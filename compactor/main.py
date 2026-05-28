"""
context-compactor: OpenAI-compatible middleware proxy in front of vLLM.

Counts tokens on incoming /v1/chat/completions requests using the target
model's tokenizer. When a conversation approaches MAX_MODEL_LEN, it asks
vLLM to summarize the older turns and replaces them with a single synthetic
system message, then forwards the trimmed request. Streaming responses are
proxied verbatim so OpenWebUI's UI keeps working.
"""

import logging
import os
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
MODEL_REPO = os.environ.get("MODEL_REPO")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
TARGET_TOKENS = int(
    os.environ.get("COMPACTOR_TARGET_TOKENS", str(int(MAX_MODEL_LEN * 0.75)))
)
KEEP_RECENT_TURNS = int(os.environ.get("COMPACTOR_KEEP_RECENT_TURNS", "4"))
SUMMARY_MAX_TOKENS = int(os.environ.get("COMPACTOR_SUMMARY_MAX_TOKENS", "1024"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("compactor")

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    if not MODEL_REPO:
        logger.warning("MODEL_REPO not set; falling back to char/4 token estimator")
        return None
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        logger.info(f"loaded tokenizer for {MODEL_REPO}")
    except Exception as e:
        logger.warning(f"could not load tokenizer for {MODEL_REPO}: {e}; using char/4 estimator")
        _tokenizer = None
    return _tokenizer


def _message_text(m: dict) -> str:
    content = m.get("content") or ""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def count_tokens(messages: list[dict]) -> int:
    tok = get_tokenizer()
    if tok is not None:
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(tok.encode(text))
        except Exception:
            total = 0
            for m in messages:
                total += len(tok.encode(_message_text(m))) + 4
            return total
    return sum(len(_message_text(m)) // 4 + 4 for m in messages)


SUMMARY_PROMPT = """You are summarizing an earlier portion of a conversation so it can be compressed into context.

Produce a concise but comprehensive summary that preserves:
- Key facts, names, numbers, decisions, and instructions given
- Any code, file paths, commands, or URLs mentioned
- The user's goals, constraints, and stated preferences
- The state of any in-progress work

Do not editorialize. Do not greet. Output only the summary."""


async def summarize(client: httpx.AsyncClient, to_summarize: list[dict]) -> str:
    transcript = "\n\n".join(
        f"[{m.get('role', 'unknown')}]: {_message_text(m)}" for m in to_summarize
    )
    payload = {
        "model": MODEL_REPO,
        "messages": [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Conversation to summarize:\n\n{transcript}"},
        ],
        "max_tokens": SUMMARY_MAX_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }
    r = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=300.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def split_messages(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) <= KEEP_RECENT_TURNS:
        return system_msgs, [], non_system
    return system_msgs, non_system[:-KEEP_RECENT_TURNS], non_system[-KEEP_RECENT_TURNS:]


async def compact_if_needed(messages: list[dict]) -> list[dict]:
    current = count_tokens(messages)
    if current <= TARGET_TOKENS:
        return messages
    system_msgs, to_summarize, keep_recent = split_messages(messages)
    if not to_summarize:
        logger.warning(
            f"over budget ({current}>{TARGET_TOKENS}) but no older turns to summarize"
        )
        return messages
    async with httpx.AsyncClient() as client:
        summary = await summarize(client, to_summarize)
    summary_msg = {
        "role": "system",
        "content": f"[Summary of earlier conversation]\n{summary}",
    }
    new_messages = system_msgs + [summary_msg] + keep_recent
    new_count = count_tokens(new_messages)
    logger.info(
        f"compacted: summarized {len(to_summarize)} messages, {current} -> {new_count} tokens"
    )
    return new_messages


app = FastAPI(title="context-compactor")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    messages = body.get("messages", [])
    try:
        body["messages"] = await compact_if_needed(messages)
    except Exception as e:
        logger.exception(f"compaction failed; forwarding original messages: {e}")

    stream = bool(body.get("stream", False))
    client = httpx.AsyncClient(timeout=None)

    if stream:
        async def event_stream():
            try:
                async with client.stream(
                    "POST", f"{VLLM_URL}/v1/chat/completions", json=body
                ) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        r = await client.post(f"{VLLM_URL}/v1/chat/completions", json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    finally:
        await client.aclose()


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{VLLM_URL}/v1/models", timeout=30.0)
        return JSONResponse(content=r.json(), status_code=r.status_code)


@app.get("/health")
async def health():
    return {"status": "ok", "vllm_url": VLLM_URL, "target_tokens": TARGET_TOKENS}
