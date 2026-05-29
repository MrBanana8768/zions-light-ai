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
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from memory import (
    ensure_storage_layout,
    list_known_conv_ids,
    resolve_conv_id,
    storage_summary,
)


def _env_int(name: str, default: int) -> int:
    """os.environ.get returns '' (not the default) when the var is set to an
    empty string, which is what .env files do for opt-in blanks. Treat empty
    as 'use the default'.
    """
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
MODEL_REPO = os.environ.get("MODEL_REPO")
MAX_MODEL_LEN = _env_int("MAX_MODEL_LEN", 32768)
TARGET_TOKENS = _env_int("COMPACTOR_TARGET_TOKENS", int(MAX_MODEL_LEN * 0.75))
KEEP_RECENT_TURNS = _env_int("COMPACTOR_KEEP_RECENT_TURNS", 4)
SUMMARY_MAX_TOKENS = _env_int("COMPACTOR_SUMMARY_MAX_TOKENS", 1024)

# V2.0 Phase 1: admin endpoint binding. Default "127.0.0.1" rejects any
# non-localhost client at the dependency layer (we still bind the FastAPI
# socket to 0.0.0.0 because uvicorn doesn't support dual-listen, but the
# admin paths return 403 unless the client IP is localhost). Set this to
# "0.0.0.0" to expose admin endpoints externally — only safe if you have
# auth/firewall in front.
ADMIN_BIND = os.environ.get("COMPACTOR_ADMIN_BIND", "127.0.0.1").strip()

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure /data/openwebui/compactor/{facts,summaries,chromadb}/
    exist. Idempotent. Shutdown: nothing yet (Phase 3 will close ChromaDB).
    """
    try:
        ensure_storage_layout()
        logger.info("storage layout ready")
    except Exception as e:
        logger.warning(f"could not initialize storage layout: {e}")
    yield


app = FastAPI(title="context-compactor", lifespan=lifespan)


def _require_localhost(request: Request) -> None:
    """FastAPI dependency: gate admin endpoints to localhost unless
    COMPACTOR_ADMIN_BIND is explicitly set to something other than 127.0.0.1.
    """
    if ADMIN_BIND != "127.0.0.1":
        return  # operator opted in to external admin access
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(
            status_code=403,
            detail=(
                "admin endpoints are localhost-only by default; "
                "set COMPACTOR_ADMIN_BIND=0.0.0.0 to expose externally"
            ),
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    messages = body.get("messages", [])

    # V2.0 Phase 1: resolve conv_id for observability. No memory operations
    # yet — Phase 2 wires this into facts injection, Phase 3 into RAG, etc.
    try:
        conv_id, source = resolve_conv_id(
            dict(request.headers), messages, body=body
        )
        logger.info(f"conv_id={conv_id} source={source} msgs={len(messages)}")
    except Exception as e:
        logger.warning(f"conv_id resolution failed: {e}")

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


# ---------------------------------------------------------------------------
# V2.0 Phase 1: admin/observability endpoints
# ---------------------------------------------------------------------------
# Localhost-only by default. Read-only at this phase (writes land in Phase 2).
# These exist so operators can answer "what conv_ids has the compactor seen?"
# from a pod shell without poking around in /data manually.

from fastapi import Depends  # noqa: E402 (kept local to the admin block)


@app.get("/admin/conversations", dependencies=[Depends(_require_localhost)])
async def admin_list_conversations():
    """List every conv_id that has any V2 state on disk. Phase 1 always
    returns [] because nothing writes yet — establishes the endpoint shape.
    """
    return {"conversations": list_known_conv_ids()}


@app.get(
    "/admin/conversations/{conv_id}",
    dependencies=[Depends(_require_localhost)],
)
async def admin_conversation_summary(conv_id: str):
    """Per-conv inventory: which files exist, sizes. Phase 2/3/4 will add
    content shape (fact count, message count, etc.).
    """
    return storage_summary(conv_id)
