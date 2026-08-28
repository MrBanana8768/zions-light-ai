"""A vLLM-SHAPED tokenizer server, small enough to run in CI.

WHY THIS EXISTS
---------------
`compactor/test_smoke.py` opens by saying it runs with "no vLLM, no GPU, no real
tokenizer", and every budget test in the suite has inherited that. The budget
code was therefore only ever asserted against the char/4 estimator — the very
estimator that was wrong. That is why a 23-51% undercount survived to
production twice (INCIDENT_2026-08-24.md, INCIDENT_2026-08-28.md).

This server closes that hole. It speaks the two endpoints the compactor
actually depends on, with the request and response shapes taken from vLLM's own
`vllm/entrypoints/openai/protocol.py`, and it answers with a REAL tokenizer, so
`count_tokens` can be measured against something that disagrees with it the way
production disagreed with it.

WHAT IT DOES NOT PROVE
----------------------
Read this before you cite a green run as evidence.

* The tokenizer here is NOT Cydonia-24B's. It is a small instruct tokenizer
  (default HuggingFaceTB/SmolLM2-135M-Instruct, 49k byte-level BPE vocabulary).
  The ABSOLUTE numbers it returns have no bearing on the production budget.
  A conversation that fits 16,384 tokens here may not fit on the pod.
* What it DOES validate is the CONTRACT and the WIRING: that the compactor asks
  the server rather than trusting itself, that it reads `.count`, that its
  scale-correction arithmetic actually lands under the limit when measured by
  the server, that the batches it packs genuinely fit, and that it degrades
  honestly when the endpoint lies or disappears.
* GENERATION_RESERVE, MAX_MODEL_LEN and the shape of the production window are
  NOT validated here. Those are properties of the deployed model.
* This is not vLLM. It reimplements two endpoints. See the README for the
  measured reason vLLM's own CPU build could not be used on this host, and for
  what that costs us.

PROTOCOL FIDELITY (verified against vllm==0.10.0, the CPU release image)
------------------------------------------------------------------------
Dumped from `vllm.entrypoints.openai.protocol` inside
public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.10.0 on 2026-08-28:

    TokenizeChatRequest fields:
        add_generation_prompt = True      <- default
        add_special_tokens    = False     <- default
        chat_template         = None
        chat_template_kwargs  = None
        continue_final_message = False
        messages              = (required)
        mm_processor_kwargs   = None
        model                 = None
        return_token_strs     = False
        tools                 = None

    TokenizeCompletionRequest fields:
        add_special_tokens, model, prompt, return_token_strs

    TokenizeResponse fields:
        count (required), max_model_len (required), tokens (required),
        token_strs = None

    ErrorResponse fields:
        object = "error", message, type, param = None, code

`/tokenize` performs NO context-length validation in vLLM — see
`serving_engine.py:606-609`, which returns early for TokenizeChatRequest. This
server matches that: /tokenize always answers, however large the input.

`/v1/chat/completions` DOES validate, and the wording is reproduced verbatim
from `serving_engine.py:621-631`. NOTE: the deployed stack pins
`vllm==0.24.0` (Dockerfile:78), not 0.10.0. The endpoint shapes are believed
stable across that range but were NOT verified against 0.24.0 from this
machine. Set FIXTURE_ERROR_STYLE=legacy to emit the older
"your prompt contains at least N input tokens" wording that
`main._CTX_OVERFLOW_RE` is written against.

FAULT INJECTION
---------------
POST /_fixture/mode  {"tokenize_mode": ..., "factor": ..., "status": ...}

    ok          normal behaviour
    wrong       return int(true_count * factor) — plausible, and wrong.
                factor 0.5 reproduces the production undercount.
    http_error  return `status` (default 400) with a vLLM-ish error body
    garbage     return 200 with a body that has no `count` key
    hang        sleep `delay` seconds (default 15) — longer than the
                compactor's 10s read timeout, so it exercises the real
                httpx.ReadTimeout path rather than a simulated one

GET /_fixture/stats returns per-endpoint call counts, so a test can assert the
compactor consults /tokenize a BOUNDED number of times and not once per
message — the cost discipline main.py:846-853 depends on.

POST /_fixture/reset clears mode and stats.
"""

import asyncio
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

TOKENIZER_ID = os.environ.get(
    "FIXTURE_TOKENIZER", "HuggingFaceTB/SmolLM2-135M-Instruct"
)
SERVED_MODEL_NAME = os.environ.get("FIXTURE_SERVED_MODEL_NAME", TOKENIZER_ID)
MAX_MODEL_LEN = int(os.environ.get("FIXTURE_MAX_MODEL_LEN", "32768"))
ERROR_STYLE = os.environ.get("FIXTURE_ERROR_STYLE", "v010")

app = FastAPI(title="vllm-shaped tokenizer fixture")

_tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

_MODE: dict[str, Any] = {
    "tokenize_mode": "ok",
    "factor": 0.5,
    "status": 400,
    "delay": 15.0,
}
_STATS: dict[str, int] = {}


def _bump(key: str) -> None:
    _STATS[key] = _STATS.get(key, 0) + 1


# --------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------


def _normalize_content(c: Any) -> str:
    """Flatten OpenAI multimodal content arrays the way a chat template sees
    them. Image parts contribute no text — this fixture has no vision tower,
    which the README states plainly as a gap."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text") or "")
        return " ".join(out)
    if c is None:
        return ""
    return str(c)


def _encode_messages(
    messages: list[dict],
    add_generation_prompt: bool = True,
    add_special_tokens: bool = False,
) -> list[int]:
    msgs = [
        {"role": m.get("role", "user"), "content": _normalize_content(m.get("content"))}
        for m in messages
    ]
    text = _tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return _tok.encode(text, add_special_tokens=add_special_tokens)


def _true_count(body: dict) -> tuple[int, list[int]]:
    if body.get("messages") is not None:
        ids = _encode_messages(
            body["messages"],
            bool(body.get("add_generation_prompt", True)),
            bool(body.get("add_special_tokens", False)),
        )
    else:
        ids = _tok.encode(
            body.get("prompt") or "",
            add_special_tokens=bool(body.get("add_special_tokens", True)),
        )
    return len(ids), ids


@app.post("/tokenize")
async def tokenize(request: Request):
    _bump("tokenize")
    body = await request.json()
    mode = _MODE["tokenize_mode"]

    if mode == "hang":
        await asyncio.sleep(float(_MODE["delay"]))
    if mode == "http_error":
        status = int(_MODE["status"])
        return JSONResponse(
            status_code=status,
            content={
                "object": "error",
                "message": "fixture: injected /tokenize failure",
                "type": "BadRequestError",
                "param": None,
                "code": status,
            },
        )
    if mode == "garbage":
        # 200, plausible-looking, and missing the one key the caller reads.
        return JSONResponse(
            status_code=200,
            content={"tokens": [1, 2, 3], "max_model_len": MAX_MODEL_LEN},
        )

    count, ids = _true_count(body)
    if mode == "wrong":
        count = int(count * float(_MODE["factor"]))

    payload: dict[str, Any] = {
        "count": count,
        "max_model_len": MAX_MODEL_LEN,
        "tokens": ids,
        "token_strs": None,
    }
    if body.get("return_token_strs"):
        payload["token_strs"] = _tok.convert_ids_to_tokens(ids)
    return JSONResponse(status_code=200, content=payload)


@app.post("/detokenize")
async def detokenize(request: Request):
    _bump("detokenize")
    body = await request.json()
    return {"prompt": _tok.decode(body.get("tokens") or [])}


# --------------------------------------------------------------------------
# OpenAI-compatible chat completions
# --------------------------------------------------------------------------


def _context_error(token_num: int, max_tokens: int | None) -> dict:
    """Reproduced from vllm/entrypoints/openai/serving_engine.py (v0.10.0)."""
    if ERROR_STYLE == "legacy":
        msg = (
            f"This model's maximum context length is {MAX_MODEL_LEN} tokens. "
            f"However, your prompt contains at least {token_num} input tokens. "
            f"Please reduce the length of the messages."
        )
    elif max_tokens is None:
        msg = (
            f"This model's maximum context length is {MAX_MODEL_LEN} tokens. "
            f"However, you requested {token_num} tokens in the messages, "
            f"Please reduce the length of the messages."
        )
    else:
        msg = (
            f"This model's maximum context length is {MAX_MODEL_LEN} tokens. "
            f"However, you requested {max_tokens + token_num} tokens "
            f"({token_num} in the messages, {max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion."
        )
    return {
        "object": "error",
        "message": msg,
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    }


_CANNED = "Acknowledged. (fixture reply — this server has no model weights.)"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _bump("chat_completions")
    body = await request.json()
    messages = body.get("messages") or []
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")

    ids = _encode_messages(messages)
    token_num = len(ids)

    # vLLM's own gate, same arithmetic: serving_engine.py:619-631
    over = (
        token_num >= MAX_MODEL_LEN
        if max_tokens is None
        else token_num + int(max_tokens) > MAX_MODEL_LEN
    )
    if over:
        return JSONResponse(
            status_code=400,
            content=_context_error(token_num, None if max_tokens is None else int(max_tokens)),
        )

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = body.get("model") or SERVED_MODEL_NAME

    if body.get("stream"):

        async def _sse():
            first = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(first)}\n\n"
            for word in _CANNED.split(" "):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            last = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": token_num,
                    "completion_tokens": len(_tok.encode(_CANNED)),
                    "total_tokens": token_num + len(_tok.encode(_CANNED)),
                },
            }
            yield f"data: {json.dumps(last)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    completion_tokens = len(_tok.encode(_CANNED))
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _CANNED},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": token_num,
            "completion_tokens": completion_tokens,
            "total_tokens": token_num + completion_tokens,
        },
    }


@app.get("/v1/models")
async def models():
    _bump("models")
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm",
                "max_model_len": MAX_MODEL_LEN,
            }
        ],
    }


@app.get("/health")
async def health():
    return JSONResponse(status_code=200, content={"status": "ok"})


# --------------------------------------------------------------------------
# fixture control plane
# --------------------------------------------------------------------------


@app.get("/_fixture/mode")
async def get_mode():
    return dict(_MODE)


@app.post("/_fixture/mode")
async def set_mode(request: Request):
    body = await request.json()
    for k in ("tokenize_mode", "factor", "status", "delay"):
        if k in body:
            _MODE[k] = body[k]
    return dict(_MODE)


@app.get("/_fixture/stats")
async def stats():
    return dict(_STATS)


@app.post("/_fixture/reset")
async def reset():
    _MODE.update({"tokenize_mode": "ok", "factor": 0.5, "status": 400, "delay": 15.0})
    _STATS.clear()
    return {"reset": True}


@app.get("/_fixture/info")
async def info():
    return {
        "tokenizer": TOKENIZER_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "max_model_len": MAX_MODEL_LEN,
        "error_style": ERROR_STYLE,
        "has_chat_template": bool(_tok.chat_template),
        "vocab_size": _tok.vocab_size,
    }
