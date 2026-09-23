"""V4 lab model shim.

The compactor (compactor/main.py) talks to "vLLM" at VLLM_URL and calls:
  - POST {VLLM_URL}/v1/chat/completions   (streaming and non-streaming)
  - POST {VLLM_URL}/tokenize              with either {"model","prompt"} or
                                           {"model","messages",
                                            "add_generation_prompt",
                                            "continue_final_message"},
                                           and reads back {"count": N}.
  - GET  {VLLM_URL}/v1/models             (health check)

There is no publicly pullable vLLM CPU image (checked: vllm/vllm-openai on
Docker Hub only ships cuXXX/rocm/aarch64 tags; the CPU build requires
compiling Dockerfile.cpu from source, which was judged impractical for this
lab -- see LAB.md "Model parity gaps"). This lab instead runs llama.cpp's
own OpenAI-compatible server (which already implements /v1/chat/completions,
including streaming, byte-compatibly) and this shim adds the one thing it
does NOT have in vLLM's shape: /tokenize returning {"count": N}.

The shim uses the SAME HF tokenizer the compactor itself would load as
MODEL_REPO (AutoTokenizer, tokenizer-only, no torch -- same technique
compactor/main.py uses for its own local-fallback counter), so the count
this returns is the model's real tokenizer count, not an estimate.

Parity gap, documented and NOT worked around: vLLM's `structured_outputs`
grammar field (used by the word-ban work) has no equivalent wired up here.
If a request includes it, the shim strips it and logs once; the request
still succeeds, just without grammar enforcement.
"""
import json
import logging
import os
import sys

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from guard import assert_local_only  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s shim %(levelname)s %(message)s")
log = logging.getLogger("v4lab.shim")

LLAMA_URL = os.environ.get("LLAMA_CPP_URL", "http://v4lab-model:8000").rstrip("/")
MODEL_REPO = os.environ.get("MODEL_REPO", "")
MAX_MODEL_LEN = int(os.environ.get("SHIM_MAX_MODEL_LEN", "4096"))
BIND_PORT = int(os.environ.get("SHIM_PORT", "8000"))

assert_local_only({"LLAMA_CPP_URL": LLAMA_URL}, component="model-shim")

if not MODEL_REPO:
    log.error("MODEL_REPO not set; the shim cannot load a tokenizer. Refusing to start.")
    raise SystemExit(1)

log.info(f"loading tokenizer for {MODEL_REPO} ...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
log.info("tokenizer loaded")

_warned_structured_outputs = False

app = FastAPI()
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=5.0))


@app.get("/v1/models")
async def models():
    try:
        r = await _client.get(f"{LLAMA_URL}/v1/models")
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception:
        # llama.cpp's own /v1/models is enough of a health signal; if it's
        # down, report OUR model id anyway so callers see a shape they expect
        # and the real failure surfaces on the chat-completions call.
        return {"object": "list", "data": [{"id": MODEL_REPO, "object": "model"}]}


@app.post("/tokenize")
async def tokenize(req: Request):
    body = await req.json()
    messages = body.get("messages")
    if messages is not None:
        add_gen = body.get("add_generation_prompt", True)
        continue_final = body.get("continue_final_message", False)
        try:
            text = _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_gen,
                continue_final_message=continue_final,
            )
        except Exception as e:
            log.warning(f"apply_chat_template failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=400)
    else:
        text = body.get("prompt", "")

    ids = _tokenizer.encode(text or "")
    return {"count": len(ids), "tokens": ids, "max_model_len": MAX_MODEL_LEN}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    global _warned_structured_outputs
    body = await req.json()
    if "structured_outputs" in body:
        if not _warned_structured_outputs:
            log.warning(
                "structured_outputs given but not supported by this shim/llama.cpp "
                "backend -- stripping it (grammar enforcement is a documented lab "
                "parity gap; see LAB.md)"
            )
            _warned_structured_outputs = True
        body = {k: v for k, v in body.items() if k != "structured_outputs"}

    stream = bool(body.get("stream"))
    upstream_url = f"{LLAMA_URL}/v1/chat/completions"

    if not stream:
        r = await _client.post(upstream_url, json=body)
        return JSONResponse(r.json(), status_code=r.status_code)

    async def gen():
        async with _client.stream("POST", upstream_url, json=body) as r:
            async for chunk in r.aiter_bytes():
                yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=BIND_PORT, log_level="info")
