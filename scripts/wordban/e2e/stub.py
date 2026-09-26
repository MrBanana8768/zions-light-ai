"""Recording stand-in for vLLM on :8000, for `wordban.py e2e`. Runs in the image's /opt/vllm-venv. Every chat body
is logged, then pushed through vLLM 0.19's OWN front half: ChatCompletionRequest validation -> to_sampling_params
-> SamplingParams.update_from_tokenizer -> the xgrammar structured-output validation + compile the engine
performs, and the probe sentences are run through the compiled grammar. No weights: the reply is canned."""
import hashlib
import json
import re
import sys
import time
import types

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import get_tokenizer
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend, validate_xgrammar_grammar

LOG = sys.argv[1]
MODEL = sys.argv[2]
PROBES = json.load(open(sys.argv[3], encoding="utf-8"))          # [[text, expected_accept], ...]
tok = get_tokenizer("/tok", tokenizer_mode="auto")
cfg = types.SimpleNamespace(structured_outputs_config=types.SimpleNamespace(disable_any_whitespace=False),
                            speculative_config=None)
BE = XgrammarBackend(vllm_config=cfg, tokenizer=tok, vocab_size=len(tok.vocab))
app = FastAPI()


def vllm_front_half(body):
    out = {}
    try:
        req = ChatCompletionRequest.model_validate(body)
        sp = req.to_sampling_params(max_tokens=int(body.get("max_tokens") or 512), default_sampling_params={})
        sp.update_from_tokenizer(tok)
        out["bad_words_token_seqs"] = len(sp.bad_words_token_ids or [])
        so = sp.structured_outputs
        if so is not None:
            validate_xgrammar_grammar(sp)
            t0 = time.time()
            BE.compile_grammar(StructuredOutputOptions.GRAMMAR, so.grammar)
            out["compile_s"] = round(time.time() - t0, 2)
            probes = []
            for text, want in PROBES:
                g = BE.compile_grammar(StructuredOutputOptions.GRAMMAR, so.grammar)
                ids = tok.encode(text=text, add_special_tokens=False)
                ok = all(g.accept_tokens("probe", [t]) for t in ids) and g.accept_tokens("probe", [tok.eos_token_id])
                probes.append({"text": text, "accepted": bool(ok), "expected": bool(want)})
            out["probes"] = probes
        out["vllm_accepts_request"] = True
    except Exception as e:                                           # noqa: BLE001
        out["vllm_accepts_request"] = False
        out["error"] = f"{type(e).__name__}: {e}"[:600]
    return out


@app.get("/health")
async def health():
    return JSONResponse({})


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "vllm", "max_model_len": 32768}]}


@app.post("/tokenize")
async def tokenize(request: Request):
    b = await request.json()
    text = b.get("prompt") or "\n".join(str(m.get("content")) for m in b.get("messages", []))
    ids = tok.encode(text=text, add_special_tokens=False)
    return {"count": len(ids) + 4 * len(b.get("messages", [])), "max_model_len": 32768, "tokens": ids}


@app.post("/detokenize")
async def detokenize(request: Request):
    b = await request.json()
    return {"prompt": tok.decode(b.get("tokens", []))}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    so = body.get("structured_outputs")
    g = so.get("grammar") if isinstance(so, dict) else None
    # the driver's marker, wherever it ended up (the compactor prepends a date line to the user message)
    tags = re.findall(r"\[wb-e2e [^\]]+\]", json.dumps(body.get("messages", []), ensure_ascii=False))
    rec = {"ua": request.headers.get("user-agent"), "stream": body.get("stream"),
           "tag": tags[-1] if tags else "",
           "bad_words": None if body.get("bad_words") is None else len(body["bad_words"]),
           "grammar_sha256": None if g is None else hashlib.sha256(g.encode()).hexdigest(),
           "grammar_chars": None if g is None else len(g), "vllm": vllm_front_half(body)}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    reply = "Stub reply."
    if body.get("stream"):
        def sse():
            chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            chunk["choices"][0] = {"index": 0, "delta": {}, "finish_reason": "stop"}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    return {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
