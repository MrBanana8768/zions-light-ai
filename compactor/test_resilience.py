"""
CPU-only Tier-1 tests for V2.3 Theme 2 vLLM-restart resilience.

Drives the real FastAPI app via TestClient with httpx.AsyncClient patched so
every call to vLLM raises httpx.ConnectError (simulating vLLM down /
restarting). Asserts the compactor returns a clean 503 (non-stream) or a
visible friendly message (stream) instead of an opaque 500.

Run: python test_resilience.py
"""

import json
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

_TMP = tempfile.mkdtemp(prefix="zions-resil-test-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "2000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # no ChromaDB in this test

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


client = TestClient(main.app)


class _UnreachableClient:
    """Stands in for httpx.AsyncClient — every vLLM call raises ConnectError.
    Supports both the bare-construct form (chat handler) and the
    `async with httpx.AsyncClient()` form (/v1/models)."""
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        raise httpx.ConnectError("All connection attempts failed")

    async def get(self, *a, **k):
        raise httpx.ConnectError("All connection attempts failed")

    def stream(self, *a, **k):
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    async def aclose(self):
        pass


# ---------------------------------------------------------------------------
# Helper shape
# ---------------------------------------------------------------------------

def test_unreachable_body_shape():
    print("\n[test] _vllm_unreachable_body: OpenAI-error shape")
    b = main._vllm_unreachable_body("ConnectError: x")
    assert_true("error" in b, "has error key")
    assert_eq(b["error"]["code"], "model_unavailable", "code")
    assert_eq(b["error"]["type"], "service_unavailable", "type")
    assert_true("retry" in b["error"]["message"].lower(), "message is retryable")
    assert_true("ConnectError" in b["error"]["detail"], "detail preserved")


def test_unreachable_stream_chunks_shape():
    print("\n[test] _vllm_unreachable_stream_chunks: 2 valid chunks w/ message")
    chunks = main._vllm_unreachable_stream_chunks("m")
    assert_eq(len(chunks), 2, "two chunks")
    assert_true(main.MODEL_RESTART_MESSAGE in chunks[0]["choices"][0]["delta"]["content"],
                "message in first chunk")
    assert_eq(chunks[1]["choices"][0]["finish_reason"], "stop", "stop in last")
    json.dumps(chunks)  # must be serializable


# ---------------------------------------------------------------------------
# Non-streaming → clean 503
# ---------------------------------------------------------------------------

def test_nonstream_returns_503_when_vllm_down():
    print("\n[test] POST /v1/chat/completions (non-stream): 503 not 500 when vLLM down")
    with patch.object(main.httpx, "AsyncClient", _UnreachableClient):
        r = client.post("/v1/chat/completions", json={
            "model": "x", "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
    assert_eq(r.status_code, 503, "503 returned")
    body = r.json()
    assert_eq(body["error"]["code"], "model_unavailable", "friendly error body")


# ---------------------------------------------------------------------------
# Streaming → visible friendly message, clean [DONE]
# ---------------------------------------------------------------------------

def test_stream_emits_friendly_message_when_vllm_down():
    print("\n[test] POST /v1/chat/completions (stream): friendly msg, not dead stream")
    with patch.object(main.httpx, "AsyncClient", _UnreachableClient):
        r = client.post("/v1/chat/completions", json={
            "model": "x", "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
    assert_eq(r.status_code, 200, "stream opens 200")
    text = r.text
    # The message contains a non-ASCII emoji that json.dumps escapes (⏳),
    # so assert on the ASCII portion which survives verbatim.
    assert_true("starting up or restarting" in text, "friendly message present in stream")
    assert_true("[DONE]" in text, "stream terminates with [DONE]")


# ---------------------------------------------------------------------------
# /v1/models → clean 503
# ---------------------------------------------------------------------------

def test_models_returns_503_when_vllm_down():
    print("\n[test] GET /v1/models: 503 not 500 when vLLM down")
    with patch.object(main.httpx, "AsyncClient", _UnreachableClient):
        r = client.get("/v1/models")
    assert_eq(r.status_code, 503, "503 returned")
    assert_eq(r.json()["error"]["code"], "model_unavailable", "friendly error body")


# ---------------------------------------------------------------------------
# Commands still work when vLLM is down (no model needed)
# ---------------------------------------------------------------------------

def test_command_works_even_when_vllm_down():
    print("\n[test] slash command (/help) succeeds even with vLLM unreachable")
    with patch.object(main.httpx, "AsyncClient", _UnreachableClient):
        r = client.post("/v1/chat/completions", json={
            "model": "x", "messages": [{"role": "user", "content": "/help"}],
            "stream": False,
        }, headers={"X-Conversation-Id": "resil-cmd"})
    assert_eq(r.status_code, 200, "command short-circuits, 200")
    content = r.json()["choices"][0]["message"]["content"]
    assert_true("/list-facts" in content, "got the help text, not a vLLM error")


def _all():
    return [
        test_unreachable_body_shape,
        test_unreachable_stream_chunks_shape,
        test_nonstream_returns_503_when_vllm_down,
        test_stream_emits_friendly_message_when_vllm_down,
        test_models_returns_503_when_vllm_down,
        test_command_works_even_when_vllm_down,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            t()
        print("\nAll resilience smoke tests passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
