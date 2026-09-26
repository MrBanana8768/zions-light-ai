"""
Tier-1 tests for v3.1.9.8 REQUEST TRACING (compactor/reqctx.py + the
chat_completions wrapper in main.py + logsetup.RequestIdFilter +
bgwork.BackgroundPool's task tagging + health.gather_health_full's
requests_in_flight/last_request_done_age_s).

The owner's ask: "enhance our logging with a guid to keep track of requests
through the entire process from sent to response". This file covers, in
order:

  [U1] id validation/minting (reqctx.new_request_id, .sane_incoming_id)
  [U2] purpose-tagged vLLM child ids (reqctx.vllm_headers)
  [U3] RequestTrace.finish is idempotent and renders the documented fields
  [U4] bgwork tags a submitted coroutine's log lines req=<id>/bg:<task>,
       and the id survives a run_in_threadpool hop (both via real
       contextvar propagation, not a mock)
  [E1] an incoming X-Request-Id is honoured; an insane one is replaced
  [E2] the response carries X-Request-Id, streaming and non-streaming
  [E3] the main vLLM call carries a bare X-Request-Id header
  [E4] the "request done" line fires exactly once, on success and on a
       vLLM-rejected (error) request, streaming and non-streaming
  [E5] requests_in_flight rises during the handler and falls back to 0;
       last_request_done_age_s becomes non-None
  [E6] OpenWebUI's chat/message id are logged when present; the
       Name/Email/Id/Role headers the SAME flag forwards are never logged,
       anywhere, even when a malicious value tries to look like a probe
  [E11] tracing-verification review F1: _unsafe_conv_id_handler reads
        request.state.req_id (set by the wrapper) rather than
        reqctx.current_request_id() (already reset to None by the time
        FastAPI's exception-handler dispatch runs it)
  [E12] tracing-verification review F3: the non-streaming path's "request
        done" line fires AFTER the JSONResponse it reports on is built,
        not before

A real client disconnect (Starlette's TestClient fully drains a streaming
body and cannot simulate a peer hanging up) is covered separately by
test_disconnect_uvloop.py's real-uvicorn/real-socket harness — see that
file's own docstring for why a mocked ASGI transport cannot answer that
question. [E4]'s "cancelled" status is instead proven by construction: a
patched _async_tail is bypassed entirely, and CancelledError/GeneratorExit
handling in event_stream() and the wrapper is unit-tested directly here at
[U3]/[U4] and via the wrapper's exception-path code, which is exercised at
[E4]'s error case (an exception is a cancellation's sibling on this code
path — both take the `except` branch that calls `trace.finish` then
re-raises).

    python test_request_tracing.py
"""

import contextlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"
os.environ["COMPACTOR_HIERARCHICAL_SUMMARY"] = "false"

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-request-tracing-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import bgwork  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import reqctx  # noqa: E402
import retrieval  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from starlette.responses import StreamingResponse  # noqa: E402

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None
memory.ensure_storage_layout()

client = TestClient(main.app, client=("127.0.0.1", 12360), raise_server_exceptions=False)


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


# ---------------------------------------------------------------------------
# Log capture — same shape as test_truncated_tail.py, EXCEPT the collector
# also carries logsetup.RequestIdFilter, since the filter in production is
# installed on the specific StreamHandler logsetup.configure() builds, not
# on every handler a logger happens to have. Attaching the same stateless
# filter (it only reads reqctx's contextvars — never the record) to this
# handler exercises the identical code path.
# ---------------------------------------------------------------------------

import logsetup  # noqa: E402


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        self.addFilter(logsetup.RequestIdFilter())

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name: str = "compactor"):
    lg = logging.getLogger(logger_name)
    handler = _Collector()
    prev_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)


def _find(records, needle):
    return next((r for r in records if needle in r.getMessage()), None)


def _all_text(records) -> str:
    return "\n".join(r.getMessage() for r in records)


# ---------------------------------------------------------------------------
# Stub vLLM backends (shape lifted from test_truncated_tail.py).
# ---------------------------------------------------------------------------

class _StubResponse:
    status_code = 200
    text = ""

    def __init__(self, content, finish_reason="stop", status_code=200):
        self._content = content
        self._fr = finish_reason
        self.status_code = status_code

    def json(self):
        if self.status_code >= 400:
            return {"error": {"message": "rejected for test", "code": self.status_code}}
        return {
            "id": "stub",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self._content},
                "finish_reason": self._fr,
            }],
            "usage": {"completion_tokens": 7},
        }


class _StubVLLM:
    reply = "hello"
    finish_reason = "stop"
    status_code = 200
    captured_headers: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None, **kwargs):
        _StubVLLM.captured_headers.append(dict(headers or {}))
        return _StubResponse(_StubVLLM.reply, _StubVLLM.finish_reason, _StubVLLM.status_code)

    async def aclose(self):
        pass


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _content(text) -> bytes:
    return _sse({"choices": [{"delta": {"content": text}}]})


def _finish(reason) -> bytes:
    return _sse({"choices": [{"delta": {}, "finish_reason": reason}]})


DONE = b"data: [DONE]\n\n"


class _StubStreamResp:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code

    async def aread(self):
        return json.dumps({"error": {"message": "rejected for test"}}).encode()

    async def aiter_raw(self):
        for c in self._chunks:
            yield c


class _StubStreamCM:
    def __init__(self, chunks, status_code=200):
        self._resp = _StubStreamResp(chunks, status_code)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _StubStreamVLLM:
    chunks: list = []
    status_code = 200
    captured_headers: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None, headers=None, **kwargs):
        _StubStreamVLLM.captured_headers.append(dict(headers or {}))
        return _StubStreamCM(list(_StubStreamVLLM.chunks), _StubStreamVLLM.status_code)

    async def aclose(self):
        pass


def _stream_of(text, finish_reason="stop"):
    return [_content(text), _finish(finish_reason), DONE]


def _post_nonstream(conv_id, reply="hello", finish_reason="stop", status_code=200, headers=None):
    _StubVLLM.reply = reply
    _StubVLLM.finish_reason = finish_reason
    _StubVLLM.status_code = status_code
    _StubVLLM.captured_headers = []
    req_headers = {"X-Conversation-Id": conv_id}
    if headers:
        req_headers.update(headers)
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), capture() as cap:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model",
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": False},
            headers=req_headers,
        )
    return r, cap.records


def _post_stream(conv_id, chunks, status_code=200, headers=None):
    _StubStreamVLLM.chunks = chunks
    _StubStreamVLLM.status_code = status_code
    _StubStreamVLLM.captured_headers = []
    req_headers = {"X-Conversation-Id": conv_id}
    if headers:
        req_headers.update(headers)
    with patch.object(main.httpx, "AsyncClient", _StubStreamVLLM), capture() as cap:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model",
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": True},
            headers=req_headers,
        )
    return r, cap.records


# ===========================================================================
# [U1] id validation/minting
# ===========================================================================

print("[U1] reqctx.new_request_id / sane_incoming_id")
assert_eq(reqctx.sane_incoming_id("abc123-DEF_45."), "abc123-DEF_45.", "a sane id round-trips")
assert_eq(reqctx.sane_incoming_id(None), None, "None -> None")
assert_eq(reqctx.sane_incoming_id(""), None, "empty -> None")
assert_eq(reqctx.sane_incoming_id("a" * 65), None, ">64 chars -> None")
assert_eq(reqctx.sane_incoming_id("a" * 64), "a" * 64, "exactly 64 chars is fine")
assert_eq(reqctx.sane_incoming_id("has a space"), None, "a space -> None")
assert_eq(reqctx.sane_incoming_id("newline\nin-it"), None, "log-injection newline -> None")
assert_eq(reqctx.sane_incoming_id("uuid-with-üñïçødé"), None, "non-ASCII -> None")

_honoured = reqctx.new_request_id("deadBEEF-1234")
assert_eq(_honoured, "deadBEEF-1234", "a sane incoming id is honoured verbatim")
_minted_a = reqctx.new_request_id("bad id; drop table")
_minted_b = reqctx.new_request_id("bad id; drop table")
assert_true(_minted_a != _minted_b, "an insane incoming id is replaced with a FRESH mint each time")
assert_true(len(_minted_a) == 32 and all(c in "0123456789abcdef" for c in _minted_a),
            "the mint is a uuid4 hex (32 lowercase hex chars)")
assert_true(len(reqctx.new_request_id(None)) == 32, "no incoming id -> a mint")


# ===========================================================================
# [U2] purpose-tagged vLLM child ids
# ===========================================================================

print("\n[U2] reqctx.vllm_headers — plain id, purpose-tagged children, no id outside a request")
assert_eq(reqctx.vllm_headers(), {}, "no request id in context -> no header at all")
_tok = reqctx.request_id_ctx.set("req-abc")
try:
    assert_eq(reqctx.vllm_headers(), {"X-Request-Id": "req-abc"},
              "main call (no purpose) -> the bare id")
    assert_eq(reqctx.vllm_headers("facts"), {"X-Request-Id": "req-abc.facts1"},
              "first facts call -> .facts1")
    assert_eq(reqctx.vllm_headers("facts"), {"X-Request-Id": "req-abc.facts2"},
              "second facts call in the SAME request -> .facts2, not a repeat")
    assert_eq(reqctx.vllm_headers("dedup"), {"X-Request-Id": "req-abc.dedup1"},
              "a different purpose starts its own counter at 1")
    assert_eq(reqctx.vllm_headers("sum"), {"X-Request-Id": "req-abc.sum1"}, ".sum1")
    assert_eq(reqctx.vllm_headers("tok"), {"X-Request-Id": "req-abc.tok1"}, ".tok1")
finally:
    reqctx.request_id_ctx.reset(_tok)
assert_eq(reqctx.vllm_headers("facts"), {},
          "outside any request's context (no id set), even a purpose call returns nothing")


# ===========================================================================
# [U3] RequestTrace — idempotent finish(), documented fields
# ===========================================================================

print("\n[U3] RequestTrace.finish is idempotent and renders the documented fields")
reqctx._reset_for_tests()
with capture() as cap:
    trace = reqctx.RequestTrace("trace-1")
    assert_eq(reqctx.in_flight_stats()["requests_in_flight"], 1,
              "constructing a trace claims an in-flight slot")
    trace.conv_id = "conv-xyz"
    trace.msgs = 3
    trace.mark("t_compact", 0.125)
    trace.mark("t_guard", 0.0009)
    trace.finish(main.logger, "ok", tokens_out=42, finish_reason="stop")
    trace.finish(main.logger, "error")  # second call: must be a silent no-op
line = _find(cap.records, "request done req=trace-1")
assert_true(line is not None, "the summary line was emitted")
msg = line.getMessage()
assert_true("conv=conv-xyz" in msg, "conv= field present")
assert_true("msgs=3" in msg, "msgs= field present")
assert_true("status=ok" in msg, "status= reflects the FIRST call, not the second")
assert_true("t_compact=0.125" in msg, "t_compact rendered to ms precision")
assert_true("t_guard=0.001" in msg, "t_guard rounds to 3 decimals")
assert_true("t_images=" not in msg, "an unmeasured phase is OMITTED, not printed as 0")
assert_true("t_total=" in msg, "t_total is always present")
assert_true("tokens_out=42" in msg, "tokens_out present when given")
assert_true("finish=stop" in msg, "finish= present when given")
assert_eq(len(cap.records) - sum(1 for r in cap.records if "request done" not in r.getMessage()),
          1, "exactly ONE 'request done' line even though finish() was called twice")
assert_eq(reqctx.in_flight_stats()["requests_in_flight"], 0,
          "the in-flight slot was released exactly once")
assert_true(reqctx.in_flight_stats()["last_request_done_age_s"] is not None,
            "last_request_done_age_s becomes known")


# ===========================================================================
# [U4] bgwork tags a background tail's log lines, and the id survives a
# run_in_threadpool hop — both via REAL contextvar propagation.
# ===========================================================================

print("\n[U4] a background tail's logs carry req=<id>/bg:<task>; run_in_threadpool keeps the id")


async def _run_u4():
    from starlette.concurrency import run_in_threadpool

    tok = reqctx.request_id_ctx.set("req-bg-test")
    try:
        with capture("compactor.bgwork") as cap_bg:
            async def _tagged_tail():
                bgwork.logger.info("bg tail probe ran")

            accepted = bgwork.pool.submit(_tagged_tail(), label="tail conv=probe")
            assert_true(accepted, "the pool accepted the probe task")
            await bgwork.pool.drain(timeout=5.0)
        line = _find(cap_bg.records, "bg tail probe ran")
        assert_true(line is not None, "the background task's own log line was captured")
        assert_eq(getattr(line, "req", None), "req-bg-test/bg:tail",
                  "it carries req=<id>/bg:tail, not the bare id")

        # run_in_threadpool: anyio propagates the current Context to the
        # worker thread by default. Assert it rather than trust the
        # platform — this is exactly the kind of "should just work" claim
        # the owner's brief calls out for verification.
        seen = await run_in_threadpool(reqctx.current_request_id)
        assert_eq(seen, "req-bg-test", "the request id survives a run_in_threadpool hop")
    finally:
        reqctx.request_id_ctx.reset(tok)


import asyncio  # noqa: E402
asyncio.run(_run_u4())


# ===========================================================================
# [E1] an incoming X-Request-Id is honoured; an insane one is replaced
# ===========================================================================

print("\n[E1] endpoint: incoming X-Request-Id honoured / insane one replaced")
r, recs = _post_nonstream("rt-e1-sane", headers={"X-Request-Id": "sane-id-123"})
assert_eq(r.status_code, 200, "200")
assert_eq(r.headers.get("x-request-id"), "sane-id-123", "the SANE incoming id is echoed back")
line = _find(recs, "conv_id=")
assert_true(line is not None and getattr(line, "req", None) == "sane-id-123",
            "log lines inside the request carry the honoured id")

r, recs = _post_nonstream("rt-e1-insane", headers={"X-Request-Id": "bad id; \n injected"})
assert_eq(r.status_code, 200, "200")
_got = r.headers.get("x-request-id")
assert_true(_got is not None and _got != "bad id; \n injected",
            "an insane incoming id is NOT echoed back — a fresh one is minted")
assert_true(len(_got) == 32, "the replacement is a uuid4 hex")


# ===========================================================================
# [E2] response header set, streaming and non-streaming
# ===========================================================================

print("\n[E2] X-Request-Id on the response, both paths")
r, _ = _post_nonstream("rt-e2-ns")
assert_true("x-request-id" in r.headers, "non-streaming response carries X-Request-Id")

r, _ = _post_stream("rt-e2-s", _stream_of("hi"))
assert_true("x-request-id" in r.headers, "streaming response carries X-Request-Id")


# ===========================================================================
# [E3] the main vLLM call carries a bare X-Request-Id header
# ===========================================================================

print("\n[E3] the main vLLM call is sent X-Request-Id (no purpose suffix)")
r, _ = _post_nonstream("rt-e3-ns", headers={"X-Request-Id": "e3-nonstream-id"})
assert_eq(len(_StubVLLM.captured_headers), 1, "exactly one vLLM call")
assert_eq(_StubVLLM.captured_headers[0].get("X-Request-Id"), "e3-nonstream-id",
          "the id forwarded to vLLM is the plain request id")

r, _ = _post_stream("rt-e3-s", _stream_of("hi"), headers={"X-Request-Id": "e3-stream-id"})
assert_eq(_StubStreamVLLM.captured_headers[0].get("X-Request-Id"), "e3-stream-id",
          "same for the streaming call")


# ===========================================================================
# [E4] the "request done" line fires EXACTLY once, success and error,
# streaming and non-streaming.
# ===========================================================================

print("\n[E4] exactly one 'request done' line per request")


def _count_done(records, req_id):
    return sum(1 for r in records if r.getMessage().startswith(f"request done req={req_id}"))


r, recs = _post_nonstream("rt-e4-ok-ns", headers={"X-Request-Id": "e4-ok-ns"})
assert_eq(_count_done(recs, "e4-ok-ns"), 1, "non-stream success: exactly one line")
assert_true("status=ok" in _find(recs, "request done").getMessage(), "...and it says ok")

r, recs = _post_nonstream(
    "rt-e4-err-ns", status_code=400, headers={"X-Request-Id": "e4-err-ns"}
)
assert_eq(_count_done(recs, "e4-err-ns"), 1, "non-stream vLLM-rejected: exactly one line")
assert_true("status=error" in _find(recs, "request done").getMessage(), "...and it says error")

r, recs = _post_stream("rt-e4-ok-s", _stream_of("hi"), headers={"X-Request-Id": "e4-ok-s"})
assert_eq(_count_done(recs, "e4-ok-s"), 1, "stream success: exactly one line")
assert_true("status=ok" in _find(recs, "request done").getMessage(), "...and it says ok")
assert_true("finish=stop" in _find(recs, "request done").getMessage(), "...with finish=stop")

r, recs = _post_stream(
    "rt-e4-err-s", [], status_code=400, headers={"X-Request-Id": "e4-err-s"}
)
assert_eq(_count_done(recs, "e4-err-s"), 1, "stream vLLM-rejected: exactly one line")
assert_true("status=error" in _find(recs, "request done").getMessage(), "...and it says error")


# ===========================================================================
# [E5] requests_in_flight rises during the handler, falls back to 0
# ===========================================================================

print("\n[E5] requests_in_flight rises during the handler and falls back to 0")
reqctx._reset_for_tests()
_StubVLLM.status_code = 200
_StubVLLM.reply = "hello"
_StubVLLM.finish_reason = "stop"

_ready = threading.Event()
_release = threading.Event()


class _SlowVLLM(_StubVLLM):
    async def post(self, url, json=None, headers=None, **kwargs):
        _ready.set()
        while not _release.is_set():
            time.sleep(0.01)
        return await super().post(url, json=json, headers=headers, **kwargs)


result_box = {}


def _drive():
    with patch.object(main.httpx, "AsyncClient", _SlowVLLM):
        result_box["r"] = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model",
                  "messages": [{"role": "user", "content": "hi"}], "stream": False},
            headers={"X-Conversation-Id": "rt-e5"},
        )


t = threading.Thread(target=_drive)
t.start()
assert_true(_ready.wait(10.0), "the slow stub was reached")
assert_eq(reqctx.in_flight_stats()["requests_in_flight"], 1,
          "in flight while the handler is still inside the vLLM call")
_release.set()
t.join(10.0)
assert_eq(result_box["r"].status_code, 200, "the (eventually unblocked) request still succeeds")
assert_eq(reqctx.in_flight_stats()["requests_in_flight"], 0,
          "back to 0 once the handler is done")
assert_true(reqctx.in_flight_stats()["last_request_done_age_s"] is not None,
            "last_request_done_age_s is known")


# ===========================================================================
# [E6] OpenWebUI chat/message id logged when present; user Name/Email/Id/
# Role headers NEVER logged, anywhere.
# ===========================================================================

print("\n[E6] OpenWebUI chat/message id logged; user identity headers never logged")
_SECRET_NAME = "Jane Q. Redacted"
_SECRET_EMAIL = "jane.q.redacted@example.com"
_SECRET_UID = "user-do-not-log-me-12345"
_SECRET_ROLE = "admin-do-not-log-me"
r, recs = _post_nonstream(
    "rt-e6",
    headers={
        "X-OpenWebUI-Chat-Id": "owui-chat-99",
        "X-OpenWebUI-Message-Id": "owui-msg-77",
        "X-OpenWebUI-User-Name": _SECRET_NAME,
        "X-OpenWebUI-User-Email": _SECRET_EMAIL,
        "X-OpenWebUI-User-Id": _SECRET_UID,
        "X-OpenWebUI-User-Role": _SECRET_ROLE,
    },
)
assert_eq(r.status_code, 200, "200")
all_text = _all_text(recs)
assert_true("owui-chat-99" in all_text, "the chat id (an opaque id) IS logged")
assert_true("owui-msg-77" in all_text, "the message id (an opaque id) IS logged")
assert_true(_SECRET_NAME not in all_text, "the user's NAME is never logged")
assert_true(_SECRET_EMAIL not in all_text, "the user's EMAIL is never logged")
assert_true(_SECRET_UID not in all_text, "the user's id header value is never logged")
assert_true(_SECRET_ROLE not in all_text, "the user's role header value is never logged")


# ===========================================================================
# [E7] /health/full actually exposes requests_in_flight / last_request_
# done_age_s — the real endpoint, not just the reqctx primitive [U3]/[E5]
# already covered.
# ===========================================================================

print("\n[E7] /health/full carries request_tracing.requests_in_flight / last_request_done_age_s")
r = client.get("/health/full")
assert_eq(r.status_code, 200, "/health/full: 200")
rt = r.json().get("request_tracing")
assert_true(isinstance(rt, dict), "request_tracing is present in the payload")
assert_eq(rt.get("requests_in_flight"), 0, "at rest: 0 in flight")
# last_request_done_age_s is None until a request has finished in this
# process — E1-E6 above already ran plenty, so by now it is a number.
assert_true(rt.get("last_request_done_age_s") is not None,
            "last_request_done_age_s is known after earlier requests in this run")


# ===========================================================================
# [E8] Hostile review round 2, B1 BLOCKER: the slash-command short-circuit
# (`/list-facts`, `/forget`, `/remember`, ...) returns its OWN
# StreamingResponse (`cmd_stream()`), built and returned entirely inside
# `_chat_completions_impl`'s command branch, which never touches `_trace`
# at all. The reviewer's own probe sent four `/list-facts` requests (one
# non-streaming, three streaming) and watched requests_in_flight go
# 0 -> 1 -> 2 -> 3 -> 4 — a permanent leak on every one, and OpenWebUI
# always streams, so this was not a corner case. No vLLM stub needed here
# at all: the whole point of the command short-circuit is that it never
# calls vLLM.
# ===========================================================================

print("\n[E8] the command path (streamed AND non-streamed) leaves requests_in_flight == 0")


def _post_command(conv_id, stream, headers=None):
    req_headers = {"X-Conversation-Id": conv_id}
    if headers:
        req_headers.update(headers)
    with capture() as cap:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model", "stream": stream,
                  "messages": [{"role": "user", "content": "/list-facts"}]},
            headers=req_headers,
        )
    return r, cap.records


for _label, _stream in (("non-streamed", False), ("streamed", True)):
    reqctx._reset_for_tests()
    r, recs = _post_command(f"rt-e8-{_stream}", _stream)
    assert_eq(r.status_code, 200, f"command path ({_label}): 200")
    assert_true("x-request-id" in r.headers,
                f"command path ({_label}): X-Request-Id on the response")
    assert_eq(reqctx.in_flight_stats()["requests_in_flight"], 0,
              f"command path ({_label}): requests_in_flight back to 0 — "
              f"this is exactly what B1 found leaking")
    done_lines = [rr for rr in recs if rr.getMessage().startswith("request done")]
    assert_eq(len(done_lines), 1,
              f"command path ({_label}): exactly one 'request done' line")
    assert_true("status=ok" in done_lines[0].getMessage(),
                f"command path ({_label}): status=ok")

# Three streamed commands back to back — the reviewer's exact probe shape
# (0 -> 1 -> 2 -> 3). Every one must land back on 0, not accumulate.
reqctx._reset_for_tests()
for i in range(3):
    r, _ = _post_command(f"rt-e8-multi-{i}", True)
    assert_eq(r.status_code, 200, f"command path (streamed #{i}): 200")
assert_eq(reqctx.in_flight_stats()["requests_in_flight"], 0,
          "three streamed commands back to back: still 0, not 3")


# ===========================================================================
# [E9] Generic guarantee: ANY StreamingResponse `_chat_completions_impl`
# returns gets finished, not just event_stream()'s own (via the richer,
# inner call) and not just the command path's (E8, via the structural
# backstop only). Proven here by monkeypatching a THIRD, made-up
# streaming code path onto the impl and confirming the wrapper still
# closes it out — this is what makes the fix "structural" rather than
# "found and patched the two known generators".
# ===========================================================================

print("\n[E9] a made-up third StreamingResponse from the impl still gets finished (structural, not per-generator)")


async def _fake_third_stream():
    yield b"data: {}\n\n"
    yield b"data: [DONE]\n\n"


_orig_impl = main._chat_completions_impl


async def _impl_returns_bare_stream(request, trace):
    # Deliberately builds a StreamingResponse with NO knowledge of `trace`
    # at all — simulating any future code path that forgets to wire up
    # tracing by hand, exactly like the command path did.
    trace.conv_id = "rt-e9"
    trace.msgs = 0
    return StreamingResponse(_fake_third_stream(), media_type="text/event-stream")


reqctx._reset_for_tests()
with patch.object(main, "_chat_completions_impl", _impl_returns_bare_stream), capture() as cap:
    r = client.post(
        "/v1/chat/completions",
        json={"model": "stub-model", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
assert_eq(r.status_code, 200, "third stream: 200")
assert_true("x-request-id" in r.headers, "third stream: X-Request-Id on the response")
assert_eq(reqctx.in_flight_stats()["requests_in_flight"], 0,
          "third stream: requests_in_flight back to 0 — the STRUCTURAL backstop, "
          "not a per-generator fix, is what caught this")
done_lines = [rr for rr in cap.records if rr.getMessage().startswith("request done")]
assert_eq(len(done_lines), 1, "third stream: exactly one 'request done' line")
assert_true("status=ok" in done_lines[0].getMessage(), "third stream: status=ok")
main._chat_completions_impl = _orig_impl


# ===========================================================================
# [E10] M1: ow_chat/ow_msg sanitized through reqctx.sane_incoming_id — the
# reviewer's exact spoof (a value containing spaces, which sane_incoming_id
# rejects) must not inject a second status=/req= pair into the log line.
# ===========================================================================

print("\n[E10] M1: a spoofed X-OpenWebUI-Chat-Id cannot inject fields into the log line")
r, recs = _post_nonstream(
    "rt-e10",
    headers={"X-OpenWebUI-Chat-Id": "abc status=ok req=deadbeefdeadbeef"},
)
assert_eq(r.status_code, 200, "200")
line = _find(recs, "conv_id=")
assert_true(line is not None, "the conv_id line was captured")
msg = line.getMessage()
assert_true("ow_chat=" not in msg,
            "an insane ow_chat value is DROPPED, not force-quoted or truncated")
assert_true("req=deadbeefdeadbeef" not in msg,
            "the spoofed req= substring never appears in the message text")
assert_true("status=ok" not in msg,
            "the spoofed status=ok substring never appears in the conv_id line")

# A SANE ow_chat (real UUID shape) still gets through, unaffected.
r, recs = _post_nonstream("rt-e10b", headers={"X-OpenWebUI-Chat-Id": "owui-chat-real-123"})
assert_eq(r.status_code, 200, "200")
line = _find(recs, "conv_id=")
assert_true(line is not None and "ow_chat=owui-chat-real-123" in line.getMessage(),
            "a SANE ow_chat value is still logged")


# ===========================================================================
# [E11] tracing-verification review F1: request.state.req_id, not the
# (already-reset) ContextVar, is what _unsafe_conv_id_handler must read.
# ===========================================================================

print("\n[E11] F1: _unsafe_conv_id_handler reads request.state.req_id")


class _FakeURL:
    path = "/v1/chat/completions"


class _FakeState:
    pass


class _FakeRequest:
    """Only `.method`, `.url.path` and `.state` are read by the handler —
    see test_config_dryrun.py's `_FakeRequest` for the same duck-typing
    approach to a Starlette Request in a pure unit test."""
    method = "POST"
    url = _FakeURL()

    def __init__(self):
        self.state = _FakeState()



from memory import UnsafeConvId  # noqa: E402

# The wrapper sets request.state.req_id BEFORE anything can reset
# reqctx's ContextVar — reproduced directly here, with the ContextVar
# confirmed empty (as it always is by the time FastAPI's own
# exception-handler dispatch runs, on the chat path: chat_completions'
# `finally:` resets it once `trace.finished` is True, which it is by then
# — see the wrapper's own comment for exactly why).
assert_eq(reqctx.current_request_id(), None, "no ContextVar id set for this test")
fake_req = _FakeRequest()
fake_req.state.req_id = "state-carried-id"
resp = asyncio.run(main._unsafe_conv_id_handler(fake_req, UnsafeConvId("test")))
assert_eq(resp.status_code, 400, "still a 400")
assert_eq(resp.headers.get("x-request-id"), "state-carried-id",
          "X-Request-Id comes from request.state.req_id, not the (empty) ContextVar")

# Fallback: a route that never went through the wrapper (no
# request.state.req_id at all) still falls back to the ContextVar when
# THAT happens to be set (unchanged behaviour for non-chat routes).
fake_req2 = _FakeRequest()
_tok = reqctx.request_id_ctx.set("ctxvar-id")
try:
    resp2 = asyncio.run(main._unsafe_conv_id_handler(fake_req2, UnsafeConvId("test")))
finally:
    reqctx.request_id_ctx.reset(_tok)
assert_eq(resp2.headers.get("x-request-id"), "ctxvar-id",
          "falls back to the ContextVar when request.state has no req_id")


# ===========================================================================
# [E12] tracing-verification review F3: finish() fires AFTER the
# JSONResponse it reports on is constructed, not before.
# ===========================================================================

print("\n[E12] F3: the non-streaming 'request done' line fires after JSONResponse is built")

_orig_json_response = main.JSONResponse


class _ExplodingJSONResponse(_orig_json_response):
    """Raises exactly on the SUCCESS non-streaming construction site
    (`status_code=200`, the one this fix reordered) — every OTHER
    JSONResponse this path can build uses a different status code (400/
    502/503), so this is selective by construction, not by luck."""

    def __init__(self, *a, **k):
        if k.get("status_code") == 200:
            raise RuntimeError("simulated JSONResponse construction failure (F3 test)")
        super().__init__(*a, **k)


with patch.object(main, "JSONResponse", _ExplodingJSONResponse), \
     patch.object(main.httpx, "AsyncClient", _StubVLLM):
    _StubVLLM.reply = "hello"
    _StubVLLM.finish_reason = "stop"
    _StubVLLM.status_code = 200
    with capture() as cap:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model", "messages": [{"role": "user", "content": "hi"}],
                  "stream": False},
            headers={"X-Conversation-Id": "rt-e12"},
        )
assert_eq(r.status_code, 500, "the simulated construction failure surfaces as a 500, not a 200")
line = _find(cap.records, "request done req=")
assert_true(line is not None, "the request done line still fired (via the wrapper's own except Exception)")
assert_true("status=error" in line.getMessage(),
            "status=error — proving _trace.finish never ran with status=ok before the "
            "exploding construction (the OLD, buggy order would have logged status=ok "
            "here, from the call that used to precede JSONResponse(...))")
assert_true("status=ok" not in line.getMessage(), "no leftover status=ok line for this request")


print("\nAll test_request_tracing.py checks passed.")
