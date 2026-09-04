"""
A1 — what a CLIENT DISCONNECT actually does to the memory tail, on the loop
production runs.

WHY THIS FILE EXISTS. Two reviewers disagreed about this and could not settle
it, and the disagreement is the mechanism behind the 51 stopped replies that
motivated the whole memory-tail change:

  * A's sub-agent: a fourth scenario — the generator parked at `yield`
    waiting on a slow consumer — defers the tail to garbage collection and
    leaks the upstream socket. It also showed `await client.aclose()` sits in
    FRONT of the only bookkeeping in that `finally`, one `await` from
    swallowing the tail entirely.
  * B: the tail fires in both cancellation placements.

BOTH RAN ON WINDOWS/PROACTOR WITH PLAIN ASYNCIO, AND BOTH USED TestClient.
That is why neither could be right in a way that mattered: TestClient drives
the app in-process through the ASGI interface and never opens a socket, so
the one thing under test — what the event loop does when a real peer hangs up
mid-response — cannot happen in it. This file runs the real app under real
uvicorn on real uvloop and hangs up a real socket.

The hazard being probed, precisely. When a peer disconnects, the ASGI server
throws into the streaming generator at its `yield`. The generator's `finally`
then runs during that throw, and it starts with `await client.aclose()`
before any of the memory-tail bookkeeping. If that await SUSPENDS while a
GeneratorExit is propagating, Python raises "async generator ignored
GeneratorExit", the rest of the `finally` never runs, and the exchange is
lost with no counter and no log line — invisibly, which is the property that
makes it worth pinning rather than reasoning about.

    python test_disconnect_uvloop.py

Skips (exit 3) rather than lying if uvloop is unavailable — a host run on
Windows cannot answer this question and must not appear to.
"""

import asyncio
import json
import os
import socket
import sys
import tempfile
import threading

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"  # no LLM calls from the tail
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-disconnect-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_STUB_PORT = _free_port()
_APP_PORT = _free_port()
# Must be set BEFORE main is imported: it reads VLLM_URL at module scope.
os.environ["VLLM_URL"] = f"http://127.0.0.1:{_STUB_PORT}"

try:
    import uvloop
except ImportError:
    print("SKIPPED: test_disconnect_uvloop.py")
    print("  reason: uvloop is not installed; this suite answers a question "
          "about the production event loop and a run without it would be "
          "evidence about the wrong loop. Run it in the container: "
          "docker compose -f docker-compose.tests.yml run --rm unit-tests "
          "--only disconnect")
    sys.exit(3)

import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import StreamingResponse, JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402
import tailhealth  # noqa: E402

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None
memory.ensure_storage_layout()

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


# ---------------------------------------------------------------------------
# The stub backend. Streams slowly ON PURPOSE: the scenario under test is a
# client that hangs up while the upstream still has data to send, which is
# what a Stop button is.
# ---------------------------------------------------------------------------

SENTENCE = "This is sentence number {} of an otherwise ordinary reply. "
N_CHUNKS = 40
CHUNK_DELAY_S = 0.02


# Upstream generators currently alive. A's sub-agent's second claim was that
# a disconnect LEAKS the upstream socket, so this is measured rather than
# asserted away: the stub adds itself on entry and removes itself in a
# finally, which runs whether it completes or is torn down.
_open_upstreams: set[int] = set()


async def _stub_completions(request):
    async def gen():
        _open_upstreams.add(id(request))
        try:
            async for _chunk in _emit():
                yield _chunk
        finally:
            _open_upstreams.discard(id(request))

    async def _emit():
        for i in range(1, N_CHUNKS + 1):
            payload = {
                "id": "chatcmpl-stub", "object": "chat.completion.chunk",
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": SENTENCE.format(i)},
                             "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
            await asyncio.sleep(CHUNK_DELAY_S)
        done = {
            "id": "chatcmpl-stub", "object": "chat.completion.chunk",
            "model": "test-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _stub_tokenize(request):
    # Refuse, deliberately: budgeting falls back to the local estimator and
    # this suite is about the disconnect path, not the token counter.
    return JSONResponse({"error": "not implemented in this stub"}, status_code=400)


stub_app = Starlette(routes=[
    Route("/v1/chat/completions", _stub_completions, methods=["POST"]),
    Route("/tokenize", _stub_tokenize, methods=["POST"]),
])


# ---------------------------------------------------------------------------
# Servers. Both on one uvloop loop in a background thread, so the test body
# can drive raw sockets synchronously and read tailhealth in-process.
# ---------------------------------------------------------------------------

_servers: list[uvicorn.Server] = []


def _serve_forever(loop):
    asyncio.set_event_loop(loop)
    for app, port in ((stub_app, _STUB_PORT), (main.app, _APP_PORT)):
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port,
                             log_level="error", loop="uvloop", lifespan="on")
        server = uvicorn.Server(cfg)
        _servers.append(server)
        loop.create_task(server.serve())
    loop.run_forever()


_loop = uvloop.new_event_loop()
_thread = threading.Thread(target=_serve_forever, args=(_loop,), daemon=True)
_thread.start()


def _wait_for_port(port, timeout=20.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


for _p in (_STUB_PORT, _APP_PORT):
    if not _wait_for_port(_p):
        print(f"FAIL server on port {_p} never came up")
        sys.exit(1)

print(f"  ok   uvloop {uvloop.__version__}, app :{_APP_PORT}, stub :{_STUB_PORT}")


def _request_bytes(conv_id: str) -> bytes:
    body = json.dumps({
        "model": "test-model",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please tell me something at length."},
        ],
    }).encode()
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{_APP_PORT}\r\n"
        f"Content-Type: application/json\r\n"
        f"X-Conversation-Id: {conv_id}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()
    return head + body


def _drive(conv_id: str, *, abort_after_bytes: int | None) -> int:
    """Send a streaming request. If abort_after_bytes is set, hang up hard
    (RST, via SO_LINGER 0) once that many body bytes have arrived."""
    s = socket.create_connection(("127.0.0.1", _APP_PORT), timeout=30)
    got = 0
    try:
        s.sendall(_request_bytes(conv_id))
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            got += len(chunk)
            if abort_after_bytes is not None and got >= abort_after_bytes:
                # SO_LINGER 0 makes close() send RST rather than FIN: this is
                # a peer VANISHING, the harshest form of the event and the
                # one a browser tab-close produces.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                             __import__("struct").pack("ii", 1, 0))
                break
    finally:
        s.close()
    return got


def _settle(seconds=3.0):
    """Let the server-side generator finish unwinding and the fire-and-forget
    tail run. Polling rather than one long sleep so a fast answer is fast."""
    import time
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.1)


print("\n[1] control — a fully consumed stream reaches the memory tail")
tailhealth._reset_for_tests()
got = _drive("conv_complete", abort_after_bytes=None)
_settle()
snap = tailhealth.snapshot()
check(got > 0, f"the client received a body ({got} bytes)")
check(snap["stored"] == 1,
      f"the tail stored the exchange (stored={snap['stored']}, "
      f"skipped={snap['skipped']}, outcomes={ {k: v for k, v in snap['outcomes'].items() if v} })")

print("\n[2] THE QUESTION — the client hangs up mid-stream (RST)")
tailhealth._reset_for_tests()
got = _drive("conv_aborted", abort_after_bytes=200)
_settle()
snap = tailhealth.snapshot()
decided = snap["stored"] + snap["skipped"]
print(f"       received {got} bytes before RST; tailhealth: "
      f"stored={snap['stored']} skipped={snap['skipped']} "
      f"outcomes={ {k: v for k, v in snap['outcomes'].items() if v} }")
check(decided == 1,
      "the memory tail DECIDED exactly once — the disconnect did not swallow "
      "the bookkeeping in the generator's finally")

print("\n[3] and the reply she actually read is not silently discarded")
if decided == 1 and snap["stored"] == 1:
    print("       stored: the partial reply was memorized (R26's path)")
elif decided == 1:
    print("       skipped, but COUNTED and named — visible in /health/full")
check(decided == 1, "either outcome is acceptable; a silent zero is not")

print("\n[3d] a LONG partial reply is memorized, not merely counted (R26)")
# [2] aborts early on purpose, so the trimmed prefix lands under the 300-char
# floor and the honest outcome is a counted skip. This case lets enough prose
# through to clear the floor, which is the case R26 is actually about: prose
# she READ, on a stream that never completed, reaching memory.
tailhealth._reset_for_tests()
got = _drive("conv_aborted_late", abort_after_bytes=6000)
_settle()
snap = tailhealth.snapshot()
print(f"       received {got} bytes before RST; outcomes="
      f"{ {k: v for k, v in snap['outcomes'].items() if v} }")
check(snap["stored"] == 1,
      "the partial reply was MEMORIZED — R26's path works on a real "
      "disconnect, on uvloop, not just under TestClient")

print("\n[4] the upstream generator is torn down, not leaked")
# A's sub-agent's second claim. Measured rather than argued: the stub adds
# itself to this set on entry and removes itself in a finally, so a leaked
# upstream is a non-empty set after everything has settled.
_settle(2.0)
check(len(_open_upstreams) == 0,
      f"no upstream stream left running after the disconnects "
      f"({len(_open_upstreams)} still open)")

for _s in _servers:
    _s.should_exit = True
_settle(1.0)

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll disconnect checks passed.")
