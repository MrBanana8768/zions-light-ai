"""
Route-level regression test for compactor/main.py's continuity-band
injection call site (v3.1.9.8 `.15g`, §17.6a; main.py ~11261-11322).

Commit 7650668 added `tokens.raw()` at that call site without `main.py`
ever importing `tokens` — a NameError on every request that has
unsummarized turns (any conversation the L1 rollup hasn't yet caught up
with), swallowed by the site's own `except Exception` into a silent
"conv=...: continuity band injection failed (non-fatal): NameError: name
'tokens' is not defined" log line. Mirrors test_time_injection.py's route
harness: a stub httpx backend records the forwarded body, `_run_memory_tail`
is spied out (background work is not what this test is about), and a real
POST goes through `/v1/chat/completions`.

Round 2 (post-review): the band shipped with no off switch. It restates
unsummarized exchanges that are ALSO forwarded verbatim (the allocator does
not yet exclude the gap from the verbatim window), with no cap on how many
band entries can pile up. `COMPACTOR_CONTINUITY_BAND` (default False) now
gates the entire block — this file asserts BOTH directions on the
FORWARDED PAYLOAD, not just the log line: with the flag on, the band
header must actually be in what reaches vLLM; with the flag off (the
shipped default), it must not, and no "band(" log line fires either.

Run: python test_continuity_band_route.py
"""

import json
import logging
import os
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="band-route-")
os.environ["COMPACTOR_CONTROL_ROOT"] = tempfile.mkdtemp(prefix="band-route-control-")

import memory  # noqa: E402

memory.ensure_storage_layout()

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import continuity  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402
import tailhealth  # noqa: E402


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


class _Backend:
    bodies: list[dict] = []
    reply = "An ordinary reply."

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        _Backend.bodies.append(json)
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {
                "role": "assistant", "content": _Backend.reply}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", url),
        )

    async def aclose(self):
        pass


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, r):
        self.records.append(r)

    def text(self):
        return "\n".join(r.getMessage() for r in self.records)


def _spy_tail(conv_id, text, **kw):
    return main.TailDecision(False, "", tailhealth.SKIPPED_EMPTY, "spy", 0)


client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)

CONVO = [
    {"role": "system", "content": "You are a patient assistant."},
    {"role": "user", "content": "Tell me about item one."},
    {"role": "assistant", "content": "Item one is a blue box."},
    {"role": "user", "content": "And item two?"},
    {"role": "assistant", "content": "Item two is a red box."},
    {"role": "user", "content": "What about item three?"},
]


def _forwarded_text(bodies: list[dict]) -> str:
    """Every message's content, across every forwarded body, concatenated
    — so a test can assert on what actually reaches vLLM rather than on a
    log line that merely claims something was injected."""
    out = []
    for body in bodies:
        for m in (body or {}).get("messages") or []:
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                out.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(out)


def post(conv_id: str, messages: list[dict], *, band_enabled: bool = False):
    _Backend.bodies = []
    cap = _Lines()
    lg = logging.getLogger("compactor")
    prev = lg.level
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    patches = [
        patch.object(main.httpx, "AsyncClient", _Backend),
        patch.object(main, "_run_memory_tail", _spy_tail),
        patch.object(main, "_fire_and_forget", lambda coro, label=None: (coro.close(), True)[1]),
        patch.object(main, "CONTINUITY_BAND_ENABLED", band_enabled),
    ]
    try:
        for p in patches:
            p.start()
        body = {"model": "test-model", "messages": json.loads(json.dumps(messages)), "stream": False}
        r = client.post(
            "/v1/chat/completions", json=body, headers={"X-Conversation-Id": conv_id},
        )
        _ = r.text
    finally:
        for p in reversed(patches):
            p.stop()
        lg.removeHandler(cap)
        lg.setLevel(prev)
    return r.status_code, list(_Backend.bodies), cap.text()


def test_main_tokens_resolves():
    print("\n[test] main.tokens resolves to the tokens module (the NameError regression)")
    assert_true(hasattr(main, "tokens"), "main.tokens exists")
    assert_true(hasattr(main.tokens, "raw"), "main.tokens.raw exists")
    # Must not raise — this is literally the call the injection site makes.
    main.tokens.raw()
    print("  ok   main.tokens.raw() callable without raising")


def _seed_unsummarized_state(conv_id: str) -> None:
    # turns_seen(5) - last_summarized_turn(3) = 2 unsummarized turns, and
    # CONVO has 5 non-system messages -> the gap slice is non-empty (see
    # main.py's _U/_gap computation just above the band-injection block).
    summarizer.save_state(conv_id, {
        "turns_seen": 5, "last_summarized_turn": 3, "l1": [], "l2": [], "l3": None,
    })


def test_band_injection_runs_without_the_nameerror_when_flag_is_on():
    print("\n[test] flag ON: a request with unsummarized turns forwards the "
          "band header, with no 'continuity band injection failed' log line")
    conv_id = "band-route-conv-on"
    _seed_unsummarized_state(conv_id)
    status, bodies, log = post(conv_id, CONVO, band_enabled=True)
    assert_true(status == 200, f"route returns 200 (got {status})")
    assert_true("continuity band injection failed" not in log,
                f"no band-injection failure logged; log tail: {log[-800:]!r}")
    assert_true("band(" in log, f"the band actually ran and logged band(N); log tail: {log[-800:]!r}")
    forwarded = _forwarded_text(bodies)
    assert_true(continuity.BAND_HEADER in forwarded,
                f"the band header reaches the FORWARDED payload, not just the log: {forwarded[-500:]!r}")


def test_band_injection_is_fully_off_by_default():
    print("\n[test] flag OFF (the shipped default): no band header forwarded, "
          "no 'band(' log line, and the block does not run at all")
    conv_id = "band-route-conv-off"
    _seed_unsummarized_state(conv_id)
    assert_true(main.CONTINUITY_BAND_ENABLED is False,
                "sanity: the module-level default really is False")

    # Task C: the log-line and forwarded-payload assertions below are
    # necessary but not sufficient — a stray "band(" phrase elsewhere, or
    # a swallowed exception, could make them pass for the wrong reason.
    # Assert the actual I/O the band block does never happens when the
    # flag is off: continuity.visible (reads the digest store) is called
    # ONLY from inside `if CONTINUITY_BAND_ENABLED:`, so it must see zero
    # calls. summarizer.load_state is trickier — it is ALSO called
    # unconditionally earlier in the request (the Phase 4 hierarchical-
    # summary section, main.py ~11243) for reasons that have nothing to do
    # with the band, so "never called" would be the wrong assertion; the
    # band's own extra `_bstate = summarizer.load_state(conv_id)` read
    # (main.py's own comment just above `if CONTINUITY_BAND_ENABLED:`:
    # "no summarizer.load_state call, no fingerprinting, no tokens.raw()")
    # is what must not happen, so the call count is pinned at exactly the
    # one earlier, unconditional read.
    with patch.object(continuity, "visible", wraps=continuity.visible) as _visible_spy, \
         patch.object(summarizer, "load_state", wraps=summarizer.load_state) as _load_state_spy:
        status, bodies, log = post(conv_id, CONVO, band_enabled=False)
        assert_true(status == 200, f"route returns 200 (got {status})")
        assert_true(_visible_spy.call_count == 0,
                    f"continuity.visible is never called when the flag is off "
                    f"(called {_visible_spy.call_count}x) — no band digest-store read")
        assert_true(_load_state_spy.call_count == 1,
                    f"summarizer.load_state is called exactly once (the "
                    f"unconditional Phase-4 read, not the band's extra "
                    f"_bstate read) when the flag is off (called "
                    f"{_load_state_spy.call_count}x)")
    assert_true("band(" not in log, f"no band(N) log line when the flag is off; log tail: {log[-800:]!r}")
    assert_true("continuity band injection failed" not in log,
                "the gated-off block cannot fail — it never runs")
    forwarded = _forwarded_text(bodies)
    assert_true(continuity.BAND_HEADER not in forwarded,
                f"no band header reaches the forwarded payload when the flag is off: {forwarded[-500:]!r}")


if __name__ == "__main__":
    test_main_tokens_resolves()
    test_band_injection_runs_without_the_nameerror_when_flag_is_on()
    test_band_injection_is_fully_off_by_default()
    print("\nAll continuity band route tests passed.")
