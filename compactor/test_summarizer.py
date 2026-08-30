"""
CPU-only smoke tests for compactor/summarizer.py (V2.0 Phase 4).

Mocks the vLLM HTTP call so no GPU / network is needed. Verifies state
storage, rollup-trigger detection, the L1/L2/L3 cascade, threshold logic,
injection block formatting, and graceful degradation.

Run:
    python test_summarizer.py
"""

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock

_TMP = tempfile.mkdtemp(prefix="compactor-test-summarizer-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
# Shrink thresholds so we exercise rollups with small fixtures.
os.environ["COMPACTOR_L1_CHUNK_SIZE"] = "4"
os.environ["COMPACTOR_L2_CHUNK_SIZE"] = "3"
os.environ["COMPACTOR_L3_CHUNK_SIZE"] = "2"

import summarizer  # noqa: E402
import logsetup  # noqa: E402
import memory  # noqa: E402


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


def _wipe():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP)
    memory.ensure_storage_layout()


# ---------------------------------------------------------------------------
# Log capture — the rollup lines are the behaviour under test below, not
# decoration. S-5 froze the hierarchy for the life of the deployment because
# a successful rollup said nothing and a latched gate said nothing either.
# ---------------------------------------------------------------------------

class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name: str = "compactor.summarizer"):
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


def find(records, needle: str):
    for r in records:
        if needle in r.getMessage():
            return r
    return None


# ---------------------------------------------------------------------------
# Mock LLM client — returns canned summaries
# ---------------------------------------------------------------------------

def _mock_client_returning(content_per_call):
    """content_per_call: either a single string (all calls return it) or
    a list (consumed in order). Returns a fake AsyncClient context manager.

    An Exception in the list is RAISED on that call instead of returned, so
    a test can fail one tier of a cascade while the earlier tiers succeed.
    """
    queue = [content_per_call] if isinstance(content_per_call, str) else list(content_per_call)

    class _Resp:
        def __init__(self, content):
            self._content = content
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    class _Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            content = queue.pop(0) if queue else "(no more canned)"
            if isinstance(content, Exception):
                raise content
            return _Resp(content)

    return _Client()


def _install_mock(content_per_call):
    """Patch httpx.AsyncClient inside summarizer to return our mock."""
    import httpx
    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: _mock_client_returning(content_per_call)
    return orig


def _install_call_recorder(reply: str = "(unexpected call)"):
    """Patch in a client that RECORDS every call and returns `reply`.
    Returns (calls, orig) where `calls` fills with the system prompt of each
    request, so a test can assert on which tier fired.

    Recording, not raising: maybe_rollup catches Exception around the whole
    cascade, so a mock that raises AssertionError to mean "should not be
    called" is swallowed and logged, and the test passes whether or not the
    call happened. Confirmed the hard way — a bare-threshold mutation of
    _needs_l3_rollup survived exactly that shape of test.
    """
    import httpx
    calls: list[str] = []

    class _Recorder:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            msgs = (kw.get("json") or {}).get("messages") or [{}]
            calls.append(msgs[0].get("content", ""))

            class _Resp:
                def raise_for_status(self): pass
                def json(self):
                    return {"choices": [{"message": {"content": reply}}]}
            return _Resp()

    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: _Recorder()
    return calls, orig


def _restore_httpx(orig):
    import httpx
    httpx.AsyncClient = orig


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _msgs(n_turns: int, system: str | None = "you are helpful"):
    """Build a [system, u1, a1, u2, a2, ...] list with n_turns non-system msgs."""
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for i in range(1, n_turns + 1):
        role = "user" if i % 2 == 1 else "assistant"
        out.append({"role": role, "content": f"msg{i}-content"})
    return out


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

def test_load_state_empty_when_no_file():
    print("\n[test] load_state returns empty skeleton when no file")
    _wipe()
    s = summarizer.load_state("never")
    assert_eq(s["conv_id"], "never", "conv_id echoed")
    assert_eq(s["l1"], [], "empty l1")
    assert_eq(s["l2"], [], "empty l2")
    assert_eq(s["l3"], None, "no l3")
    assert_eq(s["last_summarized_turn"], 0, "no turns covered yet")


def test_save_load_roundtrip():
    print("\n[test] save_state → load_state preserves content")
    _wipe()
    cid = "rt"
    state = summarizer._empty_state(cid)
    state["l1"] = [{"text": "scene 1", "first_turn": 1, "last_turn": 4}]
    state["last_summarized_turn"] = 4
    summarizer.save_state(cid, state)
    loaded = summarizer.load_state(cid)
    assert_eq(len(loaded["l1"]), 1, "one l1 chunk loaded")
    assert_eq(loaded["last_summarized_turn"], 4, "turn counter preserved")


def _unparseable_state(cid: str) -> dict:
    """A summary file mixing chunks we understand with entries we do not."""
    return {
        "conv_id": cid,
        "l1": [
            {"text": "ok", "first_turn": 1, "last_turn": 4},
            {"text": "", "first_turn": 5, "last_turn": 8},      # empty
            "not a dict",                                        # wrong type
            {"text": "no_turns"},                                # missing fields
            {"text": "also ok", "first_turn": 9, "last_turn": 12},
        ],
        "l2": [{"kind": "chapter-v2", "body": "a shape a newer build writes"}],
        "l3": None,
        "last_summarized_turn": 12,
    }


def test_load_state_parks_unrecognized_chunks():
    print("\n[test] load_state keeps malformed l1/l2 entries out of the tiers")
    _wipe()
    cid = "bad"
    summarizer.summary_path(cid).write_text(json.dumps(_unparseable_state(cid)))
    loaded = summarizer.load_state(cid)
    assert_eq(len(loaded["l1"]), 2, "filtered to 2 valid chunks")
    # They are parked, not dropped — the tiers below never see them, and the
    # next save_state puts them back (v3.1 F1b, change 4).
    parked = loaded.get(summarizer._UNRECOGNIZED) or {}
    assert_eq(len(parked.get("l1") or []), 3, "3 unrecognized l1 entries parked")
    assert_eq(len(parked.get("l2") or []), 1, "1 unrecognized l2 entry parked")


# ---------------------------------------------------------------------------
# _unrecognized round-trip (v3.1 F1b, change 4)
# ---------------------------------------------------------------------------
#
# The filter used to be silently destructive: load_state discarded whatever it
# did not recognise and the next save_state persisted the filtered list, so a
# schema change — or a single chunk written by a newer build — deleted
# summaries nobody had asked to delete. save_state runs on every rollup, so one
# read by an older build was enough. This shipped with no test: deleting the
# fold-back in _for_disk left the whole suite green.

def test_unrecognized_entries_survive_a_load_save_cycle():
    print("\n[test] load_state → save_state does not delete what it could not parse")
    _wipe()
    cid = "round-trip"
    summarizer.summary_path(cid).write_text(json.dumps(_unparseable_state(cid)))
    state = summarizer.load_state(cid)
    summarizer.save_state(cid, state)  # the destructive step, pre-v3.1

    on_disk = json.loads(summarizer.summary_path(cid).read_text(encoding="utf-8"))
    assert_eq(len(on_disk["l1"]), 5, "all 5 l1 entries back on disk")
    assert_eq(len(on_disk["l2"]), 1, "the unrecognized l2 entry back on disk")
    assert_true("not a dict" in on_disk["l1"], "the bare string survived verbatim")
    assert_true({"text": "no_turns"} in on_disk["l1"], "the partial dict survived verbatim")
    assert_true(on_disk["l2"][0]["kind"] == "chapter-v2", "the newer-build shape survived")
    # The parking key is an in-memory detail; it must not leak into the file or
    # the next reader parks the parked entries.
    assert_true(summarizer._UNRECOGNIZED not in on_disk,
                "no _unrecognized key written to disk")


def test_unrecognized_entries_survive_repeated_cycles():
    print("\n[test] the round-trip is stable — entries neither vanish nor duplicate")
    _wipe()
    cid = "round-trip-twice"
    summarizer.summary_path(cid).write_text(json.dumps(_unparseable_state(cid)))
    for _ in range(3):
        summarizer.save_state(cid, summarizer.load_state(cid))
    on_disk = json.loads(summarizer.summary_path(cid).read_text(encoding="utf-8"))
    assert_eq(len(on_disk["l1"]), 5, "still 5 l1 entries after 3 cycles")
    assert_eq(len(on_disk["l2"]), 1, "still 1 l2 entry after 3 cycles")
    reloaded = summarizer.load_state(cid)
    assert_eq(len(reloaded["l1"]), 2, "still 2 parseable chunks in the tier")


def test_unrecognized_l3_restored_when_live_l3_is_none():
    print("\n[test] an unparseable l3 is parked and folded back, not deleted")
    _wipe()
    cid = "l3-parked"
    raw = _unparseable_state(cid)
    raw["l3"] = {"summary": "an l3 shape this build does not understand"}
    summarizer.summary_path(cid).write_text(json.dumps(raw))
    state = summarizer.load_state(cid)
    assert_eq(state["l3"], None, "the unparseable l3 does not enter the tier")
    assert_eq((state.get(summarizer._UNRECOGNIZED) or {}).get("l3"), raw["l3"],
              "it is parked instead")
    summarizer.save_state(cid, state)
    on_disk = json.loads(summarizer.summary_path(cid).read_text(encoding="utf-8"))
    assert_eq(on_disk["l3"], raw["l3"], "folded back on save")


def test_real_l3_is_not_reverted_to_the_parked_one():
    print("\n[test] a rollup's real l3 is not overwritten by the parked one")
    # The one asymmetry in the fold-back: content must be preserved, but a
    # rollup that has since produced a real L3 must not be reverted to an
    # unparseable predecessor.
    _wipe()
    cid = "l3-not-reverted"
    raw = _unparseable_state(cid)
    raw["l3"] = {"summary": "the old unparseable l3"}
    summarizer.summary_path(cid).write_text(json.dumps(raw))
    state = summarizer.load_state(cid)
    state["l3"] = {"text": "a real L3 from a rollup", "first_turn": 1, "last_turn": 40}
    summarizer.save_state(cid, state)
    on_disk = json.loads(summarizer.summary_path(cid).read_text(encoding="utf-8"))
    assert_eq(on_disk["l3"]["text"], "a real L3 from a rollup", "the real L3 stands")
    assert_eq(summarizer.load_state(cid)["l3"]["text"], "a real L3 from a rollup",
              "and reloads as the live L3")


def test_clean_state_gains_no_unrecognized_key():
    print("\n[test] a state where everything parses carries no parking key")
    # Otherwise every caller that iterates or compares state dicts — and
    # state_summary, and the admin summary endpoint — starts seeing a private
    # key that was not there before.
    _wipe()
    cid = "all-clean"
    summarizer.save_state(cid, {
        "l1": [{"text": "Scene one.", "first_turn": 1, "last_turn": 20}],
        "l2": [], "l3": None, "last_summarized_turn": 20,
    })
    loaded = summarizer.load_state(cid)
    assert_true(summarizer._UNRECOGNIZED not in loaded,
                "no parking key on a fully-parseable state")
    on_disk = json.loads(summarizer.summary_path(cid).read_text(encoding="utf-8"))
    assert_true(summarizer._UNRECOGNIZED not in on_disk, "none on disk either")
    assert_eq(len(on_disk["l1"]), 1, "the one real chunk, not duplicated")


# ---------------------------------------------------------------------------
# Rollup trigger detection
# ---------------------------------------------------------------------------

def test_needs_l1_rollup_threshold():
    print("\n[test] _needs_l1_rollup respects threshold")
    state = summarizer._empty_state("c")
    state["last_summarized_turn"] = 0
    assert_eq(summarizer._needs_l1_rollup(state, 3), False, "3 < 4 → no rollup")
    assert_eq(summarizer._needs_l1_rollup(state, 4), True, "4 >= 4 → rollup")
    state["last_summarized_turn"] = 4
    assert_eq(summarizer._needs_l1_rollup(state, 7), False, "3 new < 4 → no rollup")
    assert_eq(summarizer._needs_l1_rollup(state, 8), True, "4 new turns → rollup again")


def test_needs_l2_rollup_threshold():
    print("\n[test] _needs_l2_rollup waits for enough L1 chunks")
    state = summarizer._empty_state("c")
    state["l1"] = [{"text": "x", "first_turn": 1, "last_turn": 4}] * 2
    assert_eq(summarizer._needs_l2_rollup(state), False, "2 < 3 → no L2 rollup")
    state["l1"].append({"text": "x", "first_turn": 9, "last_turn": 12})
    assert_eq(summarizer._needs_l2_rollup(state), True, "3 ≥ 3 → L2 rollup")


def test_needs_l3_rollup_threshold():
    print("\n[test] _needs_l3_rollup waits for enough L2 chapters")
    state = summarizer._empty_state("c")
    state["l2"] = [{"text": "ch", "first_turn": 1, "last_turn": 12}]
    assert_eq(summarizer._needs_l3_rollup(state), False, "1 < 2 → no L3")
    state["l2"].append({"text": "ch2", "first_turn": 13, "last_turn": 24})
    assert_eq(summarizer._needs_l3_rollup(state), True, "2 ≥ 2 → L3")
    # ...but only once for that set of chapters (S-2). _do_l3_rollup keeps the
    # L2 list, so a pure threshold is a standing condition: it stayed true
    # forever, spending one L3-sized LLM call every turn and holding
    # needs_rollup open so maybe_rollup's early exit never fired.
    state["l3"] = {"text": "theme", "first_turn": 1, "last_turn": 24}
    assert_eq(summarizer._needs_l3_rollup(state), False,
              "L3 already covers these chapters → no refresh")
    state["l2"].append({"text": "ch3", "first_turn": 25, "last_turn": 36})
    assert_eq(summarizer._needs_l3_rollup(state), True,
              "a new chapter moves the span → refresh")


# ---------------------------------------------------------------------------
# Message turn formatting
# ---------------------------------------------------------------------------

def test_format_turns_slices_correctly():
    print("\n[test] _format_turns extracts the right turn range")
    msgs = _msgs(8)
    text = summarizer._format_turns(msgs, 3, 5)
    # turn 3 = "msg3-content", 4, 5
    assert_true("msg3-content" in text, "turn 3 present")
    assert_true("msg4-content" in text, "turn 4 present")
    assert_true("msg5-content" in text, "turn 5 present")
    assert_true("msg2-content" not in text, "turn 2 excluded")
    assert_true("msg6-content" not in text, "turn 6 excluded")


def test_format_turns_skips_system():
    print("\n[test] _format_turns skips system messages, doesn't re-number")
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "another sys"},
        {"role": "user", "content": "u2"},
    ]
    text = summarizer._format_turns(msgs, 1, 3)
    assert_true("u1" in text and "a1" in text and "u2" in text, "all three non-system included")
    assert_true("sys" not in text, "system text not included")


# ---------------------------------------------------------------------------
# Injection block
# ---------------------------------------------------------------------------

def test_format_summary_block_none_when_empty():
    print("\n[test] format_summary_block returns None for empty state")
    assert_eq(summarizer.format_summary_block(summarizer._empty_state("c")), None,
              "empty -> None")


def test_format_summary_block_orders_layers():
    print("\n[test] format_summary_block: L3 → L2 → L1 in output")
    state = {
        "l1": [{"text": "scene A", "first_turn": 21, "last_turn": 24}],
        "l2": [{"text": "chapter Z", "first_turn": 1, "last_turn": 20}],
        "l3": {"text": "overall arc", "first_turn": 1, "last_turn": 100},
        "last_summarized_turn": 24,
    }
    block = summarizer.format_summary_block(state)
    assert_true("overall arc" in block, "L3 text present")
    assert_true("chapter Z" in block, "L2 text present")
    assert_true("scene A" in block, "L1 text present")
    # Most-general first
    assert_true(block.index("overall arc") < block.index("chapter Z"), "L3 before L2")
    assert_true(block.index("chapter Z") < block.index("scene A"), "L2 before L1")


# ---------------------------------------------------------------------------
# End-to-end rollup behavior (with mocked LLM)
# ---------------------------------------------------------------------------

def test_maybe_rollup_creates_l1_chunk():
    print("\n[test] maybe_rollup produces an L1 chunk when threshold met")
    _wipe()
    cid = "e2e_l1"
    msgs = _msgs(4)  # threshold is 4
    orig = _install_mock("MOCK_L1_SUMMARY")
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)
    assert_eq(len(state["l1"]), 1, "one L1 chunk produced")
    assert_eq(state["l1"][0]["text"], "MOCK_L1_SUMMARY", "L1 text from LLM")
    assert_eq(state["l1"][0]["first_turn"], 1, "first_turn 1")
    assert_eq(state["l1"][0]["last_turn"], 4, "last_turn 4")
    assert_eq(state["last_summarized_turn"], 4, "counter advanced")


def test_maybe_rollup_drains_multiple_l1():
    print("\n[test] maybe_rollup drains all eligible L1 chunks in one call")
    _wipe()
    cid = "drain"
    msgs = _msgs(12)  # 3 chunks of 4
    orig = _install_mock(["S1", "S2", "S3"])
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)
    # After drain: 3 L1 chunks would trigger L2 immediately (threshold=3).
    # So expect 0 L1 + 1 L2.
    assert_eq(len(state["l1"]), 0, "L1 drained into L2")
    assert_eq(len(state["l2"]), 1, "L2 chapter produced")


def test_maybe_rollup_l2_then_l3():
    print("\n[test] maybe_rollup cascades up to L3 when enough material")
    _wipe()
    cid = "cascade"
    msgs = _msgs(24)  # 6 L1 chunks × 4 turns. 3 L1 → L2 (×2), then 2 L2 → L3.
    canned = ["L1A", "L1B", "L1C", "L1D", "L1E", "L1F", "L2A", "L2B", "L3X"]
    orig = _install_mock(canned)
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)
    assert_eq(len(state["l1"]), 0, "all L1 chunks rolled into L2")
    # Both L2 chapters are then drained into L3 in the same pass — the same
    # consume-and-clear contract L1->L2 already had (MEMORY_REVIEW S-1/S-6
    # fix: l2 is now bounded the way l1 already was, instead of surviving
    # the L3 rollup that consumed it).
    assert_eq(len(state["l2"]), 0, "both L2 chapters drained into L3")
    assert_true(state["l3"] is not None, "L3 produced")
    assert_eq(state["l3"]["text"], "L3X", "L3 text from final LLM call")
    assert_eq(state["l3"]["first_turn"], 1, "L3 spans from the first chapter")
    assert_eq(state["l3"]["last_turn"], 24, "through the last")


def test_maybe_rollup_skips_when_not_needed():
    print("\n[test] maybe_rollup is a no-op when nothing crosses threshold")
    _wipe()
    cid = "noop"
    msgs = _msgs(2)  # below L1 threshold of 4
    calls, orig = _install_call_recorder()
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)
    assert_eq(calls, [], "no LLM call when nothing crosses a threshold")
    assert_eq(len(state["l1"]), 0, "no L1 chunks")


def test_maybe_rollup_swallows_llm_failure():
    print("\n[test] maybe_rollup never raises when LLM fails")
    _wipe()
    cid = "boom"
    msgs = _msgs(4)
    import httpx
    orig = httpx.AsyncClient

    class _Boom:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            raise RuntimeError("connection refused")

    httpx.AsyncClient = lambda *a, **kw: _Boom()
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        httpx.AsyncClient = orig
    # No crash, state remains empty-ish (no L1 produced because LLM failed).
    assert_eq(len(state["l1"]), 0, "no chunks produced on LLM failure")


# ---------------------------------------------------------------------------
# A1: the rollup input budget
# ---------------------------------------------------------------------------
#
# Until v3.1 this module had no token accounting at all: rollup input was
# bounded by turn COUNT and by nothing else. On the conversation behind
# INCIDENT_2026-08-28 twenty turns is 1.6-3.5x the window, so every L1 rollup
# 400'd — and because raise_for_status fires before the watermark write, the
# identical doomed request was re-issued on the tail of every turn, forever.
#
# Every mock above shares the blind spot that let that ship: it returns a
# canned reply no matter how large the request is. A server that always says
# yes cannot fail a test about sending it too much. So the fixture below
# CHARGES for what it is sent and REFUSES what does not fit, the way vLLM
# does, and each test asserts against a body it first proves is oversized —
# so none of them can pass vacuously.


class _FakeVllm:
    """A vLLM-shaped server with a context window it actually enforces.

    `density` is tokens per character, uniform, so a test can compute ground
    truth. Real tokenizers are not uniform; that is the point of measuring
    through /tokenize rather than a multiplier, and it is what the fixture is
    standing in for.
    """

    def __init__(self, density=1.0, window=8192, reply="SUMMARY",
                 tokenize_ok=True):
        self.density = density
        self.window = window
        self.reply = reply
        self.tokenize_ok = tokenize_ok
        self.tokenize_calls = 0
        self.chat_calls: list[tuple[str, str, int]] = []  # (system, body, cost)
        self.refusals = 0

    def cost(self, text: str) -> int:
        return int(len(text) * self.density)

    # -- the two endpoints ---------------------------------------------------

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def post(self, url, **kw):
        body = kw.get("json") or {}
        if str(url).endswith("/tokenize"):
            self.tokenize_calls += 1
            if not self.tokenize_ok:
                raise RuntimeError("connection refused")
            return _FakeResp({"count": self.cost(body.get("prompt") or "")})
        msgs = body.get("messages") or []
        system = msgs[0].get("content", "") if msgs else ""
        text = msgs[1].get("content", "") if len(msgs) > 1 else ""
        cost = self.cost(text)
        self.chat_calls.append((system, text, cost))
        if cost + int(body.get("max_tokens") or 0) > self.window:
            # The 400 the whole finding is about.
            self.refusals += 1
            raise RuntimeError(
                f"400 Bad Request: this model's maximum context length is "
                f"{self.window} tokens. However, your prompt contains {cost} "
                f"input tokens."
            )
        return _FakeResp({"choices": [{"message": {"content": self.reply}}]})

    # -- assertions the tests share -----------------------------------------

    @property
    def costs(self) -> list[int]:
        return [c for _, _, c in self.chat_calls]

    def prompts_used(self) -> list[str]:
        return [s for s, _, _ in self.chat_calls]


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def raise_for_status(self): pass

    def json(self): return self._payload


def _install_server(srv: _FakeVllm):
    import httpx
    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: srv
    return orig


@contextlib.contextmanager
def window(max_model_len: int):
    """Shrink the module's idea of the context window for one test.

    Patched rather than set through the environment because summarizer reads
    MAX_MODEL_LEN at import and _input_budget reads the module global at call
    time — which is also what lets a deployment's window change take effect
    without a code change.
    """
    prev = summarizer.MAX_MODEL_LEN
    summarizer.MAX_MODEL_LEN = max_model_len
    try:
        yield
    finally:
        summarizer.MAX_MODEL_LEN = prev


def _fat_msgs(n_turns: int, chars: int = 2000, system: str | None = "sys"):
    """Like _msgs, but each turn is big enough to matter to a budget."""
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for i in range(1, n_turns + 1):
        role = "user" if i % 2 == 1 else "assistant"
        out.append({"role": role, "content": f"t{i}." + ("x" * chars)})
    return out


def test_input_budget_is_clamped_to_the_window():
    print("\n[test] _input_budget never exceeds the window, never goes negative")
    with window(32768):
        assert_eq(summarizer._input_budget(500), 32768 - 500 - 2048,
                  "the ordinary case is window minus output minus reserve")
    # A small-context model: window minus output minus reserve is NEGATIVE, so
    # the floor takes over. Un-clamped, max(256, ...) alone would then sit
    # ABOVE the model's own window and quietly reintroduce the overflow this
    # exists to prevent — the same clamp main.HARD_INPUT_LIMIT carries.
    with window(1000):
        assert_eq(summarizer._input_budget(500), 256, "floor applies, not the negative")
    with window(200):
        assert_eq(summarizer._input_budget(0), 200,
                  "and the floor is capped by the window itself")


def test_a_body_that_fits_costs_no_tokenize_call():
    print("\n[test] a normal-sized rollup is byte-for-byte the old single call")
    # The budget must not turn every rollup into a metering exercise: when the
    # whole body fits even at the pessimistic ceiling there is nothing to
    # decide. This is the property that makes the fix free in the common case.
    _wipe()
    srv = _FakeVllm()
    orig = _install_server(srv)
    try:
        state = asyncio.run(summarizer.maybe_rollup("cheap", _msgs(4), "http://x", "m"))
    finally:
        _restore_httpx(orig)
    assert_eq(srv.tokenize_calls, 0, "no /tokenize call for a body that obviously fits")
    assert_eq(len(srv.chat_calls), 1, "exactly one summarize call — no map-reduce")
    assert_eq(srv.prompts_used(), [summarizer._PROMPT_L1], "and it is the L1 prompt")
    assert_eq(state["last_summarized_turn"], 4, "chunk produced")


def test_oversized_l1_body_is_split_instead_of_refused():
    print("\n[test] an L1 body over the window is split, and every part fits")
    _wipe()
    cid = "a1-split"
    msgs = _fat_msgs(4, chars=2000)
    srv = _FakeVllm(density=1.0, window=8192)
    with window(8192):
        budget = summarizer._input_budget(summarizer.L1_MAX_TOKENS)
        whole = srv.cost(summarizer._format_turns(msgs, 1, 4))
        # Guard against a vacuous test: the body this fixture builds must
        # really be one the pre-fix code would have sent whole and had refused.
        assert_true(whole > budget, f"the fixture is oversized ({whole} > {budget})")
        assert_true(whole + summarizer.L1_MAX_TOKENS > 8192,
                    "and the un-split body would have been a 400")
        orig = _install_server(srv)
        try:
            state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
        finally:
            _restore_httpx(orig)

    assert_eq(srv.refusals, 0, "the server refused nothing")
    assert_true(max(srv.costs) <= budget,
                f"no request exceeded the budget (largest {max(srv.costs)} <= {budget})")
    assert_true(len(srv.chat_calls) > 1, "it really did split — this is the map step")
    assert_true(summarizer._PROMPT_REDUCE in srv.prompts_used(),
                "and the parts were folded, not concatenated blind")
    # The turn range is the contract the watermark and the L2 rollup depend
    # on. Splitting the REQUEST must not change what the CHUNK claims to cover.
    assert_eq(len(state["l1"]), 1, "one chunk, however many calls it took")
    assert_eq(state["l1"][0]["first_turn"], 1, "still covers from turn 1")
    assert_eq(state["l1"][0]["last_turn"], 4, "still covers through turn 4")
    assert_eq(state["last_summarized_turn"], 4, "and the watermark advanced")


def test_the_hierarchy_no_longer_latches_on_an_oversized_conversation():
    print("\n[test] a conversation too big to summarize whole still advances")
    # This is A1 itself. Pre-fix: the 400 fires inside _llm_summarize, before
    # the watermark write, so last_summarized_turn never moves,
    # _needs_l1_rollup stays true forever, and the identical doomed request is
    # re-issued on every subsequent turn. L1 never grows, so L2 and L3 never
    # fire either. Here the backlog is drained and the second pass is quiet.
    _wipe()
    cid = "a1-latch"
    msgs = _fat_msgs(8, chars=2000)
    srv = _FakeVllm(density=1.0, window=8192)
    with window(8192):
        orig = _install_server(srv)
        try:
            state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
        finally:
            _restore_httpx(orig)
        assert_eq(srv.refusals, 0, "nothing was refused")
        assert_eq(len(state["l1"]), 2, "both 4-turn chunks were produced")
        assert_eq(state["last_summarized_turn"], 8, "the watermark caught up")
        on_disk = json.loads(summarizer.summary_path(cid).read_text(encoding="utf-8"))
        assert_eq(on_disk["last_summarized_turn"], 8, "and it is persisted")

        # The half that proves the latch is gone: the same history next turn
        # is now a no-op, rather than the same doomed work again.
        quiet = _FakeVllm(density=1.0, window=8192)
        orig = _install_server(quiet)
        try:
            asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
        finally:
            _restore_httpx(orig)
    assert_eq(quiet.chat_calls, [], "no LLM call on a turn that added nothing")


def test_a_single_turn_over_the_budget_is_truncated_not_dropped():
    print("\n[test] one enormous turn is truncated so the hierarchy keeps moving")
    # main.summarize's _chunk_to_budget deliberately does NOT truncate — it
    # gives an oversized turn its own batch and lets the call fail, because
    # compaction degrading still yields a reply. Here the same trade is wrong:
    # a rollup that cannot fit its input LATCHES. Losing the tail of one turn
    # is cheaper than losing every summary after it.
    _wipe()
    logsetup._reset_log_once_for_tests()
    cid = "a1-truncate"
    msgs = _fat_msgs(4, chars=40)
    msgs[1]["content"] = "huge." + ("y" * 8000)  # turn 1, alone over the budget
    srv = _FakeVllm(density=1.0, window=8192)
    with window(8192):
        budget = summarizer._input_budget(summarizer.L1_MAX_TOKENS)
        assert_true(srv.cost(msgs[1]["content"]) > budget,
                    "the fixture turn really does not fit on its own")
        orig = _install_server(srv)
        try:
            with capture() as cap:
                state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
        finally:
            _restore_httpx(orig)

    assert_eq(srv.refusals, 0, "the oversized turn was not sent whole")
    assert_true(max(srv.costs) <= budget, "every request still fit the budget")
    sent = "\n".join(b for _, b, _ in srv.chat_calls)
    assert_true(summarizer._TRUNCATION_NOTE.strip() in sent,
                "the model was told the turn was cut, not handed a silent stub")
    warned = find(cap.records, "has been truncated")
    assert_true(warned is not None, "and the operator was told which conversation")
    assert_eq(warned.levelno, logging.WARNING, "at WARNING — this is lossy")
    assert_eq(state["last_summarized_turn"], 4, "the hierarchy advanced anyway")


class _NonUniformVllm(_FakeVllm):
    """A server whose price per character depends on WHICH characters.

    Every fixture above charges a flat rate, and a flat rate is the one thing
    INCIDENT_2026-08-28 proves this content is not: box-drawing runs ~2.0
    tokens/char while prose runs ~0.1-0.25. A uniform mock cannot catch a bug
    whose cause is non-uniformity, and the truncation path had exactly that
    bug.
    """

    DENSE = "━"  # U+2501 — 1,710 of these in one production reply

    def cost(self, text: str) -> int:
        return int(sum(2.0 if c == self.DENSE else 0.1 for c in text))


def test_truncation_measures_the_cut_instead_of_scaling_characters():
    print("\n[test] truncating a dense-headed turn does not overflow anyway")
    # The first version of _truncate_to_budget scaled characters by the token
    # ratio and trusted the result — the A4 unit error committed inside A1's
    # own fix. It assumes tokens spread evenly across characters. A turn whose
    # DENSE part comes first prices its head far above its average, so the
    # proportional cut keeps a prefix that still overflows, the request is
    # refused, and the watermark stays put: the latch, restored by the one
    # code path that exists to prevent it.
    _wipe()
    logsetup._reset_log_once_for_tests()
    cid = "a1-nonuniform"
    srv = _NonUniformVllm(window=8192)
    dense_headed = _NonUniformVllm.DENSE * 10000 + "w" * 90000
    msgs = _fat_msgs(4, chars=40)
    msgs[1]["content"] = dense_headed

    with window(8192):
        budget = summarizer._input_budget(summarizer.L1_MAX_TOKENS)
        # The shape of the trap, stated as arithmetic so the fixture cannot
        # drift out from under the test.
        measured = srv.cost(dense_headed)
        proportional = max(1, int(len(dense_headed) * (budget / measured) * 0.9))
        assert_true(srv.cost(dense_headed[:proportional]) > budget,
                    f"a proportional cut would still cost "
                    f"{srv.cost(dense_headed[:proportional])} against {budget}")
        orig = _install_server(srv)
        try:
            state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
        finally:
            _restore_httpx(orig)

    assert_eq(srv.refusals, 0, "the cut piece actually fit")
    assert_true(max(srv.costs) <= budget,
                f"largest request {max(srv.costs)} within budget {budget}")
    assert_eq(state["last_summarized_turn"], 4, "and the hierarchy advanced")


def test_truncation_still_fits_when_tokenize_is_down():
    print("\n[test] the truncation backstop holds with no server to ask")
    # With /tokenize down every measurement is the pessimistic ceiling, so the
    # re-measure loop cannot converge. The backstop is arithmetic rather than
    # another guess: at most budget/_WORST_TOKENS_PER_CHAR characters cannot
    # exceed budget tokens unless the content beats the worst density this
    # project has ever measured.
    _wipe()
    logsetup._reset_log_once_for_tests()
    cid = "a1-trunc-blind"
    srv = _NonUniformVllm(window=8192, tokenize_ok=False)
    msgs = _fat_msgs(4, chars=40)
    msgs[1]["content"] = _NonUniformVllm.DENSE * 40000

    with window(8192):
        budget = summarizer._input_budget(summarizer.L1_MAX_TOKENS)
        orig = _install_server(srv)
        try:
            state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
        finally:
            _restore_httpx(orig)

    assert_eq(srv.refusals, 0, "nothing was refused even measuring blind")
    assert_true(max(srv.costs) <= budget,
                f"largest request {max(srv.costs)} within budget {budget}")
    assert_eq(state["last_summarized_turn"], 4, "the hierarchy advanced anyway")


def test_a_tokenize_outage_falls_back_pessimistically_never_optimistically():
    print("\n[test] when /tokenize is down the budget errs high, not low")
    # The fallback density is a ceiling (2.0 tokens/char, the worst this
    # project has measured — INCIDENT_2026-08-28's box-drawing reply), not a
    # prose multiplier. A prose multiplier is wrong on that content by ~8x and
    # would only move the failure. Being wrong HIGH over-splits; being wrong
    # low is the incident.
    _wipe()
    logsetup._reset_log_once_for_tests()
    cid = "a1-blind"
    msgs = _fat_msgs(4, chars=2000)
    srv = _FakeVllm(density=1.0, window=8192, tokenize_ok=False)
    with window(8192):
        budget = summarizer._input_budget(summarizer.L1_MAX_TOKENS)
        orig = _install_server(srv)
        try:
            with capture() as cap:
                state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
        finally:
            _restore_httpx(orig)

    assert_true(srv.tokenize_calls > 0, "it did try to measure")
    assert_eq(srv.refusals, 0, "and still never sent an over-budget request")
    assert_true(max(srv.costs) <= budget, "every request inside the budget")
    assert_eq(state["last_summarized_turn"], 4, "the rollup completed blind")
    warned = find(cap.records, "/tokenize unreachable")
    assert_true(warned is not None, "the outage is reported")
    assert_eq(warned.levelno, logging.WARNING, "at WARNING")
    assert_true(str(summarizer._WORST_TOKENS_PER_CHAR) in warned.getMessage(),
                "naming the density it fell back to")


def test_l3_input_is_bounded_too():
    print("\n[test] L3 — whose input grows without limit — is budgeted as well")
    # _do_l3_rollup joins ALL L2 chapters and nothing trims l2 (MEMORY_REVIEW
    # S-1), so this input grows with the conversation. Slower than L1's
    # overflow, same shape, and a worse ending: L3 is the tier that never
    # refires once it fails, because _needs_l3_rollup keeps returning True.
    _wipe()
    cid = "a1-l3"
    state = summarizer._empty_state(cid)
    state["l2"] = [
        {"text": f"chapter{i}." + ("z" * 3000), "first_turn": 1 + 12 * i,
         "last_turn": 12 + 12 * i}
        for i in range(3)
    ]
    state["last_summarized_turn"] = 4
    summarizer.save_state(cid, state)

    srv = _FakeVllm(density=1.0, window=8192, reply="THEME")
    with window(8192):
        budget = summarizer._input_budget(summarizer.L3_MAX_TOKENS)
        whole = srv.cost("\n\n".join(c["text"] for c in state["l2"]))
        assert_true(whole > budget, f"the chapter set is oversized ({whole} > {budget})")
        orig = _install_server(srv)
        try:
            # 4 observable turns against a watermark of 4: no L1 work is due,
            # so the only tier that can fire is L3.
            out = asyncio.run(summarizer.maybe_rollup(cid, _msgs(4), "http://x", "m"))
        finally:
            _restore_httpx(orig)

    assert_eq(srv.refusals, 0, "no L3 request was refused")
    assert_true(max(srv.costs) <= budget, "every L3 request fit the budget")
    assert_true(out["l3"] is not None, "the theme landed")
    assert_eq(out["l3"]["first_turn"], 1, "spanning the first chapter")
    assert_eq(out["l3"]["last_turn"], 36, "through the last")


def test_the_split_does_not_change_which_tier_prompt_is_used():
    print("\n[test] map uses the tier's own prompt; only the fold uses reduce")
    # A split must not silently demote L1 content to a generic summarization.
    # And the fold says "merge these consecutive parts", not "summarize this
    # again" — a second summarization pass is exactly the summary-of-summary
    # degradation the tiering exists to avoid.
    _wipe()
    srv = _FakeVllm(density=1.0, window=8192)
    with window(8192):
        orig = _install_server(srv)
        try:
            asyncio.run(summarizer.maybe_rollup("a1-prompts", _fat_msgs(4, 2000),
                                                "http://x", "m"))
        finally:
            _restore_httpx(orig)
    used = srv.prompts_used()
    assert_true(used.count(summarizer._PROMPT_L1) >= 2, "each map batch got the L1 prompt")
    assert_eq(used[-1], summarizer._PROMPT_REDUCE, "and the last call is the fold")
    assert_eq(used.count(summarizer._PROMPT_REDUCE), 1, "folded once, not repeatedly")


# ---------------------------------------------------------------------------
# S-5: the watermark latch that froze the hierarchy in production
# ---------------------------------------------------------------------------
#
# last_summarized_turn is an absolute position in whatever array the client
# sent; the gate compares it against the non-system count of the array in
# hand. When the second is smaller than the first — a bounded window, a
# deleted or edited message, a branch switch — the delta is negative and
# _needs_l1_rollup is False on this turn and every turn after it. 19.8 hours
# of production logs show every injection reading L1=5 / L2=0 while the
# conversation ran from turn ~42 to ~58: the hierarchy never rolled once.

def _stranded_state(cid: str, watermark: int = 100):
    """A conversation whose watermark is far ahead of any history a client
    is going to send back."""
    state = summarizer._empty_state(cid)
    state["l1"] = [{"text": "an earlier scene", "first_turn": 1, "last_turn": 4}]
    state["last_summarized_turn"] = watermark
    summarizer.save_state(cid, state)


def test_shortened_history_resets_the_watermark():
    print("\n[test] a history shorter than the watermark resets it, not latches")
    _wipe()
    cid = "latched"
    _stranded_state(cid)
    msgs = _msgs(8)  # 8 observable turns against a watermark of 100
    calls, orig = _install_call_recorder()
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)

    assert_eq(calls, [], "repairing the counter does not re-summarize anything")
    assert_eq(state["last_summarized_turn"], 8, "watermark pulled back to the observed count")
    assert_eq(len(state["l1"]), 1, "the stranded chunk is kept, not deleted")
    on_disk = json.loads(summarizer.summary_path(cid).read_text(encoding="utf-8"))
    assert_eq(on_disk["last_summarized_turn"], 8, "and the repair is persisted")
    assert_eq(len(on_disk["l1"]), 1, "the chunk is still on disk too")


def test_reset_watermark_lets_rollups_resume():
    print("\n[test] after the reset the hierarchy actually advances again")
    # The reset is only worth anything if the next threshold crossing rolls.
    _wipe()
    cid = "unlatched"
    _stranded_state(cid)
    orig = _install_mock("RESUMED")
    try:
        # Turn A: 8 observable turns — repairs the watermark, no material yet.
        asyncio.run(summarizer.maybe_rollup(cid, _msgs(8), "http://x", "m"))
        # Turn B: 12 observable turns — 4 new, which is the L1 threshold here.
        state = asyncio.run(summarizer.maybe_rollup(cid, _msgs(12), "http://x", "m"))
    finally:
        _restore_httpx(orig)
    assert_eq(len(state["l1"]), 2, "a new chunk on top of the kept one")
    assert_eq(state["l1"][-1]["text"], "RESUMED", "it came from the LLM")
    assert_eq(state["l1"][-1]["first_turn"], 9, "covers the turns after the reset")
    assert_eq(state["l1"][-1]["last_turn"], 12, "up to the observed count")
    assert_eq(state["last_summarized_turn"], 12, "watermark advanced")


def test_negative_delta_warns_once_per_process():
    print("\n[test] the latched gate warns at WARNING, once per process")
    # A negative delta is indistinguishable from healthy quiet in the log —
    # both are silence — which is why this ran for 19.8 hours unnoticed. It
    # is once per process because maybe_rollup is on the tail of every turn.
    _wipe()
    logsetup._reset_log_once_for_tests()
    _stranded_state("warn-a")
    _stranded_state("warn-b")
    with capture() as cap:
        asyncio.run(summarizer.maybe_rollup("warn-a", _msgs(8), "http://x", "m"))
        asyncio.run(summarizer.maybe_rollup("warn-b", _msgs(8), "http://x", "m"))
    warnings = [r for r in cap.records if r.levelno == logging.WARNING]
    assert_eq(len(warnings), 1, "exactly one warning across two stranded convs")
    assert_true("shorter than last_summarized_turn" in warnings[0].getMessage(),
                "and it names the condition")
    assert_true("100" in warnings[0].getMessage(), "reporting the stale watermark")
    logsetup._reset_log_once_for_tests()


# ---------------------------------------------------------------------------
# Rollup observability
# ---------------------------------------------------------------------------

def test_rollup_logs_a_success_line():
    print("\n[test] each tier logs when it produces something")
    # There was no success line at all, so the only evidence the hierarchy was
    # advancing was the injection counter in the request path.
    _wipe()
    cid = "logged"
    msgs = _msgs(24)  # 6 L1 chunks → 2 L2 chapters → 1 L3
    canned = ["L1A", "L1B", "L1C", "L1D", "L1E", "L1F", "L2A", "L2B", "L3X"]
    orig = _install_mock(canned)
    try:
        with capture() as cap:
            asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)

    first = find(cap.records, "L1 rollup — chunk 1 covers turns 1-4")
    assert_true(first is not None, "the first L1 rollup logged its turn range")
    assert_eq(first.levelno, logging.INFO, "at INFO — a healthy rollup is not a warning")
    assert_true(find(cap.records, "L1 rollup — chunk 3 covers turns 9-12") is not None,
                "the chunk index advances with the list")
    l2 = find(cap.records, "L2 rollup — chapter 1 covers turns 1-12")
    assert_true(l2 is not None, "the L2 chapter logged its turn range")
    assert_eq(l2.levelno, logging.INFO, "L2 at INFO too")
    l3 = find(cap.records, "L3 refresh — covers turns 1-24 over 2 chapters")
    assert_true(l3 is not None, "the L3 refresh logged its coverage")
    assert_eq(l3.levelno, logging.INFO, "L3 at INFO too")


# ---------------------------------------------------------------------------
# S-3: a failed tier must not discard the tiers that succeeded
# ---------------------------------------------------------------------------

def test_failed_l3_does_not_discard_successful_l1_and_l2():
    print("\n[test] an L3 that fails keeps the L1/L2 rollups that succeeded")
    # save_state used to sit inside the same try as the L3 call, so a single
    # oversized L3 body threw away every rollup of that pass — and, since the
    # input is identical next turn, of every pass after it, forever, while
    # spending the same LLM calls each time.
    _wipe()
    cid = "l3-fails"
    msgs = _msgs(24)
    canned = ["L1A", "L1B", "L1C", "L1D", "L1E", "L1F", "L2A", "L2B",
              RuntimeError("400 Bad Request: input too long")]
    orig = _install_mock(canned)
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)

    assert_eq(state["l3"], None, "L3 did not land")
    assert_eq(len(state["l2"]), 2, "both chapters survived in memory")
    on_disk = summarizer.load_state(cid)
    assert_eq(len(on_disk["l2"]), 2, "and both are on disk")
    assert_eq(on_disk["last_summarized_turn"], 24, "the watermark advanced on disk")
    assert_eq(on_disk["l3"], None, "no partial L3 written")


def test_failed_l3_does_not_repeat_the_same_work_forever():
    print("\n[test] the retry after a failed L3 is the L3 only")
    # The proof that the loss was permanent: with the write discarded, the
    # next turn re-ran the identical 6 L1 + 2 L2 + 1 L3 calls. With the
    # successful tiers persisted, only the L3 is outstanding.
    _wipe()
    cid = "l3-retry"
    msgs = _msgs(24)
    first_pass = ["L1A", "L1B", "L1C", "L1D", "L1E", "L1F", "L2A", "L2B",
                  RuntimeError("400 Bad Request: input too long")]
    orig = _install_mock(first_pass)
    try:
        asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)

    calls, orig = _install_call_recorder("L3X")
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)

    assert_eq(len(calls), 1, "one call on the retry, not nine")
    assert_true(calls[0] == summarizer._PROMPT_L3, "and it is the L3 that failed")
    assert_eq(state["l3"]["text"], "L3X", "which now lands")


def test_l3_does_not_refire_once_the_chapters_are_covered():
    print("\n[test] a quiet turn after L3 costs no LLM call")
    # S-2: len(l2) >= L3_CHUNK_SIZE with the L2 list retained is a standing
    # condition, so L3 regenerated on every single turn — and kept
    # needs_rollup True, defeating the early exit at the top of maybe_rollup.
    _wipe()
    cid = "l3-quiet"
    msgs = _msgs(24)
    orig = _install_mock(["L1A", "L1B", "L1C", "L1D", "L1E", "L1F",
                          "L2A", "L2B", "L3X"])
    try:
        asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)

    calls, orig = _install_call_recorder("L3-REGENERATED")
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)
    assert_eq(calls, [], "no LLM call on a turn that added nothing")
    assert_eq(state["l3"]["text"], "L3X", "the existing L3 stands unchanged")


def test_state_summary_compact():
    print("\n[test] state_summary returns admin-friendly view")
    state = {
        "l1": [{"text": "x", "first_turn": 1, "last_turn": 4}],
        "l2": [],
        "l3": {"text": "y", "first_turn": 1, "last_turn": 20},
        "last_summarized_turn": 20,
    }
    s = summarizer.state_summary(state)
    assert_eq(s["l1_chunks"], 1, "l1 count")
    assert_eq(s["l2_chapters"], 0, "l2 count")
    assert_eq(s["l3_present"], True, "l3 present flag")
    assert_eq(s["l3_turns_covered"], [1, 20], "l3 turn range")


if __name__ == "__main__":
    try:
        test_load_state_empty_when_no_file()
        test_save_load_roundtrip()
        test_load_state_parks_unrecognized_chunks()
        test_unrecognized_entries_survive_a_load_save_cycle()
        test_unrecognized_entries_survive_repeated_cycles()
        test_unrecognized_l3_restored_when_live_l3_is_none()
        test_real_l3_is_not_reverted_to_the_parked_one()
        test_clean_state_gains_no_unrecognized_key()
        test_needs_l1_rollup_threshold()
        test_needs_l2_rollup_threshold()
        test_needs_l3_rollup_threshold()
        test_format_turns_slices_correctly()
        test_format_turns_skips_system()
        test_format_summary_block_none_when_empty()
        test_format_summary_block_orders_layers()
        test_maybe_rollup_creates_l1_chunk()
        test_maybe_rollup_drains_multiple_l1()
        test_maybe_rollup_l2_then_l3()
        test_maybe_rollup_skips_when_not_needed()
        test_maybe_rollup_swallows_llm_failure()
        test_input_budget_is_clamped_to_the_window()
        test_a_body_that_fits_costs_no_tokenize_call()
        test_oversized_l1_body_is_split_instead_of_refused()
        test_the_hierarchy_no_longer_latches_on_an_oversized_conversation()
        test_a_single_turn_over_the_budget_is_truncated_not_dropped()
        test_truncation_measures_the_cut_instead_of_scaling_characters()
        test_truncation_still_fits_when_tokenize_is_down()
        test_a_tokenize_outage_falls_back_pessimistically_never_optimistically()
        test_l3_input_is_bounded_too()
        test_the_split_does_not_change_which_tier_prompt_is_used()
        test_shortened_history_resets_the_watermark()
        test_reset_watermark_lets_rollups_resume()
        test_negative_delta_warns_once_per_process()
        test_rollup_logs_a_success_line()
        test_failed_l3_does_not_discard_successful_l1_and_l2()
        test_failed_l3_does_not_repeat_the_same_work_forever()
        test_l3_does_not_refire_once_the_chapters_are_covered()
        test_state_summary_compact()
        print("\nAll summarizer smoke tests passed.")
    finally:
        if os.path.exists(_TMP):
            shutil.rmtree(_TMP, ignore_errors=True)
