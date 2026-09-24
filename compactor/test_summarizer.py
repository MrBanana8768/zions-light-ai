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
import re
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


def _install_body_recorder(reply: str = "SUMMARY"):
    """Like _install_call_recorder, but records (system prompt, body) for each
    call. The v3.1.4 capped-window tests are about WHICH TEXT reaches the
    model, not merely that a call happened: a chunk labelled 21-24 whose body
    is turns 17-20 passes every count-based assertion in this file. The system
    prompt rides along so a test can pick out one tier — an L1 assertion that
    silently matched an L2 fold would be measuring the wrong call.
    """
    import httpx
    bodies: list[str] = []

    class _Recorder:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            msgs = (kw.get("json") or {}).get("messages") or [{}, {}]
            bodies.append(
                (msgs[0].get("content", ""), msgs[-1].get("content", ""))
            )

            class _Resp:
                def raise_for_status(self): pass
                def json(self):
                    return {"choices": [{"message": {"content": reply}}]}
            return _Resp()

    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: _Recorder()
    return bodies, orig


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


def _window(total: int, cap: int, system: str | None = "you are helpful"):
    """The last `cap` non-system turns of a conversation that is `total` turns
    long — what pipelines/conversation_id_header.py's `max_turns` valve hands
    the compactor.

    Turn k has the same text at every `total`, deliberately: that is what
    OpenWebUI re-sending its own stored history looks like, and it is the only
    way an assertion can say WHICH turns a rollup summarized. _msgs(n) is
    _window(n, n).
    """
    first = max(1, total - cap + 1)
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for i in range(first, total + 1):
        role = "user" if i % 2 == 1 else "assistant"
        out.append({"role": role, "content": f"msg{i}-content"})
    return out


def _live_window(exchange: int, cap: int, system: str | None = "you are helpful"):
    """The message list main.py actually hands maybe_rollup on exchange `e`.

    NOT the same thing as _window(2*e, cap), and the difference is the whole
    of R23. The valve in pipelines/conversation_id_header.py caps what
    OPENWEBUI re-sends — the client's stored history PLUS the user turn it is
    sending, i.e. turns 1..2e-1 — and main.py then appends the assistant turn
    it has just streamed, which the valve never saw. So the window is
    `min(2e-1, cap) + 1` turns long, and on the first exchange where the cap
    bites it is one turn LONGER than the previous position while already being
    one turn SHORTER than the conversation. _window() models the transition as
    a full history followed immediately by a short window, which lands in a
    different branch entirely and is why the whole suite stayed green while
    a turn was being dropped.
    """
    hist = [
        {"role": "user" if i % 2 == 1 else "assistant",
         "content": f"msg{i}-content"}
        for i in range(1, 2 * exchange)
    ]
    if cap > 0 and len(hist) > cap:
        hist = hist[-cap:]
    out = ([{"role": "system", "content": system}] if system else []) + hist
    out.append({"role": "assistant", "content": f"msg{2 * exchange}-content"})
    return out


def _labels_logged(records) -> list[tuple[int, int]]:
    """Every span an L1 rollup CLAIMED, in the order it claimed them.

    Read out of the success line rather than out of state["l1"], because
    sampling the list cannot see a chunk that an L2 rollup folded away inside
    the same maybe_rollup call — and past the L2 threshold that is most of
    them. The label at the moment of creation is the thing under test: a
    chunk labelled 9-12 whose body is turns 10-13 is what R23 produced, and
    it is invisible to any assertion that counts chunks.
    """
    out = []
    for r in records:
        m = re.search(r"L1 rollup — chunk \d+ covers turns (\d+)-(\d+)",
                      r.getMessage())
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def _turns_in(body: str, upto: int) -> list[int]:
    """Which client turns' text is actually inside one request body."""
    return [t for t in range(1, upto + 1) if f"msg{t}-content" in body]


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


def test_shortened_history_does_not_pull_the_watermark_back():
    print("\n[test] a history shorter than the watermark does not rewind it")
    # v3.1.4 REVERSES the S-5 repair. Pulling the watermark down to the
    # observed count un-latched a ONE-OFF shrink at the price of "the turns
    # between will be summarized a second time", and under a PERMANENT cap
    # that price is the whole hierarchy: the pull-down lands once, the delta
    # is 0 from then on, and no chunk is ever produced again. The position
    # now moves forward only.
    _wipe()
    cid = "latched"
    _stranded_state(cid)
    msgs = _msgs(8)  # 8 observable turns against a watermark of 100
    calls, orig = _install_call_recorder()
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        _restore_httpx(orig)

    assert_eq(calls, [], "nothing is re-summarized")
    assert_eq(state["last_summarized_turn"], 100, "the watermark stands")
    assert_eq(state["turns_seen"], 100, "and the position is seeded from it")
    assert_eq(len(state["l1"]), 1, "the stranded chunk is kept, not deleted")
    on_disk = json.loads(summarizer.summary_path(cid).read_text(encoding="utf-8"))
    assert_eq(on_disk["last_summarized_turn"], 100, "unchanged on disk")
    assert_eq(on_disk["turns_seen"], 100, "the position IS persisted")
    assert_true(bool(on_disk["tail_fp"]), "and so is the anchor it needs")
    assert_eq(len(on_disk["l1"]), 1, "the chunk is still on disk too")


def test_a_stranded_watermark_still_unlatches():
    print("\n[test] a stranded watermark still un-latches, without a rewind")
    # S-5's own case has to keep working: the gate must open again after
    # L1_CHUNK_SIZE further turns, which is exactly what the reset bought.
    # It does — the position advances with the conversation instead of being
    # snapped to the client's array.
    _wipe()
    cid = "unlatched"
    _stranded_state(cid)          # watermark 100, one chunk covering 1-4
    orig = _install_mock("RESUMED")
    try:
        # A fixed 8-turn window over a conversation that runs 8 -> 12 turns.
        for total in (8, 10, 12):
            state = asyncio.run(
                summarizer.maybe_rollup(cid, _window(total, 8), "http://x", "m")
            )
    finally:
        _restore_httpx(orig)
    assert_eq(state["turns_seen"], 104, "four further turns past the strand")
    assert_eq(len(state["l1"]), 2, "a new chunk on top of the kept one")
    assert_eq(state["l1"][-1]["text"], "RESUMED", "it came from the LLM")
    assert_eq(state["l1"][-1]["first_turn"], 101, "covering the turns since")
    assert_eq(state["l1"][-1]["last_turn"], 104, "through the position")
    assert_eq(state["last_summarized_turn"], 104, "watermark advanced")


# ---------------------------------------------------------------------------
# v3.1.4: the PERMANENTLY capped client window
# ---------------------------------------------------------------------------
#
# pipelines/conversation_id_header.py's `max_turns` valve caps how many turns
# OpenWebUI re-sends (100 is its documented starting value). The old gate was a
# difference against the array's length, so a constant length meant a constant
# difference: _reconcile_watermark fired once, and _needs_l1_rollup was False
# on every request after it. No L1, therefore no L2, therefore no L3 — and the
# only line saying so was logsetup.log_once, i.e. once per process.
#
# The tests below run the cap at 8 turns against this file's L1_CHUNK_SIZE=4,
# the same 25:1 window-to-chunk shape production has at 100:20... (2:1 here,
# deliberately tighter, so a mapping error walks off the end of the window
# instead of landing on plausible-looking neighbouring text).

def test_a_permanently_capped_window_still_rolls_up():
    print("\n[test] a permanently fixed window does not freeze the hierarchy")
    _wipe()
    cid = "capped"
    orig = _install_mock("CHUNK")
    try:
        # 20 turns of full history first, then the valve goes on at 8 and the
        # conversation runs to turn 40 — 20 further turns, five chunks' worth.
        asyncio.run(summarizer.maybe_rollup(cid, _msgs(20), "http://x", "m"))
        for total in range(22, 41, 2):
            state = asyncio.run(
                summarizer.maybe_rollup(cid, _window(total, 8), "http://x", "m")
            )
    finally:
        _restore_httpx(orig)

    assert_eq(state["turns_seen"], 40, "the position tracked the conversation")
    assert_eq(state["last_summarized_turn"], 40,
              "and the watermark kept pace with it under the cap")
    covered = sorted(
        [(c["first_turn"], c["last_turn"]) for c in state["l1"]]
        + [(c["first_turn"], c["last_turn"]) for c in state["l2"]]
        + ([(state["l3"]["first_turn"], state["l3"]["last_turn"])]
           if state["l3"] else [])
    )
    # Contiguity is the property, not the chunk count: a position that drifts
    # by a turn or two between rollups still produces chunks, just with holes
    # between them, and a count-only assertion cannot see that.
    end = 0
    for first, last in covered:
        if first > end + 1:
            print(f"FAIL turns {end + 1}-{first - 1} are covered by nothing")
            sys.exit(1)
        end = max(end, last)
    assert_eq(end, 40, "coverage reaches the watermark with no gaps")


def test_the_capped_rollup_summarizes_the_right_turns():
    print("\n[test] under a cap the chunk's TEXT is the text it claims")
    # The subtle half. Chunk boundaries are absolute turn numbers; the client's
    # array is a suffix. Slicing turns 21-24 out of a window holding turns
    # 33-40 reads positions 21-24 of that window, i.e. nothing (or, with a
    # larger cap, some other conversation entirely) — and the stored chunk
    # would still be LABELLED 21-24 with nothing downstream able to tell.
    _wipe()
    cid = "right-text"
    bodies, orig = _install_body_recorder("CHUNK")
    try:
        asyncio.run(summarizer.maybe_rollup(cid, _msgs(20), "http://x", "m"))
        bodies.clear()
        for total in (22, 24):
            asyncio.run(
                summarizer.maybe_rollup(cid, _window(total, 8), "http://x", "m")
            )
    finally:
        _restore_httpx(orig)

    l1_bodies = [b for prompt, b in bodies if prompt == summarizer._PROMPT_L1]
    assert_eq(len(l1_bodies), 1, "exactly one L1 call for the four new turns")
    body = l1_bodies[0]
    for turn in (21, 22, 23, 24):
        assert_true(f"msg{turn}-content" in body,
                    f"turn {turn} reached the model")
    for turn in (17, 18, 19, 20):
        assert_true(f"msg{turn}-content" not in body,
                    f"turn {turn} — already summarized — did not")


def test_switching_the_cap_on_re_summarizes_nothing():
    print("\n[test] the transition to a capped window costs no rollup")
    # The migration case: a live conversation whose watermark is already far
    # ahead when the valve is switched on. The anchor matches the tail of the
    # narrower window, so the position holds and not one covered turn is sent
    # to the model a second time.
    _wipe()
    cid = "transition"
    orig = _install_mock("CHUNK")
    try:
        before = asyncio.run(summarizer.maybe_rollup(cid, _msgs(20), "http://x", "m"))
    finally:
        _restore_httpx(orig)
    assert_eq(before["last_summarized_turn"], 20, "the full-history run caught up")
    n_chunks = len(before["l1"]) + len(before["l2"])

    calls, orig = _install_call_recorder("SHOULD-NOT-HAPPEN")
    try:
        state = asyncio.run(
            summarizer.maybe_rollup(cid, _window(20, 8), "http://x", "m")
        )
    finally:
        _restore_httpx(orig)
    assert_eq(calls, [], "no LLM call on the turn the cap arrives")
    assert_eq(state["turns_seen"], 20, "the position did not move")
    assert_eq(state["last_summarized_turn"], 20, "nor the watermark")
    assert_eq(len(state["l1"]) + len(state["l2"]), n_chunks,
              "and no chunk was added")


def test_the_cap_engaging_mid_conversation_loses_no_turn():
    print("\n[test] the turn the valve first bites on is not swallowed")
    # R23. test_switching_the_cap_on_re_summarizes_nothing models the
    # transition as a full history followed immediately by a SHORT window
    # (n <= prev), which lands in the anchor branch. A sliding cap does not do
    # that. It engages gradually, and on the first request where it bites the
    # window is still LONGER than the recorded position while already being
    # SHORTER than the conversation:
    #
    #   cap 8, exchange 5 — the client stores turns 1-9, the valve trims to the
    #   last 8, the compactor appends turn 10 -> n = 9, prev = 8, truth = 10.
    #
    # The old code read "n > prev" as "the array length IS the position" and
    # took 9. One turn of position, swallowed permanently: client turn 9 was
    # summarized by no tier, and every chunk from there on was labelled one
    # turn off the text inside it.
    _wipe()
    cid = "valve"
    cap = 8
    exchanges = 12                      # 24 turns, six L1 chunks' worth
    bodies, orig = _install_body_recorder("CHUNK")
    try:
        with capture() as cap_log:
            for e in range(1, exchanges + 1):
                state = asyncio.run(
                    summarizer.maybe_rollup(cid, _live_window(e, cap),
                                            "http://x", "m")
                )
    finally:
        _restore_httpx(orig)

    truth = 2 * exchanges
    assert_eq(state["turns_seen"], truth,
              "the position tracked the conversation across the valve")

    # The two harms are separate questions and the second is the quiet one.
    # First: is every covered turn's TEXT in some chunk's request body?
    l1_bodies = [b for prompt, b in bodies if prompt == summarizer._PROMPT_L1]
    summarized = set()
    for b in l1_bodies:
        summarized.update(_turns_in(b, truth))
    watermark = state["last_summarized_turn"]
    missing = [t for t in range(1, watermark + 1) if t not in summarized]
    assert_eq(missing, [], "no client turn under the watermark went unsummarized")

    # Second: does each chunk's LABEL name the turns its body actually held?
    # A chunk labelled 9-12 whose text is turns 10-13 passes every count-based
    # assertion in this file and is what _do_l1_rollup's own comment calls
    # worse than summarizing nothing, because nothing downstream can tell.
    labels = _labels_logged(cap_log.records)
    assert_eq(len(labels), len(l1_bodies), "one claimed span per L1 request")
    assert_true(len(labels) > 0, "and rollups actually happened")
    for (first, last), body in zip(labels, l1_bodies):
        held = _turns_in(body, truth)
        assert_eq((held[0], held[-1]), (first, last),
                  f"chunk labelled {first}-{last} holds exactly those turns")


def test_a_watermark_below_its_own_chunks_is_repaired_not_discarded():
    print("\n[test] a pulled-down watermark does not silence the rollup")
    # R12, and it fires on the DOCUMENTED install path. A state file written
    # by the parent commit under max_turns has the watermark pulled down to
    # the cap by the old _reconcile_watermark, while the L1 list still holds
    # chunks labelled up to the real position. Seeding the position from
    # `max(turns_seen, last_summarized_turn)` restarted it at the cap, so the
    # next chunk was labelled cap+1..cap+20 — a label an OLD chunk already
    # owns. The duplicate-chunk guard then found it, logged a WARNING,
    # advanced the watermark and returned True, discarding a real span. For
    # every chunk until the position climbed past the highest old label:
    # reproduced at `new L1 chunks=0, LLM calls=0` for ~280 exchanges.
    #
    # Both documented cap values (100 and 60) are multiples of L1_CHUNK_SIZE,
    # so the labels collide exactly.
    _wipe()
    cid = "pulled-down"
    cap = 8
    chunk = summarizer.L1_CHUNK_SIZE
    # The conversation reached turn 40. Older chunks are already folded into a
    # chapter; the two most recent are still loose in L1 — the shape a live
    # conversation is actually in, and the shape that makes the collision
    # reachable, since the guard only looks at L1.
    summarizer.save_state(cid, {
        "l1": [{"text": "scene nine", "first_turn": 33, "last_turn": 36},
               {"text": "scene ten", "first_turn": 37, "last_turn": 40}],
        "l2": [{"text": "chapter one", "first_turn": 1, "last_turn": 32}],
        "l3": None,
        "last_summarized_turn": cap,       # PULLED DOWN from 40 by the old code
        "turns_seen": cap,
        "tail_fp": [],
    })

    bodies, orig = _install_body_recorder("NEW-CHUNK")
    try:
        with capture() as cap_log:
            # Client turns 41-48: two chunks' worth past the highest old label.
            for e in range(21, 25):
                state = asyncio.run(
                    summarizer.maybe_rollup(cid, _live_window(e, cap),
                                            "http://x", "m")
                )
    finally:
        _restore_httpx(orig)

    repaired = find(cap_log.records, "while stored chunks already cover")
    assert_true(repaired is not None,
                "the operator is told the watermark was below its own chunks")
    assert_eq(repaired.levelno, logging.WARNING, "at WARNING — it self-healed")
    discarded = find(cap_log.records, "already exists")
    assert_true(discarded is None, "and no span was discarded as a duplicate")

    l1_bodies = [b for prompt, b in bodies if prompt == summarizer._PROMPT_L1]
    assert_true(len(l1_bodies) > 0,
                "the rollup spent a real LLM call instead of skipping silently")
    labels = _labels_logged(cap_log.records)
    assert_true(len(labels) > 0, "and a new chunk past the old labels was claimed")
    assert_true(all(f > 40 for f, _ in labels),
                "every new chunk starts past the highest old label")
    assert_eq(state["turns_seen"], 48,
              "the position is the conversation's, not the cap's")

    # The label has to name the turns the body held. Advancing the watermark
    # without moving the position would have produced a chunk labelled 41-44
    # over some other four turns entirely.
    assert_eq(len(labels), len(l1_bodies), "one claimed span per L1 request")
    for (first, last), body in zip(labels, l1_bodies):
        held = _turns_in(body, 48)
        assert_eq((held[0], held[-1]), (first, last),
                  f"chunk labelled {first}-{last} holds exactly those turns")


def test_the_same_capped_window_twice_adds_nothing():
    print("\n[test] re-running the same window produces no duplicate chunk")
    # The admin-compact shape: /admin/conversations/<id>/compact loops
    # maybe_rollup over ONE reconstructed transcript until the watermark stops
    # moving. Under the old reset that loop appended a duplicate chunk set on a
    # second invocation, which cascaded into duplicate L2 chapters and a
    # duplicate-fed L3.
    _wipe()
    cid = "idempotent"
    orig = _install_mock("CHUNK")
    try:
        asyncio.run(summarizer.maybe_rollup(cid, _msgs(20), "http://x", "m"))
        for total in (22, 24):
            first = asyncio.run(
                summarizer.maybe_rollup(cid, _window(total, 8), "http://x", "m")
            )
    finally:
        _restore_httpx(orig)
    spans = [(c["first_turn"], c["last_turn"]) for c in first["l1"]]

    calls, orig = _install_call_recorder("DUPLICATE")
    try:
        again = asyncio.run(
            summarizer.maybe_rollup(cid, _window(24, 8), "http://x", "m")
        )
    finally:
        _restore_httpx(orig)
    assert_eq(calls, [], "the repeat costs no LLM call")
    assert_eq([(c["first_turn"], c["last_turn"]) for c in again["l1"]], spans,
              "and the chunk list is unchanged")
    assert_eq(again["turns_seen"], 24, "the position did not double-count")


def test_a_regenerated_reply_is_not_a_new_turn():
    print("\n[test] regenerating the last reply does not advance the position")
    # A replaced turn is not a new turn. Counted as one, the position runs 2
    # ahead of the conversation, every later chunk boundary shifts by 2, and
    # the two turns that fall through the shift are summarized by nothing.
    _wipe()
    cid = "regen"
    orig = _install_mock("CHUNK")
    try:
        asyncio.run(summarizer.maybe_rollup(cid, _msgs(20), "http://x", "m"))
        window = _window(20, 8)
        asyncio.run(summarizer.maybe_rollup(cid, window, "http://x", "m"))
        regenerated = [dict(m) for m in window]
        regenerated[-1] = {"role": "assistant", "content": "a different reply"}
        state = asyncio.run(
            summarizer.maybe_rollup(cid, regenerated, "http://x", "m")
        )
    finally:
        _restore_httpx(orig)
    assert_eq(state["turns_seen"], 20, "the position stands at 20, not 22")


def test_a_window_that_cannot_be_aligned_advances_by_one_exchange():
    print("\n[test] an unalignable window advances at the live-path rate")
    # The degradation path: if the anchor does not round-trip through the
    # client at all, the position falls back to counting calls. That is right
    # for the live tail (main.py calls maybe_rollup once per exchange) and it
    # is the branch that keeps this from being a new way to freeze — a
    # fallback of "hold" would be the 2026-09-01 defect wearing a new hat.
    _wipe()
    cid = "unalignable"
    summarizer.save_state(cid, {
        "l1": [], "l2": [], "l3": None, "last_summarized_turn": 40,
        "turns_seen": 40, "tail_fp": ["deadbeefdeadbeef"] * 4,
    })
    logsetup._reset_log_once_for_tests()
    calls, orig = _install_call_recorder()
    try:
        with capture() as cap:
            state = asyncio.run(
                summarizer.maybe_rollup(cid, _window(60, 8), "http://x", "m")
            )
    finally:
        _restore_httpx(orig)
    assert_eq(state["turns_seen"], 42, "one exchange past the recorded position")
    assert_eq(calls, [], "and no rollup yet — 2 new turns is under the threshold")
    warned = find(cap.records, "cannot be measured against it")
    assert_true(warned is not None, "the operator is told the anchor missed")
    assert_eq(warned.levelno, logging.WARNING, "at WARNING")
    logsetup._reset_log_once_for_tests()


def test_a_bounded_window_is_reported_once_per_process():
    print("\n[test] the bounded window says so, once, at INFO")
    # Once per process because it is the tail of EVERY turn once the valve is
    # on, and INFO because it is the healthy shape — the operator needs to see
    # the compactor noticed, not to be alarmed every minute.
    _wipe()
    logsetup._reset_log_once_for_tests()
    orig = _install_mock("CHUNK")
    try:
        for cid in ("bounded-a", "bounded-b"):
            asyncio.run(summarizer.maybe_rollup(cid, _msgs(20), "http://x", "m"))
        with capture() as cap:
            for cid in ("bounded-a", "bounded-b"):
                asyncio.run(
                    summarizer.maybe_rollup(cid, _window(22, 8), "http://x", "m")
                )
    finally:
        _restore_httpx(orig)
    hits = [r for r in cap.records if "bounded window" in r.getMessage()]
    assert_eq(len(hits), 1, "exactly one line across two capped conversations")
    assert_eq(hits[0].levelno, logging.INFO, "at INFO, not WARNING")
    assert_true("offset of 14" in hits[0].getMessage(),
                "naming the offset the chunk text is read at")
    logsetup._reset_log_once_for_tests()


def test_align_new_turns_unit():
    print("\n[test] _align_new_turns: exact, prefix, latest, and no match")
    a = summarizer._align_new_turns
    # The anchor ends at the previous position, so a whole-anchor match at the
    # end of the window means nothing new.
    assert_eq(a(["w", "x", "y", "z"], ["w", "x", "y", "z"]), 0, "unchanged window")
    assert_eq(a(["w", "x", "y", "z"], ["w", "x", "y", "z", "p", "q"]), 2,
              "one exchange appended")
    # Regeneration: the newest anchored turn was REPLACED, the three before it
    # were not. Without the prefix walk this reads as 1 new turn, and a
    # position 1 ahead of the conversation puts a 1-turn hole at the next
    # chunk boundary.
    assert_eq(a(["w", "x", "y", "z"], ["w", "x", "y", "REGEN"]), 0,
              "a replaced last turn is not a new turn")
    assert_eq(a(["w", "x", "y", "z"], ["w", "x", "y", "REGEN", "p", "q"]), 2,
              "and the exchange after it still counts as one")
    # Two identical stretches. Downward scan takes the LATEST, i.e. the
    # smallest advance the evidence allows: an ambiguous short turn ("ok")
    # should cost a duplicated summary, never a skipped one.
    assert_eq(a(["w", "x"], ["w", "x", "w", "x", "p"]), 1,
              "an ambiguous match resolves to the latest occurrence")
    assert_eq(a(["w", "x"], ["p", "q", "r"]), None, "no match at all")


def test_turn_fingerprints_survive_whitespace_reflow():
    print("\n[test] the anchor survives a re-flowed reply")
    # The anchor is compared ACROSS REQUESTS: what the compactor appended
    # after streaming a reply, against what OpenWebUI reads back out of its own
    # database and re-sends on the next turn. A trailing newline that does not
    # survive that round trip would make every window unalignable, and the
    # fallback (advance by one exchange) would be running permanently instead
    # of as a degradation.
    a = [{"role": "assistant", "content": "one two\nthree"}]
    b = [{"role": "assistant", "content": "  one   two \n three\n"}]
    assert_eq(summarizer._turn_fingerprints(a),
              summarizer._turn_fingerprints(b), "same turn, same fingerprint")
    c = [{"role": "user", "content": "one two\nthree"}]
    assert_true(summarizer._turn_fingerprints(a) != summarizer._turn_fingerprints(c),
                "but the role is part of the identity")
    d = [{"role": "assistant", "content": "one two three!"}]
    assert_true(summarizer._turn_fingerprints(a) != summarizer._turn_fingerprints(d),
                "and so is the text")


def test_a_duplicate_span_is_not_stored_twice():
    print("\n[test] a re-presented span is skipped — after proving it IS the same")
    # Belt and braces for the hazard the old watermark reset created: running
    # the admin drain twice appended a second identical chunk set, which
    # cascaded into duplicate L2 chapters and a duplicate-fed L3.
    #
    # REWRITTEN for v3.1.7 (R12). The old version of this test built the
    # existing chunk BY HAND — text "already summarized", span 1-4 — set
    # last_summarized_turn to 0, called it "a position that went backwards",
    # and asserted the span was skipped "without spending an LLM call to
    # prove it". That is a stronger claim than the fixture supports: nothing
    # in it established that turns 1-4 of the messages in hand were the turns
    # the existing chunk summarized. Under a pulled-down watermark they are
    # NOT — the labels collide while the text underneath them is 500 turns
    # apart — and the skip the test was pinning is exactly how R12 threw away
    # every new chunk in silence. A test that cannot tell those two cases
    # apart pins the bug as firmly as the behaviour.
    #
    # So the existing chunk is now MADE by a real rollup over known turns, and
    # its request body is kept. The skip is only correct because the body
    # proves the same four turns are already covered.
    _wipe()
    cid = "dupe-span"
    state = summarizer._empty_state(cid)
    import httpx

    bodies, orig = _install_body_recorder("FIRST-PASS")
    try:
        async def _first():
            async with httpx.AsyncClient() as client:
                return await summarizer._do_l1_rollup(
                    cid, client, "http://x", "m", state, _msgs(4), 0
                )
        assert_eq(asyncio.run(_first()), True, "the first rollup stored a chunk")
    finally:
        _restore_httpx(orig)
    assert_eq(len(state["l1"]), 1, "one chunk so far")
    assert_eq((state["l1"][0]["first_turn"], state["l1"][0]["last_turn"]), (1, 4),
              "labelled 1-4")
    held = _turns_in(bodies[0][1], 4)
    assert_eq((held[0], held[-1]), (1, 4),
              "and the turns it actually summarized were 1-4")

    # Now the re-presentation: the same four turns, with the watermark rewound
    # underneath them. THIS is the case the skip is right for.
    state["last_summarized_turn"] = 0
    calls, orig = _install_call_recorder("SHOULD-NOT-HAPPEN")
    try:
        with capture() as cap:
            async def _again():
                async with httpx.AsyncClient() as client:
                    return await summarizer._do_l1_rollup(
                        cid, client, "http://x", "m", state, _msgs(4), 0
                    )
            advanced = asyncio.run(_again())
    finally:
        _restore_httpx(orig)
    assert_eq(advanced, True, "the watermark still advances past the span")
    assert_eq(state["last_summarized_turn"], 4, "to the end of it")
    assert_eq(len(state["l1"]), 1, "and no second chunk was appended")
    assert_eq(calls, [], "without spending an LLM call to prove it")
    # Raised from WARNING in v3.1.7: with the position seeded from the chunk
    # labels and the watermark repaired against them before any rollup runs,
    # nothing on the live path can reach this line. If it is in the log, the
    # position arithmetic is wrong again and the skip is hiding how much.
    noted = find(cap.records, "already exists")
    assert_true(noted is not None, "the skip is reported")
    assert_eq(noted.levelno, logging.ERROR,
              "at ERROR — this is now an unreachable state, not routine")


def test_a_rewound_watermark_is_repaired_before_it_can_discard_a_span():
    print("\n[test] the position is seeded from the chunk labels, not just the pointers")
    # The other half of R12, and the half that makes the guard above
    # unreachable rather than merely loud. _recorded_position seeds from
    # max(turns_seen, last_summarized_turn, highest chunk label); the third
    # term is the one that survives a watermark the old code pulled down.
    _wipe()
    cid = "seeded"
    state = {
        "l1": [{"text": "a scene", "first_turn": 37, "last_turn": 40}],
        "l2": [], "l3": None,
        "last_summarized_turn": 8,       # pulled down under a cap of 8
        "turns_seen": 8,
        "tail_fp": [],
    }
    assert_eq(summarizer._recorded_position(state), 40,
              "the chunk labels outvote the pointers that were pulled down")
    assert_eq(summarizer._repair_watermark_below_chunks(cid, state), True,
              "and the watermark is raised to what the chunks already prove")
    assert_eq(state["last_summarized_turn"], 40, "to the highest label")
    # Idempotent: a healthy state is not touched, so this cannot become a
    # second way to move a watermark that is already correct.
    assert_eq(summarizer._repair_watermark_below_chunks(cid, state), False,
              "a second pass changes nothing")
    healthy = {"l1": [{"text": "x", "first_turn": 1, "last_turn": 4}],
               "l2": [], "l3": None, "last_summarized_turn": 100,
               "turns_seen": 100, "tail_fp": []}
    assert_eq(summarizer._repair_watermark_below_chunks(cid, healthy), False,
              "a watermark ABOVE its chunks is left alone — that is S-5's case")
    assert_eq(healthy["last_summarized_turn"], 100, "and stands where it was")


def test_a_backlog_past_the_window_skips_instead_of_stalling():
    print("\n[test] material that scrolled out is skipped, loudly, not stalled")
    # Reachable only after a rollup outage longer than the cap. The text is not
    # in the request and this module never held a copy, so the choice is
    # "advance past the dead span" or "never advance again" — and never
    # advancing also abandons the turns that ARE still arriving.
    _wipe()
    cid = "backlog"
    summarizer.save_state(cid, {
        "l1": [], "l2": [], "l3": None, "last_summarized_turn": 0,
        "turns_seen": 60, "tail_fp": summarizer._turn_fingerprints(
            _window(60, 8))[-4:],
    })
    orig = _install_mock("CHUNK")
    try:
        with capture() as cap:
            state = asyncio.run(
                summarizer.maybe_rollup(cid, _window(60, 8), "http://x", "m")
            )
    finally:
        _restore_httpx(orig)
    assert_eq(state["last_summarized_turn"], 60,
              "the watermark drained to the position instead of sticking at 0")
    assert_true(len(state["l1"]) + len(state["l2"]) > 0
                or state["l3"] is not None,
                "the turns still in the window were summarized")
    lost = find(cap.records, "behind the client's window and were never")
    assert_true(lost is not None, "the unrecoverable span is named")
    assert_eq(lost.levelno, logging.ERROR, "at ERROR — this is lost memory")
    first_kept = min(
        [c["first_turn"] for c in state["l1"]]
        + [c["first_turn"] for c in state["l2"]]
        + ([state["l3"]["first_turn"]] if state["l3"] else [])
    )
    assert_eq(first_kept, 53, "coverage starts at the first observable turn")


def test_a_chunk_straddling_the_window_edge_claims_only_what_it_saw():
    print("\n[test] a half-lost chunk records the span it actually summarized")
    # The other half of the backlog case, and the one that is easy to get
    # wrong quietly: the chunk's HEAD scrolled out but its TAIL is still in the
    # window, so there IS text to summarize. Recording first_turn as the
    # boundary the watermark implies would store a chunk asserting coverage of
    # turns whose text this rollup never saw — and the coverage checks in
    # test_soak_conversation.py read those labels as truth.
    _wipe()
    cid = "straddle"
    # position 60, window of 8 -> the window holds turns 53-60, so the chunk
    # after a watermark of 50 is turns 51-54: two turns gone, two still here.
    summarizer.save_state(cid, {
        "l1": [], "l2": [], "l3": None, "last_summarized_turn": 50,
        "turns_seen": 60,
        "tail_fp": summarizer._turn_fingerprints(_window(60, 8))[-4:],
    })
    bodies, orig = _install_body_recorder("CHUNK")
    try:
        with capture() as cap:
            state = asyncio.run(
                summarizer.maybe_rollup(cid, _window(60, 8), "http://x", "m")
            )
    finally:
        _restore_httpx(orig)

    first = min(c["first_turn"] for c in state["l1"] + state["l2"])
    assert_eq(first, 53, "the chunk claims only from the first turn it saw")
    l1_bodies = [b for prompt, b in bodies if prompt == summarizer._PROMPT_L1]
    assert_true("msg53-content" in l1_bodies[0],
                "and its text really does start at turn 53")
    assert_true("msg51-content" not in l1_bodies[0],
                "turn 51 was not in the window and is not in the body")
    warned = find(cap.records, "of this chunk are behind the client's window")
    assert_true(warned is not None, "the partial loss is reported")
    assert_eq(warned.levelno, logging.WARNING, "at WARNING")


def test_a_straddle_during_an_outage_does_not_claim_a_span_it_never_recorded():
    print("\n[test] a half-lost chunk during a vLLM outage records NOTHING, and says so")
    # v3.1.7, R29. Same straddle fixture as the test above, but the summarizer
    # call returns empty (an outage) instead of "CHUNK". The old code logged
    # "summarized turns 53-54 and recorded that as the chunk's span" BEFORE
    # calling _summarize_pieces, so when that call came back empty the
    # function recorded nothing while the log already said it had —
    # reproduced during triage as four such lines against l1=0, emitted
    # exactly when an operator is reading logs during an outage.
    _wipe()
    cid = "straddle-outage"
    summarizer.save_state(cid, {
        "l1": [], "l2": [], "l3": None, "last_summarized_turn": 50,
        "turns_seen": 60,
        "tail_fp": summarizer._turn_fingerprints(_window(60, 8))[-4:],
    })
    orig = _install_mock("")  # every vLLM call returns empty content
    try:
        with capture() as cap:
            state = asyncio.run(
                summarizer.maybe_rollup(cid, _window(60, 8), "http://x", "m")
            )
    finally:
        _restore_httpx(orig)

    assert_eq(state["last_summarized_turn"], 50, "the watermark did not move")
    assert_eq(state["l1"], [], "no L1 chunk was appended")
    assert_eq(state["l2"], [], "no L2 chapter either")
    lied = find(cap.records, "recorded that as the chunk's span")
    assert_true(lied is None,
                "the log must not claim a span was recorded when l1/l2 are empty")


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
        test_shortened_history_does_not_pull_the_watermark_back()
        test_a_stranded_watermark_still_unlatches()
        test_a_permanently_capped_window_still_rolls_up()
        test_the_capped_rollup_summarizes_the_right_turns()
        test_switching_the_cap_on_re_summarizes_nothing()
        test_the_cap_engaging_mid_conversation_loses_no_turn()
        test_a_watermark_below_its_own_chunks_is_repaired_not_discarded()
        test_the_same_capped_window_twice_adds_nothing()
        test_a_regenerated_reply_is_not_a_new_turn()
        test_a_window_that_cannot_be_aligned_advances_by_one_exchange()
        test_a_bounded_window_is_reported_once_per_process()
        test_align_new_turns_unit()
        test_turn_fingerprints_survive_whitespace_reflow()
        test_a_duplicate_span_is_not_stored_twice()
        test_a_rewound_watermark_is_repaired_before_it_can_discard_a_span()
        test_a_backlog_past_the_window_skips_instead_of_stalling()
        test_a_chunk_straddling_the_window_edge_claims_only_what_it_saw()
        test_a_straddle_during_an_outage_does_not_claim_a_span_it_never_recorded()
        test_rollup_logs_a_success_line()
        test_failed_l3_does_not_discard_successful_l1_and_l2()
        test_failed_l3_does_not_repeat_the_same_work_forever()
        test_l3_does_not_refire_once_the_chapters_are_covered()
        test_state_summary_compact()
        print("\nAll summarizer smoke tests passed.")
    finally:
        if os.path.exists(_TMP):
            shutil.rmtree(_TMP, ignore_errors=True)

# ---------------------------------------------------------------------------
# Mutation record for the v3.1.4 capped-window work. Each behaviour was broken
# in summarizer.py one at a time and this file re-run; the assertion named is
# the one that went red. A mutation that survives means the test is decoration.
#
#   `if n > prev:` -> `if True:`                      -> "the position is
#                                                        seeded from it"
#   the position/anchor write -> `changed = False`    -> "the position IS
#                                                        persisted"
#   prefix walk -> `for m in (len(anchor),):`         -> "the position stands
#                                                        at 20, not 22"
#   `range(n, m - 1, -1)` -> `range(m, n + 1)`        -> "an ambiguous match
#                                                        resolves to the
#                                                        latest occurrence"
#   `window_offset` -> `0` at the _do_l1_rollup call  -> "a new chunk on top
#                                                        of the kept one"
#   the scrolled-out skip -> `return False`           -> "the watermark
#                                                        drained to the
#                                                        position"
#   `covered_first = window_offset + 1` -> first_turn -> "the chunk claims
#                                                        only from the first
#                                                        turn it saw"
#   the duplicate-span guard -> `if False:`           -> "no second chunk was
#                                                        appended"
#   `max(turns_seen, watermark)` -> turns_seen only   -> "the position is
#                                                        seeded from it"
#   `_ASSUMED_NEW_TURNS = 2` -> `0`                   -> "one exchange past
#                                                        the recorded
#                                                        position"
#   fingerprint normalization removed                 -> "same turn, same
#                                                        fingerprint"
#   load_state drops turns_seen                       -> "four further turns
#                                                        past the strand"
#   load_state drops tail_fp                          -> "four further turns
#                                                        past the strand"
#
# v3.1.7, for R23 and R12. Same method, same file, re-run per mutation.
#
#   `max(n, prev + new)` -> `n if n > prev else          -> "the position
#      prev + new`  (the old two-regime rule)               tracked the
#                                                           conversation
#                                                           across the valve"
#   window_offset shifted by 1 (position stays right,    -> "no client turn
#      only the TEXT moves — proves the coverage             under the
#      assertion is not riding on the position one)         watermark went
#                                                           unsummarized"
#   _recorded_position drops the chunk-label term        -> "the chunk labels
#                                                           outvote the
#                                                           pointers that were
#                                                           pulled down"
#   _repair_watermark_below_chunks becomes a no-op       -> "the operator is
#                                                           told the watermark
#                                                           was below its own
#                                                           chunks"
#   the anchorless branch holds instead of consulting    -> "the position is
#      _highest_chunk_turn                                  the conversation's,
#                                                           not the cap's"
#   the duplicate guard logs WARNING again               -> "at ERROR — this is
#                                                           now an unreachable
#                                                           state, not routine"
# ---------------------------------------------------------------------------
