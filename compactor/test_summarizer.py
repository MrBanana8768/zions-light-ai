"""
CPU-only smoke tests for compactor/summarizer.py (V2.0 Phase 4).

Mocks the vLLM HTTP call so no GPU / network is needed. Verifies state
storage, rollup-trigger detection, the L1/L2/L3 cascade, threshold logic,
injection block formatting, and graceful degradation.

Run:
    python test_summarizer.py
"""

import asyncio
import json
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
# Mock LLM client — returns canned summaries
# ---------------------------------------------------------------------------

def _mock_client_returning(content_per_call):
    """content_per_call: either a single string (all calls return it) or
    a list (consumed in order). Returns a fake AsyncClient context manager.
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
            return _Resp(content)

    return _Client()


def _install_mock(content_per_call):
    """Patch httpx.AsyncClient inside summarizer to return our mock."""
    import httpx
    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: _mock_client_returning(content_per_call)
    return orig


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


def test_load_state_drops_corrupt_chunks():
    print("\n[test] load_state filters malformed l1/l2 entries")
    _wipe()
    cid = "bad"
    raw = {
        "conv_id": cid,
        "l1": [
            {"text": "ok", "first_turn": 1, "last_turn": 4},
            {"text": "", "first_turn": 5, "last_turn": 8},      # empty
            "not a dict",                                        # wrong type
            {"text": "no_turns"},                                # missing fields
            {"text": "also ok", "first_turn": 9, "last_turn": 12},
        ],
        "l2": [],
        "l3": None,
        "last_summarized_turn": 12,
    }
    summarizer.summary_path(cid).write_text(json.dumps(raw))
    loaded = summarizer.load_state(cid)
    assert_eq(len(loaded["l1"]), 2, "filtered to 2 valid chunks")


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
    assert_eq(len(state["l2"]), 2, "two L2 chapters produced")
    assert_true(state["l3"] is not None, "L3 produced")
    assert_eq(state["l3"]["text"], "L3X", "L3 text from final LLM call")


def test_maybe_rollup_skips_when_not_needed():
    print("\n[test] maybe_rollup is a no-op when nothing crosses threshold")
    _wipe()
    cid = "noop"
    msgs = _msgs(2)  # below L1 threshold of 4
    # No LLM calls expected — install a mock that raises if called.
    import httpx
    orig = httpx.AsyncClient

    class _ShouldNotBeCalled:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            raise AssertionError("LLM should not be called when no rollup needed")

    httpx.AsyncClient = lambda *a, **kw: _ShouldNotBeCalled()
    try:
        state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
    finally:
        httpx.AsyncClient = orig
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
        test_load_state_drops_corrupt_chunks()
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
        test_state_summary_compact()
        print("\nAll summarizer smoke tests passed.")
    finally:
        if os.path.exists(_TMP):
            shutil.rmtree(_TMP, ignore_errors=True)
