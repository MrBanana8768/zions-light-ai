"""
CPU-only smoke tests for compactor/backfill.py (V2.0 Phase 2).

Covers the backfill state machine, message-pair extraction, decision
logic for needs_backfill, and an end-to-end run with a mock vLLM client.

Run inside the compactor image or any container with the requirements
installed:
    python test_backfill.py
"""

import asyncio
import contextlib
import errno
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-backfill-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import backfill  # noqa: E402
import facts  # noqa: E402
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


def _wipe_storage():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()
    backfill._in_progress_local.clear()


# ---------------------------------------------------------------------------
# State file paths and round-trips
# ---------------------------------------------------------------------------

def test_state_path_in_facts_dir():
    print("\n[test] backfill state file lives in facts/ subdirectory")
    p = backfill._backfill_state_path("abc")
    assert_true(str(p).endswith("facts/abc.backfill.json") or
                str(p).endswith("facts\\abc.backfill.json"),
                "path ends in facts/<cid>.backfill.json")


def test_read_state_missing_returns_none():
    print("\n[test] read_state for unknown conv -> None")
    _wipe_storage()
    assert_eq(backfill.read_state("never-seen"), None, "missing state -> None")


def test_write_then_read_state_roundtrip():
    print("\n[test] _write_state -> read_state preserves fields")
    _wipe_storage()
    cid = "rt"
    backfill._write_state(cid, {
        "state": "in_progress",
        "started_at": "2026-05-28T00:00:00+00:00",
        "exchanges_done": 3,
        "exchanges_total": 10,
        "error": None,
    })
    s = backfill.read_state(cid)
    assert_eq(s["state"], "in_progress", "state preserved")
    assert_eq(s["exchanges_done"], 3, "exchanges_done preserved")
    assert_eq(s["conv_id"], cid, "conv_id auto-set")
    assert_true("updated_at" in s, "updated_at auto-set")


# ---------------------------------------------------------------------------
# Stale detection
# ---------------------------------------------------------------------------

def test_is_stale_fresh_in_progress():
    print("\n[test] is_stale=False for a freshly-updated in_progress state")
    fresh = {
        "state": "in_progress",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    assert_eq(backfill.is_stale(fresh), False, "fresh -> not stale")


def test_is_stale_old_in_progress():
    print("\n[test] is_stale=True for an in_progress state older than threshold")
    old_ts = datetime.now(timezone.utc) - timedelta(seconds=backfill._STALE_SECONDS + 60)
    old = {"state": "in_progress", "updated_at": old_ts.isoformat()}
    assert_eq(backfill.is_stale(old), True, "old -> stale")


def test_is_stale_complete_state_never_stale():
    print("\n[test] is_stale=False for complete state regardless of age")
    very_old = {
        "state": "complete",
        "updated_at": "2020-01-01T00:00:00+00:00",
    }
    assert_eq(backfill.is_stale(very_old), False, "complete is never stale")


# ---------------------------------------------------------------------------
# Message pair extraction
# ---------------------------------------------------------------------------

def test_extract_pairs_basic():
    print("\n[test] extract_user_assistant_pairs basic case")
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "how are you?"},
        {"role": "assistant", "content": "well"},
    ]
    pairs = backfill.extract_user_assistant_pairs(msgs)
    assert_eq(len(pairs), 2, "two pairs")
    assert_eq(pairs[0], ("hello", "hi"), "first pair")
    assert_eq(pairs[1], ("how are you?", "well"), "second pair")


def test_extract_pairs_skips_trailing_unmatched_user():
    print("\n[test] extract_user_assistant_pairs drops trailing user without response")
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2 — no reply yet"},
    ]
    pairs = backfill.extract_user_assistant_pairs(msgs)
    assert_eq(len(pairs), 1, "trailing unmatched user dropped")


def test_extract_pairs_handles_system_mid_conversation():
    print("\n[test] extract_user_assistant_pairs skips mid-stream system messages")
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "[summary injected]"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    pairs = backfill.extract_user_assistant_pairs(msgs)
    assert_eq(len(pairs), 2, "system in middle didn't break pairing")


def test_extract_pairs_multimodal_user_message():
    print("\n[test] extract_user_assistant_pairs handles multimodal user content")
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "what is this image"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]},
        {"role": "assistant", "content": "a cat"},
    ]
    pairs = backfill.extract_user_assistant_pairs(msgs)
    assert_eq(len(pairs), 1, "one pair from multimodal user")
    assert_true("what is this image" in pairs[0][0], "text portion extracted")


# ---------------------------------------------------------------------------
# needs_backfill decision logic
# ---------------------------------------------------------------------------

def test_needs_backfill_short_conv_returns_false():
    print("\n[test] needs_backfill=False for too-short conversation")
    _wipe_storage()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert_eq(backfill.needs_backfill("short", msgs), False, "<4 msgs -> no backfill")


def test_needs_backfill_with_existing_facts_returns_false():
    print("\n[test] needs_backfill=False when facts file already exists")
    _wipe_storage()
    cid = "has-facts"
    facts.save_facts(cid, [{"text": "x", "added_turn": 1, "last_used": 1}])
    msgs = [{"role": "user", "content": "u" * 100}] * 10
    assert_eq(backfill.needs_backfill(cid, msgs), False, "facts present -> skip backfill")


def test_needs_backfill_no_state_and_long_history_returns_true():
    print("\n[test] needs_backfill=True for V1 conv (no facts, long history)")
    _wipe_storage()
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert_eq(backfill.needs_backfill("v1-conv", msgs), True, "V1 conv -> needs backfill")


def test_needs_backfill_complete_state_returns_false():
    print("\n[test] needs_backfill=False when previous backfill marked complete")
    _wipe_storage()
    cid = "done"
    backfill._write_state(cid, {"state": "complete", "started_at": "...", "exchanges_done": 5, "exchanges_total": 5})
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 3
    assert_eq(backfill.needs_backfill(cid, msgs), False, "complete -> no retry")


def test_needs_backfill_stale_in_progress_returns_true():
    print("\n[test] needs_backfill=True when previous backfill is stale (crashed)")
    _wipe_storage()
    cid = "crashed"
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    state_path = backfill._backfill_state_path(cid)
    state_path.write_text(json.dumps({
        "conv_id": cid,
        "state": "in_progress",
        "started_at": old_ts,
        "updated_at": old_ts,
        "exchanges_done": 2,
        "exchanges_total": 10,
    }))
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 3
    assert_eq(backfill.needs_backfill(cid, msgs), True, "stale in_progress -> retry")


def test_needs_backfill_failed_state_returns_true():
    print("\n[test] needs_backfill=True when previous backfill failed")
    _wipe_storage()
    cid = "broke"
    backfill._write_state(cid, {"state": "failed", "started_at": "...", "error": "boom"})
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 3
    assert_eq(backfill.needs_backfill(cid, msgs), True, "failed -> retry")


# ---------------------------------------------------------------------------
# End-to-end run with mock vLLM
# ---------------------------------------------------------------------------

def _mock_async_client_factory(content: str):
    """Build a context-managed mock AsyncClient that returns canned content."""
    class _FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": content}}]}

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def post(self, *args, **kwargs):
            return _FakeResponse()

    return _FakeClient()


def test_run_backfill_end_to_end_writes_facts():
    print("\n[test] _run_backfill produces a facts file from message history")
    _wipe_storage()
    cid = "e2e"
    msgs = [
        {"role": "user", "content": "Tell me about Lyra."},
        {"role": "assistant", "content": "Lyra is a half-elf ranger."},
        {"role": "user", "content": "What world is this in?"},
        {"role": "assistant", "content": "It's Aethermere, a low-magic kingdom."},
    ]
    # Monkey-patch httpx.AsyncClient inside the backfill module so we don't
    # need a real vLLM.
    import httpx as real_httpx
    orig_client = real_httpx.AsyncClient

    def _fake_client(*args, **kwargs):
        return _mock_async_client_factory("- Lyra is a half-elf ranger.")

    real_httpx.AsyncClient = _fake_client
    try:
        asyncio.run(backfill._run_backfill(cid, msgs, "http://fake", "fake-model"))
    finally:
        real_httpx.AsyncClient = orig_client

    # State should be complete
    state = backfill.read_state(cid)
    assert_eq(state["state"], "complete", "state=complete after run")
    assert_eq(state["exchanges_done"], 2, "both exchanges processed")
    # Facts file should exist
    loaded = facts.load_facts(cid)
    assert_true(len(loaded) >= 1, "at least one fact recorded")


# ---------------------------------------------------------------------------
# F3 — a backfill must never destroy a store it did not read
# ---------------------------------------------------------------------------
#
# _run_backfill used to build `accumulated` from nothing and finish with
# conv_lock -> prune_facts -> save_facts. The lock wrapped the WRITE; the read
# that should have informed it never happened, so the write was a wholesale
# replacement of whatever was on disk by a reconstruction built minutes
# earlier from the CLIENT's message array (main.py:1509). Two live shapes:
# every _async_tail landing during the run was erased, and under the
# 2026-08-24 7-of-241 condition the "reconstruction" was seven messages.
#
# The fixtures below are the two real triggers from REMEDIATION.md §1.4 —
# a genuinely corrupt file and an OSError from open() — against real files
# in a real temp store. Path.stat is deliberately NOT patched: pathlib
# re-raises EIO, so a stat-patching fixture reproduces none of this.


def _key(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def corrupt(path: Path, content: str = "{ not valid js") -> Path:
    """Trigger 1: what a torn write or half-flushed MooseFS chunk leaves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@contextlib.contextmanager
def unreadable_when(path: Path, armed: dict):
    """Trigger 2: EIO from open() on one real, present, intact file — but
    only while `armed["v"]` is true, so a test can make the store go
    unreadable partway through a run that started against a readable one.

    Only reads go through builtins.open; atomic_write_json writes through
    os.fdopen, so this never blocks the state-file writes.
    """
    target = _key(path)
    real_open = open

    def _open(file, *a, **k):
        if (
            armed.get("v")
            and isinstance(file, (str, os.PathLike))
            and _key(file) == target
        ):
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real_open(file, *a, **k)

    with patch("builtins.open", _open):
        yield


def _scripted_client(contents: list[str], on_post=None):
    """AsyncClient stand-in that returns `contents` one per call (repeating
    the last), and runs `on_post()` — an async callback — before each reply.
    That callback is the interleaving hook: it is what lets a concurrent
    tail write land in the middle of a backfill, deterministically.
    """
    calls = {"n": 0}

    class _FakeResponse:
        def __init__(self, content):
            self._content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            i = calls["n"]
            calls["n"] += 1
            if on_post is not None:
                await on_post(i)
            return _FakeResponse(contents[min(i, len(contents) - 1)])

    return _FakeClient, calls


@contextlib.contextmanager
def _patched_httpx(client_cls):
    import httpx as real_httpx
    orig = real_httpx.AsyncClient
    real_httpx.AsyncClient = lambda *a, **k: client_cls()
    try:
        yield
    finally:
        real_httpx.AsyncClient = orig


_MSGS_4 = [
    {"role": "user", "content": "Tell me about Lyra."},
    {"role": "assistant", "content": "Lyra is a half-elf ranger."},
    {"role": "user", "content": "What world is this in?"},
    {"role": "assistant", "content": "It's Aethermere, a low-magic kingdom."},
]

_TAIL_FACTS = [
    {"text": "Mira's mother is called Sable.", "added_turn": 300, "last_used": 900},
    {"text": "The wedding is in the spring.", "added_turn": 302, "last_used": 901},
]


def test_merge_backfilled_existing_is_authoritative():
    print("\n[test] _merge_backfilled: disk wins, backfill only adds")
    existing = [
        {"text": "Lyra is a half-elf ranger.", "added_turn": 12, "last_used": 500},
        {"text": "Mira's mother is called Sable.", "added_turn": 14, "last_used": 501},
    ]
    accumulated = [
        # Same text the store already carries, from a reconstructed turn —
        # must NOT overwrite the real record's added_turn/last_used.
        {"text": "Lyra is a half-elf ranger.", "added_turn": 2, "last_used": 1},
        {"text": "Aethermere is a low-magic kingdom.", "added_turn": 4, "last_used": 1},
    ]
    merged = backfill._merge_backfilled(existing, accumulated)
    texts = [f["text"] for f in merged]
    assert_eq(len(merged), 3, "one genuinely new fact added, duplicate collapsed")
    assert_eq(texts[:2], [f["text"] for f in existing], "existing kept, in order")
    assert_eq(merged[0]["added_turn"], 12, "existing added_turn not rewritten")
    assert_eq(merged[0]["last_used"], 500, "existing last_used not backdated")
    assert_true("Aethermere is a low-magic kingdom." in texts, "new fact appended")


def test_backfill_merges_facts_written_while_it_ran():
    print("\n[test] concurrent tail write during a backfill — BOTH sets survive")
    _wipe_storage()
    cid = "race"

    async def scenario():
        wrote = asyncio.Event()

        async def on_post(i):
            # On the first extraction call, land a tail write the way
            # main.py's _async_tail does: locked read-modify-write. The
            # backfill's snapshot predates it and knows nothing about it.
            if i == 0:
                async with memory.conv_lock(cid):
                    facts.save_facts(cid, facts.load_facts(cid) + _TAIL_FACTS)
                wrote.set()
            await wrote.wait()

        client_cls, _ = _scripted_client(
            ["- Lyra is a half-elf ranger.", "- Aethermere is a low-magic kingdom."],
            on_post=on_post,
        )
        with _patched_httpx(client_cls):
            await backfill._run_backfill(cid, _MSGS_4, "http://fake", "fake-model")

    asyncio.run(scenario())

    texts = {f["text"] for f in facts.load_facts(cid)}
    assert_true(
        "Mira's mother is called Sable." in texts and
        "The wedding is in the spring." in texts,
        "the tail's facts survived the backfill's write",
    )
    assert_true(
        any("Lyra" in t for t in texts),
        "the backfill's own facts were written too",
    )
    assert_eq(backfill.read_state(cid)["state"], "complete", "state=complete")


def test_backfill_refuses_when_facts_already_exist():
    print("\n[test] backfill refuses against a non-empty store")
    _wipe_storage()
    cid = "live-store"
    facts.save_facts(cid, _TAIL_FACTS)
    before = facts.load_facts(cid)

    # A conversation with a real store and a TRUNCATED client array — the
    # 7-of-241 shape. If the refusal is missing this writes two reconstructed
    # facts over an established store.
    client_cls, calls = _scripted_client(["- Lyra is a half-elf ranger."])
    with _patched_httpx(client_cls):
        asyncio.run(backfill._run_backfill(cid, _MSGS_4, "http://fake", "fake-model"))

    assert_eq(calls["n"], 0, "refused before spending a single LLM call")
    assert_eq(facts.load_facts(cid), before, "store byte-for-byte unchanged")
    assert_true(
        backfill.read_state(cid) is None,
        "no backfill state written for a run that never started",
    )


def test_backfill_refuses_when_facts_unreadable():
    print("\n[test] backfill refuses against an unreadable store")
    _wipe_storage()
    cid = "corrupt-store"
    path = memory.facts_path(cid)
    corrupt(path)
    raw_before = path.read_text(encoding="utf-8")

    client_cls, calls = _scripted_client(["- Lyra is a half-elf ranger."])
    with _patched_httpx(client_cls):
        asyncio.run(backfill._run_backfill(cid, _MSGS_4, "http://fake", "fake-model"))

    assert_eq(calls["n"], 0, "refused before spending a single LLM call")
    assert_eq(path.read_text(encoding="utf-8"), raw_before,
              "the unreadable file was left exactly as found")


def test_backfill_skips_write_when_store_goes_unreadable_mid_run():
    print("\n[test] store becomes unreadable during the run -> write skipped")
    _wipe_storage()
    cid = "eio-midrun"
    path = memory.facts_path(cid)
    armed = {"v": False}

    async def scenario():
        wrote = asyncio.Event()

        async def on_post(i):
            if i == 0:
                # A tail writes real facts mid-run...
                async with memory.conv_lock(cid):
                    facts.save_facts(cid, facts.load_facts(cid) + _TAIL_FACTS)
                # ...and then the volume starts returning EIO on that file,
                # so the backfill's final read cannot see what it would be
                # replacing. That is exactly when it must not write.
                armed["v"] = True
                wrote.set()
            await wrote.wait()

        client_cls, _ = _scripted_client(
            ["- Lyra is a half-elf ranger."], on_post=on_post
        )
        with _patched_httpx(client_cls), unreadable_when(path, armed):
            await backfill._run_backfill(cid, _MSGS_4, "http://fake", "fake-model")

    asyncio.run(scenario())
    armed["v"] = False

    texts = [f["text"] for f in facts.load_facts(cid)]
    assert_eq(texts, [f["text"] for f in _TAIL_FACTS],
              "the tail's facts are still on disk, untouched")
    state = backfill.read_state(cid)
    assert_eq(state["state"], "failed", "run recorded as failed, not complete")
    # Not just "something went wrong": the state has to say the store was
    # unreadable at WRITE time, because that is the one failure an operator
    # must not read as "the backfill found nothing".
    assert_true("unreadable at write time" in (state.get("error") or ""),
                "state names the write-time read failure")
    assert_eq(state["exchanges_done"], 2, "work done is reported, not zeroed")


def test_backfill_releases_in_process_slot_on_refusal():
    print("\n[test] a refused backfill still releases its in-process slot")
    _wipe_storage()
    cid = "slot"
    facts.save_facts(cid, _TAIL_FACTS)
    backfill._in_progress_local.add(cid)

    client_cls, _ = _scripted_client(["- Lyra is a half-elf ranger."])
    with _patched_httpx(client_cls):
        asyncio.run(backfill._run_backfill(cid, _MSGS_4, "http://fake", "fake-model"))

    assert_true(cid not in backfill._in_progress_local,
                "conv_id released, so a later legitimate backfill can run")


def test_start_backfill_if_needed_idempotent_in_same_process():
    print("\n[test] start_backfill_if_needed returns False on duplicate call")
    _wipe_storage()
    cid = "dup"
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    fired = []

    def fake_fire(coro):
        # Don't actually run the coro — just count the spawn.
        fired.append(coro)
        coro.close()  # avoid "never awaited" warnings

    first = asyncio.run(backfill.start_backfill_if_needed(
        cid, msgs, "http://fake", "fake-model", fire_and_forget=fake_fire
    ))
    second = asyncio.run(backfill.start_backfill_if_needed(
        cid, msgs, "http://fake", "fake-model", fire_and_forget=fake_fire
    ))
    assert_eq(first, True, "first call starts backfill")
    assert_eq(second, False, "second call (same process, same conv) is no-op")
    assert_eq(len(fired), 1, "only one task spawned")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_state_path_in_facts_dir()
        test_read_state_missing_returns_none()
        test_write_then_read_state_roundtrip()

        test_is_stale_fresh_in_progress()
        test_is_stale_old_in_progress()
        test_is_stale_complete_state_never_stale()

        test_extract_pairs_basic()
        test_extract_pairs_skips_trailing_unmatched_user()
        test_extract_pairs_handles_system_mid_conversation()
        test_extract_pairs_multimodal_user_message()

        test_needs_backfill_short_conv_returns_false()
        test_needs_backfill_with_existing_facts_returns_false()
        test_needs_backfill_no_state_and_long_history_returns_true()
        test_needs_backfill_complete_state_returns_false()
        test_needs_backfill_stale_in_progress_returns_true()
        test_needs_backfill_failed_state_returns_true()

        test_run_backfill_end_to_end_writes_facts()

        test_merge_backfilled_existing_is_authoritative()
        test_backfill_merges_facts_written_while_it_ran()
        test_backfill_refuses_when_facts_already_exist()
        test_backfill_refuses_when_facts_unreadable()
        test_backfill_skips_write_when_store_goes_unreadable_mid_run()
        test_backfill_releases_in_process_slot_on_refusal()

        test_start_backfill_if_needed_idempotent_in_same_process()

        print("\nAll backfill smoke tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
