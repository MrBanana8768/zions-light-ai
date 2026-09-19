"""
CPU-only tests for v3.1.9.4 (P15-7): two data-loss guards on the memory-write
path that no test anywhere pinned before this file.

  - M5: main._facts_tail's `facts.prune_facts(combined, conv_id=conv_id)`
    archives evicted facts rather than deleting them. `_facts_tail` itself is
    NOT this lane's to edit (it is another lane's region of main.py) — this
    file only adds the missing test, driving the REAL tail path end to end
    with an over-budget store.
  - M9: summarizer._do_l3_rollup aborts before any state mutates when
    _archive_chapters fails. `_do_l3_rollup` IS this lane's file
    (summarizer.py) but is not itself being changed here either — per the
    finding, this is a test gap, not a code defect ("n/a — a gap in the
    gate, not a defect in the code").

Both are mutation-tested against a COPY of the tree (never via git) to prove
the new test actually pins the guard: M5 reverts `conv_id=conv_id` to a bare
call (the exact mutation the reviewer's mut15.py M5 already showed SURVIVED
the whole unit suite); M9 deletes the `return False` after a failed archive
(the exact mutation mut15.py M9 already showed SURVIVED). Both were
independently reproduced surviving in this lane's own run before the fix —
see fix-3194-mem.md.

Run: python test_v3194_mem_guards.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-mem-guards-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
# M5: a small budget so a handful of synthetic facts is already over it —
# no need for hundreds of rows to exercise real eviction.
os.environ["COMPACTOR_MAX_FACTS_TOKENS"] = "120"
# M9: keep L3 reachable without a long synthetic conversation — chapters are
# seeded directly into state, never rolled up from L1/L2 here.
os.environ["COMPACTOR_L3_CHUNK_SIZE"] = "2"

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import summarizer  # noqa: E402


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
# M5: an over-budget store through the real _facts_tail path archives,
# rather than deletes, its evictions.
# ---------------------------------------------------------------------------

async def _no_extraction(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
    """Stand in for the vLLM extraction call: no new facts, so dedup (which
    only runs when `new_entries` is non-empty) never fires either — this
    test is about prune_facts, not extraction or dedup."""
    return []


def test_over_budget_tail_archives_evicted_facts_not_deletes_them():
    print("\n[test] M5: _facts_tail's real prune_facts(..., conv_id=conv_id) call archives what it evicts")
    _wipe()
    cid = "guard-m5-over-budget"
    now = int(time.time())
    # 40 rows, ascending last_used (0 is oldest / first evicted under LRU).
    # Each row is well over COMPACTOR_MAX_FACTS_TOKENS=120 combined —
    # _estimate_tokens is char/4, and 40 rows of ~50 chars each price at
    # roughly 500+ tokens for the block, several times the budget.
    rows = [
        {"text": f"synthetic fact number {i:03d} about the garden plan", "added_turn": i + 1, "last_used": now + i}
        for i in range(40)
    ]
    facts.save_facts(cid, rows)
    assert_true(len(facts.load_facts(cid)) == 40, "seeded 40 facts directly")
    assert_eq(facts.load_archive(cid), [], "nothing archived yet")

    orig_extract = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = _no_extraction
    try:
        asyncio.run(main._facts_tail(
            cid, [], "a synthetic user turn", "a synthetic assistant reply", 41,
        ))
    finally:
        facts.extract_facts_from_exchange = orig_extract

    kept = facts.load_facts(cid)
    archived = facts.load_archive(cid)
    assert_true(len(kept) < 40, f"the store was pruned down from 40 (now {len(kept)})")
    assert_true(len(archived) > 0, "SOMETHING was evicted — the budget really was exceeded")
    assert_eq(len(kept) + len(archived), 40, "every evicted row landed in the archive — none just vanished")
    kept_texts = {f["text"] for f in kept}
    archived_texts = {f["text"] for f in archived}
    assert_eq(kept_texts & archived_texts, set(), "kept and archived are disjoint — nothing counted twice")
    # LRU: the OLDEST (lowest last_used, i.e. the earliest-numbered) rows are
    # the ones evicted.
    oldest_text = rows[0]["text"]
    assert_true(oldest_text in archived_texts, "the single oldest fact (lowest last_used) was among those evicted")


def test_under_budget_tail_archives_nothing():
    print("\n[test] CONTROL: a store comfortably under budget evicts (and archives) nothing")
    _wipe()
    cid = "guard-m5-under-budget"
    now = int(time.time())
    facts.save_facts(cid, [{"text": "one small fact", "added_turn": 1, "last_used": now}])

    orig_extract = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = _no_extraction
    try:
        asyncio.run(main._facts_tail(
            cid, [], "a synthetic user turn", "a synthetic assistant reply", 2,
        ))
    finally:
        facts.extract_facts_from_exchange = orig_extract

    assert_eq(len(facts.load_facts(cid)), 1, "the one fact is untouched")
    assert_eq(facts.load_archive(cid), [], "nothing was evicted, so nothing was archived")


# ---------------------------------------------------------------------------
# M9: a failed chapter-archive write aborts the L3 refresh before l2/l3
# mutate — proven on disk, through the real maybe_rollup, not just in the
# in-memory state dict _do_l3_rollup is handed.
# ---------------------------------------------------------------------------

def _seed_l3_ready_state(conv_id: str) -> dict:
    """Two L2 chapters (>= COMPACTOR_L3_CHUNK_SIZE=2 from this module's own
    env, set above import) and no l1 due — _needs_l3_rollup is true,
    _needs_l1_rollup and _needs_l2_rollup are both false, so maybe_rollup's
    drain reaches L3 and ONLY L3."""
    st = summarizer._empty_state(conv_id)
    st["l2"] = [
        {"text": "chapter one synthetic summary.", "first_turn": 1, "last_turn": 100},
        {"text": "chapter two synthetic summary.", "first_turn": 101, "last_turn": 200},
    ]
    st["l3"] = None
    st["last_summarized_turn"] = 200
    st["turns_seen"] = 200
    summarizer.save_state(conv_id, st)
    return st


def _msgs(n_turns: int) -> list[dict]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        out.append({"role": "user", "content": f"user turn {i}"})
        out.append({"role": "assistant", "content": f"assistant turn {i}"})
    return out


class _StubClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        class _Resp:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"choices": [{"message": {"content": "the L3 theme summary."}, "finish_reason": "stop"}]}
        if str(url).endswith("/tokenize"):
            raise RuntimeError("no /tokenize in this stub")
        return _Resp()


def test_failed_archive_write_aborts_l3_refresh_before_state_mutates():
    print("\n[test] M9: _archive_chapters raising aborts the L3 refresh — l2 and l3 are unchanged ON DISK")
    _wipe()
    cid = "guard-m9-archive-fails"
    before = _seed_l3_ready_state(cid)

    orig_archive = summarizer._archive_chapters

    def _boom(conv_id, chapters):
        raise OSError("synthetic disk failure for this test")

    summarizer._archive_chapters = _boom
    orig_client = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubClient()
    try:
        asyncio.run(summarizer.maybe_rollup(cid, _msgs(0), "http://stub:8000", "m"))
    finally:
        summarizer._archive_chapters = orig_archive
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    assert_eq(after["l2"], before["l2"], "l2 on disk is byte-identical to before the failed refresh — nothing dropped")
    assert_eq(after["l3"], before["l3"], "l3 on disk is unchanged — no partial refresh was recorded")
    # Nothing was archived either — the abort happens BEFORE the chapters
    # would have been consumed, and _archive_chapters itself never got past
    # raising, so the cold store must not exist.
    assert_eq(summarizer.load_chapter_archive(cid), [], "no chapters landed in cold storage from the aborted attempt")


def test_control_successful_archive_write_lets_l3_refresh_land():
    print("\n[test] CONTROL: when _archive_chapters succeeds, the L3 refresh lands normally (l2 clears, l3 populates)")
    _wipe()
    cid = "guard-m9-archive-succeeds"
    before = _seed_l3_ready_state(cid)

    orig_client = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubClient()
    try:
        asyncio.run(summarizer.maybe_rollup(cid, _msgs(0), "http://stub:8000", "m"))
    finally:
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    assert_eq(after["l2"], [], "l2 was consumed and cleared by the successful refresh")
    assert_true(after["l3"] is not None and after["l3"]["text"], "l3 now holds the refreshed theme summary")
    assert_eq(len(summarizer.load_chapter_archive(cid)), 2, "both chapters landed in cold storage")


def main_():
    tests = [
        test_over_budget_tail_archives_evicted_facts_not_deletes_them,
        test_under_budget_tail_archives_nothing,
        test_failed_archive_write_aborts_l3_refresh_before_state_mutates,
        test_control_successful_archive_write_lets_l3_refresh_land,
    ]
    for t in tests:
        t()
    print("\nAll v3194-mem guard tests passed.")


if __name__ == "__main__":
    main_()
