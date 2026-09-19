"""
v3.1.9.4 R5 (hostile pass #16 follow-up): P16-1 and P16-2.

P16-1 (MEDIUM, a regression from this release). B2 reordered
backfill.needs_backfill to check the backfill RECORD before the facts-file
tombstone /forget leaves. That means a conversation with a `failed`
(past its backoff) or stale `in_progress` backfill record survives a
/forget untouched, and her very next message re-extracts the whole
forgotten history — while /forget already reported a clean wipe.

Fix (backfill.py):
  - `backfill.mark_wiped(conv_id)`: called from every wipe path
    (main._clear_all_memory, commands._handle_retire's apply step,
    portability.import_conversation's W1 overwrite), inside the SAME
    locked section the wipe itself runs in. Rewrites any existing backfill
    record to the terminal "wiped" state.
  - `backfill._facts_tombstoned(conv_id)`: defense in depth, checked from
    needs_backfill's `failed`/stale-`in_progress` branches regardless of
    what the record itself says — an existing-but-EMPTY facts file is
    never written by anything except a wipe's own tombstone.

P16-2 (LOW). A /forget during a backfill only discarded its writes at the
end; the run kept calling vLLM for the rest of its (potentially
multi-hour) length. Fix: `_run_backfill`'s extraction loop now checks the
wipe generation (an unlocked dict read) at the TOP of every iteration and
stops immediately on a mismatch, writing "wiped".

Each check has a CONTROL proving the legitimate case still works: a
`failed`/stale record with NO wipe in between still retries, and a
backfill with no /forget in between still runs to completion.

Synthetic fixtures only; no HTTP calls (extraction and the vLLM client
stubbed). Run: python test_v3194_r5_p16_1_2.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-r5-p161-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "true"
os.environ["COMPACTOR_HIERARCHICAL_SUMMARY"] = "false"

import backfill  # noqa: E402
import commands  # noqa: E402
import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def _wipe_storage():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP)
    memory.ensure_storage_layout()


def _msgs(n_pairs: int) -> list[dict]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        out.append({"role": "user", "content": f"user turn {i} with enough real text to extract from"})
        out.append({"role": "assistant", "content": f"assistant reply {i} with enough real text to extract from"})
    return out


class _CountingExtraction:
    def __init__(self):
        self.calls = 0

    async def __call__(self, client, vllm_url, model, u, a, existing, **kw):
        self.calls += 1
        return [f"reconstructed synthetic fact #{self.calls}"]


def _age_updated_at(cid, seconds):
    p = backfill._backfill_state_path(cid)
    d = json.loads(p.read_text(encoding="utf-8"))
    d["updated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="seconds")
    p.write_bytes(json.dumps(d).encode("utf-8"))


def _make_failed_record(cid, *, past_backoff=True):
    backfill._write_state(cid, {
        "state": "failed", "started_at": "x", "exchanges_done": 2,
        "exchanges_total": 6, "attempts": 1, "error": "synthetic",
    })
    if past_backoff:
        _age_updated_at(cid, backfill._BACKFILL_RETRY_BACKOFF_S + 5)


def _make_stale_in_progress_record(cid):
    backfill._write_state(cid, {
        "state": "in_progress", "started_at": "x", "exchanges_done": 2,
        "exchanges_total": 6, "attempts": 1, "error": None,
    })
    _age_updated_at(cid, backfill._STALE_SECONDS + 5)


# ---------------------------------------------------------------------------
# P16-1: mark_wiped, called by every wipe path
# ---------------------------------------------------------------------------

def test_control_failed_record_past_backoff_retries_with_no_wipe_in_between():
    print("\n[test] CONTROL: a failed record past its backoff, untouched by "
          "any wipe, still retries")
    _wipe_storage()
    cid = "p161-control-failed"
    _make_failed_record(cid, past_backoff=True)
    check(backfill.needs_backfill(cid, _msgs(6)) is True,
          "needs_backfill still says yes with no wipe in the picture")


def test_control_stale_in_progress_retries_with_no_wipe_in_between():
    print("\n[test] CONTROL: a stale in_progress record, untouched by any "
          "wipe, still retries")
    _wipe_storage()
    cid = "p161-control-inprogress"
    _make_stale_in_progress_record(cid)
    check(backfill.needs_backfill(cid, _msgs(6)) is True,
          "needs_backfill still says yes with no wipe in the picture")


def test_mark_wiped_rewrites_a_failed_record_and_needs_backfill_refuses():
    print("\n[test] mark_wiped rewrites a failed (past-backoff) record to "
          "'wiped'; needs_backfill refuses to retry it")
    _wipe_storage()
    cid = "p161-failed-wiped"
    _make_failed_record(cid, past_backoff=True)
    check(backfill.needs_backfill(cid, _msgs(6)) is True, "sanity: would retry before the wipe")
    backfill.mark_wiped(cid)
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped", f"record is now 'wiped': {state!r}")
    check(backfill.needs_backfill(cid, _msgs(6)) is False,
          "needs_backfill refuses after mark_wiped")


def test_mark_wiped_rewrites_a_stale_in_progress_record_and_needs_backfill_refuses():
    print("\n[test] mark_wiped rewrites a stale in_progress record to "
          "'wiped'; needs_backfill refuses to retry it")
    _wipe_storage()
    cid = "p161-inprogress-wiped"
    _make_stale_in_progress_record(cid)
    check(backfill.needs_backfill(cid, _msgs(6)) is True, "sanity: would retry before the wipe")
    backfill.mark_wiped(cid)
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped", f"record is now 'wiped': {state!r}")
    check(backfill.needs_backfill(cid, _msgs(6)) is False,
          "needs_backfill refuses after mark_wiped")


def test_mark_wiped_is_a_no_op_with_no_record():
    print("\n[test] mark_wiped does nothing when there is no backfill record")
    _wipe_storage()
    cid = "p161-no-record"
    check(backfill.read_state(cid) is None, "sanity: no record yet")
    backfill.mark_wiped(cid)
    check(backfill.read_state(cid) is None,
          "still no record — mark_wiped did not invent one")


def test_facts_tombstone_alone_blocks_retry_even_without_mark_wiped():
    print("\n[test] defense in depth: an empty facts file blocks retry of a "
          "failed/stale record even if mark_wiped never ran (e.g. a record "
          "written before this fix shipped)")
    _wipe_storage()
    cid = "p161-tombstone-only"
    _make_failed_record(cid, past_backoff=True)
    # The tombstone a wipe leaves — written directly, bypassing mark_wiped,
    # to prove the SECOND, independent defence catches it on its own.
    facts.save_facts(cid, [])
    check(backfill.needs_backfill(cid, _msgs(6)) is False,
          "needs_backfill refuses on the tombstone alone, record untouched")
    # And the same for a stale in_progress record.
    cid2 = "p161-tombstone-only-ip"
    _make_stale_in_progress_record(cid2)
    facts.save_facts(cid2, [])
    check(backfill.needs_backfill(cid2, _msgs(6)) is False,
          "same defence for a stale in_progress record")


def test_end_to_end_forget_then_next_message_does_not_reextract():
    print("\n[test] end-to-end (order A: record predates /forget): /forget "
          "on a conv with a failed backfill record leaves needs_backfill "
          "False, and driving start_backfill_if_needed makes ZERO "
          "extraction calls")
    _wipe_storage()
    cid = "p161-e2e-failed"
    facts.save_facts(cid, [{"text": "synthetic live fact", "added_turn": 2, "last_used": 1}])
    _make_failed_record(cid, past_backoff=True)

    reply = asyncio.run(commands._handle_forget("", cid, {"clear_all_memory": main._clear_all_memory}))
    check("Forgot" in reply or "Cleared" in reply, f"reply reports a wipe: {reply!r}")
    check(facts.load_facts(cid) == [], "facts are empty after /forget")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped", f"record reads 'wiped': {state!r}")
    check(backfill.needs_backfill(cid, _msgs(6)) is False, "needs_backfill refuses on her next message")

    ex = _CountingExtraction()
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = ex
    try:
        started = asyncio.run(_drive_start(cid, _msgs(6)))
    finally:
        facts.extract_facts_from_exchange = orig
    check(started is False, "start_backfill_if_needed refuses to start")
    check(ex.calls == 0, f"zero extraction calls over the forgotten history (got {ex.calls})")
    check(facts.load_facts(cid) == [], "facts remain empty — nothing was reconstructed")


def test_end_to_end_forget_during_stale_in_progress_order_b():
    print("\n[test] end-to-end (order B: record is stale in_progress at "
          "/forget time — the redeploy-during-backfill shape): same "
          "outcome")
    _wipe_storage()
    cid = "p161-e2e-inprogress"
    facts.save_facts(cid, [{"text": "synthetic live fact", "added_turn": 2, "last_used": 1}])
    _make_stale_in_progress_record(cid)

    reply = asyncio.run(commands._handle_forget("", cid, {"clear_all_memory": main._clear_all_memory}))
    check("Forgot" in reply or "Cleared" in reply, f"reply reports a wipe: {reply!r}")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped", f"record reads 'wiped': {state!r}")
    check(backfill.needs_backfill(cid, _msgs(6)) is False, "needs_backfill refuses on her next message")

    ex = _CountingExtraction()
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = ex
    try:
        started = asyncio.run(_drive_start(cid, _msgs(6)))
    finally:
        facts.extract_facts_from_exchange = orig
    check(started is False, "start_backfill_if_needed refuses to start")
    check(ex.calls == 0, f"zero extraction calls over the forgotten history (got {ex.calls})")


async def _drive_start(cid, msgs):
    tasks = []

    def ff(coro, label=None):
        tasks.append(asyncio.ensure_future(coro))
        return True

    started = await backfill.start_backfill_if_needed(
        cid, msgs, "http://stub", "m", fire_and_forget=ff,
    )
    for t in tasks:
        await t
    return started


# ---------------------------------------------------------------------------
# P16-2: the extraction loop stops spending vLLM calls once a wipe lands
# ---------------------------------------------------------------------------

class _WipeAfterN:
    """Extractor stub that bumps this conv's wipe generation (simulating a
    concurrent /forget) right after the Nth call, then keeps counting any
    FURTHER calls the loop makes — which should be none."""

    def __init__(self, cid, n):
        self.cid = cid
        self.n = n
        self.calls = 0

    async def __call__(self, client, vllm_url, model, u, a, existing, **kw):
        self.calls += 1
        if self.calls == self.n:
            memory.bump_wipe_generation(self.cid)
        return [f"fact #{self.calls}"]


def test_control_run_backfill_completes_normally_with_no_wipe():
    print("\n[test] CONTROL: no /forget during the run — all exchanges "
          "extracted, record reads 'complete'")
    _wipe_storage()
    cid = "p162-control"
    msgs = _msgs(6)
    stub = _CountingExtraction()
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    wg = memory.current_wipe_generation(cid)
    try:
        asyncio.run(backfill._run_backfill(cid, msgs, "http://stub", "m", wipe_generation=wg))
    finally:
        facts.extract_facts_from_exchange = orig
    check(stub.calls == 6, f"all 6 exchanges extracted (got {stub.calls})")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "complete", f"record reads complete: {state!r}")


def test_wipe_mid_loop_stops_extraction_within_one_iteration():
    print("\n[test] a /forget that lands mid-loop stops the extraction "
          "loop immediately — not after every remaining exchange")
    _wipe_storage()
    cid = "p162-mid-wipe"
    n_pairs = 20
    msgs = _msgs(n_pairs)
    stub = _WipeAfterN(cid, n=3)  # wipe lands right after the 3rd call
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    wg = memory.current_wipe_generation(cid)
    try:
        asyncio.run(backfill._run_backfill(cid, msgs, "http://stub", "m", wipe_generation=wg))
    finally:
        facts.extract_facts_from_exchange = orig
    # Without the fix, this would be n_pairs (20): every remaining exchange
    # still gets extracted before the two LOCKED checks (start/end) ever
    # notice the wipe. With the fix, the loop's own unlocked check catches
    # it at the top of the NEXT iteration after the wipe lands.
    check(stub.calls <= 4, f"stopped within one iteration of the wipe landing (calls={stub.calls}, of {n_pairs} total)")
    check(stub.calls >= 3, f"at least the 3 calls before the wipe happened (calls={stub.calls})")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped", f"record reads 'wiped': {state!r}")
    check(facts.load_facts(cid) == [], "nothing from this run landed in the facts store")


def _all_tests():
    return [
        test_control_failed_record_past_backoff_retries_with_no_wipe_in_between,
        test_control_stale_in_progress_retries_with_no_wipe_in_between,
        test_mark_wiped_rewrites_a_failed_record_and_needs_backfill_refuses,
        test_mark_wiped_rewrites_a_stale_in_progress_record_and_needs_backfill_refuses,
        test_mark_wiped_is_a_no_op_with_no_record,
        test_facts_tombstone_alone_blocks_retry_even_without_mark_wiped,
        test_end_to_end_forget_then_next_message_does_not_reextract,
        test_end_to_end_forget_during_stale_in_progress_order_b,
        test_control_run_backfill_completes_normally_with_no_wipe,
        test_wipe_mid_loop_stops_extraction_within_one_iteration,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R5 P16-1/P16-2 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
