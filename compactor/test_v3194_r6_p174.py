"""
v3.1.9.4 R6 (hostile pass #17): P17-4.

Residual of P16-1/P16-2 (R5)'s restart-safety fix: a `/forget` (or the
admin facts-delete endpoint) landing DURING the LAST extraction call of a
running `backfill._run_backfill` writes `"wiped"` (via `mark_wiped`,
inside the wipe's own `conv_lock`), but the loop's own post-extraction
progress write — unconditional before this fix, with no re-check between
the extraction `await` and the write — lands right behind it and turns
`"wiped"` back into `"in_progress"`. The top-of-loop generation check only
proves no wipe had landed BEFORE that exchange's (possibly slow)
extraction call started; a wipe landing WHILE it is in flight is invisible
to it, and on the LAST exchange there is no next iteration to catch it at
the next top-of-loop check either. A process kill in the gap before the
final LOCKED check (plausible: that check waits on `conv_lock`, and a live
tail holds it across an extraction and a rollup) leaves exactly that
clobbered `"in_progress"` on disk, which later goes stale and becomes
retryable — silently re-extracting the history she just asked to forget.

Fix (`backfill.py`):
  1. Re-check the wipe generation AFTER the extraction call, before the
     progress write, mirroring the top-of-loop check (same terminal
     "wiped" write and early return on a mismatch).
  2. `_write_state_unless_wiped` — every ORDINARY state write in
     `_run_backfill` (the initial "in_progress", the per-exchange
     progress, `_write_failed_or_abandoned`, and the final "complete")
     refuses to overwrite a record that already reads "wiped" on disk,
     belt-and-suspenders for whatever gap the generation re-check alone
     does not close.

Follow-up (`main.py`): `_clear_all_memory` used to write the empty-facts
tombstone only when `n_facts > 0`, unlike `commands._wipe_all_layers` (the
chat /forget path), which writes it unconditionally. The admin DELETE
endpoint calls `_clear_all_memory` directly, so a wipe on a conversation
with NO facts left no tombstone at all — the one defense-in-depth signal
that (pre-P17-4) accidentally saved the chat path from this exact race,
per the reviewer's own probe table. `_clear_all_memory` now writes the
tombstone unconditionally too, matching `_wipe_all_layers`.

Follow-up (`backfill._facts_tombstoned`): its docstring claimed nothing
but a wipe's tombstone ever leaves an empty facts file — not literally
true (`/tidy`, a selective `/forget <substring>` matching every fact, the
admin archive-stale sweep, and an empty-bundle import can too). `/tidy`
and archive-stale are categorically different: they ARCHIVE rather than
delete, so `_facts_tombstoned` now also requires the archive sidecar to be
empty, which is what a genuine wipe (via `_wipe_all_layers`) also clears —
excluding the archive/tidy false-positive while still recognizing a real
wipe.

Real `backfill._run_backfill`, real `main._clear_all_memory` and
`commands._wipe_all_layers`. A concurrent task holds `conv_lock` the way a
live tail does (across an extraction and a rollup); the backfill task is
cancelled mid-wait to simulate a process kill, mirroring the reviewer's
own probe (`SP\\p17\\probe_p161_lastiter.py`). Synthetic content only.

Run: python test_v3194_r6_p174.py
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-r6-p174-")
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


N = 5  # exchanges — the wipe lands during the LAST one


def _msgs():
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(N):
        msgs += [
            {"role": "user", "content": f"user turn {i} with enough text to be a real exchange"},
            {"role": "assistant", "content": f"assistant reply {i} with enough text to be a real exchange"},
        ]
    return msgs


async def _wipe_during_last_call_then_kill(cid, wipe_path):
    """Runs `_run_backfill`, wiping `cid` (via `wipe_path`, "chat" or
    "admin") DURING the last extraction call, with a concurrent task
    holding `conv_lock` the way a live tail does across its own
    extraction + rollup — so the backfill's final LOCKED check has to
    wait. Cancels both tasks (a process kill) while it is waiting.
    Returns the record read right after the wipe (still inside the last
    call) and the record read right after the kill.
    """
    calls = [0]
    during_wipe_record = {}
    tail_holds = asyncio.Event()

    async def stub(client, url, model, u, a, acc, conv_id=None, **kw):
        calls[0] += 1
        if calls[0] == N:
            if wipe_path == "chat":
                await commands._wipe_all_layers(
                    cid, lambda c: main._clear_all_memory(c, source="chat-command"),
                )
            else:
                await main._clear_all_memory(cid, source="admin")
            during_wipe_record["state"] = backfill.read_state(cid)["state"]
            tail_holds.set()
            await asyncio.sleep(0.05)
        return [f"fact from exchange {calls[0]}"]

    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub

    async def live_tail_holding_lock():
        await tail_holds.wait()
        async with memory.conv_lock(cid):
            await asyncio.sleep(1.0)  # extraction + rollup, held under the lock

    try:
        wg = memory.current_wipe_generation(cid)
        t_tail = asyncio.create_task(live_tail_holding_lock())
        t_bf = asyncio.create_task(
            backfill._run_backfill(cid, _msgs(), "http://stub", "m", wipe_generation=wg)
        )
        await asyncio.sleep(0.5)
        waiting_record = backfill.read_state(cid)
        t_bf.cancel()
        t_tail.cancel()
        for t in (t_bf, t_tail):
            try:
                await t
            except asyncio.CancelledError:
                pass
    finally:
        facts.extract_facts_from_exchange = orig

    after_kill = backfill.read_state(cid)
    return during_wipe_record, waiting_record, after_kill


def _age_past_stale(cid):
    d = json.loads(backfill._backfill_state_path(cid).read_text())
    d["updated_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=backfill._STALE_SECONDS + 60)
    ).isoformat(timespec="seconds")
    backfill._backfill_state_path(cid).write_text(json.dumps(d))


def _run_scenario(label, wipe_path):
    print(f"\n[test] P17-4 ({label}): a wipe landing during the LAST "
          f"extraction call survives a kill before the final locked check")
    cid = f"p174-{wipe_path}-{label}".replace(" ", "-")
    during, waiting, after_kill = asyncio.run(
        _wipe_during_last_call_then_kill(cid, wipe_path)
    )
    check(during.get("state") == "wiped",
          f"record reads 'wiped' immediately after the wipe (during the last call): {during}")
    check(waiting.get("state") == "wiped",
          f"still 'wiped' while backfill waits on conv_lock for its final check: {waiting}")
    check(after_kill.get("state") == "wiped",
          f"STILL 'wiped' after the kill — this is the P17-4 fix: {after_kill}")

    _age_past_stale(cid)
    check(backfill.needs_backfill(cid, _msgs()) is False,
          "needs_backfill refuses to retry the now-stale 'wiped' record")

    # A live tail landing a fact AFTER the forget must not resurrect the
    # backfill either — the record's own terminal state is what matters,
    # not whether the store happens to be non-empty again.
    facts.save_facts(cid, [{"text": "post-forget fact from her next message",
                             "added_turn": 12, "last_used": 0, "pin": False}])
    check(backfill.needs_backfill(cid, _msgs()) is False,
          "needs_backfill still refuses after a post-forget live-tail fact")
    return cid


def test_p174_chat_forget_path():
    _run_scenario("case-a", "chat")


def test_p174_admin_delete_path():
    _run_scenario("case-a", "admin")


def test_p174_control_no_wipe_completes_normally():
    print("\n[test] P17-4 CONTROL: no wipe at all — the backfill still "
          "completes normally and the fix does not touch that path")
    cid = "p174-control-no-wipe"
    calls = [0]

    async def stub(client, url, model, u, a, acc, conv_id=None, **kw):
        calls[0] += 1
        return [f"fact {calls[0]}"]

    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(backfill._run_backfill(cid, _msgs(), "http://stub", "m"))
    finally:
        facts.extract_facts_from_exchange = orig

    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "complete",
          f"record reads 'complete': {state}")
    check(calls[0] == N, f"all {N} exchanges extracted: {calls[0]}")
    check(len(facts.load_facts(cid)) == N, f"all facts landed: {facts.load_facts(cid)}")


# ---------------------------------------------------------------------------
# main._clear_all_memory: the tombstone follow-up.
# ---------------------------------------------------------------------------

def test_p174_clear_all_memory_writes_tombstone_even_with_no_facts():
    print("\n[test] P17-4 follow-up: main._clear_all_memory now writes the "
          "empty-facts tombstone unconditionally, matching "
          "commands._wipe_all_layers (the chat /forget path) — not just "
          "'if n_facts > 0'")
    cid = "p174-tombstone-admin-no-facts"
    check(not facts.facts_path(cid).is_file(), "starts with no facts file at all")
    asyncio.run(main._clear_all_memory(cid, source="test"))
    check(facts.facts_path(cid).is_file() and facts.load_facts(cid) == [],
          "the tombstone now exists even though this conv never had a fact")


def test_p174_control_clear_all_memory_with_existing_facts_unaffected():
    print("\n[test] P17-4 follow-up CONTROL: a conv that DID have facts "
          "still gets the tombstone, same as before this change")
    cid = "p174-tombstone-admin-had-facts"
    facts.save_facts(cid, [{"text": "a fact", "added_turn": 1, "last_used": 1}])
    asyncio.run(main._clear_all_memory(cid, source="test"))
    check(facts.load_facts(cid) == [], "facts cleared to the tombstone")


# ---------------------------------------------------------------------------
# backfill._facts_tombstoned: the archive-sidecar follow-up.
# ---------------------------------------------------------------------------

def test_p174_tidy_or_archive_stale_is_not_a_false_tombstone():
    print("\n[test] P17-4 follow-up: _facts_tombstoned no longer treats "
          "an empty ACTIVE facts file as a wipe's tombstone when the "
          "ARCHIVE sidecar still holds the (recoverable) facts — the "
          "/tidy and archive-stale shape")
    cid = "p174-archived-not-wiped"
    facts.save_facts(cid, [])
    facts.save_archive(cid, [{"text": "archived, not deleted", "added_turn": 1,
                               "last_used": 1, "archived_at": 1}])
    check(backfill._facts_tombstoned(cid) is False,
          "NOT tombstoned: the archive still holds recoverable facts")


def test_p174_control_genuine_wipe_with_empty_archive_is_still_tombstoned():
    print("\n[test] P17-4 follow-up CONTROL: a genuine wipe (empty active "
          "facts AND empty/absent archive) still reads as tombstoned")
    cid = "p174-genuinely-wiped"
    facts.save_facts(cid, [])
    check(backfill._facts_tombstoned(cid) is True,
          "tombstoned: empty facts, no archive at all")
    facts.save_archive(cid, [])
    check(backfill._facts_tombstoned(cid) is True,
          "tombstoned: empty facts, explicitly empty archive too")


def test_p174_control_no_facts_file_at_all_is_not_tombstoned():
    print("\n[test] P17-4 follow-up CONTROL: no facts file at all (never "
          "attempted) is not a tombstone either")
    cid = "p174-never-had-facts"
    check(backfill._facts_tombstoned(cid) is False, "no file -> not tombstoned")


def _all_tests():
    return [
        test_p174_chat_forget_path,
        test_p174_admin_delete_path,
        test_p174_control_no_wipe_completes_normally,
        test_p174_clear_all_memory_writes_tombstone_even_with_no_facts,
        test_p174_control_clear_all_memory_with_existing_facts_unaffected,
        test_p174_tidy_or_archive_stale_is_not_a_false_tombstone,
        test_p174_control_genuine_wipe_with_empty_archive_is_still_tombstoned,
        test_p174_control_no_facts_file_at_all_is_not_tombstoned,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R6 P17-4 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
