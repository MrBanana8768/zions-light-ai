"""
CPU-only tests for compactor/continuity.py (v3.1.9.8 `.15g`, r10).

This suite exists because commit 7650668 shipped `record_wipe`,
`record_targeted_forget` and `retire_tombstone` calling `_ledger_path`,
which was defined nowhere — a NameError on every wipe path (`/forget`,
`/forget-all`, `/retire`) that reached the ledger. Every test below that
touches the ledger (round trip, epoch bump, targeted forget, tombstone,
reapply) would have raised NameError on the pre-fix code; that is the
regression this file guards against, not just the happy path.

Also covers: `_ledger_path`'s conv_id validation (reused from
`memory._safe_path`, the same chokepoint every other per-conversation file
goes through — see that function's docstring), the documented fail-closed
behaviour on a corrupt ledger, and (per portability.quarantine_
conversation's docstring, which names this file) that a quarantine
snapshot never carries the digest cache.

Run: python test_continuity.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

# Storage/control redirects MUST happen before importing memory/continuity
# so the module-level paths see the override.
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-continuity-")
_STORAGE_ROOT = os.path.join(_TMP_ROOT, "storage")
_CONTROL_ROOT = os.path.join(_TMP_ROOT, "control")
# v3.1.9.8 round-4 review (item 3): PENDING_ROOT is a THIRD sibling root,
# same status as STORAGE_ROOT/CONTROL_ROOT — set explicitly so this file's
# pending-intent tests never fall back to continuity._default_pending_
# root()'s shared `/tmp/compactor-pending`, which every test file that
# omits this would otherwise collide on within the same container run.
_PENDING_ROOT = os.path.join(_TMP_ROOT, "pending")
os.environ["COMPACTOR_STORAGE_ROOT"] = _STORAGE_ROOT
os.environ["COMPACTOR_CONTROL_ROOT"] = _CONTROL_ROOT
os.environ["COMPACTOR_PENDING_ROOT"] = _PENDING_ROOT

import continuity  # noqa: E402
import memory  # noqa: E402
import portability  # noqa: E402


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


class _CaptureLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _wipe_storage():
    if os.path.exists(_STORAGE_ROOT):
        shutil.rmtree(_STORAGE_ROOT)
    if os.path.exists(_CONTROL_ROOT):
        shutil.rmtree(_CONTROL_ROOT)
    if os.path.exists(_PENDING_ROOT):
        shutil.rmtree(_PENDING_ROOT)
    memory.ensure_storage_layout()


# ---------------------------------------------------------------------------
# _ledger_path — the path convention and its conv_id validation
# ---------------------------------------------------------------------------

def test_ledger_path_matches_the_documented_convention():
    print("\n[test] _ledger_path builds CONTROL_ROOT/wipes/<conv_id>.json")
    _wipe_storage()
    p = continuity._ledger_path("conv-abc123")
    assert_eq(p, continuity.CONTROL_ROOT / "wipes" / "conv-abc123.json", "path shape")


def test_ledger_path_rejects_traversal_conv_id():
    print("\n[test] a hostile conv_id ('../x') is refused, never escapes wipes/")
    _wipe_storage()
    for hostile in ("../x", "a/b", "..", "", "a\x00b"):
        try:
            continuity._ledger_path(hostile)
        except memory.UnsafeConvId:
            print(f"  ok   rejected {hostile!r}")
            continue
        print(f"FAIL: _ledger_path accepted hostile conv_id {hostile!r}")
        sys.exit(1)


def test_ledger_path_never_writes_outside_control_root_wipes():
    print("\n[test] record_wipe with a hostile conv_id writes nothing outside wipes/")
    _wipe_storage()
    before = set()
    if os.path.exists(_CONTROL_ROOT):
        before = {os.path.join(dp, f) for dp, _, fs in os.walk(_CONTROL_ROOT) for f in fs}
    for hostile in ("../escape", "a/../../b"):
        try:
            continuity.record_wipe(hostile)
        except memory.UnsafeConvId:
            pass
    after = set()
    if os.path.exists(_CONTROL_ROOT):
        after = {os.path.join(dp, f) for dp, _, fs in os.walk(_CONTROL_ROOT) for f in fs}
    assert_eq(after, before, "no new files written anywhere under CONTROL_ROOT")
    escape_target = os.path.join(_TMP_ROOT, "escape.json")
    assert_true(not os.path.exists(escape_target), "no file escaped to the parent dir")


# ---------------------------------------------------------------------------
# Ledger round trip / epoch bump — this is what NameError broke
# ---------------------------------------------------------------------------

def test_ledger_round_trip_absent_file_reads_as_epoch_zero():
    print("\n[test] no ledger file yet -> epoch 0, boundary None, no needles")
    _wipe_storage()
    assert_eq(continuity.current_ledger_epoch("fresh-conv"), 0, "absent ledger reads as epoch 0")


def test_record_wipe_bumps_and_persists_the_epoch():
    print("\n[test] record_wipe then current_ledger_epoch — the NameError regression")
    _wipe_storage()
    cid = "wipe-conv"
    epoch1 = continuity.record_wipe(cid)
    assert_eq(epoch1, 1, "first wipe -> epoch 1")
    assert_eq(continuity.current_ledger_epoch(cid), 1, "epoch persisted and readable back")
    epoch2 = continuity.record_wipe(cid, boundary={"fp_user": "a", "fp_assistant": "b"})
    assert_eq(epoch2, 2, "second wipe -> epoch 2")
    on_disk = json.loads(continuity._ledger_path(cid).read_text(encoding="utf-8"))
    assert_eq(on_disk["boundary"], {"fp_user": "a", "fp_assistant": "b"}, "boundary persisted verbatim")


def test_record_targeted_forget_appends_a_needle():
    print("\n[test] record_targeted_forget — the other NameError regression")
    _wipe_storage()
    cid = "forget-conv"
    epoch = continuity.record_targeted_forget(cid, "Lyra")
    assert_eq(epoch, 0, "recorded at the current (pre-wipe) epoch")
    on_disk = json.loads(continuity._ledger_path(cid).read_text(encoding="utf-8"))
    assert_eq([n["needle"] for n in on_disk["targeted"]], ["Lyra"], "needle stored")


def test_retire_tombstone_bumps_epoch_sets_boundary_all_keeps_needles():
    print("\n[test] retire_tombstone — the third NameError regression")
    _wipe_storage()
    cid = "retire-conv"
    continuity.record_targeted_forget(cid, "secret")
    epoch = continuity.retire_tombstone(cid)
    assert_eq(epoch, 1, "epoch bumped")
    on_disk = json.loads(continuity._ledger_path(cid).read_text(encoding="utf-8"))
    assert_eq(on_disk["boundary"], "all", "boundary set to 'all'")
    # CHANGED in v3.1.9.10 (was: "needle list cleared"). /retire now keeps the
    # chat journal as an archive of the chat, so the needles that protect a
    # restored older copy of that journal must survive the retire.
    assert_eq([n["needle"] for n in on_disk["targeted"]], ["secret"], "needle list KEPT (the journal archive stays)")
    assert_eq(on_disk["wipe_epoch"], None, "a retire is not a wipe: no journal purge floor")
    assert_true(bool(on_disk["retired_at"]), "retire time recorded")
    assert_true(continuity._ledger_path(cid).is_file(), "ledger file itself is NOT unlinked (R10-B3)")


def test_reapply_all_ledgers_finds_files_written_via_ledger_path():
    print("\n[test] reapply_all_ledgers discovers files _ledger_path wrote — proves the path convention matches")
    _wipe_storage()
    continuity.record_wipe("conv-one")
    continuity.record_wipe("conv-two")
    result = continuity.reapply_all_ledgers()
    assert_eq(result["conversations"], 2, "both ledger files found and reapplied")


# ---------------------------------------------------------------------------
# Fail-closed behaviour on a corrupt ledger (R10-B1)
# ---------------------------------------------------------------------------

def test_corrupt_ledger_fails_closed_not_epoch_zero():
    print("\n[test] a ledger file that EXISTS but is unreadable fails closed")
    _wipe_storage()
    cid = "corrupt-conv"
    p = continuity._ledger_path(cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json", encoding="utf-8")
    assert_eq(continuity.current_ledger_epoch(cid), None, "epoch is None (unknown), never 0")
    assert_eq(continuity.visible(cid), {}, "visible() returns nothing rather than guessing")
    assert_eq(continuity.apply_needles(cid, "some text"), continuity.SUPPRESSED_MARKER,
              "apply_needles suppresses rather than risks showing an un-vetted text")


def test_record_wipe_recovers_from_a_corrupt_existing_ledger():
    print("\n[test] record_wipe does not let a corrupt ledger block the wipe she asked for")
    _wipe_storage()
    cid = "corrupt-then-wipe"
    p = continuity._ledger_path(cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json", encoding="utf-8")
    epoch = continuity.record_wipe(cid)
    assert_true(epoch >= 1, "wipe proceeds despite the corrupt file")
    assert_eq(continuity.current_ledger_epoch(cid), epoch, "ledger now readable again")


# ---------------------------------------------------------------------------
# Round-2 review, item 1 (HIGH): a ledger WRITE failure must fail the band
# closed, not just degrade some future restore.
# ---------------------------------------------------------------------------
#
# The gap the reviewer found: record_wipe/record_targeted_forget/
# retire_tombstone already caught an OSError out of `_write_ledger` (Task B
# of the ORIGINAL ask, so the deletion itself is never blocked) but did
# nothing to stop `visible()`/`apply_needles` from confidently reading the
# now-STALE ledger file — with COMPACTOR_CONTINUITY_BAND on, a needle that
# never reached disk left the forgotten words fully recoverable in the band
# on her very next request, while she had already been told the forget
# succeeded. `continuity._write_ledger_best_effort` now marks the conv in
# `_ledger_write_failed`, and `_read_ledger_failing_closed` (which every
# reader in this module funnels through) treats that exactly like an
# unreadable ledger file.
#
# `_write_ledger` is monkeypatched directly (per the task's own guidance,
# preferred over chmod'ing CONTROL_ROOT for determinism, and it works the
# same whether or not the test happens to run as root).

def _seed_visible_digest(cid: str, epoch: int = 0) -> None:
    """A single digest entry stamped at `epoch`, matching what a real band
    write looks like (continuity._write_digest_store's own shape) — used
    to prove `visible()` genuinely goes from showing something to showing
    nothing and back, not just returning {} on an already-empty store."""
    continuity._write_digest_store(cid, {
        "u1": {
            "headline": "She asked about the plan.",
            "full": "She asked about the plan — answered: it starts Tuesday.",
            "epoch": epoch,
            "unit_number": 1,
            "suppressed": False,
        },
    })


def test_ledger_write_failure_fails_the_band_closed():
    print("\n[test] a ledger WRITE failure fails visible()/apply_needles "
          "closed for that conv, same as an unreadable ledger")
    _wipe_storage()
    cid = "write-fail-band"
    _seed_visible_digest(cid, epoch=0)
    # Sanity: with no ledger file at all (absent -> epoch 0), the digest
    # above is genuinely visible before anything goes wrong.
    assert_true("u1" in continuity.visible(cid), "baseline: the digest is visible")
    assert_eq(continuity.apply_needles(cid, "arbitrary text"), "arbitrary text",
              "baseline: apply_needles passes text through untouched")

    with patch.object(continuity, "_write_ledger",
                       side_effect=OSError("no space left on device")):
        # record_targeted_forget never bumps the epoch (continuity.py's own
        # docstring: "this never BUMPS the epoch"), so this isolates the
        # WRITE failure from any epoch-mismatch side effect — the digest
        # above still nominally matches epoch 0 on disk throughout.
        epoch = continuity.record_targeted_forget(cid, "the plan")
        assert_eq(epoch, 0, "the epoch itself is still honestly reported (unbumped)")

    assert_true(continuity.ledger_degraded(cid),
                "the conv is marked degraded after the failed write")
    assert_eq(continuity.visible(cid), {},
              "BUG CHECK: visible() must fail CLOSED — showing the digest here "
              "is exactly how 'the plan' could resurface after a /forget that "
              "reported success")
    assert_eq(continuity.apply_needles(cid, "arbitrary text"), continuity.SUPPRESSED_MARKER,
              "apply_needles also fails closed for this conv")

    counts = continuity.ledger_health_counts()
    assert_true(counts["write_failed_current"] >= 1, f"counted as currently degraded: {counts}")
    assert_true(counts["write_failed_ever"] >= 1, f"counted in the ever tally: {counts}")


def test_ledger_write_recovery_clears_the_degraded_mark():
    print("\n[test] a LATER successful write clears the mark and restores normal reads")
    _wipe_storage()
    cid = "write-fail-then-recover"
    _seed_visible_digest(cid, epoch=0)

    with patch.object(continuity, "_write_ledger",
                       side_effect=OSError("no space left on device")):
        continuity.record_targeted_forget(cid, "unrelated-word")
    assert_true(continuity.ledger_degraded(cid), "degraded after the failed write")
    assert_eq(continuity.visible(cid), {}, "failing closed while degraded")

    # No patch this time: the write succeeds for real. "Tuesday" appears in
    # the seeded digest's own text (_seed_visible_digest above), so a
    # successful needle here is independently verifiable through
    # suppression, not just through ledger_degraded() flipping back.
    continuity.record_targeted_forget(cid, "Tuesday")
    assert_true(not continuity.ledger_degraded(cid),
                "the mark clears the moment a write for this conv succeeds")
    assert_true("u1" in continuity.visible(cid),
                "and the band reads normally again — the digest (still epoch 0, "
                "unbumped by a targeted forget) is visible once more")
    # The needle DID persist (the recovered write), so it suppresses the
    # digest's rendered text even though the raw entry is "visible" again.
    assert_true(continuity.visible(cid)["u1"]["suppressed"],
                "and the needle that succeeded is actually applied")

    # This process's `_ledger_write_failed` is module-level and other tests
    # in this file leave their own convs degraded on purpose (e.g. the test
    # immediately above), so the global count is not 0 here — the specific
    # claim is that THIS conv dropped out of the currently-degraded set,
    # which `ledger_degraded(cid)` above already pins directly.
    counts = continuity.ledger_health_counts()
    assert_true(cid not in continuity._ledger_write_failed,
                f"this conv is not in the currently-degraded set: {counts}")
    assert_true(counts["write_failed_ever"] >= 1,
                f"but still counted in the ever tally — recovery does not erase history: {counts}")


def test_record_wipe_write_failure_marks_degraded_and_returns_none():
    print("\n[test] record_wipe: a write failure returns None (item 4) and marks degraded")
    _wipe_storage()
    cid = "record-wipe-write-fail"
    with patch.object(continuity, "_write_ledger",
                       side_effect=OSError("no space left on device")):
        result = continuity.record_wipe(cid)
    assert_eq(result, None,
              "None, not the epoch it merely INTENDED to persist (item 4)")
    assert_true(continuity.ledger_degraded(cid), "marked degraded")
    assert_true(not continuity._ledger_path(cid).is_file(),
                "the ledger FILE itself was never actually created/bumped — "
                "_write_ledger was replaced wholesale by the patch, so its "
                "real body (which creates the file) never ran")
    # And current_ledger_epoch now fails closed for this conv too (item 1),
    # exactly like an unreadable ledger — not a stale 0 that would imply
    # 'never wiped'.
    assert_eq(continuity.current_ledger_epoch(cid), None,
              "current_ledger_epoch fails closed while this conv is degraded")


def test_retire_tombstone_write_failure_marks_degraded_and_returns_none():
    print("\n[test] retire_tombstone: a write failure returns None and marks degraded too")
    _wipe_storage()
    cid = "retire-write-fail"
    with patch.object(continuity, "_write_ledger",
                       side_effect=OSError("no space left on device")):
        result = continuity.retire_tombstone(cid)
    assert_eq(result, None, "None, consistent with record_wipe (item 4)")
    assert_true(continuity.ledger_degraded(cid), "marked degraded")


# ---------------------------------------------------------------------------
# Round-3 review: PENDING INTENTS (the "fix it fully" directive). Round 2's
# process-local "degraded" mark cleared on ANY later successful write for a
# conv, even one built from the stale on-disk ledger that never carried the
# failed change — round-3's reviewer probe (scratchpad/expC/compactor/
# probe_ledger.py) demonstrated a lost needle surviving an unrelated
# successful forget (A1), a restart (A2), and a lost wipe epoch surviving a
# later unrelated forget (A3). These tests reproduce all three directly
# against continuity.py (probe_ledger.py itself is reproduced, unmodified,
# against the built image as part of the gate's own coverage).
# ---------------------------------------------------------------------------

def test_pending_intent_survives_an_unrelated_successful_forget():
    print("\n[test] round-3 A1 — a lost needle survives a later, DIFFERENT successful forget")
    _wipe_storage()
    cid = "pending-a1"
    continuity._write_digest_store(cid, {
        "u1": {
            "headline": "she mentioned her sister Maren",
            "full": "she told me about her sister Maren's diagnosis",
            "epoch": 0, "unit_number": 1, "suppressed": False,
        },
    })
    with patch.object(continuity, "_write_ledger",
                       side_effect=OSError("ro")):
        continuity.suppress_needle(cid, "Maren")
    assert_true(continuity.ledger_degraded(cid), "degraded after the failed forget")
    assert_eq(continuity.visible(cid), {}, "failing closed while degraded")

    # A DIFFERENT, unrelated forget succeeds for real.
    continuity.suppress_needle(cid, "pizza")

    assert_true(not continuity.ledger_degraded(cid), "recovered — the write landed")
    visible = continuity.visible(cid)
    assert_true("u1" in visible, "the band reads normally again")
    assert_true(
        visible["u1"]["suppressed"],
        "*** A1 BUG CHECK: Maren's needle — lost in the failed write — was "
        "carried forward by the pending intent into the unrelated 'pizza' "
        "forget's successful write, and still suppresses her words. If this "
        "fails, the needle was silently dropped and her words are visible "
        "again after an unrelated /forget.",
    )
    on_disk = continuity._read_ledger(cid)
    assert_eq(
        sorted(n["needle"] for n in on_disk["targeted"]), ["Maren", "pizza"],
        "both needles are actually persisted on disk, not just applied in memory",
    )


def test_pending_intent_survives_a_simulated_restart_via_sidecar():
    print("\n[test] round-3 A2 — a lost needle survives a restart, via the durability sidecar")
    _wipe_storage()
    cid = "pending-a2"
    continuity._write_digest_store(cid, {
        "u1": {
            "headline": "she mentioned her sister Maren",
            "full": "she told me about her sister Maren's diagnosis",
            "epoch": 0, "unit_number": 1, "suppressed": False,
        },
    })
    with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")):
        continuity.suppress_needle(cid, "Maren")
    assert_true(continuity._pending_sidecar_path(cid).is_file(),
                "the pending intent was saved to a durability sidecar")

    # Simulate a process restart: everything in-memory is gone. Only the
    # sidecar (a real file on disk) and the (still-stale) ledger file
    # survive. A fresh process would start these empty; nothing here except
    # `_pending_intents`/`_double_failure_current` needs clearing, because
    # nothing else is supposed to hold process state across a restart.
    continuity._ledger_write_failed.clear()
    continuity._pending_intents.clear()
    continuity._double_failure_current.clear()

    # Even before startup recovery runs, the band must still fail closed —
    # the sidecar FILE alone is enough (_has_pending_intent checks disk).
    assert_true(continuity.ledger_degraded(cid),
                "still degraded after 'restart', purely from the sidecar file")
    assert_eq(continuity.visible(cid), {}, "still failing closed before reapply_all_ledgers runs")

    # Startup recovery: reads the sidecar back, retries the write for real.
    result = continuity.reapply_all_ledgers()
    assert_true(result.get("pending_recovered", 0) >= 1,
                f"the sidecar's intent was recovered at startup (got {result})")
    assert_true(not continuity.ledger_degraded(cid), "recovered after startup reapply")
    assert_true(not continuity._pending_sidecar_path(cid).is_file(),
                "the sidecar is deleted once its content actually landed")
    visible = continuity.visible(cid)
    assert_true(
        visible.get("u1", {}).get("suppressed"),
        "*** A2 BUG CHECK: Maren's needle survived the simulated restart "
        "and is applied once startup recovery merges the sidecar in",
    )


def test_pending_wipe_epoch_survives_an_unrelated_successful_forget():
    print("\n[test] round-3 A3 — a lost wipe's epoch survives a later, DIFFERENT successful forget")
    _wipe_storage()
    cid = "pending-a3"
    # A pre-wipe digest, exactly the shape a stale in-flight write (or a
    # restore) could leave behind at epoch 0.
    continuity._write_digest_store(cid, {
        "u1": {"headline": "old pre-wipe", "full": "old pre-wipe content",
               "epoch": 0, "unit_number": 1, "suppressed": False},
    })
    with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")):
        result = continuity.record_wipe(cid)
    assert_eq(result, None, "record_wipe honestly reports the write failed")
    assert_true(continuity.ledger_degraded(cid), "degraded after the failed wipe")
    assert_eq(continuity.current_ledger_epoch(cid), None, "epoch read fails closed too")

    # A stale/in-flight write lands ANOTHER epoch-0 digest (as the reviewer's
    # probe does — modelling a write that was already in flight when the
    # wipe was requested), and then a completely unrelated targeted forget
    # succeeds for real.
    continuity._write_digest_store(cid, {
        "u1": {"headline": "old pre-wipe", "full": "old pre-wipe content",
               "epoch": 0, "unit_number": 1, "suppressed": False},
    })
    continuity.suppress_needle(cid, "zzz")

    assert_true(not continuity.ledger_degraded(cid), "recovered — the write landed")
    epoch = continuity.current_ledger_epoch(cid)
    assert_eq(epoch, 1, "*** A3 BUG CHECK: the wipe's epoch bump (lost in the "
              "failed write) was carried forward by the pending intent into "
              "the unrelated forget's successful write")
    assert_eq(
        list(continuity.visible(cid)), [],
        "*** A3 BUG CHECK: the pre-wipe digest (stamped epoch 0) is hidden "
        "— the wipe's epoch bump to 1 actually took effect, not silently "
        "lost",
    )


def test_double_failure_residual_is_logged_and_shown_at_health():
    print("\n[test] round-3 — a DOUBLE failure (ledger AND sidecar) is logged "
          "CRITICAL and reported at /health")
    _wipe_storage()
    cid = "double-failure"
    handler = _CaptureLogs()
    lg = logging.getLogger("compactor.continuity")
    lg.addHandler(handler)
    try:
        with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")), \
             patch.object(continuity, "_persist_pending_sidecar", return_value=False):
            continuity.suppress_needle(cid, "Maren")
    finally:
        lg.removeHandler(handler)

    assert_true(continuity.ledger_degraded(cid), "still degraded (band fails closed)")
    assert_true(continuity.ledger_double_failure(cid),
                "*** the double-failure residual is flagged — the ONE case "
                "where a restart would actually lose the pending intent")
    assert_true(
        any(r.levelno == logging.CRITICAL and "DOUBLE FAILURE" in r.getMessage() for r in handler.records),
        f"a CRITICAL log line names the double failure explicitly (got "
        f"{[r.getMessage() for r in handler.records]})",
    )
    counts = continuity.ledger_health_counts()
    assert_true(cid not in str({k: v for k, v in counts.items()}),
                "no conv id leaks into the health counts themselves")
    assert_true(counts["double_failure_current"] >= 1,
                f"*** double_failure_current reports it at /health (got {counts})")
    assert_true(counts["double_failure_ever"] >= 1,
                f"*** double_failure_ever reports it too (got {counts})")

    # Recovery: a later write succeeds for real (sidecar included) — the
    # residual clears, same as any other pending intent.
    continuity.suppress_needle(cid, "pizza")
    assert_true(not continuity.ledger_double_failure(cid), "residual clears once a write actually lands")
    assert_true(not continuity.ledger_degraded(cid), "and the conv is fully recovered")


def test_restoring_an_old_pending_sidecar_only_hides_more():
    print("\n[test] round-3 — restoring a STALE pending sidecar (e.g. from an "
          "old backup) can only hide MORE, never less")
    # BACKUP INTERACTION (round-3 design): STORAGE_ROOT (which now includes
    # ledger-pending/) is backed up. A restore from an OLD backup can bring
    # back a sidecar for a pending intent that was ALREADY merged and
    # deleted since. Simulated here by writing a sidecar directly (as if
    # restored) for a conv whose ledger ALREADY has different, unrelated
    # content on disk — the merge must only ADD the sidecar's needle and
    # RAISE the epoch if higher, never remove or lower anything already
    # there, and never reveal a digest that current state hides.
    _wipe_storage()
    cid = "stale-sidecar-restore"
    # Current, healthy ledger state: epoch 2 (two wipes happened since),
    # one needle ("current-needle") already recorded.
    continuity._write_ledger(cid, {
        "epoch": 2, "boundary": None,
        "targeted": [{"needle": "current-needle", "epoch": 0}],
    })
    # A digest from AFTER both wipes (epoch 2) — visible today.
    continuity._write_digest_store(cid, {
        "u1": {"headline": "post-wipe entry", "full": "post-wipe entry text",
               "epoch": 2, "unit_number": 1, "suppressed": False},
    })
    before = continuity.visible(cid)
    assert_true("u1" in before and not before["u1"]["suppressed"],
                "fixture: the post-wipe digest is visible and unsuppressed before the restore")

    # "Restore" a STALE sidecar: an OLDER pending intent (epoch 1 — already
    # superseded by the real epoch-2 ledger) carrying a needle that WOULD
    # match the current digest's text ("post-wipe").
    stale_pending = {
        "epoch": 1, "boundary": None,
        # CHANGED in v3.1.9.10 (was epoch 0). Point-in-time: a needle recorded
        # under epoch 0 no longer reaches a digest of epoch 2 (it was written
        # after two later wipes, so it post-dates that forget; see
        # test_a_later_epoch_postdates_an_earlier_needle). The property this
        # test pins, "a stale sidecar can only hide MORE", is kept by giving
        # the stale needle the digest's own epoch (no cutoff = a .8 needle).
        "needles": [{"needle": "post-wipe", "epoch": 2}],
    }
    continuity._persist_pending_sidecar(cid, stale_pending)
    continuity._pending_intents[cid] = dict(stale_pending)

    result = continuity.reapply_all_ledgers()
    assert_true(result.get("pending_recovered", 0) >= 1, f"the stale sidecar was processed (got {result})")

    on_disk = continuity._read_ledger(cid)
    assert_eq(on_disk["epoch"], 2,
              "*** the epoch is NOT lowered to the stale sidecar's 1 — max, never a downgrade")
    assert_eq(
        sorted(n["needle"] for n in on_disk["targeted"]),
        ["current-needle", "post-wipe"],
        "*** the stale sidecar's needle is ADDED (union), the existing one is KEPT",
    )
    after = continuity.visible(cid)
    assert_true(
        "u1" not in after or after["u1"]["suppressed"],
        "*** the merge can only hide MORE: the post-wipe digest, visible "
        "before the stale restore, is now suppressed (needle-matched) or "
        "absent — never REVEALED something that was hidden, and the only "
        "change is additional suppression",
    )
    assert_true(not continuity._pending_sidecar_path(cid).is_file(),
                "the stale sidecar is cleaned up once merged")


# ---------------------------------------------------------------------------
# Round-4 review: item 1 (HIGH) — the merge must also read the ON-DISK
# sidecar, not memory alone, and reapply_all_ledgers must be robust.
# probes N1/N1b/N4 (scratchpad/r3/p1.py).
# ---------------------------------------------------------------------------

def test_sidecar_only_pending_survives_an_unrelated_successful_write():
    print("\n[test] round-4 N1 — a sidecar that exists on disk but was NEVER "
          "loaded into memory still survives an unrelated successful write")
    # The gap round 3 missed: a failed write persists a sidecar AND records
    # the pending intent in memory. If something clears the IN-MEMORY copy
    # without ever loading the sidecar back in (a startup read that failed
    # silently, or — as here — simulated directly) the sidecar is now
    # invisible to `_pending_intents`. Before this fix, the NEXT successful
    # write for this conv only merged memory (empty) and then deleted the
    # sidecar as if it were satisfied — losing the needle it held.
    _wipe_storage()
    cid = "n1-sidecar-only"
    continuity._write_digest_store(cid, {
        "u1": {"headline": "she mentioned Maren", "full": "her sister Maren's diagnosis",
               "epoch": 0, "unit_number": 1, "suppressed": False},
    })
    with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")):
        continuity.suppress_needle(cid, "Maren")
    assert_true(continuity._pending_sidecar_path(cid).is_file(), "sidecar written")
    # Drop ONLY the in-memory copy — simulating a sidecar that was written
    # by a failed attempt but never (re)loaded into `_pending_intents`,
    # e.g. because a startup read of it failed transiently (N1b) or this
    # is a fresh process that has not run reapply_all_ledgers yet.
    continuity._pending_intents.pop(cid, None)
    assert_true(cid not in continuity._pending_intents,
                "fixture: the sidecar is NOT in memory, only on disk")
    assert_true(continuity.ledger_degraded(cid),
                "still degraded — _has_pending_intent finds the sidecar file directly")

    # A completely different, unrelated forget succeeds for real.
    continuity.suppress_needle(cid, "pizza")

    assert_true(not continuity._pending_sidecar_path(cid).is_file(),
                "the sidecar is gone now — its content was actually merged")
    on_disk = continuity._read_ledger(cid)
    assert_eq(
        sorted(n["needle"] for n in on_disk["targeted"]), ["Maren", "pizza"],
        "*** N1 BUG CHECK: Maren's needle — pending ONLY on disk, never in "
        "memory — was merged into the unrelated write, not silently "
        "dropped when its sidecar was deleted",
    )
    assert_true(not continuity.ledger_degraded(cid), "fully recovered")
    # CHANGED in v3.1.9.10 (was: apply_needles(...) == SUPPRESSED_MARKER).
    # A fallback text has no age, so a needle no longer suppresses it (that
    # was the "never remembered again" behaviour). The needle is still applied
    # to what PRE-DATES it: the digest entry, which has no `created`.
    assert_eq(continuity.apply_needles(cid, "her sister Maren's diagnosis"),
              "her sister Maren's diagnosis",
              "apply_needles passes a fallback text through (fail-closed only)")
    assert_true(continuity.visible(cid)["u1"]["suppressed"],
                "and the needle is applied to the pre-dating digest entry")


def test_transiently_unreadable_sidecar_at_startup_still_recovers_later():
    print("\n[test] round-4 N1b — a sidecar unreadable during ONE startup "
          "attempt is still recovered by a later, successful write")
    _wipe_storage()
    cid = "n1b-flaky-startup"
    continuity._write_digest_store(cid, {
        "u1": {"headline": "she mentioned Maren", "full": "her sister Maren's diagnosis",
               "epoch": 0, "unit_number": 1, "suppressed": False},
    })
    with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")):
        continuity.suppress_needle(cid, "Maren")
    continuity._pending_intents.clear()  # simulate restart

    # reapply_all_ledgers runs while the sidecar is transiently unreadable
    # (a real EIO, not a permanently corrupt file) — it must log and move
    # on, not crash, and must NOT delete the sidecar it could not read.
    real_read_text = __import__("pathlib").Path.read_text

    def flaky(self, *a, **kw):
        if "ledger-pending" in str(self):
            raise OSError(5, "EIO")
        return real_read_text(self, *a, **kw)

    with patch("pathlib.Path.read_text", flaky):
        result = continuity.reapply_all_ledgers()
    assert_eq(result.get("pending_recovered", 0), 0, "nothing recovered THIS attempt (flaky read)")
    assert_true(continuity._pending_sidecar_path(cid).is_file(),
                "the sidecar was NOT deleted just because it could not be read this time")
    assert_true(continuity.ledger_degraded(cid), "still degraded")

    # The flakiness is gone; a later unrelated forget succeeds for real and
    # merges the sidecar (read fresh — round-4's fix — not from whatever
    # reapply_all_ledgers saw during the flaky window).
    continuity.suppress_needle(cid, "pizza")
    assert_true(not continuity.ledger_degraded(cid), "recovered once the read succeeds")
    on_disk = continuity._read_ledger(cid)
    assert_eq(
        sorted(n["needle"] for n in on_disk["targeted"]), ["Maren", "pizza"],
        "*** N1b BUG CHECK: Maren's needle survived the transient read failure",
    )


def test_permanently_unreadable_sidecar_is_never_deleted_and_keeps_band_closed():
    print("\n[test] round-4 — a PERMANENTLY corrupt sidecar is never deleted, "
          "even by an otherwise-successful write, and keeps the band closed")
    _wipe_storage()
    cid = "corrupt-sidecar"
    continuity._write_digest_store(cid, {
        "u1": {"headline": "post-write entry", "full": "post-write entry text",
               "epoch": 0, "unit_number": 1, "suppressed": False},
    })
    # A genuinely corrupt sidecar — not a transient read error.
    p = continuity._pending_sidecar_path(cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json", encoding="utf-8")
    assert_true(continuity.ledger_degraded(cid), "degraded — the sidecar's mere presence fails closed")
    assert_eq(continuity.visible(cid), {}, "band fails closed while the sidecar is unreadable")

    # An UNRELATED write succeeds (the ledger itself is healthy).
    continuity.suppress_needle(cid, "something-else")

    assert_true(
        p.is_file(),
        "*** BUG CHECK: the corrupt sidecar is NEVER deleted just because "
        "a write for this conv happened to succeed — its content is "
        "unknown and might not be what just landed",
    )
    assert_true(
        continuity.ledger_degraded(cid),
        "*** BUG CHECK: the conv stays marked degraded — the band keeps "
        "failing closed until the sidecar is repaired or superseded",
    )
    assert_eq(continuity.visible(cid), {}, "band still fails closed after the 'successful' write")


def test_reapply_all_ledgers_survives_a_wipes_dir_scan_failure():
    print("\n[test] round-4 N4 — a fault in the wipes-dir scan does not skip "
          "the pending-sidecar recovery loop")
    _wipe_storage()
    cid = "n4-wipes-scan-fault"
    with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")):
        continuity.record_wipe(cid)
    assert_true(continuity._pending_sidecar_path(cid).is_file(), "sidecar written")
    continuity._pending_intents.clear()  # simulate restart

    real_is_dir = __import__("pathlib").Path.is_dir

    def bad(self):
        if str(self).endswith("wipes"):
            raise PermissionError(13, "eacces")
        return real_is_dir(self)

    with patch("pathlib.Path.is_dir", bad):
        result = continuity.reapply_all_ledgers()
    assert_true(
        result.get("pending_recovered", 0) >= 1,
        f"*** N4 BUG CHECK: the pending-sidecar loop still ran and "
        f"recovered the conv even though the (unrelated) wipes-dir scan "
        f"raised (got {result})",
    )
    assert_true(not continuity.ledger_degraded(cid), "fully recovered despite the wipes-dir fault")


# ---------------------------------------------------------------------------
# Round-4 review: item 4 (MEDIUM) — _has_pending_intent must fail CLOSED
# (True) on an unexpected stat error, not silently fail open. probe N3.
# ---------------------------------------------------------------------------

def test_has_pending_intent_fails_closed_on_a_stat_error():
    print("\n[test] round-4 N3 — _has_pending_intent fails CLOSED (True) on "
          "an unexpected stat error")
    _wipe_storage()
    cid = "n3-stat-error"
    with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")):
        continuity.suppress_needle(cid, "Maren")
    continuity._pending_intents.clear()  # sidecar-only, same shape as N1

    real_is_file = __import__("pathlib").Path.is_file

    def boom(self, *a, **kw):
        if "ledger-pending" in str(self):
            raise PermissionError(13, "eacces")
        return real_is_file(self, *a, **kw)

    with patch("pathlib.Path.is_file", boom):
        assert_true(
            continuity._has_pending_intent(cid),
            "*** N3 BUG CHECK: an unexpected stat error must be treated as "
            "'unknown, assume pending' (fail CLOSED), not 'absent' (fail "
            "open) — the old version returned False here",
        )


def test_has_pending_intent_returns_false_only_for_a_hostile_conv_id():
    print("\n[test] round-4 — the ONE case _has_pending_intent may honestly "
          "answer False from an exception is a genuinely hostile conv_id")
    assert_true(
        not continuity._has_pending_intent("../escape"),
        "a hostile conv_id cannot have a sidecar at all — False is honest here",
    )


# ---------------------------------------------------------------------------
# Round-4 review: item 6 (LOW) — on an equal epoch, keep the BROADER
# boundary; /health's pending count must include sidecar-only entries.
# ---------------------------------------------------------------------------

def test_equal_epoch_merge_keeps_the_broader_all_boundary():
    print("\n[test] round-4 — merging two failed writes at the SAME epoch "
          "keeps the broader ('all') boundary, whichever merges first")
    _wipe_storage()
    cid = "equal-epoch-boundary"
    # A failed WIPE with a partial (turn-fingerprint) boundary lands first.
    with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")):
        continuity.record_wipe(cid, boundary={"fp_user": "a", "fp_assistant": "b"})
    pending = continuity._pending_intents[cid]
    assert_eq(pending["epoch"], 1, "fixture: the wipe wanted epoch 1")
    assert_eq(pending["boundary"], {"fp_user": "a", "fp_assistant": "b"}, "fixture: partial boundary")

    # A SECOND failed op — a retire — targets the SAME epoch (simulated
    # directly via _record_pending_intent, since retire_tombstone's own
    # fresh read would normally see epoch 0 on disk and compute epoch 1
    # itself; this pins the merge rule in isolation regardless of how the
    # two epochs came to coincide).
    continuity._record_pending_intent(cid, {"epoch": 1, "boundary": "all", "targeted": []})

    merged = continuity._pending_intents[cid]
    assert_eq(merged["epoch"], 1, "epoch unchanged (still the max, not raised further)")
    assert_eq(
        merged["boundary"], "all",
        "*** the broader boundary ('all') wins on a tied epoch, not "
        "whichever attempt merged first",
    )

    # And the reverse order — "all" first, then a partial boundary at the
    # same epoch — must NOT narrow it back down.
    cid2 = "equal-epoch-boundary-reverse"
    continuity._record_pending_intent(cid2, {"epoch": 3, "boundary": "all", "targeted": []})
    continuity._record_pending_intent(cid2, {"epoch": 3, "boundary": {"fp_user": "x"}, "targeted": []})
    assert_eq(
        continuity._pending_intents[cid2]["boundary"], "all",
        "*** order independence: 'all' still wins even when it arrived FIRST",
    )


def test_health_pending_count_includes_sidecar_only_entries():
    print("\n[test] round-4 — ledger_health_counts' pending_intent_current "
          "counts a sidecar-only pending intent, not just in-memory ones")
    _wipe_storage()
    cid = "sidecar-only-health"
    with patch.object(continuity, "_write_ledger", side_effect=OSError("ro")):
        continuity.suppress_needle(cid, "Maren")
    continuity._pending_intents.pop(cid, None)  # sidecar-only now
    assert_true(cid not in continuity._pending_intents, "fixture: not in memory")
    assert_true(continuity._pending_sidecar_path(cid).is_file(), "fixture: sidecar present")

    counts = continuity.ledger_health_counts()
    assert_true(
        counts["pending_intent_current"] >= 1,
        f"*** the sidecar-only pending intent is counted at /health, not "
        f"undercounted by a bare len(_pending_intents) (got {counts})",
    )


# ---------------------------------------------------------------------------
# _write_ledger creates the parent dir if missing
# ---------------------------------------------------------------------------

def test_write_ledger_creates_missing_parent_dir():
    print("\n[test] _write_ledger creates CONTROL_ROOT/wipes/ if it does not exist yet")
    _wipe_storage()
    assert_true(not (continuity.CONTROL_ROOT / "wipes").exists(), "wipes/ absent before any write")
    continuity.record_wipe("first-ever-wipe")
    assert_true((continuity.CONTROL_ROOT / "wipes").is_dir(), "wipes/ created on demand")


# ---------------------------------------------------------------------------
# portability.quarantine_conversation must never carry the digest cache
# (portability.py:329 names this test by name)
# ---------------------------------------------------------------------------

def test_format_band_numbers_entries_exactly_once():
    print("\n[test] format_band numbers each entry once — no '1. 1. …' double-numbering")
    _wipe_storage()
    # A fallback-shaped entry, built the way main.py's band-injection call
    # site builds one: extractive_fallback's own text, unnumbered (it no
    # longer prefixes "N. " itself — format_band is the one place that
    # numbers). Round 2 fix: extractive_fallback USED to also number,
    # which produced "1. 1. she said something — answered: ...".
    fallback_text = continuity.extractive_fallback(None, "She asked about the plan.", "", 1)
    assert_true(not fallback_text.startswith("1. "),
                f"extractive_fallback no longer numbers its own output: {fallback_text!r}")
    entry = {"headline": fallback_text, "full": fallback_text, "suppressed": False}
    rendered = continuity.format_band([entry])
    line = rendered.splitlines()[1]
    assert_true(line.startswith("1. ") and not line.startswith("1. 1. "),
                f"exactly one '1. ' prefix, not two: {line!r}")


def test_quarantine_excludes_digests():
    print("\n[test] a quarantine snapshot never carries the digest cache")
    _wipe_storage()
    cid = "quarantine-digest-conv"
    memory.atomic_write_json(
        memory.digests_path(cid),
        {"u1": {"headline": "h", "full": "f", "epoch": 0, "unit_number": 1, "suppressed": False}},
    )
    result = portability.quarantine_conversation(cid, reason="test")
    bundle = json.loads(open(result["path"], encoding="utf-8").read())
    assert_true("digests" not in bundle, "no top-level 'digests' key")
    assert_true("digests" not in bundle.get("quarantine", {}), "no 'digests' key under quarantine either")


# ---------------------------------------------------------------------------
# v3.1.9.10: /forget is POINT-IN-TIME. The ledger records a cutoff per entry
# and the boot re-apply removes only what pre-dates it.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import datetime  # noqa: E402


def _iso(delta_s):
    d = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=delta_s)
    return d.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def test_targeted_forget_records_a_cutoff():
    print("\n[test] v3.1.9.10: a targeted forget records its own UTC cutoff")
    _wipe_storage()
    cid = "pit-cutoff"
    continuity.record_targeted_forget(cid, "Lyra")
    led = json.loads(continuity._ledger_path(cid).read_text(encoding="utf-8"))
    assert_eq(led["format"], 2, "ledger format 2")
    n = led["targeted"][0]
    assert_true(n["cutoff"].endswith("Z") and "T" in n["cutoff"], "cutoff is a UTC ISO timestamp")
    assert_eq(n["epoch"], 0, "and the epoch it was recorded at")


def test_reapply_removes_only_what_predates_the_forget():
    print("\n[test] v3.1.9.10: boot re-apply suppresses the OLD digest, keeps the NEW one")
    _wipe_storage()
    cid = "pit-digest"
    continuity._write_digest_store(cid, {
        "old": {"headline": "about Maren", "full": "Maren said hello", "epoch": 0, "unit_number": 1,
                "suppressed": False, "created": _iso(-3600)},
        "legacy": {"headline": "about Maren", "full": "Maren, no created stamp", "epoch": 0,
                   "unit_number": 2, "suppressed": False},
    })
    continuity.record_targeted_forget(cid, "Maren")
    # written AFTER the forget, repeating the forgotten word
    continuity._write_digest_store(cid, {
        **continuity._read_digest_store(cid),
        "new": {"headline": "Maren again", "full": "Maren again, said later", "epoch": 0,
                "unit_number": 3, "suppressed": False, "created": _iso(+60)},
    })
    continuity.reapply_all_ledgers()   # a reboot
    v = continuity.visible(cid)
    assert_true(v["old"]["suppressed"], "a pre-forget digest is suppressed")
    assert_true(v["legacy"]["suppressed"], "a digest with no `created` (age unknown) counts as pre-dating")
    assert_true(not v["new"]["suppressed"], "a post-forget digest with the same word is KEPT")
    assert_eq(v["new"]["full"], "Maren again, said later", "and its text is intact")
    continuity.reapply_all_ledgers()   # idempotent
    assert_true(not continuity.visible(cid)["new"]["suppressed"], "still kept after a second boot")


def test_generate_digest_has_no_write_time_needle_filter():
    print("\n[test] v3.1.9.10: a digest written after the forget is remembered normally")
    _wipe_storage()
    cid = "pit-gen"
    continuity._write_digest_store(cid, {
        "k1": {"headline": SUPP, "full": SUPP, "epoch": 0, "unit_number": 1,
               "suppressed": True, "created": _iso(-60)},
    })
    continuity.record_targeted_forget(cid, "Maren")
    ok = asyncio.run(continuity.generate_digest(
        cid, tok=None, her_text="I want to talk about Maren again today.",
        reply_text="Of course, Maren sounds important to you and we can talk about her.",
        unit_number=1, key="k1", epoch_at_submit=0, vllm_url=None, model=None))
    assert_true(ok, "the digest is written (returns True), not suppressed")
    e = continuity._read_digest_store(cid)["k1"]
    assert_true(not e["suppressed"] and "Maren" in e["full"], "stored with the word in it")
    assert_true(bool(e.get("created")), "and stamped with a creation time")
    continuity.reapply_all_ledgers()
    assert_true(not continuity.visible(cid)["k1"]["suppressed"], "it survives a reboot")


SUPP = continuity.SUPPRESSED_MARKER


def test_a_later_epoch_postdates_an_earlier_needle():
    print("\n[test] v3.1.9.10: an item of a later epoch post-dates an earlier needle")
    n = {"needle": "x", "epoch": 1, "cutoff": _iso(0)}
    assert_true(continuity.pre_dates(_iso(-10), 1, n), "same epoch, earlier ts: pre-dates")
    assert_true(not continuity.pre_dates(_iso(-10), 2, n), "a later epoch: post-dates whatever its ts")
    assert_true(not continuity.pre_dates(_iso(+10), 1, n), "same epoch, later ts: post-dates")
    assert_true(continuity.pre_dates(None, None, n), "unknown age: pre-dates (safe)")
    assert_true(continuity.pre_dates("garbage", 1, n), "unparseable age: pre-dates (safe)")
    assert_true(continuity.pre_dates(_iso(+10), 1, {"needle": "x", "epoch": 1}), "no cutoff (.8): everything pre-dates")


def test_full_wipe_records_a_cutoff_and_a_post_wipe_digest_survives_reboot():
    print("\n[test] v3.1.9.10: full wipe = point-in-time too")
    _wipe_storage()
    cid = "pit-wipe"
    continuity._write_digest_store(cid, {
        "pre": {"headline": "a", "full": "b", "epoch": 0, "unit_number": 1, "suppressed": False},
    })
    epoch = continuity.record_wipe(cid)
    led = json.loads(continuity._ledger_path(cid).read_text(encoding="utf-8"))
    assert_eq((led["wipe_epoch"], bool(led["wipe_at"])), (epoch, True), "wipe_epoch and wipe_at recorded")
    # the wipe deletes the store; a RESTORED pre-wipe store comes back with epoch 0
    continuity._write_digest_store(cid, {
        "pre": {"headline": "a", "full": "b", "epoch": 0, "unit_number": 1, "suppressed": False},
        "post": {"headline": "c", "full": "d", "epoch": epoch, "unit_number": 2, "suppressed": False,
                 "created": _iso(+5)},
    })
    continuity.reapply_all_ledgers()
    v = continuity.visible(cid)
    assert_eq(sorted(v), ["post"], "the restored pre-wipe digest is purged; the post-wipe one survives the reboot")


def test_legacy_v8_ledger_is_migrated_safely():
    print("\n[test] v3.1.9.10: a .8-format ledger (no cutoff, no format) migrates on boot")
    _wipe_storage()
    cid = "legacy-8"
    p = continuity._ledger_path(cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    # exactly the .8 shape
    p.write_text(json.dumps({"epoch": 2, "boundary": None,
                             "targeted": [{"needle": "Maren", "epoch": 2}]}), encoding="utf-8")
    continuity._write_digest_store(cid, {
        "old": {"headline": "Maren", "full": "Maren", "epoch": 2, "unit_number": 1, "suppressed": False,
                "created": _iso(-3600)},
        "older_epoch": {"headline": "x", "full": "y", "epoch": 1, "unit_number": 2, "suppressed": False},
    })
    # before the boot stamp: a cutoff-less needle reads as "+infinity" (.8 behaviour)
    led0 = continuity._read_ledger(cid)
    assert_eq(led0["targeted"][0].get("cutoff"), None, "read in memory: still no cutoff before the stamp")
    assert_eq(led0["wipe_epoch"], 2, "journal purge floor derived from the .8 epoch")
    before = _iso(0)
    res = continuity.reapply_all_ledgers()
    after = _iso(0)
    led = json.loads(p.read_text(encoding="utf-8"))
    assert_eq(led["format"], 2, "rewritten as format 2")
    n = led["targeted"][0]
    assert_true(n.get("legacy") is True and before <= n["cutoff"] <= after,
                f"the needle is stamped with the BOOT time and marked legacy: {n}")
    v = continuity.visible(cid)
    assert_true("older_epoch" not in v, "an older-epoch digest is purged (as in .8)")
    assert_true(v["old"]["suppressed"], "the pre-boot matching digest is suppressed (what .8 promised)")
    # ... and from now on it is point-in-time: a digest made after the stamp is kept
    continuity._write_digest_store(cid, {**continuity._read_digest_store(cid), "new": {
        "headline": "Maren", "full": "Maren again", "epoch": 2, "unit_number": 3, "suppressed": False,
        "created": _iso(+30)}})
    continuity.reapply_all_ledgers()
    assert_true(not continuity.visible(cid)["new"]["suppressed"], "post-migration content is remembered normally")
    # a retired .8 ledger (boundary "all") derives NO purge floor: retire keeps the journal
    cid2 = "legacy-8-retired"
    p2 = continuity._ledger_path(cid2)
    p2.write_text(json.dumps({"epoch": 3, "boundary": "all", "targeted": []}), encoding="utf-8")
    assert_eq(continuity._read_ledger(cid2)["wipe_epoch"], None, "a .8 retire tombstone does not purge the journal")


def test_legacy_ledger_stamp_write_failure_never_closes_the_band():
    print("\n[test] v3.1.9.10: a failed migration stamp write is logged, retried next boot, no pending intent")
    _wipe_storage()
    cid = "legacy-fail"
    p = continuity._ledger_path(cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"epoch": 0, "boundary": None, "targeted": [{"needle": "Maren", "epoch": 0}]}),
                 encoding="utf-8")
    with patch.object(continuity, "_write_ledger", side_effect=OSError(28, "full")):
        continuity.reapply_ledger(cid)
    assert_true(not continuity.ledger_degraded(cid), "no pending intent was raised for a stamp failure")
    assert_true(continuity._read_ledger(cid)["targeted"][0].get("cutoff"),
                "the in-process stamp keeps it point-in-time until the next boot")
    continuity._legacy_stamp.clear()


if __name__ == "__main__":
    try:
        test_ledger_path_matches_the_documented_convention()
        test_ledger_path_rejects_traversal_conv_id()
        test_ledger_path_never_writes_outside_control_root_wipes()

        test_ledger_round_trip_absent_file_reads_as_epoch_zero()
        test_record_wipe_bumps_and_persists_the_epoch()
        test_record_targeted_forget_appends_a_needle()
        test_retire_tombstone_bumps_epoch_sets_boundary_all_keeps_needles()
        test_reapply_all_ledgers_finds_files_written_via_ledger_path()

        test_corrupt_ledger_fails_closed_not_epoch_zero()
        test_record_wipe_recovers_from_a_corrupt_existing_ledger()

        test_ledger_write_failure_fails_the_band_closed()
        test_ledger_write_recovery_clears_the_degraded_mark()
        test_record_wipe_write_failure_marks_degraded_and_returns_none()
        test_retire_tombstone_write_failure_marks_degraded_and_returns_none()

        test_pending_intent_survives_an_unrelated_successful_forget()
        test_pending_intent_survives_a_simulated_restart_via_sidecar()
        test_pending_wipe_epoch_survives_an_unrelated_successful_forget()
        test_double_failure_residual_is_logged_and_shown_at_health()
        test_restoring_an_old_pending_sidecar_only_hides_more()

        test_sidecar_only_pending_survives_an_unrelated_successful_write()
        test_transiently_unreadable_sidecar_at_startup_still_recovers_later()
        test_permanently_unreadable_sidecar_is_never_deleted_and_keeps_band_closed()
        test_reapply_all_ledgers_survives_a_wipes_dir_scan_failure()
        test_has_pending_intent_fails_closed_on_a_stat_error()
        test_has_pending_intent_returns_false_only_for_a_hostile_conv_id()
        test_equal_epoch_merge_keeps_the_broader_all_boundary()
        test_health_pending_count_includes_sidecar_only_entries()

        test_write_ledger_creates_missing_parent_dir()

        test_format_band_numbers_entries_exactly_once()
        test_quarantine_excludes_digests()

        test_targeted_forget_records_a_cutoff()
        test_reapply_removes_only_what_predates_the_forget()
        test_generate_digest_has_no_write_time_needle_filter()
        test_a_later_epoch_postdates_an_earlier_needle()
        test_full_wipe_records_a_cutoff_and_a_post_wipe_digest_survives_reboot()
        test_legacy_v8_ledger_is_migrated_safely()
        test_legacy_ledger_stamp_write_failure_never_closes_the_band()

        print("\nAll continuity tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
