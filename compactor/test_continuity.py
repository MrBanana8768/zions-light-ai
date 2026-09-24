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
import os
import shutil
import sys
import tempfile

# Storage/control redirects MUST happen before importing memory/continuity
# so the module-level paths see the override.
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-continuity-")
_STORAGE_ROOT = os.path.join(_TMP_ROOT, "storage")
_CONTROL_ROOT = os.path.join(_TMP_ROOT, "control")
os.environ["COMPACTOR_STORAGE_ROOT"] = _STORAGE_ROOT
os.environ["COMPACTOR_CONTROL_ROOT"] = _CONTROL_ROOT

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


def _wipe_storage():
    if os.path.exists(_STORAGE_ROOT):
        shutil.rmtree(_STORAGE_ROOT)
    if os.path.exists(_CONTROL_ROOT):
        shutil.rmtree(_CONTROL_ROOT)
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


def test_retire_tombstone_bumps_epoch_sets_boundary_all_clears_needles():
    print("\n[test] retire_tombstone — the third NameError regression")
    _wipe_storage()
    cid = "retire-conv"
    continuity.record_targeted_forget(cid, "secret")
    epoch = continuity.retire_tombstone(cid)
    assert_eq(epoch, 1, "epoch bumped")
    on_disk = json.loads(continuity._ledger_path(cid).read_text(encoding="utf-8"))
    assert_eq(on_disk["boundary"], "all", "boundary set to 'all'")
    assert_eq(on_disk["targeted"], [], "needle list cleared")
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


if __name__ == "__main__":
    try:
        test_ledger_path_matches_the_documented_convention()
        test_ledger_path_rejects_traversal_conv_id()
        test_ledger_path_never_writes_outside_control_root_wipes()

        test_ledger_round_trip_absent_file_reads_as_epoch_zero()
        test_record_wipe_bumps_and_persists_the_epoch()
        test_record_targeted_forget_appends_a_needle()
        test_retire_tombstone_bumps_epoch_sets_boundary_all_clears_needles()
        test_reapply_all_ledgers_finds_files_written_via_ledger_path()

        test_corrupt_ledger_fails_closed_not_epoch_zero()
        test_record_wipe_recovers_from_a_corrupt_existing_ledger()

        test_write_ledger_creates_missing_parent_dir()

        test_format_band_numbers_entries_exactly_once()
        test_quarantine_excludes_digests()

        print("\nAll continuity tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
