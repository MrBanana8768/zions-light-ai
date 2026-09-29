"""
CPU-only Tier-1 tests for compactor.uploads_backup (v3.1.9.8, UPLOADS
INCIDENT follow-up: a SEPARATE, incremental mirror of OpenWebUI's uploaded
files — never folded into backup.py's own archive).

SYNTHETIC DATA ONLY. Every "photo" here is a few bytes this file writes
itself; nothing here reads, prints, or depends on any real file name or
file content. Assertions report counts, never names.

Covers the design's own failure paths first, per this project's doctrine
(see test_backup.py's own header) — the mass-change guard exists
specifically to replay the 2026-09-26 gzip-in-place incident (338 files
renamed to `<name>.gz` in one pass) without corrupting the manifest, and
that is the scenario most of this file is built around.

Run: python test_uploads_backup.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="zions-uploads-backup-test-"))

import uploads_backup  # noqa: E402

FAILED = 0


def assert_eq(a, b, label):
    global FAILED
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        FAILED += 1
        return
    print(f"  ok   {label}")


def assert_true(c, label):
    global FAILED
    if not c:
        print(f"FAIL {label}")
        FAILED += 1
        return
    print(f"  ok   {label}")


def _idle_ok():
    return True, "test"


def _new_dirs(tag: str) -> tuple[Path, Path]:
    uploads = _TMP / f"{tag}-uploads"
    root = _TMP / f"{tag}-mirror"
    uploads.mkdir(parents=True, exist_ok=True)
    return uploads, root


def _write(p: Path, data: bytes, mtime: float | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if mtime is not None:
        os.utime(p, (mtime, mtime))


def _objects_count(root: Path) -> int:
    d = root / uploads_backup.OBJECTS_DIRNAME
    if not d.is_dir():
        return 0
    return sum(1 for p in d.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------


def test_first_run_mirrors_everything_second_run_copies_nothing():
    uploads, root = _new_dirs("t1")
    for i in range(5):
        _write(uploads / f"abc123_photo{i}.jpg", f"jpg-bytes-{i}".encode() + b"\xff\xd8\xff extra", mtime=1_700_000_000 + i)

    r1 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r1["ok"], "first run reports ok")
    assert_eq(r1["new"], 5, "first run: all 5 uploads counted as new")
    assert_eq(_objects_count(root), 5, "first run: 5 distinct objects copied")

    manifest = uploads_backup.load_manifest(root)
    assert_eq(len(manifest["files"]), 5, "manifest tracks all 5 names")
    assert_eq(manifest["generation"], 1, "generation incremented once")

    r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r2["ok"], "second run reports ok")
    assert_eq(r2["new"], 0, "second run: nothing new")
    assert_eq(r2["changed"], 0, "second run: nothing changed")
    assert_eq(r2["bytes_copied"], 0, "second run: zero bytes copied — the incremental promise")
    assert_eq(_objects_count(root), 5, "second run: object count unchanged")


def test_new_file_gets_copied():
    uploads, root = _new_dirs("t2")
    _write(uploads / "a_one.png", b"\x89PNG\r\n\x1a\n" + b"one", mtime=1_700_000_100)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    _write(uploads / "b_two.png", b"\x89PNG\r\n\x1a\n" + b"two", mtime=1_700_000_200)
    r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r2["ok"], "second run ok")
    assert_eq(r2["new"], 1, "exactly the one new file is counted")
    assert_eq(_objects_count(root), 2, "two distinct objects on disk now")


def test_content_change_below_threshold_never_overwrites_sha():
    """One file's content changes in place (an edit, not a delete) while
    9 others stay put — 10%, under the mass-change threshold, so this is
    the ordinary per-file damage-detection path, not the guard. The
    manifest's recorded sha256 must NEVER move to the new hash; the new
    hash is recorded under `changed`, the OLD object is still present and
    restorable, and a WARNING is logged. (This is the second half of the
    "detect damage instead of backing it up" doctrine — the gzip-incident
    test above covers the guard; this covers the single-file case the
    guard is not supposed to catch.)"""
    uploads, root = _new_dirs("t3b")
    target = uploads / "edited_in_place.jpg"
    original = b"\xff\xd8\xff" + b"ORIGINAL-GOOD-BYTES"
    _write(target, original, mtime=1_700_000_500)
    for i in range(9):
        _write(uploads / f"steady_{i}.jpg", b"\xff\xd8\xff" + f"steady-{i}".encode(), mtime=1_700_000_510 + i)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    manifest0 = uploads_backup.load_manifest(root)
    original_sha = manifest0["files"]["edited_in_place.jpg"]["sha256"]
    original_object = root / uploads_backup.OBJECTS_DIRNAME / original_sha[:2] / original_sha
    assert_true(original_object.is_file(), "the original object exists before the edit")

    # Same name, DIFFERENT bytes, same size (so an mtime-only check could
    # not catch it by accident) — an in-place content edit.
    corrupted = b"\xff\xd8\xff" + b"DIFFERENT-BAD-BYTES"
    assert_eq(len(corrupted), len(original), "same size as the original (stat size alone can't tell)")
    _write(target, corrupted, mtime=1_700_000_600)  # new mtime -> triggers a rehash

    r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r2["ok"], "the below-threshold content-change run reports ok")
    assert_true(not r2["suspect"], "one file out of ten changing does not trip the mass-change guard")
    assert_eq(r2["changed"], 1, "exactly one content change is recorded")

    manifest1 = uploads_backup.load_manifest(root)
    rec = manifest1["files"]["edited_in_place.jpg"]
    assert_eq(rec["sha256"], original_sha,
              "the manifest's canonical sha256 is UNCHANGED — this is the guarantee a mutant "
              "that lets a content change overwrite it would break")
    assert_true(len(rec.get("changed", [])) == 1 and rec["changed"][0]["sha256"] != original_sha,
                "the NEW hash is recorded under 'changed', separately from the canonical sha256")
    assert_true(original_object.is_file(), "the OLD (good) object is still present after the edit")
    assert_true(original_object.read_bytes() == original, "the old object's bytes are untouched — still restorable")


def test_deleted_file_kept_then_expired():
    # Ten files in the baseline, only one deleted: 10% affected, comfortably
    # under the 20% mass-change threshold — this test is about the ordinary
    # single-delete path, not the guard (see the gzip-incident test for
    # that).
    uploads, root = _new_dirs("t3")
    target = uploads / "gone_soon.png"
    _write(target, b"\x89PNG\r\n\x1a\n" + b"bye", mtime=1_700_000_300)
    for i in range(9):
        _write(uploads / f"stays_{i}.png", b"\x89PNG\r\n\x1a\n" + f"keep-{i}".encode(), mtime=1_700_000_310 + i)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    target.unlink()

    saved_keep = uploads_backup.KEEP_DELETED_DAYS
    try:
        uploads_backup.KEEP_DELETED_DAYS = 90.0  # the shipped default
        r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
        assert_true(r2["ok"], "delete-cycle run ok")
        assert_eq(r2["deleted"], 1, "the vanished file is recorded as deleted, not dropped")
        assert_eq(r2["expired"], 0, "not old enough to expire yet at the 90-day default")
        assert_eq(_objects_count(root), 10, "the object is KEPT after a delete (all 10 still on disk)")
        manifest = uploads_backup.load_manifest(root)
        rec = manifest["files"]["gone_soon.png"]
        assert_true(rec["deleted_at"] is not None, "manifest records deleted_at")

        # Now simulate 91 days having passed by backdating deleted_at, and
        # dropping KEEP_DELETED_DAYS so THIS run's cutoff catches it.
        manifest["files"]["gone_soon.png"]["deleted_at"] = uploads_backup._iso(time.time() - 91 * 86400.0)
        uploads_backup.save_manifest(root, manifest)
        r3 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
        assert_true(r3["ok"], "expiry-cycle run ok")
        assert_eq(r3["expired"], 1, "the long-deleted record is expired")
        assert_eq(_objects_count(root), 9, "only its object is swept once expired; the other 9 remain")
        manifest2 = uploads_backup.load_manifest(root)
        assert_true("gone_soon.png" not in manifest2["files"], "the manifest entry itself is gone")
    finally:
        uploads_backup.KEEP_DELETED_DAYS = saved_keep


def test_gzip_incident_guard_trips_and_restore_recovers_byte_identical():
    """The literal 2026-09-26 incident, synthetically: every tracked upload
    is replaced by a same-named-plus-.gz file (the original name vanishes).
    This must trip the mass-change guard (write nothing to manifest.files),
    and scripts/uploads-backup.py restore --apply must bring every original
    back byte-identical from the object store, which the guard never
    touched.
    """
    uploads, root = _new_dirs("t4")
    originals = {}
    for i in range(10):
        name = f"id{i}_photo.jpg"
        data = f"jpg-content-{i}-".encode() * 20 + b"\xff\xd8\xff"
        _write(uploads / name, data, mtime=1_700_001_000 + i)
        originals[name] = data

    r1 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r1["ok"], "baseline run ok")
    assert_eq(r1["new"], 10, "baseline run mirrors all 10")
    manifest_before = json.loads((root / uploads_backup.MANIFEST_NAME).read_text())

    # Simulate: gzip every file in place (original vanishes, a NEW .gz name
    # appears). No file keeps its original name. Backdated mtimes (well
    # past SETTLE_S) so the settle window doesn't hold these back as
    # "possibly still being written" — that is a different finding, tested
    # separately below.
    for i, name in enumerate(list(originals)):
        p = uploads / name
        gz = uploads / f"{name}.gz"
        p.rename(gz)
        gz.write_bytes(b"\x1f\x8b" + b"fake-gzip-payload")  # distinct content, distinct name
        os.utime(gz, (1_700_002_000 + i, 1_700_002_000 + i))

    r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r2["ok"], "the guard reports ok (a detected, handled condition, not a crash)")
    assert_true(r2["suspect"], "the mass-change guard trips")
    assert_eq(r2["deleted"], 0, "no per-file deletions were APPLIED — those records were frozen")
    # hostile-review round 2, finding 2: NEW files are still mirrored even
    # while the guard is tripped — objects are add-only, so this is safe,
    # and withholding it would just be more to lose while an operator
    # investigates. The 10 .gz names are, from this run's point of view,
    # indistinguishable from 10 ordinary new uploads.
    assert_eq(r2["new"], 10, "the 10 new .gz names ARE still mirrored despite the guard tripping")

    manifest_after = json.loads((root / uploads_backup.MANIFEST_NAME).read_text())
    for name, rec_before in manifest_before["files"].items():
        assert_eq(manifest_after["files"][name], rec_before,
                  "each ORIGINAL tracked name's per-file record is byte-for-byte frozen")
    for i, name in enumerate(originals):
        assert_true(f"{name}.gz" in manifest_after["files"],
                    "the new .gz name is a newly tracked (but still suspect-flagged) entry")
    assert_eq(manifest_after["generation"], manifest_before["generation"] + 1,
              "generation DOES advance — new files were genuinely mirrored this run")
    assert_true(manifest_after["last_suspect_run"] is not None, "a suspect_run marker was recorded")

    hs = uploads_backup.health_snapshot(root)
    assert_true(hs["suspect"] is True, "health_snapshot surfaces suspect=true")

    # Recovery path 1: scripts/uploads-backup.py restore --apply must bring
    # every ORIGINAL name back, byte-identical, from the (untouched) object
    # store — the guard tripping never touched those objects or records.
    for name in originals:
        (uploads / name).unlink(missing_ok=True)  # the live tree still only has the .gz names
    script_ns = _load_script()
    args = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "restore", "--apply"]
    )
    rc = script_ns.cmd_restore(args)
    assert_eq(rc, 0, "restore --apply exits 0")
    for name, data in originals.items():
        got = (uploads / name).read_bytes()
        assert_eq(got, data, "restored content is byte-identical to the original")

    # Recovery path 2: once an operator has looked and confirmed the
    # deletion is legitimate (not this incident — a real one), `accept
    # --yes` marks the still-vanished tracked names deleted and clears the
    # suspect flag. Re-derive vanished names against the CURRENT uploads
    # dir (which, after the restore above, has the 10 originals back AND
    # the 10 .gz names — nothing vanished anymore), so exercise `accept` on
    # a fresh case where restore was NOT run first.
    uploads2, root2 = _new_dirs("t4b")
    for i in range(10):
        _write(uploads2 / f"k{i}.jpg", b"\xff\xd8\xff" + f"k{i}".encode(), mtime=1_700_003_000 + i)
    uploads_backup.run_once(uploads2, root2, idle_check=_idle_ok, sleep=lambda s: None)
    for i in range(3):
        (uploads2 / f"k{i}.jpg").unlink()  # 3/10 = 30% > 20% threshold
    r3 = uploads_backup.run_once(uploads2, root2, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r3["suspect"], "a legitimate 30% deletion also trips the guard")
    args2 = script_ns.build_parser().parse_args(
        ["--backup-root", str(root2), "--uploads-dir", str(uploads2), "accept", "--yes"]
    )
    rc2 = script_ns.cmd_accept(args2)
    assert_eq(rc2, 0, "accept --yes exits 0")
    manifest2 = uploads_backup.load_manifest(root2)
    for i in range(3):
        assert_true(manifest2["files"][f"k{i}.jpg"]["deleted_at"] is not None,
                    "accept marked the still-vanished tracked file deleted")
    assert_true(manifest2["last_suspect_run"] is None, "accept cleared the suspect marker")


def test_check_db_against_synthetic_sqlite_file_table():
    uploads, root = _new_dirs("t5")
    _write(uploads / "u1_a.png", b"\x89PNG\r\n\x1a\n" + b"aaa", mtime=1_700_002_000)
    _write(uploads / "u2_b.png", b"\x89PNG\r\n\x1a\n" + b"bbb", mtime=1_700_002_100)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    db_path = _TMP / "t5-webui.db"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
    con.execute("INSERT INTO file (id, path) VALUES (1, ?)", (str(uploads / "u1_a.png"),))
    con.execute("INSERT INTO file (id, path) VALUES (2, ?)", (str(uploads / "u2_b.png"),))
    con.execute("INSERT INTO file (id, path) VALUES (3, ?)", (str(uploads / "u3_missing.png"),))
    con.commit()
    con.close()

    script_ns = _load_script()
    args = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "check-db", "--db", str(db_path)]
    )
    rc = script_ns.cmd_check_db(args)
    assert_eq(rc, 1, "check-db reports a problem (one row is not mirrored/on disk)")


def test_also_mismatched_quarantines():
    uploads, root = _new_dirs("t6")
    p = uploads / "u_x.png"
    _write(p, b"\x89PNG\r\n\x1a\n" + b"original", mtime=1_700_003_000)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    # Live content now differs from what the mirror holds (simulating a
    # content edit that was NOT re-mirrored, e.g. a run skipped or the
    # guard tripped in between).
    _write(p, b"\x89PNG\r\n\x1a\n" + b"EDITED-DIFFERENT-BYTES", mtime=1_700_003_000)

    script_ns = _load_script()
    args = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "restore", "--apply", "--also-mismatched"]
    )
    rc = script_ns.cmd_restore(args)
    assert_eq(rc, 0, "restore --apply --also-mismatched exits 0")
    restored = p.read_bytes()
    assert_true(restored == b"\x89PNG\r\n\x1a\n" + b"original", "the ORIGINAL mirrored bytes are back in place")

    quarantine_dirs = list(uploads.parent.glob("uploads-quarantine-*"))
    assert_eq(len(quarantine_dirs), 1, "exactly one quarantine dir was created")
    aside = list(quarantine_dirs[0].glob("*"))
    assert_eq(len(aside), 1, "the edited file was moved aside, not deleted")
    assert_true(aside[0].read_bytes() == b"\x89PNG\r\n\x1a\n" + b"EDITED-DIFFERENT-BYTES",
                "the quarantined copy holds the edited (pre-restore) bytes")


def test_idle_gating_skips_a_busy_cycle():
    uploads, root = _new_dirs("t7")
    _write(uploads / "busy_a.png", b"\x89PNG\r\n\x1a\n" + b"x", mtime=1_700_004_000)

    calls = {"n": 0}

    def busy():
        calls["n"] += 1
        return False, "requests_in_flight is 3"

    r = uploads_backup.run_once(uploads, root, idle_check=busy, sleep=lambda s: None)
    assert_true(r["ok"], "a skipped-for-busy cycle still reports ok (not a failure)")
    assert_eq(r["skipped"], "busy", "the cycle records why it was skipped")
    assert_eq(calls["n"], 1, "the idle check was actually consulted")
    assert_eq(_objects_count(root), 0, "nothing was copied while busy")
    assert_true(not (root / uploads_backup.MANIFEST_NAME).exists(), "no manifest was even created while busy")


def test_idle_gate_http_error_fails_closed_and_mid_run_recheck_pauses():
    """hostile-review round 2, finding 4, both bullets in one test:
      (a) an HTTP error RESPONSE (503) with a non-zero requests_in_flight
          in its body must fail CLOSED (skip), not open — an unreachable
          connection still fails open (unaffected, not retested here);
      (b) a run long enough to cross IDLE_RECHECK_FILES re-polls
          idle_check() mid-run and, once busy, stops cleanly — the
          manifest still saves whatever was already applied.
    """
    import http.server
    import json as _json
    import threading

    calls = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = _json.dumps({"request_tracing": {"requests_in_flight": 2}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved_url = uploads_backup.HEALTH_URL
    try:
        uploads_backup.HEALTH_URL = f"http://127.0.0.1:{srv.server_address[1]}/health/full"
        ok, why = uploads_backup.is_idle()
        assert_true(not ok, "a 503 with requests_in_flight=2 fails CLOSED (busy), not open")
        assert_true("503" in why, "the reason names the HTTP status")
    finally:
        uploads_backup.HEALTH_URL = saved_url
        srv.shutdown()

    # (b) mid-run recheck: 30 new files, IDLE_RECHECK_FILES temporarily
    # lowered so a recheck is guaranteed within this small run, and the
    # SECOND idle_check call reports busy.
    uploads, root = _new_dirs("t7b")
    for i in range(30):
        _write(uploads / f"m{i}.jpg", b"\xff\xd8\xff" + os.urandom(50), mtime=1_700_004_100 + i)
    saved_recheck = uploads_backup.IDLE_RECHECK_FILES
    uploads_backup.IDLE_RECHECK_FILES = 10
    try:
        def idle_then_busy():
            calls.append(1)
            return (True, "idle") if len(calls) == 1 else (False, "requests_in_flight is 1")

        r = uploads_backup.run_once(uploads, root, idle_check=idle_then_busy, sleep=lambda s: None)
        assert_true(r["ok"], "a paused-mid-run cycle still reports ok")
        assert_true(r["paused"], "the report says it paused")
        assert_true(0 < r["new"] < 30, "SOME but not all files were mirrored before pausing")
        assert_true(len(calls) >= 2, "idle_check was consulted more than once in a single run")
        manifest = uploads_backup.load_manifest(root)
        assert_eq(len(manifest["files"]), r["new"], "the manifest reflects exactly the partial progress made")
    finally:
        uploads_backup.IDLE_RECHECK_FILES = saved_recheck


def test_settle_window_skips_a_recently_written_upload():
    """hostile-review round 2, finding 1: a file whose mtime is within
    SETTLE_S of "now" is skipped entirely — not mirrored, not flagged
    changed — so a half-written upload (about to receive more bytes) is
    never captured mid-write."""
    uploads, root = _new_dirs("t9")
    fresh = uploads / "still_writing.jpg"
    fresh.write_bytes(b"\xff\xd8\xff" + b"partial")  # default mtime: now

    r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r["ok"], "run ok")
    assert_eq(r["new"], 0, "the too-recent file is NOT mirrored yet")
    assert_eq(r["skipped_settling"], 1, "it is counted as settling, not silently dropped")
    manifest = uploads_backup.load_manifest(root)
    assert_true("still_writing.jpg" not in manifest["files"], "no record exists for it yet")

    # Once it's old enough (simulated by lowering SETTLE_S rather than
    # sleeping in a test), the very same file IS picked up.
    saved_settle = uploads_backup.SETTLE_S
    uploads_backup.SETTLE_S = 0.0
    try:
        r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
        assert_eq(r2["new"], 1, "now that it has settled, it is mirrored")
    finally:
        uploads_backup.SETTLE_S = saved_settle


def test_object_hash_while_copy_is_self_consistent():
    """hostile-review round 2, finding 1 (the core TOCTOU fix): the
    OBJECT's own bytes must always match its own filename (sha256) — a
    single read pass guarantees this structurally. Contrast with the
    PREVIOUS approach (hash, then separately copy — two reads of the same
    live file) reproduced inline here, which does NOT have this property:
    a file that changes content between those two reads is captured under
    the FIRST read's hash, forever, exactly the finding's own reproduction.
    """
    uploads, root = _new_dirs("t10")
    src = uploads / "race.jpg"
    content_a = b"\xff\xd8\xff" + b"A" * 5000
    src.write_bytes(content_a)

    # The fix: one function call, one read pass.
    sha, size, created, _mtime, _mtime_ns, _inode = uploads_backup._hash_and_copy_object(src, root)
    obj = uploads_backup._object_path(root, sha)
    assert_true(created, "a new object was written")
    on_disk = obj.read_bytes()
    assert_eq(hashlib.sha256(on_disk).hexdigest(), sha,
              "the object's OWN bytes always hash to its OWN filename — self-consistent by construction")
    assert_eq(on_disk, content_a, "and for this (non-racing) call, that content is exactly what was read")

    # Reproduce the PREVIOUS (buggy) shape directly, to show the contrast:
    # hash first, then race, then copy under the STALE hash.
    src2 = uploads / "race2.jpg"
    content_b_original = b"\xff\xd8\xff" + b"B" * 5000
    src2.write_bytes(content_b_original)
    stale_sha, _ = uploads_backup._sha256_file(src2)     # read #1 (the old code's hash pass)
    content_b_finished = b"\xff\xd8\xff" + b"B" * 5000 + b"-MORE-BYTES-THAT-ARRIVED-LATER"
    src2.write_bytes(content_b_finished)                  # the upload finishes mid-race
    stale_obj = uploads_backup._object_path(root, stale_sha)
    stale_obj.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src2, stale_obj)                       # read #2 (the old code's copy pass)
    on_disk_stale = stale_obj.read_bytes()
    assert_true(hashlib.sha256(on_disk_stale).hexdigest() != stale_sha,
                "REPRODUCED: under the OLD two-read pattern, the object's filename (a stale hash) "
                "does NOT match its own content — exactly finding 1's bug, shown for contrast")


def test_restore_refuses_a_damaged_object():
    """hostile-review round 2, finding 1 (restore side): a truncated/
    corrupted object on disk must never be written over a live file."""
    uploads, root = _new_dirs("t11")
    _write(uploads / "good.jpg", b"\xff\xd8\xff" + b"good-bytes", mtime=1_700_006_000)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    manifest = uploads_backup.load_manifest(root)
    sha = manifest["files"]["good.jpg"]["sha256"]
    obj = uploads_backup._object_path(root, sha)
    obj.write_bytes(b"DAMAGED-NOT-THE-REAL-CONTENT")  # corrupt the backed-up object in place
    (uploads / "good.jpg").unlink()  # now "missing" from disk -> a restore candidate

    script_ns = _load_script()
    args = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "restore", "--apply"]
    )
    rc = script_ns.cmd_restore(args)
    assert_eq(rc, 1, "restore exits 1 — a problem was found, not silently ignored")
    assert_true(not (uploads / "good.jpg").exists(),
                "the damaged object was NEVER written to the live path")


def test_restore_rejects_path_traversal_and_absolute_names():
    """hostile-review round 2, finding 5: a manifest.json entry whose name
    is `../..`-relative or an absolute path must never be joined onto
    uploads_dir and used as a restore destination — Path's own `/` operator
    silently DISCARDS the left side for an absolute right side, which is
    exactly the trap. Reject both, write nothing outside uploads_dir."""
    uploads, root = _new_dirs("t12")
    outside_root = uploads.parent / "outside"
    outside_root.mkdir(parents=True, exist_ok=True)
    escape_target = outside_root / "escaped.jpg"
    abs_target = outside_root / "abs.jpg"

    _write(uploads / "a.jpg", b"\xff\xd8\xff" + b"legit", mtime=1_700_007_000)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    manifest = json.loads((root / uploads_backup.MANIFEST_NAME).read_text())
    rec = manifest["files"].pop("a.jpg")
    manifest["files"]["../outside/escaped.jpg"] = dict(rec)
    manifest["files"][str(abs_target)] = dict(rec)
    (root / uploads_backup.MANIFEST_NAME).write_text(json.dumps(manifest))

    script_ns = _load_script()
    args = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "restore", "--apply"]
    )
    rc = script_ns.cmd_restore(args)
    assert_eq(rc, 0, "restore exits 0 — the unsafe entries were rejected, not fatal errors")
    assert_true(not escape_target.exists(), "the ../ traversal name was never written")
    assert_true(not abs_target.exists(), "the absolute-path name was never written")


def test_symlinks_and_non_regular_files_are_never_mirrored():
    """hostile-review round 2, finding 6: a symlink to a file outside
    uploads_dir, and a symlinked DIRECTORY, must both be completely
    invisible to the mirror — never hashed, never copied, never listed."""
    uploads, root = _new_dirs("t13")
    outside = uploads.parent / "outside13"
    outside.mkdir(parents=True, exist_ok=True)
    secret = outside / "secret.txt"
    secret.write_text("outside-secret-content")
    # Backdated well past SETTLE_S: `path.stat()` in run_once's classify
    # step FOLLOWS a symlink to its target's own mtime, so a just-written
    # target would otherwise be skipped by the SETTLE window (finding 1)
    # instead of exercising the symlink guard (finding 6) this test is
    # actually about — keep the two findings' tests independent.
    os.utime(secret, (1_700_007_500, 1_700_007_500))

    (uploads / "link_to_secret.jpg").symlink_to(secret)
    (uploads / "linked_dir").symlink_to(outside, target_is_directory=True)
    _write(uploads / "real.jpg", b"\xff\xd8\xff" + b"real-content", mtime=1_700_008_000)

    r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r["ok"], "run ok")
    assert_eq(r["new"], 1, "only the ONE real regular file was mirrored")
    manifest = uploads_backup.load_manifest(root)
    assert_eq(sorted(manifest["files"]), ["real.jpg"], "no symlink and nothing reached through the symlinked dir is tracked")
    secret_sha = hashlib.sha256(b"outside-secret-content").hexdigest()
    assert_true(not uploads_backup.has_object(root, secret_sha), "the secret's content was never copied into the mirror")


def test_mirror_size_cap_refuses_new_copies_over_the_limit():
    """hostile-review round 2, finding 9: a hard cap on the mirror's own
    size (objects_bytes), independent of the (on a network FS, close to
    meaningless) cluster-wide free-space number."""
    uploads, root = _new_dirs("t14")
    _write(uploads / "a.jpg", b"\xff\xd8\xff" + b"x" * 1000, mtime=1_700_009_000)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    saved_cap = uploads_backup.MAX_MIRROR_MB
    uploads_backup.MAX_MIRROR_MB = 0  # 0 MB: even the first byte is "over"
    try:
        # A cap of exactly 0 disables the guard by this module's own "cap
        # > 0" convention (see _require_mirror_size_cap) — use a value
        # that is nonzero but already exceeded instead.
        manifest = uploads_backup.load_manifest(root)
        manifest["objects_bytes"] = 999_999_999
        uploads_backup.save_manifest(root, manifest)
        uploads_backup.MAX_MIRROR_MB = 1
        _write(uploads / "b.jpg", b"\xff\xd8\xff" + b"y" * 1000, mtime=1_700_009_100)
        r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
        assert_true(not r["ok"], "the run refuses once the mirror is already over its cap")
        assert_true("MB cap" in r["detail"], "the detail explains why")
        hs = uploads_backup.health_snapshot(root)
        assert_true(hs["over_mirror_size_cap"], "health surfaces the cap breach")
    finally:
        uploads_backup.MAX_MIRROR_MB = saved_cap


def test_no_upload_name_ever_appears_in_logs_or_health_after_an_induced_error():
    """hostile-review round 2, finding 8: an exception's own str(e) can
    embed the path (and therefore her upload's own name/extension) that
    triggered it — a PermissionError or FileNotFoundError message includes
    the path by default. Every except block in this module logs/records
    only type(e).__name__ plus counts. Induce a real error (an unreadable
    directory entry masquerading as a stat failure) and check the report,
    a captured log record, AND health_snapshot for the distinctive name.
    """
    import logging as _logging

    uploads, root = _new_dirs("t15")
    secret_name = "VERY-DISTINCTIVE-UPLOAD-NAME-8f2a91.jpg"
    bad = uploads / secret_name
    _write(bad, b"\xff\xd8\xff" + b"x", mtime=1_700_009_500)

    class _Handler(_logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())

    handler = _Handler()
    logger = _logging.getLogger("compactor.uploads_backup")
    logger.addHandler(handler)

    real_stat = Path.stat

    def _boom_stat(self, *a, **k):
        if self.name == secret_name:
            raise PermissionError(f"[Errno 13] Permission denied: '{self}'")
        return real_stat(self, *a, **k)

    Path.stat = _boom_stat
    try:
        r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
        assert_true(r["ok"], "one bad file does not abort the whole run")
        assert_eq(r["new"], 0, "the one bad file was skipped, not mirrored")
        assert_true(secret_name not in json.dumps(r), "the report never contains the file's name")
        for line in handler.lines:
            assert_true(secret_name not in line, "no captured log line contains the file's name")
        hs = uploads_backup.health_snapshot(root)
        assert_true(secret_name not in json.dumps(hs), "health_snapshot never contains the file's name")
    finally:
        Path.stat = real_stat
        logger.removeHandler(handler)


def test_resurrected_file_is_undeleted_and_rehashed():
    """hostile-review round 4, finding 1 (probe redel.sh): a name the
    manifest marks deleted, that is LIVE again (restored from the mirror,
    or simply re-uploaded under the same name), must not stay marked
    deleted forever. Its record must be un-deleted, counted as
    `resurrected`, and rehashed — otherwise the guard ignores it forever,
    plain `restore` says "nothing to do" for it, and the expiry sweep
    eventually deletes its OBJECT out from under a file that is still
    live on disk.
    """
    uploads, root = _new_dirs("t19")
    target = uploads / "back_again.jpg"
    original = b"\xff\xd8\xff" + b"original-content"
    _write(target, original, mtime=1_700_014_000)
    for i in range(8):
        _write(uploads / f"steady_{i}.jpg", b"\xff\xd8\xff" + f"s{i}".encode(), mtime=1_700_014_010 + i)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    target.unlink()
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    manifest = uploads_backup.load_manifest(root)
    assert_true(manifest["files"]["back_again.jpg"]["deleted_at"] is not None, "recorded as deleted")

    # It comes back — byte-identical (e.g. an operator's own restore).
    _write(target, original, mtime=1_700_014_100)
    r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r["ok"], "run ok")
    assert_eq(r["resurrected"], 1, "counted as resurrected")
    manifest2 = uploads_backup.load_manifest(root)
    rec = manifest2["files"]["back_again.jpg"]
    assert_true(rec["deleted_at"] is None, "no longer marked deleted")

    # A subsequent plain restore must say "nothing to do" for it (it's
    # live, not missing) rather than silently leaving it perpetually
    # "deleted" — and the expiry sweep must never remove its (still
    # referenced) canonical object.
    saved_keep = uploads_backup.KEEP_DELETED_DAYS
    uploads_backup.KEEP_DELETED_DAYS = 0.00001
    try:
        uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    finally:
        uploads_backup.KEEP_DELETED_DAYS = saved_keep
    obj = uploads_backup._object_path(root, rec["sha256"])
    assert_true(obj.is_file(), "the canonical object was NOT swept — the file is live, not deleted")
    assert_eq(obj.read_bytes(), original, "and it's still the right content")


def test_pre_read_race_caught_via_settle_and_classify_sig():
    """hostile-review round 4, finding 2 (probe race.py, shape A): a
    writer that PAUSES mid-rewrite — first half written (bumping mtime to
    "now"), then a long pause before the second half — produces a file
    whose STAT already looks wrong (too-fresh mtime, or a stat that no
    longer matches what classification saw) even before a single byte is
    read for backup. The round-3 post-read-only fstat check alone allowed
    this: reading a paused-mid-rewrite file from start to end is a single,
    internally-consistent read (no fstat mismatch across THAT read), so it
    would happily hash+copy the torn A/B blend and record it as canonical.
    The fix checks the PRE-read fstat against the settle window and
    against classification's own stat before ever starting the read.
    """
    uploads, root = _new_dirs("t20")
    p = uploads / "paused.jpg"
    original = b"\xff\xd8\xff" + b"A" * 5000
    p.write_bytes(original)
    os.utime(p, (1_700_015_000, 1_700_015_000))  # old — settled at classification time

    # Classification runs first inside run_once; we simulate "the writer
    # pauses mid-rewrite AFTER classification, before the read" by
    # rewriting the file ourselves between the classify-time snapshot and
    # the copy loop — done here via a hook on _hash_and_copy_object itself
    # (the one seam run_once actually calls through), so the rewrite lands
    # exactly between class_stat capture and the read.
    orig_hac = uploads_backup._hash_and_copy_object
    state = {"done": False}

    def hooked(src, root, **kw):
        if not state["done"] and src.name == "paused.jpg":
            state["done"] = True
            with open(p, "r+b") as w:
                w.seek(3)
                w.write(b"B" * 2500)  # first half only — mtime jumps to "now"
        return orig_hac(src, root, **kw)

    uploads_backup._hash_and_copy_object = hooked
    try:
        r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    finally:
        uploads_backup._hash_and_copy_object = orig_hac

    assert_true(r["ok"], "run ok")
    assert_eq(r["new"], 0, "the paused-mid-rewrite file was NOT captured")
    assert_true(r["skipped_race"] >= 1, "counted as a detected race (settle/classify-sig, not the post-read check)")
    manifest = uploads_backup.load_manifest(root)
    assert_true("paused.jpg" not in manifest["files"], "no torn A/B blend was ever recorded as canonical")


def test_accept_gate_uses_marker_baseline_not_diluted_recount():
    """hostile-review round 4, finding 4 (probe gz.sh + acc.sh's gzacc
    path): the mass-change guard trips at 100% (20 vanished / 20 tracked)
    on a gzip-in-place incident. Because new (safe) uploads are still
    mirrored while suspect, 20 new `.gz` names get added — so by the time
    `accept` runs, its OWN fresh recount is 20 vanished / 40 NOW-tracked =
    exactly 50%, which used to slide UNDER the >50% refusal with no
    --force-large needed. `accept` must gate on max(its own pct, the
    marker's own recorded pct, affected against the marker's ORIGINAL
    baseline) instead.
    """
    uploads, root = _new_dirs("t21")
    originals = {}
    for i in range(20):
        name = f"g{i}.jpg"
        data = b"\xff\xd8\xff" + f"g{i}".encode() * 20
        _write(uploads / name, data, mtime=1_700_016_000 + i)
        originals[name] = data
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    for i, name in enumerate(list(originals)):
        p = uploads / name
        gz = uploads / f"{name}.gz"
        p.rename(gz)
        gz.write_bytes(b"\x1f\x8b" + f"gz{i}".encode())
        os.utime(gz, (1_700_017_000 + i, 1_700_017_000 + i))
    r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r["suspect"], "the guard trips at 100% (20/20)")

    script_ns = _load_script()
    args = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "accept"]
    )
    rc = script_ns.cmd_accept(args)
    assert_eq(rc, 1, "accept refuses (needs --force-large) even though its OWN fresh recount is exactly 50%")

    args_forced = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "accept", "--yes", "--force-large"]
    )
    rc2 = script_ns.cmd_accept(args_forced)
    assert_eq(rc2, 0, "with --force-large, accept applies")


def test_accept_does_not_clear_suspect_when_roll_forward_fails():
    """.9 fix (round-4 open observation, uploads-merge task 3): `accept`
    used to clear `last_suspect_run` unconditionally after its roll-forward
    loop, even when one of the changed files it was supposed to roll
    forward could not actually be written this run (here: its live mtime
    is still inside the settle window, so `_hash_and_copy_single_pass`
    correctly refuses it and returns None) — reporting success and exiting
    0 while the mirror was, in fact, left half-accepted. `accept` must
    treat that as a failure: exit 1, print which/how many roll-forwards
    failed, and leave `last_suspect_run` set (while still persisting the
    parts that DID succeed — the vanished-file deletions, and any other
    file that DID roll forward cleanly), so an operator or an automated
    re-run notices and retries instead of trusting a false "cleared".
    """
    uploads, root = _new_dirs("t22")
    originals = {}
    for i in range(10):
        name = f"h{i}.jpg"
        data = b"\xff\xd8\xff" + f"h{i}".encode() * 20
        _write(uploads / name, data, mtime=1_700_018_000 + i)
        originals[name] = data
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    # 3/10 = 30% > 20% threshold: a legitimate-looking mass deletion trips
    # the guard, exactly like test_accept_gate_uses_marker_baseline_...'s
    # own t4b case above.
    for i in range(3):
        (uploads / f"h{i}.jpg").unlink()
    # One SURVIVING tracked file's content changes, written with mtime=now
    # (well inside the default 300s settle window) — the shape that makes
    # `_hash_and_copy_single_pass` return None during accept's apply step.
    changed_name = "h5.jpg"
    _write(uploads / changed_name, b"\xff\xd8\xff" + b"EDITED-JUST-NOW" * 20)  # mtime defaults to now
    r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r["suspect"], "the 30% deletion trips the guard")

    manifest_before = uploads_backup.load_manifest(root)
    changed_rec_before = dict(manifest_before["files"][changed_name])

    script_ns = _load_script()
    args = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "accept", "--yes"]
    )
    rc = script_ns.cmd_accept(args)
    assert_eq(rc, 1, "accept exits 1 when a roll-forward could not actually be written")

    manifest_after = uploads_backup.load_manifest(root)
    assert_true(manifest_after["last_suspect_run"] is not None,
                "last_suspect_run is NOT cleared when a roll-forward failed")
    for i in range(3):
        assert_true(manifest_after["files"][f"h{i}.jpg"]["deleted_at"] is not None,
                    "the still-vanished files ARE marked deleted (the part that DID succeed persists)")
    assert_eq(manifest_after["files"][changed_name].get("cur_sha256"), changed_rec_before.get("cur_sha256"),
              "the too-fresh file's cur_sha256 is untouched — it was NOT silently rolled forward")

    # Re-running accept once the file is old enough (lower SETTLE_S, the
    # same trick the rest of this file uses to simulate the wait) now
    # succeeds and clears the flag.
    saved_settle = script_ns.SETTLE_S
    script_ns.SETTLE_S = 0.0
    try:
        rc2 = script_ns.cmd_accept(args)
    finally:
        script_ns.SETTLE_S = saved_settle
    assert_eq(rc2, 0, "accept succeeds and clears the flag once the roll-forward can actually complete")
    manifest_final = uploads_backup.load_manifest(root)
    assert_true(manifest_final["last_suspect_run"] is None, "last_suspect_run is cleared once every roll-forward succeeds")


def test_guard_stays_suspect_across_runs_c2():
    """hostile-review round 3, finding 1 (probe C2): 100 tracked uploads,
    22 gzipped-in-place (the exact incident SHAPE — 22 original names
    vanish, 22 unrelated .gz names appear). 22% trips the guard on run 0.
    The bug: a purely-recomputed-each-run guard saw the 22 newly-mirrored
    .gz names inflate `baseline` on run 1 (100 -> 122 tracked, non-deleted
    names) without inflating `affected` (still the same 22 frozen vanished
    names) — 22/122 = 18%, UNDER threshold, so run 1 silently un-suspected
    itself and marked the 22 vanished names deleted with NO operator
    action at all. Must stay suspect on every subsequent run until
    `accept`."""
    uploads, root = _new_dirs("t16")
    for i in range(100):
        _write(uploads / f"f{i}.jpg", b"\xff\xd8\xff" + f"f{i}".encode(), mtime=1_700_010_000 + i)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    for i in range(22):
        p = uploads / f"f{i}.jpg"
        data = p.read_bytes()
        p.unlink()
        gz = uploads / f"f{i}.jpg.gz"
        gz.write_bytes(b"\x1f\x8b" + data)  # a distinct new name; original vanishes
        os.utime(gz, (1_700_011_000 + i, 1_700_011_000 + i))

    for day in range(3):
        r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
        assert_true(r["suspect"], f"day {day}: still suspect")
        m = uploads_backup.load_manifest(root)
        deleted_marked = sum(1 for rec in m["files"].values() if rec.get("deleted_at"))
        assert_eq(deleted_marked, 0, f"day {day}: nothing was auto-marked deleted while suspect")
        hs = uploads_backup.health_snapshot(root)
        assert_true(hs["suspect"], f"day {day}: health_snapshot agrees it's still suspect")


def test_restore_include_deleted():
    """hostile-review round 3, finding 2: OPERATIONS.md claims a deleted
    upload is "kept 90 days so recoverable" — until this command existed,
    nothing actually restored a deleted record. `restore` (no flag) must
    still leave it alone; `--include-deleted` brings it back; `--deleted-
    since` narrows to records deleted at/after a timestamp."""
    # 9 untouched files + 1 deleted = 10% affected, comfortably under the
    # 20% mass-change threshold (see the earlier delete/expiry test for
    # the same reasoning) — this test is about --include-deleted, not the
    # guard.
    uploads, root = _new_dirs("t17")
    for i in range(9):
        _write(uploads / f"keep{i}.jpg", b"\xff\xd8\xff" + f"kept{i}".encode(), mtime=1_700_012_000 + i)
    target = uploads / "gone.jpg"
    _write(target, b"\xff\xd8\xff" + b"gone-content", mtime=1_700_012_100)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    target.unlink()
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    manifest = uploads_backup.load_manifest(root)
    assert_true(manifest["files"]["gone.jpg"]["deleted_at"] is not None, "recorded as deleted")

    script_ns = _load_script()
    args_plain = script_ns.build_parser().parse_args(
        ["--backup-root", str(root), "--uploads-dir", str(uploads), "restore", "--apply"]
    )
    script_ns.cmd_restore(args_plain)
    assert_true(not target.exists(), "without --include-deleted, a deleted upload is NOT restored")

    future_cutoff = uploads_backup._iso(time.time() + 3600)
    args_future = script_ns.build_parser().parse_args([
        "--backup-root", str(root), "--uploads-dir", str(uploads), "restore", "--apply",
        "--include-deleted", "--deleted-since", future_cutoff,
    ])
    script_ns.cmd_restore(args_future)
    assert_true(not target.exists(), "--deleted-since in the FUTURE excludes it")

    args_incl = script_ns.build_parser().parse_args([
        "--backup-root", str(root), "--uploads-dir", str(uploads), "restore", "--apply",
        "--include-deleted",
    ])
    rc = script_ns.cmd_restore(args_incl)
    assert_eq(rc, 0, "restore --include-deleted exits 0")
    assert_true(target.exists(), "--include-deleted brings the deleted upload back")
    assert_eq(target.read_bytes(), b"\xff\xd8\xff" + b"gone-content", "byte-identical")


def test_hash_and_copy_detects_inplace_rewrite_race():
    """hostile-review round 3, finding 5 (probe t.py's shape): an in-place
    rewrite that lands ENTIRELY within one read pass (same size, so
    finding-1's single-pass fix alone doesn't notice anything wrong) must
    still be caught — fstat before/after the read disagreeing on
    size/mtime_ns raises `_RaceDetected`, discarding the tmp copy. The
    file is simply skipped this run (like the settle window) and picked
    up cleanly next time once it has stopped moving."""
    uploads, root = _new_dirs("t18")
    p = uploads / "race.jpg"
    original = b"\xff\xd8\xff" + b"A" * 20000
    p.write_bytes(original)
    os.utime(p, (1_700_013_000, 1_700_013_000))

    # Same technique the probe (t.py) uses: hook hashlib.sha256 so the
    # FIRST chunk's hash.update() call — which happens strictly AFTER
    # _hash_and_copy_object's pre-read fstat, and strictly BEFORE its
    # post-read fstat — triggers an in-place, same-size rewrite. This is
    # the one race window finding-1's single read pass does NOT by itself
    # close (finding 5's own target).
    real_sha256 = hashlib.sha256
    state = {"armed": False, "fired": False}

    class _Hook:
        def __init__(self):
            self._h = real_sha256()

        def update(self, chunk):
            self._h.update(chunk)
            if state["armed"] and not state["fired"]:
                state["fired"] = True
                with open(p, "r+b") as rewriter:
                    rewriter.write(b"\xff\xd8\xff" + b"B" * 20000)

        def hexdigest(self):
            return self._h.hexdigest()

    saved_chunk = uploads_backup.CHUNK_BYTES
    uploads_backup.CHUNK_BYTES = 512  # force multiple chunks so update() fires before EOF
    orig_hac = uploads_backup._hash_and_copy_object

    def hooked(src, root, **kw):
        state["armed"] = src.name == "race.jpg"
        state["fired"] = False
        return orig_hac(src, root, **kw)

    uploads_backup._hash_and_copy_object = hooked
    uploads_backup.hashlib.sha256 = _Hook
    try:
        r1 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    finally:
        uploads_backup.hashlib.sha256 = real_sha256
        uploads_backup._hash_and_copy_object = orig_hac
        uploads_backup.CHUNK_BYTES = saved_chunk

    assert_true(r1["ok"], "run ok despite the race")
    assert_eq(r1["new"], 0, "the racing file was NOT captured this run")
    assert_true(r1["skipped_race"] >= 1, "it was counted as a detected race")
    manifest = uploads_backup.load_manifest(root)
    assert_true("race.jpg" not in manifest["files"], "no record was written for it yet")

    # Next run: the file is now stable (the in-place rewrite already
    # finished, just a moment ago in wall-clock time) — lower SETTLE_S so
    # THIS test observes "captured cleanly" without a real sleep; the
    # settle window itself is exercised by its own dedicated test.
    saved_settle = uploads_backup.SETTLE_S
    uploads_backup.SETTLE_S = 0.0
    try:
        r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    finally:
        uploads_backup.SETTLE_S = saved_settle
    assert_eq(r2["new"], 1, "captured cleanly once stable")
    rec = uploads_backup.load_manifest(root)["files"]["race.jpg"]
    obj = uploads_backup._object_path(root, rec["sha256"]).read_bytes()
    assert_eq(hashlib.sha256(obj).hexdigest(), rec["sha256"], "the object is never torn/mismatched")


def test_daemon_uploads_step_survives_base_exception():
    """hostile-review round 3, finding 7 (probe d.py): asyncio.
    CancelledError is a BaseException (not Exception) since Python 3.8 —
    `except Exception` in the daemon loop's own uploads-step call site
    missed it entirely, and it killed the WHOLE daemon loop (taking the
    main backup down with it) over a failure in the unrelated uploads
    step. `except (KeyboardInterrupt, SystemExit): raise` / `except
    BaseException` must catch it instead."""
    import asyncio
    import backup as backup_mod

    def boom(**k):
        raise asyncio.CancelledError()

    saved = backup_mod._run_daemon_uploads_step
    backup_mod._run_daemon_uploads_step = boom
    try:
        raised = False
        try:
            backup_mod._run_daemon_uploads_step()
        except asyncio.CancelledError:
            raised = True
        assert_true(raised, "CONTROL: the mock itself does raise CancelledError")

        # Reproduce the call site's own try/except exactly (rather than
        # running the real infinite run_daemon loop) — see
        # test_uploads_step_is_wired_into_the_daemon_loop_only for why a
        # structural/source check is how this suite drives run_daemon.
        daemon_src = __import__("inspect").getsource(backup_mod.run_daemon)
        assert_true("except (KeyboardInterrupt, SystemExit):" in daemon_src,
                    "run_daemon's own call site re-raises KeyboardInterrupt/SystemExit")
        assert_true("except BaseException as e:" in daemon_src,
                    "run_daemon's own call site catches BaseException around the uploads step")

        survived = False
        try:
            try:
                backup_mod._run_daemon_uploads_step()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                survived = True
        except BaseException:
            survived = False
        assert_true(survived, "the exact call-site pattern swallows CancelledError")
    finally:
        backup_mod._run_daemon_uploads_step = saved


def test_uploads_step_is_wired_into_the_daemon_loop_only():
    """hostile-review round 2, finding 7: the uploads mirror must be called
    from backup.run_daemon ONLY, after run_once() returns, never from
    run_once itself (so `backup.py --once`, `zl backup now`, and zl fix's
    pre-repair backup are untouched). Three things, in order:
      1. backup.run_once's own source contains no mention of
         uploads_backup at all (a structural guarantee that it stayed
         byte-identical to HEAD — the manual diff in this change's own
         report confirms the exact match; this is the automated backstop
         against a future regression).
      2. run_daemon's source calls `_run_daemon_uploads_step(` — and does
         so AFTER its call to `run_once()` — so the wiring is actually
         there, without ever running the real (infinite) daemon loop.
      3. `_run_daemon_uploads_step` itself, called directly with injected
         tmp paths and an injected idle_check, actually drives the mirror
         — and a raise inside it (not just an `ok: False` report) is
         swallowed by the CALLER exactly the way run_daemon's own call
         site does it, never escaping.
    Nothing here touches /data/backups or the real :8080 health endpoint —
    every path is a tmp dir, every idle check is injected.
    """
    import inspect
    import backup as backup_mod

    run_once_src = inspect.getsource(backup_mod.run_once)
    assert_true("uploads_backup" not in run_once_src,
                "run_once's own source never mentions uploads_backup")

    daemon_src = inspect.getsource(backup_mod.run_daemon)
    assert_true("_run_daemon_uploads_step(" in daemon_src,
                "run_daemon's source calls the uploads-step wiring function")
    assert_true(
        daemon_src.index("run_once()") < daemon_src.index("_run_daemon_uploads_step("),
        "the uploads step is called AFTER run_once(), textually, in run_daemon's own source",
    )

    uploads, root = _new_dirs("t8")
    _write(uploads / "a.jpg", b"\xff\xd8\xff" + b"daemon-wired", mtime=1_700_005_000)
    ureport = backup_mod._run_daemon_uploads_step(
        uploads_dir=uploads, backup_root=root, idle_check=_idle_ok,
    )
    assert_true(ureport is not None and ureport.get("ok"), "the wiring function actually ran a real cycle")
    assert_eq(ureport["new"], 1, "and that cycle mirrored the one synthetic upload")
    assert_true((root / uploads_backup.MANIFEST_NAME).exists(), "a manifest was written in the INJECTED root")

    # A raise not anticipated by uploads_backup's own error handling (not
    # an `ok: False` report — an actual exception) must be swallowed by
    # run_daemon's own call site, exactly the way the call site's inline
    # try/except is written — reproduce that exact wrapping here, driving
    # the real function (not a re-implementation of the try/except).
    saved_hook = uploads_backup.run_after_main_backup

    def _boom(*a, **k):
        raise RuntimeError("synthetic uploads-step failure (may embed a path, never logged)")

    uploads_backup.run_after_main_backup = _boom
    try:
        raised = False
        try:
            backup_mod._run_daemon_uploads_step(uploads_dir=uploads, backup_root=root, idle_check=_idle_ok)
        except Exception:
            raised = True
        assert_true(raised, "CONTROL: _run_daemon_uploads_step itself does propagate an unanticipated raise")
        # ...which is exactly why run_daemon's call site wraps it — proven
        # structurally above (the call site text) rather than by running
        # the real infinite loop.
    finally:
        uploads_backup.run_after_main_backup = saved_hook


def test_root_walk_unreadable_aborts_with_no_vanished_accounting():
    """rc-review fix, LOW-MED (probe rootwalk.sh): before this fix, the
    uploads directory ROOT itself failing to list (distinct from a
    SUBDIRECTORY failing — see `_RootWalkUnreadable`'s own docstring)
    produced an exclusion prefix of `"./"`, which matched no tracked name
    at all, so every previously-tracked file looked vanished and (at
    20%+) tripped the mass-change guard on what both this test and the
    real probe simulate as a transient permissions glitch (`chmod 311`:
    enterable, not listable). This run must instead ABORT — `ok: False`,
    a `_RootWalkUnreadable` detail, no vanished/deleted accounting at
    all, and the manifest untouched.

    MUTATION CHECK: reverting `_iter_uploads`'s onerror callback to treat
    the root the same as any other unreadable subdirectory (no
    `_RootWalkUnreadable` raise) makes this test fail — `deleted` would
    become 30 and `suspect` would become True instead of the run
    aborting."""
    uploads, root = _new_dirs("t23")
    for i in range(30):
        _write(uploads / f"r{i}.jpg", b"\xff\xd8\xff" + f"r{i}".encode() * 10, mtime=1_700_020_000 + i)
    r1 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r1["ok"], "run 1 (normal) mirrors everything cleanly")
    assert_eq(r1["new"], 30, "all 30 files mirrored on run 1")

    manifest_before = uploads_backup.load_manifest(root)
    tracked_before = sum(1 for rec in manifest_before["files"].values() if rec.get("deleted_at") is None)
    assert_eq(tracked_before, 30, "CONTROL: 30 tracked before the unreadable-root run")

    # `chmod 311` (enterable, not listable — the real reviewer probe,
    # rootwalk.sh, uses exactly this) only actually blocks a scandir when
    # the test runs as a non-root user; this suite also runs inside a
    # root gate container (docker-compose.tests.yml), where root bypasses
    # DAC permission bits entirely and os.walk would just succeed. Rather
    # than have this test's behavior depend on which UID happens to run
    # it, monkeypatch `os.walk` (as `uploads_backup` itself imports and
    # calls it) to simulate the EXACT top-level scandir failure
    # `_iter_uploads`'s own `onerror` callback has to detect — a
    # PermissionError whose `.filename` is the uploads dir itself — the
    # same shape `chmod 311` produces for a non-root caller, deterministic
    # regardless of UID.
    real_walk = uploads_backup.os.walk

    def _walk_with_root_permission_denied(top, *a, **k):
        onerror = k.get("onerror")
        if onerror is not None and Path(top) == uploads:
            onerror(PermissionError(13, "Permission denied", str(uploads)))
            return iter(())
        return real_walk(top, *a, **k)

    uploads_backup.os.walk = _walk_with_root_permission_denied
    try:
        r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    finally:
        uploads_backup.os.walk = real_walk

    assert_true(not r2["ok"], "the run reports ok=False when the root itself can't be listed")
    assert_eq(r2["deleted"], 0, "NO files are marked deleted this run")
    assert_eq(r2["walk_errors"], 0, "walk_errors stays 0 — this is a hard abort, not the per-subdirectory path")
    assert_true("_RootWalkUnreadable" in r2["detail"] or "root itself could not be listed" in r2["detail"],
                "the detail names the root-unreadable condition")

    manifest_after = uploads_backup.load_manifest(root)
    tracked_after = sum(1 for rec in manifest_after["files"].values() if rec.get("deleted_at") is None)
    assert_eq(tracked_after, 30, "all 30 files are STILL tracked (none falsely marked vanished)")
    assert_true(manifest_after["last_suspect_run"] is None, "the guard did NOT trip on a transient permissions glitch")

    status = uploads_backup._read_status(root)
    assert_true(status is not None and status["ok"] is False, "status.json reflects the abort")
    assert_eq(status.get("last_error"), "_RootWalkUnreadable", "status.json's last_error names the exception type")

    # And once the root is readable again, a normal cycle resumes cleanly.
    r3 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r3["ok"], "the very next cycle, once the root is listable again, is a normal clean run")
    assert_eq(r3["deleted"], 0, "still nothing vanished")


def test_expiry_sweep_runs_before_mirror_cap_refusal():
    """rc-review fix, LOW (task 6a): the expiry sweep used to run at the
    very END of the cycle, AFTER the free-space/mirror-size cap check —
    so once a mirror was at or over
    COMPACTOR_UPLOADS_BACKUP_MAX_MB, `_require_mirror_size_cap` raised
    `_Refusal` (caught well before the sweep's old position) on EVERY
    subsequent cycle, and the one thing that could shrink the mirror back
    under the cap (expiring records already past
    COMPACTOR_UPLOADS_BACKUP_KEEP_DELETED_DAYS and freeing their now-
    orphaned objects) could never run either. Moving the sweep before the
    cap check lets an aged-out backlog free enough bytes, in the SAME
    cycle, for that cycle's own cap check to then pass.

    MUTATION CHECK: moving the sweep back to its old (post-copy) position
    makes this test fail — the cap-triggering run would refuse (`ok:
    False`, `MB cap` in detail) with `expired: 0`, instead of expiring the
    old records and accepting the new file.
    """
    uploads, root = _new_dirs("t24")
    for i in range(10):
        _write(uploads / f"c{i}.jpg", b"\xff\xd8\xff" + os.urandom(5000), mtime=1_700_021_000 + i)
    r1 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r1["ok"] and r1["new"] == 10, "CONTROL: 10 files mirrored")

    for p in list(uploads.iterdir()):
        p.unlink()
    r2 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r2["suspect"], "100% vanished trips the guard (deleted_at is NOT applied yet — frozen)")

    # Simulate an operator having already `accept`ed this deletion (real
    # deleted_at timestamps, marker cleared) and those records having
    # since aged past KEEP_DELETED_DAYS, so the sweep below is free to
    # expire them.
    manifest = uploads_backup.load_manifest(root)
    old_iso = uploads_backup._iso(time.time() - (uploads_backup.KEEP_DELETED_DAYS + 1) * 86400.0)
    for rec in manifest["files"].values():
        rec["deleted_at"] = old_iso
    manifest["last_suspect_run"] = None
    uploads_backup.save_manifest(root, manifest)
    objects_bytes_before = manifest["objects_bytes"]
    assert_true(objects_bytes_before > 0, "CONTROL: the mirror holds real bytes before the sweep")

    saved_cap = uploads_backup.MAX_MIRROR_MB
    # A cap comfortably BELOW the current (soon-to-be-expired) usage, but
    # ABOVE what a single new file needs once the sweep frees the old
    # objects.
    uploads_backup.MAX_MIRROR_MB = max(1, int((objects_bytes_before - 2000) / (1024 * 1024)) or 1)
    # MAX_MIRROR_MB is measured in whole MB by env_int in real use, but
    # the module-level constant itself is just an int the check multiplies
    # by 1024*1024 — force a byte-precise cap here via a direct monkeypatch
    # of the check instead, so this test doesn't depend on the mirror
    # being megabytes in size.
    try:
        import types
        orig_check = uploads_backup._require_mirror_size_cap

        def _byte_precise_cap(manifest_arg):
            have = manifest_arg.get("objects_bytes", 0)
            cap = objects_bytes_before - 2000
            if have >= cap:
                raise uploads_backup._Refusal(f"refusing to copy new objects: over the test's byte-precise cap ({have} >= {cap} MB cap)")

        uploads_backup._require_mirror_size_cap = _byte_precise_cap
        try:
            _write(uploads / "new1.jpg", b"\xff\xd8\xff" + os.urandom(500), mtime=time.time() - 1000)
            r3 = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
        finally:
            uploads_backup._require_mirror_size_cap = orig_check
    finally:
        uploads_backup.MAX_MIRROR_MB = saved_cap

    assert_true(r3["ok"], "the cap-triggering cycle succeeds because the sweep already freed enough bytes")
    assert_eq(r3["expired"], 10, "the sweep expired all 10 aged-out deleted records THIS cycle")
    assert_eq(r3["new"], 1, "the new file was still copied — the cap check passed AFTER the sweep freed space")
    manifest_after = uploads_backup.load_manifest(root)
    assert_true(manifest_after["objects_bytes"] < objects_bytes_before,
                "objects_bytes dropped — the mirror actually shrank")


def test_stale_hashcopy_tmp_cleaned_up_at_run_start():
    """rc-review fix, LOW (task 6, bullet 2): a `.hashcopy.*.tmp` file
    left behind by a hard kill mid-copy (every normal exit path already
    either os.replace()s or os.unlink()s it) must be swept up the next
    time run_once starts, once it's older than 1h — never touched while
    it could still be a genuinely in-flight copy."""
    uploads, root = _new_dirs("t25")
    objects_dir = root / uploads_backup.OBJECTS_DIRNAME
    objects_dir.mkdir(parents=True)
    stale = objects_dir / ".hashcopy.deadbeef.tmp"
    stale.write_bytes(b"orphaned" * 10)
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    fresh = objects_dir / ".hashcopy.cafef00d.tmp"
    fresh.write_bytes(b"in-flight" * 10)

    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(not stale.exists(), "the stale (>1h) .hashcopy.*.tmp file was removed at run start")
    assert_true(fresh.exists(), "a fresh (<1h) .hashcopy.*.tmp file is left alone (could be genuinely in-flight)")


def test_status_json_carries_copy_errors():
    """rc-review fix, LOW (task 5's status.json shape): status.json must
    include `copy_errors` (a new field this fix round added) so the
    operator-facing `status` command in scripts/uploads-backup.py can
    show it. Induces a real copy_errors > 0 (an unreadable NEW file) and
    checks the persisted status.json directly."""
    uploads, root = _new_dirs("t26")
    good = uploads / "good.jpg"
    bad = uploads / "bad.jpg"
    _write(good, b"\xff\xd8\xff" + b"ok" * 20, mtime=1_700_022_000)
    _write(bad, b"\xff\xd8\xff" + b"bad" * 20, mtime=1_700_022_100)
    # `chmod 000` only actually blocks reads for a non-root caller; this
    # suite also runs inside a root gate container, where root bypasses
    # DAC bits entirely. Monkeypatch `_hash_and_copy_object` itself (as
    # `run_once` calls it) to raise OSError for exactly the "bad" path,
    # deterministic regardless of UID, real behavior for every other path.
    real_copy = uploads_backup._hash_and_copy_object

    def _copy_bad_fails(path, *a, **k):
        if Path(path) == bad:
            raise PermissionError(13, "Permission denied", str(bad))
        return real_copy(path, *a, **k)

    uploads_backup._hash_and_copy_object = _copy_bad_fails
    try:
        r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    finally:
        uploads_backup._hash_and_copy_object = real_copy
    assert_true(r["copy_errors"] >= 1, "CONTROL: at least one copy_errors was induced")
    status = uploads_backup._read_status(root)
    assert_true(status is not None, "status.json was written")
    assert_eq(status.get("copy_errors"), r["copy_errors"], "status.json's copy_errors matches the report")


def test_cli_accept_flags_listing_unreadable_files_as_failed_roll_forward():
    """rc-review fix, LOW (task 4, probe accunread.sh): a tracked, CURRENT
    (not vanished) file that `accept` cannot READ during its own LISTING
    pass (permission denied right then, e.g.) used to just be silently
    dropped — absent from `to_roll_forward`, absent from any count,
    `accept` could clear `last_suspect_run` and exit 0 as if everything
    had actually been rolled forward. It must instead be folded into
    `failed_roll_forward`: `accept` exits 1 and leaves the suspect marker
    set.

    MUTATION CHECK: reverting the listing loop's `except OSError:
    continue` to drop the name (not append to unreadable_at_listing)
    makes this test fail — `accept --yes` would exit 0 and clear
    last_suspect_run despite the unreadable file.
    """
    uploads, root = _new_dirs("t27")
    for i in range(20):
        _write(uploads / f"u{i}.jpg", b"\xff\xd8\xff" + f"u{i}".encode() * 20, mtime=1_700_023_000 + i)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)

    # A legitimate mass content edit (settled, old mtimes) trips the guard.
    edited_names = [f"u{i}.jpg" for i in range(10)]
    for name in edited_names:
        p = uploads / name
        with open(p, "ab") as f:
            f.write(b"EDITED")
        os.utime(p, (time.time() - 86400, time.time() - 86400))
    r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r["suspect"], "the 50% content edit trips the guard")

    blocked = uploads / edited_names[0]
    # Same root-vs-non-root concern as test_status_json_carries_copy_errors
    # above: monkeypatch the script's own `_sha256_file` (what `accept`'s
    # listing loop actually calls) to raise for exactly the "blocked" path,
    # rather than relying on `chmod 000` actually being enforced.
    script_ns = _load_script()
    real_sha256_file = script_ns._sha256_file

    def _sha256_file_blocked(path, *a, **k):
        if Path(path) == blocked:
            raise PermissionError(13, "Permission denied", str(blocked))
        return real_sha256_file(path, *a, **k)

    script_ns._sha256_file = _sha256_file_blocked
    try:
        args = script_ns.build_parser().parse_args(
            ["--backup-root", str(root), "--uploads-dir", str(uploads), "accept", "--yes", "--force-large"]
        )
        rc = script_ns.cmd_accept(args)
    finally:
        script_ns._sha256_file = real_sha256_file

    assert_eq(rc, 1, "accept exits 1 — one changed file was unreadable during listing")
    manifest_after = uploads_backup.load_manifest(root)
    assert_true(manifest_after["last_suspect_run"] is not None,
                "last_suspect_run stays set — accept did NOT fully do what was asked")
    # The other 9 readable, genuinely-changed files DID roll forward.
    rolled = sum(
        1 for name in edited_names[1:]
        if manifest_after["files"][name].get("cur_sha256") != uploads_backup.load_manifest(root)["files"][name].get("sha256")
    )
    assert_eq(rolled, 9, "the 9 readable changed files WERE rolled forward despite the 1 unreadable one")


def test_cli_accept_roll_forward_repairs_a_damaged_existing_object():
    """rc-review fix, LOW (task 6, bullet 3, scripts/uploads-backup.py
    :586): `_hash_and_copy_single_pass`'s old `created = not
    dest.is_file()` treated ANY existing file at the object path as
    already-good, with no size check — a truncated/damaged object at that
    path (this test damages it directly) would be silently accepted as
    correct forever, unlike compactor/uploads_backup.py's own
    `has_object(..., size=...)`, which this fix mirrors."""
    script_ns = _load_script()
    tmp = _TMP / "t28-standalone"
    tmp.mkdir(parents=True, exist_ok=True)
    root = tmp / "bk"
    root.mkdir()
    src = tmp / "f.jpg"
    _write(src, b"A" * 200, mtime=time.time() - 1000)

    result = script_ns._hash_and_copy_single_pass(src, root, settle_cutoff=time.time() - 300)
    assert_true(result is not None, "CONTROL: the first pass writes the object")
    sha, size, created, *_ = result
    assert_true(created, "CONTROL: first write is genuinely new")
    dest = script_ns._object_path(root, sha)
    assert_eq(dest.stat().st_size, 200, "CONTROL: the object is the right size")

    # Damage it (truncate) as if it had been corrupted on disk.
    dest.write_bytes(b"A" * 50)

    result2 = script_ns._hash_and_copy_single_pass(src, root, settle_cutoff=time.time() - 300)
    sha2, size2, created2, *_ = result2
    assert_eq(sha2, sha, "same source, same hash")
    assert_true(not created2, "not double-counted as a new object (it already existed, even if damaged)")
    assert_eq(dest.stat().st_size, 200, "the damaged object was REPAIRED (overwritten with the correct bytes)")


def test_cli_status_reads_status_json_ok_last_error_and_stale():
    """rc-review fix, LOW (task 5): the `status` CLI command must read
    status.json (not just manifest.json) and expose ok/last_error/
    copy_errors/skipped counts/degraded/stale, with rc=1 when the run
    was not ok OR degraded OR stale — not only when suspect."""
    script_ns = _load_script()
    uploads, root = _new_dirs("t29")
    _write(uploads / "s.jpg", b"\xff\xd8\xff" + b"s" * 20, mtime=1_700_024_000)
    r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    assert_true(r["ok"], "CONTROL: a clean run")

    args = script_ns.build_parser().parse_args(["--backup-root", str(root), "--uploads-dir", str(uploads), "status"])
    rc_ok = script_ns.cmd_status(args)
    assert_eq(rc_ok, 0, "status exits 0 after a clean, non-suspect, non-stale run")

    # Force a failed status.json (never mutate the manifest for this —
    # status.json is what `status` must actually read).
    manifest = uploads_backup.load_manifest(root)
    uploads_backup._write_status(
        root, ok=False, error_type="SyntheticTestFailure",
        run_at=uploads_backup._iso(time.time()), run_s=0.1,
        copy_errors=3,
    )
    rc_bad = script_ns.cmd_status(args)
    assert_eq(rc_bad, 1, "status exits 1 once status.json itself says ok=False, even though nothing is suspect")

    # And a STALE (very old last_run_at), otherwise-ok status.
    uploads_backup._write_status(
        root, ok=True, error_type=None,
        run_at=uploads_backup._iso(time.time() - 999_999_999), run_s=0.1,
    )
    rc_stale = script_ns.cmd_status(args)
    assert_eq(rc_stale, 1, "status exits 1 once the last run is stale")


_SCRIPT_MODULE = None


def _load_script():
    global _SCRIPT_MODULE
    if _SCRIPT_MODULE is not None:
        return _SCRIPT_MODULE
    import importlib.util
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "uploads-backup.py"
    spec = importlib.util.spec_from_file_location("uploads_backup_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SCRIPT_MODULE = mod
    return mod


def test_sweep_never_expires_a_live_resurrected_name():
    """rc-fix verification, finding 1: a record whose deleted_at is past the keep
    window but whose file is LIVE again (restored after a long suspect period,
    when the sweep was blocked) must not be expired by the sweep. The old code
    popped it, then the rehash did files[rel], raised KeyError and saved nothing,
    so every later cycle failed the same way.

    MUTATION CHECK: dropping `rel not in current_names` from the sweep makes the
    run raise or report not-ok here.
    """
    uploads, root = _new_dirs("t40")
    target = uploads / "long_gone.jpg"
    original = b"\xff\xd8\xff" + b"long-gone-content"
    _write(target, original, mtime=1_700_030_000)
    for i in range(8):
        _write(uploads / f"keep_{i}.jpg", b"\xff\xd8\xff" + f"k{i}".encode(), mtime=1_700_030_010 + i)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    target.unlink()
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    manifest = uploads_backup.load_manifest(root)
    manifest["files"]["long_gone.jpg"]["deleted_at"] = uploads_backup._iso(
        time.time() - (uploads_backup.KEEP_DELETED_DAYS + 1) * 86400.0)
    uploads_backup.save_manifest(root, manifest)
    _write(target, original, mtime=1_700_030_100)
    for _ in range(2):
        r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
        assert_true(r["ok"], "the run stays ok (no KeyError loop)")
    rec = uploads_backup.load_manifest(root)["files"].get("long_gone.jpg")
    assert_true(rec is not None and rec["deleted_at"] is None, "the live file is tracked again, not expired")
    assert_true(uploads_backup._object_path(root, rec["sha256"]).is_file(), "its object is still there")


def test_sweep_is_saved_even_when_the_cap_still_refuses():
    """rc-fix verification, finding 2: the sweep deletes objects on disk. If the
    size cap still refuses afterwards, the manifest must already reflect the
    sweep, or the expired records point at deleted objects and objects_bytes
    stays too high for good.

    MUTATION CHECK: removing the save after the sweep makes objects_bytes
    disagree with the bytes on disk here.
    """
    uploads, root = _new_dirs("t41")
    for i in range(6):
        _write(uploads / f"d{i}.jpg", b"\xff\xd8\xff" + os.urandom(4000), mtime=1_700_031_000 + i)
    uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    for i in range(3):
        (uploads / f"d{i}.jpg").unlink()
    manifest = uploads_backup.load_manifest(root)
    old_iso = uploads_backup._iso(time.time() - (uploads_backup.KEEP_DELETED_DAYS + 1) * 86400.0)
    for i in range(3):
        manifest["files"][f"d{i}.jpg"]["deleted_at"] = old_iso
    uploads_backup.save_manifest(root, manifest)
    _write(uploads / "new.jpg", b"\xff\xd8\xff" + os.urandom(4000), mtime=1_700_031_100)

    saved = uploads_backup._require_mirror_size_cap
    def _always_refuse(*a, **k):
        raise uploads_backup._Refusal("test: cap refuses")
    uploads_backup._require_mirror_size_cap = _always_refuse
    try:
        r = uploads_backup.run_once(uploads, root, idle_check=_idle_ok, sleep=lambda s: None)
    finally:
        uploads_backup._require_mirror_size_cap = saved
    assert_true(not r["ok"], "CONTROL: the cap refused this run")
    m = uploads_backup.load_manifest(root)
    for i in range(3):
        assert_true(f"d{i}.jpg" not in m["files"], "the expired records are gone from the saved manifest")
    on_disk = sum(p.stat().st_size for p in (root / "objects").rglob("*") if p.is_file() and not p.name.startswith("."))
    assert_eq(m["objects_bytes"], on_disk, "objects_bytes matches the bytes on disk")


def _all():
    return [
        test_first_run_mirrors_everything_second_run_copies_nothing,
        test_new_file_gets_copied,
        test_sweep_never_expires_a_live_resurrected_name,
        test_sweep_is_saved_even_when_the_cap_still_refuses,
        test_content_change_below_threshold_never_overwrites_sha,
        test_deleted_file_kept_then_expired,
        test_gzip_incident_guard_trips_and_restore_recovers_byte_identical,
        test_check_db_against_synthetic_sqlite_file_table,
        test_also_mismatched_quarantines,
        test_idle_gating_skips_a_busy_cycle,
        test_idle_gate_http_error_fails_closed_and_mid_run_recheck_pauses,
        test_settle_window_skips_a_recently_written_upload,
        test_object_hash_while_copy_is_self_consistent,
        test_restore_refuses_a_damaged_object,
        test_restore_rejects_path_traversal_and_absolute_names,
        test_symlinks_and_non_regular_files_are_never_mirrored,
        test_mirror_size_cap_refuses_new_copies_over_the_limit,
        test_no_upload_name_ever_appears_in_logs_or_health_after_an_induced_error,
        test_resurrected_file_is_undeleted_and_rehashed,
        test_pre_read_race_caught_via_settle_and_classify_sig,
        test_accept_gate_uses_marker_baseline_not_diluted_recount,
        test_accept_does_not_clear_suspect_when_roll_forward_fails,
        test_guard_stays_suspect_across_runs_c2,
        test_restore_include_deleted,
        test_hash_and_copy_detects_inplace_rewrite_race,
        test_daemon_uploads_step_survives_base_exception,
        test_uploads_step_is_wired_into_the_daemon_loop_only,
        test_root_walk_unreadable_aborts_with_no_vanished_accounting,
        test_expiry_sweep_runs_before_mirror_cap_refusal,
        test_stale_hashcopy_tmp_cleaned_up_at_run_start,
        test_status_json_carries_copy_errors,
        test_cli_accept_flags_listing_unreadable_files_as_failed_roll_forward,
        test_cli_accept_roll_forward_repairs_a_damaged_existing_object,
        test_cli_status_reads_status_json_ok_last_error_and_stale,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            print(f"\n[{t.__name__}]")
            t()
        print()
        if FAILED:
            print(f"{FAILED} FAILED")
            sys.exit(1)
        print("All uploads-backup tests passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
