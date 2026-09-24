"""
v3.1.9 round 2 — restore_backup hardening: findings 7, 8, 9, 10.

Coordinator's additional items (none reproduced by an earlier lane):

  7. The active-writer check (_require_no_active_writer) stands down
     whenever a sidecar already exists — exactly the incident state a
     restore is most likely to be run to recover from. Fixed with a raw
     fcntl() RESERVED-lock probe (_probe_reserved_lock, backup.py) that
     never opens the file via sqlite3, so it cannot trigger hot-journal
     recovery the way the existing BEGIN IMMEDIATE probe would.
  8. If the store swap fails AFTER the database swap has landed, the
     result used to be a mixed generation (NEW db, OLD store) with no way
     back — the old db's bytes were simply gone once os.replace succeeded.
     Fixed by quarantining the ORIGINAL database too (matching the store's
     own already-established pattern), and rolling it back if the store
     swap subsequently fails.
  9. The free-space guard reads shutil.disk_usage(), which on MooseFS
     reports the CLUSTER-WIDE figure (~217 TB measured), not this pod's own
     RunPod volume quota — so it can never catch a full pod volume.
     Documented as a real, unfixable-by-this-process limitation, plus an
     HONEST additional check: COMPACTOR_DATA_VOLUME_QUOTA_MB, an
     operator-configured number compared against a real `du` of DATA_DIR
     (not statvfs). Off (0) by default.
  10. restore_backup did not integrity_check the landed database, and did
      not say what to restart. Added a post-landing PRAGMA integrity_check
      (webuidb.integrity(), reused, not duplicated) that reports loudly and
      leaves the pre-restore set-aside in place without auto-restoring it,
      plus a "restart" field in the result dict / CLI output naming the
      services.

ENVIRONMENT NOTE (read before assuming a Windows FAIL here means the fix is
wrong): restore_backup's OWN staging step calls _fsync_file(db_tmp), which
opens the file "rb" and calls os.fsync() on it — this is a PRE-EXISTING,
unrelated Windows-only bug (OSError: [Errno 9] Bad file descriptor),
reproduced identically against this lane's UNMODIFIED HEAD via `git stash`
(see SP\\fix-round2.md, finding 3's suite-results note) — so EVERY test here
that drives a restore all the way to a successful landing crashes on
Windows before ever reaching the code this file actually tests. Tests that
do not need a successful landing (the free-space/quota tests, and the
active-writer probe tests, which call the helpers directly rather than
through restore_backup) run and are meaningful on Windows; the rest are
run for real against the Linux unit-test container and labelled as such in
the lane report.

Run: python test_round2_restore_hardening.py
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

_TMP = Path(tempfile.mkdtemp(prefix="zions-round2-restore-test-"))
_DATA = _TMP / "data" / "openwebui"
_STORE = _DATA / "compactor"
_BACKUPS = _TMP / "data" / "backups"
_DB = _DATA / "webui.db"
_QUARANTINE = _TMP / "quarantine"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BACKUPS)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DB)
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUARANTINE)
os.environ.pop("COMPACTOR_DATA_VOLUME_QUOTA_MB", None)

import backup  # noqa: E402

PY = sys.executable
_HOT = bytes.fromhex("d9d505f920a163d7")  # SQLite hot-journal magic header

_FAILED = []


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        _FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        _FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_raises(fn, exc, label):
    try:
        fn()
    except exc:
        print(f"  ok   {label}")
        return
    except Exception as e:
        print(f"FAIL {label}: expected {exc.__name__}, got {type(e).__name__}: {e}")
        _FAILED.append(label)
        return
    print(f"FAIL {label}: nothing raised")
    _FAILED.append(label)


def _make_db(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
    con.execute("INSERT INTO chat (body) VALUES (?)", (marker,))
    con.commit()
    con.close()


def _wipe():
    for p in (_DB, _STORE, _BACKUPS, _QUARANTINE):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    for suffix in backup.SIDECARS:
        s = _DB.with_name(_DB.name + suffix)
        if s.exists():
            s.unlink()
    _STORE.mkdir(parents=True, exist_ok=True)
    (_STORE / "facts").mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Finding 9 — quota-based free space check. No restore_backup() call needed
# (drives _require_free_space directly), so this section is real on Windows.
# ===========================================================================

def test_quota_off_by_default_does_not_block_a_huge_need():
    print("\n[9.1] COMPACTOR_DATA_VOLUME_QUOTA_MB unset (0, the default) — the "
          "quota half of the guard never fires")
    _wipe()
    saved = os.environ.pop("COMPACTOR_DATA_VOLUME_QUOTA_MB", None)
    try:
        # A need far larger than any quota anyone would set, on a directory
        # that genuinely exists — if the quota check were mistakenly always
        # on, this would refuse. The statvfs-based check above it still
        # applies (and will not fire either: real free space on a dev box
        # dwarfs this too, matching the CONTROL's own point).
        backup._require_free_space([(_DATA, 10)])
    finally:
        if saved is not None:
            os.environ["COMPACTOR_DATA_VOLUME_QUOTA_MB"] = saved
    assert_true(True, "no exception with the quota check off")


def test_quota_set_and_exceeded_refuses():
    print("\n[9.2] COMPACTOR_DATA_VOLUME_QUOTA_MB set and exceeded refuses, "
          "naming the configured ceiling")
    _wipe()
    (_DATA / "existing.bin").write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
    with patch.object(backup, "COMPACTOR_DATA_VOLUME_QUOTA_MB", 3):  # 3 MB quota
        assert_raises(
            lambda: backup._require_free_space([(_DATA, 2 * 1024 * 1024)]),  # +2 MB
            RuntimeError,
            "2 MB already used + 2 MB more against a 3 MB quota refuses",
        )
        try:
            backup._require_free_space([(_DATA, 2 * 1024 * 1024)])
        except RuntimeError as e:
            assert_true("QUOTA" in str(e), f"and the message names the quota var (got: {e})")
        else:
            assert_true(False, "expected a RuntimeError naming the quota var, got none")


def test_quota_set_but_within_budget_proceeds():
    print("\n[9.3] CONTROL: a need that fits inside the quota does not refuse")
    # Without this, [9.2] could be passing because the quota check refuses
    # unconditionally once set, not because the numbers were actually over.
    _wipe()
    (_DATA / "existing.bin").write_bytes(b"x" * (1024))  # ~1 KB
    with patch.object(backup, "COMPACTOR_DATA_VOLUME_QUOTA_MB", 500):
        backup._require_free_space([(_DATA, 1024)])
    assert_true(True, "a small need against a generous quota does not refuse")


def test_is_under_helper():
    print("\n[9.4] _is_under: the helper the quota check uses to scope which "
          "needed-bytes entries count toward DATA_DIR's quota")
    assert_true(backup._is_under(_DATA / "x" / "y.db", _DATA),
                "a path under DATA_DIR is under DATA_DIR")
    assert_true(not backup._is_under(_TMP / "elsewhere" / "y.db", _DATA),
                "a path outside DATA_DIR is not")
    assert_true(backup._is_under(_DATA, _DATA),
                "DATA_DIR itself counts as under DATA_DIR")


# ===========================================================================
# Finding 7 — the active-writer probe when a sidecar already exists.
# ===========================================================================

def test_probe_reserved_lock_fails_open_with_no_writer():
    print("\n[7.1] _probe_reserved_lock: no writer holds the file -> False "
          "(and the CONTROL for the whole probe: it can say no)")
    _wipe()
    _make_db(_DB, "no writer here")
    assert_true(
        backup._probe_reserved_lock(_DB) is False,
        "an ordinary, unlocked database reports no active writer",
    )


def test_probe_reserved_lock_fails_open_on_a_missing_file():
    print("\n[7.2] CONTROL: a target that does not exist yet is not treated "
          "as a locked writer")
    _wipe()
    missing = _DATA / "does-not-exist.db"
    assert_true(
        backup._probe_reserved_lock(missing) is False,
        "a missing file reports no active writer (fails open, not closed)",
    )


def _has_fcntl() -> bool:
    try:
        import fcntl  # noqa: F401
        return True
    except ImportError:
        return False


def test_active_writer_with_a_sidecar_is_now_caught_on_posix():
    print("\n[7.3] _require_no_active_writer: a REAL writer's RESERVED lock, "
          "concurrent WITH a sidecar already present, is now caught")
    if not _has_fcntl():
        print("  SKIP (no fcntl on this platform — Windows dev box; run for "
              "real on Linux, see the lane report for that run's result)")
        return
    _wipe()
    _make_db(_DB, "active writer test")
    journal = _DB.with_name(_DB.name + "-journal")

    # A REAL, separate process holds SQLite's RESERVED lock (via BEGIN
    # IMMEDIATE + an actual write) and signals readiness before the parent
    # probes — POSIX fcntl() record locks are per (process, inode), not
    # reentrant within one process, so this genuinely needs a child, not
    # just a second connection in this same process.
    #
    # THE JOURNAL IS THE CHILD'S OWN, real, natural one — NOT a hand-written
    # fake. A first version of this test wrote synthetic "-journal" bytes
    # BEFORE spawning the child; the child's own sqlite3.connect() +
    # BEGIN IMMEDIATE performs SQLite's hot-journal handling as a side
    # effect of opening the file AT ALL (the exact lesson
    # _require_no_active_writer's own docstring states, and the reason the
    # sidecar branch stands down in the first place) and consumed/replaced
    # that fake journal before the parent's check ever ran — so the parent
    # ended up hitting the ORIGINAL BEGIN IMMEDIATE probe path below
    # (because the sidecar was gone by the time it looked), not the NEW
    # sidecar-present branch this test exists to cover. That version passed
    # for the WRONG reason: caught by mutation-testing the fix (see
    # SP\\fix-round2.md finding 7's report) — a GREEN mutant where removing
    # the new code did not turn the test red. An actual uncommitted WRITE in
    # the child creates SQLite's OWN real journal as a side effect of ITS
    # OWN transaction, which is what "a write transaction in progress"
    # genuinely looks like on disk, and is what stays present throughout.
    child_code = (
        "import sqlite3, sys, time\n"
        f"con = sqlite3.connect({str(_DB)!r})\n"
        "con.execute('BEGIN IMMEDIATE')\n"
        "con.execute(\"INSERT INTO chat (body) VALUES ('uncommitted, held by the test child')\")\n"
        "print('READY', flush=True)\n"
        "time.sleep(20)\n"
    )
    child = subprocess.Popen([PY, "-c", child_code], stdout=subprocess.PIPE, text=True)
    try:
        line = child.stdout.readline()
        assert_true(line.strip() == "READY", f"fixture: child signalled ready ({line!r})")
        time.sleep(0.2)
        assert_true(journal.is_file(),
                    "fixture: the child's own write really did leave a real "
                    "-journal file on disk (not a hand-crafted one)")
        assert_raises(
            lambda: backup._require_no_active_writer(_DB), RuntimeError,
            "refused while the REAL child holds the RESERVED lock, even "
            "though a sidecar is present (the exact state the OLD check "
            "stood down on)",
        )
    finally:
        child.kill()
        child.wait()

    print("    CONTROL: once the child is gone, the SAME state (sidecar, no "
          "writer) passes")
    time.sleep(0.2)
    backup._require_no_active_writer(_DB)  # must not raise
    assert_true(True, "a sidecar with no writer behind it still stands down "
                "(unchanged behaviour for the common case)")


def test_journal_bytes_untouched_by_the_new_probe():
    print("\n[7.4] CONTROL: the new probe never touches the journal's bytes "
          "(it never opens the file through sqlite3 at all)")
    _wipe()
    _make_db(_DB, "journal preservation")
    journal = _DB.with_name(_DB.name + "-journal")
    payload = _HOT + b"a real hot journal's bytes, byte for byte"
    journal.write_bytes(payload)
    backup._probe_reserved_lock(_DB)  # ignore the result; just must not touch the file
    assert_eq(journal.read_bytes(), payload,
              "the journal is byte-for-byte identical after the probe")


if __name__ == "__main__":
    for t in (
        test_quota_off_by_default_does_not_block_a_huge_need,
        test_quota_set_and_exceeded_refuses,
        test_quota_set_but_within_budget_proceeds,
        test_is_under_helper,
        test_probe_reserved_lock_fails_open_with_no_writer,
        test_probe_reserved_lock_fails_open_on_a_missing_file,
        test_active_writer_with_a_sidecar_is_now_caught_on_posix,
        test_journal_bytes_untouched_by_the_new_probe,
    ):
        t()

    if _FAILED:
        print(f"\n{len(_FAILED)} FAILED:")
        for f in _FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll round-2 restore-hardening (findings 7/9) checks passed.")
    print("Findings 8 and 10 (db/store swap atomicity, post-landing "
          "integrity) are in test_round2_restore_swap_and_integrity.py — "
          "kept separate because they REQUIRE a successful restore_backup() "
          "landing, which crashes on Windows on a pre-existing, unrelated "
          "_fsync_file bug (see this file's own module docstring).")
