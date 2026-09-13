"""
p3-b (hostile pass #3, reviewer B) F13 — a SIGKILL during create_backup
leaves staging and .partial files that are never cleaned.

Run inside the compactor image or any container with the requirements
installed:
    python test_p3b_stale_debris.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-p3b-debris-test-"))
_DATA = _ROOT / "openwebui"
_BK = _ROOT / "backups"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_DATA / "compactor")
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BK)
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"
os.environ["COMPACTOR_BACKUP_STALE_DEBRIS_HOURS"] = "2"

import backup  # noqa: E402


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _age(p: Path, hours: float) -> None:
    t = time.time() - hours * 3600
    os.utime(p, (t, t))


def test_stale_staging_dir_and_partial_are_swept():
    print("\n[test] F13: a stale staging dir and .partial file are removed at the next run_once")
    _BK.mkdir(parents=True, exist_ok=True)
    stale_dir = _BK / "zions-backup-20260101-000000-abc123"
    stale_dir.mkdir()
    (stale_dir / "manifest.json").write_bytes(b"{}")
    _age(stale_dir, 3)

    stale_partial = _BK / "zions-backup-20260101-000000.tar.gz.partial"
    stale_partial.write_bytes(b"partial data")
    _age(stale_partial, 3)

    real_archive = _BK / "zions-backup-20260102-000000.tar.gz"
    real_archive.write_bytes(b"a real archive, untouched regardless of age")
    _age(real_archive, 3)

    removed = backup._sweep_stale_backup_debris(_BK)
    assert_true(stale_dir.name in removed, f"the stale staging dir was removed (got {removed})")
    assert_true(stale_partial.name in removed, f"the stale .partial was removed (got {removed})")
    assert_true(not stale_dir.exists(), "the staging dir is actually gone")
    assert_true(not stale_partial.exists(), "the partial is actually gone")
    assert_true(real_archive.exists(), "F13 fix: a real, PUBLISHED archive is never touched, regardless of age")


def test_control_recent_debris_is_left_alone():
    print("\n[test] F13 CONTROL: a staging dir/partial from a run IN PROGRESS right now is not swept")
    _BK.mkdir(parents=True, exist_ok=True)
    fresh_dir = _BK / "zions-backup-20260913-000000-xyz789"
    fresh_dir.mkdir()
    fresh_partial = _BK / "zions-backup-20260913-000000.tar.gz.partial"
    fresh_partial.write_bytes(b"in progress")
    # freshly created — mtime is "now", well under the 2h cutoff

    removed = backup._sweep_stale_backup_debris(_BK)
    assert_true(fresh_dir.name not in removed, f"CONTROL: the fresh staging dir survives (got {removed})")
    assert_true(fresh_partial.name not in removed, f"CONTROL: the fresh partial survives (got {removed})")
    assert_true(fresh_dir.exists() and fresh_partial.exists(), "CONTROL: both are still on disk")


if __name__ == "__main__":
    tests = [
        test_stale_staging_dir_and_partial_are_swept,
        test_control_recent_debris_is_left_alone,
    ]
    for t in tests:
        t()
    print("\nAll p3-b stale-debris (F13) tests passed.")
