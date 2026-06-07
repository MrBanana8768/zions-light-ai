"""
CPU-only Tier-1 tests for compactor.backup (V2.3 Theme 1).

The point of this release is the FAILURE paths, so they get the most
coverage: unverifiable archives are rejected and deleted, the disk-full
guard trips, restore refuses bad/unconfirmed input.

Sets DATA_DIR / STORAGE_ROOT / BACKUP_DIR to a tmp tree BEFORE importing
backup so module-level config points at the sandbox. Uses a real SQLite
db so the live-snapshot path is actually exercised.

Run: python test_backup.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="zions-backup-test-"))
_DATA = _TMP / "data" / "openwebui"
_STORE = _DATA / "compactor"
_BACKUPS = _TMP / "data" / "backups"
_DB = _DATA / "webui.db"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BACKUPS)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DB)
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"

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


def assert_raises(fn, exc, label):
    try:
        fn()
    except exc:
        print(f"  ok   {label}")
        return
    except Exception as e:
        print(f"FAIL {label}: expected {exc.__name__}, got {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"FAIL {label}: nothing raised")
    sys.exit(1)


def _seed_sources(*, with_db=True, facts_text="seed fact"):
    """Create a realistic source tree: a live-ish sqlite db + memory files."""
    if _STORE.exists():
        shutil.rmtree(_STORE)
    (_STORE / "facts").mkdir(parents=True, exist_ok=True)
    (_STORE / "summaries").mkdir(parents=True, exist_ok=True)
    (_STORE / "facts" / "conv1.json").write_text(
        json.dumps({"conv_id": "conv1", "facts": [{"text": facts_text}]}),
        encoding="utf-8",
    )
    if _DB.exists():
        _DB.unlink()
    if with_db:
        _DATA.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(_DB))
        con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
        con.execute("INSERT INTO chat (body) VALUES ('hello')")
        con.commit()
        con.close()


def _clean_backups():
    if _BACKUPS.exists():
        shutil.rmtree(_BACKUPS)


# ---------------------------------------------------------------------------
# Happy path + round-trip
# ---------------------------------------------------------------------------

def test_create_verify_publish_round_trip():
    print("\n[test] run_once: create → verify → publish, no .partial left behind")
    _seed_sources()
    _clean_backups()
    rep = backup.run_once()
    assert_true(rep["ok"], "run_once ok")
    assert_true(rep["verified"], "verified flag set")
    assert_true(rep["archive"].endswith(".tar.gz"), "archive is .tar.gz")
    # No leftover .partial
    partials = list(_BACKUPS.glob("*.partial"))
    assert_eq(partials, [], "no .partial files remain")
    # Archive actually exists and verifies on its own
    arch = _BACKUPS / rep["archive"]
    ok, _ = backup.verify_backup(arch)
    assert_true(ok, "published archive independently verifies")


def test_restore_round_trip_recovers_data():
    print("\n[test] restore: wipe sources, restore from archive, data comes back")
    _seed_sources(facts_text="precious memory")
    _clean_backups()
    rep = backup.run_once()
    arch = _BACKUPS / rep["archive"]

    # Simulate disaster: wipe the live store + db
    shutil.rmtree(_STORE)
    _DB.unlink()
    assert_true(not _STORE.exists(), "store wiped")

    res = backup.restore_backup(arch, confirm=True)
    assert_true(res["ok"], "restore ok")
    assert_true("compactor" in res["restored"], "compactor restored")
    assert_true("webui.db" in res["restored"], "webui.db restored")
    # The precious fact is back
    data = json.loads((_STORE / "facts" / "conv1.json").read_text())
    assert_eq(data["facts"][0]["text"], "precious memory", "fact content recovered")
    # The db is back and openable
    con = sqlite3.connect(str(_DB))
    n = con.execute("SELECT COUNT(*) FROM chat").fetchone()[0]
    con.close()
    assert_eq(n, 1, "db row recovered")


def test_backup_without_db_succeeds():
    print("\n[test] missing webui.db → backup still succeeds (memory-only)")
    _seed_sources(with_db=False)
    _clean_backups()
    rep = backup.run_once()
    assert_true(rep["ok"], "memory-only backup ok")
    ok, detail = backup.verify_backup(_BACKUPS / rep["archive"])
    assert_true(ok, "verifies")
    assert_true("absent" in detail, "detail notes db absent")


# ---------------------------------------------------------------------------
# Failure paths — the heart of this release
# ---------------------------------------------------------------------------

def test_verify_rejects_truncated_archive():
    print("\n[test] verify: a truncated/garbage archive fails cleanly (no raise)")
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    bad = _BACKUPS / "zions-backup-garbage.tar.gz"
    bad.write_bytes(b"this is not a gzip tar at all")
    ok, detail = backup.verify_backup(bad)
    assert_eq(ok, False, "garbage archive → not ok")
    assert_true("extract failed" in detail, "detail explains extract failure")


def test_verify_rejects_corrupt_memory_json():
    print("\n[test] verify: a corrupt memory JSON inside the archive fails")
    # Hand-build an archive with a broken facts file
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp())
    try:
        (staging / "compactor" / "facts").mkdir(parents=True)
        (staging / "compactor" / "facts" / "bad.json").write_text("{not valid json")
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "v1", "sources": {"webui.db": {"present": False},
                                         "compactor": {"present": True}},
        }))
        arch = _BACKUPS / "zions-backup-corruptjson.tar.gz"
        with tarfile.open(arch, "w:gz") as tar:
            tar.add(staging, arcname=".")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "corrupt json → not ok")
    assert_true("corrupt memory file" in detail, "detail names the failure")


def test_verify_rejects_bad_sqlite():
    print("\n[test] verify: manifest claims db present but it's not a real db")
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp())
    try:
        (staging / "webui.db").write_text("definitely not sqlite")
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "v1", "sources": {"webui.db": {"present": True}},
        }))
        arch = _BACKUPS / "zions-backup-baddb.tar.gz"
        with tarfile.open(arch, "w:gz") as tar:
            tar.add(staging, arcname=".")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "bad db → not ok")
    assert_true("sqlite" in detail.lower() or "integrity" in detail.lower(),
                "detail mentions sqlite/integrity")


def test_verify_rejects_missing_manifest():
    print("\n[test] verify: archive with no manifest fails")
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp())
    try:
        (staging / "compactor").mkdir()
        arch = _BACKUPS / "zions-backup-nomanifest.tar.gz"
        with tarfile.open(arch, "w:gz") as tar:
            tar.add(staging, arcname=".")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "no manifest → not ok")
    assert_true("manifest" in detail, "detail mentions manifest")


def test_run_once_discards_unverifiable_archive():
    print("\n[test] run_once: if verify fails, archive is deleted + FAIL reported")
    _seed_sources()
    _clean_backups()
    # Force verification to fail
    orig = backup.verify_backup
    backup.verify_backup = lambda p: (False, "forced failure")
    try:
        rep = backup.run_once()
    finally:
        backup.verify_backup = orig
    assert_eq(rep["ok"], False, "run reports failure")
    assert_true("VERIFICATION FAILED" in rep["detail"], "detail flags verification failure")
    # No archive (and no .partial) left — no false confidence
    leftovers = list(_BACKUPS.glob("*.tar.gz*"))
    assert_eq(leftovers, [], "unverifiable archive discarded, nothing left")


def test_min_free_guard_blocks_backup():
    print("\n[test] create_backup: disk-full guard raises rather than filling /data")
    _seed_sources()
    _clean_backups()
    orig = backup._free_mb
    backup._free_mb = lambda p: 1.0  # pretend nearly full
    try:
        assert_raises(lambda: backup.create_backup(), RuntimeError, "min-free guard trips")
    finally:
        backup._free_mb = orig


# ---------------------------------------------------------------------------
# Prune / restore gating / info
# ---------------------------------------------------------------------------

def test_prune_keeps_newest_n():
    print("\n[test] prune: keeps newest RETAIN (3), removes older")
    _seed_sources()
    _clean_backups()
    # Make 5 successful backups
    names = []
    for _ in range(5):
        import time
        time.sleep(1.05)  # ensure distinct YYYYmmdd-HHMMSS stamps
        rep = backup.run_once()
        assert_true(rep["ok"], "backup ok")
        names.append(rep["archive"])
    remaining = backup.list_backups()
    assert_eq(len(remaining), 3, "only 3 retained")
    # The 3 retained are the 3 newest
    kept = {r["name"] for r in remaining}
    assert_true(kept == set(names[-3:]), "the 3 newest are kept")


def test_restore_requires_confirm():
    print("\n[test] restore: refuses without confirm=True")
    _seed_sources()
    _clean_backups()
    rep = backup.run_once()
    arch = _BACKUPS / rep["archive"]
    assert_raises(lambda: backup.restore_backup(arch, confirm=False),
                  RuntimeError, "unconfirmed restore raises")


def test_restore_refuses_unverifiable_archive():
    print("\n[test] restore: refuses an archive that doesn't verify")
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    bad = _BACKUPS / "zions-backup-bad.tar.gz"
    bad.write_bytes(b"garbage")
    assert_raises(lambda: backup.restore_backup(bad, confirm=True),
                  RuntimeError, "unverifiable restore raises")


def test_latest_backup_info_shape():
    print("\n[test] latest_backup_info: count + latest after a backup")
    _seed_sources()
    _clean_backups()
    info0 = backup.latest_backup_info()
    assert_eq(info0["count"], 0, "zero before any backup")
    assert_eq(info0["latest"], None, "no latest")
    backup.run_once()
    info1 = backup.latest_backup_info()
    assert_eq(info1["count"], 1, "one after backup")
    assert_true(info1["latest"] is not None, "latest set")
    assert_true(info1["latest_mtime"] is not None, "mtime set")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all():
    return [
        test_create_verify_publish_round_trip,
        test_restore_round_trip_recovers_data,
        test_backup_without_db_succeeds,
        test_verify_rejects_truncated_archive,
        test_verify_rejects_corrupt_memory_json,
        test_verify_rejects_bad_sqlite,
        test_verify_rejects_missing_manifest,
        test_run_once_discards_unverifiable_archive,
        test_min_free_guard_blocks_backup,
        test_prune_keeps_newest_n,
        test_restore_requires_confirm,
        test_restore_refuses_unverifiable_archive,
        test_latest_backup_info_shape,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            t()
        print("\nAll backup smoke tests passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
