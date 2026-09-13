"""
v3.1.9 round 2 — restore_backup hardening: findings 8 and 10.

Split out of test_round2_restore_hardening.py (which holds findings 7/9):
every test in THIS file needs a restore that actually LANDS (a real,
successful os.replace of the database), and restore_backup's OWN staging
step calls _fsync_file(db_tmp) — open(p, "rb") + os.fsync() — which is a
PRE-EXISTING, unrelated Windows-only bug (OSError: [Errno 9] Bad file
descriptor), reproduced identically against this lane's UNMODIFIED HEAD via
`git stash` (see SP\\fix-round2.md, finding 3's suite-results note). So this
whole file is Linux-only in practice; it still imports and parses cleanly on
Windows, and is run for real against the Linux unit-test container (see the
lane report for that run's result).

  8. If the store swap fails AFTER the database swap has landed, the
     result used to be a mixed generation (NEW db, OLD store) with no way
     back. Fixed by quarantining the ORIGINAL database too (matching the
     store's own already-established set-aside pattern), and rolling it
     back if the store swap subsequently fails.
  10. restore_backup did not integrity_check the landed database, and did
      not say what to restart. Added a post-landing PRAGMA integrity_check
      that reports loudly and leaves the pre-restore set-aside in place,
      plus a "restart" field naming the services.

Run (Linux, inside the compactor image or the unit-tests container):
    python test_round2_restore_swap_and_integrity.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

_TMP = Path(tempfile.mkdtemp(prefix="zions-round2-swap-test-"))
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
import webuidb  # noqa: E402

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


def _db_marker(path: Path) -> str:
    con = sqlite3.connect(str(path))
    try:
        return con.execute("SELECT body FROM chat LIMIT 1").fetchone()[0]
    finally:
        con.close()


def _wipe():
    for suffix in ("",) + backup.SIDECARS:
        p = _DB.with_name(_DB.name + suffix)
        if p.exists():
            p.unlink()
    if _STORE.exists():
        shutil.rmtree(_STORE, ignore_errors=True)
    if _QUARANTINE.exists():
        shutil.rmtree(_QUARANTINE, ignore_errors=True)
    _STORE.mkdir(parents=True, exist_ok=True)
    (_STORE / "facts").mkdir(parents=True, exist_ok=True)


def _make_archive(marker: str) -> Path:
    """A real, verified archive (via run_once — the actual writer, not a
    hand-built tarball) holding a webui.db with `marker` in it."""
    _wipe()
    _make_db(_DB, marker)
    (_STORE / "facts" / "conv1.json").write_text(
        json.dumps([{"text": marker, "ts": time.time()}]), encoding="utf-8",
    )
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    for old in _BACKUPS.glob("*.tar.gz"):
        old.unlink()
    rep = backup.run_once()
    assert rep["ok"], f"fixture archive creation failed: {rep}"
    return _BACKUPS / rep["archive"]


# ===========================================================================
# Finding 8 — db/store swap atomicity
# ===========================================================================

def test_a_normal_restore_lands_both_db_and_store():
    print("\n[8.1] CONTROL: an ordinary restore lands both pieces, unaffected "
          "by the new db-quarantine step")
    arch = _make_archive("ARCHIVED generation")
    _make_db(_DB, "stale LOCAL generation")  # something different, live
    rep = backup.restore_backup(arch, confirm=True)
    assert_true(rep["ok"], f"restore reports ok (got {rep})")
    assert_eq(sorted(rep["restored"]), ["compactor", "webui.db"],
              "both pieces restored")
    assert_eq(_db_marker(_DB), "ARCHIVED generation",
              "the live database now holds the ARCHIVED generation")


def test_store_swap_failure_after_db_swap_rolls_the_db_back():
    print("\n[8.2] the STORE swap failing AFTER the DB swap landed rolls the "
          "DB back too - no mixed generation left behind")
    arch = _make_archive("NEW generation (archived)")
    _make_db(_DB, "OLD generation (live before restore)")

    real_replace = os.replace

    def _fail_store_replace(src, dst):
        # Let the FIRST os.replace (the database swap) through untouched;
        # fail only the SECOND (the store swap).
        if str(dst) == str(_STORE):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    with patch.object(backup.os, "replace", _fail_store_replace):
        assert_raises(
            lambda: backup.restore_backup(arch, confirm=True), OSError,
            "the store swap failed and restore_backup said so",
        )
    assert_eq(
        _db_marker(_DB), "OLD generation (live before restore)",
        "the database is back to its PRE-RESTORE generation - not stuck at "
        "the new one with an old store beside it (the mixed-generation "
        "defect finding 8 names)",
    )
    assert_true(
        (_STORE / "facts" / "conv1.json").exists(),
        "sanity: the store directory still exists",
    )


def test_db_swap_failure_still_restores_the_original_db_content():
    print("\n[8.3] CONTROL: the DB's OWN os.replace failing still restores "
          "the pre-restore db (the new db_aside quarantine step must not "
          "break the EXISTING recovery path for this failure)")
    arch = _make_archive("NEW generation (archived)")
    _make_db(_DB, "OLD generation, must survive a failed db replace")

    real_replace = os.replace

    def _fail_db_replace(src, dst):
        if str(dst) == str(_DB):
            raise OSError(5, "Input/output error")
        return real_replace(src, dst)

    with patch.object(backup.os, "replace", _fail_db_replace):
        assert_raises(
            lambda: backup.restore_backup(arch, confirm=True), OSError,
            "the db swap failed and restore_backup said so",
        )
    assert_eq(
        _db_marker(_DB), "OLD generation, must survive a failed db replace",
        "the ORIGINAL database content is exactly what is live afterward",
    )
    assert_eq(
        sorted(_DATA.glob("webui.db.pre-restore-*")), [],
        "no set-aside was left BESIDE the target (A3-10 still holds: it "
        "would have landed in QUARANTINE, and on this failure path it was "
        "moved straight back, so nothing lands anywhere at all)",
    )


def test_a_completely_failed_store_swap_with_no_db_in_the_archive():
    print("\n[8.4] CONTROL: have_db False (archive holds no webui.db) - the "
          "new db-rollback code in the store's except block must not crash "
          "when there was never a db_aside to begin with")
    _wipe()
    (_STORE / "facts" / "conv1.json").write_text(
        json.dumps([{"text": "store-only archive", "ts": time.time()}]),
        encoding="utf-8",
    )
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    for old in _BACKUPS.glob("*.tar.gz"):
        old.unlink()
    rep = backup.run_once()
    assert rep["ok"], f"fixture archive creation failed: {rep}"
    arch = _BACKUPS / rep["archive"]

    _make_db(_DB, "must survive - not part of this archive")

    real_replace = os.replace

    def _fail_store_replace(src, dst):
        if str(dst) == str(_STORE):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    with patch.object(backup.os, "replace", _fail_store_replace):
        assert_raises(
            lambda: backup.restore_backup(arch, confirm=True), OSError,
            "the store-only restore's store swap failed and said so, no crash",
        )
    assert_eq(_db_marker(_DB), "must survive - not part of this archive",
              "the live database (never part of this archive) is untouched")


# ===========================================================================
# Finding 10 — post-landing integrity check + "restart" naming
# ===========================================================================

def test_post_landing_integrity_check_passes_on_a_healthy_restore():
    print("\n[10.1] CONTROL: a healthy restore passes the new post-landing "
          "integrity_check and reports what to restart")
    arch = _make_archive("healthy generation")
    _make_db(_DB, "stale")
    rep = backup.restore_backup(arch, confirm=True)
    assert_true(rep["ok"], f"restore reports ok (got {rep})")
    assert_true(
        "restart" in rep and "supervisorctl" in rep["restart"],
        f"the result names the services to restart (got: {rep.get('restart')!r})",
    )
    for svc in ("openwebui", "compactor", "backup", "webuidb-sync"):
        assert_true(svc in rep["restart"], f"restart command names {svc}")


def test_post_landing_integrity_failure_is_reported_and_set_aside_kept():
    print("\n[10.2] a database that fails integrity_check immediately after "
          "landing is reported loudly, and its pre-restore set-aside is "
          "kept, not silently auto-restored")
    arch = _make_archive("archived generation")
    _make_db(_DB, "PRE-RESTORE original, must be preserved in quarantine")

    with patch.object(webuidb, "integrity", lambda p: (False, "simulated corruption")):
        assert_raises(
            lambda: backup.restore_backup(arch, confirm=True), RuntimeError,
            "the post-landing integrity_check failure is reported as a "
            "RuntimeError, not returned as ok=True",
        )
    asides = list(_QUARANTINE.glob("webui.db.pre-restore-*"))
    assert_true(
        len(asides) == 1,
        f"exactly one pre-restore database set-aside sits in quarantine "
        f"(got: {asides})",
    )
    # Guarded rather than indexing asides[0] directly: a mutant that removes
    # the db_aside quarantine step entirely (finding 8's own mutation) makes
    # `asides` EMPTY, and this check would otherwise crash with an
    # IndexError instead of failing cleanly on the check above — a
    # wrong-reason red caught by running that exact mutation (see
    # SP\\fix-round2.md finding 8/10's report).
    assert_eq(
        _db_marker(asides[0]) if asides else None,
        "PRE-RESTORE original, must be preserved in quarantine",
        "and it holds the REAL pre-restore content - nothing deleted or "
        "overwrote it",
    )


def test_post_landing_integrity_failure_does_not_touch_the_store():
    print("\n[10.3] CONTROL: when the post-landing db integrity check fails, "
          "the compactor store is never touched at all")
    arch = _make_archive("archived generation (store half)")
    _make_db(_DB, "pre-restore db")
    live_marker = "the LIVE store, must be untouched by a db-integrity failure"
    (_STORE / "facts" / "conv1.json").write_text(
        json.dumps([{"text": live_marker, "ts": time.time()}]), encoding="utf-8",
    )

    with patch.object(webuidb, "integrity", lambda p: (False, "simulated corruption")):
        assert_raises(
            lambda: backup.restore_backup(arch, confirm=True), RuntimeError,
            "restore refused on the integrity failure",
        )
    store_text = (_STORE / "facts" / "conv1.json").read_text(encoding="utf-8")
    assert_true(live_marker in store_text,
                "the live store's own content is exactly as it was - the "
                "store swap never ran")
    assert_eq(sorted(_DATA.glob("compactor.incoming-*")), [],
              "and no staged store copy was left behind either")


if __name__ == "__main__":
    for t in (
        test_a_normal_restore_lands_both_db_and_store,
        test_store_swap_failure_after_db_swap_rolls_the_db_back,
        test_db_swap_failure_still_restores_the_original_db_content,
        test_a_completely_failed_store_swap_with_no_db_in_the_archive,
        test_post_landing_integrity_check_passes_on_a_healthy_restore,
        test_post_landing_integrity_failure_is_reported_and_set_aside_kept,
        test_post_landing_integrity_failure_does_not_touch_the_store,
    ):
        t()

    if _FAILED:
        print(f"\n{len(_FAILED)} FAILED:")
        for f in _FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll round-2 restore swap-atomicity/integrity (findings 8/10) "
          "checks passed.")
