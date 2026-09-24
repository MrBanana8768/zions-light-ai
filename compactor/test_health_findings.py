"""v3.1.9 remaining health findings (fix-health lane, closing the v3.1.9 line).

Each test drives a state production actually produces (real backup.py /
webuidb.py / summarizer.py writers wherever practical) and reads it back
through gather_health_full, the function /health/full and /admin/selftest
both call.

  F1  hostile317-c F4 (HIGH): zero or stale backups read "ok" (the
      unreadable-memory half of F4 was already fixed in v3.1.8; this is the
      backups half). Grace window for a freshly booted pod is tested both
      sides.
  F2  hostile2-backup A3-8 (MEDIUM): the journal probe watched only
      WEBUI_SNAPSHOT_DB and missed the live database under the shipped
      default (WEBUI_DB_LOCAL=true).
  F3a health H-6 (MEDIUM): worst_lag had no recency filter — an ancient
      conversation's frozen lag masked a live one.
  F3b health H-7 (MEDIUM): _lag_limit inherited env_int's unranged contract.
  F3c also closes the OPEN_ISSUES2 MEDIUM "the stale-snapshot status reason
      has no end-to-end test" — F4's tests below drive webuidb.sync_once()
      for real.
  F4  webuidb MEDIUM ("the one durability alarm measures her IDLE TIME"):
      probe_snapshot compared the snapshot's mtime to wall-clock age, but
      webuidb stamps it with the LOCAL database's last-WRITE time, so a
      quiet night reads as an unbounded durability gap. Also pins the
      "three intervals" multiplier (H-5 N5) while touching this code.
  F5  hostile317-a F6 (LOW, documentation-only — see the lane report):
      pins that skipped_degenerate/too_short/no_boundary are INTENDED to
      degrade status (they are real memory loss, not faults, per
      tailhealth's own LOSSY_SKIP_OUTCOMES doctrine) so a future change to
      that doctrine is a deliberate decision, not an accident.

    python test_health_findings.py
"""

import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
_TMP_ROOT = tempfile.mkdtemp(prefix="health-findings-store-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_L1_CHUNK_SIZE"] = "40"
os.environ.pop("WEBUIDB_SYNC_ENABLED", None)

# Fixed volumes for webuidb, exactly like test_webuidb.py (its own file, not
# this lane's — duplicated here rather than imported, the same reason
# backup.py duplicates memory.py's storage-layout constants: importing a
# whole other test module for three path strings would drag its module-level
# side effects in with it).
_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="health-findings-local-"))
_SNAP_VOL = Path(tempfile.mkdtemp(prefix="health-findings-snap-"))
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL_VOL / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_VOL / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(Path(tempfile.mkdtemp(prefix="health-findings-quar-")))
os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "300"

# Fixed backup dir, separate from the compactor store above.
_BACKUP_DIR = Path(tempfile.mkdtemp(prefix="health-findings-backups-"))
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BACKUP_DIR)
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"

import memory  # noqa: E402

memory.ensure_storage_layout()

import retrieval  # noqa: E402

retrieval.conversation_doc_count = lambda conv_id: 0

import backup  # noqa: E402
import health  # noqa: E402
import summarizer  # noqa: E402
import tailhealth  # noqa: E402
import webuidb  # noqa: E402

# backup.py resolves STORAGE_ROOT/DATA_DIR/BACKUP_DIR at import time from
# env; COMPACTOR_STORAGE_ROOT above lines it up with memory.storage_root()
# automatically. BACKUP_DIR is reassigned directly (not via env — backup was
# already imported by the time this file could set COMPACTOR_BACKUP_DIR and
# have it read) to the fixed dir this file owns exclusively.
backup.BACKUP_DIR = _BACKUP_DIR

LOCAL_DB = webuidb.LOCAL_DB
SNAPSHOT_DB = webuidb.SNAPSHOT_DB

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


async def _vllm_ok(url):
    return {"ok": True, "latency_ms": 1.0, "models": ["m"], "error": None}


health.probe_vllm = _vllm_ok


def _reset_all():
    tailhealth._reset_for_tests()
    getattr(health, "_reset_hierarchy_progress_for_tests", lambda: None)()
    health._reset_process_started_at_for_tests()
    root = memory.storage_root()
    for sub in ("facts", "summaries"):
        d = root / sub
        if d.exists():
            for f in d.glob("*.json"):
                f.unlink()
    for p in (LOCAL_DB, SNAPSHOT_DB):
        for suffix in ("",) + webuidb.SIDECARS:
            f = p.with_name(p.name + suffix)
            if f.exists():
                f.unlink()
    for f in _BACKUP_DIR.glob("*"):
        if f.is_file():
            f.unlink()
    os.environ.pop("WEBUI_DB_LOCAL", None)
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("COMPACTOR_BACKUP_WEBUI_DB", None)


import asyncio  # noqa: E402


async def _full():
    return await health.gather_health_full("http://fake", 4096)


def full():
    return asyncio.run(_full())


def _has(r, needle):
    return any(needle in x for x in r["status_reasons"])


# ---------------------------------------------------------------------------
# F1 — hostile317-c F4: backups
# ---------------------------------------------------------------------------

def _make_real_backup():
    """One genuine backup.py archive, via the daemon's own create -> verify
    -> publish path (backup.run_once, not a hand-built dict). The compactor
    store is whatever COMPACTOR_STORAGE_ROOT already has (ensure_storage_layout
    made it a directory; empty is a valid, verifiable archive).

    No webui.db exists in this fixture, and since pass-3 F12 a cycle that
    cannot find one fails rather than publishing an archive without chat
    history. This test is about health reading a REAL archive, not about the
    database, so it uses that fix's explicit escape hatch for the duration of
    the one call."""
    with patch.dict(os.environ, {"COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB": "1"}):
        report = backup.run_once(_BACKUP_DIR)
    if not report.get("ok"):
        raise AssertionError(f"real backup.run_once() failed: {report}")
    return report


def test_f1_zero_backups_is_ok_during_the_grace_window():
    print("\n[F1] a freshly booted pod with zero archives is not degraded yet")
    _reset_all()
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
    }):
        health._reset_process_started_at_for_tests(time.time())  # "just booted"
        r = full()
    print(f"      backups={r['backups']} reasons={r['status_reasons']}")
    check(r["backups"].get("count") == 0, "F1 fixture: zero archives")
    check(r["status"] == "ok" and not _has(r, "zero archives"),
          "F1 GRACE: zero backups inside the first interval is not a reason")


def test_f1_zero_backups_after_the_grace_window_degrades():
    print("\n[F1] the same pod, past the grace window, still zero archives")
    # v3.1.9 (hostile pass 3, F7, part 1): the grace used to be a FULL
    # COMPACTOR_BACKUP_INTERVAL_HOURS (24h here). It is now min(interval, 2h)
    # — a cycle plus several retries, not a whole day of silence over the
    # worst durability state there is. The boundary below moves from
    # 23h/25h to just under/over 2h accordingly; see health._container_
    # started_at's docstring for why the CLOCK also changed (F7 part 2,
    # exercised separately in test_f7_grace_survives_a_compactor_only_respawn).
    _reset_all()
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
    }):
        health._reset_process_started_at_for_tests(time.time() - 25 * 3600)
        r = full()
    print(f"      reasons={r['status_reasons']}")
    check(r["status"] == "degraded" and _has(r, "zero archives"),
          "F1 FIRES: zero backups well past the grace degrades the pod")

    # BOUNDARY, both sides of the same clock: just under vs. just over the
    # new min(interval, 2h) = 7200s grace (interval is 24h here, so the cap
    # binds). Reproduces hostile317-c F4's own left side on purpose (zero
    # archives inside the grace is still "ok"); the right side is F7.
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
    }):
        health._reset_process_started_at_for_tests(time.time() - (7200 - 600))
        under = full()
    check(under["status"] == "ok",
          "F1/F7 BOUNDARY: ten minutes short of the 2h grace cap is still grace")

    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
    }):
        health._reset_process_started_at_for_tests(time.time() - (7200 + 600))
        over = full()
    check(over["status"] == "degraded" and _has(over, "zero archives"),
          "F7 BOUNDARY: ten minutes past the 2h grace cap degrades")


def test_f7_grace_uses_the_shorter_interval_when_configured_below_2h():
    print("\n[F7] a 1h backup interval caps the grace at the INTERVAL, not "
          "always at 2h — min(interval, 2h)")
    _reset_all()
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "1",
    }):
        health._reset_process_started_at_for_tests(time.time() - 3500)  # < 1h
        under = full()
        health._reset_process_started_at_for_tests(time.time() - 3700)  # > 1h
        over = full()
    check(under["status"] == "ok",
          "F7: under the 1h interval (which is below the 2h cap) is still grace")
    check(over["status"] == "degraded" and _has(over, "zero archives"),
          "F7: past the 1h interval degrades — the grace did not silently "
          "widen to 2h just because the cap exists")


def test_f7_grace_survives_a_compactor_only_respawn():
    print("\n[F7] part 2: a compactor-only respawn (this process's own start "
          "time jumping to 'now') must NOT reset the grace — the container's "
          "own clock, which does not restart with the compactor, is what "
          "the grace is measured from")
    _reset_all()
    try:
        # But THIS process (the compactor) just respawned a moment ago —
        # exactly what a crash loop or a burst of hot-patch restarts does.
        # Before this fix, _PROCESS_STARTED_AT WAS the only clock, so this
        # alone reset the grace to fresh every time, and a pod whose backups
        # had never succeeded could sit at "ok" forever through a crash loop.
        health._reset_process_started_at_for_tests(time.time())
        # The CONTAINER (backup daemon's actual environment) has been up for
        # 3 hours — well past the 2h grace — and never produced an archive.
        # Set AFTER _reset_process_started_at_for_tests, which moves BOTH
        # signals together — this independently overrides just the
        # container one, so the two clocks actually disagree, which is the
        # whole point of this test.
        health._set_container_started_at_for_tests(time.time() - 3 * 3600)
        with patch.dict(os.environ, {
            "COMPACTOR_BACKUP_ENABLED": "true",
            "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
        }):
            r = full()
        print(f"      reasons={r['status_reasons']}")
        check(r["status"] == "degraded" and _has(r, "zero archives"),
              "F7 FIXED: the container's real uptime (3h) still degrades the "
              "pod even though the compactor process itself just started — "
              "before this fix, a fresh _PROCESS_STARTED_AT alone made this 'ok'")
    finally:
        health._clear_container_started_at_override_for_tests()


def test_f7_unreadable_backup_dir_is_unobservable_not_zero():
    print("\n[F7] part 3: a backup dir this process cannot LIST reads as "
          "unobservable, not as zero archives")
    if os.name != "posix":
        print("  SKIPPED: chmod-based unreadable-directory simulation needs "
              "POSIX permission bits (Windows dev box). Linux container run "
              "exercises this for real.")
        return
    _reset_all()
    import tempfile
    locked_dir = Path(tempfile.mkdtemp(prefix="p3c-f7-locked-"))
    os.chmod(locked_dir, 0o000)
    try:
        # Root (and some container setups) ignore directory permission bits
        # entirely — confirm the lockout actually holds before trusting the
        # rest of this test, same doctrine the brief asks for everywhere
        # else (a mutation/permission change that did not apply has no
        # result).
        try:
            list(os.scandir(locked_dir))
            print("  SKIPPED: running with enough privilege that chmod 000 "
                  "did not actually block listing (root?) — cannot simulate "
                  "an unreadable directory here.")
            return
        except PermissionError:
            pass

        with patch.object(backup, "latest_backup_info",
                           lambda: {"count": 0, "latest": None,
                                    "latest_mtime": None, "dir": str(locked_dir),
                                    "error": None}), \
             patch.dict(os.environ, {
                 "COMPACTOR_BACKUP_ENABLED": "true",
                 "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
             }):
            health._reset_process_started_at_for_tests(time.time() - 25 * 3600)
            r = full()
        print(f"      reasons={r['status_reasons']}")
        check(r["status"] == "degraded" and _has(r, "backup status unobservable"),
              "F7 FIXED: an unreadable backup dir is 'unobservable', not "
              "silently treated as zero archives")
        check(not _has(r, "holds zero archives"),
              "F7: and specifically NOT the 'zero archives, nothing to "
              "restore' claim, which this process cannot actually verify")
    finally:
        os.chmod(locked_dir, 0o755)
        locked_dir.rmdir()


def test_f8_nonpositive_backup_interval_is_named_as_a_reason():
    print("\n[F8] COMPACTOR_BACKUP_INTERVAL_HOURS=0 (or negative, or non-"
          "finite) is named as its own reason, not silently clamped away")
    _reset_all()
    for bad in ("0", "-5", "nan", "inf"):
        with patch.dict(os.environ, {
            "COMPACTOR_BACKUP_ENABLED": "true",
            "COMPACTOR_BACKUP_INTERVAL_HOURS": bad,
        }):
            health._reset_process_started_at_for_tests(time.time())
            r = full()
        check(_has(r, "COMPACTOR_BACKUP_INTERVAL_HOURS"),
              f"F8: {bad!r} is named as a reason")
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
    }):
        health._reset_process_started_at_for_tests(time.time())
        control = full()
    check(not _has(control, "COMPACTOR_BACKUP_INTERVAL_HOURS"),
          "F8 CONTROL: a normal positive interval names no such reason")


def test_f1_backups_disabled_is_not_a_fault():
    print("\n[F1] COMPACTOR_BACKUP_ENABLED=false: zero archives is a choice")
    _reset_all()
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "false",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
    }):
        health._reset_process_started_at_for_tests(time.time() - 48 * 3600)
        r = full()
    check(r["status"] == "ok" and r["status_reasons"] == [],
          "F1 CONTROL: backups switched off on purpose is not a reason")


def test_f1_stale_latest_backup_degrades_and_fresh_does_not():
    print("\n[F1] the newest archive is older than 1.5x the backup interval")
    _reset_all()
    now = time.time()
    fake_archive = _BACKUP_DIR / "zions-backup-fake.tar.gz"
    fake_archive.write_bytes(b"x")
    old = now - int(2 * 3600)  # 2h old against a 1h interval, limit 1.5h
    os.utime(fake_archive, (old, old))
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "1",
    }):
        health._reset_process_started_at_for_tests(now - 48 * 3600)
        r = full()
    print(f"      backups={r['backups']} reasons={r['status_reasons']}")
    check(r["status"] == "degraded" and _has(r, "backup interval"),
          "F1 FIRES: a backup older than 1.5x the interval degrades the pod")

    fresh = now - int(0.2 * 3600)
    os.utime(fake_archive, (fresh, fresh))
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "1",
    }):
        health._reset_process_started_at_for_tests(now - 48 * 3600)
        r2 = full()
    check(r2["status"] == "ok",
          "F1 CONTROL: a backup inside 1.5x the interval is not a reason")

    # BOUNDARY, specifically between 1.0x and 1.5x: a single overrun cycle
    # (1.2x the interval) must not fire — this is the case the 1.5x
    # threshold exists to tolerate, per hostile317-c F4's own suggested
    # multiplier. A test that only checks 2x (fires) and 0.2x (quiet) cannot
    # tell a 1.5x threshold from a 1.0x one, since both agree on those two
    # points.
    at_1_2x = now - int(1.2 * 3600)
    os.utime(fake_archive, (at_1_2x, at_1_2x))
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "1",
    }):
        health._reset_process_started_at_for_tests(now - 48 * 3600)
        r3 = full()
    check(r3["status"] == "ok",
          "F1 BOUNDARY: 1.2x the interval is an overrun the 1.5x threshold "
          "must tolerate")


def test_f1_backup_probe_error_is_unobservable():
    print("\n[F1] backup.latest_backup_info() raising reaches status_reasons")
    _reset_all()

    def _raise(backup_dir=None):
        raise RuntimeError("backup dir wedged")

    with patch.object(backup, "latest_backup_info", _raise), \
         patch.dict(os.environ, {"COMPACTOR_BACKUP_ENABLED": "true"}):
        health._reset_process_started_at_for_tests(time.time() - 48 * 3600)
        r = full()
    print(f"      backups={r['backups']} reasons={r['status_reasons']}")
    check(r["status"] == "degraded" and _has(r, "backup status unobservable"),
          "F1 FIRES: an unreadable backups probe degrades the pod, not 'ok'")


def test_f1_real_archive_end_to_end_clears_the_reason():
    """MERGE-LEVEL-STYLE: built through backup.py's REAL create -> verify ->
    publish path (backup.run_once), per the brief's own instruction, not a
    hand-built {count, latest_mtime} dict."""
    print("\n[F1] a REAL backup.run_once() archive clears the reason")
    _reset_all()
    now = time.time()
    with patch.dict(os.environ, {
        "COMPACTOR_BACKUP_ENABLED": "true",
        "COMPACTOR_BACKUP_INTERVAL_HOURS": "24",
    }):
        health._reset_process_started_at_for_tests(now - 25 * 3600)
        before = full()
        check(_has(before, "zero archives"),
              "F1 fixture: zero archives still degrades before the real backup")
        report = _make_real_backup()
        print(f"      real backup.run_once() -> {report.get('archive')}")
        after = full()
    print(f"      backups={after['backups']}")
    check(after["backups"].get("count") == 1 and after["backups"].get("latest"),
          "F1: the real archive is counted by latest_backup_info()")
    check(after["status"] == "ok" and not _has(after, "zero archives")
          and not _has(after, "backup interval"),
          "F1 CONTROL: a freshly published real archive is not a reason")


# ---------------------------------------------------------------------------
# Sibling fixes (lane-health.md "Found, not fixed": the other two probe
# results whose error half was never read, alongside backups.error above)
# ---------------------------------------------------------------------------

def test_sibling_writes_unknown_is_unobservable():
    print("\n[sibling] degrade.write_state() raising reaches status_reasons")
    _reset_all()

    import degrade

    def _raise():
        raise RuntimeError("degrade wedged")

    with patch.object(degrade, "write_state", _raise):
        r = full()
    print(f"      memory_writes={r['memory_writes']} reasons={r['status_reasons']}")
    check(r["memory_writes"].get("new_memory_writes") == "unknown",
          "sibling fixture: degrade.write_state() raising reads as 'unknown'")
    check(r["status"] == "degraded" and _has(r, "new-memory write state unobservable"),
          "sibling FIXED: an unreadable write-pressure probe degrades the "
          "pod instead of reading exactly like 'writes are fine'")


def test_sibling_sqlite_journal_probe_crash_does_not_500_the_endpoint():
    print("\n[sibling] probe_sqlite_journal() itself raising a non-OSError")
    _reset_all()

    def _raise():
        raise RuntimeError("journal probe wedged")

    # Caught explicitly, not just called plainly: without the try/except
    # this is testing, the exception propagates all the way out of
    # gather_health_full (it runs inside asyncio.to_thread), which would
    # otherwise CRASH this whole test process instead of failing one named
    # check — a crash reads as "wrong reason" by this lane's own rules, so
    # the raise itself is turned into a clean, readable check result.
    with patch.object(health, "probe_sqlite_journal", _raise):
        try:
            r = full()
            raised = None
        except Exception as e:  # pragma: no cover - only under the mutant
            r = None
            raised = e
    check(raised is None,
          "sibling FIXED: gather_health_full() does not raise when "
          f"probe_sqlite_journal() itself raises (got {raised!r})")
    if raised is not None:
        return
    print(f"      sqlite_journal={r['checks']['sqlite_journal']} "
          f"reasons={r['status_reasons']}")
    check(r["checks"]["sqlite_journal"] is not None
          and r["checks"]["sqlite_journal"].get("error"),
          "sibling FIXED: a probe_sqlite_journal() crash is caught, not a "
          "500 from the whole endpoint")
    check(r["status"] == "degraded"
          and _has(r, "sqlite journal probe unobservable"),
          "sibling: and it still reaches status_reasons as unobservable")


# ---------------------------------------------------------------------------
# F2 — hostile2-backup A3-8: the journal probe watched the wrong file
# ---------------------------------------------------------------------------

_MAGIC = bytes.fromhex("d9d505f920a163d7")


def _write_hot_journal(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    journal = db_path.with_name(db_path.name + "-journal")
    journal.write_bytes(_MAGIC + bytes(20))


def _clear_journals():
    for db in (LOCAL_DB, SNAPSHOT_DB):
        j = db.with_name(db.name + "-journal")
        if j.exists():
            j.unlink()


def test_f2_hot_journal_on_the_live_local_db_is_caught_under_the_default_gate():
    print("\n[F2] WEBUI_DB_LOCAL=true (shipped default): a hot journal on "
          "WEBUI_LOCAL_DB, snapshot clean")
    _reset_all()
    _clear_journals()
    _write_hot_journal(LOCAL_DB)
    with patch.dict(os.environ, {"WEBUI_DB_LOCAL": "true"}):
        r = health.probe_sqlite_journal()
    print(f"      {r}")
    check(r.get("hot") is True and r.get("ok") is False,
          "F2 FIRES: a hot journal on the LIVE (local) db is caught")
    check((r.get("checked", {}).get("live", {}).get("path") or "").endswith(
        LOCAL_DB.name + "-journal"),
        "F2: the checked 'live' leg points at WEBUI_LOCAL_DB, not the snapshot")

    body = asyncio.run(health.gather_health_full("http://127.0.0.1:9", 4096))
    check(any("HOT SQLite rollback journal" in x for x in body["status_reasons"]),
          "F2 FIRES: and it reaches /health/full's status_reasons")
    _clear_journals()


def test_f2_control_same_file_checked_once_not_twice():
    print("\n[F2] WEBUI_DB_LOCAL=false: live == snapshot, checked once")
    _reset_all()
    _clear_journals()
    with patch.dict(os.environ, {
        "WEBUI_DB_LOCAL": "false",
        "COMPACTOR_BACKUP_WEBUI_DB": "",
        "DATABASE_URL": f"sqlite:///{SNAPSHOT_DB}",
    }):
        os.environ.pop("COMPACTOR_BACKUP_WEBUI_DB", None)
        r = health.probe_sqlite_journal()
    print(f"      checked keys={sorted(r.get('checked', {}))}")
    check(sorted(r.get("checked", {})) == ["live"],
          "F2 CONTROL: the same file under two names is checked exactly once")

    _write_hot_journal(SNAPSHOT_DB)
    with patch.dict(os.environ, {
        "WEBUI_DB_LOCAL": "false",
        "DATABASE_URL": f"sqlite:///{SNAPSHOT_DB}",
    }):
        os.environ.pop("COMPACTOR_BACKUP_WEBUI_DB", None)
        r2 = health.probe_sqlite_journal()
    check(r2.get("hot") is True,
          "F2: with WEBUI_DB_LOCAL=false the snapshot IS the live db, and a "
          "hot journal there is still caught")
    _clear_journals()


def test_f2_snapshot_also_checked_when_it_differs_and_local_is_clean():
    print("\n[F2] a hot journal on the SNAPSHOT (webuidb-sync's own writer), "
          "live clean")
    _reset_all()
    _clear_journals()
    _write_hot_journal(SNAPSHOT_DB)
    with patch.dict(os.environ, {"WEBUI_DB_LOCAL": "true"}):
        r = health.probe_sqlite_journal()
    print(f"      {r}")
    check(r.get("hot") is True,
          "F2: a hot journal on the snapshot side is caught even though the "
          "live database is clean")
    _clear_journals()


def test_f2_unresolvable_live_path_is_unobservable_not_clean():
    print("\n[F2] backup.live_webui_db() raising")
    _reset_all()
    _clear_journals()

    def _raise():
        raise RuntimeError("gate unreadable")

    with patch.object(backup, "live_webui_db", _raise):
        r = health.probe_sqlite_journal()
        print(f"      {r}")
        check(r.get("ok") is None and r.get("hot") is None and "gate unreadable" in str(r.get("error")),
              "F2: an unresolvable live path reports unobservable, not clean")
        body = asyncio.run(health.gather_health_full("http://127.0.0.1:9", 4096))
    check(any("sqlite journal probe unobservable" in x for x in body["status_reasons"]),
          "F2: and it reaches status_reasons as unobservable (sibling of H-4)")


# ---------------------------------------------------------------------------
# F3a/F3b — worst_lag recency + _lag_limit ranging
# ---------------------------------------------------------------------------

def _save_state_with_lag(cid, seen, done):
    st = summarizer.load_state(cid)
    st["turns_seen"] = seen
    st["last_summarized_turn"] = done
    summarizer.save_state(cid, st)


def test_f3a_an_ancient_conversation_does_not_mask_a_live_one():
    print("\n[F3a] a 400-day-old junk namespace vs. a live lagging conversation")
    _reset_all()
    _save_state_with_lag("ancient-junk", seen=9000, done=0)  # huge, frozen lag
    old = time.time() - 400 * 86400
    os.utime(summarizer.summary_path("ancient-junk"), (old, old))

    _save_state_with_lag("live-conv", seen=100, done=40)  # lag 60 > limit 40
    # state file mtime is "now" (just saved) — genuinely recent.

    with patch.dict(os.environ, {"COMPACTOR_L1_CHUNK_SIZE": "20"}):
        saved_chunk = summarizer.L1_CHUNK_SIZE
        summarizer.L1_CHUNK_SIZE = 20
        try:
            r = full()
        finally:
            summarizer.L1_CHUNK_SIZE = saved_chunk
    print(f"      stats.hierarchy_lag={r['stats']['hierarchy_lag']} "
          f"conv={r['stats']['hierarchy_lag_conv']}")
    print(f"      stats.hierarchy_lag_recent={r['stats']['hierarchy_lag_recent']} "
          f"conv={r['stats']['hierarchy_lag_recent_conv']}")
    print(f"      reasons={r['status_reasons']}")
    check(r["stats"]["hierarchy_lag"] == 9000
          and r["stats"]["hierarchy_lag_conv"] == "ancient-junk",
          "F3a: the raw unfiltered max is still reported in stats (diagnostic)")
    check(r["stats"]["hierarchy_lag_recent_conv"] == "live-conv",
          "F3a: the recency-filtered pair names the LIVE conversation")
    check(_has(r, "conv=live-conv") and not _has(r, "conv=ancient-junk"),
          "F3a FIXED: the status reason names the live conversation, not "
          "the ancient one that would otherwise have masked it")


def test_f3a_control_a_lagging_conversation_still_fires_when_alone():
    print("\n[F3a] CONTROL: no ancient junk present, a real lag still fires")
    _reset_all()
    _save_state_with_lag("only-conv", seen=100, done=40)
    summarizer.L1_CHUNK_SIZE, saved = 20, summarizer.L1_CHUNK_SIZE
    try:
        r = full()
    finally:
        summarizer.L1_CHUNK_SIZE = saved
    check(r["status"] == "degraded" and _has(r, "conv=only-conv"),
          "F3a CONTROL: the recency filter does not silently swallow a real, "
          "lone lag")


def test_f3b_lag_limit_is_ranged_against_a_bad_chunk_size():
    print("\n[F3b] COMPACTOR_L1_CHUNK_SIZE=0: the limit must not become 0")
    _reset_all()
    _save_state_with_lag("one-turn-drift", seen=1, done=0)  # lag 1
    saved = summarizer.L1_CHUNK_SIZE
    summarizer.L1_CHUNK_SIZE = 0
    try:
        r = full()
    finally:
        summarizer.L1_CHUNK_SIZE = saved
    print(f"      reasons={r['status_reasons']}")
    check(r["status"] == "ok",
          "F3b FIXED: COMPACTOR_L1_CHUNK_SIZE=0 no longer degrades on lag 1")


def test_f3b_lag_limit_negative_with_zero_conversations():
    print("\n[F3b] COMPACTOR_L1_CHUNK_SIZE=-5, zero conversations")
    _reset_all()
    saved = summarizer.L1_CHUNK_SIZE
    summarizer.L1_CHUNK_SIZE = -5
    try:
        r = full()
    finally:
        summarizer.L1_CHUNK_SIZE = saved
    print(f"      reasons={r['status_reasons']}")
    check(r["status"] == "ok" and not _has(r, "conv=None"),
          "F3b FIXED: a negative chunk size no longer degrades an empty "
          "store and prints conv=None")


# ---------------------------------------------------------------------------
# F4 — the durability alarm measured idle time, not publish lag
# built end-to-end through webuidb.sync_once(), the REAL writer (also closes
# the OPEN_ISSUES2 "no end-to-end test" MEDIUM for the stale-snapshot reason)
# ---------------------------------------------------------------------------

def _make_local_db(chats=1):
    LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(LOCAL_DB))
    con.execute("create table chat (id text, body text)")
    con.executemany("insert into chat values (?, ?)",
                     [(str(i), "hi") for i in range(chats)])
    con.commit()
    con.close()


def _snap_env(interval="300"):
    return {
        "WEBUIDB_SYNC_ENABLED": "true",
        "WEBUI_SNAPSHOT_DB": str(SNAPSHOT_DB),
        "WEBUI_LOCAL_DB": str(LOCAL_DB),
        "WEBUI_DB_SYNC_INTERVAL_S": interval,
    }


def test_f4_a_quiet_night_after_a_real_publish_is_not_stale():
    print("\n[F4] a REAL webuidb.sync_once() publish, then an 8h quiet night")
    _reset_all()
    _make_local_db(chats=1)
    eight_h_ago = time.time() - 8 * 3600
    os.utime(LOCAL_DB, (eight_h_ago, eight_h_ago))
    result = webuidb.sync_once()
    print(f"      sync_once() -> {result}")
    check(result.get("synced") is True, "F4 fixture: the real publish succeeded")

    with patch.dict(os.environ, _snap_env("300")):
        r = full()
    snap = r["checks"]["snapshot"] or {}
    print(f"      snapshot={snap}")
    check(snap.get("age_s", 0) > 25000,
          "F4 fixture: the OLD signal (age_s) is ~8h, exactly what used to "
          "cry wolf")
    check(snap.get("local_lag_s") is not None and abs(snap["local_lag_s"]) <= 2,
          "F4 FIXED: local_lag_s is ~0 — the durable copy is caught up with "
          "the (quiet) live database")
    check(snap.get("stale") is False and r["status"] == "ok",
          "F4 FIXED: a healthy quiet night no longer reads as a durability "
          "gap")


def test_f4_unpublished_activity_after_a_quiet_baseline_still_fires():
    print("\n[F4] REAL activity after a REAL publish, with no second sync")
    _reset_all()
    _make_local_db(chats=1)
    eight_h_ago = time.time() - 8 * 3600
    os.utime(LOCAL_DB, (eight_h_ago, eight_h_ago))
    result = webuidb.sync_once()
    check(result.get("synced") is True, "F4 fixture: baseline publish succeeded")

    # She chats now. A real write, so LOCAL_DB's own mtime moves to "now" —
    # and the daemon does NOT run again (simulating F4's own named failure
    # shape: sync_loop stalled).
    con = sqlite3.connect(str(LOCAL_DB))
    con.execute("insert into chat values ('new', 'hello')")
    con.commit()
    con.close()

    with patch.dict(os.environ, _snap_env("300")):
        r = full()
    snap = r["checks"]["snapshot"] or {}
    print(f"      snapshot={snap} reasons={r['status_reasons']}")
    check(snap.get("local_lag_s", 0) > 900,
          "F4 CONTROL: real unpublished activity produces a real, growing lag")
    check(snap.get("stale") is True and r["status"] == "degraded"
          and _has(r, "behind the live database"),
          "F4 CONTROL: and THAT is reported as a durability gap — the fix "
          "did not just silence the alarm, it retargeted it")


def test_f4_falls_back_safely_when_local_db_is_unreadable():
    print("\n[F4] WEBUI_LOCAL_DB missing: falls back to the old age check")
    _reset_all()
    SNAPSHOT_DB.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DB.write_bytes(b"x")
    old = time.time() - 4 * 300
    os.utime(SNAPSHOT_DB, (old, old))
    missing_local = str(_LOCAL_VOL / "does-not-exist" / "webui.db")
    with patch.dict(os.environ, {**_snap_env("300"), "WEBUI_LOCAL_DB": missing_local}):
        r = full()
    snap = r["checks"]["snapshot"] or {}
    print(f"      snapshot={snap}")
    check(snap.get("local_lag_s") is None,
          "F4 FALLBACK: local_lag_s is None when the local db cannot be read")
    check(snap.get("stale") is True,
          "F4 FALLBACK: staleness reverts to the old age-based comparison, "
          "not silently 'ok'")


def test_f4_three_intervals_boundary_is_pinned():
    print("\n[F4] H-5 N5: the '3x interval' multiplier, both sides")
    _reset_all()
    _make_local_db(chats=1)
    baseline = time.time() - 3 * 300  # exactly 3 intervals ago
    os.utime(LOCAL_DB, (baseline, baseline))
    result = webuidb.sync_once()
    check(result.get("synced") is True, "F4 boundary fixture: publish ok")
    # Local advances past the publish by exactly 3*interval and by one
    # second more than 3*interval, with no second publish.
    at_limit = baseline + 3 * 300
    os.utime(LOCAL_DB, (at_limit, at_limit))
    with patch.dict(os.environ, _snap_env("300")):
        at = full()
    over = at_limit + 1
    os.utime(LOCAL_DB, (over, over))
    with patch.dict(os.environ, _snap_env("300")):
        over_r = full()
    at_snap = at["checks"]["snapshot"] or {}
    over_snap = over_r["checks"]["snapshot"] or {}
    print(f"      at 3x={at_snap.get('local_lag_s')} stale={at_snap.get('stale')}; "
          f"over 3x={over_snap.get('local_lag_s')} stale={over_snap.get('stale')}")
    check(at_snap.get("stale") is False,
          "F4 BOUNDARY: exactly 3x the interval of lag is not yet stale")
    check(over_snap.get("stale") is True,
          "F4 BOUNDARY: one second past 3x the interval is stale")


# ---------------------------------------------------------------------------
# F5 — hostile317-a F6: decision pinned, not changed (see the lane report)
# ---------------------------------------------------------------------------

def test_f5_quality_gate_skips_are_intended_to_degrade_status():
    print("\n[F5] skipped_degenerate/too_short/no_boundary DO degrade status "
          "(decision: keep as-is — see the lane report)")
    _reset_all()
    tailhealth.note(tailhealth.SKIPPED_DEGENERATE, raw_chars=40, kept_chars=0)
    r = full()
    print(f"      reasons={r['status_reasons']}")
    check(r["status"] == "degraded" and _has(r, "memory tail skipping"),
          "F5: a quality-gate skip (skipped_degenerate) degrades status, by "
          "design — LOSSY_SKIP_OUTCOMES treats it as real memory loss, the "
          "300s window self-clears, and HTTP status stays 200 so nothing "
          "restarts. A precise fault-only split needs tailhealth.py, which "
          "is outside this lane's file list (see 'Found, not fixed').")


def test_g3_restore_marker_reason_fires_and_clears():
    """p4-b G3: the third of the three places this finding asks the
    in-flight/stale restore marker be checked — /health/full. Writes a
    marker with the SAME function backup.py::restore_backup uses
    (webuidb.write_restore_marker), never a hand-built file, per the
    brief's "real writer" doctrine."""
    print("\n[G3] an in-flight/stale restore marker reaches status_reasons, and clearing it clears the reason")
    _reset_all()
    r0 = full()
    check(not _has(r0, "restore marker"), "G3 fixture: no marker, no reason, before this test writes one")

    stamp = "20260913-999999-000"
    webuidb.write_restore_marker(stamp, {
        "archive": "zions-backup-20260913-000000.tar.gz",
        "target_db": str(LOCAL_DB), "staged_db_tmp": None,
        "sroot": None, "staged_store_incoming": None,
        "quarantine_dir": str(webuidb.QUARANTINE),
    })
    try:
        r = full()
        print(f"      reasons={r['status_reasons']}")
        check(r["status"] == "degraded" and _has(r, "restore marker"),
              "G3 FIRES: a present marker degrades the pod and names itself in status_reasons")
        check(_has(r, "restore-") and _has(r, ".inprogress"),
              "and the reason names the actual marker FILE, not just that one exists")
    finally:
        webuidb.remove_restore_marker(stamp)

    r_after = full()
    check(not _has(r_after, "restore marker"),
          "G3 CONTROL: removing the marker (the documented recovery) clears the reason on the next poll")


def test_p9_reuse_decline_signal_reaches_health_full():
    """P9-1/P9-2 (hostile pass #9): checks.reuse. Before this, a reuse
    decline (the stored hierarchy not fitting the stand-in's budget) had
    NO signal anywhere but an INFO log line inside compact_if_needed —
    exactly how the feature shipped silently off (P9-1: green health,
    CHANGELOG claiming it worked). `health._reuse_state()` reads
    `main.reuse_decline_state()` via sys.modules, the same call-time
    pattern `_tokenizer_state()` already uses for `checks.tokenizer` (see
    that function's docstring for why a module-scope `import main` here
    would be circular) — this test imports `main` itself (this file does
    not, at module scope, unlike test_reuse_fit.py) so the "available"
    path is actually exercised, not just the "main is not loaded" one.

    P10-3 (hostile pass #10): the recorder API this test drives changed
    shape (`_record_reuse_decline(ceiling, others)` -> the more general
    `_record_reuse_outcome(reason, ceiling, others)`), and the fields this
    test pins grew three new counters — see `test_p10_3_*` below for the
    wiring THAT split is actually for (an exception is no longer
    indistinguishable from a success). This test stays about the plumbing:
    a budget decline's numbers reach `/health/full` unchanged.
    """
    print("\n[P9-1/P9-2] checks.reuse reads main.reuse_decline_state()")
    import main  # local: this module does not import main at module scope

    # Numbers only, and cheap: call the real recorder functions directly
    # rather than driving a whole compact_if_needed request (that path is
    # test_reuse_fit.py's [10]/[11]/[12] sections' job — this test is
    # about the health WIRING, not the reuse arithmetic).
    before = health._reuse_state()
    check(before.get("available") is True,
          f"main is loaded in this process, so checks.reuse must read it "
          f"(got {before})")
    attempted_before = before["attempted"]
    declined_before = before["declined_budget"]

    main._record_reuse_attempt()
    main._record_reuse_outcome("budget", 9345, 12706)
    after = health._reuse_state()
    check(after["attempted"] == attempted_before + 1,
          "*** attempted increments")
    check(after["declined_budget"] == declined_before + 1,
          "*** declined_budget increments")
    check(after["declined_recently"] is True,
          "*** a decline just now reads as recent")
    check(after["last_reason"] == "budget",
          "*** last_reason names the budget decline specifically")
    check(
        after["last_attempt_age_s"] is not None and after["last_attempt_age_s"] < 5,
        f"*** last_attempt_age_s is a real, small number right after the "
        f"attempt just recorded (got {after['last_attempt_age_s']})",
    )
    check(after["last_declined_ceiling"] == 9345 and after["last_declined_others"] == 12706,
          "*** the two numbers that explain the decline are carried through, "
          "unchanged — no conversation text, no conv_id, anywhere in this "
          "payload")

    r = full()
    check("reuse" in r["checks"], "*** gather_health_full's checks dict carries 'reuse'")
    check(r["checks"].get("reuse", {}).get("declined_budget") == after["declined_budget"],
          "and it is the SAME live state _reuse_state() reads directly, not "
          "a stale or re-derived copy")
    check(r["status"] != "down",
          "visibility-only: a reuse decline does not itself take the pod down")


def _p10_3_history(n_exchanges: int, words: int = 200) -> list:
    """Plain old turns, no system prefix — same shape as test_reuse_fit.py's
    `history()` minus its leading system message (not needed here; this
    fixture never goes through `_enforce_hard_budget`, only
    `compact_if_needed`), duplicated rather than imported per this file's
    own convention (see the module docstring's F2 note on backup.py)."""
    out = []
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"q{i} " + ("word " * words)})
        out.append({"role": "assistant", "content": f"a{i} " + ("word " * words)})
    return out


def test_p10_3_reuse_error_state_is_not_a_silent_success():
    """P10-3 (hostile pass #10): the ONE case P9's own adversarial test
    could not tell apart from a healthy reuse — an exception raised
    partway through the reuse attempt (the `except Exception` fallback in
    `compact_if_needed` exists precisely so an optimisation can never fail
    a whole request, but the OLD accounting incremented `attempted` at a
    call site downstream of where the crash happens and never recorded a
    matching decline, so `checks.reuse` read `attempted+1, declined_budget
    +0` — indistinguishable from a real success). Drives a REAL
    `compact_if_needed` call (not the recorder functions directly, unlike
    the test above — this is the "assertion that cannot be produced by a
    crash" LOOPS5_BRIEF asks for) against a real, reuse-eligible hierarchy,
    with `summarizer.format_summary_block` made to raise partway through.
    """
    print("\n[P10-3] a crash inside the reuse attempt reads as 'error', "
          "never as a success")
    import asyncio as _asyncio
    import main  # local, same reason as the test above

    # The reuse attempt crashes and falls back to summarizing everything
    # from scratch (compact_if_needed's own except-clause guarantee) —
    # which would otherwise mean a REAL vLLM call from this offline suite.
    # Stubbed the same way test_reuse_fit.py's `_spy_summarize` is: this
    # test is about the reuse ACCOUNTING, not summarize()'s own behavior.
    _orig_summarize = main.summarize

    async def _stub_summarize(client, to_summarize):
        return "STUBBED-SUMMARY", []

    main.summarize = _stub_summarize

    conv_id = "health-p10-3-reuse-error"
    older = _p10_3_history(30, words=200)
    st = summarizer.load_state(conv_id)
    st["l1"] = [{
        "tier": "l1", "text": "L1scene " + ("filler " * 400),
        "first_turn": 1, "last_turn": len(older),
    }]
    st["last_summarized_turn"] = len(older)
    summarizer._record_chunk_fps(st, 1, len(older), older)
    summarizer.save_state(conv_id, st)

    recent = [
        {"role": "user", "content": "prev-u " + "u" * 200},
        {"role": "assistant", "content": "prev-a " + "a" * 200},
        {"role": "user", "content": "newest " + "n" * 200},
    ]
    msgs = older + recent
    check(main.count_tokens(msgs) > main.TARGET_TOKENS,
          "fixture: over TARGET, so compact_if_needed actually attempts reuse")

    before = health._reuse_state()

    _orig_format = summarizer.format_summary_block

    def _raise_partway(*a, **k):
        raise RuntimeError("P10-3 test: simulated failure inside the reuse attempt")

    summarizer.format_summary_block = _raise_partway
    try:
        stored_out: list = []
        out = _asyncio.run(main.compact_if_needed(
            list(msgs), conv_id, stored_turns_out=stored_out,
        ))
    finally:
        summarizer.format_summary_block = _orig_format
        main.summarize = _orig_summarize

    check(stored_out == [] or stored_out == [0],
          f"the crashed attempt substituted nothing (stored_turns_out="
          f"{stored_out}) — compact_if_needed's own contract for 'nothing "
          f"compacted or an exception': see that function's docstring")
    check(isinstance(out, list) and len(out) > 0,
          "*** the request itself still succeeds — an optimisation crashing "
          "must never fail the whole reply (the guarantee this except "
          "clause exists for)")

    after = health._reuse_state()
    check(after["attempted"] == before["attempted"] + 1,
          "*** attempted increments once for this request")
    check(after["errored"] == before["errored"] + 1,
          f"*** errored increments (got {after['errored']} vs "
          f"{before['errored']}) — the crash is COUNTED, not silent")
    check(after["succeeded"] == before["succeeded"],
          "*** succeeded does NOT increment — this is the exact "
          "distinction P9's own test could not make (attempted+1 with "
          "declined_budget unchanged used to read as a healthy reuse)")
    check(after["declined_budget"] == before["declined_budget"],
          "*** declined_budget (a BUDGET decline specifically) also does "
          "not increment — this was never a budget decision, it never got "
          "that far")
    check(after["last_reason"] == "error",
          f"*** last_reason is 'error', not silently absent or 'success' "
          f"(got {after['last_reason']!r})")

    r = full()
    check(r["checks"].get("reuse", {}).get("errored") == after["errored"],
          "and /health/full carries the SAME errored count, not a stale "
          "or re-derived copy")
    check(r["checks"].get("reuse", {}).get("last_reason") == "error",
          "and /health/full's last_reason also reads 'error'")
    check(r["status"] != "down",
          "visibility-only: a crashed reuse attempt does not itself take "
          "the pod down — the request still succeeded")


TESTS = [
    test_f1_zero_backups_is_ok_during_the_grace_window,
    test_f1_zero_backups_after_the_grace_window_degrades,
    test_f7_grace_uses_the_shorter_interval_when_configured_below_2h,
    test_f7_grace_survives_a_compactor_only_respawn,
    test_f7_unreadable_backup_dir_is_unobservable_not_zero,
    test_f8_nonpositive_backup_interval_is_named_as_a_reason,
    test_f1_backups_disabled_is_not_a_fault,
    test_f1_stale_latest_backup_degrades_and_fresh_does_not,
    test_f1_backup_probe_error_is_unobservable,
    test_f1_real_archive_end_to_end_clears_the_reason,
    test_sibling_writes_unknown_is_unobservable,
    test_sibling_sqlite_journal_probe_crash_does_not_500_the_endpoint,
    test_f2_hot_journal_on_the_live_local_db_is_caught_under_the_default_gate,
    test_f2_control_same_file_checked_once_not_twice,
    test_f2_snapshot_also_checked_when_it_differs_and_local_is_clean,
    test_f2_unresolvable_live_path_is_unobservable_not_clean,
    test_f3a_an_ancient_conversation_does_not_mask_a_live_one,
    test_f3a_control_a_lagging_conversation_still_fires_when_alone,
    test_f3b_lag_limit_is_ranged_against_a_bad_chunk_size,
    test_f3b_lag_limit_negative_with_zero_conversations,
    test_f4_a_quiet_night_after_a_real_publish_is_not_stale,
    test_f4_unpublished_activity_after_a_quiet_baseline_still_fires,
    test_f4_falls_back_safely_when_local_db_is_unreadable,
    test_f4_three_intervals_boundary_is_pinned,
    test_f5_quality_gate_skips_are_intended_to_degrade_status,
    test_g3_restore_marker_reason_fires_and_clears,
    test_p9_reuse_decline_signal_reaches_health_full,
    test_p10_3_reuse_error_state_is_not_a_silent_success,
]


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for t in TESTS:
        if only and only not in t.__name__:
            continue
        try:
            t()
        except Exception as e:  # a crash is reported, never scored as a pass
            import traceback
            traceback.print_exc()
            FAILED.append(f"{t.__name__} CRASHED: {type(e).__name__}: {e}")
            print(f"FAIL {t.__name__} CRASHED: {type(e).__name__}: {e}")
    print("")
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("All health-findings checks passed.")
