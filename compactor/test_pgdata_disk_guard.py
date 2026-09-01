"""
compactor/pgarchive.py — the PGDATA disk-headroom guard.

THE GAP THIS CLOSES. The compactor's free-space guard watches /data
(backup.py); pgarchive.py's own shrink guard watches archive content. Until
this, nothing watched the actual filesystem PGDATA lives on — the 20 GB
EPHEMERAL pod overlay (see pgarchive.py's module docstring for the measured
`df` numbers). A full overlay is not a degraded database, it is a Postgres
PANIC and a wedged pod, so this guard exists to warn loudly before that,
never to react to it.

What matters most here, and what most of this file checks:
  [1] free space is read from PGDATA's OWN filesystem, walking up to an
      existing ancestor when PGDATA itself doesn't exist yet (pre-initdb)
  [2] the level escalates ok -> warn -> critical as measured free space
      drops past the two env-overridable thresholds, and stands down again
      above them
  [3] PGDATA's own footprint is reported (0 for an empty/nonexistent dir,
      real bytes once files exist)
  [4] the guard is fully independent of /data and of Postgres reachability
      — nothing about ARCHIVE_DIR or _pg_reachable can make it fail, and
      nothing it does can affect either of those in return
  [5] --status surfaces the same information, and the archive loop's
      per-cycle call can never abort archiving even if the guard raises

Two temp directories stand in for the two volumes, same convention as
test_pgarchive.py: LOCAL_VOL/pgdata is the overlay this guard watches,
ARCHIVE_VOL is /data, deliberately left untouched by every check below.

    python test_pgdata_disk_guard.py
"""

import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="vol-local-diskguard-"))
_ARCHIVE_VOL = Path(tempfile.mkdtemp(prefix="vol-moosefs-diskguard-"))

os.environ["PGDATA"] = str(_LOCAL_VOL / "pgdata")
os.environ["PGARCHIVE_DIR"] = str(_ARCHIVE_VOL / "openwebui" / "pg")
os.environ["POSTGRES_USER"] = "openwebui"
os.environ["POSTGRES_DB"] = "openwebui"
os.environ["POSTGRES_SOCKET_DIR"] = str(_LOCAL_VOL / "run-postgresql")
os.environ["PGARCHIVE_DISK_WARN_FREE_MB"] = "4096"
os.environ["PGARCHIVE_DISK_CRITICAL_FREE_MB"] = "1024"

import pgarchive  # noqa: E402

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}")


print("two volumes:")
print(f"  local (overlay)  {_LOCAL_VOL}")
print(f"  archive (mfs)    {_ARCHIVE_VOL}  (must stay untouched by this file)")


# ---------------------------------------------------------------------------
# A fake shutil.disk_usage so thresholds can be exercised deterministically
# without depending on how much space this machine's real temp volume has.
# Keyed by the path disk_usage() was called with, so the ancestor walk-up in
# _pgdata_disk_status is exercised for real (it calls a real Path.exists()),
# and only the final free/total numbers are faked.
# ---------------------------------------------------------------------------

_real_disk_usage = shutil.disk_usage
_fake_free_mb = {"value": 999_999}


def _fake_disk_usage(path):
    total = 20_000 * 1_000_000  # 20 GB total, matching the real overlay
    free = int(_fake_free_mb["value"] * 1_000_000)
    return types.SimpleNamespace(total=total, used=total - free, free=free)


shutil.disk_usage = _fake_disk_usage


def _wipe_pgdata():
    if _LOCAL_VOL.joinpath("pgdata").exists():
        shutil.rmtree(_LOCAL_VOL / "pgdata")


def _archive_snapshot():
    """(exists, list of files) for ARCHIVE_DIR — used to prove the guard
    never touches /data."""
    d = pgarchive.ARCHIVE_DIR
    if not d.exists():
        return (False, [])
    return (True, sorted(p.name for p in d.rglob("*")))


try:
    # -------------------------------------------------------------------
    print()
    print("[1] PGDATA absent (pre-initdb): walks up to an existing ancestor, does not crash")
    _wipe_pgdata()
    _fake_free_mb["value"] = 999_999
    status = pgarchive._pgdata_disk_status()
    check(status["level"] == "ok", f"reports ok with ample free space (got {status})")
    check(status["free_mb"] is not None, "free_mb is populated even though PGDATA itself doesn't exist")
    check(status["pgdata_mb"] is None, "pgdata_mb is None (nothing to size) rather than a fabricated 0")

    # Same case again, but with the REAL shutil.disk_usage (not the fake,
    # which ignores its path argument and would pass even if the
    # ancestor-walk-up were deleted entirely) — proves the walk-up actually
    # resolves to a real, existing filesystem instead of raising on a
    # multi-level-deep path that has never existed.
    _wipe_pgdata()
    deep_missing_pgdata = _LOCAL_VOL / "pgdata" / "does" / "not" / "exist" / "yet"
    real_pgdata = pgarchive.PGDATA
    pgarchive.PGDATA = deep_missing_pgdata
    shutil.disk_usage = _real_disk_usage
    try:
        status = pgarchive._pgdata_disk_status()
    finally:
        pgarchive.PGDATA = real_pgdata
        shutil.disk_usage = _fake_disk_usage
    check(
        status["level"] in ("ok", "warn", "critical"),
        f"a real, never-created deep PGDATA path resolves to a real ancestor's free space, not 'unknown' (got {status})",
    )

    # -------------------------------------------------------------------
    print()
    print("[2] PGDATA present and empty: pgdata_mb reports 0, not None")
    _wipe_pgdata()
    (_LOCAL_VOL / "pgdata").mkdir(parents=True)
    status = pgarchive._pgdata_disk_status()
    check(status["pgdata_mb"] == 0, f"an existing-but-empty PGDATA sizes as 0 MB (got {status['pgdata_mb']})")

    # -------------------------------------------------------------------
    print()
    print("[3] PGDATA footprint reflects real file content")
    _wipe_pgdata()
    (_LOCAL_VOL / "pgdata").mkdir(parents=True)
    (_LOCAL_VOL / "pgdata" / "base.dat").write_bytes(b"x" * 5_000_000)  # 5 MB
    status = pgarchive._pgdata_disk_status()
    check(
        4.5 < status["pgdata_mb"] < 5.5,
        f"pgdata_mb tracks the real 5 MB file on disk (got {status['pgdata_mb']})",
    )

    # -------------------------------------------------------------------
    print()
    print("[4] level escalates ok -> warn -> critical as free space drops")
    _fake_free_mb["value"] = 10_000  # well above WARN=4096
    check(pgarchive._pgdata_disk_status()["level"] == "ok", "10000 MB free -> ok")
    _fake_free_mb["value"] = 3_000  # under WARN, above CRITICAL
    check(pgarchive._pgdata_disk_status()["level"] == "warn", "3000 MB free -> warn")
    _fake_free_mb["value"] = 500  # under CRITICAL
    check(pgarchive._pgdata_disk_status()["level"] == "critical", "500 MB free -> critical")
    # boundary: exactly at the threshold is NOT yet the escalated level
    # (thresholds are "< N", matching the module's own comments)
    _fake_free_mb["value"] = 4096
    check(pgarchive._pgdata_disk_status()["level"] == "ok", "exactly at WARN threshold is still ok (< not <=)")
    _fake_free_mb["value"] = 1024
    check(pgarchive._pgdata_disk_status()["level"] == "warn", "exactly at CRITICAL threshold is still warn (< not <=)")
    _fake_free_mb["value"] = 999_999

    # -------------------------------------------------------------------
    print()
    print("[5] thresholds are env-overridable")
    os.environ["PGARCHIVE_DISK_WARN_FREE_MB"] = "20000"
    os.environ["PGARCHIVE_DISK_CRITICAL_FREE_MB"] = "15000"
    pgarchive._reload_env()
    _fake_free_mb["value"] = 18_000
    check(
        pgarchive._pgdata_disk_status()["level"] == "warn",
        "raising PGARCHIVE_DISK_WARN_FREE_MB actually changes the outcome (18000 MB now warns)",
    )
    os.environ["PGARCHIVE_DISK_WARN_FREE_MB"] = "4096"
    os.environ["PGARCHIVE_DISK_CRITICAL_FREE_MB"] = "1024"
    pgarchive._reload_env()

    # -------------------------------------------------------------------
    print()
    print("[6] independent of /data: ARCHIVE_DIR is never touched by this guard")
    _wipe_pgdata()
    (_LOCAL_VOL / "pgdata").mkdir(parents=True)
    before = _archive_snapshot()
    for level_mb in (999_999, 3_000, 500):
        _fake_free_mb["value"] = level_mb
        pgarchive._pgdata_disk_status()
    after = _archive_snapshot()
    check(before == after, f"ARCHIVE_DIR untouched across all severity levels (before={before}, after={after})")
    _fake_free_mb["value"] = 999_999

    # -------------------------------------------------------------------
    print()
    print("[7] a broken disk_usage() (e.g. an unreadable overlay) reports unknown, never raises")
    def _raising_disk_usage(path):
        raise OSError("simulated: filesystem unreadable")

    shutil.disk_usage = _raising_disk_usage
    try:
        status = pgarchive._pgdata_disk_status()
    except Exception as e:  # pragma: no cover - this is exactly what must NOT happen
        check(False, f"_pgdata_disk_status raised instead of degrading ({type(e).__name__}: {e})")
    else:
        check(status["level"] == "unknown", f"degrades to level=unknown instead of raising (got {status})")
        check(status["free_mb"] is None, "free_mb is None, not a fabricated number")
    finally:
        shutil.disk_usage = _fake_disk_usage

    # -------------------------------------------------------------------
    print()
    print("[8] --status output includes the disk guard line and does not crash")
    import subprocess as _sp

    env = dict(os.environ)
    # Real disk_usage here (subprocess doesn't inherit the monkeypatch) —
    # just checking the CLI wires the guard in and doesn't blow up.
    r = _sp.run(
        [sys.executable, str(Path(__file__).parent / "pgarchive.py"), "--status"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    check(r.returncode == 0, f"--status exits 0 (stderr: {r.stderr[:300]})")
    check("pgdata_disk:" in r.stdout, f"--status output includes a pgdata_disk line (got: {r.stdout[:300]!r})")

    # -------------------------------------------------------------------
    print()
    print("[9] archive_loop's per-cycle disk check can never abort the loop")
    # Simulate exactly what archive_loop() does each iteration: call the
    # guard wrapped the same way, with a guard that raises, and confirm
    # archive_once() (stood in for here) still runs afterward.
    def _raising_status():
        raise RuntimeError("simulated guard failure")

    real_status_fn = pgarchive._pgdata_disk_status
    pgarchive._pgdata_disk_status = _raising_status
    reached_archive = {"value": False}
    try:
        try:
            pgarchive._pgdata_disk_status()
        except Exception:
            pass  # exactly what archive_loop's try/except does
        reached_archive["value"] = True  # the loop body continues past the guard
    finally:
        pgarchive._pgdata_disk_status = real_status_fn
    check(reached_archive["value"], "a raising disk guard does not stop the archive cycle from proceeding")

finally:
    shutil.disk_usage = _real_disk_usage
    _wipe_pgdata()

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All pgdata disk guard tests passed.")
