"""
v3.1.9.10 backup retention tiers (backup._keep_set / prune_old_backups).

The owner's ask: the last 3 hours of hourly backups, plus daily and weekly
history. Before, hourly backups + RETAIN_DAYS=14 (with the daily tier inside
the same window) + the template's floor RETAIN=28 kept ~117 archives, >20 GB.

The synthetic timeline is one archive per hour, at minute :10, for 20 days,
ending at 12:10 UTC on Tue 2026-09-29 (`NOW` is 12:30). Everything below is
worked out by hand from that calendar, NOT by re-running the implementation:

  hourly (age < 3h)            09-29 12:10, 11:10, 10:10                      3
  daily  (newest per UTC day,  09-28 .. 09-22 at 23:10                        7
          age <= 7d)           (09-29's newest is 12:10, already hourly)
  weekly (newest per ISO week) 09-20 23:10 and 09-13 23:10                    2
          (09-27 23:10 is the previous week's newest: already daily;
           the week of 08-31 lies before the timeline starts)
                                                                        = 12 of 480

    python test_backup_retention_tiers.py
"""

import datetime
import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="backup-retention-"))
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_TMP / "store")
os.environ["COMPACTOR_BACKUP_DIR"] = str(_TMP / "backups")
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_TMP / "webui.db")
os.environ["COMPACTOR_BACKUP_LOCAL_STAGING_DIR"] = str(_TMP / "staging")
os.environ["WEBUI_DB_QUARANTINE"] = str(_TMP / "q")
for k in ("COMPACTOR_BACKUP_RETAIN", "COMPACTOR_BACKUP_RETAIN_DAYS", "COMPACTOR_BACKUP_HOURLY_HOURS",
          "COMPACTOR_BACKUP_DAILY_DAYS", "COMPACTOR_BACKUP_GFS_WEEKS", "COMPACTOR_BACKUP_MIN_KEEP"):
    os.environ.pop(k, None)

import backup  # noqa: E402

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def eq(a, b, label):
    if a == b:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        FAILED.append(label)


def ts(y, mo, d, h, mi=10) -> int:
    return int(datetime.datetime(y, mo, d, h, mi, tzinfo=datetime.timezone.utc).timestamp())


NOW = ts(2026, 9, 29, 12, 30)
assert datetime.date(2026, 9, 29).weekday() == 1  # Tuesday: the calendar arithmetic above depends on it
NEWEST = ts(2026, 9, 29, 12, 10)


def timeline(hours=480):
    """Newest-first, as list_backups returns."""
    return [
        {"name": f"zions-backup-{NEWEST - i * 3600}.tar.gz", "path": "", "size_bytes": 1,
         "mtime": NEWEST - i * 3600}
        for i in range(hours)
    ]


ARCH = timeline()
BY_MTIME = {a["mtime"]: a["name"] for a in ARCH}


def names(*mtimes):
    return {BY_MTIME[m] for m in mtimes}


HOURLY = [ts(2026, 9, 29, 12), ts(2026, 9, 29, 11), ts(2026, 9, 29, 10)]
DAILY = [ts(2026, 9, d, 23) for d in (28, 27, 26, 25, 24, 23, 22)]
WEEKLY = [ts(2026, 9, 20, 23), ts(2026, 9, 13, 23)]

# ---------------------------------------------------------------------------
print("\n[R1] defaults: hourly 3h + daily 7d + weekly 8w, floor 3")
eq((backup.HOURLY_HOURS, backup.DAILY_DAYS, backup.GFS_WEEKS, backup.RETAIN_DAYS, backup.MIN_KEEP),
   (3, 7, 8, 0.0, 3), "code defaults; the age tier is OFF unless RETAIN_DAYS is set explicitly")
kept = backup._keep_set(ARCH, now=NOW, floor=3)
expected = names(*HOURLY, *DAILY, *WEEKLY)
eq(len(ARCH), 480, "fixture: 20 days of hourly archives")
eq(len(expected), 12, "hand-worked expectation: 12 archives")
eq(kept, expected, "the EXACT kept set (3 hourly + 7 daily + 2 weekly)")
check(len(kept) < 117, "far below the ~117 archives the old defaults kept")

# ---------------------------------------------------------------------------
print("\n[R2] the floor always holds")
kept = backup._keep_set(ARCH, now=NOW, floor=28)   # the template's RETAIN=28
newest28 = {a["name"] for a in ARCH[:28]}
check(newest28 <= kept, "floor 28 keeps the newest 28 archives (28 hours), swallowing the 3h tier")
eq(kept, newest28 | names(*DAILY[1:], *WEEKLY),
   "floor 28 + the daily/weekly tiers beyond it = 28 + 6 daily (09-27..09-22) + 2 weekly")
eq(len(kept), 36, "36 archives: this is what the shipped template (RETAIN=28) would still keep")
eq(backup._keep_set(ARCH, now=NOW, floor=1), backup._keep_set(ARCH, now=NOW, floor=3),
   "a floor below MIN_KEEP is raised to 3")
eq(backup._keep_set(ARCH, now=NOW, floor=0), backup._keep_set(ARCH, now=NOW, floor=3), "floor 0 is raised to 3")
ancient = [{"name": f"zions-backup-old{i}.tar.gz", "path": "", "size_bytes": 1,
            "mtime": NOW - (400 + i) * 86400} for i in range(6)]
eq(backup._keep_set(ancient, now=NOW, floor=3), {a["name"] for a in ancient[:3]},
   "six archives older than every tier: exactly the newest 3 survive")
eq(len(backup._keep_set(ancient[:2], now=NOW, floor=3)), 2, "fewer archives than the floor: all survive")
newest_only = [ARCH[0]]
eq(backup._keep_set(newest_only, now=NOW - 400 * 86400, floor=3), {ARCH[0]["name"]},
   "a single ancient archive is never pruned")

# ---------------------------------------------------------------------------
print("\n[R3] explicit COMPACTOR_BACKUP_RETAIN_DAYS keeps everything younger than that")
orig = backup.RETAIN_DAYS
try:
    backup.RETAIN_DAYS = 2.0
    kept = backup._keep_set(ARCH, now=NOW, floor=3)
    younger = {a["name"] for a in ARCH if (NOW - a["mtime"]) / 86400.0 <= 2.0}
    eq(len(younger), 48, "fixture: 48 archives are within 2 days")
    check(younger <= kept, "every archive younger than 2d is kept (the age tier)")
    eq(kept, younger | names(*DAILY[2:], *WEEKLY), "age tier + the daily archives beyond it + weekly")
    eq(len(kept), 55, "55 archives")
    backup.RETAIN_DAYS = 14.0
    kept = backup._keep_set(ARCH, now=NOW, floor=3)
    eq(len([a for a in ARCH if (NOW - a["mtime"]) / 86400.0 <= 14.0]), 336, "fixture: 336 within 14d")
    check(len(kept) >= 336, "the OLD default (14d) reproduces the old behaviour: 336+ archives when set explicitly")
finally:
    backup.RETAIN_DAYS = orig

# env parsing: unset/blank/garbage/zero/negative all mean OFF, positive turns it on
def _reload_with(env):
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(backup)
        return backup.RETAIN_DAYS, backup.HOURLY_HOURS, backup.DAILY_DAYS
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(backup)


for raw, want in ((None, 0.0), ("", 0.0), ("abc", 0.0), ("0", 0.0), ("-5", 0.0), ("14", 14.0), ("0.5", 0.5)):
    eq(_reload_with({"COMPACTOR_BACKUP_RETAIN_DAYS": raw})[0], want, f"RETAIN_DAYS={raw!r} -> {want}")
r = _reload_with({"COMPACTOR_BACKUP_HOURLY_HOURS": "6", "COMPACTOR_BACKUP_DAILY_DAYS": "10"})
eq((r[1], r[2]), (6.0, 10.0), "HOURLY_HOURS and DAILY_DAYS are configurable")
BY_MTIME = {a["mtime"]: a["name"] for a in ARCH}

# ---------------------------------------------------------------------------
print("\n[R4] windows are configurable; tiers can be off")
o = (backup.HOURLY_HOURS, backup.DAILY_DAYS)
try:
    backup.HOURLY_HOURS = 6
    kept = backup._keep_set(ARCH, now=NOW, floor=3)
    eq(len(kept), 12 + 3, "hourly 6h: three more archives (09:10, 08:10, 07:10)")
    backup.HOURLY_HOURS = 0
    kept = backup._keep_set(ARCH, now=NOW, floor=3)
    eq(kept, names(*HOURLY, *DAILY, *WEEKLY), "hourly off: the floor of 3 still keeps the same newest three")
    backup.HOURLY_HOURS, backup.DAILY_DAYS = 3, 0
    kept = backup._keep_set(ARCH, now=NOW, floor=3)
    eq(kept, names(*HOURLY, *WEEKLY, ts(2026, 9, 27, 23)),
       "daily off: hourly + weekly (09-27 23:10 is now the weekly newest of its week too)")
finally:
    backup.HOURLY_HOURS, backup.DAILY_DAYS = o

# ---------------------------------------------------------------------------
print("\n[R5] boundaries: 'younger than', one archive per UTC day")
pad = [{"name": f"p{i}", "path": "", "size_bytes": 1, "mtime": NOW - i * 60} for i in range(3)]
edge = [
    {"name": "just-inside", "path": "", "size_bytes": 1, "mtime": NOW - 3 * 3600 + 1},   # 2h59m59s old
    {"name": "exactly-3h", "path": "", "size_bytes": 1, "mtime": NOW - 3 * 3600},        # not YOUNGER than 3h
]
kept = backup._keep_set(pad + edge, now=NOW, floor=3)
check("just-inside" in kept, "2h59m59s old: kept by the hourly tier (the floor is used up by the pad)")
check("exactly-3h" not in kept, "exactly 3h old is not 'younger than' 3h, and its day/week are already claimed")
# two archives one UTC day, three days back: only the newest is claimed by the daily tier
two = [{"name": "late", "path": "", "size_bytes": 1, "mtime": ts(2026, 9, 26, 22)},
       {"name": "early", "path": "", "size_bytes": 1, "mtime": ts(2026, 9, 26, 2)}]
kept = backup._keep_set(pad + two, now=NOW, floor=3)
check("late" in kept and "early" not in kept, "two archives in one UTC day: only the newest is kept")
# UTC, not local: 23:59 and 00:01 either side of midnight are two days
mid = [{"name": "after", "path": "", "size_bytes": 1, "mtime": ts(2026, 9, 27, 0, 1)},
       {"name": "before", "path": "", "size_bytes": 1, "mtime": ts(2026, 9, 26, 23, 59)}]
kept = backup._keep_set(pad + mid, now=NOW, floor=3)
check("after" in kept and "before" in kept, "23:59 and 00:01 UTC are different days: both kept")

# ---------------------------------------------------------------------------
print("\n[R6] empty list")
eq(backup._keep_set([], now=NOW, floor=3), set(), "no archives -> empty set (no crash)")
d0 = _TMP / "empty-backups"
eq(backup.prune_old_backups(d0, now=NOW), [], "prune on a missing directory removes nothing")
d0.mkdir()
eq(backup.prune_old_backups(d0, now=NOW), [], "prune on an empty directory removes nothing")

# ---------------------------------------------------------------------------
print("\n[R7] prune on real files: never the newest, never an in-progress archive")
d = _TMP / "prune-backups"
d.mkdir()
for a in ARCH:
    p = d / a["name"]
    p.write_bytes(b"x")
    os.utime(p, (a["mtime"], a["mtime"]))
partial = d / "zions-backup-NEWER.tar.gz.partial"
partial.write_bytes(b"in progress")
os.utime(partial, (NOW, NOW))
staging = d / "zions-backup-20260929-123000-abc123"
staging.mkdir()
(staging / "payload").write_bytes(b"staging tree")
removed = backup.prune_old_backups(d, retain=3, now=NOW)
left = {p.name for p in d.glob("zions-backup-*.tar.gz")}
eq(len(removed), 480 - 12, "468 pruned")
eq(left, names(*HOURLY, *DAILY, *WEEKLY), "the remaining files are exactly the hand-worked set")
check(ARCH[0]["name"] in left, "the newest archive is never deleted")
check(partial.exists() and staging.is_dir(), "an in-progress .partial file and a staging dir are untouched")
eq(backup.prune_old_backups(d, retain=3, now=NOW), [], "idempotent: a second prune removes nothing")
# a template-style floor keeps more, and can never keep fewer
removed = backup.prune_old_backups(d, retain=0, now=NOW)
eq(removed, [], "retain=0 cannot remove anything below the 3-archive floor either")

# ---------------------------------------------------------------------------
print("\n[R9] tiers key on the timestamp in the archive NAME, not the file mtime (R3 L6)")
# The same 20-day hourly timeline as real zions-backup-YYYYMMDD-HHMMSS files,
# every one given the SAME mtime, as `cp` without -p / rsync / a tar restore
# leaves them. mtime-keyed tiers would call them all "just written" and keep
# all 480.
d9 = _TMP / "backups-r9"
shutil.rmtree(d9, ignore_errors=True)
d9.mkdir(parents=True)
same_mtime = NOW - 60
for i in range(480):
    t = NEWEST - i * 3600
    st = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    f = d9 / f"zions-backup-{st}.tar.gz"
    f.write_bytes(b"x")
    os.utime(f, (same_mtime, same_mtime))
listed = backup.list_backups(d9)
eq(len(listed), 480, "480 archives listed")
eq(listed[0]["name"], "zions-backup-20260929-121000.tar.gz", "newest-first order comes from the NAME (all mtimes equal)")
eq(listed[0]["when"], NEWEST, "`when` is the name's UTC timestamp")
removed9 = backup.prune_old_backups(d9, retain=3, now=NOW)
left9 = sorted(f.name for f in d9.glob("*.tar.gz"))
eq(len(left9), 12, "same tiering as the mtime timeline: 12 of 480 survive (3 hourly + 7 daily + 2 weekly)")
check("zions-backup-20260929-121000.tar.gz" in left9 and "zions-backup-20260928-231000.tar.gz" in left9
      and "zions-backup-20260913-231000.tar.gz" in left9, "and they are the right ones (hourly newest, daily 09-28, weekly 09-13)")
eq(len(removed9), 468, "468 removed")

# a name that does not parse falls back to the mtime
d10 = _TMP / "backups-r9b"
shutil.rmtree(d10, ignore_errors=True)
d10.mkdir(parents=True)
for nm, age_h in (("zions-backup-manual.tar.gz", 1), ("zions-backup-20261399-999999.tar.gz", 2), ("zions-backup-20260929-121000.tar.gz", 0)):
    f = d10 / nm
    f.write_bytes(b"x")
    os.utime(f, (NOW - age_h * 3600, NOW - age_h * 3600))
l10 = {e["name"]: e for e in backup.list_backups(d10)}
eq(l10["zions-backup-manual.tar.gz"]["when"], NOW - 3600, "an unparseable name falls back to mtime")
eq(l10["zions-backup-20261399-999999.tar.gz"]["when"], NOW - 7200, "an impossible date (month 13) falls back to mtime")
shutil.rmtree(d9, ignore_errors=True)
shutil.rmtree(d10, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\n[R8] the run_daemon banner reflects the tiers")
import inspect

src = inspect.getsource(backup.run_daemon)
check("hourly" in src and "daily" in src and "weekly" in src, "banner names the three tiers")

shutil.rmtree(_TMP, ignore_errors=True)
print()
if FAILED:
    print(f"FAILED {len(FAILED)}: {FAILED}")
    sys.exit(1)
print("ALL OK")
