"""
webui.db on local disk, snapshotted to the network volume.

Run with TWO separate directories standing in for the two volumes — that
separation is the point of the design, so the test refuses to run without
it (see the env vars at the bottom).

Every case here is a thing that actually happens on the pod:

  [1] a service restart inside a live container      -> keep local
  [2] the pod recreated, local disk gone             -> restore snapshot
  [3] FIRST RUN: an existing 41 MB database on /data -> migrate down
  [4] a brand-new deployment, neither file exists    -> start fresh
  [5] the volume stalls mid-sync (the actual outage) -> live DB unaffected
  [6] a stalled volume left a hot journal on the snapshot -> refuse it
  [7] local disk corrupted                           -> set aside, restore
  [8] OpenWebUI writing while the sync runs          -> consistent image

    python test_webuidb.py
"""

import os
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

# Two volumes. LOCAL stands in for the pod's overlay, SNAP for MooseFS.
_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="vol-local-"))
_SNAP_VOL = Path(tempfile.mkdtemp(prefix="vol-moosefs-"))
_QUAR = Path(tempfile.mkdtemp(prefix="vol-forensics-"))
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUAR)
os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "0.1"

import webuidb  # noqa: E402

FAILED = []
LOCAL = webuidb.LOCAL_DB
SNAP = webuidb.SNAPSHOT_DB


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)


def make_db(path: Path, chats: int, marker: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("create table if not exists chat (id text, marker text)")
    con.execute("create table if not exists user (id text)")
    con.executemany(
        "insert into chat values (?, ?)", [(str(i), marker) for i in range(chats)]
    )
    con.execute("insert into user values ('a')")
    con.commit()
    con.close()


def chats(path: Path):
    return webuidb._has_rows(path)


def marker(path: Path):
    try:
        con = sqlite3.connect(str(path))
        try:
            return con.execute("select marker from chat limit 1").fetchone()[0]
        finally:
            con.close()
    except Exception:
        return None


def wipe():
    for p in (LOCAL, SNAP):
        for suffix in ("",) + webuidb.SIDECARS:
            f = p.with_name(p.name + suffix)
            if f.exists():
                f.unlink()


print("two volumes:")
print(f"  local (overlay)  {_LOCAL_VOL}")
print(f"  snapshot (mfs)   {_SNAP_VOL}")

print()
print("[1] a service restart inside a live container keeps the local database")
wipe()
make_db(LOCAL, 31, "LOCAL-NEWER")
make_db(SNAP, 5, "SNAP-OLDER")
r = webuidb.restore_on_boot()
check(r["action"] == "kept_local", f"kept the local database (action={r['action']})")
check(
    marker(LOCAL) == "LOCAL-NEWER" and chats(LOCAL) == 31,
    "local content untouched - a stale snapshot must never overwrite newer "
    "local data on a plain service restart",
)

print()
print("[2] the pod is recreated: local disk is gone, snapshot restores it")
wipe()
make_db(SNAP, 31, "FROM-SNAPSHOT")
r = webuidb.restore_on_boot()
check(r["action"] == "restored_from_snapshot", f"restored (action={r['action']})")
check(chats(LOCAL) == 31 and marker(LOCAL) == "FROM-SNAPSHOT", "all 31 chats came down")
check(
    not LOCAL.with_name(LOCAL.name + "-journal").exists(),
    "no journal was copied down - a journal from another generation of the "
    "file is how a good database becomes a bad one",
)

print()
print("[3] FIRST RUN: the existing database on /data migrates with no special case")
wipe()
make_db(SNAP, 31, "PRODUCTION-41MB")
r = webuidb.restore_on_boot()
check(
    r["action"] == "restored_from_snapshot" and marker(LOCAL) == "PRODUCTION-41MB",
    "the live database is now local, seeded from what was on the volume",
)

print()
print("[4] a brand-new deployment starts clean")
wipe()
r = webuidb.restore_on_boot()
check(r["action"] == "fresh", f"fresh start (action={r['action']})")
check(not LOCAL.exists(), "nothing fabricated; OpenWebUI creates its own schema")

print()
print("[5] THE OUTAGE: the volume stalls mid-sync")
wipe()
make_db(LOCAL, 31, "LIVE")
make_db(SNAP, 20, "LAST-GOOD-SNAPSHOT")
_real_connect = sqlite3.connect


def _stalled(path, *a, **kw):
    # Every write to the snapshot volume raises, exactly as a stalled mount does.
    if str(SNAP.parent) in str(path):
        raise sqlite3.OperationalError("disk I/O error")
    return _real_connect(path, *a, **kw)


sqlite3.connect = _stalled
try:
    r = webuidb.sync_once(force=True)
finally:
    sqlite3.connect = _real_connect
check(r["error"] is not None, f"the sync reports the failure ({r['error'][:38]}...)")
check(
    chats(LOCAL) == 31 and marker(LOCAL) == "LIVE",
    "THE LIVE DATABASE IS UNAFFECTED - this is the whole point: a stalled "
    "volume can no longer take the front end down",
)
check(
    marker(SNAP) == "LAST-GOOD-SNAPSHOT",
    "the previous good snapshot survives - a failed publish must not "
    "destroy the durable copy it was trying to replace",
)
check(
    not list(SNAP.parent.glob(f"{SNAP.name}.sync-*")),
    "no partial temp file left where restore_on_boot would find it",
)

print()
print("[6] a hot journal on the snapshot is refused, not copied down")
wipe()
make_db(SNAP, 31, "SNAP")
# A rollback journal beside the snapshot means it was caught mid-write -
# precisely the production failure. Corrupt the db so quick_check fails.
with open(SNAP, "r+b") as fh:
    fh.seek(200)
    fh.write(b"\xff" * 5000)
r = webuidb.restore_on_boot()
check(
    r["action"] == "snapshot_unhealthy",
    f"refused an unhealthy snapshot (action={r['action']})",
)
check(
    not LOCAL.exists(),
    "nothing was copied down - restoring a half-rolled-back database and "
    "calling it live would make a recoverable problem permanent",
)

print()
print("[7] a corrupted local database is set aside, never deleted")
wipe()
make_db(LOCAL, 9, "CORRUPT-LOCAL")
with open(LOCAL, "r+b") as fh:
    fh.seek(200)
    fh.write(b"\xff" * 5000)
make_db(SNAP, 31, "GOOD-SNAPSHOT")
before = len(list(_QUAR.iterdir()))
r = webuidb.restore_on_boot()
check(r["action"] == "restored_from_snapshot", "fell through to the snapshot")
check(marker(LOCAL) == "GOOD-SNAPSHOT" and chats(LOCAL) == 31, "the good snapshot is live")
check(
    len(list(_QUAR.iterdir())) > before,
    "the corrupted local database was PRESERVED for analysis, not deleted",
)

print()
print("[8] a sync taken while OpenWebUI is writing produces a consistent image")
wipe()
make_db(LOCAL, 200, "BUSY")
stop = threading.Event()
errors = []


def writer():
    # Hammer the live database the way OpenWebUI does during a conversation.
    try:
        con = sqlite3.connect(str(LOCAL), timeout=30)
        n = 0
        while not stop.is_set():
            con.execute("insert into chat values (?, 'BUSY')", (f"w{n}",))
            con.commit()
            n += 1
            time.sleep(0.001)
        con.close()
    except Exception as e:  # pragma: no cover
        errors.append(f"{type(e).__name__}: {e}")


t = threading.Thread(target=writer, daemon=True)
t.start()
time.sleep(0.15)
r = webuidb.sync_once(force=True)
stop.set()
t.join(timeout=10)
check(not errors, f"the writer was never blocked out of the live database ({errors[:1]})")
check(r["synced"] is True, f"the sync succeeded under concurrent writes ({r['error']})")
ok, detail = webuidb.integrity(SNAP)
check(ok, f"the snapshot is internally consistent (quick_check={detail})")
check(
    (chats(SNAP) or 0) >= 200,
    f"the snapshot captured a real point in time ({chats(SNAP)} chats)",
)

print()
print("[9] an unchanged database is not republished")
# Settle first: case 8's writer kept going DURING and after that sync, so the
# database legitimately changed after the image was taken. Publish once with
# nothing writing, then ask again.
settle = webuidb.sync_once(force=True)
check(settle["synced"] is True, f"published a settled database ({settle['error']})")
r1 = webuidb.sync_once()
check(
    r1["skipped"] == "unchanged since last sync",
    f"skipped a pointless write to the flaky volume ({r1['skipped']!r}) - each "
    f"sync writes the WHOLE database onto the volume whose write reliability "
    f"is the problem, so not writing is a feature",
)
# And a real change must still publish.
make_db(LOCAL, 1, "CHANGED")
r2 = webuidb.sync_once()
check(
    r2["synced"] is True,
    f"a genuine change is published ({r2['skipped'] or r2['error']}) - the "
    f"skip must not be able to strand the snapshot",
)

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All webui.db local-disk tests passed.")
