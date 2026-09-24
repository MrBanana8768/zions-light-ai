"""ADVERSARIAL: webuidb restore/sync guards (1c8c04c) + crash consistency.

  docker compose -f docker-compose.tests.yml run --rm --entrypoint /bin/bash \
    unit-tests -c 'cp -r /src /work && cd /work/compactor && \
    /opt/compactor-venv/bin/python /work/tests/adversarial/test_adv_v319_webuidb.py'

Attacks the premises the eleven mutations could not model:
  B1  "the snapshot is written by sqlite3's backup API, which produces a
       self-contained database"  -- false on the ONE boot this commit exists
       to unblock (WEBUI_DB_LOCAL false -> true).
  B2  a LEGITIMATE shrink against the new byte guard.
  B3  SIGKILL mid-sync_once: what is left on /data.
  B4  os.replace() leaves foreign sidecars beside the new snapshot.
  B5  the WEBUI_DB_ALLOW_EMPTY_START safety claim, quantified.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="adv-webuidb-"))
(ROOT / "local").mkdir()
(ROOT / "data").mkdir()

os.environ["WEBUI_LOCAL_DB"] = str(ROOT / "local" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(ROOT / "data" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(ROOT / "data" / "quarantine")

sys.path.insert(0, "/work/compactor")
import webuidb  # noqa: E402

# The module may name the quarantine dir itself; keep it under our root.
webuidb.QUARANTINE = ROOT / "data" / "quarantine"

LOCAL = webuidb.LOCAL_DB
SNAP = webuidb.SNAPSHOT_DB

BROKEN: list[str] = []


def broke(cond, label):
    if cond:
        print(f"  *** BROKE: {label}")
        BROKEN.append(label)
    else:
        print(f"  (held)   {label}")


def note(msg):
    print(f"    . {msg}")


def make_openwebui_db(path: Path, chats: int, blob_kb: int = 4,
                      wal: bool = False, autockpt: bool = True):
    """An OpenWebUI-shaped database: one whole conversation per `chat` row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    if wal:
        con.execute("PRAGMA journal_mode=WAL")
        if not autockpt:
            con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("create table if not exists chat "
                "(id text primary key, updated_at int, chat text)")
    for i in range(chats):
        con.execute("insert or replace into chat values (?,?,?)",
                    (f"c{i}", i, "x" * (blob_kb * 1024)))
    con.commit()
    return con


def chats_in(path: Path):
    return webuidb._has_rows(path)


def reset():
    for p in (LOCAL, SNAP):
        for suf in ("", "-wal", "-shm", "-journal"):
            f = Path(str(p) + suf)
            if f.exists():
                f.unlink()
    q = webuidb.QUARANTINE
    if q.exists():
        shutil.rmtree(q)


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("B1  THE MIGRATION BOOT: THE SNAPSHOT IS *NOT* A BACKUP-API IMAGE")
print("=" * 74)
print("""
restore_on_boot: `shutil.copy2(SNAPSHOT_DB, LOCAL_DB)` with

    # Sidecars are deliberately NOT copied: the snapshot is written by
    # sqlite3's backup API, which produces a self-contained database.

That premise holds for every snapshot this module wrote. It does NOT hold on
the FIRST boot after WEBUI_DB_LOCAL flips false->true -- the commit's own
stated purpose ("what unblocks the step-6 migration"). On that boot
/data/openwebui/webui.db is the database OPENWEBUI has been writing directly
for the whole v3.1.4.x series, in WAL mode, and a RunPod redeploy is a
SIGKILL, so a -wal sits beside it.
""")

reset()
# Case (a): a hot WAL with NO other connection open -- the clean SIGKILL case.
con = make_openwebui_db(SNAP, 3, blob_kb=8, wal=True, autockpt=False)
con.execute("insert into chat values ('hot1', 99, ?)", ("y" * 8192,))
con.execute("insert into chat values ('hot2', 99, ?)", ("y" * 8192,))
con.commit()
wal_path = Path(str(SNAP) + "-wal")
note(f"before kill: main={SNAP.stat().st_size}B  "
     f"wal={wal_path.stat().st_size if wal_path.exists() else 0}B")
# Emulate SIGKILL: abandon the connection object without close(). CPython
# would finalise it at GC, so sever it the way a killed process does.
os.close(os.dup(0))  # no-op, keeps linters quiet
del con
import gc  # noqa: E402
gc.collect()
note(f"after abandoning writer: wal exists={wal_path.exists()}")

r = webuidb.restore_on_boot()
note(f"restore_on_boot -> {r['action']}")
got = chats_in(LOCAL)
note(f"local now has {got} chats (snapshot had 5: 3 + 2 written into the WAL)")
broke(got is not None and got < 5,
      f"B1a: the WAL-resident rows did not survive the restore "
      f"({got} of 5 chats)")

# Case (b): another connection is still open (OpenWebUI alive, or a killed
# process whose file lock has not been reaped) -- no checkpoint on close.
reset()
con = make_openwebui_db(SNAP, 3, blob_kb=8, wal=True, autockpt=False)
con.execute("insert into chat values ('hot1', 99, ?)", ("y" * 8192,))
con.execute("insert into chat values ('hot2', 99, ?)", ("y" * 8192,))
con.commit()
holder = sqlite3.connect(str(SNAP))          # a second, live reader
holder.execute("select 1").fetchone()
r = webuidb.restore_on_boot()
got_b = chats_in(LOCAL)
note(f"restore_on_boot -> {r['action']}; local has {got_b} chats")
broke(got_b is not None and got_b < 5,
      f"B1b: with a live second connection the WAL is not checkpointed and "
      f"the restore copies the main file alone ({got_b} of 5 chats)")
holder.close()
try:
    con.close()
except Exception:
    pass


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("B2  A LEGITIMATE SHRINK, AGAINST THE NEW BYTE GUARD")
print("=" * 74)
print("""
SHRINK_GUARD_MIN_BYTES compares SNAPSHOT_DB.stat().st_size against the new
image. She deletes one long conversation and OpenWebUI/maintenance VACUUMs --
an ordinary, intended action on a pod whose single biggest row is 32.95 MB.
The file more than halves. The guard cannot tell that from a failed restore,
so it refuses, and keeps refusing every SYNC_INTERVAL_S.
""")
reset()
webuidb._reload_env()
con = make_openwebui_db(LOCAL, 4, blob_kb=64)      # ~256 KB of blobs
con.close()
out = webuidb.sync_once(force=True)
note(f"first publish: synced={out['synced']} bytes={out['bytes']}")

con = sqlite3.connect(str(LOCAL))
con.execute("delete from chat where id in ('c0','c1','c2')")   # she clears 3
con.commit()
con.execute("VACUUM")
con.close()
note(f"after delete+VACUUM local={LOCAL.stat().st_size}B "
     f"snapshot={SNAP.stat().st_size}B chats={chats_in(LOCAL)}")

refusals = 0
quarantined = 0
for cycle in range(3):
    os.utime(LOCAL, None)                     # she keeps chatting
    out = webuidb.sync_once()
    if out["error"]:
        refusals += 1
    q = webuidb.QUARANTINE
    quarantined = len(list(q.glob("*.refused-*"))) if q.exists() else 0
note(f"refusals in 3 cycles: {refusals}; quarantine copies: {quarantined}")
broke(refusals == 3,
      f"B2: a deliberate, legitimate clear-and-VACUUM is refused on every "
      f"cycle ({refusals}/3) and the only escape is WEBUI_DB_ALLOW_SHRINK, "
      f"which also disables the unreadable-snapshot refusal")
broke(quarantined >= 3,
      f"B2b: each refusal copies the WHOLE local database to /data "
      f"({quarantined} copies in 3 cycles) -- onto the volume whose capacity "
      f"is the failure mode this module exists for")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("B3  SIGKILL MID-sync_once: WHAT IS LEFT ON /data")
print("=" * 74)
print("""
  tmp = SNAPSHOT_DB.with_name(f"{SNAPSHOT_DB.name}.sync-{os.getpid()}")

The unlink lives in an `except:` block. A SIGKILL runs no except block, and
the next boot is a NEW PID, so it writes a different temp name. Nothing in
this module, in entrypoint.sh, or in the backup scripts ever removes
`webui.db.sync-*`.
""")
reset()
con = make_openwebui_db(LOCAL, 4, blob_kb=64)
con.close()
webuidb.sync_once(force=True)

killer = f"""
import os, sys
sys.path.insert(0, "/work/compactor")
os.environ["WEBUI_LOCAL_DB"] = {str(LOCAL)!r}
os.environ["WEBUI_SNAPSHOT_DB"] = {str(SNAP)!r}
import sqlite3, webuidb
_real = sqlite3.Connection.backup
def _die(self, dst, **kw):
    _real(self, dst, **kw)
    os._exit(137)          # SIGKILL semantics: no finally, no except
sqlite3.Connection.backup = _die
os.utime({str(LOCAL)!r}, None)
webuidb.sync_once(force=True)
"""
for _ in range(3):
    subprocess.run([sys.executable, "-c", killer], capture_output=True)
strays = sorted(p.name for p in SNAP.parent.glob("webui.db.sync-*"))
note(f"strays on /data after 3 killed syncs: {strays}")
stray_bytes = sum(p.stat().st_size for p in SNAP.parent.glob("webui.db.sync-*"))
note(f"bytes of orphaned full-size database images: {stray_bytes}")
# a healthy sync afterwards must not clean them either
os.utime(LOCAL, None)
webuidb.sync_once(force=True)
still = sorted(p.name for p in SNAP.parent.glob("webui.db.sync-*"))
broke(len(still) >= 2,
      f"B3: {len(still)} orphaned full-size snapshot images accumulate on "
      f"/data, one per killed sync, and a later SUCCESSFUL sync does not "
      f"remove them ({still})")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("B4  os.replace() LEAVES FOREIGN SIDECARS BESIDE THE NEW SNAPSHOT")
print("=" * 74)
print("""
restore_on_boot refuses to copy sidecars because "a journal beside it would
belong to a different generation of the file, and applying one to the other
is how a good database becomes a bad one". sync_once creates exactly that
state: os.replace(tmp, SNAPSHOT_DB) swaps the main file and leaves any
-wal/-shm/-journal that was already there untouched.
""")
reset()
con = make_openwebui_db(SNAP, 3, blob_kb=8, wal=True, autockpt=False)
con.execute("insert into chat values ('hot', 1, 'zzz')")
con.commit()
del con
gc.collect()
sidecars_before = [Path(str(SNAP) + s).name
                   for s in webuidb.SIDECARS if Path(str(SNAP) + s).exists()]
note(f"sidecars beside the snapshot before publish: {sidecars_before}")
con = make_openwebui_db(LOCAL, 6, blob_kb=64)
con.close()
os.utime(LOCAL, None)
out = webuidb.sync_once(force=True)
sidecars_after = [Path(str(SNAP) + s).name
                  for s in webuidb.SIDECARS if Path(str(SNAP) + s).exists()]
note(f"publish synced={out['synced']} err={out['error']}")
note(f"sidecars beside the snapshot after publish: {sidecars_after}")
broke(bool(sidecars_after) and out["synced"],
      f"B4: the snapshot was replaced wholesale and {sidecars_after} from the "
      f"PREVIOUS generation were left beside it -- the exact state the "
      f"restore path refuses to create")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("B5  WEBUI_DB_ALLOW_EMPTY_START: 'THE SYNC DAEMON STAYS ON ... IS SAFE'")
print("=" * 74)
print("""
entrypoint.sh, on the escape hatch:

    "The sync daemon stays ON, and that is safe rather than an oversight:
     every route out of this state is refused by webuidb.sync_once"

The byte guard refuses only while new_bytes < prev_bytes * 0.5. An
empty-started database that simply GROWS past half the snapshot's size is
published -- with none of her history in it.
""")
reset()
con = make_openwebui_db(SNAP, 1, blob_kb=1024)     # her 1-row history
con.close()
snap_bytes = SNAP.stat().st_size
note(f"snapshot: {chats_in(SNAP)} chat, {snap_bytes} bytes")

published_at = None
for kb in (16, 128, 256, 512, 700, 900):
    if LOCAL.exists():
        LOCAL.unlink()
    con = make_openwebui_db(LOCAL, 1, blob_kb=kb)   # empty start + new chatting
    con.close()
    os.utime(LOCAL, None)
    out = webuidb.sync_once(force=True)
    ratio = LOCAL.stat().st_size / snap_bytes
    note(f"local {LOCAL.stat().st_size:>8}B ({ratio:.0%} of snapshot) -> "
         f"synced={out['synced']}")
    if out["synced"]:
        published_at = ratio
        break
broke(published_at is not None,
      f"B5: an empty-started database holding NONE of her history published "
      f"over the snapshot once it reached {published_at:.0%} of its size "
      f"-- the guard is a ratio, not a content check, so it buys days, "
      f"not safety" if published_at else "B5: did not publish")

# and the composition the error text itself suggests
os.environ["WEBUI_DB_ALLOW_SHRINK"] = "1"
webuidb._reload_env()
reset()
con = make_openwebui_db(SNAP, 1, blob_kb=1024)
con.close()
con = make_openwebui_db(LOCAL, 1, blob_kb=4)
con.close()
os.utime(LOCAL, None)
out = webuidb.sync_once(force=True)
broke(out["synced"],
      "B5b: WEBUI_DB_ALLOW_SHRINK=1 -- the value the refusal message itself "
      "tells the operator to set -- disables the byte guard AND the "
      "unreadable-snapshot refusal together, publishing a 4 KB database over "
      "a 1 MB history")
os.environ.pop("WEBUI_DB_ALLOW_SHRINK", None)
webuidb._reload_env()

print()
print("=" * 74)
print(f"BREAKS REPRODUCED: {len(BROKEN)}")
for b in BROKEN:
    print(f"  - {b}")
print("=" * 74)
