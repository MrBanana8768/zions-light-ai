"""
Adversarial: try to make the webui.db migration LOSE HER CHAT HISTORY.

The migration is the only step in the local-disk move that can destroy data,
because it is the only one where a bad local database can be published back
over a good snapshot. Everything else fails safe by construction. So this
file does not test that migration works — test_webuidb.py does that — it
tries to break it.

The attack surface, and it is small and specific:

    /data snapshot  --(1) restore-->  local live db  --(2) sync-->  /data

Step 1 can produce a local database that is WRONG BUT OPENABLE: truncated by
a full disk, half-copied when the pod was killed, or freshly created by
OpenWebUI because step 1 failed outright. Step 2 then publishes it over the
only durable copy. Every scenario below is a way to reach that state.

The property under test, stated once:

    A LOCAL DATABASE THAT IS MASSIVELY SMALLER THAN THE SNAPSHOT IT CAME
    FROM MUST NEVER SILENTLY REPLACE IT.

    python test_webuidb_migration.py
"""

import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="mig-local-"))
_SNAP_VOL = Path(tempfile.mkdtemp(prefix="mig-moosefs-"))
_QUAR = Path(tempfile.mkdtemp(prefix="mig-forensics-"))
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUAR)

import webuidb  # noqa: E402

FAILED = []
LOCAL = webuidb.LOCAL_DB
SNAP = webuidb.SNAPSHOT_DB


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)


def make_db(path: Path, n: int, marker: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("create table if not exists chat (id text, marker text, body text)")
    con.execute("create table if not exists user (id text)")
    con.executemany(
        "insert into chat values (?, ?, ?)",
        [(str(i), marker, "x" * 400) for i in range(n)],
    )
    con.execute("insert into user values ('a')")
    con.commit()
    con.close()


def chats(path: Path):
    return webuidb._has_rows(path)


def wipe():
    for p in (LOCAL, SNAP):
        for suffix in ("",) + webuidb.SIDECARS:
            f = p.with_name(p.name + suffix)
            if f.exists():
                f.unlink()


print("=" * 66)
print("ADVERSARIAL: can the migration lose her chat history?")
print("=" * 66)

# ---------------------------------------------------------------------------
print()
print("[A1] a HALF-COPIED local database must not overwrite the snapshot")
# The pod is killed mid-copy, or the overlay fills. SQLite databases are
# page-structured, so a copy truncated on a page boundary can open cleanly
# and report FEWER ROWS rather than an error. That is the dangerous shape:
# not corruption, which is caught, but plausible-looking loss.
wipe()
make_db(SNAP, 400, "HER-REAL-HISTORY")
real = chats(SNAP)
webuidb.restore_on_boot()
full_size = LOCAL.stat().st_size
with open(LOCAL, "r+b") as fh:          # truncate to a page boundary
    fh.truncate((full_size // 4096 // 3) * 4096)
partial = chats(LOCAL)
print(f"       snapshot has {real} chats; the half-copied local has {partial}")
r = webuidb.sync_once(force=True)
check(
    chats(SNAP) == real,
    f"THE SNAPSHOT SURVIVED ({chats(SNAP)} chats, was {real}) - publishing a "
    f"half-copied database over the only durable copy is the one way this "
    f"design can destroy her history",
)
check(r["synced"] is False, f"the publish was refused (synced={r['synced']})")

# ---------------------------------------------------------------------------
print()
print("[A2] a database OpenWebUI created fresh after a failed restore")
# restore_on_boot could not read the snapshot, OpenWebUI started anyway and
# built an empty schema, she sent one message. Now local has 1 chat and the
# snapshot has 400.
wipe()
make_db(SNAP, 400, "HER-REAL-HISTORY")
make_db(LOCAL, 1, "FRESH-EMPTY-START")
r = webuidb.sync_once(force=True)
check(
    chats(SNAP) == 400,
    f"THE SNAPSHOT SURVIVED ({chats(SNAP)} chats) - one new chat must not "
    f"replace four hundred",
)
check(r["synced"] is False, "the publish was refused")

# ---------------------------------------------------------------------------
print()
print("[A3] an EMPTY schema (zero chats) must not publish")
wipe()
make_db(SNAP, 400, "HER-REAL-HISTORY")
make_db(LOCAL, 0, "EMPTY")
r = webuidb.sync_once(force=True)
check(chats(SNAP) == 400 and not r["synced"], "zero-chat database refused")

# ---------------------------------------------------------------------------
print()
print("[A4] but a LEGITIMATE small change must still publish")
# The guard must not be so strict that ordinary use stops being durable.
# Deleting a few chats is normal; the snapshot must follow.
wipe()
make_db(SNAP, 400, "SNAP")
make_db(LOCAL, 380, "LOCAL-AFTER-SOME-DELETIONS")
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True and chats(SNAP) == 380,
    f"a 5% reduction published normally (synced={r['synced']}, "
    f"snapshot={chats(SNAP)}) - a guard that blocks ordinary deletion would "
    f"silently stop backing her up",
)

# ---------------------------------------------------------------------------
print()
print("[A5] growth always publishes")
wipe()
make_db(SNAP, 400, "SNAP")
make_db(LOCAL, 420, "LOCAL-GREW")
r = webuidb.sync_once(force=True)
check(r["synced"] is True and chats(SNAP) == 420, "normal growth published")

# ---------------------------------------------------------------------------
print()
print("[A6] a deliberate mass deletion can still be published, explicitly")
# If she really does clear her history, the operator must be able to make the
# snapshot follow - refusing forever would be its own failure. It just must
# not happen by accident.
wipe()
make_db(SNAP, 400, "SNAP")
make_db(LOCAL, 2, "SHE-REALLY-DELETED-THEM")
blocked = webuidb.sync_once(force=True)
check(blocked["synced"] is False, "blocked by default")
os.environ["WEBUI_DB_ALLOW_SHRINK"] = "1"
try:
    webuidb._reload_env()
    allowed = webuidb.sync_once(force=True)
finally:
    os.environ.pop("WEBUI_DB_ALLOW_SHRINK", None)
    webuidb._reload_env()
check(
    allowed["synced"] is True and chats(SNAP) == 2,
    f"an operator can override deliberately (synced={allowed['synced']})",
)

# ---------------------------------------------------------------------------
print()
print("[A7] the refused local database is PRESERVED, not discarded")
wipe()
make_db(SNAP, 400, "SNAP")
make_db(LOCAL, 1, "REFUSED-BUT-PRECIOUS")
before = len(list(_QUAR.iterdir()))
webuidb.sync_once(force=True)
check(
    len(list(_QUAR.iterdir())) > before,
    "a refused local database is copied aside - it may hold the only copy of "
    "whatever was written since the last good sync",
)

# ---------------------------------------------------------------------------
print()
print("[A8] a snapshot that vanishes mid-migration is not fatal")
wipe()
make_db(SNAP, 400, "SNAP")
r1 = webuidb.restore_on_boot()
SNAP.unlink()
r2 = webuidb.sync_once(force=True)
check(
    r1["action"] == "restored_from_snapshot" and r2["synced"] is True,
    "with no previous snapshot to compare against, the local database "
    "republishes cleanly rather than deadlocking",
)

# ---------------------------------------------------------------------------
print()
print("[A9] a zero-byte local database is not mistaken for a real one")
wipe()
make_db(SNAP, 400, "SNAP")
LOCAL.parent.mkdir(parents=True, exist_ok=True)
LOCAL.write_bytes(b"")
r = webuidb.restore_on_boot()
check(
    r["action"] == "restored_from_snapshot" and chats(LOCAL) == 400,
    f"a 0-byte local file is replaced from the snapshot (action={r['action']})",
)

# ---------------------------------------------------------------------------
print()
print("[A10] a read-only local directory fails loudly, without touching /data")
wipe()
make_db(SNAP, 400, "SNAP")
# NOT chmod: these tests run as root in the production image, and root
# ignores directory permission bits - the "read-only" directory was silently
# writable and the case proved nothing. Making the parent a regular FILE
# fails for everyone, root included.
import shutil as _sh
_sh.rmtree(LOCAL.parent, ignore_errors=True)
LOCAL.parent.write_bytes(b"not a directory")
try:
    r = webuidb.restore_on_boot()
    published = webuidb.sync_once(force=True)
finally:
    LOCAL.parent.unlink()
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
check(
    chats(SNAP) == 400,
    f"the snapshot is untouched when local disk is unusable "
    f"({chats(SNAP)} chats)",
)
check(
    not published["synced"],
    "nothing was published from a database that could not be created",
)

print()
if FAILED:
    print("!" * 66)
    for f in FAILED:
        print("FAIL " + f)
    print("!" * 66)
    sys.exit(1)
print("=" * 66)
print("The migration survived every attempt to make it lose history.")
print("=" * 66)
