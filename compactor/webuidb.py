"""
compactor.webuidb — keep OpenWebUI's SQLite database off the network volume.

THE PROBLEM. RunPod's MooseFS mount (`/data`) drops I/O occasionally. When
that lands while OpenWebUI is mid-transaction, SQLite leaves a hot rollback
journal; every subsequent open tries to roll it back, rolling back requires
WRITING, the write fails, and SQLite reports "attempt to write a readonly
database" plus disk I/O errors on plain SELECTs. Twice on 2026-08-31 (02:17
and ~04:30) that took the whole front end down.

The evidence that this is a local-vs-network problem and not a SQLite one:
through both incidents the compactor's own state on the SAME volume was
untouched — 2,080 facts, 749 indexed exchanges, zero unreadable. Many small
whole-file JSON writes survive a stalled mount. One large, continuously
journalled SQLite file does not.

THE SHAPE OF THE FIX.

    OpenWebUI  ->  /var/lib/openwebui/webui.db     (LOCAL disk, overlay)
                          |
                          |  sync_once(), every SYNC_INTERVAL_S
                          v
                   /data/openwebui/webui.db         (MooseFS, durable)

The live database and every journal it writes live on local disk, where
writes do not fail. `/data` holds a periodically refreshed snapshot, which is
what survives the pod being recreated. The snapshot keeps OpenWebUI's
original path, so reverting DATABASE_URL is a complete rollback.

NOT A SYMLINK, deliberately. Symlinking /data/openwebui/webui.db to local
disk looks equivalent and is not: SQLite derives the journal path from the
path it was given, so the journal can land next to the LINK — back on
MooseFS — and the failure we are fixing survives the fix. OpenWebUI is
pointed at a real local path via DATABASE_URL instead.

THE TRADE, stated plainly. Local disk is not persistent across pod
recreation, so SYNC_INTERVAL_S is the RPO for chat history. The daily/hourly
archive in backup.py is the second line. Postgres removes this trade
entirely and is the strategic answer; this removes the instability today.
"""

import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("compactor.webuidb")

# The live database: local disk, NOT /data.
LOCAL_DB = Path(os.environ.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db"))
# The durable snapshot: OpenWebUI's original path, so a rollback is just
# unsetting DATABASE_URL.
SNAPSHOT_DB = Path(
    os.environ.get("WEBUI_SNAPSHOT_DB", "/data/openwebui/webui.db")
)
SYNC_INTERVAL_S = float(os.environ.get("WEBUI_DB_SYNC_INTERVAL_S", "300") or 300)
# Where a local database that fails its integrity check is set aside. Never
# deleted: this project's rule is that anything removing state is reversible.
QUARANTINE = Path(os.environ.get("WEBUI_DB_QUARANTINE", "/data/forensics"))

SIDECARS = ("-journal", "-wal", "-shm")

# REGRESSION GUARD. Refuse to publish a snapshot holding less than this
# fraction of the chats the previous snapshot had.
#
# Found by adversarial test A2, and it is the one way this whole design can
# destroy her history: restore_on_boot fails (unreadable snapshot, full local
# disk, unwritable directory), OpenWebUI starts anyway and builds an empty
# schema, she sends ONE message - and the next sync cheerfully publishes a
# 1-chat database over the 400-chat snapshot that is the only durable copy.
# Every individual step behaves correctly. The composition loses everything.
#
# 0.5 is deliberately loose: ordinary use never halves a chat count, so this
# fires on catastrophe rather than on housekeeping. Deleting a few
# conversations must still reach the snapshot (A4) or the guard would quietly
# stop backing her up, which is its own data-loss mode.
SHRINK_REFUSE_BELOW = float(
    os.environ.get("WEBUI_DB_SHRINK_REFUSE_BELOW", "0.5") or 0.5
)
# Below this many chats in the PREVIOUS snapshot the ratio is meaningless
# (2 -> 1 is a 50% drop and means nothing), so the guard stands down.
SHRINK_GUARD_MIN_CHATS = int(
    os.environ.get("WEBUI_DB_SHRINK_GUARD_MIN_CHATS", "10") or 10
)
# The deliberate override: she really did clear her history and the snapshot
# must follow. Refusing forever would be its own failure.
ALLOW_SHRINK = (
    os.environ.get("WEBUI_DB_ALLOW_SHRINK", "").strip().lower()
    in ("1", "true", "yes")
)


def _reload_env() -> None:
    """Re-read the env-driven knobs. For tests, and for anyone who changes
    them without restarting the process."""
    global SHRINK_REFUSE_BELOW, SHRINK_GUARD_MIN_CHATS, ALLOW_SHRINK
    global SYNC_INTERVAL_S
    SHRINK_REFUSE_BELOW = float(
        os.environ.get("WEBUI_DB_SHRINK_REFUSE_BELOW", "0.5") or 0.5
    )
    SHRINK_GUARD_MIN_CHATS = int(
        os.environ.get("WEBUI_DB_SHRINK_GUARD_MIN_CHATS", "10") or 10
    )
    ALLOW_SHRINK = (
        os.environ.get("WEBUI_DB_ALLOW_SHRINK", "").strip().lower()
        in ("1", "true", "yes")
    )
    SYNC_INTERVAL_S = float(
        os.environ.get("WEBUI_DB_SYNC_INTERVAL_S", "300") or 300
    )


def _stamp() -> str:
    # Millisecond precision, not seconds: two set-asides inside the same
    # second collided on the filename and the second silently overwrote the
    # first. These files exist because something already went wrong; losing
    # one to a name clash is exactly the wrong time for that.
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def integrity(path: Path) -> tuple[bool, str]:
    """(ok, detail). Opening also replays/rolls back any journal, which on
    local disk always succeeds — that is the whole point of this module."""
    if not path.exists():
        return False, "missing"
    try:
        con = sqlite3.connect(str(path), timeout=30)
        try:
            verdict = con.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            con.close()
        return verdict == "ok", verdict
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _has_rows(path: Path) -> int | None:
    """Chat count, or None if unreadable. A database that opens but reports
    nothing is not something to publish over a good snapshot."""
    try:
        con = sqlite3.connect(str(path), timeout=30)
        try:
            return con.execute("select count(*) from chat").fetchone()[0]
        finally:
            con.close()
    except Exception:
        return None


def _set_aside(path: Path, why: str) -> None:
    """Move a database (and its sidecars) out of the way, never delete."""
    try:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
        stamp = _stamp()
        for suffix in ("",) + SIDECARS:
            p = path.with_name(path.name + suffix)
            if p.exists():
                shutil.move(str(p), str(QUARANTINE / f"{p.name}.{why}-{stamp}"))
        logger.warning(f"set aside {path} ({why}) -> {QUARANTINE}")
    except Exception as e:
        logger.error(f"could not set aside {path}: {type(e).__name__}: {e}")


def restore_on_boot() -> dict:
    """Make LOCAL_DB the live database before OpenWebUI starts.

    Order matters and each branch is a real case:

      1. A healthy local database wins outright. Within one container this is
         just a service restart, and local is by definition newer than any
         snapshot.
      2. A local database that fails quick_check is set aside (not deleted)
         and we fall through to the snapshot.
      3. The snapshot is copied down on a fresh container — the pod-recreate
         path, and the FIRST-RUN MIGRATION of the existing 41 MB database,
         which needs no special case because it is exactly this branch.
      4. Neither exists: a genuinely new deployment. OpenWebUI creates its
         own schema and the first sync publishes it.
    """
    result = {"action": None, "local": str(LOCAL_DB), "snapshot": str(SNAPSHOT_DB)}
    try:
        LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(
            f"cannot create {LOCAL_DB.parent} ({type(e).__name__}: {e}) — "
            f"OpenWebUI will fall back to whatever DATABASE_URL points at"
        )
        result["action"] = "error"
        return result

    if LOCAL_DB.exists():
        ok, detail = integrity(LOCAL_DB)
        # An EMPTY database passes quick_check. A 0-byte file is a valid
        # SQLite database with no tables, and so is a schema OpenWebUI built
        # after a failed restore - so "opens cleanly" is not enough to let a
        # local file win over the snapshot. Found by adversarial test A9,
        # where a 0-byte local file was kept and 400 chats on the volume were
        # ignored.
        local_chats = _has_rows(LOCAL_DB) if ok else None
        snap_chats = (
            _has_rows(SNAPSHOT_DB) if SNAPSHOT_DB.exists() else None
        )
        if ok and not local_chats and snap_chats:
            logger.error(
                f"local database opens cleanly but holds {local_chats!r} "
                f"chat(s) while the snapshot holds {snap_chats}. Treating it "
                f"as an empty shell (a 0-byte file, or a schema created after "
                f"a failed restore) and restoring the snapshot instead."
            )
            _set_aside(LOCAL_DB, "empty-shell")
        elif ok:
            logger.info(
                f"local database present and healthy ({LOCAL_DB}, "
                f"{LOCAL_DB.stat().st_size / 1e6:.1f} MB, "
                f"{local_chats} chats) — keeping it"
            )
            result["action"] = "kept_local"
            return result
        else:
            logger.error(
                f"local database failed quick_check ({detail}) — setting it "
                f"aside and restoring the snapshot"
            )
            _set_aside(LOCAL_DB, "failed-quickcheck")

    if SNAPSHOT_DB.exists():
        ok, detail = integrity(SNAPSHOT_DB)
        if not ok:
            # The snapshot lives on the flaky volume, so a hot journal here
            # is the exact production failure. Do NOT copy a half-rolled-back
            # database down and call it live.
            logger.error(
                f"snapshot {SNAPSHOT_DB} failed quick_check ({detail}). NOT "
                f"restoring it. Recover it first: "
                f"scripts/recover-webui-db.py, or restore from /data/backups."
            )
            result["action"] = "snapshot_unhealthy"
            return result
        try:
            shutil.copy2(SNAPSHOT_DB, LOCAL_DB)
            # Sidecars are deliberately NOT copied: the snapshot is written by
            # sqlite3's backup API, which produces a self-contained database.
            # A journal beside it would belong to a different generation of
            # the file, and applying one to the other is how a good database
            # becomes a bad one.
            size = LOCAL_DB.stat().st_size / 1e6
            logger.info(
                f"restored snapshot -> local ({size:.1f} MB, "
                f"{_has_rows(LOCAL_DB)} chats)"
            )
            result["action"] = "restored_from_snapshot"
        except Exception as e:
            logger.error(
                f"could not restore snapshot: {type(e).__name__}: {e}"
            )
            result["action"] = "restore_failed"
        return result

    logger.info(
        f"no local database and no snapshot — a new deployment; OpenWebUI "
        f"will create {LOCAL_DB} and the first sync will publish it"
    )
    result["action"] = "fresh"
    return result


def sync_once(force: bool = False) -> dict:
    """Publish LOCAL_DB to SNAPSHOT_DB, safely, while OpenWebUI is running.

    Three guards, each earned:

      * sqlite3's backup API, not a file copy. It takes a read lock and
        produces a consistent image of a database being written to. A cp of a
        live SQLite file can capture a torn page.
      * The snapshot is verified BEFORE it replaces the previous one, and a
        snapshot reporting zero chats is refused. Publishing a broken image
        over a good one would turn a local problem into a durable one.
      * Written to a temporary name in the destination directory and renamed
        into place, so a failure mid-write cannot leave a partial file where
        restore_on_boot will find it.
    """
    out = {"synced": False, "skipped": None, "error": None, "bytes": 0}
    if not LOCAL_DB.exists():
        out["skipped"] = "no local database yet"
        return out

    mtime = LOCAL_DB.stat().st_mtime
    if not force and SNAPSHOT_DB.exists():
        try:
            if SNAPSHOT_DB.stat().st_mtime >= mtime:
                # Nothing has been written since the last publish. Skipping
                # matters: each sync writes the whole database onto the
                # volume whose write reliability is the problem.
                out["skipped"] = "unchanged since last sync"
                return out
        except Exception:
            pass

    tmp = SNAPSHOT_DB.with_name(f"{SNAPSHOT_DB.name}.sync-{os.getpid()}")
    try:
        SNAPSHOT_DB.parent.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(str(LOCAL_DB), timeout=60)
        try:
            dst = sqlite3.connect(str(tmp), timeout=60)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()

        ok, detail = integrity(tmp)
        chats = _has_rows(tmp)
        if not ok:
            raise RuntimeError(f"snapshot failed quick_check: {detail}")
        if not chats:
            raise RuntimeError(f"snapshot reports {chats!r} chats; refusing to publish")

        # REGRESSION GUARD - see SHRINK_REFUSE_BELOW. Compare against what is
        # already published before replacing it.
        previous = _has_rows(SNAPSHOT_DB) if SNAPSHOT_DB.exists() else None
        if (
            previous
            and previous >= SHRINK_GUARD_MIN_CHATS
            and chats < previous * SHRINK_REFUSE_BELOW
            and not ALLOW_SHRINK
        ):
            # Keep the refused database. It may hold the only copy of
            # anything written since the last good sync, and this path fires
            # precisely when something has already gone wrong.
            try:
                QUARANTINE.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    LOCAL_DB, QUARANTINE / f"{LOCAL_DB.name}.refused-{_stamp()}"
                )
            except Exception:
                pass
            raise RuntimeError(
                f"REFUSING to publish: the local database has {chats} chat(s) "
                f"but the snapshot has {previous}. That is not ordinary use - "
                f"it is what a failed migration, a half-copied file or a "
                f"freshly created empty schema looks like, and publishing it "
                f"would overwrite the only durable copy of her history. The "
                f"local database has been copied to {QUARANTINE}. If this "
                f"shrink is real and intended, set WEBUI_DB_ALLOW_SHRINK=1."
            )

        os.replace(tmp, SNAPSHOT_DB)
        # Stamp the snapshot with the LOCAL mtime this image was taken from,
        # so "has anything changed since the last publish?" is a meaningful
        # question next cycle. Without this the snapshot carries the temp
        # file's own mtime, which can predate the local database's last write
        # (writes continue during the backup), and the skip never fires - so
        # every cycle rewrites the whole database onto the volume whose write
        # reliability is the entire problem.
        try:
            os.utime(SNAPSHOT_DB, (mtime, mtime))
        except Exception:
            pass  # a filesystem that refuses utime costs an extra sync, no more
        out["synced"] = True
        out["bytes"] = SNAPSHOT_DB.stat().st_size
        logger.info(
            f"published local -> snapshot ({out['bytes'] / 1e6:.1f} MB, "
            f"{chats} chats)"
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        # Expected whenever the volume is stalling — which is the condition
        # this module exists for. The LIVE database is local and unaffected;
        # only the durability window widens. Loud, but never fatal.
        logger.warning(
            f"snapshot publish failed ({out['error']}). The live database is "
            f"local and unaffected; retrying in {SYNC_INTERVAL_S:.0f}s. Chat "
            f"history is exposed to pod loss until this succeeds."
        )
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    return out


def sync_loop() -> None:
    """Daemon entry point (supervisord program `webuidb-sync`)."""
    logger.info(
        f"webui.db sync: {LOCAL_DB} -> {SNAPSHOT_DB} every "
        f"{SYNC_INTERVAL_S:.0f}s"
    )
    consecutive_failures = 0
    while True:
        time.sleep(SYNC_INTERVAL_S)
        r = sync_once()
        if r["error"]:
            consecutive_failures += 1
            if consecutive_failures in (3, 12, 48):
                logger.error(
                    f"snapshot publish has failed {consecutive_failures} times "
                    f"in a row ({consecutive_failures * SYNC_INTERVAL_S / 60:.0f} "
                    f"minutes without a durable copy of chat history). The "
                    f"volume is probably degraded; the live database is fine."
                )
        elif r["synced"]:
            consecutive_failures = 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if "--restore" in sys.argv:
        print(restore_on_boot())
    elif "--sync-once" in sys.argv:
        print(sync_once(force="--force" in sys.argv))
    elif "--status" in sys.argv:
        for label, p in (("local", LOCAL_DB), ("snapshot", SNAPSHOT_DB)):
            ok, detail = integrity(p)
            size = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "-"
            print(f"{label:9} {str(p):34} {size:>10}  quick_check={detail}  chats={_has_rows(p)}")
    else:
        sync_loop()
