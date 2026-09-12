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

from envcfg import env_float, env_int

logger = logging.getLogger("compactor.webuidb")

# The live database: local disk, NOT /data.
LOCAL_DB = Path(os.environ.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db"))
# The durable snapshot: OpenWebUI's original path, so a rollback is just
# unsetting DATABASE_URL.
SNAPSHOT_DB = Path(
    os.environ.get("WEBUI_SNAPSHOT_DB", "/data/openwebui/webui.db")
)
SYNC_INTERVAL_S = env_float("WEBUI_DB_SYNC_INTERVAL_S", 300)
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
SHRINK_REFUSE_BELOW = env_float("WEBUI_DB_SHRINK_REFUSE_BELOW", 0.5)
# The floor below which a ratio on chat COUNT says nothing. It was 10, and
# that is the number that disarmed this guard on the only pod it protects.
#
# The comparison already declines to fire at the small end without any help:
# at previous == 2 a surviving 1 chat is exactly half, not below it, and at
# previous == 1 every possible survivor is >= half of one. So a floor of 2
# costs nothing the arithmetic was not refusing anyway, and every value above
# it is the floor DELETING protection rather than adding sense to it. 10
# deleted all of it. This is a single-user pod where one conversation carries
# most of the traffic, so `chat` can sit in the single digits, and
# `previous >= 10` then makes the whole guard unreachable - permanently, and
# silently, in exactly the deployment it was written for. A FLOOR THAT
# EXCEEDS THE REAL ROW COUNT IS NOT A FLOOR, IT IS AN OFF SWITCH.
SHRINK_GUARD_MIN_CHATS = env_int("WEBUI_DB_SHRINK_GUARD_MIN_CHATS", 2)
# ...and here the row count is the wrong UNIT as well as the wrong floor, so
# there is a second measure. OpenWebUI stores an ENTIRE CONVERSATION as one
# JSON blob in one row of `chat` (scripts/chat-metrics.py reads it that way:
# `select id, updated_at, chat from chat`), and the conversation on this pod
# is tens of megabytes. A failed restore that leaves one freshly created
# conversation behind is therefore 1 row against 1 row - which NO ratio on
# counts can ever catch, at any floor - and tens of kilobytes against tens of
# megabytes, which this catches on the first sync.
#
# THE SECOND MEASURE IS STORED CONTENT, NOT FILE SIZE, AND THE DIFFERENCE IS
# THE DIFFERENCE BETWEEN A GUARD AND AN OUTAGE. This shipped comparing
# os.stat().st_size on the claim that "both files are written by sqlite3's
# backup API, which copies live pages only ... free-list churn and VACUUM do
# not move it". Both halves of that are false and were demonstrated false.
# sqlite3_backup_step copies EVERY page of the source, free-list pages
# included, so a database with 90% of its rows deleted backs up 41.026 MB ->
# 41.026 MB (ratio 1.0000) while a VACUUM of the same content gives 4.108 MB
# (ratio 0.1001). Driven against this very function: 40.07 MB -> VACUUM ->
# 10.03 MB -> REFUSED, five chats on both sides, not one byte of her history
# gone.
#
# AND VACUUM IS IN THIS REPO'S OWN RECOVERY PATH. scripts/recover-webui-db.py
# runs `con.execute("VACUUM")`; OPERATIONS.md tells the operator to
# integrity_check, VACUUM and copy back; and the refusal below names
# recover-webui-db.py in its own text. So the file-size measure blocked the
# documented recovery - every SYNC_INTERVAL_S, forever - and the only escape
# it offered was a flag that also disarmed the corruption guard.
#
# So compare what is STORED: the summed length of every column of `chat`
# (_content_bytes). A VACUUM rewrites the whole file and cannot change that
# sum; neither can free-list churn, page size, auto_vacuum, or the backup API.
# Deleting a conversation does, which is the entire point. st_size survives
# only as the cheap pre-filter for this floor - it is a stat() rather than a
# table scan, and stored content can never exceed the file holding it, so a
# file under the floor is under it on both measures and the scan is skipped.
SHRINK_GUARD_MIN_BYTES = env_int("WEBUI_DB_SHRINK_GUARD_MIN_BYTES", 65536)
# The deliberate override for the SHRINK refusal, and for nothing else: she
# really did clear her history and the snapshot must follow. Refusing forever
# would be its own failure.
#
# IT USED TO COVER THE UNREADABLE-SNAPSHOT REFUSAL TOO, on the reasoning that
# "from the operator's side they are one decision". They are not one decision,
# and the coupling is not a style point: THE SHRINK REFUSAL'S OWN TEXT TELLS
# THE OPERATOR TO SET THIS FLAG. So an operator who really did delete a
# conversation follows the instruction printed for them and, in the same
# keystroke, disarms the guard that stops a 1-chat database being published
# over her only durable copy - the guard this module exists for. An opt-in to
# a known-good shrink must not be an opt-out of corruption protection.
ALLOW_SHRINK = (
    os.environ.get("WEBUI_DB_ALLOW_SHRINK", "").strip().lower()
    in ("1", "true", "yes")
)
# The SIBLING override, separate on purpose (see above). It opens the
# unreadable-previous-snapshot refusal and nothing else, and nothing this
# module prints for any other reason names it - so if it is set, somebody read
# THAT refusal and decided about THAT risk.
ALLOW_PUBLISH_OVER_UNREADABLE = (
    os.environ.get("WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE", "").strip().lower()
    in ("1", "true", "yes")
)
# Written by entrypoint.sh when a restore failed and the operator set
# WEBUI_DB_ALLOW_EMPTY_START=true to boot anyway, onto a schema OpenWebUI
# builds from nothing. While it exists, sync_once refuses outright.
#
# It exists because the banner entrypoint.sh printed for that flag claimed
# "the sync daemon stays ON, and that is safe ... every route out of this
# state is refused by webuidb.sync_once". That was false, and measurably so:
# the shrink refusal is a RATIO, so an empty-started database is refused only
# until it grows past half the snapshot, and one was demonstrated publishing
# over the snapshot at 51% of its size. At ~2 MB/day against a 41 MB snapshot
# the flag bought about ten days, not safety - and it is a RunPod template
# variable, so once set to get a pod up it stays set through every later
# redeploy, printing its banner into a boot log nobody re-reads.
#
# ON LOCAL DISK, NOT /data, AND THAT IS THE POINT. A marker on the volume
# would outlive the incident and become the thing that blocks publishing
# forever - the defect this file is being repaired for, reintroduced by its
# own fix. On the overlay it dies with the pod, so it can only ever describe
# THIS boot. There is no env var to override it either: the escape is deleting
# the file the refusal names, which is a single act that cannot survive into
# the next incident the way an env var on a template does.
EMPTY_START_MARKER = Path(
    os.environ.get("WEBUI_DB_EMPTY_START_MARKER", "")
    or LOCAL_DB.with_name(".empty-start")
)

# What `--restore` exits with, per action restore_on_boot() can return. It is
# a module constant and not a dict literal buried in __main__ so that a test
# can assert every action string in the function appears here: see
# test_webuidb_restore_guard.py, which reads this module's own source to do
# it. An action missing from this table exits non-zero (see __main__), never
# zero - a new branch that forgets the table gets a refused boot rather than a
# silent pass.
RESTORE_EXIT_CODES = {
    "kept_local": 0,
    "restored_from_snapshot": 0,
    # A genuinely new deployment: no local database and no snapshot, so there
    # is nothing to lose and OpenWebUI building its own schema is correct.
    "fresh": 0,
    "snapshot_unhealthy": 3,
    "restore_failed": 4,
    "error": 5,
}


def _reload_env() -> None:
    """Re-read the env-driven knobs. For tests, and for anyone who changes
    them without restarting the process."""
    global SHRINK_REFUSE_BELOW, SHRINK_GUARD_MIN_CHATS, ALLOW_SHRINK
    global SHRINK_GUARD_MIN_BYTES, ALLOW_PUBLISH_OVER_UNREADABLE
    global SYNC_INTERVAL_S
    SHRINK_REFUSE_BELOW = env_float("WEBUI_DB_SHRINK_REFUSE_BELOW", 0.5)
    SHRINK_GUARD_MIN_CHATS = env_int("WEBUI_DB_SHRINK_GUARD_MIN_CHATS", 2)
    # The sibling knob. Every constant this function forgets is a knob that
    # works from the environment at import and then silently ignores a test
    # (or an operator) that sets it afterwards.
    SHRINK_GUARD_MIN_BYTES = env_int("WEBUI_DB_SHRINK_GUARD_MIN_BYTES", 65536)
    ALLOW_SHRINK = (
        os.environ.get("WEBUI_DB_ALLOW_SHRINK", "").strip().lower()
        in ("1", "true", "yes")
    )
    # The other half of the split override, and the one this function is most
    # likely to be missing next time somebody reads it: a flag reloaded here
    # and a flag that is not behave identically until the exact moment a test
    # sets the second one, which then does nothing and passes. Both refusal
    # flags are reloaded; test_webuidb_restore_guard.py [13] reads this
    # function's own source to keep it that way.
    ALLOW_PUBLISH_OVER_UNREADABLE = (
        os.environ.get("WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE", "").strip().lower()
        in ("1", "true", "yes")
    )
    SYNC_INTERVAL_S = env_float("WEBUI_DB_SYNC_INTERVAL_S", 300)


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
    nothing is not something to publish over a good snapshot.

    NONE IS AMBIGUOUS AND EVERY CALLER HAS TO SAY WHICH MEANING IT WANTS. It
    comes back for "no chat table", "corrupt header", "locked", "the volume
    stalled" - and NEVER for "zero chats", which is the integer 0. So a caller
    that tests this result for truth has collapsed None, 0 and "I could not
    look" into one answer. sync_once's regression guard did exactly that and
    stood down on the single state it exists to catch. Compare against None
    explicitly. And note this function cannot tell you whether the file is
    ABSENT - only path.exists() can, and absent and unreadable want opposite
    answers at the one call site that matters (publish vs refuse)."""
    try:
        con = sqlite3.connect(str(path), timeout=30)
        try:
            return con.execute("select count(*) from chat").fetchone()[0]
        finally:
            con.close()
    except Exception:
        return None


def _content_bytes(path: Path) -> int | None:
    """Stored length of every column of `chat`, or None if it cannot be read.

    THE SECOND MEASURE OF THE SHRINK GUARD, and it is deliberately not
    os.stat().st_size. A SQLite file's size is a function of its page churn as
    much as its content: sqlite3's backup API copies free-list pages, so a
    bloated database images at full size, and a VACUUM of exactly the same
    rows images at a tenth of it. Comparing file sizes therefore answers "has
    this file been repacked lately", and the question the guard needs answered
    is "is her conversation still in there". Summing the stored lengths asks
    the second one: a VACUUM rewrites the whole file without moving this sum
    by a byte, and deleting a conversation moves it immediately.

    Reads the column list rather than naming `chat`.`chat`, because the
    production schema (one JSON blob per conversation, per
    scripts/chat-metrics.py) and the suites' fixtures do not share column
    names, and a measure that silently returns 0 on a schema it does not
    recognise would disarm the guard exactly the way SHRINK_GUARD_MIN_CHATS=10
    once did. No `chat` table at all is None, not 0, for the same reason
    _has_rows() distinguishes them: "nothing there" and "I could not look"
    want opposite answers from the caller.

    COSTS A FULL TABLE SCAN, unlike _has_rows()'s count(*), which SQLite can
    serve from an index. Both files sync_once measures live on /data, so a
    publishing cycle at 41 MB reads roughly 82 MB more off the volume whose
    read reliability is this module's whole subject - call it double. That is
    why sync_once gates the call behind the st_size floor rather than
    measuring every cycle, and why it is not called at all on a cycle the
    mtime check skips. It is a real cost, accepted knowingly: the measure it
    replaced was free and wrong, and being wrong here means either refusing a
    documented recovery forever or publishing an empty database over her
    history.
    """
    try:
        con = sqlite3.connect(str(path), timeout=30)
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(chat)").fetchall()]
            if not cols:
                return None
            # length() is NULL for a NULL column and sum() skips a NULL row, so
            # without the coalesce a single NULL anywhere drops that whole row
            # from the measure - which is loss-shaped, in the direction that
            # makes the guard stand down.
            expr = " + ".join(
                'coalesce(length("{}"), 0)'.format(c.replace('"', '""'))
                for c in cols
            )
            row = con.execute(
                f"select coalesce(sum({expr}), 0) from chat"  # noqa: S608
            ).fetchone()
            return int(row[0])
        finally:
            con.close()
    except Exception:
        return None


def _set_aside(path: Path, why: str) -> bool:
    """Move a database (and its sidecars) out of the way, never delete.

    RETURNS FALSE IF IT COULD NOT, and the caller must then not go on to
    overwrite the file. This used to swallow the failure and return None,
    which read as "done" at both call sites in restore_on_boot: QUARANTINE is
    on /data, so a stalled or full volume - the condition this whole module
    exists for - failed the move, logged one line, and the shutil.copy2 two
    branches later landed the snapshot on top of the database we had just
    promised to preserve. That database is unreadable, not empty; a hot
    rollback journal is recoverable and may hold the only copy of everything
    written since the last sync."""
    try:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
        stamp = _stamp()
        for suffix in ("",) + SIDECARS:
            p = path.with_name(path.name + suffix)
            if p.exists():
                shutil.move(str(p), str(QUARANTINE / f"{p.name}.{why}-{stamp}"))
        logger.warning(f"set aside {path} ({why}) -> {QUARANTINE}")
        return True
    except Exception as e:
        logger.error(f"could not set aside {path}: {type(e).__name__}: {e}")
        return False


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

    Every failure RETURNS rather than raising, and that is deliberate: the
    "action" string is the contract that scripts/switch-webui-db-to-local.py
    and the suites branch on. It does mean nothing here can be detected by a
    shell - so __main__ maps each action to an exit code (RESTORE_EXIT_CODES)
    and entrypoint.sh refuses the boot on a non-zero one. Any new branch
    added below needs a row in that table; it is not optional, and an unknown
    action deliberately exits non-zero.

    The one case with no good answer is 3 failing while 1 has nothing to
    serve - an empty local database beside a damaged snapshot. It returns
    "snapshot_unhealthy" and stops the boot rather than letting OpenWebUI
    build a fresh schema: the snapshot is damaged, not gone, and it stays
    recoverable exactly as long as nothing publishes over it.
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
        if ok and not local_chats and SNAPSHOT_DB.exists() and snap_chats is None:
            # THE SIBLING OF THE SHRINK GUARD, and it arrives at the same
            # silently-empty boot by a different road. Local holds nothing to
            # serve, and the count we would have compared it against could not
            # be read - so `snap_chats` is falsy, the branch below does not
            # take, and the old code fell through to `elif ok:` and announced
            # "local database present and healthy (0 chats) - keeping it".
            # True, cheerful, and it never mentions that the only durable copy
            # of her history is unreadable.
            #
            # Damaged is not the same as merely holding no chat table: a
            # 0-byte or schema-only file on /data passes quick_check, is not
            # her history, and restoring it would gain nothing - so ask
            # integrity() rather than treating every None as a catastrophe,
            # or a stale empty file on the volume would block every boot.
            snap_ok, snap_detail = integrity(SNAPSHOT_DB)
            if not snap_ok:
                logger.error(
                    f"local database opens cleanly but holds {local_chats!r} "
                    f"chat(s), and the snapshot {SNAPSHOT_DB} failed "
                    f"quick_check ({snap_detail}) so it cannot be restored "
                    f"either. REFUSING to hand OpenWebUI an empty database "
                    f"while the durable copy is damaged: it would build a "
                    f"fresh schema, she would find her history gone, and the "
                    f"sync daemon would spend the rest of the boot trying to "
                    f"publish that. Recover it first: "
                    f"scripts/recover-webui-db.py, or restore from "
                    f"/data/backups."
                )
                result["action"] = "snapshot_unhealthy"
                return result
        if ok and not local_chats and snap_chats:
            logger.error(
                f"local database opens cleanly but holds {local_chats!r} "
                f"chat(s) while the snapshot holds {snap_chats}. Treating it "
                f"as an empty shell (a 0-byte file, or a schema created after "
                f"a failed restore) and restoring the snapshot instead."
            )
            if not _set_aside(LOCAL_DB, "empty-shell"):
                # See _set_aside's docstring. The copy2 below would land on a
                # file we just failed to preserve, and "empty shell" is our
                # reading of it, not a fact: a database whose `chat` table
                # cannot be read may still hold her accounts, settings and
                # uploads. Refuse rather than overwrite unpreserved state.
                logger.error(
                    f"could not move the empty local shell aside, so NOT "
                    f"restoring over it. Free space under {QUARANTINE} (or "
                    f"move {LOCAL_DB} by hand) and boot again."
                )
                result["action"] = "error"
                return result
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
            if not _set_aside(LOCAL_DB, "failed-quickcheck"):
                # The SIBLING of the branch above, and the more expensive one:
                # a database that fails quick_check is the recoverable case
                # this module was written for (a hot rollback journal, not
                # corruption), and it holds everything written since the last
                # sync. Overwriting it with the snapshot after failing to
                # preserve it converts a recoverable incident into a permanent
                # loss - quietly, because the move failed with one log line.
                logger.error(
                    f"could not move the failed local database aside, so NOT "
                    f"restoring over it - it is the only copy of anything "
                    f"written since the last sync and it is recoverable. Free "
                    f"space under {QUARANTINE} (or move {LOCAL_DB} by hand), "
                    f"then boot again."
                )
                result["action"] = "error"
                return result

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
            # A successful restore is the one event that ENDS an empty start:
            # local now holds exactly what the snapshot holds, so publishing it
            # back can lose nothing. Clearing the marker here is what keeps the
            # empty-start refusal from becoming the permanent block it exists
            # to replace - the operator who repairs the snapshot and re-runs
            # --restore gets their sync daemon back without having to know
            # about a dotfile. Best-effort: a marker we cannot remove costs one
            # deliberate `rm`, while a restore that failed because of one would
            # cost the boot.
            try:
                EMPTY_START_MARKER.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(
                    f"restored, but could not clear {EMPTY_START_MARKER}: "
                    f"{type(e).__name__}: {e} — the sync daemon will keep "
                    f"refusing until it is removed by hand"
                )
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

    Five guards, each earned:

      * sqlite3's backup API, not a file copy. It takes a read lock and
        produces a consistent image of a database being written to. A cp of a
        live SQLite file can capture a torn page.
      * The snapshot is verified BEFORE it replaces the previous one, and a
        snapshot reporting zero chats is refused. Publishing a broken image
        over a good one would turn a local problem into a durable one.
      * Written to a temporary name in the destination directory and renamed
        into place, so a failure mid-write cannot leave a partial file where
        restore_on_boot will find it.
      * The PREVIOUS snapshot is read before it is replaced - by chat count
        and by STORED CONTENT (not file size; see SHRINK_GUARD_MIN_BYTES), and
        "I could not read it" is a refusal rather than a shrug. The three
        bullets above all validate the NEW image; none of them ever looked at
        the file about to be destroyed, which is how a 1-chat database came to
        be publishable over a 2,300-chat one.
      * An empty start refuses outright while its marker exists, because a
        ratio cannot protect that state - it only delays it (see
        EMPTY_START_MARKER).

    EVERY ONE OF THOSE REFUSALS HAS A WAY OUT, and that is not softness. A
    guard with no exit becomes the data-loss mode it was written against: it
    stops publishing her history to /data at all, which is the same loss
    taking longer, announced only in a log line nobody is reading.
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
        if EMPTY_START_MARKER.exists() and SNAPSHOT_DB.exists():
            # THE RATIO CANNOT PROTECT THIS STATE. See EMPTY_START_MARKER: an
            # empty-started database is refused by the shrink guard only until
            # it grows past SHRINK_REFUSE_BELOW of the snapshot, and then it
            # publishes over the only copy of everything she has said. It was
            # demonstrated doing exactly that at 51%. A content comparison
            # narrows the window; it does not close it, because the content is
            # real - it is just NEW content standing where the old should be,
            # and no measure of size can tell those apart. The only honest
            # answer is that a human decides.
            #
            # Raised BEFORE the backup, not after: this refusal repeats every
            # SYNC_INTERVAL_S until someone acts, and the backup writes the
            # whole database onto /data each time it runs (~11.8 GB/day at
            # 41 MB). A guard whose refusal path fills the volume it guards
            # eventually defeats its own siblings.
            #
            # `and SNAPSHOT_DB.exists()` is not belt-and-braces. With no
            # snapshot there is no durable copy to protect and refusing would
            # mean never making one - the exact absent-vs-unreadable confusion
            # this function was repaired for, rebuilt one branch later.
            raise RuntimeError(
                f"REFUSING to publish: {EMPTY_START_MARKER} says this pod was "
                f"booted with WEBUI_DB_ALLOW_EMPTY_START=true after a failed "
                f"restore, so the local database is a schema OpenWebUI built "
                f"from nothing and {SNAPSHOT_DB} is still the only copy of her "
                f"history. Publishing local over it would complete the loss "
                f"the empty start only postponed. Recover the snapshot "
                f"(scripts/recover-webui-db.py, or /data/backups) and re-run "
                f"`webuidb.py --restore`, which clears this marker. To accept "
                f"starting over from nothing and let the empty database become "
                f"the durable one, delete {EMPTY_START_MARKER} - deliberately, "
                f"and knowing that the next sync overwrites the snapshot."
            )
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

        # REGRESSION GUARD - see SHRINK_REFUSE_BELOW. Compare the new image
        # against WHAT IS ALREADY PUBLISHED before replacing it.
        #
        # "previous" HAS THREE STATES, NOT TWO, AND THAT IS THE WHOLE GUARD.
        # This condition used to open with a bare `if previous and ...`, and
        # _has_rows() answers None both for "there is no snapshot yet" and for
        # "the snapshot is right there and I could not read it". So an
        # UNREADABLE durable copy - the precise failure this module exists for
        # - made the guard FALSY and the publish went straight over it. The
        # integrity checks above validate `tmp`, the new image; nothing in
        # this function was ever looking at the file about to be destroyed.
        #
        # Split apart, the two states want opposite answers: absent means
        # first publish, so go (test_webuidb_migration A8); unreadable means
        # the only durable copy is in trouble, so stop.
        #
        # EXCEPT THAT `previous` HAS FOUR STATES, NOT THREE, AND THE FIX FOR
        # THREE BROKE THE FOURTH. _has_rows() answers None for "no `chat`
        # table" exactly as it does for "corrupt / locked / stalled", and a
        # 0-BYTE FILE IS A VALID SQLITE DATABASE WITH NO TABLES: it opens
        # cleanly, it PASSES quick_check, and it has no chat count. Treating
        # every None as a catastrophe therefore refused an empty-but-healthy
        # snapshot every SYNC_INTERVAL_S, forever, and the durable copy was
        # never written at all - the whole failure this guard exists to
        # prevent, arrived at from the other direction, with every health
        # check green and one log line as the only signal.
        #
        # restore_on_boot's empty-local branch had already worked this out,
        # twelve lines away and in the same commit: "a 0-byte or schema-only
        # file on /data passes quick_check, is not her history, and restoring
        # it would gain nothing - so ask integrity() rather than treating
        # every None as a catastrophe". That is the sibling, and this is it
        # asking the same question. The four answers:
        #
        #   absent                 -> publish (A8 depends on it)
        #   readable, N rows       -> compare (below)
        #   unreadable (quick_check fails) -> refuse; it is RECOVERABLE until
        #                             something writes over it
        #   opens clean, no `chat` table -> publish; it holds no conversation
        #                             to lose, and refusing would strand the
        #                             snapshot on a file that will never
        #                             become readable on its own
        previous = None
        prev_bytes = 0
        if SNAPSHOT_DB.exists():
            previous = _has_rows(SNAPSHOT_DB)
            prev_bytes = SNAPSHOT_DB.stat().st_size
            if previous is None:
                prev_ok, prev_detail = integrity(SNAPSHOT_DB)
                if prev_ok:
                    # WARNING, not info: this is a normal-looking sentence for
                    # an abnormal file. It is the right answer, and it is also
                    # what a snapshot destroyed by something else looks like on
                    # the cycle before we overwrite it.
                    logger.warning(
                        f"the snapshot {SNAPSHOT_DB} "
                        f"({prev_bytes / 1e6:.1f} MB) opens cleanly but has no "
                        f"readable `chat` table - a 0-byte file, or a schema "
                        f"that never got one. There is no history there to "
                        f"lose, so this publish goes ahead and replaces it."
                    )
                elif not ALLOW_PUBLISH_OVER_UNREADABLE:
                    # Deliberately NO forensic copy of the local database here,
                    # unlike the shrink refusal below. The local file is not the
                    # suspect in this branch - the snapshot is - and this refusal
                    # repeats every SYNC_INTERVAL_S until a human acts, so copying
                    # the whole database onto /data each cycle would fill the
                    # volume that is already unhealthy. The local database stays
                    # where it is, live and unharmed.
                    #
                    # The flag named here is NOT WEBUI_DB_ALLOW_SHRINK. That
                    # one is printed by the shrink refusal below, where it is
                    # ordinary advice, and a single flag covering both meant
                    # the operator who took that advice silently disarmed THIS
                    # refusal too - the one the module exists for.
                    raise RuntimeError(
                        f"REFUSING to publish: {SNAPSHOT_DB} exists and failed "
                        f"quick_check ({prev_detail}) - a hot rollback journal, "
                        f"a corrupt header, a stalled mount. That file is the "
                        f"only durable copy of her history and it is "
                        f"RECOVERABLE in every one of those cases; overwriting "
                        f"it is not. The live database is local and "
                        f"unaffected. Recover it "
                        f"(scripts/recover-webui-db.py, or /data/backups), or "
                        f"move it aside and the next sync will republish from "
                        f"local. To publish over it deliberately, set "
                        f"WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE=1."
                    )

        if previous is not None and not ALLOW_SHRINK:
            new_bytes = tmp.stat().st_size
            # TWO MEASURES, because on this pod the row count alone cannot see
            # the loss: one conversation is one row (see SHRINK_GUARD_MIN_BYTES).
            lost_chats = (
                previous >= SHRINK_GUARD_MIN_CHATS
                and chats < previous * SHRINK_REFUSE_BELOW
            )
            # THE SECOND MEASURE IS STORED CONTENT, NOT st_size, AND THE
            # DIFFERENCE IS A SHIPPED OUTAGE. A VACUUM of unchanged content
            # took a 40.07 MB image to 10.03 MB and this refused it - five
            # chats on both sides, nothing lost - while a database with 90% of
            # its rows deleted images at full size and sails through, because
            # the backup API copies free-list pages. So the file size answered
            # "has this been repacked lately" and the guard needed "is her
            # conversation still in there". _content_bytes asks the second.
            #
            # st_size survives ONLY as the pre-filter for the floor: it is a
            # stat() rather than two table scans, and stored content can never
            # exceed the file holding it, so a snapshot under the floor by file
            # size is under it by content too and nothing is skipped that the
            # floor would not have excluded anyway.
            lost_content = False
            prev_content = new_content = None
            if prev_bytes >= SHRINK_GUARD_MIN_BYTES:
                prev_content = _content_bytes(SNAPSHOT_DB)
                new_content = _content_bytes(tmp)
                if prev_content is None or new_content is None:
                    # "I could not measure" is NOT a refusal. Turning an
                    # unreadable measurement into a permanent stop is the exact
                    # defect being repaired thirty lines above, and it costs the
                    # durable copy outright. Say so loudly and let the chat
                    # count decide - a narrower guard, honestly narrower.
                    logger.warning(
                        f"could not measure stored content (snapshot="
                        f"{prev_content!r}, new={new_content!r}); the shrink "
                        f"guard is running on chat count alone this cycle, "
                        f"which cannot see a one-row conversation being "
                        f"replaced by a one-row empty one."
                    )
                elif prev_content >= SHRINK_GUARD_MIN_BYTES:
                    lost_content = new_content < prev_content * SHRINK_REFUSE_BELOW
            if lost_chats or lost_content:
                # Keep the refused database. It may hold the only copy of
                # anything written since the last good sync, and this path
                # fires precisely when something has already gone wrong.
                try:
                    QUARANTINE.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        LOCAL_DB, QUARANTINE / f"{LOCAL_DB.name}.refused-{_stamp()}"
                    )
                except Exception:
                    pass
                _prev_desc = (
                    f"{previous} chat(s) holding {prev_content / 1e6:.1f} MB "
                    f"of content in a {prev_bytes / 1e6:.1f} MB file"
                    if prev_content is not None
                    else f"{previous} chat(s) in a {prev_bytes / 1e6:.1f} MB file"
                )
                _new_desc = (
                    f"{chats} chat(s) holding {new_content / 1e6:.1f} MB of "
                    f"content in a {new_bytes / 1e6:.1f} MB file"
                    if new_content is not None
                    else f"{chats} chat(s) in a {new_bytes / 1e6:.1f} MB file"
                )
                raise RuntimeError(
                    f"REFUSING to publish: the local database has {_new_desc} "
                    f"but the snapshot has {_prev_desc} "
                    f"({'chat count' if lost_chats else 'stored content'} "
                    f"tripped it). That is not ordinary use - it is what a "
                    f"failed migration, a half-copied file or a freshly "
                    f"created empty schema looks like, and publishing it would "
                    f"overwrite the only durable copy of her history. A VACUUM "
                    f"cannot cause this: what is compared is the stored length "
                    f"of the `chat` rows, which VACUUM rewrites without "
                    f"changing, so the recovery in OPERATIONS.md is safe to "
                    f"run and something is genuinely gone here. The local "
                    f"database has been copied to {QUARANTINE}. If this shrink "
                    f"is real and intended, set WEBUI_DB_ALLOW_SHRINK=1 (which "
                    f"opens THIS refusal only - an unreadable snapshot still "
                    f"stops the publish)."
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
        r = restore_on_boot()
        print(r)
        # THE EXIT CODE IS THE ONLY THING THE BOOT SCRIPT CAN SEE, and this
        # was `print(restore_on_boot())` with no sys.exit at all - so every
        # failure exited 0. restore_on_boot() does not raise, deliberately
        # (scripts/switch-webui-db-to-local.py branches on r["action"] and the
        # suites assert on it, so the dict is the contract), which meant
        # entrypoint.sh's `|| { WARNING }` branch could not fire for three
        # separate reasons at once. A restore that failed outright announced
        # success: OpenWebUI started onto a database that did not exist,
        # alembic built a fresh empty schema, and one message later the sync
        # daemon tried to publish that over her only durable copy.
        #
        # An action MISSING from the table exits non-zero, never zero. This
        # project's most expensive recurring defect is a rule applied at one
        # site and missed at its sibling, and a .get(..., 0) here is that
        # defect pre-installed: a future branch returning a new action string
        # would get a silent pass on the one path that loses everything.
        sys.exit(RESTORE_EXIT_CODES.get(r.get("action"), 6))
    elif "--sync-once" in sys.argv:
        r = sync_once(force="--force" in sys.argv)
        print(r)
        # The sibling, fixed at the same time and for the same reason. A
        # refused publish is the guard working, and it still must not report
        # success to a shell: anything that wraps this in `&&` - a hot-patch
        # script, a cron, an operator checking before a redeploy - would
        # otherwise read "refused to publish, her history is exposed" as
        # "done". A skip (unchanged, or no local database yet) IS success:
        # nothing needed doing.
        sys.exit(1 if r["error"] else 0)
    elif "--status" in sys.argv:
        # --status stays exit 0 even when it reports an unhealthy database,
        # deliberately: it is a report, not a gate, and the WARNING text in
        # entrypoint.sh tells an operator to run it while investigating. A
        # report that exits non-zero when it successfully reported is its own
        # trap. The gates are --restore and --sync-once above.
        for label, p in (("local", LOCAL_DB), ("snapshot", SNAPSHOT_DB)):
            ok, detail = integrity(p)
            size = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "-"
            # Stored content beside the file size, because they are not the
            # same number and the difference between them is what the shrink
            # guard now compares: a VACUUMed 4 MB file and a bloated 41 MB one
            # can hold identical history, and this is where an operator sees
            # that before believing a size.
            content = _content_bytes(p) if p.exists() else None
            content_s = "-" if content is None else f"{content / 1e6:.1f} MB"
            print(
                f"{label:9} {str(p):34} {size:>10}  quick_check={detail}  "
                f"chats={_has_rows(p)}  content={content_s}"
            )
        # The refusal an operator is most likely to be staring at when they
        # run this, and the only one whose cause is a file rather than a
        # measurement. Printed unconditionally, including when it is absent,
        # so "there is no marker" is a fact reported rather than an absence
        # inferred from silence.
        print(
            f"empty-start marker {str(EMPTY_START_MARKER):34} "
            f"{'PRESENT - sync_once is refusing; see the banner in entrypoint.sh' if EMPTY_START_MARKER.exists() else 'absent'}"
        )
    else:
        sync_loop()
