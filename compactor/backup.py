"""
compactor.backup — V2.3 Theme 1: data durability.

The `/data` volume holds two things that cannot be regenerated if lost:
OpenWebUI's `webui.db` (chat history) and `compactor/` (facts JSON, summary
state, persona text, ChromaDB vectors). Models and the torch.compile cache
are re-downloadable; these are not. This module makes timestamped,
**verified** snapshots so a corrupted file, an accidental `/forget`, or a
bad delete is recoverable.

Design principles (this is the V2.3 "failure-tested before done" release —
the safety net itself must be trustworthy):

1. **A backup that can't be verified is not a backup.** After creating an
   archive we immediately restore it to a scratch dir and assert the
   SQLite db passes `PRAGMA integrity_check` and every memory JSON parses.
   If verification fails, the archive is deleted and the run reports
   FAILURE — false confidence is worse than a known gap.
2. **Live-SQLite-safe.** `webui.db` is being written by OpenWebUI while we
   back up. A raw file copy can capture a torn page. We use SQLite's online
   backup API (`Connection.backup()`) to get a consistent snapshot.
3. **Can't fill the disk.** A min-free-space guard refuses to start a
   backup that would risk filling `/data` (a full disk is itself a failure
   mode we're trying to prevent).
4. **Atomic publish.** The archive is written to a `.partial` temp name and
   `os.replace`d into place only after it verifies — readers/pruners never
   see a half-written archive.
5. **An archive that holds nothing is a failure, not a backup.** (v3.1 F2.)
   Principle 1 was implemented as "verify what the manifest says is here",
   with the manifest written by the same run — so an archive containing
   nothing but `manifest.json` verified green, published, and pruned the
   real archives behind it. Verification now asserts *against* the
   manifest, `create_backup` raises rather than recording an absent store,
   and a payload that collapses relative to the previous archive is
   refused.
6. **Pruning is the dangerous half of this module.** (v3.1 F7 / D9.)
   Retention is by age plus a grandfather-father-son tier with a hard
   floor, and a cycle that is not fully clean does not prune at all. The
   old scheme — "keep the newest RETAIN", pruned unconditionally at the end
   of every cycle, with a cycle fired at process start — meant RETAIN
   container restarts erased every pre-incident archive.

Scope (V2.3 Theme 1, phase 1): **local backups** to a directory on the same
volume. This protects against the common, recoverable failures (corruption,
accidental delete, torn write). It does **not** survive total volume loss —
off-volume disaster recovery (object store) is required future work and
will need a migration. The `upload_hook` below is the designed-in seam.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from envcfg import env_float, env_int

logger = logging.getLogger("compactor.backup")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# What to back up. DATA_DIR is OpenWebUI's state root; webui.db lives there.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/openwebui"))


def live_webui_db() -> Path:
    """The webui.db OpenWebUI is actually reading, resolved at CALL time.

    v3.1.9 (M10 / A3-4). This was a module constant:

        WEBUI_DB = COMPACTOR_BACKUP_WEBUI_DB
                   or (_LIVE_DB if _LIVE_DB.exists() else DATA_DIR/webui.db)

    — a guess by EXISTENCE, while the deployment decides by a GATE.
    entrypoint.sh calls WEBUI_DB_LOCAL "the kill switch for the one subsystem
    here that owns where her chat history physically lives", and its
    documented rollback (`false`) points OpenWebUI at the snapshot while
    leaving /var/lib/openwebui/webui.db exactly where it was. On any restart
    that keeps the overlay the stale local file still exists, so every nightly
    archive became a copy of a database nobody had written to since the flag
    flipped, and every CLI restore landed on that abandoned file where
    OpenWebUI would never read it. The string WEBUI_DB_LOCAL did not appear
    anywhere in this file.

    It was also frozen at IMPORT, in the two longest-lived processes on the
    pod (the backup daemon and main.py), so a gate flipped under a running
    daemon was invisible to it until restart.

    Resolution order is the order of authority:

      1. COMPACTOR_BACKUP_WEBUI_DB — an explicit operator override, unchanged.
      2. DATABASE_URL, when it is a sqlite URL — that IS what OpenWebUI reads.
         entrypoint.sh derives it from the gate with `:-`, so an operator's
         own value wins there too, and this reads the result rather than
         re-deriving it. Present for the daemon (supervisord inherits
         entrypoint's exports); ABSENT for `docker exec ... backup.py`, which
         gets only the image and pod environment, so:
      3. the gate — v3.1.9 round 2: THIS is where this function used to
         re-derive WEBUI_DB_LOCAL a SECOND time
         (`os.environ.get("WEBUI_DB_LOCAL", "true") == "true"`), independently
         of webuidb.live_webui_db(), which entrypoint.sh's own boot refusal
         (finding 3, fix-webuidb.md) now normalises this same variable for —
         fold case/whitespace, accept 1/yes/on and 0/no/off, refuse to boot on
         anything else. A narrow `== "true"` compare agreed with entrypoint.sh
         for every SUPERVISED child (which only ever inherits the already-
         normalised value), but `docker exec ... backup.py` — the one case
         DATABASE_URL above is NOT inherited either, named in the case above
         as the reason this level exists — runs OUTSIDE that normalisation
         entirely, in whatever the operator's own shell has. There, this used
         to disagree with webuidb.live_webui_db() (the one other place this
         same rule is read) on `WEBUI_DB_LOCAL=1` or any other folded
         spelling: two independent readers of one rule is exactly the
         reader-disagrees-with-writer defect this function exists to close,
         reintroduced one level down from the fix. Delegates to
         webuidb.live_webui_db() now, so there is exactly one place that
         answers "which file is live" from the gate. Imported LAZILY, not at
         module scope — same reasoning _quarantine_dir gives for its own lazy
         `import webuidb`: backup.py is loaded by the CLI and the supervisord
         sidecar and must not drag in webuidb's own config surface just to
         learn one path.
    """
    explicit = os.environ.get("COMPACTOR_BACKUP_WEBUI_DB")
    if explicit:
        return Path(explicit)
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("sqlite:///"):
        # sqlite:////var/lib/x.db -> /var/lib/x.db; sqlite:///rel.db -> rel.db
        return Path(url[len("sqlite:///"):])
    import webuidb
    return webuidb.live_webui_db()


# The same three suffixes webuidb.SIDECARS sweeps, for the same reason:
# SQLite derives them from the database path it was given, so they belong
# to whatever file carries that name. Duplicated rather than imported —
# backup.py is loaded by the CLI and the daemon and must not drag in the
# sync module to learn three strings.
SIDECARS = ("-journal", "-wal", "-shm")

STORAGE_ROOT = Path(
    os.environ.get("COMPACTOR_STORAGE_ROOT", str(DATA_DIR / "compactor"))
)

# Where backups land. Default is a sibling dir on the same volume.
BACKUP_DIR = Path(os.environ.get("COMPACTOR_BACKUP_DIR", "/data/backups"))

# D2 (findings.md, the 2026-09-23 WEBUI_DB_LOCAL rehearsal). Local-disk
# scratch for _snapshot_sqlite_to_data: the destination `Connection.backup()`
# itself writes to MUST be on the same fast disk as the live database, never
# on /data — see that function's docstring for the lock this avoids holding
# open across a slow or stalling volume. A sibling of the live LOCAL
# database's own directory under the shipped default, so it is local disk
# on both WEBUI_DB_LOCAL placements (this is about where backup.py's own
# scratch copy lands, not about where the live database lives).
LOCAL_STAGING_DIR = Path(
    os.environ.get(
        "COMPACTOR_BACKUP_LOCAL_STAGING_DIR", "/var/lib/openwebui/.backup-staging"
    )
)

# How many archives to keep. No longer a cap — since v3.1 F7 this is a
# *floor* on the number retained, one of several tiers in _keep_set. As a
# cap it was the mechanism of the loss: RETAIN=7 with a prune at the end of
# every cycle and a cycle at every process start meant seven container
# restarts left seven copies of the damaged state and nothing older.
RETAIN = env_int("COMPACTOR_BACKUP_RETAIN", 7)

# Retention tiers (v3.1 F7 / D9). Keep everything younger than RETAIN_DAYS,
# plus one archive per UTC day inside that window, plus one per ISO week for
# GFS_WEEKS. Anything no tier claims is prunable.
RETAIN_DAYS = env_float("COMPACTOR_BACKUP_RETAIN_DAYS", 14)
GFS_WEEKS = env_int("COMPACTOR_BACKUP_GFS_WEEKS", 8)

# The hard floor. Never leave fewer than this many archives on disk, whatever
# their age and whatever the tiers say. Floored at 3 in code rather than in
# config: a typo in an env var must not be able to empty the backup
# directory. This is the last line between a bad cycle and total loss.
MIN_KEEP = max(3, env_int("COMPACTOR_BACKUP_MIN_KEEP", 3))

# Refuse to publish an archive whose payload is under this fraction of the
# previous one's. A store that lost half its bytes between two cycles is an
# unmounted volume, not a user deleting things. (v3.1 F2.)
MIN_PAYLOAD_RATIO = env_float("COMPACTOR_BACKUP_MIN_PAYLOAD_RATIO", 0.5)

# Daemon cadence.
INTERVAL_HOURS = env_float("COMPACTOR_BACKUP_INTERVAL_HOURS", 24)

# Refuse to back up if the target volume has less than this much free space.
# Prevents the backup process from being the thing that fills the disk.
MIN_FREE_MB = env_int("COMPACTOR_BACKUP_MIN_FREE_MB", 500)

# v3.1.9 (hostile317-c F3). How long run_daemon waits before retrying a
# FAILED cycle, instead of sleeping the full INTERVAL_HOURS. A hot MooseFS
# rollback journal or a "database is locked" episode used to cost a full 24h
# retry — exactly the state that precedes needing a backup most. Capped at
# the configured interval so a short test/dev interval is never lengthened
# by this constant.
RETRY_BACKOFF_S = env_float("COMPACTOR_BACKUP_RETRY_BACKOFF_S", 900)

# Optional off-volume target (future work). When unset, local only.
REMOTE_TARGET = os.environ.get("COMPACTOR_BACKUP_REMOTE", "").strip()

_ARCHIVE_PREFIX = "zions-backup-"
_ARCHIVE_SUFFIX = ".tar.gz"
_MANIFEST_NAME = "manifest.json"

# Manifest schema. v1 recorded `{"present": bool}` and a total file count.
# v2 adds json_files, chroma_sqlite, payload_bytes and the per-conversation
# census — everything verify_backup needs to contradict the archive it is
# looking at. Archives written before v3.1 are v1; the verifier degrades to
# the checks a v1 manifest can support rather than refusing to read them,
# because refusing to verify is refusing to restore.
_SCHEMA = "v2"

# Storage layout, duplicated from memory.py on purpose: this module is a
# standalone CLI and a supervisord sidecar, and importing the compactor
# package would drag in httpx, chromadb and the whole config surface just to
# learn three directory names.
_FACTS_SUBDIR = "facts"
_SUMMARIES_SUBDIR = "summaries"
_PERSONAS_SUBDIR = "personas"
_CHROMA_SUBDIR = "chromadb"
_CHROMA_DB_NAME = "chroma.sqlite3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _free_mb(path: Path) -> float:
    """Free space (MB) on the filesystem holding `path` (or its nearest
    existing ancestor, since the dir may not exist yet)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free / (1024 * 1024)
    except Exception as e:
        # Fail OPEN — a bad reading must not stop backups. Behaviour is
        # deliberate and unchanged; the log is not. Silently returning inf
        # left the min-free guard disabled forever with nothing said, so a
        # permanently broken disk_usage looked exactly like a roomy disk.
        # Once per process — this runs every cycle. (v3.1 P0-2b / F61.)
        import logsetup
        if logsetup.log_once("backup._free_mb"):
            logger.warning(
                f"could not read free space at {p} ({type(e).__name__}: {e}); "
                f"the min-free backup guard is disabled for this process"
            )
        return float("inf")  # can't tell → don't block


def sweep_stale_local_staging(max_age_s: float = 3600.0) -> list[str]:
    """M-1 (review1-v3197-65ea196). An orphaned `.backup-staging/webuidb-
    local-*.sqlite3` file — left behind when a cycle is SIGKILLed between
    `tempfile.mkstemp` and the `finally: local_tmp.unlink(missing_ok=True)`
    in `_snapshot_sqlite_to_data` — sits on local disk forever (mutation
    B3, "local staging file never unlinked", SURVIVED: nothing in the
    gate's own tests notices). Every orphan left there burns exactly the
    local disk space this module's own free-space check above exists to
    protect, silently shrinking the margin every later backup or sync
    cycle actually gets.

    Called once at process start (`main()`, before any argument is acted
    on) rather than every cycle: a cycle in progress right now legitimately
    has a same-shaped file open, so sweeping here rather than mid-run means
    this can only ever remove a file from a run that is not this process's
    own and is old enough (`max_age_s`, default 1h — no real backup stage
    takes anywhere near that) that nothing still using it is plausible.
    """
    removed: list[str] = []
    if not LOCAL_STAGING_DIR.is_dir():
        return removed
    now = time.time()
    for f in LOCAL_STAGING_DIR.glob("webuidb-local-*.sqlite3"):
        try:
            if now - f.stat().st_mtime > max_age_s:
                f.unlink()
                removed.append(str(f))
        except OSError:
            pass
    if removed:
        logger.warning(
            f"swept {len(removed)} orphaned local backup-staging file(s) "
            f"from a previous interrupted cycle: {removed}"
        )
    return removed


def _snapshot_sqlite(src: Path, dest: Path) -> bool:
    """Consistent online snapshot of a (possibly live) SQLite db via the
    backup API. Returns True if a snapshot was written, False if the source
    doesn't exist. Raises on a real failure.

    v3.1.9 (hostile317-c F3). A HOT ROLLBACK JOURNAL beside `src` — the
    documented MooseFS failure under WEBUI_DB_LOCAL=false, a stall that
    lands mid-transaction — makes the mode=ro open below raise
    "attempt to write a readonly database": finishing a rollback requires
    writing, and a read-only handle cannot. Before this fix that exception
    propagated out of create_backup, run_once reported ok=False, and
    run_daemon slept the FULL interval (24h default, see run_daemon) before
    trying again — so the state that precedes needing a backup most is the
    one state in which backups stopped.

    recover-webui-db.py's rule for this exact failure is the fix: a journal
    and its database are a matched pair, so copy BOTH aside and let SQLite
    finish the rollback on the disposable copy, never on the live pair. Only
    the specific readonly-database signature falls back to that path;
    anything else still raises as before, unchanged.
    """
    if not src.is_file():
        return False
    # Open read-only-ish; the backup API handles WAL + concurrent writers.
    try:
        con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(str(dest))
            try:
                con.backup(dst)
            finally:
                dst.close()
        finally:
            con.close()
        return True
    except sqlite3.OperationalError as e:
        if "readonly database" not in str(e):
            raise
        logger.warning(
            f"{src} could not be backed up read-only ({e}) — this is the hot "
            f"rollback journal signature (recover-webui-db.py), so copying "
            f"it and its sidecars aside to finish the rollback on a "
            f"disposable copy rather than touching the live pair"
        )
        return _snapshot_via_rollback_copy(src, dest)


def _snapshot_via_rollback_copy(src: Path, dest: Path) -> bool:
    """Fallback for _snapshot_sqlite when `src` has a hot rollback journal.

    Copies `src` and its SIDECARS to a scratch dir — the matched pair, never
    the live files — opens the COPY read-write so SQLite performs the same
    rollback it would perform on any ordinary open (just against disposable
    bytes this time), verifies it, then takes the online backup from that
    recovered copy. The live database and journal are only ever read here,
    never opened for writing, renamed, or deleted.
    """
    scratch = Path(tempfile.mkdtemp(prefix="zions-hotjournal-"))
    try:
        scratch_src = scratch / src.name
        shutil.copy2(src, scratch_src)
        for suffix in SIDECARS:
            side = src.with_name(src.name + suffix)
            if side.is_file():
                shutil.copy2(side, scratch / side.name)
        con = sqlite3.connect(str(scratch_src))
        try:
            row = con.execute("PRAGMA integrity_check").fetchone()
            if not row or row[0] != "ok":
                raise RuntimeError(
                    f"recovered copy of {src} failed integrity_check: {row}"
                )
            dst = sqlite3.connect(str(dest))
            try:
                con.backup(dst)
            finally:
                dst.close()
        finally:
            con.close()
        return True
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _snapshot_sqlite_to_data(
    src: Path, dest: Path, *, local_staging_dir: Path | None = None
) -> bool:
    """Like _snapshot_sqlite, but keeps /data out of the backup API's own
    lock-holding write.

    D2 (findings.md, the 2026-09-23 WEBUI_DB_LOCAL rehearsal). The old call
    was `_snapshot_sqlite(src, dest)` with `dest` already inside the /data
    staging directory create_backup builds each cycle. sqlite3's
    `Connection.backup()` defaults to `pages=-1` — one
    `sqlite3_backup_step(-1)` that holds a read lock on `src` (the LIVE
    database) until the LAST destination page is written. With `dest` on
    MooseFS that lock was held for as long as /data's writes took, and
    OpenWebUI's own busy timeout (10s) turned a slow cycle into "database is
    locked" plus multi-second commits on EVERY backup — measured (both
    WEBUI_DB_LOCAL placements): ~1 lock error plus commits from 5.9 to
    10.8s. It happens on every scheduled backup, every `--once`, and at
    boot.

    Mirrors webuidb.sync_once's own two-stage split (see that function's
    docstring for the same reasoning in more depth): the backup API is only
    ever pointed at LOCAL disk, where a live lock resolves in about a
    second regardless of /data's mood; only a lock-free plain file copy —
    no sqlite3 connection open on either end — crosses onto /data, so a
    stall there costs time, never the live database's lock.

    Falls back to the OLD direct-to-/data behaviour when local disk does
    not have room for a second copy of `src` (checked FIRST, deliberately,
    so a full local disk fails exactly as safely — and as slowly — as
    backups always have, rather than raising past this guard entirely). A
    slow backup is still a backup.
    """
    local_dir = local_staging_dir or LOCAL_STAGING_DIR
    # M-1 (review1-v3197-65ea196, X7b). This used to reserve only ~1.1x the
    # source's size - room for THIS function's own local_tmp copy, nothing
    # else. But webuidb-sync's own sync_once() stages its OWN local copy of
    # the SAME live database on the SAME local disk, on its own independent
    # schedule (every WEBUI_DB_SYNC_INTERVAL_S), and the two are not
    # coordinated: measured with 819 MB free (this check's old ~538 MB
    # threshold passed easily), the backup's local stage and webuidb-sync's
    # local temp overlapped and free space fell to 257 MB — live DB (already
    # on disk) + this function's local_tmp + webuidb-sync's own local_tmp,
    # all at once. ~0.8 GB less disk under the same timing would ENOSPC the
    # LIVE database's own journal, not merely fail this backup. Reserving
    # 2x (this copy AND a possible concurrent sync copy) plus a fixed margin
    # is the honest number for what can actually be on this disk at once.
    _CONCURRENT_COPIES = 2
    _MARGIN_MB = 32
    needed_mb = 0.0
    if src.is_file():
        needed_mb = (
            (src.stat().st_size / (1024 * 1024)) * 1.1 * _CONCURRENT_COPIES
            + _MARGIN_MB
        )
    local_free = _free_mb(local_dir) if needed_mb else float("inf")
    if needed_mb and local_free < needed_mb:
        # ERROR, not WARNING (M-1): this fallback silently brings BACK the
        # exact D2 symptom it exists to avoid — a lock error plus a
        # multi-second commit stall on the live database, on every backup
        # cycle for as long as local disk stays this full. A WARNING here
        # read like routine noise in backup.log while the live writer took
        # the hit; an operator needs this to be as loud as the problem it
        # causes.
        logger.error(
            f"only {local_free:.0f} MB free at {local_dir} (need ~"
            f"{needed_mb:.0f} MB — room for this backup's own local copy "
            f"AND a possible concurrent webuidb-sync cycle — to stage "
            f"{src} locally first) — falling back to backing it up "
            f"straight onto {dest.parent}, which holds the live "
            f"database's read lock for as long as that volume takes (D2, "
            f"findings.md). Free up local disk or raise "
            f"COMPACTOR_BACKUP_LOCAL_STAGING_DIR's volume."
        )
        return _snapshot_sqlite(src, dest)

    try:
        local_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(
            f"could not create local staging dir {local_dir} "
            f"({type(e).__name__}: {e}) — falling back to backing up "
            f"{src} straight onto {dest.parent} (D2, findings.md)"
        )
        return _snapshot_sqlite(src, dest)

    fd, tmp_name = tempfile.mkstemp(
        prefix="webuidb-local-", suffix=".sqlite3", dir=str(local_dir)
    )
    os.close(fd)
    local_tmp = Path(tmp_name)
    try:
        local_tmp.unlink()  # sqlite3.connect creates it fresh; an empty
        # file left by mkstemp is a harmless but pointless extra open.
    except OSError:
        pass

    # M-1 (review1-v3197-65ea196, X7b): serialise THIS function's local
    # stage with webuidb-sync's own — both stage a same-sized local copy of
    # the same live database on the same local disk on independent
    # schedules, and the two landing at once is exactly what turned 819 MB
    # "free" into 257 MB actually free (see the free-space check above).
    # Reusing webuidb's OWN sync flock, rather than inventing a second lock
    # file that could itself disagree with the first, asks "is a sync
    # cycle using local disk right now" the one place that already knows.
    # Non-blocking with a short bounded retry, never an indefinite wait —
    # this lock DEGRADES SAFELY (see _acquire_sync_lock's own docstring),
    # and a backup must still complete even on a platform or a filesystem
    # where flock is not available.
    import webuidb
    _sync_lock = None
    for _attempt in range(5):
        _sync_lock = webuidb._acquire_sync_lock()
        if _sync_lock is not None:
            break
        time.sleep(1.0)
    if _sync_lock is None:
        logger.warning(
            "webuidb-sync's own sync was still running after 5s; staging "
            "this backup locally anyway rather than waiting indefinitely "
            "for it (both briefly want the same local disk headroom — see "
            "the 2x free-space reservation above)"
        )
    try:
        wrote = _snapshot_sqlite(src, local_tmp)
        if not wrote:
            return False
        # STAGE 2: a plain byte copy, no sqlite3 connection open on either
        # side — the live lock (held only during the stage above) is
        # already released by the time this line runs, no matter how long
        # /data takes to accept the write.
        shutil.copy2(local_tmp, dest)
        return True
    finally:
        if _sync_lock is not None and _sync_lock is not webuidb._LOCK_DEGRADED:
            webuidb._release_sync_lock(_sync_lock)
        try:
            local_tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _tree_bytes(p: Path) -> int:
    """Total bytes of every regular file under `p` (or of `p` itself).

    p3-b F11: this used to be defined TWICE — a second, naive
    `sum(f.stat().st_size for f in root.rglob("*") if f.is_file())` later
    in the file silently replaced this one for every caller after that
    point (create_backup's own payload measure included), dropping the
    per-file OSError tolerance and the "backup sizing" warning below with
    no error of any kind — Python just keeps the last `def`. The second
    one is deleted; this is the only `_tree_bytes` in the module now.
    """
    # Undercounting here makes the payload-collapse guard *more* likely to
    # refuse a publish, so it fails safe — but it is still a plausible-looking
    # default standing in for a failure, which is the shape P0-2b exists to
    # remove. Log it so a systematically undersized archive is traceable to
    # unreadable files rather than to a shrinking store.
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError as e:
            logger.warning(f"backup sizing: could not stat {p}: {e}; counted as 0")
            return 0
    total = 0
    if p.is_dir():
        unreadable = 0
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    unreadable += 1
        if unreadable:
            logger.warning(
                f"backup sizing: {unreadable} file(s) under {p} could not be "
                f"stat'd and are counted as 0 bytes — the payload figure is a "
                f"lower bound"
            )
    return total


def _read_json(path: Path):
    """Parse a memory file, or None if it will not parse. The census must not
    be the thing that kills a backup — verify_backup fails on an unparseable
    memory file a moment later, with a message that names it."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"census: could not read {path.name}: {type(e).__name__}: {e}")
        return None


def _episodic_counts(db: Path) -> dict[str, int] | None:
    """conv_id → indexed-exchange count, read straight out of ChromaDB's own
    SQLite tables (`embedding_metadata`, the same `conv_id` metadata key
    retrieval.py writes at :172).

    Returns None meaning **unknown**, never zero, when the file is absent or
    the schema is not the one we know — a ChromaDB upgrade must degrade the
    census, not fabricate a total episodic loss and block every prune.
    """
    if not db.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT string_value, COUNT(*) FROM embedding_metadata "
                "WHERE key = 'conv_id' AND string_value IS NOT NULL "
                "GROUP BY string_value"
            ).fetchall()
        finally:
            con.close()
    except Exception as e:
        import logsetup
        if logsetup.log_once("backup._episodic_counts"):
            logger.warning(
                f"could not read episodic counts from {db} "
                f"({type(e).__name__}: {e}); manifests will carry no episodic "
                f"numbers, so an episodic-only loss will not be detected"
            )
        return None
    return {str(r[0]): int(r[1]) for r in rows}


def _text_len(obj) -> int:
    """Byte-ish length of a dict's `text` field, 0 for anything else. Used
    to build the byte-volume census fields below — deliberately len(), not
    a real UTF-8 byte count: what matters is a stable, cheap magnitude to
    compare cycle to cycle, not an exact size."""
    if isinstance(obj, dict):
        t = obj.get("text")
        if isinstance(t, str):
            return len(t)
    return 0


def _census(store: Path) -> dict:
    """Per-conversation fact / summary / persona / episodic counts.

    Computed from the *staged or extracted* tree, never from the live store,
    so the manifest describes what is actually inside the archive and
    verify_backup can recompute the identical numbers and contradict it.

    Per-conversation rather than a total: a total hides one conversation
    emptying while another grows, and one conversation is the whole product
    here. (v3.1 F2.)

    p3-b F3: `facts` and `summaries` are OLD FIELDS with their ORIGINAL
    meaning, unchanged since before v3.1.9 — active facts only, and
    len(l1)+len(l2)+bool(l3) chunks. v3.1.6.1's and v3.1.8's OWN
    verify_backup / restore_backup read these two fields by name out of any
    archive's manifest, including one this (newer) code wrote, whenever a
    pod is rolled back. hostile317-c F1's fix (below) folded archived facts
    INTO `facts`, which fixed run_once's own false alarms but silently
    changed what the field MEANS: an old reader recomputes `facts` as
    active-only from the same archive, sees a "truncated" archive on every
    conversation with any archived fact, and REFUSES it (p3-b F3, proven
    against the real v3.1.6.1-cu12/v3.1.8-cu12 images in SP\\p3-b\\xver.py).
    An archive's on-disk manifest shape has to keep meaning what an older
    binary already assumes it means — there is no version negotiation here,
    only "does the field still say what it always said". So old-shaped
    fields are never repurposed again; new information always gets a NEW
    key that an old reader simply does not look at:

      * facts     — ACTIVE facts only, exactly as v3.1.6.1/v3.1.8 compute
        it. Eviction (prune_facts with conv_id) moves entries out of this
        count into archived_facts below; that is not loss, so cross-cycle
        comparisons must read facts+archived_facts together, never this
        field alone (see _census_regressions).
      * summaries — len(l1)+len(l2)+bool(l3), exactly as v3.1.6.1/v3.1.8
        compute it. Superseded for loss-detection by summary_turn /
        summary_active_bytes below (a rollup legitimately collapses this
        count), kept only so an old reader's own shortfall check still
        finds the key it expects.
      * archived_facts — the facts.archive.json sidecar alone (p3-b F3).
        archive_facts only ADDS to it (re-archiving a fact replaces its own
        entry, never removes another's), so on its own it should not roughly
        halve between two cycles either — see _census_regressions.
      * summary_turn — `last_summarized_turn`, the highest turn any L1 chunk
        covers. Every rollup writer advances this to a chunk's own last_turn
        and none of them retreats it (summarizer.py); it is a high-water
        mark, not a count, so folding ten chunks into one chapter does not
        move it.
      * archived_chapters — the L2 chapter cold-storage sidecar
        (`summaries/<id>.archive.json`, written by _archive_chapters once L3
        has paraphrased a span). Append-only: this is the ONLY remaining
        copy of chapter-level detail once L3 absorbs it (summarizer.py:250),
        so a decrease here is real loss, not a rollup doing its job.
      * facts_bytes — sum of len(text) over active + archived facts
        (p3-b F2). A count can survive while every fact's TEXT is gutted
        (hollow entries, same length list); this is the signal that catches
        that shape, since dedup/eviction/rollup never blank a survivor's
        text, only remove or relocate whole entries.
      * summary_active_bytes — sum of len(text) over l1 + l2 + l3, ACTIVE
        state only (p3-b F2). Unlike summary_turn/archived_chapters this can
        legitimately go to zero mid-hierarchy (e.g. right after an L1->L2
        rollup with no L2->L3 yet, l1 is [] but l2 is not) — see
        _census_regressions for the narrow shape that still counts as loss.
      * archived_chapter_bytes — sum of len(text) over the archived
        chapters sidecar (p3-b F2). Append-only like archived_chapters
        itself; a legitimate L2->L3 rollup archives the FULL chapter text
        here BEFORE l3 is replaced with its paraphrase
        (summarizer._do_l3_rollup), so this total only grows or holds.
      * persona — True if a non-empty persona_text is on disk for this conv
        (p3-b F2). Not a count: there is at most one persona per
        conversation, and "present, now absent" is the whole signal.
    """
    census: dict[str, dict] = {}

    def slot(conv_id: str) -> dict:
        return census.setdefault(
            conv_id,
            {
                "facts": 0,
                "summaries": 0,
                "episodic": 0,
                "archived_facts": 0,
                "summary_turn": 0,
                "archived_chapters": 0,
                "facts_bytes": 0,
                "summary_active_bytes": 0,
                "archived_chapter_bytes": 0,
                "persona": False,
                "persona_bytes": 0,
            },
        )

    facts_dir = store / _FACTS_SUBDIR
    if facts_dir.is_dir():
        for f in sorted(facts_dir.glob("*.json")):
            # `<id>.archive.json` and other sidecars have a dot in the stem —
            # same rule memory.list_known_conv_ids uses.
            if "." in f.stem:
                continue
            data = _read_json(f)
            if isinstance(data, dict) and isinstance(data.get("facts"), list):
                entries = data["facts"]
                s = slot(f.stem)
                s["facts"] += len(entries)
                s["facts_bytes"] += sum(_text_len(e) for e in entries)
        # Archived facts get their OWN field (p3-b F3) — see _census's
        # docstring. Eviction (prune_facts with conv_id) moves entries here;
        # without reading this sidecar at all the census would watch only
        # the half of the store the daemon's own steady state empties every
        # day.
        for f in sorted(facts_dir.glob("*.archive.json")):
            conv_id = f.name[: -len(".archive.json")]
            data = _read_json(f)
            if isinstance(data, dict) and isinstance(data.get("facts"), list):
                entries = data["facts"]
                s = slot(conv_id)
                s["archived_facts"] += len(entries)
                s["facts_bytes"] += sum(_text_len(e) for e in entries)

    summaries_dir = store / _SUMMARIES_SUBDIR
    if summaries_dir.is_dir():
        for f in sorted(summaries_dir.glob("*.json")):
            if "." in f.stem:
                continue
            data = _read_json(f)
            if not isinstance(data, dict):
                continue
            s = slot(f.stem)
            turn = data.get("last_summarized_turn")
            if isinstance(turn, int):
                s["summary_turn"] = turn
            n = 0
            active_bytes = 0
            for tier in ("l1", "l2"):
                chunks = data.get(tier)
                if isinstance(chunks, list):
                    n += len(chunks)
                    active_bytes += sum(_text_len(c) for c in chunks)
            l3 = data.get("l3")
            if isinstance(l3, dict):
                n += 1
                active_bytes += _text_len(l3)
            s["summaries"] = n
            s["summary_active_bytes"] = active_bytes
        for f in sorted(summaries_dir.glob("*.archive.json")):
            conv_id = f.name[: -len(".archive.json")]
            data = _read_json(f)
            if isinstance(data, dict) and isinstance(data.get("chapters"), list):
                chapters = data["chapters"]
                s = slot(conv_id)
                s["archived_chapters"] = len(chapters)
                s["archived_chapter_bytes"] = sum(_text_len(c) for c in chapters)

    personas_dir = store / _PERSONAS_SUBDIR
    if personas_dir.is_dir():
        for f in sorted(personas_dir.glob("*.json")):
            if "." in f.stem:
                continue
            data = _read_json(f)
            if isinstance(data, dict):
                text = data.get("persona_text")
                if isinstance(text, str):
                    # p4-b G5. Recorded even when text is "" (or whitespace,
                    # which the presence flag below treats as absent) — the
                    # hwm floor on this field needs the RAW byte count, the
                    # same way facts_bytes reads the text a count-preserving
                    # rewrite can gut without moving.
                    slot(f.stem)["persona_bytes"] = len(text)
                if isinstance(text, str) and text.strip():
                    slot(f.stem)["persona"] = True

    episodic = _episodic_counts(store / _CHROMA_SUBDIR / _CHROMA_DB_NAME)
    if episodic:
        for conv_id, n in episodic.items():
            slot(conv_id)["episodic"] = n

    return census


def _census_shortfalls(expected: dict, actual: dict) -> list[str]:
    """Conversations where `actual` holds fewer of a layer than `expected`.

    One-directional on purpose. More is fine. Less is the entire signal.

    This is the STRICT comparison — every layer, any decrease — and it has
    exactly one caller: verify_backup, contradicting an archive's own
    manifest with what verify_backup just recomputed from that SAME
    archive's extracted contents. Both readings are taken from one static
    tree a few seconds apart, so there is no eviction, no dedup and no
    rollup that can happen in between — any shortfall here is the archive
    not containing what it claims to, i.e. truncation or corruption, and
    NONE of the tolerances _census_regressions (below) applies across two
    different nightly cycles belong here. Do not point run_once at this
    function — that was v3.1.9 hostile317-c F1 (see _census_regressions).

    Every numeric field _census produces is checked, byte fields included
    (p3-b F2/F3) — within one archive there is no legitimate reason for a
    freshly recomputed byte total to be lower than what the manifest claims,
    so a stricter check here costs nothing and catches truncation that a
    count alone would not (e.g. tar extracting a fact list to the right
    length with blank text).
    """
    out: list[str] = []
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return out
    for conv_id in sorted(expected):
        want = expected.get(conv_id) or {}
        have = actual.get(conv_id) or {}
        if not isinstance(want, dict):
            continue
        have = have if isinstance(have, dict) else {}
        for layer in (
            "facts", "summaries", "episodic", "archived_facts",
            "summary_turn", "archived_chapters", "facts_bytes",
            "summary_active_bytes", "archived_chapter_bytes",
        ):
            w = int(want.get(layer) or 0)
            h = int(have.get(layer) or 0)
            if h < w:
                out.append(f"{conv_id}.{layer} {w}->{h}")
        # persona is a presence flag, not a count — same "less is the
        # signal" rule, just boolean.
        if want.get("persona") and not have.get("persona"):
            out.append(f"{conv_id}.persona present->absent")
    return out


# Below this a decrease is tolerated only up to this fraction of the
# previous cycle's value — p3-b F2. 0.5 is deliberately loose: dedup is
# documented (hostile317-c logs) at a few percent a night, and eviction /
# rollup move mass between fields rather than shrinking any one of them by
# half. A real partial loss in this project's proofs (SP\p3-b\census.py)
# clears this floor by a wide margin every time; a real rollup night does
# not approach it. See _census_regressions for which field each rule reads.
_CENSUS_LOSS_FLOOR = 0.5


def _census_regressions(expected: dict, actual: dict) -> list[str]:
    """Conversations where `actual` lost something `expected` had that the
    daemon's OWN normal operation, between two nightly cycles, does not
    explain.

    v3.1.9 (hostile317-c F1). This is run_once's cross-cycle comparison —
    the previous archive's census against the new one — and it is NOT the
    same question _census_shortfalls answers. Between two cycles, normal
    operation legitimately shrinks the raw numbers: facts.prune_facts(conv_
    id=...) moves evicted facts into the `.archive.json` sidecar rather than
    deleting them; dedup permanently merges duplicate facts into fewer,
    denser entries; and every L1->L2 or L2->L3 rollup replaces N chunks with
    one chapter. Her real logs (hostile317-c) show every nightly cycle from
    2026-08-31 through 2026-09-11 reading "memory shrank" and skipping the
    prune, on exactly this shape — because the daemon was using
    _census_shortfalls's STRICT rule for a question it does not answer here.

    p3-b F2: the original fix (v3.1.9) stopped that false-alarm storm by
    flagging facts only at exactly-zero, and left summaries/personas/text
    volume out of the census entirely. That is tolerant of normal operation
    but blind to any loss that stops short of total: SP\\p3-b\\census.py
    proves five partial losses (facts 140->3; the facts archive sidecar
    deleted; L1/L2/L3 emptied with the watermark kept; persona deleted;
    every fact TEXT gutted with the count untouched) all read `ok` with an
    empty `census_regressions` and prune normally. The rules below replace
    "went to exactly zero" with floors and a narrower total-wipe check, each
    picked to fire on one of those five shapes and NOT on the normal-
    operation shapes this function's own history says never to break again:

      * facts (active+archived UNION) < half of last cycle's union — the
        union is what eviction/archiving/dedup cannot shrink except by
        actually losing something (eviction moves mass across the two
        fields without changing the union; dedup merges a few percent a
        night). Catches active-file truncation even when the archive
        sidecar is untouched (SP\\p3-b\\census.py loss1, 140->3).
      * archived_facts alone < half of last cycle's — archive_facts only
        ADDS to this sidecar, so on its own it should not roughly-halve
        either. Catches the sidecar being deleted or truncated while the
        active file is left alone, which the union rule above cannot see
        when the active file is large relative to the archive (loss2).
      * facts_bytes < half of last cycle's — a count-preserving rewrite
        that blanks every fact's text passes every count-based rule above;
        this is the only signal that reads the text itself (loss5).
      * summary hierarchy wiped to nothing while the watermark still claims
        coverage: summary_active_bytes went from > 0 to exactly 0 while
        summary_turn (this cycle) is still > 0. A real rollup can empty l1
        alone (folded into l2) or l1+l2 together (folded into l3), but
        never all three at once while last_summarized_turn keeps claiming
        turns were covered — only /forget-style clearing (which also zeroes
        summary_turn, see below) or genuine loss does that (loss3, 500
        kept, l1/l2/l3 all emptied).
      * archived_chapter_bytes — joins summary_turn / archived_chapters in
        the monotonic group below (ANY decrease flags), not a 0.5 floor.
        _do_l3_rollup archives a chapter's FULL text into this sidecar
        BEFORE replacing the active state with a shorter paraphrase
        (summarizer.py), so a real rollup only grows this total — unlike
        summary_active_bytes above, it is never allowed to legitimately
        shrink, so the stricter any-decrease rule applies, exactly as it
        already did for the chapter COUNT. (A byte UNION of active +
        archived was tried and rejected: a real, table-stakes L1->L2 rollup
        can legitimately paraphrase l1 down to a shorter l2 chapter with
        nothing archived yet at that tier — see
        test_l1_to_l2_rollup_shape_does_not_regress_the_census — so any rule
        that reads summary_active_bytes and archived_chapter_bytes together
        WILL false-fire on ordinary rollup nights. The two stay separate.)
      * persona present -> absent — at most one per conversation, so this
        is a flag, not a count (loss4). A deliberate /forget or /retire
        clears it too and SHOULD alarm once — the daemon has no way, and no
        need, to tell those apart from a bug: either way the operator is
        the one who decides whether the older archives still matter.
      * summary_turn / archived_chapters / archived_chapter_bytes —
        monotonic (a high-water mark; append-only sidecars), so ANY
        decrease is a regression.
      * episodic — unchanged: nothing in the daemon's own cycle prunes
        ChromaDB rows, so any decrease is still worth naming.
    """
    out: list[str] = []
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return out
    for conv_id in sorted(expected):
        want = expected.get(conv_id) or {}
        have = actual.get(conv_id) or {}
        if not isinstance(want, dict):
            continue
        have = have if isinstance(have, dict) else {}

        w_active = int(want.get("facts") or 0)
        h_active = int(have.get("facts") or 0)
        w_arch = int(want.get("archived_facts") or 0)
        h_arch = int(have.get("archived_facts") or 0)

        w_union = w_active + w_arch
        h_union = h_active + h_arch
        if w_union > 0 and h_union < w_union * _CENSUS_LOSS_FLOOR:
            out.append(
                f"{conv_id}.facts {w_union}->{h_union} (active+archived, "
                f"more than half gone)"
            )
        if w_arch > 0 and h_arch < w_arch * _CENSUS_LOSS_FLOOR:
            out.append(f"{conv_id}.archived_facts {w_arch}->{h_arch}")

        w_fbytes = int(want.get("facts_bytes") or 0)
        h_fbytes = int(have.get("facts_bytes") or 0)
        if w_fbytes > 0 and h_fbytes < w_fbytes * _CENSUS_LOSS_FLOOR:
            out.append(f"{conv_id}.facts_bytes {w_fbytes}->{h_fbytes} (text gutted)")

        w_sactive = int(want.get("summary_active_bytes") or 0)
        h_sactive = int(have.get("summary_active_bytes") or 0)
        h_turn = int(have.get("summary_turn") or 0)
        if w_sactive > 0 and h_sactive == 0 and h_turn > 0:
            out.append(
                f"{conv_id}.summary_active_bytes {w_sactive}->0 (hierarchy "
                f"emptied, watermark at {h_turn})"
            )

        if want.get("persona") and not have.get("persona"):
            out.append(f"{conv_id}.persona present->absent")

        for layer in (
            "summary_turn", "archived_chapters", "archived_chapter_bytes",
            "episodic",
        ):
            w = int(want.get(layer) or 0)
            h = int(have.get(layer) or 0)
            if h < w:
                out.append(f"{conv_id}.{layer} {w}->{h}")
    return out


def _census_hwm_update(prev_hwm: dict, census: dict) -> tuple[dict, list[str]]:
    """(new_hwm, losses) — the per-conversation HIGH-WATER MARK check p4-b
    G5 adds alongside _census_regressions's immediate-previous-cycle floor.

    THE GAP THIS CLOSES. _census_regressions only ever compares THIS
    cycle's numbers to the ONE cycle before it, so a loss spread thin
    enough that every single step stays above the 0.5 floor never trips
    it — each night's (already-diminished) archive quietly becomes the
    next baseline. Proof (SP\\p4-b\\census4.py): a conversation's facts
    truncated 140 -> 71 -> 36 -> 19 -> 10 -> 5 over five cycles reads
    `census_regressions: []` on every single one (the worst single-cycle
    ratio is 71/140 = 0.507, just above the floor). A persona overwritten
    with "x" and 4 of 5 L1 summary chunks dropped, each in ONE cycle
    against a field the existing rules do not track byte-for-byte at all
    (persona) or only check for a TOTAL wipe (summary_active_bytes),
    passed the same way.

    THE FIX. A high-water mark per conversation per field, carried forward
    in each archive's own manifest (census_hwm, written by create_backup)
    — so the comparison is against the BEST state seen recently, not just
    the last one. `census` is THIS cycle's freshly computed _census()
    output; `prev_hwm` is the previous archive's own `census_hwm` (or {}
    with nothing yet to compare against).

    ALARM ONCE, so the SAME already-reported loss does not nag every
    night forever (the hostile317-c F1 failure mode, on a new field): a
    field that fires THIS cycle resets its own high-water mark to this
    cycle's (lower) value, so tomorrow's comparison is against today's
    reality, not the pre-loss peak. A field that does NOT fire keeps
    climbing: new_hwm = max(prev_hwm, this cycle). First-ever sighting of a
    conversation (no prior hwm entry, or a hwm of 0 — a real high-water
    mark can never be zero for a field that has anything in it) seeds the
    mark from THIS cycle with no alarm: there is nothing yet to have
    fallen from.

    FOUR FIELDS, deliberately not every field _census produces:
      * facts_union (facts + archived_facts) — the same union
        _census_regressions's own first rule already uses, so eviction
        moving mass between the two fields still does not fire this one.
      * facts_bytes — text gutted with the count preserved.
      * persona_bytes — len(persona_text). Not covered by ANY existing
        rule byte-for-byte: the existing `persona` field is a bare
        present/absent bool, so a persona overwritten with a one-character
        stub stays "present" forever (SP\\p4-b\\census4.py's own proof).
      * summary_active_bytes — the SAME field _census_regressions's total-
        wipe rule already reads, now ALSO checked as a gradual bleed
        rather than only an all-or-nothing wipe against the watermark.

    A CONVERSATION MISSING FROM `census` ENTIRELY (deleted, retired, or a
    genuine /forget — main._clear_all_memory removes its files outright)
    is walked too, via the union of both dicts' conv_ids below, exactly
    the way _census_regressions's own iteration (over `expected`, the
    OLDER side) already catches a conversation disappearing between two
    cycles — every field reads 0 this cycle, alarms once (a deliberate
    forget/retire SHOULD alarm once, same doctrine as the existing persona
    rule), and its hwm entry is then dropped rather than carried forward
    forever once every field is back at 0.
    """
    losses: list[str] = []
    new_hwm: dict = {}
    fields = (
        "facts_union", "facts_bytes", "persona_bytes", "summary_active_bytes",
    )
    for conv_id in sorted(set(prev_hwm) | set(census)):
        want = prev_hwm.get(conv_id) or {}
        if not isinstance(want, dict):
            want = {}
        have = census.get(conv_id) or {}
        current = {
            "facts_union": int(have.get("facts") or 0) + int(have.get("archived_facts") or 0),
            "facts_bytes": int(have.get("facts_bytes") or 0),
            "persona_bytes": int(have.get("persona_bytes") or 0),
            "summary_active_bytes": int(have.get("summary_active_bytes") or 0),
        }
        conv_hwm: dict = {}
        for field in fields:
            old_h = want.get(field)
            value = current[field]
            if not isinstance(old_h, (int, float)) or old_h <= 0:
                # No baseline yet (or the mark was already reset to 0 by a
                # full removal, below) — seed it, nothing to have fallen
                # from.
                conv_hwm[field] = value
                continue
            if value < old_h * _CENSUS_LOSS_FLOOR:
                losses.append(
                    f"{conv_id}.{field} {old_h}->{value} (more than half "
                    f"below its high-water mark)"
                )
                conv_hwm[field] = value  # reset: alarm once, not every night
            else:
                conv_hwm[field] = max(old_h, value)
        # A conv_id visited only because it lingered in prev_hwm, now fully
        # gone (every field back at 0), is not worth carrying forward —
        # the loss above already fired once; keeping a {0,0,0,0} entry
        # would only cost space for a conversation that no longer exists.
        if any(conv_hwm.values()):
            new_hwm[conv_id] = conv_hwm
    return new_hwm, losses


def _webui_db_scan_for_manifest(db_path: Path) -> dict | None:
    """Chat-content summary for the manifest, so a prune can be held on
    webui.db losing HER HISTORY, not just on the compactor store shrinking
    (p4-b G1).

    Reuses webuidb._scan_chat — the SAME function webuidb.py's own
    publish-time guards (chat-count ratio, the cast-as-blob content-byte
    ratio, MAX_ROW_LOSS_BYTES per surviving conversation) already measure
    with — rather than writing a third copy of "how much of her chat
    history is actually in this file". Those guards run on the LOCAL ->
    SNAPSHOT sync path, which is OFF in production (WEBUI_DB_LOCAL=false):
    a nightly backup cycle over the SNAPSHOT itself had no equivalent at
    all. Two production-reachable shapes read `ok` and pruned behind them
    before this fix (SP\\p4-b\\emptydb.py): the database replaced by a
    fresh, empty OpenWebUI-shaped schema after a kill mid-restore leaves
    the db MISSING (G3's window) and OpenWebUI then builds one; and her one
    big conversation row gutted to `{"messages": []}` in place, with no
    VACUUM, so the FILE SIZE does not move and Connection.backup() copies
    the same free pages either way — payload_ratio read 1.0.

    Called on the STAGED SNAPSHOT COPY (db_dest, right after
    _snapshot_sqlite succeeds) — the exact bytes going into this archive,
    same doctrine as _census being computed from the staged store rather
    than the live one.

    None when webui.db has no `chat` table or could not be read at all (a
    foreign schema, or the snapshot copy itself failing a beat after
    _snapshot_sqlite succeeded) — never confused with "zero chats", which
    is a real, comparable result. See _scan_chat's own docstring for the
    None-vs-zero distinction and why it costs a full table scan.
    """
    import webuidb
    scan = webuidb._scan_chat(db_path)
    if scan is None:
        return None
    rows = scan["rows"]
    return {
        "chats": len(rows) if rows is not None else None,
        "content_bytes": scan["total"],
        "newest_updated_at": scan["newest"],
        # Per-conversation bytes (keyed by chat id), so a LATER cycle can
        # catch ONE conversation gutted in place even while the total and
        # the count barely move — many small chats can hide one large one
        # emptied (the exact "gutted in place" proof above: 51 chats,
        # payload_ratio 1.0). None when _scan_chat itself could not key the
        # rows (no id column, or a duplicate id) — see its own
        # "why_no_rows"; the caller must not read that as "zero rows".
        "row_bytes": (
            {str(k): v[0] for k, v in rows.items()} if rows is not None else None
        ),
    }


def _webui_db_regressions(prev_src: dict, new_src: dict) -> list[str]:
    """Like _census_regressions, but for webui.db's own chat CONTENT rather
    than the compactor store (p4-b G1). `prev_src`/`new_src` are each
    manifest's `sources["webui.db"]` block.

    Cross-version / pre-fix compatibility, same doctrine as p3-b F3: an
    OLD manifest (this project's own code before this fix, or an archive a
    v3.1.6.1/v3.1.8 image wrote) never has a "chats" key at all — that is
    "no baseline to compare against", not "a baseline of zero chats", so a
    missing key here is silence, never an alarm. (Mirrors read_manifest's
    own docstring: an unreadable OLD archive is a missing baseline, never a
    baseline of zero.)

    A database that WAS present and readable and now reads absent or
    unreadable is not silence, though — it is exactly the G1 "replaced by
    an empty schema" shape, and the escape hatch that lets a cycle publish
    with no webui.db at all (COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB — see G6)
    must not make THIS check go quiet the moment it is used: a chats/
    content_bytes value that cannot be read this cycle reads as 0 for the
    floor below, exactly like a genuine drop to zero would.
    """
    out: list[str] = []
    if not isinstance(prev_src, dict) or not isinstance(new_src, dict):
        return out
    if "chats" not in prev_src:
        return out

    prev_chats = prev_src.get("chats")
    if isinstance(prev_chats, int) and prev_chats > 0:
        new_chats = new_src.get("chats")
        effective = new_chats if isinstance(new_chats, int) else 0
        if effective < prev_chats * _CENSUS_LOSS_FLOOR:
            out.append(f"webui.db.chats {prev_chats}->{effective}")

    prev_bytes = prev_src.get("content_bytes")
    if isinstance(prev_bytes, int) and prev_bytes > 0:
        new_bytes = new_src.get("content_bytes")
        effective_bytes = new_bytes if isinstance(new_bytes, int) else 0
        if effective_bytes < prev_bytes * _CENSUS_LOSS_FLOOR:
            out.append(f"webui.db.content_bytes {prev_bytes}->{effective_bytes}")

    # Per-conversation loss on a SURVIVING id — the shape neither floor
    # above can see when the gutted conversation is a small fraction of a
    # large db (the proof above: one 3 MB row emptied among 51 chats moves
    # the total by well under half). Reuses webuidb.MAX_ROW_LOSS_BYTES —
    # the SAME threshold the sync-publish guard applies to this exact
    # question, read fresh per call so an operator's env override applies
    # here too, not frozen at whatever it was at import.
    prev_rows = prev_src.get("row_bytes")
    new_rows = new_src.get("row_bytes")
    if isinstance(prev_rows, dict) and isinstance(new_rows, dict):
        import webuidb
        threshold = webuidb.MAX_ROW_LOSS_BYTES
        for cid, prev_n in prev_rows.items():
            if cid not in new_rows:
                continue  # a deletion, not a row gutted in place — not this check's question
            new_n = new_rows[cid]
            if (
                isinstance(prev_n, int) and isinstance(new_n, int)
                and prev_n - new_n > threshold
            ):
                out.append(f"webui.db.row[{cid[:12]}] {prev_n}->{new_n} bytes")
    return out


def read_manifest(archive_path: Path) -> dict | None:
    """Pull manifest.json out of a published archive without unpacking the
    rest. Returns None when it is absent or unreadable — callers must treat
    that as "no baseline to compare against", never as "a baseline of zero",
    or an unreadable old archive becomes a reason to distrust a good new one.
    """
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for name in (f"./{_MANIFEST_NAME}", _MANIFEST_NAME):
                try:
                    member = tar.getmember(name)
                except KeyError:
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                return json.loads(fh.read().decode("utf-8"))
    except Exception as e:
        logger.warning(
            f"could not read a manifest from {archive_path.name} "
            f"({type(e).__name__}: {e}); this cycle has no baseline to "
            f"compare against"
        )
    return None


def list_backups(backup_dir: Path | None = None) -> list[dict]:
    """Return existing archives, newest first: [{name, path, size_bytes, mtime}]."""
    d = backup_dir or BACKUP_DIR
    if not d.exists():
        return []
    out: list[dict] = []
    for f in d.glob(f"{_ARCHIVE_PREFIX}*{_ARCHIVE_SUFFIX}"):
        try:
            st = f.stat()
            out.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": st.st_size,
                "mtime": int(st.st_mtime),
            })
        except OSError:
            continue
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

def create_backup(
    backup_dir: Path | None = None, *, prev_manifest: dict | None = None
) -> Path:
    """Build a verified-elsewhere archive of webui.db + the compactor store.

    Writes to a `.partial` temp file; the caller (run_once) verifies it and
    only then is it published via os.replace. Returns the temp path.

    Raises RuntimeError if the min-free guard trips, or if the compactor
    store is missing (v3.1 F2 — see the guard at step 2).

    `prev_manifest` (p4-b G5) is the PREVIOUS archive's manifest, when the
    caller has one — read_manifest(list_backups(d)[0]) in run_once's own
    terms. It is used for exactly one thing: carrying the per-conversation
    high-water mark (census_hwm) forward so a slow bleed across many
    cycles — each individual drop under the 0.5 floor _census_regressions
    already checks against the immediate previous cycle — still trips a
    floor eventually, against the BEST state seen recently rather than
    only the state one cycle ago. None (the default, and what every
    existing caller before this fix passes implicitly) means "no
    baseline": the archive gets THIS cycle's own values as its starting
    high-water mark, and nothing is flagged — there is nothing yet to have
    fallen from.
    """
    d = backup_dir or BACKUP_DIR
    d.mkdir(parents=True, exist_ok=True)

    free = _free_mb(d)
    if free < MIN_FREE_MB:
        raise RuntimeError(
            f"refusing to back up: only {free:.0f} MB free at {d} "
            f"(min {MIN_FREE_MB} MB) — free space before backups can run"
        )

    stamp = _now_stamp()
    staging = Path(tempfile.mkdtemp(prefix=f"{_ARCHIVE_PREFIX}{stamp}-", dir=str(d)))
    manifest: dict = {
        "created_at": int(time.time()),
        "stamp": stamp,
        "sources": {},
        "format": "tar.gz",
        "schema": _SCHEMA,
    }
    try:
        # 1. webui.db via online snapshot (live-safe)
        db_dest = staging / "webui.db"
        # Resolved per cycle, not at import: the daemon is the longest-lived
        # process on the pod, and a gate flipped under it was invisible
        # until restart (A3-9). See live_webui_db.
        live_db = live_webui_db()
        if _snapshot_sqlite_to_data(live_db, db_dest):
            manifest["sources"]["webui.db"] = {
                "present": True, "bytes": db_dest.stat().st_size,
            }
            # p4-b G1. New keys, added to the EXISTING "webui.db" block an
            # old reader already only reads "present"/"bytes" out of by
            # name (p3-b F3's doctrine) — see _webui_db_scan_for_manifest's
            # own docstring for what they hold and why.
            chat_scan = _webui_db_scan_for_manifest(db_dest)
            if chat_scan is not None:
                manifest["sources"]["webui.db"].update(chat_scan)
        else:
            manifest["sources"]["webui.db"] = {"present": False}
            # p3-b F12. This branch used to log a WARNING and carry on — the
            # same shape v3.1 F2's store-missing guard (a few lines above,
            # STORAGE_ROOT.is_dir()) was written to close, applied at one
            # call site and missed at its sibling. The store is the smaller
            # half of the payload (manifests in this pod's own logs: ~138 MB
            # db vs ~436 MB total); dropping it leaves payload_ratio well
            # above MIN_PAYLOAD_RATIO, the census has no idea webui.db
            # exists, and the archive verifies green, publishes, and prunes
            # the real archives behind it — proof: SP\p3-b\nodb.py,
            # `payload_ratio: 0.683, pruned: [3 older archives]`, the only
            # trace a single WARNING line. A missing webui.db when one is
            # expected must fail the cycle and hold the prune, exactly like
            # a missing store — raise, with the same style of escape hatch
            # ALLOW_PUBLISH_OVER_UNREADABLE gives webuidb.py, for the one
            # legitimate case: a brand-new pod where the backup daemon's
            # first cycle races OpenWebUI's own first write.
            #
            # p4-b G6. THE HATCH NOW EXPIRES: it is honoured only while
            # list_backups(d) is EMPTY — a pod that has never published a
            # single archive yet, which is the only shape the "genuinely
            # fresh deployment" case in the message below actually
            # describes. Once one archive exists, the hatch is a no-op and
            # this refuses regardless of the env var. Why: the daemon's own
            # 15-minute retry already clears the fresh-pod race unaided
            # (misc4.py's own run_daemon simulation — the first cycle fires
            # before OpenWebUI's first write lands and simply retries), so
            # the hatch was never NEEDED for the case its own message
            # names — it was advice for a race that resolves itself. An
            # operator who followed that advice anyway left it set in the
            # RunPod template, where it survives every later redeploy, and
            # from then on ANY cycle where live_webui_db() resolves to a
            # missing file (misresolution, an unmounted volume, a moved db)
            # published a memory-only archive and pruned real ones behind
            # it — this is exactly p3-b F12 again, with no signal at all
            # once set (SP\p4-b\emptydb.py case 3: `ok: true, pruned:
            # [3 older archives]`, the escape hatch its only cause). Expiry
            # closes that: the hatch can no longer disable F12's refusal
            # on a pod that has ever successfully backed up before.
            _hatch_set = os.environ.get(
                "COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB", ""
            ).strip().lower() in ("1", "true", "yes")
            if _hatch_set and not list_backups(d):
                logger.warning(
                    f"webui.db not found at {live_db} — backing up memory "
                    f"only (COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB=1 is set and "
                    f"this pod has never published an archive yet)"
                )
            else:
                raise RuntimeError(
                    f"refusing to back up: webui.db not found at {live_db} "
                    f"— the live database is missing, unmounted, or "
                    f"live_webui_db() is resolving to the wrong path. Not "
                    f"writing an archive with no chat history behind the "
                    f"good ones it would age out through retention."
                    + (
                        f" COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB=1 is set, but "
                        f"this pod has ALREADY published at least one "
                        f"archive, so the hatch no longer applies — it is "
                        f"for a pod's very first cycle only, and its own "
                        f"15-minute retry already clears that race without "
                        f"it. If webui.db is genuinely gone or moved, fix "
                        f"that; do not re-widen this hatch to work around "
                        f"it."
                        if _hatch_set
                        # p4-b G6: the advice to set the hatch for "a
                        # genuinely fresh deployment racing OpenWebUI's own
                        # first write" is deliberately REMOVED here (it used
                        # to be printed in this exact message) — the race
                        # needs no hatch at all; see the comment above.
                        else ""
                    )
                )

        # 2. compactor/ store (atomic-written files are individually consistent)
        if not STORAGE_ROOT.is_dir():
            # Was: record {"present": False} and carry on — not even a log,
            # unlike the webui.db branch above. The trigger is ENOENT: an
            # unmounted /data, a lost network volume, a COMPACTOR_STORAGE_ROOT
            # typo. The result was an archive holding nothing but
            # manifest.json, which verified green, published, logged
            # "backup ok", and pruned the real archives behind it — exactly
            # when you were going to need them. Raising is the fix: run_once's
            # except path alerts and returns before prune_old_backups is
            # reached. (v3.1 F2.)
            raise RuntimeError(
                f"refusing to back up: the compactor store {STORAGE_ROOT} is not "
                f"a directory — the memory volume is missing, unmounted, or "
                f"COMPACTOR_STORAGE_ROOT is wrong. Not writing an empty archive."
            )
        store_dest = staging / "compactor"
        # chroma.sqlite3 is a live SQLite db written by the compactor process.
        # copytree can capture a torn page, and under WAL it would pair a
        # freshly copied db with a stale -wal/-shm — worse than either alone.
        # Excluded here and snapshotted in below via the same online backup
        # API webui.db uses. (v3.1 F2; interacts with F31 — if the chroma
        # store moves off /data this path moves with it.)
        shutil.copytree(
            STORAGE_ROOT,
            store_dest,
            ignore=shutil.ignore_patterns(f"{_CHROMA_DB_NAME}*"),
        )
        chroma_src = STORAGE_ROOT / _CHROMA_SUBDIR / _CHROMA_DB_NAME
        chroma_dest = store_dest / _CHROMA_SUBDIR / _CHROMA_DB_NAME
        chroma_present = False
        if chroma_src.is_file():
            chroma_dest.parent.mkdir(parents=True, exist_ok=True)
            chroma_present = _snapshot_sqlite_to_data(chroma_src, chroma_dest)
        else:
            logger.warning(
                f"episodic store {chroma_src} not found — this archive carries "
                f"facts, summaries and personas but no embedded exchanges"
            )
        n_files = sum(1 for f in store_dest.rglob("*") if f.is_file())
        n_json = sum(1 for _ in store_dest.rglob("*.json"))
        conversations_census = _census(store_dest)
        # p4-b G5: the high-water mark, carried forward from the PREVIOUS
        # archive's own manifest (never the live store — see this
        # function's own docstring for why None means "no baseline yet").
        # See _census_hwm_update's docstring for the reset-on-fire rule
        # that gives this alarm-once semantics.
        prev_hwm = (
            ((prev_manifest or {}).get("sources", {}).get("compactor", {}) or {})
            .get("census_hwm") or {}
        )
        new_hwm, _ = _census_hwm_update(prev_hwm, conversations_census)
        manifest["sources"]["compactor"] = {
            "present": True,
            "files": n_files,
            # The count verify_backup asserts against. Deliberately not
            # `files`: that includes chroma.sqlite3 and any binary index
            # files, so comparing a parsed-JSON count to it would fail every
            # archive that has an episodic store.
            "json_files": n_json,
            "chroma_sqlite": chroma_present,
            "conversations": conversations_census,
            "census_hwm": new_hwm,
        }
        manifest["payload_bytes"] = _tree_bytes(staging)

        (staging / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # 3. tar.gz the staging dir to a .partial temp archive
        partial = d / f"{_ARCHIVE_PREFIX}{stamp}{_ARCHIVE_SUFFIX}.partial"
        with tarfile.open(partial, "w:gz") as tar:
            tar.add(staging, arcname=".")
        return partial
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_backup(archive_path: Path) -> tuple[bool, str]:
    """Restore an archive to a scratch dir and assert it's actually usable:
      - the tar opens and extracts
      - manifest.json is present and parses
      - the archive claims to hold *something*
      - if webui.db was backed up, it opens AND PRAGMA integrity_check == ok
      - if the manifest claims a compactor store, the directory is there and
        at least as many JSON files parse as the manifest counted
      - if the manifest claims chroma.sqlite3, it is there and passes
        integrity_check
      - the per-conversation census recomputed from the archive is not short
        of the census the manifest recorded
      - every *.json under compactor/ parses

    Every one of those checks reads its expectation out of the manifest and
    then tries to *contradict* it from the extracted tree. Before v3.1 the
    only manifest-driven check was webui.db, and the compactor half was a
    bare `if store.is_dir():` — so an archive of nothing but manifest.json
    returned (True, "db=absent, 0 json file(s) parsed"). (v3.1 F2.)

    Returns (ok, detail). Never raises — a failure to verify is a False, not
    an exception, so the caller can delete the bad archive and carry on.
    """
    scratch = Path(tempfile.mkdtemp(prefix="zions-verify-"))
    try:
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(scratch, filter="data")  # path-traversal safe
        except Exception as e:
            return False, f"tar extract failed: {type(e).__name__}: {e}"

        manifest_path = scratch / _MANIFEST_NAME
        if not manifest_path.is_file():
            return False, "manifest.json missing from archive"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"manifest unparseable: {e}"

        sources = manifest.get("sources", {})
        if not isinstance(sources, dict):
            return False, "manifest has no usable sources block"
        db_src = sources.get("webui.db") or {}
        store_src = sources.get("compactor") or {}
        db_expected = bool(db_src.get("present"))
        store_expected = bool(store_src.get("present"))

        # The compactor store is the half of this that cannot be regenerated
        # from anywhere else, so an archive whose own manifest records it as
        # absent is not a recovery point no matter what else it holds. This
        # is the shape the empty-backup bug produced, and archives in this
        # shape are already on disk: they must read FAIL in /admin/backups
        # rather than sit there looking like history. Backing up webui.db
        # alone is not a supported mode — create_backup raises instead.
        # (v3.1 F2.)
        if not store_expected:
            return False, (
                "manifest records no compactor store — this archive cannot "
                "restore the memory and is not a recovery point"
            )

        # SQLite integrity (only if it was supposed to be there)
        db_path = scratch / "webui.db"
        if db_expected:
            if not db_path.is_file():
                return False, "manifest says webui.db present but it's missing"
            try:
                con = sqlite3.connect(str(db_path))
                try:
                    row = con.execute("PRAGMA integrity_check").fetchone()
                finally:
                    con.close()
                if not row or row[0] != "ok":
                    return False, f"sqlite integrity_check failed: {row}"
            except Exception as e:
                return False, f"sqlite open/check failed: {type(e).__name__}: {e}"

        # The compactor store, asserted against the manifest.
        store = scratch / "compactor"
        if store_expected and not store.is_dir():
            return False, "manifest says the compactor store is present but compactor/ is missing"

        # Every memory JSON must parse
        json_checked = 0
        if store.is_dir():
            for jf in store.rglob("*.json"):
                try:
                    json.loads(jf.read_text(encoding="utf-8"))
                    json_checked += 1
                except Exception as e:
                    return False, f"corrupt memory file {jf.name}: {e}"

        if store_expected:
            expected_json = store_src.get("json_files")
            # v1 manifests recorded only a total file count, which includes
            # chroma.sqlite3 and would fail every archive that has one. They
            # get the directory check and the parse check and no count check —
            # refusing to verify an old archive is refusing to restore it.
            if isinstance(expected_json, int) and json_checked < expected_json:
                return False, (
                    f"manifest counted {expected_json} memory JSON file(s) but "
                    f"only {json_checked} are in the archive"
                )

        # ChromaDB's own SQLite. NOTE: PRAGMA integrity_check validates SQLite
        # *pages*. It says nothing about whether the application-level
        # structure inside those pages is coherent — the 2026-08-24
        # parent-pointer corruption would have passed this check green. A
        # green result here means "the file is not torn", and nothing more.
        # Do not let it stand in for "the memory is intact".
        if store_src.get("chroma_sqlite"):
            cdb = store / _CHROMA_SUBDIR / _CHROMA_DB_NAME
            if not cdb.is_file():
                return False, "manifest says chroma.sqlite3 present but it's missing"
            try:
                con = sqlite3.connect(str(cdb))
                try:
                    row = con.execute("PRAGMA integrity_check").fetchone()
                finally:
                    con.close()
                if not row or row[0] != "ok":
                    return False, f"chroma.sqlite3 integrity_check failed: {row}"
            except Exception as e:
                return False, f"chroma.sqlite3 open/check failed: {type(e).__name__}: {e}"

        # The census the manifest recorded must still be satisfiable from the
        # archive. This is the check that catches a store which extracted but
        # came back emptier than it was counted.
        shortfalls = []
        if store_expected and isinstance(store_src.get("conversations"), dict):
            shortfalls = _census_shortfalls(store_src["conversations"], _census(store))
            if shortfalls:
                return False, (
                    f"archive is short of its own manifest census: "
                    f"{', '.join(shortfalls[:5])}"
                    + (f" (+{len(shortfalls) - 5} more)" if len(shortfalls) > 5 else "")
                )

        n_convs = len(store_src.get("conversations") or {})
        return True, (
            f"db={'ok' if db_expected else 'absent'}, "
            f"chroma={'ok' if store_src.get('chroma_sqlite') else 'absent'}, "
            f"{json_checked} json file(s) parsed, {n_convs} conversation(s)"
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------

def _keep_set(
    archives: list[dict], *, now: float | None = None, floor: int | None = None
) -> set[str]:
    """Names of the archives retention claims. `archives` is newest-first, as
    list_backups returns it.

    Four tiers, unioned — an archive survives if *any* of them wants it:

      1. Floor.  The newest `floor` archives, whatever their age. Nothing
         below this line is prunable by any code path. Floored at 3.
      2. Age.    Everything younger than RETAIN_DAYS.
      3. Daily.  The newest archive of each UTC day inside RETAIN_DAYS.
      4. Weekly. The newest archive of each ISO week inside GFS_WEEKS.

    Tier 3 is redundant while tier 2 keeps everything in the same window, and
    that is deliberate: it is what the window shrinking to a day still leaves
    behind. (v3.1 F7 / D9.)
    """
    now = time.time() if now is None else now
    floor = MIN_KEEP if floor is None else max(MIN_KEEP, int(floor))
    keep: set[str] = {e["name"] for e in archives[:floor]}

    days_seen: set[tuple] = set()
    weeks_seen: set[tuple] = set()
    for entry in archives:
        age_days = (now - entry["mtime"]) / 86400.0
        when = datetime.datetime.fromtimestamp(
            entry["mtime"], datetime.timezone.utc
        )
        if age_days <= RETAIN_DAYS:
            keep.add(entry["name"])
            day = (when.year, when.month, when.day)
            if day not in days_seen:
                days_seen.add(day)
                keep.add(entry["name"])
        if age_days <= GFS_WEEKS * 7:
            iso = when.isocalendar()
            week = (iso[0], iso[1])
            if week not in weeks_seen:
                weeks_seen.add(week)
                keep.add(entry["name"])
    return keep


def prune_old_backups(
    backup_dir: Path | None = None,
    retain: int | None = None,
    *,
    now: float | None = None,
) -> list[str]:
    """Delete archives no retention tier claims. Returns names removed.

    `retain` is now a **floor** on the number kept, not a cap. It was a cap —
    "delete everything past the newest N" — called unconditionally at the end
    of every cycle, including cycles fired by a container restart. That is
    how N restarts inside one backup interval replaced every pre-incident
    archive with N copies of the damaged state. (v3.1 F7 / D9.)
    """
    archives = list_backups(backup_dir)
    floor = max(MIN_KEEP, RETAIN) if retain is None else max(MIN_KEEP, int(retain))
    keep = _keep_set(archives, now=now, floor=floor)
    removed: list[str] = []
    for entry in archives:
        if entry["name"] in keep:
            continue
        try:
            Path(entry["path"]).unlink()
            removed.append(entry["name"])
        except OSError as e:
            logger.warning(f"could not prune {entry['name']}: {e}")
    return removed


# ---------------------------------------------------------------------------
# Off-volume seam (future work)
# ---------------------------------------------------------------------------

def upload_hook(archive_path: Path) -> bool:
    """Designed-in seam for off-volume disaster recovery (object store).

    V2.3 phase 1 is local-only, so this is a no-op unless
    COMPACTOR_BACKUP_REMOTE is set — and even then it currently only logs,
    because true off-volume DR is deferred future work that will need a
    migration (provider choice + credentials + a real uploader, e.g. boto3
    or rclone). Wiring it here now means the create→verify→publish→upload
    pipeline already has the call site.
    """
    if not REMOTE_TARGET:
        return False
    logger.warning(
        f"COMPACTOR_BACKUP_REMOTE={REMOTE_TARGET!r} is set but off-volume "
        f"upload is not yet implemented (V2.3 future work). Archive "
        f"{archive_path.name} kept locally only."
    )
    return False


# ---------------------------------------------------------------------------
# Orchestration: create → verify → publish → prune
# ---------------------------------------------------------------------------

def _payload_ratio(
    prev_manifest: dict | None,
    prev_entry: dict | None,
    new_manifest: dict | None,
    new_path: Path,
) -> float | None:
    """new payload ÷ previous payload, or None when there is no comparable
    baseline (first ever backup, unreadable previous manifest, previous
    payload of zero). None means "cannot judge" — it must never be read as a
    ratio of 0, or the first backup on a fresh volume would refuse itself."""
    if prev_manifest is None and prev_entry is None:
        return None
    old = (prev_manifest or {}).get("payload_bytes")
    new = (new_manifest or {}).get("payload_bytes")
    if not isinstance(old, int) or not isinstance(new, int):
        # v1 archives carry no payload_bytes. Fall back to the compressed
        # archive size — coarser, since compression ratios move with content,
        # but always available and still catches a collapse to near-nothing.
        if not prev_entry:
            return None
        old = int(prev_entry.get("size_bytes") or 0)
        try:
            new = new_path.stat().st_size
        except OSError as e:
            # None here means "no baseline to compare against", which the
            # caller reads as "cannot judge, allow". A failed stat is a
            # different thing wearing the same value, so say so rather than
            # letting a collapse-check silently not run.
            logger.warning(
                f"backup sizing: could not stat {new_path} ({e}); the "
                f"payload-collapse check is skipped for this cycle"
            )
            return None
    if old <= 0:
        return None
    return new / old


_STALE_DEBRIS_HOURS = env_float("COMPACTOR_BACKUP_STALE_DEBRIS_HOURS", 2.0)


def _sweep_stale_backup_debris(d: Path) -> list[str]:
    """Remove `zions-backup-*` STAGING directories and `*.tar.gz.partial`
    files in `d` older than `_STALE_DEBRIS_HOURS`. Returns names removed.

    p3-b F13. A SIGKILL during create_backup (a redeploy mid-cycle skips
    every `finally`) leaves the full uncompressed staging tree (the WHOLE
    payload, ~436 MB at the manifest sizes seen in this pod's own logs)
    and/or a `*.tar.gz.partial` behind, forever: `list_backups` globs only
    `*.tar.gz`, so neither is ever listed, pruned or reported, and on
    MooseFS the MIN_FREE_MB statvfs guard never sees the space go. F10's
    96-cycles-a-day-during-a-persistent-failure shape multiplies the
    exposure. Age-gated (not "anything not the current run") because a
    concurrent create_backup legitimately has its OWN staging dir/partial
    on disk at the moment this runs — `_STALE_DEBRIS_HOURS` (default 2h)
    is well past how long a single cycle plausibly takes even on a
    degraded volume, and short enough that a killed cycle's debris does
    not sit around for the retention window's own scale of time.
    """
    removed: list[str] = []
    if not d.exists():
        return removed
    cutoff = time.time() - _STALE_DEBRIS_HOURS * 3600
    candidates = list(d.glob(f"{_ARCHIVE_PREFIX}*")) + list(d.glob("*.tar.gz.partial"))
    for f in candidates:
        # Never touch a real, published archive — only STAGING dirs
        # (`zions-backup-<stamp>-<mkdtemp suffix>`, always a directory,
        # never ending in the archive suffix) and `.partial` files.
        if f.is_file() and not f.name.endswith(".partial"):
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        try:
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
            removed.append(f.name)
        except OSError as e:
            logger.warning(f"could not remove stale backup debris {f}: {e}")
    if removed:
        logger.warning(
            f"removed {len(removed)} stale backup debris item(s) from {d} "
            f"(older than {_STALE_DEBRIS_HOURS:.0f}h — likely left by a "
            f"killed cycle): {removed}"
        )
    return removed


def run_once(backup_dir: Path | None = None) -> dict:
    """One full backup cycle. Returns a structured report. Never raises —
    failures are reported, not thrown, so the daemon keeps running.

    **Nothing is pruned unless the cycle is fully clean.** A failure, a
    refused publish, or a census that went backwards all return before
    prune_old_backups. (v3.1 F2/F7.)
    """
    d = backup_dir or BACKUP_DIR
    _sweep_stale_backup_debris(d)  # p3-b F13
    t0 = time.monotonic()
    report: dict = {"ok": False, "archive": None, "verified": False, "detail": ""}
    partial: Path | None = None
    # Read the baseline BEFORE creating the new archive, or the new one is
    # its own baseline and every comparison below is vacuous — the same
    # mistake verify_backup made with the manifest.
    existing = list_backups(d)
    # p4-b G6: loud, EVERY cycle, whenever the hatch is armed — not only on
    # the cycle it actually gets used. Before this fix the only trace of
    # COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB=1 being set was a single WARNING
    # on a cycle that happened to hit a missing webui.db; on every other
    # cycle (the overwhelming majority, once set and forgotten in a RunPod
    # template) it was completely silent — an operator watching only
    # normal "backup ok" lines had no way to notice the safety net was off
    # long before the day it mattered. Once list_backups(d) is non-empty
    # the hatch no longer even does anything (see create_backup's own
    # comment on the refusal it used to silently bypass), so this doubles
    # as a nag to go unset it.
    if os.environ.get(
        "COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB", ""
    ).strip().lower() in ("1", "true", "yes"):
        logger.warning(
            "COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB=1 is set"
            + (
                " and this pod has already published archives, so it no "
                "longer has any effect — unset it."
                if existing
                else " — a missing webui.db will publish a memory-only "
                "archive instead of refusing. This is meant for a pod's "
                "very first cycle only; unset it once the first archive "
                "exists."
            )
        )
    prev_entry = existing[0] if existing else None
    prev_manifest = read_manifest(Path(prev_entry["path"])) if prev_entry else None
    try:
        # p4-b G5: prev_manifest passed through so create_backup can carry
        # the per-conversation census high-water mark forward — see its
        # own docstring and _census_hwm_update.
        partial = create_backup(d, prev_manifest=prev_manifest)
        ok, detail = verify_backup(partial)
        report["detail"] = detail
        if not ok:
            # No false confidence — delete the unverifiable archive.
            try:
                partial.unlink()
            except OSError as e:
                logger.debug(f"could not remove unverifiable {partial.name}: {e}")
            report["detail"] = f"VERIFICATION FAILED: {detail}"
            logger.error(f"backup verification failed, archive discarded: {detail}")
            _alert_failure(report["detail"])
            return report
        report["verified"] = True
        new_manifest = read_manifest(partial)

        # Payload collapse. A store that lost more than half its bytes since
        # the last cycle is a volume that went away, not a user deleting
        # things — and publishing it would make it the newest archive and
        # push a good one a step closer to the prune. (v3.1 F2.)
        ratio = _payload_ratio(prev_manifest, prev_entry, new_manifest, partial)
        report["payload_ratio"] = None if ratio is None else round(ratio, 3)
        if ratio is not None and ratio < MIN_PAYLOAD_RATIO:
            try:
                partial.unlink()
            except OSError as e:
                logger.debug(f"could not remove shrunken {partial.name}: {e}")
            report["detail"] = (
                f"PAYLOAD COLLAPSED: new archive is {ratio:.0%} of "
                f"{prev_entry['name'] if prev_entry else 'the previous archive'} "
                f"(floor {MIN_PAYLOAD_RATIO:.0%}) — refusing to publish it or "
                f"prune behind it"
            )
            logger.error(f"backup refused: {report['detail']}")
            _alert_failure(report["detail"])
            return report

        # Publish atomically: drop the .partial suffix.
        final = partial.with_suffix("")  # strips ".partial" → ...tar.gz
        os.replace(partial, final)
        report["archive"] = final.name
        upload_hook(final)

        # Census regression against the PREVIOUS archive. The new archive is
        # published either way — it is real data and keeping it is never the
        # wrong move — but a store that went backwards is the one condition
        # under which the older archives are the valuable ones, so the prune
        # is skipped and someone is told. (v3.1 F2.)
        losses: list[str] = []
        prev_census = ((prev_manifest or {}).get("sources", {})
                       .get("compactor", {}) or {}).get("conversations")
        new_census = ((new_manifest or {}).get("sources", {})
                      .get("compactor", {}) or {}).get("conversations")
        if isinstance(prev_census, dict) and isinstance(new_census, dict):
            # v3.1.9 (hostile317-c F1): _census_regressions, NOT
            # _census_shortfalls — this is a cross-CYCLE comparison, where
            # normal eviction/dedup/rollup legitimately shrinks the raw
            # numbers; _census_shortfalls's strict "any decrease" rule is
            # for verify_backup's within-one-archive integrity check only.
            losses = _census_regressions(prev_census, new_census)
        # p4-b G1: the census above watches the compactor STORE only — it
        # has no idea webui.db, her whole chat history, exists at all. The
        # store is the SMALLER half of the real payload (this pod's own
        # manifests: ~138 MB db vs ~436 MB total), so a webui.db replaced
        # by an empty schema or gutted in place holds payload_ratio well
        # above MIN_PAYLOAD_RATIO and census_regressions empty, and prunes
        # the good archives behind it (SP\p4-b\emptydb.py). Same doctrine
        # as the census check just above: hold the prune, publish anyway,
        # tell someone — never silently lose the older archives that would
        # still have her history.
        losses += _webui_db_regressions(
            (prev_manifest or {}).get("sources", {}).get("webui.db") or {},
            (new_manifest or {}).get("sources", {}).get("webui.db") or {},
        )
        # p4-b G5: the slow-bleed check, against the high-water mark
        # create_backup already computed and baked into new_manifest (it
        # needed prev_manifest's own hwm to do that, at creation time — see
        # create_backup's docstring). Recomputed here rather than threaded
        # back out of create_backup's return value: the inputs
        # (prev_manifest's hwm, and new_census — the SAME dict
        # create_backup just wrote into the archive it built from) are
        # already both in hand, and _census_hwm_update is a pure function
        # of them, so this reproduces the identical loss list create_backup
        # itself would have seen, without changing create_backup's return
        # type for every OTHER caller (the CLI, the suites).
        if isinstance(new_census, dict):
            prev_hwm = (
                (prev_manifest or {}).get("sources", {}).get("compactor", {}) or {}
            ).get("census_hwm") or {}
            losses += _census_hwm_update(prev_hwm, new_census)[1]
        report["census_regressions"] = losses
        report["ok"] = True
        report["elapsed_s"] = round(time.monotonic() - t0, 1)
        if losses:
            report["pruned"] = []
            summary = (
                f"memory shrank since {prev_entry['name']}: "
                f"{', '.join(losses[:5])}"
                + (f" (+{len(losses) - 5} more)" if len(losses) > 5 else "")
            )
            report["detail"] = f"{detail}; {summary}"
            logger.warning(
                f"backup ok: {final.name} ({detail}); NOT pruning — {summary}"
            )
            _alert_failure(f"backup published but {summary}")
            return report
        removed = prune_old_backups(d)
        report["pruned"] = removed
        logger.info(
            f"backup ok: {final.name} ({detail}); pruned {len(removed)}; "
            f"{report['elapsed_s']}s"
        )
        return report
    except Exception as e:
        if partial and partial.exists():
            try:
                partial.unlink()
            except OSError as unlink_err:
                logger.debug(
                    f"could not remove partial {partial.name}: {unlink_err}"
                )
        report["detail"] = f"{type(e).__name__}: {e}"
        logger.error(f"backup failed: {report['detail']}")
        _alert_failure(report["detail"])
        return report


def _alert_failure(detail: str) -> None:
    """Best-effort failure alert (V2.3 Theme 4). No-op if no webhook set."""
    try:
        import alert
        alert.notify("backup", "fail", detail)
    except Exception as e:
        # The alert about a failure could itself vanish: this handler was a
        # bare `pass`, so a broken webhook, a missing alert module or a DNS
        # failure silently ate the only outbound signal the backup daemon
        # has. Not once-per-process — it fires only on a backup that already
        # failed, and every one of those is worth a line. (v3.1 P0-2b / F61.)
        logger.error(
            f"could not send backup failure alert ({type(e).__name__}: {e}); "
            f"the failure it was reporting was: {detail}"
        )


def latest_backup_info(backup_dir: Path | None = None) -> dict:
    """Summary for /health/full + admin: count + newest timestamp."""
    archives = list_backups(backup_dir)
    return {
        "count": len(archives),
        "latest": archives[0]["name"] if archives else None,
        "latest_mtime": archives[0]["mtime"] if archives else None,
        "dir": str(backup_dir or BACKUP_DIR),
    }


# ---------------------------------------------------------------------------
# Restore (destructive — for the runbook / CLI, gated by --yes)
# ---------------------------------------------------------------------------

def restore_backup(
    archive_path: Path,
    *,
    data_dir: Path | None = None,
    storage_root: Path | None = None,
    webui_db: Path | None = None,
    confirm: bool = False,
) -> dict:
    """Restore an archive over the live data locations. DESTRUCTIVE — it
    overwrites webui.db and the compactor store. Requires confirm=True.

    Verifies the archive first (won't restore an unusable backup), then
    extracts to scratch and moves the pieces into place. Returns a report.
    """
    if not confirm:
        raise RuntimeError("restore is destructive; pass confirm=True (CLI: --yes)")
    sroot = storage_root or STORAGE_ROOT

    ok, detail = verify_backup(archive_path)
    if not ok:
        raise RuntimeError(f"refusing to restore an unverifiable archive: {detail}")

    scratch = Path(tempfile.mkdtemp(prefix="zions-restore-"))
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(scratch, filter="data")  # path-traversal safe
        restored: list[str] = []
        src_db = scratch / "webui.db"
        src_store = scratch / "compactor"
        have_db = src_db.is_file()
        have_store = src_store.is_dir()

        # WHERE THE LIVE DATABASE ACTUALLY IS — resolved now, by the gate, the
        # same way create_backup resolves it. `data_dir` no longer selects the
        # target: a data root cannot say where the live database is (under
        # WEBUI_DB_LOCAL=true it is not under the data root at all), and no
        # caller in the tree ever passed it. The parameter stays so an old
        # call does not break; it is simply not evidence.
        target = webui_db or live_webui_db()
        _require_no_active_writer(target)
        stamp = _restore_stamp()

        # p4-b G3: refuse to START a new restore while an EARLIER one's
        # marker is still on disk. This run has not written ITS OWN marker
        # yet (that happens a bit further down, once staging succeeds) —
        # so any marker found here belongs to a run this process did not
        # start: one genuinely still in flight (a concurrent --restore), or
        # one left behind by an earlier kill that was never cleaned up
        # (G4). Stacking a second restore on top of live paths that might
        # already be missing or at a mixed generation compounds exactly the
        # failure the marker exists to flag, the same way
        # _require_no_active_writer above refuses to restore under an open
        # writer rather than trying to reason about what it might do.
        import webuidb
        _earlier_marker = webuidb.find_interrupted_restore()
        if _earlier_marker is not None:
            raise RuntimeError(
                f"refusing to restore: an earlier restore's marker is "
                f"still on disk at {_earlier_marker.get('marker_path')} — "
                f"either a restore is genuinely still in flight, or an "
                f"earlier one was interrupted and never cleaned up. "
                f"Resolve that first (see OPERATIONS.md's restore section "
                f"for what the marker's plan means and how to clear it "
                f"safely) before starting a new one: "
                f"{json.dumps(_earlier_marker, indent=1)}"
            )

        # ------------------------------------------------------------------
        # STAGE EVERYTHING FIRST, TOUCH NOTHING LIVE UNTIL IT IS STAGED.
        #
        # v3.1.9 (A3-2, A3-3, A3-12). The previous version was two different
        # defects of one shape — move-then-write — ten lines apart:
        #
        #   * the database's SIDECARS were renamed aside before the copy that
        #     could fail, so an ENOSPC left the original database byte-for-byte
        #     intact and stripped of its hot journal: the one file that can roll
        #     back a half-applied transaction, gone at the moment an operator
        #     is recovering from something. SQLite then opens it silently,
        #     because as far as it can see there is nothing to replay.
        #   * the STORE was `rmtree(sroot)` then `copytree` — every fact, every
        #     summary tier, every persona and ChromaDB deleted before the first
        #     byte came back, with no set-aside, in the commit whose message
        #     said it had fixed this function's atomicity. It fixed the half
        #     above and walked past the half below.
        #
        # So both halves are copied to siblings of their targets first, and
        # fsynced. Only then does anything live move, and every live move is a
        # rename on one filesystem. An interruption during staging leaves the
        # live data exactly as it was plus some `.incoming` debris. An
        # interruption during the renames leaves every piece on disk under a
        # name that says what it is, and the log line below says how to put it
        # back.
        # ------------------------------------------------------------------
        needed: list[tuple[Path, int]] = []
        if have_db:
            needed.append((target.parent, src_db.stat().st_size))
            # p3-b F11: the OLD db, if any, is about to be quarantined
            # (same volume as webuidb.QUARANTINE, never deleted) rather
            # than freed — it exists on that volume TWICE for a while,
            # which both the free-space and the quota check should see.
            if target.exists():
                try:
                    needed.append((_quarantine_dir(), target.stat().st_size))
                except OSError:
                    pass
        if have_store:
            needed.append((sroot.parent, _tree_bytes(src_store)))
            if sroot.exists():
                try:
                    needed.append((_quarantine_dir(), _tree_bytes(sroot)))
                except OSError:
                    pass
        _require_free_space(needed)

        db_tmp = target.with_name(f"{target.name}.restore-{stamp}")
        store_incoming = sroot.with_name(f"{sroot.name}.incoming-{stamp}")
        try:
            if have_db:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_db, db_tmp)
                _fsync_file(db_tmp)
            if have_store:
                sroot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_store, store_incoming)
                _fsync_tree(store_incoming)
        except BaseException:
            db_tmp.unlink(missing_ok=True)
            shutil.rmtree(store_incoming, ignore_errors=True)
            raise

        # Staged. From here every step is a rename, and the recovery path for
        # each possible stopping point is written down BEFORE the first one.
        logger.warning(
            f"restoring {archive_path.name}: staged "
            f"{'webui.db -> ' + db_tmp.name if have_db else 'no webui.db'} and "
            f"{'store -> ' + store_incoming.name if have_store else 'no store'}. "
            f"If this process dies before 'restored' is logged, nothing was "
            f"deleted: any pre-restore originals are named *.pre-restore-"
            f"{stamp} under {_quarantine_dir()}, and the staged copies are "
            f"the *.restore-{stamp} / *.incoming-{stamp} names beside their "
            f"targets."
        )

        # p3-b F4/F9. From here on, every step is a live-path move, and a
        # SIGKILL or an EIO between any two of them can leave the live db
        # missing, the live store missing, or the two at different
        # generations, with nothing on disk recording that a restore was
        # ever in flight — see webuidb.find_interrupted_restore()'s
        # docstring for the full reasoning and what remains unimplemented
        # (resuming or auto-undoing the interrupted restore; this closes
        # detection and refusal, not recovery). Written to QUARANTINE, the
        # SAME filesystem the set-asides already land on — never local
        # disk, which a redeploy does not keep. Removed ONLY on the fully-
        # successful return below: an exception here, even one this
        # function's own handler recovers from, leaves the marker and
        # refuses the next boot until a human clears it by hand, on
        # purpose (a false-positive refusal after a handled failure is
        # cheap; silently clearing evidence of a restore that might not
        # have finished cleanly is not).
        import webuidb
        webuidb.write_restore_marker(stamp, {
            "archive": archive_path.name,
            "target_db": str(target) if have_db else None,
            "staged_db_tmp": str(db_tmp) if have_db else None,
            "sroot": str(sroot) if have_store else None,
            "staged_store_incoming": str(store_incoming) if have_store else None,
            "quarantine_dir": str(_quarantine_dir()),
        })

        # v3.1.9 round 2 (finding 8): tracked OUTSIDE the `if have_db:` block
        # below so the `if have_store:` block's own failure handler can roll
        # the database back too, if the store swap fails AFTER the database
        # swap already landed — see that block for why.
        db_aside: Path | None = None
        # p3-b F5: hoisted out of the `if have_db:` block for the same
        # reason as db_aside, immediately above — it USED to be local to
        # that block, so when the store swap failed after the db swap had
        # already landed, the failure handler could put db_aside back but
        # had no way to reach the journal/-wal it had moved to quarantine a
        # few lines earlier. The live db then sat there with NO journal
        # beside it — the exact hot-journal-stripped corruption A3-2 was
        # written to prevent, rebuilt inside its own fix. See the
        # `if have_store:` block's exception handler below.
        moved: list[tuple[Path, Path]] = []

        if have_db:
            # v3.1.9 round 2 (finding 8). The ORIGINAL database is now
            # quarantined too, the SAME way the store already is a few lines
            # down — not just its sidecars. Without this, os.replace(db_tmp,
            # target) was a single atomic swap with no way back: if the
            # STORE swap that follows it then failed, the live state was
            # stuck at (NEW db, OLD store) — a mixed generation with nothing
            # to roll back to, because the old db's bytes were simply gone.
            # Preserving it here lets that failure roll the db back too,
            # matching the store's own doctrine exactly.
            if target.exists():
                db_aside = _quarantine_aside(target, stamp)

            # SIDECARS TRAVEL WITH THE DATABASE. A -journal or -wal beside the
            # target belongs to the database being REPLACED, and SQLite applies
            # it to whatever file carries that name on the next open. Renamed,
            # never deleted — and now only once the replacement is certain, and
            # put BACK if the replace itself fails, so no path through here
            # leaves the original without its journal. (`moved` is declared
            # above have_db, not here — p3-b F5.)
            for suffix in SIDECARS:
                side = target.with_name(target.name + suffix)
                if side.exists():
                    aside = _quarantine_aside(side, stamp)
                    moved.append((side, aside))
                    logger.warning(
                        f"moved {side.name} aside to {aside} before "
                        f"restoring — it belongs to the database being "
                        f"replaced, and SQLite would have applied it to the "
                        f"restored one"
                    )
            try:
                os.replace(db_tmp, target)
            except BaseException:
                _db_swap_rollback_ok = True
                if db_aside is not None:
                    try:
                        shutil.move(str(db_aside), str(target))
                    except OSError as e:
                        _db_swap_rollback_ok = False
                        logger.error(
                            f"could not move {db_aside} back to {target} "
                            f"after a failed restore ({e}); move it back by "
                            f"hand BEFORE anything opens {target.name}"
                        )
                for side, aside in reversed(moved):
                    try:
                        shutil.move(str(aside), str(side))
                    except OSError as e:
                        _db_swap_rollback_ok = False
                        logger.error(
                            f"could not put {aside} back as {side.name} "
                            f"after a failed restore ({e}); move it back by "
                            f"hand BEFORE anything opens {target.name}"
                        )
                db_tmp.unlink(missing_ok=True)
                # Nothing live has moved on the store side yet, so its staged
                # copy is only debris; leaving it would be a multi-gigabyte
                # `.incoming` nobody is told about.
                shutil.rmtree(store_incoming, ignore_errors=True)
                # p4-b G4: this handler's rollback puts the live state back
                # to EXACTLY its pre-restore condition (db + sidecars both
                # accounted for above) — the marker written a few lines above
                # `if have_db:` exists to protect against an UNKNOWN mixed
                # live state, and there is not one here when every move above
                # actually landed. Leaving the marker anyway would refuse
                # every later boot for a restore that is, on the live paths,
                # as if it never ran. Only clear it when EVERY move above is
                # confirmed back — a partial rollback is exactly the unknown
                # state the marker exists to flag, and clearing it there
                # would silently erase the one piece of evidence an operator
                # has that something is still wrong.
                if _db_swap_rollback_ok:
                    import webuidb
                    webuidb.remove_restore_marker(stamp)
                raise
            _fsync_dir(target.parent)

            # v3.1.9 round 2 (finding 10). verify_backup already ran
            # PRAGMA integrity_check against the ARCHIVE's own extracted
            # copy before any of this staging began — this checks the file
            # that actually ENDED UP live, which is a different question: a
            # bad copy2, a race, or a filesystem problem during staging or
            # the rename itself would not show up in the archive-level
            # check at all. Import lazily (same reasoning as
            # _quarantine_dir above): webuidb.integrity() is the ONE
            # existing implementation of this check, reused rather than
            # duplicated.
            import webuidb
            db_ok, db_detail = webuidb.integrity(target)
            if not db_ok:
                # LOUD, and the set-aside is left exactly where it landed —
                # NOT auto-restored: this is the only known-good pre-restore
                # copy, and silently overwriting the just-landed (corrupt)
                # file would destroy the evidence of what actually went
                # wrong. The store swap below does not run: proceeding to
                # replace the store on top of a database already known bad
                # would compound the failure, not just report it.
                logger.error(
                    f"the RESTORED database at {target} FAILED "
                    f"PRAGMA integrity_check immediately after landing "
                    f"({db_detail}). This is NOT rolled back automatically: "
                    f"the pre-restore original is preserved at {db_aside} "
                    f"(nothing will delete it) for you to restore by hand — "
                    f"`supervisorctl stop openwebui compactor backup "
                    f"webuidb-sync`, move it back over {target}, then "
                    f"restart. The store was NOT touched."
                )
                # The store's own staged copy was already built and fsynced
                # during the earlier staging phase (it runs whenever
                # have_store, unconditionally, before either swap) — this
                # raise happens BETWEEN staging and the store's own swap, so
                # nothing else would ever clean it up. Same reasoning as the
                # db-swap failure handler a few lines up: leaving it would be
                # a multi-gigabyte `.incoming` nobody is told about. Found by
                # this fix's OWN test on the real Linux container — a first
                # version of this raise left exactly this debris behind.
                shutil.rmtree(store_incoming, ignore_errors=True)
                raise RuntimeError(
                    f"restored database failed integrity_check after "
                    f"landing: {db_detail}. See the log above for the "
                    f"pre-restore copy's location."
                )
            restored.append("webui.db")

        if have_store:
            # Set aside, never deleted. The old store is the only copy of
            # everything written since the archive was taken; an operator who
            # restored the wrong archive needs it back, and this is the moment
            # nobody can tell yet whether they did.
            store_aside = None
            if sroot.exists():
                store_aside = _quarantine_aside(sroot, stamp)
            try:
                os.replace(store_incoming, sroot)
            except BaseException:
                # p3-b F5. Every move-back below goes through
                # _move_back_no_nest, never a bare shutil.move: sroot is a
                # DIRECTORY, and shutil.move lands its source INSIDE an
                # existing directory target rather than replacing it — if
                # something recreated sroot after the set-aside above (a
                # compactor not yet stopped writes a fact and
                # atomic_write_json mkdirs the store's parent again), the
                # old store used to land nested at
                # compactor/compactor.pre-restore-<stamp>/ instead of back
                # at the live path (SP\p3-b\rollback.py). Track whether
                # every step actually succeeded — the "nothing is left at a
                # mixed generation" claim below must not print unless it is
                # true.
                rollback_errors: list[str] = []
                if store_aside is not None:
                    try:
                        _move_back_no_nest(
                            store_aside, sroot, stamp=stamp,
                            label="the compactor store",
                        )
                    except OSError as e:
                        rollback_errors.append(
                            f"could not move the store back from "
                            f"{store_aside} to {sroot} ({e})"
                        )
                        logger.error(
                            f"restore rollback: {rollback_errors[-1]} — "
                            f"the pre-restore store is still at "
                            f"{store_aside}, move it back by hand"
                        )
                # v3.1.9 round 2 (found while testing finding 8, not itself
                # one of the four named findings — the same "clean up
                # staged debris on failure" doctrine this whole function
                # already uses everywhere else, missed here). os.replace
                # (store_incoming, sroot) failing leaves its SOURCE,
                # store_incoming, untouched on disk regardless of whether
                # sroot itself had something to restore — a multi-gigabyte
                # `.incoming` directory nobody is told about and nothing
                # lists or prunes (the same *shape* as hostile317-c Attack 3
                # LOW's .tmp files, a different, previously-undiscovered
                # instance).
                shutil.rmtree(store_incoming, ignore_errors=True)
                # v3.1.9 round 2 (finding 8): the store swap failed AFTER
                # the database swap already landed (have_db and db_aside is
                # only set in that case). Roll the database back too, so
                # the live state returns to the FULL pre-restore generation
                # instead of stranding a mix of (NEW db, OLD store).
                #
                # p3-b F5: the database's SIDECARS (`moved` — its -journal /
                # -wal, quarantined a few lines above have_db) were never
                # part of this rollback before. Putting the old db back
                # WITHOUT the hot journal it had is exactly the corruption
                # A3-2 exists to prevent, rebuilt inside this handler.
                #
                # p4-b G2: THE DATABASE GOES BACK FIRST, THEN ITS SIDECARS —
                # the reverse of what shipped, and the comment that used to
                # sit here ("order does not matter for correctness ...
                # nothing reopens the db until this function returns") was
                # wrong. Moving the sidecars back first lands the OLD
                # -journal beside `target` WHILE `target` STILL HOLDS THE
                # NEW DATABASE (db_aside has not moved yet) — a SIGKILL in
                # that window (a RunPod redeploy, an OOM kill) leaves a NEW
                # db + OLD journal pair that LOOKS matched. The next process
                # to open it (OpenWebUI on WEBUI_DB_LOCAL=false, where
                # nothing checks a marker before that open — G3) applies the
                # OLD journal's pages to the NEW file: proof, SP\p4-b\
                # rbkill.py — after that open `database disk image is
                # malformed`, and the pre-restore OLD database sitting in
                # forensics WITHOUT its journal (moved away out from under
                # it) reads 1,480 committed-looking rows from a transaction
                # that never committed (integrity_check still says "ok").
                # An EIO on the db move-back (no kill needed) reaches the
                # same state: the log used to tell the operator to move the
                # db back by hand without warning that a foreign journal
                # now sits beside the live file. Moving the db back FIRST
                # means the only db ever beside a foreign-looking journal
                # is the one about to receive ITS OWN journal a moment
                # later, and a kill between the two leaves OLD db + OLD
                # journal not yet reunited — recoverable by hand, not
                # silently corrupted. This is the same order the db-swap
                # handler above (the one a few lines up, for a failure
                # DURING the database's own swap) already used.
                if have_db and db_aside is not None:
                    try:
                        _move_back_no_nest(
                            db_aside, target, stamp=stamp, label="webui.db",
                        )
                    except OSError as e:
                        rollback_errors.append(
                            f"could not move webui.db back from {db_aside} "
                            f"to {target} ({e})"
                        )
                        logger.error(
                            f"restore rollback: {rollback_errors[-1]} — "
                            f"the pre-restore database is still at "
                            f"{db_aside}, move it back by hand"
                        )
                    else:
                        # Only once the OLD db is actually back at `target`
                        # do its sidecars get restored beside it — never
                        # beside whatever was there before (the NEW db, or
                        # nothing). If the db move-back above failed, the
                        # sidecars stay in quarantine too: reuniting a hot
                        # journal with the WRONG database is worse than
                        # leaving both quarantined for a human to sort out.
                        for side, aside in reversed(moved):
                            try:
                                _move_back_no_nest(
                                    aside, side, stamp=stamp,
                                    label=f"{side.name} (database sidecar)",
                                )
                            except OSError as e:
                                rollback_errors.append(
                                    f"could not put {aside} back as "
                                    f"{side.name} ({e})"
                                )
                                logger.error(
                                    f"restore rollback: {rollback_errors[-1]} "
                                    f"— move it back by hand BEFORE anything "
                                    f"opens {target.name}"
                                )
                    if rollback_errors:
                        logger.error(
                            f"the store swap failed after the database swap "
                            f"already landed, and the rollback did NOT fully "
                            f"succeed ({'; '.join(rollback_errors)}) — the "
                            f"live paths may be at a MIXED generation; check "
                            f"{target} and {sroot} by hand before restarting "
                            f"anything that reads them"
                        )
                    else:
                        logger.error(
                            f"the store swap failed after the database swap "
                            f"already landed; rolled the database back to "
                            f"its pre-restore state too, journal/-wal "
                            f"included ({target}), so nothing is left at a "
                            f"mixed generation"
                        )
                # p4-b G4: same reasoning as the db-swap handler above — a
                # FULLY successful rollback (rollback_errors == [], covering
                # both the store-only and the store+db cases: this list is
                # shared across both branches above) puts the live paths
                # back to exactly their pre-restore state, so the marker has
                # nothing left to protect against. Cleared here, once, after
                # both possible rollback branches (store-only; store+db)
                # have had their chance to append to rollback_errors —
                # never inside either branch individually, or a store-only
                # rollback with have_db False would clear the marker before
                # the (skipped) db branch even ran.
                if not rollback_errors:
                    import webuidb
                    webuidb.remove_restore_marker(stamp)
                raise
            _fsync_dir(sroot.parent)
            if store_aside is not None:
                logger.warning(
                    f"the compactor store in place before this restore is "
                    f"kept at {store_aside} — nothing deleted it, and nothing "
                    f"will; remove it by hand once the restore is confirmed"
                )
            restored.append("compactor")

        # v3.1.9 round 2 (finding 10). Every service that has the old
        # generation open, cached, or watching these paths needs a restart
        # to see the restored one at all — the same list
        # _require_no_active_writer already tells an operator to STOP before
        # running this. Named here too, so the happy path does not require
        # already knowing that list from a different error message.
        #
        # p3-b F8: `webuidb-sync` is named ONLY on a pod whose live database
        # IS webuidb.LOCAL_DB (WEBUI_DB_LOCAL=true) — the one placement
        # where that daemon's job (publish LOCAL_DB -> SNAPSHOT_DB) means
        # anything. On a WEBUI_DB_LOCAL=false pod the live database is
        # SNAPSHOT_DB itself; starting webuidb-sync there runs its first
        # cycle over whatever stale /var/lib/openwebui/webui.db survived
        # from before the flag flipped (OPERATIONS.md step 9's own
        # documented state) and publishes IT over the restore this command
        # just landed — the generation/row-loss guards compare against the
        # file being replaced, so an older, stale local file passes them
        # both and os.replace swaps the inode out from under the OpenWebUI
        # process this same restart line just started (SP\p3-b\gate.py).
        # Import lazily — same reasoning as _quarantine_dir's own `import
        # webuidb`.
        import webuidb
        services = ["openwebui", "compactor", "backup"]
        if webuidb.live_webui_db() == webuidb.LOCAL_DB:
            services.append("webuidb-sync")
        restart_cmd = "supervisorctl start " + " ".join(services)
        logger.info(
            f"restored {restored} from {archive_path.name} — restart the "
            f"services that read it: `{restart_cmd}`"
        )
        # p3-b F4/F9: every live move this run planned has now landed —
        # the ONLY point in this function that removes the marker.
        webuidb.remove_restore_marker(stamp)
        return {
            "ok": True,
            "restored": restored,
            "archive": archive_path.name,
            "restart": restart_cmd,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _restore_stamp() -> str:
    """Millisecond resolution, matching webuidb._stamp().

    Whole seconds collided: two back-to-back restores produced one
    `.pre-restore-<stamp>` name, and on POSIX a rename onto an existing name
    is a SILENT REPLACE — so the second destroyed the first set-aside, the
    older journal, the one more likely to hold the transaction an operator is
    chasing. webuidb._stamp() carries a comment saying exactly this happened
    there; this file copied its three suffixes and its reasoning and not its
    stamp. One time read, so the second and the millisecond cannot disagree.
    """
    now = time.time()
    return (
        time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
        + f"-{int(now * 1000) % 1000:03d}"
    )


def _free_name(p: Path) -> Path:
    """`p`, or `p-1`, `p-2`... — whichever does not exist yet.

    Millisecond stamps make a collision rare; this makes it impossible. A set-
    aside exists because something already went wrong, and a rename that can
    silently replace one is not a set-aside.
    """
    if not p.exists():
        return p
    n = 1
    while p.with_name(f"{p.name}-{n}").exists():
        n += 1
    return p.with_name(f"{p.name}-{n}")


def _move_back_no_nest(src: Path, dst: Path, *, stamp: str, label: str) -> None:
    """Move `src` back onto `dst` — a restore-rollback step — WITHOUT ever
    landing `src` inside `dst`.

    p3-b F5. `shutil.move(src, dst)` moves `src` INTO `dst` whenever `dst`
    already exists AND is a directory — that is shutil.move's own
    documented behaviour, not a bug in it, but it is the wrong behaviour
    for a rollback: if something recreated `dst` after this restore set it
    aside (e.g. a compactor process not yet stopped writes a fact and
    atomic_write_json mkdirs the store's parent again), the pre-restore
    original silently lands NESTED one level down
    (`compactor/compactor.pre-restore-<stamp>/`) instead of back at the
    live path — proof: SP\\p3-b\\rollback.py, `live_store_top:
    ["compactor.pre-restore-...", "facts"]`. The live store is then
    whatever thin tree got recreated, and everything else (persona,
    summaries, the rest of the facts) is invisible to the compactor.

    So: refuse to land on an occupied path. If `dst` exists, rename IT
    aside first (never delete it — it may be the only record of whatever
    got recreated), loudly, then do the move. `_free_name` makes the
    rename collision-proof the same way `_quarantine_aside` already is.
    """
    if dst.exists():
        obstacle = _free_name(dst.with_name(f"{dst.name}.failed-{stamp}"))
        logger.error(
            f"rolling back {label}: {dst} already exists (something "
            f"recreated it after this restore set it aside) — moved IT to "
            f"{obstacle} rather than nesting the pre-restore original "
            f"inside it; nothing was deleted, check {obstacle} by hand"
        )
        os.rename(dst, obstacle)
    shutil.move(str(src), str(dst))


def _quarantine_dir() -> Path:
    """webuidb.QUARANTINE, imported lazily.

    Not at module scope: backup.py is loaded by the CLI and the supervisord
    sidecar and must not drag in webuidb's own imports (which pull the sync
    daemon's config surface) just to learn one path — the same reasoning
    SIDECARS above gives for duplicating its three strings rather than
    importing it. This one constant is worth importing rather than
    duplicating: unlike SIDECARS it is operator-configurable
    (WEBUI_DB_QUARANTINE), and a copied default would silently drift from a
    changed one.
    """
    import webuidb
    return webuidb.QUARANTINE


def _quarantine_aside(path: Path, stamp: str) -> Path:
    """Move `path` (a file or a directory) to webuidb.QUARANTINE under the
    same `<name>.pre-restore-<stamp>` convention this module already used
    for a plain sibling rename, falling back to a sibling of `path` only if
    QUARANTINE itself cannot be used.

    v3.1.9 (hostile2-backup A3-10). A sibling of the RESTORE TARGET is a
    sibling of /var/lib/openwebui under the shipped default — the container
    OVERLAY, which a pod recreate destroys, and a restore runs during an
    incident, after which what follows on RunPod is a redeploy.
    webuidb.QUARANTINE (/data/forensics) is the directory this project
    already uses so a set-aside survives the incident it was set aside FOR —
    webuidb._set_aside documents exactly the same reasoning for the sync
    daemon's own set-asides, and this function copies its destination, not
    just its naming convention. shutil.move (not Path.rename / os.replace)
    because QUARANTINE is not guaranteed to share a filesystem with the
    restore target.
    """
    name = f"{path.name}.pre-restore-{stamp}"
    try:
        qdir = _quarantine_dir()
        qdir.mkdir(parents=True, exist_ok=True)
        dest = _free_name(qdir / name)
        shutil.move(str(path), str(dest))
        return dest
    except OSError as e:
        # QUARANTINE unusable — e.g. /data itself is the volume in trouble,
        # which is plausible: it is the same volume the restore is staging
        # onto. Falling back to a sibling rather than refusing the restore
        # outright keeps the destructive path available when it is needed
        # most; it is still never deleted and still findable, just not
        # survivable across a pod recreate from there. Logged loudly, not
        # silently — that silence is A3-11's other half.
        logger.error(
            f"could not move {path.name} to {_quarantine_dir()} "
            f"({type(e).__name__}: {e}); setting it aside beside {path} "
            f"instead — it will NOT survive a pod recreate from there"
        )
        dest = _free_name(path.with_name(name))
        if path.is_dir():
            shutil.move(str(path), str(dest))
        else:
            path.rename(dest)
        return dest


def list_pre_restore_asides(quarantine_dir: Path | None = None) -> list[dict]:
    """Every set-aside a restore has left behind, newest first:
    [{name, path, size_bytes, mtime}]. Directories (a set-aside compactor
    store) report the size of their whole tree.

    v3.1.9 (hostile2-backup A3-11). Before this fix nothing in the tree ever
    listed, reported or pruned a `.pre-restore-*` entry — one per restored
    sidecar/store, forever, owned by nobody, and each one counts toward
    _dir_size and so toward the space MIN_FREE_MB eventually refuses a
    *backup* over. This does not delete anything; it is the visibility the
    A3-10 move to QUARANTINE was missing without it.

    p3-b F4/F9 (hostile-D confirmed this on real data): this used to list
    ONLY `webuidb.QUARANTINE`'s `*.pre-restore-*` entries — the ORIGINALS a
    restore set aside. It never listed the STAGED copies restore_backup
    builds BESIDE the live targets before any live move (`db_tmp` /
    `store_incoming`, named `<target>.restore-<stamp>` /
    `<sroot>.incoming-<stamp>`), which a kill leaves behind just as
    permanently — or this fix's own `<path>.failed-<stamp>` (an obstacle
    _move_back_no_nest renamed aside during a rollback, F5). All three now
    show up in one place, `quarantine_dir` continues to override ONLY the
    quarantine half (existing test contract), and a failure to resolve the
    live target paths (an unset/misconfigured gate) degrades to "quarantine
    only" rather than raising — this is a listing, and a listing that
    cannot enumerate every location must still show what it can.
    """
    out: list[dict] = []

    def _add(f: Path) -> None:
        try:
            st = f.stat()
            size = st.st_size if f.is_file() else _tree_bytes(f)
        except OSError:
            return
        out.append({
            "name": f.name, "path": str(f), "size_bytes": size,
            "mtime": int(st.st_mtime),
        })

    d = quarantine_dir or _quarantine_dir()
    if d.exists():
        for f in d.glob("*.pre-restore-*"):
            _add(f)

    try:
        target = live_webui_db()
        for pattern in (f"{target.name}.restore-*", f"{target.name}.failed-*"):
            for f in target.parent.glob(pattern):
                _add(f)
    except Exception as e:
        logger.debug(f"list_pre_restore_asides: could not scan beside the live db ({e})")

    try:
        sroot = STORAGE_ROOT
        for pattern in (f"{sroot.name}.incoming-*", f"{sroot.name}.failed-*"):
            for f in sroot.parent.glob(pattern):
                _add(f)
    except Exception as e:
        logger.debug(f"list_pre_restore_asides: could not scan beside the live store ({e})")

    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def prune_pre_restore_asides(
    older_than_days: float = 30, quarantine_dir: Path | None = None
) -> list[str]:
    """Delete `.pre-restore-*` set-asides older than `older_than_days`.
    Returns names removed.

    v3.1.9 (hostile2-backup A3-11). Never called automatically — unlike
    prune_old_backups (which prunes REDUNDANT good copies), a pre-restore
    set-aside exists because a human already decided to overwrite something,
    and only a human recovering from a bad restore knows whether it is still
    needed. This is the tool for making that call explicit, not a background
    sweep that could delete the one copy of what a restore was about to
    replace before anyone confirmed the restore was right.
    """
    removed: list[str] = []
    for entry in list_pre_restore_asides(quarantine_dir):
        age_days = (time.time() - entry["mtime"]) / 86400.0
        if age_days < older_than_days:
            continue
        p = Path(entry["path"])
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(entry["name"])
        except OSError as e:
            logger.warning(f"could not prune {entry['name']}: {e}")
    return removed


# SQLite's own locking-byte offsets (sqlite3's os_unix.c / os_win.c), the
# same on every platform its default VFS runs on, independent of page size
# or schema: PENDING_BYTE = 0x40000000, RESERVED_BYTE = PENDING_BYTE + 1.
# A write transaction between BEGIN IMMEDIATE (or the first write of a plain
# BEGIN) and COMMIT/ROLLBACK holds an exclusive fcntl() record lock at
# RESERVED_BYTE. See _probe_reserved_lock.
_SQLITE_PENDING_BYTE = 0x40000000
_SQLITE_RESERVED_BYTE = _SQLITE_PENDING_BYTE + 1


def _probe_reserved_lock(target: Path) -> bool:
    """True if another process currently holds SQLite's RESERVED lock on
    `target` — a write transaction actively in progress right now.

    v3.1.9 round 2 (hostile2-backup, finding 7). _require_no_active_writer's
    own `BEGIN IMMEDIATE` probe stands down whenever a sidecar already
    exists beside `target` — precisely the incident-in-progress state a
    restore is most dangerous against, per that function's own docstring,
    because opening ANY real SQLite connection there can replay a hot
    journal as a side effect of the very check meant to be advisory (proven
    while building A3-6b, see that docstring).

    This probes the SAME lock a `BEGIN IMMEDIATE` takes — a raw POSIX
    fcntl() byte-range record lock at SQLite's own RESERVED_BYTE offset —
    but opens the file with a plain `os.open()`, never `sqlite3.connect()`.
    A raw file descriptor never invokes SQLite's pager/VFS layer, which is
    the ONLY thing that ever performs hot-journal recovery; a byte-range
    lock request touches no file content at all. This is therefore safe to
    run in EXACTLY the state the existing probe must stand down for.

    NOT a full replacement for `supervisorctl stop` (OPERATIONS.md, already
    documented before a restore): an idle reader holding only a SHARED
    lock, or a writer between BEGIN and its first actual write, holds no
    RESERVED lock yet and is invisible here too — this narrows the gap, it
    does not close it. It catches the specific, common case that matters
    most: a write transaction genuinely in flight while a journal from it
    already sits on disk, which the existing probe could not see at all in
    this state.

    Fails OPEN (returns False, "no lock detected") on any platform without
    `fcntl` (Windows — a real target only for local dev/tests here;
    production is Linux) or if the file cannot even be opened for this
    probe, since this is a best-effort ADD-ON to the real guarantee
    (`supervisorctl stop`), not a replacement for it.
    """
    try:
        import fcntl
    except ImportError:
        return False
    try:
        fd = os.open(str(target), os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, _SQLITE_RESERVED_BYTE, 0)
        except OSError:
            return True  # someone else holds the RESERVED byte right now
        else:
            # We just acquired it ourselves proving no one else held it —
            # release immediately, this is a probe, not a lock we want.
            fcntl.lockf(fd, fcntl.LOCK_UN, 1, _SQLITE_RESERVED_BYTE, 0)
            return False
    finally:
        os.close(fd)


def _require_no_active_writer(target: Path) -> None:
    """Refuse to restore if something currently holds a WRITE lock on
    `target`.

    v3.1.9 (hostile2-backup A3-6b). The atomicity fix (A3-2/A3-3) changed
    the restore from "overwrite the bytes of the live inode" to "swap in a
    new inode" (os.replace). A process that already has `target` open keeps
    writing to the OLD, now-unlinked inode and never sees the restore — and
    its next rollback journal is named from the PATH, so it lands beside the
    NEW file while describing the OLD file's page layout. That journal,
    applied to the restored database on its next open, is precisely the
    corruption this whole function exists to prevent, arriving from the
    other end.

    A `BEGIN IMMEDIATE` with no busy wait is not a perfect test — an idle
    reader that is not mid-write holds no lock SQLite can see, so
    `supervisorctl stop` (which OPERATIONS.md already documents before a
    restore) is still the real guarantee. But it catches the case that
    matters most — something actively writing through the restore — for
    the cost of one connection, and it fails LOUD instead of failing silent
    hours later.

    Skipped entirely when a sidecar already sits beside `target`. That is
    deliberate, not a gap: opening ANY read-write connection is how SQLite
    performs hot-journal recovery, which DELETES the journal as a side
    effect of the very check meant to be advisory — proven while building
    this fix, where an earlier version of this probe consumed a live
    journal before restore_backup's own sidecar-preserving code ever saw
    it. A sidecar present is either a hot rollback journal (F3's
    stalled-recovery signature) or a transaction genuinely in progress
    right now, and there is no way to tell those apart from outside without
    that same side effect — so this check stands down and leaves the
    sidecar exactly as it is for the code below to move aside intact.
    `supervisorctl stop` before a restore (OPERATIONS.md) is what actually
    closes that narrower race.
    """
    if not target.is_file():
        return
    if any(target.with_name(target.name + suffix).is_file() for suffix in SIDECARS):
        # v3.1.9 round 2 (finding 7): this used to stand down completely
        # here — exactly the incident state (a hot journal beside the
        # target) a restore is most likely to be run to recover FROM, and
        # therefore the state where an active writer matters most. A raw
        # fcntl() lock probe (see _probe_reserved_lock) works precisely
        # where the BEGIN IMMEDIATE probe below cannot enter.
        if _probe_reserved_lock(target):
            raise RuntimeError(
                f"refusing to restore: {target} is held by an active "
                f"writer (a RESERVED lock is currently set) WHILE a "
                f"sidecar also sits beside it — a write transaction in "
                f"flight during exactly the incident a restore is most "
                f"dangerous to run against. Stop the writers first: "
                f"`supervisorctl stop openwebui compactor backup "
                f"webuidb-sync`, confirm the process has actually exited "
                f"(a stop is a SIGTERM, not a guarantee), then retry."
            )
        return
    try:
        con = sqlite3.connect(str(target), timeout=0)
        try:
            con.execute("BEGIN IMMEDIATE")
            con.execute("COMMIT")
        finally:
            con.close()
    except sqlite3.OperationalError as e:
        if "locked" not in str(e):
            return
        raise RuntimeError(
            f"refusing to restore: {target} is locked by an active writer "
            f"({e}). Stop the writers first: `supervisorctl stop openwebui "
            f"compactor backup webuidb-sync`, confirm the process has "
            f"actually exited (a stop is a SIGTERM, not a guarantee), then "
            f"retry."
        ) from e
    except sqlite3.Error:
        return


def _is_under(path: Path, root: Path) -> bool:
    """True if `path` resolves to `root` or somewhere below it."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


# v3.1.9 round 2 (finding 9). See _require_free_space's own docstring for
# why this exists and why it is OFF (0) by default.
COMPACTOR_DATA_VOLUME_QUOTA_MB = env_int("COMPACTOR_DATA_VOLUME_QUOTA_MB", 0)

# p3-b F11. The quota this guards is typed into the RunPod dashboard
# against the WHOLE volume mounted at /data — the Dockerfile puts
# HF_HOME=/data/models, LOG_DIR=/data/logs, backups at /data/backups and
# forensics at /data/forensics all on that same volume, beside
# DATA_DIR (/data/openwebui). Measuring DATA_DIR alone (the old
# behaviour) meant tens of GB of model weights never counted toward the
# quota at all — proof: SP\p3-b\vacuum_payload.py, a 50,000 MB quota with
# a real 61,463 MB on the volume (a 60 GiB models dir) still read
# "ALLOWED" because DATA_DIR itself only held 23 MB. Defaults to
# DATA_DIR's parent, which is /data under the shipped layout; overridable
# because DATA_DIR's own default can be, and the two must stay in step for
# this to mean anything.
COMPACTOR_DATA_VOLUME_ROOT = Path(
    os.environ.get("COMPACTOR_DATA_VOLUME_ROOT", str(DATA_DIR.parent))
)


def _require_free_space(needed: list[tuple[Path, int]]) -> None:
    """Refuse BEFORE staging if a target volume cannot hold its copy.

    create_backup — the non-destructive half — has had a free-space guard for
    releases; restore, the destructive one, had none, and ENOSPC mid-copy was
    the named trigger for both move-then-write losses above. Staging no longer
    touches anything live, so running out of space is now merely a failed
    restore rather than a lost journal or a lost store; this makes it a clear
    one, before a multi-gigabyte copy, instead of an OSError halfway through.

    Totals per volume: when the database and the store share one, both copies
    need room at once. 10% headroom plus MIN_FREE_MB, the same floor
    create_backup keeps.

    v3.1.9 round 2 (finding 9): THIS CHECK DOES NOT MEAN WHAT IT LOOKS LIKE
    IT MEANS ON THIS POD'S ACTUAL PRODUCTION STORAGE. `shutil.disk_usage()`
    calls `statvfs()`, and on MooseFS (WEBUI_DB_LOCAL=false, this pod's real
    config — see MEMORY.md) that reports the CLUSTER-WIDE figure — measured
    at ~217 TB free — not this pod's own RunPod volume quota. A pod whose
    OWN allotment is genuinely full still sees ~217 TB "free" right up until
    the write that fails with ENOSPC/EDQUOT anyway. This function cannot
    tell the difference, and nothing in the MooseFS FUSE mount this process
    can see exposes the real per-pod quota to ask instead.

    The only ADDITIONAL check available that is honest about what it can
    and cannot promise: `COMPACTOR_DATA_VOLUME_QUOTA_MB`, a number the
    OPERATOR types in (read off the RunPod dashboard), OFF (0) by default —
    most deployments have no quota concept, and a wrong guess here is worse
    than no check at all. When set, this walks `DATA_DIR`'s own actual
    on-disk usage (`_tree_bytes`, a real `du`, NOT `statvfs` — meaningful on
    MooseFS where `statvfs` is cluster-wide and a directory walk is not) and
    compares it plus what this restore is about to add against that
    configured ceiling.

    NOT implemented, considered and rejected: a PROBE WRITE
    (`os.posix_fallocate` for the needed size) would exercise a real
    reservation instead of trusting a human-typed number — but this dev
    environment has no live MooseFS mount to verify whether MooseFS's FUSE
    implementation actually HONOURS `posix_fallocate` or merely accepts and
    ignores the call (a documented failure mode of several FUSE
    filesystems), and shipping an ENOSPC guard that LOOKS like it checked
    something real and silently does not would be worse than stating the
    limitation plainly. Per the brief: implement only what is honest: not
    built.
    """
    by_dev: dict[int, tuple[Path, int]] = {}
    for where, nbytes in needed:
        probe = where
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            dev = probe.stat().st_dev
        except OSError:
            continue
        prev = by_dev.get(dev, (probe, 0))
        by_dev[dev] = (prev[0], prev[1] + nbytes)
    for probe, nbytes in by_dev.values():
        try:
            free = shutil.disk_usage(str(probe)).free
        except OSError:
            continue
        want = int(nbytes * 1.1) + MIN_FREE_MB * 1024 * 1024
        if free < want:
            raise RuntimeError(
                f"refusing to restore: {free / 1048576:.0f} MB free at {probe}, "
                f"and staging this archive there needs about "
                f"{want / 1048576:.0f} MB. Nothing has been touched."
            )

    if COMPACTOR_DATA_VOLUME_QUOTA_MB > 0 and COMPACTOR_DATA_VOLUME_ROOT.is_dir():
        # p3-b F11: measured against the VOLUME ROOT, not DATA_DIR alone —
        # see COMPACTOR_DATA_VOLUME_ROOT's own comment. `added` now also
        # counts anything `needed` puts under the volume root, which
        # includes the quarantine dir entries `restore_backup` adds for
        # the db/store being set ASIDE (same volume), not just the
        # staging destinations.
        added = sum(
            nbytes for where, nbytes in needed
            if _is_under(where, COMPACTOR_DATA_VOLUME_ROOT)
        )
        try:
            used_bytes = _tree_bytes(COMPACTOR_DATA_VOLUME_ROOT)
        except OSError as e:
            logger.warning(
                f"COMPACTOR_DATA_VOLUME_QUOTA_MB is set but "
                f"{COMPACTOR_DATA_VOLUME_ROOT} could not be walked to "
                f"measure current usage ({e}); quota check skipped, "
                f"statvfs-based check above still applies"
            )
        else:
            want_mb = (used_bytes + added) / 1048576
            if want_mb > COMPACTOR_DATA_VOLUME_QUOTA_MB:
                raise RuntimeError(
                    f"refusing to restore: COMPACTOR_DATA_VOLUME_QUOTA_MB="
                    f"{COMPACTOR_DATA_VOLUME_QUOTA_MB}, "
                    f"{COMPACTOR_DATA_VOLUME_ROOT} already holds about "
                    f"{used_bytes / 1048576:.0f} MB, and staging this "
                    f"archive there adds about {added / 1048576:.0f} MB "
                    f"more ({want_mb:.0f} MB total, over the configured "
                    f"quota). Nothing has been touched. (statvfs-based free "
                    f"space above this pod's own quota is NOT a reliable "
                    f"signal on MooseFS — see this function's docstring.)"
                )


def _fsync_file(p: Path) -> None:
    """Durable before it is renamed into place.

    memory.atomic_write_json fsyncs the file and its directory and says why:
    /data is MooseFS, where rename and fsync guarantees are weaker than local
    POSIX. The restore's `os.replace` was never fsynced at all, so a pod killed
    after the rename could come back with a name pointing at pages that never
    reached the disk.
    """
    with open(p, "rb") as f:
        os.fsync(f.fileno())


def _fsync_tree(root: Path) -> None:
    """Every file, then every directory, under `root`.

    copytree writes each file with copy2 and fsyncs nothing, so fsyncing
    only the top directory would make the rename durable and leave the
    contents it names in the page cache. A restore is rare and manual; the
    cost of a full pass is paid once, at the moment it matters.
    """
    for f in root.rglob("*"):
        if f.is_file():
            _fsync_file(f)
    for d in sorted((x for x in root.rglob("*") if x.is_dir()), reverse=True):
        _fsync_dir(d)
    _fsync_dir(root)


def _fsync_dir(d: Path) -> None:
    """Best-effort: a filesystem that refuses a directory fsync is not a
    reason to fail a restore that has already succeeded."""
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Daemon + CLI
# ---------------------------------------------------------------------------

def _newest_archive_age_s(backup_dir: Path | None = None) -> float | None:
    """Seconds since the newest archive was written, or None if there is
    none."""
    archives = list_backups(backup_dir)
    if not archives:
        return None
    return max(0.0, time.time() - archives[0]["mtime"])


def run_daemon(interval_hours: float | None = None) -> None:
    """Periodic loop for the supervisord sidecar. Backs up every
    `interval_hours`, forever. Each cycle is wrapped so a single failure
    doesn't kill the loop."""
    interval = (INTERVAL_HOURS if interval_hours is None else interval_hours) * 3600.0
    logger.info(
        f"backup daemon started: every {interval/3600:.1f}h → {BACKUP_DIR} "
        f"(keep {RETAIN_DAYS:.0f}d + {GFS_WEEKS}w GFS, never below "
        f"{max(MIN_KEEP, RETAIN)})"
    )
    first = True
    # p3-b F10. consecutive_failures backs off the RETRY, not run_once's
    # own alerting (run_once calls _alert_failure on every failed cycle
    # already, unchanged) — see the failure branch below for why that is
    # still enough to close the finding.
    consecutive_failures = 0
    while True:
        if first:
            first = False
            # A cycle used to fire the instant this process started, and every
            # cycle ended in a prune. A restart loop — pod recreate, redeploy,
            # OOM-kill — therefore ran a whole retention window's worth of
            # cycles in minutes and left nothing but copies of the current,
            # possibly damaged, state. Skipping the boot run when a recent
            # archive already exists breaks the loop at its source; the
            # retention floor in _keep_set is the backstop. (v3.1 F7.)
            age = _newest_archive_age_s()
            if age is not None and age < interval / 2:
                logger.info(
                    f"skipping the boot-time backup: newest archive is "
                    f"{age/60:.0f} min old, under half the "
                    f"{interval/3600:.1f}h interval"
                )
                # Sleep the REMAINDER of the interval, not a fresh one.
                # Sleeping `interval` restarted the 24h timer from boot, so
                # every restart pushed the next backup further out: measured
                # in the 08-28..08-30 bundle, one completed backup in 32.5h
                # across 6 boots, worst case ~36h RPO. Nightly restarts are
                # routine here, and the deploy itself is another one.
                time.sleep(max(60.0, interval - age))
                continue
        report = run_once()
        if not report["ok"]:
            # v3.1.9 (hostile317-c F3). This used to fall through to the
            # same `time.sleep(interval)` a SUCCESSFUL cycle takes — up to
            # 24h by default. A hot rollback journal or a transient
            # "database is locked" (both live, both self-healing) turned
            # into a day-long blackout on the one signal an operator has
            # that backups stopped. Retry soon instead, capped at the
            # configured interval so this can never make a SHORT interval
            # (tests, a tuned-down deployment) longer than it already is.
            #
            # p3-b F10: a FLAT RETRY_BACKOFF_S applied to every failure
            # alike, transient or persistent. run_daemon cannot tell "a hot
            # journal that clears itself in a minute" from "one unparseable
            # memory file main.py deliberately leaves in place forever" —
            # see _clear_all_memory's own docstring, "it cannot be safely
            # rewritten from an unknown state" — so a persistent failure
            # retried every 15 minutes just the same: ~96 full archive
            # builds onto /data a day (each one stages the WHOLE payload
            # uncompressed before failing), 96 alerts, for as long as the
            # file stays broken, while webui.db and every healthy
            # conversation went unbacked the entire time (SP\p3-b\
            # retry_loop.py). Backing off exponentially per consecutive
            # failure — still capped at `interval`, so a transient fault
            # keeps its fast first retry — costs a persistent one
            # progressively less without giving up on a transient one
            # sooner. Alerting is not throttled separately here: run_once
            # already calls _alert_failure on every failed cycle
            # (unchanged), and the backoff itself is what spaces those
            # alerts out over time as failures persist — 900s, 1800s,
            # 3600s, ... capped — rather than 96 evenly-spaced ones.
            consecutive_failures += 1
            backoff = min(
                interval, RETRY_BACKOFF_S * (2 ** (consecutive_failures - 1))
            )
            logger.error(
                f"backup cycle failed ({consecutive_failures} in a row): "
                f"{report['detail']}; retrying in {backoff / 60:.0f} min "
                f"instead of the full {interval / 3600:.1f}h interval"
            )
            time.sleep(backoff)
            continue
        consecutive_failures = 0
        time.sleep(interval)


def _fmt(report: dict) -> str:
    mark = "OK" if report.get("ok") else "FAIL"
    return f"[{mark}] {report.get('archive') or '-'}  {report.get('detail', '')}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Zion's Light AI data backup.")
    p.add_argument("--once", action="store_true", help="Run one backup cycle and exit.")
    p.add_argument("--daemon", action="store_true", help="Run forever on the configured interval.")
    p.add_argument("--list", action="store_true", help="List existing backups.")
    p.add_argument("--verify", metavar="ARCHIVE", help="Verify an existing archive.")
    p.add_argument("--restore", metavar="ARCHIVE", help="Restore from an archive (DESTRUCTIVE).")
    p.add_argument("--yes", action="store_true", help="Confirm a destructive --restore.")
    p.add_argument("--json", action="store_true", help="Machine-readable output.")
    p.add_argument("--list-pre-restore", action="store_true",
                    help="List set-asides a restore has left in quarantine.")
    p.add_argument("--prune-pre-restore", metavar="DAYS", type=float,
                    help="Delete quarantined pre-restore set-asides older than DAYS.")
    args = p.parse_args(argv)

    import logsetup
    logsetup.configure()  # honors COMPACTOR_LOG_FORMAT (text/json)

    # M-1 (review1-v3197-65ea196): sweep any orphaned local staging file
    # left by a previous interrupted cycle before this one can add its own
    # — see sweep_stale_local_staging's docstring. Runs for every
    # invocation (--once, --daemon, --verify, ...), which is "at boot" for
    # both the supervisord daemon and every manual/CLI call this process
    # ever makes.
    sweep_stale_local_staging()

    if args.list:
        archives = list_backups()
        print(json.dumps(archives, indent=2) if args.json else
              "\n".join(f"{a['name']}  {a['size_bytes']} B" for a in archives) or "(none)")
        return 0
    if args.list_pre_restore:
        asides = list_pre_restore_asides()
        print(json.dumps(asides, indent=2) if args.json else
              "\n".join(f"{a['name']}  {a['size_bytes']} B" for a in asides) or "(none)")
        return 0
    if args.prune_pre_restore is not None:
        removed = prune_pre_restore_asides(args.prune_pre_restore)
        print(json.dumps(removed) if args.json else
              "\n".join(removed) or "(nothing to prune)")
        return 0
    if args.verify:
        ok, detail = verify_backup(Path(args.verify))
        print(json.dumps({"ok": ok, "detail": detail}) if args.json else f"[{'OK' if ok else 'FAIL'}] {detail}")
        return 0 if ok else 1
    if args.restore:
        try:
            rep = restore_backup(Path(args.restore), confirm=args.yes)
            if args.json:
                print(json.dumps(rep))
            else:
                print(f"restored: {rep['restored']}")
                # v3.1.9 round 2 (finding 10): the CLI's own restore output
                # names what to restart, not just the log line — this is
                # what a human doing --json=false actually reads.
                if rep.get("restart"):
                    print(f"restart the services that read it: "
                          f"`{rep['restart']}`")
            return 0
        except Exception as e:
            print(f"restore failed: {e}", file=sys.stderr)
            return 1
    if args.daemon:
        run_daemon()
        return 0
    # default: --once
    rep = run_once()
    print(json.dumps(rep, indent=2) if args.json else _fmt(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
