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

logger = logging.getLogger("compactor.backup")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# What to back up. DATA_DIR is OpenWebUI's state root; webui.db lives there.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/openwebui"))
WEBUI_DB = Path(os.environ.get("COMPACTOR_BACKUP_WEBUI_DB", str(DATA_DIR / "webui.db")))
STORAGE_ROOT = Path(
    os.environ.get("COMPACTOR_STORAGE_ROOT", str(DATA_DIR / "compactor"))
)

# Where backups land. Default is a sibling dir on the same volume.
BACKUP_DIR = Path(os.environ.get("COMPACTOR_BACKUP_DIR", "/data/backups"))

# How many archives to keep. No longer a cap — since v3.1 F7 this is a
# *floor* on the number retained, one of several tiers in _keep_set. As a
# cap it was the mechanism of the loss: RETAIN=7 with a prune at the end of
# every cycle and a cycle at every process start meant seven container
# restarts left seven copies of the damaged state and nothing older.
RETAIN = int(os.environ.get("COMPACTOR_BACKUP_RETAIN", "7") or 7)

# Retention tiers (v3.1 F7 / D9). Keep everything younger than RETAIN_DAYS,
# plus one archive per UTC day inside that window, plus one per ISO week for
# GFS_WEEKS. Anything no tier claims is prunable.
RETAIN_DAYS = float(os.environ.get("COMPACTOR_BACKUP_RETAIN_DAYS", "14") or 14)
GFS_WEEKS = int(os.environ.get("COMPACTOR_BACKUP_GFS_WEEKS", "8") or 8)

# The hard floor. Never leave fewer than this many archives on disk, whatever
# their age and whatever the tiers say. Floored at 3 in code rather than in
# config: a typo in an env var must not be able to empty the backup
# directory. This is the last line between a bad cycle and total loss.
MIN_KEEP = max(3, int(os.environ.get("COMPACTOR_BACKUP_MIN_KEEP", "3") or 3))

# Refuse to publish an archive whose payload is under this fraction of the
# previous one's. A store that lost half its bytes between two cycles is an
# unmounted volume, not a user deleting things. (v3.1 F2.)
MIN_PAYLOAD_RATIO = float(
    os.environ.get("COMPACTOR_BACKUP_MIN_PAYLOAD_RATIO", "0.5") or 0.5
)

# Daemon cadence.
INTERVAL_HOURS = float(os.environ.get("COMPACTOR_BACKUP_INTERVAL_HOURS", "24") or 24)

# Refuse to back up if the target volume has less than this much free space.
# Prevents the backup process from being the thing that fills the disk.
MIN_FREE_MB = int(os.environ.get("COMPACTOR_BACKUP_MIN_FREE_MB", "500") or 500)

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


def _snapshot_sqlite(src: Path, dest: Path) -> bool:
    """Consistent online snapshot of a (possibly live) SQLite db via the
    backup API. Returns True if a snapshot was written, False if the source
    doesn't exist. Raises on a real failure."""
    if not src.is_file():
        return False
    # Open read-only-ish; the backup API handles WAL + concurrent writers.
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


def _tree_bytes(p: Path) -> int:
    """Total bytes of every regular file under `p` (or of `p` itself)."""
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


def _census(store: Path) -> dict:
    """Per-conversation fact / summary / episodic counts.

    Computed from the *staged or extracted* tree, never from the live store,
    so the manifest describes what is actually inside the archive and
    verify_backup can recompute the identical numbers and contradict it.

    Per-conversation rather than a total: a total hides one conversation
    emptying while another grows, and one conversation is the whole product
    here. (v3.1 F2.)
    """
    census: dict[str, dict] = {}

    def slot(conv_id: str) -> dict:
        return census.setdefault(
            conv_id, {"facts": 0, "summaries": 0, "episodic": 0}
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
                slot(f.stem)["facts"] = len(data["facts"])

    summaries_dir = store / _SUMMARIES_SUBDIR
    if summaries_dir.is_dir():
        for f in sorted(summaries_dir.glob("*.json")):
            if "." in f.stem:
                continue
            data = _read_json(f)
            if not isinstance(data, dict):
                continue
            n = 0
            for tier in ("l1", "l2"):
                if isinstance(data.get(tier), list):
                    n += len(data[tier])
            if isinstance(data.get("l3"), dict):
                n += 1
            slot(f.stem)["summaries"] = n

    episodic = _episodic_counts(store / _CHROMA_SUBDIR / _CHROMA_DB_NAME)
    if episodic:
        for conv_id, n in episodic.items():
            slot(conv_id)["episodic"] = n

    return census


def _census_shortfalls(expected: dict, actual: dict) -> list[str]:
    """Conversations where `actual` holds fewer of a layer than `expected`.

    One-directional on purpose. More is fine — the live store grows between
    two archives, and growth is never the failure. Less is the entire signal
    this module exists to catch.
    """
    out: list[str] = []
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return out
    for conv_id in sorted(expected):
        want = expected.get(conv_id) or {}
        have = actual.get(conv_id) or {}
        if not isinstance(want, dict):
            continue
        for layer in ("facts", "summaries", "episodic"):
            w = int(want.get(layer) or 0)
            h = int(have.get(layer) or 0) if isinstance(have, dict) else 0
            if h < w:
                out.append(f"{conv_id}.{layer} {w}->{h}")
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

def create_backup(backup_dir: Path | None = None) -> Path:
    """Build a verified-elsewhere archive of webui.db + the compactor store.

    Writes to a `.partial` temp file; the caller (run_once) verifies it and
    only then is it published via os.replace. Returns the temp path.

    Raises RuntimeError if the min-free guard trips, or if the compactor
    store is missing (v3.1 F2 — see the guard at step 2).
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
        if _snapshot_sqlite(WEBUI_DB, db_dest):
            manifest["sources"]["webui.db"] = {
                "present": True, "bytes": db_dest.stat().st_size,
            }
        else:
            manifest["sources"]["webui.db"] = {"present": False}
            logger.warning(f"webui.db not found at {WEBUI_DB} — backing up memory only")

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
            chroma_present = _snapshot_sqlite(chroma_src, chroma_dest)
        else:
            logger.warning(
                f"episodic store {chroma_src} not found — this archive carries "
                f"facts, summaries and personas but no embedded exchanges"
            )
        n_files = sum(1 for f in store_dest.rglob("*") if f.is_file())
        n_json = sum(1 for _ in store_dest.rglob("*.json"))
        manifest["sources"]["compactor"] = {
            "present": True,
            "files": n_files,
            # The count verify_backup asserts against. Deliberately not
            # `files`: that includes chroma.sqlite3 and any binary index
            # files, so comparing a parsed-JSON count to it would fail every
            # archive that has an episodic store.
            "json_files": n_json,
            "chroma_sqlite": chroma_present,
            "conversations": _census(store_dest),
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


def run_once(backup_dir: Path | None = None) -> dict:
    """One full backup cycle. Returns a structured report. Never raises —
    failures are reported, not thrown, so the daemon keeps running.

    **Nothing is pruned unless the cycle is fully clean.** A failure, a
    refused publish, or a census that went backwards all return before
    prune_old_backups. (v3.1 F2/F7.)
    """
    d = backup_dir or BACKUP_DIR
    t0 = time.monotonic()
    report: dict = {"ok": False, "archive": None, "verified": False, "detail": ""}
    partial: Path | None = None
    # Read the baseline BEFORE creating the new archive, or the new one is
    # its own baseline and every comparison below is vacuous — the same
    # mistake verify_backup made with the manifest.
    existing = list_backups(d)
    prev_entry = existing[0] if existing else None
    prev_manifest = read_manifest(Path(prev_entry["path"])) if prev_entry else None
    try:
        partial = create_backup(d)
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
            losses = _census_shortfalls(prev_census, new_census)
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
    confirm: bool = False,
) -> dict:
    """Restore an archive over the live data locations. DESTRUCTIVE — it
    overwrites webui.db and the compactor store. Requires confirm=True.

    Verifies the archive first (won't restore an unusable backup), then
    extracts to scratch and moves the pieces into place. Returns a report.
    """
    if not confirm:
        raise RuntimeError("restore is destructive; pass confirm=True (CLI: --yes)")
    ddir = data_dir or DATA_DIR
    sroot = storage_root or STORAGE_ROOT

    ok, detail = verify_backup(archive_path)
    if not ok:
        raise RuntimeError(f"refusing to restore an unverifiable archive: {detail}")

    scratch = Path(tempfile.mkdtemp(prefix="zions-restore-"))
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(scratch, filter="data")  # path-traversal safe
        restored: list[str] = []
        # webui.db
        src_db = scratch / "webui.db"
        if src_db.is_file():
            ddir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_db, ddir / "webui.db")
            restored.append("webui.db")
        # compactor store — replace wholesale
        src_store = scratch / "compactor"
        if src_store.is_dir():
            if sroot.exists():
                shutil.rmtree(sroot)
            shutil.copytree(src_store, sroot)
            restored.append("compactor")
        logger.info(f"restored {restored} from {archive_path.name}")
        return {"ok": True, "restored": restored, "archive": archive_path.name}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


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
                time.sleep(interval)
                continue
        report = run_once()
        if not report["ok"]:
            logger.error(f"backup cycle failed: {report['detail']}")
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
    args = p.parse_args(argv)

    import logsetup
    logsetup.configure()  # honors COMPACTOR_LOG_FORMAT (text/json)

    if args.list:
        archives = list_backups()
        print(json.dumps(archives, indent=2) if args.json else
              "\n".join(f"{a['name']}  {a['size_bytes']} B" for a in archives) or "(none)")
        return 0
    if args.verify:
        ok, detail = verify_backup(Path(args.verify))
        print(json.dumps({"ok": ok, "detail": detail}) if args.json else f"[{'OK' if ok else 'FAIL'}] {detail}")
        return 0 if ok else 1
    if args.restore:
        try:
            rep = restore_backup(Path(args.restore), confirm=args.yes)
            print(json.dumps(rep) if args.json else f"restored: {rep['restored']}")
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
