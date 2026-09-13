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
      3. the gate, compared EXACTLY as entrypoint.sh compares it
         (`[ "${WEBUI_DB_LOCAL}" = "true" ]`, default true). Deliberately not a
         folded boolean: `True` means MooseFS to the shell, and a reader that
         disagreed with the writer about which file is live is the defect
         being fixed. The dialect itself is M9's problem, and the fix for it
         must change both sides at once.
    """
    explicit = os.environ.get("COMPACTOR_BACKUP_WEBUI_DB")
    if explicit:
        return Path(explicit)
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("sqlite:///"):
        # sqlite:////var/lib/x.db -> /var/lib/x.db; sqlite:///rel.db -> rel.db
        return Path(url[len("sqlite:///"):])
    if os.environ.get("WEBUI_DB_LOCAL", "true") == "true":
        return Path(os.environ.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db"))
    return Path(os.environ.get("WEBUI_SNAPSHOT_DB", str(DATA_DIR / "webui.db")))


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

    v3.1.9 (hostile317-c F1). The fields here used to be RAW COUNTS — active
    facts only, and len(l1)+len(l2)+bool(l3) chunks — and run_once's
    cross-cycle comparison flagged any decrease. That fires on the daemon's
    own designed steady
    state: facts.prune_facts(conv_id=...) moves evicted facts into the
    `.archive.json` sidecar rather than deleting them (active count drops,
    nothing is lost); dedup permanently merges duplicate facts (a smaller
    number of entries, same information); and every L1->L2 or L2->L3 rollup
    replaces N chunks with one chapter (chunk count drops, nothing is lost —
    that IS the hierarchy). Her real logs (hostile317-c) show every nightly
    cycle from 2026-08-31 through 2026-09-11 reading "memory shrank" and
    skipping the prune, on exactly this shape. So what is recorded now is
    what is actually unrecoverable if the archive did not carry it:

      * facts     — active + archived, UNIONED. Eviction moves an entry
        between the two files this reads; the union does not move with it.
        Dedup still reduces the union (two entries become one merged one),
        which is why the loss test is "went to zero", not "went down" — see
        _census_regressions.
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
    """
    census: dict[str, dict] = {}

    def slot(conv_id: str) -> dict:
        return census.setdefault(
            conv_id,
            {"facts": 0, "summary_turn": 0, "archived_chapters": 0, "episodic": 0},
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
                slot(f.stem)["facts"] += len(data["facts"])
        # Archived facts count toward the same conversation's total — see the
        # docstring above. Eviction (prune_facts with conv_id) moves entries
        # here; without this loop the census watched only the half of the
        # store the daemon's own steady state empties every day.
        for f in sorted(facts_dir.glob("*.archive.json")):
            conv_id = f.name[: -len(".archive.json")]
            data = _read_json(f)
            if isinstance(data, dict) and isinstance(data.get("facts"), list):
                slot(conv_id)["facts"] += len(data["facts"])

    summaries_dir = store / _SUMMARIES_SUBDIR
    if summaries_dir.is_dir():
        for f in sorted(summaries_dir.glob("*.json")):
            if "." in f.stem:
                continue
            data = _read_json(f)
            if not isinstance(data, dict):
                continue
            turn = data.get("last_summarized_turn")
            if isinstance(turn, int):
                slot(f.stem)["summary_turn"] = turn
        for f in sorted(summaries_dir.glob("*.archive.json")):
            conv_id = f.name[: -len(".archive.json")]
            data = _read_json(f)
            if isinstance(data, dict) and isinstance(data.get("chapters"), list):
                slot(conv_id)["archived_chapters"] = len(data["chapters"])

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
        for layer in ("facts", "summary_turn", "archived_chapters", "episodic"):
            w = int(want.get(layer) or 0)
            h = int(have.get(layer) or 0)
            if h < w:
                out.append(f"{conv_id}.{layer} {w}->{h}")
    return out


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
    So what is flagged is what is actually unrecoverable (see _census's
    docstring for what each field means):

      * facts — eviction and dedup both legitimately shrink the raw count
        (eviction moves it into the union _census now reads; dedup merges
        duplicates into fewer, denser entries). Neither can take the union
        to exactly zero while it held something — only a truly empty facts
        file AND an empty archive sidecar look like that. So this layer
        flags only w > 0 and h == 0: "the facts are gone", not "there are
        fewer of them today".
      * summary_turn / archived_chapters — both are designed to be
        monotonic (a high-water mark; an append-only sidecar), so ANY
        decrease is a regression: a rollup does not undo them, only real
        loss does.
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

        w_facts = int(want.get("facts") or 0)
        h_facts = int(have.get("facts") or 0)
        if w_facts > 0 and h_facts == 0:
            out.append(f"{conv_id}.facts {w_facts}->{h_facts} (emptied)")

        for layer in ("summary_turn", "archived_chapters", "episodic"):
            w = int(want.get(layer) or 0)
            h = int(have.get(layer) or 0)
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
        # Resolved per cycle, not at import: the daemon is the longest-lived
        # process on the pod, and a gate flipped under it was invisible
        # until restart (A3-9). See live_webui_db.
        live_db = live_webui_db()
        if _snapshot_sqlite(live_db, db_dest):
            manifest["sources"]["webui.db"] = {
                "present": True, "bytes": db_dest.stat().st_size,
            }
        else:
            manifest["sources"]["webui.db"] = {"present": False}
            logger.warning(f"webui.db not found at {live_db} — backing up memory only")

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
            # v3.1.9 (hostile317-c F1): _census_regressions, NOT
            # _census_shortfalls — this is a cross-CYCLE comparison, where
            # normal eviction/dedup/rollup legitimately shrinks the raw
            # numbers; _census_shortfalls's strict "any decrease" rule is
            # for verify_backup's within-one-archive integrity check only.
            losses = _census_regressions(prev_census, new_census)
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
        if have_store:
            needed.append((sroot.parent, _tree_bytes(src_store)))
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

        if have_db:
            # SIDECARS TRAVEL WITH THE DATABASE. A -journal or -wal beside the
            # target belongs to the database being REPLACED, and SQLite applies
            # it to whatever file carries that name on the next open. Renamed,
            # never deleted — and now only once the replacement is certain, and
            # put BACK if the replace itself fails, so no path through here
            # leaves the original without its journal.
            moved: list[tuple[Path, Path]] = []
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
                for side, aside in reversed(moved):
                    try:
                        shutil.move(str(aside), str(side))
                    except OSError as e:
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
                raise
            _fsync_dir(target.parent)
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
                if store_aside is not None:
                    shutil.move(str(store_aside), str(sroot))
                raise
            _fsync_dir(sroot.parent)
            if store_aside is not None:
                logger.warning(
                    f"the compactor store in place before this restore is "
                    f"kept at {store_aside} — nothing deleted it, and nothing "
                    f"will; remove it by hand once the restore is confirmed"
                )
            restored.append("compactor")

        logger.info(f"restored {restored} from {archive_path.name}")
        return {"ok": True, "restored": restored, "archive": archive_path.name}
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


def _tree_bytes(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


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
    """Every set-aside a restore has left in webuidb.QUARANTINE, newest
    first: [{name, path, size_bytes, mtime}]. Directories (a set-aside
    compactor store) report the size of their whole tree.

    v3.1.9 (hostile2-backup A3-11). Before this fix nothing in the tree ever
    listed, reported or pruned a `.pre-restore-*` entry — one per restored
    sidecar/store, forever, owned by nobody, and each one counts toward
    _dir_size and so toward the space MIN_FREE_MB eventually refuses a
    *backup* over. This does not delete anything; it is the visibility the
    A3-10 move to QUARANTINE was missing without it.
    """
    d = quarantine_dir or _quarantine_dir()
    if not d.exists():
        return []
    out: list[dict] = []
    for f in d.glob("*.pre-restore-*"):
        try:
            st = f.stat()
            size = st.st_size if f.is_file() else _tree_bytes(f)
        except OSError:
            continue
        out.append({
            "name": f.name, "path": str(f), "size_bytes": size,
            "mtime": int(st.st_mtime),
        })
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
            logger.error(
                f"backup cycle failed: {report['detail']}; retrying in "
                f"{min(interval, RETRY_BACKOFF_S) / 60:.0f} min instead of "
                f"the full {interval / 3600:.1f}h interval"
            )
            time.sleep(min(interval, RETRY_BACKOFF_S))
            continue
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
