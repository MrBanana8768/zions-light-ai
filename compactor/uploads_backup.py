"""
compactor.uploads_backup — a SEPARATE, incremental backup of OpenWebUI's
uploaded files (her photos), v3.1.9.8.

WHY A SEPARATE MODULE, NOT PART OF backup.py'S ARCHIVE (UPLOADS INCIDENT,
2026-09-26). The owner accidentally ran `gzip` over
`/data/openwebui/uploads` in place while trying to make her own copy: every
one of 338 files became `<name>.gz`, and OpenWebUI (whose `file` table
stores the literal path `/data/openwebui/uploads/<id>_<name>`) showed no
images at all until it was fixed by hand. backup.py's daily archive does
NOT cover this directory — uploads are write-once, ~270 MB and growing, and
tar-ing the whole tree nightly would make every archive's cost scale with
a folder that never shrinks and rarely changes. The owner asked for a
SEPARATE, incremental mirror instead, not folded into the main archive.

DESIGN, one paragraph: a content-addressed object store
(`objects/<sha256[:2]>/<sha256>`, each object a plain COPY — never a
hardlink, because an in-place edit of one twin would silently corrupt the
other) plus a JSON manifest mapping each upload's relative name to
{sha256, size, mtime, first_seen, last_seen, deleted_at}. Each run hashes
only files whose (size, mtime_ns, inode) changed since the last manifest —
that is the whole of "incremental" here; a write-once folder makes this
cheap. Deletes are kept (not purged) for
COMPACTOR_UPLOADS_BACKUP_KEEP_DELETED_DAYS so an accidental delete is
recoverable, and — the doctrine this module exists to enforce — a file
that changes CONTENT while already mirrored never overwrites the manifest's
recorded sha256; the new hash is recorded alongside it under `changed`,
the OLD (good) object is kept, and a WARNING is logged. Today's incident
replayed through this design looks exactly like what it is: 338 tracked
names vanish and 338 new, unrelated names appear in one run — which is
also precisely the shape the MASS-CHANGE GUARD below exists to catch
before it does anything destructive to the manifest.

WHAT THIS DOES NOT PROTECT AGAINST: loss of the /data volume itself. The
mirror lives on the same MooseFS mount as the uploads it protects — see
OPERATIONS.md's "Uploads backup" section. Pulling it off-pod is a
follow-up, not built here.

HOSTILE REVIEW ROUND 2 (post-merge, same day): an Opus review with two
working attack scripts (a TOCTOU race against the first draft's separate
hash-then-copy, and a synthetic replay of a legitimate 25%-of-uploads
deletion) found the first draft's guard was all-or-nothing (a real mass
DELETE froze the mirror with no way back except editing the manifest by
hand) and its hash/copy were two separate reads of the same live file (a
file finishing a write between them could be captured under the wrong
hash, permanently). Both are fixed at the point they apply below —
`_hash_and_copy_object` (one read pass), the settle window (SETTLE_S), the
mass-change guard's now-partial freeze (new files still mirrored; `accept
--yes` in the operator script recovers from a confirmed legitimate
deletion), and the cur_* fields that stop a single drift from re-counting
against the guard on every subsequent run. The round-2 review also moved
this module's own call site OUT of backup.run_once (see backup.py's
run_daemon instead) and off the event loop for its health snapshot (see
health.py's _gather_blocking) — neither is this module's own code, noted
here so the two files are read as one change.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from envcfg import env_float, env_int

logger = logging.getLogger("compactor.uploads_backup")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# OpenWebUI's own data root — same default backup.py uses (DATA_DIR), read
# independently here (not imported from backup.py) so this module has no
# import-time dependency on it; backup.py imports THIS module, lazily, at
# call time (the same "import webuidb inside the function" pattern it
# already uses elsewhere), and a two-way top-level import would be circular.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/openwebui"))
UPLOADS_DIR = Path(os.environ.get("COMPACTOR_UPLOADS_DIR", str(DATA_DIR / "uploads")))

# Where the mirror lives. A sibling of backup.py's own COMPACTOR_BACKUP_DIR
# default, but a SEPARATE tree (the owner's ask: not inside the main
# archive).
#
# v3.1.9.10: the default follows COMPACTOR_BACKUP_DIR (was the literal
# /data/backups/uploads), so a pod that redirects its backups but forgets this
# row cannot mirror uploads into, or expire files from, another pod's mirror.
UPLOADS_BACKUP_DIR = Path(
    os.environ.get(
        "COMPACTOR_UPLOADS_BACKUP_DIR",
        str(Path(os.environ.get("COMPACTOR_BACKUP_DIR") or "/data/backups") / "uploads"),
    )
)

MANIFEST_NAME = "manifest.json"
OBJECTS_DIRNAME = "objects"

# A deleted upload's object (and manifest record) is kept this long before
# being swept, in case the delete was a mistake.
KEEP_DELETED_DAYS = env_float("COMPACTOR_UPLOADS_BACKUP_KEEP_DELETED_DAYS", 90.0)

# Mass-change guard threshold, as a percentage of the previously-tracked
# (non-deleted) file count. Tripping this means: write NOTHING to the
# manifest's per-file records this run — only a `last_suspect_run` marker —
# and leave every existing object alone. This is deliberately conservative:
# a real mass edit (a directory move, a bulk rename) will also trip it, and
# that is the correct failure mode — an operator must look, not the mirror
# silently adopting whatever it sees.
MASS_CHANGE_PCT = env_float("COMPACTOR_UPLOADS_BACKUP_MASS_CHANGE_PCT", 20.0)

# Refuse a run (write nothing) if the backup volume has less than this much
# free space. Same doctrine and default as backup.MIN_FREE_MB.
MIN_FREE_MB = env_int("COMPACTOR_UPLOADS_BACKUP_MIN_FREE_MB", 500)

# Pacing: /data is a MooseFS network mount shared with live traffic. Copies
# are chunked (never one big read()/write() of a whole file) and a short
# sleep separates one file's copy from the next, so this backup does not
# turn into a burst that competes with her live request's own reads/writes.
CHUNK_BYTES = env_int("COMPACTOR_UPLOADS_BACKUP_CHUNK_BYTES", 4 * 1024 * 1024)
FILE_SLEEP_S = env_float("COMPACTOR_UPLOADS_BACKUP_FILE_SLEEP_S", 0.05)

# hostile-review round 2, finding 4: mid-run idle re-check. A single run_once
# call can process thousands of files; checking is_idle() only once, at the
# top, meant a run that STARTED idle kept copying for however long it took
# even after she started a live request mid-cycle. Re-poll after every this
# many files OR this many bytes copied (whichever comes first) and, if the
# pod has become busy, STOP CLEANLY: whatever is already written to the
# manifest stays (objects are add-only; a manifest save mid-run is always a
# consistent, smaller-than-full state), and the remaining files are picked
# up next cycle exactly as if this one had simply run out of files to find —
# no special "resume" bookkeeping needed.
IDLE_RECHECK_FILES = env_int("COMPACTOR_UPLOADS_BACKUP_RECHECK_FILES", 20)
IDLE_RECHECK_BYTES = env_int("COMPACTOR_UPLOADS_BACKUP_RECHECK_BYTES", 50 * 1024 * 1024)

# hostile-review round 2, finding 1: a file whose mtime is this recent is
# still possibly being written (an upload in progress, or an edit that has
# not finished). Such a file is skipped ENTIRELY this run — not hashed, not
# copied, not marked new or changed — so a half-written upload is never
# captured as its (wrong) partial content; it is picked up cleanly next run
# once it has stopped changing. This is a smoke detector, not a proof: a
# file that stays open for writes longer than this window is still exposed
# to the read-during-write race the hash-while-copy fix below narrows but
# cannot fully close.
SETTLE_S = env_float("COMPACTOR_UPLOADS_BACKUP_SETTLE_S", 300.0)

# hostile-review round 2, finding 9: on a network filesystem (MooseFS),
# statvfs reports the CLUSTER's free space, not this mount's actual quota —
# the existing free-space guard below is close to meaningless there. As a
# second, meaningful floor: refuse new copies once the mirror ITSELF (the
# objects/ store) would exceed this many MB. The mirror is a full copy of
# every distinct upload ever seen (plus drifted/changed copies), so it is
# expected to roughly track (and, over time with edits, exceed) the live
# uploads directory's own size — this is a backstop against runaway growth,
# not a tight budget.
MAX_MIRROR_MB = env_int("COMPACTOR_UPLOADS_BACKUP_MAX_MB", 2048)

# hostile-review round 4, finding 3: "stale" for health_snapshot — how long
# is too long since the last recorded run. The uploads mirror rides on
# backup.py's own daemon cadence (see run_daemon/_run_daemon_uploads_step),
# so its "how often should this have run" is backup.py's own interval, read
# independently here (same reasoning as DATA_DIR above: no import-time
# dependency on backup.py) rather than a separate, easy-to-forget knob.
BACKUP_INTERVAL_HOURS = env_float("COMPACTOR_BACKUP_INTERVAL_HOURS", 24.0)

# Idle gate. reqctx.requests_in_flight is a PLAIN IN-PROCESS COUNTER (see
# reqctx.py) — it lives in whichever process incremented it, which is
# main.py's uvicorn process (chat_completions' wrapper). backup.py --daemon
# runs as its OWN supervisord program, a separate OS process (see
# supervisord.conf [program:backup] vs [program:compactor]) — so importing
# reqctx directly here and reading its counter would always read 0, a false
# "idle", never having seen any real request. The pod's existing operator
# tooling (ops/zlops/common.py wait_for_no_inflight_requests) already solved
# exactly this cross-process problem the same way: poll the compactor's own
# `/health/full` over loopback HTTP and read
# `request_tracing.requests_in_flight` from the JSON. This mirrors that
# field path for consistency with the one other place in this repo that
# reads it. See the module docstring / this file's own report for the
# pushback this correction is worth flagging.
COMPACTOR_PORT = os.environ.get("COMPACTOR_PORT", "8080").strip() or "8080"
HEALTH_URL = os.environ.get(
    "COMPACTOR_UPLOADS_BACKUP_HEALTH_URL",
    f"http://127.0.0.1:{COMPACTOR_PORT}/health/full",
)
_IDLE_HTTP_TIMEOUT_S = 5.0

_ARCHIVE_HEX_RE = None  # placeholder, unused; kept out of hot path


# ---------------------------------------------------------------------------
# Idle gate
# ---------------------------------------------------------------------------


def _extract_inflight(doc: object) -> int | None:
    """Same shape as ops/zlops/common.py's own `_extract_inflight` — kept as
    an independent copy (this module must not import the `ops` package,
    which is not shipped into the image) rather than a shared import."""
    if not isinstance(doc, dict):
        return None
    rt = doc.get("request_tracing")
    if not isinstance(rt, dict):
        return None
    n = rt.get("requests_in_flight")
    return n if isinstance(n, int) else None


def is_idle() -> tuple[bool, str]:
    """True when it looks safe to run a paced backup cycle right now.

    Fails OPEN (treated as idle) only when the endpoint could not be reached
    at all (connection refused, DNS, timeout — nothing there to be busy) or
    answered 200 with the field simply absent (an older compactor, or a bare
    200 with no `request_tracing` key). This is a periodic, low-priority,
    PACED job, not a stop-the-world operator action, so the doctrine here is
    deliberately looser than ops.wait_for_no_inflight_requests's hard
    refusal for THAT one case: an unreachable health check is not evidence
    of a live request, and skipping every cycle whenever health is merely
    flaky would cost real durability for no real protection.

    hostile-review round 2, finding 4: that same looseness must NOT extend
    to an HTTP error RESPONSE (503, 500, ...) — the endpoint is reachable
    and answering, which is exactly when it is most likely to be reporting
    something wrong. The body is still read and its `request_tracing.
    requests_in_flight` still consulted when present; a NON-ZERO or
    UNPARSEABLE value on an error response fails CLOSED (skips this cycle)
    rather than open, since an error response is itself weak evidence of
    load the 200 path never has to weigh.

    hostile-review round 3, LOW: a TIMEOUT is not the same evidence as a
    refused connection. Nothing answering at all (refused/DNS/no route)
    means there is nothing there to be busy — fail OPEN, as above. A
    connection that *is* accepted but never answers within
    `_IDLE_HTTP_TIMEOUT_S` looks exactly like an overloaded process that
    cannot get to accepting/finishing this request — fail CLOSED (skip)
    instead, same direction as the HTTP-error-response case just above.
    """
    try:
        req = urllib.request.Request(HEALTH_URL, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=_IDLE_HTTP_TIMEOUT_S) as resp:
                body = resp.read()
        except urllib.error.HTTPError as he:
            body = he.read()
            try:
                doc = json.loads(body)
            except (ValueError, json.JSONDecodeError):
                return False, (
                    f"health check returned HTTP {he.code} with an unparsable "
                    f"body; skipping this cycle (fails CLOSED on an error response)"
                )
            n = _extract_inflight(doc)
            if n is None or n != 0:
                return False, (
                    f"health check returned HTTP {he.code}; requests_in_flight="
                    f"{n!r}; skipping this cycle"
                )
            return True, f"health check returned HTTP {he.code} but requests_in_flight is 0"
        except TimeoutError:
            return False, (
                f"health check timed out after {_IDLE_HTTP_TIMEOUT_S:.0f}s; "
                f"skipping this cycle (fails CLOSED on a timeout, unlike a refused connection)"
            )
        except urllib.error.URLError as ue:
            if isinstance(ue.reason, TimeoutError):
                return False, (
                    f"health check timed out after {_IDLE_HTTP_TIMEOUT_S:.0f}s; "
                    f"skipping this cycle (fails CLOSED on a timeout, unlike a refused connection)"
                )
            raise
        doc = json.loads(body)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
        return True, f"health check unreachable ({type(e).__name__}); proceeding without the idle wait"
    n = _extract_inflight(doc)
    if n is None:
        return True, "requests_in_flight not reported; proceeding without the idle wait"
    if n <= 0:
        return True, "requests_in_flight is 0"
    return False, f"requests_in_flight is {n}; skipping this cycle"


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def _empty_manifest() -> dict:
    # objects_count/objects_bytes (hostile-review round 2, finding 3): kept
    # IN the manifest, maintained incrementally as objects are added/removed
    # by run_once, so health_snapshot() never has to walk the objects/ tree
    # itself (see that function's own docstring for why that walk was the
    # actual bug — it ran on the request-serving event loop).
    return {
        "generation": 0, "files": {}, "last_suspect_run": None,
        "objects_count": 0, "objects_bytes": 0,
    }


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def load_manifest(root: Path) -> dict:
    p = _manifest_path(root)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return _empty_manifest()
    except (OSError, ValueError) as e:
        # A corrupt manifest must never look like "nothing has ever been
        # backed up" (which would re-copy every object, harmless but
        # slow) NOR crash the daemon. Log loudly and start clean; existing
        # objects are untouched either way (content-addressed, so a
        # re-scan just re-discovers them as "already have this sha").
        logger.error(f"uploads manifest at {p} is unreadable ({type(e).__name__}); starting from an empty manifest")
        return _empty_manifest()
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        logger.error(f"uploads manifest at {p} has an unexpected shape; starting from an empty manifest")
        return _empty_manifest()
    data.setdefault("generation", 0)
    data.setdefault("last_suspect_run", None)
    needs_reconcile = "objects_count" not in data or "objects_bytes" not in data
    data.setdefault("objects_count", 0)
    data.setdefault("objects_bytes", 0)
    if needs_reconcile:
        # One-time backward-compat reconciliation for a manifest written
        # before this counter existed: walk objects/ ONCE to seed the
        # counters, here in the daemon process (never on the request-
        # serving event loop — health_snapshot() below never calls this
        # walk itself, only reads the counters this seeds).
        n, total = 0, 0
        objects_dir = root / OBJECTS_DIRNAME
        if objects_dir.is_dir():
            for p in objects_dir.rglob("*"):
                if p.is_file():
                    n += 1
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
        data["objects_count"] = n
        data["objects_bytes"] = total
        # hostile-review round 4, LOW: load_manifest is a READ path — it
        # must never write, full stop, because health_snapshot() (called
        # from the request-serving process, potentially very often) calls
        # it too. Round 3 had this write the migration back immediately,
        # which technically worked but blurred that boundary. Since
        # run_once() already calls save_manifest() unconditionally at the
        # end of every successful cycle, the very next real cycle persists
        # this same reconciliation for free — the walk here simply repeats
        # (cheap relative to a whole backup cycle) until that happens,
        # exactly like a corrupt-manifest recovery already does.
    return data


def _atomic_write_json(path: Path, data: object) -> None:
    """Same crash-safe pattern as memory.atomic_write_json / backup.py's own
    fsync helpers: write a temp file in the same directory, fsync it, then
    os.replace, then fsync the directory so the rename itself is durable on
    a network mount (MooseFS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    try:
        dirfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except OSError:
        pass  # best-effort, same doctrine as backup._fsync_dir


def save_manifest(root: Path, manifest: dict) -> None:
    _atomic_write_json(_manifest_path(root), manifest)


# ---------------------------------------------------------------------------
# Cross-process lock (hostile-review round 3, LOW)
# ---------------------------------------------------------------------------

LOCK_NAME = ".uploads-backup.lock"


@contextlib.contextmanager
def locked(root: Path):
    """An advisory `flock` over one file in the mirror root, held for the
    duration of a whole run_once cycle OR a whole `accept` invocation —
    the two writers of manifest.json that are not the same process
    (the daemon's own cycle, and an operator's `scripts/uploads-backup.py
    accept` run concurrently). Without this, `accept` reading/marking
    entries while a cycle is mid-write (or vice versa) could interleave
    two read-modify-write manifest saves and silently drop one side's
    changes. BLOCKING (not a timeout/refusal) — both sides are short-lived
    and correctness matters more than never waiting; a daemon cycle and a
    manual `accept` colliding is rare enough that a brief wait is the
    right tradeoff over either racing ahead or refusing outright.

    `fcntl.flock` is per-OPEN-FILE-DESCRIPTION: closing the fd (leaving
    this context manager, even via an exception) always releases it, so
    this can never deadlock a later call in the same or another process
    even if the previous holder died uncleanly.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_NAME
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Content-addressed object store
# ---------------------------------------------------------------------------


def _object_path(root: Path, sha256: str) -> Path:
    return root / OBJECTS_DIRNAME / sha256[:2] / sha256


def has_object(root: Path, sha256: str, size: int | None = None) -> bool:
    """hostile-review round 2, finding 1: verify SIZE too, not just that a
    path exists under this sha256 — a cheap check (one stat, no re-hash)
    that catches the object having been truncated or otherwise damaged out
    from under the mirror. `size=None` (the default) keeps the old
    existence-only behavior for callers that don't have a size handy."""
    p = _object_path(root, sha256)
    if not p.is_file():
        return False
    if size is None:
        return True
    try:
        return p.stat().st_size == size
    except OSError:
        return False


def _sha256_file(path: Path) -> tuple[str, int]:
    """Hash-only, no copy. Kept for callers (verify --deep in the operator
    script has its own independent copy) that only need to confirm content,
    never to write it anywhere."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


class _RaceDetected(Exception):
    """hostile-review round 3, finding 5: an IN-PLACE rewrite (same size,
    new bytes — e.g. a same-size in-place edit, or a rewrite that happens
    to finish at the same length) can happen entirely WITHIN a single read
    pass, which finding 1's one-read-pass fix does not by itself catch: the
    hash and the copy still agree with EACH OTHER (self-consistent), but
    neither agrees with any single true state of the file, because the
    file changed mid-read. `_hash_and_copy_object` fstats the open handle
    before and after the read; if size or mtime_ns moved, this is raised
    and the tmp copy is discarded — the caller treats this exactly like
    the settle-window skip: not new, not changed, picked up cleanly next
    run once the file has stopped moving."""


def _hash_and_copy_object(
    src: Path, root: Path, *,
    settle_cutoff: float | None = None,
    classify_sig: tuple[int, int, int] | None = None,
) -> tuple[str, int, bool, float, int, int]:
    """hostile-review round 2, finding 1 (the core fix): hash WHILE copying,
    in one read pass, then name the object after the hash just computed —
    never hash first and copy second as two separate reads of the same
    live file. Two separate reads left a window where a file that changed
    IN BETWEEN them (an in-place edit racing this backup) could be copied
    under the FIRST read's hash: an object named sha(A) that actually holds
    bytes B, captured as "good" forever. A single read pass makes that
    specific race impossible — the object's name and its bytes always come
    from the same read.

    hostile-review round 3, finding 5: an in-place rewrite that happens
    WITHIN the one read pass (same open handle) is a narrower, remaining
    race — closed here by fstat-ing the open handle before and after
    reading it. A mismatch raises `_RaceDetected` (see its own docstring);
    the tmp copy is discarded, nothing is written, and the PRE-read stat
    (not a `path.stat()` taken by the caller sometime after this returns,
    which could itself have kept racing) is what a caller should record as
    this file's signature.

    hostile-review round 4, finding 2: the round-3 fstat-before/after check
    only catches a rewrite that STRADDLES the read — a writer that pauses
    mid-rewrite (old content, but a stat already reflecting the in-
    progress edit) with the pause landing entirely BEFORE this function's
    own open()/fstat() could still be captured whole, self-consistently,
    as a torn A/B blend that never existed as a real, complete file at any
    single instant. Two more (both optional, both checked BEFORE the read
    starts, so a caught case costs nothing) close that: `settle_cutoff`
    rejects a file whose PRE-read mtime is itself still within the settle
    window (the write may have only just paused, not finished), and
    `classify_sig` rejects a file whose PRE-read (size, mtime_ns, inode)
    already disagrees with what classification saw moments earlier in this
    same run_once call — either one means "something about this file
    moved between classification and this read," which is reason enough
    not to trust whatever a single read of it produces.

    Returns (sha256, size, created, mtime, mtime_ns, inode) — all from the
    PRE-read fstat. `created` is False when an object under that hash
    already existed (nothing new was written; the size check in
    `has_object` still confirms it isn't damaged) so callers can track
    objects_count/objects_bytes deltas without re-deriving them from a
    directory walk (see the manifest's own objects_count/objects_bytes,
    finding 3).
    """
    tmp_dir = root / OBJECTS_DIRNAME
    tmp_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    fd, tmp_path = tempfile.mkstemp(dir=str(tmp_dir), prefix=".hashcopy.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            pre = os.fstat(inp.fileno())
            if settle_cutoff is not None and pre.st_mtime > settle_cutoff:
                raise _RaceDetected(
                    "source file's mtime is still within the settle window at the moment of read"
                )
            if classify_sig is not None and (pre.st_size, pre.st_mtime_ns, pre.st_ino) != classify_sig:
                raise _RaceDetected(
                    "source file's stat already differs from what classification saw this same run"
                )
            while True:
                chunk = inp.read(CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                h.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
            post = os.fstat(inp.fileno())
        if (pre.st_size, pre.st_mtime_ns) != (post.st_size, post.st_mtime_ns):
            os.unlink(tmp_path)
            raise _RaceDetected(
                "source file's stat moved during the read (size/mtime changed mid-copy)"
            )
        sha = h.hexdigest()
        dest = _object_path(root, sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if has_object(root, sha, size=size):
            os.unlink(tmp_path)
            return sha, size, False, pre.st_mtime, pre.st_mtime_ns, pre.st_ino
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    try:
        dirfd = os.open(str(dest.parent), os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except OSError:
        pass
    return sha, size, True, pre.st_mtime, pre.st_mtime_ns, pre.st_ino


def _remove_object(root: Path, sha256: str) -> int:
    """Removes one object, returning the bytes freed (0 if it was already
    gone) — used by the expiry sweep to keep manifest['objects_bytes']
    accurate without a directory walk."""
    p = _object_path(root, sha256)
    try:
        n = p.stat().st_size
        p.unlink()
        return n
    except OSError:
        return 0


_STALE_HASHCOPY_TMP_S = 3600.0  # 1h — see _cleanup_stale_hashcopy_tmp


def _cleanup_stale_hashcopy_tmp(root: Path) -> int:
    """rc-review fix, LOW: `_hash_and_copy_object` writes its tmp file
    (`objects/.hashcopy.<rand>.tmp`) then either `os.replace`s it to its
    final object path or `os.unlink`s it — on every code path, including
    every `except BaseException` — so under normal operation none should
    ever survive a call. A process that dies harder than that (SIGKILL,
    an OOM kill, a pod crash mid-copy) can still leave one behind: never
    counted against objects_count/objects_bytes (only a real replace does
    that), so it is not a correctness problem, just accumulating disk
    usage on MooseFS forever with nothing to ever clean it up. Called once
    at the START of every run_once cycle, before anything else touches
    `objects/`; only removes files whose mtime is older than
    `_STALE_HASHCOPY_TMP_S` (1h) — comfortably longer than any real copy
    could still be in flight — so an in-progress copy from a CONCURRENT
    run (the advisory lock normally prevents this, but this check costs
    nothing extra to keep conservative) is never touched. Returns the
    count removed; never raises (a cleanup step must not be able to take
    a whole cycle down)."""
    objects_dir = root / OBJECTS_DIRNAME
    removed = 0
    try:
        if not objects_dir.is_dir():
            return 0
        cutoff = time.time() - _STALE_HASHCOPY_TMP_S
        for entry in objects_dir.iterdir():
            name = entry.name
            if not (name.startswith(".hashcopy.") and name.endswith(".tmp")):
                continue
            try:
                if not entry.is_file():
                    continue
                if entry.stat().st_mtime >= cutoff:
                    continue
                entry.unlink()
                removed += 1
            except OSError:
                continue
    except OSError:
        return removed
    if removed:
        logger.info(f"uploads backup: removed {removed} stale .hashcopy.*.tmp file(s) older than {_STALE_HASHCOPY_TMP_S:.0f}s")
    return removed


# ---------------------------------------------------------------------------
# Magic-byte / extension sanity check (v3.1.9.8 item 2's "suspect_content")
# ---------------------------------------------------------------------------

# Deliberately small and conservative: only extensions with a cheap, known
# magic prefix are checked at all. An unlisted extension (or none) records
# no verdict either way — this is a smoke detector for "a .jpg that is
# actually gzip" (today's literal incident), not a general file-type
# validator.
_MAGIC_BY_EXT: dict[str, tuple[bytes, ...]] = {
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".webp": (b"RIFF",),  # "WEBP" follows at offset 8; RIFF alone is enough
    ".pdf": (b"%PDF-",),
    ".gz": (b"\x1f\x8b",),
    ".bmp": (b"BM",),
}


def _magic_check(path: Path) -> dict | None:
    """None if the extension is unlisted (no verdict) or the header
    matches; else a small record describing the mismatch — no file NAME or
    content bytes beyond a short hex prefix of the header itself (never her
    data: this is the file's own magic bytes, a handful of bytes that
    identify a FORMAT, not content)."""
    ext = path.suffix.lower()
    magics = _MAGIC_BY_EXT.get(ext)
    if not magics:
        return None
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if any(head.startswith(m) for m in magics):
        return None
    return {"expected_ext": ext, "head_hex": head[:8].hex()}


# ---------------------------------------------------------------------------
# Free space
# ---------------------------------------------------------------------------


class _Refusal(RuntimeError):
    """A deliberate, SAFE-TO-SHOW refusal (free space / mirror-size cap) —
    never contains an upload's own name or content, only aggregate numbers
    and this module's own configured paths (the backup root, not her
    uploads dir's contents). run_once's own except block (finding 8) keeps
    THIS exception's str(e), unlike an arbitrary Exception, precisely
    because it is constructed only from those safe values below."""


class _RootWalkUnreadable(_Refusal):
    """rc-review fix, LOW-MED: raised by `_iter_uploads`' own `onerror`
    callback when the uploads directory ROOT ITSELF (not a subdirectory)
    could not be scanned — e.g. `chmod` leaves the directory enterable
    (execute bit, so `Path.is_dir()` above still succeeds) but not
    listable (no read bit), a transient EACCES/EIO, or any other error
    `os.walk`'s own top-level `scandir(top)` call can raise.

    Before this fix, that case was folded into the ordinary
    unreadable-subdirectory list: `Path(exc.filename).relative_to(
    uploads_dir)` for the root itself is exactly `"."`, so the vanished-
    exclusion prefix built from it was `"./"` — which matches NO tracked
    name (they are plain relative names like `"ab/cd.jpg"`, never
    `"./ab/cd.jpg"`). Every previously-tracked file therefore looked
    vanished, every run, for as long as the root stayed unlistable, which
    either tripped the mass-change guard on a transient permissions
    glitch or (once already suspect) kept it frozen forever with a
    steadily growing, entirely wrong "vanished" count.

    Subclassing `_Refusal` reuses run_once's existing `_Refusal` handling
    for free: `ok=False`, `last_error=type(e).__name__`, an alert, and —
    the whole point — NO vanished/deleted accounting happens at all this
    run (the abort happens before any of that code runs), so nothing in
    the manifest is touched. The manifest and every object on disk are
    exactly as they were; the next cycle simply tries again."""

    def __init__(self, os_error_type: str) -> None:
        super().__init__(
            f"the uploads directory root itself could not be listed "
            f"({os_error_type}); aborting this run with NO vanished/deleted "
            f"accounting rather than risk treating every tracked upload as "
            f"gone. The manifest is untouched. Will retry next cycle."
        )


def _require_free_space(root: Path, needed_bytes: int) -> None:
    """hostile-review round 2, finding 9: on a network filesystem (MooseFS),
    `statvfs`/`shutil.disk_usage` reports the CLUSTER's free space, not any
    quota this mount actually has — so this check alone means very little
    here. It is kept (a cluster genuinely out of space is still real
    signal, and this costs nothing to check), but `_require_mirror_size_cap`
    below is the guard that actually means something on this filesystem."""
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(str(probe)).free
    except OSError:
        return  # can't tell; don't block a run over an unmeasurable volume
    want = int(needed_bytes * 1.1) + MIN_FREE_MB * 1024 * 1024
    if free < want:
        raise _Refusal(
            f"refusing to run the uploads backup: {free / 1048576:.0f} MB free at "
            f"{probe}, and this run needs about {want / 1048576:.0f} MB. Nothing "
            f"has been touched."
        )


def _require_mirror_size_cap(manifest: dict) -> None:
    """hostile-review round 2, finding 9: a real cap on the mirror's OWN
    size (objects_bytes, tracked in the manifest itself — see finding 3),
    independent of the cluster-wide free-space number above. The mirror
    roughly tracks (and, as edits accumulate extra drifted-content objects,
    can exceed) the live uploads directory's own size — see OPERATIONS.md.
    Raises if already over the cap; new copies are refused, existing data
    untouched."""
    cap = MAX_MIRROR_MB * 1024 * 1024
    have = manifest.get("objects_bytes", 0)
    if cap > 0 and have >= cap:
        raise _Refusal(
            f"refusing to copy new objects: the mirror is already "
            f"{have / 1048576:.0f} MB, at or over its "
            f"COMPACTOR_UPLOADS_BACKUP_MAX_MB cap of {MAX_MIRROR_MB} MB. "
            f"Nothing new was copied (entries past the keep window were already expired "
            f"this cycle); raise the cap or wait for more deleted entries to age out."
        )


# ---------------------------------------------------------------------------
# The scan + copy cycle
# ---------------------------------------------------------------------------


def _iter_uploads(uploads_dir: Path) -> tuple[list[Path], list[str]]:
    """hostile-review round 2, finding 6: `p.is_file()` on a glob result
    FOLLOWS symlinks — a symlink to a regular file elsewhere on the host
    (or a whole symlinked subdirectory) looked exactly like a real upload,
    and this mirror would happily read and copy whatever it pointed at.
    `os.walk(followlinks=False)` (the default, made explicit here) never
    descends into a symlinked directory at all, and `lstat` + `S_ISREG` on
    each entry (never a plain `is_file()`, which stats through the link)
    rejects a symlinked FILE too, along with anything else that isn't a
    plain regular file (fifos, devices, sockets).

    hostile-review round 4, LOW: `os.walk` SILENTLY skips a directory it
    cannot list (permission denied, e.g.) unless given an `onerror`
    callback — every tracked name under such a subdirectory would then
    simply be absent from the returned list, indistinguishable from a
    real delete, and could trip (or quietly feed) the mass-change guard
    over a transient permissions glitch rather than a real change.
    Returns (files, unreadable_rel_dirs) — the second list (relative to
    `uploads_dir`) lets the caller exclude previously-tracked names under
    those specific subdirectories from `vanished` this run, rather than
    either trusting the (incomplete) listing or discarding the whole run.
    """
    if not uploads_dir.is_dir():
        return [], []
    out: list[Path] = []
    unreadable: list[str] = []
    root_str = str(uploads_dir)

    def _onerror(exc: OSError) -> None:
        # rc-review fix, LOW-MED: the ROOT itself (os.walk's own top-level
        # scandir(uploads_dir) call, not a subdirectory found while
        # walking) failing is NOT just another unreadable-subdirectory
        # entry — see _RootWalkUnreadable's own docstring for why treating
        # it as one used to be a permanent false-vanished trap.
        if exc.filename is not None and str(exc.filename) == root_str:
            raise _RootWalkUnreadable(type(exc).__name__) from exc
        try:
            rel = str(Path(exc.filename).relative_to(uploads_dir))
        except (ValueError, TypeError):
            rel = exc.filename or "?"
        unreadable.append(rel)
        logger.warning(f"uploads backup: could not list a subdirectory ({type(exc).__name__}); its files are excluded from vanished/deleted this run")

    for dirpath, _dirnames, filenames in os.walk(uploads_dir, followlinks=False, onerror=_onerror):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            out.append(p)
    return sorted(out), unreadable


def _stat_sig(st: os.stat_result) -> tuple[int, int, int]:
    return (st.st_size, st.st_mtime_ns, st.st_ino)


# hostile-review round 2, finding 1: a manifest record's TOP-LEVEL
# sha256/size/mtime/mtime_ns/inode fields are the CANONICAL, original-good
# object — set once, when a name is first mirrored, and NEVER overwritten
# after that, no matter how many times its live content later drifts. The
# cur_* fields (added lazily, only once a record's first stat/content
# touch happens) track what was last actually OBSERVED on disk — used to
# decide whether the NEXT run needs to look at this file at all (a stat-sig
# match against cur_* means "nothing happened since we last looked", not
# "this file was never touched"). Absent cur_* fields mean "never diverged
# from canonical yet" — these two helpers make that fallback the one place
# it has to be spelled out.
def _cur_sig(rec: dict) -> tuple:
    return (
        rec.get("cur_size", rec.get("size")),
        rec.get("cur_mtime_ns", rec.get("mtime_ns")),
        rec.get("cur_inode", rec.get("inode")),
    )


def _cur_sha(rec: dict) -> str | None:
    return rec.get("cur_sha256", rec.get("sha256"))


def _alert_uploads_suspect(detail: str, extra: dict) -> None:
    """hostile-review round 2, finding 2: route a suspect run through the
    same failure-alert path backup.py's own _alert_failure uses, so the
    owner finds out from the same channel a failed main backup would use —
    counts only (`extra`), never a file name. No-op if no webhook is
    configured (alert.enabled() is False); a failure to SEND the alert is
    logged, never raised — an alerter must never break the thing it
    watches, same doctrine as backup._alert_failure."""
    try:
        import alert
        alert.notify("uploads_backup", "suspect", detail, extra=extra)
    except Exception as e:
        logger.error(
            f"could not send uploads-backup suspect alert ({type(e).__name__}); "
            f"the condition it was reporting was: {detail}"
        )


def _alert_uploads_failure(error_type: str) -> None:
    """hostile-review round 3, finding 6: a run that fails outright (an
    exception, not just a busy skip or a suspect trip) needs the same
    owner-facing signal a failed main backup gets — same webhook, same
    no-op-if-unconfigured/never-raise doctrine as _alert_uploads_suspect.
    Type name only; never any exception message (see run_once's own
    except block for why)."""
    try:
        import alert
        alert.notify("uploads_backup", "fail", error_type)
    except Exception as e:
        logger.error(f"could not send uploads-backup failure alert ({type(e).__name__})")


def run_once(
    uploads_dir: Path | None = None,
    backup_root: Path | None = None,
    *,
    idle_check: Callable[[], tuple[bool, str]] = is_idle,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """One incremental mirror cycle. Never raises — failures are reported,
    like backup.run_once, so a bad cycle here can never take the process
    down or interrupt the caller's own (main backup) cycle.

    hostile-review round 2 changed this function's shape in three ways,
    each with its own inline comment at the point it applies:
      (1) hashing and copying a file are now ONE read pass (finding 1), a
          settle window skips a file that may still be mid-write, and a
          content drift updates cur_* fields so it is only ever counted
          once, not on every subsequent run;
      (2) the mass-change guard freezes only vanished/changed PER-FILE
          records — new files are still copied, since objects are add-only
          and safe (finding 2);
      (4) the idle gate is re-polled every IDLE_RECHECK_FILES/BYTES during
          the copy loops, and a newly-busy pod stops this cycle cleanly.

    hostile-review round 3 changed it again:
      (1) once suspect, STAYS suspect (frozen) until `accept` — a probe
          showed adding new (safe) files while suspect inflated the
          baseline enough to silently un-suspect on the VERY NEXT run;
      (5) `_hash_and_copy_object`'s own fstat-before/after race guard is
          respected here (`_RaceDetected` is caught, not a hard failure);
      (6) an undecodable filename is skipped and counted, never allowed to
          reach json.dump and take the whole cycle's manifest write down
          with it; a persistent status.json (not just in-process globals)
          makes ok/last_error/last_run_at/last_run_s visible cross-process;
      + a whole-cycle advisory lock (see `locked()`) shared with `accept`.
    """
    uploads_dir = Path(uploads_dir) if uploads_dir is not None else UPLOADS_DIR
    root = Path(backup_root) if backup_root is not None else UPLOADS_BACKUP_DIR
    t0 = time.monotonic()
    report: dict = {
        "ok": False, "skipped": None, "suspect": False, "detail": "",
        "new": 0, "changed": 0, "deleted": 0, "expired": 0,
        "bytes_copied": 0, "suspect_content": 0,
        "paused": False, "skipped_settling": 0,
        "skipped_undecodable": 0, "skipped_race": 0,
        "resurrected": 0, "copy_errors": 0, "walk_errors": 0,
    }

    ok_idle, why = idle_check()
    if not ok_idle:
        report["ok"] = True
        report["skipped"] = "busy"
        report["detail"] = why
        return report

    # --- mid-run idle re-check (finding 4) -----------------------------
    busy_stop = False
    _recheck_files = 0
    _recheck_bytes = 0

    def _tick(size: int) -> bool:
        """Call once per file processed in either copy loop below. Returns
        True once the pod has gone busy mid-run — the caller must stop
        immediately. Whatever is already in `files`/`manifest` at that
        point stays exactly as applied (every mutation below is either a
        brand-new object/record or an update to one already fully formed
        this run), and whatever is left simply looks unprocessed to the
        NEXT cycle, which picks it up the same way it would any other day.
        """
        nonlocal busy_stop, _recheck_files, _recheck_bytes
        if busy_stop:
            return True
        _recheck_files += 1
        _recheck_bytes += size
        if _recheck_files >= IDLE_RECHECK_FILES or _recheck_bytes >= IDLE_RECHECK_BYTES:
            _recheck_files = 0
            _recheck_bytes = 0
            still_idle, why2 = idle_check()
            if not still_idle:
                busy_stop = True
                logger.info(
                    f"uploads backup: pausing mid-run ({why2}); progress so "
                    f"far is saved, resuming next cycle"
                )
        return busy_stop

    try:
        # hostile-review round 3, LOW: shared with `accept` in the operator
        # script — see `locked()`'s own docstring for why (two writers of
        # the same manifest.json that are not the same process).
        with locked(root):
            _cleanup_stale_hashcopy_tmp(root)
            manifest = load_manifest(root)
            files = manifest["files"]
            now = time.time()
            now_iso = _iso(now)

            # hostile-review round 3, finding 1: whether this run was
            # ALREADY suspect before it did anything — read once, up
            # front, so nothing below can accidentally recompute its way
            # out of it.
            suspect_before = manifest.get("last_suspect_run") is not None

            current, unreadable_dirs = _iter_uploads(uploads_dir)
            report["walk_errors"] = len(unreadable_dirs)
            current_names = set()
            rel_to_path: dict[str, Path] = {}
            for p in current:
                rel = str(p.relative_to(uploads_dir))
                # hostile-review round 3, finding 6: a name that cannot
                # round-trip through UTF-8 (a stray non-UTF-8 byte in the
                # filename) would otherwise reach json.dump(ensure_ascii=
                # False) at save_manifest time and raise
                # UnicodeEncodeError — which used to abort the ENTIRE
                # cycle's manifest write, discarding every other file's
                # legitimate progress, EVERY run, forever (the bad name
                # never goes away on its own). Skip and COUNT it instead —
                # never print/log the name itself, only that one existed.
                try:
                    rel.encode("utf-8")
                except UnicodeEncodeError:
                    report["skipped_undecodable"] += 1
                    continue
                current_names.add(rel)
                rel_to_path[rel] = p

            # --- classify against the manifest, stat-only (no hashing) ---
            previously_tracked = {n for n, rec in files.items() if rec.get("deleted_at") is None}
            # hostile-review round 4, LOW: a name under a subdirectory
            # os.walk just failed to list is UNKNOWN this run, not
            # vanished — it must never be treated as deleted (and so
            # never feed the mass-change guard) purely because a
            # permissions glitch made it briefly invisible.
            unreadable_prefixes = tuple(d.rstrip("/") + "/" for d in unreadable_dirs)
            vanished = sorted(
                n for n in previously_tracked
                if n not in current_names and not n.startswith(unreadable_prefixes)
            )

            settle_cutoff = now - SETTLE_S
            need_rehash: list[str] = []  # tracked names whose stat signature moved
            new_names: list[str] = []    # not in the manifest at all
            resurrected_now: set[str] = set()  # tracked-but-deleted names that are LIVE again
            class_stat: dict[str, tuple[int, int, int]] = {}  # rel -> (size, mtime_ns, inode) AT CLASSIFY TIME
            for rel in sorted(current_names):
                rec = files.get(rel)
                path = rel_to_path[rel]
                try:
                    st = path.stat()
                except OSError as e:
                    logger.warning(f"uploads backup: could not stat a tracked upload ({type(e).__name__}); skipping it this run")
                    continue
                if st.st_mtime > settle_cutoff:
                    # finding 1: possibly still being written. Touch
                    # NOTHING — not new, not a rehash candidate — so a
                    # half-written upload is never captured mid-write.
                    # Picked up cleanly once it stops changing.
                    report["skipped_settling"] += 1
                    continue
                class_stat[rel] = _stat_sig(st)
                if rec is None:
                    new_names.append(rel)
                    continue
                if rec.get("deleted_at") is not None:
                    # hostile-review round 4, finding 1: a name the
                    # manifest still calls "deleted" but that is LIVE
                    # again (restored from the mirror, or simply
                    # re-uploaded under the same name) must not stay
                    # marked deleted forever — that left its record
                    # invisible to the guard, `restore` reporting "nothing
                    # to do" for it, and the expiry sweep eventually
                    # deleting its OBJECT out from under a live file. Force
                    # a rehash regardless of stat sig (`deleted_at` itself
                    # is what changed, even if content/stat happen to
                    # already match) and clear it, unconditionally, once
                    # hashed below — resurrection is exactly as safe/
                    # additive as a brand-new file, so it is never
                    # withheld by suspect state either.
                    resurrected_now.add(rel)
                    need_rehash.append(rel)
                    continue
                sig = _stat_sig(st)
                if _cur_sig(rec) != sig:
                    need_rehash.append(rel)

            # --- sweep: expire long-deleted records + orphaned objects ---
            # rc-review fix, LOW: this used to run at the very END of the
            # cycle (after new-file copying), which meant a mirror already
            # AT or OVER COMPACTOR_UPLOADS_BACKUP_MAX_MB never reached it —
            # `_require_mirror_size_cap` below raises _Refusal (caught by
            # run_once's own except block, well before this point) on
            # EVERY cycle once over cap, so the one thing that could bring
            # it back under the cap (expiring records already >
            # COMPACTOR_UPLOADS_BACKUP_KEEP_DELETED_DAYS old and freeing
            # their now-orphaned objects) could never run either: a
            # once-over-cap mirror was stuck over cap FOREVER, even as more
            # of its own tracked-deleted entries aged past the keep
            # window. Moved here, before the cap check, so an aged-out
            # backlog can shrink the mirror enough for THIS SAME cycle's
            # cap check just below to pass.
            #
            # Gated on `suspect_before` (known coming INTO this run), not
            # `suspect_active` (this run's own freshly-computed verdict,
            # not decided until the mass-change guard further down) —
            # sweeping only ever removes objects already recorded as
            # deleted_at more than KEEP_DELETED_DAYS ago by a PRIOR run,
            # nothing this run's own vanished/changed classification has
            # touched, so it carries no more risk here than it did in its
            # old position. No `busy_stop` gate is needed here either — at
            # this point in the function no copying has started yet, so
            # busy_stop cannot yet be true.
            if not suspect_before:
                keep_cutoff = now - KEEP_DELETED_DAYS * 86400.0
                # Never expire a name that is live on disk this cycle (e.g. brought back
                # by `restore --include-deleted` after a long suspect period): the
                # rehash below does files[rel] for it, and popping it here raised
                # KeyError on every later cycle (rc-fix verification, finding 1).
                expired_names = [
                    rel for rel, rec in files.items()
                    if rel not in current_names
                    and rec.get("deleted_at") and _parse_iso(rec["deleted_at"]) is not None
                    and _parse_iso(rec["deleted_at"]) < keep_cutoff
                ]
                referenced_shas = set()
                for rel, rec in files.items():
                    if rel in expired_names:
                        continue
                    referenced_shas.add(rec.get("sha256"))
                    referenced_shas.add(rec.get("cur_sha256"))
                    for c in rec.get("changed", []):
                        referenced_shas.add(c.get("sha256"))
                for rel in expired_names:
                    rec = files.pop(rel)
                    candidate_shas = {rec.get("sha256"), rec.get("cur_sha256")}
                    candidate_shas.update(c.get("sha256") for c in rec.get("changed", []))
                    for sha in candidate_shas:
                        if sha and sha not in referenced_shas:
                            freed = _remove_object(root, sha)
                            manifest["objects_bytes"] = max(0, manifest.get("objects_bytes", 0) - freed)
                            if freed:
                                manifest["objects_count"] = max(0, manifest.get("objects_count", 0) - 1)
                    report["expired"] += 1
                if expired_names:
                    # Persist the sweep now. The checks just below can still refuse,
                    # and the _Refusal path saves nothing, which left expired records
                    # pointing at deleted objects and objects_bytes too high for good
                    # (rc-fix verification, finding 2).
                    save_manifest(root, manifest)

            # --- free-space / mirror-size checks, BEFORE any copying ---
            # hostile-review round 3, LOW: this used to run only against
            # new_names' estimated bytes, AFTER the need_rehash loop had
            # already copied whatever it needed to — so a rehash-driven
            # burst of drifted-content objects could blow past the cap or
            # the free-space floor with no guard at all. One combined
            # estimate, checked before EITHER loop starts; a refusal here
            # touches nothing (no hashing/copying has happened yet).
            estimate = 0
            for rel in need_rehash:
                try:
                    estimate += rel_to_path[rel].stat().st_size
                except OSError:
                    pass
            for rel in new_names:
                try:
                    estimate += rel_to_path[rel].stat().st_size
                except OSError:
                    pass
            if not busy_stop:
                _require_free_space(root, estimate)
                _require_mirror_size_cap(manifest)

            # Rehash only what changed stat-wise — the whole of
            # "incremental". finding 1: hash WHILE copying (one read pass,
            # `_hash_and_copy_object`) instead of hashing then copying as
            # two separate reads of the same live file. finding 5: a race
            # WITHIN that one read pass (fstat before/after disagree) is
            # caught by `_hash_and_copy_object` itself and surfaces here
            # as `_RaceDetected` — treated exactly like the settle-window
            # skip: not new, not changed, picked up cleanly next run.
            changed_content: dict[str, dict] = {}  # rel -> new cur_* fields (a genuinely NEW drift)
            touched_meta: dict[str, dict] = {}      # rel -> new cur_* fields (same content, stat only)
            for rel in need_rehash:
                if busy_stop:
                    break
                path = rel_to_path[rel]
                try:
                    sha, size, created, mtime, mtime_ns, inode = _hash_and_copy_object(
                        path, root, settle_cutoff=settle_cutoff, classify_sig=class_stat.get(rel),
                    )
                except _RaceDetected:
                    report["skipped_race"] += 1
                    continue
                except OSError as e:
                    logger.warning(f"uploads backup: could not read a tracked upload to rehash it ({type(e).__name__}); leaving its record unchanged this run")
                    report["copy_errors"] += 1
                    continue
                if created:
                    manifest["objects_count"] = manifest.get("objects_count", 0) + 1
                    manifest["objects_bytes"] = manifest.get("objects_bytes", 0) + size
                rec = files[rel]
                new_cur = {
                    "cur_sha256": sha, "cur_size": size, "cur_mtime": mtime,
                    "cur_mtime_ns": mtime_ns, "cur_inode": inode,
                }
                if rel in resurrected_now:
                    # finding 1: applied immediately and UNCONDITIONALLY —
                    # never gated on suspect state (resurrection is exactly
                    # as safe/additive as a brand-new file), and never
                    # counted toward the mass-change guard's `affected`
                    # (a file coming back is not damage).
                    rec.update(new_cur)
                    rec["deleted_at"] = None
                    rec["last_seen"] = now_iso
                    report["resurrected"] += 1
                elif sha != _cur_sha(rec):
                    # A genuinely NEW drift relative to what was last
                    # observed — counts toward the mass-change guard below
                    # exactly once; once cur_sha256 is updated (in the
                    # apply step further down), the SAME still-drifted
                    # file compares equal next run and is only a metadata
                    # touch from then on (finding 1: "counted once, not
                    # every day").
                    changed_content[rel] = new_cur
                else:
                    # Same content as last observed (whether that is the
                    # original canonical content or an already-recorded
                    # drift) — just a stat touch. Not a "changed" event.
                    touched_meta[rel] = new_cur
                _tick(size)

            # --- mass-change guard ---
            affected = len(vanished) + len(changed_content)
            baseline = len(previously_tracked)
            pct = (100.0 * affected / baseline) if baseline else 0.0
            freshly_tripped = baseline > 0 and pct > MASS_CHANGE_PCT
            # hostile-review round 3, finding 1: once suspect, STAYS
            # suspect regardless of what THIS run's own percentage comes
            # out to — a probe showed that copying new (safe) files while
            # suspect inflates `baseline` (more tracked, non-deleted
            # names) without inflating `affected` (the same frozen
            # vanished/changed names, never re-counted), so a purely
            # fresh recomputation each run would silently drop BELOW
            # threshold and auto-clear the very condition an operator has
            # not yet looked at. `accept` is the only thing that clears
            # it.
            suspect_active = suspect_before or freshly_tripped

            # finding 1 (round 2): metadata-only touches are always safe
            # to apply — no content or existence claim changes, only
            # "when did we last look at this stat sig". Applied
            # regardless of suspect state.
            for rel, new_cur in touched_meta.items():
                files[rel].update(new_cur)
                files[rel]["last_seen"] = now_iso

            if freshly_tripped and not suspect_before:
                suspect = {
                    "detected_at": now_iso,
                    "vanished": len(vanished),
                    "changed": len(changed_content),
                    "baseline": baseline,
                    "pct": round(pct, 1),
                    "threshold_pct": MASS_CHANGE_PCT,
                }
                manifest["last_suspect_run"] = suspect
                detail = (
                    f"MASS CHANGE DETECTED: {affected}/{baseline} tracked uploads "
                    f"vanished or changed content in one run ({pct:.0f}% > "
                    f"{MASS_CHANGE_PCT:.0f}% threshold) — freezing the per-file "
                    f"records of every vanished/changed name; the last good "
                    f"state for each is intact. New uploads are still mirrored "
                    f"(objects are add-only). Counts only, never names, by "
                    f"design. Recover with `scripts/uploads-backup.py accept "
                    f"--yes` once you've confirmed the deletions are legitimate."
                )
                logger.error(f"uploads backup: {detail}")
                _alert_uploads_suspect(detail, {
                    "vanished": len(vanished), "changed": len(changed_content),
                    "baseline": baseline, "pct": round(pct, 1),
                })
            elif suspect_before:
                # STILL suspect — the marker's own detected_at/baseline/
                # threshold_pct are NEVER rewritten; only its live counts
                # are refreshed, against the ORIGINAL baseline, purely for
                # `status`/health visibility. Nothing in `files` is
                # touched for vanished/changed names beyond the harmless
                # touched_meta pass above.
                marker = dict(manifest.get("last_suspect_run") or {})
                orig_baseline = marker.get("baseline") or baseline
                marker["vanished"] = len(vanished)
                marker["changed"] = len(changed_content)
                if orig_baseline:
                    marker["pct"] = round(100.0 * (len(vanished) + len(changed_content)) / orig_baseline, 1)
                marker["last_checked_at"] = now_iso
                manifest["last_suspect_run"] = marker
                detail = (
                    f"STILL SUSPECT (first detected {marker.get('detected_at')}): "
                    f"{len(vanished)} vanished + {len(changed_content)} changed "
                    f"tracked names remain frozen. Recover with `scripts/"
                    f"uploads-backup.py accept --yes` once confirmed legitimate."
                )
                logger.error(f"uploads backup: {detail}")
            else:
                manifest["last_suspect_run"] = None
                for rel, new_cur in changed_content.items():
                    rec = files[rel]
                    rec.setdefault("changed", [])
                    rec["changed"].append({"sha256": new_cur["cur_sha256"], "detected_at": now_iso})
                    rec.update(new_cur)
                    rec["last_seen"] = now_iso
                    # NEVER overwrite rec["sha256"]/size/mtime/mtime_ns/
                    # inode — those stay the CANONICAL, original-good
                    # object forever. The new (drifted) content is ALSO
                    # now its own object in the store (added above, in
                    # the same read pass that hashed it) — recoverable
                    # for forensics — but nothing points the canonical
                    # fields at it.
                    report["changed"] += 1
                    logger.warning(
                        "uploads backup: a previously-mirrored upload's content changed "
                        "(sha256 differs from what was last observed); keeping the "
                        "canonical backed-up object, recording the new hash under "
                        "'changed' and as its own object. Counts only."
                    )
                for rel in vanished:
                    rec = files[rel]
                    if rec.get("deleted_at") is None:
                        rec["deleted_at"] = now_iso
                        report["deleted"] += 1

            # --- apply: new files ---
            # finding 2 (round 2): new files are copied REGARDLESS of
            # suspect_active — objects are add-only, so backing up a
            # genuinely new upload is safe even while a mass-change is
            # under investigation; withholding it would just mean more to
            # lose if the investigation takes days.
            suspect_content_count = 0
            for rel in new_names:
                if busy_stop:
                    break
                path = rel_to_path[rel]
                try:
                    sha, size, created, mtime, mtime_ns, inode = _hash_and_copy_object(
                        path, root, settle_cutoff=settle_cutoff, classify_sig=class_stat.get(rel),
                    )
                except _RaceDetected:
                    report["skipped_race"] += 1
                    continue
                except OSError as e:
                    logger.warning(f"uploads backup: could not read a new upload ({type(e).__name__}); skipping it this run")
                    report["copy_errors"] += 1
                    continue
                if created:
                    manifest["objects_count"] = manifest.get("objects_count", 0) + 1
                    manifest["objects_bytes"] = manifest.get("objects_bytes", 0) + size
                    report["bytes_copied"] += size
                magic = _magic_check(path)
                rec = {
                    "sha256": sha,
                    "size": size,
                    "mtime": mtime,
                    "mtime_ns": mtime_ns,
                    "inode": inode,
                    "first_seen": now_iso,
                    "last_seen": now_iso,
                    "deleted_at": None,
                }
                if magic is not None:
                    rec["suspect_content"] = magic
                    suspect_content_count += 1
                    logger.warning("uploads backup: a new upload's header does not match its extension (suspect_content recorded; counts only)")
                files[rel] = rec
                report["new"] += 1
                sleep(FILE_SLEEP_S)
                _tick(size)

            manifest["generation"] = manifest.get("generation", 0) + 1
            manifest["last_run_at"] = now_iso
            save_manifest(root, manifest)

            # hostile-review round 4, finding 3 + LOW: a nonzero copy_errors
            # or walk_errors means at least one live file could not be
            # backed up, or a whole subdirectory couldn't even be listed,
            # this run (unlike skipped_settling/skipped_race/
            # skipped_undecodable, which are all "deliberately not
            # touched, will resolve on its own") — that is a real, still-
            # open gap, so the run reports ok=False even though it
            # completed and saved the manifest.
            report["ok"] = report["copy_errors"] == 0 and report["walk_errors"] == 0
            report["suspect"] = suspect_active
            report["paused"] = busy_stop
            report["suspect_content"] = suspect_content_count
            report["elapsed_s"] = round(time.monotonic() - t0, 1)
            detail_bits = (
                f"new={report['new']} changed={report['changed']} "
                f"deleted={report['deleted']} expired={report['expired']} "
                f"bytes_copied={report['bytes_copied']}"
            )
            if report["resurrected"]:
                detail_bits += f" resurrected={report['resurrected']}"
            if report["skipped_undecodable"]:
                detail_bits += f" skipped_undecodable={report['skipped_undecodable']}"
            if report["skipped_race"]:
                detail_bits += f" skipped_race={report['skipped_race']}"
            if report["copy_errors"]:
                detail_bits += f" copy_errors={report['copy_errors']}"
            if report["walk_errors"]:
                detail_bits += f" walk_errors={report['walk_errors']}"
            if suspect_active:
                report["detail"] = detail
            elif busy_stop:
                report["detail"] = f"{detail_bits} (paused mid-run: pod became busy; resuming next cycle)"
            else:
                report["detail"] = detail_bits
            if report["ok"]:
                logger.info(f"uploads backup ok: {report['detail']} ({report['elapsed_s']}s)")
            else:
                logger.error(f"uploads backup completed with unbacked files: {report['detail']} ({report['elapsed_s']}s)")
            _write_status(
                root, ok=report["ok"],
                error_type=(None if report["ok"] else ("WalkErrors" if report["walk_errors"] else "CopyErrors")),
                run_at=now_iso, run_s=report["elapsed_s"],
                skipped_undecodable=report["skipped_undecodable"], skipped_race=report["skipped_race"],
                copy_errors=report["copy_errors"],
            )
            if not report["ok"]:
                # finding 3 / LOW: a nonzero copy_errors or walk_errors is
                # a real, still-open gap — surface it the same way any
                # other failed run does.
                _alert_uploads_failure("WalkErrors" if report["walk_errors"] else "CopyErrors")
            return report
    except _Refusal as e:
        # A deliberate, pre-formatted refusal (free space / mirror-size
        # cap) — see _Refusal's own docstring for why its message is safe
        # to keep, unlike the general case just below.
        report["detail"] = str(e)
        logger.error(f"uploads backup refused: {e}")
        _write_status(
            root, ok=False, error_type=type(e).__name__, run_at=_iso(time.time()), run_s=round(time.monotonic() - t0, 1),
            skipped_undecodable=report["skipped_undecodable"], skipped_race=report["skipped_race"],
            copy_errors=report["copy_errors"],
        )
        _alert_uploads_failure(type(e).__name__)
        return report
    except Exception as e:
        # hostile-review round 2, finding 8: never str(e) here — a
        # PermissionError/FileNotFoundError's message embeds the path that
        # triggered it, which can be (part of) her upload's own name. Only
        # the exception TYPE goes into the report/log; one bad file is
        # skipped elsewhere in this function specifically so it never has
        # to reach this catch-all in the first place.
        report["detail"] = f"{type(e).__name__} (message withheld: may contain a file path)"
        logger.error(f"uploads backup failed: {type(e).__name__}")
        _write_status(
            root, ok=False, error_type=type(e).__name__, run_at=_iso(time.time()), run_s=round(time.monotonic() - t0, 1),
            skipped_undecodable=report["skipped_undecodable"], skipped_race=report["skipped_race"],
            copy_errors=report["copy_errors"],
        )
        _alert_uploads_failure(type(e).__name__)
        return report


def _iso(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()


def _parse_iso(s: str) -> float | None:
    import datetime
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Called from backup.run_once (after the main backup cycle)
# ---------------------------------------------------------------------------


def run_after_main_backup(
    uploads_dir: Path | None = None,
    backup_root: Path | None = None,
    *,
    idle_check: Callable[[], tuple[bool, str]] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict:
    """The hook backup.run_daemon calls once per daemon cycle, after its own
    run_once() returns (see that module's own comment at the call site —
    hostile-review round 2, finding 7 moved this call OUT of backup.run_once
    itself and into the daemon loop only, so `backup.py --once`, `zl backup
    now` and zl fix's pre-repair backup no longer trigger it at all). A
    failure here is caught and logged by the caller and must never affect
    the main backup's report or its return value.

    `uploads_dir`/`idle_check`/`sleep` exist so a TEST can drive this
    exact same function without ever touching the real /data/openwebui/
    uploads directory or the real /health/full endpoint on :8080 — they
    default to None here and are only forwarded to run_once when a caller
    actually supplies them, so the real daemon (which supplies none of
    them) gets exactly run_once's own defaults, unchanged.
    """
    kwargs: dict = {}
    if idle_check is not None:
        kwargs["idle_check"] = idle_check
    if sleep is not None:
        kwargs["sleep"] = sleep
    return run_once(uploads_dir=uploads_dir, backup_root=backup_root, **kwargs)


# ---------------------------------------------------------------------------
# Health snapshot (for /health/full's `uploads_backup` key)
# ---------------------------------------------------------------------------

STATUS_NAME = "status.json"

_last_error: str | None = None
_last_run_at: float | None = None
_last_run_s: float | None = None


def _status_path(root: Path) -> Path:
    return root / STATUS_NAME


def _write_status(
    root: Path, *, ok: bool, error_type: str | None, run_at: str, run_s: float,
    skipped_undecodable: int = 0, skipped_race: int = 0, copy_errors: int = 0,
) -> None:
    """hostile-review round 3, finding 6: `note_run` below only ever
    updates THIS process's own module globals — fine for the daemon
    process that just ran a cycle, but /health/full is served by a
    DIFFERENT process (the compactor's own uvicorn worker) that never
    calls run_once or note_run itself, so those globals are always
    None/unset there, no matter how loudly the daemon's last cycle failed.
    A small, separately-written status.json (never inside manifest.json
    itself — this must always be writable even if something about the
    much larger main manifest write is what's currently broken) makes
    ok/last_error/last_run_at/last_run_s visible from ANY process that
    reads it, which is exactly what health_snapshot() does below.

    hostile-review round 4, finding 3: `degraded` is computed here by
    comparing this run's skipped_undecodable/skipped_race against the
    PREVIOUS status.json's own counts — the same nonzero count showing up
    two runs in a row means whatever is causing it (a persistently bad
    filename, a file that never stops moving) has not resolved on its own,
    which is worth a degraded flag even though neither counter alone fails
    the run. If the write itself fails, that must NOT be swallowed as a
    quiet warning (round 3's own bug) — a status.json an operator can't
    even trust to reflect the last run is itself a failure worth an ERROR
    and an alert, the same as any other failed run.
    """
    prev = _read_status(root) or {}
    degraded = bool(
        (skipped_undecodable and skipped_undecodable == prev.get("skipped_undecodable"))
        or (skipped_race and skipped_race == prev.get("skipped_race"))
    )
    try:
        _atomic_write_json(_status_path(root), {
            "ok": ok, "last_error": error_type, "last_run_at": run_at, "last_run_s": run_s,
            "skipped_undecodable": skipped_undecodable, "skipped_race": skipped_race,
            "copy_errors": copy_errors,
            "degraded": degraded,
        })
    except OSError as e:
        logger.error(f"uploads backup: could not write status.json ({type(e).__name__}) — this run's outcome will not be visible cross-process")
        _alert_uploads_failure(f"StatusWriteFailed({type(e).__name__})")


def _read_status(root: Path) -> dict | None:
    try:
        with open(_status_path(root), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None


def note_run(report: dict, elapsed_s: float) -> None:
    """backup.py calls this right after invoking run_after_main_backup, so
    /health/full can report the LAST run's own timing even though the
    backup daemon and the compactor's HTTP process are different processes
    (this module-level state is therefore only ever accurate from a
    process that actually called run_once itself — health_snapshot()
    below primarily reads the on-disk status.json now, which is visible
    cross-process; these globals are a same-process fallback only)."""
    global _last_error, _last_run_at, _last_run_s
    _last_run_at = time.time()
    _last_run_s = elapsed_s
    _last_error = None if report.get("ok") else report.get("detail")


def health_snapshot(backup_root: Path | None = None) -> dict:
    """For /health/full's `uploads_backup` key.

    hostile-review round 2, finding 3: this used to walk the ENTIRE
    objects/ tree with `rglob` on every call — cheap on a fresh mirror,
    NOT cheap once it holds real uploads, and the earlier docstring's claim
    that this was fine to call "from the request-serving process" was
    simply wrong once you account for that walk (this function has no
    async boundary of its own; a caller on the event loop blocks it for as
    long as the walk takes). objects_count/objects_bytes are now
    maintained incrementally IN the manifest by run_once (add on every new
    object, subtract on every sweep-removed one — see run_once and
    _remove_object), so this reads exactly one small JSON file. The
    caller-side fix — health.py now calls this from _gather_blocking, off
    the event loop, same as every other filesystem-touching health probe —
    is a second, independent layer of the same fix; either one alone would
    have been enough, and both are cheap insurance.
    """
    root = Path(backup_root) if backup_root is not None else UPLOADS_BACKUP_DIR
    # hostile-review round 3, finding 6: prefer the persisted, cross-
    # process status.json over these in-process globals, which are only
    # ever populated in whatever process actually called note_run (the
    # daemon) — never the compactor's own HTTP process serving this.
    status = _read_status(root)
    out = {
        "objects": 0, "bytes": 0, "files_tracked": 0, "deleted_tracked": 0,
        "last_run_at": (status or {}).get("last_run_at"),
        "last_run_s": (status or {}).get("last_run_s", _last_run_s),
        "suspect": False,
        "suspect_content_count": 0,
        "last_error": (status or {}).get("last_error", _last_error),
        "over_mirror_size_cap": False,
        # hostile-review round 4, finding 3:
        "skipped_undecodable": (status or {}).get("skipped_undecodable", 0),
        "skipped_race": (status or {}).get("skipped_race", 0),
        "degraded": bool((status or {}).get("degraded", False)),
        "stale": False,
    }
    try:
        manifest = load_manifest(root)
        files = manifest.get("files", {})
        out["files_tracked"] = sum(1 for r in files.values() if r.get("deleted_at") is None)
        out["deleted_tracked"] = sum(1 for r in files.values() if r.get("deleted_at") is not None)
        out["suspect_content_count"] = sum(1 for r in files.values() if r.get("suspect_content"))
        out["suspect"] = bool(manifest.get("last_suspect_run"))
        if out["last_run_at"] is None:
            out["last_run_at"] = manifest.get("last_run_at")
        out["objects"] = manifest.get("objects_count", 0)
        out["bytes"] = manifest.get("objects_bytes", 0)
        cap = MAX_MIRROR_MB * 1024 * 1024
        out["over_mirror_size_cap"] = bool(cap > 0 and out["bytes"] >= cap)
        if out["last_run_at"]:
            last_ts = _parse_iso(out["last_run_at"])
            if last_ts is not None:
                out["stale"] = (time.time() - last_ts) > (2 * BACKUP_INTERVAL_HOURS * 3600.0)
    except Exception as e:
        out["last_error"] = out["last_error"] or type(e).__name__
    return out


# ---------------------------------------------------------------------------
# CLI (manual path, per the design's "expose a manual path")
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse
    import logsetup

    logsetup.configure()
    p = argparse.ArgumentParser(description="Zion's Light AI uploads backup (manual run).")
    p.add_argument("--once", action="store_true", help="Run one mirror cycle now and exit.")
    p.add_argument("--ignore-idle", action="store_true", help="Skip the idle gate (manual, off-hours use).")
    args = p.parse_args(argv)
    if not args.once:
        p.print_help()
        return 2
    idle_check = (lambda: (True, "--ignore-idle")) if args.ignore_idle else is_idle
    report = run_once(idle_check=idle_check)
    print(f"[{'OK' if report['ok'] else 'FAIL'}] {report.get('detail', '')}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
