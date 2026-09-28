#!/usr/bin/env python3
"""Operator front-end to the uploads mirror (compactor/uploads_backup.py),
runnable directly on the pod without importing the compactor package.

    /opt/compactor-venv/bin/python scripts/uploads-backup.py status
    /opt/compactor-venv/bin/python scripts/uploads-backup.py verify
    /opt/compactor-venv/bin/python scripts/uploads-backup.py check-db --db /data/openwebui/webui.db
    /opt/compactor-venv/bin/python scripts/uploads-backup.py restore              # dry run
    /opt/compactor-venv/bin/python scripts/uploads-backup.py restore --apply

WHY A SEPARATE SCRIPT, NOT JUST `python -m compactor.uploads_backup`. The
same reason `scripts/backfill-records.py` and `scripts/clean-decoration.py`
are separate: this runs FROM A CLONED REPO CHECKOUT on the pod (see
`ops/bootstrap.sh`, `ZL_REPO`/`/opt/zl-repo`), against an OLDER installed
compactor, and must not depend on the compactor venv's own package layout
or on anything besides the Python standard library.

WHAT THIS NEVER DOES: print a file NAME or file CONTENT from her uploads.
Every command below reports COUNTS (and, for `check-db --list-ids`, numeric
OpenWebUI file ids on explicit request) — never the `<id>_<name>` strings
themselves. Tests exercise this script only against synthetic data.

Exit codes: 0 ok, 1 a problem was found or a destructive action was
refused, 2 usage error.

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import time
from pathlib import Path

DEFAULT_UPLOADS_DIR = "/data/openwebui/uploads"
DEFAULT_BACKUP_ROOT = "/data/backups/uploads"
DEFAULT_DB = "/data/openwebui/webui.db"
MANIFEST_NAME = "manifest.json"
OBJECTS_DIRNAME = "objects"
CHUNK_BYTES = 4 * 1024 * 1024
# Same default and same env var as compactor/uploads_backup.py's own
# SETTLE_S — read independently here (this script must not import the
# compactor package; see the module docstring) so `accept`'s roll-forward
# respects the same "may still be mid-write" doctrine run_once itself does.
SETTLE_S = float(os.environ.get("COMPACTOR_UPLOADS_BACKUP_SETTLE_S", "300"))


def _load_manifest(root: Path) -> dict:
    p = root / MANIFEST_NAME
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"generation": 0, "files": {}, "last_suspect_run": None, "objects_count": 0, "objects_bytes": 0}
    except (OSError, ValueError) as e:
        # hostile-review round 2, finding 8: never str(e) — swallow the
        # message, keep the type. `p` (the manifest.json path itself, not
        # any upload's name) is fine to print.
        print(f"error: manifest at {p} is unreadable ({type(e).__name__})", file=sys.stderr)
        return {"generation": 0, "files": {}, "last_suspect_run": None, "objects_count": 0, "objects_bytes": 0}
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        print(f"error: manifest at {p} has an unexpected shape", file=sys.stderr)
        return {"generation": 0, "files": {}, "last_suspect_run": None, "objects_count": 0, "objects_bytes": 0}
    return data


def _object_path(root: Path, sha256: str) -> Path:
    return root / OBJECTS_DIRNAME / sha256[:2] / sha256


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.backup_root)
    manifest = _load_manifest(root)
    files = manifest.get("files", {})
    tracked = sum(1 for r in files.values() if r.get("deleted_at") is None)
    deleted = sum(1 for r in files.values() if r.get("deleted_at") is not None)
    changed = sum(1 for r in files.values() if r.get("changed"))
    suspect_content = sum(1 for r in files.values() if r.get("suspect_content"))
    # hostile-review round 2, finding 3: read the manifest's own running
    # counters (kept in sync by every run_once cycle) instead of walking
    # objects/ here — this script is a manual, occasional operator command
    # so the walk was never the same event-loop-blocking bug health.py had,
    # but there is no reason to keep two different ways of counting the
    # same thing once one of them is already free.
    n_objects = manifest.get("objects_count", 0)
    total_bytes = manifest.get("objects_bytes", 0)
    suspect_run = manifest.get("last_suspect_run")
    print(f"backup root:        {root}")
    print(f"generation:         {manifest.get('generation', 0)}")
    print(f"last run at:        {manifest.get('last_run_at')}")
    print(f"files tracked:      {tracked}")
    print(f"files deleted:      {deleted} (kept, not yet expired)")
    print(f"files changed:      {changed} (content drift detected, old object kept)")
    print(f"suspect content:    {suspect_content}")
    print(f"objects on disk:    {n_objects}")
    print(f"bytes on disk:      {total_bytes}")
    if suspect_run:
        print(f"SUSPECT RUN:        {suspect_run}")
        print("recover with: uploads-backup.py accept --yes   (after you've confirmed it's legitimate)")
        return 1
    print("suspect run:        none")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.backup_root)
    manifest = _load_manifest(root)
    files = manifest.get("files", {})
    missing_objects = 0
    hash_mismatches = 0
    checked = 0
    for name, rec in files.items():
        sha = rec.get("sha256")
        if not sha:
            continue
        obj = _object_path(root, sha)
        if not obj.exists():
            missing_objects += 1
            continue
        checked += 1
        if args.deep:
            actual = _sha256_file(obj)
            if actual != sha:
                hash_mismatches += 1
    print(f"manifest entries:   {len(files)}")
    print(f"objects checked:    {checked}{' (deep)' if args.deep else ' (existence only)'}")
    print(f"missing objects:    {missing_objects}")
    print(f"hash mismatches:    {hash_mismatches}")
    ok = missing_objects == 0 and hash_mismatches == 0
    print("OK" if ok else "PROBLEM FOUND")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# check-db
# ---------------------------------------------------------------------------


def _webui_upload_rows(db_path: Path, *, immutable: bool = False) -> list[tuple[int, str]]:
    """[(id, path)] from OpenWebUI's `file` table, read-only. Tolerates a
    schema without a `file` table (an empty DB, or a very old one) by
    returning an empty list rather than raising — callers report that as
    zero rows, not as an error.

    hostile-review round 2, finding 10: `immutable=1` tells sqlite the file
    will NOT change for as long as this connection is open, which lets it
    skip locking and the usual "is there a newer WAL frame" checks — safe
    and a little faster for a DB that really is a static copy (e.g. one
    already extracted from a backup archive), and WRONG (can silently read
    stale or torn data) for the LIVE, currently-being-written database.
    `immutable` therefore defaults to False here: plain `mode=ro` against
    the live db is the safe default, and the one accepted trade-off is that
    sqlite may create `-shm`/`-wal` sidecar files next to it even for a
    read-only open (normal WAL-mode behavior, not a bug) — `--db-is-copy`
    opts into `immutable=1` for the (rarer, but real: an operator inspecting
    an already-extracted archive) case where that's known to be safe."""
    uri = f"file:{db_path}?mode=ro" + ("&immutable=1" if immutable else "")
    con = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        cur = con.cursor()
        try:
            cur.execute("SELECT id, path FROM file WHERE path IS NOT NULL")
        except sqlite3.OperationalError:
            return []
        return [(r[0], r[1]) for r in cur.fetchall() if r[1]]
    finally:
        con.close()


def cmd_check_db(args: argparse.Namespace) -> int:
    root = Path(args.backup_root)
    db_path = Path(args.db)
    uploads_dir = Path(args.uploads_dir)
    if not db_path.is_file():
        print(f"error: no such database: {db_path}", file=sys.stderr)
        return 2
    manifest = _load_manifest(root)
    files = manifest.get("files", {})
    by_name = {name: rec for name, rec in files.items()}

    rows = _webui_upload_rows(db_path, immutable=args.db_is_copy)
    on_disk = 0
    missing_disk = 0
    mirrored_ok = 0
    mirrored_missing = 0
    sha_mismatch = 0
    outside_uploads = 0
    bad_ids: list[int] = []
    for file_id, path_str in rows:
        p = Path(path_str)
        try:
            rel = p.relative_to(uploads_dir)
        except ValueError:
            # hostile-review round 2, finding 10: a DB path outside this
            # uploads dir (a different install) can't be checked against
            # THIS mirror at all — counted in its OWN bucket now, instead
            # of silently falling into "not mirrored" (a false positive
            # that made `ok` report a problem that wasn't one).
            outside_uploads += 1
            continue
        disk_ok = p.is_file()
        if disk_ok:
            on_disk += 1
        else:
            missing_disk += 1
        rec = by_name.get(str(rel))
        problem = False
        if rec is None:
            mirrored_missing += 1
            problem = True
        elif disk_ok:
            try:
                live_sha = _sha256_file(p)
            except OSError:
                live_sha = None
            if live_sha is not None and live_sha != rec.get("sha256"):
                sha_mismatch += 1
                problem = True
            else:
                mirrored_ok += 1
        else:
            mirrored_ok += 1  # on disk missing, but mirrored — restorable
        if problem:
            bad_ids.append(file_id)

    print(f"file-table rows:       {len(rows)}")
    print(f"outside uploads dir:   {outside_uploads} (not evaluated against this mirror)")
    print(f"on disk:               {on_disk}")
    print(f"missing from disk:     {missing_disk}")
    print(f"mirrored ok:           {mirrored_ok}")
    print(f"not mirrored:          {mirrored_missing}")
    print(f"sha mismatch:          {sha_mismatch}")
    if args.list_ids and bad_ids:
        print(f"ids with a problem:    {bad_ids}")
    ok = missing_disk == 0 and mirrored_missing == 0 and sha_mismatch == 0
    print("OK" if ok else "PROBLEM FOUND")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


def _quarantine_aside(path: Path, quarantine_dir: Path) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / path.name
    n = 1
    while dest.exists():
        dest = quarantine_dir / f"{path.name}.{n}"
        n += 1
    shutil.move(str(path), str(dest))


def _safe_dest(uploads_dir: Path, name: str) -> Path | None:
    """hostile-review round 2, finding 5: a manifest.json an attacker (or a
    bug) can write directly to disk is not a trusted input — a `name` like
    `../../etc/passwd` or an ABSOLUTE path must never be joined onto
    `uploads_dir` and used as a write target. Two things make `Path / str`
    unsafe on its own here: an absolute `name` makes `/` DISCARD the left
    side entirely (`Path("/a") / "/b" == Path("/b")`), and a relative `../`
    name resolves outside `uploads_dir` even though the join "looks"
    contained. Returns None (reject; the caller skips this record, counts
    it, never raises) unless the resolved destination is verifiably still
    inside `uploads_dir`."""
    if not name or os.path.isabs(name):
        return None
    try:
        uploads_resolved = uploads_dir.resolve()
        dest = (uploads_dir / name).resolve()
    except OSError:
        return None
    if not dest.is_relative_to(uploads_resolved):
        return None
    return dest


def cmd_restore(args: argparse.Namespace) -> int:
    root = Path(args.backup_root)
    uploads_dir = Path(args.uploads_dir)
    manifest = _load_manifest(root)
    files = manifest.get("files", {})

    # hostile-review round 3, finding 2: an accidentally-deleted upload is
    # kept (object + record) for COMPACTOR_UPLOADS_BACKUP_KEEP_DELETED_DAYS
    # (default 90) — OPERATIONS.md says this makes it "recoverable", but
    # until now there was no command that actually brought a DELETED record
    # back; `restore` always skipped them. `--include-deleted` is that
    # command; `--deleted-since` (an ISO-8601 timestamp) narrows it to
    # records deleted at or after a given time, for when only some of the
    # deleted names should come back.
    deleted_since = None
    if args.deleted_since:
        import datetime
        try:
            deleted_since = datetime.datetime.fromisoformat(args.deleted_since)
        except ValueError:
            print(f"error: --deleted-since is not a valid ISO-8601 timestamp: {args.deleted_since!r}", file=sys.stderr)
            return 2
        if deleted_since.tzinfo is None:
            # hostile-review round 4, LOW: every deleted_at this module
            # itself ever writes is timezone-AWARE UTC (_now_iso uses
            # datetime.timezone.utc) — comparing that against a naive
            # --deleted-since raises TypeError (a probe hit exactly this,
            # rc=0 with a bare traceback, no restore attempted and no
            # error message). Assume UTC for a naive input rather than
            # crash; this is what every OTHER timestamp in this tool means.
            deleted_since = deleted_since.replace(tzinfo=datetime.timezone.utc)

    # hostile-review round 2, finding 11: restoring a MISSING file (not
    # just a mismatched one) should also land with the uploads directory's
    # own ownership/mode, not whatever this process's umask/uid happens to
    # default to — there is no pre-existing `dest` to copy that from in the
    # "missing" case, so fall back to the directory's own stat.
    try:
        dir_stat = uploads_dir.stat()
    except OSError:
        dir_stat = None

    to_restore: list[tuple[str, dict, str]] = []  # (name, rec, reason)
    rejected_unsafe = 0
    for name, rec in files.items():
        if rec.get("deleted_at") is not None:
            if not args.include_deleted:
                continue  # not currently supposed to exist, and not asked for
            if deleted_since is not None:
                import datetime
                try:
                    deleted_at = datetime.datetime.fromisoformat(rec["deleted_at"])
                except (ValueError, TypeError):
                    continue
                if deleted_at < deleted_since:
                    continue
        dest = _safe_dest(uploads_dir, name)
        if dest is None:
            rejected_unsafe += 1
            continue
        sha = rec.get("sha256")
        if not sha:
            continue
        obj = _object_path(root, sha)
        if not obj.exists():
            continue  # nothing to restore FROM; verify/status already flag this
        if not dest.exists():
            to_restore.append((name, rec, "missing"))
        elif args.also_mismatched:
            try:
                live_sha = _sha256_file(dest)
            except OSError:
                live_sha = None
            if live_sha != sha:
                to_restore.append((name, rec, "mismatched"))

    n_missing = sum(1 for _, _, r in to_restore if r == "missing")
    n_mismatched = sum(1 for _, _, r in to_restore if r == "mismatched")
    print(f"would restore:         {len(to_restore)} (missing={n_missing}, mismatched={n_mismatched})")
    if rejected_unsafe:
        print(f"rejected (unsafe path): {rejected_unsafe} manifest entr{'y' if rejected_unsafe == 1 else 'ies'} "
              f"whose name did not resolve inside the uploads directory — skipped, not restored")

    if not to_restore:
        print("nothing to do")
        return 0

    if not args.apply:
        print("DRY RUN — pass --apply to actually restore")
        return 0

    stamp = _stamp()
    quarantine_dir = uploads_dir.parent / f"uploads-quarantine-{stamp}"
    restored = 0
    quarantined = 0
    errors = 0
    hash_mismatches = 0
    for name, rec, reason in to_restore:
        dest = _safe_dest(uploads_dir, name)
        if dest is None:
            continue  # re-checked; already counted above
        sha = rec["sha256"]
        obj = _object_path(root, sha)
        # hostile-review round 2, finding 1 (restore side): verify the
        # OBJECT's own hash before writing it anywhere — a damaged or
        # truncated object on disk must never silently overwrite a live
        # file with garbage. One bad object is skipped, not fatal.
        try:
            actual_sha = _sha256_file(obj)
        except OSError as e:
            errors += 1
            print(f"error reading one object before restore: {type(e).__name__}", file=sys.stderr)
            continue
        if actual_sha != sha:
            hash_mismatches += 1
            print("refusing to restore one file: its backed-up OBJECT does not match "
                  "the manifest's recorded hash (the object itself may be damaged)", file=sys.stderr)
            continue
        try:
            if reason == "mismatched" and dest.exists():
                _quarantine_aside(dest, quarantine_dir)
                quarantined += 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".uploads-restore.tmp")
            shutil.copyfile(obj, tmp)
            # Preserve mtime; owner+mode where permitted. Prefer the
            # pre-existing destination's own stat (an in-place restore of a
            # mismatched file); fall back to the uploads DIRECTORY's own
            # owner/mode for a "missing" file, which has no prior stat of
            # its own to copy (finding 11) — better than this process's
            # own default uid/umask.
            st = obj.stat()
            os.utime(tmp, (st.st_atime, rec.get("mtime", st.st_mtime)))
            try:
                orig_stat = dest.stat() if dest.exists() else dir_stat
                if orig_stat is not None:
                    os.chown(tmp, orig_stat.st_uid, orig_stat.st_gid)
                    os.chmod(tmp, stat.S_IMODE(orig_stat.st_mode))
            except (OSError, PermissionError):
                pass
            os.replace(tmp, dest)
            restored += 1
        except OSError as e:
            # hostile-review round 2, finding 8: type only — dest/obj are
            # paths built from her upload's own name.
            errors += 1
            print(f"error restoring one file: {type(e).__name__}", file=sys.stderr)

    print(f"restored:              {restored}")
    print(f"quarantined originals: {quarantined}" + (f" -> {quarantine_dir}" if quarantined else ""))
    print(f"object hash mismatch:  {hash_mismatches} (refused, not restored)")
    print(f"errors:                {errors}")
    return 0 if errors == 0 and hash_mismatches == 0 else 1


# ---------------------------------------------------------------------------
# accept — hostile-review round 2, finding 2: recovery from a CONFIRMED
# legitimate mass-change (the guard tripped, an operator looked at `status`
# and the live uploads directory and decided the deletion is real, not
# damage). Marks every still-vanished TRACKED name deleted_at=now (their
# objects are already kept, same durability as an ordinary single delete)
# and clears last_suspect_run. Never touches `changed` entries — those need
# no acceptance; they were never applied to the manifest in the first
# place while suspect (see run_once), so nothing about them needs undoing.
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: object) -> None:
    """Same crash-safe write pattern as compactor/uploads_backup.py's own
    helper of the same name (an independent copy — this script must not
    import the compactor package; see the module docstring)."""
    import tempfile
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


_LOCK_NAME = ".uploads-backup.lock"


class _locked:
    """hostile-review round 3, LOW: the SAME advisory flock the daemon's
    own run_once takes (compactor/uploads_backup.py's `locked()` — an
    independent copy here, same reasoning as _atomic_write_json above:
    this script must not import the compactor package). Without this, a
    daemon cycle and a concurrent `accept --yes` could interleave two
    read-modify-write manifest saves and silently drop one side's
    changes. BLOCKING, released unconditionally on any exit (including an
    exception), same as the daemon side."""

    def __init__(self, root: Path):
        self._root = root
        self._fd = None

    def __enter__(self):
        import fcntl
        self._root.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._root / _LOCK_NAME), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)


def _cur_sha(rec: dict) -> str | None:
    return rec.get("cur_sha256", rec.get("sha256"))


def _hash_and_copy_single_pass(path: Path, root: Path, *, settle_cutoff: float):
    """hostile-review round 4, finding 5: `accept`'s roll-forward used to
    hash a file during the (read-only, dry-run-safe) listing pass, then
    — only in the apply step, possibly much later — copy it under that
    ALREADY-COMPUTED hash via a separate, second read. A file that
    finished changing in between could be copied under a hash that no
    longer belongs to it: the exact hash-then-copy TOCTOU compactor/
    uploads_backup.py's own `_hash_and_copy_object` exists to close,
    reborn in this script's own roll-forward path. This is that same
    shape: ONE read pass computes the hash AND writes the object
    together, fstats before/after to catch an in-place rewrite landing
    inside the read, and refuses (returns None) a file whose mtime is
    still within the settle window — it may still be mid-write. Callers
    must call this only where writing is already appropriate (never
    during a dry run) and must use ITS return value as the record's new
    cur_sha256/cur_size/cur_*, never a hash computed earlier.

    Returns (sha256, size, created, mtime, mtime_ns, inode), or None if
    the file was skipped (too fresh, or moved during the read)."""
    dest_dir = root / OBJECTS_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    import tempfile
    fd, tmp_path = tempfile.mkstemp(dir=str(dest_dir), prefix=".hashcopy.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as inp:
            pre = os.fstat(inp.fileno())
            if pre.st_mtime > settle_cutoff:
                os.unlink(tmp_path)
                return None
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
            return None
        sha = h.hexdigest()
        dest = _object_path(root, sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        created = not dest.is_file()
        if created:
            os.replace(tmp_path, dest)
        else:
            os.unlink(tmp_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return sha, size, created, pre.st_mtime, pre.st_mtime_ns, pre.st_ino


def cmd_accept(args: argparse.Namespace) -> int:
    root = Path(args.backup_root)
    uploads_dir = Path(args.uploads_dir)

    # hostile-review round 3, finding 4: a missing or EMPTY --uploads-dir
    # makes every single tracked file look "vanished" — accept would then
    # mark the entire mirror deleted on a simple typo/misconfiguration.
    # Refuse outright rather than guess.
    if not uploads_dir.is_dir():
        print(f"error: --uploads-dir does not exist or is not a directory: {uploads_dir}", file=sys.stderr)
        return 2
    current_names = set()
    for dirpath, _dirnames, filenames in os.walk(uploads_dir, followlinks=False):
        for fname in filenames:
            p = Path(dirpath) / fname
            try:
                st = p.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            current_names.add(str(p.relative_to(uploads_dir)))
    if not current_names:
        print(f"error: --uploads-dir has no regular files at all ({uploads_dir}) — "
              f"refusing to treat every tracked upload as vanished", file=sys.stderr)
        return 2

    with _locked(root):
        manifest = _load_manifest(root)
        files = manifest.get("files", {})

        tracked = {n: r for n, r in files.items() if r.get("deleted_at") is None}
        vanished = sorted(n for n in tracked if n not in current_names)
        # hostile-review round 3, finding 3: `accept` must also roll the
        # CHANGED files' cur_* fields forward to the live content — a
        # legitimate mass content-edit the operator is accepting stays
        # "changed" (and keeps re-tripping the guard on the very next run)
        # forever otherwise, since cur_sha256 was never advanced while
        # frozen (see run_once: changed_content is computed every run but
        # only APPLIED to `files` when not suspect).
        to_roll_forward: list[tuple[str, str, int, float, int, int]] = []  # (name, sha, size, mtime, mtime_ns, inode)
        for n in sorted(current_names & set(tracked)):
            rec = tracked[n]
            live_path = uploads_dir / n
            try:
                live_st = live_path.stat()
            except OSError:
                continue
            try:
                live_sha = _sha256_file(live_path)
            except OSError:
                continue
            if live_sha != _cur_sha(rec):
                to_roll_forward.append((n, live_sha, live_st.st_size, live_st.st_mtime, int(live_st.st_mtime_ns), live_st.st_ino))

        suspect_run = manifest.get("last_suspect_run")
        affected = len(vanished) + len(to_roll_forward)
        baseline = len(tracked)
        pct = (100.0 * affected / baseline) if baseline else 0.0

        # hostile-review round 4, finding 4: `baseline` above is `len(
        # tracked)` computed NOW — inflated by every new file the guard
        # kept mirroring while suspect (finding 2 of round 3: new uploads
        # are never withheld). A gzip-then-accept probe showed this let a
        # 100%-at-trip-time incident recompute as exactly 50% by the time
        # `accept` ran (20 vanished + 20 newly-mirrored .gz names = 40
        # tracked), sliding UNDER the >50% refusal with no --force-large
        # needed. Gate on max(this run's own pct, the marker's OWN
        # recorded pct, affected against the marker's ORIGINAL baseline)
        # — whichever is largest wins, so a diluted recompute can never
        # under-state how big the original incident actually was.
        gate_pct = pct
        if suspect_run:
            marker_pct = suspect_run.get("pct")
            if isinstance(marker_pct, (int, float)):
                gate_pct = max(gate_pct, marker_pct)
            marker_baseline = suspect_run.get("baseline")
            if isinstance(marker_baseline, (int, float)) and marker_baseline:
                gate_pct = max(gate_pct, 100.0 * affected / marker_baseline)

        print(f"currently vanished tracked files:      {len(vanished)}")
        print(f"changed files to roll forward:         {len(to_roll_forward)}")
        print(f"suspect flag set:                      {bool(suspect_run)}")
        if suspect_run:
            # hostile-review round 3, finding 4: print the marker's own
            # recorded numbers before requiring --yes, so an operator
            # confirms against what actually tripped the guard, not just
            # what accept re-derived just now.
            print(f"  last_suspect_run.vanished:            {suspect_run.get('vanished')}")
            print(f"  last_suspect_run.changed:             {suspect_run.get('changed')}")
            print(f"  last_suspect_run.baseline:            {suspect_run.get('baseline')}")
            print(f"  last_suspect_run.pct:                 {suspect_run.get('pct')}")
            print(f"  last_suspect_run.detected_at:         {suspect_run.get('detected_at')}")
        if not vanished and not to_roll_forward and not suspect_run:
            print("nothing to accept")
            return 0

        # hostile-review round 3, finding 4: more than half of tracked
        # files affected is exactly the shape a wrong (but not empty, not
        # missing) --uploads-dir produces — e.g. pointed one directory up,
        # or at a stale copy — refuse without --force-large. (round 4:
        # gated on gate_pct, not the possibly-diluted `pct`.)
        if gate_pct > 50.0 and not args.force_large:
            print(f"error: {affected}/{baseline} tracked files ({pct:.0f}% by current count, "
                  f"{gate_pct:.0f}% against the original incident) would be affected — over 50%. "
                  f"Re-check --uploads-dir, or pass --force-large if this is really intended.", file=sys.stderr)
            return 1

        if not args.yes:
            print("DRY RUN — pass --yes to apply" + (" (and --force-large, given the size above)" if gate_pct > 50.0 else ""))
            return 0

        now_iso = _now_iso()
        for n in vanished:
            files[n]["deleted_at"] = now_iso
        # hostile-review round 4, finding 5: re-hash-and-copy each file in
        # ONE pass, right now, at apply time — never reuse the hash the
        # earlier (read-only, dry-run-safe) listing pass computed, and
        # respect the settle window here too (a file that started
        # changing again between listing and apply is skipped, not
        # captured half-written).
        settle_cutoff = time.time() - SETTLE_S
        rolled_forward = 0
        failed_roll_forward = []
        for n, _stale_sha, _stale_size, _stale_mtime, _stale_mtime_ns, _stale_inode in to_roll_forward:
            live_path = uploads_dir / n
            try:
                result = _hash_and_copy_single_pass(live_path, root, settle_cutoff=settle_cutoff)
            except OSError:
                # unreadable right now; leave the record as-is, retry next
                # time — but this IS a roll-forward failure: the accept the
                # operator asked for did not fully happen, so the suspect
                # marker below must NOT be cleared for it (see the
                # `failed_roll_forward` check just after this loop).
                failed_roll_forward.append(n)
                continue
            if result is None:
                # too fresh, or moved mid-read; leave the record as-is —
                # same reasoning: not actually rolled forward yet.
                failed_roll_forward.append(n)
                continue
            sha, size, created, mtime, mtime_ns, inode = result
            if created:
                manifest["objects_count"] = manifest.get("objects_count", 0) + 1
                manifest["objects_bytes"] = manifest.get("objects_bytes", 0) + size
            rec = files[n]
            rec["cur_sha256"] = sha
            rec["cur_size"] = size
            rec["cur_mtime"] = mtime
            rec["cur_mtime_ns"] = mtime_ns
            rec["cur_inode"] = inode
            rec["last_seen"] = now_iso
            rolled_forward += 1
        # .9 fix (round-4 open observation): `accept` used to clear
        # last_suspect_run unconditionally here, even when one or more
        # roll-forward writes above failed (unreadable right now, or too
        # fresh / moved mid-read) — silently leaving the mirror NOT fully
        # accepted while reporting the flag cleared and exiting 0. Clear
        # the marker ONLY when every roll-forward this call attempted
        # actually succeeded; otherwise report which names failed, keep
        # `last_suspect_run` set (still-vanished names are still marked
        # deleted, and every file that DID roll forward above is still
        # saved — only the marker-clear is withheld), and exit 1 so an
        # operator or script notices and retries rather than believing the
        # mirror is clean again.
        if failed_roll_forward:
            _atomic_write_json(root / MANIFEST_NAME, manifest)
            print(f"error: {len(failed_roll_forward)} changed file(s) could not be rolled forward "
                  f"this run (unreadable, or still settling/mid-write) — suspect flag NOT cleared; "
                  f"rolled forward {rolled_forward}/{len(to_roll_forward)}, marked "
                  f"{len(vanished)} file(s) deleted_at=now. Re-run accept once the files settle.",
                  file=sys.stderr)
            return 1
        manifest["last_suspect_run"] = None
        _atomic_write_json(root / MANIFEST_NAME, manifest)
        print(f"accepted: marked {len(vanished)} file(s) deleted_at=now, rolled forward "
              f"{rolled_forward} changed file(s); suspect flag cleared")
        return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup-root", default=DEFAULT_BACKUP_ROOT, help=f"default {DEFAULT_BACKUP_ROOT}")
    p.add_argument("--uploads-dir", default=DEFAULT_UPLOADS_DIR, help=f"default {DEFAULT_UPLOADS_DIR}")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("status", help="counts, bytes, last run, suspect flags")
    s.set_defaults(func=cmd_status)

    v = sub.add_parser("verify", help="every manifest entry has an object")
    v.add_argument("--deep", action="store_true", help="also re-hash each object (slower)")
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("check-db", help="compare webui.db's file table against the mirror")
    c.add_argument("--db", default=DEFAULT_DB, help=f"default {DEFAULT_DB}")
    c.add_argument("--list-ids", action="store_true", help="also print numeric file ids with a problem")
    c.add_argument("--db-is-copy", action="store_true",
                   help="the DB at --db is a static copy (already extracted from a backup), "
                        "not the live database — opens with sqlite's immutable=1 for a faster, "
                        "safe read. Do NOT pass this against the live webui.db.")
    c.set_defaults(func=cmd_check_db)

    r = sub.add_parser("restore", help="restore missing (or mismatched) uploads from the mirror")
    r.add_argument("--apply", action="store_true", help="actually restore (default: dry run)")
    r.add_argument("--also-mismatched", action="store_true",
                   help="also restore uploads whose live content no longer matches the mirror "
                        "(the existing file is quarantined aside first, never overwritten in place)")
    r.add_argument("--include-deleted", action="store_true",
                   help="also restore uploads the mirror has marked deleted (kept for "
                        "COMPACTOR_UPLOADS_BACKUP_KEEP_DELETED_DAYS specifically so this is possible)")
    r.add_argument("--deleted-since", metavar="ISO",
                   help="with --include-deleted, only restore deleted records whose deleted_at "
                        "is at or after this ISO-8601 timestamp")
    r.set_defaults(func=cmd_restore)

    a = sub.add_parser("accept", help="accept a suspect run: mark still-vanished tracked files "
                                       "deleted, roll forward accepted content changes, and clear "
                                       "the suspect flag")
    a.add_argument("--yes", action="store_true", help="actually apply (default: dry run)")
    a.add_argument("--force-large", action="store_true",
                   help="required in addition to --yes when more than half of tracked files "
                        "would be affected — a safety net against a wrong/empty --uploads-dir")
    a.set_defaults(func=cmd_accept)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "command", None):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
