#!/usr/bin/env python3
"""pod-migrate.py: move her data from the OLD pod's tree to the NEW pod's tree (v3.1.9.10).

Runs ON THE NEW POD, which sees both trees on the shared network volume:

    OLD pod   DATA_DIR=/data/openwebui        (WEBUI_DB_LOCAL=false, live DB on /data)
    NEW pod   DATA_DIR=/data/v10/openwebui    (WEBUI_DB_LOCAL=true,  live DB on local disk)

Stdlib only. Every command is a DRY RUN unless --apply is given. Nothing here ever
writes the SOURCE tree (the SQLite source is opened `mode=ro`; files are only read).
Logs carry COUNTS, table names and top-level directory names only: never an upload
file name, a conversation id or any content.

    pod-migrate.py seed    --from /data/openwebui --to /data/v10/openwebui [--apply]
    pod-migrate.py cutover --from /data/openwebui --to /data/v10/openwebui --old-pod-stopped [--apply]
    pod-migrate.py verify  --from /data/openwebui --to /data/v10/openwebui
    pod-migrate.py status  --to /data/v10/openwebui [--require cutover]
    pod-migrate.py cutover --reverse --from /data/v10/openwebui --to /data/openwebui --old-pod-stopped [--apply]

seed      A consistent copy of webui.db (sqlite3 backup API, read-only source, busy
          timeout + retries: the old pod stays live), the compactor store, its
          control/pending siblings, uploads, OpenWebUI's vector_db and the word-ban
          file. Safe to re-run: it is an incremental refresh. Writes a manifest to
          <to>/.migrate/seed.json.
cutover   The final resync, with the OLD pod frozen: a fresh DB copy, a delta of the
          trees (deleted-in-source files are MOVED to <to>/.migrate/removed-<stamp>/,
          never removed), and the LOCAL live DB on this pod REPLACED (see
          install_db(): a healthy stale local DB would otherwise win on the next boot
          and webuidb-sync would publish it over the fresh snapshot).
verify    Read-only comparison of the two trees and both databases; counts only;
          exits 1 on any mismatch. SQLite files inside the trees (chroma) are compared
          by a hash of their LOGICAL content, not by size or name.
status    Reads <to>/.migrate/seed.json only: COMPLETE or INCOMPLETE, the last step, whether
          the cutover's own verify passed. Exit 0 only for COMPLETE (a cutover needs a
          passing verify). The runbook starts no service unless this says COMPLETE.

SAFETY RAILS (v3.1.9.10 fix round B; each is a hostile-review finding)
  * The database this tool records is the one it INSTALLED (sha256 + local path, in the
    manifest). A later seed/cutover REFUSES when the new pod's live local DB no longer has
    that hash (the new pod has written), unless --discard-new-pod-writes, which prints
    what will be moved aside first. Nothing is ever deleted.
  * --reverse copies the new pod's LIVE LOCAL DB (read through the SQLite backup API,
    the services being stopped), not the snapshot on the volume: the snapshot can be
    minutes behind. It refuses if the local DB is missing.
  * cutover does not trust size+mtime: every non-SQLite file is hashed on both sides,
    every SQLite file (chroma, vector_db) is re-copied through the backup API.
  * A live seed copies in 4096-page steps with a short pause (total pause capped), retries
    a copy whose integrity_check fails (bounded), and when the source is written under the
    copy (every write restarts the backup API) escalates: fast steps with no pause, then
    one single full copy (one short read lock; the old pod's busy_timeout is 5 s). It
    reports the mode, the restarts and the longest read-lock hold.
  * SIGHUP is ignored; SIGINT/SIGTERM finish the step that must not be torn (the database
    install), stop, leave the manifest complete=false and exit 1.

THE UPLOAD PATHS. OpenWebUI stores an ABSOLUTE path per uploaded file
(`file.path` = <DATA_DIR>/uploads/<id>_<name>) and serves the file from exactly that
path. Copying the DB unchanged would leave the new pod reading (and depending on) the
OLD pod's uploads. So the copy of the DB is rewritten: `<from>/uploads/` becomes
`<to>/uploads/` in file.path (the target copy only; the source is never touched).

THE SECRET KEY. This image never reads `<DATA_DIR>/.webui_secret_key`: entrypoint.sh
always exports WEBUI_SECRET_KEY, and open-webui's own key file is cwd-relative
(/app/.webui_secret_key), used only when the variable is unset. Sessions therefore
carry over only if the NEW pod's WEBUI_SECRET_KEY env value equals the old pod's (see
RUNBOOK_POD_MIGRATION.md). --with-secret-key additionally copies a
<from>/.webui_secret_key if one exists (mode 0600); by default it is not copied.

EXIT CODES
    0   done (or, for a dry run: the same command with --apply would proceed)
    1   a step failed, a check found a mismatch, or the source changed during the run
    2   REFUSED before touching anything (a precondition failed) or bad usage
"""

from __future__ import annotations

import argparse
import datetime
import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _podpaths  # noqa: E402  (this pod's paths, not another pod's)

TOOL = "pod-migrate"
MANIFEST_DIR = ".migrate"
MANIFEST_NAME = "seed.json"
LOCK_NAME = "lock"
MANIFEST_FORMAT = 2   # 2 (fix round 2): pending_install + the go-live guard. An older manifest is REFUSED, never adopted

HER_CONV_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"

# Top-level entries of DATA_DIR that are copied as trees. Anything else in DATA_DIR
# (cache, audit.log, webui.db.bak-*, ...) is regenerable or a leftover and is only
# counted, never copied.
BASE_TREES = ("compactor", "compactor-control", "compactor-pending", "uploads", "vector_db")
TREES = BASE_TREES  # main() adds "cache" for --with-cache
SQLITE_SUFFIXES = (".sqlite3", ".sqlite")
SKIP_SUFFIXES = ("-journal", "-wal", "-shm", ".tmp", ".sync-tmp", ".migrate-tmp")
SIDECARS = ("-journal", "-wal", "-shm")

# supervisord programs on the NEW pod that read or write the data this tool replaces.
GUARDED_PROGRAMS = ("openwebui", "compactor", "webuidb-sync", "backup")
RUNNING_STATES = {"RUNNING", "STARTING", "BACKOFF", "STOPPING", "UNKNOWN"}
STOPPED_STATES = {"STOPPED", "EXITED", "FATAL"}
GUARDED_PORTS = (3000, 8080)  # OpenWebUI, compactor (loopback)

EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2


class Refused(Exception):
    """A precondition failed. Nothing was changed. Exit 2."""


class Failed(Exception):
    """A step failed or a check found a mismatch. Exit 1."""


class Interrupted(Exception):
    """SIGINT/SIGTERM arrived. The current unsafe-to-tear step was finished first. Exit 1."""


def say(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except OSError:
        pass  # the terminal went away (SIGHUP): the run must not die of a failed print


def note(msg: str) -> None:
    say(f"{TOOL}: {msg}")


_SIG = {"critical": 0, "pending": None}


def _on_signal(signum, _frame) -> None:
    if _SIG["critical"] > 0:
        _SIG["pending"] = signum  # honoured the moment the critical step ends
        return
    raise Interrupted(signal.Signals(signum).name)


@contextlib.contextmanager
def critical():
    """A step that must not be torn (replacing the live database). A signal that
    arrives inside is remembered and raised right after the step completes."""
    _SIG["critical"] += 1
    try:
        yield
    finally:
        _SIG["critical"] -= 1
        if _SIG["critical"] == 0 and _SIG["pending"] is not None:
            signum, _SIG["pending"] = _SIG["pending"], None
            raise Interrupted(signal.Signals(signum).name)


def install_signal_handlers():
    """Ignore SIGHUP (a closed terminal), handle SIGINT/SIGTERM. Returns a restore
    callable. A no-op outside the main thread."""
    saved = {}
    try:
        for sig, handler in ((signal.SIGHUP, signal.SIG_IGN), (signal.SIGINT, _on_signal), (signal.SIGTERM, _on_signal)):
            saved[sig] = signal.signal(sig, handler)
    except ValueError:  # not the main thread
        return lambda: None
    _SIG["critical"], _SIG["pending"] = 0, None

    def restore():
        for sig, old in saved.items():
            signal.signal(sig, old)
    return restore


def err_tag(e: BaseException) -> str:
    """An exception reduced to something safe to print: never its message (it can
    carry a file name)."""
    en = getattr(e, "errno", None)
    return type(e).__name__ + (f" errno={en}" if en else "")


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def is_skipped_name(name: str) -> bool:
    return name.endswith(SKIP_SUFFIXES) or name.startswith(".hashcopy.")


def is_sqlite_name(name: str) -> bool:
    return name.endswith(SQLITE_SUFFIXES)


def walk_files(root: Path) -> tuple[dict[str, os.stat_result], dict[str, int]]:
    """{relpath: stat} of every regular, non-temporary file under root, plus counts
    of what was skipped. Symlinks and special files are skipped, never followed."""
    files: dict[str, os.stat_result] = {}
    skipped = {"symlink": 0, "special": 0, "temp": 0}
    if not root.is_dir():
        return files, skipped
    stack = [root]
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                try:
                    if e.is_symlink():
                        skipped["symlink"] += 1
                    elif e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    elif e.is_file(follow_symlinks=False):
                        if is_skipped_name(e.name):
                            skipped["temp"] += 1
                        else:
                            files[os.path.relpath(e.path, root).replace(os.sep, "/")] = e.stat(follow_symlinks=False)
                    else:
                        skipped["special"] += 1
                except FileNotFoundError:
                    continue  # vanished between listing and stat: a live source
    return files, skipped


def copy_file(src: Path, dst: Path, tries: int = 3) -> tuple[str, os.stat_result]:
    """Copy src to dst (temp name + fsync + rename), preserving mtime and mode.
    Returns (sha256 of what was written, source stat). Retries when the source
    changed while it was being read; raises Failed if it will not hold still."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(tries):
        st0 = os.stat(src)
        tmp = dst.with_name(dst.name + ".migrate-tmp")
        h = hashlib.sha256()
        try:
            with open(src, "rb") as fin, open(tmp, "wb") as fout:
                for block in iter(lambda: fin.read(1 << 20), b""):
                    h.update(block)
                    fout.write(block)
                fout.flush()
                os.fsync(fout.fileno())
            st1 = os.stat(src)
            if (st1.st_size, st1.st_mtime_ns) != (st0.st_size, st0.st_mtime_ns):
                tmp.unlink(missing_ok=True)
                time.sleep(0.2)
                continue
            shutil.copymode(src, tmp)
            os.utime(tmp, ns=(st0.st_atime_ns, st0.st_mtime_ns))
            os.replace(tmp, dst)
            fsync_dir(dst.parent)
            return h.hexdigest(), st0
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    raise Failed("a source file kept changing while being read")


# --------------------------------------------------------------------------------
# sqlite
# --------------------------------------------------------------------------------

def ro_uri(path: Path) -> str:
    return "file:" + urllib.parse.quote(str(path)) + "?mode=ro"


class _BadCopy(Exception):
    """The copy finished but failed integrity_check: retried like a busy error."""


class _Restarting(Exception):
    """The source was written under the copy, again and again: every write restarts the
    backup API from page 1. Not an error: the next attempt is a faster one."""


FAST_PAGES = 4096          # 16 MB per step at 4 KiB pages: a step that takes tens of ms
PAUSE_BUDGET_S = 20.0      # the most a gentle attempt may spend pausing between steps, in total
SINGLE_MAX_S = 3.0         # the single full copy is only tried if its read lock is projected to be held <= this.
                           # The old pod's busy_timeout is 5 s: a longer hold makes HER write fail with "database is locked"


def _mode_for(attempt: int, retries: int) -> str:
    """gentle (paced) first; then fast steps with no pause; the LAST attempt is a single
    full copy (pages=-1: one read lock, no restart is possible, held for the whole copy)."""
    if attempt == 1:
        return "gentle"
    if attempt == retries and retries >= 3:
        return "single"
    return "fast"


def sqlite_backup_copy(src: Path, dst: Path, *, busy_s: float = 60.0, retries: int = 5,
                       retry_wait_s: float = 5.0, step_pages: int = FAST_PAGES, step_sleep_s: float = 0.0,
                       max_s: float = 3600.0, pause_budget_s: float = PAUSE_BUDGET_S,
                       single_max_s: float = SINGLE_MAX_S) -> dict:
    """A consistent copy of a live or frozen SQLite database via the backup API.
    The source is opened read-only (`mode=ro`); the destination is a private file
    (callers pass a LOCAL temp path, never the network volume). Retried (bounded by
    `retries`) on busy/locked/IO errors, when the finished copy fails integrity_check,
    and when the source keeps being written under the copy.

    WHY THE LADDER (v3.1.9.10 fix round 2). Every write to the source restarts the backup
    API from page 1. A paced copy of a 560 MB database takes ~2 minutes, and an idle open
    OpenWebUI tab writes last_active_at once a minute, so a paced copy NEVER finished. So:
      attempt 1  "gentle": `step_pages` per step (>= 4096 by default) with `step_sleep_s`
                 between steps, the total pause bounded by `pause_budget_s`;
      middle     "fast": 4096+ pages per step, NO pause. A step is tens of ms, so the copy
                 is short and rarely meets a write; 2 restarts in one attempt end it;
      last       "single": pages=-1, one step, one read lock for the whole copy (a few
                 seconds for 560 MB): nothing can restart it. The old pod's busy_timeout is 5 s, so it
                 is only tried when the throughput measured on the earlier attempts projects a hold of
                 at most `single_max_s` (3 s); otherwise the copy fails and says to close the app.
    The pause between steps is taken in the progress callback: sqlite3's own `sleep`
    argument only applies to BUSY/LOCKED results.
    Returns {"attempts", "seconds", "mode", "restarts", "max_step_s"} where max_step_s is the
    longest time the source's read lock was held in one step (the callback gaps, an upper bound)."""
    last: BaseException | None = None
    t_all = time.monotonic()
    restarts_total = 0
    meas = {"bytes": 0.0, "secs": 0.0, "total_bytes": 0.0}   # throughput of the steps seen so far
    for attempt in range(1, retries + 1):
        dst.unlink(missing_ok=True)
        started = time.monotonic()
        mode = _mode_for(attempt, retries)
        pages = -1 if mode == "single" else (step_pages if mode == "gentle" else max(step_pages, FAST_PAGES))
        sleep_s = step_sleep_s if mode == "gentle" else 0.0
        restart_limit = 1 if mode == "gentle" else 2
        st = {"restarts": 0, "last_remaining": None, "paused": 0.0, "max_step": 0.0, "mark": time.monotonic()}
        s = d = None
        try:
            s = sqlite3.connect(ro_uri(src), uri=True, timeout=busy_s)
            s.execute(f"PRAGMA busy_timeout={int(busy_s * 1000)}")
            d = sqlite3.connect(str(dst), timeout=busy_s)
            psize = s.execute("PRAGMA page_size").fetchone()[0]
            if mode == "single" and meas["secs"] > 0 and meas["bytes"] > 0:
                projected = meas["total_bytes"] / (meas["bytes"] / meas["secs"])
                if projected > single_max_s:
                    note(f"not trying the single full copy: it would hold the old pod's read lock for about "
                         f"{projected:.1f} s (limit {single_max_s:.0f} s; its busy_timeout is 5 s)")
                    last = _Restarting()
                    continue

            def progress(status, remaining, total, _t0=started, _st=st, _sleep=sleep_s, _limit=restart_limit):
                now = time.monotonic()
                _st["max_step"] = max(_st["max_step"], now - _st["mark"])
                meas["total_bytes"] = float(total) * psize
                if _st["last_remaining"] is not None and remaining < _st["last_remaining"] and now - _st["mark"] > 0:
                    meas["bytes"] += (_st["last_remaining"] - remaining) * psize
                    meas["secs"] += now - _st["mark"]
                if now - _t0 > max_s:
                    raise Failed("the database copy took too long")
                if _st["last_remaining"] is not None and remaining >= _st["last_remaining"]:
                    _st["restarts"] += 1     # the source was written: sqlite started over from page 1
                    if _st["restarts"] >= _limit:
                        raise _Restarting()
                _st["last_remaining"] = remaining
                if _sleep > 0 and _st["paused"] < pause_budget_s:
                    dt = min(_sleep, pause_budget_s - _st["paused"])
                    time.sleep(dt)
                    _st["paused"] += dt
                _st["mark"] = time.monotonic()

            if pages == -1:
                s.backup(d)
                st["max_step"] = time.monotonic() - started
            else:
                s.backup(d, pages=pages, progress=progress, sleep=0.05)
            d.commit()
            ok = [r[0] for r in d.execute("PRAGMA integrity_check")]
            if ok != ["ok"]:
                raise _BadCopy()
            return {"attempts": attempt, "seconds": time.monotonic() - t_all, "mode": mode,
                    "restarts": restarts_total + st["restarts"], "max_step_s": st["max_step"]}
        except Failed:
            raise
        except _Restarting as e:
            last = e
            restarts_total += st["restarts"]
            note(f"database copy attempt {attempt}/{retries} ({mode}): the source was written under the copy "
                 f"{st['restarts']}x (someone is using the old pod); "
                 f"{'retrying with ' + _mode_for(attempt + 1, retries) + ' steps' if attempt < retries else 'giving up'}")
        except (sqlite3.Error, _BadCopy) as e:
            last = e
            what = "the copy failed integrity_check" if isinstance(e, _BadCopy) else type(e).__name__
            note(f"database copy attempt {attempt}/{retries} failed ({what}); "
                 f"{'retrying' if attempt < retries else 'giving up'}")
            if attempt < retries:
                time.sleep(retry_wait_s)
        finally:
            for c in (d, s):
                try:
                    if c is not None:
                        c.close()
                except sqlite3.Error:
                    pass
    dst.unlink(missing_ok=True)
    if isinstance(last, _BadCopy):
        raise Failed(f"the copied database failed integrity_check on all {retries} attempts "
                     f"({time.monotonic() - t_all:.0f} s): is her chat active on the source? retry when she is idle")
    if isinstance(last, _Restarting):
        raise Failed(f"the source database kept changing under the copy on all {retries} attempts "
                     f"({time.monotonic() - t_all:.0f} s): she is using the old pod. Have her close the app on every device")
    raise Failed(f"could not copy the database ({type(last).__name__ if last else 'error'})")


def sqlite_logical_hash(path: Path) -> str:
    """A hash of a SQLite file's LOGICAL content (schema text, then every row of every
    table in scan order), independent of page layout, change counters and file size.
    Read-only. A backup-API copy of a database hashes equal to its source; a copy whose
    content differs does not, even at the same size and modification time."""
    h = hashlib.sha256()
    try:
        c = open_ro(path, busy_s=30)
    except sqlite3.Error as e:
        return "unreadable:" + type(e).__name__
    try:
        for name, sql in c.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            h.update(f"T\0{name}\0{sql}\0".encode("utf-8", "replace"))
            if sql and sql.upper().startswith("CREATE VIRTUAL"):
                continue  # shadow tables carry the content and are hashed on their own
            try:
                cur = c.execute(f'SELECT * FROM "{name}"')
                while True:
                    rows = cur.fetchmany(5000)
                    if not rows:
                        break
                    for r in rows:
                        h.update(repr(r).encode("utf-8", "replace"))
            except sqlite3.Error as e:
                h.update(("ERR:" + type(e).__name__).encode())
    except sqlite3.Error as e:
        return "unreadable:" + type(e).__name__
    finally:
        c.close()
    return h.hexdigest()


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
        try:
            out[name] = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except sqlite3.Error:
            out[name] = -1
    return out


def open_ro(path: Path, busy_s: float = 60.0) -> sqlite3.Connection:
    c = sqlite3.connect(ro_uri(path), uri=True, timeout=busy_s)
    c.execute(f"PRAGMA busy_timeout={int(busy_s * 1000)}")
    return c


def rewrite_file_paths(db: Path, maps: list[tuple[str, str]]) -> dict[str, int]:
    """In a PRIVATE copy of webui.db: rewrite file.path prefixes. Returns counts."""
    counts = {"rewritten": 0, "file_rows": 0}
    c = sqlite3.connect(str(db))
    try:
        cols = {r[1] for r in c.execute('PRAGMA table_info("file")')}
        if "path" not in cols:
            return counts
        counts["file_rows"] = c.execute('SELECT COUNT(*) FROM "file"').fetchone()[0]
        for old, new in maps:
            cur = c.execute(
                'UPDATE "file" SET path = ? || substr(path, ?) WHERE substr(path, 1, ?) = ?',
                (new, len(old) + 1, len(old), old),
            )
            counts["rewritten"] += cur.rowcount
        c.commit()
        ok = [r[0] for r in c.execute("PRAGMA integrity_check")]
        if ok != ["ok"]:
            raise Failed("the rewritten database failed integrity_check")
    finally:
        c.close()
    return counts


def file_path_report(db: Path, disk_root: Path | None, prefixes: dict[str, str]) -> dict[str, int]:
    """Counts of file.path rows by category, opening db read-only. `prefixes` maps a
    label to a path prefix; `disk_root` (unused rows not counted) is not needed:
    presence on disk is checked per row against the path itself."""
    out = {"rows": 0, "null": 0, "other": 0, "on_disk": 0, "missing": 0}
    for k in prefixes:
        out["under_" + k] = 0
    c = open_ro(db)
    try:
        cols = {r[1] for r in c.execute('PRAGMA table_info("file")')}
        if "path" not in cols:
            return out
        for (p,) in c.execute('SELECT path FROM "file"'):
            out["rows"] += 1
            if not p:
                out["null"] += 1
                continue
            matched = False
            for k, pre in prefixes.items():
                if p.startswith(pre):
                    out["under_" + k] += 1
                    matched = True
                    break
            if not matched:
                out["other"] += 1
            if os.path.isfile(p):
                out["on_disk"] += 1
            else:
                out["missing"] += 1
    finally:
        c.close()
    return out


# --------------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------------

def parse_supervisor_status(text: str) -> dict[str, str]:
    states: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in (RUNNING_STATES | STOPPED_STATES):
            states[parts[0]] = parts[1]
    return states


def local_services_stopped(supervisorctl: str, conf: str | None, programs=None, ports=None) -> list[str]:
    """Problems (an empty list means: all guarded programs are stopped and nothing
    listens on the guarded loopback ports). Never raises."""
    programs = GUARDED_PROGRAMS if programs is None else programs
    ports = GUARDED_PORTS if ports is None else ports
    problems: list[str] = []
    cmd = [supervisorctl] + (["-c", conf] if conf else []) + ["status", *programs]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        states = parse_supervisor_status((r.stdout or "") + "\n" + (r.stderr or ""))
    except (OSError, subprocess.SubprocessError) as e:
        return [f"cannot run supervisorctl ({type(e).__name__}); cannot confirm this pod's services are stopped"]
    for p in programs:
        st = states.get(p)
        if st is None:
            problems.append(f"cannot read the state of '{p}' from supervisorctl")
        elif st in RUNNING_STATES:
            problems.append(f"'{p}' is {st} on this pod: stop it first (supervisorctl stop {' '.join(programs)})")
        elif st not in STOPPED_STATES:
            problems.append(f"'{p}' is in state {st}, which is not a known stopped state")
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                problems.append(f"something is listening on 127.0.0.1:{port}")
        except OSError:
            pass
    return problems


def tree_signature(root: Path) -> tuple:
    """A cheap activity signature of a data tree: the DB's size/mtime, any journal
    beside it, and (count, bytes, newest mtime) of every copied subtree."""
    db = root / "webui.db"
    try:
        st = db.stat()
        dbsig = (st.st_size, st.st_mtime_ns)
    except OSError:
        dbsig = None
    journals = tuple(s for s in SIDECARS if (root / ("webui.db" + s)).exists())
    parts = []
    for t in TREES:
        n = total = newest = 0
        base = root / t
        if base.is_dir():
            stack = [base]
            while stack:
                d = stack.pop()
                try:
                    with os.scandir(d) as it:
                        for e in it:
                            try:
                                if e.is_dir(follow_symlinks=False):
                                    stack.append(Path(e.path))
                                    s = e.stat(follow_symlinks=False)
                                    newest = max(newest, s.st_mtime_ns)
                                elif e.is_file(follow_symlinks=False):
                                    s = e.stat(follow_symlinks=False)
                                    n += 1
                                    total += s.st_size
                                    newest = max(newest, s.st_mtime_ns)
                            except FileNotFoundError:
                                continue
                except FileNotFoundError:
                    continue
        parts.append((t, n, total, newest))
    return (dbsig, journals, tuple(parts))


def check_quiescent(root: Path, quiet_s: float, poll_s: float) -> list[str]:
    """Heuristic evidence that nothing is writing `root`: no journal/WAL beside
    webui.db, and an unchanged activity signature for quiet_s seconds."""
    sig0 = tree_signature(root)
    if sig0[1]:
        return ["webui.db has a -journal/-wal/-shm beside it: a writer is active, or the pod died mid-write "
                "(RUNBOOK_DB_JOURNAL.md)"]
    waited = 0.0
    while waited < quiet_s:
        step = min(poll_s, quiet_s - waited)
        time.sleep(step)
        waited += step
        if tree_signature(root) != sig0:
            return [f"the tree changed within {waited:.0f}s of watching it: its pod is still writing"]
    return []


# --------------------------------------------------------------------------------
# plan + manifest
# --------------------------------------------------------------------------------

def manifest_path(to: Path) -> Path:
    return to / MANIFEST_DIR / MANIFEST_NAME


def load_manifest(to: Path) -> dict | None:
    p = manifest_path(to)
    try:
        m = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) and m.get("format") == MANIFEST_FORMAT else None


def unusable_manifest(to: Path) -> str | None:
    """A manifest file exists but this tool cannot read it as its own format (an older
    tool's, or damaged). Returns a description, else None. Such a manifest could hide a
    completed cutover, so seed/cutover REFUSE on it whatever flags are given (--force
    included): the safe ways forward are a fresh --to directory (re-seed: cheap, and
    nothing is lost) or --reverse. There is deliberately NO override flag: an override is
    exactly what would let a tired operator overwrite her live database."""
    p = manifest_path(to)
    if not p.is_file():
        return None
    try:
        m = json.loads(p.read_text())
    except (OSError, ValueError):
        return "it is unreadable"
    if not isinstance(m, dict):
        return "it is not a manifest"
    if m.get("format") != MANIFEST_FORMAT:
        return f"it is format {m.get('format')!r}, this tool writes format {MANIFEST_FORMAT}"
    return None


def cutover_completed(m: dict | None) -> str | None:
    """The UTC time of a completed cutover recorded in the manifest, else None. After that
    the new pod may have served her: the target directory is live."""
    if not m:
        return None
    if m.get("cutover_utc"):
        return str(m["cutover_utc"])
    for r in m.get("runs") or []:
        if isinstance(r, dict) and r.get("cmd") == "cutover":
            return str(r.get("utc") or "?")
    return None


def save_manifest(to: Path, m: dict) -> None:
    d = to / MANIFEST_DIR
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (MANIFEST_NAME + ".tmp")
    m["updated_utc"] = utc_iso()
    with open(tmp, "w") as fh:
        json.dump(m, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, d / MANIFEST_NAME)
    fsync_dir(d)


BENIGN_PREFIXES = ("__selftest",)


def target_state(to: Path) -> str:
    """absent | empty | seed | pristine | dirty.

    `pristine` is a new pod's own first boot: an empty OpenWebUI schema, empty
    compactor directories and its boot self-test's probe files. Seeding over that is
    the intended first step. Anything holding chats, facts, summaries, personas,
    uploads or ledgers is `dirty`."""
    if not to.exists():
        return "absent"
    if not to.is_dir():
        return "dirty"
    if manifest_path(to).is_file():
        return "seed" if load_manifest(to) else "dirty"
    entries = [p for p in to.iterdir()]
    if not entries:
        return "empty"
    db = to / "webui.db"
    if db.is_file():
        try:
            c = open_ro(db, busy_s=5)
            try:
                counts = table_counts(c)
            finally:
                c.close()
        except sqlite3.Error:
            return "dirty"
        if any(counts.get(t, 0) > 0 for t in ("chat", "chat_message", "file", "message")):
            return "dirty"
    for t in ("uploads", "compactor", "compactor-control", "compactor-pending"):
        files, _ = walk_files(to / t)
        for rel in files:
            base = rel.rsplit("/", 1)[-1]
            top = rel.split("/", 1)[0]
            if t == "compactor" and top in ("chromadb",):
                continue  # an empty vector index is what a fresh compactor creates
            if base.startswith(BENIGN_PREFIXES):
                continue
            return "dirty"
    return "pristine"


class Plan:
    """What a run would do, as counts. The lists hold relpaths internally; only counts
    are ever printed."""

    def __init__(self) -> None:
        self.new: list[tuple[str, str]] = []          # (tree, rel)
        self.changed: list[tuple[str, str]] = []
        self.unchanged: list[tuple[str, str]] = []
        self.deleted_in_source: list[tuple[str, str]] = []
        self.target_only: list[tuple[str, str]] = []
        self.diverged: list[tuple[str, str]] = []     # target files the new pod rewrote (kept aside when replaced)
        self.skipped: dict[str, int] = {"symlink": 0, "special": 0, "temp": 0}
        self.bytes = 0

    def summary(self) -> dict[str, int]:
        return {
            "new": len(self.new), "changed": len(self.changed), "unchanged": len(self.unchanged),
            "deleted_in_source": len(self.deleted_in_source), "target_only": len(self.target_only),
            "diverged_target": len(self.diverged), "bytes_to_copy": self.bytes,
            **{"skipped_" + k: v for k, v in self.skipped.items()},
        }


def tree_pairs(src: Path, dst: Path, zl_src: Path | None, zl_dst: Path | None, with_wordban: bool):
    pairs = [(t, src / t, dst / t) for t in TREES]
    if with_wordban and zl_src is not None and zl_dst is not None:
        a, b = zl_src / "word-ban", zl_dst / "word-ban"
        try:
            same = a.exists() and b.exists() and os.path.samefile(a, b)
        except OSError:
            same = False
        if norm(str(a)) == norm(str(b)) or same:
            note("the word-ban directory is the same on both sides: not copied onto itself")
        else:
            pairs.append(("word-ban", a, b))
    return pairs


def make_plan(pairs, manifest_files: dict, strict: bool = False) -> Plan:
    """strict=True is the cutover (the source is frozen): size+mtime is NOT trusted.
    MooseFS keeps 1 s mtimes, so a same-size rewrite in the same second is invisible to
    a stat compare. Every non-SQLite file is hashed on both sides, and every SQLite
    file is re-copied through the backup API (a raw hash cannot say "same": a backup
    copy is not byte-identical to its source)."""
    plan = Plan()
    for name, s_root, d_root in pairs:
        s_files, sk = walk_files(s_root)
        d_files, _ = walk_files(d_root)
        if strict and s_files:
            note(f"hashing '{name}': {len(s_files)} file(s), {sum(x.st_size for x in s_files.values()) / 1e6:.1f} MB "
                 f"(a stat compare is not trusted at a cutover)")
        for k in plan.skipped:
            plan.skipped[k] += sk[k]
        for rel, st in sorted(s_files.items()):
            key = f"{name}/{rel}"
            m = manifest_files.get(key)
            d_st = d_files.get(rel)
            if d_st is None:
                plan.new.append((name, rel))
                plan.bytes += st.st_size
                continue
            src_same_as_seed = bool(m) and m.get("src_size") == st.st_size and m.get("src_mtime_ns") == st.st_mtime_ns
            dst_untouched = bool(m) and d_st.st_size == m.get("size") and d_st.st_mtime_ns == m.get("src_mtime_ns")
            if src_same_as_seed and dst_untouched and not strict:
                plan.unchanged.append((name, rel))
                continue
            if is_sqlite_name(rel):
                plan.changed.append((name, rel))
                if not dst_untouched:
                    plan.diverged.append((name, rel))   # the new pod rewrote it, or its origin is unknown
                plan.bytes += st.st_size
                continue
            if d_st.st_size == st.st_size and sha256_file(s_root / rel) == sha256_file(d_root / rel):
                plan.unchanged.append((name, rel))
                continue
            plan.changed.append((name, rel))
            plan.bytes += st.st_size
            if not dst_untouched:
                plan.diverged.append((name, rel))   # the new pod rewrote it, or its origin is unknown: keep the old one
        for rel in sorted(set(d_files) - set(s_files)):
            key = f"{name}/{rel}"
            m = manifest_files.get(key)
            d_st = d_files[rel]
            if m:
                plan.deleted_in_source.append((name, rel))
                if not (d_st.st_size == m.get("size") and d_st.st_mtime_ns == m.get("src_mtime_ns")):
                    plan.diverged.append((name, rel))
            else:
                plan.target_only.append((name, rel))
    return plan


def remove_empty_dirs(root: Path) -> None:
    """rmdir every empty directory under root (never root itself, never a non-empty one)."""
    for dp, dns, fns in os.walk(root, topdown=False):
        if Path(dp) == root:
            continue
        try:
            os.rmdir(dp)
        except OSError:
            pass


def apply_tree_plan(plan: Plan, pairs, manifest_files: dict, removed_dir: Path, sqlite_kw: dict,
                    tmp_dir: Path) -> dict[str, int]:
    by_name = {n: (s, d) for n, s, d in pairs}
    done = {"copied": 0, "sqlite_copied": 0, "moved_aside": 0, "errors": 0}

    def aside(name: str, rel: str) -> None:
        d_root = by_name[name][1]
        target = removed_dir / name / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(d_root / rel, target)
        done["moved_aside"] += 1

    changed_set = set(plan.changed)
    for name, rel in plan.diverged:
        if (name, rel) in changed_set and (by_name[name][1] / rel).exists():
            aside(name, rel)
    for name, rel in plan.new + plan.changed:
        s_root, d_root = by_name[name]
        key = f"{name}/{rel}"
        try:
            if is_sqlite_name(rel):
                tmp = tmp_dir / f"tree-{hashlib.sha256(key.encode()).hexdigest()[:16]}.sqlite3"
                st0 = os.stat(s_root / rel)
                sqlite_backup_copy(s_root / rel, tmp, **sqlite_kw)
                dst = d_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                part = dst.with_name(dst.name + ".migrate-tmp")
                shutil.copyfile(tmp, part)
                with open(part, "rb") as fh:
                    os.fsync(fh.fileno())
                os.utime(part, ns=(st0.st_atime_ns, st0.st_mtime_ns))
                os.replace(part, dst)
                fsync_dir(dst.parent)
                tmp.unlink(missing_ok=True)
                manifest_files[key] = {"size": dst.stat().st_size, "sha256": sha256_file(dst),
                                       "src_size": st0.st_size, "src_mtime_ns": st0.st_mtime_ns, "sqlite": True}
                done["sqlite_copied"] += 1
            else:
                sha, st0 = copy_file(s_root / rel, d_root / rel)
                manifest_files[key] = {"size": st0.st_size, "sha256": sha,
                                       "src_size": st0.st_size, "src_mtime_ns": st0.st_mtime_ns}
                done["copied"] += 1
        except (OSError, sqlite3.Error, Failed) as e:
            done["errors"] += 1
            note(f"copy failed in '{name}' ({err_tag(e)})")
    for name, rel in plan.unchanged:
        # refresh the recorded source stat so the next run can trust it
        s_root, d_root = by_name[name]
        key = f"{name}/{rel}"
        try:
            st = os.stat(s_root / rel)
            d_st = os.stat(d_root / rel)
            old = manifest_files.get(key, {})
            manifest_files[key] = {"size": d_st.st_size, "sha256": old.get("sha256") or sha256_file(d_root / rel),
                                   "src_size": st.st_size, "src_mtime_ns": st.st_mtime_ns,
                                   **({"sqlite": True} if is_sqlite_name(rel) else {})}
            if d_st.st_mtime_ns != st.st_mtime_ns:
                os.utime(d_root / rel, ns=(st.st_atime_ns, st.st_mtime_ns))
        except OSError:
            pass
    for name, rel in plan.deleted_in_source + plan.target_only:
        try:
            aside(name, rel)
            manifest_files.pop(f"{name}/{rel}", None)
        except OSError as e:
            done["errors"] += 1
            note(f"could not move a target-only file aside in '{name}' ({err_tag(e)})")
    for name, _s, d in pairs:
        if d.is_dir():
            remove_empty_dirs(d)
    return done


# --------------------------------------------------------------------------------
# installing the database on this pod (the dangerous step)
# --------------------------------------------------------------------------------

def move_aside_db_files(db: Path, dest_dir: Path) -> int:
    """Move a database file and everything SQLite or webuidb.py keeps beside it into
    dest_dir. rename, never delete. Returns how many entries were moved."""
    moved = 0
    names = [db.name] + [db.name + s for s in SIDECARS] + [db.name + ".sync-tmp", db.name + ".sync.lock"]
    for n in names:
        p = db.with_name(n)
        if p.exists() or p.is_symlink():
            dest_dir.mkdir(parents=True, exist_ok=True)
            os.replace(p, dest_dir / n)
            moved += 1
    return moved


def install_db(tmp_db: Path, snapshot: Path, local: Path | None, aside_local: Path, aside_snapshot: Path) -> dict:
    """Put a verified database copy where the NEW pod will look for it.

    With WEBUI_DB_LOCAL=true the pod's boot runs `webuidb.py --restore`, whose FIRST
    rule is "a healthy local database wins outright". A stale local file (the empty
    schema a first boot built, or the copy from seed time) would therefore beat the
    fresh snapshot, and webuidb-sync would then publish it OVER that snapshot. So:

      1. the existing LOCAL database and everything beside it (-journal/-wal/-shm,
         .sync-tmp, .sync.lock, .empty-start) is MOVED ASIDE (same-filesystem rename);
      2. the new database is copied to the local path (LOCAL FIRST: if the run dies
         between the two placements, the boot keeps the new local file and the
         daemon publishes NEW content over an OLDER snapshot, the safe direction);
      3. the SNAPSHOT is replaced (temp + fsync + sha check + rename), the old one
         moved aside;
      4. both files get the same mtime, so the daemon's "snapshot is at least as new
         as local" skip holds until she writes something.
    """
    with critical():
        return _install_db(tmp_db, snapshot, local, aside_local, aside_snapshot)


def _install_db(tmp_db: Path, snapshot: Path, local: Path | None, aside_local: Path, aside_snapshot: Path) -> dict:
    src_sha = sha256_file(tmp_db)
    out = {"local_aside": 0, "snapshot_aside": 0, "local": bool(local)}
    if local is not None:
        if local.resolve() == snapshot.resolve():
            raise Refused("the local DB path and the snapshot path are the same file")
        out["local_aside"] = move_aside_db_files(local, aside_local)
        marker = local.with_name(".empty-start")
        if marker.exists():
            aside_local.mkdir(parents=True, exist_ok=True)
            os.replace(marker, aside_local / ".empty-start")
            out["local_aside"] += 1
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + ".migrate-tmp")
        shutil.copyfile(tmp_db, part)
        with open(part, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(part, local)
        fsync_dir(local.parent)
        if sha256_file(local) != src_sha:
            raise Failed("the local database does not match the copy that was placed")
    out["snapshot_aside"] = move_aside_db_files(snapshot, aside_snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    part = snapshot.with_name(snapshot.name + ".migrate-tmp")
    shutil.copyfile(tmp_db, part)
    with open(part, "rb") as fh:
        os.fsync(fh.fileno())
    if sha256_file(part) != src_sha:
        part.unlink(missing_ok=True)
        raise Failed("the snapshot copy does not match the database that was copied")
    os.replace(part, snapshot)
    fsync_dir(snapshot.parent)
    now = time.time_ns()
    for p in ([local] if local is not None else []) + [snapshot]:
        os.utime(p, ns=(now, now))
    out["sha256"] = src_sha
    return out


# --------------------------------------------------------------------------------
# argument handling shared by seed / cutover / verify
# --------------------------------------------------------------------------------

def fold_bool(raw: str | None, default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    v = raw.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise Refused(f"WEBUI_DB_LOCAL={raw!r} is not true/false")


def resolve_local_db(args) -> Path | None:
    """The new pod's live DB path, or None when this pod keeps the DB on /data
    (WEBUI_DB_LOCAL=false)."""
    mode = args.db_mode
    if mode == "auto":
        mode = "local" if fold_bool(_podpaths.get("WEBUI_DB_LOCAL", ""), True) else "snapshot"
    if mode == "snapshot":
        return None
    return Path(args.local_db or _podpaths.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db"))


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--from", dest="src", required=True, help="the OLD pod's DATA_DIR (e.g. /data/openwebui)")
    p.add_argument("--to", dest="dst", required=True, help="the NEW pod's DATA_DIR (e.g. /data/v10/openwebui)")
    p.add_argument("--zl-from", help="the old pod's ZL_HOME (default: <from>/../zl-ops); only word-ban/ is copied")
    p.add_argument("--zl-to", help="the new pod's ZL_HOME (default: this pod's ZL_HOME)")
    p.add_argument("--no-wordban", action="store_true", help="do not copy the word-ban directory")
    p.add_argument("--with-cache", action="store_true",
                   help="also copy <from>/cache (OpenWebUI's regenerable audio/image/embedding-model cache). "
                        "Needed only if the new pod is offline (COMPACTOR_FORCE_OFFLINE) and she uses document search")
    p.add_argument("--local-data-dir", help="THIS pod's DATA_DIR (default: env, then supervisord's, then /data/openwebui)")
    p.add_argument("--json", action="store_true", help="print the final counts as one JSON line")


def add_apply_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--apply", action="store_true", help="actually do it (default: a dry run that prints the plan)")
    p.add_argument("--force", action="store_true", help="allow --to to be an existing directory that is not a seed")
    p.add_argument("--db-mode", choices=("auto", "local", "snapshot"), default="auto",
                   help="where the new pod's live DB is: local (WEBUI_DB_LOCAL=true) or the snapshot path itself")
    p.add_argument("--local-db", help="the new pod's local live DB (default $WEBUI_LOCAL_DB or /var/lib/openwebui/webui.db)")
    p.add_argument("--tmp-dir", help="local scratch directory for database copies (default: beside the local DB)")
    p.add_argument("--supervisorctl", default="supervisorctl")
    p.add_argument("--supervisor-conf", default="/etc/supervisor/conf.d/supervisord.conf")
    p.add_argument("--no-supervisor-check", action="store_true",
                   help="do not require this pod's services to be stopped (tests; never on a pod)")
    p.add_argument("--with-secret-key", action="store_true", help="also copy <from>/.webui_secret_key if it exists")
    p.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                   help="an extra file.path prefix rewrite (repeatable)")
    p.add_argument("--busy-timeout", type=float, default=60.0)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--retry-wait", type=float, default=5.0)
    p.add_argument("--step-pages", type=int, default=None,
                   help="SQLite backup-API pages per step (default 4096)")
    p.add_argument("--step-sleep", type=float, default=None,
                   help="seconds to pause between steps (default: 0.1 for a seed, 0 for a cutover; the total pause is capped at 20 s)")
    p.add_argument("--discard-new-pod-writes", action="store_true",
                   help="allow a seed/cutover to MOVE ASIDE (never delete) a local database that no longer matches the "
                        "one this tool installed, because the new pod has written to it. Prints what is moved aside")


def norm(p: str) -> Path:
    return Path(os.path.abspath(p))


def resolve_paths(args, *, writing_to: str) -> dict:
    """Validate the two trees. writing_to is "to" (seed, forward cutover) or "from"
    (reverse cutover: the local pod's tree is the source)."""
    src, dst = norm(args.src), norm(args.dst)
    if src == dst or src in dst.parents or dst in src.parents:
        raise Refused("--from and --to must be two separate directories (neither inside the other)")
    if not src.is_dir():
        raise Refused("--from is not a directory")
    local = norm(args.local_data_dir or _podpaths.data_dir())
    if writing_to == "to":
        if dst != local:
            raise Refused(
                "--to is not THIS pod's DATA_DIR: this tool only writes the tree the pod it runs on "
                "will use. If this pod's DATA_DIR is not detected (a shell without the pod's environment), "
                "pass --local-data-dir explicitly"
            )
        if src == local:
            raise Refused("--from is this pod's own DATA_DIR: the arguments look swapped")
    else:
        if src != local:
            raise Refused("--reverse: --from must be THIS pod's DATA_DIR")
    if writing_to == "to":
        zl_src = norm(args.zl_from) if args.zl_from else src.parent / "zl-ops"     # the OLD pod's ZL_HOME
        zl_dst = norm(args.zl_to) if args.zl_to else norm(_podpaths.zl_home())     # this pod's
    else:
        # reverse: the source is THIS pod, the target is the OLD pod. (Defaulting both
        # from this pod's environment made the word-ban dir copy onto itself.)
        zl_src = norm(args.zl_from) if args.zl_from else norm(_podpaths.zl_home())
        zl_dst = norm(args.zl_to) if args.zl_to else dst.parent / "zl-ops"
    return {"src": src, "dst": dst, "local": local, "zl_src": zl_src, "zl_dst": zl_dst}


def path_maps(args, src: Path, dst: Path) -> list[tuple[str, str]]:
    maps: list[tuple[str, str]] = []
    old = str(src) + "/uploads/"
    new = str(dst) + "/uploads/"
    maps.append((old, new))
    try:
        real = str(src.resolve()) + "/uploads/"
        if real != old:
            maps.append((real, new))
    except OSError:
        pass
    for spec in args.path_map:
        if "=" not in spec:
            raise Refused("--path-map must look like OLD=NEW")
        o, n = spec.split("=", 1)
        if not o or not n:
            raise Refused("--path-map must look like OLD=NEW")
        maps.append((o, n))
    return maps


def volume_age_s(path: Path) -> float | None:
    """Seconds since `path` was last written, measured against the VOLUME's clock, not this
    host's: a file is created next to the source's directory and its mtime compared with the
    path's. (The old pod's clock or this pod's may be minutes off; a wrong `age` either hides
    that she is active or cries wolf.) Falls back to this host's clock if the volume is not writable."""
    try:
        mt = path.stat().st_mtime
    except OSError:
        return None
    probe = path.parent.parent / f".pod-migrate-clock-{os.getpid()}"
    try:
        fd = os.open(str(probe), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        try:
            return os.stat(probe).st_mtime - mt
        finally:
            probe.unlink(missing_ok=True)
    except OSError:
        return time.time() - mt


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def backup_kwargs(args, kind: str) -> dict:
    """Backup-API pacing. A seed reads a LIVE database: 4096-page steps (tens of ms each) with a
    short pause, the total pause bounded, escalating to fast and then single-copy attempts if
    the source keeps being written (see sqlite_backup_copy). A cutover reads a frozen one: fast."""
    live = kind == "seed"
    pages = args.step_pages if args.step_pages else FAST_PAGES
    sleep = args.step_sleep if args.step_sleep is not None else (0.1 if live else 0.0)
    return {"busy_s": args.busy_timeout, "retries": args.retries, "retry_wait_s": args.retry_wait,
            "step_pages": pages, "step_sleep_s": sleep}


def prepare_db_copy(args, srcdb: Path, tmp_dir: Path, maps, kind: str = "cutover") -> tuple[Path, dict]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / "webui.db.migrate-copy"
    st = sqlite_backup_copy(srcdb, tmp, **backup_kwargs(args, kind))
    note(f"database copy: {st['attempts']} attempt(s), {st['seconds']:.1f} s in total, mode {st['mode']}, "
         f"{st['restarts']} restart(s), longest source read-lock hold {st['max_step_s']:.2f} s")
    if st["max_step_s"] > 3.0:
        note("WARNING: the source's read lock was held for more than 3 s in one step; the old pod's busy_timeout is 5 s. "
             "Have her close the app on every device before a seed")
    rep = rewrite_file_paths(tmp, maps)
    return tmp, rep


COUNT_TABLES = ("user", "chat", "chat_message", "file")


def db_counts(path: Path) -> dict[str, int] | None:
    try:
        c = open_ro(path, busy_s=10)
        try:
            t = table_counts(c)
        finally:
            c.close()
    except sqlite3.Error:
        return None
    return {k: t.get(k, 0) for k in COUNT_TABLES}


def new_pod_writes_problem(manifest: dict | None, local_db: Path | None, kind: str, discard: bool) -> list[str]:
    """H3: the local live DB of the new pod no longer has the hash this tool installed:
    the new pod has WRITTEN since. Returns refusal reasons (empty when it is fine, or
    when --discard-new-pod-writes was given, in which case what will move aside is printed)."""
    if manifest is None or local_db is None or not local_db.is_file():
        return []
    rec = manifest.get("db") or {}
    want = rec.get("local_sha256")
    pend = manifest.get("pending_install") or {}
    # A run killed (SIGKILL, power) between placing the new local DB and committing the manifest
    # leaves the manifest at the PREVIOUS hash. The install is recorded as pending BEFORE it starts,
    # so either hash is this tool's own file, not a write by the new pod.
    wants = {h for h in (want, pend.get("sha256") if pend.get("local_path") in (None, str(local_db)) else None) if h}
    if not wants or (want and rec.get("local_path") not in (None, str(local_db))):
        return []
    if sha256_file(local_db) in wants:
        return []
    now, was = db_counts(local_db), rec.get("counts") or {}
    delta = ", ".join(f"{k} {was.get(k, '?')} -> {(now or {}).get(k, '?')}" for k in COUNT_TABLES)
    where = "The new pod has written to its local database since this tool installed it (a test login is enough). "
    if discard:
        note(f"--discard-new-pod-writes: {where}The local database ({local_db.stat().st_size / 1e6:.1f} MB; rows: {delta}) "
             f"will be MOVED ASIDE to a .migrate-aside-<stamp> directory beside it (never deleted) and replaced.")
        return []
    return [where + f"Rows now vs when installed: {delta}. Running {kind} would move that database aside and replace it "
            "with the old pod's. If you only tested on the new pod, add --discard-new-pod-writes. If she has been using "
            "the new pod, do NOT: publish and roll back instead (RUNBOOK_POD_MIGRATION.md section 7)"]


# --------------------------------------------------------------------------------
# seed / cutover
# --------------------------------------------------------------------------------

def run_sync(args, *, kind: str) -> int:
    reverse = bool(getattr(args, "reverse", False))
    paths = resolve_paths(args, writing_to="from" if reverse else "to")
    src, dst = paths["src"], paths["dst"]
    other = dst if reverse else src  # the tree that belongs to the OTHER pod (frozen, or the rollback target)
    srcdb = src / "webui.db"
    local_db = resolve_local_db(args)   # this pod's live DB, or None when it keeps the DB on /data
    # WHICH DATABASE IS READ. Forward: the old pod's snapshot on the volume (its live file).
    # Reverse: this pod's LIVE LOCAL database (H2). The snapshot beside it can be minutes
    # behind, so copying it silently rolls her latest messages back. Local mode only; in
    # snapshot mode (WEBUI_DB_LOCAL=false) the file on the volume IS the live one.
    read_db = local_db if (reverse and local_db is not None) else srcdb
    problems: list[str] = []

    # --- preconditions ----------------------------------------------------------
    if kind == "cutover" and not args.old_pod_stopped:
        problems.append("--old-pod-stopped is required: stop the old pod's openwebui, compactor and backup "
                        "(RUNBOOK_POD_MIGRATION.md) and confirm with this flag")
    if not read_db.is_file():
        problems.append(("this pod's local live database does not exist (" + str(read_db) + "); --reverse copies it, "
                         "not the snapshot. If this pod's database really lives on the volume use --db-mode snapshot")
                        if reverse and local_db is not None else "the source has no webui.db")
    if not args.no_supervisor_check:
        problems += local_services_stopped(args.supervisorctl, args.supervisor_conf)
    state = "n/a" if reverse else target_state(dst)
    if not reverse:
        if kind == "seed" and state == "dirty" and not args.force:
            problems.append("--to exists, is not a previous seed and holds data (chats, facts, uploads ...): "
                            "refusing to merge into it (--force overrides; nothing is deleted either way)")
        if kind == "cutover" and state != "seed" and not args.force:
            problems.append(f"--to is not a seed (state: {state}); run `seed` first (--force overrides)")
    manifest = load_manifest(dst) if dst.exists() else None
    if not reverse:
        # GO-LIVE GUARD (fix round 2). Neither --force, --discard-new-pod-writes nor --db-mode
        # snapshot may reach past this: once a cutover completed here the new pod may have served her.
        bad = unusable_manifest(dst)
        if bad:
            problems.append(f"<to>/.migrate/seed.json exists but this tool cannot use it ({bad}). It may record a completed "
                            "cutover, so no flag overrides this. Seed into a NEW --to directory instead (re-seeding is cheap "
                            "and changes nothing on the old pod), or use --reverse to roll back (RUNBOOK_POD_MIGRATION.md section 7)")
        live = cutover_completed(manifest)
        if live:
            problems.append(f"a CUTOVER ALREADY COMPLETED into this directory ({live}): the new pod may have served her since. "
                            f"{kind} here is refused, and NO flag (--force, --discard-new-pod-writes, --db-mode) overrides that. "
                            "The only ways forward are --reverse (rollback, RUNBOOK_POD_MIGRATION.md section 7) or a fresh "
                            "--to directory")
        problems += new_pod_writes_problem(manifest, local_db, kind, args.discard_new_pod_writes)
    if kind == "cutover" and not problems:
        problems += check_quiescent(other, args.quiet_seconds, args.poll_interval) if (other / "webui.db").is_file() else []
    if kind == "cutover" and read_db.is_file() and any(read_db.with_name(read_db.name + s).exists() for s in SIDECARS):
        problems.append("the source webui.db has a -journal/-wal/-shm beside it (a hot journal): "
                        "the pod that owns it did not stop cleanly; follow RUNBOOK_DB_JOURNAL.md")
    if problems:
        for p in problems:
            note(f"REFUSED: {p}")
        raise Refused(f"{len(problems)} precondition(s) failed; nothing was changed")

    # --- the plan ---------------------------------------------------------------
    old_files = (manifest or {}).get("files", {})
    pairs = tree_pairs(src, dst, paths["zl_src"], paths["zl_dst"], not args.no_wordban)
    # A reverse cutover writes the OLD pod's tree, whose DB lives at <to>/webui.db itself
    # (WEBUI_DB_LOCAL=false): there is no local file to replace.
    install_local = None if reverse else local_db
    maps = path_maps(args, src, dst)
    plan = make_plan(pairs, old_files, strict=(kind == "cutover"))
    others = sorted(p.name for p in src.iterdir() if p.name not in TREES and p.name != "webui.db")
    db_size = read_db.stat().st_size
    own_tmp = not args.tmp_dir
    tmp_dir = norm(args.tmp_dir) if args.tmp_dir else (
        install_local.parent / ".migrate-tmp" if install_local else Path(tempfile.gettempdir()) / "pod-migrate-tmp")
    need = db_size * 2 + (200 << 20)
    have = free_bytes(tmp_dir)
    note(f"{kind}{' (reverse)' if reverse else ''}: target state '{state}'; source DB {db_size / 1e6:.1f} MB "
         f"({'this pod LIVE LOCAL database' if read_db != srcdb else 'the file on the volume'}); "
         f"scratch free {have / 1e6:.0f} MB (need about {need / 1e6:.0f} MB)")
    if reverse and read_db != srcdb:
        try:
            # LOGICAL hash: the local file and the snapshot are different files with different
            # page layouts and change counters even when they hold the same rows.
            same = sqlite_logical_hash(read_db) == sqlite_logical_hash(srcdb) if srcdb.is_file() else False
        except (OSError, sqlite3.Error):
            same = False
        note("the local live DB " + ("equals" if same else "DIFFERS from") + " the snapshot on the volume; "
             "the LOCAL one is what is copied back" + ("" if same else " (the snapshot is behind: expected if the last "
                                                       "sync did not run)"))
    if have < need:
        raise Refused("not enough free space for a scratch copy of the database")
    note("plan: " + ", ".join(f"{k}={v}" for k, v in plan.summary().items()))
    note(f"top-level entries of the source that are NOT copied: {len(others)} (cache, audit logs, old backups ... "
         f"regenerable or leftovers)")
    note(f"database placement: {'local ' + str(install_local) + ' + snapshot' if install_local else 'snapshot only'}; "
         f"file.path rewrite: {len(maps)} prefix map(s)")
    if kind == "seed":
        age = volume_age_s(srcdb)
        note("the seed reads the OLD pod's LIVE database: run it only while she is idle"
             + (f" (its webui.db was last written {age:.0f} s ago)" if age is not None else ""))
        if age is not None and age < 300:
            note("WARNING: the source database was written in the last 5 minutes: she may be chatting. "
                 "Wait until she is idle and re-run.")
    if args.discard_new_pod_writes and (plan.diverged or plan.target_only):
        note(f"--discard-new-pod-writes: {len(plan.diverged)} target file(s) the new pod rewrote and "
             f"{len(plan.target_only)} that exist only on the new pod will be moved to .migrate/removed-<stamp>/")
    secret_src = src / ".webui_secret_key"
    if args.with_secret_key:
        note(f"secret key file: {'will copy' if secret_src.is_file() else 'none in the source'}")
    summary = {"kind": kind, "apply": bool(args.apply), "plan": plan.summary(), "db_bytes": db_size}
    if not args.apply:
        note("DRY RUN: nothing was changed. Re-run with --apply.")
        if args.json:
            say(json.dumps(summary, sort_keys=True))
        return EXIT_OK

    # --- apply ------------------------------------------------------------------
    stamp = utc_stamp()
    n = 1
    while (dst / MANIFEST_DIR / f"removed-{stamp}").exists() or (
            install_local is not None and (install_local.parent / f".migrate-aside-{stamp}").exists()):
        n += 1   # two runs in the same second must never share (and overwrite in) one moved-aside directory
        stamp = f"{utc_stamp()}-{n}"
    dst.mkdir(parents=True, exist_ok=True)
    mdir = dst / MANIFEST_DIR
    mdir.mkdir(parents=True, exist_ok=True)
    lock = mdir / LOCK_NAME
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {utc_iso()}\n".encode())
        os.close(fd)
    except FileExistsError:
        raise Refused("another run holds <to>/.migrate/lock (remove it by hand if that run is dead)")
    restore_signals = install_signal_handlers()
    try:
        # The frozen source tree (and the database file that is read) must not move while
        # we copy FROM it.
        def signature():
            st = read_db.stat()
            return (tree_signature(src), st.st_size, st.st_mtime_ns,
                    tuple(x for x in SIDECARS if read_db.with_name(read_db.name + x).exists()))
        sig_before = signature() if kind == "cutover" else None
        if manifest is None:
            manifest = {"format": MANIFEST_FORMAT, "kind": kind, "tool": TOOL, "created_utc": utc_iso(),
                        "from": str(src), "to": str(dst), "complete": False, "files": {}, "runs": []}
        manifest["complete"] = False
        manifest["kind"] = kind
        manifest["verified"] = None
        manifest.pop("interrupted", None)
        manifest["step"] = "started"
        save_manifest(dst, manifest)

        def step(name: str) -> None:
            manifest["step"] = name
            save_manifest(dst, manifest)

        files = manifest["files"]
        try:
            step("copying the database")
            note("copying the database (read-only source, sqlite backup API) ...")
            tmp_db, rep = prepare_db_copy(args, read_db, tmp_dir, maps, kind)
            note(f"database copy verified; file.path rows rewritten: {rep['rewritten']} of {rep['file_rows']}")

            step("syncing the trees")
            removed_dir = mdir / f"removed-{stamp}"
            note("syncing the trees ...")
            done = apply_tree_plan(plan, pairs, files, removed_dir, backup_kwargs(args, kind), tmp_dir)
            note("tree sync: " + ", ".join(f"{k}={v}" for k, v in done.items()))
            if args.with_secret_key and secret_src.is_file():
                shutil.copyfile(secret_src, dst / ".webui_secret_key")
                os.chmod(dst / ".webui_secret_key", 0o600)
                note("copied .webui_secret_key")
            if done["errors"]:
                raise Failed(f"{done['errors']} file(s) could not be copied; the database was NOT installed")

            step("installing the database")
            # Recorded BEFORE the install so a SIGKILL between the local placement and the
            # manifest commit leaves a manifest that already knows the hash it is about to install.
            manifest["pending_install"] = {"sha256": sha256_file(tmp_db),
                                           "local_path": str(install_local) if install_local else None,
                                           "utc": utc_iso()}
            save_manifest(dst, manifest)
            note("installing the database ...")
            inst = install_db(tmp_db, dst / "webui.db", install_local,
                              (install_local.parent / f".migrate-aside-{stamp}") if install_local else mdir / f"aside-{stamp}",
                              mdir / f"removed-{stamp}" / "snapshot-db")
            tmp_db.unlink(missing_ok=True)
            note(f"database installed: local={'yes' if install_local else 'no'}, entries moved aside: "
                 f"local={inst['local_aside']} snapshot={inst['snapshot_aside']}")

            if sig_before is not None:
                if signature() != sig_before:
                    raise Failed("the source tree changed while the cutover ran: its pod is not frozen. "
                                 "The target now holds a partial mix; stop the old pod and re-run cutover")
            with critical():
                manifest["complete"] = True
                manifest.pop("pending_install", None)
                manifest["step"] = "installed"
                manifest["db"] = {"sha256": inst["sha256"], "src_size": db_size, "rewritten": rep["rewritten"],
                                  "local_path": str(install_local) if install_local else None,
                                  "local_sha256": inst["sha256"] if install_local else None,
                                  "counts": db_counts(dst / "webui.db"), "installed_utc": utc_iso()}
                manifest["runs"].append({"cmd": kind, "utc": utc_iso(), "plan": plan.summary()})
                if kind == "cutover":
                    manifest["cutover_utc"] = utc_iso()
                    manifest["verified"] = False
                save_manifest(dst, manifest)
        except Interrupted as e:
            manifest["complete"] = False
            manifest["interrupted"] = str(e)
            manifest["step"] = "interrupted during: " + str(manifest.get("step"))
            try:
                with critical():
                    save_manifest(dst, manifest)
            except OSError:
                pass
            raise
        finally:
            try:
                if own_tmp:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except OSError:
                pass
        summary["done"] = done
        note(f"{kind} complete. Next: `pod-migrate.py verify --from ... --to ...`")
        rc = EXIT_OK
        if kind == "cutover":
            vargs = argparse.Namespace(src=str(src), dst=str(dst), zl_from=str(paths["zl_src"]), zl_to=str(paths["zl_dst"]),
                                       no_wordban=args.no_wordban, with_cache=args.with_cache, local_data_dir=args.local_data_dir,
                                       json=False, conv_id=[HER_CONV_ID], local_db=args.local_db,
                                       db_mode="snapshot" if reverse else args.db_mode, strict_local=False,
                                       src_db=str(read_db) if read_db != srcdb else None)
            rc = run_verify(vargs)
            manifest["verified"] = rc == EXIT_OK
            manifest["step"] = "verified" if rc == EXIT_OK else "verify FAILED"
            with critical():
                save_manifest(dst, manifest)
    finally:
        restore_signals()
        try:
            lock.unlink()
        except OSError:
            pass
    if args.json:
        say(json.dumps(summary, sort_keys=True))
    return rc


# --------------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------------

def run_status(args) -> int:
    """Read-only: what does <to>/.migrate say? Exit 0 only when the last run is COMPLETE
    (a cutover additionally needs its verify to have passed), 1 when it is not (or a run
    holds the lock), 2 when there is no manifest."""
    dst = norm(args.dst)
    m = load_manifest(dst)
    bad = unusable_manifest(dst)
    if m is None and bad:
        say(f"{TOOL} status: INCOMPLETE")
        say(f"  the manifest under {dst}/{MANIFEST_DIR} cannot be used: {bad}. Do NOT start services from it; "
            "seed into a NEW --to directory (or --reverse to roll back).")
        return EXIT_FAIL
    if m is None:
        say(f"{TOOL} status: NO MANIFEST under {dst}/{MANIFEST_DIR} (nothing was seeded here). Do NOT start services.")
        return EXIT_REFUSED
    kind = m.get("kind")
    lock = dst / MANIFEST_DIR / LOCK_NAME
    running = lock.exists()
    ok = bool(m.get("complete")) and not running
    reasons = []
    if not m.get("complete"):
        reasons.append("the last run did not finish (interrupted or failed)")
    if running:
        reasons.append("a run holds the lock (still running, or it died: see the runbook)")
    if kind == "cutover" and m.get("verified") is not True:
        ok = False
        reasons.append("its verify has not passed" if m.get("verified") is False else "its verify did not run")
    if args.require and kind != args.require:
        ok = False
        reasons.append(f"the last run was '{kind}', not '{args.require}'")
    say(f"{TOOL} status: {'COMPLETE' if ok else 'INCOMPLETE'}")
    say(f"  kind={kind} complete={bool(m.get('complete'))} verified={m.get('verified')} last_step={m.get('step')!r} "
        f"updated={m.get('updated_utc')} lock={'held' if running else 'none'}")
    if m.get("interrupted"):
        say(f"  interrupted by {m['interrupted']}")
    db = m.get("db") or {}
    if db:
        say(f"  installed database: sha256={str(db.get('sha256'))[:12]}... rows={db.get('counts')} at {db.get('installed_utc')}")
    for r in reasons:
        say(f"  NOT COMPLETE: {r}")
    live = cutover_completed(m)
    if not ok and live:
        say(f"  A cutover completed here ({live}) but its verify did not pass. Do NOT re-run seed/cutover into this "
            "directory (they refuse after a completed cutover). Do NOT start services: fix the cause, then seed a NEW "
            "--to directory and cut over again, or --reverse to roll back.")
    elif not ok:
        say("  Do NOT start services. Re-run the same command; it is safe to repeat.")
    return EXIT_OK if ok else EXIT_FAIL


# --------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------

class Checks:
    def __init__(self) -> None:
        self.fail = 0
        self.rows: list[dict] = []

    def add(self, name: str, ok: bool | None, detail: str) -> None:
        tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        if ok is False:
            self.fail += 1
        self.rows.append({"check": name, "result": tag, "detail": detail})
        say(f"  {tag:4} {name}: {detail}")


def chat_stats(conn: sqlite3.Connection, conv: str) -> dict | None:
    try:
        row = conn.execute("SELECT LENGTH(chat), chat FROM chat WHERE id = ?", (conv,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    n = None
    try:
        doc = json.loads(row[1])
        n = len((doc.get("history") or {}).get("messages") or {})
    except (ValueError, TypeError, AttributeError):
        n = -1
    try:
        rows = conn.execute("SELECT COUNT(*) FROM chat_message WHERE chat_id = ?", (conv,)).fetchone()[0]
    except sqlite3.Error:
        rows = -1
    return {"json_bytes": row[0], "json_messages": n, "message_rows": rows}


def tree_digest(root: Path) -> tuple[int, int, str]:
    files, _ = walk_files(root)
    h = hashlib.sha256()
    total = 0
    for rel in sorted(files):
        if is_sqlite_name(rel):
            h.update(f"{rel}\0sqlite\0{sqlite_logical_hash(root / rel)}\0".encode())
        else:
            h.update(f"{rel}\0{sha256_file(root / rel)}\0".encode())
        total += files[rel].st_size
    return len(files), total, h.hexdigest()


def count_json_list(root: Path, sub: str) -> tuple[int, int]:
    files, _ = walk_files(root / sub)
    n = 0
    for rel in files:
        if rel.endswith(".json"):
            try:
                d = json.loads((root / sub / rel).read_text())
                if isinstance(d, dict):
                    n += len(d.get("facts") or [])
                elif isinstance(d, list):
                    n += len(d)
            except (OSError, ValueError):
                n += 0
    return len([r for r in files if r.endswith(".json")]), n


def chroma_counts(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    try:
        c = open_ro(path, busy_s=30)
        try:
            counts = table_counts(c)
        finally:
            c.close()
    except sqlite3.Error:
        return None
    return {k: v for k, v in counts.items() if k in ("embeddings", "collections", "segments", "embedding_metadata")}


def run_verify(args) -> int:
    src, dst = norm(args.src), norm(args.dst)
    if not src.is_dir() or not dst.is_dir():
        raise Refused("--from and --to must both be directories")
    ck = Checks()
    say(f"{TOOL} verify (counts only)")
    sdb, ddb = (Path(args.src_db) if getattr(args, "src_db", None) else src / "webui.db"), dst / "webui.db"
    if not sdb.is_file() or not ddb.is_file():
        ck.add("databases present", False, f"source={sdb.is_file()} target={ddb.is_file()}")
        return EXIT_FAIL
    try:
        cs, cd = open_ro(sdb), open_ro(ddb)
    except sqlite3.Error as e:
        ck.add("open databases", False, err_tag(e))
        return EXIT_FAIL
    try:
        for label, c in (("source", cs), ("target", cd)):
            try:
                res = [r[0] for r in c.execute("PRAGMA integrity_check")]
            except sqlite3.Error as e:
                res = [err_tag(e)]
            ck.add(f"integrity_check {label} snapshot", res == ["ok"], "ok" if res == ["ok"] else f"{len(res)} problem line(s)")
        ts, td = table_counts(cs), table_counts(cd)
        diffs = sorted(k for k in set(ts) | set(td) if ts.get(k) != td.get(k))
        ck.add("row counts, every table", not diffs,
               f"{len(ts)} tables compared" + ("" if not diffs else "; differ: " + ", ".join(f"{k}({ts.get(k)}->{td.get(k)})" for k in diffs)))
        for t in ("user", "chat", "chat_message", "file"):
            ck.add(f"{t} rows", ts.get(t) == td.get(t), f"source={ts.get(t)} target={td.get(t)}")
        for conv in args.conv_id:
            a, b = chat_stats(cs, conv), chat_stats(cd, conv)
            if a is None and b is None:
                ck.add("her chat", None, "not present in either database")
            else:
                ck.add("her chat (JSON history size, messages, chat_message rows)", a == b, f"source={a} target={b}")
        src_uploads = str(src) + "/uploads/"
        dst_uploads = str(dst) + "/uploads/"
        rs = file_path_report(sdb, None, {"src": src_uploads})
        rd = file_path_report(ddb, None, {"src": src_uploads, "dst": dst_uploads})
        ck.add("file rows in the source: paths on disk", None,
               f"rows={rs['rows']} under_own_uploads={rs['under_src']} on_disk={rs['on_disk']} missing={rs['missing']}")
        ck.add("file rows in the target: paths rewritten to the target", rd.get("under_src", 0) == 0,
               f"rows={rd['rows']} under_target={rd['under_dst']} still_under_old={rd['under_src']} other={rd['other']} null={rd['null']}")
        ck.add("file rows in the target: files present on disk", rd["missing"] <= rs["missing"],
               f"source on_disk={rs['on_disk']} missing={rs['missing']}; target on_disk={rd['on_disk']} missing={rd['missing']}")
        for label, c in (("source", cs), ("target", cd)):
            pass
        # any leftover mention of the old uploads prefix in chats (informational)
        for label, c, pre in (("source", cs, src_uploads), ("target", cd, src_uploads)):
            try:
                n = c.execute("SELECT COUNT(*) FROM chat WHERE instr(chat, ?) > 0", (pre,)).fetchone()[0]
            except sqlite3.Error:
                n = -1
            ck.add(f"chats mentioning the OLD uploads path ({label})", None, f"{n}")
    finally:
        cs.close()
        cd.close()

    for sub in ("facts", "summaries", "personas"):
        fs, ns = count_json_list(src / "compactor", sub)
        ft, nt = count_json_list(dst / "compactor", sub)
        ck.add(f"compactor/{sub}", (fs, ns) == (ft, nt), f"files source={fs} target={ft}; entries source={ns} target={nt}")
    for name, sp, dp in (("compactor chroma", src / "compactor/chromadb/chroma.sqlite3", dst / "compactor/chromadb/chroma.sqlite3"),
                         ("openwebui vector_db", src / "vector_db/chroma.sqlite3", dst / "vector_db/chroma.sqlite3")):
        a, b = chroma_counts(sp), chroma_counts(dp)
        ck.add(f"{name} document counts", a == b, f"source={a} target={b}")
    for t in TREES:
        a, b = tree_digest(src / t), tree_digest(dst / t)
        ck.add(f"tree '{t}' (files, bytes, sha of names+contents)", a == b,
               f"files source={a[0]} target={b[0]}; bytes source={a[1]} target={b[1]}; digest {'equal' if a[2] == b[2] else 'DIFFERENT'}")
    if not args.no_wordban:
        zl_src = norm(args.zl_from) if args.zl_from else src.parent / "zl-ops"
        zl_dst = norm(args.zl_to) if args.zl_to else norm(_podpaths.zl_home())
        a, b = tree_digest(zl_src / "word-ban"), tree_digest(zl_dst / "word-ban")
        ck.add("word-ban directory", a == b, f"files source={a[0]} target={b[0]}")
    # the local live DB on this pod, when there is one and it exists (info unless --strict-local)
    ldb = None
    try:
        ldb = resolve_local_db(args)
    except Refused:
        ldb = None
    if ldb is not None and ldb.is_file():
        try:
            cl = open_ro(ldb)
            try:
                tl = table_counts(cl)
                res = [r[0] for r in cl.execute("PRAGMA integrity_check")]
            finally:
                cl.close()
            same = tl == td
            ck.add("local live DB integrity_check", res == ["ok"], "ok" if res == ["ok"] else "problems")
            ck.add("local live DB row counts equal the snapshot", same if args.strict_local else (True if same else None),
                   "equal" if same else "differ (fine once the new pod has been writing)")
        except sqlite3.Error as e:
            ck.add("local live DB", False, err_tag(e))
    say(f"{TOOL} verify: {'FAILED' if ck.fail else 'all checks passed'} ({ck.fail} failing)")
    if args.json:
        say(json.dumps({"failing": ck.fail, "checks": ck.rows}, sort_keys=True))
    return EXIT_FAIL if ck.fail else EXIT_OK


# --------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seed", help="copy the old pod's data to the new pod (safe to re-run)", allow_abbrev=False)
    add_common(p)
    add_apply_opts(p)
    p = sub.add_parser("cutover", help="the final resync with the old pod frozen", allow_abbrev=False)
    add_common(p)
    add_apply_opts(p)
    p.add_argument("--old-pod-stopped", action="store_true",
                   help="you have stopped the other pod's openwebui, compactor and backup")
    p.add_argument("--reverse", action="store_true",
                   help="rollback: copy THIS pod's tree (--from) back into the other pod's tree (--to)")
    p.add_argument("--quiet-seconds", type=float, default=60.0,
                   help="the other pod's tree must show no change for this long (0 disables the poll)")
    p.add_argument("--poll-interval", type=float, default=10.0)
    p = sub.add_parser("status", help="COMPLETE or INCOMPLETE, from the manifest only (read-only)", allow_abbrev=False)
    p.add_argument("--to", dest="dst", required=True, help="the NEW pod's DATA_DIR (e.g. /data/v10/openwebui)")
    p.add_argument("--require", choices=("seed", "cutover"), help="also demand that the last run was this kind")
    p = sub.add_parser("verify", help="compare the two trees and databases (read-only)", allow_abbrev=False)
    add_common(p)
    p.add_argument("--conv-id", action="append", default=None, help=f"her conversation (default {HER_CONV_ID}; repeatable)")
    p.add_argument("--db-mode", choices=("auto", "local", "snapshot"), default="auto")
    p.add_argument("--local-db")
    p.add_argument("--strict-local", action="store_true", help="fail if the local live DB differs from the snapshot")
    return ap


def main(argv=None) -> int:
    global TREES
    args = build_parser().parse_args(argv)
    TREES = BASE_TREES + (("cache",) if getattr(args, "with_cache", False) else ())
    try:
        if args.cmd == "status":
            return run_status(args)
        if args.cmd == "verify":
            if not args.conv_id:
                args.conv_id = [HER_CONV_ID]
            return run_verify(args)
        return run_sync(args, kind=args.cmd)
    except Refused as e:
        note(f"REFUSED: {e}")
        return EXIT_REFUSED
    except Interrupted as e:
        note(f"INTERRUPTED by {e}: stopped cleanly. The manifest says complete=false; `pod-migrate.py status` shows it. "
             f"Do NOT start services; re-run the same command.")
        return EXIT_FAIL
    except Failed as e:
        note(f"FAILED: {e}")
        return EXIT_FAIL
    except (OSError, sqlite3.Error) as e:
        note(f"FAILED: {err_tag(e)}")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
