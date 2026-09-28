"""`zl hotfix <name> [--apply] [--yes] [--revert]` -- a pod-only, NO-IMAGE-
CHANGE patch to a file that is already baked onto the container overlay
under `/opt` (reset on every restart -- see ops/README.md), for something
urgent enough not to wait for a new image build.

Currently one entry: `backup-fast`. `compactor/backup.py`'s own
`create_backup` (v3.1.9, ~line 1277) opens its archive with
`tarfile.open(partial, "w:gz")` -- Python's default gzip compresslevel is
9. Measured on a real backup: level 9 takes ~49s, level 1 ~17s; the
archive itself grows 201->234MB. The pod reports 64-94s per backup, so
this alone is the single biggest lever available without touching the
image.

Every hotfix here follows the same shape as `zl fix`'s registry (dry run
by default; `--apply`/`--yes` to confirm; refuses cleanly instead of
guessing) but patches a FILE on the overlay rather than running an
existing repair script, so it additionally:
  - saves the original once, to `<path>.zl-orig` (never overwritten by a
    later --apply -- it is always the PRE-hotfix original), written
    atomically WITH an fsync and a round-trip check before it is trusted;
  - writes the patched file atomically (temp file + `os.replace`,
    preserving mode/owner);
  - checks the patched file actually compiles and imports the way
    supervisord's own interpreter would load it, BEFORE touching any
    running service, and restores the original on any failure (including
    a compile timeout, a missing interpreter, or anything else that check
    can raise);
  - restarts ONLY the one supervisord program the patch affects (never
    compactor/openwebui/vllm), ONLY if it was RUNNING/STARTING before,
    and never while a backup is actually in progress or another
    `zl hotfix` apply/revert is already running;
  - wraps its own stop -> start in `common.ProtectedRegion` (SIGHUP
    ignored, SIGINT/SIGTERM deferred until the start is issued and
    verified) the same way `zl fix`/`zl journal repair` do.

`--revert` undoes it (restores `<path>.zl-orig`) and restarts the same
program the same way -- but ONLY when the current on-disk file is
byte-for-byte what this hotfix itself would have produced (`orig` with
the anchor patched in); otherwise it refuses and shows a diff rather than
guessing which of two different "originals" is the right one to keep.
"""

from __future__ import annotations

import contextlib
import difflib
import fcntl
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import threading
import time

from .. import common, podenv

# --- registry ----------------------------------------------------------------

_TARGET = podenv.COMPACTOR_PKG / "backup.py"  # /opt/compactor/backup.py
_SUPERVISORD_PROGRAM = "backup"

# The EXACT anchor line in compactor/backup.py's own `create_backup`
# (v3.1.9, ~line 1277 -- `git show v3.1.9:compactor/backup.py`). Matched
# verbatim, on purpose: a fuzzy/regex match risks silently patching the
# WRONG call site (there are three other `tarfile.open(..., "r:gz")`
# reads in the same file -- never touched, they only ever READ an
# archive, an already-cheap operation compression level cannot speed up).
_ANCHOR_OLD = 'with tarfile.open(partial, "w:gz") as tar:'
_ANCHOR_NEW = 'with tarfile.open(partial, "w:gz", compresslevel=1) as tar:'

# backup.py's own naming (module-level constants `_ARCHIVE_PREFIX =
# "zions-backup-"`, `_ARCHIVE_SUFFIX = ".tar.gz"`; `create_backup` stages
# into a `tempfile.mkdtemp(prefix=f"{_ARCHIVE_PREFIX}{stamp}-", dir=str(d))`
# directory and writes `{_ARCHIVE_PREFIX}{stamp}{_ARCHIVE_SUFFIX}.partial`
# before renaming it -- both literals are duplicated here (backup.py is a
# pod-only /opt file we patch, never a package this bundle imports) so "a
# backup is in progress" can be detected without importing it.
_ARCHIVE_PREFIX = "zions-backup-"
_PARTIAL_SUFFIX = ".tar.gz.partial"

_ORIG_SUFFIX = ".zl-orig"

# Round-9 HIGH-1: bootstrap.sh's own `_HOTFIX_BOOT_TIMEOUT_S` (the whole
# `zl hotfix NAME --apply --yes` invocation, at boot) is 60s, hard-killed
# by `timeout -k 2` regardless of what is happening inside. This wait must
# leave real room, in the ordinary (non-adversarial) case, for the
# compile/import check (`_IMPORT_CHECK_TIMEOUT_S` below) and the
# stop/start round trip that follow it -- so it is deliberately shorter
# than that outer bound, not equal to it.
_WAIT_FOR_BACKUP_S = 40.0  # bound: refuse and ask the operator to re-run, never wait forever
_WAIT_PRINT_INTERVAL_S = 10.0

# Round-9 MEDIUM-2: the compile-check's own import/run subprocess timeout,
# pulled out to a module constant so a test can shrink it (a P8-style slow
# `--list` must not make the unit test suite itself slow).
_IMPORT_CHECK_TIMEOUT_S = 30.0

# Round-9 HIGH-2: exclusive lock so two `zl hotfix` invocations (apply,
# revert, or one of each) can never run concurrently -- see `_change_lock`.
# NOTE: only `zl hotfix` itself takes this lock. The brief asked for the
# same lock on `zl fix`'s and `zl journal repair`'s own apply paths "if
# that's cheap and safe; otherwise lock just hotfix and say so" -- this
# round is scoped to hotfix.py only (fix.py/journal.py are out of scope
# for this pass, per the brief's own file list), so a `zl fix --apply` or
# `zl journal repair --apply` running at the exact same moment as a
# `zl hotfix` is NOT mutually excluded by this lock. Documented here, and
# in the final report, rather than silently only half-solving it.
_CHANGE_LOCK_PATH = podenv.ZL_HOME / "locks" / "change.lock"
_CHANGE_LOCK_OVERLAY_FALLBACK = "/run/zl-hotfix-change.lock"  # never /data -- always reachable
_CHANGE_LOCK_OPEN_TIMEOUT_S = 5.0  # bound on even OPENING the lock file (ZL_HOME/data can stall)
_CHANGE_LOCK_ACQUIRE_TIMEOUT_S = 10.0  # bound on waiting for another zl hotfix to release it


def _orig_path():
    return _TARGET.with_name(_TARGET.name + _ORIG_SUFFIX)


def _backup_in_progress() -> bool:
    """A fresh staging dir (`zions-backup-<stamp>-XXXXXXXX/`) or a
    `.tar.gz.partial` file under podenv.BACKUPS -- backup.py's own
    naming for "a cycle is currently running". Anything else already
    finished (renamed to the plain `.tar.gz`) or was cleaned up.
    """
    d = podenv.BACKUPS
    if not d.is_dir():
        return False
    try:
        entries = list(d.iterdir())
    except OSError:
        return False
    for p in entries:
        if p.is_dir() and p.name.startswith(_ARCHIVE_PREFIX):
            return True
        if p.name.endswith(_PARTIAL_SUFFIX):
            return True
    return False


# --- daemon-vs-file staleness (round-9 MEDIUM-1) -----------------------------


def _supervisorctl_pid(name: str) -> int | None:
    """The real OS pid of supervisord program `name`, from `supervisorctl
    pid NAME`'s own stdout -- None if it cannot be parsed as a plain
    integer (program unknown, not running, supervisorctl unreachable,
    ...). Never guesses.
    """
    try:
        r = common.run(["supervisorctl", "pid", name], timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    text = (r.stdout or "").strip().splitlines()[0].strip() if (r.stdout or "").strip() else ""
    return int(text) if text.isdigit() else None


def _proc_start_time(pid: int) -> float | None:
    """Wall-clock start time (unix epoch seconds) of `pid`, computed from
    `/proc/<pid>/stat` field 22 (`starttime`, in clock ticks since boot)
    and `/proc/uptime` (seconds since boot) -- stdlib only, no psutil.
    Returns None if `/proc` cannot answer (process gone, container without
    `/proc`, a non-Linux dev box running the unit tests, ...) -- callers
    must treat that as "cannot tell", never as "started before" or
    "started after".
    """
    try:
        stat_text = open(f"/proc/{pid}/stat", encoding="utf-8").read()
        uptime_s = float(open("/proc/uptime", encoding="utf-8").read().split()[0])
        clk_tck = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None
    # The 2nd field (comm, in parens) can itself contain spaces/parens --
    # split from the LAST ')' so the rest are unambiguous positional fields.
    rparen = stat_text.rfind(")")
    if rparen < 0:
        return None
    fields = stat_text[rparen + 2 :].split()
    try:
        starttime_ticks = float(fields[19])  # fields[0] here is field 3 overall; field 22 is index 19
    except (IndexError, ValueError):
        return None
    if clk_tck <= 0:
        return None
    boot_time = time.time() - uptime_s
    return boot_time + starttime_ticks / clk_tck


def _daemon_started_before(mtime: float) -> bool | None:
    """True if the RUNNING `backup` program's own process start time is
    older than `mtime` (the patched file's own mtime) -- i.e. the daemon
    currently running was exec'd from the file as it was BEFORE this
    mtime, so it is still running the OLD code even though the file on
    disk has since changed. None if this cannot be determined at all
    (program not running, no pid, `/proc` unreadable) -- never guessed.
    """
    pid = _supervisorctl_pid(_SUPERVISORD_PROGRAM)
    if pid is None:
        return None
    started = _proc_start_time(pid)
    if started is None:
        return None
    return started < mtime


def backup_compression_status() -> str:
    """For `zl version`: the FILE's own state --
    "level 1 (hotfix)" / "level 1 (native)" / "default" -- plus, when the
    file is patched, whether the RUNNING daemon actually has it yet
    (round-9 MEDIUM-1): a restart this wrapper did not (yet) perform
    leaves the daemon on the old code even though the file already reads
    "level 1". Never guesses: "default" (or plain "level 1 (...)" with no
    daemon note) whenever any piece of this cannot be read/determined.

    "native" (not "hotfix") when the file already reads level-1 but no
    `.zl-orig` exists -- a future/`.8`+ image that simply ships the fix
    baked in, never patched by this wrapper at all.
    """
    try:
        text = _TARGET.read_text(encoding="utf-8")
        mtime = _TARGET.stat().st_mtime
    except OSError:
        return "default (could not read backup.py)"
    if _ANCHOR_NEW not in text:
        return "default"
    kind = "hotfix" if _orig_path().is_file() else "native"
    base = f"level 1 ({kind})"
    stale = _daemon_started_before(mtime)
    if stale is None:
        return base
    if stale:
        return f"{base} / daemon started before patch -> restart pending"
    return f"{base} / daemon has it"


def _read_target() -> str | None:
    try:
        return _TARGET.read_text(encoding="utf-8")
    except OSError as e:
        print(f"zl hotfix backup-fast: could not read {_TARGET}: {e}", file=sys.stderr)
        return None


def _atomic_write(path, text: str, *, like, fsync: bool = False) -> None:
    """Write `text` to `path` atomically (temp file in the SAME directory
    + `os.replace`, so a crash mid-write never leaves a half-written
    file), preserving `like`'s mode/owner (the original file's own
    `os.stat`, taken BEFORE this call). `fsync=True` (round-9 LOW: used
    for saving `.zl-orig`) additionally fsyncs the file's own data before
    the rename and the directory entry after it, so a crash right after
    this returns cannot lose the write to a page-cache-only state.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.chmod(tmp, like.st_mode)
        try:
            os.chown(tmp, like.st_uid, like.st_gid)
        except OSError:
            pass  # not running as root, or no such user/group any more -- best effort only
        os.replace(tmp, str(path))
        if fsync:
            try:
                dfd = os.open(str(path.parent), os.O_RDONLY)
            except OSError:
                dfd = None
            if dfd is not None:
                try:
                    os.fsync(dfd)
                except OSError:
                    pass
                finally:
                    os.close(dfd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_orig_verified(orig_path, orig_text: str, *, like, expected_patched: str) -> bool:
    """Round-9 LOW: `.zl-orig` is written atomically WITH an fsync (of
    both the file and its directory entry), then immediately re-read and
    ROUND-TRIP checked: `orig.replace(_ANCHOR_OLD, _ANCHOR_NEW, 1)` must
    reproduce, byte for byte, the exact patched content this call is
    about to write to `_TARGET` (`expected_patched`). A write/filesystem
    that silently produced something else -- or that lied about having
    completed at all -- must never be trusted as "the original is safely
    saved"; a bad `.zl-orig` is only discovered (if at all) much later, at
    `--revert` time, when it is too late to do anything but guess. Returns
    False (never raises) on any failure; the caller must refuse to
    proceed rather than apply on top of an unverified save.
    """
    try:
        _atomic_write(orig_path, orig_text, like=like, fsync=True)
        reread = orig_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return reread == orig_text and reread.replace(_ANCHOR_OLD, _ANCHOR_NEW, 1) == expected_patched


def _compiles_and_imports(path) -> tuple[bool, str]:
    """Compile-check with py_compile, then an import check with the SAME
    interpreter/cwd supervisord's own `[program:backup]` uses
    (`/opt/compactor-venv/bin/python`, `directory=/opt/compactor`) --
    `--list` is backup.py's own READ-ONLY subcommand (never `--daemon`,
    which loops forever and would never return here).

    Round-9 MEDIUM-2: every failure path here returns `(False, detail)`
    rather than letting an exception escape -- `py_compile.compile` can
    raise more than `PyCompileError`/`OSError` (round-9 review: a genuine
    "any other exception" catch-all), and the import/run subprocess can
    raise `subprocess.TimeoutExpired` (a hung `--list`, e.g. a P8-style
    slow read) or `FileNotFoundError` (the venv interpreter itself
    missing) -- neither was caught before. `_do_apply`'s own caller relies
    on this NEVER raising: every failure here must restore the original,
    not crash with the patched file already in place.

    `cfile=` (round-9 LOW) points `py_compile` at a throwaway temp file
    instead of letting it default to `/opt/compactor/__pycache__` --
    never write into the pod's own overlay package directory for a check
    this wrapper runs on every apply.
    """
    cfile_fd, cfile_path = tempfile.mkstemp(suffix=".pyc")
    os.close(cfile_fd)
    try:
        try:
            py_compile.compile(str(path), cfile=cfile_path, doraise=True)
        except py_compile.PyCompileError as e:
            return False, f"py_compile failed: {e}"
        except Exception as e:  # noqa: BLE001 -- deliberate catch-all, see docstring
            return False, f"py_compile failed unexpectedly: {e}"
    finally:
        try:
            os.unlink(cfile_path)
        except OSError:
            pass

    try:
        r = common.run(
            [str(podenv.COMPACTOR_PY), str(path), "--list", "--json"],
            timeout=_IMPORT_CHECK_TIMEOUT_S, cwd=str(podenv.COMPACTOR_PKG), env=common.merged_environ(),
        )
    except subprocess.TimeoutExpired as e:
        return False, f"import/run check ({path} --list --json) timed out after {_IMPORT_CHECK_TIMEOUT_S:.0f}s: {e}"
    except FileNotFoundError as e:
        return False, f"import/run check could not run {podenv.COMPACTOR_PY}: {e}"
    except Exception as e:  # noqa: BLE001 -- deliberate catch-all, see docstring
        return False, f"import/run check failed unexpectedly: {e}"
    if r.returncode != 0:
        return False, f"import/run check ({path} --list --json) exited {r.returncode}: {(r.stderr or r.stdout)[:400]}"
    return True, "ok"


# --- exclusive change lock (round-9 HIGH-2) ----------------------------------


class ChangeLockRefused(Exception):
    """Raised when the exclusive change lock cannot be acquired -- either
    another `zl hotfix` apply/revert is already running, or the lock file
    itself could not be reached at all.
    """


def _open_lock_file_bounded(path, timeout: float):
    """Opens (creating if needed) the lock file at `path`, bounded by
    `timeout` -- reuses `bundle.py`'s own daemon-thread-with-join idiom
    for bounding a filesystem call that might stall (MooseFS on
    `ZL_HOME`). Returns `(fd, error)`; exactly one is None.
    """
    result: dict[str, object] = {}

    def _worker() -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            result["fd"] = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            result["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, f"opening {path} timed out after {timeout:.0f}s (possible /data stall)"
    if "error" in result:
        return None, str(result["error"])
    return result.get("fd"), None


@contextlib.contextmanager
def _change_lock(*, what: str):
    """Exclusive `flock` so two `zl hotfix` invocations (apply, revert, or
    one of each) can never run at once -- see the module-level note by
    `_CHANGE_LOCK_PATH` on scope (this round: hotfix.py only).

    Opening the lock file itself is bounded (`ZL_HOME` is a MooseFS mount
    that can stall); on that timing out, falls back to a lock file on the
    container overlay (`/run`, never `/data`) instead -- still exclusive
    against every OTHER invocation that also falls back there, just not
    against one that reached the real `ZL_HOME` lock a moment earlier.
    """
    fd, err = _open_lock_file_bounded(_CHANGE_LOCK_PATH, _CHANGE_LOCK_OPEN_TIMEOUT_S)
    used_path = _CHANGE_LOCK_PATH
    if fd is None:
        from pathlib import Path as _Path

        used_path = _Path(_CHANGE_LOCK_OVERLAY_FALLBACK)
        fd, err2 = _open_lock_file_bounded(used_path, _CHANGE_LOCK_OPEN_TIMEOUT_S)
        if fd is None:
            raise ChangeLockRefused(
                f"could not open a lock file for {what} (ZL_HOME: {err}; overlay fallback: {err2})"
            )
    acquired = False
    try:
        deadline = time.monotonic() + _CHANGE_LOCK_ACQUIRE_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
        if not acquired:
            raise ChangeLockRefused(
                f"another `zl hotfix` apply/revert is already running (locked: {used_path}) -- "
                f"refusing to run {what} concurrently"
            )
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass


# --- saved --with-hotfix flag removal on revert (round-9 LOW) ---------------

_BOOTSTRAP_FLAGS_FILE = podenv.ZL_HOME / "bootstrap-flags"
_SAVED_FLAG_LINE = "--with-hotfix=backup-fast"
_FLAG_FILE_WRITE_TIMEOUT_S = 5.0  # bounded -- ZL_HOME can stall; best-effort, never blocks --revert


def _remove_saved_with_hotfix_flag() -> None:
    """`--revert` also removes bootstrap.sh's own saved `--with-hotfix
    backup-fast` flag (`$ZL_HOME/bootstrap-flags`, written by an
    interactive `bootstrap.sh --with-hotfix backup-fast` run) -- otherwise
    the NEXT restart/`bootstrap.sh` re-run would silently re-apply the
    hotfix the operator just reverted. Best-effort and bounded (the read
    +rewrite happens in a daemon thread, joined with a timeout, same
    idiom as `_open_lock_file_bounded`) -- a stalled/missing `ZL_HOME`
    must never turn a successful `--revert` into a failure; it only
    prints a warning and leaves the saved flag as-is.
    """
    result: dict[str, object] = {}

    def _worker() -> None:
        try:
            if not _BOOTSTRAP_FLAGS_FILE.is_file():
                return
            lines = _BOOTSTRAP_FLAGS_FILE.read_text(encoding="utf-8").splitlines()
            kept = [ln for ln in lines if ln.strip() != _SAVED_FLAG_LINE]
            if kept == lines:
                return  # nothing to remove
            new_text = "".join(ln + "\n" for ln in kept)
            like = _BOOTSTRAP_FLAGS_FILE.stat()
            _atomic_write(_BOOTSTRAP_FLAGS_FILE, new_text, like=like)
            result["removed"] = True
        except OSError as e:
            result["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(_FLAG_FILE_WRITE_TIMEOUT_S)
    if t.is_alive():
        common.safe_print(
            f"zl hotfix backup-fast: warning: removing the saved --with-hotfix flag from "
            f"{_BOOTSTRAP_FLAGS_FILE} timed out (possible /data stall) -- it may still be saved",
            file=sys.stderr,
        )
    elif result.get("error"):
        common.safe_print(
            f"zl hotfix backup-fast: warning: could not remove the saved --with-hotfix flag from "
            f"{_BOOTSTRAP_FLAGS_FILE}: {result['error']}",
            file=sys.stderr,
        )
    elif result.get("removed"):
        common.safe_print(
            f"zl hotfix backup-fast: removed the saved --with-hotfix=backup-fast flag from "
            f"{_BOOTSTRAP_FLAGS_FILE} -- a future restart/bootstrap re-run will not re-apply it",
        )


# --- restart (round-9 HIGH-1 + HIGH-2) ---------------------------------------


def _wait_for_backup_not_in_progress(region: "common.ProtectedRegion") -> bool:
    """Bounded wait for `_backup_in_progress()` to clear, printing a
    "do not disconnect" progress line every `_WAIT_PRINT_INTERVAL_S`
    (round-9 HIGH-1). Returns True once clear; False if the bound
    (`_WAIT_FOR_BACKUP_S`) elapsed OR a signal was deferred while waiting
    -- checked here so a Ctrl-C/hangup stops the wait promptly instead of
    running out an already-pointless timer; the CALLER still re-checks
    `region.deferred_signal` immediately after this returns, to decide how
    to report it (this function only stops waiting, it does not act on
    the signal itself).
    """
    if not _backup_in_progress():
        return True
    deadline = time.monotonic() + _WAIT_FOR_BACKUP_S
    next_print = time.monotonic()
    while _backup_in_progress():
        if region.deferred_signal is not None:
            return False
        now = time.monotonic()
        if now >= deadline:
            return False
        if now >= next_print:
            remaining = max(0, int(deadline - now))
            print(
                f"zl hotfix backup-fast: a backup is in progress under {podenv.BACKUPS} -- "
                f"waiting (up to {remaining}s left) -- do not disconnect"
            )
            next_print = now + _WAIT_PRINT_INTERVAL_S
        time.sleep(0.5)
    return True


def _restart_if_needed(region: "common.ProtectedRegion", *, apply_note: str) -> int:
    """Round-9 HIGH-1 + HIGH-2: the whole stop -> start window (plus the
    wait for any in-progress backup cycle before it) runs inside `region`
    -- a `common.ProtectedRegion` the CALLER opened (`run()`, spanning the
    entire apply/revert, not just this restart step -- see its own
    docstring on why this must start BEFORE any of the file-write/compile-
    check work that precedes a restart, not just around the stop/start
    themselves) -- the same way `zl fix`/`zl journal repair` do: SIGHUP
    ignored, SIGINT/SIGTERM deferred from before the wait/stop all the way
    through a VERIFIED start, then reflected as `exit 128+signal` (with an
    audit line -- `cli.main`'s own `except SystemExit` handler writes it,
    unchanged) rather than silently dropped or acted on mid-stop.

    Round-9 HIGH-2: restarts `backup` ONLY if it was RUNNING/STARTING
    before (reusing `common.stop_for_apply`'s own `was_running` pattern --
    it returns `[]`, without even calling `supervisorctl stop`, when the
    program was not active to begin with). If it was not running, the
    patch (already written to disk by the caller) simply takes effect the
    next time the program starts on its own.
    """
    region.set_message(
        "zl hotfix backup-fast: signal received while waiting for a backup in progress to finish -- "
        "nothing has been stopped yet; this will be acted on once the wait ends"
    )
    proceed = _wait_for_backup_not_in_progress(region)
    region.clear_message()
    if not proceed and region.deferred_signal is None:
        print(
            f"zl hotfix backup-fast: a backup is currently in progress under {podenv.BACKUPS} -- "
            f"refusing to restart `{_SUPERVISORD_PROGRAM}` while it runs; re-run in a moment",
            file=sys.stderr,
        )
        region.finish(1, restored=None)
        return 1
    if region.deferred_signal is not None:
        common.safe_print(
            "zl hotfix backup-fast: signal received during the in-progress wait -- nothing was "
            "stopped; aborting without restarting",
            file=sys.stderr,
        )
        region.finish(1, restored=None)
        return 1

    try:
        was_running = common.stop_for_apply([_SUPERVISORD_PROGRAM])
    except common.StopVerificationFailed as e:
        code, restored = common.refuse_after_stop_failure(e)
        region.finish(code, restored=restored)
        return code

    if region.deferred_signal is not None:
        sig = region.deferred_signal
        came_back = common.restart_after_apply(was_running)
        restored = common.report_restart_failures(came_back)
        common.safe_print(
            "zl hotfix backup-fast: cancelled before the restart; "
            + ("services restored" if restored else "services NOT fully restored -- see the WARNING above"),
            file=sys.stderr,
        )
        # This branch already reported and acted on the deferred signal
        # itself (computing 128+sig directly, the same convention
        # `ProtectedRegion.finish()` uses) -- clear it so `run()`'s own
        # generic `region.finish()` call, further up the stack, treats
        # this region as already settled instead of firing a second,
        # misleadingly-worded ("nothing was stopped") message on top of
        # this one.
        region.deferred_signal = None
        return 128 + sig if restored else 1

    if not was_running:
        common.safe_print(
            f"zl hotfix backup-fast: {apply_note}; `{_SUPERVISORD_PROGRAM}` is not running -- "
            "the patch takes effect when it next starts"
        )
        region.finish(0, applied=True, label="the hotfix")
        return 0

    region.set_message(
        "zl hotfix backup-fast: do not disconnect -- `backup` is being restarted to pick up the patch"
    )
    came_back = common.restart_after_apply(was_running)
    region.clear_message()
    restored = common.report_restart_failures(came_back)
    code = 0 if restored else 1
    if restored:
        common.safe_print(f"zl hotfix backup-fast: {apply_note}; `{_SUPERVISORD_PROGRAM}` restarted")
    region.finish(code, restored=restored, applied=True, label="the hotfix")

    if restored:
        _print_boot_time_backup_note()
    return code


def _print_boot_time_backup_note() -> None:
    """backup.py's own `run_daemon` skips its boot-time cycle only when
    the newest archive is under half the configured interval old -- a
    restart we just triggered is subject to that exact same check, so
    tell the operator what to expect rather than leaving them guessing.
    """
    try:
        interval_hours = float(common.proc1_environ().get("COMPACTOR_BACKUP_INTERVAL_HOURS", "24") or "24")
    except (ValueError, TypeError):
        interval_hours = 24.0
    newest = None
    if podenv.BACKUPS.is_dir():
        try:
            archives = sorted(
                (p for p in podenv.BACKUPS.iterdir() if p.name.startswith(_ARCHIVE_PREFIX) and p.is_file()),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
        except OSError:
            archives = []
        if archives:
            newest = time.time() - archives[0].stat().st_mtime
    if newest is None or newest > (interval_hours * 3600.0 / 2.0):
        common.safe_print(
            "zl hotfix backup-fast: the newest archive is more than half the backup interval old (or there is "
            "none) -- the daemon will likely take an immediate boot-time backup now that it restarted"
        )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_hotfix_audit_line(action: str, *, before_text: str, after_text: str) -> None:
    """Round-9 LOW: on top of `cli.main`'s own generic audit line
    (argv/exit_code/duration, unchanged), append ONE extra record to the
    SAME `podenv.AUDIT_LOG` with the before/after sha256 of `_TARGET` --
    the generic schema has no room for a per-command detail like this, and
    common.py's `write_audit` is shared plumbing this round does not
    touch; this is hotfix's own supplementary line, same file, same
    best-effort/never-raise contract (a MooseFS stall on the audit log
    must never abort the patch/revert it is logging).
    """
    record = {
        "time": common.utcnow_iso(),
        "hotfix": "backup-fast",
        "action": action,
        "target": str(_TARGET),
        "sha256_before": _sha256_text(before_text),
        "sha256_after": _sha256_text(after_text),
    }
    try:
        podenv.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with podenv.AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        common.safe_print(f"zl hotfix backup-fast: warning: could not write the hotfix audit line ({e}); "
                           "continuing", file=sys.stderr)


def _unified_diff(a_text: str, b_text: str, *, a_name: str, b_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            a_text.splitlines(keepends=True), b_text.splitlines(keepends=True),
            fromfile=a_name, tofile=b_name,
        )
    )


def _do_apply(text: str, *, revert: bool, region: "common.ProtectedRegion") -> int:
    """`region` (round-9: opened by `run()`, wrapping this ENTIRE call --
    not just the eventual restart) means SIGHUP is already being ignored,
    and SIGINT/SIGTERM already being deferred, before the very first byte
    of `.zl-orig`/the patched file is written -- not only once
    `_restart_if_needed` gets around to opening its own region. An earlier
    round had the region start there instead, which left a real (if
    narrow) window between process start and that point -- covering the
    patch write and the compile/import check -- where a signal fell back
    to Python's DEFAULT disposition (SIGHUP terminates the process
    outright); a real-image probe caught it happening.
    """
    orig = _orig_path()

    if revert:
        # Round-9 MEDIUM-3: --revert only when the CURRENT on-disk file is
        # byte-for-byte what re-applying the patch to `orig` would
        # produce -- never blindly overwrite with `orig` regardless of
        # what is actually there right now.
        orig_text = orig.read_text(encoding="utf-8")
        expected_patched = orig_text.replace(_ANCHOR_OLD, _ANCHOR_NEW, 1)
        if text == orig_text:
            print(f"zl hotfix backup-fast: --revert: {_TARGET} already matches the pre-hotfix original "
                  f"({orig}) -- nothing to do")
            return 0
        if text != expected_patched:
            diff = _unified_diff(
                expected_patched, text,
                a_name=f"expected ({orig} with the patch re-applied)", b_name=str(_TARGET),
            )
            print(
                f"zl hotfix backup-fast: --revert: {_TARGET} does not match what this hotfix would have "
                f"produced from {orig} -- refusing to guess which is right; diff:\n{diff}",
                file=sys.stderr,
            )
            return 1

        like = _TARGET.stat()
        _atomic_write(_TARGET, orig_text, like=like)
        ok, detail = _compiles_and_imports(_TARGET)
        if not ok:
            print(f"zl hotfix backup-fast: --revert wrote {orig}'s content back but it fails its own "
                  f"compile/import check ({detail}) -- {_TARGET} is left as reverted anyway (it is, after all, "
                  f"the file that was running before the hotfix); investigate by hand", file=sys.stderr)
            return 1
        _write_hotfix_audit_line("revert", before_text=text, after_text=orig_text)
        code = _restart_if_needed(region, apply_note=f"reverted to the pre-hotfix original ({orig})")
        if code == 0:
            _remove_saved_with_hotfix_flag()
        return code

    if not orig.is_file():
        patched_would_be = text.replace(_ANCHOR_OLD, _ANCHOR_NEW, 1)
        like_before = _TARGET.stat()
        if not _save_orig_verified(orig, text, like=like_before, expected_patched=patched_would_be):
            print(f"zl hotfix backup-fast: could not verify a saved original at {orig} round-trips back to the "
                  f"intended patched content -- refusing to apply, nothing changed", file=sys.stderr)
            return 1

    if _ANCHOR_OLD not in text:
        print(f"zl hotfix backup-fast: anchor line not found in {_TARGET} -- refusing, nothing changed",
              file=sys.stderr)
        return 1
    patched = text.replace(_ANCHOR_OLD, _ANCHOR_NEW, 1)
    like = _TARGET.stat()
    _atomic_write(_TARGET, patched, like=like)

    ok, detail = _compiles_and_imports(_TARGET)
    if not ok:
        # Restore straight from the in-memory original text (not from
        # `orig` on disk -- identical content, but this way a --revert
        # later still works even if this restore step itself is somehow
        # interrupted before it finishes writing `orig`... it already has,
        # above, so this is belt-and-braces, not load-bearing).
        _atomic_write(_TARGET, text, like=like)
        print(f"zl hotfix backup-fast: the patched file fails its own compile/import check ({detail}) -- "
              f"restored the original, nothing was applied", file=sys.stderr)
        return 1

    _write_hotfix_audit_line("apply", before_text=text, after_text=patched)

    # The "never restart while a backup is in progress" rule (checked
    # inside `_restart_if_needed`) gates the RESTART only, never the file
    # edit above -- editing the file on disk never touches an
    # already-running process's own already-loaded copy of the module (a
    # backup mid-flight keeps running unaffected either way); only THIS
    # step -- reloading the new code by restarting the program -- is
    # unsafe to do mid-cycle, so only it waits/refuses.
    return _restart_if_needed(region, apply_note="patched compresslevel=1 into backup.py's create_backup")


def run(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: zl hotfix backup-fast [--apply [--yes]] [--revert [--apply [--yes]]]", file=sys.stderr)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    name, rest = argv[0], argv[1:]
    if name != "backup-fast":
        print(f"zl hotfix: unknown hotfix {name!r} -- only 'backup-fast' exists", file=sys.stderr)
        return 2

    apply = "--apply" in rest
    yes = "--yes" in rest
    revert = "--revert" in rest
    unknown = [a for a in rest if a not in ("--apply", "--yes", "--revert")]
    if unknown:
        print(f"zl hotfix backup-fast: unknown argument(s): {' '.join(unknown)}", file=sys.stderr)
        return 2

    text = _read_target()
    if text is None:
        return 1

    if not revert:
        if _ANCHOR_NEW in text:
            # Round-9 MEDIUM-1: the FILE already reads level-1, but the
            # RUNNING daemon may still be on the old code (it was never
            # restarted since -- e.g. the file was patched by hand, or a
            # previous `zl hotfix` apply's restart step failed/was
            # skipped). Restart it, under the exact same rules as a fresh
            # apply (2: only if it was running; 1: protected region).
            mtime = _TARGET.stat().st_mtime
            stale = _daemon_started_before(mtime)
            if stale:
                print(f"zl hotfix backup-fast: {_TARGET} already has compresslevel=1, but the running "
                      f"`{_SUPERVISORD_PROGRAM}` daemon started before this file was last written -- "
                      f"it is still on the old code; restarting it now")
                if not apply:
                    print("Pass --apply (and --yes, or confirm interactively) to restart it.")
                    return 3
                try:
                    common.require_apply_confirmation(apply, yes, what=f"restart `{_SUPERVISORD_PROGRAM}` "
                                                       "to pick up the already-applied backup-fast patch")
                except common.ApplyRefused as e:
                    print(f"zl hotfix backup-fast: {e}", file=sys.stderr)
                    return 1
                with common.ProtectedRegion() as region:
                    try:
                        with _change_lock(what="restart backup for an already-applied patch"):
                            code = _restart_if_needed(region, apply_note=f"{_TARGET} already patched")
                    except ChangeLockRefused as e:
                        print(f"zl hotfix backup-fast: refusing -- {e}", file=sys.stderr)
                        region.finish(1, restored=None)
                        return 1
                    region.finish(code, restored=None)
                    return code
            print(f"zl hotfix backup-fast: already fast ({_TARGET} already has compresslevel=1) -- nothing to do")
            return 0
        old_count = text.count(_ANCHOR_OLD)
        if old_count == 0:
            print(f"zl hotfix backup-fast: anchor line not found in {_TARGET} -- refusing, nothing changed. "
                  f"Expected exactly:\n  {_ANCHOR_OLD}", file=sys.stderr)
            return 1
        if old_count > 1:
            print(f"zl hotfix backup-fast: anchor line matched {old_count} times in {_TARGET} (expected exactly "
                  f"once) -- refusing to guess which one, nothing changed", file=sys.stderr)
            return 1
        if not apply:
            print(f"zl hotfix backup-fast: DRY RUN -- would change this one line in {_TARGET}:")
            print(f"  - {_ANCHOR_OLD}")
            print(f"  + {_ANCHOR_NEW}")
            print("Pass --apply (and --yes, or confirm interactively) to actually patch it.")
            return 3
    else:
        orig = _orig_path()
        if not orig.is_file():
            print(f"zl hotfix backup-fast: --revert: {orig} does not exist -- nothing to revert to",
                  file=sys.stderr)
            return 1
        orig_text = orig.read_text(encoding="utf-8")
        if text == orig_text:
            print(f"zl hotfix backup-fast: --revert: {_TARGET} already matches the pre-hotfix original "
                  f"({orig}) -- nothing to do")
            return 0
        expected_patched = orig_text.replace(_ANCHOR_OLD, _ANCHOR_NEW, 1)
        if text != expected_patched:
            diff = _unified_diff(
                expected_patched, text,
                a_name=f"expected ({orig} with the patch re-applied)", b_name=str(_TARGET),
            )
            print(
                f"zl hotfix backup-fast: --revert: {_TARGET} does not match what this hotfix would have "
                f"produced from {orig} -- refusing to guess which is right; diff:\n{diff}",
                file=sys.stderr,
            )
            return 1
        if not apply:
            diff = _unified_diff(text, orig_text, a_name=str(_TARGET), b_name=f"{orig} (pre-hotfix original)")
            print(f"zl hotfix backup-fast: DRY RUN -- would restore {_TARGET} from {orig} and restart "
                  f"`{_SUPERVISORD_PROGRAM}`. Diff:\n{diff}")
            print("Pass --apply (and --yes, or confirm interactively) to actually revert it.")
            return 3

    try:
        common.require_apply_confirmation(
            apply, yes,
            what=(f"revert {_TARGET} to its pre-hotfix original" if revert
                  else f"patch {_TARGET}: add compresslevel=1 (hotfix backup-fast)"),
        )
    except common.ApplyRefused as e:
        print(f"zl hotfix backup-fast: {e}", file=sys.stderr)
        return 1

    # Round-9 HIGH-1: the protected region opens HERE -- before the lock,
    # before the very first byte of `.zl-orig`/the patched file is
    # written -- not just around the eventual restart (see `_do_apply`'s
    # own docstring on why: a signal landing in that earlier window used
    # to fall back to Python's default disposition, since no handler was
    # installed yet).
    with common.ProtectedRegion() as region:
        try:
            with _change_lock(what="revert" if revert else "apply backup-fast"):
                code = _do_apply(text, revert=revert, region=region)
        except ChangeLockRefused as e:
            print(f"zl hotfix backup-fast: refusing -- {e}", file=sys.stderr)
            region.finish(1, restored=None)
            return 1
        # A no-op if `_do_apply`/`_restart_if_needed` already resolved a
        # deferred signal (region.finish() raises SystemExit there, so this
        # line is never reached in that case); otherwise this is the only
        # `finish()` call for an early refusal inside `_do_apply` itself
        # (anchor missing, compile-check failure, a revert mismatch, ...),
        # none of which ever stopped anything.
        region.finish(code, restored=None)
        return code
