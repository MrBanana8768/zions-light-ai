"""
sync_once's two-stage publish and its mutual-exclusion lock.

v3.1.9, hostile pass r318-b (reviewer B), findings F3 and F4.

F3 (HIGH for the local-disk move). sync_once used to run ONE
`src.backup(dst)` straight onto /data: Python's `Connection.backup`
defaults to `pages=-1`, a single `sqlite3_backup_step(-1)` call that holds
a SHARED lock on the SOURCE (LOCAL_DB — the database OpenWebUI is actively
writing) until the LAST byte reaches the destination. In rollback-journal
mode a writer cannot commit while a SHARED lock is held elsewhere, and
OpenWebUI's own busy timeout is 10s
(DATABASE_SQLITE_PRAGMA_BUSY_TIMEOUT). Measured by the reviewer at a
10 MB/s /data throttle: 2 of 3 writer commits failed "database is locked"
for the whole length of one sync. That is the OPPOSITE of what
WEBUI_DB_LOCAL exists to buy. The fix: stage 1 backs up to LOCAL disk
(fast — source and destination share a filesystem, so LOCAL_DB's lock is
released in around a second regardless of /data's mood); every guard runs
against that local file; only THEN does a plain, lock-free byte copy reach
/data.

F4 (LOW). The old temp name was keyed on pid and swept only on a caught
exception, so a SIGKILLed sync (a RunPod redeploy mid-sync) left a ~138 MB
temp file, and its -journal, on /data forever — nothing could find it
again, because the next process has a different pid. Two temp locations
now exist (local and /data), so the fix needs BOTH to be deterministic and
swept, and — because a manual `webuidb.py --sync-once` can run while the
daemon's own loop is mid-cycle — a lock, or two deterministic-named temps
could stomp on each other instead of ever only being their own debris.
Legacy pid-keyed debris from v3.1.7/v3.1.8 pods must also be swept, by an
EXACT pattern only.

THE SEAM. A real device throttle is not available in a unit test (see the
fix-lane brief), so slowness is injected exactly where the shipped code
held the lock, and exactly where the fix moved the write:

  * `sqlite3.connect` is wrapped (via the `factory` kwarg — `sqlite3.
    Connection` itself is an immutable C type and cannot be monkeypatched
    directly) so every connection this module opens is a `_SeamConnection`
    whose `.backup(target, ...)`, when `target` is a file under the
    simulated /data directory, holds an explicit read transaction (BEGIN +
    a read) on ITSELF for the delay before delegating to the real
    backup(). `sqlite3_backup_step(-1)` (what `pages=-1`, the shipped
    default, calls) is one opaque, atomic C call with no Python-level hook
    inside it, so a literal per-page delay cannot be injected from pure
    Python without real slow device I/O; what this reproduces instead is
    backup()'s own OBSERVABLE EFFECT on a concurrent writer for that same
    duration - a SHARED lock held on the source. This is precisely
    `src.backup(dst)` writing to /data - the shipped, single-stage shape.
    It must NEVER fire against a file under the LOCAL directory, and [2]'s
    own assertion checks that directly.
  * `shutil.copy2` is wrapped to sleep first when its destination is under
    the simulated /data directory — the fixed code's stage-2 plain copy,
    the only place a slow /data is allowed to cost time under the fix.

    python test_webuidb_sync_lock.py
"""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="sl-local-"))
_SNAP_VOL = Path(tempfile.mkdtemp(prefix="sl-moosefs-"))
_QUAR = Path(tempfile.mkdtemp(prefix="sl-forensics-"))
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUAR)
os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "0.1"
for _k in list(os.environ):
    if _k.startswith("WEBUI_DB_ALLOW_") or _k in (
        "WEBUI_DB_MAX_ROW_LOSS_BYTES", "WEBUI_DB_SHRINK_REFUSE_BELOW",
        "WEBUI_DB_SHRINK_GUARD_MIN_CHATS", "WEBUI_DB_SHRINK_GUARD_MIN_BYTES",
    ):
        os.environ.pop(_k)

import webuidb  # noqa: E402

HERE = Path(__file__).resolve().parent
LOCAL = webuidb.LOCAL_DB
SNAP = webuidb.SNAPSHOT_DB
FAILED: list[str] = []

try:
    import fcntl  # noqa: F401
    HAVE_FCNTL = True
except ImportError:
    HAVE_FCNTL = False


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def note(msg):
    print(f"  note {msg}")


# open-webui 0.11.0's `chat` table, column for column (same shape as
# test_webuidb_publish_guards.py — see that file's own comment for why
# fixtures here are OpenWebUI-shaped rather than ad hoc).
OWUI_CHAT = """create table chat (
    id varchar not null primary key,
    user_id varchar,
    title text,
    chat json,
    created_at bigint,
    updated_at bigint,
    share_id text unique,
    archived boolean,
    pinned boolean,
    meta json default '{}',
    variables json,
    folder_id text,
    tasks json,
    summary text,
    current_message_id text,
    last_read_at bigint
)"""


def conversation(n_msgs: int, text: str = "m" * 1000) -> str:
    msgs = {
        f"msg-{i:05d}": {
            "id": f"msg-{i:05d}",
            "parentId": f"msg-{i - 1:05d}" if i else None,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": text,
        }
        for i in range(n_msgs)
    }
    return json.dumps({"history": {"messages": msgs}})


def owui(path: Path, rows) -> None:
    """rows: (id, updated_at, chat_json). Replaces any file there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wipe_one(path)
    con = sqlite3.connect(str(path))
    con.execute(OWUI_CHAT)
    con.execute("create table user (id text)")
    con.execute("insert into user values ('u1')")
    for cid, updated_at, body in rows:
        con.execute(
            "insert into chat (id, user_id, title, chat, created_at, updated_at, "
            "archived, pinned, meta) values (?, 'u1', 'a conversation', ?, 1, ?, "
            "0, 0, '{}')",
            (cid, body, updated_at),
        )
    con.commit()
    con.close()


def wipe_one(path: Path) -> None:
    for suffix in ("",) + webuidb.SIDECARS:
        f = path.with_name(path.name + suffix)
        if f.exists():
            f.unlink()


def wipe():
    wipe_one(LOCAL)
    wipe_one(SNAP)
    # Pre-create the /data directory, matching real deployments (entrypoint.sh
    # / a previous sync already made it). Without this, the F3 mutation
    # (stage 1 backing up directly onto /data again) fails on "unable to
    # open database file" - a missing-directory accident, not the lock
    # contention this suite exists to catch - before the slow seam is ever
    # reached.
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    # This module's own deterministic temps and lock file, and any legacy
    # debris — a scenario earlier in this file must never leak state into
    # the next one.
    for p in (
        LOCAL.with_name(LOCAL.name + webuidb._SYNC_TMP_SUFFIX),
        LOCAL.with_name(LOCAL.name + webuidb._SYNC_TMP_SUFFIX + "-journal"),
        SNAP.with_name(SNAP.name + webuidb._SYNC_TMP_SUFFIX),
        SNAP.with_name(SNAP.name + webuidb._SYNC_TMP_SUFFIX + "-journal"),
        webuidb._sync_lock_path(),
    ):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    if SNAP.parent.is_dir():
        for entry in SNAP.parent.iterdir():
            if re.match(r"^webui\.db\.sync-\d+(-journal)?$", entry.name):
                entry.unlink()
    if webuidb.EMPTY_START_MARKER.exists():
        webuidb.EMPTY_START_MARKER.unlink()
    webuidb._forensic_copy_last_monotonic = None


# ---------------------------------------------------------------------------
# The slow-/data seam (F3). See the module docstring for why `sqlite3.
# Connection` needs a factory subclass rather than a direct monkeypatch.
# ---------------------------------------------------------------------------

class _Seam:
    def __init__(self, slow_dir: Path, delay_s: float):
        self.slow_dir = slow_dir.resolve()
        self.delay_s = delay_s
        self.backup_hits: list[str] = []
        self.copy_hits: list[str] = []

    def _is_slow(self, path_str) -> bool:
        try:
            return str(Path(path_str).resolve()).startswith(str(self.slow_dir))
        except Exception:
            return False


_SEAM: _Seam | None = None
_real_connect = sqlite3.connect
_real_copy2 = shutil.copy2


class _SeamConnection(sqlite3.Connection):
    def backup(self, target, **kwargs):
        if _SEAM is not None:
            try:
                row = target.execute("PRAGMA database_list").fetchone()
                tpath = row[2] if row else None
            except Exception:
                tpath = None
            if tpath and _SEAM._is_slow(tpath):
                _SEAM.backup_hits.append(tpath)
                # THE SLOWNESS MUST SIT WHERE THE SHIPPED CODE HELD THE
                # LOCK, not merely "somewhere around this call" - a sleep
                # BEFORE calling the real backup() proved to be a wrong-
                # reason green (mutation-tested: it let the writer through
                # completely unaffected, because no lock is held until
                # backup() itself runs). `sqlite3_backup_step(-1)` (what
                # `.backup()` calls with the shipped `pages=-1` default) is
                # ONE opaque, atomic C call with no Python-level hook
                # inside it, so a literal per-page delay cannot be injected
                # from pure Python without real slow device I/O (not
                # available in a unit test). What CAN be reproduced exactly
                # is the OBSERVABLE CONSEQUENCE that call has on a
                # concurrent writer: for as long as it runs, it holds a
                # SHARED read lock on `self` (the source). An explicit
                # BEGIN + a read, held open for `delay_s`, is that same
                # lock, held for the same duration - which is precisely
                # what a concurrent writer's busy_timeout experiences
                # either way.
                self.execute("BEGIN")
                try:
                    self.execute("SELECT count(*) FROM sqlite_master")
                    time.sleep(_SEAM.delay_s)
                finally:
                    self.execute("COMMIT")
        return super().backup(target, **kwargs)


def _seam_connect(database, *a, **kw):
    kw.setdefault("factory", _SeamConnection)
    return _real_connect(database, *a, **kw)


def _seam_copy2(src, dst, *a, **kw):
    if _SEAM is not None and _SEAM._is_slow(str(dst)):
        _SEAM.copy_hits.append(str(dst))
        time.sleep(_SEAM.delay_s)
    return _real_copy2(src, dst, *a, **kw)


def install_seam(slow_dir: Path, delay_s: float) -> _Seam:
    global _SEAM
    _SEAM = _Seam(slow_dir, delay_s)
    sqlite3.connect = _seam_connect
    shutil.copy2 = _seam_copy2
    return _SEAM


def uninstall_seam() -> None:
    global _SEAM
    sqlite3.connect = _real_connect
    shutil.copy2 = _real_copy2
    _SEAM = None


# ---------------------------------------------------------------------------
# An OpenWebUI-shaped concurrent writer: busy_timeout, delete journal,
# rewriting one row in a loop under BEGIN IMMEDIATE — the same shape the
# reviewer's lm.py used (SP\\r318-b\\scripts\\lm.py, mode "move").
# ---------------------------------------------------------------------------

def make_writer(path: Path, chat_id: str, busy_timeout_ms: int, period_s: float):
    stop = threading.Event()
    results = {"commits": 0, "errors": []}

    def run():
        con = _real_connect(str(path), timeout=busy_timeout_ms / 1000.0)
        con.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        con.isolation_level = None  # autocommit; we manage BEGIN/COMMIT ourselves
        k = 0
        while not stop.is_set():
            k += 1
            try:
                con.execute("BEGIN IMMEDIATE")
                con.execute(
                    "UPDATE chat SET chat=?, updated_at=? WHERE id=?",
                    (conversation(20, text=f"w{k}" * 40), int(time.time()), chat_id),
                )
                con.execute("COMMIT")
                results["commits"] += 1
            except Exception as e:
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass
                results["errors"].append(f"{type(e).__name__}: {e}")
            time.sleep(period_s)
        con.close()

    th = threading.Thread(target=run, daemon=True)
    return th, stop, results


print("=" * 70)
print("[1] the lock file lives on LOCAL disk, never /data")
print("=" * 70)
check(
    webuidb._sync_lock_path().parent == LOCAL.parent,
    f"lock file parent ({webuidb._sync_lock_path().parent}) is LOCAL_DB's "
    f"own directory, not SNAPSHOT_DB's ({SNAP.parent}) - a lock on the "
    f"volume this module routes AROUND would itself be subject to the "
    f"same stalls",
)

print()
print("=" * 70)
print("[2] F3: a slow /data never blocks an OpenWebUI-shaped writer")
print("=" * 70)
wipe()
CHAT_ID = "the-live-conversation"
owui(LOCAL, [(CHAT_ID, 1000, conversation(1500, text="x" * 900))])
before_size_mb = LOCAL.stat().st_size / 1e6
note(f"fixture: local database is {before_size_mb:.1f} MB")

writer_th, writer_stop, writer_results = make_writer(
    LOCAL, CHAT_ID, busy_timeout_ms=1000, period_s=0.03
)
writer_th.start()
time.sleep(0.3)  # let the writer get a baseline commit or two in first

SLOW_DELAY_S = 2.5
seam = install_seam(SNAP.parent, SLOW_DELAY_S)
t0 = time.monotonic()
try:
    r = webuidb.sync_once(force=True)
finally:
    elapsed = time.monotonic() - t0
    uninstall_seam()
    writer_stop.set()
    writer_th.join(timeout=10)

check(r["synced"] is True, f"the sync itself succeeded (r={r})")
check(
    not writer_results["errors"],
    f"NO writer commit failed while /data was slow for {SLOW_DELAY_S}s "
    f"({writer_results['errors'][:3]}) - this is the finding: the shipped "
    f"code held LOCAL_DB's SHARED lock for the length of the /data write, "
    f"which is exactly what WEBUI_DB_LOCAL exists to avoid",
)
check(
    writer_results["commits"] >= 2,
    f"CONTROL: the writer was actually running and committing throughout "
    f"({writer_results['commits']} commits) - a silently-dead writer would "
    f"pass the check above for the wrong reason",
)
check(
    not seam.backup_hits,
    f"sqlite3 backup() never targeted a file under /data (hits={seam.backup_hits}) "
    f"- stage 1 backs up to LOCAL disk only; the whole point of the fix is "
    f"that /data never sees a live sqlite3 connection",
)
check(
    bool(seam.copy_hits),
    f"the seam actually engaged on the stage-2 copy (hits={seam.copy_hits}) "
    f"- a green result above with an INERT seam would prove nothing",
)
check(
    elapsed >= SLOW_DELAY_S,
    f"the sync's own wall-clock time ({elapsed:.2f}s) includes the "
    f"simulated slow /data write ({SLOW_DELAY_S}s) - the seam really did "
    f"cost the sync time, it just did not cost the WRITER anything",
)
check(
    not LOCAL.with_name(LOCAL.name + webuidb._SYNC_TMP_SUFFIX).exists()
    and not SNAP.with_name(SNAP.name + webuidb._SYNC_TMP_SUFFIX).exists(),
    "both temp files are gone after a successful publish - nothing left "
    "for the next cycle to sweep",
)

print()
print("=" * 70)
print("[3] F4: sync_once's own deterministic temps are swept at the start")
print("=" * 70)
wipe()
owui(LOCAL, [("c1", 100, conversation(20))])
local_debris = LOCAL.with_name(LOCAL.name + webuidb._SYNC_TMP_SUFFIX)
local_debris_journal = LOCAL.with_name(LOCAL.name + webuidb._SYNC_TMP_SUFFIX + "-journal")
data_debris = SNAP.with_name(SNAP.name + webuidb._SYNC_TMP_SUFFIX)
data_debris_journal = SNAP.with_name(SNAP.name + webuidb._SYNC_TMP_SUFFIX + "-journal")
SNAP.parent.mkdir(parents=True, exist_ok=True)
local_debris.write_bytes(b"leftover from a SIGKILLed sync" * 1000)
local_debris_journal.write_bytes(b"stale journal")
data_debris.write_bytes(b"leftover on /data from a SIGKILLed sync" * 1000)
data_debris_journal.write_bytes(b"stale journal")
check(
    local_debris.exists() and data_debris.exists(),
    "fixture: debris is actually on disk in both locations before the sweep",
)
webuidb._sweep_stale_sync_temps()
check(
    not local_debris.exists() and not local_debris_journal.exists(),
    "the local debris (and its -journal) is gone after the sweep",
)
check(
    not data_debris.exists() and not data_debris_journal.exists(),
    "the /data debris (and its -journal) is gone after the sweep",
)
check(
    SNAP.parent.exists(),  # sweeping must not remove the directory itself
    "CONTROL: the sweep did not touch anything it should not have "
    "(the snapshot directory itself still exists)",
)

print()
print("=" * 70)
print("[4] F4: the legacy pid-keyed pattern is swept - EXACTLY that pattern")
print("=" * 70)
wipe()
owui(LOCAL, [("c1", 100, conversation(20))])
SNAP.parent.mkdir(parents=True, exist_ok=True)
legacy = SNAP.with_name(f"{SNAP.name}.sync-4181")
legacy_journal = SNAP.with_name(f"{SNAP.name}.sync-4181-journal")
legacy.write_bytes(b"v3.1.7/v3.1.8 leftover" * 1000)
legacy_journal.write_bytes(b"stale journal")
# Decoys: everything below must SURVIVE the sweep.
decoy_snapshot_itself = SNAP  # must never be swept - it is not debris, it IS the db
decoy_nondigit = SNAP.with_name(f"{SNAP.name}.sync-abc")
decoy_trailing = SNAP.with_name(f"{SNAP.name}.sync-42-old")
decoy_nondigit.write_bytes(b"not a pid")
decoy_trailing.write_bytes(b"not the exact pattern")
# The snapshot itself must exist for this to be a meaningful decoy - use the
# migration path to put a real one there first.
owui(SNAP, [("prior", 50, conversation(5))])
check(SNAP.exists(), "fixture: a real snapshot is present before the sweep")

webuidb._sweep_stale_sync_temps()
check(
    not legacy.exists() and not legacy_journal.exists(),
    "the legacy pid-keyed debris (and its -journal) is gone",
)
check(
    decoy_nondigit.exists() and decoy_trailing.exists(),
    "CONTROL: names that only LOOK like the legacy pattern survive "
    "(non-digit suffix, and digits-plus-more) - the match must be exact",
)
check(
    SNAP.exists(),
    "CONTROL: the snapshot itself was never touched by the sweep",
)
decoy_nondigit.unlink()
decoy_trailing.unlink()

print()
print("=" * 70)
print("[4b] F4: sync_once ITSELF sweeps at the start of its own cycle")
print("=" * 70)
# [3] and [4] above call _sweep_stale_sync_temps() directly - a real unit
# test of the function, but one that CANNOT tell "sync_once calls this at
# the start of every cycle" from "sync_once doesn't call it, but happens to
# overwrite its own deterministic-named debris anyway as an ordinary side
# effect of backing up onto that same path". Legacy pid-keyed debris is
# never touched by the ordinary flow (sync_once never writes THAT name), so
# it can only disappear here if the call site inside sync_once actually
# fires - this is what distinguishes the fix from a mutant that keeps the
# sweep FUNCTION correct but never calls it (F4's own accumulation bug,
# reintroduced).
wipe()
owui(LOCAL, [("c1", 100, conversation(20))])
SNAP.parent.mkdir(parents=True, exist_ok=True)
legacy2 = SNAP.with_name(f"{SNAP.name}.sync-7777")
legacy2_journal = SNAP.with_name(f"{SNAP.name}.sync-7777-journal")
legacy2.write_bytes(b"v3.1.7/v3.1.8 leftover" * 1000)
legacy2_journal.write_bytes(b"stale journal")
check(legacy2.exists(), "fixture: legacy debris present before a real sync_once cycle")
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"the sync itself still succeeds (r={r})")
check(
    not legacy2.exists() and not legacy2_journal.exists(),
    "the legacy debris is gone after a REAL sync_once call - proving the "
    "sweep runs from sync_once's own call site, not only when called "
    "directly",
)

print()
print("=" * 70)
print("[5] F4: mutual exclusion - a second concurrent sync_once is a clean skip")
print("=" * 70)
if not HAVE_FCNTL:
    note("fcntl is not available on this platform (Windows) - flock cannot "
         "be exercised for real here; see _acquire_sync_lock's own "
         "degrade-safely doctrine. Checking only the degraded path below "
         "(production is Linux; the whole-suite Docker run covers the real "
         "lock).")
    lock = webuidb._acquire_sync_lock()
    check(
        lock is webuidb._LOCK_DEGRADED,
        "on a platform with no fcntl, the lock function degrades to "
        "_LOCK_DEGRADED rather than ever refusing a sync",
    )
    webuidb._release_sync_lock(lock)
else:
    wipe()
    owui(LOCAL, [("c1", 100, conversation(20))])
    held = webuidb._acquire_sync_lock()
    check(held not in (None, webuidb._LOCK_DEGRADED), "the first lock acquire succeeds")

    r = webuidb.sync_once(force=True)
    check(
        r["skipped"] == "another sync is in progress" and r["error"] is None
        and r["synced"] is False,
        f"a sync_once call while the lock is held is a clean SKIP, not an "
        f"error (r={r})",
    )
    check(
        not LOCAL.with_name(LOCAL.name + webuidb._SYNC_TMP_SUFFIX).exists(),
        "the blocked call never touched any temp file - it returned before "
        "sweeping or backing anything up",
    )
    webuidb._release_sync_lock(held)

    check(
        webuidb._acquire_sync_lock() is not None
        and (webuidb._release_sync_lock(webuidb._acquire_sync_lock()) or True),
        "CONTROL: after releasing, the lock is acquirable again",
    )
    r2 = webuidb.sync_once(force=True)
    check(
        r2["synced"] is True,
        f"CONTROL: with nothing else holding the lock, sync_once "
        f"publishes normally (r={r2}) - the mechanism is not 'refuse "
        f"everything'",
    )

print()
print("=" * 70)
print("[6] F4: a hard-killed sync leaves no permanent debris")
print("=" * 70)
wipe()
owui(LOCAL, [("c1", 5000, conversation(4000, text="k" * 500))])
note(f"fixture: local database is {LOCAL.stat().st_size / 1e6:.1f} MB")
child_script = HERE / "_sync_lock_kill_child.py"
ready_marker = Path(tempfile.mkdtemp(prefix="sl-ready-")) / "ready"
child_script.write_bytes((
    "import sys, time, shutil\n"
    f"sys.path.insert(0, {str(HERE)!r})\n"
    "_real_copy2 = shutil.copy2\n"
    "def _slow_copy2(src, dst, *a, **kw):\n"
    f"    open({str(ready_marker)!r}, 'w').close()\n"
    "    time.sleep(30)\n"
    "    return _real_copy2(src, dst, *a, **kw)\n"
    "shutil.copy2 = _slow_copy2\n"
    "import webuidb\n"
    "print(webuidb.sync_once(force=True))\n"
).encode("utf-8"))
env = dict(os.environ)
child = subprocess.Popen(
    [sys.executable, str(child_script)], cwd=str(HERE), env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
deadline = time.monotonic() + 15
while time.monotonic() < deadline and not ready_marker.exists():
    time.sleep(0.05)
reached_stage2 = ready_marker.exists()
child.kill()
child.wait(timeout=10)
try:
    child_script.unlink()
except OSError:
    pass

check(reached_stage2, "fixture: the child reached stage 2 before being killed")
local_tmp_path = LOCAL.with_name(LOCAL.name + webuidb._SYNC_TMP_SUFFIX)
check(
    local_tmp_path.exists(),
    "the killed child's local-disk temp is left behind, as debris - this "
    "is the shape of the finding (previously this was 138 MB on /data "
    "forever; now it is on local disk and swept, see below)",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True,
    f"a fresh sync_once, in-process, succeeds cleanly right after (r={r}) "
    f"- the debris from the killed child did not wedge anything",
)
check(
    not local_tmp_path.exists()
    and not SNAP.with_name(SNAP.name + webuidb._SYNC_TMP_SUFFIX).exists(),
    "and the debris is gone - swept at the start of this cycle, then this "
    "cycle's own temps cleaned up on its own success",
)

print()
print("=" * 70)
print("[7] a failed FILE fsync refuses the publish, not a warn-and-continue")
print("=" * 70)
# A hash match right after a fsync that failed can still be a page-cache
# read matching a page-cache read - it proves nothing about durability. See
# _fsync_file_and_dir's own docstring. This seam intercepts os.open/os.fsync/
# os.close (the exact primitives that function uses) rather than
# monkeypatching os.fsync alone, because fsync only ever receives an int fd -
# tracking which fd belongs to which PATH is the only way to fail exactly
# one file's fsync (data_tmp) without also breaking the lock file, the
# directory fsync, or anything else in the process that happens to call
# os.fsync during this run.
_real_os_open = os.open
_real_os_fsync = os.fsync
_real_os_close = os.close
_fd_to_path: dict[int, str] = {}
_fsync_fail_path: str | None = None


def _seam_os_open(path, flags, *a, **kw):
    fd = _real_os_open(path, flags, *a, **kw)
    _fd_to_path[fd] = os.fspath(path) if hasattr(path, "__fspath__") else str(path)
    return fd


def _seam_os_fsync(fd):
    if _fsync_fail_path is not None and _fd_to_path.get(fd) == _fsync_fail_path:
        raise OSError(5, "Input/output error")  # EIO - a stalling mount
    return _real_os_fsync(fd)


def _seam_os_close(fd):
    _fd_to_path.pop(fd, None)
    return _real_os_close(fd)


def install_fsync_seam(fail_path: str) -> None:
    global _fsync_fail_path
    _fsync_fail_path = fail_path
    os.open = _seam_os_open
    os.fsync = _seam_os_fsync
    os.close = _seam_os_close


def uninstall_fsync_seam() -> None:
    global _fsync_fail_path
    os.open = _real_os_open
    os.fsync = _real_os_fsync
    os.close = _real_os_close
    _fsync_fail_path = None
    _fd_to_path.clear()


wipe()
owui(LOCAL, [("c1", 100, conversation(50))])
owui(SNAP, [("prior", 90, conversation(10))])
before_snapshot_bytes = SNAP.read_bytes()
before_local_bytes = LOCAL.read_bytes()

data_tmp_path = str(SNAP.with_name(SNAP.name + webuidb._SYNC_TMP_SUFFIX))
install_fsync_seam(fail_path=data_tmp_path)
try:
    r = webuidb.sync_once(force=True)
finally:
    uninstall_fsync_seam()

check(
    r["synced"] is False and r["error"] is not None,
    f"a failed fsync of data_tmp refuses the publish (r={r})",
)
check(
    "fsync" in (r["error"] or "").lower(),
    f"...and the error names fsync, not some unrelated failure (error="
    f"{r['error']!r})",
)
check(
    SNAP.read_bytes() == before_snapshot_bytes,
    "the OLD snapshot is untouched, byte for byte - os.replace never ran",
)
check(
    not LOCAL.with_name(LOCAL.name + webuidb._SYNC_TMP_SUFFIX).exists()
    and not SNAP.with_name(SNAP.name + webuidb._SYNC_TMP_SUFFIX).exists(),
    "both temps are cleaned up despite the refusal - same finally as every "
    "other publish refusal in this function",
)
check(
    LOCAL.read_bytes() == before_local_bytes,
    "CONTROL/sanity: the live local database itself was never touched by "
    "any of this",
)

print("    CONTROL: the SAME fixture, fsync working normally, publishes")
r2 = webuidb.sync_once(force=True)
check(
    r2["synced"] is True,
    f"unpatched, this exact fixture publishes fine (r={r2}) - the refusal "
    f"above was specific to the simulated fsync failure, not something "
    f"else about this fixture",
)

print()
if FAILED:
    print("!" * 70)
    for f in FAILED:
        print("FAIL " + f)
    print("!" * 70)
    sys.exit(1)
print("All sync-lock (F3/F4) checks passed.")
