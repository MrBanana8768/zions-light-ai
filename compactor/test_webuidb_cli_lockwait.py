"""v3.1.9.10 fix round A, R3 H1: `webuidb.py --sync-once` must never report
success it did not achieve.

`--sync-once --force` used to exit 0 on `skipped: another sync is in
progress` (backup.py holds the sync flock across its MooseFS copy), so the
runbook's manual "final sync" could be a lie. Now:

  * the CLI WAITS for the flock (polling, env WEBUI_DB_SYNC_ONCE_LOCK_WAIT_S,
    default 180 s) and then publishes;
  * exit 0 ONLY for `synced: True`, or an "unchanged" skip that is VERIFIED
    (content fingerprint of local == snapshot: chats / content bytes /
    newest_update, the fields `--status` prints); everything else exits 1;
  * the SIGTERM final sync also waits (bounded) for the lock.

    python test_webuidb_cli_lockwait.py
"""

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="lw-local-"))
_SNAP_VOL = Path(tempfile.mkdtemp(prefix="lw-moosefs-"))
_QUAR = Path(tempfile.mkdtemp(prefix="lw-forensics-"))
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUAR)
os.environ["WEBUI_DB_LOCAL"] = "true"
for _k in list(os.environ):
    if _k.startswith("WEBUI_DB_ALLOW_"):
        os.environ.pop(_k)

import webuidb  # noqa: E402

try:
    import fcntl  # noqa: F401
except ImportError:
    print("no fcntl on this platform: the lock cannot be exercised; skipped")
    sys.exit(0)

HERE = Path(__file__).resolve().parent
LOCAL, SNAP = webuidb.LOCAL_DB, webuidb.SNAPSHOT_DB
FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


OWUI_CHAT = """create table chat (
    id varchar not null primary key, user_id varchar, title text, chat json,
    created_at bigint, updated_at bigint, share_id text unique, archived boolean,
    pinned boolean, meta json default '{}', variables json, folder_id text,
    tasks json, summary text, current_message_id text, last_read_at bigint)"""


def make(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suf in ("",) + webuidb.SIDECARS:
        f = path.with_name(path.name + suf)
        if f.exists():
            f.unlink()
    con = sqlite3.connect(str(path))
    con.execute(OWUI_CHAT)
    con.execute("create table user (id text)")
    for cid, upd in rows:
        con.execute(
            "insert into chat (id,user_id,title,chat,created_at,updated_at,archived,pinned,meta)"
            " values (?, 'u1','t',?,1,?,0,0,'{}')",
            (cid, json.dumps({"history": {"messages": {"m": {"content": "x" * 400}}}}), upd))
    con.commit()
    con.close()


def reset(local_rows, snap_rows):
    make(LOCAL, local_rows)
    make(SNAP, snap_rows)
    lp = webuidb._sync_lock_path()
    if lp.exists():
        lp.unlink()
    os.utime(SNAP, (time.time() - 600, time.time() - 600))


def hold_lock(seconds: float):
    """Another PROCESS holding the sync flock (as backup.py does)."""
    code = (
        "import fcntl,sys,time\n"
        f"fh=open({str(webuidb._sync_lock_path())!r},'a+')\n"
        "fcntl.flock(fh.fileno(), fcntl.LOCK_EX)\n"
        "print('held',flush=True)\n"
        f"time.sleep({seconds})\n")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def run_cli(extra_env):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(HERE), PYTHONIOENCODING="utf-8")
    env.update(extra_env)
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "webuidb.py", "--sync-once"], cwd=str(HERE),
                       env=env, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr, time.monotonic() - t0


def fp_equal():
    return webuidb.content_fingerprint(LOCAL) == webuidb.content_fingerprint(SNAP)


print("[1] the lock stays held past the wait: exit NON-ZERO, nothing published, clear message")
reset([("c1", 100), ("c2", 200)], [("c1", 100)])
h = hold_lock(60)
try:
    rc, out, err, dt = run_cli({"WEBUI_DB_SYNC_ONCE_LOCK_WAIT_S": "1.5"})
finally:
    h.kill(); h.wait()
check(rc == 1, f"exit 1 while another sync holds the lock (rc={rc})")
check("another sync is in progress" in out and "NOT PUBLISHED" in err,
      f"clear message on stderr, result on stdout (err={err.strip()!r})")
check(dt >= 1.4, f"it really waited for the lock (took {dt:.1f}s)")
check(not fp_equal(), "the snapshot was not touched (still missing c2)")

print("[2] the holder lets go inside the wait: the CLI publishes and exits 0")
reset([("c1", 100), ("c2", 200)], [("c1", 100)])
h = hold_lock(2)
try:
    rc, out, err, dt = run_cli({"WEBUI_DB_SYNC_ONCE_LOCK_WAIT_S": "30"})
finally:
    h.kill(); h.wait()
check(rc == 0 and "'synced': True" in out, f"waited, then published, exit 0 (rc={rc}, out={out.strip()!r})")
check(dt >= 1.5, f"and it did wait for the holder (took {dt:.1f}s)")
check(fp_equal(), "snapshot fingerprint now equals local (chats/content/newest_update)")

print("[3] the exit decision, unit: only synced or a VERIFIED unchanged is 0")
reset([("c1", 100)], [("c1", 100)])
base = {"synced": False, "skipped": None, "error": None, "bytes": 0}
code, _ = webuidb.sync_once_cli_exit({**base, "synced": True})
check(code == 0, "synced:True -> 0")
code, msg = webuidb.sync_once_cli_exit({**base, "skipped": "unchanged since last sync"})
check(code == 0 and "verified" in msg, f"unchanged AND fingerprints equal -> 0 ({msg})")
make(LOCAL, [("c1", 100), ("c2", 300)])
code, msg = webuidb.sync_once_cli_exit({**base, "skipped": "unchanged since last sync"})
check(code == 1 and "NOT VERIFIED" in msg, f"'unchanged' but local != snapshot -> 1 ({msg})")
code, _ = webuidb.sync_once_cli_exit({**base, "skipped": "another sync is in progress"})
check(code == 1, "busy lock -> 1")
code, _ = webuidb.sync_once_cli_exit({**base, "skipped": "no local database yet"})
check(code == 1, "no local database -> 1")
code, _ = webuidb.sync_once_cli_exit({**base, "error": "boom"})
check(code == 1, "error -> 1")
code, _ = webuidb.sync_once_cli_exit({**base})
check(code == 1, "an empty/unknown result -> 1, never a default pass")

print("[4] _acquire_sync_lock(wait_s): polls, then gives up")
reset([("c1", 100)], [("c1", 100)])
h = hold_lock(60)
try:
    t0 = time.monotonic()
    got = webuidb._acquire_sync_lock(0)
    check(got is None and time.monotonic() - t0 < 0.5, "wait 0 = the old single non-blocking try")
    t0 = time.monotonic()
    got = webuidb._acquire_sync_lock(1.0)
    check(got is None and 0.9 <= time.monotonic() - t0 < 5, "wait 1 s on a stuck holder returns None after ~1 s")
finally:
    h.kill(); h.wait()

print("[5] the SIGTERM final sync waits for the lock instead of skipping")
reset([("c1", 100), ("c2", 200)], [("c1", 100)])
os.environ["WEBUI_DB_SHUTDOWN_LOCK_WAIT_S"] = "20"
h = hold_lock(2)
try:
    webuidb._final_sync_on_shutdown()
finally:
    h.kill(); h.wait()
check(fp_equal(), "the final sync waited for the holder and then published")

reset([("c1", 100), ("c2", 200)], [("c1", 100)])
os.environ["WEBUI_DB_SHUTDOWN_LOCK_WAIT_S"] = "1"
h = hold_lock(60)
try:
    t0 = time.monotonic()
    webuidb._final_sync_on_shutdown()
    dt = time.monotonic() - t0
finally:
    h.kill(); h.wait()
check(1.0 <= dt < 10, f"bounded: a stuck holder costs ~the wait, not forever ({dt:.1f}s)")
check(not fp_equal(), "and nothing was published under the held lock")

print()
if FAILED:
    print("!" * 70)
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All --sync-once lock-wait checks passed.")
