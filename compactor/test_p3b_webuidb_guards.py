"""
p3-b (hostile pass #3, reviewer B) F6 + F7 — webuidb.sync_once/sync_loop
against a flaky snapshot mount.

Ported from SP\\p3-b\\wdb_guards.py:

  F6: a single transient read error on the FIRST connection opened against
  the snapshot ("disk I/O error", the exact stall this module exists for),
  followed by a clean second read a moment later, used to be read as "the
  snapshot opens cleanly but has no chat table" — which switched off the
  shrink ratio, the per-row loss limit and the generation guard together
  and let the publish replace 400 chats with the local database's one row.

  F7: on Python 3.12.3 (every image), Path.exists()/Path.stat() RAISE for
  EIO rather than returning False, and the mtime-skip check ran BEFORE
  sync_once's own try — so an EIO on the snapshot path escaped sync_once
  entirely and killed sync_loop (exit 1), where supervisord's autorestart
  papered over it silently: consecutive_failures lives only in the process
  that just died, so the "publish has failed N times" alarm never fires.

Run inside the compactor image or any container with the requirements
installed (Linux only — subprocess-based sync_loop check):
    python test_p3b_webuidb_guards.py
"""

import errno
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-p3b-wdb-test-"))
_LOCAL = _ROOT / "local" / "webui.db"
_SNAP = _ROOT / "data" / "openwebui" / "webui.db"
_Q = _ROOT / "data" / "forensics"

os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL)
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP)
os.environ["WEBUI_DB_QUARANTINE"] = str(_Q)
os.environ["WEBUI_DB_EMPTY_START_MARKER"] = str(_ROOT / "local" / ".empty-start")
os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "0.2"

import webuidb  # noqa: E402


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _mk(p: Path, n: int, size: int) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    c = sqlite3.connect(str(p))
    c.execute("create table chat (id text primary key, chat text, updated_at integer)")
    c.executemany(
        "insert into chat values (?,?,?)",
        [(f"id{i}", "h" * size, 1_700_000_000 + i) for i in range(n)],
    )
    c.commit()
    c.close()


def _snap_state() -> dict:
    c = sqlite3.connect(str(_SNAP))
    n = c.execute("select count(*), sum(length(chat)) from chat").fetchone()
    c.close()
    return {"chats": n[0], "content_bytes": n[1]}


def _scene() -> None:
    """Her 400-chat history on the snapshot; a failed-restore-shaped local
    db (one tiny new chat). Local is newer, so the mtime skip does not
    fire — this is the state a transient read error on the snapshot has to
    survive without publishing over."""
    _mk(_SNAP, 400, 20_000)
    _mk(_LOCAL, 1, 500)
    t = time.time() - 3600
    os.utime(_SNAP, (t, t))


def test_transient_read_error_on_snapshot_holds_the_publish():
    print("\n[test] F6: one transient 'disk I/O error' on the snapshot does not disable the guards")
    _scene()
    before = _snap_state()

    real_connect = webuidb.sqlite3.connect
    fired = {"n": 0}

    def flaky(path, *a, **k):
        if str(path) == str(_SNAP) and fired["n"] == 0:
            fired["n"] += 1
            raise sqlite3.OperationalError("disk I/O error")
        return real_connect(path, *a, **k)

    webuidb.sqlite3.connect = flaky
    try:
        r = webuidb.sync_once()
    finally:
        webuidb.sqlite3.connect = real_connect

    assert_eq(fired["n"], 1, "fixture: the fault fired exactly once")
    assert_eq(r["synced"], False,
              f"F6 fix: the publish is REFUSED, not let through unguarded (got {r})")
    assert_true(bool(r["error"]) and "could not be confirmed" in r["error"],
                f"and the refusal names the real reason (got {r['error']!r})")
    assert_eq(_snap_state(), before,
              "F6 fix: the snapshot (400 chats, her real history) is untouched")


def test_control_no_fault_still_refuses_on_its_own_terms():
    print("\n[test] F6 CONTROL: with no fault injected, the ordinary chat-count guard still refuses")
    _scene()
    before = _snap_state()
    r = webuidb.sync_once()
    assert_eq(r["synced"], False, f"CONTROL: refused (got {r})")
    assert_true("chat count" in (r["error"] or ""), f"CONTROL: for the ordinary reason (got {r['error']!r})")
    assert_eq(_snap_state(), before, "CONTROL: snapshot untouched")


def test_control_a_genuinely_empty_snapshot_still_publishes():
    print("\n[test] F6 CONTROL: a genuinely schema-only snapshot (confirmed, no fault) still publishes")
    _mk(_LOCAL, 1, 500)
    _SNAP.parent.mkdir(parents=True, exist_ok=True)
    if _SNAP.exists():
        _SNAP.unlink()
    c = sqlite3.connect(str(_SNAP))
    c.execute("create table other (x text)")
    c.commit()
    c.close()
    r = webuidb.sync_once(force=True)
    assert_eq(r["synced"], True, f"CONTROL: a genuinely empty (no chat table) snapshot still publishes (got {r})")


def test_eio_on_snapshot_stat_does_not_escape_sync_once():
    print("\n[test] F7: an EIO on Path.stat(snapshot) during the mtime-skip check does not escape sync_once")
    _scene()
    real_stat = pathlib.Path.stat

    def eio_stat(self, *a, **k):
        if str(self) == str(_SNAP):
            raise OSError(errno.EIO, "Input/output error", str(self))
        return real_stat(self, *a, **k)

    pathlib.Path.stat = eio_stat
    try:
        try:
            r = webuidb.sync_once()
        except BaseException as e:
            assert_true(False, f"F7 fix: sync_once must not raise (it did: {type(e).__name__}: {e})")
    finally:
        pathlib.Path.stat = real_stat

    assert_eq(r["synced"], False, f"F7 fix: sync_once returned a report, refused (got {r})")
    assert_true(bool(r["error"]), f"and it is recorded as a failure (got {r})")


def test_sync_loop_survives_a_sustained_eio_on_the_snapshot():
    print("\n[test] F7: sync_loop, in a real child process, survives a SUSTAINED EIO on the snapshot")
    _scene()
    child_code = (
        "import os, sys, errno, pathlib\n"
        "sys.path.insert(0, os.getcwd())\n"
        "import webuidb\n"
        "SNAP = str(webuidb.SNAPSHOT_DB)\n"
        "real = pathlib.Path.stat\n"
        "def eio(self, *a, **k):\n"
        "    if str(self) == SNAP:\n"
        "        raise OSError(errno.EIO, 'Input/output error', SNAP)\n"
        "    return real(self, *a, **k)\n"
        "pathlib.Path.stat = eio\n"
        "webuidb.sync_loop()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=dict(os.environ),
    )
    # WEBUI_DB_SYNC_INTERVAL_S=0.2 -> several cycles fit in well under a
    # second. If the pre-fix bug is present, the child exits almost
    # immediately (first EIO); waiting well past several cycle-lengths and
    # still finding it alive is the proof it did not.
    time.sleep(2.0)
    still_alive = proc.poll() is None
    proc.terminate()
    try:
        _, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr = proc.communicate()

    assert_true(still_alive,
                f"F7 fix: sync_loop is STILL RUNNING after ~10 sync cycles under sustained EIO "
                f"(it was not; stderr tail: {stderr.strip().splitlines()[-5:]})")
    cycles_logged = stderr.count("cannot stat") + stderr.count("sync_once raised")
    assert_true(cycles_logged >= 3,
                f"and it actually retried multiple times, not just survived once "
                f"(counted {cycles_logged} failure lines in stderr)")


if __name__ == "__main__":
    tests = [
        test_transient_read_error_on_snapshot_holds_the_publish,
        test_control_no_fault_still_refuses_on_its_own_terms,
        test_control_a_genuinely_empty_snapshot_still_publishes,
        test_eio_on_snapshot_stat_does_not_escape_sync_once,
        test_sync_loop_survives_a_sustained_eio_on_the_snapshot,
    ]
    for t in tests:
        t()
    print("\nAll p3-b webuidb sync guard (F6/F7) tests passed.")
