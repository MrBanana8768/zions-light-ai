"""
p3-b (hostile pass #3, reviewer B) F8 — restore_backup names webuidb-sync in
its restart line, and sync_once has no placement gate.

Ported from SP\\p3-b\\gate.py: on a WEBUI_DB_LOCAL=false pod with a stale
local file left on the overlay from an earlier WEBUI_DB_LOCAL=true boot
(OPERATIONS.md's own documented rollback state), restore_backup's printed
restart command used to name `webuidb-sync` unconditionally. Starting it
ran a sync cycle that published the stale local file OVER the just-restored
live database — the generation/row-loss guards compare against the file
being REPLACED, so an older, smaller stale file passes both, and the
os.replace swaps the inode out from under the OpenWebUI connection that the
same restart command just opened.

Run inside the compactor image or any container with the requirements
installed:
    python test_p3b_webuidb_gate.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-p3b-gate-test-"))
_LOCAL = _ROOT / "var" / "webui.db"
_SNAP = _ROOT / "data" / "openwebui" / "webui.db"
_STORE = _ROOT / "data" / "openwebui" / "compactor"

os.environ["WEBUI_DB_LOCAL"] = "false"
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL)
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP)
os.environ["WEBUI_DB_QUARANTINE"] = str(_ROOT / "data" / "forensics")
os.environ["DATA_DIR"] = str(_SNAP.parent)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"
os.environ["WEBUI_DB_EMPTY_START_MARKER"] = str(_ROOT / "var" / ".empty-start")

import backup  # noqa: E402
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


def _mk(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    c = sqlite3.connect(str(p))
    c.execute("create table chat (id text primary key, chat text, updated_at integer)")
    c.executemany("insert into chat values (?,?,?)", rows)
    c.commit()
    c.close()


def _scene():
    """A WEBUI_DB_LOCAL=false pod: the live db is the snapshot path. A
    restore lands an OLDER generation there. A stale local file — left on
    the overlay by an earlier WEBUI_DB_LOCAL=true boot, per OPERATIONS.md's
    own documented rollback state — sits at WEBUI_LOCAL_DB, newer than
    both."""
    a = _ROOT / "arch"
    _mk(a / "webui.db", [("conv", "A" * 50_000, 1000)])
    (a / "compactor" / "facts").mkdir(parents=True, exist_ok=True)
    (a / "compactor" / "facts" / "x.json").write_bytes(b'{"facts": []}')
    (a / "manifest.json").write_bytes(json.dumps({
        "sources": {
            "webui.db": {"present": True},
            "compactor": {"present": True, "json_files": 1, "conversations": {}},
        }
    }).encode())
    arch = _ROOT / "zions-backup-20260101-000000.tar.gz"
    with tarfile.open(arch, "w:gz") as t:
        t.add(a, arcname=".")
    _mk(_SNAP, [("conv", "L" * 60_000, 1500)])   # live db before the restore
    _STORE.mkdir(parents=True, exist_ok=True)
    _mk(_LOCAL, [("conv", "S" * 70_000, 2000)])  # the stale overlay file
    return arch


def test_restart_line_does_not_name_webuidb_sync_on_a_false_pod():
    print("\n[test] F8: restore_backup's restart line omits webuidb-sync on a WEBUI_DB_LOCAL=false pod")
    arch = _scene()
    assert_eq(backup.live_webui_db(), _SNAP, "PRECONDITION: the live db is the snapshot path on this placement")
    rep = backup.restore_backup(arch, confirm=True)
    assert_true("webuidb-sync" not in rep["restart"],
                f"F8 fix: webuidb-sync is not in the restart line (got {rep['restart']!r})")
    for svc in ("openwebui", "compactor", "backup"):
        assert_true(svc in rep["restart"], f"and {svc} still is (got {rep['restart']!r})")


def test_control_restart_line_names_webuidb_sync_on_a_true_pod():
    print("\n[test] F8 CONTROL: on a WEBUI_DB_LOCAL=true pod, the restart line still names webuidb-sync")
    # NOTE: webuidb.LOCAL_DB / webuidb.SNAPSHOT_DB are read from the
    # environment ONCE, at import (deliberately — see live_webui_db()'s own
    # docstring: only the GATE, WEBUI_DB_LOCAL, is re-read fresh every
    # call). Pointing WEBUI_LOCAL_DB/WEBUI_SNAPSHOT_DB at a NEW path after
    # import has no effect on those constants, so this test (and the
    # sync_once CONTROL below it) reuse THIS file's own _LOCAL/_SNAP/
    # _STORE — already bound to webuidb.LOCAL_DB/SNAPSHOT_DB — and flip
    # only WEBUI_DB_LOCAL.
    os.environ["WEBUI_DB_LOCAL"] = "true"
    try:
        a = _ROOT / "arch-true"
        shutil.rmtree(a, ignore_errors=True)
        _mk(a / "webui.db", [("conv", "A" * 50_000, 1000)])
        (a / "compactor" / "facts").mkdir(parents=True, exist_ok=True)
        (a / "compactor" / "facts" / "x.json").write_bytes(b'{"facts": []}')
        (a / "manifest.json").write_bytes(json.dumps({
            "sources": {"webui.db": {"present": True},
                        "compactor": {"present": True, "json_files": 1, "conversations": {}}}
        }).encode())
        arch = _ROOT / "zions-backup-true-20260101-000000.tar.gz"
        with tarfile.open(arch, "w:gz") as t:
            t.add(a, arcname=".")
        _mk(_LOCAL, [("conv", "L" * 60_000, 1500)])
        _STORE.mkdir(parents=True, exist_ok=True)
        assert_eq(backup.live_webui_db(), _LOCAL, "PRECONDITION: the live db is the LOCAL path on this placement")
        rep = backup.restore_backup(arch, webui_db=_LOCAL, confirm=True)
        assert_true("webuidb-sync" in rep["restart"],
                    f"CONTROL: webuidb-sync IS named when it means something (got {rep['restart']!r})")
    finally:
        os.environ["WEBUI_DB_LOCAL"] = "false"


def test_sync_once_refuses_on_a_false_pod_even_if_started_by_hand():
    print("\n[test] F8: sync_once itself refuses on a WEBUI_DB_LOCAL=false pod (autostart=false is not the only guard)")
    _scene()
    con = sqlite3.connect(str(_SNAP))
    con.execute("select 1").fetchall()
    ino_before = _SNAP.stat().st_ino
    live_before = _SNAP.read_bytes()
    t = time.time() - 60
    os.utime(_SNAP, (t, t))

    r = webuidb.sync_once()
    assert_eq(r["synced"], False, f"F8 fix: sync_once refuses outright (got {r})")
    assert_true(bool(r["error"]) and "REFUSING to sync" in r["error"],
                f"and names the placement as the reason (got {r['error']!r})")
    assert_eq(_SNAP.stat().st_ino, ino_before, "the live db's inode was never touched")
    assert_eq(_SNAP.read_bytes(), live_before, "and its bytes are exactly what they were")

    # The open connection a real OpenWebUI process would hold still works —
    # this is the thing the bug broke (READONLY_DBMOVED on the next write).
    con.execute("update chat set chat = chat || 'HER-NEW-MESSAGE', updated_at = 3000")
    con.commit()
    con.close()


def test_control_sync_once_still_works_on_a_true_pod():
    print("\n[test] F8 CONTROL: sync_once still publishes normally on a WEBUI_DB_LOCAL=true pod")
    # Same note as the CONTROL above: webuidb.LOCAL_DB/SNAPSHOT_DB are
    # frozen at import, so this reuses _LOCAL/_SNAP and flips only the gate.
    os.environ["WEBUI_DB_LOCAL"] = "true"
    try:
        _mk(_LOCAL, [("conv", "L" * 60_000, 1500)])
        if _SNAP.exists():
            _SNAP.unlink()
        r = webuidb.sync_once(force=True)
        assert_eq(r["synced"], True, f"CONTROL: publishes normally on the placement this daemon is FOR (got {r})")
    finally:
        os.environ["WEBUI_DB_LOCAL"] = "false"


if __name__ == "__main__":
    tests = [
        test_restart_line_does_not_name_webuidb_sync_on_a_false_pod,
        test_control_restart_line_names_webuidb_sync_on_a_true_pod,
        test_sync_once_refuses_on_a_false_pod_even_if_started_by_hand,
        test_control_sync_once_still_works_on_a_true_pod,
    ]
    for t in tests:
        t()
    print("\nAll p3-b webuidb placement gate (F8) tests passed.")
