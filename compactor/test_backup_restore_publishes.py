"""v3.1.9.10 fix round A, R3 H2: a local-mode restore publishes the snapshot.

On WEBUI_DB_LOCAL=true, `backup.py --restore` landed the archive in the LOCAL
database only. The snapshot on /data kept the pre-restore history, the
daemon's guards (mtime skip, row-loss, generation, shrink: a restore is an
older, smaller DB by construction) never published the restore, and the next
boot could bring the old data back. Now restore_backup publishes in the same
call (forced, guards opened for that call only) and REFUSES TO FINISH (a
clear error naming the recovery command) when the snapshot does not end up
equal to the restored DB.

    python test_backup_restore_publishes.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-restpub-test-"))
_LOCAL = _ROOT / "var" / "webui.db"
_SNAP = _ROOT / "data" / "openwebui" / "webui.db"
_STORE = _ROOT / "data" / "openwebui" / "compactor"
os.environ["WEBUI_DB_LOCAL"] = "true"
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL)
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP)
os.environ["WEBUI_DB_QUARANTINE"] = str(_ROOT / "data" / "forensics")
os.environ["DATA_DIR"] = str(_SNAP.parent)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"
os.environ["WEBUI_DB_EMPTY_START_MARKER"] = str(_ROOT / "var" / ".empty-start")
for _k in list(os.environ):
    if _k.startswith("WEBUI_DB_ALLOW_") and _k != "WEBUI_DB_ALLOW_EMPTY_START":
        os.environ.pop(_k)

import backup  # noqa: E402
import webuidb  # noqa: E402

FAILED = []


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


def _mk(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    for suf in ("",) + webuidb.SIDECARS:
        f = p.with_name(p.name + suf)
        if f.exists():
            f.unlink()
    c = sqlite3.connect(str(p))
    c.execute(OWUI_CHAT)
    c.execute("create table user (id text)")
    for cid, upd, size in rows:
        body = json.dumps({"history": {"messages": {"m": {"content": "x" * size}}}})
        c.execute("insert into chat (id,user_id,title,chat,created_at,updated_at,archived,pinned,meta)"
                  " values (?, 'u1','t',?,1,?,0,0,'{}')", (cid, body, upd))
    c.commit()
    c.close()


def _archive(rows, name):
    a = _ROOT / ("arch-" + name)
    shutil.rmtree(a, ignore_errors=True)
    _mk(a / "webui.db", rows)
    (a / "compactor" / "facts").mkdir(parents=True, exist_ok=True)
    (a / "compactor" / "facts" / "x.json").write_bytes(b'{"facts": []}')
    (a / "manifest.json").write_bytes(json.dumps({"sources": {
        "webui.db": {"present": True},
        "compactor": {"present": True, "json_files": 1, "conversations": {}}}}).encode())
    arch = _ROOT / f"zions-backup-20260101-00000{len(name)}.tar.gz"
    with tarfile.open(arch, "w:gz") as t:
        t.add(a, arcname=".")
    return arch


def _scene():
    """Live LOCAL db and the snapshot are BIG and NEWER; the archive is
    small and OLD: every guard a restore trips."""
    lp = webuidb._sync_lock_path()
    if lp.exists():
        lp.unlink()
    big = [("c1", 5000, 200_000), ("c2", 5001, 200_000)]
    _mk(_LOCAL, big)
    _mk(_SNAP, big)
    _STORE.mkdir(parents=True, exist_ok=True)
    return _archive([("c1", 1000, 2_000)], "old")


def fp(p):
    return webuidb.content_fingerprint(p)


print("[1] CONTROL: the daemon-style publish of this restored state IS refused (why H2 existed)")
arch = _scene()
saved_local = _LOCAL.read_bytes()
# Simulate what the restore lands, then ask the plain forced sync_once.
import shutil as _sh
_sh.copy2(_ROOT / "arch-old" / "webui.db", _LOCAL)
r = webuidb.sync_once(force=True)
check(r["synced"] is False and r["error"], f"forced sync_once alone refuses the small/old DB (r={r})")

print("[2] restore in local mode: the snapshot equals the restored DB afterwards")
arch = _scene()
flags_before = (webuidb.ALLOW_SHRINK, webuidb.ALLOW_ROW_LOSS, webuidb.ALLOW_OLDER_GENERATION)
rep = backup.restore_backup(arch, confirm=True)
check(rep["ok"] and rep.get("snapshot_published") is True, f"report says the snapshot was published ({rep})")
check(fp(_LOCAL) is not None and fp(_LOCAL) == fp(_SNAP),
      f"snapshot fingerprint == restored local fingerprint ({fp(_LOCAL)} vs {fp(_SNAP)})")
check(fp(_SNAP)[0] == 1, "and it is the ARCHIVE's 1 chat, not the pre-restore 2")
check((webuidb.ALLOW_SHRINK, webuidb.ALLOW_ROW_LOSS, webuidb.ALLOW_OLDER_GENERATION) == flags_before,
      "the ALLOW flags were scoped to that one call: restored afterwards")
check(webuidb.find_interrupted_restore() is None, "no restore marker left behind")

print("[3] the publish fails: the restore refuses to finish, with a clear error")
arch = _scene()
_orig = webuidb.sync_once
webuidb.sync_once = lambda force=False, lock_wait_s=0.0: {
    "synced": False, "skipped": None, "error": "REFUSING to publish: simulated", "bytes": 0}
try:
    try:
        backup.restore_backup(arch, confirm=True)
        raised = None
    except RuntimeError as e:
        raised = str(e)
finally:
    webuidb.sync_once = _orig
check(raised is not None and "FAILED" in raised and "simulated" in raised
      and "--sync-once" in raised and "PRE-restore" in raised,
      f"RuntimeError names the failure and the recovery command ({(raised or '')[:160]!r}...)")
check(fp(_LOCAL)[0] == 1 and fp(_SNAP)[0] == 2,
      "and the state is honestly as reported: local restored, snapshot still old")
check(webuidb.find_interrupted_restore() is None, "the marker is cleared (the live move did land)")
check((webuidb.ALLOW_SHRINK, webuidb.ALLOW_ROW_LOSS, webuidb.ALLOW_OLDER_GENERATION) == flags_before,
      "flags restored after a failed publish too")

print("[4] 'synced' but the fingerprints differ is also a refusal (never trust the flag alone)")
arch = _scene()
webuidb.sync_once = lambda force=False, lock_wait_s=0.0: {
    "synced": True, "skipped": None, "error": None, "bytes": 1}   # lies: published nothing
try:
    try:
        backup.restore_backup(arch, confirm=True)
        raised = None
    except RuntimeError as e:
        raised = str(e)
finally:
    webuidb.sync_once = _orig
check(raised is not None and "does not match" in raised, f"refused ({(raised or '')[:120]!r})")

print("[5] WEBUI_DB_LOCAL=false: the restore target IS the snapshot; nothing to publish, no sync_once call")
os.environ["WEBUI_DB_LOCAL"] = "false"
calls = []
webuidb.sync_once = lambda *a, **k: calls.append(1) or {"synced": True, "error": None, "skipped": None}
try:
    _mk(_SNAP, [("c1", 5000, 200_000)])
    arch = _archive([("c1", 1000, 2_000)], "old2")
    rep = backup.restore_backup(arch, confirm=True)
finally:
    webuidb.sync_once = _orig
    os.environ["WEBUI_DB_LOCAL"] = "true"
check(rep["ok"] and not calls and rep["snapshot_published"] is False,
      f"no publish attempted on a false pod (calls={len(calls)}, rep={rep})")

print()
shutil.rmtree(_ROOT, ignore_errors=True)
if FAILED:
    print("!" * 60)
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All restore-publishes-snapshot (R3 H2) checks passed.")
