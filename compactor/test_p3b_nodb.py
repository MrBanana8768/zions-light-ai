"""
p3-b (hostile pass #3, reviewer B) F12 — a cycle that cannot find webui.db
published an archive with no chat history, passed the payload guard, and
pruned behind it.

v3.1 F2's doctrine ("an archive that holds nothing is a failure") was
applied to the compactor store only. With the store the smaller half of
the real payload (manifests in SP logs: ~138 MB db vs ~436 MB total),
dropping the db alone leaves payload_ratio comfortably above
MIN_PAYLOAD_RATIO, and the census has no idea webui.db exists at all.

Ported from SP\\p3-b\\nodb.py.

Run inside the compactor image or any container with the requirements
installed:
    python test_p3b_nodb.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-p3b-nodb-test-"))
_DATA = _ROOT / "openwebui"
_STORE = _DATA / "compactor"
_BK = _ROOT / "backups"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BK)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DATA / "webui.db")
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"

import backup  # noqa: E402


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


def _build():
    shutil.rmtree(_DATA, ignore_errors=True)
    _DATA.mkdir(parents=True)
    c = sqlite3.connect(str(_DATA / "webui.db"))
    c.execute("create table chat (id text primary key, chat text)")
    c.execute("insert into chat values ('conv', ?)", ("m" * 3_000_000,))
    c.commit()
    c.close()
    (_STORE / "facts").mkdir(parents=True)
    (_STORE / "facts" / "a.json").write_bytes(json.dumps({"facts": [{"text": "f"}]}).encode())
    (_STORE / "chromadb").mkdir()
    with open(_STORE / "chromadb" / "index.bin", "wb") as f:
        f.write(os.urandom(6_500_000))


def test_missing_webui_db_refuses_the_cycle_and_holds_the_prune():
    print("\n[test] F12: a cycle that cannot find webui.db refuses, and does not prune")
    _build()
    time.sleep(1.1)
    r = backup.run_once(_BK)
    assert_true(r["ok"], f"PRECONDITION: baseline cycle with the db present is ok (got {r})")

    # three archives older than every retention tier, so a clean cycle
    # could prune them if it got that far
    for k in range(4):
        p = _BK / f"zions-backup-2026050{k}-000000.tar.gz"
        shutil.copy2(backup.list_backups(_BK)[0]["path"], p)
        t = time.time() - (100 + k) * 86400
        os.utime(p, (t, t))
    before_count = len(backup.list_backups(_BK))

    (_DATA / "webui.db").rename(_ROOT / "db-elsewhere")  # unresolvable at cycle time
    time.sleep(1.1)
    r = backup.run_once(_BK)

    assert_eq(r["ok"], False, f"F12 fix: the cycle is NOT ok (got {r})")
    assert_true(bool(r.get("detail")) and "webui.db" in r["detail"],
                f"and the detail names webui.db (got {r.get('detail')!r})")
    assert_eq(len(backup.list_backups(_BK)), before_count,
              "F12 fix: nothing was pruned behind the refused cycle")
    # No NEW archive should have been published with no chat history in it.
    newest = backup.read_manifest(Path(backup.list_backups(_BK)[0]["path"]))
    assert_true(newest["sources"]["webui.db"]["present"] is True,
                f"F12 fix: the newest published archive still has a real webui.db (got {newest['sources']['webui.db']})")


def test_control_the_escape_hatch_still_publishes_when_set():
    print("\n[test] F12 CONTROL: COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB=1 still allows a memory-only archive")
    _build()
    time.sleep(1.1)
    backup.run_once(_BK)
    (_DATA / "webui.db").rename(_ROOT / "db-elsewhere-2")
    os.environ["COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB"] = "1"
    try:
        time.sleep(1.1)
        r = backup.run_once(_BK)
        assert_true(r["ok"], f"CONTROL: the escape hatch publishes anyway (got {r})")
        newest = backup.read_manifest(Path(backup.list_backups(_BK)[0]["path"]))
        assert_eq(newest["sources"]["webui.db"]["present"], False,
                  f"CONTROL: and the manifest honestly says webui.db is absent (got {newest['sources']['webui.db']})")
    finally:
        os.environ.pop("COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB", None)


def test_control_a_healthy_cycle_with_the_db_present_still_prunes():
    print("\n[test] F12 CONTROL: an ordinary cycle with webui.db present still prunes normally")
    _build()
    time.sleep(1.1)
    backup.run_once(_BK)
    for k in range(4):
        p = _BK / f"zions-backup-2026060{k}-000000.tar.gz"
        shutil.copy2(backup.list_backups(_BK)[0]["path"], p)
        t = time.time() - (200 + k) * 86400
        os.utime(p, (t, t))
    time.sleep(1.1)
    r = backup.run_once(_BK)
    assert_true(r["ok"], f"CONTROL: ok (got {r})")
    assert_true(len(r.get("pruned") or []) > 0, f"CONTROL: the old archives really were pruned (got {r.get('pruned')})")


if __name__ == "__main__":
    tests = [
        test_missing_webui_db_refuses_the_cycle_and_holds_the_prune,
        test_control_the_escape_hatch_still_publishes_when_set,
        test_control_a_healthy_cycle_with_the_db_present_still_prunes,
    ]
    for t in tests:
        t()
    print("\nAll p3-b no-webui.db (F12) tests passed.")
