"""
p3-b (hostile pass #3, reviewer B) F4/F9 — the in-flight restore marker.

restore_backup()'s multi-step live swap (set db aside, land the new one,
integrity-check it, set the store aside, land the new store) has no single
atomic operation covering it. A SIGKILL or an EIO between any two of those
steps can leave the live db missing, the live store missing, or the two at
a MIXED generation, with nothing on disk recording that a restore was ever
in flight — the next boot starts normally onto whatever that half-finished
state happens to be. Full transactional recovery (resume or auto-undo) is
out of scope for this lane (see webuidb.py's own comment on
write_restore_marker); this proves the reduced scope that IS implemented:
a marker written before the first live move, detected by
webuidb.find_interrupted_restore(), refused by webuidb.restore_on_boot(),
removed only on full success, and the staged debris a kill leaves behind
now showing up in backup.list_pre_restore_asides() (hostile pass #3D's
real-data finding on this same gap).

Run inside the compactor image or any container with the requirements
installed (Linux — SIGKILL via subprocess):
    python test_p3b_restore_marker.py
"""

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-p3b-marker-test-"))
_DATA = _ROOT / "data" / "openwebui"
_DB = _DATA / "webui.db"
_STORE = _DATA / "compactor"
_Q = _ROOT / "data" / "forensics"
_BK = _ROOT / "data" / "backups"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["WEBUI_DB_QUARANTINE"] = str(_Q)
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"
os.environ["WEBUI_DB_LOCAL"] = "true"
os.environ["WEBUI_LOCAL_DB"] = str(_DB)
os.environ["WEBUI_SNAPSHOT_DB"] = str(_DATA / "snapshot-unused.db")

import backup  # noqa: E402
import webuidb  # noqa: E402

CID = "0123456789abcdef0123456789abcdef"


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


def _mkdb(p: Path, gen: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p))
    c.execute("create table chat (id text primary key, chat text)")
    c.execute("insert into chat values ('c', ?)", (gen * 2000,))
    c.commit()
    c.close()


def _mkstore(root: Path, gen: str) -> None:
    (root / "facts").mkdir(parents=True, exist_ok=True)
    (root / "personas").mkdir(parents=True, exist_ok=True)
    (root / "facts" / f"{CID}.json").write_bytes(
        json.dumps({"conv_id": CID, "facts": [{"text": f"{gen} {i}"} for i in range(10)]}).encode()
    )
    (root / "personas" / f"{CID}.json").write_bytes(json.dumps({"persona_text": gen}).encode())


def _build_archive() -> Path:
    a = _ROOT / "arch"
    shutil.rmtree(a, ignore_errors=True)
    a.mkdir()
    _mkdb(a / "webui.db", "NEW")
    _mkstore(a / "compactor", "NEW")
    man = {
        "schema": "v2",
        "sources": {
            "webui.db": {"present": True},
            "compactor": {
                "present": True, "json_files": 2, "chroma_sqlite": False,
                "conversations": backup._census(a / "compactor"),
            },
        },
        "payload_bytes": 1,
    }
    (a / "manifest.json").write_bytes(json.dumps(man).encode())
    _BK.mkdir(parents=True, exist_ok=True)
    arch = _BK / "zions-backup-20260101-000000.tar.gz"
    with tarfile.open(arch, "w:gz") as t:
        t.add(a, arcname=".")
    return arch


def _reset_live(gen: str = "OLD") -> Path:
    shutil.rmtree(_DATA, ignore_errors=True)
    shutil.rmtree(_Q, ignore_errors=True)
    _mkdb(_DB, gen)
    _mkstore(_STORE, gen)
    return _build_archive()


_CHILD = r'''
import os, sys, signal
sys.path.insert(0, os.getcwd())
KILL_AFTER_REPLACES = int(os.environ["KILL_AFTER_REPLACES"])
n = [0]
real_replace = os.replace
def counting_replace(src, dst):
    r = real_replace(src, dst)
    n[0] += 1
    if n[0] == KILL_AFTER_REPLACES:
        os.kill(os.getpid(), signal.SIGKILL)
    return r
os.replace = counting_replace
import backup
from pathlib import Path as P
r = backup.restore_backup(P(os.environ["ARCH"]), webui_db=P(os.environ["TARGET"]), confirm=True)
print("RESTORE_RETURNED")
'''


def _run_killed_after_n_replaces(arch: Path, n: int) -> None:
    env = dict(os.environ, KILL_AFTER_REPLACES=str(n), ARCH=str(arch), TARGET=str(_DB))
    subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=30)


def test_marker_present_after_a_kill_between_the_two_swaps():
    print("\n[test] F4/F9: a SIGKILL after the db swap lands (before the store swap) leaves the marker behind")
    arch = _reset_live()
    # The first os.replace() in restore_backup is db_tmp -> target; kill
    # right after it lands, before the store's own os.replace runs.
    _run_killed_after_n_replaces(arch, 1)

    marker = webuidb.find_interrupted_restore()
    assert_true(marker is not None, "F4/F9 fix: the marker is present after the kill")
    assert_true(marker["plan"]["archive"] == arch.name, f"and it names the right archive (got {marker})")

    boot = webuidb.restore_on_boot()
    assert_eq(boot["action"], "restore_interrupted",
              f"F4/F9 fix: restore_on_boot refuses instead of reconciling normally (got {boot})")
    assert_eq(webuidb.RESTORE_EXIT_CODES[boot["action"]] != 0, True,
              "and that action exits non-zero (entrypoint.sh would refuse to boot)")

    asides = backup.list_pre_restore_asides()
    names = [a["name"] for a in asides]
    assert_true(any(".restore-" in n or ".incoming-" in n or ".pre-restore-" in n for n in names),
                f"D-F3: the staged/set-aside debris from the kill is now listed (got {names})")


def test_no_marker_and_no_debris_after_a_clean_restore():
    print("\n[test] F4/F9 CONTROL: a clean, uninterrupted restore leaves no marker and no debris")
    arch = _reset_live()
    rep = backup.restore_backup(arch, webui_db=_DB, confirm=True)
    assert_true(rep["ok"], f"CONTROL: the restore itself succeeded (got {rep})")

    marker = webuidb.find_interrupted_restore()
    assert_true(marker is None, f"CONTROL: no marker after a clean restore (got {marker})")

    boot = webuidb.restore_on_boot()
    assert_true(boot["action"] != "restore_interrupted",
                f"CONTROL: restore_on_boot proceeds normally (got {boot['action']})")

    asides = backup.list_pre_restore_asides()
    staged_debris = [a["name"] for a in asides if ".restore-" in a["name"] or ".incoming-" in a["name"]]
    assert_eq(staged_debris, [], f"CONTROL: no staged .restore-/.incoming- debris left behind (got {staged_debris})")


def test_no_marker_for_a_kill_during_staging_before_any_live_move():
    print("\n[test] F4/F9 CONTROL: a kill during STAGING (before any live move) writes no marker")
    arch = _reset_live()
    # 0 os.replace() calls have happened by the time staging finishes —
    # this kills the process via a different hook, right after the
    # archive is opened, well before the first live-path move.
    child = r'''
import os, sys, signal, tarfile
sys.path.insert(0, os.getcwd())
real_open = tarfile.open
def killing_open(*a, **k):
    r = real_open(*a, **k)
    os.kill(os.getpid(), signal.SIGKILL)
    return r
tarfile.open = killing_open
import backup
from pathlib import Path as P
backup.restore_backup(P(os.environ["ARCH"]), webui_db=P(os.environ["TARGET"]), confirm=True)
'''
    env = dict(os.environ, ARCH=str(arch), TARGET=str(_DB))
    subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True, timeout=30)
    marker = webuidb.find_interrupted_restore()
    assert_true(marker is None,
                f"CONTROL: no marker for a kill before staging even started (got {marker}) — "
                f"nothing live was ever about to move")


if __name__ == "__main__":
    tests = [
        test_marker_present_after_a_kill_between_the_two_swaps,
        test_no_marker_and_no_debris_after_a_clean_restore,
        test_no_marker_for_a_kill_during_staging_before_any_live_move,
    ]
    for t in tests:
        t()
    print("\nAll p3-b restore-marker (F4/F9) tests passed.")
