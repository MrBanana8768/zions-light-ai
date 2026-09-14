"""
p3-b (hostile pass #3, reviewer B) F5 — restore_backup's round-2 rollback,
when the STORE swap fails AFTER the database swap already landed.

Ported from SP\\p3-b\\rollback.py (no mocked errors: a REAL hot journal, made
by killing a writer mid-transaction, and a REAL ENOTEMPTY from a recreated
store path — the two things the finding proves the old handler got wrong).

  1. The old handler restored db_aside but never the sidecars (-journal/-wal)
     it had quarantined a few lines earlier — the live db came back WITHOUT
     the journal that makes it consistent, and a live connection read
     uncommitted pages as committed.
  2. `shutil.move(store_aside, sroot)` lands `store_aside` INSIDE `sroot`
     when `sroot` already exists (a directory-move quirk, not a bug in
     shutil) — if something recreated the store path after the set-aside
     (a compactor not yet stopped, whose atomic_write_json mkdirs the
     parent), the pre-restore store silently nested one level down instead
     of landing back at the live path.

Both are proven fixed here against the SAME triggers the finding used, and
a CONTROL proves the ordinary (non-nested-obstacle) rollback path still
works.

Run inside the compactor image or any container with the requirements
installed (Linux only — sqlite3 hot-journal + POSIX kill-mid-transaction,
same as test_backup_v319.py's A3-6b fixtures):
    python test_p3b_restore_rollback.py
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="zions-p3b-rollback-test-"))
_DATA = _TMP / "data" / "openwebui"
_STORE = _DATA / "compactor"
_Q = _TMP / "data" / "forensics"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["WEBUI_DB_QUARANTINE"] = str(_Q)
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"

import backup  # noqa: E402

CID = "0123456789abcdef0123456789abcdef"
DB = _DATA / "webui.db"


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


def _mkstore(root: Path, gen: str) -> None:
    (root / "personas").mkdir(parents=True, exist_ok=True)
    (root / "facts").mkdir(parents=True, exist_ok=True)
    (root / "personas" / f"{CID}.json").write_bytes(json.dumps({"persona_text": gen}).encode())
    (root / "facts" / f"{CID}.json").write_bytes(
        json.dumps({"conv_id": CID, "facts": [{"text": gen}]}).encode()
    )


def _build_archive(root: Path) -> Path:
    a = root / "arch"
    a.mkdir()
    c = sqlite3.connect(str(a / "webui.db"))
    c.execute("create table chat (id text primary key, chat text)")
    c.execute("insert into chat values ('c','NEW')")
    c.commit()
    c.close()
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
    arch = root / "zions-backup-20260101-000000.tar.gz"
    with tarfile.open(arch, "w:gz") as t:
        t.add(a, arcname=".")
    return arch


def _build_live_db_with_real_hot_journal() -> Path:
    """4,000 committed rows, then a writer killed mid-UPDATE — a real spilled
    cache, real synced journal header (see hostile317-c hotj.py / p3-b
    hotj3.py: this is the shape that actually makes SQLite treat the
    journal as hot, unlike a one-small-INSERT kill)."""
    _DATA.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB))
    c.execute("create table chat (id text primary key, chat text)")
    c.executemany("insert into chat values (?,?)", [(f"c{i}", "x" * 900) for i in range(4000)])
    c.commit()
    c.close()
    code = (
        "import sqlite3,os\n"
        f"c=sqlite3.connect({str(DB)!r},isolation_level=None)\n"
        "c.execute('BEGIN')\n"
        "c.execute(\"UPDATE chat SET chat=replace(chat,'x','w')\")\n"
        "os._exit(9)\n"
    )
    subprocess.run([sys.executable, "-c", code])
    j = Path(str(DB) + "-journal")
    assert j.exists(), "fixture: the killed writer must leave a journal file"
    return j


def _open_live_db_counts():
    c = sqlite3.connect(str(DB))
    try:
        integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
        committed = c.execute("select count(*) from chat where chat like 'x%'").fetchone()[0]
        uncommitted = c.execute("select count(*) from chat where chat like 'w%'").fetchone()[0]
        return integrity, committed, uncommitted
    finally:
        c.close()


def test_store_swap_failure_restores_db_AND_its_journal_no_nested_store():
    print("\n[test] F5: store-swap failure after db landed rolls db+journal AND the store back, no nesting")
    j = _build_live_db_with_real_hot_journal()
    journal_before = j.read_bytes()
    assert_true(journal_before[:8] != b"\x00" * 8, "fixture: the journal header is really synced (hot)")
    _mkstore(_STORE, "OLD")
    arch = _build_archive(_TMP)

    real_aside = backup._quarantine_aside

    def aside_then_recreate(path, stamp):
        dest = real_aside(path, stamp)
        # The real trigger: a compactor process not yet stopped writes a
        # fact between the set-aside and the swap, recreating the store
        # directory (memory.atomic_write_json mkdirs the parent).
        if Path(path) == _STORE:
            (_STORE / "facts").mkdir(parents=True)
            (_STORE / "facts" / "x.json").write_bytes(
                b'{"conv_id": "x", "facts": [{"text": "written by a running compactor"}]}'
            )
        return dest

    backup._quarantine_aside = aside_then_recreate
    try:
        try:
            backup.restore_backup(arch, webui_db=DB, confirm=True)
            assert_true(False, "fixture: the recreated store must make the swap raise ENOTEMPTY")
        except OSError as e:
            assert_true("ot empty" in str(e) or "ENOTEMPTY" in str(e),
                        f"fixture: the failure is the real ENOTEMPTY (got {type(e).__name__}: {e})")
    finally:
        backup._quarantine_aside = real_aside

    # The live db is back, WITH its journal — the F5 bug's core claim.
    assert_true(DB.exists(), "the live db path is not empty after rollback")
    assert_true(j.exists(), "F5 fix: the hot journal is back beside the live db")
    assert_eq(j.read_bytes(), journal_before, "F5 fix: the journal bytes are exactly what they were")

    integrity, committed, uncommitted = _open_live_db_counts()
    assert_eq(integrity, "ok", "the rolled-back db passes integrity_check")
    assert_eq(committed, 4000, "all 4,000 originally-committed rows are there")
    assert_eq(uncommitted, 0,
              "F5 fix: NO uncommitted rows leaked in (the bug produced 2,276 — "
              "SP\\p3-b\\rollback.py)")

    # The store is back at the LIVE path, not nested one level down.
    assert_true(_STORE.exists(), "the store directory exists at the live path")
    top = sorted(p.name for p in _STORE.iterdir())
    assert_true("compactor.pre-restore-20260101-000000" not in " ".join(top),
                f"F5 fix: the old store is not nested inside itself (got {top})")
    persona_path = _STORE / "personas" / f"{CID}.json"
    assert_true(persona_path.exists(), "the OLD store's persona file is back at the live path")
    assert_eq(json.loads(persona_path.read_text())["persona_text"], "OLD",
              "F5 fix: it is genuinely the OLD generation, not the recreated stub")

    # The obstacle that WOULD have been nested is preserved, not deleted.
    failed = list(_STORE.parent.glob("compactor.failed-*"))
    assert_true(len(failed) == 1, f"the recreated store was renamed aside, not deleted (got {failed})")
    assert_true((failed[0] / "facts" / "x.json").exists(),
                "the recreated store's own content survives under the .failed- name")


def test_control_ordinary_rollback_with_no_obstacle_still_works():
    print("\n[test] F5 CONTROL: the ordinary rollback path (no recreated store) still works, no .failed- debris")
    shutil.rmtree(_TMP, ignore_errors=True)
    _TMP.mkdir(parents=True)
    _DATA.mkdir(parents=True)
    _STORE.mkdir(parents=True)
    c = sqlite3.connect(str(DB))
    c.execute("create table chat (id text primary key, chat text)")
    c.execute("insert into chat values ('c','OLD')")
    c.commit()
    c.close()
    _mkstore(_STORE, "OLD")
    arch = _build_archive(_TMP)

    # Fail the store swap a different way — no recreated directory, just an
    # ordinary injected failure — proving the CONTROL: rollback with no
    # obstacle present does not spuriously rename anything aside.
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        if str(dst) == str(_STORE) and calls["n"] == 0:
            calls["n"] += 1
            raise OSError("simulated store-swap failure, no recreation involved")
        return real_replace(src, dst)

    os.replace = flaky_replace
    try:
        try:
            backup.restore_backup(arch, webui_db=DB, confirm=True)
            assert_true(False, "fixture: the injected failure must raise")
        except OSError as e:
            assert_true("simulated" in str(e), f"fixture: got the injected error (got {e})")
    finally:
        os.replace = real_replace

    assert_true(DB.exists(), "the live db is back")
    c = sqlite3.connect(str(DB))
    try:
        integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
        row_count = c.execute("select count(*) from chat").fetchone()[0]
        row_val = c.execute("select chat from chat").fetchone()[0]
    finally:
        c.close()
    assert_eq(integrity, "ok", "CONTROL: the rolled-back db is fine")
    assert_eq(row_count, 1, "CONTROL: it is the OLD db (one row), not the NEW one")
    assert_eq(row_val, "OLD", "CONTROL: and it really is the OLD row's content")

    # No STORE obstacle existed here (nothing recreated sroot), so the store
    # rollback must not rename anything aside. The DATABASE side DOES always
    # find target occupied on this path — the db swap always lands before a
    # store failure can happen, so `target` legitimately holds the NEW db at
    # rollback time; renaming it to webui.db.failed-* (rather than silently
    # overwriting it, which the pre-fix code did) is expected and correct,
    # not debris from a missing obstacle.
    store_failed = list(_STORE.parent.glob("compactor.failed-*"))
    assert_eq(store_failed, [], "CONTROL: no store obstacle, so nothing store-side was renamed aside")
    db_failed = list(_STORE.parent.glob("webui.db.failed-*"))
    assert_true(len(db_failed) == 1,
                f"the already-landed NEW db is preserved aside, not silently overwritten (got {db_failed})")
    persona_path = _STORE / "personas" / f"{CID}.json"
    assert_eq(json.loads(persona_path.read_text())["persona_text"], "OLD",
              "CONTROL: the OLD store is back at the live path")


_G2_CHILD = r'''
import json, os, signal, sqlite3, subprocess, sys, tarfile, time
from pathlib import Path
sys.path.insert(0, os.getcwd())
import backup

CID = "0123456789abcdef0123456789abcdef"
TMP = Path(os.environ["G2_TMP"])
DATA = Path(os.environ["DATA_DIR"])
STORE = Path(os.environ["COMPACTOR_STORAGE_ROOT"])
DB = DATA / "webui.db"


def mkstore(root, gen):
    (root / "personas").mkdir(parents=True, exist_ok=True)
    (root / "facts").mkdir(parents=True, exist_ok=True)
    (root / "personas" / f"{CID}.json").write_bytes(json.dumps({"persona_text": gen}).encode())
    (root / "facts" / f"{CID}.json").write_bytes(
        json.dumps({"conv_id": CID, "facts": [{"text": gen}]}).encode())


def build_archive():
    a = TMP / "arch"
    a.mkdir()
    c = sqlite3.connect(str(a / "webui.db"))
    c.execute("create table chat (id text primary key, chat text)")
    c.execute("insert into chat values ('c','NEW')")
    c.commit(); c.close()
    mkstore(a / "compactor", "NEW")
    man = {"schema": "v2", "sources": {"webui.db": {"present": True},
           "compactor": {"present": True, "json_files": 2, "chroma_sqlite": False,
                          "conversations": backup._census(a / "compactor")}}, "payload_bytes": 1}
    (a / "manifest.json").write_bytes(json.dumps(man).encode())
    arch = TMP / "zions-backup-20260101-000000.tar.gz"
    with tarfile.open(arch, "w:gz") as t:
        t.add(a, arcname=".")
    return arch


DATA.mkdir(parents=True, exist_ok=True)
c = sqlite3.connect(str(DB))
c.execute("create table chat (id text primary key, chat text)")
c.executemany("insert into chat values (?,?)", [(f"c{i}", "x" * 900) for i in range(4000)])
c.commit(); c.close()
code = ("import sqlite3,os\n"
        f"c=sqlite3.connect({str(DB)!r},isolation_level=None)\n"
        "c.execute('BEGIN')\n"
        "c.execute(\"UPDATE chat SET chat=replace(chat,'x','w')\")\n"
        "os._exit(9)\n")
subprocess.run([sys.executable, "-c", code])
j = Path(str(DB) + "-journal")
assert j.exists(), "fixture: killed writer must leave a journal"

mkstore(STORE, "OLD")
arch = build_archive()

real_aside = backup._quarantine_aside
def aside_then_recreate(path, stamp):
    dest = real_aside(path, stamp)
    if Path(path) == STORE:
        (STORE / "facts").mkdir(parents=True)
        (STORE / "facts" / "x.json").write_bytes(b'{"facts": []}')
    return dest
backup._quarantine_aside = aside_then_recreate

real_mb = backup._move_back_no_nest
def mb(src, dst, *, stamp, label):
    # p4-b G2's kill point: right when the FIRST sidecar move-back is about
    # to run. Under the FIXED order this only happens AFTER the db's own
    # move-back (label == "webui.db") already completed — so this SIGKILL
    # proves the state right in between: db back, its journal still in
    # quarantine, nothing foreign beside the live db.
    if "(database sidecar)" in label:
        os.kill(os.getpid(), signal.SIGKILL)
    return real_mb(src, dst, stamp=stamp, label=label)
backup._move_back_no_nest = mb

backup.restore_backup(arch, webui_db=DB, confirm=True)
print("UNEXPECTED: restore_backup returned without being killed")
'''


def test_g2_kill_between_db_and_sidecar_restore_leaves_a_recoverable_not_corrupt_state():
    print("\n[test] p4-b G2: SIGKILL between the db and sidecar move-back in the "
          "store-failure rollback — db lands WITHOUT a foreign journal beside it")
    shutil.rmtree(_TMP, ignore_errors=True)
    _TMP.mkdir(parents=True)
    env = dict(
        os.environ, G2_TMP=str(_TMP), DATA_DIR=str(_DATA),
        COMPACTOR_STORAGE_ROOT=str(_STORE), WEBUI_DB_QUARANTINE=str(_Q),
        COMPACTOR_BACKUP_MIN_FREE_MB="1",
    )
    r = subprocess.run([sys.executable, "-c", _G2_CHILD], env=env,
                        cwd=os.getcwd(), capture_output=True, text=True, timeout=60)
    assert_true(r.returncode != 0, f"fixture: the child was SIGKILLed (got rc={r.returncode}, "
                f"stdout={r.stdout!r}, stderr={r.stderr[-500:]!r})")
    assert_true("UNEXPECTED" not in r.stdout, "fixture: killed before restore_backup returned")

    j = Path(str(DB) + "-journal")
    # THE CORE G2 PROPERTY: the live db is the OLD one, and there is NO
    # journal beside it right now (the state a kill in this exact window
    # leaves) — never a MISMATCHED pair (a journal beside a db it does not
    # belong to), which is what the pre-fix order could produce here.
    assert_true(DB.exists(), "the live db path is populated after the kill")
    assert_true(not j.exists(),
                "G2 fix: no journal sits beside the live db yet (it is still in "
                "quarantine) — never a FOREIGN journal beside a mismatched db")

    c = sqlite3.connect(str(DB))
    try:
        row_val = c.execute("select chat from chat limit 1").fetchone()[0]
    finally:
        c.close()
    assert_true(row_val in ("x" * 900, "w" * 900),
                f"G2 fix: the live db is genuinely the OLD generation's bytes, not "
                f"the NEW archive's db and not corrupted garbage (got {row_val[:20]!r}...)")

    # The OLD journal is exactly where the rollback left it: still in
    # quarantine, recoverable by hand (move it back beside webui.db and
    # SQLite rolls it back correctly) — never silently lost.
    quarantined_journals = list(_Q.glob("webui.db-journal.pre-restore-*"))
    assert_true(len(quarantined_journals) == 1,
                f"G2 fix: the OLD journal is preserved in quarantine, recoverable by "
                f"hand (got {quarantined_journals})")

    # Reuniting the OLD db with its OWN journal by hand (the documented
    # recovery) must roll back cleanly to the fully-committed state — proof
    # that the kill window left something RECOVERABLE, not corrupted.
    shutil.copy2(quarantined_journals[0], j)
    c = sqlite3.connect(str(DB))
    try:
        integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
        committed = c.execute("select count(*) from chat where chat like 'x%'").fetchone()[0]
    finally:
        c.close()
    assert_eq(integrity, "ok", "G2 fix: reuniting the OLD db with ITS OWN journal by "
              "hand rolls back cleanly (no page-level corruption occurred)")
    assert_eq(committed, 4000, "G2 fix: and every originally-committed row is intact")


if __name__ == "__main__":
    tests = [
        test_store_swap_failure_restores_db_AND_its_journal_no_nested_store,
        test_control_ordinary_rollback_with_no_obstacle_still_works,
        test_g2_kill_between_db_and_sidecar_restore_leaves_a_recoverable_not_corrupt_state,
    ]
    for t in tests:
        t()
    print("\nAll p3-b restore-rollback (F5) + p4-b G2 tests passed.")
