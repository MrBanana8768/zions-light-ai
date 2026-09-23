"""A sidecar with no database is not debris. It is a loaded gun pointed at
whatever lands at that path next.

v3.1.9, hostile pass #2. restore_on_boot copied a healthy snapshot to a local
path that had no database but DID have a hot -journal, then opened the copy for
its success log line. SQLite derives the journal path from the path it is
given, found one, and replayed it: 3,502,080 B / 400 chats became 933,888 B of
"database disk image is malformed". The function returned
restored_from_snapshot, RESTORE_EXIT_CODES mapped that to 0, and entrypoint.sh
booted OpenWebUI onto it. _set_aside, the one sidecar-aware function here,
moved the main file FIRST and stopped at the first failure — so a partial move
manufactured exactly that state, and its own error banner told the operator to
boot again.

THE FIXTURE IS THE HARD PART. A journal captured before SQLite's page cache
spills has a ZEROED magic header, and SQLite correctly ignores it — so a test
built on a naive journal passes against the broken code, for the wrong reason.
hot_journal() spills the cache inside the transaction (cache_size=1) and
asserts the magic before returning. Every orphan test below also carries a
control proving the same journal DOES destroy an unprotected copy, so none of
them can pass because the journal was never dangerous.

    python test_webuidb_orphans.py
"""

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(tempfile.mkdtemp(prefix="webuidb-orphans-"))
os.environ["WEBUI_LOCAL_DB"] = str(_ROOT / "local" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(_ROOT / "snap" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(_ROOT / "forensics")

import webuidb  # noqa: E402

MAGIC = bytes.fromhex("d9d505f920a163d7")
FAILED: list[str] = []


def check(cond, label):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)


def mk(path: Path, chats: int, blob_kb: int = 8):
    """OpenWebUI-shaped: one whole conversation per row of `chat`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("create table if not exists chat (id text primary key, updated_at int, chat text)")
    for i in range(chats):
        con.execute("insert or replace into chat values (?,?,?)", (f"c{i}", i, "x" * (blob_kb * 1024)))
    con.commit()
    con.close()


def hot_journal(path: Path, rows: int = 200) -> bytes:
    """A GENUINELY hot rollback journal — see the module docstring for why
    anything less passes against the broken code."""
    mk(path, rows, 4)
    con = sqlite3.connect(str(path))
    con.isolation_level = None
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA cache_size=1")
    con.execute("BEGIN IMMEDIATE")
    for i in range(rows):
        con.execute("update chat set chat=? where id=?", ("q" * 4096, f"c{i}"))
    jb = Path(str(path) + "-journal").read_bytes()
    con.rollback()
    con.close()
    assert jb[:8] == MAGIC, f"fixture: journal magic is {jb[:8].hex()}, not hot"
    return jb


LOCAL, SNAP, QUAR = webuidb.LOCAL_DB, webuidb.SNAPSHOT_DB, webuidb.QUARANTINE


def holds(p: Path, body: bytes) -> bool:
    """`p` exists and holds exactly `body`. A check that calls read_bytes()
    on a file the code under test moved away CRASHES instead of failing,
    and a mutation run then reports a traceback where it should report
    which property broke. Two checks here did exactly that."""
    return p.exists() and p.read_bytes() == body


def reset():
    for p in (LOCAL, SNAP):
        for suf in ("",) + webuidb.SIDECARS:
            Path(str(p) + suf).unlink(missing_ok=True)
        for q in p.parent.glob(p.name + ".restoring-*"):
            q.unlink()
    shutil.rmtree(QUAR, ignore_errors=True)
    webuidb.EMPTY_START_MARKER.unlink(missing_ok=True)
    webuidb._reload_env()


DONOR = hot_journal(_ROOT / "donor.db", rows=200)

# ---------------------------------------------------------------------------
print()
print("[0] CONTROL: the fixture journal really does destroy an unprotected copy")
# Without this every "held" below could be a journal SQLite was never going to
# apply. Replay it by hand into a fresh copy of the snapshot, the way the old
# restore_on_boot did, and watch it break.
reset()
mk(SNAP, 400)
_victim = _ROOT / "victim" / "webui.db"
_victim.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(SNAP, _victim)
Path(str(_victim) + "-journal").write_bytes(DONOR)
_ok, _detail = webuidb.integrity(_victim)
check(not _ok,
      f"a copy of the healthy snapshot with this journal beside it FAILS "
      f"quick_check on first open ({_detail!r}) — the fixture is loaded")

# ---------------------------------------------------------------------------
print()
print("[1] an orphan -journal at the destination does not reach the restored copy")
reset()
mk(SNAP, 400)
snap_rows = webuidb._has_rows(SNAP)
Path(str(LOCAL) + "-journal").parent.mkdir(parents=True, exist_ok=True)
Path(str(LOCAL) + "-journal").write_bytes(DONOR)
check(not LOCAL.exists(), "fixture: no local database, only its orphan journal")
r = webuidb.restore_on_boot()
check(r["action"] == "restored_from_snapshot", f"the restore completed (got {r['action']!r})")
ok, detail = webuidb.integrity(LOCAL)
check(ok, f"and the restored copy PASSES quick_check ({detail!r})")
check(webuidb._has_rows(LOCAL) == snap_rows,
      f"and holds every chat the snapshot does ({webuidb._has_rows(LOCAL)} of {snap_rows})")
aside = sorted(QUAR.glob("webui.db-journal.orphan-sidecar-*"))
check(len(aside) == 1 and aside[0].read_bytes() == DONOR,
      "the orphan was SET ASIDE, byte for byte, not deleted")

# ---------------------------------------------------------------------------
print()
print("[2] the same for an orphan -wal")
reset()
mk(SNAP, 400)
_wal_src = _ROOT / "waldonor.db"
for suf in ("", "-wal", "-shm"):
    Path(str(_wal_src) + suf).unlink(missing_ok=True)
mk(_wal_src, 50, 4)
_c = sqlite3.connect(str(_wal_src))
_c.execute("PRAGMA journal_mode=WAL")
_c.execute("PRAGMA wal_autocheckpoint=0")
for i in range(50):
    _c.execute("update chat set chat=? where id=?", ("w" * 4096, f"c{i}"))
_c.commit()
_wal_bytes = Path(str(_wal_src) + "-wal").read_bytes()
_c.close()
check(len(_wal_bytes) > 0, f"fixture: a non-empty -wal ({len(_wal_bytes)} B)")
# CONTROL, as [0] is for the journal. SQLite ignores a WAL whose header salts do
# not match the database, so a foreign -wal is not automatically dangerous —
# and if this one is not, [2] below proves nothing and must say so by failing
# here rather than by passing there.
_victim2 = _ROOT / "victim2" / "webui.db"
_victim2.parent.mkdir(parents=True, exist_ok=True)
for suf in ("", "-wal", "-shm"):
    Path(str(_victim2) + suf).unlink(missing_ok=True)
shutil.copy2(SNAP, _victim2)
Path(str(_victim2) + "-wal").write_bytes(_wal_bytes)
_ok2, _detail2 = webuidb.integrity(_victim2)
_rows2 = webuidb._has_rows(_victim2)
check(not _ok2 or _rows2 != 400,
      f"CONTROL: this -wal really does damage an unprotected copy of the "
      f"snapshot (quick_check={_detail2!r}, chats={_rows2})")
Path(str(LOCAL) + "-wal").write_bytes(_wal_bytes)
r = webuidb.restore_on_boot()
ok, detail = webuidb.integrity(LOCAL)
check(r["action"] == "restored_from_snapshot" and ok,
      f"restored and healthy with an orphan -wal present ({r['action']!r}, {detail!r})")
check(webuidb._has_rows(LOCAL) == 400, "and every chat survived")

# ---------------------------------------------------------------------------
print()
print("[3] the `fresh` branch sweeps orphans too")
# No snapshot, no local database, an orphan journal: OpenWebUI's own first open
# of the database it is about to create would replay it just the same.
reset()
Path(str(LOCAL) + "-journal").parent.mkdir(parents=True, exist_ok=True)
Path(str(LOCAL) + "-journal").write_bytes(DONOR)
r = webuidb.restore_on_boot()
check(r["action"] == "fresh", f"a new deployment (got {r['action']!r})")
check(not Path(str(LOCAL) + "-journal").exists(),
      "and no journal is waiting at the path OpenWebUI is about to create")

# ---------------------------------------------------------------------------
print()
print("[4] an orphan that CANNOT be set aside refuses the restore")
reset()
mk(SNAP, 400)
Path(str(LOCAL) + "-journal").parent.mkdir(parents=True, exist_ok=True)
Path(str(LOCAL) + "-journal").write_bytes(DONOR)
with patch.object(webuidb.shutil, "move", side_effect=OSError(28, "No space left on device")):
    r = webuidb.restore_on_boot()
check(r["action"] == "error" and webuidb.RESTORE_EXIT_CODES[r["action"]] != 0,
      f"it refused, non-zero ({r['action']!r})")
check(not LOCAL.exists(),
      "and NOTHING was copied to a path with a live journal beside it")

# ---------------------------------------------------------------------------
print()
print("[5] _clear_orphan_sidecars never touches the sidecars of a database that EXISTS")
# The defensive half. Inside restore_on_boot every caller has already found no
# database or set it aside, so there this check never decides anything — which
# is exactly why it has its own test instead of being a condition nobody can
# watch fire. A database's own hot journal is how it rolls back.
reset()
mk(LOCAL, 10)
Path(str(LOCAL) + "-journal").write_bytes(DONOR)
check(webuidb._clear_orphan_sidecars(LOCAL) is True, "it reports success")
check(holds(Path(str(LOCAL) + "-journal"), DONOR),
      "and the journal beside a PRESENT database is exactly where it was")
check(not QUAR.exists() or not any(QUAR.iterdir()), "and nothing went to quarantine")

# ---------------------------------------------------------------------------
print()
print("[6] a restored copy that fails quick_check WHERE IT LANDED is not a restore")
reset()
mk(SNAP, 400)
_real_integrity = webuidb.integrity


def _bad_after_copy(path):
    if Path(path) == LOCAL:
        return False, "database disk image is malformed"
    return _real_integrity(path)


with patch.object(webuidb, "integrity", _bad_after_copy):
    r = webuidb.restore_on_boot()
check(r["action"] == "restore_failed" and webuidb.RESTORE_EXIT_CODES[r["action"]] == 4,
      f"restore_failed, exit 4 — entrypoint.sh refuses the boot (got {r['action']!r})")
check(not LOCAL.exists(), "and the unverified copy is not left where OpenWebUI would open it")
check(any(QUAR.glob("webui.db.restore-unverified-*")),
      "it was set aside for inspection, not deleted")
check(webuidb._has_rows(SNAP) == 400, "and the snapshot on /data is untouched")

# ---------------------------------------------------------------------------
print()
print("[7] _set_aside is ALL OR NOTHING: a failure part-way moves everything back")
reset()
mk(LOCAL, 10)
Path(str(LOCAL) + "-journal").write_bytes(DONOR)
Path(str(LOCAL) + "-wal").write_bytes(b"wal bytes")
_real_move = shutil.move
_moves: list[str] = []


def _second_move_fails(src, dst, *a, **k):
    _moves.append(Path(src).name)
    if len(_moves) == 2:
        raise OSError(28, "No space left on device")
    return _real_move(src, dst, *a, **k)


with patch.object(webuidb.shutil, "move", _second_move_fails):
    res = webuidb._set_aside(LOCAL, "failed-quickcheck")
check(res is False, "it reports failure")
check(LOCAL.exists() and holds(Path(str(LOCAL) + "-journal"), DONOR)
      and Path(str(LOCAL) + "-wal").exists(),
      "and the database AND both sidecars are back together at the live path — "
      "nothing split between two directories")
check(not QUAR.exists() or not any(QUAR.iterdir()), "quarantine holds nothing")
check(_moves and _moves[0] != LOCAL.name,
      f"SIDECARS WENT FIRST ({_moves}) — so a SIGKILL between two moves, which "
      f"no rollback survives, strands a database without a journal rather than "
      f"a journal waiting for the next healthy copy")

# CONTROL: with nothing failing, all three move together.
reset()
mk(LOCAL, 10)
Path(str(LOCAL) + "-journal").write_bytes(DONOR)
check(webuidb._set_aside(LOCAL, "failed-quickcheck") is True, "CONTROL: a clean set-aside succeeds")
check(not LOCAL.exists() and len(list(QUAR.glob("webui.db*failed-quickcheck-*"))) == 2,
      "CONTROL: and the database and its journal both reached quarantine")

# ---------------------------------------------------------------------------
print()
print("[8] a SYMLINKED local database stays a symlink through a restore")
# os.replace does not follow a link at its destination; the copy2 this used to
# be did. A staged rename onto the link path silently replaced an operator's
# symlink with a regular file.
if os.name == "nt":
    print("  (POSIX-only mechanism; the suite runs on Linux — see scripts/run-tests.py)")
else:
    reset()
    mk(SNAP, 400)
    _real_target = _ROOT / "elsewhere" / "webui.db"
    _real_target.parent.mkdir(parents=True, exist_ok=True)
    _real_target.unlink(missing_ok=True)
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(_real_target), str(LOCAL))
    r = webuidb.restore_on_boot()
    check(r["action"] == "restored_from_snapshot", f"restored ({r['action']!r})")
    check(LOCAL.is_symlink(), "the local path is STILL a symlink")
    check(_real_target.is_file() and webuidb._has_rows(_real_target) == 400,
          "and the database landed at the link's target, whole")
    LOCAL.unlink()

# ---------------------------------------------------------------------------
print()
print("[D5] a HOT ROLLBACK JOURNAL ON THE SNAPSHOT ITSELF is rolled back "
      "LOCALLY - /data is never opened read-write")
# findings.md D5. restore_on_boot() used to call integrity(SNAPSHOT_DB)
# DIRECTLY: pragma quick_check opens the file where it sits, and opening a
# database with a hot rollback journal beside it makes SQLite finish that
# rollback right there - a WRITE, on /data, the one volume this whole
# module exists to keep off the live path (measured: 96.5s of a 98.7s cold
# boot restore was this single quick_check on /data). The fix copies the
# snapshot AND its sidecars to local staging FIRST, and only ever opens
# (and rolls back) the LOCAL copy.
#
# This is a MATCHING hot journal, not an orphan: hot_journal() below builds
# it against SNAP's own real content, so replaying it produces the correct,
# healthy, rolled-back database - not the deliberately-mismatched-donor
# corruption case in [0] above.
reset()
_snap_clean_bytes = hot_journal(SNAP, rows=200)
# hot_journal()'s own con.rollback() already completed the undo and removed
# the real journal - SNAP right now IS the correct post-recovery state.
# Capture it, then re-inject the same journal bytes to recreate "a crash
# landed before recovery ran", the state a real boot would find.
_snap_correct_state = SNAP.read_bytes()
Path(str(SNAP) + "-journal").write_bytes(_snap_clean_bytes)
_snap_before = SNAP.read_bytes()
_snap_journal_before = Path(str(SNAP) + "-journal").read_bytes()
check(not LOCAL.exists(), "fixture: no local database yet (pod-recreate path)")

r = webuidb.restore_on_boot()

check(r["action"] == "restored_from_snapshot", f"the restore completed (got {r['action']!r})")
check(
    SNAP.read_bytes() == _snap_before,
    "SNAPSHOT_DB on /data is BYTE-IDENTICAL to before the restore - it was "
    "never opened read-write, so its own hot journal was never touched "
    "there",
)
check(
    Path(str(SNAP) + "-journal").exists()
    and Path(str(SNAP) + "-journal").read_bytes() == _snap_journal_before,
    "and its -journal sidecar is STILL THERE on /data, byte-identical too "
    "- nothing rolled it back in place",
)
ok, detail = webuidb.integrity(LOCAL)
check(ok, f"the LOCAL copy passes quick_check ({detail!r}) - the rollback "
      f"happened there instead")
check(not Path(str(LOCAL) + "-journal").exists(),
      "and LOCAL's own journal is gone - SQLite's normal open-time recovery "
      "consumed it, on local disk")
_con = sqlite3.connect(str(LOCAL))
_rows = _con.execute("select chat from chat order by id").fetchall()
_con.close()
check(
    all(row[0].startswith("x") for row in _rows) and len(_rows) == 200,
    "and the recovered content is CORRECT - every row rolled back to its "
    "pre-transaction value, not the in-flight update the crash interrupted",
)
check(
    LOCAL.read_bytes() == _snap_correct_state,
    "CONTROL: byte-for-byte identical to hot_journal()'s own recovery of "
    "the same journal against the same starting content - the local "
    "rollback reached the same answer SQLite's own crash recovery would",
)

shutil.rmtree(_ROOT, ignore_errors=True)
if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll orphan-sidecar checks passed.")
