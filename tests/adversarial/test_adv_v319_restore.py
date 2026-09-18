"""Adversarial suite: BACKUP / RESTORE / THE STORE'S FILE IO (hostile pass #2).

Target: commit 0123135 "fix: restore the sidecars, the target, and the
atomicity", and the M10 backup half it explicitly did NOT close.

This file is a PLAIN SCRIPT, not pytest: the two suites it attacks
(compactor/test_backup.py, compactor/test_store_failures.py) are plain
scripts too, and restore_backup needs module-level config frozen at import,
which a pytest fixture cannot rewind. Run it with the unit-suite interpreter:

    /opt/compactor-venv/bin/python tests/adversarial/test_adv_v319_restore.py

Every check below asserts the behaviour OBSERVED TODAY. Where that behaviour
is the defect, the check is labelled FINDING and the docstring says what a
fix would change (which would flip the assertion). Where it is correct, the
check is labelled SOUND and must stay that way.
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "compactor"))

_TMP = Path(tempfile.mkdtemp(prefix="adv-v319-restore-"))
# Two volumes, named the way the deployment names them.
_LOCAL = _TMP / "var" / "lib" / "openwebui"          # overlay, dies with pod
_DATA = _TMP / "data" / "openwebui"                   # MooseFS, durable
_STORE = _DATA / "compactor"
_BACKUPS = _TMP / "data" / "backups"
_LOCAL_DB = _LOCAL / "webui.db"
_SNAP_DB = _DATA / "webui.db"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BACKUPS)
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL_DB)
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_DB)
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"
# DELIBERATELY NOT SET: COMPACTOR_BACKUP_WEBUI_DB. compactor/test_backup.py
# pins it, which is why that suite cannot see either half of M10 — with it
# set, WEBUI_DB and DATA_DIR/webui.db are the same file and the two branches
# of the new target resolution are indistinguishable.
os.environ.pop("COMPACTOR_BACKUP_WEBUI_DB", None)

_LOCAL.mkdir(parents=True, exist_ok=True)
_DATA.mkdir(parents=True, exist_ok=True)

import importlib  # noqa: E402

import backup  # noqa: E402
import webuidb  # noqa: E402

_PASS = 0
_FAIL = 0
_NOTES: list[str] = []

# The real rollback-journal magic, from the SQLite file-format spec. Nothing
# here may pass by treating a journal as an ordinary file.
_HOT = bytes.fromhex("d9d505f920a163d7")


def check(cond, label):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ok    {label}")
    else:
        _FAIL += 1
        print(f"  FAIL  {label}")


def note(msg):
    _NOTES.append(msg)
    print(f"  note  {msg}")


def head(title):
    print("")
    print(f"[{title}]")


def _make_db(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, chat TEXT)")
    con.execute("INSERT INTO chat (chat) VALUES (?)", (body,))
    con.commit()
    con.close()


def _read_chat(path: Path) -> str | None:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return con.execute("SELECT chat FROM chat LIMIT 1").fetchone()[0]
        finally:
            con.close()
    except Exception as e:
        return f"<unreadable: {type(e).__name__}: {e}>"


def _seed_store(text="seed fact"):
    if _STORE.exists():
        shutil.rmtree(_STORE)
    (_STORE / "facts").mkdir(parents=True, exist_ok=True)
    (_STORE / "summaries").mkdir(parents=True, exist_ok=True)
    (_STORE / "facts" / "conv1.json").write_text(
        json.dumps({"conv_id": "conv1", "facts": [{"text": text}]}),
        encoding="utf-8",
    )
    (_STORE / "summaries" / "conv1.json").write_text(
        json.dumps({"conv_id": "conv1", "l1": [], "l2": [], "l3": None}),
        encoding="utf-8",
    )


def _clean_backups():
    if _BACKUPS.exists():
        shutil.rmtree(_BACKUPS)


def _archive(db_body="live rows", fact="seed fact") -> Path:
    """One verified archive of the CURRENT live db + store."""
    _seed_store(fact)
    _make_db(backup.WEBUI_DB, db_body)
    _clean_backups()
    rep = backup.run_once()
    if not rep.get("ok"):
        raise SystemExit(f"fixture failed: run_once said {rep}")
    return _BACKUPS / rep["archive"]


# ===========================================================================
# R1. The sidecar rename happens BEFORE the copy that can fail.
# ===========================================================================

def r1_sidecars_stripped_before_the_copy_can_fail():
    """FINDING. A restore that fails leaves the ORIGINAL database stripped
    of its hot rollback journal.

    restore_backup renames -journal/-wal/-shm aside FIRST, then copies to a
    temp, then os.replace()s. If the copy fails — ENOSPC on the same volume
    the archive is on, an EIO from MooseFS, or the container being killed —
    the function raises, the CLI prints 'restore failed: ...', the target is
    byte-for-byte what it was... and its hot journal is gone.

    A hot journal is not debris. It is the ONLY thing that can roll back the
    half-applied transaction already written into the database pages. Move it
    away and the database that was recoverable is now silently inconsistent:
    SQLite opens it without complaint, because to SQLite there is nothing to
    replay.

    A fix reorders: copy to the temp first, rename the sidecars only once the
    temp is complete, then replace. That is one line moved.
    """
    head("R1 FINDING: a failed restore strips the original's hot journal")
    arch = _archive(db_body="the rows the operator still has")
    target = backup.WEBUI_DB
    journal = target.with_name(target.name + "-journal")
    journal.write_bytes(_HOT + b"the only copy of the last transaction")
    before = target.read_bytes()

    real_copy2 = shutil.copy2

    def boom(src, dst, *a, **kw):
        if ".restore-" in str(dst):
            raise OSError(28, "No space left on device")
        return real_copy2(src, dst, *a, **kw)

    shutil.copy2 = boom
    raised = None
    try:
        backup.restore_backup(arch, confirm=True)
    except Exception as e:  # noqa: BLE001
        raised = e
    finally:
        shutil.copy2 = real_copy2

    check(raised is not None,
          "the restore failed and said so (%s)" % type(raised).__name__)
    check(target.read_bytes() == before,
          "the live database is byte-for-byte untouched — 'nothing happened'")
    check(not journal.exists(),
          "FINDING: ...except its hot journal, which is GONE")
    aside = sorted(target.parent.glob(target.name + "-journal.pre-restore-*"))
    check(len(aside) == 1,
          "it is renamed beside the database, not deleted (found %d)" % len(aside))
    note("the operator now has a database SQLite will open happily whose "
         "last transaction is half-applied with no journal to roll it back; "
         "nothing in the failure message mentions the journal")
    # And the store half never ran, so the report is silent about it too.
    check(_STORE.is_dir(), "the compactor store was not reached")
    if aside:
        aside[0].unlink()


# ===========================================================================
# R2. The atomicity fix stopped at its sibling ten lines below.
# ===========================================================================

def r2_store_half_is_rmtree_then_copytree():
    """FINDING. The commit says 'IT WAS NOT ATOMIC ... It writes a sibling
    temp and os.replace()s now'. That is true of webui.db and false of the
    compactor store, which is the half that cannot be regenerated.

        if sroot.exists():
            shutil.rmtree(sroot)
        shutil.copytree(src_store, sroot)

    rmtree FIRST. There is no temp, no replace, and no set-aside — the exact
    opposite of the treatment the sidecars got twenty lines above, in the same
    function, in the same commit. Every fact, every summary tier, every
    persona and ChromaDB itself are deleted before the first byte is written
    back, and anything that interrupts the copy has destroyed them.

    webui.db at least has a second copy (the snapshot, or the local file) and
    a 41 MB archive behind it. The store's only other copy IS the archive.

    Known as REMEDIATION.md F6; reported again because 0123135 is the commit
    that claimed to fix the atomicity of this function and walked past it.
    """
    head("R2 FINDING: the store half is rmtree-then-copytree, no set-aside")
    arch = _archive(db_body="rows", fact="a fact that only exists here")
    check((_STORE / "facts" / "conv1.json").is_file(),
          "the live store is populated before the restore")

    real_copytree = shutil.copytree

    def boom(src, dst, *a, **kw):
        raise OSError(28, "No space left on device")

    shutil.copytree = boom
    raised = None
    try:
        backup.restore_backup(arch, confirm=True)
    except Exception as e:  # noqa: BLE001
        raised = e
    finally:
        shutil.copytree = real_copytree

    check(raised is not None,
          "the restore failed and said so (%s)" % type(raised).__name__)
    check(not _STORE.exists(),
          "FINDING: the ENTIRE compactor store is gone — rmtree ran, the "
          "copy did not")
    strays = sorted(p.name for p in _DATA.iterdir()
                    if "pre-restore" in p.name or "incoming" in p.name
                    or ".restore-" in p.name)
    check(strays == [],
          "and nothing was set aside anywhere: %r" % (strays,))
    note("the webui.db half of this same function renames its sidecars aside "
         "'RENAMED, NEVER DELETED' and replaces atomically; the store half "
         "deletes outright, ten lines below, in the same commit")
    _seed_store()


# ===========================================================================
# R3. The .pre-restore stamp is second-resolution.
# ===========================================================================

def r3_stamp_resolution_regresses_a_lesson_webuidb_already_learned():
    """FINDING. `time.strftime("%Y%m%d-%H%M%S", time.gmtime())` — whole
    seconds. webuidb._stamp() exists, twelve files away, at MILLISECOND
    resolution, and its comment says why:

        # Millisecond precision, not seconds: two set-asides inside the same
        # second collided on the filename and the second silently overwrote
        # the first. These files exist because something already went wrong;
        # losing one to a name clash is exactly the wrong time for that.

    backup.py duplicated webuidb.SIDECARS and its reasoning and did not
    duplicate its stamp. On POSIX Path.rename() over an existing name is a
    silent replace, so the collision loses the FIRST journal — the older one,
    the one more likely to hold the transaction the operator is chasing.

    The fix is `stamp = webuidb._stamp()`-shaped: append milliseconds.
    """
    head("R3 FINDING: second-resolution set-aside stamp, silently clobbering")
    arch = _archive(db_body="rows")
    target = backup.WEBUI_DB
    journal = target.with_name(target.name + "-journal")

    # SOUND first: the two stamps side by side, no patching.
    s_backup = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    s_webuidb = webuidb._stamp()
    note(f"backup.py stamp  = {s_backup!r} (17 chars, 1 s resolution)")
    note(f"webuidb._stamp() = {s_webuidb!r} (21 chars, 1 ms resolution)")
    check(len(s_webuidb) > len(s_backup),
          "webuidb's own stamp is strictly finer than the one backup.py wrote")

    # Two restores inside one stamp tick. strftime at second resolution
    # returns the same string for any two calls in the same second, so
    # pinning it IS the same-second case, exactly.
    real_strftime = time.strftime
    time.strftime = lambda fmt, *a: ("FROZEN" if "%H%M%S" in fmt
                                     else real_strftime(fmt, *a))
    try:
        journal.write_bytes(_HOT + b"FIRST journal - the older transaction")
        backup.restore_backup(arch, confirm=True)
        journal.write_bytes(_HOT + b"SECOND journal")
        backup.restore_backup(arch, confirm=True)
    finally:
        time.strftime = real_strftime

    aside = sorted(target.parent.glob(target.name + "-journal.pre-restore-*"))
    check(len(aside) == 1,
          "FINDING: two set-asides, ONE file on disk (found %d)" % len(aside))
    if aside:
        body = aside[0].read_bytes()
        check(b"SECOND" in body,
              "FINDING: the surviving file is the SECOND journal")
        check(b"FIRST" not in body,
              "FINDING: the FIRST journal is gone, silently, no log line")
        for p in aside:
            p.unlink()

    # And the real, unpatched timing: back-to-back restores land in one second
    # on any machine this runs on.
    for p in target.parent.glob(target.name + "-journal.pre-restore-*"):
        p.unlink()
    journal.write_bytes(_HOT + b"unpatched FIRST")
    backup.restore_backup(arch, confirm=True)
    journal.write_bytes(_HOT + b"unpatched SECOND")
    backup.restore_backup(arch, confirm=True)
    real_aside = sorted(target.parent.glob(target.name + "-journal.pre-restore-*"))
    note("unpatched back-to-back restores left %d set-aside file(s)"
         % len(real_aside))
    for p in real_aside:
        p.unlink()


# ===========================================================================
# R4. Where the set-asides land.
# ===========================================================================

def r4_set_asides_land_on_the_volume_that_dies_with_the_pod():
    """FINDING (MEDIUM). 'RENAMED, NEVER DELETED: a hot journal may hold the
    only copy of everything written since the last commit.' True for as long
    as the filesystem it is on survives.

    The aside is `side.with_name(...)` — a sibling of the TARGET. Under the
    shipped default (WEBUI_DB_LOCAL=true) the target is
    /var/lib/openwebui/webui.db on the container overlay, which a pod
    recreate destroys. webuidb._set_aside, the sibling this code says it is
    copying, moves its set-asides to QUARANTINE (/data/forensics) precisely
    so they outlive the incident.

    A restore is run during an incident, and what follows an incident on
    RunPod is a redeploy. The preserved journal is preserved until then.
    """
    head("R4 FINDING: set-asides land next to the target, not in QUARANTINE")
    arch = _archive(db_body="rows")
    target = backup.WEBUI_DB
    journal = target.with_name(target.name + "-journal")
    journal.write_bytes(_HOT + b"hot")
    backup.restore_backup(arch, confirm=True)
    aside = sorted(target.parent.glob(target.name + "-journal.pre-restore-*"))
    check(len(aside) == 1, "one set-aside was made")
    if aside:
        check(aside[0].parent == target.parent,
              "FINDING: it is a sibling of the target (%s)" % aside[0].parent)
        check(aside[0].parent != webuidb.QUARANTINE,
              "FINDING: and NOT in webuidb.QUARANTINE (%s), where the sibling "
              "this code copies puts them" % webuidb.QUARANTINE)
        note("the aside is always a sibling of the TARGET, so under the "
             "shipped gate (WEBUI_DB_LOCAL=true) it lands in "
             "/var/lib/openwebui — the container overlay, which a pod "
             "recreate destroys. R5 below shows the same resolution being "
             "used for the target, so this is not hypothetical.")
        aside[0].unlink()

    # SOUND: nothing in this tree will mistake the renamed form for a journal
    # again. SQLite derives sidecar names by exact suffix, and every sweep in
    # the repo tests the exact suffix too.
    for suffix in backup.SIDECARS:
        check(not (target.name + suffix).endswith("pre-restore"),
              f"SOUND: {suffix!r} is an exact suffix, so "
              f"'webui.db-journal.pre-restore-*' can never be re-read as a "
              f"journal")


# ===========================================================================
# R5. M10, the backup half — and which branch every caller takes.
# ===========================================================================

def r5_backup_and_restore_both_follow_exists_not_the_gate():
    """FINDING (BLOCKER-adjacent). backup.py picks the live database by
    `_LIVE_DB.exists()`. The deployment picks it by WEBUI_DB_LOCAL, which
    backup.py never reads — the string does not appear in the file.

        WEBUI_DB = Path(
            os.environ.get("COMPACTOR_BACKUP_WEBUI_DB")
            or (str(_LIVE_DB) if _LIVE_DB.exists() else str(DATA_DIR / "webui.db"))
        )

    With WEBUI_DB_LOCAL=false — the documented ROLLBACK for the one step in
    this series whose rollback is not clean — entrypoint.sh points
    DATABASE_URL at the SNAPSHOT and leaves the local file where it is. A
    stale /var/lib/openwebui/webui.db from the previous boot then still
    exists(), so:

      * create_backup archives the ABANDONED file. Every nightly tarball is a
        copy of a database nobody has written to since the flag flipped.
      * restore_backup, called from the CLI with neither webui_db nor
        data_dir, resolves the same WEBUI_DB and writes the archive back onto
        the abandoned file. OpenWebUI never sees it.

    0123135 says it 'closes the restore half of the backup.py picks the live
    DB by Path.exists() finding'. It closed the half where the target was
    hard-coded to DATA_DIR. The guess is still the guess.

    THE CALLERS, all of them:
      compactor/backup.py:1128  CLI --restore   -> neither -> WEBUI_DB  (the
                                                  guess; production path)
      compactor/test_backup.py:215,743          -> neither -> WEBUI_DB, which
                                                  that suite pins EQUAL to
                                                  DATA_DIR/webui.db
      compactor/test_backup.py:785              -> webui_db= (explicit)
      nothing at all                            -> data_dir=
    So the `data_dir is not None` branch has no caller, and the branch the
    operator actually runs is the one no test can tell apart from the other.
    """
    head("R5 FINDING: WEBUI_DB is an exists() guess, not the gate")
    check("WEBUI_DB_LOCAL" not in (_REPO / "compactor" / "backup.py").read_text(
              encoding="utf-8"),
          "FINDING: the string WEBUI_DB_LOCAL does not appear in backup.py")

    # The rollback: gate flipped to false inside a pod's life, stale local
    # file still on the overlay, live database is the snapshot.
    _make_db(_LOCAL_DB, "STALE - nobody has written here since the flag flipped")
    _make_db(_SNAP_DB, "LIVE - every conversation she has had since")
    os.environ["WEBUI_DB_LOCAL"] = "false"
    importlib.reload(backup)
    try:
        check(backup.WEBUI_DB == _LOCAL_DB,
              "FINDING: with WEBUI_DB_LOCAL=false, backup.py still resolves "
              "the live db to the abandoned local file (%s)" % backup.WEBUI_DB)

        _seed_store()
        _clean_backups()
        rep = backup.run_once()
        check(rep.get("ok"), "the backup cycle reports success")
        arch = _BACKUPS / rep["archive"]
        # What is actually inside it?
        import tarfile
        scratch = Path(tempfile.mkdtemp())
        try:
            with tarfile.open(arch, "r:gz") as tar:
                tar.extractall(scratch, filter="data")
            body = _read_chat(scratch / "webui.db")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        check(body is not None and body.startswith("STALE"),
              "FINDING: the archive holds the STALE database (%r)" % (body,))
        note("the live snapshot holds %r and no archive contains it"
             % _read_chat(_SNAP_DB))

        # ...and the restore half lands on the same wrong file.
        _make_db(_LOCAL_DB, "STALE overwritten-by-nobody")
        _make_db(_SNAP_DB, "LIVE and about to be left alone")
        backup.restore_backup(arch, confirm=True)
        check(_read_chat(_LOCAL_DB).startswith("STALE"),
              "FINDING: the restore landed on the abandoned local file")
        check(_read_chat(_SNAP_DB).startswith("LIVE"),
              "FINDING: the file OpenWebUI is reading was never touched")
        note("the operator's recovery reported success and changed nothing "
             "the user can see")
    finally:
        os.environ.pop("WEBUI_DB_LOCAL", None)
        _LOCAL_DB.unlink(missing_ok=True)
        importlib.reload(backup)


def r6_import_time_freeze():
    """FINDING (MEDIUM). WEBUI_DB is a module CONSTANT, evaluated once. The
    CLI is a fresh process so it sees the truth of the moment; the backup
    DAEMON (supervisord priority 40) and main.py (priority 15) are not, and
    priority 15 is BEFORE openwebui(20). On a first-boot pod where neither
    database exists yet, both freeze WEBUI_DB to DATA_DIR/webui.db for the
    life of the process — so /admin/backups and every nightly cycle archive
    the snapshot, whose staleness is SYNC_INTERVAL_S on top of the archive's
    own age, which is the exact thing backup.py:77's comment says it must
    not do.
    """
    head("R6 FINDING: WEBUI_DB is frozen at import, in two long-lived procs")
    _LOCAL_DB.unlink(missing_ok=True)
    os.environ["WEBUI_DB_LOCAL"] = "true"
    importlib.reload(backup)
    try:
        check(backup.WEBUI_DB == _DATA / "webui.db",
              "no local file at import -> WEBUI_DB is the snapshot")
        # OpenWebUI starts and creates it, as it does on a fresh deployment.
        _make_db(_LOCAL_DB, "the live database OpenWebUI just created")
        check(backup.WEBUI_DB == _DATA / "webui.db",
              "FINDING: the local file now exists and WEBUI_DB never moves — "
              "a re-import is the only thing that would notice")
        note("the fix is a function, not a constant: resolve the live db per "
             "call, from WEBUI_DB_LOCAL, the way entrypoint.sh does")
    finally:
        os.environ.pop("WEBUI_DB_LOCAL", None)
        importlib.reload(backup)


# ===========================================================================
# R7. The third site of the same sibling miss.
# ===========================================================================

def r7_the_third_site_health_probe_watches_the_wrong_journal():
    """FINDING (HIGH). The brief asked whether there is a THIRD site. There
    is, and it is a check that cannot fire.

    health.probe_sqlite_journal — 'Is OpenWebUI's database sitting next to a
    HOT rollback journal?' — reads WEBUI_SNAPSHOT_DB and looks for
    `<snapshot>-journal`. Under the shipped default (WEBUI_DB_LOCAL=true)
    OpenWebUI's database is /var/lib/openwebui/webui.db and its journal is
    /var/lib/openwebui/webui.db-journal. The probe never looks there.

    Its own sibling 130 lines below, probe_snapshot, DOES read the gate
    (`WEBUIDB_SYNC_ENABLED`) and says so in its docstring. probe_sqlite_journal
    was written for v3.1.5, where the snapshot WAS live; v3.1.6 moved the live
    database and the probe stayed. The 2026-09-07 incident it cites — 'an
    orphaned hot journal sat beside a database that was otherwise being
    written to normally, so nothing looked wrong' — is now undetectable
    again, on the file that matters, by the check added to detect it.

    A docstring that disagrees with its code: the docstring is right and the
    path is wrong.
    """
    head("R7 FINDING: the third site — probe_sqlite_journal reads the snapshot")
    os.environ["WEBUI_DB_LOCAL"] = "true"
    os.environ["WEBUIDB_SYNC_ENABLED"] = "true"
    import health
    importlib.reload(health)
    try:
        _make_db(_LOCAL_DB, "live")
        _make_db(_SNAP_DB, "snapshot")
        # The exact 2026-09-07 shape: a hot journal beside the LIVE database.
        live_journal = _LOCAL_DB.with_name(_LOCAL_DB.name + "-journal")
        live_journal.write_bytes(_HOT + b"an uncommitted transaction")
        res = health.probe_sqlite_journal()
        check(res["hot"] is False,
              "FINDING: hot=False with a real hot journal beside the LIVE db")
        check(res["ok"] is True,
              "FINDING: ok=True, so /health/full reports no reason at all")
        check(str(_SNAP_DB) in res["path"],
              "FINDING: the path it probed is the SNAPSHOT (%s)" % res["path"])
        note("the live journal it missed is at %s" % live_journal)

        # CONTROL: the probe is not simply broken — point it at the live
        # database and it fires immediately. The defect is the PATH.
        os.environ["WEBUI_SNAPSHOT_DB"] = str(_LOCAL_DB)
        importlib.reload(health)
        res2 = health.probe_sqlite_journal()
        check(res2["hot"] is True,
              "CONTROL: aimed at the live db the same probe says hot=True")
        live_journal.unlink()
    finally:
        os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_DB)
        os.environ.pop("WEBUI_DB_LOCAL", None)
        os.environ.pop("WEBUIDB_SYNC_ENABLED", None)
        importlib.reload(health)


# ===========================================================================
# R8. Atomic is not durable, and the project has a primitive that knows it.
# ===========================================================================

def r8_the_replace_is_never_fsynced():
    """FINDING (HIGH). 'ATOMIC. A crash mid-copy left a TRUNCATED live
    database where a whole one had been ... The temp name is a sibling so
    os.replace stays on one filesystem.'

    Atomic against another READER. Not durable against the failure the
    comment names. shutil.copy2 does not fsync, and neither does
    restore_backup: the replace can be visible in the page cache and the
    temp's data blocks never written. memory.atomic_write_json — this
    project's own primitive, in the module this same area owns — fsyncs the
    file AND the parent directory, and its comment says exactly why:

        # ... this state lives on a distributed network volume (MooseFS),
        # where rename/fsync guarantees are weaker than local POSIX.

    With WEBUI_DB_LOCAL=false the restore target IS on MooseFS. So the one
    code path whose whole purpose is surviving a crash omits the two syscalls
    the module next door added for surviving a crash on this exact volume.

    NOT DEMONSTRATED HERE, and said in those words: proving it needs a host
    that loses power between the replace and the writeback. The evidence is
    the absence of the calls, which is checkable.
    """
    head("R8 FINDING: the atomic replace is never fsynced (static evidence)")
    src = (_REPO / "compactor" / "backup.py").read_text(encoding="utf-8")
    start = src.index("def restore_backup(")
    body = src[start:src.index("\ndef ", start + 10)]
    check("os.replace(tmp, target)" in body,
          "restore_backup does os.replace(tmp, target)")
    check("fsync" not in body,
          "FINDING: and the word 'fsync' does not appear in the function")
    mem = (_REPO / "compactor" / "memory.py").read_text(encoding="utf-8")
    mstart = mem.index("def atomic_write_json(")
    mbody = mem[mstart:mem.index("\nclass ", mstart)]
    check(mbody.count("fsync") >= 2,
          "SOUND sibling: memory.atomic_write_json fsyncs the file AND the "
          "parent dir (%d mentions)" % mbody.count("fsync"))
    note("smallest fix: open the temp, os.fsync it before the replace, and "
         "fsync the target's parent dir after — three lines, copied from "
         "memory.atomic_write_json")


# ===========================================================================
# R9. os.replace is invisible to a running OpenWebUI, and copy2 was not.
# ===========================================================================

def r9_replace_hides_the_restore_from_an_open_writer():
    """FINDING (HIGH). The atomicity fix changed the restore from
    'overwrite the bytes of the live inode' to 'swap in a new inode'. A
    process holding the old file open keeps writing to the OLD, now-unlinked
    inode: the restore is invisible to it, and its next journal — named from
    the PATH, so landing beside the NEW file — describes the OLD file's page
    layout. That journal applied to the restored database is the corruption
    this commit set out to prevent, reintroduced from the other end.

    Nothing stops it. restore_backup's docstring does not mention stopping
    OpenWebUI, the CLI's --restore help is 'Restore from an archive
    (DESTRUCTIVE)' and nothing more, and the commit message's only operator
    advice is to 'check for a hot journal first'.
    """
    head("R9 FINDING: os.replace is invisible to an open writer")
    os.environ["WEBUI_DB_LOCAL"] = "true"
    _make_db(_LOCAL_DB, "rows")
    importlib.reload(backup)
    try:
        arch = _archive(db_body="the archived rows")
        target = backup.WEBUI_DB
        # A writer with the file open, exactly as OpenWebUI has it.
        con = sqlite3.connect(str(target))
        con.execute("INSERT INTO chat (chat) VALUES ('written after the archive')")
        con.commit()
        try:
            backup.restore_backup(arch, confirm=True)
            rows_on_disk = _read_chat(target)
            rows_in_writer = con.execute(
                "SELECT count(*) FROM chat").fetchone()[0]
            check(rows_on_disk == "the archived rows",
                  "the file at the path is the restored one")
            check(rows_in_writer == 2,
                  "FINDING: the open connection still sees its OWN 2 rows — "
                  "it is holding the unlinked inode and never saw the restore")
            note("the CLI reported success; the running front end is still "
                 "serving, and still writing, the database that was replaced")
            src = (_REPO / "compactor" / "backup.py").read_text(encoding="utf-8")
            fn = src[src.index("def restore_backup("):]
            fn = fn[:fn.index("\ndef ", 10)]
            for word in ("supervisorctl", "stop openwebui", "not running"):
                check(word not in fn,
                      "FINDING: restore_backup never mentions %r" % word)
        finally:
            con.close()
    finally:
        os.environ.pop("WEBUI_DB_LOCAL", None)
        importlib.reload(backup)


# ===========================================================================
# R10. What IS sound here, stated precisely.
# ===========================================================================

def r10_what_is_sound():
    """SOUND. Three attacks the brief named that this code survives."""
    head("R10 SOUND: three attacks that do not land")
    src = (_REPO / "compactor" / "backup.py").read_text(encoding="utf-8")
    fn = src[src.index("def restore_backup("):]
    fn = fn[:fn.index("\ndef ", 10)]

    # 1. Cross-filesystem os.replace. The temp is ALWAYS target.with_name(),
    #    i.e. always in target's own directory, in all three branches — there
    #    is no branch where the temp can be on a different filesystem from
    #    the target, so the MooseFS/local boundary is unreachable here.
    check('tmp = target.with_name(' in fn,
          "SOUND: the temp is target.with_name(...) — same directory in all "
          "three target branches, so os.replace cannot cross a filesystem")
    check(fn.count("tmp = ") == 1,
          "SOUND: there is exactly one temp construction (no sibling to miss)")

    # 2. The renamed sidecars being re-read as sidecars. SQLite derives
    #    sidecar names by EXACT suffix; every sweep in this repo does too.
    for f in ("compactor/webuidb.py", "scripts/recover-webui-db.py",
              "compactor/backup.py"):
        t = (_REPO / f).read_text(encoding="utf-8")
        check("with_name(DB.name + s)" in t or "path.name + suffix" in t
              or "target.name + suffix" in t or "DB.name + s" in t,
              f"SOUND: {f} matches sidecars by exact suffix, so "
              f"'*.pre-restore-*' can never be picked up as one")

    # 3. tar extraction. filter='data' on both extractall sites (verify and
    #    restore), and there are only two.
    check(src.count("extractall(") == 2 and src.count('filter="data"') == 2,
          "SOUND: both extractall sites in backup.py pass filter='data' "
          "(%d extractall / %d filtered)"
          % (src.count("extractall("), src.count('filter="data"')))

    # 4. And one thing that is NOT cleaned up, stated as fact.
    check("pre-restore" in src and src.count("pre-restore") == 1,
          "the .pre-restore- name is constructed in exactly one place")
    check("pre-restore" not in (_REPO / "compactor" / "webuidb.py").read_text(
              encoding="utf-8"),
          "FINDING (LOW): nothing anywhere else in the tree mentions "
          "'pre-restore', so nothing ever lists, reports or prunes these "
          "files — they accumulate one per restore, forever, unowned")


def main():
    print("adversarial: backup / restore / store file IO (hostile pass #2)")
    print(f"tmp tree: {_TMP}")
    for fn in (
        r1_sidecars_stripped_before_the_copy_can_fail,
        r2_store_half_is_rmtree_then_copytree,
        r3_stamp_resolution_regresses_a_lesson_webuidb_already_learned,
        r4_set_asides_land_on_the_volume_that_dies_with_the_pod,
        r5_backup_and_restore_both_follow_exists_not_the_gate,
        r6_import_time_freeze,
        r7_the_third_site_health_probe_watches_the_wrong_journal,
        r8_the_replace_is_never_fsynced,
        r9_replace_hides_the_restore_from_an_open_writer,
        r10_what_is_sound,
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
            globals()["_FAIL"] = globals()["_FAIL"] + 1
    print("")
    print(f"RESULT: {_PASS} checks passed, {_FAIL} failed")
    shutil.rmtree(_TMP, ignore_errors=True)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
