"""
v3.1.9 fix-lane tests for compactor.backup — hostile317-c F1/F3 and
hostile2-backup A3-6b/A3-10/A3-11.

Kept separate from test_backup.py (which the brief for this lane forbids
editing outside the small QUARANTINE-location updates already made there) so
each fix's proof, its control, and the "why nothing caught it before" story
stay next to each other rather than interleaved into an already-1000-line
file.

The census tests (F1) build their "normal operation" fixture with the REAL
writer, `facts.prune_facts(conv_id=...)`, not a hand-edited count — that is
what actually produces the active/archive split the census now unions. The
summarizer half is built as the exact STATE SHAPE `_do_l2_rollup` /
`_do_l3_rollup` produce (read from summarizer.py directly, cited below),
because driving the real async rollup needs a live vLLM stub that is out of
scope for a backup-module suite.

Run inside the compactor image or any container with the requirements
installed:
    python test_backup_v319.py
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

_TMP = Path(tempfile.mkdtemp(prefix="zions-backup-v319-test-"))
_DATA = _TMP / "data" / "openwebui"
_STORE = _DATA / "compactor"
_BACKUPS = _TMP / "data" / "backups"
_DB = _DATA / "webui.db"
_QUARANTINE = _TMP / "quarantine"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BACKUPS)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DB)
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUARANTINE)

import backup  # noqa: E402
import facts  # noqa: E402  (the REAL writer for the F1 fixtures)
import memory  # noqa: E402


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


def assert_raises(fn, exc, label):
    try:
        fn()
    except exc:
        print(f"  ok   {label}")
        return
    except Exception as e:
        print(f"FAIL {label}: expected {exc.__name__}, got {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"FAIL {label}: nothing raised")
    sys.exit(1)


def _wipe_store():
    if _STORE.exists():
        shutil.rmtree(_STORE)
    (_STORE / "facts").mkdir(parents=True, exist_ok=True)
    (_STORE / "summaries").mkdir(parents=True, exist_ok=True)


def _write_summary_state(conv_id: str, *, l1=None, l2=None, l3=None,
                          last_summarized_turn: int):
    p = _STORE / "summaries" / f"{conv_id}.json"
    p.write_text(json.dumps({
        "conv_id": conv_id, "l1": l1 or [], "l2": l2 or [], "l3": l3,
        "last_summarized_turn": last_summarized_turn,
    }), encoding="utf-8")


def _write_chapter_archive(conv_id: str, chapters: list[dict]):
    # summarizer._archive_chapters's own on-disk shape: {"chapters": [...]}.
    p = _STORE / "summaries" / f"{conv_id}.archive.json"
    p.write_text(json.dumps({"conv_id": conv_id, "chapters": chapters}),
                 encoding="utf-8")


# ---------------------------------------------------------------------------
# F1 — census regression false positives (hostile317-c)
# ---------------------------------------------------------------------------

def test_real_prune_facts_eviction_does_not_regress_the_census():
    """facts.prune_facts(conv_id=...) — the REAL writer — evicts to the
    archive sidecar when a conversation hits its token cap. The union the
    fixed census reads must not move; the OLD active-only census would have
    (that is F1 verbatim: her real logs show this exact shape every night
    from 2026-08-31 through 2026-09-11)."""
    print("\n[test] F1: real prune_facts eviction does not trip the guard")
    _wipe_store()
    conv_id = "conv-evict"
    initial = [
        {"text": f"fact number {i} with enough padding to cost tokens " + ("x" * 40),
         "added_turn": i, "last_used": i, "pin": False}
        for i in range(40)
    ]
    facts.save_facts(conv_id, initial)
    prev = backup._census(_STORE)
    assert_true(prev[conv_id]["facts"] == 40, "fixture: 40 active facts before eviction")

    # The real writer: a small cap forces eviction, and passing conv_id
    # archives what it evicts (facts.py:989 docstring) instead of deleting it.
    kept, dropped = facts.prune_facts(initial, max_tokens=200, conv_id=conv_id)
    assert_true(dropped > 0, "fixture: the cap actually evicted something")
    assert_true(len(kept) < 40,
                "fixture: the ACTIVE count really did shrink (the old bug's trigger)")
    facts.save_facts(conv_id, kept)

    new = backup._census(_STORE)
    # p3-b F3: `facts` reverted to its OLD, active-only meaning (an older
    # binary reads it after a rollback — see p3-b F3), so the union eviction
    # must not disturb now lives in facts + archived_facts together.
    assert_true(new[conv_id]["facts"] < 40,
                "fixture: the ACTIVE-only field really did drop (that is "
                "the point of eviction)")
    assert_eq(new[conv_id]["facts"] + new[conv_id]["archived_facts"], 40,
              "F1 fix: active+archived UNION is unchanged — eviction moved "
              "facts, it did not lose them")

    shortfalls = backup._census_regressions(prev, new)
    assert_eq(shortfalls, [],
              "F1 fix: normal eviction does not appear as a census regression")


def test_dedup_style_merge_does_not_regress_the_census():
    """Dedup permanently combines duplicate facts into fewer, denser
    entries — a real, legitimate decrease in the union itself (not just a
    move between active/archive). The finding names this explicitly as a
    false-positive source alongside eviction and rollups."""
    print("\n[test] F1: a dedup-shaped merge (fewer facts, union shrinks) does not trip the guard")
    _wipe_store()
    conv_id = "conv-dedup"
    facts.save_facts(conv_id, [
        {"text": "she was baptized on 12 April", "last_used": 1},
        {"text": "baptized 12 April (duplicate)", "last_used": 2},
        {"text": "unrelated fact", "last_used": 3},
    ])
    prev = backup._census(_STORE)
    assert_eq(prev[conv_id]["facts"], 3, "fixture: three facts before merge")

    # Dedup's own effect on disk: two entries replaced by one merged entry.
    # (dedup.py is out of this lane's file list; the fixture is the STATE
    # SHAPE it produces — a shorter list, same recoverable information.)
    facts.save_facts(conv_id, [
        {"text": "she was baptized on 12 April", "last_used": 2},
        {"text": "unrelated fact", "last_used": 3},
    ])
    new = backup._census(_STORE)
    assert_eq(new[conv_id]["facts"], 2, "fixture: the union really did shrink by one")

    shortfalls = backup._census_regressions(prev, new)
    assert_eq(shortfalls, [],
              "F1 fix: a partial union decrease (merge) is not flagged — "
              "only the union going to zero is")


def test_facts_file_emptied_still_trips_the_guard():
    """CONTROL for both tests above: the guard must still fire on the shape
    named as real loss — the union going to zero when it held something."""
    print("\n[test] F1 CONTROL: a facts file (and its archive) truly emptied still regresses")
    _wipe_store()
    conv_id = "conv-lost"
    facts.save_facts(conv_id, [{"text": "irreplaceable", "last_used": 1}])
    prev = backup._census(_STORE)
    assert_eq(prev[conv_id]["facts"], 1, "fixture: one fact before the loss")

    facts.save_facts(conv_id, [])  # the exact shape of a wiped facts file
    new = backup._census(_STORE)
    assert_eq(new[conv_id]["facts"], 0, "fixture: the union is now empty")

    shortfalls = backup._census_regressions(prev, new)
    assert_true(any(s.startswith(f"{conv_id}.facts") for s in shortfalls),
                f"F1 CONTROL: a real loss is still caught (got {shortfalls})")


def test_l1_to_l2_rollup_shape_does_not_regress_the_census():
    """summarizer._do_l2_rollup (summarizer.py:1993-2020) drops the rolled
    L1 chunks and appends one L2 chapter; last_summarized_turn is untouched
    by that rollup (it advances only in _do_l1_rollup). The OLD census
    (len(l1)+len(l2)+bool(l3)) drops from L2_CHUNK_SIZE to 1 here even
    though nothing was lost — this fixture is that exact state transition."""
    print("\n[test] F1: an L1->L2 rollup's state shape does not trip the guard")
    _wipe_store()
    conv_id = "conv-l2roll"
    l1_before = [{"text": f"scene {i}", "first_turn": i * 20 + 1, "last_turn": i * 20 + 20}
                 for i in range(10)]  # summarizer.py L2_CHUNK_SIZE default = 10
    _write_summary_state(conv_id, l1=l1_before, l2=[], l3=None,
                          last_summarized_turn=200)
    prev = backup._census(_STORE)
    assert_eq(prev[conv_id]["summary_turn"], 200, "fixture: watermark before rollup")

    # _do_l2_rollup's own transformation, verbatim: l1 = l1[L2_CHUNK_SIZE:],
    # one chapter appended to l2, last_summarized_turn NOT reassigned.
    chapter = {"text": "chapter summarizing 10 scenes", "first_turn": 1, "last_turn": 200}
    _write_summary_state(conv_id, l1=[], l2=[chapter], l3=None,
                          last_summarized_turn=200)
    new = backup._census(_STORE)

    shortfalls = backup._census_regressions(prev, new)
    assert_eq(shortfalls, [],
              "F1 fix: ten chunks collapsing into one chapter is not a "
              "regression — the watermark did not move")


def test_l2_to_l3_rollup_archives_chapters_and_does_not_regress_the_census():
    """summarizer._do_l3_rollup (summarizer.py:2098+) clears l2 and archives
    the consumed chapters via _archive_chapters — 'the chapters are archived
    precisely so that is a quality trade and not a loss' (its own docstring).
    This fixture is that exact before/after."""
    print("\n[test] F1: an L2->L3 rollup's state shape (chapters archived) does not trip the guard")
    _wipe_store()
    conv_id = "conv-l3roll"
    l2_before = [{"text": f"chapter {i}", "first_turn": i * 200 + 1, "last_turn": i * 200 + 200}
                 for i in range(5)]  # summarizer.py L3_CHUNK_SIZE default = 5
    _write_summary_state(conv_id, l1=[], l2=l2_before, l3=None,
                          last_summarized_turn=1000)
    prev = backup._census(_STORE)
    assert_eq(prev[conv_id]["archived_chapters"], 0, "fixture: nothing archived yet")

    # _do_l3_rollup's own transformation: l2 cleared, l3 becomes one object,
    # last_summarized_turn untouched, and _archive_chapters appends the
    # consumed chapters to the cold sidecar (summaries/<id>.archive.json).
    _write_summary_state(conv_id, l1=[], l2=[], l3={"text": "theme so far", "first_turn": 1},
                          last_summarized_turn=1000)
    _write_chapter_archive(conv_id, l2_before)
    new = backup._census(_STORE)
    assert_eq(new[conv_id]["archived_chapters"], 5, "fixture: five chapters now archived")

    shortfalls = backup._census_regressions(prev, new)
    assert_eq(shortfalls, [],
              "F1 fix: chapters moving from l2 into cold storage is not a regression")


def test_summary_coverage_going_backwards_still_trips_the_guard():
    """CONTROL: the finding names 'summaries' coverage going backwards' as
    the one summary-side shape that must still be caught."""
    print("\n[test] F1 CONTROL: last_summarized_turn regressing still trips the guard")
    _wipe_store()
    conv_id = "conv-coverage-lost"
    _write_summary_state(conv_id, l1=[], l2=[], l3=None, last_summarized_turn=500)
    prev = backup._census(_STORE)
    _write_summary_state(conv_id, l1=[], l2=[], l3=None, last_summarized_turn=100)
    new = backup._census(_STORE)

    shortfalls = backup._census_regressions(prev, new)
    assert_true(any(s.startswith(f"{conv_id}.summary_turn") for s in shortfalls),
                f"F1 CONTROL: a real coverage regression is still caught (got {shortfalls})")


def test_archived_chapters_disappearing_still_trips_the_guard():
    """CONTROL: the chapter archive is append-only cold storage — the only
    remaining copy of chapter-level detail once L3 has paraphrased it — so a
    decrease there is real loss, unlike the collapse in the rollup fixture
    above (which only ever ADDS to it)."""
    print("\n[test] F1 CONTROL: archived_chapters disappearing still trips the guard")
    _wipe_store()
    conv_id = "conv-chapters-lost"
    _write_summary_state(conv_id, last_summarized_turn=1000)
    _write_chapter_archive(conv_id, [{"text": "c1"}, {"text": "c2"}, {"text": "c3"}])
    prev = backup._census(_STORE)
    _write_chapter_archive(conv_id, [{"text": "c1"}])
    new = backup._census(_STORE)

    shortfalls = backup._census_regressions(prev, new)
    assert_true(any(s.startswith(f"{conv_id}.archived_chapters") for s in shortfalls),
                f"F1 CONTROL: a real archived-chapter loss is still caught (got {shortfalls})")


def test_a_conversation_disappearing_entirely_still_trips_the_guard():
    """CONTROL: 'a conversation's store gone' — the whole per-conv entry is
    absent from the new census, not just one layer of it."""
    print("\n[test] F1 CONTROL: a conversation vanishing entirely still trips the guard")
    _wipe_store()
    conv_id = "conv-vanished"
    facts.save_facts(conv_id, [{"text": "only copy", "last_used": 1}])
    _write_summary_state(conv_id, last_summarized_turn=50)
    prev = backup._census(_STORE)
    assert_true(prev[conv_id]["facts"] == 1 and prev[conv_id]["summary_turn"] == 50,
                "fixture: the conversation is fully present before")

    (_STORE / "facts" / f"{conv_id}.json").unlink()
    (_STORE / "summaries" / f"{conv_id}.json").unlink()
    new = backup._census(_STORE)
    assert_true(conv_id not in new, "fixture: the conversation is gone from the new tree")

    shortfalls = backup._census_regressions(prev, new)
    assert_true(any(s.startswith(f"{conv_id}.") for s in shortfalls),
                f"F1 CONTROL: a vanished conversation is still caught (got {shortfalls})")


# ---------------------------------------------------------------------------
# F3 — hot rollback journal makes the online-backup snapshot fail
# ---------------------------------------------------------------------------

def _make_hot_journal_pair(tmp_dir: Path, committed_rows, uncommitted_row):
    """A REAL sqlite db with a genuine HOT rollback journal — produced by
    actually SIGKILLing a child process mid-transaction (the same mechanism
    hostile317-c's own scripts/hotj.py used), not by copying files around a
    live connection in this process. That matters: while a connection is
    still open and alive, a SEPARATE reader sees "database is locked", not
    a hot journal needing recovery — hot-journal recovery is specifically
    what happens when the process that owns the lock is GONE. A same-
    process file copy cannot produce that; only losing the process can.

    p3-b F1. "A real journal file" is not the same thing as "a HOT one",
    and "a hot one" is not automatically enough to make a mutation THIS
    TEST exists to catch observable, either — both had to be learned the
    hard way, empirically, against this exact fixture:

      1. SQLite (pager.c hasHotJournal) only calls a journal hot if its
         first 8 bytes are the real magic, written ONLY when the journal
         is synced (COMMIT phase 1, or a page-cache SPILL mid-transaction).
         An earlier version of this fixture did one small uncommitted
         INSERT: no spill, no sync, a zeroed header — SQLite ignores it
         completely, `mode=ro` opens cleanly with no error at all, and the
         test could only see the readonly failure by FAKING it (patching
         sqlite3.connect). See hotj3.py's own proof table (SP\\p3-b\\
         hotj3.py) for which writer shapes are actually hot.
      2. Fixed that with `PRAGMA cache_size=10` and 200-3000 padding
         INSERTs — a real, synced, hot journal, confirmed by its magic
         bytes. But INSERTs APPEND new pages past the file's existing
         "database size in pages" field (page 1, offset 28) — a field
         only updated at COMMIT, never touched by a spill. A reader
         opening the spilled file WITHOUT its journal still only sees the
         OLD page count and never looks at the appended pages at all, so
         the finding's own sidecar-copy-deleted mutation (mut_hotj.py)
         stayed GREEN even at 3,000 padding rows: the row count the test
         asserted on happened to come out right for the wrong reason
         either way.
      3. So: an UPDATE of EXISTING, already-committed rows, not an
         INSERT of new ones. A spilled page holding an updated row is
         WITHIN the old page-count boundary — visible to any reader, with
         or without a journal — which is exactly what makes the
         difference observable, and matches the finding's own
         proven-working shapes (hotj3.py: "3,000-row UPDATE...";
         test_p3b_restore_rollback.py's `_build_live_db_with_real_hot_
         journal` in this same lane, proven against the real F5 rollback
         bug the same way).

    `committed_rows` seed the table; `uncommitted_row` REPLACES every one
    of them inside the killed transaction — the snapshot this test builds
    must show the ORIGINAL rows, not that replacement.

    Returns (db_bytes, journal_bytes) read back after the kill.
    """
    src = tmp_dir / "src.db"
    ready = tmp_dir / "ready.flag"
    script = (
        "import sqlite3\n"
        f"con = sqlite3.connect({str(src)!r}, isolation_level=None)\n"
        "con.execute('PRAGMA journal_mode=DELETE')\n"
        "con.execute('CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)')\n"
        "con.execute('BEGIN')\n"
        f"for row in {list(committed_rows)!r}:\n"
        "    con.execute('INSERT INTO chat (body) VALUES (?)', (row,))\n"
        "con.commit()\n"
        # p3-b F1: a tiny cache forces the UPDATE below to SPILL dirty
        # pages to disk mid-transaction (syncing the journal header)
        # WHILE rewriting pages that already exist — see this function's
        # own docstring, point 3, for why that is load-bearing and a
        # padding INSERT is not.
        "con.execute('PRAGMA cache_size=10')\n"
        "con.execute('BEGIN')\n"
        f"con.execute('UPDATE chat SET body = ?', ({uncommitted_row!r} + 'x' * 4000,))\n"
        f"open({str(ready)!r}, 'w').write('ready')\n"
        "import time\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-u", "-c", script])
    try:
        deadline = time.time() + 30
        while not ready.is_file():
            if time.time() > deadline:
                raise RuntimeError("fixture: child never signaled ready")
            time.sleep(0.02)
        # Give SQLite a moment past the last INSERT to make sure the spill
        # and the journal header sync are actually on disk, not just in
        # the OS page cache of a process about to disappear.
        time.sleep(0.3)
        journal = src.with_name(src.name + "-journal")
        assert journal.is_file(), "fixture: sqlite really left a journal mid-transaction"
    finally:
        proc.kill()  # SIGKILL — no chance to close() or roll back cleanly
        proc.wait(timeout=10)
    journal = src.with_name(src.name + "-journal")
    assert journal.is_file(), (
        "fixture: the journal survived the kill (it must — nothing but the "
        "OS releasing the lock should have happened)"
    )
    header = journal.read_bytes()[:8]
    _HOT_MAGIC = bytes.fromhex("d9d505f920a163d7")
    assert header == _HOT_MAGIC, (
        f"fixture: the journal header must be the REAL synced magic (a "
        f"genuinely hot journal), got {header.hex()!r} — a zeroed header "
        f"means the transaction never spilled the cache, SQLite will "
        f"ignore this journal entirely, and this test would prove nothing "
        f"(p3-b F1)"
    )
    db_bytes = src.read_bytes()
    journal_bytes = journal.read_bytes()
    return db_bytes, journal_bytes


def test_hot_journal_falls_back_to_a_recovered_copy_instead_of_failing():
    """p3-b F1. The PREVIOUS version of this test could not reproduce the
    real "attempt to write a readonly database" error because its fixture
    never produced a genuinely HOT journal (see _make_hot_journal_pair's
    docstring) — so it had to FAKE the error by patching sqlite3.connect,
    which meant it could never notice the fallback losing the journal: a
    mutant that deleted the sidecar-copy line inside
    _snapshot_via_rollback_copy stayed green (p3-b F1's own proof,
    mut_hotj.py). Now that the fixture forces a real cache spill (a
    synced, hot journal, asserted by its magic bytes), the SAME real error
    SQLite raises in production (SQLITE_READONLY_ROLLBACK, "attempt to
    write a readonly database" — confirmed independently on real user data
    by hostile pass #3D) fires naturally on the real mode=ro connect
    below. No patch on sqlite3.connect at all.
    """
    print("\n[test] F3: a hot rollback journal falls back to a recovered-copy snapshot")
    scratch = Path(tempfile.mkdtemp())
    try:
        # Enough rows that an UPDATE touching all of them, under a 10-page
        # cache, really spills — see _make_hot_journal_pair's docstring,
        # point 3, for why an UPDATE (not an INSERT) is load-bearing here.
        committed_rows = [f"original-row-{i:04d}" for i in range(2000)]
        db_bytes, journal_bytes = _make_hot_journal_pair(
            scratch, committed_rows, "UPDATED-MUST-NOT-SURVIVE-"
        )
        live_dir = scratch / "live"
        live_dir.mkdir()
        live_db = live_dir / "webui.db"
        live_journal = live_dir / "webui.db-journal"
        live_db.write_bytes(db_bytes)
        live_journal.write_bytes(journal_bytes)

        real_connect = sqlite3.connect

        # p3-b F1: proves the journal was actually NEEDED, not merely
        # present — opened WITHOUT it, the raw spilled file already shows
        # the UPDATE's uncommitted values, because an UPDATE (unlike an
        # INSERT past the old page count) rewrites pages already within
        # the database's counted size and so is visible to any reader.
        raw_copy = scratch / "raw_no_journal.db"
        raw_copy.write_bytes(db_bytes)
        con = real_connect(str(raw_copy))
        try:
            raw_rows = [r[0] for r in con.execute("SELECT body FROM chat ORDER BY id")]
        finally:
            con.close()
        assert_true(
            any(r.startswith("UPDATED-MUST-NOT-SURVIVE-") for r in raw_rows),
            f"fixture: the raw db file WITHOUT its journal already shows the "
            f"UPDATE's uncommitted values (got {raw_rows[:2]}...) — proves the "
            f"cache really did spill dirty pages INTO the committed rows, and "
            f"that the journal this test is about is not decorative"
        )

        dest = scratch / "snapshot.db"
        with patch.object(backup, "_snapshot_via_rollback_copy",
                           wraps=backup._snapshot_via_rollback_copy) as spy:
            ok = backup._snapshot_sqlite(live_db, dest)
            assert_true(spy.called,
                        "F1/F3 fix: the fallback path was actually used — "
                        "against the REAL SQLITE_READONLY_ROLLBACK error, "
                        "no patched connect() involved")
        assert_true(ok, "F3 fix: the snapshot succeeded instead of raising")
        assert_true(dest.is_file(), "a snapshot file was written")

        con = real_connect(str(dest))
        try:
            rows = [r[0] for r in con.execute("SELECT body FROM chat ORDER BY id")]
        finally:
            con.close()
        assert_eq(rows, committed_rows,
                  "the snapshot holds the ORIGINAL committed values — the "
                  "UPDATE was correctly rolled back, not fabricated into it, "
                  "unlike the raw-file values above")

        assert_eq(live_db.read_bytes(), db_bytes,
                  "the LIVE database was only ever read, never modified")
        assert_eq(live_journal.read_bytes(), journal_bytes,
                  "the LIVE journal was only ever read, never touched — "
                  "recover-webui-db.py's rule: a journal and its database "
                  "are a matched pair, and the pair here is untouched")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_a_normal_db_without_a_journal_never_uses_the_fallback():
    """CONTROL for the dispatch itself: the fast mode=ro path is untouched
    for the common case, so this fix cannot regress it."""
    print("\n[test] F3 CONTROL: a healthy db (no journal) never reaches the fallback")
    scratch = Path(tempfile.mkdtemp())
    try:
        src = scratch / "healthy.db"
        con = sqlite3.connect(str(src))
        con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
        con.execute("INSERT INTO chat (body) VALUES ('fine')")
        con.commit()
        con.close()
        dest = scratch / "snap.db"
        with patch.object(backup, "_snapshot_via_rollback_copy") as spy:
            ok = backup._snapshot_sqlite(src, dest)
        assert_true(ok, "snapshot ok")
        assert_true(not spy.called, "the fallback was never called")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_an_unrelated_operational_error_still_raises():
    """CONTROL: only the specific readonly-database signature falls back —
    anything else must still propagate, unchanged from before this fix."""
    print("\n[test] F3 CONTROL: a non-journal OperationalError still raises")
    scratch = Path(tempfile.mkdtemp())
    try:
        missing = scratch / "does-not-exist-as-a-db-file-but-is-a-file.db"
        missing.write_bytes(b"not a sqlite file at all")
        dest = scratch / "snap.db"
        assert_raises(lambda: backup._snapshot_sqlite(missing, dest),
                      sqlite3.DatabaseError,
                      "a corrupt (non-journal) file still raises, not silently recovered")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


class _StopLoop(Exception):
    pass


def test_daemon_retries_soon_after_a_failed_cycle_not_the_full_interval():
    print("\n[test] F3: run_daemon retries on a short backoff after a failed cycle, not 24h")
    sleeps: list[float] = []

    def _fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 2:
            raise _StopLoop()

    reports = iter([
        {"ok": False, "detail": "OperationalError: attempt to write a readonly database"},
        {"ok": True, "detail": "fine now"},
    ])

    with patch.object(backup.time, "sleep", _fake_sleep), \
         patch.object(backup, "run_once", lambda *a, **k: next(reports)), \
         patch.object(backup, "_newest_archive_age_s", lambda *a, **k: None):
        try:
            backup.run_daemon(interval_hours=24.0)
        except _StopLoop:
            pass

    assert_eq(len(sleeps), 2, "fixture: one failed cycle, one successful cycle")
    assert_eq(sleeps[0], backup.RETRY_BACKOFF_S,
              "F3 fix: the FAILED cycle's sleep is the short capped backoff")
    assert_eq(sleeps[1], 24.0 * 3600,
              "and the SUCCESSFUL cycle after it still sleeps the full interval")


def test_retry_backoff_is_capped_at_a_short_interval():
    """CONTROL: the backoff must never make a SHORT interval (a tuned-down
    deployment, or a test) longer than it already is."""
    print("\n[test] F3 CONTROL: the retry backoff is capped at the configured interval")
    sleeps: list[float] = []

    def _fake_sleep(s):
        sleeps.append(s)
        raise _StopLoop()

    with patch.object(backup.time, "sleep", _fake_sleep), \
         patch.object(backup, "run_once", lambda *a, **k: {"ok": False, "detail": "x"}), \
         patch.object(backup, "_newest_archive_age_s", lambda *a, **k: None):
        try:
            backup.run_daemon(interval_hours=0.05)  # 180s, well under RETRY_BACKOFF_S
        except _StopLoop:
            pass
    assert_eq(sleeps[0], 180.0, "capped at the (short) interval, not the full backoff")


# ---------------------------------------------------------------------------
# A3-6b — refuse a restore while something actively writes the target
# ---------------------------------------------------------------------------

def test_restore_refuses_while_an_active_writer_holds_the_target():
    """The check deliberately stands down once a sidecar exists (see its
    docstring: opening ANY read-write connection to probe would itself
    perform SQLite's hot-journal recovery and consume the very journal
    restore_backup exists to preserve). So the window it actually covers is
    a RESERVED lock taken but not yet backed by a journal file — BEGIN
    IMMEDIATE with no write after it, which is exactly the state a
    transaction is in for its first instant. That is deliberately not
    exercised with an INSERT: a real write would create the journal file
    the check stands down for, which would test the *other* branch."""
    print("\n[test] A3-6b: restore refuses while a writer holds a RESERVED lock, no journal yet")
    scratch = Path(tempfile.mkdtemp())
    try:
        target = scratch / "webui.db"
        con = sqlite3.connect(str(target))
        con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
        con.commit()
        con.execute("BEGIN IMMEDIATE")
        journal = target.with_name(target.name + "-journal")
        assert_true(not journal.exists(),
                    "fixture: BEGIN IMMEDIATE alone creates no journal file yet")
        try:
            assert_raises(lambda: backup._require_no_active_writer(target),
                          RuntimeError, "refused while the RESERVED lock is held")
        finally:
            con.rollback()
            con.close()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_restore_stands_down_once_a_sidecar_exists_rather_than_risk_it():
    """The other half of the same story: once a sidecar exists (a real
    write is in flight, or a hot journal is sitting there from a stall),
    the check must NOT open a probing connection — doing so would consume
    the journal via SQLite's own automatic recovery before restore_backup's
    sidecar-preserving code ever saw it. This is a control on the SKIP
    itself, not on the refusal."""
    print("\n[test] A3-6b CONTROL: a sidecar present stands the check down, untouched")
    scratch = Path(tempfile.mkdtemp())
    try:
        target = scratch / "webui.db"
        con = sqlite3.connect(str(target))
        con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
        con.commit()
        con.execute("BEGIN IMMEDIATE")
        con.execute("INSERT INTO chat (body) VALUES ('mid-write')")
        journal = target.with_name(target.name + "-journal")
        assert_true(journal.is_file(), "fixture: the write created a real journal file")
        before = journal.read_bytes()
        try:
            backup._require_no_active_writer(target)  # must not raise
            print("  ok   no exception — the check stood down")
            assert_eq(journal.read_bytes(), before,
                      "and the journal was not touched by the check itself")
        finally:
            con.rollback()
            con.close()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_restore_proceeds_when_nothing_holds_the_target():
    """CONTROL: the check must not refuse the common case — a closed db
    with nothing else attached — or every ordinary restore would be
    blocked."""
    print("\n[test] A3-6b CONTROL: nothing holding the target passes the check")
    scratch = Path(tempfile.mkdtemp())
    try:
        target = scratch / "webui.db"
        con = sqlite3.connect(str(target))
        con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
        con.commit()
        con.close()
        backup._require_no_active_writer(target)  # must not raise
        print("  ok   no exception with nothing holding the target")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_a_missing_target_is_not_a_writer_lock_failure():
    """CONTROL: a fresh deployment (no local db yet) must not be refused —
    there is nothing to be locked."""
    print("\n[test] A3-6b CONTROL: a target that does not exist yet passes the check")
    backup._require_no_active_writer(Path(tempfile.mkdtemp()) / "does-not-exist.db")
    print("  ok   no exception for a missing target")


# ---------------------------------------------------------------------------
# A3-10 — set-asides land in QUARANTINE, with a sibling fallback
# ---------------------------------------------------------------------------

def test_quarantine_aside_falls_back_to_a_sibling_when_quarantine_is_unusable():
    print("\n[test] A3-10: quarantine unusable falls back to a sibling, loudly, not silently")
    scratch = Path(tempfile.mkdtemp())
    try:
        target = scratch / "webui.db-journal"
        target.write_bytes(b"a real journal's worth of bytes")

        class _BrokenQuarantine:
            def mkdir(self, *a, **k):
                raise OSError(28, "No space left on device")

        with patch.object(backup, "_quarantine_dir", lambda: _BrokenQuarantine()):
            dest = backup._quarantine_aside(target, "20260913-000000-000")
        assert_true(not target.exists(), "the original path is gone (it was moved)")
        assert_eq(dest, scratch / "webui.db-journal.pre-restore-20260913-000000-000",
                  "fell back to a sibling of the original path")
        assert_eq(dest.read_bytes(), b"a real journal's worth of bytes",
                  "and the bytes made the trip intact")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_quarantine_aside_uses_quarantine_when_it_is_usable():
    """CONTROL for the fallback test above: the primary path is really used
    when nothing is wrong with QUARANTINE."""
    print("\n[test] A3-10 CONTROL: quarantine is used when it works")
    scratch = Path(tempfile.mkdtemp())
    qdir = scratch / "quarantine"
    try:
        target = scratch / "compactor"
        target.mkdir()
        (target / "marker.txt").write_text("hi", encoding="utf-8")
        with patch.object(backup, "_quarantine_dir", lambda: qdir):
            dest = backup._quarantine_aside(target, "20260913-000000-000")
        assert_eq(dest, qdir / "compactor.pre-restore-20260913-000000-000",
                  "landed inside quarantine, not beside the original")
        assert_true((dest / "marker.txt").is_file(), "and the directory's contents made the trip")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# A3-11 — pre-restore set-asides are listable and prunable
# ---------------------------------------------------------------------------

def test_pre_restore_asides_are_listable_and_prunable():
    print("\n[test] A3-11: pre-restore set-asides in quarantine are listable and prunable")
    qdir = Path(tempfile.mkdtemp())
    try:
        old = qdir / "webui.db-journal.pre-restore-20200101-000000-000"
        old.write_bytes(b"old journal")
        os.utime(old, (time.time() - 400 * 86400, time.time() - 400 * 86400))
        new = qdir / "compactor.pre-restore-20260913-000000-000"
        new.mkdir()
        (new / "f.json").write_text("{}", encoding="utf-8")

        listed = backup.list_pre_restore_asides(qdir)
        names = {e["name"] for e in listed}
        assert_eq(names, {old.name, new.name}, "both set-asides are listed")
        assert_true(all(e["size_bytes"] >= 0 for e in listed), "sizes are reported")

        removed = backup.prune_pre_restore_asides(30, qdir)
        assert_eq(removed, [old.name], "only the entry older than 30 days was pruned")
        assert_true(not old.exists(), "the old one is gone")
        assert_true(new.exists(), "CONTROL: the recent one survives a prune")
    finally:
        shutil.rmtree(qdir, ignore_errors=True)


def test_pre_restore_asides_empty_quarantine_is_not_an_error():
    print("\n[test] A3-11 CONTROL: an empty/missing quarantine dir lists and prunes as empty")
    missing = Path(tempfile.mkdtemp()) / "does-not-exist-yet"
    assert_eq(backup.list_pre_restore_asides(missing), [], "empty list, not an exception")
    assert_eq(backup.prune_pre_restore_asides(30, missing), [], "empty prune, not an exception")


def test_live_webui_db_agrees_with_webuidb_across_every_gate_spelling():
    """v3.1.9 round 2 (fix-webuidb.md finding 3 / "two live-db resolvers").

    backup.live_webui_db() used to re-derive the WEBUI_DB_LOCAL gate a
    second, narrower way (`os.environ.get("WEBUI_DB_LOCAL", "true") ==
    "true"`) instead of asking webuidb.live_webui_db(), the one place this
    rule is actually owned - two independent readers of one rule is the
    reader-disagrees-with-writer defect this module was already fixed for
    once (A3-4/A3-9). Proves they now agree for every spelling
    entrypoint.sh's own WEBUI_DB_LOCAL normaliser recognises, and that an
    unrecognised value raises the SAME way in both rather than one silently
    guessing while the other refuses.
    """
    import webuidb
    print("\n[test] backup.live_webui_db() and webuidb.live_webui_db() agree, "
          "for every gate spelling")
    # webuidb.LOCAL_DB / .SNAPSHOT_DB are module constants read ONCE at
    # import (webuidb has almost certainly already been imported by an
    # earlier test in this file's own suite, or by backup.py's own lazy
    # `import webuidb` elsewhere) - unlike WEBUI_DB_LOCAL (the gate), which
    # both functions deliberately re-read fresh on every call. So this test
    # compares against webuidb.LOCAL_DB/webuidb.SNAPSHOT_DB AS THEY ACTUALLY
    # ARE right now, rather than trying to override WEBUI_LOCAL_DB /
    # WEBUI_SNAPSHOT_DB via the environment post-import, which webuidb.py's
    # own design does not pick up (and rightly not: those two paths are
    # static config for the life of a process; only the gate between them
    # flips at runtime).
    keys = ("COMPACTOR_BACKUP_WEBUI_DB", "DATABASE_URL", "WEBUI_DB_LOCAL")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.pop("COMPACTOR_BACKUP_WEBUI_DB", None)
        os.environ.pop("DATABASE_URL", None)

        true_spellings = ("true", "TRUE", "True", "1", "yes", "YES", "on", " true ", "")
        false_spellings = ("false", "FALSE", "False", "0", "no", "off", " false\n")
        for spelling in true_spellings:
            os.environ["WEBUI_DB_LOCAL"] = spelling
            got = backup.live_webui_db()
            assert_eq(got, webuidb.live_webui_db(),
                      f"agree on WEBUI_DB_LOCAL={spelling!r} (true-family)")
            assert_eq(got, webuidb.LOCAL_DB,
                      f"and both resolve to webuidb.LOCAL_DB for {spelling!r}")
        for spelling in false_spellings:
            os.environ["WEBUI_DB_LOCAL"] = spelling
            got = backup.live_webui_db()
            assert_eq(got, webuidb.live_webui_db(),
                      f"agree on WEBUI_DB_LOCAL={spelling!r} (false-family)")
            assert_eq(got, webuidb.SNAPSHOT_DB,
                      f"and both resolve to webuidb.SNAPSHOT_DB for {spelling!r}")

        os.environ["WEBUI_DB_LOCAL"] = "maybe"
        assert_raises(backup.live_webui_db, RuntimeError,
                      "an unrecognised gate value raises in backup.py too, "
                      "not a silent guess")
        assert_raises(webuidb.live_webui_db, RuntimeError,
                      "...the SAME way webuidb.py itself already did")

        # CONTROL: the two operator-facing overrides that ONLY backup.py
        # honors still work, unaffected by the delegation - they deliberately
        # diverge from webuidb.live_webui_db(), documented in this
        # function's own docstring, and this must not have collapsed them
        # into the gate too.
        os.environ["WEBUI_DB_LOCAL"] = "false"  # would otherwise resolve SNAPSHOT_DB
        os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(webuidb.LOCAL_DB)
        assert_eq(backup.live_webui_db(), webuidb.LOCAL_DB,
                  "CONTROL: COMPACTOR_BACKUP_WEBUI_DB still outranks the "
                  "gate")
        os.environ.pop("COMPACTOR_BACKUP_WEBUI_DB", None)
        os.environ["DATABASE_URL"] = f"sqlite:///{webuidb.LOCAL_DB}"
        assert_eq(backup.live_webui_db(), webuidb.LOCAL_DB,
                  "CONTROL: a sqlite DATABASE_URL still outranks the gate too")
        os.environ.pop("DATABASE_URL", None)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _all():
    return [
        test_real_prune_facts_eviction_does_not_regress_the_census,
        test_dedup_style_merge_does_not_regress_the_census,
        test_facts_file_emptied_still_trips_the_guard,
        test_l1_to_l2_rollup_shape_does_not_regress_the_census,
        test_l2_to_l3_rollup_archives_chapters_and_does_not_regress_the_census,
        test_summary_coverage_going_backwards_still_trips_the_guard,
        test_archived_chapters_disappearing_still_trips_the_guard,
        test_a_conversation_disappearing_entirely_still_trips_the_guard,
        test_hot_journal_falls_back_to_a_recovered_copy_instead_of_failing,
        test_a_normal_db_without_a_journal_never_uses_the_fallback,
        test_an_unrelated_operational_error_still_raises,
        test_daemon_retries_soon_after_a_failed_cycle_not_the_full_interval,
        test_retry_backoff_is_capped_at_a_short_interval,
        test_restore_refuses_while_an_active_writer_holds_the_target,
        test_restore_stands_down_once_a_sidecar_exists_rather_than_risk_it,
        test_restore_proceeds_when_nothing_holds_the_target,
        test_a_missing_target_is_not_a_writer_lock_failure,
        test_quarantine_aside_falls_back_to_a_sibling_when_quarantine_is_unusable,
        test_quarantine_aside_uses_quarantine_when_it_is_usable,
        test_pre_restore_asides_are_listable_and_prunable,
        test_pre_restore_asides_empty_quarantine_is_not_an_error,
        test_live_webui_db_agrees_with_webuidb_across_every_gate_spelling,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            t()
        print("\nAll backup v3.1.9 fix-lane tests passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
