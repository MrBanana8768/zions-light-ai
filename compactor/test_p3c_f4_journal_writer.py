"""
Hostile pass 3 (reviewer C), F4: the HOT-journal health reason must not fire
while a live writer holds the transaction open — only when the journal is
genuinely stalled debris left behind by a dead/interrupted writer.

WHY THIS IS A SEPARATE FILE, NOT AN ADDITION TO test_sqlite_journal.py.
test_sqlite_journal.py's fixtures write the magic header directly into a
journal FILE with no real SQLite writer anywhere in the picture — that is
the only shape it tests, and it is also exactly the shape that cannot tell
this finding's fix from the pre-fix code: a magic header with no writer
process is "hot" either way (no RESERVED lock to see). Proving the fix needs
a REAL SQLite write transaction, held open by a SEPARATE PROCESS (as
OpenWebUI is), so a real fcntl() RESERVED lock exists for
backup._probe_reserved_lock to find. That needs multiprocessing with real
file I/O, which is slower and heavier than the rest of this module's tests,
hence its own file per the brief's guidance to keep new coverage isolated
and named for what it proves.

REPRODUCED AGAINST UNFIXED CODE (reviewer C, SP\\p3-c\\p3c_journal_live.py,
journal1.log): a 33MB row UPDATE inside an open transaction, writer alive,
reported health.probe_sqlite_journal hot=True and degraded /health/full with
"Stop openwebui and open the database once read-write" while the writer was
mid-write, not stalled. 71.9% of polls during a run of ordinary updates of
that row read as hot.

PLATFORM: this needs POSIX fcntl byte-range locks (backup._probe_reserved_
lock's own mechanism) and `os.fork` (via multiprocessing). Production is
Linux; SKIPS cleanly (exit 0) on any platform without `fcntl`, per
_probe_reserved_lock's own documented fail-open direction for that case —
see health._check_one_journal's F4 comment for why that direction is safe
(the pre-fix magic-only behaviour is unchanged there, not a new gap).

    python test_p3c_f4_journal_writer.py
"""

import os
import sys
import tempfile

try:
    import fcntl  # noqa: F401  (existence check only)
except ImportError:
    print("SKIPPED: test_p3c_f4_journal_writer.py — no POSIX fcntl on this "
          "platform (Windows dev box). backup._probe_reserved_lock fails "
          "open here by design (see its docstring), which means this file's "
          "whole point — proving a live writer is NOT reported hot — cannot "
          "be exercised. Production is Linux; run in the unit-tests "
          "container for a real result.")
    sys.exit(0)

import multiprocessing as mp
import sqlite3
import time
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="p3c-f4-journal-")
DB = os.path.join(TMP, "webui.db")
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = os.path.join(TMP, "store")
os.environ["WEBUI_DB_LOCAL"] = "false"
os.environ["WEBUI_SNAPSHOT_DB"] = DB
os.environ["DATABASE_URL"] = "sqlite:///" + DB
os.environ["COMPACTOR_BACKUP_ENABLED"] = "false"

import health  # noqa: E402
import backup  # noqa: E402

ROW_MB = 3  # small enough to spill the default page cache, big enough to be fast
FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def _setup_db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE chat (id TEXT PRIMARY KEY, chat TEXT)")
    c.execute("INSERT INTO chat VALUES ('c1', ?)", ("x" * (ROW_MB << 20),))
    c.commit()
    c.close()


def _held_writer(ready: "mp.synchronize.Event", release: "mp.synchronize.Event") -> None:
    """Open a write transaction on the big row and hold it, like OpenWebUI's
    chat-row UPDATE mid-flight. Big enough to spill the page cache and write
    the journal's magic header before COMMIT."""
    c = sqlite3.connect(DB, isolation_level=None)
    c.execute("BEGIN")
    c.execute("UPDATE chat SET chat=? WHERE id='c1'", ("y" * (ROW_MB << 20),))
    ready.set()
    release.wait(30)
    c.execute("COMMIT")
    c.close()


if __name__ == "__main__":
    mp.set_start_method("fork")
    _setup_db()
    print(f"python sqlite {sqlite3.sqlite_version}; row {ROW_MB}MB; db {DB}")

    journal_path = DB + "-journal"

    print("[1] CONTROL: magic header present, no writer holding the lock -> "
          "still hot (reproduces test_sqlite_journal.py's own case with the "
          "real backup-probe wired in, so this file's fix does not turn "
          "into a guard that refuses to ever fire)")
    with open(journal_path, "wb") as fh:
        fh.write(bytes.fromhex("d9d505f920a163d7") + bytes(20))
    r = health.probe_sqlite_journal()
    check(r.get("hot") is True, f"no live writer: still reported hot (got {r})")
    os.remove(journal_path)

    print("\n[2] a real write transaction, held open by a SEPARATE PROCESS, "
          "with the journal's magic header on disk -> must NOT be hot")
    ready, release = mp.Event(), mp.Event()
    p = mp.Process(target=_held_writer, args=(ready, release))
    p.start()
    ready.wait(60)
    time.sleep(0.2)  # let the write actually land and the journal sync
    try:
        check(os.path.exists(journal_path), "journal exists while the transaction is open")
        res = backup._probe_reserved_lock(Path(DB))
        check(res is True, f"CONTROL: backup._probe_reserved_lock itself sees the writer (got {res})")
        j = health.probe_sqlite_journal()
        check(j.get("hot") is False,
              f"probe_sqlite_journal does NOT report hot while the writer holds "
              f"RESERVED (got hot={j.get('hot')}, header={j.get('header')}) — "
              f"before this fix this was True (reproduced in journal1.log)")
        check(j.get("writer_active") is True,
              f"and says why: writer_active=True (got {j.get('writer_active')})")

        import asyncio
        body = asyncio.run(health.gather_health_full("http://127.0.0.1:9", 4096))
        reasons = body.get("status_reasons") or []
        check(not any("HOT SQLite rollback journal" in r for r in reasons),
              "and /health/full carries no hot-journal reason during the live write "
              "(vLLM is unreachable in this test so OTHER reasons may still degrade "
              "it — this only asserts the journal one specifically is absent)")
    finally:
        release.set()
        p.join(60)

    print("\n[3] after COMMIT, the journal is gone and the probe is clean")
    r = health.probe_sqlite_journal()
    check(r.get("hot") is False and not os.path.exists(journal_path),
          f"clean after commit (got {r})")

    print("\n[4] CONTROL: a genuinely abandoned hot journal (writer process "
          "gone, magic header left behind) is still caught")
    with open(journal_path, "wb") as fh:
        fh.write(bytes.fromhex("d9d505f920a163d7") + bytes(20))
    r = health.probe_sqlite_journal()
    check(r.get("hot") is True,
          f"an orphaned hot journal with no writer anywhere is still reported hot "
          f"(got {r}) — this is the actual 2026-09-07 incident shape, and the "
          f"whole reason this rule exists at all")
    os.remove(journal_path)

    if FAILED:
        print(f"\n{len(FAILED)} check(s) FAILED")
        sys.exit(1)
    print("\nAll F4 (hostile pass 3) journal-writer checks passed.")
