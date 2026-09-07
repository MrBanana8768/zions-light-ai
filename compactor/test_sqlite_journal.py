"""
A HOT SQLite rollback journal must degrade /health/full (v3.1.8).

WHY. Twice now an uncommitted transaction has been left beside OpenWebUI's
database on the MooseFS volume, and both times nothing said so:

  * 2026-08-31: the volume dropped I/O mid-transaction. SQLite tried to roll
    the journal back on every subsequent open; rolling back requires WRITING;
    the write failed; OpenWebUI answered "attempt to write a readonly
    database" for 24 minutes across 1,819 failed queries. The database was
    never corrupt - it was stuck mid-recovery on a filesystem that would not
    let it finish.
  * 2026-09-07: an ORPHANED hot journal sat beside a database that was being
    written to perfectly normally. Health said "ok", storage said writable,
    the memory counters were climbing. Only the journal's 8-byte header said
    otherwise.

THE HEADER IS THE SIGNAL, NOT THE FILE. In `delete` mode a journal is created
and removed around every transaction, so catching one mid-write is ordinary.
In `persist` mode one is deliberately left behind with a zeroed header. Only
the magic separates "holds an uncommitted transaction" from "is debris".

AND NOT BY OPENING THE DATABASE. A second connection cannot tell a hot
journal from a live writer's: it must take a write lock to find out,
OpenWebUI holds that lock, and the probe fails with the same "readonly
database" text either way. That ambiguity cost real time on 2026-09-07, and
it is why this reads bytes instead.

    python test_sqlite_journal.py
"""

import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-journal-")

_TMP = tempfile.mkdtemp(prefix="webuidb-")
_DB = os.path.join(_TMP, "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = _DB

import health  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


MAGIC = bytes.fromhex("d9d505f920a163d7")
JOURNAL = _DB + "-journal"

print("[1] no journal at all is the ordinary case")
if os.path.exists(JOURNAL):
    os.remove(JOURNAL)
r = health.probe_sqlite_journal()
check(r["hot"] is False and r["ok"] is True,
      f"absent journal is not hot (got {r})")

print("[2] a ZEROED journal is debris, not a pending transaction")
with open(JOURNAL, "wb") as fh:
    fh.write(bytes(28))
r = health.probe_sqlite_journal()
check(r["hot"] is False and r["ok"] is True,
      "a persist-mode leftover with a zeroed header must NOT degrade health - "
      "firing on the file's existence would cry wolf on every commit")

print("[3] the MAGIC header is a hot journal")
with open(JOURNAL, "wb") as fh:
    fh.write(MAGIC + bytes(20))
r = health.probe_sqlite_journal()
check(r["hot"] is True and r["ok"] is False,
      f"the rollback-journal magic is reported hot (got {r.get('header')})")

print("[4] and it degrades /health/full, with the recovery in the reason")
# Through the REAL assembler, not a reconstruction of it. gather_health_full
# is what /health/full and /admin/selftest both call, so anything asserted
# here is asserted about the endpoint. The journal written in [3] is still
# hot on disk at this point.
import asyncio  # noqa: E402

body = asyncio.run(health.gather_health_full("http://127.0.0.1:9", 4096))
reasons = body.get("status_reasons") or []
check(body.get("status") == "degraded",
      f"status goes degraded (got {body.get('status')!r})")
check(any("HOT SQLite rollback journal" in r for r in reasons),
      "and a reason names the hot journal")
check(any("Do NOT delete the journal" in r for r in reasons),
      "and warns against the one action that turns this into corruption")
check(((body.get("checks") or {}).get("sqlite_journal") or {}).get("hot") is True,
      "and the probe result is published under checks, beside vllm/storage/"
      "tokenize, because that is what it is")

# The control: with the journal gone, the hot-journal reason must NOT be
# there. vLLM is unreachable in this test, so the endpoint is degraded
# either way - without this the case above would pass on any degraded body.
os.remove(JOURNAL)
body2 = asyncio.run(health.gather_health_full("http://127.0.0.1:9", 4096))
check(not any("HOT SQLite rollback journal" in r
              for r in (body2.get("status_reasons") or [])),
      "and the reason disappears once the journal does - so [4] is not "
      "passing merely because something else degraded it")
with open(JOURNAL, "wb") as fh:
    fh.write(MAGIC + bytes(20))

print("[5] an unreadable journal reports that it could not answer")
os.remove(JOURNAL)
os.mkdir(JOURNAL)          # a directory where a file belongs
r = health.probe_sqlite_journal()
check(r["hot"] is None and r["ok"] is None and "error" in r,
      "a probe that cannot read must say so rather than claim healthy")
os.rmdir(JOURNAL)

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll sqlite-journal checks passed.")
