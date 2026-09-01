"""
compactor/dbselect.py — "which database does OpenWebUI actually open?"

THE STAKES. This is the riskiest decision in the whole Postgres migration:
wrongly picking Postgres when it is empty and webui.db holds her chats means
she opens the app to an empty history (dbselect.py's module docstring has
the full incident shape). No real conversation content, personal facts, or
production conversation ids appear anywhere below — this repo is public;
every count is a synthetic integer.

Two layers are checked, deliberately:
  [1] decide() directly — the actual decision table, every row of the truth
      table in decide()'s own docstring, plus the boundary cases the table
      doesn't spell out (sqlite_chats=0 with the file present, negative-ish
      malformed input).
  [2] the CLI (`python dbselect.py --pg-tables ... --sqlite-present ...
      --sqlite-chats ...`) — because that is what entrypoint.sh actually
      calls via `eval`, and a bug in argument parsing or output formatting
      would be invisible to a test that only calls decide() in-process.

MUTATION-TESTING NOTE for whoever edits decide() next: every branch below
is written so that inverting its condition (e.g. changing `> 0` to `>= 0`,
or dropping the `sqlite_present and` guard) flips at least one check from
pass to fail. That was verified by hand when this file was written — see
the task report — not just asserted here.

    python test_dbselect.py
"""

import subprocess
import sys
from pathlib import Path

import dbselect

FAILED = []
_HERE = Path(__file__).parent
_PY = sys.executable


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}")


def _run_cli(pg_tables, sqlite_present, sqlite_chats):
    r = subprocess.run(
        [_PY, str(_HERE / "dbselect.py"),
         "--pg-tables", str(pg_tables),
         "--sqlite-present", "true" if sqlite_present else "false",
         "--sqlite-chats", str(sqlite_chats)],
        capture_output=True, text=True, timeout=20,
    )
    out = {}
    for line in r.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return r.returncode, out


print()
print("[1] decide(): Postgres empty, SQLite has chats -> SQLite, sync on, migration pending")
d = dbselect.decide(pg_tables=0, sqlite_present=True, sqlite_chats=42)
check(d.database == "sqlite", f"selects sqlite (got {d.database})")
check(d.sync_enabled is True, "sync is enabled (her live chats are on ephemeral local disk)")
check(d.migration_pending is True, "migration_pending is True")
check(d.unknown_chat_count is False, "not flagged as unknown — this count IS known")

print()
print("[2] decide(): Postgres has tables -> Postgres, sync off, no migration flag")
d = dbselect.decide(pg_tables=5, sqlite_present=True, sqlite_chats=42)
check(d.database == "postgres", f"selects postgres (got {d.database})")
check(d.sync_enabled is False, "sync is disabled")
check(d.migration_pending is False, "migration_pending is False")
d2 = dbselect.decide(pg_tables=5, sqlite_present=False, sqlite_chats=None)
check(d2.database == "postgres", "postgres wins regardless of sqlite's state once it has tables")

print()
print("[3] decide(): both genuinely empty (fresh deploy) -> Postgres, no false alarm")
d = dbselect.decide(pg_tables=0, sqlite_present=True, sqlite_chats=0)
check(d.database == "postgres", f"selects postgres (got {d.database})")
check(d.migration_pending is False, "no migration-pending false alarm on a real 0")
check(d.unknown_chat_count is False, "not flagged unknown — 0 is a KNOWN count")
check(d.sync_enabled is False, "sync stays off — nothing to snapshot on a genuinely empty sqlite")

print()
print("[4] decide(): SQLite file absent entirely -> Postgres, no crash")
d = dbselect.decide(pg_tables=0, sqlite_present=False, sqlite_chats=None)
check(d.database == "postgres", f"selects postgres (got {d.database})")
check(d.migration_pending is False, "no migration flag when there is no file to migrate from")
check(d.unknown_chat_count is False, "absent file is a KNOWN empty, not unknown")
check(d.sync_enabled is False, "sync stays off — there is no local file at all to snapshot")

print()
print("[5] decide(): chat-count probe failing/unreadable -> FAILS SAFE (never switches to Postgres)")
d = dbselect.decide(pg_tables=0, sqlite_present=True, sqlite_chats=None)
check(d.database == "sqlite", f"stays on sqlite rather than guessing (got {d.database})")
check(d.unknown_chat_count is True, "flagged as unknown so entrypoint.sh can warn distinctly")
check(d.sync_enabled is True, "sync stays on — if she does have chats, they still need a durable copy")
check(d.migration_pending is False, "this is NOT the ordinary migration-pending path — it's the unknown-count path")

print()
print("[6] decide(): unknown sqlite_chats is NOT silently treated as 0 even at the boundary")
# The dangerous bug this whole module exists to prevent, spelled out
# explicitly: unknown must behave differently from a real 0 when
# postgres is empty, even though both are represented near "falsy".
unknown = dbselect.decide(pg_tables=0, sqlite_present=True, sqlite_chats=None)
known_zero = dbselect.decide(pg_tables=0, sqlite_present=True, sqlite_chats=0)
check(unknown.database != known_zero.database or unknown.unknown_chat_count != known_zero.unknown_chat_count,
      "unknown and known-zero are NOT collapsed into identical outcomes")
check(unknown.database == "sqlite" and known_zero.database == "postgres",
      f"unknown -> sqlite (safe), known-zero -> postgres (correct) (got {unknown.database}, {known_zero.database})")

print()
print("[7] the CLI wires decide() through argv/stdout correctly (this is what entrypoint.sh actually calls)")
rc, out = _run_cli(0, True, 42)
check(rc == 0, "exits 0")
check(out.get("DBSELECT_DATABASE") == "sqlite", f"CLI: migration-pending case selects sqlite (got {out})")
check(out.get("DBSELECT_MIGRATION_PENDING") == "true", f"CLI: migration_pending=true (got {out})")

rc, out = _run_cli(3, True, 42)
check(out.get("DBSELECT_DATABASE") == "postgres", f"CLI: postgres-has-tables case selects postgres (got {out})")
check(out.get("DBSELECT_SYNC_ENABLED") == "false", f"CLI: sync disabled (got {out})")

rc, out = _run_cli(0, False, "unknown")
check(out.get("DBSELECT_DATABASE") == "postgres", f"CLI: fresh deploy (no sqlite file) selects postgres (got {out})")
check(out.get("DBSELECT_SYNC_ENABLED") == "false", f"CLI: fresh deploy leaves sync off (got {out})")

rc, out = _run_cli(0, True, "unknown")
check(out.get("DBSELECT_DATABASE") == "sqlite", f"CLI: unreadable probe fails safe to sqlite (got {out})")
check(out.get("DBSELECT_UNKNOWN_CHAT_COUNT") == "true", f"CLI: flagged unknown (got {out})")

print()
print("[8] the CLI's own fail-safe: an unparseable --sqlite-chats degrades like 'unknown', not like 0")
rc, out = _run_cli(0, True, "not-a-number")
check(out.get("DBSELECT_DATABASE") == "sqlite", f"garbage input fails safe, same as an honest 'unknown' (got {out})")
check(out.get("DBSELECT_UNKNOWN_CHAT_COUNT") == "true", f"garbage input is flagged unknown, not silently read as 0 (got {out})")

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All dbselect tests passed.")
