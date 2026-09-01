"""
compactor.dbselect — "which database does OpenWebUI actually open?"

Extracted out of entrypoint.sh so the riskiest decision in the Postgres
migration is unit-testable. entrypoint.sh is bash; this repo's test suite is
plain Python scripts run with the real interpreter (see test_pgarchive.py).
Re-implementing the branching in a bash test harness would mean two copies
of the same logic that can silently drift apart — this branch's most
frequent defect — so the decision lives in exactly one place (`decide()`
below) and entrypoint.sh calls this file's CLI to get an answer instead of
branching on PG_TABLES/SQLITE_CHATS itself.

THE STAKES. Wrongly picking Postgres when Postgres is empty and webui.db
holds her chats means she opens the app to an empty history. The data is
still on disk and recoverable (scripts/migrate-webui-sqlite-to-pg.py), but
"your companion has forgotten you, ask an engineer" is not an acceptable
deploy outcome. So every branch below is written to fail toward the
database that cannot look empty when it might not be — see
DECIDE_UNKNOWN_CHATS_FAILS_SAFE below in particular.

THREE INPUTS, each already a judgment call made by entrypoint.sh's probes,
not by this module:
  pg_tables      — count of public tables in Postgres. entrypoint.sh's own
                   probe already folds "couldn't query" into 0 (Postgres
                   unreachable can't serve her chats either way, so
                   preferring sqlite in that case is always safe — see the
                   pg_tables<=0 branches below).
  sqlite_present — whether webui.db exists on disk at all. Distinguishes a
                   genuinely fresh deploy (no file -> sqlite_chats is a
                   KNOWN 0) from a file that exists but couldn't be read
                   (sqlite_chats is UNKNOWN) — those must not be treated
                   the same.
  sqlite_chats   — webuidb._has_rows()'s chat count, or None. None only
                   means anything when sqlite_present is True; combined
                   with sqlite_present=False it is the known-empty case.

CLI usage (entrypoint.sh evals this):
    python dbselect.py --pg-tables 0 --sqlite-present true --sqlite-chats 12
prints shell-sourceable KEY=VALUE lines:
    DBSELECT_DATABASE=sqlite
    DBSELECT_SYNC_ENABLED=true
    DBSELECT_MIGRATION_PENDING=true
    DBSELECT_UNKNOWN_CHAT_COUNT=false
No human-readable text is emitted here on purpose — entrypoint.sh owns the
wording of what gets printed to the boot log; this module owns only the
decision.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    database: str  # "postgres" or "sqlite"
    sync_enabled: bool  # webuidb-sync program's autostart (supervisord.conf)
    migration_pending: bool  # a human still needs to run the migration script
    unknown_chat_count: bool  # the sqlite probe failed; see decide()'s docstring


def decide(pg_tables: int, sqlite_present: bool, sqlite_chats: int | None) -> Decision:
    """The whole decision, in one place, in one function signature small
    enough to read as a truth table:

        pg_tables   sqlite_present   sqlite_chats   -> database
        --------------------------------------------------------
        > 0         (irrelevant)     (irrelevant)   -> postgres  [1]
        0           True             None           -> sqlite    [2] UNKNOWN, fail safe
        0           True             > 0             -> sqlite    [3] migration pending
        0           True             0               -> postgres  [4] both genuinely empty
        0           False            (irrelevant)    -> postgres  [4] both genuinely empty
    """
    if pg_tables > 0:
        # [1] A live database with real content always wins outright — same
        # rule pgarchive.restore_if_needed() already applies ("a healthy
        # database wins outright"). Whatever sqlite holds at this point is
        # either already migrated or moot; re-deriving a different answer
        # here would contradict that Postgres already has her history.
        return Decision(database="postgres", sync_enabled=False,
                         migration_pending=False, unknown_chat_count=False)

    if sqlite_present and sqlite_chats is None:
        # [2] THE FAIL-SAFE CASE. The sqlite file exists but the chat-count
        # probe could not read it (locked mid-write, a corrupt header, the
        # process crashing before it could print anything). We do NOT know
        # whether it holds her chats. Collapsing "unknown" to "0, so switch
        # to the empty Postgres" is exactly the bug this module exists to
        # prevent — a transient read failure would silently open her to an
        # empty app. Keep serving sqlite, which is already working, until a
        # human looks at why the probe failed.
        return Decision(database="sqlite", sync_enabled=True,
                         migration_pending=False, unknown_chat_count=True)

    if sqlite_present and (sqlite_chats or 0) > 0:
        # [3] The exact case the whole module exists for: Postgres is up
        # but empty, sqlite genuinely has her chats. Keep serving sqlite —
        # the path that already works — and let entrypoint.sh print the
        # manual migration steps.
        return Decision(database="sqlite", sync_enabled=True,
                         migration_pending=True, unknown_chat_count=False)

    # [4] Both sides are genuinely, knowably empty (fresh deploy, or sqlite
    # absent entirely) — no history to lose either way. Postgres is the
    # strategic destination, so start on it and raise no false alarm.
    return Decision(database="postgres", sync_enabled=False,
                     migration_pending=False, unknown_chat_count=False)


def _parse_sqlite_chats(raw: str) -> int | None:
    if raw.strip().lower() in ("unknown", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        # Same fail-safe rule as an actual probe failure: an unparseable
        # value must not be silently read as "0 chats, safe to switch".
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pg-tables", required=True)
    p.add_argument("--sqlite-present", required=True, choices=["true", "false"])
    p.add_argument("--sqlite-chats", required=True,
                    help='an integer, or "unknown" if the probe failed/was not run')
    args = p.parse_args(argv)

    try:
        pg_tables = int(args.pg_tables)
    except ValueError:
        # An unparseable pg_tables reading is the same "couldn't determine"
        # shape as pgarchive._table_count() returning None — entrypoint.sh's
        # own probe already collapses that to "0" before calling us (see
        # module docstring), but degrade the same way here defensively
        # rather than crash the boot script over a malformed argument.
        pg_tables = 0

    decision = decide(
        pg_tables=pg_tables,
        sqlite_present=(args.sqlite_present == "true"),
        sqlite_chats=_parse_sqlite_chats(args.sqlite_chats),
    )

    print(f"DBSELECT_DATABASE={decision.database}")
    print(f"DBSELECT_SYNC_ENABLED={'true' if decision.sync_enabled else 'false'}")
    print(f"DBSELECT_MIGRATION_PENDING={'true' if decision.migration_pending else 'false'}")
    print(f"DBSELECT_UNKNOWN_CHAT_COUNT={'true' if decision.unknown_chat_count else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
