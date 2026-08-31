#!/usr/bin/env python3
"""Copy OpenWebUI's chat history from SQLite into PostgreSQL.

    # look, change nothing (the default):
    /app/venv/bin/python /data/scripts/migrate-webui-sqlite-to-pg.py

    # do it:
    /app/venv/bin/python /data/scripts/migrate-webui-sqlite-to-pg.py --apply

Run it AFTER OpenWebUI has started once against the Postgres database, so
alembic has created the schema. This script never creates or alters a table:
translating OpenWebUI's schema by hand would be a second, worse copy of its
migrations, and it would rot. It copies ROWS into a schema OpenWebUI built.

WHY THIS EXISTS. RunPod's MooseFS mount drops I/O; SQLite responds by leaving
a hot rollback journal it cannot roll back, and the front end dies with
"attempt to write a readonly database". That happened twice on 2026-08-31.
Postgres on local disk removes the failure class. This moves her history
across.

THE SAFETY RULES, because this is the step that can lose everything:

  * The SQLite database is opened READ-ONLY and never written. If the result
    is wrong, the source is untouched and you simply try again.
  * Dry run by default. --apply is required to write anything.
  * Refuses a destination that already holds rows, unless --force. Merging
    into a live database silently duplicates or collides on primary keys.
  * Columns are matched BY NAME, never by position. A schema that gained a
    column in some version must not shift every value one place left.
  * Verifies row counts per table afterwards and reports any mismatch as a
    failure, loudly, rather than printing "done".
"""

import argparse
import gzip
import json
import os
import sqlite3
import sys
from pathlib import Path

try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError:
    print("FAIL: psycopg not importable. Run this with /app/venv/bin/python,")
    print("      which is where OpenWebUI's own Postgres driver lives.")
    sys.exit(2)

SQLITE_DB = os.environ.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db")
PG_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://openwebui@/openwebui?host=/var/run/postgresql"
)
# Tables OpenWebUI keeps but which must not be copied: alembic's own bookkeeping
# describes the DESTINATION's schema version. Overwriting it with the source's
# would tell alembic the database is at a migration it has not run.
SKIP_TABLES = {"alembic_version", "migratehistory"}


def say(msg=""):
    print(msg, flush=True)


def sqlite_tables(con) -> list[str]:
    rows = con.execute(
        "select name from sqlite_master where type='table' "
        "and name not like 'sqlite_%'"
    ).fetchall()
    return sorted(r[0] for r in rows)


def sqlite_columns(con, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def pg_columns(cur, table: str) -> dict[str, str]:
    cur.execute(
        "select column_name, data_type from information_schema.columns "
        "where table_schema='public' and table_name=%s",
        (table,),
    )
    return {name: dtype for name, dtype in cur.fetchall()}


def coerce(value, pg_type: str):
    """SQLite is dynamically typed; Postgres is not. Convert per DESTINATION
    column type — the source cannot be trusted to say what it holds."""
    if value is None:
        return None
    if pg_type == "boolean":
        # SQLite stores booleans as 0/1 (and occasionally as '0'/'1').
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "t", "yes")
        return bool(value)
    if pg_type in ("json", "jsonb"):
        # The chat bodies live here. SQLite holds them as TEXT; Postgres will
        # refuse text in a json column, so parse and hand psycopg a real
        # object. A value that is not valid JSON is wrapped rather than
        # dropped — losing a malformed chat body silently would be worse
        # than storing it as a JSON string.
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", "replace")
        if isinstance(value, str):
            try:
                return Jsonb(json.loads(value))
            except Exception:
                return Jsonb(value)
        return Jsonb(value)
    if pg_type in ("bytea",) and isinstance(value, str):
        return value.encode("utf-8", "replace")
    if pg_type in ("text", "character varying", "character") and isinstance(
        value, (bytes, bytearray)
    ):
        return value.decode("utf-8", "replace")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the destination already holds rows")
    ap.add_argument("--sqlite", default=SQLITE_DB)
    ap.add_argument("--dsn", default=PG_DSN)
    args = ap.parse_args()

    src_path = Path(args.sqlite)
    if not src_path.exists():
        say(f"FAIL: {src_path} does not exist")
        return 2

    say("=" * 68)
    say(f"{'MIGRATE' if args.apply else 'DRY RUN'}: {src_path} -> Postgres")
    say("=" * 68)

    # READ-ONLY. immutable=1 also stops SQLite trying to recover a hot
    # journal, which on a stalled mount is what fails in the first place.
    con = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    try:
        pg = psycopg.connect(args.dsn, autocommit=False)
    except Exception as e:
        say(f"FAIL: cannot connect to Postgres ({type(e).__name__}: {e})")
        say(f"      DSN: {args.dsn}")
        say("      Has OpenWebUI started once against Postgres to build the schema?")
        return 3

    copied: dict[str, int] = {}
    skipped: list[str] = []
    problems: list[str] = []

    with pg.cursor() as cur:
        src_tables = [t for t in sqlite_tables(con) if t not in SKIP_TABLES]
        say(f"source tables: {len(src_tables)}")

        # Pre-flight: is the destination already populated?
        occupied = []
        for t in src_tables:
            cols = pg_columns(cur, t)
            if not cols:
                continue
            cur.execute(f'select count(*) from "{t}"')
            n = cur.fetchone()[0]
            if n:
                occupied.append(f"{t}={n}")
        if occupied and not args.force:
            say("")
            say("FAIL: the destination already holds rows: " + ", ".join(occupied[:6]))
            say("      Migrating into a populated database duplicates rows or")
            say("      collides on primary keys. Start from an empty schema, or")
            say("      pass --force if you genuinely mean to merge.")
            return 4

        if args.apply:
            # Foreign keys are satisfied only once EVERY table is loaded, and
            # loading in dependency order would mean hard-coding OpenWebUI's
            # schema here. Deferring is simpler and cannot silently drop rows.
            cur.execute("set session_replication_role = replica")

        for table in src_tables:
            dst_cols = pg_columns(cur, table)
            if not dst_cols:
                skipped.append(f"{table} (absent in Postgres)")
                continue
            src_cols = sqlite_columns(con, table)
            shared = [c for c in src_cols if c in dst_cols]
            missing = [c for c in src_cols if c not in dst_cols]
            if missing:
                # Reported, not silently dropped: a column the destination
                # does not have is data that will not make the trip.
                problems.append(
                    f"{table}: source columns absent in Postgres, NOT copied: "
                    f"{', '.join(missing)}"
                )
            if not shared:
                skipped.append(f"{table} (no columns in common)")
                continue

            rows = con.execute(f'select * from "{table}"').fetchall()
            if not rows:
                copied[table] = 0
                continue

            collist = ", ".join(f'"{c}"' for c in shared)
            params = ", ".join(["%s"] * len(shared))
            sql = f'insert into "{table}" ({collist}) values ({params})'
            payload = [
                tuple(coerce(r[c], dst_cols[c]) for c in shared) for r in rows
            ]
            if args.apply:
                try:
                    cur.executemany(sql, payload)
                except Exception as e:
                    pg.rollback()
                    say("")
                    say(f"FAIL while copying {table}: {type(e).__name__}: {e}")
                    say("      NOTHING was committed. The SQLite source is untouched.")
                    return 5
            copied[table] = len(rows)

        if args.apply:
            cur.execute("set session_replication_role = default")
            # Identity columns keep their own counters. Without this the next
            # insert OpenWebUI makes collides with a row we just imported.
            for table in copied:
                for col in pg_columns(cur, table):
                    cur.execute(
                        "select pg_get_serial_sequence(%s, %s)", (table, col)
                    )
                    seq = cur.fetchone()[0]
                    if seq:
                        cur.execute(
                            f'select setval(%s, coalesce((select max("{col}") '
                            f'from "{table}"), 1))',
                            (seq,),
                        )
            pg.commit()

    # ---- report, and verify ------------------------------------------------
    say("")
    for t in sorted(copied):
        say(f"  {t:32} {copied[t]:>7} row(s)")
    for s in skipped:
        say(f"  SKIPPED {s}")
    for p in problems:
        say(f"  WARNING {p}")

    if not args.apply:
        say("")
        say(f"DRY RUN — nothing written. {sum(copied.values())} row(s) would move.")
        say("Re-run with --apply to migrate.")
        return 0

    say("")
    say("verifying row counts...")
    mismatches = []
    with pg.cursor() as cur:
        for t, n in copied.items():
            cur.execute(f'select count(*) from "{t}"')
            got = cur.fetchone()[0]
            if got != n:
                mismatches.append(f"{t}: copied {n}, destination has {got}")
    con.close()
    pg.close()

    if mismatches:
        say("")
        say("FAIL: row counts do not match after migration:")
        for m in mismatches:
            say(f"  {m}")
        say("The SQLite source is untouched — investigate before switching over.")
        return 6

    say(f"  all {len(copied)} table(s) match")
    say("")
    say("=" * 68)
    say(f"MIGRATED — {sum(copied.values())} row(s). SQLite source untouched at")
    say(f"{src_path} — keep it until you are satisfied.")
    say("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
