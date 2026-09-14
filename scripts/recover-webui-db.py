#!/usr/bin/env python3
"""Recover OpenWebUI's webui.db after a MooseFS I/O stall. Run it on the pod.

    /opt/compactor-venv/bin/python /data/scripts/recover-webui-db.py
    /opt/compactor-venv/bin/python /data/scripts/recover-webui-db.py --check

WHAT THIS FIXES, precisely. RunPod's MooseFS volume drops I/O occasionally.
When that lands while OpenWebUI is mid-transaction, SQLite leaves a hot
rollback journal next to the database. Every subsequent open tries to roll
that journal back, rolling back requires WRITING, the write fails on the
sick volume, and SQLite reports:

    sqlite3.OperationalError: attempt to write a readonly database
    sqlite3.OperationalError: disk I/O error      (even on plain SELECTs)

The database is NOT corrupt. It is stuck mid-recovery on a filesystem that
will not let it finish, and "readonly" is SQLite protecting the file rather
than failing. Observed twice on 2026-08-31 (02:17 and ~04:30); both times
the database came back with integrity_check = ok and nothing lost.

THE ONE THING THAT MATTERS: a journal and its database are a MATCHED PAIR.

  * Deleting a hot journal turns a recoverable database into a corrupt one -
    the journal is what the rollback needs.
  * Leaving a stale journal beside a REPLACED database corrupts that, because
    SQLite will roll back changes that belong to a different file.

So this script copies BOTH together, lets SQLite finish the rollback on
local disk where writes work, verifies, and swaps both originals out under
timestamped names. It never deletes anything.

SAFETY. Refuses to swap unless integrity_check says ok. The originals are
renamed, never removed, and /tmp/rescue keeps the verified copy. If anything
looks wrong afterwards, everything needed to go back is still on disk.
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DB = Path(os.environ.get("WEBUI_DB", "/data/openwebui/webui.db"))
RESCUE = Path("/tmp/rescue")
FORENSICS = Path(os.environ.get("WEBUI_DB_FORENSICS", "/data/forensics"))
# Journals SQLite may leave beside the database. -journal is the rollback
# journal (the one seen in production); -wal/-shm appear in WAL mode.
SIDECARS = ("-journal", "-wal", "-shm")
# The SAME check RUNBOOK_DB_JOURNAL.md route 2B tells an operator to run by
# hand after starting OpenWebUI alone ("Wait until this prints 200 ... a
# large database can take a few minutes"). r318-b F5: this script's own
# route (2A) started OpenWebUI and stopped there - never waiting for it to
# actually answer, and never saying a word about compactor/backup (see
# `main`'s own comment at [5/5] for why THIS script cannot restart those
# two itself).
OPENWEBUI_CONFIG_URL = os.environ.get(
    "OPENWEBUI_CONFIG_URL", "http://127.0.0.1:3000/api/config"
)


def say(msg: str) -> None:
    print(msg, flush=True)


def supervisor(action: str, program: str = "openwebui") -> bool:
    try:
        r = subprocess.run(
            ["supervisorctl", action, program],
            capture_output=True, text=True, timeout=120,
        )
        say(f"    supervisorctl {action} {program}: {r.stdout.strip() or r.stderr.strip()}")
        return r.returncode == 0
    except Exception as e:
        say(f"    supervisorctl {action} failed: {type(e).__name__}: {e}")
        return False


def wait_for_openwebui(timeout_s: float = 180, poll_s: float = 5) -> bool:
    """Poll OpenWebUI's own health endpoint until it answers 200, or give
    up after `timeout_s`. The SAME check the runbook's route 2B tells an
    operator to re-run by hand every 30 seconds - bounded here so this
    script cannot hang forever if OpenWebUI never comes up (a large
    database can take a few minutes to open; a script that never returns
    on a genuinely stuck boot is its own incident)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(OPENWEBUI_CONFIG_URL, timeout=10) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass  # not answering yet, or answering something other than 200
        time.sleep(poll_s)
    return False


def inspect(path: Path) -> dict:
    """Open, letting SQLite replay/roll back any journal, then report."""
    out = {"ok": False, "integrity": "?", "chats": None, "users": None}
    try:
        con = sqlite3.connect(str(path))
        out["integrity"] = con.execute("PRAGMA integrity_check").fetchone()[0]
        for table, key in (("chat", "chats"), ("user", "users")):
            try:
                out[key] = con.execute(f"select count(*) from {table}").fetchone()[0]
            except Exception:
                out[key] = None
        con.close()
        out["ok"] = out["integrity"] == "ok"
    except Exception as e:
        out["integrity"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="diagnose only: never stops OpenWebUI, never writes")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    args = ap.parse_args()

    if not DB.exists():
        say(f"[FAIL] {DB} does not exist")
        return 2

    say("=" * 62)
    say(f"webui.db recovery — {DB}")
    say("=" * 62)
    present = [s for s in SIDECARS if DB.with_name(DB.name + s).exists()]
    size_mb = DB.stat().st_size / 1e6
    say(f"  database : {size_mb:.1f} MB")
    say(f"  sidecars : {', '.join(present) if present else 'none'}")
    if present:
        say("             ^ a hot journal is the signature of this failure")

    # ---- diagnose on a COPY, never the live file -------------------------
    if args.check:
        say("")
        say("[check] copying to /tmp to inspect without touching the original")
    else:
        say("")
        say("[1/5] stopping OpenWebUI (it must not write during the copy)")
        if not args.yes:
            try:
                if input("      proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                    say("      aborted")
                    return 1
            except EOFError:
                say("      no tty; pass --yes to run unattended")
                return 1
        supervisor("stop")
        time.sleep(2)

    say("")
    say(f"[2/5] copying database AND sidecars together -> {RESCUE}")
    shutil.rmtree(RESCUE, ignore_errors=True)
    RESCUE.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(DB, RESCUE / DB.name)
        for s in present:
            shutil.copy2(DB.with_name(DB.name + s), RESCUE / (DB.name + s))
    except Exception as e:
        say(f"[FAIL] could not copy: {type(e).__name__}: {e}")
        if not args.check:
            supervisor("start")
        return 3

    say("")
    say("[3/5] opening on local disk so SQLite can finish the rollback")
    info = inspect(RESCUE / DB.name)
    say(f"      integrity : {info['integrity']}")
    say(f"      chats     : {info['chats']}")
    say(f"      users     : {info['users']}")
    left = [s for s in SIDECARS if (RESCUE / (DB.name + s)).exists()]
    say(f"      sidecars  : {', '.join(left) if left else 'none left (rollback completed)'}")

    if not info["ok"]:
        say("")
        say("[FAIL] integrity_check did not return ok — NOT swapping.")
        say("       The original is untouched. Restore from the newest archive")
        say("       in /data/backups instead, and keep /tmp/rescue for analysis.")
        if not args.check:
            supervisor("start")
        return 4
    if not info["chats"]:
        say("")
        say("[FAIL] recovered database reports no chats — NOT swapping.")
        say("       That is not the shape of this failure; stop and investigate.")
        if not args.check:
            supervisor("start")
        return 5

    if args.check:
        say("")
        say("[check] the database is recoverable. Re-run without --check to swap.")
        return 0

    try:
        con = sqlite3.connect(str(RESCUE / DB.name))
        con.execute("VACUUM")
        con.commit()
        con.close()
        say(f"      vacuumed  : {(RESCUE / DB.name).stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        say(f"      vacuum skipped ({type(e).__name__}) — not fatal, continuing")

    # ---- swap: rename BOTH originals, then install ----------------------
    say("")
    say("[4/5] swapping in the recovered database")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        DB.rename(DB.with_name(f"{DB.name}.broken-{stamp}"))
        for s in present:
            src = DB.with_name(DB.name + s)
            if src.exists():
                # Renamed, NOT deleted, and renamed so it can never be found
                # beside the new database — a stale journal applied to a
                # replaced file is how a good database becomes a bad one.
                src.rename(DB.with_name(f"{DB.name}{s}.broken-{stamp}"))
        shutil.copy2(RESCUE / DB.name, DB)
        os.chmod(DB, 0o666)
    except Exception as e:
        say(f"[FAIL] swap failed: {type(e).__name__}: {e}")
        say(f"       The verified copy is still at {RESCUE / DB.name}")
        return 6

    stale = [s for s in SIDECARS if DB.with_name(DB.name + s).exists()]
    if stale:
        say(f"[FAIL] a sidecar still sits beside the new database: {stale}")
        say("       Do NOT start OpenWebUI. Investigate before continuing.")
        return 7
    say("      no sidecar beside the new database — safe to start")

    # Move the broken pair out of the backup path so it stops riding along
    # inside every archive.
    try:
        FORENSICS.mkdir(parents=True, exist_ok=True)
        for p in DB.parent.glob(f"{DB.name}*.broken-{stamp}"):
            shutil.move(str(p), str(FORENSICS / p.name))
        say(f"      broken originals moved to {FORENSICS}")
    except Exception as e:
        say(f"      (left broken originals in place: {type(e).__name__})")

    say("")
    say("[5/5] starting OpenWebUI")
    supervisor("start")
    time.sleep(8)
    supervisor("status")

    # r318-b F5. This used to stop here - RECOVERED printed, exit 0, and
    # the script's own job was done. But this script starts ONLY openwebui
    # (see `supervisor`'s default `program="openwebui"` above, and [1/5]'s
    # stop, same default) - it never stopped compactor or backup, so it
    # cannot know whether they were running before this recovery started
    # (a runbook route that stops them by hand - 2A does exactly that -
    # leaves no record here for the script to read back). OpenWebUI's own
    # model endpoint IS the compactor, so until an operator runs the line
    # below, "RECOVERED" is true of the database and false of her ability
    # to chat.
    #
    # Bounded wait for a REAL answer, not a fixed sleep, before saying so -
    # the same check RUNBOOK_DB_JOURNAL.md route 2B has an operator poll by
    # hand ("re-run it every 30 seconds"). Verifying OpenWebUI is actually
    # serving BEFORE telling the operator to add compactor/backup's own
    # load back onto the freshly-recovered database mirrors 2B's own
    # ordering ("Start OpenWebUI alone first ... Then start the rest").
    say("")
    say("      waiting for OpenWebUI to answer "
        f"({OPENWEBUI_CONFIG_URL}) before finishing...")
    if wait_for_openwebui():
        say("      OpenWebUI answered 200")
    else:
        say("      WARNING: OpenWebUI did not answer within the wait - "
            "check it by hand (supervisorctl status; tail /data/logs/"
            "openwebui.log) before running the line below")

    say("")
    say("=" * 62)
    say(f"RECOVERED — {info['chats']} chats, {info['users']} users, integrity ok")
    say(f"verified copy kept at {RESCUE / DB.name}")
    say("=" * 62)
    say("")
    say("This script only ever starts/stops openwebui. If compactor and/or")
    say("backup were stopped for this recovery (RUNBOOK_DB_JOURNAL.md route")
    say("2A: `supervisorctl stop openwebui compactor backup`), start them")
    say("now - until you do, no chat reaches vLLM:")
    say("")
    say("    supervisorctl start compactor backup")
    say("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
