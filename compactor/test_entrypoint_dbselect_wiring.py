"""
entrypoint.sh's "WHICH DATABASE DOES OPENWEBUI ACTUALLY OPEN?" block —
bash-level wiring test.

test_dbselect.py already covers the actual decision (dbselect.decide() and
its CLI) exhaustively. What THAT file cannot catch is a bug in the bash
glue around it: the `eval "$(...)"` line breaking, `_pg_public_tables`'s or
`_sqlite_chats`'s output not reaching dbselect.py the way entrypoint.sh
intends, or the final `if`/`elif` chain picking the wrong DATABASE_URL even
though DBSELECT_DATABASE came back correct. So this file runs the REAL
lines out of entrypoint.sh under real bash — extracted by line range, not
retyped — with only `_pg_public_tables`/`_sqlite_chats`/`runuser` mocked
out (the parts that need a live Postgres/postgres OS user this test has no
business standing up). If entrypoint.sh's block ever gets edited without
this test being re-pointed at the new line range, check [0] below fails
loudly rather than silently testing stale bash.

Requires bash on PATH (Git Bash on this machine). Every scenario mirrors
one of the five the task calls out by name.

    python test_entrypoint_dbselect_wiring.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "entrypoint.sh"
COMPACTOR_DIR = Path(__file__).resolve().parent

FAILED = []


def _to_posix(path: Path) -> str:
    """Windows path -> the /c/... form Git Bash (this machine's bash)
    expects. A raw Windows path embedded in a bash script is a backslash
    minefield (bash reads \\U, \\A, ... as escapes) — this is not a style
    preference, unconverted paths make every scenario below fail before
    the actual dbselect logic is ever exercised."""
    s = str(path).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = f"/{s[0].lower()}{s[2:]}"
    return s


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}")


def _extract_block() -> str:
    """The real bash text of the decision block, sliced out of
    entrypoint.sh by its own stable markers — never retyped, so this test
    can't silently drift from what actually ships.

    Starts AFTER _pg_public_tables()/_sqlite_chats()'s own definitions
    (those two need a live `postgres` OS user and are replaced with fakes
    by the caller — see _run_scenario) and runs through the closing `fi` of
    the decision if/elif chain."""
    lines = ENTRYPOINT.read_text(encoding="utf-8").splitlines()
    # Explicit sentinels, not positional guesses. The previous markers were
    # "the PG_TABLES line" to "the [3/3] banner", and when the pg_ctl stop
    # was moved down to sit before that banner this test silently swallowed
    # it -- running a postgres shutdown it had no business running and
    # failing all five scenarios for an unrelated reason.
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith("# --- BEGIN db-decision")) + 1
    end = next(i for i, l in enumerate(lines)
               if l.strip().startswith("# --- END db-decision"))
    block = lines[start:end]
    assert any("dbselect.py" in l for l in block), (
        "extracted block no longer calls dbselect.py -- entrypoint.sh's "
        "decision block moved or was rewritten; re-point this test's "
        "extraction markers at the new location"
    )
    return "\n".join(block)


_BLOCK = _extract_block()

PYTHON_EXE = _to_posix(Path(sys.executable))


def _find_bash() -> str:
    """The bare command "bash" is not safe to trust here: Windows'
    CreateProcess searches System32 BEFORE the PATH env var, and
    System32\\bash.exe is the WSL launcher, not Git's MSYS bash — it
    parses `/c/...`-style paths and multi-line function defs differently
    and every scenario below fails with confusing "command not found"
    errors that have nothing to do with dbselect.py. Pin to a real Git
    Bash explicitly; fall back to whatever "bash" resolves to only if none
    of the usual Git-for-Windows locations exist (e.g. a non-Windows CI
    runner, where the ambiguity above doesn't exist in the first place)."""
    for candidate in (
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return "bash"


BASH_EXE = _find_bash()


def _dbselect_invocation(broken: bool) -> str:
    """How the extracted block calls dbselect. `broken` points it at a path
    that does not exist - the shape of a missing module, a broken venv, or an
    import-time traceback. All three leave the eval producing nothing."""
    if broken:
        return f'"{PYTHON_EXE}" "/nonexistent/dbselect.py"'
    return f'"{PYTHON_EXE}" "{_to_posix(COMPACTOR_DIR / "dbselect.py")}"'


def _run_scenario(*, pg_tables: int, sqlite_present: bool, sqlite_chats, chats_probe_fails: bool = False, dbselect_broken: bool = False):
    """Runs the real extracted bash block with fakes for the two functions
    that would otherwise need a live Postgres, and returns the resulting
    shell variables this test cares about."""
    tmp = Path(tempfile.mkdtemp(prefix="entrypoint-dbselect-test-"))
    webui_db = tmp / "webui.db"
    if sqlite_present:
        webui_db.write_text("not a real sqlite file, just needs to exist", encoding="utf-8")

    if chats_probe_fails:
        # Same shape as a real probe failure: the heredoc's python process
        # never gets to print anything, so command substitution captures
        # empty output — entrypoint.sh's own `${SQLITE_CHATS:-unknown}`
        # fallback is what must turn that into "unknown", not a hardcoded
        # unknown at the mock level.
        # `return 1`, not `:`. A crashed python probe prints nothing AND
        # exits non-zero; a stub that exits 0 is LAXER than the thing it
        # models, and it passed against code where `set -e` aborted the
        # whole boot on that non-zero status before the fail-safe could
        # run. A stub must be at least as harsh as production or it
        # certifies a path that does not work.
        fake_sqlite_chats_fn = '_sqlite_chats() { return 1; }'
    else:
        fake_sqlite_chats_fn = f'_sqlite_chats() {{ echo "{sqlite_chats}"; }}'

    script = f"""
set -e
POSTGRES_USER=openwebui
POSTGRES_DB=openwebui
POSTGRES_SOCKET_DIR=/tmp/fake-socket-dir
WEBUI_LOCAL_DB="{_to_posix(webui_db)}"
LOG_DIR="{_to_posix(tmp)}"

# Fakes standing in for the two probes that would otherwise require a
# live Postgres and the `postgres` OS user (see this file's docstring).
_pg_public_tables() {{ echo "{pg_tables}"; }}
{fake_sqlite_chats_fn}

# The REAL dbselect.py, invoked exactly as entrypoint.sh invokes it,
# just without the /opt/compactor-venv path (this test runs it with the
# same interpreter test_dbselect.py itself uses).
{_BLOCK.replace('/opt/compactor-venv/bin/python /opt/compactor/dbselect.py', _dbselect_invocation(dbselect_broken))}

echo "RESULT_DATABASE_URL=${{DATABASE_URL}}"
echo "RESULT_WEBUIDB_SYNC_ENABLED=${{WEBUIDB_SYNC_ENABLED}}"
echo "RESULT_MIGRATION_PENDING=${{DBSELECT_MIGRATION_PENDING}}"
echo "RESULT_UNKNOWN_CHAT_COUNT=${{DBSELECT_UNKNOWN_CHAT_COUNT}}"
"""
    r = subprocess.run(
        [BASH_EXE, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    out = {}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    out["_returncode"] = r.returncode
    out["_stdout"] = r.stdout
    out["_stderr"] = r.stderr
    return out


print()
print("[0] sanity: the extraction found the real block and it mentions dbselect.py")
check("dbselect.py" in _BLOCK, "extracted bash block references dbselect.py")
check("_pg_public_tables" in _BLOCK, "extracted bash block still defines _pg_public_tables")

print()
print("[1] Postgres empty + SQLite has chats -> SQLite selected, sync on, migration text printed")
r = _run_scenario(pg_tables=0, sqlite_present=True, sqlite_chats=7)
check(r["_returncode"] == 0, f"bash block runs cleanly (stderr: {r['_stderr'][:300]})")
check(r.get("RESULT_DATABASE_URL", "").startswith("sqlite:///"), f"DATABASE_URL is a sqlite:// URL (got {r.get('RESULT_DATABASE_URL')})")
check(r.get("RESULT_WEBUIDB_SYNC_ENABLED") == "true", f"WEBUIDB_SYNC_ENABLED=true (got {r.get('RESULT_WEBUIDB_SYNC_ENABLED')})")
check("MIGRATION PENDING" in r["_stdout"], "migration instructions are printed to the boot log")
check("migrate-webui-sqlite-to-pg.py" in r["_stdout"], "the actual migration script path is printed")

print()
print("[2] Postgres has tables -> Postgres selected, sync off")
r = _run_scenario(pg_tables=4, sqlite_present=True, sqlite_chats=7)
check(r.get("RESULT_DATABASE_URL", "").startswith("postgresql://"), f"DATABASE_URL is a postgresql:// URL (got {r.get('RESULT_DATABASE_URL')})")
check(r.get("RESULT_WEBUIDB_SYNC_ENABLED") == "false", f"WEBUIDB_SYNC_ENABLED=false (got {r.get('RESULT_WEBUIDB_SYNC_ENABLED')})")
check("MIGRATION PENDING" not in r["_stdout"], "no migration banner once postgres has tables")

print()
print("[3] both empty (genuinely fresh deploy) -> Postgres selected, no false alarm")
r = _run_scenario(pg_tables=0, sqlite_present=True, sqlite_chats=0)
check(r.get("RESULT_DATABASE_URL", "").startswith("postgresql://"), f"DATABASE_URL is postgresql:// (got {r.get('RESULT_DATABASE_URL')})")
check("MIGRATION PENDING" not in r["_stdout"], "no false alarm when sqlite genuinely has 0 chats")
check(r.get("RESULT_UNKNOWN_CHAT_COUNT") == "false", "not flagged unknown for a real, known 0")

print()
print("[4] SQLite file absent entirely -> Postgres selected, no crash")
r = _run_scenario(pg_tables=0, sqlite_present=False, sqlite_chats=0)
check(r["_returncode"] == 0, f"bash block does not crash when webui.db is missing (stderr: {r['_stderr'][:300]})")
check(r.get("RESULT_DATABASE_URL", "").startswith("postgresql://"), f"DATABASE_URL is postgresql:// (got {r.get('RESULT_DATABASE_URL')})")

print()
print("[5] the chat-count probe failing/unreadable -> fails SAFE (never silently switches to Postgres)")
r = _run_scenario(pg_tables=0, sqlite_present=True, sqlite_chats=None, chats_probe_fails=True)
check(r["_returncode"] == 0, f"bash block does not crash on a failed probe (stderr: {r['_stderr'][:300]})")
check(r.get("RESULT_DATABASE_URL", "").startswith("sqlite:///"), f"stays on sqlite:// rather than guessing (got {r.get('RESULT_DATABASE_URL')})")
check(r.get("RESULT_UNKNOWN_CHAT_COUNT") == "true", f"flagged unknown (got {r.get('RESULT_UNKNOWN_CHAT_COUNT')})")
check("could not be read" in r["_stdout"] or "WARNING" in r["_stdout"], "a distinct warning is printed for the unknown-count case, not the ordinary migration banner")

print()
print("[6] dbselect itself unavailable -> REFUSE to boot, never fall through to Postgres")
# eval discards its command substitution's exit status, so `set -e` cannot
# fire and DBSELECT_DATABASE is simply unset. Without an explicit guard the
# if/else falls through to the POSTGRES branch and hands her the empty
# database - failing OPEN in the one direction that must fail closed.
_r6 = _run_scenario(pg_tables=0, sqlite_present=True, sqlite_chats=31, dbselect_broken=True)
check(_r6["_returncode"] != 0,
            f"the block refuses to continue (rc={_r6['_returncode']})")
check("postgresql://" not in _r6.get("RESULT_DATABASE_URL", ""),
            "it did NOT silently select Postgres")
check("refusing to guess" in _r6["_stdout"].lower()
            or "no decision" in _r6["_stdout"].lower(),
            "and says why, rather than dying silently")

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All entrypoint.sh dbselect-wiring tests passed.")
