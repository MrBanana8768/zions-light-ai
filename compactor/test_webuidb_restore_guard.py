"""
The two defects that COMPOSE into losing every conversation she has.

Neither one loses anything alone, which is why both shipped. Together they
are the whole failure:

    restore_on_boot() fails  ->  it reports success (exit 0, defect 2)
      ->  OpenWebUI starts onto a database that does not exist
      ->  alembic builds a fresh empty schema
      ->  she sends one message
      ->  sync_once() publishes that 1-chat database over the snapshot
          holding every conversation she has (defect 1)

DEFECT 1, the shrink guard standing down exactly when it is needed. The
guard read `previous = _has_rows(SNAPSHOT_DB) if SNAPSHOT_DB.exists()
else None` and then opened with a bare `if previous and ...`. _has_rows()
returns None for ANY failure, so an UNREADABLE snapshot - the precise
condition this module exists for - made the guard falsy and the publish went
straight over it. The integrity checks above it validate the NEW image, never
the one about to be destroyed. Second route, needing no corruption at all:
the guard also required `previous >= 10`, and on a single-user pod where one
conversation carries most of the traffic the `chat` table sits below that, so
the guard was unreachable by arithmetic.

DEFECT 2, a failed restore being undetectable. restore_on_boot() never
raises, __main__ never exited non-zero, and `cmd | tail` reports TAIL's
status - so entrypoint.sh's `|| { WARNING }` branch could not run.

EVERY REFUSAL HERE IS PAIRED WITH A CONTROL. A guard that refuses everything
passes a test suite and destroys a deployment: it stops publishing her
history to /data at all, which is the same data loss taking longer.

    python test_webuidb_restore_guard.py
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Two volumes, same as test_webuidb.py: LOCAL stands in for the pod's
# overlay, SNAP for the MooseFS mount.
_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="guard-local-"))
_SNAP_VOL = Path(tempfile.mkdtemp(prefix="guard-moosefs-"))
_QUAR = Path(tempfile.mkdtemp(prefix="guard-forensics-"))
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUAR)

import sqlite3  # noqa: E402
import webuidb  # noqa: E402

FAILED: list[str] = []
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL = webuidb.LOCAL_DB
SNAP = webuidb.SNAPSHOT_DB


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def make_db(path: Path, n: int, marker: str = "x", body: int = 40) -> None:
    """`body` is the size of each chat's JSON blob. It matters: OpenWebUI
    stores a whole conversation as one row, so row count and byte size are
    independent axes here, and the byte guard can only be tested by moving
    one without the other."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("create table if not exists chat (id text, marker text, body text)")
    con.execute("create table if not exists user (id text)")
    con.executemany(
        "insert into chat values (?, ?, ?)",
        [(str(i), marker, "x" * body) for i in range(n)],
    )
    con.execute("insert into user values ('a')")
    con.commit()
    con.close()


def chats(path: Path):
    return webuidb._has_rows(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wipe():
    for p in (LOCAL, SNAP):
        for suffix in ("",) + webuidb.SIDECARS:
            f = p.with_name(p.name + suffix)
            if f.exists():
                f.unlink()


def make_unreadable(path: Path) -> None:
    """Zero page 1 of an existing database, in place.

    Not random bytes: a stalled MooseFS mount serves ZEROS, and a zeroed
    header is what both 2026-08-31 incidents left behind. It is also
    deterministic - the "SQLite format 3" magic is gone, so _has_rows() can
    only fail - which matters because a test that corrupts a database and
    HOPES the read fails is a test that can pass for the wrong reason.
    """
    with open(path, "r+b") as fh:
        fh.seek(0)
        fh.write(b"\x00" * 4096)


def run_restore(local: Path, snap: Path, quar: Path):
    """`webuidb.py --restore` in a fresh interpreter, returning (rc, output).

    A subprocess because THE EXIT CODE IS THE THING UNDER TEST: it is all
    entrypoint.sh can see, and an in-process call to restore_on_boot() cannot
    observe it at all.
    """
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(HERE),
        PYTHONIOENCODING="utf-8",
        WEBUI_LOCAL_DB=str(local),
        WEBUI_SNAPSHOT_DB=str(snap),
        WEBUI_DB_QUARANTINE=str(quar),
    )
    r = subprocess.run(
        [sys.executable, "webuidb.py", "--restore"],
        cwd=str(HERE), env=env, capture_output=True, text=True,
        errors="replace", timeout=120,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


print("two volumes:")
print(f"  local (overlay)  {_LOCAL_VOL}")
print(f"  snapshot (mfs)   {_SNAP_VOL}")

# ---------------------------------------------------------------------------
print()
print("[1] AN UNREADABLE PREVIOUS SNAPSHOT MUST REFUSE THE PUBLISH")
# Defect 1, the core of it. The snapshot is there and holds her history; the
# volume stalled and left its header zeroed. _has_rows() answers None, which
# the old condition read as "nothing to compare against - go ahead", and the
# publish overwrote the only durable copy with whatever local happened to be.
# An unreadable database is RECOVERABLE (recover-webui-db.py, /data/backups);
# an overwritten one is not. That asymmetry is the whole guard.
wipe()
make_db(SNAP, 40, "HER-REAL-HISTORY")
make_unreadable(SNAP)
make_db(LOCAL, 1, "FRESH-EMPTY-START")
before = digest(SNAP)
check(
    chats(SNAP) is None,
    "PRECONDITION: the snapshot's chat count really is unreadable - without "
    "this the case proves nothing, because a readable snapshot would be "
    "refused by the ordinary shrink comparison instead",
)
r = webuidb.sync_once(force=True)
check(r["synced"] is False, f"the publish was refused (synced={r['synced']})")
check(
    digest(SNAP) == before,
    "THE SNAPSHOT IS BYTE-FOR-BYTE UNTOUCHED - it is damaged, not gone, and "
    "every way it gets damaged here is recoverable until something writes "
    "over it",
)
check(
    r["error"] and "REFUSING" in r["error"],
    f"and it said why, loudly ({(r['error'] or '')[:60]}...)",
)

# ---------------------------------------------------------------------------
print()
print("[2] CONTROL: an ABSENT previous snapshot still publishes")
# The control that makes [1] mean something. _has_rows() returns None for
# BOTH "unreadable" and "there is no file", and the two want opposite
# answers: a first publish onto an empty volume must work, or the fix would
# simply stop backing her up forever. If this and [1] cannot both pass, the
# guard is reading truthiness instead of distinguishing the three states.
wipe()
make_db(LOCAL, 3, "FIRST-EVER-PUBLISH")
check(not SNAP.exists(), "PRECONDITION: there is no snapshot at all")
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True and chats(SNAP) == 3,
    f"the first publish onto a bare volume succeeded (synced={r['synced']}, "
    f"error={r['error']})",
)

# ---------------------------------------------------------------------------
print()
print("[3] CONTROL: a growing database publishes, and so does ordinary deletion")
# The second control, and it is aimed at the FLOOR CHANGE specifically.
# SHRINK_GUARD_MIN_CHATS went from 10 to 2, so counts in the 3-9 range are
# now compared where before the guard stood down. Ordinary use has to survive
# that: deleting one chat of nine must still reach the snapshot, or the guard
# has quietly stopped backing her up, which is its own data-loss mode (A4).
wipe()
make_db(SNAP, 9, "SNAP")
make_db(LOCAL, 40, "LOCAL-GREW")
r = webuidb.sync_once(force=True)
check(r["synced"] is True and chats(SNAP) == 40, "growth published")

wipe()
make_db(SNAP, 9, "SNAP")
make_db(LOCAL, 8, "LOCAL-AFTER-ONE-DELETION")
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True and chats(SNAP) == 8,
    f"9 -> 8 published normally at the new floor (synced={r['synced']}, "
    f"error={r['error']}) - the floor came down to arm the guard, not to "
    f"start refusing housekeeping",
)

# ---------------------------------------------------------------------------
print()
print("[4] the floor no longer exceeds the row count it is supposed to police")
# Defect 1's second route, which needs no corruption at all. previous=3 is
# under the OLD floor of 10, so `previous >= SHRINK_GUARD_MIN_CHATS` was
# False and the guard was unreachable: losing two thirds of her chats
# published silently. A floor that exceeds the real row count is not a floor.
wipe()
make_db(SNAP, 3, "HER-THREE-CHATS")
make_db(LOCAL, 1, "ONE-LEFT")
check(
    SNAP.stat().st_size < webuidb.SHRINK_GUARD_MIN_BYTES,
    f"PRECONDITION: both files are under the BYTE floor "
    f"({SNAP.stat().st_size} < {webuidb.SHRINK_GUARD_MIN_BYTES}), so only "
    f"the chat-count guard can fire here - otherwise this case would pass on "
    f"the other measure and prove nothing about the floor",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and chats(SNAP) == 3,
    f"3 chats -> 1 was refused (synced={r['synced']}, snapshot={chats(SNAP)})",
)

# ---------------------------------------------------------------------------
print()
print("[5] ONE conversation replaced by ONE empty conversation, caught on size")
# The shape this pod actually has. OpenWebUI keeps an entire conversation as
# one JSON blob in one row of `chat`, and hers is tens of megabytes. A failed
# restore that leaves one freshly created conversation behind is 1 row
# against 1 row, which NO ratio on counts can catch at ANY floor - and tens
# of kilobytes against tens of megabytes, which the byte ratio catches on the
# first sync. This is why lowering the floor was not enough on its own.
wipe()
make_db(SNAP, 1, "ONE-HUGE-CONVERSATION", body=300_000)
make_db(LOCAL, 1, "ONE-EMPTY-CONVERSATION", body=40)
check(
    chats(SNAP) == 1 and chats(LOCAL) == 1,
    f"PRECONDITION: one row against one row (snapshot={chats(SNAP)}, "
    f"local={chats(LOCAL)}) - the count guard CANNOT be what fires below",
)
check(
    SNAP.stat().st_size >= webuidb.SHRINK_GUARD_MIN_BYTES,
    f"PRECONDITION: the snapshot is over the byte floor "
    f"({SNAP.stat().st_size} >= {webuidb.SHRINK_GUARD_MIN_BYTES})",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and chats(SNAP) == 1 and SNAP.stat().st_size > 100_000,
    f"the empty conversation was refused on SIZE (synced={r['synced']}, "
    f"snapshot still {SNAP.stat().st_size} bytes)",
)

print("    CONTROL: the same one-row shape, growing, still publishes")
# Without this, [5] passes by refusing every single-chat database, which
# would strand the snapshot of a pod that has exactly one conversation -
# i.e. this one.
wipe()
make_db(SNAP, 1, "SMALL", body=300_000)
make_db(LOCAL, 1, "GREW", body=600_000)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True,
    f"a single conversation that GREW published normally (synced="
    f"{r['synced']}, error={r['error']})",
)

# ---------------------------------------------------------------------------
print()
print("[6] the deliberate override still opens both refusals")
# Refusing forever is its own failure: she really can clear her history, and
# a snapshot that can never be replaced is a backup that has stopped. The
# override has to cover the NEW refusal as well as the old one - a flag that
# only opens one of two doors is this project's recurring defect wearing a
# hat.
wipe()
make_db(SNAP, 40, "HER-REAL-HISTORY")
make_unreadable(SNAP)
make_db(LOCAL, 1, "DELIBERATE")
os.environ["WEBUI_DB_ALLOW_SHRINK"] = "1"
try:
    webuidb._reload_env()
    over_unreadable = webuidb.sync_once(force=True)
    wipe()
    make_db(SNAP, 40, "SNAP", body=10_000)
    make_db(LOCAL, 1, "DELIBERATE", body=40)
    over_shrink = webuidb.sync_once(force=True)
finally:
    os.environ.pop("WEBUI_DB_ALLOW_SHRINK", None)
    webuidb._reload_env()
check(
    over_unreadable["synced"] is True,
    f"WEBUI_DB_ALLOW_SHRINK=1 publishes over an unreadable snapshot "
    f"(synced={over_unreadable['synced']}, error={over_unreadable['error']})",
)
check(
    over_shrink["synced"] is True,
    f"and over a shrinking one (synced={over_shrink['synced']}, "
    f"error={over_shrink['error']})",
)
check(
    webuidb.ALLOW_SHRINK is False,
    "and the override is back off afterwards - a test that leaks this flag "
    "disarms every case that runs after it",
)

# ---------------------------------------------------------------------------
print()
print("[7] A FAILED RESTORE IS VISIBLE IN THE EXIT CODE")
# Defect 2. restore_on_boot() returns an "action" dict and never raises;
# __main__ printed it and exited 0 regardless. entrypoint.sh could therefore
# not tell a restore that worked from one that refused to touch a damaged
# snapshot, and booted OpenWebUI onto nothing either way.
_L = Path(tempfile.mkdtemp(prefix="rc-local-"))
_S = Path(tempfile.mkdtemp(prefix="rc-snap-"))
_Q = Path(tempfile.mkdtemp(prefix="rc-quar-"))
bad_snap = _S / "webui.db"
make_db(bad_snap, 40, "HER-REAL-HISTORY")
make_unreadable(bad_snap)
rc, out = run_restore(_L / "openwebui" / "webui.db", bad_snap, _Q)
check(
    rc != 0,
    f"an unhealthy snapshot exits NON-ZERO (rc={rc}) - this is the whole of "
    f"defect 2: at rc=0 the boot script's warning branch was unreachable",
)
check(
    rc == webuidb.RESTORE_EXIT_CODES["snapshot_unhealthy"],
    f"and with the documented code for the case (rc={rc}, expected "
    f"{webuidb.RESTORE_EXIT_CODES['snapshot_unhealthy']}), so entrypoint.sh "
    f"can say which failure it is instead of guessing",
)
check(
    "snapshot_unhealthy" in out,
    "and still prints the action dict the recovery scripts branch on",
)

# ---------------------------------------------------------------------------
print()
print("[8] CONTROL: a restore that works, and a new deployment, both exit 0")
# Without these, [7] passes if --restore simply fails always - which would
# turn every ordinary pod recreate and every first deploy into a pod that
# refuses to boot.
_L2 = Path(tempfile.mkdtemp(prefix="rc2-local-"))
_S2 = Path(tempfile.mkdtemp(prefix="rc2-snap-"))
_Q2 = Path(tempfile.mkdtemp(prefix="rc2-quar-"))
good_snap = _S2 / "webui.db"
good_local = _L2 / "openwebui" / "webui.db"
make_db(good_snap, 40, "HER-REAL-HISTORY")
rc, out = run_restore(good_local, good_snap, _Q2)
check(
    rc == 0 and good_local.exists() and chats(good_local) == 40,
    f"the pod-recreate path restores and exits 0 (rc={rc}, "
    f"chats={chats(good_local) if good_local.exists() else 'absent'})",
)

_L3 = Path(tempfile.mkdtemp(prefix="rc3-local-"))
_S3 = Path(tempfile.mkdtemp(prefix="rc3-snap-"))
_Q3 = Path(tempfile.mkdtemp(prefix="rc3-quar-"))
rc, out = run_restore(_L3 / "openwebui" / "webui.db", _S3 / "webui.db", _Q3)
check(
    rc == 0 and "fresh" in out,
    f"a brand-new deployment with neither file exits 0 (rc={rc}) - an empty "
    f"schema is CORRECT there and must not be confused with a lost one",
)

# ---------------------------------------------------------------------------
print()
print("[9] every action restore_on_boot can return has an exit code")
# The sibling pin. The defect class this project ships most often is a rule
# applied at one call site and missed at its twin, and the shape it would
# take here is a new `result["action"] = "..."` branch added without a row in
# RESTORE_EXIT_CODES. Read the module's own source rather than trusting that
# whoever adds one remembers this file.
src = (HERE / "webuidb.py").read_text(encoding="utf-8", errors="replace")
body = src[src.index("def restore_on_boot"):src.index("def sync_once")]
actions = set(re.findall(r"""result\["action"\]\s*=\s*["'](\w+)["']""", body))
check(
    len(actions) >= 5,
    f"found the action strings to check ({sorted(actions)})",
)
missing = sorted(actions - set(webuidb.RESTORE_EXIT_CODES))
check(not missing, f"every action has an exit code (missing: {missing})")
check(
    any(v == 0 for v in webuidb.RESTORE_EXIT_CODES.values())
    and any(v != 0 for v in webuidb.RESTORE_EXIT_CODES.values()),
    "and the table is not uniform - some actions mean go, some mean stop, so "
    "[7] and [8] cannot both be passing by accident",
)
# Read the DEFAULT out of __main__ rather than calling .get() with a literal
# here. `RESTORE_EXIT_CODES.get("nonsense", 6) != 0` would assert that
# dict.get returns its own second argument - true forever, whatever webuidb.py
# does - and pass just as happily on a `.get(action, 0)` that waves every
# unknown failure through. Ask the source what default it actually passes.
default = re.search(
    r"""RESTORE_EXIT_CODES\.get\(\s*r\.get\(["']action["']\)\s*,\s*(\d+)\s*\)""",
    src,
)
check(
    default is not None and int(default.group(1)) != 0,
    f"an UNKNOWN action maps to NON-ZERO in __main__ "
    f"(default={default.group(1) if default else 'not found'}): a branch that "
    f"forgets the table gets a refused boot, not a silent pass",
)

# ---------------------------------------------------------------------------
print()
print("[10] entrypoint.sh actually reads that exit code and refuses to boot")
# The third half of defect 2, and the one no python test can reach: the
# status of `cmd | tail` is TAIL's, always 0, and `set -o pipefail` is not
# set. Read the shipped file as text, the way test_webuidb_gate.py does.
ENTRY = (ROOT / "entrypoint.sh").read_text(encoding="utf-8", errors="replace")
# Comment lines are excluded on purpose: the block above that step QUOTES the
# broken `| tail -3 ||` pipeline it replaced, which is exactly the text this
# case is looking for. A check that cannot tell the fix from its own
# explanation of the bug fails forever on a correct file.
restore_lines = [
    ln for ln in ENTRY.splitlines()
    if "webuidb.py --restore" in ln and not ln.lstrip().startswith("#")
]
check(
    len(restore_lines) >= 1,
    "CONTROL: the restore step still exists - every other check here passes "
    "trivially if someone deletes the step instead of fixing it",
)
check(
    all("|" not in ln for ln in restore_lines),
    f"the restore's status is not taken from a pipeline ({restore_lines!r}) - "
    f"`cmd | tail` reports tail's 0 and swallows every failure to its left",
)
check(
    "|| restore_rc=$?" in ENTRY and 'if [ "${restore_rc}" -ne 0 ]' in ENTRY,
    "the exit status is captured and branched on",
)
# Anchored on an explicit string and bounded by a character count, not by
# hunting for an indented `fi`: test_webuidb_gate.py's markers exist because
# a positional slice there once swallowed a pg_ctl stop that moved into range.
_at = ENTRY.index('if [ "${WEBUI_DB_ALLOW_EMPTY_START}" != "true" ]')
refuse = ENTRY[_at:_at + 200]
check(
    "exit 1" in refuse,
    "and a failed restore STOPS THE BOOT - OpenWebUI starting onto a missing "
    "database builds an empty schema, and that empty schema is what the sync "
    "daemon then tries to publish over her history",
)
check(
    'WEBUI_DB_ALLOW_EMPTY_START="${WEBUI_DB_ALLOW_EMPTY_START:-false}"' in ENTRY,
    "with an explicit escape hatch, defaulting to refusing - a pod that can "
    "NEVER be booted after an unrecoverable snapshot is its own outage",
)

# ---------------------------------------------------------------------------
print()
print("[11] an empty local database plus a damaged snapshot is not 'kept_local'")
# The sibling inside restore_on_boot itself. It compares local against
# `snap_chats`, which is None when the snapshot is unreadable - so the
# empty-shell branch did not take, the code fell through to `elif ok:` and
# logged "local database present and healthy (0 chats) - keeping it". True,
# cheerful, and it never mentions that the only durable copy is damaged.
# Same silently-empty boot as defect 2, reached down a different road.
wipe()
make_db(LOCAL, 0, "EMPTY-SHELL")
make_db(SNAP, 40, "HER-REAL-HISTORY")
make_unreadable(SNAP)
r = webuidb.restore_on_boot()
check(
    r["action"] == "snapshot_unhealthy",
    f"the boot is stopped, not waved through (action={r['action']})",
)
check(
    LOCAL.exists() and chats(LOCAL) == 0,
    "and the local file is left exactly where it was - nothing here is "
    "destructive, so a human can still look at both",
)

print("    CONTROL: an empty local shell beside a HEALTHY snapshot still restores")
# A9's behaviour, which must survive this change: the 0-byte/empty-schema
# local file is the thing to throw away when the volume holds real history.
wipe()
make_db(LOCAL, 0, "EMPTY-SHELL")
make_db(SNAP, 40, "HER-REAL-HISTORY")
r = webuidb.restore_on_boot()
check(
    r["action"] == "restored_from_snapshot" and chats(LOCAL) == 40,
    f"the snapshot came down over the empty shell (action={r['action']}, "
    f"local={chats(LOCAL)})",
)

# ---------------------------------------------------------------------------
print()
print("[12] a set-aside that FAILED must not be followed by an overwrite")
# _set_aside() swallowed its exception and returned None, which reads as
# "done" at both call sites. QUARANTINE lives on /data, so a stalled or full
# volume - the condition this whole module exists for - failed the move,
# logged one line, and the shutil.copy2 two branches later landed the
# snapshot on top of the database we had just promised to preserve. A
# database that fails quick_check is the RECOVERABLE case (a hot rollback
# journal, not corruption) and holds everything written since the last sync.
#
# Both branches are exercised. The rule "do not overwrite state you failed to
# preserve" applied to one of two identical call sites is this project's
# single most expensive recurring defect.
_broken_quar = Path(tempfile.mkdtemp(prefix="guard-noquar-")) / "not-a-dir"
_broken_quar.write_bytes(b"a regular file, so mkdir(parents=True) cannot pass")
_real_quar = webuidb.QUARANTINE
try:
    # NOT chmod: these suites run as root in the production image and root
    # ignores directory permission bits. A regular file in the path fails for
    # everyone (the technique test_webuidb_migration A10 had to switch to).
    webuidb.QUARANTINE = _broken_quar / "forensics"

    wipe()
    make_db(LOCAL, 5, "WRITTEN-SINCE-THE-LAST-SYNC")
    with open(LOCAL, "r+b") as fh:      # fails quick_check, still recoverable
        fh.seek(200)
        fh.write(b"\xff" * 5000)
    make_db(SNAP, 40, "OLDER-SNAPSHOT")
    local_before = digest(LOCAL)
    r = webuidb.restore_on_boot()
    check(
        r["action"] == "error",
        f"a failed quarantine stops the restore (action={r['action']})",
    )
    check(
        digest(LOCAL) == local_before,
        "AND THE LOCAL DATABASE IS UNTOUCHED - it failed quick_check, which "
        "on this volume usually means a hot journal, and it is the only copy "
        "of anything written since the last sync",
    )

    wipe()
    make_db(LOCAL, 0, "EMPTY-SHELL")
    make_db(SNAP, 40, "HER-REAL-HISTORY")
    shell_before = digest(LOCAL)
    r = webuidb.restore_on_boot()
    check(
        r["action"] == "error" and digest(LOCAL) == shell_before,
        f"THE SIBLING BRANCH BEHAVES THE SAME (action={r['action']}) - an "
        f"'empty shell' is our reading of the file, not a fact about it; it "
        f"can still hold her accounts, settings and uploads",
    )
finally:
    webuidb.QUARANTINE = _real_quar

print("    CONTROL: with a working quarantine, the same corrupt local IS replaced")
# Without this, [12] passes if restore_on_boot simply errors on every
# corrupted local database - which would turn the ordinary "set it aside and
# restore" path, the one this module exists to perform, into a refused boot.
wipe()
make_db(LOCAL, 5, "CORRUPT-LOCAL")
with open(LOCAL, "r+b") as fh:
    fh.seek(200)
    fh.write(b"\xff" * 5000)
make_db(SNAP, 40, "HER-REAL-HISTORY")
quar_before = len(list(_QUAR.iterdir()))
r = webuidb.restore_on_boot()
check(
    r["action"] == "restored_from_snapshot" and chats(LOCAL) == 40,
    f"set aside and restored as normal (action={r['action']})",
)
check(
    len(list(_QUAR.iterdir())) > quar_before,
    "and the corrupted database was PRESERVED, not deleted",
)

print()
if FAILED:
    print("!" * 66)
    for f in FAILED:
        print("FAIL " + f)
    print("!" * 66)
    sys.exit(1)
print("All webui.db restore/publish guard checks passed.")
