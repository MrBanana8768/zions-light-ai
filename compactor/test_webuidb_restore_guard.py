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

AND THEN THE FIX FOR DEFECT 1 SHIPPED TWO NEW WAYS TO STOP THE DURABLE COPY
EVER BEING WRITTEN, both worse than what they replaced because the original
needed a corrupted snapshot to fire and these fire on ordinary states:

  * a 0-BYTE snapshot reads as "unreadable". It is a valid SQLite database
    with no tables: it opens, it PASSES quick_check, and _has_rows() answers
    None for "no `chat` table" exactly as it does for corruption. Every
    SYNC_INTERVAL_S, forever, refused - [3] and [3]'s controls.
  * VACUUM legitimately shrinks the file and the byte guard refused it. The
    commit asserted that sqlite3's backup API "copies live pages only ...
    VACUUM does not move it"; it copies free-list pages, so a bloated
    database images at full size and a VACUUM of the same rows images at a
    tenth. VACUUM is in this repo's own recovery path and is named in the
    refusal's own text - [5]'s VACUUM control.

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
    stores a whole conversation as one row, so row count and stored content
    are independent axes here, and the content guard can only be tested by
    moving one without the other."""
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


def content(path: Path):
    return webuidb._content_bytes(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wipe():
    for p in (LOCAL, SNAP):
        for suffix in ("",) + webuidb.SIDECARS:
            f = p.with_name(p.name + suffix)
            if f.exists():
                f.unlink()
    # Each scenario is its own incident and expects its own forensic copy
    # (FORENSIC_COPY_MIN_INTERVAL_S) — several run within the same process
    # well inside the default 3600s throttle window; without this reset a
    # later refusal's "NOTHING new copied" or "a copy was made" assertion
    # would depend on execution order rather than on what that case tests.
    webuidb._forensic_copy_last_monotonic = None
    # The empty-start marker refuses EVERY publish while it exists, so a case
    # that leaks one turns every later control into a false refusal. Same
    # reason [6] asserts the override flag is back off afterwards.
    if webuidb.EMPTY_START_MARKER.exists():
        webuidb.EMPTY_START_MARKER.unlink()


def bloat(path: Path) -> None:
    """Insert large rows and delete them again: identical content, a file
    several times the size, none of it reclaimed.

    This is what a live OpenWebUI database looks like after ordinary churn,
    and it is the state `sqlite3`'s backup API images AT FULL SIZE - the
    refuted claim was that backup() "copies live pages only"."""
    con = sqlite3.connect(str(path))
    con.executemany(
        "insert into chat values (?, ?, ?)",
        [(f"junk{i}", "JUNK", "y" * 300_000) for i in range(40)],
    )
    con.commit()
    con.execute("delete from chat where marker = 'JUNK'")
    con.commit()
    con.close()


def vacuum(path: Path) -> None:
    """Exactly what scripts/recover-webui-db.py runs and OPERATIONS.md tells
    the operator to run."""
    con = sqlite3.connect(str(path))
    con.execute("VACUUM")
    con.close()


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
check(
    webuidb.integrity(SNAP)[0] is False,
    "PRECONDITION: and it fails quick_check too. This is the second half of "
    "the same trap: the guard now asks integrity() to tell a DAMAGED "
    "snapshot from a merely EMPTY one (see [3]), so a case that only "
    "established 'no chat count' would be refused for the wrong reason - or, "
    "after the fix, published",
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
check(
    r["error"] and "WEBUI_DB_ALLOW_SHRINK" not in r["error"],
    "and it does NOT tell the operator to set WEBUI_DB_ALLOW_SHRINK. That "
    "flag is what the SHRINK refusal advises, and while one flag covered both "
    "refusals, an operator following ordinary advice about a deliberate "
    "deletion silently disarmed this one - the corruption guard the whole "
    "module exists for",
)

# ---------------------------------------------------------------------------
print()
print("[2] CONTROL: an ABSENT previous snapshot still publishes")
# The control that makes [1] mean something. _has_rows() returns None for
# BOTH "unreadable" and "there is no file", and the two want opposite
# answers: a first publish onto an empty volume must work, or the fix would
# simply stop backing her up forever. If this and [1] cannot both pass, the
# guard is reading truthiness instead of distinguishing the states.
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
print("[3] CONTROL: an EMPTY-BUT-HEALTHY snapshot publishes - it is not damaged")
# THE SECOND CONTROL ON [1], AND THE ONE WHOSE ABSENCE SHIPPED A BLOCKER.
# `previous` has FOUR states, not three, and the fix for three broke the
# fourth: absent, empty-but-healthy, unreadable, readable-with-rows.
#
# A 0-BYTE FILE IS A VALID SQLITE DATABASE WITH NO TABLES. It opens, it
# PASSES quick_check, and _has_rows() answers None for "no `chat` table"
# exactly as it does for a zeroed header. So a 0-byte /data/openwebui/webui.db
# - an os.replace that lost the volume mid-write, a touch, a half-created file
# from an earlier migration, which on a MooseFS mount is the documented
# failure class - was read as CORRUPTION and refused every SYNC_INTERVAL_S,
# forever, with the durable copy never written and every health check green.
#
# restore_on_boot had already reasoned this out twelve lines away, in the same
# commit: "a 0-byte or schema-only file on /data passes quick_check, is not
# her history, and restoring it would gain nothing - so ask integrity() rather
# than treating every None as a catastrophe". [1] and [3] cannot both pass
# unless sync_once asks the same question.
wipe()
make_db(LOCAL, 3, "HER-HISTORY")
SNAP.parent.mkdir(parents=True, exist_ok=True)
SNAP.write_bytes(b"")
check(
    chats(SNAP) is None and webuidb.integrity(SNAP)[0] is True,
    f"PRECONDITION: the 0-byte snapshot has no readable chat count "
    f"({chats(SNAP)!r}) and yet PASSES quick_check "
    f"({webuidb.integrity(SNAP)}) - that combination is the whole bug, and if "
    f"sqlite ever stopped accepting an empty file this case would be proving "
    f"nothing",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True and chats(SNAP) == 3,
    f"a 0-byte snapshot is replaced, not treated as a catastrophe "
    f"(synced={r['synced']}, error={r['error']}) - refusing here writes no "
    f"durable copy AT ALL, which is the failure this guard exists to prevent, "
    f"reached from the other side",
)

print("    CONTROL: a schema-only snapshot (a `chat` table, zero rows) publishes too")
# The sibling state, and the one the adversarial pass confirmed was already
# right: _has_rows() returns the integer 0 here, not None, so it never reached
# the unreadable branch. It is pinned anyway because the two states are one
# concept and a future edit that "simplifies" them will land on both.
wipe()
make_db(LOCAL, 3, "HER-HISTORY")
make_db(SNAP, 0, "SCHEMA-ONLY")
check(
    chats(SNAP) == 0,
    f"PRECONDITION: this one reports the INTEGER 0, not None ({chats(SNAP)!r})",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True and chats(SNAP) == 3,
    f"a schema with no chats is replaced (synced={r['synced']}, "
    f"error={r['error']})",
)

# ---------------------------------------------------------------------------
print()
print("[4] CONTROL: a growing database publishes, and so does ordinary deletion")
# The control aimed at the FLOOR CHANGE specifically. SHRINK_GUARD_MIN_CHATS
# went from 10 to 2, so counts in the 3-9 range are now compared where before
# the guard stood down. Ordinary use has to survive that: deleting one chat of
# nine must still reach the snapshot, or the guard has quietly stopped backing
# her up, which is its own data-loss mode (A4).
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
print("[5] the floor no longer exceeds the row count it is supposed to police")
# Defect 1's second route, which needs no corruption at all. previous=3 is
# under the OLD floor of 10, so `previous >= SHRINK_GUARD_MIN_CHATS` was
# False and the guard was unreachable: losing two thirds of her chats
# published silently. A floor that exceeds the real row count is not a floor.
wipe()
make_db(SNAP, 3, "HER-THREE-CHATS")
make_db(LOCAL, 1, "ONE-LEFT")
check(
    SNAP.stat().st_size < webuidb.SHRINK_GUARD_MIN_BYTES
    and content(SNAP) < webuidb.SHRINK_GUARD_MIN_BYTES,
    f"PRECONDITION: both files are under the content floor by file size "
    f"({SNAP.stat().st_size}) AND by stored content ({content(SNAP)}), so "
    f"only the chat-count guard can fire here - otherwise this case would "
    f"pass on the other measure and prove nothing about the floor",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and chats(SNAP) == 3,
    f"3 chats -> 1 was refused (synced={r['synced']}, snapshot={chats(SNAP)})",
)

# ---------------------------------------------------------------------------
print()
print("[6] ONE conversation replaced by ONE empty conversation, caught on content")
# The shape this pod actually has. OpenWebUI keeps an entire conversation as
# one JSON blob in one row of `chat`, and hers is tens of megabytes. A failed
# restore that leaves one freshly created conversation behind is 1 row
# against 1 row, which NO ratio on counts can catch at ANY floor - and tens
# of kilobytes of content against tens of megabytes, which the content ratio
# catches on the first sync. This is why lowering the floor was not enough.
wipe()
make_db(SNAP, 1, "ONE-HUGE-CONVERSATION", body=300_000)
make_db(LOCAL, 1, "ONE-EMPTY-CONVERSATION", body=40)
check(
    chats(SNAP) == 1 and chats(LOCAL) == 1,
    f"PRECONDITION: one row against one row (snapshot={chats(SNAP)}, "
    f"local={chats(LOCAL)}) - the count guard CANNOT be what fires below",
)
check(
    content(SNAP) >= webuidb.SHRINK_GUARD_MIN_BYTES,
    f"PRECONDITION: the snapshot is over the content floor "
    f"({content(SNAP)} >= {webuidb.SHRINK_GUARD_MIN_BYTES})",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and chats(SNAP) == 1 and content(SNAP) > 100_000,
    f"the empty conversation was refused on CONTENT (synced={r['synced']}, "
    f"snapshot still holds {content(SNAP)} bytes of conversation)",
)

print("    CONTROL: the same one-row shape, growing, still publishes")
# Without this, [6] passes by refusing every single-chat database, which
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

print("    CONTROL: A VACUUM THAT HALVES THE FILE AND LOSES NOTHING STILL PUBLISHES")
# THE CONTROL WHOSE ABSENCE SHIPPED A BLOCKER. This guard compared
# os.stat().st_size on the claim that both sides are "written by sqlite3's
# backup API, which copies live pages only ... free-list churn and VACUUM do
# not move it". backup() copies EVERY page including the free list, so a
# bloated database images at full size; VACUUM repacks it to a fraction. The
# measured pair: 41.026 MB -> 41.026 MB via backup (ratio 1.0000) against
# 4.108 MB via VACUUM (ratio 0.1001). Driven through this very function the
# result was 40.07 MB -> VACUUM -> 10.03 MB -> REFUSED, five chats each side,
# nothing gone.
#
# And VACUUM is not hypothetical: scripts/recover-webui-db.py runs it,
# OPERATIONS.md tells the operator to run it, and the refusal names
# recover-webui-db.py in its own text. So the guard blocked the documented
# recovery forever, and its only escape also disarmed [1].
wipe()
make_db(LOCAL, 5, "HER-REAL-HISTORY", body=300_000)
webuidb.sync_once(force=True)
bloat(LOCAL)
webuidb.sync_once(force=True)
bloated_file = SNAP.stat().st_size
bloated_content = content(SNAP)
vacuum(LOCAL)
check(
    LOCAL.stat().st_size < bloated_file * webuidb.SHRINK_REFUSE_BELOW,
    f"PRECONDITION: the VACUUM really did take the file below the refusal "
    f"ratio ({LOCAL.stat().st_size} < {bloated_file} * "
    f"{webuidb.SHRINK_REFUSE_BELOW}) - on a build where it did not, this case "
    f"would publish for the trivial reason and prove nothing",
)
check(
    content(LOCAL) == bloated_content and chats(LOCAL) == 5,
    f"PRECONDITION: and NOTHING WAS LOST doing it - {chats(LOCAL)} chats and "
    f"{content(LOCAL)} bytes of conversation, against {bloated_content} "
    f"before. This is the entire distinction the guard has to make: the same "
    f"content, compacted, is not the same thing as content that is gone",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True and content(SNAP) == bloated_content,
    f"the VACUUMed image published (synced={r['synced']}, error={r['error']}) "
    f"- the operator can follow OPERATIONS.md without the sync daemon "
    f"refusing every cycle until a human intervenes",
)

# ---------------------------------------------------------------------------
print()
print("[7] the two overrides are SEPARATE switches, and each opens only its own door")
# Refusing forever is its own failure: she really can clear her history, and a
# snapshot that can never be replaced is a backup that has stopped. But ONE
# flag for both refusals was worse than either problem it solved, because THE
# SHRINK REFUSAL'S OWN TEXT TELLS THE OPERATOR TO SET IT. An operator who
# really did delete a conversation follows the printed instruction and, in the
# same keystroke, disarms the unreadable-snapshot refusal - the guard the
# module exists for. Demonstrated: ALLOW_SHRINK=1 published a 4 KB database
# over a 1 MB history.
#
# All four cells are asserted. A 2x2 where only the diagonal is checked is the
# same defect wearing a hat: it cannot see a flag that opens both doors.
_matrix = {}
for _flag in ("WEBUI_DB_ALLOW_SHRINK", "WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE"):
    os.environ[_flag] = "1"
    try:
        webuidb._reload_env()
        wipe()
        make_db(SNAP, 40, "HER-REAL-HISTORY")
        make_unreadable(SNAP)
        make_db(LOCAL, 1, "DELIBERATE")
        _matrix[(_flag, "unreadable")] = webuidb.sync_once(force=True)
        wipe()
        make_db(SNAP, 40, "SNAP", body=10_000)
        make_db(LOCAL, 1, "DELIBERATE", body=40)
        _matrix[(_flag, "shrink")] = webuidb.sync_once(force=True)
    finally:
        os.environ.pop(_flag, None)
        webuidb._reload_env()

check(
    _matrix[("WEBUI_DB_ALLOW_SHRINK", "shrink")]["synced"] is True,
    f"ALLOW_SHRINK opens the SHRINK refusal - she really did clear her "
    f"history and the snapshot must be allowed to follow "
    f"(synced={_matrix[('WEBUI_DB_ALLOW_SHRINK', 'shrink')]['synced']})",
)
check(
    _matrix[("WEBUI_DB_ALLOW_SHRINK", "unreadable")]["synced"] is False,
    f"AND IT DOES NOT OPEN THE UNREADABLE REFUSAL "
    f"(synced={_matrix[('WEBUI_DB_ALLOW_SHRINK', 'unreadable')]['synced']}) - "
    f"opting into a known-good shrink must not be opting out of corruption "
    f"protection, which is exactly what the refusal message used to advise",
)
check(
    _matrix[("WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE", "unreadable")]["synced"]
    is True,
    f"ALLOW_PUBLISH_OVER_UNREADABLE opens the UNREADABLE refusal - a snapshot "
    f"that is genuinely unrecoverable must not block publishing forever "
    f"(error="
    f"{_matrix[('WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE', 'unreadable')]['error']})",
)
check(
    _matrix[("WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE", "shrink")]["synced"]
    is False,
    "and IT does not open the shrink refusal either - the split has to hold "
    "in both directions or it is one switch with two names",
)
check(
    webuidb.ALLOW_SHRINK is False
    and webuidb.ALLOW_PUBLISH_OVER_UNREADABLE is False,
    "and both overrides are back off afterwards - a case that leaks either "
    "flag disarms every case that runs after it",
)

# ---------------------------------------------------------------------------
print()
print("[8] A FAILED RESTORE IS VISIBLE IN THE EXIT CODE")
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
    "snapshot_unhealthy" in out,
    "and still prints the action dict the recovery scripts branch on",
)

# ---------------------------------------------------------------------------
print()
print("[9] EVERY ROW OF RESTORE_EXIT_CODES, DRIVEN FOR REAL, AGAINST ITS LITERAL")
# Four of the six rows were asserted by nothing: `"error": 5 -> 0` and
# `"restore_failed": 4 -> 0` both stayed green, and so did `kept_local`. The
# one row that looked covered was covered by `rc ==
# RESTORE_EXIT_CODES["snapshot_unhealthy"]`, which reads the value out of the
# table it is checking - change 3 to 7 and both sides move together. Only the
# separate `rc != 0` was doing any work.
#
# So every row is driven through the real subprocess and compared against a
# LITERAL. An exit code is a wire format: entrypoint.sh has a `case` arm per
# value and a human reads it out of a boot log, so the numbers are the
# contract, not an implementation detail.
_cases = []

# kept_local -> 0. A healthy local database with chats wins outright.
_k = Path(tempfile.mkdtemp(prefix="rc-kept-"))
make_db(_k / "openwebui" / "webui.db", 5, "LOCAL-IS-NEWER")
_cases.append(("kept_local", 0, run_restore(
    _k / "openwebui" / "webui.db", _k / "nosnap" / "webui.db",
    Path(tempfile.mkdtemp(prefix="rc-kept-q-")))))

# restored_from_snapshot -> 0. The pod-recreate path.
_r = Path(tempfile.mkdtemp(prefix="rc-restored-"))
make_db(_r / "snap" / "webui.db", 40, "HER-REAL-HISTORY")
_restored_local = _r / "openwebui" / "webui.db"
_cases.append(("restored_from_snapshot", 0, run_restore(
    _restored_local, _r / "snap" / "webui.db",
    Path(tempfile.mkdtemp(prefix="rc-restored-q-")))))

# fresh -> 0. A brand-new deployment with neither file.
_f = Path(tempfile.mkdtemp(prefix="rc-fresh-"))
_cases.append(("fresh", 0, run_restore(
    _f / "openwebui" / "webui.db", _f / "snap" / "webui.db",
    Path(tempfile.mkdtemp(prefix="rc-fresh-q-")))))

# snapshot_unhealthy -> 3. Reuses [8]'s subprocess rather than running it
# twice; [8] asserts only that it is non-zero, this asserts WHICH non-zero.
_cases.append(("snapshot_unhealthy", 3, (rc, out)))

# restore_failed -> 4. The snapshot is healthy; the copy to local disk fails.
#
# The state needed is narrow: Path.exists() must report the local file ABSENT
# - so the whole local block is skipped and nothing is set aside - and
# open(dst, "wb") must then fail, so shutil.copy2 raises. Every other way to
# break the copy (a directory at the path, an unwritable parent, a broken
# quarantine) is caught by an EARLIER branch and returns "error", which would
# quietly pin the wrong row while still printing an ok.
#
# The technique differs by platform because the mechanism does, not the
# intent. POSIX: a symlink into a directory that does not exist - exists()
# follows it and reports absent, open() follows it and gets ENOENT. Windows:
# a filename containing a character NTFS forbids - exists() swallows the
# OSError and reports absent, open() raises it. The suite runs on Linux
# (scripts/run-tests.py, in the unit-test image), so the first is the real
# one; the second exists so that a developer on Windows WATCHES this row too
# rather than reading a permanently red line and learning to ignore it. A row
# pinned on one platform and skipped on the other is this project's recurring
# defect with a uname in front of it.
_rf = Path(tempfile.mkdtemp(prefix="rc-failed-"))
make_db(_rf / "snap" / "webui.db", 40, "HER-REAL-HISTORY")
(_rf / "openwebui").mkdir(parents=True, exist_ok=True)
if os.name == "nt":
    _unwritable_local = _rf / "openwebui" / "web*ui.db"
else:
    _unwritable_local = _rf / "openwebui" / "webui.db"
    os.symlink(str(_rf / "no-such-directory" / "webui.db"), str(_unwritable_local))
check(
    not _unwritable_local.exists(),
    f"PRECONDITION: the local path reads as ABSENT ({_unwritable_local}) - if "
    f"it read as present, restore_on_boot would set it aside first and this "
    f"case would be pinning 'error', not 'restore_failed'",
)
_cases.append(("restore_failed", 4, run_restore(
    _unwritable_local, _rf / "snap" / "webui.db",
    Path(tempfile.mkdtemp(prefix="rc-failed-q-")))))

# error -> 5. A local database that fails quick_check plus a quarantine that
# cannot be created: _set_aside returns False and the restore must stop rather
# than overwrite a database it failed to preserve.
_e = Path(tempfile.mkdtemp(prefix="rc-error-"))
make_db(_e / "openwebui" / "webui.db", 5, "WRITTEN-SINCE-THE-LAST-SYNC")
with open(_e / "openwebui" / "webui.db", "r+b") as fh:
    fh.seek(200)
    fh.write(b"\xff" * 5000)
make_db(_e / "snap" / "webui.db", 40, "HER-REAL-HISTORY")
_blocked_quar = _e / "not-a-dir"
_blocked_quar.write_bytes(b"a regular file, so mkdir(parents=True) cannot pass")
_cases.append(("error", 5, run_restore(
    _e / "openwebui" / "webui.db", _e / "snap" / "webui.db",
    _blocked_quar / "forensics")))

for _action, _want, (_rc, _out) in _cases:
    check(
        _rc == _want and _action in _out,
        f"{_action:24} -> exit {_want} (got rc={_rc}, action in output="
        f"{_action in _out})",
    )
check(
    sorted(a for a, _, _ in _cases) == sorted(webuidb.RESTORE_EXIT_CODES),
    f"AND THAT IS EVERY ROW IN THE TABLE - driven: "
    f"{sorted(a for a, _, _ in _cases)}, table: "
    f"{sorted(webuidb.RESTORE_EXIT_CODES)}. A row nothing drives is a value "
    f"that can be changed to 0 with every check still green, which is defect "
    f"2 reinstalled one branch at a time",
)
check(
    _restored_local.exists() and chats(_restored_local) == 40,
    f"CONTROL: the restore that exited 0 actually restored something "
    f"({chats(_restored_local) if _restored_local.exists() else 'absent'} "
    f"chats) - 'exit 0' proves nothing if nothing happened",
)

# ---------------------------------------------------------------------------
print()
print("[10] every action restore_on_boot can return has an exit code")
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
    "[8] and [9] cannot both be passing by accident",
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
print("[11] entrypoint.sh actually reads that exit code and refuses to boot")
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
# The SIBLING of the exit-code table, on the other side of the process
# boundary: a value in RESTORE_EXIT_CODES with no arm in the shell's `case`
# prints "webuidb.py reported a failure this script has no text for" to the
# one person trying to understand a pod that will not boot.
_case_at = ENTRY.index('case "${restore_rc}" in')
_case_arms = set(re.findall(r"^\s+(\d+)\)", ENTRY[_case_at:_case_at + 1500], re.M))
_nonzero = {str(v) for v in webuidb.RESTORE_EXIT_CODES.values() if v != 0}
check(
    _nonzero and not (_nonzero - _case_arms),
    f"every non-zero exit code has its own `case` arm with text for a human "
    f"(table: {sorted(_nonzero)}, arms: {sorted(_case_arms)})",
)
check(
    "*)" in ENTRY[_case_at:_case_at + 1500],
    "CONTROL: and an unrecognised code still lands in the `*)` arm - the "
    "arms above are the explanation, not the gate",
)

# ---------------------------------------------------------------------------
print()
print("[12] an empty local database plus a damaged snapshot is not 'kept_local'")
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

print("    CONTROL: a HEALTHY local beside a damaged snapshot is still kept_local")
# THE CONTROL ON THE OTHER AXIS, and the one that was missing. [12] refuses on
# a damaged snapshot; without this it passes just as happily if restore_on_boot
# refuses on a damaged snapshot NO MATTER WHAT LOCAL HOLDS - which would turn
# every ordinary restart with a stale or damaged file on /data into a pod that
# will not boot, while local held every conversation intact the whole time.
# The refusal is about local having nothing to serve, not about the snapshot
# being damaged.
wipe()
make_db(LOCAL, 12, "LOCAL-HAS-EVERYTHING")
make_db(SNAP, 40, "HER-REAL-HISTORY")
make_unreadable(SNAP)
r = webuidb.restore_on_boot()
check(
    r["action"] == "kept_local" and chats(LOCAL) == 12,
    f"a healthy local database wins outright even beside an unreadable "
    f"snapshot (action={r['action']}, local={chats(LOCAL)}) - nothing here "
    f"needs the snapshot to be readable. (Not because local is 'by "
    f"definition newer': it need not be, and an older one is refused at "
    f"PUBLISH time instead - test_webuidb_publish_guards.py [N5].)",
)

# ---------------------------------------------------------------------------
print()
print("[13] _reload_env forgets no refusal flag")
# The knob-sibling, and the reason it is worth a source check rather than a
# behaviour one: a flag that _reload_env forgets works perfectly from the
# environment at import and then silently ignores every test and every
# operator that sets it afterwards. It does not fail - it passes, having done
# nothing. [7] would go green on a flag that had stopped existing.
_reload = src[src.index("def _reload_env"):src.index("def _stamp")]
_flags_read = set(re.findall(
    r"""os\.environ\.get\(\s*["'](WEBUI_DB_ALLOW_\w+)["']""", src
))
check(
    len(_flags_read) >= 2,
    f"CONTROL: there is more than one refusal flag to check ({sorted(_flags_read)}) "
    f"- on a module with none, the assertion below is vacuously true and the "
    f"split in [7] has been quietly undone",
)
_unreloaded = sorted(f for f in _flags_read if f not in _reload)
check(
    not _unreloaded,
    f"every WEBUI_DB_ALLOW_* flag read at import is re-read by _reload_env "
    f"(missing: {_unreloaded})",
)

# ---------------------------------------------------------------------------
print()
print("[14] an EMPTY START refuses every publish, because the ratio only delays one")
# entrypoint.sh's escape-hatch banner claimed "The sync daemon stays ON, and
# that is safe rather than an oversight: every route out of this state is
# refused by webuidb.sync_once". It was not safe. The shrink guard is a RATIO,
# so an empty-started database is refused only until it grows past
# SHRINK_REFUSE_BELOW of the snapshot - one was demonstrated publishing over
# the snapshot at 51% of its size, holding NONE of her history. At ~2 MB/day
# against 41 MB the flag bought about ten days, and it is a RunPod template
# variable, so once set to bring a pod up it stays set through every redeploy.
#
# A marker file is what refuses now, and the behaviour was changed rather than
# the wording. The state below is exactly the demonstrated one: local has
# grown past half the snapshot and holds nothing that was in it.
wipe()
make_db(SNAP, 1, "HER-REAL-HISTORY", body=1_000_000)
make_db(LOCAL, 1, "EMPTY-START-THAT-GREW", body=600_000)
check(
    content(LOCAL) > content(SNAP) * webuidb.SHRINK_REFUSE_BELOW,
    f"PRECONDITION: local is already past the ratio ({content(LOCAL)} > "
    f"{content(SNAP)} * {webuidb.SHRINK_REFUSE_BELOW}) - so the shrink guard "
    f"would let this through, and a refusal below is the marker's doing "
    f"rather than a case that was already covered",
)
webuidb.EMPTY_START_MARKER.parent.mkdir(parents=True, exist_ok=True)
webuidb.EMPTY_START_MARKER.write_text("restore exited 4\n", encoding="utf-8")
snap_before = digest(SNAP)
# Counted, not globbed for emptiness: the shrink refusals in [5], [6] and [7]
# have already left their own forensic copies in this directory, so an
# absolute assertion here would be false for a reason that has nothing to do
# with this case - a check passing, or failing, for the wrong reason.
quar_before = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and digest(SNAP) == snap_before,
    f"refused, and the snapshot is byte-for-byte untouched "
    f"(synced={r['synced']})",
)
check(
    r["error"] and str(webuidb.EMPTY_START_MARKER) in r["error"],
    "and the refusal names the exact file to delete - the escape is one "
    "deliberate `rm`, not an env var that survives onto the next template",
)
check(
    len(list(_QUAR.glob(f"{LOCAL.name}.refused-*"))) == quar_before,
    f"and NOTHING NEW was copied to quarantine (still {quar_before}). This "
    f"refusal repeats every SYNC_INTERVAL_S until a human acts; copying the "
    f"whole database onto /data each cycle (~11.8 GB/day at 41 MB) would fill "
    f"the volume whose capacity failure _set_aside was just hardened against",
)

print("    CONTROL: with no marker, that same database publishes")
# Without this, [14] passes if sync_once has simply stopped publishing.
wipe()
make_db(SNAP, 1, "HER-REAL-HISTORY", body=1_000_000)
make_db(LOCAL, 1, "EMPTY-START-THAT-GREW", body=600_000)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True,
    f"the marker is what refuses, not the state (synced={r['synced']}, "
    f"error={r['error']})",
)

print("    CONTROL: a marker with NO snapshot still publishes")
# The absent-vs-unreadable confusion this module was repaired for, one branch
# later: with no durable copy to protect, refusing means never making one.
wipe()
make_db(LOCAL, 1, "EMPTY-START-THAT-GREW", body=600_000)
webuidb.EMPTY_START_MARKER.write_text("restore exited 4\n", encoding="utf-8")
check(not SNAP.exists(), "PRECONDITION: there is no snapshot to protect")
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True,
    f"the first publish onto a bare volume still works (synced={r['synced']}, "
    f"error={r['error']})",
)

print("    CONTROL: a successful --restore clears the marker")
# The anti-permanence property, and the reason this is not the R11 defect
# rebuilt: a restore means local now holds exactly what the snapshot holds, so
# publishing it back can lose nothing, and the operator who repairs the
# snapshot gets their sync daemon back without knowing about a dotfile.
wipe()
webuidb.EMPTY_START_MARKER.write_text("restore exited 3\n", encoding="utf-8")
make_db(SNAP, 40, "HER-REAL-HISTORY")
r = webuidb.restore_on_boot()
check(
    r["action"] == "restored_from_snapshot"
    and not webuidb.EMPTY_START_MARKER.exists(),
    f"the marker is gone after a real restore (action={r['action']}, marker "
    f"exists={webuidb.EMPTY_START_MARKER.exists()})",
)
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"and publishing works again (error={r['error']})")

# ---------------------------------------------------------------------------
print()
print("[15] entrypoint.sh writes that marker, and fails closed if it cannot")
# The other side of [14]'s process boundary. A marker nothing writes is a
# guard that no-ops while the banner above it tells the operator they are
# protected - which is worse than no guard, because it is believed.
_hatch_at = ENTRY.index('echo "      WEBUI_DB_ALLOW_EMPTY_START=true - starting anyway."')
_hatch = ENTRY[_hatch_at:_hatch_at + 3000]
check(
    ".empty-start" in _hatch,
    "the empty-start branch writes the marker webuidb.py refuses on",
)
check(
    'empty_start_marker="$(dirname "${WEBUI_LOCAL_DB}")/.empty-start"' in _hatch,
    "ON LOCAL DISK, derived from WEBUI_LOCAL_DB - a marker on /data would "
    "outlive the incident and become the permanent block on publishing that "
    "this whole series is removing; on the overlay it dies with the pod",
)
check(
    webuidb.EMPTY_START_MARKER == LOCAL.with_name(".empty-start"),
    f"and the two sides agree on the path ({webuidb.EMPTY_START_MARKER} vs "
    f"{LOCAL.with_name('.empty-start')}) - a shell writing one name and a "
    f"daemon reading another is a guard that is off and looks on",
)
# Anchored on the write itself and bounded by a character count, the way [11]
# bounds the ALLOW_EMPTY_START refusal: hunting for a matching `fi` picks up
# whatever else has moved into range, which is how test_webuidb_gate.py's
# positional slice once swallowed a pg_ctl stop.
_write_at = _hatch.index('> "${empty_start_marker}"')
_write_block = _hatch[_write_at:_write_at + 700]
check(
    "if ! printf" in _hatch[:_write_at] and "exit 1" in _write_block,
    f"and a marker that CANNOT be written stops the boot - a guard that "
    f"silently no-ops when the disk says no leaves a pod that comes up "
    f"looking protected and is not (redirect guarded by `if !`: "
    f"{'if ! printf' in _hatch[:_write_at]}, exit 1 follows: "
    f"{'exit 1' in _write_block})",
)
check(
    "every route out of this state is refused" not in ENTRY,
    "and the false claim is gone from the banner: the sync daemon did NOT "
    "refuse every route out of this state, it refused them until local grew "
    "past half the snapshot and then published over it at 51%",
)

# ---------------------------------------------------------------------------
print()
print("[16] a snapshot that cannot be STAT'ED is refused, not read as 'fresh'")
# hostile pass #2, MEDIUM. Path.exists() swallows every OSError and answers
# False for both "nothing there" and "I could not look" - and on the volume
# whose read reliability is this module's entire subject, restore_on_boot's
# fresh-vs-restore decision read those as the SAME thing. An unstatable
# snapshot fell straight through to action="fresh", exit 0, OpenWebUI builds
# an empty schema, and because the exit code was 0 entrypoint.sh never wrote
# .empty-start - the file its own banner calls "what protects /data, not the
# shrink ratio". See _presence().
#
# The hostile pass's own proof (C10) used an ENOTDIR path (a directory
# component that is actually a file) under Linux, where stat() raises
# NotADirectoryError. VERIFIED NOT PORTABLE: on Windows the identical setup
# raises FileNotFoundError instead (WinError 3 reads as "the path does not
# exist", not "I could not check") - so that specific reproduction is
# platform-dependent for a reason that has nothing to do with the code under
# test. A fake stat() that raises a chosen OSError tests the same property
# (restore_on_boot must not collapse "cannot stat" into "absent") without
# depending on what a given OS reports for a filesystem shape no test
# environment can make a real stalled mount produce anyway.


class _UnstatableSnapshot:
    """Stands in for webuidb.SNAPSHOT_DB: .stat() always raises the given
    OSError, simulating a mount that answers an error rather than 'found'
    or 'not found'. Everything restore_on_boot does with a snapshot that
    DOES exist is unreachable through this fake (it only reaches
    _presence(), which calls .stat() and nothing else), which is exactly
    the point - this proves the FIRST branch point, not the rest of the
    restore."""

    def __init__(self, exc: OSError):
        self._exc = exc

    def stat(self):
        raise self._exc

    def __str__(self):
        return "<unstatable-snapshot-for-testing>"


wipe()  # LOCAL absent too - this is exactly the fresh-vs-restore fork
_orig_snapshot_db = webuidb.SNAPSHOT_DB
webuidb.SNAPSHOT_DB = _UnstatableSnapshot(
    OSError(5, "simulated I/O error: stalled mount")
)
try:
    r = webuidb.restore_on_boot()
finally:
    webuidb.SNAPSHOT_DB = _orig_snapshot_db
check(
    r["action"] == "error",
    f"refuses to boot rather than guess (action={r['action']!r})",
)
check(
    r["action"] != "fresh",
    "and specifically does NOT take the dangerous wrong answer - 'fresh' "
    "means OpenWebUI builds an empty schema and RESTORE_EXIT_CODES['fresh'] "
    "is 0, so entrypoint.sh would never reach the branch that writes "
    ".empty-start",
)
check(
    webuidb.RESTORE_EXIT_CODES.get(r["action"]) == 5,
    f"and that action exits 5 (error), the same code the OTHER unexpected-"
    f"filesystem-state branches in this function use (got "
    f"{webuidb.RESTORE_EXIT_CODES.get(r['action'])})",
)

print("    CONTROL: a snapshot that genuinely does not exist is still 'fresh'")
# Without this, [16] could be passing because restore_on_boot now refuses
# EVERY boot with no local database - which would "fix" the finding by
# breaking every brand-new deployment instead.
wipe()
r = webuidb.restore_on_boot()
check(
    r["action"] == "fresh",
    f"a real first boot (nothing anywhere) is still 'fresh', unchanged "
    f"(action={r['action']!r})",
)

print("    CONTROL: a snapshot that genuinely exists still restores normally")
wipe()
make_db(SNAP, 40, "HER-REAL-HISTORY")
r = webuidb.restore_on_boot()
check(
    r["action"] == "restored_from_snapshot" and chats(LOCAL) == 40,
    f"an ordinary healthy snapshot still restores (action={r['action']!r}, "
    f"chats={chats(LOCAL) if LOCAL.exists() else None!r})",
)


print()
if FAILED:
    print("!" * 66)
    for f in FAILED:
        print("FAIL " + f)
    print("!" * 66)
    sys.exit(1)
print("All webui.db restore/publish guard checks passed.")
