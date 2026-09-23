"""
compactor.webuidb — keep OpenWebUI's SQLite database off the network volume.

THE PROBLEM. RunPod's MooseFS mount (`/data`) drops I/O occasionally. When
that lands while OpenWebUI is mid-transaction, SQLite leaves a hot rollback
journal; every subsequent open tries to roll it back, rolling back requires
WRITING, the write fails, and SQLite reports "attempt to write a readonly
database" plus disk I/O errors on plain SELECTs. Twice on 2026-08-31 (02:17
and ~04:30) that took the whole front end down.

The evidence that this is a local-vs-network problem and not a SQLite one:
through both incidents the compactor's own state on the SAME volume was
untouched — 2,080 facts, 749 indexed exchanges, zero unreadable. Many small
whole-file JSON writes survive a stalled mount. One large, continuously
journalled SQLite file does not.

THE SHAPE OF THE FIX.

    OpenWebUI  ->  /var/lib/openwebui/webui.db     (LOCAL disk, overlay)
                          |
                          |  sync_once(), every SYNC_INTERVAL_S
                          v
                   /data/openwebui/webui.db         (MooseFS, durable)

The live database and every journal it writes live on local disk, where
writes do not fail. `/data` holds a periodically refreshed snapshot, which is
what survives the pod being recreated. The snapshot keeps OpenWebUI's
original path, so reverting DATABASE_URL is a complete rollback.

NOT A SYMLINK, deliberately. Symlinking /data/openwebui/webui.db to local
disk looks equivalent and is not: SQLite derives the journal path from the
path it was given, so the journal can land next to the LINK — back on
MooseFS — and the failure we are fixing survives the fix. OpenWebUI is
pointed at a real local path via DATABASE_URL instead.

THE TRADE, stated plainly. Local disk is not persistent across pod
recreation, so SYNC_INTERVAL_S is the RPO for chat history. The daily/hourly
archive in backup.py is the second line. Postgres removes this trade
entirely and is the strategic answer; this removes the instability today.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import sys
import time
from pathlib import Path

from envcfg import env_float, env_int

logger = logging.getLogger("compactor.webuidb")

# The live database: local disk, NOT /data.
LOCAL_DB = Path(os.environ.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db"))
# The durable snapshot: OpenWebUI's original path, so a rollback is just
# unsetting DATABASE_URL.
SNAPSHOT_DB = Path(
    os.environ.get("WEBUI_SNAPSHOT_DB", "/data/openwebui/webui.db")
)
SYNC_INTERVAL_S = env_float("WEBUI_DB_SYNC_INTERVAL_S", 300)
# Where a local database that fails its integrity check is set aside. Never
# deleted: this project's rule is that anything removing state is reversible.
QUARANTINE = Path(os.environ.get("WEBUI_DB_QUARANTINE", "/data/forensics"))

SIDECARS = ("-journal", "-wal", "-shm")


def live_webui_db() -> Path:
    """Where OpenWebUI's live database is RIGHT NOW - LOCAL_DB or
    SNAPSHOT_DB - read from the gate at CALL time. Not a guess.

    v3.1.9 (hostile pass #2, A3-4/A3-9, "the webuidb half"). `backup.py`
    picked the live database with `_LIVE_DB.exists()` instead of asking
    this gate: entrypoint.sh's `WEBUI_DB_LOCAL` decides where OpenWebUI's
    database physically lives, and the string `WEBUI_DB_LOCAL` did not
    appear anywhere in backup.py. The trigger is this project's own
    documented rollback - setting the flag `false` repoints DATABASE_URL at
    the snapshot but leaves the ABANDONED local file sitting on the
    container overlay, and a container restart (not a redeploy) keeps that
    overlay, so the stale file still `exists()`. Every nightly backup
    archived a database nobody had written to since the flag flipped, and
    the CLI `--restore` wrote an archive back ONTO that same abandoned
    file - OpenWebUI, reading the snapshot, never saw it.

    A3-9 is the sibling half of the same defect: `backup.py` read this at
    IMPORT and froze it in a module constant, so even the two processes
    that live for the whole life of the pod (`backup --daemon` and
    `main.py`, both started once by supervisord and never re-imported)
    could not see the flag change during their own lifetime - a constant
    cannot express "where the live database is right now" in a process the
    deployment's own kill switch can repoint under it.

    So: no caching, no import-time freeze, no existence guess. Read fresh,
    every call, and fold the value the SAME WAY entrypoint.sh's own
    WEBUI_DB_LOCAL NORMALIZATION block does (trimmed, case-insensitive,
    1/yes/on and 0/no/off) so this function and the boot script always
    agree on what a given spelling means.

    UNRECOGNISED RAISES, deliberately, rather than guessing between the two
    files the way this function exists to stop doing. In production this
    is unreachable: entrypoint.sh normalises WEBUI_DB_LOCAL to an exact
    "true" or "false" and refuses to BOOT on anything else (see its own
    block), so by the time any child process - including this one - can
    run, the environment already carries a canonical value. Reaching this
    branch means the caller is running outside that boot path entirely (a
    bare CLI invocation, a test with a stray value) - exactly the situation
    where silently picking a file has, historically, been how this defect
    shipped.
    """
    raw = os.environ.get("WEBUI_DB_LOCAL", "").strip().lower()
    if raw in ("", "true", "1", "yes", "on"):
        return LOCAL_DB
    if raw in ("false", "0", "no", "off"):
        return SNAPSHOT_DB
    raise RuntimeError(
        f"WEBUI_DB_LOCAL={raw!r} is neither a recognised true spelling "
        f"(true/1/yes/on) nor a recognised false spelling (false/0/no/off). "
        f"entrypoint.sh refuses to boot on exactly this state, so reaching "
        f"here means this process started outside that boot path. Refusing "
        f"to guess which physical file is 'live' - that guess is the "
        f"defect this function replaces (hostile pass #2, A3-4/A3-9)."
    )


# REGRESSION GUARD. Refuse to publish a snapshot holding less than this
# fraction of the chats the previous snapshot had.
#
# Found by adversarial test A2, and it is the one way this whole design can
# destroy her history: restore_on_boot fails (unreadable snapshot, full local
# disk, unwritable directory), OpenWebUI starts anyway and builds an empty
# schema, she sends ONE message - and the next sync cheerfully publishes a
# 1-chat database over the 400-chat snapshot that is the only durable copy.
# Every individual step behaves correctly. The composition loses everything.
#
# 0.5 is deliberately loose: ordinary use never halves a chat count, so this
# fires on catastrophe rather than on housekeeping. Deleting a few
# conversations must still reach the snapshot (A4) or the guard would quietly
# stop backing her up, which is its own data-loss mode.
SHRINK_REFUSE_BELOW = env_float("WEBUI_DB_SHRINK_REFUSE_BELOW", 0.5)
# The floor below which a ratio on chat COUNT says nothing. It was 10, and
# that is the number that disarmed this guard on the only pod it protects.
#
# The comparison already declines to fire at the small end without any help:
# at previous == 2 a surviving 1 chat is exactly half, not below it, and at
# previous == 1 every possible survivor is >= half of one. So a floor of 2
# costs nothing the arithmetic was not refusing anyway, and every value above
# it is the floor DELETING protection rather than adding sense to it. 10
# deleted all of it. This is a single-user pod where one conversation carries
# most of the traffic, so `chat` can sit in the single digits, and
# `previous >= 10` then makes the whole guard unreachable - permanently, and
# silently, in exactly the deployment it was written for. A FLOOR THAT
# EXCEEDS THE REAL ROW COUNT IS NOT A FLOOR, IT IS AN OFF SWITCH.
SHRINK_GUARD_MIN_CHATS = env_int("WEBUI_DB_SHRINK_GUARD_MIN_CHATS", 2)
# ...and here the row count is the wrong UNIT as well as the wrong floor, so
# there is a second measure. OpenWebUI stores an ENTIRE CONVERSATION as one
# JSON blob in one row of `chat` (scripts/chat-metrics.py reads it that way:
# `select id, updated_at, chat from chat`), and the conversation on this pod
# is tens of megabytes. A failed restore that leaves one freshly created
# conversation behind is therefore 1 row against 1 row - which NO ratio on
# counts can ever catch, at any floor - and tens of kilobytes against tens of
# megabytes, which this catches on the first sync.
#
# THE SECOND MEASURE IS STORED CONTENT, NOT FILE SIZE, AND THE DIFFERENCE IS
# THE DIFFERENCE BETWEEN A GUARD AND AN OUTAGE. This shipped comparing
# os.stat().st_size on the claim that "both files are written by sqlite3's
# backup API, which copies live pages only ... free-list churn and VACUUM do
# not move it". Both halves of that are false and were demonstrated false.
# sqlite3_backup_step copies EVERY page of the source, free-list pages
# included, so a database with 90% of its rows deleted backs up 41.026 MB ->
# 41.026 MB (ratio 1.0000) while a VACUUM of the same content gives 4.108 MB
# (ratio 0.1001). Driven against this very function: 40.07 MB -> VACUUM ->
# 10.03 MB -> REFUSED, five chats on both sides, not one byte of her history
# gone.
#
# AND VACUUM IS IN THIS REPO'S OWN RECOVERY PATH. scripts/recover-webui-db.py
# runs `con.execute("VACUUM")`; OPERATIONS.md tells the operator to
# integrity_check, VACUUM and copy back; and the refusal below names
# recover-webui-db.py in its own text. So the file-size measure blocked the
# documented recovery - every SYNC_INTERVAL_S, forever - and the only escape
# it offered was a flag that also disarmed the corruption guard.
#
# So compare what is STORED: the summed length IN BYTES of every column of
# `chat` (_content_bytes). A VACUUM rewrites the whole file and cannot change
# that sum; neither can free-list churn, page size, auto_vacuum, or the backup
# API. Deleting a conversation does, which is the entire point.
#
# st_size USED to survive as a pre-filter for this floor, skipping the table
# scan for a small snapshot. v3.1.9 removed it: the per-conversation and
# generation guards (MAX_ROW_LOSS_BYTES, ALLOW_OLDER_GENERATION) read the
# same scan and have no floor, so the pre-filter no longer saved anything but
# the scan of a file under 64 KB.
SHRINK_GUARD_MIN_BYTES = env_int("WEBUI_DB_SHRINK_GUARD_MIN_BYTES", 65536)
# The deliberate override for the SHRINK refusal, and for nothing else: she
# really did clear her history and the snapshot must follow. Refusing forever
# would be its own failure.
#
# IT USED TO COVER THE UNREADABLE-SNAPSHOT REFUSAL TOO, on the reasoning that
# "from the operator's side they are one decision". They are not one decision,
# and the coupling is not a style point: THE SHRINK REFUSAL'S OWN TEXT TELLS
# THE OPERATOR TO SET THIS FLAG. So an operator who really did delete a
# conversation follows the instruction printed for them and, in the same
# keystroke, disarms the guard that stops a 1-chat database being published
# over her only durable copy - the guard this module exists for. An opt-in to
# a known-good shrink must not be an opt-out of corruption protection.
ALLOW_SHRINK = (
    os.environ.get("WEBUI_DB_ALLOW_SHRINK", "").strip().lower()
    in ("1", "true", "yes")
)
# The SIBLING override, separate on purpose (see above). It opens the
# unreadable-previous-snapshot refusal and nothing else, and nothing this
# module prints for any other reason names it - so if it is set, somebody read
# THAT refusal and decided about THAT risk.
ALLOW_PUBLISH_OVER_UNREADABLE = (
    os.environ.get("WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE", "").strip().lower()
    in ("1", "true", "yes")
)
# PER-CONVERSATION LOSS LIMIT (v3.1.9, hostile pass #2, N4). Refuse when any
# conversation that EXISTS ON BOTH SIDES lost more than this many bytes of
# stored content, whatever the table-wide ratio says.
#
# The ratio above is the wrong instrument for this pod's table, and not by a
# margin. OpenWebUI keeps a whole conversation in ONE row of `chat`, and the
# live one was measured at 32.95 MB, so "1 row -> 1 row" is every sync, and
# a row that loses 49.5% of itself leaves the table above 0.5. Demonstrated:
# 2,020 of 4,000 messages gone from her only conversation, content ratio
# 0.505, synced=True, nothing above INFO - 16.3 MB per cycle at the live size,
# and repeatable on the remainder next cycle. The 0.5 above was justified
# entirely in counts of conversations ("ordinary use never halves a chat
# count"); nothing reasoned about content INSIDE one.
#
# Absolute, not a second ratio, because the thing being protected is an
# amount of her history, and a per-row ratio loose enough for a 30 KB chat
# would wave through megabytes of the 33 MB one. A row that is GONE is not
# covered here: that is a deletion, and deletions stay governed by the chat
# count and the content ratio (a missing row counted as "shrank to zero"
# would refuse deleting any large conversation).
#
# THE COST, named: deleting a long branch of her one conversation - more than
# about a megabyte, a few hundred long messages - is refused too, every
# SYNC_INTERVAL_S, until a human publishes it once (see ALLOW_ROW_LOSS). No
# measure of size can tell "she deleted it" from "it was truncated"; a person
# can, and the refusal tells them how. Clamped at 0: a negative limit would
# refuse every unchanged row.
MAX_ROW_LOSS_BYTES = max(0, env_int("WEBUI_DB_MAX_ROW_LOSS_BYTES", 1_000_000))
# The override for the per-row refusal and for nothing else - same split, same
# reason, as the two above: an operator who sets the flag the SHRINK refusal
# names must not disarm this one in the same keystroke, and the other way
# round. The refusal text tells the operator to set it for ONE invocation
# (`WEBUI_DB_ALLOW_ROW_LOSS=1 webuidb.py --sync-once`), not on the RunPod
# template, where it would outlive the deletion it was set for.
ALLOW_ROW_LOSS = (
    os.environ.get("WEBUI_DB_ALLOW_ROW_LOSS", "").strip().lower()
    in ("1", "true", "yes")
)
# The override for the OLDER-GENERATION refusal (N5), and for nothing else.
# Its legitimate use is exactly one act: an operator restored an archive on
# purpose and needs that older database to become the durable copy. Same
# one-invocation advice as ALLOW_ROW_LOSS.
ALLOW_OLDER_GENERATION = (
    os.environ.get("WEBUI_DB_ALLOW_OLDER_GENERATION", "").strip().lower()
    in ("1", "true", "yes")
)
# THE SHRINK REFUSAL'S FORENSIC COPY, RATE-LIMITED (hostile-lane fix, found
# by SP\lane-webuidb.md "Found, not fixed" #3, never a numbered N-finding of
# its own). "Keep the refused database" below used to run unconditionally,
# on EVERY cycle a shrink refusal (lost_chats or lost_content) stayed
# tripped, and a refusal is not a one-shot event - it repeats every
# SYNC_INTERVAL_S until an operator acts. At the measured live size (33 MB)
# and the default interval (300 s) that is ~9.5 GB/day copied onto the
# MooseFS volume this module exists to protect, for as long as the
# regression goes unaddressed - which, un-noticed, is indefinitely. The
# guard was defending her history from a bad publish and, in the same
# breath, filling the disk that publish lives on.
#
# The FIRST copy of a new refusal is still unconditional (see sync_once): it
# is the earliest evidence of whatever went wrong, and this module's own
# rule is that nothing protective is deleted or skipped outright (compare
# _set_aside: "never delete"). What is bounded is every copy AFTER that, for
# the SAME ongoing refusal, which proves nothing the first copy did not
# already prove and costs the same 33 MB again. Default 3600s (one hour)
# deliberately matches sync_loop's own "shout on the first, then hourly
# forever" cadence for consecutive publish failures, a few lines down in
# this same file - one operator-facing rhythm for this module's two
# stuck-in-a-loop alarms rather than two unrelated numbers. At the default
# SYNC_INTERVAL_S (300s) that bounds the cost to 33 MB/hour (~792 MB/day)
# worst case instead of ~9.5 GB/day, while a slowly worsening state still
# leaves an hourly trail of samples rather than one.
#
# time.monotonic(), never time.time() — deliberately, given this exact file
# already shipped one wall-clock defect (N2, a snapshot mtime in the
# future). The question this interval answers is only "how long has this
# process been retrying", which monotonic answers correctly through a clock
# step in either direction; wall-clock arithmetic would not, and getting the
# rate limit itself wrong on the flakiest volume in the system would be a
# poor place to reintroduce that bug.
FORENSIC_COPY_MIN_INTERVAL_S = max(
    0.0, env_float("WEBUI_DB_FORENSIC_COPY_MIN_INTERVAL_S", 3600)
)
# Monotonic timestamp of the last forensic copy this process made, or None
# before the first one. Deliberately NOT reset by _reload_env: that function
# re-reads ENV-DRIVEN KNOBS, and this is runtime state, not a knob - a test
# that wants a clean streak sets this back to None directly (white-box,
# consistent with how the suites already reach into EMPTY_START_MARKER and
# the ALLOW_* globals), the same way a real process only gets a clean streak
# from actually restarting.
_forensic_copy_last_monotonic: float | None = None
# How far in the future a snapshot's mtime may be before the skip stops
# trusting it (N2). Not a knob: a few seconds ahead is filesystem timestamp
# granularity or ordinary skew and costs one skipped cycle at most, and an
# hour ahead is a clock that was wrong. Without any tolerance, a filesystem
# that rounds an mtime UP would turn every cycle into a whole-database write
# onto /data - the cost the skip exists to avoid.
SNAPSHOT_MTIME_FUTURE_TOLERANCE_S = 60
# Written by entrypoint.sh when a restore failed and the operator set
# WEBUI_DB_ALLOW_EMPTY_START=true to boot anyway, onto a schema OpenWebUI
# builds from nothing. While it exists, sync_once refuses outright.
#
# It exists because the banner entrypoint.sh printed for that flag claimed
# "the sync daemon stays ON, and that is safe ... every route out of this
# state is refused by webuidb.sync_once". That was false, and measurably so:
# the shrink refusal is a RATIO, so an empty-started database is refused only
# until it grows past half the snapshot, and one was demonstrated publishing
# over the snapshot at 51% of its size. At ~2 MB/day against a 41 MB snapshot
# the flag bought about ten days, not safety - and it is a RunPod template
# variable, so once set to get a pod up it stays set through every later
# redeploy, printing its banner into a boot log nobody re-reads.
#
# ON LOCAL DISK, NOT /data, AND THAT IS THE POINT. A marker on the volume
# would outlive the incident and become the thing that blocks publishing
# forever - the defect this file is being repaired for, reintroduced by its
# own fix. On the overlay it dies with the pod, so it can only ever describe
# THIS boot. There is no env var to override it either: the escape is deleting
# the file the refusal names, which is a single act that cannot survive into
# the next incident the way an env var on a template does.
EMPTY_START_MARKER = Path(
    os.environ.get("WEBUI_DB_EMPTY_START_MARKER", "")
    or LOCAL_DB.with_name(".empty-start")
)

# What `--restore` exits with, per action restore_on_boot() can return. It is
# a module constant and not a dict literal buried in __main__ so that a test
# can assert every action string in the function appears here: see
# test_webuidb_restore_guard.py, which reads this module's own source to do
# it. An action missing from this table exits non-zero (see __main__), never
# zero - a new branch that forgets the table gets a refused boot rather than a
# silent pass.
RESTORE_EXIT_CODES = {
    "kept_local": 0,
    "restored_from_snapshot": 0,
    # A genuinely new deployment: no local database and no snapshot, so there
    # is nothing to lose and OpenWebUI building its own schema is correct.
    "fresh": 0,
    "snapshot_unhealthy": 3,
    "restore_failed": 4,
    "error": 5,
    # p3-b F4/F9: an interrupted backup.py::restore_backup() left an
    # in-flight marker — see find_interrupted_restore(). Reuses "error"'s
    # exit code (5) DELIBERATELY, not a new number: entrypoint.sh's `case`
    # arm per value (test_webuidb_restore_guard.py [11]) lives in
    # entrypoint.sh, which is outside this fix's file list (another lane
    # owns it concurrently) — a genuinely distinct code needs a matching
    # `case` arm added there. Reusing 5 costs only log-grep precision
    # (marker vs. another startup error look the same in the exit code
    # alone; the log LINE still names the marker) and needs no other
    # lane's file. See this finding's fix-lane report for the exact arm
    # to add if a distinct code is wanted later.
    "restore_interrupted": 5,
}


def _reload_env() -> None:
    """Re-read the env-driven knobs. For tests, and for anyone who changes
    them without restarting the process."""
    global SHRINK_REFUSE_BELOW, SHRINK_GUARD_MIN_CHATS, ALLOW_SHRINK
    global SHRINK_GUARD_MIN_BYTES, ALLOW_PUBLISH_OVER_UNREADABLE
    global SYNC_INTERVAL_S, MAX_ROW_LOSS_BYTES, ALLOW_ROW_LOSS
    global ALLOW_OLDER_GENERATION, FORENSIC_COPY_MIN_INTERVAL_S
    SHRINK_REFUSE_BELOW = env_float("WEBUI_DB_SHRINK_REFUSE_BELOW", 0.5)
    SHRINK_GUARD_MIN_CHATS = env_int("WEBUI_DB_SHRINK_GUARD_MIN_CHATS", 2)
    # The sibling knob. Every constant this function forgets is a knob that
    # works from the environment at import and then silently ignores a test
    # (or an operator) that sets it afterwards.
    SHRINK_GUARD_MIN_BYTES = env_int("WEBUI_DB_SHRINK_GUARD_MIN_BYTES", 65536)
    ALLOW_SHRINK = (
        os.environ.get("WEBUI_DB_ALLOW_SHRINK", "").strip().lower()
        in ("1", "true", "yes")
    )
    # The other half of the split override, and the one this function is most
    # likely to be missing next time somebody reads it: a flag reloaded here
    # and a flag that is not behave identically until the exact moment a test
    # sets the second one, which then does nothing and passes. Both refusal
    # flags are reloaded; test_webuidb_restore_guard.py [13] reads this
    # function's own source to keep it that way.
    ALLOW_PUBLISH_OVER_UNREADABLE = (
        os.environ.get("WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE", "").strip().lower()
        in ("1", "true", "yes")
    )
    SYNC_INTERVAL_S = env_float("WEBUI_DB_SYNC_INTERVAL_S", 300)
    # v3.1.9 (N4, N5): the per-row limit and both of its siblings' overrides.
    # test_webuidb_publish_guards.py [13b] reads this function's source for
    # every WEBUI_DB_* knob read at import, not only the ALLOW_ flags.
    MAX_ROW_LOSS_BYTES = max(0, env_int("WEBUI_DB_MAX_ROW_LOSS_BYTES", 1_000_000))
    ALLOW_ROW_LOSS = (
        os.environ.get("WEBUI_DB_ALLOW_ROW_LOSS", "").strip().lower()
        in ("1", "true", "yes")
    )
    ALLOW_OLDER_GENERATION = (
        os.environ.get("WEBUI_DB_ALLOW_OLDER_GENERATION", "").strip().lower()
        in ("1", "true", "yes")
    )
    # The forensic-copy rate limit (see its own comment at the definition).
    # Deliberately does NOT reset _forensic_copy_last_monotonic - that is
    # runtime state, not a knob; see the comment beside it.
    FORENSIC_COPY_MIN_INTERVAL_S = max(
        0.0, env_float("WEBUI_DB_FORENSIC_COPY_MIN_INTERVAL_S", 3600)
    )


def _stamp() -> str:
    # Millisecond precision, not seconds: two set-asides inside the same
    # second collided on the filename and the second silently overwrote the
    # first. These files exist because something already went wrong; losing
    # one to a name clash is exactly the wrong time for that.
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def integrity(path: Path) -> tuple[bool, str]:
    """(ok, detail). Opening also replays/rolls back any journal, which on
    local disk always succeeds — that is the whole point of this module."""
    if not path.exists():
        return False, "missing"
    try:
        con = sqlite3.connect(str(path), timeout=30)
        try:
            verdict = con.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            con.close()
        return verdict == "ok", verdict
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _presence(path: Path) -> tuple[bool, bool]:
    """(there, unstatable). `Path.exists()` does NOT have one uniform
    behaviour to rely on, and BOTH of its behaviours are wrong for this
    module on their own:

      * on CPython up to 3.7, `Path.exists()` caught every `OSError` and
        answered `False` for both "nothing there" and "I could not look".
      * on Python 3.12.3 (every image this project ships), it answers
        `False` only for `ENOENT`/`ENOTDIR`/`EBADF`/`ELOOP`, and RAISES for
        anything else — `EIO`, `ETIMEDOUT`, `ENOTCONN`, `EACCES` (p3-b F7).
        A version of this docstring used to claim the OLD (pre-3.7)
        behaviour applied here unconditionally; it does not, on the
        interpreter this actually runs on, and a caller that believed it
        (sync_once's mtime-skip check, before p3-b F7) crashed the whole
        process on an EIO instead of getting a clean `False`.

    Neither answer is safe alone: the old behaviour collapses "absent" and
    "unreadable" into the same `False`, and on the volume whose read
    reliability is this module's entire subject those two answer
    restore_on_boot's fresh-vs-restore decision in OPPOSITE directions; the
    3.12 behaviour instead lets an unreadable mount escape as an uncaught
    exception. This function is the one place that decides, explicitly,
    and never raises.

    Hostile pass #2, C10, proved the collapsed-`False` shape reachable: an
    unstatable snapshot path (`ENOTDIR`/`EIO` — a stalled or erroring
    MooseFS mount, once `entrypoint.sh` has already confirmed /data itself
    is writable, is the production shape; the test uses `ENOTDIR` because a
    root container cannot be denied by mode bits) made
    `SNAPSHOT_DB.exists()` answer `False`, which restore_on_boot's caller
    read as "no snapshot", falling straight through to `action="fresh"`.
    `RESTORE_EXIT_CODES["fresh"] = 0`, so entrypoint.sh never reached the
    branch that writes `.empty-start` — the file its own banner calls "what
    protects /data, not the shrink ratio". OpenWebUI built an empty schema
    with nothing guarding it but a ratio the same banner says "only delays
    an empty database, it does not stop one" (about ten days at the measured
    live size).

    This is also `_has_rows`'s own docstring arguing the reader into the
    bug: "this function cannot tell you whether the file is ABSENT — only
    path.exists() can". `path.exists()` cannot either — that claim is what
    this function replaces.

    unstatable=True means "the OS refused to say" — the caller must refuse
    rather than guess between "restore" and "fresh"; see restore_on_boot.
    EVERY caller that decides something from a path's presence on /data
    must go through this function, never a bare `.exists()` — p3-b F7 found
    two more (sync_once's mtime-skip check, restore_backup's rollback
    moves) that had not."""
    try:
        path.stat()
        return True, False
    except FileNotFoundError:
        return False, False
    except OSError:
        return False, True


# p3-b F4/F9. backup.py::restore_backup's multi-step swap (set db aside,
# land the new one, integrity-check it, set the store aside, land the new
# store) has no single atomic operation covering it — a SIGKILL (a RunPod
# redeploy, an OOM kill, a closed terminal) or an EIO between any two of
# those steps can leave the live db missing, the live store missing, or a
# MIXED generation (a NEW db beside an OLD store or vice versa), with
# nothing on disk recording that a restore was ever in flight. The next
# boot then starts normally onto whatever that half-finished state happens
# to be — silently, since nothing refuses.
#
# Full transactional recovery (resume or fully undo an interrupted restore
# from any step) is out of scope for this fix: it needs the same
# step-by-step "what was I doing" log AND replay logic on both restore_
# backup and restore_on_boot, which is a bigger change than this lane's
# remaining budget covers. What IS implemented: a marker, written to the
# SAME filesystem restore_backup already stages onto (QUARANTINE, i.e.
# /data/forensics — never local disk, which a redeploy does not keep)
# BEFORE the first live-path move, naming exactly what was about to
# happen; removed only once every live move restore_backup attempted is
# known to have either landed or been rolled back; and a LOUD refusal at
# the next boot (restore_on_boot, below) while one is present, with the
# marker's own content as the recovery instruction. A human reads it and
# finishes the move by hand — see OPERATIONS.md (this lane cannot edit it,
# so the note is not there yet; see the fix report). That is "detect and
# refuse", not "recover automatically" — see this finding's report for
# exactly what remains.
_RESTORE_MARKER_GLOB = "restore-*.inprogress"


def _restore_marker_path(stamp: str) -> Path:
    return QUARANTINE / f"restore-{stamp}.inprogress"


def write_restore_marker(stamp: str, plan: dict) -> Path | None:
    """Write the in-flight marker for a restore_backup() run, atomically.
    `plan` is whatever restore_backup wants recorded — target/aside/staged
    path names — and is written back out VERBATIM by find_interrupted_
    restore() for a human to read. Returns the marker path, or None if it
    could not be written (logged loudly; restore_backup proceeds anyway —
    refusing the restore because the ONE THING THAT COULD LATER DETECT A
    KILL could not be written would be worse than the gap it is meant to
    close).
    """
    try:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
        path = _restore_marker_path(stamp)
        tmp = path.with_suffix(".inprogress.tmp")
        payload = {"stamp": stamp, "written_at": time.time(), "plan": plan}
        tmp.write_bytes(json.dumps(payload, indent=1).encode("utf-8"))
        os.replace(tmp, path)  # atomic on the same filesystem
        try:
            fd = os.open(str(QUARANTINE), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # best-effort directory fsync; the marker's own bytes are what matters
        return path
    except OSError as e:
        logger.error(
            f"could not write the in-flight restore marker for stamp "
            f"{stamp} ({type(e).__name__}: {e}) — a SIGKILL during this "
            f"restore will NOT be detectable at the next boot. Proceeding "
            f"with the restore anyway."
        )
        return None


def remove_restore_marker(stamp: str) -> None:
    """Remove the in-flight marker — call ONLY once every live move
    restore_backup attempted for this stamp is known to have landed or
    been fully rolled back. A marker left behind after a handled failure
    (one restore_backup's own except block rolled back, imperfectly or
    not) is a deliberately conservative false positive: the alternative is
    silently clearing evidence of a restore that might not, in fact, have
    finished cleanly."""
    try:
        _restore_marker_path(stamp).unlink(missing_ok=True)
    except OSError as e:
        logger.error(
            f"could not remove the in-flight restore marker for stamp "
            f"{stamp} ({type(e).__name__}: {e}) — it will keep refusing "
            f"boots until removed by hand: {_restore_marker_path(stamp)}"
        )


def find_interrupted_restore() -> dict | None:
    """Any `restore-*.inprogress` marker in QUARANTINE, or None. Returns
    the marker's own JSON content (stamp, when it was written, and the
    plan restore_backup recorded) — that content IS the recovery
    instruction. Never raises: an unreadable QUARANTINE dir reads as "none
    found" rather than blocking a boot on a question this function cannot
    answer (unlike the moves themselves, a missed marker here is a
    detection gap, not data loss in progress)."""
    try:
        if not QUARANTINE.is_dir():
            return None
        for f in sorted(QUARANTINE.glob(_RESTORE_MARKER_GLOB)):
            try:
                return json.loads(f.read_bytes().decode("utf-8")) | {"marker_path": str(f)}
            except (OSError, ValueError):
                continue
    except OSError:
        return None
    return None


def _has_rows(path: Path) -> int | None:
    """Chat count, or None if unreadable. A database that opens but reports
    nothing is not something to publish over a good snapshot.

    NONE IS AMBIGUOUS AND EVERY CALLER HAS TO SAY WHICH MEANING IT WANTS. It
    comes back for "no chat table", "corrupt header", "locked", "the volume
    stalled" - and NEVER for "zero chats", which is the integer 0. So a caller
    that tests this result for truth has collapsed None, 0 and "I could not
    look" into one answer. sync_once's regression guard did exactly that and
    stood down on the single state it exists to catch. Compare against None
    explicitly. And note this function cannot tell you whether the file is
    ABSENT - only path.exists() can, and absent and unreadable want opposite
    answers at the one call site that matters (publish vs refuse)."""
    try:
        con = sqlite3.connect(str(path), timeout=30)
        try:
            return con.execute("select count(*) from chat").fetchone()[0]
        finally:
            con.close()
    except Exception:
        return None


def _snapshot_chat_state(path: Path) -> tuple[int | None, bool]:
    """(row_count, confirmed_no_chat_table) — read TOGETHER, in ONE sqlite3
    connection, instead of two separate reads seconds apart.

    p3-b F6. sync_once used to compute `previous = _has_rows(path)` and,
    when that came back None, open a SEPARATE connection via integrity()
    a moment later to decide whether the file "opens cleanly but has no
    chat table" — and if that second, unrelated read succeeded, treated
    the snapshot as safely empty and published straight over it. A single
    TRANSIENT failure on the first read ("disk I/O error", the stall this
    module exists for) is exactly this shape: count() fails, previous is
    None, and the second connection a moment later opens fine because
    whatever glitched has already cleared. That coincidence switched off
    the shrink ratio, the per-row loss limit and the generation guard
    together and let a publish replace 400 chats with the local
    database's one row (SP\\p3-b\\wdb_guards.py).

    So: ask the schema directly, in the connection that already tried the
    count, rather than inferring "no chat table" from that attempt's
    failure. `confirmed_no_chat_table` is True ONLY when a query against
    sqlite_master itself succeeds and reports zero matching rows — any
    exception anywhere in this function means "could not confirm", which
    the caller must treat as a refusal for this cycle, not as evidence of
    emptiness.
    """
    try:
        con = sqlite3.connect(str(path), timeout=30)
    except Exception:
        return None, False
    try:
        try:
            count = con.execute("select count(*) from chat").fetchone()[0]
            return count, False
        except Exception:
            pass
        try:
            row = con.execute(
                "select count(*) from sqlite_master where type='table' "
                "and name='chat'"
            ).fetchone()
        except Exception:
            return None, False
        return (None, True) if row and row[0] == 0 else (None, False)
    finally:
        con.close()


def _content_bytes(path: Path) -> int | None:
    """Stored BYTES of every column of `chat`, or None if it cannot be read.
    A view of _scan_chat(); see there for the unit and the cost."""
    scan = _scan_chat(path)
    return None if scan is None else scan["total"]


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _scan_chat(path: Path) -> dict | None:
    """One pass over `chat`: stored bytes per conversation and in total, and
    each conversation's updated_at. None if there is no `chat` table or it
    cannot be read.

        {"total": int,                       # bytes, every column, every row
         "rows": {id: (bytes, updated_at)},  # or None - see "why_no_rows"
         "why_no_rows": str | None,
         "has_updated_at": bool,
         "newest": number | None}            # max numeric updated_at

    BYTES, NOT CHARACTERS (v3.1.9, hostile pass #2, N3). This summed
    `length("col")`, and SQLite's length() on a TEXT value counts CHARACTERS;
    only on a BLOB does it count bytes. The result was named bytes, compared
    as bytes, and printed as MB by the shrink refusal and by --status. So a
    conversation of 300,000 CJK characters (900 KB stored) replaced by 300,000
    ASCII characters (300 KB) measured a ratio of 1.0000 and published, and
    no suite could see it: every fixture padded with ASCII, where the two
    units are the same number. `cast(x as blob)` of a TEXT value is its
    encoded bytes (UTF-8 in any database OpenWebUI creates); of an INTEGER or
    REAL it is the decimal text, exactly as length() already treated them.
    How live it is, measured rather than assumed: open-webui 0.11.0 writes
    its JSON columns (`chat`, `meta`, `tasks`, `variables`) through
    SQLAlchemy's default json.dumps, which escapes every non-ASCII character,
    so those columns ARE ascii and the error reached only the TEXT columns
    (title, summary, ...). A writer with ensure_ascii=False would expose all
    of it.

    PER ROW, keyed on `chat`.`id`, because the guards that read this need to
    compare a conversation against ITSELF: the per-row loss limit (N4) and the
    generation check (N5). "rows" is None - and the reason is in
    "why_no_rows" - when there is no `id` column or an id repeats, because a
    per-row comparison keyed on a non-identity is a comparison of nothing.
    OpenWebUI declares `id` the primary key; the suites' older fixtures do not.

    THE SECOND MEASURE OF THE SHRINK GUARD, and it is deliberately not
    os.stat().st_size. A SQLite file's size is a function of its page churn as
    much as its content: sqlite3's backup API copies free-list pages, so a
    bloated database images at full size, and a VACUUM of exactly the same
    rows images at a tenth of it. Comparing file sizes therefore answers "has
    this file been repacked lately", and the question the guard needs answered
    is "is her conversation still in there". Summing the stored lengths asks
    the second one: a VACUUM rewrites the whole file without moving this sum
    by a byte, and deleting a conversation moves it immediately.

    Reads the column list rather than naming `chat`.`chat`, because the
    production schema (one JSON blob per conversation, per
    scripts/chat-metrics.py) and the suites' fixtures do not share column
    names, and a measure that silently returns 0 on a schema it does not
    recognise would disarm the guard exactly the way SHRINK_GUARD_MIN_CHATS=10
    once did. No `chat` table at all is None, not 0, for the same reason
    _has_rows() distinguishes them: "nothing there" and "I could not look"
    want opposite answers from the caller.

    COSTS A FULL TABLE SCAN, unlike _has_rows()'s count(*), which SQLite can
    serve from an index. Both files sync_once measures live on /data, so a
    publishing cycle at 41 MB reads roughly 82 MB more off the volume whose
    read reliability is this module's whole subject - call it double. It is
    not called at all on a cycle the mtime check skips, and since v3.1.9 it is
    ONE scan per file for all three content guards rather than one per guard.
    It is a real cost, accepted knowingly: the measure it replaced was free
    and wrong, and being wrong here means either refusing a documented
    recovery forever or publishing an empty database over her history.
    """
    try:
        con = sqlite3.connect(str(path), timeout=30)
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(chat)").fetchall()]
            if not cols:
                return None
            # length() is NULL for a NULL column, so without the coalesce a
            # single NULL anywhere drops that whole row from the measure -
            # which is loss-shaped, in the direction that makes the guard
            # stand down.
            expr = " + ".join(
                'coalesce(length(cast("{}" as blob)), 0)'.format(c.replace('"', '""'))
                for c in cols
            )
            has_id = "id" in cols
            has_updated_at = "updated_at" in cols
            select = ", ".join((
                '"id"' if has_id else "NULL",
                '"updated_at"' if has_updated_at else "NULL",
                expr,
            ))
            total = 0
            newest = None
            rows: dict | None = {} if has_id else None
            why = None if has_id else "the `chat` table has no id column"
            for rid, updated_at, n in con.execute(
                f"select {select} from chat"  # noqa: S608
            ):
                total += int(n)
                if _is_number(updated_at) and (newest is None or updated_at > newest):
                    newest = updated_at
                if rows is not None:
                    if rid in rows:
                        rows = None
                        why = f"the id {str(rid)[:12]!r} appears more than once"
                    else:
                        rows[rid] = (int(n), updated_at)
            return {
                "total": total,
                "rows": rows,
                "why_no_rows": why,
                "has_updated_at": has_updated_at,
                "newest": newest,
            }
        finally:
            con.close()
    except Exception:
        return None


def _row_losses(prev_rows: dict, new_rows: dict) -> list[tuple]:
    """Conversations present on BOTH sides that lost more than
    MAX_ROW_LOSS_BYTES of stored content: [(id, prev_bytes, new_bytes)],
    largest loss first. A conversation missing from `new_rows` is a deletion
    and is deliberately not here - see MAX_ROW_LOSS_BYTES."""
    out = []
    for rid, (prev_n, _) in prev_rows.items():
        if rid not in new_rows:
            continue
        new_n = new_rows[rid][0]
        if prev_n - new_n > MAX_ROW_LOSS_BYTES:
            out.append((rid, prev_n, new_n))
    return sorted(out, key=lambda t: t[2] - t[1])


def _went_backwards(prev_rows: dict, new_rows: dict) -> tuple[list[tuple], int, int]:
    """Conversations present on BOTH sides whose updated_at is EARLIER in the
    new image than in the snapshot: ([(id, prev_updated_at, new_updated_at)],
    compared, uncomparable).

    PER CONVERSATION, NOT max(updated_at) OF THE TABLE, and the difference is
    a false refusal on ordinary use. Deleting her newest conversation lowers
    the table's maximum with nothing going backwards, and a table-wide
    comparison refuses that every cycle until her next message. A
    conversation compared against ITSELF cannot be fooled by a deletion:
    open-webui only ever writes updated_at = int(time.time()) to an existing
    row (models/chats.py, 0.11.0), so the same id carrying an earlier value is
    an earlier copy of the database, or a clock that stepped backwards.
    A value that is not a number on either side is counted, not compared."""
    back, compared, uncomparable = [], 0, 0
    for rid, (_, prev_ts) in prev_rows.items():
        if rid not in new_rows:
            continue
        new_ts = new_rows[rid][1]
        if not (_is_number(prev_ts) and _is_number(new_ts)):
            uncomparable += 1
            continue
        compared += 1
        if new_ts < prev_ts:
            back.append((rid, prev_ts, new_ts))
    return sorted(back, key=lambda t: t[2] - t[1]), compared, uncomparable


def _set_aside(path: Path, why: str) -> bool:
    """Move a database (and its sidecars) out of the way, never delete.

    RETURNS FALSE IF IT COULD NOT, and the caller must then not go on to
    overwrite the file. This used to swallow the failure and return None,
    which read as "done" at both call sites in restore_on_boot: QUARANTINE is
    on /data, so a stalled or full volume - the condition this whole module
    exists for - failed the move, logged one line, and the shutil.copy2 two
    branches later landed the snapshot on top of the database we had just
    promised to preserve. That database is unreadable, not empty; a hot
    rollback journal is recoverable and may hold the only copy of everything
    written since the last sync.

    ALL OR NOTHING (v3.1.9, hostile pass #2). It moved the main file FIRST and
    stopped at the first failure, so one ENOSPC after that move left the
    database in quarantine and its -journal / -wal ORPHANED at the live path.
    The refusal that followed was correct — and its banner said "free space
    ... then boot again", and the boot after that copied the snapshot onto
    the live path, opened it, and SQLite replayed the orphan into the healthy
    copy. The module's own remediation instruction completed the corruption.

    Two changes, and neither alone is enough:

      * a failure part-way MOVES BACK what already moved, so the pair is never
        split by an exception;
      * SIDECARS GO FIRST, for the failure no rollback can handle — SIGKILL
        between two moves, which is what a RunPod redeploy is. Stranded that
        way, the main file sits at the live path without its journal. That is
        not harmless: a hot journal is how a half-applied transaction rolls
        back, and this is the database we were quarantining because it failed
        quick_check. But the next boot re-checks it, sets it aside again, and
        nothing GOOD is destroyed. The other order strands a journal beside
        the live path with no database, and the next thing written there is
        the healthy snapshot. A hostile pass proposed the reorder alone as
        harmless; it is not harmless, it is the lesser loss, and restore_on_boot
        now clears orphans before it copies so the stranded state is caught
        either way.
    """
    try:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"could not set aside {path}: {type(e).__name__}: {e}")
        return False
    stamp = _stamp()
    moved: list[tuple[Path, Path]] = []
    try:
        for suffix in SIDECARS + ("",):
            p = path.with_name(path.name + suffix)
            if p.exists():
                dest = QUARANTINE / f"{p.name}.{why}-{stamp}"
                shutil.move(str(p), str(dest))
                moved.append((p, dest))
        logger.warning(f"set aside {path} ({why}) -> {QUARANTINE}")
        return True
    except Exception as e:
        logger.error(f"could not set aside {path}: {type(e).__name__}: {e}")
        for src, dest in reversed(moved):
            try:
                shutil.move(str(dest), str(src))
            except Exception as back:
                logger.error(
                    f"AND could not move {dest} back to {src} "
                    f"({type(back).__name__}: {back}). The database and its "
                    f"sidecars are now split between {path.parent} and "
                    f"{QUARANTINE}; reunite them by hand BEFORE anything opens "
                    f"{path}, or SQLite will apply the wrong journal."
                )
        return False


def _clear_orphan_sidecars(db: Path) -> bool:
    """Set aside any -journal/-wal/-shm beside `db` when `db` itself is ABSENT.
    False if one could not be moved — the caller must then not create or copy
    anything at `db`.

    v3.1.9 (B5). A sidecar with no database belongs to no database we are
    keeping, and SQLite derives the sidecar path from the path it is GIVEN, so
    the first open of whatever lands at `db` next replays it. restore_on_boot
    copied the snapshot there and opened it one statement later, for the
    success log line. Demonstrated: 3,502,080 B / 400 chats became 933,888 B of
    "database disk image is malformed", the function returned
    restored_from_snapshot, RESTORE_EXIT_CODES mapped that to 0, and
    entrypoint.sh booted OpenWebUI onto it. The refuse-to-boot gate could not
    fire, because the restore genuinely succeeded and then its own
    VERIFICATION destroyed it.

    Three shipped routes produce the orphan: this module's own banner ("or move
    /var/lib/openwebui/webui.db by hand" leaves the sidecars behind), a
    _set_aside that stopped part-way, and a SIGKILLed writer in WAL mode. They
    are set aside, never deleted — an orphan journal may still be what someone
    needs to recover the database it came from.

    integrity()'s docstring said opening "replays/rolls back any journal,
    which on local disk always succeeds". It does always succeed. What it
    succeeds at, when the journal is not the file's own, is destroying the
    file.

    A NO-OP WHILE `db` EXISTS, and that is the half that matters most. Those
    sidecars are the database's OWN, possibly a hot journal, and moving them
    away is the loss restore_backup was just fixed for. Every caller today
    reaches this only after finding no database or setting it aside with its
    sidecars — so inside restore_on_boot the check never decides anything.
    It lives HERE, in a helper with its own test, precisely so that it is a
    tested property rather than a condition that cannot fire: an invariant
    enforced three branches away is only a comment.
    """
    if db.exists():
        return True
    for suffix in SIDECARS:
        orphan = db.with_name(db.name + suffix)
        if orphan.exists() and not _set_aside(orphan, "orphan-sidecar"):
            logger.error(
                f"{orphan} is beside {db}, which does not exist, and could not "
                f"be moved aside. NOT creating anything at {db}: SQLite would "
                f"apply it on the first open. Move it by hand, then boot again."
            )
            return False
    return True


def restore_on_boot() -> dict:
    """Make LOCAL_DB the live database before OpenWebUI starts.

    Order matters and each branch is a real case:

      1. A healthy local database wins outright. Within one container this is
         usually just a service restart, where local is newer than any
         snapshot - but NOT "by definition", which is what this used to say.
         A local file can be OLDER than the snapshot: an archive restored by
         backup.py lands here, and so does anything put back by hand or by
         scripts/switch-webui-db-to-local.py. Keeping it is still the right
         BOOT action - it is healthy, it is what the operator put there, and
         refusing a boot on that comparison would block the documented
         restore runbook. The durable copy is guarded one step later instead:
         sync_once refuses to publish a conversation older than the
         snapshot's copy of it (ALLOW_OLDER_GENERATION), or one that lost
         more than MAX_ROW_LOSS_BYTES. Not completely - once she writes to a
         conversation its updated_at is new again, so a small enough loss in
         a conversation she keeps using is invisible to both.
      2. A local database that fails quick_check is set aside (not deleted)
         and we fall through to the snapshot.
      3. The snapshot is copied down on a fresh container — the pod-recreate
         path, and the FIRST-RUN MIGRATION of the existing 41 MB database,
         which needs no special case because it is exactly this branch.
      4. Neither exists: a genuinely new deployment. OpenWebUI creates its
         own schema and the first sync publishes it.

    Every failure RETURNS rather than raising, and that is deliberate: the
    "action" string is the contract that scripts/switch-webui-db-to-local.py
    and the suites branch on. It does mean nothing here can be detected by a
    shell - so __main__ maps each action to an exit code (RESTORE_EXIT_CODES)
    and entrypoint.sh refuses the boot on a non-zero one. Any new branch
    added below needs a row in that table; it is not optional, and an unknown
    action deliberately exits non-zero.

    The one case with no good answer is 3 failing while 1 has nothing to
    serve - an empty local database beside a damaged snapshot. It returns
    "snapshot_unhealthy" and stops the boot rather than letting OpenWebUI
    build a fresh schema: the snapshot is damaged, not gone, and it stays
    recoverable exactly as long as nothing publishes over it.
    """
    result = {"action": None, "local": str(LOCAL_DB), "snapshot": str(SNAPSHOT_DB)}

    # p3-b F4/F9, checked FIRST, before any of this function's own local-
    # vs-snapshot reconciliation touches anything: a SIGKILL or an EIO
    # mid-restore_backup() can leave the live db missing, the live store
    # missing, or the two at different generations, with nothing else on
    # disk saying so. Booting normally onto that state is silent; this
    # marker is the one thing that is NOT silent about it. See
    # find_interrupted_restore()'s docstring for what remains
    # unimplemented (resuming or auto-undoing the interrupted restore).
    interrupted = find_interrupted_restore()
    if interrupted is not None:
        logger.error(
            f"REFUSING to boot: an interrupted restore_backup() run left "
            f"{interrupted.get('marker_path')} behind — the live database "
            f"and/or compactor store may be missing or at a MIXED "
            f"generation. This is NOT auto-recovered. Read the marker "
            f"(it names exactly what was planned) and put the pieces back "
            f"by hand, then remove the marker file to allow booting again: "
            f"{json.dumps(interrupted, indent=1)}"
        )
        result["action"] = "restore_interrupted"
        result["marker"] = interrupted
        return result

    try:
        LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(
            f"cannot create {LOCAL_DB.parent} ({type(e).__name__}: {e}) — "
            f"OpenWebUI will fall back to whatever DATABASE_URL points at"
        )
        result["action"] = "error"
        return result

    if LOCAL_DB.exists():
        ok, detail = integrity(LOCAL_DB)
        # An EMPTY database passes quick_check. A 0-byte file is a valid
        # SQLite database with no tables, and so is a schema OpenWebUI built
        # after a failed restore - so "opens cleanly" is not enough to let a
        # local file win over the snapshot. Found by adversarial test A9,
        # where a 0-byte local file was kept and 400 chats on the volume were
        # ignored.
        local_chats = _has_rows(LOCAL_DB) if ok else None
        # v3.1.9 round 2: SNAPSHOT_DB.exists() swallows every OSError and
        # answers False for "I could not stat it" exactly as it does for
        # "genuinely absent" (see _presence's own docstring). This branch
        # used to read that False as "no snapshot to worry about" and fall
        # through to `elif ok:` a few lines down, handing OpenWebUI an EMPTY
        # local database while a snapshot that might hold her whole history
        # sat behind an unexamined stat error. _presence() tells the two
        # apart; see the refusal below for the unstatable case.
        snap_there, snap_unstatable = _presence(SNAPSHOT_DB)
        snap_chats = _has_rows(SNAPSHOT_DB) if snap_there else None
        if ok and not local_chats and snap_unstatable:
            logger.error(
                f"local database opens cleanly but holds {local_chats!r} "
                f"chat(s), and {SNAPSHOT_DB} could not be examined (a stat "
                f"error, not 'file not found'). REFUSING to hand OpenWebUI "
                f"an empty database while the durable copy's state is "
                f"unknown: it would build a fresh schema, and if the mount "
                f"was only stalling the snapshot's history would be gone "
                f"for nothing. Retry once the volume responds; investigate "
                f"the mount if it does not."
            )
            result["action"] = "error"
            return result
        if ok and not local_chats and snap_there and snap_chats is None:
            # THE SIBLING OF THE SHRINK GUARD, and it arrives at the same
            # silently-empty boot by a different road. Local holds nothing to
            # serve, and the count we would have compared it against could not
            # be read - so `snap_chats` is falsy, the branch below does not
            # take, and the old code fell through to `elif ok:` and announced
            # "local database present and healthy (0 chats) - keeping it".
            # True, cheerful, and it never mentions that the only durable copy
            # of her history is unreadable.
            #
            # Damaged is not the same as merely holding no chat table: a
            # 0-byte or schema-only file on /data passes quick_check, is not
            # her history, and restoring it would gain nothing - so ask
            # integrity() rather than treating every None as a catastrophe,
            # or a stale empty file on the volume would block every boot.
            snap_ok, snap_detail = integrity(SNAPSHOT_DB)
            if not snap_ok:
                logger.error(
                    f"local database opens cleanly but holds {local_chats!r} "
                    f"chat(s), and the snapshot {SNAPSHOT_DB} failed "
                    f"quick_check ({snap_detail}) so it cannot be restored "
                    f"either. REFUSING to hand OpenWebUI an empty database "
                    f"while the durable copy is damaged: it would build a "
                    f"fresh schema, she would find her history gone, and the "
                    f"sync daemon would spend the rest of the boot trying to "
                    f"publish that. Recover it first: "
                    f"scripts/recover-webui-db.py, or restore from "
                    f"/data/backups."
                )
                result["action"] = "snapshot_unhealthy"
                return result
        if ok and not local_chats and snap_chats:
            logger.error(
                f"local database opens cleanly but holds {local_chats!r} "
                f"chat(s) while the snapshot holds {snap_chats}. Treating it "
                f"as an empty shell (a 0-byte file, or a schema created after "
                f"a failed restore) and restoring the snapshot instead."
            )
            if not _set_aside(LOCAL_DB, "empty-shell"):
                # See _set_aside's docstring. The copy2 below would land on a
                # file we just failed to preserve, and "empty shell" is our
                # reading of it, not a fact: a database whose `chat` table
                # cannot be read may still hold her accounts, settings and
                # uploads. Refuse rather than overwrite unpreserved state.
                logger.error(
                    f"could not move the empty local shell aside, so NOT "
                    f"restoring over it. Free space under {QUARANTINE} (or "
                    f"move {LOCAL_DB} by hand) and boot again."
                )
                result["action"] = "error"
                return result
        elif ok:
            logger.info(
                f"local database present and healthy ({LOCAL_DB}, "
                f"{LOCAL_DB.stat().st_size / 1e6:.1f} MB, "
                f"{local_chats} chats) — keeping it"
            )
            result["action"] = "kept_local"
            return result
        else:
            logger.error(
                f"local database failed quick_check ({detail}) — setting it "
                f"aside and restoring the snapshot"
            )
            if not _set_aside(LOCAL_DB, "failed-quickcheck"):
                # The SIBLING of the branch above, and the more expensive one:
                # a database that fails quick_check is the recoverable case
                # this module was written for (a hot rollback journal, not
                # corruption), and it holds everything written since the last
                # sync. Overwriting it with the snapshot after failing to
                # preserve it converts a recoverable incident into a permanent
                # loss - quietly, because the move failed with one log line.
                logger.error(
                    f"could not move the failed local database aside, so NOT "
                    f"restoring over it - it is the only copy of anything "
                    f"written since the last sync and it is recoverable. Free "
                    f"space under {QUARANTINE} (or move {LOCAL_DB} by hand), "
                    f"then boot again."
                )
                result["action"] = "error"
                return result

    # ORPHANED SIDECARS AT THE DESTINATION GO FIRST (v3.1.9, B5). Ahead of
    # BOTH branches below: the snapshot restore opens LOCAL_DB immediately,
    # and on the `fresh` branch OpenWebUI's own first open of the database
    # it creates would replay an orphan just the same. See
    # _clear_orphan_sidecars for what this cost before it existed.
    if not _clear_orphan_sidecars(LOCAL_DB):
        result["action"] = "error"
        return result

    # v3.1.9 (hostile pass #2, MEDIUM): "there is no snapshot" and "I could
    # not look at the snapshot" must not reach the same branch below. See
    # _presence — an unstatable path used to fall straight through to
    # `action="fresh"`, exit 0, an empty schema and no .empty-start marker.
    snap_there, snap_unstatable = _presence(SNAPSHOT_DB)
    if snap_unstatable:
        logger.error(
            f"cannot stat {SNAPSHOT_DB}: the mount answered an error rather "
            f"than 'file not found'. That is NOT the same as no snapshot "
            f"existing, and treating it that way is how a stalled or "
            f"erroring mount used to boot as a 'fresh' deployment - exit 0, "
            f"an empty schema, and (because the exit code was 0) "
            f".empty-start never written, so nothing but the shrink ratio "
            f"would have stood between OpenWebUI and her history on /data. "
            f"Refusing to boot rather than guess whether her history is "
            f"there. Retry once the volume responds; if it does not, "
            f"investigate the mount before anything else."
        )
        result["action"] = "error"
        return result

    if snap_there:
        # Against the RESOLVED path. os.replace does not follow a symlink at
        # its destination — it replaces the link — while the shutil.copy2 this
        # used to be followed it and wrote the target. So a staged rename onto
        # LOCAL_DB silently turned an operator's symlinked database into a
        # regular file beside nothing, and turned a DANGLING link (the fixture
        # test_webuidb_restore_guard uses to pin restore_failed -> 4) from a
        # failed restore into a "successful" one. The first version of this
        # change did exactly that, and the exit-code row caught it.
        dest = LOCAL_DB.resolve()
        staged = dest.with_name(f"{dest.name}.restoring-{_stamp()}")

        def _cleanup_staged() -> None:
            for suffix in ("",) + SIDECARS:
                try:
                    staged.with_name(staged.name + suffix).unlink(missing_ok=True)
                except Exception:
                    pass

        # D5 (findings.md, 2026-09-23 rehearsal). integrity() USED TO run
        # directly against SNAPSHOT_DB, on /data: pragma quick_check reads
        # the whole file (measured 96.5s of a 98.7s cold boot restore), and
        # — the worse half — a HOT ROLLBACK JOURNAL beside the snapshot made
        # that same open FINISH THE ROLLBACK, which is a WRITE, on the one
        # volume this whole module exists to keep off the live write path.
        #
        # Copy first, unconditionally (a plain file copy touches no sqlite
        # lock and costs only however long /data itself takes, never LONGER
        # for holding one open), THEN check and roll back the LOCAL copy,
        # where a write costs milliseconds regardless of /data's mood.
        #
        # Sidecars ARE copied here, unlike the single self-contained-
        # snapshot copy this replaces: that assumption ("sqlite3's backup
        # API produces a self-contained database") holds for a snapshot
        # sync_once itself wrote, but not for one still carrying a hot
        # journal from being written to DIRECTLY (the WEBUI_DB_LOCAL=false
        # placement, or an operator's own tool touching /data by hand).
        # Copying the pair together and rolling back the copy is exactly
        # backup.py._snapshot_via_rollback_copy's own rule for the identical
        # failure shape.
        try:
            shutil.copy2(SNAPSHOT_DB, staged)
            for suffix in SIDECARS:
                side = SNAPSHOT_DB.with_name(SNAPSHOT_DB.name + suffix)
                if side.is_file():
                    shutil.copy2(side, staged.with_name(staged.name + suffix))
        except Exception as e:
            _cleanup_staged()
            logger.error(
                f"could not copy snapshot to local staging: "
                f"{type(e).__name__}: {e}"
            )
            result["action"] = "restore_failed"
            return result

        # LOCAL, not SNAPSHOT_DB — see the comment above. Opening `staged`
        # also replays/rolls back any -journal that came with it, on local
        # disk; /data is not written to by this call.
        ok, detail = integrity(staged)
        if not ok:
            # The snapshot lives on the flaky volume, so a hot journal here
            # is the exact production failure. Do NOT copy a half-rolled-back
            # database down and call it live.
            logger.error(
                f"snapshot {SNAPSHOT_DB} failed quick_check ({detail}) once "
                f"copied to local staging. NOT restoring it; {SNAPSHOT_DB} "
                f"on /data is untouched. Recover it first: "
                f"scripts/recover-webui-db.py, or restore from /data/backups."
            )
            _cleanup_staged()
            result["action"] = "snapshot_unhealthy"
            return result

        try:
            os.replace(staged, dest)
            # A successful rollback consumes its own -journal; a -wal/-shm
            # pair under WAL mode does not vanish on a plain open. Anything
            # left over under the STAGED name must move to match `dest`'s
            # name, or the next boot's _clear_orphan_sidecars would not
            # recognise it as belonging to LOCAL_DB.
            for suffix in SIDECARS:
                leftover = staged.with_name(staged.name + suffix)
                if leftover.exists():
                    shutil.move(
                        str(leftover), str(dest.with_name(dest.name + suffix))
                    )
        except Exception as e:
            _cleanup_staged()
            logger.error(
                f"could not restore snapshot: {type(e).__name__}: {e}"
            )
            result["action"] = "restore_failed"
            return result
        # VERIFIED AFTER IT LANDS TOO. A restore that is not checked where it
        # landed is not a restore; `staged` passed quick_check under its own
        # name, and that says nothing about the file that rename/replace
        # actually put at LOCAL_DB.
        landed_ok, landed_detail = integrity(LOCAL_DB)
        if not landed_ok:
            logger.error(
                f"the snapshot passed quick_check on {SNAPSHOT_DB} but the "
                f"restored copy at {LOCAL_DB} does NOT ({landed_detail}). "
                f"Refusing to boot OpenWebUI onto it. The snapshot on /data is "
                f"untouched."
            )
            _set_aside(LOCAL_DB, "restore-unverified")
            result["action"] = "restore_failed"
            return result
        size = LOCAL_DB.stat().st_size / 1e6
        logger.info(
            f"restored snapshot -> local ({size:.1f} MB, "
            f"{_has_rows(LOCAL_DB)} chats)"
        )
        result["action"] = "restored_from_snapshot"
        # A successful restore is the one event that ENDS an empty start:
        # local now holds exactly what the snapshot holds, so publishing it
        # back can lose nothing. Clearing the marker here is what keeps the
        # empty-start refusal from becoming the permanent block it exists
        # to replace - the operator who repairs the snapshot and re-runs
        # --restore gets their sync daemon back without having to know
        # about a dotfile. Best-effort: a marker we cannot remove costs one
        # deliberate `rm`, while a restore that failed because of one would
        # cost the boot.
        try:
            EMPTY_START_MARKER.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(
                f"restored, but could not clear {EMPTY_START_MARKER}: "
                f"{type(e).__name__}: {e} — the sync daemon will keep "
                f"refusing until it is removed by hand"
            )
        return result

    logger.info(
        f"no local database and no snapshot — a new deployment; OpenWebUI "
        f"will create {LOCAL_DB} and the first sync will publish it"
    )
    result["action"] = "fresh"
    return result


# ---------------------------------------------------------------------------
# sync_once support: the mutual-exclusion lock, deterministic temp names and
# their sweep, and the hash/fsync helpers for the /data staging copy.
# v3.1.9, hostile pass (r318-b) F3/F4.
# ---------------------------------------------------------------------------

# Deterministic, NOT pid-keyed (F4). The old name
# (`{SNAPSHOT_DB.name}.sync-{os.getpid()}`) meant a SIGKILLed sync (a RunPod
# redeploy mid-sync is exactly a SIGKILL) left a ~138 MB temp file — and its
# -journal — on /data FOREVER: nothing could find it again, because the next
# process to run has a different pid, and the cleanup that existed only ran
# in an `except` block a SIGKILL never reaches. A deterministic name means
# the NEXT sync_once can find and remove its own previous leftover (see
# _sweep_stale_sync_temps) — which only stays safe with the lock below: two
# deterministic-named temps sharing one name, with no mutual exclusion,
# would let one call delete or truncate the other's in-flight file instead
# of ever only its own debris.
_SYNC_TMP_SUFFIX = ".sync-tmp"
# The legacy pid-keyed name, v3.1.7/v3.1.8 pods only: `webui.db.sync-<digits>`
# and its `-journal`. Anchored to SNAPSHOT_DB's own name and requiring the
# suffix to be ALL DIGITS, deliberately — a bare `webui.db.sync-*` glob would
# also match this module's own new deterministic `.sync-tmp` name (not a
# problem by itself) but, more importantly, is the shape of glob that has
# swept unrelated files in this codebase before (see _RESTORE_MARKER_GLOB's
# own exact-pattern discipline). This must never match SNAPSHOT_DB itself or
# anything an operator staged by hand.
_LEGACY_SYNC_TMP_RE = re.compile(
    r"^" + re.escape(SNAPSHOT_DB.name) + r"\.sync-\d+(-journal)?$"
)

# Sentinel: flock could not be used at all on this platform/mount (see
# _acquire_sync_lock), so sync_once proceeds WITHOUT mutual exclusion. Kept
# distinct from `None` (which means "another sync holds the lock right
# now" — the caller must return immediately) and from a real file handle
# (which the caller must release).
_LOCK_DEGRADED = object()


def _sync_lock_path() -> Path:
    # ON LOCAL DISK, DELIBERATELY — never /data. A lock file on the network
    # volume would itself be subject to the same stalls this whole module
    # exists to route around (a stuck flock on a stalling MooseFS mount
    # would block every future sync forever, which is a worse failure than
    # the race this lock prevents), and a lock scoped to LOCAL_DB's own
    # directory is exactly the resource sync_once needs exclusive use of —
    # the SOURCE it is about to open a SHARED lock against.
    return LOCAL_DB.with_name(LOCAL_DB.name + ".sync.lock")


def _acquire_sync_lock():
    """Take an exclusive, non-blocking flock so two sync_once calls never
    overlap (F4: a manual `webuidb.py --sync-once` can run while the
    webuidb-sync daemon's own loop is mid-cycle).

    Returns:
      * an open file object — the lock is held; call _release_sync_lock()
        with it when the sync ends.
      * `_LOCK_DEGRADED` — flock could not be used at all (see below); the
        caller proceeds WITHOUT mutual exclusion this cycle.
      * `None` — another process holds the lock right now; the caller MUST
        return immediately without sweeping or touching any temp file.

    DEGRADES SAFELY where flock is unavailable: on any platform without
    `fcntl` (Windows — a real target only for this project's own unit
    suite; production is Linux, where LOCAL_DB's filesystem — an overlay or
    a tmpfs — always implements flock) this never refuses a sync. That is
    the safe direction for THIS lock specifically: losing mutual exclusion
    here reproduces the pre-fix behaviour (pid-keyed temp names could not
    collide because no two processes ever shared one), not a new failure
    mode, and a lock that could HANG (a blocking flock on a bad mount)
    would be strictly worse than the race it prevents — which is why this
    is LOCK_EX | LOCK_NB, never a blocking flock, even on the platform
    where it is real.
    """
    try:
        import fcntl
    except ImportError:
        logger.info(
            "sync lock: fcntl is not available on this platform (expected "
            "on Windows — production is Linux); proceeding WITHOUT mutual "
            "exclusion for sync_once. See _acquire_sync_lock's docstring."
        )
        return _LOCK_DEGRADED

    path = _sync_lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+")
    except OSError as e:
        logger.warning(
            f"sync lock: could not open {path} ({type(e).__name__}: {e}) — "
            f"proceeding WITHOUT mutual exclusion this cycle. A manual "
            f"--sync-once run at the same moment as the daemon could now "
            f"race with this one."
        )
        return _LOCK_DEGRADED

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None  # another sync_once holds it right now
    return fh


def _release_sync_lock(lock) -> None:
    if lock is None or lock is _LOCK_DEGRADED:
        return
    try:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock.close()
    except Exception:
        pass


def _sweep_stale_sync_temps() -> None:
    """Remove leftover sync temp files — this module's own (both locations)
    and the legacy pid-keyed ones on /data — before starting a new sync.

    MUST be called only while holding the sync lock (see _acquire_sync_lock)
    or with mutual exclusion genuinely unavailable (_LOCK_DEGRADED): sweeping
    a name that another sync_once is actively writing to would delete or
    truncate ITS in-flight temp file, not only ever stale debris. That is
    exactly what deterministic names would otherwise risk (see
    _SYNC_TMP_SUFFIX's own comment) — the lock is what keeps this safe.

    Never raises: a sweep that cannot run is a housekeeping miss, not a
    reason to refuse the sync it is about to make room for.
    """
    for p in (
        LOCAL_DB.with_name(LOCAL_DB.name + _SYNC_TMP_SUFFIX),
        LOCAL_DB.with_name(LOCAL_DB.name + _SYNC_TMP_SUFFIX + "-journal"),
        SNAPSHOT_DB.with_name(SNAPSHOT_DB.name + _SYNC_TMP_SUFFIX),
        SNAPSHOT_DB.with_name(SNAPSHOT_DB.name + _SYNC_TMP_SUFFIX + "-journal"),
    ):
        try:
            if p.exists():
                p.unlink()
                logger.info(f"swept stale sync temp file {p}")
        except OSError as e:
            logger.warning(
                f"could not sweep stale sync temp file {p}: "
                f"{type(e).__name__}: {e}"
            )

    # Legacy pid-keyed debris from v3.1.7/v3.1.8 pods, /data only (the old
    # code never staged anything on local disk). Matched by the EXACT
    # pattern only — see _LEGACY_SYNC_TMP_RE's own comment for why this must
    # never be a bare glob.
    try:
        snap_dir = SNAPSHOT_DB.parent
        if snap_dir.is_dir():
            for entry in snap_dir.iterdir():
                if _LEGACY_SYNC_TMP_RE.match(entry.name):
                    try:
                        entry.unlink()
                        logger.info(f"swept legacy sync temp file {entry}")
                    except OSError as e:
                        logger.warning(
                            f"could not sweep legacy sync temp {entry}: "
                            f"{type(e).__name__}: {e}"
                        )
    except OSError as e:
        logger.warning(
            f"could not scan {SNAPSHOT_DB.parent} for legacy sync temps: "
            f"{type(e).__name__}: {e}"
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_file_and_dir(path: Path) -> None:
    """fsync `path` itself — REFUSING on failure — then best-effort fsync
    its parent directory.

    THE TWO DIFFER ON PURPOSE (found in review after this lane's first
    pass, before it shipped: the file fsync used to log a warning and
    carry on, same as the directory one below it). An EIO on a stalling
    MooseFS mount means the pages this function just asked to be written
    may never have reached the volume — and the read-back sync_once does
    right after this, to hash-compare `data_tmp` against `local_tmp`
    before publishing, can be served straight from the page cache. A
    cache hit hashes identically to the file the guards already approved
    REGARDLESS of what actually made it to disk, so "warn and continue"
    let a file whose durable bytes might not exist replace the last good
    snapshot, with the hash check giving false confidence that it hadn't.
    So: a failed FILE fsync RAISES, which reaches sync_once's existing
    `except Exception` — out["error"] is set, the OLD snapshot is left
    exactly where it was (this runs before `os.replace`), and both temps
    are still cleaned up in the `finally`, the same as any other publish
    refusal in that function.

    The DIRECTORY fsync stays best-effort and does NOT raise: it exists to
    persist the directory ENTRY (so a crash right after this can't leave a
    file with no name pointing at it), not to prove the FILE's bytes are
    durable — and many filesystems, FUSE mounts (this project's own /data)
    very much included, refuse or silently no-op an fsync on a directory
    descriptor. That refusal says nothing about whether `path` itself is
    safe to hash and publish, which is the question this function actually
    needs answered before sync_once compares hashes.

    Opens the file O_RDWR, not O_RDONLY: fsync on a read-only-opened fd is
    a genuine Windows CRT limitation (`[Errno 9] Bad file descriptor`,
    reproducible on a file that exists and is fully readable — verified:
    O_RDWR fsyncs correctly on the same file where O_RDONLY does not) with
    nothing to do with whether the underlying bytes are durable; O_RDWR
    works on both platforms this project's suites run on.
    """
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError as e:
        raise RuntimeError(
            f"REFUSING to publish: could not open {path} to fsync it "
            f"({type(e).__name__}: {e}) — cannot confirm the bytes just "
            f"written to it are durable. The live database is unaffected; "
            f"the previous snapshot is untouched."
        ) from e
    try:
        os.fsync(fd)
    except OSError as e:
        raise RuntimeError(
            f"REFUSING to publish: fsync of {path} failed "
            f"({type(e).__name__}: {e}) — the bytes just written to it may "
            f"not have reached the volume, and a hash comparison right "
            f"after this could still match a page-cache copy of content "
            f"that never landed on disk. The live database is unaffected; "
            f"the previous snapshot is untouched."
        ) from e
    finally:
        os.close(fd)
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass  # best-effort; matches write_restore_marker's own directory fsync


def sync_once(force: bool = False) -> dict:
    """Publish LOCAL_DB to SNAPSHOT_DB, safely, while OpenWebUI is running.

    TWO STAGES, and the SOURCE's lock never crosses into the slow one
    (v3.1.9, hostile pass r318-b F3). This used to be one `src.backup(dst)`
    straight onto /data: Python's `Connection.backup` defaults to
    `pages=-1`, one `sqlite3_backup_step(-1)` that holds a SHARED lock on
    the SOURCE (LOCAL_DB, the database OpenWebUI is actively writing) until
    the LAST destination page is written — and the destination was MooseFS.
    In rollback-journal mode a writer cannot commit while a SHARED lock is
    held elsewhere, and OpenWebUI's own busy timeout is 10s
    (DATABASE_SQLITE_PRAGMA_BUSY_TIMEOUT), so a /data write slower than
    that turned every save into "database is locked" for the whole of it —
    measured losing 2 of 3 writer commits at a 10 MB/s throttle, for the
    entire length of the sync. That is the OPPOSITE of what WEBUI_DB_LOCAL
    exists to buy: taking MooseFS OUT of the live write path, and for the
    length of every sync it was back in that path. So:

      1. `src.backup()` into a temp file ON LOCAL DISK, beside LOCAL_DB
         (`local_tmp` below) — source and destination share a filesystem,
         so this is fast regardless of /data's mood (~1s at 138 MB,
         measured, not tens of seconds), and the SHARED lock on LOCAL_DB is
         released the moment this returns. Every guard below (integrity,
         chat count, shrink ratio, per-row loss, generation) runs against
         THIS file — by the time any of them look, LOCAL_DB is already free
         and OpenWebUI can commit again no matter how this cycle ends.
      2. Only once every guard has passed: a plain byte copy of the
         VERIFIED local file to a temp path on /data (`data_tmp`) with NO
         sqlite3 connection open anywhere — a copy touches no lock at all,
         so a stalling volume here costs time, never a lock OpenWebUI is
         waiting behind. fsynced, hash-compared against the local file it
         came from, then `os.replace`d onto SNAPSHOT_DB exactly as before.

    Six content/identity guards, each earned (all run against `local_tmp`,
    stage 1's output — see above for why that is now the right file to
    validate rather than a file already sitting on /data):

      * sqlite3's backup API, not a file copy, for STAGE 1. It takes a read
        lock and produces a consistent image of a database being written
        to. A cp of a live SQLite file can capture a torn page. (STAGE 2 is
        a plain copy deliberately — see above: by then the source is a
        finished, static local file, not a live database, so there is
        nothing left to tear.)
      * The snapshot is verified BEFORE it replaces the previous one, and a
        snapshot reporting zero chats is refused. Publishing a broken image
        over a good one would turn a local problem into a durable one.
      * Written to a temporary name in the destination directory and renamed
        into place, so a failure mid-write cannot leave a partial file where
        restore_on_boot will find it.
      * The PREVIOUS snapshot is read before it is replaced - by chat count
        and by STORED CONTENT (not file size; see SHRINK_GUARD_MIN_BYTES), and
        "I could not read it" is a refusal rather than a shrug. The three
        bullets above all validate the NEW image; none of them ever looked at
        the file about to be destroyed, which is how a 1-chat database came to
        be publishable over a 2,300-chat one.
      * An empty start refuses outright while its marker exists, because a
        ratio cannot protect that state - it only delays it (see
        EMPTY_START_MARKER).
      * A conversation that exists on both sides may not lose more than
        MAX_ROW_LOSS_BYTES, and may not go BACKWARDS in updated_at (v3.1.9).
        The count and the ratio are magnitudes of the whole table; on a pod
        whose history is one row, neither can see half of that row going, and
        no magnitude can see an older copy of the same database.

    EVERY ONE OF THOSE REFUSALS HAS A WAY OUT, and that is not softness. A
    guard with no exit becomes the data-loss mode it was written against: it
    stops publishing her history to /data at all, which is the same loss
    taking longer, announced only in a log line nobody is reading.

    MUTUAL EXCLUSION (F4): the whole of both stages runs under an exclusive
    flock (see _acquire_sync_lock) so a manual `--sync-once` overlapping the
    daemon's own cycle gets a clean "another sync is in progress" skip
    instead of racing this function's now-DETERMINISTIC temp names (see
    _SYNC_TMP_SUFFIX) — a stale one from either location is swept at the
    start of every cycle, under the same lock.
    """
    out = {"synced": False, "skipped": None, "error": None, "bytes": 0}
    if not LOCAL_DB.exists():
        out["skipped"] = "no local database yet"
        return out

    lock = _acquire_sync_lock()
    if lock is None:
        # F4: the other sync_once is doing the same job right now - this is
        # not a failure, so it is a skip (out["error"] stays None), the same
        # way "unchanged since last sync" is a skip and not an error.
        out["skipped"] = "another sync is in progress"
        return out

    local_tmp = LOCAL_DB.with_name(LOCAL_DB.name + _SYNC_TMP_SUFFIX)
    data_tmp = SNAPSHOT_DB.with_name(SNAPSHOT_DB.name + _SYNC_TMP_SUFFIX)
    try:
        # p3-b F8. "Off means off" enforced HERE, not only by supervisord's
        # autostart=false line for this program. On a WEBUI_DB_LOCAL=false
        # pod, LOCAL_DB is a STALE path nobody writes to any more — OpenWebUI
        # is reading SNAPSHOT_DB directly (live_webui_db() says so) — and
        # publishing LOCAL_DB over the snapshot would overwrite the LIVE
        # database with old, disconnected content. This used to be
        # preventable only by nobody ever manually starting this program on
        # such a pod; restore_backup's own printed restart line used to name
        # it unconditionally (see that function's fix, same finding), and
        # autostart=false does not stop a manual `supervisorctl start`. A
        # refusal here means the wrong command is merely wrong, not
        # destructive.
        #
        # CHECKED BEFORE THE SWEEP BELOW, deliberately (coordinator
        # follow-up, this lane): a pod where this refusal is about to fire
        # should not be syncing AT ALL, and the sweep — while it only ever
        # touches this module's own temp names and the exact legacy
        # pid-keyed pattern, never the live snapshot — still means an
        # unlink() on /data. "Should not be running here at all" is not the
        # same claim as "safe to touch /data first", and the refusal is
        # free to check before anything is.
        if live_webui_db() != LOCAL_DB:
            raise RuntimeError(
                f"REFUSING to sync: WEBUI_DB_LOCAL says the live database "
                f"is {SNAPSHOT_DB} (the snapshot itself), not {LOCAL_DB}. "
                f"This daemon's job is to publish {LOCAL_DB} INTO the "
                f"snapshot, which would overwrite the live database with "
                f"whatever stale content is sitting at the local path on "
                f"this placement. webuidb-sync should not be running on "
                f"this pod at all — stop it: `supervisorctl stop "
                f"webuidb-sync`."
            )

        # Sweep stale temps from a previous SIGKILLed cycle (F4) - safe only
        # because we are holding the lock (or it is genuinely unavailable,
        # in which case there is nothing safer to do than proceed anyway).
        # After the refusal above, not before it - see that check's own
        # comment.
        _sweep_stale_sync_temps()

        # p3-b F7. This mtime-skip block USED TO run here, BEFORE this
        # try — `SNAPSHOT_DB.exists()` on Python 3.12.3 (every image) RAISES
        # for EIO/ETIMEDOUT/ENOTCONN/EACCES rather than returning False (it
        # only returns False for ENOENT/ENOTDIR/EBADF/ELOOP), so a stalling
        # mount escaped this function AND sync_loop entirely: the process
        # exited 1, supervisord restarted it (autorestart=true), and
        # consecutive_failures never reached 3 because it lives only in the
        # process that just died — the "publish has failed N times" alarm
        # never printed. Moved inside the try (whose `except Exception`
        # below already turns a failure into out["error"] and a warning,
        # never a crash) and reading presence through _presence() rather
        # than a bare .exists(), so an unreadable snapshot HOLDS the
        # publish (falls through to the real attempt below, which has its
        # own _presence()-based refusal a few lines down) instead of
        # killing the loop.
        mtime = LOCAL_DB.stat().st_mtime
        if not force:
            snap_there_for_skip, _ = _presence(SNAPSHOT_DB)
            if snap_there_for_skip:
                snap_mtime = SNAPSHOT_DB.stat().st_mtime
                ahead = snap_mtime - time.time()
                if ahead > SNAPSHOT_MTIME_FUTURE_TOLERANCE_S:
                    # A SNAPSHOT DATED IN THE FUTURE IS NOT EVIDENCE OF ANYTHING
                    # (v3.1.9, hostile pass #2, N2). The skip below reads "the
                    # snapshot is at least as new as local" off the mtime, so a
                    # snapshot stamped by a clock that ran fast made EVERY cycle a
                    # skip until real time caught up - no error, so no failure
                    # count, and nothing logged at any level. And the state is
                    # durable: restore_on_boot's copy2 preserves the mtime, so a
                    # poisoned snapshot re-poisons every later pod. So: not a
                    # skip. Publish, which re-stamps it with a sane time, and say
                    # why, once per cycle it happens - a silent republish would
                    # hide the wrong clock exactly as the silent skip did.
                    logger.warning(
                        f"the snapshot {SNAPSHOT_DB} is dated {ahead:.0f}s in the "
                        f"FUTURE - a clock was wrong when it was stamped. Not "
                        f"trusting it as 'unchanged since last sync'; publishing "
                        f"now, which re-stamps it."
                    )
                elif snap_mtime >= mtime:
                    # Nothing has been written since the last publish. Skipping
                    # matters: each sync writes the whole database onto the
                    # volume whose write reliability is the problem.
                    out["skipped"] = "unchanged since last sync"
                    return out

        if EMPTY_START_MARKER.exists() and SNAPSHOT_DB.exists():
            # THE RATIO CANNOT PROTECT THIS STATE. See EMPTY_START_MARKER: an
            # empty-started database is refused by the shrink guard only until
            # it grows past SHRINK_REFUSE_BELOW of the snapshot, and then it
            # publishes over the only copy of everything she has said. It was
            # demonstrated doing exactly that at 51%. A content comparison
            # narrows the window; it does not close it, because the content is
            # real - it is just NEW content standing where the old should be,
            # and no measure of size can tell those apart. The only honest
            # answer is that a human decides.
            #
            # Raised BEFORE the backup, not after: this refusal repeats every
            # SYNC_INTERVAL_S until someone acts, and the backup writes the
            # whole database onto /data each time it runs (~11.8 GB/day at
            # 41 MB). A guard whose refusal path fills the volume it guards
            # eventually defeats its own siblings.
            #
            # `and SNAPSHOT_DB.exists()` is not belt-and-braces. With no
            # snapshot there is no durable copy to protect and refusing would
            # mean never making one - the exact absent-vs-unreadable confusion
            # this function was repaired for, rebuilt one branch later.
            raise RuntimeError(
                f"REFUSING to publish: {EMPTY_START_MARKER} says this pod was "
                f"booted with WEBUI_DB_ALLOW_EMPTY_START=true after a failed "
                f"restore, so the local database is a schema OpenWebUI built "
                f"from nothing and {SNAPSHOT_DB} is still the only copy of her "
                f"history. Publishing local over it would complete the loss "
                f"the empty start only postponed. Recover the snapshot "
                f"(scripts/recover-webui-db.py, or /data/backups) and re-run "
                f"`webuidb.py --restore`, which clears this marker. To accept "
                f"starting over from nothing and let the empty database become "
                f"the durable one, delete {EMPTY_START_MARKER} - deliberately, "
                f"and knowing that the next sync overwrites the snapshot."
            )
        # STAGE 1 (F3): local disk to local disk. LOCAL_DB's SHARED lock is
        # held only for as long as THIS takes — source and destination share
        # a filesystem, so it is fast no matter how /data is behaving. See
        # this function's own docstring for the failure this replaces.
        src = sqlite3.connect(str(LOCAL_DB), timeout=60)
        try:
            dst = sqlite3.connect(str(local_tmp), timeout=60)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()

        ok, detail = integrity(local_tmp)
        chats = _has_rows(local_tmp)
        if not ok:
            raise RuntimeError(f"snapshot failed quick_check: {detail}")
        if not chats:
            raise RuntimeError(f"snapshot reports {chats!r} chats; refusing to publish")

        # REGRESSION GUARD - see SHRINK_REFUSE_BELOW. Compare the new image
        # against WHAT IS ALREADY PUBLISHED before replacing it.
        #
        # "previous" HAS THREE STATES, NOT TWO, AND THAT IS THE WHOLE GUARD.
        # This condition used to open with a bare `if previous and ...`, and
        # _has_rows() answers None both for "there is no snapshot yet" and for
        # "the snapshot is right there and I could not read it". So an
        # UNREADABLE durable copy - the precise failure this module exists for
        # - made the guard FALSY and the publish went straight over it. The
        # integrity checks above validate `local_tmp`, the new image; nothing
        # in this function was ever looking at the file about to be destroyed.
        #
        # Split apart, the two states want opposite answers: absent means
        # first publish, so go (test_webuidb_migration A8); unreadable means
        # the only durable copy is in trouble, so stop.
        #
        # EXCEPT THAT `previous` HAS FOUR STATES, NOT THREE, AND THE FIX FOR
        # THREE BROKE THE FOURTH. _has_rows() answers None for "no `chat`
        # table" exactly as it does for "corrupt / locked / stalled", and a
        # 0-BYTE FILE IS A VALID SQLITE DATABASE WITH NO TABLES: it opens
        # cleanly, it PASSES quick_check, and it has no chat count. Treating
        # every None as a catastrophe therefore refused an empty-but-healthy
        # snapshot every SYNC_INTERVAL_S, forever, and the durable copy was
        # never written at all - the whole failure this guard exists to
        # prevent, arrived at from the other direction, with every health
        # check green and one log line as the only signal.
        #
        # restore_on_boot's empty-local branch had already worked this out,
        # twelve lines away and in the same commit: "a 0-byte or schema-only
        # file on /data passes quick_check, is not her history, and restoring
        # it would gain nothing - so ask integrity() rather than treating
        # every None as a catastrophe". That is the sibling, and this is it
        # asking the same question. The four answers:
        #
        #   absent                 -> publish (A8 depends on it)
        #   readable, N rows       -> compare (below)
        #   unreadable (quick_check fails) -> refuse; it is RECOVERABLE until
        #                             something writes over it
        #   opens clean, no `chat` table -> publish; it holds no conversation
        #                             to lose, and refusing would strand the
        #                             snapshot on a file that will never
        #                             become readable on its own
        previous = None
        prev_bytes = 0
        # v3.1.9 round 2: the SAME absent-vs-unstatable collapse _presence()
        # was added for, one guard scan over. SNAPSHOT_DB.exists() answers
        # False for "the mount errored" exactly as it does for "nothing
        # published yet" -- and `previous is None` a few lines below means
        # "first publish, nothing to compare against, every guard stands
        # down" (see the block comment above this one). A TRANSIENT stat
        # failure on the flakiest volume in the system therefore used to let
        # a publish through with the shrink ratio, the per-row loss limit
        # and the generation guard ALL skipped -- worse than the
        # restore_on_boot site this same fix closes, because that one only
        # misclassifies a boot action; this one can publish completely
        # unguarded.
        snap_there, snap_unstatable = _presence(SNAPSHOT_DB)
        if snap_unstatable:
            raise RuntimeError(
                f"REFUSING to publish: cannot stat {SNAPSHOT_DB} (the mount "
                f"answered an error rather than 'file not found'). This is "
                f"NOT the same as no snapshot existing -- treating it that "
                f"way would publish with every shrink/row-loss/generation "
                f"guard skipped, since none of them run when there is "
                f"nothing to compare against. The live database is local "
                f"and unaffected. Retry once the volume responds; "
                f"investigate the mount if it does not."
            )
        if snap_there:
            prev_bytes = SNAPSHOT_DB.stat().st_size
            # p3-b F6: `previous` and "is the chat table genuinely absent"
            # are now read TOGETHER in one connection (_snapshot_chat_state)
            # rather than inferred from two separate reads seconds apart —
            # see that function's docstring for the coincidence this closes.
            previous, confirmed_no_chat_table = _snapshot_chat_state(SNAPSHOT_DB)
            if previous is None:
                if confirmed_no_chat_table:
                    # WARNING, not info: this is a normal-looking sentence for
                    # an abnormal file. It is the right answer, and it is also
                    # what a snapshot destroyed by something else looks like on
                    # the cycle before we overwrite it.
                    logger.warning(
                        f"the snapshot {SNAPSHOT_DB} "
                        f"({prev_bytes / 1e6:.1f} MB) opens cleanly and its "
                        f"`chat` table is CONFIRMED absent (checked directly "
                        f"against sqlite_master, not inferred from a failed "
                        f"count) - a 0-byte file, or a schema that never got "
                        f"one. There is no history there to lose, so this "
                        f"publish goes ahead and replaces it."
                    )
                elif not ALLOW_PUBLISH_OVER_UNREADABLE:
                    # Deliberately NO forensic copy of the local database here,
                    # unlike the shrink refusal below. The local file is not the
                    # suspect in this branch - the snapshot is - and this refusal
                    # repeats every SYNC_INTERVAL_S until a human acts, so copying
                    # the whole database onto /data each cycle would fill the
                    # volume that is already unhealthy. The local database stays
                    # where it is, live and unharmed.
                    #
                    # The flag named here is NOT WEBUI_DB_ALLOW_SHRINK. That
                    # one is printed by the shrink refusal below, where it is
                    # ordinary advice, and a single flag covering both meant
                    # the operator who took that advice silently disarmed THIS
                    # refusal too - the one the module exists for.
                    prev_ok, prev_detail = integrity(SNAPSHOT_DB)
                    if prev_ok:
                        # p3-b F6. quick_check passing does NOT mean "safe to
                        # treat as empty" - it only means the pages are
                        # well-formed. _snapshot_chat_state already tried and
                        # failed to confirm the chat table's state directly;
                        # this used to be the exact branch that published
                        # unguarded on a coincidence (a transient read failure
                        # on the FIRST connection followed by a clean SECOND
                        # one). Refuse; the finding's own rule is "any
                        # exception is a refusal for this cycle".
                        raise RuntimeError(
                            f"REFUSING to publish: {SNAPSHOT_DB} exists and "
                            f"passes quick_check, but its `chat` table's "
                            f"state could not be confirmed either way (a "
                            f"transient read failure, not a genuinely empty "
                            f"schema). Treating an unconfirmed read as "
                            f"'nothing to lose' is exactly how a publish "
                            f"used to go out with the shrink ratio, the "
                            f"per-row loss limit and the generation guard "
                            f"ALL skipped. The live database is local and "
                            f"unaffected. Retry once the volume responds; "
                            f"investigate the mount if this persists. To "
                            f"publish over it deliberately, set "
                            f"WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE=1."
                        )
                    raise RuntimeError(
                        f"REFUSING to publish: {SNAPSHOT_DB} exists and failed "
                        f"quick_check ({prev_detail}) - a hot rollback journal, "
                        f"a corrupt header, a stalled mount. That file is the "
                        f"only durable copy of her history and it is "
                        f"RECOVERABLE in every one of those cases; overwriting "
                        f"it is not. The live database is local and "
                        f"unaffected. Recover it "
                        f"(scripts/recover-webui-db.py, or /data/backups), or "
                        f"move it aside and the next sync will republish from "
                        f"local. To publish over it deliberately, set "
                        f"WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE=1."
                    )
                else:
                    # ALLOW_PUBLISH_OVER_UNREADABLE=1: the escape hatch still
                    # works for both a genuinely corrupt snapshot AND one
                    # whose chat table's state could merely not be confirmed.
                    logger.warning(
                        f"publishing over {SNAPSHOT_DB} ({prev_bytes / 1e6:.1f} "
                        f"MB) despite its `chat` table's state being "
                        f"unreadable/unconfirmed, because "
                        f"WEBUI_DB_ALLOW_PUBLISH_OVER_UNREADABLE=1 is set"
                    )

        if previous is not None:
            new_bytes = local_tmp.stat().st_size
            # ONE SCAN PER FILE FEEDS EVERY CONTENT GUARD BELOW: the ratio,
            # the per-conversation loss limit and the generation check all
            # read the same _scan_chat result. See _content_bytes for the
            # cost, and for why the unit is BYTES.
            prev_scan = _scan_chat(SNAPSHOT_DB)
            new_scan = _scan_chat(local_tmp)
            prev_content = None if prev_scan is None else prev_scan["total"]
            new_content = None if new_scan is None else new_scan["total"]
            # TWO MEASURES OF THE WHOLE TABLE, because on this pod the row
            # count alone cannot see the loss: one conversation is one row
            # (see SHRINK_GUARD_MIN_BYTES). Both are opened by ALLOW_SHRINK.
            lost_chats = (
                not ALLOW_SHRINK
                and previous >= SHRINK_GUARD_MIN_CHATS
                and chats < previous * SHRINK_REFUSE_BELOW
            )
            # THE SECOND MEASURE IS STORED CONTENT, NOT st_size, AND THE
            # DIFFERENCE IS A SHIPPED OUTAGE. A VACUUM of unchanged content
            # took a 40.07 MB image to 10.03 MB and this refused it - five
            # chats on both sides, nothing lost - while a database with 90% of
            # its rows deleted images at full size and sails through, because
            # the backup API copies free-list pages. So the file size answered
            # "has this been repacked lately" and the guard needed "is her
            # conversation still in there". _content_bytes asks the second.
            lost_content = False
            row_losses: list[tuple] = []
            went_back: list[tuple] = []
            if prev_content is None or new_content is None:
                # "I could not measure" is NOT a refusal. Turning an
                # unreadable measurement into a permanent stop is the exact
                # defect being repaired above, and it costs the durable copy
                # outright. Say so loudly and let the chat count decide - a
                # narrower guard, honestly narrower.
                logger.warning(
                    f"could not measure stored content (snapshot="
                    f"{prev_content!r}, new={new_content!r}); the shrink "
                    f"guard is running on chat count alone this cycle, "
                    f"which cannot see a one-row conversation being "
                    f"replaced by a one-row empty one, and the per-"
                    f"conversation loss and generation guards are not "
                    f"running at all."
                )
            else:
                if not ALLOW_SHRINK and prev_content >= SHRINK_GUARD_MIN_BYTES:
                    lost_content = new_content < prev_content * SHRINK_REFUSE_BELOW
                prev_rows, new_rows = prev_scan["rows"], new_scan["rows"]
                if prev_rows is None or new_rows is None:
                    # Same principle as above: cannot compare is visible and
                    # is not a refusal.
                    logger.warning(
                        f"cannot compare conversations one by one "
                        f"({prev_scan['why_no_rows'] or new_scan['why_no_rows']}); "
                        f"the per-conversation loss limit and the generation "
                        f"guard are not running this cycle."
                    )
                else:
                    # PER CONVERSATION (N4). See MAX_ROW_LOSS_BYTES.
                    row_losses = _row_losses(prev_rows, new_rows)
                    # GENERATION (N5): the fifth state of `previous`. The
                    # four above are absent / readable / unreadable /
                    # schema-only, and every guard so far measures a
                    # MAGNITUDE. A healthy, right-sized, week-old copy of the
                    # same database is none of those - it was demonstrated
                    # publishing 380 chats over 400. Going backwards in time
                    # is not a change in size.
                    if not (prev_scan["has_updated_at"] and new_scan["has_updated_at"]):
                        # NOT a refusal. A guard that refuses every publish
                        # because a column is missing is an outage built out
                        # of a guard; the suites' own older fixtures have no
                        # updated_at, and neither might a future schema.
                        # --status reports the same fact.
                        logger.warning(
                            f"no updated_at column in "
                            f"{'the snapshot' if not prev_scan['has_updated_at'] else 'the new image'}"
                            f"; the generation guard cannot tell an older copy "
                            f"of this database from the current one and is not "
                            f"running this cycle."
                        )
                    else:
                        went_back, compared, uncomparable = _went_backwards(
                            prev_rows, new_rows
                        )
                        if uncomparable:
                            logger.warning(
                                f"{uncomparable} conversation(s) have no numeric "
                                f"updated_at on one side and were not compared "
                                f"by the generation guard ({compared} were)."
                            )
            refuse_row_loss = bool(row_losses) and not ALLOW_ROW_LOSS
            refuse_generation = bool(went_back) and not ALLOW_OLDER_GENERATION
            # An override that OPENS a tripped refusal says so. These two are
            # meant to be set for one invocation; if one has been left on a
            # template, this line is how anyone finds out.
            if row_losses and ALLOW_ROW_LOSS:
                logger.warning(
                    f"WEBUI_DB_ALLOW_ROW_LOSS is set: publishing although "
                    f"{len(row_losses)} conversation(s) lost more than "
                    f"{MAX_ROW_LOSS_BYTES / 1e6:.1f} MB each. Unset it once "
                    f"this publish is done."
                )
            if went_back and ALLOW_OLDER_GENERATION:
                logger.warning(
                    f"WEBUI_DB_ALLOW_OLDER_GENERATION is set: publishing although "
                    f"{len(went_back)} conversation(s) are older than the "
                    f"snapshot's copy. Unset it once this publish is done."
                )
            reasons: list[str] = []
            if lost_chats or lost_content:
                # Keep the refused database. It may hold the only copy of
                # anything written since the last good sync, and this path
                # fires precisely when something has already gone wrong.
                #
                # RATE-LIMITED (see FORENSIC_COPY_MIN_INTERVAL_S). This
                # refusal repeats every SYNC_INTERVAL_S until an operator
                # acts, and this used to copy the WHOLE local database here
                # on every one of those cycles - ~9.5 GB/day at the live
                # size, onto the volume this module exists to protect. The
                # FIRST copy of a new refusal (last_monotonic is None, or the
                # interval has elapsed since the last one) is unconditional -
                # it is the earliest evidence of whatever went wrong, and is
                # never skipped. Every copy after that, for the SAME ongoing
                # refusal, is throttled.
                global _forensic_copy_last_monotonic
                _forensic_now = time.monotonic()
                if (
                    _forensic_copy_last_monotonic is None
                    or _forensic_now - _forensic_copy_last_monotonic
                    >= FORENSIC_COPY_MIN_INTERVAL_S
                ):
                    try:
                        QUARANTINE.mkdir(parents=True, exist_ok=True)
                        # SIDECARS TOO (hostile pass #2 LOW, "what I checked
                        # and found sound" §sidecar audit): this copied only
                        # the main file, never `-wal`/`-shm`/`-journal`. In
                        # WAL mode recently-committed rows can live in the
                        # `-wal` file until the next checkpoint, so a plain
                        # copy of the main file can be MISSING data that is
                        # not "lost" at all - it just has not been
                        # checkpointed yet - which is exactly the wrong thing
                        # for evidence collected because something looked
                        # like loss. One stamp for the whole set, so the main
                        # file and its sidecars are named as one snapshot.
                        _forensic_stamp = _stamp()
                        shutil.copy2(
                            LOCAL_DB,
                            QUARANTINE / f"{LOCAL_DB.name}.refused-{_forensic_stamp}",
                        )
                        for _sidecar_suffix in SIDECARS:
                            _sidecar = LOCAL_DB.with_name(
                                LOCAL_DB.name + _sidecar_suffix
                            )
                            if _sidecar.exists():
                                shutil.copy2(
                                    _sidecar,
                                    QUARANTINE
                                    / f"{_sidecar.name}.refused-{_forensic_stamp}",
                                )
                        _forensic_copy_last_monotonic = _forensic_now
                    except Exception:
                        pass
                _prev_desc = (
                    f"{previous} chat(s) holding {prev_content / 1e6:.1f} MB "
                    f"of content in a {prev_bytes / 1e6:.1f} MB file"
                    if prev_content is not None
                    else f"{previous} chat(s) in a {prev_bytes / 1e6:.1f} MB file"
                )
                _new_desc = (
                    f"{chats} chat(s) holding {new_content / 1e6:.1f} MB of "
                    f"content in a {new_bytes / 1e6:.1f} MB file"
                    if new_content is not None
                    else f"{chats} chat(s) in a {new_bytes / 1e6:.1f} MB file"
                )
                reasons.append(
                    f"the local database has {_new_desc} "
                    f"but the snapshot has {_prev_desc} "
                    f"({'chat count' if lost_chats else 'stored content'} "
                    f"tripped it). That is not ordinary use - it is what a "
                    f"failed migration, a half-copied file or a freshly "
                    f"created empty schema looks like, and publishing it would "
                    f"overwrite the only durable copy of her history. A VACUUM "
                    f"cannot cause this: what is compared is the stored length "
                    f"of the `chat` rows, which VACUUM rewrites without "
                    f"changing, so the recovery in OPERATIONS.md is safe to "
                    f"run and something is genuinely gone here. The local "
                    f"database has been copied to {QUARANTINE}. If this shrink "
                    f"is real and intended, set WEBUI_DB_ALLOW_SHRINK=1 (which "
                    f"opens THIS refusal only - an unreadable snapshot still "
                    f"stops the publish)."
                )
            if refuse_row_loss:
                # NO forensic copy, unlike the shrink refusal: this one is
                # reachable by ordinary use (a deleted branch), it repeats
                # every SYNC_INTERVAL_S until a human acts, and the file that
                # holds the content in question - the snapshot - is exactly
                # the one being left untouched. A 33 MB copy onto /data every
                # five minutes would fill the volume to protect nothing.
                rid, prev_n, new_n = row_losses[0]
                reasons.append(
                    f"{len(row_losses)} conversation(s) that still exist lost "
                    f"more than {MAX_ROW_LOSS_BYTES / 1e6:.1f} MB of stored "
                    f"content each - the largest, {str(rid)[:12]!r}, went from "
                    f"{prev_n / 1e6:.1f} MB to {new_n / 1e6:.1f} MB. OpenWebUI "
                    f"keeps a whole conversation in one row, so no chat count "
                    f"can see this, and a loss under half the table does not "
                    f"move the content ratio. It is what a truncated or "
                    f"half-written conversation looks like; it is ALSO what "
                    f"deleting a long branch of one conversation looks like, "
                    f"and only a person can tell those apart. If she really "
                    f"did delete it, publish it once, deliberately: "
                    f"`WEBUI_DB_ALLOW_ROW_LOSS=1 /opt/compactor-venv/bin/python "
                    f"/opt/compactor/webuidb.py --sync-once` - set in that "
                    f"shell, NOT on the RunPod template, where it would stay "
                    f"set and wave through the next truncation too. It opens "
                    f"this refusal only."
                )
            if refuse_generation:
                rid, prev_ts, new_ts = went_back[0]
                reasons.append(
                    f"{len(went_back)} conversation(s) in the local database "
                    f"are OLDER than the same conversation in the snapshot - "
                    f"{str(rid)[:12]!r} has updated_at {prev_ts} in the "
                    f"snapshot and {new_ts} locally. OpenWebUI only ever moves "
                    f"updated_at forward, so this is an EARLIER COPY of the "
                    f"database standing where the current one should be: a "
                    f"file put back by hand, an archive from /data/backups, a "
                    f"recovery script's output. Publishing it removes "
                    f"everything since that copy from the only durable one. "
                    f"(A system clock that stepped backwards does this too, "
                    f"for as long as the step.) If the older database is "
                    f"deliberate - you restored an archive on purpose - "
                    f"publish it once: `WEBUI_DB_ALLOW_OLDER_GENERATION=1 "
                    f"/opt/compactor-venv/bin/python /opt/compactor/webuidb.py "
                    f"--sync-once`, in that shell and not on the template "
                    f"(OPERATIONS.md, 'Restore from a backup'). It opens this "
                    f"refusal only."
                )
            if reasons:
                # Every ground at once, each with its own way out. Naming only
                # the first would send the operator round the loop once per
                # guard, and an older copy of the database typically trips
                # both of the last two.
                raise RuntimeError(
                    "REFUSING to publish: " + " || AND, SEPARATELY: ".join(reasons)
                )

        # STAGE 2 (F3): local_tmp is now VERIFIED (every guard above passed).
        # Copy it to /data with NO sqlite3 connection open anywhere - LOCAL_DB
        # was only ever touched in stage 1, above, and is not reopened here.
        # A stalling /data now costs time, never a lock OpenWebUI is waiting
        # behind.
        SNAPSHOT_DB.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_tmp, data_tmp)
        _fsync_file_and_dir(data_tmp)
        # Verify the copy landed intact: a hash compare of the two (already
        # local, already-read-once-for-guards) files is cheap - SNAPSHOT_DB
        # is not yet touched, and every guard above already read local_tmp in
        # full at least once, so the hash's own read is the only new full
        # pass this costs. Cheaper and more exact than re-running
        # quick_check, which only proves data_tmp is SOME well-formed SQLite
        # file - not that it is byte-identical to what the guards approved.
        local_hash = _sha256_file(local_tmp)
        data_hash = _sha256_file(data_tmp)
        if local_hash != data_hash:
            raise RuntimeError(
                f"REFUSING to publish: the copy to {data_tmp} does not match "
                f"the verified local file (sha256 {local_hash[:12]} != "
                f"{data_hash[:12]}) - /data may be corrupting writes. Nothing "
                f"has been published; the live database is unaffected."
            )

        os.replace(data_tmp, SNAPSHOT_DB)
        # Stamp the snapshot with the LOCAL mtime this image was taken from,
        # so "has anything changed since the last publish?" is a meaningful
        # question next cycle. Without this the snapshot carries the temp
        # file's own mtime, which can predate the local database's last write
        # (writes continue during the backup), and the skip never fires - so
        # every cycle rewrites the whole database onto the volume whose write
        # reliability is the entire problem.
        #
        # NEVER LATER THAN NOW (v3.1.9, N2). A local mtime from a clock that
        # ran fast used to be copied onto the durable file, where it froze
        # every later cycle as a skip and survived redeploys through copy2.
        # Clamped, a future local mtime costs a republish per cycle until the
        # clock passes it - a cost, never a freeze.
        stamp = min(mtime, time.time())
        try:
            os.utime(SNAPSHOT_DB, (stamp, stamp))
        except Exception:
            pass  # a filesystem that refuses utime costs an extra sync, no more
        out["synced"] = True
        out["bytes"] = SNAPSHOT_DB.stat().st_size
        logger.info(
            f"published local -> snapshot ({out['bytes'] / 1e6:.1f} MB, "
            f"{chats} chats)"
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        # Expected whenever the volume is stalling — which is the condition
        # this module exists for. THE LIVE DATABASE IS LOCAL AND UNAFFECTED
        # is now actually true whatever fails here (v3.1.9, F3): stage 1
        # already released LOCAL_DB's lock before this point, so nothing
        # past it - including everything a slow /data can do - reaches back
        # to the file OpenWebUI has open. Only the durability window widens.
        # Loud, but never fatal.
        logger.warning(
            f"snapshot publish failed ({out['error']}). The live database is "
            f"local and unaffected; retrying in {SYNC_INTERVAL_S:.0f}s. Chat "
            f"history is exposed to pod loss until this succeeds."
        )
    finally:
        # Always clean up BOTH temp files on the way out, success or failure
        # (F4): leaving one behind for "the next exception handler" is
        # exactly the old pid-keyed bug's shape, just moved - a SIGKILL here
        # skips this too, which is why _sweep_stale_sync_temps() at the top
        # of the NEXT cycle is the real fix; this is the fast path that
        # usually means there is nothing for that sweep to find.
        for _p in (local_tmp, data_tmp):
            try:
                if _p.exists():
                    _p.unlink()
            except Exception:
                pass
        _release_sync_lock(lock)
    return out


class _ShutdownRequested(BaseException):
    """Raised from _sigterm_handler to unwind sync_loop's time.sleep (or an
    in-flight sync_once) so the final publish (D1, findings.md) runs
    immediately instead of waiting out the rest of the current interval.

    DELIBERATELY A BaseException, not an Exception — the same reason
    KeyboardInterrupt and SystemExit are not Exception subclasses either.
    sync_once() has its own broad `except Exception` (see its docstring:
    "not raising is the actual contract now") so it can turn a genuine
    internal failure into out["error"] instead of killing the loop: if this
    were an Exception, a SIGTERM landing WHILE sync_once() is running would
    be swallowed by that same handler and reported as an ordinary result,
    silently discarding the signal instead of triggering the final sync.

    time.sleep() is documented (PEP 475) to sleep the FULL requested time
    even across a caught signal UNLESS the handler raises — which is
    exactly why this raises rather than only setting a flag: a flag would
    still wait out up to SYNC_INTERVAL_S (300s default) before the process
    noticed, almost certainly past supervisord's stopwaitsecs.
    """


def _sigterm_handler(signum, frame) -> None:
    raise _ShutdownRequested()


def _final_sync_on_shutdown() -> None:
    """D1 (findings.md). Runs the same publish the DB-MOVE-RUNBOOK's manual
    "final sync" step performs by hand (`sync_once(force=True)`), so a
    `supervisorctl stop` — which sends SIGTERM and then waits `stopwaitsecs`
    (120s, supervisord.conf, comfortably longer than a warm publish's
    13-23s) before SIGKILLing — captures her last write automatically.

    force=True deliberately: an ordinary sync_once() would very likely hit
    the ordinary "unchanged since last sync" skip (nothing is more likely
    than there being no NEW write in the seconds since the last periodic
    publish), and this is the one caller that must not accept that skip
    quietly — if there IS something unpublished this is the last chance,
    and if there is not, force=True republishing identical content is a
    no-op cost, not a correctness problem.

    Never raises: the process is exiting either way, and an unhandled
    exception here would trade the one line an operator needs to see
    (published / skipped / error) for a stack trace in the shutdown log.
    A SECOND SIGTERM arriving while this runs is not caught here — supervisord
    does not send one before stopwaitsecs elapses, and a caller who really
    wants a hard stop should get one.
    """
    try:
        r = sync_once(force=True)
    except Exception as e:
        logger.error(f"final sync on SIGTERM raised: {type(e).__name__}: {e}")
        return
    if r["error"]:
        logger.error(f"final sync on SIGTERM did not publish: {r}")
    else:
        logger.info(f"final sync on SIGTERM: {r}")


def sync_loop() -> None:
    """Daemon entry point (supervisord program `webuidb-sync`)."""
    signal.signal(signal.SIGTERM, _sigterm_handler)
    logger.info(
        f"webui.db sync: {LOCAL_DB} -> {SNAPSHOT_DB} every "
        f"{SYNC_INTERVAL_S:.0f}s"
    )
    consecutive_failures = 0
    # v3.1.9 (hostile pass #2, MEDIUM): "sync_loop never logs or counts a
    # skip, so a permanently idle daemon is invisible." sync_once has TWO
    # skip reasons and they are not alike. "unchanged since last sync" is
    # what her being asleep looks like, correctly, every quiet night this
    # pod runs — counting or alarming on it would train an operator to
    # ignore this log within a week. "no local database yet" is not: it
    # means LOCAL_DB has never once existed this run, which is what a wrong
    # WEBUI_LOCAL_DB (or WEBUI_DB_LOCAL=false pointing the sync daemon at
    # nothing) looks like — supervisord shows RUNNING, sync_once returns
    # cleanly every cycle (error=None), and the one startup line above is
    # the only evidence for the life of the pod otherwise. Only THAT reason
    # is counted, on the same shout-on-first-then-hourly-forever shape as
    # the failure counter below.
    consecutive_no_local = 0

    def _account(r: dict) -> None:
        nonlocal consecutive_failures, consecutive_no_local
        if r["error"]:
            consecutive_failures += 1
            consecutive_no_local = 0
            # `in (3, 12, 48)` MEANT THE LOG WENT QUIET AFTER FOUR HOURS, on
            # the one condition where /data holds the only copy that survives
            # a pod recreate. A membership test says it three times and then
            # never again: at failure 49 the log is silent, /health/full reads
            # "ok", chat works perfectly, and the durability gap is unbounded.
            #
            # This is the shape bgwork.BackgroundPool.submit already uses for
            # the same problem — shout on the first, then at a fixed interval
            # forever — so the two now agree.
            if consecutive_failures == 3 or (
                consecutive_failures > 3 and consecutive_failures % 12 == 0
            ):
                logger.error(
                    f"snapshot publish has failed {consecutive_failures} times "
                    f"in a row ({consecutive_failures * SYNC_INTERVAL_S / 60:.0f} "
                    f"minutes without a durable copy of chat history). The "
                    f"volume is probably degraded; the live database is fine."
                )
        elif r["skipped"] == "no local database yet":
            consecutive_failures = 0
            consecutive_no_local += 1
            if consecutive_no_local == 3 or (
                consecutive_no_local > 3 and consecutive_no_local % 12 == 0
            ):
                logger.error(
                    f"no local database at {LOCAL_DB} for "
                    f"{consecutive_no_local} cycles in a row "
                    f"({consecutive_no_local * SYNC_INTERVAL_S / 60:.0f} "
                    f"minutes) — nothing has been published this run. This is "
                    f"what a wrong WEBUI_LOCAL_DB (or WEBUI_DB_LOCAL) looks "
                    f"like from in here: no error, supervisord shows RUNNING, "
                    f"and otherwise nothing says so. Check the path exists "
                    f"and OpenWebUI is actually writing to it."
                )
        else:
            # Either published, or the ordinary "unchanged since last sync"
            # skip — both mean nothing is currently wrong.
            consecutive_failures = 0
            consecutive_no_local = 0

    # D11/D14 (findings.md). Publish once IMMEDIATELY, rather than waiting a
    # full SYNC_INTERVAL_S before the first cycle. Before this fix,
    # /health/full's snapshot check read "stale: true" (with a huge
    # local_lag_s) for up to SYNC_INTERVAL_S after every boot — a false
    # alarm: restore_on_boot's copy2 leaves LOCAL_DB's mtime matching the
    # snapshot it was restored from, so there is nothing actually stale,
    # only nothing published YET. Safe to run this early: sync_once()
    # already treats "LOCAL_DB does not exist yet" as an ordinary skip, not
    # an error, which is what covers a boot ordering where this program's
    # priority (supervisord.conf) puts it ahead of OpenWebUI itself.
    try:
        _account(sync_once())
    except _ShutdownRequested:
        _final_sync_on_shutdown()
        return

    while True:
        # p3-b F7. sync_once's own try/except already covers everything
        # from staging onward, and moving the mtime-skip block inside it
        # (the actual F7 fix) closed the one gap that used to let an
        # OSError escape sync_once entirely and kill this loop. That
        # `except Exception` boundary is why _ShutdownRequested above is a
        # BaseException, not an Exception — see its own docstring.
        #
        # A belt-and-braces try/except Exception HERE too was tried and
        # reverted: it silently swallowed test_webuidb_publish_guards.py's
        # own `_drive_sync_loop` sentinel (`_StopLoop`, an Exception
        # subclass used to end this otherwise-infinite loop in a test with
        # `time.sleep` patched to a no-op) — turning "stop the loop" into
        # "log a failure and spin at full CPU forever", a real hang this
        # was caught doing on a real run (the whole-unit-suite pass this
        # lane finished with). A bare `except Exception` at a `while True:`
        # boundary cannot tell a genuine escaped error from a caller's own
        # control-flow exception, and the loop already has no story for how
        # a *test* is supposed to stop it short of that. sync_once() not
        # raising (Exception) is the actual contract now; if a future
        # change breaks that again, the fix belongs inside sync_once's own
        # try, the same place this one did.
        try:
            time.sleep(SYNC_INTERVAL_S)
            r = sync_once()
        except _ShutdownRequested:
            _final_sync_on_shutdown()
            return
        _account(r)


def _build_arg_parser() -> argparse.ArgumentParser:
    """D6 (findings.md). This used to be a hand-rolled `"--x" in sys.argv`
    chain with no real parser behind it: an UNRECOGNISED flag — including
    `--help` — matched none of the `elif` arms and fell straight through to
    the bare `else: sync_loop()`, so `webuidb.py --help` silently started
    the daemon loop instead of printing usage (the DB-MOVE-RUNBOOK's own
    "There is NO --sync-now. `--help`, or any unknown flag, starts a
    second daemon loop" is this exact defect, in the operator's own
    words). argparse rejects an unrecognised argument with exit 2 and
    handles `-h`/`--help` itself (prints usage, exit 0) — the fix is
    switching parsers, not hand-writing either of those two behaviours.
    """
    p = argparse.ArgumentParser(
        prog="webuidb.py",
        description=(
            "Sync OpenWebUI's live database (local disk) to the /data "
            "snapshot, or run one of the boot/CLI actions supervisord and "
            "entrypoint.sh use. With no mode flag: run the sync daemon "
            "loop (supervisord program webuidb-sync)."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-restore-marker", action="store_true",
        help="report whether an interrupted restore_backup() marker exists, then exit",
    )
    mode.add_argument(
        "--restore", action="store_true",
        help="run restore_on_boot() once, then exit (entrypoint.sh's boot step)",
    )
    mode.add_argument(
        "--sync-once", action="store_true",
        help="publish the local database to the /data snapshot once, then exit",
    )
    mode.add_argument(
        "--status", action="store_true",
        help="print the health of both the local and snapshot copies, then exit",
    )
    p.add_argument(
        "--force", action="store_true",
        help="with --sync-once, publish even if unchanged since the last sync",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    _parser = _build_arg_parser()
    _args = _parser.parse_args()
    if _args.force and not _args.sync_once:
        # Not folded into the mutually exclusive group above: --force is not
        # a MODE, it modifies --sync-once, so it needs its own check rather
        # than being forbidden from combining with every other mode too.
        _parser.error("--force is only meaningful together with --sync-once")

    if _args.check_restore_marker:
        # p4-b G3. PLACEMENT-INDEPENDENT: entrypoint.sh runs this BEFORE the
        # `if [ "${WEBUI_DB_LOCAL}" = "true" ]` split, so it also covers
        # WEBUI_DB_LOCAL=false (production, the shipped default) — the
        # placement whose boot never calls --restore / restore_on_boot at
        # all, so restore_on_boot's OWN marker check (a few lines below)
        # never runs there either. Before this, a kill mid-restore_backup()
        # on that placement booted silently onto whatever half-finished
        # state it left (G1's db-replaced-by-empty-schema shape is one
        # consequence; see that finding).
        #
        # DELIBERATELY DOES NOT OPEN webui.db, and never will:
        # find_interrupted_restore() only globs QUARANTINE (a small
        # forensics directory) and reads one small JSON file — no
        # PRAGMA quick_check, no row scan, nothing that competes for the
        # SAME file lock a hot rollback journal replay needs. This runs on
        # EVERY boot, including the overwhelming majority with no marker at
        # all, on the same MooseFS volume that took several minutes to roll
        # back a hot journal on a 138 MB webui.db twice in one afternoon
        # (2026-09-13 production incident, WEBUI_DB_LOCAL=false) — a
        # second opener here would not speed that rollback up, and could
        # only add contention to it, on the one boot where every second
        # already counts. Whether a marker that IS present can be cleared
        # automatically needs exactly that kind of open (a quick_check
        # against the target database) to answer safely — deliberately not
        # attempted here; see the banner below and restore_backup's own
        # self-clearing (this finding's other half, at the marker's write
        # site) for what closes the common case before boot ever sees it:
        # a FULLY rolled-back or fully re-verified restore removes its own
        # marker, so by the time this runs the only markers left describe a
        # restore this process cannot itself prove finished.
        interrupted = find_interrupted_restore()
        if interrupted is None:
            print("no in-flight restore marker")
            sys.exit(0)
        print(json.dumps(interrupted, indent=1))
        sys.exit(1)
    elif _args.restore:
        r = restore_on_boot()
        print(r)
        # THE EXIT CODE IS THE ONLY THING THE BOOT SCRIPT CAN SEE, and this
        # was `print(restore_on_boot())` with no sys.exit at all - so every
        # failure exited 0. restore_on_boot() does not raise, deliberately
        # (scripts/switch-webui-db-to-local.py branches on r["action"] and the
        # suites assert on it, so the dict is the contract), which meant
        # entrypoint.sh's `|| { WARNING }` branch could not fire for three
        # separate reasons at once. A restore that failed outright announced
        # success: OpenWebUI started onto a database that did not exist,
        # alembic built a fresh empty schema, and one message later the sync
        # daemon tried to publish that over her only durable copy.
        #
        # An action MISSING from the table exits non-zero, never zero. This
        # project's most expensive recurring defect is a rule applied at one
        # site and missed at its sibling, and a .get(..., 0) here is that
        # defect pre-installed: a future branch returning a new action string
        # would get a silent pass on the one path that loses everything.
        sys.exit(RESTORE_EXIT_CODES.get(r.get("action"), 6))
    elif _args.sync_once:
        # D4 (findings.md): ALWAYS force, regardless of whether the caller
        # also passed --force. This is the one-shot CLI action an operator
        # or a script runs BY HAND expecting it to actually publish - the
        # DB-MOVE-RUNBOOK's own "There is NO --sync-now" complaint is
        # exactly this: `--restore` (a backup.py::restore_backup undo, or
        # any restore that preserves the old file's mtime via copy2 or
        # tar) can leave LOCAL_DB's mtime at or before the snapshot's, so a
        # plain `--sync-once` silently hit sync_once's own "unchanged since
        # last sync" mtime-skip and printed success (exit 0) without
        # publishing anything. The DAEMON's periodic cycle (sync_loop, via
        # a bare `sync_once()`) still uses the mtime skip deliberately -
        # each cycle would otherwise rewrite the whole database onto /data
        # even when nothing changed - this is only about the explicit,
        # one-shot CLI action, which has no "next cycle" to catch up on.
        r = sync_once(force=True)
        print(r)
        # The sibling, fixed at the same time and for the same reason. A
        # refused publish is the guard working, and it still must not report
        # success to a shell: anything that wraps this in `&&` - a hot-patch
        # script, a cron, an operator checking before a redeploy - would
        # otherwise read "refused to publish, her history is exposed" as
        # "done". A skip (unchanged, or no local database yet) IS success:
        # nothing needed doing.
        sys.exit(1 if r["error"] else 0)
    elif _args.status:
        # --status stays exit 0 even when it reports an unhealthy database,
        # deliberately: it is a report, not a gate, and the WARNING text in
        # entrypoint.sh tells an operator to run it while investigating. A
        # report that exits non-zero when it successfully reported is its own
        # trap. The gates are --restore and --sync-once above.
        for label, p in (("local", LOCAL_DB), ("snapshot", SNAPSHOT_DB)):
            ok, detail = integrity(p)
            size = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "-"
            # Stored content beside the file size, because they are not the
            # same number and the difference between them is what the shrink
            # guard now compares: a VACUUMed 4 MB file and a bloated 41 MB one
            # can hold identical history, and this is where an operator sees
            # that before believing a size.
            # BYTES since v3.1.9 (see _scan_chat) - this printed a character
            # count as MB.
            scan = _scan_chat(p) if p.exists() else None
            content_s = "-" if scan is None else f"{scan['total'] / 1e6:.1f} MB"
            # The generation guard's input, and - more to the point - whether
            # it CAN run. "Cannot compare" is logged by sync_once and is not a
            # refusal; this is where it is visible without reading a log.
            if scan is None:
                newest_s = "-"
            elif not scan["has_updated_at"]:
                newest_s = "- (no updated_at column: the generation guard cannot run)"
            elif scan["newest"] is None:
                newest_s = "- (no numeric updated_at: the generation guard cannot run)"
            else:
                newest_s = str(scan["newest"])
            print(
                f"{label:9} {str(p):34} {size:>10}  quick_check={detail}  "
                f"chats={_has_rows(p)}  content={content_s}  "
                f"newest_update={newest_s}"
            )
        # The refusal an operator is most likely to be staring at when they
        # run this, and the only one whose cause is a file rather than a
        # measurement. Printed unconditionally, including when it is absent,
        # so "there is no marker" is a fact reported rather than an absence
        # inferred from silence.
        print(
            f"empty-start marker {str(EMPTY_START_MARKER):34} "
            f"{'PRESENT - sync_once is refusing; see the banner in entrypoint.sh' if EMPTY_START_MARKER.exists() else 'absent'}"
        )
    else:
        sync_loop()
