"""Shared helper: know where OpenWebUI's LIVE `webui.db` actually is, for the
operator scripts that write to it directly.

THE DEFECT THIS CLOSES (D3, `dbmove/findings.md` section 4 in the D3-move
rehearsal owned by the architect; evidence `40-tools-paths.txt`). When the
pod runs with `WEBUI_DB_LOCAL=true`, OpenWebUI's live database moves to
`/var/lib/openwebui/webui.db` (local disk); `/data/openwebui/webui.db` is
only the periodically published SNAPSHOT (`compactor/webuidb.py`,
`sync_once()`). Every writing operator script in this directory used to
accept `/data/openwebui/webui.db` unconditionally and silently edit the
SNAPSHOT. The next sync cycle then published local over it, and the repair
was gone with no warning — measured directly in the rehearsal's P3: "the
repair succeeded on the snapshot; the next cycle was a silent skip; her
next message published local over it; the repair was gone. No guard
fired." The scripts' own "database is open by another process" refusal
cannot catch this, because nothing holds the snapshot file open while
local mode is active — OpenWebUI never touches it directly.

THE FOLD. `WEBUI_DB_LOCAL` is folded exactly as `compactor/webuidb.py`'s
own `live_webui_db()` folds it (trimmed, case-insensitive; empty/unset,
"true", "1", "yes", "on" -> local; "false", "0", "no", "off" -> snapshot).
`fold_webui_db_local()` below is that same table, duplicated rather than
imported — these operator scripts run standalone on a pod with no
`compactor` package guaranteed importable (see the other scripts' own
PACKAGE RESOLUTION sections), and `git show v3.1.9:compactor/webuidb.py`
is what this was checked against.

WHY MORE THAN THE ENV VAR. A fresh Web Terminal shell does not inherit
`WEBUI_DB_LOCAL` — it is set only in supervisord's own child environment
(`entrypoint.sh`'s NORMALIZATION block writes it there), so an operator
running one of these scripts by hand, from a plain shell, sees an EMPTY
value regardless of which mode the pod is actually in. Folding empty as
"true" (matching webuidb.py) is only correct if the pod really is in
local mode; on a pod deliberately rolled back to `WEBUI_DB_LOCAL=false`,
that same empty shell value would make this module wrongly assume local
mode and refuse a perfectly correct write to `/data`. So `detect_local_mode()`
checks, in order, the one shell's own explicit value, `DATABASE_URL` (also
set by supervisord's own child environment, and readable independent of
the shell), `/proc/1/environ` (supervisord — pid 1 in this image — carries
the REAL value even when the invoking shell does not), and finally the
filesystem/process table (local file exists and something has it open),
before falling back to the same "empty means true" default `webuidb.py`
itself uses.
"""

import os
import sys
from pathlib import Path

# Same defaults and same override env vars as compactor/webuidb.py, so a
# test (or an operator) pointing that module somewhere else via
# WEBUI_LOCAL_DB/WEBUI_SNAPSHOT_DB is automatically in sync with this one.
LOCAL_DB = Path(os.environ.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db"))
SNAPSHOT_DB = Path(os.environ.get("WEBUI_SNAPSHOT_DB", "/data/openwebui/webui.db"))
FORENSICS_ROOT = Path(os.environ.get("WEBUI_DB_FORENSICS", "/data/forensics"))

# D1/the rehearsal's own "Final sync" recipe — the ONLY documented way to
# force a local-mode write to reach the durable snapshot immediately,
# rather than waiting for (or risking a pod stop before) the next
# WEBUI_DB_SYNC_INTERVAL_S cycle. There is no `--sync-now`; per the
# rehearsal, `--help` or any unknown flag starts a second daemon loop (D6).
FINAL_SYNC_COMMANDS = (
    "supervisorctl stop openwebui webuidb-sync",
    "WEBUI_DB_LOCAL=true /opt/compactor-venv/bin/python /opt/compactor/webuidb.py --sync-once --force",
)


def fold_webui_db_local(raw):
    """The exact fold `compactor/webuidb.py`'s `live_webui_db()` applies to
    `WEBUI_DB_LOCAL`: trimmed, case-insensitive; "", "true", "1", "yes",
    "on" -> True; "false", "0", "no", "off" -> False. Returns None for
    anything else (webuidb.py itself raises there; this helper only ever
    ADVISES operator scripts, so it falls through to the other detection
    methods instead of crashing one over a stray value)."""
    v = (raw or "").strip().lower()
    if v in ("", "true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return None


def _database_url_hints_local():
    """Best-effort read of `DATABASE_URL` (also set only in supervisord's
    child environment, alongside `WEBUI_DB_LOCAL` — see entrypoint.sh):
    True if it names LOCAL_DB, False if it names SNAPSHOT_DB, None if
    unset or naming neither."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return None
    if str(LOCAL_DB) in url:
        return True
    if str(SNAPSHOT_DB) in url:
        return False
    return None


def _proc1_environ_webui_db_local():
    """`WEBUI_DB_LOCAL` out of `/proc/1/environ` — pid 1 in this image is
    supervisord, which carries the REAL value even in a fresh Web Terminal
    shell that never inherited it. Best-effort: None on any failure
    (permissions, no /proc, the var simply not present there)."""
    try:
        raw = Path("/proc/1/environ").read_bytes()
    except OSError:
        return None
    for part in raw.split(b"\0"):
        if part.startswith(b"WEBUI_DB_LOCAL="):
            try:
                return part.split(b"=", 1)[1].decode("utf-8", "replace")
            except Exception:
                return None
    return None


def _local_db_open_by_a_process():
    """True if LOCAL_DB exists AND some OTHER process on this host has it
    open — a `/proc/<pid>/fd` scan, the same check the writing scripts
    already run against their own target before a write. Best-effort:
    never raises."""
    if not LOCAL_DB.exists():
        return False
    try:
        real = os.path.realpath(str(LOCAL_DB))
    except OSError:
        return False
    me = str(os.getpid())
    try:
        pids = os.listdir("/proc")
    except OSError:
        return False
    for pid in pids:
        if not pid.isdigit() or pid == me:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(f"/proc/{pid}/fd/{fd}") == real:
                    return True
            except OSError:
                pass
    return False


def detect_local_mode():
    """Returns `(is_local, reason)`. Priority, highest first — see the
    module docstring for why this order, not the env var alone:

      1. `WEBUI_DB_LOCAL` set (non-empty) in THIS process's own environment.
      2. `DATABASE_URL`, if set, naming LOCAL_DB or SNAPSHOT_DB.
      3. `/proc/1/environ`'s `WEBUI_DB_LOCAL` (supervisord's real
         environment — survives a fresh shell that inherited neither of
         the above).
      4. LOCAL_DB exists and some other process has it open.
      5. Default True — empty/unset means local, exactly like
         `webuidb.live_webui_db()`.
    """
    raw_here = os.environ.get("WEBUI_DB_LOCAL")
    if raw_here is not None and raw_here.strip() != "":
        folded = fold_webui_db_local(raw_here)
        if folded is not None:
            return folded, f"WEBUI_DB_LOCAL={raw_here!r} (this shell's own environment)"

    from_url = _database_url_hints_local()
    if from_url is not None:
        return from_url, f"DATABASE_URL points at {'local disk' if from_url else 'the snapshot'}"

    proc1_raw = _proc1_environ_webui_db_local()
    if proc1_raw is not None:
        folded = fold_webui_db_local(proc1_raw)
        if folded is not None:
            return folded, f"/proc/1/environ WEBUI_DB_LOCAL={proc1_raw!r} (supervisord's own environment)"

    if _local_db_open_by_a_process():
        return True, f"{LOCAL_DB} exists and is open by a running process"

    return True, "WEBUI_DB_LOCAL unset/empty -- defaults to local, matching webuidb.live_webui_db()"


def live_db_path():
    """`(path, is_local, reason)` -- the live database RIGHT NOW, by the
    same rule `compactor/webuidb.py`'s `live_webui_db()` uses, widened
    with the filesystem/process fallbacks above for a shell that never
    inherited the flag."""
    is_local, reason = detect_local_mode()
    return (LOCAL_DB if is_local else SNAPSHOT_DB), is_local, reason


def _same_file(a, b):
    a, b = Path(a), Path(b)
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a) == str(b)


def snapshot_problem_message(target_path, tool_name, *, dry_run):
    """The D3 message for `target_path`, or `None` if there is nothing to
    say (not the snapshot, or local mode is not active). Does not print
    or exit -- for a caller with its own reporting convention (e.g.
    `clean-decoration.py`'s JSON-aware `_fatal`/`_print_report`).
    `dry_run=True` gives the softer "warn, still allowed" wording;
    `dry_run=False` gives the refusal wording."""
    target = Path(target_path)
    if not _same_file(target, SNAPSHOT_DB):
        return None
    is_local, reason = detect_local_mode()
    if not is_local:
        return None
    live = LOCAL_DB
    if dry_run:
        return (
            f"{target} is the periodically-published SNAPSHOT, and local mode "
            f"looks ACTIVE ({reason}). The live database right now is {live} -- "
            f"point {tool_name} there instead before trusting these counts for "
            f"an --apply."
        )
    return (
        f"{target} is the periodically-published SNAPSHOT, and local mode looks "
        f"ACTIVE ({reason}). Writing here edits the snapshot only -- the next "
        f"sync cycle publishes the LOCAL database ({live}) over it and this "
        f"write is silently lost, with no warning (D3, dbmove/findings.md "
        f"section 4). Stop `openwebui` AND `webuidb-sync` first, then point "
        f"{tool_name} at the live path instead: {live}"
    )


def refuse_if_snapshot_in_local_mode(target_path, *, tool_name, dry_run):
    """Call this before any read or write against `target_path`.

    In a WRITING mode (`dry_run=False` -- `--apply`/`--restore`): prints
    the D3 refusal to stderr and exits 1 when `target_path` is the
    SNAPSHOT while local mode looks ACTIVE. Silent (no output at all)
    when the target is fine, so a script's own normal dry-run/apply
    output is not cluttered on every ordinary run.

    In a READ-ONLY dry run (`dry_run=True`): only WARNS, to stderr, and
    still lets the dry run proceed -- an operator reading a projection is
    not editing anything yet, and this dry run's own counts still need to
    reach them even if they are about to be stale.

    Does nothing at all when `target_path` is not the snapshot (including
    every test fixture that points at neither LOCAL_DB nor SNAPSHOT_DB) --
    this function only ever fires on the one specific wrong-path shape D3
    describes. For a caller that needs the message without the direct
    print/exit (e.g. to fold into its own JSON output), use
    `snapshot_problem_message` instead."""
    msg = snapshot_problem_message(target_path, tool_name, dry_run=dry_run)
    if msg is None:
        return
    print(("WARNING: " if dry_run else "REFUSING: ") + msg, file=sys.stderr)
    if not dry_run:
        sys.exit(1)


def maybe_print_final_sync_hint(tool_name, db_path):
    """Call this once a write (`--apply`/`--restore`) against `db_path` has
    actually committed successfully. Prints the exact final-sync recipe
    (D1) ONLY when `db_path` -- the file this run actually just wrote --
    IS LOCAL_DB. Deliberately gated on the real path just written, not on
    the generic `detect_local_mode()` guess: that guess defaults to True
    on an empty/unset `WEBUI_DB_LOCAL` (matching `webuidb.live_webui_db()`
    -- see the module docstring), which would otherwise print this hint
    after EVERY write anywhere, including every test fixture and every
    write to a `/data`-placed database on a host with no pod environment
    at all. Nothing publishes a write to `db_path` itself unless `db_path`
    literally is the local, ephemeral file."""
    if not _same_file(db_path, LOCAL_DB):
        return
    print("")
    print(f"{tool_name} wrote the LOCAL database ({LOCAL_DB}). Nothing publishes it to "
          f"the durable snapshot ({SNAPSHOT_DB}) until the next sync interval, and a "
          f"pod stop before then loses it outright -- run this now to make it durable:")
    for c in FINAL_SYNC_COMMANDS:
        print(f"    {c}")


def forensics_backup_dir(tool_name, stamp, db_path):
    """D10: where THIS tool's own pre-write safety backup should live when
    `db_path` IS the local, ephemeral disk file (`LOCAL_DB`) -- a backup
    left beside it would ride the same container overlay and be lost on a
    pod stop exactly like everything else there. Returns
    `/data/forensics/<tool_name>-<stamp>/` in that case (durable, matching
    the convention `scripts/fix-stale-unfinished.py` already uses), or
    `None` when `db_path` is NOT the local file -- the caller's existing
    beside-the-db backup is already wherever `db_path` itself lives, and
    this function only ever REDIRECTS local-disk backups, it never
    replaces a working scheme with a worse one."""
    if not _same_file(db_path, LOCAL_DB):
        return None
    return FORENSICS_ROOT / f"{tool_name}-{stamp}"
