"""`zl fix <name> [--apply] [--yes] [--skip-backup] [-- script args]` -- a
REGISTRY of the repair/maintenance scripts, one entry per script. This
wrapper never re-implements a script's own logic (memory.md, Mistakes log:
"two copies of one rule drifting apart is this codebase's most expensive
recurring defect") -- it only picks the interpreter, decides what to stop,
runs the unmodified script, and puts services back.

`zl fix` with no name lists the registry.

Apply flow (brief):
  1. unless --skip-backup: `compactor/backup.py --once`, require exit 0
     (safe with services up -- backup.py's own module docstring, principle
     2, "Live-SQLite-safe")
  2. record which of the entry's services are RUNNING right now
  3. stop them
  4. run the script with its apply args
  5. in a `finally`: start EXACTLY the ones that were running before --
     never one that was already stopped -- and verify they came back
  6. print the exit-code meaning (done by cli.py's own wrapper)
A KeyboardInterrupt during step 4 still runs step 5 (Python's try/finally).
Dry run = run the script without its apply args, no stops, no backup.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

from .. import common, podenv


def _live_db_writers() -> tuple[str, ...]:
    """The pair of services a repair on webui.db stops, generalized across
    pod placement: RUNBOOK_CHAT_TREE.md step 3/4 and OPERATIONS.md's
    fix-stale-unfinished.py section both stop `openwebui` + `backup` on
    this pod's own placement (WEBUI_DB_LOCAL=false); RUNBOOK_CHAT_TREE.md's
    "If this pod places webui.db on local disk" note and OPERATIONS.md's
    same section both say to stop `webuidb-sync` INSTEAD of `backup` when
    WEBUI_DB_LOCAL=true (webuidb-sync would otherwise overwrite the repair
    with the local copy on its next sync). Uses podenv's own single
    detector, not a second copy of that fold.
    """
    is_local, _why = podenv.detect_webui_db_local()
    return ("openwebui", "webuidb-sync") if is_local else ("openwebui", "backup")


@dataclasses.dataclass(frozen=True)
class FixEntry:
    name: str
    script: str  # filename under scripts/
    interpreter: Path
    read_only: bool
    dry_run_args: tuple[str, ...]
    apply_args: tuple[str, ...]
    services: Callable[[], tuple[str, ...]]  # evaluated at apply time, not import time
    services_source: str  # citation, shown in `zl fix`'s own listing
    default_args: Callable[[], tuple[str, ...]] | None  # used only if the operator gives none after `--`
    help: str
    # The SCRIPT's own write flags (from its argparse, not guessed -- review
    # B2), checked against anything passed after `--`. `--apply` is always
    # refused there regardless of this list (common.refuse_passthrough_write_flags).
    write_flags: tuple[str, ...] = ()
    # The SCRIPT's own COMPLETE declared long-option list (every flag its
    # argparse defines, read from `scripts/<script>.py`'s own
    # `add_argument` calls -- review R2-B2), used to normalize `--flag=value`
    # and unambiguous-prefix abbreviations (`--rest`, `--appl`, `--forc`)
    # before the write-flag check runs, the same way the script's own
    # argparse (abbreviations on by default) would resolve them.
    all_flags: tuple[str, ...] = ()


REGISTRY: dict[str, FixEntry] = {}


def _register(entry: FixEntry) -> None:
    REGISTRY[entry.name] = entry


_register(FixEntry(
    name="repair-chat-tree",
    script="repair-chat-tree.py",
    interpreter=podenv.WEBUI_PY,
    read_only=False,
    dry_run_args=(),
    apply_args=("--apply",),
    services=_live_db_writers,
    services_source="RUNBOOK_CHAT_TREE.md:76-77 (runbook's own description of step 3: openwebui+backup on this "
                     "placement; +webuidb-sync instead of backup when WEBUI_DB_LOCAL=true) and step 4 (line 148, the "
                     "restart)",
    default_args=lambda: (str(podenv.live_webui_db()),),
    help="relinks a chat's message tree from chat_message rows (RUNBOOK_CHAT_TREE.md)",
    write_flags=("--apply", "--restore"),
    all_flags=('--apply', '--chat', '--json', '--restore', '--sync-json-only'),
))

_register(FixEntry(
    name="fix-stale-unfinished",
    script="fix-stale-unfinished.py",
    interpreter=podenv.WEBUI_PY,
    read_only=False,
    dry_run_args=(),
    apply_args=("--apply",),
    services=_live_db_writers,
    services_source="OPERATIONS.md:2230,2235 (`supervisorctl stop openwebui backup` / `start openwebui backup`; "
                     "+webuidb-sync instead of backup when WEBUI_DB_LOCAL=true, per the script's own D3 refusal)",
    default_args=lambda: (str(podenv.live_webui_db()),),
    help="clears a message stuck 'unfinished' after a lost stream (OPERATIONS.md)",
    write_flags=("--apply", "--restore"),
    all_flags=('--all-branches', '--apply', '--branch-only', '--chat', '--json', '--min-age-minutes', '--restore'),
))

_register(FixEntry(
    name="fix-encoded-messages",
    script="fix-encoded-messages.py",
    interpreter=podenv.WEBUI_PY,
    read_only=False,
    dry_run_args=(),
    apply_args=("--apply",),
    services=_live_db_writers,
    services_source="same repair session as repair-chat-tree.py, RUNBOOK_CHAT_TREE.md step 3",
    default_args=None,  # needs BOTH `live` and `pre` (a pre-repair snapshot) -- no safe default for `pre`
    help="restores messages the retired v1 tree-repair left encoded (RUNBOOK_CHAT_TREE.md); "
         "pass -- <live db> <pre-repair snapshot> [--chat ID]",
    write_flags=("--apply", "--restore"),
    all_flags=('--apply', '--chat', '--json', '--restore'),
))

_register(FixEntry(
    name="clean-decoration",
    script="clean-decoration.py",
    interpreter=podenv.COMPACTOR_PY,
    read_only=False,
    dry_run_args=(),
    apply_args=("--apply",),
    services=lambda: ("openwebui", "compactor"),
    services_source="OPERATIONS.md:2040,2048 (`supervisorctl stop openwebui compactor` / `start openwebui compactor`)",
    default_args=None,  # --conv CONV_ID is required (no safe default)
    help="cleans stray control/decoration text from a conversation; pass -- --conv CONV_ID",
    write_flags=("--apply", "--force", "--restore"),
    all_flags=('--apply', '--compactor-pkg', '--conv', '--debug-skip-anchor-rewrite', '--force', '--health-url', '--json', '--l1-chunk-size', '--last', '--only', '--openwebui-port', '--restore', '--since-summarized', '--status-tag-pattern', '--store', '--webui-db'),
))

_register(FixEntry(
    name="import-history",
    script="import-history.py",
    interpreter=podenv.COMPACTOR_PY,
    read_only=False,
    dry_run_args=(),
    apply_args=("--apply",),
    services=lambda: ("compactor",),
    services_source="scripts/import-history.py:149-150,177-178 POD PROCEDURE steps 2 and 6 "
                     "(`supervisorctl stop compactor` before, `start compactor` after -- it writes the "
                     "live --store, and its own health probe refuses --apply while the compactor answers)",
    default_args=None,  # --webui-db (an export path) and --chat-id are required
    help="reconstructs compactor memory for one chat from a webui.db export; pass -- --webui-db PATH --chat-id ID",
    write_flags=("--apply", "--force"),
    # review LOW: this list was hand-copied from import-history.py's own
    # argparse and missed `--compactor-pkg` (added later in the same file,
    # scripts/import-history.py:1433-1439) -- harmless in the SAFE
    # direction (a real `--compactor-pkg` would have been refused by
    # `refuse_passthrough_write_flags`'s "not in all_flags at all" no-op
    # path... actually normalize_passthrough_token just returns it
    # unchanged when it's not a prefix of anything, and it is then simply
    # not in write_flags either, so it passed through fine either way; the
    # gap was cosmetic, not a hole), but it is exactly the drift
    # `test_ops_fix.py`'s AST-based check now catches automatically.
    all_flags=('--apply', '--chat-id', '--compactor-pkg', '--conv-id', '--force', '--health-url', '--json', '--max-calls', '--model', '--seconds-per-call', '--store', '--vllm-url', '--webui-db'),
))

_register(FixEntry(
    name="backfill-records",
    script="backfill-records.py",
    interpreter=podenv.COMPACTOR_PY,
    read_only=False,
    dry_run_args=(),
    apply_args=("--apply",),
    services=lambda: ("compactor",),
    services_source="OPERATIONS.md:1405-1408 (refuses to run while anything answers the compactor's /health; "
                     "stop it first: `supervisorctl stop compactor`)",
    default_args=None,  # --store has its own default; --conv is optional
    help="closes stale would-resume backfill records that already have a facts file",
    write_flags=("--apply", "--force"),
    all_flags=('--store', '--conv', '--apply', '--json', '--force', '--compactor-pkg', '--health-url'),
))

_register(FixEntry(
    name="merge-conversations",
    script="merge-conversations.py",
    interpreter=podenv.COMPACTOR_PY,
    read_only=False,
    dry_run_args=(),
    apply_args=("--apply",),
    services=lambda: ("compactor",),
    services_source="scripts/merge-conversations.py docstring (supervisorctl stop compactor ... start compactor)",
    default_args=None,  # --src and --dst are both required
    help="merges one conversation's memory into another; pass -- --src ID --dst ID",
    write_flags=("--apply", "--allow-live"),
    all_flags=('--allow-live', '--apply', '--dst', '--port', '--src'),
))

_register(FixEntry(
    name="compare-live-to-backup",
    script="compare-live-to-backup.py",
    interpreter=podenv.COMPACTOR_PY,
    read_only=True,
    dry_run_args=(),
    apply_args=(),
    services=lambda: (),
    services_source="read-only, both sides opened read-only -- never needs a stop (script's own docstring)",
    default_args=None,  # every flag defaults sensibly (--backup-dir, --live-db, --live-compactor)
    help="what is in the newest backup that is NOT live right now (read-only, safe any time)",
    write_flags=(),
    all_flags=('--backup', '--backup-dir', '--live-compactor', '--live-db'),
))

_register(FixEntry(
    name="switch-webui-db-to-local",
    script="switch-webui-db-to-local.py",
    interpreter=podenv.COMPACTOR_PY,
    read_only=False,
    dry_run_args=("--check",),
    apply_args=("--apply",),
    services=lambda: (),
    services_source="scripts/switch-webui-db-to-local.py manages its own supervisord edits/restarts; "
                     "no external stop is documented",
    default_args=None,
    help="moves webui.db to local disk with periodic /data sync (D3 fix for MooseFS-hosted SQLite); "
         "ONE-OFF per pod recreate, only when moving to local mode -- not a repeated action",
    write_flags=("--apply",),
    all_flags=('--apply', '--check'),
))


def _list_registry() -> int:
    print("zl fix <name> [--apply] [--yes] [--skip-backup] [-- script args]\n")
    width = max(len(n) for n in REGISTRY)
    for name in sorted(REGISTRY):
        entry = REGISTRY[name]
        tag = " (read-only)" if entry.read_only else ""
        print(f"  {name.ljust(width)}{tag}  {entry.help}")
        print(f"      script: scripts/{entry.script}  interpreter: {entry.interpreter}")
        if not entry.read_only:
            print(f"      stops before --apply: {entry.services_source}")
    return 0


def run(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        return _list_registry()

    name, rest = argv[0], argv[1:]
    entry = REGISTRY.get(name)
    if entry is None:
        print(f"zl fix: unknown entry {name!r}\n")
        _list_registry()
        return 2

    own, passthrough = common.split_passthrough(rest)
    apply = "--apply" in own
    yes = "--yes" in own
    skip_backup = "--skip-backup" in own
    force_busy = "--force-busy" in own
    unknown = [a for a in own if a not in ("--apply", "--yes", "--skip-backup", "--force-busy")]
    if unknown:
        print(f"zl fix {name}: unrecognized flag(s) before `--`: {unknown}")
        return 2

    # Review B2: a write flag of the SCRIPT's own after `--` must go
    # through this wrapper's own --apply gate, never sneak past it.
    gate = common.refuse_passthrough_write_flags(
        passthrough, entry.write_flags, zl_apply=apply, what=f"run {entry.script}", all_flags=entry.all_flags
    )
    if gate is not None:
        return gate

    script_path = podenv.scripts_dir() / entry.script
    if not script_path.is_file():
        print(f"zl fix {name}: {script_path} not found -- is this checkout missing scripts/?")
        return 1

    if passthrough and passthrough[0].startswith("-") and entry.default_args is not None:
        # review LOW: passthrough that is ALL FLAGS (e.g. `-- --restore
        # STAMP`) used to REPLACE default_args entirely, silently dropping
        # the required positional db path this entry's default_args exists
        # to supply -- the script then runs with no db argument at all.
        # Only prepend the default when passthrough looks flag-only (does
        # not start with a bare path/positional of its own); an operator
        # who DOES give their own positional (e.g. a different db path)
        # still gets exactly what they typed, unprefixed.
        script_args = list(entry.default_args()) + list(passthrough)
    elif passthrough:
        script_args = list(passthrough)
    elif entry.default_args is not None:
        script_args = list(entry.default_args())
    else:
        script_args = []

    if entry.read_only:
        if apply:
            print(f"zl fix {name}: this entry is read-only; --apply has no effect")
        return common.run_stream(
            [str(entry.interpreter), str(script_path), *script_args], env=common.merged_environ()
        )

    if not apply:
        return common.run_stream(
            [str(entry.interpreter), str(script_path), *entry.dry_run_args, *script_args],
            env=common.merged_environ(),
        )

    common.require_apply_confirmation(apply, yes, what=f"run {entry.script} with --apply")

    # Round-4 MEDIUM: the backup used to run HERE, before the in-flight
    # wait -- so a request that then took minutes to settle left an
    # already-taken backup stale relative to whatever changed meanwhile,
    # and (worse) burned the wait's own timeout budget after the backup
    # instead of before it. It now runs AFTER the wait settles, inside the
    # protected region, immediately before the first stop (see below).
    services = list(entry.services())
    # Review R2-B1: the protected region (SIGHUP ignored, SIGINT/SIGTERM
    # deferred rather than dropped) spans from BEFORE the first status
    # read/stop through the END of restart_after_apply -- not just the
    # child's own run -- because a hangup or Ctrl-C anywhere in that
    # window used to abandon the pod stopped with no audit line.
    with common.ProtectedRegion() as region:
        # Round-4 addendum (v3.1.9.8): check for her own in-flight
        # request(s) BEFORE the first stop -- a restart this week killed
        # one mid-stream. Inside the protected region on purpose: this
        # wait can itself run for minutes, and a hangup/Ctrl-C here must
        # be survived the same way the stop/restart phases already are.
        #
        # Round-4 blocker #3: while this wait is running, NOTHING has
        # been stopped yet, so the generic "services are being stopped/
        # restored" deferred-signal message (ProtectedRegion._handle's
        # default) is misleading here -- override it for the duration.
        region.set_message(
            "zl: signal received while waiting for her in-flight request(s) to finish -- "
            "nothing has been stopped yet; this will be acted on once the wait ends"
        )
        proceed, note = common.wait_for_no_inflight_requests(force=force_busy)
        region.clear_message()
        if not proceed:
            common.safe_print(f"zl fix {name}: refusing -- {note}", file=sys.stderr)
            region.finish(1, restored=True)
            return 1
        # Round-4 blocker #3: a signal deferred DURING the wait above (the
        # guard itself returned proceed=True on its own next reading, but
        # a Ctrl-C landed while it was still polling) must abort here --
        # nothing has been stopped, so there is nothing to restore, but
        # applying anyway after an operator hit Ctrl-C would be exactly
        # the "stops anyway" bug this review found.
        if region.deferred_signal is not None:
            common.safe_print(
                f"zl fix {name}: signal received during the in-flight wait -- nothing was stopped; "
                "aborting without applying",
                file=sys.stderr,
            )
            region.finish(1, restored=True)
            return 1

        if not skip_backup:
            common.safe_print("taking a backup first (compactor/backup.py --once) ...")
            backup_py = podenv.COMPACTOR_PKG / "backup.py"
            # Round-7: nothing has been stopped yet during the backup
            # itself either -- same reasoning as the in-flight-wait
            # `set_message` calls above; the generic "services are being
            # stopped/restored" default (ProtectedRegion._handle's own
            # fallback) would be just as misleading here, since backup.py
            # runs with every service still up. `backup.py` is its own
            # process (not run via `run_protected`, so no child is
            # registered here for the region to forward to) -- a signal
            # during it is deferred and acted on only once it exits.
            region.set_message(
                "zl: signal received while backup.py runs -- nothing has been stopped yet; "
                "this will be acted on once the backup finishes"
            )
            bcode = common.run_stream(
                [str(podenv.COMPACTOR_PY), str(backup_py), "--once"], env=common.merged_environ()
            )
            region.clear_message()
            if bcode != 0:
                common.safe_print(f"zl fix {name}: backup.py --once exited {bcode}; refusing to apply "
                                   "(pass --skip-backup to override)")
                region.finish(common.EXIT_BACKUP_FAILED, restored=True)
                return common.EXIT_BACKUP_FAILED

        # Round-6 BLOCKER: the backup itself can run for a while (compactor/
        # backup.py --once with a large webui.db), and her next request can
        # start DURING it -- the wait above only proved nothing was in
        # flight the MOMENT it finished, not that nothing started in the
        # gap between then and now. Re-check requests_in_flight, and
        # re-check for a signal deferred during either wait, immediately
        # before the first stop -- same reasoning as the first check, one
        # window later. Skipped when skip_backup is set: there is no gap to
        # re-check across in that case, and the first check above already
        # covers the window immediately before this point.
        if not skip_backup:
            region.set_message(
                "zl: signal received while waiting (again, after the backup) for her in-flight "
                "request(s) to finish -- nothing has been stopped yet; this will be acted on once "
                "the wait ends"
            )
            proceed, note = common.wait_for_no_inflight_requests(force=force_busy)
            region.clear_message()
            if not proceed:
                common.safe_print(f"zl fix {name}: refusing -- {note}", file=sys.stderr)
                region.finish(1, restored=True)
                return 1
            if region.deferred_signal is not None:
                common.safe_print(
                    f"zl fix {name}: signal received during the post-backup in-flight wait -- "
                    "nothing was stopped; aborting without applying",
                    file=sys.stderr,
                )
                region.finish(1, restored=True)
                return 1

        try:
            was_running = common.stop_for_apply(services)
        except common.StopVerificationFailed as e:
            code, restored = common.refuse_after_stop_failure(e)
            region.finish(code, restored=restored)
            return code
        # Round-6 REVERSAL: this used to say "deliberately NOT re-checking
        # region.deferred_signal here" -- a signal that arrived DURING/after
        # `stop_for_apply` (before the child had even started) was treated
        # the SAME as one arriving while the child runs: let it finish, then
        # reflect the signal only in `region.finish()`'s exit code. That is
        # wrong for THIS window specifically: nothing but the stop itself
        # has happened yet, so an operator's Ctrl-C/SIGTERM here means "stop,
        # don't apply" just as much as one during the in-flight wait above
        # does -- there is no reason to go on to run a fix the operator
        # already tried to cancel. Restore exactly what was stopped and
        # abort with the shell's 128+N convention, distinct from the
        # in-flight-wait abort's plain exit 1 above (there, nothing was
        # stopped; here, it was, and the message must say so).
        if region.deferred_signal is not None:
            sig = region.deferred_signal
            came_back = common.restart_after_apply(was_running)
            restored = common.report_restart_failures(came_back)
            common.safe_print(
                f"zl fix {name}: cancelled before the fix was applied; "
                + ("services restored" if restored else "services NOT fully restored -- see the WARNING above"),
                file=sys.stderr,
            )
            return 128 + sig if restored else 1
        # Review B1: this print USED to sit here, between `stop_for_apply`
        # succeeding and the `try` around `run_protected` -- outside any
        # protection at all. A hangup landing in that exact gap (verified
        # possible: `stop_for_apply` can block up to its own
        # `verify_timeout`, well within the window an SSH client takes to
        # notice a drop and close the pty) used to be able to raise
        # `OSError` right here, escaping this function with the services
        # already stopped and `restart_after_apply` never called. It is
        # now inside the same `try`/`finally` as the run itself, and
        # routed through `common.safe_print` (defense in depth alongside
        # `ProtectedRegion`'s own SIGHUP dup2 -- see its docstring).
        code = 1
        restored = True
        try:
            common.safe_print(f"stopped (were running before): {was_running or '(none of them were running)'}")
            code = common.run_protected(
                [str(entry.interpreter), str(script_path), *entry.apply_args, *script_args],
                name=f"fix-{name}",
                env=common.merged_environ(),
                region=region,
            )
        finally:
            came_back = common.restart_after_apply(was_running)
            restored = common.report_restart_failures(came_back)
        # Round-6 HIGH: `restored` now reflects whether every service
        # actually came back RUNNING (not just "we called restart_after_
        # apply") -- passed to `finish()` so its wording never claims
        # "restored" when one didn't, and forced into the exit code below
        # so a run that applied cleanly (child exit 0) but left a service
        # down still exits nonzero instead of a silent 0.
        if not restored and code == 0:
            code = 1
        region.finish(code, restored=restored, applied=True)
    return code
