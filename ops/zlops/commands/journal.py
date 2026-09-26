"""`zl journal check` / `zl journal repair [--apply]` -- wraps
scripts/recover-webui-db.py, the hot-SQLite-rollback-journal repair
(OPERATIONS.md "A HOT SQLite rollback journal beside webui.db").

recover-webui-db.py's own flags (its docstring / argparse, not this
wrapper's): `--check` diagnoses only, never stops OpenWebUI, never writes;
the default (no `--check`) run is the real recovery -- it stops/starts
OpenWebUI ITSELF, so this wrapper does not. `--yes` skips its own
interactive confirmation prompt (used here once WE have already confirmed).

Services this wrapper stops before the real run (RUNBOOK_DB_JOURNAL.md:139-
152: "Every writer must be stopped first: OpenWebUI, the compactor and the
backup daemon. On this pod webuidb-sync does not run" -- webuidb-sync is
added too when podenv.detect_webui_db_local() says this pod actually runs
it): `compactor`, `backup`, and `webuidb-sync` if WEBUI_DB_LOCAL is true.
OpenWebUI itself is deliberately NOT in this wrapper's stop/start list --
the script manages it.
"""

from __future__ import annotations

import sys

from .. import common, podenv

_SCRIPT = podenv.scripts_dir() / "recover-webui-db.py"
# recover-webui-db.py's own documented exit codes (every `return N` site,
# read from scripts/recover-webui-db.py -- review M1 re-reads all of them,
# not just the three the previous round tracked):
#
#   2 = webui.db does not exist (checked before anything else runs). NOT a
#       usage error -- see cli.py's `_journal_exit_note`, which fixes the
#       citation that used to print the generic "usage error" text for
#       this (`zl journal check`'s own most common failure mode).
#   1 = an earlier pre-flight refusal: a bad D3 snapshot-placement check
#       (webui.db looks like the WEBUI_DB_LOCAL snapshot), the operator
#       declining the script's own confirmation prompt, or no tty to ask
#       on. NOTHING was touched -- every one of this code's `return 1`
#       sites is strictly BEFORE the script's own `supervisor("stop")`
#       call, so OpenWebUI is exactly as it was before this ran. Review
#       M1: this used to be lumped into the leave-stopped set, which left
#       compactor/backup(/webuidb-sync) stopped and told the operator "do
#       NOT start openwebui" -- while openwebui was never stopped at all,
#       so chat was dead for no reason. Now restarts whatever THIS
#       wrapper stopped, same as any other non-leave-stopped exit.
#   3 = the copy-to-/tmp step itself failed (disk full, permissions, ...);
#       5 = the recovered copy opened but reports no chats (not this
#       failure's shape). Both call the script's OWN `supervisor("start")`
#       for OpenWebUI before returning, so this wrapper restarts its own
#       writers too, the same as success.
#   4 = integrity_check on the recovered copy did not say "ok".
#       RUNBOOK_DB_JOURNAL.md section 6: "Do not start OpenWebUI. Leave
#       the services stopped." The script's OWN `supervisor("start")`
#       call in this branch cannot be trusted -- integrity already
#       failed, restoring exactly the archive-instead situation the
#       runbook describes -- so this wrapper does NOT rely on it: it
#       treats OpenWebUI as down and leaves compactor/backup(/webuidb-
#       sync) stopped too. Review M1: 4 used to be missing from this set
#       entirely (fell through to the normal restart path).
#   6 = the rename+copy2 swap itself raised; 7 = a sidecar file still
#       sits beside the new database after the swap ("do NOT start
#       OpenWebUI. Investigate before continuing." -- the script's own
#       message). Neither of these restarts OpenWebUI itself.
#
# RUNBOOK_DB_JOURNAL.md section 6 covers 4/6/7 identically: leave the
# services stopped, do not touch webui.db further, ask for help.
_LEAVE_STOPPED_EXIT_CODES = frozenset({4, 6, 7})
# recover-webui-db.py's own argparse: only --check and --yes. Neither is a
# "write flag" in review B2's sense (this wrapper already controls --check
# itself), but --check IS refused in `repair`'s own passthrough below: it
# would silently turn an --apply'd repair into a no-op check instead of
# sneaking a write past the gate -- the opposite failure mode, still worth
# refusing outright rather than confusing the operator. Review R2-B2: the
# refusal used to be an exact-match `in` test, so `--check=1` or `--che`
# sailed straight through (recover-webui-db.py's own argparse would accept
# either as `--check`, same as every other wrapped script here).
_ALL_FLAGS = ("--check", "--yes")
_REPAIR_DISALLOWED = ("--check",)


def _writers_to_stop() -> list[str]:
    is_local, _why = podenv.detect_webui_db_local()
    names = ["compactor", "backup"]
    if is_local:
        names.append("webuidb-sync")
    return names


def _env_with_webui_db() -> dict[str, str]:
    env = common.merged_environ()  # review B3: the real WEBUI_DB_LOCAL etc., not just this shell's
    env["WEBUI_DB"] = str(podenv.live_webui_db())
    return env


def _run_check(extra: list[str]) -> int:
    return common.run_stream(
        [str(podenv.COMPACTOR_PY), str(_SCRIPT), "--check", *extra],
        env=_env_with_webui_db(),
    )


def run(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        # Round-6 cosmetic: `zl journal --help` used to fall all the way
        # through to "unknown subcommand '--help'" (exit 2) -- the same
        # bare usage line as no-argv at all, but as a REQUEST for help,
        # not a usage ERROR (exit 0, matching every other `zl` command's
        # own -h/--help convention, e.g. fix.py's `_list_registry`).
        print("usage: zl journal check | zl journal repair [--apply] [--yes]")
        return 0 if argv else 2
    sub, rest = argv[0], argv[1:]

    if sub == "check":
        return _run_check(rest)

    if sub != "repair":
        print(f"zl journal: unknown subcommand {sub!r}; usage: zl journal check | zl journal repair [--apply]")
        return 2

    apply = "--apply" in rest
    yes = "--yes" in rest
    force_busy = "--force-busy" in rest
    extra = [a for a in rest if a not in ("--apply", "--yes", "--force-busy")]

    if not apply:
        print("dry run (pass --apply to actually repair) -- running the script's own --check:")
        return _run_check(extra)

    try:
        checked = common.normalize_passthrough(extra, _ALL_FLAGS)
    except common.AmbiguousPassthroughFlag as e:
        print(f"zl journal repair: refusing -- {e}")
        return 2
    bad = [a for a in checked if a in _REPAIR_DISALLOWED]
    if bad:
        print(f"zl journal repair: refusing -- {bad} would silently turn --apply into a no-op check; "
              "use `zl journal check` for that instead")
        return 2

    common.require_apply_confirmation(apply, yes, what="repair a hot SQLite rollback journal on webui.db")

    writers = _writers_to_stop()
    # Review R2-B1 + HIGH: the protected region spans stop -> run -> restart
    # exactly like `zl fix`'s, EXCEPT this run never forwards SIGINT/SIGTERM
    # to recover-webui-db.py at all (see `run_protected`'s own docstring on
    # `forward_signals=False`) -- RUNBOOK_DB_JOURNAL.md says not to
    # interrupt its rename+copy2 swap, and the script's own `except
    # Exception` does not catch KeyboardInterrupt.
    with common.ProtectedRegion() as region:
        # Round-4 addendum (v3.1.9.8): check for her own in-flight
        # request(s) BEFORE the first stop -- same reasoning as `zl fix`
        # (common.wait_for_no_inflight_requests's own docstring): inside
        # the protected region on purpose, since this wait can itself run
        # for minutes.
        # Round-4 blocker #3: while this wait is running, NOTHING has been
        # stopped yet, so the generic "services are being stopped/
        # restored" deferred-signal message is misleading here -- override
        # it for the duration.
        region.set_message(
            "zl: signal received while waiting for her in-flight request(s) to finish -- "
            "nothing has been stopped yet; this will be acted on once the wait ends"
        )
        proceed, note = common.wait_for_no_inflight_requests(force=force_busy)
        region.clear_message()
        if not proceed:
            common.safe_print(f"zl journal repair: refusing -- {note}", file=sys.stderr)
            region.finish(1, restored=True)
            return 1
        # Round-4 blocker #3: a signal deferred DURING the wait must abort
        # here -- nothing has been stopped, so applying anyway (as the old
        # code did, having never checked this at all) is exactly the bug
        # this review found.
        if region.deferred_signal is not None:
            common.safe_print(
                "zl journal repair: signal received during the in-flight wait -- nothing was stopped; "
                "aborting without repairing",
                file=sys.stderr,
            )
            region.finish(1, restored=True)
            return 1
        try:
            was_running = common.stop_for_apply(writers)
        except common.StopVerificationFailed as e:
            code, restored = common.refuse_after_stop_failure(e)
            region.finish(code, restored=restored)
            return code
        # Round-6 REVERSAL: this used to say "deliberately NOT re-checking
        # region.deferred_signal here" -- see fix.py's own identical note
        # (same reversal, same reasoning). A signal that arrived DURING/
        # after `stop_for_apply`, before recover-webui-db.py has even
        # started, now means "stop, don't repair": restore whatever this
        # wrapper stopped and abort with 128+N, instead of going on to run
        # a repair the operator already tried to cancel. This is distinct
        # from the OTHER "signal during the child's run" case just below
        # (forward_signals=False there -- once the repair has actually
        # started, RUNBOOK_DB_JOURNAL.md's "do not interrupt" applies and
        # this window no longer does).
        if region.deferred_signal is not None:
            sig = region.deferred_signal
            came_back = common.restart_after_apply(was_running)
            restored = common.report_restart_failures(came_back)
            common.safe_print(
                "zl journal repair: cancelled before the repair was applied; "
                + ("services restored" if restored else "services NOT fully restored -- see the WARNING above"),
                file=sys.stderr,
            )
            return 128 + sig if restored else 1
        # Review B1: the print below, and the whole restart-or-leave-
        # stopped decision, USED to sit after `run_protected` returned,
        # with no `try`/`finally` of its own -- a hangup or an unexpected
        # exception anywhere between `stop_for_apply` succeeding and this
        # point (an OSError from the print itself, in the pre-dup2 world;
        # or genuinely anything else) escaped with compactor/backup(/
        # webuidb-sync) stopped and NOTHING run to bring them back. `code`
        # starts at a sentinel that is deliberately NOT in
        # `_LEAVE_STOPPED_EXIT_CODES` (a small set of positive ints) so
        # that an exception here falls into the SAME "restart" branch a
        # genuine 0/2/3/5 would -- getting writers back up is the safer
        # default when what actually happened is unknown, not leaving
        # them down on a guess.
        code = -1
        restored = True
        try:
            common.safe_print(f"stopped (were running): {was_running or '(none of them were running)'}")
            code = common.run_protected(
                [str(podenv.COMPACTOR_PY), str(_SCRIPT), "--yes", *extra],
                name="journal-repair",
                env=_env_with_webui_db(),
                region=region,
                forward_signals=False,
                no_forward_message="zl: repair in progress; will not interrupt",
            )
        finally:
            if code in _LEAVE_STOPPED_EXIT_CODES:
                # Review M1: restarting compactor/backup(/webuidb-sync)
                # after recover-webui-db.py's own 4 (integrity check
                # failed), 6 (the rename+copy2 swap itself failed) or 7
                # (a sidecar still sits beside the new database)
                # contradicts RUNBOOK_DB_JOURNAL.md section 6 ("Leave the
                # services stopped... Do not start OpenWebUI... ask for
                # help first"). Leaving compactor/backup/webuidb-sync
                # stopped here is not itself dangerous (they are not what
                # the runbook is protecting), but restarting them anyway
                # would contradict the runbook's own instruction for
                # exactly this situation, and a later manual restore
                # would then lose whatever happened in between.
                if code == 4:
                    # recover-webui-db.py:207-209: on this exact failure (integrity_check
                    # did NOT say ok) the script itself already calls
                    # `supervisor("start")` before returning 4 -- openwebui IS back up,
                    # running against the UNTOUCHED original webui.db (the swap never
                    # happened). Saying "openwebui is down" here would be simply wrong;
                    # the real caution is that webui.db itself was never repaired.
                    openwebui_note = (
                        "recover-webui-db.py already restarted openwebui itself (against the ORIGINAL, "
                        "untouched webui.db -- the swap never happened, integrity_check said NOT ok) -- "
                        "OPENWEBUI IS UP, but webui.db is exactly as broken as before this ran"
                    )
                else:
                    # codes 6/7: no `supervisor("start")` call on either of these paths
                    # (recover-webui-db.py:250/256) -- openwebui really is still down.
                    openwebui_note = (
                        "recover-webui-db.py stopped openwebui itself and never restarted it -- "
                        "OPENWEBUI IS DOWN"
                    )
                start_writers = "supervisorctl start compactor backup" + (
                    " webuidb-sync" if "webuidb-sync" in writers else ""
                )
                if code == 4:
                    # openwebui is already up (against the original webui.db) -- only
                    # the writers this wrapper itself stopped need bringing back, and
                    # only once webui.db has actually been repaired by hand.
                    chat_note = "she can chat now (against the still-broken webui.db)"
                    bring_up_order = f"    once webui.db is repaired by hand: {start_writers}\n"
                else:
                    chat_note = "she cannot chat until you bring openwebui back up by hand"
                    bring_up_order = (
                        "  once resolved by hand, bring it back up in this exact order:\n"
                        "    1. supervisorctl start openwebui\n"
                        "    2. curl -s -o /dev/null -w '%{http_code} %{time_total}s\\n' -m 10 "
                        f"http://127.0.0.1:{podenv.OPENWEBUI_PORT}/api/config   "
                        "# repeat until this prints 200\n"
                        f"    3. {start_writers}\n"
                    )
                common.safe_print(
                    f"zl journal repair: recover-webui-db.py exited {code} -- per RUNBOOK_DB_JOURNAL.md "
                    "section 6, LEAVING WRITERS STOPPED and not touching webui.db further.\n"
                    f"  {openwebui_note} -- {chat_note}.\n"
                    f"  stopped and left stopped: {was_running or '(none)'}\n"
                    "  do NOT delete the journal, the forensic copies, or any .broken-* file.\n"
                    "  save the evidence under RUNBOOK_DB_JOURNAL.md section 5 and ask for help.\n"
                    f"{bring_up_order}",
                    file=sys.stderr,
                )
            else:
                came_back = common.restart_after_apply(was_running)
                restored = common.report_restart_failures(came_back)
        if code in _LEAVE_STOPPED_EXIT_CODES:
            region.finish(code, restored=False, applied=True, label="the repair")
            return code
        # Round-6 HIGH: same reasoning as fix.py -- `restored` reflects
        # whether every writer actually came back RUNNING; force the exit
        # code nonzero when one didn't, even though recover-webui-db.py's
        # own exit code was 0/2/3/5 (none of which are in
        # _LEAVE_STOPPED_EXIT_CODES).
        if not restored and code == 0:
            code = 1
        if "webuidb-sync" in writers:
            # Review MEDIUM: on a WEBUI_DB_LOCAL=true pod, /data's snapshot
            # of webui.db is only as fresh as webuidb-sync's last publish
            # -- its own loop sleeps first, then publishes (v3.1.9
            # webuidb.py's sync loop), so /data stays STALE for up to
            # WEBUI_DB_SYNC_INTERVAL_S (300s default) after this repair. A
            # backup or a pod stop in that window would capture the stale
            # (pre-repair) snapshot, not the fix just applied.
            common.safe_print(
                f"zl journal repair: WEBUI_DB_LOCAL is true -- /data's snapshot of webui.db is now "
                f"STALE for up to WEBUI_DB_SYNC_INTERVAL_S (default 300s) until webuidb-sync publishes "
                "again on its own. To publish the repaired database immediately instead of waiting: "
                f"{podenv.COMPACTOR_PY} {podenv.COMPACTOR_PKG / 'webuidb.py'} "
                "--sync-once --force (safe with services up -- --force only skips the mtime check; "
                "the same liveness/integrity guards still apply)."
            )
        region.finish(code, restored=restored, applied=True, label="the repair")
    return code
