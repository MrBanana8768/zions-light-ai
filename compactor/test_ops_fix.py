"""Unit tests for ops/zlops/commands/fix.py -- the repair-script registry
and its apply flow: dry run touches nothing, --apply backs up first, stops
exactly the services that were running, runs the real (here: fake, for
unit-test isolation) script, and restores EXACTLY what it stopped -- never
starting one that was already stopped -- even through a KeyboardInterrupt.

Plain-script style like compactor/test_setup_sshd_script.py: fakes go on
PATH in a temp dir; exit 0 pass, 1 fail.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from zlops.commands import fix  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label, extra=""):
    if not cond:
        print(f"FAIL {label}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


_FAKE_SUPERVISORCTL = """#!/usr/bin/env python3
import json, os, sys
STATE = os.environ["FAKE_SUPERVISOR_STATE"]
def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}
def save(d):
    json.dump(d, open(STATE, "w"))
cmd = sys.argv[1] if len(sys.argv) > 1 else ""
state = load()
if cmd == "status":
    for name, s in state.items():
        print(f"{name}{' ' * 20}{s}")
elif cmd == "stop":
    for name in sys.argv[2:]:
        state[name] = "STOPPED"
    save(state)
elif cmd == "start":
    for name in sys.argv[2:]:
        state[name] = "RUNNING"
    save(state)
sys.exit(0)
"""

_FAKE_SCRIPT = """#!/usr/bin/env python3
import os, sys
if os.environ.get("FAKE_SCRIPT_MARKER"):
    with open(os.environ["FAKE_SCRIPT_MARKER"], "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\\n")
sys.exit(int(os.environ.get("FAKE_SCRIPT_EXIT", "0")))
"""


class _Fixture:
    """A temp bin dir with a fake supervisorctl (state-file backed) on
    PATH, a fake target script, and a synthetic FixEntry registered under
    REGISTRY['test-entry'] for the duration of the `with` block.
    """

    def __init__(self, initial_state: dict[str, str], read_only: bool = False, default_args=None):
        self.initial_state = initial_state
        self.read_only = read_only
        self.default_args = default_args

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        td = Path(self._td.name)
        bin_dir = td / "bin"
        bin_dir.mkdir()
        supervisorctl = bin_dir / "supervisorctl"
        supervisorctl.write_text(_FAKE_SUPERVISORCTL, encoding="utf-8")
        supervisorctl.chmod(supervisorctl.stat().st_mode | stat.S_IEXEC)
        self.script = td / "fake_fix_script.py"
        self.script.write_text(_FAKE_SCRIPT, encoding="utf-8")
        self.script.chmod(self.script.stat().st_mode | stat.S_IEXEC)
        self.marker = td / "marker.log"
        self.state_file = td / "state.json"
        self.state_file.write_text(json.dumps(self.initial_state), encoding="utf-8")

        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
                "FAKE_SUPERVISOR_STATE": str(self.state_file),
                "FAKE_SCRIPT_MARKER": str(self.marker),
            },
        )
        self._env_patch.start()
        # run_protected writes a run log under podenv.ZL_HOME/runs -- give it
        # somewhere real and writable instead of the default /data/zl-ops.
        self._zl_home_patch = mock.patch.object(fix.podenv, "ZL_HOME", td / "zl-home")
        self._zl_home_patch.start()

        entry = fix.FixEntry(
            name="test-entry",
            script=self.script.name,
            interpreter=Path(sys.executable),
            read_only=self.read_only,
            dry_run_args=(),
            apply_args=("--apply",),
            services=lambda: ("compactor", "backup"),
            services_source="test fixture",
            default_args=self.default_args,
            help="test entry",
        )
        # scripts_dir() must resolve to where our fake script actually lives.
        self._scripts_dir_patch = mock.patch.object(fix.podenv, "scripts_dir", return_value=td)
        self._scripts_dir_patch.start()
        fix.REGISTRY["test-entry"] = entry
        return self

    def __exit__(self, *exc):
        self._scripts_dir_patch.stop()
        self._zl_home_patch.stop()
        self._env_patch.stop()
        fix.REGISTRY.pop("test-entry", None)
        self._td.cleanup()
        return False

    def supervisor_state(self) -> dict[str, str]:
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def marker_lines(self) -> list[str]:
        return self.marker.read_text(encoding="utf-8").splitlines() if self.marker.exists() else []


def test_dry_run_never_touches_services_or_marker():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        code = fix.run(["test-entry"])
        assert_eq(code, 0, "dry run exits 0")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "dry run never stops a service")
        assert_true(len(fx.marker_lines()) == 1, "dry run still runs the script itself (without --apply)")
        assert_true("--apply" not in fx.marker_lines()[0], "dry run's script invocation carries no --apply")


def test_apply_stops_and_restores_exactly_what_was_running():
    # `backup` starts already STOPPED (e.g. a WEBUI_DB_LOCAL=false pod where
    # it was down for some other reason) -- it must stay stopped afterward.
    with _Fixture({"compactor": "RUNNING", "backup": "STOPPED"}) as fx:
        with mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
             mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent):
            # Give backup.py a stand-in so the backup-first step succeeds.
            (fx.script.parent / "backup.py").write_text(_FAKE_SCRIPT, encoding="utf-8")
            code = fix.run(["test-entry", "--apply", "--yes"])
        assert_eq(code, 0, "apply exits 0")
        state = fx.supervisor_state()
        assert_eq(state["compactor"], "RUNNING", "apply restores the service that WAS running")
        assert_eq(state["backup"], "STOPPED", "apply NEVER starts a service that was already stopped")
        lines = fx.marker_lines()
        assert_true(any("--apply" in ln for ln in lines), "apply's script invocation carries --apply",
                    extra="\n".join(lines))


def test_apply_refuses_without_yes_on_non_tty():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            try:
                fix.run(["test-entry", "--apply"])
            except SystemExit as e:
                assert_eq(e.code, 1, "apply without --yes on a non-TTY refuses with exit 1")
            else:
                print("FAIL apply without --yes on a non-TTY did not refuse")
                sys.exit(1)
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "a refused apply never touched a service")


def test_apply_keyboard_interrupt_during_script_still_restores_services():
    # A KeyboardInterrupt arising OUTSIDE run_protected's own signal-safe
    # window (e.g. during the backup step, or between stop and the actual
    # child exec) must still be caught by fix.py's own try/finally and
    # still restore -- run_protected itself absorbs a real SIGINT/SIGTERM
    # during the child's run without ever raising (see its own dedicated
    # tests in test_ops_common.py); this test is about fix.py's OWN
    # finally, independent of run_protected's internals.
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(fix.common, "run_protected", side_effect=KeyboardInterrupt):
            try:
                fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
            except KeyboardInterrupt:
                pass
            else:
                print("FAIL KeyboardInterrupt during the script was swallowed instead of propagating")
                sys.exit(1)
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "a KeyboardInterrupt mid-script still restarts every service that was running")


def test_apply_actually_stops_services_before_running_target():
    """H5 (review): a mutation that never calls `supervisorctl stop` must
    be caught -- not just checked by the end state (which a mutated
    "stop, then immediately restart" no-op could also satisfy). The fake
    target script itself records what supervisorctl reports AT THE TIME
    IT RUNS, proving the stop happened BEFORE the target, not just
    sometime during the whole invocation.
    """
    recording_script = """#!/usr/bin/env python3
import json, os, subprocess, sys
state = json.load(open(os.environ["FAKE_SUPERVISOR_STATE"]))
with open(os.environ["FAKE_SCRIPT_MARKER"], "a") as f:
    f.write(json.dumps(state) + "\\n")
sys.exit(int(os.environ.get("FAKE_SCRIPT_EXIT", "0")))
"""
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        fx.script.write_text(recording_script, encoding="utf-8")
        with mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
             mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent):
            code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
        assert_eq(code, 0, "apply exits 0")
        lines = fx.marker_lines()
        assert_true(len(lines) == 1, "the target ran exactly once", extra="\n".join(lines))
        state_when_target_ran = json.loads(lines[0])
        assert_eq(
            state_when_target_ran, {"compactor": "STOPPED", "backup": "STOPPED"},
            "both services were STOPPED at the moment the target script actually ran "
            "(not just stopped-then-restarted somewhere in the whole invocation)",
        )
        # --- mutation: skip stop_for_apply entirely --------------------------
        # Simulates "never actually stop services" (review H5's own listed
        # surviving mutation). With the stop skipped, the target must see
        # the services still RUNNING -- this test bites exactly there.
        real_stop_for_apply = fix.common.stop_for_apply
        with mock.patch.object(fix.common, "stop_for_apply", return_value=[]), \
             mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
             mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent):
            fx.marker.unlink()
            fx.state_file.write_text(json.dumps({"compactor": "RUNNING", "backup": "RUNNING"}), encoding="utf-8")
            fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
        mutated_lines = fx.marker_lines()
        assert_true(len(mutated_lines) == 1, "mutated run still ran the target once")
        mutated_state = json.loads(mutated_lines[0])
        assert_true(
            mutated_state != {"compactor": "STOPPED", "backup": "STOPPED"},
            "confirms the mutation actually changes what the target observed (sanity check on the test itself)",
        )
        assert_eq(fix.common.stop_for_apply, real_stop_for_apply, "sanity: the real stop_for_apply is restored")


def test_apply_backup_failure_has_its_own_distinct_exit_code():
    """H5 (review): 'ignoring backup failure' survived before because the
    fake backup and the fake target shared FAKE_SCRIPT_EXIT=1 -- a backup
    failure and a target failure were indistinguishable by exit code. Now
    they must not be: backup failure is common.EXIT_BACKUP_FAILED (5),
    never conflated with a target's own exit code (which could itself be 5
    coincidentally -- so also check no supervisorctl stop happened, since a
    real target never even runs when the backup fails).
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
             mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent), \
             mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "1"}):
            (fx.script.parent / "backup.py").write_text(_FAKE_SCRIPT, encoding="utf-8")
            code = fix.run(["test-entry", "--apply", "--yes"])
        assert_eq(code, fix.common.EXIT_BACKUP_FAILED, "a failing backup exits the DISTINCT backup-failure code")
        assert_true(code != 1, "sanity: the distinct code is not the generic refusal code")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "a refused-on-backup-failure apply never stopped a service")
        # marker also records the backup.py invocation itself ("--once");
        # the TARGET script's own invocation always carries --apply.
        assert_true(not any("--apply" in ln for ln in fx.marker_lines()),
                     "the target script never ran at all when the backup failed", extra="\n".join(fx.marker_lines()))


def test_apply_refuses_if_backup_fails_unless_skip_backup():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
             mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent), \
             mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "1"}):
            (fx.script.parent / "backup.py").write_text(_FAKE_SCRIPT, encoding="utf-8")
            code = fix.run(["test-entry", "--apply", "--yes"])
        assert_eq(code, fix.common.EXIT_BACKUP_FAILED, "a failing backup refuses the apply")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "a refused-on-backup-failure apply never stopped a service")


def test_read_only_entry_ignores_apply_and_never_touches_services():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}, read_only=True) as fx:
        code = fix.run(["test-entry", "--apply", "--yes"])
        assert_eq(code, 0, "a read-only entry with --apply still just runs (and succeeds)")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "a read-only entry never stops a service even with --apply")


def test_passthrough_write_flag_refused_without_zl_apply():
    """Review B2: `-- --apply` (the target's OWN apply-shaped flag, in this
    fixture's case a write_flags entry) must never sneak a write past our
    own gate. Uses a synthetic write flag on the fixture's own entry.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        fx_entry = fix.REGISTRY["test-entry"]
        fix.REGISTRY["test-entry"] = dataclasses.replace(fx_entry, write_flags=("--force",))
        try:
            code = fix.run(["test-entry", "--", "--force"])
            assert_eq(code, 2, "passthrough with a declared write flag, no zl --apply, refuses with exit 2")
            assert_eq(fx.marker_lines(), [], "the target never ran at all")
            assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                      "a refused passthrough never touched a service")

            code2 = fix.run(["test-entry", "--", "--apply"])
            assert_eq(code2, 2, "'--apply' itself after `--` is always refused, even standing alone")

            code3 = fix.run(["test-entry", "--apply", "--yes", "--skip-backup", "--", "--force"])
            assert_true(code3 == 0, "with zl's own --apply given too, the same passthrough now goes through")
            assert_true(any("--force" in ln for ln in fx.marker_lines()), "and the script actually saw --force")
        finally:
            fix.REGISTRY["test-entry"] = fx_entry


def test_flag_only_passthrough_keeps_the_default_positional_arg():
    """review LOW: `-- --restore STAMP` used to REPLACE default_args
    entirely, dropping the required db path default_args exists to supply
    -- the script then ran with no db argument at all. A flag-only
    passthrough must now be PREPENDED with the default instead.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}, default_args=lambda: ("DEFAULT_DB_PATH",)) as fx:
        code = fix.run(["test-entry", "--", "--restore", "STAMP"])
        assert_eq(code, 0, "dry run with flag-only passthrough exits 0")
        lines = fx.marker_lines()
        assert_true(len(lines) == 1, "the script ran once", extra="\n".join(lines))
        assert_eq(lines[0].split(), ["DEFAULT_DB_PATH", "--restore", "STAMP"],
                   "the default db path is PREPENDED, never dropped")

        # An operator who supplies their OWN positional (not flag-only)
        # still gets exactly what they typed, unprefixed.
        fx.marker.unlink()
        fix.run(["test-entry", "--", "OTHER_DB_PATH", "--restore", "STAMP2"])
        lines2 = fx.marker_lines()
        assert_eq(lines2[0].split(), ["OTHER_DB_PATH", "--restore", "STAMP2"],
                   "an explicit positional passthrough is used as-is, not prefixed with the default")


def test_real_sigterm_during_stop_phase_aborts_without_applying_then_exits_128_plus_signal():
    """R2-B1: the protected region starts BEFORE the first status read/stop.
    A real SIGTERM delivered while `stop_for_apply` is still running (no
    child registered yet) must be DEFERRED, not dropped and not left to
    kill the wrapper outright.

    Round-6 REVERSAL: this window (a signal that arrived DURING/after
    `stop_for_apply`, strictly before the child starts) used to let the
    child run to completion anyway and only reflect the signal in the
    final exit code -- round-6 switched this to an ABORT, the same as a
    signal during the in-flight wait: nothing but the stop itself has
    happened, so restore exactly what was stopped and exit 128+SIGTERM
    WITHOUT ever running the target script.
    """
    import signal
    import threading
    import time

    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        real_stop = fix.common.stop_for_apply
        entered_stop = threading.Event()

        def slow_stop(names, **kw):
            entered_stop.set()  # signal we're inside the protected window before sleeping
            time.sleep(1.2)
            return real_stop(names, **kw)

        def _send_sigterm_once_inside_stop():
            # Wait for confirmation we're actually inside `stop_for_apply`
            # (and therefore inside the protected region) before sending the
            # real signal -- a FIXED delay here would race under system
            # load (many test files run back to back), occasionally firing
            # AFTER the protected window already closed and hitting the
            # process's default (kill) disposition instead.
            entered_stop.wait(timeout=5.0)
            time.sleep(0.3)  # small margin, still well inside the 1.2s sleep
            os.kill(os.getpid(), signal.SIGTERM)

        with mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
             mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent), \
             mock.patch.object(fix.common, "stop_for_apply", side_effect=slow_stop):
            (fx.script.parent / "backup.py").write_text(_FAKE_SCRIPT, encoding="utf-8")
            timer = threading.Thread(target=_send_sigterm_once_inside_stop)
            timer.start()
            try:
                code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
            finally:
                timer.join()
            assert_eq(code, 128 + signal.SIGTERM,
                      "a real SIGTERM during the stop phase aborts and exits 128+SIGTERM once restored")
        assert_eq(fx.marker_lines(), [], "the target never ran -- cancelled before the fix was applied")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "services are fully restored even though a signal arrived mid-stop")


def test_ctrl_c_during_restart_preserves_nonzero_child_exit_code():
    """H2: a real Ctrl-C landing during `restart_after_apply` (no child is
    registered at that point -- `ProtectedRegion` defers it rather than
    forwarding it anywhere) must not be allowed to turn a real, nonzero
    exit from the TARGET SCRIPT (here: 4, "re-run") into a bare 130.
    `region.finish(code, ...)` keeps the child's own code.
    """
    import signal
    import threading
    import time

    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "4"}):
            real_restart = fix.common.restart_after_apply
            entered_restart = threading.Event()

            def slow_restart(names, **kw):
                entered_restart.set()
                time.sleep(1.2)
                return real_restart(names, **kw)

            def _send_sigint_once_inside_restart():
                entered_restart.wait(timeout=5.0)
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGINT)

            with mock.patch.object(fix.common, "restart_after_apply", side_effect=slow_restart):
                timer = threading.Thread(target=_send_sigint_once_inside_restart)
                timer.start()
                try:
                    fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
                except SystemExit as e:
                    assert_eq(e.code, 4, "the target script's own exit 4 is kept over the deferred Ctrl-C")
                else:
                    print("FAIL fix.run did not exit")
                    sys.exit(1)
                finally:
                    timer.join()
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "services were still restarted despite the deferred signal")


def test_apply_exits_nonzero_when_a_service_does_not_come_back_even_if_script_exited_0():
    """Round-7 mutation kill (mut/H2_fix_no_force_nonzero): `fix.run`'s
    own `if not restored and code == 0: code = 1` (review H2) must not be
    disabled -- a wrapped script that exits 0 but leaves a service down
    afterward (`restart_after_apply`/`report_restart_failures` says so)
    must still make `zl fix --apply` exit nonzero, or an operator relying
    on the exit code alone would believe everything came back.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "0"}):
            with mock.patch.object(fix.common, "restart_after_apply",
                                    return_value={"compactor": True, "backup": False}):
                code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
        assert_eq(code, 1, "the wrapped script's own exit 0 does not win -- a down service forces exit 1")


def test_apply_exits_0_when_every_service_actually_came_back():
    """Companion to the test above: with every service reported back
    RUNNING and the wrapped script exiting 0, `fix.run` really does exit
    0 -- proving the force-nonzero path above is conditional, not always
    on (a mutation that always forces `code = 1` would pass the test
    above but fail this one).
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "0"}):
            with mock.patch.object(fix.common, "restart_after_apply",
                                    return_value={"compactor": True, "backup": True}):
                code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
        assert_eq(code, 0, "every service back RUNNING and script exit 0 -- apply exits 0")


def test_deferred_signal_after_a_completed_run_reports_was_applied():
    """Round-7 mutation kill (mut/A1_fix_applied_false): `fix.run`'s final
    `region.finish(code, restored=restored, applied=True)` call must
    actually pass `applied=True` -- otherwise a signal deferred during the
    restart phase, after the wrapped script already ran to completion
    (exit 0), prints no "WAS APPLIED ... do not re-run it" note at all,
    and an operator going by the exit code (128+signal) alone could
    reasonably re-run something that had, in fact, already succeeded.
    """
    import io
    import signal
    import threading
    import time

    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "0"}):
            real_restart = fix.common.restart_after_apply
            entered_restart = threading.Event()

            def slow_restart(names, **kw):
                entered_restart.set()
                time.sleep(1.2)
                return real_restart(names, **kw)

            def _send_sigint_once_inside_restart():
                entered_restart.wait(timeout=5.0)
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGINT)

            printed = io.StringIO()
            with mock.patch.object(sys, "stderr", printed):
                with mock.patch.object(fix.common, "restart_after_apply", side_effect=slow_restart):
                    timer = threading.Thread(target=_send_sigint_once_inside_restart)
                    timer.start()
                    try:
                        fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
                    except SystemExit as e:
                        assert_eq(e.code, 128 + signal.SIGINT,
                                  "with child_code 0, the deferred signal becomes the exit code")
                    else:
                        print("FAIL fix.run did not exit")
                        sys.exit(1)
                    finally:
                        timer.join()
        out = printed.getvalue()
        assert_true("WAS APPLIED" in out and "do not re-run" in out,
                     "a signal deferred after a completed (exit 0) run reports WAS APPLIED / do not re-run",
                     extra=out)


def test_real_sighup_during_apply_never_kills_the_process_and_restores_services():
    """M1 (round-2 reviewer's mutation list): if `ProtectedRegion` ever
    stops actually handling SIGHUP, sending a real SIGHUP to this process
    while `fix.run` is mid-apply would kill the WHOLE TEST PROCESS
    outright (SIGHUP's default action) -- there is no softer failure mode
    to assert on; a mutated build either survives this test or doesn't
    get the chance to report anything at all. Uses a script that sleeps
    briefly so there is a reliable window to signal during, covering both
    the stop phase (before the fixture's `--skip-backup` fast path stops
    it) and the child's own run.

    Review B1 (round 3): the region's SIGHUP handler now dup2's
    /dev/null onto fds 1 and 2 (see `common.ProtectedRegion._on_hangup`)
    -- deliberately process-wide and irreversible on the real pod (there
    is no terminal to get back once sshd has actually hung up). Here,
    where the "hangup" is really just this TEST sending itself a signal,
    that dup2 would otherwise silently swallow every print from every
    test that runs afterward IN THIS SAME PROCESS for the rest of the
    file -- not a bug in the fix, but a test-harness hazard, so this test
    saves and restores its own fds 1/2 around the real SIGHUPs.
    """
    import signal
    import time

    sleepy_script = """#!/usr/bin/env python3
import sys, time
time.sleep(0.5)
sys.exit(0)
"""
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
            fx.script.write_text(sleepy_script, encoding="utf-8")
            with mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
                 mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent):
                (fx.script.parent / "backup.py").write_text(_FAKE_SCRIPT, encoding="utf-8")

                def _send_sighup_shortly():
                    time.sleep(0.1)
                    os.kill(os.getpid(), signal.SIGHUP)
                    time.sleep(0.2)
                    os.kill(os.getpid(), signal.SIGHUP)  # a second one, during the child's own run

                import threading
                t = threading.Thread(target=_send_sighup_shortly)
                t.start()
                code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
                t.join()
            # Reaching this line at all is most of the proof (a real SIGHUP
            # that was NOT ignored/handled would have killed this process
            # before it got here). The rest confirms the run still
            # completed cleanly.
            assert_eq(code, 0, "the apply still completed normally despite two real SIGHUPs")
            assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                      "services are restored -- SIGHUP never interrupted anything")
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def test_real_sigterm_while_child_runs_is_forwarded_not_deferred():
    """The OTHER half of R2-B1: once the target script is actually
    running, a SIGTERM must still be FORWARDED to it (the pre-existing
    round-1 behaviour, round-2-confirmed-fixed) rather than deferred --
    `deferred_signal` must stay None, and the child's own signal death
    is what determines the exit code (handled by cli.main's negative-code
    conversion, tested separately in test_ops_cli.py).
    """
    import signal
    import time

    with tempfile.TemporaryDirectory() as marker_td:
        started_marker = Path(marker_td) / "child.started"
        sleepy_script = f"""#!/usr/bin/env python3
import signal, sys, time
from pathlib import Path
def _quiet(signum, frame):
    sys.exit(77)  # proves the child got SIGINT, not left to the default handler
signal.signal(signal.SIGINT, _quiet)
Path({str(started_marker)!r}).write_text("1")
time.sleep(5)
sys.exit(0)
"""
        with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
            fx.script.write_text(sleepy_script, encoding="utf-8")
            with mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
                 mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent):
                (fx.script.parent / "backup.py").write_text(_FAKE_SCRIPT, encoding="utf-8")

                def _send_sigterm_once_child_started():
                    # Poll for the CHILD's own marker (written the instant it
                    # installs its SIGINT handler) instead of a fixed delay --
                    # a fixed delay would race under system load (many test
                    # files running back to back), occasionally firing before
                    # run_protected has even registered the child with the
                    # region, or long after it, both of which this test does
                    # not mean to exercise.
                    deadline = time.time() + 10.0
                    while not started_marker.exists() and time.time() < deadline:
                        time.sleep(0.02)
                    time.sleep(0.1)  # small margin once we know it's running
                    os.kill(os.getpid(), signal.SIGTERM)

                import threading
                t = threading.Thread(target=_send_sigterm_once_child_started)
                t.start()
                code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
                t.join()
            assert_eq(code, 77, "the child exited on its own SIGINT handler (proof SIGINT was actually forwarded)")
            assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                      "services restored after a forwarded signal killed the child")


def test_stop_verification_failed_refuses_before_anything_touched():
    """HIGH: an unreadable pre-stop `supervisorctl status` must refuse
    outright (StopVerificationFailed) -- never treated as 'nothing is
    running'. End-to-end through fix.run: the target must never execute
    and supervisorctl must never even be asked to stop anything.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(fix.common, "supervisorctl_status", return_value={}):
            code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
        assert_eq(code, 1, "an unreadable pre-stop status refuses the whole apply with exit 1")
        assert_eq(fx.marker_lines(), [], "the target never ran")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "nothing was touched -- the fake supervisorctl's OWN state file proves no stop/start call landed")


def test_inflight_guard_refuses_before_any_stop():
    """Round-4 addendum: `zl fix <name> --apply` must check
    `requests_in_flight` INSIDE the protected region, before the first
    stop -- a refusal here must leave every service exactly as it was and
    never touch the fake supervisorctl's own state at all.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(fix.common, "wait_for_no_inflight_requests",
                                return_value=(False, "still 2 after 300s")):
            code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
        assert_eq(code, 1, "a refused in-flight guard exits 1")
        assert_eq(fx.marker_lines(), [], "the target never ran")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "nothing was stopped -- the guard runs before the first stop")


def test_signal_during_inflight_wait_aborts_without_stopping_or_applying():
    """Round-4 blocker #3 (review4/p1/probe_ctrlc_wait.py): a signal
    deferred WHILE `wait_for_no_inflight_requests` is still running (the
    guard itself later returns proceed=True on its own next reading) must
    abort -- nothing has been stopped yet, so `zl fix --apply` must not
    go on to stop anything and run the target anyway.
    """
    import signal

    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        def fake_guard(*, force, health_url=None):
            os.kill(os.getpid(), signal.SIGTERM)
            return True, "requests_in_flight settled to 0 -- proceeding"

        with mock.patch.object(fix.common, "wait_for_no_inflight_requests", side_effect=fake_guard):
            try:
                code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
            except SystemExit as e:
                code = e.code
        assert_eq(code, 1, "the signal during the wait aborts with exit 1")
        assert_eq(fx.marker_lines(), [], "the target never ran")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "nothing was stopped -- the signal landed before the first stop")


def test_mutation_not_checking_deferred_signal_after_the_wait_is_caught():
    """Prove the test above bites for real: run an ACTUAL mutated copy of
    fix.py's OLD apply flow (identical to the real one, minus the
    `region.deferred_signal` check right after the in-flight wait) against
    the same fixture, fake supervisorctl, and signal -- not a hardcoded
    boolean (round-6 LOW: the previous version of this test could never
    fail no matter what the mutated flow did). Confirms it actually STOPS
    services and RUNS THE TARGET despite the deferred signal, then
    confirms the real (restored) `fix.run` aborts on the identical signal
    instead, touching nothing.
    """
    import signal

    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        entry = fix.REGISTRY["test-entry"]

        def mutated_apply_flow() -> int:
            """A faithful copy of fix.py's apply flow AS IT WAS BEFORE
            round-4 blocker #3: it waits for the in-flight guard, but
            never re-checks `region.deferred_signal` afterward -- so a
            signal deferred during the wait is silently lost, and the
            flow proceeds to stop services and run the target anyway.
            """
            services = list(entry.services())
            with fix.common.ProtectedRegion() as region:
                proceed, _note = fix.common.wait_for_no_inflight_requests(force=False)
                if not proceed:
                    region.finish(1, restored=True)
                    return 1
                # MUTATION: the real code's
                # `if region.deferred_signal is not None: ... abort` is
                # simply missing here.
                was_running = fix.common.stop_for_apply(services)
                code = 1
                try:
                    code = fix.common.run_protected(
                        [str(entry.interpreter), str(fix.podenv.scripts_dir() / entry.script), *entry.apply_args],
                        name="fix-test-entry-mutated", env=fix.common.merged_environ(), region=region,
                    )
                finally:
                    fix.common.restart_after_apply(was_running)
                region.finish(code, restored=True)
            return code

        def fake_guard(*, force, health_url=None):
            os.kill(os.getpid(), signal.SIGTERM)
            return True, "requests_in_flight settled to 0 -- proceeding"

        with mock.patch.object(fix.common, "wait_for_no_inflight_requests", side_effect=fake_guard):
            try:
                mutated_apply_flow()
            except SystemExit:
                pass  # finish() still reflects the deferred signal in the exit code either way
        assert_true(any("--apply" in ln for ln in fx.marker_lines()),
                    "mutated (never-checked) flow actually ran the target despite the deferred signal -- "
                    "confirms the bite", extra="\n".join(fx.marker_lines()))
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "sanity: the mutated flow still restores services (the bite is that it applied at all, "
                  "not that it left anything down)")

        fx.marker.unlink()
        fx.state_file.write_text(json.dumps({"compactor": "RUNNING", "backup": "RUNNING"}), encoding="utf-8")
        with mock.patch.object(fix.common, "wait_for_no_inflight_requests", side_effect=fake_guard):
            try:
                code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
            except SystemExit as e:
                code = e.code
        assert_eq(code, 1, "the real (restored) code aborts instead of applying")
        assert_eq(fx.marker_lines(), [], "the real (restored) code never ran the target")


def test_force_busy_flag_reaches_the_inflight_guard():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}):
        seen = {}

        def fake_guard(*, force, health_url=None):
            seen["force"] = force
            return True, "ok"

        with mock.patch.object(fix.common, "wait_for_no_inflight_requests", side_effect=fake_guard):
            code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup", "--force-busy"])
        assert_eq(code, 0, "apply with --force-busy still succeeds")
        assert_eq(seen.get("force"), True, "--force-busy reached the in-flight guard as force=True")


class _FakeHealthServer:
    """A tiny real HTTP server standing in for the compactor's
    `/health/full`, so `wait_for_no_inflight_requests` can be exercised
    for real (not mocked) across the BLOCKER tests below -- the state
    dict's `n` is read live on every request, so a test thread can flip
    it mid-backup to simulate her next request actually arriving.
    """

    def __init__(self, initial_n: int = 0):
        import http.server
        import threading

        self.state = {"n": initial_n}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(handler_self):  # noqa: N805
                import json as _json
                body = _json.dumps({"request_tracing": {"requests_in_flight": self.state["n"]}}).encode()
                handler_self.send_response(200)
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def log_message(handler_self, *a):  # noqa: N805
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


def test_backup_window_blocker_refuses_when_her_request_arrives_during_the_backup():
    """BLOCKER (round-6, review5/p1/probe_backup_window.py "request" mode):
    `requests_in_flight` reads 0 right before the backup starts (the FIRST
    wait settles), but her next request arrives WHILE `backup.py --once`
    is still running. The old code never re-checked after the backup, so
    it went on to stop services and run the target anyway. The fixed code
    must catch this in the post-backup re-check and refuse -- NOTHING
    stopped, the target NEVER runs.
    """
    import threading
    import time

    server = _FakeHealthServer(initial_n=0)
    try:
        with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx, \
             mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
             mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent), \
             mock.patch.object(fix.common.podenv, "COMPACTOR_URL", server.url), \
             mock.patch.object(fix.common, "_INFLIGHT_QUIET_S", 0.2):
            slow_backup = """#!/usr/bin/env python3
import time
time.sleep(1.0)
"""
            (fx.script.parent / "backup.py").write_text(slow_backup, encoding="utf-8")

            def _flip_busy_during_backup():
                time.sleep(0.3)
                server.state["n"] = 1  # her next message arrives mid-backup

            t = threading.Thread(target=_flip_busy_during_backup)
            t.start()
            with mock.patch.object(fix.common, "_INFLIGHT_TIMEOUT_S", 2.0):
                code = fix.run(["test-entry", "--apply", "--yes"])
            t.join()
            assert_eq(code, 1, "the post-backup re-check refuses")
            assert_eq(fx.marker_lines(), [], "the target never ran")
            assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                      "nothing was stopped -- the re-check runs before the first stop")
    finally:
        server.close()


def test_backup_window_blocker_aborts_on_sigterm_during_the_backup():
    """BLOCKER (round-6, review5/p1/probe_backup_window.py "sigterm" mode):
    a SIGTERM to the `zl` PID ITSELF (not its process group -- exactly
    what an operator's terminal delivers) arriving WHILE `backup.py
    --once` runs must be caught by the post-backup re-check's own
    `region.deferred_signal` test: nothing stopped, the target never
    runs. Exit code is the plain refusal code (1) -- same convention as
    every other "signal during a pre-stop wait" abort in this file
    (test_signal_during_inflight_wait_aborts_without_stopping_or_applying);
    128+SIGTERM is reserved for a signal AFTER `stop_for_apply` has
    already stopped something real (see the REVERSAL tests), which this
    is not.
    """
    import signal
    import threading
    import time

    server = _FakeHealthServer(initial_n=0)
    try:
        with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx, \
             mock.patch.object(fix.podenv, "COMPACTOR_PY", Path(sys.executable)), \
             mock.patch.object(fix.podenv, "COMPACTOR_PKG", fx.script.parent), \
             mock.patch.object(fix.common.podenv, "COMPACTOR_URL", server.url):
            slow_backup = """#!/usr/bin/env python3
import time
time.sleep(1.0)
"""
            (fx.script.parent / "backup.py").write_text(slow_backup, encoding="utf-8")

            def _sigterm_during_backup():
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGTERM)

            t = threading.Thread(target=_sigterm_during_backup)
            t.start()
            try:
                code = fix.run(["test-entry", "--apply", "--yes"])
            except SystemExit as e:
                # region.finish() raises SystemExit(1) here because a signal
                # WAS deferred (see its own docstring) -- same mechanism as
                # every other pre-stop abort in this file.
                code = e.code
            t.join()
            assert_eq(code, 1, "a SIGTERM during the backup aborts with the plain refusal exit code")
            assert_eq(fx.marker_lines(), [], "the target never ran")
            assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                      "nothing was stopped")
    finally:
        server.close()


def test_inflight_guard_called_exactly_once_with_skip_backup():
    """N18 (round-6 surviving mutant): the in-flight guard call must live
    ONLY inside the protected region, immediately before the first stop --
    a mutant that additionally calls it a second time BEFORE `with
    common.ProtectedRegion()` (unprotected: no signal handling installed
    yet at that point) doubles the call count without failing any
    existing assertion, since a fast, always-proceeding guard makes the
    duplicate harmless in every OTHER test. Counting calls directly is a
    fast, deterministic kill -- no signal timing or 600s hang needed.
    `--skip-backup` isolates this to exactly the ONE call the apply flow
    makes when there is no post-backup re-check (see the BLOCKER fix,
    which adds a second, equally protected call only when a backup is
    actually taken).
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(fix.common, "wait_for_no_inflight_requests",
                                return_value=(True, "ok")) as guard:
            code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup"])
        assert_eq(code, 0, "apply succeeds")
        assert_eq(guard.call_count, 1,
                   "the in-flight guard is called exactly once (inside the protected region only)")
        assert_true(any("--apply" in ln for ln in fx.marker_lines()), "and the target actually ran")


def test_mutation_forgetting_to_check_force_busy_flag_is_caught():
    """Round-4 LOW: the old version of this test only compared string
    literals against each other -- it could never fail no matter what
    fix.py actually did. Prove the test above bites for real: drive the
    real `wait_for_no_inflight_requests` with force=False (the OLD
    behaviour -- --force-busy silently ignored) against a busy compactor
    and confirm it refuses, then confirm the real `fix.run` (which
    threads --force-busy through) succeeds on the identical payload.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(fix.common, "http_get_json",
                                return_value=(200, {"request_tracing": {"requests_in_flight": 2}}, None)), \
             mock.patch.object(fix.common, "_INFLIGHT_TIMEOUT_S", 0.0):
            proceed_mutated, _note = fix.common.wait_for_no_inflight_requests(force=False)
        assert_true(not proceed_mutated,
                    "mutated (force=False always) refuses on a busy compactor -- confirms the bite")

        with mock.patch.object(fix.common, "http_get_json",
                                return_value=(200, {"request_tracing": {"requests_in_flight": 2}}, None)):
            code = fix.run(["test-entry", "--apply", "--yes", "--skip-backup", "--force-busy"])
        assert_eq(code, 0, "the real (restored) fix.run threads --force-busy through and succeeds")
        assert_true(any("--apply" in ln for ln in fx.marker_lines()), "the target actually ran")


def test_unknown_entry_lists_registry_and_exits_2():
    with _Fixture({}):
        code = fix.run(["not-a-real-entry"])
        assert_eq(code, 2, "an unknown fix entry exits 2")


def test_real_registry_entries_are_well_formed():
    """Not a fixture test -- sanity-checks the REAL registry (the 9 entries
    the brief lists), so a typo in a script name or a missing interpreter
    is caught here rather than only on the pod.
    """
    expected = {
        "repair-chat-tree", "fix-stale-unfinished", "fix-encoded-messages",
        "clean-decoration", "import-history", "backfill-records",
        "merge-conversations", "compare-live-to-backup", "switch-webui-db-to-local",
    }
    assert_eq(set(fix.REGISTRY) & expected, expected, "every entry the brief lists is registered")
    for name, entry in fix.REGISTRY.items():
        if name not in expected:
            continue
        script_path = fix.podenv.scripts_dir() / entry.script
        assert_true(script_path.is_file(), f"{name}: {script_path} exists", extra=str(script_path))
        assert_true(isinstance(entry.interpreter, Path), f"{name}: interpreter is a Path")
        if entry.read_only:
            assert_eq(entry.apply_args, (), f"{name}: read-only entries have no apply_args")
        else:
            assert_eq(entry.apply_args, ("--apply",), f"{name}: non-read-only entries apply via --apply")


def _real_argparse_long_flags(script_path: Path) -> set[str]:
    """AST-parse `ap.add_argument("--foo", ...)` calls (and the
    `add_argument("--foo", "--bar", ...)` multi-alias form) out of a real
    scripts/*.py file, without importing or executing it. Only long
    (`--foo`) options are returned -- this bundle's own passthrough
    normalization (`common.normalize_passthrough`) only ever deals with
    long options anyway.
    """
    import ast

    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flags.add(arg.value)
    return flags


def test_registry_all_flags_match_each_wrapped_scripts_real_argparse():
    """LOW (round 3): each entry's `all_flags` is hand-copied from its
    script's own argparse -- exactly the kind of duplication this
    codebase's own memory.md calls its most expensive recurring defect
    when two copies drift. This AST-parses the REAL script and compares,
    catching drift automatically instead of relying on a human re-reading
    both sides. (This is what caught import-history's missing
    `--compactor-pkg` this round.)
    """
    for name, entry in sorted(fix.REGISTRY.items()):
        if not entry.all_flags:
            continue  # no registry all_flags declared for this entry -- nothing to compare
        script_path = fix.podenv.scripts_dir() / entry.script
        if not script_path.is_file():
            continue  # covered separately by test_real_registry_entries_are_well_formed
        real_flags = _real_argparse_long_flags(script_path)
        registry_flags = set(entry.all_flags)
        missing = real_flags - registry_flags
        extra = registry_flags - real_flags
        assert_true(not missing, f"{name}: all_flags is missing real argparse flags {sorted(missing)}",
                    extra=f"script={script_path}")
        assert_true(not extra, f"{name}: all_flags claims flags the script's argparse does not declare "
                    f"{sorted(extra)}", extra=f"script={script_path}")
        for wf in entry.write_flags:
            assert_true(wf == "--apply" or wf in real_flags,
                        f"{name}: write_flags entry {wf!r} is not a real flag of the wrapped script either")


def test_mutation_stale_all_flags_entry_is_caught():
    """Prove the test above bites: temporarily point one entry's
    `all_flags` at a stale (missing a real flag) tuple, watch it fail,
    then restore.
    """
    import dataclasses as _dc

    original = fix.REGISTRY["import-history"]
    try:
        fix.REGISTRY["import-history"] = _dc.replace(
            original,
            all_flags=tuple(f for f in original.all_flags if f != "--compactor-pkg"),
        )
        script_path = fix.podenv.scripts_dir() / fix.REGISTRY["import-history"].script
        real_flags = _real_argparse_long_flags(script_path)
        missing = real_flags - set(fix.REGISTRY["import-history"].all_flags)
        assert_true("--compactor-pkg" in missing,
                    "mutated (stale) all_flags is missing --compactor-pkg -- confirms the bite")
    finally:
        fix.REGISTRY["import-history"] = original
    script_path = fix.podenv.scripts_dir() / fix.REGISTRY["import-history"].script
    real_flags = _real_argparse_long_flags(script_path)
    assert_true(not (real_flags - set(fix.REGISTRY["import-history"].all_flags)),
                "restored all_flags matches the real argparse again")


if __name__ == "__main__":
    test_dry_run_never_touches_services_or_marker()
    test_apply_stops_and_restores_exactly_what_was_running()
    test_apply_refuses_without_yes_on_non_tty()
    test_apply_keyboard_interrupt_during_script_still_restores_services()
    test_apply_actually_stops_services_before_running_target()
    test_apply_backup_failure_has_its_own_distinct_exit_code()
    test_apply_refuses_if_backup_fails_unless_skip_backup()
    test_passthrough_write_flag_refused_without_zl_apply()
    test_flag_only_passthrough_keeps_the_default_positional_arg()
    test_read_only_entry_ignores_apply_and_never_touches_services()
    test_real_sigterm_during_stop_phase_aborts_without_applying_then_exits_128_plus_signal()
    test_ctrl_c_during_restart_preserves_nonzero_child_exit_code()
    test_apply_exits_nonzero_when_a_service_does_not_come_back_even_if_script_exited_0()
    test_apply_exits_0_when_every_service_actually_came_back()
    test_deferred_signal_after_a_completed_run_reports_was_applied()
    test_real_sighup_during_apply_never_kills_the_process_and_restores_services()
    test_real_sigterm_while_child_runs_is_forwarded_not_deferred()
    test_stop_verification_failed_refuses_before_anything_touched()
    test_inflight_guard_refuses_before_any_stop()
    test_signal_during_inflight_wait_aborts_without_stopping_or_applying()
    test_mutation_not_checking_deferred_signal_after_the_wait_is_caught()
    test_backup_window_blocker_refuses_when_her_request_arrives_during_the_backup()
    test_backup_window_blocker_aborts_on_sigterm_during_the_backup()
    test_inflight_guard_called_exactly_once_with_skip_backup()
    test_force_busy_flag_reaches_the_inflight_guard()
    test_mutation_forgetting_to_check_force_busy_flag_is_caught()
    test_unknown_entry_lists_registry_and_exits_2()
    test_real_registry_entries_are_well_formed()
    test_registry_all_flags_match_each_wrapped_scripts_real_argparse()
    test_mutation_stale_all_flags_entry_is_caught()

    print("\nAll ops/commands/fix.py tests passed.")
