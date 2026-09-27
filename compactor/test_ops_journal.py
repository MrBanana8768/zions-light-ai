"""Unit tests for ops/zlops/commands/journal.py -- the recover-webui-db.py
wrapper. Round-2-review-specific behaviour this file exists to prove:

- HIGH: `zl journal repair` never forwards SIGINT/SIGTERM to
  recover-webui-db.py while it runs (RUNBOOK_DB_JOURNAL.md says not to
  interrupt its rename+copy2 swap, and the script's own `except Exception`
  does not catch KeyboardInterrupt).
- M1 (round 3; MEDIUM in round 2): on recover-webui-db.py's own exit 4, 6
  or 7, the writers it stopped (compactor/backup[/webuidb-sync]) are left
  stopped, per RUNBOOK_DB_JOURNAL.md section 6 -- restarting them would
  contradict the runbook's own "leave the services stopped" instruction.
  Exit 1 (an earlier pre-flight refusal -- NOTHING was touched, openwebui
  was never stopped) is NOT in that set: it restarts normally, the same
  as exit 3/5 (the script restarts openwebui itself before returning
  either of those).
- R2-B2: `--check=1` / `--che` after `--` on `zl journal repair` are
  recognized as `--check` (the exact same normalization every other
  wrapped-script command in this bundle now applies).

Plain-script style like compactor/test_setup_sshd_script.py: fakes go on
PATH in a temp dir; exit 0 pass, 1 fail.
"""

from __future__ import annotations

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

from zlops.commands import journal  # noqa: E402


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

_FAKE_RECOVER_SCRIPT = """#!/usr/bin/env python3
import os, sys
if os.environ.get("FAKE_SCRIPT_MARKER"):
    with open(os.environ["FAKE_SCRIPT_MARKER"], "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\\n")
sys.exit(int(os.environ.get("FAKE_SCRIPT_EXIT", "0")))
"""


class _Fixture:
    def __init__(self, initial_state: dict[str, str]):
        self.initial_state = initial_state

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        td = Path(self._td.name)
        bin_dir = td / "bin"
        bin_dir.mkdir()
        supervisorctl = bin_dir / "supervisorctl"
        supervisorctl.write_text(_FAKE_SUPERVISORCTL, encoding="utf-8")
        supervisorctl.chmod(supervisorctl.stat().st_mode | stat.S_IEXEC)
        self.script = td / "recover-webui-db.py"
        self.script.write_text(_FAKE_RECOVER_SCRIPT, encoding="utf-8")
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
        self._zl_home_patch = mock.patch.object(journal.podenv, "ZL_HOME", td / "zl-home")
        self._zl_home_patch.start()
        self._script_patch = mock.patch.object(journal, "_SCRIPT", self.script)
        self._script_patch.start()
        self._py_patch = mock.patch.object(journal.podenv, "COMPACTOR_PY", Path(sys.executable))
        self._py_patch.start()
        self._local_patch = mock.patch.object(journal.podenv, "detect_webui_db_local", return_value=(False, "test"))
        self._local_patch.start()
        self._live_db_patch = mock.patch.object(journal.podenv, "live_webui_db", return_value=td / "webui.db")
        self._live_db_patch.start()
        return self

    def __exit__(self, *exc):
        self._live_db_patch.stop()
        self._local_patch.stop()
        self._py_patch.stop()
        self._script_patch.stop()
        self._zl_home_patch.stop()
        self._env_patch.stop()
        self._td.cleanup()
        return False

    def supervisor_state(self) -> dict[str, str]:
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def marker_lines(self) -> list[str]:
        return self.marker.read_text(encoding="utf-8").splitlines() if self.marker.exists() else []


def test_successful_repair_restarts_writers():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 0, "a successful repair (exit 0) returns 0")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "writers are restarted after a successful repair")
        assert_true(any("--yes" in ln for ln in fx.marker_lines()), "the script ran with --yes")


def test_repair_exits_nonzero_when_a_writer_does_not_come_back_even_if_script_exited_0():
    """Round-7 mutation kill (mut/H3_journal_no_force_nonzero): the same
    `if not restored and code == 0: code = 1` force-nonzero guard fix.py
    has (review H2) exists here too -- recover-webui-db.py exiting 0 must
    not win over a writer that never came back RUNNING afterward.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(journal.common, "restart_after_apply",
                                return_value={"compactor": True, "backup": False}):
            code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 1, "recover-webui-db.py's own exit 0 does not win -- a down writer forces exit 1")


def test_repair_exits_0_when_every_writer_actually_came_back():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(journal.common, "restart_after_apply",
                                return_value={"compactor": True, "backup": True}):
            code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 0, "every writer back RUNNING and script exit 0 -- repair exits 0")


def test_exit_1_restarts_writers():
    """Review M1: exit 1 is recover-webui-db.py's own PRE-FLIGHT refusal
    (a bad D3 check, a declined prompt, no tty) -- every `return 1` site
    is strictly before the script's own `supervisor("stop")` call, so
    openwebui was never touched. Leaving compactor/backup stopped here
    used to kill her chat for no reason (compactor is openwebui's model
    endpoint); this wrapper now restarts whatever IT stopped, same as any
    other non-leave-stopped exit.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "1"}):
            code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 1, "recover-webui-db.py's own exit 1 propagates")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "writers are restarted after exit 1 -- nothing was touched, so nothing stays stopped")


def test_exit_4_leaves_writers_stopped():
    """Review M1: exit 4 (integrity_check on the recovered copy did not
    say "ok") was MISSING from the leave-stopped set entirely -- it fell
    through to the normal restart path even though RUNBOOK_DB_JOURNAL.md
    section 6 says the same "leave stopped, ask for help" as 6/7 do.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "4"}):
            code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 4, "recover-webui-db.py's own exit 4 (integrity check failed) propagates")
        assert_eq(fx.supervisor_state(), {"compactor": "STOPPED", "backup": "STOPPED"},
                  "writers are LEFT STOPPED after exit 4 (RUNBOOK_DB_JOURNAL.md section 6)")


def test_exit_6_leaves_writers_stopped():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "6"}):
            code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 6, "recover-webui-db.py's own exit 6 (swap failed) propagates")
        assert_eq(fx.supervisor_state(), {"compactor": "STOPPED", "backup": "STOPPED"},
                  "writers are LEFT STOPPED after exit 6")


def test_exit_7_leaves_writers_stopped():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "7"}):
            code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 7, "recover-webui-db.py's own exit 7 (stale sidecar) propagates")
        assert_eq(fx.supervisor_state(), {"compactor": "STOPPED", "backup": "STOPPED"},
                  "writers are LEFT STOPPED after exit 7")


def test_exit_4_message_says_openwebui_is_up_not_down():
    """Round-4 MEDIUM: recover-webui-db.py:207-209 already calls
    `supervisor("start")` on exit 4 (integrity_check failed) BEFORE
    returning -- openwebui is genuinely back up (against the untouched
    original webui.db). The printed message must say so, not the
    "OPENWEBUI MUST BE TREATED AS DOWN" text that only fits 6/7 (which
    never restart it).
    """
    import io
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}):
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "4"}):
            captured = io.StringIO()
            with mock.patch.object(sys, "stderr", captured):
                journal.run(["repair", "--apply", "--yes"])
        text = captured.getvalue()
    assert_true("IS UP" in text, "exit 4's message says openwebui is up", extra=text)
    assert_true("MUST BE TREATED AS DOWN" not in text,
                "exit 4's message no longer claims openwebui is down", extra=text)


def test_exit_6_message_says_openwebui_is_down():
    import io
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}):
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "6"}):
            captured = io.StringIO()
            with mock.patch.object(sys, "stderr", captured):
                journal.run(["repair", "--apply", "--yes"])
        text = captured.getvalue()
    assert_true("IS DOWN" in text, "exit 6's message still says openwebui is down", extra=text)


def test_mutation_using_the_generic_down_message_for_exit_4_is_caught():
    """Prove the exit-4 test above bites: inline the OLD (wrong) message
    text for exit 4, confirm it claims openwebui is down, then confirm
    the real code does not.
    """
    old_wrong_note = (
        "recover-webui-db.py already tried `supervisorctl start openwebui` after this "
        "failure, but its own integrity_check said NOT ok first -- do NOT trust that "
        "restart. OPENWEBUI MUST BE TREATED AS DOWN"
    )
    assert_true("MUST BE TREATED AS DOWN" in old_wrong_note,
                "mutated (old) exit-4 message wrongly claims openwebui is down -- confirms the bite")
    import io
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}):
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "4"}):
            captured = io.StringIO()
            with mock.patch.object(sys, "stderr", captured):
                journal.run(["repair", "--apply", "--yes"])
        text = captured.getvalue()
    assert_true("MUST BE TREATED AS DOWN" not in text, "the real (restored) message does not")


def test_exit_3_still_restarts_writers():
    """Exit 3 (a copy failure, pre-swap) is NOT in the leave-stopped set:
    the script's own code path already restarts OpenWebUI itself before
    returning 3 (nothing was touched), so our wrapper's normal restart of
    compactor/backup is still correct.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "3"}):
            code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 3, "exit 3 propagates")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "exit 3 is not in the leave-stopped set -- writers are restarted normally")


def test_mutation_restarting_on_leave_stopped_codes_is_caught():
    """Prove the exit-1/6/7 tests above bite: mutate `_LEAVE_STOPPED_EXIT_CODES`
    to empty, watch writers come back RUNNING after exit 6 (which the
    runbook says must NOT happen), then restore.
    """
    original = journal._LEAVE_STOPPED_EXIT_CODES
    try:
        journal._LEAVE_STOPPED_EXIT_CODES = frozenset()
        with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
            with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "6"}):
                journal.run(["repair", "--apply", "--yes"])
            assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                      "mutated (empty leave-stopped set) restarts writers even after exit 6 -- confirms the bite")
    finally:
        journal._LEAVE_STOPPED_EXIT_CODES = original
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.dict(os.environ, {"FAKE_SCRIPT_EXIT": "6"}):
            journal.run(["repair", "--apply", "--yes"])
        assert_eq(fx.supervisor_state(), {"compactor": "STOPPED", "backup": "STOPPED"},
                  "restored: writers stay stopped again after exit 6")


def test_ctrl_c_during_repair_preserves_exit_6_over_the_deferred_signal():
    """H2: a real Ctrl-C (SIGTERM here, same as the "not forwarded" test
    below -- both are deferred identically by `forward_signals=False`)
    landing while recover-webui-db.py runs, where the script itself goes
    on to exit 6 ("the rename+copy2 swap itself raised" -- a
    leave-stopped code, review M1), must not lose that 6 to a bare
    128+SIGTERM. `region.finish(code, restored=False)` keeps 6.

    Round-6 REVERSAL note: a signal landing BEFORE the child starts (still
    inside/just after `stop_for_apply`) now aborts instead (see
    test_real_sigterm_before_repair_starts_aborts_without_applying below)
    -- this test must send its signal only once the child has actually
    started, via a start marker, rather than a fixed delay that could
    otherwise race into that earlier abort window instead of the child's
    own run.
    """
    import signal
    import threading
    import time

    with tempfile.TemporaryDirectory() as marker_td:
        started_marker = Path(marker_td) / "child.started"
        script_exits_6 = f"""#!/usr/bin/env python3
import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGINT, lambda *a: None)
signal.signal(signal.SIGTERM, lambda *a: None)
Path({str(started_marker)!r}).write_text("1")
time.sleep(0.5)
sys.exit(6)
"""
        with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
            fx.script.write_text(script_exits_6, encoding="utf-8")

            def _send_sigterm_once_child_started():
                deadline = time.time() + 10.0
                while not started_marker.exists() and time.time() < deadline:
                    time.sleep(0.02)
                time.sleep(0.1)
                os.kill(os.getpid(), signal.SIGTERM)

            t = threading.Thread(target=_send_sigterm_once_child_started)
            t.start()
            try:
                journal.run(["repair", "--apply", "--yes"])
            except SystemExit as e:
                assert_eq(e.code, 6, "recover-webui-db.py's own exit 6 is kept over the deferred Ctrl-C")
            else:
                print("FAIL journal.run did not exit")
                sys.exit(1)
            finally:
                t.join()
            assert_eq(fx.supervisor_state(), {"compactor": "STOPPED", "backup": "STOPPED"},
                      "writers are still left stopped (exit 6 is a leave-stopped code)")


def test_real_sigterm_before_repair_starts_aborts_without_applying():
    """Round-6 REVERSAL: a signal that arrives DURING/after `stop_for_apply`
    but strictly BEFORE recover-webui-db.py has started must now ABORT --
    restore whatever this wrapper stopped, and exit 128+SIGTERM WITHOUT
    ever running the repair script at all. This is the mirror image of
    test_ctrl_c_during_repair_preserves_exit_6_over_the_deferred_signal
    (where the signal arrives AFTER the child starts, and the repair still
    runs to completion).
    """
    import signal
    import threading
    import time

    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        real_stop = journal.common.stop_for_apply
        entered_stop = threading.Event()

        def slow_stop(names, **kw):
            entered_stop.set()
            time.sleep(1.0)
            return real_stop(names, **kw)

        def _send_sigterm_once_inside_stop():
            entered_stop.wait(timeout=5.0)
            time.sleep(0.2)
            os.kill(os.getpid(), signal.SIGTERM)

        with mock.patch.object(journal.common, "stop_for_apply", side_effect=slow_stop):
            timer = threading.Thread(target=_send_sigterm_once_inside_stop)
            timer.start()
            try:
                code = journal.run(["repair", "--apply", "--yes"])
            finally:
                timer.join()
        assert_eq(code, 128 + signal.SIGTERM,
                  "a SIGTERM before the repair starts aborts and exits 128+SIGTERM once restored")
        assert_eq(fx.marker_lines(), [], "recover-webui-db.py never ran -- cancelled before it was applied")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "writers are fully restored even though nothing was ever applied")


def test_real_sigterm_during_repair_is_not_forwarded_to_the_child():
    """HIGH: a real SIGTERM sent while recover-webui-db.py runs must NOT
    reach it (no killpg call at all) -- it is deferred instead, exactly
    like a signal during the stop/restart phases, with a repair-specific
    message. The fake script would exit 66 if it ever saw SIGINT itself;
    seeing 0 (its normal exit) instead proves nothing was forwarded.

    Round-6: sends its signal only once the child has actually started
    (a start marker, not a fixed delay) -- see the reversal note on
    test_ctrl_c_during_repair_preserves_exit_6_over_the_deferred_signal.
    """
    import signal
    import threading
    import time

    with tempfile.TemporaryDirectory() as marker_td:
        started_marker = Path(marker_td) / "child.started"
        self_aware_script = f"""#!/usr/bin/env python3
import signal, sys, time
from pathlib import Path
def _got_sigint(signum, frame):
    sys.exit(66)  # would prove (wrongly) that SIGINT reached us
signal.signal(signal.SIGINT, _got_sigint)
Path({str(started_marker)!r}).write_text("1")
time.sleep(0.6)
sys.exit(0)
"""
        with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
            fx.script.write_text(self_aware_script, encoding="utf-8")

            def _send_sigterm_once_child_started():
                deadline = time.time() + 10.0
                while not started_marker.exists() and time.time() < deadline:
                    time.sleep(0.02)
                time.sleep(0.1)
                os.kill(os.getpid(), signal.SIGTERM)

            t = threading.Thread(target=_send_sigterm_once_child_started)
            t.start()
            try:
                journal.run(["repair", "--apply", "--yes"])
            except SystemExit as e:
                assert_eq(e.code, 128 + signal.SIGTERM,
                          "the deferred SIGTERM is acted on only after the repair (and restart) finish")
            else:
                print("FAIL journal.run did not exit on the deferred SIGTERM")
                sys.exit(1)
            finally:
                t.join()
            # The child's own exit code (0, not 66) is only observable via the
            # run log the wrapper writes -- confirm the fake script actually
            # ran to completion undisturbed rather than being cut short.
            run_logs = list((Path(journal.podenv.ZL_HOME) / "runs").glob("*"))
            assert_true(len(run_logs) >= 1, "a run log was written (the child ran)")


def test_local_mode_stops_and_restarts_webuidb_sync_and_prints_staleness_note():
    """M12 (round-2 reviewer's mutation list, 'journal ignores
    webuidb-sync' -- previously SURVIVED): on a WEBUI_DB_LOCAL=true pod,
    webuidb-sync must be stopped and restarted alongside compactor/backup,
    and the wrapper must warn that /data stays stale until the next sync
    (review MEDIUM).
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING", "webuidb-sync": "RUNNING"}) as fx:
        with mock.patch.object(journal.podenv, "detect_webui_db_local", return_value=(True, "test-local")):
            import io as _io
            captured = _io.StringIO()
            with mock.patch("sys.stdout", captured):
                code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 0, "a successful local-mode repair returns 0")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING", "webuidb-sync": "RUNNING"},
                  "webuidb-sync (stopped for the repair) is restarted, not left down")
        assert_true("WEBUI_DB_SYNC_INTERVAL_S" in captured.getvalue(),
                    "the staleness note is printed", extra=captured.getvalue())
        assert_true("--sync-once --force" in captured.getvalue(),
                     "the documented sync-now command is named", extra=captured.getvalue())


_RECORDING_RECOVER_SCRIPT = """#!/usr/bin/env python3
import json, os, sys
state = json.load(open(os.environ["FAKE_SUPERVISOR_STATE"]))
with open(os.environ["FAKE_SCRIPT_MARKER"], "a") as f:
    f.write(json.dumps(state) + "\\n")
sys.exit(int(os.environ.get("FAKE_SCRIPT_EXIT", "0")))
"""


def test_mutation_forgetting_webuidb_sync_in_local_mode_is_caught():
    """Prove the test above bites: mutate `_writers_to_stop` to ignore
    local mode entirely (the M12 mutation), confirming webuidb-sync would
    then be RUNNING at the moment the repair script actually executes --
    exactly the danger RUNBOOK_CHAT_TREE.md's local-mode note exists to
    prevent (it could overwrite the repair on its next sync) -- then
    restore. Uses a RECORDING fake script (captures supervisorctl state
    AT THE INSTANT it runs), the same technique test_ops_fix.py's own H5
    test uses, so this can't be satisfied by a stop-then-immediately-
    restart no-op.
    """
    original = journal._writers_to_stop

    def _check(expect_stopped: bool, label: str) -> None:
        with _Fixture({"compactor": "RUNNING", "backup": "RUNNING", "webuidb-sync": "RUNNING"}) as fx:
            fx.script.write_text(_RECORDING_RECOVER_SCRIPT, encoding="utf-8")
            with mock.patch.object(journal.podenv, "detect_webui_db_local", return_value=(True, "test-local")):
                journal.run(["repair", "--apply", "--yes"])
            lines = fx.marker_lines()
            assert_true(len(lines) == 1, f"{label}: the repair script ran exactly once", extra="\n".join(lines))
            state_when_ran = json.loads(lines[0])
            if expect_stopped:
                assert_eq(state_when_ran.get("webuidb-sync"), "STOPPED",
                          f"{label}: webuidb-sync was STOPPED at the moment the repair script ran")
            else:
                assert_true(state_when_ran.get("webuidb-sync") != "STOPPED",
                             f"{label}: mutated _writers_to_stop left webuidb-sync RUNNING "
                             "through the repair -- confirms the bite", extra=json.dumps(state_when_ran))

    try:
        journal._writers_to_stop = lambda: ["compactor", "backup"]  # M12: ignores local mode
        _check(expect_stopped=False, label="mutated")
    finally:
        journal._writers_to_stop = original
    _check(expect_stopped=True, label="restored")


def test_check_flag_equals_form_still_refused_after_apply():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        code = journal.run(["repair", "--apply", "--yes", "--check=1"])
        assert_eq(code, 2, "'--check=1' after `--apply` is still refused (review R2-B2 normalization)")
        assert_eq(fx.marker_lines(), [], "the script never ran")


def test_help_flag_prints_usage_and_exits_0():
    """Round-6 cosmetic: `zl journal --help`/`-h` used to fall through to
    "unknown subcommand '--help'" (exit 2) -- the same code as a genuine
    usage error, even though asking for help is not one. No-argv at all
    still exits 2 (still a usage error, not a request for help).
    """
    with _Fixture({}) as fx:
        code = journal.run(["--help"])
        assert_eq(code, 0, "`zl journal --help` exits 0")
        assert_eq(fx.marker_lines(), [], "the script never ran")

        code2 = journal.run(["-h"])
        assert_eq(code2, 0, "`zl journal -h` exits 0 too")

        code3 = journal.run([])
        assert_eq(code3, 2, "no subcommand at all is still a usage error (exit 2)")


def test_inflight_guard_refuses_before_any_stop():
    """Round-4 addendum: `zl journal repair --apply` must check
    `requests_in_flight` INSIDE the protected region, before the first
    stop -- a refusal must leave every writer exactly as it was.
    """
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        with mock.patch.object(journal.common, "wait_for_no_inflight_requests",
                                return_value=(False, "still 2 after 300s")):
            code = journal.run(["repair", "--apply", "--yes"])
        assert_eq(code, 1, "a refused in-flight guard exits 1")
        assert_eq(fx.marker_lines(), [], "recover-webui-db.py never ran")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "nothing was stopped -- the guard runs before the first stop")


def test_signal_during_inflight_wait_aborts_without_repairing():
    """Round-4 blocker #3 (review4/p1/probe_ctrlc_wait.py, `zl fix`'s own
    sibling case): a signal deferred WHILE `wait_for_no_inflight_requests`
    is still running must abort `zl journal repair --apply` too --
    nothing has been stopped yet.
    """
    import signal

    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        def fake_guard(*, force, health_url=None):
            os.kill(os.getpid(), signal.SIGTERM)
            return True, "requests_in_flight settled to 0 -- proceeding"

        with mock.patch.object(journal.common, "wait_for_no_inflight_requests", side_effect=fake_guard):
            try:
                code = journal.run(["repair", "--apply", "--yes"])
            except SystemExit as e:
                code = e.code
        assert_eq(code, 1, "the signal during the wait aborts with exit 1")
        assert_eq(fx.marker_lines(), [], "recover-webui-db.py never ran")
        assert_eq(fx.supervisor_state(), {"compactor": "RUNNING", "backup": "RUNNING"},
                  "nothing was stopped -- the signal landed before the first stop")


def test_deferred_signal_during_a_busy_refusal_says_left_untouched_not_restored():
    """F4 regression (mut.py M4: the FIRST abort call site in `zl journal
    repair --apply` -- `if not proceed: ... region.finish(1,
    restored=None)` -- mutated to `restored=True`). `region.finish()` is a
    no-op unless a signal was actually deferred at that exact point, so
    this combines a BUSY (not-proceed) in-flight guard with a SIGTERM
    delivered from inside the guard itself -- nothing was ever stopped
    here, so the message must say "left untouched", never "restored".
    """
    import io
    import signal

    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        def fake_guard(*, force, health_url=None):
            os.kill(os.getpid(), signal.SIGTERM)
            return False, "still 2 after 300s"

        printed = io.StringIO()
        with mock.patch.object(journal.common, "wait_for_no_inflight_requests", side_effect=fake_guard):
            with mock.patch.object(sys, "stderr", printed):
                try:
                    code = journal.run(["repair", "--apply", "--yes"])
                except SystemExit as e:
                    code = e.code
        assert_eq(code, 1, "a refused (busy) in-flight guard with a deferred signal still exits 1")
        assert_eq(fx.marker_lines(), [], "recover-webui-db.py never ran")
        out = printed.getvalue()
        assert_true("left untouched" in out,
                     "nothing was ever stopped here -- must say so, not 'restored'", extra=out)
        assert_true("services are restored" not in out, "never claims 'restored' here", extra=out)


def test_force_busy_flag_reaches_the_inflight_guard_and_is_not_forwarded():
    with _Fixture({"compactor": "RUNNING", "backup": "RUNNING"}) as fx:
        seen = {}

        def fake_guard(*, force, health_url=None):
            seen["force"] = force
            return True, "ok"

        with mock.patch.object(journal.common, "wait_for_no_inflight_requests", side_effect=fake_guard):
            code = journal.run(["repair", "--apply", "--yes", "--force-busy"])
        assert_eq(code, 0, "repair with --force-busy still succeeds")
        assert_eq(seen.get("force"), True, "--force-busy reached the in-flight guard as force=True")
        assert_true(all("--force-busy" not in ln for ln in fx.marker_lines()),
                    "--force-busy was never forwarded to recover-webui-db.py itself",
                    extra="\n".join(fx.marker_lines()))


if __name__ == "__main__":
    test_successful_repair_restarts_writers()
    test_repair_exits_nonzero_when_a_writer_does_not_come_back_even_if_script_exited_0()
    test_repair_exits_0_when_every_writer_actually_came_back()
    test_exit_1_restarts_writers()
    test_exit_4_leaves_writers_stopped()
    test_exit_6_leaves_writers_stopped()
    test_exit_7_leaves_writers_stopped()
    test_exit_3_still_restarts_writers()
    test_mutation_restarting_on_leave_stopped_codes_is_caught()
    test_ctrl_c_during_repair_preserves_exit_6_over_the_deferred_signal()
    test_real_sigterm_before_repair_starts_aborts_without_applying()
    test_real_sigterm_during_repair_is_not_forwarded_to_the_child()
    test_local_mode_stops_and_restarts_webuidb_sync_and_prints_staleness_note()
    test_mutation_forgetting_webuidb_sync_in_local_mode_is_caught()
    test_check_flag_equals_form_still_refused_after_apply()
    test_help_flag_prints_usage_and_exits_0()
    test_inflight_guard_refuses_before_any_stop()
    test_signal_during_inflight_wait_aborts_without_repairing()
    test_deferred_signal_during_a_busy_refusal_says_left_untouched_not_restored()
    test_force_busy_flag_reaches_the_inflight_guard_and_is_not_forwarded()

    print("\nAll ops/commands/journal.py tests passed.")
