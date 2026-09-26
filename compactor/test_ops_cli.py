"""Unit tests for ops/zlops/cli.py -- the top-level dispatcher: the help
table, unknown-command handling, and the "every invocation gets an audit
line, even a crash or Ctrl-C, and a crash is never swallowed silently" rule.

Plain-script style like compactor/test_setup_sshd_script.py: exit 0 pass,
1 fail. No docker/network needed.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import tempfile
from pathlib import Path
from unittest import mock

OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from zlops import cli, common  # noqa: E402


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


def test_every_command_name_is_unique_and_has_a_run_callable():
    names = [n for n, _g, _d, _f in cli.COMMANDS]
    assert_eq(len(names), len(set(names)), "cli.COMMANDS: every command name is unique")
    for name, group, _desc, fn in cli.COMMANDS:
        assert_true(callable(fn), f"cli.COMMANDS: {name}'s run is callable")
        assert_true(group in ("read", "change"), f"cli.COMMANDS: {name}'s group is read or change")


def test_help_lists_every_command_and_exits_0():
    buf = io.StringIO()
    with mock.patch.object(sys, "stdout", buf):
        code = cli._dispatch(["help"])
    assert_eq(code, 0, "`zl help` exits 0")
    out = buf.getvalue()
    for name, _g, _d, _f in cli.COMMANDS:
        assert_true(f"zl {name}" in out, f"`zl help` mentions {name}")
    assert_true("READ-ONLY" in out and "CHANGES THINGS" in out, "`zl help` groups by read-only vs changes-things")


def test_no_args_is_the_same_as_help():
    buf1, buf2 = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "stdout", buf1):
        cli._dispatch([])
    with mock.patch.object(sys, "stdout", buf2):
        cli._dispatch(["help"])
    assert_eq(buf1.getvalue(), buf2.getvalue(), "`zl` with no args prints the same as `zl help`")


def test_unknown_command_exits_2():
    buf = io.StringIO()
    with mock.patch.object(sys, "stderr", buf):
        code = cli._dispatch(["not-a-real-command"])
    assert_eq(code, 2, "an unknown command exits 2")
    assert_true("unknown command" in buf.getvalue(), "an unknown command says so on stderr")


def _run_main_with_tmp_home(argv, dispatch_side_effect=None):
    """Runs cli.main(argv) with podenv.AUDIT_LOG redirected into a fresh
    temp dir, optionally forcing `_dispatch` to raise/return something
    specific. Returns (exit_code_or_None, audit_lines, stdout, stderr,
    raised_exception_or_None).
    """
    with tempfile.TemporaryDirectory() as td:
        audit_log = Path(td) / "audit.log"
        out, err = io.StringIO(), io.StringIO()
        code = None
        raised = None
        with mock.patch.object(common.podenv, "AUDIT_LOG", audit_log), \
             mock.patch.object(sys, "stdout", out), \
             mock.patch.object(sys, "stderr", err):
            if dispatch_side_effect is not None:
                with mock.patch.object(cli, "_dispatch", side_effect=dispatch_side_effect):
                    try:
                        code = cli.main(argv)
                    except BaseException as e:  # noqa: BLE001 -- intentionally observing whatever escapes
                        raised = e
            else:
                code = cli.main(argv)
        lines = audit_log.read_text(encoding="utf-8").splitlines() if audit_log.exists() else []
        return code, lines, out.getvalue(), err.getvalue(), raised


def test_main_writes_one_audit_line_per_invocation():
    code, lines, _out, _err, _raised = _run_main_with_tmp_home(["help"])
    assert_eq(code, 0, "main(['help']) returns 0")
    assert_eq(len(lines), 1, "main() writes exactly one audit line")
    rec = json.loads(lines[0])
    assert_eq(rec["argv"], ["help"], "the audit line records argv")
    assert_eq(rec["exit_code"], 0, "the audit line records the exit code")


def test_main_writes_audit_line_and_prints_exit_meaning_on_unknown_command():
    code, lines, _out, err, _raised = _run_main_with_tmp_home(["bogus"])
    assert_eq(code, 2, "main(['bogus']) returns 2")
    assert_eq(len(lines), 1, "an unknown command still gets one audit line")
    assert_true("exit 2" in err, "main() prints the exit-code meaning to stderr")


def test_main_keyboard_interrupt_still_audits_130():
    code, lines, _out, err, _raised = _run_main_with_tmp_home(["fix"], dispatch_side_effect=KeyboardInterrupt)
    assert_eq(code, 130, "a KeyboardInterrupt maps to exit 130")
    assert_eq(len(lines), 1, "a KeyboardInterrupt still gets one audit line")
    rec = json.loads(lines[0])
    assert_eq(rec["exit_code"], 130, "the audit line records 130 for a KeyboardInterrupt")
    assert_true("interrupted" in err, "main() says it was interrupted")


def test_main_negative_exit_code_converts_to_128_plus_signal():
    """review LOW (common.py:55-57): a wrapped child killed by a signal
    comes back as `_dispatch` returning a NEGATIVE code (Python's own
    subprocess convention). `exit_meaning` prints that as "reported as
    exit 128+N", but `sys.exit()` (in __main__.py) passing the RAW
    negative number through gets OS-truncated to `256+code` (e.g. -2 ->
    254), not 130 -- the printed message and the real process exit code
    used to disagree. `cli.main` must convert BEFORE the audit line, the
    printed note, and its own return value all see it, so they agree.
    """
    code, lines, _out, err, _raised = _run_main_with_tmp_home(["fix"], dispatch_side_effect=lambda args: -2)
    assert_eq(code, 128 + 2, "main() converts a returncode of -2 (SIGINT) to 130")
    assert_eq(len(lines), 1, "one audit line is written")
    rec = json.loads(lines[0])
    assert_eq(rec["exit_code"], 130, "the audit line records the CONVERTED code, not -2")
    assert_true("exit 130" in err, "the printed exit-meaning matches the converted code")


def test_main_crash_still_audits_and_is_never_swallowed():
    def boom(_args):
        raise ValueError("planted crash")

    code, lines, _out, err, raised = _run_main_with_tmp_home(["fix"], dispatch_side_effect=boom)
    assert_eq(code, None, "main() does not return normally on a genuine bug")
    assert_true(raised is not None, "main() re-raises a genuine bug instead of swallowing it")
    assert_true(isinstance(raised, ValueError), "the re-raised exception is the original one, unwrapped")
    assert_eq(len(lines), 1, "a crash still gets one audit line before the traceback propagates")
    rec = json.loads(lines[0])
    assert_eq(rec["exit_code"], 1, "the audit line records exit 1 for a crash")
    assert_true("exit 1" in err, "main() prints the exit-1 meaning before re-raising")


def test_subprocess_smoke_help_and_unknown():
    """One real end-to-end invocation exactly as the pod would run it:
    `python -m zlops help`, PYTHONPATH=ops, a throwaway ZL_HOME.
    """
    with tempfile.TemporaryDirectory() as td:
        env = {"PYTHONPATH": str(OPS_DIR), "ZL_HOME": td, "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run(
            [sys.executable, "-m", "zlops", "help"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert_eq(r.returncode, 0, "subprocess: `python -m zlops help` exits 0", )
        assert_true("zl -- one command" in r.stdout, "subprocess: help banner printed",
                    extra=r.stdout[:500])

        r2 = subprocess.run(
            [sys.executable, "-m", "zlops", "not-a-command"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert_eq(r2.returncode, 2, "subprocess: an unknown command exits 2")


def test_status_brief_skips_the_audit_write():
    """review MEDIUM: `zl status --brief` is bootstrap.sh's login-hook
    command, run on every interactive SSH login -- it must not write an
    audit line (a MooseFS stall on that write must never risk hanging a
    login, and a line per login would drown out real operator actions).
    Every OTHER status invocation still gets one, same as any command.
    """
    with mock.patch.object(cli, "_dispatch", return_value=0):
        code, lines, _out, _err, _raised = _run_main_with_tmp_home(["status", "--brief"])
        assert_eq(code, 0, "main(['status', '--brief']) returns 0")
        assert_eq(lines, [], "status --brief writes NO audit line")

        code2, lines2, _out2, _err2, _raised2 = _run_main_with_tmp_home(["status"])
        assert_eq(code2, 0, "main(['status']) returns 0")
        assert_eq(len(lines2), 1, "plain `zl status` (no --brief) still gets its audit line")


def test_health_exit_notes_override_the_generic_dry_run_text():
    """review LOW: `zl health`'s exit 3 means 'warnings found', not the
    generic (and, for a read-only command, misleading) 'dry run found
    pending work'.
    """
    with mock.patch.object(cli, "_dispatch", return_value=3):
        _code, _lines, _out, err, _raised = _run_main_with_tmp_home(["health"])
        assert_true("warnings" in err, "health's exit 3 note mentions warnings", extra=err)
        assert_true("dry run found pending work" not in err,
                     "health's exit 3 note does NOT use the generic dry-run text", extra=err)

    with mock.patch.object(cli, "_dispatch", return_value=3):
        _code, _lines, _out, err2, _raised2 = _run_main_with_tmp_home(["bundle"])
        assert_true("dry run found pending work" in err2,
                     "an unrelated command's exit 3 still gets the generic text", extra=err2)


if __name__ == "__main__":
    test_every_command_name_is_unique_and_has_a_run_callable()
    test_help_lists_every_command_and_exits_0()
    test_no_args_is_the_same_as_help()
    test_unknown_command_exits_2()
    test_main_writes_one_audit_line_per_invocation()
    test_main_writes_audit_line_and_prints_exit_meaning_on_unknown_command()
    test_main_keyboard_interrupt_still_audits_130()
    test_main_negative_exit_code_converts_to_128_plus_signal()
    test_main_crash_still_audits_and_is_never_swallowed()
    test_status_brief_skips_the_audit_write()
    test_health_exit_notes_override_the_generic_dry_run_text()
    test_subprocess_smoke_help_and_unknown()

    print("\nAll ops/cli.py tests passed.")
