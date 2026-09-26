"""Unit tests for ops/zlops/commands/backup.py -- the compactor/backup.py
wrapper. Round-3 review H1: `now`/`list`/`verify` USED to forward
everything after their own arguments straight to backup.py with no gate at
all. backup.py's own argparse checks `--restore` the same way it checks
`--once`/`--list`/`--verify` -- so `zl backup now --restore ARCHIVE --yes`
reached `restore_backup` outright, and `zl backup list --prune-pre-restore
0` ran the prune. This file proves the gate: only `--json` (and any
argparse-abbreviation-normalized form of it) is accepted; `--restore` (and
any prefix of it), `--prune-pre-restore`, `--daemon`, `--list`, `--verify`,
`--yes` and `--list-pre-restore` are all refused, whatever subcommand they
follow.

Plain-script style like compactor/test_setup_sshd_script.py: fakes go on
PATH in a temp dir; exit 0 pass, 1 fail.
"""

from __future__ import annotations

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

from zlops.commands import backup  # noqa: E402


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


_FAKE_BACKUP_PY = """#!/usr/bin/env python3
import os, sys
marker = os.environ.get("FAKE_BACKUP_MARKER")
if marker:
    with open(marker, "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\\n")
sys.exit(0)
"""


class _Fixture:
    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        td = Path(self._td.name)
        self.script = td / "backup.py"
        self.script.write_text(_FAKE_BACKUP_PY, encoding="utf-8")
        self.script.chmod(self.script.stat().st_mode | stat.S_IEXEC)
        self.marker = td / "marker.log"
        self._env_patch = mock.patch.dict(os.environ, {"FAKE_BACKUP_MARKER": str(self.marker)})
        self._env_patch.start()
        self._py_patch = mock.patch.object(backup.podenv, "COMPACTOR_PY", Path(sys.executable))
        self._py_patch.start()
        self._pkg_patch = mock.patch.object(backup.podenv, "COMPACTOR_PKG", td)
        self._pkg_patch.start()
        self._script_patch = mock.patch.object(backup, "_BACKUP_PY", self.script)
        self._script_patch.start()
        return self

    def __exit__(self, *exc):
        self._script_patch.stop()
        self._pkg_patch.stop()
        self._py_patch.stop()
        self._env_patch.stop()
        self._td.cleanup()
        return False

    def marker_lines(self) -> list[str]:
        return self.marker.read_text(encoding="utf-8").splitlines() if self.marker.exists() else []


def test_now_with_json_passes_through():
    with _Fixture() as fx:
        code = backup.run(["now", "--json"])
        assert_eq(code, 0, "`zl backup now --json` succeeds")
        assert_true(any(ln == "--once --json" for ln in fx.marker_lines()),
                    "backup.py saw --once and --json", extra="\n".join(fx.marker_lines()))


def test_list_with_no_extra_flags_passes_through():
    with _Fixture() as fx:
        code = backup.run(["list"])
        assert_eq(code, 0, "`zl backup list` succeeds")
        assert_true(any(ln == "--list" for ln in fx.marker_lines()), "backup.py saw --list")


def test_now_restore_is_refused():
    with _Fixture() as fx:
        code = backup.run(["now", "--restore", "ARCHIVE", "--yes"])
        assert_eq(code, 2, "`zl backup now --restore ARCHIVE --yes` is refused")
        assert_eq(fx.marker_lines(), [], "backup.py never ran")


def test_now_restore_prefix_is_refused():
    with _Fixture() as fx:
        code = backup.run(["now", "--rest", "ARCHIVE"])
        assert_eq(code, 2, "the argparse-prefix form '--rest' is refused too, not just the exact flag")
        assert_eq(fx.marker_lines(), [], "backup.py never ran")


def test_now_restore_equals_form_is_refused():
    with _Fixture() as fx:
        code = backup.run(["now", "--restore=ARCHIVE"])
        assert_eq(code, 2, "the `--flag=value` form is refused too")
        assert_eq(fx.marker_lines(), [], "backup.py never ran")


def test_list_prune_pre_restore_is_refused():
    with _Fixture() as fx:
        code = backup.run(["list", "--prune-pre-restore", "0"])
        assert_eq(code, 2, "`zl backup list --prune-pre-restore 0` is refused")
        assert_eq(fx.marker_lines(), [], "backup.py never ran -- the prune never happened")


def test_verify_extra_write_flags_are_refused():
    with _Fixture() as fx:
        code = backup.run(["verify", "some-archive.tar.gz", "--daemon"])
        assert_eq(code, 2, "`zl backup verify ARCHIVE --daemon` is refused")
        assert_eq(fx.marker_lines(), [], "backup.py never ran")


def test_verify_json_passes_through():
    with _Fixture() as fx:
        code = backup.run(["verify", "some-archive.tar.gz", "--json"])
        assert_eq(code, 0, "`zl backup verify ARCHIVE --json` succeeds")
        assert_true(any(ln == "--verify some-archive.tar.gz --json" for ln in fx.marker_lines()),
                    "backup.py saw --verify ARCHIVE --json", extra="\n".join(fx.marker_lines()))


def test_restore_subcommand_always_refuses_regardless_of_flags():
    with _Fixture() as fx:
        code = backup.run(["restore", "--yes", "some-archive.tar.gz"])
        assert_eq(code, 1, "`zl backup restore` always refuses")
        assert_eq(fx.marker_lines(), [], "backup.py never ran")


def test_ambiguous_prefix_is_refused_not_guessed():
    with _Fixture() as fx:
        code = backup.run(["now", "--l"])  # a prefix of both --list and --list-pre-restore
        assert_eq(code, 2, "an ambiguous prefix is refused outright")
        assert_eq(fx.marker_lines(), [], "backup.py never ran")


def test_mutation_removing_the_gate_lets_restore_through():
    """Prove the tests above bite: inline the OLD (ungated) `now` handler
    -- straight passthrough, no `_gate_passthrough` call at all -- and
    show `--restore` reaches backup.py. Never touches the real
    `backup.run`; restores nothing because nothing real was mutated.
    """
    with _Fixture() as fx:
        env = backup.common.merged_environ()
        old_style_argv = [str(backup.podenv.COMPACTOR_PY), str(backup._BACKUP_PY), "--once",
                           "--restore", "ARCHIVE", "--yes"]
        code = backup.common.run_stream(old_style_argv, env=env)
        assert_eq(code, 0, "mutated (ungated) passthrough runs backup.py with --restore -- confirms the bite")
        assert_true(any("--restore" in ln for ln in fx.marker_lines()),
                    "backup.py's own argv included --restore, unrefused", extra="\n".join(fx.marker_lines()))
    # And the real, gated `now` refuses the same input:
    with _Fixture() as fx:
        code = backup.run(["now", "--restore", "ARCHIVE", "--yes"])
        assert_eq(code, 2, "the real (gated) `now` refuses instead")
        assert_eq(fx.marker_lines(), [], "backup.py never ran")


if __name__ == "__main__":
    test_now_with_json_passes_through()
    test_list_with_no_extra_flags_passes_through()
    test_now_restore_is_refused()
    test_now_restore_prefix_is_refused()
    test_now_restore_equals_form_is_refused()
    test_list_prune_pre_restore_is_refused()
    test_verify_extra_write_flags_are_refused()
    test_verify_json_passes_through()
    test_restore_subcommand_always_refuses_regardless_of_flags()
    test_ambiguous_prefix_is_refused_not_guessed()
    test_mutation_removing_the_gate_lets_restore_through()

    print("\nAll ops/commands/backup.py tests passed.")
