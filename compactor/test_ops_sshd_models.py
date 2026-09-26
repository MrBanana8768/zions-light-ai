"""Unit tests for ops/zlops/commands/sshd.py and models.py's own wiring of
the shared R2-B2 passthrough-normalization gate (the logic itself is
exhaustively tested against `common.normalize_passthrough`/
`refuse_passthrough_write_flags` directly in test_ops_common.py -- these
tests only prove each command wired its OWN `all_flags`/`write_flags`
correctly, reproducing REVIEW2's exact repro: `zl sshd -- --appl`).

Plain-script style: exit 0 pass, 1 fail.
"""

from __future__ import annotations

import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from zlops.commands import models, sshd  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def test_sshd_refuses_prefix_abbreviated_apply_without_zl_apply():
    # REVIEW2's own repro: `zl sshd -- --appl`.
    with mock.patch.object(sshd.common, "run_stream") as run_stream:
        code = sshd.run(["--", "--appl"])
        assert_eq(code, 2, "`zl sshd -- --appl` is refused without zl's own --apply")
        run_stream.assert_not_called()


def test_sshd_refuses_force_equals_form():
    with mock.patch.object(sshd.common, "run_stream") as run_stream:
        code = sshd.run(["--", "--force=1"])
        assert_eq(code, 2, "`zl sshd -- --force=1` is refused without zl's own --apply")
        run_stream.assert_not_called()


def test_sshd_allows_prefix_form_with_zl_apply_and_yes():
    with mock.patch.object(sshd.common, "run_stream", return_value=0) as run_stream, \
         mock.patch.object(sshd.podenv, "RUNPOD_PUBLIC_IP", ""), \
         mock.patch.object(sshd.podenv, "RUNPOD_TCP_PORT_22", ""):
        code = sshd.run(["--apply", "--yes", "--", "--forc"])
        assert_eq(code, 0, "with zl's own --apply+--yes, the prefix form goes through")
        run_stream.assert_called_once()


def test_models_delete_still_refused_without_apply():
    with mock.patch.object(models.common, "run_stream") as run_stream:
        code = models.run(["--delete", "coder3101/whatever"])
        assert_eq(code, 2, "`zl models --delete X` (no zl --apply) is refused")
        run_stream.assert_not_called()


def test_models_delete_equals_form_still_refused_without_apply():
    with mock.patch.object(models.common, "run_stream") as run_stream:
        code = models.run(["--delete=coder3101/whatever"])
        assert_eq(code, 2, "`zl models --delete=X` (no zl --apply) is refused")
        run_stream.assert_not_called()


def test_models_ambiguous_prefix_refused():
    # Not reachable via clean-models.sh's real flag set (no two flags share
    # a prefix today), but proves normalize_passthrough's ambiguity check
    # is actually wired into models.py, not just declared.
    with mock.patch.object(models, "_ALL_FLAGS", ("--delete", "--delete-all")):
        with mock.patch.object(models.common, "run_stream") as run_stream:
            code = models.run(["--del"])
            assert_eq(code, 2, "an ambiguous prefix is refused, not guessed")
            run_stream.assert_not_called()


if __name__ == "__main__":
    test_sshd_refuses_prefix_abbreviated_apply_without_zl_apply()
    test_sshd_refuses_force_equals_form()
    test_sshd_allows_prefix_form_with_zl_apply_and_yes()
    test_models_delete_still_refused_without_apply()
    test_models_delete_equals_form_still_refused_without_apply()
    test_models_ambiguous_prefix_refused()

    print("\nAll ops sshd/models passthrough wiring tests passed.")
