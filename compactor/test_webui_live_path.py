"""Tests for scripts/_webui_live_path.py — the shared D3 helper the four
webui.db-writing operator scripts (repair-chat-tree.py, fix-stale-
unfinished.py, fix-encoded-messages.py, clean-decoration.py) and
recover-webui-db.py use to resolve OpenWebUI's LIVE database and refuse to
silently edit the periodically-published SNAPSHOT while local mode
(`WEBUI_DB_LOCAL=true`) is active. See that module's own docstring and
`dbmove/findings.md` section 4 (D3) for the defect this closes.

Run directly:
    python compactor/test_webui_live_path.py
"""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
_MODULE = _SCRIPTS / "_webui_live_path.py"
_spec = importlib.util.spec_from_file_location("_webui_live_path_test_target", _MODULE)
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

_FAILS = []


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        _FAILS.append(label)
        return
    print(f"  ok   {label}")


def assert_true(cond, label, detail=""):
    if not cond:
        print(f"FAIL {label}" + (f" ({detail})" if detail else ""))
        _FAILS.append(label)
        return
    print(f"  ok   {label}")


class _Isolated:
    """Save/restore every piece of process-global state this module's
    functions read, so one test's monkeypatching never leaks into the
    next. Covers os.environ (the four env vars this module reads),
    L.LOCAL_DB / L.SNAPSHOT_DB / L.FORENSICS_ROOT (module attributes a
    test overrides to point at tmp files instead of real system paths),
    and pathlib.Path.read_bytes (patched only by the /proc/1/environ
    tests, to simulate a value this test process cannot actually own)."""

    ENV_KEYS = ("WEBUI_DB_LOCAL", "DATABASE_URL", "WEBUI_LOCAL_DB", "WEBUI_SNAPSHOT_DB")

    def __enter__(self):
        self._env_before = {k: os.environ.get(k) for k in self.ENV_KEYS}
        for k in self.ENV_KEYS:
            os.environ.pop(k, None)
        self._local_db = L.LOCAL_DB
        self._snapshot_db = L.SNAPSHOT_DB
        self._forensics_root = L.FORENSICS_ROOT
        self._read_bytes = Path.read_bytes
        return self

    def __exit__(self, *exc):
        for k, v in self._env_before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        L.LOCAL_DB = self._local_db
        L.SNAPSHOT_DB = self._snapshot_db
        L.FORENSICS_ROOT = self._forensics_root
        Path.read_bytes = self._read_bytes
        return False


# ===========================================================================
# fold_webui_db_local -- the exact table webuidb.live_webui_db() uses
# ===========================================================================


def test_fold_true_spellings():
    for v in ("", "true", "TRUE", " True ", "1", "yes", "YES", "on"):
        assert_eq(L.fold_webui_db_local(v), True, f"fold({v!r}) is True")


def test_fold_false_spellings():
    for v in ("false", "FALSE", " False ", "0", "no", "off"):
        assert_eq(L.fold_webui_db_local(v), False, f"fold({v!r}) is False")


def test_fold_unrecognized_is_none():
    assert_eq(L.fold_webui_db_local("banana"), None, "an unrecognized spelling folds to None, not a guess")
    assert_eq(L.fold_webui_db_local(None), True, "fold(None) treated as empty -> True")


# ===========================================================================
# detect_local_mode -- priority order
# ===========================================================================


def test_env_unset_defaults_to_local_true():
    with _Isolated():
        is_local, reason = L.detect_local_mode()
        assert_eq(is_local, True, "unset WEBUI_DB_LOCAL defaults to local (matches webuidb.live_webui_db())")
        assert_true("unset/empty" in reason, "default reason names unset/empty", reason)


def test_env_explicit_false_wins():
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "false"
        is_local, reason = L.detect_local_mode()
        assert_eq(is_local, False, "explicit WEBUI_DB_LOCAL=false in this shell is honored")
        assert_true("this shell" in reason, "reason names the shell's own environment", reason)


def test_env_explicit_true_wins_over_database_url_saying_otherwise():
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "true"
        os.environ["DATABASE_URL"] = f"sqlite:///{L.SNAPSHOT_DB}"
        is_local, reason = L.detect_local_mode()
        assert_eq(is_local, True, "this shell's own WEBUI_DB_LOCAL outranks DATABASE_URL")


def test_database_url_hints_local():
    with _Isolated():
        os.environ["DATABASE_URL"] = f"sqlite:///{L.LOCAL_DB}"
        is_local, reason = L.detect_local_mode()
        assert_eq(is_local, True, "DATABASE_URL naming LOCAL_DB is honored when WEBUI_DB_LOCAL is unset")
        assert_true("DATABASE_URL" in reason, "reason names DATABASE_URL", reason)


def test_database_url_hints_snapshot():
    with _Isolated():
        os.environ["DATABASE_URL"] = f"sqlite:///{L.SNAPSHOT_DB}"
        is_local, reason = L.detect_local_mode()
        assert_eq(is_local, False, "DATABASE_URL naming SNAPSHOT_DB is honored when WEBUI_DB_LOCAL is unset")


def test_proc1_environ_used_when_shell_env_and_database_url_are_silent():
    """The fresh-web-terminal-shell case this module exists for: nothing
    in THIS process's own environment says anything, but supervisord
    (pid 1 in the image) carries the real value -- simulated here by
    patching Path.read_bytes rather than actually owning pid 1."""
    with _Isolated():
        real_read_bytes = Path.read_bytes

        def fake_read_bytes(self):
            if str(self) == "/proc/1/environ":
                return b"PATH=/usr/bin\x00WEBUI_DB_LOCAL=false\x00OTHER=1\x00"
            return real_read_bytes(self)

        Path.read_bytes = fake_read_bytes
        is_local, reason = L.detect_local_mode()
        assert_eq(is_local, False, "/proc/1/environ's WEBUI_DB_LOCAL=false is honored")
        assert_true("/proc/1/environ" in reason, "reason names /proc/1/environ", reason)


def test_proc1_environ_missing_var_falls_through():
    with _Isolated():
        real_read_bytes = Path.read_bytes

        def fake_read_bytes(self):
            if str(self) == "/proc/1/environ":
                return b"PATH=/usr/bin\x00OTHER=1\x00"
            return real_read_bytes(self)

        Path.read_bytes = fake_read_bytes
        is_local, reason = L.detect_local_mode()
        # No WEBUI_DB_LOCAL in /proc/1/environ either -> falls through to
        # the fs/process check, then the empty-means-true default.
        assert_eq(is_local, True, "falls through to the default when /proc/1/environ has no WEBUI_DB_LOCAL either")


def test_local_file_open_by_a_process_detects_local_mode(tmp_dir):
    """Explicit scenario from the brief: an unset/empty WEBUI_DB_LOCAL,
    with the local file open by a running (OTHER) process, detects local
    mode. Spawns a real subprocess that opens the file and holds it,
    isolated from the default-true fallback by first proving detection
    also works when /proc/1/environ explicitly can't answer either."""
    local_db = tmp_dir / "webui.db"
    local_db.write_bytes(b"x" * 16)
    with _Isolated():
        L.LOCAL_DB = local_db
        real_read_bytes = Path.read_bytes

        def fake_read_bytes(self):
            if str(self) == "/proc/1/environ":
                raise OSError("no /proc/1 in this test")
            return real_read_bytes(self)

        Path.read_bytes = fake_read_bytes

        assert_true(L._local_db_open_by_a_process() is False, "not open yet -> False before the holder starts")

        holder = subprocess.Popen(
            [sys.executable, "-c",
             f"import time; f = open(r'{local_db}', 'rb'); time.sleep(20)"],
        )
        try:
            found = False
            for _ in range(50):
                if L._local_db_open_by_a_process():
                    found = True
                    break
                time.sleep(0.2)
            assert_true(found, "_local_db_open_by_a_process() finds the holder process")

            is_local, reason = L.detect_local_mode()
            assert_eq(is_local, True, "unset env + local file open by a process detects local mode")
            assert_true(str(local_db) in reason or "open by a running process" in reason,
                        "reason names the open-file signal", reason)
        finally:
            holder.terminate()
            holder.wait(timeout=5)


def test_local_file_not_open_falls_through_to_default():
    with _Isolated():
        # LOCAL_DB pointed at a path that does not exist at all.
        L.LOCAL_DB = Path("/nonexistent/for/this/test/webui.db")
        assert_true(L._local_db_open_by_a_process() is False, "a LOCAL_DB that doesn't exist is never 'open'")


# ===========================================================================
# refuse_if_snapshot_in_local_mode / snapshot_problem_message
# ===========================================================================


def test_apply_against_snapshot_in_local_mode_refuses(tmp_dir):
    snapshot = tmp_dir / "snapshot.db"
    snapshot.write_bytes(b"x")
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "true"
        L.SNAPSHOT_DB = snapshot
        msg = L.snapshot_problem_message(snapshot, "some-tool.py", dry_run=False)
        assert_true(msg is not None, "a refusal message is produced")
        assert_true(str(snapshot) in msg, "message names the snapshot path", msg)
        assert_true("some-tool.py" in msg, "message names the tool", msg)
        assert_true("openwebui" in msg and "webuidb-sync" in msg,
                    "message tells the operator to stop BOTH openwebui and webuidb-sync", msg)


def test_apply_against_local_path_in_local_mode_is_fine():
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "true"
        local = Path("/some/local/webui.db")
        L.LOCAL_DB = local
        msg = L.snapshot_problem_message(local, "some-tool.py", dry_run=False)
        assert_eq(msg, None, "writing the LOCAL path itself is never refused")


def test_dry_run_against_snapshot_in_local_mode_warns_not_refuses(tmp_dir):
    snapshot = tmp_dir / "snapshot.db"
    snapshot.write_bytes(b"x")
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "true"
        L.SNAPSHOT_DB = snapshot
        msg = L.snapshot_problem_message(snapshot, "some-tool.py", dry_run=True)
        assert_true(msg is not None, "a dry run against the snapshot in local mode still gets a message")
        assert_true("REFUS" not in msg.upper() or "refus" not in msg, "dry-run wording is a warning, not a refusal")


def test_local_mode_off_snapshot_path_is_fine(tmp_dir):
    """The 'local mode off + /data works as before' scenario: even
    pointed straight at the snapshot, nothing fires once WEBUI_DB_LOCAL
    is explicitly false."""
    snapshot = tmp_dir / "snapshot.db"
    snapshot.write_bytes(b"x")
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "false"
        L.SNAPSHOT_DB = snapshot
        msg_apply = L.snapshot_problem_message(snapshot, "some-tool.py", dry_run=False)
        msg_dry = L.snapshot_problem_message(snapshot, "some-tool.py", dry_run=True)
        assert_eq(msg_apply, None, "WEBUI_DB_LOCAL=false: --apply against /data is untouched")
        assert_eq(msg_dry, None, "WEBUI_DB_LOCAL=false: a dry run against /data is untouched")


def test_arbitrary_path_never_triggers_anything(tmp_dir):
    """Every existing test fixture in the other four scripts' suites uses
    an arbitrary tmp path -- neither LOCAL_DB nor SNAPSHOT_DB. This must
    stay completely silent regardless of local-mode detection, or every
    one of those suites' --json assertions breaks (the exact regression
    caught while building this fix)."""
    arbitrary = tmp_dir / "some_test_fixture.db"
    arbitrary.write_bytes(b"x")
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "true"  # local mode ACTIVE
        for dry_run in (True, False):
            msg = L.snapshot_problem_message(arbitrary, "some-tool.py", dry_run=dry_run)
            assert_eq(msg, None, f"an arbitrary test path never matches (dry_run={dry_run})")


def test_refuse_if_snapshot_in_local_mode_exits_1(tmp_dir):
    snapshot = tmp_dir / "snapshot.db"
    snapshot.write_bytes(b"x")
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "true"
        L.SNAPSHOT_DB = snapshot
        exited = {"code": None}
        try:
            L.refuse_if_snapshot_in_local_mode(snapshot, tool_name="some-tool.py", dry_run=False)
        except SystemExit as e:
            exited["code"] = e.code
        assert_eq(exited["code"], 1, "refuse_if_snapshot_in_local_mode exits 1 on a writing-mode refusal")


def test_refuse_if_snapshot_in_local_mode_dry_run_does_not_exit(tmp_dir):
    snapshot = tmp_dir / "snapshot.db"
    snapshot.write_bytes(b"x")
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "true"
        L.SNAPSHOT_DB = snapshot
        try:
            L.refuse_if_snapshot_in_local_mode(snapshot, tool_name="some-tool.py", dry_run=True)
            ok = True
        except SystemExit:
            ok = False
        assert_true(ok, "a dry-run warning never calls sys.exit")


# ===========================================================================
# maybe_print_final_sync_hint -- gated on the ACTUAL path just written,
# not on the generic local-mode guess (the regression this module's own
# fix caught: it used to fire on every write anywhere once the default-
# true fallback was added, corrupting every --json test fixture's stdout)
# ===========================================================================


def test_sync_hint_fires_only_for_the_local_path(tmp_dir, capsys=None):
    local = tmp_dir / "local.db"
    other = tmp_dir / "not_local.db"
    with _Isolated():
        L.LOCAL_DB = local
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            L.maybe_print_final_sync_hint("some-tool.py", other)
        assert_eq(buf.getvalue(), "", "no hint printed for a path that isn't LOCAL_DB")

        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            L.maybe_print_final_sync_hint("some-tool.py", local)
        out = buf2.getvalue()
        assert_true("supervisorctl stop openwebui webuidb-sync" in out, "hint includes the stop command", out)
        assert_true("--sync-once --force" in out, "hint includes the sync-once --force command", out)
        assert_true("some-tool.py" in out, "hint names the tool", out)


def test_sync_hint_silent_even_with_local_mode_defaulted_true(tmp_dir):
    """The regression itself: WEBUI_DB_LOCAL unset (default True) must
    NOT be enough on its own to print the hint for an unrelated path."""
    local = tmp_dir / "local.db"
    unrelated = tmp_dir / "unrelated.db"
    with _Isolated():
        L.LOCAL_DB = local
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            L.maybe_print_final_sync_hint("some-tool.py", unrelated)
        assert_eq(buf.getvalue(), "", "default-true local mode alone does not trigger the hint for another path")


# ===========================================================================
# forensics_backup_dir -- D10
# ===========================================================================


def test_forensics_backup_dir_none_when_not_local_disk(tmp_dir):
    local = tmp_dir / "local.db"
    other = tmp_dir / "elsewhere.db"
    with _Isolated():
        L.LOCAL_DB = local
        L.FORENSICS_ROOT = tmp_dir / "forensics"
        result = L.forensics_backup_dir("some-tool", "20260101T000000Z", other)
        assert_eq(result, None, "no relocation for a db_path that isn't LOCAL_DB")


def test_forensics_backup_dir_relocates_when_local_disk(tmp_dir):
    local = tmp_dir / "local.db"
    with _Isolated():
        L.LOCAL_DB = local
        forensics_root = tmp_dir / "forensics"
        L.FORENSICS_ROOT = forensics_root
        result = L.forensics_backup_dir("some-tool", "20260101T000000Z", local)
        assert_eq(result, forensics_root / "some-tool-20260101T000000Z",
                  "backup relocates to /data/forensics/<tool>-<stamp> for the local file")


def test_live_db_path_matches_detect_local_mode():
    with _Isolated():
        os.environ["WEBUI_DB_LOCAL"] = "false"
        path, is_local, reason = L.live_db_path()
        assert_eq(is_local, False, "live_db_path reports the same is_local detect_local_mode does")
        assert_eq(path, L.SNAPSHOT_DB, "live_db_path returns SNAPSHOT_DB when local mode is off")
        os.environ["WEBUI_DB_LOCAL"] = "true"
        path2, is_local2, _ = L.live_db_path()
        assert_eq(is_local2, True, "live_db_path flips to local when WEBUI_DB_LOCAL=true")
        assert_eq(path2, L.LOCAL_DB, "live_db_path returns LOCAL_DB when local mode is on")


if __name__ == "__main__":
    import tempfile

    def _with_tmp_dir(fn):
        with tempfile.TemporaryDirectory(prefix="webui-live-path-test-") as d:
            fn(Path(d))

    test_fold_true_spellings()
    test_fold_false_spellings()
    test_fold_unrecognized_is_none()
    test_env_unset_defaults_to_local_true()
    test_env_explicit_false_wins()
    test_env_explicit_true_wins_over_database_url_saying_otherwise()
    test_database_url_hints_local()
    test_database_url_hints_snapshot()
    test_proc1_environ_used_when_shell_env_and_database_url_are_silent()
    test_proc1_environ_missing_var_falls_through()
    _with_tmp_dir(test_local_file_open_by_a_process_detects_local_mode)
    test_local_file_not_open_falls_through_to_default()
    _with_tmp_dir(test_apply_against_snapshot_in_local_mode_refuses)
    test_apply_against_local_path_in_local_mode_is_fine()
    _with_tmp_dir(test_dry_run_against_snapshot_in_local_mode_warns_not_refuses)
    _with_tmp_dir(test_local_mode_off_snapshot_path_is_fine)
    _with_tmp_dir(test_arbitrary_path_never_triggers_anything)
    _with_tmp_dir(test_refuse_if_snapshot_in_local_mode_exits_1)
    _with_tmp_dir(test_refuse_if_snapshot_in_local_mode_dry_run_does_not_exit)
    _with_tmp_dir(test_sync_hint_fires_only_for_the_local_path)
    _with_tmp_dir(test_sync_hint_silent_even_with_local_mode_defaulted_true)
    _with_tmp_dir(test_forensics_backup_dir_none_when_not_local_disk)
    _with_tmp_dir(test_forensics_backup_dir_relocates_when_local_disk)
    test_live_db_path_matches_detect_local_mode()

    if _FAILS:
        print(f"\n{len(_FAILS)} FAILURE(S): {_FAILS}")
        sys.exit(1)
    print("\nAll _webui_live_path.py tests passed.")
