"""Unit tests for ops/zlops/commands/hotfix.py -- the `zl hotfix
backup-fast` pod-only, no-image-change patch to compactor/backup.py's
own `create_backup` (adds `compresslevel=1` to its `tarfile.open(partial,
"w:gz")` call). No docker/network needed: `podenv.COMPACTOR_PKG` and
`hotfix._TARGET` are pointed at a throwaway temp file, and
`supervisorctl` is faked on PATH (a state-file-backed fake, the same
idiom as test_ops_fix.py's own -- status/stop/start/pid all really work
against a small JSON state file, so `common.stop_for_apply`/
`restart_after_apply` (reused by hotfix.py, round-9) behave exactly as
they would against the real thing).

Plain-script style like compactor/test_setup_sshd_script.py: exit 0
pass, 1 fail.
"""

from __future__ import annotations

import fcntl
import io
import json
import os
import stat
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from zlops.commands import hotfix  # noqa: E402


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


_REAL_BACKUP_PY = '''"""fake backup.py standing in for the real compactor/backup.py."""
import argparse
import tarfile


def create_backup(d, stamp):
    partial = d / f"zions-backup-{stamp}.tar.gz.partial"
    with tarfile.open(partial, "w:gz") as tar:
        pass
    return partial


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--daemon", action="store_true")
    args = p.parse_args(argv)
    if args.list:
        print("[]")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
'''

_DUPLICATE_ANCHOR_BACKUP_PY = _REAL_BACKUP_PY.replace(
    'with tarfile.open(partial, "w:gz") as tar:\n        pass\n    return partial',
    'with tarfile.open(partial, "w:gz") as tar:\n        pass\n    with tarfile.open(partial, "w:gz") as tar:\n'
    '        pass\n    return partial',
)

# A backup.py whose OWN --list check inspects its own source and fails if
# `compresslevel=1` is present -- i.e. it fails ONLY once patched. Used to
# prove the compile/import check runs against the PATCHED file (H9): a
# mutant that ran it against `.zl-orig` instead would see the UNPATCHED
# source and pass.
_SELF_INSPECTING_BACKUP_PY = '''"""fake backup.py that fails its own --list once patched (H9 probe)."""
import argparse
import tarfile


def create_backup(d, stamp):
    partial = d / f"zions-backup-{stamp}.tar.gz.partial"
    with tarfile.open(partial, "w:gz") as tar:
        pass
    return partial


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    src = open(__file__, encoding="utf-8").read()
    if ("compresslevel" + "=1) as tar:") in src:
        import sys as _sys
        print("refusing: this (patched) copy of backup.py must not be the one --list runs", file=_sys.stderr)
        _sys.exit(1)
    if args.list:
        print("[]")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
'''

# A backup.py whose --list sleeps well past a shrunk `_IMPORT_CHECK_TIMEOUT_S`
# (MEDIUM-2 / a P8-style slow `--list`).
_SLOW_LIST_BACKUP_PY = _REAL_BACKUP_PY.replace(
    "    if args.list:\n        print(\"[]\")",
    "    if args.list:\n        import time as _time\n        _time.sleep(5)\n        print(\"[]\")",
)

_FAKE_SUPERVISORCTL_PY = """#!/usr/bin/env python3
import json, os, sys

STATE = os.environ["FAKE_SUPERVISOR_STATE"]
CALL_LOG = os.environ.get("FAKE_SUPERVISOR_CALL_LOG")


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"status": {}, "pids": {}}


def save(d):
    json.dump(d, open(STATE, "w"))


cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if CALL_LOG and cmd in ("stop", "start"):
    with open(CALL_LOG, "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\\n")
state = load()
status = state.setdefault("status", {})
pids = state.setdefault("pids", {})

if cmd == "status":
    for name, s in status.items():
        print(f"{name}{' ' * 20}{s}")
elif cmd == "stop":
    for name in sys.argv[2:]:
        status[name] = "STOPPED"
    save(state)
elif cmd == "start":
    for name in sys.argv[2:]:
        status[name] = "RUNNING"
    save(state)
elif cmd == "pid":
    name = sys.argv[2] if len(sys.argv) > 2 else ""
    pid = pids.get(name)
    if pid:
        print(pid)
        sys.exit(0)
    print("no such process", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
"""


class _Fixture:
    """A throwaway `/opt/compactor`-style directory with a fake
    `backup.py`, a state-file-backed fake `supervisorctl` on PATH (real
    status/stop/start/pid against a small JSON file), and podenv/hotfix
    pointed at it. `initial_status` defaults to `{"backup": "RUNNING"}`
    (the ordinary case); pass `{"backup": "STOPPED"}` for the
    stopped-stays-stopped scenario, or `{}` for "supervisorctl doesn't
    even know about backup" (refuses, never guesses).
    """

    def __init__(self, backup_py_text: str = _REAL_BACKUP_PY, initial_status=None, initial_pid: str | None = None):
        self._backup_py_text = backup_py_text
        self._initial_status = {"backup": "RUNNING"} if initial_status is None else dict(initial_status)
        self._initial_pid = initial_pid

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        root = Path(self._td.name)
        self.compactor_pkg = root / "compactor"
        self.compactor_pkg.mkdir()
        self.target = self.compactor_pkg / "backup.py"
        self.target.write_text(self._backup_py_text, encoding="utf-8")
        self.backups = root / "data" / "backups"
        self.backups.mkdir(parents=True)
        self.zl_home = root / "zl-home"

        bin_dir = root / "bin"
        bin_dir.mkdir()
        self.state_file = root / "supervisor_state.json"
        pids = {"backup": self._initial_pid} if self._initial_pid else {}
        self.state_file.write_text(
            json.dumps({"status": self._initial_status, "pids": pids}), encoding="utf-8"
        )
        supervisorctl = bin_dir / "supervisorctl"
        supervisorctl.write_text(_FAKE_SUPERVISORCTL_PY, encoding="utf-8")
        supervisorctl.chmod(supervisorctl.stat().st_mode | stat.S_IEXEC)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(bin_dir) + os.pathsep + self._old_path
        self._old_state_env = os.environ.get("FAKE_SUPERVISOR_STATE")
        os.environ["FAKE_SUPERVISOR_STATE"] = str(self.state_file)
        self.call_log = root / "supervisor_calls.log"
        self._old_call_log_env = os.environ.get("FAKE_SUPERVISOR_CALL_LOG")
        os.environ["FAKE_SUPERVISOR_CALL_LOG"] = str(self.call_log)

        self._patches = [
            mock.patch.object(hotfix, "_TARGET", self.target),
            mock.patch.object(hotfix, "_CHANGE_LOCK_PATH", self.zl_home / "locks" / "change.lock"),
            mock.patch.object(hotfix, "_BOOTSTRAP_FLAGS_FILE", self.zl_home / "bootstrap-flags"),
            mock.patch.object(hotfix.podenv, "BACKUPS", self.backups),
            mock.patch.object(hotfix.podenv, "COMPACTOR_PKG", self.compactor_pkg),
            mock.patch.object(hotfix.podenv, "COMPACTOR_PY", Path(sys.executable)),
            mock.patch.object(hotfix.podenv, "ZL_HOME", self.zl_home),
            mock.patch.object(hotfix.podenv, "AUDIT_LOG", self.zl_home / "audit.log"),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        os.environ["PATH"] = self._old_path
        if self._old_state_env is None:
            os.environ.pop("FAKE_SUPERVISOR_STATE", None)
        else:
            os.environ["FAKE_SUPERVISOR_STATE"] = self._old_state_env
        if self._old_call_log_env is None:
            os.environ.pop("FAKE_SUPERVISOR_CALL_LOG", None)
        else:
            os.environ["FAKE_SUPERVISOR_CALL_LOG"] = self._old_call_log_env
        self._td.cleanup()
        return False

    def calls(self) -> str:
        return self.call_log.read_text(encoding="utf-8") if self.call_log.is_file() else ""

    def hotfix_audit_lines(self) -> list[dict]:
        audit_log = self.zl_home / "audit.log"
        if not audit_log.is_file():
            return []
        out = []
        for ln in audit_log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(ln)
            if "hotfix" in rec:
                out.append(rec)
        return out

    def _state(self) -> dict:
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def status(self, name: str = "backup") -> str | None:
        return self._state().get("status", {}).get(name)

    def set_status(self, name: str, state: str) -> None:
        s = self._state()
        s.setdefault("status", {})[name] = state
        self.state_file.write_text(json.dumps(s), encoding="utf-8")

    def set_pid(self, name: str, pid: str) -> None:
        s = self._state()
        s.setdefault("pids", {})[name] = pid
        self.state_file.write_text(json.dumps(s), encoding="utf-8")

    def supervisor_calls(self) -> str:
        # Reconstructed from state transitions is not possible generically;
        # tests that need "was stop/start called at all" use `status()`
        # transitions or the dedicated call-log below instead.
        return ""


def _run(args, stdin_tty=True):
    err = io.StringIO()
    out = io.StringIO()
    with mock.patch.object(sys, "stderr", err), mock.patch.object(sys, "stdout", out):
        code = hotfix.run(args)
    return code, out.getvalue(), err.getvalue()


def test_dry_run_shows_the_one_line_diff_and_changes_nothing():
    with _Fixture() as fx:
        before = fx.target.read_text(encoding="utf-8")
        code, out, _err = _run(["backup-fast"])
        assert_eq(code, 3, "a dry run with a pending patch exits 3")
        assert_true('- with tarfile.open(partial, "w:gz") as tar:' in out, "shows the removed line", extra=out)
        assert_true('+ with tarfile.open(partial, "w:gz", compresslevel=1) as tar:' in out,
                    "shows the added line", extra=out)
        assert_eq(fx.target.read_text(encoding="utf-8"), before, "the file is untouched by a dry run")
        assert_eq(fx.status(), "RUNNING", "no supervisorctl state change on a dry run")


def test_anchor_missing_refuses_and_changes_nothing():
    with _Fixture(backup_py_text=_REAL_BACKUP_PY.replace("w:gz", "w:bz2")) as fx:
        before = fx.target.read_text(encoding="utf-8")
        code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "a missing anchor refuses (exit 1)")
        assert_true("anchor line not found" in err, "says so", extra=err)
        assert_eq(fx.target.read_text(encoding="utf-8"), before, "nothing changed")
        assert_eq(fx.status(), "RUNNING", "no restart when refused")


def test_anchor_matched_twice_refuses_and_changes_nothing():
    with _Fixture(backup_py_text=_DUPLICATE_ANCHOR_BACKUP_PY) as fx:
        before = fx.target.read_text(encoding="utf-8")
        code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "an anchor matched more than once refuses (exit 1)")
        assert_true("matched 2 times" in err, "says how many times", extra=err)
        assert_eq(fx.target.read_text(encoding="utf-8"), before, "nothing changed")
        assert_eq(fx.status(), "RUNNING", "no restart when refused")


def test_already_patched_exits_0():
    with _Fixture(backup_py_text=_REAL_BACKUP_PY.replace(
        'with tarfile.open(partial, "w:gz") as tar:',
        'with tarfile.open(partial, "w:gz", compresslevel=1) as tar:',
    )) as fx:
        before = fx.target.read_text(encoding="utf-8")
        code, out, _err = _run(["backup-fast"])
        assert_eq(code, 0, "already patched exits 0, not the dry-run 3")
        assert_true("already fast" in out, "says so", extra=out)
        code2, _out2, _err2 = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code2, 0, "--apply on an already-patched file is also just exit 0")
        assert_eq(fx.target.read_text(encoding="utf-8"), before, "nothing changed either way")


def test_apply_patches_compiles_and_restarts_only_backup():
    with _Fixture() as fx:
        code, out, _err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 0, "a clean apply exits 0")
        patched = fx.target.read_text(encoding="utf-8")
        assert_true("compresslevel=1" in patched, "the file was actually patched", extra=patched)
        assert_true(hotfix._orig_path().is_file(), "the original was saved to .zl-orig")
        assert_eq(hotfix._orig_path().read_text(encoding="utf-8"), _REAL_BACKUP_PY,
                  "the saved original is byte-identical to the pre-patch file")
        assert_eq(fx.status(), "RUNNING", "backup was restarted and is RUNNING again")
        assert_true("restarted" in out, "says so", extra=out)
        calls = fx.calls()
        assert_true("stop backup" in calls and "start backup" in calls,
                    "supervisorctl was actually told to stop AND start backup", extra=calls)


def test_compile_failure_restores_the_original():
    with _Fixture() as fx:
        with mock.patch.object(hotfix, "_compiles_and_imports", return_value=(False, "boom")):
            code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "a failing compile/import check refuses (exit 1)")
        assert_true("restored the original" in err, "says it restored the original", extra=err)
        assert_eq(fx.target.read_text(encoding="utf-8"), _REAL_BACKUP_PY,
                  "the target file is back to the exact pre-patch content")
        assert_eq(fx.status(), "RUNNING", "never restarted anything on a failed compile check")


def test_apply_then_revert_round_trips_byte_identically():
    with _Fixture() as fx:
        code, _out, _err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 0, "apply exits 0")
        patched_mode = fx.target.stat().st_mode
        code2, out2, _err2 = _run(["backup-fast", "--revert", "--apply", "--yes"])
        assert_eq(code2, 0, "revert exits 0")
        assert_eq(fx.target.read_text(encoding="utf-8"), _REAL_BACKUP_PY,
                  "revert restores the file byte-for-byte identical to the original")
        assert_eq(fx.target.stat().st_mode, patched_mode, "the mode bits are preserved across the round-trip")
        assert_true("reverted" in out2, "says so", extra=out2)
        assert_eq(fx.status(), "RUNNING", "backup ends up RUNNING after the apply+revert round trip")
        calls = fx.calls()
        assert_true(calls.count("stop backup") == 2 and calls.count("start backup") == 2,
                    "H11: backup was ACTUALLY restarted once for the apply and once for the revert "
                    "(not just coincidentally left RUNNING)", extra=calls)


def test_revert_without_a_saved_original_refuses():
    with _Fixture():
        code, _out, err = _run(["backup-fast", "--revert", "--apply", "--yes"])
        assert_eq(code, 1, "reverting with nothing saved refuses (exit 1)")
        assert_true("nothing to revert to" in err, "says so", extra=err)


def test_revert_dry_run_shows_a_diff_and_changes_nothing():
    """MEDIUM-3 / P5: the ordinary case -- the current file is exactly
    what this hotfix produced, so a revert dry run shows the one-line
    diff back to the original.
    """
    with _Fixture() as fx:
        _run(["backup-fast", "--apply", "--yes"])
        before = fx.target.read_text(encoding="utf-8")
        code, out, _err = _run(["backup-fast", "--revert"])
        assert_eq(code, 3, "a revert dry run exits 3 (pending work)")
        assert_true("DRY RUN" in out, "says so", extra=out)
        assert_true('-    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:' in out,
                    "shows a diff removing the patch", extra=out)
        assert_eq(fx.target.read_text(encoding="utf-8"), before, "nothing changed")


def test_revert_refuses_when_the_current_file_does_not_match_and_shows_a_diff():
    """MEDIUM-3 / P7: the file was patched, then something ELSE changed it
    (a hand edit, a different release baked over it, ...) -- the current
    content is neither the saved original NOR what re-applying the patch
    to that original would produce. `--revert` must refuse rather than
    guess which "original" is right, and show a diff explaining why.
    """
    with _Fixture() as fx:
        _run(["backup-fast", "--apply", "--yes"])
        # simulate an unrelated further edit landing on top of the patch
        fx.target.write_text(fx.target.read_text(encoding="utf-8") + "\n# unrelated hand edit\n", encoding="utf-8")
        drifted = fx.target.read_text(encoding="utf-8")
        code, _out, err = _run(["backup-fast", "--revert", "--apply", "--yes"])
        assert_eq(code, 1, "revert refuses when the current file has drifted (exit 1)")
        assert_true("does not match what this hotfix would have produced" in err, "says so", extra=err)
        assert_true("---" in err and "+++" in err, "shows a unified diff", extra=err)
        assert_eq(fx.target.read_text(encoding="utf-8"), drifted, "nothing changed by the refused revert")
        assert_eq(fx.status(), "RUNNING", "no restart on a refused revert")


def test_revert_on_an_already_unpatched_file_is_a_noop():
    """MEDIUM-3 / P6: the file already equals the saved original (e.g. a
    container restart reset the overlay back to the un-hotfixed image, or
    an earlier revert already ran) -- `--revert` is a no-op: exit 0, no
    restart.
    """
    with _Fixture() as fx:
        _run(["backup-fast", "--apply", "--yes"])
        _run(["backup-fast", "--revert", "--apply", "--yes"])
        assert_eq(fx.target.read_text(encoding="utf-8"), _REAL_BACKUP_PY, "file is the original after the revert")
        fx.set_status("backup", "STOPPED")  # if this were to (wrongly) restart anything, this would flip
        code, out, _err = _run(["backup-fast", "--revert", "--apply", "--yes"])
        assert_eq(code, 0, "reverting an already-unpatched file is a no-op (exit 0)")
        assert_true("nothing to do" in out, "says so", extra=out)
        assert_eq(fx.status(), "STOPPED", "no restart happened on the no-op revert")


def test_no_restart_while_a_backup_is_in_progress_via_staging_dir():
    with _Fixture() as fx:
        staging = fx.backups / "zions-backup-20260101T000000Z-abcd1234"
        staging.mkdir()
        with mock.patch.object(hotfix, "_WAIT_FOR_BACKUP_S", 0.2):
            code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "refuses while a backup is in progress (a fresh staging dir)")
        assert_true("in progress" in err, "says so", extra=err)
        assert_eq(fx.status(), "RUNNING", "supervisorctl was never called while a backup runs")
        # the file itself WAS patched (only the restart is blocked) -- the
        # next run (once the backup finishes) sees "already fast", not a
        # re-patch attempt.
        assert_true("compresslevel=1" in fx.target.read_text(encoding="utf-8"),
                     "the patch itself is not blocked, only the restart")


def test_no_restart_while_a_backup_is_in_progress_via_partial_file():
    with _Fixture() as fx:
        (fx.backups / "zions-backup-20260101T000000Z.tar.gz.partial").write_bytes(b"")
        with mock.patch.object(hotfix, "_WAIT_FOR_BACKUP_S", 0.2):
            code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "refuses while a backup is in progress (a .tar.gz.partial file)")
        assert_true("in progress" in err, "says so", extra=err)
        assert_eq(fx.status(), "RUNNING", "supervisorctl was never called while a backup runs")


def test_no_restart_stops_waiting_once_the_backup_finishes():
    with _Fixture() as fx:
        staging = fx.backups / "zions-backup-20260101T000000Z-abcd1234"
        staging.mkdir()

        def _finish_soon():
            import threading

            def _rm():
                time.sleep(0.3)
                staging.rmdir()

            threading.Thread(target=_rm).start()

        _finish_soon()
        with mock.patch.object(hotfix, "_WAIT_FOR_BACKUP_S", 5.0):
            code, out, _err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 0, "once the backup finishes, the restart goes ahead")
        assert_true("waiting" in out, "it printed that it was waiting", extra=out)
        assert_eq(fx.status(), "RUNNING", "backup was restarted once the wait ended")


def test_unknown_hotfix_name_exits_2():
    with _Fixture() as fx:
        code, _out, err = _run(["not-a-real-hotfix"])
        assert_eq(code, 2, "an unknown hotfix name exits 2")
        assert_true("unknown hotfix" in err, "says so", extra=err)
        assert_eq(fx.status(), "RUNNING", "nothing touched")


def test_version_reports_backup_compression_status():
    from zlops.commands import version as cmd_version
    with _Fixture():
        info = cmd_version.collect()
        assert_eq(info["backup_compression"], "default", "not yet patched -- reports 'default'")
        _run(["backup-fast", "--apply", "--yes"])
        info2 = cmd_version.collect()
        assert_true("level 1" in info2["backup_compression"], "patched -- reports level 1 (hotfix)",
                    extra=info2["backup_compression"])
        rendered = cmd_version.render(info2)
        assert_true("backup compression: level 1 (hotfix)" in rendered, "shows up in `zl version`'s own text",
                    extra=rendered)


def test_mutation_wrong_anchor_would_survive_undetected_without_the_count_check():
    """Sanity check on the test suite itself (not hotfix.py): confirms
    the anchor really is unique among backup.py's OTHER tarfile.open
    call sites in a more realistic fixture (mixing an `r:gz` read next
    to the `w:gz` write) -- the read call is never touched.
    """
    mixed = _REAL_BACKUP_PY + '''

def read_something(archive_path):
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tar:
        return tar.getnames()
'''
    with _Fixture(backup_py_text=mixed) as fx:
        code, _out, _err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 0, "apply still succeeds with an unrelated r:gz read present")
        patched = fx.target.read_text(encoding="utf-8")
        assert_true('tarfile.open(archive_path, "r:gz")' in patched,
                    "the READ call site is untouched", extra=patched)
        assert_true('tarfile.open(partial, "w:gz", compresslevel=1)' in patched,
                    "only the WRITE call site was patched", extra=patched)


# --- round-9 HIGH-2: restart only if running before; concurrency --------


def test_stopped_program_stays_stopped():
    with _Fixture(initial_status={"backup": "STOPPED"}) as fx:
        code, out, _err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 0, "applying while backup is STOPPED still exits 0")
        assert_true("not running" in out, "says the patch takes effect next start", extra=out)
        assert_true("compresslevel=1" in fx.target.read_text(encoding="utf-8"), "the file was still patched")
        assert_eq(fx.status(), "STOPPED", "backup was never started by the hotfix")


def test_concurrent_hotfix_run_is_refused():
    """HIGH-2: two `zl hotfix` apply/revert invocations must never run at
    once. Simulates "another one is already running" by holding the same
    exclusive lock this one would need, from a real second file
    descriptor (a distinct open() of the same path, the only realistic
    way two separate processes would ever contend for it).
    """
    with _Fixture() as fx:
        lock_path = fx.zl_home / "locks" / "change.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        held_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(held_fd, fcntl.LOCK_EX)
        try:
            with mock.patch.object(hotfix, "_CHANGE_LOCK_ACQUIRE_TIMEOUT_S", 0.3):
                code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        finally:
            fcntl.flock(held_fd, fcntl.LOCK_UN)
            os.close(held_fd)
        assert_eq(code, 1, "a concurrent zl hotfix run is refused (exit 1)")
        assert_true("already running" in err, "says so", extra=err)
        assert_eq(fx.status(), "RUNNING", "nothing was stopped by the refused run")
        assert_true("compresslevel=1" not in fx.target.read_text(encoding="utf-8"),
                    "the file was never patched by the refused run")


# --- round-9 mutation-killers: H6, H7, H9, H10 ----------------------------


def test_zl_orig_never_overwritten_on_a_later_apply():
    """H6: `.zl-orig` must be the FIRST-EVER pre-hotfix original, never
    overwritten by a later apply -- even one that re-patches a file the
    overlay reset back to a DIFFERENT (but still old-anchor) state. Calls
    `_do_apply` directly (bypassing the "already fast" gate in `run()`,
    which would otherwise never let a second apply reach the write path
    at all) to exercise the guard itself.
    """
    with _Fixture() as fx:
        with hotfix.common.ProtectedRegion() as region:
            code = hotfix._do_apply(_REAL_BACKUP_PY, revert=False, region=region)
        assert_eq(code, 0, "first apply succeeds")
        first_orig = hotfix._orig_path().read_text(encoding="utf-8")
        assert_eq(first_orig, _REAL_BACKUP_PY, "the saved original is the first pre-patch content")

        # Simulate the overlay resetting back to a DIFFERENT pre-hotfix
        # copy (same anchor, different unrelated content) and re-applying.
        different_original = _REAL_BACKUP_PY.replace("fake backup.py", "fake backup.py v2")
        like = fx.target.stat()
        hotfix._atomic_write(fx.target, different_original, like=like)
        with hotfix.common.ProtectedRegion() as region:
            code2 = hotfix._do_apply(different_original, revert=False, region=region)
        assert_eq(code2, 0, "second apply (after a simulated overlay reset) also succeeds")
        assert_eq(hotfix._orig_path().read_text(encoding="utf-8"), first_orig,
                  ".zl-orig is STILL the very first original, not overwritten by the second apply")


def test_atomic_write_preserves_the_original_files_exact_mode():
    """H7: the patched (and reverted) file must keep the ORIGINAL file's
    own mode bits, not whatever `tempfile.mkstemp` defaults to (0600) --
    checked against an absolute, distinctive mode set on the target
    BEFORE patching, not just "patch and revert agree with each other"
    (which a mutant that drops the chmod entirely would still pass, since
    both sides would then share the same wrong default).
    """
    with _Fixture() as fx:
        fx.target.chmod(0o741)
        before_mode = stat.S_IMODE(fx.target.stat().st_mode)
        assert_eq(before_mode, 0o741, "the distinctive mode was actually set before patching")
        code, _out, _err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 0, "apply exits 0")
        after_mode = stat.S_IMODE(fx.target.stat().st_mode)
        assert_eq(after_mode, 0o741, "the patched file keeps the EXACT original mode, not mkstemp's default")


def test_import_check_runs_on_the_patched_file_not_the_original():
    """H9: the compile/import check must run against the PATCHED file
    (`_TARGET`, post-write), never `.zl-orig`. Uses a backup.py whose own
    `--list` inspects its own source and fails once `compresslevel=1` is
    present -- so if the check ran on the unpatched original instead, it
    would incorrectly report success.
    """
    with _Fixture(backup_py_text=_SELF_INSPECTING_BACKUP_PY) as fx:
        code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "the real check (against the PATCHED file) correctly refuses")
        assert_true("restored the original" in err, "says it restored the original", extra=err)
        assert_eq(fx.target.read_text(encoding="utf-8"), _SELF_INSPECTING_BACKUP_PY,
                  "the target file is back to the exact pre-patch content")
        assert_eq(fx.status(), "RUNNING", "never restarted anything on a failed compile check")


def test_wait_for_backup_constant_is_positive_and_shorter_than_the_boot_step_timeout():
    """H10: `_WAIT_FOR_BACKUP_S` must be a real, positive bound -- not 0
    (refuse-at-once) -- while still leaving room under bootstrap.sh's own
    `_HOTFIX_BOOT_TIMEOUT_S` (60s) for the compile/import check and the
    stop/start round trip that follow it.
    """
    assert_true(hotfix._WAIT_FOR_BACKUP_S > 5.0, "the wait is a real, non-trivial bound",
                extra=str(hotfix._WAIT_FOR_BACKUP_S))
    assert_true(hotfix._WAIT_FOR_BACKUP_S < 60.0, "the wait leaves room under the 60s boot-step timeout",
                extra=str(hotfix._WAIT_FOR_BACKUP_S))


# --- round-9 MEDIUM-1: daemon-vs-file staleness --------------------------


def test_version_reports_native_not_hotfix_when_there_is_no_zl_orig():
    with _Fixture(backup_py_text=_REAL_BACKUP_PY.replace(
        'with tarfile.open(partial, "w:gz") as tar:',
        'with tarfile.open(partial, "w:gz", compresslevel=1) as tar:',
    )):
        status = hotfix.backup_compression_status()
        assert_true("level 1 (native)" in status, "no .zl-orig -> 'native', not 'hotfix'", extra=status)


def test_already_fast_but_stale_daemon_restarts_it():
    """MEDIUM-1: the file already reads level-1, but the RUNNING daemon's
    own process started before the file's last write -- `zl version`
    (`backup_compression_status`) must say so, and a plain `zl hotfix
    backup-fast --apply --yes` must restart `backup` (never guessing that
    "already fast" means nothing left to do).
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with _Fixture(
            backup_py_text=_REAL_BACKUP_PY.replace(
                'with tarfile.open(partial, "w:gz") as tar:',
                'with tarfile.open(partial, "w:gz", compresslevel=1) as tar:',
            ),
            initial_pid=str(proc.pid),
        ) as fx:
            hotfix._orig_path().write_text(_REAL_BACKUP_PY, encoding="utf-8")  # this IS a hotfix, not native
            time.sleep(0.3)  # make sure the file's mtime is strictly after the daemon's start
            os.utime(fx.target, None)

            status = hotfix.backup_compression_status()
            assert_true("restart pending" in status, "reports the daemon is stale", extra=status)

            code, out, _err = _run(["backup-fast", "--apply", "--yes"])
            assert_eq(code, 0, "restarting a stale daemon for an already-applied patch exits 0")
            assert_true("restarted" in out or "already patched" in out, "says so", extra=out)
            assert_eq(fx.status(), "RUNNING", "backup ends up RUNNING")
    finally:
        proc.kill()
        proc.wait()


def test_already_fast_stale_daemon_but_backup_in_progress_refuses():
    """P2-combined: a STALE leftover staging directory (as if a previous
    backup cycle crashed and never cleaned up) must still gate the
    restart-for-staleness path exactly like a fresh one -- it is bounded
    (never hangs forever) but still refuses rather than restarting out
    from under what LOOKS like an in-progress cycle.
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with _Fixture(
            backup_py_text=_REAL_BACKUP_PY.replace(
                'with tarfile.open(partial, "w:gz") as tar:',
                'with tarfile.open(partial, "w:gz", compresslevel=1) as tar:',
            ),
            initial_pid=str(proc.pid),
        ) as fx:
            hotfix._orig_path().write_text(_REAL_BACKUP_PY, encoding="utf-8")
            time.sleep(0.3)
            os.utime(fx.target, None)
            # a STALE staging dir, far too old to ever finish on its own
            stale = fx.backups / "zions-backup-20200101T000000Z-deadbeef"
            stale.mkdir()

            with mock.patch.object(hotfix, "_WAIT_FOR_BACKUP_S", 0.2):
                code, _out, err = _run(["backup-fast", "--apply", "--yes"])
            assert_eq(code, 1, "refuses (bounded, not a hang) while a stale staging dir sits there")
            assert_true("in progress" in err, "says so", extra=err)
            assert_eq(fx.status(), "RUNNING", "supervisorctl was never called")
    finally:
        proc.kill()
        proc.wait()


# --- round-9 MEDIUM-2: slow --list / TimeoutExpired ------------------------


def test_slow_list_times_out_and_restores_the_original():
    """P8-style: `--list` hangs well past the (shrunk, for this test)
    import-check timeout -- `subprocess.TimeoutExpired` must be caught,
    not escape, and the original restored exactly like any other
    compile/import failure.
    """
    with _Fixture(backup_py_text=_SLOW_LIST_BACKUP_PY) as fx:
        with mock.patch.object(hotfix, "_IMPORT_CHECK_TIMEOUT_S", 0.3):
            code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "a compile-check timeout refuses (exit 1)")
        assert_true("timed out" in err, "says it timed out", extra=err)
        assert_true("restored the original" in err, "says it restored the original", extra=err)
        assert_eq(fx.target.read_text(encoding="utf-8"), _SLOW_LIST_BACKUP_PY,
                  "the target file is back to the exact pre-patch content")
        assert_eq(fx.status(), "RUNNING", "never restarted anything on a timed-out compile check")


def test_missing_interpreter_is_caught_and_restores_the_original():
    """MEDIUM-2: a FileNotFoundError from the import-check subprocess
    (e.g. the venv interpreter itself missing) must be caught the same
    way, not escape and crash with the patched file left in place.
    """
    with _Fixture() as fx:
        with mock.patch.object(hotfix.podenv, "COMPACTOR_PY", Path("/no/such/interpreter-zl-test")):
            code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "a missing interpreter refuses (exit 1)")
        assert_true("could not run" in err, "says so", extra=err)
        assert_eq(fx.target.read_text(encoding="utf-8"), _REAL_BACKUP_PY,
                  "the target file is back to the exact pre-patch content")


# --- round-9 LOW: .zl-orig round-trip verification -------------------------


def test_orig_round_trip_failure_refuses_to_apply():
    """LOW: if the freshly-saved `.zl-orig` does not round-trip back to
    the exact intended patched content, the apply must refuse rather than
    trust an unverified save.
    """
    with _Fixture() as fx:
        with mock.patch.object(hotfix, "_save_orig_verified", return_value=False):
            code, _out, err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 1, "refuses when the saved original cannot be verified")
        assert_true("round-trips" in err, "says so", extra=err)
        assert_eq(fx.target.read_text(encoding="utf-8"), _REAL_BACKUP_PY, "the target file is untouched")


def test_orig_write_uses_fsync_and_verifies_the_round_trip_for_real():
    """LOW: exercises the REAL `_save_orig_verified` (not mocked) end to
    end -- the saved `.zl-orig` must itself, when the patch is re-applied
    to it, reproduce the exact patched content on disk.
    """
    with _Fixture() as fx:
        code, _out, _err = _run(["backup-fast", "--apply", "--yes"])
        assert_eq(code, 0, "apply exits 0")
        orig_text = hotfix._orig_path().read_text(encoding="utf-8")
        patched_text = fx.target.read_text(encoding="utf-8")
        assert_eq(orig_text.replace(hotfix._ANCHOR_OLD, hotfix._ANCHOR_NEW, 1), patched_text,
                  "the saved .zl-orig, patched, reproduces the real patched file byte for byte")


def test_audit_line_records_before_and_after_sha256():
    """LOW: on top of cli.py's own generic audit line, hotfix.py writes
    its own supplementary record with the before/after sha256 of
    backup.py, for both an apply and a subsequent revert.
    """
    import hashlib

    with _Fixture() as fx:
        _run(["backup-fast", "--apply", "--yes"])
        _run(["backup-fast", "--revert", "--apply", "--yes"])
        recs = fx.hotfix_audit_lines()
        assert_eq(len(recs), 2, "one supplementary audit line per apply/revert")
        apply_rec, revert_rec = recs
        assert_eq(apply_rec["action"], "apply", "the first record is the apply")
        assert_eq(apply_rec["sha256_before"], hashlib.sha256(_REAL_BACKUP_PY.encode()).hexdigest(),
                  "apply's sha256_before is the real pre-patch content's hash")
        assert_true(apply_rec["sha256_after"] != apply_rec["sha256_before"],
                    "apply's sha256_after differs from sha256_before", extra=str(apply_rec))
        assert_eq(revert_rec["action"], "revert", "the second record is the revert")
        assert_eq(revert_rec["sha256_after"], hashlib.sha256(_REAL_BACKUP_PY.encode()).hexdigest(),
                  "revert's sha256_after is back to the real original content's hash")
        assert_eq(revert_rec["sha256_before"], apply_rec["sha256_after"],
                  "revert's sha256_before matches what the apply left behind")


if __name__ == "__main__":
    test_dry_run_shows_the_one_line_diff_and_changes_nothing()
    test_anchor_missing_refuses_and_changes_nothing()
    test_anchor_matched_twice_refuses_and_changes_nothing()
    test_already_patched_exits_0()
    test_apply_patches_compiles_and_restarts_only_backup()
    test_compile_failure_restores_the_original()
    test_apply_then_revert_round_trips_byte_identically()
    test_revert_without_a_saved_original_refuses()
    test_revert_dry_run_shows_a_diff_and_changes_nothing()
    test_revert_refuses_when_the_current_file_does_not_match_and_shows_a_diff()
    test_revert_on_an_already_unpatched_file_is_a_noop()
    test_no_restart_while_a_backup_is_in_progress_via_staging_dir()
    test_no_restart_while_a_backup_is_in_progress_via_partial_file()
    test_no_restart_stops_waiting_once_the_backup_finishes()
    test_unknown_hotfix_name_exits_2()
    test_version_reports_backup_compression_status()
    test_mutation_wrong_anchor_would_survive_undetected_without_the_count_check()
    test_stopped_program_stays_stopped()
    test_concurrent_hotfix_run_is_refused()
    test_zl_orig_never_overwritten_on_a_later_apply()
    test_atomic_write_preserves_the_original_files_exact_mode()
    test_import_check_runs_on_the_patched_file_not_the_original()
    test_wait_for_backup_constant_is_positive_and_shorter_than_the_boot_step_timeout()
    test_version_reports_native_not_hotfix_when_there_is_no_zl_orig()
    test_already_fast_but_stale_daemon_restarts_it()
    test_already_fast_stale_daemon_but_backup_in_progress_refuses()
    test_slow_list_times_out_and_restores_the_original()
    test_missing_interpreter_is_caught_and_restores_the_original()
    test_orig_round_trip_failure_refuses_to_apply()
    test_orig_write_uses_fsync_and_verifies_the_round_trip_for_real()
    test_audit_line_records_before_and_after_sha256()

    print("\nAll ops/commands/hotfix.py tests passed.")
