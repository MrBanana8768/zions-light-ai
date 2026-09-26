"""Unit tests for ops/zlops/commands/shadow.py -- review H1: a pidfile
alone is not enough to trust "this pid is still our shadow process" after
a pod restart (container overlay resets, PIDs restart from low numbers,
but /data -- and the pidfile on it -- survives). `status`/`down` must
verify pid + boot_id + process start time + cmdline all still agree before
trusting the record, and `down` must repeat `up`'s STORE-equality guard
immediately before `rmtree`.

Plain-script style like compactor/test_setup_sshd_script.py: exit 0 pass,
1 fail. No docker/network needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from zlops.commands import shadow  # noqa: E402


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


class _Fixture:
    """A real, long-lived child process (`sleep`) standing in for the
    shadow uvicorn, plus temp STORE/SHADOW_STORE and pidfile paths.
    """

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        td = Path(self._td.name)
        self.pidfile = td / "shadow.pid"
        self.store = td / "store"
        self.shadow_store = td / "shadow-store"
        self.checkout = td / "compactor-shadow"
        self.checkout.mkdir()
        self.store.mkdir()

        self._patches = [
            mock.patch.object(shadow, "_PIDFILE", self.pidfile),
            mock.patch.object(shadow, "_CHECKOUT", self.checkout),
            mock.patch.object(shadow.podenv, "STORE", self.store),
            mock.patch.object(shadow.podenv, "SHADOW_STORE", self.shadow_store),
        ]
        for p in self._patches:
            p.start()

        # A real process whose /proc/<pid>/cmdline genuinely contains the
        # SHADOW_PORT and checkout path, like the real uvicorn invocation.
        marker = f"--shadow-marker--port-{shadow.podenv.SHADOW_PORT}--{self.checkout}--"
        self.proc = subprocess.Popen(["sleep", "60"], env={**os.environ, "ZL_SHADOW_MARKER": marker})
        # cmdline won't literally contain the marker for `sleep 60`, so
        # patch _cmdline to behave like it would for the real invocation.
        self._cmdline_patch = mock.patch.object(
            shadow, "_cmdline",
            side_effect=lambda pid: (
                f"/opt/compactor-venv/bin/python -m uvicorn main:app --app-dir {self.checkout}/compactor "
                f"--port {shadow.podenv.SHADOW_PORT}"
                if pid == self.proc.pid else ""
            ),
        )
        self._cmdline_patch.start()
        return self

    def __exit__(self, *exc):
        self._cmdline_patch.stop()
        for p in reversed(self._patches):
            p.stop()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._td.cleanup()
        return False

    def write_record(self, **overrides) -> None:
        record = {
            "pid": self.proc.pid,
            "boot_id": shadow._boot_id(),
            "start_time": shadow._proc_start_time(self.proc.pid),
            "cmdline": shadow._cmdline(self.proc.pid),
        }
        record.update(overrides)
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        self.pidfile.write_text(json.dumps(record), encoding="utf-8")


def test_write_and_read_pid_record_round_trips():
    with _Fixture() as fx:
        shadow._write_pid_record(fx.proc.pid)
        record = shadow._read_pid_record()
        assert_eq(record["pid"], fx.proc.pid, "pid recorded")
        assert_true(record["boot_id"], "boot_id recorded (non-empty)")
        assert_true(record["start_time"], "start_time recorded (non-empty)")
        assert_true(str(shadow.podenv.SHADOW_PORT) in record["cmdline"], "cmdline recorded",
                    extra=record["cmdline"])


def test_matching_record_is_trusted():
    with _Fixture() as fx:
        fx.write_record()
        record = shadow._read_pid_record()
        assert_true(shadow._record_is_our_live_process(record), "a genuinely matching record is trusted")


def test_wrong_boot_id_is_never_trusted():
    """The core H1 scenario: a pod restart changes the boot id, while the
    pidfile (on /data) survives. A record whose boot_id no longer matches
    must never be trusted, even if its pid happens to be alive right now
    (holding an unrelated process after the restart).
    """
    with _Fixture() as fx:
        fx.write_record(boot_id="00000000-0000-0000-0000-000000000000")
        record = shadow._read_pid_record()
        assert_true(not shadow._record_is_our_live_process(record),
                    "a stale boot_id is never trusted, even with a live pid")


def test_wrong_start_time_is_never_trusted():
    """The pid got reused by an unrelated process (possible even within
    the SAME boot): start_time no longer matches.
    """
    with _Fixture() as fx:
        fx.write_record(start_time="1")
        record = shadow._read_pid_record()
        assert_true(not shadow._record_is_our_live_process(record),
                    "a mismatched start_time is never trusted")


def test_wrong_cmdline_is_never_trusted():
    """The identity check reads the pid's CURRENT /proc/<pid>/cmdline (not
    just whatever was last recorded -- a pid can be reused by something
    else without the pidfile ever being rewritten), so this mutates the
    LIVE cmdline the identity check actually reads.
    """
    with _Fixture() as fx:
        fx.write_record()
        record = shadow._read_pid_record()
        with mock.patch.object(shadow, "_cmdline", return_value="/usr/bin/some-unrelated-process --nothing"):
            assert_true(not shadow._record_is_our_live_process(record),
                        "a live cmdline that doesn't look like our own shadow uvicorn is never trusted")


def test_status_refuses_on_untrusted_record():
    with _Fixture() as fx:
        fx.write_record(boot_id="stale")
        code = shadow._status([])
        assert_eq(code, 1, "status refuses (exit 1) when the record cannot be verified")


def test_down_never_signals_an_untrusted_pid():
    """H1: down must not kill whatever now holds a reused/restart-crossed
    pid. Mutation-style proof: an untrusted record's pid (our own real
    child, standing in for "some unrelated process") must be left alone --
    still alive after `down`.

    Review LOW (round 2): the OLD behaviour also unconditionally removed
    the pidfile AND rmtree'd SHADOW_STORE even after refusing to signal --
    if that pid genuinely is still some live process (not actually our
    shadow, but still running), it could still be using that storage.
    `down` must now refuse outright (exit 1) instead, and leave BOTH the
    pidfile and SHADOW_STORE untouched.
    """
    with _Fixture() as fx:
        fx.write_record(boot_id="stale-from-before-a-restart")
        fx.shadow_store.mkdir(exist_ok=True)
        (fx.shadow_store / "sentinel.txt").write_text("might still be in use", encoding="utf-8")
        code = shadow._down([])
        assert_eq(code, 1, "down refuses (exit 1) when it could not verify the recorded pid")
        assert_true(fx.proc.poll() is None, "the (untrusted) process was NEVER signalled -- still alive",
                    extra=f"poll()={fx.proc.poll()}")
        assert_true(fx.pidfile.exists(), "the pidfile is left in place, not silently cleaned up")
        assert_true((fx.shadow_store / "sentinel.txt").exists(),
                     "SHADOW_STORE is NOT removed when down refused to signal an unverifiable pid")


def test_down_signals_a_trusted_pid():
    with _Fixture() as fx:
        fx.write_record()
        code = shadow._down([])
        assert_eq(code, 0, "down exits 0")
        for _ in range(20):
            if fx.proc.poll() is not None:
                break
            time.sleep(0.2)
        assert_true(fx.proc.poll() is not None, "a genuinely-verified process IS signalled and exits")


def test_down_repeats_the_store_equality_guard_before_rmtree():
    """H1: `down` must repeat the SAME STORE-equality guard `up` uses,
    immediately before rmtree -- not just trust a check made earlier.
    Mutation-style proof: point SHADOW_STORE at STORE itself (the unsafe
    configuration `up` would refuse outright) and confirm `down` refuses
    too, rather than deleting her live memory.
    """
    with _Fixture() as fx:
        with mock.patch.object(shadow.podenv, "SHADOW_STORE", fx.store):  # now == STORE: unsafe
            fx.store.mkdir(exist_ok=True)
            (fx.store / "sentinel.txt").write_text("her live memory", encoding="utf-8")
            code = shadow._down([])
            assert_eq(code, 1, "down refuses (exit 1) when SHADOW_STORE resolves to the live STORE")
            assert_true((fx.store / "sentinel.txt").exists(),
                        "the live STORE directory was NOT removed")


if __name__ == "__main__":
    test_write_and_read_pid_record_round_trips()
    test_matching_record_is_trusted()
    test_wrong_boot_id_is_never_trusted()
    test_wrong_start_time_is_never_trusted()
    test_wrong_cmdline_is_never_trusted()
    test_status_refuses_on_untrusted_record()
    test_down_never_signals_an_untrusted_pid()
    test_down_signals_a_trusted_pid()
    test_down_repeats_the_store_equality_guard_before_rmtree()

    print("\nAll ops/commands/shadow.py tests passed.")
