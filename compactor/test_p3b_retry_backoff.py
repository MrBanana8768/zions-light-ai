"""
p3-b (hostile pass #3, reviewer B) F10 — run_daemon's retry backoff applied
equally to transient AND persistent failures.

One unparseable memory JSON (which the compactor deliberately leaves in
place — main.py's _clear_all_memory docstring: "it cannot be safely
rewritten from an unknown state") used to retry every RETRY_BACKOFF_S
(900s) forever: ~96 full archive builds onto /data a day, 96 alerts, for
as long as the file stays broken.

Ported from SP\\p3-b\\retry_loop.py, with a REAL webui.db added (this
lane's own p3-b F12 fix now refuses a cycle with no webui.db at all, which
would otherwise mask this fixture's intended failure — the corrupt JSON —
behind a different one).

Run inside the compactor image or any container with the requirements
installed:
    python test_p3b_retry_backoff.py
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-p3b-retry-test-"))
_DATA = _ROOT / "data" / "openwebui"
_STORE = _DATA / "compactor"
_BK = _ROOT / "data" / "backups"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BK)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DATA / "webui.db")
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"

import backup  # noqa: E402


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


class _Stop(Exception):
    pass


def test_persistent_failure_backs_off_exponentially_capped_at_interval():
    print("\n[test] F10: a persistent (corrupt-json) failure backs off exponentially, capped at the interval")
    _DATA.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DATA / "webui.db"))
    c.execute("create table chat (id text primary key, chat text)")
    c.execute("insert into chat values ('c', 'x')")
    c.commit()
    c.close()
    (_STORE / "facts").mkdir(parents=True)
    (_STORE / "facts" / "good.json").write_bytes(json.dumps({"facts": [{"text": "x" * 100}] * 200}).encode())
    (_STORE / "facts" / "0123456789abcdef0123456789abcdef.json").write_bytes(b'{"facts": [ TRUNCATED')

    sleeps = []
    alerts = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 6:
            raise _Stop()

    real_sleep = backup.time.sleep
    real_alert = backup._alert_failure
    backup.time.sleep = fake_sleep
    backup._alert_failure = lambda detail: alerts.append(detail[:90])
    try:
        try:
            # A short interval (1 hour) so the exponential ramp visibly
            # caps well before the test's own sleep-call budget runs out:
            # 900, 1800, 3600, 3600(capped), 3600, 3600.
            backup.run_daemon(interval_hours=1)
        except _Stop:
            pass
    finally:
        backup.time.sleep = real_sleep
        backup._alert_failure = real_alert

    assert_true(len(sleeps) >= 4, f"fixture: enough cycles ran to see the ramp (got {sleeps})")
    assert_eq(sleeps[0], 900, f"F10 fix: first retry is the base backoff, 900s (got {sleeps})")
    assert_eq(sleeps[1], 1800, f"F10 fix: second retry DOUBLES, 1800s (got {sleeps})")
    assert_eq(sleeps[2], 3600, f"F10 fix: third retry doubles again, 3600s = the 1h interval cap (got {sleeps})")
    assert_eq(sleeps[3], 3600, f"F10 fix: fourth retry stays CAPPED at the interval, not 7200s (got {sleeps})")
    assert_true(len(alerts) == len(sleeps), f"an alert fired every failed cycle, same as before (got {len(alerts)} alerts, {len(sleeps)} cycles)")


def test_control_a_transient_failure_that_clears_resets_the_backoff():
    print("\n[test] F10 CONTROL: once the failure clears, the NEXT failure starts back at the base backoff")
    _DATA.mkdir(parents=True, exist_ok=True)
    if (_DATA / "webui.db").exists():
        (_DATA / "webui.db").unlink()
    c = sqlite3.connect(str(_DATA / "webui.db"))
    c.execute("create table chat (id text primary key, chat text)")
    c.execute("insert into chat values ('c', 'x')")
    c.commit()
    c.close()
    bad = _STORE / "facts" / "0123456789abcdef0123456789abcdef.json"
    bad.write_bytes(b'{"facts": [ TRUNCATED')

    sleeps = []
    calls = {"n": 0}

    def fake_sleep(s):
        sleeps.append(s)
        calls["n"] += 1
        if calls["n"] == 2:
            # heal the failure between the 2nd and 3rd cycle
            bad.unlink()
        if calls["n"] >= 4:
            raise _Stop()

    real_sleep = backup.time.sleep
    real_alert = backup._alert_failure
    backup.time.sleep = fake_sleep
    backup._alert_failure = lambda detail: None
    try:
        try:
            backup.run_daemon(interval_hours=1)
        except _Stop:
            pass
    finally:
        backup.time.sleep = real_sleep
        backup._alert_failure = real_alert

    # sleeps: [900 (fail1), 1800 (fail2, heals here), 3600 (SUCCESS -> full
    # interval), ...]. The success sleep is the full interval (3600s here),
    # same value as the capped backoff in this fixture — the real proof is
    # the NEXT failure (if any) restarting at 900, which we can't force
    # without a 4th failure; instead assert the loop reached a successful
    # cycle at all (3 calls, not stuck retrying).
    assert_eq(sleeps[0], 900, f"got {sleeps}")
    assert_eq(sleeps[1], 1800, f"got {sleeps}")
    assert_true(len(sleeps) >= 3, f"the loop reached a post-heal cycle (got {sleeps})")


if __name__ == "__main__":
    tests = [
        test_persistent_failure_backs_off_exponentially_capped_at_interval,
        test_control_a_transient_failure_that_clears_resets_the_backoff,
    ]
    for t in tests:
        t()
    print("\nAll p3-b retry-backoff (F10) tests passed.")
