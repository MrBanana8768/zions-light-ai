"""
CPU-only tests for the v3.1 log-level demotions (REMEDIATION.md changes 4
and 5).

These assert a level, which is an odd thing to assert until you look at what
the log-sweep found: the compactor's operational log was loud enough on a
healthy pod that nobody read it, and the token-counter fallback ran unnoticed
for months behind lines exactly like these. A WARNING that fires on every
clean boot, or on every write to a network volume that does not support
O_DIRECTORY fsync, is not a warning — it is noise that trains the operator to
skip the whole file. So the level IS the behaviour here, and a silent
promotion back to WARNING would undo the change with nothing to catch it.

Two demotions are covered:
  - memory.atomic_write_json's two Class B handlers (post-write directory
    fsync, orphan temp-file cleanup) log at DEBUG. Behaviour otherwise
    unchanged: the write still succeeds / the original exception still
    propagates.
  - main's lifespan announces the budget-margin reset at INFO.

Run: python test_log_levels.py
"""

import asyncio
import contextlib
import errno
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-log-levels-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "2000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main  # noqa: E402
import memory  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name: str):
    """Collect every record a logger emits, at DEBUG and above.

    The logger's own level is forced to DEBUG for the duration — otherwise a
    demoted line would simply not be emitted under the default level and the
    test could not tell "logged at DEBUG" from "not logged at all", which are
    very different outcomes for change 4.
    """
    lg = logging.getLogger(logger_name)
    handler = _Collector()
    prev_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)


def find(records, needle: str) -> logging.LogRecord | None:
    for r in records:
        if needle in r.getMessage():
            return r
    return None


def _wipe_storage():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


# ---------------------------------------------------------------------------
# Change 4a — the post-write directory fsync
# ---------------------------------------------------------------------------

def test_directory_fsync_failure_logs_at_debug():
    print("\n[test] a failed directory fsync logs at DEBUG and does not fail the write")
    _wipe_storage()
    target = Path(_TMP_ROOT) / "facts" / "fsync-debug.json"
    real_open = os.open
    dir_key = os.path.normcase(os.path.abspath(str(target.parent)))

    def _open(path, *a, **k):
        # Only the O_RDONLY open of the PARENT DIRECTORY fails. The temp file
        # that tempfile.mkstemp opens goes through untouched, so the write
        # itself genuinely succeeds and only the durability barrier is lost —
        # which is the case the DEBUG line is for.
        if os.path.normcase(os.path.abspath(str(path))) == dir_key:
            raise OSError(errno.EINVAL, "Invalid argument", str(path))
        return real_open(path, *a, **k)

    with capture("compactor.memory") as cap:
        with patch("os.open", _open):
            memory.atomic_write_json(target, {"facts": [{"text": "survived"}]})

    rec = find(cap.records, "directory fsync skipped")
    assert_true(rec is not None, "the skipped-fsync line was logged at all")
    assert_eq(rec.levelname, "DEBUG", "logged at DEBUG, not WARNING")
    assert_true(not [r for r in cap.records if r.levelno >= logging.WARNING],
                "nothing at WARNING or above for a write that succeeded")
    # Behaviour unchanged: this is a durability barrier, not the write.
    assert_true(target.is_file(), "the file was still written")
    assert_eq(json.loads(target.read_text(encoding="utf-8"))["facts"][0]["text"],
              "survived", "and its contents are correct")


# ---------------------------------------------------------------------------
# Change 4b — orphan temp-file cleanup
# ---------------------------------------------------------------------------

def test_orphan_temp_cleanup_failure_logs_at_debug():
    print("\n[test] a failed orphan-temp cleanup logs at DEBUG and re-raises the original")
    _wipe_storage()
    target = Path(_TMP_ROOT) / "facts" / "orphan-debug.json"

    def _boom_unlink(path, *a, **k):
        raise OSError(errno.EACCES, "Permission denied", str(path))

    # json.dump fails first (the real error), then the cleanup fails too. The
    # cleanup's own failure must not shadow the original exception.
    with capture("compactor.memory") as cap:
        with patch("json.dump", side_effect=ValueError("not serializable")):
            with patch("os.unlink", _boom_unlink):
                try:
                    memory.atomic_write_json(target, {"facts": []})
                except ValueError as e:
                    raised = e
                except Exception as e:
                    print(f"FAIL the original exception was shadowed by "
                          f"{type(e).__name__}: {e}")
                    sys.exit(1)
                else:
                    print("FAIL atomic_write_json did not raise at all")
                    sys.exit(1)

    assert_eq(str(raised), "not serializable",
              "the original write error propagates, not the cleanup's")
    rec = find(cap.records, "orphan temp file left behind")
    assert_true(rec is not None, "the orphan-temp line was logged at all")
    assert_eq(rec.levelname, "DEBUG", "logged at DEBUG, not WARNING")


def test_healthy_write_says_nothing_to_the_operator():
    print("\n[test] a healthy atomic write emits nothing at INFO or above")
    # The other half of a demotion: the path that runs thousands of times a day
    # must not put a line in front of the operator.
    #
    # Asserted at INFO-and-above rather than "no records at all", because the
    # directory-fsync line legitimately fires on some platforms for every
    # write. On Windows it always does: os.open(dir, O_RDONLY) is EACCES there,
    # a directory is not openable as a file. That is the same shape as the
    # MooseFS case the DEBUG demotion was made for, and it is a fair
    # demonstration of why: at WARNING this line would fire on every single
    # write on such a host.
    _wipe_storage()
    target = Path(_TMP_ROOT) / "facts" / "quiet.json"
    with capture("compactor.memory") as cap:
        memory.atomic_write_json(target, {"facts": []})
    loud = [f"{r.levelname}: {r.getMessage()}"
            for r in cap.records if r.levelno >= logging.INFO]
    if loud:
        print("FAIL a healthy write emitted operator-visible lines:")
        for line in loud:
            print(f"       {line}")
        sys.exit(1)
    print("  ok   nothing at INFO or above for a healthy write")
    assert_true(target.is_file(), "and the write happened (fixture sanity)")


# ---------------------------------------------------------------------------
# Change 5 — the boot budget-margin announcement
# ---------------------------------------------------------------------------

def test_budget_margin_boot_line_is_info():
    print("\n[test] the boot budget-margin line is INFO, not WARNING")

    async def boot():
        async with main.lifespan(main.app):
            pass

    with capture("compactor") as cap:
        asyncio.run(boot())

    rec = find(cap.records, "context calibration starts at")
    assert_true(rec is not None, "the calibration line is still announced")
    assert_eq(rec.levelname, "INFO", "logged at INFO, not WARNING")
    assert_true("budget margin" in rec.getMessage(),
                "it still says what does not survive the restart")


def test_clean_boot_emits_no_warnings():
    print("\n[test] a clean boot produces no WARNING-or-above lines")
    # This is the actual reason for the demotion: a warning present on every
    # healthy boot is one the operator learns to scroll past, and that habit
    # is what let the token-counter fallback run unnoticed. Asserting the
    # level of one line does not protect that property; asserting the whole
    # clean boot is quiet does.
    async def boot():
        async with main.lifespan(main.app):
            pass

    with capture("compactor") as cap:
        asyncio.run(boot())

    noisy = [f"{r.levelname}: {r.getMessage()}"
             for r in cap.records if r.levelno >= logging.WARNING]
    if noisy:
        print("FAIL clean boot emitted WARNING-or-above:")
        for line in noisy:
            print(f"       {line}")
        sys.exit(1)
    print("  ok   nothing at WARNING or above on a clean boot")
    assert_true(len(cap.records) > 0, "the boot did log something (fixture sanity)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_directory_fsync_failure_logs_at_debug,
        test_orphan_temp_cleanup_failure_logs_at_debug,
        test_healthy_write_says_nothing_to_the_operator,
        test_budget_margin_boot_line_is_info,
        test_clean_boot_emits_no_warnings,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll log-level tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
