"""
CPU-only Tier-1 tests for compactor.logsetup (V2.3 Theme 4).

Verifies the text/JSON formatter switch, JSON line shape (incl. exception
attachment), that configure() is idempotent (no duplicate handlers), that the
handler writes to stdout (v3.1 P0-2), and the log_once gate (v3.1 P0-2b).

Run: python test_logsetup.py
"""

import io
import json
import logging
import os
import sys

import logsetup


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


def _capture(fmt: str, emit) -> str:
    """Configure with COMPACTOR_LOG_FORMAT=fmt, redirect the root handler to
    a buffer, run emit(logger), return captured text."""
    os.environ["COMPACTOR_LOG_FORMAT"] = fmt
    logsetup.configure()
    root = logging.getLogger()
    buf = io.StringIO()
    # Point the single handler installed by configure() at our buffer.
    root.handlers[0].stream = buf
    emit(logging.getLogger("compactor.test"))
    return buf.getvalue()


def test_text_format_default():
    print("\n[test] text format: plain line with level + message")
    out = _capture("text", lambda lg: lg.info("hello text"))
    assert_true("hello text" in out, "message present")
    assert_true("INFO" in out, "level present")
    assert_true(not out.strip().startswith("{"), "not JSON")


def test_json_format_is_valid_json():
    print("\n[test] json format: each line parses with expected fields")
    out = _capture("json", lambda lg: lg.warning("hello json"))
    line = out.strip().splitlines()[-1]
    rec = json.loads(line)
    assert_eq(rec["level"], "WARNING", "level field")
    assert_eq(rec["logger"], "compactor.test", "logger field")
    assert_eq(rec["message"], "hello json", "message field")
    assert_true("ts" in rec and "T" in rec["ts"], "iso timestamp present")


def test_json_includes_exception():
    print("\n[test] json format: exc_info attaches a traceback")

    def emit(lg):
        try:
            raise ValueError("boom")
        except ValueError:
            lg.exception("caught it")

    out = _capture("json", emit)
    rec = json.loads(out.strip().splitlines()[-1])
    assert_eq(rec["message"], "caught it", "message")
    assert_true("exc" in rec, "exc field present")
    assert_true("ValueError" in rec["exc"] and "boom" in rec["exc"], "traceback text")


def test_unknown_format_falls_back_to_text():
    print("\n[test] unknown COMPACTOR_LOG_FORMAT → text (not a crash)")
    out = _capture("yaml-lol", lambda lg: lg.info("fallback"))
    assert_true("fallback" in out, "message present")
    assert_true(not out.strip().startswith("{"), "fell back to text")


def test_configure_is_idempotent():
    print("\n[test] configure() doesn't stack duplicate handlers")
    os.environ["COMPACTOR_LOG_FORMAT"] = "text"
    logsetup.configure()
    logsetup.configure()
    logsetup.configure()
    assert_eq(len(logging.getLogger().handlers), 1, "exactly one handler after 3 calls")


def test_handler_writes_to_stdout():
    """v3.1 P0-2 / F17a. StreamHandler() defaults to stderr, which under
    supervisord is compactor-error.log — so every operational line landed in
    the file named 'error' while compactor.log, the one OPERATIONS.md:44
    tells the operator to read, held uvicorn access noise. This asserts the
    stream, not just the format, so the runbook stays true."""
    print("\n[test] configure() attaches the handler to stdout, not stderr")
    os.environ["COMPACTOR_LOG_FORMAT"] = "text"
    logsetup.configure()
    handler = logging.getLogger().handlers[0]
    assert_true(handler.stream is sys.stdout, "handler stream is sys.stdout")
    assert_true(handler.stream is not sys.stderr, "handler stream is NOT sys.stderr")


def test_log_once_gates_repeat_call_sites():
    print("\n[test] log_once: True once per key, False forever after")
    logsetup._reset_log_once_for_tests()
    assert_eq(logsetup.log_once("site.a"), True, "first call for a key -> True")
    assert_eq(logsetup.log_once("site.a"), False, "second call for same key -> False")
    assert_eq(logsetup.log_once("site.a"), False, "third call for same key -> False")
    assert_eq(logsetup.log_once("site.b"), True, "a different key is independent")


def test_log_once_survives_reconfigure():
    """configure() clears handlers; it must not clear the gate, or a second
    entry point calling configure() would re-open every throttled call site."""
    print("\n[test] log_once state is not reset by configure()")
    logsetup._reset_log_once_for_tests()
    assert_eq(logsetup.log_once("site.c"), True, "first call -> True")
    logsetup.configure()
    assert_eq(logsetup.log_once("site.c"), False, "still gated after configure()")


def _all():
    return [
        test_text_format_default,
        test_json_format_is_valid_json,
        test_json_includes_exception,
        test_unknown_format_falls_back_to_text,
        test_configure_is_idempotent,
        test_handler_writes_to_stdout,
        test_log_once_gates_repeat_call_sites,
        test_log_once_survives_reconfigure,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll logsetup smoke tests passed.")
