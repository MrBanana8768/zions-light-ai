"""
CPU-only tests for compactor.tokenhealth, and for summarizer's wiring to it
(v3.1 remediation residual: summarizer's /tokenize failures used to report
through logsetup.log_once, which fires exactly once for the life of the
process and then never again — so a /tokenize outage starting after that
first line was invisible in the log AND absent from every health surface,
because nothing fed a counter /health/full could read).

This file proves three things the fix claims:
  1. tokenhealth counts a real streak/degraded-since, readable via
     source_health() / summarizer.tokenize_health() — the (a) half.
  2. A persistent degradation is NOT silenced forever: it speaks again once
     its rate-limit window rolls over, unlike log_once — the (b) half.
  3. Within one window, repeated failures still collapse to ONE line, so the
     fix does not trade "silent forever" for "a line per call".

Run: python test_tokenhealth.py
"""

import asyncio
import contextlib
import logging
import sys
import time

import logsetup
import summarizer
import tokenhealth


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


def _reset():
    tokenhealth._reset_for_tests()
    logsetup._reset_log_once_for_tests()


# ---------------------------------------------------------------------------
# Log capture (same shape as test_summarizer.py's — scoped to
# "compactor.summarizer", because that is the logger the fix is required to
# use: a warning logged under "compactor.tokenhealth" would never propagate
# there, since the two are siblings, not parent/child, in the logger
# hierarchy).
# ---------------------------------------------------------------------------

class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name: str = "compactor.summarizer"):
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


def find(records, needle: str):
    for r in records:
        if needle in r.getMessage():
            return r
    return None


# ---------------------------------------------------------------------------
# Fake httpx client — /tokenize always fails (or always succeeds), nothing
# else is called from _count_tokens directly.
# ---------------------------------------------------------------------------

class _FailingClient:
    def __init__(self, exc: Exception | None = None, status: int | None = None):
        self.exc = exc
        self.status = status
        self.calls = 0

    async def post(self, url, **kw):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return _Resp(self.status, {})


class _OkClient:
    def __init__(self, count: int = 42):
        self.count = count
        self.calls = 0

    async def post(self, url, **kw):
        self.calls += 1
        return _Resp(200, {"count": self.count})


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# tokenhealth unit tests
# ---------------------------------------------------------------------------

def test_fresh_source_is_healthy():
    print("\n[test] an unknown source reports healthy, not unknown-as-broken")
    _reset()
    h = tokenhealth.source_health("nobody-has-called-this-yet")
    assert_eq(h["ok"], True, "ok")
    assert_eq(h["consecutive_failures"], 0, "no failures")
    assert_eq(h["degraded_since"], None, "never degraded")


def test_note_failure_returns_a_message_on_first_call():
    print("\n[test] the first failure in a window returns a message to log")
    _reset()
    msg = tokenhealth.note_failure("src-a", "k1", "/tokenize unreachable (boom)")
    assert_true(msg is not None, "a message is due")
    assert_true("/tokenize unreachable (boom)" in msg, "caller's detail survives verbatim")
    assert_true("1 consecutive failure" in msg, "names the streak")
    h = tokenhealth.source_health("src-a")
    assert_eq(h["ok"], False, "now unhealthy")
    assert_eq(h["consecutive_failures"], 1, "streak of 1")


def test_note_failure_is_rate_limited_within_one_window():
    print("\n[test] repeated failures in the same window collapse to one line")
    _reset()
    first = tokenhealth.note_failure("src-b", "k1", "detail", warn_interval_s=300)
    second = tokenhealth.note_failure("src-b", "k1", "detail", warn_interval_s=300)
    third = tokenhealth.note_failure("src-b", "k1", "detail", warn_interval_s=300)
    assert_true(first is not None, "first call speaks")
    assert_eq(second, None, "second call in-window is silenced")
    assert_eq(third, None, "third call in-window is silenced")
    # But the streak keeps counting even while silenced — this is the whole
    # point: /health/full must see 3, not 1, even though only one line was
    # logged.
    assert_eq(tokenhealth.source_health("src-b")["consecutive_failures"], 3,
               "streak counts every failure, logged or not")


def test_note_failure_speaks_again_once_the_window_rolls_over():
    print("\n[test] NOT log_once: a persistent outage speaks again later")
    # This is the (b) half of the fix. log_once would return False forever
    # after the first call for this key, for the life of the process. A
    # tiny warn_interval_s and a real sleep crossing the bucket boundary
    # proves this mechanism is not that.
    _reset()
    interval = 0.05
    first = tokenhealth.note_failure("src-c", "k1", "detail", warn_interval_s=interval)
    immediate_repeat = tokenhealth.note_failure("src-c", "k1", "detail", warn_interval_s=interval)
    time.sleep(interval * 3)
    later = tokenhealth.note_failure("src-c", "k1", "detail", warn_interval_s=interval)
    assert_true(first is not None, "first call speaks")
    assert_eq(immediate_repeat, None, "immediate repeat is silenced")
    assert_true(later is not None, "a call after the window rolled over speaks again")
    assert_true("consecutive failure" in later, "and still reports the (now larger) streak")


def test_note_success_clears_and_reports_recovery_once():
    print("\n[test] note_success clears the streak and reports recovery once")
    _reset()
    tokenhealth.note_failure("src-d", "k1", "detail")
    tokenhealth.note_failure("src-d", "k1", "detail", warn_interval_s=0.001)
    recovery = tokenhealth.note_success("src-d")
    assert_true(recovery is not None, "a recovery message is produced")
    assert_true("answering again" in recovery, "names the recovery")
    h = tokenhealth.source_health("src-d")
    assert_eq(h["ok"], True, "healthy again")
    assert_eq(h["consecutive_failures"], 0, "streak cleared")
    again = tokenhealth.note_success("src-d")
    assert_eq(again, None, "calling success again on an already-healthy source is a no-op")


def test_source_health_staleness_for_not_every_request_sources():
    print("\n[test] stale_after_s: an old failure reads as unconfirmed, not ongoing")
    _reset()
    tokenhealth.note_failure("src-e", "k1", "detail")
    fresh = tokenhealth.source_health("src-e", stale_after_s=1000)
    assert_eq(fresh["ok"], False, "not stale yet -> still reported as failing")
    stale = tokenhealth.source_health("src-e", stale_after_s=0)
    assert_eq(stale["ok"], True, "older than the staleness window -> unconfirmed, reported healthy")
    assert_eq(stale["consecutive_failures"], 0, "and the count reads as 0")


def test_reset_for_tests_clears_named_and_all_sources():
    print("\n[test] _reset_for_tests clears one source, or everything")
    _reset()
    tokenhealth.note_failure("src-f", "k1", "detail")
    tokenhealth.note_failure("src-g", "k1", "detail")
    tokenhealth._reset_for_tests("src-f")
    assert_eq(tokenhealth.source_health("src-f")["consecutive_failures"], 0, "src-f cleared")
    assert_eq(tokenhealth.source_health("src-g")["consecutive_failures"], 1, "src-g untouched")
    tokenhealth._reset_for_tests()
    assert_eq(tokenhealth.source_health("src-g")["consecutive_failures"], 0, "and a bare reset clears everything")


# ---------------------------------------------------------------------------
# summarizer wiring: _count_tokens <-> tokenhealth <-> tokenize_health()
# ---------------------------------------------------------------------------

def test_summarizer_count_tokens_feeds_tokenize_health_on_failure():
    print("\n[test] a summarizer._count_tokens failure is visible on tokenize_health()")
    _reset()
    assert_eq(summarizer.tokenize_health()["ok"], True, "starts healthy")
    client = _FailingClient(exc=RuntimeError("connection refused"))
    n = asyncio.run(summarizer._count_tokens(client, "http://x", "m", "some text"))
    assert_true(n > 0, "still returns the pessimistic estimate, not a crash")
    h = summarizer.tokenize_health()
    assert_eq(h["ok"], False, "tokenize_health now reports unhealthy")
    assert_eq(h["consecutive_failures"], 1, "one failure counted")


def test_summarizer_count_tokens_recovers_tokenize_health_on_success():
    print("\n[test] a subsequent success clears tokenize_health()")
    _reset()
    client = _FailingClient(exc=RuntimeError("boom"))
    asyncio.run(summarizer._count_tokens(client, "http://x", "m", "text"))
    assert_eq(summarizer.tokenize_health()["ok"], False, "unhealthy after the failure")
    ok_client = _OkClient(count=7)
    n = asyncio.run(summarizer._count_tokens(ok_client, "http://x", "m", "text"))
    assert_eq(n, 7, "the real count is used, not the pessimistic fallback")
    assert_eq(summarizer.tokenize_health()["ok"], True, "healthy again")
    assert_eq(summarizer.tokenize_health()["consecutive_failures"], 0, "streak cleared")


def test_summarizer_count_tokens_warns_at_warning_and_names_the_outage():
    print("\n[test] the log line still fires, at WARNING, naming /tokenize")
    _reset()
    client = _FailingClient(exc=RuntimeError("connection refused"))
    with capture() as cap:
        asyncio.run(summarizer._count_tokens(client, "http://x", "m", "text"))
    warned = find(cap.records, "/tokenize unreachable")
    assert_true(warned is not None, "a warning line names the outage")
    assert_eq(warned.levelno, logging.WARNING, "at WARNING")


def test_summarizer_tokenize_failure_is_not_silenced_forever():
    print("\n[test] RESIDUAL 1(b): a persistent outage is not silenced after the first line")
    # This is the direct proof that log_once's per-process-lifetime memory is
    # gone from this call site. Under the OLD code, the second capture below
    # would see NOTHING — log_once("summarizer.tokenize.error") had already
    # returned True once, permanently, for this process.
    _reset()
    orig_interval = summarizer.TOKENIZE_WARN_INTERVAL_S
    summarizer.TOKENIZE_WARN_INTERVAL_S = 0.05
    try:
        client = _FailingClient(exc=RuntimeError("connection refused"))
        with capture() as cap1:
            asyncio.run(summarizer._count_tokens(client, "http://x", "m", "text"))
        first_warned = find(cap1.records, "/tokenize unreachable")
        assert_true(first_warned is not None, "first outage line fires")

        time.sleep(summarizer.TOKENIZE_WARN_INTERVAL_S * 3)

        with capture() as cap2:
            asyncio.run(summarizer._count_tokens(client, "http://x", "m", "text"))
        second_warned = find(cap2.records, "/tokenize unreachable")
        assert_true(second_warned is not None,
                     "a SECOND outage line fires once the window rolls over — "
                     "log_once would have silenced this forever")
        assert_true("2 consecutive failure" in second_warned.getMessage(),
                     "and the streak has grown, not reset")
    finally:
        summarizer.TOKENIZE_WARN_INTERVAL_S = orig_interval


def test_summarizer_tokenize_failure_still_collapses_within_one_window():
    print("\n[test] ...but still just ONE line per window, not one per call")
    _reset()
    orig_interval = summarizer.TOKENIZE_WARN_INTERVAL_S
    summarizer.TOKENIZE_WARN_INTERVAL_S = 300
    try:
        client = _FailingClient(exc=RuntimeError("connection refused"))
        with capture() as cap:
            for _ in range(5):
                asyncio.run(summarizer._count_tokens(client, "http://x", "m", "text"))
        warnings = [r for r in cap.records if r.levelno == logging.WARNING]
        assert_eq(len(warnings), 1, "5 failures in one window -> exactly 1 line")
        assert_eq(summarizer.tokenize_health()["consecutive_failures"], 5,
                   "but the health counter still saw all 5")
    finally:
        summarizer.TOKENIZE_WARN_INTERVAL_S = orig_interval


def _all():
    return [
        test_fresh_source_is_healthy,
        test_note_failure_returns_a_message_on_first_call,
        test_note_failure_is_rate_limited_within_one_window,
        test_note_failure_speaks_again_once_the_window_rolls_over,
        test_note_success_clears_and_reports_recovery_once,
        test_source_health_staleness_for_not_every_request_sources,
        test_reset_for_tests_clears_named_and_all_sources,
        test_summarizer_count_tokens_feeds_tokenize_health_on_failure,
        test_summarizer_count_tokens_recovers_tokenize_health_on_success,
        test_summarizer_count_tokens_warns_at_warning_and_names_the_outage,
        test_summarizer_tokenize_failure_is_not_silenced_forever,
        test_summarizer_tokenize_failure_still_collapses_within_one_window,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll tokenhealth smoke tests passed.")
