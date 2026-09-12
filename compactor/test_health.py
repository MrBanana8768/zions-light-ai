"""
CPU-only Tier-1 tests for compactor.health.

Mocks httpx for vLLM probe; uses a tmpdir as STORAGE_ROOT for the
real storage probe (so we exercise the actual fs path). Stubs
retrieval/summarizer counters where they'd otherwise need ChromaDB.

Run: python test_health.py
"""

import asyncio
import contextlib
import os
import shutil
import sys
import tempfile
import time
import threading
from unittest.mock import AsyncMock, MagicMock, patch

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_health_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import bgwork  # noqa: E402
import facts  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

# Stub the retrieval count so health doesn't fail trying to init ChromaDB
retrieval.conversation_doc_count = lambda conv_id: 0

import health  # noqa: E402


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


# ---------------------------------------------------------------------------
# probe_vllm
# ---------------------------------------------------------------------------

def test_probe_vllm_ok():
    print("\n[test] probe_vllm reports ok=True with model list on 200")

    async def go():
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={
            "data": [{"id": "magnum-v4-12b"}, {"id": "another"}]
        })
        with patch("health.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=mock_resp)
            return await health.probe_vllm("http://fake:8000")

    r = asyncio.run(go())
    assert_eq(r["ok"], True, "ok=True")
    assert_eq(r["models"], ["magnum-v4-12b", "another"], "models extracted")
    assert_eq(r["error"], None, "no error")


def test_probe_vllm_4xx():
    print("\n[test] probe_vllm reports ok=False on HTTP 5xx")

    async def go():
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("health.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=mock_resp)
            return await health.probe_vllm("http://fake:8000")

    r = asyncio.run(go())
    assert_eq(r["ok"], False, "ok=False on 503")
    assert_eq(r["models"], [], "empty model list")
    assert_true("503" in (r["error"] or ""), "error mentions code")


def test_probe_vllm_network_error():
    print("\n[test] probe_vllm catches network exceptions as ok=False")

    async def go():
        with patch("health.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(side_effect=ConnectionError("boom"))
            return await health.probe_vllm("http://fake:8000")

    r = asyncio.run(go())
    assert_eq(r["ok"], False, "ok=False on exception")
    assert_true("boom" in (r["error"] or ""), "error includes underlying message")


def test_probe_vllm_empty_model_list():
    print("\n[test] probe_vllm reports ok=False when no models listed")

    async def go():
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"data": []})
        with patch("health.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=mock_resp)
            return await health.probe_vllm("http://fake:8000")

    r = asyncio.run(go())
    assert_eq(r["ok"], False, "ok=False on empty list")
    assert_true("no models" in (r["error"] or ""), "error mentions no models")


# ---------------------------------------------------------------------------
# probe_storage
# ---------------------------------------------------------------------------

def test_probe_storage_writable():
    print("\n[test] probe_storage reports ok=True on writable mount")
    r = health.probe_storage()
    assert_eq(r["ok"], True, "tmpdir is writable")
    assert_eq(r["writable"], True, "writable flag set")
    assert_true(r["root"] == _TMP_ROOT, "root path reported")
    assert_eq(r["error"], None, "no error")


def test_probe_storage_reports_disk_usage():
    print("\n[test] probe_storage reports free_gb and total_gb when available")
    r = health.probe_storage()
    # shutil.disk_usage works on POSIX (linux test container); just verify
    # the fields are present and reasonable. We don't assert specific
    # values because they depend on the host disk.
    assert_true(r["free_gb"] is None or r["free_gb"] >= 0, "free_gb >= 0 or None")
    assert_true(r["total_gb"] is None or r["total_gb"] > 0, "total_gb > 0 or None")


# ---------------------------------------------------------------------------
# gather_memory_stats
# ---------------------------------------------------------------------------

def test_gather_memory_stats_empty():
    print("\n[test] gather_memory_stats handles empty storage")
    # Clear out any prior test files
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    os.makedirs(_TMP_ROOT, exist_ok=True)
    memory.ensure_storage_layout()
    s = health.gather_memory_stats()
    assert_eq(s["conversations"], 0, "zero convs")
    assert_eq(s["facts_total"], 0, "zero facts")
    assert_eq(s["indexed_exchanges_total"], 0, "zero episodic")


def test_gather_memory_stats_counts_across_convs():
    print("\n[test] gather_memory_stats aggregates across multiple convs")
    memory.ensure_storage_layout()
    facts.save_facts(
        "stats-A",
        [{"text": "a1", "added_turn": 0, "last_used": 0},
         {"text": "a2", "added_turn": 0, "last_used": 0}],
    )
    facts.save_facts(
        "stats-B",
        [{"text": "b1", "added_turn": 0, "last_used": 0}],
    )
    s = health.gather_memory_stats()
    assert_eq(s["conversations"], 2, "2 convs known")
    assert_eq(s["facts_total"], 3, "3 facts total across both")
    assert_eq(s["unreadable"], {"facts": 0, "episodic": 0, "summaries": 0},
              "nothing unreadable on a healthy store")


# ---------------------------------------------------------------------------
# gather_memory_stats — the corruption the health endpoint exists to catch
# (v3.1 P0-2b / F61)
# ---------------------------------------------------------------------------

def test_unreadable_facts_are_counted_not_hidden():
    """The defect this replaces: an unreadable facts file was skipped by a
    bare `pass`, so facts_total simply read LOWER while status stayed "ok".
    A silently smaller number is the failure mode — the total must come with
    a count of what could not be read."""
    print("\n[test] gather_memory_stats counts an unreadable facts file")
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    os.makedirs(_TMP_ROOT, exist_ok=True)
    memory.ensure_storage_layout()
    facts.save_facts("readable", [{"text": "r1", "added_turn": 0, "last_used": 0}])
    facts.save_facts("broken", [{"text": "b1", "added_turn": 0, "last_used": 0}])

    real_load = facts.load_facts

    def _load(cid):
        if cid == "broken":
            raise OSError(5, "Input/output error")
        return real_load(cid)

    facts.load_facts = _load
    try:
        s = health.gather_memory_stats()
    finally:
        facts.load_facts = real_load

    assert_eq(s["conversations"], 2, "both convs still enumerated")
    assert_eq(s["facts_total"], 1, "only the readable conv's fact counted")
    assert_eq(s["unreadable"]["facts"], 1, "the unreadable one is COUNTED")


def test_unreadable_summary_is_counted():
    print("\n[test] gather_memory_stats counts an unreadable summary state")
    real_load = summarizer.load_state

    def _load(cid):
        raise OSError(5, "Input/output error")

    summarizer.load_state = _load
    try:
        s = health.gather_memory_stats()
    finally:
        summarizer.load_state = real_load

    assert_eq(s["unreadable"]["summaries"], s["conversations"],
              "every conv's summary counted as unreadable")
    assert_eq(s["summaries_with_l1"], 0, "none counted as having l1")


def test_dead_vector_store_is_not_reported_as_empty():
    """conversation_doc_count returns None when Chroma is unavailable. If
    NOTHING could be counted, indexed_exchanges_total must be None — a dead
    vector store reporting `0` beside `"status": "ok"` is the blind spot."""
    print("\n[test] dead vector store -> indexed_exchanges_total is None, not 0")
    real_count = retrieval.conversation_doc_count
    retrieval.conversation_doc_count = lambda cid: None
    try:
        s = health.gather_memory_stats()
    finally:
        retrieval.conversation_doc_count = real_count

    assert_eq(s["indexed_exchanges_total"], None, "unknown, not 0")
    assert_eq(s["unreadable"]["episodic"], s["conversations"], "all convs uncounted")


def test_empty_vector_store_still_reports_zero():
    print("\n[test] healthy but empty vector store still reports 0")
    s = health.gather_memory_stats()
    assert_eq(s["indexed_exchanges_total"], 0, "0 means genuinely empty")
    assert_eq(s["unreadable"]["episodic"], 0, "nothing uncounted")


# ---------------------------------------------------------------------------
# gather_health_full — aggregate status logic
# ---------------------------------------------------------------------------

def test_status_ok_when_all_pass():
    print("\n[test] gather_health_full status='ok' when vllm + storage both pass")

    async def go():
        with patch("health.probe_vllm", new=AsyncMock(return_value={
            "ok": True, "latency_ms": 10.0, "models": ["m"], "error": None,
        })):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "ok", "status=ok")
    assert_eq(r["checks"]["vllm"]["ok"], True, "vllm check reported")
    assert_eq(r["checks"]["storage"]["ok"], True, "storage check reported")
    assert_eq(r["config"]["vllm_url"], "http://fake", "config echoed")
    assert_eq(r["config"]["target_tokens"], 4096, "target_tokens echoed")


def test_status_degraded_when_vllm_unreachable():
    print("\n[test] gather_health_full status='degraded' when vllm fails but storage ok")

    async def go():
        with patch("health.probe_vllm", new=AsyncMock(return_value={
            "ok": False, "latency_ms": 3000.0, "models": [], "error": "timeout",
        })):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "degraded", "status=degraded when vllm down")


def test_status_down_when_storage_broken():
    print("\n[test] gather_health_full status='down' when storage breaks")

    async def go():
        with patch("health.probe_storage", return_value={
            "ok": False, "writable": False, "root": "/x",
            "free_gb": None, "total_gb": None, "error": "EROFS",
        }):
            with patch("health.probe_vllm", new=AsyncMock(return_value={
                "ok": True, "latency_ms": 5.0, "models": ["m"], "error": None,
            })):
                return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "down", "status=down on storage failure")


# ---------------------------------------------------------------------------
# status_to_http_code
# ---------------------------------------------------------------------------

def test_status_to_http_code_mapping():
    print("\n[test] status_to_http_code: ok/degraded → 200, down → 503")
    assert_eq(health.status_to_http_code("ok"), 200, "ok → 200")
    assert_eq(health.status_to_http_code("degraded"), 200, "degraded → 200 (don't kill container)")
    assert_eq(health.status_to_http_code("down"), 503, "down → 503")
    assert_eq(health.status_to_http_code("unknown"), 200, "unknown → 200 (default ok)")


# ---------------------------------------------------------------------------
# Shedding must reach `status` (v3.1 A11 / incident C2)
#
# The defect: bgwork.pool.stats() was computed here, placed in the payload,
# and never consulted. The pool could be dropping every fact extraction,
# episodic index and summary rollup on the floor and /health/full answered
# "ok" — which is what the endpoint did during both incidents, while the user
# played the part of the monitoring. These tests pin the whole chain: the
# pool's shed reaches `status`, `status_reasons` says which condition it was,
# and an unreadable pool does not read as a healthy one.
# ---------------------------------------------------------------------------

class _FakePool:
    """Stands in for bgwork.pool. Only stats() is reached from health."""

    def __init__(self, stats):
        self._stats = stats

    def stats(self):
        return self._stats


class _BrokenPool:
    """A pool whose stats() blows up — the real shape of "we could not read
    the pool", which is what health's try/except around it is for."""

    def stats(self):
        raise RuntimeError("pool is wedged")


@contextlib.contextmanager
def _pool_reporting(stats):
    real = bgwork.pool
    bgwork.pool = stats if isinstance(stats, _BrokenPool) else _FakePool(stats)
    try:
        yield
    finally:
        bgwork.pool = real


def _quiet_pool(**over):
    """A pool that is doing its job: nothing shed, nothing pending."""
    base = {
        "outstanding": 0, "max_concurrent": 4, "max_outstanding": 64,
        "submitted": 10, "completed": 10, "shed": 0,
        "seconds_since_last_shed": None, "shed_recently": False,
        "shed_window_s": 300.0, "at_capacity": False,
    }
    base.update(over)
    return base


def _healthy_vllm():
    return patch("health.probe_vllm", new=AsyncMock(return_value={
        "ok": True, "latency_ms": 10.0, "models": ["m"], "error": None,
    }))


def test_shedding_degrades_the_status():
    print("\n[test] a shedding background pool makes /health/full 'degraded'")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool(
            shed=37, seconds_since_last_shed=12.0, shed_recently=True,
            outstanding=64, at_capacity=True,
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "degraded", "shedding is NOT 'ok'")
    joined = " | ".join(r["status_reasons"])
    assert_true("shedding" in joined, "the reason names shedding")
    assert_true("37" in joined, "and how many tails were dropped")
    assert_true("12.0" in joined, "and how long ago the last one was")


def test_a_pool_that_shed_long_ago_is_ok_again():
    """Shedding degrades while it is happening and for one window after, then
    clears on its own. A health warning that only a restart can silence gets
    ignored, which is how the endpoint became decoration in the first place."""
    print("\n[test] a shed outside the window is 'ok' again, count intact")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool(
            shed=37, seconds_since_last_shed=901.0, shed_recently=False,
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "ok", "aged-out shedding does not pin 'degraded'")
    assert_eq(r["status_reasons"], [], "and nothing is reported")
    assert_eq(r["background_work"]["shed"], 37,
              "the cumulative count is still in the payload as history")


def test_at_capacity_alone_does_not_degrade():
    """A pool that touches its ceiling and drains again lost nothing. If it
    stays full the shed follows within one submission and `shed_recently`
    picks it up then — degrading on `at_capacity` would just flap.

    Unlike its neighbours this one also passes against the pre-v3.1 code,
    because it pins the boundary of the fix rather than the fix: it fails
    only if someone later widens the degrade condition to "the pool looks
    busy". Kept deliberately."""
    print("\n[test] at_capacity without a shed stays 'ok'")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool(
            outstanding=64, at_capacity=True,
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "ok", "full but not dropping is still ok")


def test_unreadable_pool_is_not_reported_as_healthy():
    """Same doctrine as indexed_exchanges_total: unknown is not fine. If we
    cannot read the pool at all we cannot say it isn't shedding."""
    print("\n[test] a pool we cannot read degrades rather than reading 'ok'")

    async def go():
        with _healthy_vllm(), _pool_reporting(_BrokenPool()):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "degraded", "unobservable != healthy")
    assert_true(any("unobservable" in x for x in r["status_reasons"]),
                "and the reason says we could not see it")


def test_status_reasons_accumulate():
    """Two things wrong must report as two things, not as whichever the
    if/elif chain happened to reach first."""
    print("\n[test] concurrent degradations all appear in status_reasons")

    async def go():
        with patch("health.probe_vllm", new=AsyncMock(return_value={
            "ok": False, "latency_ms": 3000.0, "models": [], "error": "timeout",
        })), _pool_reporting(_quiet_pool(
            shed=3, seconds_since_last_shed=1.0, shed_recently=True,
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "degraded", "degraded")
    assert_eq(len(r["status_reasons"]), 2, "BOTH conditions listed")
    joined = " | ".join(r["status_reasons"])
    assert_true("vLLM" in joined, "vLLM named")
    assert_true("shedding" in joined, "shedding named")


def test_shedding_still_answers_200():
    """Deliberate: shedding is backpressure. Restarting the container in the
    middle of it kills every in-flight chat AND guarantees the loss of every
    tail still outstanding — the harm we were trying to report. The signal
    belongs in the body, not in the HEALTHCHECK's exit code."""
    print("\n[test] a shedding box stays HEALTHY to Docker; the body carries it")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool(
            shed=5, seconds_since_last_shed=2.0, shed_recently=True,
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(health.status_to_http_code(r["status"]), 200,
              "200 — do not restart a box that is merely under load")
    assert_true(r["status_reasons"], "but the body is not silent about it")


def test_ok_carries_an_empty_reason_list():
    """`status_reasons` is always present, so a consumer can read it without
    a key check and a healthy run is distinguishable from an old payload."""
    print("\n[test] a healthy report still carries status_reasons: []")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool()):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "ok", "ok")
    assert_eq(r["status_reasons"], [], "empty list, not absent")


def test_broken_storage_reports_its_reason_too():
    print("\n[test] status='down' also says why")

    async def go():
        with patch("health.probe_storage", return_value={
            "ok": False, "writable": False, "root": "/x",
            "free_gb": None, "total_gb": None, "error": "EROFS",
        }), _healthy_vllm(), _pool_reporting(_quiet_pool()):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "down", "down")
    assert_true(any("storage" in x for x in r["status_reasons"]), "storage named")


# ---------------------------------------------------------------------------
# Memory-tail skips must reach `status` (v3.1.4)
#
# The defect: a reply that did not enter memory — she hit Stop, vLLM hit the
# ceiling, the accumulator dropped a chunk — was a WARNING line and nothing
# else. 63 skipped exchanges in one 2026-09-01 log window, more than half her
# recent conversation, and /health/full said ok throughout. These pin the
# chain the same way the shedding tests above do: the counter reaches
# `status`, `status_reasons` says which condition it was and how many, a
# stale skip does not pin the endpoint degraded forever, and an unreadable
# counter does not read as a healthy one. One test uses the REAL tailhealth
# module rather than a fake, so a health.py that reads a fake correctly but
# never imports the real counter cannot pass.
# ---------------------------------------------------------------------------

import tailhealth  # noqa: E402


def _quiet_tail(**over):
    """A tail that is doing its job: everything stored, nothing skipped."""
    base = {
        "stored": 40, "skipped": 0, "outcomes": {}, "consecutive_skips": 0,
        "last_skip_outcome": None, "seconds_since_last_skip": None,
        "skipped_recently": False, "skip_window_s": 300.0,
        "raw_chars": 0, "kept_chars": 0, "trimmed_raw_chars": 0,
        "trimmed_kept_chars": 0, "trim_retention": None,
    }
    base.update(over)
    return base


class _BrokenTail:
    def __call__(self):
        raise RuntimeError("counter is wedged")


@contextlib.contextmanager
def _tail_reporting(snap):
    fn = snap if isinstance(snap, _BrokenTail) else (lambda **kw: snap)
    with patch("tailhealth.snapshot", new=fn):
        yield


def test_tail_skips_degrade_the_status():
    print("\n[test] recent memory-tail skips make /health/full 'degraded'")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool()), _tail_reporting(_quiet_tail(
            skipped=63, consecutive_skips=7, seconds_since_last_skip=12.0,
            skipped_recently=True, last_skip_outcome="skipped_no_boundary",
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "degraded", "skipping is NOT 'ok'")
    joined = " | ".join(r["status_reasons"])
    assert_true("memory tail" in joined, "the reason names the memory tail")
    assert_true("63" in joined, "and how many replies were not memorized")
    assert_true("7 consecutive" in joined, "and the streak")
    assert_true("12.0" in joined, "and how long ago the last one was")
    assert_true("skipped_no_boundary" in joined, "and the last machine outcome")
    assert_eq(r["memory_tail"]["skipped"], 63, "the block is in the payload")


def test_a_tail_skip_long_ago_is_ok_again():
    """The stale-skip case. Degrading on the CUMULATIVE count would pin the
    endpoint 'degraded' from the first skip until the next restart — and a
    warning that is always on is a warning nobody reads, which is how the
    endpoint became decoration in the first place (v3.1 A11)."""
    print("\n[test] a skip outside the window is 'ok' again, count intact")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool()), _tail_reporting(_quiet_tail(
            skipped=63, consecutive_skips=63, seconds_since_last_skip=901.0,
            skipped_recently=False, last_skip_outcome="skipped_no_boundary",
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "ok", "an aged-out skip does not pin 'degraded'")
    assert_eq(r["status_reasons"], [], "and nothing is reported")
    assert_eq(r["memory_tail"]["skipped"], 63,
              "the cumulative count is still in the payload as history")


def test_unreadable_tail_counter_is_not_reported_as_healthy():
    print("\n[test] a tail counter we cannot read degrades rather than reading 'ok'")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool()), _tail_reporting(_BrokenTail()):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "degraded", "unobservable != healthy")
    assert_true(any("memory tail unobservable" in x for x in r["status_reasons"]),
                "and the reason says we could not see it")
    assert_true("error" in r["memory_tail"], "the payload carries the error")


def test_tail_skips_still_answer_200():
    """Same deliberate choice as shedding: the signal belongs in the body.
    Restarting the container would not bring the skipped memories back and
    would kill every in-flight chat."""
    print("\n[test] a skipping tail stays HEALTHY to Docker; the body carries it")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool()), _tail_reporting(_quiet_tail(
            skipped=3, consecutive_skips=3, seconds_since_last_skip=2.0,
            skipped_recently=True, last_skip_outcome="skipped_too_short",
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(health.status_to_http_code(r["status"]), 200, "200 — do not restart")
    assert_true(r["status_reasons"], "but the body is not silent about it")


def test_tail_reason_joins_the_others():
    """Three things wrong report as three, not as whichever came first."""
    print("\n[test] a tail skip is listed alongside vLLM and shedding")

    async def go():
        with patch("health.probe_vllm", new=AsyncMock(return_value={
            "ok": False, "latency_ms": 3000.0, "models": [], "error": "timeout",
        })), _pool_reporting(_quiet_pool(
            shed=3, seconds_since_last_shed=1.0, shed_recently=True,
        )), _tail_reporting(_quiet_tail(
            skipped=1, consecutive_skips=1, seconds_since_last_skip=1.0,
            skipped_recently=True, last_skip_outcome="skipped_holed",
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(len(r["status_reasons"]), 3, "ALL THREE conditions listed")


def test_the_real_counter_is_wired():
    """Not a fake: drive the real tailhealth module and read it back through
    health. A health.py that handled a patched snapshot correctly but never
    imported the real one would pass every test above and still be blind."""
    print("\n[test] the REAL tailhealth counter reaches /health/full")
    tailhealth._reset_for_tests()
    try:
        async def go():
            with _healthy_vllm(), _pool_reporting(_quiet_pool()):
                before = await health.gather_health_full("http://fake", 4096)
                tailhealth.note(tailhealth.SKIPPED_NO_BOUNDARY, raw_chars=812, kept_chars=0)
                tailhealth.note(tailhealth.SKIPPED_TOO_SHORT, raw_chars=400, kept_chars=0)
                after = await health.gather_health_full("http://fake", 4096)
                return before, after

        before, after = asyncio.run(go())
        assert_eq(before["status"], "ok", "fresh counter: ok")
        assert_eq(before["memory_tail"]["skipped"], 0, "and the block reads 0")
        assert_eq(after["status"], "degraded", "two real skips: degraded")
        assert_eq(after["memory_tail"]["skipped"], 2, "the real count is in the payload")
        assert_eq(after["memory_tail"]["last_skip_outcome"], "skipped_too_short",
                  "with the real last outcome")
        assert_true(any("2 reply(ies)" in x for x in after["status_reasons"]),
                    "and the reason carries the real number")
    finally:
        tailhealth._reset_for_tests()


def test_an_empty_reply_skip_does_not_degrade_status():
    """v3.1.7, R27. SKIPPED_EMPTY is she-hit-Stop-before-the-first-token:
    there was never any text to memorize, so it is the one skip label that
    carries no loss. Keying the degrade decision off "any skip" made a run of
    these — the release's own figure is 51 of 63 skips in one window were
    manual stops — pin /health/full degraded for a turn that lost nothing.
    Mirrors test_tail_skips_degrade_the_status exactly except for the
    outcome label, so the only variable is whether that label degrades."""
    print("\n[test] an empty-reply skip alone does NOT degrade /health/full")

    async def go():
        with _healthy_vllm(), _pool_reporting(_quiet_pool()), _tail_reporting(_quiet_tail(
            skipped=63, consecutive_skips=63, seconds_since_last_skip=0.0,
            skipped_recently=False, last_skip_outcome="skipped_empty",
        )):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "ok", "an empty-reply skip is not memory loss")
    assert_eq(r["status_reasons"], [], "so nothing is reported")
    assert_eq(r["memory_tail"]["skipped"], 63,
              "the cumulative count is still in the payload as history")


def test_the_real_counter_does_not_degrade_on_empty_skips_alone():
    """Not a fake: drive the real tailhealth.note() with SKIPPED_EMPTY only,
    the way test_the_real_counter_is_wired drives it with lossy outcomes. A
    health.py/tailhealth.py pair that passed every mocked test above but
    still computed `skipped_recently` off ANY skip internally would be blind
    to this."""
    print("\n[test] the REAL tailhealth counter: empty skips alone stay 'ok'")
    tailhealth._reset_for_tests()
    try:
        async def go():
            with _healthy_vllm(), _pool_reporting(_quiet_pool()):
                for _ in range(63):
                    tailhealth.note(tailhealth.SKIPPED_EMPTY, raw_chars=0, kept_chars=0)
                return await health.gather_health_full("http://fake", 4096)

        r = asyncio.run(go())
        assert_eq(r["status"], "ok", "63 real empty skips: still ok")
        assert_eq(r["status_reasons"], [], "nothing reported")
        assert_eq(r["memory_tail"]["skipped"], 63, "but the record shows all 63")
        assert_eq(r["memory_tail"]["last_skip_outcome"], "skipped_empty",
                  "with the real last outcome")

        # A LOSSY skip afterward still degrades — the fix must not have made
        # tailhealth blind to real loss, only to the zero-loss label.
        async def go2():
            with _healthy_vllm(), _pool_reporting(_quiet_pool()):
                tailhealth.note(tailhealth.SKIPPED_NO_BOUNDARY, raw_chars=100, kept_chars=0)
                return await health.gather_health_full("http://fake", 4096)

        r2 = asyncio.run(go2())
        assert_eq(r2["status"], "degraded", "a real lossy skip still degrades")
    finally:
        tailhealth._reset_for_tests()


# ---------------------------------------------------------------------------
# The store scan must not run on the event loop (v3.1 A12)
# ---------------------------------------------------------------------------

def test_blocking_probes_run_off_the_event_loop():
    """`gather_memory_stats` is O(conversations) and reads three layers each;
    measured at ~100 ms for 1000 conversations on this image, every 30 s from
    the Docker HEALTHCHECK, on the one event loop this process has
    (supervisord starts uvicorn with no --workers). Every concurrent chat
    stalled for the length of the scan.

    This asserts the structure rather than the duration: the filesystem probes
    must execute on some thread that is not the loop's. A timing assertion
    would be flaky; a thread-identity assertion fails the moment someone puts
    the scan back inline."""
    print("\n[test] the filesystem probes execute off the loop thread")

    seen = {}

    def _recording_storage():
        seen["storage"] = threading.get_ident()
        return {"ok": True, "writable": True, "root": _TMP_ROOT,
                "free_gb": 1.0, "total_gb": 2.0, "error": None}

    def _recording_stats():
        seen["stats"] = threading.get_ident()
        return {"conversations": 0, "facts_total": 0,
                "indexed_exchanges_total": 0, "summaries_with_l1": 0,
                "summaries_with_l3": 0,
                "unreadable": {"facts": 0, "episodic": 0, "summaries": 0}}

    async def go():
        seen["loop"] = threading.get_ident()
        with patch("health.probe_storage", new=_recording_storage), \
             patch("health.gather_memory_stats", new=_recording_stats), \
             _healthy_vllm(), _pool_reporting(_quiet_pool()):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "ok", "the report still comes back intact")
    assert_true(seen["storage"] != seen["loop"],
                "probe_storage ran on a worker thread, not the loop")
    assert_true(seen["stats"] != seen["loop"],
                "gather_memory_stats ran on a worker thread, not the loop")
    assert_eq(seen["storage"], seen["stats"],
              "and both in ONE hop — not a thread per probe")


def test_loop_stays_responsive_while_the_scan_runs():
    """The same point stated as behaviour rather than as thread identity:
    another coroutine gets to run WHILE the store scan is still in progress.

    The mechanism is a deadlock that only resolves one way round. The scan
    blocks on `release`, which only the ticker coroutine can set. If the scan
    is on the loop, the ticker cannot run until the scan returns, so the wait
    times out and `released` is False — the scan finishes first and the loop
    was frozen for its whole duration. Off the loop, the ticker runs, sets
    `release`, and the scan returns immediately.

    Note the earlier version of this test asserted only that the ticker had
    run by the END, which the blocking arrangement also satisfies — it just
    took the full timeout to get there. It passed against the pre-fix code."""
    print("\n[test] another coroutine progresses DURING the store scan")

    started = threading.Event()
    release = threading.Event()
    observed = {}

    def _slow_stats():
        started.set()
        # 2 s only matters on the failure path; a healthy hop releases at once.
        observed["released_during_scan"] = release.wait(2.0)
        return {"conversations": 0, "facts_total": 0,
                "indexed_exchanges_total": 0, "summaries_with_l1": 0,
                "summaries_with_l3": 0,
                "unreadable": {"facts": 0, "episodic": 0, "summaries": 0}}

    async def ticker():
        # Wait until the scan is genuinely underway before releasing it, so
        # this cannot pass by simply running first.
        while not started.is_set():
            await asyncio.sleep(0.005)
        release.set()

    async def go():
        with patch("health.gather_memory_stats", new=_slow_stats), \
             _healthy_vllm(), _pool_reporting(_quiet_pool()):
            report, _ = await asyncio.gather(
                health.gather_health_full("http://fake", 4096), ticker()
            )
            return report

    r = asyncio.run(go())
    assert_eq(observed["released_during_scan"], True,
              "the loop ran another coroutine mid-scan, not after it")
    assert_eq(r["status"], "ok", "and the health report completed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_probe_vllm_ok,
        test_probe_vllm_4xx,
        test_probe_vllm_network_error,
        test_probe_vllm_empty_model_list,
        test_probe_storage_writable,
        test_probe_storage_reports_disk_usage,
        test_gather_memory_stats_empty,
        test_gather_memory_stats_counts_across_convs,
        # Order matters: the first of these reseeds the store with the two
        # convs ("readable" / "broken") the rest of the group counts against.
        test_unreadable_facts_are_counted_not_hidden,
        test_unreadable_summary_is_counted,
        test_dead_vector_store_is_not_reported_as_empty,
        test_empty_vector_store_still_reports_zero,
        test_status_ok_when_all_pass,
        test_status_degraded_when_vllm_unreachable,
        test_status_down_when_storage_broken,
        test_status_to_http_code_mapping,
        # v3.1 A11 — shedding has to reach `status`, and say so.
        test_shedding_degrades_the_status,
        test_a_pool_that_shed_long_ago_is_ok_again,
        test_at_capacity_alone_does_not_degrade,
        test_unreadable_pool_is_not_reported_as_healthy,
        test_status_reasons_accumulate,
        test_shedding_still_answers_200,
        test_ok_carries_an_empty_reason_list,
        test_broken_storage_reports_its_reason_too,
        # v3.1.4 — memory-tail skips have to reach `status`, and say so.
        test_tail_skips_degrade_the_status,
        test_a_tail_skip_long_ago_is_ok_again,
        test_unreadable_tail_counter_is_not_reported_as_healthy,
        test_tail_skips_still_answer_200,
        test_tail_reason_joins_the_others,
        test_the_real_counter_is_wired,
        # v3.1.7, R27 — SKIPPED_EMPTY carries no loss and must not degrade.
        test_an_empty_reply_skip_does_not_degrade_status,
        test_the_real_counter_does_not_degrade_on_empty_skips_alone,
        # v3.1 A12 — the store scan must not block the one event loop.
        test_blocking_probes_run_off_the_event_loop,
        test_loop_stays_responsive_while_the_scan_runs,
        test_snapshot_probe_is_silent_when_the_gate_is_off,
        test_snapshot_probe_flags_a_snapshot_that_stopped_refreshing,
        test_snapshot_probe_treats_a_missing_snapshot_as_stale,
        test_hierarchy_lag_is_measured_and_reported,
        test_hierarchy_lag_surfaces_as_a_status_reason,
    ]


def test_snapshot_probe_is_silent_when_the_gate_is_off():
    """With WEBUI_DB_LOCAL=false the snapshot IS the live database.

    Its mtime moves on every message, so "stale" would be meaningless, and a
    number nobody should read is worse than no number. watched=False says so.
    """
    print("")
    print("[test] snapshot probe: not watched when the sync daemon is off")
    with patch.dict(os.environ, {"WEBUIDB_SYNC_ENABLED": "false"}):
        out = health.probe_snapshot()
    assert_eq(out["watched"], False, "watched is False")
    assert_eq(out["stale"], False, "and it does not claim staleness")
    assert_eq(out["age_s"], None, "and reports no age rather than a wrong one")


def test_snapshot_probe_flags_a_snapshot_that_stopped_refreshing():
    """The one condition where /data holds no copy that survives a recreate.

    sync_loop shouted three times and went quiet; this endpoint never looked.
    The result was a pod reporting "ok", chat working perfectly, and an
    unbounded durability gap.
    """
    print("")
    print("[test] snapshot probe: a stale snapshot is flagged, a fresh one is not")
    snap = os.path.join(_TMP_ROOT, "snapshot-webui.db")
    with open(snap, "wb") as fh:
        fh.write(b"x")
    env = {
        "WEBUIDB_SYNC_ENABLED": "true",
        "WEBUI_SNAPSHOT_DB": snap,
        "WEBUI_DB_SYNC_INTERVAL_S": "300",
    }

    # Fresh: just written, well inside one interval.
    with patch.dict(os.environ, env):
        fresh = health.probe_snapshot()
    assert_eq(fresh["watched"], True, "watched is True when the daemon owns it")
    assert_eq(fresh["stale"], False, "a just-written snapshot is not stale")

    # Stale: four intervals of age, past the three-interval threshold.
    old = time.time() - (4 * 300)
    os.utime(snap, (old, old))
    with patch.dict(os.environ, env):
        gone = health.probe_snapshot()
    assert_eq(gone["stale"], True, "four intervals of age reads as stale")
    assert_true(gone["age_s"] >= 3 * 300, "and the age is reported")

    # CONTROL on the threshold itself: two intervals must NOT trip it, or the
    # check is just "any age at all" and cries wolf on every slow cycle.
    recent = time.time() - (2 * 300)
    os.utime(snap, (recent, recent))
    with patch.dict(os.environ, env):
        ok = health.probe_snapshot()
    assert_eq(ok["stale"], False,
              "two intervals is an overrun, not a dead daemon")


def test_snapshot_probe_treats_a_missing_snapshot_as_stale():
    """Missing is worse than stale, not better.

    The daemon is supposed to be writing this file. Reporting a quiet False on
    an OSError would be the same silence the probe exists to end.
    """
    print("")
    print("[test] snapshot probe: an absent snapshot is not a quiet pass")
    with patch.dict(os.environ, {
        "WEBUIDB_SYNC_ENABLED": "true",
        "WEBUI_SNAPSHOT_DB": os.path.join(_TMP_ROOT, "does-not-exist.db"),
    }):
        out = health.probe_snapshot()
    assert_eq(out["stale"], True, "absent reads as stale")
    assert_true(out["error"], "with the reason attached")


def test_hierarchy_lag_is_measured_and_reported():
    """The rollup's only observable.

    v3.1.8's skip-path rollup has six ways to do nothing, five of them
    completely silent, and the failure it exists to fix - a soak ending with
    last_summarized_turn still 0 - reported status "ok" with empty reasons.
    """
    print("")
    print("[test] hierarchy lag: measured by the scan and named by conversation")
    cid = "lagging_conv"
    st = summarizer.load_state(cid)
    st["l1"] = [{"text": "a scene", "first_turn": 1, "last_turn": 20}]
    st["last_summarized_turn"] = 20
    st["turns_seen"] = 20 + (3 * summarizer.L1_CHUNK_SIZE)
    summarizer.save_state(cid, st)

    stats = health.gather_memory_stats()
    assert_eq(stats["hierarchy_lag"], 3 * summarizer.L1_CHUNK_SIZE,
              "the scan measured turns_seen - last_summarized_turn")
    assert_eq(stats["hierarchy_lag_conv"], cid, "and named the conversation")

    # CONTROL: a conversation keeping pace must NOT be reported, or this
    # degrades every healthy pod and gets switched off within a week.
    st["last_summarized_turn"] = st["turns_seen"]
    summarizer.save_state(cid, st)
    caught_up = health.gather_memory_stats()
    assert_eq(caught_up["hierarchy_lag"], 0,
              "a hierarchy that has kept pace reports no lag")


def test_hierarchy_lag_surfaces_as_a_status_reason():
    """Measuring it is half the fix; SAYING it is the half that matters.

    stats.hierarchy_lag in the payload is what stats.unreadable used to be -
    a number nobody reads, inside a body whose top line says everything is
    fine. This asserts the reason, and asserts a caught-up hierarchy stays
    quiet, because a check that degrades a healthy pod is switched off within
    a week.
    """
    print("")
    print("[test] hierarchy lag: degrades the pod, and only when it should")
    cid = "lag_reason_conv"
    st = summarizer.load_state(cid)
    st["l1"] = [{"text": "a scene", "first_turn": 1, "last_turn": 20}]
    st["last_summarized_turn"] = 20
    st["turns_seen"] = 20 + (3 * summarizer.L1_CHUNK_SIZE)
    summarizer.save_state(cid, st)

    async def go():
        with patch("health.probe_vllm", new=AsyncMock(return_value={
            "ok": True, "latency_ms": 10.0, "models": ["m"], "error": None,
        })):
            return await health.gather_health_full("http://fake", 4096)

    r = asyncio.run(go())
    assert_eq(r["status"], "degraded", "a lagging hierarchy degrades the pod")
    assert_true(
        any("summary hierarchy is" in x for x in r["status_reasons"]),
        "and says so by name: %r" % (r["status_reasons"],),
    )

    # CONTROL, and it is ONE CHUNK of lag rather than zero on purpose.
    # The rollup fires on the tail, so the newest turns are always ahead
    # of the watermark - a hierarchy is never caught up in the strict
    # sense. A control at zero would pass with the threshold set to 0 and
    # would leave every healthy pod degraded. This pins the boundary.
    st["last_summarized_turn"] = st["turns_seen"] - summarizer.L1_CHUNK_SIZE
    summarizer.save_state(cid, st)
    r2 = asyncio.run(go())
    assert_true(
        not any("summary hierarchy is" in x for x in r2["status_reasons"]),
        "one chunk of ordinary drift raises no reason: %r"
        % (r2["status_reasons"],),
    )


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll health smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
