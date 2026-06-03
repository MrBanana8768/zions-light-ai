"""
CPU-only Tier-1 tests for compactor.health.

Mocks httpx for vLLM probe; uses a tmpdir as STORAGE_ROOT for the
real storage probe (so we exercise the actual fs path). Stubs
retrieval/summarizer counters where they'd otherwise need ChromaDB.

Run: python test_health.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_health_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

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
        test_status_ok_when_all_pass,
        test_status_degraded_when_vllm_unreachable,
        test_status_down_when_storage_broken,
        test_status_to_http_code_mapping,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll health smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
