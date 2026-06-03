"""
CPU-only Tier-1 tests for compactor.selftest.

Mocks httpx for the compactor/vLLM HTTP probes; uses real tmpdir
storage for the facts round-trip check (verifying the actual write
path). Verifies:
  - run_selftest() aggregates correctly (pass/fail status)
  - individual checks degrade to ok=False rather than raising
  - report shape matches the documented schema
  - facts_round_trip cleans up the sentinel even on failure
  - CLI flag parsing + report rendering

Run: python test_selftest.py
"""

import asyncio
import io
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_selftest_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import facts  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

# Stub the retrieval count to avoid ChromaDB init in CPU tests
retrieval.conversation_doc_count = lambda conv_id: 0

import selftest  # noqa: E402


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
# facts_round_trip — real storage, no mocks
# ---------------------------------------------------------------------------

def test_facts_round_trip_succeeds_with_real_storage():
    print("\n[test] _check_facts_round_trip succeeds against real tmpdir storage")
    memory.ensure_storage_layout()
    ok, detail = selftest._check_facts_round_trip()
    assert_eq(ok, True, "facts round-trip ok=True")
    assert_true("sentinel=" in detail, "detail mentions sentinel")
    # Verify cleanup happened — sentinel conv should have no facts left
    assert_eq(
        facts.load_facts(selftest.SELFTEST_CONV_ID), [],
        "sentinel cleaned up",
    )


def test_facts_round_trip_cleans_up_even_on_inner_failure():
    print("\n[test] _check_facts_round_trip cleans up sentinel after failure")
    memory.ensure_storage_layout()
    # Pre-pollute the sentinel so we can verify wipe-on-cleanup
    facts.save_facts(
        selftest.SELFTEST_CONV_ID,
        [{"text": "leftover from prior crash", "added_turn": 0, "last_used": 0}],
    )
    ok, _ = selftest._check_facts_round_trip()
    assert_eq(ok, True, "still passes (cleanup runs first)")
    assert_eq(
        facts.load_facts(selftest.SELFTEST_CONV_ID), [],
        "pollution cleaned up",
    )


# ---------------------------------------------------------------------------
# HTTP-based checks
# ---------------------------------------------------------------------------

def test_compactor_health_check_200():
    print("\n[test] _check_compactor_health: 200 → ok")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        client.get = AsyncMock(return_value=resp)
        return await selftest._check_compactor_health(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, True, "ok=True on 200")
    assert_true("200" in detail, "detail mentions code")


def test_compactor_health_check_500():
    print("\n[test] _check_compactor_health: 500 → fail")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "internal error"
        client.get = AsyncMock(return_value=resp)
        return await selftest._check_compactor_health(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False on 500")


def test_chat_round_trip_well_formed_response():
    print("\n[test] _check_chat_round_trip extracts content from valid response")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Hi"}}]
        })
        client.post = AsyncMock(return_value=resp)
        client.delete = AsyncMock(return_value=MagicMock(status_code=200))
        return await selftest._check_chat_round_trip(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, True, "ok=True on valid completion")
    assert_true("response_len=" in detail, "reports response length")


def test_chat_round_trip_malformed_response():
    print("\n[test] _check_chat_round_trip fails gracefully on malformed body")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(return_value={"wrong_shape": True})
        resp.text = '{"wrong_shape": true}'
        client.post = AsyncMock(return_value=resp)
        client.delete = AsyncMock(return_value=MagicMock(status_code=200))
        return await selftest._check_chat_round_trip(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False on malformed response")
    assert_true("malformed" in detail.lower(), "detail explains the failure")


def test_admin_localhost_200():
    print("\n[test] _check_admin_localhost: 200 → ok")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        client.get = AsyncMock(return_value=resp)
        return await selftest._check_admin_localhost(client)

    ok, _ = asyncio.run(go())
    assert_eq(ok, True, "ok=True")


def test_admin_localhost_403_fails():
    print("\n[test] _check_admin_localhost: 403 → fail (gating misconfigured?)")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 403
        resp.text = "forbidden"
        client.get = AsyncMock(return_value=resp)
        return await selftest._check_admin_localhost(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False on 403")
    assert_true("403" in detail, "detail mentions code")


# ---------------------------------------------------------------------------
# run_selftest — aggregate
# ---------------------------------------------------------------------------

def test_run_selftest_all_passing():
    print("\n[test] run_selftest: status='pass' when all checks succeed")

    # Mock every async check to succeed
    async def go():
        with patch.object(selftest, "_check_vllm_models",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_compactor_health",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_admin_localhost",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_chat_round_trip",
                          new=AsyncMock(return_value=(True, "ok"))):
            return await selftest.run_selftest(do_round_trip=True)

    report = asyncio.run(go())
    assert_eq(report["status"], "pass", "status=pass")
    assert_eq(report["summary"]["failed"], 0, "0 failed")
    assert_eq(report["summary"]["passed"], 6, "6 checks passed")
    assert_eq(report["summary"]["total"], 6, "6 total")


def test_run_selftest_one_failure_flips_status():
    print("\n[test] run_selftest: one fail → status='fail' overall")

    async def go():
        with patch.object(selftest, "_check_vllm_models",
                          new=AsyncMock(return_value=(False, "down"))), \
             patch.object(selftest, "_check_compactor_health",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_admin_localhost",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_chat_round_trip",
                          new=AsyncMock(return_value=(True, "ok"))):
            return await selftest.run_selftest(do_round_trip=True)

    report = asyncio.run(go())
    assert_eq(report["status"], "fail", "status=fail when any check fails")
    assert_eq(report["summary"]["failed"], 1, "1 failed")
    # vllm_models is the one we made fail — find it
    vllm_check = next(c for c in report["checks"] if c["name"] == "vllm_models")
    assert_eq(vllm_check["ok"], False, "vllm_models marked failed")
    assert_true("down" in vllm_check["detail"], "detail preserved")


def test_run_selftest_skip_round_trip():
    print("\n[test] run_selftest(do_round_trip=False) omits the chat check")

    async def go():
        with patch.object(selftest, "_check_vllm_models",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_compactor_health",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_admin_localhost",
                          new=AsyncMock(return_value=(True, "ok"))):
            return await selftest.run_selftest(do_round_trip=False)

    report = asyncio.run(go())
    assert_eq(report["summary"]["total"], 5, "5 checks (chat skipped)")
    names = [c["name"] for c in report["checks"]]
    assert_true("chat_round_trip" not in names, "chat_round_trip not in checks")


def test_run_selftest_inner_exception_becomes_ok_false():
    print("\n[test] run_selftest: an inner exception is caught, not propagated")

    async def boom():
        raise RuntimeError("explosion")

    async def go():
        with patch.object(selftest, "_check_vllm_models",
                          new=AsyncMock(side_effect=RuntimeError("explosion"))), \
             patch.object(selftest, "_check_compactor_health",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_admin_localhost",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_chat_round_trip",
                          new=AsyncMock(return_value=(True, "ok"))):
            return await selftest.run_selftest(do_round_trip=True)

    report = asyncio.run(go())
    # Should NOT raise — must produce a report with vllm_models failed
    assert_eq(report["status"], "fail", "status=fail")
    vllm_check = next(c for c in report["checks"] if c["name"] == "vllm_models")
    assert_eq(vllm_check["ok"], False, "vllm marked failed via exception path")
    assert_true("RuntimeError" in vllm_check["detail"], "exception type preserved")


# ---------------------------------------------------------------------------
# wait_for_vllm_ready — two-phase readiness probe (V2.1 Phase 6.1)
# ---------------------------------------------------------------------------

def test_wait_for_vllm_ready_succeeds_when_both_phases_pass():
    print("\n[test] wait_for_vllm_ready: /v1/models 200 + chat 200 → ready")

    models_resp = MagicMock(status_code=200)
    models_resp.json = MagicMock(return_value={"data": [{"id": "x"}]})
    chat_resp = MagicMock(status_code=200)

    async def go():
        with patch("selftest.httpx.AsyncClient") as MockClient, \
             patch.object(selftest, "WAIT_FOR_READY_POLL_INTERVAL_S", 0.01):
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=models_resp)
            instance.post = AsyncMock(return_value=chat_resp)
            return await selftest.wait_for_vllm_ready(timeout_s=5.0)

    ready = asyncio.run(go())
    assert_eq(ready, True, "returns True when both phases pass")


def test_wait_for_vllm_ready_keeps_polling_when_models_404():
    print("\n[test] wait_for_vllm_ready: /v1/models 404 → keep polling, timeout=False")

    models_resp = MagicMock(status_code=404)

    async def go():
        with patch("selftest.httpx.AsyncClient") as MockClient, \
             patch.object(selftest, "WAIT_FOR_READY_POLL_INTERVAL_S", 0.01):
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=models_resp)
            instance.post = AsyncMock()  # should never be called
            ready = await selftest.wait_for_vllm_ready(timeout_s=0.2)
            # Verify Phase 2 was never even attempted
            instance.post.assert_not_called()
            return ready

    ready = asyncio.run(go())
    assert_eq(ready, False, "returns False on Phase 1 timeout")


def test_wait_for_vllm_ready_keeps_polling_when_chat_503():
    print("\n[test] wait_for_vllm_ready: /v1/models 200 + chat 503 → keep polling")

    models_resp = MagicMock(status_code=200)
    models_resp.json = MagicMock(return_value={"data": [{"id": "x"}]})
    chat_503 = MagicMock(status_code=503)

    async def go():
        with patch("selftest.httpx.AsyncClient") as MockClient, \
             patch.object(selftest, "WAIT_FOR_READY_POLL_INTERVAL_S", 0.01):
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=models_resp)
            instance.post = AsyncMock(return_value=chat_503)
            ready = await selftest.wait_for_vllm_ready(timeout_s=0.2)
            # Phase 2 should have been called multiple times
            assert_true(instance.post.call_count >= 1, "Phase 2 attempted at least once")
            return ready

    ready = asyncio.run(go())
    assert_eq(ready, False, "returns False when chat keeps 503-ing")


def test_wait_for_vllm_ready_succeeds_after_engine_warmup():
    print("\n[test] wait_for_vllm_ready: chat 503 then 200 → ready after warmup")

    models_resp = MagicMock(status_code=200)
    models_resp.json = MagicMock(return_value={"data": [{"id": "x"}]})
    chat_responses = [MagicMock(status_code=503), MagicMock(status_code=200)]

    async def go():
        with patch("selftest.httpx.AsyncClient") as MockClient, \
             patch.object(selftest, "WAIT_FOR_READY_POLL_INTERVAL_S", 0.01):
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=models_resp)
            instance.post = AsyncMock(side_effect=chat_responses)
            return await selftest.wait_for_vllm_ready(timeout_s=5.0)

    ready = asyncio.run(go())
    assert_eq(ready, True, "returns True once engine recovers from 503")


def test_wait_for_vllm_ready_empty_model_list_keeps_polling():
    print("\n[test] wait_for_vllm_ready: /v1/models 200 + empty list → keep polling")

    models_resp = MagicMock(status_code=200)
    models_resp.json = MagicMock(return_value={"data": []})

    async def go():
        with patch("selftest.httpx.AsyncClient") as MockClient, \
             patch.object(selftest, "WAIT_FOR_READY_POLL_INTERVAL_S", 0.01):
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=models_resp)
            instance.post = AsyncMock()
            ready = await selftest.wait_for_vllm_ready(timeout_s=0.2)
            instance.post.assert_not_called()  # never advance to Phase 2
            return ready

    ready = asyncio.run(go())
    assert_eq(ready, False, "empty model list does NOT advance to Phase 2")


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def test_format_report_human_includes_check_names():
    print("\n[test] _format_report_human contains every check name + status")
    report = {
        "status": "pass",
        "checks": [
            {"name": "storage", "ok": True, "latency_ms": 1.2, "detail": "ok"},
            {"name": "vllm_models", "ok": True, "latency_ms": 12.3, "detail": "ok"},
        ],
        "summary": {"passed": 2, "failed": 0, "total": 2},
    }
    out = selftest._format_report_human(report)
    assert_true("storage" in out, "storage line present")
    assert_true("vllm_models" in out, "vllm line present")
    assert_true("PASS" in out, "status line present")
    assert_true("2/2 passed" in out, "summary line present")


def test_format_report_human_marks_failures():
    print("\n[test] _format_report_human shows FAIL for failed checks")
    report = {
        "status": "fail",
        "checks": [
            {"name": "vllm_models", "ok": False, "latency_ms": 3000.0, "detail": "timeout"},
        ],
        "summary": {"passed": 0, "failed": 1, "total": 1},
    }
    out = selftest._format_report_human(report)
    assert_true("FAIL" in out, "FAIL marker present")
    assert_true("timeout" in out, "detail surfaced")


# ---------------------------------------------------------------------------
# CLI entry — exit code mapping
# ---------------------------------------------------------------------------

def test_cli_exits_0_on_pass():
    print("\n[test] CLI main() returns 0 when all checks pass")
    fake_report = {
        "status": "pass",
        "checks": [{"name": "s", "ok": True, "latency_ms": 1.0, "detail": ""}],
        "summary": {"passed": 1, "failed": 0, "total": 1},
    }

    async def fake_run(*, do_round_trip):
        return fake_report

    with patch.object(selftest, "run_selftest", side_effect=fake_run):
        # Capture stdout to avoid polluting test output
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            code = selftest.main(["--no-round-trip"])
        finally:
            sys.stdout = old
        assert_eq(code, 0, "exit 0 on pass")


def test_cli_exits_1_on_fail():
    print("\n[test] CLI main() returns 1 when any check fails")
    fake_report = {
        "status": "fail",
        "checks": [{"name": "s", "ok": False, "latency_ms": 1.0, "detail": "boom"}],
        "summary": {"passed": 0, "failed": 1, "total": 1},
    }

    async def fake_run(*, do_round_trip):
        return fake_report

    with patch.object(selftest, "run_selftest", side_effect=fake_run):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            code = selftest.main(["--no-round-trip"])
        finally:
            sys.stdout = old
        assert_eq(code, 1, "exit 1 on fail")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_facts_round_trip_succeeds_with_real_storage,
        test_facts_round_trip_cleans_up_even_on_inner_failure,
        test_compactor_health_check_200,
        test_compactor_health_check_500,
        test_chat_round_trip_well_formed_response,
        test_chat_round_trip_malformed_response,
        test_admin_localhost_200,
        test_admin_localhost_403_fails,
        test_run_selftest_all_passing,
        test_run_selftest_one_failure_flips_status,
        test_run_selftest_skip_round_trip,
        test_run_selftest_inner_exception_becomes_ok_false,
        test_wait_for_vllm_ready_succeeds_when_both_phases_pass,
        test_wait_for_vllm_ready_keeps_polling_when_models_404,
        test_wait_for_vllm_ready_keeps_polling_when_chat_503,
        test_wait_for_vllm_ready_succeeds_after_engine_warmup,
        test_wait_for_vllm_ready_empty_model_list_keeps_polling,
        test_format_report_human_includes_check_names,
        test_format_report_human_marks_failures,
        test_cli_exits_0_on_pass,
        test_cli_exits_1_on_fail,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll selftest smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
