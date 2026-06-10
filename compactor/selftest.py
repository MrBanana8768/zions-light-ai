"""
compactor.selftest — V2.1 Phase 6 Step 2: live-stack self-test harness.

Two invocation paths:

  1. Auto on boot. Runs as a supervisord one-shot (priority=30, after
     vllm/compactor/openwebui are RUNNING) with `--on-boot --wait-for-ready`.
     If self-test fails it logs FAIL to /var/log/supervisor/selftest.log
     but does NOT take down the pod — operator just sees the failure on
     the next operational check. The boot run is best-effort observability,
     not a gate.

  2. On-demand via GET /admin/selftest (localhost-only). The on-demand
     path skips wait-for-ready (the stack is assumed up) and returns the
     JSON report directly to the caller.

Each check produces:
    {"name": str, "ok": bool, "latency_ms": float, "detail": str}

Aggregate report:
    {
        "status": "pass" | "fail",
        "checks": [<check>, ...],
        "summary": {"passed": int, "failed": int, "total": int}
    }

Would have caught the V2.0 Phase 4.1 Mistral template bug automatically:
the chat_round_trip check uses a fact-state-touching prompt, so once
state populated, the second self-test pass would have hit the HTTP 400
alternation error and flipped to FAIL before the operator noticed.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Callable, Awaitable

import httpx

# These are only needed for the storage + facts checks. Importing at
# module load is fine because selftest.py runs in the same compactor-venv
# as the compactor process and uses the same /opt/compactor source dir.
import facts
import health
import memory

logger = logging.getLogger("compactor.selftest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# In-pod defaults: both compactor + vLLM are reachable on localhost. The
# on-demand /admin/selftest path doesn't pass these in (uses globals).
COMPACTOR_URL = os.environ.get("COMPACTOR_URL", "http://127.0.0.1:8080").rstrip("/")
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_REPO = os.environ.get("MODEL_REPO", "").strip()

# Sentinel conv_id for the facts round-trip. Never touched by real
# traffic, cleaned up at the end of each run.
SELFTEST_CONV_ID = "__selftest__"

# How long to wait for vLLM to come up before giving up.
WAIT_FOR_READY_TIMEOUT_S = float(
    os.environ.get("COMPACTOR_SELFTEST_WAIT_TIMEOUT_S", "600.0")
)
WAIT_FOR_READY_POLL_INTERVAL_S = 5.0

# Round-trip request timeout (real LLM call, can be slow on cold start).
ROUND_TRIP_TIMEOUT_S = float(
    os.environ.get("COMPACTOR_SELFTEST_ROUND_TRIP_TIMEOUT_S", "180.0")
)

# V3.2 — STT (Whisper) service probe. Gated on STT_ENABLED so the check is only
# added when the speech service is actually part of the deployment: the image
# sets STT_ENABLED=true, while unit tests and STT-disabled pods leave it
# unset/false, so run_selftest keeps its original check count. STT_URL/STT_PORT
# are inherited from the container env.
STT_URL = (
    os.environ.get("STT_URL")
    or f"http://127.0.0.1:{os.environ.get('STT_PORT', '9000')}"
).rstrip("/")
STT_ENABLED = os.environ.get("STT_ENABLED", "false").strip().lower() == "true"
STT_TIMEOUT_S = float(os.environ.get("COMPACTOR_SELFTEST_STT_TIMEOUT_S", "30.0"))


# ---------------------------------------------------------------------------
# CheckResult helpers
# ---------------------------------------------------------------------------

def _check(name: str, ok: bool, latency_ms: float, detail: str = "") -> dict:
    return {
        "name": name,
        "ok": bool(ok),
        "latency_ms": round(latency_ms, 1),
        "detail": detail,
    }


async def _timed_async(name: str, fn: Callable[[], Awaitable[tuple[bool, str]]]) -> dict:
    """Run an async check, time it, catch any exception as ok=False."""
    t0 = time.monotonic()
    try:
        ok, detail = await fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    return _check(name, ok, (time.monotonic() - t0) * 1000.0, detail)


def _timed_sync(name: str, fn: Callable[[], tuple[bool, str]]) -> dict:
    """Run a sync check, time it, catch any exception as ok=False."""
    t0 = time.monotonic()
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    return _check(name, ok, (time.monotonic() - t0) * 1000.0, detail)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_storage() -> tuple[bool, str]:
    """Reuse health.probe_storage — single source of truth for "is the
    persistent volume actually writable."
    """
    r = health.probe_storage()
    if r["ok"]:
        return True, f"root={r['root']} free={r.get('free_gb')}GB"
    return False, r.get("error") or "unknown"


async def _check_vllm_models(client: httpx.AsyncClient) -> tuple[bool, str]:
    r = await health.probe_vllm(VLLM_URL)
    if r["ok"]:
        return True, f"models={r['models']}"
    return False, r.get("error") or "unknown"


async def _check_compactor_health(client: httpx.AsyncClient) -> tuple[bool, str]:
    r = await client.get(f"{COMPACTOR_URL}/health", timeout=5.0)
    if r.status_code == 200:
        return True, f"HTTP {r.status_code}"
    return False, f"HTTP {r.status_code}: {r.text[:120]}"


async def _check_chat_round_trip(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Real chat through compactor → vLLM with max_tokens=1. Uses a
    unique conv_id per run so the conversation never accumulates state.
    """
    one_shot_conv = f"__selftest_oneshot_{uuid.uuid4().hex[:8]}__"
    payload = {
        "model": MODEL_REPO or "default",
        "messages": [{"role": "user", "content": "Reply with exactly one word."}],
        "max_tokens": 1,
        "stream": False,
    }
    headers = {"X-Conversation-Id": one_shot_conv}
    r = await client.post(
        f"{COMPACTOR_URL}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=ROUND_TRIP_TIMEOUT_S,
    )
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        return False, f"malformed response: {e}; body={r.text[:200]}"
    # Cleanup: forget this transient conv if possible.
    try:
        await client.delete(
            f"{COMPACTOR_URL}/admin/conversations/{one_shot_conv}/facts",
            timeout=5.0,
        )
    except Exception:
        pass
    return True, f"response_len={len(content)}"


def _check_facts_round_trip() -> tuple[bool, str]:
    """Write a sentinel fact, read it back, delete it. Direct module calls
    (not HTTP) — exercises the storage write path that the async tail uses.
    """
    sentinel_text = f"selftest sentinel {uuid.uuid4().hex[:8]}"
    now = int(time.time())
    try:
        # Start clean (defensive — prior crash may have left state).
        facts.save_facts(SELFTEST_CONV_ID, [])
        # Write
        facts.save_facts(
            SELFTEST_CONV_ID,
            [{"text": sentinel_text, "added_turn": 0, "last_used": now}],
        )
        # Read back
        loaded = facts.load_facts(SELFTEST_CONV_ID)
        if not loaded or loaded[0].get("text") != sentinel_text:
            return False, f"readback mismatch: {loaded!r}"
        return True, f"sentinel='{sentinel_text}'"
    finally:
        # Cleanup — even on failure, leave no junk behind.
        try:
            facts.save_facts(SELFTEST_CONV_ID, [])
        except Exception:
            pass


async def _check_admin_localhost(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Admin endpoint should respond 200 (we're 127.0.0.1)."""
    r = await client.get(
        f"{COMPACTOR_URL}/admin/conversations", timeout=5.0
    )
    if r.status_code == 200:
        return True, f"HTTP {r.status_code}"
    return False, f"HTTP {r.status_code}: {r.text[:120]}"


def _tiny_wav_bytes(seconds: float = 0.3, rate: int = 16000) -> bytes:
    """A short silent mono 16-bit PCM WAV — valid for ffmpeg/Whisper to decode.
    Silence transcribes to empty text; the STT check asserts the response is
    *well-formed* (HTTP 200 + a string `text` field), which proves the service
    decodes audio and runs the model end-to-end — not just that the port is
    open. (Quality, as opposed to liveness, is measured by the tests/eval set.)
    """
    buf = io.BytesIO()
    import wave  # stdlib, only needed here
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(b"\x00\x00" * int(seconds * rate))
    w.close()
    return buf.getvalue()


async def _check_stt(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Functional STT probe: POST a tiny WAV to the Whisper service and assert a
    well-formed OpenAI transcription response. Catches the 'service running but
    broken' failure mode that a port/health check alone would miss. (STT loads a
    small model in seconds; vLLM readiness — which gates the boot self-test —
    takes far longer, so STT is reliably up by the time this runs.)
    """
    files = {"file": ("probe.wav", _tiny_wav_bytes(), "audio/wav")}
    form = {"model": "whisper-1", "response_format": "json"}
    r = await client.post(
        f"{STT_URL}/v1/audio/transcriptions",
        files=files,
        data=form,
        timeout=STT_TIMEOUT_S,
    )
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:160]}"
    try:
        body = r.json()
    except Exception as e:
        return False, f"malformed response: {e}"
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        return False, f"missing 'text' field: {str(body)[:120]}"
    return True, f"transcribed probe ok (text_len={len(body['text'])})"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def wait_for_vllm_ready(timeout_s: float = WAIT_FOR_READY_TIMEOUT_S) -> bool:
    """Two-phase readiness probe — declare vLLM ready only when it can
    actually serve a completion, not just bind a port.

    Phase 1: GET /v1/models 200 with a non-empty model list. Cheap, no
        GPU work — confirms the API server has come up.
    Phase 2: POST /v1/chat/completions with max_tokens=1. Confirms the
        engine is past weight-load + KV-cache init + CUDA graph capture
        and can actually generate. This is the bit that catches the
        boot race: on a cold start, vLLM's API server registers
        /v1/models early but completions return 503 (or hang) until the
        engine finishes loading the model to GPU.

    Without Phase 2, the post-boot self-test's chat_round_trip would
    eventually succeed via its 180s timeout, but flaky one-shot results
    undermine the whole point of the self-test as a deploy canary.
    """
    deadline = time.monotonic() + timeout_s
    models_ready = False
    async with httpx.AsyncClient(timeout=10.0) as c:
        while time.monotonic() < deadline:
            # Phase 1 — API server listing the model
            if not models_ready:
                try:
                    r = await c.get(f"{VLLM_URL}/v1/models")
                    if r.status_code == 200 and (r.json().get("data") or []):
                        models_ready = True
                        logger.info("vLLM /v1/models responding — probing engine readiness")
                except Exception:
                    pass

            # Phase 2 — engine actually completing
            if models_ready:
                try:
                    probe = await c.post(
                        f"{VLLM_URL}/v1/chat/completions",
                        json={
                            "model": MODEL_REPO or "default",
                            "messages": [{"role": "user", "content": "ok"}],
                            "max_tokens": 1,
                            "stream": False,
                        },
                        timeout=30.0,
                    )
                    if probe.status_code == 200:
                        elapsed = timeout_s - (deadline - time.monotonic())
                        logger.info(f"vLLM fully ready (completions live) after {elapsed:.0f}s")
                        return True
                    # 503 / 5xx → engine still warming. Keep polling.
                except Exception:
                    # Network errors during warmup are expected — keep polling.
                    pass

            await asyncio.sleep(WAIT_FOR_READY_POLL_INTERVAL_S)
    logger.warning(f"vLLM did not become ready within {timeout_s}s")
    return False


async def run_selftest(*, do_round_trip: bool = True) -> dict:
    """Execute the full check battery and return a structured report.

    do_round_trip=False skips the chat call — useful for quick smoke tests
    that just want to know storage + endpoints are alive.
    """
    checks: list[dict] = []
    checks.append(_timed_sync("storage", _check_storage))
    checks.append(_timed_sync("facts_round_trip", _check_facts_round_trip))
    async with httpx.AsyncClient() as client:
        checks.append(await _timed_async(
            "vllm_models", lambda: _check_vllm_models(client)
        ))
        checks.append(await _timed_async(
            "compactor_health", lambda: _check_compactor_health(client)
        ))
        checks.append(await _timed_async(
            "admin_localhost", lambda: _check_admin_localhost(client)
        ))
        if STT_ENABLED:
            checks.append(await _timed_async(
                "stt", lambda: _check_stt(client)
            ))
        if do_round_trip:
            checks.append(await _timed_async(
                "chat_round_trip", lambda: _check_chat_round_trip(client)
            ))
    passed = sum(1 for c in checks if c["ok"])
    failed = sum(1 for c in checks if not c["ok"])
    return {
        "status": "pass" if failed == 0 else "fail",
        "checks": checks,
        "summary": {"passed": passed, "failed": failed, "total": len(checks)},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_report_human(report: dict) -> str:
    """Render the report as a readable table for stdout / log file."""
    lines = []
    header = f"=== SELFTEST {report['status'].upper()} ==="
    lines.append(header)
    width_name = max((len(c["name"]) for c in report["checks"]), default=20)
    for c in report["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        lines.append(
            f"  [{mark}] {c['name']:<{width_name}}  {c['latency_ms']:>8.1f}ms  {c['detail']}"
        )
    s = report["summary"]
    lines.append(f"=== {s['passed']}/{s['total']} passed, {s['failed']} failed ===")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live-stack self-test for the Zion's Light AI compactor.",
    )
    parser.add_argument(
        "--on-boot",
        action="store_true",
        help="Mark this run as the post-boot one-shot (cosmetic — affects log labeling).",
    )
    parser.add_argument(
        "--wait-for-ready",
        action="store_true",
        help="Poll vLLM /v1/models until ready before running the chat round-trip "
             "(default timeout: 600s). Use on cold boot where the model may still "
             "be loading.",
    )
    parser.add_argument(
        "--no-round-trip",
        action="store_true",
        help="Skip the real LLM round-trip check. Useful for quick smoke checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw JSON report on stdout instead of the human table.",
    )
    args = parser.parse_args(argv)

    import logsetup
    logsetup.configure()  # honors COMPACTOR_LOG_FORMAT (text/json)

    label = "ON-BOOT" if args.on_boot else "ON-DEMAND"
    logger.info(f"selftest starting ({label})")

    async def _go() -> dict:
        if args.wait_for_ready:
            await wait_for_vllm_ready()
        return await run_selftest(do_round_trip=not args.no_round_trip)

    report = asyncio.run(_go())

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report_human(report))

    # V2.3 Theme 4: alert on failure (no-op unless COMPACTOR_ALERT_WEBHOOK set).
    if report["status"] != "pass":
        failed = [c["name"] for c in report.get("checks", []) if not c["ok"]]
        try:
            import alert
            alert.notify(
                "selftest", "fail",
                f"{label} self-test failed: {', '.join(failed) or 'unknown'}",
                extra={"summary": report.get("summary")},
            )
        except Exception:
            pass

    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
