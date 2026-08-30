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
from pathlib import Path
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

# V4: if the compactor's API-key gate is enabled (apiauth), the self-test's own
# /v1 chat round-trip must carry the key or it'd get a 401. Unset = no header
# (the current single-container deploy, where auth is off).
COMPACTOR_API_KEY = os.environ.get("COMPACTOR_API_KEY", "").strip()

# Sentinel conv_id for the facts round-trip. Never touched by real
# traffic, cleaned up at the end of each run.
SELFTEST_CONV_ID = "__selftest__"

# How long to wait for the compactor's post-response background work to finish
# before cleaning up the one-shot conversation. The tail is submitted BEFORE
# the response is returned, so it is already outstanding by the time we read
# the body — but it makes an LLM extraction call, so the old
# delete-immediately cleanup always won the race and the tail then re-created
# the file. That left one orphaned conversation per boot, forever: 17 of the
# 124 buckets in the 2026-08-24 store were __selftest_oneshot_* (v3.1 F23).
SELFTEST_TAIL_DRAIN_TIMEOUT_S = float(
    os.environ.get("COMPACTOR_SELFTEST_TAIL_DRAIN_TIMEOUT_S", "120.0")
)
SELFTEST_TAIL_DRAIN_POLL_S = 1.0

# After the cleanup, look once more. The drain is a good signal on a quiet pod
# and a weak one on a busy pod (/admin/selftest runs the same battery in-
# process while real traffic is in flight), so the thing actually asserted is
# that the conversation is gone AND stays gone — which catches a late tail
# write however the drain went.
SELFTEST_CLEANUP_SETTLE_S = float(
    os.environ.get("COMPACTOR_SELFTEST_CLEANUP_SETTLE_S", "2.0")
)

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

# V3.3 — TTS (Piper) service probe. Same gating discipline as STT: only added
# when the speech-output service is part of the deployment (image sets
# TTS_ENABLED=true; unit tests / TTS-disabled pods leave it unset/false).
TTS_URL = (
    os.environ.get("TTS_URL")
    or f"http://127.0.0.1:{os.environ.get('TTS_PORT', '9001')}"
).rstrip("/")
TTS_ENABLED = os.environ.get("TTS_ENABLED", "false").strip().lower() == "true"
TTS_TIMEOUT_S = float(os.environ.get("COMPACTOR_SELFTEST_TTS_TIMEOUT_S", "30.0"))

# Spoken by the TTS service to make the STT probe audio. It has to be long
# enough and speech-like enough to survive Whisper's VAD filter, which is what
# made the old probe vacuous — see _speech_probe_wav.
STT_PROBE_TEXT = os.environ.get(
    "COMPACTOR_SELFTEST_STT_PROBE_TEXT",
    "The quick brown fox jumps over the lazy dog.",
).strip()


# ---------------------------------------------------------------------------
# Leave-nothing-behind cleanup
# ---------------------------------------------------------------------------
#
# Every self-test conversation is scratch. It must not survive the run: a boot
# that mints a bucket and leaves it turns the store into mostly test residue
# (2026-08-24: 124 buckets, only 33 of them real conversations), and every one
# of those buckets is then walked by the O(N) health scan and listed to anyone
# inspecting the store during an incident.

def _conv_artifact_paths(conv_id: str) -> list[Path]:
    """Every file a conversation can leave on disk.

    Kept in one place because the layers are spread across four modules and a
    cleanup that misses one is indistinguishable from a cleanup that worked.
    The backfill sidecar is built here rather than imported from backfill.py —
    importing that module for a path would pull the lazy-backfill machinery
    into the self-test process for no reason.
    """
    return [
        memory.facts_path(conv_id),
        memory.facts_archive_path(conv_id),
        memory.summary_path(conv_id),
        memory.persona_path(conv_id),
        memory.storage_root() / "facts" / f"{conv_id}.backfill.json",
    ]


def _conv_residue(conv_id: str) -> list[str]:
    """Names of the conversation's artifact files that exist right now."""
    out: list[str] = []
    for p in _conv_artifact_paths(conv_id):
        try:
            if p.exists():
                out.append(p.name)
        except Exception as e:
            # A path we cannot even stat is a path we cannot vouch for.
            logger.warning(f"selftest cleanup could not stat {p}: {e}")
            out.append(p.name)
    return out


def _purge_conv_files(conv_id: str) -> list[str]:
    """Unlink every artifact file for `conv_id`. Returns the names that are
    still there afterwards — empty means the store is clean.

    Unlink, not `save_facts(conv_id, [])`: an empty facts file is still a file,
    and `memory.list_known_conv_ids` globs `facts/*.json`, so an emptied
    conversation is counted forever. That is how `facts/__selftest__.json`
    became a permanent resident (v3.1 F23 / D10).
    """
    for p in _conv_artifact_paths(conv_id):
        try:
            p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"selftest cleanup could not remove {p}: {e}")
    return _conv_residue(conv_id)


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


async def _wait_for_tail_drain(client: httpx.AsyncClient) -> tuple[str, int | None]:
    """Block until the compactor's background pool is empty.

    Returns ("drained"|"timeout"|"unobservable", outstanding). The pool count
    comes from /health/full's `background_work` block (bgwork.pool.stats), the
    only outside view of whether the post-response tail has finished. Boot is
    the quiet moment by definition — the self-test's own chat is normally the
    only thing in flight — so an empty pool means our tail is done.
    """
    deadline = time.monotonic() + SELFTEST_TAIL_DRAIN_TIMEOUT_S
    outstanding: int | None = None
    while time.monotonic() < deadline:
        try:
            r = await client.get(f"{COMPACTOR_URL}/health/full", timeout=10.0)
            bg = (r.json() or {}).get("background_work") or {}
        except Exception as e:
            logger.debug(f"tail-drain poll failed: {e}")
            await asyncio.sleep(SELFTEST_TAIL_DRAIN_POLL_S)
            continue
        if "outstanding" not in bg:
            # The pool could not report (bgwork import failed, older shape).
            # Waiting out the full timeout would delay every boot for a number
            # we are never going to see, so stop and say we could not look.
            logger.warning(
                f"background pool depth not reported by /health/full "
                f"({str(bg)[:120]}); cleaning up without waiting for the tail"
            )
            return "unobservable", None
        outstanding = bg["outstanding"]
        if outstanding == 0:
            return "drained", 0
        await asyncio.sleep(SELFTEST_TAIL_DRAIN_POLL_S)
    logger.warning(
        f"background pool still had {outstanding} task(s) outstanding after "
        f"{SELFTEST_TAIL_DRAIN_TIMEOUT_S}s"
    )
    return "timeout", outstanding


async def _check_chat_round_trip(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Real chat through compactor → vLLM with max_tokens=1, followed by a
    cleanup that actually holds.

    Uses a unique conv_id per run, then removes it. The order matters and is
    the whole fix for F23: wait for the async tail to finish, THEN delete.
    Deleting first — what this did — lost the race every time, because the
    tail is fired before the response returns and takes an LLM call, so it
    landed after the delete and wrote the conversation back. The DELETE is
    still what clears ChromaDB (the compactor process owns that client; a
    second process must not open it), and the unlink afterwards removes the
    empty facts file the DELETE leaves behind.
    """
    one_shot_conv = f"__selftest_oneshot_{uuid.uuid4().hex[:8]}__"
    payload = {
        "model": MODEL_REPO or "default",
        "messages": [{"role": "user", "content": "Reply with exactly one word."}],
        "max_tokens": 1,
        "stream": False,
    }
    headers = {"X-Conversation-Id": one_shot_conv}
    if COMPACTOR_API_KEY:
        headers["Authorization"] = f"Bearer {COMPACTOR_API_KEY}"
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

    drain, outstanding = await _wait_for_tail_drain(client)
    try:
        await client.delete(
            f"{COMPACTOR_URL}/admin/conversations/{one_shot_conv}/facts",
            timeout=30.0,
        )
    except Exception as e:
        # The unlink below still removes the on-disk layers; only ChromaDB
        # needs the compactor process, so say which layer is in doubt.
        logger.warning(
            f"one-shot cleanup of {one_shot_conv}: admin forget failed ({e}); "
            f"episodic embeddings may remain"
        )
    left = _purge_conv_files(one_shot_conv)
    # Then look again. A drain timeout is not itself reported as a failure —
    # on-demand runs share the pool with live traffic, so a busy pool is
    # normal there — but a file that comes BACK after the purge is the late
    # tail write this whole sequence exists to prevent, and that is a failure
    # whatever the drain said.
    await asyncio.sleep(SELFTEST_CLEANUP_SETTLE_S)
    reappeared = _conv_residue(one_shot_conv)
    if reappeared:
        # Don't leave the bucket behind while reporting that it was left behind.
        _purge_conv_files(one_shot_conv)
    if left or reappeared:
        return False, (
            f"response_len={len(content)}; one-shot conversation survived "
            f"cleanup (drain={drain}, outstanding={outstanding}, "
            f"unremovable={left or '-'}, rewritten={reappeared or '-'}) — "
            f"every boot leaks a bucket"
        )
    return True, f"response_len={len(content)} cleanup={drain} store_left_clean=yes"


def _check_facts_round_trip() -> tuple[bool, str]:
    """Write a sentinel fact, read it back, delete it. Direct module calls
    (not HTTP) — exercises the storage write path that the async tail uses.

    Both the start-clean and the cleanup unlink rather than writing an empty
    facts list: the old `save_facts(SELFTEST_CONV_ID, [])` on either side left
    `facts/__selftest__.json` in place, and an existing-but-empty file is a
    listed conversation as far as list_known_conv_ids is concerned. The check
    now fails if anything survives, so a regression here is visible in the boot
    report instead of only in the bucket count months later.
    """
    sentinel_text = f"selftest sentinel {uuid.uuid4().hex[:8]}"
    now = int(time.time())
    try:
        # Start clean (defensive — a prior crash may have left state).
        _purge_conv_files(SELFTEST_CONV_ID)
        # Write
        facts.save_facts(
            SELFTEST_CONV_ID,
            [{"text": sentinel_text, "added_turn": 0, "last_used": now}],
        )
        # Read back
        loaded = facts.load_facts(SELFTEST_CONV_ID)
        if not loaded or loaded[0].get("text") != sentinel_text:
            return False, f"readback mismatch: {loaded!r}"
    finally:
        # Cleanup — even on failure, leave no junk behind.
        left = _purge_conv_files(SELFTEST_CONV_ID)
    if left:
        return False, f"sentinel files survived cleanup: {', '.join(left)}"
    return True, f"sentinel='{sentinel_text}' store_left_clean=yes"


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

    Fallback probe only. It cannot assert a transcription: the STT service runs
    with WHISPER_VAD_FILTER on, and VAD drops the whole clip, so a healthy
    service answers `{"text": ""}`. Measured in the v3.0.5 image: this exact
    0.3s probe is 9,644 bytes in and 0 chars out, which is what production was
    recording as `[PASS] stt ... text_len=0`. A synthetic tone does not help —
    a 1s 440Hz sine measured the same way also transcribes to "" — so there is
    no purely-local audio that survives VAD, and the real probe comes from the
    TTS service instead (_speech_probe_wav).
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


async def _speech_probe_wav(client: httpx.AsyncClient) -> bytes | None:
    """Render STT_PROBE_TEXT to real speech with the Piper service, for use as
    the STT probe. Returns None if TTS is not part of this deployment or the
    synthesis did not produce audio.

    Piper is the only speech source available at boot: the compactor venv has
    no TTS engine, and shipping a recorded clip in the repo would fix the voice
    and the language into the image. tts runs at priority=26 and selftest at
    30, so it is already up when this fires. A None here is not reported as an
    STT failure — the tts check owns that — it just costs the STT check its
    transcription assertion, which _check_stt says out loud.
    """
    if not TTS_ENABLED:
        return None
    try:
        r = await client.post(
            f"{TTS_URL}/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": STT_PROBE_TEXT,
                "response_format": "wav",
            },
            timeout=TTS_TIMEOUT_S,
        )
    except Exception as e:
        logger.warning(f"STT speech probe: TTS synthesis failed: {e}")
        return None
    if r.status_code != 200 or not r.content:
        logger.warning(
            f"STT speech probe: TTS returned HTTP {r.status_code} "
            f"({len(r.content)} bytes)"
        )
        return None
    if not r.headers.get("content-type", "").startswith("audio/"):
        logger.warning(
            f"STT speech probe: TTS returned non-audio content-type "
            f"{r.headers.get('content-type')!r}"
        )
        return None
    return r.content


async def _check_stt(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Functional STT probe: speak a known sentence through the TTS service,
    POST that audio to the Whisper service, and assert it comes back as text.

    The assertion is what changed in v3.1. This check used to POST 0.3s of
    silence and assert only that the reply was well-formed, so it passed while
    stt-error.log recorded `VAD filter removed 00:00.300 of audio` and
    `9644 bytes -> 0 chars` — the probe never reached the model, and the check
    could not have told anyone. A pass now means audio went in and words came
    out. Content is deliberately not asserted (the voice and the model are both
    swappable per pod); the transcript is in the detail so a garbage one is
    readable in the boot log.
    """
    speech = await _speech_probe_wav(client)
    files = {"file": ("probe.wav", speech or _tiny_wav_bytes(), "audio/wav")}
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
    text = body["text"].strip()
    if speech is None:
        # No speech to send, so no transcription to demand. Liveness only —
        # say so rather than reporting a text_len=0 pass as if it meant
        # something.
        return True, (
            f"liveness only: no speech probe available "
            f"(TTS_ENABLED={str(TTS_ENABLED).lower()}), so a silent clip was "
            f"sent and only the response shape was checked; text_len={len(text)}"
        )
    if not text:
        return False, (
            f"spoke {len(speech)} bytes of synthesized audio and got an empty "
            f"transcription — the service answered, but nothing reached the "
            f"model (VAD, decode or weights)"
        )
    return True, f"transcribed spoken probe: {text[:80]!r}"


async def _check_tts(client: httpx.AsyncClient) -> tuple[bool, str]:
    """Functional TTS probe: POST a tiny text to the Piper service and assert it
    returns non-empty audio with an audio/* content type. Catches the 'service
    running but broken' failure a port/health check alone would miss."""
    r = await client.post(
        f"{TTS_URL}/v1/audio/speech",
        json={"model": "tts-1", "input": "ok", "response_format": "wav"},
        timeout=TTS_TIMEOUT_S,
    )
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:160]}"
    ctype = r.headers.get("content-type", "")
    if not ctype.startswith("audio/"):
        return False, f"non-audio content-type: {ctype!r}"
    if not r.content:
        return False, "empty audio body"
    return True, f"synthesized probe ok (bytes={len(r.content)}, {ctype})"


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
                except Exception as e:
                    # Expected while vLLM is still binding its port.
                    logger.debug(f"vLLM /v1/models not up yet: {e}")

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
                except Exception as e:
                    # Network errors during warmup are expected — keep polling.
                    logger.debug(f"vLLM completion probe not ready yet: {e}")

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
        if TTS_ENABLED:
            checks.append(await _timed_async(
                "tts", lambda: _check_tts(client)
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
        except Exception as e:
            # The report is already on stdout and the exit code is nonzero,
            # so the failure itself is not lost — only the notification is.
            logger.debug(f"selftest failure alert not sent: {e}")

    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
