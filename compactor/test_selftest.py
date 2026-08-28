"""
CPU-only Tier-1 tests for compactor.selftest.

Mocks httpx for the compactor/vLLM HTTP probes; uses real tmpdir
storage for the facts round-trip check (verifying the actual write
path). Verifies:
  - run_selftest() aggregates correctly (pass/fail status)
  - individual checks degrade to ok=False rather than raising
  - report shape matches the documented schema
  - facts_round_trip cleans up the sentinel even on failure
  - a self-test run leaves NO conversation behind (v3.1 F23)
  - the STT probe asserts a transcription, not just a 200 (v3.1)
  - CLI flag parsing + report rendering

Run: python test_selftest.py
"""

import asyncio
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path
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
# Leave-nothing-behind cleanup (v3.1 F23)
# ---------------------------------------------------------------------------

def _write_every_layer(conv_id):
    """Put one file in every storage layer a conversation can occupy."""
    memory.ensure_storage_layout()
    for p in selftest._conv_artifact_paths(conv_id):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")


def test_purge_conv_files_removes_every_layer():
    print("\n[test] _purge_conv_files removes facts, archive, summary, persona, backfill")
    conv = "__purge_probe__"
    _write_every_layer(conv)
    paths = selftest._conv_artifact_paths(conv)
    assert_eq(len(paths), 5, "five artifact paths tracked")
    assert_true(all(p.is_file() for p in paths), "all five written")
    left = selftest._purge_conv_files(conv)
    assert_eq(left, [], "nothing reported as surviving")
    assert_true(not any(p.exists() for p in paths), "all five gone from disk")


def test_conv_residue_lists_only_what_exists():
    print("\n[test] _conv_residue names the artifact files that are present")
    conv = "__residue_probe__"
    memory.ensure_storage_layout()
    assert_eq(selftest._conv_residue(conv), [], "clean store reports nothing")
    memory.facts_path(conv).write_text("{}", encoding="utf-8")
    assert_eq(selftest._conv_residue(conv), [f"{conv}.json"], "the facts file is named")
    selftest._purge_conv_files(conv)


def test_purge_conv_files_reports_what_it_could_not_remove():
    print("\n[test] _purge_conv_files reports survivors instead of claiming success")
    conv = "__purge_survivor__"
    memory.ensure_storage_layout()
    # A directory cannot be unlink()ed — a stand-in for any path the purge
    # cannot clear (permissions, a stalled mount).
    stuck = Path(_TMP_ROOT) / "stuck_dir"
    stuck.mkdir(exist_ok=True)
    with patch.object(selftest, "_conv_artifact_paths", lambda c: [stuck]):
        left = selftest._purge_conv_files(conv)
    assert_eq(left, ["stuck_dir"], "survivor named in the return value")


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


def test_facts_round_trip_leaves_no_file_at_all():
    print("\n[test] _check_facts_round_trip unlinks the sentinel, not empties it")
    memory.ensure_storage_layout()
    ok, _ = selftest._check_facts_round_trip()
    assert_eq(ok, True, "round-trip ok")
    # The regression this guards: save_facts(conv, []) left an existing-but-
    # empty facts/__selftest__.json, which list_known_conv_ids counts forever.
    assert_eq(
        memory.facts_path(selftest.SELFTEST_CONV_ID).exists(), False,
        "facts/__selftest__.json does not exist",
    )
    assert_true(
        selftest.SELFTEST_CONV_ID not in memory.list_known_conv_ids(),
        "sentinel is not a listed conversation",
    )


def test_facts_round_trip_fails_when_cleanup_leaves_something():
    print("\n[test] _check_facts_round_trip fails if an artifact survives cleanup")
    memory.ensure_storage_layout()
    stuck = Path(_TMP_ROOT) / "stuck_dir_facts"
    stuck.mkdir(exist_ok=True)
    real = memory.facts_path(selftest.SELFTEST_CONV_ID)
    # Real facts path still purged (so this test leaves the store clean); the
    # extra unremovable path is what the check has to notice.
    with patch.object(selftest, "_conv_artifact_paths", lambda c: [real, stuck]):
        ok, detail = selftest._check_facts_round_trip()
    assert_eq(ok, False, "ok=False when an artifact survives")
    assert_true("stuck_dir_facts" in detail, "detail names the survivor")
    assert_eq(real.exists(), False, "the real sentinel file was still removed")


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
    assert_eq(
        memory.facts_path(selftest.SELFTEST_CONV_ID).exists(), False,
        "pollution file removed, not emptied",
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


def _chat_client(chat_body, *, outstanding=(0,), calls=None):
    """A mock httpx client for the chat round-trip: POST returns a completion,
    GET /health/full reports the background pool depth (one entry per poll,
    last value repeats), DELETE succeeds. `calls` collects the call order.
    """
    client = MagicMock()
    log = calls if calls is not None else []
    depths = list(outstanding)

    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=chat_body)
    resp.text = str(chat_body)

    async def _post(url, **kw):
        log.append(("post", url))
        return resp

    async def _get(url, **kw):
        log.append(("get", url))
        depth = depths.pop(0) if len(depths) > 1 else depths[0]
        health_resp = MagicMock(status_code=200)
        health_resp.json = MagicMock(
            return_value={"background_work": {"outstanding": depth}}
        )
        return health_resp

    async def _delete(url, **kw):
        log.append(("delete", url))
        return MagicMock(status_code=200)

    client.post = AsyncMock(side_effect=_post)
    client.get = AsyncMock(side_effect=_get)
    client.delete = AsyncMock(side_effect=_delete)
    return client


def test_chat_round_trip_well_formed_response():
    print("\n[test] _check_chat_round_trip extracts content from valid response")

    async def go():
        client = _chat_client({"choices": [{"message": {"content": "Hi"}}]})
        with patch.object(selftest, "SELFTEST_CLEANUP_SETTLE_S", 0.0):
            return await selftest._check_chat_round_trip(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, True, "ok=True on valid completion")
    assert_true("response_len=" in detail, "reports response length")
    assert_true("store_left_clean=yes" in detail, "reports the store is clean")


def test_chat_round_trip_malformed_response():
    print("\n[test] _check_chat_round_trip fails gracefully on malformed body")

    async def go():
        client = _chat_client({"wrong_shape": True})
        return await selftest._check_chat_round_trip(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False on malformed response")
    assert_true("malformed" in detail.lower(), "detail explains the failure")


def test_chat_round_trip_waits_for_the_tail_before_deleting():
    print("\n[test] _check_chat_round_trip drains the tail BEFORE the delete (F23)")
    calls = []

    async def go():
        # Pool is busy for the first two polls, then empty.
        client = _chat_client(
            {"choices": [{"message": {"content": "Hi"}}]},
            outstanding=(1, 1, 0),
            calls=calls,
        )
        with patch.object(selftest, "SELFTEST_TAIL_DRAIN_POLL_S", 0.01), \
             patch.object(selftest, "SELFTEST_CLEANUP_SETTLE_S", 0.0):
            return await selftest._check_chat_round_trip(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, True, "ok=True")
    kinds = [k for k, _ in calls]
    assert_eq(kinds[0], "post", "chat is posted first")
    assert_true("get" in kinds, "the background pool was polled")
    assert_eq(kinds[-1], "delete", "the delete happens last, after the drain")
    assert_true(
        kinds.index("delete") > max(i for i, k in enumerate(kinds) if k == "get"),
        "every pool poll precedes the delete",
    )
    assert_true("cleanup=drained" in detail, "detail records that the tail drained")


def test_chat_round_trip_fails_when_the_conversation_survives():
    print("\n[test] _check_chat_round_trip fails when the one-shot conv is left behind")

    async def go():
        client = _chat_client({"choices": [{"message": {"content": "Hi"}}]})
        with patch.object(selftest, "_purge_conv_files",
                          return_value=["__selftest_oneshot_abc__.json"]), \
             patch.object(selftest, "_conv_residue", return_value=[]), \
             patch.object(selftest, "SELFTEST_CLEANUP_SETTLE_S", 0.0):
            return await selftest._check_chat_round_trip(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False when a bucket is leaked")
    assert_true("survived cleanup" in detail, "detail says the conv survived")
    assert_true("unremovable=" in detail, "detail distinguishes an unremovable file")


def test_chat_round_trip_reports_but_survives_a_busy_pool():
    print("\n[test] _check_chat_round_trip: drain timeout is reported, not failed")

    async def go():
        # /admin/selftest runs this battery in-process on a live pod, where the
        # background pool belongs to real traffic too. A pool that never empties
        # is not by itself evidence that this conversation leaked.
        client = _chat_client(
            {"choices": [{"message": {"content": "Hi"}}]}, outstanding=(3,)
        )
        with patch.object(selftest, "SELFTEST_TAIL_DRAIN_TIMEOUT_S", 0.05), \
             patch.object(selftest, "SELFTEST_TAIL_DRAIN_POLL_S", 0.01), \
             patch.object(selftest, "SELFTEST_CLEANUP_SETTLE_S", 0.0):
            return await selftest._check_chat_round_trip(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, True, "ok=True — nothing was actually left behind")
    assert_true("cleanup=timeout" in detail, "the busy pool is stated in the detail")


def test_chat_round_trip_fails_when_the_tail_writes_the_conv_back():
    print("\n[test] _check_chat_round_trip fails when a late tail re-creates the conv")

    async def go():
        client = _chat_client({"choices": [{"message": {"content": "Hi"}}]})
        # Clean right after the purge, then the tail's write lands: exactly the
        # F23 sequence, which a purge-and-trust cleanup cannot see.
        with patch.object(selftest, "_conv_residue",
                          side_effect=[[], ["__selftest_oneshot_x__.json"], []]), \
             patch.object(selftest, "SELFTEST_CLEANUP_SETTLE_S", 0.0):
            return await selftest._check_chat_round_trip(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False when the conversation comes back")
    assert_true("rewritten=" in detail, "detail distinguishes a late write")
    assert_true("survived cleanup" in detail, "detail says the conv survived")


# ---------------------------------------------------------------------------
# _wait_for_tail_drain
# ---------------------------------------------------------------------------

def test_wait_for_tail_drain_returns_drained_on_empty_pool():
    print("\n[test] _wait_for_tail_drain: outstanding=0 → drained")

    async def go():
        client = _chat_client({}, outstanding=(0,))
        return await selftest._wait_for_tail_drain(client)

    state, outstanding = asyncio.run(go())
    assert_eq(state, "drained", "state=drained")
    assert_eq(outstanding, 0, "outstanding=0")


def test_wait_for_tail_drain_times_out_on_busy_pool():
    print("\n[test] _wait_for_tail_drain: pool never empties → timeout")

    async def go():
        client = _chat_client({}, outstanding=(2,))
        with patch.object(selftest, "SELFTEST_TAIL_DRAIN_TIMEOUT_S", 0.05), \
             patch.object(selftest, "SELFTEST_TAIL_DRAIN_POLL_S", 0.01):
            return await selftest._wait_for_tail_drain(client)

    state, outstanding = asyncio.run(go())
    assert_eq(state, "timeout", "state=timeout")
    assert_eq(outstanding, 2, "last observed depth reported")


def test_wait_for_tail_drain_gives_up_immediately_when_depth_unreported():
    print("\n[test] _wait_for_tail_drain: no pool stats → unobservable, no waiting")

    async def go():
        client = MagicMock()
        resp = MagicMock(status_code=200)
        resp.json = MagicMock(
            return_value={"background_work": {"error": "ImportError: bgwork"}}
        )
        client.get = AsyncMock(return_value=resp)
        # A long timeout: the point is that it returns without burning it.
        with patch.object(selftest, "SELFTEST_TAIL_DRAIN_TIMEOUT_S", 300.0):
            out = await selftest._wait_for_tail_drain(client)
        return out, client.get.call_count

    (state, outstanding), n_polls = asyncio.run(go())
    assert_eq(state, "unobservable", "state=unobservable")
    assert_eq(outstanding, None, "no depth to report")
    assert_eq(n_polls, 1, "polled once, did not wait out the timeout")


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
# STT functional probe (V3.2)
# ---------------------------------------------------------------------------

def _speech_client(*, tts=None, stt=None):
    """Mock client that answers the TTS synthesis POST and the STT
    transcription POST differently, dispatching on the URL path.

    tts: (status_code, content, content_type) or None for "TTS not consulted"
    stt: (status_code, json_body) — json_body may be any object
    """
    client = MagicMock()

    async def _post(url, **kw):
        resp = MagicMock()
        if "/audio/speech" in url:
            code, content, ctype = tts
            resp.status_code = code
            resp.content = content
            resp.headers = {"content-type": ctype}
            resp.text = "tts"
            return resp
        code, body = stt
        resp.status_code = code
        resp.json = MagicMock(return_value=body)
        resp.text = str(body)
        return resp

    client.post = AsyncMock(side_effect=_post)
    return client


def test_stt_asserts_a_real_transcription_of_synthesized_speech():
    print("\n[test] _check_stt: speaks a probe via TTS and asserts the text")

    async def go():
        client = _speech_client(
            tts=(200, b"RIFF....WAVE....", "audio/wav"),
            stt=(200, {"text": " The quick brown fox. "}),
        )
        with patch.object(selftest, "TTS_ENABLED", True):
            out = await selftest._check_stt(client)
        return out, client.post.call_count

    (ok, detail), n_posts = asyncio.run(go())
    assert_eq(ok, True, "ok=True on a real transcription")
    assert_eq(n_posts, 2, "synthesized the probe, then transcribed it")
    assert_true("quick brown fox" in detail, "the transcript is in the detail")


def test_stt_fails_when_the_spoken_probe_transcribes_to_nothing():
    print("\n[test] _check_stt: speech in, empty text out → FAIL (the v3.0.5 bug)")

    async def go():
        # Exactly the production shape: HTTP 200, well-formed body, no words.
        client = _speech_client(
            tts=(200, b"RIFF....WAVE....", "audio/wav"),
            stt=(200, {"text": ""}),
        )
        with patch.object(selftest, "TTS_ENABLED", True):
            return await selftest._check_stt(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "an empty transcription of real speech is a failure")
    assert_true("empty transcription" in detail, "detail names the failure")


def test_stt_is_liveness_only_and_says_so_when_tts_is_disabled():
    print("\n[test] _check_stt: no TTS → silent probe, liveness only, stated")

    async def go():
        client = _speech_client(stt=(200, {"text": ""}))
        with patch.object(selftest, "TTS_ENABLED", False):
            out = await selftest._check_stt(client)
        return out, client.post.call_count

    (ok, detail), n_posts = asyncio.run(go())
    assert_eq(ok, True, "still a pass — TTS-off pods are a valid deployment")
    assert_eq(n_posts, 1, "TTS was never consulted")
    assert_true("liveness only" in detail, "the weaker assertion is stated")


def test_stt_falls_back_to_liveness_when_tts_synthesis_fails():
    print("\n[test] _check_stt: TTS enabled but broken → liveness only, not a fake pass")

    async def go():
        client = _speech_client(
            tts=(503, b"", "application/json"),
            stt=(200, {"text": ""}),
        )
        with patch.object(selftest, "TTS_ENABLED", True):
            return await selftest._check_stt(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, True, "the tts check owns a TTS failure, not this one")
    assert_true("liveness only" in detail, "the degraded assertion is stated")


def test_speech_probe_rejects_non_audio_body():
    print("\n[test] _speech_probe_wav: TTS returns JSON → None, no probe")

    async def go():
        client = _speech_client(tts=(200, b'{"error":"x"}', "application/json"))
        with patch.object(selftest, "TTS_ENABLED", True):
            return await selftest._speech_probe_wav(client)

    assert_eq(asyncio.run(go()), None, "non-audio content-type yields no probe")


def test_stt_check_503_fails():
    print("\n[test] _check_stt: 503 (model loading) → fail")

    async def go():
        client = _speech_client(
            tts=(200, b"RIFF....WAVE....", "audio/wav"),
            stt=(503, {"error": "model not ready"}),
        )
        with patch.object(selftest, "TTS_ENABLED", True):
            return await selftest._check_stt(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False on 503")
    assert_true("503" in detail, "detail mentions code")


def test_stt_check_malformed_fails():
    print("\n[test] _check_stt: 200 but no 'text' field → fail")

    async def go():
        client = _speech_client(
            tts=(200, b"RIFF....WAVE....", "audio/wav"),
            stt=(200, {"oops": 1}),
        )
        with patch.object(selftest, "TTS_ENABLED", True):
            return await selftest._check_stt(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False when 'text' missing")
    assert_true("text" in detail.lower(), "detail explains missing field")


def test_tiny_wav_is_a_decodable_wav():
    print("\n[test] _tiny_wav_bytes produces a real mono 16-bit WAV")
    import wave
    raw = selftest._tiny_wav_bytes()
    with wave.open(io.BytesIO(raw), "rb") as w:
        assert_eq(w.getnchannels(), 1, "mono")
        assert_eq(w.getsampwidth(), 2, "16-bit")
        assert_eq(w.getframerate(), 16000, "16 kHz")
        assert_true(w.getnframes() > 0, "non-empty")


def test_run_selftest_includes_stt_when_enabled():
    print("\n[test] run_selftest adds the stt check only when STT_ENABLED")

    async def go():
        # Both flags pinned. They come from the container environment, and the
        # image sets STT_ENABLED=true and TTS_ENABLED=true — so a check count
        # that leaned on the ambient value passed in a bare venv and failed in
        # the real image, which is where this suite is supposed to be run.
        with patch.object(selftest, "STT_ENABLED", True), \
             patch.object(selftest, "TTS_ENABLED", False), \
             patch.object(selftest, "_check_vllm_models",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_compactor_health",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_admin_localhost",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_stt",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_chat_round_trip",
                          new=AsyncMock(return_value=(True, "ok"))):
            return await selftest.run_selftest(do_round_trip=True)

    report = asyncio.run(go())
    names = [c["name"] for c in report["checks"]]
    assert_true("stt" in names, "stt check present when enabled")
    assert_true("tts" not in names, "tts check absent when disabled")
    assert_eq(report["summary"]["total"], 7, "7 checks (6 core + stt)")


# ---------------------------------------------------------------------------
# TTS functional probe (V3.3)
# ---------------------------------------------------------------------------

def test_tts_check_200_audio():
    print("\n[test] _check_tts: 200 + audio/* + bytes → ok")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "audio/wav"}
        resp.content = b"RIFF....WAVE...."
        client.post = AsyncMock(return_value=resp)
        return await selftest._check_tts(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, True, "ok=True on well-formed audio response")
    assert_true("bytes=" in detail, "reports audio byte count")


def test_tts_check_non_audio_content_type_fails():
    print("\n[test] _check_tts: 200 but content-type not audio/* → fail")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        resp.content = b'{"error":"x"}'
        client.post = AsyncMock(return_value=resp)
        return await selftest._check_tts(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False on non-audio content type")
    assert_true("content-type" in detail.lower(), "detail explains the mismatch")


def test_tts_check_503_fails():
    print("\n[test] _check_tts: 503 (voice loading) → fail")

    async def go():
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 503
        resp.text = "voice not ready"
        client.post = AsyncMock(return_value=resp)
        return await selftest._check_tts(client)

    ok, detail = asyncio.run(go())
    assert_eq(ok, False, "ok=False on 503")
    assert_true("503" in detail, "detail mentions code")


def test_run_selftest_includes_tts_when_enabled():
    print("\n[test] run_selftest adds the tts check only when TTS_ENABLED")

    async def go():
        with patch.object(selftest, "TTS_ENABLED", True), \
             patch.object(selftest, "STT_ENABLED", False), \
             patch.object(selftest, "_check_vllm_models",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_compactor_health",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_admin_localhost",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_tts",
                          new=AsyncMock(return_value=(True, "ok"))), \
             patch.object(selftest, "_check_chat_round_trip",
                          new=AsyncMock(return_value=(True, "ok"))):
            return await selftest.run_selftest(do_round_trip=True)

    report = asyncio.run(go())
    names = [c["name"] for c in report["checks"]]
    assert_true("tts" in names, "tts check present when enabled")
    assert_true("stt" not in names, "stt check absent when disabled")
    assert_eq(report["summary"]["total"], 7, "7 checks (6 core + tts)")


# ---------------------------------------------------------------------------
# run_selftest — aggregate
# ---------------------------------------------------------------------------

def test_run_selftest_all_passing():
    print("\n[test] run_selftest: status='pass' when all checks succeed")

    # Mock every async check to succeed
    async def go():
        with patch.object(selftest, "STT_ENABLED", False), \
             patch.object(selftest, "TTS_ENABLED", False), \
             patch.object(selftest, "_check_vllm_models",
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
        with patch.object(selftest, "STT_ENABLED", False), \
             patch.object(selftest, "TTS_ENABLED", False), \
             patch.object(selftest, "_check_vllm_models",
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
        with patch.object(selftest, "STT_ENABLED", False), \
             patch.object(selftest, "TTS_ENABLED", False), \
             patch.object(selftest, "_check_vllm_models",
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
        with patch.object(selftest, "STT_ENABLED", False), \
             patch.object(selftest, "TTS_ENABLED", False), \
             patch.object(selftest, "_check_vllm_models",
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
        test_purge_conv_files_removes_every_layer,
        test_conv_residue_lists_only_what_exists,
        test_purge_conv_files_reports_what_it_could_not_remove,
        test_facts_round_trip_succeeds_with_real_storage,
        test_facts_round_trip_leaves_no_file_at_all,
        test_facts_round_trip_fails_when_cleanup_leaves_something,
        test_facts_round_trip_cleans_up_even_on_inner_failure,
        test_compactor_health_check_200,
        test_compactor_health_check_500,
        test_chat_round_trip_well_formed_response,
        test_chat_round_trip_malformed_response,
        test_chat_round_trip_waits_for_the_tail_before_deleting,
        test_chat_round_trip_fails_when_the_conversation_survives,
        test_chat_round_trip_fails_when_the_tail_writes_the_conv_back,
        test_chat_round_trip_reports_but_survives_a_busy_pool,
        test_wait_for_tail_drain_returns_drained_on_empty_pool,
        test_wait_for_tail_drain_times_out_on_busy_pool,
        test_wait_for_tail_drain_gives_up_immediately_when_depth_unreported,
        test_admin_localhost_200,
        test_admin_localhost_403_fails,
        test_stt_asserts_a_real_transcription_of_synthesized_speech,
        test_stt_fails_when_the_spoken_probe_transcribes_to_nothing,
        test_stt_is_liveness_only_and_says_so_when_tts_is_disabled,
        test_stt_falls_back_to_liveness_when_tts_synthesis_fails,
        test_speech_probe_rejects_non_audio_body,
        test_stt_check_503_fails,
        test_stt_check_malformed_fails,
        test_tiny_wav_is_a_decodable_wav,
        test_run_selftest_includes_stt_when_enabled,
        test_tts_check_200_audio,
        test_tts_check_non_audio_content_type_fails,
        test_tts_check_503_fails,
        test_run_selftest_includes_tts_when_enabled,
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
