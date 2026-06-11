"""
CPU-only Tier-1 tests for V3.3 (Text-to-Speech / Piper) service.

Exercises the request->response CONTRACT of tts.server with a FAKE engine — no
Piper, no onnxruntime, no real voice model, no audio synthesis. Verifies:
  1. WAV <-> PCM helpers round-trip
  2. _encode renders/falls back per response_format (wav / pcm / unknown)
  3. POST /v1/audio/speech core: success (audio/wav), pcm format, and the
     400 (empty input) / 503 (voice not ready) / 500 (synth error) paths
  4. malformed `speed` degrades to the default instead of crashing

Deps: fastapi only (no python-multipart — the speech endpoint takes a JSON
body, not a multipart form). piper-tts is NOT needed — get_engine is patched.
Run: pip install fastapi && python test_tts.py
"""

import json
import os
import sys

# Never warm up a real voice if a lifespan ever runs.
os.environ["TTS_WARMUP_ON_START"] = "false"

import server  # noqa: E402


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


# A small valid WAV (0.1s of silence @ 22050) for canned engine output.
_CANNED_WAV = server._pcm_to_wav(b"\x00\x00" * 2205, sample_rate=22050)


class FakeVoice:
    def synthesize(self, text, wav_file, **kwargs):
        raise AssertionError("real synth should not run; _synthesize_wav is patched")


def _patch_ok():
    server._engine = None
    server.get_engine = lambda: FakeVoice()
    server._synthesize_wav = lambda text, speed=1.0: _CANNED_WAV


def _body_json(resp):
    return json.loads(bytes(resp.body).decode())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_pcm_wav_roundtrip():
    print("\n[test] _pcm_to_wav / _wav_to_pcm round-trip")
    pcm = b"\x01\x02\x03\x04" * 50
    wav = server._pcm_to_wav(pcm, sample_rate=22050)
    assert_true(wav[:4] == b"RIFF", "valid RIFF/WAV header")
    assert_eq(server._wav_to_pcm(wav), pcm, "pcm survives wrap+unwrap")


def test_encode_wav_passthrough():
    print("\n[test] _encode: wav -> passthrough audio/wav")
    out, ct = server._encode(_CANNED_WAV, "wav")
    assert_eq(out, _CANNED_WAV, "wav bytes unchanged")
    assert_eq(ct, "audio/wav", "audio/wav content type")


def test_encode_pcm():
    print("\n[test] _encode: pcm -> raw frames, audio/L16")
    out, ct = server._encode(_CANNED_WAV, "pcm")
    assert_eq(out, server._wav_to_pcm(_CANNED_WAV), "raw pcm frames")
    assert_eq(ct, "audio/L16", "audio/L16 content type")


def test_encode_unknown_falls_back_to_wav():
    print("\n[test] _encode: unknown format -> wav fallback")
    out, ct = server._encode(_CANNED_WAV, "bogus")
    assert_eq(ct, "audio/wav", "fallback content type")
    assert_eq(out, _CANNED_WAV, "fallback returns wav bytes")


# ---------------------------------------------------------------------------
# /v1/audio/speech core
# ---------------------------------------------------------------------------

def test_speech_success_wav():
    print("\n[test] _speech: valid input -> 200 audio/wav")
    _patch_ok()
    resp = server._speech({"input": "hello world", "response_format": "wav"})
    assert_eq(resp.status_code, 200, "200 OK")
    assert_eq(resp.media_type, "audio/wav", "audio/wav")
    assert_eq(bytes(resp.body), _CANNED_WAV, "returns synthesized audio")


def test_speech_pcm_format():
    print("\n[test] _speech: response_format=pcm -> audio/L16")
    _patch_ok()
    resp = server._speech({"input": "hi", "response_format": "pcm"})
    assert_eq(resp.status_code, 200, "200 OK")
    assert_eq(resp.media_type, "audio/L16", "audio/L16 for pcm")


def test_speech_empty_input_400():
    print("\n[test] _speech: empty input -> 400")
    _patch_ok()
    resp = server._speech({"input": "   "})
    assert_eq(resp.status_code, 400, "400 on empty input")
    assert_true("error" in _body_json(resp), "error body")


def test_speech_voice_not_ready_503():
    print("\n[test] _speech: engine load failure -> 503")

    def boom():
        raise RuntimeError("voice still loading")

    server._engine = None
    server.get_engine = boom
    resp = server._speech({"input": "hello"})
    assert_eq(resp.status_code, 503, "503 when voice not ready")


def test_speech_synth_error_500():
    print("\n[test] _speech: synthesis error -> 500")
    server._engine = None
    server.get_engine = lambda: FakeVoice()

    def boom(text, speed=1.0):
        raise RuntimeError("onnx blew up")

    server._synthesize_wav = boom
    resp = server._speech({"input": "hello"})
    assert_eq(resp.status_code, 500, "500 on synthesis failure")


def test_speech_bad_speed_defaults():
    print("\n[test] _speech: malformed speed degrades to default, still 200")
    _patch_ok()
    resp = server._speech({"input": "hi", "speed": "fast"})
    assert_eq(resp.status_code, 200, "bad speed doesn't crash")


def _all():
    return [
        test_pcm_wav_roundtrip,
        test_encode_wav_passthrough,
        test_encode_pcm,
        test_encode_unknown_falls_back_to_wav,
        test_speech_success_wav,
        test_speech_pcm_format,
        test_speech_empty_input_400,
        test_speech_voice_not_ready_503,
        test_speech_synth_error_500,
        test_speech_bad_speed_defaults,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll TTS smoke tests passed.")
