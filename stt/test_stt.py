"""
CPU-only Tier-1 tests for V3.2 (Speech-to-Text / Whisper) service.

Exercises the request->response CONTRACT of stt.server with a FAKE whisper
model — no GPU, no faster-whisper, no real audio. Verifies:
  1. each OpenAI response_format renders correctly (json / text / verbose_json
     / srt / vtt, plus unknown -> json fallback)
  2. timestamp + subtitle formatting are correct
  3. language / prompt / task pass through to the model
  4. error paths: empty audio -> 400, model-not-ready -> 503

Deps: fastapi + python-multipart (FastAPI needs python-multipart at import to
register the File/Form routes). faster-whisper is NOT needed — get_model is
patched. Run:
    pip install fastapi python-multipart && python stt/test_stt.py
"""

import json
import os
import sys
from types import SimpleNamespace

# Never warm up a real model if a lifespan ever runs.
os.environ["WHISPER_WARMUP_ON_START"] = "false"
os.environ["WHISPER_VAD_FILTER"] = "false"  # keep fake-model kwargs predictable

import server  # noqa: E402


# ---------------------------------------------------------------------------
# assert helpers (match the project's Tier-1 style)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


class FakeModel:
    """Stands in for faster_whisper.WhisperModel. Records the last transcribe
    kwargs so we can assert pass-through."""

    def __init__(self, segments=None, info=None):
        self._segments = segments if segments is not None else [
            _seg(0.0, 1.0, " Hello"),
            _seg(1.0, 2.0, " world"),
        ]
        self._info = info if info is not None else SimpleNamespace(language="en", duration=2.0)
        self.last_kwargs = None

    def transcribe(self, audio, **kwargs):
        self.last_kwargs = kwargs
        # faster-whisper returns a lazy generator + an info object
        return iter(self._segments), self._info


def _patch_model(model):
    server._model = None  # reset singleton
    server.get_model = lambda: model


def _body_json(resp):
    return json.loads(bytes(resp.body).decode())


def _body_text(resp):
    return bytes(resp.body).decode()


# ---------------------------------------------------------------------------
# response_format rendering
# ---------------------------------------------------------------------------

def test_json_format():
    print("\n[test] response_format=json -> {'text': ...}")
    _patch_model(FakeModel())
    resp = server._transcribe(b"AUDIO", response_format="json")
    assert_eq(resp.status_code, 200, "200 OK")
    assert_eq(_body_json(resp), {"text": "Hello world"}, "concatenated + stripped text")


def test_default_format_is_json():
    print("\n[test] unknown/empty response_format falls back to json")
    _patch_model(FakeModel())
    resp = server._transcribe(b"AUDIO", response_format="")
    assert_eq(_body_json(resp)["text"], "Hello world", "empty -> json")
    resp2 = server._transcribe(b"AUDIO", response_format="bogus")
    assert_eq(_body_json(resp2)["text"], "Hello world", "bogus -> json")


def test_text_format():
    print("\n[test] response_format=text -> plain text body")
    _patch_model(FakeModel())
    resp = server._transcribe(b"AUDIO", response_format="text")
    assert_eq(_body_text(resp), "Hello world", "raw text body")


def test_verbose_json_format():
    print("\n[test] response_format=verbose_json -> segments + language + duration")
    _patch_model(FakeModel())
    resp = server._transcribe(b"AUDIO", response_format="verbose_json")
    data = _body_json(resp)
    assert_eq(data["language"], "en", "language echoed")
    assert_eq(data["duration"], 2.0, "duration echoed")
    assert_eq(data["text"], "Hello world", "full text present")
    assert_eq(len(data["segments"]), 2, "two segments")
    assert_eq(data["segments"][0]["id"], 0, "segment id 0")
    assert_eq(data["segments"][1]["text"], "world", "segment text stripped")


def test_srt_format():
    print("\n[test] response_format=srt -> numbered cues with , ms separator")
    _patch_model(FakeModel())
    resp = server._transcribe(b"AUDIO", response_format="srt")
    body = _body_text(resp)
    assert_true(body.startswith("1\n"), "first cue index")
    assert_true("00:00:00,000 --> 00:00:01,000" in body, "SRT timestamp w/ comma")
    assert_true("Hello" in body and "world" in body, "cue text present")


def test_vtt_format():
    print("\n[test] response_format=vtt -> WEBVTT header with . ms separator")
    _patch_model(FakeModel())
    resp = server._transcribe(b"AUDIO", response_format="vtt")
    body = _body_text(resp)
    assert_true(body.startswith("WEBVTT"), "WEBVTT header")
    assert_true("00:00:01.000 --> 00:00:02.000" in body, "VTT timestamp w/ dot")


# ---------------------------------------------------------------------------
# timestamp formatting
# ---------------------------------------------------------------------------

def test_fmt_ts():
    print("\n[test] _fmt_ts formats HH:MM:SS<sep>mmm")
    assert_eq(server._fmt_ts(0, ","), "00:00:00,000", "zero")
    assert_eq(server._fmt_ts(3661.5, ","), "01:01:01,500", "1h1m1.5s comma")
    assert_eq(server._fmt_ts(3661.5, "."), "01:01:01.500", "dot separator")
    assert_eq(server._fmt_ts(-5, ","), "00:00:00,000", "negative clamps to zero")
    assert_eq(server._fmt_ts(None, ","), "00:00:00,000", "None clamps to zero")


# ---------------------------------------------------------------------------
# pass-through
# ---------------------------------------------------------------------------

def test_param_passthrough():
    print("\n[test] language / prompt / task forwarded to the model")
    m = FakeModel()
    _patch_model(m)
    server._transcribe(b"AUDIO", task="translate", language="es", prompt="context here")
    assert_eq(m.last_kwargs["task"], "translate", "task forwarded")
    assert_eq(m.last_kwargs["language"], "es", "language forwarded")
    assert_eq(m.last_kwargs["initial_prompt"], "context here", "prompt -> initial_prompt")
    # empty language -> None (auto-detect), empty prompt -> None
    server._transcribe(b"AUDIO", language="", prompt="")
    assert_eq(m.last_kwargs["language"], None, "empty language -> None")
    assert_eq(m.last_kwargs["initial_prompt"], None, "empty prompt -> None")


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------

def test_empty_audio_400():
    print("\n[test] empty audio -> 400")
    _patch_model(FakeModel())
    resp = server._transcribe(b"", response_format="json")
    assert_eq(resp.status_code, 400, "400 on empty")
    assert_true("error" in _body_json(resp), "error body")


def test_model_not_ready_503():
    print("\n[test] model load failure -> 503")

    def boom():
        raise RuntimeError("still loading")

    server._model = None
    server.get_model = boom
    resp = server._transcribe(b"AUDIO", response_format="json")
    assert_eq(resp.status_code, 503, "503 when model not ready")


def test_transcribe_exception_500():
    print("\n[test] transcription error -> 500")

    class BoomModel:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("decode failed")

    _patch_model(BoomModel())
    resp = server._transcribe(b"AUDIO", response_format="json")
    assert_eq(resp.status_code, 500, "500 on transcription failure")


def _all():
    return [
        test_json_format,
        test_default_format_is_json,
        test_text_format,
        test_verbose_json_format,
        test_srt_format,
        test_vtt_format,
        test_fmt_ts,
        test_param_passthrough,
        test_empty_audio_400,
        test_model_not_ready_503,
        test_transcribe_exception_500,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll STT smoke tests passed.")
