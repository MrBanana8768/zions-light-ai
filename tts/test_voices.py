"""
CPU-only Tier-1 tests for N6's /v1/audio/voices endpoint.

THE FAILURE THIS IS ABOUT. OpenWebUI's "OpenAI" TTS engine calls
GET {api_base_url}/audio/voices before it will let a user pick a voice at
all (backend/open_webui/routers/audio.py's get_available_voices, openai
branch) and expects back {"voices": [{"id": ..., "name": ...}, ...]}. Piper
synthesis never needed this endpoint — TTS_VOICE names the one voice this
process loads — so it was simply absent, and OpenWebUI logged a 404 on it
ten times in the review window even though /v1/audio/speech worked fine.

Deps: fastapi only — no piper-tts, no onnxruntime, no real voice files.
list_voices() is pure filesystem enumeration; only the ".onnx"/".onnx.json"
NAMES matter, never their contents.

Run: python test_voices.py
"""

import os
import shutil
import sys
import tempfile

os.environ["TTS_WARMUP_ON_START"] = "false"

import server  # noqa: E402

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"FAIL   {label}")


def _touch(d, name):
    open(os.path.join(d, name), "w").close()


# ---------------------------------------------------------------------------
# list_voices()
# ---------------------------------------------------------------------------

print("[1] list_voices() reads the voice directory")

_tmp = tempfile.mkdtemp(prefix="tts_voices_test_")
_orig_dir = server.TTS_VOICE_DIR
_orig_voice = server.TTS_VOICE

try:
    server.TTS_VOICE_DIR = _tmp
    server.TTS_VOICE = "en_US-lessac-medium"

    # Empty directory -> falls back to advertising the configured voice,
    # not an empty list (an empty picker is worse than a one-item picker
    # that just happens to match what /v1/audio/speech will actually use).
    check(
        server.list_voices() == [{"id": "en_US-lessac-medium", "name": "en_US-lessac-medium"}],
        "empty voice dir falls back to TTS_VOICE, not []",
    )

    # A matched pair (.onnx + its .onnx.json sibling) is listed.
    _touch(_tmp, "en_US-lessac-medium.onnx")
    _touch(_tmp, "en_US-lessac-medium.onnx.json")
    check(
        server.list_voices() == [{"id": "en_US-lessac-medium", "name": "en_US-lessac-medium"}],
        "a matched .onnx/.onnx.json pair is listed by its stem",
    )

    # An .onnx with NO config sibling is not advertised — PiperVoice.load
    # needs both, so listing it would offer a choice that 500s on synthesis.
    _touch(_tmp, "orphan-voice.onnx")
    check(
        server.list_voices() == [{"id": "en_US-lessac-medium", "name": "en_US-lessac-medium"}],
        "an .onnx with no .onnx.json sibling is skipped",
    )

    # A second complete pair is listed alongside the first, sorted.
    _touch(_tmp, "en_US-amy-medium.onnx")
    _touch(_tmp, "en_US-amy-medium.onnx.json")
    check(
        server.list_voices() == [
            {"id": "en_US-amy-medium", "name": "en_US-amy-medium"},
            {"id": "en_US-lessac-medium", "name": "en_US-lessac-medium"},
        ],
        "multiple prebaked voices are all listed, sorted by id",
    )

    # A lone .onnx.json with no matching .onnx is not a voice either.
    _touch(_tmp, "config-only.onnx.json")
    check(
        len(server.list_voices()) == 2,
        "a lone .onnx.json with no .onnx sibling contributes nothing",
    )

    # Unreadable directory -> same TTS_VOICE fallback, never an exception.
    server.TTS_VOICE_DIR = os.path.join(_tmp, "does-not-exist")
    check(
        server.list_voices() == [{"id": "en_US-lessac-medium", "name": "en_US-lessac-medium"}],
        "a missing voice dir falls back to TTS_VOICE instead of raising",
    )
finally:
    server.TTS_VOICE_DIR = _orig_dir
    server.TTS_VOICE = _orig_voice
    shutil.rmtree(_tmp, ignore_errors=True)

# ---------------------------------------------------------------------------
# GET /v1/audio/voices — shape lock against what OpenWebUI actually parses
# ---------------------------------------------------------------------------

print()
print("[2] GET /v1/audio/voices response shape")


def _run_endpoint():
    import asyncio

    return asyncio.run(server.voices_endpoint())


_tmp2 = tempfile.mkdtemp(prefix="tts_voices_test2_")
try:
    server.TTS_VOICE_DIR = _tmp2
    server.TTS_VOICE = "en_US-lessac-medium"
    _touch(_tmp2, "en_US-lessac-medium.onnx")
    _touch(_tmp2, "en_US-lessac-medium.onnx.json")

    body = _run_endpoint()
    check("voices" in body, "top-level 'voices' key present")
    check(isinstance(body["voices"], list), "'voices' is a list")
    check(len(body["voices"]) == 1, "one voice reported")
    v = body["voices"][0]
    check(
        set(v.keys()) == {"id", "name"},
        "each voice is exactly {id, name} — the two keys OpenWebUI's "
        "get_available_voices reads (`{v['id']: v['name'] for v in "
        "data['voices']}`)",
    )
    check(v["id"] == "en_US-lessac-medium" and v["name"] == "en_US-lessac-medium", "id/name populated")
finally:
    server.TTS_VOICE_DIR = _orig_dir
    server.TTS_VOICE = _orig_voice
    shutil.rmtree(_tmp2, ignore_errors=True)

# ---------------------------------------------------------------------------
# Sibling check: /v1/audio/speech must still work — this endpoint must not
# have broken the one that already worked.
# ---------------------------------------------------------------------------

print()
print("[3] /v1/audio/speech is untouched")

_CANNED_WAV = server._pcm_to_wav(b"\x00\x00" * 2205, sample_rate=22050)


class _FakeVoice:
    pass


server._engine = None
server.get_engine = lambda: _FakeVoice()
server._synthesize_wav = lambda text, speed=1.0: _CANNED_WAV
resp = server._speech({"input": "still works", "response_format": "wav"})
check(resp.status_code == 200, "synthesis still returns 200")
check(bytes(resp.body) == _CANNED_WAV, "synthesis still returns audio")

print()
if FAILED:
    print(f"{len(FAILED)} FAILURE(S):")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("All TTS voice-listing tests passed.")
