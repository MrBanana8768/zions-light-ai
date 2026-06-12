"""
tts.server — V3.3 Text-to-Speech service.

A thin, OpenAI-compatible speech service in front of **Piper** (a fast,
onnxruntime-based neural TTS). Exposes:

    POST /v1/audio/speech   — text -> spoken audio (OpenAI shape)
    GET  /v1/models         — lists the loaded voice/model id
    GET  /health            — readiness probe

so OpenWebUI's "OpenAI" TTS engine — and any OpenAI audio client — can talk to
it unchanged.

Design notes (mirrors the STT service deliberately)
---------------------------------------------------
* **Own venv, own process.** Runs in /opt/tts-venv as its own supervisord
  program, so its deps can NEVER disturb the vLLM or compactor venvs.
* **Torch-free + CPU.** Piper runs on onnxruntime — no torch, no GPU — so it
  keeps the image lean and never competes with vLLM for VRAM. (Kokoro is a
  higher-quality alternative but pulls torch; left as a documented swap.)
* **Independent of the memory pipeline.** This service does text -> audio only.
  The model's reply text is produced through the normal compactor path; only
  the *spoken rendering* of it comes through here.
* **Testable core.** The engine is imported lazily (inside get_engine) and the
  request logic lives in plain functions (_speech / _encode / _pcm_to_wav ...),
  so Tier-1 tests exercise the request->response contract with a fake engine
  and never load Piper or a voice model.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import wave
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("tts.server")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


# Piper voice: a "<name>.onnx" file (with its "<name>.onnx.json" sibling) under
# TTS_VOICE_DIR. The default is prebaked into the image; swap via TTS_VOICE.
TTS_VOICE = _env("TTS_VOICE", "en_US-lessac-medium")
TTS_VOICE_DIR = _env("TTS_VOICE_DIR", "/opt/tts-voices")
TTS_MODEL_ID = _env("TTS_MODEL_ID", "tts-1")  # what /v1/models reports; OpenWebUI sends this back

TTS_HOST = _env("TTS_HOST", "0.0.0.0")
TTS_PORT = int(_env("TTS_PORT", "9001"))

# Load the voice during startup so /health reflects true readiness. Tests set
# this false; they patch get_engine.
TTS_WARMUP_ON_START = _env("TTS_WARMUP_ON_START", "true").lower() != "false"

CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/L16",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
}


# ---------------------------------------------------------------------------
# Engine (lazy singleton — keeps Tier-1 free of Piper / onnxruntime)
# ---------------------------------------------------------------------------

_engine = None


def _voice_path() -> str:
    return os.path.join(TTS_VOICE_DIR, f"{TTS_VOICE}.onnx")


def get_engine():
    """Return the loaded Piper voice, importing piper on first use. Lazy import
    is deliberate so unit tests can patch this function and run with only
    fastapi installed."""
    global _engine
    if _engine is not None:
        return _engine
    from piper import PiperVoice  # noqa: WPS433 (intentional lazy import)

    path = _voice_path()
    logger.info("loading piper voice=%s from %s", TTS_VOICE, path)
    _engine = PiperVoice.load(path)  # finds the .onnx.json config alongside
    logger.info("piper voice loaded")
    return _engine


def engine_ready() -> bool:
    return _engine is not None


# ---------------------------------------------------------------------------
# Audio helpers (pure functions — unit tested directly)
# ---------------------------------------------------------------------------

def _pcm_to_wav(pcm: bytes, sample_rate: int = 22050, channels: int = 1, sampwidth: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_to_pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes())


def _ffmpeg_encode(wav_bytes: bytes, fmt: str):
    """Convert WAV -> fmt via ffmpeg if it's on PATH; else None (caller falls
    back to wav). Keeps mp3/opus/aac/flac support optional, not a hard dep."""
    if not shutil.which("ffmpeg"):
        return None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", fmt, "pipe:1"],
            input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        return proc.stdout
    except Exception as e:
        logger.warning("ffmpeg encode to %s failed: %s", fmt, e)
        return None


def _encode(wav_bytes: bytes, response_format: str):
    """Render WAV bytes in the requested OpenAI format. Returns (bytes, content_type).
    Unknown / unsupported-without-ffmpeg formats fall back to WAV."""
    fmt = (response_format or "wav").lower()
    if fmt in ("wav", "wave"):
        return wav_bytes, CONTENT_TYPES["wav"]
    if fmt == "pcm":
        return _wav_to_pcm(wav_bytes), CONTENT_TYPES["pcm"]
    if fmt in ("mp3", "opus", "aac", "flac"):
        out = _ffmpeg_encode(wav_bytes, fmt)
        if out is not None:
            return out, CONTENT_TYPES[fmt]
        logger.warning("returning wav for requested '%s' (ffmpeg unavailable)", fmt)
        return wav_bytes, CONTENT_TYPES["wav"]
    return wav_bytes, CONTENT_TYPES["wav"]


# ---------------------------------------------------------------------------
# Synthesis (sync; runs in a threadpool from the endpoint)
# ---------------------------------------------------------------------------

def _synthesize_wav(text: str, *, speed: float = 1.0) -> bytes:
    """Render `text` to 16-bit PCM WAV bytes via Piper. Engine call isolated
    here so Tier-1 can mock it without Piper installed.

    Piper's `synthesize_wav(text, wav_file, syn_config=...)` writes a complete
    WAV to an open `wave.Wave_write`. `speed` maps to `SynthesisConfig.length_scale`
    (higher = slower), so length_scale = 1/speed. (Verified against piper-tts
    1.4.2 in the built image.)
    """
    voice = get_engine()
    syn_config = None
    if speed and speed > 0 and speed != 1.0:
        try:
            from piper import SynthesisConfig
            syn_config = SynthesisConfig(length_scale=1.0 / speed)
        except Exception:
            syn_config = None  # speed is best-effort; never break synthesis over it
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)
    return buf.getvalue()


def _speech(body: dict):
    """Core of POST /v1/audio/speech — pure enough to unit-test directly."""
    text = (body.get("input") or "").strip()
    if not text:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "missing 'input' text", "type": "invalid_request_error"}},
        )
    response_format = body.get("response_format") or "wav"
    try:
        speed = float(body.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    try:
        get_engine()
    except Exception as e:
        logger.error("voice not ready: %s", e)
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"voice not ready: {e}", "type": "server_error"}},
        )
    try:
        wav = _synthesize_wav(text, speed=speed)
    except Exception as e:
        logger.exception("synthesis failed")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"synthesis failed: {e}", "type": "server_error"}},
        )
    audio, content_type = _encode(wav, response_format)
    logger.info("spoke %d chars -> %d bytes (%s)", len(text), len(audio), content_type)
    return Response(content=audio, media_type=content_type)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if TTS_WARMUP_ON_START:
        try:
            get_engine()
        except Exception as e:
            logger.error("warmup failed (will retry on first request): %s", e)
    yield


app = FastAPI(title="Zion's Light AI — TTS (Piper)", lifespan=lifespan)


@app.get("/health")
async def health():
    ready = engine_ready()
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "loading", "ready": ready, "voice": TTS_VOICE},
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": TTS_MODEL_ID, "object": "model", "owned_by": "piper"}],
    }


@app.post("/v1/audio/speech")
async def speech_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await run_in_threadpool(_speech, body)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("starting TTS service on %s:%d (voice=%s)", TTS_HOST, TTS_PORT, TTS_VOICE)
    import uvicorn

    uvicorn.run(app, host=TTS_HOST, port=TTS_PORT, log_level="info")


if __name__ == "__main__":
    main()
