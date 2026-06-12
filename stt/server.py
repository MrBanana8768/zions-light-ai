"""
stt.server — V3.2 Speech-to-Text service (Whisper).

A thin, OpenAI-compatible transcription service in front of faster-whisper
(CTranslate2). Exposes:

    POST /v1/audio/transcriptions   — audio -> text (OpenAI shape)
    POST /v1/audio/translations     — audio -> English text
    GET  /v1/models                 — lists the loaded model
    GET  /health                    — readiness probe

so OpenWebUI's "OpenAI" STT engine — and any OpenAI audio client — can talk
to it unchanged.

Design notes
------------
* **Own venv, own process.** Runs in /opt/whisper-venv as its own supervisord
  program, so its deps (faster-whisper, ctranslate2, av) can NEVER disturb the
  vLLM or compactor venvs. Same isolation discipline as the rest of the stack.
* **Independent of the memory pipeline.** This service does audio -> text only.
  The transcribed text then becomes an ordinary chat message that flows through
  the compactor like anything the user typed — no special wiring needed.
* **CPU by default.** vLLM reserves ~90% of the GPU (GPU_MEMORY_UTILIZATION),
  so transcribing on the GPU would fight it for VRAM and risks the A40 OOM we
  fought before. faster-whisper is fast on CPU with int8 for the small/base
  models, so WHISPER_DEVICE defaults to "cpu". Flip to "cuda" only with real
  headroom (bigger card, or a lowered vLLM utilization).
* **Testable core.** The model is imported lazily (inside get_model) and the
  transcription logic lives in plain functions (_transcribe / _format_response
  / _to_srt ...), so Tier-1 tests exercise the request->response contract with
  a fake model and never load faster-whisper or touch a GPU.
"""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("stt.server")


# ---------------------------------------------------------------------------
# Configuration (env, with sane defaults)
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


# Whisper model size or HF repo. "base" is a good speed/quality balance on CPU
# and is prebaked into the image. "small"/"medium"/"large-v3" are bigger and
# better; pick large-v3 + WHISPER_DEVICE=cuda only with VRAM headroom.
WHISPER_MODEL = _env("WHISPER_MODEL", "base")
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu").lower()
# Empty -> auto: int8 on CPU, float16 on CUDA.
WHISPER_COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "")
WHISPER_DOWNLOAD_ROOT = _env("WHISPER_DOWNLOAD_ROOT", "/opt/whisper-models")
WHISPER_BEAM_SIZE = int(_env("WHISPER_BEAM_SIZE", "5"))
# Voice-activity-detection filter trims silence/noise -> far fewer
# hallucinated transcripts on near-silent clips. faster-whisper bundles the
# VAD model (no runtime download). Disable with WHISPER_VAD_FILTER=false.
WHISPER_VAD_FILTER = _env("WHISPER_VAD_FILTER", "true").lower() != "false"

# Public model id reported by /v1/models. OpenWebUI's OpenAI STT engine sends
# model="whisper-1" by default; we accept any string and use the loaded model.
WHISPER_MODEL_ID = _env("WHISPER_MODEL_ID", "whisper-1")

STT_HOST = _env("STT_HOST", "0.0.0.0")
STT_PORT = int(_env("STT_PORT", "9000"))

# Load the model during startup so /health reflects true readiness (the boot
# self-test relies on this). Tests set this false; they patch get_model.
WHISPER_WARMUP_ON_START = _env("WHISPER_WARMUP_ON_START", "true").lower() != "false"


# ---------------------------------------------------------------------------
# Model (lazy singleton — keeps Tier-1 free of faster-whisper / GPU)
# ---------------------------------------------------------------------------

_model = None


def get_model():
    """Return the loaded WhisperModel, importing faster-whisper on first use.

    Lazy import is deliberate: importing this module (and the FastAPI app) must
    NOT pull in ctranslate2 / faster-whisper, so unit tests can patch this
    function and run with only fastapi installed.
    """
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel  # noqa: WPS433 (intentional lazy import)

    compute = WHISPER_COMPUTE_TYPE or ("float16" if WHISPER_DEVICE == "cuda" else "int8")
    logger.info(
        "loading whisper model=%s device=%s compute=%s root=%s",
        WHISPER_MODEL, WHISPER_DEVICE, compute, WHISPER_DOWNLOAD_ROOT,
    )
    _model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=compute,
        download_root=WHISPER_DOWNLOAD_ROOT,
    )
    logger.info("whisper model loaded")
    return _model


def model_ready() -> bool:
    return _model is not None


# ---------------------------------------------------------------------------
# Subtitle / response formatting (pure functions — unit tested directly)
# ---------------------------------------------------------------------------

def _fmt_ts(seconds, sep: str = ",") -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep ',' for SRT, '.' for VTT)."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = 0.0
    if s < 0:
        s = 0.0
    ms_total = int(round(s * 1000.0))
    h, ms_total = divmod(ms_total, 3_600_000)
    m, ms_total = divmod(ms_total, 60_000)
    sec, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}{sep}{ms:03d}"


def _to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts(seg.start, ',')} --> {_fmt_ts(seg.end, ',')}")
        lines.append((seg.text or "").strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_fmt_ts(seg.start, '.')} --> {_fmt_ts(seg.end, '.')}")
        lines.append((seg.text or "").strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _format_response(response_format: str, text: str, segments, info):
    """Render a transcription result in the requested OpenAI format."""
    fmt = (response_format or "json").lower()
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "srt":
        return PlainTextResponse(_to_srt(segments), media_type="application/x-subrip")
    if fmt == "vtt":
        return PlainTextResponse(_to_vtt(segments), media_type="text/vtt")
    if fmt == "verbose_json":
        return JSONResponse({
            "task": "transcribe",
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "text": text,
            "segments": [
                {
                    "id": i,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": (seg.text or "").strip(),
                }
                for i, seg in enumerate(segments)
            ],
        })
    # default + unknown -> minimal JSON ({"text": ...})
    return JSONResponse({"text": text})


# ---------------------------------------------------------------------------
# Transcription orchestration (sync; runs in a threadpool from the endpoint)
# ---------------------------------------------------------------------------

def _transcribe(
    raw: bytes,
    *,
    task: str = "transcribe",
    language: str = "",
    prompt: str = "",
    response_format: str = "json",
    temperature: float = 0.0,
):
    if not raw:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "empty audio file", "type": "invalid_request_error"}},
        )
    try:
        model = get_model()
    except Exception as e:  # model still loading / failed to load
        logger.error("model not ready: %s", e)
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"model not ready: {e}", "type": "server_error"}},
        )
    try:
        segments, info = model.transcribe(
            io.BytesIO(raw),
            task=task,
            language=(language or None),
            initial_prompt=(prompt or None),
            beam_size=WHISPER_BEAM_SIZE,
            temperature=temperature,
            vad_filter=WHISPER_VAD_FILTER,
        )
        seg_list = list(segments)  # faster-whisper yields lazily; materialize
    except Exception as e:
        logger.exception("transcription failed")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"transcription failed: {e}", "type": "server_error"}},
        )
    text = "".join((seg.text or "") for seg in seg_list).strip()
    logger.info("transcribed %d bytes -> %d chars (%d segments)", len(raw), len(text), len(seg_list))
    return _format_response(response_format, text, seg_list, info)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if WHISPER_WARMUP_ON_START:
        try:
            get_model()
        except Exception as e:
            logger.error("warmup failed (will retry on first request): %s", e)
    yield


app = FastAPI(title="Zion's Light AI — STT (Whisper)", lifespan=lifespan)


@app.get("/health")
async def health():
    ready = model_ready()
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "loading",
            "ready": ready,
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
        },
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": WHISPER_MODEL_ID, "object": "model", "owned_by": "whisper"}],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions_endpoint(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default=""),
    prompt: str = Form(default=""),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    raw = await file.read()
    return await run_in_threadpool(
        _transcribe,
        raw,
        task="transcribe",
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
    )


@app.post("/v1/audio/translations")
async def translations_endpoint(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    prompt: str = Form(default=""),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    raw = await file.read()
    return await run_in_threadpool(
        _transcribe,
        raw,
        task="translate",
        language="",
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "starting STT service on %s:%d (model=%s device=%s)",
        STT_HOST, STT_PORT, WHISPER_MODEL, WHISPER_DEVICE,
    )
    import uvicorn

    uvicorn.run(app, host=STT_HOST, port=STT_PORT, log_level="info")


if __name__ == "__main__":
    main()
