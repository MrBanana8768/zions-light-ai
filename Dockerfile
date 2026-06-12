# Dockerfile for vLLM + context-compactor + OpenWebUI on Runpod
# Optimized for NVIDIA A40 (48GB VRAM) and larger.
#
# CUDA base + torch wheels are parametric via build args so the same
# Dockerfile can build cu128 (default, RunPod driver 570 compatible) and
# cu130 variants without source changes:
#
#   docker build .                                       # cu128 default
#   docker build --build-arg CUDA_BASE_IMAGE=nvidia/cuda:12.9.1-runtime-ubuntu24.04 \
#                --build-arg TORCH_CUDA=cu128 .          # newer CUDA runtime, same wheels
#   docker build --build-arg CUDA_BASE_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu24.04 \
#                --build-arg TORCH_CUDA=cu130 \
#                --build-arg VLLM_VERSION=0.21.0 .       # full cu130 variant (driver 580+)

ARG CUDA_BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Runtime dependencies.
# - binutils: for strip during the install/cleanup layers (~10 MB).
# - build-essential + python3-dev: required at runtime by Triton's JIT,
#   which compiles per-kernel C source during CUDA graph capture.
# - apt-get upgrade pulls in CVE patches for the base image's installed
#   packages (gnupg2 etc.). One layer, picks up any patched versions
#   released since the base image was published.
# - apt-get clean + autoremove + lists prune keeps the layer slim.
# ~200 MB total — necessary tax for vLLM on a slim base.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        curl \
        wget \
        git \
        libgomp1 \
        supervisor \
        binutils \
        build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Persistent data root — mounted as a single network volume in production.
# Holds both the model cache (/data/models -> HF_HOME) and OpenWebUI state
# (/data/openwebui -> DATA_DIR). entrypoint.sh creates the subdirs on first run.
RUN mkdir -p /data

ENV VLLM_VENV=/opt/vllm-venv

# vLLM + torch CUDA target. Both parametric so the cu130 variant can be
# built from the same source. Defaults aligned for RunPod's driver-570
# fleet (CUDA 12.8 max). CVE-2026-22778 (Critical 9.8) requires vllm >=
# 0.14.1. Bump VLLM_VERSION + TORCH_CUDA together when RunPod rolls out
# driver 580+.
ARG VLLM_VERSION=0.14.1
ARG TORCH_CUDA=cu128

# =============================================================================
# vLLM venv — ONLY vLLM. As of V2.0 Phase 3 the compactor has its OWN venv
# (below), so the compactor's deps (chromadb, fastembed, etc.) can NEVER
# disturb vLLM's torch/transformers pins. This permanently closes the
# dependency-coupling bug class that caused the V1.9.x fire drills.
# Installed + stripped + cache-cleared in one layer so unstripped libs and
# .pyc caches never get committed. --extra-index-url pins the torch CUDA
# channel. pip/setuptools/wheel bumped here too (Scout-flagged Highs).
# =============================================================================
RUN python3 -m venv /opt/vllm-venv && \
    /opt/vllm-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/vllm-venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
        vllm==${VLLM_VERSION} && \
    find /opt/vllm-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/vllm-venv -name "*.pyc" -delete && \
    find /opt/vllm-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# =============================================================================
# Compactor venv — fully decoupled from vLLM. Holds fastapi/uvicorn/httpx
# (proxy), transformers (tokenizer-only, no torch), chromadb (vector store)
# and fastembed (bge-small embeddings via ONNX runtime — no torch, keeps
# this venv lean). COPY requirements separately so editing the Python
# sources later doesn't bust this layer.
# =============================================================================
COPY compactor/requirements.txt /opt/compactor/requirements.txt
RUN python3 -m venv /opt/compactor-venv && \
    /opt/compactor-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/compactor-venv/bin/pip install --no-cache-dir -r /opt/compactor/requirements.txt && \
    find /opt/compactor-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/compactor-venv -name "*.pyc" -delete && \
    find /opt/compactor-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Pre-download the bge-small ONNX embedding model into the image so the
# first request pays no download. Static weights belong in the image, not
# on the /data volume. FASTEMBED_CACHE_PATH (ENV section below) points here.
RUN /opt/compactor-venv/bin/python -c \
    "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/embeddings')" && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# =============================================================================
# Whisper (STT) venv — V3.2. Fully decoupled from vLLM AND the compactor:
# faster-whisper pulls ctranslate2 + av + onnxruntime into its OWN venv, so its
# deps can never disturb vLLM's torch pins or the compactor's. av ships ffmpeg
# in its wheel, so no apt ffmpeg is needed. Same install+strip+clean atomic
# pattern as the other venvs.
# =============================================================================
COPY stt/requirements.txt /opt/stt/requirements.txt
RUN python3 -m venv /opt/whisper-venv && \
    /opt/whisper-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/whisper-venv/bin/pip install --no-cache-dir -r /opt/stt/requirements.txt && \
    find /opt/whisper-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/whisper-venv -name "*.pyc" -delete && \
    find /opt/whisper-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Pre-download the default Whisper model ("base") into the image so the first
# transcription pays no download — static small models belong in the image, not
# on /data (same principle as the bge embeddings). Bigger models (small/medium/
# large-v3) and a persistent /data root are configurable via WHISPER_MODEL /
# WHISPER_DOWNLOAD_ROOT. Built on CPU (no GPU at build time), int8 weights.
RUN /opt/whisper-venv/bin/python -c \
    "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8', download_root='/opt/whisper-models')" && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# =============================================================================
# TTS (Piper) venv — V3.3. Own venv, torch-free (Piper is onnxruntime-based),
# CPU only — never competes with vLLM for VRAM and keeps the image lean. Same
# install+strip+clean atomic pattern as the other venvs.
# =============================================================================
COPY tts/requirements.txt /opt/tts/requirements.txt
RUN python3 -m venv /opt/tts-venv && \
    /opt/tts-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/tts-venv/bin/pip install --no-cache-dir -r /opt/tts/requirements.txt && \
    find /opt/tts-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/tts-venv -name "*.pyc" -delete && \
    find /opt/tts-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Pre-download the default Piper voice (en_US-lessac-medium, ~63 MB) into the
# image so the first speech pays no download — a static model belongs in the
# image (same principle as the bge + whisper models). Swap via TTS_VOICE (+ a
# /data TTS_VOICE_DIR for other voices). Voices: huggingface.co/rhasspy/piper-voices.
RUN mkdir -p /opt/tts-voices && \
    wget -q -O /opt/tts-voices/en_US-lessac-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
    wget -q -O /opt/tts-voices/en_US-lessac-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# =============================================================================
# OpenWebUI venv — kept isolated from vLLM's pytorch pin. Same install+strip
# atomic pattern. Also bumps pip/setuptools/wheel here (Scout-flagged Highs
# live in both venvs since each has its own copy).
# =============================================================================
WORKDIR /app
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /app/venv/bin/pip install --no-cache-dir open-webui && \
    find /app/venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /app/venv -name "*.pyc" -delete && \
    find /app/venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*
# Note: OpenWebUI data lives at DATA_DIR=/data/openwebui (on the persistent
# volume), created by entrypoint.sh at boot. No /app/data dirs needed —
# that was the pre-single-volume layout (removed in V2.2 cleanup).

# Compactor sources copied AFTER the expensive install layer so editing
# the Python files doesn't invalidate the vllm install cache. List each
# runtime module explicitly to avoid pulling test_*.py and V2_PLAN.md
# into the production image.
COPY compactor/main.py /opt/compactor/main.py
COPY compactor/memory.py /opt/compactor/memory.py
COPY compactor/facts.py /opt/compactor/facts.py
COPY compactor/backfill.py /opt/compactor/backfill.py
COPY compactor/retrieval.py /opt/compactor/retrieval.py
COPY compactor/summarizer.py /opt/compactor/summarizer.py
COPY compactor/health.py /opt/compactor/health.py
COPY compactor/selftest.py /opt/compactor/selftest.py
COPY compactor/portability.py /opt/compactor/portability.py
COPY compactor/dedup.py /opt/compactor/dedup.py
COPY compactor/commands.py /opt/compactor/commands.py
COPY compactor/persona.py /opt/compactor/persona.py
COPY compactor/backup.py /opt/compactor/backup.py
COPY compactor/degrade.py /opt/compactor/degrade.py
COPY compactor/bgwork.py /opt/compactor/bgwork.py
COPY compactor/logsetup.py /opt/compactor/logsetup.py
COPY compactor/alert.py /opt/compactor/alert.py

# V3.2 — STT service source (copied late, after its venv, for cache efficiency).
COPY stt/server.py /opt/stt/server.py

# V3.3 — TTS service source (copied late, after its venv, for cache efficiency).
COPY tts/server.py /opt/tts/server.py

# =============================================================================
# Supervisor
# =============================================================================
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# =============================================================================
# Model configuration — override via .env or Runpod template
# Default: anthracite-org/magnum-v4-22b (creative writing fine-tune, lightly
# aligned). On A40 use VLLM_EXTRA_ARGS="--quantization fp8" to fit 32K context;
# without quantization, drop MAX_MODEL_LEN to 8192.
# Any HuggingFace causal-LM repo that vLLM supports works here.
# =============================================================================
ENV MODEL_REPO="anthracite-org/magnum-v4-22b"
ENV HF_HOME="/data/models"
# Note: TRANSFORMERS_CACHE was removed in v1.9.1 — deprecated in transformers
# v5, HF_HOME is the modern equivalent and is read by both transformers
# and huggingface_hub.

# vLLM server settings
ENV VLLM_HOST="0.0.0.0"
ENV VLLM_PORT="8000"
ENV MAX_MODEL_LEN="32768"
ENV GPU_MEMORY_UTILIZATION="0.90"
ENV VLLM_EXTRA_ARGS=""

# context-compactor settings (port 8080 — what OpenWebUI talks to)
ENV COMPACTOR_HOST="0.0.0.0"
ENV COMPACTOR_PORT="8080"
# V2.1 Phase 6 Step 2: post-boot self-test auto-runs as a supervisord
# one-shot. Disable per-pod (e.g. for CI containers) by setting to "false".
ENV COMPACTOR_SELFTEST_ON_BOOT="true"
# V2.3 Theme 1: periodic data-durability backup daemon. Disable per-pod
# (e.g. CI containers) by setting to "false".
ENV COMPACTOR_BACKUP_ENABLED="true"
ENV COMPACTOR_TARGET_TOKENS=""
ENV COMPACTOR_KEEP_RECENT_TURNS="4"
ENV COMPACTOR_SUMMARY_MAX_TOKENS="1024"
ENV VLLM_URL="http://localhost:8000"

# V2.0 Phase 2 — facts memory
ENV COMPACTOR_FACTS_EXTRACTION="true"
ENV COMPACTOR_MAX_FACTS_TOKENS="1500"
ENV COMPACTOR_ADMIN_BIND="127.0.0.1"

# V2.0 Phase 3 — episodic memory (RAG). Embedding model baked into the
# image at /opt/embeddings; FASTEMBED_CACHE_PATH points there so no
# runtime download. RAG can be disabled with COMPACTOR_RAG_ENABLED=false.
ENV COMPACTOR_RAG_ENABLED="true"
ENV COMPACTOR_RAG_TOP_K="5"
ENV COMPACTOR_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
ENV FASTEMBED_CACHE_PATH="/opt/embeddings"

# V3.2 — Speech-to-text (Whisper) service. Runs in its own venv on STT_PORT.
# CPU by default so it never competes with vLLM for VRAM. Default model "base"
# is prebaked at /opt/whisper-models; swap via WHISPER_MODEL (+ WHISPER_DEVICE=
# cuda and/or a /data WHISPER_DOWNLOAD_ROOT for larger persistent models).
# Disable the whole service per-pod with STT_ENABLED=false.
ENV STT_ENABLED="true"
ENV WHISPER_MODEL="base"
ENV WHISPER_DEVICE="cpu"
ENV WHISPER_DOWNLOAD_ROOT="/opt/whisper-models"
ENV WHISPER_MODEL_ID="whisper-1"
ENV STT_HOST="0.0.0.0"
ENV STT_PORT="9000"

# V3.3 — Text-to-speech (Piper) service. Own venv on TTS_PORT, CPU + torch-free.
# Default voice en_US-lessac-medium is prebaked at /opt/tts-voices; swap via
# TTS_VOICE (+ a /data TTS_VOICE_DIR for other voices). Disable per-pod with
# TTS_ENABLED=false.
ENV TTS_ENABLED="true"
ENV TTS_VOICE="en_US-lessac-medium"
ENV TTS_VOICE_DIR="/opt/tts-voices"
ENV TTS_MODEL_ID="tts-1"
ENV TTS_HOST="0.0.0.0"
ENV TTS_PORT="9001"

# OpenWebUI settings — points at the compactor, not vLLM directly
ENV OPENWEBUI_PORT="3000"
ENV WEBUI_SECRET_KEY=""
ENV OLLAMA_BASE_URL=""
ENV OPENAI_API_BASE_URL="http://localhost:8080/v1"
ENV OPENAI_API_KEY="not-needed"
ENV ENABLE_OLLAMA_API="false"
ENV ENABLE_OPENAI_API="true"
ENV DATA_DIR="/data/openwebui"
ENV WEBUI_AUTH="true"

# V3.2 — wire OpenWebUI's STT to the local Whisper service (OpenAI engine).
# Disable voice input per-pod by setting AUDIO_STT_ENGINE="" (empty).
ENV AUDIO_STT_ENGINE="openai"
ENV AUDIO_STT_OPENAI_API_BASE_URL="http://localhost:9000/v1"
ENV AUDIO_STT_OPENAI_API_KEY="not-needed"
ENV AUDIO_STT_MODEL="whisper-1"

# V3.3 — wire OpenWebUI's TTS to the local Piper service (OpenAI engine).
# Disable voice output per-pod by setting AUDIO_TTS_ENGINE="" (empty). The voice
# field is sent by OpenWebUI but ignored by the service (it uses TTS_VOICE).
ENV AUDIO_TTS_ENGINE="openai"
ENV AUDIO_TTS_OPENAI_API_BASE_URL="http://localhost:9001/v1"
ENV AUDIO_TTS_OPENAI_API_KEY="not-needed"
ENV AUDIO_TTS_MODEL="tts-1"
ENV AUDIO_TTS_VOICE="alloy"

# 3000 — OpenWebUI (user-facing)
# 8080 — context-compactor (OpenAI-compatible, what OpenWebUI talks to)
# 8000 — vLLM (internal; can also be exposed for direct API access)
# 9000 — STT / Whisper (OpenAI audio API; OpenWebUI talks here for voice input)
# 9001 — TTS / Piper (OpenAI audio API; OpenWebUI talks here for voice output)
EXPOSE 8000 8080 3000 9000 9001

# V2.1 Phase 6: switch from `curl :3000` (OpenWebUI login page) to the
# compactor's /health/full deep probe. The old check stayed "healthy"
# even when vLLM was FATAL because OpenWebUI's login page kept serving;
# /health/full returns 503 when storage breaks and reports degraded
# status when vLLM is unreachable. start-period=300s covers model load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health/full || exit 1

ENTRYPOINT ["/entrypoint.sh"]
