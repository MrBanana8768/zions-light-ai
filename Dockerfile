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
# vLLM venv + compactor python deps — installed, stripped, and cache-cleared
# in a SINGLE layer so unstripped libs and .pyc caches never get committed.
# --extra-index-url pulls PyTorch from the chosen CUDA channel explicitly to
# keep pip from resolving torch to a different cu variant.
# COPY just the requirements file (not full compactor/) so editing main.py
# later doesn't bust this expensive layer.
# Also bumps pip/setuptools/wheel as part of the same layer — both resolve
# Scout-flagged Highs (CVE in setuptools 68.x and wheel 0.42.x).
# =============================================================================
COPY compactor/requirements.txt /opt/compactor/requirements.txt
RUN python3 -m venv /opt/vllm-venv && \
    /opt/vllm-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/vllm-venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
        vllm==${VLLM_VERSION} && \
    /opt/vllm-venv/bin/pip install --no-cache-dir -r /opt/compactor/requirements.txt && \
    find /opt/vllm-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/vllm-venv -name "*.pyc" -delete && \
    find /opt/vllm-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

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

# 3000 — OpenWebUI (user-facing)
# 8080 — context-compactor (OpenAI-compatible, what OpenWebUI talks to)
# 8000 — vLLM (internal; can also be exposed for direct API access)
EXPOSE 8000 8080 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
