# Dockerfile for vLLM + context-compactor + OpenWebUI on Runpod
# Optimized for NVIDIA A40 (48GB VRAM) and larger
# CUDA 12.6.3 runtime on Ubuntu 24.04 (Python 3.12)

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Runtime dependencies (+ binutils for strip during install layers, kept ~10MB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    libgomp1 \
    supervisor \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Persistent data root — mounted as a single network volume in production.
# Holds both the model cache (/data/models -> HF_HOME) and OpenWebUI state
# (/data/openwebui -> DATA_DIR). entrypoint.sh creates the subdirs on first run.
RUN mkdir -p /data

ENV VLLM_VENV=/opt/vllm-venv

# =============================================================================
# vLLM venv + compactor python deps — installed, stripped, and cache-cleared
# in a SINGLE layer so unstripped libs and .pyc caches never get committed.
# COPY just the requirements file (not full compactor/) so editing main.py
# later doesn't bust this expensive layer.
# =============================================================================
COPY compactor/requirements.txt /opt/compactor/requirements.txt
RUN python3 -m venv /opt/vllm-venv && \
    /opt/vllm-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/vllm-venv/bin/pip install --no-cache-dir vllm && \
    /opt/vllm-venv/bin/pip install --no-cache-dir -r /opt/compactor/requirements.txt && \
    find /opt/vllm-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/vllm-venv -name "*.pyc" -delete && \
    find /opt/vllm-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# =============================================================================
# OpenWebUI venv — kept isolated from vLLM's pytorch pin. Same install+strip
# atomic pattern.
# =============================================================================
WORKDIR /app
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir open-webui && \
    find /app/venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /app/venv -name "*.pyc" -delete && \
    find /app/venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/* && \
    mkdir -p /app/data /app/data/uploads /app/data/cache

# Compactor sources copied AFTER the expensive install layer so editing
# main.py doesn't invalidate the vllm install cache.
COPY compactor/main.py /opt/compactor/main.py

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
ENV TRANSFORMERS_CACHE="/data/models"

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
