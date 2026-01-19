# Multi-stage Dockerfile for llama.cpp + OpenWebUI on Runpod
# Optimized for NVIDIA A40 (48GB VRAM)
# Uses CUDA 12.6.3 on Ubuntu 24.04 (Python 3.12 for Open WebUI compatibility)

# =============================================================================
# Stage 1: Build llama.cpp with CUDA support
# =============================================================================
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS llama-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp with CUDA support
WORKDIR /build

# Create symlinks to CUDA stub library for linking during build
# The real libcuda.so will be provided by the NVIDIA driver at runtime
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build build --config Release -j$(nproc) --target llama-server llama-cli

# =============================================================================
# Stage 2: Runtime image with llama.cpp + OpenWebUI
# =============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime dependencies (Ubuntu 24.04 has Python 3.12 which satisfies Open WebUI requirements)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    libgomp1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy llama.cpp binaries and libraries from builder
COPY --from=llama-builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=llama-builder /build/llama.cpp/build/bin/llama-cli /usr/local/bin/

# Copy shared libraries (use shell to handle globs)
RUN mkdir -p /tmp/libs
COPY --from=llama-builder /build/llama.cpp/build/ /tmp/build/
RUN find /tmp/build -name "*.so*" -exec cp {} /usr/local/lib/ \; && rm -rf /tmp/build

# Update library cache
RUN ldconfig

# Create model directory
RUN mkdir -p /models

# =============================================================================
# Model Configuration (defaults - override via .env or environment)
# =============================================================================
# Default: Qwen3-24B-A4B MoE Q6_K for production (A40/A100)
# Override MODEL_REPO and MODEL_FILE for different models
# MODEL_PATH is computed at runtime in entrypoint.sh
ENV MODEL_REPO="DavidAU/Qwen3-24B-A4B-Freedom-Thinking-Abliterated-Heretic-NEO-Imatrix-GGUF"
ENV MODEL_FILE="Qwen3-24B-A4B-Freedom-Think-Ablit-Heretic-Neo-D_AU-Q6_K-imat.gguf"

# =============================================================================
# Install OpenWebUI
# =============================================================================
WORKDIR /app

# Create virtual environment for OpenWebUI
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install OpenWebUI
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir open-webui

# Create data directories for OpenWebUI
RUN mkdir -p /app/data /app/data/uploads /app/data/cache

# =============================================================================
# Configure Supervisor for process management
# =============================================================================
RUN mkdir -p /var/log/supervisor

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# =============================================================================
# Environment Variables
# =============================================================================
# llama.cpp server settings (optimized for A40)
ENV LLAMA_HOST="0.0.0.0"
ENV LLAMA_PORT="8080"
ENV LLAMA_CTX_SIZE="32768"
ENV LLAMA_N_GPU_LAYERS="999"
ENV LLAMA_PARALLEL="4"
ENV LLAMA_CONT_BATCHING="true"

# OpenWebUI settings
ENV OPENWEBUI_PORT="3000"
ENV WEBUI_SECRET_KEY=""
ENV OLLAMA_BASE_URL=""
ENV OPENAI_API_BASE_URL="http://localhost:8080/v1"
ENV OPENAI_API_KEY="not-needed"
ENV ENABLE_OLLAMA_API="false"
ENV ENABLE_OPENAI_API="true"
ENV DEFAULT_MODELS="qwen3-24b-a4b"
ENV DATA_DIR="/app/data"
ENV WEBUI_AUTH="true"

# Expose ports
# 8080 - llama.cpp API server (OpenAI-compatible)
# 3000 - OpenWebUI frontend
EXPOSE 8080 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
