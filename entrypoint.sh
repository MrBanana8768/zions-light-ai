#!/bin/bash
set -e

echo "=============================================="
echo "  Zion's Light AI - Startup"
echo "  Backend: vLLM"
echo "  Model:   ${MODEL_REPO}"
echo "  Cache:   ${HF_HOME}"
echo "  Ctx:    ${MAX_MODEL_LEN} tokens (compactor target: ${COMPACTOR_TARGET_TOKENS:-auto})"
echo "=============================================="

# =============================================================================
# Preflight checks — fail loud and fast with actionable messages instead of
# letting vLLM crash 2-3 minutes into its startup with a cryptic stack trace.
# =============================================================================
echo "[1/3] Preflight checks..."

# Check 1: /data volume is writable. If not, the pod has no persistence and
# both model cache + OpenWebUI state will be lost on every restart.
if ! touch /data/.write-test 2>/dev/null; then
    echo "      ERROR: /data is not writable. Did you attach a Network Volume?"
    echo "             Expected mount: /data (single shared volume per RUNPOD_DEPLOY.md)"
    exit 1
fi
rm -f /data/.write-test
echo "      /data is writable"

# Create persistent subdirs on the volume (empty on first attach).
mkdir -p "${HF_HOME}" "${DATA_DIR}" /data/vllm-compile-cache

# Check 2: GPU is visible. nvidia-smi runs cleanly = host driver passthrough
# is working. If this fails, the container was started without --gpus all
# (or RunPod's equivalent).
if ! nvidia-smi >/dev/null 2>&1; then
    echo "      ERROR: nvidia-smi failed. No GPU passthrough?"
    echo "             For RunPod: confirm pod has a GPU attached."
    echo "             For local docker: use 'docker compose up' (compose file requests GPU)."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
echo "      GPU: ${GPU_NAME} (${GPU_MEM} MiB), driver ${DRIVER}"

# Check 3: driver version satisfies what our torch/vLLM wheels need. The CUDA
# channel this image was built against is baked in as ${TORCH_CUDA}; CUDA 13
# (cu130) needs a much newer driver than CUDA 12 (cu128/cu126). Bail with an
# actionable message rather than letting torch/vLLM crash later with "NVIDIA
# driver too old" or "libcudart.so.NN: cannot open shared object file".
DRIVER_MAJOR=$(echo "${DRIVER}" | cut -d. -f1)
case "${TORCH_CUDA:-cu128}" in
    cu130) MIN_DRIVER=580; CUDA_LABEL="CUDA 13 (cu130)" ;;
    cu128) MIN_DRIVER=525; CUDA_LABEL="CUDA 12.8 (cu128)" ;;
    cu126) MIN_DRIVER=525; CUDA_LABEL="CUDA 12.6 (cu126)" ;;
    *)     MIN_DRIVER=525; CUDA_LABEL="${TORCH_CUDA:-unknown}" ;;
esac
if [ "${DRIVER_MAJOR}" -lt "${MIN_DRIVER}" ] 2>/dev/null; then
    echo "      ERROR: Driver ${DRIVER} is too old for this image (${CUDA_LABEL})."
    echo "             Need driver >= ${MIN_DRIVER}."
    if [ "${MIN_DRIVER}" -ge 580 ]; then
        echo "             This is the CUDA-13 build; it needs a driver-580+ host."
        echo "             Deploy on a newer host, or build the CUDA-12 fallback:"
        echo "               --build-arg CUDA_BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \\"
        echo "               --build-arg TORCH_CUDA=cu128 --build-arg VLLM_VERSION=0.19.0"
    else
        echo "             Pick a RunPod GPU/host with a newer driver."
    fi
    exit 1
fi
echo "      driver ${DRIVER} OK for ${CUDA_LABEL} (needs >= ${MIN_DRIVER})"

# Symlink vLLM's torch.compile cache onto the persistent volume. Without
# this, every cold start re-runs the 60-120s CUDA graph capture even
# though the cache key would have hit. Symlink is idempotent — re-runs
# are no-ops.
if [ ! -L /root/.cache/vllm ]; then
    mkdir -p /root/.cache
    rm -rf /root/.cache/vllm
    ln -s /data/vllm-compile-cache /root/.cache/vllm
    echo "      torch.compile cache linked to /data/vllm-compile-cache"
fi

echo ""

# =============================================================================
# Network / HuggingFace reachability check.
# =============================================================================
echo "[2/3] Checking HuggingFace connectivity..."
HF_READY=false
for i in $(seq 1 30); do
    if curl -sf --max-time 5 "https://huggingface.co" > /dev/null 2>&1; then
        echo "      HuggingFace is reachable."
        HF_READY=true
        break
    else
        echo "      Waiting for network... (attempt $i/30)"
        sleep 2
    fi
done

if [ "$HF_READY" = false ]; then
    echo "ERROR: Cannot reach HuggingFace after 60 seconds. Check network connectivity."
    exit 1
fi

# Generate OpenWebUI secret key if not set
if [ -z "${WEBUI_SECRET_KEY}" ]; then
    export WEBUI_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    echo "      Generated WebUI secret key"
fi

# =============================================================================
# Hand off to supervisord. vLLM, context-compactor, and OpenWebUI all run
# as supervised child processes from here.
# =============================================================================
echo ""
echo "[3/3] Starting services..."
echo "      - vLLM             on port ${VLLM_PORT}      (internal)"
echo "      - context-compactor on port ${COMPACTOR_PORT}  (OpenWebUI talks here)"
echo "      - OpenWebUI        on port ${OPENWEBUI_PORT}  (user-facing)"
echo ""
echo "      Note: vLLM downloads model weights on first run; first startup"
echo "      may take 5-15 minutes depending on model size and network speed."
echo "      Weights are cached to ${HF_HOME} (persist via volume mount)."
echo "      torch.compile cache lives at /data/vllm-compile-cache — second"
echo "      and later cold starts skip the 60-120s CUDA graph capture."
echo "=============================================="

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
