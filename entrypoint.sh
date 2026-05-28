#!/bin/bash
set -e

echo "=============================================="
echo "  Zion's Light AI - Startup"
echo "  Backend: vLLM"
echo "  Model:   ${MODEL_REPO}"
echo "  Cache:   ${HF_HOME}"
echo "  Ctx:    ${MAX_MODEL_LEN} tokens (compactor target: ${COMPACTOR_TARGET_TOKENS:-auto})"
echo "=============================================="

# Create both subdirs on the persistent volume (empty on first attach)
mkdir -p "${HF_HOME}" "${DATA_DIR}"

# Wait for network/HuggingFace to be reachable (handles cold starts on Runpod)
echo "[1/2] Checking HuggingFace connectivity..."
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

echo "[2/2] Starting services..."
echo "      - vLLM             on port ${VLLM_PORT}      (internal)"
echo "      - context-compactor on port ${COMPACTOR_PORT}  (OpenWebUI talks here)"
echo "      - OpenWebUI        on port ${OPENWEBUI_PORT}  (user-facing)"
echo ""
echo "      Note: vLLM downloads model weights on first run; first startup"
echo "      may take 5-15 minutes depending on model size and network speed."
echo "      Weights are cached to ${HF_HOME} (persist via volume mount)."
echo "=============================================="

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
