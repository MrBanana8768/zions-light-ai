#!/bin/bash
set -e

# Compute MODEL_PATH from MODEL_FILE if not explicitly set
export MODEL_PATH="${MODEL_PATH:-/models/${MODEL_FILE}}"

echo "=============================================="
echo "  Zion's Light AI - Startup"
echo "  Model: ${MODEL_FILE}"
echo "  Path:  ${MODEL_PATH}"
echo "=============================================="

# =============================================================================
# Download model if not present or empty
# =============================================================================
# Create models directory if it doesn't exist
mkdir -p /models

# Check if model exists AND has size > 0 (handles failed downloads that left empty files)
if [ -s "${MODEL_PATH}" ]; then
    echo "[1/3] Model already present, skipping download."
else
    # Wait for network/HuggingFace to be reachable (handles cold starts)
    echo "[1/3] Checking HuggingFace connectivity..."
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
    # Clean up any empty/partial file from failed download
    rm -f "${MODEL_PATH}"

    echo "[1/3] Downloading model from HuggingFace..."
    echo "      Repository: ${MODEL_REPO}"
    echo "      File: ${MODEL_FILE}"

    # Download with retries for transient network issues
    MAX_RETRIES=3
    RETRY_DELAY=10

    for i in $(seq 1 $MAX_RETRIES); do
        echo "      Attempt $i of $MAX_RETRIES..."

        if wget --progress=bar:force:noscroll \
            --tries=3 \
            --timeout=30 \
            "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" \
            -O "${MODEL_PATH}"; then
            echo "      Download complete!"
            break
        else
            echo "      Download failed, cleaning up..."
            rm -f "${MODEL_PATH}"

            if [ $i -lt $MAX_RETRIES ]; then
                echo "      Retrying in ${RETRY_DELAY} seconds..."
                sleep $RETRY_DELAY
            fi
        fi
    done
fi

# Verify model file exists and has size > 0
if [ ! -s "${MODEL_PATH}" ]; then
    echo "ERROR: Model file is missing or empty after all download attempts!"
    exit 1
fi

MODEL_SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
echo "      Model size: ${MODEL_SIZE}"

# =============================================================================
# Generate OpenWebUI secret key if not set
# =============================================================================
if [ -z "${WEBUI_SECRET_KEY}" ]; then
    export WEBUI_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    echo "[2/3] Generated WebUI secret key"
else
    echo "[2/3] Using provided WebUI secret key"
fi

# =============================================================================
# Start services via Supervisor
# =============================================================================
echo "[3/3] Starting services..."
echo "      - llama.cpp server on port ${LLAMA_PORT}"
echo "      - OpenWebUI on port ${OPENWEBUI_PORT}"
echo "=============================================="

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
