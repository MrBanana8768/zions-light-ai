#!/usr/bin/env bash
# V4 lab hard safety rule, enforced in code: the compactor may never talk to
# the live pod's vLLM. Check VLLM_URL before doing anything else, refuse to
# start otherwise.
set -euo pipefail

python3 /opt/compactor/lab_guard.py compactor "VLLM_URL=${VLLM_URL:-}"

cd /opt/compactor
exec python3 -m uvicorn main:app \
    --host "${COMPACTOR_HOST:-0.0.0.0}" \
    --port "${COMPACTOR_PORT:-8080}"
