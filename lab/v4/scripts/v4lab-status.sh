#!/usr/bin/env bash
# Quick health check across the stack. Prints container state, then hits
# each service's own health endpoint.
set -euo pipefail
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"
ENV_FILE="${LAB_DIR}/.env.lab"
[ -f "${ENV_FILE}" ] || ENV_FILE=/dev/null

DC() { docker compose -f "${LAB_DIR}/docker-compose.lab.yml" -p v4lab --env-file "${ENV_FILE}" "$@"; }

echo "== containers =="
DC ps

echo ""
echo "== synapse =="
curl -sf http://localhost:18008/_matrix/client/versions >/dev/null && echo "OK (18008)" || echo "DOWN"

echo "== element =="
curl -sf http://localhost:18009/ >/dev/null && echo "OK (18009)" || echo "DOWN"

echo "== compactor =="
curl -sf http://localhost:18080/health >/dev/null && echo "OK (18080)" || echo "DOWN"
curl -s http://localhost:18080/health/full 2>/dev/null | head -c 500; echo ""

echo "== model shim -> llama.cpp =="
DC exec -T v4lab-model-shim curl -sf http://v4lab-model:8000/v1/models >/dev/null 2>&1 && echo "OK" || echo "DOWN or still loading the model"

echo ""
echo "== logs: use e.g. =="
echo "  docker compose -f ${LAB_DIR}/docker-compose.lab.yml -p v4lab logs -f v4lab-bot"
echo "  docker compose -f ${LAB_DIR}/docker-compose.lab.yml -p v4lab logs -f v4lab-compactor"
