#!/usr/bin/env bash
# Stops the lab. Containers only -- data (Postgres volume, Synapse
# data-run, her data copies) survives. Use v4lab-reset.sh to wipe state.
set -euo pipefail
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"
ENV_FILE="${LAB_DIR}/.env.lab"
[ -f "${ENV_FILE}" ] || ENV_FILE=/dev/null
docker compose -f "${LAB_DIR}/docker-compose.lab.yml" \
    -p v4lab --env-file "${ENV_FILE}" --profile tools down
echo "==> down. Volumes and data copies were left in place."
