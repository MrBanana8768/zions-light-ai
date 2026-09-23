#!/usr/bin/env bash
# Wipes the lab's OWN state back to fresh: containers, volumes, Synapse
# data-run, the importer's journal, and the room-id/admin-token lines in
# .env.lab (so v4lab-up.sh recreates the room next time). Does NOT touch
# her data copies under LAB_DATA_DIR (pass --purge-data to also delete
# those) and never touches the read-only pod-export source.
set -euo pipefail
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"
ENV_FILE="${LAB_DIR}/.env.lab"
[ -f "${ENV_FILE}" ] || ENV_FILE=/dev/null

docker compose -f "${LAB_DIR}/docker-compose.lab.yml" \
    -p v4lab --env-file "${ENV_FILE}" --profile tools down -v

rm -rf "${LAB_DIR}/synapse/data-run"
rm -rf "${LAB_DIR}/importer/out"

if [ -f "${ENV_FILE}" ]; then
    sed -i '/^LAB_HER_ROOM_ID=/d; /^LAB_ADMIN_TOKEN=/d' "${ENV_FILE}"
fi

if [ "${1:-}" = "--purge-data" ]; then
    DEST="${LAB_DATA_DIR:-/home/drew/scratch/v4lab/data}"
    echo "==> --purge-data: removing ${DEST} (the copies, NOT the read-only source)"
    rm -rf "${DEST}"
fi

echo "==> reset complete. Run v4lab-up.sh to bring the lab back up."
