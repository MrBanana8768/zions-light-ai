#!/usr/bin/env bash
# Imports a COPY of her history into her lab room. Resumable -- re-run to
# continue after any failure; the importer's own journal (lab/v4/importer/
# out/journal.db) tracks per-turn progress.
#
#   bash lab/v4/scripts/v4lab-import.sh              # full import (~3,950 turns)
#   bash lab/v4/scripts/v4lab-import.sh --last 200    # fast partial import for a quick test
#   bash lab/v4/scripts/v4lab-import.sh --count-only  # just report the branch length
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"
ENV_FILE="${LAB_DIR}/.env.lab"

if [ ! -f "${ENV_FILE}" ]; then
    echo "REFUSING: ${ENV_FILE} not found. Run v4lab-up.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1090
set -a; source "${ENV_FILE}"; set +a

if [ -z "${LAB_HER_ROOM_ID:-}" ]; then
    echo "REFUSING: LAB_HER_ROOM_ID is not set in ${ENV_FILE}. Run v4lab-up.sh first." >&2
    exit 1
fi
if [ ! -f "${LAB_DATA_DIR:-/home/drew/scratch/v4lab/data}/webui.db" ]; then
    echo "REFUSING: no copy of webui.db found. Run v4lab-copy-data.sh first." >&2
    exit 1
fi

mkdir -p "${LAB_DIR}/importer/out"

docker compose -f "${LAB_DIR}/docker-compose.lab.yml" \
    -p v4lab --env-file "${ENV_FILE}" --profile tools \
    run --build --rm v4lab-importer python3 import_history.py \
    --db /data/her-copy/webui.db \
    --room "${LAB_HER_ROOM_ID}" \
    --her-password "${LAB_HER_PASSWORD}" \
    "$@"
