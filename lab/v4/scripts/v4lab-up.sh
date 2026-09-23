#!/usr/bin/env bash
# One-command up for the V4 local lab. Idempotent: safe to re-run.
#
#   bash lab/v4/scripts/v4lab-up.sh
#
# Brings up Postgres, Synapse, Element, the tiny model + shim and the real
# compactor; creates the three test users; creates her E2EE room (once);
# then starts the bot. See LAB.md for what to do next (log in, import).
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"
ENV_FILE="${LAB_DIR}/.env.lab"

rand_pw() { python3 -c "import secrets; print(secrets.token_urlsafe(18))"; }

if [ ! -f "${ENV_FILE}" ]; then
    echo "==> creating ${ENV_FILE} with fresh random passwords (never committed) ..."
    cp "${LAB_DIR}/.env.lab.example" "${ENV_FILE}"
    sed -i "s|^LAB_HER_PASSWORD=.*|LAB_HER_PASSWORD=$(rand_pw)|" "${ENV_FILE}"
    sed -i "s|^LAB_OWNER_PASSWORD=.*|LAB_OWNER_PASSWORD=$(rand_pw)|" "${ENV_FILE}"
    sed -i "s|^LAB_BOT_PASSWORD=.*|LAB_BOT_PASSWORD=$(rand_pw)|" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
fi
# shellcheck disable=SC1090
set -a; source "${ENV_FILE}"; set +a

DC() { docker compose -f "${LAB_DIR}/docker-compose.lab.yml" -p v4lab --env-file "${ENV_FILE}" "$@"; }

echo "==> starting Postgres ..."
DC up -d v4lab-postgres
for i in $(seq 1 30); do
    DC exec -T v4lab-postgres pg_isready -U synapse >/dev/null 2>&1 && break
    sleep 2
done

echo "==> ensuring the bot's crypto-store database exists ..."
DC exec -T v4lab-postgres psql -U synapse -d synapse -tc \
    "SELECT 1 FROM pg_database WHERE datname='v4lab_botcrypto'" | grep -q 1 || \
    DC exec -T v4lab-postgres psql -U synapse -d synapse -c "CREATE DATABASE v4lab_botcrypto"

echo "==> initializing Synapse data dir (signing key + config) ..."
mkdir -p "${LAB_DIR}/synapse/data-run"
if [ ! -f "${LAB_DIR}/synapse/data-run/localhost.signing.key" ]; then
    docker run --rm \
        -v "${LAB_DIR}/synapse/data-run:/data" \
        -e SYNAPSE_SERVER_NAME=localhost \
        -e SYNAPSE_REPORT_STATS=no \
        matrixdotorg/synapse:latest generate
fi
# `generate` (and Synapse itself) chowns /data to its in-container uid
# (991). Rather than fight that (Synapse READS these files as uid 991 at
# every container start, so chowning to the host user breaks the next
# boot), make the whole dir world-rwX: cheap, unconditional, and safe --
# this is a throwaway lab signing key, not a production secret.
docker run --rm -v "${LAB_DIR}/synapse/data-run:/data" alpine \
    chmod -R a+rwX /data
cp "${LAB_DIR}/synapse/homeserver.lab.yaml" "${LAB_DIR}/synapse/data-run/homeserver.yaml"
cp "${LAB_DIR}/synapse/lab.log.config" "${LAB_DIR}/synapse/data-run/lab.log.config"
cp "${LAB_DIR}/synapse/appservice-registration.lab.yaml" "${LAB_DIR}/synapse/data-run/appservice-registration.lab.yaml"

echo "==> starting Synapse, Element, the model, the shim and the compactor ..."
DC up -d --build v4lab-synapse v4lab-element v4lab-model v4lab-model-shim v4lab-compactor

echo "==> waiting for Synapse to answer ..."
for i in $(seq 1 60); do
    DC exec -T v4lab-synapse curl -sf http://localhost:8008/_matrix/client/versions >/dev/null 2>&1 && break
    sleep 2
done

bash "${HERE}/v4lab-create-users.sh"
# shellcheck disable=SC1090
set -a; source "${ENV_FILE}"; set +a

if [ -z "${LAB_HER_ROOM_ID:-}" ]; then
    echo "==> creating her E2EE room ..."
    ROOM_ID=$(python3 "${HERE}/create_room.py" \
        --owner-password "${LAB_OWNER_PASSWORD}" \
        --her-password "${LAB_HER_PASSWORD}" | tail -1)
    if grep -q '^LAB_HER_ROOM_ID=' "${ENV_FILE}"; then
        sed -i "s|^LAB_HER_ROOM_ID=.*|LAB_HER_ROOM_ID=${ROOM_ID}|" "${ENV_FILE}"
    else
        echo "LAB_HER_ROOM_ID=${ROOM_ID}" >> "${ENV_FILE}"
    fi
    echo "    room created: ${ROOM_ID}"
else
    echo "==> her room already exists: ${LAB_HER_ROOM_ID}"
fi
# Re-source: the branch above only wrote LAB_HER_ROOM_ID to the FILE, and
# docker compose's --env-file interpolation is unreliable about picking up
# a file changed after the shell first sourced it. Exporting it here is
# what actually gets it into the bot container's environment.
# shellcheck disable=SC1090
set -a; source "${ENV_FILE}"; set +a

echo "==> starting the bot ..."
DC up -d --build v4lab-bot

echo ""
echo "==> up. Element:  http://localhost:18009"
echo "    Synapse:      http://localhost:18008"
echo "    Compactor:    http://localhost:18080"
echo "    Test users and their passwords are in ${ENV_FILE} (not in git)."
echo "    Next: scripts/v4lab-import.sh to import a copy of her history."
