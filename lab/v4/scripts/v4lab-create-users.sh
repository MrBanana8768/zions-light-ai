#!/usr/bin/env bash
# Creates the three test users with the ADMIN CLI (registration is closed
# on this homeserver -- see synapse/homeserver.lab.yaml). Idempotent: an
# already-registered user is reported and skipped, not treated as failure.
# Also sets Synapse's per-user rate-limit override for @bot:localhost.
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${HERE}/.." && pwd)"
ENV_FILE="${LAB_DIR}/.env.lab"

if [ ! -f "${ENV_FILE}" ]; then
    echo "REFUSING: ${ENV_FILE} not found. Run v4lab-up.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1090
set -a; source "${ENV_FILE}"; set +a

DC() { docker compose -f "${LAB_DIR}/docker-compose.lab.yml" -p v4lab --env-file "${ENV_FILE}" "$@"; }

register() {
    local username="$1" password="$2" admin_flag="$3"
    echo "==> registering @${username}:localhost ..."
    if DC exec -T v4lab-synapse register_new_matrix_user \
        -u "${username}" -p "${password}" "${admin_flag}" \
        -c /data/homeserver.yaml http://localhost:8008 2>&1 | tee /dev/stderr | grep -qi "success"; then
        echo "    registered."
    else
        echo "    already exists or registration reported an issue (continuing; re-run is idempotent)."
    fi
}

register her "${LAB_HER_PASSWORD}" --no-admin
register owner "${LAB_OWNER_PASSWORD}" --admin
register bot "${LAB_BOT_PASSWORD}" --no-admin

echo "==> logging in as @owner to get an admin token ..."
ADMIN_LOGIN=$(DC exec -T v4lab-synapse curl -s -X POST http://localhost:8008/_matrix/client/v3/login \
    -H "Content-Type: application/json" \
    -d "{\"type\":\"m.login.password\",\"identifier\":{\"type\":\"m.id.user\",\"user\":\"owner\"},\"password\":\"${LAB_OWNER_PASSWORD}\"}")
ADMIN_TOKEN=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['access_token'])" "${ADMIN_LOGIN}" 2>/dev/null || true)

if [ -z "${ADMIN_TOKEN}" ]; then
    echo "WARNING: could not extract an admin token; skipping the bot rate-limit override." >&2
else
    echo "==> setting the per-user rate-limit override for @bot:localhost ..."
    DC exec -T v4lab-synapse curl -s -X POST \
        "http://localhost:8008/_synapse/admin/v1/users/@bot:localhost/override_ratelimit" \
        -H "Authorization: Bearer ${ADMIN_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{"messages_per_second": 0, "burst_count": 0}' \
        && echo "    override set (0/0 = unlimited, Synapse's own convention)."

    # Persist the admin token for create_room.py and later admin calls.
    if grep -q '^LAB_ADMIN_TOKEN=' "${ENV_FILE}"; then
        sed -i "s|^LAB_ADMIN_TOKEN=.*|LAB_ADMIN_TOKEN=${ADMIN_TOKEN}|" "${ENV_FILE}"
    else
        echo "LAB_ADMIN_TOKEN=${ADMIN_TOKEN}" >> "${ENV_FILE}"
    fi
fi

echo "==> users ready: @her:localhost, @owner:localhost, @bot:localhost"
