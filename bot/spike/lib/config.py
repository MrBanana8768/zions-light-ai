"""Shared config for the Phase-1 spike scripts.

Everything here points at the throwaway `zlaspike` dev stack copied into
SP\\v4spike\\chat-app and run inside WSL Debian's native Docker. None of this
touches production values.
"""
import os

# Synapse, as seen from other containers on the zlaspike_default network.
HS_URL = os.environ.get("SPIKE_HS_URL", "http://synapse:8008")
SERVER_NAME = "localhost"

ALICE_MXID = "@alice_spike:localhost"
ALICE_PASSWORD = os.environ.get("SPIKE_ALICE_PASSWORD", "CHANGE_ME")

BOT_MXID = "@zla_bot:localhost"
BOT_PASSWORD = os.environ.get("SPIKE_BOT_PASSWORD", "CHANGE_ME")

ADMIN_MXID = "@spike_admin:localhost"
ADMIN_PASSWORD = os.environ.get("SPIKE_ADMIN_PASSWORD", "CHANGE_ME")

AS_TOKEN = os.environ.get("SPIKE_AS_TOKEN", "CHANGE_ME_as_token")
HS_TOKEN = os.environ.get("SPIKE_HS_TOKEN", "CHANGE_ME_hs_token")
AS_BOT_MXID = "@zlaspike_as_bot:localhost"

PG_DSN = os.environ.get(
    "SPIKE_PG_DSN", "postgresql://synapse:devpassword@postgres:5432/synapse"
)
# Separate logical "schema" via table prefix isn't supported by mautrix's
# asyncpg store directly, so instead each identity gets its own account_id
# row inside the shared crypto_* tables (mautrix's intended usage for
# multi-account/bridge processes) inside a dedicated botcrypto database.
PG_CRYPTO_DSN = os.environ.get(
    "SPIKE_PG_CRYPTO_DSN", "postgresql://synapse:devpassword@postgres:5432/botcrypto"
)

STUB_COMPACTOR_URL = os.environ.get("SPIKE_COMPACTOR_URL", "http://stub-compactor:8090/v1/chat")
COMPACTOR_BEARER = "spike-compactor-key-do-not-use-in-prod"
