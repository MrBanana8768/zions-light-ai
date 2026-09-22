"""Phase-1 spike bot: claims 1 and 4 of V4_MATRIX_CLIENT.md.

An ordinary logged-in Matrix user, ONE device, mautrix-python's OlmMachine
backed by Postgres (PgCryptoStore + PgCryptoStateStore), auto-join on
invite, decrypt incoming messages, call the stub compactor with
X-Conversation-Id + Authorization: Bearer, encrypt and send the reply.

Fixed device_id across restarts so re-logging in after a container rebuild
reuses the same E2EE device identity instead of minting a new one (that's
the whole point of claim 4 -- the user's client should not see a "new
unverified device" warning after a rebuild).
"""
import asyncio
import logging
import sys
import time

import aiohttp
from mautrix.client import Client
from mautrix.crypto import OlmMachine, PgCryptoStore
from mautrix.crypto.store.asyncpg import PgCryptoStateStore
from mautrix.types import EventType, Membership, MessageEvent, StateEvent, TextMessageEventContent
from mautrix.util.async_db import Database

sys.path.insert(0, "/work")
from lib import config
from lib.convid import conversation_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("spike.bot")

DEVICE_ID = "ZLASPIKE_BOT_DEVICE"  # fixed across restarts/rebuilds -- see claim 4
PICKLE_KEY = "zlaspike-bot-pickle-key-not-for-prod"


async def main() -> None:
    db_crypto = Database.create(
        url=config.PG_CRYPTO_DSN, upgrade_table=PgCryptoStore.upgrade_table
    )
    db_state = Database.create(
        url=config.PG_CRYPTO_DSN, upgrade_table=PgCryptoStateStore.upgrade_table
    )
    await db_crypto.start()
    await db_state.start()

    crypto_store = PgCryptoStore(config.BOT_MXID, PICKLE_KEY, db_crypto)
    state_store = PgCryptoStateStore(db_state)

    client = Client(
        base_url=config.HS_URL,
        mxid=config.BOT_MXID,
        device_id=DEVICE_ID,
        sync_store=crypto_store,
        state_store=state_store,
    )

    login_resp = await client.login(
        password=config.BOT_PASSWORD,
        device_id=DEVICE_ID,
        device_name="zlaspike bot device",
    )
    log.info(f"Logged in as {login_resp.user_id} device={login_resp.device_id}")

    machine = OlmMachine(client, crypto_store, state_store)
    await machine.load()
    client.crypto = machine
    await machine.share_keys()
    log.info(f"Olm identity key: {machine.account.identity_key}")

    http_session = aiohttp.ClientSession()

    async def on_member(evt: StateEvent) -> None:
        if evt.state_key == config.BOT_MXID and evt.content.membership == Membership.INVITE:
            log.info(f"Invited to {evt.room_id}, joining")
            await client.join_room(evt.room_id)

    start_ts_ms = int(time.time() * 1000)

    async def on_message(evt: MessageEvent) -> None:
        if evt.sender == config.BOT_MXID:
            return
        if evt.timestamp < start_ts_ms:
            # Don't replay/reply to backlog (e.g. the claim-2 import history) on
            # (re)connect -- only react to messages sent after this process started.
            log.info(f"Ignoring pre-start message {evt.event_id} (ts={evt.timestamp})")
            return
        body = evt.content.body if hasattr(evt.content, "body") else ""
        log.info(f"Decrypted message from {evt.sender} in {evt.room_id}: {body!r}")
        conv_id = conversation_id(evt.room_id)
        try:
            async with http_session.post(
                config.STUB_COMPACTOR_URL,
                json={"message": body, "room_id": evt.room_id, "sender": evt.sender},
                headers={
                    "X-Conversation-Id": conv_id,
                    "Authorization": f"Bearer {config.COMPACTOR_BEARER}",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                reply = data.get("reply", "(no reply)")
        except Exception:
            log.exception("Compactor call failed")
            reply = "Sorry, I could not reach the compactor."
        await client.send_text(evt.room_id, reply)
        log.info(f"Replied (encrypted) in {evt.room_id}")

    client.add_event_handler(EventType.ROOM_MEMBER, on_member)
    client.add_event_handler(EventType.ROOM_MESSAGE, on_message)

    log.info("Starting sync loop")
    await client.start(None)


if __name__ == "__main__":
    asyncio.run(main())
