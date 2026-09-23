"""V4 local lab bot -- L2.

An ordinary logged-in Matrix user (mautrix-python's OlmMachine, Postgres-
backed crypto store, ONE pinned device id), in her E2EE room, talking to
the REAL compactor (this branch's compactor/) over X-Conversation-Id.

What this does, matching the L2 spec:
  - joins the E2EE room the setup script created and invited it to;
  - maps her room EXPLICITLY to HER_CONV_ID (the real conversation id --
    safe here only because COMPACTOR_STORAGE_ROOT points at a COPY of her
    memory, which the guard below cannot see, but the localhost/private
    check on every URL is what keeps this whole process from ever reaching
    the live compactor);
  - sends X-Conversation-Id, a typing indicator, and streams the reply back
    as edits at about 1/s;
  - passes `!` commands through unchanged -- the compactor parses those
    itself from the message text (commands.parse_command in main.py);
  - posts a failure notice in the room on any error instead of going silent;
  - ignores messages sent before this process started, so a history import
    running through the importer's own client never triggers a reply.

Streaming-by-edit mechanics (Matrix): send a placeholder, then repeatedly
send `m.replace` edits with the accumulated text, at most once per second,
with a final edit that always fires once the compactor's stream ends. This
lab does not implement the full V4_DESIGN split-by-encrypted-size scheme
(the 40,000-byte budget) -- her real turns are long enough to hit that in
production, but the lab's tiny model's replies are short. Documented gap;
see LAB.md.
"""
import asyncio
import json
import logging
import os
import sys
import time

import aiohttp
from mautrix.client import Client
from mautrix.crypto import OlmMachine, PgCryptoStore
from mautrix.crypto.store.asyncpg import PgCryptoStateStore
from mautrix.types import (
    EventType,
    Membership,
    MessageEvent,
    RelatesTo,
    RelationType,
    StateEvent,
    TextMessageEventContent,
)
from mautrix.util.async_db import Database

sys.path.insert(0, os.path.dirname(__file__))
from guard import assert_local_only  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s lab_bot %(levelname)s %(message)s")
log = logging.getLogger("v4lab.bot")

SYNAPSE_URL = os.environ.get("SYNAPSE_URL", "http://v4lab-synapse:8008")
COMPACTOR_URL = os.environ.get("COMPACTOR_URL", "http://v4lab-compactor:8080").rstrip("/")
BOT_MXID = os.environ.get("BOT_MXID", "@bot:localhost")
BOT_PASSWORD = os.environ.get("BOT_PASSWORD")
PG_CRYPTO_DSN = os.environ.get("PG_CRYPTO_DSN")
HER_ROOM_ID = os.environ.get("HER_ROOM_ID", "")
HER_CONV_ID = os.environ.get("HER_CONV_ID", "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e")

assert_local_only(
    {"SYNAPSE_URL": SYNAPSE_URL, "COMPACTOR_URL": COMPACTOR_URL},
    component="bot",
)

DEVICE_ID = "V4LAB_BOT_DEVICE"  # fixed across restarts -- claim-4 style device pinning
PICKLE_KEY = "v4lab-bot-pickle-key-lab-only-not-for-prod"

EDIT_INTERVAL_S = 1.0
TYPING_REFRESH_S = 20.0


async def main() -> None:
    if not BOT_PASSWORD:
        log.error("BOT_PASSWORD not set; refusing to start.")
        raise SystemExit(1)
    if not PG_CRYPTO_DSN:
        log.error("PG_CRYPTO_DSN not set; refusing to start.")
        raise SystemExit(1)

    db_crypto = Database.create(url=PG_CRYPTO_DSN, upgrade_table=PgCryptoStore.upgrade_table)
    db_state = Database.create(url=PG_CRYPTO_DSN, upgrade_table=PgCryptoStateStore.upgrade_table)
    await db_crypto.start()
    await db_state.start()

    crypto_store = PgCryptoStore(BOT_MXID, PICKLE_KEY, db_crypto)
    state_store = PgCryptoStateStore(db_state)

    client = Client(
        base_url=SYNAPSE_URL,
        mxid=BOT_MXID,
        device_id=DEVICE_ID,
        sync_store=crypto_store,
        state_store=state_store,
    )

    login_resp = await client.login(
        password=BOT_PASSWORD, device_id=DEVICE_ID, device_name="v4lab bot",
    )
    log.info(f"logged in as {login_resp.user_id} device={login_resp.device_id}")

    machine = OlmMachine(client, crypto_store, state_store)
    await machine.load()
    client.crypto = machine
    await machine.share_keys()
    log.info(f"olm identity key: {machine.account.identity_key}")

    http = aiohttp.ClientSession()
    start_ts_ms = int(time.time() * 1000)

    async def on_member(evt: StateEvent) -> None:
        if evt.state_key == BOT_MXID and evt.content.membership == Membership.INVITE:
            log.info(f"invited to {evt.room_id}, joining")
            await client.join_room(evt.room_id)

    async def send_failure_notice(room_id: str, reason: str) -> None:
        try:
            await client.send_text(room_id, f"[v4lab-bot] Sorry, something went wrong: {reason}")
        except Exception:
            log.exception("even the failure notice failed to send")

    async def send_edit(room_id: str, target_event_id: str, text: str) -> None:
        content = TextMessageEventContent(msgtype="m.text", body=f"* {text}")
        content.set_edit(target_event_id)
        await client.send_message_event(room_id, EventType.ROOM_MESSAGE, content)

    async def keep_typing(room_id: str, stop: asyncio.Event) -> None:
        try:
            while not stop.is_set():
                await client.set_typing(room_id, timeout=int((TYPING_REFRESH_S + 5) * 1000))
                try:
                    await asyncio.wait_for(stop.wait(), timeout=TYPING_REFRESH_S)
                except asyncio.TimeoutError:
                    pass
        finally:
            try:
                await client.set_typing(room_id, timeout=0)
            except Exception:
                pass

    async def stream_reply(room_id: str, conv_id: str, user_text: str, placeholder_event_id: str) -> None:
        full_text = ""
        last_edit_at = 0.0
        got_any = False
        try:
            payload = {
                "model": "lab",
                "messages": [{"role": "user", "content": user_text}],
                "stream": True,
            }
            headers = {"X-Conversation-Id": conv_id, "Content-Type": "application/json"}
            async with http.post(
                f"{COMPACTOR_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=600),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"compactor returned HTTP {resp.status}: {body[:300]}")
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if not delta:
                        continue
                    full_text += delta
                    got_any = True
                    now = time.monotonic()
                    if now - last_edit_at >= EDIT_INTERVAL_S:
                        last_edit_at = now
                        await send_edit(room_id, placeholder_event_id, full_text)
            if not got_any:
                full_text = "(no reply)"
            # The final edit always happens, regardless of the 1/s cadence above.
            await send_edit(room_id, placeholder_event_id, full_text)
        except Exception as e:
            log.exception("stream_reply failed")
            await send_failure_notice(room_id, str(e))

    async def on_message(evt: MessageEvent) -> None:
        if evt.sender == BOT_MXID:
            return
        if evt.timestamp < start_ts_ms:
            log.info(f"ignoring pre-start event {evt.event_id} (ts={evt.timestamp})")
            return
        if HER_ROOM_ID and evt.room_id != HER_ROOM_ID:
            log.info(f"ignoring message in unmapped room {evt.room_id}")
            return
        body = evt.content.body if hasattr(evt.content, "body") else ""
        if not body:
            return
        log.info(f"decrypted message from {evt.sender} in {evt.room_id} ({len(body)} chars)")

        placeholder = await client.send_text(evt.room_id, "...")
        placeholder_event_id = placeholder.event_id if hasattr(placeholder, "event_id") else placeholder

        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(keep_typing(evt.room_id, stop_typing))
        try:
            await stream_reply(evt.room_id, HER_CONV_ID, body, placeholder_event_id)
        finally:
            stop_typing.set()
            await typing_task

    client.add_event_handler(EventType.ROOM_MEMBER, on_member)
    client.add_event_handler(EventType.ROOM_MESSAGE, on_message)

    log.info(f"v4lab bot ready. her_room={HER_ROOM_ID or '(none configured)'} her_conv_id={HER_CONV_ID}")
    await client.start(None)


if __name__ == "__main__":
    asyncio.run(main())
