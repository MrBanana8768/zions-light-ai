"""Small helpers around matrix-nio for the scripted "human" side of the
spike (alice). Kept separate from the bot, which uses mautrix-python, so
claim 1's round trip exercises two independently-implemented E2EE stacks
talking to each other -- not the same code decrypting its own ciphertext.
"""
from __future__ import annotations

import asyncio
import os
import shutil

from nio import (
    AsyncClient,
    AsyncClientConfig,
    InviteMemberEvent,
    JoinResponse,
    LoginResponse,
    MegolmEvent,
    RoomMessageText,
    RoomSendResponse,
)


async def make_client(homeserver: str, user: str, password: str, store_dir: str, device_name: str,
                       fresh_store: bool = False) -> AsyncClient:
    if fresh_store and os.path.isdir(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir, exist_ok=True)
    config = AsyncClientConfig(store_sync_tokens=True, encryption_enabled=True)
    client = AsyncClient(homeserver, user, store_path=store_dir, config=config)
    resp = await client.login(password, device_name=device_name)
    if not isinstance(resp, LoginResponse):
        raise RuntimeError(f"login failed: {resp}")
    if client.should_upload_keys:
        await client.keys_upload()
    return client


async def accept_all_invites(client: AsyncClient) -> list[str]:
    joined = []
    for room_id in list(client.invited_rooms.keys()):
        resp = await client.join(room_id)
        if isinstance(resp, JoinResponse):
            joined.append(room_id)
    return joined


async def wait_for_join(client: AsyncClient, room_id: str, timeout: float = 20.0) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await client.sync(timeout=2000)
        await accept_all_invites(client)
        if room_id in client.rooms:
            return True
        await asyncio.sleep(0.5)
    return False
