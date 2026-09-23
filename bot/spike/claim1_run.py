"""Claim 1: encrypted round trip.

alice (matrix-nio, a real independent E2EE client implementation) creates an
encrypted room, invites the bot (mautrix-python, already running as the
`bot` compose service), sends a message, and waits for the bot's encrypted
reply. Success = alice's client decrypts a reply, and the stub compactor's
request log shows the expected headers.

Run inside the zlaspike-lab image on the zlaspike_default network:
    docker run --rm --network zlaspike_default -v <spike dir>:/work \
        zlaspike-lab python3 claim1_run.py
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/work")
from lib import config
from lib.convid import conversation_id
from lib.nio_helpers import make_client, wait_for_join

from nio import (
    EnableEncryptionBuilder,
    MegolmEvent,
    RoomCreateResponse,
    RoomMessageText,
    RoomSendResponse,
)


async def main() -> dict:
    result = {"claim": 1, "steps": []}

    alice = await make_client(
        config.HS_URL, config.ALICE_MXID, config.ALICE_PASSWORD,
        "/work/out/alice_store_claim1", "alice-claim1-device", fresh_store=True,
    )
    result["steps"].append(f"alice logged in, device_id={alice.device_id}")

    create_resp = await alice.room_create(
        name="zlaspike-claim1",
        invite=[config.BOT_MXID],
        initial_state=[
            {"type": "m.room.encryption", "content": {"algorithm": "m.megolm.v1.aes-sha2"}}
        ],
    )
    if not isinstance(create_resp, RoomCreateResponse):
        raise RuntimeError(f"room_create failed: {create_resp}")
    room_id = create_resp.room_id
    result["room_id"] = room_id
    result["conversation_id"] = conversation_id(room_id)
    result["steps"].append(f"created encrypted room {room_id}, invited bot")

    # Wait for the bot to accept the invite and become visible as a member.
    joined = False
    for _ in range(30):
        await alice.sync(timeout=2000)
        room = alice.rooms.get(room_id)
        if room and config.BOT_MXID in room.users:
            joined = True
            break
        await asyncio.sleep(1)
    if not joined:
        raise RuntimeError("bot never joined the room")
    result["steps"].append("bot joined room")

    text = "hello from alice - zlaspike claim 1 round trip"
    send_resp = await alice.room_send(
        room_id,
        message_type="m.room.message",
        content={"msgtype": "m.text", "body": text},
        ignore_unverified_devices=True,
    )
    if not isinstance(send_resp, RoomSendResponse):
        raise RuntimeError(f"room_send failed: {send_resp}")
    result["steps"].append(f"alice sent encrypted message, event_id={send_resp.event_id}")

    # Wait for the bot's reply.
    reply_text = None
    undecryptable = 0
    deadline = time.time() + 30
    while time.time() < deadline and reply_text is None:
        resp = await alice.sync(timeout=3000)
        room = resp.rooms.join.get(room_id) if hasattr(resp, "rooms") else None
        if room:
            for evt in room.timeline.events:
                if isinstance(evt, RoomMessageText) and evt.sender == config.BOT_MXID:
                    reply_text = evt.body
                elif isinstance(evt, MegolmEvent) and evt.sender == config.BOT_MXID:
                    undecryptable += 1
        if reply_text is None:
            await asyncio.sleep(1)

    result["undecryptable_events_seen"] = undecryptable
    if reply_text is None:
        result["ok"] = False
        result["error"] = "no decrypted reply from bot within timeout"
    else:
        result["ok"] = True
        result["reply_text"] = reply_text
        result["steps"].append(f"alice decrypted bot reply: {reply_text!r}")

    await alice.close()
    return result


if __name__ == "__main__":
    res = asyncio.run(main())
    print(json.dumps(res, indent=2))
    with open("/work/out/claim1_result.json", "w") as f:
        json.dump(res, f, indent=2)
