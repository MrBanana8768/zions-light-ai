"""Claim 4: the bot survives a restart and a container rebuild -- it should
keep decrypting new messages and the user's client should keep seeing the
same (already-known) device rather than a scary new one.

Usage: python3 claim4_run.py <phase>
  phase = "post_restart" | "post_rebuild"

Both phases do the same thing: a brand-new alice nio device joins the
existing import room (bot is already a member from claim2) and sends a
fresh, *live* (non-backdated) message, then waits for the bot's encrypted
reply. What's compared across phases (from the orchestration script) is the
bot's Olm identity key and device_id, which must stay identical for this to
mean anything.
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/work")
from lib import config
from lib.nio_helpers import make_client

from nio import MegolmEvent, RoomMessageText, RoomSendResponse


async def main(phase: str) -> dict:
    with open("/work/out/claim2_ground_truth.json") as f:
        room_id = json.load(f)["room_id"]

    result = {"claim": 4, "phase": phase, "room_id": room_id, "steps": []}

    alice = await make_client(
        config.HS_URL, config.ALICE_MXID, config.ALICE_PASSWORD,
        f"/work/out/alice_store_claim4_{phase}", f"alice-claim4-{phase}", fresh_store=True,
    )
    result["steps"].append(f"alice logged in on new device {alice.device_id}")

    await alice.sync(timeout=5000, full_state=True)
    if room_id not in alice.rooms:
        raise RuntimeError("alice's new device doesn't see the import room -- unexpected")

    # Look up the bot's current device id(s) as alice sees them (identity-key
    # comparison across phases is done from the bot's own logs instead --
    # more direct evidence than re-deriving it through nio's device store).
    keys_resp = await alice.keys_query()
    devmap = getattr(keys_resp, "device_keys", None) or {}
    result["bot_devices_seen_by_alice"] = list(devmap.get(config.BOT_MXID, {}).keys())

    text = f"zlaspike claim4 live message ({phase}) at {time.time()}"
    send_resp = await alice.room_send(
        room_id,
        message_type="m.room.message",
        content={"msgtype": "m.text", "body": text},
        ignore_unverified_devices=True,
    )
    if not isinstance(send_resp, RoomSendResponse):
        raise RuntimeError(f"room_send failed: {send_resp}")
    result["steps"].append(f"sent live message, event_id={send_resp.event_id}")

    reply_text = None
    deadline = time.time() + 30
    while time.time() < deadline and reply_text is None:
        resp = await alice.sync(timeout=3000)
        room = resp.rooms.join.get(room_id) if hasattr(resp, "rooms") else None
        if room:
            for evt in room.timeline.events:
                if isinstance(evt, RoomMessageText) and evt.sender == config.BOT_MXID:
                    if evt.body != getattr(main, "_last_seen_reply", None):
                        reply_text = evt.body
        if reply_text is None:
            await asyncio.sleep(1)

    result["ok"] = reply_text is not None
    result["reply_text"] = reply_text
    await alice.close()
    return result


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "post_restart"
    res = asyncio.run(main(phase))
    print(json.dumps(res, indent=2))
    with open(f"/work/out/claim4_result_{phase}.json", "w") as f:
        json.dump(res, f, indent=2)
