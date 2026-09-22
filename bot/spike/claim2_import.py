"""Claim 2: faithful import of ~4,000 synthetic messages, alternating two
senders, each encrypted as its own sender, each carrying an original
timestamp via the appservice `ts` param.

Also does the alice-side half of claim 3's setup (creates her key backup
and uploads every session used during the import to it), because that has
to happen *during* the import (we hold the plaintext session keys in memory
right here) -- there's no way to do it after the fact without the importer's
material, which is exactly the point being tested.

Ground truth for every message (sender, body, original ts) is written to
out/claim2_ground_truth.json so claim3_verify.py can check decrypted output
against it independently.
"""
import asyncio
import json
import random
import sys
import time

import aiohttp

sys.path.insert(0, "/work")
from lib import backup, config
from lib.olm_sender import ImportSender

ROOM_ALIAS_NAME = "zlaspike-import-room"
NUM_MESSAGES = 4000
IMPORT_ALICE_DEVICE = "ZLASPIKE_IMPORT_ALICE_DEV"
IMPORT_BOT_DEVICE = "ZLASPIKE_IMPORT_BOT_DEV"

FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog while synthetic spike data "
    "flows through the phase one importer proving faithful history migration "
    "works end to end for encrypted rooms without touching any real"
).split()


def filler_body(i: int, sender_label: str) -> str:
    rnd = random.Random(i)
    words = [rnd.choice(FILLER_WORDS) for _ in range(rnd.randint(6, 16))]
    return f"[synthetic #{i} from {sender_label}] " + " ".join(words)


async def as_request(session, method, path, mxid, ts=None, json_body=None, params=None):
    p = dict(params or {})
    p["user_id"] = mxid
    if ts is not None:
        p["ts"] = ts
    headers = {"Authorization": f"Bearer {config.AS_TOKEN}"}
    async with session.request(
        method, f"{config.HS_URL}{path}", params=p, json=json_body, headers=headers
    ) as resp:
        text = await resp.text()
        if resp.status >= 300:
            return resp.status, text
        try:
            return resp.status, json.loads(text)
        except Exception:
            return resp.status, text


async def login_as(session, mxid_localpart, device_id):
    headers = {"Authorization": f"Bearer {config.AS_TOKEN}"}
    body = {
        "type": "m.login.application_service",
        "identifier": {"type": "m.id.user", "user": mxid_localpart},
        "device_id": device_id,
    }
    async with session.post(
        f"{config.HS_URL}/_matrix/client/v3/login", json=body, headers=headers
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"AS login as {mxid_localpart} failed: {data}")
        return data["access_token"]


async def upload_device_keys(session, access_token, device_keys):
    headers = {"Authorization": f"Bearer {access_token}"}
    async with session.post(
        f"{config.HS_URL}/_matrix/client/v3/keys/upload",
        json={"device_keys": device_keys},
        headers=headers,
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"keys/upload failed: {data}")
        return data


async def main():
    result = {"claim": 2, "steps": [], "throttled_count": 0, "retries": 0}
    async with aiohttp.ClientSession() as session:
        alice_token = await login_as(session, "alice_spike", IMPORT_ALICE_DEVICE)
        bot_token = await login_as(session, "zla_bot", IMPORT_BOT_DEVICE)
        result["steps"].append("created importer devices for both senders via AS login")

        alice_sender = ImportSender(config.ALICE_MXID, IMPORT_ALICE_DEVICE)
        bot_sender = ImportSender(config.BOT_MXID, IMPORT_BOT_DEVICE)

        await upload_device_keys(session, alice_token, alice_sender.device_keys_payload())
        await upload_device_keys(session, bot_token, bot_sender.device_keys_payload())
        result["steps"].append("uploaded identity keys for both importer devices")

        # Create the room and get both participants in, all via AS masquerade
        # -- no need for the live bot process to be involved at all.
        status, room_resp = await as_request(
            session,
            "POST",
            "/_matrix/client/v3/createRoom",
            config.ALICE_MXID,
            json_body={
                "name": "zlaspike import room",
                "invite": [config.BOT_MXID],
                "initial_state": [
                    {
                        "type": "m.room.encryption",
                        "content": {"algorithm": "m.megolm.v1.aes-sha2"},
                    }
                ],
            },
        )
        if status != 200:
            raise RuntimeError(f"createRoom failed: {room_resp}")
        room_id = room_resp["room_id"]
        result["room_id"] = room_id
        result["steps"].append(f"created room {room_id}")

        status, _ = await as_request(
            session, "POST", f"/_matrix/client/v3/join/{room_id}", config.BOT_MXID
        )
        if status != 200:
            raise RuntimeError(f"bot join failed: {_}")
        result["steps"].append("bot joined room (direct AS join, no live process needed)")

        # ---- generate + send ~4000 alternating, backdated, encrypted messages ----
        now_ms = int(time.time() * 1000)
        span_ms = 90 * 24 * 3600 * 1000  # spread over the last 90 days
        start_ms = now_ms - span_ms
        ground_truth = []
        t0 = time.time()
        rate_limited_hits = 0
        cur_ts = start_ms
        for i in range(NUM_MESSAGES):
            is_alice = i % 2 == 0
            sender = alice_sender if is_alice else bot_sender
            mxid = config.ALICE_MXID if is_alice else config.BOT_MXID
            label = "alice" if is_alice else "bot"
            cur_ts += random.randint(20_000, 240_000)  # 20s - 4min gaps
            cur_ts = min(cur_ts, now_ms - 1000)
            body = filler_body(i, label)
            content = {"msgtype": "m.text", "body": body}
            encrypted_content = sender.encrypt(room_id, "m.room.message", content)
            txn_id = f"zlaspikeimport{i}"

            attempt = 0
            while True:
                status, resp = await as_request(
                    session,
                    "PUT",
                    f"/_matrix/client/v3/rooms/{room_id}/send/m.room.encrypted/{txn_id}",
                    mxid,
                    ts=cur_ts,
                    json_body=encrypted_content,
                )
                if status == 429:
                    rate_limited_hits += 1
                    retry_after = 0.5
                    if isinstance(resp, dict):
                        retry_after = max(retry_after, resp.get("retry_after_ms", 500) / 1000)
                    await asyncio.sleep(retry_after)
                    attempt += 1
                    result["retries"] += 1
                    if attempt > 10:
                        raise RuntimeError(f"gave up after repeated 429s at message {i}")
                    continue
                if status != 200:
                    raise RuntimeError(f"send failed at message {i}: {status} {resp}")
                break

            ground_truth.append(
                {
                    "index": i,
                    "sender": mxid,
                    "body": body,
                    "ts": cur_ts,
                    "event_id": resp["event_id"],
                }
            )
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  sent {i+1}/{NUM_MESSAGES} in {elapsed:.1f}s", flush=True)

        elapsed = time.time() - t0
        result["throttled_count"] = rate_limited_hits
        result["import_duration_seconds"] = elapsed
        result["messages_per_second"] = NUM_MESSAGES / elapsed
        result["steps"].append(
            f"sent {NUM_MESSAGES} messages in {elapsed:.1f}s "
            f"({NUM_MESSAGES/elapsed:.1f} msg/s), {rate_limited_hits} rate-limit hits"
        )

        with open("/work/out/claim2_ground_truth.json", "w") as f:
            json.dump({"room_id": room_id, "messages": ground_truth}, f)

        # ---- create alice's key backup and upload every session used ----
        backup_version, alice_backup_pickle = await backup.create_backup_version(
            session, config.HS_URL, alice_token
        )
        result["backup_version"] = backup_version.version
        result["steps"].append(f"created alice key backup version {backup_version.version}")

        with open("/work/out/alice_backup_pickle.bin", "wb") as f:
            f.write(alice_backup_pickle)

        n_alice_sessions = len(alice_sender.export_all_sessions_for_backup(room_id))
        n_bot_sessions = len(bot_sender.export_all_sessions_for_backup(room_id))
        for entry in alice_sender.export_all_sessions_for_backup(
            room_id
        ) + bot_sender.export_all_sessions_for_backup(room_id):
            await backup.upload_room_key(
                session,
                config.HS_URL,
                alice_token,
                backup_version.version,
                room_id,
                entry["session_id"],
                backup_version.public_key,
                entry["session_data"],
            )
        result["steps"].append(
            f"uploaded {n_alice_sessions} alice-sender sessions + "
            f"{n_bot_sessions} bot-sender sessions to alice's key backup "
            f"({n_alice_sessions + n_bot_sessions} megolm session rotations "
            f"total across {NUM_MESSAGES} messages)"
        )
        result["session_count"] = n_alice_sessions + n_bot_sessions
        result["ok"] = True

    print(json.dumps(result, indent=2))
    with open("/work/out/claim2_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
