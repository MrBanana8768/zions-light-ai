"""Claim 3b step 2: import synthetic history into a fresh encrypted room
whose members are a real Element user (elem_owner11, who set up her own
Secure Backup through Element's UI in claim3b_setup_backup.py) and the bot.

Corrected design: the importer NEVER creates a backup version. It fetches
the CURRENT version's public key (a GET, no secret material) and uploads
session keys against it. This script asserts, from its own HTTP call log,
that it made zero POST calls to /room_keys/version.
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

NUM_MESSAGES = 600
IMPORT_ALICE_DEVICE = "ZLASPIKE_3BFINAL_ELEM_DEV"
IMPORT_BOT_DEVICE = "ZLASPIKE_3BFINAL_BOT_DEV"
ELEM_MXID = "@elem_owner11:localhost"

FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog while synthetic spike data "
    "flows through the phase one importer proving faithful history migration "
    "works end to end for encrypted rooms without touching any real"
).split()

HTTP_CALL_LOG = []


class LoggingSession(aiohttp.ClientSession):
    def _request(self, method, url, **kwargs):
        HTTP_CALL_LOG.append((method, str(url)))
        return super()._request(method, url, **kwargs)


def filler_body(i: int, sender_label: str) -> str:
    rnd = random.Random(i + 99000)
    words = [rnd.choice(FILLER_WORDS) for _ in range(rnd.randint(6, 16))]
    return f"[synthetic 3b #{i} from {sender_label}] " + " ".join(words)


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
    result = {"claim": "3b-import", "steps": [], "throttled_count": 0, "retries": 0}
    async with LoggingSession() as session:
        elem_token = await login_as(session, "elem_owner11", IMPORT_ALICE_DEVICE)
        bot_token = await login_as(session, "zla_bot", IMPORT_BOT_DEVICE)
        result["steps"].append("created importer devices for both senders via AS login")

        elem_sender = ImportSender(ELEM_MXID, IMPORT_ALICE_DEVICE)
        bot_sender = ImportSender(config.BOT_MXID, IMPORT_BOT_DEVICE)

        await upload_device_keys(session, elem_token, elem_sender.device_keys_payload())
        await upload_device_keys(session, bot_token, bot_sender.device_keys_payload())
        result["steps"].append("uploaded identity keys for both importer devices")

        status, room_resp = await as_request(
            session,
            "POST",
            "/_matrix/client/v3/createRoom",
            ELEM_MXID,
            json_body={
                "name": "zlaspike 3b FINAL import room",
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
        result["steps"].append("bot joined room")

        now_ms = int(time.time() * 1000)
        span_ms = 30 * 24 * 3600 * 1000  # spread over the last 30 days
        start_ms = now_ms - span_ms
        ground_truth = []
        t0 = time.time()
        cur_ts = start_ms
        for i in range(NUM_MESSAGES):
            is_elem = i % 2 == 0
            sender = elem_sender if is_elem else bot_sender
            mxid = ELEM_MXID if is_elem else config.BOT_MXID
            label = "elem_owner11" if is_elem else "bot"
            cur_ts += random.randint(20_000, 240_000)
            cur_ts = min(cur_ts, now_ms - 1000)
            body = filler_body(i, label)
            content = {"msgtype": "m.text", "body": body}
            encrypted_content = sender.encrypt(room_id, "m.room.message", content)
            txn_id = f"zlaspike3b{i}"

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
                    result["retries"] += 1
                    result["throttled_count"] += 1
                    await asyncio.sleep(0.5)
                    attempt += 1
                    if attempt > 10:
                        raise RuntimeError(f"gave up after repeated 429s at message {i}")
                    continue
                if status != 200:
                    raise RuntimeError(f"send failed at message {i}: {status} {resp}")
                break

            ground_truth.append(
                {"index": i, "sender": mxid, "body": body, "ts": cur_ts, "event_id": resp["event_id"]}
            )
            if (i + 1) % 200 == 0:
                print(f"  sent {i+1}/{NUM_MESSAGES} in {time.time()-t0:.1f}s", flush=True)

        elapsed = time.time() - t0
        result["import_duration_seconds"] = elapsed
        result["messages_per_second"] = NUM_MESSAGES / elapsed
        result["steps"].append(f"sent {NUM_MESSAGES} messages in {elapsed:.1f}s")

        with open("/work/out/claim3b_ground_truth.json", "w") as f:
            json.dump({"room_id": room_id, "messages": ground_truth}, f)

        # ---- fetch elem_owner11's EXISTING backup version (created by her own
        # Element client) and upload every session used to it. No version
        # creation, ever. ----
        backup_version = await backup.get_backup_version(session, config.HS_URL, elem_token)
        result["backup_version"] = backup_version.version
        result["backup_public_key"] = backup_version.public_key
        result["steps"].append(
            f"fetched EXISTING elem_owner11 backup version {backup_version.version} "
            "(created by Element, not by this importer)"
        )

        elem_sessions = elem_sender.export_all_sessions_for_backup(room_id)
        bot_sessions = bot_sender.export_all_sessions_for_backup(room_id)
        for entry in elem_sessions + bot_sessions:
            await backup.upload_room_key(
                session,
                config.HS_URL,
                elem_token,
                backup_version.version,
                room_id,
                entry["session_id"],
                backup_version.public_key,
                entry["session_data"],
            )
        result["steps"].append(
            f"uploaded {len(elem_sessions)} elem-sender sessions + "
            f"{len(bot_sessions)} bot-sender sessions to elem_owner11's EXISTING backup"
        )
        result["session_count"] = len(elem_sessions) + len(bot_sessions)

        # ---- the assertion the corrected design lives or dies on ----
        version_posts = [
            (m, u) for (m, u) in HTTP_CALL_LOG if m == "POST" and "/room_keys/version" in u
        ]
        result["room_keys_version_post_calls"] = version_posts
        result["zero_backup_version_posts"] = len(version_posts) == 0
        result["total_http_calls"] = len(HTTP_CALL_LOG)
        result["ok"] = len(version_posts) == 0

    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=2))
    for s in result["steps"]:
        print(" -", s)
    with open("/work/out/claim3b_import_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
