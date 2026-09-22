"""Claim 3 -- the central go/no-go.

1. Delete every one of alice's devices (not just the importer's -- see
   fix-v4-spike.md for why: this rules out the new device getting lucky via
   leftover key-gossip from a still-alive old device instead of genuinely
   via key backup).
2. Log alice in on a brand-new device (fresh password login, no prior
   store, no prior olm state of any kind).
3. Restore her key backup using the recovery material from claim2 (the
   pickled PkDecryption -- see lib/backup.py for why a pickle stands in for
   the human-typable recovery key here).
4. Paginate the room's full history from scratch and decrypt every event
   using only sessions recovered from backup.
5. Compare against out/claim2_ground_truth.json: every message present,
   right order, right sender, right original timestamp, right body.
"""
import asyncio
import json
import sys

import aiohttp
from mautrix.crypto.sessions import InboundGroupSession

sys.path.insert(0, "/work")
from lib import backup, config


async def admin_login(session) -> str:
    async with session.post(
        f"{config.HS_URL}/_matrix/client/v3/login",
        json={
            "type": "m.login.password",
            "identifier": {"type": "m.id.user", "user": "spike_admin"},
            "password": config.ADMIN_PASSWORD,
        },
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"admin login failed: {data}")
        return data["access_token"]


async def list_devices(session, admin_token, user_id):
    async with session.get(
        f"{config.HS_URL}/_synapse/admin/v2/users/{user_id}/devices",
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"list devices failed: {data}")
        return [d["device_id"] for d in data["devices"]]


async def delete_devices(session, admin_token, user_id, device_ids):
    if not device_ids:
        return
    async with session.post(
        f"{config.HS_URL}/_synapse/admin/v2/users/{user_id}/delete_devices",
        json={"devices": device_ids},
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as resp:
        text = await resp.text()
        if resp.status not in (200, 201):
            raise RuntimeError(f"delete_devices failed: {resp.status} {text}")


async def password_login(session, user_localpart, password, device_name):
    async with session.post(
        f"{config.HS_URL}/_matrix/client/v3/login",
        json={
            "type": "m.login.password",
            "identifier": {"type": "m.id.user", "user": user_localpart},
            "password": password,
            "initial_device_display_name": device_name,
        },
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"login failed: {data}")
        return data


async def get_all_room_events(session, token, room_id):
    events = []
    params = {"dir": "b", "limit": "1000"}
    url = f"{config.HS_URL}/_matrix/client/v3/rooms/{room_id}/messages"
    from_token = None
    while True:
        p = dict(params)
        if from_token:
            p["from"] = from_token
        async with session.get(
            url, params=p, headers={"Authorization": f"Bearer {token}"}
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"get_messages failed: {data}")
        chunk = data.get("chunk", [])
        events.extend(chunk)
        from_token = data.get("end")
        if not chunk or "end" not in data:
            break
        if len(chunk) < 1000:
            break
    return events


async def main():
    result = {"claim": 3, "steps": []}
    with open("/work/out/claim2_ground_truth.json") as f:
        gt = json.load(f)
    room_id = gt["room_id"]
    ground_truth = gt["messages"]

    with open("/work/out/claim2_result.json") as f:
        claim2 = json.load(f)
    backup_version = claim2["backup_version"]

    with open("/work/out/alice_backup_pickle.bin", "rb") as f:
        pickled = f.read()

    async with aiohttp.ClientSession() as session:
        admin_token = await admin_login(session)
        result["steps"].append("admin logged in")

        before_devices = await list_devices(session, admin_token, config.ALICE_MXID)
        result["alice_devices_before_deletion"] = before_devices
        await delete_devices(session, admin_token, config.ALICE_MXID, before_devices)
        after_devices = await list_devices(session, admin_token, config.ALICE_MXID)
        result["alice_devices_after_deletion"] = after_devices
        result["steps"].append(
            f"deleted ALL of alice's devices ({before_devices}) -- "
            f"remaining: {after_devices}"
        )
        if after_devices:
            raise RuntimeError("device deletion did not remove all devices")

        # Brand new device: fresh password login, nothing carried over.
        login_data = await password_login(
            session, "alice_spike", config.ALICE_PASSWORD, "zlaspike-brand-new-device"
        )
        new_device_id = login_data["device_id"]
        new_token = login_data["access_token"]
        result["new_device_id"] = new_device_id
        result["steps"].append(f"logged in on brand-new device {new_device_id}")

        # Restore key backup.
        server_version = await backup.get_backup_version(session, config.HS_URL, new_token)
        assert server_version.version == backup_version
        decryption = backup.load_decryption(pickled)
        assert decryption.public_key == server_version.public_key
        result["steps"].append(
            f"fetched backup version {server_version.version}, "
            "recovery material matches the published public key"
        )

        all_keys = await backup.download_all_keys(
            session, config.HS_URL, new_token, backup_version
        )
        room_keys = all_keys["rooms"][room_id]["sessions"]
        result["backed_up_session_count"] = len(room_keys)

        inbound_sessions = {}
        for session_id, entry in room_keys.items():
            session_data = backup.decrypt_session_data(decryption, entry)
            inbound = InboundGroupSession.import_session(
                session_key=session_data["session_key"],
                signing_key=session_data["sender_claimed_keys"]["ed25519"],
                sender_key=session_data["sender_key"],
                room_id=room_id,
            )
            inbound_sessions[session_id] = inbound
        result["steps"].append(
            f"restored {len(inbound_sessions)} megolm sessions from backup on the new device"
        )

        # Pull every event in the room from scratch, oldest first.
        raw_events = await get_all_room_events(session, new_token, room_id)
        raw_events.reverse()  # messages endpoint with dir=b returns newest-first
        result["steps"].append(f"paginated {len(raw_events)} raw timeline events")

        decrypted = []
        failures = []
        for evt in raw_events:
            if evt.get("type") != "m.room.encrypted":
                continue
            content = evt["content"]
            sid = content["session_id"]
            inbound = inbound_sessions.get(sid)
            if not inbound:
                failures.append({"event_id": evt["event_id"], "reason": "no session in backup"})
                continue
            try:
                plaintext, index = inbound.decrypt(content["ciphertext"])
                payload = json.loads(plaintext)
                decrypted.append(
                    {
                        "event_id": evt["event_id"],
                        "sender": evt["sender"],
                        "ts": evt["origin_server_ts"],
                        "body": payload["content"]["body"],
                        "index": index,
                    }
                )
            except Exception as e:
                failures.append({"event_id": evt["event_id"], "reason": str(e)})

        result["decrypted_count"] = len(decrypted)
        result["failed_count"] = len(failures)
        result["failures_sample"] = failures[:10]

        # Compare against ground truth.
        gt_by_event = {m["event_id"]: m for m in ground_truth}
        mismatches = []
        for i, d in enumerate(decrypted):
            gtm = gt_by_event.get(d["event_id"])
            if gtm is None:
                mismatches.append({"event_id": d["event_id"], "reason": "not in ground truth"})
                continue
            if gtm["body"] != d["body"] or gtm["sender"] != d["sender"] or gtm["ts"] != d["ts"]:
                mismatches.append(
                    {"event_id": d["event_id"], "expected": gtm, "got": d}
                )

        # Order check: decrypted list should be in non-decreasing ts order
        # (we paginated oldest-first) and should match ground truth order.
        order_ok = [d["event_id"] for d in decrypted] == [m["event_id"] for m in ground_truth]

        result["total_expected"] = len(ground_truth)
        result["mismatch_count"] = len(mismatches)
        result["mismatches_sample"] = mismatches[:10]
        result["order_matches_ground_truth"] = order_ok
        result["ok"] = (
            result["decrypted_count"] == result["total_expected"]
            and result["failed_count"] == 0
            and result["mismatch_count"] == 0
            and order_ok
        )

    print(json.dumps(result, indent=2)[:4000])
    with open("/work/out/claim3_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
