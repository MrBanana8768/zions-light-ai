#!/usr/bin/env python3
"""One-off setup: create her E2EE room, invite @her and @bot, have @her
auto-join (accept the invite) so the room is ready for the importer and for
her opening Element for the first time. Plain REST, stdlib only -- no matrix
SDK needed for this step.

Usage: python3 create_room.py --owner-password ... --her-password ...
Prints the created room_id on the last line of stdout; the caller
(v4lab-up.sh) captures it into .env.lab as LAB_HER_ROOM_ID.
"""
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")
from guard import assert_local_only  # noqa: E402

SERVER_NAME = "localhost"


def call(base_url, method, path, token=None, body=None):
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def login(base_url, user, password, device_id=None):
    body = {
        "type": "m.login.password",
        "identifier": {"type": "m.id.user", "user": user},
        "password": password,
    }
    if device_id:
        body["device_id"] = device_id
    status, data = call(base_url, "POST", "/_matrix/client/v3/login", body=body)
    if status != 200:
        raise RuntimeError(f"login as {user!r} failed: HTTP {status} {data}")
    return data["access_token"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:18008")
    ap.add_argument("--owner-user", default=f"@owner:{SERVER_NAME}")
    ap.add_argument("--her-user", default=f"@her:{SERVER_NAME}")
    ap.add_argument("--bot-user", default=f"@bot:{SERVER_NAME}")
    ap.add_argument("--owner-password", required=True)
    ap.add_argument("--her-password", required=True)
    ap.add_argument("--existing-room-id", default="")
    args = ap.parse_args()

    assert_local_only({"SYNAPSE_BASE_URL": args.base_url}, component="create_room.py")

    if args.existing_room_id:
        print(f"reusing existing room {args.existing_room_id}", file=sys.stderr)
        print(args.existing_room_id)
        return 0

    owner_token = login(args.base_url, args.owner_user, args.owner_password)

    status, data = call(
        args.base_url,
        "POST",
        "/_matrix/client/v3/createRoom",
        token=owner_token,
        body={
            "name": "Her V4 Lab Room",
            "preset": "private_chat",
            "invite": [args.her_user, args.bot_user],
            "initial_state": [
                {
                    "type": "m.room.encryption",
                    "state_key": "",
                    "content": {"algorithm": "m.megolm.v1.aes-sha2"},
                }
            ],
        },
    )
    if status != 200:
        raise RuntimeError(f"createRoom failed: HTTP {status} {data}")
    room_id = data["room_id"]
    print(f"created room {room_id}, invited {args.her_user} and {args.bot_user}", file=sys.stderr)

    her_token = login(args.base_url, args.her_user, args.her_password)
    status, data = call(
        args.base_url, "POST", f"/_matrix/client/v3/join/{urllib.parse.quote(room_id)}",
        token=her_token, body={},
    )
    if status != 200:
        raise RuntimeError(f"her join failed: HTTP {status} {data}")
    print(f"{args.her_user} joined {room_id}", file=sys.stderr)

    # Bot joins on its own (on_member invite handler) once its process is
    # running and syncs -- no need to force it here.
    print(room_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
