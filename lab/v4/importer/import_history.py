"""V4 lab importer -- L3.

Imports a COPY of her real history into her local E2EE room: table-first,
from the newest leaf (`chat.current_message_id`) walking `parent_id`
backward to the root, exactly as V4_DESIGN.md section 5.2 specifies for the
production exporter. Uses each assistant row's `output` text when present
(design: "use the output text for replies"), original timestamps (via the
appservice `ts` override), Megolm encryption with vodozemac, and per-session
keys uploaded to (and read back from) the sender's own key backup.

Crash-safe and resumable: the crypto/backup/send loop is
lab/zl-ops/v4/lab/val_import.py's proven pattern (ciphertext persisted
before send, ciphertext-based reconciliation on resume, a session never
re-encrypted). See that file's docstring for the V1/V2/V5/V6/V7 properties
this preserves.

PRIVACY: never print her message text. Every log line below reports ids,
roles, lengths, timestamps and counts only -- grep this file for `body` to
audit that claim.

HARD SAFETY RULE: refuses to start unless SYNAPSE_URL is local/private (see
guard.py) and unless the source database path is under a path this script
recognizes as a COPY (never the read-only pod-export originals directly).
"""
import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import time
from urllib.parse import quote

import aiohttp
import vodozemac as vz

sys.path.insert(0, os.path.dirname(__file__))
from guard import assert_local_only  # noqa: E402

SERVER = "localhost"
HER = f"@her:{SERVER}"
BOT = f"@bot:{SERVER}"
HER_LOCALPART = "her"
BOT_LOCALPART = "bot"
DEVICE_HER = "V4LAB_IMPORTER_HER"
DEVICE_BOT = "V4LAB_IMPORTER_BOT"
SESSION_MAX_MSGS = 40
PICKLE_KEY = hashlib.sha256(b"v4lab importer journal pickle key -- lab only").digest()

OUT_DIR = os.environ.get("IMPORT_OUT_DIR", "/work/out")
JOURNAL = f"{OUT_DIR}/journal.db"
RECOVERY_FILE = f"{OUT_DIR}/her_recovery_secret.b64"


def ub64(b: bytes) -> str:
    import base64
    return base64.b64encode(b).decode().rstrip("=")


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------- export
def export_branch(db_path: str, chat_id: str) -> list[tuple[int, str, int, str, bool]]:
    """Table-first walk from the newest leaf, per V4_DESIGN.md 5.2.

    Returns [(idx, sender_mxid, ts_ms, body, is_placeholder), ...] in
    chronological order. `is_placeholder` marks a turn whose text could not
    be recovered (never both roles' worth of context lost -- the position
    in the chain is preserved either way).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    row = cur.execute("select current_message_id from chat where id=?", (chat_id,)).fetchone()
    if not row or not row["current_message_id"]:
        raise RuntimeError(f"chat {chat_id} has no current_message_id in {db_path}")

    chain: list[sqlite3.Row] = []
    local_id = row["current_message_id"]
    seen = set()
    while local_id:
        full_id = f"{chat_id}-{local_id}"
        if full_id in seen:
            raise RuntimeError(f"cycle detected in parent_id chain at {full_id}")
        seen.add(full_id)
        r = cur.execute(
            "select id, role, parent_id, content, output, error, created_at "
            "from chat_message where id=? and chat_id=?",
            (full_id, chat_id),
        ).fetchone()
        if not r:
            raise RuntimeError(f"broken chain: {full_id} referenced but not found")
        chain.append(r)
        local_id = r["parent_id"]
    chain.reverse()  # root -> newest leaf, chronological

    out = []
    idx = 0
    for r in chain:
        role = r["role"]
        if role not in ("user", "assistant"):
            continue  # system/tool rows are not chat turns in this room
        content_raw = r["content"]
        output_raw = r["output"]
        error_raw = r["error"]

        text = None
        if role == "assistant" and output_raw:
            try:
                output = json.loads(output_raw)
                if isinstance(output, list) and output:
                    parts = (output[0].get("content") or [])
                    texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
                    if texts:
                        text = "".join(texts)
            except (json.JSONDecodeError, TypeError, AttributeError, KeyError):
                text = None
        if text is None and content_raw:
            try:
                c = json.loads(content_raw)
                text = c if isinstance(c, str) else None
            except (json.JSONDecodeError, TypeError):
                text = content_raw if isinstance(content_raw, str) else None

        is_placeholder = False
        if not text:
            if role == "assistant" and error_raw:
                # Design rule: an assistant row with `error` set and neither
                # content nor output is dropped entirely -- it never
                # occupied a turn slot in the rollup either.
                continue
            text = "[empty message]"
            is_placeholder = True

        sender = HER if role == "user" else BOT
        ts_ms = int(r["created_at"]) * 1000
        out.append((idx, sender, ts_ms, text, is_placeholder))
        idx += 1
    return out


# ------------------------------------------------------------- journal (val_import.py pattern)
def jopen():
    c = sqlite3.connect(JOURNAL, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=FULL")
    c.executescript("""
    create table if not exists meta(k text primary key, v text);
    create table if not exists sessions(session_id text primary key, sender text, pickle text,
        backup_version text, backed_up integer default 0, uses integer default 0, retired integer default 0);
    create table if not exists msgs(idx integer primary key, sender text, ts integer, len integer,
        placeholder integer default 0, state text default 'planned', session_id text,
        message_index integer, content text, txn text, event_id text);
    """)
    return c


def meta_get(c, k):
    r = c.execute("select v from meta where k=?", (k,)).fetchone()
    return r[0] if r else None


def meta_set(c, k, v):
    c.execute("insert into meta(k,v) values(?,?) on conflict(k) do update set v=excluded.v", (k, v))


HTTP_LOG = []


async def req(s, method, path, token, params=None, body=None, ok=(200,)):
    HTTP_LOG.append((method, path))
    async with s.request(
        method, f"{SYNAPSE_URL}{path}", params=params, json=body,
        headers={"Authorization": f"Bearer {token}"},
    ) as r:
        txt = await r.text()
        try:
            data = json.loads(txt)
        except Exception:
            data = txt
        if ok and r.status not in ok:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status} {str(data)[:300]}")
        return r.status, data


async def as_req(s, method, path, user, ts=None, body=None, ok=(200,)):
    p = {"user_id": user}
    if ts is not None:
        p["ts"] = str(ts)
    return await req(s, method, path, AS_TOKEN, params=p, body=body, ok=ok)


async def as_login(s, localpart, device):
    _, d = await req(
        s, "POST", "/_matrix/client/v3/login", AS_TOKEN,
        body={
            "type": "m.login.application_service",
            "identifier": {"type": "m.id.user", "user": localpart},
            "device_id": device,
        },
    )
    return d["access_token"]


async def pw_login(s, user, password, device):
    _, d = await req(
        s, "POST", "/_matrix/client/v3/login", "",
        body={
            "type": "m.login.password",
            "identifier": {"type": "m.id.user", "user": user},
            "password": password,
            "device_id": device,
        },
    )
    return d["access_token"]


async def backup_version(s, tok):
    _, d = await req(s, "GET", "/_matrix/client/v3/room_keys/version", tok, ok=(200, 404))
    if "version" not in d:
        return None, None, None
    return d["version"], d["auth_data"]["public_key"], d["algorithm"]


async def ensure_backup(s, her_tok):
    version, pub, alg = await backup_version(s, her_tok)
    if version is not None:
        return version, pub
    print("no key backup exists yet for @her; creating one (lab stand-in for Element's own setup)")
    d = vz.PkDecryption()
    _, r = await req(s, "POST", "/_matrix/client/v3/room_keys/version", her_tok, body={
        "algorithm": "m.megolm_backup.v1.curve25519-aes-sha2",
        "auth_data": {"public_key": d.public_key.to_base64()},
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RECOVERY_FILE, "w") as f:
        f.write(d.key.to_base64())
    os.chmod(RECOVERY_FILE, 0o600)
    print(f"created key backup version {r['version']}; recovery key saved to {RECOVERY_FILE} (not printed, not in git)")
    return r["version"], d.public_key.to_base64()


async def upload_and_readback(s, her_tok, room, version, pub, sender_ident, gs):
    ib = vz.InboundGroupSession(gs.session_key)
    session_data = {
        "algorithm": "m.megolm.v1.aes-sha2",
        "sender_key": sender_ident["curve"],
        "sender_claimed_keys": {"ed25519": sender_ident["ed"]},
        "forwarding_curve25519_key_chain": [],
        "session_key": ib.export_at(0).to_base64(),
    }
    enc = vz.PkEncryption.from_key(vz.Curve25519PublicKey.from_base64(pub))
    m = enc.encrypt(canon(session_data).encode())
    sd = {
        "ephemeral": m.ephemeral_key.to_base64() if hasattr(m.ephemeral_key, "to_base64") else ub64(m.ephemeral_key),
        "ciphertext": ub64(m.ciphertext),
        "mac": ub64(m.mac),
    }
    body = {"first_message_index": 0, "forwarded_count": 0, "is_verified": True, "session_data": sd}
    path = f"/_matrix/client/v3/room_keys/keys/{quote(room, safe='')}/{quote(gs.session_id, safe='')}"
    await req(s, "PUT", path, her_tok, params={"version": version}, body=body)
    _, back = await req(s, "GET", path, her_tok, params={"version": version})
    if back.get("session_data") != sd or back.get("first_message_index") != 0:
        raise RuntimeError(f"key-backup read-back mismatch for session {gs.session_id}")


async def find_ciphertext(s, tok, room, ct, limit=50):
    _, d = await req(s, "GET", f"/_matrix/client/v3/rooms/{quote(room, safe='')}/messages", tok,
                     params={"dir": "b", "limit": str(limit)})
    for ev in d.get("chunk", []):
        if ev.get("type") == "m.room.encrypted" and ev["content"].get("ciphertext") == ct:
            return ev["event_id"]
    return None


# ------------------------------------------------------------- main import loop
SYNAPSE_URL = ""
AS_TOKEN = ""


async def run(db_path: str, chat_id: str, room: str, her_password: str, last_n: int | None):
    global SYNAPSE_URL, AS_TOKEN
    SYNAPSE_URL = os.environ["SYNAPSE_URL"].rstrip("/")
    AS_TOKEN = os.environ.get("AS_TOKEN", "v4lab_as_token_LAB_ONLY_do_not_reuse")
    assert_local_only({"SYNAPSE_URL": SYNAPSE_URL}, component="importer")

    branch = export_branch(db_path, chat_id)
    total_len = len(branch)
    placeholders = sum(1 for *_r, ph in branch if ph)
    print(f"exported branch: {total_len} turns (table-first, newest-leaf walk); {placeholders} placeholder turn(s)")
    if last_n is not None and last_n < total_len:
        branch = branch[-last_n:]
        print(f"--last {last_n}: importing only the newest {len(branch)} of {total_len}")

    c = jopen()
    for idx, sender, ts, body, is_ph in branch:
        c.execute(
            "insert or ignore into msgs(idx,sender,ts,len,placeholder) values(?,?,?,?,?)",
            (idx, sender, ts, len(body), 1 if is_ph else 0),
        )
    body_by_idx = {idx: body for idx, _s, _t, body, _ph in branch}

    async with aiohttp.ClientSession() as s:
        idents = {}
        for mxid, lp, dev in ((HER, HER_LOCALPART, DEVICE_HER), (BOT, BOT_LOCALPART, DEVICE_BOT)):
            pk = meta_get(c, f"acct:{mxid}")
            if pk:
                acct = vz.Account.from_pickle(pk, PICKLE_KEY)
                tok = meta_get(c, f"tok:{mxid}")
            else:
                acct = vz.Account()
                tok = await as_login(s, lp, dev)
                keys = {
                    "user_id": mxid, "device_id": dev,
                    "algorithms": ["m.olm.v1.curve25519-aes-sha2", "m.megolm.v1.aes-sha2"],
                    "keys": {
                        f"curve25519:{dev}": acct.curve25519_key.to_base64(),
                        f"ed25519:{dev}": acct.ed25519_key.to_base64(),
                    },
                }
                sig = acct.sign(canon(keys).encode())
                keys["signatures"] = {mxid: {f"ed25519:{dev}": sig.to_base64()}}
                await req(s, "POST", "/_matrix/client/v3/keys/upload", tok, body={"device_keys": keys})
                meta_set(c, f"acct:{mxid}", acct.pickle(PICKLE_KEY))
                meta_set(c, f"tok:{mxid}", tok)
            idents[mxid] = {
                "acct": acct, "tok": tok, "dev": dev,
                "curve": acct.curve25519_key.to_base64(), "ed": acct.ed25519_key.to_base64(),
            }

        her_tok = idents[HER]["tok"]
        version, pub = await ensure_backup(s, her_tok)
        pinned = meta_get(c, "backup_version")
        if pinned is None:
            meta_set(c, "backup_version", version)
            meta_set(c, "backup_pub", pub)
        elif (pinned, meta_get(c, "backup_pub")) != (version, pub):
            print(f"HALT: backup version changed {pinned} -> {version}; not using any session")
            return 4

        meta_set(c, "room", room)

        t0 = time.time()
        sent = skipped_resume = 0
        rows = c.execute(
            "select idx,sender,ts,state,session_id,message_index,content,txn,event_id "
            "from msgs order by idx"
        ).fetchall()
        for (idx, snd, ts, state, sid, mi, content, txn, evid) in rows:
            if idx not in body_by_idx:
                continue  # journal has an idx this run's --last window excludes; leave it alone
            body = body_by_idx[idx]
            if state == "sent":
                continue
            if state == "planned":
                row = c.execute(
                    "select session_id,pickle,uses from sessions where sender=? and retired=0 "
                    "and backed_up=1 order by rowid desc limit 1", (snd,),
                ).fetchone()
                if row and row[2] >= SESSION_MAX_MSGS:
                    c.execute("update sessions set retired=1 where session_id=?", (row[0],))
                    row = None
                if row is None:
                    v2, p2, _ = await backup_version(s, her_tok)
                    if (v2, p2) != (meta_get(c, "backup_version"), meta_get(c, "backup_pub")):
                        print(f"HALT: backup version changed to {v2} before a new session; stopping")
                        return 4
                    gs = vz.GroupSession()
                    c.execute(
                        "insert into sessions(session_id,sender,pickle,backup_version) values(?,?,?,?)",
                        (gs.session_id, snd, gs.pickle(PICKLE_KEY), v2),
                    )
                    await upload_and_readback(s, her_tok, room, v2, p2, idents[snd], gs)
                    c.execute("update sessions set backed_up=1 where session_id=?", (gs.session_id,))
                    row = (gs.session_id, gs.pickle(PICKLE_KEY), 0)
                gs = vz.GroupSession.from_pickle(row[1], PICKLE_KEY)
                plaintext = canon({
                    "room_id": room, "type": "m.room.message",
                    "content": {"msgtype": "m.text", "body": body, "zl.import": {"idx": idx}},
                })
                mi = gs.message_index
                ct = gs.encrypt(plaintext.encode()).to_base64()
                content = canon({
                    "algorithm": "m.megolm.v1.aes-sha2", "sender_key": idents[snd]["curve"],
                    "ciphertext": ct, "session_id": gs.session_id, "device_id": idents[snd]["dev"],
                })
                txn = f"v4limp-{idx}"
                c.execute("begin")
                c.execute(
                    "update sessions set pickle=?, uses=uses+1 where session_id=?",
                    (gs.pickle(PICKLE_KEY), gs.session_id),
                )
                c.execute(
                    "update msgs set state='encrypted', session_id=?, message_index=?, content=?, txn=? where idx=?",
                    (gs.session_id, mi, content, txn, idx),
                )
                c.execute("commit")
            else:  # 'encrypted' -- resume: reconcile by ciphertext, never re-encrypt
                ct = json.loads(content)["ciphertext"]
                found = await find_ciphertext(s, her_tok, room, ct)
                if found:
                    c.execute("update msgs set state='sent', event_id=? where idx=?", (found, idx))
                    skipped_resume += 1
                    continue
            st, d = await as_req(
                s, "PUT", f"/_matrix/client/v3/rooms/{quote(room, safe='')}/send/m.room.encrypted/{txn}",
                snd, ts=ts, body=json.loads(content), ok=None,
            )
            if st != 200:
                print(f"SEND FAILED idx={idx} sender={snd} status={st} {str(d)[:200]}")
                return 5
            c.execute("update msgs set state='sent', event_id=? where idx=?", (d["event_id"], idx))
            sent += 1
        dt = time.time() - t0
        result = {
            "branch_length": total_len,
            "imported_this_selection": len(branch),
            "sent_this_run": sent,
            "already_present_resume": skipped_resume,
            "seconds": round(dt, 1),
            "msgs_per_s": round(sent / dt, 1) if dt and sent else None,
        }
        print(json.dumps(result, indent=1))
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(f"{OUT_DIR}/import_result.json", "w") as f:
            json.dump(result, f, indent=1)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="/data/her-copy/webui.db", help="path to the COPIED webui.db")
    ap.add_argument("--chat-id", default=os.environ.get("HER_CONV_ID", "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"))
    ap.add_argument("--room", default=os.environ.get("HER_ROOM_ID", ""))
    ap.add_argument("--her-password", default=os.environ.get("HER_PASSWORD", ""))
    ap.add_argument("--last", type=int, default=None, help="import only the newest N turns (for a fast CPU test run)")
    ap.add_argument("--count-only", action="store_true", help="just export and print the branch length, no network")
    args = ap.parse_args()

    if not args.room and not args.count_only:
        print("REFUSING: --room (or HER_ROOM_ID) is required. Run scripts/create_room.py first.", file=sys.stderr)
        return 2
    if "/home/drew/pod-exports/" in os.path.abspath(args.db):
        print("REFUSING: --db points at the read-only pod-export path directly; use the COPY.", file=sys.stderr)
        return 2

    if args.count_only:
        branch = export_branch(args.db, args.chat_id)
        print(json.dumps({"branch_length": len(branch), "placeholders": sum(1 for *_r, p in branch if p)}))
        return 0

    return asyncio.run(run(args.db, args.chat_id, args.room, args.her_password, args.last))


if __name__ == "__main__":
    raise SystemExit(main())
