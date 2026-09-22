"""Server-side key backup (m.megolm_backup.v1.curve25519-aes-sha2), hand-rolled.

CORRECTED DESIGN (per hostile review of the first pass): the importer must
never create a backup version. In production her Element owns the backup --
it creates the version, signs it with her cross-signing keys, and holds the
recovery key in 4S/secret storage. An importer that creates its own version
would supersede hers with one she has no recovery key for. Uploading a
session only needs the CURRENT version's public key from `auth_data`
(a GET, no secret material), so `create_backup_version` below is kept only
as a record of the wrong first draft -- claim3b_import.py never calls it.
Everything the importer does now is achievable with a version's public key
alone; it never holds, generates, or sees any private backup material.

(Earlier draft also had the wrong algorithm id here --
"m.megolm_backup.v1.curve25519.aes-sha2" (period) instead of the real
"m.megolm_backup.v1.curve25519-aes-sha2" (hyphen), confirmed against what
Element itself POSTs when a real user sets up backup. This never surfaced
in claims 2/3 because Synapse stores the algorithm string opaquely and my
own verifier used the same wrong string consistently on both ends -- exactly
the kind of self-consistent bug claim 3b exists to catch.)

Both mautrix-python 0.20.8 and matrix-nio 0.26.0 were grepped source-wide for
"room_keys"/"backup" and neither has any reference (see fix-v4-spike.md,
claim 3). Neither library can create a backup version, upload a session, or
restore one. Everything in this file exists to answer "if it doesn't, what
does, and at what cost."

The cost turned out to be low: the backup algorithm Matrix specifies
(curve25519 ECDH -> HKDF -> AES-CBC -> HMAC, wrapped as
{ciphertext, mac, ephemeral}) is *exactly* libolm's "PK encryption" primitive
-- the same one used for the very first (pre-Rust) Element/Riot key-backup
implementations. python-olm, which mautrix-python already depends on for
Megolm/Olm, exposes it directly as olm.PkEncryption / olm.PkDecryption. No
extra crypto library, no hand-rolled AES/HMAC.

One real gap: python-olm's PkDecryption never exposes the raw 32-byte
Curve25519 private scalar, only an opaque pickle (passphrase-encrypted). A
production client has to render that scalar as the human-typable "Recovery
Key" (a specific base58 encoding with a parity byte, documented in the
spec's "Key export format" section) so the owner can write it down and type
it into Element on a new device. Neither mautrix-python nor matrix-nio
implement that encoding either. For this spike we transport the *decryption
capability* between processes (standing in for "her devices") as a
passphrase-protected libolm pickle blob written to a file, instead of a
typed recovery key. The cryptographic secret and the encrypt/decrypt
algorithm are identical either way; only the human-facing display encoding
is skipped, which does not change whether the underlying claim
(post-hoc decryptability) holds. Implementing the base58 recovery-key
encoding for real is maybe 40 lines and is called out in the report as
follow-up work, not a blocker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import aiohttp
import olm


@dataclass
class BackupVersion:
    version: str
    public_key: str


async def create_backup_version(session: aiohttp.ClientSession, hs_url: str, token: str) -> tuple[BackupVersion, bytes]:
    """Create a new backup version. Returns the version info and the pickled
    (passphrase-free -- see below) decryption key material.

    Uses a fixed, spike-only pickle passphrase. That passphrase plus the
    pickle bytes together stand in for the "Recovery Key" the owner would
    keep in production.
    """
    decryption = olm.PkDecryption()
    pubkey = decryption.public_key
    body = {
        "algorithm": "m.megolm_backup.v1.curve25519.aes-sha2",
        "auth_data": {"public_key": pubkey},
    }
    async with session.post(
        f"{hs_url}/_matrix/client/v3/room_keys/version",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    pickled = decryption.pickle(RECOVERY_PASSPHRASE)
    return BackupVersion(version=data["version"], public_key=pubkey), pickled


RECOVERY_PASSPHRASE = "zlaspike-recovery-passphrase-not-for-prod"


async def get_backup_version(session: aiohttp.ClientSession, hs_url: str, token: str) -> BackupVersion:
    async with session.get(
        f"{hs_url}/_matrix/client/v3/room_keys/version",
        headers={"Authorization": f"Bearer {token}"},
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return BackupVersion(version=data["version"], public_key=data["auth_data"]["public_key"])


def encrypt_session_data(public_key: str, session_data: dict[str, Any]) -> dict[str, Any]:
    enc = olm.PkEncryption(public_key)
    msg = enc.encrypt(json.dumps(session_data))
    return {"ciphertext": msg.ciphertext, "mac": msg.mac, "ephemeral": msg.ephemeral_key}


async def upload_room_key(
    session: aiohttp.ClientSession,
    hs_url: str,
    token: str,
    version: str,
    room_id: str,
    session_id: str,
    backup_public_key: str,
    session_data: dict[str, Any],
    first_message_index: int = 0,
    forwarded_count: int = 0,
    is_verified: bool = True,
) -> None:
    payload = {
        "first_message_index": first_message_index,
        "forwarded_count": forwarded_count,
        "is_verified": is_verified,
        "session_data": encrypt_session_data(backup_public_key, session_data),
    }
    # session_id is base64 and can contain '/', which would otherwise be read
    # as an extra path segment and 404 with M_UNRECOGNIZED -- quote(safe="")
    # percent-encodes it (and room_id's '!'/':') so the path has exactly two
    # segments like the route expects.
    async with session.put(
        f"{hs_url}/_matrix/client/v3/room_keys/keys/{quote(room_id, safe='')}/{quote(session_id, safe='')}",
        params={"version": version},
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"backup upload failed {resp.status}: {text}")


async def download_all_keys(
    session: aiohttp.ClientSession, hs_url: str, token: str, version: str
) -> dict[str, Any]:
    async with session.get(
        f"{hs_url}/_matrix/client/v3/room_keys/keys",
        params={"version": version},
        headers={"Authorization": f"Bearer {token}"},
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


def load_decryption(pickled: bytes) -> olm.PkDecryption:
    return olm.PkDecryption.from_pickle(pickled, RECOVERY_PASSPHRASE)


def decrypt_session_data(decryption: olm.PkDecryption, entry: dict[str, Any]) -> dict[str, Any]:
    sd = entry["session_data"]
    msg = olm.PkMessage(
        ephemeral_key=sd["ephemeral"],
        mac=sd["mac"],
        ciphertext=sd["ciphertext"],
    )
    plaintext = decryption.decrypt(msg)
    return json.loads(plaintext)
