"""Direct Megolm session management for the one-shot importer.

Why not mautrix's OlmMachine (which the bot in bot.py uses)? OlmMachine's
`share_group_session` shares each Megolm session by encrypting it with Olm
to *currently online, already-known* devices via to-device messages. That's
the right model for a live chat participant. It is the wrong model for a
bulk historical importer: the recipient (her) is not online during the
import, and even if she were, to-device sharing only reaches her *current*
device -- exactly the thing claim 3 says must not be required. So the
importer creates and exports Megolm sessions directly with the same
primitives OlmMachine itself is built on (mautrix.crypto.OlmAccount,
OutboundGroupSession, InboundGroupSession -- all straight from
mautrix-python/python-olm) and hands the session keys to the recipient via
server-side key backup instead of to-device sharing.
"""
from __future__ import annotations

import json
from typing import Any

from mautrix.crypto import OlmAccount
from mautrix.crypto.sessions import InboundGroupSession, OutboundGroupSession
from mautrix.types import EncryptedMegolmEventContent, EventType, SessionID


class ImportSender:
    """One Megolm-encrypting identity (one sender, one importer-created device)."""

    def __init__(self, mxid: str, device_id: str):
        self.mxid = mxid
        self.device_id = device_id
        self.account = OlmAccount()
        self._sessions: dict[str, OutboundGroupSession] = {}
        # Every InboundGroupSession we generate along the way -- these are what
        # get exported into key backup, keyed by (room_id, session_id).
        self.inbound_sessions: dict[tuple[str, str], InboundGroupSession] = {}

    def device_keys_payload(self) -> dict[str, Any]:
        return self.account.get_device_keys(self.mxid, self.device_id)

    def _new_session(self, room_id: str) -> OutboundGroupSession:
        out = OutboundGroupSession(room_id)
        out.shared = True  # we're bypassing to-device sharing entirely; see module docstring
        inbound = InboundGroupSession(
            session_key=out.session_key,
            signing_key=self.account.signing_key,
            sender_key=self.account.identity_key,
            room_id=room_id,
        )
        self.inbound_sessions[(room_id, out.id)] = inbound
        self._sessions[room_id] = out
        return out

    def encrypt(self, room_id: str, event_type: str, content: dict[str, Any]) -> dict[str, Any]:
        session = self._sessions.get(room_id)
        if session is None or session.expired:
            session = self._new_session(room_id)
        plaintext = json.dumps({"room_id": room_id, "type": event_type, "content": content})
        ciphertext = session.encrypt(plaintext)
        return EncryptedMegolmEventContent(
            sender_key=self.account.identity_key,
            device_id=self.device_id,
            ciphertext=ciphertext,
            session_id=SessionID(session.id),
        ).serialize()

    def export_all_sessions_for_backup(self, room_id: str) -> list[dict[str, Any]]:
        """Session export payloads (session_data, pre-encryption) for every
        Megolm session used in the given room, ready for lib.backup.upload_room_key.
        """
        out = []
        for (r_id, session_id), inbound in self.inbound_sessions.items():
            if r_id != room_id:
                continue
            out.append(
                {
                    "session_id": session_id,
                    "session_data": {
                        "algorithm": "m.megolm.v1.aes-sha2",
                        "sender_key": inbound.sender_key,
                        "sender_claimed_keys": {"ed25519": inbound.signing_key},
                        "forwarding_curve25519_key_chain": [],
                        "session_key": inbound.export_session(0),
                    },
                }
            )
        return out
