import hashlib


def conversation_id(room_id: str) -> str:
    """X-Conversation-Id: mx-<first 32 hex of sha256(room id)>, per V4_MATRIX_CLIENT.md."""
    digest = hashlib.sha256(room_id.encode("utf-8")).hexdigest()
    return f"mx-{digest[:32]}"
