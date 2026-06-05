"""
compactor.persona — V2.1 Phase 8: personas as first-class memory.

A persona is a long, durable system message describing the model's role,
voice, setting, or character (e.g. "You are a hardboiled noir detective
named Sam Cole..." or a character bible for an RPG protagonist). In
plain V2.0, persona text:
  - gets sent to vLLM as part of the request (good)
  - also flows into the message accumulator that summarizer.maybe_rollup
    compresses into L1/L2/L3 (bad — wastes summary budget on text we
    already inject every turn)
  - counts against context budget twice in effect (once as the actual
    system prompt, once via the summary)

Phase 8 fixes this by recognizing persona text as a separate memory
layer that:
  1. Auto-detects from the first system message when long enough
     (AUTO_DETECT_MIN_CHARS, default 200). First detection per conv
     stores it; later requests with matching hash skip the re-store.
  2. Stores in its own sidecar at <storage>/personas/<conv>.json
  3. Gets injected as a separate system block in the combined system
     message (alongside facts/RAG/summary)
  4. NEVER summarized — summarizer is told to skip system messages
     anyway; this just makes the "skip" explicit
  5. NEVER evicted — separate from the LRU fact budget

User control:
  - GET/POST/DELETE /admin/conversations/<id>/persona
  - GET /admin/personas — library across all convs
  - POST /admin/conversations/<id>/inherit-persona — clone from another conv

Storage shape:
    {
      "conv_id":      "<str>",
      "persona_text": "<str>",
      "set_at":       <unix_ts>,
      "source":       "auto" | "admin" | "inherited",
      "hash":         "<sha256 hex of persona_text>"
    }
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from memory import (
    STORAGE_ROOT,
    atomic_write_json,
    persona_path,
    read_json,
)

logger = logging.getLogger("compactor.persona")

# Minimum length for auto-detection of a system message as a persona.
# Short system prompts ("be concise", "respond in JSON") aren't personas
# — they're per-request style guidance. The threshold is a heuristic.
AUTO_DETECT_MIN_CHARS = int(
    os.environ.get("COMPACTOR_PERSONA_AUTO_DETECT_MIN_CHARS", "200") or 200
)

# Feature gate. Persona detection + injection can be disabled per-pod
# if the operator wants V2.0 behavior.
_ENABLED = (
    os.environ.get("COMPACTOR_PERSONA_ENABLED", "true").lower() != "false"
)


def enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def load_persona(conv_id: str) -> dict | None:
    """Return the persona record for a conv, or None if not set."""
    data = read_json(persona_path(conv_id), default=None)
    if not isinstance(data, dict):
        return None
    text = data.get("persona_text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "conv_id": conv_id,
        "persona_text": text,
        "set_at": int(data.get("set_at", 0)),
        "source": str(data.get("source", "unknown")),
        "hash": str(data.get("hash") or _hash(text)),
    }


def get_persona_text(conv_id: str) -> str | None:
    """Convenience: return just the text, or None. Used by main.py
    injection path."""
    rec = load_persona(conv_id)
    return rec["persona_text"] if rec else None


def save_persona(conv_id: str, text: str, *, source: str = "admin") -> dict:
    """Persist a persona record. Returns the saved record. Idempotent on
    matching hash — if the same text is already stored, the timestamp
    and source are NOT updated (avoids churn from auto-detect path)."""
    text = text.strip()
    if not text:
        raise ValueError("persona text must be non-empty")
    new_hash = _hash(text)
    existing = load_persona(conv_id)
    if existing and existing.get("hash") == new_hash:
        return existing
    record = {
        "conv_id": conv_id,
        "persona_text": text,
        "set_at": int(time.time()),
        "source": source,
        "hash": new_hash,
    }
    atomic_write_json(persona_path(conv_id), record)
    logger.info(
        f"conv={conv_id}: persona saved ({len(text)} chars, source={source})"
    )
    return record


def clear_persona(conv_id: str) -> bool:
    """Delete the persona file. Returns True if a file existed."""
    p = persona_path(conv_id)
    if not p.is_file():
        return False
    try:
        p.unlink()
        logger.info(f"conv={conv_id}: persona cleared")
        return True
    except OSError as e:
        logger.warning(f"conv={conv_id}: persona clear failed: {e}")
        return False


def list_personas() -> list[dict]:
    """Return one record per conversation that has a persona. The library
    view — used by /admin/personas. Returns lightweight records (length
    + metadata only, NOT the full text) so the listing stays compact.
    """
    pdir = STORAGE_ROOT / "personas"
    if not pdir.exists():
        return []
    out: list[dict] = []
    for f in pdir.glob("*.json"):
        if "." in f.stem:
            continue  # skip future sidecars
        conv_id = f.stem
        rec = load_persona(conv_id)
        if not rec:
            continue
        out.append({
            "conv_id": conv_id,
            "length": len(rec["persona_text"]),
            "set_at": rec["set_at"],
            "source": rec["source"],
            "hash": rec["hash"][:12],
        })
    out.sort(key=lambda r: r["set_at"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Auto-detection from request messages
# ---------------------------------------------------------------------------

def detect_persona_in_messages(messages: list[dict]) -> str | None:
    """Look at the first message. If it's a system message whose content
    is long enough to be a persona (≥ AUTO_DETECT_MIN_CHARS), return its
    text. Otherwise return None.

    Multimodal content (list of parts) collapsed to concatenated text.
    """
    if not messages or not _ENABLED:
        return None
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return None
    content = first.get("content")
    if isinstance(content, list):
        # OpenAI multimodal parts — join all text parts.
        text = " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    elif isinstance(content, str):
        text = content
    else:
        return None
    text = text.strip()
    if len(text) < AUTO_DETECT_MIN_CHARS:
        return None
    return text


def _extract_first_system_text(messages: list[dict]) -> str | None:
    """Return the text of messages[0] if it's a system message, else None."""
    if not messages:
        return None
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return None
    content = first.get("content")
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ).strip() or None
    if isinstance(content, str):
        return content.strip() or None
    return None


def text_to_inject(conv_id: str, messages: list[dict]) -> str | None:
    """Return the persona text the chat handler should add as an injected
    block, or None if injection would duplicate what's already in the
    request.

    Logic: if a stored persona exists AND its hash differs from whatever's
    in messages[0], inject the stored one (covers admin-set / inherited
    personas not present in the request payload). If hashes match, vLLM
    is already seeing the persona via messages[0] and we'd be doubling
    up — return None.
    """
    if not _ENABLED:
        return None
    rec = load_persona(conv_id)
    if not rec:
        return None
    first_text = _extract_first_system_text(messages)
    if first_text and _hash(first_text) == rec["hash"]:
        # Already in the request — no need to re-inject.
        return None
    return rec["persona_text"]


def auto_capture_persona(conv_id: str, messages: list[dict]) -> dict | None:
    """If messages contain an unrecognized long system prompt, save it
    as the conv's persona. Returns the saved record, or None if there's
    nothing to capture (no persona-like message, or it matches what's
    already stored).

    Called per-request on the chat handler hot path — must be cheap.
    Idempotent: matching hash → no-op.
    """
    if not _ENABLED:
        return None
    candidate = detect_persona_in_messages(messages)
    if not candidate:
        return None
    existing = load_persona(conv_id)
    new_hash = _hash(candidate)
    if existing and existing.get("hash") == new_hash:
        # Same persona as before — no-op.
        return None
    return save_persona(conv_id, candidate, source="auto")


# ---------------------------------------------------------------------------
# Injection block formatting
# ---------------------------------------------------------------------------

_PERSONA_BLOCK_HEADER = (
    "[Persona / role context for this conversation — treat this as the "
    "primary identity and voice you should maintain]"
)


def format_persona_block(text: str | None) -> str | None:
    """Render persona text as a system-message body for injection. None
    if no persona. The header makes it explicit to the model that this
    block is the durable role context, not a per-turn instruction."""
    if not text:
        return None
    return f"{_PERSONA_BLOCK_HEADER}\n{text}"
