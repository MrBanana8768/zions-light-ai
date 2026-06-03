"""
compactor.facts — Persistent facts memory (V2.0 Phase 2, "semantic" layer).

Storage shape on disk (one JSON file per conversation):
    {
      "conv_id": "abc123...",
      "updated_at": "2026-05-28T05:00:00Z",
      "facts": [
        { "text": "Protagonist is Lyra, half-elf ranger, age 23.",
          "added_turn": 5,
          "last_used": 1748419200 },
        ...
      ]
    }

Each fact is one short bullet extracted by the LLM from a single exchange
(user message + assistant response). Facts are appended over time; LRU
eviction by `last_used` keeps the total under COMPACTOR_MAX_FACTS_TOKENS.

Lifecycle:
  1. Request arrives → load_facts(conv_id) → inject into request as system block
  2. Mark all loaded facts as `last_used = now` (LRU tracking)
  3. After response streams back → extract_facts_from_exchange() in async tail
  4. Append new facts → prune to budget → save_facts()

All file writes go through memory.atomic_write_json() for crash safety.
All read/write pairs are serialized per-conv via memory.conv_lock().
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from memory import (
    atomic_write_json,
    conv_lock,
    facts_archive_path,
    facts_path,
    read_json,
)

logger = logging.getLogger("compactor.facts")


# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

# Approximate token budget for the facts block injected into every request.
# We use char/4 as a fast estimator — precision doesn't matter for a soft
# cap. ~1500 tokens ≈ 6000 chars ≈ 100-150 short bullets.
_MAX_FACTS_TOKENS = int(os.environ.get("COMPACTOR_MAX_FACTS_TOKENS", "1500") or 1500)

# Max tokens the LLM produces per extraction call. Each call should yield
# at most a handful of bullets, so this is intentionally tight.
_EXTRACTION_MAX_TOKENS = int(
    os.environ.get("COMPACTOR_FACTS_EXTRACTION_MAX_TOKENS", "256") or 256
)

# Whether to even run fact extraction. Off → facts memory becomes append-only
# from manual /remember commands (V2.1 territory). Default on.
_EXTRACTION_ENABLED = (
    os.environ.get("COMPACTOR_FACTS_EXTRACTION", "true").lower() != "false"
)


def extraction_enabled() -> bool:
    return _EXTRACTION_ENABLED


# ---------------------------------------------------------------------------
# Fact shape + token estimation
# ---------------------------------------------------------------------------

# A fact is a dict — using TypedDict-style for clarity but plain dict for
# JSON round-trip simplicity.
#   { "text": str, "added_turn": int, "last_used": int (unix ts) }


def _now_unix() -> int:
    return int(time.time())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _estimate_tokens(text: str) -> int:
    """Approximate token count via char/4. Good enough for budget enforcement;
    we don't need precision because the cap itself is a soft target.
    """
    return len(text) // 4


# ---------------------------------------------------------------------------
# I/O — round-trips through memory.atomic_write_json
# ---------------------------------------------------------------------------

def load_facts(conv_id: str) -> list[dict]:
    """Return the current facts list for a conversation. Empty list if no
    facts file exists yet (or if the file is corrupted — logged).
    """
    data = read_json(facts_path(conv_id), default={})
    facts = data.get("facts", []) if isinstance(data, dict) else []
    # Defensive: ensure each entry has the expected shape; drop malformed.
    valid: list[dict] = []
    for f in facts:
        if isinstance(f, dict) and isinstance(f.get("text"), str) and f["text"].strip():
            valid.append({
                "text": f["text"],
                "added_turn": int(f.get("added_turn", 0)),
                "last_used": int(f.get("last_used", 0)),
            })
    return valid


def save_facts(conv_id: str, facts: list[dict]) -> None:
    """Persist a facts list. Atomic write — readers always see a coherent
    state. Caller is responsible for any pruning before calling.
    """
    data = {
        "conv_id": conv_id,
        "updated_at": _now_iso(),
        "facts": facts,
    }
    atomic_write_json(facts_path(conv_id), data)


# ---------------------------------------------------------------------------
# Stale-fact archival (V2.1 Phase 7 Step 2)
# ---------------------------------------------------------------------------
#
# The LRU budget in prune_facts evicts purely on storage pressure. Archival
# is the time-based companion: facts not retrieved-and-injected in N days
# get moved to a cold-storage sidecar file. They're still recoverable via
# restore_from_archive — the user just doesn't pay context-window cost for
# old facts that the model hasn't needed.
#
# Why a separate file vs a flag on the existing record: a flag would still
# count against prune_facts's token budget and would still appear in
# /admin/facts listings. Moving to a sidecar keeps the active set lean and
# makes the cold/hot distinction obvious in any tooling that walks storage.

ARCHIVE_DEFAULT_DAYS = int(
    os.environ.get("COMPACTOR_ARCHIVE_DEFAULT_DAYS", "90") or 90
)


def load_archive(conv_id: str) -> list[dict]:
    """Return the archived facts list for a conv. Empty if no archive yet."""
    data = read_json(facts_archive_path(conv_id), default={})
    archived = data.get("facts", []) if isinstance(data, dict) else []
    valid: list[dict] = []
    for f in archived:
        if (
            isinstance(f, dict)
            and isinstance(f.get("text"), str)
            and f["text"].strip()
        ):
            valid.append({
                "text": f["text"],
                "added_turn": int(f.get("added_turn", 0)),
                "last_used": int(f.get("last_used", 0)),
                "archived_at": int(f.get("archived_at", 0)),
            })
    return valid


def save_archive(conv_id: str, facts: list[dict]) -> None:
    """Persist the archive sidecar. Atomic — readers always see coherent state."""
    data = {
        "conv_id": conv_id,
        "updated_at": _now_iso(),
        "facts": facts,
    }
    atomic_write_json(facts_archive_path(conv_id), data)


def archive_stale_facts(
    conv_id: str, *, older_than_days: int = ARCHIVE_DEFAULT_DAYS
) -> tuple[int, int]:
    """Move facts with `last_used` older than the cutoff from active storage
    to the archive sidecar. Returns (kept_count, archived_count).

    Callers should serialize via conv_lock — concurrent extraction tail
    could otherwise see torn state mid-move. Idempotent: running twice with
    the same cutoff archives the same set on first call, zero on second.
    """
    if older_than_days < 0:
        return len(load_facts(conv_id)), 0
    cutoff = _now_unix() - (older_than_days * 86400)
    active = load_facts(conv_id)
    if not active:
        return 0, 0
    stale = [f for f in active if f.get("last_used", 0) < cutoff]
    if not stale:
        return len(active), 0
    fresh = [f for f in active if f.get("last_used", 0) >= cutoff]
    # Stamp archived_at so restore can tell when each fact was retired.
    archive_ts = _now_unix()
    archived_entries = [
        {**f, "archived_at": archive_ts} for f in stale
    ]
    # Merge with any existing archive — accumulating, not overwriting.
    existing_archive = load_archive(conv_id)
    save_archive(conv_id, existing_archive + archived_entries)
    save_facts(conv_id, fresh)
    logger.info(
        f"conv={conv_id}: archived {len(stale)} stale fact(s) "
        f"(cutoff: {older_than_days}d)"
    )
    return len(fresh), len(stale)


def restore_from_archive(
    conv_id: str, *, text_substring: str | None = None
) -> int:
    """Move matching archive entries back to active facts. Returns the
    number restored.

      text_substring=None — restore all archived facts
      text_substring="..."  — restore only facts whose text contains the
                              substring (case-insensitive)

    Caller serializes via conv_lock. Restored facts get their `last_used`
    bumped to now so they don't immediately re-archive on the next pass.
    The `archived_at` field is dropped (the fact is hot again).
    """
    archived = load_archive(conv_id)
    if not archived:
        return 0
    if text_substring:
        needle = text_substring.lower()
        to_restore = [f for f in archived if needle in f.get("text", "").lower()]
        remaining = [f for f in archived if needle not in f.get("text", "").lower()]
    else:
        to_restore = list(archived)
        remaining = []
    if not to_restore:
        return 0
    now = _now_unix()
    refreshed = [
        {
            "text": f["text"],
            "added_turn": f.get("added_turn", 0),
            "last_used": now,
        }
        for f in to_restore
    ]
    active = load_facts(conv_id)
    save_facts(conv_id, active + refreshed)
    save_archive(conv_id, remaining)
    logger.info(
        f"conv={conv_id}: restored {len(refreshed)} fact(s) from archive"
    )
    return len(refreshed)


async def with_facts_lock(conv_id: str, fn):
    """Run `fn` (an async callable) while holding the per-conv lock. Use
    for any read-modify-write sequence on facts to prevent torn updates
    between concurrent writers (e.g., new-request extraction vs. backfill).
    """
    async with conv_lock(conv_id):
        return await fn()


# ---------------------------------------------------------------------------
# Pruning — LRU by last_used
# ---------------------------------------------------------------------------

def prune_facts(
    facts: list[dict],
    max_tokens: int = _MAX_FACTS_TOKENS,
) -> tuple[list[dict], int]:
    """Trim facts down to fit max_tokens. LRU eviction by `last_used`
    (least-recently-used dropped first). Returns (kept, dropped_count).

    The eviction order is stable: facts with identical last_used preserve
    insertion order, so manual /remember additions (V2.1) won't randomly
    lose to extraction-time additions of the same timestamp.
    """
    if not facts:
        return [], 0
    total = sum(_estimate_tokens(f["text"]) for f in facts)
    if total <= max_tokens:
        return facts, 0

    # Sort by last_used ascending (oldest first), then by added_turn for stability
    sorted_facts = sorted(facts, key=lambda f: (f["last_used"], f["added_turn"]))
    kept_reversed: list[dict] = []
    running = 0
    # Walk from most-recently-used backward, keeping facts that fit
    for f in reversed(sorted_facts):
        cost = _estimate_tokens(f["text"])
        if running + cost <= max_tokens:
            kept_reversed.append(f)
            running += cost
    # Restore original-ish ordering by added_turn for stable injection
    kept = sorted(kept_reversed, key=lambda f: f["added_turn"])
    return kept, len(facts) - len(kept)


# ---------------------------------------------------------------------------
# Injection — turn facts into a system message block for the LLM request
# ---------------------------------------------------------------------------

_FACTS_BLOCK_HEADER = (
    "[Persistent facts about this conversation — established earlier, "
    "maintain consistency with these]"
)


def format_facts_block(facts: list[dict]) -> str | None:
    """Render facts as a system-message body. Returns None if no facts.
    Caller wraps in {"role": "system", "content": <this>}.
    """
    if not facts:
        return None
    lines = [_FACTS_BLOCK_HEADER]
    for f in facts:
        lines.append(f"- {f['text']}")
    return "\n".join(lines)


def touch_facts(facts: list[dict], now: int | None = None) -> list[dict]:
    """Mark every fact as just-used (for LRU). Mutates the list in place
    AND returns it for chaining. Call after injecting facts into a request
    so the eviction order reflects actual usage.
    """
    ts = now if now is not None else _now_unix()
    for f in facts:
        f["last_used"] = ts
    return facts


# ---------------------------------------------------------------------------
# Extraction — async LLM call against vLLM
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You extract persistent facts from a conversation exchange so they can be remembered for the rest of the conversation.

DEFAULT BEHAVIOR: extract every concrete piece of information the USER stated. Bias toward extracting. The cost of missing a fact is high; the cost of a slightly trivial fact is low.

Extract:
- Named entities the user introduced (characters, places, items, factions, projects, names)
- User preferences or instructions ("write in past tense", "avoid romance subplots")
- World/setting/story details the user established (magic systems, rules, technologies)
- Decisions the user made (story choices, plot directions, design choices)
- Constraints the user set (genre, tone, content limits)

OUTPUT FORMAT — STRICT:
- One fact per line, each line prefixed with "- "
- Each fact: ONE concise sentence under 20 words
- Output ONLY bullets — no preamble, no commentary, no headings, no closing remark
- Do NOT restate facts already in the EXISTING FACTS list below
- Do NOT extract things the assistant invented; only what the user stated or confirmed

ONLY return the literal word NONE (no other characters) when the user's message contained zero concrete information — e.g. just "ok", "thanks", "continue", or a one-word reaction. If the user named anything, expressed any preference, or stated any detail, extract it. When in doubt, extract."""


def _build_extraction_messages(
    user_msg: str, assistant_msg: str, existing_facts: list[dict]
) -> list[dict]:
    """Build the LLM request payload for one extraction call."""
    existing_block = "\n".join(f"- {f['text']}" for f in existing_facts) or "(none)"
    user_content = (
        f"EXISTING FACTS:\n{existing_block}\n\n"
        f"LATEST EXCHANGE:\n[user]: {user_msg}\n[assistant]: {assistant_msg}"
    )
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_extraction_output(raw: str) -> list[str]:
    """Parse the LLM's output into a clean list of fact strings. Handles:
    - "NONE" (any casing, with/without trailing punctuation) → []
    - Bullets prefixed with -, *, • → stripped
    - Numbered lists "1. ..." → stripped
    - Blank lines → skipped
    - Lines too short to be a fact (< 6 chars) → skipped
    """
    if not raw or not raw.strip():
        return []
    cleaned = raw.strip()
    if cleaned.upper().rstrip(".").strip() == "NONE":
        return []
    out: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading bullets / numbering
        for prefix in ("- ", "* ", "• ", "– "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        else:
            # Strip "1. ", "2. ", etc.
            if len(line) >= 3 and line[0].isdigit() and line[1:3] in (". ", ") "):
                line = line[3:].strip()
        if len(line) >= 6:
            out.append(line)
    return out


async def extract_facts_from_exchange(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    user_msg: str,
    assistant_msg: str,
    existing_facts: list[dict],
    *,
    timeout: float = 120.0,
) -> list[str]:
    """Call vLLM to extract new facts from one user/assistant exchange.

    Returns a list of fact strings (possibly empty). Caller is responsible
    for assigning added_turn / last_used and appending to the facts file.

    Errors (network, vLLM 5xx, parse failures) return [] — fact extraction
    must NEVER block or break the user's chat flow. Failures get logged.
    """
    if not user_msg or not assistant_msg:
        return []
    payload = {
        "model": model,
        "messages": _build_extraction_messages(user_msg, assistant_msg, existing_facts),
        "max_tokens": _EXTRACTION_MAX_TOKENS,
        # temp 0.0: extraction is structured-output, not creative writing.
        # We want the same input to always produce the same facts. The
        # previous 0.2 produced ~35% NONE rate on Magnum-12B with fact-rich
        # prompts — pure model variance, not a real "no facts" signal.
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = await client.post(
            f"{vllm_url}/v1/chat/completions", json=payload, timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        raw = data["choices"][0]["message"]["content"]
        facts = _parse_extraction_output(raw)
        if facts:
            logger.info(f"extracted {len(facts)} new fact(s)")
        else:
            # Empty result has two distinct causes; log both so the
            # silence isn't a diagnostic dead-end during integration runs.
            snippet = (raw or "").strip().replace("\n", " ")[:120]
            logger.info(
                f"extracted 0 fact(s) — model returned: {snippet!r}"
            )
        return facts
    except Exception as e:
        logger.warning(f"fact extraction failed (non-fatal): {e}")
        return []


# ---------------------------------------------------------------------------
# Convenience: complete the read-extract-prune-write cycle
# ---------------------------------------------------------------------------

async def record_facts_for_exchange(
    conv_id: str,
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    user_msg: str,
    assistant_msg: str,
    turn_index: int,
) -> int:
    """The full async-tail facts loop: load existing, extract new from
    the exchange, append, prune to budget, write back atomically.
    Serialized per-conv via the conv_lock to prevent torn updates.

    Returns the number of NEW facts added. Always safe to call — never
    raises (failures logged + return 0).
    """
    async def _run() -> int:
        try:
            existing = load_facts(conv_id)
            new_strs = await extract_facts_from_exchange(
                client, vllm_url, model, user_msg, assistant_msg, existing
            )
            if not new_strs:
                return 0
            now = _now_unix()
            new_entries = [
                {"text": s, "added_turn": turn_index, "last_used": now}
                for s in new_strs
            ]
            combined = existing + new_entries
            kept, dropped = prune_facts(combined)
            save_facts(conv_id, kept)
            if dropped:
                logger.info(
                    f"conv={conv_id}: +{len(new_entries)} facts, pruned {dropped} oldest"
                )
            return len(new_entries)
        except Exception as e:
            logger.exception(f"record_facts_for_exchange failed (non-fatal): {e}")
            return 0

    return await with_facts_lock(conv_id, _run)
