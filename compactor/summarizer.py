"""
compactor.summarizer — Hierarchical "working" memory (V2.0 Phase 4).

The third memory layer (alongside facts in facts.py and episodic in
retrieval.py). Replaces v1's single-shot flat summary with tiered summaries
that preserve narrative continuity at multiple resolutions:

    L1: 20-turn "chunk" summaries (recent narrative beats)
    L2: ~10×L1 "chapter" summaries (story arcs)
    L3: whole-conversation theme/state (highest level, optional)

Why tiered:
- v1's flat summary repeatedly re-summarizes already-summarized content,
  losing specifics with each pass ("summary-of-summary degradation").
- Tiered summaries roll older content into denser representations without
  re-touching it — once an L1 chunk is created from turns 1-20, it never
  gets re-summarized; only when 10+ L1 chunks exist do they roll into L2.
- Total injected size stays bounded (~5K tokens worst case: L3 + latest
  L2 + a handful of unrolled L1 chunks).

Storage (one JSON per conv):
    /data/openwebui/compactor/summaries/<conv_id>.json
    {
      "conv_id": "...",
      "updated_at": "ISO",
      "l1": [{"text": "...", "first_turn": 1, "last_turn": 20}, ...],
      "l2": [{"text": "...", "first_turn": 1, "last_turn": 200}, ...],
      "l3": {"text": "...", "first_turn": 1, "last_turn": 1000} | null,
      "last_summarized_turn": 20  # highest turn covered by any L1 chunk
    }

Lifecycle:
  request time (sync, cheap): load_state → format injection block from
    existing L3+L2+(unrolled L1s) → inject as system message.
  post-response (async, may do LLM calls): maybe_rollup checks thresholds
    and triggers L1 / L2 / L3 rollups if enough new material accumulated.

All operations degrade to safe no-ops on failure — chat never breaks because
the summarizer hit a problem.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from memory import atomic_write_json, conv_lock, read_json, storage_root

logger = logging.getLogger("compactor.summarizer")


# ---------------------------------------------------------------------------
# Configuration (env-overridable, sensible defaults)
# ---------------------------------------------------------------------------

L1_CHUNK_SIZE = int(os.environ.get("COMPACTOR_L1_CHUNK_SIZE", "20") or 20)
L2_CHUNK_SIZE = int(os.environ.get("COMPACTOR_L2_CHUNK_SIZE", "10") or 10)
L3_CHUNK_SIZE = int(os.environ.get("COMPACTOR_L3_CHUNK_SIZE", "5") or 5)

# Per-tier token budget for the LLM's output (input tokens depend on how
# much we're summarizing). L3 is largest because it must represent the
# whole conversation; L1 is smallest because each chunk is one "scene."
L1_MAX_TOKENS = int(os.environ.get("COMPACTOR_L1_MAX_TOKENS", "500") or 500)
L2_MAX_TOKENS = int(os.environ.get("COMPACTOR_L2_MAX_TOKENS", "1200") or 1200)
L3_MAX_TOKENS = int(os.environ.get("COMPACTOR_L3_MAX_TOKENS", "2000") or 2000)

# Master switch — set false to fall back to v1 flat summary (or no summary).
ENABLED = os.environ.get("COMPACTOR_HIERARCHICAL_SUMMARY", "true").lower() != "false"


def enabled() -> bool:
    return ENABLED


# ---------------------------------------------------------------------------
# Storage paths + helpers
# ---------------------------------------------------------------------------

def summary_path(conv_id: str):
    """File path for this conversation's hierarchical summary state.
    Kept alongside facts (in the summaries/ subdir per the V2.0 layout).
    """
    return storage_root() / "summaries" / f"{conv_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_state(conv_id: str) -> dict:
    return {
        "conv_id": conv_id,
        "updated_at": _now_iso(),
        "l1": [],
        "l2": [],
        "l3": None,
        "last_summarized_turn": 0,
    }


def load_state(conv_id: str) -> dict:
    """Return current summary state. Empty (but well-formed) skeleton if
    no file exists or it's corrupt — failures are non-fatal."""
    data = read_json(summary_path(conv_id), default=None)
    if not isinstance(data, dict):
        return _empty_state(conv_id)
    # Defensive: ensure all top-level keys exist with the right types.
    state = _empty_state(conv_id)
    if isinstance(data.get("l1"), list):
        state["l1"] = [x for x in data["l1"] if _is_chunk(x)]
    if isinstance(data.get("l2"), list):
        state["l2"] = [x for x in data["l2"] if _is_chunk(x)]
    if isinstance(data.get("l3"), dict) and _is_chunk(data["l3"]):
        state["l3"] = data["l3"]
    if isinstance(data.get("last_summarized_turn"), int):
        state["last_summarized_turn"] = data["last_summarized_turn"]
    return state


def save_state(conv_id: str, state: dict) -> None:
    state["conv_id"] = conv_id
    state["updated_at"] = _now_iso()
    atomic_write_json(summary_path(conv_id), state)


def _is_chunk(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and isinstance(x.get("text"), str)
        and x["text"].strip()
        and isinstance(x.get("first_turn"), int)
        and isinstance(x.get("last_turn"), int)
    )


# ---------------------------------------------------------------------------
# Injection — format the existing summary stack as a system message
# ---------------------------------------------------------------------------

_BLOCK_HEADER = (
    "[Hierarchical summary of earlier portions of this conversation, ordered "
    "by recency — use them for continuity. Older summaries are denser; the "
    "L3 line (if present) is the whole-conversation theme.]"
)


def format_summary_block(state: dict) -> str | None:
    """Render the current summary stack into a single system-message body.
    Returns None if there's nothing to inject.

    Order in the rendered block (most-general → most-specific):
      1. L3 (whole-conversation theme), if any
      2. L2 chapters in chronological order
      3. L1 chunks in chronological order
    The most-recent L1s are what the model needs most for continuity, so
    they come last (right before the recent raw turns will appear in the
    final message list).
    """
    has_l3 = state.get("l3") is not None
    l2 = state.get("l2") or []
    l1 = state.get("l1") or []
    if not (has_l3 or l2 or l1):
        return None

    lines = [_BLOCK_HEADER]
    if has_l3:
        l3 = state["l3"]
        lines.append(f"\n--- conversation-wide theme (turns {l3.get('first_turn','?')}-{l3.get('last_turn','?')}) ---")
        lines.append(l3.get("text", ""))
    if l2:
        for ch in l2:
            lines.append(f"\n--- chapter (turns {ch.get('first_turn','?')}-{ch.get('last_turn','?')}) ---")
            lines.append(ch.get("text", ""))
    if l1:
        for ch in l1:
            lines.append(f"\n--- scene (turns {ch.get('first_turn','?')}-{ch.get('last_turn','?')}) ---")
            lines.append(ch.get("text", ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rollup trigger detection
# ---------------------------------------------------------------------------

def _needs_l1_rollup(state: dict, current_turn_count: int) -> bool:
    """True if there are >= L1_CHUNK_SIZE turns past last_summarized_turn."""
    last = state.get("last_summarized_turn", 0)
    return (current_turn_count - last) >= L1_CHUNK_SIZE


def _needs_l2_rollup(state: dict) -> bool:
    """True if accumulated L1 chunks have crossed the L2 threshold."""
    return len(state.get("l1", [])) >= L2_CHUNK_SIZE


def _needs_l3_rollup(state: dict) -> bool:
    """True if accumulated L2 chapters have crossed the L3 threshold."""
    return len(state.get("l2", [])) >= L3_CHUNK_SIZE


def needs_rollup(state: dict, current_turn_count: int) -> bool:
    """Public: any tier needs work?"""
    return (
        _needs_l1_rollup(state, current_turn_count)
        or _needs_l2_rollup(state)
        or _needs_l3_rollup(state)
    )


# ---------------------------------------------------------------------------
# Message ↔ turn helpers
# ---------------------------------------------------------------------------

def _message_text(m: dict) -> str:
    content = m.get("content") or ""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _format_turns(messages: list[dict], first_turn: int, last_turn: int) -> str:
    """Render the slice of messages corresponding to turn indices
    [first_turn .. last_turn] (1-indexed, system messages skipped) as
    a flat transcript suitable for the LLM to summarize.
    """
    # Walk messages assigning turn indices to non-system entries.
    parts: list[str] = []
    idx = 0
    for m in messages:
        if m.get("role") == "system":
            continue
        idx += 1
        if idx < first_turn:
            continue
        if idx > last_turn:
            break
        role = m.get("role", "unknown")
        parts.append(f"[{role}]: {_message_text(m)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM-driven summarization (one call per rollup)
# ---------------------------------------------------------------------------

_PROMPT_L1 = """Summarize the following conversation excerpt for long-term recall. Preserve:
- Names, places, decisions, and concrete details.
- The user's stated preferences and goals.
- Code, file paths, commands, URLs, or numeric values mentioned.
- Plot/story beats if this is creative writing.
Do not greet, editorialize, or hedge. Output the summary only."""

_PROMPT_L2 = """You are summarizing several earlier per-scene summaries into one "chapter-level" summary. Preserve continuity at the chapter scale: characters, settings, decisions, ongoing threads. Drop scene-by-scene minutiae but keep names and concrete decisions. Output the chapter summary only — no preamble, no hedging."""

_PROMPT_L3 = """You are producing the whole-conversation "theme" summary from a list of chapter-level summaries. Capture the high-level arc, the user's overarching goals, persistent constraints, and the cast of named entities. This will be injected on every future request, so be concise but never vague. Output the theme summary only."""


async def _llm_summarize(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    system_prompt: str,
    body_text: str,
    max_tokens: int,
    *,
    timeout: float = 300.0,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": body_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    r = await client.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"].strip()
    return out


# ---------------------------------------------------------------------------
# Rollup orchestration
# ---------------------------------------------------------------------------

async def _do_l1_rollup(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    state: dict,
    messages: list[dict],
) -> bool:
    """Roll the next L1_CHUNK_SIZE turns after last_summarized_turn into a
    new L1 chunk. Returns True if a chunk was produced.
    """
    last = state.get("last_summarized_turn", 0)
    first_turn = last + 1
    last_turn = last + L1_CHUNK_SIZE
    body = _format_turns(messages, first_turn, last_turn)
    if not body.strip():
        return False
    text = await _llm_summarize(
        client, vllm_url, model, _PROMPT_L1, body, L1_MAX_TOKENS
    )
    if not text:
        return False
    state["l1"].append({
        "text": text, "first_turn": first_turn, "last_turn": last_turn,
    })
    state["last_summarized_turn"] = last_turn
    return True


async def _do_l2_rollup(
    client: httpx.AsyncClient, vllm_url: str, model: str, state: dict,
) -> bool:
    """Roll the OLDEST L2_CHUNK_SIZE L1 chunks into one L2 chapter, dropping
    them from the L1 list. Returns True if a chapter was produced.
    """
    l1 = state.get("l1") or []
    if len(l1) < L2_CHUNK_SIZE:
        return False
    chunks = l1[:L2_CHUNK_SIZE]
    body = "\n\n".join(
        f"--- scene (turns {c['first_turn']}-{c['last_turn']}) ---\n{c['text']}"
        for c in chunks
    )
    text = await _llm_summarize(
        client, vllm_url, model, _PROMPT_L2, body, L2_MAX_TOKENS
    )
    if not text:
        return False
    state["l2"].append({
        "text": text,
        "first_turn": chunks[0]["first_turn"],
        "last_turn": chunks[-1]["last_turn"],
    })
    state["l1"] = l1[L2_CHUNK_SIZE:]  # drop the rolled-up chunks
    return True


async def _do_l3_rollup(
    client: httpx.AsyncClient, vllm_url: str, model: str, state: dict,
) -> bool:
    """Roll all L2 chapters into / refresh L3. Unlike L1→L2, this keeps the
    L2 list (so the next request still has the chapters available) and
    just refreshes the L3 theme. L3 is a single object, not a list.
    """
    l2 = state.get("l2") or []
    if len(l2) < L3_CHUNK_SIZE:
        return False
    body = "\n\n".join(
        f"--- chapter (turns {c['first_turn']}-{c['last_turn']}) ---\n{c['text']}"
        for c in l2
    )
    text = await _llm_summarize(
        client, vllm_url, model, _PROMPT_L3, body, L3_MAX_TOKENS
    )
    if not text:
        return False
    state["l3"] = {
        "text": text,
        "first_turn": l2[0]["first_turn"],
        "last_turn": l2[-1]["last_turn"],
    }
    return True


async def maybe_rollup(
    conv_id: str,
    messages: list[dict],
    vllm_url: str,
    model: str,
) -> dict:
    """Public entry point. Loads state, runs whichever tier(s) need work,
    saves atomically. Held under conv_lock so concurrent rollups can't tear
    state. Returns the new state. Never raises (failures logged + state
    returned best-effort).

    `messages` is the FULL message history (caller usually has the request's
    messages list right there), so L1 rollups can format the exact turns
    that need summarizing.

    `current_turn_count` is derived from messages (non-system count) so the
    caller doesn't have to track it.
    """
    current_turns = sum(1 for m in messages if m.get("role") != "system")

    async with conv_lock(conv_id):
        state = load_state(conv_id)

        if not needs_rollup(state, current_turns):
            return state

        try:
            async with httpx.AsyncClient() as client:
                # Drain L1 rollups until either caught up or no more material.
                while _needs_l1_rollup(state, current_turns):
                    ok = await _do_l1_rollup(client, vllm_url, model, state, messages)
                    if not ok:
                        break

                # Drain L2 rollups while threshold met.
                while _needs_l2_rollup(state):
                    ok = await _do_l2_rollup(client, vllm_url, model, state)
                    if not ok:
                        break

                # L3 is at most one rollup per call (refresh, not stack).
                if _needs_l3_rollup(state):
                    await _do_l3_rollup(client, vllm_url, model, state)
        except Exception as e:
            logger.exception(f"conv={conv_id}: rollup failed mid-flight: {e}")

        # Save whatever we got, even partial — better than losing work.
        try:
            save_state(conv_id, state)
        except Exception as e:
            logger.warning(f"conv={conv_id}: save_state failed: {e}")

        return state


# ---------------------------------------------------------------------------
# Diagnostics for admin endpoint
# ---------------------------------------------------------------------------

def state_summary(state: dict) -> dict:
    """Compact, JSON-serializable view of state for /admin/conversations/<id>.
    """
    l3 = state.get("l3")
    return {
        "l1_chunks": len(state.get("l1") or []),
        "l2_chapters": len(state.get("l2") or []),
        "l3_present": l3 is not None,
        "last_summarized_turn": state.get("last_summarized_turn", 0),
        "l3_turns_covered": (
            [l3.get("first_turn"), l3.get("last_turn")] if l3 else None
        ),
    }
