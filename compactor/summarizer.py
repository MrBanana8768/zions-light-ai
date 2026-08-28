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
- Total injected size was INTENDED to stay bounded (~5K tokens worst case:
  L3 + latest L2 + a handful of unrolled L1 chunks). It is not: nothing
  trims l2 and format_summary_block renders every chapter, so this layer
  grows without limit (MEMORY_REVIEW S-1/S-6). Read the line above as the
  design, not as a property you may rely on.

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

import logsetup
from memory import atomic_write_json, conv_lock, read_json_strict, storage_root

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


# Entries that fail _is_chunk are parked under this key by load_state and
# folded back in by save_state, instead of being dropped. The filter used to
# be silently destructive: load_state discarded whatever it didn't recognise
# and the next save_state persisted the filtered list, so a schema change —
# or a single chunk written by a newer build — deleted summaries nobody had
# asked to delete. Round-tripping them verbatim costs nothing and the tiers
# below never see them (v3.1 F1b, change 4).
_UNRECOGNIZED = "_unrecognized"


def load_state(conv_id: str) -> dict:
    """Return current summary state. Empty (but well-formed) skeleton if
    no file exists.

    Raises memory.StoreUnreadable if the file IS there and could not be
    read. Handing back the skeleton for that case is what let one misread
    replace an entire L1/L2/L3 hierarchy with a summary of the client's
    current window — worse than the facts equivalent, because summaries are
    replaced wholesale rather than merged (v3.1 F1b).
    """
    data = read_json_strict(summary_path(conv_id), default=None)
    if not isinstance(data, dict):
        return _empty_state(conv_id)
    # Defensive: ensure all top-level keys exist with the right types.
    state = _empty_state(conv_id)
    parked: dict = {"l1": [], "l2": [], "l3": None}
    for tier in ("l1", "l2"):
        if isinstance(data.get(tier), list):
            state[tier] = [x for x in data[tier] if _is_chunk(x)]
            parked[tier] = [x for x in data[tier] if not _is_chunk(x)]
    if isinstance(data.get("l3"), dict) and _is_chunk(data["l3"]):
        state["l3"] = data["l3"]
    elif data.get("l3") is not None:
        parked["l3"] = data["l3"]
    if isinstance(data.get("last_summarized_turn"), int):
        state["last_summarized_turn"] = data["last_summarized_turn"]
    if parked["l1"] or parked["l2"] or parked["l3"] is not None:
        state[_UNRECOGNIZED] = parked
    return state


def _for_disk(state: dict) -> dict:
    """The on-disk form of `state`: parked entries folded back, private key
    gone. They land after the chunks we do understand — position isn't
    preserved, content is, which is what "don't delete what you can't
    parse" actually requires. Nothing renders them either way.
    """
    parked = state.get(_UNRECOGNIZED)
    out = {k: v for k, v in state.items() if k != _UNRECOGNIZED}
    if not isinstance(parked, dict):
        return out
    for tier in ("l1", "l2"):
        extra = parked.get(tier) or []
        if extra:
            out[tier] = list(out.get(tier) or []) + list(extra)
    # Only restore a parked l3 if the live state still has none — a rollup
    # that produced a real L3 must not be reverted to the unparseable one.
    if parked.get("l3") is not None and out.get("l3") is None:
        out["l3"] = parked["l3"]
    return out


def save_state(conv_id: str, state: dict) -> None:
    state["conv_id"] = conv_id
    state["updated_at"] = _now_iso()
    atomic_write_json(summary_path(conv_id), _for_disk(state))


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


def _reconcile_watermark(state: dict, current_turn_count: int) -> bool:
    """Pull last_summarized_turn back to what the history actually contains.
    Returns True if the watermark moved.

    `last_summarized_turn` is an absolute position in whatever array the
    client sent (S-5 / REMEDIATION F14). Whenever the observed history is
    SHORTER than it — a client sending a bounded window, a user deleting or
    editing messages, a branch switch — `current_turn_count - last` is
    negative, so `_needs_l1_rollup` is False on this turn and on every turn
    after it. The hierarchy stops advancing permanently and silently. That
    is not hypothetical: 19.8 hours of production logs show every summary
    injection reading L1=5 / L2=0 while the conversation ran from turn ~42
    to ~58.

    Resetting to the observed count un-latches the gate without
    re-summarizing anything: rollups resume once L1_CHUNK_SIZE new turns
    arrive. If the history later grows past the old watermark again, the
    turns between will be summarized a second time — accepting a duplicate
    chunk is the cheap half of the trade against a hierarchy that never
    moves again.

    The L1 chunks covering turns that are no longer observable are KEPT.
    They are the only surviving record of that material, and deleting
    summaries to repair a counter is exactly how the five destructive
    memory paths removed earlier on this branch started. Their turn labels
    stay wrong until D1 gives turns durable identities; this function fixes
    the stall, not the units.
    """
    last = state.get("last_summarized_turn", 0)
    if current_turn_count >= last:
        return False
    state["last_summarized_turn"] = current_turn_count
    return True


def _needs_l2_rollup(state: dict) -> bool:
    """True if accumulated L1 chunks have crossed the L2 threshold."""
    return len(state.get("l1", [])) >= L2_CHUNK_SIZE


def _needs_l3_rollup(state: dict) -> bool:
    """True if enough L2 chapters exist AND L3 does not already cover them.

    The threshold alone is a standing condition, not an event (MEMORY_REVIEW
    S-2): `_do_l3_rollup` keeps the L2 list, unlike L1→L2 which drops what it
    consumed, so from the L3_CHUNK_SIZE-th chapter onward `len(l2) >=
    L3_CHUNK_SIZE` never clears. Every turn then spent one L3_MAX_TOKENS LLM
    call re-paraphrasing the same chapters, and kept `needs_rollup` True, so
    maybe_rollup's early exit never fired either. Comparing L3's recorded
    span against the current chapter span gives L3 the event semantics L1→L2
    always had: refresh when the chapters have actually moved.
    """
    l2 = state.get("l2") or []
    if len(l2) < L3_CHUNK_SIZE:
        return False
    l3 = state.get("l3")
    if not isinstance(l3, dict):
        return True
    return (
        l3.get("first_turn") != l2[0].get("first_turn")
        or l3.get("last_turn") != l2[-1].get("last_turn")
    )


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
    conv_id: str,
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
    # A rollup had no success line of its own, so the only evidence the
    # hierarchy was advancing was the injection counter — which is why S-5
    # froze it for the life of the deployment without anyone noticing.
    logger.info(
        f"conv={conv_id}: L1 rollup — chunk {len(state['l1'])} covers turns "
        f"{first_turn}-{last_turn}"
    )
    return True


async def _do_l2_rollup(
    conv_id: str, client: httpx.AsyncClient, vllm_url: str, model: str, state: dict,
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
    logger.info(
        f"conv={conv_id}: L2 rollup — chapter {len(state['l2'])} covers turns "
        f"{chunks[0]['first_turn']}-{chunks[-1]['last_turn']} from "
        f"{len(chunks)} L1 chunks"
    )
    return True


async def _do_l3_rollup(
    conv_id: str, client: httpx.AsyncClient, vllm_url: str, model: str, state: dict,
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
    logger.info(
        f"conv={conv_id}: L3 refresh — covers turns {l2[0]['first_turn']}-"
        f"{l2[-1]['last_turn']} over {len(l2)} chapters"
    )
    return True


async def maybe_rollup(
    conv_id: str,
    messages: list[dict],
    vllm_url: str,
    model: str,
) -> dict:
    """Public entry point. Loads state, runs whichever tier(s) need work,
    saves atomically. Held under conv_lock so concurrent rollups can't tear
    state. Returns the new state. An LLM failure is logged and swallowed;
    an unreadable state file propagates memory.StoreUnreadable, because the
    one thing this function must never do is write a state it could not
    read (v3.1 F1b/G3). The caller's tail already treats that as a
    non-fatal skipped rollup. Tiers that completed before a failure are
    persisted; only the tier that failed retries on the next turn.

    `messages` is the FULL message history (caller usually has the request's
    messages list right there), so L1 rollups can format the exact turns
    that need summarizing.

    `current_turn_count` is derived from messages (non-system count) so the
    caller doesn't have to track it.
    """
    current_turns = sum(1 for m in messages if m.get("role") != "system")

    async with conv_lock(conv_id):
        state = load_state(conv_id)

        stale = state.get("last_summarized_turn", 0)
        changed = _reconcile_watermark(state, current_turns)
        if changed and logsetup.log_once("summarizer.watermark.reset"):
            # WARNING, and separate from the quiet path: a negative delta
            # reads exactly like "not enough new material" from the outside,
            # and that is why it went unnoticed. Once per process because
            # this is on the tail of every turn (v3.1 P0-2b).
            logger.warning(
                f"conv={conv_id}: observed history ({current_turns} turns) is "
                f"shorter than last_summarized_turn ({stale}); the L1 gate was "
                f"latched off and has been reset to {current_turns} — earlier "
                f"chunks are kept, and their turn labels no longer line up "
                f"with this history"
            )

        if needs_rollup(state, current_turns):
            try:
                async with httpx.AsyncClient() as client:
                    # Drain L1 rollups until either caught up or no more material.
                    while _needs_l1_rollup(state, current_turns):
                        if not await _do_l1_rollup(
                            conv_id, client, vllm_url, model, state, messages
                        ):
                            break
                        changed = True

                    # Drain L2 rollups while threshold met.
                    while _needs_l2_rollup(state):
                        if not await _do_l2_rollup(
                            conv_id, client, vllm_url, model, state
                        ):
                            break
                        changed = True

                    # L3 is at most one rollup per call (refresh, not stack).
                    if _needs_l3_rollup(state):
                        if await _do_l3_rollup(conv_id, client, vllm_url, model, state):
                            changed = True
            except Exception as e:
                logger.exception(f"conv={conv_id}: rollup failed mid-flight: {e}")

        # The write sits OUTSIDE the rollup try (v3.1 G3, revised for
        # MEMORY_REVIEW S-3). G3 moved it in because a rollup that died
        # mid-flight still persisted whatever `state` held, and a load that
        # returned the empty skeleton on a misread made that skeleton the
        # thing written. The load is the half that got fixed: load_state now
        # raises StoreUnreadable and is called above, outside every try, so
        # nothing here can reach save_state with a state it did not read.
        # What was left was the other half — an L3 failure discarding the L1
        # rollups that had already succeeded, on every turn, forever. Each
        # _do_*_rollup mutates `state` only after its own LLM call returns,
        # so `state` here is always a consistent prefix of successful
        # rollups whether or not a later tier raised.
        if changed:
            try:
                save_state(conv_id, state)
            except Exception as e:
                logger.exception(f"conv={conv_id}: rollup state write failed: {e}")

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
