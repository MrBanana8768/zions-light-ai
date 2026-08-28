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

import asyncio
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

# The model's context window, and the slack left inside it for a
# summarization call's system prompt, wrapper text and chat-template framing.
# Same env vars main.py reads, deliberately: the two summarization paths must
# not be tunable apart, and a rollup that budgets against a different window
# than the request path is the same defect in a second place.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768") or 32768)
SUMMARY_INPUT_RESERVE = int(
    os.environ.get("COMPACTOR_SUMMARY_INPUT_RESERVE", "2048") or 2048
)

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


def _turn_pieces(messages: list[dict], first_turn: int, last_turn: int) -> list[str]:
    """The slice of messages for turn indices [first_turn .. last_turn]
    (1-indexed, system messages skipped), one rendered string per turn.

    Split rather than joined because the budget below has to be able to pack
    these into batches that fit the model's window. A turn is the smallest
    unit this module will divide the transcript into.
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
    return parts


def _format_turns(messages: list[dict], first_turn: int, last_turn: int) -> str:
    """The same slice as one flat transcript.

    No longer on the rollup path — since A1 every tier goes through
    `_turn_pieces` so an oversized slice can be split. Kept because it is
    exactly what the pre-A1 code sent in one unbudgeted call, which makes it
    the ground truth a budget test measures itself against.
    """
    return "\n\n".join(_turn_pieces(messages, first_turn, last_turn))


# ---------------------------------------------------------------------------
# Input budget (v3.1 A1)
# ---------------------------------------------------------------------------
#
# Until v3.1 this module had NO token accounting whatsoever. Rollup input was
# bounded by turn COUNT (L1_CHUNK_SIZE) and by nothing else, so the size of a
# rollup request was whatever the user's twenty turns happened to weigh. On the
# conversation behind INCIDENT_2026-08-28 an assistant turn measures
# 7,513-11,347 tokens; ten of them plus ten user turns is multiples of the
# 32,768-token window. Every L1 rollup therefore 400'd — and because
# `r.raise_for_status()` fires before the watermark write, `last_summarized_turn`
# never advanced, `_needs_l1_rollup` stayed true forever, and the identical
# doomed request was re-issued on the tail of every subsequent turn. L1 never
# grew, so L2 and L3 never fired either. The whole hierarchy was dead, loudly,
# for the life of the conversation.
#
# WHY THE COUNTER IS AN HTTP CALL AND NOT AN IMPORT
# -------------------------------------------------
# `main.py` already has `count_tokens` / `count_tokens_exact`, and this module
# cannot use them: `main` imports `summarizer`, so importing back is a cycle.
# The clean answer is to extract them into a shared `tokens.py` that both
# import — a cross-module refactor, and not one to make from inside this file.
#
# But `count_tokens_exact` is not really a library function. It is one POST to
# `{vllm_url}/tokenize`, and this module is *already* holding `vllm_url`, the
# model name, and an open `httpx.AsyncClient` at every place a budget decision
# is made. So it asks the server directly. That is not a duplicate of main's
# arithmetic; it is the same question put to the same process, on a connection
# we already have, and it sidesteps the import cycle entirely.
#
# Copying main's LOCAL path (transformers + char/4) instead would have been the
# wrong half to copy: that estimator is the thing INCIDENT_2026-08-28 is about.
# It read 34-51% LOW on this model's assistant content.
#
# And unlike main.py — which is counting from sync call sites and has to go
# through `run_in_threadpool` — every caller here is already async, so the call
# is awaited on the loop and blocks nothing.

# Fallback density for when /tokenize cannot answer, in tokens per character.
#
# NOT a prose multiplier. INCIDENT_2026-08-28:35-37 measures one production
# reply carrying 1,710 x U+2501 plus 441 x U+2500 — 2,151 characters that vLLM
# charged roughly 4,275 tokens, i.e. ~1.99 tokens per character. A multiplier
# tuned for prose (~0.25) is wrong on that content by nearly 8x and would only
# move the failure. 2.0 is the worst density this project has actually
# measured, used as a ceiling.
#
# It is a ceiling, not a proof — a pathological input could beat it. What it
# buys is that being wrong here degrades to the PRE-FIX behaviour (an oversized
# request, a 400, a retry next turn) rather than to silent loss, and it is only
# reachable when /tokenize is down.
_WORST_TOKENS_PER_CHAR = 2.0

_TRUNCATION_NOTE = "\n\n[... truncated to fit the summarizer's input budget ...]"


def _pessimistic_tokens(text: str) -> int:
    """The most this text could plausibly cost. Used to decide whether it is
    worth asking the server at all."""
    return int(len(text) * _WORST_TOKENS_PER_CHAR)


def _input_budget(max_tokens: int) -> int:
    """Tokens available for a summarization call's INPUT, given what the call
    reserves for its own output.

    Clamped the way main.HARD_INPUT_LIMIT is clamped, and for the same reason:
    a bare floor could sit above the model's own window on a small-context
    model and quietly reintroduce the overflow this exists to prevent.
    """
    return min(
        MAX_MODEL_LEN,
        max(256, MAX_MODEL_LEN - max_tokens - SUMMARY_INPUT_RESERVE),
    )


async def _count_tokens(
    client: httpx.AsyncClient, vllm_url: str, model: str, text: str
) -> int:
    """What vLLM will charge for `text`. Asks vLLM; falls back to the
    pessimistic ceiling, never to an optimistic guess.
    """
    if not text:
        return 0
    try:
        r = await client.post(
            f"{vllm_url}/tokenize",
            json={"model": model, "prompt": text},
            timeout=httpx.Timeout(connect=2.0, read=10.0, write=10.0, pool=2.0),
        )
        if getattr(r, "status_code", 200) == 200:
            n = (r.json() or {}).get("count")
            if isinstance(n, (int, float)):
                return int(n)
        if logsetup.log_once("summarizer.tokenize.http"):
            logger.warning(
                f"/tokenize did not answer with a count (status "
                f"{getattr(r, 'status_code', '?')}); rollup input is being "
                f"budgeted at {_WORST_TOKENS_PER_CHAR} tokens/char instead, so "
                f"rollups will over-split until this recovers"
            )
    except Exception as e:
        if logsetup.log_once("summarizer.tokenize.error"):
            logger.warning(
                f"/tokenize unreachable ({type(e).__name__}: {e}); rollup input "
                f"is being budgeted at {_WORST_TOKENS_PER_CHAR} tokens/char "
                f"instead, so rollups will over-split until this recovers"
            )
    return _pessimistic_tokens(text)


async def _truncate_to_budget(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    piece: str,
    measured: int,
    budget: int,
) -> str:
    """Cut a single piece that does not fit even on its own.

    main.summarize's `_chunk_to_budget` deliberately does NOT do this — it
    gives an oversized turn its own batch, lets the call fail, and degrades.
    That is the right trade there, because compaction degrading means
    forwarding the original messages and the user still gets a reply.

    Here the same trade is wrong. A rollup that cannot fit its input does not
    degrade, it LATCHES: the watermark never advances and the hierarchy is dead
    from that turn on (A1). Losing the tail of one enormous turn is cheaper
    than losing every summary after it, so this truncates, says so in the text
    it hands the model, and the caller logs it.

    WHY THIS RE-MEASURES INSTEAD OF CUTTING BY PROPORTION
    -----------------------------------------------------
    The first version of this scaled characters by the token ratio —
    `len(piece) * budget / measured * 0.9` — and trusted the result. That is
    the A4 unit error committed inside A1's own fix: it assumes tokens are
    spread evenly across the characters, and the entire subject of
    INCIDENT_2026-08-28 is that they are not. A turn whose DENSE part comes
    first (a box-drawing table, then prose) prices its head far above its
    average, so a proportional cut keeps a prefix that still overflows.

    Measured, on a turn of 10k box-drawing characters followed by 90k of
    prose against a 5,644-token budget: the proportional cut kept 17,515
    characters costing 20,751 tokens — 3.7x over — the request was refused,
    and the watermark stayed at 0. The latch, straight back, by the one code
    path that exists to prevent it.

    So the cut is measured, not assumed: shrink, ask, repeat. The final
    backstop is arithmetic rather than another guess — at most
    `budget / _WORST_TOKENS_PER_CHAR` characters cannot exceed `budget` tokens
    unless the content beats the worst density this project has ever measured.
    """
    if measured <= 0:
        return piece

    text, m = piece, measured
    # Three rounds is enough for any density profile to converge from above,
    # and each round only costs one /tokenize on an already-rare path.
    for _ in range(3):
        # 0.9 for the framing and separators the measurement does not include.
        keep = max(1, int(len(text) * (budget / m) * 0.9))
        if keep >= len(text):
            break
        text = text[:keep]
        m = await _count_tokens(client, vllm_url, model, text + _TRUNCATION_NOTE)
        if m <= budget:
            return text + _TRUNCATION_NOTE

    # Still over (or /tokenize is down and every answer is the pessimistic
    # ceiling): fall back to the cut that cannot be wrong about density.
    hard = max(1, int(budget / _WORST_TOKENS_PER_CHAR) - len(_TRUNCATION_NOTE))
    return text[:hard] + _TRUNCATION_NOTE


async def _batch_to_budget(
    conv_id: str,
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    pieces: list[str],
    budget: int,
) -> list[list[str]]:
    """Split `pieces` into consecutive batches that each fit `budget` tokens.

    Cost note: the whole-body check first. When everything fits even at the
    pessimistic ceiling there is nothing to decide and the split costs ZERO
    /tokenize calls. Be honest about how far that reaches, though: the
    short-circuit is `len * 2.0 <= budget`, so at the shipped defaults
    (budget 30,220) it covers bodies under ~15,110 CHARACTERS — a short
    conversation, not every ordinary one. A chatty 20-turn slice will measure
    its pieces, one call each. That is one cheap localhost round-trip per turn
    on the background tail, and it is the price of not guessing; the ceiling is
    set for the worst content this project has measured, not for prose.
    """
    joined = "\n\n".join(pieces)
    if _pessimistic_tokens(joined) <= budget:
        return [pieces]

    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for p in pieces:
        t = await _count_tokens(client, vllm_url, model, p)
        if t > budget:
            # Does not fit on its own. Flush what we have, then truncate it.
            if current:
                batches.append(current)
                current, current_tokens = [], 0
            logger.warning(
                f"conv={conv_id}: a single turn measures {t} tokens against a "
                f"{budget}-token summarization budget; it has been truncated "
                f"for the rollup so the hierarchy keeps advancing — the stored "
                f"summary covers only the beginning of that turn"
            )
            batches.append([
                await _truncate_to_budget(client, vllm_url, model, p, t, budget)
            ])
            continue
        if current and current_tokens + t > budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(p)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


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

# Used only by the reduce step, when one tier's input was too large for a
# single call and had to be summarized in parts. The parts are consecutive
# slices of ONE unit (one scene, one chapter, one theme), so the instruction is
# "fold", not "summarize again" — a second summarization pass is exactly the
# summary-of-summary degradation this module's tiering exists to avoid.
_PROMPT_REDUCE = """The following are consecutive partial summaries of a single stretch of one conversation, in order. Merge them into one continuous summary of that stretch. Keep every name, decision, concrete detail and numeric value that appears in any part; drop only repetition between the parts. Do not add framing, headings, or commentary. Output the merged summary only."""


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
    data = r.json() or {}
    choices = data.get("choices") or []
    if not choices:
        # A 200 with no choices (an error-shaped body, most often) used to
        # surface as a bare IndexError inside maybe_rollup's blanket handler —
        # a stack trace per turn that named the wrong thing. Say what happened.
        # Same guard main._summarize_once already carries.
        raise ValueError(
            f"vLLM returned no choices for a rollup summarize: {str(data)[:200]}"
        )
    return ((choices[0].get("message") or {}).get("content") or "").strip()


async def _summarize_pieces(
    conv_id: str,
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    system_prompt: str,
    pieces: list[str],
    max_tokens: int,
) -> str:
    """Summarize `pieces` into one string, MAP-REDUCE so the request can never
    itself exceed the model's context window (v3.1 A1).

    Every tier goes through here. L1's input is user turns, which are bounded
    by nothing at all; L3's input is every L2 chapter ever written, which grows
    without limit as the conversation does. L2 is the only tier whose input is
    bounded by construction (L2_CHUNK_SIZE chunks, each capped at
    L1_MAX_TOKENS output) and it uses the same path anyway, because a tier that
    is safe today by arithmetic nobody re-checks is how this module got here.

    The batch count stays at 1 for any normal conversation, so the common case
    is byte-for-byte the old single call.
    """
    pieces = [p for p in pieces if p and p.strip()]
    if not pieces:
        return ""
    budget = _input_budget(max_tokens)
    batches = await _batch_to_budget(
        conv_id, client, vllm_url, model, pieces, budget
    )

    async def _call(prompt: str, batch: list[str]) -> str:
        return await _llm_summarize(
            client, vllm_url, model, prompt, "\n\n".join(batch), max_tokens
        )

    if len(batches) == 1:
        return await _call(system_prompt, batches[0])

    logger.info(
        f"conv={conv_id}: rollup input exceeds the {budget}-token budget — "
        f"map-reduce over {len(batches)} batches"
    )
    # Map. Concurrent because vLLM batches fine, bounded by a small semaphore
    # so one huge backlog can't monopolize the engine — the tail already holds
    # conv_lock for the duration of this call.
    sem = asyncio.Semaphore(4)

    async def _bounded(prompt: str, batch: list[str]) -> str:
        async with sem:
            return await _call(prompt, batch)

    parts = [
        p for p in await asyncio.gather(*(_bounded(system_prompt, b) for b in batches))
        if p
    ]
    if not parts:
        return ""

    # Reduce, in bounded rounds, never handing a call more than it can take.
    # If folding can make no further progress the parts are concatenated: a
    # longer chunk than the tier intended, but a complete one, and the rollup
    # still advances the watermark. Silence would not.
    rounds = 0
    while len(parts) > 1 and rounds < 3:
        rounds += 1
        groups = await _batch_to_budget(
            conv_id, client, vllm_url, model, parts, budget
        )
        if all(len(g) == 1 for g in groups):
            break
        try:
            folded = await asyncio.gather(
                *(_bounded(_PROMPT_REDUCE, g) for g in groups)
            )
        except Exception as e:
            logger.warning(
                f"conv={conv_id}: rollup reduce round {rounds} failed, keeping "
                f"the partial summaries concatenated: {e}"
            )
            break
        parts = [p for p in folded if p] or parts
    return "\n\n".join(parts)


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
    # One piece per turn, so an oversized slice can be split rather than sent
    # whole and refused. The chunk still COVERS first_turn..last_turn either
    # way — the turn range is the contract the watermark and the L2 rollup
    # depend on, and splitting the request must not change it (v3.1 A1).
    pieces = _turn_pieces(messages, first_turn, last_turn)
    if not any(p.strip() for p in pieces):
        return False
    text = await _summarize_pieces(
        conv_id, client, vllm_url, model, _PROMPT_L1, pieces, L1_MAX_TOKENS
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
    pieces = [
        f"--- scene (turns {c['first_turn']}-{c['last_turn']}) ---\n{c['text']}"
        for c in chunks
    ]
    text = await _summarize_pieces(
        conv_id, client, vllm_url, model, _PROMPT_L2, pieces, L2_MAX_TOKENS
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
    # ALL chapters, and nothing trims l2 (MEMORY_REVIEW S-1). So this input
    # grows without bound as the conversation does: at L2_MAX_TOKENS=1200 a
    # 25-chapter conversation is already a ~30,000-token request. Slower than
    # L1's overflow, same shape, same permanent-death ending — L3 is the tier
    # that never refires once it fails, because _needs_l3_rollup keeps
    # returning True and the call keeps being refused (v3.1 A1).
    pieces = [
        f"--- chapter (turns {c['first_turn']}-{c['last_turn']}) ---\n{c['text']}"
        for c in l2
    ]
    text = await _summarize_pieces(
        conv_id, client, vllm_url, model, _PROMPT_L3, pieces, L3_MAX_TOKENS
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
