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
- Total injected size stays bounded, the same way at every tier: l1 drains
  into l2 once L2_CHUNK_SIZE chunks accumulate (_do_l2_rollup), and l2 now
  drains into l3 the same way once L3_CHUNK_SIZE chapters accumulate
  (_do_l3_rollup) — MEMORY_REVIEW S-1/S-6's fix. Before this, _do_l3_rollup
  refreshed l3 but kept every l2 chapter it had just folded in, so l2 grew
  by one chapter per L2_CHUNK_SIZE*L1_CHUNK_SIZE turns for the life of the
  conversation and so did the L3 input, the state file, and the injected
  block. Measured on a synthetic 240-turn run at this module's test
  thresholds (L1=4/L2=3/L3=2): len(l2) reached 20 and was still climbing,
  never once trimmed. With the drain, l2 is bounded to at most
  L3_CHUNK_SIZE-1 chapters, the same shape l1's bound already had. On top of
  that, format_summary_block enforces its own COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS
  ceiling (default 12,000 - above the
  tiers' own worst-case product of 11,300, so it backstops misconfiguration
  instead of firing in normal operation) as a
  backstop against the tiers' bounds ever being right in theory and wrong in
  practice — see its docstring for how it chooses what to keep when they
  don't fit.

Storage (one JSON per conv):
    /data/openwebui/compactor/summaries/<conv_id>.json
    {
      "conv_id": "...",
      "updated_at": "ISO",
      "l1": [{"text": "...", "first_turn": 1, "last_turn": 20}, ...],
      "l2": [{"text": "...", "first_turn": 1, "last_turn": 200}, ...],
      "l3": {"text": "...", "first_turn": 1, "last_turn": 1000} | null,
      "last_summarized_turn": 20,  # highest turn covered by any L1 chunk
      "turns_seen": 44,            # monotonic conversational position (v3.1.4)
      "tail_fp": ["ab12…", ...]    # content anchor for the last few turns
    }

`turns_seen` and `tail_fp` are the compactor's OWN answer to "how far has
this conversation got", replacing the client's `len(messages)`. See
_observed_position for why the client's array cannot be that authority.

Lifecycle:
  request time (sync, cheap): load_state → format injection block from
    existing L3+L2+(unrolled L1s) → inject as system message.
  post-response (async, may do LLM calls): maybe_rollup checks thresholds
    and triggers L1 / L2 / L3 rollups if enough new material accumulated.

All operations degrade to safe no-ops on failure — chat never breaks because
the summarizer hit a problem.
"""

import asyncio
import hashlib
import logging
import os
import unicodedata
from datetime import datetime, timezone
from typing import Any

import httpx

import logsetup
import textclean
import tokens
import tokenhealth
from envcfg import env_float, env_int
from memory import (
    atomic_write_json,
    conv_lock,
    read_json_strict,
    storage_root,
    summary_archive_path,
)

logger = logging.getLogger("compactor.summarizer")


# ---------------------------------------------------------------------------
# Configuration (env-overridable, sensible defaults)
# ---------------------------------------------------------------------------

L1_CHUNK_SIZE = env_int("COMPACTOR_L1_CHUNK_SIZE", 20)
L2_CHUNK_SIZE = env_int("COMPACTOR_L2_CHUNK_SIZE", 10)
L3_CHUNK_SIZE = env_int("COMPACTOR_L3_CHUNK_SIZE", 5)

# Per-tier token budget for the LLM's output (input tokens depend on how
# much we're summarizing). L3 is largest because it must represent the
# whole conversation; L1 is smallest because each chunk is one "scene."
L1_MAX_TOKENS = env_int("COMPACTOR_L1_MAX_TOKENS", 500)
L2_MAX_TOKENS = env_int("COMPACTOR_L2_MAX_TOKENS", 1200)
L3_MAX_TOKENS = env_int("COMPACTOR_L3_MAX_TOKENS", 2000)

# Same env var and default main.py reads for its own /tokenize call sites
# (main.py:693, TOKENIZE_WARN_INTERVAL_S) — deliberately, not independently
# tuned: an operator setting this once should govern every /tokenize
# dependency in the process, not just the ones main.py happens to own.
TOKENIZE_WARN_INTERVAL_S = env_float("COMPACTOR_TOKENIZE_WARN_INTERVAL_S", 300)

# Hard ceiling on the rendered injection block (see format_summary_block).
# 5000 is the figure this module's own docstring always claimed as the
# intended worst case (L3 + latest L2 + a handful of unrolled L1 chunks) —
# this makes it a real, enforced number instead of an unverified comment
# (MEMORY_REVIEW S-1/S-6).
SUMMARY_BLOCK_MAX_TOKENS = env_int("COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS", 12000)

# The model's context window, and the slack left inside it for a
# summarization call's system prompt, wrapper text and chat-template framing.
# Same env vars main.py reads, deliberately: the two summarization paths must
# not be tunable apart, and a rollup that budgets against a different window
# than the request path is the same defect in a second place.
MAX_MODEL_LEN = env_int("MAX_MODEL_LEN", 32768)
SUMMARY_INPUT_RESERVE = env_int("COMPACTOR_SUMMARY_INPUT_RESERVE", 2048)

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
        "turns_seen": 0,
        "tail_fp": [],
        # v3.1.7 (R18). The anchor alone cannot tell "the client re-sent the
        # same window" from "the client appended one more exchange" when the
        # conversation's tail REPEATS — two consecutive degenerate replies
        # redact to byte-identical placeholders, and "ok"/"Sure." twice does
        # it without any redaction. These two say whether the window itself
        # changed: under a cap the first turn slides out on every exchange,
        # and an unchanged window keeps both its head and its length.
        "head_fp": "",
        "window_turns": 0,
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
    data = read_json_strict(summary_path(conv_id), default=None, expect=dict)
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
    # v3.1.4. Absent on every file written before this release, which is why
    # _observed_position seeds from last_summarized_turn rather than from 0:
    # seeding at 0 would make the first post-upgrade turn look like a brand-new
    # conversation and re-summarize turns 1-20 of a 651-turn history.
    if isinstance(data.get("turns_seen"), int):
        state["turns_seen"] = data["turns_seen"]
    if isinstance(data.get("tail_fp"), list):
        state["tail_fp"] = [x for x in data["tail_fp"] if isinstance(x, str)]
    # v3.1.7. Absent on every file written before this release. Missing, they
    # read as "the window is not known to be unchanged", which is the side
    # that duplicates a summary rather than the side that stalls the position.
    if isinstance(data.get("head_fp"), str):
        state["head_fp"] = data["head_fp"]
    if isinstance(data.get("window_turns"), int):
        state["window_turns"] = data["window_turns"]
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

# v3.1.5 — this block's authority is over what HAPPENED, and nothing else.
# "use them for continuity" was doing double duty: continuity of events is
# wanted, continuity of phrasing is not, and the header did not distinguish
# them. Naming it as background the model already holds also discourages
# recapping it back at the user, which is its own species of repetition. See
# persona.py's _PERSONA_BLOCK_HEADER for the division of labour between the
# four injected blocks.
_BLOCK_HEADER = (
    "[Hierarchical summary of earlier portions of this conversation, ordered "
    "by recency — background you already hold, for continuity of events. "
    "Older summaries are denser; the L3 line (if present) is the "
    "whole-conversation theme.]"
)


def _estimate_block_tokens(text: str) -> int:
    """Cheap, HTTP-free CEILING estimate of what vLLM will charge for `text`.

    format_summary_block is a synchronous, no-I/O read on the request hot
    path (see main.py's comment at its call site: "no LLM call on the hot
    path"), so it cannot ask /tokenize the way _count_tokens does — a
    blocking POST has no place inside a sync function called from an async
    request handler. A flat chars/4 estimate is the thing
    INCIDENT_2026-08-28 is about: it read up to 7.74x low on this model's
    decoration characters. So this uses the same split-by-character-class
    ceiling retrieval.py's `_estimate_tokens` (A4) already validated under
    the identical constraint (no tokenizer, no HTTP, called synchronously):
    ASCII priced at chars/4 (the measured prose density on this deployment),
    non-ASCII priced at one token per UTF-8 byte — a byte-level BPE cannot
    cost more than that per byte, so it can only over-count decoration, never
    under-count it the way a flat multiplier does.

    Duplicated here rather than imported from retrieval.py: this module's
    few sync budget primitives (_pessimistic_tokens and this) stay together,
    and summarizer.py does not otherwise depend on retrieval.py's private
    helpers.
    NON-ASCII LETTERS ARE PRICED PER CHARACTER, NOT PER BYTE, and that
    distinction is the whole point. The per-byte ceiling was written for
    DECORATION - box-drawing runs, where it is roughly right. Applied to
    natural non-Latin script it is wildly pessimistic, because those
    characters are 2-3 UTF-8 bytes each and tekken encodes them far better
    than one token per byte. Measured in the production image against the
    real tekken vocabulary, shipped estimator vs ground truth:

        prose       1848 chars   real   364   shipped   462   1.27x
        greek       1520 chars   real  1164   shipped  2720   2.34x
        hebrew      1120 chars   real  1124   shipped  2030   1.81x
        cjk         1000 chars   real   703   shipped  3000   4.27x
        decoration   400 chars   real   803   shipped  1200   1.49x

    That over-pricing is not academic here: this user quotes scripture, and
    a summary block of 2,823 REAL tokens - inside both this cap and the
    accurate /tokenize budget downstream - priced out at 13,282 and was
    dropped in full. She would have silently lost her entire summary memory
    on exactly the conversations she cares most about, with the per-turn log
    line still reporting the chunks as injected.

    One token per CHARACTER is still a true ceiling for every script
    measured (Greek 0.77 tokens/char, Hebrew 1.00, CJK 0.70), while cutting
    the bias to 1.10-1.42x. Decoration keeps the per-byte ceiling, which is
    what it was for.
    """
    exact = None
    try:
        if tokens.is_available():
            exact = tokens.count([{"role": "user", "content": text}])
    except Exception:
        exact = None
    if exact is not None:
        return exact

    # Fallback only. No single multiplier fits: measured tokens-per-character
    # for non-ASCII letters ranges from 0.40 (Russian) to 1.16 (Hebrew with
    # niqqud), and emoji land just above one token per BYTE. 1.25 per
    # letter/mark is a ceiling on every script measured; combining marks are
    # counted with letters because Hebrew points are category Mn, not
    # isalpha(), and pricing them as decoration is what made the first
    # attempt at this fix under-count Hebrew by 11%.
    ascii_n = script_chars = decor_bytes = 0
    for c in text:
        if ord(c) < 128:
            ascii_n += 1
        elif unicodedata.category(c)[0] in ("L", "M"):
            script_chars += 1
        else:
            decor_bytes += len(c.encode("utf-8", "surrogatepass"))
    # decor gets 5% headroom: emoji measured at 1.001 tokens/byte, i.e. the
    # bare per-byte rule is not quite a ceiling for them.
    return ascii_n // 4 + int(script_chars * 1.25) + int(decor_bytes * 1.05) + 1


def _summary_line(kind: str, chunk: dict) -> tuple[str, str]:
    """The (header, body) pair format_summary_block renders for one chunk —
    factored out so the cost estimate and the render use IDENTICAL text."""
    header = (
        f"\n--- {kind} (turns {chunk.get('first_turn', '?')}-"
        f"{chunk.get('last_turn', '?')}) ---"
    )
    return header, chunk.get("text", "")


def format_summary_block(state: dict, max_tokens: int | None = None) -> str | None:
    """Render the current summary stack into a single system-message body.
    Returns None if there's nothing to inject.

    Order in the rendered block (most-general → most-specific):
      1. L3 (whole-conversation theme), if any
      2. L2 chapters in chronological order
      3. L1 chunks in chronological order
    The most-recent L1s are what the model needs most for continuity, so
    they come last (right before the recent raw turns will appear in the
    final message list).

    The whole block is capped at SUMMARY_BLOCK_MAX_TOKENS (MEMORY_REVIEW
    S-1/S-6's other half): l1 and l2 are now bounded by construction (see
    _do_l2_rollup / _do_l3_rollup) — but that bound is LARGER than the cap
    was originally set to, so "never reached" was false as shipped. At
    defaults the bounded state's own capacity is 9*L1_MAX + 4*L2_MAX +
    L3_MAX = 11,300 tokens; against the original 5,000 cap it fired on every
    request above roughly 45% tier fill and dropped every L2 chapter above
    75%. Since L1 is selected before L2, chapters got only what L3 and all
    of L1 left over — a tier that could be created, never injected, and then
    consumed by the next L3 refresh. The default is now 12,000, above that
    capacity, so the cap is what it was meant to be: a backstop against
    misconfiguration, not a routine amputation.

    "Bounded by construction" is still exactly the kind of claim
    this module has been burned by before (_summarize_pieces's own docstring:
    "a tier that is safe today by arithmetic nobody re-checks is how this
    module got here"), and L1_CHUNK_SIZE/L2_CHUNK_SIZE/L3_CHUNK_SIZE/
    *_MAX_TOKENS are five independently-configurable env vars whose product
    is what actually bounds this block. This is the backstop that holds even
    if that arithmetic is ever wrong, or configured wrong, again.

    What gets dropped when it doesn't fit, in priority order (highest first):
      1. L3 — a single object, already capped at L3_MAX_TOKENS, and the
         cheapest way to keep the whole-conversation throughline; almost
         never the thing that has to give.
      2. L1 chunks, NEWEST first — "the most-recent L1s are what the model
         needs most for continuity" (above) makes them the second-highest
         priority to keep, so a squeeze drops the OLDEST scenes first.
      3. L2 chapters, NEWEST first — the middle tier: by the time a chapter
         is old enough to be first in this list, an L3 refresh has usually
         already folded it into the theme, so it is the most redundant
         content to lose and goes first.
    Selection order and render order differ on purpose: what to KEEP is
    decided newest-first (recency is what makes content worth keeping under
    a squeeze); what gets SENT stays general-to-specific (L3, L2, L1) either
    way, because that is what the model reads best regardless of how much of
    each tier survived the cut.
    """
    has_l3 = state.get("l3") is not None
    l2 = state.get("l2") or []
    l1 = state.get("l1") or []
    if not (has_l3 or l2 or l1):
        return None

    # The caller's cap wins when it is tighter. SUMMARY_BLOCK_MAX_TOKENS
    # alone (12,000) is larger than main.py's whole injection budget (8,192
    # at production config, shared with persona, facts and retrieval), so
    # capping only here meant _bound_injected_blocks downstream dropped
    # WHOLE layers to make room: measured, facts stopped reaching the model
    # from ~50% tier fill and everything but persona was gone at ~70%. The
    # caller passes its real share so the trimming happens HERE,
    # newest-kept and graceful, instead of down there, whole-layer and
    # blind.
    budget = (
        min(SUMMARY_BLOCK_MAX_TOKENS, max_tokens)
        if max_tokens is not None
        else SUMMARY_BLOCK_MAX_TOKENS
    )
    used = _estimate_block_tokens(_BLOCK_HEADER)

    l3_line: tuple[str, str] | None = None
    if has_l3:
        header, body = _summary_line("conversation-wide theme", state["l3"])
        cost = _estimate_block_tokens(header) + _estimate_block_tokens(body)
        if used + cost <= budget:
            l3_line = (header, body)
            used += cost

    l1_keep = [False] * len(l1)
    for i in range(len(l1) - 1, -1, -1):
        header, body = _summary_line("scene", l1[i])
        cost = _estimate_block_tokens(header) + _estimate_block_tokens(body)
        if used + cost > budget:
            break
        l1_keep[i] = True
        used += cost

    l2_keep = [False] * len(l2)
    for i in range(len(l2) - 1, -1, -1):
        header, body = _summary_line("chapter", l2[i])
        cost = _estimate_block_tokens(header) + _estimate_block_tokens(body)
        if used + cost > budget:
            break
        l2_keep[i] = True
        used += cost

    dropped_l3 = 1 if (has_l3 and l3_line is None) else 0
    dropped_l1 = l1_keep.count(False)
    dropped_l2 = l2_keep.count(False)
    if dropped_l3 or dropped_l1 or dropped_l2:
        # WARNING, not INFO: main.py logs the analogous event ("memory the
        # user believes the assistant has and the model is not going to
        # see") at WARNING, and this is the same event one layer down.
        logger.warning(
            f"summary block: dropped {dropped_l3 + dropped_l1 + dropped_l2} "
            f"tier item(s) to fit the {budget}-token block budget "
            f"(COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS) — kept "
            f"{'L3, ' if l3_line else ('no L3, ' if has_l3 else '')}"
            f"{len(l2) - dropped_l2}/{len(l2)} chapter(s), "
            f"{len(l1) - dropped_l1}/{len(l1)} scene(s)"
        )

    lines = [_BLOCK_HEADER]
    if l3_line:
        lines.extend(l3_line)
    for i, keep in enumerate(l2_keep):
        if keep:
            header, body = _summary_line("chapter", l2[i])
            lines.append(header)
            lines.append(body)
    for i, keep in enumerate(l1_keep):
        if keep:
            header, body = _summary_line("scene", l1[i])
            lines.append(header)
            lines.append(body)
    if len(lines) == 1:
        # Everything was dropped (an absurdly small budget, or a single
        # chunk larger than the whole block cap). Consistent with the
        # "nothing to inject" contract rather than sending a bare header.
        return None
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rollup trigger detection
# ---------------------------------------------------------------------------

def _needs_l1_rollup(state: dict, current_turn_count: int) -> bool:
    """True if there are >= L1_CHUNK_SIZE turns past last_summarized_turn.

    `current_turn_count` is the conversation's POSITION (_observed_position),
    not `len(messages)`. Handed the client's array length instead, this gate
    latches shut forever the moment the client starts sending a bounded
    window — see _observed_position.
    """
    last = state.get("last_summarized_turn", 0)
    return (current_turn_count - last) >= L1_CHUNK_SIZE


def _needs_l2_rollup(state: dict) -> bool:
    """True if accumulated L1 chunks have crossed the L2 threshold."""
    return len(state.get("l1", [])) >= L2_CHUNK_SIZE


def _needs_l3_rollup(state: dict) -> bool:
    """True if enough L2 chapters exist AND L3 does not already cover them.

    The threshold alone used to be a standing condition, not an event
    (MEMORY_REVIEW S-2): `_do_l3_rollup` used to keep the L2 list after
    folding it into L3, unlike L1→L2 which drops what it consumed, so from
    the L3_CHUNK_SIZE-th chapter onward `len(l2) >= L3_CHUNK_SIZE` never
    cleared. Every turn then spent one L3_MAX_TOKENS LLM call re-paraphrasing
    the same chapters, and kept `needs_rollup` True, so maybe_rollup's early
    exit never fired either.

    `_do_l3_rollup` now drops the L2 chapters it consumes on success, the
    same contract L1→L2 already had (MEMORY_REVIEW S-1/S-6), so after a
    successful refresh `len(l2)` drops below `L3_CHUNK_SIZE` and this
    function's first check already returns False — the span comparison below
    now mainly guards the case a refresh has NOT yet happened (l2 sitting at
    or above threshold with a stale or absent l3, e.g. after a prior L3
    failure left l2 non-empty): refresh only when the chapters on hand are
    not what the recorded l3 (if any) already covers, rather than on the bare
    threshold alone.
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
# Conversational position (v3.1.4)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# Until v3.1.4 the rollup gate compared `last_summarized_turn` against
# `len([m for m in messages if m["role"] != "system"])` — a measurement of the
# CLIENT'S ARRAY. pipelines/conversation_id_header.py's `max_turns` valve caps
# that array at a constant (100 is the documented starting value, ~50
# exchanges), and a constant is fatal to a gate expressed as a difference:
#
#   watermark 651, window pinned at 100 -> the old _reconcile_watermark pulled
#   the watermark down to 100 once, and from then on (observed - watermark) is
#   0 on EVERY subsequent request. _needs_l1_rollup is False forever, so no L1
#   chunk is ever produced, so no L2, so no L3. Simulated against this module
#   with the window pinned: the hierarchy freezes permanently, and the one
#   warning that says so is logsetup.log_once — one line per process, then
#   silence.
#
# That matters more than it sounds: with request-path compaction latched off,
# the L1/L2/L3 hierarchy is the one memory layer keeping pace in production
# (lastturn=651 against msgs=664 on 2026-09-01).
#
# THE FIX IS retrieval._next_turn_index's, APPLIED TO THE SUMMARIZER
# ------------------------------------------------------------------
# retrieval.py:294 solved exactly this for the episodic index: allocate from
# the STORE's own maximum rather than from the request, "because a deletion, an
# edit, a branch switch or a bounded client window all shrink len(messages)+1",
# and let the request only ever push the sequence FORWARD. `turns_seen` is that
# counter for the summarizer: persisted per conversation, monotonic, and owned
# by the compactor.
#
# The one thing a counter cannot do on its own is notice that new material
# arrived while the array length stayed put. That is what `tail_fp` is for —
# the same content-addressed identity D1 gave episodic rows, used here to align
# this request's window against the last one we saw.

# How many trailing turns the anchor records. Four, not one: a single
# fingerprint is enough to detect that SOMETHING changed but not how much, and
# the prefix walk in _align_new_turns needs the older elements to survive a
# regeneration (which rewrites the newest turn and nothing else).
_ANCHOR_TURNS = 4

# What to assume advanced when the anchor cannot be found anywhere in the
# window. main.py's tail calls maybe_rollup exactly once per exchange, with the
# user turn and the assistant turn it just produced, so one exchange is the
# real per-call rate; assuming it degrades the mechanism to "count the calls",
# which is right for the live path and idempotent for the admin-compact loop
# (there the window is unchanged, so the anchor matches and this never runs).
_ASSUMED_NEW_TURNS = 2

# How many trailing turns are fingerprinted per call (v3.1.7, R17). The whole
# history used to be hashed on the event loop inside conv_lock, on EVERY turn:
# measured 47 ms per call at 660 turns of her real reply length, linear in
# history and growing for the life of the conversation. The alignment below
# only ever needs the anchor plus whatever arrived after it, and main.py calls
# maybe_rollup once per exchange, so 64 turns is 32 exchanges of slack against
# a per-call rate of one. Beyond that the anchor falls off the end and the
# call degrades to _ASSUMED_NEW_TURNS with the warning that already exists —
# the same degradation as an anchor the client never echoed back.
_FINGERPRINT_TAIL_TURNS = 64


def _image_only_marker(m: dict) -> str:
    """What the EPISODIC STORE will remember an image-only turn as, or "".

    The twin of main._memorable_user_text / main._message_image_count, and it
    has to stay byte-identical to them (R15). `_message_text` is "" for a
    content list with no text part, while the store holds "[shared 1 image]"
    — so the live path fingerprinted one string and /admin/compact's
    reconstruction fingerprinted another, and a mismatch in anchor[0] defeats
    every prefix in _align_candidates. The position then inflated by 2 per
    rollup and stayed inflated, so window_offset subtracted 2 forever and two
    turns were summarized twice.

    It cannot import main: main imports summarizer, so the constant is
    duplicated here rather than shared. If main's marker changes, change this
    one in the same commit — the fingerprints must agree.
    """
    content = m.get("content")
    if not isinstance(content, list):
        return ""
    n = 0
    for c in content:
        if not isinstance(c, dict):
            continue
        if c.get("type") in ("image_url", "image", "input_image") or "image_url" in c:
            n += 1
    if not n:
        return ""
    return f"[shared {n} image{'s' if n > 1 else ''}]"


def _turn_fingerprints(messages: list[dict]) -> list[str]:
    """One short content hash per observed turn, oldest first, system skipped.

    Whitespace-normalized before hashing. The anchor is compared across two
    different HTTP requests — what the compactor appended after streaming a
    reply on turn N, against what OpenWebUI reads back out of its own database
    and re-sends on turn N+1 — and a re-flowed trailing newline must not read
    as a different turn. Truncated to 16 hex chars: at ~10^4 turns per
    conversation the collision probability is ~10^-11, and the whole anchor
    lives in a state file that is read and written on every rollup.

    A turn with images and no text is fingerprinted as the episodic store
    will remember it (see _image_only_marker), not as the empty string its
    request shape reduces to. The marker never leaves this function — it is
    hashed, never stored and never summarized — so nothing can extract it as
    a fact.
    """
    out: list[str] = []
    for m in messages:
        if m.get("role") == "system":
            continue
        text = " ".join(_message_text(m).split())
        if not text:
            text = _image_only_marker(m)
        payload = f"{m.get('role', 'unknown')}\x00{text}"
        out.append(
            hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        )
    return out


def _align_candidates(anchor: list[str], fps: list[str]) -> list[int]:
    """Every "how many turns at the END of `fps` are new" that the anchor
    supports, sorted ascending and de-duplicated. Empty if it cannot be told.

    `anchor` is oldest-first and ends at the previous position, so a match on
    the whole anchor ending at window slot j means slot j IS the previous
    position and everything after it is new.

    PREFIXES ARE TRIED, and that is what makes a regeneration cost nothing.
    Regenerating the last reply rewrites the newest turn and leaves the three
    before it alone: the full anchor is nowhere in the new window, the
    3-element prefix is, ending one slot earlier — so `new` comes out 0, which
    is the truth (a replaced turn is not a new turn). Without the prefix walk
    that reads as one fresh exchange, and a position that is 2 ahead of
    reality shifts every later chunk boundary by 2, which is a 2-turn HOLE in
    what the hierarchy summarizes.

    EVERY match is scored, not just the first one found. Until v3.1.7 the walk
    returned the first hit of the LONGEST matching prefix, which is a
    different rule from the one the docstring claimed ("the latest
    occurrence"): anchor [A,B,C,D] against [A,B,C,D,X,Y,A,B,C] returned 5
    because the 4-element prefix matches early, while the 3-element prefix
    matches at the very end for 0 (R20). Returning the candidate list lets the
    caller apply the recency rule uniformly AND see when the evidence is
    ambiguous, which is the only way to tell a repeating tail (R18) from a
    window that genuinely did not move.
    """
    n = len(fps)
    out: set[int] = set()
    for m in range(len(anchor), 0, -1):
        prefix = anchor[:m]
        for j in range(n, m - 1, -1):
            if fps[j - m:j] == prefix:
                # The prefix ends (len(anchor) - m) turns before the previous
                # position, so those turns are already accounted for.
                out.add(max(0, n - j - (len(anchor) - m)))
    return sorted(out)


def _align_new_turns(anchor: list[str], fps: list[str]) -> int | None:
    """The SMALLEST advance the anchor supports, or None if it supports none.

    Smallest, because direction matters: over-counting drops turns for good,
    under-counting merely summarizes some of them twice. A short repeated turn
    ("ok") that collides with an older one must land on the side that
    duplicates rather than the side that loses.

    _observed_position uses _align_candidates directly — it holds the extra
    evidence (the window's own length and head) needed to decide whether a
    zero-advance answer is credible when the tail repeats. This function is
    that rule with no extra evidence, and is what the unit test pins.
    """
    cands = _align_candidates(anchor, fps)
    return cands[0] if cands else None


def _highest_chunk_turn(state: dict) -> int:
    """The furthest turn any stored chunk claims to cover, across all tiers.

    The chunk list is the RECORD of what was summarized; `last_summarized_turn`
    is a pointer derived from it. When the two disagree the chunks are the
    survivors — they carry text, the pointer carries none — so this is a lower
    bound on the conversation's position that no watermark edit can erase.
    """
    highest = 0
    for c in list(state.get("l1") or []) + list(state.get("l2") or []):
        lt = c.get("last_turn") if isinstance(c, dict) else None
        if isinstance(lt, int) and lt > highest:
            highest = lt
    l3 = state.get("l3")
    if isinstance(l3, dict) and isinstance(l3.get("last_turn"), int):
        highest = max(highest, l3["last_turn"])
    return highest


def _recorded_position(state: dict) -> int:
    """The furthest turn this conversation is KNOWN to have reached.

    Three sources, and the third is the v3.1.7 repair (R12). `turns_seen` is
    the compactor's own counter; `last_summarized_turn` covers a state file
    written before that counter existed; and the CHUNK LABELS cover a state
    file whose watermark was pulled DOWN by the old _reconcile_watermark
    (S-5 / REMEDIATION F14) while the chunks it had already written kept their
    real labels. Seeding from the first two alone restarted the position at
    the cap on the first post-upgrade request, which is what made every new
    L1 chunk collide with an old label and be discarded.
    """
    return max(
        int(state.get("turns_seen") or 0),
        int(state.get("last_summarized_turn") or 0),
        _highest_chunk_turn(state),
    )


def _repair_watermark_below_chunks(conv_id: str, state: dict) -> bool:
    """Raise a watermark that sits below the chunks it is supposed to track.

    Every writer in this module advances `last_summarized_turn` to a chunk's
    `last_turn` in the same state dict it appends the chunk to, so the
    watermark is never legitimately below the highest label. Below it means
    the file was written by the old _reconcile_watermark, which pulled the
    watermark down to the client's array length when a cap shortened it.

    Left alone, that costs twice over: turns 101-280 are re-summarized from
    the WRONG text (the offset arithmetic no longer matches the labels), and
    then each new chunk collides with an existing label and is thrown away by
    the duplicate guard in _do_l1_rollup — silently, for as many turns as the
    old chunks span. Raising the pointer to what the chunks already prove is
    the only repair that loses nothing: the text under those labels is
    summarized, and this module still holds the summaries.

    Returns True if it changed anything, so the caller persists it.
    """
    highest = _highest_chunk_turn(state)
    last = int(state.get("last_summarized_turn") or 0)
    if highest <= last:
        return False
    state["last_summarized_turn"] = highest
    logger.warning(
        f"conv={conv_id}: the watermark was at turn {last} while stored "
        f"chunks already cover through turn {highest} — a state file written "
        f"by the pre-v3.1.4 watermark reset. Advancing it to {highest}, which "
        f"is what the chunks themselves record; without this every new chunk "
        f"would be labelled over an existing one and discarded"
    )
    return True


def _observed_position(conv_id: str, state: dict, messages: list[dict]) -> int:
    """This conversation's monotonic turn position, updated in `state`.

    THE INVARIANTS, because several separate defects lived in the arithmetic
    below and the next person needs to be able to check it:

      I1. The position is the number of turns the CONVERSATION has reached.
          It never moves backwards. That is the deliberate reversal of
          _reconcile_watermark (S-5 / REMEDIATION F14), which pulled the
          watermark down to the observed count: under a PERSISTENT cap the
          pull-down happens once and the rollup gate then never opens again.

      I2. `n`, the count of non-system messages, is a LOWER BOUND on the
          position and nothing more. Every turn in the window is a real turn
          of this conversation, so the conversation has at least n of them —
          but a capped window is a SUFFIX, so n says nothing about how many
          came before. Until v3.1.7 `n > prev` was read as "the array length
          IS the position", which is true only while the window is unbounded.
          A sliding cap does not snap from full history to short window in one
          step: on the first request where it bites, n is still greater than
          prev while ALREADY BEING SHORTER than the truth (cap 100, exchange
          51: the client stores 100 turns, appends turn 101, the valve trims
          to the last 100, the compactor appends turn 102 -> n = 101 against a
          truth of 102). One turn of position was swallowed permanently, the
          chunk labelled 101-120 held turns 102-121, and nothing said so
          (R23).

      I3. `prev + aligned` is the other lower bound: `prev` is monotonic and
          `aligned` counts the turns past the anchor's match. So the position
          is `max(n, prev + new)` — the larger of two lower bounds, which is
          exact whenever either source is exact and never over-counts.

      I4. Over-counting drops turns for good; under-counting only summarizes
          some of them twice. Where the evidence is ambiguous, take the
          smaller — EXCEPT that "0 new turns" under a cap is not merely
          conservative, it is a stall: the position is the only thing that
          advances, so a zero that repeats forever is the frozen hierarchy
          this release exists to fix, reached by a different route (R18).
          A zero is therefore only believed when it is unambiguous, or when
          the window itself is unchanged (same head, same length) AND is not
          a strict suffix of a longer conversation. The second half is not
          decoration: a capped window that has filled with byte-identical
          exchanges is byte-identical to the one before it, so "unchanged"
          alone reads a live stall as the admin drain and freezes the
          position for as long as the loop runs.

      I5. An empty window is not evidence of anything. A request with no
          non-system turns at all must leave both the position and the anchor
          exactly as they were — advancing by an exchange invents two turns,
          and overwriting the anchor with [] destroys the only thing that can
          align the NEXT window (R21).

      I6. `prev` comes from _recorded_position, which counts the CHUNK LABELS
          as well as the two pointers. A chunk labelled a..b is evidence that
          b turns existed and were summarized, and it is the only one of the
          three that the old _reconcile_watermark could not erase. Seeding
          from the pointers alone restarted an upgraded conversation at the
          cap, so every new chunk was labelled over one that already existed
          and the duplicate guard threw it away — silently, for hundreds of
          turns (R12). Where there is no anchor, the labels are also the only
          evidence that the window in hand is a SUFFIX rather than the whole
          conversation; see the branch below.
    """
    turns = [m for m in messages if m.get("role") != "system"]
    n = len(turns)
    prev = _recorded_position(state)

    if not turns:
        # I5. Neither the position nor the anchor may be touched.
        state["turns_seen"] = prev
        return prev

    # R17: only the tail is hashed. The alignment below cannot use a slot
    # older than the anchor plus everything appended after it, and hashing the
    # whole history cost 47 ms per call at 660 turns, on the event loop,
    # inside conv_lock, every single turn.
    fps = _turn_fingerprints(turns[-_FINGERPRINT_TAIL_TURNS:])
    head_fp = _turn_fingerprints(turns[:1])[0]
    anchor = [x for x in (state.get("tail_fp") or []) if isinstance(x, str)]
    # Same head, same length: the client re-sent the window it sent last time.
    # Under a cap the head slides out on every exchange, so this is USUALLY a
    # reliable negative — and it is the only content evidence that separates
    # the admin drain (the same transcript, looped) from a live turn whose
    # tail repeats.
    #
    # USUALLY, and the exception is the whole point of the second clause
    # below. Once a capped window has filled with byte-identical exchanges —
    # a model looping, or `_redact_degenerate_turns` replacing every reply
    # with the same placeholder — the window slides by two turns per exchange
    # onto content that is period-2 identical, so the head hash repeats and
    # the length is pinned at the cap. The two arrays are then equal BYTE FOR
    # BYTE, and no content test of any width can tell them apart: comparing
    # the whole window, or the whole fingerprint tail, gives the same answer
    # as comparing the head. Measured at cap 20: the position advanced for
    # the first ten repeated exchanges, then stalled permanently at turn 40
    # while the conversation ran on to 68 (R18, second route).
    #
    # WHAT DOES SEPARATE THEM IS NOT CONTENT. `n < prev` says the window is a
    # strict SUFFIX of a longer conversation — invariant I2's own reading of a
    # bounded window, on the evidence rather than on the outcome (the
    # "bounded window" line below tests `position > n`, which is the same
    # judgement AFTER `new` has been chosen and so cannot inform the choice).
    # The admin drain cannot be in that state: /admin/compact
    # refuses (409) unless the rebuilt transcript REACHES
    # _recorded_position, so throughout its loop n >= prev — and holding
    # keeps it there, because holding leaves the position at max(n, prev) = n.
    # So a repeating window with n < prev is a live capped turn, and the zero
    # is the coincidence.
    #
    # The trade, stated because it is a trade: a state file whose watermark is
    # stranded ABOVE its true position (the S-5 case) also reads n < prev, so
    # if such a conversation ALSO has a repeating tail AND the client re-sends
    # a byte-identical window, this advances 2 turns it should not have. That
    # costs two turns of coverage once per duplicate request; the stall it
    # replaces costs every turn of the hierarchy for as long as the loop runs,
    # which is the failure this release exists to fix.
    window_unchanged = (
        n == int(state.get("window_turns") or 0)
        and head_fp == (state.get("head_fp") or "")
    )
    window_is_a_suffix = n < prev

    if not anchor:
        # First sight of this conversation with no anchor to compare against —
        # a state file from before v3.1.4, or a conversation whose very first
        # observation is already capped.
        #
        # The question is whether the window in hand ALREADY contains this
        # exchange's two turns (so holding is right) or is a suffix that sits
        # past everything recorded (so holding loses two turns that are never
        # repaid). `prev` cannot answer it: a watermark can be stranded above
        # a genuinely shorter history, which is the S-5 case, and treating
        # that as proof of a bounded window pushes the position past a history
        # that has not caught up to it.
        #
        # THE CHUNK LABELS CAN. A stored chunk labelled a..b is evidence that
        # b turns of this conversation existed and were summarized — evidence
        # that carries text, unlike the watermark. So:
        if n < _highest_chunk_turn(state):
            # The window provably cannot hold the whole conversation: the
            # chunks alone account for more turns than the client sent. It is
            # a suffix, so this call's exchange is past everything recorded,
            # and main.py calls maybe_rollup once per exchange. Without this
            # the R12 upgrade path (chunks to turn 660, watermark pulled down
            # to the cap) came back with every chunk labelled two turns off
            # the text inside it.
            new = _ASSUMED_NEW_TURNS
        else:
            # The window could contain everything the chunks prove exists, so
            # the watermark is the suspect number, not the window. HOLD, and
            # let THIS call lay the anchor down; from the next call on the
            # alignment is exact. Holding is the under-counting side
            # (invariant I4), and it is also what keeps /admin/compact's drain
            # loop still — it re-presents ONE unchanged transcript, which
            # n >= _highest_chunk_turn always describes.
            new = 0
    else:
        cands = _align_candidates(anchor, fps)
        if not cands:
            new = _ASSUMED_NEW_TURNS
            if logsetup.log_once("summarizer.position.unaligned"):
                logger.warning(
                    f"conv={conv_id}: none of the {len(anchor)} anchored "
                    f"turns appear in the {n}-turn window the client sent, "
                    f"so the conversation's position ({prev}) cannot be "
                    f"measured against it; advancing by "
                    f"{_ASSUMED_NEW_TURNS} (one exchange) per rollup call "
                    f"instead. Summary turn labels will drift from the "
                    f"client's numbering, which costs nothing, but a "
                    f"repeat of this line means the anchor is not "
                    f"round-tripping through the client"
                )
        elif (
            cands[0] == 0
            and len(cands) > 1
            and not (window_unchanged and not window_is_a_suffix)
        ):
            # I4. The anchor occurs at the very end AND earlier, so "nothing
            # advanced" and "one exchange advanced" are equally consistent
            # with it — a tail of byte-identical exchanges, which
            # _redact_degenerate_turns manufactures out of two consecutive
            # degenerate replies. Either the window's head and length say it
            # is not the same window, or the window is a strict suffix of a
            # longer conversation and so cannot be the admin drain's
            # re-presented transcript — see window_unchanged above. Either
            # way the zero is the coincidence, not the truth.
            new = cands[1]
            if logsetup.log_once("summarizer.position.repeating_tail"):
                logger.info(
                    f"conv={conv_id}: the last {len(anchor)} turns of this "
                    f"conversation repeat earlier ones, so the anchor alone "
                    f"cannot say whether the window moved; "
                    + (
                        f"the window is {n} turns against a position of "
                        f"{prev}, so it is a suffix of a longer conversation "
                        f"and not a re-presented transcript"
                        if window_is_a_suffix
                        else "the window itself changed"
                    )
                    + f", so taking {new} new turns rather than 0. "
                    f"Reading 0 here would stop the position advancing, and "
                    f"under a cap the position is the only thing that does"
                )
        else:
            new = cands[0]

    position = max(n, prev + new)
    if position > n and logsetup.log_once("summarizer.position.bounded"):
        # Once per process: this is the tail of EVERY turn under a cap.
        # It is the healthy shape, not a fault — logged so that an
        # operator turning max_turns on can see the compactor noticed.
        logger.info(
            f"conv={conv_id}: the client is sending a bounded window "
            f"({n} turns) while the conversation is at turn {position}; "
            f"rollups are driven by the compactor's own counter from here "
            f"on, and chunk text is read at an offset of {position - n}"
        )

    state["turns_seen"] = position
    state["tail_fp"] = fps[-_ANCHOR_TURNS:]
    state["head_fp"] = head_fp
    state["window_turns"] = n
    return position


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


# The name this module's own /tokenize failures are tracked under in
# tokenhealth's per-source registry. See tokenize_health() below for what
# reads it back out.
_TOKENIZE_SOURCE = "summarizer"


def tokenize_health() -> dict:
    """This module's own /tokenize dependency state, tokenhealth-shaped.

    CONSUMED by /health/full: main._summarizer_tokenize_failing_now() and
    main._summarizer_degraded_since() fold this into that endpoint's `ok`,
    `consecutive_failures`, `degraded_since` and `degraded_for_s`. (This
    docstring said "not yet consumed" for about an hour after it became
    false - the wiring landed in the same diff. A13's lesson was that a
    health signal nothing reads is indistinguishable from one that does not
    exist; a docstring claiming it is unread is the same failure wearing a
    different hat.)

    Passes `stale_after_s=TOKENIZE_WARN_INTERVAL_S` into source_health
    because /tokenize is NOT asked on every request here — only when a
    rollup is actually summarizing (main._text_tokenize_failing_now carries
    the identical doctrine for its own not-every-request form): a streak
    whose last failure is long past means "not asked lately," not
    "recovered," and reporting either one as the other is asserting a fact
    that isn't observable from here.
    """
    return tokenhealth.source_health(
        _TOKENIZE_SOURCE, stale_after_s=TOKENIZE_WARN_INTERVAL_S
    )


async def _count_tokens(
    client: httpx.AsyncClient, vllm_url: str, model: str, text: str
) -> int:
    """What vLLM will charge for `text`. Asks vLLM; falls back to the
    pessimistic ceiling, never to an optimistic guess.

    Failures and recoveries are counted through tokenhealth (v3.1 remediation
    residual). Before this, a failure here reported through
    `logsetup.log_once` — a gate that fires ONE line for the life of the
    process and then nothing, ever again. An outage starting any time after
    that first line was invisible in the log AND absent from every health
    surface, because nothing here fed a counter anything could read.
    tokenhealth.note_failure/note_success keep the "don't spam a line per
    call" property (still rate-limited, at TOKENIZE_WARN_INTERVAL_S — the
    same interval main.py tunes its own /tokenize reporting with) while
    remaining observable for as long as the degradation actually lasts, and
    they feed the streak this module's tokenize_health() reads back out.
    `logger.warning` stays a call made HERE, on this module's own logger,
    rather than inside tokenhealth — see tokenhealth.py's module docstring
    for why (logger-hierarchy propagation makes that mandatory, not stylistic:
    a warning logged under a sibling logger would never reach this module's
    own captured tests).
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
                msg = tokenhealth.note_success(_TOKENIZE_SOURCE)
                if msg:
                    logger.warning(msg)
                return int(n)
        status = getattr(r, "status_code", "?")
        msg = tokenhealth.note_failure(
            _TOKENIZE_SOURCE,
            f"http.{status}",
            f"/tokenize did not answer with a count (status {status}); "
            f"rollup input is being budgeted at {_WORST_TOKENS_PER_CHAR} "
            f"tokens/char instead, so rollups will over-split until this "
            f"recovers",
            warn_interval_s=TOKENIZE_WARN_INTERVAL_S,
        )
        if msg:
            logger.warning(msg)
    except Exception as e:
        msg = tokenhealth.note_failure(
            _TOKENIZE_SOURCE,
            f"error.{type(e).__name__}",
            f"/tokenize unreachable ({type(e).__name__}: {e}); rollup input "
            f"is being budgeted at {_WORST_TOKENS_PER_CHAR} tokens/char "
            f"instead, so rollups will over-split until this recovers",
            warn_interval_s=TOKENIZE_WARN_INTERVAL_S,
        )
        if msg:
            logger.warning(msg)
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
    """Every tier's summary text comes from here, so this is where
    rule/box decoration comes off it (v3.1.8).

    A WRAPPER rather than a strip at each `return`. _summarize_pieces_raw
    has five return paths and four call sites; applying the rule at any
    subset of them is the fix-one-site-miss-the-sibling defect this
    codebase has paid for more than a dozen times. One seam, all tiers,
    every path.

    Measured before this landed: 3 of 14 summary files carried box
    characters, 1,173 in all, including the live summary of the
    conversation in daily use - which is injected into every request. A
    model shown its own decoration in its memory block keeps producing
    it, whatever the system prompt asks for.
    """
    return textclean.strip_rule_decoration(
        await _summarize_pieces_raw(
            conv_id, client, vllm_url, model, system_prompt, pieces,
            max_tokens,
        )
    )


async def _summarize_pieces_raw(
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

    raw = await asyncio.gather(*(_bounded(system_prompt, b) for b in batches))
    _empty = sum(1 for p in raw if not (p or "").strip())
    if _empty:
        # ANY empty map batch fails the WHOLE call - not just the all-empty
        # case. Filtering the empty part out stored an L1 chunk claiming
        # first_turn..last_turn while its text covered only the batches that
        # answered, and the watermark then advanced past turns that were
        # never summarized and never retried: permanent, unlogged loss in
        # the stored hierarchy. The same defect this branch fixed at both
        # main.summarize returns, at this sibling. Returning "" makes the
        # tier return False, nothing advances, and the rollup retries next
        # turn.
        logger.warning(
            f"conv={conv_id}: {_empty} of {len(batches)} rollup map "
            f"batch(es) returned empty content - failing the whole call so "
            f"no chunk is stored claiming a span its text does not cover"
        )
        return ""
    parts = list(raw)

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
        if any(not (p or "").strip() for p in folded):
            # A partial-empty fold quietly deletes whichever group came back
            # blank. The pre-fold parts are all non-empty (the map phase
            # guarantees it), so concatenating them is complete, just longer.
            logger.warning(
                f"conv={conv_id}: rollup reduce round {rounds} returned "
                f"empty content for a group - keeping the "
                f"{len(parts)} partial(s) concatenated instead"
            )
            break
        parts = list(folded)
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
    window_offset: int = 0,
) -> bool:
    """Roll the next L1_CHUNK_SIZE turns after last_summarized_turn into a
    new L1 chunk. Returns True if the watermark advanced.

    `window_offset` is (position - len(window)): how many turns of this
    conversation sit BEFORE the first turn the client sent. It is 0 for a
    client re-sending the whole history, which is why every existing caller and
    test that omits it gets byte-identical behaviour. Under a cap it is the
    number that turns a turn LABEL into an index into the array in hand —
    without it the chunk boundaries are absolute positions in an array that no
    longer starts at turn 1, and _turn_pieces would summarize the wrong text
    while labelling it correctly, which is worse than summarizing nothing
    because nothing downstream can tell.
    """
    last = state.get("last_summarized_turn", 0)
    first_turn = last + 1
    last_turn = last + L1_CHUNK_SIZE
    pos_first = first_turn - window_offset
    pos_last = last_turn - window_offset

    if pos_last < 1:
        # This whole chunk scrolled out of the client's window before it was
        # ever summarized — only reachable when the backlog exceeds the cap
        # plus L1_CHUNK_SIZE (119 turns at max_turns=100), i.e. after a long
        # rollup outage. The text is not in the request and this module never
        # held a copy, so there is nothing to summarize. Skipping the dead span
        # is the only alternative to a hierarchy that is stuck on it forever,
        # and a hierarchy that stops advancing also stops recording the turns
        # that ARE still arriving.
        state["last_summarized_turn"] = window_offset
        logger.error(
            f"conv={conv_id}: turns {first_turn}-{window_offset} are behind "
            f"the client's window and were never summarized; the watermark "
            f"has been advanced past them so newer turns are not lost too. "
            f"POST /admin/conversations/{conv_id}/compact rebuilds from the "
            f"episodic store, which may still hold that text"
        )
        return True

    covered_first = first_turn
    partial = pos_first < 1
    if partial:
        pos_first = 1
        covered_first = window_offset + 1

    # One piece per turn, so an oversized slice can be split rather than sent
    # whole and refused. The chunk still COVERS first_turn..last_turn either
    # way — the turn range is the contract the watermark and the L2 rollup
    # depend on, and splitting the request must not change it (v3.1 A1).
    pieces = _turn_pieces(messages, pos_first, pos_last)
    if not any(p.strip() for p in pieces):
        return False
    if any(
        c.get("first_turn") == covered_first and c.get("last_turn") == last_turn
        for c in state.get("l1") or []
    ):
        # Belt and braces against the hazard the old watermark reset created:
        # an operator running /admin/conversations/<id>/compact twice appended
        # a second identical chunk set, which then cascaded into duplicate L2
        # chapters and a duplicate-fed L3.
        #
        # ERROR, not WARNING, since v3.1.7. The watermark is now repaired
        # against the chunk list before any position is derived from it
        # (_repair_watermark_below_chunks), and _recorded_position seeds from
        # the labels too, so the only way to reach this line is a genuine
        # re-presentation of a span the labels already own — the idempotent
        # admin drain. Under the pre-v3.1.7 code this fired on the FIRST
        # rollup of every upgraded capped conversation and threw away a real
        # span each time, at WARNING, for hundreds of turns (R12). If it is
        # in the log now, the position arithmetic is wrong again, and the
        # skip below is hiding how much.
        state["last_summarized_turn"] = last_turn
        logger.error(
            f"conv={conv_id}: an L1 chunk covering turns {covered_first}-"
            f"{last_turn} already exists; advancing the watermark past it "
            f"instead of storing a duplicate. This should be unreachable — "
            f"the position only moves forward and is seeded from the chunk "
            f"labels — so check turns_seen against the l1 spans in "
            f"GET /admin/conversations/{conv_id}"
        )
        return True
    text = await _summarize_pieces(
        conv_id, client, vllm_url, model, _PROMPT_L1, pieces, L1_MAX_TOKENS
    )
    if not text:
        return False
    if partial:
        # BELOW the summarization, not above it (v3.1.7, R29). Logged first,
        # this line announced "recording that as the chunk's span" and then
        # the vLLM call returned empty and recorded nothing — four such lines
        # against l1=0 during an outage, which is exactly when an operator is
        # reading the log.
        logger.warning(
            f"conv={conv_id}: turns {first_turn}-{window_offset} of this "
            f"chunk are behind the client's window; summarized turns "
            f"{covered_first}-{last_turn} and recorded that as the "
            f"chunk's span rather than claiming coverage of text this rollup "
            f"never saw"
        )
    state["l1"].append({
        "text": text, "first_turn": covered_first, "last_turn": last_turn,
    })
    state["last_summarized_turn"] = last_turn
    # A rollup had no success line of its own, so the only evidence the
    # hierarchy was advancing was the injection counter — which is why S-5
    # froze it for the life of the deployment without anyone noticing.
    logger.info(
        f"conv={conv_id}: L1 rollup — chunk {len(state['l1'])} covers turns "
        f"{covered_first}-{last_turn}"
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


def _chapter_piece(c: dict) -> str:
    return (
        f"--- chapter (turns {c['first_turn']}-{c['last_turn']}) ---\n"
        f"{c['text']}"
    )


def _archive_chapters(conv_id: str, chapters: list[dict]) -> None:
    """Append L2 chapters to the cold-storage sidecar, newest last.

    Read back with load_chapter_archive(). Nothing injects these - they cost
    no context - but they are the raw material an operator (or a future
    re-derivation) needs after L3 has paraphrased them several generations
    deep.
    """
    if not chapters:
        return
    path = summary_archive_path(conv_id)
    existing = read_json_strict(path, default={}, expect=dict)
    rows = existing.get("chapters") if isinstance(existing, dict) else None
    if not isinstance(rows, list):
        rows = []
    # Dedupe against what is already stored: the archive now runs BEFORE
    # save_state, so a failed state save retries the whole refresh next turn
    # and would re-archive the same chapters (measured: two refreshes of the
    # same state produced 5 exact duplicate rows). Identity is the full
    # (span, text) triple - two different chapters legitimately covering the
    # same span must both survive.
    _seen = {(r.get("first_turn"), r.get("last_turn"), r.get("text", ""))
             for r in rows}
    rows.extend(
        {
            "text": c.get("text", ""),
            "first_turn": c.get("first_turn"),
            "last_turn": c.get("last_turn"),
        }
        for c in chapters
        if (c.get("first_turn"), c.get("last_turn"), c.get("text", ""))
        not in _seen
    )
    atomic_write_json(path, {"chapters": rows})
    logger.info(
        f"conv={conv_id}: archived {len(chapters)} consumed chapter(s) "
        f"({len(rows)} total in cold storage)"
    )


def load_chapter_archive(conv_id: str) -> list[dict]:
    """Every L2 chapter ever consumed by an L3 refresh, oldest first."""
    data = read_json_strict(summary_archive_path(conv_id), default={}, expect=dict)
    rows = data.get("chapters") if isinstance(data, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


async def _do_l3_rollup(
    conv_id: str, client: httpx.AsyncClient, vllm_url: str, model: str, state: dict,
) -> bool:
    """Roll ALL current L2 chapters into / refresh L3, then DROP them from
    l2 — the same consume-and-clear contract L1→L2 already has, so this tier
    is bounded the same way L1 is (MEMORY_REVIEW S-1/S-6).

    Before this fix, a successful refresh kept every L2 chapter it had just
    folded in, so l2 grew by one chapter every L2_CHUNK_SIZE*L1_CHUNK_SIZE
    turns for the life of the conversation — unbounded in the state file AND
    in format_summary_block's injected block, and unbounded L3 INPUT too (at
    L2_MAX_TOKENS=1200 a 25-chapter conversation was already a ~30,000-token
    request). Measured on a synthetic 240-turn run at this module's test
    thresholds (L1=4/L2=3/L3=2): len(l2) reached 20 and was still climbing.
    L3 is a single object, not a list, so what "bounded" means for L2 here is
    "at most L3_CHUNK_SIZE-1 chapters awaiting the next refresh" — the same
    shape l1 already had relative to L2_CHUNK_SIZE.

    The trade this makes explicit: once a span of chapters is folded into
    L3, the CHAPTER-level detail for that span is gone from state and from
    injection - only L3's denser paraphrase of it remains. That is the same
    lossy-on-purpose compression this module's docstring already describes
    for L1->L2 ("roll older content into denser representations without
    re-touching it"), now actually applied at the L2->L3 boundary instead of
    stopping short of it.

    THE PRIOR L3 IS CARRIED FORWARD AS AN INPUT, and that is load-bearing.
    L1->L2 APPENDS to a list, so dropping its inputs loses nothing. L3 is a
    single object that is REPLACED, so clearing l2 without feeding the old
    L3 back in would make each refresh summarize only the newest
    L3_CHUNK_SIZE chapters and overwrite everything earlier: the first
    refresh covers turns 1-N, the second silently replaces it with a summary
    covering only N+1-M. That is permanent, unannounced deletion of the
    oldest history in the system - strictly worse than the unbounded growth
    this fix set out to solve. The two tiers do NOT have the same contract,
    and the difference is list-versus-object.

    So the refresh input is (previous L3 + the pending chapters), and
    first_turn is inherited from the previous L3 rather than taken from the
    chapter list, and L3 keeps covering turn 1 through now.

    On INPUT SIZE, stated carefully because the earlier wording was wrong:
    stage 2 is bounded (one L3 body plus one reduced chapter summary, both
    capped at L3_MAX_TOKENS). Stage 1 is NOT - it takes every pending
    chapter, and a single maybe_rollup over a long history from empty state
    (the backfill and admin-compact shape) can present far more than
    L3_CHUNK_SIZE of them. That is why stage 1 goes through _summarize_pieces,
    which map-reduces; the earlier claim of a hard per-refresh bound was
    measured false at 10x.

    The cost this makes explicit: the previous L3 is re-summarized each
    refresh, so the oldest material gains one generation of paraphrase per
    refresh rather than being re-derived from chapters each time. The
    chapters are archived (see _archive_chapters) precisely so that is a
    quality trade and not a loss - the source survives in cold storage.
    """
    l2 = state.get("l2") or []
    if len(l2) < L3_CHUNK_SIZE:
        return False
    prior = state.get("l3") if isinstance(state.get("l3"), dict) else None
    prior_piece = None
    if prior and (prior.get("text") or "").strip():
        # First, so the model reads the story in order and the older
        # material is not competing for attention at the end of the prompt.
        prior_piece = (
            f"--- the story so far (turns {prior.get('first_turn','?')}-"
            f"{prior.get('last_turn','?')}) ---\n{prior['text']}"
        )
    # TWO-STAGE when a prior L3 exists, and this is what makes the span
    # below honest rather than merely hopeful.
    #
    # _summarize_pieces map-reduces whenever its input exceeds the budget,
    # and its reduce drops empty parts - so a 200-with-empty-content on the
    # batch carrying "the story so far" would discard the prior L3 while
    # first_turn still claimed to cover it. L3 would then assert coverage of
    # turns its text does not describe, which is worse than losing them
    # because nothing downstream can tell.
    #
    # Reducing the CHAPTERS first (map-reduce is fine there - every chapter
    # is an input, none is privileged) and only then folding the prior L3
    # into a second call means the final text always comes from a call that
    # contained it: both inputs are bounded by L3_MAX_TOKENS, so that second
    # call is a single batch whenever a real token count is available. (The
    # pessimistic no-tokenizer fallback prices at 2 tokens/char and CAN
    # split it - which degrades to concatenation of non-empty parts, not
    # loss, per _summarize_pieces' partial-empty rules.) An earlier attempt
    # instead fed only the
    # chapters that fit one batch, which broke the guarantee that an
    # oversized chapter set is still covered in full.
    text = await _summarize_pieces(
        conv_id, client, vllm_url, model, _PROMPT_L3,
        [_chapter_piece(c) for c in l2], L3_MAX_TOKENS,
    )
    if text and prior_piece:
        text = await _summarize_pieces(
            conv_id, client, vllm_url, model, _PROMPT_L3,
            [prior_piece, f"--- newer chapters ---{chr(10)}{text}"],
            L3_MAX_TOKENS,
        )
    if not text:
        return False
    # ARCHIVE FIRST, and a failed archive ABORTS the refresh before any
    # state mutates. The previous order (write l3, clear l2, then try to
    # archive) meant an archive-write failure still dropped the chapters
    # with no cold copy - loudly, but violating the invariant all the same:
    # eviction MOVES memory, it never unlinks it. Aborting wastes the LLM
    # calls that produced `text`, and that is the right trade: the refresh
    # retries next turn, and a disk that cannot take the archive write is a
    # bigger problem than a deferred rollup.
    try:
        _archive_chapters(conv_id, l2)
    except Exception as e:
        logger.error(
            f"conv={conv_id}: could not archive the {len(l2)} chapter(s) "
            f"this L3 refresh would consume ({type(e).__name__}: {e}) - "
            f"ABORTING the refresh so nothing is dropped without a cold "
            f"copy; it will retry next turn"
        )
        return False
    state["l3"] = {
        "text": text,
        # Inherit the START of coverage. Taking l2[0] here would move the
        # span forward on every refresh and quietly discard everything
        # before it - see this function's docstring.
        "first_turn": (
            prior.get("first_turn")
            if prior and prior.get("first_turn") is not None
            else l2[0]["first_turn"]
        ),
        "last_turn": l2[-1]["last_turn"],
    }
    state["l2"] = []  # consumed; archived above (pre-mutation), bounds l2 like l1
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

    `messages` is whatever history the client sent — the FULL array from a
    client that re-sends everything, or a bounded window from one that does
    not. Which of the two it is no longer decides whether rollups happen:
    _observed_position owns the conversation's position and `window_offset`
    maps it back onto the array in hand (v3.1.4).
    """
    async with conv_lock(conv_id):
        state = load_state(conv_id)

        # Before anything reads the watermark: a file written by the old
        # _reconcile_watermark can have it BELOW the chunks it wrote, and
        # every number below is derived from it (v3.1.7, R12).
        changed = _repair_watermark_below_chunks(conv_id, state)

        before_position = state.get("turns_seen")
        before_anchor = state.get("tail_fp")
        before_window = (state.get("head_fp"), state.get("window_turns"))
        current_turns = _observed_position(conv_id, state, messages)
        # The position and the anchor are useless unless they are PERSISTED:
        # unwritten, the next call re-seeds from last_summarized_turn, finds no
        # anchor, and holds — which is the frozen hierarchy this release
        # exists to fix, reintroduced by an unsaved counter. The window
        # signature rides along for the same reason: unwritten, every
        # repeating tail reads as ambiguous forever (R18).
        changed = changed or (
            before_position != state["turns_seen"]
            or before_anchor != state["tail_fp"]
            or before_window != (state.get("head_fp"), state.get("window_turns"))
        )
        # How many turns of this conversation sit before the array's first
        # turn. Computed ONCE and held constant for the whole drain: the
        # contiguity of consecutive L1 chunks is exactly the property that a
        # varying offset would break.
        window_offset = current_turns - sum(
            1 for m in messages if m.get("role") != "system"
        )

        if needs_rollup(state, current_turns):
            try:
                async with httpx.AsyncClient() as client:
                    # Drain L1 rollups until either caught up or no more material.
                    while _needs_l1_rollup(state, current_turns):
                        if not await _do_l1_rollup(
                            conv_id, client, vllm_url, model, state, messages,
                            window_offset,
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
        # The pair is what an operator needs to read together: a watermark
        # that is not moving is only a fault if turns_seen IS.
        "turns_seen": state.get("turns_seen", 0),
        "l3_turns_covered": (
            [l3.get("first_turn"), l3.get("last_turn")] if l3 else None
        ),
    }
