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
import bisect
import contextlib
import contextvars
import hashlib
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.concurrency import run_in_threadpool

import logsetup
import textclean
import tokens
import tokenhealth
from envcfg import env_float, env_int
from memory import (
    StoreUnreadable,
    atomic_write_json,
    conv_lock,
    current_wipe_generation,
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

# Same env var and default main.py reads for its own /tokenize call sites,
# under its own module-level TOKENIZE_WARN_INTERVAL_S — deliberately, not
# independently tuned: an operator setting this once should govern every
# /tokenize dependency in the process, not just the ones main.py happens to
# own. (hostile2-config: this pair used to disagree — main.py parsed the
# variable with int(), so any non-integer value there silently reverted to
# the 300 default while this module applied it correctly. Fixed by reading
# it with env_float in both places; test_tail_tokenize_warn_interval.py
# pins the two modules' parsing against each other directly. A line-number
# citation was here and went stale the first time either file was edited
# above it — the constant's own name does not.)
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
        # v3.1.9. One fingerprint per covered position, position 1 first,
        # written by the chunk that covers it from the turn that chunk read
        # (as the client sees it), concatenated; _FP_UNKNOWN where no
        # record-writing chunk read the position. Absent on every file
        # written before this release, and absent reads as NO EVIDENCE: the
        # reuse path declines until a rollup records it (or adopts it once,
        # _adopt_legacy_record), which costs a summarization call and cannot
        # cost a turn. See _record_chunk_fps.
        "covered_fps": "",
        # hostile pass #4 (reviewer A F6). Turns a chunk read OUT OF POSITION:
        # request turns that paired with nothing although they sit inside
        # the covered span (the text after a delete or regenerate of a
        # chunk-closing exchange, an edit, a legacy position no evidence
        # backs), re-read by the next L1 chunk as extra pieces. One
        # [after_position, owner_last_turn, fingerprint] per turn: it sorts
        # after record entry `after_position`, and it counts only while the
        # chunk that read it (ending at owner_last_turn) is inside the
        # covered prefix. See _record_sequence and _patch_candidates.
        "covered_extra": [],
        # hostile pass #4 (reviewer A F2/F3). Set once the one-shot adoption
        # of pre-v3.1.9 chunks has run, so it cannot run again whatever the
        # leading entries of the record hold. See _adopt_legacy_record.
        "legacy_adopted": False,
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
        # v3.1.9. The FOURTH site of A3-1, and the one on the hot path; the
        # hostile pass named three and not this. A tier that is not a list
        # was neither loaded NOR parked — the F1b parking below only ever
        # saw lists — so it silently read as [] and the next save_state
        # erased that tier of the hierarchy. Parking cannot hold it either:
        # _for_disk folds parked tiers back with list(), which turns a dict
        # into its keys. So it raises, exactly as an unreadable file already
        # does from this function — no caller gains a new obligation.
        v = data.get(tier, [])
        if not isinstance(v, list):
            raise StoreUnreadable(
                summary_path(conv_id),
                TypeError(f'"{tier}" is {type(v).__name__}, not a list'),
            )
        state[tier] = [x for x in v if _is_chunk(x)]
        parked[tier] = [x for x in v if not _is_chunk(x)]
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
    # v3.1.9. Validated as a whole (see _covered_fps). The digest keys an
    # earlier unreleased cut of this branch wrote (`covered_fp`,
    # `covered_fp_turns`, `covered_fp_marks`) are ignored and dropped on the
    # next save: the first two were built over redacted text and can never
    # match a request.
    if _covered_fps(data):
        state["covered_fps"] = data["covered_fps"]
    # hostile pass #4. Invalid rows are dropped one by one rather than voiding
    # the list: an extra entry is only ever a reason to REPLACE a turn, so a
    # row that is not read costs a refresh, never a turn.
    state["covered_extra"] = _covered_extra(data)
    if isinstance(data.get("legacy_adopted"), bool):
        state["legacy_adopted"] = data["legacy_adopted"]
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


def format_summary_block(
    state: dict, max_tokens: int | None = None, *, all_or_nothing: bool = False
) -> str | None:
    """Render the current summary stack into a single system-message body.
    Returns None if there's nothing to inject.

    all_or_nothing=True returns None instead of a partial block whenever the
    budget forced ANY tier item out. It exists for one caller and one reason:
    compact_if_needed REMOVES the turns this block stands in for, and a squeeze
    here drops the OLDEST scenes first (see below) — the same end of the
    conversation compaction removes from the array. A partial block there is
    not a smaller summary, it is turns deleted from the array and absent from
    the stand-in: gone from the request entirely, with the log still reporting
    them as "covered by stored summaries". Demonstrated at 9 L1 / 4 L2 / 1 L3,
    a state at its documented capacity, via _estimate_block_tokens pricing
    non-ASCII per UTF-8 BYTE while L1_MAX_TOKENS bounds output TOKENS.

    Every other caller injects this block ALONGSIDE the turns rather than
    instead of them, so a partial block is a smaller summary and nothing more.
    They keep the default.

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
        if all_or_nothing:
            logger.warning(
                "and the caller asked for all-or-nothing, so NOTHING is "
                "returned: it substitutes this block for turns it removes "
                "from the array, and a block missing its oldest scenes "
                "cannot stand in for the oldest turns"
            )
            return None

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


def rollup_due_tiers(state: dict, current_turn_count: int) -> dict[str, bool]:
    """Public: which tier(s) need work, individually — {"l1": bool, "l2":
    bool, "l3": bool}. For a DIAGNOSTIC that must describe what is
    actually pending (main._rollup_hierarchy's catch-up INFO line,
    hostile pass #5 C5-7/E8), not for the drain itself, which reads the
    private `_needs_*` gates directly against a `state` that changes
    mid-call — this is a point-in-time snapshot a caller takes AFTER a
    pass, when it is safe to read once. One seam so a future rule change
    to any `_needs_*` gate cannot drift from what this reports, the same
    fix-one-site-miss-the-sibling concern `needs_rollup` above already
    avoids by delegating rather than re-deriving.
    """
    return {
        "l1": _needs_l1_rollup(state, current_turn_count),
        "l2": _needs_l2_rollup(state),
        "l3": _needs_l3_rollup(state),
    }


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


# Width of one covered-turn fingerprint in `covered_fps`, matching
# _turn_fingerprints.
_FP_WIDTH = 16

# A record entry for a position no record-writing chunk read: a hole in the
# chain, or a chunk written before v3.1.9 that no request has vouched for yet.
# Not hex, so no request turn's fingerprint can ever equal it: the turn at
# that position is summarized fresh, never replaced (see _record_chunk_fps).
_FP_UNKNOWN = "-" * _FP_WIDTH

# The note main._apply_image_retention leaves in place of a demoted image. It
# cannot import main (main imports summarizer), so the format is duplicated
# here, exactly like _image_only_marker; if main's note changes, change this
# pattern AND _RETENTION_NOTE_TAIL in the same commit or every demotion reads
# as an edited turn.
_RETENTION_NOTE = re.compile(
    r"\s*\[(\d+) images? shared earlier in this conversation\]\s*$"
)
# hostile pass #3 (reviewer A F6): what every note ends with. A text that does
# not end with this (after trailing whitespace) cannot contain a note, and the
# regex above is not run on it. See _covered_turn_fingerprint.
_RETENTION_NOTE_TAIL = "shared earlier in this conversation]"
# How far back from the end the note is searched for. The note is at most
# ~50 characters plus its digits; a longer window only changes how much
# leading whitespace the `\s*` absorbs, and whitespace is normalized away.
_RETENTION_NOTE_WINDOW = 256

# Memo of string-content fingerprints, keyed by (role, len, hash(text)).
# hostile pass #3 (reviewer A F6): _coverage_plan fingerprints every turn
# about to be removed on EVERY compacting request — measured 458 ms of
# GIL-bound CPU at 1,990 turns (2.2 s at her longest replies), stalling the
# event loop 84-141 ms from inside the threadpool. Nearly all of those turns
# are byte-identical to the previous request's. hash() of a str is one C pass
# (no regex, no split/join), so a steady conversation pays one pass per turn.
#
# Collision risk, stated because a wrong hit here would bless a changed turn:
# CPython's str hash is SipHash keyed per process, 64 bits, and the key adds
# role and length, so a wrong hit needs a 64-bit collision between an edited
# turn and its original of the same length — the same odds as the 64-bit
# truncated sha256 the fingerprint itself already is. Bounded: cleared at
# _FP_MEMO_MAX entries (one conversation of ~2,000 turns is 2,000 entries).
_FP_MEMO: dict[tuple[str, int, int], str] = {}
_FP_MEMO_MAX = 50_000


def _covered_turn_fingerprint(m: dict) -> str:
    """One covered-turn fingerprint. See _covered_turn_fingerprints."""
    role = str(m.get("role", "unknown"))
    content = m.get("content")
    key = None
    if isinstance(content, str):
        key = (role, len(content), hash(content))
        hit = _FP_MEMO.get(key)
        if hit is not None:
            return hit
    text = _message_text(m)
    images = 0
    note = None
    # The cheap test first (F6): 60% of this function's cost was the regex
    # attempting `\s*` at every whitespace position of every reply. It can
    # only match a text whose stripped end is the note's own end.
    if text.rstrip().endswith(_RETENTION_NOTE_TAIL):
        note = _RETENTION_NOTE.search(
            text, max(0, len(text) - _RETENTION_NOTE_WINDOW)
        )
    if note is not None:
        images = int(note.group(1))
        text = text[: note.start()]
    elif isinstance(content, list):
        images = sum(
            1 for c in content
            if isinstance(c, dict)
            and (c.get("type") in ("image_url", "image", "input_image")
                 or "image_url" in c)
        )
    payload = f"{role}\x00{' '.join(text.split())}\x00{images}"
    fp = hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()[:_FP_WIDTH]
    if key is not None:
        if len(_FP_MEMO) >= _FP_MEMO_MAX:
            _FP_MEMO.clear()
        _FP_MEMO[key] = fp
    return fp


def _covered_turn_fingerprints(messages: list[dict]) -> list[str]:
    """One fingerprint per non-system turn, for the covered-turns record ONLY.

    Not _turn_fingerprints, which feeds the position anchor and must not
    change under existing state files. This one has to be stable across
    something the anchor never compares across: IMAGE RETENTION. An image
    turn arrives with its image parts; a later upload demotes it to its text
    plus a "[1 image shared earlier in this conversation]" note, and the same
    turn now reads differently in every later request. Keyed on the raw text
    that would read as an edit on every conversation that ever shared two
    pictures. So a turn is its role, its whitespace-normalized text with any
    retention note removed, and its image count — the note's count when
    demoted, the parts' count when not — which is the same triple either way.
    """
    return [
        _covered_turn_fingerprint(m) for m in messages if m.get("role") != "system"
    ]


def _covered_fps(state: dict) -> list[str]:
    """The covered-turn record, position 1 first, validated.

    Each entry is a 16-hex fingerprint or _FP_UNKNOWN. Stored as ONE string
    rather than a JSON list: her conversation is ~1,900 turns and this is
    read on every request, so it is parsed as one token rather than 1,900.
    Anything that is not a whole number of well-formed entries voids the
    record — a partly unreadable record was written by something that did
    not finish, and reads as no evidence rather than as some of it.
    """
    raw = state.get("covered_fps")
    if not isinstance(raw, str) or len(raw) % _FP_WIDTH:
        return []
    if raw.strip("0123456789abcdef-"):
        return []
    entries = [raw[i:i + _FP_WIDTH] for i in range(0, len(raw), _FP_WIDTH)]
    if "-" in raw and any("-" in e and e != _FP_UNKNOWN for e in entries):
        return []
    return entries


def _record_chunk_fps(
    state: dict, first_turn: int, last_turn: int, turns: list[dict]
) -> bool:
    """Record what ONE chunk read: `turns` are the turns it summarized for
    positions first_turn..last_turn, as the client sees them. True if the
    record changed.

    hostile pass #3 (reviewer A, F1 and F9, both BLOCKER). THE FOURTH SHAPE,
    and why the third was wrong. The third recorded position N from whatever
    request next carried position N (`_catch_up_covered_fp`), so a chunk was
    routinely ahead of its record, and anything that changed in between was
    blessed:

      * F1. Every L1 chunk ends on the reply that was JUST streamed, which is
        not in the request, so its fingerprint was taken from the next
        request. Delete the last exchange, regenerate the chunk-closing
        reply, or delete only that reply, before the next message: the next
        request's turn at that position was recorded as covered, and from
        then on it matched its record and was replaced by a summary of the
        turn she had deleted. Lost for the life of the conversation.
      * F9. /admin/.../compact summarizes a transcript rebuilt from the
        episodic store, with a placeholder pair for every exchange the store
        never indexed, and passed no raw array, so it recorded nothing. The
        next live rollup then recorded every rebuilt position from her real
        turns, and the exchanges the rebuild only had placeholders for were
        deleted from every later request.

    So the record is written HERE, by the chunk, in the same state mutation
    that appends the chunk, from the turns the chunk read. No later event
    writes an entry for a position a chunk already covers, except the
    one-shot legacy adoption (_adopt_legacy_record). What "the turn it read,
    as the client sees it" means per caller is decided by maybe_rollup: the
    raw request plus the reply AS STREAMED for the live tail, the unredacted
    snapshot for the backfill, and the rollup input itself for a caller with
    no raw array (the admin rebuild), where a placeholder is recorded as a
    placeholder and so never matches the real turn.

    Append-only: an entry that already holds a fingerprint is never
    rewritten (a turn edited after its chunk keeps its original fingerprint,
    so exactly that turn reads as changed). Positions below `first_turn` that
    no entry covers are padded with _FP_UNKNOWN. An _FP_UNKNOWN entry is
    filled when a chunk reads that position, because the chunk then did read
    it.
    """
    if first_turn < 1 or last_turn < first_turn:
        return False
    fps = _covered_turn_fingerprints(turns)
    if len(fps) != last_turn - first_turn + 1:
        # The chunk's span and the turns handed in disagree: record nothing
        # rather than something misaligned. Its positions stay unrecorded
        # (and are padded UNKNOWN by the next chunk), which costs refreshes
        # and never a turn.
        return False
    entries = _covered_fps(state)
    before = "".join(entries)
    if len(entries) < first_turn - 1:
        entries.extend([_FP_UNKNOWN] * (first_turn - 1 - len(entries)))
    for k, fp in enumerate(fps, start=first_turn):
        if k <= len(entries):
            if entries[k - 1] == _FP_UNKNOWN:
                entries[k - 1] = fp
        else:
            entries.append(fp)
    after = "".join(entries)
    state["covered_fps"] = after
    return after != before


def _covered_extra(state: dict) -> list[tuple[int, int, str]]:
    """The out-of-position record (`covered_extra`), validated row by row.

    Each row is [after_position, owner_last_turn, fingerprint]: a turn some
    L1 chunk read as an extra piece (_patch_candidates), which sorts after
    covered position `after_position` and was read by the chunk ending at
    `owner_last_turn`. A malformed row is skipped, not the list: a row is
    only ever a reason to replace a turn, so an unread row costs a refresh.
    """
    raw = state.get("covered_extra")
    if not isinstance(raw, list):
        return []
    out: list[tuple[int, int, str]] = []
    for row in raw:
        if not (isinstance(row, (list, tuple)) and len(row) == 3):
            continue
        after, owner, fp = row
        if (
            type(after) is int and type(owner) is int and isinstance(fp, str)
            and after >= 0 and owner >= 1 and len(fp) == _FP_WIDTH
            and not fp.strip("0123456789abcdef")
        ):
            out.append((after, owner, fp))
    return out


def _record_sequence(state: dict) -> tuple[list[str], list[int]]:
    """The record as the pairing reads it: (fingerprints, positions), in
    conversation order.

    The chunk-written entries for positions 1..eff (eff: what the unbroken
    chunk chain from turn 1 backs, _covered_prefix), with every
    out-of-position row (_covered_extra) inserted after the position it sorts
    after. `positions[i]` is the covered position of sequence element i, or
    for an extra row the position it sorts after. An extra row counts only
    while the chunk that read it is inside that chain: a chunk parked by
    load_state or cut off by a hole takes its extras with it.
    """
    entries = _covered_fps(state)
    eff = min(_covered_prefix(state), len(entries))
    if eff <= 0:
        return [], []
    extras: dict[int, list[str]] = {}
    for after, owner, fp in _covered_extra(state):
        if owner <= eff and after <= eff:
            extras.setdefault(after, []).append(fp)
    seq: list[str] = list(extras.get(0, ()))
    pos: list[int] = [0] * len(seq)
    for k in range(1, eff + 1):
        seq.append(entries[k - 1])
        pos.append(k)
        for fp in extras.get(k, ()):
            seq.append(fp)
            pos.append(k)
    return seq, pos


def _record_patch_fps(
    state: dict, owner_last_turn: int, rows: list[tuple[int, str]]
) -> bool:
    """Append out-of-position rows (after_position, fingerprint) read by the
    chunk ending at `owner_last_turn`. True if anything was added.

    Counted, not de-duplicated: a row already held for the same
    (after_position, fingerprint) is not written again, but two identical
    turns read together ("ok", "ok") need two rows, because one entry pairs
    with one turn (_pairing) and the second would stay unpaired — refreshed
    on every request and re-read by every chunk, for good."""
    existing = _covered_extra(state)
    held: dict[tuple[int, str], int] = {}
    for a, _o, f in existing:
        held[(a, f)] = held.get((a, f), 0) + 1
    out = [list(r) for r in existing]
    added = False
    seen: dict[tuple[int, str], int] = {}
    for after, fp in rows:
        if fp == _FP_UNKNOWN:
            continue
        key = (int(after), fp)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= held.get(key, 0):
            continue
        out.append([int(after), int(owner_last_turn), fp])
        added = True
    state["covered_extra"] = out
    return added


def _legacy_unread_positions(state: dict, legacy_watermark: int | None) -> set[int]:
    """Covered positions a pre-v3.1.9 state file PROVES no chunk read.

    hostile pass #4 (reviewer A F2). A v3.1.6.1 file records no text, but its
    shape keeps three traces of the traffic that put unread turns under
    chunk labels:

      * a watermark BELOW the highest label (read before
        _repair_watermark_below_chunks raises it). v3.1.6.1's
        _reconcile_watermark set it to the array length on the tail after a
        delete or an edit-and-resend, so the exchange of that tail (w-1, w)
        and everything after it is text no chunk read — unless a chunk has
        closed at w since, in which case only what is after w is unread and
        the chunk's own start is the second trace;
      * a span that starts INSIDE a span written before it (41-60, then
        47-66): the pull-down again, and s-2, s-1 are that tail's exchange;
      * the closing exchange of every chunk still in l1, and of the furthest
        span: a regenerate of the reply that closed a chunk, or a delete of
        that exchange and a new message, moves no watermark on v3.1.6.1
        (39 + 1 = 40 is not below 40), so nothing else can tell it from the
        exchange the chunk read.

    What no trace can show, stated so nobody reads this as complete: an
    in-place edit (OpenWebUI's edit-and-Save without a new branch), and a
    regenerate of a closing reply whose chunk has since been consumed into
    an L2 chapter. See _adopt_legacy_record for why those are adopted.
    """
    unread: set[int] = set()
    l1 = [c for c in (state.get("l1") or []) if isinstance(c, dict)]
    l2 = [c for c in (state.get("l2") or []) if isinstance(c, dict)]
    l3 = state.get("l3") if isinstance(state.get("l3"), dict) else None
    ordered = ([l3] if l3 else []) + l2 + l1
    spans = [
        (c.get("first_turn"), c.get("last_turn")) for c in ordered
        if isinstance(c.get("first_turn"), int) and isinstance(c.get("last_turn"), int)
    ]
    highest = _highest_chunk_turn(state)
    if isinstance(legacy_watermark, int) and 0 < legacy_watermark < highest:
        w = legacy_watermark
        closed_at_w = bool(spans) and spans[-1][1] == w
        unread.update(range(max(1, w + 1 if closed_at_w else w - 1), highest + 1))
    reach = 0
    for ft, lt in spans:
        if reach and ft <= reach:
            unread.update(p for p in (ft - 2, ft - 1) if p >= 1)
        reach = max(reach, lt)
    # The closing exchange of the furthest span too, whatever its tier. This
    # also absorbs the one error the position can carry into adoption: with
    # no anchor in the file, _observed_position HOLDS where the truth may be
    # one exchange on, so the offset can read up to _ASSUMED_NEW_TURNS low
    # and window turn i land up to two positions below its own. Only the
    # last two covered positions can then receive a turn past the chain,
    # and they are never adopted.
    for lt in [c.get("last_turn") for c in l1] + [highest]:
        if isinstance(lt, int):
            unread.update(p for p in range(lt - _ASSUMED_NEW_TURNS + 1, lt + 1) if p >= 1)
    return unread


def _adopt_legacy_record(
    state: dict,
    request_turns: list[dict],
    window_offset: int,
    legacy_watermark: int | None = None,
) -> bool:
    """One-shot adoption of chunks written before v3.1.9, which recorded
    nothing. True if the record changed.

    A pre-v3.1.9 hierarchy has no evidence of what its chunks read, and her
    conversation is ~1,900 turns of such chunks: never adopting them means the
    reuse path refreshes her whole history on every request, which is the
    4-call cap refusal this feature exists to end. So the array in hand is
    believed ONCE, narrowed so that it cannot become F1 or F9 again:

      * ONCE, by the `legacy_adopted` flag, and only over the LEADING
        unrecorded run (an empty record, or the _FP_UNKNOWN padding a
        raw-less admin rebuild wrote below its own chunks).
      * only positions chunks covered BEFORE this call (maybe_rollup calls it
        ahead of its own rollups, whose chunks record themselves).
      * only the request's own turns, never the reply appended after them:
        a chunk that closed on a reply is exactly what a regenerate replaces.
      * AT ANY WINDOW OFFSET o (hostile pass #4, reviewer A F3/F8): position
        p takes request_turns[p - o - 1], and positions 1..o stay UNKNOWN. It
        used to require o == 0, and o is never 0 again once the position has
        run ahead of the array — v3.1.9 seeds the position from the highest
        chunk label, and v3.1.6.1 kept every label while pulling its
        watermark down after a delete or edit-and-resend. Upgrade inside that
        window and adoption never ran: the whole legacy span was refreshed
        on every request, for good. For a CAPPED client this reads exactly
        the part of the window the chunks cover (position o+1 is window turn
        1). For a full-history client whose array shrank it under-adopts by
        o turns, never over: those are re-read by the next L1 chunks
        (_patch_candidates).
      * never a position _legacy_unread_positions proves no chunk read,
        under either reading of the array (window turn i as position o+1+i,
        or as position i+1).

    WHAT IT STILL BELIEVES (reviewer A F2), and why that is the least bad
    option. An in-place edit made on v3.1.6.1, and a regenerate of a
    closing reply whose chunk was since folded into an L2 chapter, leave no
    trace, and are adopted as if their chunk had read them. Two alternatives
    were assessed:

      * the episodic store as evidence (adopt a position only when its text
        matches an indexed exchange). It cannot see the regenerate or the
        edit-and-resend: the store is append-only across branches, so the
        regenerated reply and the resent turn are indexed too. It does see
        an in-place edit — and it also rejects every exchange memory never
        stored, trimmed or was damaged on: before v3.1.4 a cut reply was
        not stored at all (51 Stops and 12 ceilings in one 2026-09-01 log
        window, more than half of that window's exchanges), pre-D1 rows
        were overwritten in place, and a trimmed reply matches the re-sent
        one only as a prefix, which is no evidence about the tail. Every
        rejected position is refreshed on every request until re-read, and
        at that rate the refreshed span is the cap refusal again;
      * a rebuild of the whole covered span from the current array, the
        complete fix, which is ~95 L1-sized summarization calls at her
        length and is the admin endpoint's job, not a request tail's.

    And relative to what she runs: v3.1.6.1 does not deliver an old
    correction either. At ~1,700 messages its compaction needs far more than
    MAX_SUMMARY_CALLS_PER_REQUEST batches, refuses, and the guard sheds the
    older turns; the model receives the injected hierarchy, whose chunks
    are these same summaries of the text before the correction. An adopted
    position changes what reaches the model only for a turn the guard would
    have kept verbatim — the newest few — and the closing exchanges of the
    chunks still in l1 are exactly the ones this refuses.
    """
    if state.get("legacy_adopted") or len(request_turns) < 1:
        return False
    entries = _covered_fps(state)
    lead = 0
    while lead < len(entries) and entries[lead] == _FP_UNKNOWN:
        lead += 1
    if entries and lead == 0:
        return False
    upto = _covered_prefix(state)
    if lead < len(entries):
        upto = min(upto, lead)
    if upto <= 0:
        return False
    o = max(0, int(window_offset))
    unread = _legacy_unread_positions(state, legacy_watermark)
    adopted: list[str] = []
    for p in range(1, upto + 1):
        i = p - o - 1
        if 0 <= i < len(request_turns) and p not in unread and (i + 1) not in unread:
            adopted.append(_covered_turn_fingerprint(request_turns[i]))
        else:
            adopted.append(_FP_UNKNOWN)
    state["covered_fps"] = "".join(adopted + entries[upto:])
    state["legacy_adopted"] = True
    return True


def _increasing_anchors(cand: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The longest subsequence of `cand` (request index ascending) whose
    record indices strictly increase. Patience sorting, O(n log n)."""
    tails_k: list[int] = []
    tails_i: list[int] = []
    prev = [-1] * len(cand)
    for i, (_j, k) in enumerate(cand):
        p = bisect.bisect_left(tails_k, k)
        if p > 0:
            prev[i] = tails_i[p - 1]
        if p == len(tails_k):
            tails_k.append(k)
            tails_i.append(i)
        else:
            tails_k[p] = k
            tails_i[p] = i
    out: list[tuple[int, int]] = []
    i = tails_i[-1] if tails_i else -1
    while i >= 0:
        out.append(cand[i])
        i = prev[i]
    out.reverse()
    return out


def _pairing(record: list[str], now: list[str]) -> dict[int, int]:
    """{index into `now`: index into `record`}: which request turn each
    record entry vouches for, IN ORDER, each entry used at most once.

    hostile pass #4 (reviewer A F4). The previous shape was set membership:
    any turn whose fingerprint was ANYWHERE in the record was replaced. A
    repeated "yes" to a new question, or a paste sent twice, was then
    replaced by the summary of its earlier twin in another context — for
    good when it sat where no chunk would ever read it (after a delete of the
    exchange that closed a chunk).

    ORDER-PRESERVING, AND NOT SequenceMatcher. The first cut of the content
    gate used difflib, which junks any element repeated in over 1% of a long
    sequence ("continue", "ok", identical redaction placeholders — refreshed
    on every request after any delete) and measured 679 ms at 1,990 turns.
    This is patience alignment:

      1. ANCHORS: turns whose fingerprint occurs exactly once in `now` and
         exactly once in `record`; of those, the longest run whose record
         indices increase. Her replies are long and unique, so anchors are
         dense — roughly every other turn.
      2. GAPS: between two consecutive anchors, each remaining turn pairs with
         the FIRST unused entry holding its fingerprint inside the same gap of
         the record, in order (sorted index lists + bisect). A "continue"
         between two unique replies sits in a one-turn gap and pairs; a
         "yes" whose only twin is outside its gap pairs with nothing.

    A turn is still replaced only when its own content is an entry a chunk
    wrote (safety is unchanged); the order only removes pairings. A delete
    costs nothing and an edit costs the edited turn, as before. Inside one
    very long gap with no unique turns at all, the greedy step can pair a
    turn with a later twin and leave the turns between unpaired: that costs
    refreshes, which the next L1 chunk re-reads, never a turn.
    """
    rec_idx: dict[str, list[int]] = {}
    for k, fp in enumerate(record):
        if fp != _FP_UNKNOWN:
            rec_idx.setdefault(fp, []).append(k)
    if not rec_idx:
        return {}
    now_count: dict[str, int] = {}
    for fp in now:
        now_count[fp] = now_count.get(fp, 0) + 1
    cand = [
        (j, rec_idx[fp][0]) for j, fp in enumerate(now)
        if now_count[fp] == 1 and len(rec_idx.get(fp, ())) == 1
    ]
    anchors = _increasing_anchors(cand)
    pairs: dict[int, int] = {}
    bounds = [(-1, -1)] + anchors + [(len(now), len(record))]
    for (j0, k0), (j1, k1) in zip(bounds, bounds[1:]):
        last_k = k0
        for j in range(j0 + 1, j1):
            lst = rec_idx.get(now[j])
            if not lst:
                continue
            p = bisect.bisect_right(lst, last_k)
            if p < len(lst) and lst[p] < k1:
                pairs[j] = lst[p]
                last_k = lst[p]
        if j1 < len(now):
            pairs[j1] = k1
    return pairs


def _paired_turns(record: list[str], now: list[str]) -> set[int]:
    """Indices into `now` that _pairing pairs with an entry of `record`.

    hostile pass #3 (reviewer A F2/F3/F7) made this content-based: the third
    shape compared turn N of the request with record entry N, so one delete
    shifted every later turn and the refreshed span grew until the cap
    refused it. hostile pass #4 (F4) made it ordered; see _pairing.
    """
    return set(_pairing(record, now))


def _has_image_parts(m: dict) -> bool:
    content = m.get("content")
    return isinstance(content, list) and any(
        isinstance(c, dict)
        and (c.get("type") in ("image_url", "image", "input_image") or "image_url" in c)
        for c in content
    )


def _patch_candidates(
    state: dict, raw_turns: list[dict], first_read: int
) -> list[tuple[int, int]]:
    """[(index into raw_turns, after_position)]: the turns the next L1 chunk
    re-reads as extra pieces, oldest first, at most L1_CHUNK_SIZE.

    hostile pass #4 (reviewer A F6). A turn inside the covered span that
    pairs with nothing is summarized fresh by compact_if_needed on every
    request, and nothing ever re-read it: the turns written after a delete of
    a chunk-closing exchange (+2 per delete), a regenerated closing reply
    (+1), an edited turn, an admin rebuild's placeholder positions, a legacy
    position no evidence backs. Measured growth +2/+1 per event with no decay
    across an L2 rollup; projected at her edit rate, most messages paying 3
    summarization calls in about two months and the 4-call refusal in four
    to six.

    Computed the way the gate computes it (_pairing over _record_sequence):
    every unpaired text turn BEFORE `first_read`, the first raw index this
    call's chunk reads in position order. That is the gate's refreshed set
    (unpaired turns before the last paired one) plus the unpaired turns
    between the last paired one and where the chunk starts — the turns
    written after a delete of the closing exchange sit exactly there, and
    no later chunk would ever read them. Bounding by the last paired turn
    instead left them for a second L1 cycle (measured). Nothing at or after
    `first_read` is a candidate, so one chunk never reads a turn twice. The
    chunk records each in `covered_extra` after the covered position of the
    paired turn before it, so the next request pairs them. Image turns are
    skipped (compaction never removes one).
    """
    seq, seq_pos = _record_sequence(state)
    if not seq or not raw_turns:
        return []
    pairs = _pairing(seq, _covered_turn_fingerprints(raw_turns))
    out: list[tuple[int, int]] = []
    after = 0
    for j in range(min(len(raw_turns), max(0, first_read))):
        if j in pairs:
            after = seq_pos[pairs[j]]
            continue
        m = raw_turns[j]
        if _has_image_parts(m) or not _message_text(m).strip():
            continue
        out.append((j, after))
        if len(out) >= L1_CHUNK_SIZE:
            break
    return out


def _coverage_plan(state: dict, to_summarize: list[dict]) -> tuple[int, set[int]]:
    """(covered, changed): the leading span of `to_summarize`'s non-system
    turns the reuse path may take, and the 0-based indices inside it that it
    must summarize fresh instead of replacing.

    A turn is replaced only when it is PAIRED (_pairing, order-preserving)
    with an equal entry of the record sequence (_record_sequence: the
    entries the unbroken chunk chain from turn 1 backs, plus the turns a
    chunk re-read out of position). Every entry was written from the text a
    chunk read (_record_chunk_fps, _record_patch_fps), so a replaced turn's
    content is in the stored summary. `covered` ends after the last replaced
    turn; every other turn before it is `changed`. That holds for a capped
    window, a truncated head, a delete, a regenerate, an edit and a shorter
    branch alike, which is why no length comparison guards this any more
    (F2/F3/F7): a window whose turns the record does not hold simply reads
    as changed.

    (0, set()) when there is no record (no evidence) or nothing pairs.
    Pure; hashing is memoized (F6), and the gate still runs it in the
    threadpool.
    """
    seq, _pos = _record_sequence(state)
    if not seq:
        return 0, set()
    non_system = [m for m in to_summarize if m.get("role") != "system"]
    if not non_system:
        return 0, set()
    now = _covered_turn_fingerprints(non_system)
    matched = _paired_turns(seq, now)
    if not matched:
        return 0, set()
    covered = max(matched) + 1
    return covered, set(range(covered)) - matched


def _covered_prefix(state: dict) -> int:
    """The furthest turn N such that turns 1..N are covered by an UNBROKEN
    chain of stored summaries. 0 when the chain does not start at turn 1.

    _highest_chunk_turn above answers where coverage ENDS. Nothing asked
    where it STARTS, and two shipped paths in _do_l1_rollup deliberately
    leave a hole:

      * `pos_last < 1` advances `last_summarized_turn` to `window_offset`
        and appends NO chunk. The span is gone on purpose — the text was
        never in a request and this module never held a copy — and the error
        line says so, pointing at /admin/.../compact to rebuild it.
      * `partial` records `first_turn = window_offset + 1`, deliberately
        NARROWER than `last + 1`, precisely so the chunk does not claim
        coverage of text the rollup never saw.

    Both are correct as rollup behaviour. Both are invisible to
    _highest_chunk_turn, which takes a max over `last_turn` and never looks
    at `first_turn` at all. So a reader asking "how many turns can I replace
    with stored summary text?" got a number that counted straight across the
    hole, and the reuse path in compact_if_needed deleted turns no chunk
    represents — logged as "N covered by stored summaries". An adversarial
    pass demonstrated 20 such turns.

    A gap also appears with no outage at all: load_state parks a chunk whose
    shape it cannot parse (v3.1 F1b, correct on its own terms) and the
    survivors either side are contiguous with each other but not with turn 1.

    l1 + l2 + l3. An L2 rollup CONSUMES its inputs (`state["l1"] =
    l1[L2_CHUNK_SIZE:]`) and an L3 refresh consumes every L2 chapter, so a span
    can live in any tier. The first version EXCLUDED l3 as "a claim about a
    claim" and argued that excluding it could only make this number smaller,
    which is the safe direction. It is — and it also made the number ZERO for
    every conversation after its first L3 refresh, because the refresh removes
    the L2 chapters from 1 onward and leaves l3 as the only span that starts at
    turn 1. Reuse was then off for good on exactly the conversations long
    enough to need it. l3's first_turn is inherited from the previous l3, which
    took it from l2[0] at the first refresh, so it is measured once and carried;
    and what makes a claimed span trustworthy for substitution is the covered-
    turn record (_coverage_plan), not this walk.

    Spans may overlap, nest and arrive in any order, so the walk sorts and
    takes `max` rather than requiring `first_turn == reach + 1`.
    """
    spans: list[tuple[int, int]] = []
    tiers = list(state.get("l1") or []) + list(state.get("l2") or [])
    if isinstance(state.get("l3"), dict):
        tiers.append(state["l3"])
    for c in tiers:
        if not isinstance(c, dict):
            continue
        ft, lt = c.get("first_turn"), c.get("last_turn")
        if isinstance(ft, int) and isinstance(lt, int) and ft >= 1 and lt >= ft:
            spans.append((ft, lt))
    reach = 0
    for ft, lt in sorted(spans):
        # A chunk starting past the end of what we have proven leaves a hole,
        # and everything after it is unreachable from turn 1 no matter how
        # much of it there is.
        if ft > reach + 1:
            break
        if lt > reach:
            reach = lt
    return reach


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


def recorded_position(state: dict) -> int:
    """Public wrapper over `_recorded_position`, for callers outside this
    module that need to know how far a conversation has already been
    tracked WITHOUT calling `maybe_rollup` (v3.1.9.3 / hostile317-b F3).

    backfill.py used to read this before calling `maybe_rollup` with its
    kickoff snapshot. That was check-then-act outside conv_lock (hostile pass
    #3, reviewer E F3); the comparison now happens inside maybe_rollup
    (`skip_if_position_past`). Kept for diagnostics and tests.
    """
    return _recorded_position(state)


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
            # hostile pass #5 (C5-2): this fires for ANY oversized piece
            # this function is handed, not only a literal conversation
            # turn — L3's stage 2 (_do_l3_rollup) passes a prior-L3 body or
            # a chapter-summary part here too, and "a single turn measures
            # N tokens" pointed an operator investigating an L3 loss at her
            # chat instead of at the rollup's own intermediate summaries.
            logger.warning(
                f"conv={conv_id}: a single rollup input piece measures {t} "
                f"tokens against a {budget}-token summarization budget; it "
                f"has been truncated for the rollup so the hierarchy keeps "
                f"advancing — the stored summary covers only the beginning "
                f"of that piece"
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

# v3.1.9.4 (P15-6). Every tier's prompt used to ask for content ("preserve
# names, places...") with no LENGTH target at all, so the only thing bounding
# the reply was the hard `max_tokens` cap passed to vLLM — every prompt below
# is a normal-length request TO THE MODEL, and a model that is not told to be
# brief writes until it is cut off. `_target_words` gives each tier a target
# comfortably under its own cap (see L1_MAX_TOKENS/L2_MAX_TOKENS/L3_MAX_TOKENS
# above), in WORDS rather than tokens because that is the unit a prompt can
# ask a model to reason about — tokens are a serving-side accounting unit the
# model does not see. The 0.6 factor is deliberately conservative (English
# prose runs closer to 0.75 words/token, so "0.6 * max_tokens words" leaves
# real headroom under the cap) — this is a steering hint, not a budget the
# rest of this module relies on; _llm_summarize's retry-then-trim below is
# what actually GUARANTEES a stored summary never ends mid-sentence, for the
# reply that ignores the hint anyway.
def _target_words(max_tokens: int) -> int:
    return max(40, int(max_tokens * 0.6))


_PROMPT_L1 = f"""Summarize the following conversation excerpt for long-term recall. Preserve:
- Names, places, decisions, and concrete details.
- The user's stated preferences and goals.
- Code, file paths, commands, URLs, or numeric values mentioned.
- Plot/story beats if this is creative writing.
Keep it to roughly {_target_words(L1_MAX_TOKENS)} words or fewer — well under your length limit, so you finish with a complete final sentence rather than being cut off partway through.
Do not greet, editorialize, or hedge. Output the summary only."""

_PROMPT_L2 = f"""You are summarizing several earlier per-scene summaries into one "chapter-level" summary. Preserve continuity at the chapter scale: characters, settings, decisions, ongoing threads. Drop scene-by-scene minutiae but keep names and concrete decisions. Keep it to roughly {_target_words(L2_MAX_TOKENS)} words or fewer — well under your length limit, so you finish with a complete final sentence rather than being cut off partway through. Output the chapter summary only — no preamble, no hedging."""

_PROMPT_L3 = f"""You are producing the whole-conversation "theme" summary from a list of chapter-level summaries. Capture the high-level arc, the user's overarching goals, persistent constraints, and the cast of named entities. This will be injected on every future request, so be concise but never vague. Keep it to roughly {_target_words(L3_MAX_TOKENS)} words or fewer — well under your length limit, so you finish with a complete final sentence rather than being cut off partway through. Output the theme summary only."""

# Used only by the reduce step, when one tier's input was too large for a
# single call and had to be summarized in parts. The parts are consecutive
# slices of ONE unit (one scene, one chapter, one theme), so the instruction is
# "fold", not "summarize again" — a second summarization pass is exactly the
# summary-of-summary degradation this module's tiering exists to avoid.
_PROMPT_REDUCE = """The following are consecutive partial summaries of a single stretch of one conversation, in order. Merge them into one continuous summary of that stretch. Keep every name, decision, concrete detail and numeric value that appears in any part; drop only repetition between the parts. Do not add framing, headings, or commentary. Output the merged summary only."""


# v3.1.9.4 (P15-6). `_llm_summarize` is called through one seam (`_call`,
# inside `_summarize_pieces_raw`) but MONKEYPATCHED WHOLESALE — replaced with
# a fixed-signature stub, not wrapped — by test_admin_compact.py,
# test_p3c_admin_fuzz.py, test_p4c_compact_verdict.py, test_p5_drain.py,
# test_tail_catchup.py and others. Its call signature (six positional args,
# keyword-only `timeout`) and its return type (a plain `str`) are therefore a
# contract this fix must not touch — a conv_id/tier PARAMETER here would be a
# TypeError against every one of those stubs the moment `_call` passed it,
# exactly the "a parameter threaded through a stubbed seam breaks every fixed-
# signature stub" hazard `_vllm_call_budget`'s own block comment (below)
# already names for its sibling functions. So this reads `conv_id`/tier the
# same way that budget reads its ceiling: a CONTEXTVAR (`_rollup_log_ctx`),
# set by `_call` immediately around its real call into this function and
# absent (None) for every monkeypatched test, which never reaches this body
# at all.
_SENTENCE_END_RE = re.compile(
    r"""[.!?]["'”’)\]»*_~`]*(?=\s|\Z)"""
    r"""|[。！？]["”’」』)）]*"""
)


def _trim_to_last_sentence(text: str) -> str:
    """The longest prefix of `text` ending on a sentence boundary, or ""
    when there is none — there IS no non-empty prefix that ends on a
    boundary if the text never reaches one, so "" is the correct answer to
    the question THIS function asks (it has no fallback of its own; see
    `_trim_best_effort` for the chain that keeps the hierarchy advancing
    when a cut reply never reaches a sentence boundary at all).

    A deliberately SIMPLER sibling of main.trim_to_last_sentence (that
    function's own docstring covers fence-awareness, abbreviation and
    initials handling — main.py is another lane's region here, and importing
    it from summarizer.py would be a cycle: main.py imports this module, not
    the other way around). What this module needs is narrower: the input is
    always a MODEL-WRITTEN summary (never the user's or the model's raw
    conversational text, which is where an abbreviation like "Dr." or "e.g."
    actually shows up often enough to matter), so a plain terminator-plus-
    whitespace boundary is enough to satisfy the one hard requirement this
    exists for — never store text that ends mid-sentence. Being slightly
    over-eager about what counts as a boundary (treating "Dr." as one, say)
    only means trimming a little earlier than strictly necessary, which is
    the safe direction of the same trade-off, not a correctness gap.
    """
    last = None
    for m in _SENTENCE_END_RE.finditer(text):
        last = m.end()
    return "" if last is None else text[:last]


def _trim_best_effort(text: str) -> str:
    """Never come back empty on a non-empty `text`, even when it has NO
    sentence boundary anywhere — a cut summary written as a bullet or
    numbered list (routine: see _PROMPT_L1's own "Plot/story beats" and
    "concrete details" asks, which invite exactly that shape) can run for
    hundreds of characters with no `.`/`!`/`?` at all, and
    `_trim_to_last_sentence` alone returned "" for every one of them —
    which `_llm_summarize` used to treat as "nothing usable", refusing the
    WHOLE unit. That is not a rare degenerate case, it is a routine one,
    and refusing it routinely is the exact stall this fix exists to avoid.

    Fallback chain, best first, each strictly safer to trim to than the one
    before it:
      1. `_trim_to_last_sentence` — a real sentence boundary, unchanged.
      2. The longest prefix ending on a LINE break. A cut bullet/numbered
         list still has complete lines up to the one the cut landed inside;
         the boundary is honest (nothing on either side of the cut is
         invented), it just is not a sentence.
      3. The longest prefix ending on WHITESPACE (the last complete word),
         with " …" appended — explicit rather than silent, so nothing
         downstream mistakes this for a naturally short, complete summary.
         The one case with no internal whitespace at all (a single unbroken
         token) returns that whole token with " …" appended: there is no
         smaller safe boundary to cut to, and it is still better than "".

    Returns "" only when `text` itself is empty or whitespace-only — the
    same "nothing was said at all" case an empty reply already is, and the
    ONLY case `_llm_summarize` still refuses the unit for.
    """
    if not text.strip():
        return ""
    by_sentence = _trim_to_last_sentence(text)
    if by_sentence:
        return by_sentence
    lines = text.split("\n")
    if len(lines) > 1:
        by_line = "\n".join(lines[:-1]).rstrip()
        if by_line:
            return by_line
    words = text.split()
    if len(words) > 1:
        return " ".join(words[:-1]) + " …"
    return text.strip() + " …"


def _retry_suffix(target_words: int) -> str:
    """The one line added to the system prompt on the single retry a cut
    reply gets — see _llm_summarize's block comment for why the retry keeps
    the SAME max_tokens and changes only this instruction."""
    return (
        "\n\nYour previous attempt at this ran past its length limit and "
        "was cut off mid-sentence. This time, keep it to roughly "
        f"{target_words} words or fewer — well under the limit — and make "
        "sure your LAST sentence is complete."
    )

# v3.1.9.4 (P15-6). How many rollup summary calls (any tier, any conv) were
# cut at max_tokens and STILL had nothing better than a fallback-trimmed
# result after the one retry (or after skipping it for lack of budget) — see
# _trim_best_effort — i.e. how many times a stored summary is the model's
# output trimmed to a sentence, line or word boundary rather than its full
# intended text. Zero on a healthy deployment; a number that climbs is the
# operator-visible signal this defect had NONE of before (the finding's own
# words: "no log line, no counter"). Process-local and reset on restart, the
# same scope tokenhealth's counters have — see tokenize_health() above for
# the established pattern of exposing a module counter for main.py/health.py
# (another lane's region) to surface at /health/full; nothing in this module
# reads it back.
_truncated_summary_calls = 0


def truncated_summary_count() -> int:
    """How many _llm_summarize calls, across every tier, were cut at
    max_tokens and still had no better than a fallback-trimmed result after
    the retry (or after the retry was skipped for lack of budget) — see the
    block comment above `_llm_summarize`. A caller in health.py can surface
    this; this module does not read it back itself."""
    return _truncated_summary_calls


_rollup_log_ctx: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "summarizer_rollup_log_ctx", default=None
)


def _tier_of(system_prompt: str) -> str:
    """Which tier a system prompt belongs to, for logging only. The four
    prompts below are the only ones this module ever passes to
    _llm_summarize, so an exact string match is unambiguous; anything else
    (only reachable if a future caller adds a fifth) logs as "?" rather than
    raising over a log line. Same purpose, same shape, as test_p5_drain.py's
    own `_tier_of` helper."""
    return {
        _PROMPT_L1: "L1", _PROMPT_L2: "L2", _PROMPT_L3: "L3",
        _PROMPT_REDUCE: "reduce",
    }.get(system_prompt, "?")


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
    """Summarize once, and never hand back text a `finish_reason=length`
    reply cut mid-sentence — siblings that already refuse this outright:
    facts.py's extraction call ("A reply cut off at _EXTRACTION_MAX_TOKENS
    ends mid-line ... a truncation which does not announce itself becomes a
    wrong fact") and dedup.py's merge call (refused as "truncated"). This
    tier could not simply copy that refusal: unlike a single fact or a single
    merge decision, a REFUSED rollup unit does not retry in isolation — see
    _summarize_pieces_raw's map-reduce, where ANY empty batch fails the whole
    tier and the watermark does not advance for it. If cut summaries turn out
    to be ROUTINE at these caps (the finding's real-data indication: most of
    her live L1/L2 summaries end without terminal punctuation), refusing
    every one of them would routinely stall the hierarchy — worse than the
    defect this fixes. So the shape is graceful, not a hard refusal:

      1. One real call at the caller's max_tokens.
      2. If it finished normally (`finish_reason` anything but "length"),
         return it untouched — this is the ordinary path and it is unchanged.
      3. If it was cut, retry ONCE at the SAME max_tokens — the cap IS the
         real budget (what vLLM is actually willing to generate against this
         request); LOWERING it on retry guarantees nothing, because the
         system prompt still carries its ORIGINAL word target
         (_target_words(max_tokens), roughly 0.8x max_tokens in real
         tokens), so a tighter cap and an unchanged instruction just
         contradict each other and the retry gets cut too, one sentence
         earlier, for no reason. What actually changes on retry is the
         INSTRUCTION: `_retry_suffix` asks for roughly HALF the tier's
         ordinary word target, at the SAME cap — an achievable ask instead
         of a shrunk budget the prompt was never told to fit. Skipped
         entirely if the shared vLLM-call budget (`_vllm_call_budget`, set
         by a caller via `vllm_call_budget_ctx`/`maybe_rollup`'s own
         parameter) has nothing left: `_call`'s own decrement only ever
         accounts for the ONE guaranteed call, so this function decrements
         the same budget itself for the retry — an uncounted second call
         would silently let a caller's `{"max_calls": N}` spend N+1 real
         calls. No `await` between the check and the decrement, matching
         `_call`'s own reasoning for why that is race-safe under concurrent
         map-phase callers.
      4. If the retry ALSO finished normally, return it (a shorter but
         complete summary — better than the fuller-but-cut first attempt).
      5. Otherwise (the retry was cut too, OR was skipped for budget): this
         is where the finding's defect used to land silently. Trim EVERY
         candidate actually in hand (`text`, and `retry_text` if a retry
         ran) with `_trim_best_effort` — the fallback chain that finds a
         line or word boundary when there is no sentence boundary at all —
         and keep whichever TRIMMED result is LONGER, not automatically the
         retry: the tighter retry target sometimes produces a cut reply
         that trims to LESS usable content than the first attempt trimmed
         would have, and always preferring "the newest attempt" would throw
         that material away for no reason. Log a WARNING naming the
         conversation and tier (from `_rollup_log_ctx`, set by `_call`) and
         count it (truncated_summary_count). The unit still advances — it
         just covers slightly less than the model tried to say.
      6. The one case this does NOT paper over: EVERY candidate in hand is
         itself empty or whitespace-only (a reply that said nothing at
         all — `_trim_best_effort` returns "" only for that). That is the
         same "reply carries no usable content" shape facts.py and dedup.py
         already refuse, so this returns "" too — the tier fails this unit
         and retries next turn, same as an empty reply always has. This is
         NOT the same case as "no sentence boundary": a cut bullet list is
         routine and _trim_best_effort keeps it (falling back to a line or
         word boundary); only a genuinely content-free reply reaches "".
    """
    async def _one_call(extra_system: str = "") -> tuple[str, str | None]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt + extra_system},
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
            # surface as a bare IndexError inside maybe_rollup's blanket
            # handler — a stack trace per turn that named the wrong thing.
            # Say what happened. Same guard main._summarize_once already
            # carries.
            raise ValueError(
                f"vLLM returned no choices for a rollup summarize: {str(data)[:200]}"
            )
        choice = choices[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        return text, choice.get("finish_reason")

    text, finish_reason = await _one_call()
    if finish_reason != "length":
        return text

    # Cut. Retry ONCE at the same max_tokens (see the docstring above for
    # why lowering it would not help), gated on the shared vLLM-call budget
    # actually having room for a second real call.
    retry_text: str | None = None
    retry_finish: str | None = None
    _budget = _vllm_call_budget.get()
    if _budget is None or _budget["remaining"] > 0:
        if _budget is not None:
            _budget["remaining"] -= 1
        retry_words = max(20, _target_words(max_tokens) // 2)
        retry_text, retry_finish = await _one_call(_retry_suffix(retry_words))
        if retry_finish != "length":
            return retry_text

    # Still cut (or no budget left for a retry at all). Trim every candidate
    # actually in hand and keep the longer trimmed result.
    candidates = [text] + ([retry_text] if retry_text is not None else [])
    trimmed = [_trim_best_effort(c) for c in candidates]
    best_idx = max(range(len(trimmed)), key=lambda i: len(trimmed[i]))
    best = trimmed[best_idx]
    if not best:
        # Every candidate was itself empty or whitespace-only — nothing
        # usable anywhere, not even a fallback boundary to trim to.
        return ""

    global _truncated_summary_calls
    _truncated_summary_calls += 1
    _ctx = _rollup_log_ctx.get() or {}
    _retry_note = (
        "retried at the same cap with a tighter word target, still cut"
        if retry_text is not None
        else "no vLLM-call budget left for a retry"
    )
    logger.warning(
        f"conv={_ctx.get('conv_id', '?')}: {_ctx.get('tier', '?')}-tier rollup "
        f"summary was cut at max_tokens={max_tokens} ({_retry_note}) — kept "
        f"trimmed to {len(best)} of {len(candidates[best_idx])} chars rather "
        f"than stored past where the model stopped"
    )
    return best


# ---------------------------------------------------------------------------
# v3.1.9 (hostile pass 4, F5). A budget on REAL vLLM summarization calls
# (_llm_summarize invocations), independent of how many rollup PASSES the
# caller makes (maybe_rollup, called once per pass) and independent of how
# many tiers or map-reduce batches one pass touches internally. Before this,
# `max_calls` on /compact counted calls to maybe_rollup itself, and ONE
# maybe_rollup call drains every L1 and L2 tier that is due in its own
# internal `while` loops — `{"max_calls": 1}` on a deep backlog still ran
# however many vLLM calls the whole backlog needed, not one.
#
# A CONTEXTVAR, not a parameter threaded through every intermediate function
# (_do_l1_rollup, _do_l2_rollup, _do_l3_rollup, _summarize_pieces,
# _summarize_pieces_raw): every one of those five is monkeypatched with a
# fixed-signature stub somewhere in this test suite (nine files, at last
# count — test_compaction_reuse.py, test_l3_coverage.py,
# test_p3a_reuse_endpoint.py, test_p3a_reuse_traffic.py,
# test_p4a_reuse_order.py, test_review_fixes.py, test_soak_conversation.py,
# test_time_memory.py, and main.admin_compact's own caller stubs
# maybe_rollup wholesale in test_admin_compact.py [5c]). PLUS `maybe_rollup`
# ITSELF, stubbed wholesale (not just one of the five below it) by at least
# two more callers with their own fixed signature: test_admin_compact.py
# again, and — found the hard way, by the v3.1.9 tail-catch-up feature
# passing `vllm_call_budget=` as a keyword on the TAIL's own call to
# maybe_rollup and breaking it — test_degenerate_skip.py's
# `spy_maybe_rollup(cid, messages, vllm_url, model, *, raw_messages=None)`.
# main._rollup_hierarchy uses the context-manager form for exactly the
# reason main.admin_compact already did (see that function's own comment on
# the point): a keyword added to the maybe_rollup CALL breaks any stub of
# maybe_rollup ITSELF, not just of what it calls internally. Adding a keyword
# argument to any of their signatures breaks every stub that does not also
# grow that keyword — which is every one of them, since none takes
# `**kwargs`. A contextvar needs no call-site change anywhere in that chain:
# it is read at the one real HTTP-call site (`_call`, inside
# `_summarize_pieces_raw`, for accounting) and at the L1/L2/L3 drain in
# `_maybe_rollup_body` (via `_budget_allows_unit`, for the unit-boundary
# gate itself — v3.1.9, tail catch-up), and a stub that replaces anything
# ABOVE `_call` in the chain never reaches ITS read at all — exactly
# correct, because a stub that does not make real vLLM calls has nothing to
# bound. `_maybe_rollup_body` is not one of the nine stubbed functions
# above, so the gate's own call site needed no signature change either.
#
# maybe_rollup's own `vllm_call_budget` PARAMETER (see its docstring) sets
# this for the duration of its call, for a caller that CAN pass a keyword.
# `vllm_call_budget_ctx` is the lower-level context-manager form for a
# caller that must keep calling maybe_rollup with today's exact signature
# (main.admin_compact, for the stub-compatibility reason above) — both set
# the same underlying mechanism, and either can be read back afterward for
# how many calls were actually spent and whether the budget ran out.
_vllm_call_budget: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "summarizer_vllm_call_budget", default=None
)


@contextlib.contextmanager
def vllm_call_budget_ctx(max_calls: int):
    """Bound the number of REAL vLLM summarization calls a `maybe_rollup`
    call spends, to AT MOST `max_calls` PLUS one unit's own call cost —
    never a hard per-call ceiling. (hostile pass #5, E9: this docstring
    used to promise per-call refusal; the tail catch-up feature moved the
    gate to the UNIT boundary, and this is the corrected contract.)

    Yields the mutable dict `{"remaining": int, "exhausted": bool}`; read it
    after the block to see how many calls are left (0 or negative if the
    budget was spent, possibly past zero — see the overshoot note below)
    and whether anything was still due when the drain stopped checking.
    `remaining` is decremented once per REAL call to `_llm_summarize`,
    wherever in the L1/L2/L3 drain (including a map-reduce split within any
    one tier) it happens — never once per rollup pass or per tier, both of
    which can spend zero-or-more real calls.

    THE GATE IS AT THE UNIT BOUNDARY, NOT PER CALL. The only reader that
    refuses anything is `_budget_allows_unit`, checked by the L1/L2/L3
    drain in `_maybe_rollup_body` immediately before a unit (one L1 chunk,
    one L2 fold, the L3 refresh) is allowed to START — never inside
    `_call` itself, which only decrements. A unit that is allowed to start
    is GUARANTEED to finish, so the true overshoot on a call that spends
    down to (or past) zero is AT MOST one unit's own calls, not zero — see
    `_budget_allows_unit`'s own docstring for why a strict per-call refusal
    livelocked a budget smaller than one unit's cost. A DIRECT call into
    `_do_l1_rollup`/`_do_l2_rollup`/`_do_l3_rollup`, or into
    `_summarize_pieces`/`_summarize_pieces_raw`, from inside this block
    is NOT bounded at all — those functions do not check
    `_budget_allows_unit` themselves, only `_maybe_rollup_body`'s drain
    does, so a caller reaching for a bounded direct tier call must go
    through `maybe_rollup` (or replicate the unit-boundary check itself).

    A state mutation for a chunk/chapter/refresh is only ever written AFTER
    its summarize call returns non-empty text (see _do_l1_rollup,
    _do_l2_rollup, _do_l3_rollup) — an LLM failure must not record a chunk
    it did not produce — so a call that ran out of budget mid-unit and
    still finished (the guaranteed-finish trade above) either records real
    progress or records nothing, never a chunk covering text it did not
    summarize. A resumed call (the next /compact request, or the next
    chat-path tail) re-reads state from disk and continues from the real
    watermark, same as any other partial-progress rollup already does.
    """
    budget = {"remaining": max_calls, "exhausted": False}
    token = _vllm_call_budget.set(budget)
    try:
        yield budget
    finally:
        _vllm_call_budget.reset(token)


# v3.1.9.4 (R1 / P15-5 follow-up). SAME shape as _vllm_call_budget just
# above, and for the identical reason: maybe_rollup is monkeypatched
# WHOLESALE by test doubles this module does not own, so a new keyword on
# ITS signature would TypeError every one of them. A contextvar set around
# the call, read by _maybe_rollup_body itself (not by maybe_rollup's thin
# wrapper — this one has no per-call decrement to own, so there is nothing
# for a wrapper layer to do), survives that: a stub which replaces
# maybe_rollup wholesale never reads it, which is correct — a stub making
# no real writes has nothing to discard.
_wipe_generation: "contextvars.ContextVar[int | None]" = contextvars.ContextVar(
    "summarizer_wipe_generation", default=None
)


@contextlib.contextmanager
def wipe_generation_ctx(generation: int | None):
    """Set the conversation's wipe generation, as captured by the caller at
    the moment its background tail was SUBMITTED, for the duration of one
    `maybe_rollup` call. `_maybe_rollup_body` reads it back (via
    `_wipe_generation.get()`) immediately after loading state, under
    conv_lock, and discards the ENTIRE rollup — no tier runs, no chapter
    archive write, no save_state — if it disagrees with
    `memory.current_wipe_generation(conv_id)` at that moment: a /forget (or
    any other wipe) that ran after this tail was submitted must not have
    its deletion undone by a rollup that reads stale content and writes it
    back.

    `generation=None` (main.admin_compact, which calls `maybe_rollup`
    directly with no context manager at all and leaves this at its
    contextvar default of None) disables the check entirely — an admin
    compact is not a background tail racing a wipe; an operator is waiting
    on it. backfill.py's rollup DOES set it (v3.1.9.4 W2): a backfill runs
    in the background for up to hours and is exactly the stale work a wipe
    must be able to stop. A caller inside this SAME `with` block that itself passes
    `generation=None` (a tail that never captured one — see
    `main._facts_tail`'s identical `wipe_generation: int | None = None`
    convention) gets the same opt-out, for the same reason: nothing
    changes for a direct test call or a caller not participating.
    """
    token = _wipe_generation.set(generation)
    try:
        yield
    finally:
        _wipe_generation.reset(token)


def _budget_allows_unit() -> bool:
    """True if the current vLLM call budget (if any) has room to START a new
    rollup UNIT — one L1 chunk, one L2 fold, or the L3 refresh.

    v3.1.9 (tail catch-up). Checked ONCE per unit, immediately BEFORE that
    unit's first real vLLM call, by the L1/L2/L3 drain in
    `_maybe_rollup_body` — never per real call within a unit. Per-call was
    F5's original shape (see the check this replaced in
    `_summarize_pieces_raw`'s `_call`, and the comment left there): it
    could return "" to any ONE map batch inside a unit that had already
    spent calls, and `_summarize_pieces_raw`'s map-reduce already treats
    ANY empty map batch as a whole-unit failure — correct, elsewhere, for a
    real LLM failure, but here it meant a budget smaller than one unit's
    own call cost NEVER advanced: a 20-turn L1 chunk needing 2 map batches
    + 1 reduce = 3 calls is ordinary at L1_MAX_TOKENS=500 (more under the
    pessimistic /tokenize-down scale), so a budget of 1 or 2 hit the same
    exhausted-mid-chunk failure, recorded nothing (by design — see
    _summarize_pieces_raw), and represented the IDENTICAL too-expensive
    chunk again next turn. Not slow progress: LIVELOCK, forever, on any
    backlog whose chunks cost more than the configured budget.

    Gating at the unit boundary instead means every unit that is ALLOWED to
    start is GUARANTEED to finish — `remaining` is spent, possibly past
    zero, but a started unit is never refused mid-flight. The documented
    cost is an overshoot of AT MOST one unit's own calls: once `remaining`
    has reached zero or below, the very NEXT check (on the next unit, not
    this one) sees `remaining <= 0` and refuses, so only one unit per call
    can ever run past the budget, never an unbounded number of them.

    `remaining <= 0` refuses cleanly when the caller opened the budget at
    ZERO on purpose — admin /compact's documented `{"max_calls": 0}` ("run
    the guards, make no real calls", hostile pass 2 MEDIUM) — because 0 is
    never `> 0`, so the first unit is never allowed to start either. The
    tail's own default (COMPACTOR_TAIL_ROLLUP_MAX_CALLS=4, main.py) is
    never 0 unless an operator sets it that way on purpose, in which case
    it means the same thing admin /compact's 0 already does.
    """
    budget = _vllm_call_budget.get()
    if budget is None:
        return True
    if budget["remaining"] > 0:
        return True
    budget["exhausted"] = True
    return False


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
        # F5: the ONE real HTTP-call site every tier's every batch goes
        # through — see the block comment above _vllm_call_budget for why
        # the accounting lives here and nowhere else in the chain.
        #
        # v3.1.9 (tail catch-up): the REFUSAL that used to live here —
        # return "" once `remaining <= 0` — moved to the UNIT boundary
        # (`_budget_allows_unit`, checked by the L1/L2/L3 drain in
        # `_maybe_rollup_body` before a unit starts, not here). See that
        # function's docstring for why: refusing mid-unit is exactly the
        # shape that livelocked a budget smaller than one unit's call cost,
        # because this map-reduce already treats any empty map batch as a
        # whole-unit failure. A unit that was allowed to start now always
        # finishes; `remaining` still decrements for every real call this
        # unit makes, including past zero (the documented overshoot), so
        # the accounting `vllm_call_budget_ctx` promises ("remaining
        # decremented once per real call") holds unchanged — only the
        # refusal moved.
        #
        # No `await` between the read and the decrement, so concurrent
        # map-phase callers (asyncio.gather below) cannot race past each
        # other onto the same unit of budget — asyncio only yields at an
        # `await`.
        _vllm_budget = _vllm_call_budget.get()
        if _vllm_budget is not None:
            _vllm_budget["remaining"] -= 1
        # P15-6: conv_id/tier for _llm_summarize's own WARNING and counter on
        # a cut-then-still-cut reply, via a contextvar for the identical
        # reason `_vllm_call_budget` is one — see _llm_summarize's own block
        # comment. Set only around the real call, so a stub that replaces
        # _llm_summarize wholesale never observes it either way.
        _log_token = _rollup_log_ctx.set({"conv_id": conv_id, "tier": _tier_of(prompt)})
        try:
            return await _llm_summarize(
                client, vllm_url, model, prompt, "\n\n".join(batch), max_tokens
            )
        finally:
            _rollup_log_ctx.reset(_log_token)

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
    #
    # v3.1.9 (hostile pass #5, C5-2). A give-up here (every batch already a
    # singleton -- the parts on hand do not fit TOGETHER under `budget`,
    # routine whenever /tokenize is down) used to concatenate on the spot.
    # That let a tier's own output grow by roughly one part's width every
    # time this ran, and a caller that feeds its own output back in as a
    # LATER input (_do_l3_rollup's prior-L3 fold is the one that does)
    # compounded it further, refresh after refresh, without limit -- the
    # reduce's own "give up and concatenate" being the source of the
    # unbounded growth a downstream truncate-to-budget was then silently
    # cutting back down (C5-2's original finding). Now a give-up first
    # tries PAIRING adjacent parts and folding two at a time, TOLERATING
    # that a pair may price over `budget` under the pessimistic per-char
    # estimate: `_WORST_TOKENS_PER_CHAR` is a worst-case INPUT-budgeting
    # guess, not a measurement of what the model's REAL context window
    # (MAX_MODEL_LEN) can actually hold, and two of this reduce's own
    # bounded-output parts (each capped at `max_tokens` real model tokens)
    # fit the real window far more often than the pessimistic estimate
    # admits. Pairwise folding converges to ONE part in ceil(log2(N))
    # rounds regardless of how oversized the pessimistic estimate makes
    # each part look. This only changes behaviour ON THE GIVE-UP PATH -- a
    # healthy /tokenize essentially never reaches it, so L1/L2's ordinary
    # folding (and L3's, when /tokenize is up) is unaffected. If pairing
    # still leaves one part with nothing left to fold against (an odd
    # leftover with no partner), that is as far as this reduce can bring
    # it, and it is concatenated same as before -- see the round-count
    # ceiling below, sized for pairwise convergence rather than the
    # smaller ordinary case.
    rounds = 0
    max_rounds = max(3, len(parts).bit_length() + 1)
    while len(parts) > 1 and rounds < max_rounds:
        rounds += 1
        groups = await _batch_to_budget(
            conv_id, client, vllm_url, model, parts, budget
        )
        if all(len(g) == 1 for g in groups):
            groups = [parts[i:i + 2] for i in range(0, len(parts), 2)]
            if all(len(g) == 1 for g in groups):
                # Only reachable with a single leftover part and nothing
                # to pair it against -- already as folded as it gets.
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
    raw_turns: list[dict] | None = None,
    patch: list[tuple[int, int]] | None = None,
) -> bool:
    """Roll the next L1_CHUNK_SIZE turns after last_summarized_turn into a
    new L1 chunk. Returns True if the watermark advanced.

    `raw_turns`, if given, is the non-system turns of `messages` as the
    client sees them, index for index (maybe_rollup builds and checks it).
    This chunk's covered-turn record is written from it, or from `messages`
    itself when it is None — see _record_chunk_fps.

    `patch`, if given (with raw_turns), is [(non-system index, after
    position)] from _patch_candidates: older turns inside the covered span
    that pair with nothing. The chunk reads them as extra pieces ahead of its
    own turns and records them out of position (_record_patch_fps) in the
    same mutation. Its label, and so the tiling, L2 and L3, are unchanged.

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
    # hostile pass #4 (reviewer A F6): the older unpaired turns this chunk
    # re-reads, AHEAD of its own turns because they are older. Read from
    # `messages` (the rollup view: a loop redacted exactly as it would be at
    # its own position), recorded from `raw_turns` below, the same split as
    # the chunk's own turns. Only with raw_turns: without the client's text
    # (the admin rebuild) there is nothing to pair a re-read against.
    _ns_msgs = [m for m in messages if m.get("role") != "system"]
    _patch = [
        (j, after) for j, after in (patch or [])
        if raw_turns is not None and 0 <= j < min(len(_ns_msgs), len(raw_turns))
        and j < pos_first - 1
    ]
    if _patch:
        pieces = [
            f"[{_ns_msgs[j].get('role', 'unknown')}] (an earlier turn, as it "
            f"reads now): {_message_text(_ns_msgs[j])}"
            for j, _a in _patch
        ] + pieces
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
    # THE RECORD IS WRITTEN WITH THE CHUNK (hostile pass #3, F1/F9): in the
    # same mutation, from the turns `pieces` came from — positions
    # covered_first..last_turn are array turns pos_first..pos_last, through
    # the same window_offset that chose the text. Never from a later request:
    # that is what blessed a deleted or regenerated turn (F1) and every
    # position an admin rebuild summarized from placeholders (F9).
    #
    # From `raw_turns` when the caller has the client's own text, because
    # `messages` is the rollup input — degenerate replies redacted — which no
    # request carries, and recording that switched reuse off on ordinary
    # traffic (the 2026-09-12 soak). The raw text carries ONE documented
    # trade: a reply reply_is_degenerate calls a loop is read as its clean
    # sentence head (or the placeholder when it has none), so what follows
    # its last sentence boundary is replaced by a summary that omitted it
    # (main._redact_degenerate_turns). A STOPPED reply is no longer a second
    # trade (hostile pass #4, reviewer A F1): the tail appends it as it
    # streamed, so the chunk that closes on it reads the tail it records.
    _src = (
        raw_turns if raw_turns is not None
        else [m for m in messages if m.get("role") != "system"]
    )
    _record_chunk_fps(
        state, covered_first, last_turn, _src[pos_first - 1:pos_last]
    )
    if _patch:
        _record_patch_fps(
            state, last_turn,
            [(after, _covered_turn_fingerprint(raw_turns[j])) for j, after in _patch],
        )
        logger.info(
            f"conv={conv_id}: the L1 chunk for turns {covered_first}-"
            f"{last_turn} also re-read {len(_patch)} earlier turn(s) that no "
            f"stored summary held as they read now; they are reused from "
            f"the next request on instead of summarized fresh on every one"
        )
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
    # v3.1.9 (A3-1). A READ-MODIFY-WRITE, so the fallback here was not a
    # misread but a deletion: a wrong-type "chapters" became [] and the
    # atomic_write_json below replaced the whole cold chapter store with
    # this refresh's rows. After L3 has paraphrased a span this sidecar is
    # the ONLY copy of its chapter-level detail. Raising aborts the L3
    # refresh before it consumes anything — its caller already treats any
    # archive failure that way — so the chapters stay in l2, uncompressed
    # and intact, until someone looks at the file.
    rows = existing.get("chapters", [])
    if not isinstance(rows, list):
        raise StoreUnreadable(
            path, TypeError(f'"chapters" is {type(rows).__name__}, not a list')
        )
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
    # v3.1.9 (A3-1): the same rule as _archive_chapters above and
    # facts.load_archive — an operator asking for the cold chapters must be
    # told the file is unreadable, not that there are none.
    rows = data.get("chapters", [])
    if not isinstance(rows, list):
        raise StoreUnreadable(
            summary_archive_path(conv_id),
            TypeError(f'"chapters" is {type(rows).__name__}, not a list'),
        )
    return [r for r in rows if isinstance(r, dict)]


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
        # hostile pass #5 (C5-2): BOTH halves of stage 2's input are split
        # back into pieces on their own "\n\n" join points before being
        # handed to stage 2's map-reduce — not just the newer-chapters
        # half. Stage 1's own map-reduce (_summarize_pieces_raw, shared
        # with every tier) already gives up and CONCATENATES its parts,
        # with "\n\n" as the join, whenever its reduce cannot fold them
        # further — routine whenever /tokenize is down and the chapter
        # count is not tiny (measured: 2-3 parts, each already at or near
        # L3_MAX_TOKENS, so the concatenation is 2-3x one part's own
        # bound). The first cut of this fix split only `text` (this
        # refresh's newer-chapters half) and left `prior_piece` (the
        # PREVIOUS refresh's stored L3 text) as one atomic string — which
        # still broke, one refresh later: a give-up concatenation is
        # exactly what gets STORED as state["l3"]["text"] below, so the
        # NEXT refresh's `prior` is frequently ALREADY an oversized
        # multi-part blob, and wrapping THAT as one piece is the identical
        # "one blob priced as a whole and hard-truncated" failure, just
        # moved from the newer half to the prior half (measured: 0/4
        # refreshes truncated with only the newer half split; 3/4 with
        # both, once a give-up concatenation had a chance to compound).
        # Splitting BOTH halves the same way is symmetric and self-
        # healing across any number of refreshes: whatever shape a PRIOR
        # refresh's own give-up left in storage, splitting on its join
        # recovers pieces that individually fit the budget by
        # construction (each is itself the bounded output of one map or
        # reduce call), so nothing is ever handed to `_batch_to_budget` as
        # a piece larger than one of those calls could have produced —
        # not on this refresh, and not on any refresh after it.
        #
        # No harm in the ordinary case: an unsplit single-call answer that
        # happens to contain its own blank-line paragraph breaks is only
        # split into SMALLER pieces, which `_batch_to_budget` batches back
        # together under budget the same way it always groups any
        # ordinary multi-piece input. This is a split of TEXT ALREADY
        # PRODUCED (an ordinary string operation on `prior["text"]`), not
        # a second seam alongside `_summarize_pieces` — test_l3_
        # coverage.py's own monkeypatch of that ONE name is what every
        # caller of this function must keep working through.
        _prior_body = (prior.get("text") or "").strip() if prior else ""
        _prior_parts = (
            [p for p in _prior_body.split("\n\n") if p.strip()] or [_prior_body]
        )
        _prior_header = (
            f"the story so far (turns {prior.get('first_turn','?')}-"
            f"{prior.get('last_turn','?')})" if prior else "the story so far"
        )
        _prior_pieces = [
            (
                f"--- {_prior_header} (part {i} of {len(_prior_parts)}) ---"
                if len(_prior_parts) > 1 else f"--- {_prior_header} ---"
            ) + f"{chr(10)}{p}"
            for i, p in enumerate(_prior_parts, 1)
        ]
        _newer_parts = [p for p in text.split("\n\n") if p.strip()] or [text]
        _newer_pieces = [
            (
                f"--- newer chapters (part {i} of {len(_newer_parts)}) ---"
                if len(_newer_parts) > 1 else "--- newer chapters ---"
            ) + f"{chr(10)}{p}"
            for i, p in enumerate(_newer_parts, 1)
        ]
        text = await _summarize_pieces(
            conv_id, client, vllm_url, model, _PROMPT_L3,
            _prior_pieces + _newer_pieces,
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
    *,
    raw_messages: list[dict] | None = None,
    reply_as_streamed: str | None = None,
    skip_if_position_past: int | None = None,
    skipped_at: list | None = None,
    vllm_call_budget: dict | None = None,
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

    `raw_messages` is the request exactly as the client sent it, WITHOUT the
    reply the live tail appends to `messages`. `reply_as_streamed` is that
    reply as the client received it, when it differs from the text appended
    (a stopped reply is trimmed to its last sentence for memory; OpenWebUI
    keeps and re-sends what streamed). Together they are what the covered-turn
    record is written from (_record_chunk_fps). With no `raw_messages` (the
    admin rebuild) the record is written from `messages` itself.

    `skip_if_position_past`: return the loaded state untouched, writing
    nothing, if the conversation's recorded position is already past it, and
    append that position to `skipped_at`. For a caller holding a SNAPSHOT
    (the backfill): the comparison has to be made here, under conv_lock after
    the load, because one made before taking the lock let a live tail queued
    on it run in between (hostile pass #3, reviewer E F3).

    `vllm_call_budget`: optional mutable {"remaining": int, "exhausted":
    bool} (v3.1.9, hostile pass 4, F5). Bounds REAL vLLM summarization
    calls made during THIS call, across L1/L2/L3 and any internal
    map-reduce split — not rollup passes (a caller making several
    maybe_rollup calls decides its own pass count) and not tiers (one tier
    can spend zero calls, if nothing is due, or several, if its input maps
    to more than one batch). "remaining" is decremented once per real
    call; "exhausted" is set True if the budget ran out before every tier
    that needed a rollup got one. None (the default) is unlimited: today's
    behaviour, byte-for-byte — a caller that does not pass this sees no
    change at all. See `vllm_call_budget_ctx` (above _summarize_pieces)
    for the equivalent context-manager form, for a caller that cannot add
    a keyword to ITS OWN call to this function because something upstream
    of it stubs this function WHOLESALE in a test with a fixed-argument
    signature: main.admin_compact (test_admin_compact.py's own stub) and
    main._rollup_hierarchy (v3.1.9, tail catch-up — test_degenerate_skip.py's
    `spy_maybe_rollup`) both use the context-manager form for exactly this
    reason. backfill.py's one call (v3.1.9, tail catch-up) passes this
    keyword directly instead: nothing in its own test coverage stubs
    maybe_rollup wholesale, so it has no fixed signature to preserve.

    v3.1.9 (tail catch-up): the budget is checked at the UNIT boundary —
    before each L1 chunk, L2 fold, or the L3 refresh starts — not before
    each real call within one. See `_budget_allows_unit`'s docstring for
    why: a per-call check livelocks whenever a unit's own call cost (a
    20-turn L1 chunk needing map-reduce is ordinarily 2-3 calls) exceeds
    the budget, because this module's map-reduce already fails the WHOLE
    unit on any single empty batch, and an exhausted-mid-unit refusal is
    indistinguishable from a real LLM failure to that check. So a unit
    that is allowed to START always FINISHES — "remaining" can go
    negative, documenting an overshoot of at most one unit's own calls,
    never more, because the NEXT unit's boundary check sees the negative
    balance and refuses. This is what turns "process a bounded number of
    calls per turn" into a guarantee that a tail with work due always
    completes at least one whole L1 chunk (or one L2 fold, or the L3
    refresh, when no L1 chunk is due) — the property a caller bounding
    per-turn work over a persistent backlog actually needs.
    """
    _budget_token = None
    if vllm_call_budget is not None:
        _budget_token = _vllm_call_budget.set(vllm_call_budget)
    try:
        return await _maybe_rollup_body(
            conv_id, messages, vllm_url, model,
            raw_messages=raw_messages,
            reply_as_streamed=reply_as_streamed,
            skip_if_position_past=skip_if_position_past,
            skipped_at=skipped_at,
        )
    finally:
        if _budget_token is not None:
            _vllm_call_budget.reset(_budget_token)


async def _maybe_rollup_body(
    conv_id: str,
    messages: list[dict],
    vllm_url: str,
    model: str,
    *,
    raw_messages: list[dict] | None = None,
    reply_as_streamed: str | None = None,
    skip_if_position_past: int | None = None,
    skipped_at: list | None = None,
) -> dict:
    """The actual rollup logic, unchanged by F5's split — maybe_rollup
    (above) is now a thin wrapper that sets/resets the vLLM call budget's
    contextvar around this call and otherwise passes every argument
    through untouched. Split out rather than wrapping the body inline so
    the diff for F5 is "one function extracted, one small wrapper added",
    not a full reindent of ~170 lines under a new try/finally.
    """
    async with conv_lock(conv_id):
        # OFF THE EVENT LOOP (v3.1.9.2). Benchmarked on the v3.1.9 harness:
        # this read and the save_state below cost 3.8-4.9 ms of BLOCKING loop
        # time on every turn the tail runs, unconditionally, producing p99
        # loop lateness of 7.2 ms with spikes to 21.9 ms. The transfer
        # function into request lateness measured 1:1 — every millisecond
        # blocked here is a millisecond added to whatever else the loop was
        # serving. Wrapped, a 500 ms stall becomes 0.75 ms p99.
        #
        # This was NOT where the plan said the cost was. v3.1.9 originally
        # targeted _is_repeat_task_traffic's load_state, which measures
        # 0.000 ms on any ongoing conversation because _has_conversational_
        # history returns before the disk read. That would have moved ~2% of
        # the blocking. The measurement is the reason this line changed and
        # that one did not.
        #
        # conv_lock is an asyncio.Lock and it stays HELD across the await,
        # which is the point: the IO moves to a worker, the serialisation
        # that stops concurrent rollups tearing the file does not.
        state = await run_in_threadpool(load_state, conv_id)

        # v3.1.9.4 (R1 / P15-5 follow-up). Checked right after the load, so
        # `state` below is always something real to return, and before
        # anything else in this function runs — no tier check, no
        # watermark repair, no LLM call, no _archive_chapters, no
        # save_state. See wipe_generation_ctx's own docstring for why a
        # single check here, this early, is enough: nothing else may hold
        # conv_lock(conv_id) while this section does, so a wipe's bump (see
        # memory.bump_wipe_generation) either already happened — and this
        # call discards, correctly, because the wipe's own deletes are
        # either already done or queued right behind it on this exact lock
        # — or has not happened yet, in which case this call is free to
        # proceed and whatever it writes is exactly what a wipe arriving
        # afterward is supposed to clear.
        _wgen = _wipe_generation.get()
        if _wgen is not None and _wgen != current_wipe_generation(conv_id):
            logger.warning(
                f"conv={conv_id}: discarding a summary rollup — a wipe ran "
                f"after this tail was submitted (generation {_wgen} != "
                f"current {current_wipe_generation(conv_id)}); the turns it "
                f"would have summarized were already asked to be forgotten"
            )
            return state

        if (
            skip_if_position_past is not None
            and _recorded_position(state) > skip_if_position_past
        ):
            if skipped_at is not None:
                skipped_at.append(_recorded_position(state))
            return state

        # Before anything reads the watermark: a file written by the old
        # _reconcile_watermark can have it BELOW the chunks it wrote, and
        # every number below is derived from it (v3.1.7, R12). The value it
        # had is kept first: it is one of the traces the one-shot legacy
        # adoption reads (hostile pass #4, _legacy_unread_positions).
        _legacy_watermark = state.get("last_summarized_turn")
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

        # The turns the covered-turn record is written from, index-aligned
        # with `messages`' non-system turns (hostile pass #3, F1/F9). The
        # rollup input differs from the raw request only in CONTENT —
        # redaction swaps a reply's text, the tail appends one reply — so
        # anything else is a caller bug, and recording from `messages` itself
        # (raw_turns None) is the safe fallback: the chunk did read that text.
        raw_turns: list[dict] | None = None
        request_turns: list[dict] = []
        if raw_messages is not None:
            _rt = [m for m in raw_messages if m.get("role") != "system"]
            _ns = [m for m in messages if m.get("role") != "system"]
            if len(_rt) == len(_ns):
                request_turns = _rt
                raw_turns = _rt
            elif (
                len(_rt) == len(_ns) - 1
                and _ns[-1].get("role") == "assistant"
            ):
                # The reply this tail appended. As STREAMED, because that is
                # what OpenWebUI stores and re-sends: a stopped reply's
                # trimmed text would read as changed on every later request.
                _reply = _ns[-1]
                if reply_as_streamed is not None:
                    _reply = {**_reply, "content": reply_as_streamed}
                request_turns = _rt
                raw_turns = _rt + [_reply]
            elif logsetup.log_once(f"summarizer.raw_misaligned.{conv_id}"):
                logger.warning(
                    f"conv={conv_id}: the raw request has {len(_rt)} turn(s) "
                    f"against {len(_ns)} in the rollup input, which should "
                    f"differ by at most the appended reply; the covered-turn "
                    f"record is written from the rollup input instead, so "
                    f"redacted or trimmed turns will be summarized fresh "
                    f"rather than reused"
                )
            if request_turns and await run_in_threadpool(
                _adopt_legacy_record, state, request_turns, window_offset,
                _legacy_watermark if isinstance(_legacy_watermark, int) else None,
            ):
                changed = True

        if needs_rollup(state, current_turns):
            try:
                # hostile pass #4 (reviewer A F6): the refreshed turns the
                # FIRST chunk of this drain re-reads. Computed once, against
                # the record as it stands before this call's chunks, and
                # handed to one chunk only, so no turn is read twice.
                _patch: list[tuple[int, int]] = []
                if raw_turns is not None and _needs_l1_rollup(state, current_turns):
                    _first_read = (
                        int(state.get("last_summarized_turn", 0)) + 1
                        - window_offset - 1
                    )
                    _patch = await run_in_threadpool(
                        _patch_candidates, state, raw_turns, _first_read
                    )
                async with httpx.AsyncClient() as client:
                    # v3.1.9 (tail catch-up). ONE loop, priority order
                    # L3 > L2 > L1 — not the old L1-then-L2-then-L3 shape,
                    # and not three separate while loops any more.
                    #
                    # WHY THE ORDER FLIPPED. Under an unlimited budget
                    # (before this feature) it never mattered: L1 fully
                    # drained, then L2 fully drained whatever that produced,
                    # then L3 ran once — every tier was fully caught up by
                    # the time the call returned regardless of which order
                    # got there. Under a PER-TURN budget, a backlog deep
                    # enough to outlast the budget never reaches "L1 fully
                    # drained" in one call — so draining L1 first would
                    # spend the WHOLE per-turn budget on L1, every turn, for
                    # as long as the L1 backlog outlasts L2's threshold.
                    # `state["l1"]` is injected into every request
                    # (format_summary_block, one line per chunk) — L1's own
                    # bound on that injection is L2_CHUNK_SIZE, enforced by
                    # L2 folding chunks out of it, and NEVER while L2 is
                    # starved for budget. Checking the UPPER tier first,
                    # every iteration of this loop, means l1 is folded into
                    # l2 (and l2 into l3) the moment either crosses
                    # threshold, whether or not L1 itself is still behind —
                    # so injection stays bounded for the FULL length of a
                    # catch-up that can span many turns, not just at the end
                    # of it.
                    #
                    # `_l3_done` caps L3 at one refresh per call, the same
                    # contract the old `if` (not `while`) already gave it.
                    #
                    # v3.1.9 (hostile follow-up): stated precisely, because
                    # the first cut of this comment argued it wrong. A
                    # SECOND refresh later in the same call would NOT be
                    # re-folding what the first one already covered —
                    # `_do_l3_rollup` clears l2 on success, so a second
                    # trigger means genuinely NEW chapters arrived since —
                    # it would fold them SEPARATELY from the first refresh's
                    # batch, in a different map-reduce grouping, producing a
                    # DIFFERENT L3 text than one consolidated refresh over
                    # everything the call produced would have. That is the
                    # real reason this caps at one per call rather than
                    # looping: not "wasted work", but "a second refresh
                    # inside one call is not equivalent to the single
                    # consolidated one the OLD L1-then-L2-then-L3 order
                    # always produced" — so capping and deferring the
                    # remainder to the NEXT call is the closer match, and is
                    # exactly what the old order already did whenever ONE
                    # call's L1/L2 work produced more L2 growth than a
                    # single refresh needed to consume (old code's own
                    # trailing `if` also ran only once, catching whatever
                    # existed in l2 AT THAT POINT — the same one-shot shape,
                    # just checked after L1/L2 instead of interleaved with
                    # them).
                    #
                    # DOES THIS REACH GENUINE STEADY STATE (not behind)?
                    # Confirmed no, by construction, not by luck — and
                    # proven directly in test_tail_catchup.py [3b], not just
                    # argued here. For `_l3_done` to defer anything, L3 must
                    # already be due (len(l2) >= L3_CHUNK_SIZE) EITHER at
                    # this call's start OR a second time after this call's
                    # own L1/L2 work. Under ample (non-exhausted) budget —
                    # true for any conversation that is not behind — this
                    # very loop's only "nothing left to do" exit already
                    # resolves L3 (to len(l2)==0) before any call returns,
                    # so nothing is ever left over FOR a later call to find
                    # already due. And a single ordinary turn (one exchange)
                    # advances the observed position by one exchange, so one
                    # ordinary call can produce AT MOST one new L1 chunk and
                    # therefore at most one new L2 fold — never two
                    # independent threshold crossings for `_l3_done` to
                    # ration between. The "lag one call behind" shape this
                    # cap can produce is real, but only for a call that
                    # itself processes many chunks at once — a deep catch-up
                    # under a tight budget, or an admin/backfill rebuild
                    # from the episodic store — never steady, one-exchange-
                    # at-a-time chat, which is what "not behind" means.
                    #
                    # v3.1.9 (hostile pass #5, C5-1/E1). `_l3_failed` /
                    # `_l2_failed` — a failed unit no longer ends the whole
                    # pass. Before this fix, ANY upper-tier failure (a torn
                    # summaries/<conv>.archive.json sidecar, A3-1's
                    # deliberate raise on a wrong-typed "chapters"; an
                    # archive write that keeps failing; an L3/L2 reply that
                    # strips to empty) hit the `break` that used to sit in
                    # its branch below and exited the loop for the rest of
                    # THIS call — and because the SAME drain runs on every
                    # later call too, a persistently failing tier froze
                    # every LOWER tier forever: L1 stopped advancing from
                    # that turn on, hierarchy_lag grew without bound, and
                    # the "/compact drains the backlog" advice health gives
                    # for a stuck hierarchy runs this identical drain and
                    # hits the identical abort first, doing nothing.
                    # Regression against v3.1.7/v3.1.8/21645f2, which ran
                    # three SEPARATE `while`/`if` loops (L1 then L2 then
                    # L3) — a broken upper tier there could only ever defer
                    # ITSELF, never block a tier checked earlier.
                    #
                    # `_needs_l3_rollup(state)` (and `_needs_l2_rollup`)
                    # already guarantee something was due before
                    # `_do_l3_rollup`/`_do_l2_rollup` was called, so a
                    # False return here is ALWAYS a real failure, never
                    # "nothing to do" — see those functions' own early
                    # returns, which are the same conditions these gates
                    # check. Marking the tier failed FOR THIS CALL and
                    # `continue`ing to the tier below (instead of `break`)
                    # restores the old shape: a broken tier costs only its
                    # own retries next call, and every lower tier keeps
                    # covering every turn the way the three-loop version
                    # always did. L1 has no lower tier to fall through to,
                    # so its own failure still ends the pass, unchanged.
                    _l3_done = False
                    _l3_failed = False
                    _l2_failed = False
                    while True:
                        if not _l3_done and not _l3_failed and _needs_l3_rollup(state):
                            if not _budget_allows_unit():
                                break
                            _l3_done = True
                            if not await _do_l3_rollup(
                                conv_id, client, vllm_url, model, state
                            ):
                                _l3_failed = True
                                _mark_tier_failed(conv_id, "l3")
                                if logsetup.log_once(
                                    f"summarizer.tier_stuck.l3.{conv_id}"
                                ):
                                    logger.error(
                                        f"conv={conv_id}: L3 refresh failed "
                                        f"and will be retried next turn "
                                        f"without blocking L1/L2 — see the "
                                        f"archive-abort or empty-reply "
                                        f"warning above (or its absence) "
                                        f"for why; a refresh failing on "
                                        f"EVERY attempt usually means a "
                                        f"torn archive sidecar that needs "
                                        f"an operator (this message prints "
                                        f"once per conversation)"
                                    )
                                continue
                            _mark_tier_recovered(conv_id, "l3")
                            changed = True
                        elif not _l2_failed and _needs_l2_rollup(state):
                            if not _budget_allows_unit():
                                break
                            if not await _do_l2_rollup(
                                conv_id, client, vllm_url, model, state
                            ):
                                _l2_failed = True
                                _mark_tier_failed(conv_id, "l2")
                                if logsetup.log_once(
                                    f"summarizer.tier_stuck.l2.{conv_id}"
                                ):
                                    logger.error(
                                        f"conv={conv_id}: L2 fold failed "
                                        f"and will be retried next turn "
                                        f"without blocking L1 — a fold "
                                        f"failing on EVERY attempt usually "
                                        f"means the model is returning "
                                        f"empty content for this "
                                        f"conversation's chapters (this "
                                        f"message prints once per "
                                        f"conversation)"
                                    )
                                continue
                            _mark_tier_recovered(conv_id, "l2")
                            changed = True
                        elif _needs_l1_rollup(state, current_turns):
                            if not _budget_allows_unit():
                                break
                            if not await _do_l1_rollup(
                                conv_id, client, vllm_url, model, state, messages,
                                window_offset, raw_turns, _patch,
                            ):
                                # L1 has no lower tier to fall through to;
                                # unchanged from before this fix.
                                break
                            _patch = []
                            changed = True
                        else:
                            # Nothing left DUE, or every due tier this call
                            # already ran or has already failed once. The
                            # ONLY exit that means "no work is
                            # outstanding"; every other `break` above means
                            # "work remains but the budget said stop" or
                            # "L1 itself failed" — a failed L2/L3 no longer
                            # reaches this branch on its own; it falls
                            # through to the tier below instead (see above).
                            break
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
        # rollups whether or not a later tier raised. That now includes the
        # covered-turn record, which _do_l1_rollup writes in the same mutation
        # as its chunk (hostile pass #3, F1/F9). There is deliberately no
        # catch-up here any more: recording a position from a LATER call's
        # array is what blessed a deleted reply and every placeholder an
        # admin rebuild summarized.

        if changed:
            try:
                # The expensive half: tempfile + fsync + rename, measured at
                # 3.8-4.9 ms on the loop. Same reasoning as the load above,
                # and the same conv_lock held across the await.
                await run_in_threadpool(save_state, conv_id, state)
            except Exception as e:
                logger.exception(f"conv={conv_id}: rollup state write failed: {e}")

        return state


# ---------------------------------------------------------------------------
# Tail catch-up progress — process-local, per conversation (v3.1.9, hostile
# follow-up on the tail-catch-up feature)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. health.py's hierarchy_lag reason first tried to tell a
# CONVERGING catch-up (self-healing, no operator action needed) from a
# STUCK one (needs `/compact`) by comparing hierarchy_lag_recent poll to
# poll. That is WRONG, and wrong in a way that only showed up against a
# real pod: the Dockerfile HEALTHCHECK polls /health/full every 30 s, and
# she sends a message — the only thing that ever advances a rollup — every
# few minutes at most. So on a real pod, almost every pair of CONSECUTIVE
# polls sees the exact same lag: not because the catch-up stalled, but
# because nothing has happened between the two polls at all. The
# "converging" wording appeared on exactly the one poll right after a tail
# happened to run, then flipped back to the actionable "/compact" wording
# for the next several dozen polls until her next message — poll-cadence
# dependent, flapping, and printing the "run /compact" advice on nearly
# every poll during a real, healthy, self-healing catch-up. The project's
# own test for it polled once per turn, which is exactly the one cadence
# that hid the bug.
#
# THE FIX: evidence the TAIL itself records, independent of how often
# anything polls /health/full. Every BUDGETED pass (main._rollup_hierarchy,
# and backfill.py's one-shot rollup) reports here whether the watermark it
# just tried to advance for `conv_id` actually moved. health.py reads it
# back and asks two cadence-independent questions: "did the watermark
# advance RECENTLY, in wall-clock time?" (converging) and "have PASSES
# happened, with work still due, without an advance?" (stuck) — both
# keyed to how often SHE chats (a tail pass only ever happens on her
# turn), never to how often the HEALTHCHECK polls.
#
# WHY summarizer.py AND NOT A NEW MODULE. health.py cannot import main
# (main imports health) — the same constraint tailhealth.py's own
# docstring names for the identical reason. This module is not that
# precedent's twin by accident: both `main._rollup_hierarchy` (the writer)
# and `health.py` (the reader) already import `summarizer` for unrelated
# reasons, so no new import edge is needed anywhere. Keyed by conv_id,
# which is not a new privacy surface here — gather_memory_stats already
# puts a bare conv_id in this same endpoint's payload
# (hierarchy_lag_conv/hierarchy_lag_recent_conv).
#
# PROCESS RESTARTS LOSE THIS. There is no disk-backed version, on purpose:
# it exists to answer "is the CURRENT process's tail actively making
# progress", and a value surviving a restart would describe a process that
# no longer exists. A conv_id with no entry means "no budgeted pass has
# run for it in this process yet" — see catchup_progress_for's own
# docstring for what health.py does with that (the answer is: treat it the
# same as "not converging", the safe default — see that function's own
# comment for why).
_catchup_progress: dict[str, dict[str, Any]] = {}


def record_catchup_pass(
    conv_id: str, before_watermark: int, after_watermark: int, work_due_after: bool,
) -> None:
    """Called after every BUDGETED maybe_rollup pass (main._rollup_hierarchy,
    backfill.py) — never after an unbounded one (admin /compact's default
    max_calls=200 already drains everything in a few passes; there is
    nothing for a catch-up-rate signal to describe there).

    `before_watermark`/`after_watermark` are last_summarized_turn before and
    after this pass. `work_due_after` is whether the hierarchy still needs a
    rollup once this pass finished (summarizer.needs_rollup on the returned
    state) — a pass that advanced the watermark AND still has more due is
    still "progress", tracked the same as any other advance.
    """
    now = time.monotonic()
    entry = _catchup_progress.setdefault(conv_id, {
        "last_advance_monotonic": None,
        "passes_since_advance": 0,
    })
    if after_watermark > before_watermark:
        entry["last_advance_monotonic"] = now
        entry["passes_since_advance"] = 0
    elif work_due_after:
        # Only a pass that HAD work due and made none counts against the
        # stall counter — a pass with nothing due (an ordinary, caught-up
        # turn) is not evidence of anything stalling.
        entry["passes_since_advance"] += 1
    entry["watermark"] = after_watermark
    entry["work_due"] = work_due_after
    entry["last_pass_monotonic"] = now


def catchup_progress_for(conv_id: str) -> dict[str, Any] | None:
    """This process's evidence for `conv_id`, or None if no budgeted pass has
    run for it since this process started (a fresh boot, or a conv_id that
    has simply never been behind). health.py's own comment at the call site
    is where "None" gets turned into a behaviour — this function only
    reports what is known, never guesses at what a restart erased.

    A shallow copy: callers get a snapshot, not a handle into the live dict
    a later pass could mutate under them mid-read.
    """
    entry = _catchup_progress.get(conv_id)
    return dict(entry) if entry is not None else None


def _reset_catchup_progress_for_tests() -> None:
    _catchup_progress.clear()
    _tier_failure.clear()


# ---------------------------------------------------------------------------
# Which tier, if any, is stuck? (v3.1.9, hostile pass #5, C5-1/E1)
# ---------------------------------------------------------------------------
#
# Companion to _catchup_progress above — same contract: in-process only,
# keyed by conv_id, lost on restart. WHY IT EXISTS: the L3>L2>L1 drain in
# `_maybe_rollup_body` now falls through past a failed upper tier instead
# of freezing the whole hierarchy behind it (see that loop's own comment),
# which fixes the freeze but leaves a NEW question an operator needs
# answered: which tier is the one that keeps failing? `passes_since_
# advance` (_catchup_progress) cannot say — a conversation can rack up
# stalled passes for a reason that has nothing to do with any tier being
# broken (no budget ever allocated, a degrade-guard pause). This dict
# exists so health.py's "stuck" reason can name the actual failing tier
# instead of pointing at `/compact`, which runs this identical drain and
# hits the identical failure first — advice that cannot help.
#
# Set the moment a tier's unit fails; cleared the moment that SAME tier
# next succeeds (not cleared by a different tier succeeding — L1 advancing
# while L3 keeps failing says nothing about L3). A conv_id with no entry
# means "no tier has failed for it in THIS process" — the same safe
# default `catchup_progress_for`'s own docstring gives its sibling.
_tier_failure: dict[str, str] = {}


def _mark_tier_failed(conv_id: str, tier: str) -> None:
    _tier_failure[conv_id] = tier


def _mark_tier_recovered(conv_id: str, tier: str) -> None:
    if _tier_failure.get(conv_id) == tier:
        del _tier_failure[conv_id]


def failing_tier_for(conv_id: str) -> str | None:
    """"l2" or "l3" if that tier's unit most recently failed for this
    conversation, in THIS process, and has not since succeeded; None if no
    failure is on record (including "never observed" — a restart or a
    conv_id this process has not rolled up). See the _tier_failure block
    comment above for the full contract.
    """
    return _tier_failure.get(conv_id)


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
