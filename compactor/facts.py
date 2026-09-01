"""
compactor.facts — Persistent facts memory (V2.0 Phase 2, "semantic" layer).

Storage shape on disk (one JSON file per conversation):
    {
      "conv_id": "abc123...",
      "updated_at": "2026-05-28T05:00:00Z",
      "facts": [
        { "text": "Protagonist is Lyra, half-elf ranger, age 23.",
          "added_turn": 5,
          "last_used": 1748419200,
          "pin": false },
        ...
      ]
    }

Each fact is one short bullet extracted by the LLM from a single exchange
(user message + assistant response). Facts are appended over time; LRU
eviction by `last_used` keeps the STORE under COMPACTOR_MAX_FACTS_TOKENS.
Eviction MOVES facts to the archive sidecar — it does not delete them
(v3.1 F9).

v3.1.4 F1 — the store cap and the INJECTED block are separate budgets as of
this release, not one knob doing two jobs. COMPACTOR_MAX_FACTS_TOKENS still
bounds what prune_facts keeps on disk; COMPACTOR_INJECT_FACTS_TOKENS (new,
much smaller — target ~300-400 tokens) bounds what select_for_injection puts
in THIS turn's prompt. select_for_injection ranks the non-pinned facts by
relevance to the current turn (reusing retrieval.py's CPU-only bge-small
embedder) rather than injecting the whole store, and a `pin`-flagged
identity-tier fact always makes it in regardless of ranking or budget. See
select_for_injection's own docstring for the full mechanism and its
graceful-degradation contract.

Lifecycle:
  1. Request arrives → load_facts(conv_id) → select_for_injection(query_text=
     <this turn's text>) → inject the selected subset as a system block
  2. Mark ONLY the injected subset as `last_used = now` (LRU tracking).
     Touching the whole store is what made `last_used` uniform across every
     fact, which collapsed the eviction sort key onto `added_turn` and made
     "LRU" mean "drop the conversation's oldest, most foundational facts"
     (v3.1 F9) — injecting only the top-K-relevant-plus-pinned subset (F1)
     is what keeps that touch meaningfully selective turn over turn, instead
     of everything being "recently used" because everything was injected.
  3. After response streams back → extract_facts_from_exchange() in async tail.
     Its INPUT is budgeted too (_fit_extraction_input): the store it is handed
     may be the whole file, and the reply it is handed is unbounded.
  4. Append new facts → prune to the STORE budget (archiving the evictions,
     COMPACTOR_MAX_FACTS_TOKENS — unrelated to the injection budget above) →
     save_facts()

All file writes go through memory.atomic_write_json() for crash safety.
All read/write pairs are serialized per-conv via memory.conv_lock().
"""

import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

import tokens
from memory import (
    StoreUnreadable,
    atomic_write_json,
    conv_lock,
    facts_archive_path,
    facts_path,
    read_json_strict,
)

# v3.1.4 F1. retrieval.py does not import this module (directly or
# transitively — its only local import is `from memory import
# STORAGE_ROOT`), so there is no cycle: dedup.py already imports BOTH
# `facts as facts_module` and `retrieval as retrieval_module` side by side
# and that has never been a problem. Mirrors dedup.py's own import exactly,
# including the reason retrieval.py's import is always cheap and safe: the
# heavy deps (fastembed, chromadb) are lazy-imported inside
# retrieval._try_init(), never at module import time, so `import retrieval`
# itself cannot fail or block. Ranking still degrades gracefully if THAT
# lazy init fails later — see _relevance_order.
import retrieval as retrieval_module

logger = logging.getLogger("compactor.facts")


# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

# v3.1.4 F1 — two knobs, two jobs. Through v3.1.3 COMPACTOR_MAX_FACTS_TOKENS
# was BOTH of these at once: prune_facts' STORE cap (how much a conversation
# may remember, active-set-on-disk) and select_for_injection's INJECTION cap
# (how much goes into every single prompt). Because the injection cap equaled
# the store cap, the entire active set was injected every turn, so the entire
# active set was TOUCHED every turn (see touch_facts), so `last_used` carried
# no signal at all and LRU eviction collapsed onto `added_turn` — N3's
# measured 70% FIFO churn (5,341 extracted, 3,714 evicted) selecting for
# nothing but age. See select_for_injection's docstring for the fix.
#
# Deliberately NOT derived from each other in either direction: the deploy is
# about to raise the store cap from 1500 to ~3000-6000 (N3/F1 part 3), and an
# injection default computed as some fraction of the store cap would silently
# re-grow the injected block right along with it, reproducing the exact
# coupling this split exists to remove. Each is an independent, absolute
# token count with its own sane fallback if the operator sets neither.

# STORE cap. How much a conversation may remember, active-set-on-disk.
# prune_facts' default budget. We use char/4 as a fast estimator — precision
# doesn't matter for a soft cap. ~1500 tokens ≈ 6000 chars ≈ 100-150 short
# bullets.
_MAX_FACTS_TOKENS = int(os.environ.get("COMPACTOR_MAX_FACTS_TOKENS", "1500") or 1500)

# INJECTION cap. How much goes into THIS turn's prompt. select_for_injection's
# default budget. Small enough that only what's actually relevant to the
# current turn (plus the pinned identity tier, see select_for_injection) gets
# touched, which is what gives `last_used` a real recency signal again.
# Independent of _MAX_FACTS_TOKENS by design — see the block comment above.
#
# 800 -> 400 in v3.1.5. F1 part 1 specified 300-400 and main.py's call site
# has documented the default as 400 since it landed; 800 was a v3.1.4 review
# value that the surrounding comments were never updated to match, so the
# code and its own documentation disagreed until now.
#
# 400 is what the user asked for on 2026-08-31, reporting replies had gone
# formulaic. Production was injecting a median of 91 fact bullets per turn
# (p90 103, max 179) under a header that asked the model to maintain
# consistency with them — see persona.py's _PERSONA_BLOCK_HEADER for the
# header half of that fix. Halving the budget matters less than what the
# ranking then does with it: because main.py passes query_text at both call
# sites, the surviving ~26 are the ones THIS turn is about, so the block
# rotates with the topic instead of presenting the same 91 lines every turn.
# Without that wiring this change would be strictly worse than 800 — it would
# keep a fixed most-recently-used prefix forever — which is why
# test_facts_wiring.py exists.
_INJECT_FACTS_TOKENS = int(
    os.environ.get("COMPACTOR_INJECT_FACTS_TOKENS", "400") or 400
)

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

# INPUT budget for one extraction call — the counterpart to _EXTRACTION_MAX_TOKENS
# above, which only ever bounded the OUTPUT.
#
# Until v3.1 there was none. The payload is the extraction system prompt + the
# ENTIRE fact store + the full user turn + the full assistant reply, and none of
# it was counted. The chat path sends no max_tokens, so vLLM may generate a reply
# of nearly the whole window and extraction then stacks the prompt and the store
# on top of that reply. Two calls in the 2026-08-24 window were rejected at
# 33,790 and 33,581 input tokens against a 32,768 window. A rejection is not a
# partial result: those exchanges' facts were never extracted, and nothing
# retried them.
#
# Same shape as main.summarize's own input budget (main.py's
# `budget = min(MAX_MODEL_LEN, max(256, MAX_MODEL_LEN - out - reserve))`), read
# from the same env var so the two cannot drift apart on a re-sized deployment.
_MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768") or 32768)
# Slack left inside the window for the chat template's per-message framing.
# Same default and same purpose as COMPACTOR_SUMMARY_INPUT_RESERVE. This is a
# FIXED, real-token cost (the template's own scaffolding) — see
# _ASSISTANT_CONTENT_ESTIMATE_LOW_FRACTION below for the separate, PROPORTIONAL
# correction this reserve cannot do on its own.
_EXTRACTION_INPUT_RESERVE = int(
    os.environ.get("COMPACTOR_FACTS_INPUT_RESERVE", "2048") or 2048
)
# Clamped for the same reason as main.HARD_INPUT_LIMIT: a bare floor could sit
# ABOVE the model's own window on a small-context model, which would budget
# nothing at all. This is a REAL-token ceiling — the model's own window minus
# what the reply and the framing need.
_EXTRACTION_INPUT_BUDGET_REAL_TOKENS = min(
    _MAX_MODEL_LEN,
    max(256, _MAX_MODEL_LEN - _EXTRACTION_MAX_TOKENS - _EXTRACTION_INPUT_RESERVE),
)

# _fit_extraction_input below (and everything it calls) measures exclusively in
# _estimate_tokens' unit — char/4 — never in real tokens. Comparing that
# estimate directly against _EXTRACTION_INPUT_BUDGET_REAL_TOKENS, as this
# module did through v3.1, silently treated "1 estimated token" as "1 real
# token" charged by vLLM. This module's own docstring already named the gap
# ("this module has no tokenizer at all") but the only thing bridging it was
# the FIXED reserve above, which cannot correct a PROPORTIONAL error: on a
# large enough payload no fixed number of tokens of slack survives a
# percentage under-count, no matter how generous.
#
# MEASURED, INSIDE THE PRODUCTION IMAGE, AGAINST THE REAL TOKENIZER (tokens.py
# + mistral_common's bundled tekken vocabulary, the same family vLLM uses) —
# not cited from a document. An earlier draft of this fix used "51% low"
# handed down as a general figure for this model's assistant content and
# a synthetic repeated-lorem-ipsum reproduction built on it. Measuring that
# same synthetic text against the real tokenizer disproved the premise: char/4
# read it 53% HIGH, not low — repeated text compresses hard under a real BPE
# vocabulary in a way char/4 cannot see. So a single fixed "% low" cannot be
# right for all content, because the true direction and size depend on SHAPE:
#
#   ordinary structured prose (paragraphs, some markdown, numbers) ..  ~6.5% low
#   prose with an occasional divider mixed in ....................... ~16% HIGH
#   repetitive/degenerate filler ..................................... ~53% HIGH
#   pure box-drawing decoration ...................................... ~87% low
#
# The decoration figure independently reproduces this file's own
# INCIDENT_2026-08-28 measurement (2,151 chars / ~4,275 tokens, ~87.4% low)
# to within a rounding error — two separate measurements agreeing is the
# strongest evidence in this file for any number. Ordinary content is close
# enough that no correction is really needed; decoration is the only content
# shape actually seen to blow the estimate this badly, and it is exactly the
# shape this module's own history keeps producing: the extraction INPUT is
# the raw prior turn, unfiltered — is_storable_fact's structural filter only
# ever runs on what extraction OUTPUTS, so a decoration-heavy prior reply
# reaches this budget check completely unfiltered.
#
# A one-directional worst case would be no worse than imprecise; ORDINARY
# content reading HIGH under char/4 is what makes it actually safe to say
# direction 2 (a fact wrongly evicted or trimmed for looking bigger than it
# is) is not a live risk from this estimator: on the content this store
# actually holds, char/4 already errs toward "looks bigger", not smaller.
# Only decoration flips that, and only on the INPUT side, which is exactly
# what this constant now defends — see also the tokens.count() backstop in
# extract_facts_from_exchange below, which is the actual fix for the common
# case: this constant is what's left standing when that backstop cannot run.
#
# So this budget is deliberately calibrated to the WORST case actually
# measured (decoration, ~87%), not to "typical" content — because typical
# content does not need defending, and this is the one number used when
# nothing can verify the real cost before the request goes out.
_ASSISTANT_CONTENT_ESTIMATE_LOW_FRACTION = float(
    os.environ.get("COMPACTOR_FACTS_ESTIMATE_LOW_FRACTION", "0.87") or 0.87
)
_EXTRACTION_INPUT_BUDGET = max(
    256,
    int(
        _EXTRACTION_INPUT_BUDGET_REAL_TOKENS
        * (1 - _ASSISTANT_CONTENT_ESTIMATE_LOW_FRACTION)
    ),
)

# How many times extract_facts_from_exchange will re-trim and re-measure
# against the real tokenizer (tokens.count()) before giving up and sending
# its best effort. Bounded because a real disagreement should converge in a
# step or two once corrected toward the true ratio; a cap turns "does not
# converge" into one WARNING log rather than a request that never goes out.
_MAX_REAL_TOKEN_MEASURE_RETRIES = 3


def _extraction_input_budget_estimate_units() -> int:
    """The FIRST-PASS estimate-unit budget handed to _fit_extraction_input.

    Measured (see the trimming-impact numbers in the fix that added this):
    starting from the conservative, worst-case-calibrated
    _EXTRACTION_INPUT_BUDGET unconditionally — even when the real tokenizer
    can verify and correct the result afterward — cut ordinary, undecorated
    long exchanges down to roughly 15% of their real size for no reason: the
    backstop below only ever SHRINKS the budget on disagreement, it never
    widens it back up, so a pessimistic starting point survives even when
    the very first real measurement would have confirmed there was 10x the
    room actually needed.

    So: generous (the bare real ceiling, in estimate-units) whenever the real
    tokenizer is available to check the result — the backstop is what keeps
    that safe, not this number. Conservative (the measured-worst-case
    fallback) only when nothing can verify anything, which is the one
    situation this number actually has to be right without help.
    """
    if tokens.is_available():
        return _EXTRACTION_INPUT_BUDGET_REAL_TOKENS
    return _EXTRACTION_INPUT_BUDGET


def extraction_enabled() -> bool:
    return _EXTRACTION_ENABLED


# ---------------------------------------------------------------------------
# Fact shape + token estimation
# ---------------------------------------------------------------------------

# A fact is a dict — using TypedDict-style for clarity but plain dict for
# JSON round-trip simplicity.
#   { "text": str, "added_turn": int, "last_used": int (unix ts),
#     "pin": bool }
#
# `last_used`: unix seconds, set by touch_facts() on the facts INJECTED into
# a turn. One unit, one writer. Safe to compare across facts.
#
# `pin`: v3.1.4 F1 part 2. True marks an identity-tier fact (who she is, who
# the owner is to her, a standing preference) that select_for_injection must
# ALWAYS include, bypassing relevance ranking and the token budget entirely
# — top-K-by-relevance alone can drop "her name is X" on a turn about dinner,
# which is the "she forgot me" failure wearing a relevance-scoring hat.
# Defaults to False; absent on every record written before this field
# existed, and load_facts/load_archive/restore_from_archive all read it via
# `.get("pin", False)` so old records load exactly as before. Set with
# set_pinned(); reached from the /pin and /unpin commands in commands.py,
# and shown as "[pinned]" by /list-facts.
#
# A pinned fact is, by construction, in select_for_injection's output on
# EVERY turn (never excluded by budget or by ranking), so it is touched on
# every turn too (main.py touches only what was injected — v3.1 F9). That
# alone is what keeps prune_facts' plain LRU from ever choosing a pinned
# fact as an eviction candidate: no special case needed in prune_facts or
# _lru_split, the store cap's eviction policy is UNCHANGED by this field.
#
# `added_turn`: NOT one unit. Do not compare two facts' added_turn unless you
# know they came from the same writer. Unifying it is the D1 identity work;
# until that lands, the four writers and what each actually stores are:
#
#   main._async_tail          `turn_index`, which the chat handler computes as
#                             `len(messages) + 1` — a count of MESSAGES in the
#                             array THE CLIENT SENT, not turns and not the
#                             conversation's true length. On 2026-08-24 the
#                             client sent 7 of 241 messages, so facts extracted
#                             late in a long conversation were stamped 8.
#   commands._handle_remember `ctx.get("turn_index", 0)` — the same expression
#                             handed through from the chat handler, so the same
#                             message-count unit, except that the `.get`
#                             default of 0 stamps a manually-remembered fact as
#                             the oldest thing in the store if ctx ever lacks
#                             the key.
#   backfill._run_backfill    `i * 2`, where i is the 1-based index of the
#                             user/assistant PAIR within the slice backfill
#                             happened to be handed. Message-units by
#                             construction, but numbered from that slice's
#                             start rather than the conversation's.
#   dedup._merge_metadata     `min(added_turn)` over the merged cluster — a
#                             derived value, and the one place where the units
#                             above get mixed into a single number.
#
# portability.import_conversation restores whatever a bundle carried, so an
# imported conversation's added_turn values were minted by another
# conversation's writers entirely. selftest writes a literal 0.
#
# Consequence for this module: added_turn is a stable tie-breaker and an
# injection-order hint. It is NOT a recency signal, and prune_facts must
# never be allowed to fall back onto it as one — see prune_facts.


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
    facts file exists yet.

    Raises memory.StoreUnreadable if the file IS there and could not be
    read. It used to return [] for that too, which every caller that then
    saved read as "this conversation has no facts" and wrote back over the
    real ones (v3.1 F1a). Callers that merely display or count facts should
    catch it and say so; callers that write must abort.
    """
    data = read_json_strict(facts_path(conv_id), default={})
    facts = data.get("facts", []) if isinstance(data, dict) else []
    # Defensive: ensure each entry has the expected shape; drop malformed.
    valid: list[dict] = []
    for f in facts:
        if isinstance(f, dict) and isinstance(f.get("text"), str) and f["text"].strip():
            valid.append({
                "text": f["text"],
                "added_turn": int(f.get("added_turn", 0)),
                "last_used": int(f.get("last_used", 0)),
                # v3.1.4 F1: absent on every record written before this field
                # existed (`.get(..., False)` — see the fact-shape comment
                # above), so an old store round-trips as entirely unpinned.
                "pin": bool(f.get("pin", False)),
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
    """Return the archived facts list for a conv. Empty if no archive yet.

    Raises memory.StoreUnreadable on an unreadable sidecar, same contract
    as load_facts. restore_from_archive reads BOTH halves and writes both
    back, so a silent empty here lost the cold store and the active set in
    one call (v3.1 F1e).
    """
    data = read_json_strict(facts_archive_path(conv_id), default={})
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
                # v3.1.4 F1: a pinned fact can still land in the sidecar via
                # the time-based sweep (archive_stale_facts) — pin exempts a
                # fact from injection ranking, not from that separate,
                # unchanged policy — so the pin has to survive the trip
                # there and back via restore_from_archive.
                "pin": bool(f.get("pin", False)),
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


def archive_facts(conv_id: str, entries: list[dict]) -> int:
    """Move `entries` into the cold-storage sidecar. Returns how many landed.

    The single way a fact leaves the active set without the user asking —
    both the time-based sweep (archive_stale_facts) and the budget eviction
    in prune_facts go through here, so "evicted" always means "recoverable
    via restore_from_archive" and never "unlinked" (v3.1 F9).

    Writes ONLY the sidecar. The caller still owns the active-set write, and
    must do it AFTER this returns: a crash in between then leaves the fact in
    both files, which restore_from_archive's caller can sort out. The other
    order loses it outright.

    Re-archiving a fact that is already in the sidecar replaces the old entry
    rather than appending a second copy — a fact can be evicted, restored and
    evicted again, and the sidecar should not grow a copy per round trip.
    """
    if not entries:
        return 0
    archive_ts = _now_unix()
    incoming = [{**f, "archived_at": archive_ts} for f in entries]
    incoming_texts = {f.get("text") for f in incoming}
    existing = load_archive(conv_id)
    save_archive(
        conv_id,
        [f for f in existing if f.get("text") not in incoming_texts] + incoming,
    )
    return len(incoming)


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
    # Sidecar first, active set second — see archive_facts on the ordering.
    archive_facts(conv_id, stale)
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
            "pin": bool(f.get("pin", False)),
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

def _fact_bullet_tokens(text: str) -> int:
    """Estimated cost of one fact AS ACTUALLY RENDERED in the injected block:
    the "- " prefix and its line break, not the bare fact text.

    _lru_split priced a fact by `_estimate_tokens(f["text"])` alone through
    v3.1 — the same unit gap _fit_extraction_input already corrects for its
    own payload (see its "Settle that difference against the assembled
    payload" comment), just never applied here. Reproduced: 150 short facts
    at a 1500-token budget were all kept believing the total was exactly
    1500 (bare text), while format_facts_block's real rendering of that same
    set measured 1674 — an 11.6% overshoot of a budget the code believed it
    was honouring, purely from missing the "- " prefix, the per-line break,
    and the header (see _FACTS_BLOCK_HEADER_TOKENS below), independent of
    any question about char/4's accuracy as a tokenizer.

    Includes a trailing newline even though format_facts_block's join only
    puts one BETWEEN lines (the last line has none): a 1-character-per-fact
    over-count in the estimator's own unit, which only ever makes this
    module keep fewer facts than the true rendering needs, never more —
    the safe direction for a budget, and negligible for a soft cap.
    """
    return _estimate_tokens(f"- {text}\n")


def _lru_split(
    facts: list[dict], max_tokens: int
) -> tuple[list[dict], list[dict]]:
    """Split `facts` into (fits_the_budget, does_not) by LRU on `last_used`.

    The one place the budget order is decided, so injection and eviction can
    never disagree about which facts are the working set: select_for_injection
    takes the first half, prune_facts archives the second.

    Both halves come back sorted by added_turn, which is an ordering hint
    only — see the field notes above on why it is not a recency signal. The
    tie-break on `last_used` keeps eviction deterministic between facts
    touched in the same second (a manual /remember and an extraction landing
    on the same turn), it does not decide recency.

    Budgets against the block format_facts_block ACTUALLY renders — the
    header once, plus each fact as "- <text>\n" — not the bare fact text.
    The header is fixed and renders exactly once whenever any fact survives
    (format_facts_block never emits a header with zero bullets under it), so
    a budget that cannot afford the header cannot afford to inject anything:
    that is not a smaller injection, it is no injection, and every fact goes
    to eviction rather than a partial, header-less fragment nobody asked for.
    """
    if not facts:
        return [], []
    if _FACTS_BLOCK_HEADER_TOKENS > max_tokens:
        # Degenerate budget: too small even for the header alone. Correct
        # behaviour is to inject nothing, not a header over an empty body —
        # so every fact is an eviction candidate here, not a partial fit.
        return [], sorted(facts, key=lambda f: f["added_turn"])

    def _rendered_cost(subset: list[dict]) -> int:
        block = format_facts_block(subset)
        return _estimate_tokens(block) if block else 0

    if _rendered_cost(facts) <= max_tokens:
        return list(facts), []

    body_budget = max_tokens - _FACTS_BLOCK_HEADER_TOKENS
    # Sort by last_used ascending (oldest first), then by added_turn for stability
    # PINNED FACTS SORT LAST, i.e. they are evicted last of all. The
    # fact-shape note above used to argue that a pin needs no eviction
    # exemption because a pinned fact is injected every turn and therefore
    # always freshly touched. Measured false in two windows:
    #
    #   1. /pin sets the flag but not last_used, so pinning a STALE fact
    #      leaves it an eviction candidate until its first protected
    #      injection - and the extraction tail can prune in between.
    #   2. Touch is conditional on the facts layer surviving
    #      _bound_injected_blocks (main.py), which drops whole layers; the
    #      production logs show 87 over-budget drops in one window. A pinned
    #      fact ages normally across those turns.
    #
    # Repro before this fix: a pinned fact with a stale last_used, against
    # 200 fresh facts, was archived - silently no longer injected, which is
    # identity loss with no error. Recoverable from the sidecar, but nothing
    # would have said so.
    #
    # The exemption is in the SORT rather than a separate carve-out so
    # injection and eviction still read the same ordering - the invariant
    # this function exists to hold.
    sorted_facts = sorted(
        facts,
        key=lambda f: (bool(f.get("pin")), f["last_used"], f["added_turn"]),
    )
    kept_reversed: list[dict] = []
    running = 0
    # Walk from most-recently-used backward, keeping facts that fit. This
    # sums PER-FACT floors as a fast approximation to pick candidates without
    # rebuilding the whole rendered string on every step.
    for f in reversed(sorted_facts):
        cost = _fact_bullet_tokens(f["text"])
        if running + cost <= body_budget:
            kept_reversed.append(f)
            running += cost

    # Settle against the real render before trusting the approximation: a sum
    # of per-fact floors is always <= the floor of their true combined
    # length, so the fast walk above can under-count by a few tokens once
    # enough facts accumulate, letting the real block land a token or two
    # over `max_tokens` even though the approximation said it fit. Same
    # principle as _fit_extraction_input settling against _assembled_tokens
    # rather than trusting a per-part cost model. kept_reversed was built
    # most-recently-used-first, so popping the tail drops the
    # least-recently-used fact still standing — LRU order is preserved.
    while kept_reversed and _rendered_cost(kept_reversed) > max_tokens:
        kept_reversed.pop()

    kept_ids = {id(f) for f in kept_reversed}
    # Restore original-ish ordering by added_turn for stable injection
    kept = sorted(kept_reversed, key=lambda f: f["added_turn"])
    evicted = sorted(
        (f for f in facts if id(f) not in kept_ids), key=lambda f: f["added_turn"]
    )
    return kept, evicted


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Doesn't assume inputs are pre-normalized — same
    reasoning and same shape as dedup._cosine; duplicated rather than
    imported because dedup.py imports facts.py already (facts_module) and
    reaching back the other way for one four-line function is not worth
    inventing a shared-utility module for.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _relevance_order(
    candidates: list[dict],
    query_text: str | None,
    embedder: Callable[[list[str]], list[list[float]] | None] | None,
) -> list[dict] | None:
    """Return `candidates` sorted best-match-first against `query_text`, or
    None when ranking cannot run (no query, no embedder, embedding failed,
    or a result shaped wrong) — callers degrade to LRU order on None.

    One batched embedding call for the query AND every candidate's text
    together (`[query_text] + texts`), matching dedup._embed_facts' batching
    reasoning: fastembed/bge-small is CPU-milliseconds either way, but one
    call is still cheaper than two, and it means the query and the facts are
    embedded by the exact same call, not two that could observe the
    embedder in different states.

    `embedder` lets a caller (or a test) supply its own — see
    select_for_injection's docstring on why this exists ALONGSIDE the direct
    `import retrieval` at the top of this module rather than instead of it.
    """
    if not query_text or not candidates:
        return None
    embed_fn = embedder if embedder is not None else retrieval_module._embed
    texts = [f.get("text", "") or "" for f in candidates]
    if not all(texts):
        # Defensive, same as dedup._embed_facts: an empty text would embed
        # as "" and cluster/score meaninglessly against everything.
        return None
    try:
        vecs = embed_fn([query_text] + texts)
    except Exception as e:
        logger.warning(
            f"facts relevance ranking: embedder raised ({e}); falling back "
            f"to LRU for this turn — chat is unaffected"
        )
        return None
    if not vecs or len(vecs) != len(texts) + 1:
        # retrieval._embed's own contract: None on failure. A short list
        # would be a caller-supplied embedder that dropped rows silently —
        # treat it exactly the same as unavailable rather than guess which
        # rows are missing.
        return None
    query_vec, fact_vecs = vecs[0], vecs[1:]
    scored = list(zip(candidates, fact_vecs))
    # Stable sort: candidates already carry the store's added_turn order, so
    # equal-scoring facts (all-zero mock vectors in a test, or a genuine
    # exact tie) keep that order rather than an arbitrary one.
    scored.sort(key=lambda pair: _cosine(query_vec, pair[1]), reverse=True)
    return [f for f, _vec in scored]


def _greedy_fill_by_priority(
    priority: list[dict], bullets_budget: int
) -> list[dict]:
    """Keep facts from `priority` (already ordered best-first) greedily
    until the summed PER-FACT bullet cost would exceed `bullets_budget` —
    a budget for the rendered "- <text>\\n" lines ALONE, no header. Same
    per-fact-floor approximation _lru_split uses (and the same reason: fast,
    and the caller settles against the real combined render afterward,
    because a sum of floors can under-count once enough facts accumulate).

    Restores added_turn order in the result — same convention as
    _lru_split's `kept` — so injection order doesn't reshuffle with every
    turn's ranking.
    """
    if bullets_budget <= 0:
        return []
    kept: list[dict] = []
    running = 0
    for f in priority:
        cost = _fact_bullet_tokens(f["text"])
        if running + cost <= bullets_budget:
            kept.append(f)
            running += cost
    return sorted(kept, key=lambda f: f["added_turn"])


def _settle_against_budget(
    pinned: list[dict], selected_rest: list[dict], max_tokens: int
) -> list[dict]:
    """Trim `selected_rest` until `pinned + selected_rest` actually renders
    within `max_tokens`, then return the combined list.

    Same principle _lru_split's own settle step uses for its per-fact-floor
    approximation: a sum of per-fact floors is always <= the floor of their
    true combined length, so `_greedy_fill_by_priority`'s fast walk can
    under-count by a token or two once enough bullets accumulate, letting
    the real block land a token over `max_tokens` even though the
    approximation said it fit. `pinned` is untouched — pinned facts never
    yield to this settle step (see select_for_injection's docstring: the
    only case a pinned fact goes unrendered is pinned-alone-over-budget,
    handled before this is ever called).

    `selected_rest` is in added_turn order (see _greedy_fill_by_priority),
    not lowest-priority-last order — re-deriving true priority order here
    is overkill for a settle step that only ever trims a token or two of
    slop, so this drops the last (highest added_turn) survivor each round,
    the same bounded-iteration convergence _lru_split's own settle loop
    relies on.
    """
    while selected_rest and _estimate_tokens(
        format_facts_block(pinned + selected_rest)
    ) > max_tokens:
        selected_rest = selected_rest[:-1]
    return pinned + selected_rest


def _select_rest(
    candidates: list[dict],
    bullets_budget: int,
    query_text: str | None,
    embedder: Callable[[list[str]], list[list[float]] | None] | None,
) -> list[dict]:
    """Select from the NON-pinned facts against a bullets-only budget (no
    header — the header is accounted once by select_for_injection). Ranks by
    relevance to `query_text` when possible; degrades to LRU order (most-
    recently-used first, same key _lru_split sorts eviction by) otherwise —
    covers "no query_text", "no embedder available", and "embedding call
    failed" with one fallback path.
    """
    if not candidates or bullets_budget <= 0:
        return []
    priority = _relevance_order(candidates, query_text, embedder)
    if priority is None:
        priority = sorted(
            candidates, key=lambda f: (f["last_used"], f["added_turn"]), reverse=True
        )
    return _greedy_fill_by_priority(priority, bullets_budget)


# The pin guarantee's REAL contract, stated once here because the docstring
# below promises "always injected" and one layer up can still void it:
# main._bound_injected_blocks drops WHOLE LAYERS when the combined injection
# budget is exceeded, and facts is priority 2 of 4. So over-pinning converts
# "a slightly oversized facts block" into "no facts at all this turn, pinned
# included" - the opposite of the intent. Pin the handful that must always
# land, not everything that seems important.
def select_for_injection(
    facts: list[dict],
    max_tokens: int = _INJECT_FACTS_TOKENS,
    *,
    query_text: str | None = None,
    embedder: Callable[[list[str]], list[list[float]] | None] | None = None,
) -> list[dict]:
    """Return the subset of `facts` to inject into this turn's prompt.

    v3.1.4 F1. Two tiers:

      1. PINNED facts (`f["pin"]` truthy — see the fact-shape comment near
         the top of this module) are ALWAYS included, bypassing ranking and
         the budget split entirely. Only when the pinned set alone costs
         MORE than `max_tokens` does this go over budget — deliberately: a
         dropped identity fact is the "she forgot me" failure this tier
         exists to prevent, and that failure is worse than a slightly
         oversized system block. Logged when it happens.
      2. The REST is ranked against `query_text` (typically the current
         user turn) using the SAME bge-small embedding retrieval.py already
         computes for episodic retrieval — `retrieval._embed`, CPU-only,
         milliseconds, no GPU, no new dependency — and the top-scoring facts
         that fit the remaining budget are kept. `embedder` overrides which
         embedding function is used (tests use this instead of
         monkeypatching `retrieval._embed`, though `patch.object(retrieval,
         "_embed", ...)` — this module's own import of retrieval, mirroring
         dedup.py's — works too).

    GRACEFUL DEGRADATION, exactly as this codebase's other embedding
    consumer (retrieval.py's own docstring: "Everything degrades to a safe
    no-op"):
      - `query_text=None` (the default — EVERY caller that has not been
        updated to pass the current turn's text) → no ranking is attempted.
        With no pinned facts either, this is BYTE-FOR-BYTE the pre-F1
        behaviour: `_lru_split(facts, max_tokens)`'s kept half, nothing
        else. This is the fast path taken on every call site until main.py
        is updated to pass `query_text=<current user turn>` — see this
        module's F1 report for the exact one-line change.
      - `query_text` given but embedding unavailable/fails this turn → the
        REST falls back to LRU order (most-recently-used first), same as
        before. Chat is never blocked or broken by a memory-ranking
        failure.
      - Pinned facts present → always included regardless of either case
        above; only the REST's selection method depends on query_text/
        embedder availability.

    Callers touch (and only touch) what this returns. That is what gives
    `last_used` any signal at all: while the store fits the budget this is
    the whole store and nothing is evicted anyway, but the moment it does
    not — which is exactly when eviction starts choosing — the facts left
    out stop being refreshed and become the eviction candidates, instead of
    every fact carrying an identical timestamp and eviction falling through
    to added_turn (v3.1 F9). Ranking by relevance instead of injecting
    everything is what makes that happen in practice, not just in theory —
    only what's actually relevant (plus what's pinned) gets touched, so LRU
    finally selects for "keeps mattering" instead of "was added early."

    The returned dicts are the SAME objects as the ones passed in, so
    touch_facts() on this subset updates them in the caller's full list too.
    """
    if not facts:
        return []

    pinned = sorted(
        (f for f in facts if f.get("pin")), key=lambda f: f["added_turn"]
    )

    if not pinned and not query_text:
        # Nothing pinned (every record written before this field existed —
        # or simply a conversation that has never used it — reads as
        # unpinned) and no ranking requested: the exact pre-F1 code path,
        # unconditionally. This is the guarantee "callers that pass nothing
        # get the current behaviour" rests on.
        kept, _ = _lru_split(facts, max_tokens)
        return kept

    rest = [f for f in facts if not f.get("pin")]

    if not pinned:
        # Ranking requested, nothing pinned to reserve budget for: rest gets
        # the WHOLE budget, header included — same header accounting
        # _lru_split itself uses, so hand it the header allowance directly
        # rather than inventing a second copy of that degenerate-budget
        # check ("too small even for the header" — see _lru_split).
        if _FACTS_BLOCK_HEADER_TOKENS > max_tokens:
            return []
        selected_rest = _select_rest(
            rest, max_tokens - _FACTS_BLOCK_HEADER_TOKENS, query_text, embedder
        )
        return _settle_against_budget([], selected_rest, max_tokens)

    pinned_cost = _estimate_tokens(format_facts_block(pinned))  # header + pinned bullets
    if pinned_cost >= max_tokens:
        if pinned_cost > max_tokens:
            logger.warning(
                f"{len(pinned)} pinned fact(s) alone cost ~{pinned_cost} "
                f"estimated tokens against a {max_tokens}-token injection "
                f"budget; injecting them anyway rather than dropping an "
                f"identity-tier fact — consider raising "
                f"COMPACTOR_INJECT_FACTS_TOKENS or pinning fewer facts"
            )
        return pinned

    # Budget left for `rest`, in bullets-only units: pinned_cost already
    # paid for the header once, and the final combined block pays it only
    # once too (format_facts_block never emits the header twice).
    rest_bullets_budget = max_tokens - pinned_cost
    selected_rest = _select_rest(rest, rest_bullets_budget, query_text, embedder)
    return _settle_against_budget(pinned, selected_rest, max_tokens)


def prune_facts(
    facts: list[dict],
    max_tokens: int = _MAX_FACTS_TOKENS,
    *,
    conv_id: str | None = None,
) -> tuple[list[dict], int]:
    """Trim facts down to fit max_tokens. LRU eviction by `last_used`
    (least-recently-used dropped first). Returns (kept, dropped_count).

    Pass `conv_id` and the evicted facts are moved to the archive sidecar
    before this returns — recoverable with restore_from_archive. Without it
    they are only reachable from this log line, because archive_stale_facts
    and restore_from_archive are wired to admin endpoints and nothing calls
    them on a schedule, so an eviction here was a permanent delete (v3.1 F9).
    Every caller should pass it.

    If the archive write fails, NOTHING is evicted: the store stays over
    budget and the caller writes it back whole. Over budget is a soft cap on
    a block select_for_injection bounds anyway; losing the facts is not
    recoverable at all.
    """
    kept, evicted = _lru_split(facts, max_tokens)
    if not evicted:
        return kept, 0

    if conv_id is None:
        # No sidecar to write to. Preserve the pre-v3.1 behaviour rather than
        # leaving the caller with an over-budget list it does not expect, but
        # put the texts in the log so the eviction is at least forensically
        # recoverable.
        # Bounded deliberately. Dumping every text unbounded measured 40 KB per
        # call on a 300-fact store — ~3.8 MB per hundred turns into the same
        # log a human is meant to find this line in. A forensic record that
        # buries the record is not one.
        _SHOWN = 5
        shown = "; ".join(repr(f["text"]) for f in evicted[:_SHOWN])
        more = (
            f" … and {len(evicted) - _SHOWN} more (not logged; pass conv_id to "
            f"archive them instead)"
            if len(evicted) > _SHOWN else ""
        )
        logger.warning(
            f"prune_facts called without conv_id — {len(evicted)} fact(s) "
            f"evicted with no archive to land in: {shown}{more}"
        )
        return kept, len(evicted)

    try:
        archive_facts(conv_id, evicted)
    except Exception as e:
        logger.error(
            f"conv={conv_id}: could not archive {len(evicted)} evicted fact(s) "
            f"({e}); kept them in the active store over budget rather than "
            f"deleting them"
        )
        return list(facts), 0

    logger.info(
        f"conv={conv_id}: archived {len(evicted)} fact(s) evicted for budget "
        f"({len(kept)} still active)"
    )
    return kept, len(evicted)


# ---------------------------------------------------------------------------
# Injection — turn facts into a system message block for the LLM request
# ---------------------------------------------------------------------------

# v3.1.5 — this block's authority is over what is TRUE, and nothing else.
# It used to end "maintain consistency with these", which arrived every turn
# above ~91 bullets and was read as a request for consistency of EXPRESSION
# as well as of fact. See persona.py's _PERSONA_BLOCK_HEADER for the full
# division of labour between the four injected blocks and why the wording is
# positive rather than prohibitive.
_FACTS_BLOCK_HEADER = (
    "[Persistent facts about this conversation — established earlier. Stay "
    "accurate to these; the wording is yours.]"
)
# Measured off the header itself, not hand-counted, so editing the header
# text cannot silently invalidate the budget in _lru_split — the same
# reasoning as _extraction_overhead_tokens() below for the extraction
# payload's own fixed overhead. +1 char for the line break the header always
# carries when any fact follows it (format_facts_block never emits the
# header alone), a safe, negligible over-count in the same direction as
# _fact_bullet_tokens' own trailing newline.
_FACTS_BLOCK_HEADER_TOKENS = _estimate_tokens(_FACTS_BLOCK_HEADER + "\n")


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
    """Mark every fact in the list you pass as just-used (for LRU). Mutates
    in place AND returns the list for chaining.

    Pass the INJECTED subset — what select_for_injection() returned — not the
    whole store. Touching everything loaded is not LRU tracking; it stamps
    every fact with the same second on every single request, which is how
    eviction came to sort on added_turn and delete the conversation's
    foundational facts (v3.1 F9). The subset shares its dicts with the full
    list, so the untouched facts keep their real last_used and the caller can
    still save the whole thing.
    """
    ts = now if now is not None else _now_unix()
    for f in facts:
        f["last_used"] = ts
    return facts


def set_pinned(
    facts: list[dict], *, text_substring: str, pinned: bool = True
) -> int:
    """Set (or clear) the `pin` flag on every fact whose text contains
    `text_substring` (case-insensitive substring match — same convention as
    restore_from_archive's text_substring). Returns how many facts changed.

    Mutates the dicts in place and returns a count, the same contract as
    touch_facts: this does not read or write storage itself. Caller owns
    load → set_pinned → save_facts under conv_lock, same as every other
    read-modify-write in this module.

    The data-layer primitive behind the /pin and /unpin commands
    (commands.py), which hold conv_lock around load -> set_pinned -> save.
    Callers must do the same: an unlocked read-modify-write here races the
    extraction tail in both directions.
    """
    needle = text_substring.lower()
    changed = 0
    for f in facts:
        if needle in f.get("text", "").lower():
            if bool(f.get("pin", False)) != pinned:
                changed += 1
            f["pin"] = pinned
    return changed


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


_TRIM_NOTE = "\n[...trimmed to fit the fact-extraction input budget]"


def _assembled_tokens(
    user_msg: str, assistant_msg: str, existing_facts: list[dict]
) -> int:
    """Estimated size of the request as _build_extraction_messages will
    actually assemble it — system prompt, scaffolding, per-fact bullets and all.

    Every shedding decision below is checked against this rather than against a
    sum of the parts, for the same reason main._enforce_hard_budget verifies
    with a real count after it sheds: the framing is not free, and a budget
    that only counts the content is the budget that let 33k-token payloads out.
    """
    return sum(
        _estimate_tokens(m["content"])
        for m in _build_extraction_messages(user_msg, assistant_msg, existing_facts)
    )


def _extraction_overhead_tokens() -> int:
    """Everything in the payload that is neither the store nor the exchange:
    the system prompt plus the literal scaffolding.

    Measured off the templates themselves rather than written down as a
    constant, so editing either one cannot silently invalidate the budget.
    """
    return _assembled_tokens("", "", [])


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut `text` down to roughly `max_tokens`, keeping the head and saying so.

    Head rather than tail: the extraction prompt asks for what the USER
    stated, and a reply states its answer up front and elaborates below.
    Marked rather than silent: a sentence that stops mid-word looks to the
    model like all there was, which is how a truncation becomes a wrong fact.
    Never returns "" — extract_facts_from_exchange short-circuits on an empty
    message, and shedding a turn into a silent no-op is the failure mode this
    whole budget exists to remove.
    """
    if _estimate_tokens(text) <= max_tokens:
        return text
    # Leave room for the note itself, plus one token for the char/4 remainder
    # the slice can carry past the boundary.
    keep_chars = max(0, max_tokens - _estimate_tokens(_TRIM_NOTE) - 1) * 4
    return (text[:keep_chars].rstrip() + _TRIM_NOTE).lstrip()


def _fit_extraction_input(
    user_msg: str,
    assistant_msg: str,
    existing_facts: list[dict],
    budget: int,
) -> tuple[str, str, list[dict], str | None]:
    """Trim one extraction payload to `budget` estimated tokens.

    Returns (user_msg, assistant_msg, existing_facts, note); `note` is None
    when everything fit as passed and a description of what was shed otherwise.

    Shedding order is by value, cheapest first — the same principle as
    main._enforce_hard_budget, with this call's own ordering:

      1. the fact store. It is already on disk. Anything the model re-extracts
         because it could not see it here is caught by dedup on the way back
         in, so the cost of shedding it is a duplicate, not a loss.
      2. the assistant reply. The prompt explicitly asks for what the USER
         stated and not what the assistant invented, and the reply is also the
         unbounded half — the chat path sends no max_tokens.
      3. the user message, last. It is the new information and the reason the
         call exists.

    The store is narrowed through select_for_injection so eviction, injection
    and extraction all agree on which facts are the working set, rather than
    this function inventing a third opinion.
    """
    if _assembled_tokens(user_msg, assistant_msg, existing_facts) <= budget:
        return user_msg, assistant_msg, existing_facts, None

    notes: list[str] = []

    # 1. The store gets whatever the exchange and the scaffolding leave it.
    kept = select_for_injection(
        existing_facts,
        max(0, budget - _assembled_tokens(user_msg, assistant_msg, [])),
    )
    # select_for_injection sizes a fact as its text alone — the unit it shares
    # with prune_facts — while this payload renders it as "- <text>\n". Settle
    # that difference against the assembled payload instead of carrying a
    # second cost model around. It is the per-bullet framing, so this drops a
    # few facts and stops; `kept` shrinks every round, so it terminates.
    while kept and _assembled_tokens(user_msg, assistant_msg, kept) > budget:
        lru = min(kept, key=lambda f: (f["last_used"], f["added_turn"]))
        kept = [f for f in kept if f is not lru]
    if len(kept) != len(existing_facts):
        notes.append(f"store {len(existing_facts)}->{len(kept)} fact(s)")

    # 2. The assistant reply. Two rounds: one to make the cut, one to absorb
    #    the cost of the note the cut adds.
    a_before = _estimate_tokens(assistant_msg)
    for _ in range(2):
        over = _assembled_tokens(user_msg, assistant_msg, kept) - budget
        if over <= 0:
            break
        assistant_msg = _truncate_to_tokens(
            assistant_msg, _estimate_tokens(assistant_msg) - over
        )
    a_after = _estimate_tokens(assistant_msg)
    if a_after != a_before:
        notes.append(f"assistant reply {a_before}->{a_after} tok")

    # 3. The user message, last.
    u_before = _estimate_tokens(user_msg)
    for _ in range(2):
        over = _assembled_tokens(user_msg, assistant_msg, kept) - budget
        if over <= 0:
            break
        user_msg = _truncate_to_tokens(user_msg, _estimate_tokens(user_msg) - over)
    u_after = _estimate_tokens(user_msg)
    if u_after != u_before:
        notes.append(f"user turn {u_before}->{u_after} tok")

    if _assembled_tokens(user_msg, assistant_msg, kept) > budget:
        # Reachable only when the window itself is smaller than the extraction
        # prompt, i.e. a misconfigured MAX_MODEL_LEN. Say so here, because the
        # next thing in the log will be the rejection and nothing else would
        # explain why shedding everything did not help.
        notes.append(
            f"STILL over the {budget}-token budget with nothing left to shed — "
            f"the configured window cannot hold the extraction prompt itself"
        )

    return user_msg, assistant_msg, kept, "; ".join(notes) or None


# ---------------------------------------------------------------------------
# What is not a fact — the write-path structure filter (v3.1 D5)
# ---------------------------------------------------------------------------
#
# HOW THE SCAFFOLDING GOT IN. Not by splitting an assistant reply: nothing in
# this codebase does that. `_parse_extraction_output` splits the EXTRACTION
# MODEL'S OWN OUTPUT on newlines, and until this filter it kept every line of
# six characters or more whether or not the model had honoured the "- " bullet
# contract the system prompt demands. The extractor is handed the assistant's
# reply verbatim in the `[assistant]:` half of its input; when that reply is a
# formatted dashboard, the extractor mirrors the formatting — a heading, a rule,
# a fenced block — and the parser promoted every one of those lines to a stored
# fact. `_EXTRACTION_MAX_TOKENS` (256) then cut the reply mid-structure, which
# is why the store holds openers like a bare fence line and a JSON key with an
# unclosed bracket rather than whole blocks.
#
# So there are two failures, and only the second one is ours to fix here: the
# model breaks its output contract, and the parser has no contract to break.
# A prompt cannot be relied on; a structural gate can.
#
# WHY IT MATTERS BEYOND TIDINESS. `_estimate_tokens` prices a fact at char/4.
# INCIDENT_2026-08-28 measured 1,710 U+2501 plus 441 U+2500 — 2,151 characters —
# at roughly 4,275 real tokens, i.e. ~1.99 tokens per character where char/4
# assumes 0.25. Decoration is therefore underpriced by about 8x by the very
# estimator that enforces _MAX_FACTS_TOKENS, so a store full of rules and
# dashboard rows passes a 1,500-token budget while costing tens of thousands.
# A horizontal rule stored as memory is the box-drawing input that caused the
# incident, re-injected on every turn — the system feeding itself its own worst
# input.
#
# THE RULES ARE DELIBERATELY STRUCTURAL AND CONSERVATIVE. Over-filtering
# destroys real memory, and the memory here is not recoverable from anywhere
# else. Every rule below is a syntax judgement, never a topic judgement:
# an emoji alone does not disqualify a line, a colon alone does not, an
# ALL-CAPS acronym alone does not, and a percentage alone does not. The
# dashboard rule fires only on the conjunction of all three of an ALL-CAPS
# label, a value with no lowercase prose in it, and a metric or an arrow —
# so a synthetic "MOOD: 7/10 -> 9/10" is refused while a synthetic
# "PTSD: symptoms improved 40% since 2019" is kept.
#
# WRITE PATH ONLY. This is deliberately NOT applied in load_facts or
# save_facts, which is the tempting place to put it. Every turn's tail does
# load → append → prune → save, so a filter there would silently delete
# already-stored entries on the next unrelated write — an irreversible
# cleanup of a live store, smuggled in as a parser fix. Cleaning what is
# already stored is a separate, deliberate, reversible operation.

# A fenced block opener/closer: ``` or ~~~ (any longer run too).
_FENCE_RE = re.compile(r"^(?:`{3,}|~{3,})")

# An ATX markdown heading: one to six '#' followed by space or end of line.
# "#1 priority ..." is not a heading and is not matched.
_HEADING_RE = re.compile(r"^#{1,6}(?:\s|$)")

# A JSON object key at the start of the line: "some_key": <json value>
#
# The colon alone is NOT enough, and the earlier comment here ("English
# sentences do not open with a double-quoted token followed by a colon") is
# simply false of the gloss form. These are facts, and this filter would have
# deleted them:
#     "Little Bear": her grandmother's name for her
#     "One day at a time": the phrase she repeats when panicking
# Given what this store holds — coping phrases, names of the dead — a quoted
# phrase followed by an explanation is among the LAST things that should be
# discarded. The old rule was also inconsistent on typography: it matched the
# ASCII quote only, so the same fact with curly quotes was kept.
#
# So require the remainder to look like JSON rather than like prose. A key
# followed by an object, array, string, number, boolean, null, or nothing is
# scaffolding; a key followed by a lowercase English word is a definition.
#
# This matters more under /retire than at extraction: here a false positive
# drops one incoming line, there it DELETES something already stored.
_JSON_KEY_RE = re.compile(
    r'^"[^"]*"\s*:\s*(?:[\[{"]|-?\d|true\b|false\b|null\b|$)'
)

# An ALL-CAPS label followed by a colon, optionally behind a few leading
# non-alphanumerics (an emoji, a bullet glyph, a box character). The label
# alone proves nothing — see _reject_reason for the conjunction it is part of.
_DASHBOARD_LABEL_RE = re.compile(
    r'^[^0-9A-Za-z]{0,8}([A-Z][A-Z0-9 _/&.\'"-]{1,39}):\s*(\S.*)$'
)

# Transition arrows, the shape of a "was -> now" dashboard cell.
_ARROW_RE = re.compile(r"(?:-{1,2}>|={1,2}>|→|⇒|⇨|⟶)")

# A percentage attached to a number ("88%"), not a bare '%' character.
_PERCENT_RE = re.compile(r"\d\s*%")


def _reject_reason(text: str) -> str | None:
    """Return why `text` is structure rather than a fact, or None if it is
    storable. The reason string is for logs; callers should treat any
    non-None as "do not store".
    """
    line = text.strip()
    if not line:
        return "empty"
    # No letter and no digit in any script: horizontal rules of U+2501/U+2500,
    # "---", "***", "===", "|---|---|", a lone bracket, an emoji-only line.
    # A fact in any language contains at least one alphanumeric character.
    if not any(ch.isalnum() for ch in line):
        return "no alphanumeric content"
    if _FENCE_RE.match(line):
        return "code fence"
    if _HEADING_RE.match(line):
        return "markdown heading"
    if _JSON_KEY_RE.match(line):
        return "json object key"
    # A line that ends on an opening bracket is an unclosed structure opener,
    # which is what a 256-token cut through a JSON block leaves behind.
    if line.endswith(("[", "{")):
        return "unclosed structure opener"
    m = _DASHBOARD_LABEL_RE.match(line)
    if m:
        value = m.group(2)
        if not any(ch.islower() for ch in value) and (
            _ARROW_RE.search(value) or _PERCENT_RE.search(value)
        ):
            return "status-dashboard line"
    return None


def is_storable_fact(text: str) -> bool:
    """True when `text` is shaped like a fact rather than like markup.

    Public because this module is not the only write path into the store:
    commands._handle_remember, backfill and dedup's merged canonical text all
    mint fact text too, and they should share one definition rather than grow
    three. Read-only callers must NOT use it to filter what is already stored.
    """
    return _reject_reason(text) is None


def _parse_extraction_output(raw: str, *, where: str = "") -> list[str]:
    """Parse the LLM's output into a clean list of fact strings. Handles:
    - "NONE" (any casing, with/without trailing punctuation) → []
    - Bullets prefixed with -, *, • → stripped
    - Numbered lists "1. ..." → stripped
    - Blank lines → skipped
    - Lines too short to be a fact (< 6 chars) → skipped
    - Lines that are markup rather than a fact → skipped, and logged (D5)

    The markup check runs AFTER the bullet prefix is stripped, because a model
    that mirrors formatting also bullets it: "- ## Current Status" has to be
    refused on the heading, not accepted on the dash.
    """
    if not raw or not raw.strip():
        return []
    cleaned = raw.strip()
    if cleaned.upper().rstrip(".").strip() == "NONE":
        return []
    out: list[str] = []
    rejected: list[tuple[str, str]] = []
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
        if len(line) < 6:
            continue
        reason = _reject_reason(line)
        if reason:
            rejected.append((reason, line))
            continue
        out.append(line)
    if rejected:
        # Named and bounded. Silence here is what let half a store fill with
        # markup unnoticed; an unbounded dump is what prune_facts already
        # learned not to do.
        _SHOWN = 3
        shown = "; ".join(
            f"{reason}: {line[:60]!r}" for reason, line in rejected[:_SHOWN]
        )
        more = (
            f" … and {len(rejected) - _SHOWN} more"
            if len(rejected) > _SHOWN else ""
        )
        prefix = f"{where}: " if where else ""
        logger.info(
            f"{prefix}extraction returned {len(rejected)} line(s) of markup "
            f"rather than facts; not stored — {shown}{more}"
        )
    return out


async def extract_facts_from_exchange(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    user_msg: str,
    assistant_msg: str,
    existing_facts: list[dict],
    *,
    conv_id: str | None = None,
    max_input_tokens: int | None = None,
    timeout: float = 120.0,
) -> list[str]:
    """Call vLLM to extract new facts from one user/assistant exchange.

    Returns a list of fact strings (possibly empty). Caller is responsible
    for assigning added_turn / last_used and appending to the facts file.

    Errors (network, vLLM 5xx, parse failures) return [] — fact extraction
    must NEVER block or break the user's chat flow. Failures get logged.

    `existing_facts` may be the whole store: the input budget below narrows it
    here, so a caller that has not yet learned to pass a bounded set cannot
    push this call past the model's window. Passing the bounded set — what
    select_for_injection returned for this turn — is still better, because
    then the store the extractor is told about is the same one the model just
    saw.

    `conv_id` is used for logging only, and only this module's own logging. It
    is what makes a failure attributable: without it the warning this used to
    emit named no conversation, so a lost extraction could not be traced to
    the turn that lost it.

    `max_input_tokens`, when given, only overrides the INITIAL estimate-unit
    guess handed to _fit_extraction_input — the module's own default
    (`_extraction_input_budget_estimate_units()`) picks generous or
    conservative depending on whether tokens.py can verify the result
    afterward, and a caller skips that choice by supplying its own number
    directly. The real-tokenizer backstop below still runs regardless of
    where the initial guess came from: it protects the model's REAL window,
    which no estimate-unit budget — caller-supplied or not — should be able
    to sign away.
    """
    if not user_msg or not assistant_msg:
        return []
    # A lost extraction is a lost memory, so every line below has to say WHOSE.
    where = f"conv={conv_id}" if conv_id else "conv=? (caller passed none)"
    budget = (
        _extraction_input_budget_estimate_units()
        if max_input_tokens is None else max_input_tokens
    )
    trimmed_user, trimmed_assistant, trimmed_facts, trim_note = _fit_extraction_input(
        user_msg, assistant_msg, existing_facts, budget
    )
    if trim_note:
        logger.info(f"{where}: extraction input trimmed to budget — {trim_note}")

    # Real-measurement backstop. _fit_extraction_input only ever sees the
    # char/4 estimate, which _ASSISTANT_CONTENT_ESTIMATE_LOW_FRACTION above
    # already prices for the worst case MEASURED across content shapes — but
    # "worst case measured in general" is still a guess about THIS specific
    # exchange. When the real tokenizer is available (tokens.py — the same
    # tekken vocabulary vLLM uses), verify directly and re-trim toward the
    # true ratio rather than trust either direction of the guess: this is
    # what actually fixes the common case. The constant above is what is left
    # standing when this cannot run at all (mistral_common absent, no cached
    # tekken.json, or a non-Mistral model) — see tokens.py's own doctrine,
    # "everything degrades to None".
    for _retry in range(_MAX_REAL_TOKEN_MEASURE_RETRIES):
        real = tokens.count(
            _build_extraction_messages(trimmed_user, trimmed_assistant, trimmed_facts)
        )
        if real is None or real <= _EXTRACTION_INPUT_BUDGET_REAL_TOKENS:
            break
        # The estimate under-priced THIS exchange. Shrinking `budget`
        # PROPORTIONALLY TO ITSELF (budget * ceiling / real) does not
        # converge when the true density is far off char/4's: on pure
        # decoration the reply alone can estimate at a few thousand tokens
        # while costing tens of thousands for real, so a budget shrunk by
        # only the ceiling/real ratio can still land far ABOVE that small
        # estimate — _fit_extraction_input's own "does it already fit?"
        # check then sees nothing to shed, `real` never moves, and the loop
        # burns its retries without changing anything (measured: this
        # reproduces the 2026-08-24 class of incident again for decoration
        # specifically, silently, even with this backstop wired in).
        #
        # So derive the new target from the OBSERVED density of what was
        # actually just measured (real tokens per estimated token for THIS
        # content), not from the ceiling/real ratio applied to the old
        # budget. That number, divided into the ceiling, is the estimate-unit
        # target that forces the SAME shedding logic to cut deep enough —
        # assuming the density stays roughly uniform in what gets shed next,
        # which is the same assumption _fit_extraction_input's own char/4
        # estimate already makes throughout.
        current_estimate = _assembled_tokens(trimmed_user, trimmed_assistant, trimmed_facts)
        if current_estimate <= 0:
            break  # nothing left to shrink an estimate-unit budget against
        density = real / current_estimate
        budget = max(256, int(_EXTRACTION_INPUT_BUDGET_REAL_TOKENS / density * 0.9))
        trimmed_user, trimmed_assistant, trimmed_facts, trim_note = _fit_extraction_input(
            user_msg, assistant_msg, existing_facts, budget
        )
        logger.info(
            f"{where}: real tokenizer measured {real} tokens against a "
            f"{_EXTRACTION_INPUT_BUDGET_REAL_TOKENS}-token window ({density:.2f} "
            f"real tokens per estimated token on this content) — the char/4 "
            f"estimate under-priced this exchange; re-trimmed to a "
            f"{budget}-token estimate budget"
        )
    else:
        logger.warning(
            f"{where}: real tokenizer still reports over budget after "
            f"{_MAX_REAL_TOKEN_MEASURE_RETRIES} re-trim(s); sending the most-"
            f"trimmed attempt rather than holding the exchange forever"
        )

    payload = {
        "model": model,
        "messages": _build_extraction_messages(trimmed_user, trimmed_assistant, trimmed_facts),
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
        choice = data["choices"][0]
        raw = as_sent = choice["message"]["content"]
        # A reply cut off at _EXTRACTION_MAX_TOKENS ends mid-line, and that
        # partial line is not a fact — it is the fragment that put an unclosed
        # JSON opener in the store (D5). Drop only the incomplete tail, and
        # only when the model actually ran out of budget: this module already
        # holds that a truncation which does not announce itself becomes a
        # wrong fact (see _truncate_to_tokens). A trailing newline means the
        # last line completed before the cut, so nothing is dropped.
        if choice.get("finish_reason") == "length" and raw and not raw.endswith("\n"):
            head, sep, tail = raw.rpartition("\n")
            raw = head if sep else ""
            logger.info(
                f"{where}: extraction hit the {_EXTRACTION_MAX_TOKENS}-token "
                f"output cap; dropped the incomplete final line "
                f"({len(tail)} chars) rather than storing a half fact"
            )
        facts = _parse_extraction_output(raw, where=where)
        if facts:
            logger.info(f"{where}: extracted {len(facts)} new fact(s)")
        else:
            # Empty result has two distinct causes; log both so the
            # silence isn't a diagnostic dead-end during integration runs.
            # `as_sent`, not the tail-trimmed `raw`: when zero facts survive
            # this line is the only record of what the model actually said.
            snippet = (as_sent or "").strip().replace("\n", " ")[:120]
            logger.info(
                f"{where}: extracted 0 fact(s) — model returned: {snippet!r}"
            )
        return facts
    except Exception as e:
        # ERROR, not WARNING. This is the whole of what the 2026-08-24 window
        # left behind for two rejected extractions: one unattributed warning
        # each, no "+N facts" line after it, no retry and no alert. The chat
        # response had already gone out, so nothing else in the system knew
        # anything was missing. The facts are gone for good: backfill is the
        # only other thing that ever re-reads an old exchange, and
        # backfill.needs_backfill returns False as soon as a conversation has
        # a facts file, so it will never revisit this turn.
        logger.error(
            f"{where}: fact extraction FAILED ({e}) — this exchange's facts "
            f"are lost; there is no retry"
        )
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
                client, vllm_url, model, user_msg, assistant_msg, existing,
                conv_id=conv_id,
            )
            if not new_strs:
                return 0
            now = _now_unix()
            new_entries = [
                {"text": s, "added_turn": turn_index, "last_used": now, "pin": False}
                for s in new_strs
            ]
            combined = existing + new_entries
            kept, dropped = prune_facts(combined, conv_id=conv_id)
            save_facts(conv_id, kept)
            if dropped:
                logger.info(
                    f"conv={conv_id}: +{len(new_entries)} facts, archived {dropped} "
                    f"least-recently-used"
                )
            return len(new_entries)
        except StoreUnreadable as e:
            # The existing store is there and we could not read it, so
            # `existing` is unknown, not empty. Writing now would replace the
            # whole store with this one exchange (v3.1 F1a).
            logger.error(
                f"conv={conv_id}: facts file unreadable ({e}); skipped the "
                f"fact write rather than overwriting it"
            )
            return 0
        except Exception as e:
            logger.exception(f"record_facts_for_exchange failed (non-fatal): {e}")
            return 0

    return await with_facts_lock(conv_id, _run)
