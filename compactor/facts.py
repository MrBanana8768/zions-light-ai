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
eviction by `last_used` keeps the injected block under
COMPACTOR_MAX_FACTS_TOKENS. Eviction MOVES facts to the archive sidecar —
it does not delete them (v3.1 F9).

Lifecycle:
  1. Request arrives → load_facts(conv_id) → select_for_injection() → inject
     the selected subset as a system block
  2. Mark ONLY the injected subset as `last_used = now` (LRU tracking).
     Touching the whole store is what made `last_used` uniform across every
     fact, which collapsed the eviction sort key onto `added_turn` and made
     "LRU" mean "drop the conversation's oldest, most foundational facts"
     (v3.1 F9).
  3. After response streams back → extract_facts_from_exchange() in async tail.
     Its INPUT is budgeted too (_fit_extraction_input): the store it is handed
     may be the whole file, and the reply it is handed is unbounded.
  4. Append new facts → prune to budget (archiving the evictions) → save_facts()

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
    StoreUnreadable,
    atomic_write_json,
    conv_lock,
    facts_archive_path,
    facts_path,
    read_json_strict,
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
# Slack left inside the window for the chat template's per-message framing and
# for the estimator's error. Same default and same purpose as
# COMPACTOR_SUMMARY_INPUT_RESERVE; this module has no tokenizer at all, so its
# char/4 estimate is the coarser of the two and this reserve is the only thing
# absorbing that difference.
_EXTRACTION_INPUT_RESERVE = int(
    os.environ.get("COMPACTOR_FACTS_INPUT_RESERVE", "2048") or 2048
)
# Clamped for the same reason as main.HARD_INPUT_LIMIT: a bare floor could sit
# ABOVE the model's own window on a small-context model, which would budget
# nothing at all.
_EXTRACTION_INPUT_BUDGET = min(
    _MAX_MODEL_LEN,
    max(256, _MAX_MODEL_LEN - _EXTRACTION_MAX_TOKENS - _EXTRACTION_INPUT_RESERVE),
)


def extraction_enabled() -> bool:
    return _EXTRACTION_ENABLED


# ---------------------------------------------------------------------------
# Fact shape + token estimation
# ---------------------------------------------------------------------------

# A fact is a dict — using TypedDict-style for clarity but plain dict for
# JSON round-trip simplicity.
#   { "text": str, "added_turn": int, "last_used": int (unix ts) }
#
# `last_used`: unix seconds, set by touch_facts() on the facts INJECTED into
# a turn. One unit, one writer. Safe to compare across facts.
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
    """
    if not facts:
        return [], []
    total = sum(_estimate_tokens(f["text"]) for f in facts)
    if total <= max_tokens:
        return list(facts), []

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
    kept_ids = {id(f) for f in kept_reversed}
    # Restore original-ish ordering by added_turn for stable injection
    kept = sorted(kept_reversed, key=lambda f: f["added_turn"])
    evicted = sorted(
        (f for f in facts if id(f) not in kept_ids), key=lambda f: f["added_turn"]
    )
    return kept, evicted


def select_for_injection(
    facts: list[dict],
    max_tokens: int = _MAX_FACTS_TOKENS,
) -> list[dict]:
    """Return the subset of `facts` to inject into this turn's prompt.

    Callers touch (and only touch) what this returns. That is what gives
    `last_used` any signal at all: while the store fits the budget this is
    the whole store and nothing is evicted anyway, but the moment it does
    not — which is exactly when eviction starts choosing — the facts left
    out stop being refreshed and become the eviction candidates, instead of
    every fact carrying an identical timestamp and eviction falling through
    to added_turn (v3.1 F9).

    The returned dicts are the SAME objects as the ones passed in, so
    touch_facts() on this subset updates them in the caller's full list too.
    """
    kept, _ = _lru_split(facts, max_tokens)
    return kept


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
    """
    if not user_msg or not assistant_msg:
        return []
    # A lost extraction is a lost memory, so every line below has to say WHOSE.
    where = f"conv={conv_id}" if conv_id else "conv=? (caller passed none)"
    user_msg, assistant_msg, existing_facts, trim_note = _fit_extraction_input(
        user_msg,
        assistant_msg,
        existing_facts,
        _EXTRACTION_INPUT_BUDGET if max_input_tokens is None else max_input_tokens,
    )
    if trim_note:
        logger.info(f"{where}: extraction input trimmed to budget — {trim_note}")
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
            logger.info(f"{where}: extracted {len(facts)} new fact(s)")
        else:
            # Empty result has two distinct causes; log both so the
            # silence isn't a diagnostic dead-end during integration runs.
            snippet = (raw or "").strip().replace("\n", " ")[:120]
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
                {"text": s, "added_turn": turn_index, "last_used": now}
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
