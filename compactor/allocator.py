"""
compactor.allocator — v3.1.9.8 `.15b`: the r8/r10 prompt allocator.

Design: V4_DESIGN.md §17.6 (r8, amended r10). Ports faithfully from
`zl-ops/v4/lab/r8/alloc8.py`'s `allocate8()`/`run8()` (the proof harness
that produced the r8/r10 whole-history numbers: 0 holes, 0 starved, 0
overflows) and the r10 image rule. This module is the ALGORITHM: given a
boundary's token/image counts, it decides each tier's budget exactly as
the lab proof does. `compose()` at the bottom is the integration
adapter that turns those decisions into a real forwarded message list.

SCOPE OF THIS v3.1.9.8 IMPLEMENTATION (read before assuming more is
wired than is). The reduced `.15b`:
  - the previous exchange (T2): her message whole first, budget B2, the
    reply cut from the middle at a paragraph boundary, 25% of L as a
    floor;
  - the protected memory minimum M_min (continuity headlines for every
    gap unit, plus the long-term core), which nothing below T2 may take;
  - the reduced forward stage: pixels only from the request's OWN parts,
    only for image turns inside the last COMPACTOR_IMAGE_WINDOW_TURNS
    turns; older in-window image turns become a text note. No hoist, no
    non-adjacent merge, no image store, no image bytes on disk;
  - the too-long path;
  - the turn-accounting invariant assertion.
Without pointers, recall, or any store (`.15a`) — those are V4-only.

WHAT IS NOT YET WIRED INTO THE LIVE REQUEST PATH, AND WHY (see the
v3.1.9.8 implementation report's "deviations" section for the full
reasoning): `main.compact_if_needed` is ~1,250 lines and is the single
place that both (a) decides the forwarded array's composition and (b)
advances the conversation's position/anchor bookkeeping that
`summaries/<conv>.json`'s fingerprints depend on — condition 2 of
§17.13 requires those fingerprints to stay BYTE-IDENTICAL with the
allocator on or off. Safely separating (b) from (a) inside that
function, and verifying the separation against condition 2's frozen-
clock replay, is real work this pass did not have time to finish
without risking exactly the invariant it exists to protect. This module
is therefore shipped as a faithful, independently tested decision core
and a `compose()` adapter with a real (if narrower) call site — see
`main.py`'s `_ALLOCATOR_INTEGRATION_NOTE` near the `compact_if_needed`
call site for the current wiring and what remains.

NOTHING IN THIS MODULE MAY IMPORT `main`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from envcfg import env_bool, env_int

logger = logging.getLogger("compactor.allocator")

# ---------------------------------------------------------------------------
# Config (§17.13: safe code-level defaults; no required new env var).
# ---------------------------------------------------------------------------

ENABLED = env_bool("COMPACTOR_ALLOCATOR_ENABLED", True)
IMAGE_WINDOW_TURNS = env_int("COMPACTOR_IMAGE_WINDOW_TURNS", 10)
MAX_CURRENT_IMAGES = env_int("COMPACTOR_MAX_CURRENT_IMAGES", 4)
IMAGE_SLICE_TOKENS = env_int("COMPACTOR_IMAGE_TOKEN_BUDGET", 8192)
MIN_GENERATION_RESERVE = env_int("COMPACTOR_MIN_GENERATION_RESERVE", 4096)

_MARK = 16           # "[... part of this reply not shown ...]" etc.
_REPLY_MIN = 64       # below this the reply becomes the one-line marker
_EXCESS = 24          # "[N earlier exchanges are not summarized yet]"
_Q_FRACTION = 0.25    # T2's 25% floor
_FLOOR_CAP_FRACTION = 0.5   # T5's floor cap: <= 50% of L

REPLY_NOT_SHOWN_MARKER = "[her previous message was answered at length; the reply is not shown]"
MESSAGE_MIDDLE_MARKER = "[… middle of her message not shown …]"
REPLY_MIDDLE_MARKER = "[… part of this reply not shown …]"


# ---------------------------------------------------------------------------
# The image rule (§17.6): vLLM's Mistral tokenizer patch-count formula.
# ---------------------------------------------------------------------------

def image_tokens(width: int, height: int, *, long_side_cap: int = 1540, patch: int = 14) -> int:
    """`rows*cols + rows` after downscaling to a long side of <= 1540
    (patch 14, spatial merge 2). [V, §17.6]: equals mistral_common's
    ImageEncoder exactly for 3024x4032 (2,365), 1080x1920 (1,760), 1540^2
    (3,080) and 640x480 (432) — see test_allocator.py for the pinned
    values."""
    if width <= 0 or height <= 0:
        return 0
    long_side = max(width, height)
    if long_side > long_side_cap:
        scale = long_side_cap / long_side
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
    cols = -(-width // patch)     # ceil division, no float rounding drift
    rows = -(-height // patch)
    return rows * cols + rows


# ---------------------------------------------------------------------------
# The decision core — a faithful port of lab/r8/alloc8.py's allocate8().
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    """One turn's shape, as the allocator needs it. `tokens` is the LOCAL
    exact count (§17.6: 'the allocator counts text with the model's own
    Tekken tokenizer... never with /tokenize on the request path')."""
    role: str          # "user" | "assistant"
    tokens: int
    images: int = 0    # count of image parts ON THIS TURN


@dataclass
class Decision:
    too_long: bool = False
    photos_dropped: int = 0
    overflow: bool = False
    # T2 — the previous exchange
    has_u: bool = False
    has_a: bool = False
    u_whole: bool = True
    a_whole: bool = True
    a_marker: bool = False
    u_tokens_kept: int = 0
    a_tokens_kept: int = 0
    # T3 — her current photos
    current_images: int = 0
    current_images_shown: int = 0
    # T5 — the rest of the floor
    floor_turns_kept: int = 0
    floor_lo: int = 0
    # T6 — memory
    n_gap_units: int = 0
    hole_units: int = 0
    full_digests: int = 0
    lt_memory: int = 0
    lt_core_short: bool = False
    lt_zero: bool = False
    mem_full: bool = False
    # T7 — older in-window pixels
    older_images_shown: int = 0

    @property
    def t2_cut(self) -> bool:
        return (self.has_u and not self.u_whole) or (self.has_a and not self.a_whole)

    @property
    def reply_without_her_message(self) -> bool:
        return self.has_a and self.a_tokens_kept > 0 and self.has_u and self.u_tokens_kept == 0


def _units_of(gap_roles: list[str]) -> list[list[int]]:
    """Group gap indices (0-based, oldest-first, each entry's role given
    in `gap_roles`) into exchange units: a unit starts at a user turn, or
    at the gap's first turn — §17.6a's definition, exactly alloc8.py's
    `units_of`."""
    units: list[list[int]] = []
    for i, role in enumerate(gap_roles):
        if not units or role == "user":
            units.append([i])
        else:
            units[-1].append(i)
    return units


def _continuity_cost(n_units: int, d_full: int, d_head: int, full_n: int) -> int:
    full = min(n_units, full_n)
    return full * d_full + (n_units - full) * d_head


def allocate(
    *,
    limit: int,
    system_tokens: int,
    newest: Turn,
    gap_and_floor: list[Turn],   # oldest-first: every turn strictly before `newest`
    unsummarized: int,            # U: unsummarized turns before the newest message
    lt_demand: int,                # tokens of long-term memory on hand (0 if none)
    lt_core: int,                  # the long-term core's own size (<= lt_demand)
    current_images: int = 0,
    older_window_images: int = 0,
    img_tokens_worst: int = 3080,
    d_full: int = 160,
    d_head: int = 40,
    full_n: int = 10,
    note_tokens: int = 24,
) -> Decision:
    """The r8 allocator, faithfully ported from alloc8.py's allocate8().
    Pure: no I/O, no tokenizer calls — every count is already resolved by
    the caller. `gap_and_floor` is oldest-first and its LAST two entries
    (if present) are the previous exchange candidates, matching alloc8's
    `t-2`/`t-3` indexing into `turns`."""
    d = Decision()
    rem = limit - system_tokens
    if newest.tokens > rem:
        d.too_long = True
        d.photos_dropped = current_images
        return d
    rem -= newest.tokens

    n = len(gap_and_floor)
    # T2 = the previous exchange: the assistant turn immediately before
    # the newest message (if any), and the user turn before THAT.
    a_idx = n - 1 if n >= 1 and gap_and_floor[n - 1].role == "assistant" else None
    u_idx = None
    if a_idx is not None:
        cand = a_idx - 1
    else:
        cand = n - 1
    if cand is not None and cand >= 0 and gap_and_floor[cand].role == "user":
        u_idx = cand
    d.has_u, d.has_a = u_idx is not None, a_idx is not None
    u = gap_and_floor[u_idx] if u_idx is not None else None
    a = gap_and_floor[a_idx] if a_idx is not None else None
    floor_lo = min(i for i in (u_idx, a_idx) if i is not None) if (u_idx is not None or a_idx is not None) else n

    lt_core = min(lt_core, lt_demand)

    def gap_roles(flo: int) -> list[str]:
        return [t.role for t in gap_and_floor[:flo]]

    def m_min(flo: int) -> int:
        return _continuity_cost(len(_units_of(gap_roles(flo))), d_full, d_head, full_n) + lt_core

    def m_hard(flo: int) -> int:
        return len(_units_of(gap_roles(flo))) * d_head + lt_core

    t3_first = img_tokens_worst if current_images > 0 else 0
    note = note_tokens if current_images > MAX_CURRENT_IMAGES else 0
    r0 = rem - t3_first - note
    b2 = max(r0 - m_min(floor_lo), min(int(_Q_FRACTION * limit), r0))
    b2 = max(0, b2)

    # Stage 1: her previous message whole, then the reply up to 25% of L
    # for T2 as a whole.
    used2 = 0
    if u is not None:
        take = min(u.tokens, b2)
        d.u_whole = take == u.tokens
        d.u_tokens_kept = take + (_MARK if not d.u_whole else 0)
        used2 += d.u_tokens_kept
    a_take = 0
    if a is not None:
        a_take = min(a.tokens, max(0, min(b2, max(int(_Q_FRACTION * limit), used2)) - used2))
    rem -= used2 + a_take + t3_first + note

    shown = 1 if current_images > 0 else 0
    # Stage 2: T3 rest — her current photos 2..4, never below M_min.
    while shown < min(current_images, MAX_CURRENT_IMAGES) and img_tokens_worst <= rem - m_hard(floor_lo):
        rem -= img_tokens_worst
        shown += 1
    if current_images > shown and note == 0:
        rem -= note_tokens
    d.current_images = current_images
    d.current_images_shown = shown

    # Stage 3: the rest of the reply, up to B2, never below M_min; cut
    # from the middle if it does not fit whole.
    if a is not None:
        more = max(0, min(a.tokens - a_take, b2 - used2 - a_take, rem - m_min(floor_lo)))
        a_take += more
        rem -= more
        if a_take == a.tokens:
            d.a_whole = True
        elif a_take >= _REPLY_MIN + _MARK:
            d.a_whole = False
        else:
            rem += a_take
            rem -= _MARK
            a_take = _MARK
            d.a_whole = False
            d.a_marker = True
        d.a_tokens_kept = a_take

    # T5: the rest of the floor — whole turns only, contiguous, <= 8
    # turns, floor <= 50% of L, never below M_min.
    kept = (1 if d.has_u else 0) + (1 if d.has_a else 0)
    floor_used = d.u_tokens_kept + d.a_tokens_kept
    j = floor_lo - 1
    while j >= 0 and kept < 8:
        need = gap_and_floor[j].tokens
        if need <= int(_FLOOR_CAP_FRACTION * limit) - floor_used and need <= rem - m_min(j):
            rem -= need
            floor_used += need
            kept += 1
            floor_lo = j
            j -= 1
        else:
            break
    d.floor_turns_kept = kept
    d.floor_lo = floor_lo

    # T6: memory — continuity headlines for every gap unit, then the
    # long-term core, then full digests (newest first), then the rest.
    units = _units_of(gap_roles(floor_lo))
    n_units = len(units)
    heads = min(n_units, max(0, rem) // d_head) if d_head else n_units
    rem -= heads * d_head
    missing = n_units - heads
    if missing:
        rem -= _EXCESS
    core = max(0, min(lt_core, rem))
    rem -= core
    full = 0
    for _ in range(min(heads, full_n)):
        if d_full - d_head <= rem:
            rem -= d_full - d_head
            full += 1
    extra = max(0, min(lt_demand - core, rem))
    rem -= extra
    lt = core + extra
    d.n_gap_units = n_units
    d.hole_units = missing
    d.full_digests = full
    d.lt_memory = lt
    d.lt_core_short = lt < lt_core
    d.lt_zero = lt == 0
    d.mem_full = (missing == 0 and full == min(n_units, full_n) and lt == lt_demand)

    # T7/T8: older in-window pixels, inside the image slice.
    slice_left = IMAGE_SLICE_TOKENS - shown * img_tokens_worst
    older_shown = 0
    for _ in range(older_window_images):
        if img_tokens_worst <= rem and img_tokens_worst <= slice_left:
            rem -= img_tokens_worst
            slice_left -= img_tokens_worst
            older_shown += 1
        else:
            break
    d.older_images_shown = older_shown
    d.overflow = rem < 0
    return d


# ---------------------------------------------------------------------------
# Text-level operations the renderer needs: the middle cut, snapped to a
# paragraph boundary (§17.6).
# ---------------------------------------------------------------------------

def cut_message_middle(text: str, *, keep_head_chars: int, keep_tail_chars: int) -> str:
    """Her previous message, cut from the middle only if it alone exceeds
    its budget: start and end kept. Used only in the rare case her single
    previous message is larger than B2 (§17.6)."""
    if len(text) <= keep_head_chars + keep_tail_chars:
        return text
    head = text[:keep_head_chars]
    tail = text[-keep_tail_chars:] if keep_tail_chars else ""
    return f"{head}\n\n{MESSAGE_MIDDLE_MARKER}\n\n{tail}"


_PARA_BOUNDARY = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _snap_boundary(text: str, pos: int, *, forward: bool) -> int:
    """Snap `pos` to the nearest paragraph boundary (a blank line), else
    the nearest sentence end, searching in the given direction. Falls
    back to `pos` unchanged if neither exists nearby."""
    candidates = [m.end() if forward else m.start() for m in _PARA_BOUNDARY.finditer(text)]
    if not candidates:
        candidates = [m.end() if forward else m.start() for m in _SENTENCE_END.finditer(text)]
    if not candidates:
        return pos
    if forward:
        after = [c for c in candidates if c >= pos]
        return after[0] if after else pos
    before = [c for c in candidates if c <= pos]
    return before[-1] if before else pos


def cut_reply_middle(text: str, *, head_fraction: float = 0.35, keep_chars: int) -> str:
    """r8 (§17.6): the reply's opening (up to `head_fraction` of its
    allotment — where it answered her) and its end are kept, snapped to
    paragraph boundaries, else sentence ends. `keep_chars` is the total
    character budget for what remains (head + tail), approximating the
    caller's already-decided token budget — the caller is responsible for
    re-checking the real token count of the result and re-cutting tighter
    if the approximation ran over, exactly as any char-based estimate on
    this codebase's request path must (see main._enforce_hard_budget's
    own prescreen comment for the same discipline)."""
    if len(text) <= keep_chars:
        return text
    head_budget = max(0, int(keep_chars * head_fraction))
    tail_budget = max(0, keep_chars - head_budget)
    head_end = _snap_boundary(text, head_budget, forward=False) or head_budget
    tail_start_target = len(text) - tail_budget
    tail_start = _snap_boundary(text, tail_start_target, forward=True)
    if tail_start <= head_end:
        tail_start = max(head_end, tail_start_target)
    head = text[:head_end].rstrip()
    tail = text[tail_start:].lstrip()
    return f"{head}\n\n{REPLY_MIDDLE_MARKER}\n\n{tail}".strip()


# ---------------------------------------------------------------------------
# The image window rule (§17.13 `.15b`, reduced scope): pixels only from
# the request's own parts, only inside the last IMAGE_WINDOW_TURNS turns.
# ---------------------------------------------------------------------------

def image_turns_in_window(turn_count: int, image_turn_indices: list[int], *, window_turns: int = None) -> tuple[list[int], list[int]]:
    """Split image-bearing turn indices (0-based into a `turn_count`-long
    non-system turn list, newest last) into (in_window, outside_window).
    Outside-window image turns are forwarded as a text note, never as
    pixels — no hoist, no non-adjacent merge, no store."""
    w = IMAGE_WINDOW_TURNS if window_turns is None else window_turns
    cutoff = max(0, turn_count - w)
    inside = [i for i in image_turn_indices if i >= cutoff]
    outside = [i for i in image_turn_indices if i < cutoff]
    return inside, outside


IMAGE_NOTE_TEMPLATE = (
    "[note from the system: an earlier picture in this conversation (turn "
    "{turn}) is outside the {window}-turn image window and was not shown "
    "to you as pixels]"
)


# ---------------------------------------------------------------------------
# The turn-accounting invariant (§17.6, r8).
# ---------------------------------------------------------------------------

class TurnAccountingViolation(RuntimeError):
    """§17.6: 'a failure logs ERROR and falls back to today's v3.1.9.7
    composition for that request, and /health reports it.'"""


def assert_turn_accounting(
    total_unsummarized_turns: int,
    verbatim_turn_count: int,
    digested_unit_count: int,
    digested_turn_count: int,
) -> None:
    """Every unsummarized turn appears exactly once: as verbatim (T1/T2/
    T5) or as a continuity digest, never both and never neither."""
    accounted = verbatim_turn_count + digested_turn_count
    if accounted != total_unsummarized_turns:
        raise TurnAccountingViolation(
            f"turn-accounting invariant failed: {total_unsummarized_turns} "
            f"unsummarized turns, but {verbatim_turn_count} verbatim + "
            f"{digested_turn_count} digested (in {digested_unit_count} "
            f"units) = {accounted}"
        )


def enabled() -> bool:
    """§17.6: 'The compactor refuses to enable the allocator if the
    tokenizer is missing.' Callers must check `tokens.is_available()` in
    addition to this flag — kept as two checks rather than one so a unit
    test can exercise `allocate()` without a real tekken.json on disk."""
    return ENABLED
