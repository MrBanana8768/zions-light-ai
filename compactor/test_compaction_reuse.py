"""Compaction stops re-summarizing turns the hierarchy already covers.

v3.1.9.1. MEASURED, not proposed: on 2026-09-11 a conversation at 61 messages
carried 104,917 tokens and paid 117 SECONDS of compaction before generation —
four concurrent LLM calls over 56 turns — while `maybe_rollup` had already
folded turns 1-40 into persisted L1 chunks and the request was injecting them
separately. The same history, summarized twice, one of the two thrown away and
recomputed every single turn.

main.py said why it happened, at the comment explaining why summarizing a
prefix does not converge:

    "compact_if_needed is a pure function of the client's message array,
     nothing records where summarization stopped, so the SAME oldest batches
     are re-summarized every turn forever."

Something does record it. This file pins that compaction now reads it.

WHAT IT HAS TO PIN, and the second is as important as the first:

  * fewer LLM calls when the hierarchy covers part of the span;
  * NO turn removed without a stand-in IN THE SAME ARRAY. The separately
    injected summary block is capped at 60% of the injection budget and can be
    shed downstream; if compaction leaned on it, a later shedding stage could
    take the stand-in and leave a hole. That is 2026-08-24.

    python test_compaction_reuse.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compact-reuse-")
# Small, so a modest fixture crosses the compaction threshold.
os.environ["COMPACTOR_TARGET_TOKENS"] = "500"

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def history(n_exchanges: int) -> list[dict]:
    """n exchanges of fat turns, so compaction definitely triggers."""
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"question {i} " + ("word " * 60)})
        out.append({"role": "assistant", "content": f"answer {i} " + ("word " * 60)})
    return out


CALLS: list[list[str]] = []


async def _spy_summarize(client, to_summarize):
    """Stand in for summarize(). Records WHAT it was asked to compress."""
    CALLS.append([str(m.get("content", ""))[:24] for m in to_summarize])
    return "FRESHLY-SUMMARIZED", []


_real = main.summarize
main.summarize = _spy_summarize

CONV = "reuse_conv"
MSGS = history(24)          # 48 non-system turns, well over the target

print("[0] control — with no stored hierarchy, everything is summarized")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS), CONV))
check(len(CALLS) == 1, f"summarize was called (got {len(CALLS)})")
baseline = len(CALLS[0]) if CALLS else 0
check(baseline > 0, f"and it was handed {baseline} turns to compress")
check(any("FRESHLY-SUMMARIZED" in str(m.get("content", "")) for m in out),
      "and the fresh summary reached the returned array")

print("[1] with a hierarchy covering the oldest span, it summarizes LESS")
state = summarizer.load_state(CONV)
state["l1"] = [
    {"text": "STORED-CHUNK-ONE", "first_turn": 1, "last_turn": 20},
    {"text": "STORED-CHUNK-TWO", "first_turn": 21, "last_turn": 40},
]
state["last_summarized_turn"] = 40
# THE ANCHOR IS NOW LOAD-BEARING, and its absence here made this fixture
# describe a state production never writes. maybe_rollup sets tail_fp on every
# rollup (`state["tail_fp"] = fps[-_ANCHOR_TURNS:]`), so a hierarchy that
# exists AT ALL has one. compact_if_needed refuses to substitute without it,
# because the length test it used to rely on proves a length relation while
# the substitution needs a content one — an adversarial pass replaced 20 of 30
# live turns with an abandoned branch's summary through exactly that gap.
#
# Built the way maybe_rollup builds it: the fingerprints of the last
# _ANCHOR_TURNS turns of the array as it stood when the rollup ran, which for
# this state is turns 1..40.
_ns = [m for m in MSGS if m.get("role") != "system"]
state["tail_fp"] = summarizer._turn_fingerprints(_ns[:40])[-summarizer._ANCHOR_TURNS:]
summarizer.save_state(CONV, state)

CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS), CONV))
covered_call = len(CALLS[0]) if CALLS else 0
check(covered_call < baseline,
      f"fewer turns went to the LLM ({covered_call} against {baseline}) — this "
      f"is the 117 seconds")
check(any("STORED-CHUNK" in str(m.get("content", "")) for m in out),
      "and the stored text is IN THE RETURNED ARRAY — the stand-in travels "
      "with the removal, so a later shedding stage cannot separate them")

print("[2] the oldest turns are the ones that came off the shelf")
if CALLS and CALLS[0]:
    first_sent = CALLS[0][0]
    check("question 0" not in first_sent and "answer 0" not in first_sent,
          f"the LLM no longer starts from turn 0 (got {first_sent!r})")

print("[3] it never silently drops a turn")
# Every non-system turn must be accounted for: still present verbatim, or
# represented by text in the array. Nothing may simply vanish.
returned = " ".join(str(m.get("content", "")) for m in out)
check("STORED-CHUNK-ONE" in returned and "STORED-CHUNK-TWO" in returned,
      "both stored chunks are present, so the span they cover is represented")

print("[4] no conv_id, or an unreadable store, falls back to today")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS), None))
check(len(CALLS) == 1 and len(CALLS[0]) == baseline,
      f"without a conv_id it summarizes exactly as before ({len(CALLS[0]) if CALLS else 0} "
      f"against {baseline}) — the feature can only remove LLM calls, never add")

print("[5] a CAPPED client is declined, not guessed at")
# The safety property, and the one a mutation found missing. Turn numbers
# stop being array indices the moment a client sends a bounded window: the
# array is a SUFFIX, so its leading messages are NOT turns 1..n. Claiming
# the hierarchy covers them would replace live turns with a summary of
# different ones - silent loss, not slowness. Under-claiming costs a little
# speed, so that is the direction this must fail in.
SHORT = history(8)          # 16 non-system turns against a position of 40
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(SHORT), CONV))
check(len(CALLS) == 1,
      f"summarize still ran (got {len(CALLS)} call(s))")
if CALLS:
    _sent = CALLS[0]
    check(any("question 0" in t for t in _sent),
          "and it was handed the window's OWN oldest turn, not a span the "
          "hierarchy claimed — the suffix guard declined rather than "
          "mapping turn numbers onto indices that do not mean the same thing")
check(not any("STORED-CHUNK" in str(m.get("content", "")) for m in out),
      "and no stored text was substituted into a window it cannot speak for")


# ---------------------------------------------------------------------------
# THE THREE GUARDS AN ADVERSARIAL PASS PUT HERE (v3.1.9.3). Each one closes a
# DEMONSTRATED break, and each is a different way the same habit went wrong:
# deleting turns on the strength of a number never checked against the array
# in hand, the block actually rendered, or the content the chunk was made
# from. Every one has a CONTROL beside it, because a guard that declines
# everything passes a "nothing was substituted" check perfectly.
# ---------------------------------------------------------------------------


def _seed(conv: str, msgs: list[dict], chunks: list[dict], *, anchor_through: int):
    """A hierarchy the way maybe_rollup would have left it."""
    st = summarizer.load_state(conv)
    st["l1"] = chunks
    st["last_summarized_turn"] = chunks[-1]["last_turn"]
    ns = [m for m in msgs if m.get("role") != "system"]
    st["tail_fp"] = summarizer._turn_fingerprints(
        ns[:anchor_through]
    )[-summarizer._ANCHOR_TURNS:]
    summarizer.save_state(conv, st)
    return st


print("[6] a SQUEEZED summary block does not stand in for removed turns")
# format_summary_block drops the OLDEST scenes under budget pressure — the
# same end of the conversation compaction removes. Demonstrated break: 80
# turns deleted from the array and absent from the block, on a state at its
# documented capacity, with the log still calling them covered.
CONV_SQ = "reuse_squeezed"
MSGS_SQ = history(24)
# Sized so the two scenes TOGETHER exceed SUMMARY_BLOCK_MAX_TOKENS (12,000)
# while each one alone fits — that is the shape that drops the oldest scene
# rather than returning nothing. The first draft used 4,000 chars and the
# block fit comfortably, so the test failed by never squeezing at all.
_fat = "z" * 30000
_seed(CONV_SQ, MSGS_SQ,
      [{"text": f"SQUEEZED-{i} {_fat}", "first_turn": i * 20 + 1,
        "last_turn": (i + 1) * 20} for i in range(2)],
      anchor_through=40)
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_SQ), CONV_SQ))
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "the oldest turn was handed to summarize() rather than deleted — a "
      "block that lost its oldest scenes cannot stand in for the oldest turns")
check(not any("SQUEEZED-" in str(m.get("content", "")) for m in out),
      "and no partial block reached the array")

print("[7] an array that does NOT align with the stored anchor is refused")
# OpenWebUI keeps branches in ONE chat, so conv_id never changes. The length
# test is satisfied by a DIFFERENT branch of the same conversation, and the
# demonstrated break replaced 20 of 30 live turns with the abandoned branch's
# summary. tail_fp is the content evidence; _align_candidates reads it.
CONV_BR = "reuse_branch"
MSGS_BR = history(24)
_seed(CONV_BR, MSGS_BR,
      [{"text": "BRANCH-A-CHUNK", "first_turn": 1, "last_turn": 40}],
      anchor_through=40)
_st_br = summarizer.load_state(CONV_BR)
_st_br["tail_fp"] = ["deadbeef", "cafebabe", "f00dface", "badc0ffe"]
summarizer.save_state(CONV_BR, _st_br)
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_BR), CONV_BR))
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "an anchor that appears nowhere in this array declines the substitution")
check(not any("BRANCH-A-CHUNK" in str(m.get("content", "")) for m in out),
      "and the other branch's summary did not reach the array")

# THE CONTROL for [6] and [7] together. Same shape, nothing wrong with it, and
# it MUST still reuse — otherwise both checks above pass by the feature being
# dead, which is exactly how this suite was green while reuse did nothing.
print("[8] control — a sound hierarchy with a matching anchor STILL reuses")
CONV_OK = "reuse_control"
MSGS_OK = history(24)
_seed(CONV_OK, MSGS_OK,
      [{"text": "SOUND-CHUNK", "first_turn": 1, "last_turn": 40}],
      anchor_through=40)
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_OK), CONV_OK))
check(len(CALLS) == 1 and not any("question 0" in t for t in CALLS[0]),
      "the covered oldest turns were NOT re-summarized — reuse still applies")
check(any("SOUND-CHUNK" in str(m.get("content", "")) for m in out),
      "and the stored text travelled into the array with the removal")

print("[9] an IMAGE turn inside the covered span does not shift the mapping")
# _covered counts every non-system turn; text_only has image turns REMOVED, so
# min(_covered, len(text_only)) indexes one unit into the other and deletes one
# live turn per image in the span — demonstrated at 1 and 5 turns of overrun,
# reachable at the shipped MAX_RETAINED_IMAGES=1.
#
# Turn 2 is an image, chunks cover turns 1-40, so 39 of the first 40 turns are
# text. Correct: text_only[39:] starts at turn 41. Wrong: text_only[40:] starts
# at turn 42 and turn 41 is in NEITHER the array nor the summary.
CONV_IMG = "reuse_image"
MSGS_IMG = history(24)
MSGS_IMG[2] = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "here is the sketch"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ],
}
_seed(CONV_IMG, MSGS_IMG,
      [{"text": "IMG-SPAN-CHUNK", "first_turn": 1, "last_turn": 40}],
      anchor_through=40)
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_IMG), CONV_IMG))
_sent_img = CALLS[0] if CALLS else []
check(any("question 20" in t for t in _sent_img),
      "turn 41 — the first turn past the claimed coverage — still reached "
      "summarize(); the image did not push the boundary one turn deep into "
      "live history")
# CONTROL: the covered span really was reused, so [9] is not passing because
# the substitution declined outright.
check(not any("question 0" in t for t in _sent_img),
      "and the covered oldest turns were still NOT re-summarized")

print("[10] _covered_prefix: coverage that does not reach turn 1 is not coverage")
# B3/B4. _highest_chunk_turn answers where coverage ENDS; the substitution
# deletes turns from the START. Two shipped rollup paths leave a hole on
# purpose (pos_last < 1 advances the watermark with NO chunk; `partial`
# records a deliberately narrower first_turn) and load_state parks an
# unparseable chunk, which makes one with no outage at all. The adversarial
# pass deleted 20 turns across such a hole and logged them as covered.
_cp = summarizer._covered_prefix
# CONTROL FIRST: an unbroken chain still returns its full reach, so every
# refusal below is measured against a function that can say yes.
check(_cp({"l1": [{"first_turn": 1, "last_turn": 20},
                  {"first_turn": 21, "last_turn": 40}]}) == 40,
      "an unbroken chain from turn 1 covers its whole span")
check(_cp({"l1": [{"first_turn": 21, "last_turn": 40}]}) == 0,
      "a chain that starts at turn 21 covers NOTHING — this is the "
      "`pos_last < 1` path, which advances the watermark and appends no chunk")
check(_cp({"l1": [{"first_turn": 1, "last_turn": 10},
                  {"first_turn": 31, "last_turn": 44}]}) == 10,
      "a hole at 11-30 stops the count at 10, not 44 — this is the parked "
      "chunk, and 44 is what _highest_chunk_turn answered")
check(_cp({"l1": [{"first_turn": 11, "last_turn": 20}],
           "l2": [{"first_turn": 1, "last_turn": 10}]}) == 20,
      "the chain may span tiers — an L2 rollup CONSUMES its L1 inputs, so a "
      "span lives in whichever tier last touched it")
check(_cp({"l1": [{"first_turn": 21, "last_turn": 40},
                  {"first_turn": 1, "last_turn": 20}]}) == 40,
      "order does not matter; the walk sorts")
check(_cp({"l1": [{"first_turn": 1, "last_turn": 20},
                  {"first_turn": 15, "last_turn": 30}]}) == 30,
      "overlapping spans are contiguous, not a hole")
check(_cp({"l1": [{"first_turn": 1, "last_turn": 40},
                  {"first_turn": 5, "last_turn": 10}]}) == 40,
      "a nested span cannot pull the reach back down")
check(_cp({"l1": [{"first_turn": 1, "last_turn": 20}, "not-a-dict",
                  {"first_turn": 21, "last_turn": 40}]}) == 40,
      "a non-dict entry is skipped, not fatal")
check(_cp({"l1": [{"first_turn": "1", "last_turn": 20}]}) == 0,
      "a string turn number is not an int and claims nothing")
check(_cp({"l1": [{"first_turn": 0, "last_turn": 20}]}) == 0,
      "turn 0 does not exist, so a span claiming it claims nothing")
check(_cp({"l1": [{"first_turn": 30, "last_turn": 5}]}) == 0,
      "a span whose end precedes its start claims nothing")
check(_cp({}) == 0 and _cp({"l1": None, "l2": None}) == 0,
      "an empty or null state covers nothing")
# l3 is excluded DELIBERATELY: it inherits first_turn from the previous l3
# rather than measuring it. Pin that, so a later change to include it is a
# decision and not a drift.
check(_cp({"l3": {"first_turn": 1, "last_turn": 999}}) == 0,
      "l3 alone claims nothing — its span is inherited, not measured")

print("[11] _aligns_fully: one repeated short turn is not the same conversation")
# B2. The gate read `bool(_align_candidates(...))`, and _align_candidates
# tries every prefix down to length ONE. tail_fp's first element is a USER
# turn, and in a companion chat "ok" is typed more than once, so a single
# collision put 20 of 30 exchanges from one branch under the other's summary
# with candidates == [0], which is truthy.
_af = summarizer._aligns_fully
_ANCH = ["aa", "bb", "cc", "dd"]
# CONTROL FIRST.
check(_af(_ANCH, ["xx", "aa", "bb", "cc", "dd"]) is True,
      "the whole anchor present contiguously aligns")
check(_af(_ANCH, ["aa", "bb", "cc", "dd"]) is True,
      "...including when it is the entire window")
check(_af(_ANCH, ["aa", "bb", "cc", "dd", "yy", "zz"]) is True,
      "...and when the window has moved on past it")
check(_af(_ANCH, ["zz", "aa", "yy"]) is False,
      "a ONE-element match is refused — this is the 'ok' collision")
check(_af(_ANCH, ["aa", "bb", "zz"]) is False,
      "a two-element prefix is refused too")
check(_af(_ANCH, ["aa", "bb", "cc"]) is False,
      "three of four is still not the anchor")
check(_af(_ANCH, ["aa", "zz", "bb", "cc", "dd"]) is False,
      "the run must be CONTIGUOUS — an inserted turn breaks it")
check(_af([], ["aa", "bb"]) is False,
      "an empty anchor is no evidence")
check(_af(_ANCH, []) is False and _af(_ANCH, ["aa", "bb"]) is False,
      "a window shorter than the anchor cannot contain it")
# AND the primitive is UNCHANGED, so this is a new predicate rather than a
# weakened shared function. _observed_position's recency rule depends on
# seeing the short candidates.
check(summarizer._align_candidates(_ANCH, ["zz", "aa", "yy"]) != [],
      "_align_candidates still accepts the 1-element match it is built to "
      "find — the fix is a separate predicate, not a broken primitive")

print("[12] a capped window that DOES align is still refused on length")
# The gate ced4520 proved with a mutation, re-proved. That mutation went RED
# then; f78a389 later added `_aligned` as a conjunct IN FRONT of it, and [5]'s
# fixture declines on ALIGNMENT — so the length gate has been untested since,
# and dropping it left this suite GREEN. Mutation evidence expires when a
# conjunct lands in front of it.
#
# The fixture matters: a real capped window ALIGNS BY CONSTRUCTION, because a
# cap sends the last max_turns turns and the anchor is the previous request's
# tail. So the anchor here is built from the WINDOW's own tail, not from turns
# that appear nowhere in it, and the length comparison is the only thing left
# standing between this array and the substitution.
CONV_CAP2 = "reuse_capped_aligning"
MSGS_CAP2 = history(24)
_ns_cap2 = [m for m in MSGS_CAP2 if m.get("role") != "system"]
_seed(CONV_CAP2, MSGS_CAP2,
      [{"text": "CAPPED-CHUNK", "first_turn": 1, "last_turn": 40}],
      anchor_through=40)
_st_cap2 = summarizer.load_state(CONV_CAP2)
# turns_seen past the window's length is what a capped client looks like: the
# conversation is known to have reached 48, the client is sending 16.
_st_cap2["turns_seen"] = 48
_WINDOW = _ns_cap2[-16:]
_st_cap2["tail_fp"] = summarizer._turn_fingerprints(
    _WINDOW)[-summarizer._ANCHOR_TURNS:]
summarizer.save_state(CONV_CAP2, _st_cap2)
# FIXTURE CHECK: the two gates must be in the states this case claims, or it
# proves nothing about which one fired.
check(summarizer._aligns_fully(
          _st_cap2["tail_fp"],
          summarizer._turn_fingerprints(
              _WINDOW[-summarizer._FINGERPRINT_TAIL_TURNS:])) is True,
      "fixture: this window ALIGNS — so the refusal below cannot come from "
      "the alignment gate")
check(len(_WINDOW) < summarizer._recorded_position(_st_cap2),
      "fixture: and it is SHORTER than the recorded position, so the length "
      "gate is the one under test")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(_WINDOW), CONV_CAP2))
check(not any("CAPPED-CHUNK" in str(m.get("content", "")) for m in out),
      "a window that aligns but is shorter than the recorded position is "
      "REFUSED — turn numbers are not array indices once a client caps, and "
      "the stored summary covers turns this array does not contain")

print("[13] a HOLE in the stored coverage stops the substitution at the hole")
# [10] proves _covered_prefix. It does NOT prove the gate calls it: reverting
# the one call site to _highest_chunk_turn left this suite GREEN, which is the
# same defect as running the right input and asserting somewhere else. So
# assert it END TO END, through compact_if_needed, on a state with a hole.
#
# Chunks cover 1-10 and 31-44. _highest_chunk_turn says 44; the real coverage
# is 10, and turns 11-30 are represented by nothing at all.
CONV_HOLE = "reuse_hole"
MSGS_HOLE = history(24)
_seed(CONV_HOLE, MSGS_HOLE,
      [{"text": "HOLE-CHUNK-A", "first_turn": 1, "last_turn": 10},
       {"text": "HOLE-CHUNK-B", "first_turn": 31, "last_turn": 44}],
      anchor_through=44)
_st_hole = summarizer.load_state(CONV_HOLE)
# FIXTURE CHECK: the two readings must actually disagree here, or [13] proves
# nothing about which one the gate used.
check(summarizer._highest_chunk_turn(_st_hole) == 44
      and summarizer._covered_prefix(_st_hole) == 10,
      "fixture: the highest label says 44 and the contiguous prefix says 10")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_HOLE), CONV_HOLE))
check(len(CALLS) == 1 and any("question 5" in t for t in CALLS[0]),
      "turn 11 — the first turn INSIDE the hole — reached summarize() rather "
      "than being deleted under a summary that does not mention it")
check(len(CALLS) == 1 and any("question 15" in t for t in CALLS[0]),
      "and so did turn 31, which HOLE-CHUNK-B claims: a chunk unreachable "
      "from turn 1 cannot stand in for its own span either, because the "
      "turns before it would have to go with it")
# CONTROL: the part that IS contiguously covered is still reused, so [13] is
# not passing because the substitution declined outright.
check(not any("question 0" in t for t in CALLS[0]),
      "CONTROL: turns 1-10 were still reused — the gate narrowed, it did not "
      "give up")

print("[14] an anchor matching ONE turn of the window is refused end to end")
# [11] proves _aligns_fully. Reverting the call site to the old
# `bool(_align_candidates(...))` left this suite GREEN, so prove it through
# compact_if_needed too. This is the shipped 'ok' collision: the anchor's
# first element is a real turn of this array and the other three are
# impossible, which is what a repeated short user turn looks like.
CONV_OK = "reuse_ok_collision"
MSGS_OK = history(24)
_seed(CONV_OK, MSGS_OK,
      [{"text": "COLLIDE-CHUNK", "first_turn": 1, "last_turn": 40}],
      anchor_through=40)
_st_ok = summarizer.load_state(CONV_OK)
_ns_ok = [m for m in MSGS_OK if m.get("role") != "system"]
_fps_ok = summarizer._turn_fingerprints(_ns_ok)
# Mid-window deliberately, NOT at the tail: a collision on the last turn makes
# _align_candidates answer 0, which is falsy, so a mutant reading the old
# non-emptiness rule would decline for the wrong reason and [14] would pass
# without testing anything.
_st_ok["tail_fp"] = [_fps_ok[20], "deadbeef", "cafebabe", "f00dface"]
summarizer.save_state(CONV_OK, _st_ok)
_window_fps = summarizer._turn_fingerprints(
    _ns_ok[-summarizer._FINGERPRINT_TAIL_TURNS:])
# FIXTURE CHECK: the OLD rule must accept this and the new one refuse it.
# Without this the case passes for any anchor at all.
check(summarizer._align_candidates(_st_ok["tail_fp"], _window_fps) != [],
      "fixture: the old non-emptiness rule ACCEPTS this anchor")
check(0 not in summarizer._align_candidates(_st_ok["tail_fp"], _window_fps),
      "fixture: and not via a zero candidate, so a falsy answer cannot be "
      "what declines it")
check(summarizer._aligns_fully(_st_ok["tail_fp"], _window_fps) is False,
      "fixture: and the full-anchor rule refuses it")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_OK), CONV_OK))
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "one matching turn is not evidence this is the same branch: the oldest "
      "turn went to summarize() rather than being replaced")
check(not any("COLLIDE-CHUNK" in str(m.get("content", "")) for m in out),
      "and the stored summary did not reach the array")

main.summarize = _real

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll compaction-reuse checks passed.")
