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
import json
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

import httpx  # noqa: E402
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
# AND THE COVERED-TURNS DIGEST, for the same reason (v3.1.9, B1). _do_l1_rollup
# extends it immediately after appending each chunk, so a hierarchy written by
# this release carries one; without it the gate reads NO EVIDENCE and declines,
# which is correct for a pre-v3.1.9 file and would make this fixture describe
# one. Built by the rollup's own function, once per chunk, in order.
for _c in state["l1"]:
    summarizer._extend_covered_fp(state, MSGS, _c["first_turn"], _c["last_turn"], 0)
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


def _seed(conv: str, msgs: list[dict], chunks: list[dict], *, anchor_through: int,
          digest: bool = True):
    """A hierarchy the way maybe_rollup would have left it.

    The covered-turns digest is built by calling the SAME function the rollup
    calls, once per chunk in order, rather than by recomputing it some other
    way here. A fixture that assembles state by a second route is how this
    file got its last defect: [6]'s first version seeded no tail_fp at all,
    which is a state production never produces, and the resulting red read as
    the fix being wrong. If _extend_covered_fp refuses a span — a hole, a
    capped window — the fixture inherits that refusal, which is the point.

    `digest=False` seeds a file written BEFORE v3.1.9, which every existing
    conversation on disk is until its next rollup.
    """
    st = summarizer.load_state(conv)
    st["l1"] = chunks
    st["last_summarized_turn"] = chunks[-1]["last_turn"]
    ns = [m for m in msgs if m.get("role") != "system"]
    st["tail_fp"] = summarizer._turn_fingerprints(
        ns[:anchor_through]
    )[-summarizer._ANCHOR_TURNS:]
    if digest:
        for c in sorted(chunks, key=lambda c: c["first_turn"]):
            summarizer._extend_covered_fp(
                st, msgs, c["first_turn"], c["last_turn"], 0
            )
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

print("[12] an array SHORTER than the recorded position is refused on length")
# The gate ced4520 proved with a mutation, re-proved TWICE. That mutation went
# RED then; f78a389 added `_aligned` in front of it and it went GREEN; this
# suite's first fix built a capped window whose anchor aligned and it went RED
# again — and then B1's digest landed in front of BOTH and it went GREEN a
# second time, within the hour, because a sliding-window cap can never match a
# digest of turns 1..N. Mutation evidence expires every time a stronger gate
# lands ahead of a weaker one.
#
# So the fixture must be one where EVERY OTHER GATE PASSES and length alone
# refuses: a TRUNCATED array — the head of the conversation, not a sliding
# window over its tail. A client that replays the beginning and stops (an
# import, a replay, a history cut at a length limit) produces exactly this:
# the first turns are genuine, so the digest matches; the anchor is its own
# tail, so it aligns; and it is shorter than the conversation is KNOWN to be.
CONV_TRUNC = "reuse_truncated_head"
_FULL = history(24)                                  # the conversation: 48 turns
_ns_full = [m for m in _FULL if m.get("role") != "system"]
_TRUNC = [_FULL[0]] + _ns_full[:20]                  # the request: turns 1..20
_seed(CONV_TRUNC, _TRUNC,
      [{"text": "TRUNC-CHUNK", "first_turn": 1, "last_turn": 10}],
      anchor_through=20)
_st_tr = summarizer.load_state(CONV_TRUNC)
_st_tr["turns_seen"] = 48                            # known to have reached 48
summarizer.save_state(CONV_TRUNC, _st_tr)
_ns_tr = [m for m in _TRUNC if m.get("role") != "system"]
_, _ts_tr, _ = main.split_messages(list(_TRUNC))
# FIXTURE CHECKS: every gate but length must PASS, or this proves nothing
# about which one refused.
check(summarizer._covered_prefix(_st_tr) == 10,
      "fixture: the contiguous prefix is 10")
check(_st_tr["covered_fp_turns"] == 10 and summarizer._covered_fp_over(
          _ts_tr, 10) == _st_tr["covered_fp"],
      "fixture: the digest covers 10 turns and MATCHES this array's first 10")
check(summarizer._aligns_fully(
          _st_tr["tail_fp"],
          summarizer._turn_fingerprints(
              _ns_tr[-summarizer._FINGERPRINT_TAIL_TURNS:])) is True,
      "fixture: the anchor aligns fully")
check(len(_ns_tr) < summarizer._recorded_position(_st_tr),
      "fixture: and the array is shorter than the recorded position — length "
      "is the only gate left that can refuse")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(_TRUNC), CONV_TRUNC))
check(not any("TRUNC-CHUNK" in str(m.get("content", "")) for m in out),
      "an array shorter than the conversation is known to be is REFUSED even "
      "when its head matches and its tail aligns")
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "and its oldest turn went to summarize() rather than being replaced")

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

print("[13b] a digest that claims MORE than the chunks can back is refused")
# B1's digest now caps the reuse at a hole by itself — _extend_covered_fp
# refuses a non-contiguous span — so [13] passes whether or not the gate uses
# _covered_prefix, and reverting that call site went GREEN again. The prefix
# check is defence in depth against a digest that is WRONG: extended across a
# hole by a future bug, or written by a build that disagrees about spans. So
# build exactly that: chunks with a hole at 11-30, and a digest that is a
# perfectly valid fold over turns 1..44 of this very array. Only the
# contiguous-prefix check can refuse it, and it has to.
CONV_LIE = "reuse_digest_overclaims"
MSGS_LIE = history(24)
_seed(CONV_LIE, MSGS_LIE,
      [{"text": "LIE-CHUNK-A", "first_turn": 1, "last_turn": 10},
       {"text": "LIE-CHUNK-B", "first_turn": 31, "last_turn": 44}],
      anchor_through=44, digest=False)
_st_lie = summarizer.load_state(CONV_LIE)
_, _ts_lie, _ = main.split_messages(list(MSGS_LIE))
_st_lie["covered_fp"] = summarizer._covered_fp_over(_ts_lie, 44)
_st_lie["covered_fp_turns"] = 44
summarizer.save_state(CONV_LIE, _st_lie)
check(summarizer._covered_prefix(_st_lie) == 10
      and summarizer._covered_fp_over(_ts_lie, 44) == _st_lie["covered_fp"],
      "fixture: the digest is internally VALID for 44 turns, while the chunks "
      "can only back 10")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_LIE), CONV_LIE))
check(len(CALLS) == 1 and any("question 5" in t for t in CALLS[0]),
      "turn 11, inside the hole, still reached summarize() — a digest cannot "
      "vouch for turns no chunk summarized, however well it hashes")

print("[15] an EDITED earlier turn is not replaced by the summary of its old text")
# B1, the only unrecoverable break in this path. OpenWebUI's edit-without-
# regenerate rewrites one message in the middle and keeps everything after it,
# so the tail is byte-identical: _aligned passes, _covered_prefix passes, the
# length gate passes. The stored summary describes the PRE-EDIT text, the
# substitution deletes the corrected turn, and the hierarchy never re-reads
# that span because last_summarized_turn is already past it. The correction is
# gone for the life of the conversation.
CONV_EDIT = "reuse_edited_turn"
MSGS_EDIT = history(24)
_seed(CONV_EDIT, MSGS_EDIT,
      [{"text": "PRE-EDIT-CHUNK", "first_turn": 1, "last_turn": 20},
       {"text": "PRE-EDIT-CHUNK-2", "first_turn": 21, "last_turn": 40}],
      anchor_through=40)
# CONTROL FIRST, on the identical unedited array: it must reuse, or every
# refusal below passes by the feature being off.
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_EDIT), CONV_EDIT))
check(any("PRE-EDIT-CHUNK" in str(m.get("content", "")) for m in out),
      "CONTROL: the unedited array still reuses the stored summaries")
# Turn 5 is the user's 3rd message (turn 2k+1 is question k): MSGS_EDIT[5]
# counting the system message at index 0.
_EDITED = [dict(m) for m in MSGS_EDIT]
check(_EDITED[5]["role"] == "user" and "question 2" in _EDITED[5]["content"],
      "fixture: index 5 is turn 5, the user's 'question 2'")
_EDITED[5]["content"] = "CORRECTED-FACT my sister is Sarah, not Sara " + ("word " * 60)
_ns_ed = [m for m in _EDITED if m.get("role") != "system"]
_st_ed = summarizer.load_state(CONV_EDIT)
check(summarizer._aligns_fully(
          _st_ed["tail_fp"],
          summarizer._turn_fingerprints(
              _ns_ed[-summarizer._FINGERPRINT_TAIL_TURNS:])) is True
      and summarizer._covered_prefix(_st_ed) == 40
      and len(_ns_ed) >= summarizer._recorded_position(_st_ed),
      "fixture: after the edit, the tail still aligns, the prefix is still 40 "
      "and the length still passes — every pre-B1 gate would substitute")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(_EDITED), CONV_EDIT))
_sent_ed = CALLS[0] if CALLS else []
check(any("CORRECTED-FACT" in t for t in _sent_ed)
      or any("CORRECTED-FACT" in str(m.get("content", "")) for m in out),
      "the corrected turn survives — in the summarize() input or verbatim in "
      "the array, anywhere but deleted")
check(not any("PRE-EDIT-CHUNK" in str(m.get("content", "")) for m in out),
      "and the summary of the text the user CORRECTED did not stand in for it")

print("[16] a state written before v3.1.9 has no digest, and declines")
# Every conversation on disk today. Absent must read as NO EVIDENCE — the
# direction that costs one summarization call, not the one that costs turns.
CONV_OLD = "reuse_pre_v319_state"
MSGS_OLD = history(24)
_seed(CONV_OLD, MSGS_OLD,
      [{"text": "OLD-FORMAT-CHUNK", "first_turn": 1, "last_turn": 40}],
      anchor_through=40, digest=False)
check(summarizer.load_state(CONV_OLD).get("covered_fp") == "",
      "fixture: this state carries no digest")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_OLD), CONV_OLD))
check(not any("OLD-FORMAT-CHUNK" in str(m.get("content", "")) for m in out),
      "a pre-v3.1.9 hierarchy is not substituted on the strength of gates that "
      "cannot see an edit")
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "and the turns went to summarize(), exactly as before the feature")
# A half-written pair reads as nothing, not as half of something.
_st_half = summarizer.load_state(CONV_OLD)
_st_half["covered_fp"] = "0123456789abcdef"
summarizer.save_state(CONV_OLD, _st_half)
_raw = json.loads(summarizer.summary_path(CONV_OLD).read_text(encoding="utf-8"))
_raw.pop("covered_fp_turns", None)
summarizer.summary_path(CONV_OLD).write_text(json.dumps(_raw), encoding="utf-8")
check(summarizer.load_state(CONV_OLD).get("covered_fp") == "",
      "a digest on disk WITHOUT its turn count loads as no digest at all")

print("[17] the digest the ROLLUP writes is the digest the GATE computes")
# Every case above seeds through _seed, which calls _extend_covered_fp with
# window_offset 0 directly. That proves the reader agrees with a helper. It
# does not prove it agrees with _do_l1_rollup, which is the only writer
# production has — and a writer and reader that each agree with a fixture but
# not with each other is a gate that declines forever, silently, which is the
# "reuse permanently off with zero log signal" shape a hostile pass already
# found here once. So drive the real rollup, twice, with the LLM stubbed, and
# hand its state straight to the gate.
CONV_RT = "reuse_roundtrip"
MSGS_RT = history(24)
_real_pieces = summarizer._summarize_pieces


async def _stub_pieces(conv_id, client, vllm_url, model, prompt, pieces, max_tokens):
    return f"ROUNDTRIP-CHUNK over {len(pieces)} turn(s)"


summarizer._summarize_pieces = _stub_pieces
try:
    _st_rt = summarizer.load_state(CONV_RT)

    async def _two_rollups():
        async with httpx.AsyncClient() as _c:
            a = await summarizer._do_l1_rollup(CONV_RT, _c, "http://stub", "m", _st_rt, MSGS_RT)
            b = await summarizer._do_l1_rollup(CONV_RT, _c, "http://stub", "m", _st_rt, MSGS_RT)
        return a, b

    _adv = asyncio.run(_two_rollups())
finally:
    summarizer._summarize_pieces = _real_pieces
check(_adv == (True, True) and len(_st_rt["l1"]) == 2,
      f"fixture: two real L1 rollups advanced (got {_adv}, "
      f"{len(_st_rt['l1'])} chunk(s))")
check(_st_rt.get("covered_fp_turns") == 2 * summarizer.L1_CHUNK_SIZE,
      "the rollup extended the digest over exactly the turns it summarized "
      f"(got {_st_rt.get('covered_fp_turns')})")
_, _ts_rt, _ = main.split_messages(list(MSGS_RT))
check(summarizer._covered_fp_over(_ts_rt, _st_rt["covered_fp_turns"])
      == _st_rt["covered_fp"],
      "and the gate's own computation over the same request AGREES with it — "
      "writer and reader, not writer and fixture")
_st_rt["tail_fp"] = summarizer._turn_fingerprints(
    [m for m in MSGS_RT if m.get("role") != "system"][:40]
)[-summarizer._ANCHOR_TURNS:]
summarizer.save_state(CONV_RT, _st_rt)
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_RT), CONV_RT))
check(any("ROUNDTRIP-CHUNK" in str(m.get("content", "")) for m in out),
      "end to end: state written by the real rollup is REUSED by the real gate")

main.summarize = _real

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll compaction-reuse checks passed.")
