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

main.summarize = _real

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll compaction-reuse checks passed.")
