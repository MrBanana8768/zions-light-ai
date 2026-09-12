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

main.summarize = _real

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll compaction-reuse checks passed.")
