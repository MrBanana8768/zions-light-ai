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

AND, since the second reuse fix, a third: reuse must SURVIVE ordinary traffic.
Two cuts of the content gate were each green here and each switched reuse off
in the 112-turn soak, because every fixture built the record by a route
production does not take. [18]-[22] drive the real writer through the real
transformations — redaction, a stopped reply, an L3 refresh, image retention,
the backfill — and assert reuse still fires afterwards.

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

import backfill  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def history(n_exchanges: int, tag: str = "") -> list[dict]:
    """n exchanges of fat turns, so compaction definitely triggers."""
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"{tag}question {i} " + ("word " * 60)})
        out.append({"role": "assistant", "content": f"{tag}answer {i} " + ("word " * 60)})
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


def _seed(conv: str, msgs: list[dict], chunks: list[dict], *, record: bool = True):
    """A hierarchy the way maybe_rollup would have left it.

    The covered-turn record is written by the SAME function maybe_rollup
    calls (_catch_up_covered_fp), over the array handed in as the raw request,
    rather than assembled some other way here. Two cuts of this gate were
    green in this file and dead in the soak because the fixture recorded by a
    route production does not take. If the writer refuses a span — a hole, a
    capped window — the fixture inherits that refusal, which is the point.

    `record=False` seeds a file written BEFORE v3.1.9, which every existing
    conversation on disk is until its next rollup.
    """
    st = summarizer.load_state(conv)
    st["l1"] = chunks
    st["last_summarized_turn"] = max(c["last_turn"] for c in chunks)
    if record:
        summarizer._catch_up_covered_fp(st, msgs, 0)
    summarizer.save_state(conv, st)
    return st


def _plan(conv: str, msgs: list[dict]):
    _, ts, _ = main.split_messages(list(msgs))
    return summarizer._coverage_plan(summarizer.load_state(conv), ts)


print("[1] with a hierarchy covering the oldest span, it summarizes LESS")
_seed(CONV, MSGS, [
    {"text": "STORED-CHUNK-ONE", "first_turn": 1, "last_turn": 20},
    {"text": "STORED-CHUNK-TWO", "first_turn": 21, "last_turn": 40},
])
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
# Turn numbers stop being array indices the moment a client sends a bounded
# window: the array is a SUFFIX, so its leading messages are NOT turns 1..n.
SHORT = history(8)          # 16 non-system turns against a position of 40
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(SHORT), CONV))
check(len(CALLS) == 1,
      f"summarize still ran (got {len(CALLS)} call(s))")
if CALLS:
    check(any("question 0" in t for t in CALLS[0]),
          "and it was handed the window's OWN oldest turn, not a span the "
          "hierarchy claimed")
check(not any("STORED-CHUNK" in str(m.get("content", "")) for m in out),
      "and no stored text was substituted into a window it cannot speak for")

print("[6] a SQUEEZED summary block does not stand in for removed turns")
# format_summary_block drops the OLDEST scenes under budget pressure — the
# same end of the conversation compaction removes. Sized so the two scenes
# TOGETHER exceed SUMMARY_BLOCK_MAX_TOKENS while each alone fits.
CONV_SQ = "reuse_squeezed"
MSGS_SQ = history(24)
_fat = "z" * 30000
_seed(CONV_SQ, MSGS_SQ,
      [{"text": f"SQUEEZED-{i} {_fat}", "first_turn": i * 20 + 1,
        "last_turn": (i + 1) * 20} for i in range(2)])
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_SQ), CONV_SQ))
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "the oldest turn was handed to summarize() rather than deleted — a "
      "block that lost its oldest scenes cannot stand in for the oldest turns")
check(not any("SQUEEZED-" in str(m.get("content", "")) for m in out),
      "and no partial block reached the array")

print("[7] ANOTHER BRANCH under the same conversation id keeps its turns")
# OpenWebUI keeps branches in ONE chat, so conv_id never changes. The shipped
# break replaced 20 of 30 live branch-B turns with branch A's summary. Branch
# A is recorded; branch B shares turns 1-20 and differs from 21 on.
CONV_BR = "reuse_branch"
MSGS_BR_A = history(24)
_seed(CONV_BR, MSGS_BR_A,
      [{"text": "BRANCH-A-CHUNK-1", "first_turn": 1, "last_turn": 20},
       {"text": "BRANCH-A-CHUNK-2", "first_turn": 21, "last_turn": 40}])
_ns_b = [m for m in history(24, tag="B-") if m.get("role") != "system"]
MSGS_BR_B = [MSGS_BR_A[0]] + [m for m in MSGS_BR_A[1:21]] + _ns_b[20:]
_cov_br, _chg_br = _plan(CONV_BR, MSGS_BR_B)
check(_cov_br == 40 and _chg_br == set(range(20, 40)),
      f"fixture: turns 21-40 read as changed and 1-20 do not (got {_cov_br}, "
      f"{sorted(_chg_br)[:3]}..{len(_chg_br)})")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_BR_B), CONV_BR))
_sent_br = CALLS[0] if CALLS else []
check(all(any(f"B-{w} {k} " in t for t in _sent_br)
          for k in range(10, 20) for w in ("question", "answer")),
      "every branch-B turn inside A's covered span reached summarize() — none "
      "was deleted under the other branch's summary")
check(not any("question 0 " in t and not t.startswith("B-") for t in _sent_br),
      "CONTROL: the shared turns 1-20 still came off the shelf")

print("[8] control — a sound hierarchy STILL reuses, with nothing refreshed")
CONV_OK = "reuse_control"
MSGS_OK = history(24)
_seed(CONV_OK, MSGS_OK, [{"text": "SOUND-CHUNK", "first_turn": 1, "last_turn": 40}])
check(_plan(CONV_OK, MSGS_OK) == (40, set()),
      "fixture: 40 covered, none changed")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_OK), CONV_OK))
check(len(CALLS) == 1 and not any("question 0" in t for t in CALLS[0]),
      "the covered oldest turns were NOT re-summarized — reuse still applies")
check(len(CALLS) == 1 and len(CALLS[0]) == 4,
      f"and exactly the 4 uncovered older turns went fresh (got "
      f"{len(CALLS[0]) if CALLS else None})")
check(any("SOUND-CHUNK" in str(m.get("content", "")) for m in out),
      "and the stored text travelled into the array with the removal")

print("[9] an IMAGE turn inside the covered span does not shift the mapping")
# _covered counts every non-system turn; text_only has image turns REMOVED.
# Turn 2 is an image, chunks cover 1-40: text_only[39:] starts at turn 41.
CONV_IMG = "reuse_image"
MSGS_IMG = history(24)
MSGS_IMG[2] = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "here is the sketch"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ],
}
_seed(CONV_IMG, MSGS_IMG, [{"text": "IMG-SPAN-CHUNK", "first_turn": 1, "last_turn": 40}])
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_IMG), CONV_IMG))
_sent_img = CALLS[0] if CALLS else []
check(any("question 20" in t for t in _sent_img),
      "turn 41 — the first turn past the claimed coverage — still reached "
      "summarize(); the image did not push the boundary one turn deep")
check(not any("question 0" in t for t in _sent_img),
      "CONTROL: and the covered oldest turns were still NOT re-summarized")

print("[10] _covered_prefix: coverage that does not reach turn 1 is not coverage")
_cp = summarizer._covered_prefix
check(_cp({"l1": [{"first_turn": 1, "last_turn": 20},
                  {"first_turn": 21, "last_turn": 40}]}) == 40,
      "CONTROL: an unbroken chain from turn 1 covers its whole span")
check(_cp({"l1": [{"first_turn": 21, "last_turn": 40}]}) == 0,
      "a chain that starts at turn 21 covers NOTHING — the `pos_last < 1` path")
check(_cp({"l1": [{"first_turn": 1, "last_turn": 10},
                  {"first_turn": 31, "last_turn": 44}]}) == 10,
      "a hole at 11-30 stops the count at 10, not 44")
check(_cp({"l1": [{"first_turn": 11, "last_turn": 20}],
           "l2": [{"first_turn": 1, "last_turn": 10}]}) == 20,
      "the chain may span tiers — an L2 rollup CONSUMES its L1 inputs")
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
# l3 IS included. Excluding it zeroed coverage for every conversation after
# its first L3 refresh, which consumes the L2 chapters from turn 1 onward.
check(_cp({"l3": {"first_turn": 1, "last_turn": 200},
           "l1": [{"first_turn": 201, "last_turn": 220}]}) == 220,
      "an L3 span from turn 1 is coverage, and later L1 chunks extend it")
check(_cp({"l3": "not-a-dict", "l1": [{"first_turn": 1, "last_turn": 20}]}) == 20,
      "a malformed l3 is skipped, not fatal")

print("[11] the record itself: validation, and what counts as the same turn")
_cf = summarizer._covered_fps
check(_cf({"covered_fps": "0123456789abcdef" * 3}) == ["0123456789abcdef"] * 3,
      "CONTROL: a whole number of hex fingerprints reads back in order")
check(_cf({"covered_fps": "0123456789abcdef" + "0123"}) == [],
      "a torn record (not a whole number of fingerprints) is no evidence")
check(_cf({"covered_fps": "0123456789abcdeg"}) == [],
      "a non-hex record is no evidence")
check(_cf({"covered_fps": ["0123456789abcdef"]}) == [] and _cf({}) == [],
      "a record of the wrong type, or none, is no evidence")
_ctf = summarizer._covered_turn_fingerprints
_img_turn = {"role": "user", "content": [
    {"type": "text", "text": "look at this"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
_demoted, _n_dem = main._apply_image_retention(
    [_img_turn, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]}])
check(_n_dem == 1 and isinstance(_demoted[0]["content"], str),
      "fixture: main's REAL retention demoted the older image to its text note")
check(_ctf([_img_turn]) == _ctf([_demoted[0]]),
      "a turn demoted by image retention is the SAME turn — main's note format "
      "and summarizer's pattern agree")
check(_ctf([{"role": "user", "content": "look at this"}]) != _ctf([_img_turn]),
      "but the same text WITHOUT its image is a different turn")
check(_ctf([{"role": "user", "content": "a  b\n"}]) == _ctf([{"role": "user", "content": "a b"}]),
      "whitespace is normalized, so a re-flowed trailing newline is not an edit")
check(_ctf([{"role": "user", "content": "x"}]) != _ctf([{"role": "assistant", "content": "x"}]),
      "role is part of the turn")
_st_win = {"l1": [{"text": "W", "first_turn": 1, "last_turn": 20}]}
check(summarizer._catch_up_covered_fp(_st_win, history(24), 8) is False
      and "covered_fps" not in _st_win,
      "a BOUNDED window (offset 8) records nothing: its turn N is not raw[N-1]")
check(summarizer._catch_up_covered_fp(_st_win, history(24), 0) is True
      and len(summarizer._covered_fps(_st_win)) == 20,
      "CONTROL: the same array at offset 0 records the 20 covered turns")

print("[12] an array SHORTER than the recorded position is refused on length")
# Mutation evidence expires every time a stronger gate lands ahead of a weaker
# one, so the fixture is one where EVERY OTHER GATE PASSES and length alone
# refuses: a TRUNCATED array — the head of the conversation, not a sliding
# window over its tail. Its first turns are genuine, so the record matches.
CONV_TRUNC = "reuse_truncated_head"
_FULL = history(24)                                  # the conversation: 48 turns
_ns_full = [m for m in _FULL if m.get("role") != "system"]
_TRUNC = [_FULL[0]] + _ns_full[:20]                  # the request: turns 1..20
_seed(CONV_TRUNC, _TRUNC, [{"text": "TRUNC-CHUNK", "first_turn": 1, "last_turn": 10}])
_st_tr = summarizer.load_state(CONV_TRUNC)
_st_tr["turns_seen"] = 48                            # known to have reached 48
summarizer.save_state(CONV_TRUNC, _st_tr)
_ns_tr = [m for m in _TRUNC if m.get("role") != "system"]
check(_plan(CONV_TRUNC, _TRUNC) == (10, set()),
      "fixture: the record vouches for 10 turns and none changed")
check(len(_ns_tr) < summarizer._recorded_position(_st_tr),
      "fixture: and the array is shorter than the recorded position — length "
      "is the only gate left that can refuse")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(_TRUNC), CONV_TRUNC))
check(not any("TRUNC-CHUNK" in str(m.get("content", "")) for m in out),
      "an array shorter than the conversation is known to be is REFUSED even "
      "when its head matches")
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "and its oldest turn went to summarize() rather than being replaced")

print("[13] a HOLE in the stored coverage stops the substitution at the hole")
# Chunks cover 1-10 and 31-44. _highest_chunk_turn says 44; the real coverage
# is 10, and turns 11-30 are represented by nothing at all.
CONV_HOLE = "reuse_hole"
MSGS_HOLE = history(24)
_seed(CONV_HOLE, MSGS_HOLE,
      [{"text": "HOLE-CHUNK-A", "first_turn": 1, "last_turn": 10},
       {"text": "HOLE-CHUNK-B", "first_turn": 31, "last_turn": 44}])
_st_hole = summarizer.load_state(CONV_HOLE)
check(summarizer._highest_chunk_turn(_st_hole) == 44
      and summarizer._covered_prefix(_st_hole) == 10,
      "fixture: the highest label says 44 and the contiguous prefix says 10")
check(len(summarizer._covered_fps(_st_hole)) == 10,
      "and the WRITER refused to record past the hole")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_HOLE), CONV_HOLE))
check(len(CALLS) == 1 and any("question 5" in t for t in CALLS[0]),
      "turn 11 — the first turn INSIDE the hole — reached summarize()")
check(len(CALLS) == 1 and any("question 15" in t for t in CALLS[0]),
      "and so did turn 31, which HOLE-CHUNK-B claims but cannot reach from turn 1")
check(len(CALLS) == 1 and not any("question 0" in t for t in CALLS[0]),
      "CONTROL: turns 1-10 were still reused — the gate narrowed, it did not "
      "give up")

print("[13b] a record that claims MORE than the chunks can back is capped")
# [13] passes whether or not the GATE reads _covered_prefix, because the
# writer already stopped at the hole. The gate's own cap is defence against a
# record that is WRONG — written by a future bug, or a build that disagrees
# about spans. So build exactly that: a hole at 11-30 and a record that is a
# perfectly valid fingerprint list of turns 1..44 of this very array.
CONV_LIE = "reuse_record_overclaims"
MSGS_LIE = history(24)
_seed(CONV_LIE, MSGS_LIE,
      [{"text": "LIE-CHUNK-A", "first_turn": 1, "last_turn": 10},
       {"text": "LIE-CHUNK-B", "first_turn": 31, "last_turn": 44}], record=False)
_st_lie = summarizer.load_state(CONV_LIE)
_st_lie["covered_fps"] = "".join(summarizer._covered_turn_fingerprints(
    [m for m in MSGS_LIE if m.get("role") != "system"][:44]))
summarizer.save_state(CONV_LIE, _st_lie)
check(len(summarizer._covered_fps(summarizer.load_state(CONV_LIE))) == 44,
      "fixture: the record is internally VALID for 44 turns, while the chunks "
      "can only back 10")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_LIE), CONV_LIE))
check(len(CALLS) == 1 and any("question 5" in t for t in CALLS[0]),
      "turn 11, inside the hole, still reached summarize() — a record cannot "
      "vouch for turns no chunk summarized, however well it hashes")

print("[14] an array that shares NOTHING with the record reuses nothing")
# Every covered turn changed: the stored text describes none of this array,
# and substituting it beside a full fresh summary would only cost tokens.
CONV_ALL = "reuse_all_changed"
_seed(CONV_ALL, history(24), [{"text": "ALIEN-CHUNK", "first_turn": 1, "last_turn": 40}])
MSGS_ALIEN = history(24, tag="Z-")
check(_plan(CONV_ALL, MSGS_ALIEN) == (0, set()),
      "fixture: every covered turn changed, so the plan covers nothing")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_ALIEN), CONV_ALL))
check(not any("ALIEN-CHUNK" in str(m.get("content", "")) for m in out),
      "no stored text reached the array")
check(len(CALLS) == 1 and len(CALLS[0]) == baseline,
      f"and the whole older span was summarized fresh ({len(CALLS[0]) if CALLS else 0} "
      f"against {baseline})")

print("[15] an EDITED earlier turn keeps its correction, and costs only itself")
# B1, the only unrecoverable break in this path. OpenWebUI's edit-without-
# regenerate rewrites one message in the middle and keeps everything after it.
# The stored summary describes the PRE-EDIT text; replacing the corrected turn
# with it loses the correction for good.
#
# And the second half, which the checkpoint cut got wrong: the edit must cost
# THAT TURN, not every covered turn after it, and not for the rest of the
# conversation.
CONV_EDIT = "reuse_edited_turn"
MSGS_EDIT = history(24)
_seed(CONV_EDIT, MSGS_EDIT,
      [{"text": "PRE-EDIT-CHUNK", "first_turn": 1, "last_turn": 20},
       {"text": "PRE-EDIT-CHUNK-2", "first_turn": 21, "last_turn": 40}])
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_EDIT), CONV_EDIT))
check(any("PRE-EDIT-CHUNK" in str(m.get("content", "")) for m in out),
      "CONTROL: the unedited array reuses the stored summaries")
_EDITED = [dict(m) for m in MSGS_EDIT]
check(_EDITED[5]["role"] == "user" and "question 2" in _EDITED[5]["content"],
      "fixture: index 5 is turn 5, the user's 'question 2'")
_EDITED[5]["content"] = "CORRECTED-FACT my sister is Sarah, not Sara " + ("word " * 60)
check(_plan(CONV_EDIT, _EDITED) == (40, {4}),
      "exactly turn 5 reads as changed, and the other 39 covered turns do not")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(_EDITED), CONV_EDIT))
_sent_ed = CALLS[0] if CALLS else []
check(any("CORRECTED-FACT" in t for t in _sent_ed),
      "the corrected turn went to summarize() — it was not deleted under the "
      "summary of the text the user corrected")
check(len(_sent_ed) == 5 and not any("question 0" in t for t in _sent_ed),
      f"and ONLY it plus the 4 uncovered turns went fresh (got {len(_sent_ed)}): "
      f"every unchanged covered turn still came off the shelf")
check(_sent_ed[:1] and "CORRECTED-FACT" in _sent_ed[0],
      "the refreshed turn is ahead of the newer uncovered ones — chronological")
# NOT FROZEN. The conversation grows, a rollup covers 41-60, and the writer
# records those turns from the edited array; turn 5 still reads as changed
# and every other covered turn still reuses.
_GROWN = _EDITED + [m for m in history(34) if m.get("role") != "system"][48:]
_st_g = summarizer.load_state(CONV_EDIT)
_st_g["l1"].append({"text": "POST-EDIT-CHUNK", "first_turn": 41, "last_turn": 60})
_st_g["last_summarized_turn"] = 60
check(summarizer._catch_up_covered_fp(_st_g, _GROWN, 0) is True
      and len(summarizer._covered_fps(_st_g)) == 60,
      "the record keeps growing after an edit — it is not frozen at the edit")
summarizer.save_state(CONV_EDIT, _st_g)
check(_plan(CONV_EDIT, _GROWN) == (60, {4}),
      "and on the grown conversation the edit still costs exactly one turn")

print("[16] a state written before v3.1.9 has no record, and declines")
CONV_OLD = "reuse_pre_v319_state"
MSGS_OLD = history(24)
_seed(CONV_OLD, MSGS_OLD,
      [{"text": "OLD-FORMAT-CHUNK", "first_turn": 1, "last_turn": 40}], record=False)
check(summarizer.load_state(CONV_OLD).get("covered_fps") == "",
      "fixture: this state carries no record")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_OLD), CONV_OLD))
check(not any("OLD-FORMAT-CHUNK" in str(m.get("content", "")) for m in out),
      "a pre-v3.1.9 hierarchy is not substituted without evidence")
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "and the turns went to summarize(), exactly as before the feature")
_raw = json.loads(summarizer.summary_path(CONV_OLD).read_text(encoding="utf-8"))
_raw["covered_fps"] = "0123456789abcdef" * 39 + "01234567"
summarizer.summary_path(CONV_OLD).write_text(json.dumps(_raw), encoding="utf-8")
check(summarizer.load_state(CONV_OLD).get("covered_fps") == "",
      "a torn record on disk loads as no record at all")


# ---------------------------------------------------------------------------
# THE REAL WRITER, through the transformations production applies. Every case
# above records through _catch_up_covered_fp directly. The two earlier cuts of
# this gate each agreed with a fixture like that and disagreed with the
# production writer, and the soak found it, not this file. From here on the
# record is written by maybe_rollup / _rollup_hierarchy / the backfill.
# ---------------------------------------------------------------------------

_real_pieces = summarizer._summarize_pieces


async def _stub_pieces(conv_id, client, vllm_url, model, prompt, pieces, max_tokens):
    return f"ROLLED-CHUNK over {len(pieces)} piece(s)"


summarizer._summarize_pieces = _stub_pieces


def _request(n_exchanges: int, tag: str = "") -> list[dict]:
    """A request as the client sends it: the history plus a new user turn."""
    h = history(n_exchanges, tag)
    return h + [{"role": "user", "content": f"{tag}question {n_exchanges} " + ("word " * 60)}]


try:
    print("[17] the record maybe_rollup writes is the record the gate reads")
    CONV_RT = "reuse_roundtrip"
    MSGS_RT = history(24)
    _st_rt = asyncio.run(summarizer.maybe_rollup(
        CONV_RT, MSGS_RT, "http://stub", "m", raw_messages=MSGS_RT))
    check(len(_st_rt["l1"]) == 2 and summarizer._covered_prefix(_st_rt) == 40,
          f"fixture: two real L1 rollups covered 1-40 (got {len(_st_rt['l1'])} "
          f"chunk(s), prefix {summarizer._covered_prefix(_st_rt)})")
    check(len(summarizer._covered_fps(_st_rt)) == 40,
          "the rollup recorded exactly the turns its chunks cover")
    CALLS.clear()
    out = asyncio.run(main.compact_if_needed(list(MSGS_RT), CONV_RT))
    check(any("ROLLED-CHUNK" in str(m.get("content", "")) for m in out)
          and CALLS and not any("question 0" in t for t in CALLS[0]),
          "end to end: state written by the real rollup is REUSED by the real gate")
    _st_none = asyncio.run(summarizer.maybe_rollup(
        "reuse_no_raw", history(24), "http://stub", "m"))
    check(len(_st_none["l1"]) == 2 and summarizer._covered_fps(_st_none) == [],
          "a caller with no raw array (the admin rebuild) records nothing")

    print("[18] a REDACTED degenerate reply inside the covered span does not stop reuse")
    # The production shape that switched reuse off: _rollup_hierarchy hands
    # maybe_rollup a history with every degenerate reply replaced by a
    # placeholder, and the next request carries the reply as it was.
    RULE = "━"
    DEGENERATE = "# Status\n\n```\n" + RULE * 569 + "\n"
    check(main.reply_is_degenerate(DEGENERATE), "fixture: the reply is degenerate")
    CONV_RD = "reuse_redacted"
    REQ_RD = _request(23)
    REQ_RD[10] = {"role": "assistant", "content": DEGENERATE}   # turn 10
    asyncio.run(main._rollup_hierarchy(CONV_RD, REQ_RD, "answer 23 " + "word " * 60))
    _st_rd = summarizer.load_state(CONV_RD)
    check(summarizer._covered_prefix(_st_rd) == 40,
          f"fixture: the real tail rolled up 1-40 (got {summarizer._covered_prefix(_st_rd)})")
    NEXT_RD = list(REQ_RD) + [{"role": "assistant", "content": "answer 23 " + "word " * 60},
                              {"role": "user", "content": "question 24 " + "word " * 60}]
    check(_plan(CONV_RD, NEXT_RD) == (40, set()),
          f"the next request's raw turns match the record, degenerate reply "
          f"included (got {_plan(CONV_RD, NEXT_RD)})")
    CALLS.clear()
    out = asyncio.run(main.compact_if_needed(list(NEXT_RD), CONV_RD))
    check(any("ROLLED-CHUNK" in str(m.get("content", "")) for m in out)
          and CALLS and not any("# Status" in t or "question 0" in t for t in CALLS[0]),
          "and reuse FIRES: no covered turn, degenerate or not, was re-summarized")

    print("[19] a STOPPED (trimmed) reply that a chunk covers does not stop reuse")
    # The rollup summarizes the reply as streamed; OpenWebUI re-sends what it
    # saved. The record must follow the request, on the request that carries
    # the reply back.
    CONV_TR = "reuse_trimmed"
    REQ_TR = _request(29)                       # 59 turns; the reply is turn 60
    _full_reply = "answer 29 " + "word " * 60
    asyncio.run(main._rollup_hierarchy(CONV_TR, REQ_TR, "answer 29 word word"))
    _st_t1 = summarizer.load_state(CONV_TR)
    check(summarizer._covered_prefix(_st_t1) == 60
          and len(summarizer._covered_fps(_st_t1)) == 59,
          f"fixture: chunks cover the trimmed reply (60) and the record stops at "
          f"the 59 turns the request carried (got {summarizer._covered_prefix(_st_t1)}, "
          f"{len(summarizer._covered_fps(_st_t1))})")
    REQ_TR2 = list(REQ_TR) + [{"role": "assistant", "content": _full_reply},
                              {"role": "user", "content": "question 30 " + "word " * 60}]
    asyncio.run(main._rollup_hierarchy(CONV_TR, REQ_TR2, "answer 30 " + "word " * 60))
    check(len(summarizer._covered_fps(summarizer.load_state(CONV_TR))) == 60,
          "the next tail recorded turn 60 from the request that carried it back")
    REQ_TR3 = list(REQ_TR2) + [{"role": "assistant", "content": "answer 30 " + "word " * 60},
                               {"role": "user", "content": "question 31 " + "word " * 60}]
    check(_plan(CONV_TR, REQ_TR3) == (60, set()),
          f"and the full reply OpenWebUI re-sends matches it (got {_plan(CONV_TR, REQ_TR3)})")

    print("[20] a conversation past its first L3 refresh still reuses")
    # _covered_prefix excluded l3, and an L3 refresh consumes the L2 chapters
    # from turn 1 onward — so coverage read 0 for every long conversation.
    CONV_L3 = "reuse_after_l3"
    MSGS_L3 = history(24)
    _st_l3 = summarizer.load_state(CONV_L3)
    _st_l3["l3"] = {"text": "L3-OVERVIEW", "first_turn": 1, "last_turn": 30}
    _st_l3["l1"] = [{"text": "L1-AFTER-L3", "first_turn": 31, "last_turn": 40}]
    _st_l3["last_summarized_turn"] = 40
    summarizer._catch_up_covered_fp(_st_l3, MSGS_L3, 0)
    summarizer.save_state(CONV_L3, _st_l3)
    check(_plan(CONV_L3, MSGS_L3) == (40, set()),
          "the L3 span counts: 40 covered, none changed")
    CALLS.clear()
    out = asyncio.run(main.compact_if_needed(list(MSGS_L3), CONV_L3))
    check(CALLS and not any("question 0" in t for t in CALLS[0])
          and any("L3-OVERVIEW" in str(m.get("content", "")) for m in out),
          "and reuse fires on it")

    print("[21] an image DEMOTED by retention inside the covered span does not stop reuse")
    CONV_DM = "reuse_demoted_image"
    REQ_DM = _request(23)
    REQ_DM[3] = {"role": "user", "content": [
        {"type": "text", "text": "question 1 look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    _req_dm, _ = main._apply_image_retention(REQ_DM)
    asyncio.run(main._rollup_hierarchy(CONV_DM, _req_dm, "answer 23 " + "word " * 60))
    check(summarizer._covered_prefix(summarizer.load_state(CONV_DM)) == 40,
          "fixture: rolled up 1-40 while turn 3 still carried its image")
    NEXT_DM = list(REQ_DM) + [
        {"role": "assistant", "content": "answer 23 " + "word " * 60},
        {"role": "user", "content": [
            {"type": "text", "text": "question 24 and another"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]}]
    _next_dm, _n_dm = main._apply_image_retention(NEXT_DM)
    check(_n_dm == 1 and isinstance(_next_dm[3]["content"], str),
          "fixture: the new upload demoted turn 3 to its text note")
    check(_plan(CONV_DM, _next_dm) == (40, set()),
          f"the demoted turn is the same turn (got {_plan(CONV_DM, _next_dm)})")

    print("[22] the backfill records from the UNREDACTED request")
    CONV_BF = "reuse_backfill_raw"
    REQ_BF = _request(23)
    REQ_BF[10] = {"role": "assistant", "content": DEGENERATE}
    _spawned: list = []
    _started = asyncio.run(backfill.start_backfill_if_needed(
        CONV_BF, list(REQ_BF), "http://stub", "m",
        fire_and_forget=lambda coro: _spawned.append(coro) or True,
        redact=main._redact_degenerate_turns,
    ))
    check(_started and len(_spawned) == 1, "fixture: a backfill was scheduled")
    if _spawned:
        _loc = _spawned[0].cr_frame.f_locals

        def _turn10(ms):
            ns = [m for m in (ms or []) if m.get("role") != "system"]
            return ns[9].get("content") if len(ns) > 9 else None

        check(_turn10(_loc.get("raw_messages")) == DEGENERATE,
              "raw_messages carries the reply exactly as the client sent it")
        check(_turn10(_loc.get("messages")) not in (None, DEGENERATE),
              "CONTROL: while the rollup input is still the redacted copy")
        _spawned[0].close()
    backfill._in_progress_local.discard(CONV_BF)
finally:
    summarizer._summarize_pieces = _real_pieces

main.summarize = _real

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll compaction-reuse checks passed.")
