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
production does not take. [17]-[22] drive the real writer through the real
transformations — redaction, a stopped reply, an L3 refresh, image retention,
the backfill — and assert reuse still fires afterwards.

AND, since hostile pass #3, a fourth: a turn is replaced only when a chunk
READ that exact content. The record is written by the chunk that covers a
position, in the same call, from the turn it read (summarizer._record_chunk_fps),
and the gate pairs turns with it by content, not by position. Every fixture
below records through that writer. test_p3a_reuse_traffic.py drives the
delete / regenerate / edit / admin-rebuild traffic that broke the third cut,
and test_p3a_reuse_endpoint.py the route-level contracts.

    python test_compaction_reuse.py
"""

import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compact-reuse-")
# Small, so a modest fixture crosses the compaction threshold. 1000 rather
# than 500 since hostile pass #3 F5: the stand-in is budgeted against what
# TARGET leaves beside the recent turns and ONE fresh summary, and at 500 with
# the default 1024-token summary reserve nothing could ever fit.
os.environ["COMPACTOR_TARGET_TOKENS"] = "1000"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "64"
# The stand-in budget prices a preserved image at this estimate; at the
# default 4096 one image leaves a 1000-token TARGET no room at all.
os.environ["COMPACTOR_IMAGE_TOKENS"] = "64"

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


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name: str = "compactor"):
    lg = logging.getLogger(logger_name)
    handler = _Collector()
    prev_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)


def _find(records, needle):
    return next((r for r in records if needle in r.getMessage()), None)


def history(n_exchanges: int, tag: str = "") -> list[dict]:
    """n exchanges of fat turns, so compaction definitely triggers."""
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"{tag}question {i} " + ("word " * 60)})
        out.append({"role": "assistant", "content": f"{tag}answer {i} " + ("word " * 60)})
    return out


def _ns(msgs: list[dict]) -> list[dict]:
    return [m for m in msgs if m.get("role") != "system"]


CALLS: list[list[str]] = []
CALLS_FULL: list[list[str]] = []


async def _spy_summarize(client, to_summarize):
    """Stand in for summarize(). Records WHAT it was asked to compress."""
    CALLS.append([str(m.get("content", ""))[:24] for m in to_summarize])
    CALLS_FULL.append([summarizer._message_text(m) for m in to_summarize])
    return "FRESHLY-SUMMARIZED", []


_real = main.summarize
main.summarize = _spy_summarize


def _lost_turns(conv: str, req: list[dict], out: list[dict]) -> list[str]:
    """THE ORACLE: every non-system turn of `req` must be represented in what
    compaction returned — verbatim in `out`, handed to summarize() on this
    call, or replaced while its OWN content is a record entry the chunk chain
    backs AND the stored summaries travelled into `out`. Returns the 24-char
    heads of every turn that is none of the three."""
    st = summarizer.load_state(conv)
    rec = summarizer._covered_fps(st)
    eff = min(summarizer._covered_prefix(st), len(rec))
    backed = {e for e in rec[:eff] if e != summarizer._FP_UNKNOWN}
    chunks = list(st.get("l1") or []) + list(st.get("l2") or [])
    if isinstance(st.get("l3"), dict):
        chunks.append(st["l3"])
    blob = "\n".join(summarizer._message_text(m) for m in out)
    stand_in = bool(chunks) and all(c["text"] in blob for c in chunks)
    sent = set(CALLS_FULL[-1]) if CALLS_FULL else set()
    lost = []
    for m in _ns(req):
        if m in out or summarizer._message_text(m) in sent:
            continue
        if stand_in and summarizer._covered_turn_fingerprint(m) in backed:
            continue
        lost.append(summarizer._message_text(m)[:24])
    return lost


def _run(msgs, conv, **kw):
    CALLS.clear()
    CALLS_FULL.clear()
    return asyncio.run(main.compact_if_needed(list(msgs), conv, **kw))


CONV = "reuse_conv"
MSGS = history(24)          # 48 non-system turns, well over the target

print("[0] control — with no stored hierarchy, everything is summarized")
out = _run(MSGS, CONV)
check(len(CALLS) == 1, f"summarize was called (got {len(CALLS)})")
baseline = len(CALLS[0]) if CALLS else 0
check(baseline > 0, f"and it was handed {baseline} turns to compress")
check(any("FRESHLY-SUMMARIZED" in str(m.get("content", "")) for m in out),
      "and the fresh summary reached the returned array")


def _seed(conv: str, msgs: list[dict], chunks: list[dict], *, record: bool = True):
    """A hierarchy the way maybe_rollup would have left it.

    The covered-turn record is written by the SAME function _do_l1_rollup
    calls when it appends a chunk (_record_chunk_fps), once per chunk, from
    the turns that chunk covers in `msgs`. Two cuts of this gate were green in
    this file and dead in the soak because the fixture recorded by a route
    production does not take; a third blessed turns because production itself
    recorded from a LATER request than the chunk. If the writer pads a hole
    with UNKNOWN, the fixture inherits that, which is the point.

    `record=False` seeds a file written BEFORE v3.1.9, which every existing
    conversation on disk is until its next rollup.
    """
    st = summarizer.load_state(conv)
    st["l1"] = chunks
    st["last_summarized_turn"] = max(c["last_turn"] for c in chunks)
    if record:
        ns = _ns(msgs)
        for c in sorted(chunks, key=lambda c: c["first_turn"]):
            summarizer._record_chunk_fps(
                st, c["first_turn"], c["last_turn"],
                ns[c["first_turn"] - 1:c["last_turn"]])
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
out = _run(MSGS, CONV)
covered_call = len(CALLS[0]) if CALLS else 0
check(covered_call < baseline,
      f"fewer turns went to the LLM ({covered_call} against {baseline}) — this "
      f"is the 117 seconds")
check(any("STORED-CHUNK" in str(m.get("content", "")) for m in out),
      "and the stored text is IN THE RETURNED ARRAY — the stand-in travels "
      "with the removal, so a later shedding stage cannot separate them")
check(_lost_turns(CONV, MSGS, out) == [], "and the oracle finds no lost turn")

print("[2] the oldest turns are the ones that came off the shelf")
if CALLS and CALLS[0]:
    first_sent = CALLS[0][0]
    check("question 0" not in first_sent and "answer 0" not in first_sent,
          f"the LLM no longer starts from turn 0 (got {first_sent!r})")

print("[3] it never silently drops a turn")
returned = " ".join(str(m.get("content", "")) for m in out)
check("STORED-CHUNK-ONE" in returned and "STORED-CHUNK-TWO" in returned,
      "both stored chunks are present, so the span they cover is represented")

print("[4] no conv_id falls back to today")
out = _run(MSGS, None)
check(len(CALLS) == 1 and len(CALLS[0]) == baseline,
      f"without a conv_id it summarizes exactly as before ({len(CALLS[0]) if CALLS else 0} "
      f"against {baseline}) — the feature can only remove LLM calls, never add")

print("[5] a CAPPED window: the turns the record holds come off the shelf, "
      "and no turn is lost")
# hostile pass #3 (F2/F3/F7). A capped window used to be refused outright by a
# length gate, because its turn N is not conversation turn N and the gate
# compared by position. Nothing compares by position now: a window turn is
# replaced only if its own content is a record entry. The window below is
# turns 17-48 of the conversation; 17-40 are recorded, 41-48 are not.
WINDOW = [MSGS[0]] + _ns(MSGS)[16:48]
out = _run(WINDOW, CONV)
check(_lost_turns(CONV, WINDOW, out) == [],
      f"*** no window turn is lost (lost: {_lost_turns(CONV, WINDOW, out)})")
check(any("STORED-CHUNK" in str(m.get("content", "")) for m in out),
      "the window's recorded turns were replaced by the stored summaries")
check(len(CALLS) == 1 and any("question 20" in t for t in CALLS[0])
      and not any("question 8 " in t for t in CALLS[0]),
      "turn 41 (unrecorded) went to summarize(); turn 17 (recorded) did not")
ALIEN_WINDOW = [MSGS[0]] + _ns(history(24, tag="W-"))[16:48]
out = _run(ALIEN_WINDOW, CONV)
check(not any("STORED-CHUNK" in str(m.get("content", "")) for m in out)
      and len(CALLS) == 1 and any("W-question 8 " in t for t in CALLS[0]),
      "CONTROL: a window of turns the record does not hold replaces nothing")

print("[6] a SQUEEZED summary block does not stand in for removed turns")
# format_summary_block drops the OLDEST scenes under budget pressure — the
# same end of the conversation compaction removes. The budget that binds is
# the stand-in budget (TARGET minus system, images, recent turns and one fresh
# summary, F5); the two scenes are sized so each fits it and both do not.
CONV_SQ = "reuse_squeezed"
MSGS_SQ = history(24)
_sys_sq, _ts_sq, _kr_sq = main.split_messages(list(MSGS_SQ))
_budget_sq = (main.TARGET_TOKENS - main.count_tokens(_sys_sq + _kr_sq)
              - main.SUMMARY_MAX_TOKENS)
_fat = "z " * int(_budget_sq * 1.2)
_seed(CONV_SQ, MSGS_SQ,
      [{"text": f"SQUEEZED-{i} {_fat}", "first_turn": i * 20 + 1,
        "last_turn": (i + 1) * 20} for i in range(2)])
_st_sq = summarizer.load_state(CONV_SQ)
check(summarizer.format_summary_block(_st_sq, _budget_sq, all_or_nothing=True) is None
      and summarizer.format_summary_block(_st_sq, _budget_sq) is not None
      and summarizer.format_summary_block(
          {"l1": _st_sq["l1"][1:]}, _budget_sq, all_or_nothing=True) is not None,
      f"fixture: at the {_budget_sq}-token stand-in budget one scene fits and "
      f"two do not, and a squeeze would return a partial block")
out = _run(MSGS_SQ, CONV_SQ)
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "the oldest turn was handed to summarize() rather than deleted — a "
      "block that lost its oldest scenes cannot stand in for the oldest turns")
check(not any("SQUEEZED-" in str(m.get("content", "")) for m in out),
      "and no partial block reached the array")

print("[6b] F5: the stand-in cannot push the recent turns out of the window")
# hostile pass #3 F5 (pass #2 H5). The stand-in was rendered against the flat
# 12,000-token block cap whatever else the array held; one long reply in the
# recent window then put compaction's own output past the limit and the guard
# shed her last two turns first. It now gets what TARGET leaves.
CONV_FAT = "reuse_fat_recent"
MSGS_FAT = history(24)
_seed(CONV_FAT, MSGS_FAT, [{"text": "FAT-RECENT-CHUNK " + "y " * 150,
                            "first_turn": 1, "last_turn": 40}])
out = _run(MSGS_FAT, CONV_FAT)
check(any("FAT-RECENT-CHUNK" in str(m.get("content", "")) for m in out),
      "CONTROL: with ordinary recent turns this stand-in fits and is reused")
FAT = [dict(m) for m in MSGS_FAT]
FAT[-2] = {"role": "user", "content": "question 23 " + "long " * 700}
with capture() as records:
    out = _run(FAT, CONV_FAT)
check(not any("FAT-RECENT-CHUNK" in str(m.get("content", "")) for m in out),
      "*** F5: beside a recent turn that leaves no room, the stand-in is NOT used")
check(any("long long" in str(m.get("content", "")) for m in out),
      "and the long recent turn is still in the array")
check(_find(records.records, "do not fit whole") is not None,
      "and the decline says why, with the numbers")
check(_lost_turns(CONV_FAT, FAT, out) == [], "and nothing was lost")

print("[7] ANOTHER BRANCH under the same conversation id keeps its turns")
# OpenWebUI keeps branches in ONE chat, so conv_id never changes. The shipped
# break replaced 20 of 30 live branch-B turns with branch A's summary. Branch
# A is recorded; branch B shares turns 1-20 and differs from 21 on.
CONV_BR = "reuse_branch"
MSGS_BR_A = history(24)
_seed(CONV_BR, MSGS_BR_A,
      [{"text": "BRANCH-A-CHUNK-1", "first_turn": 1, "last_turn": 20},
       {"text": "BRANCH-A-CHUNK-2", "first_turn": 21, "last_turn": 40}])
_ns_b = _ns(history(24, tag="B-"))
MSGS_BR_B = [MSGS_BR_A[0]] + [m for m in MSGS_BR_A[1:21]] + _ns_b[20:]
_cov_br, _chg_br = _plan(CONV_BR, MSGS_BR_B)
check((_cov_br, _chg_br) == (20, set()),
      f"fixture: turns 1-20 pair with the record and no branch-B turn does "
      f"(got {_cov_br}, {sorted(_chg_br)[:3]}..{len(_chg_br)})")
out = _run(MSGS_BR_B, CONV_BR)
_sent_br = CALLS[0] if CALLS else []
check(all(any(f"B-{w} {k} " in t for t in _sent_br)
          for k in range(10, 20) for w in ("question", "answer")),
      "every branch-B turn inside A's covered span reached summarize() — none "
      "was deleted under the other branch's summary")
check(not any("question 0 " in t and not t.startswith("B-") for t in _sent_br),
      "CONTROL: the shared turns 1-20 still came off the shelf")
check(_lost_turns(CONV_BR, MSGS_BR_B, out) == [], "and the oracle finds no lost turn")

print("[8] control — a sound hierarchy STILL reuses, with nothing refreshed")
CONV_OK = "reuse_control"
MSGS_OK = history(24)
_seed(CONV_OK, MSGS_OK, [{"text": "SOUND-CHUNK", "first_turn": 1, "last_turn": 40}])
check(_plan(CONV_OK, MSGS_OK) == (40, set()),
      "fixture: 40 covered, none changed")
out = _run(MSGS_OK, CONV_OK)
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
out = _run(MSGS_IMG, CONV_IMG)
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

print("[11] the record itself: validation, the writer, and what counts as the same turn")
_cf = summarizer._covered_fps
_U = summarizer._FP_UNKNOWN
check(_cf({"covered_fps": "0123456789abcdef" * 3}) == ["0123456789abcdef"] * 3,
      "CONTROL: a whole number of hex fingerprints reads back in order")
check(_cf({"covered_fps": "0123456789abcdef" + _U}) == ["0123456789abcdef", _U],
      "CONTROL: an UNKNOWN entry is a valid entry")
check(_cf({"covered_fps": "0123456789abcdef" + "0123"}) == [],
      "a torn record (not a whole number of fingerprints) is no evidence")
check(_cf({"covered_fps": "0123456789abcdeg"}) == [],
      "a non-hex record is no evidence")
check(_cf({"covered_fps": "0123456789abcde-"}) == [],
      "an entry that is part hex and part UNKNOWN marker is no evidence")
check(_cf({"covered_fps": ["0123456789abcdef"]}) == [] and _cf({}) == [],
      "a record of the wrong type, or none, is no evidence")
_ctf = summarizer._covered_turn_fingerprints
_w = {"l1": []}
_h24 = _ns(history(24))
check(summarizer._record_chunk_fps(_w, 21, 40, _h24[20:40]) is True
      and _cf(_w)[:20] == [_U] * 20 and _cf(_w)[20:] == _ctf(_h24[20:40]),
      "the writer pads the positions below a chunk that no chunk read with UNKNOWN")
check(summarizer._record_chunk_fps(_w, 1, 20, _h24[:20]) is True
      and _cf(_w) == _ctf(_h24[:40]),
      "a chunk that reads an UNKNOWN position fills it")
check(summarizer._record_chunk_fps(_w, 21, 40, _ns(history(24, tag="X-"))[20:40]) is False
      and _cf(_w) == _ctf(_h24[:40]),
      "an entry that holds a fingerprint is never rewritten (append-only)")
check(summarizer._record_chunk_fps(_w, 41, 60, _h24[40:44]) is False
      and len(_cf(_w)) == 40,
      "a span the turns handed in do not fill records nothing")
check(_U not in _ctf(_h24), "no turn's fingerprint can equal the UNKNOWN marker")
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
check(_ctf([{**_demoted[0], "content": _demoted[0]["content"] + "  \n"}]) == _ctf([_img_turn]),
      "and still with trailing whitespace after the note (F6's cheap pre-check)")
_long_demoted = {"role": "user", "content": ("ramble " * 200) + _demoted[0]["content"][len("look at this"):]}
check(_ctf([_long_demoted]) == _ctf([{"role": "user", "content": [
    {"type": "text", "text": ("ramble " * 200).rstrip()},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,CCCC"}}]}]),
      "and on a turn far longer than the note's search window")
check(_ctf([{"role": "user", "content": "look at this"}]) != _ctf([_img_turn]),
      "but the same text WITHOUT its image is a different turn")
check(_ctf([{"role": "user", "content": "[1 image shared earlier in this conversation] and then more"}])
      != _ctf([{"role": "user", "content": "and then more"}]),
      "a note that is not at the END is text, not a demotion")
check(_ctf([{"role": "user", "content": "a  b\n"}]) == _ctf([{"role": "user", "content": "a b"}]),
      "whitespace is normalized, so a re-flowed trailing newline is not an edit")
check(_ctf([{"role": "user", "content": "x"}]) != _ctf([{"role": "assistant", "content": "x"}]),
      "role is part of the turn")
_memo_a = {"role": "user", "content": "memo probe " + "q " * 50}
_memo_b = {"role": "user", "content": "memo probf " + "q " * 50}
check(_ctf([_memo_a]) == _ctf([dict(_memo_a)]) and _ctf([_memo_a]) != _ctf([_memo_b])
      and _ctf([_memo_b]) != _ctf([_memo_a]),
      "the fingerprint memo (F6) returns the same answer for equal text and a "
      "different one for a same-length edit, in either order")

print("[12] a TRUNCATED HEAD is reused for the turns it carries, and loses none")
# The third cut refused this on length alone ("the array is shorter than the
# conversation is known to be"). Its turns ARE turns 1..20, and the chunk read
# turns 1..10: replacing those ten with the summary that read them is exactly
# reuse. [5] is the capped (suffix) case; test_p3a_reuse_traffic.py drives the
# edit-and-resend and delete traffic that produces both shapes.
CONV_TRUNC = "reuse_truncated_head"
_FULL = history(24)                                  # the conversation: 48 turns
_TRUNC = [_FULL[0]] + _ns(_FULL)[:20]                # the request: turns 1..20
_seed(CONV_TRUNC, _FULL, [{"text": "TRUNC-CHUNK", "first_turn": 1, "last_turn": 10}])
_st_tr = summarizer.load_state(CONV_TRUNC)
_st_tr["turns_seen"] = 48                            # known to have reached 48
summarizer.save_state(CONV_TRUNC, _st_tr)
check(_plan(CONV_TRUNC, _TRUNC) == (10, set()),
      "fixture: the record vouches for 10 turns and none changed")
out = _run(_TRUNC, CONV_TRUNC)
check(any("TRUNC-CHUNK" in str(m.get("content", "")) for m in out),
      "the truncated head's recorded turns came off the shelf")
check(len(CALLS) == 1 and any("question 5" in t for t in CALLS[0])
      and not any("question 0 " in t for t in CALLS[0]),
      "turn 11 went to summarize(), turn 1 did not")
check(_lost_turns(CONV_TRUNC, _TRUNC, out) == [], "and the oracle finds no lost turn")

print("[12b] F7: tool-role turns in a full history do not switch reuse off")
# The length gate counted user/assistant turns against a position that counts
# every non-system role: two tool messages and it declined for good.
CONV_TOOL = "reuse_tool_turns"
_TOOLS = history(24)
_TOOLS[7] = {"role": "tool", "content": "tool result one " + "t " * 40, "tool_call_id": "a"}
_TOOLS[9] = {"role": "tool", "content": "tool result two " + "t " * 40, "tool_call_id": "b"}
_seed(CONV_TOOL, _TOOLS, [{"text": "TOOL-CHUNK", "first_turn": 1, "last_turn": 40}])
_st_tool = summarizer.load_state(CONV_TOOL)
_st_tool["turns_seen"] = 48
summarizer.save_state(CONV_TOOL, _st_tool)
out = _run(_TOOLS, CONV_TOOL)
check(any("TOOL-CHUNK" in str(m.get("content", "")) for m in out)
      and len(CALLS) == 1 and len(CALLS[0]) == 4,
      "*** F7: the stored summaries are reused and only the 4 uncovered turns go fresh")
check(_lost_turns(CONV_TOOL, _TOOLS, out) == [], "and the oracle finds no lost turn")

print("[12c] a partial reuse says so on EVERY request, not once per process")
CONV_LOUD = "reuse_loud"
_seed(CONV_LOUD, MSGS, [{"text": "LOUD-CHUNK", "first_turn": 1, "last_turn": 40}])
_LOUD = [dict(m) for m in MSGS]
_LOUD[9] = {"role": "assistant", "content": "answer 4 EDITED " + "word " * 60}
_hits = 0
for _ in range(2):
    with capture() as records:
        out = _run(_LOUD, CONV_LOUD)
    _hits += 1 if _find(records.records, "not in the stored summaries' covered-turn record") else 0
check(_hits == 2, f"both requests logged the refreshed turn (got {_hits})")

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
check(summarizer._covered_fps(_st_hole)[10:30] == [_U] * 20,
      "and the WRITER marked the hole's positions UNKNOWN")
out = _run(MSGS_HOLE, CONV_HOLE)
check(len(CALLS) == 1 and any("question 5" in t for t in CALLS[0]),
      "turn 11 — the first turn INSIDE the hole — reached summarize()")
check(len(CALLS) == 1 and any("question 15" in t for t in CALLS[0]),
      "and so did turn 31, which HOLE-CHUNK-B claims but cannot reach from turn 1")
check(len(CALLS) == 1 and not any("question 0" in t for t in CALLS[0]),
      "CONTROL: turns 1-10 were still reused — the gate narrowed, it did not "
      "give up")

print("[13b] a record that claims MORE than the chunks can back is capped")
# [13] passes whether or not the GATE reads _covered_prefix, because the
# writer marked the hole UNKNOWN. The gate's own cap is defence against a
# record that is WRONG. So build exactly that: a hole at 11-30 and a record
# that is a perfectly valid fingerprint list of turns 1..44 of this very array.
CONV_LIE = "reuse_record_overclaims"
MSGS_LIE = history(24)
_seed(CONV_LIE, MSGS_LIE,
      [{"text": "LIE-CHUNK-A", "first_turn": 1, "last_turn": 10},
       {"text": "LIE-CHUNK-B", "first_turn": 31, "last_turn": 44}], record=False)
_st_lie = summarizer.load_state(CONV_LIE)
_st_lie["covered_fps"] = "".join(_ctf(_ns(MSGS_LIE)[:44]))
summarizer.save_state(CONV_LIE, _st_lie)
check(len(summarizer._covered_fps(summarizer.load_state(CONV_LIE))) == 44,
      "fixture: the record is internally VALID for 44 turns, while the chunks "
      "can only back 10")
out = _run(MSGS_LIE, CONV_LIE)
check(len(CALLS) == 1 and any("question 5" in t for t in CALLS[0]),
      "turn 11, inside the hole, still reached summarize() — a record cannot "
      "vouch for turns no chunk summarized, however well it hashes")

print("[13c] an UNKNOWN entry inside the chain is refreshed, and only it")
CONV_UNK = "reuse_unknown_entry"
_seed(CONV_UNK, MSGS, [{"text": "UNK-CHUNK", "first_turn": 1, "last_turn": 40}])
_st_unk = summarizer.load_state(CONV_UNK)
_ent = summarizer._covered_fps(_st_unk)
_ent[6] = _U
_st_unk["covered_fps"] = "".join(_ent)
summarizer.save_state(CONV_UNK, _st_unk)
check(_plan(CONV_UNK, MSGS) == (40, {6}),
      f"exactly turn 7 reads as changed (got {_plan(CONV_UNK, MSGS)})")
out = _run(MSGS, CONV_UNK)
check(len(CALLS) == 1 and any("question 3 " in t for t in CALLS[0]) and len(CALLS[0]) == 5,
      "turn 7 went fresh with the 4 uncovered turns, and nothing else did")

print("[14] an array that shares NOTHING with the record reuses nothing")
CONV_ALL = "reuse_all_changed"
_seed(CONV_ALL, history(24), [{"text": "ALIEN-CHUNK", "first_turn": 1, "last_turn": 40}])
MSGS_ALIEN = history(24, tag="Z-")
check(_plan(CONV_ALL, MSGS_ALIEN) == (0, set()),
      "fixture: nothing pairs, so the plan covers nothing")
out = _run(MSGS_ALIEN, CONV_ALL)
check(not any("ALIEN-CHUNK" in str(m.get("content", "")) for m in out),
      "no stored text reached the array")
check(len(CALLS) == 1 and len(CALLS[0]) == baseline,
      f"and the whole older span was summarized fresh ({len(CALLS[0]) if CALLS else 0} "
      f"against {baseline})")

print("[15] an EDITED earlier turn keeps its correction, and costs only itself")
# B1, the only unrecoverable break in this path. OpenWebUI's edit-without-
# regenerate rewrites one message in the middle and keeps everything after it.
# The stored summary describes the PRE-EDIT text; replacing the corrected turn
# with it loses the correction for good. And the edit must cost THAT TURN, not
# every covered turn after it, and not for the rest of the conversation.
CONV_EDIT = "reuse_edited_turn"
MSGS_EDIT = history(24)
_seed(CONV_EDIT, MSGS_EDIT,
      [{"text": "PRE-EDIT-CHUNK", "first_turn": 1, "last_turn": 20},
       {"text": "PRE-EDIT-CHUNK-2", "first_turn": 21, "last_turn": 40}])
out = _run(MSGS_EDIT, CONV_EDIT)
check(any("PRE-EDIT-CHUNK" in str(m.get("content", "")) for m in out),
      "CONTROL: the unedited array reuses the stored summaries")
_EDITED = [dict(m) for m in MSGS_EDIT]
check(_EDITED[5]["role"] == "user" and "question 2" in _EDITED[5]["content"],
      "fixture: index 5 is turn 5, the user's 'question 2'")
_EDITED[5]["content"] = "CORRECTED-FACT the lighthouse is on the north cape " + ("word " * 60)
check(_plan(CONV_EDIT, _EDITED) == (40, {4}),
      "exactly turn 5 reads as changed, and the other 39 covered turns do not")
out = _run(_EDITED, CONV_EDIT)
_sent_ed = CALLS[0] if CALLS else []
check(any("CORRECTED-FACT" in t for t in _sent_ed),
      "the corrected turn went to summarize() — it was not deleted under the "
      "summary of the text the user corrected")
check(len(_sent_ed) == 5 and not any("question 0" in t for t in _sent_ed),
      f"and ONLY it plus the 4 uncovered turns went fresh (got {len(_sent_ed)}): "
      f"every unchanged covered turn still came off the shelf")
check(_sent_ed[:1] and "CORRECTED-FACT" in _sent_ed[0],
      "the refreshed turn is ahead of the newer uncovered ones — chronological")
# NOT FROZEN. The conversation grows, a rollup covers 41-60 from the edited
# array; turn 5 still reads as changed and every other covered turn reuses.
_GROWN = _EDITED + _ns(history(34))[48:]
_st_g = summarizer.load_state(CONV_EDIT)
_st_g["l1"].append({"text": "POST-EDIT-CHUNK", "first_turn": 41, "last_turn": 60})
_st_g["last_summarized_turn"] = 60
check(summarizer._record_chunk_fps(_st_g, 41, 60, _ns(_GROWN)[40:60]) is True
      and len(summarizer._covered_fps(_st_g)) == 60,
      "the record keeps growing after an edit — it is not frozen at the edit")
summarizer.save_state(CONV_EDIT, _st_g)
check(_plan(CONV_EDIT, _GROWN) == (60, {4}),
      "and on the grown conversation the edit still costs exactly one turn")

print("[15b] a DELETE among many IDENTICAL turns still pairs every one of them")
# A first cut paired turns with difflib's SequenceMatcher, which junks any
# element repeated in more than 1% of a 200+-element sequence: her "continue"
# or a run of identical redaction placeholders then read as changed and were
# refreshed on every request after any delete, and 1,990 turns half of which
# were one placeholder took 679 ms. Pairing is set membership now.
import time as _time  # noqa: E402
CONV_REP = "reuse_repeated_turns"
_REP = [{"role": "system", "content": "you are a companion"}]
for i in range(130):
    _REP.append({"role": "user", "content": "continue"})
    _REP.append({"role": "assistant", "content": f"answer {i} " + "word " * 60})
_seed(CONV_REP, _REP, [{"text": f"REP-CHUNK-{k}", "first_turn": 20 * k + 1,
                        "last_turn": 20 * (k + 1)} for k in range(12)])
_REP_DEL = [_REP[0]] + _ns(_REP)[2:]          # she deleted the first exchange
_cov_rep, _chg_rep = _plan(CONV_REP, _REP_DEL)
check(_cov_rep >= 238 and not any(c < 238 for c in _chg_rep),
      f"*** every repeated 'continue' still pairs after the delete (covered "
      f"{_cov_rep}, changed below 238: {sorted(c for c in _chg_rep if c < 238)})")
_big_rec = [f"{i:016x}" for i in range(1990)]
_big_now = [("f" * 16 if i % 2 else r) for i, r in enumerate(_big_rec[1:])]
_t0 = _time.perf_counter()
summarizer._paired_turns(_big_rec + ["f" * 16], _big_now)
_dt_pair = _time.perf_counter() - _t0
check(_dt_pair < 0.2,
      f"and pairing 1,990 turns, half one repeated fingerprint, is cheap "
      f"({_dt_pair * 1000:.1f} ms)")

print("[16] a state written before v3.1.9 has no record, and declines")
CONV_OLD = "reuse_pre_v319_state"
MSGS_OLD = history(24)
_seed(CONV_OLD, MSGS_OLD,
      [{"text": "OLD-FORMAT-CHUNK", "first_turn": 1, "last_turn": 40}], record=False)
check(summarizer.load_state(CONV_OLD).get("covered_fps") == "",
      "fixture: this state carries no record")
out = _run(MSGS_OLD, CONV_OLD)
check(not any("OLD-FORMAT-CHUNK" in str(m.get("content", "")) for m in out),
      "a pre-v3.1.9 hierarchy is not substituted without evidence")
check(len(CALLS) == 1 and any("question 0" in t for t in CALLS[0]),
      "and the turns went to summarize(), exactly as before the feature")
_raw = json.loads(summarizer.summary_path(CONV_OLD).read_text(encoding="utf-8"))
_raw["covered_fps"] = "0123456789abcdef" * 39 + "01234567"
summarizer.summary_path(CONV_OLD).write_text(json.dumps(_raw), encoding="utf-8")
check(summarizer.load_state(CONV_OLD).get("covered_fps") == "",
      "a torn record on disk loads as no record at all")

print("[16b] stored_turns_out reports what the RETURNED array reused, and "
      "nothing when there is no such array")
CONV_M1 = "reuse_m1_outparam"
_seed(CONV_M1, MSGS, [
    {"text": "M1-CHUNK-ONE", "first_turn": 1, "last_turn": 20},
    {"text": "M1-CHUNK-TWO", "first_turn": 21, "last_turn": 40},
])
_out_reused: list = []
out = _run(MSGS, CONV_M1, stored_turns_out=_out_reused)
check(any("M1-CHUNK" in str(m.get("content", "")) for m in out),
      "fixture: this call DID reuse the stored hierarchy")
check(_out_reused == [40],
      f"*** M1: stored_turns_out reports the 40 turns actually reused (got {_out_reused})")
_out_none_conv: list = []
_run(MSGS, None, stored_turns_out=_out_none_conv)
check(_out_none_conv == [0],
      f"CONTROL: no conv_id -> stored_turns_out reports 0 (got {_out_none_conv})")
_out_declined: list = []
_run(history(24, tag="Q-"), CONV_M1, stored_turns_out=_out_declined)
check(_out_declined == [0],
      f"CONTROL: turns the record does not hold -> 0 (got {_out_declined})")
_out_early: list = []
_run(history(2), CONV_M1, stored_turns_out=_out_early)
check(_out_early == [],
      f"an early return (under TARGET) leaves it empty (got {_out_early})")
out_default = _run(MSGS, CONV_M1)
check(any("M1-CHUNK" in str(m.get("content", "")) for m in out_default),
      "CONTROL: omitting stored_turns_out entirely still reuses normally")


async def _raising_summarize(client, to_summarize):
    raise RuntimeError("summarization backend refused the call")

main.summarize = _raising_summarize
_out_raised: list = []
try:
    asyncio.run(main.compact_if_needed(list(MSGS), CONV_M1, stored_turns_out=_out_raised))
    _raised = False
except RuntimeError:
    _raised = True
main.summarize = _spy_summarize
check(_raised, "fixture: summarize() raised out of compact_if_needed")
check(_out_raised == [],
      f"*** F4: a compaction that RAISED leaves stored_turns_out empty, so the "
      f"caller injects its own summary (got {_out_raised})")


# ---------------------------------------------------------------------------
# THE REAL WRITER, through the transformations production applies. From here
# on the record is written by maybe_rollup / _rollup_hierarchy / the backfill.
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
    check(summarizer._covered_fps(_st_rt) == _ctf(_ns(MSGS_RT)[:40]),
          "the rollup recorded exactly the turns its chunks read")
    out = _run(MSGS_RT, CONV_RT)
    check(any("ROLLED-CHUNK" in str(m.get("content", "")) for m in out)
          and CALLS and not any("question 0" in t for t in CALLS[0]),
          "end to end: state written by the real rollup is REUSED by the real gate")
    _REBUILT = history(24)
    _REBUILT[5] = {"role": "user", "content": "[this turn was not recorded]"}
    _st_none = asyncio.run(summarizer.maybe_rollup(
        "reuse_no_raw", _REBUILT, "http://stub", "m"))
    check(len(_st_none["l1"]) == 2
          and summarizer._covered_fps(_st_none) == _ctf(_ns(_REBUILT)[:40]),
          "*** F9: a caller with no raw array (the admin rebuild) records the "
          "text its chunks READ — a placeholder is recorded as a placeholder")
    check(_plan("reuse_no_raw", history(24)) == (40, {4}),
          "so the client's real turn at that position reads as changed, never blessed")

    print("[18] a REDACTED degenerate reply inside the covered span does not stop reuse")
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
    out = _run(NEXT_RD, CONV_RD)
    check(any("ROLLED-CHUNK" in str(m.get("content", "")) for m in out)
          and CALLS and not any("# Status" in t or "question 0" in t for t in CALLS[0]),
          "and reuse FIRES: no covered turn, degenerate or not, was re-summarized")

    print("[19] a STOPPED reply that closes a chunk is recorded AS STREAMED, at once")
    # F1. The chunk summarizes the trimmed reply; OpenWebUI keeps and re-sends
    # what streamed. The record is written with the chunk, from the streamed
    # text — never from the next request, which is what blessed a turn that
    # replaced it.
    CONV_TR = "reuse_trimmed"
    REQ_TR = _request(29)                       # 59 turns; the reply is turn 60
    _full_reply = "answer 29 " + "word " * 60
    asyncio.run(main._rollup_hierarchy(CONV_TR, REQ_TR, "answer 29 word word",
                                       reply_as_streamed=_full_reply))
    _st_t1 = summarizer.load_state(CONV_TR)
    check(summarizer._covered_prefix(_st_t1) == 60
          and summarizer._covered_fps(_st_t1)[59] == _ctf([{"role": "assistant", "content": _full_reply}])[0],
          f"the chunk closed on turn 60 and recorded the STREAMED reply in the "
          f"same call (prefix {summarizer._covered_prefix(_st_t1)}, record "
          f"{len(summarizer._covered_fps(_st_t1))})")
    REQ_TR2 = list(REQ_TR) + [{"role": "assistant", "content": _full_reply},
                              {"role": "user", "content": "question 30 " + "word " * 60}]
    REQ_TR3 = list(REQ_TR2) + [{"role": "assistant", "content": "answer 30 " + "word " * 60},
                               {"role": "user", "content": "question 31 " + "word " * 60}]
    check(_plan(CONV_TR, REQ_TR3) == (60, set()),
          f"and the full reply OpenWebUI re-sends matches it (got {_plan(CONV_TR, REQ_TR3)})")
    CONV_TR_NO = "reuse_trimmed_unstreamed"
    asyncio.run(main._rollup_hierarchy(CONV_TR_NO, REQ_TR, "answer 29 word word"))
    check(_plan(CONV_TR_NO, REQ_TR3) == (59, set()),
          f"CONTROL: recorded from the trimmed text instead, the re-sent reply "
          f"pairs with nothing, so the reuse span stops before it (got "
          f"{_plan(CONV_TR_NO, REQ_TR3)})")
    out = _run(REQ_TR3, CONV_TR_NO)
    check(len(CALLS) == 1 and any(t.startswith("answer 29 word word word") for t in CALLS_FULL[0])
          and _lost_turns(CONV_TR_NO, REQ_TR3, out) == [],
          "and it is summarized fresh on every request instead of reused: never lost")

    print("[20] a conversation past its first L3 refresh still reuses")
    CONV_L3 = "reuse_after_l3"
    MSGS_L3 = history(24)
    _st_l3 = summarizer.load_state(CONV_L3)
    _st_l3["l3"] = {"text": "L3-OVERVIEW", "first_turn": 1, "last_turn": 30}
    _st_l3["l1"] = [{"text": "L1-AFTER-L3", "first_turn": 31, "last_turn": 40}]
    _st_l3["last_summarized_turn"] = 40
    summarizer._record_chunk_fps(_st_l3, 1, 40, _ns(MSGS_L3)[:40])
    summarizer.save_state(CONV_L3, _st_l3)
    check(_plan(CONV_L3, MSGS_L3) == (40, set()),
          "the L3 span counts: 40 covered, none changed")
    out = _run(MSGS_L3, CONV_L3)
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

    print("[16c] a pre-v3.1.9 hierarchy is adopted ONCE, from the request, never the reply")
    # Her conversation is ~1,900 turns of chunks written before the record
    # existed; never adopting them is the 4-call cap refusal for good.
    CONV_LEG = "reuse_legacy_adopt"
    _LEG = history(24)
    _seed(CONV_LEG, _LEG, [{"text": "LEGACY-CHUNK", "first_turn": 1, "last_turn": 40}],
          record=False)
    _st_leg = summarizer.load_state(CONV_LEG)
    _st_leg["turns_seen"] = 48
    summarizer.save_state(CONV_LEG, _st_leg)
    _REQ_LEG = _LEG + [{"role": "user", "content": "question 24 " + "word " * 60}]
    asyncio.run(main._rollup_hierarchy(CONV_LEG, _REQ_LEG, "answer 24 " + "word " * 60))
    check(summarizer._covered_fps(summarizer.load_state(CONV_LEG)) == _ctf(_ns(_LEG)[:40]),
          "the first live rollup adopted positions 1-40 from the request")
    out = _run(_REQ_LEG, CONV_LEG)
    check(any("LEGACY-CHUNK" in str(m.get("content", "")) for m in out),
          "and reuse fires on the adopted hierarchy")
    CONV_LEG2 = "reuse_legacy_regenerate"
    _LEG2 = history(20)                     # a legacy chunk closed on turn 40
    _seed(CONV_LEG2, _LEG2, [{"text": "LEGACY-CHUNK-2", "first_turn": 1, "last_turn": 40}],
          record=False)
    _REGEN = _LEG2[:-1]                     # regenerate: re-send up to turn 39
    asyncio.run(main._rollup_hierarchy(CONV_LEG2, _REGEN, "answer 19 REGENERATED " + "word " * 60))
    _rec_leg2 = summarizer._covered_fps(summarizer.load_state(CONV_LEG2))
    check(len(_rec_leg2) == 39,
          f"*** the appended reply is never adopted: 39 positions, not 40 (got {len(_rec_leg2)})")
    _AFTER = _REGEN + [{"role": "assistant", "content": "answer 19 REGENERATED " + "word " * 60}] + _ns(history(24))[40:46]
    _AFTER = [_LEG2[0]] + _ns(_AFTER)
    asyncio.run(main._rollup_hierarchy(CONV_LEG2, _AFTER, "answer 23 " + "word " * 60))
    check(len(summarizer._covered_fps(summarizer.load_state(CONV_LEG2))) == 39,
          "and a later call does not adopt it either (one-shot)")

    print("[22] the backfill: the live tail owns the rollup in production order, "
          "and the backfill records from the UNREDACTED snapshot when it runs")
    import facts as _facts_mod

    async def _no_facts(*_a, **_k):
        return []

    _real_extract = _facts_mod.extract_facts_from_exchange
    _facts_mod.extract_facts_from_exchange = _no_facts
    try:
        for _live_first in (True, False):
            CONV_BF = f"reuse_backfill_{'tail' if _live_first else 'alone'}"
            REQ_BF = _request(23)
            REQ_BF[10] = {"role": "assistant", "content": DEGENERATE}
            _spawned: list = []
            _started = asyncio.run(backfill.start_backfill_if_needed(
                CONV_BF, list(REQ_BF), "http://stub", "m",
                fire_and_forget=lambda coro: _spawned.append(coro) or True,
                redact=main._redact_degenerate_turns,
            ))
            check(_started and len(_spawned) == 1, f"fixture: a backfill was scheduled ({CONV_BF})")
            if _live_first:
                # PRODUCTION ORDER: the kickoff request's own reply streams in
                # seconds and its tail rolls up minutes before the backfill
                # finishes extracting facts.
                asyncio.run(main._rollup_hierarchy(CONV_BF, REQ_BF, "answer 23 " + "word " * 60))
            _before_bf = summarizer.load_state(CONV_BF)
            with capture("compactor.backfill") as _bfrec:
                asyncio.run(_spawned[0]) if _spawned else None
            _after_bf = summarizer.load_state(CONV_BF)
            if _live_first:
                check(_find(_bfrec.records, "summary rollup skipped") is not None
                      and _after_bf.get("tail_fp") == _before_bf.get("tail_fp")
                      and _after_bf.get("covered_fps") == _before_bf.get("covered_fps"),
                      "production order: the backfill skips its rollup, and the live "
                      "tail's anchor and record are untouched")
            else:
                check(summarizer._covered_prefix(_after_bf) == 40,
                      "no live tail: the backfill's own rollup covered 1-40")
                check(summarizer._covered_fps(_after_bf)[9] == _ctf([REQ_BF[10]])[0],
                      "and recorded turn 10 from the UNREDACTED snapshot (the "
                      "degenerate reply as the client sends it), not the placeholder")
            backfill._in_progress_local.discard(CONV_BF)

        print("[22b] reviewer E F3: a live tail QUEUED on the backfill's facts-merge "
              "lock still keeps its anchor")
        # The stale-snapshot check used to be read outside conv_lock: a live
        # tail created while the backfill's facts merge held the lock ran its
        # rollup after the merge released, the backfill read the OLD position
        # in between, then rolled up its 38-turn snapshot over the live 40.
        # Adapted from reviewer E's p3e_c6_backfill_toctou.py; the live tail's
        # summarization call is held open so the backfill reaches its check
        # while the tail is mid-rollup.
        CONV_Q = "reuse_backfill_queued"
        _snap = _ns(history(19))            # 38 turns
        _live = _ns(history(20))            # 40 turns

        async def _queued():
            await summarizer.maybe_rollup(CONV_Q, _snap, "http://stub", "m", raw_messages=_snap)
            _res: dict = {}
            _tasks: list = []

            async def _slow_pieces(conv_id, client, vllm_url, model, prompt, pieces, max_tokens):
                await asyncio.sleep(0.3)
                return f"LIVE over {len(pieces)} piece(s)"

            async def _live_tail():
                summarizer._summarize_pieces = _slow_pieces
                try:
                    s_ = await summarizer.maybe_rollup(CONV_Q, _live, "http://stub", "m",
                                                       raw_messages=_live)
                finally:
                    summarizer._summarize_pieces = _stub_pieces
                _res["tail_fp"] = list(s_.get("tail_fp") or [])
                _res["turns_seen"] = s_.get("turns_seen")

            _real_merge = backfill._merge_backfilled

            def _merge_spy(on_disk, accumulated):
                _tasks.append(asyncio.get_running_loop().create_task(_live_tail()))
                return _real_merge(on_disk, accumulated)

            backfill._merge_backfilled = _merge_spy
            try:
                await backfill._run_backfill(CONV_Q, _snap, "http://stub", "m", raw_messages=_snap)
            finally:
                backfill._merge_backfilled = _real_merge
            await asyncio.gather(*_tasks)
            return _res

        _q = asyncio.run(_queued())
        _after_q = summarizer.load_state(CONV_Q)
        check(_q.get("turns_seen") == 40, f"fixture: the live tail reached 40 (got {_q.get('turns_seen')})")
        check(_after_q.get("tail_fp") == _q.get("tail_fp") and _after_q.get("window_turns") == 40,
              f"*** E-F3: the live 40-turn anchor survives the stale backfill (window_turns "
              f"{_after_q.get('window_turns')})")
        backfill._in_progress_local.discard(CONV_Q)
    finally:
        _facts_mod.extract_facts_from_exchange = _real_extract
finally:
    summarizer._summarize_pieces = _real_pieces

main.summarize = _real

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll compaction-reuse checks passed.")
