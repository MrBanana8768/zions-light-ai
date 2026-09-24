"""
v3.1.9 (hostile pass #5, lane p5-drain): a failing upper tier must not
freeze the tiers below it, an L3 refresh under a failing /tokenize must not
silently truncate content it claims to cover, and health must not let one
converging conversation hide a genuinely stuck one.

C5-1/E1 (HIGH, regression vs v3.1.7/v3.1.8/21645f2): the single L3>L2>L1
drain loop `summarizer._maybe_rollup_body` added for the tail catch-up
feature `break`s the WHOLE pass the moment any upper-tier unit fails (a
torn archive sidecar, an archive write that keeps failing, or an L3/L2
reply that strips to empty). Before v3.1.9, L1/L2/L3 were three SEPARATE
while/if loops, so a broken upper tier could only ever defer itself. Because
the SAME drain runs on every LATER call too, and because it re-tries the
BROKEN tier FIRST every time, a persistently failing tier froze every lower
tier forever -- L1 stopped advancing from that turn on, and the
"/compact drains the backlog" advice health gives for a stuck hierarchy
runs the identical drain and hits the identical failure first, doing
nothing. Fixed: a failed unit is marked failed FOR THIS CALL and the drain
falls through to the tier below instead of ending the pass (summarizer.py,
`_maybe_rollup_body`'s while loop).

C5-2 (MEDIUM, pre-existing): `_do_l3_rollup`'s stage 2 used to hand stage
1's output on as ONE piece, even when stage 1's own map-reduce had already
given up and CONCATENATED 2-3 of its own parts (routine whenever /tokenize
is down and the chapter count is not tiny). `_batch_to_budget` priced the
whole blob, found it over ITS OWN budget, and `_truncate_to_budget`
hard-cut it -- silently, while L3 kept claiming coverage from turn 1.
Fixed in two parts: `_do_l3_rollup` splits BOTH the prior-L3 text and the
newer-chapters text back into pieces on their own "\\n\\n" joins before
handing them to stage 2 (so `_batch_to_budget` batches them normally
instead of truncating one oversized blob); and `_summarize_pieces_raw`'s
shared reduce (used by every tier) now pairs adjacent parts and folds
them two at a time on a give-up instead of concatenating immediately, so
a tier's own output stays near one call's bounded size instead of
growing without limit across repeated refreshes (an unbounded-growth
regression the first cut of this fix introduced and [5]'s own
boundedness check catches).

E6 (LOW-MEDIUM): health.py's hierarchy-lag reason judged only the single
worst-lag RECENT conversation; a converging catch-up with a larger lag
could mask a genuinely stuck one with a smaller lag for as long as it kept
advancing within `_CATCHUP_RECENT_S`. Fixed: every recent conversation over
the limit is judged, and the reason names the worst one that is NOT
converging.

Drives the REAL summarizer/main/health chain end to end -- only
`summarizer._llm_summarize`, `summarizer._batch_to_budget` and (for C5-2
only) `summarizer._count_tokens` are doubled, the same "double only the LLM
seams" doctrine test_tail_catchup.py already uses.

    python test_p5_drain.py
"""
import asyncio
import logging
import os
import random
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="p5-drain-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"

import facts  # noqa: E402
import health  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import summarizer  # noqa: E402

memory.ensure_storage_layout()


async def _noop_extract(client, vllm_url, model, user_msg, assistant_msg,
                         existing_facts, **kw) -> list[str]:
    return []


facts.extract_facts_from_exchange = _noop_extract

FAILED: list[str] = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


# ---------------------------------------------------------------------------
# Stub LLM seam -- fails FOREVER for one chosen tier's prompt, succeeds for
# every other. Keyed on the REAL system prompts (_PROMPT_L1/_PROMPT_L2/
# _PROMPT_L3), so "L3 fails on every attempt" here is exactly the
# reviewers' own reproduction trigger: an empty reply, indistinguishable to
# this module from a torn archive sidecar aborting the refresh before any
# text is even produced (both make _do_l3_rollup return False).
# ---------------------------------------------------------------------------
LLM_CALLS: list[str] = []  # tier labels, in call order, across the whole run
_FAIL_TIER = {"l3": False, "l2": False}


def _tier_of(system_prompt: str) -> str:
    return {
        summarizer._PROMPT_L1: "l1",
        summarizer._PROMPT_L2: "l2",
        summarizer._PROMPT_L3: "l3",
        summarizer._PROMPT_REDUCE: "reduce",
    }.get(system_prompt, "?")


async def _stub_llm_summarize(client, vllm_url, model, system_prompt, body_text,
                               max_tokens, *, timeout=300.0):
    tier = _tier_of(system_prompt)
    LLM_CALLS.append(tier)
    if _FAIL_TIER.get(tier):
        return ""
    return f"summary of {len(body_text)} chars (#{len(LLM_CALLS)})"


summarizer._llm_summarize = _stub_llm_summarize

# Saved BEFORE stubbing below, not re-read off `summarizer` later -- once
# `summarizer._batch_to_budget` is overwritten, the module has no other
# reference to the real one, so section [5] restores THIS name, not
# "whatever summarizer._batch_to_budget happens to hold at that point"
# (that mistake made an earlier draft of this file silently run section
# [5] against the ONE-BATCH stub below and observe zero truncations for
# the wrong reason -- the real map-reduce never ran at all).
_REAL_BATCH_TO_BUDGET = summarizer._batch_to_budget


async def _stub_batch_to_budget(conv_id, client, vllm_url, model, pieces, budget):
    # One call per unit -- this file is about TIER interaction, not unit
    # cost (test_tail_catchup.py already covers the map-reduce cost shape).
    return [pieces]


summarizer._batch_to_budget = _stub_batch_to_budget


def reset_all(l1=5, l2=3, l3=2):
    summarizer.L1_CHUNK_SIZE = l1
    summarizer.L2_CHUNK_SIZE = l2
    summarizer.L3_CHUNK_SIZE = l3
    _FAIL_TIER["l3"] = False
    _FAIL_TIER["l2"] = False
    LLM_CALLS.clear()
    summarizer._reset_catchup_progress_for_tests()
    health._reset_hierarchy_progress_for_tests()


def fresh_state(conv_id: str) -> None:
    summarizer.save_state(
        conv_id, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0}
    )


def build_messages(n_messages: int) -> list[dict]:
    """One message == one turn, matching test_tail_catchup.py's own
    build_messages (see that file's docstring for why: _turn_pieces counts
    per MESSAGE, not per exchange)."""
    msgs = [{"role": "system", "content": "You are a patient assistant."}]
    for i in range(1, n_messages + 1):
        role = "user" if i % 2 == 1 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}: detail about item {i}."})
    return msgs


def run_tail(conv_id: str, messages: list[dict], *, max_calls: int) -> tuple[dict, list[str]]:
    """One call to the REAL tail path. Returns (state after, tier labels of
    the real calls THIS call made)."""
    before = len(LLM_CALLS)
    prior = main.TAIL_ROLLUP_MAX_CALLS
    main.TAIL_ROLLUP_MAX_CALLS = max_calls
    try:
        asyncio.run(main._rollup_hierarchy(conv_id, messages, None))
    finally:
        main.TAIL_ROLLUP_MAX_CALLS = prior
    return summarizer.load_state(conv_id), LLM_CALLS[before:]


# ===========================================================================
# [1] C5-1/E1: an L3 refresh that fails on EVERY attempt must not block L1
#     or L2 -- the HIGH regression.
# ===========================================================================
print("[1] a permanently failing L3 refresh does not freeze L1/L2 "
      "(C5-1/E1, control: v3.1.7 let L1 progress past a failing L3)")

reset_all(l1=5, l2=3, l3=2)
CID_1 = "p5drain-l3-stuck"
BACKLOG_1 = 20 * summarizer.L1_CHUNK_SIZE  # 20 L1 chunks -> 6 L2 chapters -> L3 due repeatedly
MSGS_1 = build_messages(BACKLOG_1)
fresh_state(CID_1)
_FAIL_TIER["l3"] = True

watermarks_1: list[int] = []
l2_counts_1: list[int] = []
l3_attempted_then_fell_through = False
for turn in range(1, 16):
    st, calls = run_tail(CID_1, MSGS_1, max_calls=3)
    watermarks_1.append(st.get("last_summarized_turn", 0))
    l2_counts_1.append(len(st.get("l2") or []))
    # Fall-through evidence: an "l3" attempt in THIS pass followed by an
    # "l1" or "l2" attempt in the SAME pass -- proves the loop did not
    # `break` the moment L3 failed.
    if "l3" in calls:
        i = calls.index("l3")
        if any(c in ("l1", "l2") for c in calls[i + 1:]):
            l3_attempted_then_fell_through = True
    if not summarizer.needs_rollup(st, st.get("turns_seen", 0)) and st.get("l3") is None:
        # Only l3 can still be "needed" forever; if l1/l2 are both clear
        # this conversation has done everything it can.
        if not summarizer._needs_l1_rollup(st, st.get("turns_seen", 0)) and \
           not summarizer._needs_l2_rollup(st):
            break

check(watermarks_1[-1] == BACKLOG_1,
      f"L1 fully caught up to turn {BACKLOG_1} despite L3 failing on EVERY "
      f"attempt (reached {watermarks_1[-1]} over {len(watermarks_1)} "
      f"turn(s)): {watermarks_1}")
check(all(watermarks_1[i] >= watermarks_1[i - 1] for i in range(1, len(watermarks_1))),
      "the watermark never went backwards while L3 kept failing")
check(l3_attempted_then_fell_through,
      "at least one pass tried L3 (and it failed), then fell through to "
      "L1/L2 in the SAME pass -- the actual C5-1/E1 mechanism, not just "
      "an across-call retry")
check(l2_counts_1[-1] > summarizer.L3_CHUNK_SIZE - 1,
      f"L2 kept folding past its normal L3_CHUNK_SIZE-1 bound because L3 "
      f"never consumed it (got {l2_counts_1[-1]} chapters) -- the accepted "
      f"v3.1.7-shape trade for a persistently broken upper tier, not a "
      f"NEW regression")
check(summarizer.load_state(CID_1).get("l3") is None,
      "L3 itself never succeeded (it cannot, by construction of this test)")
# Budget semantics unchanged: no pass spent more than its budget, since
# every unit here costs exactly 1 stubbed call (see _stub_batch_to_budget).
print(f"      watermarks: {watermarks_1}")
print(f"      l2 chapter counts: {l2_counts_1}")

print()
print("[1-control] the SAME setup with NO tier failing reaches L3 too -- "
      "the fix does not just refuse every failure, it still lets a "
      "healthy hierarchy converge all the way")
reset_all(l1=5, l2=3, l3=2)
CID_1C = "p5drain-l3-control"
MSGS_1C = build_messages(BACKLOG_1)
fresh_state(CID_1C)
st_c = None
for _ in range(30):
    st_c, _ = run_tail(CID_1C, MSGS_1C, max_calls=3)
    if not summarizer.needs_rollup(st_c, st_c.get("turns_seen", 0)):
        break
check(st_c is not None and st_c.get("last_summarized_turn", 0) == BACKLOG_1,
      f"CONTROL: fully caught up to {BACKLOG_1} "
      f"(got {st_c and st_c.get('last_summarized_turn')})")
check(st_c is not None and st_c.get("l3") is not None,
      "CONTROL: L3 itself completed when nothing was failing -- the guard "
      "says yes when it should, not just no")
check(st_c is not None and len(st_c.get("l2") or []) < summarizer.L3_CHUNK_SIZE,
      f"CONTROL: L2 stayed bounded under L3_CHUNK_SIZE once L3 could "
      f"consume it (got {len(st_c.get('l2') or [])}) -- the injection-size "
      f"bound the priority order exists for is unchanged for a healthy "
      f"catch-up")


# ===========================================================================
# [2] C5-1/E1: the SAME fall-through, one tier down -- a permanently
#     failing L2 fold must not block L1.
# ===========================================================================
print()
print("[2] a permanently failing L2 fold does not freeze L1 (one tier down)")

reset_all(l1=5, l2=3, l3=1000)  # L3 out of reach: this is purely an L2/L1 test
CID_2 = "p5drain-l2-stuck"
BACKLOG_2 = 10 * summarizer.L1_CHUNK_SIZE
MSGS_2 = build_messages(BACKLOG_2)
fresh_state(CID_2)
_FAIL_TIER["l2"] = True

watermarks_2: list[int] = []
for _ in range(10):
    st, calls = run_tail(CID_2, MSGS_2, max_calls=3)
    watermarks_2.append(st.get("last_summarized_turn", 0))
    if not summarizer._needs_l1_rollup(st, st.get("turns_seen", 0)):
        break

check(watermarks_2[-1] == BACKLOG_2,
      f"L1 fully caught up to turn {BACKLOG_2} despite L2 failing on "
      f"EVERY attempt (reached {watermarks_2[-1]}): {watermarks_2}")
final_2 = summarizer.load_state(CID_2)
check(len(final_2.get("l1") or []) > summarizer.L2_CHUNK_SIZE - 1,
      f"L1's own chunk list kept growing past its normal L2_CHUNK_SIZE-1 "
      f"bound because L2 never consumed it (got "
      f"{len(final_2.get('l1') or [])} chunks) -- same accepted trade as "
      f"[1], one tier down")
check(not (final_2.get("l2") or []),
      "L2 itself never produced a chapter (it cannot, by construction)")


# ===========================================================================
# [3] C5-1/E1: the failing tier is VISIBLE -- summarizer.failing_tier_for,
#     logged once, and health's reason names the tier instead of a /compact
#     no-op.
# ===========================================================================
print()
print("[3] the failing tier is visible: failing_tier_for, once-per-conv "
      "logging, and health names it instead of a dead /compact suggestion")

reset_all(l1=4, l2=2, l3=2)
CID_3 = "p5drain-l3-visible"
BACKLOG_3 = summarizer.L2_CHUNK_SIZE * summarizer.L1_CHUNK_SIZE * summarizer.L3_CHUNK_SIZE
MSGS_3 = build_messages(BACKLOG_3)
fresh_state(CID_3)
_FAIL_TIER["l3"] = True

check(summarizer.failing_tier_for(CID_3) is None,
      "fixture: no failure on record before any pass has run")

# Drain L1/L2 fully first (a few generous passes), then keep polling with
# the SAME messages so L1 has nothing left to do but L3 keeps failing --
# this is what drives health.HIERARCHY_STALL_DECISIONS worth of
# no-advance passes without an artificial hook into the counter.
for _ in range(10):
    st3, _ = run_tail(CID_3, MSGS_3, max_calls=10)
    if not summarizer._needs_l1_rollup(st3, st3.get("turns_seen", 0)) and \
       not summarizer._needs_l2_rollup(st3):
        break

check(summarizer.failing_tier_for(CID_3) == "l3",
      f"failing_tier_for names L3 the moment it has failed, independent "
      f"of whether L1 is still stalled overall (got "
      f"{summarizer.failing_tier_for(CID_3)!r})")

# Capture ERROR log lines to prove the "once per conversation" property
# (logsetup.log_once) -- many more passes below must not print a second one.
class _ErrCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "L3 refresh failed" in msg:
            self.lines.append(msg)


_err = _ErrCounter()
logging.getLogger("compactor.summarizer").addHandler(_err)
try:
    for _ in range(health.HIERARCHY_STALL_DECISIONS + 5):
        st3, _ = run_tail(CID_3, MSGS_3, max_calls=3)
finally:
    logging.getLogger("compactor.summarizer").removeHandler(_err)

check(len(_err.lines) <= 1,
      f"the L3-stuck ERROR line printed at most once across "
      f"{health.HIERARCHY_STALL_DECISIONS + 5} more passes, not once per "
      f"pass (got {len(_err.lines)}: {_err.lines})")

# health's hierarchy_lag is computed purely from the L1 watermark
# (turns_seen - last_summarized_turn) -- and this fix's whole point is
# that L1 keeps draining past a broken L3, so by now this conversation's
# OWN lag has settled near 0 (L1 has nothing left to do), the same as any
# other live, self-healing conversation. THAT is the fix working, not a
# gap: a live conversation whose L1 keeps pace is not what an operator
# needs paged for. What health.py's "stuck" reason is for is a
# conversation that genuinely stopped covering new turns -- so the rest
# of this check seeds exactly that shape directly (the same technique
# [4] below and the reviewer's own mask.py use): a real recorded lag over
# the limit, plus HIERARCHY_STALL_DECISIONS real no-advance passes,
# reusing the L3 failure this section already drove through the REAL
# drain above (failing_tier_for(CID_3) is real evidence, not seeded).
st3 = summarizer.load_state(CID_3)
st3["turns_seen"] = st3.get("last_summarized_turn", 0) + 2 * summarizer.L1_CHUNK_SIZE + 10
summarizer.save_state(CID_3, st3)
_wm3 = st3.get("last_summarized_turn", 0)
# Age the real drive's own last advance out of the "recent" window before
# recording the no-advance passes -- otherwise _catchup_verdict reads
# "converging" from that real, but now stale, advance forever (this whole
# test runs in well under health._CATCHUP_RECENT_S of wall-clock time).
# Exactly what test_tail_catchup.py [6] does to age an advance directly,
# rather than a real 15-minute sleep.
if CID_3 in summarizer._catchup_progress:
    summarizer._catchup_progress[CID_3]["last_advance_monotonic"] -= (
        health._CATCHUP_RECENT_S + 1
    )
for _ in range(health.HIERARCHY_STALL_DECISIONS):
    summarizer.record_catchup_pass(CID_3, _wm3, _wm3, True)

payload_3 = asyncio.run(health.gather_health_full(main.VLLM_URL, 4000))
reason_3 = next(
    (r for r in (payload_3.get("status_reasons") or []) if "turns behind" in r), ""
)
verdict_3 = (
    (payload_3.get("checks") or {}).get("hierarchy", {}).get("catching_up") or {}
).get("verdict")
print(f"      verdict={verdict_3!r}")
print(f"      reason={reason_3!r}")
check(verdict_3 == "stuck",
      f"health reads this conversation as stuck once L1 has nothing left "
      f"but L3 keeps failing (got {verdict_3!r})")
check("L3" in reason_3 and "failed every attempt" in reason_3,
      f"the reason NAMES the L3 tier instead of the bare /compact advice "
      f"(got {reason_3!r})")
check("will NOT clear this backlog" in reason_3,
      "the reason says /compact will NOT help here -- it runs the "
      "identical drain and hits the identical failure first")


# ===========================================================================
# [4] E6: a converging conversation with a LARGER lag must not hide a
#     genuinely stuck one with a smaller lag.
# ===========================================================================
print()
print("[4] E6: a converging conversation does not mask a stuck one")

reset_all(l1=20, l2=10_000, l3=10_000)
LIMIT_4 = 2 * summarizer.L1_CHUNK_SIZE  # 40, matches health's own computation

CID_A = "p5drain-e6-stuck"
CID_B = "p5drain-e6-converging"
summarizer.save_state(CID_A, {
    "l1": [], "l2": [], "l3": None,
    "last_summarized_turn": 0, "turns_seen": LIMIT_4 + 50,
})
summarizer.save_state(CID_B, {
    "l1": [], "l2": [], "l3": None,
    "last_summarized_turn": 0, "turns_seen": LIMIT_4 + 300,
})
# A: stuck -- HIERARCHY_STALL_DECISIONS no-advance passes with work due.
for _ in range(health.HIERARCHY_STALL_DECISIONS):
    summarizer.record_catchup_pass(CID_A, 0, 0, True)
# B: converging -- a recent advance, and a MUCH larger lag than A.
summarizer.record_catchup_pass(CID_B, 0, 20, True)

check(health._catchup_verdict(CID_A) == "stuck", "fixture: A reads stuck")
check(health._catchup_verdict(CID_B) == "converging", "fixture: B reads converging")
check(LIMIT_4 + 300 - 20 > LIMIT_4 + 50,
      "fixture: B's lag is genuinely larger than A's -- B would have won "
      "the OLD single-worst-lag comparison")

payload_4 = asyncio.run(health.gather_health_full(main.VLLM_URL, 4000))
reasons_4 = payload_4.get("status_reasons") or []
lag_reasons_4 = [r for r in reasons_4 if "turns behind" in r]
worst_conv_4 = (
    (payload_4.get("checks") or {}).get("hierarchy", {}).get("catching_up") or {}
).get("conv")
print(f"      single-worst conv (unchanged field): {worst_conv_4!r}")
print(f"      lag reason(s): {lag_reasons_4}")
check(worst_conv_4 == CID_B,
      f"the pre-existing single-worst field still names B (unchanged "
      f"shape for anything already reading it) -- got {worst_conv_4!r}")
check(any(f"conv={CID_A}" in r for r in lag_reasons_4),
      f"but the STATUS REASON names the STUCK conversation A, not just "
      f"the larger-lag converging one B -- this is the E6 fix "
      f"(got {lag_reasons_4})")
check(not any(f"conv={CID_B}" in r for r in lag_reasons_4),
      "and does not ALSO fire a reason for the converging B -- a "
      "self-healing catch-up is still not an alarm")

health._reset_hierarchy_progress_for_tests()
summarizer._reset_catchup_progress_for_tests()


# ===========================================================================
# [5] C5-2: an L3 refresh under a failing /tokenize must not silently
#     truncate content it claims to cover.
# ===========================================================================
print()
print("[5] C5-2: an L3 refresh with /tokenize down never truncates a piece "
      "it claims to cover")

# Fresh chunk sizes -- this section calls _do_l3_rollup directly (L1/L2
# never run), but leaving L3_CHUNK_SIZE at whatever an earlier section set
# it to (up to 10,000 in [4]) would build a 10,000-chapter fixture here by
# accident and make this section minutes slower for no reason.
summarizer.L3_CHUNK_SIZE = 5

rng = random.Random(5150)
VOCAB = ("the a she he they river lantern quiet morning garden letter window "
         "road said walked remembered carefully suddenly because although "
         "beneath over silver old new broken warm cold distant familiar "
         "Elena Marcus harbor tower promise secret journey storm candle").split()


def prose(chars: int) -> str:
    out, n = [], 0
    while n < chars:
        s = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(8, 22)))
        s = s[0].upper() + s[1:] + rng.choice([".", ".", ".", "?", "!"])
        out.append(s)
        n += len(s) + 1
    return " ".join(out)[:chars]


TRUNC_MSGS: list[str] = []


class _TruncCollector(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "has been truncated" in msg:
            TRUNC_MSGS.append(msg)


_trunc_handler = _TruncCollector()
logging.getLogger("compactor").addHandler(_trunc_handler)

# /tokenize DOWN: _count_tokens always falls back to the pessimistic
# ceiling, exactly what a real /tokenize outage produces (summarizer.
# _count_tokens' own except/status branches both end at _pessimistic_
# tokens -- this stub skips the network round trip, not the arithmetic).
async def _count_tokens_down(client, vllm_url, model, text):
    return summarizer._pessimistic_tokens(text) if text else 0


_orig_count_tokens = summarizer._count_tokens
summarizer._count_tokens = _count_tokens_down
summarizer._batch_to_budget = _REAL_BATCH_TO_BUDGET  # the real map-reduce batcher, not the stub


async def _stub_llm_long(client, vllm_url, model, system_prompt, body_text, max_tokens, **kw):
    # A model that uses ~100% of its output allowance, ~4 chars/token --
    # the OUTF=1.0 shape the reviewer's own l3loss.py measured the worst
    # loss under. Long enough that L3_CHUNK_SIZE chapters need more than
    # one map batch under the pessimistic 2.0-tokens/char pricing.
    return prose(max_tokens * 4)


summarizer._llm_summarize = _stub_llm_long

try:
    CID_5 = "p5drain-l3-trunc"
    turn = 0
    spans: list[tuple[int, int]] = []
    l3_lens: list[int] = []
    for r in range(4):
        st5 = summarizer.load_state(CID_5) if r else {
            "l1": [], "l3": None, "l2": [], "last_summarized_turn": 0,
        }
        st5["l2"] = [
            {
                "text": prose(int(1200 * 4)),
                "first_turn": turn + 1 + 200 * i,
                "last_turn": turn + 200 * (i + 1),
            }
            for i in range(summarizer.L3_CHUNK_SIZE)
        ]
        turn += 1000
        summarizer.save_state(CID_5, st5)
        s5 = summarizer.load_state(CID_5)

        async def _go():
            async with summarizer.httpx.AsyncClient() as c:
                return await summarizer._do_l3_rollup(
                    CID_5, c, "http://127.0.0.1:1", "m", s5
                )

        ok5 = asyncio.run(_go())
        summarizer.save_state(CID_5, s5)
        check(ok5, f"refresh {r + 1}: the L3 refresh itself succeeded")
        l3 = s5["l3"]
        spans.append((l3["first_turn"], l3["last_turn"]))
        l3_lens.append(len(l3["text"]))
        print(f"      refresh {r + 1}: span={l3['first_turn']}-{l3['last_turn']} "
              f"chars={len(l3['text'])} truncations_so_far={len(TRUNC_MSGS)}")

    # A second way this could have "fixed" C5-2 badly: never truncate, but
    # let the give-up concatenation this fix's first cut left in place
    # (see summarizer._summarize_pieces_raw's reduce, C5-2) grow L3's own
    # stored text by roughly one part's width every refresh, compounding
    # forever (measured against an earlier draft of this fix: 24003 ->
    # 48009 -> 72013 -> 96020 chars over these same 4 refreshes). The
    # pairwise-fold fallback in the shared reduce is what stops that --
    # every refresh here should land back near ONE call's own bounded
    # output size (L3_MAX_TOKENS*4 chars, the stub's own output length),
    # not grow with the refresh count.
    check(all(n < summarizer.L3_MAX_TOKENS * 4 * 2 for n in l3_lens),
          f"L3's stored text stays near one call's bounded output size on "
          f"EVERY refresh, not just the ones that happen not to truncate "
          f"-- no unbounded growth traded in for the fixed truncation "
          f"(L3_MAX_TOKENS*4*2={summarizer.L3_MAX_TOKENS * 4 * 2} ceiling, "
          f"got lengths {l3_lens})")

    check(spans[0][0] == 1,
          f"the first refresh's L3 claims coverage from turn 1 (got "
          f"{spans[0]})")
    check(all(s[0] == 1 for s in spans),
          f"EVERY refresh's L3 still claims coverage from turn 1 (span "
          f"first_turn never moves forward) -- unchanged by this fix, "
          f"still the pre-existing contract: {spans}")
    check(len(TRUNC_MSGS) == 0,
          f"NO piece was truncated across 4 successive refreshes with "
          f"/tokenize down and full-length chapter summaries -- the exact "
          f"shape the reviewer measured losing up to 46% of the newest "
          f"chapters and 19% of the prior story (got {len(TRUNC_MSGS)} "
          f"truncation(s): {TRUNC_MSGS})")
    check(all("a single rollup input piece measures" not in m for m in TRUNC_MSGS),
          "control on the reworded warning text itself: it did not fire, "
          "but if it ever does again it must say 'a single rollup input "
          "piece', not 'a single turn' (C5-2's wording fix)")

    print()
    print("[5b] the split protects an ALREADY-oversized prior L3 too -- "
          "e.g. one written before this fix, or by an older pod version, "
          "not only one this process just produced")
    # Directly seeded, not produced by a refresh: this is what a give-up
    # concatenation from BEFORE this fix (or a cross-version restore) left
    # on disk -- three "parts" joined with the same "\n\n" a give-up
    # produces, individually bounded, together far over budget. The
    # pairwise-fold fix above only protects output THIS process produces;
    # a state already shaped like this on disk needs the SPLIT in
    # _do_l3_rollup specifically, independent of the reduce fix -- this is
    # what makes that half of the fix earn its place rather than being
    # redundant with the other half (see the mutation table in the
    # report: reverting the split ALONE does not turn red once the
    # pairwise-fold fix stays in place, because this process never
    # produces an oversized blob to split any more -- this case exercises
    # the split against a blob this process did NOT produce).
    _old_part = prose(1200 * 4)
    _oversized_prior_text = "\n\n".join([_old_part, _old_part, _old_part])
    CID_5B = "p5drain-l3-trunc-oldstate"
    summarizer.save_state(CID_5B, {
        "l1": [], "l2": [
            {
                "text": prose(int(1200 * 4)),
                "first_turn": 5001 + 200 * i,
                "last_turn": 5200 + 200 * i,
            }
            for i in range(summarizer.L3_CHUNK_SIZE)
        ],
        "l3": {"text": _oversized_prior_text, "first_turn": 1, "last_turn": 5000},
        "last_summarized_turn": 6000,
    })
    _before_5b = len(TRUNC_MSGS)
    s5b = summarizer.load_state(CID_5B)

    async def _go_5b():
        async with summarizer.httpx.AsyncClient() as c:
            return await summarizer._do_l3_rollup(
                CID_5B, c, "http://127.0.0.1:1", "m", s5b
            )

    ok5b = asyncio.run(_go_5b())
    check(ok5b, "the refresh over an already-oversized prior L3 succeeded")
    check(s5b.get("l3", {}).get("first_turn") == 1,
          f"and still claims coverage from turn 1 (got "
          f"{s5b.get('l3', {}).get('first_turn')})")
    check(len(TRUNC_MSGS) == _before_5b,
          f"and did not truncate the pre-existing oversized prior L3 "
          f"(got {len(TRUNC_MSGS) - _before_5b} new truncation(s): "
          f"{TRUNC_MSGS[_before_5b:]})")
finally:
    summarizer._count_tokens = _orig_count_tokens
    summarizer._llm_summarize = _stub_llm_summarize
    summarizer._batch_to_budget = _stub_batch_to_budget
    logging.getLogger("compactor").removeHandler(_trunc_handler)


print()
if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("All p5-drain checks passed.")
