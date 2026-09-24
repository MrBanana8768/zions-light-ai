"""
v3.1.9 (V4_ROADMAP tail line): bounded hierarchy catch-up.

Owner's problem: when the L1/L2/L3 summary hierarchy falls far behind (days
of rollup failures, a vLLM outage, an upgrade that finds a stale watermark),
`summarizer.maybe_rollup`'s `while _needs_l1_rollup` / `while _needs_l2_rollup`
drain used to catch up ALL AT ONCE on the background tail -- however many
vLLM calls that took, on the same single GPU she is chatting on. This proves
the fix: the chat-path tail (main._rollup_hierarchy) now passes a per-turn
vLLM-call budget (main.TAIL_ROLLUP_MAX_CALLS, env COMPACTOR_TAIL_ROLLUP_MAX_
CALLS) into maybe_rollup, and the budget is enforced at the UNIT boundary
(summarizer._budget_allows_unit) rather than per real call -- the shape that
GUARANTEES at least one whole L1 chunk (or L2 fold, or L3 refresh) of
progress every turn that has work due, with a documented overshoot of at
most one unit's own calls, rather than livelocking on a backlog whose chunks
cost more than the configured budget.

Drives THROUGH THE REAL TAIL PATH: main._rollup_hierarchy, the exact
function both `_async_tail` call sites use (main.py ~5390), calling the real
summarizer.maybe_rollup / _maybe_rollup_body / _do_l1_rollup / _do_l2_rollup
/ _do_l3_rollup / _summarize_pieces_raw chain unchanged. Only the two LLM
seams are doubled (summarizer._llm_summarize, so no real vLLM is needed; and
summarizer._batch_to_budget, so a chunk's real-call COST is a controlled
knob -- 1 call or 3 -- rather than a function of this fixture's fake content
against a real /tokenize) -- the same "double only the LLM seams" doctrine
test_time_memory.py and test_p4c_compact_verdict.py already use.

    python test_tail_catchup.py
"""
import asyncio
import logging
import os
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="tail-catchup-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"

import backfill  # noqa: E402
import facts  # noqa: E402
import health  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import summarizer  # noqa: E402

memory.ensure_storage_layout()

# [5] drives backfill._run_backfill, whose facts-extraction half is not what
# this file tests (only the summarizer-rollup budget at its end is) and
# whose real shape is a per-exchange vLLM call regardless of
# COMPACTOR_FACTS_EXTRACTION (that env var gates a DIFFERENT decision —
# whether the extracted facts are considered a legitimate backfill target at
# all — not whether backfill attempts the call). Stubbed to empty/fast so
# [5] measures the rollup's own call count, not ~30 real connection timeouts.
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
# The two doubled LLM seams
# ---------------------------------------------------------------------------

# One entry per REAL call to summarizer._llm_summarize, in order:
# (conv-agnostic -- the fixture is single-conversation per test).
LLM_CALLS: list[str] = []
# If > 0, the next N real calls return "" (an LLM answering with nothing --
# indistinguishable, to this module, from a map batch genuinely failing) and
# the counter decrements once per call regardless of conv_id.
_FAIL_NEXT_CALLS = {"n": 0}


async def _stub_llm_summarize(client, vllm_url, model, system_prompt, body_text,
                               max_tokens, *, timeout=300.0):
    LLM_CALLS.append(body_text)
    if _FAIL_NEXT_CALLS["n"] > 0:
        _FAIL_NEXT_CALLS["n"] -= 1
        return ""
    return f"summary #{len(LLM_CALLS)} of {len(body_text)} chars"


summarizer._llm_summarize = _stub_llm_summarize

# Controls how many real calls ONE L1 chunk's map phase costs. `n=1` (the
# default): the whole chunk fits one batch, 1 call, no reduce -- the ordinary
# case for a real conversation whose turns are individually small next to
# L1_MAX_TOKENS. `n=2` or `n=3`: split into that many map batches PLUS one
# reduce call (2 -> 3 calls total; 3 -> 4), modelling the ~1,650-token-turn,
# 2-map-plus-1-reduce shape the brief and _budget_allows_unit's own docstring
# measure. Distinguishing "the top-level map call for one unit" from "an
# internal reduce round" by PIECE COUNT (`original_len`), set once per call
# below -- see the two branches' comments.
_FORCE_SPLIT = {"n": 1, "original_len": None}


def set_unit_cost(n: int) -> None:
    """How many real calls the NEXT unit(s) this stub sees will cost:
    `n` map batches (+ 1 reduce call whenever n > 1). Call before each
    run_tail() whose unit cost matters to the test; it is read once per
    _do_*_rollup call via `original_len` below and is NOT auto-reset --
    tests that need to go back to n=1 do so explicitly, the same
    "nothing is implicit" discipline this project's fixtures already use.
    """
    _FORCE_SPLIT["n"] = n
    _FORCE_SPLIT["original_len"] = None  # re-learned on the next top-level call


async def _stub_batch_to_budget(conv_id, client, vllm_url, model, pieces, budget):
    n = _FORCE_SPLIT["n"]
    if n <= 1 or len(pieces) <= 1:
        return [pieces]
    if _FORCE_SPLIT["original_len"] is None:
        # First call this unit: this IS the top-level map call (its own
        # piece count is whatever _do_l1_rollup/_do_l2_rollup/_do_l3_rollup
        # handed _summarize_pieces_raw). Remember it so a LATER call in the
        # same unit -- a reduce round, whose `pieces` are the map phase's
        # OWN output count, always smaller -- is recognised as the reduce
        # phase instead and kept together (see below): the test's requested
        # cost is exactly n map batches + 1 reduce call, not n^2 splits.
        _FORCE_SPLIT["original_len"] = len(pieces)
    if len(pieces) == _FORCE_SPLIT["original_len"]:
        size = max(1, -(-len(pieces) // n))  # ceil division
        return [pieces[i:i + size] for i in range(0, len(pieces), size)]
    # A reduce round (or _do_l3_rollup's second, prior-L3-folding call,
    # which always hands exactly 2 pieces): keep everything in ONE group so
    # it costs exactly one more real call, not a further split.
    return [pieces]


summarizer._batch_to_budget = _stub_batch_to_budget

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def fresh_state(conv_id: str) -> None:
    summarizer.save_state(
        conv_id, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0}
    )


def build_messages(n_messages: int) -> list[dict]:
    """`n_messages` non-system messages, alternating user/assistant.

    THIS PROJECT'S "turn" IS ONE MESSAGE, not one exchange: _turn_pieces
    walks non-system messages incrementing its index once per message
    (summarizer.py), and last_summarized_turn / L1_CHUNK_SIZE / _needs_l1_
    rollup all count in that same unit. `n_messages` is chosen so a backlog
    expressed as `k * summarizer.L1_CHUNK_SIZE` lands on an EXACT chunk
    boundary with no partial remainder — building it as `k` exchanges
    (`2*k` messages) here would silently double every count below it.
    """
    msgs = [{"role": "system", "content": "You are a patient assistant."}]
    for i in range(1, n_messages + 1):
        role = "user" if i % 2 == 1 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}: detail about item {i}."})
    return msgs


def run_tail(conv_id: str, messages: list[dict], *, max_calls: int | None) -> tuple[dict, int]:
    """One call to the REAL tail path (main._rollup_hierarchy), with
    `main.TAIL_ROLLUP_MAX_CALLS` set to `max_calls` for the duration (None
    leaves it at whatever the module currently has -- used only by the
    unbounded CONTROL, which patches it out entirely below). Returns
    (state after the call, real LLM calls made BY THIS CALL).
    """
    before = len(LLM_CALLS)
    prior = main.TAIL_ROLLUP_MAX_CALLS
    if max_calls is not None:
        main.TAIL_ROLLUP_MAX_CALLS = max_calls
    try:
        asyncio.run(main._rollup_hierarchy(conv_id, messages, None))
    finally:
        main.TAIL_ROLLUP_MAX_CALLS = prior
    return summarizer.load_state(conv_id), len(LLM_CALLS) - before


async def run_unbounded(conv_id: str, messages: list[dict]) -> dict:
    """The old, pre-budget behaviour: maybe_rollup with NO budget at all,
    called repeatedly until nothing more is due -- what an unbounded single
    drain converges to. Used as the CONTROL final state.
    """
    st = await summarizer.maybe_rollup(conv_id, messages, main.VLLM_URL, main.MODEL_REPO or "")
    while summarizer.needs_rollup(st, st.get("turns_seen", 0)):
        st = await summarizer.maybe_rollup(conv_id, messages, main.VLLM_URL, main.MODEL_REPO or "")
    return st


def reset_all(l1=20, l2=10, l3=5):
    summarizer.L1_CHUNK_SIZE = l1
    summarizer.L2_CHUNK_SIZE = l2
    summarizer.L3_CHUNK_SIZE = l3
    set_unit_cost(1)
    _FAIL_NEXT_CALLS["n"] = 0
    LLM_CALLS.clear()


# ===========================================================================
# (a) A hierarchy far behind catches up over successive turns.
# ===========================================================================
print("[a] a hierarchy far behind catches up over successive bounded turns")

# L2/L3 pushed out of reach so this scenario is purely about L1/watermark --
# tier interaction under a budget is its own test below (see "[3]").
reset_all(l1=10, l2=10_000, l3=10_000)
CID_A = "catchup-a"
BACKLOG_A = 19 * summarizer.L1_CHUNK_SIZE  # 19 whole chunks behind, nothing partial
MSGS_A = build_messages(BACKLOG_A)
fresh_state(CID_A)

watermarks = []
calls_per_turn = []
turn = 0
while True:
    turn += 1
    st, calls = run_tail(CID_A, MSGS_A, max_calls=4)
    watermarks.append(st.get("last_summarized_turn", 0))
    calls_per_turn.append(calls)
    if not summarizer.needs_rollup(st, st.get("turns_seen", 0)):
        break
    if turn > 50:
        break  # runaway guard for the test itself, not the feature

check(watermarks[-1] == BACKLOG_A,
      f"the hierarchy fully caught up to turn {BACKLOG_A} "
      f"(reached {watermarks[-1]} over {turn} turn(s))")
check(all(watermarks[i] > watermarks[i - 1] for i in range(1, len(watermarks))),
      f"the watermark strictly advanced EVERY turn while behind "
      f"({watermarks})")
# Overshoot bound: cost-1 chunks under budget=4 means up to 4 chunks/turn,
# never a 5th (the check refuses the 5th chunk's START once remaining<=0).
check(all(c <= 4 for c in calls_per_turn),
      f"no turn spent more real calls than its 4-call budget (cost-1 "
      f"chunks here, so no overshoot was even possible): {calls_per_turn}")
print(f"      {turn} turn(s), calls per turn: {calls_per_turn}")


# ===========================================================================
# (b) THE LIVELOCK CASE: budget smaller than one unit's own cost still
#     advances one whole chunk per turn.
# ===========================================================================
print()
print("[b] livelock case: budget=1, chunks costing 3 real calls each -- "
      "still advances one whole chunk per turn, never stalls")

reset_all(l1=10, l2=10_000, l3=10_000)
set_unit_cost(2)  # 2 map batches + 1 reduce = 3 real calls per L1 chunk
CID_B = "catchup-b"
BACKLOG_B = 6 * summarizer.L1_CHUNK_SIZE
MSGS_B = build_messages(BACKLOG_B)
fresh_state(CID_B)

wm_b = []
calls_b = []
for _ in range(10):
    st, calls = run_tail(CID_B, MSGS_B, max_calls=1)
    wm_b.append(st.get("last_summarized_turn", 0))
    calls_b.append(calls)
    if not summarizer.needs_rollup(st, st.get("turns_seen", 0)):
        break

check(wm_b[-1] == BACKLOG_B,
      f"budget=1 against 3-call chunks still fully caught up to "
      f"{BACKLOG_B} ({wm_b})")
check(all(wm_b[i] - wm_b[i - 1] == summarizer.L1_CHUNK_SIZE for i in range(1, len(wm_b))),
      f"EXACTLY one whole chunk advanced per turn, never zero, never more "
      f"than the documented one-unit overshoot ({wm_b})")
check(all(c == 3 for c in calls_b),
      f"every turn spent exactly the one unit's real cost (3), the "
      f"documented overshoot of budget(1) + 2 -- never more, never less: "
      f"{calls_b}")
print(f"      {len(wm_b)} turn(s) to catch up {BACKLOG_B} turns at 1 "
      f"chunk/turn: watermarks {wm_b}, calls {calls_b}")


# ===========================================================================
# (c) A map batch failing mid-chunk records NOTHING; the next turn redoes
#     that chunk.
# ===========================================================================
print()
print("[c] a map batch failing mid-chunk records nothing; the next turn "
      "redoes the identical chunk")

reset_all(l1=10, l2=10_000, l3=10_000)
set_unit_cost(2)  # 2 map batches, no reduce needed to observe the failure
CID_C = "catchup-c"
MSGS_C = build_messages(3 * summarizer.L1_CHUNK_SIZE)
fresh_state(CID_C)

_FAIL_NEXT_CALLS["n"] = 1  # one of the two map batches comes back empty
# max_calls=2, not generous: a budget tight enough that at most ONE chunk's
# worth of work can start this call -- a generous budget here would let a
# SECOND chunk begin after the first (failing) one aborts, and this test
# is about ONE chunk's fate, not the whole backlog's.
st_fail, calls_fail = run_tail(CID_C, MSGS_C, max_calls=2)
check(st_fail.get("last_summarized_turn", 0) == 0,
      f"the watermark did NOT advance on the pass with a failing map batch "
      f"(got {st_fail.get('last_summarized_turn')})")
check(not (st_fail.get("l1") or []),
      f"no L1 chunk was recorded from the failing pass (got "
      f"{st_fail.get('l1')})")
check(calls_fail == 2,
      f"both map batches were attempted (2 real calls) before the unit "
      f"failed as a whole (got {calls_fail})")

# The NEXT turn (no more forced failures): the identical chunk is redone,
# not skipped and not duplicated.
_FAIL_NEXT_CALLS["n"] = 0
st_retry, calls_retry = run_tail(CID_C, MSGS_C, max_calls=2)
check(st_retry.get("last_summarized_turn", 0) == summarizer.L1_CHUNK_SIZE,
      f"the retry turn recorded the SAME chunk (turns 1-"
      f"{summarizer.L1_CHUNK_SIZE}), not a gap or a duplicate (watermark="
      f"{st_retry.get('last_summarized_turn')})")
check(len(st_retry.get("l1") or []) == 1
      and (st_retry["l1"][0].get("first_turn"), st_retry["l1"][0].get("last_turn"))
      == (1, summarizer.L1_CHUNK_SIZE),
      f"exactly one chunk, covering exactly turns 1-{summarizer.L1_CHUNK_SIZE} "
      f"(got {st_retry.get('l1')})")
print(f"      failing pass: 0 recorded, {calls_fail} calls spent; retry "
      f"pass: 1 chunk recorded, watermark={st_retry.get('last_summarized_turn')}")


# ===========================================================================
# (d) CONTROL: a conversation that is NOT behind behaves byte-for-byte the
#     same whether or not a budget is passed.
# ===========================================================================
print()
print("[d] CONTROL: an ordinary (not-behind) turn is unaffected by the "
      "budget -- same calls, same writes, with or without one")

reset_all(l1=10, l2=10_000, l3=10_000)
CID_D_BUDGET = "catchup-d-budget"
CID_D_UNBOUNDED = "catchup-d-unbounded"
# Exactly one chunk's worth due -- well inside ANY sane budget, the ordinary
# steady-state shape (at most one L1 chunk due per real turn).
MSGS_D = build_messages(summarizer.L1_CHUNK_SIZE)
fresh_state(CID_D_BUDGET)
fresh_state(CID_D_UNBOUNDED)

LLM_CALLS.clear()
st_budget, calls_budget = run_tail(CID_D_BUDGET, MSGS_D, max_calls=4)
calls_after_budget = len(LLM_CALLS)

LLM_CALLS.clear()
st_unbounded = asyncio.run(run_unbounded(CID_D_UNBOUNDED, MSGS_D))
calls_unbounded = len(LLM_CALLS)

check(calls_budget == calls_unbounded,
      f"the same number of real calls were made with a budget (4) as "
      f"with none (got budget={calls_budget}, unbounded={calls_unbounded})")
check(st_budget.get("last_summarized_turn") == st_unbounded.get("last_summarized_turn")
      and [c.get("text") for c in (st_budget.get("l1") or [])]
      == [c.get("text") for c in (st_unbounded.get("l1") or [])],
      f"identical resulting state -- watermark "
      f"{st_budget.get('last_summarized_turn')} vs "
      f"{st_unbounded.get('last_summarized_turn')}, same chunk text")


# ===========================================================================
# (e) Request-path reuse resumes once the hierarchy has caught up.
# ===========================================================================
print()
print("[e] once caught up, the request path's compaction reuses the "
      "hierarchy instead of re-summarizing")

reset_all(l1=10, l2=10_000, l3=10_000)
CID_E = "catchup-e"
BACKLOG_E = 4 * summarizer.L1_CHUNK_SIZE
MSGS_E = build_messages(BACKLOG_E)
fresh_state(CID_E)
for _ in range(10):
    st, _ = run_tail(CID_E, MSGS_E, max_calls=4)
    if not summarizer.needs_rollup(st, st.get("turns_seen", 0)):
        break
check(not summarizer.needs_rollup(st, st.get("turns_seen", 0)),
      "fixture: the hierarchy is fully caught up before checking reuse")

# main.compact_if_needed's own reuse gate is exercised the same way
# test_p4c_compact_verdict.py exercises the request path: through a real
# admin.post/client.post. Here we call summarize()'s reuse helper directly
# -- summarizer.format_summary_block plus the coverage machinery is what
# tells the request path "these turns are already covered" -- because
# standing up a full /v1/chat/completions round trip needs episodic-store
# and degenerate-detection fixtures this file does not otherwise carry, and
# the property under test is entirely inside summarizer's own coverage
# check, not main's routing of it.
_final_state = summarizer.load_state(CID_E)
# _coverage_plan pairs `to_summarize`'s turns against the covered-turn
# record maybe_rollup wrote — the SAME MSGS_E array whose text the chunks
# above were built from, so every turn should pair and nothing should need
# a fresh read.
_covered, _refreshed = summarizer._coverage_plan(_final_state, MSGS_E)
check(_covered == BACKLOG_E and len(_refreshed) == 0,
      f"once caught up, compaction's coverage plan reads ALL {BACKLOG_E} "
      f"turns as covered by stored summaries, none needing a fresh "
      f"re-summarize (covered={_covered}, refreshed={len(_refreshed)})")
_block = summarizer.format_summary_block(_final_state, 4000)
check(bool(_block) and "summary #" in (_block or ""),
      "the injected summary block is built from the STORED chunk text "
      "(the stub's own marker), not re-summarized fresh")
print(f"      {len(_refreshed)} of {BACKLOG_E} turns need a fresh read on "
      f"the request path once caught up (the rest reuse stored summaries)")


# ===========================================================================
# [3] Tier order under a budget: L2/L3 are never starved, and l1/l2 stay
#     bounded near their fold thresholds for the WHOLE catch-up, not just
#     at the end of it.
# ===========================================================================
print()
print("[3] tier order: L2/L3 fold on schedule during a deep L1 catch-up "
      "instead of waiting for it to finish -- l1/l2 stay bounded throughout")

reset_all(l1=5, l2=3, l3=2)
CID_T = "catchup-tiers"
# Deep enough to build several L2 chapters and at least one L3 refresh
# while L1 is still nowhere near caught up: 40 chunks' worth.
BACKLOG_T = 40 * summarizer.L1_CHUNK_SIZE
MSGS_T = build_messages(BACKLOG_T)
fresh_state(CID_T)

l1_lens = []
l2_lens = []
for _ in range(400):
    st, _ = run_tail(CID_T, MSGS_T, max_calls=2)  # small: many turns to observe
    l1_lens.append(len(st.get("l1") or []))
    l2_lens.append(len(st.get("l2") or []))
    if not summarizer.needs_rollup(st, st.get("turns_seen", 0)):
        break

check(max(l1_lens) <= summarizer.L2_CHUNK_SIZE,
      f"l1 (injected into every request) never grew past L2_CHUNK_SIZE "
      f"({summarizer.L2_CHUNK_SIZE}) at any point in the catch-up -- "
      f"observed max {max(l1_lens)} over {len(l1_lens)} turns")
check(max(l2_lens) <= summarizer.L3_CHUNK_SIZE,
      f"l2 never grew past L3_CHUNK_SIZE ({summarizer.L3_CHUNK_SIZE}) at "
      f"any point -- observed max {max(l2_lens)}")
check(st.get("l3") is not None,
      "L3 was reached at least once during this catch-up (it was not "
      "starved for the whole run)")
check(st.get("last_summarized_turn") == BACKLOG_T,
      f"L1 still fully caught up by the end despite L2/L3 taking priority "
      f"whenever due (watermark={st.get('last_summarized_turn')})")
print(f"      {len(l1_lens)} turns; max(len(l1))={max(l1_lens)}, "
      f"max(len(l2))={max(l2_lens)}, l3 reached={st.get('l3') is not None}")


# ===========================================================================
# [3b] hostile follow-up: does priority order L3>L2>L1 change STEADY-STATE
#     (never-behind) behaviour when an L2 fold crosses the L3 threshold?
#
# THE CONCERN: L3 is checked FIRST every loop iteration, and `_l3_done`
# caps it at one refresh per maybe_rollup call. If L3 were ALREADY due at
# the very START of a call (leftover from an earlier call), and THIS call's
# own L1/L2 work then produced a SECOND L2 fold that crosses the L3
# threshold again, `_l3_done` would defer that second refresh to the NEXT
# call -- L3 "lagging one call behind" relative to the OLD L1-then-L2-then-
# L3 order, whose single trailing check always ran AFTER all of that call's
# L1/L2 work, catching everything in one consolidated refresh.
#
# WHY THIS DOES NOT REACH GENUINE STEADY STATE. For L3 to be due at a
# call's START, some EARLIER call must have left l2 sitting at/above
# L3_CHUNK_SIZE without refreshing it -- which never happens under ample
# (non-exhausted) budget: this priority loop only STOPS via a `break`, and
# the only `break` that means "nothing left to do" is the one reached when
# NEITHER L3 nor L2 nor L1 is due -- so with budget that never binds, L3 is
# always resolved (to len(l2)==0) before any call returns, every time,
# including this one's own leftover from any earlier call. And for the
# SECOND concern -- new L1/L2 growth WITHIN one call re-crossing the
# threshold -- an ordinary turn (one exchange) advances the observed
# position by exactly one exchange, so a single ordinary tail call can
# produce AT MOST one new L1 chunk (needs_l1_rollup only becomes true once
# every L1_CHUNK_SIZE messages) and therefore at most one new L2 fold: two
# fold events, each independently crossing L3_CHUNK_SIZE, cannot both occur
# from ONE exchange's worth of new material. The "lag a call behind" shape
# is real, but only for a CALL that itself processes many chunks at once
# (a deep catch-up, or an admin/backfill rebuild from the episodic store) --
# never for one exchange at a time, which is what "not behind" means.
#
# Proven directly, not argued: drive TURN-BY-TURN growth (one exchange
# appended, one tail call, exactly production's shape) through many L1/L2/L3
# boundary crossings with a budget generous enough to never bind, and assert
# l2 is NEVER observed at or above L3_CHUNK_SIZE at any call boundary --
# if L3 ever lagged a call behind here, this is exactly what would show it.
# ===========================================================================
print()
print("[3b] steady-state (one exchange per turn) never leaves l2 at/above "
      "L3_CHUNK_SIZE across a call boundary -- L3 does not lag a turn "
      "behind an L2 fold that crosses its threshold")

reset_all(l1=4, l2=2, l3=2)
CID_S = "catchup-steady-l3"
fresh_state(CID_S)
_steady_msgs: list[dict] = [{"role": "system", "content": "You are a patient assistant."}]
_l2_at_boundary: list[int] = []
_l3_refreshes = 0
_prev_l3_text = None
for _turn in range(1, 101):  # 100 exchanges: several L3_CHUNK_SIZE crossings
    _steady_msgs.append({"role": "user", "content": f"Message {2 * _turn - 1}."})
    _steady_msgs.append({"role": "assistant", "content": f"Message {2 * _turn}."})
    # Budget far larger than any ONE ordinary turn could possibly need (at
    # most 1 L1 chunk + 1 L2 fold + 1 L3 refresh = 3 units here), so this
    # never once binds -- the "ample/unbounded" precondition the argument
    # above depends on.
    st, _ = run_tail(CID_S, _steady_msgs, max_calls=20)
    _l2_at_boundary.append(len(st.get("l2") or []))
    if st.get("l3") and st["l3"].get("text") != _prev_l3_text:
        _l3_refreshes += 1
        _prev_l3_text = st["l3"].get("text")

check(max(_l2_at_boundary) < summarizer.L3_CHUNK_SIZE,
      f"l2 was NEVER observed at or above L3_CHUNK_SIZE "
      f"({summarizer.L3_CHUNK_SIZE}) at the end of any of {len(_l2_at_boundary)} "
      f"ordinary turns -- L3 never lagged a turn behind an L2 fold "
      f"(observed l2 lengths: {sorted(set(_l2_at_boundary))})")
check(_l3_refreshes >= 3,
      f"fixture: at least a few real L3 refreshes actually happened over "
      f"100 turns, so the check above is not vacuously true (got "
      f"{_l3_refreshes})")
print(f"      100 turns, {_l3_refreshes} L3 refresh(es), l2 lengths seen: "
      f"{sorted(set(_l2_at_boundary))}")


# ===========================================================================
# [4] Visible catch-up: one INFO line per tail while behind; none once
#     caught up (does not alarm on ordinary operation).
# ===========================================================================
print()
print("[4] one INFO catch-up line per tail while behind; none once settled")

reset_all(l1=10, l2=10_000, l3=10_000)
CID_L = "catchup-log"
MSGS_L = build_messages(3 * summarizer.L1_CHUNK_SIZE)
fresh_state(CID_L)


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, r):
        self.lines.append(r.getMessage())


_cap = _Cap()
_lg = logging.getLogger("compactor")
_prev_level = _lg.level
_lg.addHandler(_cap)
_lg.setLevel(logging.DEBUG)
st_l1, _ = run_tail(CID_L, MSGS_L, max_calls=1)  # deliberately under-budget
_behind_log = "\n".join(_cap.lines)
check("hierarchy catch-up in progress" in _behind_log,
      "a turn that leaves the hierarchy still behind logs the catch-up "
      "progress line")
check("turn(s) still uncovered" in _behind_log,
      "the catch-up line names how many turns are still uncovered")

_cap.lines.clear()
for _ in range(10):
    st_l1, _ = run_tail(CID_L, MSGS_L, max_calls=10)
    if not summarizer.needs_rollup(st_l1, st_l1.get("turns_seen", 0)):
        break
_cap.lines.clear()
st_settled, _ = run_tail(CID_L, MSGS_L, max_calls=10)
_settled_log = "\n".join(_cap.lines)
_lg.removeHandler(_cap)
_lg.setLevel(_prev_level)
check("hierarchy catch-up in progress" not in _settled_log,
      "once fully caught up, an ordinary turn prints NO catch-up line "
      "(does not alarm on normal operation)")


# ===========================================================================
# [5] backfill.py: its single maybe_rollup call is bounded the same way,
#     and the live tail finishes what it leaves behind.
# ===========================================================================
print()
print("[5] backfill's one-shot summary rollup is bounded too; the live "
      "tail finishes what it leaves behind")

reset_all(l1=10, l2=10_000, l3=10_000)
CID_BF = "catchup-backfill"
BACKLOG_BF = 6 * summarizer.L1_CHUNK_SIZE
MSGS_BF = build_messages(BACKLOG_BF)
fresh_state(CID_BF)

before_bf = len(LLM_CALLS)
asyncio.run(backfill._run_backfill(CID_BF, MSGS_BF, main.VLLM_URL, main.MODEL_REPO))
calls_bf = len(LLM_CALLS) - before_bf
state_bf = summarizer.load_state(CID_BF)

check(calls_bf <= backfill._TAIL_ROLLUP_MAX_CALLS + summarizer.L1_CHUNK_SIZE,
      f"backfill's rollup spent a bounded number of calls (its own budget "
      f"plus the one-unit overshoot), not however many the whole "
      f"{BACKLOG_BF}-turn backlog needed (got {calls_bf} calls)")
check(0 < state_bf.get("last_summarized_turn", 0) < BACKLOG_BF,
      f"backfill made SOME progress but did NOT fully drain a backlog "
      f"deeper than its budget (watermark={state_bf.get('last_summarized_turn')} "
      f"of {BACKLOG_BF})")
print(f"      backfill: {calls_bf} real call(s), watermark "
      f"{state_bf.get('last_summarized_turn')} of {BACKLOG_BF}")

# The live tail picks up exactly where backfill left off and finishes it —
# the same persisted-watermark convergence proven in [a], now handed off
# from one caller to another.
for _ in range(20):
    st_bf, _ = run_tail(CID_BF, MSGS_BF, max_calls=4)
    if not summarizer.needs_rollup(st_bf, st_bf.get("turns_seen", 0)):
        break
check(st_bf.get("last_summarized_turn") == BACKLOG_BF,
      f"the live tail finished the backlog backfill left behind "
      f"(reached {st_bf.get('last_summarized_turn')} of {BACKLOG_BF})")


# ===========================================================================
# [6] health.py: the hierarchy_lag verdict is evidence-based (the TAIL's own
#     process-local record), NOT poll-to-poll -- so it is stable across
#     however many times /health/full happens to be polled between her
#     turns, distinguishes a converging catch-up from a stuck one, and
#     never hides real lag.
# ===========================================================================
print()
print("[6] health.py: the catch-up verdict is evidence-based, stable "
      "across many polls, never hides real lag")


def _clear_all_conv_state() -> None:
    """hostile pass #5 (E6): this section's checks depend on the health
    scan seeing ONLY the conversation(s) each check sets up -- e.g. "a
    converging catch-up does not fire the actionable reason" assumes
    nothing else on disk is a DIFFERENT, non-converging conversation over
    the lag limit. That was true by ACCIDENT before this fix, because the
    scan only ever named the single largest-lag conversation (E6's own
    defect: a smaller stuck lag could never surface once anything larger
    existed). Now every over-limit recent conversation is judged, so a
    leftover conv_id from an EARLIER section above (its state file still
    on disk, mtime still "recent") with no catch-up evidence would read
    'unknown' and correctly fire its own reason -- correct behaviour, but
    it would contaminate an isolation test that never meant to exercise
    it. Clearing every summary state file before this section starts (the
    same technique test_health_findings.py's own _reset_all uses) keeps
    this section about the ONE conversation it is actually testing.
    """
    d = memory.storage_root() / "summaries"
    if d.exists():
        for f in d.glob("*.json"):
            f.unlink()


_clear_all_conv_state()


async def _poll_health() -> dict:
    return await health.gather_health_full(main.VLLM_URL, 4000)


def _catching_up(payload: dict) -> dict | None:
    return (payload.get("checks") or {}).get("hierarchy", {}).get("catching_up")


def _lag_reason(payload: dict) -> str:
    return next(
        (r for r in (payload.get("status_reasons") or []) if "turns behind" in r), ""
    )


# ---- unit-level: summarizer's own record/read pair --------------------
summarizer._reset_catchup_progress_for_tests()
check(summarizer.catchup_progress_for("nobody-yet") is None,
      "fixture: a conv_id with no recorded pass has no evidence")
check(health._catchup_verdict("nobody-yet") == "unknown",
      "no evidence at all -> unknown (the restart-safe default)")
check(health._catchup_verdict(None) == "unknown",
      "no worst-lag conv at all -> unknown")

summarizer.record_catchup_pass("conv-u", 0, 20, True)  # advanced, more due
check(health._catchup_verdict("conv-u") == "converging",
      "a pass that just advanced the watermark: converging")

# Age that advance OUT of the "recent" window directly, rather than a real
# 15-minute sleep: this is exactly what "the advance happened a while ago"
# looks like from _catchup_verdict's own perspective (it reads the SAME
# last_advance_monotonic field either way), not a different code path or a
# fixture shortcut.
summarizer._catchup_progress["conv-u"]["last_advance_monotonic"] -= (
    health._CATCHUP_RECENT_S + 1
)
check(health._catchup_verdict("conv-u") == "unknown",
      "the SAME advance, once older than the converging window: no longer "
      "converging, and not yet 'stuck' either (0 no-advance passes "
      "recorded since it) -- the safe actionable default in between")

for _ in range(health.HIERARCHY_STALL_DECISIONS - 1):
    summarizer.record_catchup_pass("conv-u", 20, 20, True)  # due, no advance
check(health._catchup_verdict("conv-u") == "unknown",
      f"fewer than HIERARCHY_STALL_DECISIONS "
      f"({health.HIERARCHY_STALL_DECISIONS}) no-advance passes since the "
      f"aged-out advance: not yet 'stuck'")
summarizer.record_catchup_pass("conv-u", 20, 20, True)  # the Nth: stuck
check(health._catchup_verdict("conv-u") == "stuck",
      f"{health.HIERARCHY_STALL_DECISIONS} consecutive no-advance passes "
      f"with work due: stuck")
summarizer.record_catchup_pass("conv-u", 20, 40, True)  # advances again
check(health._catchup_verdict("conv-u") == "converging",
      "a later advance resets the stall count -- converging again, not "
      "still 'stuck' from before")
summarizer._reset_catchup_progress_for_tests()

# ---- MANY health polls per turn: the verdict must not flap -------------
# The regression this whole section exists for: the ORIGINAL implementation
# compared hierarchy_lag_recent poll to poll, so of two consecutive polls
# with NOTHING happening between them, the FIRST poll after a real advance
# read "converging" and every one after it (lag unchanged from the last
# poll, not shrinking further) flipped back to the actionable wording --
# poll-cadence dependent, not evidence dependent. This drives MANY real
# health polls with nothing else happening in between and asserts every
# poll gives the SAME verdict, both before evidence exists and right after
# a real tail advance -- proving the fix is poll-cadence independent, not
# just poll-once-per-turn green (which is exactly what let the original bug
# ship).
reset_all(l1=5, l2=10_000, l3=10_000)
CID_H = "catchup-health"
MSGS_H = build_messages(20 * summarizer.L1_CHUNK_SIZE)
# Seeded directly on disk (turns_seen far past last_summarized_turn), NOT
# via a real tail pass -- this conv is behind, but no BUDGETED pass has run
# for it in this process, so summarizer has no catchup-progress evidence
# for it at all yet. Matches "just noticed behind" (the scan finding it)
# more than "a tail already tried and made no progress", and is the
# cleanest way to get a stable pre-evidence baseline without a fresh advance
# from run_tail muddying it (a fresh conv's FIRST tail pass always advances
# from 0, which would already read as converging).
summarizer.save_state(CID_H, {
    "l1": [], "l2": [], "l3": None,
    "last_summarized_turn": 0, "turns_seen": 100,
})
health._reset_hierarchy_progress_for_tests()
summarizer._reset_catchup_progress_for_tests()

_polls_no_evidence = [asyncio.run(_poll_health()) for _ in range(10)]
_verdicts_1 = [(_catching_up(p) or {}).get("verdict") for p in _polls_no_evidence]
check(len(set(_verdicts_1)) == 1 and _verdicts_1[0] == "unknown",
      f"10 health polls against a conv that is behind but has no "
      f"catch-up evidence yet: all 10 read 'unknown', stable "
      f"(got {_verdicts_1})")
check(all(_lag_reason(p) for p in _polls_no_evidence),
      "and the lag reason fires on every one of those 10 polls -- not "
      "hidden by however many times it is asked")

run_tail(CID_H, MSGS_H, max_calls=2)  # a real tail pass: advances the watermark
_polls_after_advance = [asyncio.run(_poll_health()) for _ in range(10)]
_verdicts_2 = [
    (_catching_up(p) or {}).get("verdict") for p in _polls_after_advance
]
check(len(set(_verdicts_2)) == 1 and _verdicts_2[0] == "converging",
      f"10 health polls right after a tail pass that advanced the "
      f"watermark: all 10 read 'converging', not just the first poll "
      f"(got {_verdicts_2}) -- the exact scenario that flapped before "
      f"this fix")
check(all(not _lag_reason(p) for p in _polls_after_advance),
      "and none of those 10 polls carry the actionable /compact reason "
      "-- status is not degraded for a catch-up that is provably "
      "self-healing right now")

# ---- stuck case: passes with work due, never advancing ------------------
summarizer._reset_catchup_progress_for_tests()
for _ in range(health.HIERARCHY_STALL_DECISIONS):
    summarizer.record_catchup_pass(CID_H, 100, 100, True)
_stuck_payload = asyncio.run(_poll_health())
check((_catching_up(_stuck_payload) or {}).get("verdict") == "stuck",
      f"after {health.HIERARCHY_STALL_DECISIONS} no-advance passes with "
      f"work due, this conv's catch-up reads as stuck")
check(bool(_lag_reason(_stuck_payload)),
      "and the actionable /compact reason fires for a genuinely stuck "
      "catch-up")

# ---- converging case: does not alarm -------------------------------------
summarizer._reset_catchup_progress_for_tests()
summarizer.record_catchup_pass(CID_H, 100, 120, True)
_converging_payload = asyncio.run(_poll_health())
check((_catching_up(_converging_payload) or {}).get("verdict") == "converging",
      "a just-advanced watermark reads as converging")
check(not _lag_reason(_converging_payload),
      "and does NOT fire the actionable reason -- a self-healing "
      "catch-up is not an alarm")

# ---- restart behaviour: lost evidence defaults to the SAFE (actionable)
# reading, never to "converging" -----------------------------------------
summarizer._reset_catchup_progress_for_tests()  # simulates a fresh process
_restart_payload = asyncio.run(_poll_health())
check((_catching_up(_restart_payload) or {}).get("verdict") == "unknown",
      "right after a (simulated) restart, no evidence yet: unknown")
check(bool(_lag_reason(_restart_payload)),
      "unknown degrades status the SAME way 'stuck' does -- a restart "
      "must never let a genuinely stuck backlog read as self-healing "
      "just because nothing has been observed about it yet in the new "
      "process (does not hide a stuck lag forever)")
# hostile pass #5 (C5-7): the text is no longer identical, ON PURPOSE.
# "Unknown" is NOT evidence of a stall (most often a recent restart with
# no catch-up passes recorded yet) and must not assert one as fact the
# way the old shared wording did; "stuck" is real evidence, so it still
# gets the actionable /compact wording (or, when a tier is known to be
# failing, names that tier instead -- see health._catchup_reason). Both
# still DEGRADE STATUS identically (checked above and by [F3a]/[F3b] in
# test_health_findings.py) -- what changed is the wording, not whether it
# alarms.
check(_lag_reason(_restart_payload) != _lag_reason(_stuck_payload),
      "unknown and stuck now fire DIFFERENT reason text -- 'unknown' says "
      "there is no evidence yet rather than asserting a stall that was "
      "never observed, while 'stuck' keeps the actionable wording (or "
      "names the failing tier)")
check("No catch-up evidence" in _lag_reason(_restart_payload),
      f"the unknown reason says plainly that nothing has been observed "
      f"yet, rather than reusing 'stuck''s wording (got "
      f"{_lag_reason(_restart_payload)!r})")
check("/compact" in _lag_reason(_stuck_payload),
      f"the stuck reason (no tier failure on record for this conv) still "
      f"names /compact as the remedy (got {_lag_reason(_stuck_payload)!r})")

health._reset_hierarchy_progress_for_tests()
summarizer._reset_catchup_progress_for_tests()


# ===========================================================================
# [7] hostile follow-up: the catch-up INFO line cannot raise INTO the tail,
#     even when maybe_rollup returns a state shaped unexpectedly (missing
#     turns_seen, or a non-int value) -- the tail must have work to do on
#     the NEXT turn regardless.
# ===========================================================================
print()
print("[7] a malformed state from maybe_rollup cannot crash the tail; the "
      "NEXT turn still works")

reset_all(l1=5, l2=10_000, l3=10_000)
CID_M = "catchup-malformed"
# Deep enough that ONE budgeted pass (max_calls=4 below, cost-1 chunks) does
# NOT fully drain it -- genuine work must remain for "the next turn" to do,
# which is the actual property under test ("the tail after it still has
# work to do" — a malformed pass that happened to finish the whole backlog
# would prove nothing about whether a LATER turn still functions).
MSGS_M = build_messages(10 * summarizer.L1_CHUNK_SIZE)
fresh_state(CID_M)

_real_maybe_rollup = summarizer.maybe_rollup


async def _malformed_maybe_rollup(*a, **kw):
    st = await _real_maybe_rollup(*a, **kw)
    # A real, saved state -- maybe_rollup's own contract already ran and
    # persisted it -- just handed back to the caller missing/wrong-typed
    # the two fields the catch-up log line reads. This is exactly the
    # shape a future change three call-levels away could produce without
    # meaning to.
    _bad = dict(st)
    _bad.pop("turns_seen", None)
    _bad["last_summarized_turn"] = "not-an-int"
    return _bad


summarizer.maybe_rollup = _malformed_maybe_rollup
try:
    _raised = False
    try:
        run_tail(CID_M, MSGS_M, max_calls=4)
    except Exception:
        _raised = True
finally:
    summarizer.maybe_rollup = _real_maybe_rollup

check(not _raised,
      "a state missing turns_seen / with a non-int last_summarized_turn "
      "did not raise out of main._rollup_hierarchy")

# The tail after it still has work to do: a NORMAL subsequent turn (real
# maybe_rollup restored above) must still work -- nothing about the
# malformed pass left the module, the lock, or the conv's own state
# unusable for the next one.
_st_after, _calls_after = run_tail(CID_M, MSGS_M, max_calls=4)
check(_calls_after > 0 and _st_after.get("last_summarized_turn", 0) > 0,
      f"the NEXT turn (real maybe_rollup) makes real progress -- the tail "
      f"was not left broken by the malformed pass before it "
      f"(calls={_calls_after}, watermark={_st_after.get('last_summarized_turn')})")


# ===========================================================================
if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll tail catch-up checks passed.")
