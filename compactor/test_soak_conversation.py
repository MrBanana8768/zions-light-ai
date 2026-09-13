"""
End-to-end soak: a conversation that GROWS, against a real token-counting server.

## Why this exists

Every test in this repo before it checks a component. Not one drives a
conversation from turn 1 to turn N and asks whether the thing still works — and
that is exactly the shape of every failure this project has had:

  - 2026-08-28: the counter under-read, so the summarizer built over-budget
    batches, so compaction 400'd, so the guard shed 80 turns per request. No
    single component was broken. The COMPOSITION was, and only after a
    conversation got long enough.
  - 2026-08-29: /tokenize refused an assistant-final list. Same cascade, new
    trigger, found in production rather than in CI.

A component test cannot see either. Both need a conversation with a history
long enough to compact, replies expensive enough to matter, and a server that
charges real tokens and refuses what does not fit.

## What makes it adversarial rather than merely long

The fixture generates replies containing the four things that actually broke
production, not lorem alone:

  - box-drawing rules (U+2501) — where local and server tokenizers diverge most
  - emoji — the same, by a different route
  - markdown headings and code fences — what fact extraction stored as memory
  - fabricated numeric status lines — the loop where the model reads its own
    invented figures back as fact

Prose is where tokenizers agree. A soak written on prose alone would pass while
production burned, which is the mistake the contract harness documents.

## Running it

    docker compose -f docker-compose.tokenizer-contract.yml up -d tokenize-fixture
    VLLM_URL=http://localhost:18000 MODEL_REPO=fixture-model \\
      python test_soak_conversation.py

With no fixture reachable it SKIPS and exits 3 (the runner's SKIP code; see
_skip). The ORACLE SELF-TEST runs first, before the fixture is probed, so even
a run that skips has proved every assertion below can still say no:

    SOAK_ORACLE_SELFTEST_ONLY=1 python test_soak_conversation.py

## Three phases, and why one is not enough

  1. STORE   — prose replies (reply_looping=False). The memory tail must
               actually store: every turn is a `stored` decision and the tail
               itself writes facts.
  2. REFUSE  — repetition-loop replies (reply_looping=True). Every turn is
               refused, and the hierarchy must keep rolling up anyway
               (rollup-on-skip, v3.1.8).
  3. STORE   — prose again. The tail must come back after a refusal streak.

Until v3.1.9 the whole run was phase 2. The tail stored nothing on 22 of 22
turns, the facts the soak asserted on came from the BACKFILL, and 21 of 22
assistant turns reached the rollup as redaction placeholders — and it all
printed `ok`.

## What it does NOT prove

The fixture's tokenizer is not Cydonia's, so absolute token numbers mean
nothing for production. What it proves is that the SYSTEM holds together over a
growing conversation against a server that charges real tokens — the property
no other test in this repo asserts.
"""

import os
import sys
import tempfile

# --- Environment must be set before main is imported ------------------------
os.environ.setdefault("VLLM_URL", "http://localhost:18000")
os.environ.setdefault("MODEL_REPO", "fixture-model")
# A small window so compaction fires within a soak-sized run instead of after
# hundreds of turns. The RATIOS are what this test asserts, not the absolutes.
# Forced, not setdefault: the production image BAKES MAX_MODEL_LEN=32768 as an
# ENV, so setdefault silently loses and the soak would run against a window
# four times too large — compaction would never fire and the run would assert
# nothing while reporting success. Override with SOAK_MAX_MODEL_LEN.
os.environ["MAX_MODEL_LEN"] = os.environ.get("SOAK_MAX_MODEL_LEN", "8192")
os.environ["COMPACTOR_GENERATION_RESERVE"] = os.environ.get(
    "SOAK_GENERATION_RESERVE", "2048")
# No network on the budgeting path.
#
# The first real soak run blocked for 231 SECONDS inside get_tokenizer, which
# tried to resolve MODEL_REPO against huggingface.co and hit a 429. The fixture
# connection dropped while it was stalled and the run failed at turn 10 for an
# entirely environmental reason.
#
# That is REMEDIATION F33 (the tokenizer load is retried per request and its
# failure is indistinguishable from not-yet-loaded) composing with F25 (nothing
# sets HF_HUB_OFFLINE even when every weight is cached). Latent in production
# only because the model IS cached there. Setting these keeps the soak
# measuring the compactor rather than huggingface.co's rate limiter.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_STORE = tempfile.mkdtemp(prefix="soak-store-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _STORE

# THE L2 AND L3 FAN-OUT, forced small (v3.1.9). L1_CHUNK_SIZE is left at the
# production 20 on purpose: it is the one that sets chunk boundaries, the lag
# and the coverage slack, and it is what compaction's reuse path reads. L2 and
# L3 are COUNTS of lower-tier entries and nothing else — at production values
# (10 and 5) the second tier needs 100 turns and the third 500, so the soak
# never built either, and `_summary_coverage_gaps`'s l2 loop and its entire l3
# branch never executed while the soak printed a coverage claim naming all
# three tiers. At 2 and 2 the same code builds an L2 chapter every 20 turns
# and refreshes L3 every 40, which a 112-turn run reaches twice — the second
# refresh is the one that folds a PRIOR L3 back in (the two-stage path). This
# module's own docstring measured L3 behaviour at L1=4/L2=3/L3=2 for the same
# reason. Forced, not setdefault, for MAX_MODEL_LEN's reason. _tier_plan below
# FAILS the soak if the configured turns cannot build every tier.
os.environ["COMPACTOR_L2_CHUNK_SIZE"] = os.environ.get("SOAK_L2_CHUNK_SIZE", "2")
os.environ["COMPACTOR_L3_CHUNK_SIZE"] = os.environ.get("SOAK_L3_CHUNK_SIZE", "2")

import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from unittest.mock import patch  # noqa: E402

import httpx  # noqa: E402

FIXTURE_URL = os.environ.get("FIXTURE_URL", os.environ["VLLM_URL"]).rstrip("/")
# ONE default, and it is the compose file's. The module said 40 while
# docker-compose.tokenizer-contract.yml said 22, and the override silently put
# the lag guard out of reach (A1). The two now agree, and _lag_guard_unreachable
# / _tier_plan fail the run if either is ever overridden into vacuity again.
TURNS = int(os.environ.get("SOAK_TURNS", "112"))
REPLY_CHARS = int(os.environ.get("SOAK_REPLY_CHARS", "1400"))
# Phase lengths. Turns 1..STORE_TURNS store, the next REFUSE_TURNS refuse, the
# rest store again. See the module docstring and _phase_problems.
STORE_TURNS = int(os.environ.get("SOAK_STORE_TURNS", "30"))
REFUSE_TURNS = int(os.environ.get("SOAK_REFUSE_TURNS", "40"))


def _phase_of(n: int) -> str:
    if n <= STORE_TURNS:
        return "store-1"
    if n <= STORE_TURNS + REFUSE_TURNS:
        return "refuse"
    return "store-2"


# ===========================================================================
# THE ORACLES — pure functions, no product import, no fixture.
#
# Every end-of-run claim this soak prints is decided by one of these, and
# every one of them is exercised by _oracle_selftest() on constructed rows and
# states BEFORE the fixture is even probed. Three of this file's assertions
# shipped unable to fail in the configuration the stack runs (hostile pass #2,
# items A1-A3), and a soak is the one suite whose oracle cannot be
# mutation-tested by breaking the product: the product is the whole stack. So
# the oracle is tested here instead, against the exact defect shapes, with a
# CONTROL for each proving it can still say yes.
#
# Why in this file and not a separate test: the soak must never run with an
# oracle that cannot fail. A separate file can be skipped, forgotten, or left
# out of the runner's discovery (scripts/run-tests.py only discovers
# compactor/test_*.py, and puts this file in SATURATION); code at the top of
# the soak cannot be separated from it. And because it runs before the fixture
# probe, the unit stack — which SKIPs this file for want of a fixture — still
# proves the oracle on every run.
# ===========================================================================

def _lag_budget(l1_chunk_size: int) -> int:
    """How far `history - watermark` may legitimately reach, in message-units.

    ONE L1 chunk. It used to be `L1_CHUNK_SIZE*2 + KEEP_RECENT_TURNS*2` = 48,
    which no 22-turn run can exceed (A1). The tight bound is a property of the
    drain, not a guess: maybe_rollup loops `_do_l1_rollup` while
    `position - watermark >= L1_CHUNK_SIZE`, so after any completed tail the
    lag is at most L1_CHUNK_SIZE-1 against the messages the rollup saw. The
    skip path rolls up WITHOUT the refused reply (2n-1 units at turn n) and
    `rows` records history before the reply is appended (also 2n-1), so the
    worst healthy lag is 19 — exactly what the shipped soak measured ("worst
    lag 19 of 48"). KEEP_RECENT_TURNS never entered it: it governs what the
    GUARD forwards, not what the summarizer must have consumed.
    """
    return l1_chunk_size


def _lag_guard_unreachable(turns: int, l1_chunk_size: int, budget: int) -> list[str]:
    """Problems if the lag guard could not fire at this turn count.

    Not "could a watermark frozen at 0 trip it" — the separate `watermark <= 0`
    check already catches TOTAL death. The shape that got through was ONE
    rollup and then death, so that is the shape this requires the run to be
    long enough to see: `rows` records history 2n-1, the first rollup pins the
    watermark at L1_CHUNK_SIZE, so the largest lag that shape can show is
    (2*turns - 1) - L1_CHUNK_SIZE.
    """
    worst = (2 * turns - 1) - l1_chunk_size
    if worst <= budget:
        return [
            f"SOAK_TURNS={turns} caps the lag a hierarchy that rolled up ONCE "
            f"and then died can show at {worst}, at or below the budget of "
            f"{budget}: the lag guard cannot fire. Need SOAK_TURNS > "
            f"{(budget + l1_chunk_size + 1) // 2}."
        ]
    return []


def _lagging_rows(rows: list[dict], budget: int) -> list[dict]:
    return [r for r in rows if (r["history"] - r["watermark"]) > budget]


def _summary_coverage_gaps(st: dict, conversation_turns: int, slack: int) -> list[str]:
    """Every problem with how the summary hierarchy represents this
    conversation. [] means every turn old enough to have been summarised is
    covered by exactly one real, non-empty entry, contiguously from turn 1.

    `conversation_turns` is the SOAK's count of non-system message-units it
    has shown the compactor — deliberately not the state's own `turns_seen`,
    which is a product-written number and the kind of thing that freezes.
    `slack` is how many trailing turns may legitimately be unsummarised: one
    L1 chunk (see _lag_budget for why that is exact).

    WHAT THE SHIPPED VERSION COULD NOT SEE (A2), both proven below:
      * turns STRANDED ABOVE A FROZEN WATERMARK. It returned [] on
        `watermark <= 0` and its only completeness test was
        `covered < watermark`, so everything the watermark had not passed was
        outside its domain — and a watermark that stops while the conversation
        grows is exactly the 2026-08-28 incident its own comment cites. Now:
        the other end of the range is the conversation, not the watermark.
      * an IMPOSSIBLE SPAN. `covered = max(covered, lt)` trusted each chunk's
        own last_turn, so one chunk claiming 1-999999 satisfied every later
        comparison for free. Now: a span must lie inside the conversation to
        count as coverage at all, and spans must TILE — a gap and an overlap
        are both reported, so `covered` can only ever be the end of a chain
        this function walked link by link.

    What stays from the shipped version, because it was right: l3 counts (an
    L3 refresh consumes and clears l2, so after one every turn behind it lives
    in l3 alone), and an entry with empty text covers nothing.

    Never raises on a type-wrong state: a traceback would replace the one
    line that says which invariant broke.
    """
    def _int(v):
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    problems: list[str] = []
    watermark = _int(st.get("last_summarized_turn")) or 0
    seen = _int(st.get("turns_seen")) or 0
    if seen > conversation_turns:
        problems.append(
            f"turns_seen={seen} but the conversation has only "
            f"{conversation_turns} message-units — the compactor believes it "
            f"has seen turns that were never sent"
        )

    spans: list[tuple[int, int, str]] = []

    def _take(tier: str, c) -> None:
        if not isinstance(c, dict):
            problems.append(f"{tier} holds a non-object entry: {c!r:.80}")
            return
        ft, lt = _int(c.get("first_turn")), _int(c.get("last_turn"))
        text = c.get("text")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{tier} entry {ft}-{lt} has EMPTY text")
            return
        if ft is None or lt is None:
            problems.append(f"{tier} entry has no integer span "
                            f"({c.get('first_turn')!r}-{c.get('last_turn')!r})")
            return
        if ft < 1 or lt < ft:
            problems.append(f"{tier} entry claims the impossible span {ft}-{lt}")
            return
        if lt > conversation_turns:
            problems.append(
                f"{tier} entry claims turns {ft}-{lt} but the conversation "
                f"has only {conversation_turns} — a span past the end of the "
                f"conversation is not coverage, and is not counted as any"
            )
            return
        spans.append((ft, lt, tier))

    _l3 = st.get("l3")
    if _l3 is not None:
        _take("l3", _l3)
    for tier in ("l1", "l2"):
        entries = st.get(tier) or []
        if not isinstance(entries, list):
            problems.append(f"{tier} is a {type(entries).__name__}, not a list")
            continue
        for c in entries:
            _take(tier, c)

    covered = 0
    for ft, lt, tier in sorted(spans):
        if ft > covered + 1:
            problems.append(
                f"gap in coverage: turns {covered + 1}-{ft - 1} are claimed "
                f"by nothing (l1, l2 or l3)"
            )
        elif ft <= covered:
            problems.append(
                f"overlap: {tier} entry {ft}-{lt} re-claims turns "
                f"{ft}-{min(lt, covered)} that an earlier entry already covers"
            )
        covered = max(covered, lt)

    if covered < watermark:
        problems.append(
            f"coverage stops at turn {covered} but last_summarized_turn "
            f"says {watermark} — {watermark - covered} turn(s) were passed "
            f"by the watermark and represented nowhere"
        )
    if conversation_turns - covered > slack:
        problems.append(
            f"turns {covered + 1}-{conversation_turns} "
            f"({conversation_turns - covered} message-units) are old enough to "
            f"have been summarised (slack {slack}) and no entry covers them — "
            f"the hierarchy stopped while the conversation grew, which is "
            f"2026-08-28 (watermark {watermark})"
        )
    return problems


def _phase_problems(rows: list[dict], final_fact_count: int) -> list[str]:
    """Did the run exercise BOTH sides of the memory tail? (A3)

    Each row is one turn: `phase`, `stored`/`skipped` (tailhealth deltas —
    the counter /health/full publishes, one decision per exchange),
    `tail_new_facts` (fact texts that appeared DURING the tail's own
    _facts_tail call, so the backfill — an independent writer that fills an
    empty store once — cannot be mistaken for it), and `watermark_advance`.

    The shipped soak asserted only `final_fact_count > 0`, which the backfill
    satisfied on a run where the tail refused 22 of 22 turns.
    """
    problems: list[str] = []
    if final_fact_count <= 0:
        problems.append("no facts were stored across the whole run")
    phases: dict[str, list[dict]] = {}
    for r in rows:
        phases.setdefault(r["phase"], []).append(r)
    if not any(p.startswith("store") for p in phases):
        problems.append("the run had no STORE phase: the storing path never ran")
    if "refuse" not in phases:
        problems.append("the run had no REFUSE phase: rollup-on-skip never ran")
    for r in rows:
        if r["stored"] + r["skipped"] != 1:
            problems.append(
                f"turn {r['turn']}: {r['stored']} stored + {r['skipped']} "
                f"skipped tail decisions — exactly one per exchange is the "
                f"contract, and without it no count below means anything"
            )
    for name, prs in phases.items():
        n = len(prs)
        stored = sum(r["stored"] for r in prs)
        if name.startswith("store"):
            refused = [r["turn"] for r in prs if r["skipped"]]
            if refused:
                problems.append(
                    f"{name}: the tail refused {len(refused)} of {n} prose "
                    f"turns (turns {refused[:8]}) — the storing path is not "
                    f"what this phase exercised"
                )
            if stored and sum(r["tail_new_facts"] for r in prs) <= 0:
                problems.append(
                    f"{name}: {stored} exchange(s) were counted `stored` and "
                    f"the tail's own fact extraction wrote nothing — the "
                    f"decision says stored, the store says otherwise"
                )
        else:
            if stored:
                problems.append(
                    f"{name}: {stored} of {n} repetition-loop replies were "
                    f"STORED — the refusal this phase exists to exercise did "
                    f"not happen"
                )
            skipped_turns = [r for r in prs if r["skipped"]]
            if skipped_turns and not any(r["watermark_advance"] > 0
                                         for r in skipped_turns):
                problems.append(
                    f"{name}: {len(skipped_turns)} consecutive refused turns "
                    f"and the watermark never advanced on one of them — "
                    f"rollup-on-skip is dead, which is the v3.1.8 frozen "
                    f"hierarchy"
                )
    return problems


def _tier_plan(turns: int, l1: int, l2: int, l3: int) -> dict:
    """What a healthy hierarchy is GUARANTEED to have built by the last turn.

    2*turns - 1 units: what the last turn's rollup is guaranteed to see (the
    skip path excludes the refused reply). The drain is exact: L1 while a
    chunk's worth is pending, L2 while L2_CHUNK_SIZE chunks exist, L3 once
    L3_CHUNK_SIZE chapters exist (consuming them).
    """
    units = 2 * turns - 1
    chunks = units // l1
    chapters = chunks // l2
    return {
        "l1_chunks": chunks,
        "l2_chapters": chapters,
        "l3_refreshes": chapters // l3,
        "final_l1": chunks % l2,
        "final_l2": chapters % l3,
    }


def _tier_reach_problems(plan: dict) -> list[str]:
    """Problems if the configuration cannot build every tier the coverage
    claim names. Two L3 refreshes, because only the SECOND folds a prior L3
    back in; and a non-empty l1 and l2 at the end, so the final coverage walk
    has to tile all three tiers at once."""
    problems = []
    if plan["l3_refreshes"] < 2:
        problems.append(
            f"the configuration reaches {plan['l3_refreshes']} L3 refresh(es) "
            f"(need 2: the second is the one that folds a prior L3 back in)")
    if plan["final_l2"] < 1:
        problems.append("the run ends with l2 empty, so the final coverage "
                        "walk never tiles l3 against l2")
    if plan["final_l1"] < 1:
        problems.append("the run ends with l1 empty, so the final coverage "
                        "walk never tiles l2 against l1")
    return problems


def _tier_observed_problems(plan: dict, observed: dict) -> list[str]:
    """Problems if the run built LESS than a healthy hierarchy must have.
    Checked against the arithmetic, not against 'something exists': a
    hierarchy that stops at L1 while the plan says two L3 refreshes is the
    dead-upper-tier shape, and 'L1>=1' reads it as advanced."""
    problems = []
    for key, what in (("l1_chunks", "L1 chunks written"),
                      ("l2_chapters", "L2 chapters written"),
                      ("l3_refreshes", "L3 refreshes")):
        if observed.get(key, 0) < plan[key]:
            problems.append(f"{what}: {observed.get(key, 0)}, but a healthy "
                            f"hierarchy must have reached {plan[key]}")
    return problems


def _rollup_input_problems(l1_inputs: list[tuple[int, int]]) -> list[str]:
    """`l1_inputs` is (placeholder pieces, total pieces) for every L1
    summarization call the run made. Problems if NO chunk summarised real
    text: then 'nothing vanished' is a claim about redaction placeholders
    (A3 — 21 of 22 assistant turns were placeholders in the shipped run)."""
    if not l1_inputs:
        return ["the run made no L1 summarization call at all"]
    if not any(ph == 0 for ph, _ in l1_inputs):
        return [
            f"every one of the {len(l1_inputs)} L1 chunk(s) summarised "
            f"redaction placeholders ({sum(ph for ph, _ in l1_inputs)} of "
            f"{sum(t for _, t in l1_inputs)} pieces) — no chunk in this run "
            f"summarised a real reply"
        ]
    return []


def _oracle_selftest() -> tuple[list[str], int]:
    """Each oracle against the defect shape it exists for (must go RED) and a
    healthy control (must stay GREEN). Returns the cases that came out wrong.
    """
    failures: list[str] = []
    cases = [0]

    def expect(name: str, oracle, want_red: bool) -> None:
        # `oracle` is a zero-argument callable, not its result, so an oracle
        # that RAISES is recorded as this case failing, by name, instead of
        # killing the self-test with a traceback that names nothing.
        cases[0] += 1
        try:
            problems = oracle()
        except Exception as e:
            failures.append(f"{name}: the oracle RAISED {type(e).__name__}: {e}")
            return
        red = bool(problems)
        if red != want_red:
            failures.append(
                f"{name}: expected {'RED' if want_red else 'GREEN'}, got "
                f"{'RED ' + repr(problems) if red else 'GREEN'}")

    L1, SLACK = 20, 20

    # ---- A1: the lag guard ------------------------------------------------
    # The shipped shape: 22 turns, one rollup at turn 11, then the hierarchy
    # dies. rows record history 2n-1 and the post-tail watermark.
    frozen = [{"turn": n, "history": 2 * n - 1,
               "watermark": 20 if n >= 11 else 0} for n in range(3, 23)]
    healthy = [{"turn": n, "history": 2 * n - 1,
                "watermark": ((2 * n - 1) // 20) * 20} for n in range(3, 23)]
    expect("A1 lag: watermark frozen after one rollup, 22 turns",
           lambda: _lagging_rows(frozen, _lag_budget(L1)), True)
    expect("A1 lag CONTROL: watermark advancing every chunk",
           lambda: _lagging_rows(healthy, _lag_budget(L1)), False)
    # THE ONE STATE WHERE THE LAG CHECK DECIDES ALONE. The per-turn coverage
    # check runs first and catches every frozen-hierarchy shape before the
    # end-of-run lag check is reached, so on the real stack the lag check can
    # only ever be the one that fires when the WATERMARK is stuck behind
    # chunks that already tile further — the pre-v3.1.4 watermark reset
    # (R12, _repair_watermark_below_chunks). Coverage is green on that state
    # (covered 40 of 44, watermark 20 < covered), so these two cases together
    # prove the lag check is not dead weight behind it.
    stuck_behind_chunks = {"last_summarized_turn": 20,
                           "l1": [{"text": "s", "first_turn": 1, "last_turn": 20},
                                  {"text": "s", "first_turn": 21, "last_turn": 40}],
                           "turns_seen": 44}
    expect("A1 lag decides alone: coverage CONTROL is green on the stuck-watermark state",
           lambda: _summary_coverage_gaps(stuck_behind_chunks, 44, L1), False)
    expect("A1 lag decides alone: watermark stuck at 20 behind chunks tiling to 40",
           lambda: _lagging_rows([{"turn": 22, "history": 43, "watermark": 20}],
                                 _lag_budget(L1)), True)
    expect("A1 reach: SOAK_TURNS=22 against the shipped budget of 48",
           lambda: _lag_guard_unreachable(22, L1, 48), True)
    expect("A1 reach: SOAK_TURNS=20 against a budget of 20",
           lambda: _lag_guard_unreachable(20, L1, _lag_budget(L1)), True)
    expect("A1 reach CONTROL: SOAK_TURNS=112 against a budget of 20",
           lambda: _lag_guard_unreachable(112, L1, _lag_budget(L1)), False)

    # ---- A2: the coverage oracle -----------------------------------------
    def ch(ft, lt, text="a real summary of that stretch"):
        return {"text": text, "first_turn": ft, "last_turn": lt}

    expect("A2 stranded above a frozen watermark (one rollup, then death)",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 20, "l1": [ch(1, 20)],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 watermark frozen at 0 on a 44-unit conversation",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 0, "l1": [],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 bogus last_turn 999999",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 999999)], "turns_seen": 44},
                                          44, SLACK), True)
    # Past this point each RED case is built so that ONE condition decides
    # it, and soak-mutate.py drops each condition in turn to prove it.
    expect("A2 bogus span after a real tiling (1-20, 21-40, 41-999999)",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20), ch(21, 40),
                                                  ch(41, 999999)],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 backwards span (45-41) after a real tiling",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20), ch(21, 40),
                                                  ch(45, 41)],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 non-integer span after a real tiling",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20), ch(21, 40),
                                                  ch("41", 44)],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 watermark past coverage, nothing stranded (alone)",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20)], "turns_seen": 38},
                                          38, SLACK), True)
    expect("A2 empty text on the pending chunk, nothing else wrong (alone)",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 20,
                                           "l1": [ch(1, 20), ch(21, 40, "")],
                                           "turns_seen": 40}, 40, SLACK), True)
    expect("A2 overlapping spans (1-20, 11-40)",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20), ch(11, 40)],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 turns_seen past the conversation",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20), ch(21, 40)],
                                           "turns_seen": 90}, 44, SLACK), True)
    # The four positive controls the hostile pass confirmed on the shipped
    # oracle — kept, so the rewrite cannot lose what already worked.
    expect("A2 kept: watermark 40 past a single 1-20 chunk",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40, "l1": [ch(1, 20)],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 kept: interior hole 21-30",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 60,
                                           "l1": [ch(1, 20), ch(31, 60)],
                                           "turns_seen": 64}, 64, SLACK), True)
    expect("A2 kept: empty chunk text",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20), ch(21, 40, "")],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 kept: whitespace chunk text",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20), ch(21, 40, "  \n")],
                                           "turns_seen": 44}, 44, SLACK), True)
    expect("A2 type-wrong state does not raise and is reported (alone)",
           lambda: _summary_coverage_gaps({"last_summarized_turn": "40",
                                           "l1": {"c1": ch(1, 40)}}, 20, SLACK),
           True)
    expect("A2 a non-object entry in l1 (alone)",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 20,
                                           "l1": [ch(1, 20), "junk"],
                                           "turns_seen": 24}, 24, SLACK), True)
    # Controls — the healthy shapes a real run produces at each tier.
    expect("A2 CONTROL: two contiguous L1 chunks, 4 units pending",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 40,
                                           "l1": [ch(1, 20), ch(21, 40)],
                                           "turns_seen": 44}, 44, SLACK), False)
    expect("A2 CONTROL: skip-path boundary, exactly one chunk pending",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 20,
                                           "l1": [ch(1, 20)], "turns_seen": 39},
                                          40, SLACK), False)
    expect("A2 CONTROL: young conversation, nothing summarised yet",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 0, "l1": [],
                                           "turns_seen": 20}, 20, SLACK), False)
    expect("A2 CONTROL: post-L3 shape, l3 + l2 + l1 tiling",
           lambda: _summary_coverage_gaps({"last_summarized_turn": 220,
                                           "l3": ch(1, 160), "l2": [ch(161, 200)],
                                           "l1": [ch(201, 220)], "turns_seen": 224},
                                          224, SLACK), False)

    # ---- A3: both sides of the tail, every tier, real rollup input -------
    def row(n, phase, stored, tail_facts=0, adv=0):
        return {"turn": n, "phase": phase, "stored": 1 if stored else 0,
                "skipped": 0 if stored else 1, "tail_new_facts": tail_facts,
                "watermark_advance": adv}

    # The shipped run: every turn refused, 3 facts from the backfill.
    all_refused = [row(n, "refuse", False, 0, 20 if n in (11, 21) else 0)
                   for n in range(1, 23)]
    expect("A3 every turn refused; facts came from the backfill",
           lambda: _phase_problems(all_refused, 3), True)
    good = ([row(n, "store-1", True, 1 if n == 1 else 0) for n in range(1, 31)]
            + [row(n, "refuse", False, 0, 20 if n % 10 == 1 else 0)
               for n in range(31, 71)]
            + [row(n, "store-2", True, 1 if n == 71 else 0)
               for n in range(71, 113)])
    expect("A3 CONTROL: store / refuse / store, all healthy",
           lambda: _phase_problems(good, 5), False)
    one_refused = [dict(r) for r in good]
    one_refused[4].update(stored=0, skipped=1)
    expect("A3 a prose turn refused in a STORE phase",
           lambda: _phase_problems(one_refused, 5), True)
    no_writes = [dict(r, tail_new_facts=0) for r in good]
    expect("A3 decisions say stored, the tail wrote no facts",
           lambda: _phase_problems(no_writes, 3), True)
    dead_skip_rollup = [dict(r, watermark_advance=0) if r["phase"] == "refuse"
                        else r for r in good]
    expect("A3 refuse phase with no rollup on any refused turn",
           lambda: _phase_problems(dead_skip_rollup, 5), True)
    no_decision = [dict(r) for r in good]
    no_decision[40].update(stored=0, skipped=0)
    expect("A3 a turn with no tail decision at all",
           lambda: _phase_problems(no_decision, 5), True)
    loop_stored = [dict(r) for r in good]
    loop_stored[44].update(stored=1, skipped=0)
    expect("A3 a repetition-loop reply STORED in the refuse phase",
           lambda: _phase_problems(loop_stored, 5), True)
    expect("A3 no REFUSE phase at all",
           lambda: _phase_problems([r for r in good if r["phase"] != "refuse"], 5),
           True)
    expect("A3 an empty fact store at the end, all else healthy",
           lambda: _phase_problems(good, 0), True)

    expect("A3 tiers: 22 turns at production fan-out (20/10/5) builds no L2",
           lambda: _tier_reach_problems(_tier_plan(22, 20, 10, 5)), True)
    # One case per condition, each the ONLY condition that fails in it.
    expect("A3 tiers: 80 turns at 20/2/2 — one L3 refresh (alone)",
           lambda: _tier_reach_problems(_tier_plan(80, 20, 2, 2)), True)
    expect("A3 tiers: 91 turns at 20/2/2 — ends with l2 empty (alone)",
           lambda: _tier_reach_problems(_tier_plan(91, 20, 2, 2)), True)
    expect("A3 tiers: 101 turns at 20/2/2 — ends with l1 empty (alone)",
           lambda: _tier_reach_problems(_tier_plan(101, 20, 2, 2)), True)
    expect("A3 tiers CONTROL: 112 turns at 20/2/2",
           lambda: _tier_reach_problems(_tier_plan(112, 20, 2, 2)), False)
    plan = _tier_plan(112, 20, 2, 2)
    expect("A3 tiers: hierarchy stopped at L1 while the plan says L3 twice",
           lambda: _tier_observed_problems(plan, {"l1_chunks": 11, "l2_chapters": 0,
                                                  "l3_refreshes": 0}), True)
    expect("A3 tiers: L2 built, L3 never refreshed",
           lambda: _tier_observed_problems(plan, {"l1_chunks": 11, "l2_chapters": 5,
                                                  "l3_refreshes": 0}), True)
    expect("A3 tiers CONTROL: observed matches the plan",
           lambda: _tier_observed_problems(plan, dict(plan)), False)

    expect("A3 rollup input: every L1 chunk summarised placeholders",
           lambda: _rollup_input_problems([(10, 20), (11, 20)]), True)
    expect("A3 rollup input CONTROL: one chunk of real text",
           lambda: _rollup_input_problems([(0, 20), (11, 20)]), False)
    return failures, cases[0]


_selftest_failures, _selftest_cases = _oracle_selftest()
if _selftest_failures:
    print("FAIL the soak's own oracle cannot tell a broken run from a healthy one")
    for _f in _selftest_failures:
        print(f"     {_f}")
    print("     A soak whose assertions cannot fail proves nothing; refusing to "
          "run it.")
    sys.exit(1)
print(f"[soak] oracle self-test passed: {_selftest_cases} cases, every "
      f"assertion went red on its defect shape and stayed green on its control")
if os.environ.get("SOAK_ORACLE_SELFTEST_ONLY"):
    sys.exit(0)


def _skip(reason: str) -> None:
    print("SKIPPED: test_soak_conversation.py")
    print(f"  {reason}")
    print("  Start the fixture with:")
    print("    docker compose -f docker-compose.tokenizer-contract.yml "
          "up -d tokenize-fixture")
    # EXIT 3, NOT 0. A suite that reports success without running its checks
    # is worse than one that fails: "all suites pass" then means nothing, and
    # that is exactly what it meant here — every green run in this branch
    # excluded this file. 3 is the runner's SKIP code (scripts/run-tests.py);
    # set COMPACTOR_ALLOW_FIXTURE_SKIP=1 to opt into exit 0 for a context that
    # genuinely cannot start docker.
    sys.exit(0 if os.environ.get("COMPACTOR_ALLOW_FIXTURE_SKIP") else 3)


try:
    _mode = httpx.get(f"{FIXTURE_URL}/_fixture/mode", timeout=3.0)
    _mode.raise_for_status()
    _mode = _mode.json()
except Exception as e:
    _skip(f"{FIXTURE_URL} unreachable ({type(e).__name__}: {e})")

# A STALE FIXTURE IS NOT A SKIP — same doctrine as test_tokenizer_contract.
# Silently soaking against a fixture that cannot generate adversarial replies
# would report coverage this run does not have.
#
# reply_looping, not reply_chars: the needle has to be the NEWEST key the soak
# depends on. An image with reply_chars but no reply_looping passed the old
# check, and its set_mode whitelist silently drops the key — so every STORE
# phase would get the repetition loop and the soak would report the storing
# path broken when the fixture was simply old.
_missing = [k for k in ("reply_chars", "reply_looping") if k not in _mode]
if _missing:
    print(f"FAIL the fixture image is stale: /_fixture/mode has no {_missing}.")
    print("     Rebuild it:  docker compose -f docker-compose.tokenizer-contract.yml "
          "build tokenize-fixture")
    sys.exit(1)

from fastapi.testclient import TestClient  # noqa: E402

import memory  # noqa: E402

memory.ensure_storage_layout()

import facts as facts_mod  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402

client = TestClient(main.app, client=("127.0.0.1", 12345),
                    raise_server_exceptions=False)

CONV = "soak_conversation"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def text(self) -> str:
        return "\n".join(r.getMessage() for r in self.records)


_pending: list = []


def _defer_tail(coro, label=None):
    """Collect the memory tail instead of firing it into the background.

    The tail is what writes facts, indexes episodic and rolls up summaries. In
    production it is fire-and-forget; in a soak it has to COMPLETE before the
    next turn, or the run measures a system whose memory never caught up and
    every assertion about accumulation is meaningless.
    """
    _pending.append(coro)
    # TRUE, and it is not a formality. v3.1.8 gave _fire_and_forget a bool
    # contract - False means the pool SHED the tail - and a double that
    # returns None reads as shed, so the caller counts the exchange lost and
    # skips it. Same defect already fixed in _spy_fire/_no_schedule.
    return True


def _drain_tails() -> None:
    while _pending:
        coro = _pending.pop(0)
        try:
            asyncio.run(coro)
        except Exception as e:  # a tail failure is a finding, not a crash
            print(f"  NOTE tail raised: {type(e).__name__}: {e}")


def _reset_fixture() -> None:
    """Put the fixture back in an honest mode before soaking.

    The contract tests deliberately leave it lying — a 0.5 count factor, a 400
    status, an artificial delay. Soaking against that measures the fixture's
    sabotage rather than the compactor, and would either fail for the wrong
    reason or, worse, pass while asserting nothing.
    """
    # /_fixture/reset clears STATS, not the mode — the sabotage settings
    # survive it, so they have to be set back by name.
    httpx.post(f'{FIXTURE_URL}/_fixture/reset', timeout=5.0)
    httpx.post(
        f'{FIXTURE_URL}/_fixture/mode',
        json={'tokenize_mode': 'ok', 'factor': 1.0, 'status': 200,
              'delay': 0.0, 'assistant_final_400': False},
        timeout=5.0,
    )
    m = httpx.get(f'{FIXTURE_URL}/_fixture/mode', timeout=5.0).json()
    if m.get('tokenize_mode') != 'ok' or float(m.get('factor', 1)) != 1.0:
        fail('the fixture would not return to an honest mode', repr(m))
    if m.get('assistant_final_400'):
        fail('the fixture is still refusing assistant-final lists',
             'that is the D1 sabotage mode; a soak cannot run against it')


def _fixture_stats() -> dict:
    try:
        return httpx.get(f"{FIXTURE_URL}/_fixture/stats", timeout=5.0).json()
    except Exception:
        return {}


def _set_reply(seq: int, looping: bool) -> None:
    # reply_looping EXPLICITLY, every turn, and read back from set_mode's own
    # response. The shipped soak never set it and got the fixture's default
    # (True) on every turn, which is how a whole run became one refusal
    # streak without anything saying so.
    m = httpx.post(f"{FIXTURE_URL}/_fixture/mode",
                   json={"reply_chars": REPLY_CHARS, "reply_seq": seq,
                         "reply_looping": looping}, timeout=5.0).json()
    if m.get("reply_looping") is not looping or m.get("reply_seq") != seq:
        fail(f"turn {seq}: the fixture did not take reply_looping={looping}",
             repr(m))


def _turn(n: int, history: list[dict], looping: bool
          ) -> tuple[int, str, list[dict], str, int]:
    """One real POST through the compactor to the fixture.

    Returns (status, log text, forwarded payload, reply text, LLM calls made
    ON THE REQUEST PATH) — the last one measured before the memory tail runs,
    because the tail does not block the user."""
    _set_reply(n, looping)
    _calls_at_start = _fixture_stats().get("chat_completions", 0)
    cap = _Capture()
    lg = logging.getLogger("compactor")
    lg.addHandler(cap)
    forwarded: list[dict] = []

    _real_post = main.httpx.AsyncClient.post

    async def _spy(self, url, **kw):
        if url.endswith("/v1/chat/completions"):
            forwarded.append(kw.get("json") or {})
        return await _real_post(self, url, **kw)

    try:
        with patch.object(main, "_fire_and_forget", _defer_tail), \
             patch.object(main.httpx.AsyncClient, "post", _spy):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "fixture-model", "messages": history,
                      "stream": False},
                headers={"X-Conversation-Id": CONV},
            )
    finally:
        lg.removeHandler(cap)
    # Counted HERE, before the tail is drained. The tail (fact extraction,
    # dedup, rollup) is fire-and-forget in production and does NOT block the
    # reply; this soak runs it synchronously only so memory accumulates
    # deterministically. Counting it against the request-path budget measures
    # the wrong thing — and the first version of this assertion did exactly
    # that, then blamed the code for it.
    request_calls = _fixture_stats().get("chat_completions", 0) - _calls_at_start
    _drain_tails()
    sent = forwarded[-1].get("messages", []) if forwarded else []
    reply = ""
    try:
        reply = (r.json()["choices"][0]["message"]["content"]) or ""
    except Exception:
        pass
    return r.status_code, cap.text(), sent, reply, request_calls


def fail(label: str, detail: str = "") -> None:
    print(f"FAIL {label}")
    if detail:
        print(f"     {detail}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# The soak
# ---------------------------------------------------------------------------

import tailhealth  # noqa: E402

_L1 = summarizer.L1_CHUNK_SIZE
_LAG_BUDGET = _lag_budget(_L1)
# Coverage slack is one L1 chunk, for the reason _lag_budget gives.
_SLACK = _L1
_PLAN = _tier_plan(TURNS, _L1, summarizer.L2_CHUNK_SIZE, summarizer.L3_CHUNK_SIZE)

# ---- configuration self-checks, BEFORE the run ---------------------------
# A1's defect was not in the check, it was in the configuration: an override
# of SOAK_TURNS made the check arithmetically unable to fire and nothing said
# so. Every assertion below that depends on how long the run is, or how the
# phases fall, is checked for reachability here, and an unreachable one FAILS
# the soak rather than letting it print `ok`.
_config_problems = (_lag_guard_unreachable(TURNS, _L1, _LAG_BUDGET)
                    + _tier_reach_problems(_PLAN))
if STORE_TURNS < 1:
    _config_problems.append(f"SOAK_STORE_TURNS={STORE_TURNS}: no first STORE phase")
if 2 * REFUSE_TURNS < _L1 + 2:
    _config_problems.append(
        f"SOAK_REFUSE_TURNS={REFUSE_TURNS}: a refusal streak that short is not "
        f"guaranteed to cross an L1 boundary, so rollup-on-skip may never run "
        f"inside it (need > {_L1 // 2} turns)")
if TURNS - STORE_TURNS - REFUSE_TURNS < 1:
    _config_problems.append(
        f"SOAK_TURNS={TURNS} leaves no second STORE phase after "
        f"{STORE_TURNS}+{REFUSE_TURNS}: recovery after a refusal streak is untested")
if _config_problems:
    fail("the configured run cannot exercise what this soak asserts",
         "; ".join(_config_problems))

# ---- instrumentation (wrappers only: every call reaches the real code) ----
#
# _facts_tail: the fact texts that appear DURING the tail's own extraction.
# The backfill is a second, independent writer that fills an empty store once;
# a non-empty store is therefore not evidence the tail stored anything, and in
# the shipped run it was not (3 facts, all backfill, 22 of 22 turns refused).
_tail_new_facts = [0]
_real_facts_tail = main._facts_tail


async def _facts_tail_spy(conv_id, *args, **kwargs):
    before = {f.get("text") for f in facts_mod.load_facts(conv_id)}
    try:
        return await _real_facts_tail(conv_id, *args, **kwargs)
    finally:
        after = {f.get("text") for f in facts_mod.load_facts(conv_id)}
        _tail_new_facts[0] += len(after - before)


main._facts_tail = _facts_tail_spy

# _summarize_pieces: how many of each L1 chunk's input pieces were redaction
# placeholders. A chunk of placeholders is non-empty, so the coverage oracle
# accepts it — and in the shipped run 21 of 22 assistant turns were
# placeholders by the time the rollup saw them.
_l1_inputs: list[tuple[int, int]] = []
_real_summarize_pieces = summarizer._summarize_pieces


async def _summarize_pieces_spy(conv_id, client_, vllm_url, model, system_prompt,
                                pieces, max_tokens):
    if system_prompt == summarizer._PROMPT_L1:
        _l1_inputs.append((
            sum(1 for p in pieces if main._DEGENERATE_HISTORY_PLACEHOLDER in p),
            len(pieces),
        ))
    return await _real_summarize_pieces(conv_id, client_, vllm_url, model,
                                        system_prompt, pieces, max_tokens)


summarizer._summarize_pieces = _summarize_pieces_spy

# The tier writers: counted from the STATE they mutate, not from log wording.
# With L2_CHUNK_SIZE=2 the second chunk of every pair is written and consumed
# in the same drain, so no state snapshot between turns ever shows it; only
# the writer sees it happen.
_observed = {"l1_chunks": 0, "l2_chapters": 0, "l3_refreshes": 0}
_real_l1, _real_l2, _real_l3 = (summarizer._do_l1_rollup,
                                summarizer._do_l2_rollup,
                                summarizer._do_l3_rollup)


async def _l1_spy(conv_id, client_, vllm_url, model, state, *args, **kwargs):
    before = len(state.get("l1") or [])
    ok = await _real_l1(conv_id, client_, vllm_url, model, state, *args, **kwargs)
    _observed["l1_chunks"] += max(0, len(state.get("l1") or []) - before)
    return ok


async def _l2_spy(conv_id, client_, vllm_url, model, state):
    before = len(state.get("l2") or [])
    ok = await _real_l2(conv_id, client_, vllm_url, model, state)
    _observed["l2_chapters"] += max(0, len(state.get("l2") or []) - before)
    return ok


async def _l3_spy(conv_id, client_, vllm_url, model, state):
    before = state.get("l3")
    ok = await _real_l3(conv_id, client_, vllm_url, model, state)
    if state.get("l3") is not before and isinstance(state.get("l3"), dict):
        _observed["l3_refreshes"] += 1
    return ok


summarizer._do_l1_rollup, summarizer._do_l2_rollup, summarizer._do_l3_rollup = (
    _l1_spy, _l2_spy, _l3_spy)

print(f"[soak] {TURNS} turns, ~{REPLY_CHARS}-char adversarial replies, "
      f"window {os.environ['MAX_MODEL_LEN']}, store {_STORE}")
print(f"[soak] phases: store 1-{STORE_TURNS}, refuse "
      f"{STORE_TURNS + 1}-{STORE_TURNS + REFUSE_TURNS}, store "
      f"{STORE_TURNS + REFUSE_TURNS + 1}-{TURNS}")
print(f"[soak] hierarchy L1={_L1} L2={summarizer.L2_CHUNK_SIZE} "
      f"L3={summarizer.L3_CHUNK_SIZE}; a healthy run must write "
      f"{_PLAN['l1_chunks']} L1 chunks, {_PLAN['l2_chapters']} L2 chapters, "
      f"{_PLAN['l3_refreshes']} L3 refreshes; lag budget {_LAG_BUDGET}")
print(f"[soak] fixture: {FIXTURE_URL}")
_reset_fixture()

history: list[dict] = []
rows: list[dict] = []
calls_seen: list[tuple] = []
compaction_fired = False
_prev_watermark = 0

for n in range(1, TURNS + 1):
    phase = _phase_of(n)
    history.append({"role": "user",
                    "content": f"Turn {n}. Tell me about item {n} in detail."})
    _th0 = tailhealth.snapshot()
    _facts0 = _tail_new_facts[0]
    _t0 = time.monotonic()
    status, log, sent, reply, _calls = _turn(n, history,
                                              looping=(phase == "refuse"))
    _elapsed = time.monotonic() - _t0
    _th1 = tailhealth.snapshot()

    # ---- invariants that must hold on EVERY turn --------------------------
    # Each one is a failure this project has actually shipped.

    if status != 200:
        fail(f"turn {n}: backend rejected the request (HTTP {status})",
             "A conversation that grows must never become unanswerable. "
             "This is the 2026-08-29 shape.")

    for needle, why in (
        ("/tokenize degraded",
         "the counter fell back to an estimator that reads up to 51% low "
         "(D1: an assistant-final list refused by the chat template)"),
        ("compaction failed",
         "compaction fell through and forwarded the original messages "
         "(the head of the 2026-08-28 cascade)"),
        ("hard budget FAILED to fit",
         "the guard could not fit the payload and forwarded anyway"),
        ("REQUEST REJECTED",
         "the turn produced no reply, no facts and no episodic write"),
        (", margin ",
         "the calibration margin latched and is now narrowing the window "
         "for every conversation in the process (D4)"),
    ):
        if needle in log:
            fail(f"turn {n}: {needle!r} appeared", why)

    # THE ASSERTION THAT WAS MISSING, and its absence is why 2026-08-29
    # happened. Compaction runs on the REQUEST PATH. A conversation with a
    # summarization backlog produced 33 LLM calls on one request — eight
    # minutes with a dead composer, and the user got no reply at all. Nothing
    # here measured how much work a single turn did, so a green soak said
    # everything was fine.
    #
    # The budget is the cap plus one: the cap bounds summarization calls, and
    # the user's own reply is the +1. Fact extraction and dedup run on the
    # background tail, which the soak drains separately after the turn.
    _budget = main.MAX_SUMMARY_CALLS_PER_REQUEST + 1
    if _calls > _budget:
        fail(f"turn {n}: one request made {_calls} LLM calls (budget {_budget})",
             f"compaction is unbounded on the request path. At ~1024 output "
             f"tokens per call on a 24B model this is minutes of latency for a "
             f"user who is watching a blank composer. See "
             f"MAX_SUMMARY_CALLS_PER_REQUEST.")
    calls_seen.append((n, _calls, round(_elapsed, 1)))

    if "hard budget enforced" in log:
        compaction_fired = True  # shedding implies we got past the target

    if "summarize:" in log:
        compaction_fired = True

    # The counter and the log line must agree. "skipping memory tail" is the
    # phrase the pod is grepped for, and tailhealth is what /health/full
    # publishes; a skip one of them sees and the other does not is the
    # silent-skip class, and it would make every phase count below a guess.
    _stored = _th1["stored"] - _th0["stored"]
    _skipped = _th1["skipped"] - _th0["skipped"]
    _skip_lines = log.count("skipping memory tail")
    if _skip_lines != _skipped:
        fail(f"turn {n}: tailhealth counted {_skipped} skip(s) but the log "
             f"carries {_skip_lines} 'skipping memory tail' line(s)",
             "the published counter and the grepped log disagree about "
             "whether this exchange reached memory")

    # The REAL reply, not a placeholder. A soak that appends "(reply)" grows a
    # 47-message conversation weighing 1,211 tokens, never reaches the
    # compaction trigger, and asserts nothing while looking busy — which is
    # precisely the class of test this file exists to replace.
    if not reply:
        fail(f"turn {n}: the backend returned no assistant text",
             "the soak cannot grow a conversation it never receives")

    st = summarizer.load_state(CONV)
    _wm = st.get("last_summarized_turn", 0)
    nonsys = [m for m in sent if m.get("role") != "system"]
    # `history` BEFORE the reply is appended: 2n-1, which is what both the lag
    # budget and _lag_guard_unreachable are computed against.
    rows.append({"turn": n, "phase": phase, "sent_msgs": len(sent),
                 "sent_nonsys": len(nonsys), "history": len(history),
                 "watermark": _wm, "stored": _stored, "skipped": _skipped,
                 "tail_new_facts": _tail_new_facts[0] - _facts0,
                 "watermark_advance": _wm - _prev_watermark})
    _prev_watermark = _wm
    history.append({"role": "assistant", "content": reply})

    # COVERAGE ON EVERY TURN, not once at the end. A hierarchy that tiles
    # correctly at turn 112 can have stranded turns at turn 60 that a later
    # rollup happened to cover, and the shipped end-of-run check could see
    # neither that nor the turns above a frozen watermark (A2).
    _gaps = _summary_coverage_gaps(st, len(history), _SLACK)
    if _gaps:
        fail(f"turn {n} ({phase}): history was consumed without being "
             f"represented anywhere",
             "; ".join(_gaps) + f"  [state: watermark={_wm} "
             f"turns_seen={st.get('turns_seen')} "
             f"l1={[(c.get('first_turn'), c.get('last_turn')) for c in st.get('l1') or []]} "
             f"l2={[(c.get('first_turn'), c.get('last_turn')) for c in st.get('l2') or []]} "
             f"l3={(st.get('l3') or {}).get('first_turn')}-"
             f"{(st.get('l3') or {}).get('last_turn')}]")

    if n % 10 == 0:
        _l3 = st.get("l3") if isinstance(st.get("l3"), dict) else None
        print(f"  turn {n:>3} [{phase:<7}]: history={len(history):>3} "
              f"forwarded={len(sent):>3} facts={len(facts_mod.load_facts(CONV)):>3} "
              f"L1={len(st.get('l1') or [])} L2={len(st.get('l2') or [])} "
              f"L3={'%s-%s' % (_l3.get('first_turn'), _l3.get('last_turn')) if _l3 else 'n'} "
              f"watermark={_wm}")

# ---------------------------------------------------------------------------
# End-of-run invariants
# ---------------------------------------------------------------------------

print()
print("[soak] end-of-run checks")
if calls_seen:
    _worst = max(calls_seen, key=lambda x: x[1])
    _slow = max(calls_seen, key=lambda x: x[2])
    print(f"  ok   LLM calls per request stayed bounded "
          f"(worst {_worst[1]} at turn {_worst[0]}, budget "
          f"{main.MAX_SUMMARY_CALLS_PER_REQUEST + 1}; slowest turn "
          f"{_slow[2]}s at turn {_slow[0]})")

if not compaction_fired:
    fail("the run never exceeded the compaction target",
         f"{TURNS} turns of ~{REPLY_CHARS} chars did not fill a "
         f"{os.environ['MAX_MODEL_LEN']}-token window. Raise SOAK_TURNS or "
         f"SOAK_REPLY_CHARS — a soak that never compacts proves nothing.")
print("  ok   the conversation actually got big enough to exercise compaction")

# Context delivery — and the right property is NOT "many raw turns arrived".
#
# A guard that sheds 36 of 40 messages is behaving CORRECTLY if compaction
# summarised them first: the older material is still represented, just densely.
# That is the whole design. The 2026-08-28 failure was not shedding — it was
# shedding turns that nothing had summarised, because compaction had died. The
# turns were simply gone, and the log said "enforced" either way.
#
# So the invariant is COVERAGE: every turn is either forwarded verbatim or sits
# below the summarizer's watermark. A turn that is neither has left the system.
substantive = [r for r in rows if r["history"] >= 6]
if not substantive:
    fail("the run never built a conversation worth measuring", "raise SOAK_TURNS")

# The property is that the SUMMARIZER KEEPS PACE — not that every message is
# covered at every instant.
#
# My first version asserted `watermark + forwarded >= history` and fired at
# turn 19 with history=37, watermark=20. That is not loss: L1_CHUNK_SIZE is 20
# message-units, so a lag of up to one unfilled chunk is the design working.
# Asserting otherwise would have made this test red on a healthy system, which
# is how a suite teaches people to ignore it.
#
# The incident looked different in a way this DOES catch: the watermark froze
# (compaction was failing) while the conversation grew, so the lag climbed
# without bound — production reached history=116 against a watermark that had
# stopped, and the guard shed everything above it. So: bound the lag, and
# require the watermark to actually move. The budget is ONE chunk (see
# _lag_budget), and _lag_guard_unreachable already failed the run above if
# this turn count could not show a hierarchy that rolled up once and died.
lagging = _lagging_rows(substantive, _LAG_BUDGET)
if lagging:
    w = max(lagging, key=lambda r: r["history"] - r["watermark"])
    fail(f"turn {w['turn']}: the summarizer fell {w['history'] - w['watermark']} "
         f"message-units behind (budget {_LAG_BUDGET})",
         f"history={w['history']} watermark={w['watermark']} "
         f"forwarded={w['sent_nonsys']}. A watermark that stops advancing while "
         f"the conversation grows means compaction is failing, and everything "
         f"above it is being DROPPED by the guard rather than summarised. That "
         f"is 2026-08-28, and it logs as 'hard budget enforced' either way.")

# There used to be a second check here, `substantive[-1]["watermark"] <= 0`
# ("the summarizer watermark never advanced"). With a one-chunk budget it can
# no longer decide anything: a final watermark of 0 is a lag of 2*TURNS-1 on
# the last row, and _lag_guard_unreachable has already refused any TURNS for
# which that is not over budget, so `lagging` above always fails first. A
# check that cannot fire is the defect class A1 was; it is removed rather
# than left as decoration.

worst = max(substantive, key=lambda r: r["history"] - r["watermark"])
print(f"  ok   the summarizer kept pace (worst lag {worst['history'] - worst['watermark']} "
      f"of {_LAG_BUDGET} at turn {worst['turn']}; final watermark "
      f"{substantive[-1]['watermark']}; a hierarchy that rolled up once and "
      f"died would have shown {(2 * TURNS - 1) - _L1})")

# BOTH SIDES OF THE MEMORY TAIL (A3). The shipped soak's only memory
# assertion was a non-empty fact store, which the backfill filled on a run
# whose tail refused every turn.
stored = facts_mod.load_facts(CONV)
_phase_p = _phase_problems(rows, len(stored))
if _phase_p:
    fail("the run did not exercise both sides of the memory tail",
         "; ".join(_phase_p))
for _name in ("store-1", "refuse", "store-2"):
    _prs = [r for r in rows if r["phase"] == _name]
    if _name == "refuse":
        print(f"  ok   {_name}: {sum(r['skipped'] for r in _prs)}/{len(_prs)} "
              f"repetition-loop replies refused, and the hierarchy still "
              f"rolled up on {sum(1 for r in _prs if r['watermark_advance'] > 0)} "
              f"of them (rollup-on-skip)")
    else:
        print(f"  ok   {_name}: {sum(r['stored'] for r in _prs)}/{len(_prs)} "
              f"prose replies stored, and the tail's own extraction wrote "
              f"{sum(r['tail_new_facts'] for r in _prs)} new fact(s)")
print(f"  ok   facts accumulated ({len(stored)} on disk, "
      f"{_tail_new_facts[0]} written by the tail itself)")

# ...and must not contain what the model decorates with. This is the loop where
# the system reads its own scaffolding back as established truth.
bad = [f for f in stored if not facts_mod.is_storable_fact(f.get("text", ""))]
if bad:
    fail(f"{len(bad)} stored fact(s) are markup, not facts",
         "every reply in this run contained a code fence, a heading and a "
         "62-character box-drawing rule; none of them may be memory")
print("  ok   no markup reached the fact store")

state = summarizer.load_state(CONV)

# "compaction fired but the summary hierarchy produced NOTHING" used to be a
# check here (and before that a NOTE, which could not fail at all). It is
# subsumed twice over and could no longer decide anything: with no L1 chunk
# the per-turn coverage check fails as soon as the conversation passes one
# chunk plus slack (turn 11), long before the window fills and
# `compaction_fired` is set; and _tier_observed_problems below requires the
# exact number of L1 chunks the arithmetic says, not merely one.
# EVERY TIER THE COVERAGE CLAIM NAMES MUST HAVE BEEN BUILT, as many times as
# the arithmetic says. The shipped run ended at L1=2 L2=0 L3=n, so the l2 loop
# and the l3 branch of the coverage oracle never executed, and the claim
# printed below named all three anyway.
_tier_p = _tier_observed_problems(_PLAN, _observed)
if _tier_p:
    fail("the summary hierarchy built less than a healthy one must have",
         "; ".join(_tier_p) + f"  [final state: l1={len(state.get('l1') or [])} "
         f"l2={len(state.get('l2') or [])} l3={'set' if state.get('l3') else 'none'}]")
print(f"  ok   the summary hierarchy built every tier ({_observed['l1_chunks']} L1 "
      f"chunks, {_observed['l2_chapters']} L2 chapters, "
      f"{_observed['l3_refreshes']} L3 refreshes; plan "
      f"{_PLAN['l1_chunks']}/{_PLAN['l2_chapters']}/{_PLAN['l3_refreshes']})")

# REAL INPUT, NOT PLACEHOLDERS. A chunk built from redaction placeholders is
# non-empty and the coverage oracle accepts it; the refusal phase makes such
# chunks on purpose. What must also be true is that the run built chunks
# from real replies, or "nothing vanished" is a statement about placeholders.
_input_p = _rollup_input_problems(_l1_inputs)
if _input_p:
    fail("no L1 chunk in this run summarised real text", "; ".join(_input_p))
_clean = sum(1 for ph, _ in _l1_inputs if ph == 0)
print(f"  ok   {_clean} of {len(_l1_inputs)} L1 summarization(s) read only real "
      f"replies; {len(_l1_inputs) - _clean} read redaction placeholders "
      f"({sum(ph for ph, _ in _l1_inputs)} placeholder piece(s) in all — the "
      f"refusal phase, by design)")

# The property [substantive]/[lagging] above did not check: not just THAT a
# chunk exists, but that the chunks which exist actually TILE the turns the
# conversation has, with real content in each one. A bug that advances
# last_summarized_turn without writing (or while writing an EMPTY) L1/L2 entry
# passes every check above — the watermark moves, so the lag stays bounded —
# while a stretch of history in the middle is consumed and represented
# nowhere. That is exactly what v3.1.3 (f8614b6, "a failed summary must not
# delete the turns it failed on") fixed, and exactly the shape of "history
# consumed without being represented anywhere": the watermark is evidence of
# PROGRESS, not of COVERAGE. Already asserted on every turn in the loop; this
# is the final state, stated as what it proves.
gaps = _summary_coverage_gaps(state, len(history), _SLACK)
if gaps:
    fail("history was consumed without being represented anywhere",
         "; ".join(gaps))
_final_l3 = state.get("l3") if isinstance(state.get("l3"), dict) else {}
print(f"  ok   on all {TURNS} turns, every message-unit older than one L1 chunk "
      f"was covered by exactly one non-empty entry, contiguously from turn 1 — "
      f"final tiling L3 {_final_l3.get('first_turn')}-{_final_l3.get('last_turn')}"
      f" + L2 {[(c['first_turn'], c['last_turn']) for c in state.get('l2') or []]}"
      f" + L1 {[(c['first_turn'], c['last_turn']) for c in state.get('l1') or []]}"
      f" of {len(history)} units")

print()
print(f"All soak checks passed over {TURNS} turns.")
print("REMINDER: the fixture's tokenizer is not Cydonia's. This proves the "
      "system holds together over a growing conversation, NOT that the "
      "production token numbers are right. The fixture's summaries are "
      "adversarial replies, not summaries: coverage here is about which turns "
      "each entry CLAIMS and that it is non-empty, not what it says.")
