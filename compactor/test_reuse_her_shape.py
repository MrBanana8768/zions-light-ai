"""
CPU-only Tier-1 test (v3.1.9.8, item 5). NOT a reproduction of production:
it drives `compact_if_needed` directly on a hand-built, IDEALIZED synthetic
state -- a hierarchy sized and shaped by hand to resemble the numbers in her
real shape, not extracted from her real data or her real request path. What
it proves is narrower than "the fix works for her": under BOTH the RunPod
template's settings and the owner's live pod settings, an idealized
conversation with this shape (1 L3 + 2 L2 chapters + 4 L1 scenes, ~4,400
turns, ~15 uncovered) takes the stand-in reuse path rather than declining.

v3.1.9.8 round-2 (hostile review, item 6): earlier revisions of this
docstring claimed this test "verifies the reuse path fires on her shape" --
overstated. This file passes UNCHANGED on the pre-fix HEAD too (none of
items 1-4's fixes touch the reuse-fit decision this file drives), so it does
not GUARD any of those fixes; it is a separate, additive check that the
idealized shape clears the reuse-fit arithmetic, kept for that value alone.

THE ARITHMETIC (verified empirically below, and reported here so a reader
does not have to re-derive it):

  RunPod template (runpod.env.template, the ACTIVE/uncommented lines only --
  COMPACTOR_INJECTION_BUDGET_FRACTION and COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS
  are commented out there, so those two are this deployment's plain
  defaults, 0.5 and 12000):
    MAX_MODEL_LEN=32768, COMPACTOR_GENERATION_RESERVE=12000,
    COMPACTOR_MAX_RETAINED_IMAGES=1 (the task brief said 2 for this pair;
    the template itself has 1 -- checked directly, see below),
    COMPACTOR_STANDIN_BUDGET_FRACTION=1.0.
      effective_limit = min(32768, 32768-12000)            = 20768
      inject_budget   = int(20768 * 0.5)                    = 10384
      standin ceiling = min(12000, 1.0 * 10384)             = 10384

  Owner's live pod settings (memory.md, 2026-09-25: COMPACTOR_MAX_RETAINED_
  IMAGES=1 and COMPACTOR_GENERATION_RESERVE=8000 set in supervisord, every
  other knob left at the template's values):
      effective_limit = min(32768, 32768-8000)              = 24768
      inject_budget   = int(24768 * 0.5)                    = 12384
      standin ceiling = min(12000, 1.0 * 12384)             = 12000

  This file's synthetic hierarchy (1 L3 + 2 L2 chapters + 4 L1 scenes,
  covering all but the newest 15 of ~4,400 turns) renders at 10,731 tokens
  unbounded -- close to her measured shape (~11k) and, empirically, still
  under BOTH ceilings above once rendered at each one's own budget (the
  render drops the oldest tier item first when squeezed; at these sizes
  neither ceiling squeezes it). Both settings are asserted to take the
  reuse path below -- if either did not, this file would say so via the
  MUST NOT DECLINE check, per the coordinator's instruction: report the
  arithmetic and do not touch the thresholds.

NOTE ON COMPACTOR_MAX_RETAINED_IMAGES: the task brief names 2 for the
RunPod-template pair. runpod.env.template's own active line is
`COMPACTOR_MAX_RETAINED_IMAGES=1` (checked directly). This test uses the
template's REAL value (1) for both settings, since that is what actually
ships; it does not affect the reuse arithmetic above (no images are used in
this fixture).

Also asserts: no full-history /tokenize call is made under either setting
(count_tokens_exact is spied on; every call it sees is far smaller than the
~4,400-turn history -- the reuse decision for this shape never has to price
`_fresh_span_preview` at all, since the uncovered tail is only ~15 turns and
the certain-floor gate, item 4's own fix, would in any case stop a huge one
from reaching /tokenize).

Run: python test_reuse_her_shape.py
"""

import asyncio
import logging
import os
import sys
import tempfile

os.environ.pop("MODEL_REPO", None)  # char/4 local estimator, no tokenizer download
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="reuse-her-shape-")

# Settings common to BOTH scenarios below (the template's other active
# lines; neither scenario overrides these).
os.environ["MAX_MODEL_LEN"] = "32768"
os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "4"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "1024"
os.environ["COMPACTOR_SUMMARY_INPUT_RESERVE"] = "2048"
os.environ["COMPACTOR_MAX_RETAINED_IMAGES"] = "1"
os.environ["COMPACTOR_STANDIN_BUDGET_FRACTION"] = "1.0"
os.environ["COMPACTOR_MAX_SUMMARY_CALLS"] = "4"
os.environ["COMPACTOR_GENERATION_RESERVE"] = "12000"  # overridden per-scenario below

import memory  # noqa: E402
memory.ensure_storage_layout()
import main  # noqa: E402
import summarizer  # noqa: E402

FAILED = False


def check(cond, label, detail=""):
    global FAILED
    status = "ok  " if cond else "FAIL"
    print(f"  {status} {label}" + (f" ({detail})" if detail and not cond else ""))
    if not cond:
        FAILED = True


def _ns(msgs):
    return [m for m in msgs if m.get("role") != "system"]


def _history(n_exchanges, words=15):
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"q{i} " + ("w " * words)})
        out.append({"role": "assistant", "content": f"a{i} " + ("w " * words)})
    return out


def _seed_hierarchy(conv, msgs, l1_chunks, l2_chunks, l3_chunk):
    st = summarizer.load_state(conv)
    st["l1"] = l1_chunks
    st["l2"] = l2_chunks
    st["l3"] = l3_chunk
    st["last_summarized_turn"] = max(c["last_turn"] for c in l1_chunks)
    ns = _ns(msgs)
    for c in sorted(l2_chunks + l1_chunks, key=lambda c: c["first_turn"]):
        summarizer._record_chunk_fps(
            st, c["first_turn"], c["last_turn"], ns[c["first_turn"] - 1:c["last_turn"]]
        )
    summarizer.save_state(conv, st)
    return st


# 2,200 exchanges = 4,400 turns; the newest 15 turns are left UNCOVERED by
# the hierarchy (her shape: a small catch-up tail, not a backlog).
N_PAIRS = 2200
UNCOVERED = 15
MSGS = _history(N_PAIRS, words=15)
_NS = _ns(MSGS)
TOTAL_TURNS = len(_NS)
COVERED_LAST = TOTAL_TURNS - UNCOVERED
check(TOTAL_TURNS == 4400, "fixture: ~4,400 turns", f"got {TOTAL_TURNS}")

# 1 L3 theme + 2 L2 chapters + 4 L1 scenes, sized (by hand, in characters --
# _estimate_block_tokens prices ASCII at chars/4) so the whole hierarchy
# renders near her measured ~11k-token shape.
_SEG1 = COVERED_LAST // 4
L1_CHUNKS = [
    {
        "text": f"L1-{i} " + "s" * 4600,
        "first_turn": i * _SEG1 + 1,
        "last_turn": (i + 1) * _SEG1 if i < 3 else COVERED_LAST,
    }
    for i in range(4)
]
_SEG2 = COVERED_LAST // 2
L2_CHUNKS = [
    {"text": "L2-0 " + "c" * 8000, "first_turn": 1, "last_turn": _SEG2},
    {"text": "L2-1 " + "c" * 8000, "first_turn": _SEG2 + 1, "last_turn": COVERED_LAST},
]
L3_CHUNK = {"text": "L3 " + "t" * 8000, "first_turn": 1, "last_turn": COVERED_LAST}


async def _fake_summarize(client, to_summarize):
    """Stands in for the one real summarize() call the ~15-turn uncovered
    tail still needs (the fresh span this small is expected to be
    summarized for real in production; this file is not testing THAT
    call, only that the stand-in reuse path is taken for the covered
    bulk). No network reachable in this test process regardless."""
    return "FRESH-SUMMARY-OF-THE-TAIL", []


class _ExactSpy:
    """Records every count_tokens_exact call's payload SIZE (never content
    -- project rule, and irrelevant here) so this file can assert none of
    them is anywhere near full-history size."""

    def __init__(self):
        self.call_sizes: list[int] = []

    def __call__(self, messages, *a, **k):
        self.call_sizes.append(len(messages) if messages else 0)
        return None  # /tokenize unreachable in this test process anyway


def _run_scenario(name: str, *, generation_reserve: int):
    print(f"\n[scenario] {name} (COMPACTOR_GENERATION_RESERVE={generation_reserve})")
    os.environ["COMPACTOR_GENERATION_RESERVE"] = str(generation_reserve)
    main.GENERATION_RESERVE = generation_reserve  # module global re-read, matches
    # how test_reuse_fit.py and others reload numeric globals for a scenario
    # (main.py reads these as module-level constants at import time; tests
    # that vary them post-import poke the same names main.py's own code
    # reads, exactly like test_reuse_fit.py's sibling scenario tests do).

    # v3.1.9.8 round-2 (hostile review, item 6): call PRODUCTION's own
    # `_effective_limit` (main.py, right beside HARD_INPUT_LIMIT) instead of
    # a hand-rolled copy of its formula -- the earlier copy here omitted the
    # `max(GENERATION_RESERVE, req_max_tokens)` term entirely, silently
    # correct only because this fixture never sets `max_tokens` (so
    # req_max_tokens=0 either way). Passed explicitly for the same reason:
    # a reader should see this fixture's req_max_tokens is deliberately 0,
    # not merely omitted.
    effective_limit = main._effective_limit(req_max_tokens=0)
    inject_budget = int(effective_limit * main.INJECTION_BUDGET_FRACTION)
    standin_ceiling = main._standin_reuse_ceiling(inject_budget)
    print(
        f"    effective_limit={effective_limit} inject_budget={inject_budget} "
        f"standin_ceiling={standin_ceiling}"
    )

    conv = f"reuse-her-shape-{name}"
    _seed_hierarchy(conv, MSGS, L1_CHUNKS, L2_CHUNKS, L3_CHUNK)
    st = summarizer.load_state(conv)
    hier_tokens = summarizer._estimate_block_tokens(
        summarizer.format_summary_block(st, 10**9) or ""
    )
    print(f"    hierarchy renders at {hier_tokens} tokens (unbounded)")

    exact_spy = _ExactSpy()
    real_summarize = main.summarize
    real_exact = main.count_tokens_exact
    main.summarize = _fake_summarize
    main.count_tokens_exact = exact_spy

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    handler = _Capture()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    try:
        stored_out: list = []
        out = asyncio.run(
            main.compact_if_needed(
                list(MSGS), conv, stored_turns_out=stored_out, inject_budget=inject_budget
            )
        )
    finally:
        lg.removeHandler(handler)
        main.summarize = real_summarize
        main.count_tokens_exact = real_exact

    cap_refusal = any("per-request cap" in r for r in records)
    all_or_nothing_decline = any("do not fit whole" in r for r in records)
    reused = bool(stored_out) and stored_out[0] > 0

    check(
        reused,
        f"{name}: the idealized shape takes the reuse path "
        f"(not the 'compaction skipped' decline)",
        f"stored_out={stored_out}",
    )
    check(
        not cap_refusal,
        f"{name}: no 'over the N-call per-request cap' refusal",
    )
    check(
        not all_or_nothing_decline,
        f"{name}: no all-or-nothing 'do not fit whole' decline",
    )
    check(
        stored_out and stored_out[0] == COVERED_LAST,
        f"{name}: every covered turn ({COVERED_LAST}) is reused, none re-summarized",
        f"stored_out={stored_out}",
    )
    max_exact_call = max(exact_spy.call_sizes, default=0)
    check(
        max_exact_call < 100,
        f"{name}: no full-history /tokenize call (max call size {max_exact_call} "
        f"messages, vs {TOTAL_TURNS} turns in the real history)",
        f"call_sizes={exact_spy.call_sizes}",
    )
    check(len(out) > 0, f"{name}: compact_if_needed returned a non-empty array")


_run_scenario("runpod-template", generation_reserve=12000)
_run_scenario("owner-live", generation_reserve=8000)

# Restore the module global other tests importing main afterward in the
# same process would see (this file's scenarios each set it directly).
main.GENERATION_RESERVE = int(os.environ.get("COMPACTOR_GENERATION_RESERVE", "12000"))

if FAILED:
    print(
        "\nFAILED -- if this is a genuine decline under one or both settings, "
        "STOP: report the arithmetic above rather than adjusting thresholds "
        "to force a pass (per the task's explicit instruction)."
    )
    sys.exit(1)
print(
    "\nThe idealized shape takes the reuse path under both settings. This "
    "is not a claim about her actual conversation or actual request path."
)
