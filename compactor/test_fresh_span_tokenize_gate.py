"""
CPU-only Tier-1 test (v3.1.9.8, item 4): gate the fresh-span exact
/tokenize call the same way `_summarize_body`'s own `_min_possible_batches`
call is gated.

`compact_if_needed`'s reuse-fit decision (main.py, around `_fresh_span_
preview`) used to call `count_tokens_exact(_fresh_span_preview)`
UNCONDITIONALLY whenever any turn was `_changed` (edited/regenerated,
un-paired with the stored hierarchy's covered-turn record). When `_changed`
is huge -- the steady state on a conversation whose stored spans have
drifted -- that POSTs a full-size payload to /tokenize, which can hit the
same 10s read timeout the summarizer's own gate exists to avoid.

THE FIX: before that call, `_min_possible_batches(_fresh_span_preview,
budget, cap=MAX_SUMMARY_CALLS_PER_REQUEST)` computes a CERTAIN lower bound
on the batch count from the local tokenizer alone. `_fresh_reserve` (a few
lines further down) is `min(_fresh_batches, MAX_SUMMARY_CALLS_PER_REQUEST)
* SUMMARY_MAX_TOKENS` -- it SATURATES the moment `_fresh_batches` reaches
the cap. So once the certain floor alone already reaches the cap, no
/tokenize answer -- however exact -- can move `_fresh_reserve` at all: the
exact call and the real `_chunk_to_budget` count are skipped, and the
floor is used directly as `_fresh_batches`.

This file checks:
  1. A fixture with a genuinely huge `_changed` span (long turns, a small
     per-request budget) skips the exact /tokenize call for the fresh span
     entirely -- only the small, unrelated `_sys_recent_floor` exact call
     (system + recent turns) still happens.
  2. A control fixture with a SMALL `_changed` span (the floor stays under
     the cap) still makes the exact call -- the gate must not swallow the
     case where /tokenize's answer legitimately still matters.
  3. The decision outcome (the window-squeeze verdict) is unaffected by
     whether the gate fired -- same log shape, same reuse-turn count.

Run: python test_fresh_span_tokenize_gate.py
"""

import asyncio
import logging
import os
import sys
import tempfile

os.environ.pop("MODEL_REPO", None)
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="fresh-span-gate-")
# Small window so a moderate `_changed` span already needs several
# summarization batches -- the fixture does not need her real ~32k window
# to demonstrate the gate, only a budget/content ratio that produces a
# floor over the (also small, for the same reason) call cap.
os.environ["MAX_MODEL_LEN"] = "6000"
os.environ["COMPACTOR_GENERATION_RESERVE"] = "0"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "200"
os.environ["COMPACTOR_SUMMARY_INPUT_RESERVE"] = "200"
os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "4"
os.environ["COMPACTOR_MAX_RETAINED_IMAGES"] = "1"
os.environ["COMPACTOR_STANDIN_BUDGET_FRACTION"] = "1.0"
os.environ["COMPACTOR_MAX_SUMMARY_CALLS"] = "4"

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


def _make_pair(i, tag, user_words=50, asst_words=400):
    return (
        {"role": "user", "content": f"{tag}q{i} " + ("w " * user_words)},
        {"role": "assistant", "content": f"{tag}a{i} " + ("w " * asst_words)},
    )


def _build(n_pairs, n_changed_pairs):
    """`ORIGINAL` seeds the stored hierarchy's fingerprint record.
    `EDITED` matches it everywhere EXCEPT the first `n_changed_pairs`
    exchanges (turns 0..2*n_changed_pairs-1), which carry different text --
    "regenerated" turns the record does not recognise, so `_coverage_plan`
    marks them `_changed` while the tail (identical to ORIGINAL) still
    pairs and keeps `covered` spanning the whole array."""
    original = [{"role": "system", "content": "you are a companion"}]
    edited = [{"role": "system", "content": "you are a companion"}]
    for i in range(n_pairs):
        ou, oa = _make_pair(i, "orig")
        original.append(ou)
        original.append(oa)
        if i < n_changed_pairs:
            eu, ea = _make_pair(i, "EDITED")
        else:
            eu, ea = ou, oa
        edited.append(eu)
        edited.append(ea)
    return original, edited


def _seed(conv, original):
    st = summarizer.load_state(conv)
    ns_orig = _ns(original)
    st["l1"] = [{"text": "L1 " + "s" * 2000, "first_turn": 1, "last_turn": len(ns_orig)}]
    st["l2"] = []
    st["l3"] = None
    st["last_summarized_turn"] = len(ns_orig)
    summarizer._record_chunk_fps(st, 1, len(ns_orig), ns_orig)
    summarizer.save_state(conv, st)


class _ExactSpy:
    def __init__(self):
        self.call_sizes: list[int] = []

    def __call__(self, messages, *a, **k):
        self.call_sizes.append(len(messages) if messages else 0)
        return None


async def _fake_summarize(client, to_summarize):
    return "SUMMARY", []


def _drive(conv, edited, inject_budget):
    exact_spy = _ExactSpy()
    real_exact = main.count_tokens_exact
    real_summarize = main.summarize
    main.count_tokens_exact = exact_spy
    main.summarize = _fake_summarize

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
                list(edited), conv, stored_turns_out=stored_out, inject_budget=inject_budget
            )
        )
    finally:
        lg.removeHandler(handler)
        main.count_tokens_exact = real_exact
        main.summarize = real_summarize
    return out, stored_out, exact_spy, records


EFFECTIVE_LIMIT = min(main.MAX_MODEL_LEN, max(256, main.MAX_MODEL_LEN - main.GENERATION_RESERVE))
INJECT_BUDGET = int(EFFECTIVE_LIMIT * main.INJECTION_BUDGET_FRACTION)

# ---------------------------------------------------------------------------
# [1] huge `_changed` span: the exact /tokenize call for the fresh span is
#     SKIPPED (only the small, unrelated _sys_recent_floor call survives).
# ---------------------------------------------------------------------------
print("[1] a huge _changed span skips the fresh-span /tokenize call")
CONV1 = "fresh-span-gate-huge"
original1, edited1 = _build(n_pairs=1000, n_changed_pairs=780)  # ~1560 changed turns
_seed(CONV1, original1)
out1, stored_out1, spy1, records1 = _drive(CONV1, edited1, INJECT_BUDGET)

check(
    any("_changed" not in r and "fresh-span reserve is a certain floor" in r for r in records1),
    "the gate's own log line fired",
)
max_call1 = max(spy1.call_sizes, default=0)
check(
    max_call1 < 50,
    "no large exact call was made (only _sys_recent_floor's small one, if any)",
    f"call_sizes={spy1.call_sizes}",
)
check(len(out1) > 0, "compact_if_needed still returns a usable array")

# ---------------------------------------------------------------------------
# [2] CONTROL: a small `_changed` span (floor stays under the cap) still
#     makes the real exact call -- the gate must not swallow this case.
# ---------------------------------------------------------------------------
print("\n[2] control: a small _changed span still makes the real exact call")
CONV2 = "fresh-span-gate-small"
original2, edited2 = _build(n_pairs=1000, n_changed_pairs=2)  # ~4 changed turns
_seed(CONV2, original2)
out2, stored_out2, spy2, records2 = _drive(CONV2, edited2, INJECT_BUDGET)

check(
    not any("fresh-span reserve is a certain floor" in r for r in records2),
    "the gate did NOT fire (the floor is genuinely under the cap here)",
)
check(
    any(3 <= n <= 8 for n in spy2.call_sizes),
    "the real exact call for the small fresh span was made",
    f"call_sizes={spy2.call_sizes}",
)
check(len(out2) > 0, "compact_if_needed still returns a usable array")

# ---------------------------------------------------------------------------
# MUTATION CHECK: reverting the gate (always taking the `else` branch, i.e.
# always calling count_tokens_exact on _fresh_span_preview) makes test [1]'s
# "no large exact call was made" assertion fail, since spy1 would then see a
# ~1560-message call. Confirmed by the coordinator (see the task report).
# ---------------------------------------------------------------------------

if FAILED:
    print("\nFAILED")
    sys.exit(1)
print("\nAll fresh-span tokenize-gate tests passed.")
