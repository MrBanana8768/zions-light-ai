"""
CPU-only Tier-1 test (v3.1.9.8): `_min_possible_batches` early exit.

`_min_possible_batches` (main.py) proves, from the local tokenizer alone,
that a summarization backlog needs more `_chunk_to_budget` batches than
`MAX_SUMMARY_CALLS_PER_REQUEST` allows -- sparing a pointless full /tokenize
call on a payload that was always going to refuse. But it used to walk EVERY
turn to compute the exact minimum, even once the running count had already
certainly exceeded the cap: on her ~4,400-turn backlog, ~4,400 per-message
local-tokenizer calls to prove something the first dozen turns already
proved.

The fix adds an optional `cap` parameter: the loop returns as soon as the
running batch count exceeds `cap`, instead of continuing to the end. The
function's own docstring proof still holds on a partial walk -- walking
further turns can only ADD batches to what has already been counted, never
remove one, so a running count already over `cap` remains a valid CERTAIN
lower bound whether or not the rest of `turns` is ever priced.

This file checks:
  1. Same verdict as before (`cap`-vs-no-cap agree on both examples this
     RCA's own test file already established -- one confirmed-under-cap
     fixture, one confirmed-over-cap fixture -- see
     test_rca_20260925_tokenize_precut.py's [3]/[4]) across several fixture
     shapes.
  2. The early return itself: a counter fake for count_tokens proves far
     fewer per-message tokenizer calls happen when `cap` is given and the
     backlog is large, than when it is not.

Run: python test_min_possible_batches_early_exit.py
"""

import os
import sys
import tempfile

os.environ.pop("MODEL_REPO", None)  # char/4 estimator, no tokenizer download
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="min-batches-early-exit-")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["MAX_MODEL_LEN"] = "32768"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "600"
os.environ["COMPACTOR_SUMMARY_INPUT_RESERVE"] = "600"
os.environ["COMPACTOR_MAX_SUMMARY_CALLS"] = "4"

import main  # noqa: E402

FAILED = False


def check(cond, label, detail=""):
    global FAILED
    status = "ok  " if cond else "FAIL"
    print(f"  {status} {label}" + (f" ({detail})" if detail and not cond else ""))
    if not cond:
        FAILED = True


def _user_turn(i, words=3):
    return {"role": "user", "content": f"idx{i} " + " ".join(["hi"] * words)}


def _assistant_turn(i, words=90):
    return {"role": "assistant", "content": f"idx{i} " + " ".join(["word"] * words)}


def _backlog(n_pairs, user_words=3, assistant_words=90):
    out = []
    for i in range(n_pairs):
        out.append(_user_turn(i, user_words))
        out.append(_assistant_turn(i, assistant_words))
    return out


class _CountingTokenizer:
    """Wraps a real callable and counts how many times it is invoked with a
    SINGLE-message list -- the shape `_min_possible_batches` calls
    `count_tokens([m])` with, once per turn it actually visits."""

    def __init__(self, real):
        self.real = real
        self.singleton_calls = 0

    def __call__(self, messages):
        if len(messages) == 1:
            self.singleton_calls += 1
        return self.real(messages)


budget = min(
    main.MAX_MODEL_LEN,
    max(256, main.MAX_MODEL_LEN - main.SUMMARY_MAX_TOKENS - main.SUMMARY_INPUT_RESERVE),
)
cap = max(1, main.MAX_SUMMARY_CALLS_PER_REQUEST)

print(f"[setup] budget={budget} cap={cap} (MAX_SUMMARY_CALLS_PER_REQUEST)")

# ---------------------------------------------------------------------------
# [1] same verdict, cap vs no cap, across shapes clearly under and clearly
#     over the cap.
# ---------------------------------------------------------------------------
print("\n[1] the early-exit verdict matches the full-walk verdict")

fixtures = {
    "small (well under cap)": _backlog(5, user_words=3, assistant_words=10),
    "short-message, under cap (RCA finding 1 fixture)":
        _backlog(2000, user_words=4, assistant_words=4),
    "genuinely oversized (RCA finding 1's regression case)":
        _backlog(20000, user_words=4, assistant_words=4),
    "her shape: ~4,400 turns, long assistant replies":
        _backlog(2200, user_words=50, assistant_words=800),
    "empty": [],
}

for name, turns in fixtures.items():
    no_cap = main._min_possible_batches(turns, budget)
    with_cap = main._min_possible_batches(turns, budget, cap)
    verdict_no_cap = no_cap > cap
    verdict_with_cap = with_cap > cap
    check(
        verdict_no_cap == verdict_with_cap,
        f"{name}: same over/under-cap verdict",
        f"no_cap={no_cap} with_cap={with_cap} cap={cap}",
    )
    # the exact count only has to agree while it's still under the cap --
    # once over, cap-vs-no-cap are allowed to diverge (that's the whole
    # point: the capped version stops early), but the capped result must
    # never be BELOW the true no-cap floor while both are <= cap, and
    # must never claim <= cap when the true floor is over it.
    if not verdict_no_cap:
        check(
            with_cap == no_cap,
            f"{name}: exact count matches when under the cap (no early exit fires)",
            f"no_cap={no_cap} with_cap={with_cap}",
        )
    check(
        with_cap <= max(no_cap, cap + 1),
        f"{name}: the capped floor never overstates the true floor",
        f"no_cap={no_cap} with_cap={with_cap}",
    )

# ---------------------------------------------------------------------------
# [2] the early exit really visits fewer turns on a large, over-cap backlog.
# ---------------------------------------------------------------------------
print("\n[2] the early exit skips per-message tokenizer calls once certain")

her_shape = _backlog(2200, user_words=50, assistant_words=800)  # ~4,400 turns
real_count_tokens = main.count_tokens

counter_no_cap = _CountingTokenizer(real_count_tokens)
main.count_tokens = counter_no_cap
verdict_no_cap = main._min_possible_batches(her_shape, budget)
main.count_tokens = real_count_tokens

counter_with_cap = _CountingTokenizer(real_count_tokens)
main.count_tokens = counter_with_cap
verdict_with_cap = main._min_possible_batches(her_shape, budget, cap)
main.count_tokens = real_count_tokens

check(verdict_no_cap > cap, "fixture: this backlog genuinely exceeds the cap",
      f"verdict_no_cap={verdict_no_cap}")
check(
    counter_no_cap.singleton_calls == len(her_shape),
    "no cap: every turn is priced (the old, full-walk behaviour)",
    f"singleton_calls={counter_no_cap.singleton_calls} turns={len(her_shape)}",
)
check(
    counter_with_cap.singleton_calls < counter_no_cap.singleton_calls,
    "*** THE FIX: with a cap, far fewer turns are priced before returning",
    f"with_cap calls={counter_with_cap.singleton_calls} vs "
    f"no_cap calls={counter_no_cap.singleton_calls}",
)
check(
    counter_with_cap.singleton_calls < len(her_shape) // 4,
    "the early exit fires within a small fraction of the backlog, not most of it",
    f"singleton_calls={counter_with_cap.singleton_calls} of {len(her_shape)} turns",
)

# ---------------------------------------------------------------------------
# MUTATION CHECK (documented, not auto-applied here): reverting the early
# exit -- i.e. removing the `if cap is not None and n_batches > cap: return
# n_batches` line, restoring the unconditional walk-to-the-end -- makes
# test [2]'s "far fewer turns are priced" assertion fail, because
# counter_with_cap.singleton_calls would then equal len(her_shape) exactly
# like counter_no_cap's. See CHANGELOG / the coordinator's mutation run for
# the confirmed result.
# ---------------------------------------------------------------------------

if FAILED:
    print("\nFAILED")
    sys.exit(1)
print("\nAll _min_possible_batches early-exit tests passed.")
