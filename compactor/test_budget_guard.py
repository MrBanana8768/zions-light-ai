"""
CPU-only Tier-1 tests for the context-budget guards.

Regression cover for the 2026-08-13 production failure, where the component
whose job is to keep requests inside the context window overflowed it:

  1. A long conversation packed EVERY older turn into one summarization prompt
     -> that call exceeded MAX_MODEL_LEN -> 400.
  2. Compaction caught the 400 and "degraded" by forwarding the ORIGINAL
     oversized messages.
  3. Memory injection then added 100 facts + retrieved exchanges on top.
  4. The real chat request 400'd: "maximum context length is 32768 tokens...
     your prompt contains at least 32769 input tokens".

So: _chunk_to_budget bounds the summarizer's input, and _enforce_hard_budget is
the final pre-flight that guarantees what we forward can actually be served.

Run inside the compactor image or any container with the requirements:
    python test_budget_guard.py
"""

import os
import sys

# Small, predictable window. No MODEL_REPO -> char/4 token estimator.
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"
os.environ["COMPACTOR_GENERATION_RESERVE"] = "200"   # HARD_INPUT_LIMIT = 800
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "100"
os.environ["COMPACTOR_SUMMARY_INPUT_RESERVE"] = "100"

import main  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def user(text):
    return {"role": "user", "content": text}


def big(role, approx_tokens):
    # char/4 estimator -> 4 chars ~= 1 token
    return {"role": role, "content": "x" * (approx_tokens * 4)}


def test_hard_limit_configured():
    print("\n[test] hard budget config")
    assert_eq(main.HARD_INPUT_LIMIT, 800, "HARD_INPUT_LIMIT = MAX_MODEL_LEN - reserve")


def test_under_budget_is_untouched():
    print("\n[test] _enforce_hard_budget — under budget passes through unchanged")
    msgs = [{"role": "system", "content": "persona"}, user("hi")]
    out = main._enforce_hard_budget(msgs)
    assert_eq(out, msgs, "identical list returned")


def test_the_production_scenario():
    print("\n[test] _enforce_hard_budget — the 2026-08-13 overflow scenario")
    # Oversized injected memory (facts+RAG+summary) on top of a long history —
    # exactly the shape that produced "at least 32769 input tokens".
    msgs = [
        {"role": "system", "content": "P" * (300 * 4)},   # persona
        {"role": "system", "content": "F" * (400 * 4)},   # injected memory
    ] + [big("user" if i % 2 == 0 else "assistant", 100) for i in range(10)]

    before = main.count_tokens(msgs)
    assert_true(before > main.HARD_INPUT_LIMIT, f"starts over budget ({before})")

    out = main._enforce_hard_budget(msgs)
    after = main.count_tokens(out)
    assert_true(after <= main.HARD_INPUT_LIMIT, f"ends within budget ({after})")

    print("\n[test] _enforce_hard_budget — the newest turn is never dropped")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "last turn preserved intact")


def test_pathological_single_huge_turn():
    print("\n[test] _enforce_hard_budget — one enormous turn still terminates")
    # A single user message larger than the whole window. We cannot drop it
    # (it is the newest turn), so the guard must trim blocks, give up, and
    # RETURN rather than spin forever.
    msgs = [
        {"role": "system", "content": "S" * (900 * 4)},
        big("user", 2000),
    ]
    out = main._enforce_hard_budget(msgs)
    assert_true(isinstance(out, list) and len(out) >= 1, "returned a list, did not hang")
    assert_true(out[-1]["role"] == "user", "the user's own turn survives")


def test_chunk_to_budget():
    print("\n[test] _chunk_to_budget — splits oversized input for the summarizer")
    turns = [big("user", 100) for _ in range(10)]   # ~1000 tokens total
    batches = main._chunk_to_budget(turns, 300)
    assert_true(len(batches) > 1, f"split into multiple batches ({len(batches)})")
    assert_eq(sum(len(b) for b in batches), 10, "no turn lost across batches")
    for b in batches:
        # Each batch fits, unless it is a single oversized turn (allowed).
        assert_true(
            main.count_tokens(b) <= 300 or len(b) == 1,
            "batch within budget (or a lone oversized turn)",
        )

    print("\n[test] _chunk_to_budget — small input stays one batch")
    assert_eq(len(main._chunk_to_budget([user("hi"), user("there")], 1000)), 1, "single batch")

    print("\n[test] _chunk_to_budget — a lone oversized turn is not dropped")
    batches = main._chunk_to_budget([big("user", 5000)], 100)
    assert_eq(len(batches), 1, "one batch")
    assert_eq(len(batches[0]), 1, "the turn is kept, not silently discarded")


if __name__ == "__main__":
    test_hard_limit_configured()
    test_under_budget_is_untouched()
    test_the_production_scenario()
    test_pathological_single_huge_turn()
    test_chunk_to_budget()
    print("\nAll budget-guard tests passed.")
