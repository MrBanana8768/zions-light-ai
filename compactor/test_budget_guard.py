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
os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "4"      # even — the blocker's trigger

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


def _assert_template_valid(out, label_prefix):
    """The Mistral-family invariants the rc6 review found violated: the first
    non-system message must be a USER turn, and non-system roles must
    alternate strictly."""
    roles = [m["role"] for m in out if m.get("role") != "system"]
    assert_true(roles, f"{label_prefix}: at least one non-system turn survives")
    assert_eq(roles[0], "user", f"{label_prefix}: first non-system turn is user")
    for a, b in zip(roles, roles[1:]):
        assert_true(a != b, f"{label_prefix}: roles alternate ({a}->{b})")


def test_the_production_scenario():
    print("\n[test] _enforce_hard_budget — the 2026-08-13 overflow scenario")
    # Oversized injected memory (facts+RAG+summary) on top of a long history —
    # exactly the shape that produced "at least 32769 input tokens". 11 turns:
    # user-first alternation ending on the NEW USER TURN, as real traffic does
    # (the rc6 review caught the old fixture ending on an assistant turn —
    # itself template-invalid).
    msgs = [
        {"role": "system", "content": "P" * (300 * 4)},   # persona
        {"role": "system", "content": "F" * (400 * 4)},   # injected memory
    ] + [big("user" if i % 2 == 0 else "assistant", 100) for i in range(11)]

    before = main.count_tokens(msgs)
    assert_true(before > main.HARD_INPUT_LIMIT, f"starts over budget ({before})")

    out = main._enforce_hard_budget(msgs)
    after = main.count_tokens(out)
    assert_true(after <= main.HARD_INPUT_LIMIT, f"ends within budget ({after})")

    print("\n[test] _enforce_hard_budget — the newest turn is never dropped")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "last turn preserved intact")

    print("\n[test] _enforce_hard_budget — role alternation survives shedding")
    # The rc6 review's confirmed HIGH: stopping mid-pair left the conversation
    # assistant-first, manufacturing the Mistral 400 the guard exists to stop.
    _assert_template_valid(out, "post-shed")


def test_per_request_limit():
    print("\n[test] _enforce_hard_budget — per-request limit parameter")
    # A client-requested max_tokens shrinks the effective input limit; the
    # guard must honor the caller-supplied limit, not the module constant.
    msgs = [{"role": "system", "content": "S" * (100 * 4)}] + [
        big("user" if i % 2 == 0 else "assistant", 60) for i in range(11)
    ]
    out = main._enforce_hard_budget(msgs, 300)
    assert_true(main.count_tokens(out) <= 300, "honors the tighter explicit limit")
    _assert_template_valid(out, "tight-limit")


def test_tokenization_cost_is_bounded():
    print("\n[test] _enforce_hard_budget — full-list tokenizations are O(1), not O(drops)")
    # The rc6 review's other confirmed HIGH: the old loop re-tokenized the
    # ENTIRE message list once per dropped message (O(N^2) on the event loop).
    # Now: one prescreen (no tokenizer), one entry count, per-message counts,
    # and a bounded number of verification counts.
    msgs = [{"role": "system", "content": "S" * (50 * 4)}] + [
        big("user" if i % 2 == 0 else "assistant", 40) for i in range(41)
    ]  # ~1700 tokens; needs ~25 drops to fit 800
    calls = {"full": 0}
    orig = main.count_tokens

    def counting(m):
        if len(m) > 1:
            calls["full"] += 1
        return orig(m)

    main.count_tokens = counting
    try:
        out = main._enforce_hard_budget(msgs)
    finally:
        main.count_tokens = orig
    assert_true(main.count_tokens(out) <= main.HARD_INPUT_LIMIT, "fits after shedding")
    assert_true(
        calls["full"] <= 8,
        f"full-list tokenizations bounded (got {calls['full']}, want <=8)",
    )
    _assert_template_valid(out, "many-drops")


def test_split_messages_keeps_user_first():
    print("\n[test] split_messages — compaction window starts on a USER turn (the blocker)")
    # The rc6 review's BLOCKER: with even KEEP_RECENT_TURNS (4) and a real
    # request's ODD non-system count, keep_recent always began with an
    # assistant turn — every successful compaction emitted a template-invalid
    # conversation. Latent since V1, shielded by the summarize-overflow bug.
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(7):  # u,a,u,a,u,a,u — history pairs + the new user turn
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i}"})
    system_msgs, to_summarize, keep_recent = main.split_messages(msgs)
    assert_eq(len(system_msgs), 1, "system preserved")
    assert_eq(keep_recent[0]["role"], "user", "keep window starts with a user turn")
    assert_eq(
        len(to_summarize) + len(keep_recent), 7, "no non-system turn lost in the split"
    )
    roles = [m["role"] for m in keep_recent]
    for a, b in zip(roles, roles[1:]):
        assert_true(a != b, f"keep window alternates ({a}->{b})")

    print("\n[test] split_messages — short conversations untouched")
    short = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    s, t, k = main.split_messages(short)
    assert_eq((len(s), len(t), len(k)), (1, 0, 1), "under-threshold passthrough")


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
    test_per_request_limit()
    test_tokenization_cost_is_bounded()
    test_split_messages_keeps_user_first()
    test_pathological_single_huge_turn()
    test_chunk_to_budget()
    print("\nAll budget-guard tests passed.")
