"""
CPU-only smoke tests for the compactor.

Exercises the pure-Python code paths (no vLLM, no GPU, no real tokenizer).
The HuggingFace tokenizer load is skipped by leaving MODEL_REPO unset, so
count_tokens falls back to the char/4 estimator.

Run inside the compactor image or any container with the requirements installed:
    python test_smoke.py
"""

import asyncio
import os
import sys

# Force the tokenizer-free fallback path before importing main
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"
os.environ["COMPACTOR_TARGET_TOKENS"] = "500"
os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "2"

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


def test_message_text_handles_multimodal():
    print("\n[test] _message_text handles multimodal content arrays")
    assert_eq(main._message_text({"content": "hello"}), "hello", "plain string")
    assert_eq(main._message_text({"content": None}), "", "None content -> empty")
    multimodal = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert_eq(main._message_text(multimodal), "a b", "list-of-parts joined")


def test_count_tokens_fallback_estimator():
    print("\n[test] count_tokens uses char/4 estimator when no tokenizer")
    msgs = [{"role": "user", "content": "hello world" * 10}]  # 110 chars
    # 110/4 + 4 overhead = 31
    assert_eq(main.count_tokens(msgs), 31, "estimator math")


def test_split_keeps_systems_separate():
    print("\n[test] split_messages preserves system msgs and isolates recent turns")
    msgs = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    system_msgs, to_summarize, keep_recent = main.split_messages(msgs)
    assert_eq(len(system_msgs), 1, "1 system message")
    assert_eq(len(to_summarize), 3, "3 older non-system msgs go to summary")
    assert_eq(len(keep_recent), 2, "KEEP_RECENT_TURNS=2 preserved verbatim")
    assert_eq(keep_recent[-1]["content"], "u3", "most-recent message is u3")


def test_split_no_summary_when_under_keep():
    print("\n[test] split_messages returns no to_summarize when conversation is short")
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    _, to_summarize, keep_recent = main.split_messages(msgs)
    assert_eq(len(to_summarize), 0, "nothing to summarize")
    assert_eq(len(keep_recent), 2, "both kept verbatim")


def test_compact_no_op_when_under_budget():
    print("\n[test] compact_if_needed is a no-op when under target budget")
    msgs = [{"role": "user", "content": "short"}]  # well under 500
    out = asyncio.run(main.compact_if_needed(msgs))
    assert_true(out is msgs, "same list returned unchanged")


def test_app_routes_registered():
    print("\n[test] FastAPI app exposes the expected OpenAI-compatible routes")
    routes = {r.path for r in main.app.routes if hasattr(r, "path")}
    assert_true("/v1/chat/completions" in routes, "/v1/chat/completions present")
    assert_true("/v1/models" in routes, "/v1/models present")
    assert_true("/health" in routes, "/health present")


def test_env_defaults_applied():
    print("\n[test] env-driven defaults computed correctly")
    assert_eq(main.MAX_MODEL_LEN, 1000, "MAX_MODEL_LEN from env")
    assert_eq(main.TARGET_TOKENS, 500, "TARGET_TOKENS from env")
    assert_eq(main.KEEP_RECENT_TURNS, 2, "KEEP_RECENT_TURNS from env")


if __name__ == "__main__":
    test_message_text_handles_multimodal()
    test_count_tokens_fallback_estimator()
    test_split_keeps_systems_separate()
    test_split_no_summary_when_under_keep()
    test_compact_no_op_when_under_budget()
    test_app_routes_registered()
    test_env_defaults_applied()
    print("\nAll smoke tests passed.")
