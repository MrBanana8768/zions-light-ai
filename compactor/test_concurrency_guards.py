"""
CPU-only Tier-1 tests for the RC5 robustness guards.

Covers the two pure helpers added after the 2026-08-10 audit:
  - _merge_touched: the lost-update fix for the async facts tail. The request
    path reads facts OUTSIDE the per-conv lock, so the tail holds a stale
    snapshot; writing it back would erase facts a concurrent tail persisted.
  - _merge_adjacent_system_messages: Mistral-family templates reject runs of
    consecutive system messages with a 400, and V1 compaction + memory
    injection can each add one independently.

Run inside the compactor image or any container with the requirements:
    python test_concurrency_guards.py
"""

import os
import sys

# Force the tokenizer-free fallback path before importing main
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"
os.environ["COMPACTOR_TARGET_TOKENS"] = "500"

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


def f(text, last_used=100, added_turn=1):
    return {"text": text, "added_turn": added_turn, "last_used": last_used}


def test_merge_touched_preserves_concurrent_writes():
    print("\n[test] _merge_touched — the lost-update scenario")
    # The exact audit scenario: our snapshot is [f1]; meanwhile another tail
    # persisted f2. Building on the snapshot would drop f2 forever.
    snapshot = [f("f1", last_used=100)]
    fresh_from_disk = [f("f1", last_used=100), f("f2", last_used=150)]
    merged = main._merge_touched(fresh_from_disk, snapshot)
    assert_eq([m["text"] for m in merged], ["f1", "f2"], "concurrent fact survives")

    print("\n[test] _merge_touched — LRU touch carries over")
    snapshot = [f("f1", last_used=999)]           # request path touched it
    fresh_from_disk = [f("f1", last_used=100)]
    merged = main._merge_touched(fresh_from_disk, snapshot)
    assert_eq(merged[0]["last_used"], 999, "newer touch applied")

    print("\n[test] _merge_touched — never backdates a fresher touch")
    snapshot = [f("f1", last_used=100)]           # our snapshot is older
    fresh_from_disk = [f("f1", last_used=500)]    # someone touched it since
    merged = main._merge_touched(fresh_from_disk, snapshot)
    assert_eq(merged[0]["last_used"], 500, "fresher last_used kept")

    print("\n[test] _merge_touched — forgotten facts stay forgotten")
    # Disk is authoritative for membership: a fact deleted via /forget while
    # the tail was in flight must NOT be resurrected from the snapshot.
    snapshot = [f("f1"), f("deleted-by-forget")]
    fresh_from_disk = [f("f1")]
    merged = main._merge_touched(fresh_from_disk, snapshot)
    assert_eq([m["text"] for m in merged], ["f1"], "deleted fact not resurrected")

    print("\n[test] _merge_touched — edge cases")
    assert_eq(main._merge_touched([], []), [], "both empty")
    assert_eq([m["text"] for m in main._merge_touched([f("a")], [])], ["a"], "empty snapshot")
    assert_eq(main._merge_touched([], [f("a")]), [], "empty disk stays empty")


def test_merge_adjacent_system_messages():
    print("\n[test] _merge_adjacent_system_messages — the Mistral 400 scenario")
    # compaction summary + injected memory block + original system prompt
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": "[Summary of earlier conversation]"},
        {"role": "system", "content": "[facts]"},
        {"role": "user", "content": "hi"},
    ]
    out = main._merge_adjacent_system_messages(msgs)
    assert_eq(len(out), 2, "three system messages collapse to one")
    assert_eq(out[0]["role"], "system", "merged message is system")
    assert_true("persona" in out[0]["content"], "first content preserved")
    assert_true("[facts]" in out[0]["content"], "last content preserved")
    assert_eq(out[1]["role"], "user", "user message untouched")

    print("\n[test] _merge_adjacent_system_messages — non-adjacent left alone")
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "b"},
    ]
    assert_eq(len(main._merge_adjacent_system_messages(msgs)), 3, "separated systems kept")

    print("\n[test] _merge_adjacent_system_messages — text-only list content is flattened")
    # rc6 review: OpenAI content-parts system prompts with NO images must be
    # flattened and merged — leaving them unmerged preserves the adjacent-
    # system run this function exists to prevent.
    msgs = [
        {"role": "system", "content": "text"},
        {"role": "system", "content": [{"type": "text", "text": "part"}]},
    ]
    out = main._merge_adjacent_system_messages(msgs)
    assert_eq(len(out), 1, "text-only parts system merged into the run")
    assert_true("part" in out[0]["content"], "flattened text preserved")

    print("\n[test] _merge_adjacent_system_messages — image content not string-joined")
    # A genuinely image-bearing system message must NOT be collapsed — that
    # would destroy the image parts (V3.1 vision).
    msgs = [
        {"role": "system", "content": "text"},
        {"role": "system", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
    ]
    out = main._merge_adjacent_system_messages(msgs)
    assert_eq(len(out), 2, "image-bearing system message preserved separately")

    print("\n[test] _merge_adjacent_system_messages — passthroughs")
    assert_eq(main._merge_adjacent_system_messages([]), [], "empty list")
    one = [{"role": "user", "content": "hi"}]
    assert_eq(main._merge_adjacent_system_messages(one), one, "single message")


if __name__ == "__main__":
    test_merge_touched_preserves_concurrent_writes()
    test_merge_adjacent_system_messages()
    print("\nAll concurrency/guard tests passed.")
