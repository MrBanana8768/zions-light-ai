"""
CPU-only Tier-1 tests for V3.1 (Vision) compactor handling.

Verifies the two things that matter when a vision-language model is in play:
  1. count_tokens accounts for image token cost (so VLM budgets don't
     silently overflow the real context window).
  2. compact_if_needed PRESERVES image-bearing turns verbatim instead of
     summarizing them to text (which would destroy the image forever).

No GPU, no real model — summarize() is mocked. Run: python test_vision.py
"""

import asyncio
import os
import sys

# Force the char/4 estimator (no tokenizer) + a small budget so compaction
# triggers, with a known per-image token cost. Set before importing main.
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"
os.environ["COMPACTOR_TARGET_TOKENS"] = "200"
os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "2"
os.environ["COMPACTOR_IMAGE_TOKENS"] = "100"

import main  # noqa: E402


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _img_msg(text, marker=""):
    """A user message with text + one image part (OpenAI multimodal shape)."""
    return {"role": "user", "content": [
        {"type": "text", "text": (text + " " + marker).strip()},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_image_count_and_has_image():
    print("\n[test] _message_image_count / _message_has_image")
    assert_eq(main._message_image_count({"content": "plain"}), 0, "plain str -> 0")
    assert_eq(main._message_image_count({"content": None}), 0, "None -> 0")
    text_only = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert_eq(main._message_image_count(text_only), 0, "text-only parts -> 0")
    one_img = {"content": [{"type": "text", "text": "q"},
                           {"type": "image_url", "image_url": {"url": "x"}}]}
    assert_eq(main._message_image_count(one_img), 1, "one image part -> 1")
    two_img = {"content": [{"type": "image_url", "image_url": {"url": "x"}},
                           {"type": "image_url", "image_url": {"url": "y"}}]}
    assert_eq(main._message_image_count(two_img), 2, "two image parts -> 2")
    assert_eq(main._message_has_image(one_img), True, "has_image True")
    assert_eq(main._message_has_image(text_only), False, "has_image False")


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

def test_count_tokens_adds_image_cost():
    print("\n[test] count_tokens adds per-image token estimate")
    text_only = [{"role": "user", "content": "hello world"}]  # 11 chars -> 11//4+4 = 6
    assert_eq(main.count_tokens(text_only), 6, "text-only baseline")
    with_img = [{"role": "user", "content": [
        {"type": "text", "text": "hello world"},
        {"type": "image_url", "image_url": {"url": "x"}},
    ]}]
    # text view is "hello world " (12 chars — the image part joins as an
    # empty string + separator) -> 12//4+4 = 7, plus 1 image * 100 = 107
    assert_eq(main.count_tokens(with_img), 107, "one image adds IMAGE_TOKEN_ESTIMATE")
    two = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "image_url", "image_url": {"url": "y"}},
    ]}]
    # text view "" -> 0//4+4 = 4, + 2*100 = 204
    assert_eq(main.count_tokens(two), 204, "two images add 2x estimate")


# ---------------------------------------------------------------------------
# Compaction preserves image turns
# ---------------------------------------------------------------------------

def test_compaction_preserves_image_turns():
    print("\n[test] compact_if_needed keeps image turns verbatim, summarizes text")

    async def fake_summarize(client, to_summarize):
        return "SUMMARY"

    orig = main.summarize
    main.summarize = fake_summarize
    try:
        long = "x" * 400  # ~104 tokens each, forces over-budget
        img = _img_msg("describe this", marker="IMG-MARKER")
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": long},        # t1  (older, text)
            {"role": "assistant", "content": long},   # t2
            {"role": "user", "content": long},        # t3
            img,                                      # older image turn
            {"role": "user", "content": "recent-1"},  # keep_recent
            {"role": "assistant", "content": "recent-2"},
        ]
        out = asyncio.run(main.compact_if_needed(msgs))
    finally:
        main.summarize = orig

    # Order: system → summary → image turn → recent(2)
    assert_eq(len(out), 5, "5 messages after compaction")
    assert_eq(out[0]["content"], "sys", "system preserved first")
    assert_true(out[1]["content"].startswith("[Summary of earlier conversation]"),
                "summary block second")
    assert_true("SUMMARY" in out[1]["content"], "summary text present")
    assert_true(main._message_has_image(out[2]), "image turn preserved verbatim (3rd)")
    assert_true("IMG-MARKER" in main._message_text(out[2]), "the right image turn")
    assert_eq(out[3]["content"], "recent-1", "recent-1 kept")
    assert_eq(out[4]["content"], "recent-2", "recent-2 kept")
    # The long text turns must NOT survive verbatim
    assert_true(all(m.get("content") != long for m in out), "long text turns summarized away")


def test_compaction_all_images_kept_unchanged():
    print("\n[test] compact_if_needed: all older turns are images -> kept verbatim, no summary")

    async def boom_summarize(client, to_summarize):
        raise AssertionError("summarize must not be called when nothing is text-only")

    orig = main.summarize
    main.summarize = boom_summarize
    try:
        msgs = [
            {"role": "system", "content": "sys"},
            _img_msg("a", "I1"),
            _img_msg("b", "I2"),
            _img_msg("c", "I3"),   # 3 images * 100 = 300 tokens > TARGET 200
            {"role": "user", "content": "recent-1"},
            {"role": "assistant", "content": "recent-2"},
        ]
        out = asyncio.run(main.compact_if_needed(msgs))
    finally:
        main.summarize = orig

    assert_true(out is msgs, "original list returned unchanged (nothing summarizable)")


def _all():
    return [
        test_image_count_and_has_image,
        test_count_tokens_adds_image_cost,
        test_compaction_preserves_image_turns,
        test_compaction_all_images_kept_unchanged,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll vision smoke tests passed.")
