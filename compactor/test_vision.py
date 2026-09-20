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
import base64
import os
import struct
import sys
import zlib

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
# Token accounting once opencv lets the REAL chat template succeed (v3.1.9.3)
#
# The tests above cover count_tokens() with NO tokenizer (char/4). These
# cover the tokenizer-present, template-SUCCEEDS branch, which is what
# changes once compactor/requirements.txt installs opencv-python-headless
# (mistral_common can then tokenize an actual image; see that file's
# comment and SP\fix-3193-opencv.md for the measurement this fix is built
# from). No network, no real mistral_common, no GPU: a fake tokenizer class
# (this file's `FramingTokenizer`-style pattern, matching
# test_budget_guard.py) reproduces two things measured against the REAL
# served model's tokenizer files, so the test is honest about what
# production actually does rather than an idealized encode():
#   1. the template renders each image as real marker tokens whose TOTAL
#      COUNT matches vLLM's own usage.prompt_tokens exactly, at every size
#      scripts/probe-vision.py measures (110/380/1406/3080 at
#      256/512/1024/2048px);
#   2. re-encoding that rendered STRING (what count_tokens() does,
#      tokenize=False then encode()) does NOT price those markers at the 1
#      token each they really are in the vocabulary -- it costs 4/7/5 raw
#      BPE tokens per [IMG]/[IMG_BREAK]/[IMG_END] occurrence instead,
#      measured against the real model (SP\fix-3193-opencv.md,
#      opencv_marker_check.log). count_tokens()'s fix has to correct for
#      BOTH or the tolerance check below would pass for the wrong reason.
# ---------------------------------------------------------------------------

# scripts/probe-vision.py's own generator: a gradient PNG, not a flat fill,
# so a real width/height round-trips through the file's own IHDR chunk.
# Reproduced here (not imported) because that script talks to a live vLLM
# over HTTP and this file never does.
def _synthetic_png(w: int, h: int) -> bytes:
    raw = b""
    for y in range(h):
        row = bytes([(x * 7 + y * 3) % 256 for x in range(w) for _ in (0, 1, 2)])
        raw += b"\x00" + row

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _png_dims(data: bytes) -> tuple[int, int]:
    """Width/height from a PNG's IHDR chunk -- same field probe-vision.py's
    own identify() reads, simplified for this file's own generator only."""
    return struct.unpack(">II", data[16:24])


def _image_msg_with_real_png(size: int) -> dict:
    """A user message carrying an actual synthetic PNG at `size`x`size`, not
    a 4-byte placeholder -- so the fake tokenizer below determines the image's
    cost from the image's own real dimensions, the way the real pipeline
    does, rather than a side-channel test hint."""
    png = _synthetic_png(size, size)
    b64 = base64.b64encode(png).decode()
    return {"role": "user", "content": [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}


class VisionTemplateTokenizer:
    """A tokenizer that renders and (mis)encodes images the way the REAL
    served model's does, once opencv lets its chat template run (measured
    against the actual tokenizer files -- SP\\fix-3193-opencv.md). Plain text
    keeps this file's usual 4-chars-per-token convention."""

    # measured, scripts/probe-vision.py: real per-image cost (vLLM's own
    # usage.prompt_tokens), at the sizes it probes.
    REAL_COST = {256: 110, 512: 380, 1024: 1406, 2048: 3080}
    # measured, this exact model (SP\fix-3193-opencv.md,
    # opencv_marker_check.log): raw BPE tokens a plain encode() call assigns
    # to one occurrence of each marker, instead of the 1 it really is.
    RAW_MARKER_COST = {"[IMG]": 4, "[IMG_BREAK]": 7, "[IMG_END]": 5}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                             continue_final_message=False):
        # v3195-main M3: count_tokens now always passes
        # continue_final_message (previously it never did at all). Accepted
        # and otherwise ignored here: this double's rendering never
        # depended on it.
        parts = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif p.get("type") == "image_url":
                        url = p.get("image_url", {}).get("url", "")
                        b64 = url.split(",", 1)[1] if "," in url else ""
                        w, h = _png_dims(base64.b64decode(b64)) if b64 else (0, 0)
                        n = self.REAL_COST.get(w, 0)
                        # n-1 [IMG] + one [IMG_END] -- matches the real
                        # template's shape (exactly one END per image,
                        # verified at every size tested).
                        parts.append("[IMG]" * max(0, n - 1) + ("[IMG_END]" if n else ""))
            else:
                parts.append(content or "")
        return " ".join(parts)

    def encode(self, text):
        n = 0
        rest = text
        for marker, raw_cost in self.RAW_MARKER_COST.items():
            count = rest.count(marker)
            n += count * raw_cost
            rest = rest.replace(marker, "")
        n += len(rest) // 4
        return list(range(n))


class RaisingTokenizer:
    """Simulates tier 1 failing -- e.g. opencv genuinely absent, ImportError
    on an image-bearing apply_chat_template call. encode() alone (used by
    the except branch's per-message loop) still works, matching what
    get_tokenizer() actually returns: a real, loaded tokenizer whose
    apply_chat_template call is what fails, not the object itself."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        raise ImportError("`opencv` is not installed. Please install it with `pip install mistral-common[opencv]`")

    def encode(self, text):
        return list(range(len(text) // 4))


def test_count_tokens_prices_images_at_real_cost_once_template_succeeds():
    print("\n[test] count_tokens: template success prices images near their REAL cost")
    orig_tok = main._tokenizer
    try:
        main._tokenizer = VisionTemplateTokenizer()
        for size, real_cost in VisionTemplateTokenizer.REAL_COST.items():
            msgs = [_image_msg_with_real_png(size)]
            counted = main.count_tokens(msgs)
            # Tolerance: a handful of tokens for the "look at this" text plus
            # template framing (baseline, no image, under this same stub) --
            # NOT a percentage of real_cost, because the point of this fix is
            # that the image portion is no longer priced by a multiplier at
            # all. Stated explicitly: within TEXT_MARGIN of real_cost, where
            # TEXT_MARGIN is generous enough for the surrounding text/framing
            # and small enough that a return of the 2-2.3x inflation (or the
            # double-count) blows through it by thousands of tokens.
            TEXT_MARGIN = 20
            assert_true(
                real_cost <= counted <= real_cost + TEXT_MARGIN,
                f"{size}px: counted={counted}, real_cost={real_cost} "
                f"(want within +{TEXT_MARGIN} of real, never under)",
            )
            # And explicitly NOT anywhere near the flat estimate stacked on
            # top, or the un-corrected re-encode inflation -- the two
            # regressions this fix exists to prevent.
            image_tokens_flat = main.IMAGE_TOKEN_ESTIMATE
            assert_true(
                counted < real_cost + image_tokens_flat,
                f"{size}px: {counted} must not still include the flat "
                f"{image_tokens_flat}-token estimate on top of a priced image",
            )
    finally:
        main._tokenizer = orig_tok


def test_count_tokens_text_only_unchanged_with_template():
    print("\n[test] count_tokens: text-only list is IDENTICAL with the template present")
    orig_tok = main._tokenizer
    try:
        main._tokenizer = VisionTemplateTokenizer()
        msgs = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello there, how are you today"},
        ]
        counted = main.count_tokens(msgs)
        # Manually what the OLD (and still current, for text-only) formula
        # computes: len(encode(apply_chat_template(msgs))) -- no markers
        # present, so the fix's branch takes the untouched `else` path.
        tok = main._tokenizer
        expected = len(tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
        assert_eq(counted, expected, "text-only count via the template path is unchanged by this fix")
    finally:
        main._tokenizer = orig_tok


def test_count_tokens_template_failure_unchanged():
    print("\n[test] count_tokens: template FAILING still charges exactly today's flat estimate")
    orig_tok = main._tokenizer
    try:
        main._tokenizer = RaisingTokenizer()
        msgs = [_image_msg_with_real_png(1024)]
        counted = main.count_tokens(msgs)
        tok = main._tokenizer
        n_images = sum(main._message_image_count(m) for m in msgs)
        expected = (
            sum(len(tok.encode(main._message_text(m))) + 4 for m in msgs)
            + n_images * main.IMAGE_TOKEN_ESTIMATE
        )
        assert_eq(counted, expected,
                   "except-branch (template failing) formula is byte-for-byte unchanged")
    finally:
        main._tokenizer = orig_tok


def test_count_tokens_literal_img_end_text_not_treated_as_a_marker():
    print("\n[test] count_tokens: a TEXT mention of '[IMG_END]' with no real image is not stripped")
    orig_tok = main._tokenizer
    try:
        main._tokenizer = VisionTemplateTokenizer()
        # No image_url part anywhere -- n_images is 0 -- but the text itself
        # names the literal marker (e.g. someone discussing this very fix).
        # It must be encoded like any other text, not stripped out and
        # replaced with a single "priced marker" token.
        msgs = [{"role": "user", "content": "what does [IMG_END] mean here?"}]
        counted = main.count_tokens(msgs)
        tok = main._tokenizer
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        expected = len(tok.encode(rendered))  # the untouched, un-stripped encode
        assert_eq(counted, expected,
                   "a literal '[IMG_END]' in ordinary text is encoded normally, not stripped")
    finally:
        main._tokenizer = orig_tok


# ---------------------------------------------------------------------------
# Compaction preserves image turns
# ---------------------------------------------------------------------------

def test_compaction_preserves_image_turns():
    print("\n[test] compact_if_needed keeps image turns verbatim, summarizes text")

    async def fake_summarize(client, to_summarize):
        # (summary, deferred). v3.1.3 bounded the summarization calls one
        # request may make and returns the turns it chose NOT to summarize, so
        # the caller can forward them verbatim. A stub returning a bare string
        # unpacks into its characters and fails with 'too many values'.
        return "SUMMARY", []

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
        test_count_tokens_prices_images_at_real_cost_once_template_succeeds,
        test_count_tokens_text_only_unchanged_with_template,
        test_count_tokens_template_failure_unchanged,
        test_count_tokens_literal_img_end_text_not_treated_as_a_marker,
        test_compaction_preserves_image_turns,
        test_compaction_all_images_kept_unchanged,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll vision smoke tests passed.")
