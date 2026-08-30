"""
CPU-only tests for format_summary_block's own token cap (MEMORY_REVIEW
S-1/S-6, the "other half"): even with l1/l2 now bounded by construction
(see test_l2_bound.py), format_summary_block used to render every chapter
and chunk with no ceiling of its own — "the only uncapped push layer" per
the review. Since L1_CHUNK_SIZE/L2_CHUNK_SIZE/L3_CHUNK_SIZE/*_MAX_TOKENS are
five independently-configurable env vars, a block bounded only by their
product is one misconfiguration away from being unbounded again. This
verifies the block itself now enforces SUMMARY_BLOCK_MAX_TOKENS directly,
dropping the OLDEST content first within each tier (newest-first keep) when
it doesn't fit.

Synthetic lorem-ipsum-style content only — no real conversation content.

Run: python test_summary_block_budget.py
"""

import sys

import summarizer


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


def _lorem(n_chars: int) -> str:
    unit = "lorem ipsum dolor sit amet consectetur adipiscing elit "
    return (unit * (n_chars // len(unit) + 1))[:n_chars]


def _chunk(text: str, first: int, last: int) -> dict:
    return {"text": text, "first_turn": first, "last_turn": last}


def _with_budget(tokens: int):
    """Context manager: temporarily set summarizer.SUMMARY_BLOCK_MAX_TOKENS."""
    class _Ctx:
        def __enter__(self):
            self.prev = summarizer.SUMMARY_BLOCK_MAX_TOKENS
            summarizer.SUMMARY_BLOCK_MAX_TOKENS = tokens
            return self

        def __exit__(self, *a):
            summarizer.SUMMARY_BLOCK_MAX_TOKENS = self.prev
            return False

    return _Ctx()


def test_small_state_is_unaffected_by_the_cap():
    print("\n[test] a state well under budget renders exactly as before")
    state = {
        "l1": [_chunk("scene A", 21, 24)],
        "l2": [_chunk("chapter Z", 1, 20)],
        "l3": {"text": "overall arc", "first_turn": 1, "last_turn": 100},
        "last_summarized_turn": 24,
    }
    block = summarizer.format_summary_block(state)
    assert_true("overall arc" in block, "L3 text present")
    assert_true("chapter Z" in block, "L2 text present")
    assert_true("scene A" in block, "L1 text present")
    assert_true(block.index("overall arc") < block.index("chapter Z") < block.index("scene A"),
                "render order stays L3 -> L2 -> L1")


def test_oversized_l1_keeps_the_newest_and_drops_the_oldest():
    print("\n[test] over budget: L1 is trimmed newest-first")
    # Each chunk ~ 500 chars -> a bit over 125 tokens by the ASCII/4 estimate;
    # 20 of them is comfortably over a small budget.
    l1 = [_chunk(f"scene-{i} " + _lorem(500), i, i) for i in range(20)]
    state = {"l1": l1, "l2": [], "l3": None, "last_summarized_turn": 20}
    with _with_budget(600):
        block = summarizer.format_summary_block(state)
    assert_true(block is not None, "something still fits")
    assert_true("scene-19 " in block, "the NEWEST scene survives")
    assert_true("scene-18 " in block, "and the one before it, if it fits")
    assert_true("scene-0 " not in block, "the OLDEST scene was dropped first")


def test_l3_is_kept_over_l2_when_squeezed():
    print("\n[test] L3 (cheap, whole-conversation) is the last thing dropped")
    l2 = [_chunk(f"chapter-{i} " + _lorem(2000), i, i) for i in range(10)]
    state = {
        "l1": [],
        "l2": l2,
        "l3": {"text": "the whole-conversation theme", "first_turn": 1, "last_turn": 200},
        "last_summarized_turn": 200,
    }
    # Budget large enough for L3 plus a little, but nowhere near all 10
    # oversized chapters.
    with _with_budget(400):
        block = summarizer.format_summary_block(state)
    assert_true(block is not None, "something still fits")
    assert_true("the whole-conversation theme" in block,
                "L3 survives even though most chapters do not fit")


def test_newest_chapter_survives_over_oldest_when_l2_alone_is_squeezed():
    print("\n[test] over budget: L2 is also trimmed newest-first")
    l2 = [_chunk(f"chapter-{i} " + _lorem(500), i, i) for i in range(10)]
    state = {"l1": [], "l2": l2, "l3": None, "last_summarized_turn": 10}
    with _with_budget(400):
        block = summarizer.format_summary_block(state)
    assert_true(block is not None, "something still fits")
    assert_true("chapter-9 " in block, "the newest chapter survives")
    assert_true("chapter-0 " not in block, "the oldest chapter is dropped first")


def test_the_rendered_block_never_wildly_exceeds_the_budget():
    print("\n[test] the estimated cost of what IS kept stays near the budget, not the input size")
    l1 = [_chunk(f"scene-{i} " + _lorem(3000), i, i) for i in range(15)]
    l2 = [_chunk(f"chapter-{i} " + _lorem(3000), i, i) for i in range(15)]
    state = {
        "l1": l1, "l2": l2,
        "l3": {"text": "theme " + _lorem(500), "first_turn": 1, "last_turn": 300},
        "last_summarized_turn": 300,
    }
    budget = 2000
    with _with_budget(budget):
        block = summarizer.format_summary_block(state)
    cost = summarizer._estimate_block_tokens(block)
    # Selection is greedy per-item, so the final kept set can land a little
    # under budget (an item that didn't fit is skipped, not partially kept)
    # but must never land far over it — the whole point of the cap.
    assert_true(cost <= budget * 1.05,
                f"rendered block costs ~{cost} tokens against a {budget}-token budget")
    unbounded_cost = summarizer._estimate_block_tokens(
        "\n".join(c["text"] for c in l1 + l2)
    )
    assert_true(cost < unbounded_cost,
                "the capped block is meaningfully smaller than the uncapped input")


def test_a_single_chunk_larger_than_the_whole_budget_degrades_to_none_not_a_crash():
    print("\n[test] pathological: nothing fits -> None, not an exception")
    state = {
        "l1": [_chunk(_lorem(100000), 1, 1)],
        "l2": [], "l3": None, "last_summarized_turn": 1,
    }
    with _with_budget(10):
        block = summarizer.format_summary_block(state)
    assert_eq(block, None, "degrades to no injection rather than a bare header or a crash")


def test_default_budget_exceeds_the_bounded_state_capacity():
    print(chr(10) + "[test] the default cap is a backstop, not a routine amputation")
    # This asserted 5000 - "the figure this module always claimed" - which
    # was the wrong test: it checked the constant against a docstring rather
    # than against the thing the constant has to accommodate.
    #
    # l1 and l2 are bounded by construction, so the block's worst case is
    # L2_CHUNK_SIZE-1 scenes + L3_CHUNK_SIZE-1 chapters + one theme. At
    # defaults that is 9*500 + 4*1200 + 2000 = 11,300 tokens. A 5,000 cap
    # sits BELOW it, so the cap fired in normal operation instead of as a
    # backstop: measured, above ~45% tier fill it dropped content on every
    # request, and above ~75% it dropped every L2 chapter - and since L1 is
    # selected first, chapters got only what L3 and all of L1 left over.
    # Combined with the L3 refresh consuming l2, a chapter could be created,
    # never injected, and archived without ever having been seen.
    #
    # The property is not "equals 5000". It is "leaves room for a full
    # bounded state".
    worst_case = (
        (summarizer.L2_CHUNK_SIZE - 1) * summarizer.L1_MAX_TOKENS
        + (summarizer.L3_CHUNK_SIZE - 1) * summarizer.L2_MAX_TOKENS
        + summarizer.L3_MAX_TOKENS
    )
    assert_true(
        summarizer.SUMMARY_BLOCK_MAX_TOKENS >= worst_case,
        f"the cap ({summarizer.SUMMARY_BLOCK_MAX_TOKENS}) leaves room for "
        f"a full bounded state ({worst_case}), so it backstops "
        f"misconfiguration rather than dropping real memory every request",
    )

def _all():
    return [
        test_small_state_is_unaffected_by_the_cap,
        test_oversized_l1_keeps_the_newest_and_drops_the_oldest,
        test_l3_is_kept_over_l2_when_squeezed,
        test_newest_chapter_survives_over_oldest_when_l2_alone_is_squeezed,
        test_the_rendered_block_never_wildly_exceeds_the_budget,
        test_a_single_chunk_larger_than_the_whole_budget_degrades_to_none_not_a_crash,
        test_default_budget_exceeds_the_bounded_state_capacity,
    ]


# ---------------------------------------------------------------------------
# Non-Latin script must not be priced out of the block (review F1)
# ---------------------------------------------------------------------------
# The cap's estimator used to price every non-ASCII byte as one token. Those
# characters are 2-3 UTF-8 bytes each, so Greek came out 2.34x and CJK 4.27x
# over their real cost, and a summary block of 2,823 REAL tokens - inside both
# this cap and the accurate /tokenize budget downstream - priced at 13,282 and
# was dropped in full. This user quotes scripture, so that is her entire
# summary memory silently missing from the conversations she cares most about.
#
# Every case here is deliberately NOT lorem ipsum. The original test was
# ASCII-only and would have passed with the defect present.

GREEK = "Οὕτως γὰρ ἠγάπησεν ὁ θεὸς τὸν κόσμον. "
HEBREW = "בְּרֵאשִׁית בָּרָא אֱלֹהִים. "
CJK = "你好世界我们一起学习。"


def test_non_latin_script_is_not_priced_out_of_the_block():
    print(chr(10) + "[test] a scripture-heavy block still gets injected")
    for name, sample in (("greek", GREEK), ("hebrew", HEBREW), ("cjk", CJK)):
        # ~4,000 characters: a realistic L3 theme, far inside the cap.
        body = (sample * 60)[:4000]
        est = summarizer._estimate_block_tokens(body)
        # The estimate must be a CEILING (never under-count - that is the
        # 2026-08-28 failure) without being so far above the truth that it
        # prices real memory out of the block. When the local tokenizer is
        # present the estimate IS ground truth; otherwise the fallback's
        # measured worst case across nine scripts is 3.9x, so 4x is the
        # bound that separates "pessimistic" from "will drop what fits".
        assert_true(
            est <= len(body) * 4,
            f"{name}: {len(body)} chars estimated at {est} tokens - more "
            f"than 4x the character count, which is worse than the measured "
            f"worst case. The cap will drop memory that fits.",
        )
        state = {"l1": [], "l2": [],
                 "l3": {"text": body, "first_turn": 1, "last_turn": 200}}
        block = summarizer.format_summary_block(state)
        assert_true(
            block is not None and body[:40] in block,
            f"{name}: the whole summary block was dropped for a "
            f"{len(body)}-char theme that fits the cap",
        )
        print(f"  ok   {name}: {len(body)} chars -> {est} est tokens, injected")


def test_decoration_keeps_its_pessimistic_ceiling():
    print(chr(10) + "[test] decoration is still priced pessimistically")
    # The per-byte ceiling exists FOR this: box-drawing really does cost
    # about one token per byte, and under-counting it is what
    # INCIDENT_2026-08-28 was about. Narrowing the rule to non-ASCII
    # LETTERS must not relax it here.
    rule = "━" * 400
    est = summarizer._estimate_block_tokens(rule)
    assert_true(
        est >= len(rule),
        f"decoration priced at {est} for {len(rule)} chars - below one "
        f"token per character; the ceiling that exists for decoration was "
        f"relaxed along with the one for letters",
    )
    print(f"  ok   400 box-drawing chars -> {est} est tokens (still a ceiling)")


if __name__ == "__main__":
    test_non_latin_script_is_not_priced_out_of_the_block()
    test_decoration_keeps_its_pessimistic_ceiling()
    for t in _all():
        t()
    print("\nAll summary-block-budget tests passed.")
