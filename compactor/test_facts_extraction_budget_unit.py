"""
A unit-mismatch regression test for facts.py's extraction-input budget.

_fit_extraction_input and everything it calls (`_assembled_tokens`,
`_estimate_tokens`, `_truncate_to_tokens`) measure exclusively in char/4
estimated tokens. Through v3.1, `_EXTRACTION_INPUT_BUDGET` was derived
straight from MAX_MODEL_LEN -- a REAL token count charged by vLLM -- and
compared directly against that estimate, bridged only by a flat 2048-token
reserve. A flat reserve cannot correct a PROPORTIONAL estimator error: no
fixed amount of slack survives a percentage under-count once the payload is
large enough.

The worst-case ratio used below (WORST_CASE_LOW_FRACTION) is MEASURED, inside
the production image, against the real tokenizer (tokens.py + mistral_common's
bundled tekken vocabulary -- the same family vLLM uses), not cited from a
document. An earlier version of this file cited "51% low" as a general figure
for this model's assistant content and built a synthetic repeated-lorem-ipsum
reproduction on it. Measuring THAT SAME synthetic text against the real
tokenizer disproved the premise: char/4 read it 53% HIGH, not low --
repetitive filler compresses hard under a real BPE vocabulary in a way char/4
cannot see. The true direction and size depend on content SHAPE:

    ordinary structured prose (paragraphs, markdown, numbers)  ~6.5% low
    prose with an occasional divider mixed in                  ~16%  HIGH
    repetitive/degenerate filler                                ~53%  HIGH
    pure box-drawing decoration                                 ~87% low

Decoration independently reproduces facts.py's own INCIDENT_2026-08-28 figure
(~87.4% low) to within rounding -- two separate measurements agreeing is used
here as the worst case, because it is the one content shape actually observed
to blow the estimate this badly, and it is exactly what reaches this budget
check unfiltered: extraction's INPUT is the raw prior turn, and
is_storable_fact's structural filter only ever runs on what extraction
OUTPUTS, never on what it is fed.

Ordinary content reading HIGH under char/4 (not low) is itself a finding: it
means direction 2 (a fact wrongly evicted or trimmed because the estimator
made it look BIGGER than it is) is not a live risk from this estimator on the
content this store actually holds -- only decoration flips the direction, and
only on this unfiltered input side.

The real fix for the common case is facts.py's tokens.count() backstop in
extract_facts_from_exchange (verifies and re-trims using the REAL tokenizer
when it is available); this budget constant is what is left standing when
that backstop cannot run at all.

All content below is synthetic lorem-ipsum filler; no production data.

    python test_facts_extraction_budget_unit.py
"""

import os
import sys

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")

import facts  # noqa: E402

# The project's own independently measured worst case (see facts.py's
# _ASSISTANT_CONTENT_ESTIMATE_LOW_FRACTION and its comment for the full
# measurement table): box-drawing decoration reads ~87% low vs. the real
# tokenizer. Used here to SIMULATE what vLLM would really charge for a
# payload the estimator scored a given size. Hardcoded rather than imported
# from facts.py so this test still means something if that constant drifts —
# but it should track the same measured worst case, not a re-guessed one.
_WORST_CASE_LOW_FRACTION = 0.87


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def simulated_worst_case_real_tokens(estimated_tokens: int) -> int:
    return int(estimated_tokens / (1 - _WORST_CASE_LOW_FRACTION))


def test_budget_is_denominated_to_survive_the_worst_case():
    print("\n[test] _EXTRACTION_INPUT_BUDGET leaves room for the documented worst case")
    # The real ceiling (MAX_MODEL_LEN minus output minus framing reserve) is a
    # REAL-token quantity; the budget this module actually enforces must be
    # smaller than that real ceiling, because it is compared against an
    # ESTIMATE that reads low, never high, on this content.
    assert_true(
        facts._EXTRACTION_INPUT_BUDGET < facts._EXTRACTION_INPUT_BUDGET_REAL_TOKENS,
        "the enforced budget is strictly less than the real-token ceiling "
        "(an estimate that reads low needs a smaller allowance, not the same one)",
    )
    assert_true(facts._EXTRACTION_INPUT_BUDGET >= 256, "the floor still holds")


def test_a_long_ordinary_reply_cannot_defeat_the_real_window():
    print("\n[test] a long, undecorated synthetic reply stays inside MAX_MODEL_LEN "
          "even under the worst documented estimator error")
    # Long, entirely ordinary content -- no box-drawing, no adversarial
    # shapes, just repeated lorem-ipsum prose. Large enough that the OLD
    # (pre-fix) budget -- the bare real-token ceiling -- would have let it
    # through with no shedding at all.
    assistant_msg = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 1600
    user_msg = "please summarize the plan so far in detail"
    existing = [
        {"text": f"established synthetic fact {i:04d} about the plot",
         "added_turn": i, "last_used": 1000 + i}
        for i in range(20)
    ]

    # Reproduce the pre-fix comparison directly: what the estimate-only check
    # against the bare real ceiling would have allowed through unshed.
    pre_fix_budget = facts._EXTRACTION_INPUT_BUDGET_REAL_TOKENS
    u_old, a_old, kept_old, note_old = facts._fit_extraction_input(
        user_msg, assistant_msg, existing, pre_fix_budget
    )
    old_estimate = facts._assembled_tokens(u_old, a_old, kept_old)
    old_worst_case_real = simulated_worst_case_real_tokens(old_estimate)
    assert_true(
        old_estimate <= pre_fix_budget,
        "prep: the pre-fix comparison would have let this payload through "
        "unshed (confirms the scenario actually exercises the gap)",
    )
    assert_true(
        old_worst_case_real > facts._MAX_MODEL_LEN,
        f"prep: that same payload's worst-case real cost ({old_worst_case_real}) "
        f"exceeds MAX_MODEL_LEN ({facts._MAX_MODEL_LEN}) -- the defect this "
        f"fix closes",
    )

    # Now the actual, current behaviour with the real (fixed) budget.
    u, a, kept, note = facts._fit_extraction_input(
        user_msg, assistant_msg, existing, facts._EXTRACTION_INPUT_BUDGET
    )
    estimate = facts._assembled_tokens(u, a, kept)
    worst_case_real = simulated_worst_case_real_tokens(estimate)
    assert_true(
        estimate <= facts._EXTRACTION_INPUT_BUDGET,
        "the trimmed payload still fits the budget it was given",
    )
    assert_true(
        worst_case_real <= facts._MAX_MODEL_LEN,
        f"the SAME payload's worst-case real cost ({worst_case_real}) now "
        f"fits inside MAX_MODEL_LEN ({facts._MAX_MODEL_LEN})",
    )


def test_a_small_ordinary_exchange_is_untouched():
    print("\n[test] a small, everyday exchange is not needlessly shed")
    # Guards against an over-correction: a fix that shrinks the budget to
    # near-zero would "pass" the tests above by shedding everything, which is
    # its own regression (extraction quality, not safety). A short, normal
    # exchange must still reach the model whole.
    user_msg = "Her sister is named Isolde and runs the mill at Varrow Ford."
    assistant_msg = "Isolde keeps the mill turning while the protagonist is away."
    existing = [
        {"text": "The protagonist carries a synthetic yew bow.",
         "added_turn": 1, "last_used": 1},
    ]
    u, a, kept, note = facts._fit_extraction_input(
        user_msg, assistant_msg, existing, facts._EXTRACTION_INPUT_BUDGET
    )
    assert_true(note is None, "nothing needed shedding for an ordinary short exchange")
    assert_true(u == user_msg, "the user turn is untouched")
    assert_true(a == assistant_msg, "the assistant reply is untouched")
    assert_true(len(kept) == len(existing), "the whole small store is untouched")


# ---------------------------------------------------------------------------
# The tokens.count() real-measurement backstop in extract_facts_from_exchange
# ---------------------------------------------------------------------------
#
# tokens.count() is unavailable in this test environment (no cached
# tekken.json, mistral_common not installed) exactly as it would be on a pod
# whose model is not cached yet or is not a Mistral model — tokens.py's own
# "everything degrades to None" doctrine. So these tests mock tokens.count
# directly to exercise the backstop's three outcomes without needing the real
# tokenizer file.

import asyncio  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402


def _mock_client_returning(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": content}}]})
    client = MagicMock()
    client.post = AsyncMock(return_value=mock_response)
    return client


def _sent_messages(client) -> list[dict]:
    return client.post.call_args.kwargs["json"]["messages"]


def test_backstop_retrims_when_the_real_tokenizer_disagrees():
    print("\n[test] backstop: a real-measured overage triggers a re-trim, not a blind send")
    client = _mock_client_returning("NONE")
    # First call: "real" tokenizer reports far over the real ceiling. Second
    # call (after the backstop shrinks the budget and re-trims): reports
    # comfortably under. tokens.count is called once per _retry iteration.
    over = facts._EXTRACTION_INPUT_BUDGET_REAL_TOKENS * 3
    under = facts._EXTRACTION_INPUT_BUDGET_REAL_TOKENS // 2
    with patch.object(facts.tokens, "count", side_effect=[over, under]) as mock_count:
        asyncio.run(facts.extract_facts_from_exchange(
            client, "http://fake", "fake-model",
            "a synthetic user turn about the plot",
            "a synthetic assistant reply " * 50,
            [], conv_id="backstop",
        ))
    assert_true(mock_count.call_count == 2, "measured once, disagreed, re-trimmed, measured again")
    assert_true(client.post.called, "the call still went out after converging")


def test_backstop_is_a_no_op_when_the_real_tokenizer_is_unavailable():
    print("\n[test] backstop: tokens.count()->None leaves the estimate-based trim untouched")
    # This is the actual CI/test-environment behaviour (no cached tekken.json)
    # and must be behaviourally IDENTICAL to calling _fit_extraction_input
    # alone — the backstop must add nothing when it cannot measure anything.
    #
    # "Unavailable" has to be faked on BOTH signals the production code
    # actually reads, not just tokens.count(). extract_facts_from_exchange
    # picks its STARTING budget from tokens.is_available() (generous when
    # True, conservative when False — see
    # facts._extraction_input_budget_estimate_units), and only THEN runs the
    # backstop against tokens.count(). On a box with a real tekken.json
    # staged (this repo's reproduction of the pod — see tokens.py's module
    # docstring), is_available() is truthfully True even while this test
    # mocks count() to return None, so patching count() alone leaves the
    # module picking the GENEROUS starting budget — a real, reachable
    # divergence (the tokenizer loaded fine but THIS measurement failed) —
    # while the "baseline" below is computed at the CONSERVATIVE constant,
    # comparing two different starting budgets and failing on every box that
    # actually has a cached tokenizer. Patch both so the scenario means what
    # its own name says regardless of the host's cache.
    client_with_backstop = _mock_client_returning("NONE")
    client_baseline = _mock_client_returning("NONE")
    user_msg = "a synthetic user turn"
    assistant_msg = "a synthetic assistant reply " * 2000  # large enough to need shedding
    existing = [{"text": f"synthetic fact {i}", "added_turn": i, "last_used": i} for i in range(30)]

    with patch.object(facts.tokens, "is_available", return_value=False), \
         patch.object(facts.tokens, "count", return_value=None) as mock_count:
        asyncio.run(facts.extract_facts_from_exchange(
            client_with_backstop, "http://fake", "fake-model", user_msg, assistant_msg,
            list(existing), conv_id="backstop-none",
        ))
    assert_true(mock_count.called, "the backstop still tries to measure once")

    # Baseline: the plain pre-backstop code path (_fit_extraction_input at the
    # module's default budget), assembled the same way extract_facts_from_exchange
    # itself assembles a payload.
    u, a, kept, _ = facts._fit_extraction_input(
        user_msg, assistant_msg, list(existing), facts._EXTRACTION_INPUT_BUDGET
    )
    baseline_messages = facts._build_extraction_messages(u, a, kept)

    assert_eq(
        _sent_messages(client_with_backstop), baseline_messages,
        "identical payload with or without the backstop when tokens.count() is None",
    )


def test_generous_start_when_real_tokenizer_available_avoids_needless_trim():
    print("\n[test] a real tokenizer available means ordinary long content is not pre-emptively shed")
    # Measured regression this guards: an earlier version of this fix always
    # started _fit_extraction_input from the conservative, worst-case-
    # calibrated _EXTRACTION_INPUT_BUDGET, even when the real tokenizer was
    # available to verify the result afterward. Since the backstop above
    # only ever SHRINKS on disagreement and never widens back up, that
    # pessimistic starting point survived untouched -- cutting an ordinary,
    # undecorated long reply down to roughly 15% of its real size for no
    # reason, confirmed against the real Mistral tokenizer inside the
    # production image. The fix: start from the GENEROUS real ceiling
    # whenever tokens.is_available() is True, and let the backstop -- not
    # this starting guess -- be what keeps that safe.
    client = _mock_client_returning("NONE")
    user_msg = "a synthetic user turn"
    # Long, ordinary, non-decorated synthetic content -- ~21,000 chars, the
    # "long" size from the measured trimming-impact comparison.
    assistant_msg = "an ordinary long synthetic reply about the plan " * 430
    existing = [{"text": f"synthetic fact {i}", "added_turn": i, "last_used": i} for i in range(50)]

    # A generous, roughly-accurate real count for this ordinary content
    # (char/4 reads close to accurate on undecorated prose -- see the
    # measured table in facts.py's calibration comment) — comfortably under
    # the real ceiling, so a correctly-generous start sheds nothing at all.
    def realistic_real_count(messages):
        content = "".join(m.get("content", "") for m in messages)
        return facts._estimate_tokens(content)

    with patch.object(facts.tokens, "is_available", return_value=True), \
         patch.object(facts.tokens, "count", side_effect=realistic_real_count):
        asyncio.run(facts.extract_facts_from_exchange(
            client, "http://fake", "fake-model", user_msg, assistant_msg, list(existing),
            conv_id="generous-start",
        ))
    sent = _sent_messages(client)
    sent_content = "".join(m.get("content", "") for m in sent)
    assert_true(
        assistant_msg in sent_content,
        "the ordinary long reply reached the model whole — nothing was shed "
        "that the real tokenizer confirmed was unnecessary",
    )
    assert_true(facts._TRIM_NOTE not in sent_content, "no trim note — nothing needed trimming")


def test_backstop_gives_up_and_warns_after_the_retry_cap():
    print("\n[test] backstop: a real tokenizer that never agrees gets a warning, not a hang")
    client = _mock_client_returning("NONE")
    # Always over budget, no matter how much gets shed -- the pathological
    # case the retry cap exists for.
    always_over = facts._EXTRACTION_INPUT_BUDGET_REAL_TOKENS * 5
    with patch.object(facts.tokens, "count", return_value=always_over) as mock_count, \
         patch.object(facts.logger, "warning") as warn:
        asyncio.run(facts.extract_facts_from_exchange(
            client, "http://fake", "fake-model",
            "a synthetic user turn", "a synthetic assistant reply " * 50,
            [], conv_id="backstop-stuck",
        ))
    assert_eq(
        mock_count.call_count, facts._MAX_REAL_TOKEN_MEASURE_RETRIES,
        "stopped at the retry cap rather than looping forever",
    )
    assert_true(warn.called, "gave up with a WARNING rather than silently sending the last attempt")
    assert_true(client.post.called, "the call still went out (best effort beats holding it forever)")


def test_backstop_converges_on_a_severe_density_mismatch():
    print("\n[test] backstop converges when real tokens run far off char/4 throughout (decoration-shaped)")
    # Reproduces the actual bug this fix closed, not a canned scenario: an
    # EARLIER version of the backstop shrunk the estimate-unit budget by
    # (ceiling / real) applied to ITSELF each retry. For content whose real
    # density is far from char/4's assumption throughout (measured: ~7.9x on
    # box-drawing), that shrink lands the new budget still far ABOVE the
    # content's own (tiny) char/4 estimate, so _fit_extraction_input's "does
    # it already fit?" check keeps saying yes, nothing gets shed, `real`
    # never moves, and the retry cap burns out with the request still over
    # budget -- silently, even with a real tokenizer wired in. The fix
    # derives the new target from the OBSERVED density of what was just
    # measured instead of re-scaling the old budget.
    DENSITY = 7.0  # close to the measured box-drawing ratio (~7.9x)
    client = _mock_client_returning("NONE")

    def fake_count(messages):
        content = "".join(m.get("content", "") for m in messages)
        return int(facts._estimate_tokens(content) * DENSITY)

    # Sized so its OWN char/4 estimate (~5500) sits well under the real
    # ceiling, matching the real decoration measurement's proportions: the
    # old formula's first shrink (ceiling * ceiling/real * 0.9) lands well
    # ABOVE an estimate this small relative to the ceiling, so it never
    # triggers shedding at all -- a larger estimate here would let the old
    # formula converge by coincidence and this test would not discriminate.
    assistant_msg = "synthetic decoration-shaped filler content " * 500
    # is_available() must read True here too: the bug this guards only shows
    # up when extract_facts_from_exchange starts from the GENEROUS budget
    # (real tokenizer available, see _extraction_input_budget_estimate_units)
    # and relies entirely on this loop to correct it. In an environment where
    # is_available() is already False (no mistral_common installed, as in
    # this test's own process), the FALLBACK budget alone would shed enough
    # up front that this loop's formula would never even get exercised.
    with patch.object(facts.tokens, "is_available", return_value=True), \
         patch.object(facts.tokens, "count", side_effect=fake_count) as mock_count:
        asyncio.run(facts.extract_facts_from_exchange(
            client, "http://fake", "fake-model",
            "a synthetic user turn", assistant_msg, [], conv_id="density-mismatch",
        ))
    real_sent = fake_count(_sent_messages(client))
    assert_true(
        real_sent <= facts._EXTRACTION_INPUT_BUDGET_REAL_TOKENS,
        f"converged under the real ceiling ({real_sent} <= "
        f"{facts._EXTRACTION_INPUT_BUDGET_REAL_TOKENS}) within "
        f"{facts._MAX_REAL_TOKEN_MEASURE_RETRIES} retries, at a {DENSITY}x "
        f"density mismatch",
    )
    assert_true(
        mock_count.call_count <= facts._MAX_REAL_TOKEN_MEASURE_RETRIES,
        "converged within the retry cap, not by exhausting it and giving up",
    )


if __name__ == "__main__":
    test_budget_is_denominated_to_survive_the_worst_case()
    test_a_long_ordinary_reply_cannot_defeat_the_real_window()
    test_a_small_ordinary_exchange_is_untouched()
    test_backstop_retrims_when_the_real_tokenizer_disagrees()
    test_backstop_is_a_no_op_when_the_real_tokenizer_is_unavailable()
    test_generous_start_when_real_tokenizer_available_avoids_needless_trim()
    test_backstop_gives_up_and_warns_after_the_retry_cap()
    test_backstop_converges_on_a_severe_density_mismatch()
    print("\nAll extraction-budget unit tests passed.")
