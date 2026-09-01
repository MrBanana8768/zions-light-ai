"""
CPU-only Tier-1 tests for the four injected block headers.

These four strings are the only prompt text this service writes itself, and
they go into every single request the user's companion answers. They have no
unit behaviour to test, so nothing else in the suite notices when one
changes — which is how the v3.1.5 defect survived: from V2.0 to v3.1.4 all
four headers asked for CONSISTENCY in one wording or another, and by
2026-08-31 the model was reading ~91 fact bullets plus up to 1500 tokens of
its own verbatim past replies under those instructions every turn. The user
reported the replies had gone formulaic. The 08-29 degeneration detector was
silent throughout, because this was not degeneration: the prompt was asking
for the sameness and getting it.

THE DESIGN RULE (stated in full at persona.py's _PERSONA_BLOCK_HEADER):
each block claims exactly one kind of authority.

    persona    — identity and VOICE.  The ONLY block that governs how she
                 speaks.
    facts      — what is TRUE.
    retrieval  — what was SAID.
    summary    — what HAPPENED.

WHY THIS FILE PINS EXACT STRINGS. There is no assertion that captures "this
paragraph does not nudge a language model toward repeating itself" — that is
a judgement, and pretending otherwise would buy a test that passes while the
property it names is false. So this pins the bytes instead, which buys the
one thing that IS mechanically enforceable and is what actually failed here:
no one can change what ships into her prompt without the diff landing in
front of a reviewer alongside the rule above. Re-pin deliberately; do not
loosen to a substring match.

The budget bounds below are ordinary numeric invariants and are not pins.

Run: python test_block_headers.py
"""

import os
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_headers_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import facts  # noqa: E402
import persona  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}:\n  expected {expected!r}\n  got      {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


# ---------------------------------------------------------------------------
# The pins
# ---------------------------------------------------------------------------


def test_persona_header_owns_the_voice():
    print("\n[test] persona header — the one block that governs voice")
    assert_eq(
        persona._PERSONA_BLOCK_HEADER,
        "[Persona / role context for this conversation — this is where your "
        "identity and voice come from. Hold the character steady; let the "
        "phrasing vary from turn to turn.]",
        "persona header verbatim",
    )


def test_facts_header_claims_truth_only():
    print("\n[test] facts header — authority over what is true, not how to say it")
    assert_eq(
        facts._FACTS_BLOCK_HEADER,
        "[Persistent facts about this conversation — established earlier. "
        "Stay accurate to these; the wording is yours.]",
        "facts header verbatim",
    )


def test_retrieval_header_claims_recall_only():
    print("\n[test] retrieval header — authority over what was said")
    # The highest-risk block of the four: it puts the model's OWN past prose
    # in the prompt, so whatever this header asks for, it asks for while
    # showing the model exactly how it phrased things last time.
    assert_eq(
        retrieval._RETRIEVAL_BLOCK_HEADER,
        "[Relevant earlier exchanges from this conversation, retrieved by "
        "similarity — use them for accurate recall of what happened. Say the "
        "next thing in your own words.]",
        "retrieval header verbatim",
    )


def test_summary_header_claims_events_only():
    print("\n[test] summary header — authority over what happened")
    assert_eq(
        summarizer._BLOCK_HEADER,
        "[Hierarchical summary of earlier portions of this conversation, "
        "ordered by recency — background you already hold, for continuity of "
        "events. Older summaries are denser; the L3 line (if present) is the "
        "whole-conversation theme.]",
        "summary header verbatim",
    )


# ---------------------------------------------------------------------------
# The block labels other suites match on
# ---------------------------------------------------------------------------


def test_block_labels_survive_rewording():
    print("\n[test] each header still opens with its block's label")
    # test_facts.py and test_retrieval.py assert on these substrings, and
    # OPERATIONS.md names them. A reword is free to change the guidance after
    # the em-dash; changing the LABEL breaks other suites and the runbook, so
    # it is called out separately here rather than discovered downstream.
    assert_true(
        facts._FACTS_BLOCK_HEADER.startswith("[Persistent facts"),
        "facts label intact (test_facts.py matches this)",
    )
    assert_true(
        retrieval._RETRIEVAL_BLOCK_HEADER.startswith("[Relevant earlier exchanges"),
        "retrieval label intact (test_retrieval.py matches this)",
    )
    assert_true(
        summarizer._BLOCK_HEADER.startswith("[Hierarchical summary"),
        "summary label intact",
    )
    assert_true(
        persona._PERSONA_BLOCK_HEADER.startswith("[Persona / role context"),
        "persona label intact",
    )


# ---------------------------------------------------------------------------
# Budget invariants — not pins
# ---------------------------------------------------------------------------


def test_facts_header_leaves_room_for_facts():
    print("\n[test] the facts header cannot eat the injection budget")
    # _FACTS_BLOCK_HEADER_TOKENS is charged once against
    # COMPACTOR_INJECT_FACTS_TOKENS before a single bullet is priced, and
    # v3.1.5 lowers that budget to 400 in production. A header that grows
    # without anyone noticing spends the budget on itself and silently
    # injects fewer facts — the failure mode _lru_split's "too small even for
    # the header" guard already refuses outright at the extreme, so the
    # dangerous region is the quiet middle, which is what this bounds.
    budget = 400
    cost = facts._FACTS_BLOCK_HEADER_TOKENS
    assert_true(cost < budget * 0.10, f"header is {cost} tokens, under 10% of {budget}")
    assert_true(
        facts._estimate_tokens(facts._FACTS_BLOCK_HEADER + "\n") == cost,
        "the constant is measured off the header, not hand-counted",
    )


def test_retrieval_and_summary_headers_are_small_against_their_budgets():
    print("\n[test] retrieval/summary headers are small against their budgets")
    # Both are charged once per block, against budgets an order of magnitude
    # larger than the facts one (retrieval: 1500 by default). Far looser
    # bound, same purpose: catch a header that ran away.
    r = retrieval._estimate_tokens(retrieval._RETRIEVAL_BLOCK_HEADER)
    s = summarizer._estimate_block_tokens(summarizer._BLOCK_HEADER)
    assert_true(r < 150, f"retrieval header {r} tokens")
    assert_true(s < 150, f"summary header {s} tokens")


# ---------------------------------------------------------------------------
# The blocks actually render with the headers above
# ---------------------------------------------------------------------------


def test_headers_reach_the_rendered_blocks():
    print("\n[test] the constants are what the rendered blocks actually carry")
    # Pinning a constant nothing reads would be a test of nothing. Each
    # formatter is exercised so the pins above are pins on shipped prompt
    # text.
    fb = facts.format_facts_block([{"text": "x", "added_turn": 1, "last_used": 1}])
    assert_true(fb.startswith(facts._FACTS_BLOCK_HEADER), "facts block carries it")

    rb = retrieval.format_retrieval_block(
        [{"turn_index": 1, "document": "x", "distance": 0.1}]
    )
    assert_true(
        rb.startswith(retrieval._RETRIEVAL_BLOCK_HEADER), "retrieval block carries it"
    )

    pb = persona.format_persona_block("x")
    assert_true(
        pb.startswith(persona._PERSONA_BLOCK_HEADER), "persona block carries it"
    )


if __name__ == "__main__":
    import shutil

    try:
        for t in (
            test_persona_header_owns_the_voice,
            test_facts_header_claims_truth_only,
            test_retrieval_header_claims_recall_only,
            test_summary_header_claims_events_only,
            test_block_labels_survive_rewording,
            test_facts_header_leaves_room_for_facts,
            test_retrieval_and_summary_headers_are_small_against_their_budgets,
            test_headers_reach_the_rendered_blocks,
        ):
            t()
        print("\nAll block-header tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
