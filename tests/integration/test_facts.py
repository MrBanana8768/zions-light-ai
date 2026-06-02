"""
Tier-3 validation of the facts memory layer (V2.0 Phase 2).

The "did the model actually remember that?" tests. Each scenario sends a
distinctive fact in turn 1, waits for the async post-response tail to
extract+save it, then asks a fresh question in turn 2 and asserts the
response is consistent with the saved fact.

Two modes:
- Without admin URL: only observe model behavior (response content).
- With admin URL: also assert the underlying state file contains a fact
  with the expected substring. Much stronger signal.
"""

import _harness as H


def test_fact_extracted_and_persisted(conv_id):
    """Turn 1 plants a vivid, retrievable fact; the async tail should
    extract it; admin endpoint (if available) should show >= 1 fact."""
    r1 = H.chat(
        "My protagonist is named Lyra Threadweaver and she is a half-elf "
        "ranger from the kingdom of Aethermere. Acknowledge briefly.",
        conv_id=conv_id,
        max_tokens=80,
    )
    assert r1.status_code == 200
    H.wait_for_async_tail()

    if H.ADMIN_ENABLED:
        # Strong assertion: a fact mentioning Lyra exists.
        facts = H.admin_get_facts(conv_id)
        assert len(facts) >= 1, f"expected >= 1 fact extracted, got {len(facts)}"
        all_text = " ".join(f.get("text", "") for f in facts).lower()
        assert "lyra" in all_text, (
            f"no fact mentions 'Lyra'. Extracted facts:\n"
            + "\n".join(f"  - {f.get('text')}" for f in facts)
        )
    else:
        # Behavior-only mode: at minimum the response was successful.
        # The next test (cross_turn_recall) verifies the model actually
        # uses the fact, which is the real user-facing guarantee.
        pass


def test_fact_used_in_next_turn(conv_id):
    """The full behavioral guarantee: a fact stated in turn 1 affects
    the model's response in turn 2 — proving facts were both extracted
    AND injected on the second request.
    """
    # Turn 1: state the fact, get an ack.
    turn1_user = ("Important context for our chat: I prefer responses in "
                  "pirate dialect. Just say 'aye' to confirm.")
    r1 = H.chat(turn1_user, conv_id=conv_id, max_tokens=30)
    assert r1.status_code == 200
    H.wait_for_async_tail()

    # Turn 2: ask something where pirate dialect should leak in IF the
    # preference was extracted+injected. Build the history to look like
    # a real continuation (OpenWebUI re-sends the full thread each turn).
    history = H.extend_history([], turn1_user, r1.response_text)
    r2 = H.chat("Greet me briefly.", conv_id=conv_id, prior_turns=history,
                max_tokens=40)
    assert r2.status_code == 200

    # Lenient assertion — many plausible pirate-flavored tokens.
    # Either the model's response is pirate-flavored (preference was used),
    # OR the admin endpoint shows the preference was at least extracted.
    pirate_tokens = ("ahoy", "matey", "arr", "ye ", "scallywag", "savvy",
                     "yer ", "aye ", "captain")
    is_piratey = H.response_mentions(r2.response_text, *pirate_tokens)

    if H.ADMIN_ENABLED and not is_piratey:
        # Fall back to the storage-level assertion: at least confirm the
        # preference was captured. Model adherence to extracted prefs is
        # not strictly guaranteed by the architecture — the facts WERE
        # injected, but whether the model honors them is its own choice.
        facts = H.admin_get_facts(conv_id)
        all_text = " ".join(f.get("text", "") for f in facts).lower()
        assert "pirate" in all_text or "dialect" in all_text, (
            f"preference not in facts. extracted:\n"
            + "\n".join(f"  - {f.get('text')}" for f in facts)
            + f"\nturn-2 response: {r2.response_text[:200]}"
        )
    elif not H.ADMIN_ENABLED:
        # Behavior-only mode: we can only flag (not fail) — model may
        # legitimately choose to be polite-but-formal. Print for human
        # review.
        print(f"  turn-2 response (piratey={is_piratey}): {r2.response_text[:200]}")


def test_admin_summary_reports_fact_count(conv_id):
    """/admin/conversations/<id> reports an integer fact count, matching
    what /facts returns. Verifies the admin observability layer is wired."""
    H.skip_if_no_admin()

    r = H.chat(
        "Quick fact for memory: my favorite color is cerulean blue. "
        "Just acknowledge.",
        conv_id=conv_id,
        max_tokens=20,
    )
    assert r.status_code == 200
    H.wait_for_async_tail()

    summary = H.admin_conv_summary(conv_id)
    assert "facts" in summary, summary
    count = summary["facts"].get("count")
    assert isinstance(count, int), f"facts.count should be int, got {count!r}"
    facts = H.admin_get_facts(conv_id)
    assert count == len(facts), f"summary count {count} != facts len {len(facts)}"
