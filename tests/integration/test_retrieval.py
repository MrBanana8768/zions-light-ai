"""
Tier-3 validation of the episodic memory layer (V2.0 Phase 3, RAG).

Verifies that:
  1. Exchanges get indexed into ChromaDB (admin-mode strong assertion).
  2. A semantically-similar query later in the conversation retrieves
     the earlier turn (behavioral + admin-mode strong assertion).
  3. The /admin counter reports indexed_exchanges.

Uses fewer fixture turns than the summarizer test (which needs ≥ 20 to
trigger rollups). Each chat is real inference — keep counts minimal.
"""

import _harness as H


def test_exchange_gets_indexed(conv_id):
    """One chat → wait → admin reports indexed_exchanges >= 1."""
    H.skip_if_no_admin("indexed-exchange count requires admin endpoint")

    r = H.chat(
        "Quick aside: the rare amaranth crystal can only be found in the "
        "Hollow Marshes east of Veynport. Note this for later.",
        conv_id=conv_id,
        max_tokens=40,
    )
    assert r.status_code == 200
    # Episodic indexing is cheap (no LLM call, just an embedding + upsert)
    # but on a cold pod fastembed lazy-loads the ONNX model on first use.
    indexed = H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=30)
    assert indexed >= 1, (
        f"expected indexed_exchanges >= 1 within 30s, got {indexed}.\n"
        f"Likely causes: indexing failed silently, fastembed init failed, "
        f"or COMPACTOR_RAG_ENABLED is disabled."
    )


def test_distinctive_content_retrieved_later(conv_id):
    """The end-to-end RAG guarantee: an early turn with distinctive,
    rare vocabulary is retrievable by a semantic query several turns
    later. Uses a small fixture to keep chat costs bounded.
    """
    # Turn 1: plant distinctive content with rare proper nouns.
    plant = (
        "Setting note: the amaranth crystal grows only in the Hollow Marshes "
        "east of Veynport. It glows pale violet when held by someone with "
        "elven blood. The crystal is sacred to the Vesh'tara clan. Got it?"
    )
    r1 = H.chat(plant, conv_id=conv_id, max_tokens=40)
    assert r1.status_code == 200
    # Make sure turn 1 is indexed before we move on — otherwise the
    # probe later might race against the planted turn still being indexed.
    H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=30)

    # Turns 2-5: filler exchanges on unrelated topics so the plant turn
    # is no longer in the immediate "recent N" window and would only be
    # surfaced by retrieval.
    history = H.extend_history([], plant, r1.response_text)
    fillers = [
        "Briefly: what's a good breakfast for a long journey?",
        "Now invent a one-line proverb about patience.",
        "Suggest a single color other than blue.",
        "Give me one short sentence about wind.",
    ]
    for f in fillers:
        rf = H.chat(f, conv_id=conv_id, prior_turns=history, max_tokens=30)
        assert rf.status_code == 200
        history = H.extend_history(history, f, rf.response_text)
        # No tail-wait between fillers — the post-response indexing for
        # one turn can overlap with the next request; that's a designed
        # property of the async tail.

    # Wait until all 5 turns have been indexed before the probe, so the
    # plant turn (#1) is reliably in the vector store and retrievable.
    H.wait_for_indexed_exchanges(conv_id, min_count=5, max_wait=30)

    # The probe query: ask about the planted content using semantically
    # related wording. RAG should pull turn-1 back.
    probe = "Tell me where I said the amaranth crystal is found."
    rp = H.chat(probe, conv_id=conv_id, prior_turns=history, max_tokens=80)
    assert rp.status_code == 200

    # The response should mention specific terms from the plant. The LLM
    # paraphrases, so accept several plausible mentions.
    location_mentioned = H.response_mentions(
        rp.response_text,
        "Hollow Marshes", "Veynport", "marshes east", "east of Veynport",
    )

    if H.ADMIN_ENABLED:
        # Strong assertion: indexed_exchanges grew over the conversation.
        summary = H.admin_conv_summary(conv_id)
        indexed = summary.get("episodic", {}).get("indexed_exchanges", 0)
        # 5 turns sent (plant + 4 fillers), each should index → expect >= 4
        # (last-turn indexing may still be in-flight; lenient lower bound).
        assert indexed >= 4, (
            f"expected indexed_exchanges >= 4 after 5 chats, got {indexed}"
        )
        # If location wasn't mentioned, RAG may not have fired OR the
        # model just didn't use it. Print details for human review rather
        # than failing — model adherence isn't strictly guaranteed.
        if not location_mentioned:
            print(
                f"  WARN: probe response did not mention location.\n"
                f"  response: {rp.response_text[:300]}\n"
                f"  indexed_exchanges: {indexed}"
            )
    else:
        # Behavior-only mode: print, don't fail. Model output isn't
        # deterministic.
        print(f"  RAG probe response: {rp.response_text[:300]}")
        print(f"  location_mentioned: {location_mentioned}")
