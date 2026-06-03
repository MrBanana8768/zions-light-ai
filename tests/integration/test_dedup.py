"""
Tier-3 validation of /admin/conversations/<id>/dedup (V2.1 Phase 7 Step 1).

Drives the live stack to populate a conv with near-duplicate facts,
runs the dedup endpoint, and asserts the count went down. Localhost-only.

Inline dedup (the after-extraction trigger) is hard to test deterministically
in Tier-3 because it depends on Magnum-12B repeatedly extracting paraphrases
of the same statement, which the extractor isn't designed to do (it sees
EXISTING FACTS and is told not to restate). So Tier-3 focuses on the
on-demand path which is fully controllable via the harness.
"""

import uuid

import _harness as H


def test_dedup_endpoint_is_noop_when_lt_2_facts(conv_id):
    """A conv with 0 or 1 fact has nothing to merge — endpoint should
    return a zero-removed response, not crash."""
    H.skip_if_no_admin()
    result = H.admin_dedup(conv_id)
    assert result["conv_id"] == conv_id
    assert result["removed"] == 0, f"unexpected removals: {result!r}"
    assert result["before"] == result["after"], result


def test_dedup_endpoint_returns_full_shape(conv_id):
    """Shape lock: every field UI consumers depend on must be present."""
    H.skip_if_no_admin()
    result = H.admin_dedup(conv_id)
    for key in ("conv_id", "before", "after", "removed"):
        assert key in result, f"missing {key!r}: {result!r}"
    assert isinstance(result["before"], int)
    assert isinstance(result["after"], int)
    assert isinstance(result["removed"], int)


def test_dedup_merges_seeded_duplicates_via_import(conv_id):
    """Seed a conv with three near-identical facts via the import endpoint
    (bypasses the extractor's own dedup behavior), then run dedup and
    verify at least one was merged.

    Uses /admin/import to plant facts deterministically — the extractor
    is too smart to produce dupes on its own when EXISTING FACTS is in
    the prompt.
    """
    H.skip_if_no_admin()
    target = f"itest-dedup-{uuid.uuid4().hex[:8]}"
    bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "seeded",
        "facts": [
            {"text": "Lyra is a half-elf ranger", "added_turn": 0, "last_used": 100},
            {"text": "Lyra is half elf and is a ranger", "added_turn": 1, "last_used": 101},
            {"text": "The protagonist Lyra is a half-elven ranger", "added_turn": 2, "last_used": 102},
        ],
        "summary_state": {},
        "episodic": [],
    }
    try:
        status, _ = H.admin_import(bundle, target_conv_id=target, overwrite=False)
        assert status == 200, f"import failed: {status}"
        before_facts = H.admin_get_facts(target)
        assert len(before_facts) == 3, "prep: 3 facts seeded"

        result = H.admin_dedup(target)
        # Hybrid dedup may or may not merge all three depending on model
        # judgment — but with these three nearly-identical phrasings, we
        # expect at least one merge (3→2 or 3→1).
        assert result["after"] <= 2, (
            f"expected at least one merge (3 → ≤2), got "
            f"before={result['before']} after={result['after']}"
        )
        assert result["removed"] >= 1, f"removed=0 — model returned KEEP for all? {result!r}"
        # Re-fetch to confirm the active facts file matches
        after_facts = H.admin_get_facts(target)
        assert len(after_facts) == result["after"], (
            f"endpoint reports after={result['after']} but admin_get_facts "
            f"returns {len(after_facts)}"
        )
    finally:
        H.admin_safe_forget(target)


def test_dedup_preserves_unrelated_facts(conv_id):
    """Seed a conv with facts that are CLEARLY unrelated — dedup must
    not merge them. Guards against false-positive merging by the LLM."""
    H.skip_if_no_admin()
    target = f"itest-dedup-distinct-{uuid.uuid4().hex[:8]}"
    bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "seeded",
        "facts": [
            {"text": "User likes pirate dialect", "added_turn": 0, "last_used": 100},
            {"text": "The setting is Aethermere", "added_turn": 1, "last_used": 101},
            {"text": "Hippogriffs are a key creature in the world", "added_turn": 2, "last_used": 102},
        ],
        "summary_state": {},
        "episodic": [],
    }
    try:
        status, _ = H.admin_import(bundle, target_conv_id=target, overwrite=False)
        assert status == 200
        result = H.admin_dedup(target)
        # These three facts are about totally different things; cosine
        # similarity shouldn't even cluster them, let alone the LLM merge.
        assert result["removed"] == 0, (
            f"unrelated facts got merged — dedup is over-eager. "
            f"result: {result!r}"
        )
        assert result["after"] == 3, result
    finally:
        H.admin_safe_forget(target)
