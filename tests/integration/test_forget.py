"""
Tier-3 validation of the /forget endpoint (Phase 2-4 combined behavior).

DELETE /admin/conversations/<id>/facts is documented as a "full three-
layer reset" — it must clear:
  1. facts JSON   (Phase 2)
  2. ChromaDB     (Phase 3)
  3. summary JSON (Phase 4)

Admin-only by design (every assertion needs admin endpoints).
"""

import _harness as H


def test_forget_clears_facts_and_episodic(conv_id):
    """Populate facts + episodic memory in one chat, then forget, then
    confirm both counters drop to zero."""
    H.skip_if_no_admin("forget verification requires admin endpoint")

    H.chat(
        "Important: my favorite mythical creature is the hippogriff. "
        "Got it?",
        conv_id=conv_id,
        max_tokens=30,
    )
    H.wait_for_async_tail()

    before = H.admin_conv_summary(conv_id)
    facts_before = before.get("facts", {}).get("count", 0) or 0
    indexed_before = before.get("episodic", {}).get("indexed_exchanges", 0) or 0
    # At least one of the two should have populated for the test to be
    # meaningful — otherwise we'd be asserting zero stayed zero.
    assert facts_before + indexed_before >= 1, (
        f"nothing got persisted to forget — before state: {before}"
    )

    result = H.admin_forget(conv_id)
    assert result.get("conv_id") == conv_id
    assert result.get("forgotten_facts") == facts_before
    assert result.get("forgotten_episodic") == indexed_before

    after = H.admin_conv_summary(conv_id)
    assert (after.get("facts", {}).get("count") or 0) == 0, (
        f"facts not cleared after forget: {after}"
    )
    assert (after.get("episodic", {}).get("indexed_exchanges") or 0) == 0, (
        f"episodic not cleared after forget: {after}"
    )


def test_forget_response_shape(conv_id):
    """The DELETE response carries the three forgotten_* counters, so
    callers can log/show what was actually wiped."""
    H.skip_if_no_admin()
    H.chat("Just a quick test message, please reply briefly.",
           conv_id=conv_id, max_tokens=20)
    H.wait_for_async_tail()

    result = H.admin_forget(conv_id)
    for key in ("conv_id", "forgotten_facts", "forgotten_episodic",
                "forgotten_summary"):
        assert key in result, f"forget response missing '{key}': {result}"
    assert isinstance(result["forgotten_facts"], int)
    assert isinstance(result["forgotten_episodic"], int)
    assert isinstance(result["forgotten_summary"], bool)


def test_forget_is_idempotent(conv_id):
    """Calling forget twice in a row must not error (second call sees
    nothing to delete, returns zeros)."""
    H.skip_if_no_admin()
    H.chat("brief msg", conv_id=conv_id, max_tokens=10)
    H.wait_for_async_tail()

    H.admin_forget(conv_id)  # first call wipes
    second = H.admin_forget(conv_id)  # second call should be a no-op
    assert second.get("forgotten_facts") == 0
    assert second.get("forgotten_episodic") == 0
    assert second.get("forgotten_summary") is False
