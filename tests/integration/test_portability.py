"""
Tier-3 validation of portability endpoints (V2.1 Phase 6 Step 3).

Exercises export → forget → import round-trip + fork operations
against a live deploy. Localhost-only — all three endpoints are
admin.
"""

import uuid

import _harness as H


def test_export_empty_conv_returns_full_shape(conv_id):
    """Even a conv with zero state must export a full-shape bundle so
    import logic doesn't need defensive .get() calls."""
    H.skip_if_no_admin()
    bundle = H.admin_export(conv_id)
    for key in ("version", "exported_at", "source_conv_id",
                "facts", "summary_state", "episodic"):
        assert key in bundle, f"missing {key!r}: {bundle!r}"
    assert bundle["version"] == "v2.1", bundle["version"]
    assert bundle["source_conv_id"] == conv_id
    assert bundle["facts"] == []
    assert bundle["episodic"] == []
    assert isinstance(bundle["summary_state"], dict)


def test_export_after_chat_includes_facts_and_episodic(conv_id):
    """Drive one fact-rich chat, then verify the export bundle picks up
    both the extracted fact and the indexed episodic exchange."""
    H.skip_if_no_admin()
    H.chat(
        "Important: my protagonist is named Lyra Threadweaver, a half-elf "
        "ranger from Aethermere. Please confirm noted.",
        conv_id=conv_id,
        max_tokens=40,
    )
    H.wait_for_facts(conv_id, min_count=1, max_wait=30)

    bundle = H.admin_export(conv_id)
    assert len(bundle["facts"]) >= 1, f"expected facts: {bundle!r}"
    assert len(bundle["episodic"]) >= 1, f"expected episodic: {bundle!r}"


def test_import_into_fresh_conv_restores_state(conv_id):
    """Round-trip: chat → export → forget → import (under new id)
    → verify state lands cleanly in the new conv."""
    H.skip_if_no_admin()
    H.chat(
        "Note this for memory: my favorite mythical creature is a hippogriff. "
        "Acknowledge briefly.",
        conv_id=conv_id,
        max_tokens=30,
    )
    H.wait_for_facts(conv_id, min_count=1, max_wait=30)

    bundle = H.admin_export(conv_id)
    src_facts = len(bundle["facts"])
    src_episodic = len(bundle["episodic"])
    assert src_facts >= 1 and src_episodic >= 1, (
        f"prep: source conv must have state to test round-trip; "
        f"facts={src_facts} episodic={src_episodic}"
    )

    # Import into a fresh target id — fresh = no existing state
    target = f"itest-import-{uuid.uuid4().hex[:8]}"
    try:
        status, result = H.admin_import(bundle, target_conv_id=target, overwrite=False)
        assert status == 200, f"import failed: HTTP {status} body={result!r}"
        assert result["conv_id"] == target
        assert result["imported"]["facts"] == src_facts
        # Re-export from target and compare to original
        round_trip_bundle = H.admin_export(target)
        assert len(round_trip_bundle["facts"]) == src_facts, (
            f"fact count mismatch after round-trip: "
            f"src={src_facts} dst={len(round_trip_bundle['facts'])}"
        )
        # The episodic count should match too (modulo re-embedding fidelity,
        # but document text is preserved exactly).
        assert len(round_trip_bundle["episodic"]) == src_episodic
    finally:
        H.admin_safe_forget(target)


def test_import_refuses_overwrite_by_default(conv_id):
    """Importing into a conv that already has state must fail with HTTP 400
    unless overwrite=true. Protects against accidental wipes."""
    H.skip_if_no_admin()
    H.chat("brief msg", conv_id=conv_id, max_tokens=10)
    H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=30)

    # Build a tiny valid bundle and try to drop it on the populated conv
    fake_bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "other",
        "facts": [{"text": "replacement", "added_turn": 0, "last_used": 0}],
        "summary_state": {},
        "episodic": [],
    }
    status, body = H.admin_import(fake_bundle, target_conv_id=conv_id, overwrite=False)
    assert status == 400, (
        f"expected 400 refusal, got {status}: {body!r}"
    )
    assert "existing state" in (body.get("detail") or "").lower(), body


def test_import_rejects_wrong_version_bundle():
    """A bundle with unsupported version must be rejected with HTTP 400."""
    H.skip_if_no_admin()
    bad = {
        "version": "v99.0",
        "facts": [],
        "summary_state": {},
        "episodic": [],
    }
    target = f"itest-badver-{uuid.uuid4().hex[:6]}"
    status, body = H.admin_import(bad, target_conv_id=target)
    assert status == 400, f"expected 400, got {status}: {body!r}"


def test_fork_clones_state_into_new_conv(conv_id):
    """Fork should produce a new conv_id with the same memory state.
    Mutating the fork must not affect the parent."""
    H.skip_if_no_admin()
    H.chat(
        "Setup: the secret password is 'mockingbird'. Acknowledge briefly.",
        conv_id=conv_id,
        max_tokens=30,
    )
    H.wait_for_facts(conv_id, min_count=1, max_wait=30)
    parent_facts_before = H.admin_get_facts(conv_id)
    assert len(parent_facts_before) >= 1, "prep: parent must have state"

    result = H.admin_fork(conv_id)
    new_id = result["conv_id"]
    try:
        assert result["forked_from"] == conv_id
        assert new_id.startswith(f"{conv_id}__fork_"), f"unexpected id: {new_id}"
        fork_facts = H.admin_get_facts(new_id)
        assert len(fork_facts) == len(parent_facts_before), (
            f"fork should have parent's facts: "
            f"parent={len(parent_facts_before)} fork={len(fork_facts)}"
        )
        # Mutate the fork — parent must be untouched
        H.admin_forget(new_id)
        parent_facts_after = H.admin_get_facts(conv_id)
        assert len(parent_facts_after) == len(parent_facts_before), (
            "forgetting fork affected parent — fork is NOT isolated"
        )
    finally:
        H.admin_safe_forget(new_id)
