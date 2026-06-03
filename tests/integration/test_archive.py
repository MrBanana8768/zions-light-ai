"""
Tier-3 validation of stale-fact archival endpoints (V2.1 Phase 7 Step 2).

GET  /admin/conversations/<id>/archive
POST /admin/conversations/<id>/archive?older_than_days=N
POST /admin/conversations/<id>/restore  body: {text_substring?}

We seed state via /admin/import (rather than driving real chats and
waiting for last_used to age) so tests are deterministic and don't
need a time machine — the bundle includes last_used timestamps we
control directly.
"""

import time
import uuid

import _harness as H


def test_archive_no_facts_returns_empty(conv_id):
    """Empty conv → archive endpoint returns empty list (not 404)."""
    H.skip_if_no_admin()
    archived = H.admin_get_archive(conv_id)
    assert archived == [], f"expected empty list: {archived!r}"


def test_archive_stale_moves_old_facts_to_sidecar():
    """Seed facts with old last_used → archive pass moves them off
    the active list, restore brings them back."""
    H.skip_if_no_admin()
    target = f"itest-archive-{uuid.uuid4().hex[:8]}"
    now = int(time.time())
    old_ts = now - (100 * 86400)  # 100 days old
    bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "seeded",
        "facts": [
            {"text": "fresh fact", "added_turn": 0, "last_used": now},
            {"text": "ancient fact one", "added_turn": 1, "last_used": old_ts},
            {"text": "ancient fact two", "added_turn": 2, "last_used": old_ts},
        ],
        "summary_state": {},
        "episodic": [],
    }
    try:
        status, _ = H.admin_import(bundle, target_conv_id=target, overwrite=False)
        assert status == 200
        before = H.admin_get_facts(target)
        assert len(before) == 3, "prep: 3 facts seeded"

        # Archive everything older than 30 days
        result = H.admin_archive_stale(target, older_than_days=30)
        assert result["kept"] == 1, f"expected 1 kept (fresh only): {result!r}"
        assert result["archived"] == 2, f"expected 2 archived: {result!r}"

        # Confirm active facts dropped to 1
        after = H.admin_get_facts(target)
        assert len(after) == 1
        assert after[0]["text"] == "fresh fact"

        # Confirm archive sidecar has both ancients
        archived = H.admin_get_archive(target)
        assert len(archived) == 2
        archived_texts = {f["text"] for f in archived}
        assert archived_texts == {"ancient fact one", "ancient fact two"}
        for f in archived:
            assert "archived_at" in f, f"missing archived_at: {f!r}"
            assert f["archived_at"] > 0
    finally:
        H.admin_safe_forget(target)


def test_restore_all_brings_archived_back():
    """Archive everything, then restore all — active count returns to
    original, archive is empty."""
    H.skip_if_no_admin()
    target = f"itest-restore-{uuid.uuid4().hex[:8]}"
    now = int(time.time())
    old_ts = now - (100 * 86400)
    bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "seeded",
        "facts": [
            {"text": "A", "added_turn": 0, "last_used": old_ts},
            {"text": "B", "added_turn": 1, "last_used": old_ts},
        ],
        "summary_state": {},
        "episodic": [],
    }
    try:
        H.admin_import(bundle, target_conv_id=target, overwrite=False)
        H.admin_archive_stale(target, older_than_days=30)
        assert len(H.admin_get_facts(target)) == 0, "prep: active empty"
        assert len(H.admin_get_archive(target)) == 2, "prep: archive full"

        result = H.admin_restore_from_archive(target)  # no filter = all
        assert result["restored"] == 2, f"expected 2 restored: {result!r}"
        assert result["filter"] is None
        assert len(H.admin_get_facts(target)) == 2
        assert len(H.admin_get_archive(target)) == 0
    finally:
        H.admin_safe_forget(target)


def test_restore_with_substring_filter_is_case_insensitive():
    """Filter restore by substring — case-insensitive contains match."""
    H.skip_if_no_admin()
    target = f"itest-restore-filt-{uuid.uuid4().hex[:8]}"
    now = int(time.time())
    old_ts = now - (100 * 86400)
    bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "seeded",
        "facts": [
            {"text": "Lyra is a ranger", "added_turn": 0, "last_used": old_ts},
            {"text": "Aethermere is the kingdom", "added_turn": 1, "last_used": old_ts},
            {"text": "Hippogriffs exist", "added_turn": 2, "last_used": old_ts},
        ],
        "summary_state": {},
        "episodic": [],
    }
    try:
        H.admin_import(bundle, target_conv_id=target, overwrite=False)
        H.admin_archive_stale(target, older_than_days=30)
        # Use upper-case substring to verify case-insensitivity
        result = H.admin_restore_from_archive(target, text_substring="LYRA")
        assert result["restored"] == 1, f"expected 1 restored: {result!r}"
        assert result["filter"] == "LYRA"
        active = H.admin_get_facts(target)
        assert len(active) == 1
        assert active[0]["text"] == "Lyra is a ranger"
        # Other two stay archived
        assert len(H.admin_get_archive(target)) == 2
    finally:
        H.admin_safe_forget(target)


def test_archive_is_idempotent():
    """Running archive twice with the same cutoff = second pass archives 0."""
    H.skip_if_no_admin()
    target = f"itest-archive-idem-{uuid.uuid4().hex[:8]}"
    now = int(time.time())
    old_ts = now - (100 * 86400)
    bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "seeded",
        "facts": [{"text": "old", "added_turn": 0, "last_used": old_ts}],
        "summary_state": {},
        "episodic": [],
    }
    try:
        H.admin_import(bundle, target_conv_id=target, overwrite=False)
        r1 = H.admin_archive_stale(target, older_than_days=30)
        assert r1["archived"] == 1
        r2 = H.admin_archive_stale(target, older_than_days=30)
        assert r2["archived"] == 0, f"second pass should archive 0: {r2!r}"
    finally:
        H.admin_safe_forget(target)
