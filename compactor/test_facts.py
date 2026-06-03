"""
CPU-only smoke tests for compactor/facts.py (V2.0 Phase 2).

Covers I/O round-trips, atomic-write semantics, LRU pruning, the
extraction prompt parser, and the end-to-end record_facts_for_exchange
flow with a mock vLLM client.

Run inside the compactor image or any container with the requirements
installed:
    python test_facts.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Storage redirect MUST happen before importing memory/facts so the
# module-level paths see the override.
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-facts-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import facts  # noqa: E402
import memory  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _wipe_storage():
    """Clean slate between tests."""
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


# ---------------------------------------------------------------------------
# Atomic write + load semantics
# ---------------------------------------------------------------------------

def test_load_facts_missing_file_returns_empty():
    print("\n[test] load_facts returns [] for a conv with no file")
    _wipe_storage()
    assert_eq(facts.load_facts("never-seen"), [], "missing file -> []")


def test_load_facts_corrupted_file_returns_empty():
    print("\n[test] load_facts returns [] for a corrupted JSON file")
    _wipe_storage()
    facts_path = memory.facts_path("corrupt")
    facts_path.write_text("{ not valid json")
    assert_eq(facts.load_facts("corrupt"), [], "corrupt file -> [] (logged, not raised)")


def test_save_load_roundtrip():
    print("\n[test] save_facts -> load_facts preserves content")
    _wipe_storage()
    cid = "rt"
    fixture = [
        {"text": "Protagonist is Lyra.", "added_turn": 5, "last_used": 1748000000},
        {"text": "Setting: low-magic medieval.", "added_turn": 7, "last_used": 1748000100},
    ]
    facts.save_facts(cid, fixture)
    loaded = facts.load_facts(cid)
    assert_eq(len(loaded), 2, "two facts loaded")
    assert_eq(loaded[0]["text"], "Protagonist is Lyra.", "text preserved")
    assert_eq(loaded[1]["added_turn"], 7, "added_turn preserved")


def test_save_facts_is_atomic_via_temp_file():
    print("\n[test] save_facts uses temp-then-rename (no torn writes)")
    _wipe_storage()
    cid = "atomic"
    # Write a baseline
    facts.save_facts(cid, [{"text": "first", "added_turn": 1, "last_used": 1}])
    facts_p = memory.facts_path(cid)
    original_content = facts_p.read_text()
    # Simulate atomic_write_json's behavior: temp file appears momentarily.
    # We can't easily test the atomicity directly without filesystem injection,
    # but we CAN verify no leftover .tmp files exist after a normal write.
    facts.save_facts(cid, [{"text": "second", "added_turn": 2, "last_used": 2}])
    leftover_tmps = list(facts_p.parent.glob("*.tmp"))
    assert_eq(len(leftover_tmps), 0, "no .tmp leftovers after successful write")
    # Confirm new content is what landed.
    loaded = facts.load_facts(cid)
    assert_eq(loaded[0]["text"], "second", "second write overwrote first")
    assert_true(facts_p.read_text() != original_content, "file content actually changed")


def test_load_facts_drops_malformed_entries():
    print("\n[test] load_facts filters malformed facts (defensive)")
    _wipe_storage()
    cid = "mixed"
    # Manually write a facts file with some malformed entries
    raw = {
        "conv_id": cid,
        "updated_at": "2026-05-28T00:00:00+00:00",
        "facts": [
            {"text": "valid", "added_turn": 1, "last_used": 100},
            {"text": "", "added_turn": 2, "last_used": 200},          # empty text
            "not a dict",                                              # wrong type
            {"added_turn": 3, "last_used": 300},                       # missing text
            {"text": "  ", "added_turn": 4, "last_used": 400},         # whitespace-only
            {"text": "also valid", "added_turn": 5, "last_used": 500},
        ],
    }
    memory.facts_path(cid).write_text(json.dumps(raw))
    loaded = facts.load_facts(cid)
    assert_eq(len(loaded), 2, "kept only the 2 valid entries")
    assert_eq(loaded[0]["text"], "valid", "first valid preserved")
    assert_eq(loaded[1]["text"], "also valid", "second valid preserved")


# ---------------------------------------------------------------------------
# Pruning (LRU)
# ---------------------------------------------------------------------------

def test_prune_facts_no_op_under_budget():
    print("\n[test] prune_facts is a no-op when total under budget")
    items = [{"text": "short", "added_turn": 1, "last_used": 1}]
    kept, dropped = facts.prune_facts(items, max_tokens=1000)
    assert_eq(len(kept), 1, "kept everything")
    assert_eq(dropped, 0, "dropped 0")


def test_prune_facts_lru_eviction():
    print("\n[test] prune_facts drops least-recently-used first")
    # Each fact ≈ 100 chars → 25 tokens. Budget 25 = only 1 fact fits.
    items = [
        {"text": "x" * 100, "added_turn": 1, "last_used": 100},  # oldest used → drop
        {"text": "y" * 100, "added_turn": 2, "last_used": 500},  # mid → drop
        {"text": "z" * 100, "added_turn": 3, "last_used": 999},  # newest → keep
    ]
    kept, dropped = facts.prune_facts(items, max_tokens=25)
    assert_eq(dropped, 2, "evicted 2 oldest")
    assert_eq(len(kept), 1, "1 fact survives")
    assert_eq(kept[0]["text"], "z" * 100, "most-recently-used preserved")

    # Also verify intermediate budget keeps 2 most-recent.
    kept2, dropped2 = facts.prune_facts(items, max_tokens=50)
    assert_eq(dropped2, 1, "budget=50 evicts only the oldest")
    assert_eq(len(kept2), 2, "2 facts survive")
    # Restored to added_turn order after eviction
    assert_eq([f["text"] for f in kept2], ["y" * 100, "z" * 100],
              "kept in added_turn order: y then z")


def test_prune_facts_empty_input():
    print("\n[test] prune_facts handles empty input")
    kept, dropped = facts.prune_facts([], max_tokens=1000)
    assert_eq(kept, [], "empty in -> empty out")
    assert_eq(dropped, 0, "nothing dropped")


# ---------------------------------------------------------------------------
# Touch + injection block
# ---------------------------------------------------------------------------

def test_touch_facts_updates_timestamps():
    print("\n[test] touch_facts marks every fact as just-used")
    items = [
        {"text": "a", "added_turn": 1, "last_used": 0},
        {"text": "b", "added_turn": 2, "last_used": 0},
    ]
    facts.touch_facts(items, now=12345)
    assert_eq(items[0]["last_used"], 12345, "first fact touched")
    assert_eq(items[1]["last_used"], 12345, "second fact touched")


def test_format_facts_block_empty():
    print("\n[test] format_facts_block returns None for no facts")
    assert_eq(facts.format_facts_block([]), None, "empty -> None")


def test_format_facts_block_renders_bullets():
    print("\n[test] format_facts_block renders header + bullets")
    items = [
        {"text": "fact one", "added_turn": 1, "last_used": 1},
        {"text": "fact two", "added_turn": 2, "last_used": 2},
    ]
    block = facts.format_facts_block(items)
    assert_true("[Persistent facts" in block, "header present")
    assert_true("- fact one" in block, "first bullet present")
    assert_true("- fact two" in block, "second bullet present")


# ---------------------------------------------------------------------------
# Extraction prompt parser
# ---------------------------------------------------------------------------

def test_parse_extraction_NONE_returns_empty():
    print("\n[test] _parse_extraction_output handles NONE")
    assert_eq(facts._parse_extraction_output("NONE"), [], "NONE -> []")
    assert_eq(facts._parse_extraction_output("None."), [], "None. -> []")
    assert_eq(facts._parse_extraction_output("none"), [], "lowercase none -> []")
    assert_eq(facts._parse_extraction_output(""), [], "empty -> []")


def test_parse_extraction_dash_bullets():
    print("\n[test] _parse_extraction_output strips dash bullets")
    raw = "- The protagonist is Lyra.\n- Setting is a medieval kingdom."
    parsed = facts._parse_extraction_output(raw)
    assert_eq(len(parsed), 2, "two facts parsed")
    assert_eq(parsed[0], "The protagonist is Lyra.", "first stripped")
    assert_eq(parsed[1], "Setting is a medieval kingdom.", "second stripped")


def test_parse_extraction_numbered_list():
    print("\n[test] _parse_extraction_output strips numbered prefixes")
    raw = "1. Fact one.\n2. Fact two."
    parsed = facts._parse_extraction_output(raw)
    assert_eq(parsed, ["Fact one.", "Fact two."], "numbers stripped")


def test_parse_extraction_mixed_and_blank_lines():
    print("\n[test] _parse_extraction_output handles mixed input")
    # All bullets >= 6 chars after stripping the prefix (per the parser's
    # too-short filter). Tests: dash, asterisk, unicode bullet, en-dash.
    raw = "- first fact\n\n- second fact\n   \n* third alt bullet\n• fourth unicode bullet"
    parsed = facts._parse_extraction_output(raw)
    assert_eq(len(parsed), 4, "blank lines skipped, all bullet styles handled")


def test_parse_extraction_drops_too_short():
    print("\n[test] _parse_extraction_output drops lines < 6 chars")
    raw = "- ok\n- this is fine"
    parsed = facts._parse_extraction_output(raw)
    assert_eq(len(parsed), 1, "short '- ok' dropped, long line kept")


# ---------------------------------------------------------------------------
# Extraction with mock vLLM client
# ---------------------------------------------------------------------------

def _mock_client_returning(content: str) -> MagicMock:
    """Build a fake httpx.AsyncClient whose .post returns a canned response."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={
        "choices": [{"message": {"content": content}}]
    })
    client = MagicMock()
    client.post = AsyncMock(return_value=mock_response)
    return client


def _mock_client_raising(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(side_effect=exc)
    return client


def test_extract_facts_from_exchange_success():
    print("\n[test] extract_facts_from_exchange parses canned LLM response")
    client = _mock_client_returning(
        "- The character is named Lyra.\n- She is half-elf."
    )
    out = asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model",
        "Tell me about Lyra.", "Lyra is half-elf.", []
    ))
    assert_eq(len(out), 2, "two facts extracted")


def test_extract_facts_from_exchange_NONE():
    print("\n[test] extract_facts_from_exchange handles NONE response")
    client = _mock_client_returning("NONE")
    out = asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model", "hi", "hello", []
    ))
    assert_eq(out, [], "NONE -> []")


def test_extract_facts_from_exchange_network_failure():
    print("\n[test] extract_facts_from_exchange swallows network errors")
    client = _mock_client_raising(RuntimeError("connection refused"))
    out = asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model", "hi", "hello", []
    ))
    assert_eq(out, [], "failure -> [] (never raises to caller)")


def test_extract_facts_from_exchange_empty_inputs_short_circuit():
    print("\n[test] extract_facts_from_exchange short-circuits empty inputs")
    client = _mock_client_returning("- something")
    out_user = asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model", "", "assistant", []
    ))
    out_asst = asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model", "user", "", []
    ))
    assert_eq(out_user, [], "no user msg -> [] (no LLM call)")
    assert_eq(out_asst, [], "no assistant msg -> [] (no LLM call)")


def test_record_facts_end_to_end():
    print("\n[test] record_facts_for_exchange: extract → prune → save round-trip")
    _wipe_storage()
    cid = "e2e"
    client = _mock_client_returning(
        "- Character Lyra is a ranger.\n- The setting is Aethermere."
    )

    n = asyncio.run(facts.record_facts_for_exchange(
        cid, client, "http://fake", "fake-model",
        user_msg="Who is Lyra?",
        assistant_msg="A half-elf ranger from Aethermere.",
        turn_index=4,
    ))
    assert_eq(n, 2, "2 new facts added")
    loaded = facts.load_facts(cid)
    assert_eq(len(loaded), 2, "2 facts on disk")
    assert_eq(loaded[0]["added_turn"], 4, "turn_index recorded")


# ---------------------------------------------------------------------------
# V2.1 Phase 7 Step 2 — stale-fact archival
# ---------------------------------------------------------------------------

def test_archive_no_facts_is_noop():
    print("\n[test] archive_stale_facts: empty conv → (0, 0)")
    _wipe_storage()
    kept, archived = facts.archive_stale_facts("empty", older_than_days=30)
    assert_eq(kept, 0, "kept=0")
    assert_eq(archived, 0, "archived=0")


def test_archive_all_facts_fresh_is_noop():
    print("\n[test] archive_stale_facts: all facts fresh → 0 archived")
    _wipe_storage()
    now = int(time.time())
    facts.save_facts("fresh-conv", [
        {"text": "still-warm", "added_turn": 1, "last_used": now},
        {"text": "also-warm", "added_turn": 2, "last_used": now - 10},
    ])
    kept, archived = facts.archive_stale_facts("fresh-conv", older_than_days=30)
    assert_eq(kept, 2, "both kept")
    assert_eq(archived, 0, "none archived")
    assert_eq(len(facts.load_archive("fresh-conv")), 0, "archive empty")


def test_archive_moves_stale_facts_to_sidecar():
    print("\n[test] archive_stale_facts: stale facts moved to .archive.json")
    _wipe_storage()
    now = int(time.time())
    stale_ts = now - (100 * 86400)  # 100 days old
    facts.save_facts("mixed", [
        {"text": "fresh", "added_turn": 1, "last_used": now},
        {"text": "ancient", "added_turn": 2, "last_used": stale_ts},
    ])
    kept, archived = facts.archive_stale_facts("mixed", older_than_days=30)
    assert_eq(kept, 1, "1 fresh kept")
    assert_eq(archived, 1, "1 stale archived")
    active = facts.load_facts("mixed")
    assert_eq(len(active), 1, "active has 1 fact")
    assert_eq(active[0]["text"], "fresh", "fresh fact remains active")
    archived_list = facts.load_archive("mixed")
    assert_eq(len(archived_list), 1, "archive has 1 fact")
    assert_eq(archived_list[0]["text"], "ancient", "ancient fact archived")
    assert_true(archived_list[0]["archived_at"] > 0, "archived_at stamped")


def test_archive_accumulates_across_passes():
    print("\n[test] archive_stale_facts: subsequent passes append to archive")
    _wipe_storage()
    now = int(time.time())
    old1 = now - (100 * 86400)
    facts.save_facts("accum", [{"text": "old-A", "added_turn": 1, "last_used": old1}])
    facts.archive_stale_facts("accum", older_than_days=30)
    # Second pass with a new stale fact
    facts.save_facts("accum", [{"text": "old-B", "added_turn": 2, "last_used": old1}])
    kept, archived = facts.archive_stale_facts("accum", older_than_days=30)
    assert_eq(archived, 1, "one new archived")
    a = facts.load_archive("accum")
    assert_eq(len(a), 2, "archive has both A and B (accumulated)")


def test_archive_is_idempotent():
    print("\n[test] archive_stale_facts: re-running with same cutoff is a no-op")
    _wipe_storage()
    now = int(time.time())
    old = now - (100 * 86400)
    facts.save_facts("idem", [{"text": "ancient", "added_turn": 0, "last_used": old}])
    facts.archive_stale_facts("idem", older_than_days=30)
    k2, a2 = facts.archive_stale_facts("idem", older_than_days=30)
    assert_eq(a2, 0, "second pass archives 0")


def test_restore_all_from_archive():
    print("\n[test] restore_from_archive: no filter → restores everything")
    _wipe_storage()
    now = int(time.time())
    old = now - (100 * 86400)
    facts.save_facts("restore-all", [{"text": "ancient", "added_turn": 0, "last_used": old}])
    facts.archive_stale_facts("restore-all", older_than_days=30)
    assert_eq(len(facts.load_facts("restore-all")), 0, "prep: active empty after archive")
    restored = facts.restore_from_archive("restore-all")
    assert_eq(restored, 1, "1 restored")
    active = facts.load_facts("restore-all")
    assert_eq(len(active), 1, "active has 1 fact again")
    assert_eq(active[0]["text"], "ancient", "text preserved")
    assert_true(active[0]["last_used"] > old, "last_used refreshed (no immediate re-archive)")
    assert_true("archived_at" not in active[0], "archived_at dropped on restore")
    assert_eq(len(facts.load_archive("restore-all")), 0, "archive empty after full restore")


def test_restore_with_substring_filter():
    print("\n[test] restore_from_archive: substring filter is case-insensitive")
    _wipe_storage()
    now = int(time.time())
    old = now - (100 * 86400)
    facts.save_facts("filt", [
        {"text": "Lyra is a ranger", "added_turn": 1, "last_used": old},
        {"text": "Aethermere is a kingdom", "added_turn": 2, "last_used": old},
        {"text": "Hippogriffs exist", "added_turn": 3, "last_used": old},
    ])
    facts.archive_stale_facts("filt", older_than_days=30)
    n = facts.restore_from_archive("filt", text_substring="LYRA")
    assert_eq(n, 1, "1 restored matching 'LYRA' (case-insensitive)")
    active = facts.load_facts("filt")
    assert_eq(len(active), 1, "1 active")
    assert_eq(active[0]["text"], "Lyra is a ranger", "right fact restored")
    assert_eq(len(facts.load_archive("filt")), 2, "two remain archived")


def test_restore_substring_no_match_returns_zero():
    print("\n[test] restore_from_archive: no matches → 0, archive untouched")
    _wipe_storage()
    now = int(time.time())
    old = now - (100 * 86400)
    facts.save_facts("nomatch", [{"text": "x", "added_turn": 0, "last_used": old}])
    facts.archive_stale_facts("nomatch", older_than_days=30)
    n = facts.restore_from_archive("nomatch", text_substring="not-present")
    assert_eq(n, 0, "0 restored")
    assert_eq(len(facts.load_archive("nomatch")), 1, "archive still has the fact")


def test_archive_path_is_sidecar_not_facts_file():
    print("\n[test] archive sidecar isn't mistaken for a separate conv by listing")
    _wipe_storage()
    now = int(time.time())
    facts.save_facts("L", [{"text": "x", "added_turn": 0, "last_used": now}])
    facts.save_archive("L", [{"text": "old", "added_turn": 0, "last_used": 0, "archived_at": 0}])
    ids = memory.list_known_conv_ids()
    assert_eq(ids, ["L"], "only one conv listed (sidecar excluded)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_load_facts_missing_file_returns_empty()
        test_load_facts_corrupted_file_returns_empty()
        test_save_load_roundtrip()
        test_save_facts_is_atomic_via_temp_file()
        test_load_facts_drops_malformed_entries()

        test_prune_facts_no_op_under_budget()
        test_prune_facts_lru_eviction()
        test_prune_facts_empty_input()

        test_touch_facts_updates_timestamps()
        test_format_facts_block_empty()
        test_format_facts_block_renders_bullets()

        test_parse_extraction_NONE_returns_empty()
        test_parse_extraction_dash_bullets()
        test_parse_extraction_numbered_list()
        test_parse_extraction_mixed_and_blank_lines()
        test_parse_extraction_drops_too_short()

        test_extract_facts_from_exchange_success()
        test_extract_facts_from_exchange_NONE()
        test_extract_facts_from_exchange_network_failure()
        test_extract_facts_from_exchange_empty_inputs_short_circuit()

        test_record_facts_end_to_end()

        # V2.1 Phase 7 Step 2 — stale-fact archival
        test_archive_no_facts_is_noop()
        test_archive_all_facts_fresh_is_noop()
        test_archive_moves_stale_facts_to_sidecar()
        test_archive_accumulates_across_passes()
        test_archive_is_idempotent()
        test_restore_all_from_archive()
        test_restore_with_substring_filter()
        test_restore_substring_no_match_returns_zero()
        test_archive_path_is_sidecar_not_facts_file()

        print("\nAll facts smoke tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
