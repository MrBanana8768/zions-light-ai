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

        print("\nAll facts smoke tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
