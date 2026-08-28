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


def test_load_facts_corrupted_file_raises():
    print("\n[test] load_facts raises StoreUnreadable for a corrupted JSON file")
    # This test asserted `corrupt file -> []` until v3.1. That was the F1a
    # contract written down as a requirement: every caller that saves read
    # the [] as "no facts here" and wrote it back over the real store. An
    # unreadable file must be distinguishable from an absent one — the test
    # above covers absent, which still returns [] and always will.
    _wipe_storage()
    facts_path = memory.facts_path("corrupt")
    facts_path.write_text("{ not valid json")
    try:
        got = facts.load_facts("corrupt")
    except memory.StoreUnreadable:
        print("  ok   corrupt file -> StoreUnreadable")
        return
    print(f"FAIL corrupt file must not read as a fact store, got {got!r}")
    sys.exit(1)


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
# v3.1 F9 — "LRU" eviction was not LRU, and it deleted permanently
# ---------------------------------------------------------------------------
#
# Two halves to the defect, so two halves to the coverage:
#
#   1. The request path loaded the whole store and touched the whole store,
#      so last_used was the same second on every fact at all times. The sort
#      key (last_used, added_turn) collapsed onto added_turn and eviction
#      became "drop whatever was added earliest" — the conversation's
#      foundational facts, every turn, past the token cap.
#   2. Nothing calls archive_stale_facts automatically (admin endpoints
#      only), so an eviction was an unlink with no cold-storage fallback.
#
# test_prune_facts_lru_eviction above still passes and always did; it
# constructs the differentiated last_used values the live pipeline could
# never produce. These tests drive the pipeline's own sequence instead.


def _f(text, added_turn, last_used):
    return {"text": text, "added_turn": added_turn, "last_used": last_used}


def test_select_for_injection_is_the_whole_store_under_budget():
    print("\n[test] select_for_injection returns everything when it fits")
    items = [_f("x" * 100, i, 500) for i in range(3)]
    selected = facts.select_for_injection(items, max_tokens=1000)
    assert_eq(len(selected), 3, "all 3 selected")
    assert_true(
        all(a is b for a, b in zip(selected, items)),
        "selection shares the caller's dicts (so touching it touches the store)",
    )


def test_turn_touches_only_the_injected_facts():
    print("\n[test] a turn touches only the injected facts, not all 200")
    _wipe_storage()
    cid = "lru-200"
    now = int(time.time())
    stale = now - 10000
    # The three the model has actually been using are the FOUNDATIONAL ones
    # — added_turn 0-2, the ones the old sort key evicted first. Each fact is
    # 100 chars ≈ 25 tokens, so a 75-token budget admits exactly three.
    store = [_f("who she is " + "0" * 89, 0, now)]
    store += [_f("where she lives " + "1" * 84, 1, now)]
    store += [_f("what she wants " + "2" * 85, 2, now)]
    store += [_f(f"passing detail {i:03d} " * 5, i, stale) for i in range(3, 200)]
    assert_eq(len(store), 200, "prep: 200 facts")
    facts.save_facts(cid, store)

    # --- request path ---
    on_disk = facts.load_facts(cid)
    injected = facts.select_for_injection(on_disk, max_tokens=75)
    assert_eq(len(injected), 3, "3 facts injected")
    facts.touch_facts(injected, now=now + 500)
    block = facts.format_facts_block(injected)
    assert_true("who she is" in block, "the injected block carries the identity facts")

    # The injected subset shares its dicts with the full list, so the touch
    # lands in `on_disk` without touching the other 197. main.py's tail
    # re-reads under the lock and carries these forward by text via
    # _merge_touched, which only ever moves last_used forward — same result.
    touched = [f for f in on_disk if f["last_used"] == now + 500]
    untouched = [f for f in on_disk if f["last_used"] != now + 500]
    assert_eq(len(touched), 3, "exactly 3 facts touched")
    assert_eq(len(untouched), 197, "the other 197 keep their old last_used")
    assert_true(
        all(f["last_used"] == stale for f in untouched),
        "untouched facts still carry the ORIGINAL last_used, not now",
    )
    assert_eq(
        sorted(f["added_turn"] for f in touched), [0, 1, 2],
        "the touched facts are the foundational ones the model is using",
    )


def test_eviction_archives_instead_of_deleting():
    print("\n[test] over-budget eviction lands in the archive file, not /dev/null")
    _wipe_storage()
    cid = "lru-200"
    now = int(time.time())
    stale = now - 10000
    store = [_f("who she is " + "0" * 89, 0, now)]
    store += [_f("where she lives " + "1" * 84, 1, now)]
    store += [_f("what she wants " + "2" * 85, 2, now)]
    store += [_f(f"passing detail {i:03d} " * 5, i, stale) for i in range(3, 200)]
    facts.save_facts(cid, store)

    kept, dropped = facts.prune_facts(store, max_tokens=75, conv_id=cid)
    facts.save_facts(cid, kept)
    assert_eq(len(kept), 3, "3 facts survive the budget")
    assert_eq(dropped, 197, "197 evicted")
    assert_eq(
        sorted(f["added_turn"] for f in kept), [0, 1, 2],
        "recently-used foundational facts survive — NOT evicted as 'oldest'",
    )

    # Read the sidecar off disk, not through load_archive — the claim is
    # that the facts are on the filesystem.
    sidecar = memory.facts_archive_path(cid)
    assert_true(sidecar.is_file(), "the archive sidecar exists after an eviction")
    raw = json.loads(sidecar.read_text())
    archived_texts = {f["text"] for f in raw["facts"]}
    assert_eq(len(archived_texts), 197, "all 197 evicted facts are in the file")
    evicted_texts = {f["text"] for f in store if f["added_turn"] >= 3}
    assert_eq(archived_texts, evicted_texts, "the archive holds exactly the evicted set")
    assert_true(
        all(f["archived_at"] > 0 for f in raw["facts"]), "archived_at stamped"
    )

    # And they are recoverable the way the user would recover them.
    restored = facts.restore_from_archive(cid, text_substring="passing detail 042")
    assert_eq(restored, 1, "an evicted fact restores from cold storage")


def test_eviction_without_conv_id_logs_the_texts():
    print("\n[test] prune_facts with no conv_id warns and names what it dropped")
    items = [_f("secret ingredient is nutmeg" + "y" * 70, 1, 100), _f("x" * 100, 2, 500)]
    with patch.object(facts.logger, "warning") as warn:
        kept, dropped = facts.prune_facts(items, max_tokens=25)
    assert_eq(dropped, 1, "1 evicted")
    assert_eq(len(kept), 1, "1 kept")
    assert_true(warn.called, "the unarchivable eviction is logged at WARNING")
    msg = warn.call_args[0][0]
    assert_true("no archive to land in" in msg, "warning says why it could not archive")
    assert_true("secret ingredient" in msg, "warning carries the dropped text verbatim")


def test_eviction_keeps_facts_when_the_archive_write_fails():
    print("\n[test] a failed archive write evicts NOTHING")
    _wipe_storage()
    cid = "archive-broken"
    items = [_f("x" * 100, 1, 100), _f("y" * 100, 2, 500), _f("z" * 100, 3, 999)]
    with patch.object(facts, "save_archive", side_effect=OSError("EIO")):
        kept, dropped = facts.prune_facts(items, max_tokens=25, conv_id=cid)
    assert_eq(dropped, 0, "nothing reported dropped")
    assert_eq(len(kept), 3, "all 3 facts returned — over budget beats destroyed")
    assert_eq(
        [f["text"] for f in kept], [f["text"] for f in items],
        "the caller gets its store back intact",
    )


def test_re_archiving_a_fact_replaces_rather_than_duplicates():
    print("\n[test] archiving the same fact twice keeps one entry, not two")
    _wipe_storage()
    cid = "roundtrip"
    # A fact evicted for budget, re-established by the user (or re-extracted
    # from a later exchange), then evicted again — without a restore in
    # between, so copy #1 is still sitting in the sidecar.
    facts.archive_facts(cid, [_f("Lyra is a half-elf ranger", 1, 100)])
    facts.archive_facts(cid, [_f("Lyra is a half-elf ranger", 1, 200)])
    archived = facts.load_archive(cid)
    assert_eq(len(archived), 1, "one entry, not one per eviction")
    assert_eq(archived[0]["last_used"], 200, "the newer entry is the one kept")
    # And a full restore brings back one copy, not N.
    assert_eq(facts.restore_from_archive(cid), 1, "restores a single copy")


def test_record_facts_for_exchange_archives_its_evictions():
    print("\n[test] record_facts_for_exchange routes eviction to the archive")
    _wipe_storage()
    cid = "tail-evict"
    now = int(time.time())
    # 100 facts x ~100 chars = ~2500 tokens, well past the real 1500-token
    # default. No patching: prune_facts binds _MAX_FACTS_TOKENS as a default
    # argument at import, so a patched module attribute would not be read.
    seeded = [_f(f"old fact {i:03d} " + "o" * 85, i, now - 10000) for i in range(100)]
    facts.save_facts(cid, seeded)
    client = _mock_client_returning("- Character Lyra is a ranger.")
    n = asyncio.run(facts.record_facts_for_exchange(
        cid, client, "http://fake", "fake-model",
        user_msg="Who is Lyra?",
        assistant_msg="A half-elf ranger.",
        turn_index=99,
    ))
    assert_eq(n, 1, "1 new fact added")
    active = facts.load_facts(cid)
    archived = facts.load_archive(cid)
    assert_true(len(active) < 101, "the store was pruned to the token budget")
    assert_true(len(archived) > 0, "the evicted facts went to cold storage")
    assert_eq(len(active) + len(archived), 101, "every fact is still somewhere")


# ---------------------------------------------------------------------------
# v3.1 — fact extraction had no INPUT budget
# ---------------------------------------------------------------------------
#
# _EXTRACTION_MAX_TOKENS bounded the output and nothing bounded the input. The
# payload is the extraction system prompt + the ENTIRE fact store + the full
# user turn + the full assistant reply; the chat path sends no max_tokens, so
# vLLM may generate a reply of nearly the whole window and extraction then
# stacks the prompt and the store on top of it. Two calls in the 2026-08-24
# window were rejected at 33,790 and 33,581 input tokens against a 32,768
# window.
#
# What a rejection cost, precisely: the warning fired and no "+N facts" line
# followed it. The exchange's facts were never extracted, nothing retried, and
# the warning named no conversation — so the log could not say which
# conversation had lost the turn. Hence the third test here.


def _payload_of(client) -> dict:
    """The JSON body the mocked client was actually asked to POST."""
    return client.post.call_args.kwargs["json"]


def _payload_tokens(payload: dict) -> int:
    """Size the assembled request the way facts.py sizes everything else."""
    return sum(facts._estimate_tokens(m["content"]) for m in payload["messages"])


def _oversized_store() -> list[dict]:
    """A fact store that on its own exceeds the real extraction input budget."""
    store = [_f(f"established fact {i:04d} " + "z" * 80, i, 1000 + i) for i in range(1500)]
    assert_true(
        sum(facts._estimate_tokens(x["text"]) for x in store)
        > facts._EXTRACTION_INPUT_BUDGET,
        "prep: the store alone is over the extraction input budget",
    )
    return store


def test_oversized_store_is_trimmed_not_rejected():
    print("\n[test] an over-budget store is trimmed to fit, and the call still happens")
    client = _mock_client_returning("- Lyra carries a yew bow.")
    out = asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model",
        "What does Lyra carry?", "A yew bow she made herself.",
        _oversized_store(), conv_id="budget",
    ))
    assert_eq(out, ["Lyra carries a yew bow."], "the extraction returned its fact")
    assert_true(client.post.called, "the call went out — trimmed, not skipped")
    assert_true(
        _payload_tokens(_payload_of(client)) <= facts._EXTRACTION_INPUT_BUDGET,
        "the assembled payload fits the input budget",
    )


def test_the_exchange_survives_trimming_in_preference_to_the_store():
    print("\n[test] trimming sheds the store first — the exchange is the new information")
    client = _mock_client_returning("NONE")
    store = _oversized_store()
    user = "Her sister is named Isolde and she runs the mill at Varrow Ford."
    asst = "Isolde keeps the mill turning while Lyra is away."
    asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model", user, asst, store, conv_id="budget",
    ))
    body = _payload_of(client)["messages"][-1]["content"]
    assert_true(user in body, "the user turn reached the model verbatim")
    assert_true(asst in body, "the assistant reply reached the model verbatim")
    assert_true(
        facts._TRIM_NOTE not in body,
        "neither half of the exchange was truncated to make room",
    )
    # The store is what paid for it. Not zero — it gets whatever the exchange
    # left — but far less than the 1500 it was handed.
    kept = [x for x in store if f"- {x['text']}" in body]
    shed = [x for x in store if f"- {x['text']}" not in body]
    assert_true(kept, "the store was narrowed, not emptied")
    assert_true(shed, "the store is what got shed")
    assert_true(
        min(x["last_used"] for x in kept) > max(x["last_used"] for x in shed),
        "the facts kept are the most-recently-used ones — the same order "
        "select_for_injection and prune_facts use, not an arbitrary slice",
    )


def test_the_assistant_reply_is_shed_before_the_user_turn():
    print("\n[test] when the exchange itself does not fit, the reply goes first")
    client = _mock_client_returning("NONE")
    user = "Remember: the Varrow Ford mill burned down in the third winter."
    asst = "Understood. " + "The mill is gone. " * 4000
    # A budget the exchange alone cannot fit, passed explicitly so the test does
    # not depend on MAX_MODEL_LEN.
    asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model", user, asst, [],
        conv_id="budget", max_input_tokens=facts._extraction_overhead_tokens() + 600,
    ))
    body = _payload_of(client)["messages"][-1]["content"]
    assert_true(user in body, "the user turn survives whole")
    assert_true(asst not in body, "the assistant reply did not")
    assert_true(facts._TRIM_NOTE in body, "and the model is told it was truncated")
    assert_true(
        _payload_tokens(_payload_of(client))
        <= facts._extraction_overhead_tokens() + 600,
        "the trimmed payload fits the budget it was given",
    )


def test_a_user_turn_larger_than_the_budget_is_truncated_not_dropped():
    print("\n[test] even an oversized user turn is truncated, never emptied")
    client = _mock_client_returning("NONE")
    asyncio.run(facts.extract_facts_from_exchange(
        client, "http://fake", "fake-model",
        "Lyra " * 8000, "Noted.", [],
        conv_id="budget", max_input_tokens=facts._extraction_overhead_tokens() + 300,
    ))
    body = _payload_of(client)["messages"][-1]["content"]
    # An empty message would short-circuit the whole call on the next turn and
    # turn a budget overflow back into a silently lost exchange.
    assert_true("[user]: Lyra" in body, "the user turn still carries its content")
    assert_true(facts._TRIM_NOTE in body, "truncation is marked, not silent")


def test_extraction_failure_is_an_error_that_names_the_conversation():
    print("\n[test] a failed extraction logs at ERROR and says which conversation")
    client = _mock_client_raising(RuntimeError("400 Bad Request: maximum context length"))
    with patch.object(facts.logger, "error") as err, \
         patch.object(facts.logger, "warning") as warn:
        out = asyncio.run(facts.extract_facts_from_exchange(
            client, "http://fake", "fake-model", "hi", "hello", [],
            conv_id="conv-abc123",
        ))
    assert_eq(out, [], "still non-fatal to the chat path")
    assert_true(err.called, "a lost extraction is an ERROR, not a WARNING")
    assert_true(not warn.called, "and not also a warning")
    msg = err.call_args[0][0]
    assert_true("conv=conv-abc123" in msg, "the log names the conversation")
    assert_true("lost" in msg, "the log says the facts are gone, not merely that a call failed")


def test_extraction_failure_without_conv_id_says_the_caller_gave_none():
    print("\n[test] a caller that passes no conv_id is named as the reason")
    # main.py's async tail is that caller at HEAD — it has conv_id in scope and
    # does not pass it. The line must not read as if the conversation were
    # unknowable.
    client = _mock_client_raising(RuntimeError("connection refused"))
    with patch.object(facts.logger, "error") as err:
        asyncio.run(facts.extract_facts_from_exchange(
            client, "http://fake", "fake-model", "hi", "hello", [],
        ))
    assert_true("caller passed none" in err.call_args[0][0], "says who is at fault")


def test_record_facts_for_exchange_passes_its_conv_id_through():
    print("\n[test] record_facts_for_exchange attributes its own failures")
    _wipe_storage()
    client = _mock_client_raising(RuntimeError("connection refused"))
    with patch.object(facts.logger, "error") as err:
        n = asyncio.run(facts.record_facts_for_exchange(
            "attributed", client, "http://fake", "fake-model",
            user_msg="Who is Lyra?", assistant_msg="A ranger.", turn_index=1,
        ))
    assert_eq(n, 0, "no facts added")
    assert_true("conv=attributed" in err.call_args[0][0], "the conv_id reached the log")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_load_facts_missing_file_returns_empty()
        test_load_facts_corrupted_file_raises()
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

        # v3.1 F9 — LRU eviction is not LRU, and it deletes permanently
        test_select_for_injection_is_the_whole_store_under_budget()
        test_turn_touches_only_the_injected_facts()
        test_eviction_archives_instead_of_deleting()
        test_eviction_without_conv_id_logs_the_texts()
        test_eviction_keeps_facts_when_the_archive_write_fails()
        test_re_archiving_a_fact_replaces_rather_than_duplicates()
        test_record_facts_for_exchange_archives_its_evictions()

        # v3.1 — fact extraction had no input budget
        test_oversized_store_is_trimmed_not_rejected()
        test_the_exchange_survives_trimming_in_preference_to_the_store()
        test_the_assistant_reply_is_shed_before_the_user_turn()
        test_a_user_turn_larger_than_the_budget_is_truncated_not_dropped()
        test_extraction_failure_is_an_error_that_names_the_conversation()
        test_extraction_failure_without_conv_id_says_the_caller_gave_none()
        test_record_facts_for_exchange_passes_its_conv_id_through()

        print("\nAll facts smoke tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
