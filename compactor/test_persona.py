"""
CPU-only Tier-1 tests for compactor.persona.

Real tmpdir storage (no mocks needed — persona module is pure I/O +
hashing). Covers:
  - I/O round-trip + idempotent save on matching hash
  - clear / list
  - auto-detection from messages (threshold, role check, multimodal)
  - text_to_inject hash-matching guard (no double-injection)
  - format_persona_block output
  - library listing

Run: python test_persona.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_persona_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_PERSONA_AUTO_DETECT_MIN_CHARS"] = "50"  # shorter for tests

import memory  # noqa: E402
import persona  # noqa: E402


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


def _wipe():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


# ---------------------------------------------------------------------------
# I/O round-trip
# ---------------------------------------------------------------------------

def test_load_missing_returns_none():
    print("\n[test] load_persona returns None when no file exists")
    _wipe()
    assert_eq(persona.load_persona("nope"), None, "missing → None")
    assert_eq(persona.get_persona_text("nope"), None, "convenience → None")


def test_save_and_load_round_trip():
    print("\n[test] save_persona / load_persona round-trip")
    _wipe()
    saved = persona.save_persona("rt", "You are a wandering bard.", source="admin")
    assert_eq(saved["persona_text"], "You are a wandering bard.", "text returned")
    assert_eq(saved["source"], "admin", "source recorded")
    assert_true(saved["hash"], "hash computed")
    loaded = persona.load_persona("rt")
    assert_eq(loaded["persona_text"], "You are a wandering bard.", "load matches")
    assert_eq(loaded["hash"], saved["hash"], "hash stable")


def test_save_strips_whitespace():
    print("\n[test] save_persona strips surrounding whitespace")
    _wipe()
    persona.save_persona("ws", "   leading and trailing space   ")
    assert_eq(
        persona.get_persona_text("ws"),
        "leading and trailing space",
        "whitespace stripped",
    )


def test_save_rejects_empty():
    print("\n[test] save_persona raises ValueError on empty text")
    _wipe()
    try:
        persona.save_persona("e", "   ")
        print("FAIL: no exception raised")
        sys.exit(1)
    except ValueError:
        print("  ok   ValueError raised on empty text")


def test_save_idempotent_on_matching_hash():
    print("\n[test] save_persona with same text → returns existing record, no timestamp churn")
    _wipe()
    first = persona.save_persona("idem", "stable text", source="admin")
    import time
    time.sleep(0.01)  # ensure clock advance
    second = persona.save_persona("idem", "stable text", source="auto")
    # set_at should be the first save, not updated
    assert_eq(first["set_at"], second["set_at"], "set_at unchanged")
    # source should also not be updated (auto-capture path shouldn't overwrite admin)
    assert_eq(second["source"], first["source"], "source preserved")


def test_save_replaces_on_different_text():
    print("\n[test] save_persona with different text overwrites prior record")
    _wipe()
    persona.save_persona("rep", "first text version", source="admin")
    rec = persona.save_persona("rep", "second text version", source="admin")
    assert_eq(rec["persona_text"], "second text version", "text updated")
    loaded = persona.load_persona("rep")
    assert_eq(loaded["persona_text"], "second text version", "load shows new")


# ---------------------------------------------------------------------------
# clear / list
# ---------------------------------------------------------------------------

def test_clear_returns_true_when_existed():
    print("\n[test] clear_persona returns True after a save")
    _wipe()
    persona.save_persona("clr", "to be cleared")
    assert_eq(persona.clear_persona("clr"), True, "True on existing")
    assert_eq(persona.load_persona("clr"), None, "load now None")


def test_clear_returns_false_when_absent():
    print("\n[test] clear_persona on non-existent conv → False, no error")
    _wipe()
    assert_eq(persona.clear_persona("never-existed"), False, "False on missing")


def test_list_personas_empty():
    print("\n[test] list_personas: returns [] when no personas stored")
    _wipe()
    assert_eq(persona.list_personas(), [], "empty list")


def test_list_personas_sorted_by_set_at():
    print("\n[test] list_personas: sorted newest-first; returns metadata only (no text)")
    _wipe()
    persona.save_persona("a", "alpha persona text here long enough")
    import time; time.sleep(0.01)
    persona.save_persona("b", "beta persona text here long enough")
    out = persona.list_personas()
    assert_eq(len(out), 2, "two entries")
    assert_eq(out[0]["conv_id"], "b", "newest first")
    assert_eq(out[1]["conv_id"], "a", "oldest last")
    # Each entry: lightweight (no full text)
    for entry in out:
        assert_true("length" in entry, f"length present: {entry!r}")
        assert_true("set_at" in entry, "set_at present")
        assert_true("source" in entry, "source present")
        assert_true("hash" in entry, "hash present (short)")
        assert_true("persona_text" not in entry, "full text NOT included (library view)")


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def test_detect_empty_messages():
    print("\n[test] detect_persona_in_messages: empty messages → None")
    assert_eq(persona.detect_persona_in_messages([]), None, "empty list")
    assert_eq(persona.detect_persona_in_messages([{"role": "user", "content": "hi"}]),
              None, "no system message")


def test_detect_short_system_message_ignored():
    print("\n[test] detect_persona_in_messages: short system message NOT a persona")
    msgs = [{"role": "system", "content": "be concise"}]
    assert_eq(persona.detect_persona_in_messages(msgs), None, "too short → None")


def test_detect_long_system_message_returned():
    print("\n[test] detect_persona_in_messages: long system message returned")
    text = "You are a wandering bard from the kingdom of Aethermere. " * 3  # > 50 chars
    msgs = [{"role": "system", "content": text}]
    out = persona.detect_persona_in_messages(msgs)
    assert_eq(out, text.strip(), "text returned and stripped")


def test_detect_multimodal_content_joined():
    print("\n[test] detect_persona_in_messages: multimodal text parts joined")
    parts = [
        {"type": "text", "text": "You are a wandering bard "},
        {"type": "text", "text": "from the kingdom of Aethermere."},
    ]
    # Long enough only when joined
    msgs = [{"role": "system", "content": parts + parts}]
    out = persona.detect_persona_in_messages(msgs)
    assert_true(out and "wandering bard" in out, "joined text returned")


def test_detect_ignores_user_role_at_position_0():
    print("\n[test] detect_persona_in_messages: first message non-system → None")
    long_text = "x" * 300
    msgs = [{"role": "user", "content": long_text}]
    assert_eq(persona.detect_persona_in_messages(msgs), None, "user-role ignored")


def test_auto_capture_stores_on_first_sight():
    print("\n[test] auto_capture_persona: first sight stores it")
    _wipe()
    text = "You are a wandering bard from Aethermere. " * 3
    msgs = [{"role": "system", "content": text}]
    saved = persona.auto_capture_persona("auto1", msgs)
    assert_true(saved is not None, "returns saved record")
    assert_eq(saved["source"], "auto", "source=auto")
    assert_eq(persona.get_persona_text("auto1"), text.strip(), "stored")


def test_auto_capture_noop_on_matching_hash():
    print("\n[test] auto_capture_persona: matching hash → no-op, returns None")
    _wipe()
    text = "You are a wandering bard from Aethermere. " * 3
    msgs = [{"role": "system", "content": text}]
    persona.auto_capture_persona("auto2", msgs)
    second = persona.auto_capture_persona("auto2", msgs)
    assert_eq(second, None, "second call → None (no churn)")


def test_auto_capture_replaces_on_different_text():
    print("\n[test] auto_capture_persona: new text → overwrites")
    _wipe()
    msgs1 = [{"role": "system", "content": "Old persona text. " * 5}]
    msgs2 = [{"role": "system", "content": "New persona text. " * 5}]
    persona.auto_capture_persona("auto3", msgs1)
    rec = persona.auto_capture_persona("auto3", msgs2)
    assert_true(rec is not None, "second save happens")
    assert_true("New persona" in persona.get_persona_text("auto3"), "new text stored")


def test_auto_capture_disabled_when_persona_disabled(monkeypatch=None):
    print("\n[test] auto_capture_persona: returns None when feature disabled")
    _wipe()
    original = persona._ENABLED
    try:
        persona._ENABLED = False
        msgs = [{"role": "system", "content": "x" * 300}]
        assert_eq(persona.auto_capture_persona("d", msgs), None, "disabled → None")
        assert_eq(persona.load_persona("d"), None, "nothing stored")
    finally:
        persona._ENABLED = original


# ---------------------------------------------------------------------------
# text_to_inject — double-injection guard
# ---------------------------------------------------------------------------

def test_text_to_inject_none_when_no_persona():
    print("\n[test] text_to_inject: None when no persona stored")
    _wipe()
    assert_eq(persona.text_to_inject("none", []), None, "no record → None")


def test_text_to_inject_none_when_already_in_request():
    print("\n[test] text_to_inject: None when persona already in messages[0]")
    _wipe()
    text = "You are a noir detective. " * 5
    persona.save_persona("dup", text)
    msgs = [{"role": "system", "content": text}]
    out = persona.text_to_inject("dup", msgs)
    assert_eq(out, None, "no double-injection — already in request")


def test_text_to_inject_returns_text_when_not_in_request():
    print("\n[test] text_to_inject: returns text when not in messages (admin-set / inherited)")
    _wipe()
    text = "You are a noir detective. " * 5
    persona.save_persona("missing", text, source="admin")
    msgs = [{"role": "user", "content": "hi"}]  # no system message
    out = persona.text_to_inject("missing", msgs)
    assert_eq(out, text.strip(), "injectable text returned")


def test_text_to_inject_returns_text_when_different_system_msg():
    print("\n[test] text_to_inject: returns text when messages[0] is a DIFFERENT system msg")
    _wipe()
    persona.save_persona("diff", "Stored persona text. " * 5)
    msgs = [{"role": "system", "content": "be concise"}]  # short, unrelated
    out = persona.text_to_inject("diff", msgs)
    assert_true(out, "returns text — first sys msg doesn't match stored")


# ---------------------------------------------------------------------------
# format_persona_block
# ---------------------------------------------------------------------------

def test_format_persona_block_none():
    print("\n[test] format_persona_block(None) → None")
    assert_eq(persona.format_persona_block(None), None, "None in → None out")
    assert_eq(persona.format_persona_block(""), None, "empty in → None out")


def test_format_persona_block_includes_header():
    print("\n[test] format_persona_block prepends an explicit header")
    out = persona.format_persona_block("Bard from Aethermere")
    assert_true("[Persona" in out, "header present")
    assert_true("Bard from Aethermere" in out, "body present")


# ---------------------------------------------------------------------------
# enabled() flag
# ---------------------------------------------------------------------------

def test_enabled_reflects_module_state():
    print("\n[test] persona.enabled() reflects _ENABLED")
    original = persona._ENABLED
    try:
        persona._ENABLED = True
        assert_eq(persona.enabled(), True, "True when enabled")
        persona._ENABLED = False
        assert_eq(persona.enabled(), False, "False when disabled")
    finally:
        persona._ENABLED = original


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_load_missing_returns_none,
        test_save_and_load_round_trip,
        test_save_strips_whitespace,
        test_save_rejects_empty,
        test_save_idempotent_on_matching_hash,
        test_save_replaces_on_different_text,
        test_clear_returns_true_when_existed,
        test_clear_returns_false_when_absent,
        test_list_personas_empty,
        test_list_personas_sorted_by_set_at,
        test_detect_empty_messages,
        test_detect_short_system_message_ignored,
        test_detect_long_system_message_returned,
        test_detect_multimodal_content_joined,
        test_detect_ignores_user_role_at_position_0,
        test_auto_capture_stores_on_first_sight,
        test_auto_capture_noop_on_matching_hash,
        test_auto_capture_replaces_on_different_text,
        test_auto_capture_disabled_when_persona_disabled,
        test_text_to_inject_none_when_no_persona,
        test_text_to_inject_none_when_already_in_request,
        test_text_to_inject_returns_text_when_not_in_request,
        test_text_to_inject_returns_text_when_different_system_msg,
        test_format_persona_block_none,
        test_format_persona_block_includes_header,
        test_enabled_reflects_module_state,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll persona smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
