"""
CPU-only Tier-1 tests for compactor.commands.

Covers:
  - parse_command: prefix detection + aliases + non-command pass-through
  - each command handler against real tmpdir storage
  - build_synthetic_completion / build_synthetic_completion_stream shape

Run: python test_commands.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_commands_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import commands  # noqa: E402
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


def _wipe():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------

def test_parse_empty_returns_none():
    print("\n[test] parse_command: empty/None → (None, '')")
    assert_eq(commands.parse_command(""), (None, ""), "empty")
    assert_eq(commands.parse_command(None), (None, ""), "None")


def test_parse_non_slash_returns_none():
    print("\n[test] parse_command: messages without leading `/` pass through")
    assert_eq(commands.parse_command("hello"), (None, ""), "plain")
    assert_eq(commands.parse_command("   plain text"), (None, ""), "leading ws then plain")


def test_parse_unknown_slash_returns_none():
    print("\n[test] parse_command: unknown command → pass through (no false match)")
    # Critical: paths like /usr/bin must NOT trigger commands
    assert_eq(commands.parse_command("/usr/bin/foo"), (None, ""), "path-like")
    assert_eq(commands.parse_command("/nonsense"), (None, ""), "unknown command")
    assert_eq(commands.parse_command("/  "), (None, ""), "just slash + spaces")


def test_parse_canonical_commands():
    print("\n[test] parse_command: canonical command names")
    assert_eq(commands.parse_command("/help"), ("help", ""), "help")
    assert_eq(commands.parse_command("/list-facts"), ("list-facts", ""), "list-facts")
    assert_eq(commands.parse_command("/remember a fact"), ("remember", "a fact"), "remember with arg")
    assert_eq(commands.parse_command("/forget"), ("forget", ""), "forget no arg")
    assert_eq(commands.parse_command("/forget Lyra"), ("forget", "Lyra"), "forget with arg")
    assert_eq(commands.parse_command("/why"), ("why", ""), "why")


def test_parse_aliases_resolve_to_canonical():
    print("\n[test] parse_command: aliases resolve correctly")
    assert_eq(commands.parse_command("/facts"), ("list-facts", ""), "/facts → list-facts")
    assert_eq(commands.parse_command("/archive"), ("list-archive", ""), "/archive → list-archive")
    assert_eq(commands.parse_command("/why-did-you-say-that"),
              ("why", ""), "long form why")
    assert_eq(commands.parse_command("/?"), ("help", ""), "? → help")


def test_parse_is_case_insensitive():
    print("\n[test] parse_command: command name is case-insensitive")
    assert_eq(commands.parse_command("/Help"), ("help", ""), "/Help")
    assert_eq(commands.parse_command("/LIST-FACTS"), ("list-facts", ""), "ALL CAPS")
    assert_eq(commands.parse_command("/ReMeMbEr foo"), ("remember", "foo"), "mixed case")


def test_parse_preserves_arg_text():
    print("\n[test] parse_command: arg preserves original casing and inner whitespace")
    cmd, arg = commands.parse_command("/remember Lyra  is a   HALF-ELF.")
    assert_eq(cmd, "remember", "command name")
    assert_eq(arg, "Lyra  is a   HALF-ELF.", "arg preserved with inner whitespace")


def test_parse_handles_leading_whitespace():
    print("\n[test] parse_command: tolerates leading whitespace before slash")
    cmd, arg = commands.parse_command("   /list-facts")
    assert_eq(cmd, "list-facts", "matched after leading whitespace")


# ---------------------------------------------------------------------------
# handle_command — /help
# ---------------------------------------------------------------------------

def test_help_lists_all_commands():
    print("\n[test] /help mentions every documented command")
    out = asyncio.run(commands.handle_command("help", "", "any-conv"))
    for token in ("/list-facts", "/list-archive", "/remember",
                  "/forget", "/why", "/help"):
        assert_true(token in out, f"help mentions {token}")


# ---------------------------------------------------------------------------
# handle_command — /list-facts
# ---------------------------------------------------------------------------

def test_list_facts_empty_conv():
    print("\n[test] /list-facts on empty conv → friendly message")
    _wipe()
    out = asyncio.run(commands.handle_command("list-facts", "", "empty"))
    assert_true("No facts" in out, "empty message returned")


def test_list_facts_renders_all_entries():
    print("\n[test] /list-facts renders every fact as a bullet")
    _wipe()
    facts.save_facts("c1", [
        {"text": "Alpha", "added_turn": 0, "last_used": 100},
        {"text": "Beta", "added_turn": 1, "last_used": 101},
    ])
    out = asyncio.run(commands.handle_command("list-facts", "", "c1"))
    assert_true("Alpha" in out, "first fact rendered")
    assert_true("Beta" in out, "second fact rendered")
    assert_true("(2)" in out, "count rendered")


# ---------------------------------------------------------------------------
# handle_command — /list-archive
# ---------------------------------------------------------------------------

def test_list_archive_empty():
    print("\n[test] /list-archive empty → friendly message")
    _wipe()
    out = asyncio.run(commands.handle_command("list-archive", "", "x"))
    assert_true("No archived" in out, "empty message")


def test_list_archive_renders_entries():
    print("\n[test] /list-archive renders archive sidecar contents")
    _wipe()
    facts.save_archive("a1", [
        {"text": "OldThing", "added_turn": 0, "last_used": 0, "archived_at": 100},
    ])
    out = asyncio.run(commands.handle_command("list-archive", "", "a1"))
    assert_true("OldThing" in out, "archive entry rendered")


# ---------------------------------------------------------------------------
# handle_command — /remember
# ---------------------------------------------------------------------------

def test_remember_requires_arg():
    print("\n[test] /remember without arg returns usage hint")
    out = asyncio.run(commands.handle_command("remember", "", "any"))
    assert_true("Usage" in out, "usage hint")


def test_remember_persists_fact():
    print("\n[test] /remember <text> appends a fact and persists it")
    _wipe()
    out = asyncio.run(commands.handle_command(
        "remember", "Lyra is left-handed", "rmb", ctx={"turn_index": 7},
    ))
    assert_true("Remembered" in out, "confirmation in output")
    loaded = facts.load_facts("rmb")
    assert_eq(len(loaded), 1, "1 fact stored")
    assert_eq(loaded[0]["text"], "Lyra is left-handed", "text matches")
    assert_eq(loaded[0]["added_turn"], 7, "turn_index from ctx")


def test_remember_rejects_too_long():
    print("\n[test] /remember rejects facts over 500 chars")
    _wipe()
    out = asyncio.run(commands.handle_command(
        "remember", "x" * 600, "long", ctx={"turn_index": 1},
    ))
    assert_true("too long" in out.lower(), "rejection message")
    assert_eq(facts.load_facts("long"), [], "not persisted")


# ---------------------------------------------------------------------------
# handle_command — /forget
# ---------------------------------------------------------------------------

def test_forget_with_substring_removes_only_matches():
    print("\n[test] /forget <substring> removes only matching facts")
    _wipe()
    facts.save_facts("fg", [
        {"text": "Lyra is a ranger", "added_turn": 0, "last_used": 0},
        {"text": "Aethermere is the kingdom", "added_turn": 1, "last_used": 0},
        {"text": "Lyra has a hawk companion", "added_turn": 2, "last_used": 0},
    ])
    out = asyncio.run(commands.handle_command("forget", "Lyra", "fg"))
    assert_true("Forgot 2 fact(s)" in out, f"correct count message: {out!r}")
    remaining = facts.load_facts("fg")
    assert_eq(len(remaining), 1, "1 fact remains")
    assert_eq(remaining[0]["text"], "Aethermere is the kingdom",
              "non-matching fact preserved")


def test_forget_substring_no_match():
    print("\n[test] /forget <substring> with no matches → 0 removed message")
    _wipe()
    facts.save_facts("fg2", [{"text": "X", "added_turn": 0, "last_used": 0}])
    out = asyncio.run(commands.handle_command("forget", "MissingThing", "fg2"))
    assert_true("No facts matched" in out, "explicit no-match message")
    assert_eq(len(facts.load_facts("fg2")), 1, "fact untouched")


def test_forget_no_arg_invokes_clear_all_helper():
    print("\n[test] /forget (no arg) calls clear_all_memory helper from ctx")
    _wipe()
    cleared = {"called": False, "conv_id": None}

    async def fake_clear(cid):
        cleared["called"] = True
        cleared["conv_id"] = cid
        return {
            "conv_id": cid,
            "forgotten_facts": 3,
            "forgotten_episodic": 5,
            "forgotten_summary": True,
        }

    out = asyncio.run(commands.handle_command(
        "forget", "", "wipe-me", ctx={"clear_all_memory": fake_clear},
    ))
    assert_eq(cleared["called"], True, "helper invoked")
    assert_eq(cleared["conv_id"], "wipe-me", "conv_id passed")
    assert_true("3 fact(s)" in out, "fact count in output")
    assert_true("5 indexed" in out, "episodic count in output")
    assert_true("summary state" in out, "summary mentioned")


def test_forget_no_arg_without_helper_returns_error():
    print("\n[test] /forget (no arg) without clear_all_memory ctx → error message")
    out = asyncio.run(commands.handle_command("forget", "", "x", ctx={}))
    assert_true("ERROR" in out, "helper-missing error")


def test_forget_no_arg_nothing_to_clear():
    print("\n[test] /forget (no arg) on empty conv → friendly nothing message")
    async def fake_clear(cid):
        return {
            "conv_id": cid,
            "forgotten_facts": 0,
            "forgotten_episodic": 0,
            "forgotten_summary": False,
        }

    out = asyncio.run(commands.handle_command(
        "forget", "", "x", ctx={"clear_all_memory": fake_clear},
    ))
    assert_true("Nothing to forget" in out, "empty-conv message")


# ---------------------------------------------------------------------------
# handle_command — /why
# ---------------------------------------------------------------------------

def test_why_shows_memory_state():
    print("\n[test] /why summarizes facts + summary stack")
    _wipe()
    facts.save_facts("why-conv", [
        {"text": "Lyra is half-elf", "added_turn": 0, "last_used": 100},
    ])
    out = asyncio.run(commands.handle_command("why", "", "why-conv"))
    assert_true("Memory state" in out, "header present")
    assert_true("Lyra" in out, "fact shown")
    assert_true("Summary stack" in out, "summary line present")
    assert_true("Indexed exchanges" in out, "episodic line present")


def test_why_with_no_state_shows_none_markers():
    print("\n[test] /why on empty conv shows '(none)' markers")
    _wipe()
    out = asyncio.run(commands.handle_command("why", "", "fresh"))
    assert_true("(none)" in out, "none markers present")


# ---------------------------------------------------------------------------
# handle_command — unknown
# ---------------------------------------------------------------------------

def test_unknown_canonical_returns_hint():
    print("\n[test] handle_command with unknown canonical name → hint")
    out = asyncio.run(commands.handle_command("xyzzy", "", "any"))
    assert_true("Unknown command" in out, "unknown-command hint")
    assert_true("/help" in out, "points to /help")


def test_handler_exception_caught():
    print("\n[test] handle_command catches handler exceptions")

    # Inject a handler that raises by patching the dispatch table
    original = commands._HANDLERS.get("help")
    try:
        async def boom(arg, conv_id, ctx):
            raise RuntimeError("explosion")
        commands._HANDLERS["help"] = boom
        out = asyncio.run(commands.handle_command("help", "", "any"))
        assert_true("Command failed" in out, "failure message returned")
        assert_true("RuntimeError" in out, "exception type surfaced")
    finally:
        if original is not None:
            commands._HANDLERS["help"] = original


# ---------------------------------------------------------------------------
# build_synthetic_completion shape
# ---------------------------------------------------------------------------

def test_synthetic_completion_shape():
    print("\n[test] build_synthetic_completion has all OpenAI-shape fields")
    out = commands.build_synthetic_completion("hello", "magnum-12b")
    assert_eq(out["object"], "chat.completion", "object field")
    assert_eq(out["model"], "magnum-12b", "model field")
    assert_true("id" in out and out["id"].startswith("chatcmpl-cmd-"), "id prefix")
    assert_eq(len(out["choices"]), 1, "one choice")
    choice = out["choices"][0]
    assert_eq(choice["message"]["role"], "assistant", "assistant role")
    assert_eq(choice["message"]["content"], "hello", "content preserved")
    assert_eq(choice["finish_reason"], "stop", "finish_reason=stop")
    assert_eq(out["usage"]["total_tokens"], 0, "zero token usage")


def test_synthetic_completion_handles_empty_model():
    print("\n[test] build_synthetic_completion tolerates empty/None model")
    out = commands.build_synthetic_completion("x", "")
    assert_true(out["model"], "fallback model name applied")
    assert_eq(out["model"], "compactor-command", "explicit fallback")


def test_synthetic_completion_stream_shape():
    print("\n[test] build_synthetic_completion_stream returns 2 valid chunks")
    chunks = commands.build_synthetic_completion_stream("hi", "m")
    assert_eq(len(chunks), 2, "two SSE chunks (content + stop)")
    first, last = chunks
    assert_eq(first["object"], "chat.completion.chunk", "chunk object")
    assert_eq(first["choices"][0]["delta"]["content"], "hi", "content in first chunk")
    assert_eq(last["choices"][0]["finish_reason"], "stop", "stop in last chunk")
    # Each chunk must be JSON-serializable (it's what main.py emits via SSE)
    json.dumps(first)
    json.dumps(last)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_parse_empty_returns_none,
        test_parse_non_slash_returns_none,
        test_parse_unknown_slash_returns_none,
        test_parse_canonical_commands,
        test_parse_aliases_resolve_to_canonical,
        test_parse_is_case_insensitive,
        test_parse_preserves_arg_text,
        test_parse_handles_leading_whitespace,
        test_help_lists_all_commands,
        test_list_facts_empty_conv,
        test_list_facts_renders_all_entries,
        test_list_archive_empty,
        test_list_archive_renders_entries,
        test_remember_requires_arg,
        test_remember_persists_fact,
        test_remember_rejects_too_long,
        test_forget_with_substring_removes_only_matches,
        test_forget_substring_no_match,
        test_forget_no_arg_invokes_clear_all_helper,
        test_forget_no_arg_without_helper_returns_error,
        test_forget_no_arg_nothing_to_clear,
        test_why_shows_memory_state,
        test_why_with_no_state_shows_none_markers,
        test_unknown_canonical_returns_hint,
        test_handler_exception_caught,
        test_synthetic_completion_shape,
        test_synthetic_completion_handles_empty_model,
        test_synthetic_completion_stream_shape,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll commands smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
