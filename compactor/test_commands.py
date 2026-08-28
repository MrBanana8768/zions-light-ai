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

import backfill  # noqa: E402
import bgwork  # noqa: E402
import commands  # noqa: E402
import dedup  # noqa: E402
import facts  # noqa: E402
import memory  # noqa: E402
import summarizer  # noqa: E402


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
# handle_command — /forget, v3.1 A3: the reply must describe what is gone,
# not what the wipe intended
# ---------------------------------------------------------------------------

def _null_clear(**over):
    """A clear_all_memory helper that reports a clean, empty wipe. Used to
    isolate the layers /forget is responsible for on its own — anything the
    reply says under this helper came from the disk, not from the counters."""
    async def fake_clear(cid):
        base = {
            "conv_id": cid,
            "forgotten_facts": 0,
            "forgotten_episodic": 0,
            "forgotten_summary": False,
            "forgotten_persona": False,
            "unreadable": [],
        }
        base.update(over)
        return base
    return fake_clear


def test_forget_clears_the_archive_sidecar():
    print("\n[test] /forget (no arg) clears the archive sidecar and says so")
    # A3. prune_facts and archive_stale_facts MOVE evicted facts into this
    # sidecar rather than deleting them, and restore_from_archive puts them
    # back into the injected set. Before this fix nothing in the codebase
    # deleted an archived fact, ever — /forget included.
    _wipe()
    cid = "arch-wipe"
    facts.save_archive(cid, [
        {"text": "Her mother's name is Selene.", "added_turn": 1,
         "last_used": 1, "archived_at": 1},
        {"text": "The inn burned down in winter.", "added_turn": 2,
         "last_used": 2, "archived_at": 2},
    ])
    out = asyncio.run(commands.handle_command(
        "forget", "", cid, ctx={"clear_all_memory": _null_clear()},
    ))
    assert_eq(facts.load_archive(cid), [], "archive is empty on disk")
    assert_true("2 archived fact(s)" in out, f"archive counted in reply: {out!r}")


def test_forget_on_archive_only_conv_does_not_claim_it_was_empty():
    print("\n[test] /forget on an archive-only conv must not say 'no stored memory'")
    # The exact measured sentence this fixes: the conversation's remaining
    # memory was archived, /forget answered "Nothing to forget — this
    # conversation had no stored memory", and /list-archive listed it on the
    # next line.
    _wipe()
    cid = "arch-only"
    facts.save_archive(cid, [{"text": "Her mother's name is Selene.",
                              "added_turn": 1, "last_used": 1, "archived_at": 1}])
    out = asyncio.run(commands.handle_command(
        "forget", "", cid, ctx={"clear_all_memory": _null_clear()},
    ))
    assert_true("no stored memory" not in out.lower(),
                f"does not claim the conversation was empty: {out!r}")
    listing = asyncio.run(commands.handle_command("list-archive", "", cid, {}))
    assert_true("Selene" not in listing,
                f"/list-archive no longer shows the forgotten fact: {listing!r}")


def test_forget_reports_persona_it_deleted():
    print("\n[test] /forget names the persona it just deleted")
    # Under-reporting a deletion is the same defect as over-reporting one:
    # _clear_all_memory has always returned forgotten_persona and this handler
    # dropped it, so a persona-only conversation had its persona deleted and
    # was told it "had no stored memory".
    _wipe()
    out = asyncio.run(commands.handle_command(
        "forget", "", "persona-only",
        ctx={"clear_all_memory": _null_clear(forgotten_persona=True)},
    ))
    assert_true("persona" in out.lower(), f"persona named in reply: {out!r}")
    assert_true("no stored memory" not in out.lower(),
                f"does not claim the conversation was empty: {out!r}")


def test_forget_reports_memory_that_survived_the_wipe():
    print("\n[test] /forget reports a layer still on disk instead of a clean sweep")
    # This is the shape a racing background tail leaves behind, without needing
    # concurrency to produce it: the wipe helper reports success and the layer
    # is on disk when the user reads the answer. The reply is built from the
    # second read, not the first.
    #
    # The summary layer, not facts, because /forget's last act is to write an
    # empty facts store (the backfill tombstone) — a fact that survives the
    # wipe is deleted by that write rather than reported, which is the better
    # outcome and is asserted separately. Nothing writes over the summary
    # file, so it is the layer that can still reach the "still storing"
    # sentence, and this test exists to prove that sentence can still fire.
    _wipe()
    cid = "survivor"
    summarizer.save_state(cid, {
        "l1": [{"text": "Scene one.", "first_turn": 1, "last_turn": 20}],
        "l2": [], "l3": None, "last_summarized_turn": 20,
    })
    out = asyncio.run(commands.handle_command(
        "forget", "", cid,
        ctx={"clear_all_memory": _null_clear(forgotten_summary=True)},
    ))
    assert_true("still storing" in out.lower(),
                f"reply admits memory survived: {out!r}")
    assert_true("summary state" in out, f"names what survived: {out!r}")


def test_forget_deletes_a_fact_that_landed_behind_the_wipe():
    print("\n[test] /forget deletes a fact written behind it, rather than reporting it")
    # The measured race, reduced to its residue: the wipe helper reports a
    # clean sweep and a fact is on disk afterwards — what a parked tail's
    # extraction leaves. /forget's final write is an empty facts store, so the
    # fact is gone and the reply is both clean and true. Before this, the
    # reply was "Forgot: 2 fact(s)." with the tail's fact sitting in the file.
    _wipe()
    cid = "late-write"
    facts.save_facts(cid, [
        {"text": "fact extracted by the parked tail", "added_turn": 5,
         "last_used": 5},
    ])
    out = asyncio.run(commands.handle_command(
        "forget", "", cid,
        ctx={"clear_all_memory": _null_clear(forgotten_facts=2)},
    ))
    assert_eq(facts.load_facts(cid), [], "the late fact is off disk")
    assert_true("still storing" not in out.lower(),
                f"nothing left to warn about: {out!r}")


def test_forget_leaves_an_empty_facts_store_so_backfill_cannot_resurrect():
    print("\n[test] /forget closes the lazy-backfill gate on the wiped history")
    # Measured before the fix: a conversation whose memory was a summary and a
    # persona had no facts file, so /forget replied "Forgot: summary state,
    # persona." and backfill.needs_backfill was still True — the next request
    # would start a background extraction over the whole message history the
    # user had just asked to be forgotten, and write the result to disk.
    _wipe()
    cid = "bf-gate"
    msgs = [
        {"role": "user", "content": "My protagonist Lyra is a half-elf ranger."},
        {"role": "assistant", "content": "Where is she from?"},
        {"role": "user", "content": "Aethermere. Her mother Selene died there."},
        {"role": "assistant", "content": "A strong hook."},
    ]
    summarizer.save_state(cid, {
        "l1": [{"text": "Lyra, Aethermere, Selene.", "first_turn": 1, "last_turn": 4}],
        "l2": [], "l3": None, "last_summarized_turn": 4,
    })
    assert_true(backfill.needs_backfill(cid, msgs),
                "precondition: the history was backfillable before the wipe")
    out = asyncio.run(commands.handle_command(
        "forget", "", cid,
        ctx={"clear_all_memory": _null_clear(forgotten_summary=True)},
    ))
    assert_true(memory.facts_path(cid).is_file(),
                f"an empty facts store is on disk: {out!r}")
    assert_eq(facts.load_facts(cid), [], "and it is empty")
    assert_true(not backfill.needs_backfill(cid, msgs),
                "the backfill can no longer reconstruct the forgotten history")


def test_forget_leaves_an_unreadable_facts_file_untouched():
    print("\n[test] the backfill tombstone never overwrites an unreadable facts file")
    # F1's rule outranks the tombstone: a file whose contents are unknown is
    # not rewritten from a guess, even to write an empty store over it. The
    # cost is that the backfill gate stays open for that conversation, which
    # is why the layer is named in the reply.
    _wipe()
    cid = "tombstone-corrupt"
    p = memory.facts_path(cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid js", encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    out = asyncio.run(commands.handle_command(
        "forget", "", cid, ctx={"clear_all_memory": _null_clear(unreadable=["facts"])},
    ))
    assert_eq(p.read_text(encoding="utf-8"), before,
              "corrupt facts file left byte-identical")
    assert_true("could not read" in out.lower() and "facts" in out,
                f"names the layer it left alone: {out!r}")


def test_forget_clears_the_dedup_refusal_memo():
    print("\n[test] /forget drops this conversation's dedup refusal memo")
    # Per-conversation state derived from the conversation's facts. Nothing can
    # come back out of it — hashes and one-word reasons — so it is cleared
    # silently and not counted in the reply; it is cleared at all because a
    # wipe is exactly the "replaces a conversation wholesale" caller
    # reset_refusal_memo's docstring is written for.
    _wipe()
    dedup.reset_refusal_memo()
    dedup._memo_put("memo-mine", "clusterkey", "keep")
    dedup._memo_put("memo-other", "clusterkey", "keep")
    asyncio.run(commands.handle_command(
        "forget", "", "memo-mine", ctx={"clear_all_memory": _null_clear()},
    ))
    assert_eq(dedup._memo_get("memo-mine", "clusterkey"), None,
              "the wiped conversation's memo is gone")
    assert_eq(dedup._memo_get("memo-other", "clusterkey"), "keep",
              "another conversation's memo is untouched")


def test_forget_retries_once_when_the_first_pass_leaves_something():
    print("\n[test] /forget wipes a second time rather than telling the user to")
    # "Please run /forget again" is advice the command can take itself. The
    # shape it addresses is a tail landing between the wipe and the
    # verification, and a second pass clears it. Bounded at one: the summary
    # here survives pass 1 and is cleared by pass 2, and the reply comes back
    # clean with the counters of both passes summed.
    _wipe()
    cid = "retry-once"
    summarizer.save_state(cid, {
        "l1": [{"text": "Scene one.", "first_turn": 1, "last_turn": 20}],
        "l2": [], "l3": None, "last_summarized_turn": 20,
    })
    calls = []

    async def clear_second_time_only(c):
        calls.append(c)
        if len(calls) >= 2:
            summarizer.summary_path(c).unlink(missing_ok=True)
        return {"conv_id": c, "forgotten_facts": 1, "forgotten_episodic": 0,
                "forgotten_summary": True, "forgotten_persona": False,
                "unreadable": []}

    out = asyncio.run(commands.handle_command(
        "forget", "", cid, ctx={"clear_all_memory": clear_second_time_only},
    ))
    assert_eq(len(calls), 2, "the wipe ran twice")
    assert_true("still storing" not in out.lower(),
                f"the retry cleared it, so no warning: {out!r}")
    assert_true("2 fact(s)" in out,
                f"both passes' counters are summed, not the first pass's: {out!r}")


def test_forget_does_not_retry_when_the_first_pass_was_clean():
    print("\n[test] the ordinary /forget still wipes exactly once")
    # The retry is for the residue case only. An extra full wipe on every
    # /forget would double the log lines and the disk writes for a command
    # that almost always succeeds on the first pass.
    _wipe()
    calls = []

    async def counting_clear(c):
        calls.append(c)
        return {"conv_id": c, "forgotten_facts": 2, "forgotten_episodic": 0,
                "forgotten_summary": False, "forgotten_persona": False,
                "unreadable": []}

    asyncio.run(commands.handle_command(
        "forget", "", "clean-pass", ctx={"clear_all_memory": counting_clear},
    ))
    assert_eq(len(calls), 1, "one pass, no retry")


def test_forget_drains_background_work_before_wiping():
    print("\n[test] /forget drains in-flight tails BEFORE it deletes, and again after")
    # The first drain is the fix for the race: draining only after the wipe
    # would let a parked tail — one that has not executed a line, holding a
    # facts snapshot older than the command — write its extraction on top of
    # it. The second drain is what makes the verification meaningful: a tail
    # submitted while the wipe was running (it holds conv_lock across an
    # extraction and a rollup) is not in the first drain's set, and reading the
    # disk while that is still in flight measures a moving state.
    _wipe()
    order = []
    real_drain = bgwork.pool.drain

    async def spy_drain(timeout=10.0):
        order.append("drain")
        return await real_drain(timeout=timeout)

    async def fake_clear(cid):
        order.append("clear")
        return {"conv_id": cid, "forgotten_facts": 1, "forgotten_episodic": 0,
                "forgotten_summary": False, "forgotten_persona": False,
                "unreadable": []}

    bgwork.pool.drain = spy_drain
    try:
        asyncio.run(commands.handle_command(
            "forget", "", "drain-order", ctx={"clear_all_memory": fake_clear},
        ))
    finally:
        bgwork.pool.drain = real_drain
    assert_eq(order, ["drain", "clear", "drain"],
              "drained, cleared, drained again before verifying")


def test_forget_says_so_when_background_work_did_not_settle():
    print("\n[test] /forget admits it when the drain did not finish")
    # Honest partial success. The wipe happened; the guarantee that nothing
    # re-adds behind it did not, and the user is the one who will see the
    # difference.
    _wipe()
    real_stats = bgwork.pool.stats
    bgwork.pool.stats = lambda: {"outstanding": 2}
    try:
        out = asyncio.run(commands.handle_command(
            "forget", "", "unsettled",
            ctx={"clear_all_memory": _null_clear(forgotten_facts=2)},
        ))
    finally:
        bgwork.pool.stats = real_stats
    assert_true("2 fact(s)" in out, f"still reports the real wipe: {out!r}")
    assert_true("may reappear" in out.lower(),
                f"reply declines to promise a clean sweep: {out!r}")


def test_forget_reports_an_unreadable_archive_without_rewriting_it():
    print("\n[test] /forget with a corrupt archive names it and leaves it alone")
    # Same rule the facts layer follows (F1): a file whose contents are unknown
    # is never rewritten from a guess, and the user is told which layer that
    # was rather than given a clean-sweep sentence.
    _wipe()
    cid = "arch-corrupt"
    p = memory.facts_archive_path(cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid js", encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    out = asyncio.run(commands.handle_command(
        "forget", "", cid, ctx={"clear_all_memory": _null_clear()},
    ))
    assert_true("archived facts" in out, f"names the archive layer: {out!r}")
    assert_true("no stored memory" not in out.lower(),
                f"does not claim the conversation was empty: {out!r}")
    assert_eq(p.read_text(encoding="utf-8"), before,
              "corrupt archive left byte-identical")


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
        # v3.1 A3. These were written, left out of this list, and therefore
        # never ran — the exact failure mode the item is about, one level up:
        # a fix that reports itself as tested. Registered, and the list is now
        # asserted to hold every test_* in the module (see
        # test_every_test_in_this_module_is_registered) so the next one cannot
        # go missing quietly.
        test_forget_clears_the_archive_sidecar,
        test_forget_on_archive_only_conv_does_not_claim_it_was_empty,
        test_forget_reports_persona_it_deleted,
        test_forget_reports_memory_that_survived_the_wipe,
        test_forget_deletes_a_fact_that_landed_behind_the_wipe,
        test_forget_leaves_an_empty_facts_store_so_backfill_cannot_resurrect,
        test_forget_leaves_an_unreadable_facts_file_untouched,
        test_forget_clears_the_dedup_refusal_memo,
        test_forget_retries_once_when_the_first_pass_leaves_something,
        test_forget_does_not_retry_when_the_first_pass_was_clean,
        test_forget_drains_background_work_before_wiping,
        test_forget_says_so_when_background_work_did_not_settle,
        test_forget_reports_an_unreadable_archive_without_rewriting_it,
        test_why_shows_memory_state,
        test_why_with_no_state_shows_none_markers,
        test_unknown_canonical_returns_hint,
        test_handler_exception_caught,
        test_synthetic_completion_shape,
        test_synthetic_completion_handles_empty_model,
        test_synthetic_completion_stream_shape,
        test_every_test_in_this_module_is_registered,
    ]


def test_every_test_in_this_module_is_registered():
    print("\n[test] every test_* in this file is in the runner list")
    # Seven A3 tests were added to this file and never added to _all_tests().
    # They were dead code: the suite reported PASS and none of them had run.
    # A missing registration is invisible in exactly the way a missing test is
    # not, so the list checks itself.
    defined = {
        name for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    }
    registered = {t.__name__ for t in _all_tests()}
    missing = sorted(defined - registered)
    assert_eq(missing, [], "no test defined here is left out of the runner")


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll commands smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
