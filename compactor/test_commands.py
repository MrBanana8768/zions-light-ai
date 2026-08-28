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
import portability  # noqa: E402
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
# /tidy — v3.1 D6
# ---------------------------------------------------------------------------
#
# Every fact string below is invented for this file. The store this feature is
# built for holds a real person's memory of her own life; nothing from it, and
# nothing shaped like it, belongs in a test fixture, a commit message or a log
# line. The "real fact" rows here are deliberately mundane fiction-workshop
# material — their job is to sit in the store and NOT be removed.
#
# The portability.quarantine_conversation tests live here rather than in
# test_portability.py because that file is owned elsewhere this cycle. They
# belong beside the export tests; move them when the ownership allows.

_REAL_FACTS = [
    "The protagonist is a lighthouse keeper named Idris.",
    "The story is set on a fictional island called Brannock.",
    "The user wants the prose written in past tense.",
    "The user does not want a romance subplot.",
    "Idris keeps a logbook bound in blue cloth.",
    "The second act should open with the storm.",
]

# Rows that are provably content-free: the extractor's own format vocabulary,
# and rows with no letter and no digit anywhere.
_GARBAGE_SCAFFOLD = ["NONE", "No new facts.", "EXISTING FACTS:", "assistant"]
_GARBAGE_EMPTY = ["- -", "...", "**", "━━━━━━━━"]


def _rows(texts, *, start_turn=1, last_used=1000):
    return [
        {"text": t, "added_turn": start_turn + i, "last_used": last_used + i}
        for i, t in enumerate(texts)
    ]


def _tidy(arg, cid):
    return asyncio.run(commands.handle_command("tidy", arg, cid, ctx={}))


def _remove_section(out):
    """Only the WOULD REMOVE block, so a test can assert a string is absent
    from the removal list without it accidentally matching the KEEPING list."""
    if "WOULD REMOVE" not in out:
        return ""
    return out.split("WOULD REMOVE", 1)[1].split("KEEPING", 1)[0]


def _code_from(out):
    marker = "/tidy apply "
    assert_true(marker in out, "dry run offered a confirmation code")
    return out.split(marker, 1)[1].split()[0]


def test_parse_tidy_and_aliases():
    print("\n[test] parse_command: /tidy and its aliases, and near-misses pass through")
    assert_eq(commands.parse_command("/tidy"), ("tidy", ""), "/tidy")
    assert_eq(commands.parse_command("/tidy apply abc123"),
              ("tidy", "apply abc123"), "/tidy apply <code>")
    assert_eq(commands.parse_command("/tidy-facts"), ("tidy", ""), "/tidy-facts alias")
    assert_eq(commands.parse_command("/cleanup"), ("tidy", ""), "/cleanup alias")
    # A near-miss must pass through to the model rather than resolve to a
    # command that can delete facts.
    assert_eq(commands.parse_command("/tidyup"), (None, ""), "/tidyup is not /tidy")
    assert_eq(commands.parse_command("/t"), (None, ""), "no single-letter alias")


def test_tidy_dry_run_changes_nothing():
    print("\n[test] /tidy is a dry run: the store is byte-identical afterwards")
    _wipe()
    cid = "tidy-dry"
    rows = _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD + _GARBAGE_EMPTY)
    facts.save_facts(cid, rows)
    before = facts.load_facts(cid)
    out = _tidy("", cid)
    assert_true("DRY RUN" in out, f"reply announces a dry run: {out[:80]!r}")
    assert_eq(facts.load_facts(cid), before, "facts unchanged by the dry run")
    assert_eq(facts.load_archive(cid), [], "nothing archived by the dry run")
    assert_eq(portability.list_quarantine(cid), [], "no snapshot written by a dry run")


def test_tidy_never_proposes_a_real_fact_for_removal():
    print("\n[test] /tidy: no real fact appears in the WOULD REMOVE section")
    _wipe()
    cid = "tidy-conservative"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD + _GARBAGE_EMPTY))
    section = _remove_section(_tidy("", cid))
    for text in _REAL_FACTS:
        assert_true(text not in section, f"real fact not proposed for removal: {text!r}")
    for text in _GARBAGE_SCAFFOLD + _GARBAGE_EMPTY:
        assert_true(repr(text) in section, f"garbage row shown verbatim: {text!r}")


def test_tidy_groups_removals_under_the_rule_that_matched():
    print("\n[test] /tidy groups each proposed removal under its rule + reason")
    # The requirement is that a BAD RULE is visible before it runs, which means
    # the operator has to be able to see which rule claimed which row.
    _wipe()
    cid = "tidy-groups"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD + _GARBAGE_EMPTY))
    out = _tidy("", cid)
    assert_true("[scaffolding]" in out, "scaffolding group present")
    assert_true("[no-content]" in out, "no-content group present")
    assert_true("extractor's own format vocabulary" in out, "rule explained in prose")
    assert_true("no letter and no digit" in out, "no-content rule explained in prose")


def test_tidy_keeps_and_reports_ambiguous_rows():
    print("\n[test] /tidy KEEPS ambiguous rows and reports them separately")
    # The whole safety premise: when a row is odd but not provably garbage it
    # stays, and the operator is told about it. Every row here is odd.
    _wipe()
    cid = "tidy-ambiguous"
    ambiguous = [
        "1997-04-12",                                     # numeric-only
        "[user]: Idris hates the fog",                    # transcript-fragment
        "I cannot determine any facts from this exchange.",  # meta-commentary
        "━━━━━━━━ Chapter Three ━━━━━━━━",                # decorative
        "Blue cloth.",                                    # very-short
        "x" * (commands.TIDY_OVERSIZED_CHARS + 20),       # oversized
    ]
    facts.save_facts(cid, _rows(_REAL_FACTS + ambiguous))
    out = _tidy("", cid)
    assert_true("WOULD REMOVE nothing" in out, f"nothing removed: {out}")
    assert_true("KEEPING" in out, "ambiguous rows reported")
    for rule in ("numeric-only", "transcript-fragment", "meta-commentary",
                 "decorative", "very-short", "oversized"):
        assert_true(f"[{rule}]" in out, f"{rule} reported as kept-but-odd")


def test_tidy_reports_near_duplicates_without_removing_them():
    print("\n[test] /tidy flags near-duplicates but will not pick a wording")
    _wipe()
    cid = "tidy-near"
    facts.save_facts(cid, _rows([
        "Idris keeps a logbook bound in blue cloth.",
        "Idris keeps a logbook bound in blue cloth",   # no full stop
        "The story is set on a fictional island called Brannock.",
    ]))
    out = _tidy("", cid)
    assert_true("WOULD REMOVE nothing" in out, f"no removal proposed: {out}")
    assert_true("[near-duplicate]" in out, "near-duplicate flagged for a human")


def test_tidy_removes_exact_duplicates_keeping_the_longest_lived_copy():
    print("\n[test] /tidy collapses byte-identical rows, keeping the survivor "
          "that would have outlived the others")
    _wipe()
    cid = "tidy-dupes"
    dup = "Idris keeps a logbook bound in blue cloth."
    facts.save_facts(cid, [
        {"text": dup, "added_turn": 3, "last_used": 100},
        {"text": dup, "added_turn": 9, "last_used": 400},   # survivor
        {"text": dup, "added_turn": 5, "last_used": 400},
        {"text": "The second act should open with the storm.",
         "added_turn": 2, "last_used": 50},
    ])
    out = _tidy("", cid)
    assert_true("[duplicate] 2 row(s)" in out, f"two duplicates proposed: {out}")
    reply = _tidy(f"apply {_code_from(out)}", cid)
    remaining = facts.load_facts(cid)
    assert_eq(len(remaining), 2, f"one copy left plus the other fact: {reply}")
    survivor = [f for f in remaining if f["text"] == dup][0]
    # Eviction sorts on (last_used, added_turn) — keeping the maximum means
    # this collapse can never move a fact FORWARD in the eviction queue.
    assert_eq((survivor["last_used"], survivor["added_turn"]), (400, 9),
              "survivor is the highest (last_used, added_turn) copy")


def test_tidy_apply_without_a_code_refuses():
    print("\n[test] /tidy apply with no code changes nothing")
    _wipe()
    cid = "tidy-nocode"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD))
    before = facts.load_facts(cid)
    out = _tidy("apply", cid)
    assert_true("needs the code" in out, f"refused with a reason: {out!r}")
    assert_eq(facts.load_facts(cid), before, "store untouched")


def test_tidy_apply_with_a_stale_code_refuses_and_reprints_the_plan():
    print("\n[test] /tidy apply refuses a code issued for a different removal set")
    _wipe()
    cid = "tidy-stale"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD))
    code = _code_from(_tidy("", cid))
    # A new GARBAGE row lands — the removal set has changed, so the code the
    # operator is holding no longer describes what would happen.
    facts.save_facts(cid, facts.load_facts(cid) + _rows(["NONE"], start_turn=99))
    before = facts.load_facts(cid)
    out = _tidy(f"apply {code}", cid)
    assert_true("out of date" in out, f"refused as stale: {out[:120]!r}")
    assert_true("DRY RUN" in out, "a fresh plan is printed instead")
    assert_eq(facts.load_facts(cid), before, "nothing removed")


def test_tidy_code_survives_an_unrelated_write():
    print("\n[test] a new GOOD fact does not invalidate the confirmation code")
    # The code is a compare-and-swap over the REMOVAL SET, not over the whole
    # store. A tail adding a real fact between the dry run and the confirmation
    # is the normal case on a live conversation; invalidating there would make
    # the command unusable without making it safer.
    _wipe()
    cid = "tidy-benign"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD))
    code = _code_from(_tidy("", cid))
    facts.save_facts(cid, facts.load_facts(cid) + _rows(
        ["Brannock has one harbour, on the eastern shore."], start_turn=99))
    out = _tidy(f"apply {code}", cid)
    assert_true(out.startswith("Removed "), f"applied: {out[:120]!r}")
    texts = [f["text"] for f in facts.load_facts(cid)]
    assert_true("Brannock has one harbour, on the eastern shore." in texts,
                "the fact added in between survived")
    for t in _GARBAGE_SCAFFOLD:
        assert_true(t not in texts, f"garbage removed: {t!r}")


def test_tidy_apply_archives_to_the_sidecar_and_is_restorable():
    print("\n[test] /tidy apply moves rows to the archive; they can be restored")
    _wipe()
    cid = "tidy-restore"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD))
    out = _tidy(f"apply {_code_from(_tidy('', cid))}", cid)
    archived = {f["text"] for f in facts.load_archive(cid)}
    assert_eq(archived, set(_GARBAGE_SCAFFOLD), "every removed row is in the archive")
    assert_true("/list-archive" in out, f"reply says where they went: {out!r}")
    # Nothing was unlinked: restore_from_archive puts a row back, unchanged.
    n = facts.restore_from_archive(cid, text_substring="EXISTING FACTS:")
    assert_eq(n, 1, "one row restored")
    assert_true("EXISTING FACTS:" in [f["text"] for f in facts.load_facts(cid)],
                "the restored row is back in the active set")


def test_tidy_apply_writes_a_verified_snapshot_that_import_can_read():
    print("\n[test] /tidy apply publishes a snapshot import_conversation accepts")
    _wipe()
    cid = "tidy-snap"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD))
    n_before = len(facts.load_facts(cid))
    out = _tidy(f"apply {_code_from(_tidy('', cid))}", cid)
    snaps = portability.list_quarantine(cid)
    assert_eq(len(snaps), 1, "exactly one snapshot published")
    assert_true(str(snaps[0]) in out, "the reply names the snapshot path")
    assert_eq(list(snaps[0].parent.glob("*.partial")), [], "no .partial left behind")
    bundle = json.loads(snaps[0].read_text(encoding="utf-8"))
    assert_eq(len(bundle["facts"]), n_before, "snapshot holds the pre-op store")
    # The recovery route is the existing import path, with no new format.
    res = portability.import_conversation(
        bundle, target_conv_id="tidy-snap-restored", overwrite=False)
    assert_eq(res["imported"]["facts"], n_before, "snapshot restores every row")


def test_tidy_apply_aborts_when_the_snapshot_cannot_be_written():
    print("\n[test] /tidy apply removes nothing if the pre-op snapshot fails")
    _wipe()
    cid = "tidy-snapfail"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD))
    code = _code_from(_tidy("", cid))
    before = facts.load_facts(cid)
    real = portability.quarantine_conversation

    def boom(conv_id, *, reason):
        raise portability.QuarantineError("simulated: no space left on device")

    portability.quarantine_conversation = boom
    try:
        out = _tidy(f"apply {code}", cid)
    finally:
        portability.quarantine_conversation = real
    assert_true("changed nothing" in out, f"abort is stated plainly: {out!r}")
    assert_eq(facts.load_facts(cid), before, "store untouched")
    assert_eq(facts.load_archive(cid), [], "nothing archived")


def test_tidy_refuses_a_plan_that_takes_more_than_half_the_store():
    print("\n[test] /tidy refuses a plan whose blast radius is over the cap")
    # A rule that matches half a conversation's memory is a broken rule. The
    # operator must see a refusal, not a large success message.
    _wipe()
    cid = "tidy-blast"
    garbage = _rows(["..."] * 15 + ["- -"] * 3, start_turn=1)
    for i, r in enumerate(garbage):
        r["text"] = f"{'.' * (3 + i)}"     # distinct, all content-free
    facts.save_facts(cid, garbage + _rows(_REAL_FACTS[:4], start_turn=100))
    dry = _tidy("", cid)
    assert_true("I will NOT apply this plan" in dry, f"dry run warns: {dry}")
    before = facts.load_facts(cid)
    # There is no code on offer, but a stale one from elsewhere must not work
    # either — the cap is checked on the apply path, not only in the renderer.
    plan = commands._tidy_plan(before)
    out = _tidy(f"apply {plan['token']}", cid)
    assert_true("Refusing" in out, f"apply refused: {out!r}")
    assert_eq(facts.load_facts(cid), before, "store untouched")


def test_tidy_is_idempotent():
    print("\n[test] /tidy apply twice removes the same rows once")
    _wipe()
    cid = "tidy-idem"
    # Enough real facts that eight garbage rows stay under the blast-radius
    # cap — the cap is exercised on its own in
    # test_tidy_refuses_a_plan_that_takes_more_than_half_the_store.
    more = [
        "Brannock has one harbour, on the eastern shore.",
        "The lamp room was rebuilt after the fire.",
        "Idris's sister Maren visits every spring.",
        "Chapter one should end on the wreck.",
    ]
    facts.save_facts(
        cid, _rows(_REAL_FACTS + more + _GARBAGE_SCAFFOLD + _GARBAGE_EMPTY))
    _tidy(f"apply {_code_from(_tidy('', cid))}", cid)
    after_first = facts.load_facts(cid)
    archive_first = facts.load_archive(cid)
    second = _tidy("", cid)
    assert_true("WOULD REMOVE nothing" in second, f"second pass is a no-op: {second}")
    out = _tidy("apply deadbeef1234", cid)
    assert_true("Nothing to remove" in out, f"apply is a no-op too: {out!r}")
    assert_eq(facts.load_facts(cid), after_first, "active set unchanged")
    assert_eq(facts.load_archive(cid), archive_first, "archive not re-appended")


def test_tidy_on_an_empty_store_says_so():
    print("\n[test] /tidy on a conversation with no facts")
    _wipe()
    assert_true("nothing to tidy" in _tidy("", "tidy-empty").lower(), "dry run")
    assert_true("nothing to tidy" in _tidy("apply abc", "tidy-empty").lower(), "apply")


def test_tidy_rejects_an_unknown_subcommand():
    print("\n[test] /tidy with an unrecognized option does not fall through to apply")
    _wipe()
    cid = "tidy-badopt"
    facts.save_facts(cid, _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD))
    before = facts.load_facts(cid)
    out = _tidy("delete-everything", cid)
    assert_true("Unknown option" in out, f"refused: {out!r}")
    assert_eq(facts.load_facts(cid), before, "store untouched")


def test_tidy_unreadable_store_is_not_rewritten():
    print("\n[test] /tidy on an unreadable facts file changes nothing and says so")
    _wipe()
    cid = "tidy-corrupt"
    memory.facts_path(cid).parent.mkdir(parents=True, exist_ok=True)
    memory.facts_path(cid).write_text("{not json", encoding="utf-8")
    out = _tidy("", cid)
    assert_true("couldn't read" in out, f"plain-language store error: {out!r}")
    assert_true("Nothing has been deleted" in out, "reassures nothing was lost")
    assert_eq(memory.facts_path(cid).read_text(encoding="utf-8"), "{not json",
              "the unreadable file is left exactly as it was")


def test_every_removal_rule_has_a_printed_explanation():
    print("\n[test] every rule the classifier can emit has operator-facing prose")
    # A rule with no explanation is a rule the operator cannot review, and an
    # unreviewable rule is how a bad one gets confirmed.
    emitted = set(commands._TIDY_REMOVE_ORDER) | set(commands._TIDY_FLAG_ORDER)
    missing = sorted(emitted - set(commands._TIDY_RULE_TEXT))
    assert_eq(missing, [], "no rule id is missing from _TIDY_RULE_TEXT")


def test_quarantine_refuses_to_publish_a_snapshot_short_of_the_store():
    print("\n[test] quarantine_conversation contradicts its own export")
    # export_conversation wraps every layer in `except Exception` and degrades
    # to an empty value, so a conversation whose facts file raised produces a
    # valid, empty, importable bundle. Publishing that as "your data is safe"
    # is backup.py F2 exactly. The verify step exists to catch it.
    _wipe()
    cid = "tidy-shortsnap"
    facts.save_facts(cid, _rows(_REAL_FACTS))
    real = portability.export_conversation

    def lossy(conv_id):
        b = real(conv_id)
        b["facts"] = []
        return b

    portability.export_conversation = lossy
    try:
        raised = False
        try:
            portability.quarantine_conversation(cid, reason="test")
        except portability.QuarantineError:
            raised = True
    finally:
        portability.export_conversation = real
    assert_true(raised, "a snapshot short of the store is refused")
    assert_eq(portability.list_quarantine(cid), [], "nothing published")
    assert_eq(list(portability.quarantine_dir().glob("*.partial")), [],
              "no .partial left behind on the failure path")


def test_quarantine_propagates_an_unreadable_facts_file():
    print("\n[test] quarantine_conversation raises on an unreadable facts file")
    # It must not fall back to "0 facts expected" and then congratulate itself
    # for capturing 0 — the operation that follows is about to rewrite that
    # very file.
    _wipe()
    cid = "tidy-quarantine-corrupt"
    memory.facts_path(cid).parent.mkdir(parents=True, exist_ok=True)
    memory.facts_path(cid).write_text("{not json", encoding="utf-8")
    raised = False
    try:
        portability.quarantine_conversation(cid, reason="test")
    except memory.StoreUnreadable:
        raised = True
    assert_true(raised, "StoreUnreadable propagates to the caller")
    assert_eq(portability.list_quarantine(cid), [], "nothing published")


def test_tidy_finishes_an_interrupted_apply():
    print("\n[test] /tidy resumes cleanly from a crash between the two writes")
    # The apply writes the sidecar first and the active set second, so an
    # interruption in between leaves a row in BOTH files. That is the
    # recoverable order (archive_facts's docstring: the other one loses it
    # outright), and re-running has to finish the job without doubling the
    # archive.
    _wipe()
    cid = "tidy-interrupted"
    rows = _rows(_REAL_FACTS + _GARBAGE_SCAFFOLD)
    facts.save_facts(cid, rows)
    facts.archive_facts(cid, [f for f in rows if f["text"] in _GARBAGE_SCAFFOLD])
    assert_eq(len(facts.load_facts(cid)), len(rows), "active set still complete")
    assert_eq(len(facts.load_archive(cid)), len(_GARBAGE_SCAFFOLD), "sidecar written")
    _tidy(f"apply {_code_from(_tidy('', cid))}", cid)
    assert_eq(len(facts.load_facts(cid)), len(_REAL_FACTS), "active set finished")
    assert_eq(len(facts.load_archive(cid)), len(_GARBAGE_SCAFFOLD),
              "archive holds one copy of each row, not two")


def test_tidy_never_elides_the_removal_list():
    print("\n[test] /tidy shows every proposed removal, however many there are")
    # A row confirmed without being seen is the failure this command exists to
    # prevent, so the removal section is never truncated. The flagged list is,
    # because it is a reading list and not a consent surface.
    _wipe()
    cid = "tidy-longlist"
    n = commands.TIDY_MAX_ROWS_SHOWN + 7
    garbage = [{"text": "." * (3 + i), "added_turn": i, "last_used": i}
               for i in range(n)]
    real = _rows([f"Brannock landmark number {i} is on the map." for i in range(n * 3)],
                 start_turn=500)
    facts.save_facts(cid, garbage + real)
    out = _tidy("", cid)
    section = _remove_section(out)
    assert_true("not shown" not in section, "removal list is complete")
    for g in garbage:
        assert_true(repr(g["text"]) in section, "every removal row is printed")
    flagged = out.split("KEEPING", 1)[1] if "KEEPING" in out else ""
    assert_true("not shown" in flagged or not flagged,
                "the flagged reading list is the part that may be capped")


def test_quarantine_does_not_overwrite_a_snapshot_from_the_same_second():
    print("\n[test] two snapshots in one second get two names")
    # The stamp has second resolution. A collision would mean the second run
    # silently overwriting the snapshot that makes the first run reversible.
    _wipe()
    cid = "tidy-collide"
    facts.save_facts(cid, _rows(_REAL_FACTS))
    a = portability.quarantine_conversation(cid, reason="test")
    b = portability.quarantine_conversation(cid, reason="test")
    assert_true(a["path"] != b["path"], "the second snapshot got its own name")
    assert_eq(len(portability.list_quarantine(cid)), 2, "both are on disk")


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
        # v3.1 D6 — /tidy
        test_parse_tidy_and_aliases,
        test_tidy_dry_run_changes_nothing,
        test_tidy_never_proposes_a_real_fact_for_removal,
        test_tidy_groups_removals_under_the_rule_that_matched,
        test_tidy_keeps_and_reports_ambiguous_rows,
        test_tidy_reports_near_duplicates_without_removing_them,
        test_tidy_removes_exact_duplicates_keeping_the_longest_lived_copy,
        test_tidy_apply_without_a_code_refuses,
        test_tidy_apply_with_a_stale_code_refuses_and_reprints_the_plan,
        test_tidy_code_survives_an_unrelated_write,
        test_tidy_apply_archives_to_the_sidecar_and_is_restorable,
        test_tidy_apply_writes_a_verified_snapshot_that_import_can_read,
        test_tidy_apply_aborts_when_the_snapshot_cannot_be_written,
        test_tidy_refuses_a_plan_that_takes_more_than_half_the_store,
        test_tidy_is_idempotent,
        test_tidy_on_an_empty_store_says_so,
        test_tidy_rejects_an_unknown_subcommand,
        test_tidy_unreadable_store_is_not_rewritten,
        test_every_removal_rule_has_a_printed_explanation,
        test_quarantine_refuses_to_publish_a_snapshot_short_of_the_store,
        test_quarantine_propagates_an_unreadable_facts_file,
        test_tidy_finishes_an_interrupted_apply,
        test_tidy_never_elides_the_removal_list,
        test_quarantine_does_not_overwrite_a_snapshot_from_the_same_second,
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
