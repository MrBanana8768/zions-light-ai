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
import persona  # noqa: E402
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
# /retire — v3.1 D8
# ---------------------------------------------------------------------------
#
# Same privacy rule as the /tidy block above, and it matters more here because
# this operation names a conversation id in its own syntax. Every id in this
# file is invented. Nothing from the live store, and no id from it, appears
# anywhere in this repository.
#
# The fixtures reuse _REAL_FACTS / _GARBAGE_SCAFFOLD / _GARBAGE_EMPTY on
# purpose: if a row is safe from /tidy it must be safe from /retire, and the
# shared fixture is what makes a divergence show up as a failure rather than as
# two files quietly disagreeing about what a fact is.

# Markup that facts.is_storable_fact refuses and D6's rules do not — the half
# of the classifier that /tidy does not have.
_GARBAGE_MARKUP = [
    "```json",
    "## Current Status",
    '"energy_level": 88,',
    "ENERGY LEVEL: 88% -> 92%",
    "The following data was recorded: {",
]


def _retire(arg, cid):
    return asyncio.run(commands.handle_command("retire", arg, cid, ctx={}))


def _retire_code(out, source):
    marker = f"/retire {source} apply "
    assert_true(marker in out, f"dry run offered a confirmation code: {out[-400:]}")
    return out.split(marker, 1)[1].split()[0]


def _retire_apply(source, dest):
    return _retire(
        f"{source} apply {_retire_code(_retire(source, dest), source)}", dest)


def _drop_section(out):
    """Only the WOULD NOT MOVE block, so a test can assert a string is absent
    from the dropped list without it matching the moved list."""
    if "WOULD NOT MOVE" not in out:
        return ""
    return out.split("WOULD NOT MOVE", 1)[1].split("MOVING ANYWAY", 1)[0]


def test_parse_retire_and_aliases():
    print("\n[test] parse_command: /retire, its aliases, and near-misses")
    assert_eq(commands.parse_command("/retire abc"), ("retire", "abc"), "/retire <id>")
    assert_eq(commands.parse_command("/retire abc apply c0ffee"),
              ("retire", "abc apply c0ffee"), "/retire <id> apply <code>")
    assert_eq(commands.parse_command("/retire-conversation x"),
              ("retire", "x"), "long alias")
    # A near-miss must reach the model, not a command that empties a store.
    assert_eq(commands.parse_command("/retired"), (None, ""), "/retired is not /retire")
    assert_eq(commands.parse_command("/r"), (None, ""), "no single-letter alias")


def test_retire_without_a_source_explains_itself():
    print("\n[test] /retire with no argument prints usage and changes nothing")
    _wipe()
    out = _retire("", "dest-1")
    assert_true("Usage:" in out, f"usage printed: {out[:80]!r}")
    assert_true("the one that gets emptied" in out, "says which side is emptied")


def test_retire_refuses_its_own_conversation():
    print("\n[test] /retire cannot name the conversation it was typed in")
    _wipe()
    facts.save_facts("self-1", _rows(_REAL_FACTS))
    before = facts.load_facts("self-1")
    out = _retire("self-1", "self-1")
    assert_true("cannot be its own destination" in out, f"refused: {out!r}")
    assert_true("/forget" in out, "points at the command that does do this")
    assert_eq(facts.load_facts("self-1"), before, "store untouched")


def test_retire_rejects_anything_that_is_not_a_conversation_id():
    print("\n[test] /retire refuses a source id outside the filesystem charset")
    # memory._sanitize STRIPS bad characters on the request path, where the
    # cost of a mangled id is the wrong bucket. Here it would be the wrong
    # conversation emptied, so this refuses instead.
    _wipe()
    for bad in ("../../etc/passwd", "a/b", "x" * 65, "sem;colon", "dot.dot",
                "..", "%2e%2e"):
        out = _retire(bad, "dest-1")
        assert_true("is not a conversation id" in out, f"refused {bad!r}: {out[:60]!r}")
    # An id with whitespace in it cannot reach the charset check at all — the
    # arg is split on whitespace first, so the second word lands in the option
    # slot. It still refuses, and still changes nothing, which is the property
    # that matters; asserting the exact wording here would be asserting the
    # tokenizer rather than the guard.
    out = _retire("with space", "dest-1")
    assert_true("Unknown option" in out and "changed nothing" in out,
                f"a two-word id refuses too: {out[:80]!r}")


def test_retire_on_a_source_with_no_memory_says_so():
    print("\n[test] /retire on a conversation that has nothing")
    _wipe()
    out = _retire("ghost-1", "dest-1")
    assert_true("no stored memory of any kind" in out, f"says so: {out!r}")
    assert_eq(portability.list_quarantine("ghost-1"), [], "no snapshot written")


def test_retire_dry_run_changes_nothing():
    print("\n[test] /retire is a dry run: both stores are byte-identical after")
    _wipe()
    src, dst = "phantom-1", "dest-1"
    facts.save_facts(src, _rows(_REAL_FACTS + _GARBAGE_MARKUP + _GARBAGE_SCAFFOLD))
    facts.archive_facts(src, _rows(["Idris was born in the village of Cairn."]))
    facts.save_facts(dst, _rows(["The user prefers short chapters."], start_turn=50))
    persona.save_persona(src, "You are a patient writing companion.")
    summarizer.save_state(src, {"l1": [{"text": "a chunk", "first_turn": 1,
                                       "last_turn": 20}], "l2": [], "l3": None})
    before_src, before_arch = facts.load_facts(src), facts.load_archive(src)
    before_dst = facts.load_facts(dst)

    out = _retire(src, dst)
    assert_true("DRY RUN" in out, f"announces a dry run: {out[:80]!r}")
    assert_eq(facts.load_facts(src), before_src, "source facts unchanged")
    assert_eq(facts.load_archive(src), before_arch, "source archive unchanged")
    assert_eq(facts.load_facts(dst), before_dst, "destination unchanged")
    assert_true(persona.load_persona(src) is not None, "persona still there")
    assert_true(bool(summarizer.load_state(src).get("l1")), "summary still there")
    assert_eq(portability.list_quarantine(src), [], "no snapshot written by a dry run")
    # Requirement 7: the plan has to name what else is keyed to that id.
    for layer in ("summary state", "indexed exchanges", "persona",
                  "lazy-backfill state"):
        assert_true(f"  {layer}" in out, f"the plan reports the {layer} layer")


def test_retire_moves_real_facts_and_leaves_markup_behind():
    print("\n[test] /retire migrates genuine facts and drops only provable markup")
    _wipe()
    src, dst = "phantom-2", "dest-2"
    facts.save_facts(src, _rows(_REAL_FACTS + _GARBAGE_MARKUP
                                + _GARBAGE_SCAFFOLD + _GARBAGE_EMPTY))
    facts.save_facts(dst, _rows(["The user prefers short chapters."], start_turn=50))
    dry = _retire(src, dst)
    dropped = _drop_section(dry)
    for text in _REAL_FACTS:
        assert_true(text not in dropped, f"real fact never dropped: {text!r}")
    for text in _GARBAGE_MARKUP:
        assert_true(repr(text) in dropped, f"markup shown verbatim as dropped: {text!r}")
    assert_true("[markup]" in dry, "the is_storable_fact rule is named")
    assert_true("facts.is_storable_fact" in dry, "the plan says which predicate ran")

    out = _retire(f"{src} apply {_retire_code(dry, src)}", dst)
    assert_true(out.startswith("Retired "), f"applied: {out[:160]!r}")
    landed = [f["text"] for f in facts.load_facts(dst)]
    for text in _REAL_FACTS:
        assert_true(text in landed, f"migrated: {text!r}")
    for text in _GARBAGE_MARKUP + _GARBAGE_SCAFFOLD + _GARBAGE_EMPTY:
        assert_true(text not in landed, f"not migrated: {text!r}")
    assert_true("The user prefers short chapters." in landed,
                "the destination's own fact is still there")
    assert_eq(facts.load_facts(src), [], "source facts emptied")
    assert_true(memory.facts_path(src).is_file(),
                "an EMPTY facts file is left behind, not an unlinked one")


def test_retire_keeps_every_row_it_cannot_prove_is_garbage():
    print("\n[test] /retire migrates ambiguous rows rather than dropping them")
    # Requirement 6, and the whole premise: wrongly keeping garbage costs
    # tokens, wrongly dropping a row costs her a memory of her own life.
    _wipe()
    src, dst = "phantom-3", "dest-3"
    ambiguous = [
        "1997-04-12",
        "[user]: Idris hates the fog",
        "I cannot determine any facts from this exchange.",
        "Blue cloth.",
        "x" * (commands.TIDY_OVERSIZED_CHARS + 20),
    ]
    facts.save_facts(src, _rows(ambiguous))
    dry = _retire(src, dst)
    assert_true("WOULD NOT MOVE nothing" in dry, f"nothing dropped: {dry}")
    assert_true("MOVING ANYWAY" in dry, "they are surfaced as a reading list")
    _retire(f"{src} apply {_retire_code(dry, src)}", dst)
    landed = [f["text"] for f in facts.load_facts(dst)]
    for text in ambiguous:
        assert_true(text in landed, f"ambiguous row kept: {text[:40]!r}")


def test_retire_drops_only_byte_identical_destination_duplicates():
    print("\n[test] /retire drops an exact duplicate of a destination fact and "
          "KEEPS a near one")
    _wipe()
    src, dst = "phantom-4", "dest-4"
    exact = "Idris keeps a logbook bound in blue cloth."
    near = "idris keeps a logbook bound in blue cloth"   # case + full stop only
    facts.save_facts(src, _rows([exact, near, "The second act opens with a storm."]))
    facts.save_facts(dst, _rows([exact], start_turn=50))
    dry = _retire(src, dst)
    assert_true("[already-in-destination]" in dry, f"exact duplicate named: {dry}")
    assert_true("[near-duplicate-in-destination]" in dry, "near one flagged, not dropped")
    assert_true(repr(near) not in _drop_section(dry), "the near one is not in the drop list")
    _retire(f"{src} apply {_retire_code(dry, src)}", dst)
    landed = [f["text"] for f in facts.load_facts(dst)]
    assert_eq(landed.count(exact), 1, "the exact duplicate did not double")
    assert_true(near in landed, "the near-duplicate was MOVED, not discarded")


def test_retire_drops_a_fact_the_destination_already_has_in_cold_storage():
    print("\n[test] /retire treats the destination's archive as 'already present'")
    _wipe()
    src, dst = "phantom-5", "dest-5"
    text = "The story is set on a fictional island called Brannock."
    facts.save_facts(src, _rows([text]))
    facts.archive_facts(dst, _rows([text], start_turn=7))
    dry = _retire(src, dst)
    assert_true("[already-in-destination-archive]" in dry, f"named: {dry}")
    assert_true("/list-archive" in dry, "the reply says where the copy is")
    _retire(f"{src} apply {_retire_code(dry, src)}", dst)
    assert_eq([f["text"] for f in facts.load_facts(dst)], [],
              "not resurrected into the active set")
    assert_eq([f["text"] for f in facts.load_archive(dst)], [text],
              "the destination's cold copy is untouched and unduplicated")


def test_retire_moves_archived_rows_into_the_destination_archive():
    print("\n[test] /retire keeps cold facts cold — the sidecar is memory too")
    _wipe()
    src, dst = "phantom-6", "dest-6"
    cold = "Idris's father kept the light before him."
    hot = "The protagonist is a lighthouse keeper named Idris."
    facts.save_facts(src, _rows([hot]))
    facts.archive_facts(src, [{"text": cold, "added_turn": 4, "last_used": 1234}])
    _retire_apply(src, dst)
    assert_eq([f["text"] for f in facts.load_facts(dst)], [hot], "hot stays hot")
    dst_arch = facts.load_archive(dst)
    assert_eq([f["text"] for f in dst_arch], [cold], "cold stays cold")
    assert_eq((dst_arch[0]["added_turn"], dst_arch[0]["last_used"]), (4, 1234),
              "the archived row's own metadata came with it")
    assert_eq(facts.load_archive(src), [], "the source sidecar is emptied")


def test_retire_prefers_the_hot_copy_when_a_fact_is_in_both_source_layers():
    print("\n[test] /retire migrates the active copy and drops its cold twin")
    _wipe()
    src, dst = "phantom-7", "dest-7"
    text = "The user wants the prose written in past tense."
    # The archived twin deliberately carries the HIGHER (last_used, added_turn),
    # so D6's survivor rule on its own would pick the cold copy. Being in the
    # active set has to outrank that: the active copy is the one she is
    # currently being reminded of, and moving it to cold storage would take a
    # live fact out of the injected set on a technicality.
    facts.save_facts(src, [{"text": text, "added_turn": 1, "last_used": 100}])
    facts.archive_facts(src, [{"text": text, "added_turn": 9, "last_used": 900}])
    dry = _retire(src, dst)
    assert_true("[duplicate-in-source]" in dry, f"the twin is named: {dry}")
    _retire(f"{src} apply {_retire_code(dry, src)}", dst)
    assert_eq([f["text"] for f in facts.load_facts(dst)], [text], "one copy, hot")
    assert_eq(facts.load_archive(dst), [], "nothing landed in cold storage")


def test_retire_preserves_last_used_and_added_turn():
    print("\n[test] /retire moves a fact's metadata with it, unchanged")
    _wipe()
    src, dst = "phantom-8", "dest-8"
    facts.save_facts(src, [{"text": "Idris repaints the lamp housing each spring.",
                            "added_turn": 3, "last_used": 1700000042}])
    dry = _retire(src, dst)
    assert_true("last_used" in dry and "added_turn" in dry,
                "the plan states what happens to each field")
    assert_true("NOT meaningful here" in dry,
                "and says plainly which one does not survive as a signal")
    _retire(f"{src} apply {_retire_code(dry, src)}", dst)
    moved = facts.load_facts(dst)[0]
    assert_eq((moved["added_turn"], moved["last_used"]), (3, 1700000042),
              "both fields carried verbatim")


def test_retire_clears_every_layer_keyed_to_the_source():
    print("\n[test] /retire clears summary, episodic, persona and backfill state")
    _wipe()
    src, dst = "phantom-9", "dest-9"
    facts.save_facts(src, _rows(_REAL_FACTS[:2]))
    facts.archive_facts(src, _rows(["Cairn is three hours away by boat."]))
    persona.save_persona(src, "You are a patient writing companion.")
    summarizer.save_state(src, {"l1": [{"text": "a chunk", "first_turn": 1,
                                       "last_turn": 20}], "l2": [], "l3": None})
    backfill_sidecar = memory.storage_root() / "facts" / f"{src}.backfill.json"
    memory.atomic_write_json(backfill_sidecar, {"state": "failed"})

    out = _retire_apply(src, dst)
    assert_true(out.startswith("Retired "), f"applied: {out[:200]!r}")
    assert_eq(facts.load_facts(src), [], "facts gone")
    assert_eq(facts.load_archive(src), [], "archive sidecar gone")
    assert_true(persona.load_persona(src) is None, "persona gone")
    assert_true(not summarizer.load_state(src).get("l1"), "summary gone")
    assert_true(not backfill_sidecar.is_file(), "backfill sidecar gone")
    assert_true("Not everything went" not in out, f"nothing left behind: {out}")


def test_retire_snapshot_carries_the_archive_and_the_persona():
    print("\n[test] the pre-removal snapshot holds the two layers the bundle "
          "schema has no key for")
    # D19: "/forget can destroy a persona that no export can back up and no
    # import can restore." A snapshot missing the layers the operation deletes
    # is not an archive-before-removing.
    _wipe()
    src, dst = "phantom-10", "dest-10"
    cold = "Cairn is three hours away by boat."
    facts.save_facts(src, _rows(_REAL_FACTS[:2] + _GARBAGE_MARKUP))
    facts.archive_facts(src, _rows([cold]))
    persona.save_persona(src, "You are a patient writing companion.")
    _retire_apply(src, dst)
    snaps = portability.list_quarantine(src)
    assert_eq(len(snaps), 1, "exactly one snapshot published")
    assert_eq(list(snaps[0].parent.glob("*.partial")), [], "no .partial left behind")
    bundle = json.loads(snaps[0].read_text(encoding="utf-8"))
    q = bundle["quarantine"]
    assert_eq([f["text"] for f in q["archive"]], [cold], "the sidecar is in the snapshot")
    assert_eq(q["persona"]["persona_text"], "You are a patient writing companion.",
              "the persona is in the snapshot")
    # And the garbage: requirement 4 is "archives everything, including the
    # garbage", so a wrong classification is recoverable and not merely visible.
    snap_texts = {f["text"] for f in bundle["facts"]}
    for text in _GARBAGE_MARKUP:
        assert_true(text in snap_texts, f"dropped row is recoverable: {text!r}")
    # Still a valid v2.1 bundle: the extra keys did not break the import path.
    res = portability.import_conversation(
        bundle, target_conv_id="phantom-10-restored", overwrite=False)
    assert_eq(res["imported"]["facts"], len(_REAL_FACTS[:2] + _GARBAGE_MARKUP),
              "the snapshot restores through the existing import, unchanged")


def test_retire_aborts_when_the_snapshot_cannot_be_written():
    print("\n[test] /retire moves nothing if the pre-op snapshot fails")
    _wipe()
    src, dst = "phantom-11", "dest-11"
    facts.save_facts(src, _rows(_REAL_FACTS))
    code = _retire_code(_retire(src, dst), src)
    real = portability.quarantine_conversation

    def boom(conv_id, *, reason):
        raise portability.QuarantineError("simulated: no space left on device")

    portability.quarantine_conversation = boom
    try:
        out = _retire(f"{src} apply {code}", dst)
    finally:
        portability.quarantine_conversation = real
    assert_true("changed nothing" in out, f"abort stated plainly: {out!r}")
    assert_eq(len(facts.load_facts(src)), len(_REAL_FACTS), "source untouched")
    assert_eq(facts.load_facts(dst), [], "destination untouched")


def test_retire_refuses_when_a_layer_could_not_be_accounted_for():
    print("\n[test] /retire refuses rather than delete a layer it could not back up")
    # /tidy notes an unverified layer and carries on, because it only touches
    # facts. Retirement deletes every layer, so an unverified one is a refusal.
    _wipe()
    src, dst = "phantom-12", "dest-12"
    facts.save_facts(src, _rows(_REAL_FACTS))
    code = _retire_code(_retire(src, dst), src)
    real = portability.quarantine_conversation

    def half_blind(conv_id, *, reason):
        res = real(conv_id, reason=reason)
        res["unverified_layers"] = ["episodic (vector store unavailable)"]
        return res

    portability.quarantine_conversation = half_blind
    try:
        out = _retire(f"{src} apply {code}", dst)
    finally:
        portability.quarantine_conversation = real
    assert_true("I have changed nothing" in out, f"refused: {out!r}")
    assert_true("vector store unavailable" in out, "names the layer it could not prove")
    assert_eq(len(facts.load_facts(src)), len(_REAL_FACTS), "source untouched")
    assert_eq(facts.load_facts(dst), [], "destination untouched")


def test_retire_apply_without_a_code_refuses():
    print("\n[test] /retire ... apply with no code changes nothing")
    _wipe()
    src, dst = "phantom-13", "dest-13"
    facts.save_facts(src, _rows(_REAL_FACTS))
    out = _retire(f"{src} apply", dst)
    assert_true("needs the code" in out, f"refused with a reason: {out!r}")
    assert_eq(len(facts.load_facts(src)), len(_REAL_FACTS), "source untouched")


def test_retire_stale_code_refuses_and_reprints_the_plan():
    print("\n[test] /retire refuses a code issued for a different plan")
    _wipe()
    src, dst = "phantom-14", "dest-14"
    facts.save_facts(src, _rows(_REAL_FACTS))
    code = _retire_code(_retire(src, dst), src)
    facts.save_facts(src, facts.load_facts(src)
                     + _rows(["Brannock has one harbour."], start_turn=99))
    before = facts.load_facts(src)
    out = _retire(f"{src} apply {code}", dst)
    assert_true("out of date" in out, f"refused as stale: {out[:140]!r}")
    assert_true("DRY RUN" in out, "a fresh plan is printed instead")
    assert_eq(facts.load_facts(src), before, "nothing moved")
    assert_eq(facts.load_facts(dst), [], "nothing landed")


def test_retire_code_covers_the_destination_too():
    print("\n[test] the same source plan aimed at a different destination "
          "needs a different code")
    # A code that only hashed the source would let a plan reviewed against one
    # conversation be applied into another.
    _wipe()
    src = "phantom-15"
    facts.save_facts(src, _rows(_REAL_FACTS))
    a = _retire_code(_retire(src, "dest-15a"), src)
    b = _retire_code(_retire(src, "dest-15b"), src)
    assert_true(a != b, "the destination is part of the code")
    out = _retire(f"{src} apply {a}", "dest-15b")
    assert_true("out of date" in out, f"the other destination's code is refused: {out[:120]!r}")
    assert_eq(facts.load_facts("dest-15b"), [], "nothing landed")


def test_retire_is_idempotent():
    print("\n[test] /retire applied twice moves the facts once")
    _wipe()
    src, dst = "phantom-16", "dest-16"
    facts.save_facts(src, _rows(_REAL_FACTS + _GARBAGE_MARKUP))
    _retire_apply(src, dst)
    after_first = [f["text"] for f in facts.load_facts(dst)]
    second = _retire(src, dst)
    assert_true("no stored memory of any kind" in second,
                f"the second dry run has nothing to do: {second[:160]!r}")
    assert_eq([f["text"] for f in facts.load_facts(dst)], after_first,
              "the destination did not change")


def test_retire_finishes_an_interrupted_apply():
    print("\n[test] /retire resumes cleanly from a crash between the two halves")
    # Interrupted after the destination write and before the source clear, the
    # rows are in both places. A re-run must recognise that as "already in the
    # destination" and finish the removal — never duplicate, never re-drop.
    _wipe()
    src, dst = "phantom-17", "dest-17"
    facts.save_facts(src, _rows(_REAL_FACTS + _GARBAGE_MARKUP))
    # Exactly the state a crash in the middle leaves: destination written,
    # source still full.
    facts.save_facts(dst, _rows(_REAL_FACTS, start_turn=50))
    out = _retire_apply(src, dst)
    assert_true(out.startswith("Retired "), f"resumed and finished: {out[:200]!r}")
    landed = [f["text"] for f in facts.load_facts(dst)]
    assert_eq(len(landed), len(_REAL_FACTS), "no fact was duplicated")
    for text in _REAL_FACTS:
        assert_eq(landed.count(text), 1, f"exactly one copy of {text[:30]!r}")
    assert_eq(facts.load_facts(src), [], "and the source is emptied this time")


def test_retire_writes_the_destination_before_it_touches_the_source():
    print("\n[test] /retire crashing between the halves loses nothing")
    # "Migrates before it removes" as a testable property rather than as a
    # comment: fail the FIRST write to the source, which is the write that
    # immediately follows the destination writes. In the correct order the
    # source is still intact at that instant. In the reverse order the same
    # crash would have emptied it with nothing landed anywhere.
    _wipe()
    src, dst = "phantom-21", "dest-21"
    facts.save_facts(src, _rows(_REAL_FACTS))
    facts.archive_facts(src, _rows(["Cairn is three hours away by boat."]))
    code = _retire_code(_retire(src, dst), src)
    real_save_archive = facts.save_archive

    def fail_on_source(conv_id, rows):
        if conv_id == src:
            raise OSError("simulated: power loss between the two halves")
        return real_save_archive(conv_id, rows)

    commands.facts_module.save_archive = fail_on_source
    try:
        out = _retire(f"{src} apply {code}", dst)
    finally:
        commands.facts_module.save_archive = real_save_archive
    assert_true("Command failed" in out, f"the crash is not swallowed: {out!r}")
    assert_eq(len(facts.load_facts(src)), len(_REAL_FACTS),
              "the source still holds every active fact")
    assert_eq(len(facts.load_archive(src)), 1, "and its archived one")
    assert_eq(len(facts.load_facts(dst)), len(_REAL_FACTS),
              "the destination already has them — nothing is anywhere else")
    # And the resume finishes the job without duplicating anything.
    out2 = _retire_apply(src, dst)
    assert_true(out2.startswith("Retired "), f"resumed: {out2[:120]!r}")
    assert_eq(len(facts.load_facts(dst)), len(_REAL_FACTS), "still no duplicates")
    assert_eq(facts.load_facts(src), [], "and the source is empty now")


def test_retire_serializes_against_the_destination_conversation():
    print("\n[test] /retire parks on the DESTINATION's lock, not just the source's")
    # The destination is a live conversation with its own extraction tail. An
    # unlocked append lands under a tail that is parked mid-sequence on its own
    # pre-read snapshot, and that tail's next save deletes the migrated facts —
    # the F22/F3 shape, with a whole conversation's memory as the payload.
    _wipe()
    src, dst = "phantom-22", "dest-22"
    facts.save_facts(src, _rows(_REAL_FACTS[:3]))
    code = _retire_code(_retire(src, dst), src)

    async def scenario():
        async with memory.conv_lock(dst):
            task = asyncio.ensure_future(
                commands.handle_command("retire", f"{src} apply {code}", dst, ctx={}))
            for _ in range(50):
                await asyncio.sleep(0)
            parked = not task.done()
            landed_while_parked = len(facts.load_facts(dst))
        return parked, landed_while_parked, await task

    parked, landed, out = asyncio.run(scenario())
    assert_true(parked, "the apply parked while the destination lock was held")
    assert_eq(landed, 0, "and wrote nothing to the destination while parked")
    assert_true(out.startswith("Retired "), f"then finished once released: {out[:120]!r}")
    assert_eq(len(facts.load_facts(dst)), 3, "the facts landed after the lock cleared")


def test_retire_never_elides_the_dropped_list():
    print("\n[test] /retire prints every dropped row, however many there are")
    # The rule D6 states and this inherits: a row the operator confirms without
    # having seen is the exact failure the command exists to prevent.
    _wipe()
    src, dst = "phantom-18", "dest-18"
    junk = [f'"metric_{i}": {i},' for i in range(commands.TIDY_MAX_ROWS_SHOWN * 2)]
    facts.save_facts(src, _rows(junk))
    out = _retire(src, dst)
    assert_true("not shown" not in _drop_section(out),
                "the dropped list is never elided")
    for text in junk:
        assert_true(repr(text) in out, f"row shown verbatim: {text!r}")


def test_retire_warns_when_the_destination_would_overflow_its_facts_budget():
    print("\n[test] /retire says so when the move pushes the destination past "
          "what it can inject")
    _wipe()
    src, dst = "phantom-19", "dest-19"
    bulk = [f"Chapter {i} covers the crossing to Cairn and back again, "
            f"with the storm arriving on the return leg. " * 3 for i in range(40)]
    facts.save_facts(src, _rows(bulk))
    out = _retire(src, dst)
    assert_true("Heads up" in out, f"the overflow is disclosed: {out[-900:]}")
    assert_true("restored" in out, "and says the overflow is recoverable")


def test_retire_reports_the_bucket_will_re_form():
    print("\n[test] the reply does not claim the conversation is gone for good")
    _wipe()
    src, dst = "phantom-20", "dest-20"
    facts.save_facts(src, _rows(_REAL_FACTS[:2]))
    out = _retire_apply(src, dst)
    assert_true("start filling again" in out,
                f"the reply says what it cannot promise: {out}")


def test_every_retire_rule_has_a_printed_explanation():
    print("\n[test] every /retire rule id has an operator-readable sentence")
    ids = set(commands._RETIRE_DROP_ORDER) | set(commands._RETIRE_FLAG_ORDER)
    missing = sorted(i for i in ids if i not in commands._RETIRE_RULE_TEXT)
    assert_eq(missing, [], "no rule can reach the report as a bare id")


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
        # v3.1 D8 — /retire
        test_parse_retire_and_aliases,
        test_retire_without_a_source_explains_itself,
        test_retire_refuses_its_own_conversation,
        test_retire_rejects_anything_that_is_not_a_conversation_id,
        test_retire_on_a_source_with_no_memory_says_so,
        test_retire_dry_run_changes_nothing,
        test_retire_moves_real_facts_and_leaves_markup_behind,
        test_retire_keeps_every_row_it_cannot_prove_is_garbage,
        test_retire_drops_only_byte_identical_destination_duplicates,
        test_retire_drops_a_fact_the_destination_already_has_in_cold_storage,
        test_retire_moves_archived_rows_into_the_destination_archive,
        test_retire_prefers_the_hot_copy_when_a_fact_is_in_both_source_layers,
        test_retire_preserves_last_used_and_added_turn,
        test_retire_clears_every_layer_keyed_to_the_source,
        test_retire_snapshot_carries_the_archive_and_the_persona,
        test_retire_aborts_when_the_snapshot_cannot_be_written,
        test_retire_refuses_when_a_layer_could_not_be_accounted_for,
        test_retire_apply_without_a_code_refuses,
        test_retire_stale_code_refuses_and_reprints_the_plan,
        test_retire_code_covers_the_destination_too,
        test_retire_is_idempotent,
        test_retire_finishes_an_interrupted_apply,
        test_retire_writes_the_destination_before_it_touches_the_source,
        test_retire_serializes_against_the_destination_conversation,
        test_retire_never_elides_the_dropped_list,
        test_retire_warns_when_the_destination_would_overflow_its_facts_budget,
        test_retire_reports_the_bucket_will_re_form,
        test_every_retire_rule_has_a_printed_explanation,
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
