"""
CPU-only tests for what a user gets when a memory file cannot be read
(REMEDIATION.md v3.1, changes 2 and 3).

Two surfaces, one underlying condition:

  - main._clear_all_memory: an unreadable facts file must not abort the whole
    wipe. The user asked for this data to be gone; refusing to clear the three
    layers we CAN read leaves more behind than clearing them does. It clears
    what it can, reports "unreadable": ["facts"], and leaves the facts file
    itself on disk — it cannot be safely rewritten from an unknown state.

  - the healthy full-stack /forget: same HTTP surface, nothing faked, asserted
    against the disk rather than against the reply. A3's defect was that those
    two disagreed — the archive sidecar and the lazy-backfill gate both
    survived a wipe that reported a clean sweep — so the assertions here read
    the store back after the command rather than trusting its counters.

  - the slash-command reply: a real person types these into a chat box. What
    reaches her must say that nothing was deleted and that she did nothing
    wrong. The absolute path and the exception class belong in the log, and
    only in the log. Asserting the path is absent from the reply is the actual
    requirement, so that is what is asserted — not merely that some friendly
    words are present.

The corrupt-file trigger is the same one test_store_failures.py uses and for
the same reason (REMEDIATION.md §1.4): a genuinely corrupt file on disk is
what a torn write leaves behind, and it reaches read_json_strict's
JSONDecodeError arm without any patching of Path.stat.

Run: python test_degraded_forget.py
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, patch

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-degraded-forget-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "2000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_PERSONA_AUTO_DETECT_MIN_CHARS"] = "50"

import backfill  # noqa: E402
import continuity  # noqa: E402
import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

# raise_server_exceptions=False so an unhandled exception inside the handler
# arrives as a real 500 response instead of propagating into the test runner.
# A user-facing surface that 500s is a result worth asserting on, not a crash
# worth hiding.
client = TestClient(main.app, raise_server_exceptions=False)

# Separate client for the admin surface: client=127.0.0.1 satisfies
# _require_localhost without loosening COMPACTOR_ADMIN_BIND, so the gate stays
# genuinely under test on the chat client above.
admin_client = TestClient(
    main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False
)


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
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


def corrupt(path, content: str = "{ not valid js"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _seed_other_layers(cid: str) -> None:
    """Give the conv a real summary and a real persona so the assertions below
    can tell "cleared" from "was never there"."""
    summarizer.save_state(cid, {
        "l1": [{"text": "Scene one.", "first_turn": 1, "last_turn": 20}],
        "l2": [], "l3": None, "last_summarized_turn": 20,
    })
    persona.save_persona(
        cid, "You are Sam Cole, a hardboiled detective in a rain-slick city.",
        source="admin",
    )


def assert_no_path_leak(text: str, path, label: str) -> None:
    """The reply must not carry the storage path in any form a chat client
    would render. Checked in three shapes because a Windows path survives a
    round trip through json.dumps as escaped backslashes, and because the
    bare filename alone is still an internal detail the user cannot act on.
    """
    p = str(path)
    variants = {p, p.replace("\\", "/"), p.replace("\\", "\\\\")}
    for v in variants:
        if v in text:
            print(f"FAIL {label}: reply contains the storage path {v!r}\n"
                  f"       reply was: {text!r}")
            sys.exit(1)
    if _TMP_ROOT in text or _TMP_ROOT.replace("\\", "/") in text:
        print(f"FAIL {label}: reply contains the storage root\n"
              f"       reply was: {text!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def _command_reply(text: str, conv_id: str) -> str:
    r = client.post("/v1/chat/completions", json={
        "model": "x",
        "messages": [{"role": "user", "content": text}],
        "stream": False,
    }, headers={"X-Conversation-Id": conv_id})
    assert_eq(r.status_code, 200, f"{text} short-circuits with 200")
    return r.json()["choices"][0]["message"]["content"]


def _wipe(cid: str) -> dict:
    """Drive _clear_all_memory and report a raise as a failed assertion rather
    than an unhandled traceback. It is documented not to raise on an
    unreadable layer, so a raise is this file's finding to state, not a
    fixture problem to swallow."""
    try:
        return asyncio.run(main._clear_all_memory(cid, source="test"))
    except Exception as e:
        print(f"FAIL _clear_all_memory raised {type(e).__name__}: {e}\n"
              f"       it is documented to clear what it can and report the "
              f"rest in 'unreadable'")
        sys.exit(1)


# ---------------------------------------------------------------------------
# B — _clear_all_memory clears what it can read
# ---------------------------------------------------------------------------

def test_clear_all_memory_partial_wipe_on_corrupt_facts():
    print("\n[test] a corrupt facts file does not abort the wipe of the other layers")
    _wipe_storage()
    cid = "partial-wipe"
    before = corrupt(memory.facts_path(cid)).read_text(encoding="utf-8")
    _seed_other_layers(cid)
    # A non-zero episodic count proves the wipe reached the layer AFTER the
    # facts failure rather than unwinding at it. The real function returns 0
    # here (chromadb absent), which cannot distinguish "cleared nothing" from
    # "never ran".
    with patch.object(retrieval, "forget_conversation", lambda c: 7):
        result = _wipe(cid)

    assert_eq(result["unreadable"], ["facts"], "unreadable names exactly facts")
    assert_eq(result["forgotten_facts"], 0, "no fact count claimed")
    assert_eq(result["forgotten_episodic"], 7, "episodic still cleared")
    assert_eq(result["forgotten_summary"], True, "summary still cleared")
    assert_eq(result["forgotten_persona"], True, "persona still cleared")
    assert_true(not memory.summary_path(cid).is_file(), "summary file gone")
    assert_true(not memory.persona_path(cid).is_file(), "persona file gone")
    # The point of leaving it: its real contents are unknown, so there is no
    # safe value to rewrite it with. Deleting it would be a decision made on
    # data nobody has read.
    assert_true(memory.facts_path(cid).is_file(), "facts file still in place")
    assert_eq(memory.facts_path(cid).read_text(encoding="utf-8"), before,
              "facts file byte-identical — not rewritten from an unknown state")


def test_clear_all_memory_partial_wipe_on_facts_write_failure():
    print("\n[test] round-4 item 2 — a facts WRITE failure (not a read "
          "failure) does not abort the wipe of the other layers either")
    # v3.1.9.8 round-4 review ("audit every write on the wipe and forget
    # paths"): the sibling of test_clear_all_memory_partial_wipe_on_
    # corrupt_facts above — that one covers load_facts raising
    # StoreUnreadable (a READ failure); this covers facts.save_facts
    # itself raising a plain OSError (a WRITE failure, e.g. ENOSPC) on a
    # facts file that reads FINE. Before this fix, only StoreUnreadable
    # was caught around this call — an OSError from the write propagated
    # straight out of _clear_all_memory, aborting episodic/summary/
    # persona/chapters before any of them ran.
    _wipe_storage()
    cid = "facts-write-fails"
    facts.save_facts(cid, [
        {"text": "her sister Maren has cancer", "added_turn": 1, "last_used": 1},
    ])
    _seed_other_layers(cid)

    with patch.object(retrieval, "forget_conversation", lambda c: 7), \
         patch.object(facts, "save_facts", side_effect=OSError(28, "No space left on device")):
        result = _wipe(cid)

    # v3.1.9.8 (reply-wording fix, post round-4): a WRITE failure (this one
    # — save_facts itself raised) is reported separately from a READ
    # failure now. `unreadable` means the facts file could not be READ
    # (load_facts raised StoreUnreadable) — that never happened here, the
    # read above this succeeded fine. Folding a write failure into
    # `unreadable` is exactly what made commands._handle_forget tell her
    # "I could not READ the stored facts" for a delete that failed, not a
    # read.
    assert_eq(result["unreadable"], [],
              "*** BUG CHECK: a write failure must not be reported as unreadable")
    assert_eq(result["write_failed"], ["facts"],
              "*** BUG CHECK: reported as write_failed/unconfirmed, not silently skipped")
    assert_eq(result["forgotten_facts"], 0, "no fact count claimed — the write never confirmed landing")
    assert_eq(result["forgotten_episodic"], 7,
              "*** BUG CHECK: episodic still cleared — the OSError did not abort the function")
    assert_eq(result["forgotten_summary"], True, "summary still cleared")
    assert_eq(result["forgotten_persona"], True, "persona still cleared")
    assert_true(not memory.summary_path(cid).is_file(), "summary file gone")
    assert_true(not memory.persona_path(cid).is_file(), "persona file gone")


def test_clear_all_memory_does_not_raise_on_corrupt_facts():
    print("\n[test] the corrupt-facts wipe returns a dict rather than raising")
    _wipe_storage()
    cid = "no-raise"
    corrupt(memory.facts_path(cid))
    result = _wipe(cid)
    assert_true(isinstance(result, dict), "returned a counters dict")
    assert_eq(result["unreadable"], ["facts"], "unreadable reported in the dict")


def test_clear_all_memory_healthy_reports_nothing_unreadable():
    print("\n[test] the healthy wipe still reports an empty unreadable list")
    _wipe_storage()
    cid = "healthy-wipe"
    facts.save_facts(cid, [
        {"text": "Lyra is a half-elf ranger.", "added_turn": 1, "last_used": 1},
        {"text": "The setting is Aethermere.", "added_turn": 2, "last_used": 2},
    ])
    _seed_other_layers(cid)
    result = _wipe(cid)
    assert_eq(result["unreadable"], [], "nothing unreadable")
    assert_eq(result["forgotten_facts"], 2, "both facts counted")
    assert_eq(facts.load_facts(cid), [], "facts cleared on disk")
    assert_true(not memory.summary_path(cid).is_file(), "summary file gone")
    assert_true(not memory.persona_path(cid).is_file(), "persona file gone")


def test_clear_all_memory_absent_facts_is_not_unreadable():
    print("\n[test] a conv with no facts file at all is not reported unreadable")
    # The distinction the whole v3.1 chain rests on: absent is empty, and only
    # unreadable is unreadable. If this ever starts reporting ["facts"], every
    # /forget on a fresh conversation tells the user something is broken.
    _wipe_storage()
    result = _wipe("never-seen")
    assert_eq(result["unreadable"], [], "absent facts file -> unreadable is empty")
    assert_eq(result["forgotten_facts"], 0, "nothing to forget")


def test_admin_forget_endpoint_reports_unreadable():
    print("\n[test] DELETE /admin/conversations/{id}/facts -> 200 carrying unreadable")
    # The operator-facing half of the same contract. A partial wipe is a
    # result, not a server error: the caller needs the counters AND the list
    # of layers that could not be read, which a 500 gives it neither of.
    _wipe_storage()
    cid = "admin-forget-corrupt"
    corrupt(memory.facts_path(cid))
    r = admin_client.delete(f"/admin/conversations/{cid}/facts")
    assert_eq(r.status_code, 200, "200, not 500")
    assert_eq(r.json()["unreadable"], ["facts"],
              "response body names the layer it could not read")


def test_forget_command_must_not_report_a_clean_wipe():
    print("\n[test] /forget must not say 'nothing to forget' when a layer was unreadable")
    # main.py:1924 states the contract this asserts: "Callers must not report a
    # clean wipe when this is non-empty." A facts file with unknown contents is
    # sitting on disk; telling the user the conversation had no stored memory
    # is the single most misleading thing this surface can say.
    _wipe_storage()
    cid = "forget-cmd-corrupt"
    corrupt(memory.facts_path(cid))
    reply = _command_reply("/forget", cid)
    assert_true(
        "no stored memory" not in reply.lower(),
        f"reply does not claim the conversation had no stored memory: {reply!r}",
    )
    assert_true(
        any(w in reply.lower() for w in ("could not", "couldn't", "unreadable",
                                         "unable")),
        f"reply says a layer could not be read: {reply!r}",
    )


def test_real_forget_clears_every_layer_end_to_end():
    print("\n[test] the real /forget leaves no layer behind (no fakes in the path)")
    # Everything else in this file that exercises /forget does it on a store
    # that is already broken. This one runs the healthy path through the same
    # HTTP surface, with main._clear_all_memory really wired, and asserts
    # against the disk rather than against the reply — the A3 defect was
    # precisely that those two disagreed.
    #
    # The archive sidecar is the layer the wipe never touched: prune_facts and
    # archive_stale_facts MOVE evicted facts there rather than unlinking them
    # (F9), and nothing in the codebase had ever deleted one.
    _wipe_storage()
    cid = "forget-e2e"
    facts.save_facts(cid, [
        {"text": "Lyra is a half-elf ranger.", "added_turn": 1, "last_used": 1},
    ])
    facts.save_archive(cid, [
        {"text": "Her mother's name is Selene.", "added_turn": 1,
         "last_used": 1, "archived_at": 1},
    ])
    _seed_other_layers(cid)

    reply = _command_reply("/forget", cid)

    assert_eq(facts.load_facts(cid), [], "facts gone")
    assert_eq(facts.load_archive(cid), [], "archive sidecar gone")
    assert_true(not memory.summary_path(cid).is_file(), "summary gone")
    assert_true(not memory.persona_path(cid).is_file(), "persona gone")
    assert_true("still storing" not in reply.lower(),
                f"and the reply agrees with the disk: {reply!r}")
    for expected in ("1 fact(s)", "1 archived fact(s)", "summary state", "persona"):
        assert_true(expected in reply, f"reply names {expected}: {reply!r}")


def test_real_forget_closes_the_backfill_gate():
    print("\n[test] the real /forget stops the lazy backfill re-extracting the history")
    # The path that put memory back without any concurrency at all.
    # backfill.needs_backfill gates on facts_path(conv_id).is_file(); a
    # conversation whose memory was a summary and a persona has no facts file,
    # so the wipe left the gate open and the NEXT request would start a
    # background extraction over the whole message history — the history the
    # user had just asked to be forgotten — and write the result to disk.
    #
    # Driven through the HTTP surface so the assertion is about what a user
    # typing /forget actually gets, not about the handler in isolation.
    _wipe_storage()
    cid = "forget-backfill-gate"
    msgs = [
        {"role": "user", "content": "My protagonist Lyra is a half-elf ranger."},
        {"role": "assistant", "content": "Where is she from?"},
        {"role": "user", "content": "Aethermere. Her mother Selene died there."},
        {"role": "assistant", "content": "A strong hook."},
    ]
    _seed_other_layers(cid)
    assert_true(backfill.needs_backfill(cid, msgs),
                "precondition: the forgotten history was backfillable")

    reply = _command_reply("/forget", cid)

    assert_true(memory.facts_path(cid).is_file(),
                f"the wipe left an empty facts store behind: {reply!r}")
    assert_eq(facts.load_facts(cid), [], "and it is empty")
    assert_true(not backfill.needs_backfill(cid, msgs),
                "the backfill can no longer reconstruct the forgotten history")


# ---------------------------------------------------------------------------
# B2 — a ledger WRITE failure must never block the deletion she asked for
# ---------------------------------------------------------------------------
#
# `_ledger_path` (continuity.py) lives in CONTROL_ROOT. Both wipe paths write
# the ledger BEFORE deleting: commands._handle_forget's selective branch
# calls continuity.suppress_needle -> record_targeted_forget before
# facts_module.save_facts; the full wipe's continuity.wipe_conversation_band
# calls record_wipe before it unlinks any memory_layer_paths. Before this
# fix, an OSError out of `_write_ledger` (an unwritable CONTROL_ROOT, a full
# disk) propagated straight out of record_wipe/record_targeted_forget,
# raising through commands._handle_forget or main._clear_all_memory before
# either reached its own deletion — so /forget raised, and her facts were
# NOT removed. record_wipe/record_targeted_forget/retire_tombstone now catch
# that failure (via `continuity._write_ledger_best_effort`), log it loudly at
# ERROR, and let the deletion proceed regardless — the same "never block a
# wipe she asked for" principle `record_wipe` already applies to an
# unreadable EXISTING ledger, extended to a WRITE failure.
#
# Per the task's own guidance: monkeypatching `continuity._write_ledger` to
# raise OSError is preferred over chmod'ing CONTROL_ROOT to 0500, both
# because it is deterministic (a chmod is a no-op for a process running as
# root, which would silently skip the scenario rather than test it) and
# because it isolates the fault to exactly the write this fix is about.


def _capture_compactor_log():
    handler = logging.StreamHandler(stream=open(os.devnull, "w"))
    records: list[logging.LogRecord] = []

    class _Recorder(logging.Handler):
        def emit(self, record):
            records.append(record)

    rec = _Recorder()
    lg = logging.getLogger("compactor")
    lg.addHandler(rec)
    return lg, rec, records


def test_selective_forget_removes_facts_even_when_the_ledger_write_fails():
    print("\n[test] /forget X removes the fact even when the wipe ledger cannot be written")
    _wipe_storage()
    cid = "forget-selective-ledger-failure"
    facts.save_facts(cid, [
        {"text": "Lyra is a half-elf ranger.", "added_turn": 1, "last_used": 1},
    ])

    lg, rec, records = _capture_compactor_log()
    try:
        with patch.object(
            continuity, "_write_ledger",
            side_effect=OSError("[Errno 13] Permission denied: CONTROL_ROOT/wipes"),
        ):
            reply = _command_reply("/forget Lyra", cid)
    finally:
        lg.removeHandler(rec)

    assert_eq(facts.load_facts(cid), [], "the fact is removed despite the ledger write failure")
    assert_true("Forgot" in reply, f"the command reports success: {reply!r}")
    # v3.1.9.8 round-2/round-3 review (item 1/3, HIGH/MEDIUM): she deserves
    # to know the recording itself did not fully stick — but round 3's
    # pending-intent design means the band still fails closed for this
    # conv (the sidecar caught it), so the note is calm ("stays paused"),
    # NOT a claim that the words could come back — that claim is reserved
    # for the double-failure residual only (see commands._ledger_pending_
    # note's docstring and test_continuity.py's double-failure test).
    assert_true(
        "saved durably" in reply.lower() and "paused" in reply.lower(),
        f"the reply carries the calm, accurate note (see _ledger_pending_"
        f"note): {reply!r}",
    )
    assert_true(
        "come back" not in reply.lower() and "reappear" not in reply.lower(),
        f"*** the reply must NOT warn the words could come back — the "
        f"sidecar caught this pending intent, so nothing should reappear "
        f"(got {reply!r})",
    )
    assert_true(
        any(
            r.levelno == logging.ERROR
            and "ledger" in r.getMessage().lower()
            and cid in r.getMessage()
            for r in records
        ),
        "the ledger write failure is logged loudly (ERROR, naming the conv)",
    )
    assert_true(
        any("pending intent" in r.getMessage().lower() for r in records if r.levelno == logging.ERROR),
        "the log line names the pending-intent mechanism that is now "
        "carrying this forward, not just a bare 'degraded' claim",
    )


def test_full_forget_removes_facts_even_when_the_ledger_write_fails():
    print("\n[test] full /forget clears memory even when the wipe ledger cannot be written")
    _wipe_storage()
    cid = "forget-full-ledger-failure"
    facts.save_facts(cid, [
        {"text": "Lyra is a half-elf ranger.", "added_turn": 1, "last_used": 1},
    ])
    _seed_other_layers(cid)

    lg, rec, records = _capture_compactor_log()
    try:
        with patch.object(
            continuity, "_write_ledger",
            side_effect=OSError("[Errno 13] Permission denied: CONTROL_ROOT/wipes"),
        ):
            reply = _command_reply("/forget", cid)
    finally:
        lg.removeHandler(rec)

    assert_eq(facts.load_facts(cid), [], "facts are gone despite the ledger write failure")
    assert_true(not memory.summary_path(cid).is_file(), "summary is gone too")
    assert_true(not memory.persona_path(cid).is_file(), "persona is gone too")
    assert_true("still storing" not in reply.lower(), f"the reply agrees the wipe went through: {reply!r}")
    # v3.1.9.8 round-2/round-3 review (item 1/3, HIGH/MEDIUM): same calm,
    # accurate note as the selective branch — see that test's comment for
    # why "stays paused" replaces the old, overstated "could come back"
    # (round 2) / "storage problem ... restored from a backup" (round 1)
    # wording.
    assert_true(
        "saved durably" in reply.lower() and "paused" in reply.lower(),
        f"the reply carries the calm, accurate note: {reply!r}",
    )
    assert_true(
        "come back" not in reply.lower() and "reappear" not in reply.lower(),
        f"*** the reply must NOT warn the words could come back — the "
        f"sidecar caught this pending intent (got {reply!r})",
    )
    assert_true(
        any(
            r.levelno == logging.ERROR
            and "ledger" in r.getMessage().lower()
            and cid in r.getMessage()
            for r in records
        ),
        "the ledger write failure is logged loudly (ERROR, naming the conv)",
    )
    assert_true(
        any("pending intent" in r.getMessage().lower() for r in records if r.levelno == logging.ERROR),
        "the log line names the pending-intent mechanism",
    )


def test_ledger_write_failure_does_not_bump_the_epoch():
    print("\n[test] a failed ledger write leaves the persisted epoch unchanged "
          "(best-effort, not a lie)")
    # v3.1.9.8 round-2 review (item 4, LOW): record_wipe used to return the
    # epoch it merely INTENDED to persist even when the write failed — a
    # caller trusting that number would believe something was on disk that
    # never got there. It now returns None on a failed write instead (and
    # the conv is marked degraded, which fails current_ledger_epoch closed
    # too — see test_continuity.py's item-1 tests for that half).
    _wipe_storage()
    cid = "forget-ledger-failure-epoch"
    with patch.object(continuity, "_write_ledger", side_effect=OSError("no space left on device")):
        got = continuity.record_wipe(cid)
    assert_eq(got, None, "record_wipe returns None, not a number nothing backs")
    assert_true(continuity.ledger_degraded(cid), "and the conv is marked degraded")
    assert_true(not continuity._ledger_path(cid).is_file(),
                "the ledger file itself was never written")


# ---------------------------------------------------------------------------
# Round-4 review, item 2 (HIGH — probe r3/p2.py): "audit every write on the
# wipe and forget paths — none may abort the deletion." Reproduces the
# reviewer's own probe scope: every `memory.atomic_write_json` call AND
# `continuity._write_ledger` fail with ENOSPC — this hits the wipe LEDGER,
# the wipe-MARKER cache (continuity.py, the specific bug the probe found),
# and the backfill sidecar state (backfill.mark_wiped), all of which sit
# BEFORE the facts/summary/persona/chapters deletes in the wipe sequence.
# facts.save_facts is deliberately NOT affected by this patch (it imports
# `atomic_write_json` by name at module load, a separate binding from
# `memory.atomic_write_json` — the same reason the probe's own facts file
# ends up genuinely empty), matching a realistic fault where the WIPE
# BOOKKEEPING's own storage (a different volume, in production terms) is
# what is out of space, not the facts store itself.
# ---------------------------------------------------------------------------

def _enospc_on_bookkeeping_writes():
    """Context manager-ish pair of patches, matching probe r3/p2.py
    exactly: every `memory.atomic_write_json` call and
    `continuity._write_ledger` call raises ENOSPC."""
    return (
        patch.object(memory, "atomic_write_json", side_effect=OSError(28, "No space left on device")),
        patch.object(continuity, "_write_ledger", side_effect=OSError(28, "No space left on device")),
    )


def test_full_wipe_still_deletes_her_data_under_enospc_on_every_bookkeeping_write():
    print("\n[test] round-4 item 2 — a full /forget still deletes her data "
          "when the ledger, wipe marker AND backfill state all hit ENOSPC")
    _wipe_storage()
    cid = "enospc-full-wipe"
    facts.save_facts(cid, [
        {"text": "her sister Maren has cancer", "added_turn": 1, "last_used": 1},
    ])
    _seed_other_layers(cid)

    lg, rec, records = _capture_compactor_log()
    p1, p2 = _enospc_on_bookkeeping_writes()
    try:
        with p1, p2:
            reply = _command_reply("/forget", cid)
    finally:
        lg.removeHandler(rec)

    # THE bug (probe p2.py): before this fix, record_wipe's marker write
    # raised straight out of wipe_conversation_band, aborting the wipe
    # before any deletion ran at all.
    assert_eq(facts.load_facts(cid), [],
              "*** BUG CHECK: facts are gone despite ENOSPC on every "
              "bookkeeping write — the wipe was never aborted")
    assert_true(not memory.summary_path(cid).is_file(), "summary is gone too")
    assert_true(not memory.persona_path(cid).is_file(), "persona is gone too")
    assert_true("Forgot" in reply or "Cleared" in reply,
                f"the reply reports what actually happened, truthfully: {reply!r}")
    assert_true("still storing" not in reply.lower(),
                f"the reply does not falsely claim residue: {reply!r}")
    # The log must not claim "proceeding with the deletion anyway" while
    # secretly having aborted — it can only say that truthfully if the
    # deletion that follows the log line actually runs, which this
    # asserts indirectly via the facts/summary/persona checks above.
    assert_true(
        any(r.levelno == logging.ERROR and "wipe-marker" in r.getMessage().lower() for r in records),
        f"the wipe-marker write failure is itself logged (not silently "
        f"swallowed) (got {[r.getMessage() for r in records if r.levelno == logging.ERROR]})",
    )


def test_selective_forget_still_deletes_her_data_under_enospc_on_every_bookkeeping_write():
    print("\n[test] round-4 item 2 — a selective /forget X still deletes the "
          "fact when the ledger hits ENOSPC (facts.save_facts itself unaffected)")
    _wipe_storage()
    cid = "enospc-selective-forget"
    facts.save_facts(cid, [
        {"text": "her sister Maren has cancer", "added_turn": 1, "last_used": 1},
    ])

    lg, rec, records = _capture_compactor_log()
    p1, p2 = _enospc_on_bookkeeping_writes()
    try:
        with p1, p2:
            reply = _command_reply("/forget Maren", cid)
    finally:
        lg.removeHandler(rec)

    assert_eq(facts.load_facts(cid), [],
              "the fact is removed despite ENOSPC on the ledger write")
    assert_true("Forgot" in reply, f"the reply reports success: {reply!r}")
    # ENOSPC on `memory.atomic_write_json` also breaks the pending-intent
    # durability sidecar (`_persist_pending_sidecar` uses it too), so this
    # lands in the DOUBLE-FAILURE note ("could not be saved at all right
    # now"), not the single-failure "stays paused" one — either is an
    # honest degraded-recording note; only "come back"/"restarted" is
    # specific to the double-failure residual this scenario actually hits.
    assert_true(
        "saved durably" in reply.lower() or "paused" in reply.lower()
        or ("saved" in reply.lower() and "restart" in reply.lower()),
        f"and honestly notes the recording itself is degraded: {reply!r}",
    )
    assert_true(
        any(r.levelno >= logging.ERROR and cid in r.getMessage() for r in records),
        "the ledger write failure is logged (ERROR for a single failure, "
        "CRITICAL for the double-failure residual this scenario hits)",
    )


# ---------------------------------------------------------------------------
# C — the slash-command reply on StoreUnreadable
# ---------------------------------------------------------------------------

def test_command_reply_hides_path_when_persona_unreadable():
    print("\n[test] /why with a corrupt persona -> plain language, no path, no class name")
    _wipe_storage()
    cid = "cmd-persona-corrupt"
    p = corrupt(memory.persona_path(cid))
    reply = _command_reply("/why", cid)
    assert_no_path_leak(reply, p, "no storage path in the reply")
    assert_true("StoreUnreadable" not in reply, "no exception class name in the reply")
    assert_true("JSONDecodeError" not in reply, "no underlying exception name either")
    assert_true("Nothing has been deleted" in reply,
                f"reply reassures that nothing was deleted: {reply!r}")


def test_command_reply_streams_the_same_plain_language():
    print("\n[test] the streaming path carries the same plain-language reply")
    # OpenWebUI streams by default, so the non-streaming assertion alone would
    # leave the shape most users actually see untested.
    _wipe_storage()
    cid = "cmd-persona-corrupt-stream"
    p = corrupt(memory.persona_path(cid))
    r = client.post("/v1/chat/completions", json={
        "model": "x",
        "messages": [{"role": "user", "content": "/why"}],
        "stream": True,
    }, headers={"X-Conversation-Id": cid})
    assert_eq(r.status_code, 200, "stream opens 200")
    # Pull the assistant text back out of the SSE frames so the assertion runs
    # against what the client renders, not against the escaped wire bytes.
    content = ""
    for line in r.text.splitlines():
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: "):])
        content += chunk["choices"][0].get("delta", {}).get("content", "") or ""
    assert_true("Nothing has been deleted" in content,
                f"plain language present in the stream: {content!r}")
    assert_no_path_leak(content, p, "no storage path in the streamed reply")


def test_command_reply_hides_path_when_facts_unreadable():
    print("\n[test] /list-facts with a corrupt facts file -> plain language, no path")
    # Same condition, different layer, and the one a user is most likely to
    # hit: /list-facts is the command people type when they suspect memory
    # trouble in the first place.
    _wipe_storage()
    cid = "cmd-facts-corrupt"
    p = corrupt(memory.facts_path(cid))
    reply = _command_reply("/list-facts", cid)
    assert_no_path_leak(reply, p, "no storage path in the reply")
    assert_true("StoreUnreadable" not in reply, "no exception class name in the reply")
    assert_true("Nothing has been deleted" in reply,
                f"reply reassures that nothing was deleted: {reply!r}")


def test_command_reply_unaffected_when_store_is_healthy():
    print("\n[test] a healthy store still gets the normal command output")
    _wipe_storage()
    cid = "cmd-healthy"
    facts.save_facts(cid, [{"text": "Lyra is a half-elf ranger.",
                            "added_turn": 1, "last_used": 1}])
    reply = _command_reply("/list-facts", cid)
    assert_true("Lyra is a half-elf ranger." in reply, "the fact is listed")
    assert_true("Nothing has been deleted" not in reply,
                "the degraded message does not fire on a healthy store")


def test_forget_command_write_failure_says_could_not_remove_not_read():
    print("\n[test] a facts WRITE failure says 'could not remove', not 'could "
          "not read', and keeps the truthful still-storing note")
    # v3.1.9.8 (reply-wording fix, post round-4). Before this, a facts-wipe
    # WRITE failure (save_facts raising OSError — the file read fine; only
    # the delete-by-overwrite failed) was folded into the same `unreadable`
    # bucket as a genuine READ failure (StoreUnreadable), so the reply told
    # her "I could not read the stored facts" for a delete that failed, not
    # a read. The facts were never actually removed, so _memory_residue's
    # own re-check (unchanged by this fix) still finds them and appends the
    # "still storing... run /forget again" note — that part must survive.
    _wipe_storage()
    cid = "forget-cmd-write-fails"
    facts.save_facts(cid, [
        {"text": "her sister Maren has cancer", "added_turn": 1, "last_used": 1},
    ])
    with patch.object(facts, "save_facts",
                       side_effect=OSError(28, "No space left on device")):
        reply = _command_reply("/forget", cid)

    # v3.1.9.8 round-2 review (item 1, MED): "could not remove" ALONE also
    # matches the older, unrelated "still storing... I could not remove
    # that in this pass" sentence (_handle_forget's `still` branch), which
    # fires anyway because the facts genuinely were not removed — so this
    # assertion passed even with the new sentence deleted entirely
    # (reviewer mutant m1b, EXIT=0). Assert on text unique to the NEW
    # sentence instead: "could not remove the stored facts" only appears
    # in the write_failed clause this fix added.
    assert_true(
        "could not remove the stored facts" in reply.lower(),
        f"*** BUG CHECK: reply describes the write failure as a removal "
        f"failure, in the NEW sentence this fix added (not just the "
        f"pre-existing 'still storing... run /forget again' note): {reply!r}",
    )
    assert_true(
        "i could not read the stored" not in reply.lower(),
        f"*** BUG CHECK: reply must not misdescribe a write failure as a "
        f"read failure: {reply!r}",
    )
    assert_true(
        "still storing" in reply.lower() and "run /forget again" in reply.lower(),
        f"reply keeps the truthful still-storing/run-again note: {reply!r}",
    )


def test_forget_reply_truthful_across_read_write_failure_combinations():
    print("\n[test] the /forget reply is truthful for every combination of "
          "a read failure and a write failure (table-driven)")
    # v3.1.9.8 round-2 review (item 6, LOW). Before this, the write-failure
    # sentence unconditionally said "even though the rest of this wipe went
    # through" — false whenever a READ failure (summary unreadable, say)
    # ALSO left another layer untouched: that directly contradicts the
    # "I could not read the stored summary... so I left it untouched"
    # sentence appended just above it. Four cases below; each asserts only
    # what is actually true for that combination.
    read_err = lambda cid: memory.StoreUnreadable(
        memory.summary_path(cid), ValueError("bad json"))
    write_err = OSError(28, "No space left on device")

    def _seed(cid: str) -> None:
        facts.save_facts(cid, [
            {"text": "her sister Maren has cancer", "added_turn": 1, "last_used": 1},
        ])

    # --- A: read failure only (summary unreadable), no write failure -----
    _wipe_storage()
    cid = "table-read-only"
    _seed(cid)
    with patch.object(summarizer, "load_state", side_effect=read_err(cid)):
        reply = _command_reply("/forget", cid)
    assert_true("i could not read the stored summary" in reply.lower(),
                f"[A] names the unreadable summary: {reply!r}")
    assert_true("could not remove the stored" not in reply.lower(),
                f"[A] *** BUG CHECK: no write-failure sentence when nothing "
                f"failed to WRITE: {reply!r}")
    assert_true("even though the rest of this wipe went through" not in reply.lower(),
                f"[A] *** BUG CHECK: the write-failure-only claim must not "
                f"appear without a write failure: {reply!r}")

    # --- B: write failure only, no read failure ---------------------------
    _wipe_storage()
    cid = "table-write-only"
    _seed(cid)
    with patch.object(facts, "save_facts", side_effect=write_err):
        reply = _command_reply("/forget", cid)
    assert_true("could not remove the stored facts" in reply.lower(),
                f"[B] names the write failure: {reply!r}")
    assert_true("i could not read the stored" not in reply.lower(),
                f"[B] *** BUG CHECK: no read-failure sentence when nothing "
                f"failed to READ: {reply!r}")
    assert_true("every other layer was cleared" in reply.lower(),
                f"[B] the 'rest of the wipe went through' claim IS true "
                f"here (nothing else failed), so it should still appear: "
                f"{reply!r}")

    # --- C: BOTH a read failure and a write failure ------------------------
    _wipe_storage()
    cid = "table-both"
    _seed(cid)
    with patch.object(summarizer, "load_state", side_effect=read_err(cid)), \
         patch.object(facts, "save_facts", side_effect=write_err):
        reply = _command_reply("/forget", cid)
    assert_true("i could not read the stored summary" in reply.lower(),
                f"[C] names the unreadable summary: {reply!r}")
    assert_true("could not remove the stored facts" in reply.lower(),
                f"[C] names the write failure: {reply!r}")
    assert_true(
        "even though the rest of this wipe went through" not in reply.lower()
        and "every other layer was cleared" not in reply.lower(),
        f"[C] *** BUG CHECK: must NOT claim every other layer was cleared "
        f"— the summary was left untouched too (contradicts the sentence "
        f"immediately above it): {reply!r}",
    )
    assert_true("some other memory layers were also left untouched" in reply.lower(),
                f"[C] the write-failure sentence points at the read-failure "
                f"reason instead of a false blanket claim: {reply!r}")

    # --- D: write failure on the first pass, success on the bounded retry -
    _wipe_storage()
    cid = "table-write-then-retry-success"
    _seed(cid)
    # One full _wipe_all_layers pass calls facts.save_facts up to twice
    # (once inside _clear_all_memory, once for the empty-facts tombstone —
    # see _wipe_all_layers). Failing both calls in pass 1 leaves the facts
    # genuinely still on disk, which is what makes _handle_forget's own
    # bounded retry actually fire; succeeding both calls in pass 2 (the
    # retry) is what "success on the retry" means here. A plain
    # side_effect=[..., None, None] would make the mock swallow those
    # calls silently (returning None without writing anything) — the
    # "success" calls must actually reach the real save_facts so the
    # facts genuinely land empty on disk, which is what the retry
    # succeeding actually means.
    _real_save_facts = facts.save_facts
    _calls = {"n": 0}

    def _flaky_save_facts(conv_id_arg, value):
        _calls["n"] += 1
        if _calls["n"] <= 2:
            raise write_err
        return _real_save_facts(conv_id_arg, value)

    with patch.object(facts, "save_facts", side_effect=_flaky_save_facts):
        reply = _command_reply("/forget", cid)
    assert_true(
        "could not remove" not in reply.lower(),
        f"[D] *** BUG CHECK: the retry succeeded — must not still claim a "
        f"removal failure that already healed: {reply!r}",
    )
    assert_true(
        "still storing" not in reply.lower(),
        f"[D] *** BUG CHECK: the retry succeeded — must not still claim "
        f"facts are still stored: {reply!r}",
    )
    assert_true(
        "fact" in reply.lower() and "forgot" in reply.lower(),
        f"[D] the eventual success is reported as an ordinary forget: {reply!r}",
    )
    assert_eq(facts.load_facts(cid), [], "[D] facts are actually empty on disk")

    # --- E: _clear_all_memory's own write fails, but the SAME pass's
    #        empty-facts tombstone (moments later, same _wipe_all_layers
    #        call) succeeds — no second full pass involved at all. -------
    _wipe_storage()
    cid = "table-write-heals-within-one-pass"
    _seed(cid)
    _real_save_facts_e = facts.save_facts
    _calls_e = {"n": 0}

    def _heals_on_second_call(conv_id_arg, value):
        _calls_e["n"] += 1
        if _calls_e["n"] == 1:
            raise write_err
        return _real_save_facts_e(conv_id_arg, value)

    with patch.object(facts, "save_facts", side_effect=_heals_on_second_call):
        reply = _command_reply("/forget", cid)
    assert_true(
        "could not remove" not in reply.lower(),
        f"[E] *** BUG CHECK: _clear_all_memory's own write failed, but the "
        f"SAME pass's empty-facts tombstone write moments later actually "
        f"landed — write_failed must reflect the CURRENT disk truth, not "
        f"the stale first attempt: {reply!r}",
    )
    assert_eq(facts.load_facts(cid), [], "[E] facts are actually empty on disk")


def test_facts_write_failure_alerts_async_once_per_conv():
    print("\n[test] a facts-wipe write failure alerts via notify_async "
          "(never the sync notify — this call site holds conv_lock on the "
          "event loop), exactly once per conversation")
    # v3.1.9.8 round-2 review (item 5, LOW): the original version of this
    # fix called the SYNC `alert.notify` (a blocking httpx.post, up to
    # COMPACTOR_ALERT_TIMEOUT_S) from inside `async def _clear_all_memory`
    # while holding conv_lock — blocking the event loop, and every OTHER
    # conversation's request behind it, for the duration of the HTTP call.
    # Fixed to `await alert.notify_async(...)`, gated on the conv_id being
    # NEWLY added to `_facts_wipe_write_failed_ever` (a still-failing write
    # retried by _wipe_all_layers must not alert twice for the same fault).
    _wipe_storage()
    cid = "facts-write-alert-async"
    facts.save_facts(cid, [{"text": "x", "added_turn": 1, "last_used": 1}])
    import alert
    with patch.object(alert, "notify_async", new_callable=AsyncMock) as mock_async, \
         patch.object(alert, "notify") as mock_sync, \
         patch.object(facts, "save_facts",
                       side_effect=OSError(28, "No space left on device")):
        # Two wipes against the SAME still-broken conv_id, the way
        # _wipe_all_layers' own bounded retry does within one /forget.
        _wipe(cid)
        _wipe(cid)

    assert_true(not mock_sync.called,
                f"*** BUG CHECK: the blocking sync alert.notify must never "
                f"be called from this async, lock-holding call site: "
                f"{mock_sync.call_args_list!r}")
    assert_eq(mock_async.call_count, 1,
              f"*** BUG CHECK: alert.notify_async must fire exactly once "
              f"for this conv_id, not once per wipe pass: "
              f"{mock_async.call_args_list!r}")
    call_args = mock_async.call_args
    args = call_args.args
    kwargs = call_args.kwargs
    assert_eq(args[0] if args else kwargs.get("service"), "compactor",
              "alert.notify_async's service arg")
    detail = args[2] if len(args) > 2 else kwargs.get("detail", "")
    assert_true("facts-wipe write failed" in detail,
                f"alert detail names the failure: {call_args!r}")
    assert_true(cid in detail, f"alert detail names the conversation: {call_args!r}")


def test_health_full_counts_facts_write_failures():
    print("\n[test] /health/full reports the facts-wipe write-failure count "
          "next to the ledger counts")
    # v3.1.9.8 round-2 review (item 1, MED): `_facts_wipe_write_failed_ever`
    # is a process-wide set that accumulates across every test in this
    # file's run (it is never reset — same as the ledger counters
    # `checks.ledger` already follows this doctrine for). Asserting
    # `>= 1` passes even if THIS test's own failure never actually landed
    # in the set (an earlier test's failure alone would satisfy it), so
    # the BEFORE/AFTER delta is what actually proves this failure was
    # counted.
    r_before = client.get("/health/full")
    assert_eq(r_before.status_code, 200, "200 (before)")
    before = (r_before.json().get("checks", {}).get("facts_write") or {}).get(
        "wipe_write_failed_ever", 0)

    _wipe_storage()
    cid = "facts-write-health"
    facts.save_facts(cid, [{"text": "x", "added_turn": 1, "last_used": 1}])
    with patch.object(facts, "save_facts",
                       side_effect=OSError(28, "No space left on device")):
        _wipe(cid)

    r = client.get("/health/full")
    assert_eq(r.status_code, 200, "200")
    body = r.json()
    fw = body.get("checks", {}).get("facts_write")
    assert_true(fw is not None, f"checks.facts_write is present: {body.get('checks', {}).keys()}")
    assert_true(fw.get("available"), f"*** BUG CHECK: facts_write check not available: {fw}")
    after = fw.get("wipe_write_failed_ever", 0)
    assert_eq(
        after, before + 1,
        f"*** BUG CHECK: the count did not move by exactly 1 for THIS "
        f"conversation's failure (before={before}, after={after})",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    # "does not raise" runs first deliberately: if the wipe blows up, that is
    # the one fact worth reading, and every assertion after it is downstream
    # noise. The runner is fail-fast, so order is the diagnosis.
    return [
        test_clear_all_memory_does_not_raise_on_corrupt_facts,
        test_clear_all_memory_partial_wipe_on_corrupt_facts,
        test_clear_all_memory_partial_wipe_on_facts_write_failure,
        test_clear_all_memory_healthy_reports_nothing_unreadable,
        test_clear_all_memory_absent_facts_is_not_unreadable,
        test_admin_forget_endpoint_reports_unreadable,
        test_forget_command_must_not_report_a_clean_wipe,
        test_real_forget_clears_every_layer_end_to_end,
        test_real_forget_closes_the_backfill_gate,
        test_selective_forget_removes_facts_even_when_the_ledger_write_fails,
        test_full_forget_removes_facts_even_when_the_ledger_write_fails,
        test_ledger_write_failure_does_not_bump_the_epoch,
        test_full_wipe_still_deletes_her_data_under_enospc_on_every_bookkeeping_write,
        test_selective_forget_still_deletes_her_data_under_enospc_on_every_bookkeeping_write,
        test_command_reply_hides_path_when_persona_unreadable,
        test_command_reply_streams_the_same_plain_language,
        test_command_reply_hides_path_when_facts_unreadable,
        test_command_reply_unaffected_when_store_is_healthy,
        test_forget_command_write_failure_says_could_not_remove_not_read,
        test_forget_reply_truthful_across_read_write_failure_combinations,
        test_facts_write_failure_alerts_async_once_per_conv,
        test_health_full_counts_facts_write_failures,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll degraded-forget tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
