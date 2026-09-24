"""
CPU-only Tier-1 tests for compactor.portability.

Round-trips export → import → export and verifies the second export
equals the first (modulo timestamp). Uses an isolated tmpdir as
COMPACTOR_STORAGE_ROOT and a stubbed retrieval module — no ChromaDB,
no embeddings.

Run: python test_portability.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile

# Isolate storage to a tmpdir BEFORE importing memory.
_TMP_ROOT = tempfile.mkdtemp(prefix="zions_portability_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # we'll stub retrieval

import facts  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

# Stub retrieval's ChromaDB integration with a pure-Python dict so tests
# don't need fastembed/chromadb available. Each test resets the stub.
_STUB_STORE: dict[str, list[dict]] = {}


def _stub_export(conv_id):
    return list(_STUB_STORE.get(conv_id, []))


def _stub_import(conv_id, turn_index, document):
    _STUB_STORE.setdefault(conv_id, []).append(
        {"turn_index": turn_index, "document": document}
    )
    return True


def _stub_count(conv_id):
    return len(_STUB_STORE.get(conv_id, []))


def _stub_forget(conv_id):
    n = len(_STUB_STORE.get(conv_id, []))
    _STUB_STORE.pop(conv_id, None)
    return n


retrieval.export_indexed_exchanges = _stub_export
retrieval.import_indexed_exchange = _stub_import
retrieval.conversation_doc_count = _stub_count
retrieval.forget_conversation = _stub_forget

import portability  # noqa: E402 — must import after stubs are wired


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

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


def assert_raises(fn, exc_type, label):
    try:
        fn()
    except exc_type:
        print(f"  ok   {label}")
        return
    except Exception as e:
        print(f"FAIL {label}: expected {exc_type.__name__}, got {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"FAIL {label}: expected {exc_type.__name__}, nothing raised")
    sys.exit(1)


def reset_state(conv_id):
    """Wipe any state for a conv between tests."""
    facts.save_facts(conv_id, [])
    summarizer.save_state(conv_id, {})
    _STUB_STORE.pop(conv_id, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_export_empty_conv():
    print("\n[test] export of empty conv produces full-shape bundle")
    memory.ensure_storage_layout()
    reset_state("empty-conv")
    b = portability.export_conversation("empty-conv")
    assert_eq(b["version"], "v2.1", "version")
    assert_eq(b["source_conv_id"], "empty-conv", "source_conv_id")
    assert_eq(b["facts"], [], "facts list empty")
    assert_eq(b["episodic"], [], "episodic list empty")
    assert_true(isinstance(b["summary_state"], dict), "summary_state is dict")
    assert_true(isinstance(b["exported_at"], int), "exported_at is int")


def test_export_populated_conv():
    print("\n[test] export captures facts + episodic + summary")
    memory.ensure_storage_layout()
    reset_state("conv-A")
    facts.save_facts(
        "conv-A",
        [
            {"text": "Lyra is half-elf", "added_turn": 0, "last_used": 100},
            {"text": "Setting: Aethermere", "added_turn": 1, "last_used": 200},
        ],
    )
    chunk1 = {"text": "chunk one summary", "first_turn": 0, "last_turn": 9}
    chunk2 = {"text": "chunk two summary", "first_turn": 10, "last_turn": 19}
    summarizer.save_state("conv-A", {"l1": [chunk1, chunk2], "l2": [], "l3": None})
    _stub_import("conv-A", 2, "[user]: hi\n[assistant]: hello")
    _stub_import("conv-A", 4, "[user]: bye\n[assistant]: goodbye")

    b = portability.export_conversation("conv-A")
    assert_eq(len(b["facts"]), 2, "2 facts exported")
    assert_eq(len(b["episodic"]), 2, "2 episodic entries exported")
    assert_eq(b["episodic"][0]["turn_index"], 2, "first episodic turn=2 (sorted)")
    assert_eq(len(b["summary_state"]["l1"]), 2, "summary l1 has 2 chunks")
    assert_eq(b["summary_state"]["l1"][0]["text"], "chunk one summary", "chunk1 text preserved")


def test_import_round_trip_to_new_conv():
    print("\n[test] import to a fresh conv_id round-trips state cleanly")
    memory.ensure_storage_layout()
    reset_state("src-conv")
    reset_state("dst-conv")
    facts.save_facts(
        "src-conv",
        [{"text": "fact one", "added_turn": 0, "last_used": 100}],
    )
    summarizer.save_state(
        "src-conv",
        {
            "l1": [{"text": "a chunk", "first_turn": 0, "last_turn": 9}],
            "l2": [],
            "l3": None,
        },
    )
    _stub_import("src-conv", 1, "[user]: q\n[assistant]: a")

    bundle = portability.export_conversation("src-conv")
    result = portability.import_conversation(
        bundle, target_conv_id="dst-conv", overwrite=False
    )
    assert_eq(result["conv_id"], "dst-conv", "result conv_id")
    assert_eq(result["imported"]["facts"], 1, "imported 1 fact")
    assert_eq(result["imported"]["episodic"], 1, "imported 1 episodic")
    assert_true(result["imported"]["summary"], "imported summary")
    assert_eq(result["overwrote_existing"], False, "no overwrite on fresh target")

    # Verify destination state matches source
    assert_eq(len(facts.load_facts("dst-conv")), 1, "dst has 1 fact")
    assert_eq(_stub_count("dst-conv"), 1, "dst has 1 episodic")
    dst_l1 = summarizer.load_state("dst-conv")["l1"]
    assert_eq(len(dst_l1), 1, "dst summary l1 has 1 chunk")
    assert_eq(dst_l1[0]["text"], "a chunk", "dst summary l1 chunk text matches")


def test_import_refuses_overwrite_without_flag():
    print("\n[test] import refuses to clobber existing state by default")
    memory.ensure_storage_layout()
    reset_state("existing-conv")
    facts.save_facts(
        "existing-conv",
        [{"text": "do not lose me", "added_turn": 0, "last_used": 100}],
    )
    bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "other",
        "facts": [{"text": "replacement", "added_turn": 0, "last_used": 0}],
        "summary_state": {},
        "episodic": [],
    }
    assert_raises(
        lambda: portability.import_conversation(
            bundle, target_conv_id="existing-conv", overwrite=False
        ),
        portability.ImportError_,
        "import refuses without overwrite=True",
    )
    # And verifies it didn't already touch state
    f = facts.load_facts("existing-conv")
    assert_eq(f[0]["text"], "do not lose me", "existing facts intact after refusal")


def test_import_overwrite_replaces_wholesale():
    print("\n[test] import with overwrite=True replaces existing state")
    memory.ensure_storage_layout()
    reset_state("clobber-conv")
    facts.save_facts(
        "clobber-conv",
        [{"text": "old", "added_turn": 0, "last_used": 100}],
    )
    _stub_import("clobber-conv", 0, "old episodic")
    bundle = {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "other",
        "facts": [{"text": "new", "added_turn": 5, "last_used": 200}],
        "summary_state": {"l1": ["new chunk"], "l2": [], "l3": None},
        "episodic": [{"turn_index": 9, "document": "new episodic"}],
    }
    result = portability.import_conversation(
        bundle, target_conv_id="clobber-conv", overwrite=True
    )
    assert_eq(result["overwrote_existing"], True, "overwrote_existing flag")
    f = facts.load_facts("clobber-conv")
    assert_eq(f[0]["text"], "new", "facts replaced")
    assert_eq(_stub_count("clobber-conv"), 1, "episodic replaced (count=1)")
    assert_eq(_STUB_STORE["clobber-conv"][0]["document"], "new episodic", "episodic content")


def test_import_rejects_wrong_version():
    print("\n[test] import rejects bundles with mismatched version")
    bundle = {"version": "v9.9", "facts": [], "summary_state": {}, "episodic": []}
    assert_raises(
        lambda: portability.import_conversation(bundle, target_conv_id="x"),
        portability.ImportError_,
        "wrong version rejected",
    )


def test_import_rejects_missing_keys():
    print("\n[test] import rejects bundles missing required keys")
    bundle = {"version": "v2.1", "facts": []}  # missing summary_state + episodic
    assert_raises(
        lambda: portability.import_conversation(bundle, target_conv_id="x"),
        portability.ImportError_,
        "missing keys rejected",
    )


def test_import_rejects_non_dict():
    print("\n[test] import rejects non-dict bundle")
    assert_raises(
        lambda: portability.import_conversation("not a dict", target_conv_id="x"),
        portability.ImportError_,
        "string bundle rejected",
    )


def test_fork_creates_independent_copy():
    print("\n[test] fork clones state into a new conv_id with default suffix")
    memory.ensure_storage_layout()
    reset_state("parent-conv")
    facts.save_facts(
        "parent-conv",
        [{"text": "shared truth", "added_turn": 0, "last_used": 100}],
    )
    result = portability.fork_conversation("parent-conv")
    new_id = result["conv_id"]
    assert_true(new_id.startswith("parent-conv__fork_"), "new id has fork prefix")
    assert_eq(result["forked_from"], "parent-conv", "forked_from tag set")
    assert_eq(len(facts.load_facts(new_id)), 1, "fork has parent's facts")
    # Mutate the fork — parent should be untouched
    facts.save_facts(new_id, [])
    assert_eq(len(facts.load_facts("parent-conv")), 1, "parent untouched after fork mutation")


def test_fork_with_explicit_new_id():
    print("\n[test] fork with explicit new_conv_id uses that id")
    memory.ensure_storage_layout()
    reset_state("parent-2")
    reset_state("custom-fork")
    facts.save_facts(
        "parent-2",
        [{"text": "x", "added_turn": 0, "last_used": 0}],
    )
    result = portability.fork_conversation("parent-2", new_conv_id="custom-fork")
    assert_eq(result["conv_id"], "custom-fork", "uses custom id")


def test_export_is_json_serializable():
    print("\n[test] export output round-trips through json.dumps")
    memory.ensure_storage_layout()
    reset_state("json-conv")
    facts.save_facts(
        "json-conv",
        [{"text": "x", "added_turn": 0, "last_used": 0}],
    )
    bundle = portability.export_conversation("json-conv")
    s = json.dumps(bundle)
    again = json.loads(s)
    assert_eq(again["source_conv_id"], "json-conv", "round-trip preserves id")


# ---------------------------------------------------------------------------
# hostile pass 3, F9: export_conversation's best-effort facts/summary read
# used to make merge and fork treat an UNREADABLE source as "0 facts" —
# silently, no error — instead of "unknown is not empty", the doctrine
# import's pre-flight / health / quarantine already follow.
# ---------------------------------------------------------------------------

def _corrupt_facts_file(conv_id):
    """A torn write: bytes on disk that are not valid JSON at all. Triggers
    memory.StoreUnreadable from facts.load_facts, not a missing-file []."""
    memory.facts_path(conv_id).write_bytes(b"{not valid json")


def test_export_default_is_best_effort_on_a_corrupt_facts_file():
    print("\n[test] F9 CONTROL: export_conversation(strict=False, the "
          "default) still degrades gracefully on a corrupt facts file — "
          "GET /admin/.../export must not start 400ing on a read a human "
          "was just inspecting")
    memory.ensure_storage_layout()
    reset_state("f9-corrupt-1")
    _corrupt_facts_file("f9-corrupt-1")
    b = portability.export_conversation("f9-corrupt-1")
    assert_eq(b["facts"], [], "best-effort default still returns an empty list, not raise")


def test_export_strict_raises_on_unreadable_facts():
    print("\n[test] F9: export_conversation(strict=True) raises "
          "memory.StoreUnreadable on a corrupt facts file instead of "
          "returning an empty list")
    memory.ensure_storage_layout()
    reset_state("f9-corrupt-2")
    _corrupt_facts_file("f9-corrupt-2")
    assert_raises(
        lambda: portability.export_conversation("f9-corrupt-2", strict=True),
        memory.StoreUnreadable,
        "strict export raises rather than silently reporting 0 facts",
    )


def test_merge_refuses_on_unreadable_source_facts():
    print("\n[test] F9: merge_conversation refuses (ValueError, 400-mapped) "
          "rather than reporting src_facts=0 when the source facts file "
          "cannot be read")
    memory.ensure_storage_layout()
    reset_state("f9-merge-src")
    reset_state("f9-merge-dst")
    _corrupt_facts_file("f9-merge-src")
    # Give the source SOME episodic content too, so the pre-fix code's
    # "nothing to merge" early-exit (empty facts AND empty episodic) would
    # not have masked this — the corrupted facts file is the only reason
    # this must refuse.
    _STUB_STORE["f9-merge-src"] = [{"turn_index": 1, "document": "[user]: q\n[assistant]: a"}]
    assert_raises(
        lambda: portability.merge_conversation("f9-merge-src", "f9-merge-dst", dry_run=True),
        ValueError,
        "merge refuses on an unreadable source rather than treating it as "
        "0 facts and proceeding on episodic alone",
    )


def test_fork_refuses_on_unreadable_source_facts():
    print("\n[test] F9: fork_conversation refuses (ImportError_, 400-mapped) "
          "rather than writing an empty-facts tombstone for the fork")
    memory.ensure_storage_layout()
    reset_state("f9-fork-src")
    _corrupt_facts_file("f9-fork-src")
    assert_raises(
        lambda: portability.fork_conversation("f9-fork-src"),
        portability.ImportError_,
        "fork refuses on an unreadable source rather than writing an "
        "empty-facts fork (which would also block needs_backfill later)",
    )


# ---------------------------------------------------------------------------
# merge_conversation — R5 (v3.1.7): pin/last_used must survive a collision,
# not be silently discarded because dst's copy "wins".
# ---------------------------------------------------------------------------

def test_merge_conversation_adds_new_facts_and_leaves_source_intact():
    print("\n[test] merge_conversation unions non-colliding facts, src untouched")
    memory.ensure_storage_layout()
    reset_state("merge-src-1")
    reset_state("merge-dst-1")
    facts.save_facts(
        "merge-src-1",
        [{"text": "The lighthouse has a red door.", "added_turn": 5, "last_used": 300}],
    )
    facts.save_facts(
        "merge-dst-1",
        [{"text": "The user prefers past tense.", "added_turn": 1, "last_used": 100}],
    )
    result = portability.merge_conversation("merge-src-1", "merge-dst-1", dry_run=False)
    assert_eq(result["facts_added"], 1, "one new fact added")
    dst_texts = {f["text"] for f in facts.load_facts("merge-dst-1")}
    assert_true("The lighthouse has a red door." in dst_texts, "new fact landed in dst")
    assert_true("The user prefers past tense." in dst_texts, "dst's own fact still there")
    assert_eq(len(facts.load_facts("merge-src-1")), 1, "source untouched by the merge")


def test_merge_conversation_pins_the_destination_copy_on_collision():
    print("\n[test] R5: a pinned source fact merging into an unpinned dst copy "
          "comes out pinned, not silently un-pinned")
    memory.ensure_storage_layout()
    reset_state("merge-src-2")
    reset_state("merge-dst-2")
    # _fact_key casefolds and collapses whitespace, so this is deliberately
    # NOT byte-identical text — the backlog is explicit that R5 fires on
    # non-identical pairs too.
    facts.save_facts(
        "merge-src-2",
        [{"text": "Her name is Elena and she goes by El.",
          "added_turn": 3, "last_used": 500, "pin": True}],
    )
    facts.save_facts(
        "merge-dst-2",
        [{"text": "her name is elena and she goes by el.",
          "added_turn": 40, "last_used": 900, "pin": False}],
    )
    result = portability.merge_conversation("merge-src-2", "merge-dst-2", dry_run=False)
    after = facts.load_facts("merge-dst-2")
    assert_eq(len(after), 1, "still exactly one row, not two")
    # The core R5 assertion, checked before anything about the reporting
    # fields below — this is the one that must fail loudly on old code, not
    # as a side effect of a KeyError on a field the fix happens to add.
    assert_true(after[0]["pin"], "destination's copy is now pinned")
    assert_eq(result["facts_added"], 0, "no NEW row — the key already existed")
    assert_eq(result.get("facts_pin_or_recency_updated"), 1, "the collision is reported")
    # last_used: the fresher of the two (dst's 900 already beat src's 500).
    assert_eq(after[0]["last_used"], 900, "last_used is the max of the two")
    # added_turn: dst's own value survives untouched — merging two different
    # conversations' turn numbering is not meaningful (see
    # _merge_fact_pin_and_recency's docstring).
    assert_eq(after[0]["added_turn"], 40, "added_turn is dst's own, not touched")
    # dst's own wording survives — the two texts differed only in case/full
    # stop, and the key match does not mean byte-identical text.
    assert_eq(after[0]["text"], "her name is elena and she goes by el.",
              "dst's own wording is kept")


def test_merge_conversation_last_used_takes_the_max_either_direction():
    print("\n[test] merge collision keeps the fresher last_used, whichever side it's on")
    memory.ensure_storage_layout()
    reset_state("merge-src-3")
    reset_state("merge-dst-3")
    facts.save_facts(
        "merge-src-3",
        [{"text": "The story is set on Brannock.", "added_turn": 1, "last_used": 9999}],
    )
    facts.save_facts(
        "merge-dst-3",
        [{"text": "The story is set on Brannock.", "added_turn": 1, "last_used": 100}],
    )
    portability.merge_conversation("merge-src-3", "merge-dst-3", dry_run=False)
    after = facts.load_facts("merge-dst-3")
    assert_eq(after[0]["last_used"], 9999, "src's fresher last_used wins even though src is not kept as the row")


def test_merge_conversation_dry_run_previews_the_pin_update_without_writing():
    print("\n[test] merge_conversation dry_run reports the pin update but changes nothing")
    memory.ensure_storage_layout()
    reset_state("merge-src-4")
    reset_state("merge-dst-4")
    facts.save_facts(
        "merge-src-4",
        [{"text": "Idris keeps a logbook.", "added_turn": 1, "last_used": 50, "pin": True}],
    )
    facts.save_facts(
        "merge-dst-4",
        [{"text": "Idris keeps a logbook.", "added_turn": 1, "last_used": 50, "pin": False}],
    )
    result = portability.merge_conversation("merge-src-4", "merge-dst-4", dry_run=True)
    assert_eq(result["dry_run"], True, "dry_run flag echoed")
    assert_eq(result.get("facts_pin_or_recency_to_update"), 1, "preview reports the pending pin update")
    after = facts.load_facts("merge-dst-4")
    assert_eq(after[0]["pin"], False, "dry run changed nothing on disk")


def test_merge_conversation_byte_identical_duplicate_is_a_true_no_op():
    print("\n[test] a genuinely identical pair (same pin, same or lower last_used) "
          "is reported as skipped, not as an update")
    memory.ensure_storage_layout()
    reset_state("merge-src-5")
    reset_state("merge-dst-5")
    facts.save_facts(
        "merge-src-5",
        [{"text": "Setting: Aethermere.", "added_turn": 9, "last_used": 100, "pin": True}],
    )
    facts.save_facts(
        "merge-dst-5",
        [{"text": "Setting: Aethermere.", "added_turn": 1, "last_used": 100, "pin": True}],
    )
    result = portability.merge_conversation("merge-src-5", "merge-dst-5", dry_run=False)
    assert_eq(result.get("facts_pin_or_recency_updated"), 0, "nothing actually changed")
    assert_eq(result["facts_skipped_duplicate"], 1, "counted as a true duplicate")


# ---------------------------------------------------------------------------
# hostile pass 3, reviewer D F5: merge_conversation guarded only the
# DESTINATION's lock, not the SOURCE's. On the identity runbook's own R3
# (reverse merge), src IS the new uuid, and R2's first message under it
# holds conv_lock(uuid) for the whole summary-rebuild drain (10-30 minutes
# per the runbook) while her NEXT messages queue their episodic/fact tails
# behind that same lock. A reverse merge run in that window reads an
# INCOMPLETE src snapshot, commits, and reports success — the queued tail's
# writes then land under the uuid after the header is already gone,
# stranded (recoverable by re-running the merge, but silently incomplete
# the first time).
# ---------------------------------------------------------------------------

def test_merge_refuses_while_source_has_a_write_in_flight():
    print("\n[test] hostile pass 3 (reviewer D) F5: merge_conversation "
          "refuses while the SOURCE (not just the destination) has a "
          "memory write in flight — the identity runbook's R3 (reverse "
          "merge), run while the uuid's own rebuild tail is still draining")
    memory.ensure_storage_layout()
    reset_state("f5d-src")
    reset_state("f5d-dst")
    facts.save_facts("f5d-src", [{"text": "queued fact", "added_turn": 1, "last_used": 1, "pin": False}])
    facts.save_facts("f5d-dst", [{"text": "dst fact", "added_turn": 1, "last_used": 1, "pin": False}])

    async def _while_src_locked():
        async with memory.conv_lock("f5d-src"):
            try:
                portability.merge_conversation("f5d-src", "f5d-dst", dry_run=False)
                return False
            except ValueError:
                return True

    assert_true(
        asyncio.run(_while_src_locked()),
        "refuses (ValueError, 400-mapped) while a tail holds conv_lock on "
        "the SOURCE — before this fix, only the destination's lock was "
        "checked and this merge would have proceeded, reading whatever the "
        "queued tail had NOT yet written",
    )
    # Nothing was written to dst — a mid-air refusal must not partially land.
    assert_eq(len(facts.load_facts("f5d-dst")), 1, "dst untouched by the refused attempt")


def test_merge_still_refuses_while_dest_has_a_write_in_flight():
    print("\n[test] CONTROL: the pre-existing destination-lock refusal is "
          "unchanged by adding the source check next to it")
    memory.ensure_storage_layout()
    reset_state("f5d-src2")
    reset_state("f5d-dst2")
    facts.save_facts("f5d-src2", [{"text": "src fact", "added_turn": 1, "last_used": 1, "pin": False}])
    facts.save_facts("f5d-dst2", [{"text": "dst fact", "added_turn": 1, "last_used": 1, "pin": False}])

    async def _while_dst_locked():
        async with memory.conv_lock("f5d-dst2"):
            try:
                portability.merge_conversation("f5d-src2", "f5d-dst2", dry_run=False)
                return False
            except ValueError:
                return True

    assert_true(asyncio.run(_while_dst_locked()), "CONTROL: dst-lock refusal still fires")


def test_merge_commits_normally_once_both_locks_are_free():
    print("\n[test] CONTROL: with neither lock held, the merge commits "
          "normally — this is a refusal on CONTENTION, not a new blanket "
          "refusal on every merge")
    memory.ensure_storage_layout()
    reset_state("f5d-src3")
    reset_state("f5d-dst3")
    facts.save_facts("f5d-src3", [{"text": "src fact", "added_turn": 1, "last_used": 1, "pin": False}])
    reset_state("f5d-dst3")
    result = portability.merge_conversation("f5d-src3", "f5d-dst3", dry_run=False)
    assert_eq(result["facts_added"], 1, "CONTROL: an uncontended merge still commits")


# ---------------------------------------------------------------------------
# merge_conversation LRU floor — v3.1.9 F6 (hostile pass 2, review B):
# merging the hash-id store into the new chat-id store used to keep each
# merged fact's OWN last_used. A backfill's fresh re-extractions under the
# new id then won facts.prune_facts's LRU on the very next write, and a
# reviewed real merge archived 115 of a user's 136 original facts.
#
# These tests use facts.prune_facts as the REAL evictor (not a hand-rolled
# stand-in), exactly the brief's instruction, so the LRU ordering exercised
# here is the one production actually runs.
# ---------------------------------------------------------------------------

def _bulk_facts(n, prefix, lu_base, lu_spread, turn_base=0):
    """n synthetic facts shaped like the reviewed real store: distinct text,
    a spread of last_used values, no pins. `lu_base`/`lu_spread` control
    where in "time" this batch sits relative to another batch — the whole
    point of these tests is comparing two batches with DIFFERENT last_used
    neighborhoods, the way "her hours-old originals" and "backfill's
    just-minted facts" sit in production.
    """
    out = []
    for i in range(n):
        out.append({
            "text": f"{prefix} fact {i}: a distinct detail long enough to "
                    f"cost real budget once rendered as a bullet line.",
            "added_turn": turn_base + i,
            "last_used": lu_base + (i * lu_spread // max(n, 1)),
            "pin": False,
        })
    return out


def test_merge_conversation_f6_new_facts_survive_lru_against_fresher_backfill():
    print("\n[test] F6: merged-in facts are not archived wholesale just because "
          "a same-moment backfill minted fresher last_used values in dst")
    memory.ensure_storage_layout()
    reset_state("merge-src-f6")
    reset_state("merge-dst-f6")
    # src: 40 facts with OLD last_used (~1000-2000) — "her originals",
    # hours old by the time the merge runs.
    facts.save_facts("merge-src-f6", _bulk_facts(40, "hers", lu_base=1000, lu_spread=1000))
    # dst: 60 facts with FRESH last_used (~99000-100000) — "backfill's",
    # minted moments before the merge. Calibrated (not arbitrary): 60 of
    # this exact bullet shape ALREADY slightly exceeds the default
    # COMPACTOR_MAX_FACTS_TOKENS budget on its own (measured: 60 alone ->
    # 57 kept, 3 dropped) — the same shape as the reviewed real store,
    # where backfill's own batch was already most of the way to the cap
    # before anything was merged in. This matters: with dst's own batch
    # already filling the budget, an unprotected merge leaves ZERO slack
    # for anything merged in, which is what makes this test actually
    # discriminate the fix from the bug (a smaller/looser dst batch leaves
    # enough spare capacity that some of "hers" survives by sheer room
    # either way, and the assertion below would pass for the wrong reason).
    facts.save_facts("merge-dst-f6", _bulk_facts(60, "backfill", lu_base=99000, lu_spread=1000))

    # v3.1.9 (hostile pass 3, F6): the floor is opt-in now (see
    # portability.merge_conversation's docstring) — this test exercises
    # exactly the id-migration/backfill shape it was built for, so it asks
    # for it explicitly.
    result = portability.merge_conversation(
        "merge-src-f6", "merge-dst-f6", dry_run=False, refresh_last_used=True
    )
    assert_eq(result["facts_added"], 40, "all 40 of her facts were new (no collisions)")

    merged = facts.load_facts("merge-dst-f6")
    assert_eq(len(merged), 100, "post-merge store holds both batches")

    # The default budget forces real eviction — the CONTROL half: this must
    # still be able to say yes, not just refuse to evict anything.
    kept, dropped = facts.prune_facts(merged, conv_id="merge-dst-f6")
    assert_true(dropped > 0, "CONTROL: eviction still happens under the real budget")
    assert_true(len(kept) < len(merged), "CONTROL: not everything survives")

    hers_kept = sum(1 for f in kept if f["text"].startswith("hers"))
    hers_archived = 40 - hers_kept
    # Before the fix this was 0/40 (every one of "hers" lost the LRU race to
    # "backfill" — measured directly against the unfixed code with this
    # exact fixture before the fix landed), matching the shape of the
    # reviewed incident (115/136 archived, a large majority). The floor
    # does not guarantee ALL of hers survive every possible store shape —
    # it stops hers being the ONE-SIDED loser of a race it was never a
    # real participant in, which this fixture is calibrated to show clearly:
    # fixed, hers_kept measures 40/40.
    assert_true(
        hers_kept >= 30,
        f"a floor-protected merge keeps the large majority of the merged-in "
        f"facts against a dst batch that was already at/over budget before "
        f"the merge (kept {hers_kept}/40, archived {hers_archived}/40)",
    )


def test_merge_conversation_f6_no_backfill_competition_all_of_hers_survive():
    print("\n[test] F6: with an EMPTY dst (the recommended merge-before-message "
          "order — no backfill has run), the floor is a no-op and normal "
          "budget math applies")
    memory.ensure_storage_layout()
    reset_state("merge-src-f6b")
    reset_state("merge-dst-f6b")
    facts.save_facts("merge-src-f6b", _bulk_facts(20, "hers", lu_base=1000, lu_spread=1000))
    # dst starts genuinely empty — no backfill has run, matching F3's fix
    # (merging first means the destination already has a facts file, so the
    # backfill trigger never fires).
    facts.save_facts("merge-dst-f6b", [])

    result = portability.merge_conversation("merge-src-f6b", "merge-dst-f6b", dry_run=False)
    assert_eq(result["facts_added"], 20, "all 20 landed, nothing to collide with")

    merged = facts.load_facts("merge-dst-f6b")
    # Budget comfortably fits all 20 — nothing should be evicted, floor or
    # not, proving the floor logic does not invent phantom pressure when
    # there was none.
    kept, dropped = facts.prune_facts(merged, conv_id="merge-dst-f6b")
    assert_eq(dropped, 0, "nothing evicted when the store fits the default budget")
    assert_eq(len(kept), 20, "all of hers present")


def test_merge_conversation_f6_collision_last_used_is_not_floored():
    print("\n[test] F6's floor applies ONLY to brand-new facts, never to a "
          "collision fold — a colliding row keeps max(dst, src), not the "
          "destination's unrelated newest fact")
    memory.ensure_storage_layout()
    reset_state("merge-src-f6c")
    reset_state("merge-dst-f6c")
    # Two facts in dst: one with a very high last_used (sets the floor a
    # naive implementation might apply everywhere), and one that will
    # COLLIDE with src's fact at a much lower last_used than the floor.
    facts.save_facts(
        "merge-dst-f6c",
        [
            {"text": "Unrelated fresh fact.", "added_turn": 0, "last_used": 90000, "pin": False},
            {"text": "The story is set on Brannock.", "added_turn": 1, "last_used": 200, "pin": False},
        ],
    )
    facts.save_facts(
        "merge-src-f6c",
        [{"text": "The story is set on Brannock.", "added_turn": 1, "last_used": 300, "pin": False}],
    )
    portability.merge_conversation("merge-src-f6c", "merge-dst-f6c", dry_run=False)
    after = {f["text"]: f for f in facts.load_facts("merge-dst-f6c")}
    assert_eq(len(after), 2, "still two rows — the collision did not add a third")
    assert_eq(
        after["The story is set on Brannock."]["last_used"], 300,
        "collision keeps max(dst=200, src=300) — NOT floored up to the "
        "unrelated fact's 90000",
    )
    assert_eq(
        after["Unrelated fresh fact."]["last_used"], 90000,
        "the untouched dst fact is exactly untouched",
    )


def test_merge_conversation_f6_reports_facts_over_budget_after_merge():
    print("\n[test] F6: the merge response says how many facts are already "
          "over budget right after it lands, in both dry-run and commit")
    memory.ensure_storage_layout()
    reset_state("merge-src-f6d")
    reset_state("merge-dst-f6d")
    facts.save_facts("merge-src-f6d", _bulk_facts(40, "hers", lu_base=1000, lu_spread=1000))
    facts.save_facts("merge-dst-f6d", _bulk_facts(25, "backfill", lu_base=99000, lu_spread=1000))

    # Same numbers dry or committed — the preview must not undercount.
    dry = portability.merge_conversation("merge-src-f6d", "merge-dst-f6d", dry_run=True)
    assert_true("facts_over_budget_after_merge" in dry, "dry-run response carries the field")
    assert_true(dry["facts_over_budget_after_merge"] > 0,
                f"dry-run: the merged 40+25 facts over the "
                f"{facts._MAX_FACTS_TOKENS}-token "
                f"default budget is reported as already over "
                f"(got {dry['facts_over_budget_after_merge']})")

    live = portability.merge_conversation("merge-src-f6d", "merge-dst-f6d", dry_run=False)
    assert_eq(dry["facts_over_budget_after_merge"], live["facts_over_budget_after_merge"],
               "dry-run preview matches the committed reality (same inputs, "
               "same LRU split)")


# ---------------------------------------------------------------------------
# hostile pass 3, F6: the SAME automatic floor that fixes the backfill shape
# above INVERTS eviction for RUNBOOK_MEMORY_IDENTITY.md's "Older forks" step
# — merging an abandoned fork into a primary conversation still being
# chatted in, no backfill involved. These tests use facts.prune_facts as the
# real evictor, same as the F6 tests above.
# ---------------------------------------------------------------------------

def test_merge_conversation_f6c_default_does_not_invert_older_forks_eviction():
    print("\n[test] F6 (pass 3): the default (no refresh_last_used) merge does "
          "NOT float 14-day-old fork facts above her own real, more-recent "
          "ones — the runbook's own 'expect most of what they add to be "
          "evicted' promise")
    memory.ensure_storage_layout()
    reset_state("merge-src-f6e")
    reset_state("merge-dst-f6e")
    # dst ("her" primary, still being chatted in): 70 facts spread 1-4 days
    # old, matching the runbook shape (a real working set, not a backfill).
    facts.save_facts("merge-dst-f6e", _bulk_facts(
        70, "hers", lu_base=1_000_000, lu_spread=300_000,  # ~ "1-4 days ago" scale
    ))
    # src (the older, abandoned fork): 50 facts, all OLDER than any of dst's
    # — 13-14 days old, exactly what the runbook tells the operator to
    # expect gets evicted first.
    facts.save_facts("merge-src-f6e", _bulk_facts(
        50, "fork", lu_base=100_000, lu_spread=50_000,  # older than dst's 1,000,000+ range
    ))

    # CONTROL: without the fix, an unconditional floor would stamp every one
    # of the 50 fork facts up to dst's own newest last_used, inverting whose
    # facts look oldest. Confirm the default call does not ask for that.
    result = portability.merge_conversation("merge-src-f6e", "merge-dst-f6e", dry_run=False)
    assert_eq(result["refresh_last_used"], False, "opt-in flag defaults off")

    merged = facts.load_facts("merge-dst-f6e")
    assert_eq(len(merged), 120, "both batches present")

    kept, dropped = facts.prune_facts(merged, conv_id="merge-dst-f6e")
    assert_true(dropped > 0, "CONTROL: eviction still happens under the real budget")

    fork_kept = sum(1 for f in kept if f["text"].startswith("fork"))
    hers_kept = sum(1 for f in kept if f["text"].startswith("hers"))
    # Before this fix (floor unconditional): fork facts got floored to "now"
    # and out-ranked hers, so hers_kept was the one that collapsed. Fixed:
    # the fork's genuinely older facts are the ones the real LRU sheds first.
    assert_true(
        fork_kept < 50,
        f"the 14-day-old fork facts are evicted first under real LRU "
        f"(fork_kept={fork_kept}/50)",
    )
    assert_true(
        hers_kept > fork_kept,
        f"her more-recent facts outlive the fork's older ones "
        f"(hers_kept={hers_kept}/70, fork_kept={fork_kept}/50) — inverted "
        f"before this fix",
    )


def test_merge_conversation_f6c_refresh_last_used_opt_in_still_floors():
    print("\n[test] F6 (pass 3) CONTROL: refresh_last_used=True still floors "
          "merged-in facts for the id-migration/backfill recovery it exists for")
    memory.ensure_storage_layout()
    reset_state("merge-src-f6f")
    reset_state("merge-dst-f6f")
    facts.save_facts("merge-src-f6f", _bulk_facts(40, "hers", lu_base=1000, lu_spread=1000))
    facts.save_facts("merge-dst-f6f", _bulk_facts(60, "backfill", lu_base=99000, lu_spread=1000))

    result = portability.merge_conversation(
        "merge-src-f6f", "merge-dst-f6f", dry_run=False, refresh_last_used=True
    )
    assert_eq(result["refresh_last_used"], True, "the opt-in flag round-trips in the response")

    merged = facts.load_facts("merge-dst-f6f")
    kept, dropped = facts.prune_facts(merged, conv_id="merge-dst-f6f")
    assert_true(dropped > 0, "CONTROL: eviction still happens")
    hers_kept = sum(1 for f in kept if f["text"].startswith("hers"))
    assert_true(
        hers_kept >= 30,
        f"opt-in still protects the merged-in originals against a fresher "
        f"backfill batch (hers_kept={hers_kept}/40)",
    )


def test_merge_conversation_f6c_reports_eviction_by_origin():
    print("\n[test] F6 (pass 3): facts_over_budget_after_merge is split by "
          "origin so the operator can see WHICH side would be archived")
    memory.ensure_storage_layout()
    reset_state("merge-src-f6g")
    reset_state("merge-dst-f6g")
    facts.save_facts("merge-dst-f6g", _bulk_facts(70, "hers", lu_base=1_000_000, lu_spread=300_000))
    facts.save_facts("merge-src-f6g", _bulk_facts(50, "fork", lu_base=100_000, lu_spread=50_000))

    dry = portability.merge_conversation("merge-src-f6g", "merge-dst-f6g", dry_run=True)
    for key in (
        "facts_over_budget_after_merge",
        "dst_facts_evicted_after_merge",
        "merged_facts_evicted_after_merge",
    ):
        assert_true(key in dry, f"dry-run response carries {key}")
    assert_eq(
        dry["dst_facts_evicted_after_merge"] + dry["merged_facts_evicted_after_merge"],
        dry["facts_over_budget_after_merge"],
        "the per-origin split sums back to the total",
    )
    # The whole point of the finding: under the default (no floor), the
    # evicted set should be overwhelmingly the OLD FORK facts, not hers.
    assert_true(
        dry["merged_facts_evicted_after_merge"] > dry["dst_facts_evicted_after_merge"],
        f"the fork's older facts dominate the eviction, not her own "
        f"(dst={dry['dst_facts_evicted_after_merge']}, "
        f"merged={dry['merged_facts_evicted_after_merge']})",
    )

    live = portability.merge_conversation("merge-src-f6g", "merge-dst-f6g", dry_run=False)
    assert_eq(
        dry["dst_facts_evicted_after_merge"], live["dst_facts_evicted_after_merge"],
        "dry-run per-origin preview matches the committed reality",
    )
    assert_eq(
        dry["merged_facts_evicted_after_merge"], live["merged_facts_evicted_after_merge"],
        "dry-run per-origin preview matches the committed reality",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_export_empty_conv,
        test_export_populated_conv,
        test_import_round_trip_to_new_conv,
        test_import_refuses_overwrite_without_flag,
        test_import_overwrite_replaces_wholesale,
        test_import_rejects_wrong_version,
        test_import_rejects_missing_keys,
        test_import_rejects_non_dict,
        test_fork_creates_independent_copy,
        test_fork_with_explicit_new_id,
        test_export_is_json_serializable,
        test_export_default_is_best_effort_on_a_corrupt_facts_file,
        test_export_strict_raises_on_unreadable_facts,
        test_merge_refuses_on_unreadable_source_facts,
        test_fork_refuses_on_unreadable_source_facts,
        test_merge_conversation_adds_new_facts_and_leaves_source_intact,
        test_merge_conversation_pins_the_destination_copy_on_collision,
        test_merge_conversation_last_used_takes_the_max_either_direction,
        test_merge_conversation_dry_run_previews_the_pin_update_without_writing,
        test_merge_conversation_byte_identical_duplicate_is_a_true_no_op,
        test_merge_refuses_while_source_has_a_write_in_flight,
        test_merge_still_refuses_while_dest_has_a_write_in_flight,
        test_merge_commits_normally_once_both_locks_are_free,
        test_merge_conversation_f6_new_facts_survive_lru_against_fresher_backfill,
        test_merge_conversation_f6_no_backfill_competition_all_of_hers_survive,
        test_merge_conversation_f6_collision_last_used_is_not_floored,
        test_merge_conversation_f6_reports_facts_over_budget_after_merge,
        test_merge_conversation_f6c_default_does_not_invert_older_forks_eviction,
        test_merge_conversation_f6c_refresh_last_used_opt_in_still_floors,
        test_merge_conversation_f6c_reports_eviction_by_origin,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll portability smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
