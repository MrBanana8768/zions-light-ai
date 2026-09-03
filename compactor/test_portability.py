"""
CPU-only Tier-1 tests for compactor.portability.

Round-trips export → import → export and verifies the second export
equals the first (modulo timestamp). Uses an isolated tmpdir as
COMPACTOR_STORAGE_ROOT and a stubbed retrieval module — no ChromaDB,
no embeddings.

Run: python test_portability.py
"""

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
        test_merge_conversation_adds_new_facts_and_leaves_source_intact,
        test_merge_conversation_pins_the_destination_copy_on_collision,
        test_merge_conversation_last_used_takes_the_max_either_direction,
        test_merge_conversation_dry_run_previews_the_pin_update_without_writing,
        test_merge_conversation_byte_identical_duplicate_is_a_true_no_op,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll portability smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
