"""
N6 store pollution cleanup: portability.find_test_conversations /
cleanup_test_conversations.

THE FAILURE THIS GUARDS AGAINST. Production carries 129 "conversations" for
~26 real ones — CLONE_CONV_ID_HERE (a runbook placeholder pasted
unsubstituted), 17 __selftest_oneshot_* (one per boot before F23),
~75 itest-* (the Tier-3 integration harness). The one way this cleanup can
go wrong is catastrophic and asymmetric with "leaves some junk behind": a
false positive deletes a real conversation's memory. So this file spends
most of its weight on the matcher's near-misses and on proving a
substantial conversation is kept even when its id matches a pattern
exactly, before it ever gets to the happy path.

Three parts:
  [1] the matcher — realistic ids that must match, and near-misses that
      must NOT, run against the same patterns the real code uses
  [2] the substantial-content refusal — a pattern match alone is never
      sufficient
  [3] end-to-end dry-run vs live cleanup, including partial-failure
      handling (quarantine ok / wipe fails leaves the snapshot behind and
      reports it, never silently)

    python test_n6_cleanup.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_n6_cleanup_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # stub retrieval, no chromadb

import facts  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

# Stub retrieval's ChromaDB integration with a pure-Python dict — same
# approach test_portability.py uses, for the same reason (no fastembed/
# chromadb needed for this file's assertions).
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

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"FAIL   {label}")


def reset_state(conv_id):
    """Unlink every artifact file for conv_id — NOT save_facts(cid, []).

    An emptied-but-present facts file still counts as a known conv_id
    (memory.list_known_conv_ids globs facts/*.json), which is precisely
    the F23/D10 leak selftest._purge_conv_files exists to avoid. This test
    reuses many short conv_ids across sections, so a reset that left empty
    files behind would make an earlier section's ids resurface as
    zero-fact, still-itest-shaped candidates in a later section's scan.
    """
    for p in (
        memory.facts_path(conv_id),
        memory.facts_archive_path(conv_id),
        memory.summary_path(conv_id),
        memory.persona_path(conv_id),
    ):
        p.unlink(missing_ok=True)
    _STUB_STORE.pop(conv_id, None)


memory.ensure_storage_layout()

# ---------------------------------------------------------------------------
# [1] the matcher
# ---------------------------------------------------------------------------

print("[1] the matcher")

POSITIVES = [
    ("CLONE_CONV_ID_HERE", "the runbook placeholder itself"),
    ("__selftest_oneshot_1a2b3c4d__", "selftest.py's minted shape, 8 hex"),
    ("__selftest_oneshot_deadbeef__", "selftest.py's minted shape, all-letter hex"),
    ("itest-a1b2c3d4e5f6", "bare harness default, hex[:12]"),
    ("itest-archive-a1b2c3d4", "test_archive.py's archive fixture, hex[:8]"),
    ("itest-restore-filt-a1b2c3d4", "test_archive.py's filtered-restore fixture"),
    ("itest-dedup-distinct-a1b2c3d4", "test_dedup.py's distinct-facts fixture"),
    ("itest-persona-behave-a1b2c3d4", "test_persona.py's behavior fixture"),
    ("itest-badver-a1b2c3", "test_portability.py's short suffix, hex[:6]"),
]
for cid, why in POSITIVES:
    check(
        portability._test_conv_match_reason(cid) is not None,
        f"matches: {cid!r} ({why})",
    )

NEGATIVES = [
    ("clone_conv_id_here", "wrong case — the sanitizer preserves case, so the "
     "real placeholder is uppercase only"),
    ("CLONE_CONV_ID_HERE_2", "not an exact literal match"),
    ("__selftest__", "the FIXED sentinel — its own round trip purges both "
     "sides, and it isn't part of what N6 counted as pollution"),
    ("__selftest_oneshot_1a2b3c4d", "missing the trailing __"),
    ("__selftest_oneshot_1A2B3C4D__", "uppercase hex — uuid4().hex is always lowercase"),
    ("__selftest_oneshot_1a2b3c4__", "7 hex chars, selftest.py always mints 8"),
    ("itestx-a1b2c3d4", "no hyphen after itest — not the harness prefix"),
    ("my-itest-a1b2c3d4", "itest- is not a PREFIX here"),
    ("itest-PROD-a1b2c3d4", "uppercase descriptive segment — harness segments are lowercase"),
    ("itest-my-report", "no trailing hex run — could be a real id that merely "
     "starts with the word itest"),
    ("itest-abcdef1234567890abcdef", "24 hex chars — longer than any harness suffix"),
    ("550e8400-e29b-41d4-a716-446655440000", "a real OpenWebUI-shaped UUID"),
    ("a1b2c3d4e5f68899", "a real hash-fallback 16-hex id — no itest- prefix"),
    ("itest-g1h2i3", "g/h/i are not hex digits"),
]
for cid, why in NEGATIVES:
    check(
        portability._test_conv_match_reason(cid) is None,
        f"does NOT match: {cid!r} ({why})",
    )

# ---------------------------------------------------------------------------
# [2] substantial content beats a pattern match
# ---------------------------------------------------------------------------

print()
print("[2] a matched id holding real content is kept, not removed")

reset_state("itest-tiny-a1b2c3d4")
facts.save_facts(
    "itest-tiny-a1b2c3d4",
    [{"text": "seeded fact", "added_turn": 0, "last_used": 100}],
)
matches = {m["conv_id"]: m for m in portability.find_test_conversations()}
check(
    matches["itest-tiny-a1b2c3d4"]["safe_to_remove"] is True,
    "1 fact (test-suite-shaped) is safe to remove",
)

reset_state("itest-big-a1b2c3d4")
facts.save_facts(
    "itest-big-a1b2c3d4",
    [
        {"text": f"fact {i}", "added_turn": i, "last_used": 100 + i}
        for i in range(15)
    ],
)
matches = {m["conv_id"]: m for m in portability.find_test_conversations()}
check(
    matches["itest-big-a1b2c3d4"]["safe_to_remove"] is False,
    "15 facts (> SUBSTANTIAL_FACTS=10) is KEPT despite matching itest-*",
)
check(
    any("15 active fact" in r for r in matches["itest-big-a1b2c3d4"]["reasons_kept"]),
    "the kept reason names the actual count",
)

reset_state("itest-persona-a1b2c3d4")
facts.save_facts("itest-persona-a1b2c3d4", [])  # tombstone so it's a "known" conv
persona.save_persona("itest-persona-a1b2c3d4", "a persona nobody should lose")
matches = {m["conv_id"]: m for m in portability.find_test_conversations()}
check(
    matches["itest-persona-a1b2c3d4"]["safe_to_remove"] is False,
    "a stored persona is KEPT regardless of fact count",
)
persona.clear_persona("itest-persona-a1b2c3d4")

reset_state("itest-archived-a1b2c3d4")
facts.save_facts("itest-archived-a1b2c3d4", [])  # tombstone so it's a "known" conv
facts.save_archive(
    "itest-archived-a1b2c3d4",
    [
        {"text": f"archived {i}", "added_turn": i, "last_used": i}
        for i in range(12)
    ],
)
matches = {m["conv_id"]: m for m in portability.find_test_conversations()}
check(
    matches["itest-archived-a1b2c3d4"]["safe_to_remove"] is False,
    "12 archived facts (> SUBSTANTIAL_ARCHIVED_FACTS=10) is KEPT",
)

reset_state("itest-episodic-a1b2c3d4")
# memory.list_known_conv_ids only globs facts/ and summaries/ — a conv_id
# whose only layer is episodic wouldn't be scanned at all without this
# tombstone, same reason commands.py leaves one behind after a real wipe.
facts.save_facts("itest-episodic-a1b2c3d4", [])
for i in range(11):
    _stub_import("itest-episodic-a1b2c3d4", i, f"[user]: hi {i}")
matches = {m["conv_id"]: m for m in portability.find_test_conversations()}
check(
    matches["itest-episodic-a1b2c3d4"]["safe_to_remove"] is False,
    "11 indexed exchanges (> SUBSTANTIAL_EPISODIC=10) is KEPT",
)

for cid in (
    "itest-tiny-a1b2c3d4",
    "itest-big-a1b2c3d4",
    "itest-persona-a1b2c3d4",
    "itest-archived-a1b2c3d4",
    "itest-episodic-a1b2c3d4",
):
    reset_state(cid)

# A real, un-matched conv_id with a big store is never even a candidate —
# proves the "matches nothing" branch, not just the "matches but kept" one.
reset_state("real-conv-abc123")
facts.save_facts(
    "real-conv-abc123",
    [{"text": f"fact {i}", "added_turn": i, "last_used": i} for i in range(50)],
)
ids_seen = {m["conv_id"] for m in portability.find_test_conversations()}
check(
    "real-conv-abc123" not in ids_seen,
    "a real conv_id with 50 facts never appears in find_test_conversations at all",
)
reset_state("real-conv-abc123")

# ---------------------------------------------------------------------------
# [3] end to end: dry-run vs live, and partial failure
# ---------------------------------------------------------------------------

print()
print("[3] cleanup_test_conversations")


async def _fake_wipe_ok(conv_id):
    n = len(facts.load_facts(conv_id))
    facts.save_facts(conv_id, [])
    return {
        "forgotten_facts": n,
        "forgotten_episodic": 0,
        "forgotten_summary": False,
        "forgotten_persona": False,
        "unreadable": [],
    }


async def _fake_wipe_boom(conv_id):
    raise RuntimeError("simulated wipe failure")


def _seed_batch():
    reset_state("itest-cleanup-a1b2c3d4")
    facts.save_facts(
        "itest-cleanup-a1b2c3d4",
        [{"text": "small", "added_turn": 0, "last_used": 0}],
    )
    reset_state("itest-cleanup-kept-a1b2c3d4")
    facts.save_facts(
        "itest-cleanup-kept-a1b2c3d4",
        [{"text": f"f{i}", "added_turn": i, "last_used": i} for i in range(20)],
    )
    reset_state("CLONE_CONV_ID_HERE")
    facts.save_facts(
        "CLONE_CONV_ID_HERE",
        [{"text": "placeholder residue", "added_turn": 0, "last_used": 0}],
    )
    reset_state("a-real-conversation-id")
    facts.save_facts(
        "a-real-conversation-id",
        [{"text": "her real memory", "added_turn": 0, "last_used": 0}],
    )


_seed_batch()
dry = asyncio.run(portability.cleanup_test_conversations(dry_run=True))
check(dry["dry_run"] is True, "dry_run flag echoed")
check(
    {c["conv_id"] for c in dry["would_remove"]}
    == {"itest-cleanup-a1b2c3d4", "CLONE_CONV_ID_HERE"},
    "dry run lists exactly the two safe matches, not the kept one or the real id",
)
check(
    {c["conv_id"] for c in dry["kept"]} == {"itest-cleanup-kept-a1b2c3d4"},
    "dry run separately lists the matched-but-kept id, with a reason",
)
check(
    len(facts.load_facts("itest-cleanup-a1b2c3d4")) == 1
    and len(facts.load_facts("a-real-conversation-id")) == 1,
    "dry run touched NOTHING — both still hold their seeded fact",
)

no_helper_raised = False
try:
    asyncio.run(portability.cleanup_test_conversations(dry_run=False))
except ValueError:
    no_helper_raised = True
check(no_helper_raised, "dry_run=False without wipe_layers refuses outright")

live = asyncio.run(
    portability.cleanup_test_conversations(dry_run=False, wipe_layers=_fake_wipe_ok)
)
check(len(live["removed"]) == 2, "both safe matches were removed")
check(live["errors"] == [], "no errors on the happy path")
check(
    facts.load_facts("itest-cleanup-a1b2c3d4") == [],
    "the removed test conv's facts are actually gone",
)
check(
    len(facts.load_facts("a-real-conversation-id")) == 1,
    "the real, unmatched conversation is completely untouched",
)
check(
    len(facts.load_facts("itest-cleanup-kept-a1b2c3d4")) == 20,
    "the matched-but-substantial conversation is completely untouched",
)
snaps = portability.list_quarantine("itest-cleanup-a1b2c3d4")
check(
    len(snaps) == 1,
    "a quarantine snapshot exists for the removed conv before its facts were cleared",
)

print()
print("[3b] wipe failure after quarantine leaves the snapshot, reports it, doesn't lose data")
_seed_batch()
live2 = asyncio.run(
    portability.cleanup_test_conversations(dry_run=False, wipe_layers=_fake_wipe_boom)
)
check(live2["removed"] == [], "nothing reported removed when every wipe fails")
check(
    len(live2["errors"]) == 2 and all(e["stage"] == "wipe" for e in live2["errors"]),
    "both failures are reported at the wipe stage",
)
check(
    all(e.get("quarantine_path") for e in live2["errors"]),
    "each error carries the quarantine path — the snapshot is not lost track of",
)
check(
    len(facts.load_facts("itest-cleanup-a1b2c3d4")) == 1,
    "a wipe failure leaves the conversation's facts INTACT, not half-deleted",
)

for cid in (
    "itest-cleanup-a1b2c3d4",
    "itest-cleanup-kept-a1b2c3d4",
    "CLONE_CONV_ID_HERE",
    "a-real-conversation-id",
):
    reset_state(cid)

print()
if FAILED:
    print(f"{len(FAILED)} FAILURE(S):")
    for f in FAILED:
        print("  - " + f)
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    sys.exit(1)

shutil.rmtree(_TMP_ROOT, ignore_errors=True)
print("All N6 cleanup tests passed.")
