"""
CPU-only smoke tests for compactor/retrieval.py (V2.0 Phase 3).

retrieval.py imports fastembed + chromadb lazily (inside _try_init), so this
test runs WITHOUT those heavy deps installed — it either exercises the
graceful-degradation path (deps unavailable → safe no-ops) or injects mock
embedder/collection objects to test the index/query/forget logic directly.

The middle of the file covers v3.1 D1 — content-addressed document ids. See the
block comment above test_deleted_message_does_not_overwrite_an_existing_exchange
for the production sequence those tests are reconstructed from.

The second half of the file covers the v3.1 retrieval budget
(MAX_RETRIEVAL_TOKENS / COMPACTOR_MAX_RETRIEVAL_TOKENS), added after the
2026-08-27 production 400: facts were capped, summary chunks were capped, and
retrieved exchanges were capped by nothing. See
test_regression_20260827_three_exchanges_cannot_overflow_the_window.

Run:
    python test_retrieval.py
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-retrieval-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import logsetup  # noqa: E402
import retrieval  # noqa: E402


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


# ---------------------------------------------------------------------------
# Mocks — let us test logic without fastembed/chromadb installed
# ---------------------------------------------------------------------------

class _MockVec:
    def __init__(self, data):
        self._data = data
    def tolist(self):
        return self._data


class MockEmbedder:
    """Returns a deterministic fake vector per input text."""
    def __init__(self):
        self.calls = []
    def embed(self, texts):
        self.calls.append(list(texts))
        for t in texts:
            yield _MockVec([float(len(t)), 0.0, 1.0])


class MockCollection:
    """Records upsert/query/get/delete and returns canned query results."""
    def __init__(self):
        self.upserts = []
        self.deleted_ids = []
        self._store = {}  # id -> (doc, meta)
        self.canned_query = None  # set per-test
        self.n_results_seen = []  # every n_results this mock was asked for

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, _id in enumerate(ids):
            self._store[_id] = (documents[i], metadatas[i])
        self.upserts.append({"ids": ids, "metadatas": metadatas})

    def query(self, query_embeddings, n_results, where):
        """Conv-filtered rows, in insertion order, TRUNCATED TO n_results.

        v3.1 A7. This mock took `n_results` and ignored it, returning every
        conv-matching row however few were asked for — so a retrieve() that
        asked for k and then filtered looked identical to one that over-fetched,
        and `test_retrieve_excludes_recent_turns` passed either way. Exactly the
        blind spot that let char/4 stand in for a tokenizer for two incidents:
        **the mock was more generous than production.** Real chroma returns at
        most n_results rows, and that limit is the whole subject of A7, so it
        has to be modelled here or the fix is unverifiable.

        Insertion order stands in for similarity rank. The mock embedder returns
        a length-derived vector and a flat 0.1 distance, so there is no real
        ranking to reproduce; a test that cares which candidates come back first
        controls it by seeding in the order it wants (see the A7 tests)."""
        self.n_results_seen.append(n_results)
        if self.canned_query is not None:
            return self.canned_query
        cid = where.get("conv_id")
        ids, docs, metas, dists = [], [], [], []
        for _id, (doc, meta) in self._store.items():
            if meta.get("conv_id") == cid:
                ids.append(_id); docs.append(doc); metas.append(meta); dists.append(0.1)
            if len(ids) >= n_results:
                break
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}

    def get(self, where=None, ids=None, include=None):
        """Mirrors the three call shapes retrieval.py uses against the real
        chromadb 1.5.9, verified in angreg/zions-light-ai:v3.0.5-cu12:

            get(where={"conv_id": ...})                  -> ids+docs+metas
            get(where={"conv_id": ...}, include=["metadatas"])
            get(ids=[...], include=[])                   -> the D1 exists probe

        Real chroma returns only the requested fields and drops unknown ids
        from the result rather than raising, which is what makes the probe a
        membership test."""
        if ids is not None:
            matched = [i for i in ids if i in self._store]
        else:
            cid = (where or {}).get("conv_id")
            matched = [
                i for i, (_, m) in self._store.items() if m.get("conv_id") == cid
            ]
        out = {"ids": matched}
        if include is None or "documents" in include:
            out["documents"] = [self._store[i][0] for i in matched]
        if include is None or "metadatas" in include:
            out["metadatas"] = [self._store[i][1] for i in matched]
        return out

    def delete(self, ids):
        for i in ids:
            self.deleted_ids.append(i)
            self._store.pop(i, None)

    def turn_indices(self, conv_id):
        """Stored ordering metadata for a conv, in insertion order."""
        return [
            m["turn_index"]
            for (_, m) in self._store.values()
            if m.get("conv_id") == conv_id
        ]


def _install_mocks():
    """Force retrieval into 'available' state with mock backends."""
    retrieval._available = True
    retrieval._embedder = MockEmbedder()
    retrieval._chroma_collection = MockCollection()
    return retrieval._embedder, retrieval._chroma_collection


def _force_unavailable():
    retrieval._available = False
    retrieval._embedder = None
    retrieval._chroma_collection = None


def _seed_legacy_row(col, conv_id, turn_index, document):
    """Write a PRE-D1 row straight into the store: id `{conv_id}::{N}`.

    These are what the live ChromaDB is full of, and D1 does not migrate them —
    the requirement is that they keep working beside the new hashed ids. Seeded
    directly rather than through index_exchange because index_exchange can no
    longer produce this id format, which is the whole point of the fix."""
    col._store[f"{conv_id}::{turn_index}"] = (
        document, {"conv_id": conv_id, "turn_index": turn_index}
    )


# ---------------------------------------------------------------------------
# Pure helpers (no backend needed)
# ---------------------------------------------------------------------------

def test_exchange_doc_format():
    print("\n[test] _exchange_doc renders canonical user/assistant text")
    doc = retrieval._exchange_doc("hello", "hi there")
    assert_true("[user]: hello" in doc, "user line present")
    assert_true("[assistant]: hi there" in doc, "assistant line present")


def test_doc_id_is_content_addressed():
    print("\n[test] _doc_id hashes the exchange text, not the client's turn index")
    doc = retrieval._exchange_doc("who is Lyra?", "a half-elf ranger")
    got = retrieval._doc_id("abc", doc)

    assert_true(got.startswith("abc::"), f"conversation-scoped prefix: {got!r}")
    assert_eq(len(got.split("::", 1)[1]), 16, "16 hex chars of sha256")
    assert_eq(got, retrieval._doc_id("abc", doc), "stable across calls")
    assert_true(got != retrieval._doc_id("abd", doc), "same text, other conv -> other id")
    assert_true(
        got != retrieval._doc_id("abc", doc + "."),
        "one character of difference -> its own row",
    )
    # The id takes nothing from the request at all — the parameter it used to
    # take is gone. index_exchange still accepts turn_index for ordering; the
    # tests below are what hold it out of the identity.


def test_format_retrieval_block_empty():
    print("\n[test] format_retrieval_block returns None for no hits")
    assert_eq(retrieval.format_retrieval_block([]), None, "empty -> None")


def test_format_retrieval_block_orders_by_turn():
    print("\n[test] format_retrieval_block orders chronologically + has header")
    hits = [
        {"turn_index": 50, "document": "later", "distance": 0.2},
        {"turn_index": 10, "document": "earlier", "distance": 0.1},
    ]
    block = retrieval.format_retrieval_block(hits)
    assert_true("Relevant earlier exchanges" in block, "header present")
    # "earlier" (turn 10) must appear before "later" (turn 50)
    assert_true(block.index("earlier") < block.index("later"), "chronological order")


# ---------------------------------------------------------------------------
# Degraded mode (deps unavailable) — the safety contract
# ---------------------------------------------------------------------------

def test_degraded_index_returns_false():
    print("\n[test] index_exchange returns False when retrieval unavailable")
    _force_unavailable()
    assert_eq(retrieval.index_exchange("c", 2, "u", "a"), False, "no-op index")


def test_degraded_retrieve_returns_empty():
    print("\n[test] retrieve returns [] when retrieval unavailable")
    _force_unavailable()
    assert_eq(retrieval.retrieve("c", "query"), [], "no-op retrieve")


def test_degraded_forget_returns_zero():
    print("\n[test] forget_conversation returns 0 when unavailable")
    _force_unavailable()
    assert_eq(retrieval.forget_conversation("c"), 0, "no-op forget")


def test_degraded_doc_count_returns_none_not_zero():
    """v3.1 P0-2b / F61. Returning 0 for an unavailable store made a dead
    ChromaDB indistinguishable from an empty one — /health/full printed
    `indexed_exchanges_total: 0` beside `"status": "ok"`. None means
    unknown; 0 must mean genuinely empty."""
    print("\n[test] conversation_doc_count returns None (not 0) when unavailable")
    _force_unavailable()
    assert_eq(retrieval.conversation_doc_count("c"), None, "unavailable -> None")


def test_doc_count_returns_none_on_query_failure():
    print("\n[test] conversation_doc_count returns None when the store raises")
    emb, col = _install_mocks()

    def _boom(where):
        raise RuntimeError("chroma is on fire")

    col.get = _boom
    assert_eq(retrieval.conversation_doc_count("conv1"), None, "raised -> None")


def test_doc_count_zero_means_genuinely_empty():
    print("\n[test] conversation_doc_count returns 0 for a healthy, empty conv")
    emb, col = _install_mocks()
    retrieval.index_exchange("populated", 2, "u", "a")
    assert_eq(retrieval.conversation_doc_count("never-seen"), 0, "empty -> 0, not None")
    assert_eq(retrieval.conversation_doc_count("populated"), 1, "one indexed exchange")


# ---------------------------------------------------------------------------
# Index / retrieve / forget with mock backends
# ---------------------------------------------------------------------------

def test_index_exchange_upserts():
    print("\n[test] index_exchange embeds + upserts with conv metadata")
    emb, col = _install_mocks()
    ok = retrieval.index_exchange("conv1", 4, "who is Lyra?", "a half-elf ranger")
    doc = retrieval._exchange_doc("who is Lyra?", "a half-elf ranger")
    assert_eq(ok, True, "index succeeded")
    assert_eq(len(col.upserts), 1, "one upsert call")
    assert_eq(col.upserts[0]["ids"], [retrieval._doc_id("conv1", doc)], "content-addressed id")
    assert_eq(col.upserts[0]["metadatas"][0]["conv_id"], "conv1", "conv_id in metadata")
    # Nothing stored for conv1 yet, so the ordinal seeds from the caller's
    # turn_index. Later exchanges take the store's max as a floor instead.
    assert_eq(col.upserts[0]["metadatas"][0]["turn_index"], 4, "turn_index in metadata")


def test_index_exchange_skips_empty():
    print("\n[test] index_exchange skips empty user/assistant text")
    emb, col = _install_mocks()
    assert_eq(retrieval.index_exchange("c", 2, "", "a"), False, "empty user -> skip")
    assert_eq(retrieval.index_exchange("c", 2, "u", ""), False, "empty assistant -> skip")
    assert_eq(len(col.upserts), 0, "no upserts for empty input")


def test_retrieve_returns_matches():
    print("\n[test] retrieve returns indexed exchanges for the conv")
    emb, col = _install_mocks()
    retrieval.index_exchange("conv1", 2, "u1", "a1")
    retrieval.index_exchange("conv1", 4, "u2", "a2")
    retrieval.index_exchange("other", 2, "x", "y")  # different conv
    hits = retrieval.retrieve("conv1", "query text", k=5)
    assert_eq(len(hits), 2, "only conv1's two exchanges")
    turns = sorted(h["turn_index"] for h in hits)
    assert_eq(turns, [2, 4], "correct turn indices")


def test_retrieve_excludes_recent_turns():
    print("\n[test] retrieve drops turns >= exclude_turns_from")
    emb, col = _install_mocks()
    # Seeded as pre-D1 rows: post-D1 the ordinal is allocated from the store's
    # own max, so index_exchange could not produce a 2/20 gap in two calls.
    _seed_legacy_row(col, "conv1", 2, retrieval._exchange_doc("old", "old-a"))
    _seed_legacy_row(col, "conv1", 20, retrieval._exchange_doc("recent", "recent-a"))
    # 14, not the 10 this test used before A6. The store's max is 20, so a
    # cutoff of 10 sits 10 message-units below it — further than the 8-unit
    # recent window a consistent caller could ever put it, which is now read as
    # the client's framing having drifted from the store's and the filter is
    # ignored. 14 is 6 below, i.e. an ordinary in-frame cutoff, which is what
    # this test is about. The out-of-frame case has its own test below.
    hits = retrieval.retrieve("conv1", "q", k=5, exclude_turns_from=14)
    assert_eq(len(hits), 1, "recent turn (20) excluded")
    assert_eq(hits[0]["turn_index"], 2, "only the old turn remains")


def test_retrieve_empty_query():
    print("\n[test] retrieve returns [] for empty query text")
    _install_mocks()
    assert_eq(retrieval.retrieve("conv1", ""), [], "empty query -> []")


def test_forget_conversation_deletes():
    print("\n[test] forget_conversation deletes all of a conv's exchanges")
    emb, col = _install_mocks()
    retrieval.index_exchange("conv1", 2, "u1", "a1")
    retrieval.index_exchange("conv1", 4, "u2", "a2")
    retrieval.index_exchange("keep", 2, "x", "y")
    n = retrieval.forget_conversation("conv1")
    assert_eq(n, 2, "deleted 2 from conv1")
    # 'keep' conv survives
    assert_eq(retrieval.conversation_doc_count("keep"), 1, "other conv untouched")
    assert_eq(retrieval.conversation_doc_count("conv1"), 0, "conv1 now empty")


# ---------------------------------------------------------------------------
# v3.1 D1 — content-addressed document ids
# ---------------------------------------------------------------------------
#
# The highest-severity finding in the v3.1 bundle, and the only one whose
# damage is unrecoverable. `_doc_id` was `{conv_id}::{turn_index}` where
# turn_index is main.py's `len(messages)+1` — a measurement of the CLIENT'S
# ARRAY. Deleting messages in OpenWebUI shortens that array, so the index goes
# down and the upsert lands on an id that already holds a different exchange.
#
# Measured in production. Five DELETE /api/v1/chats/…/messages/… between
# 06:15:29 and 06:15:44 produced this indexed sequence:
#
#     42, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58
#
# The real turn ~42, written 06:14:54, was destroyed at 14:49:22 by the second
# 42. The phantom conversation is the same failure with the array pinned short:
# sixteen writes to index 2, leaving one document and fifteen destructions.
#
# These four tests are the ones named in the D1 task. Each was mutation-checked
# against the fixed code.


def test_deleted_message_does_not_overwrite_an_existing_exchange():
    print("\n[test] D1 — a shortened client array adds an exchange, never replaces one")
    emb, col = _install_mocks()

    # 06:14:54 — the client's array is 41 messages long. This is the exchange
    # that production lost.
    doomed = "what did she say about the road?"
    retrieval.index_exchange("prod", 42, doomed, "that it was watched")

    # 06:15:29-06:15:44 — five messages deleted. turn_index drops and then
    # climbs back, which is the measured sequence 42, 34, 36, 38, 40, 42. The
    # SIXTH write is the one that mattered: pre-D1 it reused `prod::42` and
    # overwrote the exchange above at 14:49:22. The shortening alone is
    # harmless; it is the climb back onto an occupied id that destroys.
    for nominal, text in (
        (34, "and the watchers?"),
        (36, "how many of them?"),
        (38, "did they follow?"),
        (40, "to the bridge?"),
        (42, "and then?"),
    ):
        retrieval.index_exchange("prod", nominal, text, "mm")

    assert_eq(retrieval.conversation_doc_count("prod"), 6, "six exchanges, none replaced")
    docs = [doc for doc, _ in col._store.values()]
    assert_true(
        any(doomed in d for d in docs),
        "the exchange at the reused index survived the second write to it",
    )
    assert_true(any("and then?" in d for d in docs), "and the later one was stored too")
    # Ordinals came from the store, stepping by one exchange, never from the
    # request — so the stored sequence cannot run backwards even while the
    # client's array does.
    assert_eq(
        col.turn_indices("prod"), [42, 44, 46, 48, 50, 52], "ordering only moves forward"
    )


def test_identical_text_reindexed_is_idempotent():
    print("\n[test] D1 — re-indexing identical text is one row and no second embed")
    emb, col = _install_mocks()

    retrieval.index_exchange("c", 10, "the same question", "the same answer")
    embeds_after_first = len(emb.calls)

    # A retry, a replayed request, a re-import — same text, any turn index.
    ok = retrieval.index_exchange("c", 999, "the same question", "the same answer")

    assert_eq(ok, True, "reported as indexed, because it is")
    assert_eq(retrieval.conversation_doc_count("c"), 1, "still one row")
    assert_eq(len(col.upserts), 1, "no second upsert")
    assert_eq(len(emb.calls), embeds_after_first, "the exists probe skipped the embed")
    assert_eq(col.turn_indices("c"), [10], "the stored ordinal is untouched")


def test_distinct_text_always_gets_a_row():
    print("\n[test] D1 — distinct text always gets its own row, however the index moves")
    emb, col = _install_mocks()

    # Turn indices deliberately hostile: descending, repeating, and colliding
    # with each other — every shape a deleting/editing/windowing client makes.
    for nominal, text in (
        (42, "the first thing she said"),
        (34, "the second thing she said"),
        (34, "the third thing she said"),
        (2, "the fourth thing she said"),
        (2, "the fourth thing she said."),  # one character apart
    ):
        assert_eq(
            retrieval.index_exchange("c", nominal, text, "mm"), True, f"indexed {text!r}"
        )

    assert_eq(retrieval.conversation_doc_count("c"), 5, "five distinct exchanges, five rows")
    assert_eq(
        col.turn_indices("c"), [42, 44, 46, 48, 50], "ordinals ascend by one exchange"
    )


def test_sixteen_writes_at_one_turn_index_are_sixteen_rows():
    print("\n[test] D1 — REGRESSION: the phantom conversation's 16 writes to index 2")
    # conv 31365d633335bbd0 in production: 105 facts and ONE episodic row. The
    # client's array never grew past two messages, so turn_index was 2 on every
    # request and every exchange landed on `31365d633335bbd0::2`. Fifteen
    # exchanges were destroyed by the sixteenth. This is the test that fails if
    # document identity is ever derived from the request again.
    emb, col = _install_mocks()
    conv = "31365d633335bbd0"

    for n in range(16):
        retrieval.index_exchange(
            conv, 2, f"question {n} about the northern road", f"answer {n}"
        )

    assert_eq(retrieval.conversation_doc_count(conv), 16, "sixteen rows, not one")
    assert_eq(len(emb.calls), 16, "each distinct exchange was embedded once")
    ordinals = col.turn_indices(conv)
    assert_eq(len(set(ordinals)), 16, "no two exchanges share an ordinal")
    assert_eq(ordinals, sorted(ordinals), "and they ascend in arrival order")

    # Every one of the sixteen is retrievable, which is the user-visible point.
    hits = retrieval.retrieve(conv, "the northern road", k=20)
    assert_eq(len(hits), 16, "all sixteen are retrievable")


def test_legacy_rows_coexist_and_the_ordinal_continues_past_them():
    print("\n[test] D1 — pre-D1 `{conv}::{N}` rows survive beside hashed ids")
    # There is no migration: the live store is full of these and they are the
    # only copy of what they hold. Every reader filters on the conv_id METADATA
    # — verified across retrieve, forget_conversation, conversation_doc_count
    # and export_indexed_exchanges — so the id format is free to differ.
    emb, col = _install_mocks()
    _seed_legacy_row(col, "old", 56, retrieval._exchange_doc("legacy u", "legacy a"))
    _seed_legacy_row(col, "old", 58, retrieval._exchange_doc("legacy u2", "legacy a2"))

    retrieval.index_exchange("old", 4, "a new question", "a new answer")

    assert_eq(retrieval.conversation_doc_count("old"), 3, "legacy rows still counted")
    assert_true("old::56" in col._store, "legacy id untouched")
    assert_eq(
        max(col.turn_indices("old")), 60, "the new row continues past the legacy max"
    )
    assert_eq(len(retrieval.retrieve("old", "q", k=10)), 3, "all three retrievable")
    assert_eq(retrieval.forget_conversation("old"), 3, "and /forget clears both formats")


def test_a_request_ahead_of_the_store_pulls_the_ordinal_forward():
    print("\n[test] D1 — the store's max is a floor, not a ceiling")
    # A conversation damaged pre-D1: one surviving row pinned at the index the
    # collapse landed on, while the live conversation is hundreds of messages
    # along. Numbering new rows 4, 6, 8… here would leave every stored ordinal
    # far below main.py's `recent_cutoff = turn_index - KEEP_RECENT_TURNS*2`,
    # so nothing would ever be excluded as recent and the filter would stop
    # meaning anything. The request may raise the sequence; it may never lower
    # it, which is the property that stops the overwriting.
    emb, col = _install_mocks()
    _seed_legacy_row(col, "damaged", 2, retrieval._exchange_doc("survivor", "row"))

    retrieval.index_exchange("damaged", 300, "back to the road", "it is still watched")
    assert_eq(max(col.turn_indices("damaged")), 300, "the request pulled it forward")

    # And it still cannot go backwards: the next request arrives shorter.
    retrieval.index_exchange("damaged", 12, "one more thing", "go on")
    assert_eq(
        sorted(col.turn_indices("damaged")), [2, 300, 302], "a shorter array cannot pull back"
    )


def test_import_keeps_the_bundle_turn_index_and_shares_identity_with_indexing():
    print("\n[test] D1 — import preserves bundle ordering; ids match live indexing")
    emb, col = _install_mocks()
    doc = retrieval._exchange_doc("bundled u", "bundled a")

    assert_eq(retrieval.import_indexed_exchange("c", 12, doc), True, "imported")
    assert_eq(col.upserts[0]["ids"], [retrieval._doc_id("c", doc)], "content-addressed")
    # The bundle's turn_index is the SOURCE conversation's ordering and it is
    # what export_indexed_exchanges sorts on. Re-allocating it here would make
    # the round-trip lossy, so import is the one writer that keeps the caller's.
    assert_eq(col.upserts[0]["metadatas"][0]["turn_index"], 12, "bundle ordering kept")

    # The same exchange arriving live is the same row, not a duplicate.
    assert_eq(retrieval.index_exchange("c", 77, "bundled u", "bundled a"), True, "no-op")
    assert_eq(retrieval.conversation_doc_count("c"), 1, "one row, not two")
    assert_eq(col.turn_indices("c"), [12], "and the bundle's ordinal stands")


# ---------------------------------------------------------------------------
# v3.1 A6 — the cutoff is client-derived, and the store's is not
# ---------------------------------------------------------------------------
#
# `exclude_turns_from` arrives as main.py's
# `max(0, turn_index - KEEP_RECENT_TURNS * 2)` where `turn_index` is
# `len(messages) + 1` — a measurement of the CLIENT'S ARRAY. Stored ordinals are
# allocated from the store's own maximum (D1, _next_turn_index). Two
# authorities, one comparison.
#
# REMEDIATION P0-3 deferred this half explicitly and D1 landed without it. P4
# repaired `cutoff == 0` only; under a bounded client window the cutoff is a
# POSITIVE number that conditional does not touch, and every hit is suppressed
# on every request, indefinitely.


def _seed_ordinals(col, conv_id, ordinals):
    """Seed one row per ordinal, in the order given."""
    for n in ordinals:
        _seed_legacy_row(
            col, conv_id, n, retrieval._exchange_doc(f"u{n}", f"a{n}")
        )


def test_a_bounded_client_window_no_longer_suppresses_every_hit():
    print("\n[test] A6 — REGRESSION: a bounded client window kept 0 of 12 hits")
    # The v3.1 preflight's reproduction, verbatim: a client sending a window of
    # 20 messages against stored ordinals [21, 23, ... 43]. turn_index is 21,
    # so the cutoff is max(0, 21 - 8) = 13 — positive, so P4's `> 0` guard does
    # not fire — and every stored ordinal is >= 13. Result before A6: an empty
    # block on every request, forever, with `0retr` the only trace.
    emb, col = _install_mocks()
    logsetup._reset_log_once_for_tests()
    ordinals = list(range(21, 44, 2))
    assert_eq(len(ordinals), 12, "fixture: twelve stored exchanges")
    _seed_ordinals(col, "windowed", ordinals)

    # Teeth check: the cutoff really does exclude all twelve, so this fixture
    # fails on code without the fix rather than passing vacuously.
    assert_true(
        all(n >= 13 for n in ordinals),
        "fixture: the client's cutoff of 13 is below every stored ordinal",
    )
    assert_true(
        retrieval._cutoff_is_out_of_frame("windowed", 13),
        "the store's max is further than one recent window above the cutoff",
    )

    hits = retrieval.retrieve("windowed", "q", k=5, exclude_turns_from=13)
    assert_eq(len(hits), 5, "the filter was ignored, so k hits come back")
    assert_true(
        all(h["turn_index"] in ordinals for h in hits), "and they are real rows"
    )


def test_the_out_of_frame_cutoff_says_so_in_the_log():
    print("\n[test] A6 — ignoring the caller's filter is announced, not silent")
    # A retrieval layer that quietly overrides its caller is how you get an
    # operator debugging "why is it repeating itself" with nothing to grep.
    emb, col = _install_mocks()
    logsetup._reset_log_once_for_tests()
    _seed_ordinals(col, "loud", list(range(21, 44, 2)))
    cap = _LogCapture()
    try:
        retrieval.retrieve("loud", "q", k=5, exclude_turns_from=13)
    finally:
        cap.stop()
    warnings = cap.messages(logging.WARNING)
    assert_eq(len(warnings), 1, "exactly one WARNING")
    assert_true("exclude_turns_from=13" in warnings[0], f"names the cutoff: {warnings[0]!r}")
    assert_true("turn_seq" in warnings[0], "and points at the real fix")


def test_a_short_conversation_keeps_its_filter():
    print("\n[test] A6 — an in-frame cutoff is still honoured, not overridden")
    # The failure mode A6 guards against must not become "the filter never
    # applies". Where the two framings agree — a short conversation whose every
    # stored exchange really is in the request verbatim — the store's max sits
    # exactly one recent window above the cutoff and the filter stands.
    emb, col = _install_mocks()
    logsetup._reset_log_once_for_tests()
    _seed_ordinals(col, "short", [3, 5, 7, 9, 11, 13])
    # Client array of 12 messages -> turn_index 13 -> cutoff 13 - 8 = 5.
    assert_eq(
        retrieval._cutoff_is_out_of_frame("short", 5), False, "13 - 5 == the window"
    )
    hits = retrieval.retrieve("short", "q", k=5, exclude_turns_from=5)
    assert_eq(
        sorted(h["turn_index"] for h in hits), [3], "only the pre-window exchange"
    )


def test_cutoff_zero_still_excludes_nothing():
    print("\n[test] A6 — P4's `cutoff == 0 means exclude nothing` is unchanged")
    emb, col = _install_mocks()
    logsetup._reset_log_once_for_tests()
    _seed_ordinals(col, "zero", [2, 4, 6])
    hits = retrieval.retrieve("zero", "q", k=5, exclude_turns_from=0)
    assert_eq(sorted(h["turn_index"] for h in hits), [2, 4, 6], "all three survive")
    # And no override warning: nothing was dropped, so nothing needed rescuing.
    cap = _LogCapture()
    try:
        retrieval.retrieve("zero", "q", k=5, exclude_turns_from=0)
    finally:
        cap.stop()
    assert_eq(cap.messages(logging.WARNING), [], "silent — the filter never bit")


def test_the_frame_check_is_only_paid_when_the_caller_lost_hits():
    print("\n[test] A6 — no extra store read unless the filter actually cost hits")
    # _cutoff_is_out_of_frame reads the store, and this runs on the request hot
    # path. Two cases must not pay for it: a cutoff that excluded nothing (there
    # is nothing to be suppressing), and a cutoff that excluded rows but still
    # returned k (the over-fetch already made it whole). The second is the long
    # healthy conversation — the one with the most metadata to scan.
    emb, col = _install_mocks()
    logsetup._reset_log_once_for_tests()
    calls = []
    real_get = col.get

    def counting_get(where=None, ids=None, include=None):
        calls.append(where)
        return real_get(where=where, ids=ids, include=include)

    col.get = counting_get

    _seed_ordinals(col, "cheap", [2, 4, 6])
    retrieval.retrieve("cheap", "q", k=5, exclude_turns_from=100)
    assert_eq(calls, [], "everything is below the cutoff -> no frame check")

    # 4 recent rows ranked first, then 8 older ones: k+4=9 fetched, 4 dropped,
    # 5 kept — the caller got what it asked for, so nothing needs rescuing.
    _seed_ordinals(col, "healthy", [46, 48, 50, 52])
    _seed_ordinals(col, "healthy", list(range(20, 36, 2)))
    hits = retrieval.retrieve("healthy", "q", k=5, exclude_turns_from=46)
    assert_eq(len(hits), 5, "precondition: the over-fetch covered the exclusions")
    assert_eq(calls, [], "k hits returned -> still no frame check")

    # And the case that does need it: fewer than k survive.
    retrieval.retrieve("cheap", "q", k=5, exclude_turns_from=4)
    assert_eq(len(calls), 1, "a filter that cost hits -> exactly one frame check")


# ---------------------------------------------------------------------------
# v3.1 A7 — over-fetch, so a filtered slot is replaced rather than deleted
# ---------------------------------------------------------------------------


def test_the_mock_honours_n_results():
    print("\n[test] A7 — the fake collection truncates to n_results, as chroma does")
    # This assertion is the precondition for every A7 test below. The mock used
    # to take n_results and ignore it, which made over-fetch unobservable: the
    # filtered-slot bug and its fix produced identical results here. Assert the
    # mock's fidelity directly so a future edit that re-loosens it fails LOUDLY
    # rather than quietly re-blinding the suite.
    emb, col = _install_mocks()
    _seed_ordinals(col, "many", list(range(2, 42, 2)))  # 20 rows
    res = col.query(query_embeddings=[[0.0]], n_results=3, where={"conv_id": "many"})
    assert_eq(len(res["ids"][0]), 3, "asked for 3, got 3 — not all 20")


def test_a_filtered_slot_is_replaced_not_deleted():
    print("\n[test] A7 — filtering k candidates used to return fewer than k")
    emb, col = _install_mocks()
    logsetup._reset_log_once_for_tests()
    # Seed the four rows the recent window covers FIRST, so they occupy the top
    # of the candidate ranking — which is the realistic case, since the turns
    # closest to the current question are usually the ones most like it, and
    # exclude_turns_from exists precisely because they are already in the
    # request verbatim. Then nine older rows behind them.
    recent = [46, 48, 50, 52]
    older = list(range(20, 38, 2))
    _seed_ordinals(col, "deep", recent)
    _seed_ordinals(col, "deep", older)
    # Store max 52, cutoff 46 -> 6 units, inside the window: in frame, so the
    # filter genuinely applies and this is testing over-fetch, not A6.
    assert_eq(retrieval._cutoff_is_out_of_frame("deep", 46), False, "in frame")

    hits = retrieval.retrieve("deep", "q", k=5, exclude_turns_from=46)

    # Pre-A7: n_results=5 returned exactly the four recent rows plus one older,
    # the filter dropped four, and the caller got ONE hit where it asked for
    # five — with four perfectly good older exchanges sitting unqueried.
    assert_eq(len(hits), 5, "five asked for, five returned")
    assert_true(
        all(h["turn_index"] < 46 for h in hits), "and none from the recent window"
    )
    assert_eq(col.n_results_seen[-1], 9, "asked chroma for k + 4, not k")


def test_overfetch_is_bounded_and_trimmed_to_k():
    print("\n[test] A7 — the over-fetch is k + one recent window, and k is honoured")
    emb, col = _install_mocks()
    logsetup._reset_log_once_for_tests()
    _seed_ordinals(col, "big", list(range(2, 62, 2)))  # 30 rows
    hits = retrieval.retrieve("big", "q", k=5, exclude_turns_from=100)
    assert_eq(len(hits), 5, "never more than k, whatever was fetched")
    assert_eq(
        col.n_results_seen[-1],
        5 + retrieval._OVERFETCH,
        "bounded by k + _OVERFETCH, not by the conversation",
    )


def test_no_cutoff_means_no_overfetch():
    print("\n[test] A7 — nothing to replace, so nothing extra is fetched")
    emb, col = _install_mocks()
    _seed_ordinals(col, "plain", list(range(2, 42, 2)))
    hits = retrieval.retrieve("plain", "q", k=5)
    assert_eq(len(hits), 5, "k hits")
    assert_eq(col.n_results_seen[-1], 5, "asked for exactly k")


def test_query_result_parsing_robustness():
    print("\n[test] retrieve tolerates malformed/empty chroma responses")
    emb, col = _install_mocks()
    col.canned_query = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert_eq(retrieval.retrieve("c", "q"), [], "empty result lists -> []")
    col.canned_query = {}  # totally empty dict
    assert_eq(retrieval.retrieve("c", "q"), [], "empty dict -> [] (no crash)")


# ---------------------------------------------------------------------------
# The v3.1 retrieval budget (fix A)
# ---------------------------------------------------------------------------
#
# Production, 2026-08-27, conv ef1755fd144a228a on a 32,768-token window. Same
# conversation, same ~102 facts, same L1=5 summary stack, three requests:
#
#     06:44  [102 fact(s) 1 retr sum(L1=5)]  -> ok
#     22:03  [104 fact(s) 0 retr sum(L1=5)]  -> ok
#     23:51  [102 fact(s) 3 retr sum(L1=5)]  -> vLLM 400, 33,127 input tokens
#
# Compaction had already cut that request 76,104 -> 9,915 tokens; injection put
# it over the window on its own. The only variable across the three was the
# retrieved-hit count — and retrieval was the one injected layer with no cap.
#
# v3.1 A4: that cap shipped denominated in CHARACTERS (MAX_RETRIEVAL_TOKENS * 4)
# while its log line reported a token figure, so it did not cap tokens at all —
# see the A4 section further down. These tests are now written in TOKENS, as
# measured by retrieval._estimate_tokens, which is the unit the budget is in.

# The literal appended to a truncated exchange. Mirrored from retrieval.py
# rather than imported — it is not exported, and a silent change to the words
# the model actually reads should fail a test.
_TRUNCATION_MARKER = "[...truncated to fit the retrieval budget]"

_HEADER_LEN = len(retrieval._RETRIEVAL_BLOCK_HEADER)
_HEADER_TOKENS = retrieval._estimate_tokens(retrieval._RETRIEVAL_BLOCK_HEADER)


def _tokens(text):
    """The module's own measure, so a test asserts against the unit the budget
    is denominated in rather than against a second opinion."""
    return retrieval._estimate_tokens(text)


def _truncated_ceiling(budget):
    """The most a block ending in a truncated exchange may measure.

    Two bounded overshoots, both by design:
      * the marker is appended AFTER the room is computed, so it is not paid for
        (as it was not under the character budget either);
      * `_estimate_tokens` floor-divides the ASCII term, so measuring the joined
        block can come out up to one token per join above the sum of the parts
        the loop priced. The block has four ASCII groups — header, separator,
        body, marker — hence three.
    """
    return budget + _tokens(_TRUNCATION_MARKER) + 3


def _hit(turn_index, document, distance=0.1):
    return {"turn_index": turn_index, "document": document, "distance": distance}


def _sep(turn_index):
    return f"--- (turn ~{turn_index}) ---"


def _sep_len(turn_index):
    """Length of the per-exchange separator, measured the way the module
    measures it."""
    return len(_sep(turn_index))


def _hit_cost(turn_index, document):
    """What format_retrieval_block charges for one rendered exchange: the
    separator, the document, and the two newlines that join them — priced as
    the one string that actually gets emitted."""
    return _tokens(f"{_sep(turn_index)}\n{document}\n")


def _uncapped_block(hits):
    """What format_retrieval_block rendered BEFORE the cap: the header, then
    every hit whole, in turn order. The reference output for "the cap must be
    invisible when it does not bind"."""
    lines = [retrieval._RETRIEVAL_BLOCK_HEADER]
    for r in sorted(hits, key=lambda h: h["turn_index"]):
        lines.append(f"--- (turn ~{r['turn_index']}) ---")
        lines.append(r["document"])
    return "\n".join(lines)


class _LogCapture(logging.Handler):
    """Records compactor.retrieval's own log lines so a test can assert the
    cap announced itself.

    retrieval.py never configures logging (main.py does, via logsetup), so its
    logger sits at NOTSET and inherits the root's WARNING — an INFO line would
    be dropped before any handler saw it. Raise the level here and put it back
    in stop(), the way test_budget_guard.py restores the root stream.
    """

    def __init__(self):
        logging.Handler.__init__(self)
        self.records = []
        self._logger = retrieval.logger
        self._prev_level = self._logger.level
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(self)

    def emit(self, record):
        self.records.append(record)

    def stop(self):
        self._logger.removeHandler(self)
        self._logger.setLevel(self._prev_level)

    def messages(self, level):
        return [r.getMessage() for r in self.records if r.levelno == level]


def _with_budget(tokens):
    """Point MAX_RETRIEVAL_TOKENS at `tokens` for one test; returns the old
    value for the caller to restore in a finally. format_retrieval_block reads
    the module global on every call, so this is enough to exercise a budget.
    The ENV VAR is a different question — read once at import, and covered in
    a subprocess below."""
    prev = retrieval.MAX_RETRIEVAL_TOKENS
    retrieval.MAX_RETRIEVAL_TOKENS = tokens
    return prev


def test_cap_invisible_when_it_does_not_bind():
    print("\n[test] retrieval cap — under budget, output is identical to pre-cap")
    # A cap that changes the block when it does not bind is a silent rewrite of
    # every request. Under budget the bytes must not move at all.
    # Pinned rather than assumed: an ambient COMPACTOR_MAX_RETRIEVAL_TOKENS in
    # the shell must not be able to change what this fixture means.
    prev = _with_budget(1500)
    hits = [_hit(10, "A" * 300), _hit(30, "B" * 300), _hit(20, "C" * 300)]
    cap = _LogCapture()
    try:
        # 33-token header + 3 * 80 = 273 of a 1,500-token budget. (Was 272:
        # v3.1.3 split the estimator's non-ASCII pricing into script-vs-
        # decoration and added a +1 ceiling guard on any text containing
        # non-ASCII; the header's em-dash pays it. The pin exists to stop an
        # ambient env var changing what the fixture means, not to freeze the
        # estimator.)
        assert_eq(_HEADER_TOKENS + sum(_hit_cost(h["turn_index"], h["document"])
                                       for h in hits), 273, "fixture is well under")
        block = retrieval.format_retrieval_block(hits)
    finally:
        cap.stop()
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    assert_eq(block, _uncapped_block(hits), "byte-identical to the uncapped render")
    assert_eq(cap.messages(logging.INFO), [], "silent when nothing is dropped")


def test_cap_drops_later_hits_and_says_so():
    print("\n[test] retrieval cap — over budget: later hits drop, earlier survive whole")
    prev = _with_budget(1500)
    # 32 + 700 + 700 = 1,432 tokens fits; a third 700 would reach 2,132.
    hits = [
        _hit(10, "EARLY" + "a" * 2775),
        _hit(20, "MIDDLE" + "b" * 2774),
        _hit(30, "LATE" + "c" * 2776),
    ]
    assert_eq(
        [_hit_cost(h["turn_index"], h["document"]) for h in hits],
        [700, 700, 700],
        "fixture: three equal 700-token exchanges",
    )
    cap = _LogCapture()
    try:
        block = retrieval.format_retrieval_block(hits)
    finally:
        cap.stop()
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    assert_true(hits[0]["document"] in block, "first exchange present, whole")
    assert_true(hits[1]["document"] in block, "second exchange present, whole")
    assert_true("LATE" not in block, "third exchange dropped entirely")
    assert_true(_TRUNCATION_MARKER not in block, "survivors are kept whole, not trimmed")
    assert_true(_tokens(block) <= 1500, f"block within budget ({_tokens(block)} tokens)")

    # Dropping retrieved context silently is how you get an operator debugging
    # "the model forgot" with nothing in the log to point at.
    info = cap.messages(logging.INFO)
    assert_eq(len(info), 1, "exactly one INFO line")
    assert_true("kept 2 of 3" in info[0], f"names kept-of-given: {info[0]!r}")
    assert_true("COMPACTOR_MAX_RETRIEVAL_TOKENS" in info[0], "names the knob to raise")


def test_oversized_first_hit_is_truncated_not_dropped():
    print("\n[test] retrieval cap — a lone oversized exchange keeps its opening")
    prev = _with_budget(1500)
    # One exchange bigger than the entire budget. Dropping it would return no
    # block at all; the opening is where "what were we talking about" lives.
    doc = "OPENING " + "a" * 9000 + " ENDING"
    try:
        block = retrieval.format_retrieval_block([_hit(7, doc)])
    finally:
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    assert_true(block is not None, "not dropped — a partial exchange beats none")
    assert_true(block.startswith(retrieval._RETRIEVAL_BLOCK_HEADER), "header first")
    assert_true("OPENING" in block, "the opening survives")
    assert_true("ENDING" not in block, "the tail is gone")
    assert_true(_TRUNCATION_MARKER in block, "and it says so, rather than ending mid-word")

    # room = 1500 - 33 header - 4 (separator + its two newlines) = 1,463 tokens
    # of document. (Header was 32 before v3.1.3's +1 non-ASCII ceiling guard;
    # its em-dash pays it - same shift as the fixture pin above.) Asserted as
    # a property rather than as a character offset: A4's whole subject is
    # that there is no fixed chars-per-token to slice at.
    room = 1500 - _HEADER_TOKENS - _tokens(f"{_sep(7)}\n\n")
    assert_eq(room, 1463, "room left for the document, in tokens")
    body = block.split(f"{_sep(7)}\n", 1)[1][: -len("\n" + _TRUNCATION_MARKER)]
    assert_true(_tokens(body) <= room, f"kept body fits the room ({_tokens(body)})")
    # Maximal, not merely safe: a cap that keeps 10 tokens of a 2,258-token
    # exchange also "fits". The next character must not have fitted.
    assert_true(
        _tokens(doc[: len(body) + 1]) > room,
        "and it is the LONGEST prefix that fits — one more character does not",
    )

    # The marker is appended AFTER room is computed, so a truncated block runs
    # one marker over budget. Bounded and harmless, but it does mean this is the
    # one path where the block exceeds the cap.
    assert_true(
        _tokens(block) <= _truncated_ceiling(1500),
        f"overshoot bounded by the marker ({_tokens(block)} tokens)",
    )


def test_oversized_first_hit_with_no_room_returns_none():
    print("\n[test] retrieval cap — too little room to be worth truncating -> None")
    # A header with nothing under it is pure cost: it tells the model relevant
    # earlier exchanges exist and then shows it none of them. Below the
    # 200-char threshold the whole block must go away.
    ti = 7
    threshold = retrieval._MIN_TRUNCATED_TOKENS
    fixed = _HEADER_TOKENS + _tokens(f"{_sep(ti)}\n\n")
    tight = fixed + threshold        # leaves exactly the threshold — not enough
    loose = fixed + threshold + 1    # one token more — just enough

    prev = _with_budget(tight)
    try:
        room = tight - fixed
        assert_true(0 < room <= threshold, f"precondition: room={room}, at the threshold")
        assert_eq(
            retrieval.format_retrieval_block([_hit(ti, "a" * 4000)]),
            None,
            "no block at all, not a header-only block",
        )
    finally:
        retrieval.MAX_RETRIEVAL_TOKENS = prev

    print("\n[test] retrieval cap — just over the threshold, a stub is worth emitting")
    prev = _with_budget(loose)
    try:
        room = loose - fixed
        assert_true(room > threshold, f"precondition: room={room}, just over the threshold")
        block = retrieval.format_retrieval_block([_hit(ti, "a" * 4000)])
        assert_true(block is not None, "a stub is emitted")
        assert_true(_TRUNCATION_MARKER in block, "and marked as truncated")
    finally:
        retrieval.MAX_RETRIEVAL_TOKENS = prev


def test_cap_keeps_turn_order_and_does_not_reverse_it():
    print("\n[test] retrieval cap — survivors stay in ascending turn order")
    prev = _with_budget(1500)
    # Handed over newest-first, as a similarity ranking may well arrive. Two of
    # the three fit; the survivors must be the two OLDEST, still ascending —
    # the cap must not quietly reverse the chronology the header promises.
    hits = [
        _hit(90, "LATE" + "c" * 2776),
        _hit(10, "EARLY" + "a" * 2775),
        _hit(50, "MIDDLE" + "b" * 2774),
    ]
    try:
        block = retrieval.format_retrieval_block(hits)
    finally:
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    assert_true("EARLY" in block and "MIDDLE" in block, "the two oldest survive")
    assert_true("LATE" not in block, "the newest is the one shed")
    assert_true(block.index("EARLY") < block.index("MIDDLE"), "ascending, not reversed")
    assert_true(
        block.index("(turn ~10)") < block.index("(turn ~50)"),
        "separators agree with the bodies",
    )


# Same fixture as _probe_with_env's caller: five 400-char hits.
_ENV_PROBE = """
import json
import retrieval

hits = [{"turn_index": i, "document": "d" * 400} for i in range(5)]
block = retrieval.format_retrieval_block(hits)
print(json.dumps({
    "max": retrieval.MAX_RETRIEVAL_TOKENS,
    "chars": len(block) if block is not None else None,
}))
"""


def _probe_with_env(value):
    """Import retrieval in a CHILD process with COMPACTOR_MAX_RETRIEVAL_TOKENS
    set to `value` (None = unset) and report what the module resolved.

    A subprocess because the knob is read at import: this module imported
    retrieval at line 24, and re-importing it in-process would not re-read the
    environment. COMPACTOR_STORAGE_ROOT is inherited, so the child writes
    nothing outside this file's temp root.
    """
    env = dict(os.environ)
    if value is None:
        env.pop("COMPACTOR_MAX_RETRIEVAL_TOKENS", None)
    else:
        env["COMPACTOR_MAX_RETRIEVAL_TOKENS"] = value
    proc = subprocess.run(
        [sys.executable, "-c", _ENV_PROBE],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"FAIL env probe {value!r}: exit {proc.returncode}\n{proc.stderr.strip()}")
        sys.exit(1)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_budget_knob_is_read_from_the_environment():
    print("\n[test] retrieval cap — COMPACTOR_MAX_RETRIEVAL_TOKENS is honoured")
    hits = [_hit(i, "d" * 400) for i in range(5)]  # mirrors _ENV_PROBE

    default = _probe_with_env(None)
    assert_eq(default["max"], 1500, "unset -> the documented 1500 default")
    assert_eq(default["chars"], len(_uncapped_block(hits)), "all five hits fit at 1500")

    tight = _probe_with_env("200")
    assert_eq(tight["max"], 200, "the env value wins")
    # 200-token budget: 32 header + one 104-token hit = 136; a second reaches 240.
    assert_eq(_hit_cost(0, "d" * 400), 104, "fixture: 104 tokens per hit")
    assert_eq(tight["chars"], len(_uncapped_block(hits[:1])), "budget cut it to one hit")

    print("\n[test] retrieval cap — an empty value falls back to the default")
    assert_eq(_probe_with_env("")["max"], 1500, "'' -> 1500, not 0")

    print("\n[test] retrieval cap — 0 and negative values suppress injection")
    # Neither crashes, and neither reads as "unlimited": a budget of zero or
    # below cannot fit even the header, kept stays 0, and the block comes back
    # None. Injection is simply off — the only sane reading of "zero tokens of
    # retrieval", and the same end state as COMPACTOR_RAG_ENABLED=false.
    for value in ("0", "-1", "-1500"):
        probe = _probe_with_env(value)
        assert_eq(probe["max"], int(value), f"{value} parsed as given")
        assert_eq(probe["chars"], None, f"{value} -> no retrieval block injected")


def test_regression_20260827_three_exchanges_cannot_overflow_the_window():
    print("\n[test] retrieval cap — REGRESSION: the 2026-08-27 three-hit overflow")
    # The failing request at 23:51, reconstructed from the compactor log:
    #
    #     window                          32,768 tokens
    #     conversation after compaction    9,915  (76,104 before)
    #     facts                         <= 1,500  (COMPACTOR_MAX_FACTS_TOKENS)
    #     summary, L1=5                 <= 2,500  (COMPACTOR_L1_MAX_TOKENS * 5)
    #     3 retrieved exchanges          UNCAPPED
    #     -----------------------------------------
    #     vLLM saw                        33,127 -> 400
    #
    # Every other layer was already bounded; retrieval was the one that could
    # grow without limit, and three whole user+assistant pairs from a model
    # that writes at length is all it took. This is the test that fails if the
    # cap is ever removed, weakened, or made conditional.
    WINDOW = 32768
    COMPACTED_CONVERSATION = 9915
    CAPPED_FACTS = 1500
    CAPPED_SUMMARY = 2500

    def exchange(turn, chars):
        body = f"turn {turn}: " + (
            "she said the northern road was watched, and the watchers were patient. "
            * 500
        )
        return f"[user]: what happened on the northern road?\n[assistant]: {body}"[:chars]

    prev = _with_budget(1500)
    hits = [_hit(t, exchange(t, 25600)) for t in (11, 47, 88)]  # ~6,400 tokens each
    cap = _LogCapture()
    try:
        block = retrieval.format_retrieval_block(hits)
    finally:
        cap.stop()
        retrieval.MAX_RETRIEVAL_TOKENS = prev

    uncapped_tokens = _tokens(_uncapped_block(hits))
    capped_tokens = _tokens(block)
    before = COMPACTED_CONVERSATION + CAPPED_FACTS + CAPPED_SUMMARY + uncapped_tokens
    after = COMPACTED_CONVERSATION + CAPPED_FACTS + CAPPED_SUMMARY + capped_tokens

    assert_true(
        before > WINDOW,
        f"fixture reproduces the failure uncapped ({before} > {WINDOW} tokens)",
    )
    assert_true(
        capped_tokens <= _truncated_ceiling(1500),
        f"retrieval held to its budget ({capped_tokens} tokens, {len(block)} chars)",
    )
    assert_true(
        after <= WINDOW,
        f"the same request now fits the window ({after} <= {WINDOW} tokens)",
    )
    assert_true(
        any("of 3 exchange(s)" in m for m in cap.messages(logging.INFO)),
        "and the operator is told what was left out",
    )


# ---------------------------------------------------------------------------
# v3.1 A4 — the cap was denominated in characters, so it did not cap tokens
# ---------------------------------------------------------------------------
#
# `budget = MAX_RETRIEVAL_TOKENS * 4`, summed against `len(sep) + len(doc) + 2`,
# then logged as "within the 1500-TOKEN budget". Characters are the one unit
# that cannot see decoration, and this model decorates.
#
# The fixtures below are box-drawing and CJK on purpose. Prose is where every
# character-based estimate is right; a budget test written on prose passes and
# proves nothing, which is the same blind spot that let char/4 stand in for a
# tokenizer through two incidents (test_tokenizer_contract.py's opening
# docstring says so at length).

# INCIDENT_2026-08-28:35-37. One production assistant reply carried 1,710 U+2501
# and 441 U+2500. The same SHAPE as test_tokenizer_contract.DECORATIVE_REPLY,
# sized so that three of them fit the SHIPPED character budget of 6,000 with
# room to spare (3 * 1,955 + 119 = 5,984) — the teeth check below needs the old
# rule to admit all three, or it is not demonstrating what the old rule did.
_BOX_RULE = "━" * 1450 + "\n" + "─" * 450 + "\nHere is the table you asked for.\n"

# "Please look at the light of God." — three UTF-8 bytes per character, none of
# them ASCII, so the character count sees a third of what the encoder does.
_CJK = "神の光を見てください。" * 200


def _vllm_measured_tokens(text):
    """What vLLM was MEASURED to charge, reconstructed per character class.

    Not the module's estimate — a second opinion, so the test is not simply
    agreeing with the code under test. Both densities are measurements, not
    guesses: INCIDENT_2026-08-28 prices 2,151 box-drawing characters at ~4,275
    tokens (1.99 tokens/char), and main.count_tokens_exact's docstring records
    4.10 chars/token on this deployment's ASCII chat transcript.
    """
    ascii_chars = len(text.encode("ascii", "ignore"))
    return int(ascii_chars / 4.10 + (len(text) - ascii_chars) * 1.99)


def test_the_reconstruction_matches_the_incident():
    print("\n[test] A4 — the fixture's token density is the one production measured")
    # Anchor the second opinion before using it to judge anything.
    reply = "━" * 1710 + "\n" + "─" * 441
    assert_eq(len(reply) - len(reply.encode("ascii", "ignore")), 2151, "2,151 non-ASCII")
    measured = _vllm_measured_tokens(reply)
    assert_true(
        abs(measured - 4275) < 45, f"lands on the incident's ~4,275 tokens: {measured}"
    )
    # And this is the number the shipped cap could not see.
    assert_true(
        measured / max(1, len(reply) // 4) > 7.5,
        f"chars/4 undercounts it by >7.5x ({len(reply) // 4} vs {measured})",
    )


def test_the_budget_is_denominated_in_tokens_not_characters():
    print("\n[test] A4 — REGRESSION: a decorative block no longer blows the cap")
    prev = _with_budget(1500)
    hits = [_hit(t, _BOX_RULE) for t in (11, 47, 88)]
    try:
        # TEETH CHECK. Reproduce the SHIPPED rule — a character budget of
        # MAX * 4 — and show it admits all three exchanges, then price what it
        # admitted. A suite that only asserts the fixed path passes just as
        # happily on code that never had the fix.
        char_budget = 1500 * 4
        used, admitted = _HEADER_LEN, 0
        for h in sorted(hits, key=lambda x: x["turn_index"]):
            cost = _sep_len(h["turn_index"]) + len(h["document"]) + 2
            if used + cost > char_budget:
                break
            used += cost
            admitted += 1
        assert_eq(admitted, 3, "the character rule admitted every exchange")
        real = _vllm_measured_tokens(_uncapped_block(hits))
        assert_true(
            real > 1500 * 7,
            f"…and what it admitted really costs ~{real} tokens against 1,500",
        )

        block = retrieval.format_retrieval_block(hits)
    finally:
        retrieval.MAX_RETRIEVAL_TOKENS = prev

    # The fix. The estimate is a ceiling on decoration (one token per UTF-8
    # byte), so holding the block to the budget by that measure holds it by the
    # measured density too — which is the property that matters.
    assert_true(
        _tokens(block) <= _truncated_ceiling(1500),
        f"held to the token budget ({_tokens(block)} tokens)",
    )
    assert_true(
        _vllm_measured_tokens(block) <= 1500,
        f"and under it by the measured density too "
        f"({_vllm_measured_tokens(block)} tokens)",
    )


def test_cjk_is_capped_as_well_as_box_drawing():
    print("\n[test] A4 — the cap holds on CJK, which chars/4 also undercounts")
    prev = _with_budget(1500)
    try:
        # 2,200 characters, 6,600 UTF-8 bytes: the character rule prices this at
        # 550 tokens and admits four of them.
        assert_eq(len(_CJK) // 4, 550, "fixture: the character rule sees 550 tokens")
        block = retrieval.format_retrieval_block([_hit(t, _CJK) for t in (1, 2, 3, 4)])
    finally:
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    assert_true(
        _tokens(block) <= _truncated_ceiling(1500),
        f"held to the token budget ({_tokens(block)} tokens)",
    )
    assert_true(_TRUNCATION_MARKER in block, "one exchange, truncated — not four whole")


def test_prose_renders_exactly_as_it_did_before_the_unit_fix():
    print("\n[test] A4 — the common case does not shrink: ASCII is byte-identical")
    # The unit fix must not cost the layer its ordinary job. For pure ASCII the
    # new measure is the old one — chars/4 — so a prose block that fitted the
    # character budget still fits the token budget, byte for byte. If this ever
    # fails, the fix has started charging normal chat for decoration it has not
    # got.
    prose = (
        "The quick brown fox jumps over the lazy dog, and the dog, being lazy, "
        "does not object to this arrangement in the slightest degree. "
    ) * 14
    hits = [_hit(t, prose) for t in (10, 20, 30)]
    # 32 + 3 * 463 = 1,421 tokens: comfortably inside, so the cap must be
    # invisible here whatever unit it is denominated in.
    assert_eq(_hit_cost(10, prose), 463, "fixture: 463 tokens per prose exchange")
    prev = _with_budget(1500)
    try:
        block = retrieval.format_retrieval_block(hits)
    finally:
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    assert_eq(block, _uncapped_block(hits), "all three whole, byte-identical")
    assert_eq(
        _tokens(prose), len(prose) // 4, "pure ASCII is priced exactly as chars/4"
    )


def test_a_caller_supplied_counter_is_used_and_named():
    print("\n[test] A4 — an exact counter can be injected across the module boundary")
    # `main` imports `retrieval`, so `retrieval` cannot import
    # `main.count_tokens_exact`. The seam is a parameter instead. Nothing passes
    # one today; this test is what stops the seam rotting before it is wired.
    prev = _with_budget(600)
    hits = [_hit(10, "a" * 400), _hit(20, "b" * 400)]
    seen = []

    def counter(text):
        seen.append(text)
        return len(text)  # deliberately harsh: one token per character

    cap = _LogCapture()
    try:
        block = retrieval.format_retrieval_block(hits, count_tokens=counter)
    finally:
        cap.stop()
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    assert_true(seen, "the injected counter was actually consulted")
    # At one token per character a 600-token budget spends 119 on the header and
    # 420 on the first exchange, leaving no room for the second. The default
    # estimate would have fitted both twice over, so this result can only come
    # from the caller's measure actually being the one in force.
    assert_true(
        _HEADER_TOKENS + 2 * _hit_cost(10, "a" * 400) < 600,
        "teeth: the default estimate would have fitted both",
    )
    assert_true("a" * 400 in block, "the first exchange fits")
    assert_true("b" * 400 not in block, "the second does not, on the caller's measure")
    info = cap.messages(logging.INFO)
    assert_eq(len(info), 1, "one INFO line")
    assert_true(
        "caller-supplied counter" in info[0],
        f"and it names which counter decided: {info[0]!r}",
    )


def test_the_log_line_no_longer_reports_characters_as_tokens():
    print("\n[test] A4 — the dropped-hits line names its unit and its measure")
    prev = _with_budget(1500)
    hits = [_hit(t, _BOX_RULE) for t in (11, 47)]
    cap = _LogCapture()
    try:
        retrieval.format_retrieval_block(hits)
    finally:
        cap.stop()
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    info = cap.messages(logging.INFO)
    assert_eq(len(info), 1, "exactly one INFO line")
    assert_true("token(s) by the local estimate" in info[0], f"names the measure: {info[0]!r}")
    assert_true("COMPACTOR_MAX_RETRIEVAL_TOKENS" in info[0], "and the knob to raise")


if __name__ == "__main__":
    try:
        test_exchange_doc_format()
        test_doc_id_is_content_addressed()
        test_format_retrieval_block_empty()
        test_format_retrieval_block_orders_by_turn()

        test_degraded_index_returns_false()
        test_degraded_retrieve_returns_empty()
        test_degraded_forget_returns_zero()
        test_degraded_doc_count_returns_none_not_zero()
        test_doc_count_returns_none_on_query_failure()
        test_doc_count_zero_means_genuinely_empty()

        test_index_exchange_upserts()
        test_index_exchange_skips_empty()
        test_retrieve_returns_matches()
        test_retrieve_excludes_recent_turns()
        test_retrieve_empty_query()
        test_forget_conversation_deletes()

        test_deleted_message_does_not_overwrite_an_existing_exchange()
        test_identical_text_reindexed_is_idempotent()
        test_distinct_text_always_gets_a_row()
        test_sixteen_writes_at_one_turn_index_are_sixteen_rows()
        test_legacy_rows_coexist_and_the_ordinal_continues_past_them()
        test_a_request_ahead_of_the_store_pulls_the_ordinal_forward()
        test_import_keeps_the_bundle_turn_index_and_shares_identity_with_indexing()

        test_a_bounded_client_window_no_longer_suppresses_every_hit()
        test_the_out_of_frame_cutoff_says_so_in_the_log()
        test_a_short_conversation_keeps_its_filter()
        test_cutoff_zero_still_excludes_nothing()
        test_the_frame_check_is_only_paid_when_the_caller_lost_hits()

        test_the_mock_honours_n_results()
        test_a_filtered_slot_is_replaced_not_deleted()
        test_overfetch_is_bounded_and_trimmed_to_k()
        test_no_cutoff_means_no_overfetch()

        test_query_result_parsing_robustness()

        test_cap_invisible_when_it_does_not_bind()
        test_cap_drops_later_hits_and_says_so()
        test_oversized_first_hit_is_truncated_not_dropped()
        test_oversized_first_hit_with_no_room_returns_none()
        test_cap_keeps_turn_order_and_does_not_reverse_it()
        test_budget_knob_is_read_from_the_environment()
        test_regression_20260827_three_exchanges_cannot_overflow_the_window()

        test_the_reconstruction_matches_the_incident()
        test_the_budget_is_denominated_in_tokens_not_characters()
        test_cjk_is_capped_as_well_as_box_drawing()
        test_prose_renders_exactly_as_it_did_before_the_unit_fix()
        test_a_caller_supplied_counter_is_used_and_named()
        test_the_log_line_no_longer_reports_characters_as_tokens()

        print("\nAll retrieval smoke tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
