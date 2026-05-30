"""
CPU-only smoke tests for compactor/retrieval.py (V2.0 Phase 3).

retrieval.py imports fastembed + chromadb lazily (inside _try_init), so this
test runs WITHOUT those heavy deps installed — it either exercises the
graceful-degradation path (deps unavailable → safe no-ops) or injects mock
embedder/collection objects to test the index/query/forget logic directly.

Run:
    python test_retrieval.py
"""

import os
import shutil
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-retrieval-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

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

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, _id in enumerate(ids):
            self._store[_id] = (documents[i], metadatas[i])
        self.upserts.append({"ids": ids, "metadatas": metadatas})

    def query(self, query_embeddings, n_results, where):
        if self.canned_query is not None:
            return self.canned_query
        # Default: return everything matching the conv_id filter
        cid = where.get("conv_id")
        ids, docs, metas, dists = [], [], [], []
        for _id, (doc, meta) in self._store.items():
            if meta.get("conv_id") == cid:
                ids.append(_id); docs.append(doc); metas.append(meta); dists.append(0.1)
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}

    def get(self, where):
        cid = where.get("conv_id")
        ids = [i for i, (_, m) in self._store.items() if m.get("conv_id") == cid]
        return {"ids": ids}

    def delete(self, ids):
        for i in ids:
            self.deleted_ids.append(i)
            self._store.pop(i, None)


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


# ---------------------------------------------------------------------------
# Pure helpers (no backend needed)
# ---------------------------------------------------------------------------

def test_exchange_doc_format():
    print("\n[test] _exchange_doc renders canonical user/assistant text")
    doc = retrieval._exchange_doc("hello", "hi there")
    assert_true("[user]: hello" in doc, "user line present")
    assert_true("[assistant]: hi there" in doc, "assistant line present")


def test_doc_id_stable():
    print("\n[test] _doc_id is stable + unique per (conv, turn)")
    assert_eq(retrieval._doc_id("abc", 4), "abc::4", "id format")
    assert_true(retrieval._doc_id("abc", 4) != retrieval._doc_id("abc", 6), "distinct turns")


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


# ---------------------------------------------------------------------------
# Index / retrieve / forget with mock backends
# ---------------------------------------------------------------------------

def test_index_exchange_upserts():
    print("\n[test] index_exchange embeds + upserts with conv metadata")
    emb, col = _install_mocks()
    ok = retrieval.index_exchange("conv1", 4, "who is Lyra?", "a half-elf ranger")
    assert_eq(ok, True, "index succeeded")
    assert_eq(len(col.upserts), 1, "one upsert call")
    assert_eq(col.upserts[0]["ids"], ["conv1::4"], "correct doc id")
    assert_eq(col.upserts[0]["metadatas"][0]["conv_id"], "conv1", "conv_id in metadata")
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
    retrieval.index_exchange("conv1", 2, "old", "old-a")
    retrieval.index_exchange("conv1", 20, "recent", "recent-a")
    hits = retrieval.retrieve("conv1", "q", k=5, exclude_turns_from=10)
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


def test_query_result_parsing_robustness():
    print("\n[test] retrieve tolerates malformed/empty chroma responses")
    emb, col = _install_mocks()
    col.canned_query = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert_eq(retrieval.retrieve("c", "q"), [], "empty result lists -> []")
    col.canned_query = {}  # totally empty dict
    assert_eq(retrieval.retrieve("c", "q"), [], "empty dict -> [] (no crash)")


if __name__ == "__main__":
    try:
        test_exchange_doc_format()
        test_doc_id_stable()
        test_format_retrieval_block_empty()
        test_format_retrieval_block_orders_by_turn()

        test_degraded_index_returns_false()
        test_degraded_retrieve_returns_empty()
        test_degraded_forget_returns_zero()

        test_index_exchange_upserts()
        test_index_exchange_skips_empty()
        test_retrieve_returns_matches()
        test_retrieve_excludes_recent_turns()
        test_retrieve_empty_query()
        test_forget_conversation_deletes()
        test_query_result_parsing_robustness()

        print("\nAll retrieval smoke tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
