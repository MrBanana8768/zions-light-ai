"""
CPU-only smoke tests for compactor/retrieval.py (V2.0 Phase 3).

retrieval.py imports fastembed + chromadb lazily (inside _try_init), so this
test runs WITHOUT those heavy deps installed — it either exercises the
graceful-degradation path (deps unavailable → safe no-ops) or injects mock
embedder/collection objects to test the index/query/forget logic directly.

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
# format_retrieval_block now enforces one, as a CHARACTER budget of
# MAX_RETRIEVAL_TOKENS * 4 (the module has no tokenizer, deliberately).

# The literal appended to a truncated exchange. Mirrored from retrieval.py
# rather than imported — it is not exported, and a silent change to the words
# the model actually reads should fail a test.
_TRUNCATION_MARKER = "[...truncated to fit the retrieval budget]"

_HEADER_LEN = len(retrieval._RETRIEVAL_BLOCK_HEADER)


def _hit(turn_index, document, distance=0.1):
    return {"turn_index": turn_index, "document": document, "distance": distance}


def _sep_len(turn_index):
    """Length of the per-exchange separator, measured the way the module
    measures it."""
    return len(f"--- (turn ~{turn_index}) ---")


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
        # 119 header + 3 * (17 sep + 2 newlines + 300 doc) = 1076 of 6000 chars.
        block = retrieval.format_retrieval_block(hits)
    finally:
        cap.stop()
        retrieval.MAX_RETRIEVAL_TOKENS = prev
    assert_eq(block, _uncapped_block(hits), "byte-identical to the uncapped render")
    assert_eq(cap.messages(logging.INFO), [], "silent when nothing is dropped")


def test_cap_drops_later_hits_and_says_so():
    print("\n[test] retrieval cap — over budget: later hits drop, earlier survive whole")
    prev = _with_budget(1500)  # 6000-char budget
    # 119 + 2519 + 2519 = 5157 fits; a third 2519 would reach 7676.
    hits = [
        _hit(10, "EARLY" + "a" * 2495),
        _hit(20, "MIDDLE" + "b" * 2494),
        _hit(30, "LATE" + "c" * 2496),
    ]
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
    assert_true(len(block) <= 1500 * 4, f"block within budget ({len(block)} chars)")

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

    # room = 6000 - 119 header - 17 sep - 2 newlines = 5862 chars of document.
    room = 1500 * 4 - _HEADER_LEN - _sep_len(7) - 2
    assert_eq(
        len(block),
        _HEADER_LEN + 1 + _sep_len(7) + 1 + room + 1 + len(_TRUNCATION_MARKER),
        "cut to exactly the room left, plus the marker",
    )
    # The marker is appended AFTER room is computed, so a truncated block runs
    # one marker over budget (41 chars against 6000). Bounded and harmless, but
    # it does mean this is the one path where the block exceeds the cap.
    assert_true(
        len(block) <= 1500 * 4 + len(_TRUNCATION_MARKER) + 1,
        f"overshoot bounded by the marker ({len(block)} chars)",
    )


def test_oversized_first_hit_with_no_room_returns_none():
    print("\n[test] retrieval cap — too little room to be worth truncating -> None")
    # A header with nothing under it is pure cost: it tells the model relevant
    # earlier exchanges exist and then shows it none of them. Below the
    # 200-char threshold the whole block must go away.
    ti = 7
    fixed = _HEADER_LEN + _sep_len(ti) + 2
    tight = (fixed + 198) // 4              # leaves ~198 chars of room
    loose = -(-(fixed + 202) // 4)          # leaves ~202 — just over

    prev = _with_budget(tight)
    try:
        room = tight * 4 - fixed
        assert_true(0 < room <= 200, f"precondition: room={room}, at/under the threshold")
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
        room = loose * 4 - fixed
        assert_true(room > 200, f"precondition: room={room}, just over the threshold")
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
        _hit(90, "LATE" + "c" * 2496),
        _hit(10, "EARLY" + "a" * 2495),
        _hit(50, "MIDDLE" + "b" * 2494),
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
    # 800-char budget: header + one 419-char hit = 538; a second reaches 957.
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

    uncapped_tokens = len(_uncapped_block(hits)) // 4
    capped_tokens = len(block) // 4
    before = COMPACTED_CONVERSATION + CAPPED_FACTS + CAPPED_SUMMARY + uncapped_tokens
    after = COMPACTED_CONVERSATION + CAPPED_FACTS + CAPPED_SUMMARY + capped_tokens

    assert_true(
        before > WINDOW,
        f"fixture reproduces the failure uncapped ({before} > {WINDOW} tokens)",
    )
    assert_true(
        len(block) <= 1500 * 4 + len(_TRUNCATION_MARKER) + 1,
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


if __name__ == "__main__":
    try:
        test_exchange_doc_format()
        test_doc_id_stable()
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
        test_query_result_parsing_robustness()

        test_cap_invisible_when_it_does_not_bind()
        test_cap_drops_later_hits_and_says_so()
        test_oversized_first_hit_is_truncated_not_dropped()
        test_oversized_first_hit_with_no_room_returns_none()
        test_cap_keeps_turn_order_and_does_not_reverse_it()
        test_budget_knob_is_read_from_the_environment()
        test_regression_20260827_three_exchanges_cannot_overflow_the_window()

        print("\nAll retrieval smoke tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
