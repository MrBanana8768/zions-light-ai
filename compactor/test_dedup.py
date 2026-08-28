"""
CPU-only Tier-1 tests for compactor.dedup.

Mocks retrieval._embed to control which facts cluster together; mocks
the LLM HTTP call to control merge/KEEP decisions. Exercises the
hybrid pipeline end-to-end without touching ChromaDB or vLLM.

Run: python test_dedup.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_dedup_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # skip ChromaDB init in retrieval

import retrieval  # noqa: E402
import dedup  # noqa: E402


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


def _fact(text: str, added_turn: int = 0, last_used: int = 0) -> dict:
    return {"text": text, "added_turn": added_turn, "last_used": last_used}


def _mock_embed_returns(*vectors):
    """Patch retrieval._embed to return the given vectors in order."""
    return patch.object(retrieval, "_embed", lambda texts: list(vectors[:len(texts)]))


def _mock_chat_response(content: str, finish_reason: str | None = "stop"):
    """Build a MagicMock that mimics an httpx response with the given chat content.

    finish_reason defaults to "stop" because that is what vLLM actually
    sends; pass "length" for a truncated reply, or None to omit the key
    entirely (a backend that doesn't report it must not disable dedup).
    """
    choice = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    r = MagicMock(status_code=200)
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value={"choices": [choice]})
    return r


# ---------------------------------------------------------------------------
# Stage 1 — clustering
# ---------------------------------------------------------------------------

def test_cosine_basics():
    print("\n[test] _cosine math")
    assert_eq(round(dedup._cosine([1, 0], [1, 0]), 4), 1.0, "identical vectors → 1.0")
    assert_eq(round(dedup._cosine([1, 0], [0, 1]), 4), 0.0, "orthogonal → 0.0")
    assert_eq(round(dedup._cosine([1, 0], [-1, 0]), 4), -1.0, "opposite → -1.0")
    assert_eq(dedup._cosine([0, 0], [1, 1]), 0.0, "zero vector → 0.0 (no nan)")


def test_no_clusters_when_facts_dissimilar():
    print("\n[test] find_candidate_clusters: no clusters when all dissimilar")
    facts = [_fact("a"), _fact("b"), _fact("c")]
    # Orthogonal vectors → cosine = 0 → no clustering
    with _mock_embed_returns([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        clusters = dedup.find_candidate_clusters(facts)
    assert_eq(clusters, [], "no clusters")


def test_cluster_two_similar_facts():
    print("\n[test] find_candidate_clusters: two near-identical facts cluster")
    facts = [_fact("Lyra is half elf"), _fact("Lyra is a half-elf")]
    with _mock_embed_returns([0.9, 0.1], [0.95, 0.05]):
        clusters = dedup.find_candidate_clusters(facts)
    assert_eq(len(clusters), 1, "one cluster found")
    assert_eq(sorted(clusters[0]), [0, 1], "cluster contains both indices")


def test_singleton_facts_not_returned():
    print("\n[test] find_candidate_clusters: singletons not returned")
    facts = [_fact("alone"), _fact("paired-1"), _fact("paired-2")]
    # vec[0] orthogonal to others; vec[1] ~ vec[2]
    with _mock_embed_returns([1, 0, 0], [0, 0.9, 0], [0, 0.95, 0]):
        clusters = dedup.find_candidate_clusters(facts)
    assert_eq(len(clusters), 1, "only the pair returned, not the singleton")


def test_transitive_clustering():
    print("\n[test] find_candidate_clusters: A~B and B~C cluster even if A!~C")
    facts = [_fact("a"), _fact("b"), _fact("c")]
    # cos(a,b)=0.9, cos(b,c)=0.9, cos(a,c)~0.6 (below threshold)
    # All three should still cluster via transitive closure
    with _mock_embed_returns([1.0, 0.0], [0.8, 0.6], [0.6, 0.8]):
        clusters = dedup.find_candidate_clusters(facts, threshold=0.7)
    assert_eq(len(clusters), 1, "one transitive cluster")
    assert_eq(sorted(clusters[0]), [0, 1, 2], "all three included")


def test_split_cluster_never_exceeds_cap_or_orphans_a_fact():
    print("\n[test] _split_cluster: 2..MAX_CLUSTER_SIZE, every member kept")
    cap = dedup.MAX_CLUSTER_SIZE
    for n in range(2, 21):
        parts = dedup._split_cluster(list(range(n)))
        flat = [i for p in parts for i in p]
        assert_true(sorted(flat) == list(range(n)), f"n={n}: every index kept once")
        assert_true(max(len(p) for p in parts) <= cap, f"n={n}: no part over the cap")
        # A part of 1 has nothing to merge with and would be dropped by the
        # >=2 filter — that is a fact silently excluded from the pass.
        assert_true(min(len(p) for p in parts) >= 2, f"n={n}: no orphaned member")


def test_oversized_transitive_cluster_is_split():
    print("\n[test] find_candidate_clusters: V7 caps the transitive cluster")
    facts = [_fact(f"Lyra fact {i}") for i in range(9)]
    # All nine mutually similar (cos >= 0.99) — one transitive cluster of 9
    # before V7, which one LLM reply could have replaced wholesale.
    vecs = [[1.0, i * 0.01] for i in range(9)]
    with _mock_embed_returns(*vecs):
        clusters = dedup.find_candidate_clusters(facts)
    sizes = sorted(len(c) for c in clusters)
    assert_true(all(s <= dedup.MAX_CLUSTER_SIZE for s in sizes),
                f"no cluster over the cap (sizes={sizes})")
    assert_eq(len(clusters), 3, "9 facts → 3 sub-clusters")
    flat = sorted(i for c in clusters for i in c)
    assert_eq(flat, list(range(9)), "all nine facts still candidates")


def test_clustering_returns_empty_when_embedding_unavailable():
    print("\n[test] find_candidate_clusters: returns [] when embeddings fail")
    facts = [_fact("a"), _fact("b")]
    with patch.object(retrieval, "_embed", lambda texts: None):
        clusters = dedup.find_candidate_clusters(facts)
    assert_eq(clusters, [], "embeddings unavailable → no clusters")


def test_clustering_skips_facts_with_empty_text():
    print("\n[test] find_candidate_clusters: empty-text fact short-circuits")
    facts = [_fact("real"), _fact("")]  # empty text fact
    # Without the empty-text guard, "" would embed to something and might
    # cluster with anything else trivially.
    with patch.object(retrieval, "_embed",
                      lambda texts: [[1, 0], [0.5, 0.5]]):
        clusters = dedup.find_candidate_clusters(facts)
    # _embed_facts returns None on empty-text → no clusters
    assert_eq(clusters, [], "empty-text fact triggers safe no-op")


# ---------------------------------------------------------------------------
# Stage 2 — LLM merge call
# ---------------------------------------------------------------------------

def test_llm_merge_returns_text_on_merge():
    print("\n[test] llm_merge_candidate: returns merged text when LLM agrees")
    cluster = [_fact("Lyra is half elf"), _fact("Lyra is a half-elf")]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("Lyra is a half-elf"))
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    out = asyncio.run(go())
    assert_eq(out, ("Lyra is a half-elf", "merged"), "merged text returned")


def test_llm_merge_returns_none_on_keep():
    print("\n[test] llm_merge_candidate: returns None when LLM says KEEP")
    cluster = [
        _fact("user wants third-person past tense"),
        _fact("user wants first-person present tense"),
    ]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    out = asyncio.run(go())
    assert_eq(out, (None, "keep"), "KEEP → None, reason 'keep'")


def test_llm_merge_handles_keep_with_punctuation():
    print("\n[test] llm_merge_candidate: tolerates 'KEEP.' / 'Keep — different'")
    cluster = [_fact("x"), _fact("y")]

    async def go(resp):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(resp))
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    assert_eq(asyncio.run(go("KEEP.")), (None, "keep"), "KEEP. → None")
    assert_eq(asyncio.run(go("Keep — they are different")), (None, "keep"),
              "case+punct → None")


def test_llm_merge_handles_trailing_keep():
    print("\n[test] llm_merge_candidate: D39 — KEEP at the END is still KEEP")
    # The bug: head.startswith("KEEP") tolerated a leading KEEP and missed a
    # trailing one, so this response was treated as the merged canonical
    # fact and replaced every fact in the cluster with itself.
    cluster = [
        _fact("user wants third-person past tense"),
        _fact("user wants first-person present tense"),
    ]

    async def go(resp):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(resp))
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    assert_eq(asyncio.run(go("These are different, KEEP")), (None, "keep"),
              "trailing KEEP → None")
    assert_eq(asyncio.run(go("They say different things - keep")), (None, "keep"),
              "trailing lowercase keep → None")
    assert_eq(asyncio.run(go("2. KEEP")), (None, "keep"), "numbered KEEP → None")


def test_llm_merge_returns_none_on_truncated_response():
    print("\n[test] llm_merge_candidate: finish_reason 'length' → None")
    cluster = [
        _fact("Lyra is a half-elf ranger who lives in Aethermere"),
        _fact("Lyra is a half-elf ranger and prefers the northern road"),
    ]
    truncated = "Lyra is a half-elf ranger who lives in Aethermere and prefers"

    async def go(finish_reason):
        client = MagicMock()
        client.post = AsyncMock(
            return_value=_mock_chat_response(truncated, finish_reason=finish_reason)
        )
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    assert_eq(asyncio.run(go("length")), (None, "truncated"),
              "truncated merge rejected")
    # The same text is a fine merge when the model actually finished; the
    # guard must key on finish_reason, not on the content.
    assert_eq(asyncio.run(go("stop")), (truncated, "merged"),
              "completed merge accepted")
    assert_eq(asyncio.run(go(None)), (truncated, "merged"),
              "missing finish_reason is not evidence of truncation")


def test_llm_merge_returns_none_on_trivially_short_merge():
    print("\n[test] llm_merge_candidate: merge far shorter than its inputs → None")
    cluster = [
        _fact("Lyra is a half-elf ranger from Aethermere"),      # 40 chars
        _fact("Lyra Threadweaver is a half-elf ranger"),         # 37 chars
    ]

    async def go(resp):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(resp))
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    # 7 chars: clears the len<6 guard, so only the V7 length floor can
    # catch it. Collapsing two full facts to this is a summary, not a merge.
    assert_eq(asyncio.run(go("Lyra is")), (None, "collapsed"),
              "collapsed merge rejected")
    assert_eq(asyncio.run(go("Lyra Threadweaver is a half-elf ranger from Aethermere")),
              ("Lyra Threadweaver is a half-elf ranger from Aethermere", "merged"),
              "a real merge of the same cluster still passes")


def test_llm_merge_strips_bullet_prefix():
    print("\n[test] llm_merge_candidate: strips bullet/number prefix from merged text")
    cluster = [_fact("a"), _fact("b")]

    async def go(resp):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(resp))
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    assert_eq(asyncio.run(go("- merged fact")), ("merged fact", "merged"),
              "dash bullet stripped")
    assert_eq(asyncio.run(go("1. merged fact")), ("merged fact", "merged"),
              "numbered list stripped")


def test_llm_merge_returns_none_on_too_short_response():
    print("\n[test] llm_merge_candidate: rejects sub-6-char responses")
    cluster = [_fact("a"), _fact("b")]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("ok"))
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    assert_eq(asyncio.run(go()), (None, "short"), "too-short response → None")


def test_llm_merge_returns_none_on_network_failure():
    print("\n[test] llm_merge_candidate: network failure → None (cluster preserved)")
    cluster = [_fact("a"), _fact("b")]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(side_effect=ConnectionError("boom"))
        return await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)

    # The reason matters as much as the None: "error" is the one refusal
    # the memo must not remember, or a single vLLM blip pins a mergeable
    # cluster shut for the life of the process.
    assert_eq(asyncio.run(go()), (None, "error"), "exception caught, cluster preserved")


def test_llm_merge_returns_none_on_single_fact_cluster():
    print("\n[test] llm_merge_candidate: <2 facts is a no-op (no LLM call)")
    cluster = [_fact("only one")]

    async def go():
        client = MagicMock()
        client.post = AsyncMock()  # would fail loudly if called
        out = await dedup.llm_merge_candidate(client, "http://fake", "m", cluster)
        client.post.assert_not_called()
        return out

    assert_eq(asyncio.run(go()), (None, "singleton"),
              "singleton → no call, returns None")


# ---------------------------------------------------------------------------
# _merge_metadata
# ---------------------------------------------------------------------------

def test_merge_metadata_preserves_oldest_added_turn():
    print("\n[test] _merge_metadata: added_turn = min, last_used = max")
    cluster = [
        _fact("a", added_turn=5, last_used=10),
        _fact("b", added_turn=2, last_used=15),
        _fact("c", added_turn=8, last_used=12),
    ]
    out = dedup._merge_metadata(cluster, "merged")
    assert_eq(out["text"], "merged", "text preserved")
    assert_eq(out["added_turn"], 2, "added_turn = min (earliest origin)")
    assert_eq(out["last_used"], 15, "last_used = max (most recent)")


# ---------------------------------------------------------------------------
# End-to-end dedup_facts
# ---------------------------------------------------------------------------

def test_dedup_facts_short_circuits_when_lt_2():
    print("\n[test] dedup_facts: <2 facts → no-op, no embedding call")
    async def go():
        client = MagicMock()
        with patch.object(retrieval, "_embed",
                          MagicMock(side_effect=AssertionError("should not call"))):
            out, removed = await dedup.dedup_facts(client, "http://x", "m", [])
            assert removed == 0
            out2, r2 = await dedup.dedup_facts(client, "http://x", "m", [_fact("solo")])
            assert r2 == 0
            return out, out2

    a, b = asyncio.run(go())
    assert_eq(a, [], "empty in → empty out")
    assert_eq(len(b), 1, "single fact unchanged")


def test_dedup_facts_merges_when_llm_agrees():
    print("\n[test] dedup_facts: cluster found + LLM merges → fact list shrinks")
    facts = [
        _fact("Lyra is half elf", added_turn=1, last_used=10),
        _fact("Lyra is a half-elf", added_turn=2, last_used=15),
        _fact("the protagonist is named Lyra Threadweaver", added_turn=3, last_used=20),
    ]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(
            "Lyra Threadweaver is a half-elf protagonist"
        ))
        # First two facts cluster, third is alone
        with _mock_embed_returns([0.9, 0.1, 0.0], [0.95, 0.1, 0.0], [0.0, 0.0, 1.0]):
            return await dedup.dedup_facts(client, "http://x", "m", facts)

    deduped, removed = asyncio.run(go())
    assert_eq(removed, 1, "1 fact removed (3 → 2)")
    assert_eq(len(deduped), 2, "two facts remain")
    # The merged fact should carry min added_turn from the cluster (1)
    merged = [f for f in deduped if "half-elf" in f["text"]]
    assert_eq(len(merged), 1, "exactly one merged fact contains 'half-elf'")
    assert_eq(merged[0]["added_turn"], 1, "merged fact carries earliest added_turn")


def test_dedup_facts_keeps_all_when_llm_says_keep():
    print("\n[test] dedup_facts: cluster found but LLM says KEEP → all preserved")
    facts = [
        _fact("user wants third-person past", added_turn=1, last_used=10),
        _fact("user wants first-person present", added_turn=2, last_used=15),
    ]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with _mock_embed_returns([0.9, 0.1], [0.85, 0.15]):
            return await dedup.dedup_facts(client, "http://x", "m", facts)

    deduped, removed = asyncio.run(go())
    assert_eq(removed, 0, "0 removed when LLM says KEEP")
    assert_eq(len(deduped), 2, "both facts preserved")


def test_dedup_facts_no_op_when_no_clusters():
    print("\n[test] dedup_facts: no candidate clusters → 0 LLM calls")
    facts = [_fact("a"), _fact("b")]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        # Orthogonal vectors → no clusters
        with _mock_embed_returns([1, 0], [0, 1]):
            out, removed = await dedup.dedup_facts(client, "http://x", "m", facts)
            client.post.assert_not_called()
            return out, removed

    out, removed = asyncio.run(go())
    assert_eq(removed, 0, "no removals")
    assert_eq(len(out), 2, "all facts kept")


def test_dedup_facts_returns_input_on_clustering_failure():
    print("\n[test] dedup_facts: clustering raises → input returned unchanged")
    facts = [_fact("a"), _fact("b")]

    async def go():
        client = MagicMock()
        with patch.object(dedup, "find_candidate_clusters",
                          MagicMock(side_effect=RuntimeError("boom"))):
            return await dedup.dedup_facts(client, "http://x", "m", facts)

    out, removed = asyncio.run(go())
    assert_eq(removed, 0, "failure → 0 removed")
    assert_eq(len(out), 2, "facts unchanged")


def test_dedup_facts_respects_max_llm_calls_cap():
    print("\n[test] dedup_facts: hits LLM call cap → remaining clusters deferred")
    # 4 pairs, each its own cluster of 2 = 4 candidate clusters
    facts = [
        _fact("a1"), _fact("a2"),
        _fact("b1"), _fact("b2"),
        _fact("c1"), _fact("c2"),
        _fact("d1"), _fact("d2"),
    ]
    # Make pairs cluster: (0,1), (2,3), (4,5), (6,7)
    vecs = [
        [1, 0, 0, 0], [1, 0.05, 0, 0],
        [0, 1, 0, 0], [0, 1.05, 0, 0],
        [0, 0, 1, 0], [0, 0, 1.05, 0],
        [0, 0, 0, 1], [0, 0, 0, 1.05],
    ]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("merged"))
        with patch.object(retrieval, "_embed", lambda texts: vecs), \
             patch.object(dedup, "MAX_LLM_CALLS_PER_PASS", 2):
            out, removed = await dedup.dedup_facts(client, "http://x", "m", facts)
            return out, removed, client.post.call_count

    out, removed, call_count = asyncio.run(go())
    assert_eq(call_count, 2, "LLM called exactly 2 times (cap honored)")
    # 2 clusters merged (each 2 facts → 1) = 2 facts removed total
    assert_eq(removed, 2, "2 facts removed in this pass")
    # 4 remaining (untouched clusters) + 2 merged = 6 facts
    assert_eq(len(out), 6, "remaining clusters preserved for next pass")


def test_dedup_facts_bounds_what_one_llm_reply_can_replace():
    print("\n[test] dedup_facts: V7 — one LLM reply can't replace ten facts")
    # Ten mutually similar facts. Before V7 this was one cluster, one call,
    # and one <=60-token answer replacing all ten.
    facts = [_fact(f"Lyra detail number {i}", added_turn=i) for i in range(10)]
    vecs = [[1.0, i * 0.01] for i in range(10)]
    seen_sizes = []
    real_merge = dedup.llm_merge_candidate

    async def spy(client, vllm_url, model, cluster_facts):
        seen_sizes.append(len(cluster_facts))
        return await real_merge(client, vllm_url, model, cluster_facts)

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(
            "Lyra has several established details"
        ))
        with patch.object(retrieval, "_embed", lambda texts: vecs), \
             patch.object(dedup, "llm_merge_candidate", spy):
            return await dedup.dedup_facts(client, "http://x", "m", facts)

    out, removed = asyncio.run(go())
    assert_true(max(seen_sizes) <= dedup.MAX_CLUSTER_SIZE,
                f"no reply saw more than the cap (sizes={seen_sizes})")
    assert_eq(sum(seen_sizes), 10, "every fact was still a merge candidate")
    assert_eq(len(out), len(seen_sizes), "one canonical fact per sub-cluster")
    assert_eq(removed, 10 - len(seen_sizes), "removed count matches")


def test_dedup_facts_logs_removed_texts_for_forensics():
    print("\n[test] dedup_facts: removed fact texts are logged at INFO")
    # The facts file is overwritten by the caller and there is no tombstone,
    # so this log line is the only remaining copy of what a merge destroyed.
    facts = [
        _fact("Lyra is a half-elf ranger from Aethermere", added_turn=1),
        _fact("Lyra Threadweaver is a half-elf ranger", added_turn=2),
    ]

    async def go(log):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(
            "Lyra Threadweaver is a half-elf ranger from Aethermere"
        ))
        with _mock_embed_returns([1.0, 0.0], [1.0, 0.01]), \
             patch.object(dedup, "logger", log):
            return await dedup.dedup_facts(client, "http://x", "m", facts)

    log = MagicMock()
    _out, removed = asyncio.run(go(log))
    assert_eq(removed, 1, "one fact removed")
    logged = " ".join(str(c) for c in log.info.call_args_list)
    for f in facts:
        assert_true(f["text"] in logged,
                    f"removed text recoverable from the log: {f['text']!r}")
    # The productive case reports the same counters as the silent one.
    line = _pass_lines(log)[0]
    assert_true("1 LLM call(s)" in line, "pass line counts the call")
    assert_true("1 merge(s)" in line, "pass line counts the merge")
    assert_true("1 fact(s) removed" in line, "pass line counts the removal")


def test_dedup_facts_preserves_cluster_on_trailing_keep():
    print("\n[test] dedup_facts: D39 end-to-end — trailing KEEP loses nothing")
    facts = [
        _fact("user wants third-person past", added_turn=1),
        _fact("user wants first-person present", added_turn=2),
        _fact("user wants second-person future", added_turn=3),
    ]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(
            "These are different, KEEP"
        ))
        with _mock_embed_returns([1.0, 0.0], [1.0, 0.01], [1.0, 0.02]):
            return await dedup.dedup_facts(client, "http://x", "m", facts)

    out, removed = asyncio.run(go())
    assert_eq(removed, 0, "0 removed on a trailing KEEP")
    assert_eq(sorted(f["text"] for f in out),
              sorted(f["text"] for f in facts),
              "all three original facts survive verbatim")


def test_dedup_facts_preserves_cluster_on_truncated_merge():
    print("\n[test] dedup_facts: truncated merge end-to-end → nothing removed")
    facts = [
        _fact("Lyra is a half-elf ranger who lives in Aethermere", added_turn=1),
        _fact("Lyra is a half-elf ranger and prefers the north road", added_turn=2),
    ]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response(
            "Lyra is a half-elf ranger who lives in Aethermere and prefers",
            finish_reason="length",
        ))
        with _mock_embed_returns([1.0, 0.0], [1.0, 0.01]):
            return await dedup.dedup_facts(client, "http://x", "m", facts)

    out, removed = asyncio.run(go())
    assert_eq(removed, 0, "0 removed when the merge was cut off")
    assert_eq(len(out), 2, "both facts survive")


# ---------------------------------------------------------------------------
# I-6 — every pass logs
# ---------------------------------------------------------------------------

def _pass_lines(log) -> list[str]:
    """The per-pass summary lines emitted at INFO."""
    return [str(c) for c in log.info.call_args_list if "dedup pass:" in str(c)]


def test_pass_with_zero_merges_still_logs():
    print("\n[test] dedup_facts: a pass that merged nothing still logs")
    # The 19h47m window: 197 of 301 vLLM calls were dedup, and roughly 23
    # of ~30 passes logged nothing at all, because the old summary line
    # was inside `if removed > 0`. A pass that spends calls and merges
    # nothing is the expensive case; it is the one that must be visible.
    facts = [
        _fact("user wants third-person past", added_turn=1),
        _fact("user wants first-person present", added_turn=2),
    ]

    async def go(log):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with _mock_embed_returns([1.0, 0.0], [1.0, 0.01]), \
             patch.object(dedup, "logger", log):
            return await dedup.dedup_facts(
                client, "http://x", "m", facts, conv_id="c-log"
            )

    log = MagicMock()
    _out, removed = asyncio.run(go(log))
    assert_eq(removed, 0, "nothing merged")
    lines = _pass_lines(log)
    assert_eq(len(lines), 1, "exactly one pass line")
    line = lines[0]
    assert_true("conv=c-log" in line, "pass line names the conversation")
    assert_true("2 fact(s) in" in line, "input size reported")
    assert_true("1 candidate cluster(s)" in line, "clusters considered reported")
    assert_true("1 LLM call(s)" in line, "LLM calls spent reported")
    assert_true("0 merge(s)" in line, "merges made reported")


def test_every_return_path_logs_a_pass():
    print("\n[test] dedup_facts: no return path is silent")
    # Each of these used to leave no trace whatsoever. They are cheap, but
    # "the cheap case is the one that logs" was the bug — the guarantee
    # worth having is that a pass is never silent, not that a costly one
    # is loud.
    async def go(log, facts, vecs):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with patch.object(retrieval, "_embed", lambda texts: vecs), \
             patch.object(dedup, "logger", log):
            return await dedup.dedup_facts(
                client, "http://x", "m", facts, conv_id="c-paths"
            )

    log = MagicMock()
    asyncio.run(go(log, [], []))
    assert_eq(len(_pass_lines(log)), 1, "empty input logs a pass")

    log = MagicMock()
    asyncio.run(go(log, [_fact("solo")], [[1, 0]]))
    assert_eq(len(_pass_lines(log)), 1, "single-fact input logs a pass")

    log = MagicMock()
    asyncio.run(go(log, [_fact("a"), _fact("b")], [[1, 0], [0, 1]]))
    lines = _pass_lines(log)
    assert_eq(len(lines), 1, "no-candidate-clusters pass logs")
    assert_true("0 candidate cluster(s)" in lines[0], "reports zero clusters")

    log = MagicMock()
    with patch.object(dedup, "find_candidate_clusters",
                      MagicMock(side_effect=RuntimeError("boom"))):
        asyncio.run(go(log, [_fact("a"), _fact("b")], [[1, 0], [0, 1]]))
    assert_eq(len(_pass_lines(log)), 1, "clustering failure logs a pass")


def test_pass_is_logged_even_when_the_pass_raises():
    print("\n[test] dedup_facts: an unexpected raise still leaves a pass line")
    # The wrapper logs in `finally` precisely so that a future edit to the
    # pass body cannot reintroduce a silent path.
    async def go(log):
        client = MagicMock()
        with patch.object(dedup, "_dedup_pass",
                          AsyncMock(side_effect=RuntimeError("boom"))), \
             patch.object(dedup, "logger", log):
            try:
                await dedup.dedup_facts(
                    client, "http://x", "m", [_fact("a"), _fact("b")],
                    conv_id="c-raise",
                )
            except RuntimeError:
                return "raised"
        return "swallowed"

    log = MagicMock()
    assert_eq(asyncio.run(go(log)), "raised", "the exception still propagates")
    assert_eq(len(_pass_lines(log)), 1, "and the pass was still reported")


def test_pass_line_says_when_the_memo_is_off():
    print("\n[test] dedup_facts: a conv_id-less caller is named in the log")
    # An un-updated caller gets no memoisation, which is the difference
    # between a handful of LLM calls a day and a few hundred. That must be
    # visible in the log rather than inferred from the call volume.
    async def go(log, **kw):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with _mock_embed_returns([1.0, 0.0], [1.0, 0.01]), \
             patch.object(dedup, "logger", log):
            return await dedup.dedup_facts(
                client, "http://x", "m", [_fact("aaa"), _fact("aab")], **kw
            )

    log = MagicMock()
    asyncio.run(go(log))
    assert_true("refusal memo off" in _pass_lines(log)[0],
                "no conv_id → the line says the memo is off")

    log = MagicMock()
    asyncio.run(go(log, conv_id="c-on"))
    assert_true("refusal memo off" not in _pass_lines(log)[0],
                "with a conv_id the marker is absent")


# ---------------------------------------------------------------------------
# I-6 — refusal memo
# ---------------------------------------------------------------------------

_KEEP_PAIR = [
    _fact("user wants third-person past", added_turn=1),
    _fact("user wants first-person present", added_turn=2),
]
_KEEP_PAIR_VECS = [[1.0, 0.0], [1.0, 0.01]]


async def _two_passes(client, facts, vecs, conv_ids):
    """Run dedup once per conv_id in order, returning post-call counts."""
    counts = []
    with patch.object(retrieval, "_embed", lambda texts: vecs[:len(texts)]):
        for cid in conv_ids:
            await dedup.dedup_facts(client, "http://x", "m", facts, conv_id=cid)
            counts.append(client.post.call_count)
    return counts


def test_refused_cluster_is_not_resent_on_the_next_pass():
    print("\n[test] dedup_facts: I-6 — a KEEP is bought once, not every turn")

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        counts = await _two_passes(
            client, _KEEP_PAIR, _KEEP_PAIR_VECS, ["c1", "c1", "c1"]
        )
        return counts

    counts = asyncio.run(go())
    assert_eq(counts[0], 1, "first pass asks the model")
    assert_eq(counts[1], 1, "second pass does not ask again")
    assert_eq(counts[2], 1, "and neither does the third")


def test_refusal_memo_survives_but_facts_are_still_returned_intact():
    print("\n[test] dedup_facts: a memoised skip preserves the cluster")
    # A memo hit must behave exactly like the KEEP it stands in for: the
    # facts come back, all of them, unchanged.
    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with patch.object(retrieval, "_embed", lambda texts: _KEEP_PAIR_VECS):
            await dedup.dedup_facts(
                client, "http://x", "m", _KEEP_PAIR, conv_id="c-intact"
            )
            return await dedup.dedup_facts(
                client, "http://x", "m", _KEEP_PAIR, conv_id="c-intact"
            )

    out, removed = asyncio.run(go())
    assert_eq(removed, 0, "0 removed on the memoised pass")
    assert_eq(sorted(f["text"] for f in out),
              sorted(f["text"] for f in _KEEP_PAIR),
              "both facts survive the skip verbatim")


def test_refusal_memo_does_not_leak_across_conv_ids():
    print("\n[test] dedup_facts: one conversation's KEEP is not another's")
    # Facts are per-conversation; a decision about one conversation's facts
    # is not evidence about another's, and this codebase's expensive
    # failures have all been cross-conversation bleed.
    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        return await _two_passes(
            client, _KEEP_PAIR, _KEEP_PAIR_VECS, ["conv-a", "conv-b"]
        )

    counts = asyncio.run(go())
    assert_eq(counts[0], 1, "conv-a asks")
    assert_eq(counts[1], 2, "conv-b asks for itself, memo not shared")


def test_backend_failure_is_not_memoised():
    print("\n[test] dedup_facts: a timeout is not a decision")
    # "error" is excluded from _MEMOISABLE_REFUSALS deliberately: if a
    # failed call were remembered, one bad minute on vLLM would pin a
    # mergeable cluster shut for the life of the process.
    async def go():
        client = MagicMock()
        client.post = AsyncMock(side_effect=ConnectionError("vllm down"))
        return await _two_passes(
            client, _KEEP_PAIR, _KEEP_PAIR_VECS, ["c-err", "c-err"]
        )

    counts = asyncio.run(go())
    assert_eq(counts[0], 1, "first pass tries")
    assert_eq(counts[1], 2, "second pass retries — the cluster is not condemned")


def test_truncated_and_collapsed_refusals_are_memoised():
    print("\n[test] dedup_facts: a deterministic bad reply is not re-bought")
    # temperature is 0.0: the same cluster produces the same truncated or
    # collapsed reply every time, so re-sending it buys the same refusal.
    long_pair = [
        _fact("Lyra is a half-elf ranger who lives in Aethermere", added_turn=1),
        _fact("Lyra is a half-elf ranger and prefers the north road", added_turn=2),
    ]

    async def go(resp, finish_reason):
        client = MagicMock()
        client.post = AsyncMock(
            return_value=_mock_chat_response(resp, finish_reason=finish_reason)
        )
        return await _two_passes(
            client, long_pair, _KEEP_PAIR_VECS, ["c-t", "c-t"]
        )

    counts = asyncio.run(go(
        "Lyra is a half-elf ranger who lives in Aethermere and prefers", "length"
    ))
    assert_eq(counts, [1, 1], "truncated reply memoised")

    dedup.reset_refusal_memo()
    counts = asyncio.run(go("Lyra.", "stop"))
    assert_eq(counts, [1, 1], "collapsed reply memoised")


def test_memo_key_changes_when_the_cluster_does():
    print("\n[test] dedup_facts: a cluster that gained a fact is a new question")
    # The memo is keyed on the whole cluster, not on pairs, because "are
    # these two the same?" and "are these three the same?" are different
    # questions and the second one deserves its own call.
    pair = list(_KEEP_PAIR)
    trio = pair + [_fact("user wants second-person future", added_turn=3)]

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with patch.object(retrieval, "_embed",
                          lambda texts: [[1.0, i * 0.01] for i in range(len(texts))]):
            await dedup.dedup_facts(client, "http://x", "m", pair, conv_id="c-grow")
            first = client.post.call_count
            await dedup.dedup_facts(client, "http://x", "m", trio, conv_id="c-grow")
            return first, client.post.call_count

    first, second = asyncio.run(go())
    assert_eq(first, 1, "the pair was asked about")
    assert_eq(second, 2, "the trio is asked about too, not memo-skipped")


def test_memo_skip_does_not_spend_the_call_cap():
    print("\n[test] dedup_facts: the cap reaches new clusters, not old refusals")
    # Clusters are offered to the cap oldest-first, so without the memo the
    # same first N refused clusters spend the entire budget every pass and
    # a cluster involving a newly added fact is never looked at. This is
    # the starvation half of I-6.
    facts = []
    vecs = []
    for k in range(6):
        facts.append(_fact(f"g{k}-one"))
        facts.append(_fact(f"g{k}-two"))
        base = [0.0] * 6
        base[k] = 1.0
        vecs.append(list(base))
        near = list(base)
        near[k] = 1.05
        vecs.append(near)

    asked: list[str] = []
    real_merge = dedup.llm_merge_candidate

    async def spy(client, vllm_url, model, cluster_facts):
        asked.append(cluster_facts[0]["text"])
        return await real_merge(client, vllm_url, model, cluster_facts)

    async def go():
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with patch.object(retrieval, "_embed", lambda texts: vecs), \
             patch.object(dedup, "llm_merge_candidate", spy), \
             patch.object(dedup, "MAX_LLM_CALLS_PER_PASS", 2):
            for _ in range(3):
                await dedup.dedup_facts(
                    client, "http://x", "m", facts, conv_id="c-cap"
                )

    asyncio.run(go())
    assert_eq(asked, ["g0-one", "g1-one", "g2-one",
                      "g3-one", "g4-one", "g5-one"],
              "three capped passes worked through all six clusters, none twice")


def test_pass_line_reports_memo_skips_and_deferrals():
    print("\n[test] dedup_facts: skips and deferrals are counted in the log")
    facts = []
    vecs = []
    for k in range(3):
        facts.append(_fact(f"h{k}-one"))
        facts.append(_fact(f"h{k}-two"))
        base = [0.0] * 3
        base[k] = 1.0
        vecs.append(list(base))
        near = list(base)
        near[k] = 1.05
        vecs.append(near)

    async def go(log):
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with patch.object(retrieval, "_embed", lambda texts: vecs), \
             patch.object(dedup, "MAX_LLM_CALLS_PER_PASS", 1), \
             patch.object(dedup, "logger", log):
            await dedup.dedup_facts(client, "http://x", "m", facts, conv_id="c-rep")
            await dedup.dedup_facts(client, "http://x", "m", facts, conv_id="c-rep")

    log = MagicMock()
    asyncio.run(go(log))
    first, second = _pass_lines(log)
    assert_true("0 already-refused skip(s)" in first, "nothing memoised yet")
    assert_true("2 cluster(s) deferred" in first, "two clusters deferred by the cap")
    assert_true("1 already-refused skip(s)" in second, "the first refusal is skipped")
    assert_true("1 cluster(s) deferred" in second,
                "a memo skip is not counted as a deferral")


def test_cluster_key_is_order_independent_and_collision_resistant():
    print("\n[test] _cluster_key: same facts → same key, different facts → not")
    ab = dedup._cluster_key([_fact("alpha"), _fact("beta")])
    ba = dedup._cluster_key([_fact("beta"), _fact("alpha")])
    assert_eq(ab, ba, "order does not change the key")
    assert_true(ab != dedup._cluster_key([_fact("alpha"), _fact("gamma")]),
                "a different member changes the key")
    # The NUL join: without a separator these two clusters would hash the
    # same and one would inherit the other's refusal.
    assert_true(dedup._cluster_key([_fact("ab"), _fact("c")]) !=
                dedup._cluster_key([_fact("a"), _fact("bc")]),
                "concatenation ambiguity does not collide")


def test_memo_is_bounded_in_both_dimensions():
    print("\n[test] refusal memo: LRU-bounded per conversation and overall")
    # Forgetting a refusal costs one LLM call and never costs a fact, so
    # eviction can be crude — but it has to happen.
    with patch.object(dedup, "MEMO_MAX_CLUSTERS_PER_CONV", 3):
        for k in range(5):
            dedup._memo_put("c-bound", f"key{k}", "keep")
        assert_eq(len(dedup._REFUSAL_MEMO["c-bound"]), 3, "per-conv cap holds")
        assert_eq(dedup._memo_get("c-bound", "key0"), None, "oldest evicted")
        assert_eq(dedup._memo_get("c-bound", "key4"), "keep", "newest retained")

    dedup.reset_refusal_memo()
    with patch.object(dedup, "MEMO_MAX_CONVS", 2):
        for k in range(4):
            dedup._memo_put(f"conv{k}", "key", "keep")
        assert_eq(len(dedup._REFUSAL_MEMO), 2, "conversation cap holds")
        assert_eq(dedup._memo_get("conv0", "key"), None, "oldest conv evicted")
        assert_eq(dedup._memo_get("conv3", "key"), "keep", "newest conv retained")


def test_memo_is_a_no_op_without_a_conv_id():
    print("\n[test] refusal memo: no conv_id → no entry, no shared bucket")
    dedup._memo_put(None, "key", "keep")
    dedup._memo_put("", "key", "keep")
    assert_eq(len(dedup._REFUSAL_MEMO), 0, "nothing stored under a null key")
    assert_eq(dedup._memo_get(None, "key"), None, "and nothing retrievable")


def test_reset_refusal_memo_scopes_correctly():
    print("\n[test] reset_refusal_memo: one conversation, or all of them")
    dedup._memo_put("r1", "k", "keep")
    dedup._memo_put("r2", "k", "keep")
    dedup.reset_refusal_memo("r1")
    assert_eq(dedup._memo_get("r1", "k"), None, "named conv cleared")
    assert_eq(dedup._memo_get("r2", "k"), "keep", "the other is untouched")
    dedup.reset_refusal_memo()
    assert_eq(len(dedup._REFUSAL_MEMO), 0, "no argument clears everything")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_cosine_basics,
        test_no_clusters_when_facts_dissimilar,
        test_cluster_two_similar_facts,
        test_singleton_facts_not_returned,
        test_transitive_clustering,
        test_split_cluster_never_exceeds_cap_or_orphans_a_fact,
        test_oversized_transitive_cluster_is_split,
        test_clustering_returns_empty_when_embedding_unavailable,
        test_clustering_skips_facts_with_empty_text,
        test_llm_merge_returns_text_on_merge,
        test_llm_merge_returns_none_on_keep,
        test_llm_merge_handles_keep_with_punctuation,
        test_llm_merge_handles_trailing_keep,
        test_llm_merge_returns_none_on_truncated_response,
        test_llm_merge_returns_none_on_trivially_short_merge,
        test_llm_merge_strips_bullet_prefix,
        test_llm_merge_returns_none_on_too_short_response,
        test_llm_merge_returns_none_on_network_failure,
        test_llm_merge_returns_none_on_single_fact_cluster,
        test_merge_metadata_preserves_oldest_added_turn,
        test_dedup_facts_short_circuits_when_lt_2,
        test_dedup_facts_merges_when_llm_agrees,
        test_dedup_facts_keeps_all_when_llm_says_keep,
        test_dedup_facts_no_op_when_no_clusters,
        test_dedup_facts_returns_input_on_clustering_failure,
        test_dedup_facts_respects_max_llm_calls_cap,
        test_dedup_facts_bounds_what_one_llm_reply_can_replace,
        test_dedup_facts_logs_removed_texts_for_forensics,
        test_dedup_facts_preserves_cluster_on_trailing_keep,
        test_dedup_facts_preserves_cluster_on_truncated_merge,
        test_pass_with_zero_merges_still_logs,
        test_every_return_path_logs_a_pass,
        test_pass_is_logged_even_when_the_pass_raises,
        test_pass_line_says_when_the_memo_is_off,
        test_refused_cluster_is_not_resent_on_the_next_pass,
        test_refusal_memo_survives_but_facts_are_still_returned_intact,
        test_refusal_memo_does_not_leak_across_conv_ids,
        test_backend_failure_is_not_memoised,
        test_truncated_and_collapsed_refusals_are_memoised,
        test_memo_key_changes_when_the_cluster_does,
        test_memo_skip_does_not_spend_the_call_cap,
        test_pass_line_reports_memo_skips_and_deferrals,
        test_cluster_key_is_order_independent_and_collision_resistant,
        test_memo_is_bounded_in_both_dimensions,
        test_memo_is_a_no_op_without_a_conv_id,
        test_reset_refusal_memo_scopes_correctly,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            # The memo is process-scoped by design, so it is shared state
            # between tests. Clear it or a later test inherits an earlier
            # one's refusals and passes for the wrong reason.
            dedup.reset_refusal_memo()
            t()
        print("\nAll dedup smoke tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
