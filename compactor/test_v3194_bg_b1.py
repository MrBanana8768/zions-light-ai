"""
v3.1.9.4 B1 (P15-4): fact-selection and dedup-clustering cost her ~30s per
reply in production because facts._relevance_order and dedup._embed_facts
each re-embedded the WHOLE non-pinned fact store, inline, on every call.
Measured (SP\\3194-bg\\b1-before.log / b1-after.log, this workstation,
Docker unit image, synthetic 190-fact store shaped like her real one):
select_for_injection went from ~5.3s EVERY call to ~20ms once the store is
warm; dedup.find_candidate_clusters from ~6.2-7.3s to sub-second (see
fix-3194-bg.md B1 for the full numbers).

This file proves, with synthetic text only:
  1. retrieval._embed_cached is a correct, bounded, (model, text)-keyed
     cache: a warm hit never calls the embedder, an edited text is a new
     key (never a stale vector), a model-identity change is a new key,
     the cache never exceeds its bound, and a failed fetch never corrupts
     or evicts what was already cached.
  2. facts.select_for_injection's DEFAULT (no embedder=) path returns
     BIT-FOR-BIT the same order whether the cache is cold or warm, and a
     warm call only asks the embedder for genuinely new/changed text.
  3. dedup._embed_facts shares the SAME cache — a fact embedded once by
     selection is not re-embedded by dedup.
  4. The event-loop-blocking claim and its fix: a slow embedder blocks a
     concurrent asyncio task when called directly, and does not when
     driven through run_in_threadpool — the exact pattern _dedup_pass now
     uses internally, and the exact pattern the ~9393 call site in
     main.py's chat_completions should use (see fix-3194-bg.md B1 for the
     literal replacement line; main.py:chat_completions is out of this
     lane's file ownership).

Run: python test_v3194_bg_b1.py
"""
import asyncio
import os
import sys
import tempfile
import time
from unittest.mock import patch

_TMP_ROOT = tempfile.mkdtemp(prefix="v3194-bg-b1-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # never touch the real model

import dedup  # noqa: E402
import facts  # noqa: E402
import retrieval  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402


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


def _fact(text, added_turn=0, last_used=0, pin=False):
    return {"text": text, "added_turn": added_turn, "last_used": last_used, "pin": pin}


def _counting_embed(dim=4):
    """A deterministic, content-hashed embedder that counts how many
    texts it was actually asked to embed — the thing every test below
    checks to prove the cache is doing its job, not just returning the
    right numbers by coincidence."""
    calls = {"n": 0, "texts": []}

    def embed(texts):
        calls["n"] += len(texts)
        calls["texts"].extend(texts)
        out = []
        for t in texts:
            h = sum(t.encode("utf-8"))
            out.append([((h >> (4 * i)) % 97) / 97.0 for i in range(dim)])
        return out

    return embed, calls


# ---------------------------------------------------------------------------
# 1. retrieval._embed_cached
# ---------------------------------------------------------------------------

def test_cache_cold_then_warm_hit_calls_embedder_zero_times():
    print("\n[test] _embed_cached: a repeated identical text list makes NO "
          "second call to the embedder")
    retrieval.reset_vector_cache()
    embed, calls = _counting_embed()
    with patch.object(retrieval, "_embed", embed):
        v1 = retrieval._embed_cached(["alpha", "beta", "gamma"])
        n_after_first = calls["n"]
        v2 = retrieval._embed_cached(["alpha", "beta", "gamma"])
    assert_eq(n_after_first, 3, "first call embeds all three texts")
    assert_eq(calls["n"], 3, "second call embeds NOTHING new — full cache hit")
    assert_eq(v1, v2, "and the vectors returned are identical either way")


def test_cache_only_embeds_the_new_or_changed_text():
    print("\n[test] _embed_cached: growing a batch by one new text embeds "
          "ONLY that text, not the whole batch again")
    retrieval.reset_vector_cache()
    embed, calls = _counting_embed()
    with patch.object(retrieval, "_embed", embed):
        retrieval._embed_cached(["a", "b", "c"])
        n1 = calls["n"]
        retrieval._embed_cached(["a", "b", "c", "d"])
        n2 = calls["n"]
    assert_eq(n1, 3, "first batch: 3 embedded")
    assert_eq(n2 - n1, 1, "second batch: exactly 1 NEW text embedded, not 4")


def test_edited_text_is_a_new_key_never_a_stale_vector():
    print("\n[test] _embed_cached: a byte-for-byte text edit is a cache "
          "MISS, never the old text's vector under the new spelling")
    retrieval.reset_vector_cache()
    embed, calls = _counting_embed()
    with patch.object(retrieval, "_embed", embed):
        v_before = retrieval._embed_cached(["the cat sat"])[0]
        n1 = calls["n"]
        v_after = retrieval._embed_cached(["the cat sat."])[0]  # one char added
        n2 = calls["n"]
    assert_eq(n2 - n1, 1, "the edited text triggered a fresh embed, not a cache hit")
    assert_true(v_before != v_after,
                "different text produced a different (correctly re-embedded) vector")


def test_model_identity_change_is_a_new_key():
    print("\n[test] _embed_cached: swapping COMPACTOR_EMBEDDING_MODEL "
          "identity cannot serve another model's cached vector")
    retrieval.reset_vector_cache()
    embed, calls = _counting_embed()
    with patch.object(retrieval, "_embed", embed):
        with patch.object(retrieval, "EMBEDDING_MODEL", "model-a"):
            retrieval._embed_cached(["same text"])
            n1 = calls["n"]
        with patch.object(retrieval, "EMBEDDING_MODEL", "model-b"):
            retrieval._embed_cached(["same text"])
            n2 = calls["n"]
    assert_eq(n2 - n1, 1, "a different model identity re-embeds rather than reusing model-a's vector")


def test_cache_is_bounded():
    print("\n[test] _embed_cached: the cache never exceeds its configured bound")
    retrieval.reset_vector_cache()
    embed, _calls = _counting_embed()
    cap = 25
    with patch.object(retrieval, "_VECTOR_CACHE_MAX", cap), \
         patch.object(retrieval, "_embed", embed):
        for i in range(cap * 3):
            retrieval._embed_cached([f"distinct fact number {i}"])
            assert_true(len(retrieval._vector_cache) <= cap,
                        f"cache size never exceeds {cap} (i={i})")
    assert_eq(len(retrieval._vector_cache), cap, "settles at exactly the cap")


def test_cache_bound_evicts_least_recently_used():
    print("\n[test] _embed_cached: eviction under the bound is LRU, not FIFO-of-insertion-only")
    retrieval.reset_vector_cache()
    embed, calls = _counting_embed()
    with patch.object(retrieval, "_VECTOR_CACHE_MAX", 3), \
         patch.object(retrieval, "_embed", embed):
        retrieval._embed_cached(["x1", "x2", "x3"])
        retrieval._embed_cached(["x1"])  # touch x1 -> most-recently-used
        retrieval._embed_cached(["x4"])  # forces one eviction: x2 (oldest untouched)
        n_before = calls["n"]
        retrieval._embed_cached(["x1"])  # still cached
        n_after_x1 = calls["n"]
        retrieval._embed_cached(["x2"])  # was evicted -> re-embedded
        n_after_x2 = calls["n"]
    assert_eq(n_after_x1, n_before, "x1 (recently touched) survived eviction")
    assert_eq(n_after_x2, n_before + 1, "x2 (least recently used) was evicted and re-embedded")


def test_failed_fetch_returns_none_without_corrupting_existing_cache():
    print("\n[test] _embed_cached: a failure on the MISSING portion returns "
          "None for this call but does not evict or corrupt earlier hits")
    retrieval.reset_vector_cache()
    embed, calls = _counting_embed()
    with patch.object(retrieval, "_embed", embed):
        retrieval._embed_cached(["known-1", "known-2"])
    with patch.object(retrieval, "_embed", lambda texts: None):
        result = retrieval._embed_cached(["known-1", "brand-new"])
    assert_eq(result, None, "a failure on the new text fails the WHOLE call (all-or-nothing, same as _embed)")
    with patch.object(retrieval, "_embed", embed):
        n_before = calls["n"]
        again = retrieval._embed_cached(["known-1", "known-2"])
        n_after = calls["n"]
    assert_eq(n_after, n_before, "the two previously-cached texts are still cached — the failure did not evict them")
    assert_true(again is not None and len(again) == 2, "and they still resolve correctly")


def test_reset_vector_cache_clears_everything():
    print("\n[test] reset_vector_cache: test-only hook actually empties the cache")
    embed, calls = _counting_embed()
    with patch.object(retrieval, "_embed", embed):
        retrieval._embed_cached(["one", "two"])
    assert_true(len(retrieval._vector_cache) > 0, "cache is populated")
    retrieval.reset_vector_cache()
    assert_eq(len(retrieval._vector_cache), 0, "cache is empty after reset")


# ---------------------------------------------------------------------------
# 2. facts.select_for_injection: identical results, cold vs warm
# ---------------------------------------------------------------------------

def _synthetic_store(n=40):
    topics = ["alpha", "beta", "gamma", "delta"]
    return [
        _fact(f"[{topics[i % len(topics)]}] synthetic fact number {i} about nothing in particular",
              added_turn=i, last_used=1000 + i)
        for i in range(n)
    ]


def test_selection_is_identical_cold_vs_warm_cache():
    print("\n[test] select_for_injection: SAME order and SAME facts whether "
          "the vector cache starts cold or already warm (v3.1.9.4 B1's "
          "own proof requirement — the cache must never change what gets "
          "selected, only how fast)")
    store = _synthetic_store(40)
    embed, calls = _counting_embed()
    query = "[gamma] anything about gamma"

    retrieval.reset_vector_cache()
    with patch.object(retrieval, "_embed", embed):
        cold = facts.select_for_injection(list(store), max_tokens=300, query_text=query)
    cold_calls = calls["n"]

    # Warm run: same store (same objects would touch last_used identically;
    # use fresh copies with identical text so this is a clean "same store,
    # cache already warm from an earlier turn" scenario).
    warm_store = [dict(f) for f in store]
    with patch.object(retrieval, "_embed", embed):
        warm = facts.select_for_injection(warm_store, max_tokens=300, query_text=query)
    warm_calls = calls["n"] - cold_calls

    assert_eq([f["text"] for f in cold], [f["text"] for f in warm],
              "identical selection, cold vs warm cache")
    assert_true(warm_calls <= 1,
                f"warm run embedded ~nothing new (embedded {warm_calls} texts); "
                f"cold run embedded {cold_calls}")


def test_selection_warm_run_only_embeds_the_query_when_store_unchanged():
    print("\n[test] select_for_injection: an unchanged store, new query "
          "each turn — only the query is ever embedded fresh")
    store = _synthetic_store(40)
    embed, calls = _counting_embed()
    retrieval.reset_vector_cache()
    with patch.object(retrieval, "_embed", embed):
        facts.select_for_injection([dict(f) for f in store], max_tokens=300,
                                    query_text="[alpha] turn one")
        n1 = calls["n"]
        facts.select_for_injection([dict(f) for f in store], max_tokens=300,
                                    query_text="[beta] a completely different turn")
        n2 = calls["n"]
    assert_eq(n2 - n1, 1, "the second call embedded exactly one new text: its own query")


def test_selection_embeds_only_the_new_fact_when_one_is_added():
    print("\n[test] select_for_injection: one new fact added to an "
          "otherwise-unchanged store embeds only that one fact plus the query")
    store = _synthetic_store(40)
    embed, calls = _counting_embed()
    retrieval.reset_vector_cache()
    with patch.object(retrieval, "_embed", embed):
        facts.select_for_injection([dict(f) for f in store], max_tokens=1000,
                                    query_text="[alpha] turn one")
        n1 = calls["n"]
        grown = [dict(f) for f in store] + [_fact("[alpha] a brand new fact this turn", 99, 2000)]
        facts.select_for_injection(grown, max_tokens=1000, query_text="[alpha] turn one")
        n2 = calls["n"]
    # +1 for the new fact. The query text is IDENTICAL to the first call
    # ("[alpha] turn one"), and queries are embedded uncached/fresh every
    # time by design (see _embed_cached's docstring) — so it costs one
    # more embed call too, for a total of 2, never anything close to 41.
    assert_eq(n2 - n1, 2, "exactly the new fact + the (uncached-by-design) query, not the whole store")


def test_a_supplied_embedder_still_bypasses_the_cache_entirely():
    print("\n[test] select_for_injection: embedder= override still gets "
          "the ORIGINAL one-call, uncached contract (tests rely on this)")
    store = _synthetic_store(5)
    calls = {"n": 0}

    def mock(texts):
        calls["n"] += 1
        return [[1.0, 0.0] for _ in texts]

    retrieval.reset_vector_cache()
    facts.select_for_injection(store, max_tokens=300, query_text="q", embedder=mock)
    facts.select_for_injection(store, max_tokens=300, query_text="q", embedder=mock)
    assert_eq(calls["n"], 2, "embedder= is called fresh every time — never routed through the cache")


# ---------------------------------------------------------------------------
# 3. dedup shares the same cache as selection
# ---------------------------------------------------------------------------

def test_dedup_and_selection_share_the_vector_cache():
    print("\n[test] a fact embedded once by select_for_injection is NOT "
          "re-embedded by dedup.find_candidate_clusters — same process-wide "
          "cache, exactly the sharing v3.1.9.4 B1 asked for")
    store = _synthetic_store(20)
    embed, calls = _counting_embed()
    retrieval.reset_vector_cache()
    with patch.object(retrieval, "_embed", embed):
        facts.select_for_injection([dict(f) for f in store], max_tokens=2000,
                                    query_text="[alpha] anything")
        n_after_selection = calls["n"]
        dedup.find_candidate_clusters([dict(f) for f in store])
        n_after_dedup = calls["n"]
    assert_eq(n_after_dedup, n_after_selection,
              "dedup embedded NOTHING new — every fact text was already cached by selection")


def test_dedup_embeds_only_new_facts_across_repeated_tail_passes():
    print("\n[test] find_candidate_clusters: a store that grows by one fact "
          "per pass (the tail's real shape) embeds only the new fact each pass")
    store = _synthetic_store(15)
    embed, calls = _counting_embed()
    retrieval.reset_vector_cache()
    with patch.object(retrieval, "_embed", embed):
        dedup.find_candidate_clusters(list(store))
        n1 = calls["n"]
        store = store + [_fact("[alpha] pass two's new fact", 900, 900)]
        dedup.find_candidate_clusters(list(store))
        n2 = calls["n"]
        store = store + [_fact("[beta] pass three's new fact", 901, 901)]
        dedup.find_candidate_clusters(list(store))
        n3 = calls["n"]
    assert_eq(n1, 15, "first pass embeds the whole starting store")
    assert_eq(n2 - n1, 1, "second pass embeds only its one new fact")
    assert_eq(n3 - n2, 1, "third pass embeds only ITS one new fact")


# ---------------------------------------------------------------------------
# 4. Off the event loop: run_in_threadpool actually unblocks a concurrent task
# ---------------------------------------------------------------------------

def _install_slow_embedder(seconds):
    """A synchronous, CPU/IO-bound-shaped stand-in for the real fastembed
    call: it just blocks for `seconds` — long enough that, if something
    calls it directly on the event loop, a concurrent asyncio task will
    visibly stall behind it."""
    def slow(texts):
        time.sleep(seconds)
        return [[1.0, 0.0] for _ in texts]
    return slow


async def _ticks_during(coro_factory, ticker_interval=0.02):
    """Run `coro_factory()` while a ticker counts how many times it got to
    run concurrently. Returns (result, tick_count)."""
    ticks = {"n": 0}
    stop = asyncio.Event()

    async def ticker():
        while not stop.is_set():
            ticks["n"] += 1
            await asyncio.sleep(ticker_interval)

    t = asyncio.create_task(ticker())
    try:
        result = await coro_factory()
    finally:
        stop.set()
        await asyncio.sleep(0)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    return result, ticks["n"]


def test_direct_blocking_call_stalls_a_concurrent_task():
    print("\n[test] CONTROL: calling find_candidate_clusters's embedding "
          "step directly (no threadpool) stalls a concurrent asyncio task "
          "for its whole duration — this is the defect P15-4 measured "
          "(every OTHER request, including /health, waits behind it)")
    retrieval.reset_vector_cache()
    slow = _install_slow_embedder(0.3)

    async def go():
        with patch.object(retrieval, "_embed", slow):
            return dedup.find_candidate_clusters(
                [_fact(f"blocking-{i}", i, i) for i in range(3)]
            )

    _result, ticks = asyncio.run(_ticks_during(go, ticker_interval=0.02))
    # 0.3s of dead time at a 0.02s tick interval would allow ~15 ticks if
    # nothing stalled; a direct call yields control to the loop ZERO times
    # during the sleep, so the ticker gets at most the ticks before/after.
    assert_true(ticks <= 2, f"the ticker barely ran while blocked ({ticks} ticks) — the defect, reproduced")


def test_run_in_threadpool_does_not_stall_a_concurrent_task():
    print("\n[test] FIX: the SAME slow embedding call, driven through "
          "run_in_threadpool exactly as _dedup_pass now does internally, "
          "lets a concurrent asyncio task keep running")
    retrieval.reset_vector_cache()
    slow = _install_slow_embedder(0.3)

    async def go():
        with patch.object(retrieval, "_embed", slow):
            return await run_in_threadpool(
                dedup.find_candidate_clusters,
                [_fact(f"unblocked-{i}", i, i) for i in range(3)],
            )

    _result, ticks = asyncio.run(_ticks_during(go, ticker_interval=0.02))
    assert_true(ticks >= 8, f"the ticker kept running while the embed call was off-loaded ({ticks} ticks)")


def test_dedup_pass_itself_does_not_block_the_loop():
    print("\n[test] _dedup_pass (the real function, not a stand-in) does "
          "not block a concurrent task during clustering — proves the "
          "actual fix in dedup.py, not just the pattern")
    retrieval.reset_vector_cache()
    slow = _install_slow_embedder(0.3)
    import httpx
    from unittest.mock import MagicMock, AsyncMock

    async def go():
        client = MagicMock()
        client.post = AsyncMock()
        with patch.object(retrieval, "_embed", slow):
            stats = {"facts": 3, "clusters": 0, "memo_skips": 0, "calls": 0,
                      "merges": 0, "removed": 0, "deferred": 0}
            return await dedup._dedup_pass(
                client, "http://x", "m",
                [_fact(f"real-{i}", i, i) for i in range(3)], None, stats,
            )

    _result, ticks = asyncio.run(_ticks_during(go, ticker_interval=0.02))
    assert_true(ticks >= 8, f"_dedup_pass's own clustering call did not stall the loop ({ticks} ticks)")


def test_request_path_pattern_select_for_injection_via_run_in_threadpool():
    print("\n[test] the ~main.py:9393 recommendation: wrapping "
          "facts.select_for_injection(touched_facts, query_text=...) in "
          "run_in_threadpool, exactly as the report recommends, drives "
          "the REAL function and does not stall a concurrent task")
    store = _synthetic_store(30)
    slow = _install_slow_embedder(0.3)
    retrieval.reset_vector_cache()

    async def blocking():
        with patch.object(retrieval, "_embed", slow):
            return facts.select_for_injection(
                [dict(f) for f in store], query_text="a new turn's text"
            )

    async def fixed():
        with patch.object(retrieval, "_embed", slow):
            return await run_in_threadpool(
                facts.select_for_injection,
                [dict(f) for f in store],
                query_text="a new turn's text",
            )

    _r1, ticks_blocking = asyncio.run(_ticks_during(blocking, ticker_interval=0.02))
    retrieval.reset_vector_cache()
    _r2, ticks_fixed = asyncio.run(_ticks_during(fixed, ticker_interval=0.02))
    assert_true(ticks_blocking <= 2,
                f"CONTROL: the direct call (today's main.py:9393) stalls the loop ({ticks_blocking} ticks)")
    assert_true(ticks_fixed >= 8,
                f"FIX: run_in_threadpool(facts.select_for_injection, ...) does not ({ticks_fixed} ticks)")


# ---------------------------------------------------------------------------
# 5. The docstrings no longer claim the cost is negligible
# ---------------------------------------------------------------------------

def test_cpu_milliseconds_docstring_is_gone():
    print("\n[test] the 'CPU-milliseconds either way' claim is corrected, "
          "not just moved (the corrected docstring is allowed to MENTION "
          "the old phrase while explaining it is wrong; it must not still "
          "ASSERT it)")
    assert_true("CPU-milliseconds either way" not in (facts._relevance_order.__doc__ or ""),
                "_relevance_order no longer claims the embedding cost is negligible")
    assert_true("milliseconds, no GPU, no new dependency —" not in (facts.select_for_injection.__doc__ or ""),
                "select_for_injection no longer claims it either")
    assert_true("NOT" in (facts._relevance_order.__doc__ or "")
                and "12.5s" in (facts._relevance_order.__doc__ or ""),
                "the corrected docstring states the measured cost instead")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_cache_cold_then_warm_hit_calls_embedder_zero_times,
        test_cache_only_embeds_the_new_or_changed_text,
        test_edited_text_is_a_new_key_never_a_stale_vector,
        test_model_identity_change_is_a_new_key,
        test_cache_is_bounded,
        test_cache_bound_evicts_least_recently_used,
        test_failed_fetch_returns_none_without_corrupting_existing_cache,
        test_reset_vector_cache_clears_everything,
        test_selection_is_identical_cold_vs_warm_cache,
        test_selection_warm_run_only_embeds_the_query_when_store_unchanged,
        test_selection_embeds_only_the_new_fact_when_one_is_added,
        test_a_supplied_embedder_still_bypasses_the_cache_entirely,
        test_dedup_and_selection_share_the_vector_cache,
        test_dedup_embeds_only_new_facts_across_repeated_tail_passes,
        test_direct_blocking_call_stalls_a_concurrent_task,
        test_run_in_threadpool_does_not_stall_a_concurrent_task,
        test_dedup_pass_itself_does_not_block_the_loop,
        test_request_path_pattern_select_for_injection_via_run_in_threadpool,
        test_cpu_milliseconds_docstring_is_gone,
    ]


if __name__ == "__main__":
    import shutil
    try:
        for t in _all_tests():
            retrieval.reset_vector_cache()
            dedup.reset_refusal_memo()
            t()
        print("\nAll v3.1.9.4 B1 tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
