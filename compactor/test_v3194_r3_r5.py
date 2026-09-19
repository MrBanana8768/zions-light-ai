"""
v3.1.9.4 R5: small fixes.

1. retrieval._embed_cached: the all-hits path used to look entries up
   under ONE lock acquisition and move_to_end them under a SEPARATE one, so
   a concurrent eviction in between raised KeyError (the caller's `except
   Exception` then degraded that turn to LRU order or skipped dedup). Fixed
   by making the lookup and the move_to_end ONE critical section (all-hits
   path), and by tolerating a missing key in the miss-path's own trailing
   move_to_end loop (which touches hit-texts read outside the lock that
   writes fresh vectors).
2. main.chat_completions calls facts.select_for_injection through
   `await run_in_threadpool(...)` instead of a bare call — the one embedding
   -costed request-path call site B1 (round 2) named but could not fix
   (main.py was another lane's file then).
3. main's `lifespan` shutdown passes `cancel_on_timeout=True` to
   bgwork.pool.drain, restoring pre-B3 shutdown behaviour (B3's new default
   is correct for /forget, wrong for a process that is exiting regardless).

Run: python test_v3194_r3_r5.py
"""

import asyncio
import os
import sys
import tempfile
import threading
import time

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-r3-r5-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import bgwork  # noqa: E402
import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

memory.ensure_storage_layout()

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def _vec_for(text: str) -> list[float]:
    return [float(len(text)), float(sum(map(ord, text)) % 997)]


def _stub_embed(texts):
    return [_vec_for(t) for t in texts]


# ---------------------------------------------------------------------------
# 1a. retrieval._embed_cached: the miss-path tolerates a key evicted, by a
#     concurrent caller, between the lookup and the trailing move_to_end
#     loop (deterministic simulation via a side-effecting stub, rather than
#     hoping for real thread timing).
# ---------------------------------------------------------------------------

def test_miss_path_tolerates_a_hit_key_evicted_during_the_embed_call():
    print("\n[test] R5: a HIT text evicted by a concurrent caller while this "
          "call's own miss is being embedded does not raise — the trailing "
          "move_to_end loop tolerates the missing key")
    retrieval.reset_vector_cache()
    orig_embed = retrieval._embed
    orig_max = retrieval._VECTOR_CACHE_MAX

    # Seed one hit.
    retrieval._embed = _stub_embed
    try:
        retrieval._embed_cached(["already-cached-text"])
    finally:
        retrieval._embed = orig_embed
    check(("test-model" if False else retrieval.EMBEDDING_MODEL, "already-cached-text") in retrieval._vector_cache
          or True, "seed call completed")  # sanity no-op; real check below

    def _embed_and_evict(texts):
        # Simulates: while THIS call's miss is being embedded (a real model
        # call takes real time, off any lock), a concurrent caller runs its
        # own miss path and evicts "already-cached-text" via popitem.
        key = (retrieval.EMBEDDING_MODEL, "already-cached-text")
        retrieval._vector_cache.pop(key, None)
        return _stub_embed(texts)

    retrieval._embed = _embed_and_evict
    try:
        out = retrieval._embed_cached(["already-cached-text", "a-brand-new-miss"])
    except KeyError as e:
        out = None
        check(False, f"raised KeyError instead of tolerating the evicted key: {e!r}")
    finally:
        retrieval._embed = orig_embed
        retrieval._VECTOR_CACHE_MAX = orig_max

    check(out is not None, "no exception propagated")
    if out is not None:
        check(len(out) == 2 and all(v is not None for v in out),
              f"both vectors still returned correctly despite the mid-call eviction: {out!r}")


def test_control_normal_all_hits_and_mixed_lookups_still_work():
    print("\n[test] CONTROL: ordinary (non-racing) lookups are unaffected — "
          "all-hits and mixed hit/miss both return correct, cached vectors")
    retrieval.reset_vector_cache()
    retrieval._embed = _stub_embed
    try:
        first = retrieval._embed_cached(["alpha", "beta"])
        check(first == [_vec_for("alpha"), _vec_for("beta")], "cold call embeds both")
        calls = []
        real_embed = retrieval._embed

        def _counting_embed(texts):
            calls.append(list(texts))
            return real_embed(texts)
        retrieval._embed = _counting_embed
        second = retrieval._embed_cached(["alpha", "beta"])
        check(second == first, "warm all-hits call returns identical vectors")
        check(calls == [], "warm all-hits call embeds NOTHING — served from cache")

        third = retrieval._embed_cached(["alpha", "gamma"])
        check(calls == [["gamma"]], "mixed call embeds only the miss")
        check(third[0] == _vec_for("alpha") and third[1] == _vec_for("gamma"),
              "mixed call returns the right vector for each text")
    finally:
        retrieval._embed = _stub_embed
        retrieval.reset_vector_cache()


def test_all_hits_path_is_one_critical_section_under_real_concurrent_threads():
    print("\n[test] R5: hammering the cache from two real threads — one doing "
          "all-hits lookups on a key, the other repeatedly evicting/"
          "re-adding it — never raises, for hundreds of iterations")
    retrieval.reset_vector_cache()
    orig_max = retrieval._VECTOR_CACHE_MAX
    retrieval._VECTOR_CACHE_MAX = 2  # tiny, so a miss reliably evicts
    retrieval._embed = _stub_embed
    key_text = "hammered-key"
    retrieval._embed_cached([key_text])  # seed it

    errors: list[BaseException] = []
    stop = threading.Event()

    def _reader():
        while not stop.is_set():
            try:
                retrieval._embed_cached([key_text])
            except BaseException as e:  # noqa: BLE001 - the exact thing under test
                errors.append(e)
                return

    def _evictor():
        i = 0
        while not stop.is_set():
            try:
                retrieval._embed_cached([f"filler-{i}"])
                i += 1
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                return

    threads = [threading.Thread(target=_reader), threading.Thread(target=_evictor)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    retrieval._VECTOR_CACHE_MAX = orig_max
    check(errors == [], f"no exception from either thread across the stress run: {errors!r}")


# ---------------------------------------------------------------------------
# 2. chat_completions routes select_for_injection through run_in_threadpool
# ---------------------------------------------------------------------------

def test_select_for_injection_call_site_uses_run_in_threadpool():
    print("\n[test] R5: main.chat_completions' select_for_injection call site "
          "routes through run_in_threadpool, not a bare call")
    import inspect
    src = inspect.getsource(main.chat_completions)
    idx_pool = src.find("run_in_threadpool")
    idx_call = src.find("facts.select_for_injection")
    check(idx_pool != -1 and idx_call != -1, "both names appear in chat_completions' source")
    check(0 <= idx_call - idx_pool < 150,
          f"run_in_threadpool wraps the select_for_injection call directly "
          f"(offset {idx_call - idx_pool if idx_pool != -1 and idx_call != -1 else 'n/a'})")


def test_select_for_injection_does_not_block_the_loop_under_a_slow_relevance_order():
    print("\n[test] R5: driven directly — a slow facts._relevance_order does "
          "not stall a concurrent asyncio task when reached via "
          "run_in_threadpool (the exact fix B1 applied to the sibling call "
          "inside _facts_tail, now applied here too)")

    def _slow_relevance_order(*a, **kw):
        time.sleep(0.3)
        return None

    orig = facts._relevance_order
    facts._relevance_order = _slow_relevance_order
    ticks = {"n": 0}

    async def ticker():
        for _ in range(30):
            await asyncio.sleep(0.01)
            ticks["n"] += 1

    async def go():
        t = asyncio.create_task(ticker())
        await run_in_threadpool_shim(
            facts.select_for_injection,
            [{"text": "a", "last_used": 1, "added_turn": 1}],
            query_text="q",
        )
        await asyncio.sleep(0.05)
        t.cancel()

    # main.run_in_threadpool is starlette's; import the same one directly so
    # this test exercises the identical primitive the fix uses.
    from starlette.concurrency import run_in_threadpool as run_in_threadpool_shim

    try:
        asyncio.run(go())
    finally:
        facts._relevance_order = orig

    check(ticks["n"] >= 15,
          f"the concurrent ticker kept advancing while the slow call ran off-loop: {ticks['n']} ticks in ~0.3s")


# ---------------------------------------------------------------------------
# 3. lifespan shutdown passes cancel_on_timeout=True
# ---------------------------------------------------------------------------

def test_lifespan_shutdown_drain_passes_cancel_on_timeout_true():
    print("\n[test] R5: main's lifespan shutdown restores the old "
          "cancel-on-timeout behaviour (B3's new default is right for "
          "/forget, wrong for a process that is exiting regardless)")
    import inspect
    src = inspect.getsource(main.lifespan)
    check("bgwork.pool.drain(timeout=10.0, cancel_on_timeout=True)" in src,
          f"the exact call with cancel_on_timeout=True is present in lifespan's source")


def test_control_settle_background_work_unaffected_still_defaults_false():
    print("\n[test] CONTROL: commands._settle_background_work's /forget drain "
          "is UNCHANGED by this — it still calls drain with no "
          "cancel_on_timeout, keeping B3's safe (non-cancelling) default")
    import commands
    import inspect
    src = inspect.getsource(commands._settle_background_work)
    check("cancel_on_timeout" not in src,
          "the /forget drain does not pass cancel_on_timeout at all — B3's default (False) still applies there")


def _all_tests():
    return [
        test_miss_path_tolerates_a_hit_key_evicted_during_the_embed_call,
        test_control_normal_all_hits_and_mixed_lookups_still_work,
        test_all_hits_path_is_one_critical_section_under_real_concurrent_threads,
        test_select_for_injection_call_site_uses_run_in_threadpool,
        test_select_for_injection_does_not_block_the_loop_under_a_slow_relevance_order,
        test_lifespan_shutdown_drain_passes_cancel_on_timeout_true,
        test_control_settle_background_work_unaffected_still_defaults_false,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R5 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
