"""
v3.1.9.4 R1 (P15-5 follow-up): a per-conversation wipe generation.

Round 2 (B3) made bgwork.BackgroundPool.drain WAIT for outstanding tails on
timeout instead of cancelling them — correct for every OTHER conversation's
in-flight work, which must not be killed just because one conversation typed
/forget. But it reopened a narrower hole for the SAME conversation: a tail
submitted before /forget arrived, still parked on the pool's concurrency
semaphore (or mid an extraction call) when commands._settle_background_
work's drain times out, is now left running to completion — and it can
still acquire conv_lock and write, with nothing left to stop it, AFTER the
wipe already ran. The shape this file proves against:

  1. A tail captures the conversation's wipe generation at the moment it is
     SUBMITTED to the pool (main._run_memory_tail, synchronously, before
     either _fire_and_forget call) — not when it happens to start running.
  2. Every wipe path (main._clear_all_memory — the chokepoint /forget, the
     admin facts-delete endpoint and the self-test cleanup all call through
     — and commands._handle_retire's apply step) bumps the conversation's
     generation, inside the SAME conv_lock section as its deletes, before
     any of them run.
  3. Every write site the tail reaches (main._facts_tail's two locked
     blocks, main._async_tail's episodic-index block, and
     summarizer._maybe_rollup_body's one save_state) re-reads the CURRENT
     generation under conv_lock, immediately before writing, and discards
     the write on a mismatch.

Each write site gets: a CONTROL (no wipe in between — writes normally), the
defect shape (wipe happens after the tail's generation was captured, before
the tail's write runs — nothing lands), and one cross-conversation proof
(another conversation's tail, with its own matching generation, is
unaffected by this one's wipe). All fixtures are synthetic; no HTTP calls
happen (extraction, indexing and summarization are stubbed/spied) except a
fake in-process stub client for the summary-rollup checks, matching
test_v3194_mem_guards.py's own pattern.

Run: python test_v3194_r3_r1.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-r3-r1-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "true"
# L3-ready-state trick, matching test_v3194_mem_guards.py: two seeded L2
# chapters plus this small chunk size reaches maybe_rollup's L3 tier with no
# long synthetic conversation needed.
os.environ["COMPACTOR_L3_CHUNK_SIZE"] = "2"
# Same idea for L1: a handful of synthetic turns reaches _needs_l1_rollup
# with no long synthetic conversation needed.
os.environ["COMPACTOR_L1_CHUNK_SIZE"] = "2"

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import commands  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def _wipe_storage():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP)
    memory.ensure_storage_layout()


async def _no_extraction(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
    return []


class _NewFactExtraction:
    """Stands in for the vLLM extraction call: always reports exactly one
    new fact, so a landed write is unambiguous (the store goes from empty to
    non-empty) and a discarded write is equally unambiguous (stays empty)."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def __call__(self, client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
        self.calls += 1
        return [self.text]


# ---------------------------------------------------------------------------
# 1. main._facts_tail — the main (extraction) locked block
# ---------------------------------------------------------------------------

def test_control_facts_tail_writes_normally_with_no_wipe_in_between():
    print("\n[test] CONTROL: _facts_tail's write lands when nothing wiped the conversation in between")
    _wipe_storage()
    cid = "r1-facts-control"
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "a fresh conversation starts at generation 0")

    stub = _NewFactExtraction("The control fact should land.")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(main._facts_tail(
            cid, [], "a user turn", "an assistant reply", 2,
            wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(stub.calls == 1, "the extraction call actually ran")
    kept = facts.load_facts(cid)
    check(len(kept) == 1 and kept[0]["text"] == "The control fact should land.",
          f"the new fact landed on disk: {kept!r}")


def test_facts_tail_discards_its_write_after_a_wipe_ran_since_submission():
    print("\n[test] R1: a tail's captured generation is stale after /forget ran in "
          "between — nothing it would have written reaches disk")
    _wipe_storage()
    cid = "r1-facts-parked-tail"
    # Seed a fact so the wipe has something real to clear, mirroring
    # production: a conversation the user is /forget-ing already has memory.
    facts.save_facts(cid, [{"text": "pre-existing fact she wants gone", "added_turn": 1, "last_used": 1}])

    # Step 1: the tail is SUBMITTED — captures the generation as it stands
    # right now (this is main._run_memory_tail's own first statement; here
    # done explicitly so the ordering below is unambiguous).
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no wipe has happened yet")

    # Step 2: the tail is parked (nothing runs yet — this whole test is
    # synchronous up to this point, exactly like the request path between
    # _run_memory_tail's capture and bgwork.pool actually scheduling the
    # coroutine).

    # Step 3: /forget runs to completion BEFORE the parked tail gets its
    # turn — the real wipe, the same one every /forget and the admin
    # facts-delete endpoint call.
    result = asyncio.run(main._clear_all_memory(cid, source="test"))
    check(result["forgotten_facts"] == 1, "the wipe cleared the pre-existing fact")
    check(memory.current_wipe_generation(cid) == 1, "the wipe bumped the generation")

    # Step 4: the parked tail FINALLY runs, long after being submitted, with
    # its now-stale captured generation. It would extract and store a new
    # fact if this fix did not exist.
    stub = _NewFactExtraction("A fact from memory she just asked to forget.")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(main._facts_tail(
            cid, [], "a user turn", "an assistant reply", 2,
            wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(stub.calls == 0, "the extraction call never even ran — discarded before spending a vLLM call")
    check(facts.load_facts(cid) == [], "nothing landed on disk: the forgotten memory stayed forgotten")


def test_facts_tail_extraction_disabled_touched_save_also_discarded():
    print("\n[test] R1: the OTHER locked block in _facts_tail (extraction "
          "disabled — LRU touched-save) is guarded too")
    _wipe_storage()
    cid = "r1-facts-touched-save"
    facts.save_facts(cid, [{"text": "a fact to forget", "added_turn": 1, "last_used": 1}])
    wipe_generation = memory.current_wipe_generation(cid)

    asyncio.run(main._clear_all_memory(cid, source="test"))
    check(memory.current_wipe_generation(cid) == 1, "wipe bumped the generation")

    orig = facts.extraction_enabled
    facts.extraction_enabled = lambda: False
    try:
        asyncio.run(main._facts_tail(
            cid, [{"text": "a fact to forget", "added_turn": 1, "last_used": 1}],
            "a user turn", "an assistant reply", 2,
            wipe_generation=wipe_generation,
        ))
    finally:
        facts.extraction_enabled = orig

    check(facts.load_facts(cid) == [], "the touched-save did not resurrect the forgotten fact")


def test_another_conversations_facts_tail_is_unaffected_by_this_ones_wipe():
    print("\n[test] a wipe on conversation A does not discard conversation B's tail")
    _wipe_storage()
    conv_a, conv_b = "r1-facts-conv-a", "r1-facts-conv-b"
    facts.save_facts(conv_a, [{"text": "A's fact", "added_turn": 1, "last_used": 1}])

    wipe_generation_b = memory.current_wipe_generation(conv_b)
    check(wipe_generation_b == 0, "B starts at generation 0, same as any fresh conversation")

    asyncio.run(main._clear_all_memory(conv_a, source="test"))
    check(memory.current_wipe_generation(conv_a) == 1, "A's generation moved")
    check(memory.current_wipe_generation(conv_b) == 0, "B's generation is untouched by A's wipe")

    stub = _NewFactExtraction("B's own new fact.")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(main._facts_tail(
            conv_b, [], "a user turn", "an assistant reply", 2,
            wipe_generation=wipe_generation_b,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(stub.calls == 1, "B's extraction ran normally")
    kept = facts.load_facts(conv_b)
    check(len(kept) == 1 and kept[0]["text"] == "B's own new fact.",
          "B's write landed — A's /forget never touched it")


async def _run_async_tail_at_generation(wipe_generation, *args, **kwargs):
    """main._async_tail no longer takes wipe_generation as a keyword (it is
    replaced wholesale by three test doubles elsewhere with a fixed
    signature — see main._tail_wipe_generation's own block comment). The
    real submission path (main._run_memory_tail) sets the
    main._tail_wipe_generation contextvar around the call that submits the
    tail; this helper does the identical thing for a test driving
    _async_tail directly."""
    token = main._tail_wipe_generation.set(wipe_generation)
    try:
        return await main._async_tail(*args, **kwargs)
    finally:
        main._tail_wipe_generation.reset(token)


# ---------------------------------------------------------------------------
# 2. main._async_tail's episodic-indexing block
# ---------------------------------------------------------------------------

def test_control_episodic_index_writes_normally_with_no_wipe_in_between():
    print("\n[test] CONTROL: episodic indexing runs when nothing wiped the conversation")
    _wipe_storage()
    cid = "r1-episodic-control"
    wipe_generation = memory.current_wipe_generation(cid)

    calls = []

    def _spy_index(conv_id, turn_index, user_text, assistant_text):
        calls.append((conv_id, turn_index, user_text, assistant_text))
        return True

    orig_index = retrieval.index_exchange
    orig_extract = facts.extract_facts_from_exchange
    retrieval.index_exchange = _spy_index
    facts.extract_facts_from_exchange = _no_extraction
    try:
        asyncio.run(_run_async_tail_at_generation(
            wipe_generation,
            cid, [], "a user turn", "an assistant reply", 2, [],
        ))
    finally:
        retrieval.index_exchange = orig_index
        facts.extract_facts_from_exchange = orig_extract

    check(len(calls) == 1, "index_exchange was called exactly once")


def test_episodic_index_discarded_after_a_wipe_ran_since_submission():
    print("\n[test] R1: episodic indexing is discarded when a wipe ran after this tail was submitted")
    _wipe_storage()
    cid = "r1-episodic-parked-tail"
    wipe_generation = memory.current_wipe_generation(cid)

    asyncio.run(main._clear_all_memory(cid, source="test"))
    check(memory.current_wipe_generation(cid) == 1, "wipe bumped the generation (nothing existed, but the bump still happens)")

    calls = []

    def _spy_index(conv_id, turn_index, user_text, assistant_text):
        calls.append((conv_id, turn_index, user_text, assistant_text))
        return True

    orig_index = retrieval.index_exchange
    orig_extract = facts.extract_facts_from_exchange
    retrieval.index_exchange = _spy_index
    facts.extract_facts_from_exchange = _no_extraction
    try:
        asyncio.run(_run_async_tail_at_generation(
            wipe_generation,
            cid, [], "a user turn", "an assistant reply", 2, [],
        ))
    finally:
        retrieval.index_exchange = orig_index
        facts.extract_facts_from_exchange = orig_extract

    check(len(calls) == 0, "index_exchange was never called — the write was discarded before it could happen")


# ---------------------------------------------------------------------------
# 3. summarizer._maybe_rollup_body's save_state, reached through
#    main._rollup_hierarchy (the real tail path, including the contextvar
#    wiring — summarizer.maybe_rollup is monkeypatched wholesale by other
#    test files, so this is the mechanism that has to actually work).
# ---------------------------------------------------------------------------

def _seed_l3_ready_state(conv_id: str) -> dict:
    st = summarizer._empty_state(conv_id)
    st["l2"] = [
        {"text": "chapter one synthetic summary.", "first_turn": 1, "last_turn": 100},
        {"text": "chapter two synthetic summary.", "first_turn": 101, "last_turn": 200},
    ]
    st["l3"] = None
    st["last_summarized_turn"] = 200
    st["turns_seen"] = 200
    summarizer.save_state(conv_id, st)
    return st


def _msgs(n_turns: int) -> list[dict]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        out.append({"role": "user", "content": f"user turn {i}"})
        out.append({"role": "assistant", "content": f"assistant turn {i}"})
    return out


class _StubClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        class _Resp:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"choices": [{"message": {"content": "the L3 theme summary."}, "finish_reason": "stop"}]}
        if str(url).endswith("/tokenize"):
            raise RuntimeError("no /tokenize in this stub")
        return _Resp()


def test_control_summary_rollup_writes_normally_with_no_wipe_in_between():
    print("\n[test] CONTROL: the real _rollup_hierarchy path lands an L3 refresh when nothing wiped in between")
    _wipe_storage()
    cid = "r1-rollup-control"
    before = _seed_l3_ready_state(cid)
    wipe_generation = memory.current_wipe_generation(cid)

    orig_client = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubClient()
    try:
        asyncio.run(main._rollup_hierarchy(
            cid, _msgs(0), None, wipe_generation=wipe_generation,
        ))
    finally:
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    check(after["l2"] == [], "l2 was consumed by the successful refresh")
    check(after["l3"] is not None and after["l3"]["text"], "l3 now holds the refreshed theme summary")


def test_summary_rollup_discarded_after_a_wipe_ran_since_submission():
    print("\n[test] R1: a summary rollup is discarded whole when a wipe ran after "
          "this tail was submitted — l2/l3 stay exactly as the wipe left them")
    _wipe_storage()
    cid = "r1-rollup-parked-tail"
    before = _seed_l3_ready_state(cid)
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no wipe yet")

    # The real wipe: deletes the summary state file outright.
    result = asyncio.run(main._clear_all_memory(cid, source="test"))
    check(result["forgotten_summary"] is True, "the wipe cleared the summary state")
    check(memory.current_wipe_generation(cid) == 1, "wipe bumped the generation")
    after_wipe = summarizer.load_state(cid)
    check(not after_wipe.get("l1") and not after_wipe.get("l2") and not after_wipe.get("l3"),
          "the summary state is empty right after the wipe")

    orig_client = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubClient()
    try:
        asyncio.run(main._rollup_hierarchy(
            cid, _msgs(0), None, wipe_generation=wipe_generation,
        ))
    finally:
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    check(not after.get("l1"), "l1 is still empty — the discarded rollup wrote nothing")
    check(not after.get("l2"), "l2 is still empty — the two 'forgotten' chapters were never re-summarized in")
    check(after.get("l3") is None, "l3 is still None — no theme summary was written from data that was wiped")
    check(summarizer.load_chapter_archive(cid) == [], "nothing landed in the chapter archive either")


def test_summary_rollup_does_not_resurrect_forgotten_conversation_content():
    print("\n[test] R1 (the case that actually matters): a parked tail's STALE "
          "pre-wipe message history must not be summarized into a FRESH L1 "
          "chunk after the conversation was wiped — the wipe deletes the "
          "existing summary STATE, but the tail still carries the old "
          "MESSAGES as an argument, and those alone are enough to build a "
          "brand new chunk if the generation check does not stop it")
    _wipe_storage()
    cid = "r1-rollup-resurrection"
    # No state seeded — the wipe has nothing to delete here; that is the
    # point. The danger is not "did the wipe's own deletion survive", it is
    # "does the discarded rollup call refuse to CREATE new summary content
    # from conversation the tail captured before the wipe".
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no wipe yet")

    asyncio.run(main._clear_all_memory(cid, source="test"))
    check(memory.current_wipe_generation(cid) == 1, "wipe bumped the generation")

    # The STALE history the parked tail is still holding — built with
    # COMPACTOR_L1_CHUNK_SIZE=2, two synthetic turns is already enough to
    # trigger _needs_l1_rollup on a from-scratch state.
    stale_messages = _msgs(2)

    orig_client = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubClient()
    try:
        asyncio.run(main._rollup_hierarchy(
            cid, stale_messages, "the final synthetic assistant reply",
            wipe_generation=wipe_generation,
        ))
    finally:
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    check(not after.get("l1"),
          f"no L1 chunk was built from the forgotten conversation's stale history: {after.get('l1')!r}")
    check(after.get("last_summarized_turn", 0) == 0,
          "the watermark did not advance either — nothing about this call was allowed to happen")


def test_control_summary_rollup_builds_a_fresh_chunk_with_no_wipe_in_between():
    print("\n[test] CONTROL: the identical from-scratch L1 rollup lands normally when nothing wiped in between")
    _wipe_storage()
    cid = "r1-rollup-fresh-control"
    wipe_generation = memory.current_wipe_generation(cid)
    messages = _msgs(2)

    orig_client = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubClient()
    try:
        asyncio.run(main._rollup_hierarchy(
            cid, messages, "the final synthetic assistant reply",
            wipe_generation=wipe_generation,
        ))
    finally:
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    check(len(after.get("l1") or []) >= 1, f"an L1 chunk WAS built: {after.get('l1')!r}")


def test_another_conversations_rollup_is_unaffected_by_this_ones_wipe():
    print("\n[test] a wipe on conversation A does not discard conversation B's summary rollup")
    _wipe_storage()
    conv_a, conv_b = "r1-rollup-conv-a", "r1-rollup-conv-b"
    _seed_l3_ready_state(conv_a)
    before_b = _seed_l3_ready_state(conv_b)
    wipe_generation_b = memory.current_wipe_generation(conv_b)

    asyncio.run(main._clear_all_memory(conv_a, source="test"))
    check(memory.current_wipe_generation(conv_b) == 0, "B's generation is untouched by A's wipe")

    orig_client = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubClient()
    try:
        asyncio.run(main._rollup_hierarchy(
            conv_b, _msgs(0), None, wipe_generation=wipe_generation_b,
        ))
    finally:
        summarizer.httpx.AsyncClient = orig_client

    after_b = summarizer.load_state(conv_b)
    check(after_b["l2"] == [], "B's refresh landed normally")
    check(after_b["l3"] is not None, "B's l3 was written — A's /forget never touched it")


# ---------------------------------------------------------------------------
# 4. main._run_memory_tail: the submission-time capture itself, and that it
#    is threaded to BOTH fire_and_forget call sites (the full-tail path and
#    the rollup-only "reply was skipped" path).
# ---------------------------------------------------------------------------

def test_run_memory_tail_captures_generation_before_submitting_the_full_tail():
    print("\n[test] _run_memory_tail sets main._tail_wipe_generation to the "
          "pre-wipe generation for the duration of the call that submits "
          "the full tail — proven by reading the contextvar from inside the "
          "spy that replaces _fire_and_forget (main._async_tail no longer "
          "takes wipe_generation as a keyword — see the block comment on "
          "main._tail_wipe_generation for why)")
    _wipe_storage()
    cid = "r1-run-tail-capture"
    facts.save_facts(cid, [{"text": "a fact to forget", "added_turn": 1, "last_used": 1}])

    captured = {}

    def _spy_fire_and_forget(coro, label=None):
        # Don't actually run it — bgwork isn't under test here, only that
        # _run_memory_tail set the contextvar correctly BEFORE calling this.
        captured["coro"] = coro
        captured["label"] = label
        captured["wipe_generation"] = main._tail_wipe_generation.get()
        coro.close()
        return True

    orig = main._fire_and_forget
    main._fire_and_forget = _spy_fire_and_forget
    try:
        # Generation captured here (0) is BEFORE the wipe below.
        decision = main._run_memory_tail(
            cid, "a real assistant reply", finished=True, truncated=False, holed=False,
            touched_facts=[{"text": "a fact to forget", "added_turn": 1, "last_used": 1}],
            last_user_text="a user message", turn_index=2,
            messages=[{"role": "user", "content": "a user message"}],
            injected_facts=None,
        )
    finally:
        main._fire_and_forget = orig

    check(decision.store is True, "the reply was accepted for storage")
    check(captured.get("coro") is not None, "a tail coroutine was submitted")
    check(captured.get("wipe_generation") == 0,
          f"the contextvar read 0 inside the spy (captured before any wipe): {captured.get('wipe_generation')!r}")
    check(main._tail_wipe_generation.get() is None,
          "and it is reset back to the default once _run_memory_tail returns — it must not leak into whatever runs next in this same task")


def test_run_memory_tail_captures_generation_for_the_rollup_only_path():
    print("\n[test] _run_memory_tail also threads wipe_generation into the "
          "rollup-only fire_and_forget (the 'reply was skipped, roll up the "
          "history anyway' path)")
    _wipe_storage()
    cid = "r1-run-tail-rollup-only"

    captured = {}

    def _spy_fire_and_forget(coro, label=None):
        captured["coro"] = coro
        captured["label"] = label
        captured["wipe_generation"] = (
            coro.cr_frame.f_locals.get("wipe_generation")
            if coro.cr_frame is not None else "<no frame>"
        )
        coro.close()
        return True

    orig = main._fire_and_forget
    main._fire_and_forget = _spy_fire_and_forget
    try:
        # holed=True with non-empty text: decide_memory_tail refuses to
        # store it (SKIPPED_HOLED) but raw_chars > 0 (computed BEFORE the
        # holed check), which is the discriminator _run_memory_tail's own
        # comment names for firing the rollup-only path — with prior
        # conversational history, the watermark must not freeze just
        # because THIS reply could not be trusted.
        decision = main._run_memory_tail(
            cid, "a real reply that had a hole in its stream",
            finished=True, truncated=False, holed=True,
            touched_facts=[], last_user_text="a user message", turn_index=4,
            messages=[
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "a user message"},
            ],
            injected_facts=None,
        )
    finally:
        main._fire_and_forget = orig

    check(decision.store is False, "a holed reply is not stored")
    check(captured.get("coro") is not None,
          "the rollup-only coroutine was still submitted (history exists to roll up)")
    check(captured.get("wipe_generation") == 0,
          f"the rollup-only path was handed generation 0 too: {captured.get('wipe_generation')!r}")


# ---------------------------------------------------------------------------
# 5. /retire's apply step also bumps the source conversation's generation
#    (commands._handle_retire) — the other wipe path this brief names.
# ---------------------------------------------------------------------------

def _retire(arg, cid):
    return asyncio.run(commands.handle_command("retire", arg, cid, ctx={}))


def _retire_code(out, source):
    marker = f"/retire {source} apply "
    assert marker in out, f"dry run offered no confirmation code: {out[-400:]}"
    return out.split(marker, 1)[1].split()[0]


def _retire_apply(source, dest):
    return _retire(f"{source} apply {_retire_code(_retire(source, dest), source)}", dest)


def test_retire_apply_bumps_the_source_conversations_generation():
    print("\n[test] /retire ... apply bumps the SOURCE conversation's wipe generation")
    _wipe_storage()
    src, dst = "r1-retire-src", "r1-retire-dst"
    facts.save_facts(src, [{"text": "Her mother's name is Selene.", "added_turn": 1, "last_used": 1}])
    check(memory.current_wipe_generation(src) == 0, "source starts at generation 0")

    out = _retire_apply(src, dst)
    check(out.startswith("Retired "), f"applied cleanly: {out[:200]!r}")
    check(memory.current_wipe_generation(src) >= 1,
          f"the source's generation moved after /retire apply: {memory.current_wipe_generation(src)!r}")
    # The destination is a different conversation — /retire moving memory
    # INTO it is not a wipe of it, so its own generation must not move.
    check(memory.current_wipe_generation(dst) == 0,
          "the destination's generation is untouched — receiving memory is not being wiped")


def _all_tests():
    return [
        test_control_facts_tail_writes_normally_with_no_wipe_in_between,
        test_facts_tail_discards_its_write_after_a_wipe_ran_since_submission,
        test_facts_tail_extraction_disabled_touched_save_also_discarded,
        test_another_conversations_facts_tail_is_unaffected_by_this_ones_wipe,
        test_control_episodic_index_writes_normally_with_no_wipe_in_between,
        test_episodic_index_discarded_after_a_wipe_ran_since_submission,
        test_control_summary_rollup_writes_normally_with_no_wipe_in_between,
        test_summary_rollup_discarded_after_a_wipe_ran_since_submission,
        test_summary_rollup_does_not_resurrect_forgotten_conversation_content,
        test_control_summary_rollup_builds_a_fresh_chunk_with_no_wipe_in_between,
        test_another_conversations_rollup_is_unaffected_by_this_ones_wipe,
        test_run_memory_tail_captures_generation_before_submitting_the_full_tail,
        test_run_memory_tail_captures_generation_for_the_rollup_only_path,
        test_retire_apply_bumps_the_source_conversations_generation,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R1 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
