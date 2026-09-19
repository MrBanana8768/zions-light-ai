"""
v3.1.9.4 R4 / W2 (P15-5 follow-up, "Found, not fixed" from R1):
backfill.py's background writes must not undo a wipe either.

`backfill._run_backfill` (submitted via the same `fire_and_forget` the live
memory tail uses, in `backfill.start_backfill_if_needed`) can run for HOURS
(P15-2's own real-data figures — up to 2h43m for 195 exchanges). A /forget
(or /retire, or an overwrite import) that lands mid-run left it free to keep
extracting facts from the conversation's pre-wipe history and write them
straight back — the single longest-lived background writer this codebase
has, and the one R1's wipe-generation counter did not cover.

The shape this file proves, mirroring test_v3194_r3_r1.py's own structure:

  1. `start_backfill_if_needed` captures the conversation's wipe generation
     at the moment it hands the run to `fire_and_forget` — submission, not
     whenever the task actually gets a turn.
  2. `_run_backfill` re-checks it under `conv_lock`, TWICE: once right after
     its first lock (before spending a single vLLM call), and again right
     before its actual `facts_module.save_facts` write (catching a wipe that
     arrived during the — possibly hours-long — extraction loop).
  3. On a mismatch, the run is discarded and its record is written as the
     new terminal state "wiped" — NOT "failed" (which `_backoff_ready`
     would eventually retry) and NOT "abandoned" (the wrong story) — and
     `needs_backfill` refuses to retry a "wiped" record ever again, the
     same permanent weight as "complete".
  4. The hierarchical-summary half of a backfill (`summarizer.maybe_rollup`)
     is covered too, via the same `wipe_generation_ctx` mechanism the live
     tail's `_rollup_hierarchy` already uses.

Each check gets a CONTROL (no wipe in between — the backfill completes and
its record reads "complete", not "wiped"). All fixtures are synthetic; no
HTTP calls happen (extraction and the vLLM client are stubbed).

Run: python test_v3194_r4_w2.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-r4-w2-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "true"
os.environ["COMPACTOR_L1_CHUNK_SIZE"] = "2"

import backfill  # noqa: E402
import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
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


def _msgs(n_pairs: int) -> list[dict]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        out.append({"role": "user", "content": f"user turn {i} with enough real text to extract from"})
        out.append({"role": "assistant", "content": f"assistant reply {i} with enough real text to extract from"})
    return out


class _NewFactExtraction:
    """Always reports exactly one new fact per exchange, so a landed write
    is unambiguous and a discarded one is equally unambiguous."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def __call__(self, client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
        self.calls += 1
        return [f"{self.text} #{self.calls}"]


class _StubRollupClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        class _Resp:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"choices": [{"message": {"content": "a backfilled theme summary."}, "finish_reason": "stop"}]}
        if str(url).endswith("/tokenize"):
            raise RuntimeError("no /tokenize in this stub")
        return _Resp()


# ---------------------------------------------------------------------------
# 1. start_backfill_if_needed captures the generation at submission and
#    threads it through.
# ---------------------------------------------------------------------------

def test_start_backfill_if_needed_captures_generation_before_submitting():
    print("\n[test] start_backfill_if_needed captures the wipe generation "
          "synchronously, before handing the run to fire_and_forget")
    _wipe_storage()
    cid = "w2-capture"

    captured = {}

    def _spy_fire_and_forget(coro, label=None):
        captured["wipe_generation"] = coro.cr_frame.f_locals.get("wipe_generation")
        coro.close()
        return True

    started = asyncio.run(backfill.start_backfill_if_needed(
        cid, _msgs(3), "http://stub", "m", fire_and_forget=_spy_fire_and_forget,
    ))
    check(started is True, "a backfill was started (fresh V1-shaped conversation)")
    check(captured.get("wipe_generation") == 0,
          f"generation 0 was threaded to _run_backfill (before any wipe): {captured.get('wipe_generation')!r}")

    # Now bump and start a second, unrelated backfill-eligible conversation
    # to prove the capture reads the CURRENT value, not a stale constant.
    cid2 = "w2-capture-2"
    memory.bump_wipe_generation(cid2)
    captured2 = {}

    def _spy2(coro, label=None):
        captured2["wipe_generation"] = coro.cr_frame.f_locals.get("wipe_generation")
        coro.close()
        return True

    asyncio.run(backfill.start_backfill_if_needed(
        cid2, _msgs(3), "http://stub", "m", fire_and_forget=_spy2,
    ))
    check(captured2.get("wipe_generation") == 1,
          f"a conversation already at generation 1 is captured as 1, not 0: {captured2.get('wipe_generation')!r}")


# ---------------------------------------------------------------------------
# 2. CONTROL: a backfill that runs with no wipe in between completes
#    normally and its record says "complete".
# ---------------------------------------------------------------------------

def test_control_backfill_completes_normally_with_no_wipe_in_between():
    print("\n[test] CONTROL: a backfill with no wipe in between completes and writes facts")
    _wipe_storage()
    cid = "w2-control"
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "fresh conversation starts at generation 0")

    stub = _NewFactExtraction("a real fact")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(backfill._run_backfill(
            cid, _msgs(3), "http://stub", "m", wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(stub.calls == 3, f"all three exchanges were extracted: {stub.calls}")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "complete",
          f"the record reads complete: {state!r}")
    kept = facts.load_facts(cid)
    check(len(kept) == 3, f"three facts landed: {kept!r}")


# ---------------------------------------------------------------------------
# 3. The defect shape, early check: a wipe that already happened by the
#    time the backfill gets its first turn discards before spending any
#    vLLM calls at all.
# ---------------------------------------------------------------------------

def test_backfill_discarded_before_starting_when_wipe_ran_before_first_turn():
    print("\n[test] W2: a backfill whose captured generation is already stale "
          "by the time it gets its first turn on the event loop discards "
          "before extracting a single exchange")
    _wipe_storage()
    cid = "w2-early-discard"
    facts.save_facts(cid, [{"text": "pre-existing fact she wants gone", "added_turn": 1, "last_used": 1}])

    # Captured at "submission" (generation 0).
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no wipe yet")

    # The wipe runs BEFORE the parked backfill gets its turn.
    asyncio.run(main._clear_all_memory(cid, source="test"))
    check(memory.current_wipe_generation(cid) == 1, "the wipe bumped the generation")

    stub = _NewFactExtraction("resurrected fact")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(backfill._run_backfill(
            cid, _msgs(3), "http://stub", "m", wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(stub.calls == 0, "the extraction call never ran — discarded before spending a single vLLM call")
    check(facts.load_facts(cid) == [], "nothing landed: the forgotten memory stayed forgotten")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped",
          f"the record reads 'wiped', not 'failed' or 'abandoned': {state!r}")
    check(backfill.needs_backfill(cid, _msgs(3)) is False,
          "needs_backfill refuses to retry a 'wiped' record")


def test_wiped_record_is_a_permanent_refusal_on_its_own_terms():
    print("\n[test] W2 (+ v3.1.9.4 R6 / P17-4 follow-up): needs_backfill "
          "refuses a 'wiped' record because it IS 'wiped' — not merely as "
          "a side effect of the empty facts.json tombstone `_clear_all_"
          "memory` also leaves behind. Before P17-4's follow-up,"
          "`_clear_all_memory` wrote that tombstone only when facts existed"
          " to clear, so a conv_id that never had any facts left no facts"
          " file at all and this isolation was incidental. It now writes"
          " the tombstone UNCONDITIONALLY (matching commands._wipe_all_"
          "layers, the chat /forget path, for the same defense-in-depth"
          " reason _facts_tombstoned's own docstring gives), so this test"
          " deletes that tombstone straight back out after the wipe to"
          " isolate the record's own state as the thing actually being"
          " checked — needs_backfill's own docstring confirms the RECORD"
          " is checked before any facts-file signal (B2's reordering).")
    _wipe_storage()
    cid = "w2-wiped-terminal-in-isolation"
    check(memory.current_wipe_generation(cid) == 0, "starts at generation 0")
    wipe_generation = memory.current_wipe_generation(cid)

    # A wipe with nothing to clear (no facts, no summary, no episodic) still
    # bumps the generation — main._clear_all_memory does this unconditionally
    # (see its own comment: "FIRST statement inside the lock"). Called
    # directly here for the same reason.
    asyncio.run(main._clear_all_memory(cid, source="test"))
    check(memory.current_wipe_generation(cid) == 1, "the wipe bumped the generation")
    # v3.1.9.4 (R6 / P17-4 follow-up): the tombstone now exists even though
    # this conv_id never had a fact — see the docstring above.
    check(facts.facts_path(cid).is_file() and facts.load_facts(cid) == [],
          "the empty-facts tombstone now exists unconditionally")
    # Remove it: isolates the record's own "wiped" state as the ONLY signal
    # left for needs_backfill to refuse on.
    facts.facts_path(cid).unlink()
    check(not facts.facts_path(cid).is_file(), "tombstone removed for isolation")

    stub = _NewFactExtraction("resurrected fact")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(backfill._run_backfill(
            cid, _msgs(3), "http://stub", "m", wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(not facts.facts_path(cid).is_file(), "still no facts file after the discarded run")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped", f"the record reads 'wiped': {state!r}")
    check(
        backfill.needs_backfill(cid, _msgs(3)) is False,
        "needs_backfill refuses — and it cannot be falling through to the "
        "'no facts file' branch (that branch would say True), so this is "
        "the 'wiped' state itself being checked, with the tombstone "
        "removed out from under it",
    )


# ---------------------------------------------------------------------------
# 4. The defect shape, late check: a wipe that arrives DURING the
#    extraction loop (after the early check passed) still stops the final
#    write.
# ---------------------------------------------------------------------------

def test_backfill_discarded_at_final_write_when_wipe_ran_during_the_loop():
    print("\n[test] W2 (+ v3.1.9.4 R5 / P16-2): a wipe that lands DURING "
          "the extraction loop (after the early check already passed) "
          "now stops the loop itself at the top of the NEXT iteration, "
          "rather than running to completion and only being caught by "
          "the final-write check — the real 'a backfill can run for "
          "hours' shape, and P16-2's own finding: before this fix, every "
          "remaining exchange still spent a real vLLM call before the "
          "final check discarded the whole result anyway")
    _wipe_storage()
    cid = "w2-mid-run-discard"
    # No pre-existing facts seeded here (unlike the early-discard test
    # above): _run_backfill refuses outright against a non-empty store
    # unless resuming (v3.1 F3 — see its own comment), which would stop
    # this test before it ever reached the loop this check exists to
    # cover. Starting empty still proves the fix: without it, the wipe
    # would leave the accumulated facts to land in this empty store at the
    # final write below.
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no wipe yet — the early check will pass")

    # The extraction stub itself performs the wipe partway through the
    # loop — modeling "a /forget arrived while the backfill was still
    # mid-run", the exact production shape (P15-2: up to 2h43m runs).
    # asyncio.run cannot be nested inside the running loop this stub is
    # already awaited from, so the wipe is done directly (conv_lock +
    # bump_wipe_generation + the empty-facts write _clear_all_memory itself
    # performs) rather than by calling main._clear_all_memory — the only
    # thing that matters for this check is the generation moving and the
    # store actually going empty, both of which this reproduces exactly.
    calls = {"n": 0}

    async def _wipe_mid_loop(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            async with memory.conv_lock(cid):
                memory.bump_wipe_generation(cid)
                facts.save_facts(cid, [])
        return [f"fact from exchange {calls['n']}"]

    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = _wipe_mid_loop
    try:
        asyncio.run(backfill._run_backfill(
            cid, _msgs(4), "http://stub", "m", wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    # v3.1.9.4 (R5 / P16-2 fix). The loop used to keep running to
    # completion (all 4 exchanges) regardless of the wipe, and only the
    # LOCKED final-write check discarded the result — burning 2 wasted
    # vLLM calls here (exchanges 3 and 4), and up to the rest of a
    # multi-hour run in production. The per-iteration check now catches
    # the mismatch at the top of the NEXT iteration after the wipe lands
    # (iteration 3, right after exchange 2's call did the wipe) and stops
    # immediately — no iteration 3 or 4 call is ever made.
    check(calls["n"] == 2, f"the loop stopped at the wipe rather than running to completion: {calls['n']}")
    check(memory.current_wipe_generation(cid) == 1, "the mid-loop wipe bumped the generation")
    check(facts.load_facts(cid) == [],
          f"the accumulated facts (built from pre-wipe history) never reached disk: {facts.load_facts(cid)!r}")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped",
          f"the record reads 'wiped', written by the per-iteration check itself: {state!r}")


def test_backfill_discarded_at_final_write_when_wipe_ran_after_the_last_extraction_call():
    print("\n[test] W2, still covered directly: a wipe landing AFTER the "
          "LAST extraction call (so there is no 'next iteration' for "
          "P16-2's per-iteration check to catch it on) is still caught by "
          "the original, LOCKED final-write check — defense in depth for "
          "the one window the per-iteration check cannot see into")
    _wipe_storage()
    cid = "w2-mid-run-discard-last"
    wipe_generation = memory.current_wipe_generation(cid)

    calls = {"n": 0}

    async def _wipe_after_last(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
        calls["n"] += 1
        if calls["n"] == 4:  # the LAST exchange for _msgs(4)
            async with memory.conv_lock(cid):
                memory.bump_wipe_generation(cid)
                facts.save_facts(cid, [])
        return [f"fact from exchange {calls['n']}"]

    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = _wipe_after_last
    try:
        asyncio.run(backfill._run_backfill(
            cid, _msgs(4), "http://stub", "m", wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(calls["n"] == 4, f"all 4 exchanges were attempted (the wipe lands on the last one): {calls['n']}")
    check(memory.current_wipe_generation(cid) == 1, "the wipe bumped the generation")
    check(facts.load_facts(cid) == [],
          f"the accumulated facts never reached disk — the final-write check caught it: {facts.load_facts(cid)!r}")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped", f"the record reads 'wiped': {state!r}")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped",
          f"the record reads 'wiped': {state!r}")


# ---------------------------------------------------------------------------
# 5. The summary-rollup half is covered too.
# ---------------------------------------------------------------------------

def test_backfill_summary_rollup_discarded_after_a_wipe_ran_since_submission():
    print("\n[test] W2: backfill's own summarizer.maybe_rollup call is "
          "covered by wipe_generation_ctx too — no L1 chunk is resurrected "
          "from the pre-wipe history it was handed at kickoff")
    _wipe_storage()
    cid = "w2-rollup-discard"
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no wipe yet")

    asyncio.run(main._clear_all_memory(cid, source="test"))
    check(memory.current_wipe_generation(cid) == 1, "the wipe bumped the generation")

    # No facts to extract (extraction disabled would also skip the rollup
    # entirely) — use a message count under _MIN_MESSAGES_FOR_BACKFILL's
    # concern is irrelevant here since we call _run_backfill directly, not
    # through needs_backfill. Facts extraction returns nothing so this test
    # isolates the rollup half.
    async def _no_facts(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
        return []

    orig_extract = facts.extract_facts_from_exchange
    orig_client = summarizer.httpx.AsyncClient
    facts.extract_facts_from_exchange = _no_facts
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubRollupClient()
    try:
        asyncio.run(backfill._run_backfill(
            cid, _msgs(2), "http://stub", "m", wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig_extract
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    check(not after.get("l1"), f"no L1 chunk was built from the forgotten conversation's stale history: {after.get('l1')!r}")
    check(after.get("last_summarized_turn", 0) == 0, "the watermark did not move")


def test_control_backfill_summary_rollup_builds_normally_with_no_wipe():
    print("\n[test] CONTROL: backfill's summary rollup builds a chunk normally when nothing wiped in between")
    _wipe_storage()
    cid = "w2-rollup-control"
    wipe_generation = memory.current_wipe_generation(cid)

    async def _no_facts(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
        return []

    orig_extract = facts.extract_facts_from_exchange
    orig_client = summarizer.httpx.AsyncClient
    facts.extract_facts_from_exchange = _no_facts
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubRollupClient()
    try:
        asyncio.run(backfill._run_backfill(
            cid, _msgs(2), "http://stub", "m", wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig_extract
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    check(len(after.get("l1") or []) >= 1, f"an L1 chunk WAS built: {after.get('l1')!r}")


def test_rollup_wiring_alone_discards_a_wipe_that_arrives_after_the_facts_write():
    print("\n[test] W2: the wipe_generation_ctx wiring around backfill's own "
          "maybe_rollup call catches a wipe that arrives AFTER the facts "
          "phase already completed cleanly — isolating this check from the "
          "two facts-side checks above, which would already have discarded "
          "the whole run if the wipe had landed any earlier")
    _wipe_storage()
    cid = "w2-rollup-race"
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no wipe yet")

    async def _no_facts(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
        return []

    # summarizer.load_state(conv_id) is the first thing _run_backfill calls
    # AFTER the facts phase has fully committed (both of its own checks
    # already passed) and BEFORE maybe_rollup even starts — exactly where a
    # wipe racing in after the facts write but before the rollup call would
    # land in production. Wrapped to inject the wipe at precisely that
    # instant instead of before _run_backfill is even called.
    orig_load_state = summarizer.load_state
    injected = {"done": False}

    def _load_state_then_wipe(conv_id_arg):
        if not injected["done"] and conv_id_arg == cid:
            injected["done"] = True
            memory.bump_wipe_generation(cid)
        return orig_load_state(conv_id_arg)

    orig_extract = facts.extract_facts_from_exchange
    orig_client = summarizer.httpx.AsyncClient
    facts.extract_facts_from_exchange = _no_facts
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubRollupClient()
    summarizer.load_state = _load_state_then_wipe
    try:
        asyncio.run(backfill._run_backfill(
            cid, _msgs(2), "http://stub", "m", wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig_extract
        summarizer.httpx.AsyncClient = orig_client
        summarizer.load_state = orig_load_state

    check(injected["done"], "the wipe was actually injected mid-flight")
    check(memory.current_wipe_generation(cid) == 1, "the injected wipe bumped the generation")
    after = orig_load_state(cid)
    check(not after.get("l1"),
          f"no L1 chunk was built — the rollup was discarded despite the facts phase completing cleanly: {after.get('l1')!r}")
    check(after.get("last_summarized_turn", 0) == 0, "the watermark did not move")


# ---------------------------------------------------------------------------
# 6. Another conversation's backfill is unaffected.
# ---------------------------------------------------------------------------

def test_another_conversations_backfill_is_unaffected_by_this_ones_wipe():
    print("\n[test] a wipe on conversation A does not discard conversation B's backfill")
    _wipe_storage()
    conv_a, conv_b = "w2-conv-a", "w2-conv-b"
    facts.save_facts(conv_a, [{"text": "A's fact", "added_turn": 1, "last_used": 1}])

    wipe_generation_b = memory.current_wipe_generation(conv_b)
    check(wipe_generation_b == 0, "B starts at generation 0")

    asyncio.run(main._clear_all_memory(conv_a, source="test"))
    check(memory.current_wipe_generation(conv_a) == 1, "A's generation moved")
    check(memory.current_wipe_generation(conv_b) == 0, "B's generation is untouched by A's wipe")

    stub = _NewFactExtraction("B's own fact")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(backfill._run_backfill(
            conv_b, _msgs(2), "http://stub", "m", wipe_generation=wipe_generation_b,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(stub.calls == 2, "B's extraction ran normally")
    state = backfill.read_state(conv_b)
    check(state is not None and state.get("state") == "complete", f"B's record reads complete: {state!r}")
    check(len(facts.load_facts(conv_b)) == 2, "B's facts landed — A's wipe never touched it")


def _all_tests():
    return [
        test_start_backfill_if_needed_captures_generation_before_submitting,
        test_control_backfill_completes_normally_with_no_wipe_in_between,
        test_backfill_discarded_before_starting_when_wipe_ran_before_first_turn,
        test_wiped_record_is_a_permanent_refusal_on_its_own_terms,
        test_backfill_discarded_at_final_write_when_wipe_ran_during_the_loop,
        test_backfill_discarded_at_final_write_when_wipe_ran_after_the_last_extraction_call,
        test_backfill_summary_rollup_discarded_after_a_wipe_ran_since_submission,
        test_control_backfill_summary_rollup_builds_normally_with_no_wipe,
        test_rollup_wiring_alone_discards_a_wipe_that_arrives_after_the_facts_write,
        test_another_conversations_backfill_is_unaffected_by_this_ones_wipe,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R4/W2 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
