"""
v3.1.9.4 R4 / W1 (P15-5 follow-up, "Found, not fixed" from R1):
portability.import_conversation's overwrite clear is the same class of wipe
path main._clear_all_memory and commands._handle_retire's apply step already
bump for, and it did not.

The shape: `import_conversation(bundle, target_conv_id=X, overwrite=True)`
replaces X's facts (save_facts, which atomically overwrites the file) and,
when X had existing/unverifiable state, also clears X's facts archive,
chapter archive and persona. None of that bumped X's wipe generation, so a
memory tail submitted against X BEFORE the import — still parked on the
pool's concurrency semaphore, or mid a vLLM extraction call — captured a
generation that stayed valid straight through the import, and its eventual
write landed on top of the freshly-imported store: the target conversation's
OLD, pre-import facts reappearing after an operator explicitly replaced them.

Each check: a CONTROL (no import overwrite in between — the tail's write
lands normally), the defect shape (import runs after the tail's generation
was captured, before the tail's write runs — nothing lands), and a check
that an overwrite onto a target with nothing to replace does not bump at all
(the bump is conditioned on actually wiping something, matching
main._clear_all_memory's "only when there is something to clear" shape one
layer up — though that one always bumps; here the condition is intrinsic:
"is this overwrite actually replacing state").

All fixtures are synthetic; no HTTP calls happen (extraction is stubbed).

Run: python test_v3194_r4_w1.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-r4-w1-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "true"

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
import portability  # noqa: E402
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


def _bundle(facts_list, source_conv_id="src-not-used"):
    return {
        "version": portability.BUNDLE_VERSION,
        "exported_at": 0,
        "source_conv_id": source_conv_id,
        "facts": facts_list,
        "summary_state": {},
        "episodic": [],
    }


class _NewFactExtraction:
    """Same shape as test_v3194_r3_r1.py's own: always reports exactly one
    new fact, so a landed write is unambiguous and a discarded write is
    equally unambiguous (the store stays exactly what the import left)."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def __call__(self, client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
        self.calls += 1
        return [self.text]


# ---------------------------------------------------------------------------
# 1. CONTROL: a tail's write lands normally when no overwrite import runs in
#    between.
# ---------------------------------------------------------------------------

def test_control_facts_tail_writes_normally_with_no_import_in_between():
    print("\n[test] CONTROL: a tail's write lands when nothing overwrote the conversation in between")
    _wipe_storage()
    cid = "w1-control"
    facts.save_facts(cid, [{"text": "an existing fact", "added_turn": 1, "last_used": 1}])
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "a fresh conversation starts at generation 0")

    stub = _NewFactExtraction("The control fact should land.")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(main._facts_tail(
            cid, [{"text": "an existing fact", "added_turn": 1, "last_used": 1}],
            "a user turn", "an assistant reply", 2,
            wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(stub.calls == 1, "the extraction call actually ran")
    kept = facts.load_facts(cid)
    check(
        any(f["text"] == "The control fact should land." for f in kept),
        f"the new fact landed on disk: {kept!r}",
    )


# ---------------------------------------------------------------------------
# 2. The defect shape: a tail parked before an overwrite import must not
#    undo it.
# ---------------------------------------------------------------------------

def test_overwrite_import_bumps_the_generation():
    print("\n[test] W1: an overwrite import onto a target with existing state bumps its wipe generation")
    _wipe_storage()
    cid = "w1-bumps"
    facts.save_facts(cid, [{"text": "her old fact, about to be replaced", "added_turn": 1, "last_used": 1}])
    check(memory.current_wipe_generation(cid) == 0, "starts at generation 0")

    result = portability.import_conversation(
        _bundle([{"text": "the imported fact", "added_turn": 5, "last_used": 5}]),
        target_conv_id=cid, overwrite=True,
    )
    check(result["overwrote_existing"] is True, "the import reports it replaced existing state")
    check(memory.current_wipe_generation(cid) == 1, "the overwrite bumped the generation")


def test_facts_tail_discards_its_write_after_an_overwrite_import_ran_since_submission():
    print("\n[test] W1: a tail's captured generation is stale after an overwrite import ran "
          "in between — nothing it would have written reaches disk, and the imported facts "
          "are exactly what survives")
    _wipe_storage()
    cid = "w1-parked-tail"
    facts.save_facts(cid, [{"text": "her old fact, about to be replaced", "added_turn": 1, "last_used": 1}])

    # Step 1: the tail is SUBMITTED — captures the generation as it stands
    # right now, exactly like main._run_memory_tail's own first statement.
    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no import has happened yet")

    # Step 2: the tail is parked (nothing runs yet — synchronous up to here,
    # exactly like the request path between capture and bgwork.pool actually
    # scheduling the coroutine).

    # Step 3: an operator runs an overwrite import, BEFORE the parked tail
    # gets its turn — the exact production shape: a restore, a migration, a
    # cross-pod move landing on a conv_id that still has a live tail behind
    # it from before the restore was kicked off.
    result = portability.import_conversation(
        _bundle([{"text": "the imported fact", "added_turn": 5, "last_used": 5}]),
        target_conv_id=cid, overwrite=True,
    )
    check(result["overwrote_existing"] is True, "the import replaced the target's existing state")
    check(memory.current_wipe_generation(cid) == 1, "the import bumped the generation")
    check(
        [f["text"] for f in facts.load_facts(cid)] == ["the imported fact"],
        f"only the imported fact is on disk right after the import: {facts.load_facts(cid)!r}",
    )

    # Step 4: the parked tail FINALLY runs, long after being submitted, with
    # its now-stale captured generation. It would extract and store a new
    # fact (built from the pre-import exchange) if this fix did not exist —
    # landing beside, or worse racing to overwrite, the bundle that was just
    # imported.
    stub = _NewFactExtraction("A fact from the conversation the import just replaced.")
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = stub
    try:
        asyncio.run(main._facts_tail(
            cid, [{"text": "her old fact, about to be replaced", "added_turn": 1, "last_used": 1}],
            "a user turn", "an assistant reply", 2,
            wipe_generation=wipe_generation,
        ))
    finally:
        facts.extract_facts_from_exchange = orig

    check(stub.calls == 0, "the extraction call never even ran — discarded before spending a vLLM call")
    check(
        [f["text"] for f in facts.load_facts(cid)] == ["the imported fact"],
        f"still exactly the imported fact — the parked tail's write never landed: {facts.load_facts(cid)!r}",
    )


def test_archive_chapter_and_persona_are_also_protected():
    print("\n[test] W1: the archive/chapter/persona clear that rides along with an "
          "overwrite import is covered by the SAME bump (one generation, one check) "
          "— proven via the summary-rollup write site, the other layer a parked tail "
          "can resurrect")
    _wipe_storage()
    cid = "w1-rollup-parked-tail"
    # Seed L3-ready summary state, matching test_v3194_r3_r1.py's own trick.
    # import_conversation's pre-flight only checks state.get("l1") (not l2/
    # l3/persona) to decide `pre_existing` — see its own comment — so l1
    # must be non-empty here for the overwrite branch (and therefore the
    # bump) to fire at all.
    st = summarizer._empty_state(cid)
    st["l1"] = [{"text": "an old L1 chunk.", "first_turn": 1, "last_turn": 50}]
    st["l2"] = [
        {"text": "chapter one.", "first_turn": 1, "last_turn": 100},
        {"text": "chapter two.", "first_turn": 101, "last_turn": 200},
    ]
    st["l3"] = None
    st["last_summarized_turn"] = 200
    st["turns_seen"] = 200
    summarizer.save_state(cid, st)
    persona.save_persona(cid, "a persona the import is about to replace", source="test")

    wipe_generation = memory.current_wipe_generation(cid)
    check(wipe_generation == 0, "no import has happened yet")

    result = portability.import_conversation(
        _bundle([]), target_conv_id=cid, overwrite=True,
    )
    check(result["overwrote_existing"] is True, "the import replaced the target's existing state")
    check(memory.current_wipe_generation(cid) == 1, "the import bumped the generation")
    check(persona.load_persona(cid) is None, "the old persona was cleared by the overwrite")

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
                    return {"choices": [{"message": {"content": "resurrected theme summary."}, "finish_reason": "stop"}]}
            if str(url).endswith("/tokenize"):
                raise RuntimeError("no /tokenize in this stub")
            return _Resp()

    # The parked tail still carries the OLD l2 state's worth of stale
    # messages — main._rollup_hierarchy is the real submission path,
    # including the wipe_generation contextvar wiring.
    stale_messages = [{"role": "system", "content": "sys"}]
    for i in range(2):
        stale_messages.append({"role": "user", "content": f"turn {i}"})
        stale_messages.append({"role": "assistant", "content": f"reply {i}"})

    orig_client = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _StubClient()
    try:
        asyncio.run(main._rollup_hierarchy(
            cid, stale_messages, "final synthetic reply", wipe_generation=wipe_generation,
        ))
    finally:
        summarizer.httpx.AsyncClient = orig_client

    after = summarizer.load_state(cid)
    check(not after.get("l1"), f"no L1 chunk was resurrected from the pre-import history: {after.get('l1')!r}")
    check(after.get("last_summarized_turn", 0) == 0, "the watermark did not move — the parked rollup was discarded whole")


def test_overwrite_onto_a_genuinely_empty_target_does_not_bump():
    print("\n[test] an overwrite=True import onto a target with nothing pre-existing "
          "does not bump — there is nothing for a parked tail to undo")
    _wipe_storage()
    cid = "w1-fresh-target"
    check(memory.current_wipe_generation(cid) == 0, "starts at generation 0")

    result = portability.import_conversation(
        _bundle([{"text": "brand new fact", "added_turn": 1, "last_used": 1}]),
        target_conv_id=cid, overwrite=True,
    )
    check(result["overwrote_existing"] is False, "the import correctly reports nothing was replaced")
    check(memory.current_wipe_generation(cid) == 0, "the generation did not move — nothing was wiped")


def test_import_without_overwrite_onto_an_empty_target_does_not_bump():
    print("\n[test] CONTROL: the ordinary overwrite=False import path (the common case: "
          "restoring into a fresh conv_id) is unaffected and does not bump either")
    _wipe_storage()
    cid = "w1-plain-import"
    check(memory.current_wipe_generation(cid) == 0, "starts at generation 0")

    result = portability.import_conversation(
        _bundle([{"text": "a fact", "added_turn": 1, "last_used": 1}]),
        target_conv_id=cid, overwrite=False,
    )
    check(result["overwrote_existing"] is False, "nothing existed to overwrite")
    check(memory.current_wipe_generation(cid) == 0, "the generation did not move")
    check([f["text"] for f in facts.load_facts(cid)] == ["a fact"], "the import still landed normally")


def test_another_conversations_tail_is_unaffected_by_this_ones_overwrite_import():
    print("\n[test] an overwrite import on conversation A does not discard conversation B's tail")
    _wipe_storage()
    conv_a, conv_b = "w1-conv-a", "w1-conv-b"
    facts.save_facts(conv_a, [{"text": "A's old fact", "added_turn": 1, "last_used": 1}])

    wipe_generation_b = memory.current_wipe_generation(conv_b)
    check(wipe_generation_b == 0, "B starts at generation 0")

    portability.import_conversation(
        _bundle([{"text": "A's imported fact", "added_turn": 5, "last_used": 5}]),
        target_conv_id=conv_a, overwrite=True,
    )
    check(memory.current_wipe_generation(conv_a) == 1, "A's generation moved")
    check(memory.current_wipe_generation(conv_b) == 0, "B's generation is untouched by A's import")

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
          "B's write landed — A's overwrite import never touched it")


def _all_tests():
    return [
        test_control_facts_tail_writes_normally_with_no_import_in_between,
        test_overwrite_import_bumps_the_generation,
        test_facts_tail_discards_its_write_after_an_overwrite_import_ran_since_submission,
        test_archive_chapter_and_persona_are_also_protected,
        test_overwrite_onto_a_genuinely_empty_target_does_not_bump,
        test_import_without_overwrite_onto_an_empty_target_does_not_bump,
        test_another_conversations_tail_is_unaffected_by_this_ones_overwrite_import,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R4/W1 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
