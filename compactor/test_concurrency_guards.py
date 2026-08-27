"""
CPU-only Tier-1 tests for the RC5 robustness guards.

Covers the two pure helpers added after the 2026-08-10 audit:
  - _merge_touched: the lost-update fix for the async facts tail. The request
    path reads facts OUTSIDE the per-conv lock, so the tail holds a stale
    snapshot; writing it back would erase facts a concurrent tail persisted.
  - _merge_adjacent_system_messages: Mistral-family templates reject runs of
    consecutive system messages with a 400, and V1 compaction + memory
    injection can each add one independently.

...and, since v3.1 F22/D18, the three write paths that took no conv_lock at
all. Those tests are not helper tests: they run a real _async_tail and a real
command handler against a real temp store on one event loop and drive the
interleaving with an asyncio.Event.

Run inside the compactor image or any container with the requirements:
    python test_concurrency_guards.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

# Force the tokenizer-free fallback path before importing main
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"
os.environ["COMPACTOR_TARGET_TOKENS"] = "500"
# The F22/D18 tests write real files. RAG off keeps retrieval a fast no-op
# (and makes conversation_doc_count return 0-means-empty, not None-means-
# unknown, so the import pre-flight has nothing to be unsure about).
_TMP_ROOT = tempfile.mkdtemp(prefix="zions_concurrency_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main  # noqa: E402
import commands  # noqa: E402
import dedup  # noqa: E402
import degrade  # noqa: E402
import facts as facts_module  # noqa: E402
import memory  # noqa: E402
import portability  # noqa: E402
import summarizer  # noqa: E402


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


def f(text, last_used=100, added_turn=1):
    return {"text": text, "added_turn": added_turn, "last_used": last_used}


def test_merge_touched_preserves_concurrent_writes():
    print("\n[test] _merge_touched — the lost-update scenario")
    # The exact audit scenario: our snapshot is [f1]; meanwhile another tail
    # persisted f2. Building on the snapshot would drop f2 forever.
    snapshot = [f("f1", last_used=100)]
    fresh_from_disk = [f("f1", last_used=100), f("f2", last_used=150)]
    merged = main._merge_touched(fresh_from_disk, snapshot)
    assert_eq([m["text"] for m in merged], ["f1", "f2"], "concurrent fact survives")

    print("\n[test] _merge_touched — LRU touch carries over")
    snapshot = [f("f1", last_used=999)]           # request path touched it
    fresh_from_disk = [f("f1", last_used=100)]
    merged = main._merge_touched(fresh_from_disk, snapshot)
    assert_eq(merged[0]["last_used"], 999, "newer touch applied")

    print("\n[test] _merge_touched — never backdates a fresher touch")
    snapshot = [f("f1", last_used=100)]           # our snapshot is older
    fresh_from_disk = [f("f1", last_used=500)]    # someone touched it since
    merged = main._merge_touched(fresh_from_disk, snapshot)
    assert_eq(merged[0]["last_used"], 500, "fresher last_used kept")

    print("\n[test] _merge_touched — forgotten facts stay forgotten")
    # Disk is authoritative for membership: a fact deleted via /forget while
    # the tail was in flight must NOT be resurrected from the snapshot.
    snapshot = [f("f1"), f("deleted-by-forget")]
    fresh_from_disk = [f("f1")]
    merged = main._merge_touched(fresh_from_disk, snapshot)
    assert_eq([m["text"] for m in merged], ["f1"], "deleted fact not resurrected")

    print("\n[test] _merge_touched — edge cases")
    assert_eq(main._merge_touched([], []), [], "both empty")
    assert_eq([m["text"] for m in main._merge_touched([f("a")], [])], ["a"], "empty snapshot")
    assert_eq(main._merge_touched([], [f("a")]), [], "empty disk stays empty")


def test_merge_adjacent_system_messages():
    print("\n[test] _merge_adjacent_system_messages — the Mistral 400 scenario")
    # compaction summary + injected memory block + original system prompt
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": "[Summary of earlier conversation]"},
        {"role": "system", "content": "[facts]"},
        {"role": "user", "content": "hi"},
    ]
    out = main._merge_adjacent_system_messages(msgs)
    assert_eq(len(out), 2, "three system messages collapse to one")
    assert_eq(out[0]["role"], "system", "merged message is system")
    assert_true("persona" in out[0]["content"], "first content preserved")
    assert_true("[facts]" in out[0]["content"], "last content preserved")
    assert_eq(out[1]["role"], "user", "user message untouched")

    print("\n[test] _merge_adjacent_system_messages — non-adjacent left alone")
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "b"},
    ]
    assert_eq(len(main._merge_adjacent_system_messages(msgs)), 3, "separated systems kept")

    print("\n[test] _merge_adjacent_system_messages — text-only list content is flattened")
    # rc6 review: OpenAI content-parts system prompts with NO images must be
    # flattened and merged — leaving them unmerged preserves the adjacent-
    # system run this function exists to prevent.
    msgs = [
        {"role": "system", "content": "text"},
        {"role": "system", "content": [{"type": "text", "text": "part"}]},
    ]
    out = main._merge_adjacent_system_messages(msgs)
    assert_eq(len(out), 1, "text-only parts system merged into the run")
    assert_true("part" in out[0]["content"], "flattened text preserved")

    print("\n[test] _merge_adjacent_system_messages — image content not string-joined")
    # A genuinely image-bearing system message must NOT be collapsed — that
    # would destroy the image parts (V3.1 vision).
    msgs = [
        {"role": "system", "content": "text"},
        {"role": "system", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
    ]
    out = main._merge_adjacent_system_messages(msgs)
    assert_eq(len(out), 2, "image-bearing system message preserved separately")

    print("\n[test] _merge_adjacent_system_messages — passthroughs")
    assert_eq(main._merge_adjacent_system_messages([]), [], "empty list")
    one = [{"role": "user", "content": "hi"}]
    assert_eq(main._merge_adjacent_system_messages(one), one, "single message")


# ---------------------------------------------------------------------------
# v3.1 F22 + D18 + D49 — the write paths that took no conv_lock
# ---------------------------------------------------------------------------
#
# The window these tests drive is exact, and worth stating because "the tail
# re-reads under its lock, so a concurrent write is safe" is the reasoning that
# left the lock off the command handlers for four releases.
#
# _async_tail re-reads facts INSIDE its lock (main.py, `combined = ...`) and
# then awaits AGAIN — inline dedup — before it prunes and saves. A writer that
# lands in that second gap is invisible to the re-read and is overwritten by
# the save. The re-read narrows the window; only the lock closes it.
#
# The command handlers have no await of their own between load and save, so
# nothing can interleave with them. That is precisely why the missing lock was
# invisible in every non-concurrent test: the handler is atomic on the event
# loop and always ends up with a correct-looking file. The lock is what makes
# it QUEUE behind the parked tail instead of reading in front of it.


def _wipe():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


async def _race_against_parked_tail(conv_id, snapshot, extracted, contender):
    """Run _async_tail to the point where it holds conv_lock and has already
    done its re-read, start `contender()` there, then let the tail finish.

    Extraction and dedup are the tail's two awaits; both are stubbed, so the
    only thing this exercises is the write ordering. Returns the contender's
    result.
    """
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fake_extract(*_a, **_k):
        return [extracted]

    async def fake_dedup(_client, _url, _model, combined):
        # The tail is now holding conv_lock, past its re-read, and parked.
        entered.set()
        await release.wait()
        return combined, 0

    with patch.object(facts_module, "extract_facts_from_exchange", fake_extract), \
         patch.object(dedup, "dedup_facts", fake_dedup), \
         patch.object(degrade, "guard", lambda *_a, **_k: True), \
         patch.object(summarizer, "enabled", lambda: False):
        tail = asyncio.create_task(
            main._async_tail(conv_id, snapshot, "user turn", "assistant turn", 9, [])
        )
        await asyncio.wait_for(entered.wait(), timeout=10)
        contender_task = asyncio.create_task(contender())
        # Let the contender reach its first blocking point: the lock acquire
        # once F22 is fixed, or straight through load->save without it.
        for _ in range(10):
            await asyncio.sleep(0)
        release.set()
        result = await asyncio.wait_for(contender_task, timeout=10)
        await asyncio.wait_for(tail, timeout=10)
    return result


def _texts(conv_id) -> set:
    return {x["text"] for x in facts_module.load_facts(conv_id)}


def test_commands_share_the_one_lock_registry():
    print("\n[test] commands.conv_lock is memory's, not a second registry")
    # The remediation plan proposed injecting the lock through ctx to dodge an
    # import cycle with main.py. There is no cycle — conv_lock lives in
    # memory.py and main.py only re-exports it — and a ctx key a caller can
    # forget is an unlocked write waiting to happen. This asserts the thing
    # that actually matters: one lock object per conv_id, shared by all three.
    assert_true(commands.conv_lock is memory.conv_lock, "commands uses memory's conv_lock")
    assert_true(main.conv_lock is memory.conv_lock, "main uses memory's conv_lock")
    assert_true(
        commands.conv_lock("shared-id") is main.conv_lock("shared-id"),
        "same lock object for the same conv_id",
    )


def test_remember_survives_a_tail_parked_mid_write():
    print("\n[test] F22 — /remember racing _async_tail: both facts survive")
    _wipe()
    cid = "race-remember"
    snapshot = [f("established fact", last_used=100)]
    facts_module.save_facts(cid, snapshot)

    async def contender():
        return await commands.handle_command(
            "remember", "the ring is silver", cid, ctx={"turn_index": 10}
        )

    out = asyncio.run(
        _race_against_parked_tail(cid, snapshot, "tail extracted this", contender)
    )
    texts = _texts(cid)
    assert_true("Remembered" in out, "/remember reported success to the user")
    # The reported consequence, both directions at once.
    assert_true("the ring is silver" in texts, "the user's /remember fact survived")
    assert_true("tail extracted this" in texts, "the tail's extraction survived")
    assert_true("established fact" in texts, "the pre-existing fact survived")


def test_selective_forget_is_not_undone_by_a_parked_tail():
    print("\n[test] F22 — /forget <substring> racing _async_tail stays forgotten")
    _wipe()
    cid = "race-forget"
    snapshot = [
        f("Lyra is a half-elf ranger", last_used=100),
        f("Aethermere is a coastal city", last_used=100),
    ]
    facts_module.save_facts(cid, snapshot)

    async def contender():
        return await commands.handle_command("forget", "Lyra", cid)

    out = asyncio.run(
        _race_against_parked_tail(cid, snapshot, "tail extracted this", contender)
    )
    texts = _texts(cid)
    assert_true("Forgot 1" in out, "/forget reported one removal")
    # Unlocked, the tail's pre-forget view was written back on top and the fact
    # the user explicitly deleted reappeared on her next turn.
    assert_true("Lyra is a half-elf ranger" not in texts, "the forgotten fact stayed forgotten")
    assert_true("Aethermere is a coastal city" in texts, "the untargeted fact was not collateral")
    assert_true("tail extracted this" in texts, "the tail's extraction survived")


def test_import_never_reports_success_with_the_bundle_gone():
    print("\n[test] D18 — import racing _async_tail refuses instead of vanishing")
    _wipe()
    cid = "race-import"
    snapshot = [f("established fact", last_used=100)]
    facts_module.save_facts(cid, snapshot)
    bundle = {
        "version": portability.BUNDLE_VERSION,
        "exported_at": 0,
        "source_conv_id": cid,
        "facts": [f("bundle fact 1"), f("bundle fact 2")],
        "summary_state": {},
        "episodic": [],
    }
    outcome = {"raised": None, "returned": None}

    async def contender():
        try:
            outcome["returned"] = portability.import_conversation(
                bundle, target_conv_id=cid, overwrite=True
            )
        except portability.ImportError_ as e:
            outcome["raised"] = str(e)
        return outcome

    asyncio.run(
        _race_against_parked_tail(cid, snapshot, "tail extracted this", contender)
    )
    texts = _texts(cid)
    landed = {"bundle fact 1", "bundle fact 2"} <= texts

    # Stated as the thing that must never happen rather than as an expected
    # store, because the two outcomes leave DIFFERENT stores and only one of
    # them is a bug: refusing is fine, landing is fine, "reported
    # overwrote_existing: true and then a parked tail wrote over it" is D18.
    assert_true(
        outcome["raised"] is not None or landed,
        "import never reports success while its bundle is not on disk",
    )
    assert_true("in flight" in (outcome["raised"] or ""), "the refusal names the conflict")
    assert_true("tail extracted this" in texts, "the tail's own write was not disturbed")

    # And the refusal is transient, not a broken endpoint: the same call
    # succeeds once the writer has let go.
    result = portability.import_conversation(bundle, target_conv_id=cid, overwrite=True)
    assert_eq(result["imported"]["facts"], 2, "the retry lands the bundle")
    assert_eq(_texts(cid), {"bundle fact 1", "bundle fact 2"}, "the retry replaced the store")


def test_index_exchange_runs_under_the_conv_lock():
    print("\n[test] D49 — episodic indexing holds conv_lock while it upserts")
    _wipe()
    cid = "race-index"
    # /forget holds conv_lock across retrieval.forget_conversation. Indexing
    # outside the lock lands after that delete and puts the exchange the user
    # asked to forget back into the vector store — idempotent doc ids make two
    # tails safe against each other and do nothing about this.
    held = {"v": False}

    def spy_index(conv_id, *_a, **_k):
        held["v"] = memory.conv_lock(conv_id).locked()
        return False

    async def run():
        with patch.object(main.retrieval, "index_exchange", spy_index), \
             patch.object(degrade, "guard", lambda *_a, **_k: True), \
             patch.object(summarizer, "enabled", lambda: False), \
             patch.object(facts_module, "extraction_enabled", lambda: False):
            await main._async_tail(cid, [], "user turn", "assistant turn", 3, [])

    asyncio.run(run())
    assert_true(held["v"], "conv_lock was held during index_exchange")


# ---------------------------------------------------------------------------
# v3.1 F9, caller half — the request path touches only what it injects
# ---------------------------------------------------------------------------
#
# facts.select_for_injection exists and is tested in test_facts.py. This is the
# half that made F9 a live bug: main.py loaded the store, touched ALL of it and
# injected ALL of it. Every fact then carried the same second, last_used stopped
# being a recency signal, and eviction fell through to added_turn — deleting the
# conversation's foundational facts first. A test on select_for_injection alone
# cannot see that; only the call site can.


class _FakeVllmResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class _FakeVllmClient:
    """httpx.AsyncClient stand-in for the proxy path. Records the body that
    reached vLLM — which is where the injected facts block is observable, and
    the only place it ever is."""

    def __init__(self, sink):
        self._sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def aclose(self):
        return None

    async def post(self, _url, json=None, **_k):
        self._sink.append(json or {})
        return _FakeVllmResponse()


def test_request_injects_and_touches_only_the_working_set():
    print("\n[test] F9 caller half — only the injected subset is touched and sent")
    _wipe()
    cid = "inject-subset"
    in_set = "Lyra carries a silver ring"
    crowded_out = "the harbour bell rings at dusk"
    facts_module.save_facts(cid, [f(in_set, last_used=100), f(crowded_out, last_used=100)])

    # Stand in for the budget rather than shrinking it: _MAX_FACTS_TOKENS is
    # bound as a default argument at import, so patching the module global
    # would not reach select_for_injection anyway.
    def only_first(fs, *_a, **_k):
        return fs[:1]

    touched: dict = {}
    real_touch = facts_module.touch_facts

    def spy_touch(fs, now=None):
        touched["texts"] = [x["text"] for x in fs]
        return real_touch(fs, now)

    sent: list = []
    from fastapi.testclient import TestClient
    with patch.object(main.facts, "select_for_injection", only_first), \
         patch.object(main.facts, "touch_facts", spy_touch), \
         patch.object(main.facts, "extraction_enabled", lambda: False), \
         patch.object(summarizer, "enabled", lambda: False), \
         patch.object(main.httpx, "AsyncClient", lambda *_a, **_k: _FakeVllmClient(sent)):
        r = TestClient(main.app, raise_server_exceptions=False).post(
            "/v1/chat/completions",
            json={"model": "m", "stream": False,
                  "messages": [{"role": "user", "content": "tell me about the ring"}]},
            headers={"X-Conversation-Id": cid},
        )

    assert_eq(r.status_code, 200, "the proxied request succeeded")
    assert_true(bool(sent), "a body reached the vLLM stand-in")
    forwarded = "\n".join(
        m.get("content") or "" for m in sent[0].get("messages", [])
        if isinstance(m.get("content"), str)
    )
    assert_true(in_set in forwarded, "the working-set fact was injected")
    assert_true(crowded_out not in forwarded, "the crowded-out fact was NOT injected")
    assert_eq(touched.get("texts"), [in_set], "only the injected fact was touched")


if __name__ == "__main__":
    test_merge_touched_preserves_concurrent_writes()
    test_merge_adjacent_system_messages()
    test_commands_share_the_one_lock_registry()
    test_remember_survives_a_tail_parked_mid_write()
    test_selective_forget_is_not_undone_by_a_parked_tail()
    test_import_never_reports_success_with_the_bundle_gone()
    test_index_exchange_runs_under_the_conv_lock()
    test_request_injects_and_touches_only_the_working_set()
    print("\nAll concurrency/guard tests passed.")
