"""
CPU-only tests for what a user gets when a memory file cannot be read
(REMEDIATION.md v3.1, changes 2 and 3).

Two surfaces, one underlying condition:

  - main._clear_all_memory: an unreadable facts file must not abort the whole
    wipe. The user asked for this data to be gone; refusing to clear the three
    layers we CAN read leaves more behind than clearing them does. It clears
    what it can, reports "unreadable": ["facts"], and leaves the facts file
    itself on disk — it cannot be safely rewritten from an unknown state.

  - the slash-command reply: a real person types these into a chat box. What
    reaches her must say that nothing was deleted and that she did nothing
    wrong. The absolute path and the exception class belong in the log, and
    only in the log. Asserting the path is absent from the reply is the actual
    requirement, so that is what is asserted — not merely that some friendly
    words are present.

The corrupt-file trigger is the same one test_store_failures.py uses and for
the same reason (REMEDIATION.md §1.4): a genuinely corrupt file on disk is
what a torn write leaves behind, and it reaches read_json_strict's
JSONDecodeError arm without any patching of Path.stat.

Run: python test_degraded_forget.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-degraded-forget-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "2000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_PERSONA_AUTO_DETECT_MIN_CHARS"] = "50"

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

# raise_server_exceptions=False so an unhandled exception inside the handler
# arrives as a real 500 response instead of propagating into the test runner.
# A user-facing surface that 500s is a result worth asserting on, not a crash
# worth hiding.
client = TestClient(main.app, raise_server_exceptions=False)

# Separate client for the admin surface: client=127.0.0.1 satisfies
# _require_localhost without loosening COMPACTOR_ADMIN_BIND, so the gate stays
# genuinely under test on the chat client above.
admin_client = TestClient(
    main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False
)


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


def _wipe_storage():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


def corrupt(path, content: str = "{ not valid js"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _seed_other_layers(cid: str) -> None:
    """Give the conv a real summary and a real persona so the assertions below
    can tell "cleared" from "was never there"."""
    summarizer.save_state(cid, {
        "l1": [{"text": "Scene one.", "first_turn": 1, "last_turn": 20}],
        "l2": [], "l3": None, "last_summarized_turn": 20,
    })
    persona.save_persona(
        cid, "You are Sam Cole, a hardboiled detective in a rain-slick city.",
        source="admin",
    )


def assert_no_path_leak(text: str, path, label: str) -> None:
    """The reply must not carry the storage path in any form a chat client
    would render. Checked in three shapes because a Windows path survives a
    round trip through json.dumps as escaped backslashes, and because the
    bare filename alone is still an internal detail the user cannot act on.
    """
    p = str(path)
    variants = {p, p.replace("\\", "/"), p.replace("\\", "\\\\")}
    for v in variants:
        if v in text:
            print(f"FAIL {label}: reply contains the storage path {v!r}\n"
                  f"       reply was: {text!r}")
            sys.exit(1)
    if _TMP_ROOT in text or _TMP_ROOT.replace("\\", "/") in text:
        print(f"FAIL {label}: reply contains the storage root\n"
              f"       reply was: {text!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def _command_reply(text: str, conv_id: str) -> str:
    r = client.post("/v1/chat/completions", json={
        "model": "x",
        "messages": [{"role": "user", "content": text}],
        "stream": False,
    }, headers={"X-Conversation-Id": conv_id})
    assert_eq(r.status_code, 200, f"{text} short-circuits with 200")
    return r.json()["choices"][0]["message"]["content"]


def _wipe(cid: str) -> dict:
    """Drive _clear_all_memory and report a raise as a failed assertion rather
    than an unhandled traceback. It is documented not to raise on an
    unreadable layer, so a raise is this file's finding to state, not a
    fixture problem to swallow."""
    try:
        return asyncio.run(main._clear_all_memory(cid, source="test"))
    except Exception as e:
        print(f"FAIL _clear_all_memory raised {type(e).__name__}: {e}\n"
              f"       it is documented to clear what it can and report the "
              f"rest in 'unreadable'")
        sys.exit(1)


# ---------------------------------------------------------------------------
# B — _clear_all_memory clears what it can read
# ---------------------------------------------------------------------------

def test_clear_all_memory_partial_wipe_on_corrupt_facts():
    print("\n[test] a corrupt facts file does not abort the wipe of the other layers")
    _wipe_storage()
    cid = "partial-wipe"
    before = corrupt(memory.facts_path(cid)).read_text(encoding="utf-8")
    _seed_other_layers(cid)
    # A non-zero episodic count proves the wipe reached the layer AFTER the
    # facts failure rather than unwinding at it. The real function returns 0
    # here (chromadb absent), which cannot distinguish "cleared nothing" from
    # "never ran".
    with patch.object(retrieval, "forget_conversation", lambda c: 7):
        result = _wipe(cid)

    assert_eq(result["unreadable"], ["facts"], "unreadable names exactly facts")
    assert_eq(result["forgotten_facts"], 0, "no fact count claimed")
    assert_eq(result["forgotten_episodic"], 7, "episodic still cleared")
    assert_eq(result["forgotten_summary"], True, "summary still cleared")
    assert_eq(result["forgotten_persona"], True, "persona still cleared")
    assert_true(not memory.summary_path(cid).is_file(), "summary file gone")
    assert_true(not memory.persona_path(cid).is_file(), "persona file gone")
    # The point of leaving it: its real contents are unknown, so there is no
    # safe value to rewrite it with. Deleting it would be a decision made on
    # data nobody has read.
    assert_true(memory.facts_path(cid).is_file(), "facts file still in place")
    assert_eq(memory.facts_path(cid).read_text(encoding="utf-8"), before,
              "facts file byte-identical — not rewritten from an unknown state")


def test_clear_all_memory_does_not_raise_on_corrupt_facts():
    print("\n[test] the corrupt-facts wipe returns a dict rather than raising")
    _wipe_storage()
    cid = "no-raise"
    corrupt(memory.facts_path(cid))
    result = _wipe(cid)
    assert_true(isinstance(result, dict), "returned a counters dict")
    assert_eq(result["unreadable"], ["facts"], "unreadable reported in the dict")


def test_clear_all_memory_healthy_reports_nothing_unreadable():
    print("\n[test] the healthy wipe still reports an empty unreadable list")
    _wipe_storage()
    cid = "healthy-wipe"
    facts.save_facts(cid, [
        {"text": "Lyra is a half-elf ranger.", "added_turn": 1, "last_used": 1},
        {"text": "The setting is Aethermere.", "added_turn": 2, "last_used": 2},
    ])
    _seed_other_layers(cid)
    result = _wipe(cid)
    assert_eq(result["unreadable"], [], "nothing unreadable")
    assert_eq(result["forgotten_facts"], 2, "both facts counted")
    assert_eq(facts.load_facts(cid), [], "facts cleared on disk")
    assert_true(not memory.summary_path(cid).is_file(), "summary file gone")
    assert_true(not memory.persona_path(cid).is_file(), "persona file gone")


def test_clear_all_memory_absent_facts_is_not_unreadable():
    print("\n[test] a conv with no facts file at all is not reported unreadable")
    # The distinction the whole v3.1 chain rests on: absent is empty, and only
    # unreadable is unreadable. If this ever starts reporting ["facts"], every
    # /forget on a fresh conversation tells the user something is broken.
    _wipe_storage()
    result = _wipe("never-seen")
    assert_eq(result["unreadable"], [], "absent facts file -> unreadable is empty")
    assert_eq(result["forgotten_facts"], 0, "nothing to forget")


def test_admin_forget_endpoint_reports_unreadable():
    print("\n[test] DELETE /admin/conversations/{id}/facts -> 200 carrying unreadable")
    # The operator-facing half of the same contract. A partial wipe is a
    # result, not a server error: the caller needs the counters AND the list
    # of layers that could not be read, which a 500 gives it neither of.
    _wipe_storage()
    cid = "admin-forget-corrupt"
    corrupt(memory.facts_path(cid))
    r = admin_client.delete(f"/admin/conversations/{cid}/facts")
    assert_eq(r.status_code, 200, "200, not 500")
    assert_eq(r.json()["unreadable"], ["facts"],
              "response body names the layer it could not read")


def test_forget_command_must_not_report_a_clean_wipe():
    print("\n[test] /forget must not say 'nothing to forget' when a layer was unreadable")
    # main.py:1924 states the contract this asserts: "Callers must not report a
    # clean wipe when this is non-empty." A facts file with unknown contents is
    # sitting on disk; telling the user the conversation had no stored memory
    # is the single most misleading thing this surface can say.
    _wipe_storage()
    cid = "forget-cmd-corrupt"
    corrupt(memory.facts_path(cid))
    reply = _command_reply("/forget", cid)
    assert_true(
        "no stored memory" not in reply.lower(),
        f"reply does not claim the conversation had no stored memory: {reply!r}",
    )
    assert_true(
        any(w in reply.lower() for w in ("could not", "couldn't", "unreadable",
                                         "unable")),
        f"reply says a layer could not be read: {reply!r}",
    )


# ---------------------------------------------------------------------------
# C — the slash-command reply on StoreUnreadable
# ---------------------------------------------------------------------------

def test_command_reply_hides_path_when_persona_unreadable():
    print("\n[test] /why with a corrupt persona -> plain language, no path, no class name")
    _wipe_storage()
    cid = "cmd-persona-corrupt"
    p = corrupt(memory.persona_path(cid))
    reply = _command_reply("/why", cid)
    assert_no_path_leak(reply, p, "no storage path in the reply")
    assert_true("StoreUnreadable" not in reply, "no exception class name in the reply")
    assert_true("JSONDecodeError" not in reply, "no underlying exception name either")
    assert_true("Nothing has been deleted" in reply,
                f"reply reassures that nothing was deleted: {reply!r}")


def test_command_reply_streams_the_same_plain_language():
    print("\n[test] the streaming path carries the same plain-language reply")
    # OpenWebUI streams by default, so the non-streaming assertion alone would
    # leave the shape most users actually see untested.
    _wipe_storage()
    cid = "cmd-persona-corrupt-stream"
    p = corrupt(memory.persona_path(cid))
    r = client.post("/v1/chat/completions", json={
        "model": "x",
        "messages": [{"role": "user", "content": "/why"}],
        "stream": True,
    }, headers={"X-Conversation-Id": cid})
    assert_eq(r.status_code, 200, "stream opens 200")
    # Pull the assistant text back out of the SSE frames so the assertion runs
    # against what the client renders, not against the escaped wire bytes.
    content = ""
    for line in r.text.splitlines():
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: "):])
        content += chunk["choices"][0].get("delta", {}).get("content", "") or ""
    assert_true("Nothing has been deleted" in content,
                f"plain language present in the stream: {content!r}")
    assert_no_path_leak(content, p, "no storage path in the streamed reply")


def test_command_reply_hides_path_when_facts_unreadable():
    print("\n[test] /list-facts with a corrupt facts file -> plain language, no path")
    # Same condition, different layer, and the one a user is most likely to
    # hit: /list-facts is the command people type when they suspect memory
    # trouble in the first place.
    _wipe_storage()
    cid = "cmd-facts-corrupt"
    p = corrupt(memory.facts_path(cid))
    reply = _command_reply("/list-facts", cid)
    assert_no_path_leak(reply, p, "no storage path in the reply")
    assert_true("StoreUnreadable" not in reply, "no exception class name in the reply")
    assert_true("Nothing has been deleted" in reply,
                f"reply reassures that nothing was deleted: {reply!r}")


def test_command_reply_unaffected_when_store_is_healthy():
    print("\n[test] a healthy store still gets the normal command output")
    _wipe_storage()
    cid = "cmd-healthy"
    facts.save_facts(cid, [{"text": "Lyra is a half-elf ranger.",
                            "added_turn": 1, "last_used": 1}])
    reply = _command_reply("/list-facts", cid)
    assert_true("Lyra is a half-elf ranger." in reply, "the fact is listed")
    assert_true("Nothing has been deleted" not in reply,
                "the degraded message does not fire on a healthy store")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    # "does not raise" runs first deliberately: if the wipe blows up, that is
    # the one fact worth reading, and every assertion after it is downstream
    # noise. The runner is fail-fast, so order is the diagnosis.
    return [
        test_clear_all_memory_does_not_raise_on_corrupt_facts,
        test_clear_all_memory_partial_wipe_on_corrupt_facts,
        test_clear_all_memory_healthy_reports_nothing_unreadable,
        test_clear_all_memory_absent_facts_is_not_unreadable,
        test_admin_forget_endpoint_reports_unreadable,
        test_forget_command_must_not_report_a_clean_wipe,
        test_command_reply_hides_path_when_persona_unreadable,
        test_command_reply_streams_the_same_plain_language,
        test_command_reply_hides_path_when_facts_unreadable,
        test_command_reply_unaffected_when_store_is_healthy,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll degraded-forget tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
