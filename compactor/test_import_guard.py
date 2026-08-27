"""
CPU-only tests for the v3.1 import/fork pre-flight guard (portability.py).

This file exists because test_portability.py could not see the regression it
was supposed to be covering. That file replaces retrieval.conversation_doc_count
wholesale with a stub that returns len(dict) — an int, always — so the one
value the guard now has to reason about, None, never reached the code under
test. Meanwhile the real function returns None whenever the vector store is
unavailable (retrieval.py:250), and the pre-flight did
`conversation_doc_count(target) > 0`. int > None is a TypeError, the endpoint
had no handler for it, and every import and every fork answered HTTP 500 for
as long as the retrieval latch was tripped.

So: conversation_doc_count is NOT stubbed at module scope here. Retrieval is
left ENABLED while its dependencies are absent, so _try_init() fails and the
real function returns None on its own — the production shape of the outage,
reproduced rather than imitated. The two tests that need a genuine integer
patch it for the duration of that one test and say why, which is the opposite
of erasing the None case globally.

NOT `COMPACTOR_RAG_ENABLED=false`, which an earlier draft used and which no
longer produces None at all. Disabling retrieval deliberately means a
conversation genuinely has nothing indexed, so v3.1 returns 0 for that — and it
must, because reading a supported configuration as "cannot verify" made the
pre-flight refuse every import and every fork with a 400, including onto
freshly-minted fork ids that could not possibly be occupied. Enabled-but-
unreachable is the only state that legitimately yields None, and it is the one
this file needs.

The guard's contract, from portability.py:146-185: "I could not check" must be
read as "occupied", never as "empty". An import that guesses wrong here does
not fail — it silently wipes a live conversation.

Run: python test_import_guard.py
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import patch

# Storage redirect before importing anything that resolves a module-level path.
# RAG ENABLED + the retrieval deps absent is what makes conversation_doc_count
# return None for real: _try_init() tries, and latches unavailable on the first
# missing dep — fastembed, since it is imported before chromadb; either one
# trips it. Setting RAG to "false" instead returns 0 by design and would
# silently stop this file testing anything. MODEL_REPO unset so importing main
# doesn't reach for a tokenizer.
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-import-guard-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "2000"
os.environ["COMPACTOR_RAG_ENABLED"] = "true"

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import portability  # noqa: E402
import retrieval  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

# client=127.0.0.1 so _require_localhost is satisfied without loosening
# COMPACTOR_ADMIN_BIND — the gate stays genuinely under test.
# raise_server_exceptions=False so an unhandled exception inside the endpoint
# comes back as a real 500 response instead of propagating into the test. That
# is the whole point of the endpoint assertions below: 400 and 500 have to be
# distinguishable.
client = TestClient(
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


def assert_raises(fn, exc_type, label):
    """Returns the exception so callers can assert on its message."""
    try:
        fn()
    except exc_type as e:
        print(f"  ok   {label}")
        return e
    except Exception as e:
        print(f"FAIL {label}: expected {exc_type.__name__}, "
              f"got {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"FAIL {label}: expected {exc_type.__name__}, nothing raised")
    sys.exit(1)


def _wipe_storage():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


def corrupt(path, content: str = "{ not valid js"):
    """A genuinely corrupt file on disk — what a torn write leaves behind, and
    what makes load_facts / load_state raise StoreUnreadable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _bundle(*, facts_list=None, summary=None, episodic=None, source="src-conv"):
    return {
        "version": portability.BUNDLE_VERSION,
        "exported_at": 0,
        "source_conv_id": source,
        "facts": facts_list if facts_list is not None else [
            {"text": "Lyra is a half-elf ranger.", "added_turn": 1,
             "last_used": 1748000000},
        ],
        "summary_state": summary if summary is not None else {},
        "episodic": episodic if episodic is not None else [],
    }


def _counting(n: int):
    """A conversation_doc_count that returns a real integer.

    Patched in for exactly the two tests that are about the healthy path, and
    only for the duration of those tests. Left in place at module scope this
    would recreate test_portability.py's blind spot.
    """
    return lambda conv_id: n


# ---------------------------------------------------------------------------
# Fixture self-check — the None case has to be real, not arranged
# ---------------------------------------------------------------------------

def test_doc_count_really_returns_none_here():
    print("\n[test] the unstubbed conversation_doc_count returns None in this fixture")
    # If this ever starts returning an int, every refusal test below stops
    # testing the regression and starts testing the happy path while still
    # printing ok. Assert the precondition instead of assuming it.
    got = retrieval.conversation_doc_count("anything")
    assert_eq(got, None, "retrieval unavailable -> conversation_doc_count() is None")


# ---------------------------------------------------------------------------
# The regression: unverifiable episodic layer
# ---------------------------------------------------------------------------

def test_import_refuses_when_episodic_unverifiable():
    print("\n[test] doc_count None + overwrite=False -> ImportError_, not TypeError")
    _wipe_storage()
    e = assert_raises(
        lambda: portability.import_conversation(
            _bundle(), target_conv_id="unverifiable-target", overwrite=False
        ),
        portability.ImportError_,
        "raises ImportError_ (the pre-v3.1 code raised TypeError here)",
    )
    msg = str(e)
    assert_true("episodic" in msg,
                f"message names the layer that could not be verified: {msg!r}")
    assert_true("unverifiable-target" in msg, "message names the target conv_id")
    assert_true("overwrite" in msg, "message tells the caller how to proceed")


def test_import_refuses_does_not_touch_target():
    print("\n[test] a refusal writes nothing — the target is left exactly as found")
    _wipe_storage()
    cid = "untouched-target"
    facts.save_facts(cid, [{"text": "do not lose me", "added_turn": 0,
                            "last_used": 100}])
    before = memory.facts_path(cid).read_text(encoding="utf-8")
    err = assert_raises(
        lambda: portability.import_conversation(
            _bundle(), target_conv_id=cid, overwrite=False
        ),
        portability.ImportError_,
        "refused",
    )
    # An earlier version of this test passed for the wrong reason: the target
    # holds real facts, so the "has existing state" branch fires first and the
    # unverifiable branch — the one this file exists to cover — was never
    # reached. Pin which refusal we got, or this silently stops testing the
    # guard the moment the ordering changes.
    assert_true("cannot verify" in str(err).lower(),
                "refused on 'unverifiable', not merely on 'has existing state'")
    assert_eq(memory.facts_path(cid).read_text(encoding="utf-8"), before,
              "facts file byte-identical after the refusal")


def test_import_proceeds_when_unverifiable_and_overwrite():
    print("\n[test] doc_count None + overwrite=True -> the import runs")
    _wipe_storage()
    cid = "unverifiable-forced"
    result = portability.import_conversation(
        _bundle(), target_conv_id=cid, overwrite=True
    )
    assert_eq(result["conv_id"], cid, "result names the target")
    assert_eq(result["imported"]["facts"], 1, "one fact imported")
    landed = facts.load_facts(cid)
    assert_eq(len(landed), 1, "the fact is on disk")
    assert_eq(landed[0]["text"], "Lyra is a half-elf ranger.", "fact text landed")


# ---------------------------------------------------------------------------
# Unverifiable facts / summaries — the StoreUnreadable half of the guard
# ---------------------------------------------------------------------------

def test_import_refuses_when_facts_file_corrupt():
    print("\n[test] corrupt facts file + overwrite=False -> ImportError_ naming facts")
    _wipe_storage()
    cid = "corrupt-facts-target"
    before = corrupt(memory.facts_path(cid)).read_text(encoding="utf-8")
    # doc_count pinned to 0 so "facts" is the ONLY thing that can be
    # unverifiable — otherwise the message names two layers and the assertion
    # below would pass for the wrong reason.
    with patch.object(retrieval, "conversation_doc_count", _counting(0)):
        e = assert_raises(
            lambda: portability.import_conversation(
                _bundle(), target_conv_id=cid, overwrite=False
            ),
            portability.ImportError_,
            "raises ImportError_",
        )
    assert_true("facts" in str(e), f"message names facts: {str(e)!r}")
    assert_true("episodic" not in str(e),
                "message does NOT name episodic — that layer verified clean")
    assert_eq(memory.facts_path(cid).read_text(encoding="utf-8"), before,
              "corrupt facts file left exactly as found")


def test_import_refuses_when_summaries_corrupt():
    print("\n[test] corrupt summary file + overwrite=False -> ImportError_ naming summaries")
    _wipe_storage()
    cid = "corrupt-summary-target"
    before = corrupt(memory.summary_path(cid)).read_text(encoding="utf-8")
    with patch.object(retrieval, "conversation_doc_count", _counting(0)):
        e = assert_raises(
            lambda: portability.import_conversation(
                _bundle(), target_conv_id=cid, overwrite=False
            ),
            portability.ImportError_,
            "raises ImportError_",
        )
    assert_true("summaries" in str(e), f"message names summaries: {str(e)!r}")
    assert_eq(memory.summary_path(cid).read_text(encoding="utf-8"), before,
              "corrupt summary file left exactly as found")


# ---------------------------------------------------------------------------
# The healthy paths still behave exactly as they did before v3.1
# ---------------------------------------------------------------------------

def test_genuinely_empty_target_imports():
    print("\n[test] doc_count 0 + no facts + no summary -> imports without overwrite")
    _wipe_storage()
    cid = "genuinely-empty"
    with patch.object(retrieval, "conversation_doc_count", _counting(0)):
        result = portability.import_conversation(
            _bundle(), target_conv_id=cid, overwrite=False
        )
    assert_eq(result["imported"]["facts"], 1, "one fact imported")
    assert_eq(result["overwrote_existing"], False, "nothing was overwritten")
    assert_eq(len(facts.load_facts(cid)), 1, "fact is on disk")


def test_genuinely_occupied_target_refuses():
    print("\n[test] doc_count 0 but facts present -> refuses without overwrite")
    _wipe_storage()
    cid = "genuinely-occupied"
    facts.save_facts(cid, [{"text": "an established fact", "added_turn": 0,
                            "last_used": 100}])
    with patch.object(retrieval, "conversation_doc_count", _counting(0)):
        e = assert_raises(
            lambda: portability.import_conversation(
                _bundle(), target_conv_id=cid, overwrite=False
            ),
            portability.ImportError_,
            "refuses on genuine existing state",
        )
    assert_true("existing state" in str(e),
                f"message says existing state, not unverifiable: {str(e)!r}")
    assert_eq(facts.load_facts(cid)[0]["text"], "an established fact",
              "the established fact survived")


def test_genuinely_occupied_episodic_refuses():
    print("\n[test] doc_count > 0 -> refuses without overwrite (0 vs None vs N)")
    _wipe_storage()
    with patch.object(retrieval, "conversation_doc_count", _counting(3)):
        e = assert_raises(
            lambda: portability.import_conversation(
                _bundle(), target_conv_id="episodic-occupied", overwrite=False
            ),
            portability.ImportError_,
            "refuses on a non-zero indexed count",
        )
    assert_true("existing state" in str(e),
                "a countable non-empty store is 'existing state', not 'cannot verify'")


def test_genuinely_occupied_target_overwrites_with_flag():
    print("\n[test] doc_count 0 + facts present + overwrite=True -> replaces")
    _wipe_storage()
    cid = "occupied-overwrite"
    facts.save_facts(cid, [{"text": "old fact", "added_turn": 0, "last_used": 1}])
    with patch.object(retrieval, "conversation_doc_count", _counting(0)):
        result = portability.import_conversation(
            _bundle(), target_conv_id=cid, overwrite=True
        )
    assert_eq(result["overwrote_existing"], True, "overwrote_existing reported")
    assert_eq(facts.load_facts(cid)[0]["text"], "Lyra is a half-elf ranger.",
              "facts replaced wholesale")


# ---------------------------------------------------------------------------
# Fork rides the same guard — it calls import_conversation(overwrite=False)
# ---------------------------------------------------------------------------

def test_fork_refuses_rather_than_typeerrors():
    print("\n[test] fork with the vector store down -> ImportError_, not TypeError")
    # fork_conversation always imports with overwrite=False, so before v3.1
    # every fork attempted while the retrieval latch was tripped raised
    # TypeError out of the endpoint. There is no overwrite escape hatch on the
    # fork endpoint at all, so this path was 500 with no way around it.
    _wipe_storage()
    facts.save_facts("fork-parent", [{"text": "shared truth", "added_turn": 0,
                                      "last_used": 100}])
    e = assert_raises(
        lambda: portability.fork_conversation("fork-parent"),
        portability.ImportError_,
        "fork raises ImportError_",
    )
    assert_true("episodic" in str(e), "fork's message names the unverifiable layer")


# ---------------------------------------------------------------------------
# Endpoint mapping — ImportError_ is a 400, an unhandled TypeError is a 500
# ---------------------------------------------------------------------------

def test_import_endpoint_returns_400_not_500():
    print("\n[test] POST /admin/conversations/import -> 400 when the target is unverifiable")
    _wipe_storage()
    r = client.post("/admin/conversations/import", json={
        "bundle": _bundle(),
        "target_conv_id": "endpoint-unverifiable",
        "overwrite": False,
    })
    assert_eq(r.status_code, 400, "400, not 500 (this was the live regression)")
    detail = r.json().get("detail", "")
    assert_true("episodic" in detail, f"detail names the unverifiable layer: {detail!r}")


def test_import_endpoint_succeeds_with_overwrite():
    print("\n[test] POST /admin/conversations/import -> 200 with overwrite=true")
    _wipe_storage()
    r = client.post("/admin/conversations/import", json={
        "bundle": _bundle(),
        "target_conv_id": "endpoint-forced",
        "overwrite": True,
    })
    assert_eq(r.status_code, 200, "200 when the operator has said overwrite")
    assert_eq(r.json()["imported"]["facts"], 1, "one fact imported")


def test_fork_endpoint_returns_400_not_500():
    print("\n[test] POST /admin/conversations/{id}/fork -> 400 when unverifiable")
    _wipe_storage()
    facts.save_facts("endpoint-fork-parent", [
        {"text": "shared truth", "added_turn": 0, "last_used": 100},
    ])
    r = client.post("/admin/conversations/endpoint-fork-parent/fork", json={})
    assert_eq(r.status_code, 400, "400, not 500")


def test_import_endpoint_400_on_corrupt_facts():
    print("\n[test] POST /admin/conversations/import -> 400 when the target's facts are corrupt")
    _wipe_storage()
    cid = "endpoint-corrupt-facts"
    corrupt(memory.facts_path(cid))
    with patch.object(retrieval, "conversation_doc_count", _counting(0)):
        r = client.post("/admin/conversations/import", json={
            "bundle": _bundle(),
            "target_conv_id": cid,
            "overwrite": False,
        })
    assert_eq(r.status_code, 400, "400, not 500")
    assert_true("facts" in r.json().get("detail", ""), "detail names facts")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_doc_count_really_returns_none_here,
        test_import_refuses_when_episodic_unverifiable,
        test_import_refuses_does_not_touch_target,
        test_import_proceeds_when_unverifiable_and_overwrite,
        test_import_refuses_when_facts_file_corrupt,
        test_import_refuses_when_summaries_corrupt,
        test_genuinely_empty_target_imports,
        test_genuinely_occupied_target_refuses,
        test_genuinely_occupied_episodic_refuses,
        test_genuinely_occupied_target_overwrites_with_flag,
        test_fork_refuses_rather_than_typeerrors,
        test_import_endpoint_returns_400_not_500,
        test_import_endpoint_succeeds_with_overwrite,
        test_fork_endpoint_returns_400_not_500,
        test_import_endpoint_400_on_corrupt_facts,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll import-guard tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
