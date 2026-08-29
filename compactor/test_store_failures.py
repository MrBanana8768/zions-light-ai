"""
CPU-only failure-injection tests for the store read/write paths
(REMEDIATION.md F32 / item V1).

Every S1 finding in REMEDIATION.md §3 rides on one primitive: read_json's
"return the default on any error" contract (memory.py:259-270), applied by
callers that then atomically write back. Nothing in compactor/test_*.py
exercised that path — grep found zero occurrences of OSError in any test —
so those findings were arguments in a document rather than failing tests.
This file is the fixture that makes them fail.

Per REMEDIATION.md §1.4, and this is the whole reason the fixture looks the
way it does: Path.is_file() does NOT swallow EIO. CPython's pathlib ignores
only ENOENT, ENOTDIR, EBADF and ELOOP; everything else re-raises, so a stat
failure never reaches read_json's own handler at all. A fixture that patches
Path.stat reproduces NONE of these findings and gives false confidence. The
two real triggers are a genuinely corrupt/truncated file on disk and an
OSError raised from open() or json.load() — MooseFS's characteristic
metadata-fine/chunk-unavailable failure. Both are injected here, against
real files in a real temp store.

STATUS: F1 (read_json_strict + abort-on-UNREADABLE at the write-back sites),
G2 and G3 have landed, and this file is green. It was written red on purpose
— F32 lands before F1 so the fix can be watched going green (§4, constraint
3) — so the per-section headers below still say "EXPECTED TO FAIL until F1".
Read those as the history of each assertion, not as its current state. A red
run now is a regression, not the plan.

The four test_absent_* cases were the exception: they had to pass before F1
and they still do. A genuinely missing file IS an empty store, that is
correct, and F1 must not turn it into an error.

Run: python test_store_failures.py
"""

import asyncio
import contextlib
import errno
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

# Storage redirect MUST happen before importing the modules under test so
# their module-level paths see the override. MODEL_REPO stays unset and RAG
# stays off so importing main doesn't reach for a tokenizer or ChromaDB.
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-store-failures-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "2000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
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


def _wipe_storage():
    """Clean slate between tests."""
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


# ---------------------------------------------------------------------------
# The flaky_fs fixture — the two triggers from §1.4, and nothing else
# ---------------------------------------------------------------------------

def _key(p) -> str:
    """Comparable form of a path. abspath, not resolve(): STORAGE_ROOT is
    stored as a plain string and every path under test is derived from it,
    so the two always agree without dragging symlink resolution in."""
    return os.path.normcase(os.path.abspath(str(p)))


def corrupt(path: Path, content: str = "{ not valid js") -> Path:
    """Trigger 1: a genuinely corrupt / truncated file on disk. This is the
    dominant trigger per §1.4 — it is what a torn write or a half-flushed
    MooseFS chunk actually leaves behind, and it reaches read_json's
    json.JSONDecodeError arm.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@contextlib.contextmanager
def unreadable(path: Path, at: str = "open"):
    """Trigger 2: EIO at read time on one real, present, intact file.

    `at="open"` raises from open(); `at="load"` raises from the read inside
    json.load(). Both land in read_json's OSError arm. Deliberately NOT
    patching Path.stat / Path.is_file — see §1.4: that path re-raises and so
    reproduces none of these findings.

    The file stays genuinely on disk and genuinely intact for the duration,
    which is what makes the write-back tests below meaningful: they can
    assert that the real contents are still there afterwards.
    """
    target = _key(path)
    real_open = open
    real_load = json.load

    def _open(file, *a, **k):
        if isinstance(file, (str, os.PathLike)) and _key(file) == target:
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real_open(file, *a, **k)

    def _load(fp, *a, **k):
        name = getattr(fp, "name", None)
        if isinstance(name, (str, os.PathLike)) and _key(name) == target:
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real_load(fp, *a, **k)

    if at == "open":
        with patch("builtins.open", _open):
            yield
    else:
        with patch("json.load", _load):
            yield


# ---------------------------------------------------------------------------
# Stand-in vLLM clients — the `async with` form the tail and rollup use
# ---------------------------------------------------------------------------

def _client_returning(content: str):
    """httpx.AsyncClient stand-in whose every call returns one canned
    completion. test_facts.py's MagicMock client can't be used here: the
    tail constructs its client with `async with`."""
    class _Canned:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json = MagicMock(return_value={
                "choices": [{"message": {"content": content}}]
            })
            return r
    return _Canned


def _client_raising(exc: Exception):
    class _Broken:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise exc
    return _Broken


def run_write_back(call):
    """Drive a read-modify-write path for its effect on disk.

    Whether the caller ends up raising or returning once F1 lands is F1's
    business — main.py's tail swallows either. Every assertion below is
    about the bytes on disk afterwards, which is §6.2's pass condition and
    is true under any shape the primitive takes.
    """
    try:
        call()
    except Exception as e:
        print(f"  note write path raised {type(e).__name__} "
              f"(fine — the assertion is about the file)")


def _turns(n: int) -> list[dict]:
    """n non-system messages — what maybe_rollup counts to decide on L1."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Loader discrimination — is "unreadable" told apart from "empty"?
# ---------------------------------------------------------------------------
#
# These assert the weakest true form of the finding, so they survive whatever
# shape F1 gives read_json_strict: a loader may raise, or return a status, or
# return anything at all — it just may not answer with the same value it
# gives for a file that legitimately isn't there. That answer is what the
# five write-back callers then persist.
#
# ALL EXPECTED TO FAIL until F1.

def _empty_facts(v) -> bool:
    return v == []


def _empty_state(s) -> bool:
    return (
        isinstance(s, dict)
        and not s.get("l1")
        and not s.get("l2")
        and s.get("l3") is None
        and s.get("last_summarized_turn") == 0
    )


def _no_persona(v) -> bool:
    return v is None


def assert_distinguishes_absent(call, is_absent, label):
    """The loader must not answer "nothing stored" when the truth is
    "could not read". Raising counts as distinguishing."""
    try:
        got = call()
    except Exception as e:
        print(f"  ok   {label} (raised {type(e).__name__})")
        return
    if is_absent(got):
        print(f"FAIL {label}: returned the absent-file default — "
              f"unreadable is indistinguishable from empty")
        sys.exit(1)
    print(f"  ok   {label} (returned a distinguishable value)")


def assert_reads_as_absent(call, is_absent, label):
    try:
        got = call()
    except Exception as e:
        print(f"FAIL {label}: raised {type(e).__name__}: {e}")
        sys.exit(1)
    if not is_absent(got):
        print(f"FAIL {label}: expected the absent-file default, got {got!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def test_load_facts_corrupt_is_not_empty():
    print("\n[test] F1a: a corrupt facts file is not an empty fact store")
    _wipe_storage()
    corrupt(memory.facts_path("corrupt-facts"))
    assert_distinguishes_absent(
        lambda: facts.load_facts("corrupt-facts"), _empty_facts,
        "load_facts(corrupt) != load_facts(absent)",
    )


def test_load_facts_unreadable_is_not_empty():
    print("\n[test] F1a: an EIO on open() is not an empty fact store")
    _wipe_storage()
    cid = "eio-facts"
    facts.save_facts(cid, [{"text": "Lyra is a half-elf ranger.",
                            "added_turn": 1, "last_used": 1748000000}])
    with unreadable(memory.facts_path(cid)):
        assert_distinguishes_absent(
            lambda: facts.load_facts(cid), _empty_facts,
            "load_facts(EIO on open) != load_facts(absent)",
        )


def test_load_facts_unreadable_at_json_load_is_not_empty():
    print("\n[test] F1a: an EIO from the read inside json.load() likewise")
    # The other half of §1.4's read-time trigger: open() succeeds against
    # cached metadata and the failure surfaces when the chunk is actually
    # fetched. read_json catches OSError on both, so both must be covered.
    _wipe_storage()
    cid = "eio-facts-load"
    facts.save_facts(cid, [{"text": "Setting is Aethermere.",
                            "added_turn": 1, "last_used": 1748000000}])
    with unreadable(memory.facts_path(cid), at="load"):
        assert_distinguishes_absent(
            lambda: facts.load_facts(cid), _empty_facts,
            "load_facts(EIO in json.load) != load_facts(absent)",
        )


def test_load_archive_corrupt_is_not_empty():
    print("\n[test] F1e: a corrupt archive sidecar is not an empty archive")
    # restore_from_archive reads both halves and writes both back, so an
    # empty read here loses the cold store as well as the active set.
    _wipe_storage()
    corrupt(memory.facts_archive_path("corrupt-archive"))
    assert_distinguishes_absent(
        lambda: facts.load_archive("corrupt-archive"), _empty_facts,
        "load_archive(corrupt) != load_archive(absent)",
    )


def test_load_archive_unreadable_is_not_empty():
    print("\n[test] F1e: an EIO on the archive sidecar is not an empty archive")
    _wipe_storage()
    cid = "eio-archive"
    facts.save_archive(cid, [{"text": "Retired fact.", "added_turn": 1,
                              "last_used": 0, "archived_at": 1748000000}])
    with unreadable(memory.facts_archive_path(cid)):
        assert_distinguishes_absent(
            lambda: facts.load_archive(cid), _empty_facts,
            "load_archive(EIO) != load_archive(absent)",
        )


def test_load_state_corrupt_is_not_empty():
    print("\n[test] F1b: a corrupt summary file is not an empty L1/L2/L3 stack")
    _wipe_storage()
    corrupt(memory.summary_path("corrupt-state"))
    assert_distinguishes_absent(
        lambda: summarizer.load_state("corrupt-state"), _empty_state,
        "load_state(corrupt) != _empty_state",
    )


def test_load_state_unreadable_is_not_empty():
    print("\n[test] F1b: an EIO on the summary file is not an empty stack")
    _wipe_storage()
    cid = "eio-state"
    summarizer.save_state(cid, {
        "l1": [{"text": "Scene one.", "first_turn": 1, "last_turn": 20}],
        "l2": [], "l3": None, "last_summarized_turn": 20,
    })
    with unreadable(memory.summary_path(cid)):
        assert_distinguishes_absent(
            lambda: summarizer.load_state(cid), _empty_state,
            "load_state(EIO) != _empty_state",
        )


def test_load_persona_corrupt_is_not_absent():
    print("\n[test] F1c: a corrupt persona file is not 'no persona set'")
    _wipe_storage()
    corrupt(memory.persona_path("corrupt-persona"))
    assert_distinguishes_absent(
        lambda: persona.load_persona("corrupt-persona"), _no_persona,
        "load_persona(corrupt) is not None",
    )


def test_load_persona_unreadable_is_not_absent():
    print("\n[test] F1c: an EIO on the persona file is not 'no persona set'")
    _wipe_storage()
    cid = "eio-persona"
    persona.save_persona(cid, "You are Sam Cole, " + ("a hardboiled detective. " * 12),
                         source="admin")
    with unreadable(memory.persona_path(cid)):
        assert_distinguishes_absent(
            lambda: persona.load_persona(cid), _no_persona,
            "load_persona(EIO) is not None",
        )


# ---------------------------------------------------------------------------
# The regression guard — absent really is empty
# ---------------------------------------------------------------------------
#
# THESE FOUR MUST PASS TODAY AND MUST STILL PASS AFTER F1. A conversation
# that has never stored anything has no file, and that is the by-design row
# in §1.4's trigger table. If F1 makes any of these raise or report
# UNREADABLE, every new conversation breaks on its first turn.

def test_absent_facts_file_is_still_empty():
    print("\n[test] an absent facts file is still an empty fact store")
    _wipe_storage()
    assert_reads_as_absent(
        lambda: facts.load_facts("never-seen"), _empty_facts,
        "load_facts(absent) -> []",
    )


def test_absent_archive_file_is_still_empty():
    print("\n[test] an absent archive sidecar is still an empty archive")
    _wipe_storage()
    assert_reads_as_absent(
        lambda: facts.load_archive("never-seen"), _empty_facts,
        "load_archive(absent) -> []",
    )


def test_absent_summary_file_is_still_empty():
    print("\n[test] an absent summary file is still an empty stack")
    _wipe_storage()
    assert_reads_as_absent(
        lambda: summarizer.load_state("never-seen"), _empty_state,
        "load_state(absent) -> empty skeleton",
    )


def test_absent_persona_file_is_still_absent():
    print("\n[test] an absent persona file is still 'no persona set'")
    _wipe_storage()
    assert_reads_as_absent(
        lambda: persona.load_persona("never-seen"), _no_persona,
        "load_persona(absent) -> None",
    )


# ---------------------------------------------------------------------------
# Write-back durability — the S1 itself
# ---------------------------------------------------------------------------
#
# The findings are not that a read returns the wrong value; they are that the
# caller then atomically overwrites the real file with what it read. These
# assert the §6.2 pass condition directly: after the turn, the file on disk
# is unchanged. That holds whatever shape F1 takes, so these do not need
# rewriting when the primitive lands.
#
# ALL EXPECTED TO FAIL until F1 + G2 + G3.

def test_async_tail_keeps_facts_after_unreadable_read():
    print("\n[test] F1a: the tail must not replace 12 facts with 1 after an EIO")
    # This is the incident shape: 105 established facts, one unreadable read,
    # one atomic write, 1 fact left. touched_facts=[] because the request
    # path's own load_facts hit the same EIO and injected nothing.
    _wipe_storage()
    cid = "tail-eio-facts"
    facts.save_facts(cid, [
        {"text": f"Established fact {i}.", "added_turn": i,
         "last_used": 1748000000 + i}
        for i in range(1, 13)
    ])
    with unreadable(memory.facts_path(cid)):
        with patch.object(main.httpx, "AsyncClient",
                          _client_returning("- A brand new fact from this turn.")):
            run_write_back(lambda: asyncio.run(main._async_tail(
                cid, [], "and then?", "Lyra drew her bow.", 13,
                [{"role": "user", "content": "and then?"}],
            )))
    assert_eq(len(facts.load_facts(cid)), 12,
              "all 12 facts survive an unreadable read (write aborted)")


def test_async_tail_leaves_corrupt_facts_file_untouched():
    print("\n[test] F1a: the tail must not overwrite a corrupt facts file")
    # §6.2's pass condition: the file is still the corrupt bytes afterwards.
    # A file that suddenly parses cleanly is the failure — it means the real
    # store was replaced by a summary of this one turn.
    _wipe_storage()
    cid = "tail-corrupt-facts"
    before = corrupt(memory.facts_path(cid)).read_text(encoding="utf-8")
    with patch.object(main.httpx, "AsyncClient",
                      _client_returning("- A brand new fact from this turn.")):
        run_write_back(lambda: asyncio.run(main._async_tail(
            cid, [], "and then?", "Lyra drew her bow.", 13,
            [{"role": "user", "content": "and then?"}],
        )))
    assert_eq(memory.facts_path(cid).read_text(encoding="utf-8"), before,
              "corrupt facts file left exactly as found")


def test_async_tail_writes_no_facts_file_when_nothing_extracted():
    print("\n[test] G2: a turn that extracts nothing must not write a facts file")
    # No early return on empty new_strs, so combined can be [] and
    # save_facts(conv_id, []) runs on every exchange. That is what makes
    # F1a re-fire every turn, and what grows facts/ without bound (D10).
    _wipe_storage()
    cid = "tail-nothing-extracted"
    with patch.object(main.httpx, "AsyncClient", _client_returning("NONE")):
        run_write_back(lambda: asyncio.run(main._async_tail(
            cid, [], "ok", "Understood.", 2,
            [{"role": "user", "content": "ok"}],
        )))
    assert_true(not memory.facts_path(cid).is_file(),
                "no facts file created for a zero-fact exchange")


def test_rollup_keeps_summary_stack_after_unreadable_read():
    print("\n[test] F1b/G3: the rollup must not flatten the L1 stack after an EIO")
    # G3: save_state sits outside the try that catches rollup failure, so a
    # misread plus an LLM failure writes _empty_state over the real file with
    # zero LLM involvement. The real state says no rollup is due at all
    # (24 turns, last_summarized_turn=40) — only the misread triggers it.
    _wipe_storage()
    cid = "rollup-eio"
    summarizer.save_state(cid, {
        "l1": [
            {"text": "Scene one: the road to Aethermere.", "first_turn": 1, "last_turn": 20},
            {"text": "Scene two: the ranger's oath.", "first_turn": 21, "last_turn": 40},
        ],
        "l2": [], "l3": None, "last_summarized_turn": 40,
    })
    with unreadable(memory.summary_path(cid)):
        with patch.object(summarizer.httpx, "AsyncClient",
                          _client_raising(httpx.ConnectError("refused"))):
            run_write_back(lambda: asyncio.run(summarizer.maybe_rollup(
                cid, _turns(24), "http://fake", "fake-model"
            )))
    after = summarizer.load_state(cid)
    assert_eq(len(after.get("l1") or []), 2, "both L1 chunks survive")
    assert_eq(after.get("last_summarized_turn"), 40, "last_summarized_turn intact")


def test_rollup_leaves_corrupt_summary_file_untouched():
    print("\n[test] F1b/G3: the rollup must not overwrite a corrupt summary file")
    _wipe_storage()
    cid = "rollup-corrupt"
    before = corrupt(memory.summary_path(cid)).read_text(encoding="utf-8")
    with patch.object(summarizer.httpx, "AsyncClient",
                      _client_raising(httpx.ConnectError("refused"))):
        run_write_back(lambda: asyncio.run(summarizer.maybe_rollup(
            cid, _turns(24), "http://fake", "fake-model"
        )))
    assert_eq(memory.summary_path(cid).read_text(encoding="utf-8"), before,
              "corrupt summary file left exactly as found")


def test_auto_capture_keeps_persona_after_unreadable_read():
    print("\n[test] F1c: auto-capture must not replace an admin persona after an EIO")
    # Worst of the three in one respect: this runs on the request path,
    # before vLLM is even called.
    #
    # It does NOT also cover change 6 (a deliberately-set "admin" or
    # "inherited" record is not replaced by auto-capture), though the comment
    # here used to claim it did. Under the EIO, load_persona raises before
    # auto_capture_persona reaches the source check at all — this test passes
    # identically with source="auto", so it says nothing about change 6.
    # That behaviour is covered in test_persona.py by
    # test_auto_capture_declines_to_replace_admin_persona and its
    # inherited / override_managed siblings.
    _wipe_storage()
    cid = "persona-eio"
    # .strip() to match what save_persona stores — it strips before writing.
    stored = ("You are Sam Cole, "
              + "a hardboiled detective in a rain-slick city. " * 8).strip()
    persona.save_persona(cid, stored, source="admin")
    incoming = "You are a cheerful assistant. " * 12
    with unreadable(memory.persona_path(cid)):
        run_write_back(lambda: persona.auto_capture_persona(
            cid, [{"role": "system", "content": incoming}]))
    rec = persona.load_persona(cid)
    assert_eq(rec["persona_text"], stored, "stored persona text unchanged")
    assert_eq(rec["source"], "admin", "admin record not downgraded to auto")


def test_auto_capture_leaves_corrupt_persona_file_untouched():
    print("\n[test] F1c: auto-capture must not overwrite a corrupt persona file")
    _wipe_storage()
    cid = "persona-corrupt"
    before = corrupt(memory.persona_path(cid)).read_text(encoding="utf-8")
    incoming = "You are a cheerful assistant. " * 12
    run_write_back(lambda: persona.auto_capture_persona(
        cid, [{"role": "system", "content": incoming}]))
    assert_eq(memory.persona_path(cid).read_text(encoding="utf-8"), before,
              "corrupt persona file left exactly as found")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Loader discrimination — expected to fail until F1.
        test_load_facts_corrupt_is_not_empty()
        test_load_facts_unreadable_is_not_empty()
        test_load_facts_unreadable_at_json_load_is_not_empty()
        test_load_archive_corrupt_is_not_empty()
        test_load_archive_unreadable_is_not_empty()
        test_load_state_corrupt_is_not_empty()
        test_load_state_unreadable_is_not_empty()
        test_load_persona_corrupt_is_not_absent()
        test_load_persona_unreadable_is_not_absent()

        # Regression guard — must pass today and after F1.
        test_absent_facts_file_is_still_empty()
        test_absent_archive_file_is_still_empty()
        test_absent_summary_file_is_still_empty()
        test_absent_persona_file_is_still_absent()

        # Write-back durability — expected to fail until F1 + G2 + G3.
        test_async_tail_keeps_facts_after_unreadable_read()
        test_async_tail_leaves_corrupt_facts_file_untouched()
        test_async_tail_writes_no_facts_file_when_nothing_extracted()
        test_rollup_keeps_summary_stack_after_unreadable_read()
        test_rollup_leaves_corrupt_summary_file_untouched()
        test_auto_capture_keeps_persona_after_unreadable_read()
        test_auto_capture_leaves_corrupt_persona_file_untouched()

        print("\nAll store-failure tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
