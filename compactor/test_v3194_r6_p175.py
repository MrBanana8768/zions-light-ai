"""
v3.1.9.4 R6 (hostile pass #17): P17-5 (documentation).

`commands._handle_forget`'s "summary will rebuild" note only showed when
`totals["forgotten_summary"] or totals["forgotten_chapters"]` — but the
rebuild it describes (the hierarchy re-folding the pre-/forget transcript
OpenWebUI resends on the next turn) happens whenever the chat continues,
regardless of whether a summary already existed to clear. A /forget on a
short chat that had facts but no summary YET got no note, even though the
exact same rebuild would happen. Fix: gate on `parts` (something was
actually forgotten) instead — not shown on a true no-op ("Nothing to
forget").

Direct calls to `commands._handle_forget` with a stubbed `clear_all_memory`
in `ctx`, mirroring test_commands.py's own shape. Synthetic content only.

Run: python test_v3194_r6_p175.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-v3194-r6-p175-")
import memory  # noqa: E402
memory.ensure_storage_layout()
import commands  # noqa: E402
import facts  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


_NOTE_SNIPPET = "the summary will start rebuilding itself"


def _clear_all_stub(result):
    async def _stub(conv_id):
        return dict(result)
    return _stub


def _base_result(**overrides):
    r = {
        "conv_id": "x",
        "forgotten_facts": 0,
        "forgotten_episodic": 0,
        "forgotten_summary": False,
        "forgotten_chapters": False,
        "forgotten_persona": False,
        "unreadable": [],
    }
    r.update(overrides)
    return r


def test_note_shown_when_facts_forgotten_but_no_summary_existed():
    print("\n[test] P17-5: /forget on a chat with FACTS but NO summary yet "
          "now gets the rebuild note too — the exact case the finding named")
    cid = "p175-facts-only"
    facts.save_facts(cid, [{"text": "a fact", "added_turn": 1, "last_used": 1}])
    ctx = {"clear_all_memory": _clear_all_stub(_base_result(forgotten_facts=1))}
    reply = asyncio.run(commands._handle_forget("", cid, ctx))
    check("Forgot: 1 fact(s)." in reply, f"reports the forgotten fact: {reply!r}")
    check(_NOTE_SNIPPET in reply,
          f"the rebuild note IS present (was missing pre-fix): {reply!r}")


def test_control_note_shown_when_summary_actually_cleared():
    print("\n[test] P17-5 CONTROL: /forget that DID clear a summary still "
          "gets the note (unchanged from before this fix)")
    cid = "p175-summary-cleared"
    ctx = {"clear_all_memory": _clear_all_stub(_base_result(forgotten_summary=True))}
    reply = asyncio.run(commands._handle_forget("", cid, ctx))
    check("summary state" in reply, f"reports the cleared summary: {reply!r}")
    check(_NOTE_SNIPPET in reply, f"the rebuild note is present: {reply!r}")


def test_control_no_note_on_a_true_no_op_forget():
    print("\n[test] P17-5 CONTROL: a /forget with NOTHING to forget at all "
          "gets no note (there is nothing that could rebuild)")
    cid = "p175-nothing-to-forget"
    ctx = {"clear_all_memory": _clear_all_stub(_base_result())}
    reply = asyncio.run(commands._handle_forget("", cid, ctx))
    check(reply == "Nothing to forget — this conversation had no stored memory.",
          f"exact no-op reply: {reply!r}")
    check(_NOTE_SNIPPET not in reply, f"no rebuild note on a no-op: {reply!r}")


def test_note_shown_for_persona_or_episodic_only_too():
    print("\n[test] P17-5: the note is not narrowly tied to facts either — "
          "any real forgotten content (persona-only, here) gets it")
    cid = "p175-persona-only"
    ctx = {"clear_all_memory": _clear_all_stub(_base_result(forgotten_persona=True))}
    reply = asyncio.run(commands._handle_forget("", cid, ctx))
    check("persona" in reply, f"reports the cleared persona: {reply!r}")
    check(_NOTE_SNIPPET in reply, f"the rebuild note is present: {reply!r}")


# ---------------------------------------------------------------------------
# backfill._facts_tombstoned docstring correction — behavioral checks are in
# test_v3194_r6_p174.py (the archive-sidecar fix); this just confirms the
# docstring itself names all four non-wipe paths, so a future reader is not
# misled the way the reviewer's own review of the premise caught.
# ---------------------------------------------------------------------------

def test_facts_tombstoned_docstring_names_the_non_wipe_paths():
    print("\n[test] P17-5: backfill._facts_tombstoned's docstring no longer "
          "claims a wipe's tombstone is the ONLY thing that ever leaves an "
          "empty facts file — it names the other paths and says which are "
          "still treated as tombstones and which are not")
    import backfill
    doc = backfill._facts_tombstoned.__doc__ or ""
    for term in ("/tidy", "archive-stale", "selective", "empty-bundle"):
        check(term in doc, f"docstring mentions {term!r}")
    check("nothing in this codebase ever writes an empty facts file except" not in doc.lower(),
          "the old, overstated claim is gone")


def _all_tests():
    return [
        test_note_shown_when_facts_forgotten_but_no_summary_existed,
        test_control_note_shown_when_summary_actually_cleared,
        test_control_no_note_on_a_true_no_op_forget,
        test_note_shown_for_persona_or_episodic_only_too,
        test_facts_tombstoned_docstring_names_the_non_wipe_paths,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R6 P17-5 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
