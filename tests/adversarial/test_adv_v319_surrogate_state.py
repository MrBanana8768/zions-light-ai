"""AREA 5 hostile pass #2: M3 (the lone surrogate at the UNGUARDED body
siblings) and admin_compact's other unguarded conversion.

NEW FILE. No shipped module is edited. The state-write mechanism is proved
against the REAL compactor/memory.py (stdlib-only, so it imports anywhere)
into a scratch storage root. The reachability from /admin/conversations/import
is established by reading the shipped call sequence and is quoted in the
report; the write that fails, and what it leaves on disk, is executed here.

Run:  python tests/adversarial/test_adv_v319_surrogate_state.py
"""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "compactor" / "main.py"

FAILED: list[str] = []

try:  # a Windows console defaults to cp1252 and cannot print U+1F600
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass


def check(cond, label):
    print(f"   {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED.append(label)


def eq(got, want, label):
    check(got == want, f"{label}  (got {got!r}, want {want!r})")


SCRATCH = Path(tempfile.mkdtemp(prefix="adv-a5-"))
os.environ["COMPACTOR_STORAGE_ROOT"] = str(SCRATCH)
sys.path.insert(0, str(ROOT / "compactor"))
import memory  # noqa: E402  the real module


LONE = "\ud83d"          # a valid JSON \ud83d, a valid Python str, unencodable
PAIRED = "😀"  # the legal pair json.loads folds into U+1F600


# ===========================================================================
# [A5-11]  atomic_write_json cannot write a lone surrogate, and the exception
#          is a ValueError sibling nobody in the admin chain names
# ===========================================================================

def test_atomic_write_json_raises_unicodeencodeerror_on_a_lone_surrogate():
    print()
    print("[A5-11] memory.atomic_write_json vs a lone surrogate")

    # Control: the legal pair writes fine, so the case below is the surrogate
    # and not 'non-ASCII'.
    ok_path = SCRATCH / "ok.json"
    memory.atomic_write_json(ok_path, {"facts": [{"text": PAIRED}]})
    eq(json.loads(ok_path.read_text(encoding="utf-8"))["facts"][0]["text"],
       "\U0001f600", "CONTROL: a PAIRED surrogate (an ordinary emoji) writes "
                      "and round-trips")

    bad_path = SCRATCH / "bad.json"
    raised = None
    try:
        memory.atomic_write_json(bad_path, {"facts": [{"text": LONE}]})
    except Exception as e:
        raised = e
    check(raised is not None, "the write RAISES")
    eq(type(raised).__name__, "UnicodeEncodeError",
       "F21: it raises UnicodeEncodeError — `json.dump(..., "
       "ensure_ascii=False)` into a utf-8 file, memory.py:330")
    check(isinstance(raised, ValueError),
          "F21b: UnicodeEncodeError IS a ValueError, so it is the exact "
          "sibling shape 843bf9d's own M4 was about — and M4 fixed the READ "
          "side (json.load) while the WRITE side raises the same family")
    check(not isinstance(raised, (json.JSONDecodeError, OSError)),
          "F21c: ...and it is neither JSONDecodeError nor OSError, which is "
          "what every handler in the admin chain names")
    check(not bad_path.exists(),
          "the destination is not created — atomic_write_json's cleanup "
          "holds, so this is a FAILED WRITE, not a torn file")
    leftovers = [p.name for p in SCRATCH.glob("bad.json.*")]
    eq(leftovers, [], "and the temp file is unlinked")


# ===========================================================================
# [A5-12]  The guard is at chat_completions and at NONE of its siblings
# ===========================================================================

def test_the_surrogate_guard_has_seven_siblings_and_guards_one():
    print()
    print("[A5-12] the body-parsing siblings of chat_completions")

    src = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(src, filename="main.py")

    # Every async endpoint that parses a request body.
    parsers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            seg = ast.get_source_segment(src, node) or ""
            if "await request.json()" in seg or "await request.body()" in seg:
                guarded = 'b"\\\\u" in _raw' in seg or "unpaired_surrogate" in seg
                parsers.append((node.name, node.lineno, guarded))
    print()
    print("       handler                              line   surrogate guard?")
    print("       " + "-" * 62)
    for name, ln, g in sorted(parsers, key=lambda x: x[1]):
        print(f"       {name:<36} {ln:>5}   {'YES' if g else 'no'}")

    guarded = [p for p in parsers if p[2]]
    unguarded = [p for p in parsers if not p[2]]
    eq([p[0] for p in guarded], ["chat_completions"],
       "exactly one handler carries the guard")
    check(len(unguarded) >= 5,
          f"F22: {len(unguarded)} sibling handlers parse a body and none "
          f"carries it: {[p[0] for p in unguarded]}")
    check(any(p[0] == "admin_import_conversation" for p in unguarded),
          "F22b: admin_import_conversation is one of them — the endpoint "
          "whose whole job is to write three memory layers wholesale")


# ===========================================================================
# [A5-13]  admin_import's write ORDER turns that 500 into data loss
# ===========================================================================

def test_import_clears_before_it_writes_so_a_failed_write_loses_the_conv():
    print()
    print("[A5-13] portability.import_conversation: clear-then-write order")

    psrc = (ROOT / "compactor" / "portability.py").read_text(encoding="utf-8")
    i_clear = psrc.index("facts.save_facts(target, [])")
    i_forget = psrc.index("retrieval.forget_conversation(target)")
    i_write = psrc.index('facts.save_facts(target, list(bundle.get("facts", [])))')
    i_state = psrc.index('summarizer.save_state(target, dict(bundle.get("summary_state", {})))')
    check(i_clear < i_forget < i_write < i_state,
          "the shipped order is: wipe facts -> wipe episodic -> write "
          "bundle facts -> write bundle summary state")

    # The endpoint's handler list.
    msrc = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(msrc)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "admin_import_conversation")
    seg = ast.get_source_segment(msrc, node) or ""
    check("except (portability.ImportError_, UnsafeConvId)" in seg,
          "the endpoint catches exactly (ImportError_, UnsafeConvId)")
    check("UnicodeEncodeError" not in seg and "ValueError" not in seg,
          "F23: it names neither UnicodeEncodeError nor ValueError, so the "
          "failed write at step 3 escapes as an unhandled 500")

    # And nothing validates the text on the way in.
    check("_validate_bundle" in psrc, "_validate_bundle exists")
    v = ast.get_source_segment(psrc, next(
        n for n in ast.parse(psrc).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_bundle")) or ""
    check("encode" not in v and "surrogate" not in v,
          "F23b: _validate_bundle is a SHAPE check only — no encodability "
          "check anywhere, so the surrogate reaches save_facts intact")

    # Execute the consequence at the layer that actually fails, with the real
    # primitives, in the shipped order.
    conv = "adv-a5-import"
    memory.atomic_write_json(memory.facts_path(conv), {
        "conv_id": conv, "updated_at": "t0",
        "facts": [{"text": f"her real fact {i}"} for i in range(105)]})
    before = json.loads(memory.facts_path(conv).read_text(encoding="utf-8"))
    eq(len(before["facts"]), 105, "the conversation starts with 105 facts")

    # step 1 of the shipped sequence: the overwrite clear.
    memory.atomic_write_json(memory.facts_path(conv), {
        "conv_id": conv, "updated_at": "t1", "facts": []})
    # (step 2, retrieval.forget_conversation, wipes the episodic store; not
    #  executed here because it needs chromadb. It is unconditional in the
    #  same `if` block, so it lands before step 3 either way.)
    # step 3: write the bundle's facts, one of which carries the surrogate.
    boom = None
    try:
        memory.atomic_write_json(memory.facts_path(conv), {
            "conv_id": conv, "updated_at": "t2",
            "facts": [{"text": "an imported fact"}, {"text": LONE}]})
    except Exception as e:
        boom = e
    check(type(boom).__name__ == "UnicodeEncodeError",
          "step 3 raises UnicodeEncodeError")
    after = json.loads(memory.facts_path(conv).read_text(encoding="utf-8"))
    eq(after["facts"], [],
       "F24 (BLOCKER): the facts file is left EMPTY. Step 1's wipe landed, "
       "step 3's write did not, and the operator gets an HTTP 500 from the "
       "endpoint whose name is 'import'. 105 facts and the whole episodic "
       "index are gone, replaced by nothing, by the restore tool")
    check(after["updated_at"] == "t1",
          "and the file's own updated_at proves which write survived")


# ===========================================================================
# [A5-14]  admin_compact's OTHER unguarded conversion, one line above the fix
# ===========================================================================

def test_max_calls_is_unguarded_and_zero_means_two_hundred():
    print()
    print("[A5-14] admin_compact: `int(body.get('max_calls') or 200)`")

    src = MAIN.read_text(encoding="utf-8")
    needle = 'max_calls = int(body.get("max_calls") or 200)'
    eq(src.count(needle), 1, "the line is present exactly once")
    # and it sits immediately above the line 843bf9d changed.
    i_mc = src.index(needle)
    i_dr = src.index("dry_run = _dry_run_from(request, body, default=False)")
    check(0 < i_dr - i_mc < 300,
          "F25: it is the line ABOVE the dry_run fix, in the same handler, "
          "in the same commit's diff hunk")

    # Its sibling in chat_completions IS guarded. Same conversion, same
    # source (a client-supplied body), one wrapped and one not.
    check('try:\n        req_max_tokens = int(body.get("max_tokens") or 0)\n'
          '    except (TypeError, ValueError):' in src,
          "F25b: chat_completions wraps the identical shape in "
          "`except (TypeError, ValueError)` — the rule exists in this file "
          "and was applied at one site")

    def shipped(body):
        """The shipped expression, evaluated. Not a paraphrase: it is the
        text of the line, exec'd."""
        ns = {"body": body}
        exec(compile(needle.strip(), "<max_calls>", "exec"), ns)
        return ns["max_calls"]

    print()
    print("       body                          max_calls")
    print("       " + "-" * 48)
    for b, want in [({}, 200), ({"max_calls": 5}, 5), ({"max_calls": "7"}, 7),
                    ({"max_calls": 0}, 200), ({"max_calls": False}, 200),
                    ({"max_calls": None}, 200), ({"max_calls": -5}, -5),
                    ({"max_calls": 1.9}, 1), ({"max_calls": 10**9}, 10**9)]:
        got = shipped(b)
        print(f"       {str(b):<30}{got}")
        eq(got, want, f"{b} -> {want}")

    # {"max_calls": {}} and {"max_calls": ""} are FALSY, so `or 200` catches
    # them and they land on 200 — same silent-200 shape as the 0 case below.
    for b in [{"max_calls": {}}, {"max_calls": ""}]:
        eq(shipped(b), 200, f"{b} -> 200 (falsy, so `or 200` swallows it)")

    for b in [{"max_calls": "abc"}, {"max_calls": {"n": 1}},
              {"max_calls": [1]}, {"max_calls": "1e6"}]:
        raised = None
        try:
            shipped(b)
        except Exception as e:
            raised = type(e).__name__
        print(f"       {str(b):<30}RAISES {raised}")
        check(raised in ("ValueError", "TypeError"),
              f"F25c: {b} is an unhandled {raised} -> HTTP 500 from an admin "
              f"endpoint")

    check(shipped({"max_calls": 0}) == 200,
          "F26: `{'max_calls': 0}` means 200. An operator asking for ZERO "
          "summarization calls — the other way to say 'just show me the "
          "plan' — gets the full 200-call LIVE run, because `or 200` cannot "
          "tell 0 from absent. That is M6's own failure mode, surviving in "
          "the handler M6 was fixed in")
    check(shipped({"max_calls": 10**9}) == 10**9,
          "F26b: and there is no upper clamp — max_calls is bounded only by "
          "the `while calls < max_calls` loop, so a typed-in 1000000000 is "
          "accepted as written")


ALL = [
    test_atomic_write_json_raises_unicodeencodeerror_on_a_lone_surrogate,
    test_the_surrogate_guard_has_seven_siblings_and_guards_one,
    test_import_clears_before_it_writes_so_a_failed_write_loses_the_conv,
    test_max_calls_is_unguarded_and_zero_means_two_hundred,
]

if __name__ == "__main__":
    print("=" * 74)
    print("AREA 5 hostile pass #2 — the lone surrogate and the state it loses")
    print("=" * 74)
    print(f"scratch storage root: {SCRATCH}")
    for t in ALL:
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} assertion(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("all assertions held (each encodes a finding; see the report)")
