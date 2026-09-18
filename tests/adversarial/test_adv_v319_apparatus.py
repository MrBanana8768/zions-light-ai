"""Hostile pass #2, AREA 6: attacks on the TEST APPARATUS itself.

Everything else in v3.1.9 is validated BY the apparatus, so a lie here
invalidates every green result above it. These cases attack the oracles, not
the compactor:

  P1  test_soak_conversation's lag budget is arithmetically unreachable at the
      SOAK_TURNS the compose file actually runs.
  P2  _summary_coverage_gaps -- the soak's strongest claim -- is conditional on
      the watermark, which is the thing that breaks in the incident it cites.
  P3  _summary_coverage_gaps raises instead of failing on three state shapes a
      crash-torn or type-wrong state file produces.
  P4  the fixture's looping=False LCG has a strictly alternating low bit.
  P5  adversarial_replies.__exit__ restores two of the three mode keys its
      __enter__ sets; reply_looping is not restored.

NO DOCKER, NO COMPACTOR. Every case here is a pure function lifted out of the
file that defines it, so this runs anywhere python does:

    python tests/adversarial/test_adv_v319_apparatus.py

Each test asserts the property that SHOULD hold, so it is red while the defect
stands and green when it is closed. That is deliberate and it differs from
test_adv_v319_reuse/webuidb/loop.py, which print BROKE and then call
sys.exit(0) unconditionally -- see the report for why that matters.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _lift(path: pathlib.Path, names: set[str]) -> dict:
    """Exec only the named top-level defs/assigns out of `path`.

    The soak and the fixture both run their whole program at import time, so
    importing them is not an option. Lifting the node keeps the test honest:
    it runs the SHIPPED source of the function, not a copy of it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "_lifted", "re": re}
    for node in tree.body:
        keep = (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ) or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in names for t in node.targets)
        )
        if not keep:
            continue
        mod = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(mod)
        try:
            exec(compile(mod, str(path), "exec"), ns)
        except Exception:
            pass
    return ns


SOAK = ROOT / "compactor" / "test_soak_conversation.py"
FIXTURE = ROOT / "testfixtures" / "tokenizer-contract" / "fixture_server.py"
REGTEXT = ROOT / "tests" / "integration" / "test_regression_text.py"
COMPOSE_CONTRACT = ROOT / "docker-compose.tokenizer-contract.yml"

_gaps = _lift(SOAK, {"_summary_coverage_gaps"})["_summary_coverage_gaps"]
_fx = _lift(FIXTURE, {"_body_words", "_LOREM"})


def _chunk(first, last, text="a real summary of that stretch"):
    return {"text": text, "first_turn": first, "last_turn": last}


# ---------------------------------------------------------------------------
# P1  the lag budget cannot fire at the SOAK_TURNS the compose file runs
# ---------------------------------------------------------------------------

def test_p1_soak_lag_budget_is_reachable_at_the_configured_turn_count():
    """The soak's guard against the 2026-08-28 shape must be able to fire.

    _LAG_BUDGET = L1_CHUNK_SIZE*2 + KEEP_RECENT_TURNS*2 = 20*2 + 4*2 = 48.
    `rows` records len(history) BEFORE the assistant reply is appended, so at
    turn n it is 2n-1, and the watermark floors at 0. The largest lag the run
    can ever produce is therefore 2*SOAK_TURNS - 1.

    docker-compose.tokenizer-contract.yml sets SOAK_TURNS=22 -> max lag 43,
    against a budget of 48. A watermark frozen at 0 for the entire run does
    not trip it. The module default of 40 does (max lag 79).
    """
    turns = None
    for line in COMPOSE_CONTRACT.read_text(encoding="utf-8").splitlines():
        if "SOAK_TURNS:" in line:
            turns = int(line.split(":", 1)[1].strip().strip('"').strip("'"))
    assert turns is not None, "SOAK_TURNS is not set in the contract compose file"

    l1_chunk_size, keep_recent = 20, 4          # summarizer/main defaults
    lag_budget = l1_chunk_size * 2 + keep_recent * 2
    max_possible_lag = 2 * turns - 1
    assert max_possible_lag > lag_budget, (
        f"SOAK_TURNS={turns} caps the observable lag at {max_possible_lag}, "
        f"below _LAG_BUDGET={lag_budget}. The soak's `lagging` check is "
        f"arithmetically unreachable: the watermark can freeze at 0 for the "
        f"whole run and the check still passes. It needs "
        f"SOAK_TURNS > {(lag_budget + 1) // 2} to be able to fire at all."
    )


# ---------------------------------------------------------------------------
# P2  the coverage oracle is conditional on the watermark
# ---------------------------------------------------------------------------

def test_p2_coverage_oracle_sees_turns_stranded_above_a_frozen_watermark():
    """The soak's strongest claim must see the shape its own comment names.

    Its comment: "history was consumed without being represented anywhere",
    and it cites 2026-08-28. In that incident the watermark FROZE while the
    conversation grew and the guard shed everything above it.

    `_summary_coverage_gaps` returns early on `watermark <= 0` and its only
    completeness test is `covered < watermark`, so turns ABOVE the watermark
    are outside its domain by construction. One rollup followed by total
    hierarchy death is reported as full coverage.
    """
    stranded = _gaps({
        "last_summarized_turn": 20,        # advanced once, then froze
        "l1": [_chunk(1, 20)],
        "l2": [],
        "turns_seen": 44,                  # the conversation kept growing
        "window_turns": 44,
    })
    assert stranded, (
        "a state whose watermark froze at 20 while the conversation reached "
        "turn 44 is reported as fully covered. 24 turns were shed by the hard "
        "budget guard and are represented nowhere, which is 2026-08-28, and "
        "the assertion that exists for it cannot see past the watermark."
    )


def test_p2b_coverage_oracle_rejects_an_impossible_chunk_span():
    """One chunk with an over-wide last_turn makes coverage unfalsifiable.

    `covered = max(covered, lt)` trusts the chunk's own bounds, so a rollup
    defect that writes a last_turn past the end of the conversation satisfies
    every later comparison for free.
    """
    assert _gaps({
        "last_summarized_turn": 40,
        "l1": [_chunk(1, 999999)],
        "turns_seen": 44,
    }), (
        "an l1 chunk claiming turns 1-999999 on a 44-turn conversation is "
        "accepted as coverage. Nothing bounds a chunk's span against "
        "turns_seen, so the oracle cannot fail once one bad span exists."
    )


# ---------------------------------------------------------------------------
# P3  the oracle raises rather than failing on plausible bad state
# ---------------------------------------------------------------------------

def test_p3_coverage_oracle_does_not_raise_on_a_type_wrong_state_file():
    """A torn or type-wrong state file must produce a FAIL, not a traceback.

    The state is JSON on a volume a RunPod redeploy can kill mid-write. All
    three shapes below come back from json.load without complaint and make
    `_summary_coverage_gaps` raise, so the soak dies with an uncaught
    exception instead of printing which invariant broke.
    """
    bad = {
        "watermark as a string": {"last_summarized_turn": "40",
                                  "l1": [_chunk(1, 20)]},
        "chunk text as a dict": {"last_summarized_turn": 40,
                                 "l1": [{"text": {"summary": "x"},
                                         "first_turn": 1, "last_turn": 40}]},
        "l1 as a dict of chunks": {"last_summarized_turn": 40,
                                   "l1": {"c1": _chunk(1, 40)}},
    }
    raised = {}
    for label, st in bad.items():
        try:
            _gaps(st)
        except Exception as e:
            raised[label] = f"{type(e).__name__}: {e}"
    assert not raised, (
        "the coverage oracle raises instead of reporting a problem:\n  "
        + "\n  ".join(f"{k} -> {v}" for k, v in raised.items())
    )


# ---------------------------------------------------------------------------
# P4  the looping=False LCG's low bit strictly alternates
# ---------------------------------------------------------------------------

def test_p4_non_looping_padding_has_no_fixed_period_in_its_low_bit():
    """eabacb2's LCG must not have an obvious short-period structure.

    seed = (seed*1103515245 + 12345) & 0x7FFFFFFF has a congruent to 1 mod 4
    and c odd, so modulo 2 it is seed -> seed+1: the low bit alternates on
    every single step, forever. len(_LOREM) is 34 = 2*17, so
    `seed % len(_LOREM)` inherits it and the padding's word index alternates
    even/odd with period 2 by construction.

    This does NOT reintroduce the tail loop (measured: _tail_loop_span is 0 at
    1,400 / 4,000 / 12,000 chars for seq 0, 1, 9 and 47), but it means the
    fixture's "non-repeating" padding is half-deterministic, and any future
    degeneracy rule keyed on positional structure would be exercised against a
    sequence carrying a period-2 component.
    """
    lorem = _fx["_LOREM"]
    seed = (9 * 7 + 1) & 0x7FFFFFFF
    parities = []
    for _ in range(400):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        parities.append((seed % len(lorem)) % 2)
    alternating = all(
        parities[i] != parities[i + 1] for i in range(len(parities) - 1)
    )
    assert not alternating, (
        "the word index chosen by the looping=False LCG alternates even/odd on "
        "every step for all 400 steps measured. A truncated LCG's low bits are "
        "known-bad; random.Random(n).choice is deterministic in n, which is the "
        "only property the docstring asks for, and has no such structure."
    )


# ---------------------------------------------------------------------------
# P5  adversarial_replies restores two of the three keys it sets
# ---------------------------------------------------------------------------

def test_p5_adversarial_replies_restores_every_mode_key_it_sets():
    """__enter__ and __exit__ must name the same fixture mode keys.

    eabacb2 added reply_looping to __enter__'s fixture_mode_set, to the
    fixture's set_mode whitelist and to _MODE -- and not to __exit__, whose
    own comment says "Restore rather than zero: another agent may have had a
    mode set". The fixture is one container for the whole stack and _MODE is
    process-global, so the False set by
    test_decorated_prose_reply_is_still_memorized outlives the test.
    """
    tree = ast.parse(REGTEXT.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "adversarial_replies")

    def keys_set(method: str) -> set[str]:
        fn = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == method)
        out: set[str] = set()
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "fixture_mode_set"):
                out |= {k.arg for k in node.keywords if k.arg}
        return out

    entered, exited = keys_set("__enter__"), keys_set("__exit__")
    assert entered <= exited, (
        f"__enter__ sets {sorted(entered)} but __exit__ restores only "
        f"{sorted(exited)}; {sorted(entered - exited)} leaks to every later "
        f"consumer of this fixture container."
    )


if __name__ == "__main__":
    failures = []
    for _name, _fn in sorted(globals().items()):
        if not (_name.startswith("test_") and callable(_fn)):
            continue
        try:
            _fn()
            print(f"  PASS  {_name}")
        except AssertionError as _e:
            print(f"  FAIL  {_name}")
            print(f"        {_e}")
            failures.append(_name)
    print()
    print(f"{len(failures)} apparatus defect(s) reproduced.")
    sys.exit(1 if failures else 0)
