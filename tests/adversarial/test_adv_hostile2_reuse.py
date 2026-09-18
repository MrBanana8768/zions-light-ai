"""ADVERSARIAL pass #2 on the compaction reuse path (ced4520 + f78a389).

Run inside the unit-suite image, which can import `main`:

  docker run --rm --network none -v <worktree>:/src:ro \
    --entrypoint /bin/bash zla-feat-compact-unit-tests:latest \
    -c 'cp -r /src /work && cd /work/compactor && \
        /opt/compactor-venv/bin/python \
        /work/tests/adversarial/test_adv_hostile2_reuse.py'

f78a389 added three gates in front of the substitution:

    if _covered > 0 and _n >= summarizer._recorded_position(_st) and _aligned:

and rendered the stand-in with all_or_nothing=True.

THE FIRST PASS attacked the gap between a LENGTH relation and a CONTENT one.
This pass attacks the gap between the CONTENT relation that is actually
checked (does the stored anchor appear ANYWHERE in the last 64 turns) and
the one the substitution needs (are turns 1..`_covered` of THIS array the
turns the hierarchy summarized, all of them, and unchanged).

Every break below passes all three gates.
"""

import asyncio
import json
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="h2-reuse-")
os.environ["COMPACTOR_TARGET_TOKENS"] = "500"

sys.path.insert(0, "/work/compactor")

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

BROKEN: list[str] = []
HELD: list[str] = []
FIXTURE_FAILED: list[str] = []


def broke(cond, label):
    """cond True == the break reproduced."""
    if cond:
        print(f"  *** BROKE: {label}")
        BROKEN.append(label)
    else:
        print(f"  (held)    {label}")
        HELD.append(label)


def fixture(cond, label):
    """A property the attack DEPENDS on. False == the attack never ran."""
    if cond:
        print(f"  fixture ok   {label}")
    else:
        print(f"  FIXTURE FAILED   {label}")
        FIXTURE_FAILED.append(label)


CALLS: list[list[str]] = []


async def _spy_summarize(client, to_summarize):
    CALLS.append([str(m.get("content", ""))[:60] for m in to_summarize])
    return "FRESHLY-SUMMARIZED", []


main.summarize = _spy_summarize


def history(n_exchanges: int, tag: str = "") -> list[dict]:
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n_exchanges):
        out.append({"role": "user",
                    "content": f"question {tag}{i} " + ("word " * 60)})
        out.append({"role": "assistant",
                    "content": f"answer {tag}{i} " + ("word " * 60)})
    return out


def flat(msgs) -> str:
    parts = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
    return " ".join(parts)


def everything_the_model_will_see(out, calls) -> str:
    """The returned array PLUS whatever was handed to summarize().

    A turn that appears in neither is gone from the request with no stand-in.
    """
    sent = " ".join(" ".join(c) for c in calls)
    return flat(out) + " " + sent


def seed(conv, msgs, chunks, *, anchor_through, turns_seen=None):
    """A hierarchy the way maybe_rollup leaves one: chunks, a watermark, and
    tail_fp = the fingerprints of the last _ANCHOR_TURNS turns of the array
    as it stood when the rollup ran (summarizer.py:1111)."""
    st = summarizer.load_state(conv)
    st["l1"] = chunks
    st["last_summarized_turn"] = chunks[-1]["last_turn"]
    ns = [m for m in msgs if m.get("role") != "system"]
    st["tail_fp"] = summarizer._turn_fingerprints(
        ns[:anchor_through]
    )[-summarizer._ANCHOR_TURNS:]
    st["turns_seen"] = anchor_through if turns_seen is None else turns_seen
    summarizer.save_state(conv, st)
    return st


def gates(conv, msgs):
    """Evaluate the three gates exactly as compact_if_needed does."""
    st = summarizer.load_state(conv)
    ns = [m for m in msgs if m.get("role") != "system"]
    covered = summarizer._highest_chunk_turn(st)
    n = len(ns)
    rec = summarizer._recorded_position(st)
    anchor = st.get("tail_fp") or []
    cands = summarizer._align_candidates(
        anchor,
        summarizer._turn_fingerprints(ns[-summarizer._FINGERPRINT_TAIL_TURNS:]),
    )
    return {
        "covered": covered, "n": n, "recorded": rec,
        "anchor_len": len(anchor), "candidates": cands,
        "aligned": bool(anchor) and bool(cands),
        "passes": covered > 0 and n >= rec and bool(anchor) and bool(cands),
    }


# ===========================================================================
print()
print("=" * 74)
print("H1  A MID-CONVERSATION EDIT: THE ANCHOR IS THE TAIL, THE REMOVAL IS")
print("    THE HEAD, AND NOTHING COMPARES THE HEAD TO ANYTHING")
print("=" * 74)
print("""
_aligned fingerprints `_non_system[-_FINGERPRINT_TAIL_TURNS:]` — the TAIL.
The substitution deletes `to_summarize[:_covered]` — the HEAD. No check
relates the two. OpenWebUI lets the user edit an earlier message and save
without regenerating, which rewrites ONE message in the middle and leaves
the tail byte-identical. All three gates pass and the hierarchy's summary of
the PRE-EDIT text replaces the corrected turn.
""")

CONV_H1 = "h2_edit"
MSGS_H1 = history(24)
seed(CONV_H1, MSGS_H1,
     [{"text": "PRE-EDIT-SUMMARY-OF-TURNS-1-TO-40",
       "first_turn": 1, "last_turn": 40}],
     anchor_through=40)

# The user rewinds to turn 5 (non-system index 4) and corrects a fact.
EDITED = [dict(m) for m in MSGS_H1]
EDITED[1 + 4] = {
    "role": "user",
    "content": "CORRECTED-FACT my sister is Sarah not Sara " + ("word " * 60),
}
g = gates(CONV_H1, EDITED)
print(f"  gates: {g}")
fixture(g["passes"], "all three gates pass on the EDITED array")

CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(EDITED), CONV_H1))
seen = everything_the_model_will_see(out, CALLS)
fixture("PRE-EDIT-SUMMARY" in flat(out), "reuse actually fired (control)")
broke("CORRECTED-FACT" not in seen,
      "H1: the corrected turn is in NEITHER the returned array NOR the "
      "summarize() input — the model is told the pre-edit version instead, "
      "and the hierarchy will never re-read that span")


# ===========================================================================
print()
print("=" * 74)
print("H2  THE FORK GUARD IS DEFEATED BY ONE REPEATED SHORT TURN")
print("=" * 74)
print("""
_align_candidates tries EVERY PREFIX of the anchor, down to length 1, at
EVERY position in the 64-turn window. So `_aligned` is True as soon as ONE
fingerprint — anchor[0], a USER turn — reappears anywhere recent. In a
companion chat the user types "ok" more than once. That is the whole
content evidence, and it re-opens A3 (20 of 30 branch-B exchanges replaced
by branch A's summary) exactly as f78a389 closed it.

Note `_aligned = bool(_align_candidates(...))`: the candidate list is only
tested for EMPTINESS. A candidate list of [0] is truthy.
""")

CONV_H2 = "h2_fork"
BR_A = history(24, tag="A")
BR_A[1 + 36] = {"role": "user", "content": "ok"}      # turn 37, in the anchor
BR_B = history(24, tag="B")
BR_B[1 + 44] = {"role": "user", "content": "ok"}      # turn 45, a later "ok"

seed(CONV_H2, BR_A,
     [{"text": "BRANCH-A-CHUNK-COVERING-1-TO-40",
       "first_turn": 1, "last_turn": 40}],
     anchor_through=40)
_st_h2 = summarizer.load_state(CONV_H2)
print(f"  anchor = {_st_h2['tail_fp']}")
print(f"  fp of a bare user 'ok' = "
      f"{summarizer._turn_fingerprints([{'role': 'user', 'content': 'ok'}])}")
g = gates(CONV_H2, BR_B)
print(f"  gates on branch B: {g}")
fixture(g["anchor_len"] == 4, "the anchor really has 4 elements")
fixture(g["passes"],
        "all three gates pass on a DIFFERENT branch whose only shared "
        "content is one 'ok'")

CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(BR_B), CONV_H2))
seen = everything_the_model_will_see(out, CALLS)
missing_b = [i for i in range(24)
             if f"question B{i} " not in seen and f"answer B{i} " not in seen]
print(f"  branch-B exchanges absent from the whole request: {missing_b}")
broke(len(missing_b) > 0 and "BRANCH-A-CHUNK" in flat(out),
      f"H2: {len(missing_b)} branch-B exchanges deleted and replaced by "
      f"branch A's summary — the content gate passed on a single repeated "
      f"'ok'")

# Control: without the shared "ok" the same fork IS refused, so H2 is the
# prefix walk and not the gate being dead.
CONV_H2C = "h2_fork_control"
BR_B2 = history(24, tag="B")
seed(CONV_H2C, BR_A,
     [{"text": "BRANCH-A-CHUNK-COVERING-1-TO-40",
       "first_turn": 1, "last_turn": 40}],
     anchor_through=40)
CALLS.clear()
out_c = asyncio.run(main.compact_if_needed(list(BR_B2), CONV_H2C))
fixture("BRANCH-A-CHUNK" not in flat(out_c),
        "CONTROL: the same fork with NO shared turn is still refused")


# ===========================================================================
print()
print("=" * 74)
print("H3  _highest_chunk_turn PROVES WHERE COVERAGE ENDS. NOTHING ASKS")
print("    WHERE IT STARTS. A HEAD GAP DELETES TURNS WITH NO STAND-IN")
print("=" * 74)
print("""
`stored_turns` counts from index 0 of to_summarize, i.e. it assumes the
hierarchy covers turn 1. Two shipped paths in _do_l1_rollup leave a state
where it does not:

  * summarizer.py:1633  `if pos_last < 1:` sets last_summarized_turn =
    window_offset and returns True WITHOUT appending a chunk — turns
    1..window_offset are never summarized, by design, loudly, and no chunk
    claims them.
  * summarizer.py:1655  the `partial` branch sets covered_first =
    window_offset + 1, so the chunk HONESTLY records that it covers only
    from there.

Either way l1[0]["first_turn"] > 1 and _highest_chunk_turn is unchanged.
all_or_nothing cannot see it: nothing was dropped by the budget.
""")

CONV_H3 = "h2_headgap"
MSGS_H3 = history(24)
seed(CONV_H3, MSGS_H3,
     [{"text": "CHUNK-COVERING-ONLY-21-TO-40",
       "first_turn": 21, "last_turn": 40}],
     anchor_through=40)
g = gates(CONV_H3, MSGS_H3)
print(f"  gates: {g}")
fixture(g["passes"], "all three gates pass on the head-gapped hierarchy")

CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_H3), CONV_H3))
seen = everything_the_model_will_see(out, CALLS)
fixture("CHUNK-COVERING-ONLY-21-TO-40" in flat(out),
        "reuse fired and the stored text is in the array (control)")
lost = [i for i in range(10)
        if f"question {i} " not in seen and f"answer {i} " not in seen]
print(f"  exchanges 0-9 (turns 1-20) absent from the whole request: {lost}")
broke(len(lost) >= 5,
      f"H3: {len(lost)} exchanges ({len(lost) * 2} turns) that NO chunk "
      f"claims were deleted from the array and never handed to summarize() "
      f"— logged as 'covered by stored summaries'")


# ===========================================================================
print()
print("=" * 74)
print("H4  THE SAME BLINDNESS IN THE MIDDLE: load_state PARKS A CHUNK IT")
print("    CANNOT PARSE AND _highest_chunk_turn NEVER NOTICES THE HOLE")
print("=" * 74)
print("""
load_state files any entry failing _is_chunk under `_unrecognized` — by
design, so a schema change does not delete summaries (v3.1 F1b). _is_chunk
requires text.strip() truthy and both labels int. A parked MIDDLE chunk
leaves l1 = [1-10, 31-44] while _highest_chunk_turn is still 44, so the
substitution deletes turns 11-30 and the block that replaces them does not
mention them. Written straight to the state file, because that is the shape
a partly-written or cross-version file has on disk.
""")

CONV_H4 = "h2_midgap"
MSGS_H4 = history(24)
seed(CONV_H4, MSGS_H4,
     [{"text": "PLACEHOLDER", "first_turn": 1, "last_turn": 44}],
     anchor_through=40)
_p = summarizer.summary_path(CONV_H4)
_raw = json.loads(open(_p, "r", encoding="utf-8").read())
_raw["l1"] = [
    {"text": "GOOD-CHUNK-1-TO-10", "first_turn": 1, "last_turn": 10},
    {"text": "", "first_turn": 11, "last_turn": 30},          # unparseable
    {"text": "GOOD-CHUNK-31-TO-44", "first_turn": 31, "last_turn": 44},
]
_raw["last_summarized_turn"] = 44
open(_p, "w", encoding="utf-8").write(json.dumps(_raw))

_st4 = summarizer.load_state(CONV_H4)
print(f"  after load_state: l1 spans = "
      f"{[(c['first_turn'], c['last_turn']) for c in _st4['l1']]}, "
      f"parked = {len((_st4.get('_unrecognized') or {}).get('l1') or [])}")
fixture(len(_st4["l1"]) == 2 and summarizer._highest_chunk_turn(_st4) == 44,
        "the middle chunk was parked and _highest_chunk_turn is still 44")
g = gates(CONV_H4, MSGS_H4)
print(f"  gates: {g}")
fixture(g["passes"], "all three gates pass on the gapped hierarchy")

CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_H4), CONV_H4))
seen = everything_the_model_will_see(out, CALLS)
fixture("GOOD-CHUNK-31-TO-44" in flat(out), "reuse fired (control)")
lost = [i for i in range(5, 15)
        if f"question {i} " not in seen and f"answer {i} " not in seen]
print(f"  exchanges 5-14 (turns 11-30) absent from the whole request: {lost}")
broke(len(lost) >= 5,
      f"H4: {len(lost)} exchanges inside the PARKED chunk's span were "
      f"deleted with no stand-in anywhere in the request")


# ===========================================================================
print()
print("=" * 74)
print("H5  THE STAND-IN DOES NOT TRAVEL WITH THE REMOVAL. THE HARD BUDGET")
print("    GUARD IS EXPLICITLY ALLOWED TO SPEND COMPACTION'S OWN BLOCK")
print("=" * 74)
print("""
ced4520's safety argument, verbatim: "Stored text goes into the array this
function RETURNS, never into the separately-injected summary block - that
block is capped at 60% of the injection budget and can be trimmed or shed
downstream, so leaning on it to carry turns compaction removed would leave a
silent hole the moment it was shed."

_droppable_system_indices' own docstring answers it, main.py:3287:
"Everything after that prefix is ours - injected memory, AND COMPACTION'S
OWN SUMMARY BLOCK." The caller passes protect_system = the number of system
messages the CLIENT sent, so index 0 is protected and compaction's block,
which sits at index 1, is droppable and trimmable like any injected layer.
""")

CONV_H5 = "h2_shed"
MSGS_H5 = history(24)
seed(CONV_H5, MSGS_H5,
     [{"text": "SOUND-CHUNK-COVERING-1-TO-40 " + ("detail " * 400),
       "first_turn": 1, "last_turn": 40}],
     anchor_through=40)
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_H5), CONV_H5))
fixture("SOUND-CHUNK" in flat(out), "reuse fired; the stand-in is in the array")
_sys_idx = [i for i, m in enumerate(out) if m.get("role") == "system"]
_block_idx = [i for i, m in enumerate(out)
              if "SOUND-CHUNK" in str(m.get("content", ""))]
_caller_system = sum(1 for m in MSGS_H5 if m.get("role") == "system")
_droppable = main._droppable_system_indices(out, _caller_system)
print(f"  system indices {_sys_idx}, stand-in at {_block_idx}, "
      f"caller_system={_caller_system}, droppable={_droppable}")
broke(bool(set(_block_idx) & set(_droppable)),
      "H5a: the message carrying the stand-in for 40 removed turns is in "
      "_droppable_system_indices — the guard may delete it outright")

_before_len = max(len(str(m.get("content", ""))) for m in out
                  if "SOUND-CHUNK" in str(m.get("content", "")))
for _limit in (600, 300, 150, 80):
    _report: dict = {}
    _after = main._enforce_hard_budget(list(out), _limit, _caller_system,
                                       _report)
    _carriers = [str(m.get("content", "")) for m in _after
                 if "SOUND-CHUNK" in str(m.get("content", ""))]
    _after_len = max((len(c) for c in _carriers), default=0)
    print(f"  limit={_limit}: stand-in {_before_len} -> {_after_len} chars, "
          f"dropped_blocks={_report.get('dropped_blocks')}, "
          f"trimmed_blocks={_report.get('trimmed_blocks')}")
    if _after_len == 0:
        broke(True,
              f"H5b: _enforce_hard_budget(limit={_limit}) removed the "
              f"stand-in outright, leaving the 40 turns compaction deleted "
              f"represented by nothing at all")
        break
    if _after_len < _before_len:
        broke(True,
              f"H5b: _enforce_hard_budget(limit={_limit}) TRIMMED the "
              f"stand-in from {_before_len} to {_after_len} chars — the "
              f"stored hierarchy is cut mid-text while the turns it stands "
              f"in for are already gone from the array")
        break
else:
    broke(False, "H5b: the guard never touched the stand-in at any limit")


# ===========================================================================
print()
print("=" * 74)
print("H6  ON A REUSING TURN THE HIERARCHY IS RENDERED TWICE, AND THE TWO")
print("    RENDERS NOW DISAGREE ABOUT WHAT IT SAYS")
print("=" * 74)
print("""
Reported unfixed by the last pass. f78a389 made it sharper rather than
milder: main.py:1777 renders the block at SUMMARY_BLOCK_MAX_TOKENS with
all_or_nothing=True, main.py:5521 renders the SAME state at
min(SUMMARY_BLOCK_MAX_TOKENS, inject_budget*0.6) with the default. So on one
request the model gets a full hierarchy inside the array AND a
differently-trimmed copy of it as an injected block.
""")
_st6 = summarizer._empty_state("h2_double")
_st6["l1"] = [{"text": f"SCENE-{i} " + ("prose " * 500),
               "first_turn": i * 20 + 1, "last_turn": (i + 1) * 20}
              for i in range(9)]
_st6["last_summarized_turn"] = 180
_array_copy = summarizer.format_summary_block(
    _st6, summarizer.SUMMARY_BLOCK_MAX_TOKENS, all_or_nothing=True) or ""
_inject_copy = summarizer.format_summary_block(_st6, int(8192 * 0.6)) or ""
_in_array = {i for i in range(9) if f"SCENE-{i}" in _array_copy}
_in_inject = {i for i in range(9) if f"SCENE-{i}" in _inject_copy}
print(f"  array copy carries scenes {sorted(_in_array)}")
print(f"  injected copy carries scenes {sorted(_in_inject)}")
broke(bool(_in_array) and bool(_in_inject) and _in_array != _in_inject,
      f"H6: one request, two renders of one hierarchy, disagreeing — array "
      f"{len(_in_array)} scenes vs injected {len(_in_inject)}; the "
      f"{len(_in_array & _in_inject)} they share are sent twice")


# ===========================================================================
print()
print("=" * 74)
print("H7  GATE [2] AT EXACT EQUALITY, AND A LONGER ARRAY FOR AN UNRELATED")
print("    REASON")
print("=" * 74)
print("""
A CAPPED WINDOW ALIGNS BY CONSTRUCTION. The window a cap produces is the
LAST max_turns turns, and the anchor is the tail of the previous request —
so it is INSIDE the window and `_aligned` is True for every capped client.
The whole defence against a capped client is therefore the single length
comparison `_n >= _recorded_position`, and `_n` counts every message whose
role is not "system". Any extra role — tool, developer, function — lifts it
without adding one turn the hierarchy could have summarized.

The window below is a genuine suffix: turns 33-48 of a 48-turn
conversation, the way a max_turns=16 cap would send it. The hierarchy
covers turns 1-40. Its leading messages are turns 33..., NOT turns 1...
""")
CONV_H7 = "h2_capped"
FULL = history(24)
_ns_full = [m for m in FULL if m.get("role") != "system"]
WINDOW = [FULL[0]] + _ns_full[-16:]      # turns 33-48, plus the system msg
seed(CONV_H7, FULL,
     [{"text": "CHUNK-1-TO-40", "first_turn": 1, "last_turn": 40}],
     anchor_through=48)                  # the anchor is the LIVE tail
_g_before = gates(CONV_H7, WINDOW)
print(f"  capped window as shipped: {_g_before}")
fixture(_g_before["aligned"],
        "the capped window ALIGNS — the anchor is inside it by construction")
fixture(not _g_before["passes"],
        "and only the length gate declines it")
PADDED = WINDOW + [
    {"role": "tool", "content": f"tool result {i}"} for i in range(34)
]
_g_after = gates(CONV_H7, PADDED)
print(f"  same window + 34 tool turns: {_g_after}")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(PADDED), CONV_H7))
seen = everything_the_model_will_see(out, CALLS)
_lost = [i for i in range(16, 22)
         if f"question {i} " not in seen and f"answer {i} " not in seen]
print(f"  exchanges 16-21 (the WINDOW's own oldest turns, i.e. real turns "
      f"33-44) absent from the request: {_lost}")
broke(_g_after["passes"] and "CHUNK-1-TO-40" in flat(out) and bool(_lost),
      f"H7: tool-role padding lifts a capped window over the only gate that "
      f"declines it; the substitution then treats the window's leading "
      f"messages as turns 1..40 and deletes {len(_lost)} exchanges of real "
      f"turns 33+ under a summary of turns 1-40")


# ===========================================================================
print()
print("=" * 74)
print("H8  THE CAPPED-CLIENT GATE IS UNCOVERED: test_compaction_reuse [5]")
print("    DECLINES FOR THE WRONG REASON")
print("=" * 74)
print("""
Mutation `_covered > 0 and _n >= _recorded_position(_st) and _aligned`
-> `_covered > 0 and _aligned` leaves test_compaction_reuse.py GREEN
(needle occurrence count 1, verified before apply). ced4520 reported that
same mutation RED — correctly, at the time. f78a389 then added `_aligned`,
and [5]'s fixture sends history(8) against an anchor taken from turns 37-40
of history(24): those four turns appear nowhere in the 16-turn window, so
[5] now declines on ALIGNMENT and its conclusion about the length gate
comes from the wrong conjunct.

Below: [5]'s own fixture, with the gates printed one at a time; then a
capped window that DOES align, which is what a real cap produces. That
second case is the guard the suite is missing — it HOLDS as shipped and
goes red under the mutation above.
""")
CONV_H8 = "h2_five"
MSGS_H8 = history(24)
seed(CONV_H8, MSGS_H8,
     [{"text": "STORED-CHUNK-ONE", "first_turn": 1, "last_turn": 20},
      {"text": "STORED-CHUNK-TWO", "first_turn": 21, "last_turn": 40}],
     anchor_through=40)
_g5 = gates(CONV_H8, history(8))
print(f"  test [5]'s exact fixture: {_g5}")
broke(not _g5["aligned"],
      "H8a: test_compaction_reuse [5] declines because _aligned is False, "
      "not because the length gate fired — the capped-client gate ced4520 "
      "added has no test that isolates it")

CONV_H8B = "h2_capped_aligning"
FULL8 = history(24)
_ns8 = [m for m in FULL8 if m.get("role") != "system"]
WINDOW8 = [FULL8[0]] + _ns8[-16:]
seed(CONV_H8B, FULL8,
     [{"text": "CAPPED-CHUNK-1-TO-40", "first_turn": 1, "last_turn": 40}],
     anchor_through=48)
_g8 = gates(CONV_H8B, WINDOW8)
print(f"  a capped window that aligns: {_g8}")
fixture(_g8["aligned"] and _g8["covered"] > 0,
        "two of three gates pass; only the length gate stands between this "
        "window and the substitution")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(WINDOW8), CONV_H8B))
seen = everything_the_model_will_see(out, CALLS)
_win_lost = [i for i in range(16, 22)
             if f"question {i} " not in seen and f"answer {i} " not in seen]
broke(bool(_win_lost) or "CAPPED-CHUNK" in flat(out),
      f"H8b: the aligning capped window lost {len(_win_lost)} of its own "
      f"exchanges to a summary of turns 1-40 (THIS IS THE MISSING GUARD: it "
      f"holds as shipped and goes red when the length gate is removed)")


# ===========================================================================
print()
print("=" * 74)
print("H9  REUSE CAN BE PERMANENTLY OFF AND NOTHING SAYS SO")
print("=" * 74)
print("""
`turns_seen` is MONOTONIC and advances by _ASSUMED_NEW_TURNS (2) whenever
the anchor cannot be found in the window (summarizer.py:686). Once it sits
above the array's own length, `_n >= _recorded_position` is false FOREVER
and the 117-second regression is back.

The only log evidence reuse ever happened is the optional
", N covered by stored summaries" suffix. There is no line on the declining
path at all, so "reuse declined this turn", "this conversation has no
hierarchy yet" and "conv_id was None" are textually identical in the log.
That is the shape the 2026-08-29 outage had: a feature silently doing
nothing while every line reads healthy.
""")
CONV_H9 = "h2_silent"
MSGS_H9 = history(24)
seed(CONV_H9, MSGS_H9,
     [{"text": "NEVER-REUSED-CHUNK", "first_turn": 1, "last_turn": 40}],
     anchor_through=40, turns_seen=49)      # ONE past the array's 48
_g9 = gates(CONV_H9, MSGS_H9)
print(f"  gates with turns_seen=49 against n=48: {_g9}")
CALLS.clear()
out = asyncio.run(main.compact_if_needed(list(MSGS_H9), CONV_H9))
_declined = (not any("NEVER-REUSED-CHUNK" in str(m.get("content", ""))
                     for m in out)) and bool(CALLS) and len(CALLS[0]) == 44
broke(_declined,
      "H9: turns_seen one turn ahead of the array disables reuse completely "
      "and the compaction log line is byte-identical in shape to a healthy "
      "no-hierarchy turn — no counter, no WARNING, nothing to alert on")


# ===========================================================================
print()
print("=" * 74)
print(f"BREAKS REPRODUCED: {len(BROKEN)}")
for b in BROKEN:
    print(f"  - {b}")
print(f"HELD: {len(HELD)}")
for h in HELD:
    print(f"  - {h}")
if FIXTURE_FAILED:
    print(f"FIXTURES THAT FAILED (those attacks never ran): "
          f"{len(FIXTURE_FAILED)}")
    for f in FIXTURE_FAILED:
        print(f"  - {f}")
print("=" * 74)
sys.exit(0)
