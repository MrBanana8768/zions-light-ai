"""ADVERSARIAL: compact_if_needed's stored-summary substitution (ced4520).

Run inside the unit-suite image, which can import `main`:

  docker compose -f docker-compose.tests.yml run --rm --entrypoint /bin/bash \
    unit-tests -c 'cp -r /src /work && cd /work/compactor && \
    /opt/compactor-venv/bin/python /work/tests/adversarial/test_adv_v319_reuse.py'

ced4520 deletes `stored_turns` live messages from the array and substitutes
`format_summary_block(...)` in their place. Its stated safety property is:

    "THE STAND-IN TRAVELS WITH THE REMOVAL ... Under-claiming coverage costs
     a little speed; over-claiming replaces live turns with a summary of
     different ones."

and its only guard is

    if _covered > 0 and _n >= summarizer._recorded_position(_st):

This file attacks the gap between what that guard proves (a LENGTH relation)
and what the substitution needs (a CONTENT/INDEX relation).
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="adv-reuse-")
os.environ["COMPACTOR_TARGET_TOKENS"] = "500"

sys.path.insert(0, "/work/compactor")

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []
BROKEN: list[str] = []


def broke(cond, label):
    """cond True == the break reproduced."""
    if cond:
        print(f"  *** BROKE: {label}")
        BROKEN.append(label)
    else:
        print(f"  (held)   {label}")


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


CALLS: list[list[str]] = []


async def _spy_summarize(client, to_summarize):
    CALLS.append([str(m.get("content", ""))[:40] for m in to_summarize])
    return "FRESHLY-SUMMARIZED", []


main.summarize = _spy_summarize


def history(n_exchanges: int, tag: str = "") -> list[dict]:
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n_exchanges):
        out.append({"role": "user",
                    "content": f"U{tag}{i} " + ("word " * 30)})
        out.append({"role": "assistant",
                    "content": f"A{tag}{i} " + ("word " * 30)})
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


def fresh_state(conv: str) -> dict:
    p = summarizer._state_path(conv) if hasattr(summarizer, "_state_path") else None
    st = summarizer._empty_state(conv)
    return st


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("A1  THE STAND-IN IS TRIMMED OLDEST-FIRST; THE REMOVAL IS NOT TRIMMED")
print("=" * 74)
print("""
format_summary_block drops OLDEST L1 scenes first when over budget
(summarizer.py:452-457, 'a squeeze drops the OLDEST scenes first').
compact_if_needed removes the OLDEST `stored_turns` messages. The two are
inverted: the turns removed are exactly the ones whose stand-in is dropped.
The only guard is `if not stored_text` -- it fires only when EVERY tier was
dropped, never on a partial trim.
""")

# 24 L1 chunks x 20 turns = turns 1..480, each chunk ~500 tokens of text.
CONV_A1 = "adv_a1_squeeze"
st = summarizer._empty_state(CONV_A1)
CHUNK_BODY = "scene text " * 180          # ~2000 chars ~= 500 est. tokens
st["l1"] = [
    {"text": f"MARKER-{i:03d} " + CHUNK_BODY,
     "first_turn": i * 20 + 1, "last_turn": (i + 1) * 20}
    for i in range(24)
]
st["last_summarized_turn"] = 480
summarizer.save_state(CONV_A1, st)

_block = summarizer.format_summary_block(st, summarizer.SUMMARY_BLOCK_MAX_TOKENS)
kept = [i for i in range(24) if f"MARKER-{i:03d}" in (_block or "")]
dropped = [i for i in range(24) if i not in kept]
print(f"  format_summary_block kept {len(kept)}/24 scenes, dropped {dropped}")

MSGS_A1 = history(250)                     # 500 non-system turns >= 480
CALLS.clear()
out_a1 = asyncio.run(main.compact_if_needed(list(MSGS_A1), CONV_A1))
txt_a1 = flat(out_a1)

_covered = summarizer._highest_chunk_turn(st)
_n = len([m for m in MSGS_A1 if m.get("role") != "system"])
print(f"  guard: _covered={_covered}, _n={_n}, "
      f"_recorded_position={summarizer._recorded_position(st)} "
      f"-> substitution {'APPLIES' if _n >= summarizer._recorded_position(st) else 'declines'}")

lost_spans = []
for i in dropped:
    # turns (i*20+1)..((i+1)*20); turn k is non-system index k-1.
    # exchange e: user = turn 2e+1, assistant = turn 2e+2
    first_turn = i * 20 + 1
    e = (first_turn - 1) // 2
    tok = f"U{e} "
    if tok not in txt_a1 and f"MARKER-{i:03d}" not in txt_a1:
        lost_spans.append(i)

print(f"  scenes whose TEXT was removed and whose SUMMARY was also dropped: "
      f"{lost_spans}")
broke(len(lost_spans) > 0,
      f"A1: {len(lost_spans) * 20} turns removed from the array with NO "
      f"stand-in anywhere in it (scenes {lost_spans})")
check(len(CALLS) == 1, "summarize still ran for the uncovered tail")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("A1b  THE SAME SQUEEZE AT DEFAULT CAPACITY, VIA NON-ASCII")
print("=" * 74)
print("""
_estimate_block_tokens prices non-ASCII at ONE TOKEN PER UTF-8 BYTE
(summarizer.py:284-300). L1_MAX_TOKENS bounds the model's OUTPUT tokens, not
its bytes. A hierarchy summarizing a conversation in a non-Latin script -- or
merely one with emoji -- is therefore priced 2-4x its real size, so the
docstring's '11,300 capacity vs 12,000 budget' headroom (6%) is gone and the
cap fires on a state that is WITHIN its documented bounds.
""")

CONV_A1B = "adv_a1b_unicode"
st2 = summarizer._empty_state(CONV_A1B)
# 9 L1 + 4 L2 + L3: exactly the documented maximum shape, nothing over-full.
CYR = "разговор о жизни и о том что было раньше "     # 2 bytes/char
st2["l1"] = [
    {"text": f"SCENE-{i} " + CYR * 30, "first_turn": i * 20 + 1,
     "last_turn": (i + 1) * 20}
    for i in range(9)
]
st2["l2"] = [
    {"text": f"CHAPTER-{i} " + CYR * 70, "first_turn": 1, "last_turn": 180}
    for i in range(4)
]
st2["l3"] = {"text": "THEME " + CYR * 120, "first_turn": 1, "last_turn": 180}
st2["last_summarized_turn"] = 180
summarizer.save_state(CONV_A1B, st2)

blk2 = summarizer.format_summary_block(st2, summarizer.SUMMARY_BLOCK_MAX_TOKENS)
kept2 = [i for i in range(9) if f"SCENE-{i}" in (blk2 or "")]
dropped2 = [i for i in range(9) if i not in kept2]
print(f"  a state at its DOCUMENTED capacity (9 L1 / 4 L2 / 1 L3) kept "
      f"{len(kept2)}/9 scenes; dropped {dropped2}")

MSGS_A1B = history(100)                   # 200 non-system turns >= 180
CALLS.clear()
out_a1b = asyncio.run(main.compact_if_needed(list(MSGS_A1B), CONV_A1B))
txt_a1b = flat(out_a1b)
lost2 = []
for i in dropped2:
    e = (i * 20 + 1 - 1) // 2
    if f"U{e} " not in txt_a1b and f"SCENE-{i}" not in txt_a1b:
        lost2.append(i)
broke(len(lost2) > 0,
      f"A1b: {len(lost2) * 20} turns removed with no stand-in, on a state "
      f"that never exceeded its documented bounds (scenes {lost2})")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("A2  TURN NUMBERS vs ARRAY INDICES: ONE IMAGE TURN SHIFTS THE MAPPING")
print("=" * 74)
print("""
_covered is a TURN number. The summarizer counts turns as ALL non-system
messages (_observed_position: `[m for m in messages if role != system]`).
compact_if_needed indexes into `text_only`, which has the IMAGE turns removed
(main.py:1666). So text_only[:_covered] runs PAST turn _covered by exactly
the number of image turns inside the covered span -- and those extra turns
are live, uncovered, and deleted.

Reachable at the DEFAULT config: COMPACTOR_MAX_RETAINED_IMAGES=1 leaves
exactly one image-bearing turn, and _apply_image_retention keeps the most
RECENT one, which sits in to_summarize as soon as it is older than
KEEP_RECENT_TURNS(=4).
""")

IMG = {"role": "user", "content": [
    {"type": "text", "text": "look at this"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
]}
check(main._message_has_image(IMG), "fixture really is an image turn")

CONV_A2 = "adv_a2_image"
st3 = summarizer._empty_state(CONV_A2)
st3["l1"] = [{"text": "COVERS-1-TO-40", "first_turn": 1, "last_turn": 40}]
st3["last_summarized_turn"] = 40
summarizer.save_state(CONV_A2, st3)

MSGS_A2 = history(30)                     # 60 non-system turns
# Put ONE image turn early, inside the covered span (non-system index 2 ->
# turn 3). Everything else is text.
MSGS_A2[1 + 2] = IMG                      # +1 for the system message

CALLS.clear()
out_a2 = asyncio.run(main.compact_if_needed(list(MSGS_A2), CONV_A2))
txt_a2 = flat(out_a2)
sent_a2 = CALLS[0] if CALLS else []
sent_txt = " ".join(sent_a2)

# turn 41 = non-system index 40 = exchange 20's USER message = "U20 "
# The chunk claims only through turn 40 (= "A19").
print(f"  chunk covers turns 1-40 (i.e. through 'A19'); "
      f"turn 41 is 'U20'")
print(f"  first thing handed to summarize(): {sent_a2[0] if sent_a2 else None!r}")
gone = ("U20 " not in txt_a2) and ("U20 " not in sent_txt)
broke(gone,
      "A2: turn 41 ('U20') was deleted from the array and never handed to "
      "summarize(), although the hierarchy only claims turns 1-40 -- one "
      "live turn replaced by a summary that does not cover it")

# scale it: N images in the covered span -> N turns past the coverage
CONV_A2B = "adv_a2b_image_scale"
st4 = summarizer._empty_state(CONV_A2B)
st4["l1"] = [{"text": "COVERS-1-TO-40", "first_turn": 1, "last_turn": 40}]
st4["last_summarized_turn"] = 40
summarizer.save_state(CONV_A2B, st4)
MSGS_A2B = history(30)
for idx in (2, 4, 6, 8, 10):              # 5 image turns, all inside 1..40
    MSGS_A2B[1 + idx] = dict(IMG)
CALLS.clear()
out_a2b = asyncio.run(main.compact_if_needed(list(MSGS_A2B), CONV_A2B))
txt_a2b = flat(out_a2b) + " " + " ".join(CALLS[0] if CALLS else [])
lost_turns = [t for t in range(41, 46)
              if (f"U{(t - 1) // 2} " if t % 2 else f"A{(t - 2) // 2} ") not in txt_a2b]
broke(len(lost_turns) >= 3,
      f"A2b: with 5 retained image turns the overrun is {len(lost_turns)} "
      f"live turns past the claimed coverage (turns {lost_turns})")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("A3  THE GUARD CHECKS LENGTH, NEVER CONTENT (branch / regenerate)")
print("=" * 74)
print("""
OpenWebUI keeps branches inside ONE chat, so conv_id does not change when the
user edits an earlier message and regenerates. The hierarchy still holds
chunks summarizing the ABANDONED branch's turns. `_n >= _recorded_position`
is satisfied as soon as the new branch is at least as long, and the stored
summary of branch A's turns then replaces branch B's live turns.

The state file already carries `tail_fp` -- content fingerprints of the array
that was summarized. The new code does not look at them.
""")

CONV_A3 = "adv_a3_fork"
st5 = summarizer._empty_state(CONV_A3)
st5["l1"] = [
    {"text": "BRANCH-A-SCENE-1", "first_turn": 1, "last_turn": 20},
    {"text": "BRANCH-A-SCENE-2", "first_turn": 21, "last_turn": 40},
]
st5["last_summarized_turn"] = 40
st5["turns_seen"] = 40
summarizer.save_state(CONV_A3, st5)

# Branch B: the user rewound to turn 20 and said something different.
BRANCH_B = history(30, tag="B")
CALLS.clear()
out_a3 = asyncio.run(main.compact_if_needed(list(BRANCH_B), CONV_A3))
txt_a3 = flat(out_a3) + " " + " ".join(CALLS[0] if CALLS else [])
missing_b = [i for i in range(20) if f"UB{i} " not in txt_a3]
print(f"  branch-B exchanges absent from everything the model will see: "
      f"{missing_b[:8]}{'...' if len(missing_b) > 8 else ''} "
      f"({len(missing_b)} of 30)")
broke(len(missing_b) > 0 and "BRANCH-A-SCENE-1" in flat(out_a3),
      f"A3: {len(missing_b)} branch-B exchanges deleted and replaced by a "
      f"summary of branch A -- the guard passed on LENGTH alone "
      f"(_n={len([m for m in BRANCH_B if m.get('role') != 'system'])} >= "
      f"_recorded_position={summarizer._recorded_position(st5)})")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("A4  THE HIERARCHY IS NOW RENDERED INTO THE REQUEST TWICE")
print("=" * 74)
print("""
main.py:1707 renders format_summary_block(state, SUMMARY_BLOCK_MAX_TOKENS)
into the array compact_if_needed RETURNS. main.py:5406 renders
format_summary_block(state, min(SUMMARY_BLOCK_MAX_TOKENS, budget*0.6)) into
injected_blocks on the SAME request, AFTER compaction. Nothing tells either
about the other, so the same L1/L2/L3 text is sent to the model twice, and
the second copy competes with persona/facts/retrieval inside
_bound_injected_blocks.
""")
_in_array = summarizer.format_summary_block(
    st5, summarizer.SUMMARY_BLOCK_MAX_TOKENS) or ""
_injected = summarizer.format_summary_block(st5, int(8192 * 0.6)) or ""
dupe = bool(_in_array) and bool(_injected) and (
    "BRANCH-A-SCENE-2" in _in_array and "BRANCH-A-SCENE-2" in _injected)
broke(dupe and "BRANCH-A-SCENE-2" in flat(out_a3),
      "A4: the same scene text is in BOTH the compacted array and the "
      "separately-injected summary block on one request")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"BREAKS REPRODUCED: {len(BROKEN)}")
for b in BROKEN:
    print(f"  - {b}")
if FAILED:
    print(f"HARNESS CHECKS FAILED: {len(FAILED)}")
    for f in FAILED:
        print(f"  - {f}")
print("=" * 74)
sys.exit(0)
