"""
L3 must never stop covering the beginning of the conversation.

The summary hierarchy is l1 chunks -> l2 chapters -> a single l3 theme. When
l2 was bounded (MEMORY_REVIEW S-1) by clearing it after each L3 refresh, the
refresh still built its input from the pending chapters ALONE and overwrote
`state["l3"]` wholesale, taking first_turn from `l2[0]`. So:

    refresh 1: l3 covers turns 1-N,        l2 cleared
    refresh 2: l3 REPLACED, covers N+1-M   <- turns 1-N deleted, silently

L1->L2 can drop its inputs safely because it APPENDS to a list. L3 is a single
object that is REPLACED, so it must be fed back into its own refresh. The two
tiers do not have the same contract, and the difference is list-versus-object.
This asserts the property directly, because the bug is invisible from any
single refresh - it only appears on the SECOND one.

    python test_l3_coverage.py
"""

import asyncio
import os
import sys

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")

import summarizer  # noqa: E402

FAILED = []
SEEN_INPUTS = []


async def _fake_summarize_pieces(conv_id, client, vllm_url, model, prompt,
                                 pieces, max_tokens):
    """Record what the tier was asked to summarize, and echo a marker."""
    SEEN_INPUTS.append(list(pieces))
    return "THEME(" + " | ".join(p.splitlines()[0] for p in pieces) + ")"


def chapter(first, last):
    return {"text": f"chapter covering {first}-{last}", "first_turn": first,
            "last_turn": last}


def run_refresh(state):
    return asyncio.run(
        summarizer._do_l3_rollup("t", None, "http://stub", "m", state)
    )


orig = summarizer._summarize_pieces
summarizer._summarize_pieces = _fake_summarize_pieces
try:
    N = summarizer.L3_CHUNK_SIZE

    print("[1] the first refresh covers the chapters it consumed")
    state = {"l1": [], "l2": [chapter(1 + 10 * i, 10 + 10 * i) for i in range(N)],
             "l3": None}
    if not run_refresh(state):
        FAILED.append("the first refresh did not run at all")
    else:
        l3 = state["l3"]
        print(f"  ok   l3 covers turns {l3['first_turn']}-{l3['last_turn']}, "
              f"l2 now holds {len(state['l2'])}")
        if l3["first_turn"] != 1:
            FAILED.append(f"first refresh: first_turn {l3['first_turn']} != 1")

    print()
    print("[2] the SECOND refresh must still cover turn 1 - the actual bug")
    first_text = state["l3"]["text"]
    base = 10 * N
    state["l2"] = [chapter(base + 1 + 10 * i, base + 10 + 10 * i) for i in range(N)]
    if not run_refresh(state):
        FAILED.append("the second refresh did not run at all")
    else:
        l3 = state["l3"]
        print(f"  ok   l3 covers turns {l3['first_turn']}-{l3['last_turn']}")
        if l3["first_turn"] != 1:
            FAILED.append(
                f"second refresh: l3 now starts at turn {l3['first_turn']}, "
                f"not 1 - everything before it was DELETED"
            )
        if l3["last_turn"] != base + 10 * N:
            FAILED.append(
                f"second refresh: last_turn {l3['last_turn']} did not advance"
            )

    print()
    print("[3] the previous L3 was actually fed back in as input")
    # The span alone could be inherited while the TEXT is still discarded,
    # which would leave l3 claiming a coverage it does not describe.
    last_input = SEEN_INPUTS[-1]
    carried = any(first_text in p for p in last_input)
    if not carried:
        FAILED.append(
            "the second refresh did not receive the previous L3 text as "
            "input - the span says turns 1+, but the content does not "
            "describe them"
        )
    else:
        print(f"  ok   the prior L3 body was input piece 1 of {len(last_input)}")

    print()
    print("[4] the input stays bounded across many refreshes")
    widths = []
    for r in range(6):
        b = 10 * N * (r + 2)
        state["l2"] = [chapter(b + 1 + 10 * i, b + 10 + 10 * i) for i in range(N)]
        run_refresh(state)
        widths.append(len(SEEN_INPUTS[-1]))
        if state["l3"]["first_turn"] != 1:
            FAILED.append(
                f"refresh {r + 3}: coverage start slipped to "
                f"{state['l3']['first_turn']}"
            )
    if max(widths) > N + 1:
        FAILED.append(f"input grew unbounded: piece counts {widths}")
    else:
        print(f"  ok   input piece count stayed at {widths} "
              f"(prior L3 + {N} chapters), l2 bounded at {len(state['l2'])}")

    print()
    print("[5] a first-ever refresh with no prior L3 still works")
    fresh = {"l1": [], "l2": [chapter(1, 10)] * N, "l3": None}
    if not run_refresh(fresh) or fresh["l3"]["first_turn"] != 1:
        FAILED.append("a refresh with no prior L3 regressed")
    else:
        print("  ok   no prior L3 is not an error")
finally:
    summarizer._summarize_pieces = orig

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All L3 coverage tests passed.")
