"""
summarize() must never lose a turn.

Every turn handed to summarize() is either represented in the returned summary
or handed back in the deferred list. Never neither.

WHY THIS EXISTS. _summarize_once returns "" for an HTTP 200 whose content is
empty - no exception, nothing logged, and vLLM does return those (a generation
that hits the stop token immediately, an empty choices[0].message.content).
summarize() then returned ("", []) and compact_if_needed built a payload with
no summary block and no older turns. Every summarized turn deleted from the
request, silently, with a log line reading "compacted: summarized 80 text
turn(s)".

It is tested at BOTH returns on purpose. The bug was found at the multi-batch
return, fixed there, and the single-batch fast path one screen above it was
missed - the same fix-one-site-miss-the-sibling defect as the two /tokenize
outages. A test that only exercises one path would have passed over the bug.

    python test_summarize_invariant.py
"""

import asyncio
import os
import sys

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")

import main  # noqa: E402

FAILED = []


def turns(n):
    return [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"turn {i} " + "filler " * 40}
        for i in range(n)
    ]


def run(label, n, reply, *, expect_summary):
    """Drive summarize() with a stub _summarize_once and check the invariant."""
    async def stub(client, batch):
        return reply

    original = main._summarize_once
    main._summarize_once = stub
    try:
        msgs = turns(n)
        summary, deferred = asyncio.run(main.summarize(None, msgs))
    finally:
        main._summarize_once = original

    got_summary = bool((summary or "").strip())
    if got_summary != expect_summary:
        FAILED.append(
            f"{label}: summary present={got_summary}, wanted {expect_summary}"
        )
        return
    if not got_summary and len(deferred) != len(msgs):
        FAILED.append(
            f"{label}: no summary, but only {len(deferred)} of {len(msgs)} "
            f"turn(s) handed back - {len(msgs) - len(deferred)} LOST"
        )
        return
    where = "in the summary" if got_summary else f"{len(deferred)} deferred"
    print(f"  ok   {label}  ({where})")


print("[1] an empty 200 must not delete turns - single batch")
# One batch is the fast-path return. This is the case that was still broken
# after the first fix.
run("8 turns, empty reply", 8, "", expect_summary=False)
run("8 turns, whitespace-only reply", 8, chr(32)*3 + chr(10) + chr(9), expect_summary=False)

print()
print("[2] an empty 200 must not delete turns - many batches")
# Enough turns to force _chunk_to_budget past one batch and through the
# map/reduce return at the bottom of the function.
run("400 turns, empty reply", 400, "", expect_summary=False)
run("400 turns, whitespace-only reply", 400, chr(32)*2 + chr(10), expect_summary=False)

print()
print("[3] a real summary still summarizes")
run("8 turns, real reply", 8, "They talked about her week.", expect_summary=True)
run("400 turns, real reply", 400, "They talked about her week.",
    expect_summary=True)

print()
print("[4] the invariant, stated directly over every shape")
# Not a restatement of the cases above: this asserts the property itself, so a
# future path added to summarize() is covered without anyone remembering to
# add a case here.
for n in (2, 8, 40, 200, 400):
    for reply in ("", "   ", "a real summary"):
        async def stub(client, batch, _r=reply):
            return _r
        original = main._summarize_once
        main._summarize_once = stub
        try:
            msgs = turns(n)
            summary, deferred = asyncio.run(main.summarize(None, msgs))
        finally:
            main._summarize_once = original
        if not (summary or "").strip() and len(deferred) != len(msgs):
            FAILED.append(
                f"n={n} reply={reply!r}: {len(msgs) - len(deferred)} turn(s) "
                f"neither summarized nor deferred"
            )
print("  ok   no shape produces a turn that is neither summarized nor deferred")

print()
if FAILED:
    for f in FAILED:
        print(f"FAIL {f}")
    sys.exit(1)
print("All summarize invariant tests passed.")
