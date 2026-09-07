"""A skipped memory tail must still advance the hierarchy (v3.1.8).

THE DEFECT. `_run_memory_tail` decides whether a reply enters memory. When it
says no — a repetition loop, a reply with no usable text — it returned before
firing the tail at all, and the hierarchical rollup is inside that tail. So
one bad reply skipped the rollup as well as the reply.

That is not a rounding error. The rollup does not summarize the CURRENT
reply; it summarizes turns already in the history, and `_redact_degenerate_turns`
is how it handles degenerate ones. Coupling them meant a model that loops for
n turns froze the watermark for n turns, with no floor and nothing to recover
it afterwards, because the rollup is only ever driven from the tail.

MEASURED, not reasoned: the soak ran 22 turns against a fixture emitting
adversarial replies, logged "14 consecutive memory-tail skip(s)", and finished
with `last_summarized_turn` still 0 — "the hierarchy is dead", which is the
2026-08-28 failure this release exists to fix, reached by a third route. The
same 14-loop shape appears in the production logs of 08-29 and 09-01.

WHAT THIS PINS. Both halves, because a fix that advanced the watermark by
summarizing the loop would be worse than the freeze:

  * the rollup runs even when the tail is skipped;
  * the skipped reply is NOT in what gets rolled up.

    python test_rollup_on_skip.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="rollup-skip-")

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


CONV = "rollup_on_skip"
RULE = "━"
# Long enough to trip the detector, from test_degenerate_reply's own corpus:
# the shortest real degenerate run measured was 386 characters.
DEGENERATE = "# Status\n\n```\n" + RULE * 569 + "\n"
HISTORY = [
    {"role": "user", "content": "Tell me about the garden."},
    {"role": "assistant", "content": "The roses are doing well this year."},
    {"role": "user", "content": "And the tea?"},
]

check(main.reply_is_degenerate(DEGENERATE),
      "the fixture reply is degenerate — without this the test proves nothing")

print("[1] a skipped tail still schedules the hierarchy rollup")

scheduled: list = []


def _spy_fire(coro, label=None):
    scheduled.append((coro, label))
    # True, not None: v3.1.8 gave _fire_and_forget a bool contract and a
    # double that returns None reads as "the pool shed it".
    return True


rolled: list = []


async def _spy_rollup(conv_id, messages, vllm_url, model):
    rolled.append(list(messages))
    return summarizer.load_state(conv_id)


_real_fire = main._fire_and_forget
_real_rollup = summarizer.maybe_rollup
main._fire_and_forget = _spy_fire
summarizer.maybe_rollup = _spy_rollup
try:
    decision = main._run_memory_tail(
        CONV,
        DEGENERATE,
        finished=True,
        truncated=False,
        holed=False,
        touched_facts=[],
        last_user_text="And the tea?",
        turn_index=2,
        messages=list(HISTORY),
        injected_facts=None,
    )
    check(not decision.store,
          "the reply itself is still refused — this fix must not smuggle a "
          "repetition loop into memory")
    check(len(scheduled) == 1,
          f"exactly one piece of background work was scheduled "
          f"(got {len(scheduled)})")
    if scheduled:
        check("rollup" in (scheduled[0][1] or ""),
              f"and it is labelled as the rollup (got {scheduled[0][1]!r}) — "
              f"the label is what the pool's shed warning names")
        asyncio.run(scheduled[0][0])

    print("[2] and the degenerate reply is NOT in what gets rolled up")
    check(len(rolled) == 1, f"maybe_rollup was called once (got {len(rolled)})")
    if rolled:
        texts = [str(m.get("content", "")) for m in rolled[0]]
        check(not any(RULE * 100 in t for t in texts),
              "no run of rule characters reached the rollup input — the reply "
              "was excluded, not summarized")
        check(any("roses" in t for t in texts),
              "while the sound history that PRECEDES it did reach it, which is "
              "the whole point of running the rollup at all")
finally:
    main._fire_and_forget = _real_fire
    summarizer.maybe_rollup = _real_rollup

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll rollup-on-skip checks passed.")
