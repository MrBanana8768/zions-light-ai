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


# ---------------------------------------------------------------------------
# THE GATE. Two conditions decide whether the skip path rolls up at all: the
# model produced something (raw_chars), and there is an earlier exchange to
# summarize (_has_conversational_history). THREE separate defects have come
# out of this one gate and every one of them was a condition that could not
# fire - first two labels `_run_memory_tail` computes only behind
# `if decision.store`, then a `not _task_traffic` conjunct the history check
# in front of it had already guaranteed. A gate with no test is how the first
# shipped: the broken version passed every suite in the repo.
# ---------------------------------------------------------------------------


def _fire_once(**patches):
    """Run one skipped-tail decision, return the labels scheduled."""
    labels: list = []

    def _spy(coro, label=None):
        labels.append(label)
        coro.close()
        return True

    saved = {k: getattr(main, k) for k in patches}
    main._fire_and_forget = _spy
    for k, v in patches.items():
        setattr(main, k, v)
    try:
        main._run_memory_tail(
            CONV, DEGENERATE, finished=True, truncated=False, holed=False,
            touched_facts=[], last_user_text="And the tea?", turn_index=2,
            messages=list(HISTORY), injected_facts=None,
        )
    finally:
        for k, v in saved.items():
            setattr(main, k, v)
        main._fire_and_forget = _real_fire
    return labels


print("[3] the history check SUBSUMES the task-traffic check")
# This step used to patch _is_repeat_task_traffic to return True while
# handing the gate an array WITH history - a combination the real predicate
# cannot produce, so it asserted nothing and stayed green with the conjunct
# deleted. What actually keeps task traffic out is the history check: the
# predicate opens with `if _has_conversational_history(messages): return
# False`, so by the time `and not _task_traffic` was reached the conjunct in
# front of it had already guaranteed the answer. That is why the dead
# conjunct could be removed, and the subsumption is pinned HERE rather than
# left in a comment. Take the early return out of _is_repeat_task_traffic
# and this goes red - which is the signal that the gate needs a
# task-traffic condition of its own back.
#
# THE POSITION HAS TO BE SEEDED PAST THE THRESHOLD. The predicate ends with
# `return position >= TASK_TRAFFIC_MIN_POSITION`, so a conversation that has
# not reached it answers False for BOTH reasons and the check passes whether
# the early return is there or not. The first draft of this step did exactly
# that and stayed green under the mutation - the same vacuous shape it was
# written to replace. Seeding the position above the line leaves the early
# return as the ONLY thing that can make the answer False.
SUB_CONV = "rollup_subsumption"
_sub = summarizer.load_state(SUB_CONV)
_sub["turns_seen"] = main.TASK_TRAFFIC_MIN_POSITION + 2
summarizer.save_state(SUB_CONV, _sub)
check(main._is_repeat_task_traffic(SUB_CONV, [{"role": "user", "content": "Title?"}]),
      "the seeded conv_id IS task traffic when the array carries no assistant "
      "turn - without this the next check proves nothing")
check(not main._is_repeat_task_traffic(SUB_CONV, list(HISTORY)),
      "and the SAME conv_id at the SAME position answers False as soon as the "
      "array carries history - the subsumption the removed conjunct rested on")
check(_fire_once() != [],
      "and the gate still schedules on that array - so [3] is not passing "
      "because the gate refuses everything")

print("[4] disk pressure does NOT roll up")
# A rollup WRITES state, so it is subject to the same pause as every other
# new-memory write. The check lives inside _rollup_hierarchy, so it holds
# for BOTH callers rather than the one that remembered.
rolled.clear()
_real_guard = main.degrade.guard
main.degrade.guard = lambda _label: False
summarizer.maybe_rollup = _spy_rollup
try:
    for coro, _label in [(main._rollup_hierarchy(CONV, list(HISTORY), None), None)]:
        asyncio.run(coro)
finally:
    main.degrade.guard = _real_guard
    summarizer.maybe_rollup = _real_rollup
check(rolled == [],
      "no rollup happens while degrade.guard has writes paused")

print("[5] a reply that never arrived does NOT roll up")
# A backend rejection adds no turn to roll up and needs the same backend the
# rollup would call, so during a vLLM outage every 400 would fire a
# summarization at the process that is already failing. The discriminator is
# raw_chars: nothing arrived, so there is nothing new to summarize.
_empty: list = []


def _spy_empty(coro, label=None):
    _empty.append(label)
    coro.close()
    return True


main._fire_and_forget = _spy_empty
try:
    _d = main._run_memory_tail(
        CONV, "", finished=True, truncated=False, holed=False,
        touched_facts=[], last_user_text="And the tea?", turn_index=2,
        messages=list(HISTORY), injected_facts=None,
    )
finally:
    main._fire_and_forget = _real_fire
check(_d.raw_chars == 0,
      "(the empty reply really does report 0 raw chars)")
check(_empty == [],
      "nothing is scheduled when the model produced no reply at all")

print("[6] the REAL predicate, not a lambda, closes the gate")
# [3] proves the gate calls something; it does not prove the something is
# right. This drives _is_repeat_task_traffic itself, under the state it was
# written for: a stable conv_id past TASK_TRAFFIC_MIN_POSITION receiving a
# request with no assistant turn - OpenWebUI asking for a title. That array
# is NOT the conversation, and rolling it up is what review showed could
# burn 21 real turns.
TASK_CONV = "rollup_task_traffic"
_st = summarizer.load_state(TASK_CONV)
_st["turns_seen"] = main.TASK_TRAFFIC_MIN_POSITION + 2
summarizer.save_state(TASK_CONV, _st)
TASK_MESSAGES = [{"role": "user", "content": "Generate a title."}]
check(main._is_repeat_task_traffic(TASK_CONV, TASK_MESSAGES),
      "the real predicate calls this task traffic — without it [6] proves "
      "nothing")

_task_labels: list = []


def _spy_task(coro, label=None):
    _task_labels.append(label)
    coro.close()
    return True


main._fire_and_forget = _spy_task
try:
    main._run_memory_tail(
        TASK_CONV, DEGENERATE, finished=True, truncated=False, holed=False,
        touched_facts=[], last_user_text="Generate a title.", turn_index=2,
        messages=list(TASK_MESSAGES), injected_facts=None,
    )
finally:
    main._fire_and_forget = _real_fire
check(_task_labels == [],
      "no rollup is scheduled for it, through the production predicate "
      "rather than a stand-in")

print("[7] a first turn with NO history does not roll up")
# Found by the R8 integration tests, not by this file, which is the point:
# they post a SINGLE user message and assert a skipped tail leaves the
# store untouched. The rollup fired anyway and wrote turns_seen=1 - it
# cannot build a chunk from one message, so the write was cost with no
# benefit, and it created summary state for a conversation that stored
# nothing (adversarial F5).
FIRST = [{"role": "user", "content": "hello for the very first time"}]
_first: list = []


def _spy_first(coro, label=None):
    _first.append(label)
    coro.close()
    return True


main._fire_and_forget = _spy_first
try:
    main._run_memory_tail(
        "first_turn_conv", DEGENERATE, finished=True, truncated=False,
        holed=False, touched_facts=[], last_user_text="hello", turn_index=1,
        messages=list(FIRST), injected_facts=None,
    )
finally:
    main._fire_and_forget = _real_fire
check(_first == [],
      "nothing is scheduled when the array carries no prior assistant turn "
      "- there is no earlier exchange to summarize")

# THE CONTROL, so [7] cannot pass by the gate refusing everything. The same
# degenerate reply, with one completed exchange behind it, still rolls up -
# which is the 14-consecutive-skips case this feature exists for.
WITH_HIST = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "an earlier answer"},
    {"role": "user", "content": "and now"},
]
_hist: list = []


def _spy_hist(coro, label=None):
    _hist.append(label)
    coro.close()
    return True


main._fire_and_forget = _spy_hist
try:
    main._run_memory_tail(
        "hist_conv", DEGENERATE, finished=True, truncated=False,
        holed=False, touched_facts=[], last_user_text="and now", turn_index=3,
        messages=list(WITH_HIST), injected_facts=None,
    )
finally:
    main._fire_and_forget = _real_fire
check(_hist != [],
      "but a looping reply WITH a history behind it still rolls up - the "
      "guard narrowed the case, it did not remove the feature")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll rollup-on-skip checks passed.")
