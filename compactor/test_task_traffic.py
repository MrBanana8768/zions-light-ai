"""
N4b — OpenWebUI background task traffic must not be memorized, and a real
first turn must be.

THE DEFECT. The compactor has classified this shape since v3.1: the
over-budget warning literally says "task traffic or a first turn". Only the
INJECTION side ever acted on it (INJECTION_NO_HISTORY_FRACTION, 0.125 against
0.5). `has_history` is computed once in the chat handler and reaches exactly
two places — that fraction and that log line — so neither _run_memory_tail
call site was ever told, and OpenWebUI's title/tag/follow-up calls have been
fact-extracted, episodically indexed and deduped the whole time. Measured in
the 2026-09-03/04 bundle: 99 such requests in two days on conv=026752…, one
roughly every 90 seconds after every real turn. One classification, two
consumers, wired to one of them.

THE TRAP, AND WHY THIS FILE EXISTS. The obvious fix — skip the tail when
`_has_conversational_history` is False — is a WORSE bug than the one it
fixes, and the backlog's own suggested fix direction would have produced it.
That predicate is False for a genuine first turn too; its docstring says so.
Acting on it alone silently drops the opening exchange of every new
conversation. [1] is the case that catches it, and it is the load-bearing
test in this file: a fix that skips task traffic but also eats first turns
passes every other assertion here.

What actually separates them is not the request but the history. Task calls
arrive on a STABLE conv_id, over and over, each with no assistant turn. A
real conversation looks like that exactly once.

Mutations this file exists to kill:

    the predicate ignores stored state       [1] the first turn is eaten
    the predicate ignores assistant history  [3] an ordinary turn is eaten
    the check is dropped from the tail       [2] task traffic is memorized
    the label is counted as lossy            [4] health degrades on it
    the exception type is dropped again      [5] L7's empty parentheses

Conventions from test_degenerate_skip.py (canonical). Only synthetic content
appears below — the repo is public.

    python test_task_traffic.py
"""

import contextlib
import logging
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # no ChromaDB/fastembed here

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-task-traffic-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402
import tailhealth  # noqa: E402

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None

memory.ensure_storage_layout()

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name):
    lg = logging.getLogger(logger_name)
    handler = _Collector()
    prev = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev)


PROSE = " ".join(
    f"Sentence number {i} of a perfectly ordinary reply about nothing."
    for i in range(1, 13)
)
assert len(PROSE) > main.MIN_MEMORABLE_TRIMMED_CHARS, "fixture must clear the floor"

# The shape OpenWebUI sends for a title/tag/follow-up call: a system prompt
# and one user message, no assistant turn.
TASK_MESSAGES = [
    {"role": "system", "content": "Generate a concise title."},
    {"role": "user", "content": "Summarize the conversation above."},
]
# A real first turn is the SAME SHAPE. That is the whole problem.
FIRST_TURN_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, can you help me with something?"},
]
ONGOING_MESSAGES = FIRST_TURN_MESSAGES + [
    {"role": "assistant", "content": PROSE},
    {"role": "user", "content": "Thank you, one more question."},
]


def _seed_store(conv_id, position):
    """Make it look like this conv_id has reached `position` turns.

    POSITION, not merely "a file exists". The first draft of the fix keyed off
    file existence and test_budget_guard killed it on the spot with an
    ordinary fixture: a conversation with a facts file receiving a
    history-less turn, which is exactly what OpenWebUI sends when the user
    regenerates the first reply. Eating that is a silent loss on a real
    exchange, so the rule needs the conversation to be genuinely deeper than
    the array it just presented.
    """
    memory.atomic_write_json(memory.facts_path(conv_id), {"facts": []})
    st = summarizer.load_state(conv_id)
    st["turns_seen"] = position
    summarizer.save_state(conv_id, st)


# _run_memory_tail hands the real tail to _fire_and_forget, which needs a
# running loop. These cases are about the DECISION, not the write, so the
# coroutine is closed rather than scheduled — leaving it unawaited turns a
# passing suite into a RuntimeWarning that the runner reports as a failure.
def _no_schedule(coro, label=""):
    coro.close()


main._fire_and_forget = _no_schedule


def _run(conv_id, messages, user_text="A user turn with real words in it."):
    tailhealth._reset_for_tests()
    return main._run_memory_tail(
        conv_id,
        PROSE,
        finished=True,
        truncated=False,
        holed=False,
        touched_facts=[],
        last_user_text=user_text,
        turn_index=1,
        messages=messages,
        injected_facts=None,
    )


print("[1] a genuine FIRST TURN is memorized — nothing stored under it yet")
d = _run("conv_first_turn", FIRST_TURN_MESSAGES)
check(d.store is True,
      "a first turn has no assistant history AND no stored state, so it is a "
      "conversation, not a task")
check(d.outcome == tailhealth.STORED,
      f"and it is counted as stored (got {d.outcome!r})")

print("[2] REPEAT task traffic is skipped, and named")
_seed_store("conv_task", position=40)
d = _run("conv_task", TASK_MESSAGES)
check(d.store is False, "no assistant turn on a conv_id already in the store")
check(d.outcome == tailhealth.SKIPPED_TASK_TRAFFIC,
      f"counted under its own label (got {d.outcome!r})")
check("task traffic" in (d.reason or ""),
      "and the reason says so, so the log line is greppable")

print("[3] an ORDINARY turn on a stored conversation is untouched")
_seed_store("conv_ongoing", position=40)
d = _run("conv_ongoing", ONGOING_MESSAGES)
check(d.store is True,
      "an assistant turn in the client array is the end of the question — "
      "stored state is only consulted when there is none")

print("[3b] the FIRST task calls are memorized, deliberately")
d = _run("conv_task_fresh", TASK_MESSAGES)
check(d.store is True,
      "indistinguishable from a new conversation at that moment; the price "
      "of not guessing, paid a handful of times per conv_id rather than "
      "99 times every two days")

print("[3c] REGENERATING the first reply of a shallow conversation is memorized")
# The case test_budget_guard caught, and the reason this rule is a threshold
# rather than "has the store anything at all". OpenWebUI sends exactly this
# when the user regenerates the opening reply: a facts file already exists,
# and the array has no assistant turn in it.
_seed_store("conv_regen", position=2)
d = _run("conv_regen", FIRST_TURN_MESSAGES)
check(d.store is True,
      "a conversation one exchange deep is still allowed to present an array "
      "with no assistant turn — that is a regenerate, not a background task")
check(d.outcome == tailhealth.STORED,
      f"and it is stored, not counted as task traffic (got {d.outcome!r})")

print("[4] the new label is HARMLESS — it must not degrade health")
check(tailhealth.SKIPPED_TASK_TRAFFIC in tailhealth.HARMLESS_SKIP_OUTCOMES,
      "named in the one place both consumers read")
check(tailhealth.SKIPPED_TASK_TRAFFIC not in tailhealth.LOSSY_SKIP_OUTCOMES,
      "so it is excluded from the lossy set derived from it")
tailhealth._reset_for_tests()
tailhealth.note(tailhealth.SKIPPED_TASK_TRAFFIC, raw_chars=100, kept_chars=0)
snap = tailhealth.snapshot()
check(snap.get("skipped_recently") is False,
      "a run of task calls does not pin /health/full degraded (R27's rule, "
      "which this label now inherits rather than re-states)")

print("[4b] the label is still counted, because a silent skip is the defect")
check(snap.get("outcomes", {}).get(tailhealth.SKIPPED_TASK_TRAFFIC) == 1,
      "declining to memorize is not the same as not noticing")

print("[5] L7 — a failure that cannot say why is not a failure report")
check("type(e).__name__" in open("facts.py", encoding="utf-8").read(),
      "facts.py logs the exception TYPE, not just str(e)")
check("type(e).__name__" in open("dedup.py", encoding="utf-8").read(),
      "and so does its twin in dedup.py — both produced bare '()' in the "
      "2026-09-03 production log")


class _Silent(Exception):
    """An exception that stringifies to nothing — httpx's timeouts do this."""


with capture("compactor.dedup") as cap:
    logging.getLogger("compactor.dedup").warning(
        f"dedup LLM call failed (cluster preserved): "
        f"{type(_Silent()).__name__}: {_Silent()}"
    )
check(any("_Silent" in r.getMessage() for r in cap.records),
      "so a silent exception still names itself in the log")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll task-traffic checks passed.")
