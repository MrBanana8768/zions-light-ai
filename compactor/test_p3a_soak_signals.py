"""The soak's reuse oracle reads the REAL signals compaction emits.

hostile pass #3, reviewer E F2. test_soak_conversation.py decides whether a
turn "was refused over the per-request cap" and how many turns it "reused".
The refusal half matched one phrase of one WARNING; no real run had ever
produced that phrase in the soak, and rewording the warning passed the
2026-09-12 regression (reuse_mutate's mRawRollup) green. Its self-test only
fed hand-built rows, so it proved the oracle and not the wiring.

This file takes the soak's own parser (`_turn_signals`, lifted out of
test_soak_conversation.py by AST so the soak is not executed) and feeds it
what the REAL code emits: a real summarize() cap refusal, a real reusing
compaction's log line, a real ordinary compaction. Reword the log line the
parser reads, or stop counting refusals, and this goes red in the unit suite.

    python test_p3a_soak_signals.py
"""

import ast
import asyncio
import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="p3a-signals-")
os.environ["COMPACTOR_TARGET_TOKENS"] = "1000"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "64"

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


def _lift(path: Path, names: set[str]) -> dict:
    """Execute only the named top-level defs/assignments of `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets):
            keep.append(node)
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    missing = names - set(ns)
    if missing:
        raise SystemExit(f"FAIL the soak no longer defines {sorted(missing)}")
    return ns


SOAK = _lift(Path(__file__).with_name("test_soak_conversation.py"),
             {"_turn_signals", "_REUSED_RE", "_COMPACTED_NEEDLE"})
signals = SOAK["_turn_signals"]


@contextlib.contextmanager
def logs():
    lines: list[str] = []

    class H(logging.Handler):
        def emit(self, r):
            lines.append(r.getMessage())

    lg = logging.getLogger("compactor")
    h = H(level=logging.DEBUG)
    prev = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield lines
    finally:
        lg.removeHandler(h)
        lg.setLevel(prev)


def history(n, tag=""):
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n):
        out.append({"role": "user", "content": f"{tag}question {i} " + "word " * 60})
        out.append({"role": "assistant", "content": f"{tag}answer {i} " + "word " * 60})
    return out


def turn_signals_for(msgs, conv, **kw):
    before = main.compaction_counters()
    with logs() as lines:
        asyncio.run(main.compact_if_needed(list(msgs), conv, **kw))
    after = main.compaction_counters()
    return signals("\n".join(lines), before, after), lines, before, after


print("[1] a REAL cap refusal reads as refused, from the counter")
_real_once = main._summarize_once
_real_cap = main.MAX_SUMMARY_CALLS_PER_REQUEST
_real_exact = main.count_tokens_exact


async def _never(client, turns):
    raise AssertionError("a refused request must not call the model")


main._summarize_once = _never
main.MAX_SUMMARY_CALLS_PER_REQUEST = 1
main.count_tokens_exact = lambda ms: None
try:
    # Batches are sized to MAX_MODEL_LEN - reserves, so make the span wider
    # than one batch at the pessimistic 2.0 scale.
    big = history(400)
    sig, lines, before, after = turn_signals_for(big, None)
finally:
    main._summarize_once = _real_once
    main.MAX_SUMMARY_CALLS_PER_REQUEST = _real_cap
    main.count_tokens_exact = _real_exact
check(after["cap_refused"] == before["cap_refused"] + 1,
      f"fixture: summarize() refused once over the cap (counter "
      f"{before['cap_refused']} -> {after['cap_refused']})")
check(any("compaction skipped" in l for l in lines),
      "fixture: and logged its refusal")
check(sig["cap_refused"] is True,
      f"*** E-F2: the soak's parser reads that turn as cap-refused (got {sig})")
check(sig["compacted"] is True,
      "and as compacted: the refused turn still logs its compaction line")

print("[2] a REAL reusing compaction reads as reused, with its count")
_real_summarize = main.summarize


async def _fresh(client, turns):
    return "FRESH", []


main.summarize = _fresh
try:
    CONV = "signals_reuse"
    MSGS = history(24)
    st = summarizer.load_state(CONV)
    st["l1"] = [{"text": "SIGNAL-CHUNK", "first_turn": 1, "last_turn": 40}]
    st["last_summarized_turn"] = 40
    summarizer._record_chunk_fps(st, 1, 40, [m for m in MSGS if m["role"] != "system"][:40])
    summarizer.save_state(CONV, st)
    out_n: list = []
    sig, lines, before, after = turn_signals_for(MSGS, CONV, stored_turns_out=out_n)
    check(out_n == [40], f"fixture: the real gate reused 40 turns (got {out_n})")
    check(sig["reused"] == 40 and sig["compacted"] is True,
          f"*** the soak's parser reads the real log line as 40 reused (got {sig}; "
          f"lines: {[l for l in lines if 'compacted:' in l]})")
    check(sig["cap_refused"] is False,
          "CONTROL: an ordinary compaction is not a refusal")

    print("[3] CONTROL: a compaction that reused nothing reads as zero")
    sig, lines, before, after = turn_signals_for(history(24, tag="N-"), "signals_none")
    check(sig == {"compacted": True, "cap_refused": False, "reused": 0},
          f"compacted, not refused, 0 reused (got {sig})")
finally:
    main.summarize = _real_summarize

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll soak-signal checks passed.")
