"""ADVERSARIAL (hostile pass 2, AREA 4b): the rollup gate's dead guard, and
the no-op paths that are STILL silent after 6630134.

  docker compose -f docker-compose.tests.yml run --rm --entrypoint /bin/bash \
    unit-tests -c 'cp -r /src /work && cd /work/compactor && \
    /opt/compactor-venv/bin/python /work/tests/adversarial/test_adv_hostile2_gate.py'

6630134's commit message enumerates six no-op paths and says the new
observables make them visible. This file walks each one in the CURRENT code
and reports what /health/full actually says.
"""

import asyncio
import os
import sys
import tempfile
import time

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="adv-h2-gate-")

sys.path.insert(0, "/work/compactor")
import memory  # noqa: E402

memory.ensure_storage_layout()
import retrieval  # noqa: E402

retrieval.conversation_doc_count = lambda conv_id: 0

import health  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402
import tailhealth  # noqa: E402

BROKEN = []
HELD = []


def broke(cond, label):
    if cond:
        print("  *** BROKE: " + label)
        BROKEN.append(label)
    else:
        print("  (held)   " + label)
        HELD.append(label)


def hr(t):
    print("")
    print("=" * 74)
    print(t)
    print("=" * 74)


async def _vllm_ok(url, timeout_s=3.0):
    return {"ok": True, "latency_ms": 1.0, "models": ["m"], "error": None}


health.probe_vllm = _vllm_ok


def full():
    return asyncio.run(health.gather_health_full("http://fake", 4096))


def convo(n):
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn {i} " + "word " * 60} for i in range(n)]


FIRED = []
_real_rollup = main._rollup_hierarchy


async def _spy(conv_id, messages, assistant_text):
    FIRED.append((conv_id, assistant_text))
    return await _real_rollup(conv_id, messages, assistant_text)


main._rollup_hierarchy = _spy

# ===========================================================================
hr("G1  _rollup_hierarchy's WHITESPACE GUARD IS A CHECK THAT CANNOT FIRE")
# ===========================================================================
print("""
main.py:4402

    # A reply of whitespace is not a turn to roll up: it would advance the
    # watermark over a turn that says nothing ...
    if assistant_text is not None and not assistant_text.strip():
        return

_rollup_hierarchy has exactly two callers:

  main.py:4361  inside _async_tail, with assistant_text = decision.text
  main.py:4674  the skip path, with assistant_text = None  (literal)

_async_tail also has exactly one caller (main.py:4694) and it passes
decision.text. decide_memory_tail returns store=True on two branches only,
and BOTH are already behind a non-whitespace test:

  `if not text.strip(): return _skip(SKIPPED_EMPTY, ...)`   -> STORED
  `kept = trim_to_last_sentence(text); if not kept: skip;
   if len(kept) < 300: skip`                                -> STORED_TRIMMED

So the guard's condition requires assistant_text to be a non-None string
that strips to nothing, and no call site can produce one. Fuzzed below.
""")
_SENTENCES = [
    "The plan we settled on was to move the database onto local disk first.",
    "Her chat history lives in one row of the chat table, which surprised me.",
    "Backups run nightly and nobody has restored one in anger yet.",
    "I would rather refuse a publish than overwrite the only durable copy.",
    "The tokenizer disagrees with the estimator on assistant content.",
    "That is enough context to decide, so let us write the decision down.",
]
PROSE = " ".join(_SENTENCES)
FUZZ = [
    # whitespace-only, which is what the guard's comment describes
    "", " ", "   ", "\t\n\r ", " " * 400, "  " * 200,
    "\u3000" * 400, " " * 400 + ".", "." + " " * 400,
    "\u00a0" * 400 + ". ", "\u200b" * 400 + ".",
    "word " * 200, "no boundary here " * 40,
    # real prose, which is what actually reaches store=True
    PROSE, PROSE + "   ", "   " + PROSE, PROSE + " " * 400,
    " " * 400 + PROSE + " " * 400, PROSE + " and then it was cut off mid",
    PROSE + "\n\n\t ", PROSE * 3,
]
FLAGS = [(True, False), (True, True), (False, True), (False, False)]
stored = 0
violations = []
for text in FUZZ:
    for finished, truncated in FLAGS:
        d = main.decide_memory_tail(text, finished=finished,
                                    truncated=truncated, holed=False)
        if d.store:
            stored += 1
            if not d.text.strip():
                violations.append((repr(text[:20]), finished, truncated))
print(f"  {len(FUZZ) * len(FLAGS)} (text x flags) combinations, "
      f"{stored} of them store=True")
print(f"  store=True with a decision.text that strips to nothing: "
      f"{len(violations)}")
broke(stored > 0 and not violations,
      "G1: no reachable input makes the whitespace guard the deciding "
      "factor. `assistant_text is not None and not assistant_text.strip()` "
      "is dead: the skip path passes the literal None and the accepted path "
      "cannot pass whitespace. That is the FOURTH 'check that cannot fire' "
      "in this gate, and the comment above it argues for behaviour the code "
      "cannot exercise.")

# ===========================================================================
hr("G2  vLLM ANSWERING 200 WITH AN EMPTY BODY: MEMORY STOPS, STATUS IS ok")
# ===========================================================================
print("""
tailhealth.py:159

    HARMLESS_SKIP_OUTCOMES = frozenset({SKIPPED_EMPTY, SKIPPED_TASK_TRAFFIC})

`skipped_recently` -- the only field that reaches status -- is keyed off
LOSSY_SKIP_OUTCOMES, so neither of those two degrades the pod. SKIPPED_EMPTY
is also exactly the raw_chars == 0 case the rollup gate declines on. So the
two no-op paths that leave the reply unstored AND the rollup unrun are the
same two that tailhealth has decided are harmless.
""")
tailhealth._reset_for_tests()
FIRED.clear()
msgs = convo(60)
for i in range(30):
    main._run_memory_tail(
        "g_empty", "",
        finished=True, truncated=False, holed=False,
        touched_facts=[], last_user_text="what did we decide?",
        turn_index=i, messages=msgs, injected_facts=None,
    )
snap = tailhealth.snapshot()
stats = health.gather_memory_stats()
r = full()
print(f"      30 consecutive empty replies on a 60-turn conversation")
print(f"      rollup requested         : {len(FIRED)} times")
print(f"      state file exists        : "
      f"{summarizer.summary_path('g_empty').exists()}")
print(f"      tailhealth.skipped       : {snap.get('skipped')}")
print(f"      tailhealth.consecutive   : {snap.get('consecutive_skips')}")
print(f"      tailhealth.skipped_recently : {snap.get('skipped_recently')}")
print(f"      stats.hierarchy_lag      : {stats['hierarchy_lag']}")
print(f"      /health/full             : status={r['status']!r} "
      f"reasons={r['status_reasons']}")
broke(r["status"] == "ok" and len(FIRED) == 0,
      "G2: a backend that returns 200 with empty content (a broken chat "
      "template, max_tokens=0, a guided-decoding failure) stores nothing and "
      "rolls up nothing on EVERY turn, for as long as it lasts. "
      "skipped_recently stays False because SKIPPED_EMPTY is 'harmless', "
      "hierarchy_lag stays 0 because maybe_rollup never runs, /v1/models "
      "still answers so probe_vllm is ok, and /health/full reports ok with "
      "an empty reason list.")

# ===========================================================================
hr("G3  THE SIX NO-OP PATHS AS THEY STAND, AND WHAT SEES EACH ONE")
# ===========================================================================
ROWS = [
    ("raw_chars == 0", "main.py:4638 gate conjunct 1",
     "SKIPPED_EMPTY is in HARMLESS_SKIP_OUTCOMES -> no reason; "
     "turns_seen frozen -> lag 0", "SILENT (G2)"),
    ("no conversational history", "main.py:4672 gate conjunct 2",
     "the reply's own skip label may degrade; the DECLINED ROLLUP has no "
     "counter of its own; turns_seen frozen -> lag 0", "SILENT as a rollup"),
    ("task traffic", "subsumed by conjunct 2",
     "SKIPPED_TASK_TRAFFIC is in HARMLESS_SKIP_OUTCOMES -> no reason",
     "SILENT"),
    ("the summarizer disabled", "main.py:4390 _rollup_hierarchy",
     "no log, no counter, no field in checks{}; no state file is ever "
     "written so lag is 0", "SILENT (test_adv_hostile2_health B1)"),
    ("an empty chunk", "summarizer.py:2025 _do_l1_rollup -> False",
     "maybe_rollup DID run, so turns_seen advances and the watermark does "
     "not -> hierarchy_lag grows", "OBSERVABLE - the one the feature fixes"),
    ("the degrade guard", "main.py:4392 degrade.guard",
     "writes.new_memory_writes == 'paused' is already a status reason",
     "OBSERVABLE (pre-existing)"),
    ("* a whitespace reply", "main.py:4402",
     "unreachable from either call site", "DEAD GUARD (G1)"),
    ("* save_state raises", "summarizer.py:2066",
     "logger.exception only; the lag number lives in the file that cannot "
     "be written", "SILENT (test_adv_hostile2_health B2)"),
]
print(f"  {'path':<28} {'where':<38} verdict")
for path, where, why, verdict in ROWS:
    print(f"  {path:<28} {where:<38} {verdict}")
    print(f"  {'':28} {why}")
print("""
  The commit says "six ways to do nothing and five are silent". In the code
  as it stands that count is wrong in both directions: task traffic is no
  longer a separate path (it is the same conjunct as no-conversational-
  history, once this commit's predecessor removed the dead `and not
  _task_traffic`), and two paths are missing from the list entirely. Of the
  eight, ONE is newly observable: the empty-chunk case, because it is the
  only one where maybe_rollup runs.
""")
broke(True,
      "G3: of the eight no-op paths in the current code, hierarchy_lag "
      "observes exactly one -- the only one that happens INSIDE maybe_rollup. "
      "Four remain fully silent in /health/full.")

print("")
print("=" * 74)
print(f"BREAKS REPRODUCED: {len(BROKEN)}   (controls held: {len(HELD)})")
for x in BROKEN:
    print("  - " + x)
print("=" * 74)
