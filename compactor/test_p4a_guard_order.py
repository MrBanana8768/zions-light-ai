"""The hard-budget guard on a compacted array: persona and pinned facts are
not spent on turns that are dropped anyway.

hostile pass #4, reviewer A F5. Hostile pass #3's F5 made the guard spend
injected memory (persona, pinned facts, retrieval) BEFORE any turn whenever
compaction's stand-in is in the array, because the turns such an array keeps
are the ones no summary covers: her last message and the reply she is
answering. True in steady reuse. False in the cap-refusal state — a stand-in
present AND summarize() refusing the fresh span over the per-request call
cap, which hands every refreshed or uncovered turn back verbatim. Those are
tens to thousands of old turns, far more than all injected memory. Measured
at the shipped limit (20,768): the guard halved and dropped the whole memory
block, then dropped 100 old turns anyway; the pre-F5 order dropped 104 and
kept persona and facts.

Both states, through the REAL compact_if_needed and the order chat_completions
uses (compact -> inject one memory block -> guard with protect_system=1):

  [1] cap refusal: persona and facts survive, the old deferred turns go, the
      newest turn and the stand-in stay.
  [2] steady reuse: the recent turns survive and memory is what is spent
      (the F5 order, unchanged).

No LLM is reachable: /tokenize answers None (the char/4 counter), the real
summarize() refuses over the cap before any call, and in [2] summarize() is a
stub. Synthetic text only.

    python test_p4a_guard_order.py
"""

import asyncio
import os
import random
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ["VLLM_URL"] = "http://127.0.0.1:9"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="p4a-guard-")
os.environ["MAX_MODEL_LEN"] = "37152"

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

main.count_tokens_exact = lambda ms: None

FAILED: list[str] = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


W = ["river", "lantern", "quiet", "morning", "garden", "letter", "stone", "window",
     "silver", "harbor", "meadow", "candle", "thread", "bridge", "winter", "orchard"]
LIMIT = 20768                      # MAX_MODEL_LEN 37152 - reserve 16384, as shipped


def body(seed, chars):
    r = random.Random(str(seed))
    out, n = [], 0
    while n < chars:
        w = r.choice(W)
        out.append(w)
        n += len(w) + 1
    return " ".join(out) + "."


def history(exchanges, reply_chars=6600, user_chars=800):
    msgs = [{"role": "system", "content": "You are her companion."}]
    for k in range(exchanges):
        msgs.append({"role": "user", "content": f"U{k} " + body(("u", k), user_chars)})
        msgs.append({"role": "assistant", "content": f"A{k} " + body(("a", k), reply_chars)})
    msgs.append({"role": "user", "content": "NEWEST " + body("newest", 400)})
    return msgs


MEMORY = ("PERSONA-MARKER " + body("persona", 2400) + "\n\n"
          + "PINNED-FACTS-MARKER " + body("facts", 6000) + "\n\n"
          + "RETRIEVAL-MARKER " + body("retr", 5000))


def seed_hierarchy(conv, msgs, covered_exchanges):
    ns = [m for m in msgs if m["role"] != "system"]
    st = summarizer.load_state(conv)
    n_cov = covered_exchanges * 2
    st["l1"] = [{"text": "STORED-SCENES " + body("s", 4000), "first_turn": 1, "last_turn": n_cov}]
    st["last_summarized_turn"] = n_cov
    st["turns_seen"] = len(ns) - 1
    summarizer._record_chunk_fps(st, 1, n_cov, ns[:n_cov])
    summarizer.save_state(conv, st)


def guard(out, limit):
    payload = main.inject_system_block(out, MEMORY)
    rep: dict = {}
    g = main._enforce_hard_budget(payload, limit, 1, rep)
    text = "\n".join(summarizer._message_text(m) for m in g)
    return g, rep, text


async def refusal():
    print("[1] CAP REFUSAL: a stand-in beside 120 deferred turns")
    conv = "guard_refusal"
    msgs = history(80)
    seed_hierarchy(conv, msgs, 20)
    before = main.compaction_counters().get("cap_refused", 0)
    out_n: list = []
    out = await main.compact_if_needed(list(msgs), conv, stored_turns_out=out_n)
    n_turns = sum(1 for m in out if m["role"] != "system")
    check(out_n == [40] and any(main._is_compaction_standin(m) for m in out)
          and main.compaction_counters().get("cap_refused", 0) == before + 1 and n_turns > 100,
          f"fixture: reuse held the stand-in (stored {out_n}) AND summarize() refused the rest "
          f"over the cap ({n_turns} turns left in the array)")
    g, rep, text = guard(out, LIMIT)
    print(f"       guard: {rep}")
    check(rep.get("fits") is True, "the guard fits the payload")
    check("PERSONA-MARKER" in text and "PINNED-FACTS-MARKER" in text,
          "*** F5: persona and pinned facts are forwarded — not spent on turns that were "
          "going to be dropped anyway")
    check((rep.get("dropped_turns") or 0) >= 90,
          f"and the old deferred turns are what went ({rep.get('dropped_turns')} dropped)")
    check("STORED-SCENES" in text and "NEWEST" in text and "A79 " in text and "U79 " in text,
          "and the stand-in, her newest message, her previous one and the reply she is "
          "answering all stay")


async def steady():
    print("[2] STEADY REUSE: nothing deferred, memory is spent before the recent turns")
    conv = "guard_steady"
    msgs = history(80)
    seed_hierarchy(conv, msgs, 76)

    async def fake_summarize(client, turns):
        return "FRESH-SUMMARY " + body("fresh", 1200), []

    real = main.summarize
    main.summarize = fake_summarize
    try:
        out_n: list = []
        out = await main.compact_if_needed(list(msgs), conv, stored_turns_out=out_n)
    finally:
        main.summarize = real
    array_tokens = main.count_tokens(out)
    memory_tokens = main.count_tokens([{"role": "system", "content": MEMORY}])
    limit = array_tokens + memory_tokens // 2
    check(out_n == [152] and not any(("U10 " in summarizer._message_text(m)) for m in out),
          f"fixture: steady reuse ({out_n} turns off the shelf, nothing deferred)")
    g, rep, text = guard(out, limit)
    print(f"       guard at limit {limit}: {rep}")
    check(rep.get("fits") is True and (rep.get("dropped_turns") or 0) == 0,
          f"CONTROL (F5 order): no turn is dropped ({rep.get('dropped_turns')})")
    check("A79 " in text and "U79 " in text and "NEWEST" in text and "STORED-SCENES" in text,
          "and her previous message, the reply she is answering and the stand-in stay")
    check((rep.get("trimmed_blocks") or 0) + (rep.get("dropped_blocks") or 0) >= 1
          and "PERSONA-MARKER" in text,
          "memory is what was spent (trimmed from the end), persona first in line kept")


try:
    asyncio.run(refusal())
    asyncio.run(steady())
except Exception:
    import traceback
    traceback.print_exc()
    FAILED.append("crashed")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll guard-order checks passed.")
