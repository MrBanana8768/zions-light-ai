"""ADVERSARIAL (hostile pass 2, AREA 4): the health observables added by
6630134, and the rollup gate they are supposed to observe.

  docker compose -f docker-compose.tests.yml run --rm --entrypoint /bin/bash \
    unit-tests -c 'cp -r /src /work && cd /work/compactor && \
    /opt/compactor-venv/bin/python /work/tests/adversarial/test_adv_hostile2_health.py'

6630134 claims: "v3.1.8's skip-path rollup has six ways to do nothing and
five are silent ... It is reported as hierarchy_lag and, past the threshold,
as a status reason". The central question this file asks is WHO WRITES THE
OBSERVABLE, because hierarchy_lag is turns_seen - last_summarized_turn and
both of those numbers are written by maybe_rollup -- the function whose
absence the signal exists to report.
"""

import asyncio
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="adv-h2-health-")
os.environ["COMPACTOR_TARGET_TOKENS"] = "500"

sys.path.insert(0, "/work/compactor")
import memory  # noqa: E402

memory.ensure_storage_layout()
import retrieval  # noqa: E402

retrieval.conversation_doc_count = lambda conv_id: 0

import bgwork  # noqa: E402
import degrade  # noqa: E402
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


def hr(title):
    print("")
    print("=" * 74)
    print(title)
    print("=" * 74)


async def _vllm_ok(url, timeout_s=3.0):
    return {"ok": True, "latency_ms": 1.0, "models": ["m"], "error": None}


health.probe_vllm = _vllm_ok


def full():
    """/health/full as the operator sees it, with only vLLM stubbed."""
    return asyncio.run(health.gather_health_full("http://fake", 4096))


def wipe_store():
    root = memory.storage_root()
    for sub in ("facts", "summaries"):
        d = root / sub
        if d.exists():
            for f in d.glob("*.json"):
                f.unlink()


def put_state(cid, turns_seen, last_summarized, age_days=0.0):
    st = summarizer._empty_state(cid)
    st["turns_seen"] = turns_seen
    st["last_summarized_turn"] = last_summarized
    if last_summarized:
        st["l1"] = [{"text": "a scene", "first_turn": 1,
                     "last_turn": last_summarized}]
    summarizer.save_state(cid, st)
    if age_days:
        p = summarizer.summary_path(cid)
        t = time.time() - age_days * 86400
        os.utime(p, (t, t))


def convo(n_turns, with_assistant=True):
    """n_turns non-system messages, alternating user/assistant."""
    out = []
    for i in range(n_turns):
        role = "user" if (i % 2 == 0 or not with_assistant) else "assistant"
        out.append({"role": role, "content": f"turn {i} " + ("word " * 60)})
    return out


# ===========================================================================
hr("A  THE ROLLUP GATE: EVERY CONJUNCT, AND THE STATE THAT DECIDES IT")
# ===========================================================================
print("""
    if (
        decision.raw_chars > 0                       # A1
        and _has_conversational_history(messages)    # A2  <- 'DO NOT DELETE'
        and not _fire_and_forget(                    # A3
            _rollup_hierarchy(conv_id, messages, None), ...)
    ):

A3 is where the rollup is actually REQUESTED; A1 and A2 short-circuit before
the coroutine is ever constructed. Verified by spy, not by reading the
comment.
""")

FIRED = []
_real_rollup = main._rollup_hierarchy


async def _spy_rollup(conv_id, messages, assistant_text):
    FIRED.append(conv_id)
    return await _real_rollup(conv_id, messages, assistant_text)


main._rollup_hierarchy = _spy_rollup

NO_BOUNDARY = "x" * 600  # trims to nothing -> SKIPPED_NO_BOUNDARY, raw>0


async def run_tail(conv_id, text, messages, finished=False, truncated=True):
    d = main._run_memory_tail(
        conv_id, text,
        finished=finished, truncated=truncated, holed=False,
        touched_facts=[], last_user_text="what did we decide about the plan?",
        turn_index=1, messages=messages, injected_facts=None,
    )
    await asyncio.sleep(0.25)
    return d


CASES = [
    ("A1 deciding: raw_chars == 0, history present",
     "a_raw0", "", convo(6), False),
    ("A2 deciding: raw_chars > 0, NO assistant turn",
     "a_nohist", NO_BOUNDARY, convo(1, with_assistant=False), False),
    ("both pass: raw_chars > 0 and history present",
     "a_both", NO_BOUNDARY, convo(6), True),
]
for label, cid, text, msgs, expect_fired in CASES:
    FIRED.clear()
    d = asyncio.run(run_tail(cid, text, msgs))
    fired = bool(FIRED)
    print(f"  {label}")
    print(f"      outcome={d.outcome} raw_chars={d.raw_chars} store={d.store} "
          f"rollup_requested={fired}")
    broke(fired != expect_fired,
          f"A: {label} -> rollup_requested={fired}, expected {expect_fired}")

st = summarizer.summary_path("a_nohist")
print(f"\n  summary state file for the history-less skip: exists={st.exists()}")
broke(st.exists(),
      "A: R8 regression -- a history-less skipped turn wrote summary state")
print("""
  VERDICT: all three conjuncts have a state in which they are the deciding
  factor, so there is no dead conjunct in the gate as it stands. The
  'DO NOT DELETE' comment on A2 is correct and is now pinned by a test
  rather than by prose.

  The removed `and not _task_traffic` really was dead: _is_repeat_task_traffic
  opens with `if _has_conversational_history(messages): return False`, and A2
  is evaluated on the SAME array, so past A2 it is guaranteed False.
""")

# ===========================================================================
hr("B  hierarchy_lag IS WRITTEN BY THE FUNCTION WHOSE ABSENCE IT REPORTS")
# ===========================================================================
print("""
turns_seen is assigned in exactly one place reachable in production:
summarizer._observed_position, and _observed_position has exactly one caller,
summarizer.maybe_rollup:2004. last_summarized_turn is written only by the
_do_*_rollup functions, also inside maybe_rollup.

So hierarchy_lag can only become non-zero if maybe_rollup RAN. Every no-op
that happens BEFORE maybe_rollup -- which is five of the commit's own six --
leaves both numbers frozen, and a frozen pair reads as lag 0: caught up.
""")

wipe_store()
tailhealth._reset_for_tests()

print("B1  the master switch off: 22 accepted turns, nothing written, ok")
summarizer.ENABLED = False
msgs = convo(60)
for _ in range(22):
    asyncio.run(_real_rollup("b_disabled", msgs, "A complete reply."))
summarizer.ENABLED = True
stats = health.gather_memory_stats()
r = full()
print(f"      state file exists : {summarizer.summary_path('b_disabled').exists()}")
print(f"      hierarchy_lag     : {stats['hierarchy_lag']}")
print(f"      summaries_with_l1 : {stats['summaries_with_l1']}")
print(f"      /health/full      : status={r['status']!r} reasons={r['status_reasons']}")
broke(r["status"] == "ok" and stats["hierarchy_lag"] == 0,
      "B1: COMPACTOR_HIERARCHICAL_SUMMARY=false -> the whole hierarchy is "
      "dead, 22 turns produce no state at all, hierarchy_lag reads 0 and "
      "/health/full reports ok with an empty reason list. No field anywhere "
      "in the payload names the switch.")
print(f"      any 'summar' key in checks: "
      f"{[k for k in r['checks'] if 'summar' in k or 'hier' in k]}")

print("")
print("B2  every save_state fails: the conversation is 60 turns past the")
print("    watermark and hierarchy_lag reads 0")
wipe_store()
tailhealth._reset_for_tests()
_real_save = summarizer.save_state


def _boom(conv_id, state):
    raise OSError("simulated: the state write cannot land")


summarizer.save_state = _boom
for _ in range(22):
    asyncio.run(_real_rollup("b_nosave", msgs, "A complete reply."))
summarizer.save_state = _real_save
stats = health.gather_memory_stats()
r = full()
print(f"      state file exists : {summarizer.summary_path('b_nosave').exists()}")
print(f"      hierarchy_lag     : {stats['hierarchy_lag']}  "
      f"(the conversation is at turn {len([m for m in msgs])})")
print(f"      /health/full      : status={r['status']!r} reasons={r['status_reasons']}")
broke(r["status"] == "ok" and stats["hierarchy_lag"] == 0,
      "B2: the lag signal lives IN the file whose write is failing, so the "
      "single most common way the hierarchy freezes permanently is the one "
      "state it is structurally unable to report. 60 turns unsummarized, "
      "hierarchy_lag 0, /health/full ok.")

print("")
print("B3  CONTROL -- the one class it does catch: maybe_rollup ran, the LLM")
print("    call failed, so turns_seen advanced and the watermark did not")
wipe_store()
tailhealth._reset_for_tests()
asyncio.run(_real_rollup("b_llmdown", msgs, "A complete reply."))
disk = json.loads(summarizer.summary_path("b_llmdown").read_text())
stats = health.gather_memory_stats()
r = full()
print(f"      on disk           : turns_seen={disk.get('turns_seen')} "
      f"last_summarized_turn={disk.get('last_summarized_turn')}")
print(f"      hierarchy_lag     : {stats['hierarchy_lag']} "
      f"conv={stats['hierarchy_lag_conv']}")
print(f"      /health/full      : status={r['status']!r}")
for x in r["status_reasons"]:
    print(f"        - {x}")
broke(not any("summary hierarchy is" in x for x in r["status_reasons"]),
      "B3: the control failed -- an LLM-side rollup failure should raise the "
      "lag reason")

# ===========================================================================
hr("C  worst_lag HAS NO RECENCY FILTER, AND NAMES ONLY ONE CONVERSATION")
# ===========================================================================
print("""
memory.list_known_conv_ids() globs every *.json under facts/ and summaries/
with no mtime, no activity and no liveness filter. The live store was
measured at 62% test junk -- 90 of 145 namespaces. worst_lag is a single
global max over all of them, and the reason names that one conversation.
""")
wipe_store()
tailhealth._reset_for_tests()
put_state("junk_from_a_soak_2025", turns_seen=5000, last_summarized=0,
          age_days=400)
put_state("the_live_conversation", turns_seen=200, last_summarized=100)
stats = health.gather_memory_stats()
r = full()
print(f"      hierarchy_lag      : {stats['hierarchy_lag']}")
print(f"      hierarchy_lag_conv : {stats['hierarchy_lag_conv']}")
print(f"      state file mtime   : "
      f"{(time.time() - os.path.getmtime(summarizer.summary_path('junk_from_a_soak_2025'))) / 86400:.0f} "
      f"days old")
for x in r["status_reasons"]:
    print(f"        - {x}")
named_junk = stats["hierarchy_lag_conv"] == "junk_from_a_soak_2025"
mentions_live = any("the_live_conversation" in x for x in r["status_reasons"])
broke(named_junk and not mentions_live,
      "C: a state file 400 days old with lag 5000 pins hierarchy_lag "
      "permanently, the reason instructs the operator to drain a conversation "
      "that has no messages left to drain, and the genuinely lagging live "
      "conversation (lag 100, also past the limit) is never named. Permanent "
      "red is indistinguishable from ignored.")

# ===========================================================================
hr("D  _lag_limit = 2 * L1_CHUNK_SIZE, AND L1_CHUNK_SIZE IS AN ENV KNOB")
# ===========================================================================
print("""
summarizer.py:95  L1_CHUNK_SIZE = env_int("COMPACTOR_L1_CHUNK_SIZE", 20)
envcfg.env_int's documented contract: "env_int / env_float do NOT police
range. An explicit 0, a negative ... is returned exactly as given."
health.py:562     _lag_limit = 2 * summarizer.L1_CHUNK_SIZE
""")
_orig_chunk = summarizer.L1_CHUNK_SIZE
wipe_store()
tailhealth._reset_for_tests()
put_state("d_ordinary_drift", turns_seen=21, last_summarized=20)  # lag 1
for chunk in (20, 4, 1, 0):
    summarizer.L1_CHUNK_SIZE = chunk
    r = full()
    lagline = [x for x in r["status_reasons"] if "summary hierarchy" in x]
    print(f"      COMPACTOR_L1_CHUNK_SIZE={chunk:<4} limit={2 * chunk:<4} "
          f"status={r['status']:<9} lag_reason={bool(lagline)}")
    if lagline:
        print(f"          {lagline[0][:96]}")
summarizer.L1_CHUNK_SIZE = 0
r0 = full()
broke(r0["status"] == "degraded",
      "D1: COMPACTOR_L1_CHUNK_SIZE=0 makes the limit 0, so ONE turn of the "
      "ordinary tail drift the comment itself calls healthy degrades the pod "
      "on every probe, forever")

print("")
print("D2  a NEGATIVE chunk size degrades a pod with NO conversations at all")
wipe_store()
summarizer.L1_CHUNK_SIZE = -20
rneg = full()
print(f"      conversations={health.gather_memory_stats()['conversations']} "
      f"status={rneg['status']!r}")
for x in rneg["status_reasons"]:
    print(f"        - {x}")
broke(rneg["status"] == "degraded",
      "D2: worst_lag is initialised to 0 and the guard is `_lag > _lag_limit`, "
      "so a negative limit fires on an EMPTY store: the reason reads 'the "
      "summary hierarchy is 0 turns behind on conv=None'")
summarizer.L1_CHUNK_SIZE = _orig_chunk

# ===========================================================================
hr("E  probe_snapshot's interval COMES FROM env_float, WHICH POLICES NOTHING")
# ===========================================================================
print("""
health.py:301  interval = env_float("WEBUI_DB_SYNC_INTERVAL_S", 300.0)
health.py:303  "interval_s": round(interval),
health.py:319  out["stale"] = age > 3 * interval
""")
snap = os.path.join(tempfile.mkdtemp(prefix="adv-h2-snap-"), "webui.db")
with open(snap, "wb") as fh:
    fh.write(b"x")

for raw in ("300", "0", "-1", "inf", "1e400", "nan", "not-a-number"):
    os.environ["WEBUIDB_SYNC_ENABLED"] = "true"
    os.environ["WEBUI_SNAPSHOT_DB"] = snap
    os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = raw
    try:
        out = health.probe_snapshot()
        print(f"      {raw:<14} -> watched={out['watched']} "
              f"age_s={out['age_s']} stale={out['stale']} "
              f"interval_s={out['interval_s']}")
    except Exception as e:
        print(f"      {raw:<14} -> RAISED {type(e).__name__}: {e}")

os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "0"
broke(health.probe_snapshot()["stale"] is True,
      "E1: WEBUI_DB_SYNC_INTERVAL_S=0 makes the threshold 0, so a snapshot "
      "written one second ago is stale and /health/full is degraded forever")

print("")
print("E2  a non-finite interval kills the probe, and the error is NOT a reason")
os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "inf"
blocking = health._gather_blocking()
print(f"      _gather_blocking()['snapshot'] = {blocking['snapshot']}")
r = full()
print(f"      /health/full: status={r['status']!r} reasons={r['status_reasons']}")
print(f"      checks.snapshot = {r['checks'].get('snapshot')}")
broke(r["status"] == "ok" and blocking["snapshot"].get("error"),
      "E2: round(inf) raises OverflowError (round(nan) raises ValueError), "
      "past probe_snapshot's `except OSError`, into _gather_blocking's "
      "catch-all, which sets stale=False. gather_health_full reads only "
      "`_snap.get('stale')`, so the staleness probe is dead and /health/full "
      "says ok. Its siblings bg/mt BOTH raise an 'unobservable' reason for "
      "exactly this condition -- the doctrine is applied twice and missed "
      "here.")
os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "300"
os.environ["WEBUIDB_SYNC_ENABLED"] = "false"

# ===========================================================================
hr("F  THE WORST STATE I CAN BUILD, AND WHAT /health/full SAYS ABOUT IT")
# ===========================================================================
wipe_store()
tailhealth._reset_for_tests()
degrade._reset_cache_for_tests()
summarizer.ENABLED = False
os.environ["WEBUIDB_SYNC_ENABLED"] = "true"
os.environ["WEBUI_SNAPSHOT_DB"] = snap
os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "inf"
stale = time.time() - 90000
os.utime(snap, (stale, stale))
for _ in range(22):
    asyncio.run(_real_rollup("f_worst", convo(60), "A complete reply."))
r = full()
print("  state of the pod:")
print("    - the summary hierarchy is switched off; 22 turns produced nothing")
print("    - the /data snapshot has not been written for 25 hours")
print("    - WEBUI_DB_SYNC_INTERVAL_S is a value the probe cannot round")
print("")
print(f"  /health/full status        : {r['status']!r}")
print(f"  /health/full status_reasons: {r['status_reasons']}")
print(f"  checks.snapshot            : {r['checks'].get('snapshot')}")
print(f"  stats.hierarchy_lag        : {r['stats']['hierarchy_lag'] if 'stats' in r else health.gather_memory_stats()['hierarchy_lag']}")
broke(r["status"] == "ok",
      "F: both new observables are simultaneously dead and /health/full "
      "answers status 'ok' with an empty status_reasons and a green "
      "HEALTHCHECK -- the exact report 6630134 exists to make impossible")
summarizer.ENABLED = True
os.environ["WEBUIDB_SYNC_ENABLED"] = "false"
os.environ["WEBUI_DB_SYNC_INTERVAL_S"] = "300"

print("")
print("=" * 74)
print(f"BREAKS REPRODUCED: {len(BROKEN)}   (controls held: {len(HELD)})")
for x in BROKEN:
    print("  - " + x)
print("=" * 74)
