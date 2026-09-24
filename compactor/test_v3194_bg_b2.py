"""
v3.1.9.4 B2 (P15-2 + P15-8): a redeploy mid-backfill used to abandon it
forever, and COMPACTOR_FACTS_EXTRACTION=false did not stop it from
spending vLLM calls anyway.

Part A mirrors the reviewer's SIGKILL-a-child shape (SP\\p15\\
p15_b2_backfill.py) as a subprocess test: a child process starts a
backfill, its OWN kickoff-reply tail writes one fact (what production
does within seconds of the reply), then the child is SIGKILLed part-way
through the backfill. A fresh process (the redeployed pod) must now see
`needs_backfill() == True` and a subsequent run must merge the rest of
the history in, rather than being permanently blocked by the live tail's
facts file existing on disk. Part B proves COMPACTOR_FACTS_EXTRACTION=false
stops needs_backfill from ever starting one. Synthetic text only.

Run: python test_v3194_bg_b2.py
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

_TMP_ROOT = tempfile.mkdtemp(prefix="v3194-bg-b2-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_HIERARCHICAL_SUMMARY"] = "false"

import backfill  # noqa: E402
import facts  # noqa: E402
import memory  # noqa: E402

FAILED = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label, flush=True)
    if not cond:
        FAILED.append(label)


def _wipe():
    import shutil
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()
    backfill._in_progress_local.clear()


# ---------------------------------------------------------------------------
# Part A: SIGKILL-a-child (subprocess), the reviewer's shape
# ---------------------------------------------------------------------------

_CHILD_SCRIPT = r'''
import asyncio, os, sys, time
sys.path.insert(0, os.environ["COMPACTOR_SRC"])
import backfill, facts, memory

CID = "b2conv"
N_PAIRS = 40
MSGS = [{"role": "system", "content": "sys"}]
for i in range(N_PAIRS):
    MSGS.append({"role": "user", "content": f"user message {i} with synthetic detail {i}"})
    MSGS.append({"role": "assistant", "content": f"assistant reply {i} acknowledging detail {i}."})
MSGS.append({"role": "user", "content": "the new message that triggered the kickoff"})

async def slow_extract(client, url, model, user_text, asst_text, existing, **kw):
    await asyncio.sleep(0.2)
    return [f"history fact from: {user_text[:24]}"]

facts.extract_facts_from_exchange = slow_extract
backfill.facts_module.extract_facts_from_exchange = slow_extract

async def run():
    tasks = []
    started = await backfill.start_backfill_if_needed(
        CID, MSGS, "http://stub:8000", "m",
        fire_and_forget=lambda coro, label=None: tasks.append(asyncio.create_task(coro)) or True,
    )
    print(f"CHILD kickoff started={started}", flush=True)
    await asyncio.sleep(0.5)
    async with memory.conv_lock(CID):
        facts.save_facts(CID, [{"text": "a fact the live tail extracted", "added_turn": 83, "last_used": int(time.time())}])
    print("CHILD live tail wrote its fact", flush=True)
    await asyncio.sleep(60)

asyncio.run(run())
'''


def part_a():
    print("\n[Part A] SIGKILL-a-child: a redeploy mid-backfill must be retried, not abandoned")
    _wipe()
    child_path = os.path.join(_TMP_ROOT, "_child.py")
    with open(child_path, "w", encoding="utf-8") as f:
        f.write(_CHILD_SCRIPT)

    src = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, COMPACTOR_SRC=src)
    p = subprocess.Popen([sys.executable, child_path], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    deadline = time.time() + 30
    st = None
    while time.time() < deadline:
        st = backfill.read_state("b2conv")
        if st and st.get("exchanges_done", 0) >= 8:
            break
        time.sleep(0.05)
    # p.kill() is SIGKILL on POSIX (Linux/Docker — the authoritative
    # platform per SP\V3194_SHARED.md; rc == -9). Windows has no SIGKILL;
    # p.kill() there is TerminateProcess, close enough for a provisional
    # local run but not the same guarantee — this whole test's real
    # verdict is the Linux/Docker run (see fix-3194-bg.md).
    p.kill()
    out, _ = p.communicate()
    child_lines = [l for l in out.splitlines() if l.startswith("CHILD")]
    print("  child output:", " | ".join(child_lines))
    print(f"  child rc={p.returncode} (SIGKILL expected: -9 on POSIX)")
    if os.name == "posix":
        check(p.returncode == -9, "child was actually SIGKILLed (rc=-9), not a clean exit")
    else:
        check(p.returncode != 0, "child was forcibly terminated (Windows: TerminateProcess, not a clean exit)")

    st = backfill.read_state("b2conv")
    print(f"  after kill: state={st.get('state') if st else None} "
          f"done={st.get('exchanges_done') if st else None}/{st.get('exchanges_total') if st else None}")
    check(bool(st) and st.get("state") == "in_progress",
          "record is stuck in_progress after the kill (the P15-2 shape)")
    check(bool(st) and st.get("exchanges_done", 0) >= 1,
          "the kill landed mid-run, not before it started")

    on_disk = facts.load_facts("b2conv")
    history_facts_before = [f for f in on_disk if f["text"].startswith("history fact")]
    check(len(on_disk) == 1 and not history_facts_before,
          "only the live tail's own fact is on disk — the backfill's history extraction died with the child")

    # Fresh-process view: no in-memory state survives a redeploy. A
    # just-killed record is NOT yet stale (_STALE_SECONDS=600 by default,
    # and no real time has passed) — that is CORRECT: an in_progress
    # record younger than the threshold might still be a live run
    # somewhere else, so needs_backfill must still say False here. The
    # real-world case P15-2 is about is a pod that comes back up minutes
    # later, well past the threshold — simulated the same way the
    # reviewer's own script does it (SP\p15\p15_b2_backfill.py):
    # shorten _STALE_SECONDS to 0 and let a moment pass.
    st_before_stale = backfill.read_state("b2conv")
    check(backfill.needs_backfill("b2conv", [{"role": "user", "content": "u"},
                                              {"role": "assistant", "content": "a"}] * 3) is False,
          "a JUST-killed record (not yet past _STALE_SECONDS) correctly still says no retry yet")
    backfill._STALE_SECONDS = 0
    time.sleep(0.05)
    check(backfill.is_stale(st_before_stale) is True,
          "sanity: the same record now reads as stale once the threshold is lowered")

    backfill._in_progress_local.clear()
    MSGS = [{"role": "system", "content": "sys"}]
    for i in range(40):
        MSGS.append({"role": "user", "content": f"user message {i} with synthetic detail {i}"})
        MSGS.append({"role": "assistant", "content": f"assistant reply {i} acknowledging detail {i}."})
    MSGS.append({"role": "assistant", "content": "reply."})
    MSGS.append({"role": "user", "content": "next"})

    needs = backfill.needs_backfill("b2conv", MSGS)
    check(needs is True,
          "THE FIX: needs_backfill() is True after the redeploy — a facts "
          "file existing (the live tail's write) no longer permanently "
          "blocks the retry; the abandoned in_progress record does")

    async def fast_extract(client, url, model, user_text, asst_text, existing, **kw):
        return [f"history fact from: {user_text[:24]}"]
    backfill.facts_module.extract_facts_from_exchange = fast_extract
    facts.extract_facts_from_exchange = fast_extract

    async def resume():
        tasks = []
        started2 = await backfill.start_backfill_if_needed(
            "b2conv", MSGS, "http://stub:8000", "m",
            fire_and_forget=lambda coro, label=None: tasks.append(asyncio.create_task(coro)) or True,
        )
        if tasks:
            await asyncio.gather(*tasks)
        return started2

    started2 = asyncio.run(resume())
    check(started2 is True, "the resumed backfill actually starts")

    final = facts.load_facts("b2conv")
    final_history = [f for f in final if f["text"].startswith("history fact")]
    live_tail_fact = [f for f in final if f["text"] == "a fact the live tail extracted"]
    check(len(final_history) >= 30,
          f"the resumed run actually extracted the history ({len(final_history)} history facts recovered)")
    check(len(live_tail_fact) == 1,
          "the live tail's original fact survived the resume — merge, not replace")
    st_final = backfill.read_state("b2conv")
    check(bool(st_final) and st_final.get("state") == "complete",
          "the backfill record now reads complete, not stuck in_progress forever")


# ---------------------------------------------------------------------------
# Part A control: a normal (non-abandoned) backfill is unaffected
# ---------------------------------------------------------------------------

def part_a_control():
    print("\n[Part A CONTROL] a fresh V1 conversation with no prior attempt "
          "still backfills normally (the fix must not make everything "
          "look like a resume)")
    _wipe()

    async def fast_extract(client, url, model, user_text, asst_text, existing, **kw):
        return [f"ctl fact from: {user_text[:24]}"]
    backfill.facts_module.extract_facts_from_exchange = fast_extract
    facts.extract_facts_from_exchange = fast_extract

    MSGS = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "another turn"}, {"role": "assistant", "content": "sure"}]
    check(backfill.needs_backfill("ctl-conv", MSGS) is True, "a genuine V1 conversation still needs backfill")

    async def go():
        tasks = []
        started = await backfill.start_backfill_if_needed(
            "ctl-conv", MSGS, "http://stub:8000", "m",
            fire_and_forget=lambda coro, label=None: tasks.append(asyncio.create_task(coro)) or True,
        )
        if tasks:
            await asyncio.gather(*tasks)
        return started

    started = asyncio.run(go())
    check(started is True, "control backfill starts")
    st = backfill.read_state("ctl-conv")
    check(bool(st) and st.get("state") == "complete", "control backfill completes normally")
    check(len(facts.load_facts("ctl-conv")) >= 1, "control backfill wrote facts")


# ---------------------------------------------------------------------------
# Part A refusal still holds: a store with facts and NO backfill record
# (not a resume) still refuses, exactly as before this fix
# ---------------------------------------------------------------------------

def part_a_refusal_unaffected():
    print("\n[Part A: refusal unaffected] a store with facts and NO backfill "
          "record at all is still refused — the fix only exempts a genuine "
          "resume, not every non-empty store")
    _wipe()
    facts.save_facts("live-store", [{"text": "already here", "added_turn": 1, "last_used": 1}])
    before = facts.load_facts("live-store")
    MSGS = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 3

    called = {"n": 0}

    async def spy(client, url, model, user_text, asst_text, existing, **kw):
        called["n"] += 1
        return ["should not run"]
    backfill.facts_module.extract_facts_from_exchange = spy
    facts.extract_facts_from_exchange = spy

    asyncio.run(backfill._run_backfill("live-store", MSGS, "http://fake", "fake-model"))
    check(called["n"] == 0, "refused before spending a single extraction call")
    check(facts.load_facts("live-store") == before, "store byte-for-byte unchanged")
    check(backfill.read_state("live-store") is None,
          "no backfill state written for a run that never started")


# ---------------------------------------------------------------------------
# Part B: COMPACTOR_FACTS_EXTRACTION=false stops backfill entirely (P15-8)
# ---------------------------------------------------------------------------

def part_b():
    print("\n[Part B] COMPACTOR_FACTS_EXTRACTION=false must stop the lazy "
          "backfill, not just the live tail")
    _wipe()
    MSGS = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "another turn"}, {"role": "assistant", "content": "sure"}]

    called = {"n": 0}

    async def spy(client, url, model, user_text, asst_text, existing, **kw):
        called["n"] += 1
        return ["should never be extracted"]

    with patch.object(facts, "_EXTRACTION_ENABLED", False):
        check(facts.extraction_enabled() is False, "sanity: extraction really is off")
        check(backfill.needs_backfill("ext-off-conv", MSGS) is False,
              "THE FIX: needs_backfill() refuses when extraction is disabled, "
              "even for an otherwise-eligible V1 conversation")

        with patch.object(backfill.facts_module, "extract_facts_from_exchange", spy):
            calls = []
            started = asyncio.run(backfill.start_backfill_if_needed(
                "ext-off-conv", MSGS, "http://stub:8000", "m",
                fire_and_forget=lambda coro, label=None: calls.append(asyncio.ensure_future(coro)) or True,
            ))
    check(started is False, "start_backfill_if_needed does not start a backfill")
    check(len(calls) == 0, "no background task was even spawned")
    check(called["n"] == 0, "no extraction call was made")
    check(backfill.read_state("ext-off-conv") is None, "no backfill record was written")


def part_b_control():
    print("\n[Part B CONTROL] with extraction back on, the identical "
          "conversation shape DOES need a backfill")
    _wipe()
    check(facts.extraction_enabled() is True, "sanity: extraction is on by default in this file")
    MSGS = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "another turn"}, {"role": "assistant", "content": "sure"}]
    check(backfill.needs_backfill("ext-on-conv", MSGS) is True,
          "CONTROL: needs_backfill() is True with extraction enabled")


if __name__ == "__main__":
    import shutil
    try:
        part_a()
        part_a_control()
        part_a_refusal_unaffected()
        part_b()
        part_b_control()
        print("\nRESULT:", "all v3.1.9.4 B2 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    sys.exit(0 if not FAILED else 1)
