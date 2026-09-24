"""Health signals that used to read green while memory stopped (v3.1.9 HIGH).

Every test here drives a state that production actually produces and reads it
back through `gather_health_full`, the function /health/full returns. Each new
status reason has three things pinned: it FIRES in the broken state, it does
NOT fire in the matching healthy state, and (in the lane report) a mutation
that removes it turns the firing check red.

  N6  the summary hierarchy stalls in a way `hierarchy_lag` cannot see,
      because the lag is written by the rollup whose absence it reports
  N7  a backend that returns no text, reply after reply
  N8  COMPACTOR_HIERARCHICAL_SUMMARY=false had no observable at all
  N9  a non-finite / non-positive WEBUI_DB_SYNC_INTERVAL_S, a snapshot
      probe that raised with no status branch, and a snapshot mtime in the
      FUTURE read as fresh
  TOK the local tokenizer's load state (contract: main.tokenizer_state())

    python test_health_signals.py
"""

import asyncio
import contextlib
import os
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
_TMP_ROOT = tempfile.mkdtemp(prefix="health-signals-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
# Two chunks of lag is the EXISTING lag reason's limit (2 * L1_CHUNK_SIZE).
# The real-turn tests below drive 25 exchanges (51 turns) against a stub vLLM
# that cannot summarize, so at the default chunk of 20 the existing lag reason
# would fire in the CONTROL and the control could not say "ok". 40 puts that
# limit at 80, out of the way, without changing anything the new signals read.
os.environ["COMPACTOR_L1_CHUNK_SIZE"] = "40"
# No daemon unless a test says so.
os.environ.pop("WEBUIDB_SYNC_ENABLED", None)

import memory  # noqa: E402

memory.ensure_storage_layout()

import retrieval  # noqa: E402

retrieval.conversation_doc_count = lambda conv_id: 0

import bgwork  # noqa: E402
import health  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402
import tailhealth  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


async def _vllm_ok(url):
    return {"ok": True, "latency_ms": 1.0, "models": ["m"], "error": None}


health.probe_vllm = _vllm_ok


def _reset_all():
    tailhealth._reset_for_tests()
    getattr(health, "_reset_hierarchy_progress_for_tests", lambda: None)()
    root = memory.storage_root()
    for sub in ("facts", "summaries"):
        d = root / sub
        if d.exists():
            for f in d.glob("*.json"):
                f.unlink()


async def _full():
    return await health.gather_health_full("http://fake", 4096)


def full():
    return asyncio.run(_full())


def _has(r, needle):
    return any(needle in x for x in r["status_reasons"])


# ---------------------------------------------------------------------------
# Real turns through the real tail
# ---------------------------------------------------------------------------

def _history(i):
    """The array a client sends for exchange i: i prior exchanges + a user."""
    msgs = []
    for k in range(i):
        msgs.append({"role": "user",
                     "content": f"Question {k}: what happened in the garden on day {k}?"})
        msgs.append({"role": "assistant", "content": _reply(k)})
    msgs.append({"role": "user",
                 "content": f"Question {i}: what happened in the garden on day {i}?"})
    return msgs


def _reply(i):
    return (
        f"On day {i} the roses opened along the east wall. "
        f"The tea was ready by {i % 12 + 1} o'clock, and we talked about "
        f"moving the bench into the shade. Nothing else of note happened."
    )


async def _drive(conv, start, n):
    """n real exchanges through main._run_memory_tail, each finished and
    stored, each tail run to completion before the next — the order a human
    conversation produces."""
    for i in range(start, start + n):
        d = main._run_memory_tail(
            conv, _reply(i),
            finished=True, truncated=False, holed=False,
            touched_facts=[], last_user_text=f"Question {i}: what happened?",
            turn_index=2 * i + 2, messages=_history(i), injected_facts=None,
        )
        assert d.outcome == tailhealth.STORED, d
        await bgwork.pool.drain(60)


# ---------------------------------------------------------------------------
# N6 — the hierarchy stops advancing and hierarchy_lag cannot see it
# ---------------------------------------------------------------------------

def test_n6_a_state_write_that_never_lands_degrades():
    print("\n[N6] 25 real turns whose rollup state write fails every time")
    _reset_all()
    real_save = summarizer.save_state

    def _boom(conv_id, state):
        raise OSError("simulated: the state write cannot land")

    async def go():
        first = await _full()               # the HEALTHCHECK has been polling
        summarizer.save_state = _boom
        try:
            await _drive("n6_nosave", 0, 25)
        finally:
            summarizer.save_state = real_save
        return first, await _full()

    first, r = asyncio.run(go())
    print(f"      stats.hierarchy_lag = {r['stats'].get('hierarchy_lag')}")
    print(f"      checks.hierarchy    = {r['checks'].get('hierarchy')}")
    print(f"      status={r['status']!r} reasons={r['status_reasons']}")
    check(first["status"] == "ok", "N6 fixture: the pod starts ok")
    check(r["stats"].get("hierarchy_lag") == 0,
          "N6 fixture: hierarchy_lag reads 0 - the blind spot is real")
    check(r["status"] == "degraded",
          "N6 FIRES: 25 turns with no hierarchy progress degrade the pod")
    check(_has(r, "has not advanced"),
          "N6 FIRES: and the reason says the hierarchy has not advanced")


def test_n6_control_healthy_turns_stay_ok():
    print("\n[N6] CONTROL: the same 25 real turns with working state writes")
    _reset_all()

    async def go():
        await _full()
        await _drive("n6_healthy", 0, 25)
        return await _full()

    r = asyncio.run(go())
    print(f"      checks.hierarchy = {r['checks'].get('hierarchy')}")
    print(f"      status={r['status']!r} reasons={r['status_reasons']}")
    check(r["status"] == "ok" and r["status_reasons"] == [],
          "N6 CONTROL: a healthy conversation of 25 turns is ok, no reasons")
    check(not _has(r, "has not advanced"),
          "N6 CONTROL: no stall reason on a hierarchy that is advancing")


def test_n6_control_polled_every_turn_stays_ok():
    """The HEALTHCHECK polls every 30 s, i.e. between most turns. Progress
    observed at each poll must keep resetting the count."""
    print("\n[N6] CONTROL: polled between every turn, the count keeps resetting")
    _reset_all()

    async def go():
        worst = 0
        await _full()
        for i in range(25):
            await _drive("n6_polled", i, 1)
            r = await _full()
            h = r["checks"].get("hierarchy") or {}
            worst = max(worst, h.get("decisions_since_progress") or 0)
        return r, worst

    r, worst = asyncio.run(go())
    print(f"      worst decisions_since_progress across 25 polls = {worst}")
    check(r["status"] == "ok", "N6 CONTROL: polled every turn, still ok")
    check(worst <= 1,
          "N6 CONTROL: each observed write resets the count (worst <= 1)")


def test_n6_threshold_boundary():
    """limit-1 decisions with no progress are quiet; the limit-th fires.
    Decisions counted with nothing written is exactly what production holds
    when every rollup no-ops, whatever the route."""
    print("\n[N6] the stall threshold, pinned from both sides")
    _reset_all()
    limit = getattr(health, "HIERARCHY_STALL_DECISIONS", 20)

    async def go():
        await _full()
        for _ in range(limit - 1):
            tailhealth.note(tailhealth.STORED, raw_chars=500, kept_chars=500)
        below = await _full()
        tailhealth.note(tailhealth.STORED, raw_chars=500, kept_chars=500)
        at = await _full()
        return below, at

    below, at = asyncio.run(go())
    print(f"      limit={limit} below={below['checks'].get('hierarchy')}")
    check(limit == 20, "N6: the limit is 20 decisions")
    check(not _has(below, "has not advanced"),
          "N6 BOUNDARY: limit-1 decisions without progress stay quiet")
    check(_has(at, "has not advanced"),
          "N6 BOUNDARY: the limit-th decision without progress fires")


def test_n6_harmless_outcomes_do_not_count():
    print("\n[N6] CONTROL: task traffic and empty replies never roll up, so "
          "they are not counted")
    _reset_all()

    async def go():
        await _full()
        for _ in range(60):
            tailhealth.note(tailhealth.SKIPPED_TASK_TRAFFIC,
                            raw_chars=300, kept_chars=0)
        for _ in range(5):
            tailhealth.note(tailhealth.SKIPPED_EMPTY, raw_chars=0, kept_chars=0)
        return await _full()

    r = asyncio.run(go())
    h = r["checks"].get("hierarchy") or {}
    print(f"      checks.hierarchy = {h}")
    check(h.get("decisions_since_progress") == 0,
          "N6 CONTROL: 60 task-traffic + 5 empty decisions count as 0")
    check(r["status"] == "ok", "N6 CONTROL: and the pod stays ok")


def test_n6_progress_on_disk_resets_the_count():
    print("\n[N6] CONTROL: a state write seen between polls resets the count")
    _reset_all()
    limit = getattr(health, "HIERARCHY_STALL_DECISIONS", 20)

    async def go():
        await _full()
        for _ in range(limit - 1):
            tailhealth.note(tailhealth.STORED, raw_chars=500, kept_chars=500)
        st = summarizer.load_state("n6_progress")
        st["turns_seen"] = 7
        summarizer.save_state("n6_progress", st)
        await _full()
        for _ in range(limit - 1):
            tailhealth.note(tailhealth.STORED, raw_chars=500, kept_chars=500)
        return await _full()

    r = asyncio.run(go())
    h = r["checks"].get("hierarchy") or {}
    print(f"      checks.hierarchy = {h}")
    check(h.get("decisions_since_progress") == limit - 1,
          "N6: the count restarted at the observed write")
    check(not _has(r, "has not advanced"),
          "N6 CONTROL: 2*(limit-1) decisions split by a write stay quiet")


def test_n6_unreadable_halves_report_none_and_do_not_crash():
    print("\n[N6] an unreadable tail counter or scan: None, not 0, and no crash")
    _reset_all()

    def _raise(**kw):
        raise RuntimeError("counter is wedged")

    with patch("tailhealth.snapshot", new=_raise):
        r = full()
    h = r["checks"].get("hierarchy") or {}
    print(f"      tail wedged: checks.hierarchy={h} reasons={r['status_reasons']}")
    check(h.get("decisions_since_progress") is None,
          "N6: a tail counter that raised reads as not-measured (None)")
    check(_has(r, "memory tail unobservable"),
          "N6: and the existing tail reason still reports it")

    real_stats = health.gather_memory_stats

    def _stats_without_fingerprint():
        s = real_stats()
        s.pop("_hierarchy_fingerprint", None)
        return s

    with patch("health.gather_memory_stats", new=_stats_without_fingerprint):
        r2 = full()
    h2 = r2["checks"].get("hierarchy") or {}
    check(h2.get("decisions_since_progress") is None,
          "N6: a scan with no fingerprint reads as not-measured (None)")
    check("_hierarchy_fingerprint" not in full()["stats"],
          "N6: the private fingerprint never reaches the payload")


def test_n6_a_reset_counter_restarts_the_base():
    """tailhealth's counter going DOWN (a reset) must restart the base, or
    every decision after it is measured against a number it can never reach."""
    print("\n[N6] a counter that went down restarts the base")
    _reset_all()
    limit = getattr(health, "HIERARCHY_STALL_DECISIONS", 20)

    async def go():
        await _full()
        for _ in range(limit - 1):
            tailhealth.note(tailhealth.STORED, raw_chars=500, kept_chars=500)
        # Progress observed HERE, so the base moves up to limit-1. Without
        # this the base is still 0 and a reset changes nothing - the first
        # version of this test passed with the reset branch deleted.
        st = summarizer.load_state("n6_reset")
        st["turns_seen"] = 3
        summarizer.save_state("n6_reset", st)
        mid = await _full()
        tailhealth._reset_for_tests()       # the counter restarts; the disk does not
        await _full()
        for _ in range(limit):
            tailhealth.note(tailhealth.STORED, raw_chars=500, kept_chars=500)
        return mid, await _full()

    mid, r = asyncio.run(go())
    h = r["checks"].get("hierarchy") or {}
    print(f"      at the write: {mid['checks'].get('hierarchy')}")
    print(f"      checks.hierarchy={h}")
    check((mid["checks"].get("hierarchy") or {}).get("decisions_since_progress") == 0,
          "N6 fixture: the write moved the base up to the current count")
    check(h.get("decisions_since_progress") == limit,
          "N6: counted from the reset, not from the old base")
    check(_has(r, "has not advanced"),
          "N6: so limit decisions after a reset still fire")


# ---------------------------------------------------------------------------
# N8 — the master switch is visible, and does not cry wolf when set on purpose
# ---------------------------------------------------------------------------

def test_n8_switch_off_is_visible_and_does_not_fire_the_stall():
    print("\n[N8] COMPACTOR_HIERARCHICAL_SUMMARY=false, 25 real turns")
    _reset_all()

    async def go():
        await _full()
        summarizer.ENABLED = False
        try:
            await _drive("n8_off", 0, 25)
            return await _full()
        finally:
            summarizer.ENABLED = True

    r = asyncio.run(go())
    print(f"      config={r['config']} checks.hierarchy={r['checks'].get('hierarchy')}")
    print(f"      state file exists={summarizer.summary_path('n8_off').exists()}")
    print(f"      status={r['status']!r} reasons={r['status_reasons']}")
    check(not summarizer.summary_path("n8_off").exists(),
          "N8 fixture: the switch really stopped every write")
    check(r["config"].get("hierarchical_summary") is False,
          "N8: config.hierarchical_summary reports the switch as False")
    check((r["checks"].get("hierarchy") or {}).get("enabled") is False,
          "N8: checks.hierarchy.enabled reports False")
    check(not _has(r, "has not advanced"),
          "N8: a hierarchy switched off on purpose is not reported as stalled")

    r2 = full()
    check(r2["config"].get("hierarchical_summary") is True,
          "N8 CONTROL: with the switch on, config reports True")


# ---------------------------------------------------------------------------
# N7 — a run of replies with no text
# ---------------------------------------------------------------------------

def test_n7_thirty_empty_replies_degrade():
    print("\n[N7] 30 consecutive empty replies on a 60-turn conversation")
    _reset_all()
    msgs = _history(30)

    async def go():
        for i in range(30):
            main._run_memory_tail(
                "n7_empty", "",
                finished=True, truncated=False, holed=False,
                touched_facts=[], last_user_text="what did we decide?",
                turn_index=i, messages=msgs, injected_facts=None,
            )
        return await _full()

    r = asyncio.run(go())
    print(f"      memory_tail.consecutive_empty_replies = "
          f"{r['memory_tail'].get('consecutive_empty_replies')}")
    print(f"      status={r['status']!r} reasons={r['status_reasons']}")
    check(r["status"] == "degraded",
          "N7 FIRES: 30 empty replies in a row degrade the pod")
    check(_has(r, "no text"),
          "N7 FIRES: and the reason says the backend returned no text")


def test_n7_threshold_boundary_and_reset():
    print("\n[N7] the empty-run threshold, and what ends a run")
    _reset_all()
    limit = getattr(tailhealth, "EMPTY_RUN_DEGRADE", 10)

    def empties(n):
        for _ in range(n):
            tailhealth.note(tailhealth.SKIPPED_EMPTY, raw_chars=0, kept_chars=0)

    empties(limit - 1)
    below = full()
    empties(1)
    at = full()
    tailhealth.note(tailhealth.STORED, raw_chars=900, kept_chars=900)
    after_reply = full()
    _reset_all()
    empties(limit - 1)
    tailhealth.note(tailhealth.SKIPPED_TASK_TRAFFIC, raw_chars=120, kept_chars=0)
    empties(limit - 1)
    split = full()
    print(f"      limit={limit} at: {at['status_reasons']}")
    check(limit == 10, "N7: the limit is 10 consecutive empty replies")
    check(below["status"] == "ok" and not _has(below, "no text"),
          "N7 BOUNDARY: limit-1 empty replies stay ok (a few Stops are not an outage)")
    check(_has(at, "no text"),
          "N7 BOUNDARY: the limit-th consecutive empty reply fires")
    check(not _has(after_reply, "no text") and after_reply["status"] == "ok",
          "N7 CONTROL: one reply with text ends the run and clears the reason")
    check(not _has(split, "no text"),
          "N7 CONTROL: a task-traffic reply WITH text also ends the run")


# ---------------------------------------------------------------------------
# N9 — the snapshot probe
# ---------------------------------------------------------------------------

_SNAP = os.path.join(_TMP_ROOT, "snapshot-webui.db")


def _snap_env(interval):
    return {
        "WEBUIDB_SYNC_ENABLED": "true",
        "WEBUI_SNAPSHOT_DB": _SNAP,
        "WEBUI_DB_SYNC_INTERVAL_S": interval,
    }


def _write_snap(age_s):
    with open(_SNAP, "wb") as fh:
        fh.write(b"x")
    t = time.time() - age_s
    os.utime(_SNAP, (t, t))


def test_n9_nonfinite_interval_does_not_kill_the_probe():
    print("\n[N9] inf / 1e400 / nan interval, through gather_health_full")
    _reset_all()
    for raw in ("inf", "1e400", "nan"):
        _write_snap(4 * 300)
        with patch.dict(os.environ, _snap_env(raw)):
            r = full()
        snap = r["checks"]["snapshot"] or {}
        print(f"      {raw:>6}: snapshot={snap} reasons={r['status_reasons']}")
        check(snap.get("watched") is True and snap.get("stale") is True,
              f"N9 FIRES [{raw}]: a 1200 s old snapshot is still judged stale")
        check(r["status"] == "degraded" and _has(r, "snapshot"),
              f"N9 FIRES [{raw}]: and the pod is degraded with a snapshot reason")
        _write_snap(10)
        with patch.dict(os.environ, _snap_env(raw)):
            ok = full()
        check(ok["status"] == "ok" and (ok["checks"]["snapshot"] or {}).get("stale") is False,
              f"N9 CONTROL [{raw}]: a fresh snapshot under the same bad interval is ok")


def test_n9_zero_and_negative_interval_do_not_pin_degraded():
    print("\n[N9] 0 / -1 interval: a fresh snapshot must not read stale")
    _reset_all()
    for raw in ("0", "-1"):
        _write_snap(1)
        with patch.dict(os.environ, _snap_env(raw)):
            r = full()
        snap = r["checks"]["snapshot"] or {}
        print(f"      {raw:>3}: snapshot={snap}")
        check(snap.get("stale") is False and r["status"] == "ok",
              f"N9 [{raw}]: a snapshot written a second ago is not stale")
        _write_snap(4 * 300)
        with patch.dict(os.environ, _snap_env(raw)):
            r2 = full()
        check((r2["checks"]["snapshot"] or {}).get("stale") is True,
              f"N9 CONTROL [{raw}]: the clamp did not disable staleness")


def test_n9_a_probe_that_raises_is_a_status_reason():
    print("\n[N9] probe_snapshot raising reaches status_reasons")
    _reset_all()

    def _raise():
        raise RuntimeError("probe is wedged")

    with patch("health.probe_snapshot", new=_raise):
        r = full()
    print(f"      snapshot={r['checks']['snapshot']} reasons={r['status_reasons']}")
    check(r["status"] == "degraded",
          "N9 FIRES: a snapshot probe that raised degrades the pod")
    check(_has(r, "snapshot probe unobservable"),
          "N9 FIRES: and the reason says the snapshot could not be observed")

    with patch.dict(os.environ, {"WEBUIDB_SYNC_ENABLED": "false"}):
        r2 = full()
    check(r2["status"] == "ok" and r2["status_reasons"] == [],
          "N9 CONTROL: an unwatched snapshot (daemon off) raises nothing")


def test_n9_a_snapshot_from_the_future_is_stale_and_names_the_clock():
    print("\n[N9] a snapshot mtime an hour in the future")
    _reset_all()
    _write_snap(-3600)
    with patch.dict(os.environ, _snap_env("300")):
        r = full()
    snap = r["checks"]["snapshot"] or {}
    print(f"      snapshot={snap}")
    print(f"      reasons={r['status_reasons']}")
    check(snap.get("stale") is True,
          "N9 FIRES: a snapshot an hour in the future reads stale, not fresh")
    check(r["status"] == "degraded" and _has(r, "clock"),
          "N9 FIRES: and the reason names the clock")

    # From above: thirty seconds ahead is already a clock fault, so the
    # tolerance cannot quietly widen into a window that hides one.
    _write_snap(-30)
    with patch.dict(os.environ, _snap_env("300")):
        half_min = full()
    check((half_min["checks"]["snapshot"] or {}).get("stale") is True
          and _has(half_min, "clock"),
          "N9 FIRES: a snapshot 30 s in the future is a clock fault too")

    _write_snap(5)
    with patch.dict(os.environ, _snap_env("300")):
        ok = full()
    check(ok["status"] == "ok" and not _has(ok, "clock"),
          "N9 CONTROL: a snapshot 5 s in the past is fresh")

    # Timestamp rounding on the network filesystem can put a just-published
    # mtime a fraction of a second ahead; that must not cry wolf.
    _write_snap(-0.5)
    with patch.dict(os.environ, _snap_env("300")):
        tiny = full()
    check(tiny["status"] == "ok" and not _has(tiny, "clock"),
          "N9 CONTROL: half a second of timestamp rounding is not a clock fault")


def test_n9_missing_snapshot_reason_is_readable():
    print("\n[N9] a missing snapshot's reason carries the error, not 'Nones old'")
    _reset_all()
    with patch.dict(os.environ, {
        "WEBUIDB_SYNC_ENABLED": "true",
        "WEBUI_SNAPSHOT_DB": os.path.join(_TMP_ROOT, "absent.db"),
    }):
        r = full()
    print(f"      reasons={r['status_reasons']}")
    check(r["status"] == "degraded", "N9: a missing snapshot degrades")
    check(not _has(r, "Nones old"), "N9: and the reason does not read 'Nones old'")


# ---------------------------------------------------------------------------
# TOK — main.tokenizer_state() (the config lane's contract)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _tokenizer_state(fn):
    had = hasattr(main, "tokenizer_state")
    old = getattr(main, "tokenizer_state", None)
    if fn is None:
        if had:
            delattr(main, "tokenizer_state")
    else:
        main.tokenizer_state = fn
    try:
        yield
    finally:
        if had:
            main.tokenizer_state = old
        elif hasattr(main, "tokenizer_state"):
            delattr(main, "tokenizer_state")


def test_tok_absent_contract_reports_unavailable():
    print("\n[TOK] main has no tokenizer_state yet")
    _reset_all()
    with _tokenizer_state(None):
        r = full()
    tok = r["checks"].get("tokenizer")
    print(f"      checks.tokenizer={tok}")
    check(isinstance(tok, dict) and tok.get("available") is False and tok.get("reason"),
          "TOK: absent contract reports available=false with a reason")
    check(r["status"] == "ok", "TOK: an absent contract is not a status reason")

    with patch.dict(sys.modules):
        sys.modules.pop("main", None)
        r2 = full()
        imported_main = "main" in sys.modules
    tok2 = r2["checks"].get("tokenizer")
    check(isinstance(tok2, dict) and tok2.get("available") is False,
          "TOK: outside the app process (main not loaded) reports unavailable")
    check(not imported_main,
          "TOK: health did not import main itself (a probe must not boot the app)")


def test_tok_loaded_is_reported_and_ok():
    print("\n[TOK] CONTROL: a loaded tokenizer")
    _reset_all()
    state = {"loaded": True, "last_error": None, "failed_at": None,
             "next_retry_at": None}
    with _tokenizer_state(lambda: dict(state)):
        r = full()
    tok = r["checks"].get("tokenizer") or {}
    print(f"      checks.tokenizer={tok}")
    check(tok.get("available") is True and tok.get("loaded") is True,
          "TOK CONTROL: the state is carried into checks.tokenizer")
    check(r["status"] == "ok" and r["status_reasons"] == [],
          "TOK CONTROL: a loaded tokenizer is ok with no reasons")

    pending = {"loaded": False, "last_error": None, "failed_at": None,
               "next_retry_at": None}
    with _tokenizer_state(lambda: dict(pending)):
        r2 = full()
    check(r2["status"] == "ok",
          "TOK CONTROL: not-yet-attempted (no error) is not a status reason")


def test_tok_failed_load_is_a_status_reason():
    print("\n[TOK] a tokenizer that failed to load")
    _reset_all()
    now = time.time()
    state = {"loaded": False, "last_error": "OSError: no such repo",
             "failed_at": now - 30, "next_retry_at": now + 90}
    with _tokenizer_state(lambda: dict(state)):
        r = full()
    print(f"      reasons={r['status_reasons']}")
    check(r["status"] == "degraded",
          "TOK FIRES: a failed tokenizer load degrades the pod")
    check(_has(r, "local tokenizer"),
          "TOK FIRES: and the reason names the local tokenizer")


def test_tok_state_that_raises_is_unobservable():
    print("\n[TOK] tokenizer_state() raising")
    _reset_all()

    def _raise():
        raise RuntimeError("wedged")

    with _tokenizer_state(_raise):
        r = full()
    tok = r["checks"].get("tokenizer") or {}
    print(f"      checks.tokenizer={tok} reasons={r['status_reasons']}")
    check(tok.get("available") is False and "wedged" in str(tok.get("reason")),
          "TOK: the error is carried in the payload")
    check(_has(r, "tokenizer state unobservable"),
          "TOK FIRES: an unreadable tokenizer state is a status reason")

    with _tokenizer_state(lambda: None):
        r2 = full()
    print(f"      returns None: checks.tokenizer={r2['checks'].get('tokenizer')}")
    check(_has(r2, "tokenizer state unobservable"),
          "TOK FIRES: a tokenizer_state() that breaks its contract is unobservable")


def test_tok_real_contract_end_to_end():
    """MERGE-LEVEL: the config lane's REAL tokenizer_state() feeding the health
    lane's REAL reason, with nothing mocked between them.

    Every TOK test above substitutes a hand-built state dict, and the config
    lane's own tests assert on the dict without ever rendering it. So each lane
    was proven against its own idea of the other. They disagreed: the config
    lane stored next_retry_at on time.monotonic(), this lane computes
    `next_retry_at - time.time()`, and the rendered reason read "next retry in
    0s" for the life of the process. Found at the merge, fixed there (the state
    is now converted to wall clock), and pinned here, because a contract
    between two lanes is checked by nobody except whoever joins them.
    """
    print("\n[TOK] MERGE: real get_tokenizer miss -> real tokenizer_state -> real reason")
    _reset_all()
    import re
    import types

    calls = []

    class _Boom:
        @staticmethod
        def from_pretrained(repo):
            calls.append(repo)
            raise OSError("no weights here (merge test)")

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = _Boom
    saved = {k: getattr(main, k) for k in (
        "_tokenizer", "_TOKENIZER_TRIED", "MODEL_REPO", "_TOKENIZER_LAST_ERROR",
        "_TOKENIZER_FAILED_AT", "_TOKENIZER_NEXT_RETRY_AT", "_TOKENIZER_RETRY_S")}
    saved_mod = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    main._tokenizer = None
    main._TOKENIZER_TRIED = False
    main.MODEL_REPO = "does/not-exist"
    main._TOKENIZER_LAST_ERROR = None
    main._TOKENIZER_FAILED_AT = None
    main._TOKENIZER_NEXT_RETRY_AT = None
    main._TOKENIZER_RETRY_S = main._TOKENIZER_RETRY_FLOOR_S
    try:
        check(hasattr(main, "tokenizer_state"),
              "MERGE: main.tokenizer_state exists on the merged tree")
        check(main.get_tokenizer() is None and len(calls) == 1,
              "MERGE fixture: one real failed load happened")
        st = main.tokenizer_state()
        check(sorted(st) == ["failed_at", "last_error", "loaded", "next_retry_at"],
              f"MERGE: the contract carries exactly the four agreed keys (got {sorted(st)})")
        r = full()
    finally:
        for k, v in saved.items():
            setattr(main, k, v)
        if saved_mod is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved_mod
    print(f"      reasons={r['status_reasons']}")
    check(r["status"] == "degraded" and _has(r, "local tokenizer"),
          "MERGE: the real failure reaches /health/full as the tokenizer reason")
    text = " ".join(r["status_reasons"])
    m = re.search(r"next retry in (\d+)s", text)
    floor = main._TOKENIZER_RETRY_FLOOR_S
    check(m is not None and floor - 5 <= int(m.group(1)) <= floor + 1,
          f"MERGE: the countdown is the real backoff (~{floor:.0f}s), not the "
          f"monotonic/wall mismatch's permanent 0s (got {m.group(0) if m else None!r})")


TESTS = [
    test_n6_a_state_write_that_never_lands_degrades,
    test_n6_control_healthy_turns_stay_ok,
    test_n6_control_polled_every_turn_stays_ok,
    test_n6_threshold_boundary,
    test_n6_harmless_outcomes_do_not_count,
    test_n6_progress_on_disk_resets_the_count,
    test_n6_unreadable_halves_report_none_and_do_not_crash,
    test_n6_a_reset_counter_restarts_the_base,
    test_n8_switch_off_is_visible_and_does_not_fire_the_stall,
    test_n7_thirty_empty_replies_degrade,
    test_n7_threshold_boundary_and_reset,
    test_n9_nonfinite_interval_does_not_kill_the_probe,
    test_n9_zero_and_negative_interval_do_not_pin_degraded,
    test_n9_a_probe_that_raises_is_a_status_reason,
    test_n9_a_snapshot_from_the_future_is_stale_and_names_the_clock,
    test_n9_missing_snapshot_reason_is_readable,
    test_tok_absent_contract_reports_unavailable,
    test_tok_loaded_is_reported_and_ok,
    test_tok_failed_load_is_a_status_reason,
    test_tok_state_that_raises_is_unobservable,
    test_tok_real_contract_end_to_end,
]


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for t in TESTS:
        if only and only not in t.__name__:
            continue
        try:
            t()
        except Exception as e:  # a crash is reported, never scored as a pass
            import traceback
            traceback.print_exc()
            FAILED.append(f"{t.__name__} CRASHED: {type(e).__name__}: {e}")
            print(f"FAIL {t.__name__} CRASHED: {type(e).__name__}: {e}")
    print("")
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("All health-signal checks passed.")
