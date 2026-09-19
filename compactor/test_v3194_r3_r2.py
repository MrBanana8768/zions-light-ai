"""
v3.1.9.4 R2 (P15-2 follow-up): a backfill that keeps failing must not retry
forever.

Round 2 (B2) made a `failed` (or stale `in_progress`) backfill record retry
on the next eligible request — correct for a transient failure (a redeploy,
an OOM, a momentary vLLM outage), but a backfill that fails DETERMINISTICALLY
would now re-run its full multi-hour, multi-vLLM-call extraction on every
single eligible request forever, with no bound at all.

This file proves:
  1. THE CAP: three failed attempts in a row abandon the backfill
     permanently (new terminal state "abandoned"), with one WARNING logged
     naming the conversation, the attempt count and the last error.
  2. THE BACKOFF: a `failed` record does not retry on the very next request
     — it waits `_BACKFILL_RETRY_BACKOFF_S * 2 ** (attempts - 1)` seconds
     since its last update, doubling per attempt already spent.
  3. A SUCCESSFUL RETRY STILL COMPLETES: attempt 1 fails, attempt 2 (once
     eligible) succeeds — "complete", not capped, not abandoned, and
     `needs_backfill` refuses any further attempt afterward for the
     ordinary "already done" reason.
  4. The crash (stale in_progress) case is capped the same way, logged once
     rather than silently forever.

All fixtures are synthetic. Extraction is stubbed (no real HTTP calls).

Run: python test_v3194_r3_r2.py
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-r3-r2-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "true"
os.environ["COMPACTOR_HIERARCHICAL_SUMMARY"] = "false"  # isolate: this item is not about the rollup half

import backfill  # noqa: E402
import facts  # noqa: E402
import memory  # noqa: E402
import logsetup  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def _wipe():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP)
    memory.ensure_storage_layout()
    logsetup._reset_log_once_for_tests()


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _capture():
    lg = logging.getLogger("compactor.backfill")
    h = _Collector()
    lg.addHandler(h)
    return lg, h


_MSGS = [{"role": "user", "content": f"question {i}"} if i % 2 == 0
         else {"role": "assistant", "content": f"answer {i}"} for i in range(10)]


async def _no_new_facts(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
    return []


async def _always_succeeds(client, vllm_url, model, user_msg, assistant_msg, existing_facts, **kw):
    return [f"a fact about {user_msg[:20]}"]


def _prune_facts_deterministic_failure(*a, **kw):
    raise RuntimeError("synthetic deterministic prune failure")


def _run_failing_backfill(cid):
    """A backfill that fails DETERMINISTICALLY, every single attempt — the
    shape this item exists for. Per-exchange extraction failures alone do
    NOT fail the whole run (_run_backfill's own per-exchange try/except
    logs and continues — 'we keep going and save whatever we got'), so the
    failure is injected at the final prune/save step instead, which is
    outside that per-exchange guard and reaches _run_backfill's OUTER
    except — the same site a real deterministic failure (a malformed
    history, a store this conversation can never satisfy) would reach."""
    orig_extract = facts.extract_facts_from_exchange
    orig_prune = facts.prune_facts
    facts.extract_facts_from_exchange = _no_new_facts
    facts.prune_facts = _prune_facts_deterministic_failure
    try:
        asyncio.run(backfill._run_backfill(cid, list(_MSGS), "http://stub:8000", "m"))
    finally:
        facts.extract_facts_from_exchange = orig_extract
        facts.prune_facts = orig_prune


def _run_succeeding_backfill(cid):
    orig = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = _always_succeeds
    try:
        asyncio.run(backfill._run_backfill(cid, list(_MSGS), "http://stub:8000", "m"))
    finally:
        facts.extract_facts_from_exchange = orig


def _age_state(cid, seconds_ago):
    """Rewrite the record's updated_at directly — backfill._write_state
    always stamps the CURRENT time, so it cannot be used to construct an
    aged fixture (same technique test_backfill.py's own stale-in_progress
    test uses)."""
    p = backfill._backfill_state_path(cid)
    state = json.loads(p.read_text(encoding="utf-8"))
    old = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    state["updated_at"] = old
    p.write_text(json.dumps(state), encoding="utf-8")
    return state


# ---------------------------------------------------------------------------
# 1. The cap: three failures in a row -> abandoned, one WARNING, permanent
# ---------------------------------------------------------------------------

def test_three_deterministic_failures_abandon_the_backfill_permanently():
    print("\n[test] R2: a backfill that fails 3 times in a row is ABANDONED — "
          "not retried a 4th time — with one WARNING an operator can find")
    _wipe()
    cid = "r2-cap-abandon"
    lg, collector = _capture()
    try:
        for attempt in (1, 2, 3):
            check(backfill.needs_backfill(cid, _MSGS) is True,
                  f"eligible before attempt {attempt}")
            _run_failing_backfill(cid)
            state = backfill.read_state(cid)
            check(state["attempts"] == attempt, f"attempts recorded as {attempt}")
            if attempt < backfill._MAX_BACKFILL_ATTEMPTS:
                check(state["state"] == "failed", f"attempt {attempt}: state is 'failed' (under the cap)")
                # Not yet eligible again — the backoff has not elapsed.
                check(backfill.needs_backfill(cid, _MSGS) is False,
                      f"not yet eligible again right after attempt {attempt} (backoff)")
                # Fast-forward past this attempt's own backoff window so the
                # NEXT attempt in this loop is allowed to run.
                backoff = backfill._BACKFILL_RETRY_BACKOFF_S * (2 ** (attempt - 1))
                _age_state(cid, backoff + 30)
            else:
                check(state["state"] == "abandoned",
                      f"attempt {attempt} (the cap): state is 'abandoned', not 'failed'")
    finally:
        lg.removeHandler(collector)

    check(backfill.needs_backfill(cid, _MSGS) is False, "abandoned -> never eligible again")
    warnings = [r for r in collector.records if r.levelno == logging.WARNING
                and "abandoned" in r.getMessage()]
    check(len(warnings) == 1, f"exactly one WARNING logged for the abandonment: {len(warnings)}")
    if warnings:
        msg = warnings[0].getMessage()
        check(cid in msg, "the WARNING names the conversation")
        check(str(backfill._MAX_BACKFILL_ATTEMPTS) in msg, "the WARNING names the cap")
        check("synthetic deterministic prune failure" in msg, "the WARNING names the last error")


def test_control_successful_retry_still_completes():
    print("\n[test] CONTROL: attempt 1 fails, attempt 2 (once eligible) "
          "succeeds — completes normally, not capped, not abandoned")
    _wipe()
    cid = "r2-successful-retry"
    _run_failing_backfill(cid)
    state = backfill.read_state(cid)
    check(state["state"] == "failed", "attempt 1 failed")
    check(state["attempts"] == 1, "attempts == 1")
    check(backfill.needs_backfill(cid, _MSGS) is False, "not eligible yet (backoff)")
    _age_state(cid, backfill._BACKFILL_RETRY_BACKOFF_S + 30)
    check(backfill.needs_backfill(cid, _MSGS) is True, "eligible once the backoff elapses")

    _run_succeeding_backfill(cid)
    state = backfill.read_state(cid)
    check(state["state"] == "complete", "attempt 2 completed")
    check(state["attempts"] == 2, "attempts == 2 (recorded, informational)")
    check(len(facts.load_facts(cid)) > 0, "facts were actually written by the successful attempt")
    check(backfill.needs_backfill(cid, _MSGS) is False,
          "complete -> not eligible again (the ordinary reason, unrelated to the cap)")


# ---------------------------------------------------------------------------
# 2. The backoff: doubles per attempt
# ---------------------------------------------------------------------------

def test_backoff_doubles_per_attempt():
    print("\n[test] R2: the backoff window doubles with each recorded attempt")
    _wipe()
    cid = "r2-backoff-doubles"
    base = backfill._BACKFILL_RETRY_BACKOFF_S

    # attempts=1: base window
    state1 = {"state": "failed", "attempts": 1}
    check(backfill._backoff_ready({**state1, "updated_at":
          (datetime.now(timezone.utc) - timedelta(seconds=base - 30)).isoformat()}) is False,
          "attempt 1: not ready just under the base window")
    check(backfill._backoff_ready({**state1, "updated_at":
          (datetime.now(timezone.utc) - timedelta(seconds=base + 30)).isoformat()}) is True,
          "attempt 1: ready just past the base window")

    # attempts=2: 2x window
    state2 = {"state": "failed", "attempts": 2}
    check(backfill._backoff_ready({**state2, "updated_at":
          (datetime.now(timezone.utc) - timedelta(seconds=base + 30)).isoformat()}) is False,
          "attempt 2: the base window alone is NOT enough — must wait 2x")
    check(backfill._backoff_ready({**state2, "updated_at":
          (datetime.now(timezone.utc) - timedelta(seconds=2 * base + 30)).isoformat()}) is True,
          "attempt 2: ready past 2x the base window")


def test_needs_backfill_agrees_with_backoff_ready():
    print("\n[test] needs_backfill's 'failed' branch is exactly _backoff_ready, "
          "driven through the real on-disk record")
    _wipe()
    cid = "r2-needs-backfill-backoff"
    _run_failing_backfill(cid)
    check(backfill.needs_backfill(cid, _MSGS) is False, "fresh failure: not eligible")
    _age_state(cid, backfill._BACKFILL_RETRY_BACKOFF_S + 30)
    check(backfill.needs_backfill(cid, _MSGS) is True, "backed off: eligible")


# ---------------------------------------------------------------------------
# 3. A crash (stale in_progress) already at the cap is refused, logged once
# ---------------------------------------------------------------------------

def test_stale_in_progress_at_the_cap_is_refused_and_logged_once():
    print("\n[test] R2: a crash on what was already the Nth attempt is not "
          "retried a 4th time, and is logged once (not silently forever)")
    _wipe()
    cid = "r2-crash-at-cap"
    old_ts = (
        datetime.now(timezone.utc)
        - timedelta(seconds=backfill._STALE_SECONDS + 60)
    ).isoformat()
    state_path = backfill._backfill_state_path(cid)
    state_path.write_text(json.dumps({
        "conv_id": cid, "state": "in_progress", "started_at": old_ts,
        "updated_at": old_ts, "exchanges_done": 2, "exchanges_total": 10,
        "attempts": backfill._MAX_BACKFILL_ATTEMPTS,
    }))
    lg, collector = _capture()
    try:
        check(backfill.needs_backfill(cid, _MSGS) is False,
              "stale AND already at the cap -> refused, not retried")
        # Called again (a second request on the same stuck conversation) —
        # the log line must not repeat.
        backfill.needs_backfill(cid, _MSGS)
        backfill.needs_backfill(cid, _MSGS)
    finally:
        lg.removeHandler(collector)
    warnings = [r for r in collector.records if r.levelno == logging.WARNING
                and cid in r.getMessage() and "crashed" in r.getMessage()]
    check(len(warnings) == 1, f"logged exactly once despite 3 needs_backfill calls: {len(warnings)}")


def test_control_stale_in_progress_under_the_cap_still_retries():
    print("\n[test] CONTROL: a crash under the cap (attempts 1 of 3) still retries normally")
    _wipe()
    cid = "r2-crash-under-cap"
    old_ts = (
        datetime.now(timezone.utc)
        - timedelta(seconds=backfill._STALE_SECONDS + 60)
    ).isoformat()
    state_path = backfill._backfill_state_path(cid)
    state_path.write_text(json.dumps({
        "conv_id": cid, "state": "in_progress", "started_at": old_ts,
        "updated_at": old_ts, "exchanges_done": 2, "exchanges_total": 10,
        "attempts": 1,
    }))
    check(backfill.needs_backfill(cid, _MSGS) is True, "stale, under the cap -> retries")


def test_env_overridable_cap_and_backoff():
    print("\n[test] the cap and the backoff base are both env-overridable "
          "(matching this module's other retry/budget knobs)")
    check(backfill._MAX_BACKFILL_ATTEMPTS == int(
        os.environ.get("COMPACTOR_BACKFILL_MAX_ATTEMPTS", "3")),
        "cap reads COMPACTOR_BACKFILL_MAX_ATTEMPTS (default 3)")
    check(backfill._BACKFILL_RETRY_BACKOFF_S == int(
        os.environ.get("COMPACTOR_BACKFILL_RETRY_BACKOFF_S", str(backfill._STALE_SECONDS))),
        "backoff base reads COMPACTOR_BACKFILL_RETRY_BACKOFF_S (default _STALE_SECONDS)")


def _all_tests():
    return [
        test_three_deterministic_failures_abandon_the_backfill_permanently,
        test_control_successful_retry_still_completes,
        test_backoff_doubles_per_attempt,
        test_needs_backfill_agrees_with_backoff_ready,
        test_stale_in_progress_at_the_cap_is_refused_and_logged_once,
        test_control_stale_in_progress_under_the_cap_still_retries,
        test_env_overridable_cap_and_backoff,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R2 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
