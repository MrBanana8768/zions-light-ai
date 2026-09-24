"""
v3.1.9.4 R4: surface summarizer.truncated_summary_count() (round 2's M1,
P15-6) and main.truncated_compaction_summary_count() (R3, this lane) at
/health/full, as `checks.truncated_summaries` — visibility only, matching
the checks.tokenizer/checks.reuse/checks.budget_margin doctrine: the value
never moves `status`.

Run: python test_v3194_r3_r4.py
"""

import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, patch

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-v3194-r3-r4-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")

import memory  # noqa: E402

memory.ensure_storage_layout()

import retrieval  # noqa: E402

retrieval.conversation_doc_count = lambda conv_id: 0

import health  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


async def _full():
    with patch("health.probe_vllm", new=AsyncMock(return_value={
        "ok": True, "latency_ms": 1.0, "models": ["m"], "error": None,
    })):
        return await health.gather_health_full("http://fake", 4096)


def full():
    return asyncio.run(_full())


# ---------------------------------------------------------------------------
# _truncated_summary_state() directly
# ---------------------------------------------------------------------------

def test_state_reads_both_real_counters():
    print("\n[test] health._truncated_summary_state() reads both real, live counters")
    before_h = summarizer.truncated_summary_count()
    before_c = main.truncated_compaction_summary_count()
    st = health._truncated_summary_state()
    check(st.get("available") is True, f"available: {st!r}")
    check(st.get("hierarchy") == before_h,
          f"hierarchy matches summarizer.truncated_summary_count() exactly: {st!r}")
    check(st.get("compaction") == before_c,
          f"compaction matches main.truncated_compaction_summary_count() exactly: {st!r}")


def test_state_reflects_a_live_increment_of_each_counter():
    print("\n[test] the state reflects a real increment of EACH counter — not a "
          "stale snapshot or a re-derived copy")
    summarizer._truncated_summary_calls += 1
    main._truncated_compaction_summary_calls += 1
    try:
        st = health._truncated_summary_state()
        check(st.get("hierarchy") == summarizer.truncated_summary_count(),
              f"hierarchy tracks the live global: {st!r}")
        check(st.get("compaction") == main.truncated_compaction_summary_count(),
              f"compaction tracks the live global: {st!r}")
    finally:
        summarizer._truncated_summary_calls -= 1
        main._truncated_compaction_summary_calls -= 1


def test_state_degrades_gracefully_when_main_is_not_loaded():
    print("\n[test] CONTROL: with main absent from sys.modules, compaction reads "
          "None with a reason rather than raising — hierarchy is unaffected "
          "(summarizer.py is imported normally, no sys.modules dance needed)")
    saved = sys.modules.pop("main", None)
    try:
        st = health._truncated_summary_state()
        check(st.get("available") is True, f"still available (hierarchy alone is enough): {st!r}")
        check(st.get("compaction") is None, f"compaction is None: {st!r}")
        check(bool(st.get("compaction_reason")), f"a reason is given: {st!r}")
        check(st.get("hierarchy") == summarizer.truncated_summary_count(),
              "hierarchy is unaffected by main's absence")
    finally:
        if saved is not None:
            sys.modules["main"] = saved


def test_state_degrades_gracefully_when_the_function_is_missing():
    print("\n[test] CONTROL: an older/different main.py build with no "
          "truncated_compaction_summary_count() reads None with a reason, not a crash")
    had = hasattr(main, "truncated_compaction_summary_count")
    orig = getattr(main, "truncated_compaction_summary_count", None)
    if had:
        del main.truncated_compaction_summary_count
    try:
        st = health._truncated_summary_state()
        check(st.get("compaction") is None, f"compaction is None: {st!r}")
        check("not present in this build" in (st.get("compaction_reason") or ""),
              f"reason names the missing function: {st!r}")
    finally:
        if had:
            main.truncated_compaction_summary_count = orig


# ---------------------------------------------------------------------------
# Wired into gather_health_full, visibility only
# ---------------------------------------------------------------------------

def test_reaches_health_full_checks_dict():
    print("\n[test] gather_health_full's checks dict carries 'truncated_summaries'")
    r = full()
    check("truncated_summaries" in r["checks"],
          "*** the key is present")
    check(r["checks"]["truncated_summaries"].get("available") is True,
          f"and it is the live state: {r['checks']['truncated_summaries']!r}")


def test_a_high_count_does_not_move_status():
    print("\n[test] R4: a nonzero (even large) truncated-summary count never "
          "moves status — visibility only, same doctrine as checks.reuse/"
          "checks.budget_margin")
    summarizer._truncated_summary_calls += 500
    main._truncated_compaction_summary_calls += 500
    try:
        r = full()
        check(r["checks"]["truncated_summaries"]["hierarchy"] >= 500,
              f"the large count DOES reach the payload: {r['checks']['truncated_summaries']!r}")
        check(r["status"] == "ok",
              f"*** status is still 'ok' despite the large count: {r['status']!r}")
        check(not any("truncated" in reason.lower() for reason in r["status_reasons"]),
              f"*** no status_reasons line mentions it either: {r['status_reasons']!r}")
    finally:
        summarizer._truncated_summary_calls -= 500
        main._truncated_compaction_summary_calls -= 500


def test_control_a_zero_count_also_does_not_move_status():
    print("\n[test] CONTROL: a zero count (the healthy case) also reads status='ok'")
    r = full()
    check(r["status"] == "ok", f"status: {r['status']!r}")


def _all_tests():
    return [
        test_state_reads_both_real_counters,
        test_state_reflects_a_live_increment_of_each_counter,
        test_state_degrades_gracefully_when_main_is_not_loaded,
        test_state_degrades_gracefully_when_the_function_is_missing,
        test_reaches_health_full_checks_dict,
        test_a_high_count_does_not_move_status,
        test_control_a_zero_count_also_does_not_move_status,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R4 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
