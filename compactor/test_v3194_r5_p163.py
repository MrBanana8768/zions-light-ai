"""
v3.1.9.4 R5 (hostile pass #16 follow-up): P16-3.

`main._shed_last_resort` ran steps 2 (spend injected memory) and 3 (spend
the recent window, except the newest — U_prev/A_prev) back to back in the
SAME round, both decided from that round's `per` ESTIMATES with no
measurement in between. A prior round's rescale (`per-item estimate *=
actual/estimated`, applied uniformly to every remaining class) can
under-price memory relative to its real cost when an EARLIER round's
turn-drops were over-priced — round 2 then "spends" memory on paper in
step 2, comes up short of that round's (badly-scaled) target, and step 3
drops U_prev/A_prev in the SAME round when the real, measured saving from
steps 1+2 together would already have met the target — breaking the
owner's standing rule ("her own previous exchange ... is kept over
injected memory; the newest turn is never dropped") in the one function
written to enforce it.

Fix: once step 2 has spent anything this round, measure steps 1+2's REAL
combined effect before step 3 (or step 4) is allowed to touch anything —
at most one extra `measure` call per round, still inside
`_G3A_MEASURE_CAP`.

Direct call to `_shed_last_resort`, the same way test_v3194_guard_g3a.py
does. `measure` returns the TRUE token count; `per` starts as the true
per-item cost, then old turns (indices 3-6) are over-priced by a factor —
this is exactly the reviewer's own reproduction fixture
(SP\\p16\\test_p16_g3a_same_round.py), formalized here with assertions, a
CONTROL, and a sweep. Synthetic text only.

Run: python test_v3194_r5_p163.py
"""

import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-v3194-r5-p163-")
import memory  # noqa: E402
memory.ensure_storage_layout()
import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def _build(over_factor):
    """9 real turns plus 3 system blocks (caller prompt, facts, retrieval).
    KEEP_RECENT_TURNS=4 aligns the recent window to the last 4 non-system
    messages (old-u-2, old-a-2, PREV-U, PREV-A) — PREV-U/PREV-A sit INSIDE
    that window, one turn short of the newest, which is exactly the shape
    step 3 (never the newest) is allowed to touch.

    `true` is each message's real size; `per` starts equal to it, then
    indices 3-6 (the two old exchanges ABOVE the aligned floor) are scaled
    by `over_factor` — an over-estimate, the documented shape (this repo's
    own compaction-standin undercount can produce it; the reviewer's own
    finding shows it can also arrive via a PRIOR round's uniform rescale).
    Facts/retrieval (indices 1, 2) are priced exactly, matching the
    reviewer's [G3a-mem]/[G3a-order] baseline.
    """
    msgs = [
        {"role": "system", "content": "caller system prompt"},
        {"role": "system", "content": "FACTS block synthetic"},
        {"role": "system", "content": "RETRIEVAL block synthetic"},
        {"role": "user", "content": "old-u-1"},
        {"role": "assistant", "content": "old-a-1"},
        {"role": "user", "content": "old-u-2"},
        {"role": "assistant", "content": "old-a-2"},
        {"role": "user", "content": "PREV-U"},
        {"role": "assistant", "content": "PREV-A"},
        {"role": "user", "content": "NEWEST-U"},
    ]
    true = [100, 800, 800, 100, 100, 100, 100, 300, 300, 100]
    per = list(true)
    for i in (3, 4, 5, 6):
        per[i] = true[i] * over_factor
    size = {m["content"]: t for m, t in zip(msgs, true)}

    def measure(ms):
        return sum(size[m["content"]] for m in ms), "exact"

    return msgs, per, measure


def _run(over_factor, limit=1600):
    msgs, per, measure = _build(over_factor)
    running, _ = measure(msgs)
    out, _per, r, _c, d, sd = main._shed_last_resort(
        msgs, per, running, limit, 1, True, "exact", 0, 0, measure,
    )
    kept = {m["content"] for m in out}
    return {
        "running": r,
        "dropped": d,
        "sys_dropped": sd,
        "prev_u": "PREV-U" in kept,
        "prev_a": "PREV-A" in kept,
        "newest": "NEWEST-U" in kept,
        "facts": "FACTS block synthetic" in kept,
        "retrieval": "RETRIEVAL block synthetic" in kept,
    }


def test_control_accurate_estimates_keep_prev_exchange():
    print("\n[test] CONTROL: accurate per-class estimates (over_factor=1) — "
          "memory pays, U_prev/A_prev survive, exactly as G3a always did")
    r = _run(1)
    check(r["running"] <= 1600, f"under/at limit: {r['running']}")
    check(r["prev_u"] and r["prev_a"], f"PREV-U/PREV-A kept: {r}")
    check(r["newest"], "the newest turn is never touched")


def test_control_mild_skew_still_keeps_prev_exchange():
    print("\n[test] CONTROL: a 1.25x-2x skew (below the level that breaks "
          "the old code) still keeps U_prev/A_prev")
    for factor in (1.25, 1.5, 2):
        r = _run(factor)
        check(r["prev_u"] and r["prev_a"],
              f"over_factor={factor}: PREV-U/PREV-A kept: {r}")


def test_severe_skew_no_longer_drops_prev_exchange():
    print("\n[test] a 3x-4x old-turn over-price — the reviewer's own "
          "'lost at 3-4x' reproduction — no longer drops U_prev/A_prev")
    for factor in (3, 4):
        r = _run(factor)
        check(r["prev_u"] and r["prev_a"],
              f"over_factor={factor}: PREV-U/PREV-A kept (was lost pre-fix): {r}")
        # Memory paid instead — the owner's rule, not merely "nothing was
        # dropped": both facts and retrieval are gone rather than her
        # previous exchange.
        check(not r["facts"] and not r["retrieval"],
              f"over_factor={factor}: memory absorbed the shortfall instead: {r}")
        check(r["running"] <= 1600,
              f"over_factor={factor}: still converged to at/under the limit: {r}")


def test_checkpoint_stops_as_soon_as_steps_one_two_real_effect_suffices():
    print("\n[test] the mid-round checkpoint's REAL measurement, not the "
          "skewed estimate, decides whether step 3 is needed at all")
    r = _run(4)
    # steps 1+2's true combined saving (facts 800 + retrieval 800 dropped
    # by round 2, on top of round 1's real 400) brings this well under the
    # 1600 limit on its own -- the checkpoint should stop there rather than
    # additionally spending U_prev/A_prev for a target that measurement
    # already shows is met.
    check(r["running"] < 1600,
          f"real measurement found room to spare, not just barely at the limit: {r}")


def _all_tests():
    return [
        test_control_accurate_estimates_keep_prev_exchange,
        test_control_mild_skew_still_keeps_prev_exchange,
        test_severe_skew_no_longer_drops_prev_exchange,
        test_checkpoint_stops_as_soon_as_steps_one_two_real_effect_suffices,
    ]


if __name__ == "__main__":
    print("KEEP_RECENT_TURNS", main.KEEP_RECENT_TURNS)
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R5 P16-3 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
