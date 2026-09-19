"""
v3.1.9.4 R6 (hostile pass #17): P17-2 and P17-3.

`main._shed_last_resort`'s R5/P16-3 checkpoint (a mid-round `measure` call
between step 2 and step 3, added so step 3 would not drop U_prev/A_prev on
a stale per-item estimate) only covers the step2->step3 transition. Two
residual defects:

P17-2: once that checkpoint has run (or was never needed because step 2's
own estimate already looked sufficient), steps 3 and 4 can STILL run back
to back in the SAME round on stale, rescaled `per`-estimates with no
measurement between them — so step 4 can spend the PROTECTED compaction
stand-in even though the REAL saving from step 3 alone (dropping
U_prev/A_prev) already meets the limit. Reproduced with the reviewer's own
"protected stand-in" fixture (`SP\\p17\\probe_p163_standin.py`): a 3x-6x
old-turn over-price converges to 200 tokens with the stand-in GONE, when
dropping U_prev/A_prev alone reaches 700 (comfortably under an 800 limit).

Fix: a SECOND checkpoint, symmetric to the first, runs after step 3 and
before step 4 (same gating: something changed, the round's estimated
target is not yet met, this is not the last round, and there is
measurement-cap room). On a real measurement <= limit, step 4 is skipped
outright. On a real measurement still over limit, the round's remaining
target/freed are RE-BASED on the measured gap (not the stale per-item
estimate) — the reviewer's own suggested two-line fix, now applied at
both checkpoints and reused by the end-of-round measurement (via a trial
snapshot comparison) to avoid a redundant third measure call.

P17-3: gating a checkpoint on `_g3a_measures < _G3A_MEASURE_CAP` lets it
fire on the LAST round this pass is allowed to run and push the total
measurement count one past `_G3A_MEASURE_CAP` — contradicting the
function's own docstring ("still inside `_G3A_MEASURE_CAP`"). Fixed by
also gating both checkpoints on `not _g3a_last_round`: the last round
already spends everything eligible unconditionally (steps 2/4's own
`_g3a_last_round` bypass), so a checkpoint there decides nothing and
firing it anyway is exactly how the cap got exceeded.

Direct calls to `_shed_last_resort`, the same way test_v3194_r5_p163.py
and test_v3194_guard_g3a.py do. Synthetic content only.

Run: python test_v3194_r6_p172_p173.py
"""

import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-v3194-r6-p172-")
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


# ---------------------------------------------------------------------------
# P17-2: the reviewer's own "protected stand-in" fixture
# (SP\p17\probe_p163_standin.py), formalized here with assertions.
# ---------------------------------------------------------------------------

SI = main.COMPACTION_SUMMARY_HEADER + " synthetic stand-in"


def _build_standin_fixture(over_factor):
    """11 messages: caller prompt, the protected compaction stand-in,
    facts, retrieval, two old exchanges (over-priced by `over_factor`,
    modeling a prior round's rescale skew — see P16-3's own fixture),
    PREV-U/PREV-A (the recent window's older-than-newest pair), NEWEST.

    `true` is the real size of each message; `measure` returns the exact
    sum. Dropping U_prev/A_prev ALONE after the old turns + facts +
    retrieval are gone reaches 700 real tokens, comfortably under an
    800-token limit — the stand-in should never need to go.
    """
    msgs = [
        {"role": "system", "content": "caller system prompt"},
        {"role": "system", "content": SI},
        {"role": "system", "content": "FACTS block synthetic"},
        {"role": "system", "content": "RETRIEVAL block synthetic"},
        {"role": "user", "content": "old-u-1"},
        {"role": "assistant", "content": "old-a-1"},
        {"role": "user", "content": "old-u-2"},
        {"role": "assistant", "content": "old-a-2"},
        {"role": "user", "content": "PREV-U"},
        {"role": "assistant", "content": "PREV-A"},
        {"role": "user", "content": "NEWEST"},
    ]
    true = [100, 500, 800, 800, 100, 100, 100, 100, 300, 300, 100]
    per = list(true)
    for i in (4, 5, 6, 7):
        per[i] = true[i] * over_factor
    size = {m["content"]: t for m, t in zip(msgs, true)}
    calls = [0]

    def measure(ms):
        calls[0] += 1
        return sum(size[m["content"]] for m in ms), "exact"

    return msgs, per, measure, calls, sum(true)


def _run_standin(over_factor, limit=800):
    msgs, per, measure, calls, start = _build_standin_fixture(over_factor)
    out, _per, r, _c, d, sd = main._shed_last_resort(
        msgs, per, start, limit, 1, True, "exact", 0, 0, measure,
    )
    kept = {m["content"] for m in out}
    return {
        "running": r,
        "calls": calls[0],
        "standin_kept": SI in kept,
        "prev_u": "PREV-U" in kept,
        "prev_a": "PREV-A" in kept,
        "newest": "NEWEST" in kept,
    }


def test_p172_control_mild_skew_never_needed_the_checkpoint():
    print("\n[test] P17-2 CONTROL: 1x-2x old-turn over-price never needed "
          "either checkpoint — the stand-in survives, same as before this "
          "fix")
    for f in (1, 2):
        r = _run_standin(f)
        check(r["standin_kept"], f"over_factor={f}: stand-in kept: {r}")
        check(r["running"] <= 800, f"over_factor={f}: converged: {r}")
        check(r["newest"], f"over_factor={f}: newest turn never touched")


def test_p172_severe_skew_no_longer_drops_the_protected_standin():
    print("\n[test] P17-2: a 3x-6x old-turn over-price — the reviewer's own "
          "reproduction — no longer spends the protected stand-in when "
          "dropping U_prev/A_prev alone already fits")
    for f in (3, 4, 6):
        r = _run_standin(f)
        check(r["standin_kept"], f"over_factor={f}: stand-in KEPT (was lost pre-fix): {r}")
        check(not r["prev_u"] and not r["prev_a"],
              f"over_factor={f}: U_prev/A_prev pay instead: {r}")
        check(r["running"] == 700,
              f"over_factor={f}: converges to the real 700-token floor, not "
              f"the stand-in's extra headroom: {r}")
        check(r["newest"], f"over_factor={f}: newest turn never touched")


def test_p172_measurement_cap_still_respected():
    print("\n[test] P17-2: the extra checkpoint costs at most one more "
          "measure() call per round and never exceeds _G3A_MEASURE_CAP")
    for f in (1, 2, 3, 4, 6):
        r = _run_standin(f)
        check(r["calls"] <= main._G3A_MEASURE_CAP,
              f"over_factor={f}: calls={r['calls']} <= cap "
              f"{main._G3A_MEASURE_CAP}")


# ---------------------------------------------------------------------------
# P17-3: the reviewer's pinned-count probe (SP\p17\probe_p163_cap.py) —
# measure() pinned so it never converges, so the round loop always runs the
# full _G3A_MEASURE_CAP worth of rounds and any checkpoint firing on the
# LAST one pushes the total past the cap.
# ---------------------------------------------------------------------------

def _run_pinned(n_old_pairs, limit=10000, k_over_limit=3):
    msgs = [
        {"role": "system", "content": "caller"},
        {"role": "system", "content": "FACTS"},
        {"role": "system", "content": "RETRIEVAL"},
    ]
    for i in range(n_old_pairs):
        msgs += [
            {"role": "user", "content": f"old-u-{i}"},
            {"role": "assistant", "content": f"old-a-{i}"},
        ]
    msgs += [
        {"role": "user", "content": "w-u"}, {"role": "assistant", "content": "w-a"},
        {"role": "user", "content": "PREV-U"}, {"role": "assistant", "content": "PREV-A"},
        {"role": "user", "content": "NEWEST"},
    ]
    per = [5] * len(msgs)
    calls = [0]

    def measure(ms):
        calls[0] += 1
        return limit + k_over_limit, "exact"  # never converges

    out, _per, r, _c, d, sd = main._shed_last_resort(
        msgs, per, limit + k_over_limit, limit, 1, True, "exact", 0, 0, measure,
    )
    return calls[0], r, d, sd


def test_p173_measurement_count_never_exceeds_the_cap():
    print("\n[test] P17-3: measure() pinned above the limit forever (the "
          "'never converges' shape test_budget_guard.py itself uses) — "
          "the checkpoint(s) must never push the total past "
          "_G3A_MEASURE_CAP, for a sweep of old-pair counts")
    over_cap_hits = []
    for n in range(0, 12):
        calls, running, dropped, sys_dropped = _run_pinned(n)
        if calls > main._G3A_MEASURE_CAP:
            over_cap_hits.append((n, calls))
        check(calls <= main._G3A_MEASURE_CAP,
              f"old_pairs={n}: calls={calls} <= cap {main._G3A_MEASURE_CAP}")
    check(not over_cap_hits, f"no fixture exceeded the cap: {over_cap_hits}")


def test_p173_seven_old_pairs_is_the_reviewers_own_pinned_reproduction():
    print("\n[test] P17-3: the reviewer's own named case — 7 old pairs, "
          "head made 7 calls where base made 6 — now stays at the cap (6)")
    calls, running, dropped, sys_dropped = _run_pinned(7)
    check(calls == main._G3A_MEASURE_CAP,
          f"exactly {main._G3A_MEASURE_CAP} calls, not "
          f"{main._G3A_MEASURE_CAP + 1}: got {calls}")


def test_p173_control_fixture_that_converges_is_unaffected():
    print("\n[test] P17-3 CONTROL: a fixture that DOES converge (p16's own "
          "over-estimate shape, from test_v3194_r5_p163.py) is unaffected "
          "by the last-round checkpoint gate — it never gets near the cap "
          "in the first place")
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
        per[i] = true[i] * 4
    size = {m["content"]: t for m, t in zip(msgs, true)}
    calls = [0]

    def measure(ms):
        calls[0] += 1
        return sum(size[m["content"]] for m in ms), "exact"

    running = sum(true)
    out, _per, r, _c, d, sd = main._shed_last_resort(
        msgs, per, running, 1600, 1, True, "exact", 0, 0, measure,
    )
    kept = {m["content"] for m in out}
    check(r <= 1600, f"converged: {r}")
    check(calls[0] < main._G3A_MEASURE_CAP,
          f"converged well before the cap: {calls[0]} calls")
    check("PREV-U" in kept and "PREV-A" in kept, f"U_prev/A_prev survive: {kept}")


def _all_tests():
    return [
        test_p172_control_mild_skew_never_needed_the_checkpoint,
        test_p172_severe_skew_no_longer_drops_the_protected_standin,
        test_p172_measurement_cap_still_respected,
        test_p173_measurement_count_never_exceeds_the_cap,
        test_p173_seven_old_pairs_is_the_reviewers_own_pinned_reproduction,
        test_p173_control_fixture_that_converges_is_unaffected,
    ]


if __name__ == "__main__":
    print("CAP", main._G3A_MEASURE_CAP, "KEEP_RECENT_TURNS", main.KEEP_RECENT_TURNS)
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R6 P17-2/P17-3 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
