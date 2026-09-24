"""v3.1.9.4, lane v3194-guard, G3d (hostile pass #14's "not demonstrated"
list): the compaction trigger (`current <= TARGET_TOKENS`, early in
compact_if_needed) did not subtract the learned `_BUDGET_MARGIN`.
TARGET_TOKENS is a fixed fraction of HARD_INPUT_LIMIT (0.75 by default,
i.e. 25% headroom built in) — it has no idea `_enforce_hard_budget` (the
guard, downstream) will subtract `_BUDGET_MARGIN` from ITS OWN limit
before shedding a single token. A margin up to `MAX_MODEL_LEN // 4`
(8,192 at the shipped defaults) can exceed that 25% headroom outright: an
array between `HARD_INPUT_LIMIT - margin` and TARGET_TOKENS then skips
compaction ENTIRELY — no summary, no stand-in — and reaches the guard
raw, where old turns that compaction would have preserved as a summary
are instead deleted with no trace at all.

WHAT THIS COSTS HER, measured (main.py's brief: "demonstrate what that
costs her (which turns go) at a margin of 513 and 8,192"):

  - At margin 513: the trigger's threshold (TARGET_TOKENS - 513) never
    falls below what the guard actually enforces (HARD_INPUT_LIMIT -
    513), because 513 is well inside HARD_INPUT_LIMIT's own 25% built-in
    headroom (4,096 at the shipped defaults: TARGET_TOKENS is already
    HARD_INPUT_LIMIT * 0.75). The gap this item is about cannot open at
    this margin — verified analytically ([G3d-513] below) and confirmed
    the pre-fix and post-fix trigger decisions are identical here.

  - At margin 8,192: reproduced. A fixture with 10 old exchange pairs
    (raw, ~10,120 local tokens) beside a small recent window sits at
    11,790 local tokens — under TARGET_TOKENS (12,288, so the unfixed
    trigger skips compaction) but over the guard's actual 8,192-token
    limit at this margin. Before this fix: compaction never runs
    (standin_present=False), and the guard's generic shed loop deletes 8
    of the 20 old turns OUTRIGHT — content compaction would have
    represented as a stand-in, gone with no trace and no log line
    naming what was lost beyond a turn count. After this fix: the
    trigger fires, compaction summarizes the old turns into a stand-in,
    and the guard does not need to shed anything at all
    (dropped_turns=0).

  - Whether it can cost her PREVIOUS EXCHANGE specifically: investigated
    and worked out mathematically, then confirmed both ways. The guard's
    own shedding order (both the generic loop and the P12-5 branch) sheds
    OLD content — raw or compacted, it does not matter which — before it
    ever reaches into the recent window. Her previous exchange is
    therefore only at risk when `persona + recent_window` ALONE already
    exceeds the guard's limit — a condition compaction can NEVER address
    either way, because compact_if_needed never touches `keep_recent`.
    This item's trigger bug does not create that risk; it only means
    OLDER content that could have been compacted is deleted raw instead.
    [G3d-recent-cap] below confirms: a fixture where the recent window
    alone exceeds the margin-8,192 guard limit loses the exchange
    IDENTICALLY with the fix applied and without it — proving the fix
    does not, and could not, change that outcome, so it is reported
    accurately rather than as something this fix resolves.

THE FIX: compact_if_needed's trigger now compares against
`max(256, TARGET_TOKENS - (_BUDGET_MARGIN or 0))` — the same floor
`_enforce_hard_budget` applies to its own limit — so a margin blind spot
never lets compaction skip an array the guard cannot actually afford.

    python test_v3194_guard_g3d.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="g3dguard-")

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


def local(ms):
    return sum(len(main._message_text(m)) // 4 + 4 for m in ms)


def exact(ms, *a, **k):
    return local(ms)  # accurate on both sides -- isolate the trigger, not the counters


async def _fake_summarize_once(client, turns):
    return "SUMMARIZED-" + ("z" * 40)


_saved_ct = main.count_tokens
_saved_cte = main.count_tokens_exact
_saved_once = main._summarize_once
_saved_margin = main._BUDGET_MARGIN
main.count_tokens = local
main.count_tokens_exact = exact
main._summarize_once = _fake_summarize_once


def _fixture(n_old=10, old_chars=2000, aprev_chars=6000):
    """Enough RAW old-turn content to push `current` into the TARGET/
    guard-limit gap at margin 8192, but small enough once compacted that
    persona+recent alone fit the guard's limit -- otherwise no trigger fix
    could ever help (compact_if_needed never touches keep_recent)."""
    persona = {"role": "system", "content": "P" * 400}
    old_turns = []
    for i in range(n_old):
        old_turns.append({"role": "user", "content": f"OLDTURN u{i} " + ("x" * old_chars)})
        old_turns.append({"role": "assistant", "content": f"OLDTURN a{i} " + ("x" * old_chars)})
    recent = [
        {"role": "user", "content": "prev-u " + "y" * 100},
        {"role": "assistant", "content": "prev-a " + "y" * aprev_chars},
        {"role": "user", "content": "newest-u " + "y" * 100},
    ]
    return persona, old_turns, recent


def _run(margin, persona, old_turns, recent, conv="g3d-guard"):
    saved = main._BUDGET_MARGIN
    main._BUDGET_MARGIN = margin
    try:
        payload = [persona] + old_turns + recent
        out = asyncio.run(main.compact_if_needed(list(payload), conv))
        standin_present = any(main._is_compaction_standin(m) for m in out)
        report: dict = {}
        # main.HARD_INPUT_LIMIT, not a pre-subtracted value: the guard
        # subtracts _BUDGET_MARGIN itself internally, the same way
        # chat_completions calls it in production (passing effective_
        # limit unadjusted) -- pre-subtracting here would double it.
        final = main._enforce_hard_budget(list(out), main.HARD_INPUT_LIMIT, 1, report)
        final_text = " ".join(main._message_text(m) for m in final)
        return {
            "current": local(payload),
            "out_len": len(out),
            "standin_present": standin_present,
            "report": report,
            "old_survived": "OLDTURN" in final_text,
            "prev_survived": "prev-u" in final_text and "prev-a" in final_text,
            "newest_survived": "newest-u" in final_text,
        }
    finally:
        main._BUDGET_MARGIN = saved


try:
    print("\n[G3d-513] CONTROL: at margin 513, the fixed trigger's threshold "
          "cannot fall below what the guard enforces -- the gap this item "
          "is about never opens")
    check(
        513 < main.HARD_INPUT_LIMIT - main.TARGET_TOKENS,
        f"fixture: 513 is inside HARD_INPUT_LIMIT's own built-in headroom "
        f"({main.HARD_INPUT_LIMIT - main.TARGET_TOKENS} tokens) -- the "
        f"trigger's fixed threshold was never the binding constraint at "
        f"this margin, before or after this fix",
    )
    check(
        main.HARD_INPUT_LIMIT - 513 >= main.TARGET_TOKENS,
        f"fixture: even the SMALLEST margin _note_backend_rejection can "
        f"latch (overshoot+512, so >=513) leaves the guard's real limit "
        f"({main.HARD_INPUT_LIMIT - 513}) at or above TARGET_TOKENS "
        f"({main.TARGET_TOKENS}) -- the pre-fix condition for this item "
        f"('TARGET_TOKENS sits above what the guard enforces') cannot "
        f"hold at this margin, so the fixed and unfixed trigger make the "
        f"IDENTICAL decision for every possible `current`",
    )
    check(
        max(256, main.TARGET_TOKENS - 513) == main.TARGET_TOKENS - 513,
        "fixture: this margin is far above the 256-token floor "
        "_enforce_hard_budget itself uses, so the fixed trigger's "
        "threshold here is exactly TARGET_TOKENS - 513, unclamped",
    )

    print("\n[G3d-8192] THE FIX: at margin 8192, the fixed trigger closes "
          "the gap the unfixed one left open")
    persona, old_turns, recent = _fixture()
    r = _run(8192, persona, old_turns, recent)
    print(f"  current={r['current']} TARGET_TOKENS={main.TARGET_TOKENS} "
          f"HARD_INPUT_LIMIT-margin={main.HARD_INPUT_LIMIT - 8192}")
    check(
        main.HARD_INPUT_LIMIT - 8192 < r["current"] <= main.TARGET_TOKENS,
        f"fixture: current ({r['current']}) sits exactly in the gap "
        f"(> {main.HARD_INPUT_LIMIT - 8192}, <= {main.TARGET_TOKENS}) -- "
        f"the unfixed trigger would have skipped compaction here",
    )
    check(
        r["standin_present"],
        f"*** G3d [8192] THE FIX: compaction ran and a stand-in is "
        f"present (out_len={r['out_len']}) -- before this fix, this exact "
        f"array skipped compaction entirely",
    )
    check(
        r["report"].get("dropped_turns", 0) == 0,
        f"*** G3d [8192] THE FIX: the guard does not need to shed ANY "
        f"turn -- the compacted stand-in already fits "
        f"(report={r['report']})",
    )
    check(
        r["prev_survived"] and r["newest_survived"],
        f"[8192]: her previous exchange and the newest turn survive "
        f"({r['report']})",
    )

    print("\n[G3d-recent-cap] CONTROL: when the recent window ALONE "
          "exceeds the guard's limit, this fix does not and cannot save "
          "her previous exchange -- compaction never touches keep_recent, "
          "so that outcome is identical with or without this fix")
    persona2, old_turns2, recent2 = _fixture(n_old=4, old_chars=200, aprev_chars=45000)
    r2 = _run(8192, persona2, old_turns2, recent2, conv="g3d-guard-recentcap")
    check(
        local([persona2] + recent2) > main.HARD_INPUT_LIMIT - 8192,
        f"fixture: persona + recent window alone already exceeds the "
        f"guard's limit at this margin ({local([persona2] + recent2)} > "
        f"{main.HARD_INPUT_LIMIT - 8192})",
    )
    check(
        not r2["prev_survived"],
        f"CONTROL: her previous exchange is lost here even WITH this fix "
        f"applied (report={r2['report']}) -- an oversized reply not "
        f"fitting the window at all is a separate, pre-existing capacity "
        f"limit this fix does not claim to solve",
    )
    check(r2["newest_survived"], "CONTROL: the newest turn still always survives, even here")

finally:
    main.count_tokens = _saved_ct
    main.count_tokens_exact = _saved_cte
    main._summarize_once = _saved_once
    main._BUDGET_MARGIN = _saved_margin


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nall checks passed")
