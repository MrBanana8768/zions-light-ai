"""v3.1.9.4, lane v3194-guard, G3a (hostile pass #14's "not demonstrated"
list; REWORKED after coordinator review found the first draft unsafe): the
post-round forced drop in _enforce_hard_budget (the "v3.1 D3" block,
main.py, after the six-round loop) only ever touched SYSTEM content. If
the six rounds exhaust their budget with non-newest TURNS still present,
this block went straight for the compaction stand-in without ever
shedding those turns first -- "older turns first, never the newest" (this
guard's own stated shedding order) did not apply to it.

p14's own hostile pass could not construct a payload that exhausts six
rounds with turns above the newest still present ("not demonstrated
either way"). [G3a] below does: a stand-in whose LOCAL price reads far
under its true cost (this repo's own documented 34-51% undercount on
LLM-generated prose, realistic for a real compacted summary) skews the
single global `scale` _enforce_hard_budget computes ONCE at the top, so
the per-round arithmetic OVER-prices ordinary turns and the round loop's
own `_cut` estimate stops short every round, converging geometrically
without ever finishing.

THE FIX IS `_shed_last_resort` (main.py, extracted out of
`_enforce_hard_budget`'s body so it can be called and tested directly,
isolated from the six-round loop that decides WHEN it runs but never HOW):
four explicit priority steps, oldest turns above the protected recent
window first, then spendable injected memory, then the recent window's
own turns except the newest, then the protected stand-in truly last.

THE FIRST DRAFT WAS WRONG: it shed every non-newest turn, oldest first,
before ever reaching for injected memory at all — so on an array that
still carried facts or retrieval when it ran, it dropped her previous
exchange AHEAD of memory, exactly what P12-5 and the owner's standing
rule forbid ("her previous exchange is kept over injected memory; the
newest turn is never dropped"). [G3a-mem] below is the reproduction of
THAT defect and the proof it is fixed: facts and retrieval present
alongside a leftover old turn, and memory must pay, not the exchange.

[G3a-order] proves the full four-step priority precisely, at three
thresholds of one fixture. [G3a-linear] proves the pass is linear in the
turn count, not quadratic (test_p5_guard.py's own `[4]` technique — a
message subclass that counts `.get("role")` calls, a property of the
CODE PATH, not wall-clock timing noisy under this shared VM's load).
[G3a-cap] proves the number of exact measurements this pass can spend is
bounded even when the measurement itself is adversarially useless (an
exact stub returning the SAME fixed value regardless of content --
test_budget_guard.py's own technique, which is what exposed a second
defect in the first corrected draft: steps 2/4 were ALSO target-limited
like the turn-shedding steps, so a measurement that never reflects
content made them spend only one block per round and exhaust the
six-round cap with memory still held -- the exact "guard holding memory
it was allowed to spend" bug v3.1 D3 exists to prevent. The coordinator's
fix at the time made steps 2 and 4 UNCONDITIONAL once reached instead).

v3.1.9.4, lane v3194-r3 (R6): steps 2 and 4 are TARGET-LIMITED again,
like steps 1 and 3 — the unconditional shape above reintroduced exactly
what the forced drop THIS pass replaced was written to avoid, in that
forced drop's own comment: "Dropping EVERYTHING is wasteful: measured by
review, a payload over by 550 tokens lost persona, facts and summary
when one 1500-token block covered it." The round-cap risk [G3a-cap]
above exists to bound is real under an adversarial measurement, but this
pass's own cross-round rescale (which corrects every remaining per-item
estimate by measured-vs-estimated after each round — see [G3a-rescale]
below) is the same mechanism steps 1 and 3 have relied on for that risk
since G3a shipped; [G3a-unconditional] below is rewritten to pin the
MINIMUM instead of the maximum: two spendable blocks, only one needed,
and only that one is dropped.

    python test_v3194_guard_g3a.py
"""

import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="g3aguard-")

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


def _texts(msgs):
    return [m.get("content") for m in msgs]


OLD_TAG = "OLDTURN"


def _exact_standin_4x(ms, *a, **k):
    """The stand-in reads 4x low locally (a realistic ratio for this
    model's assistant-authored prose, per count_tokens_exact's own
    docstring); everything else (turns, persona) prices exactly at its
    local estimate."""
    total = 0
    for m in ms:
        txt = main._message_text(m)
        lp = len(txt) // 4 + 4
        total += lp * 4 if main.COMPACTION_SUMMARY_HEADER in txt else lp
    return total


def _fixture(n_old=60, standin_chars=80000):
    standin = {
        "role": "system",
        "content": main.COMPACTION_SUMMARY_HEADER + "\n" + ("S" * standin_chars),
    }
    persona = {"role": "system", "content": "P" * 200}
    old_turns = []
    for i in range(n_old):
        old_turns.append({"role": "user", "content": f"{OLD_TAG} u{i} " + ("x" * 400)})
        old_turns.append({"role": "assistant", "content": f"{OLD_TAG} a{i} " + ("x" * 400)})
    recent = [
        {"role": "user", "content": "prev-u " + "y" * 100},
        {"role": "assistant", "content": "prev-a " + "y" * 100},
        {"role": "user", "content": "newest-u " + "y" * 100},
    ]
    return persona, standin, old_turns, recent


_saved_count_tokens = main.count_tokens
_saved_count_tokens_exact = main.count_tokens_exact
main.count_tokens = local
main.count_tokens_exact = _exact_standin_4x

try:
    # -------------------------------------------------------------------
    # [G3a] THE FIX: cutting old turns alone is enough (the limit sits
    # between "everything but the standin fits" and "the standin alone
    # would also fit") -- the guard must reach that state WITHOUT ever
    # touching the stand-in. End to end, through the REAL
    # _enforce_hard_budget (six-round loop included), proving
    # `_shed_last_resort` is correctly wired in, not just correct in
    # isolation.
    # -------------------------------------------------------------------
    print("\n[G3a] the post-round forced drop sheds old turns before ever "
          "reaching for the compaction stand-in")
    persona, standin, old_turns, recent = _fixture(n_old=60, standin_chars=80000)
    payload = [persona, standin] + old_turns + recent
    standin_exact = _exact_standin_4x([standin])
    persona_exact = _exact_standin_4x([persona])
    recent_exact = _exact_standin_4x(recent)
    limit = standin_exact + persona_exact + recent_exact + 2000
    check(
        limit < _exact_standin_4x(payload),
        f"fixture: the full payload does not already fit (limit={limit}, "
        f"total={_exact_standin_4x(payload)})",
    )
    report: dict = {}
    out = main._enforce_hard_budget(list(payload), limit, 1, report)
    out_text = " ".join(main._message_text(m) for m in out)
    standin_survived = any(main._is_compaction_standin(m) for m in out)
    old_turns_survived = OLD_TAG in out_text
    prev_survived = "prev-u" in out_text and "prev-a" in out_text
    newest_survived = "newest-u" in out_text
    check(report.get("fits") is True, f"the guard fits the payload ({report})")
    check(
        standin_survived,
        f"*** THE FIX: the compaction stand-in survives — cutting old "
        f"turns alone was enough (report={report})",
    )
    check(
        report.get("dropped_turns", 0) > 0,
        f"fixture: old turns WERE the thing shed to make room, not just "
        f"left alone because the payload already fit "
        f"(dropped_turns={report.get('dropped_turns')})",
    )
    check(
        old_turns_survived,
        "fixture: SOME old turns still remain — the fix stops cutting "
        "once it fits, it does not need to shed every last one",
    )
    check(prev_survived, f"her previous exchange survives ({report})")
    check(newest_survived, "the newest turn always survives")

    # -------------------------------------------------------------------
    # [G3a-CONTROL] when NOTHING (all old turns gone) can make it fit, the
    # guard still spends the stand-in — "last resort" must not become
    # "never". The newest turn still always survives.
    # -------------------------------------------------------------------
    print("\n[G3a-CONTROL] with every old turn already shed and still over "
          "budget, the guard still spends the stand-in")
    persona2, standin2, old_turns2, recent2 = _fixture(n_old=60, standin_chars=400000)
    payload2 = [persona2, standin2] + old_turns2 + recent2
    persona2_exact = _exact_standin_4x([persona2])
    recent2_exact = _exact_standin_4x(recent2)
    # Below even (persona + recent) alone plus a cushion — no amount of
    # turn-shedding (short of the forbidden newest) can ever satisfy this
    # while the stand-in survives.
    limit2 = persona2_exact + recent2_exact + 500
    report2: dict = {}
    out2 = main._enforce_hard_budget(list(payload2), limit2, 1, report2)
    out2_text = " ".join(main._message_text(m) for m in out2)
    standin2_survived = any(main._is_compaction_standin(m) for m in out2)
    newest2_survived = "newest-u" in out2_text
    check(report2.get("fits") is True, f"the guard fits the payload ({report2})")
    check(
        not standin2_survived,
        f"*** CONTROL: with nothing else left, the guard still spends the "
        f"stand-in — this fix does not turn 'last resort' into 'never' "
        f"(report={report2})",
    )
    check(newest2_survived, "CONTROL: the newest turn always survives, even here")

finally:
    main.count_tokens = _saved_count_tokens
    main.count_tokens_exact = _saved_count_tokens_exact


# ---------------------------------------------------------------------------
# From here on, tests call `main._shed_last_resort` DIRECTLY — the state
# the six-round loop above it leaves behind, isolated from that loop's own
# separate correctness (already covered by [G3a]/[G3a-CONTROL] above,
# which prove the wiring). Direct calls give exact, deterministic control
# over `msgs`/`per`/`running`/`limit` and the `measure` callable, which is
# what makes the ordering, linear-time and measurement-cap checks below
# precise rather than incidental.
# ---------------------------------------------------------------------------

def _msg(role, tag, cost, cost_map):
    """A message tagged with a unique content string, recorded in
    cost_map so a `measure` stub can look up its true cost by content."""
    content = f"{tag}:{len(cost_map)}"
    cost_map[content] = cost
    return {"role": role, "content": content}


def _real_measure(cost_map):
    """A WORKING (non-adversarial) measure(msgs) -> (total, 'stub'),
    summing the true cost of whatever content is still present — used
    where the test wants normal, single-round convergence."""
    def _measure(msgs):
        return sum(cost_map[m["content"]] for m in msgs), "stub"
    return _measure


# ---------------------------------------------------------------------------
# [G3a-mem] THE REPRODUCTION AND THE FIX FOR THE COORDINATOR'S OWN FINDING:
# a leftover old turn ABOVE the floor, plus spendable facts and retrieval,
# beside her previous exchange. Cutting the old turn alone is not enough;
# memory (facts + retrieval) is. The previous exchange must survive and
# memory must actually be spent — the first draft dropped the exchange
# here instead, because it shed every non-newest turn before ever
# reaching for memory.
# ---------------------------------------------------------------------------
print("\n[G3a-mem] memory pays before her previous exchange, when facts "
      "and retrieval are present and could cover the gap")
_mem_costs: dict = {}
_persona = _msg("system", "PERSONA", 50, _mem_costs)
_old_u = _msg("user", "OLDU", 300, _mem_costs)
_old_a = _msg("assistant", "OLDA", 300, _mem_costs)
_facts = _msg("system", "FACTS", 400, _mem_costs)
_retrieval = _msg("system", "RETRIEVAL", 400, _mem_costs)
_prev_u = _msg("user", "PREVU", 50, _mem_costs)
_prev_a = _msg("assistant", "PREVA", 5000, _mem_costs)
_newest_u = _msg("user", "NEWESTU", 50, _mem_costs)

_mem_msgs = [_persona, _old_u, _old_a, _facts, _retrieval, _prev_u, _prev_a, _newest_u]
_mem_per = [_mem_costs[m["content"]] for m in _mem_msgs]
_mem_total = sum(_mem_per)
# v3.1.9.4 (v3194-r3, R6): step 2 is now TARGET-LIMITED, so a gap that
# freeing the old pair PLUS FACTS ALONE already closes (600 + 400 = 1000)
# must leave RETRIEVAL untouched — this fixture's job is to need BOTH
# memory blocks, not just their sum, so the gap has to sit strictly ABOVE
# what facts alone (combined with the old pair) provides. Freeing the old
# pair alone (600) is NOT enough; the old pair plus FACTS alone (1000) is
# STILL not enough; the old pair plus facts+retrieval (600 + 800 = 1400)
# IS -- so step 3 (the recent window, which holds her previous exchange)
# must never even be reached, and BOTH memory blocks are required, not
# just offered.
_mem_limit = _mem_total - 1200
check(
    1000 < _mem_total - _mem_limit <= 1400,
    f"fixture: the gap ({_mem_total - _mem_limit}) needs the old pair AND "
    f"BOTH memory blocks (facts alone is not enough), but not the recent window",
)
_mem_out_msgs, _mem_out_per, _mem_out_running, _mem_counter, _mem_dropped, _mem_sys_dropped = (
    main._shed_last_resort(
        list(_mem_msgs), list(_mem_per), _mem_total, _mem_limit,
        protect_system=1, standin_protected=True, counter="stub",
        dropped=0, sys_dropped=0, measure=_real_measure(_mem_costs),
    )
)
_mem_out_texts = _texts(_mem_out_msgs)
check(_mem_out_running <= _mem_limit, f"fits (running={_mem_out_running}, limit={_mem_limit})")
check(
    "FACTS:3" not in _mem_out_texts and "RETRIEVAL:4" not in _mem_out_texts,
    f"*** [G3a-mem] THE FIX: memory (facts and retrieval) was spent "
    f"(out={_mem_out_texts})",
)
check(
    any(t.startswith("PREVU") for t in _mem_out_texts)
    and any(t.startswith("PREVA") for t in _mem_out_texts),
    f"*** [G3a-mem] THE FIX: her previous exchange survives — before this "
    f"fix, the first draft shed it here AHEAD of memory (out={_mem_out_texts})",
)
check(
    any(t.startswith("NEWESTU") for t in _mem_out_texts),
    "[G3a-mem]: the newest turn always survives",
)
check(
    not any(t.startswith("OLDU") or t.startswith("OLDA") for t in _mem_out_texts),
    f"fixture: the old pair (step 1) was shed too (out={_mem_out_texts})",
)
check(_mem_sys_dropped == 2, f"[G3a-mem]: exactly the 2 memory blocks were dropped (sys_dropped={_mem_sys_dropped})")


# ---------------------------------------------------------------------------
# [G3a-order] the full four-step priority, precisely, at three thresholds
# of ONE fixture: (a) old turn + memory is enough -> recent window and
# stand-in both survive; (b) also needs the recent window (except the
# newest) -> memory AND those turns go, stand-in still survives; (c) also
# needs the stand-in -> everything but the persona and the newest turn is
# gone.
# ---------------------------------------------------------------------------
print("\n[G3a-order] the full four-step priority: turns above the floor, "
      "then memory, then the recent window except the newest, then the "
      "protected stand-in — never the newest, never the persona")
_ord_costs: dict = {}
_o_persona = _msg("system", "PERSONA", 100, _ord_costs)
_o_standin = {
    "role": "system",
    "content": main.COMPACTION_SUMMARY_HEADER + "\nSTANDIN",
}
_ord_costs[_o_standin["content"]] = 9000
_o_old_u = _msg("user", "OLDU", 300, _ord_costs)
_o_old_a = _msg("assistant", "OLDA", 300, _ord_costs)
_o_facts = _msg("system", "FACTS", 400, _ord_costs)
_o_retrieval = _msg("system", "RETRIEVAL", 400, _ord_costs)
_o_prev_u = _msg("user", "PREVU", 50, _ord_costs)
_o_prev_a = _msg("assistant", "PREVA", 700, _ord_costs)
_o_newest_u = _msg("user", "NEWESTU", 50, _ord_costs)

_ord_msgs = [
    _o_persona, _o_standin, _o_old_u, _o_old_a, _o_facts, _o_retrieval,
    _o_prev_u, _o_prev_a, _o_newest_u,
]
_ord_per = [_ord_costs[m["content"]] for m in _ord_msgs]
_ord_total = sum(_ord_per)
_P, _SC, _OU, _OA, _F, _R, _PU, _PA, _NU = _ord_per


def _run_order(limit):
    out_msgs, *_ = main._shed_last_resort(
        list(_ord_msgs), list(_ord_per), _ord_total, limit,
        protect_system=1, standin_protected=True, counter="stub",
        dropped=0, sys_dropped=0, measure=_real_measure(_ord_costs),
    )
    tags = {t.split(":")[0] for t in _texts(out_msgs) if isinstance(t, str) and ":" in t}
    standin_survived = any(main._is_compaction_standin(m) for m in out_msgs)
    return tags, standin_survived


# (a) step 1 + step 2 (memory) is enough; step 3/4 never reached.
_lim_a = _P + _SC + _PU + _PA + _NU
_tags_a, _standin_a = _run_order(_lim_a)
check(
    _tags_a == {"PERSONA", "PREVU", "PREVA", "NEWESTU"} and _standin_a,
    f"*** [G3a-order] (a) old turns + memory freed the gap: the recent "
    f"window AND the stand-in both survive (tags={sorted(_tags_a)}, "
    f"standin_survived={_standin_a})",
)

# (b) step 1 + step 2 not enough; step 3 (recent window except newest) is
#     also needed, but the stand-in (step 4) must still survive.
_lim_b = _P + _SC + _NU
_tags_b, _standin_b = _run_order(_lim_b)
check(
    _tags_b == {"PERSONA", "NEWESTU"} and _standin_b,
    f"*** [G3a-order] (b) needed the recent window too: PREVU/PREVA are "
    f"gone but the stand-in STILL survives (tags={sorted(_tags_b)}, "
    f"standin_survived={_standin_b})",
)

# (c) nothing but the persona and the newest turn is enough; the
#     protected stand-in is spent too, truly last.
_lim_c = _P + _NU
_tags_c, _standin_c = _run_order(_lim_c)
check(
    _tags_c == {"PERSONA", "NEWESTU"} and not _standin_c,
    f"*** [G3a-order] (c) only the persona and the newest turn survive — "
    f"the stand-in went too, as the true last resort (tags={sorted(_tags_c)}, "
    f"standin_survived={_standin_c})",
)


# ---------------------------------------------------------------------------
# [G3a-linear] the pass is LINEAR in the turn count, not quadratic —
# test_p5_guard.py's own `[4]` technique: a message subclass counts calls
# to `.get("role")`, a property of the CODE PATH (index lists built once
# per round, cut by pointer) rather than wall-clock time, which is noisy
# under this shared VM's load. A per-drop rebuild (the P12-5 branch's own
# pre-fix regression, hostile pass #5) would make this count scale WITH n
# (c = count/n growing); the fixed shape keeps c flat.
# ---------------------------------------------------------------------------
print("\n[G3a-linear] the last-resort pass is linear in the turn count, "
      "not quadratic (deterministic op count, test_p5_guard.py [4]'s own "
      "technique)")

_LIN_C_BOUND = 10
_LIN_COUNT = [0]


class _CountingDict(dict):
    def get(self, key, default=None):
        if key == "role":
            _LIN_COUNT[0] += 1
        return dict.get(self, key, default)


def _linear_fixture(n_old):
    costs: dict = {}
    persona = _CountingDict(_msg("system", "PERSONA", 10, costs))
    old_turns = []
    for i in range(n_old):
        old_turns.append(_CountingDict(_msg("user", f"OLDU{i}", 10, costs)))
        old_turns.append(_CountingDict(_msg("assistant", f"OLDA{i}", 10, costs)))
    recent = [
        _CountingDict(_msg("user", "PREVU", 10, costs)),
        _CountingDict(_msg("assistant", "PREVA", 10, costs)),
        _CountingDict(_msg("user", "NEWESTU", 10, costs)),
    ]
    msgs = [persona] + old_turns + recent
    per = [costs[m["content"]] for m in msgs]
    return msgs, per, sum(per), costs


for _n_old, _label in ((250, "n~500"), (1000, "n~2000")):
    _lin_msgs, _lin_per, _lin_total, _lin_costs = _linear_fixture(_n_old)
    _n = len(_lin_msgs)
    # Shed ALL old turns (the whole gap between total and persona+recent).
    _lin_limit = _lin_costs["PERSONA:0"] + 30  # persona + the 3 recent turns
    _LIN_COUNT[0] = 0
    _lo, _lp, _lr, _lc, _ld, _lsd = main._shed_last_resort(
        _lin_msgs, _lin_per, _lin_total, _lin_limit,
        protect_system=1, standin_protected=True, counter="stub",
        dropped=0, sys_dropped=0, measure=_real_measure(_lin_costs),
    )
    check(_lr <= _lin_limit, f"fixture ({_label}): converged (running={_lr}, limit={_lin_limit})")
    check(_lsd == 0, f"fixture ({_label}): no memory to spend, so the count below is step 1 alone ({_lsd})")
    _c = _LIN_COUNT[0] / _n
    check(
        _LIN_COUNT[0] <= _LIN_C_BOUND * _n,
        f"*** [G3a-linear] {_label} (n={_n}): {_LIN_COUNT[0]} .get('role') "
        f"calls, c=count/n={_c:.2f}, within the C_BOUND={_LIN_C_BOUND} "
        f"linear envelope (a per-drop-rebuild regression would grow c WITH n)",
    )


# ---------------------------------------------------------------------------
# [G3a-cap] the number of exact measurements is bounded even when the
# measurement itself is adversarially useless — a stub that returns the
# SAME fixed value regardless of content (test_budget_guard.py's own
# technique). This is what caught the second defect: a target-limited
# memory-spend step converges only as fast as `running` shrinks, and a
# measurement that never shrinks made it spend one block per round,
# exhausting the cap with memory still held.
# ---------------------------------------------------------------------------
print("\n[G3a-cap] the number of exact measurements never exceeds "
      f"_G3A_MEASURE_CAP ({main._G3A_MEASURE_CAP}), even against a "
      "measurement that never reflects content")

_cap_costs: dict = {}
_cap_persona = _msg("system", "PERSONA", 10, _cap_costs)
_cap_blocks = [_msg("system", f"BLOCK{i}", 10, _cap_costs) for i in range(20)]
_cap_newest = _msg("user", "NEWESTU", 10, _cap_costs)
_cap_msgs = [_cap_persona] + _cap_blocks + [_cap_newest]
_cap_per = [_cap_costs[m["content"]] for m in _cap_msgs]
_cap_total = sum(_cap_per)

_cap_calls = [0]


def _broken_measure(msgs):
    """Adversarial: ALWAYS reports the same huge fixed value, regardless
    of what got cut — exactly the shape test_budget_guard.py's own
    'the last resort spends every injected block' fixture uses."""
    _cap_calls[0] += 1
    return _cap_total, "stub"


_cap_out_msgs, _cap_out_per, _cap_out_running, _cap_counter, _cap_dropped, _cap_sys_dropped = (
    main._shed_last_resort(
        list(_cap_msgs), list(_cap_per), _cap_total, _cap_total - 5000,
        protect_system=1, standin_protected=True, counter="stub",
        dropped=0, sys_dropped=0, measure=_broken_measure,
    )
)
check(
    _cap_calls[0] <= main._G3A_MEASURE_CAP,
    f"*** [G3a-cap] only {_cap_calls[0]} measurement(s) were made against "
    f"the {main._G3A_MEASURE_CAP}-round cap, even though the measurement "
    f"never once confirmed progress",
)
check(
    not any(t.startswith("BLOCK") for t in _texts(_cap_out_msgs)),
    f"*** [G3a-cap] every spendable block was still dropped despite the "
    f"broken measurement — the 5000-token target here is far beyond the "
    f"200 tokens all 20 blocks together are worth, so target-limited step "
    f"2 (v3.1.9.4, R6) still exhausts them all; it stops early only when "
    f"the target really is smaller than what remains, which [G3a-"
    f"unconditional] below is what actually pins "
    f"(out={sorted(_texts(_cap_out_msgs))})",
)
check(
    any(t.startswith("NEWESTU") for t in _texts(_cap_out_msgs))
    and any(t.startswith("PERSONA") for t in _texts(_cap_out_msgs)),
    "[G3a-cap]: the persona and the newest turn always survive",
)


# ---------------------------------------------------------------------------
# [G3a-unconditional] v3.1.9.4 (v3194-r3, R6) REWORKED: this used to pin
# step 2 dropping EVERY eligible block once reached, even when one block
# already closed the gap -- the exact waste the forced drop this pass
# replaced was written to avoid, in that forced drop's own comment:
# "Dropping EVERYTHING is wasteful: measured by review, a payload over by
# 550 tokens lost persona, facts and summary when one 1500-token block
# covered it." Steps 2 and 4 are now TARGET-LIMITED, like steps 1 and 3
# always were (main.py, _shed_last_resort, R6's own comment there has the
# full reasoning, including why the round-cap risk the ORIGINAL
# unconditional shape was defending against is bounded, not eliminated, by
# this pass's own cross-round rescale). This test now pins THE MINIMUM:
# with a working (non-adversarial) measurement and TWO spendable blocks
# where only the larger one is needed to close the gap, only that one is
# dropped -- the smaller one survives.
# ---------------------------------------------------------------------------
print("\n[G3a-unconditional] step 2 drops only as much spendable memory as "
      "the gap needs -- one block closes it, the other survives")

_unc_costs: dict = {}
_unc_persona = _msg("system", "PERSONA", 50, _unc_costs)
_unc_big = _msg("system", "BIGBLOCK", 1000, _unc_costs)
_unc_small = _msg("system", "SMALLBLOCK", 1000, _unc_costs)
_unc_newest = _msg("user", "NEWESTU", 50, _unc_costs)
_unc_msgs = [_unc_persona, _unc_big, _unc_small, _unc_newest]
_unc_per = [_unc_costs[m["content"]] for m in _unc_msgs]
_unc_total = sum(_unc_per)  # 50 + 1000 + 1000 + 50 = 2100

# The gap (600) sits strictly under BIGBLOCK's own cost (1000) -- BIGBLOCK
# ALONE clears it (there is no step 1 content here to help: both messages
# above the recent window are protected persona/newest, so step 2 carries
# the whole gap). SMALLBLOCK must never be touched.
_unc_limit = _unc_total - 600
check(
    0 < _unc_total - _unc_limit < 1000,
    f"fixture: the gap ({_unc_total - _unc_limit}) is smaller than either "
    f"single spendable block (1000), so ONE closes it",
)

_unc_calls = [0]
_unc_real_measure = _real_measure(_unc_costs)


def _unc_counting_measure(msgs):
    _unc_calls[0] += 1
    return _unc_real_measure(msgs)


_unc_out_msgs, _unc_out_per, _unc_out_running, _unc_counter, _unc_dropped, _unc_sys_dropped = (
    main._shed_last_resort(
        list(_unc_msgs), list(_unc_per), _unc_total, _unc_limit,
        protect_system=1, standin_protected=True, counter="stub",
        dropped=0, sys_dropped=0, measure=_unc_counting_measure,
    )
)
_unc_out_texts = _texts(_unc_out_msgs)
check(_unc_out_running <= _unc_limit, f"fits (running={_unc_out_running}, limit={_unc_limit})")
check(
    "BIGBLOCK:1" not in _unc_out_texts,
    f"*** [G3a-unconditional] THE FIX: BIGBLOCK alone closed the gap and "
    f"was dropped (out={_unc_out_texts})",
)
check(
    "SMALLBLOCK:2" in _unc_out_texts,
    f"*** [G3a-unconditional] THE FIX: SMALLBLOCK was NEVER TOUCHED -- the "
    f"pre-fix shape would have dropped it too, the instant step 2 was "
    f"reached at all, even though BIGBLOCK alone already closed the gap "
    f"(out={_unc_out_texts})",
)
check(_unc_sys_dropped == 1, f"[G3a-unconditional]: exactly ONE memory block was dropped (sys_dropped={_unc_sys_dropped})")
check(
    _unc_calls[0] <= main._G3A_MEASURE_CAP,
    f"[G3a-unconditional]: still within the measurement cap ({_unc_calls[0]} calls)",
)
check(
    any(t.startswith("NEWESTU") for t in _unc_out_texts)
    and any(t.startswith("PERSONA") for t in _unc_out_texts),
    "[G3a-unconditional]: the persona and the newest turn always survive",
)


# ---------------------------------------------------------------------------
# [G3a-step4-target] v3.1.9.4 (v3194-r3, R6): step 4's OWN target-limiting,
# isolated from step 2. Production only ever offers step 4 one candidate
# (the single real compaction stand-in), so [G3a-order] above cannot
# distinguish target-limited from unconditional there -- with one item,
# both shapes drop it or keep it identically. Two synthetic
# stand-in-shaped blocks (recognised by _is_compaction_standin on CONTENT
# alone, per that function's own docstring) isolate step 4's loop the same
# way [G3a-unconditional] isolates step 2's.
# ---------------------------------------------------------------------------
print("\n[G3a-step4-target] step 4 drops only as many protected-stand-in-"
      "shaped blocks as the gap needs")

_s4_costs: dict = {}
_s4_persona = _msg("system", "PERSONA", 50, _s4_costs)
_s4_standin_big = {"role": "system", "content": main.COMPACTION_SUMMARY_HEADER + "\nSTANDINBIG"}
_s4_costs[_s4_standin_big["content"]] = 1000
_s4_standin_small = {"role": "system", "content": main.COMPACTION_SUMMARY_HEADER + "\nSTANDINSMALL"}
_s4_costs[_s4_standin_small["content"]] = 1000
_s4_newest = _msg("user", "NEWESTU", 50, _s4_costs)
_s4_msgs = [_s4_persona, _s4_standin_big, _s4_standin_small, _s4_newest]
_s4_per = [_s4_costs[m["content"]] for m in _s4_msgs]
_s4_total = sum(_s4_per)  # 50 + 1000 + 1000 + 50 = 2100
_s4_limit = _s4_total - 600  # gap 600 < either single stand-in's 1000

_s4_out_msgs, *_s4_rest, _s4_sys_dropped = main._shed_last_resort(
    list(_s4_msgs), list(_s4_per), _s4_total, _s4_limit,
    protect_system=1, standin_protected=True, counter="stub",
    dropped=0, sys_dropped=0, measure=_real_measure(_s4_costs),
)
_s4_out_texts = _texts(_s4_out_msgs)
_s4_standins_left = sum(1 for m in _s4_out_msgs if main._is_compaction_standin(m))
check(
    _s4_standins_left == 1,
    f"*** [G3a-step4-target] THE FIX: exactly ONE of the two stand-in-shaped "
    f"blocks survives — the other alone closed the gap "
    f"(standins_left={_s4_standins_left}, out={_s4_out_texts})",
)
check(_s4_sys_dropped == 1, f"[G3a-step4-target]: exactly one block dropped (sys_dropped={_s4_sys_dropped})")
check(
    any(t.startswith("NEWESTU") for t in _s4_out_texts)
    and any(t.startswith("PERSONA") for t in _s4_out_texts),
    "[G3a-step4-target]: the persona and the newest turn always survive",
)


# ---------------------------------------------------------------------------
# [G3a-rescale] after each real measurement, the remaining per-item
# ESTIMATES are rescaled by (what this round actually freed) / (what its
# estimate said it would free) -- without this, a systematic
# miscalibration (this fixture: old turns whose ESTIMATE reads 100x their
# real cost -- the same SHAPE as the stand-in's own 4x-UNDER reproduction
# in [G3a], just the opposite direction and exaggerated to force it past
# six rounds instead of converging within them) makes every round cut only
# ONE pair -- the estimate always looks "big enough" against a shrinking
# real target that never actually needed that much -- and the pass hits
# the measurement cap without converging. WITH rescale, round 1's
# real-vs-estimate ratio corrects the remaining estimates to reality and
# the SAME fixture converges in two rounds.
# ---------------------------------------------------------------------------
print("\n[G3a-rescale] per-item estimates are corrected against reality "
      "after each round, so a systematic miscalibration still converges "
      "well inside the measurement cap")


def _rescale_fixture():
    costs: dict = {}
    persona = _msg("system", "PERSONA", 10, costs)
    old_turns = []
    for i in range(20):
        old_turns.append(_msg("user", f"OLDU{i}", 10, costs))
        old_turns.append(_msg("assistant", f"OLDA{i}", 10, costs))
    recent = [
        _msg("user", "PREVU", 10, costs),
        _msg("assistant", "PREVA", 10, costs),
        _msg("user", "NEWESTU", 10, costs),
    ]
    msgs = [persona] + old_turns + recent
    real_per = [costs[m["content"]] for m in msgs]
    # Estimates: old turns read 100x their real cost (1000 vs. 10); every
    # other message's estimate is accurate -- isolates the miscalibration
    # to exactly the content step 1 decides about.
    est_per = [
        1000 if (m["content"].startswith("OLDU") or m["content"].startswith("OLDA"))
        else rc
        for m, rc in zip(msgs, real_per)
    ]
    return msgs, est_per, sum(real_per), costs


_re_msgs, _re_per, _re_total, _re_costs = _rescale_fixture()
# persona(10) + 20 pairs*20(400) + recent(30) = 440; need 15 of the 20
# pairs' real cost (300) shed -- more than one round's worth of progress
# if step 1 keeps stopping after a single (over-estimated) pair.
_re_limit = _re_total - 300
_re_calls = [0]


def _re_measure(msgs):
    _re_calls[0] += 1
    return sum(_re_costs[m["content"]] for m in msgs), "stub"


_re_out_msgs, _re_out_per, _re_out_running, _re_counter, _re_dropped, _re_sys_dropped = (
    main._shed_last_resort(
        list(_re_msgs), list(_re_per), _re_total, _re_limit,
        protect_system=1, standin_protected=True, counter="stub",
        dropped=0, sys_dropped=0, measure=_re_measure,
    )
)
check(
    _re_out_running <= _re_limit,
    f"*** [G3a-rescale] THE FIX: converged (running={_re_out_running}, "
    f"limit={_re_limit}) -- without rescaling, this 300-token real gap "
    f"needs about fifteen rounds (one 20-token pair per round against a "
    f"stale 1000-token estimate) and hits the "
    f"{main._G3A_MEASURE_CAP}-round cap still over limit",
)
check(
    _re_calls[0] <= main._G3A_MEASURE_CAP,
    f"[G3a-rescale]: within the measurement cap ({_re_calls[0]} calls)",
)
_re_old_remaining = sum(
    1 for t in _texts(_re_out_msgs)
    if t.startswith("OLDU") or t.startswith("OLDA")
)
check(
    0 < _re_old_remaining < 40,
    f"fixture: only PART of the old turns were needed, not all of them "
    f"({_re_old_remaining} of 40 remain)",
)
check(
    any(t.startswith("PREVU") for t in _texts(_re_out_msgs))
    and any(t.startswith("PREVA") for t in _texts(_re_out_msgs))
    and any(t.startswith("NEWESTU") for t in _texts(_re_out_msgs)),
    "[G3a-rescale]: her previous exchange and the newest turn survive",
)


# ---------------------------------------------------------------------------
# [G3a-cap-forced] the measurement count stays within the cap even against
# a measurement that never reflects any cut (so `running > limit` never
# resolves on its own) -- at this size (100 pairs) rescale's own reaction
# to a perfectly flat measurement (a zero real-vs-estimate ratio, which
# floors every remaining per-item estimate to 1) converges this particular
# fixture in two rounds, well under the cap; [G3a-cap-binding] right below
# uses a larger fixture where the cap itself -- not convergence -- is what
# stops the pass.
# ---------------------------------------------------------------------------
print("\n[G3a-cap-forced] the measurement cap is the thing that stops the "
      "loop when there is always more to cut and the measurement never "
      "shows progress")

_cf_costs: dict = {}
_cf_persona = _msg("system", "PERSONA", 10, _cf_costs)
_cf_pairs = []
for i in range(100):
    _cf_pairs.append(_msg("user", f"OLDU{i}", 10, _cf_costs))
    _cf_pairs.append(_msg("assistant", f"OLDA{i}", 10, _cf_costs))
_cf_newest = _msg("user", "NEWESTU", 10, _cf_costs)
_cf_msgs = [_cf_persona] + _cf_pairs + [_cf_newest]
_cf_per = [_cf_costs[m["content"]] for m in _cf_msgs]
_cf_total = sum(_cf_per)  # 10 + 2000 + 10 = 2020

_cf_calls = [0]


def _cf_broken_measure(msgs):
    _cf_calls[0] += 1
    return _cf_total, "stub"  # never reflects what was actually cut


# Target 300 real tokens/round -> 15 pairs/round (target-limited step 1,
# unchanged and correct for TURNS) -- the ~99 available pairs need 7
# rounds to exhaust, one more than the cap.
_cf_limit = _cf_total - 300
_cf_out_msgs, _cf_out_per, _cf_out_running, _cf_counter, _cf_dropped, _cf_sys_dropped = (
    main._shed_last_resort(
        list(_cf_msgs), list(_cf_per), _cf_total, _cf_limit,
        protect_system=1, standin_protected=True, counter="stub",
        dropped=0, sys_dropped=0, measure=_cf_broken_measure,
    )
)
_cf_old_remaining = sum(
    1 for t in _texts(_cf_out_msgs)
    if t.startswith("OLDU") or t.startswith("OLDA")
)
check(
    _cf_calls[0] <= main._G3A_MEASURE_CAP,
    f"*** [G3a-cap-forced] THE CAP: exactly the {main._G3A_MEASURE_CAP}-"
    f"round budget was spent, not one round more, even though 100 old "
    f"pairs and an always-over-limit measurement would otherwise keep "
    f"this pass cutting indefinitely ({_cf_calls[0]} calls; "
    f"{_cf_old_remaining} old turn(s) of 200 remain, forwarded as the "
    f"best effort)",
)
check(
    any(t.startswith("NEWESTU") for t in _texts(_cf_out_msgs))
    and any(t.startswith("PERSONA") for t in _texts(_cf_out_msgs)),
    "[G3a-cap-forced]: the persona and the newest turn always survive",
)


# ---------------------------------------------------------------------------
# [G3a-cap-binding] the cap is the thing that stops this pass, not
# incidental convergence -- 2,000 old pairs (far more than rescale's own
# post-round floor of 1-per-item can clear in the remaining five rounds
# once round 1's real-vs-estimate ratio corrects them) against a
# measurement that never reflects any cut. Removing (or loosening) the
# cap's own loop condition lets this run for many more rounds instead of
# stopping at exactly `_G3A_MEASURE_CAP` -- proven empirically (probe
# script, kept in scratch): the unmutated pass here takes exactly 6 calls
# and leaves 2,470 of 4,000 old turns; with the cap's loop condition
# loosened it takes 15 calls and clears every one of them. [G3a-cap-forced]
# above cannot show this -- at 100 pairs it converges in two rounds
# regardless of the cap's value.
# ---------------------------------------------------------------------------
print("\n[G3a-cap-binding] the measurement cap -- not convergence -- is "
      "what stops this pass, on a fixture too large for rescale's own "
      "post-round correction to clear within the remaining rounds")

_cb_costs: dict = {}
_cb_persona = _msg("system", "PERSONA", 10, _cb_costs)
_cb_pairs = []
for i in range(2000):
    _cb_pairs.append(_msg("user", f"OLDU{i}", 10, _cb_costs))
    _cb_pairs.append(_msg("assistant", f"OLDA{i}", 10, _cb_costs))
_cb_newest = _msg("user", "NEWESTU", 10, _cb_costs)
_cb_msgs = [_cb_persona] + _cb_pairs + [_cb_newest]
_cb_per = [_cb_costs[m["content"]] for m in _cb_msgs]
_cb_total = sum(_cb_per)

_cb_calls = [0]


def _cb_broken_measure(msgs):
    _cb_calls[0] += 1
    return _cb_total, "stub"  # never reflects what was actually cut


_cb_limit = _cb_total - 300
_cb_out_msgs, _cb_out_per, _cb_out_running, _cb_counter, _cb_dropped, _cb_sys_dropped = (
    main._shed_last_resort(
        list(_cb_msgs), list(_cb_per), _cb_total, _cb_limit,
        protect_system=1, standin_protected=True, counter="stub",
        dropped=0, sys_dropped=0, measure=_cb_broken_measure,
    )
)
check(
    _cb_calls[0] == main._G3A_MEASURE_CAP,
    f"*** [G3a-cap-binding] THE CAP: exactly {main._G3A_MEASURE_CAP} "
    f"measurements were made, neither more (an unbounded loop condition) "
    f"nor fewer (an early, unrelated exit) -- {_cb_calls[0]} calls",
)
check(
    any(t.startswith("NEWESTU") for t in _texts(_cb_out_msgs))
    and any(t.startswith("PERSONA") for t in _texts(_cb_out_msgs)),
    "[G3a-cap-binding]: the persona and the newest turn always survive",
)


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nall checks passed")
