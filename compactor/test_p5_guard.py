"""The hard-budget guard on a compacted array: shed whole old exchanges
before injected memory, never a lone turn — hostile pass #5, reviewer A F4.

pass-4 F5 (test_p4a_guard_order.py) made the guard, when a compaction
stand-in sits in the array, shed the old turns that would have to go EVEN
WITH EVERY SPENDABLE BLOCK DROPPED before spending injected memory (facts,
retrieval) — because in the cap-refusal regime (a stand-in AND summarize()
refusing the fresh span over its per-request call cap) the turns left are
tens to thousands of old, un-summarized exchanges, far more than all
injected memory, and spending memory first bought nothing.

That pre-shed loop had two bugs, both in `_enforce_hard_budget`'s
compacted-standin branch (main.py):

  (a) It stopped the moment cutting ALL injected memory could cover the
      rest ("running - _memory <= limit"). "Memory COULD pay it" is not
      "nothing more is owed" — it left LESS THAN ONE old exchange
      undropped, every time, and let memory absorb that remainder. At her
      numbers (facts 400 + retrieval 1,500 tokens, old exchanges 1,800
      tokens each): memory was cut or gone on 16 of 40 payload sizes and
      whole on 0, while shedding one more old exchange instead of touching
      memory would have fit the same limit on 38 of those 16
      (SP\\p5-a\\p5a_guard.py, SP\\p5-a-findings.md F4).
  (b) It dropped ONE message at a time, so it could stop having removed an
      old USER turn but not yet its reply — the pair the role-alternation
      repair (further down the guard) would go on to delete anyway, AFTER
      memory had already been halved and dropped to cover the tokens that
      orphaned reply cost. Traced at the shipped limit (p5a_guard_trace.py):
      1,798 tokens — 9% of the 20,768-token window — sat unused because the
      space the repair freed arrived after the memory spend it would have
      made unnecessary.

The fix: shed every FULL old exchange (a user turn together with its reply,
if the reply immediately follows — never split) down to KEEP_RECENT_TURNS,
before memory is touched at all; memory is spent only once that floor is
reached and the array still does not fit.

This file is the reviewer's own 40-size sweep, turned into a test: the
memory-whole count is the assertion (0 of 40 before this fix, per
SP\\p5-a-findings.md; 40 of 40 after — SP\\fix-p5-guard\\p5a_guard_postfix.log),
plus two controls the sweep alone does not cover:

  [1] the sweep: memory is kept whole on every payload size, old exchanges
      are always shed in pairs, and the array is never left assistant-first
  [2] CONTROL: a payload with nothing to shed at all (no deferred history)
      passes through completely untouched
  [3] CONTROL: a payload with nothing SHEDDABLE (no deferred history, and
      the recent turns are protected) still gets memory cut when that is
      the only way left to fit — this fix narrows when memory is spent, it
      does not make the guard refuse to spend it
  [4] the pre-shed loop is linear in the turn count, not quadratic — proved
      by a DETERMINISTIC operation count (how many times a message's role
      is inspected), not a wall-clock timing, so this holds regardless of
      machine speed or load. A first draft of this fix rebuilt the
      non-system index list AND called `del msgs[i]` / `del per[i]` on
      every iteration; caught before it shipped and replaced with a single
      scan, a pointer walk, and one rebuild. This section is that
      regression's regression test.

    python test_p5_guard.py
"""

import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://127.0.0.1:9")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="p5guard-")

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


def tokens(msgs) -> int:
    """One token per UTF-8 byte of each message's text, plus 4 per message —
    deterministic, so the guard's own arithmetic and this file's
    expectations agree by construction (same convention as
    test_time_budget.py and the reviewer's p5a_guard.py / p5a_guard_trace.py)."""
    return sum(len(main._message_text(m).encode("utf-8")) + 4 for m in msgs)


main.count_tokens_exact = lambda ms, *a, **k: tokens(ms)
main.count_tokens = lambda ms: tokens(ms)
main._BUDGET_MARGIN = 0

check(main.KEEP_RECENT_TURNS == 4, f"fixture: KEEP_RECENT_TURNS ({main.KEEP_RECENT_TURNS})")

LIMIT = 20768 - 90  # the shipped v3.1.9 effective limit, less a time-line reserve
FACTS = "[Facts]\n" + "".join(
    f"- FACT{i:02d} she likes item {i} very much indeed.\n" for i in range(9)
)
FACTS = FACTS + "f" * (400 - len(FACTS))
RETR = "[Retrieved]\n" + "RETRIEVAL " + "r" * (1500 - 22)
MEM = FACTS + "\n\n" + RETR


def build(newest_pad, deferred_exchanges=40):
    """A compaction stand-in beside `deferred_exchanges` old, un-summarized
    exchanges (the cap-refusal regime pass-4 F5 targeted): her caller system
    prompt, the stand-in, injected memory, the deferred history, the
    previous exchange, and a newest user turn padded to `newest_pad`."""
    msgs = [
        {"role": "system", "content": "P" * 1200},
        {"role": "system", "content": main.COMPACTION_SUMMARY_HEADER + "\n" + "s" * 6000},
        {"role": "system", "content": MEM},
    ]
    for i in range(deferred_exchanges):
        msgs.append({"role": "user", "content": f"old-u{i} " + "u" * 150})
        msgs.append({"role": "assistant", "content": f"old-a{i} " + "a" * 1650})
    msgs.append({"role": "user", "content": "prev-u " + "u" * 150})
    msgs.append({"role": "assistant", "content": "prev-a " + "a" * 1650})
    msgs.append({"role": "user", "content": "newest " + "n" * newest_pad})
    return msgs


def run(newest_pad, deferred_exchanges=40):
    m = build(newest_pad, deferred_exchanges)
    rep: dict = {}
    out = main._enforce_hard_budget(m, LIMIT, 1, rep)
    mem_out = [
        x for x in out if x.get("role") == "system" and "[Facts]" in (x.get("content") or "")
    ]
    mem_tokens = tokens(mem_out) - 4 * len(mem_out) if mem_out else 0
    facts_whole = bool(mem_out) and all(f"FACT{i:02d}" in mem_out[0]["content"] for i in range(9))
    return out, rep, mem_tokens, facts_whole


# ===========================================================================
print("[1] the reviewer's 40-size sweep — memory-whole count is the assertion")
rows = []
for pad in range(100, 1900, 45):
    out, rep, mem_tokens, facts_whole = run(pad)
    idxs = [i for i, m in enumerate(out) if m.get("role") != "system"]
    first_is_user = bool(idxs) and out[idxs[0]].get("role") == "user"
    check(
        first_is_user,
        f"pad={pad}: the array is never left assistant-first (role alternation "
        f"intact without needing the repair to fix an orphan THIS branch made)",
    )
    check(
        (rep["dropped_turns"] or 0) % 2 == 0,
        f"pad={pad}: old exchanges are shed in WHOLE pairs, never a lone turn "
        f"({rep['dropped_turns']} dropped)",
    )
    check(rep.get("fits") is True, f"pad={pad}: the guard fits the payload ({rep})")
    rows.append((pad, rep["dropped_turns"], mem_tokens, facts_whole))

n = len(rows)
memory_whole = sum(1 for r in rows if r[2] >= len(MEM))
facts_ok = sum(1 for r in rows if r[3])
check(n == 40, f"fixture: the reviewer's sweep is 40 payload sizes ({n})")
check(
    memory_whole == 40,
    f"*** F4: memory (facts + retrieval, {len(MEM)} tokens) is kept WHOLE on "
    f"every payload size in the cap-refusal regime ({memory_whole}/{n} — was "
    f"0/40 before this fix; SP\\p5-a-findings.md F4, SP\\p5-a\\p5a_guard.py)",
)
check(facts_ok == 40, f"facts are never cut or dropped on any sweep point ({facts_ok}/{n})")


# ===========================================================================
print("[2] CONTROL: nothing to shed at all — the payload passes through untouched")
# No deferred history (just the previous exchange and the newest turn, both
# inside KEEP_RECENT_TURNS), and small enough that the whole array — caller
# system, stand-in, memory, three recent turns — fits without spending
# anything. Proves the fix does not shed or trim when there is no need to.
out, rep, mem_tokens, facts_whole = run(newest_pad=100, deferred_exchanges=0)
check(
    rep.get("dropped_turns") == 0 and rep.get("trimmed_blocks") == 0
    and rep.get("dropped_blocks") == 0,
    f"nothing was shed, trimmed or dropped ({rep})",
)
check(mem_tokens >= len(MEM) and facts_whole, "memory is untouched")
check(rep.get("fits") is True, "and it fits")
_in = build(newest_pad=100, deferred_exchanges=0)
check(len(out) == len(_in), f"the array is the same length in and out ({len(_in)} -> {len(out)})")


# ===========================================================================
print("[3] CONTROL: nothing SHEDDABLE — memory legitimately must be cut")
# Still no deferred history, but the newest turn alone is now large enough
# that the array does not fit even with every recent turn kept (as it must
# be — KEEP_RECENT_TURNS protects them, and the newest turn is never
# dropped). There is no old exchange here to shed at all, so the fix's
# reordering changes nothing: memory is spent, exactly as it was before this
# fix and exactly as F4 says it still should be once shedding is exhausted.
out, rep, mem_tokens, facts_whole = run(newest_pad=10500, deferred_exchanges=0)
check(
    rep.get("dropped_turns") == 0,
    f"fixture: no old exchange exists to shed ({rep.get('dropped_turns')} dropped)",
)
check(
    mem_tokens < len(MEM),
    f"*** memory IS cut here — correctly: nothing else was left to spend "
    f"({mem_tokens}/{len(MEM)} kept)",
)
check(rep.get("fits") is True, f"and the guard still gets the payload to fit ({rep})")
idxs = [i for i, m in enumerate(out) if m.get("role") != "system"]
check(
    bool(idxs) and out[idxs[0]].get("role") == "user" and any("newest" in main._message_text(m) for m in out),
    "her previous exchange and newest turn both survive — only memory paid",
)


# ===========================================================================
print("[4] the pre-shed loop is LINEAR, not quadratic (deterministic op count)")
# Review follow-up on this fix (see SP\fix-p5-guard.md). The first draft's
# pre-shed loop rebuilt the non-system index list, AND called `del msgs[i]`
# / `del per[i]`, on every single iteration. In this regime the array can
# hold thousands of turns with hundreds needing to go (her live chat: ~480
# turns today, ~1,900 in the archive, ~470 shed per request in this
# regime), all on the request path, holding the GIL (reviewer pass-3 F6
# measured 0.46s of GIL-bound CPU in the reuse gate on far less data than
# this — SP\p3-a-findings.md F6). The fix scans once, walks a plain integer
# pointer to decide the cut, and rebuilds `msgs`/`per` exactly once.
#
# A WALL-CLOCK bound was tried first and rejected on review: measured
# fixed 6-38 ms and the quadratic mutant 70-229 ms in the same shared-VM
# container this suite runs in (four separate runs each) — close enough,
# and noisy enough under load from other lanes' Docker work, that a fixed
# mutant could pass on a quiet run and the real fix could fail on a loaded
# one. Wall time is not what changed; the SHAPE of the work did. So this
# counts it directly.
#
# `.get("role")` is exactly what both versions use to decide which message
# is next: the fixed loop calls it once per message to build `_turn_idxs`
# and twice per exchange evaluated thereafter (O(n) total); the version
# this replaces rebuilt `idxs = [... m.get("role") ...]` — a FULL scan of
# the (shrinking) array — on every one of up to ~n/2 iterations (O(n^2)
# total). Wrapping every message dict in a subclass that counts calls to
# `.get("role")`, with no other behavioural change (it IS a dict; `main.py`
# never knows the difference), makes that count a property of the CODE
# PATH, not the machine — identical on a quiet box, a loaded one, or a
# slower CPU architecture entirely.
#
# The bound (measured directly below, both sizes, this run): at n=504 the
# fixed loop makes exactly 1,514 calls (c = count/n = 3.00); at n=2,004 it
# makes exactly 6,014 (c = 3.00 again — the SAME c at 4x the size is the
# definition of linear). The quadratic mutant, same fixture
# (SP\p5-guard-count\count_probe.py / count_probe_mutant.py, run once to
# calibrate this bound, not part of the suite): c = 129.36 at n=504 and
# c = 504.47 at n=2,004 — c itself scales with n, the signature of O(n^2).
# C_BOUND=10 below has more than 3x headroom above the fixed loop's actual
# c (3.00) and is more than 12x under the mutant's SMALLER-n c (129.36, the
# harder of the two for this bound to catch) — the mutant fails this check
# even at n=504, well before n=2,000 is needed to make the point.
C_BOUND = 10
_COUNT = [0]


class _CountingDict(dict):
    """A message dict that is, in every other respect, a dict — `main.py`
    never branches on the type, only on `.get(...)`/`[...]`/`isinstance(x,
    dict)`, and a dict subclass satisfies all three. Counts only
    `.get("role")`, the specific operation whose call COUNT (not its
    wall-clock cost) distinguishes the fixed loop from the regression it
    replaces."""

    def get(self, key, default=None):
        if key == "role":
            _COUNT[0] += 1
        return dict.get(self, key, default)


def _counting_build(newest_pad, deferred_exchanges):
    return [_CountingDict(m) for m in build(newest_pad, deferred_exchanges)]


_OPCOUNT_LIMIT = 20768 - 90
for _deferred, _label in ((249, "n~500"), (999, "n~2000")):
    _m = _counting_build(newest_pad=300, deferred_exchanges=_deferred)
    _n = len(_m)
    _COUNT[0] = 0
    _rep: dict = {}
    main._enforce_hard_budget(_m, _OPCOUNT_LIMIT, 1, _rep)
    _c = _COUNT[0] / _n
    check(
        _rep.get("trimmed_blocks") == 0 and _rep.get("dropped_blocks") == 0,
        f"fixture ({_label}, n={_n}): memory untouched, so the count below is "
        f"the pre-shed loop's alone, not diluted by the trim/drop stages "
        f"({_rep})",
    )
    check(
        _COUNT[0] <= C_BOUND * _n,
        f"*** {_label} (n={_n}): {_COUNT[0]} .get('role') calls, c=count/n="
        f"{_c:.2f}, within the C_BOUND={C_BOUND} linear envelope (a "
        f"per-iteration-rebuild regression measures c=129-504 on this same "
        f"fixture, growing WITH n — see SP\\p5-guard-count\\count_probe.py)",
    )


print()
if FAILED:
    print(f"FAILED: {len(FAILED)} check(s)")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("All pass-5 guard-order checks passed.")
