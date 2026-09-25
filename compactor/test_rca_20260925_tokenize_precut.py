"""2026-09-25 RCA (her chat: 4,196 resent messages, ~7.8M tokens, every
request). Two pointless full-payload /tokenize calls per request, each a
plain synchronous POST with a 10s read timeout, on a payload that was never
going to be summarized or forwarded whole:

  1. `summarize()` (`_summarize_body`, main.py) measured the FULL backlog's
     exact token count before checking whether the backlog even fits under
     `MAX_SUMMARY_CALLS_PER_REQUEST` — a check that, on a backlog this size,
     was always going to refuse regardless of what /tokenize said.

  2. `_enforce_hard_budget`'s first ground-truth measurement
     (`count_tokens_exact(messages)`) priced the WHOLE conversation, when
     the per-message shedding loop right below it (and, on a payload this
     starved, `_shed_last_resort`) was always going to shed nearly all of
     it anyway, oldest turns first, down to a small fixed floor.

THE FIX:

  1. `_min_possible_batches` proves, from the local tokenizer alone, that
     the backlog needs more summarization calls than
     `MAX_SUMMARY_CALLS_PER_REQUEST` allows, and `summarize()` refuses on
     that floor before ever calling /tokenize.

  2. `_precut_to_local_ceiling` sheds old turns (never a system message,
     never into the aligned recent-turn floor, whole (user, assistant)
     exchanges only) down to roughly `_PESSIMISTIC_SUMMARY_SCALE` (2.0x)
     times `limit`, on local counts alone, before `_enforce_hard_budget`'s
     first real /tokenize call.

ROUND 2 (hostile review): the FIRST cut of both fixes carried the OLD
calibration (count_tokens_exact's 2026-08-28 comment: local reads ~50% LOW
on assistant content, ~10% high on user/system — a transformers-converted
tokenizer this deployment no longer runs). The SERVED tokenizer is now
mistral_common-backed: it never reads low, only high, by a roughly fixed
per-message framing cost that a per-message `count_tokens([m])` call pays
for itself every time instead of once for the whole conversation — a small
fraction of a long message, a large one of a short one (hostile review
measured a real/local ratio as low as 0.63 on 2-6-word messages). Findings
fixed here:

  - finding 1: `_min_possible_batches`'s certainty margin
    (`_LOCAL_OVERCOUNT_CEILING`) is now 2.0 — a provably-safe margin over
    the worst OVERcount ratio actually measured (1/0.63 ~= 1.59x), not the
    stale 1.10 that produced wrong refusals (a floor of >=5 batches where
    the real count needed only 3-4).
  - finding 2: `_precut_to_local_ceiling` cuts whole (user, assistant)
    exchanges only (never a lone turn, which could leave an assistant
    turn first); a payload the pre-cut alone already brings under `limit`
    now still runs fill-back and logs the normal "hard budget enforced"
    WARNING, instead of returning early behind an INFO line that promised
    a detail line which never came; fill-back is handed the ORIGINAL
    (pre-pre-cut) message list, so pre-cut turns are restorable too.
  - finding 3: the WARNING's `{total} ->` prefix now includes the
    ORIGINAL local count when pre-cut ran, so the line does not read as
    if the whole request were only as large as the pre-cut remainder.
  - finding 4: the pre-cut TRIGGER now measures with the real local
    tokenizer (`count_tokens`, the same whole-list call made moments
    later anyway), not the char/4 `_fast_token_estimate`, which
    undercounts scripts this model prices far above 4 chars/token
    (fully-pointed Hebrew, emoji, box-drawing) and could leave the
    trigger unreached between 2x and 8x the true limit.

ROUND 3 (hostile re-review of round 2's own fix): round 2's 2.0 ceiling was
ITSELF still wrong, on content shorter than anything round 2 measured.
Round 2 measured its worst case on 2-6-WORD messages through the
per-message SINGLETON call path; round 3 measured degenerate ONE-TOKEN and
EMPTY turns ("ok"/"sure", "k"/"k", ""/"") against vLLM's own tokenization
in the prod image and found ratios of 0.43 / 0.38 / 0.30 respectively —
worse than round 2's 0.63, and worse than the 2.0 margin covered (needs
>=1/0.30 ~= 3.33x). Two consequences, both fixed here:

  - `_LOCAL_OVERCOUNT_CEILING` is now 4.0 (safe over the 3.33x worst
    measured), so `_min_possible_batches` no longer falsely refuses on a
    backlog of one-token/empty exchanges whose true batch count fits the
    cap (case [3b] below: 15,000 "ok"/"sure"-shaped exchanges).
  - the pre-cut trigger and target in `_enforce_hard_budget` were reusing
    `_PESSIMISTIC_SUMMARY_SCALE` (2.0) — a same-looking but UNRELATED
    constant that sizes an unrelated pessimistic map-reduce estimate
    elsewhere in this file, not this file's overcount margin. Both now use
    `_LOCAL_OVERCOUNT_CEILING`, so a payload whose true total already fits
    under `limit` is not needlessly pre-cut just because its local count
    (itself inflated by the same framing overhead) crosses a threshold
    that was never a valid proxy for "still over `limit`" (case [10]
    below: 3,600 "k"/"k" exchanges).

The test fakes below model the CORRECTED calibration: a whole-list
`count_tokens(messages)` call is accurate (matches the prod "token scale
1.00x" reading on ordinary content); a SINGLE-message `count_tokens([m])`
call reads high by a fixed framing cost, worst on short messages — the
bucketed scale figures (0.63 / 0.77 / 0.89 / 0.98) are the hostile
reviewer's own measured numbers from `floor.py`, not invented here. A
separate, even-shorter bucket (round 3's own measured 0.43 / 0.38 / 0.30
for "ok"/"sure", "k"/"k" and empty turns) is added where those specific
cases are tested ([3b], [10]).

/tokenize itself is never wired through httpx in these fakes (main.py's
`count_tokens_exact` is replaced directly, the same technique
test_v3194_guard_g3a.py and others in this suite use) — so "/tokenize was
never called" is checked by spying on `main.count_tokens_exact` itself,
not on the HTTP layer beneath a function these fakes bypass entirely.

    python test_rca_20260925_tokenize_precut.py
"""

import asyncio
import logging
import os
import sys
import tempfile

os.environ.pop("MODEL_REPO", None)  # char/4 local estimator — no tokenizer download
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="rca0925-precut-")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["MAX_MODEL_LEN"] = "32768"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "600"
os.environ["COMPACTOR_SUMMARY_INPUT_RESERVE"] = "600"
os.environ["COMPACTOR_MAX_SUMMARY_CALLS"] = "4"

import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}{': ' + detail if detail else ''}")
        FAILED.append(label)


# =============================================================================
# CORRECTED calibration (hostile review round 2): the served tokenizer is
# mistral_common-backed. A WHOLE-LIST count_tokens(messages) call is
# accurate; a SINGLE-message count_tokens([m]) call reads high by a fixed
# per-call framing cost, worse the shorter the message. Bucket scale
# figures are the reviewer's own (floor.py): 0.63 for 2-6 words, 0.77 for
# 5-15, 0.89 for 10-40, ~0.98 (the "+0.4% to +2.2%" figure) beyond that.
# =============================================================================
def _word_count(m):
    text = m.get("content")
    text = text if isinstance(text, str) else ""
    return max(1, len(text.split()))


def _scale_for_words(n):
    if n <= 6:
        return 0.63
    if n <= 15:
        return 0.77
    if n <= 40:
        return 0.89
    return 0.98


def fake_count_tokens(messages):
    """The local tokenizer: accurate on a whole list, high on a singleton."""
    if len(messages) == 1:
        n = _word_count(messages[0])
        return max(n, int(round(n / _scale_for_words(n))))
    return sum(_word_count(m) for m in messages)


def fake_count_tokens_exact(messages, add_generation_prompt=None):
    """vLLM's own /tokenize: always the true whole-conversation count."""
    if not messages:
        return 0
    return sum(_word_count(m) for m in messages)


_real_count_tokens = main.count_tokens
_real_count_tokens_exact = main.count_tokens_exact
main.count_tokens = fake_count_tokens
main.count_tokens_exact = fake_count_tokens_exact


def _assistant_turn(i, words=90):
    return {"role": "assistant", "content": f"idx{i} " + " ".join(["word"] * words)}


def _user_turn(i, words=3):
    return {"role": "user", "content": f"idx{i} " + " ".join(["hi"] * words)}


def _huge_backlog(n_pairs=2095, user_words=3, assistant_words=90):
    out = []
    for i in range(n_pairs):
        out.append(_user_turn(i, user_words))
        out.append(_assistant_turn(i, assistant_words))
    return out


class _Capture:
    def __enter__(self):
        self.records = []
        self.handler = logging.Handler()
        self.handler.emit = lambda r: self.records.append(r.getMessage())
        logging.getLogger("compactor").addHandler(self.handler)
        return self

    def __exit__(self, *a):
        logging.getLogger("compactor").removeHandler(self.handler)

    def text(self):
        return "\n".join(self.records)


class _ExactSpy:
    """Wraps main.count_tokens_exact, counting calls and their sizes —
    the real thing /tokenize would be asked to measure, without going
    through httpx (these fakes replace count_tokens_exact directly, the
    same technique this suite already uses elsewhere)."""

    def __init__(self, real):
        self.calls: list[int] = []
        self._real = real

    def __call__(self, messages, add_generation_prompt=None):
        self.calls.append(len(messages))
        return self._real(messages, add_generation_prompt)


class _RefusingClient:
    """A chat client that fails the test if summarize() ever reaches it —
    used only where the backlog is expected to refuse before any call."""

    async def post(self, *a, **kw):
        raise AssertionError("summarize() must not reach the chat endpoint here")


class _SummarizingClient:
    """A chat client that always answers with a canned summary — used
    where summarize() is expected to actually run map-reduce or a single
    call."""

    async def post(self, *a, **kw):
        class R:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "a summary of it all"},
                                      "finish_reason": "stop"}]}

            def raise_for_status(self):
                pass

        return R()


async def _run(turns, client):
    return await main.summarize(client, turns)


# =============================================================================
# [1] summarize() refuses a huge backlog WITHOUT calling /tokenize at all.
# =============================================================================
print("\n[1] summarize() refuses a starved backlog on the local floor alone")

huge = _huge_backlog(n_pairs=2095, user_words=50, assistant_words=2000)

exact_spy = _ExactSpy(fake_count_tokens_exact)
main.count_tokens_exact = exact_spy

before = main.compaction_counters()["cap_refused"]
with _Capture() as logs:
    out, deferred = asyncio.run(_run(huge, _RefusingClient()))
after = main.compaction_counters()["cap_refused"]

check(len(exact_spy.calls) == 0, "/tokenize (count_tokens_exact) was never "
      "called for the huge backlog", f"calls: {exact_spy.calls}")
check(after == before + 1, "cap_refused counted the refusal",
      f"before={before} after={after}")
check(out == "" and deferred == huge,
      "the huge backlog is returned verbatim, deferred, exactly as the "
      "post-batching cap refusal already did")
check("compaction skipped" in logs.text(),
      "the refusal path logs the SAME shape as the post-batching refusal",
      logs.text()[:300])
check(
    "at least" in logs.text() and "without spending a /tokenize call" in logs.text(),
    "and says it is a certain floor, not the real count",
    logs.text()[:400],
)

main.count_tokens_exact = fake_count_tokens_exact


# =============================================================================
# [2] a normal-size backlog still measures with /tokenize, unchanged
# =============================================================================
print("\n[2] a normal-size backlog still measures with /tokenize, unchanged")

small = _huge_backlog(n_pairs=7, user_words=20, assistant_words=40)
exact_spy2 = _ExactSpy(fake_count_tokens_exact)
main.count_tokens_exact = exact_spy2

out2, deferred2 = asyncio.run(_run(small, _SummarizingClient()))
check(len(exact_spy2.calls) >= 1, "the normal-size path still calls /tokenize",
      f"calls={len(exact_spy2.calls)}")
check(deferred2 == [], "a small backlog is fully summarized, nothing deferred")
check(out2 == "a summary of it all", "and the summary content is unchanged")

main.count_tokens_exact = fake_count_tokens_exact


# =============================================================================
# [3] finding 1, THE REGRESSION: a backlog of SHORT messages, shaped so the
#     STALE 1.10 ceiling's floor would have exceeded the cap while the TRUE
#     (uninflated) batch count does not.
#
# Measured directly against `_min_possible_batches`, not through the full
# `summarize()` path: `_chunk_to_budget` (used both by the post-batching
# cap check and by map-reduce itself) prices each turn with the SAME
# per-message `count_tokens([m])` singleton call `_min_possible_batches`
# does, so it inherits a version of the same inflation — a separate,
# pre-existing characteristic of `_chunk_to_budget`'s OWN batching that is
# not one of this round's findings and not touched here (this file's fixed
# calibration bucket, applied uniformly to every turn, makes the two track
# too closely to isolate one from the other end to end; the reviewer's own
# real-tokenizer measurement did not have that artifact, since real
# content is not uniformly 2-6 words). Comparing the certain floor
# directly against a TRUE, uninflated reference isolates exactly what
# `_LOCAL_OVERCOUNT_CEILING` governs.
# =============================================================================
print("\n[3] the certain floor is not falsely over cap on short messages "
      "(finding 1)")

budget = min(main.MAX_MODEL_LEN,
             max(256, main.MAX_MODEL_LEN - main.SUMMARY_MAX_TOKENS - main.SUMMARY_INPUT_RESERVE))
cap = main.MAX_SUMMARY_CALLS_PER_REQUEST


def _true_batches_reference(turns, budget):
    """The same greedy packing `_min_possible_batches` uses, but on TRUE
    (uninflated) per-message word counts — the reference this test's
    "not falsely refused" claim is measured against."""
    n_batches = 1
    current = 0
    has_item = False
    for m in turns:
        t = _word_count(m)
        if has_item and current + t > budget:
            n_batches += 1
            current = 0
            has_item = False
        current += t
        has_item = True
    return n_batches


def _short_backlog(n_pairs):
    return _huge_backlog(n_pairs=n_pairs, user_words=4, assistant_words=4)


# Short (2-6-word) turns — the reviewer's own worst bucket (scale 0.63) —
# sized so the TRUE batch count sits comfortably under the cap.
short_backlog = _short_backlog(11000)
true_ref_batches = _true_batches_reference(short_backlog, budget)
floor_batches = main._min_possible_batches(short_backlog, budget)
check(
    true_ref_batches <= cap,
    "fixture: the TRUE (uninflated) batch count is under the cap",
    f"true_ref_batches={true_ref_batches}",
)
check(
    floor_batches <= cap,
    "*** THE FIX: the certain floor does not exceed the cap when the true "
    "count does not either",
    f"floor_batches={floor_batches}, true_ref_batches={true_ref_batches}, "
    f"cap={cap}",
)


# =============================================================================
# [4] a genuinely oversized SHORT-message backlog still refuses (certainty
#     is not lost — only the false positives are).
# =============================================================================
print("\n[4] a genuinely oversized short-message backlog still refuses")

tiny_huge = _short_backlog(40000)
realb4 = len(main._chunk_to_budget(tiny_huge, budget, 1.0))
floor_batches4 = main._min_possible_batches(tiny_huge, budget)
check(
    realb4 > cap,
    "fixture: this backlog genuinely needs more than the cap, on the real "
    "(_chunk_to_budget) count too",
    f"realb={realb4}",
)
check(
    floor_batches4 > cap,
    "the certain floor still exceeds the cap for a genuine case",
    f"floor_batches={floor_batches4}",
)

exact_spy4 = _ExactSpy(fake_count_tokens_exact)
main.count_tokens_exact = exact_spy4

before4 = main.compaction_counters()["cap_refused"]
out4, deferred4 = asyncio.run(_run(tiny_huge, _RefusingClient()))
after4 = main.compaction_counters()["cap_refused"]
check(
    after4 == before4 + 1,
    "a genuinely huge short-message backlog is still refused",
    f"before={before4} after={after4}",
)
check(len(exact_spy4.calls) == 0, "and still without calling /tokenize",
      f"calls={exact_spy4.calls}")

main.count_tokens_exact = fake_count_tokens_exact


# =============================================================================
# ROUND 3 calibration: the hostile reviewer measured DEGENERATE (near-empty)
# turns directly against vLLM's own tokenizer in the prod image
# (angreg/zions-light-ai@sha256:f3fafa3e1949…998d2, v3.1.9-cu12, the
# mistral_common/tekken tokenizer this deployment serves) — not the
# word-count buckets above, since "ok", "sure" and "k" are each "1 word" to
# a word-count model but price very differently in a real chat template.
# Measured real/local ratios: 0.43 for "ok"/"sure", 0.38 for "k"/"k", 0.30
# for empty turns ("", "") — the worst of the three, and the one
# `_LOCAL_OVERCOUNT_CEILING` (4.0) is now sized against.
#
# This also demonstrated round 2's OWN assumption was wrong: round 2's
# comments (and `fake_count_tokens` above) treat a WHOLE-LIST
# `count_tokens(messages)` call as ACCURATE, with only a SINGLETON
# `count_tokens([m])` call reading high. On degenerate content the
# WHOLE-LIST call reads high too (the prod reproduction: 3,600 "k"/"k"
# exchanges measured local=47,227 whole-list vs real=18,406 — ratio
# 18406/47227 ~= 0.39, matching the reviewer's separately-measured 0.38).
# These fakes model that: BOTH the whole-list and the singleton call
# overcount by the same fixed per-turn amount, since the tokenizer's
# per-message framing tokens do not shrink away when messages are grouped
# in the same list — only the true content itself does.
# =============================================================================
def _make_degenerate_fakes(true_per_turn, local_per_turn):
    def fake_local(messages):
        return int(round(len(messages) * local_per_turn))

    def fake_exact(messages, add_generation_prompt=None):
        if not messages:
            return 0
        return int(round(len(messages) * true_per_turn))

    return fake_local, fake_exact


# =============================================================================
# [3b] finding 1 (round 3): the certain floor is not falsely over cap on
#      ONE-TOKEN exchanges. Reproduces the hostile reviewer's own numbers:
#      15,000 "k"/"k" exchanges (measured ratio 0.38) produced a floor of 5
#      against a 4-call cap under the STALE 2.0 ceiling, while the true
#      count needed only 4 batches.
# =============================================================================
print("\n[3b] the certain floor is not falsely over cap on ONE-TOKEN "
      "exchanges (round 3, measured ratio 0.38)")

# Clean numbers reproducing the measured ratio exactly (0.38) and, at
# n=15,000 exchanges, the reviewer's own reported floor/real split
# (floor=5, real=4) under the stale 2.0 ceiling — verified directly below,
# not asserted from the constant's current value.
_K_LOCAL_PER_TURN = 10.0
_K_TRUE_PER_TURN = 3.8  # ratio 3.8/10.0 = 0.38, the reviewer's own number
_k_local, _k_exact = _make_degenerate_fakes(_K_TRUE_PER_TURN, _K_LOCAL_PER_TURN)

k_backlog = []
for _ in range(15000):
    k_backlog += [{"role": "user", "content": "k"}, {"role": "assistant", "content": "k"}]

main.count_tokens = _k_local
main.count_tokens_exact = _k_exact

true_batches_3b = -(-len(k_backlog) * _K_TRUE_PER_TURN // budget)  # ceil division
floor_batches_3b = main._min_possible_batches(k_backlog, budget)
check(
    true_batches_3b <= cap,
    "fixture: the TRUE batch count for 15,000 one-token exchanges is under "
    "the cap",
    f"true_batches_3b={true_batches_3b}",
)
check(
    floor_batches_3b <= cap,
    "*** THE FIX: the certain floor does not falsely exceed the cap on "
    "one-token exchanges (round 3) — this check FAILS at the stale 2.0 "
    "ceiling (floor=5 there, verified separately by mutation)",
    f"floor_batches_3b={floor_batches_3b}, true_batches_3b={true_batches_3b}, "
    f"cap={cap}",
)

exact_spy3b = _ExactSpy(_k_exact)
main.count_tokens_exact = exact_spy3b
before3b = main.compaction_counters()["cap_refused"]
out3b, deferred3b = asyncio.run(_run(k_backlog, _SummarizingClient()))
after3b = main.compaction_counters()["cap_refused"]
check(
    after3b == before3b,
    "*** THE FIX: 15,000 one-token exchanges are NOT wrongly refused "
    "(cap_refused must not increment)",
    f"before={before3b} after={after3b}",
)
check(deferred3b == [], "and the backlog is actually summarized, nothing "
      "deferred by a false refusal", f"deferred={len(deferred3b)}")

main.count_tokens = fake_count_tokens
main.count_tokens_exact = fake_count_tokens_exact


# =============================================================================
# [5] finding 2, THE REGRESSION: with turns priced near the framing floor,
#     an earlier per-turn pre-cut could land mid-pair. The fixed pre-cut
#     must never forward an assistant-first array.
# =============================================================================
print("\n[5] pre-cut never leaves an assistant turn first (finding 2)")

tiny_payload = (
    [{"role": "system", "content": "system prompt"}]
    + _huge_backlog(n_pairs=8000, user_words=1, assistant_words=1)
    + [{"role": "user", "content": "recent q"}, {"role": "assistant", "content": "recent a"}]
)
LIMIT5 = 2000
report5: dict = {}
result5 = main._enforce_hard_budget(list(tiny_payload), limit=LIMIT5, protect_system=1,
                                     report=report5)
turns5 = [m for m in result5 if m.get("role") != "system"]
check(
    bool(turns5) and turns5[0]["role"] == "user",
    "*** THE FIX: the forwarded result never starts on an assistant turn",
    f"first turn role: {turns5[0]['role'] if turns5 else None}",
)
check(
    fake_count_tokens_exact(result5) <= LIMIT5,
    "and it never exceeds the verified limit",
    f"{fake_count_tokens_exact(result5)} vs {LIMIT5}",
)

# [5b] the SAME defect, forced deterministically: with uniform per-message
# costs, a naive single-turn cut only lands mid-pair when the ceiling
# happens to fall between an ODD number of turns' worth of content and an
# even one — [5] above hit that by chance (8000 tiny pairs); this
# constructs the ceiling explicitly, one message short of a full pair, so
# a per-turn cutter is FORCED to stop after exactly 1 turn (odd) if it is
# not pair-aware. Calls `_precut_to_local_ceiling` directly so the
# boundary is exact, not incidental.
print("\n[5b] pre-cut, forced to the exact odd-cut boundary, still cuts "
      "whole pairs only (finding 2)")

_p5b_payload = (
    [{"role": "system", "content": "system prompt"}]
    + _huge_backlog(n_pairs=50, user_words=5, assistant_words=5)
    + [{"role": "user", "content": "recent q"}, {"role": "assistant", "content": "recent a"}]
)
_p5b_turns = [m for m in _p5b_payload if m.get("role") != "system"]
_p5b_floor = set(main._aligned_recent_tail(_p5b_payload))
_p5b_above_floor = [
    i for i, m in enumerate(_p5b_payload)
    if m.get("role") != "system" and i not in _p5b_floor
]
_p5b_per = [main.count_tokens([m]) for m in _p5b_payload]
_p5b_total = sum(_p5b_per)
# One message short of cutting the full first pair — a naive per-turn
# cutter stops at cut=1 (odd) here; the pair-aware version must still cut
# the whole first pair (cut=2) rather than stop mid-pair.
_p5b_ceiling = _p5b_total - _p5b_per[_p5b_above_floor[0]]

_p5b_result, _p5b_cut = main._precut_to_local_ceiling(list(_p5b_payload), _p5b_ceiling)
_p5b_result_turns = [m for m in _p5b_result if m.get("role") != "system"]
check(
    bool(_p5b_result_turns) and _p5b_result_turns[0]["role"] == "user",
    "*** THE FIX: at the exact odd-cut boundary, the result still starts "
    "on a user turn",
    f"cut={_p5b_cut}, first role={_p5b_result_turns[0]['role'] if _p5b_result_turns else None}",
)
check(
    _p5b_cut % 2 == 0,
    "and the cut count itself is a whole number of exchanges (even)",
    f"cut={_p5b_cut}",
)


# =============================================================================
# [6] finding 2, THE REGRESSION: when pre-cut ALONE already makes the
#     payload fit — so the round loop below has NOTHING left to shed — the
#     standard WARNING (not just the INFO promise) must still fire, and
#     fill-back must still run on the ORIGINAL (pre-pre-cut) list, so it
#     can restore turns pre-cut itself removed, not only ones a round-loop
#     shed would have (there is no round-loop shed here at all).
#
# Needs its own, deliberately EXTREME calibration: with the file's shared
# worst-case bucket (round 3's own worst measured case, empty turns, ratio
# ~0.30 i.e. ~3.33x), pre-cut stopping at a local sum of
# `_LOCAL_OVERCOUNT_CEILING` (4.0x) `limit` still leaves a TRUE remainder of
# at least 4.0x`limit`/3.33 ~= 1.2x`limit` — i.e. still over `limit`, so the
# round loop always has SOME real work left under that calibration (as
# [7]/[8]/[9] above all show). To reach the specific "pre-cut alone already
# fits" branch this finding is about, this fixture uses a per-message
# overcount ratio of 8x on singleton calls — well past anything measured,
# but the EXACT case finding 2's bullets 2 and 3 need isolated: with ratio
# 8x, a local sum of `_LOCAL_OVERCOUNT_CEILING` (4.0x) `limit` implies a
# true remainder of at most 4.0x`limit`/8 = 0.5x`limit`, comfortably
# already under `limit`.
# =============================================================================
print("\n[6] pre-cut-alone-fits still logs the normal WARNING and lets "
      "fill-back reach PRE-CUT turns (finding 2)")

_EXTREME_RATIO = 8.0


def _extreme_local(messages):
    if len(messages) == 1:
        return int(_word_count(messages[0]) * _EXTREME_RATIO)
    return sum(_word_count(m) for m in messages)


def _extreme_exact(messages, add_generation_prompt=None):
    if not messages:
        return 0
    return sum(_word_count(m) for m in messages)


payload6 = (
    [{"role": "system", "content": "system prompt"}]
    + _huge_backlog(n_pairs=3000, user_words=3, assistant_words=3)
    + [{"role": "user", "content": "recent q"}, {"role": "assistant", "content": "recent a"}]
)
LIMIT6 = 4000
report6: dict = {}
_fillback_calls = {"n": 0}
_real_fill_back = main._fill_back_dropped_turns


def _spy_fill_back(*a, **kw):
    _fillback_calls["n"] += 1
    return _real_fill_back(*a, **kw)


main.count_tokens = _extreme_local
main.count_tokens_exact = _extreme_exact
main._fill_back_dropped_turns = _spy_fill_back
with _Capture() as logs6:
    result6 = main._enforce_hard_budget(list(payload6), limit=LIMIT6, protect_system=1,
                                         report=report6)
main._fill_back_dropped_turns = _real_fill_back
main.count_tokens = fake_count_tokens
main.count_tokens_exact = fake_count_tokens_exact

turns6_before_fillback_pool = len(payload6) - 1 - 2  # everything but system + recent pair
check(report6.get("fits") is True, "fixture: the guard fits the payload",
      f"{report6}")
check(
    "hard budget enforced:" in logs6.text(),
    "*** THE FIX: the normal WARNING fires even when pre-cut alone made it "
    "fit — not only the INFO promise",
    logs6.text()[-800:],
)
check(
    _fillback_calls["n"] >= 1,
    "*** THE FIX: fill-back still ran on this path",
    f"calls={_fillback_calls['n']}",
)
check(
    report6.get("dropped_turns", 0) > 0,
    "and the report's dropped_turns reflects the real drop, not a bare 0",
    f"{report6}",
)
# finding 3: the WARNING's `{total} ->` prefix must include the ORIGINAL
# (pre-pre-cut) local count, not just the post-pre-cut remainder — else the
# line reads as if the whole request were only as large as what pre-cut
# left behind. Computed independently here (the same whole-list call
# `_enforce_hard_budget` itself makes to decide the pre-cut trigger), not
# read back off the report, so a regression that drops the prefix (or
# prints the wrong number) is caught either way.
_expected_original6 = _extreme_local(payload6)
check(
    f"{_expected_original6} (original local count, pre-cut before "
    "measurement) ->" in logs6.text(),
    "*** THE FIX: the WARNING states the ORIGINAL local count (finding 3), "
    "not just the post-pre-cut remainder",
    f"expected {_expected_original6!r} in original-count prefix; "
    f"{logs6.text()[-800:]}",
)
kept_turns6 = [m for m in result6 if m.get("role") != "system"]
# hostile review round 2 (of THIS fix): a weaker check here — "more than
# the newest 2 turns survive" — passes even when fill-back restores
# NOTHING, because the pre-cut ceiling (_LOCAL_OVERCOUNT_CEILING x limit)
# already leaves several hundred turns beyond the floor before fill-back
# ever runs. The actual
# proof that fill-back reached INTO pre-cut's shed set (not just the
# round loop's, which is empty here) is that it restored something at
# all — visible only in its own "filled back N" log line, since
# `_fill_back_dropped_turns` logs only when `restored > 0`. Reverting the
# ORIGINAL-list fix (passing the post-precut list as "original") makes
# every turn already spoken for, so the diff this pass computes is empty,
# it restores 0, and this line never appears.
check(
    "filled back" in logs6.text() and "0 previously-shed" not in logs6.text(),
    "*** THE FIX: fill-back actually restored something — reachable only "
    "because it was given the ORIGINAL (pre-pre-cut) list, since the "
    "round loop itself shed nothing here",
    logs6.text()[-500:],
)
check(
    len(kept_turns6) > 2,
    "fixture: more than the newest pair survives",
    f"kept {len(kept_turns6)} turn(s)",
)


# =============================================================================
# [7] _enforce_hard_budget pre-cuts on local counts before its first
#     /tokenize call — the original end-to-end reproduction, moderate
#     (not 1-2 word) content.
# =============================================================================
print("\n[7] _enforce_hard_budget pre-cuts on local counts before its first "
      "/tokenize call")

payload7 = (
    [{"role": "system", "content": "system prompt " * 10}]
    + _huge_backlog(n_pairs=2095, user_words=30, assistant_words=90)
    + [
        {"role": "user", "content": "recent q1"},
        {"role": "assistant", "content": "recent a1"},
    ]
)
LIMIT7 = 20672
exact_spy7 = _ExactSpy(fake_count_tokens_exact)
main.count_tokens_exact = exact_spy7
report7: dict = {}
result7 = main._enforce_hard_budget(list(payload7), limit=LIMIT7, protect_system=1,
                                     report=report7)
main.count_tokens_exact = fake_count_tokens_exact

check(len(exact_spy7.calls) >= 1, "at least one /tokenize (exact) call was made",
      f"calls={exact_spy7.calls}")
check(
    exact_spy7.calls[0] < len(payload7) // 2,
    "*** THE FIX: the FIRST /tokenize call measures the PRE-CUT remainder, "
    "not the full ~4,200-message payload",
    f"first call size {exact_spy7.calls[0]} vs full payload {len(payload7)}",
)
check(report7.get("fits") is True, "the guard still fits the payload", f"{report7}")
check(
    fake_count_tokens_exact(result7) <= LIMIT7,
    "the forwarded result is within the verified limit",
    f"measured {fake_count_tokens_exact(result7)} vs limit {LIMIT7}",
)


# =============================================================================
# [8] a normal-size payload's guard behavior is unchanged by the pre-cut.
# =============================================================================
print("\n[8] a normal-size payload's guard behavior is unchanged by the pre-cut")

small_payload = (
    [{"role": "system", "content": "system prompt " * 10}]
    + _huge_backlog(n_pairs=6, user_words=30, assistant_words=90)
    + [
        {"role": "user", "content": "recent q1"},
        {"role": "assistant", "content": "recent a1"},
    ]
)
small_limit = 6000
exact_spy8 = _ExactSpy(fake_count_tokens_exact)
main.count_tokens_exact = exact_spy8
report8: dict = {}
result8 = main._enforce_hard_budget(list(small_payload), limit=small_limit,
                                     protect_system=1, report=report8)
main.count_tokens_exact = fake_count_tokens_exact
check(
    exact_spy8.calls and exact_spy8.calls[0] == len(small_payload),
    "a payload under the pre-cut threshold is measured whole, exactly as "
    "before this fix",
    f"first call size {exact_spy8.calls[0] if exact_spy8.calls else None} "
    f"vs payload {len(small_payload)}",
)
check(report8.get("measured", 10**9) <= small_limit, "and it still fits",
      f"{report8}")


# =============================================================================
# [9] finding 4: the pre-cut TRIGGER uses the real local tokenizer, not
#     chars/4 — a dense script (no word-boundary spaces, standing in for
#     fully-pointed Hebrew, which this model's real tokenizer prices near
#     1 token/char) must still trigger pre-cut once genuinely over the
#     `_LOCAL_OVERCOUNT_CEILING`-multiple-of-limit ceiling on a REAL local
#     count, even though a char/4 proxy would have read it as much smaller.
# =============================================================================
print("\n[9] the pre-cut trigger uses the real local tokenizer, not chars/4 "
      "(finding 4)")


def _word_count_dense(m):
    # 1 "token" per character (no spaces) — a separate fake so this case
    # does not depend on the word-based model above, and directly proves
    # the trigger reads a real per-character-priced script correctly.
    text = m.get("content")
    text = text if isinstance(text, str) else ""
    return max(1, len(text))


def _fake_count_tokens_dense(messages):
    if len(messages) == 1:
        return _word_count_dense(messages[0])
    return sum(_word_count_dense(m) for m in messages)


def _fake_count_tokens_exact_dense(messages, add_generation_prompt=None):
    if not messages:
        return 0
    return sum(_word_count_dense(m) for m in messages)


def _dense_backlog(n_pairs, tokens_per_turn):
    out = []
    for i in range(n_pairs):
        out.append({"role": "user", "content": "א" * tokens_per_turn})
        out.append({"role": "assistant", "content": "א" * tokens_per_turn})
    return out


main.count_tokens = _fake_count_tokens_dense
main.count_tokens_exact = _fake_count_tokens_exact_dense

LIMIT9 = 5000
dense_payload = (
    [{"role": "system", "content": "s"}]
    + _dense_backlog(n_pairs=10, tokens_per_turn=1200)  # 20 turns * 1200 = 24,000 real tokens
    + [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
)
real_total_9 = _fake_count_tokens_exact_dense(dense_payload)
check(
    real_total_9 > LIMIT9 * main._LOCAL_OVERCOUNT_CEILING,
    "fixture: the dense-script payload is genuinely over the "
    "_LOCAL_OVERCOUNT_CEILING-multiple of the limit on a REAL local count",
    f"real_total={real_total_9}, "
    f"ceiling={LIMIT9 * main._LOCAL_OVERCOUNT_CEILING}",
)
exact_spy9 = _ExactSpy(_fake_count_tokens_exact_dense)
main.count_tokens_exact = exact_spy9
report9: dict = {}
result9 = main._enforce_hard_budget(list(dense_payload), limit=LIMIT9, protect_system=1,
                                     report=report9)
check(
    exact_spy9.calls and exact_spy9.calls[0] < len(dense_payload),
    "*** THE FIX: the dense-script payload still triggers pre-cut — the "
    "first /tokenize call is on fewer messages than the full payload",
    f"first call size {exact_spy9.calls[0] if exact_spy9.calls else None} "
    f"vs full payload {len(dense_payload)}",
)
check(
    _fake_count_tokens_exact_dense(result9) <= LIMIT9,
    "and the result still fits the verified limit",
    f"{_fake_count_tokens_exact_dense(result9)} vs {LIMIT9}",
)


# =============================================================================
# [10] the pre-cut trigger/target (round 3) reproduces the reviewer's own
#      numbers: 3,600 "k"/"k" exchanges (measured ratio 0.38) whose TRUE
#      total (18,000) already fits under a prod-shaped 20,672-token limit
#      must NOT be pre-cut at all — the stale ceiling (whether the old 2.0
#      `_LOCAL_OVERCOUNT_CEILING`, or reusing the unrelated
#      `_PESSIMISTIC_SUMMARY_SCALE`) pre-cut roughly 2,878 of these turns
#      in the real incident even though nothing needed to be shed.
# =============================================================================
print("\n[10] the pre-cut trigger/target does not over-cut ONE-TOKEN "
      "exchanges whose true total already fits (round 3, finding 2)")

_K10_LOCAL_PER_TURN = 6.5789  # local whole-list reads this high per turn
_K10_TRUE_PER_TURN = 2.5      # true (vLLM) cost per turn -> ratio ~0.38
_k10_local, _k10_exact = _make_degenerate_fakes(_K10_TRUE_PER_TURN, _K10_LOCAL_PER_TURN)

LIMIT10 = 20672  # the prod limit named in the 2026-09-25 RCA
k10_payload = (
    [{"role": "system", "content": "s"}]
    + [m for _ in range(3600) for m in (
        {"role": "user", "content": "k"}, {"role": "assistant", "content": "k"}
    )]
    + [{"role": "user", "content": "k"}, {"role": "assistant", "content": "k"}]
)
turns10 = [m for m in k10_payload if m.get("role") != "system"]

main.count_tokens = _k10_local
main.count_tokens_exact = _k10_exact

check(
    _k10_exact(k10_payload) <= LIMIT10,
    "fixture: the TRUE total of 3,600+1 'k'/'k' exchanges already fits "
    "the limit — nothing needs to be shed at all",
    f"real_total={_k10_exact(k10_payload)}, limit={LIMIT10}",
)
check(
    _k10_local(k10_payload) > LIMIT10 * 2.0,
    "fixture: the (inflated) WHOLE-LIST local count is still over the "
    "STALE 2.0x ceiling — this is exactly what used to wrongly trigger "
    "pre-cut",
    f"local_total={_k10_local(k10_payload)}, 2x limit={LIMIT10 * 2.0}",
)

report10: dict = {}
result10 = main._enforce_hard_budget(list(k10_payload), limit=LIMIT10, protect_system=1,
                                      report=report10)
check(
    report10.get("dropped_turns", -1) == 0,
    "*** THE FIX: nothing is pre-cut (or shed) when the true total already "
    "fits — this check FAILS at the stale 2.0 ceiling (an actual mutation "
    "run drops ~2,878 turns here)",
    f"{report10}",
)
check(
    len(result10) == len(k10_payload),
    "and every turn survives",
    f"kept {len(result10)} of {len(k10_payload)}",
)

main.count_tokens = fake_count_tokens
main.count_tokens_exact = fake_count_tokens_exact


# =============================================================================
# [11] REGRESSION GUARD: her actual payload (the 2026-09-25 RCA: ~4,196
#      messages, ~7.8M tokens) must still be refused by summarize() and
#      still pre-cut (and still fit) through _enforce_hard_budget, exactly
#      as before round 3 — the new, larger ceiling must not let a
#      genuinely enormous backlog slip past either guard.
# =============================================================================
print("\n[11] her actual payload is still refused and still pre-cut "
      "(regression guard, unchanged by round 3)")

her_backlog = _huge_backlog(n_pairs=2098, user_words=200, assistant_words=2500)
before11 = main.compaction_counters()["cap_refused"]
out11, deferred11 = asyncio.run(_run(her_backlog, _RefusingClient()))
after11 = main.compaction_counters()["cap_refused"]
check(
    after11 == before11 + 1,
    "*** her payload is still refused by summarize() without calling "
    "/tokenize",
    f"before={before11} after={after11}",
)

her_payload = (
    [{"role": "system", "content": "system prompt " * 10}]
    + her_backlog
    + [{"role": "user", "content": "recent q1"}, {"role": "assistant", "content": "recent a1"}]
)
LIMIT11 = 20672
exact_spy11 = _ExactSpy(fake_count_tokens_exact)
main.count_tokens_exact = exact_spy11
report11: dict = {}
result11 = main._enforce_hard_budget(list(her_payload), limit=LIMIT11, protect_system=1,
                                      report=report11)
main.count_tokens_exact = fake_count_tokens_exact
check(
    exact_spy11.calls and exact_spy11.calls[0] < len(her_payload) // 2,
    "*** her payload is still pre-cut before the first /tokenize call — "
    "the first call measures a fraction of the ~4,200-message original",
    f"first call size {exact_spy11.calls[0] if exact_spy11.calls else None} "
    f"vs full payload {len(her_payload)}",
)
check(report11.get("fits") is True, "and the result still fits the limit",
      f"{report11}")
check(
    fake_count_tokens_exact(result11) <= LIMIT11,
    "verified directly too",
    f"{fake_count_tokens_exact(result11)} vs {LIMIT11}",
)


main.count_tokens = _real_count_tokens
main.count_tokens_exact = _real_count_tokens_exact


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nall checks passed")
