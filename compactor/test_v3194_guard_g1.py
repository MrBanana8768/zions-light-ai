"""v3.1.9.4, lane v3194-guard, G1 (hostile pass #14, P14-2): the fresh-span
preview in compact_if_needed and summarize() each measured /tokenize
SEPARATELY on the SAME list, a few dozen lines apart in the same request.
If the preview's POST answered and summarize()'s own call failed a moment
later, summarize() fell back to the pessimistic 2.0x scale, packed the same
turns into MORE map-reduce batches than the preview's reserve priced for,
and could return a summary up to one SUMMARY_MAX_TOKENS batch (2,048 tokens
at the shipped default) larger than what the window check reserved. The
stand-in then carries tokens nobody reserved room for; P12-5's order sheds
memory first and, if that is not enough, her previous exchange, to find it.

Reproduced against 26ea2f2 (unfixed) with SP\\p14\\flap.py: 4 of 24
synthetic spans came back over their reserve, every one on the "preview
answers, summarize() FAILS" schedule; the reverse flap (preview fails,
summarize() answers) over-reserves, which is the safe direction, and
both-answer / both-fail already agreed by construction (same scale, same
budget expression, same _chunk_to_budget — p14's own "Verified sound"
finding). flap.py CANNOT exercise a fix placed inside count_tokens_exact:
its stub replaces main.count_tokens_exact WHOLESALE (`main.
count_tokens_exact = exact`), so both the preview's and summarize()'s calls
go through the SAME stub function, never the real one. This file patches
httpx.post instead, one level down, so the real count_tokens_exact (and
its memo) actually runs, and counts real POSTs to prove the mechanism, not
just the output numbers.

THE FIX (main.py, count_tokens_exact, ~line 1316): a small, content-keyed
memo of SUCCESSFUL /tokenize answers, held in a REQUEST-scoped
`contextvars.ContextVar` (`_COUNT_TOKENS_EXACT_MEMO_CTX`), turned on by
`compact_if_needed` at its own top. The preview's list
(`_fresh_span_preview`) and summarize()'s list (`to_summarize`) are the
SAME messages in the SAME order (p14's own Q2 finding), so once the
preview's call answers, summarize()'s later call on that content is
served from the memo — no second POST, so no second chance to fail. A
FAILURE is never memoized (see the comment beside _TOKENIZER_TRIED for why
caching a miss would be worse than this bug).

NOT a process-wide cache: a first draft was (a bare bounded dict, alive
for the process's whole life) and it broke two EXISTING suites this lane
does not own — test_tokenize_flags.py's `[6]` and test_tokenize_repair.py's
`test_every_empty_shape_a_client_can_send_is_repaired` both call `main.
count_tokens_exact` directly, outside any request, reusing content earlier
sections had already measured, and asserted on a /tokenize call a stub
captured; a memo hit meant no call was made and the stub's capture stayed
empty. The context-scoped design fixes that categorically: the memo only
activates for the ONE call chain this fix is about (compact_if_needed's
own preview, then summarize()), and every other caller — a direct unit
test, `_enforce_hard_budget`'s own calls, `_sent_token_size` — sees
exactly the un-memoized behaviour it always had. `[G1f]` below proves
this directly.

Chosen over plumbing a `scale=` argument into summarize() (P13's other
candidate, and the brief's other option): that would change `summarize
(client, fresh_input)`'s call shape, which 34+ stubs across this suite
replace with a fixed two-argument function (`grep -rn "main.summarize ="
compactor tests`) — breaking that shape here breaks every one of them for a
fix that belongs one level down, at the one function both callers already
share.

    python test_v3194_guard_g1.py
"""

import asyncio
import sys
import tempfile
import os

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="g1guard-")
os.environ["MAX_MODEL_LEN"] = "32768"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "1024"

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
    return sum(len(main._message_text(m)) + 4 for m in ms)


main.count_tokens = local

_SAVED_POST = main.httpx.post
_SAVED_SUMMARIZE_ONCE = main._summarize_once
_SAVED_COUNT_TOKENS_EXACT = main.count_tokens_exact

SCHED: list[bool] = []
POST_CALLS = {"n": 0}


class _FakeResp:
    def __init__(self, n):
        self.status_code = 200
        self._n = n
        self.text = ""

    def json(self):
        return {"count": self._n}


def _fake_post(url, json=None, timeout=None):
    """Answers or fails according to SCHED, one entry per REAL call — so a
    memo hit (no call made) leaves the next scheduled entry for whoever
    calls next, exactly like a live process where a cached list simply never
    reaches the network a second time."""
    POST_CALLS["n"] += 1
    ans = SCHED.pop(0) if SCHED else True
    if not ans:
        raise ConnectionError("simulated /tokenize outage")
    return _FakeResp(local(json["messages"]))


main.httpx.post = _fake_post


async def _fake_summarize_once(client, turns):
    return "w" * (main.SUMMARY_MAX_TOKENS - 4)  # a full-length summary


main._summarize_once = _fake_summarize_once

BUDGET = min(
    main.MAX_MODEL_LEN,
    max(256, main.MAX_MODEL_LEN - main.SUMMARY_MAX_TOKENS - main.SUMMARY_INPUT_RESERVE),
)


def _activate_memo():
    """Turn the count_tokens_exact memo ON for what follows in THIS
    (synchronous) context, the way compact_if_needed does at its own top
    — a fresh dict, since nothing else does this for a direct unit-test
    call. Returns the dict, for tests that need to inspect it."""
    d: dict = {}
    main._COUNT_TOKENS_EXACT_MEMO_CTX.set(d)
    return d


def _preview(span):
    """compact_if_needed's own preview arithmetic (main.py's
    `_fresh_span_preview` block), through the REAL count_tokens_exact so the
    memo is live."""
    fl = local(span)
    fe = main.count_tokens_exact(span)
    sc = (fe / fl) if (fe is not None and fl > 0) else main._PESSIMISTIC_SUMMARY_SCALE
    b = len(main._chunk_to_budget(span, BUDGET, sc))
    return b, min(b, main.MAX_SUMMARY_CALLS_PER_REQUEST) * main.SUMMARY_MAX_TOKENS


def _span_of(n, each):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i} " + "x" * (each - 8)}
        for i in range(n)
    ]


SPANS = [(8, 7000), (10, 3300), (20, 1650), (20, 2600), (30, 1650), (12, 4800)]
SCHEDULES = {
    "steady": [True, True],
    "flapA": [True, False],   # THE dangerous direction: preview answers, summarize() fails
    "flapB": [False, True],   # the reverse flap: already safe (over-reserves)
    "down": [False, False],
}

print("\n[G1] P14-2: the preview and summarize() never disagree about the "
      "SAME list's batch count, even when one /tokenize call fails and the "
      "other does not")

_rows_over = 0
_flapA_single_post = True
for n, each in SPANS:
    for sched_name, sched in SCHEDULES.items():
        span = _span_of(n, each)
        # One request = one memo lifetime: activate it fresh for this row,
        # exactly as compact_if_needed does at the top of one real call —
        # asyncio.run() below creates its own Task, which copies the
        # AMBIENT context (contextvars.copy_context() at Task creation),
        # so this activation, made in the calling thread just before, is
        # what the preview call AND summarize()'s task-based call both see.
        _activate_memo()
        SCHED[:] = list(sched)
        POST_CALLS["n"] = 0
        pb, reserve = _preview(span)
        summary, deferred = asyncio.run(main.summarize(None, span))
        real = local([{"role": "system", "content": summary}]) - 4 if summary else 0
        excess = max(0, real - reserve)
        _rows_over += excess > 0
        check(
            excess == 0,
            f"[G1] {n}x{each} {sched_name}: summarize() ({real} tok) fits "
            f"the preview's reserve ({reserve} tok), no turns deferred "
            f"unexpectedly (deferred={len(deferred)}, posts={POST_CALLS['n']})",
        )
        if sched_name == "flapA":
            # THE mechanism, not just the outcome: a memo hit means the
            # second (scheduled-to-fail) POST is never attempted at all.
            # Asserting the call count catches a mutant that happens to
            # produce the right numbers some other way.
            _flapA_single_post = _flapA_single_post and POST_CALLS["n"] == 1
            check(
                POST_CALLS["n"] == 1,
                f"[G1] {n}x{each} flapA: only ONE real /tokenize POST was "
                f"made (posts={POST_CALLS['n']}) — summarize()'s call was "
                f"served from the memo, never reaching the scheduled "
                f"failure",
            )

check(_rows_over == 0, f"[G1] CONTROL: 0 of {len(SPANS) * len(SCHEDULES)} rows over their reserve (was 4/24 before the fix)")


# ---------------------------------------------------------------------------
# [G1b] CONTROL: a failure is never memoized. Two calls on the SAME content,
# first scheduled to fail, second to answer — the first must not poison the
# second, and the memo must hold the SECOND (successful) answer, not "no
# answer" and not silence.
# ---------------------------------------------------------------------------
_g1b_memo = _activate_memo()
_g1b_span = _span_of(4, 500)
SCHED[:] = [False]
POST_CALLS["n"] = 0
_g1b_first = main.count_tokens_exact(_g1b_span)
check(_g1b_first is None, f"[G1b] a failed call returns None ({_g1b_first!r})")
_g1b_key = main._count_tokens_exact_memo_key(_g1b_span, None)
check(
    _g1b_key not in _g1b_memo,
    "[G1b] a FAILURE is never memoized — the key is absent after a failed call",
)
SCHED[:] = [True]
_g1b_second = main.count_tokens_exact(_g1b_span)
check(_g1b_second == local(_g1b_span), f"[G1b] the retry on the same content answers for real ({_g1b_second})")
check(
    _g1b_memo.get(_g1b_key) == _g1b_second,
    "[G1b] the SUCCESSFUL answer is now memoized",
)
POST_CALLS["n"] = 0
_g1b_third = main.count_tokens_exact(_g1b_span)
check(_g1b_third == _g1b_second, "[G1b] a third call on the same content returns the memoized value")
check(POST_CALLS["n"] == 0, f"[G1b] the third call made NO real POST (posts={POST_CALLS['n']}) — served from the memo")


# ---------------------------------------------------------------------------
# [G1c] CONTROL: different content never collides. Two distinct spans both
# get their own, independent measurement.
# ---------------------------------------------------------------------------
_g1c_memo = _activate_memo()
SCHED[:] = [True, True]
_g1c_a = _span_of(3, 300)
_g1c_b = _span_of(5, 700)
_g1c_ra = main.count_tokens_exact(_g1c_a)
_g1c_rb = main.count_tokens_exact(_g1c_b)
check(_g1c_ra == local(_g1c_a) and _g1c_rb == local(_g1c_b), "[G1c] two distinct spans each measured for real")
check(len(_g1c_memo) == 2, f"[G1c] the memo holds two independent entries (size={len(_g1c_memo)})")


# ---------------------------------------------------------------------------
# [G1d] CONTROL: the key covers add_generation_prompt, not just content — an
# assistant-final list measured with the flag explicitly True and explicitly
# False must not collide (count_tokens_exact's own docstring: the two
# request bodies differ in BOTH fields whenever this flag differs).
# ---------------------------------------------------------------------------
_activate_memo()


def _agp_sensitive_post(url, json=None, timeout=None):
    POST_CALLS["n"] += 1
    base = local(json["messages"])
    return _FakeResp(base + (500 if json.get("add_generation_prompt") else 0))


main.httpx.post = _agp_sensitive_post
_g1d_span = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
POST_CALLS["n"] = 0
_g1d_true = main.count_tokens_exact(_g1d_span, add_generation_prompt=True)
_g1d_false = main.count_tokens_exact(_g1d_span, add_generation_prompt=False)
check(_g1d_true != _g1d_false, f"[G1d] add_generation_prompt=True ({_g1d_true}) and False ({_g1d_false}) do not collide")
check(POST_CALLS["n"] == 2, f"[G1d] both were measured for real, neither served the other's cached value (posts={POST_CALLS['n']})")
main.httpx.post = _fake_post


# ---------------------------------------------------------------------------
# [G1e] CONTROL: with the memo NEVER activated (the state every caller other
# than compact_if_needed is always in — _enforce_hard_budget's own calls, a
# direct unit test), count_tokens_exact behaves exactly as it did before
# this fix: every call is a real POST, nothing is cached across calls, even
# on IDENTICAL content. This is what keeps test_tokenize_flags.py's `[6]`
# and test_tokenize_repair.py's empty-shape suite (neither owned by this
# lane) working unmodified — see this file's own module docstring.
# ---------------------------------------------------------------------------
main._COUNT_TOKENS_EXACT_MEMO_CTX.set(None)  # explicitly OFF, not just unset
SCHED[:] = [True, True]
POST_CALLS["n"] = 0
_g1e_span = _span_of(2, 300)
_g1e_first = main.count_tokens_exact(_g1e_span)
_g1e_second = main.count_tokens_exact(_g1e_span)
check(
    POST_CALLS["n"] == 2,
    f"*** [G1e] CONTROL: with the memo not activated, the SAME content is "
    f"measured for real TWICE (posts={POST_CALLS['n']}) — the pre-fix "
    f"behaviour every caller outside compact_if_needed still gets, so a "
    f"direct test asserting 'a POST was made' is never silently skipped",
)
check(_g1e_first == _g1e_second, "[G1e]: both calls still agree on the answer, just not via a cache")


# ---------------------------------------------------------------------------
# [G1f] *** THE CROSS-LANE FIX: the exact shape test_tokenize_flags.py's
# `[6]` and test_tokenize_repair.py's empty-shape suite hit — a direct
# caller re-measuring content an EARLIER, unrelated direct call already
# measured, with the memo never activated (its default OFF state, since
# these callers never go through compact_if_needed at all). Reproduced
# against a first draft of this fix (a process-wide, always-on memo): the
# second call returned the cached value with NO POST, so a stub capturing
# "the last body sent" saw nothing and `b["add_generation_prompt"]` raised
# TypeError on None. Fixed by scoping the memo to a context compact_if_
# needed turns on; a plain module-level call (this file's own default
# state, same as those two suites') never turns it on at all.
# ---------------------------------------------------------------------------
_saved_default = main._COUNT_TOKENS_EXACT_MEMO_CTX.get()
main._COUNT_TOKENS_EXACT_MEMO_CTX.set(None)
SCHED[:] = [True]
POST_CALLS["n"] = 0
_g1f_span = [{"role": "user", "content": "hello"}]
main.count_tokens_exact(_g1f_span)  # an EARLIER, unrelated section's own call
SCHED[:] = [True]
_g1f_seen = {}


def _g1f_post(url, json=None, timeout=None):
    _g1f_seen["json"] = json
    return _FakeResp(local(json["messages"]))


main.httpx.post = _g1f_post
main.count_tokens_exact(_g1f_span)  # the SAME content, a later, unrelated call
check(
    "json" in _g1f_seen and _g1f_seen["json"] is not None,
    "*** [G1f] THE CROSS-LANE FIX: a later, unrelated direct call on "
    "content an earlier direct call already measured still reaches "
    "/tokenize for real (outside compact_if_needed, the memo is never on)",
)
main.httpx.post = _fake_post


main.httpx.post = _SAVED_POST
main._summarize_once = _SAVED_SUMMARIZE_ONCE
main.count_tokens_exact = _SAVED_COUNT_TOKENS_EXACT
main._COUNT_TOKENS_EXACT_MEMO_CTX.set(None)

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nall checks passed")
