"""Adversary: make the compactor's DEPENDENCIES misbehave.

The backend, the disk, the clock, the process. Not the client — three other
files own that.

WHAT THIS FILE CAN AND CANNOT REACH
-----------------------------------
The adversary container has an HTTP client and the fixture's fault-injection
control plane. It has NO docker socket (deliberately — see its Dockerfile). So
every attack that requires stopping, pausing, killing or restarting a container,
and every attack that requires a small filesystem, was run BY HAND from the host
and written up in `findings/`. Each such case appears here as a `skip` that
names the finding file and the exact host commands, so a reader of the suite
alone still learns that the attack was made and what it found:

    faults-04-unbounded-read.md          docker pause -> the request never returns
    faults-05-unreachable-finish-stop.md docker stop  -> finish_reason "stop"
    faults-06-healthcheck-docstring.md   docker stop  -> degraded, HTTP 200
    faults-08-disk-pressure-SOUND.md     a 300 MB tmpfs compactor
    faults-10-drain-abandons.md          docker stop with tails in flight

ORDERING MATTERS IN THIS FILE, and it is not cosmetic.

`_BUDGET_MARGIN` is a module GLOBAL in the compactor with no reset short of ~250
successful requests or a restart. The test that proves a lying /tokenize can
latch it therefore POISONS the process for everything after it, so it is last,
and it says so. The flood test likewise leaves `background_work.shed_recently`
true for 300 s, so nothing after it may assert `status == "ok"`.
"""

import json
import threading
import time

import pytest

try:  # conftest is importable under pytest's default prepend import mode
    from conftest import record
except Exception:  # pragma: no cover - belt and braces; findings must land
    import pathlib

    def record(name: str, detail: str) -> None:
        try:
            p = pathlib.Path("/findings")
            p.mkdir(parents=True, exist_ok=True)
            with (p / f"{name}.md").open("a", encoding="utf-8") as fh:
                fh.write(detail.rstrip() + "\n\n")
        except OSError:
            pass


MODEL = "fixture-model"

# HARD_INPUT_LIMIT on this stack: MAX_MODEL_LEN 32768 - GENERATION_RESERVE 8192.
HARD_INPUT_LIMIT = 24576
MAX_MODEL_LEN = 32768
TARGET_TOKENS = 18432
# MAX_MODEL_LEN // 4 — the ceiling _note_backend_rejection may latch to.
MARGIN_CEILING = MAX_MODEL_LEN // 4

_WORD = "quick brown fox jumps over lazy dogs "

# The content class this deployment's model actually writes, and the one the
# char/4 estimator is worst on: `count_tokens_exact`'s docstring measures one
# 17,930-character reply holding 1,710 U+2501 and 441 U+2500. Three UTF-8 bytes
# and about one token per character, so char/4 reads roughly a quarter of the
# truth. Used deliberately wherever a payload must be large in TOKENS while
# staying small in CHARACTERS.
_RULE = "━" * 60 + "\n"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _true_count(fixture_client, messages) -> int:
    """What the backend will CHARGE for `messages`. The oracle, asked directly,
    so no assertion below depends on the compactor's own arithmetic."""
    r = fixture_client.post(
        "/tokenize",
        json={"model": MODEL, "messages": messages, "add_generation_prompt": True},
    )
    r.raise_for_status()
    return int(r.json()["count"])


def _user_turn_of(fixture_client, target_tokens: int) -> list[dict]:
    """One user message whose chat-templated count is EXACTLY target_tokens.

    Binary search on characters against the fixture's own tokenizer. Exact,
    because the whole point of a boundary case is the boundary.
    """
    lo, hi = 1, 400_000
    while lo < hi:
        mid = (lo + hi) // 2
        msgs = [{"role": "user", "content": (_WORD * (mid // len(_WORD) + 1))[:mid]}]
        if _true_count(fixture_client, msgs) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return [{"role": "user", "content": (_WORD * (lo // len(_WORD) + 1))[:lo]}]


def _build_multi_turn(chars_per_turn: int, pairs: int, unit: str) -> list[dict]:
    body = (unit * (chars_per_turn // len(unit) + 1))[:chars_per_turn]
    msgs: list[dict] = []
    for i in range(pairs):
        msgs.append({"role": "user", "content": f"turn {i}. {body}"})
        msgs.append({"role": "assistant", "content": f"reply {i}. {body}"})
    msgs.append({"role": "user", "content": "and so what should I do next?"})
    return msgs


def _multi_turn_of(
    fixture_client, target_tokens: int, pairs: int = 4, unit: str = _WORD
) -> list[dict]:
    """A user/assistant conversation of AT LEAST target_tokens, ending on a
    short user turn — a shape the hard-budget guard is ALLOWED to shed from. A
    single-turn payload would prove nothing about shedding, because the guard
    may never drop the newest turn.

    Binary-searched against the fixture's own tokenizer rather than estimated:
    a chars-per-token guess is exactly the thing this project keeps getting
    wrong, and a test whose payload lands somewhere near the boundary is not
    testing the boundary.
    """
    lo, hi = 16, 400_000
    while lo < hi:
        mid = (lo + hi) // 2
        if _true_count(fixture_client, _build_multi_turn(mid, pairs, unit)) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return _build_multi_turn(lo, pairs, unit)


def _build_user_only_multi_turn(chars_per_turn: int, pairs: int, unit: str) -> list[dict]:
    """Same shape as `_build_multi_turn`, but `unit`'s dense content sits ONLY
    in USER turns; assistant turns are short, ordinary filler (advfix, P8-3
    hardening — see `_user_only_multi_turn_of`'s docstring for why).
    """
    body = (unit * (chars_per_turn // len(unit) + 1))[:chars_per_turn]
    msgs: list[dict] = []
    for i in range(pairs):
        msgs.append({"role": "user", "content": f"turn {i}. {body}"})
        msgs.append({"role": "assistant", "content": f"reply {i}. Understood, noted."})
    msgs.append({"role": "user", "content": "and so what should I do next?"})
    return msgs


def _user_only_multi_turn_of(
    fixture_client, target_tokens: int, pairs: int = 4, unit: str = _WORD
) -> list[dict]:
    """`_multi_turn_of`, but the dense `unit` content is confined to USER
    turns (advfix hardening, P8 gate).

    test_compaction_triggers_on_the_discredited_counter (F-13) used
    `_multi_turn_of(..., unit=_RULE)`, which puts the box-drawing payload in
    BOTH roles via `_build_multi_turn`. v3.1.9.2 added
    `_redact_forwarded_loop_replies`, which rewrites every non-newest
    ASSISTANT turn `reply_is_degenerate` flags before forwarding — and
    `_DECOR_CHARS` (main.py:2302-2304) classifies the ENTIRE box-drawing
    block U+2500-U+259F as decoration, so a box-drawing-dense assistant turn
    trips the decor-fraction rule regardless of repetition and gets shrunk.
    That shrink cuts `usage.prompt_tokens` for a reason that has nothing to
    do with F-13 (whether `compact_if_needed`'s TRIGGER — `count_tokens(...)
    <= TARGET_TOKENS` at main.py:1913-1915 — still runs on the char/4
    estimate instead of vLLM's ground truth), and was being misread as "the
    compaction trigger no longer runs on char/4" (advfix P8 gate: a test that
    ran the right input and, once a second subsystem entered the picture,
    ended up asserting something else). Reproduced with a plain chat request
    (no fault injection) against this exact fixture: `checks.tokenizer` never
    loads in this stack (MODEL_REPO=fixture-model is not a real cached HF
    repo under HF_HUB_OFFLINE=1), so `count_tokens` is ALWAYS the char/4
    estimator here — nothing about the trigger changed; only the forwarded
    text did, and only because it was an assistant turn.

    Keeping the dense content in USER turns (never touched by
    `_redact_forwarded_loop_replies` — see main.py's "only non-newest
    ASSISTANT turns" comments) removes that confound and isolates F-13's
    actual question.
    """
    lo, hi = 16, 400_000
    while lo < hi:
        mid = (lo + hi) // 2
        if _true_count(fixture_client, _build_user_only_multi_turn(mid, pairs, unit)) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return _build_user_only_multi_turn(lo, pairs, unit)


def _chat(client, messages, conv: str, **extra):
    body = {"model": MODEL, "messages": messages, "stream": False}
    body.update(extra)
    return client.post(
        "/v1/chat/completions", headers={"X-Conversation-Id": conv}, json=body
    )


def _forwarded_prompt_tokens(resp) -> int | None:
    """What the backend says it was actually charged for THIS request.

    The fixture echoes real usage, so this is a black-box measurement of what
    the compactor forwarded — no log scraping, no internals.
    """
    try:
        return int(resp.json()["usage"]["prompt_tokens"])
    except Exception:
        return None


def _current_time_line(client) -> str | None:
    """The current-time line the compactor is adding right now (v3.1.9), or
    None when the feature is off or the field is absent."""
    ti = (client.get("/health/full").json().get("config") or {}).get("time_injection") or {}
    return ti.get("current_line") if ti.get("enabled") else None


def _health(client) -> dict:
    r = client.get("/health/full")
    assert r.status_code in (200, 503), f"/health/full answered {r.status_code}"
    return r.json()


def _set_mode(fixture_client, **kw) -> None:
    fixture_client.post("/_fixture/mode", json=kw)


# Reasons that ALWAYS appear in THIS stack's /health/full, independent of any
# fault this file injects, and must not be mistaken for evidence about the
# thing under test (advfix, P8 gate). Confirmed with a plain, unfaulted chat
# request against this exact compose file: `checks.tokenizer` never loads,
# because MODEL_REPO=fixture-model names no real, cacheable HF repo and the
# stack runs HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 with nothing baked in to
# satisfy it (testfixtures/unit-suite/Dockerfile caches only the fastembed
# retrieval model). The very first request that calls count_tokens() latches
# "the local tokenizer failed to load ... the request budget sheds turns by
# them." into status_reasons for good (health.py:1842-1856) — a real, v3.1.9
# signal about a DIFFERENT counter (main.get_tokenizer's local estimator) than
# either test below is manipulating (vLLM's own /tokenize, or
# compact_if_needed's trigger), and it happens to contain both "budget" and
# "estimate", which a naive substring filter reads as the specific reason
# these tests are checking for absence of. Filtering on it by its own fixed
# text (rather than widening the keyword list, which would just as easily
# swallow a real regression) keeps both tests sensitive to an actual new
# reason while not being fooled by this one.
_KNOWN_TOKENIZER_NOISE = "the local tokenizer failed to load"


def _non_noise_reasons(reasons: list[str]) -> list[str]:
    """`reasons` minus the always-on local-tokenizer-load noise described
    above. Exists so both call sites filter identically — see the constant's
    comment for why a second, differently-worded copy would be how they drift.
    """
    return [x for x in reasons if _KNOWN_TOKENIZER_NOISE not in x]


def test_non_noise_reasons_ignores_only_the_known_tokenizer_noise():
    """CONTROL for `_non_noise_reasons` (advfix, P8 gate). Pure Python, no
    stack — this is testing the FILTER two other tests rely on, not the
    compactor, so it must not need the compactor to run.

    Guards both directions:
      - a status_reasons list holding ONLY the known, unrelated tokenizer-load
        noise must filter to empty (the false positive this fixes: without
        the filter, `test_an_inflated_tokenize_count_destroys_turns_while_
        health_says_ok` and `test_compaction_triggers_on_the_discredited_
        counter` both reported "GOOD NEWS" from this reason alone, proven
        live against an UNFAULTED request — see `_KNOWN_TOKENIZER_NOISE`);
      - a status_reasons list holding a GENUINE hard-budget-shed reason next
        to that same noise must still come through, so a real fix to F-01/
        F-13 is not silently swallowed by this filter. This is the CONTROL a
        guard that refuses everything would fail.

    FAILS IF: `_non_noise_reasons` starts dropping (or stops dropping) the
    known noise string, or starts dropping anything else.
    """
    noise = (
        "the local tokenizer failed to load (We couldn't connect to "
        "'https://huggingface.co' ...; next retry in 30s). Token counts "
        "that do not come from /tokenize are running on the char/4 "
        "estimate, and the request budget sheds turns by them."
    )
    real = "hard budget shed 12 turn(s) because /tokenize disagreed with itself"

    assert _non_noise_reasons([noise]) == [], (
        "the known tokenizer-load noise alone must filter to nothing"
    )
    assert _non_noise_reasons([]) == [], "no reasons in, none out"
    assert _non_noise_reasons([noise, real]) == [real], (
        "a genuine reason beside the noise must survive the filter — this is "
        "the control that proves the filter cannot pass every request"
    )
    assert _non_noise_reasons([real]) == [real], (
        "a genuine reason with no noise present must survive unchanged"
    )


# ---------------------------------------------------------------------------
# 1. /tokenize answers 200 with a number that cannot be true
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factor, label",
    [(0.0, "zero"), (-1.0, "negative")],
    ids=["count_is_zero", "count_is_negative"],
)
def test_an_impossible_tokenize_count_is_believed_and_reported_healthy(
    client, fixture_client, factor, label
):
    """F-01. A count of 0 (or below 0) for a non-empty message list is
    IMPOSSIBLE, not merely implausible. count_tokens_exact validates only
    `isinstance(n, (int, float))`, so it is taken as ground truth, propagated
    into the shedding `scale`, and `/health/full` calls the endpoint healthy.

    `checks.tokenize.ok` is the field this project added because two outages
    ran in a degraded counting mode with every health surface green. It covers
    "the endpoint did not answer" and NOT "the endpoint answered, confidently,
    with a lie" — which is the 2026-08-28 shape.

    FAILS IF: count_tokens_exact starts rejecting a non-positive count for
    non-empty input and routes it through _note_tokenize_failure, so that
    tokenize.ok goes false (or any new field says the answer was refused).
    """
    # Big enough to defeat the char/4 prescreen (limit // 8), so /tokenize is
    # genuinely consulted.
    msgs = _user_turn_of(fixture_client, 6000)
    true_n = _true_count(fixture_client, msgs)  # BEFORE the lie is installed
    _set_mode(fixture_client, tokenize_mode="wrong", factor=factor)

    r = _chat(client, msgs, f"advf-impossible-{label}")
    assert r.status_code < 500, f"compactor 5xx'd on a lying oracle: {r.status_code}"

    tok = _health(client)["checks"]["tokenize"]
    detail = (
        f"- /tokenize forced to answer `int(true * {factor})` for a "
        f"{true_n}-token payload; compactor "
        f"answered HTTP {r.status_code} and /health/full reported "
        f"checks.tokenize = {json.dumps(tok)}"
    )
    record("faults-01-tokenize-lies", detail)

    assert tok["ok"] is True and tok["consecutive_failures"] == 0, (
        "GOOD NEWS, and this test is now stale: the compactor noticed that "
        f"/tokenize answered with an impossible count. tokenize={tok}"
    )


def test_an_inflated_tokenize_count_destroys_turns_while_health_says_ok(
    client, fixture_client
):
    """F-01, aggravated. `factor=1000` makes every payload look enormous. The
    guard sheds the user's oldest turns to fit a fictional budget, logs
    "hard budget FAILED to fit" at ERROR, forwards anyway, and answers 200.

    Measured black-box: the backend is charged for FEWER tokens than the client
    sent, i.e. turns were dropped. `/health/full` reports no reason for it —
    there is no budget/shedding field in the payload at all.

    ADVFIX (P8 gate): filters `_non_noise_reasons` out of the check below.
    Every request in this stack latches "the local tokenizer failed to load
    ... the request budget sheds turns by them." (see
    `_KNOWN_TOKENIZER_NOISE`'s comment) — a real v3.1.9 reason, but about
    main.get_tokenizer's local estimator, which this test never touches; it
    only contains "budget" by coincidence. Without the filter this test
    reported "GOOD NEWS" for a reason that has nothing to do with the lying
    /tokenize oracle it drives — confirmed by re-running with NO fault
    injected at all: the same reason appears from the first request. The
    _enforce_hard_budget guard's shedding (driven by vLLM's own, lied-to
    /tokenize — see _measure's "ground truth" comment, main.py ~5523) is
    still invisible in status_reasons once that noise is excluded.

    FAILS IF: shedding (or a failed shed) reaches /health/full's
    `status_reasons` as anything other than the known tokenizer-load noise,
    or the compactor stops believing a 1000x count.
    """
    msgs = _multi_turn_of(fixture_client, 12000)
    sent = _true_count(fixture_client, msgs)

    honest = _chat(client, msgs, "advf-inflate-control")
    assert honest.status_code == 200
    honest_tokens = _forwarded_prompt_tokens(honest)

    _set_mode(fixture_client, tokenize_mode="wrong", factor=1000.0)
    lied = _chat(client, msgs, "advf-inflate-victim")
    assert lied.status_code < 500, f"compactor 5xx'd: {lied.status_code}"
    lied_tokens = _forwarded_prompt_tokens(lied)

    h = _health(client)
    record(
        "faults-01-tokenize-lies",
        f"- inflated count (factor 1000): client sent {sent} true tokens; "
        f"with an honest /tokenize the backend was charged {honest_tokens}; "
        f"with the lie it was charged {lied_tokens}. HTTP {lied.status_code}. "
        f"/health/full status={h['status']} reasons={json.dumps(h['status_reasons'])}",
    )

    assert lied_tokens is not None and honest_tokens is not None
    assert lied_tokens < honest_tokens, (
        "expected the inflated count to make the guard shed turns; it did not "
        f"({lied_tokens} vs {honest_tokens})"
    )
    budget_reason = [
        x for x in _non_noise_reasons(h["status_reasons"])
        if "budget" in x or "shed" in x.lower()
    ]
    assert not budget_reason, (
        "GOOD NEWS, and this test is now stale: /health/full grew a reason for "
        f"hard-budget shedding: {budget_reason}"
    )


# ---------------------------------------------------------------------------
# 2. an exchange the backend refuses is counted nowhere
# ---------------------------------------------------------------------------


def test_a_backend_rejected_turn_is_counted_nowhere(client, fixture_client):
    """F-03. A payload whose FINAL user turn alone exceeds the window cannot be
    compacted (no older turns) and cannot be shed (the guard never drops the
    newest turn), so it is forwarded and the backend refuses it.

    The compactor's own log calls this "this turn produced no reply, no facts
    and no episodic write, and nothing retries it". `/health/full` has no
    counter that moves: `memory_tail` is reached only by exchanges that got as
    far as a reply, so the ledger `stored + skipped == exchanges` that
    test_saturation.py asserts is exact only over the exchanges that survived.

    FAILS IF: any counter in /health/full increments when the backend rejects a
    request.
    """
    msgs = _user_turn_of(fixture_client, MAX_MODEL_LEN + 4000)
    before = _health(client)["memory_tail"]

    r = _chat(client, msgs, "advf-rejected")
    assert r.status_code == 400, (
        f"expected the backend's context-length 400 to be relayed; got "
        f"{r.status_code}: {r.text[:200]}"
    )
    assert r.status_code < 500, "the compactor must relay, not crash"

    time.sleep(3)
    after = _health(client)["memory_tail"]

    record(
        "faults-03-rejected-turn-uncounted",
        f"- a {_true_count(fixture_client, msgs)}-token single-turn payload was "
        f"relayed as HTTP {r.status_code}. memory_tail before="
        f"{json.dumps({k: before[k] for k in ('stored', 'skipped')})} after="
        f"{json.dumps({k: after[k] for k in ('stored', 'skipped')})}; "
        f"outcomes delta="
        f"{ {k: after['outcomes'][k] - before['outcomes'][k] for k in after['outcomes']} }",
    )

    moved = {
        k: after["outcomes"][k] - before["outcomes"][k]
        for k in after["outcomes"]
        if after["outcomes"][k] != before["outcomes"][k]
    }
    assert not moved and after["stored"] == before["stored"], (
        "GOOD NEWS, and this test is now stale: a backend rejection now moves a "
        f"counter: {moved}"
    )


# ---------------------------------------------------------------------------
# 3. what the boundary actually does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [TARGET_TOKENS, HARD_INPUT_LIMIT - 1, HARD_INPUT_LIMIT, HARD_INPUT_LIMIT + 1],
    ids=["at_target", "one_under_hard", "exactly_hard", "one_over_hard"],
)
def test_budget_boundaries_are_exact(client, fixture_client, target):
    """SOUND. A single-turn payload sized to the token, measured by the
    fixture's own tokenizer, is forwarded and charged for EXACTLY that many
    tokens — no off-by-one shed at the limit, no crash on either side of it.

    (Over the limit it is forwarded whole because the guard may never drop the
    newest turn. That is documented behaviour and is logged at ERROR.)

    FAILS IF: the guard starts trimming a single-turn payload, or the boundary
    arithmetic moves by one.
    """
    msgs = _user_turn_of(fixture_client, target)
    assert _true_count(fixture_client, msgs) == target

    # v3.1.9: the compactor dates the newest user turn with one line
    # (main._inject_time_line) whenever the payload fits beside the line's
    # reserve, so a payload of `target` tokens is charged `target` plus that
    # line's exact cost. The line is read from /health/full on both sides of
    # the request, because it carries the minute; a payload the guard measured
    # into the reserve band or over the limit is forwarded undated, at exactly
    # `target`. Either way the count is EXACT - the boundary property holds.
    lines = {_current_time_line(client)}
    r = _chat(client, msgs, f"advf-edge-{target}")
    lines.add(_current_time_line(client))
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    allowed = {target}
    for line in lines - {None}:
        dated = [{**msgs[-1], "content": line + "\n\n" + msgs[-1]["content"]}]
        allowed.add(_true_count(fixture_client, dated))
    assert _forwarded_prompt_tokens(r) in allowed, (
        f"the compactor forwarded {_forwarded_prompt_tokens(r)} tokens for a "
        f"payload of exactly {target} (allowed: {sorted(allowed)}, undated or "
        f"dated with {sorted(lines - {None})})"
    )


def test_compaction_triggers_on_the_discredited_counter(client, fixture_client):
    """F-13. Two payloads of IDENTICAL true size, both above
    COMPACTOR_TARGET_TOKENS, get opposite treatment because `compact_if_needed`
    still triggers on `count_tokens` — the char/4 estimator P0-0c discredited —
    while `_enforce_hard_budget` was given vLLM's own count.

    The discriminator is characters per token. Plain English prices near 4
    chars/token so char/4 agrees; box-drawing rules price near 1 char/token, so
    char/4 reads about a quarter of the truth. The second is not a contrived
    input: `count_tokens_exact`'s own docstring measures one production reply
    holding 1,710 U+2501 and 441 U+2500.

    Measured black-box through `usage.prompt_tokens` — what the backend was
    actually charged — so nothing here depends on reading a log.

    In THIS stack the estimator reads HIGH on English, so the visible failure is
    compaction NOT firing on the box-drawing payload. On production content the
    estimator reads LOW, which is the same defect pointing the failing way.

    ADVFIX (P8 gate), read before touching this test again: the box-drawing
    payload used to go through `_multi_turn_of` (BOTH roles). v3.1.9.2 added
    `_redact_forwarded_loop_replies`, which rewrites every non-newest
    ASSISTANT turn that `reply_is_degenerate` flags before forwarding, and
    `_DECOR_CHARS` (main.py:2302-2304) makes the ENTIRE box-drawing block
    U+2500-U+259F "decoration" regardless of repetition — so a box-drawing
    assistant turn is shrunk by an UNRELATED subsystem, which cut
    `usage.prompt_tokens` for the rules payload and read as "the compaction
    trigger no longer runs on char/4". It does: `compact_if_needed` still
    computes `current = count_tokens(messages)` (char/4 whenever the local
    tokenizer is unavailable, which it always is in this stack — see
    `_KNOWN_TOKENIZER_NOISE`) and compares it to TARGET_TOKENS unchanged
    (main.py:1913-1915); nothing there consults count_tokens_exact.
    `test_a_lying_tokenize_oracle_can_latch_the_margin` (same file) still
    relies on this exact payload shape staying UNDER the compaction trigger
    ("at ~1 char per token the rules payload is far under the compaction
    trigger") and still passes, which is the same fact from the other side.
    `_user_only_multi_turn_of` keeps the dense content in USER turns, which
    that redaction path never touches, removing the confound.

    FAILS IF: compact_if_needed starts consulting count_tokens_exact.
    """
    target = (TARGET_TOKENS + HARD_INPUT_LIMIT) // 2  # above the compaction
    # target, below the hard budget, so ONLY the compaction decision is on trial

    english = _user_only_multi_turn_of(fixture_client, target, unit=_WORD)
    rules = _user_only_multi_turn_of(fixture_client, target, unit=_RULE)
    n_english = _true_count(fixture_client, english)
    n_rules = _true_count(fixture_client, rules)
    assert abs(n_english - n_rules) <= 8, (
        f"the two payloads must be the same true size to compare: "
        f"{n_english} vs {n_rules}"
    )
    assert n_english > TARGET_TOKENS, "both payloads must be over the compaction target"

    r_eng = _chat(client, english, "advf-trigger-english")
    r_rul = _chat(client, rules, "advf-trigger-rules")
    assert r_eng.status_code == 200 and r_rul.status_code == 200
    fwd_eng = _forwarded_prompt_tokens(r_eng)
    fwd_rul = _forwarded_prompt_tokens(r_rul)

    chars_eng = sum(len(m["content"]) for m in english)
    chars_rul = sum(len(m["content"]) for m in rules)
    h = _health(client)
    record(
        "faults-13-compaction-trigger",
        f"- both payloads {n_rules} true tokens (target {TARGET_TOKENS}). "
        f"english: {chars_eng} chars (char/4={chars_eng // 4}) -> forwarded "
        f"{fwd_eng}. rules: {chars_rul} chars (char/4={chars_rul // 4}) -> "
        f"forwarded {fwd_rul}. /health/full status={h['status']} "
        f"reasons={json.dumps(h['status_reasons'])}",
    )

    assert fwd_eng is not None and fwd_rul is not None
    assert fwd_eng < n_english * 0.8, (
        f"expected the English payload to be compacted ({n_english} true -> "
        f"{fwd_eng} forwarded)"
    )
    assert fwd_rul >= n_rules - 8, (
        "GOOD NEWS, and this test is now stale: the box-drawing payload was "
        f"compacted too ({n_rules} true -> {fwd_rul} forwarded), so the "
        "compaction trigger no longer runs on the char/4 estimator."
    )
    non_noise = [x for x in _non_noise_reasons(h["status_reasons"]) if "budget" in x or "compact" in x]
    assert not non_noise, (
        f"GOOD NEWS: /health/full now says something about compaction not "
        f"firing: {non_noise}"
    )


def test_a_max_tokens_larger_than_the_window_is_silently_rewritten(
    client, fixture_client
):
    """F-11. `max_tokens` above MAX_MODEL_LEN // 2 is clamped and the clamped
    value is written back into the forwarded body. The clamp is correct; it is
    also silent — no log line, no response field, nothing telling the client it
    will not get the completion length it asked for.

    Proven black-box: with max_tokens=40000 on a 32768 window the fixture's own
    gate (`prompt + max_tokens > MAX_MODEL_LEN`) would refuse the request, and
    it does not — so the value that reached it was not 40000.

    FAILS IF: the rewrite starts announcing itself, or stops happening (the
    request would then be refused).
    """
    msgs = _user_turn_of(fixture_client, 2000)
    r = _chat(client, msgs, "advf-mt-huge", max_tokens=40000)
    assert r.status_code == 200, (
        "expected the clamp to keep this request acceptable to the backend; got "
        f"{r.status_code}: {r.text[:300]}"
    )
    record(
        "faults-11-max-tokens-clamp",
        "- max_tokens=40000 on a 32768 window: HTTP 200, so the forwarded value "
        "was rewritten (2000 + 40000 would have tripped the backend's own gate). "
        "Nothing in the response says so.",
    )


# ---------------------------------------------------------------------------
# 4. a 4xx stream IS error-typed. Its siblings are not — see findings.
# ---------------------------------------------------------------------------


def test_a_4xx_stream_is_error_typed(client, fixture_client):
    """SOUND, and the control case for F-05.

    `_request_rejected_stream_chunks` marks a refused request with
    `finish_reason: "error"` and a top-level `error` object, so a client can
    tell a failed turn from a reply. This asserts that half works.

    The other half does NOT — `_vllm_unreachable_stream_chunks`, used when the
    backend 5xx's or disappears mid-stream, still ends `finish_reason: "stop"`
    with no error object, which is exactly what the function above documents as
    the incident. That is unreachable from this container (it needs the backend
    stopped); see findings/faults-05-unreachable-finish-stop.md.

    FAILS IF: the 4xx stream stops carrying finish_reason "error".
    """
    msgs = _user_turn_of(fixture_client, MAX_MODEL_LEN + 4000)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Conversation-Id": "advf-4xx-stream"},
        json={"model": MODEL, "messages": msgs, "stream": True},
    ) as r:
        assert r.status_code == 200, "a refused stream still commits HTTP 200"
        text = "".join(chunk for chunk in r.iter_text())

    events = [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    assert events, f"no SSE events at all: {text[:300]!r}"
    finishes = [
        c.get("finish_reason")
        for e in events
        for c in (e.get("choices") or [])
    ]
    assert "error" in finishes, (
        "a stream the backend REFUSED ended without finish_reason 'error'; "
        f"finish reasons were {finishes}. That is the INCIDENT 4.2 shape."
    )
    assert any("error" in e for e in events), "no top-level error object on the pair"


# ---------------------------------------------------------------------------
# 5. what happens to memory when the pool sheds
# ---------------------------------------------------------------------------


def test_shed_tails_are_counted_as_stored(client, fixture_client):
    """F-07, ADVFIX REWRITE (P8 gate) — the original F-07 defect is FIXED;
    this now pins the correct behaviour instead. Drive more concurrent turns
    than the pool's outstanding ceiling (default 64). Tails past the ceiling
    are CLOSED UNRUN — no facts, no episodic index, no rollup.

    Proven fixed live (`findings/faults-07-shed-counted-as-stored.md`, 160
    concurrent turns, shed=80): `memory_tail.stored` moved by exactly the 80
    that actually ran (not by all 160), `memory_tail.skipped` moved by the 80
    that were shed, `tail["outcomes"]["skipped_shed"]` now exists and moved by
    80, and BOTH `background_work` and `memory_tail`'s own status_reasons name
    the loss — "memory tail skipping: ... that skip's outcome skipped_shed"
    alongside "background work shedding: 80 task(s) dropped". Before this
    fix, `tailhealth` had no `skipped_shed` outcome at all and `stored`
    silently absorbed the shed ones; see the old assertions this replaces in
    git history for exactly what that looked like.

    NOTE: this test leaves `shed_recently` true for ~300 s, so nothing after it
    may assert status == "ok".

    FAILS IF: a shed tail is counted as `stored` again, `skipped_shed` stops
    existing or stops moving with `shed`, or status_reasons stops naming it.
    """
    n = 160
    # Settle first: a tail still outstanding from an earlier case would show up
    # in this one's deltas and make the ledger arithmetic below a guess.
    for _ in range(60):
        if _health(client)["background_work"]["outstanding"] == 0:
            break
        time.sleep(2)
    h0 = _health(client)
    before_bg, before_tail = h0["background_work"], h0["memory_tail"]

    results: list = []
    lock = threading.Lock()

    def one(i: int) -> None:
        try:
            r = _chat(
                client,
                [{"role": "user", "content": f"flood {i}: my lucky number is {i}"}],
                f"advf-flood-{i}",
            )
            with lock:
                results.append(r.status_code)
        except Exception as e:  # a proxy under load must not drop connections
            with lock:
                results.append(type(e).__name__)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(x == 200 for x in results), (
        f"the compactor did not stay a well-behaved proxy under {n} concurrent "
        f"requests: {sorted({str(x) for x in results})}"
    )

    # let the accepted tails finish so `outstanding` settles
    for _ in range(60):
        if _health(client)["background_work"]["outstanding"] == 0:
            break
        time.sleep(2)

    h = _health(client)
    bg, tail = h["background_work"], h["memory_tail"]
    shed = bg["shed"] - before_bg["shed"]
    stored = tail["stored"] - before_tail["stored"]
    skipped = tail["skipped"] - before_tail["skipped"]

    record(
        "faults-07-shed-counted-as-stored",
        f"- {n} concurrent turns, all HTTP 200. bg delta: shed={shed}, "
        f"submitted={bg['submitted'] - before_bg['submitted']}, "
        f"completed={bg['completed'] - before_bg['completed']}. "
        f"memory_tail delta: stored={stored}, skipped={skipped}. "
        f"status={h['status']} reasons={json.dumps(h['status_reasons'])}",
    )

    if shed == 0:
        pytest.skip(
            f"the pool never reached its ceiling on this run (shed=0 of {n} "
            f"submissions); the box was fast enough to drain them. Re-run with "
            f"more concurrency to reproduce faults-07."
        )

    before_skipped_shed = before_tail.get("outcomes", {}).get("skipped_shed", 0)
    after_skipped_shed = tail.get("outcomes", {}).get("skipped_shed", 0)
    skipped_shed_delta = after_skipped_shed - before_skipped_shed

    # Ledger conservation, same invariant test_saturation.py pins elsewhere:
    # every submitted exchange is either stored or skipped, never both, never
    # neither.
    assert stored + skipped == n, (
        f"sanity: {n} submitted must split exactly into stored+skipped; got "
        f"stored={stored} skipped={skipped}"
    )
    # The FIX (was the finding): shed tails no longer inflate `stored` — they
    # move `skipped`, specifically via the `skipped_shed` outcome, one per
    # shed tail.
    assert stored == n - shed, (
        f"GOOD NEWS did not fully land: expected memory_tail.stored to count "
        f"only the {n - shed} exchanges that actually ran (n={n}, shed="
        f"{shed}); it counted {stored}"
    )
    assert skipped == shed, (
        f"expected memory_tail.skipped to count exactly the {shed} shed "
        f"exchanges; it counted {skipped}"
    )
    assert skipped_shed_delta == shed, (
        f"expected outcomes.skipped_shed to move by exactly {shed} (the shed "
        f"count); it moved by {skipped_shed_delta}. outcomes={tail['outcomes']}"
    )
    # The positive claim (CONTROL): a tail that actually ran is still counted
    # as stored, not swept into skipped_shed too — this guard is not simply
    # refusing everything.
    assert stored > 0, "sanity: some tails should have run and stored under n=160"
    assert any("shedding" in reason for reason in h["status_reasons"]), (
        "background_work's own loss is not even reported in status_reasons"
    )
    assert any("skipped_shed" in reason for reason in h["status_reasons"]), (
        "memory_tail's status_reasons no longer names skipped_shed by name — "
        "the fix that closed F-07 (the loss visible in ONE dict and not the "
        "other) has regressed"
    )


# ---------------------------------------------------------------------------
# 6. the health endpoint's silences
# ---------------------------------------------------------------------------


def test_backups_are_reported_but_never_judged(client):
    """F-09. `/health/full` carries a `backups` block with `count`, `latest` and
    `latest_mtime`, and `gather_health_full` never consults any of them when it
    builds `status_reasons`. A pod that has never produced a backup, or whose
    newest is days old, is not degraded by it.

    The other reporting path is `backup._alert_failure` -> `alert.notify`, which
    returns immediately when COMPACTOR_ALERT_WEBHOOK is unset — the default, and
    unset in every compose file in this tree. Both mechanisms are off.

    FAILS IF: a backup-staleness (or backup-absence) reason appears in
    status_reasons.
    """
    h = _health(client)
    backups = h["backups"]
    assert "count" in backups and "latest_mtime" in backups

    reasons = " ".join(h["status_reasons"]).lower()
    record(
        "faults-09-backups-never-judged",
        f"- backups={json.dumps(backups)} status={h['status']} "
        f"reasons={json.dumps(h['status_reasons'])}",
    )
    assert "backup" not in reasons, (
        "GOOD NEWS, and this test is now stale: /health/full now judges backups: "
        f"{h['status_reasons']}"
    )


def test_a_degraded_status_never_changes_the_health_http_code(client, fixture_client):
    """F-06, the half reachable from here.

    `status_to_http_code` returns 503 only for "down" (storage unwritable).
    Every other degradation — including vLLM being completely unreachable —
    answers 200, so `curl -f http://localhost:8080/health/full`, which is the
    Dockerfile's HEALTHCHECK, stays green.

    The behaviour is deliberate and well argued. What is wrong is
    `health_full`'s own docstring, which says "the container goes unhealthy when
    vLLM is FATAL". It does not; measured with the backend stopped, see
    findings/faults-06-healthcheck-docstring.md.

    Here we prove the mapping with a degradation this container CAN cause: a
    /tokenize outage.

    FAILS IF: the status/HTTP mapping changes, or that docstring is corrected
    and this case is re-read.
    """
    # Built BEFORE the outage is installed: this helper asks /tokenize itself.
    msgs = _user_turn_of(fixture_client, 6000)
    _set_mode(fixture_client, tokenize_mode="http_error", status=503)
    _chat(client, msgs, "advf-degrade-code")

    r = client.get("/health/full")
    body = r.json()
    assert body["status"] == "degraded", f"expected degraded, got {body['status']}"
    assert any("/tokenize" in x for x in body["status_reasons"])
    assert r.status_code == 200, (
        "a degraded compactor answered non-200 on /health/full — the Docker "
        "HEALTHCHECK semantics just changed"
    )


@pytest.mark.parametrize(
    "mode, kw",
    [
        ("http_error", {"status": 400}),
        ("http_error", {"status": 429}),
        ("http_error", {"status": 500}),
        ("http_error", {"status": 503}),
        ("garbage", {}),
        ("hang", {"delay": 12.0}),
    ],
    ids=["400", "429", "500", "503", "garbage_body", "hang_past_read_timeout"],
)
def test_a_tokenize_outage_degrades_visibly_and_recovers(
    client, fixture_client, mode, kw
):
    """SOUND. Every way of NOT answering /tokenize — 4xx, 5xx, a 200 with no
    `count`, and a hang longer than the 10 s read timeout — degrades
    `/health/full` with an accurate reason, keeps chat serving, and clears
    itself on the next success.

    This is the half of the claim that holds. The half that does not is
    test_an_impossible_tokenize_count_... above: answering WRONGLY is invisible.

    FAILS IF: a /tokenize outage stops reaching status_reasons, or chat starts
    5xx'ing when the counter is unavailable, or recovery stops clearing.
    """
    # Built BEFORE the outage is installed: this helper asks /tokenize itself.
    msgs = _user_turn_of(fixture_client, 6000)
    _set_mode(fixture_client, tokenize_mode=mode, **kw)

    r = _chat(client, msgs, f"advf-outage-{mode}-{kw}")
    assert r.status_code < 500, f"chat 5xx'd during a /tokenize outage: {r.status_code}"

    tok = _health(client)["checks"]["tokenize"]
    assert tok["ok"] is False and tok["consecutive_failures"] >= 1, (
        f"a {mode} {kw} /tokenize was not reported as degraded: {tok}"
    )
    assert tok["degraded_since"] is not None

    _set_mode(fixture_client, tokenize_mode="ok")
    _chat(client, msgs, f"advf-outage-{mode}-recovery")
    tok = _health(client)["checks"]["tokenize"]
    assert tok["ok"] is True and tok["consecutive_failures"] == 0, (
        f"/tokenize recovered but the health endpoint did not: {tok}"
    )


# ---------------------------------------------------------------------------
# 7. host-only attacks, recorded here so the suite says they were made
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finding, how",
    [
        (
            "faults-04-unbounded-read.md",
            "docker pause adv-faults-vllm-fixture-1; POST a chat completion. "
            "A paused container still ACCEPTS (the listen queue is kernel side) "
            "so connect=10.0 is satisfied and read=None never expires: measured "
            "100.1 s with no answer, bounded only by the client's patience.",
        ),
        (
            "faults-05-unreachable-finish-stop.md",
            "docker stop adv-faults-vllm-fixture-1; POST a STREAMING completion. "
            "HTTP 200, an assistant delta reading 'The model backend is starting "
            "up or restarting', finish_reason 'stop', no error object — the exact "
            "shape _request_rejected_stream_chunks' docstring calls the incident.",
        ),
        (
            "faults-06-healthcheck-docstring.md",
            "docker stop adv-faults-vllm-fixture-1; GET /health/full -> 200 "
            "'degraded'. health_full's docstring claims the container goes "
            "unhealthy when vLLM is FATAL. It does not.",
        ),
        (
            "faults-08-disk-pressure-SOUND.md",
            "a second compactor on a 300 MB tmpfs; fill to 120 MB free -> "
            "skipped_disk_pressure counted and two accurate status reasons; fill "
            "to 0 -> status 'down'; free it -> writes resume; no corruption.",
        ),
        (
            "faults-10-drain-abandons.md",
            "reply_chars=160000 to make tails slow, 6 concurrent turns, then "
            "docker stop: 'background tasks didn't finish in 10.0s; abandoning', "
            "5 tails lost, memory_tail.stored had already said 6, and the restart "
            "erased the counters that would have shown it.",
        ),
    ],
)
def test_host_only_attack_is_recorded(finding, how):
    """These need a docker socket the adversary container deliberately does not
    have. They were run by hand from the host; the write-ups are on disk."""
    pytest.skip(f"host-only, see findings/{finding} — {how}")


# ---------------------------------------------------------------------------
# 8. LAST, because it poisons the process
# ---------------------------------------------------------------------------


def test_one_lying_tokenize_latches_the_process_wide_budget_margin(
    client, fixture_client
):
    """F-02. THIS TEST POISONS THE COMPACTOR PROCESS. It is last on purpose.

    v3.1 D4 added `guard_measured_overflow` so a rejection the guard ALREADY
    predicted cannot widen `_BUDGET_MARGIN` — because the margin is a module
    global and one unfittable conversation must not narrow the window for every
    other one. The protection covers an honest backend and not a dishonest one:
    make /tokenize under-report and the guard measures `fits`, so
    `guard_measured_overflow` is False and the margin learns the whole
    overshoot in one step, straight to its MAX_MODEL_LEN // 4 ceiling.

    Nothing reports it. `/health/full` has no margin field; recovery is 50
    accepted requests per halving (~250 requests) or a restart.

    Measured black-box through the fixture's echoed usage.prompt_tokens: a
    victim conversation between the poisoned limit and the real one is forwarded
    whole before, and shed after.

    FAILS IF: /health/full grows a budget-margin field, or the calibration stops
    learning from a rejection whose measurement came from a non-positive count.
    """
    # Sized to sit between (HARD_INPUT_LIMIT - MARGIN_CEILING) and TARGET_TOKENS,
    # so only the poisoned guard can shed it.
    #
    # Built from box-drawing rules, and that is load-bearing: compaction
    # triggers on char/4 (see test_compaction_triggers_on_the_discredited_counter),
    # and an English payload of this many TOKENS is long enough in CHARACTERS to
    # trip that trigger and be summarized before the guard ever sees it — which
    # is exactly what happened on the first run of this case. At ~1 char per
    # token the rules payload is far under the compaction trigger and far over
    # the poisoned budget, so the guard is the only thing that can move it.
    victim_target = (HARD_INPUT_LIMIT - MARGIN_CEILING + TARGET_TOKENS) // 2
    victim = _multi_turn_of(fixture_client, victim_target, unit=_RULE)
    victim_true = _true_count(fixture_client, victim)
    assert HARD_INPUT_LIMIT - MARGIN_CEILING < victim_true < TARGET_TOKENS, (
        f"victim payload landed at {victim_true}, outside the window this test "
        f"needs ({HARD_INPUT_LIMIT - MARGIN_CEILING} < n < {TARGET_TOKENS})"
    )

    before = _chat(client, victim, "advf-margin-victim-before")
    assert before.status_code == 200, before.text[:200]
    before_tokens = _forwarded_prompt_tokens(before)
    if before_tokens is None or before_tokens < victim_true - 64:
        pytest.skip(
            "the budget margin was already non-zero when this test started "
            f"(sent {victim_true}, forwarded {before_tokens}); it cannot be "
            "reset from outside the process. Recreate the compactor and re-run."
        )

    # --- poison: one request the backend refuses, measured by a lying oracle --
    _set_mode(fixture_client, tokenize_mode="wrong", factor=0.0)
    huge = _user_turn_of(fixture_client, MAX_MODEL_LEN + 7000)  # one turn: no
    # older turns to summarize, and the guard may never drop the newest turn,
    # so this reaches the backend and is refused there.
    poison = _chat(client, huge, "advf-margin-poison")
    assert poison.status_code == 400, (
        "the poisoning request was supposed to be refused by the backend; got "
        f"{poison.status_code}"
    )
    _set_mode(fixture_client, tokenize_mode="ok")

    after = _chat(client, victim, "advf-margin-victim-after")
    assert after.status_code == 200, after.text[:200]
    after_tokens = _forwarded_prompt_tokens(after)

    h = _health(client)
    record(
        "faults-02-margin-latch",
        f"- victim conversation of {victim_true} true tokens: forwarded "
        f"{before_tokens} tokens BEFORE the poisoning and {after_tokens} AFTER "
        f"one rejected request whose /tokenize answered 0. "
        f"/health/full status={h['status']} "
        f"reasons={json.dumps(h['status_reasons'])}; "
        f"top-level keys={sorted(h.keys())}; config={json.dumps(h['config'])}",
    )

    assert after_tokens is not None
    assert after_tokens < before_tokens, (
        "GOOD NEWS, and this test is now stale: the margin did not latch from a "
        f"rejection measured by a lying /tokenize ({before_tokens} -> "
        f"{after_tokens})"
    )
    assert after_tokens <= HARD_INPUT_LIMIT - MARGIN_CEILING + 64, (
        f"expected the margin to latch to its {MARGIN_CEILING} ceiling; the "
        f"victim was shed to {after_tokens}"
    )
    assert "margin" not in json.dumps(h).lower(), (
        "GOOD NEWS, and this test is now stale: /health/full now mentions the "
        "budget margin"
    )
