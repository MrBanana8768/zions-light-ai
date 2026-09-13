"""
Tier-3 regression net for the MEMORY TAIL and HEALTH-HONESTY defects fixed in
v3.1.7 / v3.1.8.

Every defect below is FIXED. This file exists so it stays fixed. Each case
names the R-number it pins (the section headings in V314_BACKLOG.md) and, in
its docstring, what would make it fail.

  R8    `tailhealth` counted `stored` on decide_memory_tail's verdict and THEN
        fired the tail, which had exits that stored nothing. /health/full
        reported a healthy tail for an exchange that never reached memory.
        Pinned by: an exchange with no pairable user text must be counted as a
        SKIP with the R8 label, and the conversation's store must be empty —
        with a normal exchange as the control that proves both halves move.

  R26   vLLM dying mid-stream skipped the tail with no counter and no log line
        ("0 decisions, 0 counter movement, an empty grep"). Pinned by: an
        interrupted stream still produces a COUNTED, NAMED tail decision.

  R27   An empty reply (Stop before the first token) armed the degrade flag
        for five minutes claiming memory was lost, when nothing was. Pinned
        by: a HARMLESS skip advances the general skip clock but NOT the lossy
        one — with a lossy skip as the control that proves the lossy clock
        does move when there is a real loss.

  R28   `tailhealth.note` raised out of a `finally` on a non-numeric count and
        left the ledger inconsistent. Pinned via the API by the invariant the
        fix is written to preserve: sum(outcomes) == stored + skipped, checked
        after every one of a run of odd exchanges.

  A2 / N4b (v3.1.8)
        OpenWebUI title/tag/follow-up traffic was fact-extracted, indexed and
        deduped. Now skipped under `skipped_task_traffic` — but the rule is a
        THRESHOLD (`_recorded_position >= 4`), not "no assistant turn in the
        array", because a genuine FIRST TURN and a REGENERATE of the first
        reply send exactly the same shape and must still be memorized. All
        three cases are pinned end to end.

BLACK BOX, like the rest of this directory: no compactor import, public API
only, plus the localhost-gated /admin endpoints the harness already wraps.

TWO THINGS ABOUT WHAT CAN BE OBSERVED FROM OUT HERE, because they shape every
assertion in the file.

1. /health/full's `memory_tail` block is PROCESS-GLOBAL. It counts every
   exchange the compactor has handled, including other suites' and other
   people's running against the same deployment. So counter assertions are
   `>=` deltas taken across a narrow window, never `==`, and wherever a claim
   can be made against the CONVERSATION instead (admin_conv_summary: facts
   count, episodic.indexed_exchanges, summary.turns_seen) it is made there —
   that state is per-conv and immune to concurrent traffic. The two places a
   global counter is asserted to be UNCHANGED are marked; concurrent traffic
   could in principle false-FAIL them, which is the safe direction.

2. The local integration fixture has NO MODEL WEIGHTS. Every completion is the
   same ~65-character canned string, so nothing here asserts on reply content,
   and a trimmed partial reply can never clear MIN_MEMORABLE_TRIMMED_CHARS
   (300). `episodic.indexed_exchanges` and `summary.turns_seen` both advance
   without weights, and they are what "it reached the store" means here.
"""

from __future__ import annotations

import time

import httpx
import pytest

import _harness as H

# ---------------------------------------------------------------------------
# Local helpers. Deliberately in this file rather than in _harness.py: the
# harness is shared with every other suite in this directory and nothing here
# has a second caller.
# ---------------------------------------------------------------------------

# The two tailhealth labels that v3.1.8 calls HARMLESS_SKIP_OUTCOMES — a skip
# that cost the user nothing. Spelled out here rather than imported, because
# importing it would mean importing compactor code, which is the one thing
# this directory does not do. If these names drift, the cases below fail on
# the missing key rather than passing quietly.
HARMLESS_SKIPS = ("skipped_empty", "skipped_task_traffic")


def _retry_transport(fn, what: str, *, attempts: int = 4, gap: float = 3.0):
    """Run `fn`, retrying only through TRANSPORT-level faults.

    Not a way of making a flaky assertion pass — nothing here retries a
    failed check. This retries a connect/read failure against the deployment
    itself, which on a stack shared with other suites is a real and frequent
    event (a saturated single-worker fixture, a read timeout on /health/full)
    and says nothing about the property under test. A test that fell over on
    one of those would be reporting "the memory tail regressed" for "the
    machine was busy", which is the exact confusion this project keeps paying
    for. Exhausting the attempts FAILS, loudly, naming the last fault.
    """
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except httpx.TransportError as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(gap)
    pytest.fail(
        f"{what} kept failing at the transport level after {attempts} "
        f"attempts — the deployment is not answering, so nothing in this "
        f"test can be concluded. Last fault: {last}"
    )


def _retry_503(fn, what: str, *, attempts: int = 4, gap: float = 3.0) -> int:
    """As above, for a call that returns an HTTP status.

    A 503 from /v1/chat/completions is the compactor's "vLLM unreachable"
    branch, which returns BEFORE the tail (`No async tail — there's no
    assistant turn`, main.py). So a retried request leaves no decision in the
    ledger and no row in the store: retrying is clean, and the alternative is
    a test that reports a memory-tail regression because the fixture was
    momentarily busy.
    """
    last = None
    for _ in range(attempts):
        try:
            status = fn()
        except httpx.TransportError as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if status != 503:
                return status
            last = "HTTP 503 (the compactor could not reach vLLM)"
        time.sleep(gap)
    pytest.fail(f"{what} never got past a transient backend failure: {last}")


def _tail() -> dict:
    """The `memory_tail` block from /health/full.

    Fails loudly rather than returning {} — an absent block would otherwise
    turn every delta below into 0 - 0 == 0 and every case in this file into a
    test that passes without running its check. That failure mode has bitten
    this project three times (V314_BACKLOG R6), so it is checked once, here,
    where every case goes through it.
    """
    status, body = _retry_transport(H.health_full, "GET /health/full")
    assert isinstance(body, dict) and body, (
        f"/health/full returned no JSON body (HTTP {status})"
    )
    mt = body.get("memory_tail")
    assert isinstance(mt, dict) and mt, (
        f"/health/full has no usable memory_tail block: {body.get('memory_tail')!r}"
    )
    assert "error" not in mt, (
        f"memory_tail is unobservable, so nothing in this file can be "
        f"asserted: {mt['error']!r}"
    )
    for key in ("stored", "skipped", "outcomes", "seconds_since_last_skip",
                "seconds_since_last_lossy_skip", "skipped_recently",
                "skip_window_s"):
        assert key in mt, f"memory_tail is missing {key!r}: {mt!r}"
    return mt


def _outcome(mt: dict, name: str) -> int:
    """One outcome counter, insisting the LABEL exists.

    `.get(name, 0)` would make a deleted label read as a flat zero and turn
    "this outcome never fired" into "this outcome is not a thing any more" —
    the same silent pass. Three of the labels asserted in this file
    (skipped_disk_pressure, skipped_no_user_text, skipped_task_traffic) were
    introduced BY the fixes being pinned, so their presence is part of what is
    being pinned.
    """
    outcomes = mt.get("outcomes") or {}
    assert name in outcomes, (
        f"tailhealth no longer publishes the {name!r} outcome; "
        f"published labels: {sorted(outcomes)}"
    )
    return int(outcomes[name])


def _decisions(mt: dict) -> int:
    """Total tail decisions the ledger has recorded."""
    return int(mt["stored"]) + int(mt["skipped"])


def _assert_ledger_reconciles(mt: dict, when: str) -> None:
    """sum(outcomes) == stored + skipped, the R28 invariant.

    Safe to assert under concurrent traffic: tailhealth.snapshot() and
    tailhealth.note() both run on the event loop with no await between their
    field reads/writes, so a snapshot cannot catch a half-applied note().
    """
    outcomes = mt.get("outcomes") or {}
    total = sum(int(v) for v in outcomes.values())
    assert total == _decisions(mt), (
        f"tail ledger is incoherent {when}: sum(outcomes)={total} but "
        f"stored({mt['stored']}) + skipped({mt['skipped']})={_decisions(mt)}.\n"
        f"outcomes: {outcomes!r}"
    )


def _conv(conv_id: str) -> dict:
    """(facts_count, episodic_count, turns_seen) for one conversation.

    Each is asserted to be an INTEGER. admin_conversation_summary reports
    `None` — deliberately, and says so in its own comments — when a layer is
    unreadable, precisely so "we could not look" stays distinguishable from
    "there is nothing there". A test that treated None as 0 would report a
    clean store during exactly the incident it is meant to catch.
    """
    s = _retry_transport(
        lambda: H.admin_conv_summary(conv_id),
        f"GET /admin/conversations/{conv_id}",
    )
    facts = (s.get("facts") or {}).get("count")
    episodic = (s.get("episodic") or {}).get("indexed_exchanges")
    summary = s.get("summary") or {}
    turns = summary.get("turns_seen")
    assert isinstance(facts, int), f"facts.count unreadable for {conv_id}: {s!r}"
    assert isinstance(episodic, int), f"episodic unreadable for {conv_id}: {s!r}"
    assert isinstance(turns, int), f"turns_seen unreadable for {conv_id}: {s!r}"
    return {"facts": facts, "episodic": episodic, "turns_seen": turns}


def _wait_for_turns_seen(conv_id: str, want: int, *, max_wait: float = 30.0) -> int:
    """Poll until summary.turns_seen reaches `want`. Returns what it reached.

    turns_seen is the sharpest "the tail actually ran" signal available to a
    black-box caller: summarizer.maybe_rollup is the last of _async_tail's
    three jobs and it persists the observed position on every fired tail, so
    the counter moves whether or not there was enough material to roll up.
    It moves for NO other reason — a skipped tail leaves it exactly where it
    was.
    """
    deadline = time.monotonic() + max_wait
    seen = -1
    while True:
        try:
            seen = _conv(conv_id)["turns_seen"]
        except Exception:
            seen = -1
        if seen >= want or time.monotonic() >= deadline:
            return seen
        time.sleep(1.0)


def _settle() -> None:
    """Give a fired tail time to land before asserting that NOTHING landed.

    A negative assertion cannot be reached by polling — there is no state to
    poll for — so it gets a flat wait, and a generous one. The tail is three
    jobs deep (episodic embed, extraction, rollup) and this suite's whole
    point is being right rather than quick.

    max(H.TAIL_WAIT, 60), NOT a bare H.wait_for_async_tail(). TAIL_WAIT is
    ZIONS_TEST_TAIL_WAIT, which the `model` profile sets to 240 but the
    DEFAULT `integration-tests` service in docker-compose.integration.yml
    never sets — so on every default-profile run this was a flat 8 SECONDS
    for a three-job tail, on four call sites that guard a NEGATIVE assertion
    (nothing landed). It was green only because the weightless fixture
    answers in microseconds and the tail is usually done well inside 8s; it
    was a wrong-reason pass waiting for the first slow run. 60s keeps the
    fast profile fast enough while no longer being shorter than the tail
    itself is documented to be; the `model` profile's 240 already clears it.
    """
    H.wait_for_async_tail(seconds=max(H.TAIL_WAIT, 60.0))


# Slack for "the newest skip happened after this test fired its trigger":
# seconds_since_last_skip is rounded to 0.1 s by the endpoint, and the two
# clocks being compared are the same host's.
_SKIP_CLOCK_SLACK_S = 2.0


def _newest_skip_is_ours(since_any, triggered_at: float) -> bool:
    """True when the most recent skip (of the clock read) happened AFTER
    `triggered_at` (a time.monotonic() taken just before this test's trigger).

    It used to be `since_any <= 60`. That was only ever a proxy for "after
    the trigger", and it stopped being one the moment _settle became a
    60-second wait: every run then read 60.1-60.2 s and failed by
    construction (merged v3.1.9 integration run). Measuring the bound from
    the trigger keeps the check exactly as strict however long _settle is.
    """
    return (isinstance(since_any, (int, float))
            and since_any <= (time.monotonic() - triggered_at) + _SKIP_CLOCK_SLACK_S)


def _stream_and_abort(conv_id: str, prompt: str) -> int:
    """Open a streaming completion and hang up WITHOUT reading a single byte
    of the body. Returns the HTTP status of the response headers.

    Why this USUALLY interrupts the compactor mid-stream rather than racing
    it: StreamingResponse commits its headers BEFORE the generator produces
    anything, and the generator's first chunk cannot exist until the
    compactor's own request to vLLM has come back. So the client is normally
    holding the headers, and closing, before there is any reply text to
    accumulate.

    THIS IS NOT A GUARANTEE, and a docstring here once called it "structural,
    not lucky" on the strength of 3/3 on an idle machine. That claim covers
    only the header-commit half; the other half is the CLIENT's own
    socket-close scheduling, which this function does not control, and one
    measured run under load lost the race (`resp.close()` landed after the
    first chunk arrived). Callers that depend on the abort beating the first
    chunk must not assume it from this function's name — see
    test_r26_interrupted_stream_still_produces_a_counted_named_decision for
    the shape of a caller that accounts for both outcomes.

    _harness.chat() is non-streaming, so this is raw httpx against
    H.BASE_URL — the harness README's own instruction for streaming cases.
    """
    body = {
        "model": H.resolve_model(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": True,
        "metadata": {"chat_id": conv_id},
    }

    def _once() -> int:
        with httpx.Client(base_url=H.BASE_URL, timeout=H.TIMEOUT) as c:
            with c.stream(
                "POST", "/v1/chat/completions", json=body,
                headers={"X-Conversation-Id": conv_id},
            ) as resp:
                status = resp.status_code
                resp.close()
        return status

    return _retry_transport(_once, f"streaming completion on {conv_id}")


def _stream_fully(conv_id: str, prompt: str) -> tuple[int, int]:
    """The control for the case above: consume the whole stream, including
    the terminating [DONE]. Returns (status, bytes read)."""
    body = {
        "model": H.resolve_model(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": True,
        "metadata": {"chat_id": conv_id},
    }
    def _once() -> tuple[int, int]:
        n = 0
        with httpx.Client(base_url=H.BASE_URL, timeout=H.TIMEOUT) as c:
            with c.stream(
                "POST", "/v1/chat/completions", json=body,
                headers={"X-Conversation-Id": conv_id},
            ) as resp:
                status = resp.status_code
                for raw in resp.iter_raw():
                    n += len(raw)
        return status, n

    return _retry_transport(_once, f"streaming completion on {conv_id}")


def _post_raw(conv_id: str, messages: list[dict], *, max_tokens: int = 64) -> int:
    """Non-streaming completion with a hand-built messages array — for the
    message shapes _harness.chat() cannot express (content as a parts list).
    Returns the HTTP status."""
    body = {
        "model": H.resolve_model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "metadata": {"chat_id": conv_id},
    }
    def _once() -> int:
        with httpx.Client(base_url=H.BASE_URL, timeout=H.TIMEOUT) as c:
            r = c.post("/v1/chat/completions", json=body,
                       headers={"X-Conversation-Id": conv_id})
        return r.status_code

    return _retry_503(_once, f"non-streaming completion on {conv_id}")


def _chat(user_msg: str, *, conv_id: str, **kw):
    """H.chat, insisting on a 200 and retried through a transient 503.

    Every case in this file wants a completed exchange as its SETUP; a
    backend blip during setup is not the thing being measured.
    """
    holder: dict = {}

    def _once() -> int:
        holder["r"] = H.chat(user_msg, conv_id=conv_id, **kw)
        return holder["r"].status_code

    status = _retry_503(_once, f"chat completion on {conv_id}")
    assert status == 200, f"chat on {conv_id} returned HTTP {status}"
    return holder["r"]


def _drive_to_position(conv_id: str, exchanges: int) -> int:
    """Drive `exchanges` real exchanges, resending the whole thread each turn
    the way OpenWebUI does, and return the recorded position.

    Two message-units per exchange, so two exchanges is position 4 — which is
    TASK_TRAFFIC_MIN_POSITION exactly. The caller asserts the position it
    needs rather than assuming this arithmetic: if _recorded_position ever
    counts something else, the N4b cases must fail loudly on the precondition
    and not quietly test the wrong side of the threshold.
    """
    history: list[dict] = []
    for i in range(exchanges):
        msg = f"Thread turn {i}: tell me about the tide tables."
        r = _chat(msg, conv_id=conv_id, prior_turns=history, max_tokens=48)
        assert r.status_code == 200, f"turn {i} failed: HTTP {r.status_code}"
        history = H.extend_history(history, msg, r.response_text)
        _wait_for_turns_seen(conv_id, 2 * (i + 1))
    _settle()
    return _conv(conv_id)["turns_seen"]


# ---------------------------------------------------------------------------
# R8 — what /health/full says about the tail must match what is in the store
# ---------------------------------------------------------------------------


def test_health_full_publishes_the_vocabulary_these_fixes_introduced():
    """Schema canary for R8, R27 and N4b, and the first thing to read when
    anything below fails.

    Each of these names arrived WITH one of the fixes: `skipped_disk_pressure`
    and `skipped_no_user_text` are R8's two hoisted conditions,
    `skipped_task_traffic` is N4b's, and `seconds_since_last_lossy_skip` is
    R27's second clock. On the code that shipped before them, this test fails
    on a missing key — which is what makes the delta assertions in the rest of
    the file statements about behaviour rather than about `.get(k, 0)`.

    WHAT WOULD MAKE THIS FAIL: reverting any of the three fixes, or renaming a
    published label without updating this file.
    """
    mt = _tail()
    for label in ("stored", "stored_trimmed", "skipped_disk_pressure",
                  "skipped_no_user_text", *HARMLESS_SKIPS):
        _outcome(mt, label)
    # R27's separate clock. _tail() already insists the key is present; this
    # says why it has to be, and that it is a number or an honest None rather
    # than an alias of the general clock's value.
    lossy = mt["seconds_since_last_lossy_skip"]
    assert lossy is None or isinstance(lossy, (int, float)), repr(lossy)
    assert isinstance(mt["skipped_recently"], bool), repr(mt["skipped_recently"])
    assert float(mt["skip_window_s"]) > 0, (
        f"the degrade window is {mt['skip_window_s']!r}; a non-positive window "
        f"makes `skipped_recently` permanently false and silently disables "
        f"the whole signal (tailhealth._window_s)"
    )
    _assert_ledger_reconciles(mt, "on a bare read of /health/full")


def test_r8_control_a_normal_exchange_is_counted_stored_and_reaches_the_store():
    """CONTROL for the R8 case below, and the thing that makes it non-vacuous.

    An ordinary exchange must (a) move the global `stored` counter and (b)
    leave a row in the conversation's store. Without this, "the store is
    empty" in the next test would prove nothing — it would be equally
    consistent with a deployment where the store never fills at all.

    WHAT WOULD MAKE THIS FAIL: episodic indexing broken or disabled; the tail
    not firing on the non-streaming path; the `stored` counter not being
    published. Verified non-vacuous by construction — the next test drives
    the same endpoint and asserts the OPPOSITE of every line here.
    """
    H.skip_if_no_admin("R8 pins health output against the store")
    conv_id = H.fresh_conv_id()
    try:
        before = _tail()
        _assert_ledger_reconciles(before, "before a normal exchange")
        assert _conv(conv_id) == {"facts": 0, "episodic": 0, "turns_seen": 0}, (
            "a fresh conv_id already has state — the sentinel id collided"
        )

        r = _chat(
            "Control exchange: my ferry leaves from the north quay at dawn.",
            conv_id=conv_id, max_tokens=64,
        )
        assert r.status_code == 200, r.status_code

        # Poll rather than sleep: this is a POSITIVE assertion, so it can be
        # waited for, and the fast path stays fast.
        H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=30)
        _wait_for_turns_seen(conv_id, 2)
        after = _tail()
        _assert_ledger_reconciles(after, "after a normal exchange")

        state = _conv(conv_id)
        assert state["episodic"] >= 1, (
            f"a normal exchange left nothing in the episodic store: {state!r}"
        )
        assert state["turns_seen"] >= 2, (
            f"a normal exchange did not advance the recorded position: {state!r}"
        )
        # `>=`, not `== +1`: the counter is process-global and other traffic
        # against this deployment lands in it too. See the module docstring.
        assert _outcome(after, "stored") > _outcome(before, "stored"), (
            "the `stored` outcome did not move for an exchange that demonstrably "
            "reached the store"
        )
    finally:
        H.admin_safe_forget(conv_id)


@pytest.mark.parametrize(
    "shape,last_user_content",
    [
        # A user turn of nothing but whitespace. _has_pairable_user_text's
        # docstring calls this out by name: two siblings used to disagree about
        # exactly this shape, and the one that let it through was the one that
        # writes to the store.
        ("whitespace-only", "   \n  "),
        # A multimodal parts LIST carrying no text field and no part
        # _message_image_count recognises. _tail_store_blocked names this as
        # the reachable case: it falls through _extract_last_user_text and then
        # through _memorable_user_text, which only substitutes a marker when it
        # can count images. Before R8 the tail took a bare `return` here with
        # no log line of any kind, and the counter said `stored`.
        ("parts-list-with-no-text", [{"type": "input_audio", "data": "zzz"}]),
    ],
)
def test_r8_exchange_that_cannot_be_stored_is_counted_as_a_skip_not_a_store(
    shape, last_user_content
):
    """R8. An exchange the tail will store NOTHING for must be counted as a
    skip, under the label that says why, and the store must agree.

    This is R8 exactly: `decide_memory_tail` judges the REPLY and says
    "store" (the reply is a fine, finished, non-degenerate string), and only
    then does the tail discover it has no user text to pair it with. Before
    v3.1.7 `stored` had already been counted by that point, so /health/full
    reported a healthy tail for an exchange that never got near memory.

    WHAT WOULD MAKE THIS FAIL:
      * the `skipped_no_user_text` label being removed — _outcome() asserts
        the key exists rather than defaulting it to 0;
      * the condition moving back INSIDE _async_tail, so the request path
        counts `stored` and this counter never moves;
      * anything reaching the store for this conversation, which is the other
        half of the same claim and is checked per-conv, so concurrent traffic
        cannot mask it.
    The control above proves both halves DO move for an ordinary exchange, so
    neither assertion here is vacuously satisfiable.
    """
    H.skip_if_no_admin("R8 pins health output against the store")
    conv_id = H.fresh_conv_id()
    try:
        before = _tail()
        _assert_ledger_reconciles(before, f"before the {shape} exchange")

        status = _post_raw(
            conv_id, [{"role": "user", "content": last_user_content}]
        )
        # The USER still gets a reply — this is a memory decision, not a
        # request failure, and a 4xx here would mean the test is exercising
        # request validation instead of the tail.
        assert status == 200, f"{shape} request was rejected: HTTP {status}"

        _settle()
        after = _tail()
        _assert_ledger_reconciles(after, f"after the {shape} exchange")

        assert _outcome(after, "skipped_no_user_text") > _outcome(
            before, "skipped_no_user_text"
        ), (
            f"a {shape} exchange did not register as skipped_no_user_text. "
            f"before={before['outcomes']!r}\nafter={after['outcomes']!r}"
        )

        state = _conv(conv_id)
        assert state == {"facts": 0, "episodic": 0, "turns_seen": 0}, (
            f"the tail was counted as a skip but something reached the store "
            f"for this conversation anyway: {state!r} — this is R8 with the "
            f"sign reversed and is just as dishonest"
        )
    finally:
        H.admin_safe_forget(conv_id)


# ---------------------------------------------------------------------------
# R26 / R27 — an interrupted stream, and what it does to health
# ---------------------------------------------------------------------------


def test_r26_control_a_fully_consumed_stream_is_stored():
    """CONTROL for R26. A streaming exchange read to completion stores.

    Its job is to show that the streaming path reaches the store at all, so
    that the interrupted case below is a statement about the interruption
    rather than about streaming being broken.

    WHAT WOULD MAKE THIS FAIL: the streaming call site not invoking
    _run_memory_tail, or episodic indexing being off.
    """
    H.skip_if_no_admin("needs the store to check what was written")
    conv_id = H.fresh_conv_id()
    try:
        before = _tail()
        status, nbytes = _stream_fully(
            conv_id, "Streaming control: describe the harbour at dusk."
        )
        assert status == 200, status
        assert nbytes > 0, "the control stream delivered no bytes to the client"

        H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=30)
        _wait_for_turns_seen(conv_id, 2)
        after = _tail()
        _assert_ledger_reconciles(after, "after a fully consumed stream")

        state = _conv(conv_id)
        assert state["episodic"] >= 1, (
            f"a completed stream left nothing in the store: {state!r}"
        )
        assert _outcome(after, "stored") > _outcome(before, "stored")
    finally:
        H.admin_safe_forget(conv_id)


def test_r26_interrupted_stream_still_produces_a_counted_named_decision():
    """R26. A stream that ends early must still reach the tail's ledger.

    The defect was measured as "decide_memory_tail called: 0 times,
    tailhealth snapshot changed: False, log lines with 'memory tail': []" —
    the interruption did not merely lose the exchange, it lost the RECORD
    that anything had happened. The fix removed `and not vllm_failed` from
    the streaming call site so the tail decides on whatever did arrive.

    The interruption here is client-side (see _stream_and_abort for why that
    lands deterministically). Both interruptions arrive at the same `finally`
    and the same _run_memory_tail call, and the client-side one is the only
    half a black-box caller can produce against a fixture it must not
    reconfigure — see the report note on what this does NOT cover.

    WHAT WOULD MAKE THIS FAIL: restoring the `and not vllm_failed` guard, or
    any change that lets an interrupted stream leave the `finally` without a
    note() — the decision total would not move and the outcome label would
    not be published.
    """
    H.skip_if_no_admin("needs the store to check nothing was written")
    conv_id = H.fresh_conv_id()
    try:
        before = _tail()
        _assert_ledger_reconciles(before, "before an interrupted stream")

        status = _stream_and_abort(
            conv_id, "Interrupted stream: start telling me about the tides."
        )
        assert status == 200, f"the stream never opened: HTTP {status}"

        _settle()
        after = _tail()
        _assert_ledger_reconciles(after, "after an interrupted stream")

        # The core R26 claim: a decision was MADE. Under the defect this
        # number did not move at all for an interrupted stream.
        assert _decisions(after) > _decisions(before), (
            "an interrupted stream produced no tail decision at all — the "
            "exchange left no record that it had happened, which is R26 "
            "verbatim.\n"
            f"before stored={before['stored']} skipped={before['skipped']}\n"
            f"after  stored={after['stored']} skipped={after['skipped']}"
        )
        # And it was NAMED — but WHICH label wins is a race, not a guarantee,
        # and that used to be asserted as if it were structural.
        # _stream_and_abort's docstring argued the client's close always
        # lands before the compactor's first chunk because StreamingResponse
        # commits its headers first; that covers the header commit but not
        # the socket-close scheduling on the client's own side, and on one
        # measured run the close lost the race: the outcome came back
        # 'stored', not 'skipped_empty', because the accumulator already held
        # the first chunk. A bare `skipped_empty > before` then fails and
        # reads as an R26 regression when the actual cause is the harness
        # losing a timing race it does not control.
        #
        # The R26 claim itself (a decision was made at all — the assertion
        # above) is race-free and is not weakened by any of this. What is
        # race-dependent is only WHICH label fired, so accept either winner
        # and require it to agree with what the store actually holds — that
        # agreement is the property worth pinning; a specific label winning a
        # client-side timing race is not.
        stored_delta = _outcome(after, "stored") - _outcome(before, "stored")
        skipped_empty_delta = (
            _outcome(after, "skipped_empty") - _outcome(before, "skipped_empty")
        )
        state = _conv(conv_id)
        if skipped_empty_delta > 0:
            # The close won the race, as documented: nothing may have been
            # written, since there was no reply text to write.
            assert state == {"facts": 0, "episodic": 0, "turns_seen": 0}, (
                f"an aborted stream labelled skipped_empty wrote to the "
                f"store anyway: {state!r}"
            )
        elif stored_delta > 0:
            # The close lost the race: the first chunk landed before the
            # abort reached the server, so the honest label is 'stored', and
            # the store must show the exchange really arrived.
            assert state["episodic"] >= 1, (
                f"the interrupted stream was labelled 'stored' but nothing "
                f"reached the store: {state!r} — that combination is not a "
                f"race, it is R26 with a different outcome name"
            )
        else:
            raise AssertionError(
                "the interrupted stream was counted (a decision fired) but "
                "under neither 'skipped_empty' nor 'stored'; outcome deltas: "
                f"{ {k: v - before['outcomes'].get(k, 0) for k, v in after['outcomes'].items() if v != before['outcomes'].get(k, 0)} }"
            )
    finally:
        H.admin_safe_forget(conv_id)


def test_r27_a_harmless_skip_does_not_arm_the_degrade_flag():
    """R27. An empty reply must not make /health/full claim memory was lost.

    Reachable whenever the user hits Stop before the first token; the
    release's own figure is that 51 of 63 skips in one window were manual
    stops, and each of them used to pin /health/full to `degraded` for the
    full five-minute window over a turn that lost nothing.

    Asserted through the two CLOCKS rather than through `skipped_recently`
    directly, and that is deliberate. `skipped_recently` is a global, windowed
    flag: a genuinely lossy skip from any other traffic against this
    deployment — including this file's own R8 cases — legitimately holds it
    true for five minutes, so reading it as "R27 is broken" would be a test
    that fails for the correct behaviour of a neighbour. The invariant that is
    actually R27 is finer: a harmless skip advances `seconds_since_last_skip`
    (the general record) and must NOT advance
    `seconds_since_last_lossy_skip` (the degrade signal).

    WHAT WOULD MAKE THIS FAIL: dropping skipped_empty out of
    HARMLESS_SKIP_OUTCOMES, or keying the lossy clock off "any skip" again —
    the two clocks would then read the same and the strict comparison below
    would fail. The lossy control at the end of this test proves the lossy
    clock is not simply frozen, so the comparison is not vacuous.
    """
    H.skip_if_no_admin("uses the store to confirm the harmless skip lost nothing")
    conv_id = H.fresh_conv_id()
    lossy_conv = H.fresh_conv_id()
    try:
        # A pause first, so that any lossy skip already on the clock (this
        # file's R8 cases, or another suite's) is measurably OLDER than the
        # harmless one we are about to cause. Without the gap, "strictly
        # greater" could be decided by rounding to one decimal place.
        time.sleep(5)

        before = _tail()
        triggered_at = time.monotonic()
        status = _stream_and_abort(conv_id, "Stop me before I start.")
        assert status == 200, status
        _settle()
        after = _tail()
        _assert_ledger_reconciles(after, "after a harmless skip")

        assert _outcome(after, "skipped_empty") > _outcome(before, "skipped_empty"), (
            "the setup did not actually produce a harmless skip, so the rest "
            "of this test would be asserting nothing. NOTE: _stream_and_abort "
            "is not guaranteed to beat the first chunk (see its docstring) — "
            "if the outcome delta below shows 'stored' instead, the client "
            "lost that race rather than R27 being broken; rerun to confirm "
            "before treating this as a regression.\n"
            f"outcome deltas: "
            f"{ {k: v - before['outcomes'].get(k, 0) for k, v in after['outcomes'].items() if v != before['outcomes'].get(k, 0)} }"
        )
        # It really was harmless: nothing was lost, because nothing existed.
        assert _conv(conv_id) == {"facts": 0, "episodic": 0, "turns_seen": 0}

        since_any = after["seconds_since_last_skip"]
        since_lossy = after["seconds_since_last_lossy_skip"]
        assert isinstance(since_any, (int, float)), (
            f"the general skip clock did not move for a skip: {after!r}"
        )
        assert _newest_skip_is_ours(since_any, triggered_at), (
            f"seconds_since_last_skip is {since_any}s, more than has passed "
            f"since this test fired its trigger — the skip this test just "
            f"caused is not the most recent one, so the comparison below "
            f"would be about someone else's skip"
        )
        assert since_lossy is None or since_lossy > since_any, (
            f"a harmless skip advanced the LOSSY clock "
            f"(since_lossy={since_lossy}s vs since_any={since_any}s). That is "
            f"R27: /health/full is about to report 'reply(ies) not memorized' "
            f"for a turn that had nothing to memorize."
        )
        # The published flag must be exactly the lossy clock's verdict — the
        # field health.py keys `degraded` off, and the field an empty reply
        # must not be able to set on its own.
        window = float(after["skip_window_s"])
        expected = since_lossy is not None and since_lossy <= window
        assert after["skipped_recently"] is expected, (
            f"skipped_recently={after['skipped_recently']} does not follow "
            f"from since_lossy={since_lossy} against window={window}"
        )

        # ---- control: a LOSSY skip must move the clock the harmless one did
        # not. Without this, "the lossy clock stayed old" would be equally
        # consistent with a lossy clock that never moves at all.
        lossy_before = _tail()
        lossy_triggered_at = time.monotonic()
        status = _post_raw(lossy_conv, [{"role": "user", "content": "   "}])
        assert status == 200, status
        _settle()
        lossy_after = _tail()
        assert _outcome(lossy_after, "skipped_no_user_text") > _outcome(
            lossy_before, "skipped_no_user_text"
        ), "the lossy control did not produce a lossy skip"
        lossy_since = lossy_after["seconds_since_last_lossy_skip"]
        assert _newest_skip_is_ours(lossy_since, lossy_triggered_at), (
            f"a LOSSY skip left the lossy clock at {lossy_since!r}. The clock "
            f"is not tracking loss at all, which would make the harmless-skip "
            f"assertion above meaningless."
        )
        assert lossy_after["skipped_recently"] is True, (
            "a reply that was genuinely not memorized did not arm the degrade "
            "flag — R27's fix has been taken too far in the other direction"
        )
    finally:
        H.admin_safe_forget(conv_id)
        H.admin_safe_forget(lossy_conv)


# ---------------------------------------------------------------------------
# A2 / N4b — the task-traffic THRESHOLD, all three cases
# ---------------------------------------------------------------------------


def test_n4b_a_genuine_first_turn_is_memorized():
    """A2/N4b, case 1 of 3. A first turn looks EXACTLY like task traffic —
    one user message, no assistant turn behind it — and must still be
    memorized.

    This is the case that killed the backlog's own suggested fix direction:
    `_has_conversational_history` is False here too, so skipping the tail on
    that predicate alone silently drops the opening exchange of every new
    conversation.

    WHAT WOULD MAKE THIS FAIL: keying the skip off the request shape instead
    of the recorded position; lowering TASK_TRAFFIC_MIN_POSITION to 0. Either
    would leave the store empty, which the assertions below reject.
    """
    H.skip_if_no_admin("the claim is about what reached the store")
    conv_id = H.fresh_conv_id()
    try:
        assert _conv(conv_id)["turns_seen"] == 0, (
            "precondition: the conversation must be at position 0 for this to "
            "be a genuine first turn"
        )
        before = _tail()

        # No prior_turns: the array is one user message and nothing else,
        # byte-for-byte the shape OpenWebUI's title call sends.
        r = _chat(
            "First turn: the lighthouse keeper is called Mrs Ondermay.",
            conv_id=conv_id, prior_turns=[], max_tokens=64,
        )
        assert r.status_code == 200, r.status_code

        H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=30)
        _wait_for_turns_seen(conv_id, 2)
        after = _tail()

        state = _conv(conv_id)
        assert state["episodic"] >= 1, (
            f"the opening exchange of a new conversation was not memorized: "
            f"{state!r}"
        )
        assert state["turns_seen"] >= 2, (
            f"the tail did not run for the first turn: {state!r}"
        )
        assert _outcome(after, "skipped_task_traffic") == _outcome(
            before, "skipped_task_traffic"
        ), (
            "a genuine first turn was counted as task traffic. (Global "
            "counter: a concurrent task-traffic skip elsewhere would also "
            "trip this, which is the safe direction — the store assertions "
            "above are the concurrency-immune half of the claim.)"
        )
    finally:
        H.admin_safe_forget(conv_id)


def test_n4b_a_shallow_regenerate_is_memorized():
    """A2/N4b, case 2 of 3. Regenerating the first reply resends the same
    history-less array against a conv_id the store already knows — and must
    still be memorized.

    This is the case that killed the SECOND implementation: keying off "have
    we stored anything under this conv_id" eats exactly this shape, and
    test_budget_guard caught it within one run. The threshold exists for this
    case: two message-units is below TASK_TRAFFIC_MIN_POSITION (4).

    TWO SUB-CASES, and the second is the one with teeth. A byte-identical
    regenerate cannot be witnessed in the store at all — episodic doc ids are
    content-addressed (retrieval._doc_id), so re-storing the same exchange is
    correctly a no-op and adds no row — so it is checked against the global
    counters. The edited resend (same classification input: no assistant
    turn, position 2; different text) IS witnessable, and carries the claim.

    WHAT WOULD MAKE THIS FAIL: raising TASK_TRAFFIC_MIN_POSITION's effective
    bar to 2 or below, or going back to a bare "is there anything in the
    store" test — the edited resend would then be skipped and the episodic
    count would stay at 1.
    """
    H.skip_if_no_admin("the claim is about what reached the store")
    conv_id = H.fresh_conv_id()
    try:
        opening = "Regenerate case: the capital of my island is Verrindale."
        r = _chat(opening, conv_id=conv_id, prior_turns=[], max_tokens=64)
        assert r.status_code == 200
        H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=30)
        _wait_for_turns_seen(conv_id, 2)
        _settle()

        state = _conv(conv_id)
        # Precondition, asserted rather than assumed: this test is only about
        # the threshold if the conversation is genuinely BELOW it.
        assert 0 < state["turns_seen"] < 4, (
            f"precondition: the conversation must sit below "
            f"TASK_TRAFFIC_MIN_POSITION (4) after one exchange; it is at "
            f"{state['turns_seen']}. The threshold arithmetic has changed and "
            f"this test is no longer testing the shallow side of it."
        )
        baseline_episodic = state["episodic"]

        # --- sub-case A: the byte-identical regenerate --------------------
        before = _tail()
        r = _chat(opening, conv_id=conv_id, prior_turns=[], max_tokens=64)
        assert r.status_code == 200
        _settle()
        after = _tail()
        assert _outcome(after, "skipped_task_traffic") == _outcome(
            before, "skipped_task_traffic"
        ), (
            "regenerating the first reply of a one-exchange conversation was "
            "counted as task traffic — the position threshold has collapsed "
            "back into 'have we stored anything for this conv_id'. (Global "
            "counter; a concurrent task-traffic skip would also trip it.)"
        )
        assert _outcome(after, "stored") > _outcome(before, "stored"), (
            "the regenerate produced no stored decision at all"
        )

        # --- sub-case B: the edited resend, which the store can witness ----
        edited = opening + " It sits on the northern shore."
        before = _tail()
        r = _chat(edited, conv_id=conv_id, prior_turns=[], max_tokens=64)
        assert r.status_code == 200
        H.wait_for_indexed_exchanges(
            conv_id, min_count=baseline_episodic + 1, max_wait=30
        )
        _settle()

        state = _conv(conv_id)
        assert state["episodic"] >= baseline_episodic + 1, (
            f"a history-less turn on a conversation only one exchange deep "
            f"was not memorized: episodic stayed at {state['episodic']} "
            f"(baseline {baseline_episodic}). That is the silent memory loss "
            f"the threshold exists to prevent."
        )
        assert state["turns_seen"] >= 4, (
            f"the tail did not run for the edited resend: {state!r}"
        )
    finally:
        H.admin_safe_forget(conv_id)


def test_n4b_repeat_task_traffic_past_the_threshold_is_skipped_and_counted():
    """A2/N4b, case 3 of 3. The same shape, on a conversation that is two
    exchanges deep, IS task traffic: skipped, counted, and named.

    99 requests in two days on one conv_id were being fact-extracted,
    episodically indexed and deduped — N3's "second treadmill", against 0
    real exchanges. A conversation that has genuinely got two exchanges deep
    cannot honestly present an array with no assistant turn in it.

    WHAT WOULD MAKE THIS FAIL: removing the skip (the episodic count would
    grow and turns_seen would advance); removing the COUNTER (the outcome
    delta would not move, which is what made the original defect invisible);
    or moving the check after the tail has fired.

    Non-vacuous by construction: the two tests above drive a request of the
    IDENTICAL shape below the threshold and assert the opposite outcome, so
    what is being pinned here is the threshold itself, not "history-less
    requests are dropped".
    """
    H.skip_if_no_admin("the claim is about what did NOT reach the store")
    conv_id = H.fresh_conv_id()
    try:
        position = _drive_to_position(conv_id, exchanges=2)
        assert position >= 4, (
            f"precondition: two exchanges must put the conversation at or past "
            f"TASK_TRAFFIC_MIN_POSITION (4); it is at {position}. Either the "
            f"position arithmetic changed or the tail did not run — either "
            f"way this test would otherwise be testing the shallow side of "
            f"the threshold and passing for the wrong reason."
        )
        state_before = _conv(conv_id)
        before = _tail()
        triggered_at = time.monotonic()

        # The real shape: OpenWebUI's title generator, one user message, no
        # assistant turn, on the conv_id it has been shadowing all along.
        r = _chat(
            "### Task:\nGenerate a concise 3-5 word title for this chat.",
            conv_id=conv_id, prior_turns=[], max_tokens=24,
        )
        assert r.status_code == 200, r.status_code

        _settle()
        after = _tail()
        _assert_ledger_reconciles(after, "after task traffic")

        assert _outcome(after, "skipped_task_traffic") > _outcome(
            before, "skipped_task_traffic"
        ), (
            "task traffic past the threshold was not counted under "
            "skipped_task_traffic. Counting it is what stops it looking like "
            "one of the lossy skips — and what made the original defect "
            "visible at all."
        )

        state_after = _conv(conv_id)
        assert state_after["episodic"] == state_before["episodic"], (
            f"task traffic was episodically indexed anyway: "
            f"{state_before['episodic']} -> {state_after['episodic']}"
        )
        assert state_after["turns_seen"] == state_before["turns_seen"], (
            f"the tail ran for task traffic: turns_seen "
            f"{state_before['turns_seen']} -> {state_after['turns_seen']}"
        )
        assert state_after["facts"] == state_before["facts"], (
            f"task traffic was fact-extracted: {state_before['facts']} -> "
            f"{state_after['facts']}"
        )

        # v3.1.8 put skipped_task_traffic in HARMLESS_SKIP_OUTCOMES: declining
        # to remember a background task is not a loss and must not degrade
        # health. Same two-clock argument as the R27 case.
        since_any = after["seconds_since_last_skip"]
        since_lossy = after["seconds_since_last_lossy_skip"]
        assert _newest_skip_is_ours(since_any, triggered_at), (
            f"the skip this test caused is not the most recent one "
            f"(since_any={since_any}, older than this test's trigger); the "
            f"clock comparison below would be about someone else's skip"
        )
        assert since_lossy is None or since_lossy > since_any, (
            f"a task-traffic skip advanced the LOSSY clock "
            f"(since_lossy={since_lossy} vs since_any={since_any}) — "
            f"/health/full will report a memory loss for traffic the "
            f"compactor deliberately declined to remember"
        )
    finally:
        H.admin_safe_forget(conv_id)


# ---------------------------------------------------------------------------
# R28 / the ledger — the invariant that makes every counter above trustworthy
# ---------------------------------------------------------------------------


def test_r28_tail_ledger_reconciles_across_a_run_of_odd_exchanges():
    """R28, and the reconciliation invariant.

    `tailhealth.note` incremented the OUTCOME counter before a bare `int()`
    that could raise — out of a `finally`, the one place the module's
    docstring promises it never will — leaving the published block with
    sum(outcomes) != stored + skipped. The fix coerces safely (`_safe_int`)
    and makes the outcome tally the LAST mutation in the function, so that a
    future raise leaves the ledger stale rather than wrong.

    The raising input is not reachable from the API today (both call sites
    pass real ints), so what a black-box test can pin is the invariant the fix
    exists to preserve — checked after every step of a deliberately mixed run,
    because a run of ordinary exchanges would only ever exercise one branch of
    note().

    The six steps cover both storing branches and three distinct skip
    branches, which between them touch every mutation note() performs:
    stored/skipped, the outcome tally, the char totals, the consecutive-skip
    streak (set and reset), and both clocks.

    WHAT WOULD MAKE THIS FAIL: incrementing the outcome tally without the
    matching stored/skipped increment (R28 exactly), double-counting an
    exchange at the two call sites, or any new skip path that returns from
    _run_memory_tail without calling note() — the second assertion at the end
    catches that one, since the decision total would grow by less than the
    number of exchanges driven.
    """
    H.skip_if_no_admin("drives the admin store between steps")

    start = _tail()
    _assert_ledger_reconciles(start, "at the start of the mixed run")
    convs: list[str] = []

    def _fresh() -> str:
        cid = H.fresh_conv_id()
        convs.append(cid)
        return cid

    try:
        deep = _fresh()
        position = _drive_to_position(deep, exchanges=2)
        assert position >= 4, (
            f"precondition: the task-traffic step needs a conversation past "
            f"position 4; got {position}"
        )
        _assert_ledger_reconciles(_tail(), "after seeding the deep conversation")

        # Every step is (label, callable). Each drives exactly ONE exchange
        # through the tail, so the decision total must grow by at least
        # len(steps) over the run.
        steps = [
            ("plain non-streaming exchange",
             lambda: _chat("Odd run 1: the bell rings twice at noon.",
                            conv_id=_fresh(), max_tokens=64).status_code),
            ("unicode-heavy exchange",
             lambda: _chat("Odd run 2: 潮汐表 — señora Ünal's ferry ⛴ leaves 06:30. "
                            "Noted?", conv_id=_fresh(), max_tokens=64).status_code),
            ("whitespace-only user turn (lossy skip)",
             lambda: _post_raw(_fresh(), [{"role": "user", "content": "  \t "}])),
            ("parts list with no text part (lossy skip)",
             lambda: _post_raw(_fresh(), [{"role": "user",
                                           "content": [{"type": "input_audio",
                                                        "data": "zz"}]}])),
            ("interrupted stream (harmless skip)",
             lambda: _stream_and_abort(_fresh(), "Odd run 5: begin, then stop.")),
            ("repeat task traffic (harmless skip)",
             lambda: _chat("### Task:\nGenerate a concise title.",
                            conv_id=deep, prior_turns=[], max_tokens=24).status_code),
        ]

        for label, run in steps:
            status = run()
            assert status == 200, f"step {label!r} did not return 200: {status}"
            _settle()
            _assert_ledger_reconciles(_tail(), f"after {label}")

        end = _tail()
        _assert_ledger_reconciles(end, "at the end of the mixed run")

        # Every exchange that reached the tail is in the ledger exactly once,
        # so the totals must have grown by AT LEAST the number driven here.
        # `>=` and not `==` only because the counter is process-global; the
        # direction that would catch a lost decision is the one being checked.
        drove = len(steps)
        grew = _decisions(end) - _decisions(start)
        assert grew >= drove, (
            f"{drove} exchanges reached the tail but the ledger only grew by "
            f"{grew}. At least one decision was never counted, which is the "
            f"silent-skip class this whole module exists to end."
        )
        # And the outcome tally grew by the same amount as the totals — the
        # two halves of the invariant moving together, not merely agreeing on
        # a value they were both already at.
        def _sum(mt: dict) -> int:
            return sum(int(v) for v in (mt.get("outcomes") or {}).values())

        assert _sum(end) - _sum(start) == grew, (
            f"outcome tally grew by {_sum(end) - _sum(start)} while "
            f"stored+skipped grew by {grew} — the two halves of the ledger "
            f"disagree about the same run of exchanges (R28)."
        )
    finally:
        for cid in convs:
            H.admin_safe_forget(cid)


@pytest.mark.skip(
    reason="Needs a reply longer than MIN_MEMORABLE_TRIMMED_CHARS (300) that "
           "ends on a sentence boundary. The local integration fixture has no "
           "model weights and answers every completion with the same ~65-char "
           "canned string, so a trimmed partial can never clear the floor, and "
           "the only lever that would change it (POST /_fixture/mode "
           "reply_chars) is GLOBAL state on a fixture shared with other suites. "
           "The 'long partial reply is stored' half of R26 is pinned instead by "
           "compactor/test_disconnect_uvloop.py, which hangs up a real socket "
           "with SO_LINGER 0 under real uvicorn/uvloop and measures "
           "stored_trimmed at 6,139 bytes (V314_BACKLOG A1). Un-skip this when "
           "running the --profile model stack, whose 0.5B GGUF produces real "
           "prose."
)
def test_r26_long_partial_reply_is_stored_via_the_trim_path():
    """R26's other half: an interrupted stream carrying ENOUGH prose is
    memorized rather than discarded — the trim path keeps everything up to the
    last sentence boundary.

    Left as an executable statement of the missing coverage rather than
    deleted, so the gap is visible in the run summary as a SKIP with a reason
    instead of being invisible.
    """
    H.skip_if_no_admin()
    conv_id = H.fresh_conv_id()
    try:
        before = _tail()
        # Read part of the body, then hang up — with a long reply this leaves
        # the accumulator holding several complete sentences.
        body = {
            "model": H.resolve_model(),
            "messages": [{"role": "user", "content":
                          "Write several paragraphs about the tide tables."}],
            "max_tokens": 1024,
            "stream": True,
            "metadata": {"chat_id": conv_id},
        }
        seen = 0
        with httpx.Client(base_url=H.BASE_URL, timeout=H.TIMEOUT) as c:
            with c.stream("POST", "/v1/chat/completions", json=body,
                          headers={"X-Conversation-Id": conv_id}) as resp:
                for raw in resp.iter_raw():
                    seen += len(raw)
                    if seen > 4000:
                        break
                resp.close()
        assert seen > 400, (
            f"the reply was only {seen} bytes — too short to carry a trimmed "
            f"prefix past the 300-char floor, so this test cannot make its "
            f"claim on this stack"
        )
        H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=30)
        after = _tail()
        assert _outcome(after, "stored_trimmed") > _outcome(before, "stored_trimmed")
        assert _conv(conv_id)["episodic"] >= 1
    finally:
        H.admin_safe_forget(conv_id)

