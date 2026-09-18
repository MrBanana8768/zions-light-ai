"""ADVERSARIAL: black-box coverage for the v3.1.9.1/.2 line (advfix, P8 gate
item 4).

p8's gate note: "Nothing in any stack exercises v3.1.9.1/.2." Three gaps
named explicitly:

  1. a request carrying repeat_penalty (and both names) — is the Ollama ->
     vLLM sampling-param translation actually reaching the wire?
  2. a conversation whose history contains a degenerate reply — what reaches
     the vLLM fixture (the forwarded-window loop redaction, v3.1.9.2)?
  3. reuse driven with a stored hierarchy — no cap refusal, and the stand-in
     is on the wire.

`tests/adversarial/test_adv_v319_*.py` (P8 gate item 2) do NOT cover any of
this — they attack `ced4520`'s reuse CONTENT/INDEX guard (test_adv_v319_
reuse.py), the event loop and degenerate message ARRAYS (test_adv_v319_
loop.py), and webui.db sync (test_adv_v319_webuidb.py), none of which is the
sampling translation, the forwarded-window redaction, or the reuse budget
this file covers — so this is new coverage, not an extension of theirs.

WHY A FIXTURE CHANGE WAS NEEDED. Before this file, nothing in the fixture
recorded what a /v1/chat/completions BODY actually contained — only what came
back (`usage.prompt_tokens`) could be inspected, which proves a token COUNT
but not which sampling keys or which message shapes arrived. `GET
/_fixture/last_chats` (testfixtures/tokenizer-contract/fixture_server.py) is
new, purely additive (a new list, one call at the top of the existing
handler, one new GET endpoint, cleared by the existing POST /_fixture/reset),
and deliberately carries no message CONTENT — only role, length, and any
`ADVFIX-WIRE-MARKER-*` token a test embedded on purpose — so this fixture
never becomes a second place production or synthetic prose ends up recorded.
"""

import time
import uuid

import pytest

from conftest import MODEL, record  # noqa: F401

TARGET_TOKENS = 18432  # COMPACTOR_TARGET_TOKENS default at HARD_INPUT_LIMIT*0.75
_WORD = "quick brown fox jumps over lazy dogs "
_MARKER = "ADVFIX-WIRE-MARKER-"


def _chat(client, messages, conv: str, **extra):
    body = {"model": MODEL, "messages": messages, "stream": False}
    body.update(extra)
    return client.post(
        "/v1/chat/completions", headers={"X-Conversation-Id": conv}, json=body
    )


def _true_count(fixture_client, messages) -> int:
    r = fixture_client.post(
        "/tokenize",
        json={"model": MODEL, "messages": messages, "add_generation_prompt": True},
    )
    r.raise_for_status()
    return int(r.json()["count"])


def _forwarded_prompt_tokens(resp) -> int | None:
    try:
        return int(resp.json()["usage"]["prompt_tokens"])
    except Exception:
        return None


def _forwarded_match(chats: list[dict], marker: str) -> dict | None:
    """The record for the FORWARDED chat request carrying `marker`.

    The compactor makes its own /v1/chat/completions calls to the same
    fixture (summarization, fact extraction) from the async tail, and
    those carry the turn text — marker included — as INPUT. They land
    AFTER the forwarded request, so `matches[-1]` picked a summarizer call:
    it reported `sampling={'temperature': 0.0}` and no repetition_penalty,
    which looked exactly like the translation failing to reach vLLM.

    P9-6 (hostile pass #9): the original fix for that used a PRIMARY rule
    (marker on the LAST message — the forwarded array's last message is
    always the newest user turn) with a FALLBACK to "the earliest match"
    for tests whose marker sits elsewhere (e.g. inside an assistant turn
    mid-history). The fallback is an ORDERING assumption, not a
    discriminator — it happens to hold today because compaction makes no
    backend call for a sub-TARGET conversation, but if that ever stops
    being true the fallback would silently pick a summarizer/facts payload
    again, the exact bug this function exists to avoid, with no test able
    to tell.

    Fixed: identify the forward POSITIVELY instead. Both known internal
    callers set an explicit `temperature` on every request —
    `_summarize_once` (main.py) sends 0.2, facts extraction
    (facts.py::extract_facts) sends 0.0 — because both want deterministic-
    ish structured output, not creative sampling. The forwarded request is
    the CLIENT'S own body verbatim (main.py's two real forwarding call
    sites post `body` unmodified) with whatever sampling keys the client
    sent, which in this suite's `_chat()` helper never includes
    `temperature` unless a test explicitly asks for it. So: the forward is
    the marked record whose `sampling` has NO `temperature` key at all —
    true regardless of where in the array the marker sits. Asserts exactly
    one such record matches, rather than trusting index 0 of however many
    there are (a genuine ambiguity — two candidates — is now a test
    failure, not a silent pick).
    """
    marked = [
        c for c in chats
        if any(marker in (m.get("markers") or []) for m in c.get("messages", []))
    ]
    forwarded = [c for c in marked if "temperature" not in (c.get("sampling") or {})]
    assert len(forwarded) <= 1, (
        f"P9-6: {len(forwarded)} records carrying marker {marker!r} all lack "
        f"an explicit temperature — cannot tell which is the real forward "
        f"(records: {forwarded})"
    )
    return forwarded[0] if forwarded else None


def _last_chats(fixture_client) -> list[dict]:
    r = fixture_client.get("/_fixture/last_chats")
    r.raise_for_status()
    return r.json()["chats"]


def _settle(client, timeout=120) -> bool:
    """Wait for background tail work to finish, same pattern
    test_adv_faults.py's flood test uses — needed before a second request can
    see the hierarchy the first request's tail wrote."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/health/full").json()["background_work"]["outstanding"] == 0:
            return True
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# 1. repeat_penalty -> repetition_penalty translation reaches the wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body_extra, expect_repetition_penalty, expect_repeat_penalty_key, label",
    [
        ({"repeat_penalty": 1.3}, 1.3, False, "repeat_only"),
        (
            {"repeat_penalty": 1.5, "repetition_penalty": 1.1},
            1.1,
            False,
            "both_present_repetition_wins",
        ),
        ({"repeat_penalty": "not-a-number"}, None, False, "repeat_invalid_dropped"),
    ],
    ids=["repeat_only", "both_present_repetition_wins", "repeat_invalid_dropped"],
)
def test_repeat_penalty_translation_reaches_the_wire(
    client, fixture_client, body_extra, expect_repetition_penalty,
    expect_repeat_penalty_key, label,
):
    """v3.1.9.2 (advfix, P8 gate item 4). `_translate_ollama_sampling_params`
    (main.py ~7483) claims to translate Ollama's `repeat_penalty` into vLLM's
    `repetition_penalty` before forwarding, dropping `repeat_penalty` either
    way, and to prefer an existing valid `repetition_penalty` when both are
    sent. Nothing in any stack drove this end to end and inspected the
    forwarded body — /health/full and usage.prompt_tokens cannot see a
    sampling key. This asserts what actually reaches "vLLM" via the fixture's
    new `/_fixture/last_chats` introspection.

    FAILS IF: repeat_penalty reaches the wire under its own name, the
    translated/kept repetition_penalty value is wrong, or an invalid
    repeat_penalty with no repetition_penalty present is forwarded anyway.
    """
    conv = f"advcov-repeat-{label}-{uuid.uuid4().hex[:8]}"
    marker = f"{_MARKER}{label}"
    msgs = [{"role": "user", "content": f"{marker} hello, please respond briefly"}]
    r = _chat(client, msgs, conv, **body_extra)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"

    chats = _last_chats(fixture_client)
    forwarded = _forwarded_match(chats, marker)
    assert forwarded is not None, (
        f"no /_fixture/last_chats entry carries marker {marker!r} — the "
        f"fixture never saw this request forwarded at all"
    )
    wire = forwarded["sampling"]
    record(
        "advcov-repeat-penalty",
        f"[{label}] sent={body_extra} wire_sampling={wire}",
    )

    assert "repeat_penalty" not in wire, (
        f"GOOD NEWS did not land: repeat_penalty reached the wire under its "
        f"own name ({wire}) — vLLM 0.19 does not recognise it and silently "
        f"ignores it, so the client's requested penalty was never honoured"
    )
    if expect_repetition_penalty is None:
        assert "repetition_penalty" not in wire, (
            f"an invalid repeat_penalty with no repetition_penalty present "
            f"should be dropped entirely, not forwarded: {wire}"
        )
    else:
        assert wire.get("repetition_penalty") == pytest.approx(
            expect_repetition_penalty
        ), f"expected repetition_penalty={expect_repetition_penalty}, wire={wire}"


# ---------------------------------------------------------------------------
# 2. a degenerate reply in history — what reaches the vLLM fixture
# ---------------------------------------------------------------------------


def test_degenerate_reply_in_history_is_redacted_on_the_wire(client, fixture_client):
    """v3.1.9.2 (advfix, P8 gate item 4). `_redact_forwarded_loop_replies`
    (main.py:7397, called at main.py:8305-8307) rewrites every non-newest
    ASSISTANT turn `reply_is_degenerate` flags, AFTER compaction and memory
    injection, BEFORE `_enforce_hard_budget` — so the guard measures what is
    actually sent. Nothing in any stack drove a real chat_completions request
    whose HISTORY contains a degenerate reply and inspected what left the
    process. This does, via `/_fixture/last_chats`.

    The conversation is kept small (well under TARGET_TOKENS) and the marker
    tokens are plain identifier-shaped tokens so ONLY the redaction path —
    not compact_if_needed's summarization — can be responsible for any
    marker loss: a short conversation is returned unchanged by
    compact_if_needed's own trigger (main.py:1913-1915).

    The loop uses an underscore-joined identifier run
    (`batch_handler_shared_batch_handler_shared_...`), the exact shape
    INCIDENT_2026-08-29 was written for and `_TOKEN_RUN_RE`
    (DEGENERATE_TOKEN_RUN_CHARS=120) targets — not a char-run or a fenced
    box, so this is insensitive to the fence-exemption history (P8-2/P8-7)
    this lane did not touch.

    FAILS IF: the assistant turn's forwarded length is not meaningfully
    smaller than what was sent (the loop was forwarded verbatim), or the
    fixture never sees the request at all (the redaction path crashed and
    nothing was forwarded, or the marker convention broke).

    A first version of this test put a marker token immediately AFTER the
    loop and asserted it was gone — and that assertion was WRONG, caught by
    running it: `_cut_degenerate_span_once` (main.py:3494) only takes the
    "nothing after it worth keeping" tail branch when the matched span
    reaches the (stripped) end of the text; text after the span in the
    ordinary mid-reply case (`post = text[end:]`) is DELIBERATELY kept
    verbatim beside the cut marker (P8-4's own subject). A marker placed
    right after the loop lands in `post` and is SUPPOSED to survive — that
    is not a redaction failure, it was a wrong test. Fixed by ending the
    message exactly at the loop's own end (a genuine tail-loop, the shape
    `_TAIL_LOOP_WINDOW`/the "runs to the end" branch is for) and measuring
    length instead of planting a marker inside the part meant to be cut.
    """
    conv = f"advcov-degenreply-{uuid.uuid4().hex[:8]}"
    head_marker = f"{_MARKER}CLEANHEAD"
    # Well over MIN_MEMORABLE_TRIMMED_CHARS(300, main.py:3851) once trimmed to
    # the last sentence boundary — a first version of this test sized the
    # head to "around 300" and it landed just under after trimming, which
    # fell to the whole-reply placeholder instead of a kept, cut head; sized
    # generously here instead of tuned to the exact floor.
    clean_head = (
        f"Here is an ordinary opening sentence that explains something "
        f"useful and unremarkable, with enough length to clear the "
        f"redaction's floor for kept clean text before a cut. "
        f"It continues for a few more ordinary sentences so the kept head is "
        f"comfortably over the floor even after trimming to a sentence "
        f"boundary. Here is another plain sentence adding more harmless "
        f"length. And one more, for good measure, well past the floor. "
        f"Marker token {head_marker} sits inside this complete sentence so "
        f"trimming to the last sentence boundary keeps it. "
    )
    # 21 repeats of a 20-char alphanumeric+underscore token = ~440 chars,
    # comfortably over DEGENERATE_TOKEN_RUN_CHARS(120); _TOKEN_RUN_RE requires
    # an alphanumeric in the repeated unit, which this has. Runs to the very
    # end of the message (no trailing marker) — a genuine tail loop.
    loop = ("batch_handler_shared " * 21).rstrip()
    assistant_degenerate = f"{clean_head}{loop}"
    original_len = len(assistant_degenerate)

    msgs = [
        {"role": "user", "content": "Tell me about your batch handler design."},
        {"role": "assistant", "content": assistant_degenerate},
        {"role": "user", "content": "Thanks — anything else I should know?"},
    ]
    n_true = _true_count(fixture_client, msgs)
    assert n_true < TARGET_TOKENS, (
        f"fixture too large ({n_true} tokens) — compaction would confound the "
        f"redaction-only signal this test needs"
    )

    r = _chat(client, msgs, conv)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"

    chats = _last_chats(fixture_client)
    wire = _forwarded_match(chats, head_marker)
    assert wire is not None, "no /_fixture/last_chats entry carries the clean-head marker"
    assistant_wire = [m for m in wire["messages"] if m["role"] == "assistant"]
    record(
        "advcov-degenerate-reply-wire",
        f"true_tokens={n_true} forwarded={_forwarded_prompt_tokens(r)} "
        f"original_assistant_len={original_len} "
        f"assistant_turns_on_wire={assistant_wire}",
    )

    assert assistant_wire, "the assistant turn itself is missing from the wire"
    wire_len = assistant_wire[0]["len"]
    assert wire_len < original_len * 0.6, (
        f"THE LOOP WAS NOT REDACTED: the degenerate assistant turn was sent "
        f"as {original_len} chars and reached the wire as {wire_len} — not "
        f"meaningfully shorter, so the forwarded-window redaction did not "
        f"act on this shape"
    )
    # The clean head must still be legible after the cut (this project's own
    # design goal — a redacted reply is not supposed to erase content that
    # was never part of the loop).
    assert head_marker in {
        mk for m in wire["messages"] for mk in m.get("markers", [])
    }, "the clean head marker did not survive the cut"


# ---------------------------------------------------------------------------
# 3. reuse driven with a stored hierarchy
# ---------------------------------------------------------------------------


def _pairs_of(n: int, start: int = 0) -> list[dict]:
    msgs = []
    for i in range(start, start + n):
        msgs.append({"role": "user", "content": f"turn {i}. " + _WORD * 30})
        msgs.append({"role": "assistant", "content": f"reply {i}. " + _WORD * 30})
    return msgs


def test_reuse_with_a_stored_hierarchy_no_cap_refusal_stand_in_on_wire(
    client, fixture_client
):
    """v3.1.9.1/.2 (advfix, P8 gate item 4). `compact_if_needed`'s reuse path
    (main.py ~1938-2160) substitutes a stand-in rendered from the STORED
    summary hierarchy for turns the hierarchy already covers, instead of
    re-summarizing them from scratch — but only once a prior turn's
    background tail work has actually written that hierarchy
    (`summarizer.maybe_rollup`). No stack drives two requests on the same
    conv_id with a settle in between and inspects the second one.

    Request 1 is long enough (many pairs of ~200-token turns) to force
    compact_if_needed to summarize the older span from nothing — that
    summarize() call, and the request's own forward, both land on
    `/_fixture/stats.chat_completions` (the fixture serves both). After
    settling (background_work.outstanding == 0, so maybe_rollup's write has
    landed), request 2 reuses the SAME leading turns verbatim (unchanged
    objects — content fingerprints must match for the coverage record to
    pair them) plus a few new pairs, and should need at most a small,
    bounded number of ADDITIONAL backend calls for the new tail — not a
    second full re-summarization of the whole older span.

    "No cap refusal": request 2 must still answer 200 with real content, not
    a refusal — `compact_if_needed`'s all_or_nothing stand-in has a declined-
    reuse fallback (full summarize) specifically so a squeezed budget cannot
    refuse the turn; this proves that fallback (or the reuse path itself)
    keeps the request alive under a repeated, growing conversation.

    "The stand-in is on the wire": request 2's forwarded prompt is far
    smaller than its true (uncompacted) token count — the older span did not
    go out verbatim.

    P9-4 (hostile pass #9): the assertion below used to be `fwd2 <
    n2_true * 0.5` alone — satisfied by a factor of 20 with NO stored
    hierarchy at all (`stored_turns_out=[0]`), because a full decline
    still summarizes the older span from scratch and returns a short
    array; a small forwarded prompt and a real 200 are what the DECLINED
    path produces too, not evidence reuse specifically fired. Fixed:
    `/health/full`'s `checks.reuse` (main.py `reuse_decline_state()`,
    hostile pass #9's P9-1/P9-2 fix, same compactor process this test
    already drives via `client`) is the compactor's OWN bookkeeping of
    whether a stand-in was attempted and whether it was declined for
    budget — evidence the declined path cannot fake, unlike anything
    inferable from the wire alone. Request 2 must show `attempted`
    increment by exactly one (an attempt was made) and `declined_budget`
    NOT increment (it was not declined) — i.e., the stand-in fit and was
    used. `reuse_decline_state()`'s own contract (numbers only: attempted/
    declined_budget/declined_recently/last_declined_ceiling/
    last_declined_others, no conversation text) is exercised directly, end
    to end through `compact_if_needed`, by `compactor/test_reuse_fit.py`
    section `[10]` and its mutation table (this lane's report,
    SP\\fix-loops4.md) — including the exact "declines, then reuses" shape
    this assertion depends on. This file's own suite runs inside a real
    docker-compose adversarial stack (`-p loops4`) rather than as a
    unit-level `pytest`, which I did not stand up for this specific change
    (heavy — the lane brief's own warning about three concurrent stacks
    crashing the shared VM applies); the logic is the same p9-findings.md
    itself used to prove this exact finding ("proven at the
    compact_if_needed level instead, which is the layer the assertion
    actually depends on").

    P11-2 (hostile pass #11) RE-OPENED the discriminator above, in the
    commit that says it made reuse failures visible (v3.1.9.1 -> v3.1.9.2,
    P10-3): `attempted` moved to count every request that reaches the top
    of `compact_if_needed`'s `if conv_id:` block (a live QUESTION, not yet
    an answer), and only a BUDGET decline moves `declined_budget` any more
    — a brand-new conversation with no stored hierarchy at all ("no_state"),
    a hierarchy that covers none of this array ("no_coverage"), and the
    reuse block's own `except` firing ("error") each now produce
    `attempted +1, declined_budget +0` too, which the OLD assertion pair
    reads as "the stand-in fit and was used" for all three. Proven at the
    `compact_if_needed` + `reuse_decline_state()` layer (same layer, same
    method as P9-4's own proof above — this repo's established way to
    check a `checks.reuse` claim no black-box HTTP fixture fault-injection
    hook can drive; SP\\gatecov\\p112_repro.py, synthetic only, no real
    data): the OLD pair (`attempted == before+1` AND `declined_budget ==
    before`) PASSES on all three, and ALSO on a fourth shape this test
    cannot construct at all through real HTTP — reuse recorded "success"
    (`checks.reuse.succeeded` incremented) and THEN the fresh-span
    `summarize()` call 503s, `compact_if_needed` raises, and
    `chat_completions` forwards the original array (P11-1, a SEPARATE
    finding: `_reuse_reason` is fixed as "success" before that call runs,
    at main.py's `finally` block around line 2507, so a fresh-span failure
    after it changes nothing this test can read — that recording-order
    defect belongs to lane reuse's `compact_if_needed`, not to this file).

    FIXED: the discriminator now requires `succeeded == before + 1` AND
    `last_reason == "success"` — no_state/no_coverage/error each leave
    `succeeded` unchanged and set a different `last_reason`, so all three
    now correctly fail this assertion (proven red, synthetic, in
    p112_repro.py's phases A/B/C). The fourth shape (success recorded,
    then the fresh-span summarize fails) is NOT closed by this file alone
    — it needs P11-1 landing in `compact_if_needed` so a post-recording
    failure is reported as "error" rather than "success"; once that lands,
    THIS SAME assertion (unchanged) closes it too. Two dedicated tests
    below (`test_reuse_discriminator_reports_no_state_not_success` and
    `..._no_coverage_not_success`) drive the no_state and no_coverage
    shapes through REAL `/v1/chat/completions` calls (no fault-injection
    hook needed — a brand-new conv_id and an unrelated-content conv_id are
    both ordinary client behaviour), so those two are covered end to end
    by this pytest suite, not only by the standalone script.
    """
    conv = f"advcov-reuse-{uuid.uuid4().hex[:8]}"

    msgs1 = _pairs_of(50) + [{"role": "user", "content": "and so what should I do next?"}]
    n1_true = _true_count(fixture_client, msgs1)
    assert n1_true > TARGET_TOKENS, (
        f"fixture too small ({n1_true} true tokens) to force first-time "
        f"summarization — widen _pairs_of's count"
    )

    stats_before_1 = fixture_client.get("/_fixture/stats").json()
    r1 = _chat(client, msgs1, conv)
    assert r1.status_code == 200, f"request 1: HTTP {r1.status_code}: {r1.text[:300]}"
    assert _settle(client), "background tail work never drained after request 1"
    stats_after_1 = fixture_client.get("/_fixture/stats").json()
    calls_1 = stats_after_1.get("chat_completions", 0) - stats_before_1.get("chat_completions", 0)

    # Request 2: the SAME leading turns (unchanged) plus a short new tail,
    # ending on a new final user turn — the shape reuse is supposed to cover.
    msgs2 = msgs1[:-1] + _pairs_of(4, start=50) + [
        {"role": "user", "content": "one more thing — what about tomorrow?"}
    ]
    n2_true = _true_count(fixture_client, msgs2)

    # P9-4: the compactor's OWN reuse bookkeeping, before/after request 2.
    # `checks.reuse` may be `{"available": False, ...}` on a build without
    # main.reuse_decline_state() (main.py cannot be assumed to be exactly
    # HEAD in every environment this runs against) — treat that as a hard
    # failure for THIS assertion rather than silently skipping it, since a
    # missing signal is exactly the P9-1 failure mode (invisible, not
    # absent).
    reuse_before = client.get("/health/full").json()["checks"].get("reuse") or {}
    assert reuse_before.get("available"), (
        f"P9-4 precondition: /health/full's checks.reuse is not available "
        f"({reuse_before}) — main.reuse_decline_state() is missing or the "
        f"wiring broke; this test cannot tell reuse from decline without it"
    )

    stats_before_2 = fixture_client.get("/_fixture/stats").json()
    r2 = _chat(client, msgs2, conv)
    calls_2_immediate = (
        fixture_client.get("/_fixture/stats").json().get("chat_completions", 0)
        - stats_before_2.get("chat_completions", 0)
    )
    reuse_after = client.get("/health/full").json()["checks"].get("reuse") or {}

    assert r2.status_code == 200, (
        f"NO CAP REFUSAL: request 2 (a growing conversation on the same "
        f"conv_id, with a stored hierarchy behind it) must still answer 200; "
        f"got {r2.status_code}: {r2.text[:400]}"
    )
    reply2 = (r2.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    assert reply2, "request 2 answered 200 with no content — a refusal in disguise"

    # P9-4 (hostile pass #9): the hard proof. `fwd2 < n2_true * 0.5` below
    # is satisfied by the DECLINED path too (a full re-summarize also
    # shrinks the forwarded prompt) — it stays as supporting evidence, not
    # the discriminator.
    #
    # P11-2 (hostile pass #11) RE-OPENED this: `attempted` incrementing by
    # exactly one and `declined_budget` NOT moving stopped being evidence
    # only a successful stand-in produces the moment P10-3 gave no_state,
    # no_coverage and error their OWN counters instead of folding them into
    # `declined_budget` — all three now also produce `attempted +1,
    # declined_budget +0` (see this test's own docstring for the proof).
    # Kept below as a WEAKER, supporting check (still true of a genuine
    # success, still useful in a failure's error message), but no longer
    # the discriminator.
    assert reuse_after.get("attempted") == reuse_before.get("attempted", 0) + 1, (
        f"request 2 did not register a reuse attempt at all "
        f"(before={reuse_before}, after={reuse_after}) — either no stored "
        f"hierarchy covers anything yet (settle after request 1 did not "
        f"land, or the hierarchy genuinely does not cover these turns), or "
        f"main.compact_if_needed's reuse block was not reached"
    )
    # THE DISCRIMINATOR (P11-2's fix): `succeeded` is the ONE counter that
    # moves on exactly one `_reuse_reason` value ("success" — see
    # main.py's `_record_reuse_outcome`), and `last_reason` names it
    # directly rather than requiring the reader to rule out every OTHER
    # reason by checking a counter that reason does not touch. no_state,
    # no_coverage, error and a BUDGET decline all leave `succeeded`
    # unchanged and set `last_reason` to their own name — none of them can
    # produce this pair.
    assert reuse_after.get("succeeded") == reuse_before.get("succeeded", 0) + 1, (
        f"THE REUSE CODE PATH DID NOT SUCCEED: request 2's `succeeded` "
        f"counter did not move (before={reuse_before}, after={reuse_after}) "
        f"— whatever produced the small forwarded prompt below, it was not "
        f"a stand-in that rendered; this is exactly the P9-1/P9-2/P11-2 "
        f"failure mode (the feature silently not firing, or firing and "
        f"then not being distinguishable from a decline) and NOT what this "
        f"test's name claims to cover"
    )
    assert reuse_after.get("last_reason") == "success", (
        f"request 2's last recorded reuse outcome was "
        f"{reuse_after.get('last_reason')!r}, not 'success' "
        f"(before={reuse_before}, after={reuse_after}) — declined_budget, "
        f"no_state, no_coverage and error are all DECLINES/faults, not the "
        f"successful stand-in this test's name claims to prove. NOTE "
        f"(P11-1, not fixed by this file): if lane reuse's fix for P11-1 "
        f"has not landed yet, `last_reason` can still read 'success' here "
        f"even when the fresh-span summarize() call AFTER this point 5xxs "
        f"and the original array was forwarded instead — see this test's "
        f"docstring for why that specific shape needs main.py's fix too"
    )

    fwd2 = _forwarded_prompt_tokens(r2)
    record(
        "advcov-reuse",
        f"n1_true={n1_true} calls_for_request_1(incl. its own forward)={calls_1} "
        f"n2_true={n2_true} fwd2={fwd2} "
        f"calls_for_request_2_before_its_tail_settles(incl. its own forward)="
        f"{calls_2_immediate} (NOT hard-asserted — see docstring) "
        f"reuse_before={reuse_before} reuse_after={reuse_after} (P9-4: the "
        f"hard-asserted evidence that the reuse code path specifically fired)",
    )
    assert fwd2 is not None and fwd2 < n2_true * 0.5, (
        f"STAND-IN NOT ON THE WIRE: request 2 forwarded {fwd2} of {n2_true} "
        f"true tokens — the older span was not meaningfully compacted/"
        f"substituted"
    )
    assert _settle(client), "background tail work never drained after request 2"


# ---------------------------------------------------------------------------
# 3b. P11-2: the discriminator must NOT read a decline as a success
# ---------------------------------------------------------------------------
#
# Two of P11-2's four false-positive shapes (no_state, no_coverage) are
# ordinary client behaviour — no fault-injection hook needed — so they are
# driven here through the real endpoint, same as the test above. The other
# two (the reuse block's own `except` firing; reuse recorded "success" and
# THEN the fresh-span summarize() call fails) need a fault this fixture has
# no hook for on /v1/chat/completions (only /tokenize has one — see
# testfixtures/tokenizer-contract/fixture_server.py's `/_fixture/mode`), so
# they are proven at the `compact_if_needed` + `reuse_decline_state()` layer
# instead (SP\\gatecov\\p112_repro.py, synthetic only) — the same layer, and
# the same reason, as the test above's own docstring already documents for
# P9-4's original proof.


def test_reuse_discriminator_reports_no_state_not_success(client, fixture_client):
    """P11-2. A brand-new conv_id's FIRST request, already over TARGET_TOKENS,
    reaches `compact_if_needed`'s reuse block (an attempt is a live question
    the instant there is a conv_id, older text turns, and the request is over
    budget — see `_record_reuse_attempt`'s own comment) and finds no stored
    hierarchy at all: `_covered == 0` and nothing in the covered-turn record,
    so `_reuse_reason = "no_state"` (main.py ~2272-2277). No settle, no
    second request, no fault injection — this is what any brand-new
    conversation's first oversized request does.

    FAILS IF: `succeeded` moves, or `last_reason` reads "success" — the
    OLD discriminator (`attempted +1, declined_budget +0`) PASSES on this
    shape too (proof in the test above's docstring), which is exactly what
    P11-2 is about: a request that never had anything to reuse must not
    read as a successful reuse.
    """
    conv = f"advcov-reuse-nostate-{uuid.uuid4().hex[:8]}"
    msgs = _pairs_of(50) + [{"role": "user", "content": "one more thing?"}]
    n_true = _true_count(fixture_client, msgs)
    assert n_true > TARGET_TOKENS, (
        f"fixture too small ({n_true} true tokens) to force compaction on "
        f"a first request — widen _pairs_of's count"
    )

    reuse_before = client.get("/health/full").json()["checks"].get("reuse") or {}
    assert reuse_before.get("available"), (
        f"precondition: /health/full's checks.reuse is not available "
        f"({reuse_before}) — main.reuse_decline_state() is missing or the "
        f"wiring broke"
    )
    r = _chat(client, msgs, conv)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    reuse_after = client.get("/health/full").json()["checks"].get("reuse") or {}

    record(
        "advcov-reuse-discriminator",
        f"[no_state] n_true={n_true} reuse_before={reuse_before} "
        f"reuse_after={reuse_after}",
    )
    assert reuse_after.get("last_reason") == "no_state", (
        f"setup did not reach the shape this test needs — expected "
        f"last_reason='no_state' on a brand-new conv_id's first oversized "
        f"request, got {reuse_after.get('last_reason')!r} "
        f"(before={reuse_before}, after={reuse_after})"
    )
    assert reuse_after.get("succeeded") == reuse_before.get("succeeded", 0), (
        f"P11-2: a conv_id with NOTHING stored yet must not increment "
        f"`succeeded` (before={reuse_before}, after={reuse_after})"
    )
    assert _settle(client), "background tail work never drained"


def test_reuse_discriminator_reports_no_coverage_not_success(client, fixture_client):
    """P11-2. Request 1 builds a real stored hierarchy for a conv_id (settled,
    same as the happy-path test above). Request 2, on the SAME conv_id,
    replaces the ENTIRE history with unrelated content sharing no text with
    request 1 — the "different branch / delete-and-regenerate" shape
    `_coverage_plan` is written to detect (main.py ~2257-2264): a hierarchy
    EXISTS (`summarizer._covered_fps` is non-empty) but covers none of THIS
    array, so `_reuse_reason = "no_coverage"`.

    FAILS IF: `succeeded` moves, or `last_reason` reads "success" — same
    P11-2 shape as the no_state test above, on the other counter P10-3
    split out.
    """
    conv = f"advcov-reuse-nocoverage-{uuid.uuid4().hex[:8]}"
    msgs1 = _pairs_of(50) + [{"role": "user", "content": "and so what should I do next?"}]
    n1_true = _true_count(fixture_client, msgs1)
    assert n1_true > TARGET_TOKENS, (
        f"fixture too small ({n1_true} true tokens) to force first-time "
        f"summarization — widen _pairs_of's count"
    )
    r1 = _chat(client, msgs1, conv)
    assert r1.status_code == 200, f"request 1: HTTP {r1.status_code}: {r1.text[:300]}"
    assert _settle(client), "background tail work never drained after request 1"

    # Unrelated content on the SAME conv_id — a different word bank and a
    # disjoint turn-number range, so no turn's fingerprint can pair with the
    # covered-turn record request 1's tail wrote.
    unrelated_word = "papaya quokka xylophone zeppelin umbrella vortex "
    msgs2 = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"unrelated turn {i}. " + unrelated_word * 30}
        for i in range(100)
    ] + [{"role": "user", "content": "one more thing?"}]
    n2_true = _true_count(fixture_client, msgs2)
    assert n2_true > TARGET_TOKENS, (
        f"fixture too small ({n2_true} true tokens) — widen the unrelated "
        f"turn count"
    )

    reuse_before = client.get("/health/full").json()["checks"].get("reuse") or {}
    r2 = _chat(client, msgs2, conv)
    assert r2.status_code == 200, f"request 2: HTTP {r2.status_code}: {r2.text[:300]}"
    reuse_after = client.get("/health/full").json()["checks"].get("reuse") or {}

    record(
        "advcov-reuse-discriminator",
        f"[no_coverage] n1_true={n1_true} n2_true={n2_true} "
        f"reuse_before={reuse_before} reuse_after={reuse_after}",
    )
    assert reuse_after.get("last_reason") == "no_coverage", (
        f"setup did not reach the shape this test needs — expected "
        f"last_reason='no_coverage' when request 2 replaces the whole "
        f"history with unrelated content, got "
        f"{reuse_after.get('last_reason')!r} (before={reuse_before}, "
        f"after={reuse_after}) — if this reads 'no_state', request 1's "
        f"hierarchy never got built/settled; if it reads something else, "
        f"the unrelated content accidentally paired with the covered-turn "
        f"record"
    )
    assert reuse_after.get("succeeded") == reuse_before.get("succeeded", 0), (
        f"P11-2: a hierarchy that covers NONE of this array must not "
        f"increment `succeeded` (before={reuse_before}, after={reuse_after})"
    )
    assert _settle(client), "background tail work never drained after request 2"
