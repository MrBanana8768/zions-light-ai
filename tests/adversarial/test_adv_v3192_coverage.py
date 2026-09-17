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
    fixture (summarization, fact extraction, persona) from the async tail,
    and those carry the turn text — marker included — as INPUT. They land
    AFTER the forwarded request, so `matches[-1]` picked a summarizer call:
    it reported `sampling={'temperature': 0.0}` and no repetition_penalty,
    which looked exactly like the translation failing to reach vLLM. The
    forwarded array is the one whose LAST message is the newest user turn
    (the one the marker is on); a summarizer payload ends with its own
    instruction. Fall back to the earliest match, since the forward always
    precedes the tail it triggers.
    """
    tail_marked = [
        c for c in chats
        if c.get("messages") and marker in (c["messages"][-1].get("markers") or [])
    ]
    if tail_marked:
        return tail_marked[0]
    any_marked = [
        c for c in chats
        if any(marker in (m.get("markers") or []) for m in c.get("messages", []))
    ]
    return any_marked[0] if any_marked else None


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

    NOT PROVEN (documented, not asserted): that the reuse CODE PATH
    specifically fired rather than a second fresh summarize — the fixture
    has no way to distinguish those from outside (both produce a small
    forwarded prompt and a real 200). The chat_completions call-count
    comparison below is the closest available black-box signal and is
    reported, not hard-asserted, because fact extraction and episodic
    indexing in the async tail also call the same endpoint and add noise
    unrelated to reuse.
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

    stats_before_2 = fixture_client.get("/_fixture/stats").json()
    r2 = _chat(client, msgs2, conv)
    calls_2_immediate = (
        fixture_client.get("/_fixture/stats").json().get("chat_completions", 0)
        - stats_before_2.get("chat_completions", 0)
    )

    assert r2.status_code == 200, (
        f"NO CAP REFUSAL: request 2 (a growing conversation on the same "
        f"conv_id, with a stored hierarchy behind it) must still answer 200; "
        f"got {r2.status_code}: {r2.text[:400]}"
    )
    reply2 = (r2.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    assert reply2, "request 2 answered 200 with no content — a refusal in disguise"

    fwd2 = _forwarded_prompt_tokens(r2)
    record(
        "advcov-reuse",
        f"n1_true={n1_true} calls_for_request_1(incl. its own forward)={calls_1} "
        f"n2_true={n2_true} fwd2={fwd2} "
        f"calls_for_request_2_before_its_tail_settles(incl. its own forward)="
        f"{calls_2_immediate} (NOT hard-asserted — see docstring)",
    )
    assert fwd2 is not None and fwd2 < n2_true * 0.5, (
        f"STAND-IN NOT ON THE WIRE: request 2 forwarded {fwd2} of {n2_true} "
        f"true tokens — the older span was not meaningfully compacted/"
        f"substituted"
    )
    assert _settle(client), "background tail work never drained after request 2"
