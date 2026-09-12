"""
Tier-3 regression net for the TEXT-HANDLING defects fixed in v3.1.7/v3.1.8.

Every defect below was found in production or in review, fixed, and is now
only as fixed as the thing that keeps it fixed. This file is that thing, at
the black-box level: it imports no compactor code and hits the public API
(plus the vLLM fixture's own control plane, which is part of the stack, not
part of the compactor).

What each case pins, and what would make it fail:

  R7/R14 — SseAccumulator corrupted a UTF-8 character split across two raw
    reads.  test_multibyte_reply_reaches_memory_intact drives a REAL SSE
    stream and asserts the multibyte characters the client received are the
    ones the memory store holds, with no U+FFFD anywhere.
    test_sse_wire_carries_raw_multibyte_bytes is the other half — the byte
    boundary itself — and it SKIPS LOUDLY rather than passing quietly when
    the stack cannot put a multibyte character on the wire at all. Read its
    docstring before believing this file covers R7 end to end; it does not,
    and it says why.

  v3.1.8 — rule/box decoration reached MEMORY and was injected back into
    every prompt.  test_decoration_never_reaches_the_fact_store and
    test_decoration_never_reaches_the_summary drive the two write paths
    `compactor/textclean.py` guards (facts._parse_extraction_output and
    summarizer._summarize_pieces) with a reply that is 17% box-drawing, and
    assert the borders are gone AND the words are not. Both halves matter:
    a fix that ate the words would pass a one-sided test.

  R9/R19/R24/R25 — ordinary prose judged degenerate and then PERMANENTLY
    redacted from memory.  test_decorated_prose_reply_is_still_memorized
    pins the reachable half of that: a heavily decorated but genuinely
    prose reply is STORED. The abbreviation / long-enumeration / em-dash
    shapes are NOT reachable from here — see that test's docstring, and
    `compactor/test_degenerate_reply.py`, which owns them.

  N1/R3 — an interior assistant turn with empty content made vLLM reject
    the whole request.  test_interior_empty_assistant_turn_completes drives
    every emptiness shape the repair covers, plus the non-text-part shape it
    must NOT treat as empty, and asserts the exchange completes and still
    reaches memory. Its docstring records what this stack cannot prove.

RUNNING IT (the stack must already be up; do not rebuild it under other
agents):

    docker compose -f docker-compose.integration.yml run --rm --no-deps \\
        integration-tests -k regression_text -v

The fixture-mode helper below mutates SHARED state on the vLLM fixture.
Every use restores it, and every use verifies the mode actually took —
because a silently-canned reply would make the decoration cases assert
nothing at all.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

import _harness as H

# The fixture's own port, inside the compose network. The tests container
# shares the compactor's network namespace, so this name resolves for both.
#
# THIS FILE NEEDS THE WEIGHTLESS STACK AND MUST NOT BE GUARDED WITH
# H.requires_real_model(). Every decoration case here drives the fixture's
# adversarial generator through /_fixture/mode, and `vllm-fixture` is a
# service of the DEFAULT compose profile only — the `model` profile builds
# `model-fixture` instead, a different hostname that this name does not
# resolve to. So the real-weights profile is the one place these cases cannot
# run, and a real-weights guard on them would skip the only profile that can.
#
# Said here because a v3.1.9 review listed
# test_decorated_prose_reply_is_still_memorized among the tests that "fail
# against the weightless fixture" and should carry the guard. It does not:
# its assertion is `indexed_exchanges >= 1`, episodic indexing is an
# embedding and an upsert with no generation in it, and its controls are on
# the generator's deterministic output. Nothing in it reads a model's words.
FIXTURE_URL = "http://vllm-fixture:8000"

# Generous, deliberately. The stub answers in microseconds but the memory
# tail behind each turn does not, this stack is shared with other agents,
# and the harness default (ZIONS_TEST_TIMEOUT=30) has already turned a busy
# compactor into a ReadTimeout that looks nothing like the defect under test.
SLOW_TIMEOUT = 180.0

# U+2500-U+259F: box drawing + block elements. The same range
# compactor/textclean.py builds _RULE_CHARS from, written out here rather
# than imported, because this suite does not import compactor code.
_BOX_LO, _BOX_HI = 0x2500, 0x259F

REPLACEMENT_CHAR = "�"


# ---------------------------------------------------------------------------
# Helpers (local on purpose — _harness.py is shared with two other agents)
# ---------------------------------------------------------------------------


def box_chars(text: str) -> list[str]:
    """Every box-drawing/block character in `text`. Returned as a list, not
    a count, so a failure message can name what it found."""
    return [c for c in (text or "") if _BOX_LO <= ord(c) <= _BOX_HI]


def non_ascii(text: str) -> set[str]:
    return {c for c in (text or "") if ord(c) > 127}


def fixture_mode_get() -> dict:
    with httpx.Client(base_url=FIXTURE_URL, timeout=30.0) as c:
        r = c.get("/_fixture/mode")
        r.raise_for_status()
        return r.json()


def fixture_mode_set(**kw) -> dict:
    with httpx.Client(base_url=FIXTURE_URL, timeout=30.0) as c:
        r = c.post("/_fixture/mode", json=kw)
        r.raise_for_status()
        return r.json()


def fixture_reply_text(reply_chars: int, reply_seq: int) -> str:
    """The exact string the fixture's adversarial generator will produce for
    (reply_chars, reply_seq), fetched from the fixture itself.

    Deterministic in those two numbers — `_adversarial_reply` is a pure
    function of them and ignores the prompt — so this is the text the
    compactor's chat, extraction AND summarize calls all receive. Fetching
    it rather than hard-coding it is what lets the assertions below say
    "the words that went in came out" instead of guessing at needles that
    might silently stop appearing.
    """
    with httpx.Client(base_url=FIXTURE_URL, timeout=30.0) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "fixture-model",
                "messages": [{"role": "user", "content": "probe"}],
                "max_tokens": 64,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class adversarial_replies:
    """Context manager: make every fixture completion an adversarial reply.

    SHARED STATE. The fixture is one container for the whole stack, so this
    both restores the previous value on exit and VERIFIES on entry that the
    mode took — another agent resetting it mid-test would otherwise turn a
    decoration assertion into an assertion about the canned one-liner, which
    contains no decoration and would therefore pass having tested nothing.
    That is the exact failure this project has been bitten by twice.
    """

    def __init__(self, reply_chars: int, reply_seq: int,
                 looping: bool = True) -> None:
        self.reply_chars = reply_chars
        self.reply_seq = reply_seq
        # looping=False asks the fixture for NON-cycling padding. The
        # default walk repeats the same ~212-char phrase every 34 words,
        # which v3.1.8's fourth degeneracy rule (a phrase repeating to the
        # end for 400+ chars) correctly refuses — so a long decorated
        # reply cannot be legitimate prose unless this is False.
        self.looping = looping
        self.raw = ""

    def __enter__(self) -> "adversarial_replies":
        self._before = fixture_mode_get()
        fixture_mode_set(reply_chars=self.reply_chars, reply_seq=self.reply_seq,
                         reply_looping=self.looping)
        self.raw = fixture_reply_text(self.reply_chars, self.reply_seq)
        if not box_chars(self.raw):
            self.__exit__(None, None, None)
            pytest.fail(
                "the fixture is not producing decorated replies "
                f"(reply_chars={self.reply_chars}, got {len(self.raw)} chars "
                f"with no box-drawing characters). Everything downstream of "
                f"this would assert nothing. Mode now: {fixture_mode_get()}"
            )
        return self

    def __exit__(self, *exc) -> None:
        # Restore rather than zero: another agent may have had a mode set.
        fixture_mode_set(
            reply_chars=int(self._before.get("reply_chars") or 0),
            reply_seq=int(self._before.get("reply_seq") or 0),
        )


def chat_slow(
    user_msg: str,
    *,
    conv_id: str,
    prior_turns: list | None = None,
    max_tokens: int = 512,
) -> tuple[int, str]:
    """H.chat with a timeout this stack can actually meet. Returns
    (status_code, reply_text) — the status is part of what N1 asserts, so it
    must not be swallowed."""
    msgs = list(prior_turns or []) + [{"role": "user", "content": user_msg}]
    body = {
        "model": H.resolve_model(),
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "metadata": {"chat_id": conv_id},
    }
    with httpx.Client(base_url=H.BASE_URL, timeout=SLOW_TIMEOUT) as c:
        r = c.post(
            "/v1/chat/completions", json=body,
            headers={"X-Conversation-Id": conv_id},
        )
    text = ""
    try:
        text = (r.json().get("choices") or [{}])[0].get(
            "message", {}).get("content") or ""
    except Exception:
        pass
    return r.status_code, text


def stream_chat(user_msg: str, *, conv_id: str, max_tokens: int = 1024):
    """Drive the STREAMING path and return (status, raw_bytes, reply_text).

    `H.chat` is non-streaming and the accumulator only exists on the
    streaming path, so R7/R14 cannot be reached through the harness. The raw
    bytes are returned as well as the text because the compactor forwards
    the upstream chunks verbatim (`yield chunk` beside `accumulator.feed`),
    which makes them the closest observation available of what the
    accumulator was actually fed.
    """
    body = {
        "model": H.resolve_model(),
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "metadata": {"chat_id": conv_id},
    }
    raw = bytearray()
    with httpx.Client(base_url=H.BASE_URL, timeout=SLOW_TIMEOUT) as c:
        with c.stream(
            "POST", "/v1/chat/completions", json=body,
            headers={"X-Conversation-Id": conv_id},
        ) as r:
            status = r.status_code
            for chunk in r.iter_raw():
                raw += chunk
    text = []
    for line in bytes(raw).decode("utf-8", "replace").split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
        if isinstance(delta.get("content"), str):
            text.append(delta["content"])
    return status, bytes(raw), "".join(text)


def stored_documents(conv_id: str) -> list[str]:
    """The episodic store's own text for this conversation.

    This is the strongest "what actually reached memory" observable that
    needs no model weights: episodic indexing is an embedding and an upsert,
    with no LLM call in it, so it happens on this stack exactly as it does
    in production — unlike fact extraction and summarization, whose CONTENT
    is whatever the stub was told to say.
    """
    return [
        e.get("document", "")
        for e in (H.admin_export(conv_id).get("episodic") or [])
    ]


# ---------------------------------------------------------------------------
# R7 / R14 — a multibyte character split across a raw read boundary
# ---------------------------------------------------------------------------


def test_multibyte_reply_reaches_memory_intact(conv_id):
    """The end-to-end half of R7/R14: what the client read is what memory got.

    `SseAccumulator` decoded every raw chunk independently, so a UTF-8
    sequence straddling two TCP reads became U+FFFD on both sides — and the
    corrupted text, not the text the client received, is what went to
    memory. The fix holds one incremental decoder for the life of the
    accumulator and flushes it in finalize().

    WHAT WOULD MAKE THIS FAIL: any character the client saw going missing
    or arriving as U+FFFD in the episodic document. That covers the
    incremental decoder being reverted to a per-chunk `decode(...,
    "replace")` WHEN a split occurs, finalize() not being called (a
    character pending at end-of-stream would be dropped), and the store
    itself mangling non-ASCII on the way to disk.

    WHAT IT DOES NOT COVER, stated because a green here is easy to
    over-read: on this stack a split cannot occur — see
    test_sse_wire_carries_raw_multibyte_bytes. The needle assertion below
    is real, but it is about the decode/store round trip, not about the
    boundary. The boundary is pinned at the unit level by
    compactor/test_sse_accumulator.py.
    """
    H.skip_if_no_admin("reading what reached the store needs the admin API")

    status, raw, reply = stream_chat(
        "Please answer briefly.", conv_id=conv_id, max_tokens=1024,
    )
    assert status == 200, f"streaming request returned {status}"

    # CONTROL. Without this the test would "pass" on a reply that contained
    # no multibyte character at all — an empty needle set makes every
    # assertion below vacuously true, which is precisely the shape this
    # project keeps catching in its own suites.
    needles = non_ascii(reply)
    assert needles, (
        "the reply the client received contains no non-ASCII character, so "
        "there is nothing for this test to check. The fixture's canned reply "
        "normally carries an em dash. Reply was:\n"
        f"{reply[:300]!r}"
    )

    docs = stored_documents(conv_id) if H.wait_for_indexed_exchanges(
        conv_id, min_count=1, max_wait=60) else []
    assert docs, (
        "the exchange never reached the episodic store, so this test cannot "
        "say anything about what memory holds. Check /health/full's "
        "memory_tail outcomes."
    )
    stored = "\n".join(docs)

    assert REPLACEMENT_CHAR not in stored, (
        f"U+FFFD in stored text — a character was decoded across a boundary "
        f"and replaced. Stored:\n{stored[:500]!r}"
    )
    missing = sorted(c for c in needles if c not in stored)
    assert not missing, (
        f"characters the client received are absent from memory: "
        f"{[f'U+{ord(c):04X}' for c in missing]}\n"
        f"client reply: {reply[:200]!r}\nstored: {stored[:300]!r}"
    )


def test_sse_wire_carries_raw_multibyte_bytes(conv_id):
    """The boundary half of R7/R14 — and an honest SKIP when it is absent.

    R7 is a defect about BYTES: a multibyte character whose encoding lands
    partly in one `aiter_raw()` chunk and partly in the next. To reproduce
    it end to end the upstream SSE stream has to carry raw UTF-8 in the
    first place.

    It does not. `testfixtures/tokenizer-contract/fixture_server.py` builds
    every streamed event as `f"data: {json.dumps(chunk)}\\n\\n"`, and
    `json.dumps` defaults to `ensure_ascii=True`, so `💚`, `━` and `—` go on
    the wire as `\\uXXXX` escapes and are only turned back into characters
    by `json.loads` INSIDE the accumulator, long after any read boundary.
    Measured on this stack: 2,704 bytes over 13 chunks, 0 of them > 127.

    So this test measures the precondition and SKIPS, loudly and by name,
    when it is unmet — rather than asserting something that cannot fail. If
    the fixture ever emits `ensure_ascii=False` (or a real vLLM is put
    behind this suite), the skip lifts by itself and the assertions below
    become the end-to-end pin R7 deserves.

    WHAT WOULD MAKE THIS FAIL once it stops skipping: a per-chunk decode
    turning a straddling character into U+FFFD in the stored text.
    """
    H.skip_if_no_admin("reading what reached the store needs the admin API")

    # A long reply gives the sender something to coalesce and the reader a
    # reason to split — a one-line reply arrives in a single read and could
    # never straddle anything even on a wire that carried real UTF-8.
    status, raw, reply = stream_chat(
        "Please answer at length.", conv_id=conv_id, max_tokens=4096,
    )
    assert status == 200, f"streaming request returned {status}"

    wire_non_ascii = sum(1 for b in raw if b > 127)
    if wire_non_ascii == 0:
        pytest.skip(
            "the byte-split case cannot be produced on this stack: the "
            f"upstream SSE stream carried {len(raw)} bytes and none of them "
            "was > 127. fixture_server.py serialises every event with "
            "json.dumps(...) at its ensure_ascii=True default, so no "
            "multibyte character is ever on the wire to be split. R7/R14's "
            "byte boundary is pinned instead by "
            "compactor/test_sse_accumulator.py; this test starts working "
            "the day the fixture (or a real vLLM) puts UTF-8 on the wire."
        )

    if not H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=60):
        pytest.fail("the exchange never reached the episodic store")
    stored = "\n".join(stored_documents(conv_id))
    assert REPLACEMENT_CHAR not in stored, (
        f"U+FFFD in stored text after {wire_non_ascii} non-ASCII bytes "
        f"crossed the wire — a character was split across a read boundary "
        f"and decoded per-chunk. Stored:\n{stored[:500]!r}"
    )
    missing = sorted(c for c in non_ascii(reply) if c not in stored)
    assert not missing, (
        f"characters the client received are absent from memory: "
        f"{[f'U+{ord(c):04X}' for c in missing]}"
    )


# ---------------------------------------------------------------------------
# v3.1.8 — rule/box decoration must not reach memory
# ---------------------------------------------------------------------------


def test_decoration_never_reaches_the_fact_store(conv_id):
    """Fact write path: `facts._parse_extraction_output` stores the STRIPPED
    line (v3.1.8), and `_reject_reason` refuses a line with no alphanumeric
    content at all.

    Measured 2026-09-04 over 976 real replies: 28.4% carried box-drawing
    characters and 8 of 156 fact files held 3,119 of them — re-injected into
    every prompt, teaching the model to keep drawing them.

    The reply here is the fixture's adversarial generator: a 62-character
    U+2501 rule (twice), a markdown heading, a JSON fence, a status
    dashboard and prose. Extraction runs against exactly that text, because
    extraction is itself a completion and the fixture answers every
    completion the same way.

    WHAT WOULD MAKE THIS FAIL: any of the guards on that path being lost —
    `_reject_reason`'s "no alphanumeric content" rule (which is what keeps
    the two bare rules out) or the `strip_rule_decoration` call that
    replaced storing `line` verbatim. Either one gone and a fact carries
    U+2501.

    HONEST LIMIT, measured rather than guessed: the fixture emits no MIXED
    line (`━━━ Current Status ━━━`), which is the exact shape v3.1.8 added
    the strip for. Every decorated line it does emit is refused one step
    earlier by `_reject_reason` (bare rule -> "no alphanumeric content",
    `# Status Report` -> heading, the dashboard line -> status dashboard).
    This case was RUN against a compactor built before v3.1.8, with no
    textclean.py in it at all, and PASSED — so what it pins is
    `_reject_reason`, not `strip_rule_decoration`. The strip itself is
    covered by compactor/test_rule_decoration.py, and end to end by
    test_decoration_never_reaches_the_summary below.
    """
    H.skip_if_no_admin("reading the fact store needs the admin API")

    with adversarial_replies(reply_chars=900, reply_seq=5) as adv:
        status, reply = chat_slow(
            "Give me your status report, please.", conv_id=conv_id,
        )
        assert status == 200, f"chat returned {status}"
        # CONTROL, on the reply the client actually got rather than on the
        # generator's own output: this is the proof that decoration entered
        # the system on this run at all.
        assert box_chars(reply), (
            "the reply carried no decoration, so nothing downstream is being "
            f"tested. Reply:\n{reply[:300]!r}"
        )
        # INSIDE the context, and that is not tidiness. Fact extraction is a
        # completion of its own, fired by the async memory tail AFTER the
        # response returns; restoring the fixture mode first means the
        # extractor is answered with the canned one-liner and every
        # assertion below is about a string with no decoration in it. Cost
        # one failing run to learn.
        facts = H.wait_for_facts(conv_id, min_count=1, max_wait=60)

    assert facts, (
        "no fact was extracted from a decorated reply, so this test cannot "
        "say whether decoration would have survived into one. That is a "
        "different failure from the one under test — check extraction is "
        "enabled and the memory tail stored this exchange."
    )

    # A second control, on the extraction call rather than on the visible
    # reply: the extractor must have been answered adversarially too, or the
    # box-character assertion below is checking a string that never had a
    # border in it.
    all_fact_text = " ".join(f.get("text", "") for f in facts)
    assert "no model weights" not in all_fact_text, (
        "the extractor was answered with the fixture's CANNED reply, not the "
        "decorated one — the fixture mode was restored (or clobbered by "
        "another agent) before the async memory tail ran. Nothing below "
        "would be tested. Facts:\n"
        + "\n".join(f"  - {f.get('text', '')[:120]!r}" for f in facts)
    )

    for f in facts:
        found = box_chars(f.get("text", ""))
        assert not found, (
            f"a stored fact carries {len(found)} box-drawing character(s) "
            f"({sorted(set(found))}) — decoration reached the fact store and "
            f"will be injected into every subsequent prompt.\n"
            f"fact: {f.get('text', '')[:300]!r}"
        )

    # THE OTHER HALF. A "fix" that dropped the decorated lines wholesale,
    # or stripped them down to nothing, would pass every assertion above
    # and lose the words — which is worse than the defect. So require that
    # prose from the reply actually survived into the store.
    assert "eiusmod tempor incididunt" in all_fact_text, (
        "the decoration is gone but so are the words: no fact contains the "
        "prose that was in the reply. Facts:\n"
        + "\n".join(f"  - {f.get('text', '')[:120]!r}" for f in facts)
    )


def test_decoration_never_reaches_the_summary(conv_id):
    """Summary write path: `summarizer._summarize_pieces` strips decoration
    off EVERY tier's text before it is stored (v3.1.8).

    Measured 2026-09-04: 3 of 14 summary files carried 1,173 box characters,
    including the live summary of the conversation in daily use — and the
    summary block is injected on every single request.

    Driven through import + `/admin/.../compact` rather than by sending 20
    chat turns. That is not a shortcut around the code under test: the
    endpoint rebuilds the transcript and runs the SAME rollup, so
    `_summarize_pieces` is the function that produces the chunk text either
    way. It is a shortcut around twenty turns on a shared stack, which cost
    ~4 minutes and hit a backend ConnectTimeout when measured.

    WHAT WOULD MAKE THIS FAIL: the strip being removed, or moved from the
    `_summarize_pieces` wrapper down onto a subset of
    `_summarize_pieces_raw`'s five return paths (the fix-one-site-miss-the-
    sibling shape this project has paid for repeatedly) — the L1 chunk would
    then carry the generator's 124 U+2501 characters. Verified failing on
    this stack against a compactor container built before v3.1.8: the L1
    text came back with all 124 of them.
    """
    H.skip_if_no_admin("import + compact + summary state all need admin")

    # Ten indexed exchanges at turns 2..20 rebuild to exactly the 20-message
    # transcript one L1 chunk covers, with no placeholder padding (the
    # endpoint refuses a rebuild that is more gap than transcript).
    bundle = {
        "version": "v2.1",
        "exported_at": int(time.time()),
        "source_conv_id": f"{conv_id}-seed",
        "facts": [],
        "summary_state": {},
        "episodic": [
            {
                "turn_index": 2 * i,
                "document": (
                    f"[user]: Question {i} about the garden.\n"
                    f"[assistant]: Answer {i}. She waters the roses early."
                ),
            }
            for i in range(1, 11)
        ],
    }
    status, body = H.admin_import(bundle, target_conv_id=conv_id, overwrite=True)
    assert status == 200, f"import failed: {status} {body}"
    assert (body.get("imported") or {}).get("episodic") == 10, body

    with adversarial_replies(reply_chars=900, reply_seq=6) as adv:
        raw_summary_reply = adv.raw
        with httpx.Client(base_url=H.ADMIN_URL, timeout=SLOW_TIMEOUT) as c:
            r = c.post(
                f"/admin/conversations/{conv_id}/compact", json={"dry_run": False},
            )
        assert r.status_code == 200, f"compact refused: {r.status_code} {r.text[:400]}"
        plan = r.json()
        assert plan.get("l1_after", 0) >= 1, (
            f"no L1 chunk was produced, so there is no summary to inspect: {plan}"
        )

    # CONTROL: the text the summarizer was answered with really did carry
    # decoration. Without this the assertion below could pass because the
    # fixture went canned, not because the strip worked.
    assert box_chars(raw_summary_reply), (
        "the summarize call would not have returned any decoration; nothing "
        "below is being tested"
    )

    state = H.admin_get_summary(conv_id)
    chunks = list(state.get("l1") or []) + list(state.get("l2") or [])
    if state.get("l3"):
        chunks.append(state["l3"])
    assert chunks, f"summary state holds no chunk: {json.dumps(state)[:400]}"

    for ch in chunks:
        found = box_chars(ch.get("text", ""))
        assert not found, (
            f"a stored summary chunk carries {len(found)} box-drawing "
            f"character(s) ({sorted(set(found))}). The summary block is "
            f"injected on every request, so this is decoration being fed "
            f"back to the model every turn.\n"
            f"If this compactor was built before v3.1.8 it has no "
            f"compactor/textclean.py and cannot pass — restart the stack "
            f"from the current tree.\n"
            f"chunk text: {ch.get('text', '')[:300]!r}"
        )

    # THE WORDS HALF, measured rather than sampled: nearly every
    # alphanumeric character of the model's answer must still be there.
    # Stripping borders removes no letters and no digits, so a big drop here
    # means the strip ate content.
    def alnum(s: str) -> int:
        return sum(1 for c in s if c.isalnum())

    kept = sum(alnum(ch.get("text", "")) for ch in chunks)
    sent = alnum(raw_summary_reply)
    assert kept >= 0.9 * sent, (
        f"the decoration is gone but so are the words: {kept} alphanumeric "
        f"characters survived of {sent} the model returned. Stripping rules "
        f"must not remove letters or digits."
    )


# ---------------------------------------------------------------------------
# R9 / R19 / R24 / R25 — degeneracy false positives
# ---------------------------------------------------------------------------


def test_decorated_prose_reply_is_still_memorized(conv_id):
    """A decorated reply is a STYLE, not a repetition loop — it must be
    stored.

    `reply_is_degenerate` refusing a reply is not a skipped write: via
    `_redact_degenerate_turns` the reply is replaced by a placeholder in
    every future rollup, backfill and admin compact, permanently. That is
    why the R9/R19/R24/R25 family is about FALSE POSITIVES, and why the
    thresholds are calibrated rather than chosen.

    The reply driven here is the shape the calibration is about: 124
    box-drawing characters (~17% decoration, against a 45% limit), two
    62-character single-character runs (against a 250 limit), a markdown
    heading, a JSON fence, and a single unbroken 500+ character prose line.
    Storing it is the assertion.

    WHAT WOULD MAKE THIS FAIL: any tightening back toward the pre-v3.1.7
    behaviour — DEGENERATE_DECOR_FRACTION dropping under ~0.18,
    DEGENERATE_RUN_CHARS under 62, the structural rules losing their
    DEGENERATE_MIN_CHARS floor (R19) or their "runs to the end of the
    reply" requirement (R9) — turns this exchange into
    `skipped_degenerate`, and `indexed_exchanges` stays at 0.

    WHAT THIS CANNOT PIN, and it is the larger half of the family: R24
    (`Dr.` / `9 a.m.` collapsing the fragment-line mean), R9's 50+ item
    legitimate enumeration, and R25's em dash before an abbreviation are
    all judgements about the ASSISTANT REPLY's text, and this fixture has
    no weights — its reply is a canned string or this generator, never
    anything derived from the prompt. There is no black-box lever that puts
    chosen prose into an assistant reply on this stack, so those three
    shapes are unreachable from here. They are owned by
    compactor/test_degenerate_reply.py and compactor/test_sentence_trim.py.
    The same wall makes the TRUE POSITIVE unreachable: the generator's
    longest single-character run is 62 and its longest repeated-token run is
    one word, so nothing it can emit trips any rule.
    """
    H.skip_if_no_admin("indexed_exchanges is an admin observable")

    with adversarial_replies(reply_chars=1400, reply_seq=9,
                             looping=False) as adv:
        status, reply = chat_slow(
            "Tell me how the project is going.", conv_id=conv_id,
        )
        assert status == 200, f"chat returned {status}"

    # CONTROLS on the reply itself, so a green cannot be read as "the
    # detector passed prose" when it was actually handed a bare one-liner.
    found = box_chars(reply)
    assert len(found) >= 100, (
        f"expected a heavily decorated reply, got {len(found)} box "
        f"characters in {len(reply)} chars"
    )
    assert len(reply) >= 300, (
        f"reply is {len(reply)} chars, under DEGENERATE_MIN_CHARS (300) — "
        f"the structural rules would not even be consulted, so this test "
        f"would prove nothing"
    )

    indexed = H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=60)
    assert indexed >= 1, (
        "a decorated prose reply was NOT stored. If /health/full's "
        "memory_tail shows skipped_degenerate moving, the degeneracy "
        "detector has regressed toward the false positives R9/R19/R24/R25 "
        "were fixed for — a reply refused here is redacted from every "
        "future summary, permanently, not merely unremembered."
    )


# ---------------------------------------------------------------------------
# N1 / R3 — the empty-assistant repair
# ---------------------------------------------------------------------------


# Every emptiness shape `main.assistant_content_is_empty` covers, plus the
# one it must refuse to call empty. Named, because a parametrize id of
# "content4" in a failure report tells nobody which shape broke.
EMPTY_SHAPES = [
    ("empty-string", ""),
    ("whitespace-only", "   \n  "),
    ("none", None),
    ("empty-list", []),
    ("blank-text-list", [{"type": "text", "text": "   "}]),
]


@pytest.mark.parametrize("name,content", EMPTY_SHAPES, ids=[s[0] for s in EMPTY_SHAPES])
def test_interior_empty_assistant_turn_completes(conv_id, name, content):
    """N1: an INTERIOR assistant turn with empty content made vLLM's mistral
    template reject the whole request — "Invalid assistant message:
    role='assistant' content=''" — 102 times in two days, each one a dead
    turn with no reply, no memory write and an HTTP 200 already committed.
    The repair space-fills (never drops: dropping breaks user/assistant
    alternation) in the same pass as the tail repair.

    Every shape the shared predicate covers is driven here, interior — one
    complete exchange before it and a live user message after it, which is
    the exact production shape: a cancelled stream leaves an empty assistant
    turn, then she types again and it becomes interior.

    WHAT WOULD MAKE THIS FAIL: the repair raising, dropping the turn in a
    way that breaks the payload, or the exchange failing to reach memory.

    WHAT THIS STACK CANNOT PROVE, said plainly because a green here is
    weaker than it looks: the fixture models three of vLLM's template
    refusals (`_template_refusal`) and the empty-assistant-content one is
    not among them, and it only applies them on /tokenize. So an
    unrepaired payload would ALSO return 200 here. This case pins that the
    repair does not itself break the request or the memory tail across all
    six shapes; the refusal it exists for is pinned by
    compactor/test_payload_tail.py against vLLM's own template stack.
    """
    H.skip_if_no_admin("the memory assertion needs the admin API")

    assistant_turn = {"role": "assistant"}
    if name != "missing-key":
        assistant_turn["content"] = content

    prior = [
        {"role": "user", "content": "First question about the trip."},
        {"role": "assistant", "content": "First answer, complete and ordinary."},
        {"role": "user", "content": "Second question, the one whose reply was cut."},
        assistant_turn,
    ]
    status, reply = chat_slow(
        "Third question, typed after the cancelled reply.",
        conv_id=conv_id, prior_turns=prior, max_tokens=64,
    )
    assert status == 200, (
        f"an interior empty assistant turn ({name}) produced HTTP {status}. "
        f"This is the N1 shape: the request must survive it."
    )
    assert reply.strip(), (
        f"({name}) HTTP 200 with an empty reply — the 'dead turn with a "
        f"committed 200' that made this defect invisible in production"
    )
    assert H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=60) >= 1, (
        f"({name}) the exchange completed but never reached memory"
    )


def test_missing_content_key_completes(conv_id):
    """The sixth shape: an assistant turn with no `content` key at all.

    Split out rather than parametrized because it is a different JSON shape
    (a key absent, not a value present) and building it inside the
    parametrize would need a sentinel that reads worse than a second test.
    Same guarantee, same reason — see the docstring above.
    """
    H.skip_if_no_admin("the memory assertion needs the admin API")

    prior = [
        {"role": "user", "content": "First question about the trip."},
        {"role": "assistant", "content": "First answer, complete and ordinary."},
        {"role": "user", "content": "Second question, the one whose reply was cut."},
        {"role": "assistant"},
    ]
    status, reply = chat_slow(
        "Third question, typed after the cancelled reply.",
        conv_id=conv_id, prior_turns=prior, max_tokens=64,
    )
    assert status == 200, f"a content-less assistant turn produced HTTP {status}"
    assert reply.strip(), "HTTP 200 with an empty reply"
    assert H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=60) >= 1


def test_assistant_turn_with_a_non_text_part_is_not_empty(conv_id):
    """R3's other half: a list with ANY non-text part is never empty.

    `_repair_template_invalid_tail` tested emptiness two ways in one
    function — the pop used `_message_text().strip()`, which joins text
    parts and ignores images, so an assistant turn carrying only an image
    read as empty and was POPPED, destroying the image, three lines before
    the fill carefully refused to touch that shape. The two rules are now
    one predicate.

    WHAT WOULD MAKE THIS FAIL: the request erroring, or the exchange not
    reaching memory. A silent pop is not directly observable from outside —
    the fixture accepts a non-alternating message list (measured: two
    consecutive user turns return 200), so the corruption a pop causes
    cannot be seen here. The mutation coverage for the predicate itself
    lives in compactor/test_payload_tail.py; this case is the end-to-end
    guarantee that the shape does not blow up the request path.
    """
    H.skip_if_no_admin("the memory assertion needs the admin API")

    prior = [
        {"role": "user", "content": "Here is a picture, what do you think?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                }
            ],
        },
        {"role": "user", "content": "And the second one?"},
        {"role": "assistant", "content": "The second one is brighter."},
    ]
    status, reply = chat_slow(
        "What did you make of the first picture?",
        conv_id=conv_id, prior_turns=prior, max_tokens=64,
    )
    assert status == 200, (
        f"an assistant turn holding only an image produced HTTP {status}"
    )
    assert reply.strip(), "HTTP 200 with an empty reply"
    assert H.wait_for_indexed_exchanges(conv_id, min_count=1, max_wait=60) >= 1, (
        "the exchange completed but never reached memory"
    )
