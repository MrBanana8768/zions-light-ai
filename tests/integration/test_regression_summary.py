"""
Tier-3 regression net for the SUMMARIZER-POSITION and ADMIN-ENDPOINT defects
fixed in v3.1.7. One case (or one small group) per defect, all black box.

WHAT EACH CASE PINS
-------------------
  R23  test_r23_bounded_window_position_matches_the_client_turn_count
       A client turn was silently lost the moment the `max_turns` valve
       engaged: `n > prev` was read as "the array length IS the position",
       true only for an unbounded window, so at the first request where the
       cap bit the position came out one turn short — permanently — and the
       chunk labelled 101-120 held turns 102-121.

  R12  test_r12_a_watermark_below_the_chunks_does_not_discard_new_spans
       A watermark pulled down to the cap by the pre-v3.1.4 code made every
       new L1 chunk collide with an existing label and be discarded, silently,
       for ~280 exchanges. Pinned as COVERAGE: every turn the watermark has
       passed is covered by a real, non-empty summary entry, and the new span
       starts where the seeded ones stop.

  R13  test_r13_a_gap_in_the_episodic_index_can_be_compacted
       test_r13_a_gapless_store_needs_no_placeholders            (control)
       test_r13_a_store_short_of_the_position_still_refuses      (refusal kept)
       test_r13_more_placeholder_than_transcript_still_refuses   (refusal kept)
       `/admin/conversations/<id>/compact` 409'd whenever any exchange had
       ever missed the episodic index — which is every real conversation —
       on the one rebuild-from-store recovery path there is. It now rebuilds
       BY SLOT from `turn_index` and fills the holes with placeholders.

  R4   test_r4_query_string_dry_run_false_commits
       test_r4_body_dry_run_false_commits
       test_r4_default_is_a_dry_run_that_changes_nothing         (control)
       test_r4_an_explicit_body_overrides_the_query_string
       `?dry_run=false` — the form the install runbook documents — was read
       from the JSON body only, so the documented "commit the merge" command
       was a second dry run: HTTP 200, plausible counts, nothing changed.

  R10  test_r10_equality_is_admitted_and_the_labels_start_at_turn_one
       The `len(messages) < recorded_position` guard is CORRECT and stays `<`
       — equality is the healthy case and means the rebuild aligns exactly.
       The defect was one layer downstream: the chat path's anchor, measured
       against a rebuild it does not belong to, pushed the position to n + 2
       so chunk 1-20 came back labelled 3-20 holding turns 1-18.

  R5   test_r5_a_pinned_fact_survives_a_merge_still_pinned
       test_r5_a_pinned_fact_survives_a_retire_still_pinned
       `merge_conversation` unioned facts with the destination winning
       outright and no pin union, so a pinned source fact merged into a
       conversation holding an unpinned copy came out unpinned — on the
       documented install runbook step. `/retire` step 3 had the same defect.

HOW ALIGNMENT IS ASSERTED HERE, AND WHY IT IS NOT ASSERTED ON CHUNK TEXT
-----------------------------------------------------------------------
The whole nature of R23 is that chunk LABELS look right whether or not the
fix works, so a suite that reads labels back pins nothing. The load-bearing
property is what a chunk CONTAINS against what its label claims.

On this stack that text cannot be read. The fixture has no model weights, so
every chunk's `text` is the same canned string, and no endpoint exposes the
summarization prompt either — the fixture's `/_fixture/shapes` records
`/tokenize` request SHAPES (flags, roles, counts) and never content. There is
no route to the bytes.

What IS observable is the entire mapping from a label to the text it reads.
`summarizer.maybe_rollup` computes, once per call and constant for the drain:

    window_offset = turns_seen - (non-system turns in the array)
    chunk text    = array[first_turn - window_offset .. last_turn - window_offset]

Both inputs are exactly known here: the array length is what the TEST sent
(the test is the client), and `turns_seen` is published verbatim by
GET /admin/conversations/<id>/summary. So

    assert turns_seen == <the number of turns this test actually sent>

is an assertion about the offset, and therefore about which turns every label
maps onto. It is not an assertion about the label. It is also precisely the
number R23 got wrong: 101 against a truth of 102, one short, forever.

Every case that touches position therefore asserts the position exactly, and
carries a companion assertion that makes the exactness non-trivial — see each
case's "CONTROL" comment for what would make it fail.

WHAT THE FIXTURE MEANS FOR ASSERTIONS
-------------------------------------
Every completion is canned, so nothing here asserts on summary PROSE — only
on structure: chunk boundaries, coverage, counts, positions, refusals, and
the pin flag. The one thing asserted about text is that it is non-empty,
which is the difference between a recorded span and a silently empty one.

These tests drive tens of chat turns each and are NOT marked `slow`: the
whole point of the file is that it runs as a gate. Against the local
integration stack each turn is a canned reply and costs milliseconds.
"""

from __future__ import annotations

import re
import time

import httpx
import pytest

import _harness as H


# ---------------------------------------------------------------------------
# Local helpers. Everything this file needs that _harness does not already
# expose lives HERE — _harness.py and conftest.py are shared with other work
# in flight and are not mine to touch.
# ---------------------------------------------------------------------------

# The compact endpoint drains rollups in a loop and the merge endpoint
# re-embeds exchanges; both are slower than a chat turn even against a canned
# fixture, so they get their own ceiling rather than the 30s the stack sets
# for chat.
_ADMIN_TIMEOUT = max(H.TIMEOUT, 300.0)

# What the summarizer's defaults make a chunk. Read from the source of truth
# (compactor/summarizer.py L1_CHUNK_SIZE) at the time of writing; the stack
# sets no COMPACTOR_L1_CHUNK_SIZE override, and _assert_tiled_coverage below
# fails loudly (rather than skipping) if the deployment disagrees.
L1_CHUNK = 20

# portability.BUNDLE_VERSION. _validate_bundle is a strict equality check, so
# a mismatch here surfaces as a 400 with the expected/actual pair in it — a
# loud failure, not a silent skip.
BUNDLE_VERSION = "v2.1"


def _admin_client() -> httpx.Client:
    assert H.ADMIN_URL, "admin URL required — call H.skip_if_no_admin() first"
    return httpx.Client(base_url=H.ADMIN_URL, timeout=_ADMIN_TIMEOUT)


def admin_merge(
    src: str,
    dst: str,
    *,
    query: dict | None = None,
    body: dict | None = None,
    send_body: bool = True,
) -> tuple[int, dict]:
    """POST /admin/conversations/<src>/merge-into/<dst> → (status, json).

    `send_body=False` sends no request body AT ALL, which is what an operator
    typing the runbook's curl actually produces and the shape R4 is about —
    passing `json={}` instead would still hand the handler a parseable body
    and quietly change the case under test.
    """
    with _admin_client() as c:
        r = c.post(
            f"/admin/conversations/{src}/merge-into/{dst}",
            params=query or {},
            **({"json": body if body is not None else {}} if send_body else {}),
        )
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {}


def admin_compact(
    conv_id: str, *, dry_run: bool = False, max_calls: int = 200
) -> tuple[int, dict]:
    """POST /admin/conversations/<id>/compact → (status, json).

    Status is part of the contract here — 409 is the refusal two of these
    cases exist to keep — so it is returned rather than raised on.
    """
    with _admin_client() as c:
        r = c.post(
            f"/admin/conversations/{conv_id}/compact",
            json={"dry_run": dry_run, "max_calls": max_calls},
        )
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {}


def _empty_summary_state(conv_id: str, **overrides) -> dict:
    """A well-formed summary state to hand to /admin/conversations/import.

    Mirrors summarizer._empty_state's shape exactly. Written out here rather
    than exported from anywhere because this suite imports no compactor code:
    if the on-disk shape ever changes, this file is supposed to notice by
    failing, not by silently importing the new one.
    """
    state = {
        "conv_id": conv_id,
        "l1": [],
        "l2": [],
        "l3": None,
        "last_summarized_turn": 0,
        "turns_seen": 0,
        "tail_fp": [],
        "head_fp": "",
        "window_turns": 0,
    }
    state.update(overrides)
    return state


def _seed_conversation(
    conv_id: str,
    *,
    facts: list[dict] | None = None,
    summary_state: dict | None = None,
    episodic: list[dict] | None = None,
) -> None:
    """Put a conversation into an exact known state through the PUBLIC import
    endpoint, and fail loudly if it did not land.

    This is how the upgrade-path shapes get built. R12's state file (chunks
    to turn 660, watermark pulled down to the cap) and R13's gapped episodic
    store are both artefacts of history that a live conversation cannot be
    driven into from a clean stack in any reasonable time — but they are
    exactly what the endpoints under test have to survive, and the import
    endpoint is a supported public route to them.

    `overwrite=True` unconditionally: on a conversation that is genuinely
    empty the flag is a no-op, and on one where the vector store could not be
    probed the import would otherwise refuse (portability.import_conversation
    treats "cannot verify" as occupied). A refusal here would look exactly
    like a test that skipped its own setup.
    """
    bundle = {
        "version": BUNDLE_VERSION,
        "exported_at": int(time.time()),
        "source_conv_id": conv_id,
        "facts": facts or [],
        "summary_state": summary_state or _empty_summary_state(conv_id),
        "episodic": episodic or [],
    }
    status, body = H.admin_import(bundle, target_conv_id=conv_id, overwrite=True)
    assert status == 200, f"seeding {conv_id} failed: HTTP {status} {body}"
    imported = body.get("imported") or {}
    # The count is checked, not just the status: import_conversation skips an
    # episodic entry it cannot embed and reports 200 regardless, and a case
    # whose store silently came up short would refuse or pass for the wrong
    # reason.
    assert imported.get("episodic") == len(episodic or []), (
        f"seeding {conv_id}: expected {len(episodic or [])} episodic rows to "
        f"import, endpoint reported {imported.get('episodic')} — {body}"
    )


def _chunks(state: dict) -> list[dict]:
    """Every stored L1 + L2 chunk. L3 is deliberately excluded: it is a
    whole-conversation refresh that overlaps everything below it, so folding
    it into a tiling check would make the check meaningless. Callers assert
    l3 is None so its arrival fails the test rather than skewing it."""
    return list(state.get("l1") or []) + list(state.get("l2") or [])


def _assert_tiled_coverage(state: dict, *, through: int, what: str) -> None:
    """Every turn from 1 to `through` is covered by exactly one non-empty
    chunk, the spans are contiguous, and no label is used twice.

    This is R12's shape of assertion (compare compactor/test_saturation.py's
    reconciliation): a silently discarded chunk shows up as a HOLE between two
    spans, or as a run of turns past the last span that the watermark claims
    to have summarized.
    """
    assert state.get("l3") is None, (
        f"{what}: an L3 chapter appeared; this case's coverage arithmetic "
        f"assumes it has not, so it is a failure and not a skip. state={state}"
    )
    spans = sorted(
        (c["first_turn"], c["last_turn"], c.get("text") or "")
        for c in _chunks(state)
    )
    assert spans, f"{what}: no summary chunks at all (state={state})"
    assert spans[0][0] == 1, (
        f"{what}: coverage starts at turn {spans[0][0]}, not turn 1 — turns "
        f"1-{spans[0][0] - 1} are covered by nothing. spans={[s[:2] for s in spans]}"
    )
    prev_last = 0
    for first, last, text in spans:
        assert first == prev_last + 1, (
            f"{what}: chunk {first}-{last} does not continue from turn "
            f"{prev_last} — {'a gap' if first > prev_last + 1 else 'an overlap'} "
            f"in the coverage. spans={[s[:2] for s in spans]}"
        )
        assert last >= first, f"{what}: inverted span {first}-{last}"
        assert text.strip(), (
            f"{what}: chunk {first}-{last} has empty text — a span recorded "
            f"with nothing under it is the failure, not the fix"
        )
        prev_last = last
    assert prev_last == through, (
        f"{what}: coverage stops at turn {prev_last} but the conversation is "
        f"summarized through turn {through} — turns {prev_last + 1}-{through} "
        f"are covered by nothing. spans={[s[:2] for s in spans]}"
    )


def _wait_for_position(
    conv_id: str, *, at_least: int, max_wait: float = 8.0
) -> int:
    """Poll the summary state until `turns_seen` reaches `at_least`.

    The rollup runs in main.py's post-response tail, so the state for turn N
    can still be in flight when turn N+1 is sent. Waiting per turn keeps the
    position arithmetic measured on settled state instead of on a race.

    Returns whatever it last saw — the CALLER asserts. A helper that asserted
    here would turn "the position never arrived" into an error inside a
    fixture rather than a failure on the case that cares.
    """
    deadline = time.monotonic() + max_wait
    seen = -1
    while True:
        try:
            seen = int(H.admin_get_summary(conv_id).get("turns_seen") or 0)
        except Exception:
            seen = -1
        if seen >= at_least or time.monotonic() >= deadline:
            return seen
        time.sleep(0.25)


def _drive(
    conv_id: str,
    exchanges: int,
    *,
    cap: int | None = None,
    tag: str = "t",
    history: list[dict] | None = None,
) -> list[dict]:
    """Send `exchanges` real chat turns and return the FULL history.

    `cap` simulates OpenWebUI's `max_turns` valve: the client keeps the whole
    conversation but sends only the last `cap` turns of it. That is the shape
    v3.1.4 exists for and the shape R23 broke — and note it is applied AFTER
    appending the new user turn, exactly as the valve does, so on the request
    where the cap first bites the window is a strict suffix that no longer
    starts at turn 1.

    Every user turn is distinct so that _turn_fingerprints produces a unique
    anchor per turn. If they repeated, the alignment would land in the
    ambiguous-tail branch (R18) and this file would be measuring that instead.
    """
    hist = list(history or [])
    for i in range(exchanges):
        n_prior = len(hist)
        user = (
            f"{tag}-{n_prior // 2 + 1}: reply with the word "
            f"ok-{tag}-{n_prior // 2 + 1} and nothing else."
        )
        window = hist + [{"role": "user", "content": user}]
        if cap is not None:
            window = window[-cap:]
        r = H.chat(user, conv_id=conv_id, prior_turns=window[:-1], max_tokens=16)
        assert r.status_code == 200, (
            f"chat turn {n_prior // 2 + 1} returned {r.status_code}: {r.raw}"
        )
        assert r.response_text.strip(), (
            f"chat turn {n_prior // 2 + 1} returned an empty reply; the "
            f"memory tail skips a whitespace reply, so this case would be "
            f"measuring a conversation it did not drive"
        )
        hist = H.extend_history(hist, user, r.response_text)
        # Settle the tail before the next turn (see _wait_for_position).
        _wait_for_position(conv_id, at_least=len(hist), max_wait=8.0)
    return hist


def _fresh(request) -> str:
    """A sentinel conv_id cleaned up at teardown. The conftest `conv_id`
    fixture gives one per test; several cases here need two or three, and a
    merge/retire case is not a test of anything if its second conversation
    leaks into the next run's store."""
    cid = H.fresh_conv_id()
    request.addfinalizer(lambda: H.admin_safe_forget(cid))
    return cid


@pytest.fixture
def extra_conv(request):
    """Factory for additional cleaned-up conv_ids."""
    return lambda: _fresh(request)


# ---------------------------------------------------------------------------
# R23 — a client turn silently lost the moment the max_turns valve engages
# ---------------------------------------------------------------------------

def test_r23_bounded_window_position_matches_the_client_turn_count(conv_id):
    """With a BOUNDED window the recorded position must equal the number of
    turns the client actually sent, and the chunks must tile it.

    THE SHAPE, and why the cap is even. The loss needs `prev < n < truth`:
    the array is longer than the last recorded position (so the old
    `n > prev` branch fires and reads the length AS the position) while
    already being SHORTER than the truth (because the valve trimmed it). With
    an even cap C the client's pre-trim array at exchange e is 2e-1 turns, so
    on the first exchange where 2e-1 > C the valve removes an ODD number of
    turns and lands the array exactly one short — R23's cap-100 exchange-51
    arithmetic, at cap 20 and exchange 11. An odd cap trims by an even amount
    and never reproduces it, which is why both documented cap values (100 and
    60) are even and this one is too.

    WHAT WOULD MAKE THIS FAIL. Restore `if n > prev: position = n` without
    consulting the anchor and `turns_seen` comes back 59 against 60 — one
    turn swallowed at exchange 11 and never repaid. Every later chunk then
    reads its text one turn early, which is the harm; the labels stay
    plausible, which is why nothing downstream can tell.

    CONTROL, so the exactness is not trivially satisfied: the last assertion
    checks the position EXCEEDS the window the client sent. If the cap had not
    bitten — if this were an unbounded client — the position would equal the
    array length and `turns_seen == 2 * exchanges` would be true for a reason
    that has nothing to do with the fix.
    """
    H.skip_if_no_admin("summary state inspection requires admin endpoint")

    cap = 20
    exchanges = 30
    expected_position = 2 * exchanges          # the client sent 60 turns

    _drive(conv_id, exchanges, cap=cap, tag="r23")
    settled = _wait_for_position(
        conv_id, at_least=expected_position, max_wait=30.0
    )
    state = H.admin_get_summary(conv_id)

    assert settled == expected_position, (
        f"the client sent {expected_position} turns under a {cap}-turn cap, "
        f"but the conversation's recorded position is {settled}. A position "
        f"short by one is R23: the window trimmed by an odd number of turns "
        f"on the exchange where the cap first bit, and the array length was "
        f"read as the position. Every chunk below then reads its text "
        f"{expected_position - settled} turn(s) early while its label says "
        f"otherwise. state={state}"
    )

    # CONTROL. The window really was bounded, so the position above could only
    # have come from the compactor's own counter — not from the array length.
    window_turns = int(state.get("window_turns") or 0)
    assert 0 < window_turns < settled, (
        f"the cap never engaged: the last window was {window_turns} turns "
        f"against a position of {settled}, so this ran as an UNBOUNDED client "
        f"and proves nothing about R23. state={state}"
    )

    # And the hierarchy that position drives: three chunks tiling 1-60, each
    # with something under it.
    watermark = int(state.get("last_summarized_turn") or 0)
    assert watermark == (expected_position // L1_CHUNK) * L1_CHUNK, (
        f"expected the watermark at turn "
        f"{(expected_position // L1_CHUNK) * L1_CHUNK} after "
        f"{expected_position} turns, got {watermark}. state={state}"
    )
    _assert_tiled_coverage(state, through=watermark, what="R23 bounded window")


# ---------------------------------------------------------------------------
# R12 — a pulled-down watermark made the duplicate-label guard discard every
#       new L1 chunk, silently
# ---------------------------------------------------------------------------

def test_r12_a_watermark_below_the_chunks_does_not_discard_new_spans(conv_id):
    """The upgrade shape, exactly: chunks labelled well past the cap, with the
    watermark pulled DOWN to the cap by the pre-v3.1.4 `_reconcile_watermark`.
    Driving a capped client on top of that must keep COVERING new turns.

    R12's own reproduction is the measurement this mirrors: 33 old chunks to
    turn 660, watermark 100, capped 100-turn client, 30 new exchanges →
    `new L1 chunks = 0`, stuck for ~280 exchanges. Scaled down here to 8
    chunks reaching turn 160 with the watermark at 100 and a ~100-turn window,
    which is the same relationship between the three numbers (the labels run
    past the cap; the pointer sits below them; the client's window is shorter
    than both) and keeps the L1 count under L2_CHUNK_SIZE so no L2 rollup
    reshapes the tiers mid-case.

    WHAT WOULD MAKE THIS FAIL. Seed `_observed_position` from
    `max(turns_seen, last_summarized_turn)` alone — i.e. drop the chunk-label
    term from `_recorded_position` and the `_repair_watermark_below_chunks`
    call — and the position restarts at 100, the next chunk is labelled
    101-120, that label already exists, the duplicate guard advances the
    watermark and stores nothing, and coverage stops dead at turn 160 while
    the watermark walks up through spans nothing new was written for. The
    coverage assertion below reports it as a hole; the count assertion
    reports it as zero new chunks, which is what R12 measured.

    CONTROL: the pre-drive assertions establish that the seeded state really
    is the R12 shape (watermark strictly BELOW the highest label). Without
    that, a green result would only prove that a healthy conversation stays
    healthy.
    """
    H.skip_if_no_admin("summary state inspection requires admin endpoint")

    seeded_chunks = 8                      # labels 1-20 .. 141-160
    highest_seeded = seeded_chunks * L1_CHUNK
    pulled_down_watermark = 100            # a documented cap value
    window = 99                            # client turns per request; +1 reply

    l1 = [
        {
            "text": f"(seeded L1 chunk covering turns {i * L1_CHUNK + 1}"
                    f"-{(i + 1) * L1_CHUNK})",
            "first_turn": i * L1_CHUNK + 1,
            "last_turn": (i + 1) * L1_CHUNK,
        }
        for i in range(seeded_chunks)
    ]
    _seed_conversation(
        conv_id,
        summary_state=_empty_summary_state(
            conv_id,
            l1=l1,
            # The pointer the old _reconcile_watermark left behind, below the
            # chunks it had already written. `turns_seen` absent (0) because
            # the field did not exist when such a file was written.
            last_summarized_turn=pulled_down_watermark,
            turns_seen=0,
        ),
    )

    before = H.admin_get_summary(conv_id)
    assert len(before.get("l1") or []) == seeded_chunks, (
        f"seed did not land: {before}"
    )
    assert int(before["last_summarized_turn"]) < highest_seeded, (
        f"seed is not the R12 shape — the watermark "
        f"({before['last_summarized_turn']}) must sit BELOW the highest chunk "
        f"label ({highest_seeded}) for this case to mean anything"
    )

    # A fabricated history the client "already has". Its content is never
    # asserted on; it exists so the window the client sends is a genuine
    # bounded SUFFIX (n < the recorded position), which is the state R12's
    # duplicate guard fired in. Distinct per turn for the anchor's sake.
    history: list[dict] = []
    for i in range(49):
        history += [
            {"role": "user", "content": f"prior exchange {i}: what comes next?"},
            {"role": "assistant", "content": f"prior exchange {i}: the answer."},
        ]

    # Enough exchanges to walk the position from 160 past 180, i.e. past the
    # first new chunk boundary. Two turns per exchange, plus slack.
    exchanges = 14
    for e in range(exchanges):
        user = f"r12-{e}: reply with the word ok-r12-{e} and nothing else."
        win = (history + [{"role": "user", "content": user}])[-window:]
        r = H.chat(user, conv_id=conv_id, prior_turns=win[:-1], max_tokens=16)
        assert r.status_code == 200, f"r12 turn {e} → {r.status_code}: {r.raw}"
        history = H.extend_history(history, user, r.response_text)
        _wait_for_position(
            conv_id, at_least=highest_seeded + 2 * (e + 1), max_wait=8.0
        )

    after = H.admin_get_summary(conv_id)
    chunks_after = _chunks(after)

    assert len(chunks_after) > seeded_chunks, (
        f"no new summary span was stored in {exchanges} exchanges — "
        f"{len(chunks_after)} chunks before and after. This is R12 verbatim: "
        f"the new chunk collided with an existing label and was discarded, "
        f"and the only sign is that coverage stopped moving. state={after}"
    )

    new_spans = sorted(
        (c["first_turn"], c["last_turn"])
        for c in chunks_after
        if c["first_turn"] > highest_seeded
    )
    assert new_spans, (
        f"the chunk count grew but nothing covers a turn past "
        f"{highest_seeded}: the new spans were labelled over the seeded ones. "
        f"spans={sorted((c['first_turn'], c['last_turn']) for c in chunks_after)}"
    )
    assert new_spans[0] == (highest_seeded + 1, highest_seeded + L1_CHUNK), (
        f"the first new span is {new_spans[0]}, not "
        f"{(highest_seeded + 1, highest_seeded + L1_CHUNK)} — it does not "
        f"continue from the seeded chunks, so some turns between them are "
        f"covered by nothing"
    )

    _assert_tiled_coverage(
        after,
        through=int(after["last_summarized_turn"]),
        what="R12 upgrade path",
    )


# ---------------------------------------------------------------------------
# R13 — /admin/compact refused for every real conversation
# ---------------------------------------------------------------------------

def _indexed_rows(conv_id: str, exchanges: int) -> list[dict]:
    """Drive a full-history conversation and return its episodic rows.

    Full history (no cap) on purpose: it makes the recorded position exactly
    2 * exchanges and the turn_index spacing exactly one step per exchange,
    which is what the rebuild-by-slot cases need to be arithmetic rather than
    guesswork.
    """
    _drive(conv_id, exchanges, tag="src")
    got = H.wait_for_indexed_exchanges(conv_id, min_count=exchanges, max_wait=60.0)
    assert got == exchanges, (
        f"prep: expected {exchanges} indexed exchanges for {conv_id}, got "
        f"{got}. Every R13 case is about a store with a KNOWN shape, so an "
        f"unexpected one is a failure and not something to work around."
    )
    bundle = H.admin_export(conv_id)
    rows = sorted(
        bundle.get("episodic") or [], key=lambda e: int(e.get("turn_index") or 0)
    )
    assert len(rows) == exchanges, f"prep: export returned {len(rows)} rows"
    return rows


def test_r13_a_gap_in_the_episodic_index_can_be_compacted(conv_id, extra_conv):
    """One exchange missing from the episodic index must no longer refuse.

    That is the whole of R13: `turns_seen` counts every exchange including the
    ones the memory tail skipped and the pool shed, the episodic store holds
    only the indexed ones, and the old guard compared a CONCATENATION against
    the position — so a single gap anywhere 409'd, which is every real
    conversation (63 skips in one measured window), on the one
    rebuild-from-store recovery path there is.

    The gapped store is built by exporting a healthy conversation, dropping
    one MIDDLE row, and importing the rest — `retrieval.import_indexed_exchange`
    keeps the bundle's `turn_index` verbatim, so the hole lands where the
    dropped exchange was. A middle row, not the first: the rebuild anchors on
    the LOWEST stored index, so dropping the head moves the anchor instead of
    opening a gap, and would be testing the other refusal.

    WHAT WOULD MAKE THIS FAIL. Rebuild by concatenation instead of by slot —
    the pre-R13 behaviour — and the array is 22 messages against a recorded
    position of 24, `len(messages) < _pos` fires, and this comes back 409.
    """
    H.skip_if_no_admin("admin compact requires admin endpoint")

    exchanges = 12
    rows = _indexed_rows(conv_id, exchanges)

    target = extra_conv()
    gapped = [r for i, r in enumerate(rows) if i != 5]   # drop exchange 6
    _seed_conversation(
        target,
        summary_state=_empty_summary_state(target, turns_seen=2 * exchanges),
        episodic=gapped,
    )

    status, plan = admin_compact(target, dry_run=True)
    assert status == 200, (
        f"a single gap in the episodic index still refuses the compaction "
        f"that exists to repair it — HTTP {status}: {plan}"
    )
    assert plan["recorded_position"] == 2 * exchanges, plan
    assert plan["gap_exchanges"] == 1, (
        f"expected exactly one placeholder exchange for the row that was "
        f"dropped, plan reports {plan['gap_exchanges']}: {plan}"
    )
    assert plan["gap_turns"] == 2, plan
    assert plan["reconstructed_messages"] == 2 * exchanges, (
        f"the rebuild is {plan['reconstructed_messages']} messages against a "
        f"position of {2 * exchanges} — filling the hole in place is the "
        f"whole fix, and it did not restore the length: {plan}"
    )

    # And it actually runs, laying the chunks down from turn 1.
    status, done = admin_compact(target, dry_run=False)
    assert status == 200, f"live compaction refused: HTTP {status}: {done}"
    state = H.admin_get_summary(target)
    _assert_tiled_coverage(
        state,
        through=int(state["last_summarized_turn"]),
        what="R13 gapped rebuild",
    )
    assert int(state["last_summarized_turn"]) == L1_CHUNK, (
        f"expected one L1 chunk covering turns 1-{L1_CHUNK} out of "
        f"{2 * exchanges} rebuilt turns, watermark is "
        f"{state['last_summarized_turn']}: {state}"
    )


def test_r13_a_gapless_store_needs_no_placeholders(conv_id, extra_conv):
    """CONTROL for the case above: the SAME rows with nothing dropped rebuild
    with zero placeholders.

    Without this, `gap_exchanges == 1` proves nothing — a rebuild that
    reported one placeholder for every conversation, or a `gap_turns` field
    hard-coded to 2, would satisfy the case above just as well. This pins the
    2 placeholder turns to the row that was removed and to nothing else.
    """
    H.skip_if_no_admin("admin compact requires admin endpoint")

    exchanges = 12
    rows = _indexed_rows(conv_id, exchanges)

    target = extra_conv()
    _seed_conversation(
        target,
        summary_state=_empty_summary_state(target, turns_seen=2 * exchanges),
        episodic=rows,
    )
    status, plan = admin_compact(target, dry_run=True)
    assert status == 200, f"HTTP {status}: {plan}"
    assert plan["gap_turns"] == 0, (
        f"a complete store still reports {plan['gap_turns']} placeholder "
        f"turns, so the gap count in the sibling case is not measuring the "
        f"missing row: {plan}"
    )
    assert plan["reconstructed_messages"] == 2 * exchanges, plan


def test_r13_a_store_short_of_the_position_still_refuses(conv_id, extra_conv):
    """The refusal that MUST survive: a store that does not REACH the
    recorded position.

    R13 narrowed when the 409 is right; it did not remove it. Placeholders
    restore alignment for holes INSIDE the store, but nothing invents a
    missing tail (or head) — running anyway would summarize text that is not
    the text the chunk labels claim, which is the failure the refusal's own
    comment calls worse than no chunk.

    WHAT WOULD MAKE THIS FAIL. Drop the guard, or weaken it to compare
    against `last_summarized_turn` instead of `_recorded_position`, and this
    comes back 200 having summarized 24 rebuilt turns under labels that
    belong to a 60-turn conversation.
    """
    H.skip_if_no_admin("admin compact requires admin endpoint")

    exchanges = 12
    rows = _indexed_rows(conv_id, exchanges)

    target = extra_conv()
    _seed_conversation(
        target,
        # The store reaches turn 24; the conversation is recorded at turn 60.
        summary_state=_empty_summary_state(target, turns_seen=60),
        episodic=rows,
    )
    status, body = admin_compact(target, dry_run=True)
    assert status == 409, (
        f"a rebuild 36 turns short of the recorded position was ADMITTED "
        f"(HTTP {status}) — every chunk it writes is labelled against turns "
        f"it does not contain: {body}"
    )
    detail = str(body.get("detail", ""))
    assert "recorded position" in detail and "60" in detail, (
        f"the refusal does not name the position it refused against, so an "
        f"operator cannot tell it from the placeholder refusal: {detail!r}"
    )


def test_r13_more_placeholder_than_transcript_still_refuses(conv_id, extra_conv):
    """The second refusal, and the only new one: a reconstruction that is more
    placeholder than transcript is not a transcript.

    Summarizing it would spend one vLLM call per chunk to record that nothing
    is known, advance the watermark past turns nothing will ever summarize,
    and store that as memory. It is also what bounds the rebuilt array against
    a single corrupt `turn_index`.

    Built by re-indexing three real rows far apart — 2, 20, 40 — so the
    rebuild is 40 turns of which 34 are placeholder. `turns_seen` is left at 0
    so the FIRST refusal cannot fire and mask this one.

    WHAT WOULD MAKE THIS FAIL. Remove the `gap_turns > _real_turns` check and
    this returns 200 with a plan that cheerfully reports 34 placeholder turns.
    """
    H.skip_if_no_admin("admin compact requires admin endpoint")

    rows = _indexed_rows(conv_id, 6)

    target = extra_conv()
    sparse = [
        {"turn_index": ti, "document": rows[i]["document"]}
        for i, ti in enumerate((2, 20, 40))
    ]
    _seed_conversation(
        target,
        summary_state=_empty_summary_state(target, turns_seen=0),
        episodic=sparse,
    )
    status, body = admin_compact(target, dry_run=True)
    assert status == 409, (
        f"a rebuild that is 34 placeholder turns against 6 recorded ones was "
        f"ADMITTED (HTTP {status}): {body}"
    )
    detail = str(body.get("detail", ""))
    assert "placeholder" in detail, (
        f"the refusal does not say it refused on placeholder density: {detail!r}"
    )


# ---------------------------------------------------------------------------
# R4 — ?dry_run=false was read from the JSON body only
# ---------------------------------------------------------------------------

def _seed_fact(conv_id: str, text: str) -> None:
    """Store one fact through the user-facing /remember command.

    Not through the import endpoint: this is the fact whose MOVEMENT the R4
    cases measure, and a fact written by the same endpoint family under test
    would make "it arrived" a weaker claim. /remember is also how the row gets
    an unpinned shape without depending on extraction, which this fixture
    cannot do (no model weights).
    """
    r = H.chat(f"/remember {text}", conv_id=conv_id, max_tokens=64)
    assert r.status_code == 200 and "Remembered" in r.response_text, (
        f"seeding a fact into {conv_id} failed: {r.status_code} "
        f"{r.response_text[:200]!r}"
    )


def _fact_texts(conv_id: str) -> list[str]:
    return [f["text"] for f in H.admin_get_facts(conv_id)]


def test_r4_query_string_dry_run_false_commits(conv_id, extra_conv):
    """`POST .../merge-into/<dst>?dry_run=false` with NO body must COMMIT.

    This is the form `pipelines/conversation_id_header.py` install step 8
    documents, and the step whose entire purpose is to un-fork her memory
    before the history cap goes on. `admin_merge` read `dry_run` from the JSON
    body only, so the documented command returned HTTP 200 with plausible
    counts and changed nothing — and capping before a real merge is what makes
    the loss permanent.

    WHAT WOULD MAKE THIS FAIL. Delete the `request.query_params` branch and
    this comes back `dry_run: true` with the destination still empty.
    """
    H.skip_if_no_admin("merge verification requires admin endpoint")

    src = conv_id
    dst = extra_conv()
    marker = "Marguerite keeps her grandmother's cello in the north room"
    _seed_fact(src, marker)

    # CONTROL: the destination does not already hold it, so its presence
    # afterwards can only have come from the merge.
    assert marker not in _fact_texts(dst), "control: dst must start without it"

    status, body = admin_merge(src, dst, query={"dry_run": "false"}, send_body=False)
    assert status == 200, f"HTTP {status}: {body}"
    assert body.get("dry_run") is False, (
        f"the query form was ignored — the endpoint still ran a dry run: {body}"
    )
    assert body.get("facts_added") == 1, (
        f"a committed merge reports what it WROTE (facts_added); this body "
        f"has only the preview counters, which is the R4 symptom: {body}"
    )
    assert marker in _fact_texts(dst), (
        f"HTTP 200, plausible counts, and nothing changed — the fact is not "
        f"in {dst}: {_fact_texts(dst)}"
    )


def test_r4_body_dry_run_false_commits(conv_id, extra_conv):
    """The JSON-body form must keep committing — the query form is an
    addition, not a replacement, and this is the form every existing caller
    and the endpoint's own docstring use.

    WHAT WOULD MAKE THIS FAIL. Any change that reads only the query string
    (the mirror-image regression) leaves the destination empty here.
    """
    H.skip_if_no_admin("merge verification requires admin endpoint")

    src = conv_id
    dst = extra_conv()
    marker = "The north room floorboard by the window creaks on the third step"
    _seed_fact(src, marker)
    assert marker not in _fact_texts(dst), "control: dst must start without it"

    status, body = admin_merge(src, dst, body={"dry_run": False})
    assert status == 200, f"HTTP {status}: {body}"
    assert body.get("dry_run") is False, body
    assert marker in _fact_texts(dst), (
        f"body form did not commit: {_fact_texts(dst)}"
    )


def test_r4_default_is_a_dry_run_that_changes_nothing(conv_id, extra_conv):
    """CONTROL for both cases above, and a contract in its own right: with no
    `dry_run` anywhere the endpoint previews and writes NOTHING.

    Without this the two commit cases prove only that the fact reached the
    destination somehow. With it, they prove the commit flag is what put it
    there — because the identical request minus the flag leaves the
    destination untouched.

    It also pins the safer default the endpoint deliberately keeps (the
    compact endpoint next door defaults the other way and surprised an
    operator into a live run).
    """
    H.skip_if_no_admin("merge verification requires admin endpoint")

    src = conv_id
    dst = extra_conv()
    marker = "The cello case has a brass plate reading 1911"
    _seed_fact(src, marker)

    status, body = admin_merge(src, dst, send_body=False)
    assert status == 200, f"HTTP {status}: {body}"
    assert body.get("dry_run") is True, f"default must be a dry run: {body}"
    assert body.get("facts_to_add") == 1, (
        f"the preview must still SAY what it would do: {body}"
    )
    assert "facts_added" not in body, (
        f"a dry run reported a write counter: {body}"
    )
    assert marker not in _fact_texts(dst), (
        f"a dry run modified the destination: {_fact_texts(dst)}"
    )

    # ?dry_run=true is the same answer by the other route.
    status, body = admin_merge(src, dst, query={"dry_run": "true"}, send_body=False)
    assert status == 200 and body.get("dry_run") is True, body
    assert marker not in _fact_texts(dst), _fact_texts(dst)


def test_r4_an_explicit_body_overrides_the_query_string(conv_id, extra_conv):
    """When both are present the BODY wins — an explicit JSON body is the more
    deliberate of the two, and the endpoint's docstring commits to it.

    Pinned because the safe direction matters: a `{"dry_run": true}` body sent
    to a URL that still carries `?dry_run=false` from a previous shell line
    must not commit.
    """
    H.skip_if_no_admin("merge verification requires admin endpoint")

    src = conv_id
    dst = extra_conv()
    marker = "The rosin tin lives in the pocket behind the neck"
    _seed_fact(src, marker)

    status, body = admin_merge(
        src, dst, query={"dry_run": "false"}, body={"dry_run": True}
    )
    assert status == 200, f"HTTP {status}: {body}"
    assert body.get("dry_run") is True, (
        f"the query string overrode an explicit body: {body}"
    )
    assert marker not in _fact_texts(dst), _fact_texts(dst)


# ---------------------------------------------------------------------------
# R10 — chunk labelling at the equality case
# ---------------------------------------------------------------------------

def test_r10_equality_is_admitted_and_the_labels_start_at_turn_one(
    conv_id, extra_conv
):
    """`len(messages) == recorded_position` is the HEALTHY case: it means the
    rebuild aligns exactly. It must be admitted, and the chunks it produces
    must be labelled against the text they actually contain.

    TWO THINGS ARE PINNED AND THEY ARE DIFFERENT.

    1. The guard stays `<`. Mutate it to `<=` and this case comes back 409 —
       and with it every one-gap rebuild R13 exists to admit, i.e. every real
       conversation. (V314_BACKLOG's re-derivation: that mutation fails 15
       assertions in the unit suite for the same reason.)

    2. The defect one layer downstream. `_observed_position` aligns the array
       against `tail_fp` — an anchor the CHAT path left behind from a bounded
       live window — and when it cannot align, falls back to
       `_ASSUMED_NEW_TURNS = 2`. At equality that makes the position n + 2,
       `window_offset` 2, and `_do_l1_rollup` reads chunk 1-20's text at array
       slots -1..18: it clamps, labels the chunk 3-20, fills it with turns
       1-18, leaves turns 1-2 covered by nothing and `turns_seen` inflated for
       the conversation's life. The fix drops the chat path's anchor under
       `conv_lock` before the drain, so the rebuild is measured on its own
       terms.

    The anchor is seeded EXPLICITLY with fingerprints that cannot match
    anything in the rebuilt transcript, because that is the state a bounded
    live window leaves behind and it is the only state in which the defect is
    reachable. Seeding it is also this case's CONTROL: the assertions before
    the compaction prove the anchor was really there, so a green result cannot
    come from an empty `tail_fp` that made the drop a no-op.

    WHAT WOULD MAKE THIS FAIL. Remove the anchor drop and the first chunk
    comes back labelled 3-20 with `turns_seen` at 26 instead of 24.
    """
    H.skip_if_no_admin("admin compact requires admin endpoint")

    exchanges = 12
    rows = _indexed_rows(conv_id, exchanges)
    position = 2 * exchanges

    target = extra_conv()
    # Four fingerprints of the right SHAPE (16 hex chars, as
    # _turn_fingerprints emits) and of content that appears nowhere in the
    # rebuilt transcript, so _align_candidates returns nothing and the
    # unalignable-anchor path is the one taken.
    stale_anchor = ["dead" + f"{i:012x}" for i in range(4)]
    _seed_conversation(
        target,
        summary_state=_empty_summary_state(
            target,
            turns_seen=position,          # equality: the rebuild is 24 turns
            tail_fp=stale_anchor,
            head_fp="beef000000000000",
            window_turns=21,              # what a bounded live window recorded
        ),
        episodic=rows,
    )

    # CONTROL: the state really carries a live anchor, so the drop below has
    # something to drop.
    seeded = H.admin_get_summary(target)
    assert seeded.get("tail_fp") == stale_anchor, (
        f"the stale anchor did not survive seeding, so this case cannot "
        f"reach the defect: {seeded}"
    )

    status, plan = admin_compact(target, dry_run=False)
    assert status == 200, (
        f"the equality case was REFUSED (HTTP {status}). Equality means the "
        f"rebuild aligns exactly — it is the case the endpoint exists to "
        f"serve, and refusing it refuses every real conversation: {plan}"
    )
    assert plan["reconstructed_messages"] == plan["recorded_position"] == position, (
        f"this case is only about equality if the two are equal: {plan}"
    )
    assert plan["gap_turns"] == 0, plan

    state = H.admin_get_summary(target)
    spans = sorted((c["first_turn"], c["last_turn"]) for c in _chunks(state))
    assert spans, f"the drain produced no chunk at all: {state}"
    assert spans[0] == (1, L1_CHUNK), (
        f"the first chunk is labelled {spans[0]}, not (1, {L1_CHUNK}). A "
        f"first_turn of 3 is R10: the rebuild was measured against the chat "
        f"path's anchor, the position came out {position} + 2, and the chunk "
        f"holds turns 1-18 under a label that claims 3-20 — with turns 1-2 "
        f"then covered by nothing. state={state}"
    )
    assert int(state["turns_seen"]) == position, (
        f"the position is {state['turns_seen']} against a rebuild of exactly "
        f"{position} turns. An inflated position is not a one-off: it stays "
        f"inflated for the conversation's life and shifts every later chunk. "
        f"state={state}"
    )
    _assert_tiled_coverage(
        state, through=int(state["last_summarized_turn"]), what="R10 equality"
    )

    # The drain replaced the borrowed anchor with the rebuild's own — the
    # other half of "measured on its own terms".
    assert state.get("tail_fp") != stale_anchor, (
        f"the chat path's anchor is still in place after the drain: {state}"
    )


# ---------------------------------------------------------------------------
# R5 — merge_conversation silently un-pinned facts
# ---------------------------------------------------------------------------

def _pinned_fact(text: str) -> dict:
    """A fact record in the on-disk shape, pinned.

    Seeded through the import endpoint rather than through `/pin`: see the
    note in the report — `pin`/`unpin` are absent from `commands._ALIASES`, so
    `parse_command` never resolves them and the chat command cannot reach
    `_handle_pin` at all. Import is the only route this suite has to a pinned
    row, and it is a public admin endpoint.
    """
    return {
        "text": text,
        "added_turn": 1,
        "last_used": int(time.time()),
        "pin": True,
    }


def _fact_named(conv_id: str, text: str) -> dict:
    matches = [f for f in H.admin_get_facts(conv_id) if f["text"] == text]
    assert len(matches) == 1, (
        f"expected exactly one fact reading {text!r} in {conv_id}, found "
        f"{len(matches)}: {H.admin_get_facts(conv_id)}"
    )
    return matches[0]


def test_r5_a_pinned_fact_survives_a_merge_still_pinned(conv_id, extra_conv):
    """A pinned SOURCE fact merged into a conversation holding an unpinned
    copy of the same text must come out PINNED.

    Facts are unioned on `_fact_key` (casefolded, whitespace-collapsed) with
    the destination's wording winning on collision. Until v3.1.7 the
    destination won outright, taking its unpinned flag with it — so following
    the project's own install runbook un-pinned identity facts, and a pinned
    fact is the only tier relevance ranking can never drop.

    WHAT WOULD MAKE THIS FAIL. Replace `_merge_fact_pin_and_recency` with
    "keep dst" and the destination's copy comes back `pin: false`.

    CONTROL: the destination's copy is asserted UNPINNED immediately before
    the merge. Without that, `pin is True` afterwards could just be the row
    the destination always had.
    """
    H.skip_if_no_admin("fact inspection requires admin endpoint")

    src = conv_id
    dst = extra_conv()
    marker = "Her name is Marguerite Vaillancourt and she was born in Trois-Rivieres"

    _seed_conversation(src, facts=[_pinned_fact(marker)])
    _seed_fact(dst, marker)                     # same text, unpinned

    assert _fact_named(src, marker)["pin"] is True, "seed: src fact must be pinned"
    assert _fact_named(dst, marker)["pin"] is False, (
        "control: the destination's copy must start UNPINNED, otherwise the "
        "assertion after the merge is satisfied by the seed"
    )

    status, body = admin_merge(src, dst, body={"dry_run": False})
    assert status == 200, f"HTTP {status}: {body}"
    assert body.get("dry_run") is False, body

    after = H.admin_get_facts(dst)
    assert len(after) == 1, (
        f"the collision was not recognised — the destination now holds "
        f"{len(after)} copies: {after}"
    )
    assert after[0]["pin"] is True, (
        f"the merge un-pinned an identity fact: {after[0]}. This is R5, and "
        f"it fires on the documented install path (step 8) that must run "
        f"before the history cap is enabled."
    )


def test_r5_a_pinned_fact_survives_a_retire_still_pinned(conv_id, extra_conv):
    """`/retire` step 3 — "already in the destination" — had the same defect:
    a pinned source row byte-identical to an unpinned destination row was
    dropped outright, taking its pin with it. The row still does not migrate
    (the text is genuinely redundant), but its pin must be folded into the
    destination's copy.

    Driven through the real two-step command surface, code and all, because
    the confirmation code is a hash of the plan and a test that skipped it
    would not be exercising the path an operator walks.

    WHAT WOULD MAKE THIS FAIL. Restore step 3's outright drop and the
    destination's copy is still `pin: false` after the apply.

    CONTROL: as above, the destination's copy is asserted unpinned first.
    """
    H.skip_if_no_admin("fact inspection requires admin endpoint")

    src = extra_conv()
    dst = conv_id
    marker = "She plays the cello left-handed and has since she was nine"

    _seed_conversation(src, facts=[_pinned_fact(marker)])
    _seed_fact(dst, marker)

    assert _fact_named(dst, marker)["pin"] is False, (
        "control: the destination's copy must start UNPINNED"
    )

    plan = H.chat(f"/retire {src}", conv_id=dst, max_tokens=64)
    assert plan.status_code == 200, plan.raw
    match = re.search(
        rf"/retire\s+{re.escape(src)}\s+apply\s+([0-9a-f]{{6,}})",
        plan.response_text,
    )
    assert match, (
        f"the dry run did not issue a confirmation code, so the apply below "
        f"cannot run and this case would pass without testing anything:\n"
        f"{plan.response_text[:800]}"
    )
    code = match.group(1)

    applied = H.chat(f"/retire {src} apply {code}", conv_id=dst, max_tokens=64)
    assert applied.status_code == 200, applied.raw
    assert "out of date" not in applied.response_text, (
        f"the plan changed between the dry run and the apply, so nothing "
        f"moved:\n{applied.response_text[:800]}"
    )

    after = H.admin_get_facts(dst)
    assert len(after) == 1, (
        f"the destination should still hold exactly one copy of the fact "
        f"(the source row is redundant by text): {after}"
    )
    assert after[0]["text"] == marker, after[0]
    assert after[0]["pin"] is True, (
        f"/retire dropped a pinned source row onto an unpinned destination "
        f"copy and lost the pin: {after[0]}"
    )
