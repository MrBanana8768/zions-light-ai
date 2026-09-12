"""Runtime coverage for the lazy V1-conversation backfill.

WHY THIS FILE EXISTS. `backfill.py` has a 642-line unit suite, and every one
of its model calls is a monkey-patched `httpx.AsyncClient`. So what is proven
today is the state machine — trigger conditions, stale detection, the merge —
against a stub that always answers. What was never exercised anywhere is the
thing the module is FOR: that a conversation which arrives with history and no
facts file ends up with facts extracted from that history, through a real
generation, against a running compactor.

That gap matters more than a normal one, because the backfill is invisible.
It runs as a background task, the request that triggered it returns without
facts by design, and nothing surfaces in /health/full. A backfill that starts,
produces nothing, and writes "complete" is indistinguishable from one that
worked until somebody reads the facts store.

WHAT CANNOT BE ASSERTED FROM HERE, stated so the absence is not mistaken for
coverage: the `facts/<id>.backfill.json` sidecar lives under the compactor's
storage root, inside its container, and no admin endpoint exposes it. These
tests therefore observe the backfill only through its effect on the facts
store. State transitions (in_progress -> complete, stale retry) remain
unit-tested only.

That limit is why the v3.1 F3 case is TWO tests rather than one. A refusal
has no signal of its own out here; what distinguishes a refusal from a
completed job is whether the seeded history's invented strings reached the
facts store, and only a real generation can put them there. So the
non-destructive half runs on every integration run and the refusal half asks
for weights — rather than one test asserting a subset relation that both
answers satisfy.
"""

from __future__ import annotations

import time

import _harness as H


# A V1-shaped history: several completed exchanges carrying content that the
# CURRENT message never mentions. That separation is the whole design of test
# [1] — the ordinary memory tail only ever sees the exchange it just handled,
# so a fact about this material can only have come from walking the history.
_SEED_HISTORY = [
    {"role": "user",
     "content": "Before we start, some background on the campaign setting. "
                "The capital city is called Quillhaven Reach, and it sits on "
                "a salt lake that freezes every autumn."},
    {"role": "assistant",
     "content": "Noted — Quillhaven Reach on a seasonally frozen salt lake."},
    {"role": "user",
     "content": "The ruling family is House Vantreel. Their sigil is a brass "
                "heron. They have held the city for eleven generations."},
    {"role": "assistant",
     "content": "Understood: House Vantreel, brass heron sigil, eleven "
                "generations in power."},
    {"role": "user",
     "content": "One more thing: the city's harbour is named Cold Anchor, and "
                "it is the only ice-free berth in the region."},
    {"role": "assistant",
     "content": "Got it — Cold Anchor is the region's only ice-free berth."},
]

# Distinctive tokens from the HISTORY above. Deliberately invented strings, so
# a match cannot be the model reciting something it knew already.
_HISTORY_NEEDLES = ("Quillhaven", "Vantreel", "Cold Anchor", "brass heron")


def _facts_text(facts: list[dict]) -> str:
    return " ".join(str(f.get("text", "")) for f in facts).lower()


def _settle(conv_id: str, *, quiet_for: float = 25.0, ceiling: float = 240.0) -> int:
    """Block until the fact count stops changing, and return it.

    NOT wait_for_facts(min_count=1). That returns on the FIRST fact to appear,
    which during a backfill is the middle of the job — it walks the history
    exchange by exchange and saves as it goes. Reading a count there and
    comparing it with a later one measures the backfill's PROGRESS and calls
    it duplication. That is exactly how this test failed the first time it ran
    against real weights: 1 -> 6, which was the job finishing.

    'Stopped changing' is the only end-of-backfill signal available from out
    here: the facts/<id>.backfill.json sidecar lives inside the compactor's
    container and no admin endpoint exposes its state.
    """
    deadline = time.time() + ceiling
    last = -1
    unchanged_since = 0.0
    while time.time() < deadline:
        n = len(H.admin_get_facts(conv_id))
        if n != last:
            last, unchanged_since = n, time.time()
        elif time.time() - unchanged_since >= quiet_for:
            return n
        time.sleep(5.0)
    return len(H.admin_get_facts(conv_id))


def test_backfill_reconstructs_facts_from_history_alone(conv_id):
    """A conversation that arrives WITH history and WITHOUT a facts file gets
    facts extracted from that history.

    This is the module's entire purpose and nothing has ever run it end to
    end. The current message is deliberately about something unrelated, so a
    fact naming the history's content cannot have come from the ordinary tail.
    """
    H.skip_if_no_admin("backfill is observed through /admin facts")
    H.requires_real_model(
        "the assertion is that an extracted fact is ABOUT the seeded history"
    )

    # Precondition, and it is load-bearing: needs_backfill() refuses outright
    # if a facts file already exists, so a dirty conv_id would make this test
    # pass or fail for reasons unrelated to the backfill.
    assert H.admin_get_facts(conv_id) == [], (
        "this conv must start with no facts — needs_backfill() returns False "
        "the moment a facts file exists, so a dirty id silently voids the test"
    )

    r = H.chat(
        "Ignore the setting for a moment — what is a good way to structure a "
        "session zero with new players?",
        conv_id=conv_id,
        prior_turns=_SEED_HISTORY,
        max_tokens=80,
    )
    assert r.status_code == 200

    # The triggering request returns WITHOUT facts by design; the work lands
    # later. POLL_CEILING tracks ZIONS_TEST_TAIL_WAIT, which the model profile
    # sets to 240 because a 0.5B model on CPU extracts slowly.
    facts = H.wait_for_facts(conv_id, min_count=1, max_wait=H.POLL_CEILING)
    assert facts, (
        "no facts appeared after a request carrying 6 messages of history "
        "against an empty store. Either the backfill never started (check "
        "'lazy backfill started in background' in the compactor log), or it "
        "ran and extracted nothing."
    )

    blob = _facts_text(facts)
    hits = [n for n in _HISTORY_NEEDLES if n.lower() in blob]
    assert hits, (
        f"facts were written, but none mentions anything from the seeded "
        f"history. Got: {[f.get('text') for f in facts]!r}\n"
        f"That is the failure mode this test exists for: the ordinary memory "
        f"tail extracting from the CURRENT exchange only, while the backfill "
        f"did nothing. Expected one of {list(_HISTORY_NEEDLES)}."
    )


def _plant_then_send_history(conv_id: str) -> tuple[set[str], list[dict]]:
    """Plant one fact with /remember, then hand the same conversation a long
    history, and return (the planted texts, the facts afterwards).

    Shared by the two halves of v3.1 F3 below. /remember is a slash command
    handled without a generation, so the plant is deterministic under either
    profile — that is what lets the first half run on every integration run.
    """
    H.chat("/remember The sommelier's name is Idris Vale.",
           conv_id=conv_id, max_tokens=40)
    planted = H.wait_for_facts(conv_id, min_count=1, max_wait=H.POLL_CEILING)
    assert planted, "/remember did not produce a fact; test cannot proceed"
    before = {str(f.get("text", "")) for f in planted}

    # needs_backfill() must decline on the facts file alone; if it somehow
    # starts, _run_backfill re-checks under the lock and refuses there.
    r = H.chat(
        "Anyway, what should I pair with a smoked trout course?",
        conv_id=conv_id,
        prior_turns=_SEED_HISTORY,
        max_tokens=80,
    )
    assert r.status_code == 200

    # _settle, NOT a flat sleep, and that is the difference between the two
    # assertions below meaning something and not.
    #
    # This used to be wait_for_async_tail(90) — a guess at how long a backfill
    # that WRONGLY ran would need before its facts showed up. A guess that is
    # too short turns "the history never reached the store" into "the history
    # had not reached the store YET", which is the same assertion passing for
    # a completely different reason, on the slow CPU profile where the
    # difference is most likely. _settle waits until the fact count stops
    # moving: it comes back in about 30s when nothing is happening (the
    # refusal, and the common case), and keeps waiting for as long as anything
    # is still saving. It is also FASTER than the 90s it replaces in the
    # ordinary case, which matters — two 240s sleeps in one suite is what
    # pushed the first real-weights run past its budget and got it SIGKILLed.
    #
    # BOUND, stated because it is the one way these can still go green for the
    # wrong reason: _settle gives up at its 240s ceiling, so a backfill slower
    # than that would not be seen from here. Nothing out here can do better —
    # the sidecar that knows is inside the container (see the module
    # docstring).
    _settle(conv_id)
    return before, H.admin_get_facts(conv_id)


def test_backfill_never_overwrites_an_existing_store(conv_id):
    """v3.1 F3, the half that needs no weights: nothing DESTROYS a fact.

    Runs under BOTH profiles, which is why it is separate from the refusal
    below: the plant is a slash command, the assertion is a subset check, and
    neither depends on what any generation said.

    WHAT IT PINS. Every writer on a request carrying a long history —
    `_merge_touched` in the async tail, `prune_facts`' eviction, and
    `_merge_backfilled` if a backfill ever does run here — leaves a fact that
    was already on disk alone. The F3 defect persisted `accumulated`
    wholesale and erased everything written underneath it while the backfill
    ran; this is the shape that catches that class.

    WHAT IT DOES NOT PIN, AND USED TO SAY IT DID. Its docstring called the
    REFUSAL its safety property, and `before <= after` cannot see a refusal:
    `_merge_backfilled` only ever appends, so "the backfill declined" and
    "the backfill walked all six seeded messages and added six facts" satisfy
    it identically. Reviewed 2026-09-11; the refusal moved to the test below,
    which can actually observe it.
    """
    H.skip_if_no_admin("needs /admin facts to read the store back")

    before, after_facts = _plant_then_send_history(conv_id)
    after = {str(f.get("text", "")) for f in after_facts}
    assert before <= after, (
        f"a fact present before the history arrived is gone afterwards.\n"
        f"  lost: {before - after!r}\n"
        f"Every path that writes facts on this request is additive by "
        f"contract; losing a planted fact means one of them replaced rather "
        f"than merged."
    )


def test_backfill_refuses_a_conversation_that_already_has_facts(conv_id):
    """v3.1 F3, the refusal itself: a populated store is not reconstructed.

    Reconstructing history over a store that already holds real memory would
    replace something known with something guessed, so `needs_backfill`
    declines on the facts file and `_run_backfill` re-checks under the lock
    and returns before it extracts anything.

    HOW A REFUSAL IS TOLD FROM A NO-OP, which is the whole point and is what
    the subset check above could not do. The seeded history carries invented
    strings that the current message never mentions, and the ordinary memory
    tail only ever sees the exchange it just handled. So a fact naming
    Quillhaven, Vantreel, Cold Anchor or the brass heron can only have come
    from walking the HISTORY — i.e. from a backfill that ran when it should
    have refused. Their absence is the refusal, observed.

    THE CONTROL FOR THIS ASSERTION IS TEST [1] in this file
    (`..._reconstructs_facts_from_history_alone`): it drives the SAME history
    with the SAME needles against an EMPTY store and asserts they DO arrive.
    Without it, "no needle appeared" would be satisfied by extraction being
    broken, or by the needles having quietly stopped being distinctive. If
    [1] is red, read nothing into this one.

    REAL WEIGHTS, and the reason is exact rather than cautious: against the
    weightless fixture every completion — the extraction call included —
    returns the same canned string, so a backfill that DID run over the
    history could not produce a needle either. There is no channel that
    distinguishes them from out here: the `facts/<id>.backfill.json` sidecar
    that records `state` and `exchanges_done` lives inside the compactor's
    container and no admin endpoint exposes it, and /health/full carries no
    backfill counter. Under the fixture this assertion cannot fail, so it
    skips rather than reporting a pass it did not earn.
    """
    H.skip_if_no_admin("needs /admin facts to read the store back")
    H.requires_real_model(
        "a refused backfill is told from a completed one by whether the "
        "SEEDED HISTORY reached the facts store, which needs a real extraction"
    )

    before, after_facts = _plant_then_send_history(conv_id)

    blob = _facts_text(after_facts)
    leaked = [n for n in _HISTORY_NEEDLES if n.lower() in blob]
    assert not leaked, (
        f"the backfill extracted from the seeded history even though the "
        f"store already held {len(before)} fact(s).\n"
        f"  history material that reached the store: {leaked}\n"
        f"  all facts now: {[f.get('text') for f in after_facts]!r}\n"
        f"That is v3.1 F3: needs_backfill() is supposed to decline on the "
        f"facts file, and _run_backfill to refuse again under the lock. A "
        f"reconstruction has replaced nothing yet — the merge is additive — "
        f"but it is now guessing alongside what she actually said."
    )


def test_backfill_does_not_duplicate_on_repeat_requests(conv_id):
    """Two requests carrying the same history must not extract it twice.

    `maybe_start_backfill` is documented idempotent — concurrent callers for
    one conv start a single task, and a 'complete' state stops later ones.
    Duplicated memory is the visible symptom when that fails, and dedup would
    then have to clean up after it.
    """
    H.skip_if_no_admin("backfill is observed through /admin facts")
    H.requires_real_model("needs a real extraction to have anything to double")

    H.chat("What is a session zero?", conv_id=conv_id,
           prior_turns=_SEED_HISTORY, max_tokens=60)
    _settle(conv_id)

    H.chat("And how long should it run?", conv_id=conv_id,
           prior_turns=H.extend_history(_SEED_HISTORY, "What is a session zero?",
                                        "A planning session before play begins."),
           max_tokens=60)
    _settle(conv_id)

    # COUNTS ARE THE WRONG UNIT, and the first version of this test used them:
    # it compared a count taken after wait_for_facts(min_count=1) — which
    # returns on the FIRST fact, i.e. the middle of a backfill — against a
    # settled count, and read the job finishing (1 -> 6) as duplication.
    #
    # Duplication means the same CLAIM stored twice, so compare texts. That is
    # also timing-independent: a late-arriving fact changes a count, but it
    # cannot turn a set of distinct texts into a set with repeats.
    texts = [
        " ".join(str(f.get("text", "")).lower().split())
        for f in H.admin_get_facts(conv_id)
    ]
    texts = [t for t in texts if t]
    dupes = sorted({t for t in texts if texts.count(t) > 1})
    assert not dupes, (
        f"the same fact text is stored more than once after a second request "
        f"carrying the same history:\n  {dupes!r}\n"
        f"all {len(texts)} fact(s): {texts!r}\n"
        f"A second backfill re-extracted history the first had already done; "
        f"the 'complete' state should have stopped it."
    )
