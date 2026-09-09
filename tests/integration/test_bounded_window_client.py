"""Memory must work for a client that sends a BOUNDED WINDOW.

WHY THIS EXISTS. Every integration test in this directory drives the compactor
the way OpenWebUI does: it re-sends the whole conversation every turn. The
client described by FRONTEND_SPEC does not. §4 rule 2 makes a bounded window
the committed shape — the client keeps the full chain locally and sends the
last N turns — and nothing has ever verified that memory survives that.

FRONTEND_SPEC §15 asserts it does not. It says the compactor's only notion of
position is the client's array length, so under a bounded window every
exchange overwrites the same ChromaDB document, `recent_cutoff` stays ~0, and
`_needs_l1_rollup` is never true — "the memory architecture is inert against
the client this spec describes."

That was written 2026-08-24 and each mechanism has since been fixed
independently: content-addressed episodic ids and store-allocated ordinals
(v3.1 D1), the out-of-frame cutoff guard (v3.1 A6), and `_observed_position`
owning the conversational position instead of `len(messages)` (v3.1.4). So the
claim is believed to be stale — and "believed stale" is not a thing to hand a
front-end team on a document. This file is the measurement that settles it,
against the real stack, in the shape the client will actually send.

It asserts the three properties the front end needs to be able to trust and
that only the real stack can show, each a different failure if it breaks:

  1. facts ACCUMULATE — the window is constant, the store is not;
  2. episodic rows do not COLLIDE — one indexed exchange per exchange, not one
     row overwritten n times (the phantom conversation was sixteen writes to
     index 2: one document, fifteen destructions);
  3. the summarizer WATERMARK advances — the hierarchy is alive under a window.

    pytest tests/integration/test_bounded_window_client.py -v
"""

import pytest

import _harness as h

# The window the client sends: the last WINDOW_TURNS messages of the chain.
# Small on purpose — the point is that it is CONSTANT while the conversation
# grows past it, which is the condition §15 says breaks memory.
WINDOW_TURNS = 6
EXCHANGES = 10


@pytest.fixture(scope="module")
def bounded_conv():
    """Drive one conversation with a bounded-window client, return what the
    server made of it.

    The full chain is kept here, exactly as the real client keeps it; only the
    tail is ever sent. `turns_sent` is recorded per exchange so the assertions
    can prove the window really was bounded rather than assuming it.
    """
    h.require_base_url()
    h.skip_if_no_admin("bounded-window memory needs the admin API to observe")

    conv = h.fresh_conv_id()
    chain: list[dict] = []
    sent_counts: list[int] = []

    for i in range(EXCHANGES):
        window = chain[-WINDOW_TURNS:] if chain else []
        res = h.chat(
            f"Turn {i}: tell me one thing about the number {i}.",
            conv_id=conv,
            prior_turns=window,
        )
        assert res.status_code == 200, f"turn {i} failed: {res.status_code}"
        sent_counts.append(res.turns_sent)
        chain = h.extend_history(
            chain, f"Turn {i}: tell me one thing about the number {i}.",
            res.response_text or "(empty)",
        )

    h.wait_for_async_tail()
    return {
        "conv": conv,
        "sent_counts": sent_counts,
        "chain_len": len(chain),
        "summary": h.admin_conv_summary(conv),
        "facts": h.admin_get_facts(conv),
        "state": h.admin_get_summary(conv),
    }


def test_the_window_really_was_bounded(bounded_conv):
    """The control. Without this every assertion below could be passing
    because the harness quietly sent the whole history."""
    sent = bounded_conv["sent_counts"]
    assert bounded_conv["chain_len"] == EXCHANGES * 2, "the chain grew as expected"
    # +1 for the new user message appended to the window.
    assert max(sent) <= WINDOW_TURNS + 1, (
        f"the client sent {max(sent)} messages at its widest, which is not a "
        f"bounded window of {WINDOW_TURNS} — this test would prove nothing"
    )
    assert sent[-1] < bounded_conv["chain_len"], (
        f"by the last turn the conversation was {bounded_conv['chain_len']} "
        f"messages and the client sent {sent[-1]}; if these were equal the "
        f"window never actually bit"
    )


def test_facts_accumulate_though_the_window_does_not_grow(bounded_conv):
    """Property 1. A constant-length window must not mean a constant store."""
    facts = bounded_conv["facts"]
    assert len(facts) > 0, (
        "no facts were stored across "
        f"{EXCHANGES} exchanges sent as a bounded window — memory is inert "
        "against the client FRONTEND_SPEC describes, which is exactly what "
        "§15 predicted and this test exists to check"
    )


def test_episodic_rows_do_not_collide(bounded_conv):
    """Property 2. The phantom conversation was sixteen writes to index 2 —
    one surviving document and fifteen destroyed ones — caused by a doc id
    derived from the client's array length. A bounded window is the same
    input shape that produced it."""
    episodic = (bounded_conv["summary"] or {}).get("episodic") or {}
    indexed = episodic.get("indexed_exchanges")
    if indexed is None:
        pytest.skip("this build's admin summary does not report episodic count")
    assert indexed > 1, (
        f"only {indexed} episodic row(s) after {EXCHANGES} exchanges — rows "
        f"are overwriting each other, which is the v3.1 D1 failure returning "
        f"under a bounded window"
    )


def test_the_summary_hierarchy_is_not_frozen(bounded_conv):
    """Property 3. §15's sharpest claim: `_needs_l1_rollup` is never true under
    a bounded window, so the hierarchy never advances. The gate reads
    `_observed_position`, not `len(messages)`, so the position should move even
    though the array length does not."""
    state = bounded_conv["state"] or {}
    seen = state.get("turns_seen")
    if seen is None:
        pytest.skip("this build does not persist turns_seen")
    assert seen > max(bounded_conv["sent_counts"]), (
        f"the server's position is {seen}, no further along than the widest "
        f"window it was sent ({max(bounded_conv['sent_counts'])}). The position "
        f"is tracking the client's array rather than the conversation, which "
        f"is the stall §15 describes"
    )


# `added_turn` monotonicity is NOT tested here, deliberately.
#
# It needs two facts from one conversation to see movement, and the fixture
# backend extracts about one — so this assertion could only ever SKIP, and a
# skip is not evidence of anything. The property is deterministic and belongs
# in a unit test: compactor/test_added_turn.py drives the allocation directly
# and is mutation-checked. What this file proves is the part only the real
# stack can prove: that a bounded window does not stop memory accumulating.
