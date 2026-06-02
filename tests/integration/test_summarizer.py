"""
Tier-3 validation of the hierarchical summary layer (V2.0 Phase 4).

The default L1 chunk size is 20 turns — so triggering even ONE L1
rollup requires sending 20 chat requests. At ~10-30s per chat, that's
5-10 minutes of real time per test. We mark these `slow` so the default
pytest run skips them; run with `-m slow` (or `-m "slow or not slow"`)
to include them when you want full coverage.

The Tier-1 test_summarizer.py already proves the state machine works
in isolation with mocked thresholds (4/3/2) and a mocked LLM — these
Tier-3 tests only need to verify the wiring (real vLLM → real rollup →
state actually written to /data).
"""

import pytest

import _harness as H


pytestmark = pytest.mark.slow


def _drive_turns(conv_id: str, n: int) -> list[dict]:
    """Send N chat turns on filler topics. Returns the accumulated history
    so the test can keep extending it. Uses tiny max_tokens to keep each
    turn fast."""
    history: list[dict] = []
    for i in range(n):
        prompt = f"Reply with the single word 'ok-{i}' and nothing else."
        r = H.chat(prompt, conv_id=conv_id, prior_turns=history,
                   max_tokens=10)
        assert r.status_code == 200, f"chat turn {i} returned {r.status_code}"
        history = H.extend_history(history, prompt, r.response_text)
    return history


def test_l1_rollup_fires_after_threshold(conv_id):
    """Drive ≥ 20 turns, wait for async tail, assert summary state has
    at least one L1 chunk. Admin-required because this is verified via
    the state file."""
    H.skip_if_no_admin("summary state inspection requires admin endpoint")

    # 22 turns: well past the default L1 threshold (20) so even a slightly
    # off-by-one or in-flight rollup is caught.
    _drive_turns(conv_id, 22)

    # Generous tail wait — last rollup could be in flight, and L1 rollups
    # involve a real LLM call (a summarization request).
    H.wait_for_async_tail(seconds=30)

    state = H.admin_get_summary(conv_id)
    l1 = state.get("l1") or []
    assert len(l1) >= 1, (
        f"expected at least 1 L1 chunk after 22 turns, got {len(l1)}\n"
        f"state: {state}"
    )
    last = state.get("last_summarized_turn", 0)
    assert last >= 20, (
        f"expected last_summarized_turn >= 20, got {last}\nstate: {state}"
    )
    # The chunk should cover turns 1-20 (or thereabouts).
    chunk = l1[0]
    assert chunk.get("first_turn") == 1, f"unexpected first_turn: {chunk}"
    assert chunk.get("last_turn") == 20, f"unexpected last_turn: {chunk}"
    assert chunk.get("text"), "L1 chunk has empty text"
    print(f"  L1 chunk produced: turns {chunk['first_turn']}-{chunk['last_turn']}, "
          f"{len(chunk['text'])} chars")


def test_summary_state_appears_in_admin_endpoint(conv_id):
    """After a rollup, /admin/conversations/<id> reflects the new state."""
    H.skip_if_no_admin()
    _drive_turns(conv_id, 22)
    H.wait_for_async_tail(seconds=30)

    summary = H.admin_conv_summary(conv_id)
    sstate = summary.get("summary") or {}
    assert sstate.get("l1_chunks", 0) >= 1, f"summary.l1_chunks not surfaced: {summary}"
    assert sstate.get("last_summarized_turn", 0) >= 20
