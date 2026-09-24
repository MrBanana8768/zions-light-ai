"""
compactor/test_tail_backfill_position.py

hostile317-b F3 — a lazy backfill's late `maybe_rollup` call can land minutes
after the request that triggered it, with THAT request's now-stale array.
Meanwhile a real, concurrent live tail can have advanced the conversation's
position in the interim. `_observed_position` (summarizer.py) has no way to
tell "a short window because the client capped it" (a SUFFIX — its own tail
IS the true current tail) from "a short window because this snapshot
predates turns that already landed" (a PREFIX — its tail is stale, superseded
content) — both present as `n < prev` with a partial anchor match landing
at the array's own end. Calling `maybe_rollup` with the stale snapshot
therefore overwrites state["tail_fp"] / ["head_fp"] / ["window_turns"] with
that snapshot's OWN (stale) tail, even though `turns_seen` itself stays
correct (position math never goes backwards). Every later request's
`_observed_position` then aligns against the WRONG anchor, and the offset
compounds: chunk labels stop matching the text they claim to cover.

The fix (backfill.py, `_run_backfill`): before calling `maybe_rollup` with
the kickoff-time snapshot, re-read the conversation's already-recorded
position and compare it against the snapshot's own turn count. If the
conversation has already reached further than the snapshot could represent,
skip the rollup call entirely rather than handing `_observed_position` a
window it cannot correctly place. The facts backfill itself is unaffected;
only the summary-position update is skipped, and the very next live turn
rolls up normally against the real, current window.

This test drives the REAL `summarizer.maybe_rollup` (not a stub) and a REAL
live tail call interleaved between the backfill's snapshot and its own late
rollup call, exactly as production timing would have it.

Run:
    python test_tail_backfill_position.py
"""

import asyncio
import contextlib
import os
import shutil
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-tail-backfill-position-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import backfill  # noqa: E402
import memory  # noqa: E402
import summarizer  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _wipe_storage():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()
    backfill._in_progress_local.clear()


def _exchange(n_text: str, a_text: str) -> list[dict]:
    return [
        {"role": "user", "content": n_text},
        {"role": "assistant", "content": a_text},
    ]


# The "true" conversation: 4 exchanges (8 turns). The backfill's snapshot
# (taken at kickoff) is only the first 3 exchanges (6 turns) — the 4th
# exchange is what the concurrent live tail appends while the backfill runs.
_EXCHANGES = [
    ("Tell me about Lyra.", "Lyra is a half-elf ranger."),
    ("What world is this in?", "It's Aethermere, a low-magic kingdom."),
    ("Who is her mentor?", "An old druid named Cael."),
    ("What happens next?", "She sets out for the Sundered Vale."),
]


def _messages_through(n_exchanges: int) -> list[dict]:
    out: list[dict] = []
    for u, a in _EXCHANGES[:n_exchanges]:
        out.extend(_exchange(u, a))
    return out


def _mock_async_client_factory(content: str):
    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return _FakeResponse()

    return _FakeClient()


@contextlib.contextmanager
def _patched_httpx(content: str):
    import httpx as real_httpx
    orig = real_httpx.AsyncClient

    def _fake_client(*a, **k):
        return _mock_async_client_factory(content)

    real_httpx.AsyncClient = _fake_client
    try:
        yield
    finally:
        real_httpx.AsyncClient = orig


def test_stale_backfill_snapshot_does_not_corrupt_anchor():
    print("\n[test] a stale backfill snapshot must not overwrite the live anchor")
    _wipe_storage()
    cid = "stale-backfill"

    snapshot = _messages_through(3)   # 6 turns — what the backfill kicks off with
    full = _messages_through(4)       # 8 turns — the true conversation once the
                                       # live tail's 4th exchange lands

    async def scenario():
        # 1. A REAL live tail advances the conversation's position BEFORE the
        #    backfill's own rollup call runs — exactly the race hostile317-b
        #    demonstrated (backfill snapshotted at request N, live chat
        #    continues to N+1 while the backfill is still extracting facts).
        live_state = await summarizer.maybe_rollup(
            cid, full, "http://fake", "fake-model", raw_messages=full
        )
        assert_eq(live_state["turns_seen"], 8, "live tail recorded the true position")
        live_tail_fp = list(live_state["tail_fp"])
        live_window_turns = live_state["window_turns"]
        live_head_fp = live_state["head_fp"]
        assert_true(live_tail_fp, "live tail laid down a real anchor")

        # 2. The backfill's OWN late `maybe_rollup` call, with the STALE
        #    6-turn snapshot taken at kickoff — run through the real
        #    `_run_backfill`, not a hand-rolled call, so the guard under test
        #    is the one shipped code actually takes.
        with _patched_httpx("- Lyra is a half-elf ranger."):
            await backfill._run_backfill(
                cid, snapshot, "http://fake", "fake-model", raw_messages=snapshot
            )

        return live_tail_fp, live_window_turns, live_head_fp

    live_tail_fp, live_window_turns, live_head_fp = asyncio.run(scenario())

    after = summarizer.load_state(cid)
    assert_eq(after["turns_seen"], 8,
              "position still correct (turns_seen never regresses even unfixed)")
    assert_eq(after["tail_fp"], live_tail_fp,
              "*** F3: the stale 6-turn snapshot must NOT overwrite the live "
              "8-turn anchor")
    assert_eq(after["window_turns"], live_window_turns,
              "*** F3: window_turns must still reflect the live window, not "
              "the stale snapshot's 6")
    assert_eq(after["head_fp"], live_head_fp,
              "*** F3: head_fp must still reflect the live window's head")

    # And the NEXT live turn must still align correctly against the
    # untouched anchor — the actual consequence hostile317-b measured
    # ("every later chunk label covers different text").
    async def next_turn():
        nxt = _messages_through(4)
        nxt.extend(_exchange("One more thing?", "Yes, Cael left a warning."))
        return await summarizer.maybe_rollup(
            cid, nxt, "http://fake", "fake-model", raw_messages=nxt
        )

    nxt_state = asyncio.run(next_turn())
    assert_eq(nxt_state["turns_seen"], 10,
              "the position after the stale backfill call still advances "
              "exactly one exchange per exchange — no permanent +1/-1 offset")


def test_control_non_stale_backfill_still_rolls_up():
    print("\n[test] CONTROL: a backfill snapshot that is NOT stale still runs "
          "its rollup")
    _wipe_storage()
    cid = "fresh-backfill"
    snapshot = _messages_through(3)  # 6 turns, and nothing else has touched
                                      # this conv_id yet — recorded position
                                      # is 0, so the guard must NOT refuse.

    async def scenario():
        with _patched_httpx("- Lyra is a half-elf ranger."):
            await backfill._run_backfill(
                cid, snapshot, "http://fake", "fake-model", raw_messages=snapshot
            )

    asyncio.run(scenario())

    after = summarizer.load_state(cid)
    assert_eq(after["turns_seen"], 6,
              "fixture ok   CONTROL: a genuinely fresh backfill snapshot still "
              "lays down the position (the guard is not a blanket refusal)")
    assert_true(after["tail_fp"], "CONTROL: the anchor was recorded")
    assert_eq(after["window_turns"], 6, "CONTROL: window_turns matches the snapshot")


if __name__ == "__main__":
    try:
        test_control_non_stale_backfill_still_rolls_up()
        test_stale_backfill_snapshot_does_not_corrupt_anchor()
        print("\nAll test_tail_backfill_position checks passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
