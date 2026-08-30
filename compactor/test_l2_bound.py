"""
CPU-only test for the L2 tier's bound (MEMORY_REVIEW S-1/S-6).

A prior review found that `_do_l3_rollup` folds every L2 chapter into L3 but
never removes them from `state["l2"]`, unlike L1->L2 (`_do_l2_rollup`, which
drops the L1 chunks it consumes). If true, `state["l2"]` — and therefore
`format_summary_block`'s injected block, and the on-disk state file — grows
by one chapter every L2_CHUNK_SIZE*L1_CHUNK_SIZE turns for the life of a
conversation.

This drives MANY turns through the REAL rollup machinery
(summarizer.maybe_rollup, with only the HTTP layer mocked) and measures
len(state["l2"]) over time, rather than asserting on the code shape. Content
is synthetic lorem-ipsum-style text only — no real conversation content, per
project policy.

Run: python test_l2_bound.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="compactor-test-l2bound-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
# Small thresholds, same style as test_summarizer.py, so a run of a few
# hundred turns exercises many L1/L2/L3 cycles instead of one.
os.environ["COMPACTOR_L1_CHUNK_SIZE"] = "4"
os.environ["COMPACTOR_L2_CHUNK_SIZE"] = "3"
os.environ["COMPACTOR_L3_CHUNK_SIZE"] = "2"

import summarizer  # noqa: E402
import memory  # noqa: E402


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _wipe():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP)
    memory.ensure_storage_layout()


class _Resp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _CountingMockClient:
    """Every chat-completions call returns a short synthetic summary. Bodies
    here are tiny (lorem-ipsum turns), so nothing exercises /tokenize."""

    def __init__(self):
        self.n = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        self.n += 1
        return _Resp(f"synthetic-summary-{self.n}")


def _install():
    import httpx
    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: _CountingMockClient()
    return orig


def _restore(orig):
    import httpx
    httpx.AsyncClient = orig


def test_l2_stays_bounded_over_many_rollup_cycles():
    print("\n[test] len(state['l2']) stays bounded across 240 synthetic turns")
    _wipe()
    cid = "l2-bound-many-turns"
    orig = _install()
    l2_lengths = []
    try:
        msgs = [{"role": "system", "content": "lorem ipsum dolor sit amet"}]
        total_turns = 0
        # 60 cycles * 4 turns/cycle (L1_CHUNK_SIZE) = 240 turns, which at
        # L2_CHUNK_SIZE=3 crosses 20 L2-chapter boundaries, and at
        # L3_CHUNK_SIZE=2 crosses ~10 L3-refresh boundaries — more than
        # enough passes for unbounded growth (pre-fix: reached 20 and
        # climbing) to show up if it still existed.
        for _cycle in range(60):
            for _ in range(4):
                total_turns += 1
                role = "user" if total_turns % 2 == 1 else "assistant"
                msgs.append({
                    "role": role,
                    "content": f"lorem-ipsum turn {total_turns} consectetur adipiscing elit",
                })
            state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
            l2_lengths.append(len(state.get("l2") or []))
    finally:
        _restore(orig)

    max_l2 = max(l2_lengths)
    final_l2 = l2_lengths[-1]
    print(f"  .    len(l2) over the run: min={min(l2_lengths)} max={max_l2} "
          f"final={final_l2} (L3_CHUNK_SIZE={summarizer.L3_CHUNK_SIZE})")

    # The bound this fix claims: l2 never reaches L3_CHUNK_SIZE for more than
    # the instant inside one maybe_rollup call before L3 drains it — so
    # OBSERVED (post-call) length never reaches the threshold at all in
    # steady state, the same shape l1 already has relative to L2_CHUNK_SIZE.
    assert_true(max_l2 < summarizer.L3_CHUNK_SIZE,
                f"max len(l2)={max_l2} stays under L3_CHUNK_SIZE "
                f"({summarizer.L3_CHUNK_SIZE}) across the whole run")

    # The pre-fix shape was monotonic, unbounded growth (verified separately
    # against the unfixed code: reached 20 after 240 turns and still
    # climbing). Assert the STRONGER, directly-relevant property here: the
    # tail of the run is not sitting at a high-water mark still climbing.
    tail = l2_lengths[-6:]
    assert_true(max(tail) < summarizer.L3_CHUNK_SIZE,
                f"the last 6 measurements ({tail}) show l2 has plateaued, not "
                f"still growing")

    # And the state actually written to disk carries the same bound — this
    # is what makes the state FILE bounded too, not just the in-memory value
    # for one instant.
    on_disk = summarizer.load_state(cid)
    assert_true(len(on_disk.get("l2") or []) < summarizer.L3_CHUNK_SIZE,
                "the on-disk l2 is bounded too")


def test_l1_and_l3_are_unaffected_by_the_l2_bound():
    print("\n[test] l1 stays bounded as before, and l3 keeps advancing")
    _wipe()
    cid = "l2-bound-siblings"
    orig = _install()
    try:
        msgs = [{"role": "system", "content": "lorem ipsum dolor sit amet"}]
        total_turns = 0
        last_l3_span = None
        for _cycle in range(30):
            for _ in range(4):
                total_turns += 1
                role = "user" if total_turns % 2 == 1 else "assistant"
                msgs.append({
                    "role": role,
                    "content": f"lorem-ipsum turn {total_turns} consectetur",
                })
            state = asyncio.run(summarizer.maybe_rollup(cid, msgs, "http://x", "m"))
            assert_true(len(state.get("l1") or []) < summarizer.L2_CHUNK_SIZE,
                        f"l1 stays within its own bound at turn {total_turns}")
            if state.get("l3"):
                span = (state["l3"]["first_turn"], state["l3"]["last_turn"])
                if last_l3_span is not None:
                    assert_true(span[1] >= last_l3_span[1],
                                "l3's covered span never regresses")
                last_l3_span = span
    finally:
        _restore(orig)
    assert_true(last_l3_span is not None, "l3 was produced at least once")
    assert_true(last_l3_span[1] == total_turns,
                "l3's final span reaches the last turn summarized")


def _all():
    return [
        test_l2_stays_bounded_over_many_rollup_cycles,
        test_l1_and_l3_are_unaffected_by_the_l2_bound,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            t()
        print("\nAll L2-bound tests passed.")
    finally:
        if os.path.exists(_TMP):
            shutil.rmtree(_TMP, ignore_errors=True)
