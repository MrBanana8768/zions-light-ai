"""The position-tracking EDGES: R15, R16, R17, R18, R20, R21.

Companion to test_summarizer.py, which owns the mainline position behaviour
(R23/R12: the two lower bounds, the chunk-label seed, the watermark repair).
This file owns the edges around it — the shapes that only bite once
`max_turns` is on, and that the v3.1.7 position pass fixed in code and left
without a single test:

    R15  an image-only turn hashes differently on the live path and in
         /admin/compact's reconstruction, so anchor[0] never matches
    R16  an anchorless capped upgrade holds two turns it never repays
         (a TRADE, pinned here so it cannot be changed by accident)
    R17  the whole history hashed on the event loop inside conv_lock
    R18  a capped window of byte-identical exchanges stalls the position
    R20  the anchor's "latest occurrence" rule vs a long early prefix
    R21  an empty or system-only window invents two turns and wipes the anchor

Run:
    python test_position_edges.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="compactor-test-position-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
# Small enough that a handful of fixture turns crosses the L1 gate.
os.environ["COMPACTOR_L1_CHUNK_SIZE"] = "4"
os.environ["COMPACTOR_L2_CHUNK_SIZE"] = "3"
os.environ["COMPACTOR_L3_CHUNK_SIZE"] = "2"

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


def _wipe():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP, ignore_errors=True)
    memory.ensure_storage_layout()


def _ex(u: str, a: str) -> list[dict]:
    return [{"role": "user", "content": u},
            {"role": "assistant", "content": a}]


def _state(**kw) -> dict:
    base = {"turns_seen": 0, "last_summarized_turn": 0, "tail_fp": []}
    base.update(kw)
    return base


def _install_mock(reply: str = "CHUNK"):
    """Patch httpx.AsyncClient so no rollup reaches the network."""
    import httpx

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": reply}}]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            return _Resp()

    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: _Client()
    return orig


def _restore_httpx(orig):
    import httpx
    httpx.AsyncClient = orig


# ---------------------------------------------------------------------------
# R15 — the image-only turn
# ---------------------------------------------------------------------------

_IMAGE_TURN = {
    "role": "user",
    "content": [{"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,AAAA"}}],
}
# What main._memorable_user_text writes into the episodic store for that turn,
# and therefore what main._rebuild_transcript_by_slot hands back to the
# summarizer on /admin/compact.
_AS_THE_STORE_REMEMBERS_IT = {"role": "user", "content": "[shared 1 image]"}


def test_an_image_only_turn_hashes_as_the_store_will_remember_it():
    print("\n[test] R15: the live window and the store agree on an image turn")
    live = summarizer._turn_fingerprints([_IMAGE_TURN])
    rebuilt = summarizer._turn_fingerprints([_AS_THE_STORE_REMEMBERS_IT])
    assert_eq(live, rebuilt,
              "an image-only turn has ONE fingerprint on both paths")
    # Two images, and the plural — the marker is compared byte for byte
    # against main.py's, so the 's' matters.
    two = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "a"}},
        {"type": "image_url", "image_url": {"url": "b"}},
    ]}
    assert_eq(summarizer._turn_fingerprints([two]),
              summarizer._turn_fingerprints(
                  [{"role": "user", "content": "[shared 2 images]"}]),
              "and the plural marker matches too")
    # A captioned image is NOT substituted: the store keeps the caption, so
    # substituting here would break the agreement in the other direction.
    captioned = {"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "a"}},
    ]}
    assert_eq(summarizer._turn_fingerprints([captioned]),
              summarizer._turn_fingerprints(
                  [{"role": "user", "content": "what is this?"}]),
              "a captioned image still hashes as its caption")
    # The marker is a hash input and nothing else — it must never be able to
    # reach a summary, a fact or a store.
    assert_true(all(len(f) == 16 for f in summarizer._turn_fingerprints(
        [_IMAGE_TURN])), "the marker leaves the function only as a hash")


def test_an_image_turn_at_the_head_of_the_anchor_does_not_inflate_the_position():
    print("\n[test] R15: anchor[0] on an image turn survives the rebuild")
    # anchor[0] is the OLDEST of the four anchored slots, and every prefix
    # _align_candidates tries starts there — so a mismatch in that one slot
    # defeats all four, the window reads as unalignable, and the position
    # takes _ASSUMED_NEW_TURNS on a call where nothing advanced. The drain
    # loop re-presents ONE transcript, so that inflation repeats per call and
    # persists in the state file: window_offset is then permanently wrong and
    # two turns are summarized twice.
    #
    # Put the image turn exactly at anchor[0]: fourth from the end.
    live = (_ex("first", "one") + _ex("second", "two")
            + [_IMAGE_TURN, {"role": "assistant", "content": "A tabby cat."}]
            + _ex("thanks", "welcome"))
    rebuilt = (_ex("first", "one") + _ex("second", "two")
               + [_AS_THE_STORE_REMEMBERS_IT,
                  {"role": "assistant", "content": "A tabby cat."}]
               + _ex("thanks", "welcome"))
    st = _state()
    live_pos = summarizer._observed_position("r15", st, live)
    assert_eq(live_pos, 8, "the live window puts the conversation at turn 8")
    assert_eq(st["tail_fp"], summarizer._turn_fingerprints(live[-4:]),
              "and the anchor's oldest slot IS the image turn")
    # Now /admin/compact re-presents the same conversation, from the store.
    # Truth: nothing advanced.
    for call in range(3):
        pos = summarizer._observed_position("r15", st, rebuilt)
        assert_eq(pos, 8, f"the rebuild adds nothing (drain call {call + 1})")


# ---------------------------------------------------------------------------
# R21 — the empty and system-only window
# ---------------------------------------------------------------------------

def test_an_empty_window_moves_nothing():
    print("\n[test] R21: an empty or system-only window is not evidence")
    for label, msgs in (("empty", []),
                        ("system only", [{"role": "system", "content": "s"}]),
                        ("two system turns",
                         [{"role": "system", "content": "a"},
                          {"role": "system", "content": "b"}])):
        st = _state(turns_seen=100, tail_fp=["a", "b", "c", "d"],
                    head_fp="hh", window_turns=100)
        pos = summarizer._observed_position("r21", st, msgs)
        assert_eq(pos, 100, f"{label}: the position stands at 100")
        assert_eq(st["tail_fp"], ["a", "b", "c", "d"],
                  f"{label}: the anchor is untouched")
        assert_eq(st["turns_seen"], 100, f"{label}: and so is the counter")
        # The window signature must survive too: overwritten with an empty
        # window's shape, the NEXT real window compares against a lie.
        assert_eq((st.get("head_fp"), st.get("window_turns")), ("hh", 100),
                  f"{label}: the window signature is untouched")


def test_an_empty_window_still_seeds_the_counter_from_the_chunks():
    print("\n[test] R21: holding is not the same as ignoring the state")
    # A pre-v3.1.4 file has no turns_seen at all. Holding must still write
    # back what _recorded_position knows, or the next call re-derives it.
    st = {"last_summarized_turn": 0, "tail_fp": ["a"],
          "l1": [{"first_turn": 1, "last_turn": 40, "summary": "s"}]}
    pos = summarizer._observed_position("r21b", st, [])
    assert_eq(pos, 40, "the chunk labels still set the position")
    assert_eq(st["turns_seen"], 40, "and it is written back")
    assert_eq(st["tail_fp"], ["a"], "with the anchor left alone")


# ---------------------------------------------------------------------------
# R20 — which match wins
# ---------------------------------------------------------------------------

def test_a_long_early_prefix_does_not_beat_a_short_late_one():
    print("\n[test] R20: every match is scored, and the smallest wins")
    anchor = ["A", "B", "C", "D"]
    fps = ["A", "B", "C", "D", "X", "Y", "A", "B", "C"]
    # The full anchor matches at the START (4 turns in, 5 new after it); the
    # 3-element prefix matches at the very END, which ends one slot before the
    # previous position and so means 0 new. Both are consistent with the
    # anchor. The old walk returned the first hit of the LONGEST prefix and so
    # answered 5, which is the opposite of the rule it documented.
    assert_eq(summarizer._align_candidates(anchor, fps), [0, 5],
              "both readings are reported, ascending")
    assert_eq(summarizer._align_new_turns(anchor, fps), 0,
              "and the rule takes the smallest: duplicate, never drop")
    # The rest of the alignment contract, so a mutation here cannot pass by
    # breaking only the case above.
    assert_eq(summarizer._align_new_turns(["w", "x", "y", "z"],
                                          ["w", "x", "y", "z"]), 0,
              "an unchanged window is 0 new")
    assert_eq(summarizer._align_new_turns(["w", "x", "y", "z"],
                                          ["w", "x", "y", "z", "p", "q"]), 2,
              "one appended exchange is 2 new")
    assert_eq(summarizer._align_new_turns(["w", "x", "y", "z"],
                                          ["w", "x", "y", "REGEN"]), 0,
              "a regenerated last turn is not a new turn")
    assert_eq(summarizer._align_new_turns(["w", "x"], ["p", "q", "r"]), None,
              "and no match at all is None, not 0")


# ---------------------------------------------------------------------------
# R18 — the repeating window under a cap
# ---------------------------------------------------------------------------

def _capped_repeat_positions(cap: int, real_exchanges: int, repeats: int):
    """Positions reported over `repeats` byte-identical exchanges, after
    `real_exchanges` distinct ones, through a window pinned at `cap`."""
    st = _state()
    hist: list[dict] = []
    for i in range(real_exchanges):
        hist += _ex(f"real question {i}", f"real answer {i}")
    summarizer._observed_position("r18", st, hist[-cap:])
    out = []
    for _ in range(repeats):
        hist += _ex("ok", "Sure.")
        out.append(summarizer._observed_position("r18", st, hist[-cap:]))
    return out


def test_a_repeating_tail_under_a_cap_still_advances():
    print("\n[test] R18: a repeating TAIL does not stall the position")
    # Two identical exchanges make the whole anchor match at the end of the
    # window (0 new) and again two turns earlier (2 new). Under a cap the
    # position is the only thing that advances, so believing the 0 is the
    # frozen hierarchy this release exists to fix, reached by another route.
    pos = _capped_repeat_positions(cap=20, real_exchanges=20, repeats=4)
    assert_eq(pos, [22, 24, 26, 28], "each repeated exchange still counts")


def test_a_window_that_is_ENTIRELY_a_repeat_still_advances():
    print("\n[test] R18: and neither does a window that is ALL repeats")
    # The second route, and the one the first fix missed. After cap/2
    # identical exchanges the window has filled with them, so it slides onto
    # period-2 identical content: the head hash repeats and the length is
    # pinned, which is byte-for-byte the signature of "the client re-sent the
    # window it sent last time". No content test of any width can separate
    # the two — `n < prev` (a strict suffix, which the admin drain cannot be)
    # is what does. Measured before the fix: 40, 40, 40, ... forever.
    pos = _capped_repeat_positions(cap=8, real_exchanges=6, repeats=8)
    assert_eq(pos, [10, 12, 14, 16, 18, 20, 22, 24],
              "the position keeps pace once the window is all repeats")
    assert_true(len(set(pos)) == len(pos), "no two calls report the same turn")


def test_the_admin_drain_still_holds_on_a_repeating_transcript():
    print("\n[test] R18: the drain's re-presented transcript still adds 0")
    # The other side of the same trade, and the reason the zero cannot simply
    # be disbelieved: /admin/compact loops maybe_rollup over ONE transcript
    # until the watermark stops moving. If a repeating tail made each pass
    # look like a new exchange, the position would run past the transcript and
    # chunks would be labelled onto text that is not theirs — which is the
    # thing that endpoint's own 409 exists to prevent.
    st = _state()
    msgs = _distinct(1, 6)
    # THREE identical exchanges, not two: with only two, the four-slot anchor
    # occurs once and the alignment is unambiguous, so this passes without
    # exercising the rule at all. Three is what makes the anchor match at the
    # end AND two turns earlier — the ambiguity the branch is about. (Found
    # by a mutation that survived: n <= prev instead of n < prev.)
    msgs += _ex("ok", "Sure.") * 3
    assert_eq(summarizer._align_candidates(
        summarizer._turn_fingerprints(msgs[-summarizer._ANCHOR_TURNS:]),
        summarizer._turn_fingerprints(msgs)), [0, 2],
        "the fixture really is ambiguous")
    seen = [summarizer._observed_position("r18d", st, msgs) for _ in range(6)]
    assert_eq(seen, [18] * 6, "six drain passes over one transcript add nothing")


def test_a_repeating_capped_window_reopens_the_l1_gate():
    print("\n[test] R18: the consequence — the hierarchy keeps rolling up")
    # The position number is the mechanism; THIS is the failure. Asserted
    # end to end so a fix that keeps the number moving but stops the gate
    # opening cannot pass.
    _wipe()
    cid = "r18-gate"
    orig = _install_mock("CHUNK")
    try:
        hist: list[dict] = []
        for i in range(10):
            hist += _ex(f"real question {i}", f"real answer {i}")
        asyncio.run(summarizer.maybe_rollup(cid, hist[-8:], "http://x", "m"))
        for _ in range(12):
            hist += _ex("ok", "Sure.")
            state = asyncio.run(
                summarizer.maybe_rollup(cid, hist[-8:], "http://x", "m"))
    finally:
        _restore_httpx(orig)
    assert_eq(state["turns_seen"], 32, "the position reached the tail of the loop")
    # The watermark is the gate's own output: it only moves when _do_l1_rollup
    # stores a chunk. Latched, it stops at the turn the position stalled on.
    assert_eq(state["last_summarized_turn"], 32,
              "and the watermark kept pace rather than latching")
    assert_true(state["l3"] is not None,
                "the whole cascade fired — L1 into L2 into L3")
    assert_true(summarizer.format_summary_block(state) is not None,
                "and there is a summary block to inject")


# ---------------------------------------------------------------------------
# R17 — the hashing cost
# ---------------------------------------------------------------------------

def test_only_the_tail_of_the_history_is_hashed():
    print("\n[test] R17: the hash is bounded, not linear in the history")
    # 25 ms per call at 700 messages, on the event loop, inside conv_lock,
    # every turn — and growing for the life of the conversation. Asserted
    # structurally rather than by a clock: a timing assertion on a shared CI
    # box is a flake, and what actually matters is that the bound exists.
    sizes: list[int] = []
    real = summarizer._turn_fingerprints

    def spy(messages):
        sizes.append(len(messages))
        return real(messages)

    summarizer._turn_fingerprints = spy
    try:
        big: list[dict] = []
        for i in range(350):
            big += _ex(f"q{i}", f"a{i}")
        st = _state()
        summarizer._observed_position("r17", st, big)
    finally:
        summarizer._turn_fingerprints = real
    assert_eq(len(big), 700, "the fixture is a 700-turn history")
    assert_true(max(sizes) <= summarizer._FINGERPRINT_TAIL_TURNS,
                f"no call hashes more than {summarizer._FINGERPRINT_TAIL_TURNS}"
                f" turns (largest was {max(sizes)})")
    assert_true(sum(sizes) <= summarizer._FINGERPRINT_TAIL_TURNS + 1,
                f"and the whole call hashes at most "
                f"{summarizer._FINGERPRINT_TAIL_TURNS + 1} turns in total "
                f"(it hashed {sum(sizes)})")


def test_bounding_the_hash_does_not_break_the_alignment():
    print("\n[test] R17: and the anchor still aligns against a long history")
    # The bound is only safe because the anchor is the tail of the PREVIOUS
    # window and main.py calls maybe_rollup once per exchange, so what the
    # alignment needs is always within the last few turns. Pinned end to end:
    # a full-history client with 700 turns must still report an exchange as
    # one exchange, not as _ASSUMED_NEW_TURNS off a failed alignment.
    st = _state()
    big: list[dict] = []
    for i in range(350):
        big += _ex(f"q{i}", f"a{i}")
    assert_eq(summarizer._observed_position("r17b", st, big), 700,
              "a 700-turn history reads as turn 700")
    big += _ex("q350", "a350")
    assert_eq(summarizer._observed_position("r17b", st, big), 702,
              "and one more exchange is two more turns")
    # A regeneration of the newest reply is still free at this length.
    big[-1] = {"role": "assistant", "content": "a350 regenerated"}
    assert_eq(summarizer._observed_position("r17b", st, big), 702,
              "a regenerated reply is still not a new turn")


# ---------------------------------------------------------------------------
# R16 — the anchorless capped upgrade. A TRADE, pinned so it stays deliberate.
# ---------------------------------------------------------------------------

def _distinct(first_exchange: int, count: int) -> list[dict]:
    """`count` exchanges of distinct content, numbered from `first_exchange`.

    Distinct on purpose: a fixture of identical turns is the R18 shape, and it
    would exercise the repeating-tail rule rather than the branch under test.
    """
    out: list[dict] = []
    for i in range(first_exchange, first_exchange + count):
        out += _ex(f"question {i}", f"answer {i}")
    return out


def test_an_anchorless_upgrade_advances_when_the_chunks_prove_a_suffix():
    print("\n[test] R16: chunks past the window prove the window is a suffix")
    st = _state(last_summarized_turn=100,
                l1=[{"first_turn": 641, "last_turn": 660, "summary": "s"}])
    pos = summarizer._observed_position("r16a", st, _distinct(281, 50))
    assert_eq(pos, 662, "the position is the conversation's, not the cap's")


def test_an_anchorless_upgrade_with_no_chunks_HOLDS_deliberately():
    print("\n[test] R16: with no anchor and no chunks the hold stands")
    # THIS IS A TRADE, NOT AN OVERSIGHT, and it is pinned so that changing it
    # has to be a decision. With no anchor and no chunk labels there is
    # nothing that can say whether the window in hand already contains this
    # exchange's two turns (so holding is right) or is a suffix sitting past
    # everything recorded (so holding under-counts by two, once, at upgrade).
    # `prev` cannot answer it: a watermark stranded ABOVE a genuinely shorter
    # history is the S-5 case, and reading that as proof of a bounded window
    # pushes the position past a history that has not caught up to it —
    # over-counting, which drops turns for good, to avoid under-counting,
    # which only summarizes some twice (invariant I4).
    st = _state(last_summarized_turn=100)
    window = _distinct(1, 50)
    pos = summarizer._observed_position("r16b", st, window)
    assert_eq(pos, 100, "the position holds at the recorded watermark")
    assert_eq(len(st["tail_fp"]), summarizer._ANCHOR_TURNS,
              "and THIS call lays the anchor down")
    # The hold is once, and it is never repaid — that is the whole of R16.
    # From the next call on the alignment is exact, so the deficit stays at
    # exactly two turns rather than growing.
    nxt = window[2:] + _ex("question 51", "answer 51")
    assert_eq(summarizer._observed_position("r16b", st, nxt), 102,
              "the very next call measures exactly")
    nxt = nxt[2:] + _ex("question 52", "answer 52")
    assert_eq(summarizer._observed_position("r16b", st, nxt), 104,
              "and the one after it, so the hold does not compound")


if __name__ == "__main__":
    try:
        _wipe()
        test_an_image_only_turn_hashes_as_the_store_will_remember_it()
        test_an_image_turn_at_the_head_of_the_anchor_does_not_inflate_the_position()
        test_an_empty_window_moves_nothing()
        test_an_empty_window_still_seeds_the_counter_from_the_chunks()
        test_a_long_early_prefix_does_not_beat_a_short_late_one()
        test_a_repeating_tail_under_a_cap_still_advances()
        test_a_window_that_is_ENTIRELY_a_repeat_still_advances()
        test_the_admin_drain_still_holds_on_a_repeating_transcript()
        test_a_repeating_capped_window_reopens_the_l1_gate()
        test_only_the_tail_of_the_history_is_hashed()
        test_bounding_the_hash_does_not_break_the_alignment()
        test_an_anchorless_upgrade_advances_when_the_chunks_prove_a_suffix()
        test_an_anchorless_upgrade_with_no_chunks_HOLDS_deliberately()
        print("\nAll position-edge tests passed.")
    finally:
        if os.path.exists(_TMP):
            shutil.rmtree(_TMP, ignore_errors=True)

# ---------------------------------------------------------------------------
# Mutation record for the v3.1.7 position edges. Each behaviour was broken in
# summarizer.py one at a time and both this file and test_summarizer.py re-run;
# the assertion named is the one that went red. A mutation that survives means
# the test is decoration.
#
#   the image-marker substitution deleted from      -> "an image-only turn has
#     _turn_fingerprints                               ONE fingerprint on both
#                                                      paths"
#   _image_only_marker loses its plural             -> "and the plural marker
#                                                      matches too"
#   `if not turns:` -> `if False:`                  -> IndexError on head_fp
#                                                      (the guard is what
#                                                      makes the rest legal)
#   the empty branch clears tail_fp as well         -> "empty: the anchor is
#                                                      untouched"
#   _align_candidates returns on its first hit      -> "both readings are
#     (the pre-v3.1.7 walk)                            reported, ascending"
#   _align_new_turns takes cands[-1]                -> "and the rule takes the
#                                                      smallest: duplicate,
#                                                      never drop"
#   the suffix clause dropped from the ambiguous-   -> "the position keeps pace
#     zero test (the first R18 fix on its own)         once the window is all
#                                                      repeats" ([10,12,14,16,
#                                                      16,16,16,16])
#   the ambiguous-zero branch removed entirely      -> "each repeated exchange
#                                                      still counts"
#   `n < prev` -> `n <= prev`                       -> "six drain passes over
#                                                      one transcript add
#                                                      nothing"
#   the fingerprint tail bound removed              -> "no call hashes more
#                                                      than 64 turns"
#   the anchorless branch stops consulting          -> "the position is the
#     _highest_chunk_turn                              conversation's, not the
#                                                      cap's"
#   the anchorless hold advances instead            -> "the position holds at
#                                                      the recorded watermark"
#
# One of these SURVIVED on the first sweep and the reason is worth keeping:
# `n <= prev` was green because the drain fixture used only TWO identical
# trailing exchanges, which the four-slot anchor matches exactly once — so the
# ambiguous branch the mutation lives on was never entered. Three identical
# exchanges is the smallest fixture that reaches it. A fixture that does not
# reach the code under test is the same failure as a missing test.
# ---------------------------------------------------------------------------
