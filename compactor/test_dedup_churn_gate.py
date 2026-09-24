"""
compactor.dedup — F1 part 4: gating the dedup treadmill.

V314_BACKLOG F1 part 4, backed by N3's 48h production measurement: 234
dedup passes, 2,024 LLM calls, 55 merges (2.7% yield), "transitive cluster
exceeds the cap" firing on essentially every pass. The refusal memo
(dedup.py:183-235) is logically correct — a cluster with EXACT unchanged
membership since a refused verdict is skipped for free — but a store that
grows one turn's worth of facts (15-22) at a time keeps handing an
oversized transitive cluster to `_split_cluster`, whose old balanced
algorithm rebalanced every sub-cluster boundary from the current total
size. That reshuffled which facts shared a sub-cluster on almost every
pass, so the memo rarely got to reuse a verdict even when most of a
cluster's members hadn't changed at all.

This file measures the fix (front-loaded, memo-stable splitting) against
the defect it replaces, on synthetic workloads shaped like the production
pattern N3 describes, and reports two different things honestly:

  - Under an UNCAPPED call budget, the fix issues strictly no more calls
    than the old split, and typically a few percent fewer — a single
    steadily-growing blob rarely gives the old split room to reshuffle
    more than one boundary at a time, so the ceiling on this measurement
    is modest and the tests say so rather than asserting otherwise.
  - Under a SATURATED call budget (N3's actual regime: 8.65/10 calls per
    pass, "exceeds the cap" firing on essentially every pass) the capped
    call COUNT is identical by construction — both algorithms spend the
    whole budget every pass once there is enough genuinely new material
    to fill it, and there almost always is. What the fix changes there is
    the DEFERRED backlog: fewer of the 10 calls are spent re-litigating a
    reshuffled-but-already-decided shape, so more of them resolve
    genuinely new material, and the backlog of never-yet-examined
    clusters grows measurably slower (~13% slower in the workload below).
    That is real — it is N3's own symptom ("deferred clusters grew from 4
    to 23") — but it is a throughput/fairness improvement, not a
    guaranteed reduction in raw GPU call volume at today's churn rate.

A genuine merge still goes through once the underlying facts actually are
the same, proven separately below.

No production data: every fact string below is synthetic.

Run: python test_dedup_churn_gate.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

_TMP_ROOT = tempfile.mkdtemp(prefix="zions_dedup_churn_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # skip ChromaDB init in retrieval

import retrieval  # noqa: E402
import dedup  # noqa: E402


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


def _fact(text: str, added_turn: int = 0, last_used: int = 0) -> dict:
    return {"text": text, "added_turn": added_turn, "last_used": last_used}


def _mock_chat_response(content: str, finish_reason: str | None = "stop"):
    choice = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    r = MagicMock(status_code=200)
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value={"choices": [choice]})
    return r


# ---------------------------------------------------------------------------
# The pre-F1-part-4 balanced split, kept here ONLY as a measurement
# baseline so the "before" side of the before/after comparison below is
# the actual old behaviour and not a hand-wave. Not exported, not used by
# any non-test code.
# ---------------------------------------------------------------------------

def _old_balanced_split(cluster: list[int]) -> list[list[int]]:
    n = len(cluster)
    if n <= dedup.MAX_CLUSTER_SIZE:
        return [cluster]
    chunks = -(-n // dedup.MAX_CLUSTER_SIZE)  # ceil
    base, extra = divmod(n, chunks)
    out: list[list[int]] = []
    start = 0
    for k in range(chunks):
        size = base + (1 if k < extra else 0)
        out.append(cluster[start:start + size])
        start += size
    return out


# ---------------------------------------------------------------------------
# _split_cluster itself: front-loaded prefix is stable as the group grows
# ---------------------------------------------------------------------------

def test_split_cluster_leading_full_chunk_is_stable_as_the_group_grows():
    print("\n[test] _split_cluster: a full leading chunk keeps its exact "
          "membership as later members join")
    cap = dedup.MAX_CLUSTER_SIZE
    # Grow the group one member at a time, exactly the production pattern
    # (extraction appends new facts to a store that already clusters). The
    # very first growth step (cap -> cap+1) is the one place even the new
    # split still reshapes the leading chunk (it has to become full for
    # the first time) — every step after that must leave it untouched, and
    # this is checked unconditionally, not only when the chunk happens to
    # already be full, so a regression that lets it wobble again would be
    # caught rather than silently skipped.
    prior_first_chunk = dedup._split_cluster(list(range(cap + 2)))[0]
    for n in range(cap + 3, 4 * cap):  # 7 .. 15 for cap=4
        parts = dedup._split_cluster(list(range(n)))
        first = parts[0]
        assert_eq(first, prior_first_chunk,
                  f"n={n}: leading full chunk unchanged from n={n - 1}")
        prior_first_chunk = first
        # Invariants the old split also guaranteed — still true here.
        flat = sorted(i for p in parts for i in p)
        assert_eq(flat, list(range(n)), f"n={n}: every member still present")
        assert_true(all(len(p) <= cap for p in parts), f"n={n}: no part over cap")
        assert_true(all(len(p) >= 2 for p in parts), f"n={n}: no orphaned member")


def _one_step_survival(splitfn, lo: int, hi: int) -> tuple[int, int]:
    """Over n in [lo, hi), split a group of n and of n+1 (one growth
    step) and count how many of the n-split's sub-clusters reappear
    byte-identical in the n+1-split — the exact condition the refusal
    memo needs to skip a re-ask. Returns (survived, total)."""
    survived = 0
    total = 0
    for n in range(lo, hi):
        before = splitfn(list(range(n)))
        after = [tuple(c) for c in splitfn(list(range(n + 1)))]
        for c in before:
            total += 1
            if tuple(c) in after:
                survived += 1
    return survived, total


def test_split_cluster_old_balanced_split_reshuffled_more_often():
    print("\n[test] characterizing the defect: a growth step invalidates "
          "more already-formed sub-clusters under the OLD balanced split "
          "than under the front-loaded one")
    lo, hi = dedup.MAX_CLUSTER_SIZE + 1, 30
    old_survived, old_total = _one_step_survival(_old_balanced_split, lo, hi)
    new_survived, new_total = _one_step_survival(dedup._split_cluster, lo, hi)
    old_rate = old_survived / old_total
    new_rate = new_survived / new_total
    print(f"  old balanced:  {old_survived}/{old_total} sub-clusters survive "
          f"one growth step ({old_rate:.0%})")
    print(f"  new frontload: {new_survived}/{new_total} sub-clusters survive "
          f"one growth step ({new_rate:.0%})")
    assert_true(new_rate > old_rate,
                 "front-loaded splitting survives strictly more growth "
                 "steps than the balanced split it replaced")


# ---------------------------------------------------------------------------
# End-to-end: the memo actually gets reused for the stable prefix
# ---------------------------------------------------------------------------

def _tight_cluster_vecs(n: int) -> list[list[float]]:
    """n vectors, all mutually far above SIMILARITY_THRESHOLD (one giant
    transitive cluster regardless of n), mirroring N3's observation that
    the store forms blobs well past MAX_CLUSTER_SIZE almost every pass."""
    return [[1.0, i * 0.001] for i in range(n)]


async def _run_growing_blob(conv_id: str, start: int, end: int):
    """Simulate `end - start` passes of a transitive blob that gains one
    member per pass, all-KEEP (the 97.3%-of-clusters case). Returns the
    per-pass count of LLM calls actually issued (memo hits are free and do
    not appear here)."""
    calls_per_pass = []
    facts_so_far = [_fact(f"related-fact-{i}", added_turn=i) for i in range(start)]
    for n in range(start, end):
        facts_so_far.append(_fact(f"related-fact-{n}", added_turn=n))
        client = MagicMock()
        client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
        with patch.object(retrieval, "_embed",
                           lambda texts, _n=len(facts_so_far): _tight_cluster_vecs(_n)), \
             patch.object(dedup, "MAX_LLM_CALLS_PER_PASS", 1000):
            result, removed = await dedup.dedup_facts(
                client, "http://x", "m", facts_so_far, conv_id=conv_id
            )
        calls_per_pass.append(client.post.call_count)
        facts_so_far = result  # unchanged on an all-KEEP pass, but stay honest
    return calls_per_pass


def test_growing_blob_settles_to_one_fresh_call_per_pass():
    print("\n[test] dedup_facts: once the leading chunk fills, growth by "
          "one member costs at most a small, bounded number of fresh "
          "calls per pass — not one per existing sub-cluster")
    dedup.reset_refusal_memo()
    calls = asyncio.run(_run_growing_blob("c-grow-blob", 5, 21))
    print(f"  per-pass fresh LLM calls (n=5..20 growth): {calls}")
    # A single new member can, at worst, force re-asking about the one
    # tail chunk it joined/extended plus a repair-shifted neighbour. It
    # must never scale with the growing TOTAL number of sub-clusters —
    # that scaling is exactly what made deferred clusters balloon in
    # production. Cap generously at 2 to allow for the occasional 1-member
    # borrow-repair touching an adjacent chunk.
    assert_true(max(calls[3:]) <= 2,
                 "steady-state passes cost O(1) fresh calls, not O(chunks)")


def test_call_count_reduction_vs_old_balanced_split():
    print("\n[test] MEASUREMENT: total LLM calls over a 20-pass growing-blob "
          "workload, old balanced split vs. new front-loaded split")
    # Measured honestly (see the report, not asserted here as a target):
    # under an UNCAPPED per-pass budget this is a small, single-digit-percent
    # reduction (a single steadily-growing blob rarely gives the old split
    # room to reshuffle more than one boundary at a time). The larger,
    # measured effect is the deferred-backlog test below, where the call
    # budget is the realistic MAX_LLM_CALLS_PER_PASS default and genuinely
    # new material is competing with reshuffled old material for it. This
    # test's job is narrower and unconditional: prove the new split never
    # costs MORE than the old one on the same workload.
    dedup.reset_refusal_memo()
    with patch.object(dedup, "_split_cluster", _old_balanced_split):
        before = asyncio.run(_run_growing_blob("c-measure-before", 5, 25))
    dedup.reset_refusal_memo()
    after = asyncio.run(_run_growing_blob("c-measure-after", 5, 25))

    total_before = sum(before)
    total_after = sum(after)
    print(f"  BEFORE (old balanced split): {before}  total={total_before}")
    print(f"  AFTER  (new front-loaded):   {after}  total={total_after}")
    reduction = 1 - (total_after / total_before) if total_before else 0.0
    print(f"  reduction: {reduction:.0%}")
    assert_true(total_after <= total_before,
                 "new split never issues more LLM calls than the old one "
                 "on the same workload")


def test_deferred_backlog_shrinks_under_a_saturated_call_budget():
    print("\n[test] MEASUREMENT: with the call budget saturated every pass "
          "(N3's actual regime — 8.65/10 calls/pass, near the cap), the "
          "front-loaded split still reduces the DEFERRED backlog even "
          "though the capped call count itself cannot go lower")
    # 12 independently-clustering topics, 15 new members distributed across
    # them per pass — enough simultaneous genuinely-new material to
    # saturate MAX_LLM_CALLS_PER_PASS=10 on every single pass, matching
    # N3's near-saturated 8.65/10 average and "exceeds the cap" firing on
    # essentially every pass. Under saturation the CALL count is identical
    # by construction (both algorithms spend the whole budget every pass);
    # what differs is how much of that spend goes to genuinely new
    # questions vs. re-litigating a reshuffled old one — visible only in
    # how fast the un-examined backlog grows.
    n_blobs = 12
    growth_per_pass = 15
    passes = 60
    cap = 10

    def vec_for(blob, member_id, dims):
        v = [0.0] * dims
        v[blob] = 1.0
        v[-1] = (member_id % 1000) * 0.0005
        return v

    async def run(splitfn):
        dims = n_blobs + 1
        with patch.object(dedup, "_split_cluster", splitfn):
            blobs = [5] * n_blobs
            deferred_hist = []
            turn = 0
            for p in range(passes):
                for _ in range(growth_per_pass):
                    blobs[turn % n_blobs] += 1
                    turn += 1
                facts, vecs = [], []
                for b in range(n_blobs):
                    for m in range(blobs[b]):
                        facts.append(_fact(f"blob{b}-fact{m}", added_turn=turn))
                        vecs.append(vec_for(b, m, dims))
                client = MagicMock()
                client.post = AsyncMock(return_value=_mock_chat_response("KEEP"))
                stats = {"facts": len(facts), "clusters": 0, "memo_skips": 0,
                          "calls": 0, "merges": 0, "removed": 0, "deferred": 0}
                with patch.object(retrieval, "_embed", lambda texts, v=vecs: v), \
                     patch.object(dedup, "MAX_LLM_CALLS_PER_PASS", cap):
                    await dedup._dedup_pass(
                        client, "http://x", "m", facts, "c-deferred-sim", stats
                    )
                assert_eq(stats["calls"], cap, "the budget really is saturated this pass")
                deferred_hist.append(stats["deferred"])
            return deferred_hist

    dedup.reset_refusal_memo()
    old_deferred = asyncio.run(run(_old_balanced_split))
    dedup.reset_refusal_memo()
    new_deferred = asyncio.run(run(dedup._split_cluster))

    window = 15  # steady state — early passes are still filling the store
    old_sum = sum(old_deferred[-window:])
    new_sum = sum(new_deferred[-window:])
    print(f"  OLD deferred, last {window} passes: {old_deferred[-window:]} sum={old_sum}")
    print(f"  NEW deferred, last {window} passes: {new_deferred[-window:]} sum={new_sum}")
    print(f"  backlog reduction: {100 * (1 - new_sum / old_sum):.0f}%")
    assert_true(new_sum < old_sum * 0.95,
                 "front-loaded splitting leaves a meaningfully smaller "
                 "unexamined backlog under a saturated call budget")


# ---------------------------------------------------------------------------
# No loss of genuine merges
# ---------------------------------------------------------------------------

def test_genuine_merge_still_happens_inside_a_split_cluster():
    print("\n[test] dedup_facts: a real duplicate pair still merges when it "
          "shares a transitive blob with distinct-but-related facts")
    # Six facts, all mutually within threshold (one blob, over the cap of
    # 4, so it gets split). Front-loaded splitting puts the first four
    # (unrelated filler — the related-but-distinct shape N3 measured as
    # the dominant 97.3% case) in one sub-cluster and the last two (a
    # genuine paraphrase pair) in their own — so the LLM is asked about
    # the duplicate pair ALONE, the same as any 2-fact cluster, and a
    # MERGE verdict only ever removes those two, never the filler.
    cluster_facts = [
        _fact("Lyra's sister Wren is a human bard.", added_turn=0),
        _fact("The campaign is set in Aethermere.", added_turn=1),
        _fact("Kestrel carries a longbow.", added_turn=2),
        _fact("The story is currently in autumn.", added_turn=3),
        _fact("Lyra is a half-elf ranger.", added_turn=4),
        _fact("Lyra is a half-elven ranger.", added_turn=5),
    ]
    assert_eq(
        dedup._split_cluster(list(range(6))), [[0, 1, 2, 3], [4, 5]],
        "sanity: the duplicate pair (indices 4,5) is its own sub-cluster",
    )

    async def post(url, json=None, **kw):
        content = json["messages"][0]["content"]
        if "Lyra is a half-elf ranger." in content and \
           "Lyra is a half-elven ranger." in content:
            return _mock_chat_response("Lyra is a half-elf ranger.")
        return _mock_chat_response("KEEP")

    client = MagicMock()
    client.post = AsyncMock(side_effect=post)

    async def go():
        with patch.object(retrieval, "_embed",
                           lambda texts: _tight_cluster_vecs(len(texts))):
            return await dedup.dedup_facts(
                client, "http://x", "m", cluster_facts, conv_id="c-genuine-merge"
            )

    dedup.reset_refusal_memo()
    out, removed = asyncio.run(go())
    assert_true(removed >= 1, "at least one fact was removed by the merge")
    assert_true(
        any(f["text"] == "Lyra is a half-elf ranger." for f in out),
        "the merged canonical fact is present in the result",
    )
    assert_true(
        not any(f["text"] == "Lyra is a half-elven ranger." for f in out),
        "the paraphrase it merged with is gone",
    )
    # And the unrelated filler facts all survive — the split cap still
    # bounds blast radius to the sub-cluster the pair actually landed in.
    for text in (
        "Lyra's sister Wren is a human bard.",
        "The campaign is set in Aethermere.",
        "Kestrel carries a longbow.",
        "The story is currently in autumn.",
    ):
        assert_true(any(f["text"] == text for f in out),
                     f"unrelated fact survives: {text!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_split_cluster_leading_full_chunk_is_stable_as_the_group_grows,
        test_split_cluster_old_balanced_split_reshuffled_more_often,
        test_growing_blob_settles_to_one_fresh_call_per_pass,
        test_call_count_reduction_vs_old_balanced_split,
        test_deferred_backlog_shrinks_under_a_saturated_call_budget,
        test_genuine_merge_still_happens_inside_a_split_cluster,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            dedup.reset_refusal_memo()
            t()
        print("\nAll dedup churn-gate tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
