"""
CPU-only tests for compactor/facts.py's v3.1.4 F1 work: decoupling the
STORE cap from the INJECTION budget, top-K relevance ranking of the
injected block, and the pinned always-inject identity tier.

Companion to test_facts.py (which still owns the pre-F1 LRU/archive/
extraction coverage) — kept as its own file per the branch's "new tests are
new files" convention, since this module edits nothing test_facts.py
already owns.

Every fact text below is synthetic lorem-ipsum, tagged with a bracketed
fake "topic" ([HOME]/[HOBBY]/[MISC]) purely so a deterministic mock
embedder can score it — never real conversation content (repo is public).

Run inside the compactor image or any container with the requirements
installed:
    python test_facts_injection.py
"""

import os
import sys
import tempfile
import time
from unittest.mock import patch

# Storage redirect MUST happen before importing memory/facts so the
# module-level paths see the override — same convention as test_facts.py.
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-facts-injection-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import facts  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402


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
        import shutil
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


def _f(text, added_turn, last_used, pin=False):
    return {"text": text, "added_turn": added_turn, "last_used": last_used, "pin": pin}


# ---------------------------------------------------------------------------
# A tiny deterministic "embedder" — no fastembed dependency in the test.
# Vectors are one-hot by bracketed topic tag, so cosine similarity is exactly
# 1.0 for a matching topic and exactly 0.0 otherwise. Real bge-small never
# produces exact ties like this; tests that need an exact ranking outcome
# size the budget tightly enough that the tie doesn't matter (see comments
# at each call site below) rather than pretending the mock is realistic.
# ---------------------------------------------------------------------------

_AXES = {"HOME": [1.0, 0.0, 0.0], "HOBBY": [0.0, 1.0, 0.0], "MISC": [0.0, 0.0, 1.0]}


def _topic_of(text: str) -> str:
    return text.split("]", 1)[0].lstrip("[")


def _mock_embed(texts: list[str]) -> list[list[float]] | None:
    return [_AXES[_topic_of(t)] for t in texts]


# ---------------------------------------------------------------------------
# Part 3 — the store cap and the injection budget are independent knobs
# ---------------------------------------------------------------------------

def test_store_cap_and_injection_cap_are_independent_env_vars():
    print("\n[test] COMPACTOR_MAX_FACTS_TOKENS and COMPACTOR_INJECT_FACTS_TOKENS "
          "are two different constants, not one knob doing two jobs")
    assert_true(
        facts._MAX_FACTS_TOKENS != facts._INJECT_FACTS_TOKENS,
        "store cap and injection cap have different default values",
    )
    assert_eq(facts.prune_facts.__defaults__[0], facts._MAX_FACTS_TOKENS,
              "prune_facts (store cap enforcement) defaults to the STORE budget")
    assert_true(
        facts._INJECT_FACTS_TOKENS < facts._MAX_FACTS_TOKENS,
        "the injection default is the smaller of the two, as F1 asks for "
        "(target ~300-400 tokens vs a much larger store)",
    )


def test_select_for_injection_default_is_the_injection_cap_not_the_store_cap():
    print("\n[test] select_for_injection's own default is COMPACTOR_INJECT_FACTS_TOKENS")
    default = facts.select_for_injection.__kwdefaults__ or {}
    # max_tokens is positional-or-keyword with a default; inspect via the
    # function's __defaults__ (positional defaults) since it's declared
    # before the keyword-only args.
    positional_defaults = facts.select_for_injection.__defaults__
    assert_eq(positional_defaults, (facts._INJECT_FACTS_TOKENS,),
              "select_for_injection's max_tokens default is the injection cap")


# ---------------------------------------------------------------------------
# Part 2 — pin durability across save/load/archive/restore
# ---------------------------------------------------------------------------

def test_pin_round_trips_through_save_and_load():
    print("\n[test] pin=True survives a save_facts -> load_facts round trip")
    _wipe_storage()
    cid = "pin-roundtrip"
    facts.save_facts(cid, [
        _f("her name is Placeholder", 0, 100, pin=True),
        _f("she likes lorem ipsum", 1, 100, pin=False),
    ])
    loaded = facts.load_facts(cid)
    assert_eq(len(loaded), 2, "both facts loaded")
    by_text = {f["text"]: f for f in loaded}
    assert_eq(by_text["her name is Placeholder"]["pin"], True, "pinned fact stays pinned")
    assert_eq(by_text["she likes lorem ipsum"]["pin"], False, "unpinned fact stays unpinned")


def test_legacy_records_without_pin_field_load_as_unpinned():
    print("\n[test] a record written by pre-F1 code (no 'pin' key at all) "
          "loads as pin=False, not an error")
    _wipe_storage()
    cid = "pin-legacy"
    # Write the exact shape save_facts produced before this field existed —
    # no "pin" key anywhere, simulating every one of the 5,341 facts already
    # on disk in production.
    legacy_path = memory.facts_path(cid)
    import json
    legacy_path.write_text(json.dumps({
        "conv_id": cid,
        "updated_at": "2026-05-28T05:00:00Z",
        "facts": [
            {"text": "a fact from before pin existed", "added_turn": 3, "last_used": 500},
        ],
    }))
    loaded = facts.load_facts(cid)
    assert_eq(len(loaded), 1, "the legacy fact loaded")
    assert_eq(loaded[0]["pin"], False, "missing 'pin' key defaults to False, not a crash")
    assert_eq(loaded[0]["text"], "a fact from before pin existed", "text preserved")
    assert_eq(loaded[0]["added_turn"], 3, "added_turn preserved")
    assert_eq(loaded[0]["last_used"], 500, "last_used preserved")

    # And it keeps round-tripping cleanly from here on.
    facts.save_facts(cid, loaded)
    reloaded = facts.load_facts(cid)
    assert_eq(reloaded, loaded, "re-saving a migrated legacy record is stable")


def test_pin_round_trips_through_archive_and_restore():
    print("\n[test] pin survives archive_stale_facts -> restore_from_archive")
    _wipe_storage()
    cid = "pin-archive"
    now = int(time.time())
    stale = now - 1_000_000
    facts.save_facts(cid, [
        _f("pinned but stale", 0, stale, pin=True),
        _f("unpinned and stale", 1, stale, pin=False),
    ])
    kept, archived = facts.archive_stale_facts(cid, older_than_days=1)
    assert_eq(kept, 0, "both facts were stale enough to archive")
    assert_eq(archived, 2, "both moved to the sidecar")

    sidecar = facts.load_archive(cid)
    by_text = {f["text"]: f for f in sidecar}
    assert_eq(by_text["pinned but stale"]["pin"], True, "pin preserved in the archive sidecar")
    assert_eq(by_text["unpinned and stale"]["pin"], False, "non-pin preserved too")

    restored = facts.restore_from_archive(cid)
    assert_eq(restored, 2, "both facts restored")
    active = {f["text"]: f for f in facts.load_facts(cid)}
    assert_eq(active["pinned but stale"]["pin"], True,
              "the restored fact is STILL pinned after the round trip")


def test_set_pinned_sets_and_clears_by_substring():
    print("\n[test] set_pinned() flips the pin flag by case-insensitive substring")
    items = [
        _f("Her Name is Placeholder", 0, 1),
        _f("she likes lorem ipsum hobbies", 1, 1),
        _f("another name-adjacent fact", 2, 1),
    ]
    changed = facts.set_pinned(items, text_substring="name", pinned=True)
    assert_eq(changed, 2, "two facts matched 'name' case-insensitively")
    assert_eq(items[0]["pin"], True, "first match pinned")
    assert_eq(items[2]["pin"], True, "second match pinned")
    assert_eq(items[1]["pin"], False, "non-matching fact untouched")

    changed_again = facts.set_pinned(items, text_substring="name", pinned=True)
    assert_eq(changed_again, 0, "re-pinning an already-pinned fact reports 0 changed")

    unpinned = facts.set_pinned(items, text_substring="Placeholder", pinned=False)
    assert_eq(unpinned, 1, "unpinning one fact by substring")
    assert_eq(items[0]["pin"], False, "it is actually unpinned now")


# ---------------------------------------------------------------------------
# Backward compatibility — "callers that pass nothing get the current
# behaviour" (F1's explicit requirement)
# ---------------------------------------------------------------------------

def test_no_query_no_pins_matches_the_pre_f1_lru_split_exactly():
    print("\n[test] query_text=None + nothing pinned == byte-for-byte the old "
          "_lru_split behaviour (the graceful-degradation contract)")
    items = [
        _f("x" * 100, 1, 100),
        _f("y" * 100, 2, 500),
        _f("z" * 100, 3, 999),
    ]
    for budget in (10, 24, 25, 50, 74, 75, 1000):
        expected, _ = facts._lru_split(list(items), budget)
        got = facts.select_for_injection(list(items), max_tokens=budget)
        assert_eq(
            [f["text"] for f in got], [f["text"] for f in expected],
            f"select_for_injection(query_text=None) == _lru_split at budget={budget}",
        )


def test_no_query_text_still_uses_lru_even_when_embedder_is_available():
    print("\n[test] not passing query_text means no ranking attempt at all, "
          "even with a working embedder sitting right there")
    items = [_f("[HOME] a", 0, 100), _f("[MISC] b", 1, 999)]
    # If this accidentally ranked, the embedder would push [HOME] itself only
    # for a HOME query -- but there IS no query, so LRU (last_used) must
    # decide: item at last_used=999 survives a budget for exactly one.
    budget = facts._FACTS_BLOCK_HEADER_TOKENS + facts._fact_bullet_tokens("[MISC] b") + 1
    got = facts.select_for_injection(items, max_tokens=budget, embedder=_mock_embed)
    assert_eq(len(got), 1, "one fact fits")
    assert_eq(got[0]["text"], "[MISC] b", "LRU (not relevance) decided, because query_text was never given")


# ---------------------------------------------------------------------------
# Part 1 — top-K relevance ranking
# ---------------------------------------------------------------------------

def test_relevant_facts_beat_irrelevant_ones_within_a_tight_budget():
    print("\n[test] a tight budget keeps the query-relevant facts and drops "
          "the irrelevant ones, not the other way round")
    items = [
        _f("[HOME] lorem ipsum home fact one about the house " * 2, 0, 100),
        _f("[HOME] lorem ipsum home fact two about the house " * 2, 1, 200),
        _f("[MISC] lorem ipsum unrelated filler fact one " * 2, 2, 900),  # newest, most recently used
        _f("[MISC] lorem ipsum unrelated filler fact two " * 2, 3, 950),  # newest of all
    ]
    home_only = [f for f in items if f["text"].startswith("[HOME]")]
    # Exactly enough room for the header + both HOME bullets, no slack for a
    # third bullet of any topic -- removes the mock embedder's exact-tie
    # ambiguity between MISC facts entirely.
    tight_budget = facts._estimate_tokens(facts.format_facts_block(home_only))
    got = facts.select_for_injection(
        items, max_tokens=tight_budget, query_text="[HOME] tell me about the house",
        embedder=_mock_embed,
    )
    assert_eq(
        sorted(f["text"] for f in got), sorted(f["text"] for f in home_only),
        "both HOME facts selected, both MISC facts excluded -- despite the "
        "MISC facts being NEWER and more recently used, which is exactly "
        "the axis a pure-LRU/pure-age policy would have picked instead",
    )


def test_ranking_falls_back_to_lru_when_embedder_returns_none():
    print("\n[test] embedder unavailable this turn -> falls back to LRU, chat unaffected")
    items = [_f("[HOME] a", 0, 100), _f("[MISC] b", 1, 999)]
    budget = facts._FACTS_BLOCK_HEADER_TOKENS + facts._fact_bullet_tokens("[MISC] b") + 1
    got = facts.select_for_injection(
        items, max_tokens=budget, query_text="[HOME] anything",
        embedder=lambda texts: None,
    )
    assert_eq(len(got), 1, "one fact fits")
    assert_eq(got[0]["text"], "[MISC] b",
              "LRU decided (most recently used), NOT relevance -- embedder returned None")


def test_ranking_falls_back_to_lru_when_embedder_raises():
    print("\n[test] embedder raising -> falls back to LRU, does not propagate")
    def _boom(texts):
        raise RuntimeError("synthetic embedder failure")
    items = [_f("[HOME] a", 0, 100), _f("[MISC] b", 1, 999)]
    budget = facts._FACTS_BLOCK_HEADER_TOKENS + facts._fact_bullet_tokens("[MISC] b") + 1
    got = facts.select_for_injection(
        items, max_tokens=budget, query_text="[HOME] anything", embedder=_boom,
    )
    assert_eq(got[0]["text"], "[MISC] b", "LRU fallback survived the embedder raising")


def test_ranking_falls_back_when_vector_count_mismatches():
    print("\n[test] an embedder returning the wrong number of vectors is "
          "treated as unavailable, not trusted partially")
    items = [_f("[HOME] a", 0, 100), _f("[MISC] b", 1, 999)]
    budget = facts._FACTS_BLOCK_HEADER_TOKENS + facts._fact_bullet_tokens("[MISC] b") + 1
    got = facts.select_for_injection(
        items, max_tokens=budget, query_text="[HOME] anything",
        embedder=lambda texts: [[1.0, 0.0]],  # only 1 vector for query+2 facts
    )
    assert_eq(got[0]["text"], "[MISC] b", "LRU fallback used on a malformed embedder result")


def test_default_embedder_wiring_reaches_retrieval_module():
    print("\n[test] with no embedder= override, select_for_injection reaches "
          "retrieval._embed -- the actual production wiring")
    items = [
        _f("[HOME] lorem ipsum home fact " * 2, 0, 100),
        _f("[MISC] lorem ipsum misc fact " * 2, 1, 900),
    ]
    home_only = [f for f in items if f["text"].startswith("[HOME]")]
    tight_budget = facts._estimate_tokens(facts.format_facts_block(home_only))
    with patch.object(retrieval, "_embed", _mock_embed):
        got = facts.select_for_injection(
            items, max_tokens=tight_budget, query_text="[HOME] anything",
        )
    assert_eq([f["text"] for f in got], [f["text"] for f in home_only],
              "select_for_injection used retrieval._embed via facts.retrieval_module, unprompted")


# ---------------------------------------------------------------------------
# Part 2 (continued) — the pinned tier bypasses ranking and the budget
# ---------------------------------------------------------------------------

def test_pinned_fact_survives_a_query_it_has_nothing_to_do_with():
    print("\n[test] pure top-K would drop an off-topic identity fact -- pin "
          "keeps it in regardless (the 'she forgot me' failure this exists to stop)")
    identity = _f("[MISC] her name is Placeholder", 0, 1, pin=True)
    items = [identity] + [
        _f(f"[HOME] lorem ipsum home fact {i} about the house " * 2, i, 100 + i)
        for i in range(1, 4)
    ]
    got = facts.select_for_injection(
        items, max_tokens=1000, query_text="[HOME] tell me about the house",
        embedder=_mock_embed,
    )
    assert_true(
        any(f["text"] == identity["text"] for f in got),
        "the identity fact is present even though the query is entirely about HOME, "
        "not MISC, and pure ranking would have scored it 0.0",
    )


def test_pinned_facts_always_included_even_over_budget():
    print("\n[test] pinned facts alone exceeding the injection budget still "
          "all get injected -- losing an identity fact is worse than overshooting a soft cap")
    pinned = [_f(f"[HOME] pinned identity fact number {i} " * 3, i, 1, pin=True) for i in range(5)]
    tiny_budget = facts._FACTS_BLOCK_HEADER_TOKENS + 5  # far too small for 5 long pinned bullets
    got = facts.select_for_injection(pinned, max_tokens=tiny_budget)
    assert_eq(len(got), 5, "all 5 pinned facts injected despite the budget")
    assert_true(
        facts._estimate_tokens(facts.format_facts_block(got)) > tiny_budget,
        "the resulting block genuinely exceeds the nominal budget -- confirms this "
        "path did not silently drop a pinned fact to make the numbers fit",
    )


def test_pin_tier_reserves_room_so_rest_still_fits_the_stated_budget():
    print("\n[test] when the pinned tier + selected rest both fit, the WHOLE "
          "combined block still respects max_tokens (only over-budget pin-only case doesn't)")
    identity = _f("[MISC] her name is Placeholder", 0, 1, pin=True)
    rest = [_f(f"[HOME] home fact {i} about the house and the garden " * 2, i, 100 + i)
            for i in range(1, 6)]
    budget = 300
    got = facts.select_for_injection(
        [identity] + rest, max_tokens=budget, query_text="[HOME] about the house",
        embedder=_mock_embed,
    )
    assert_true(any(f["text"] == identity["text"] for f in got), "identity fact present")
    assert_true(
        facts._estimate_tokens(facts.format_facts_block(got)) <= budget,
        "the combined pinned+ranked block fits the stated budget when it is not "
        "pinned-facts-alone that overshoots",
    )


def test_touching_the_injected_set_touches_the_pinned_facts_too():
    print("\n[test] pinned facts are part of what gets touched -- this is what "
          "keeps them out of LRU eviction with NO special case in prune_facts")
    now = int(time.time())
    identity = _f("[MISC] her name is Placeholder", 0, now - 100000, pin=True)  # very stale last_used
    rest = [_f(f"[HOME] home fact {i} " * 3, i, now) for i in range(1, 4)]
    store = [identity] + rest
    injected = facts.select_for_injection(
        store, max_tokens=1000, query_text="[HOME] about the house", embedder=_mock_embed,
    )
    facts.touch_facts(injected, now=now + 500)
    assert_eq(identity["last_used"], now + 500,
              "the pinned fact -- despite an ancient last_used going in -- was touched, "
              "because it was part of the injected (and therefore touched) set")


# ---------------------------------------------------------------------------
# End-to-end measurement — before/after injection size, and LRU tracking
# relevance rather than age across several simulated turns (N3/F1's claim,
# reproduced on synthetic data at the ~80-fact scale N3 measured)
# ---------------------------------------------------------------------------

def test_measured_injection_size_before_and_after():
    print("\n[test] MEASURED: injected block size, pre-F1 shape vs F1 top-K+pin")
    _wipe_storage()
    cid = "measure-size"
    now = int(time.time())
    store = [_f("[MISC] her name is Placeholder, this is her identity fact", 0, now, pin=True)]
    store += [_f(f"[HOME] lorem ipsum dolor sit amet home fact number {i}", i, now)
              for i in range(1, 6)]
    store += [_f(f"[MISC] lorem ipsum dolor sit amet filler fact number {i}", i, now)
              for i in range(6, 80)]
    assert_eq(len(store), 80, "prep: 80 facts, matching N3's measured ~80 active facts")
    facts.save_facts(cid, store)
    on_disk = facts.load_facts(cid)

    # BEFORE: the literal pre-F1 call shape -- the whole store against the
    # combined 1500-token cap this module used for both jobs at once.
    before = facts._lru_split(on_disk, 1500)[0]
    before_tokens = facts._estimate_tokens(facts.format_facts_block(before))
    print(f"  MEASURED before (whole store, old combined 1500-tok cap): "
          f"{len(before)} facts / ~{before_tokens} tokens")
    assert_eq(len(before), 80, "pre-F1 injects the entire active set")

    # AFTER: top-K + pin against the new, independent injection default.
    after = facts.select_for_injection(
        on_disk, query_text="[HOME] anything about home",
        embedder=_mock_embed,
    )
    after_tokens = facts._estimate_tokens(facts.format_facts_block(after))
    print(f"  MEASURED after  (top-K + pin, new {facts._INJECT_FACTS_TOKENS}-tok cap): "
          f"{len(after)} facts / ~{after_tokens} tokens")
    assert_true(after_tokens <= facts._INJECT_FACTS_TOKENS,
                "the new injection fits the new, much smaller default budget")
    assert_true(len(after) < len(before),
                "MEASURED: fewer facts injected than the pre-F1 whole-store behaviour")
    # The PROPERTY, not a percentage tied to one default. F1's objective is
    # that injection is meaningfully smaller than the STORE cap, because that
    # gap is what makes `last_used` carry signal and stops LRU collapsing to
    # FIFO. Any injection cap below the store cap achieves it; the exact
    # ratio is an operational dial (COMPACTOR_INJECT_FACTS_TOKENS), and it
    # moved 400 -> 800 on review: the latency case for 400 was near-zero
    # (those tokens are prefill on a turn whose p50 is 95s) while the cost of
    # a ranking miss is this user's standing complaint, forgetting. Asserting
    # "shrank 60%" pinned the test to a dial rather than to the goal.
    assert_true(facts._INJECT_FACTS_TOKENS < facts._MAX_FACTS_TOKENS,
                f"the injection cap ({facts._INJECT_FACTS_TOKENS}) is below the "
                f"store cap ({facts._MAX_FACTS_TOKENS}) - the gap is what gives "
                f"last_used meaning")
    assert_true(after_tokens < before_tokens,
                f"MEASURED: injected token size shrank "
                f"({before_tokens} -> {after_tokens})")


def test_lru_now_selects_by_relevance_not_age_after_several_turns():
    print("\n[test] MEASURED: after several turns of on-topic queries, eviction "
          "keeps the OLD-but-relevant facts and drops the NEW-but-irrelevant "
          "ones -- N3's 'LRU degenerates to FIFO by added_turn' failure, reversed")
    _wipe_storage()
    cid = "measure-lru-relevance"
    now = int(time.time())

    # HOME facts are the OLDEST in the store (added_turn 0-2) -- exactly what
    # a pure-age/FIFO policy evicts FIRST. MISC facts are added LATER
    # (added_turn 3-79) and are never once relevant to any turn below --
    # exactly what a pure-age/FIFO policy would PROTECT, wrongly.
    home = [_f(f"[HOME] home fact {i} about the house " * 2, i, now) for i in range(3)]
    misc = [_f(f"[MISC] filler fact {i} " * 2, i, now) for i in range(3, 80)]
    facts.save_facts(cid, home + misc)

    # Five turns, every one of them a HOME-relevant query, tight injection
    # budget so ranking (not accidental ties) decides what's touched. Sized
    # off the REAL combined render (not a sum of per-fact floors) so it is
    # exactly enough for the 3 HOME facts and nothing else, with no
    # floor-rounding ambiguity at the boundary.
    sim_budget = facts._estimate_tokens(facts.format_facts_block(home))
    for i in range(5):
        on_disk = facts.load_facts(cid)
        injected = facts.select_for_injection(
            on_disk, max_tokens=sim_budget,
            query_text="[HOME] tell me about the house", embedder=_mock_embed,
        )
        facts.touch_facts(injected, now=now + 1000 + i)
        facts.save_facts(cid, on_disk)

    store_after_turns = facts.load_facts(cid)
    touched = {f["text"] for f in store_after_turns if f["last_used"] > now}
    print(f"  MEASURED: {len(touched)}/{len(store_after_turns)} facts touched "
          f"across 5 HOME-relevant turns")
    assert_eq(touched, {f["text"] for f in home},
              "only the 3 HOME facts were ever touched -- the 77 MISC facts, "
              "despite being newer, were never once relevant")

    # Now enforce the STORE cap (eviction, unrelated to injection). A
    # pre-F1 store (touch-everything every turn) would have every fact's
    # last_used pinned to the same instant, collapsing this sort onto
    # added_turn and evicting the HOME facts FIRST (they're the oldest).
    tight_store_budget = sim_budget
    kept, dropped = facts.prune_facts(store_after_turns, max_tokens=tight_store_budget, conv_id=cid)
    kept_topics = sorted({_topic_of(f["text"]) for f in kept})
    print(f"  MEASURED: pruned to {len(kept)}/{len(store_after_turns)} facts "
          f"(dropped {dropped}); topics kept = {kept_topics}")
    assert_eq(kept_topics, ["HOME"],
              "eviction kept the OLDEST facts in the store because they were the ones "
              "actually used -- the exact opposite of the FIFO-by-added_turn failure "
              "N3 measured (5,341 extracted / 3,714 evicted / 70%, selecting for nothing but age)")
    # The evicted facts are still recoverable -- eviction/archival policy is
    # unchanged by F1.
    archived_texts = {f["text"] for f in facts.load_archive(cid)}
    assert_eq(archived_texts, {f["text"] for f in misc},
              "every evicted MISC fact landed in the archive sidecar, not deleted")


if __name__ == "__main__":
    try:
        test_store_cap_and_injection_cap_are_independent_env_vars()
        test_select_for_injection_default_is_the_injection_cap_not_the_store_cap()

        test_pin_round_trips_through_save_and_load()
        test_legacy_records_without_pin_field_load_as_unpinned()
        test_pin_round_trips_through_archive_and_restore()
        test_set_pinned_sets_and_clears_by_substring()

        test_no_query_no_pins_matches_the_pre_f1_lru_split_exactly()
        test_no_query_text_still_uses_lru_even_when_embedder_is_available()

        test_relevant_facts_beat_irrelevant_ones_within_a_tight_budget()
        test_ranking_falls_back_to_lru_when_embedder_returns_none()
        test_ranking_falls_back_to_lru_when_embedder_raises()
        test_ranking_falls_back_when_vector_count_mismatches()
        test_default_embedder_wiring_reaches_retrieval_module()

        test_pinned_fact_survives_a_query_it_has_nothing_to_do_with()
        test_pinned_facts_always_included_even_over_budget()
        test_pin_tier_reserves_room_so_rest_still_fits_the_stated_budget()
        test_touching_the_injected_set_touches_the_pinned_facts_too()

        test_measured_injection_size_before_and_after()
        test_lru_now_selects_by_relevance_not_age_after_several_turns()

        print("\nALL PASS (test_facts_injection.py)")
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL (uncaught exception): {e}")
        sys.exit(1)
