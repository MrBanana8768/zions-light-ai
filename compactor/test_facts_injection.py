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
    print("\n[test] pin survives eviction into the archive sidecar -> "
          "restore_from_archive")
    # v3.1.9 (hostile pass 4, F8d). archive_stale_facts (the TIME-based
    # sweep) now exempts pinned facts outright -- a pin means "do not
    # remove this by age/pressure alone", the rule _lru_split's own sort
    # already states for the BUDGET-based evictor (prune_facts): pins sort
    # LAST, i.e. they are evicted only once every non-pinned fact is
    # already gone and the budget is STILL too tight. This test used to
    # get its pinned fixture into the archive via archive_stale_facts,
    # which is exactly the route F8d closes -- rewritten to use the real
    # route a pin can still legitimately end up in the archive by: budget
    # eviction so tight that not even the LRU-favoured pinned fact fits
    # (the field note directly above _lru_split's own sort documents this
    # exact shape from production: "a pinned fact with a stale last_used,
    # against 200 fresh facts, was archived").
    _wipe_storage()
    cid = "pin-archive"
    now = int(time.time())
    stale = now - 1_000_000
    pinned = _f("pinned but stale", 0, stale, pin=True)
    unpinned = _f("unpinned and stale", 1, stale, pin=False)
    # body_budget = max_tokens - _FACTS_BLOCK_HEADER_TOKENS = 0: no fact's
    # rendered bullet (always > 0 tokens) can fit, pinned or not. Equal
    # to, not less than, the header floor -- so this is NOT the degenerate
    # "budget too small even for the header" branch (that returns early
    # sorted by added_turn only, bypassing the pin-aware sort/walk this
    # test means to exercise); it is the ordinary path finding it can
    # afford zero facts.
    kept, dropped = facts.prune_facts(
        [pinned, unpinned], max_tokens=facts._FACTS_BLOCK_HEADER_TOKENS,
        conv_id=cid,
    )
    assert_eq(kept, [], "a budget of exactly the header floor fits no fact")
    assert_eq(dropped, 2, "both facts evicted, pinned included")

    sidecar = facts.load_archive(cid)
    by_text = {f["text"]: f for f in sidecar}
    assert_true("pinned but stale" in by_text,
                "the pinned fact reached the sidecar via real budget eviction")
    assert_eq(by_text["pinned but stale"]["pin"], True, "pin preserved in the archive sidecar")
    assert_eq(by_text["unpinned and stale"]["pin"], False, "non-pin preserved too")

    restored = facts.restore_from_archive(cid)
    assert_eq(restored, 2, "both facts restored")
    active = {f["text"]: f for f in facts.load_facts(cid)}
    assert_eq(active["pinned but stale"]["pin"], True,
              "the restored fact is STILL pinned after the round trip")


def test_archive_stale_facts_exempts_pinned_facts_at_older_than_days_0():
    print("\n[test] archive_stale_facts(older_than_days=0) keeps a pinned "
          "fact active; an unpinned one just as stale is still archived "
          "(F8d, hostile pass 4)")
    _wipe_storage()
    cid = "pin-sweep-exempt"
    now = int(time.time())
    stale = now - 1_000_000
    facts.save_facts(cid, [
        _f("pinned and stale", 0, stale, pin=True),
        _f("unpinned and stale", 1, stale, pin=False),
    ])
    kept, archived = facts.archive_stale_facts(cid, older_than_days=0)
    assert_eq(kept, 1, "the pinned fact stays active — the sweep never touches it")
    assert_eq(archived, 1, "CONTROL: the equally-stale UNPINNED fact is still archived")

    active = {f["text"]: f for f in facts.load_facts(cid)}
    assert_true("pinned and stale" in active,
                "the pinned fact is still in the active set")
    assert_true("unpinned and stale" not in active,
                "the unpinned fact left the active set")

    sidecar = {f["text"]: f for f in facts.load_archive(cid)}
    assert_true("pinned and stale" not in sidecar,
                "the pinned fact never reached the sidecar")
    assert_true("unpinned and stale" in sidecar,
                "CONTROL: the unpinned fact did")

    # And it holds at the (unusually generous) 90-day default too, not just
    # at the literal older_than_days=0 the finding's own proof used.
    _wipe_storage()
    cid2 = "pin-sweep-exempt-default"
    ancient = now - (200 * 86400)
    facts.save_facts(cid2, [
        _f("pinned and ancient", 0, ancient, pin=True),
        _f("unpinned and ancient", 1, ancient, pin=False),
    ])
    kept2, archived2 = facts.archive_stale_facts(cid2)  # default 90 days
    assert_eq(kept2, 1, "CONTROL: the pinned fact is exempt at the default cutoff too")
    assert_eq(archived2, 1, "the unpinned fact is still archived at the default cutoff")


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
# Part 2 (continued) — the pin's EVICTION exemption, which is a different
# mechanism from the injection tier above and fails in a different place
# ---------------------------------------------------------------------------

def test_a_stale_pinned_fact_is_not_evicted_by_fresher_unpinned_ones():
    print("\n[test] a pinned fact with an ancient last_used survives eviction "
          "against 20 fresh facts")
    # The exemption lives in _lru_split's sort key, not in a carve-out, and
    # this is the failure it was added for: /pin sets the flag and does NOT
    # touch last_used, and the touch that would refresh it is conditional on
    # the facts layer surviving main._bound_injected_blocks (87 over-budget
    # drops in one production window). So a pinned fact can be both the
    # least-recently-used row in the store and the one row that must never
    # leave it. Before the exemption, the store below archived it: silently no
    # longer injected, recoverable from the sidecar but with nothing anywhere
    # saying so.
    _wipe_storage()
    cid = "pin-eviction"
    now = int(time.time())
    identity = _f("[MISC] her name is Placeholder", 0, 1, pin=True)  # ancient
    fresh = [
        _f(f"[HOME] lorem ipsum house fact {i} about the garden " * 2, i, now + i)
        for i in range(1, 21)
    ]
    store = [identity] + fresh
    # A budget that fits only a handful of the 21, so eviction genuinely has to
    # choose. Without a binding budget this test would pass by keeping
    # everything.
    tight = facts._FACTS_BLOCK_HEADER_TOKENS + 5 * facts._fact_bullet_tokens(
        fresh[0]["text"]
    )
    kept, dropped = facts.prune_facts(store, max_tokens=tight, conv_id=cid)
    assert_true(dropped > 0, f"fixture: the budget really binds ({dropped} evicted)")
    assert_true(
        any(f["text"] == identity["text"] for f in kept),
        "the stale PINNED fact is still in the active store",
    )
    archived = {f["text"] for f in facts.load_archive(cid)}
    assert_true(
        identity["text"] not in archived,
        "and it did not go to the archive sidecar either",
    )
    assert_true(
        any(f["text"] in archived for f in fresh),
        "while unpinned facts — every one of them fresher — did",
    )


def test_the_pin_exemption_does_not_disturb_ordinary_lru_order():
    print("\n[test] with nothing pinned, eviction is exactly LRU as before")
    # The pin term is the highest-order element of the sort key, so it can only
    # ever act as a tie-break above recency. On a store with no pins — every
    # record written before the field existed — the order must be unchanged.
    now = int(time.time())
    cold = _f("[MISC] the coldest fact " * 4, 0, now - 10000)
    warm = [_f(f"[HOME] fact {i} " * 8, i, now + i) for i in range(1, 6)]
    tight = facts._FACTS_BLOCK_HEADER_TOKENS + 3 * facts._fact_bullet_tokens(
        warm[0]["text"]
    )
    kept, dropped = facts.prune_facts([cold] + warm, max_tokens=tight)
    assert_true(dropped > 0, "fixture: the budget binds here too")
    assert_true(
        all(f["text"] != cold["text"] for f in kept),
        "the least-recently-used fact is still the first one evicted",
    )


# ---------------------------------------------------------------------------
# Part 1 (continued) — the settle step, which is what makes the fast
# per-fact-floor walk safe to use as a budget
# ---------------------------------------------------------------------------

def test_the_selected_block_is_settled_against_its_real_rendering():
    print("\n[test] the greedy per-fact-floor walk can overshoot; the settle "
          "step is what brings the block back inside the budget")
    # _fact_bullet_tokens floors each bullet at char/4, and a sum of floors is
    # always <= the floor of the true combined length. Every bullet below is
    # 40 characters, i.e. 43 with its "- " and newline, i.e. 3 characters of
    # remainder each that the per-fact walk throws away — 10 facts, 30 lost
    # characters, and the block the model would actually be sent measures 135
    # estimated tokens against a 128-token budget the approximation believed
    # it had honoured. Without the settle loop, select_for_injection returns
    # an over-budget block while reporting that it fit.
    texts = [
        ("[MISC] synthetic filler fact number %02d" % i).ljust(40, "x")
        for i in range(10)
    ]
    assert_true(all(len(t) == 40 for t in texts), "fixture: 40-char bullets")
    budget = facts._FACTS_BLOCK_HEADER_TOKENS + sum(
        facts._fact_bullet_tokens(t) for t in texts
    )

    def _check(items, label):
        # embedder returning None takes the LRU fallback inside _select_rest,
        # so this measures the budget arithmetic and not a ranking.
        got = facts.select_for_injection(
            items, max_tokens=budget, query_text="a query",
            embedder=lambda _texts: None,
        )
        assert_true(
            len(got) < len(items),
            f"{label}: the settle step actually dropped something "
            f"({len(got)} of {len(items)})",
        )
        assert_true(
            facts._estimate_tokens(facts.format_facts_block(got)) <= budget,
            f"{label}: the block as RENDERED fits the stated budget",
        )

    unpinned = [_f(t, i, 100 + i) for i, t in enumerate(texts)]
    # Fixture guard: the whole set really is over budget as rendered, which is
    # the only reason there is anything for the settle step to do.
    assert_true(
        facts._estimate_tokens(facts.format_facts_block(unpinned)) > budget,
        "fixture: the greedy walk's own budget is genuinely an under-count",
    )
    _check(unpinned, "no pins")

    # And the pinned branch, which reaches the same settle step by a different
    # route (pinned cost paid first, rest given what is left) — this codebase's
    # recurring defect is a fix that lands on one of two paths.
    pinned = [_f(texts[0], 0, 100, pin=True)] + [
        _f(t, i, 100 + i) for i, t in enumerate(texts[1:], start=1)
    ]
    _check(pinned, "one pinned")


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
        test_archive_stale_facts_exempts_pinned_facts_at_older_than_days_0()
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

        test_a_stale_pinned_fact_is_not_evicted_by_fresher_unpinned_ones()
        test_the_pin_exemption_does_not_disturb_ordinary_lru_order()
        test_the_selected_block_is_settled_against_its_real_rendering()

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
