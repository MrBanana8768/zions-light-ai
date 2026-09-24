"""
CPU-only Tier-1 tests for the `.15p` identity guard (v3.1.9.8, §17.6 r9/r10;
commit 7650668, main.py ~7025-7054, 7657, 7756, 11447-11551). No dedicated
suite covered this before now — test_budget_guard.py exercises the
POSITIONAL `protect_system` guard at length, but only one line
(test_call_site_passes_the_callers_system_count) even mentions
`_caller_system_hashes`, and nothing in this repo called
`_droppable_system_indices`, `_is_known_injected_system_block` or the
`caller_system_rewritten` fallback branch directly.

Covers, in order:

  1. the caller's own system message(s), protected BY HASH, survive the
     hard-budget guard wherever they sit and however many there are —
     the whole point of moving from position to identity (r9 HIGH-A);
  2. system blocks the compactor itself injects (persona, facts,
     retrieval, hierarchical summary, the `.15g` band, the compaction
     stand-in) are droppable when their hash is not in the protected set;
  3. the reviewer-flagged edge: her OWN system prompt happens to CONTAIN
     the literal text of one of those injected headers (she is free to
     write about the assistant's memory mechanism in her own system
     prompt). In the `caller_system_rewritten` fallback path (r10, LOW),
     content-based matching (first `_is_known_injected_system_block` by
     substring; then, round 1 of this file's own fix, "does it still
     match a hash from request entry") both broke the moment a pre-guard
     transform edited a message that also happened to contain — or,
     after the edit, still contained — one of our header strings (round-2
     review's `expA` probes: a merge that appends an injected facts block
     onto her own prompt, and a bare whitespace-strip of a prompt that
     quotes a header; both lost her content under either approach even
     though it was never ours to spend). Section 3 proves the round-2
     fix: `_enforce_hard_budget`'s new `injected_system_hashes` parameter
     — hashes of the blocks THIS request's caller (chat_completions)
     itself injected, recorded at the moment each was created — is the
     ONLY thing the fallback trusts to tell ours from hers; content is
     never consulted for the drop decision again.
  4. the WARNING a direct (positional-int) caller draws is logged once per
     PROCESS via logsetup.log_once, not once per call/request;
  5. edge inputs: an empty frozenset (nothing protected — everything
     droppable) and a caller who sent zero system messages at all
     (`_caller_system_hashes` on a system-less array is the empty set,
     which composes the same way).

Run: python test_identity_guard.py
"""

import logging
import os
import sys
import tempfile

# Small, predictable window, matching test_budget_guard.py's config so the
# fixtures below (char/4 estimator, no MODEL_REPO) are directly comparable.
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"
os.environ["COMPACTOR_GENERATION_RESERVE"] = "200"   # HARD_INPUT_LIMIT = 800
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "100"
os.environ["COMPACTOR_SUMMARY_INPUT_RESERVE"] = "100"
os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "4"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-identity-guard-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_CONTROL_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-identity-guard-control-")

import facts  # noqa: E402
import logsetup  # noqa: E402
import main  # noqa: E402
import persona  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402
import continuity  # noqa: E402

# Same R22 rationale as test_budget_guard.py: nothing here points VLLM_URL
# at a live server, so an unstubbed /tokenize call is a dead TCP handshake
# stacked on every test. Fail it fast instead of waiting the ~4.2s out.
_real_httpx_post = main.httpx.post


def _fail_tokenize_fast(url, *args, **kwargs):
    if "/tokenize" in url:
        raise main.httpx.ConnectError(
            "connection refused (stubbed for test speed — see R22)"
        )
    return _real_httpx_post(url, *args, **kwargs)  # pragma: no cover


main.httpx.post = _fail_tokenize_fast

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None


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


def big(role, approx_tokens):
    # char/4 estimator -> 4 chars ~= 1 token
    return {"role": role, "content": "x" * (approx_tokens * 4)}


def sysmsg(text):
    return {"role": "system", "content": text}


class _CaptureLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _with_logs(fn, *a, **kw):
    """Run `fn(*a, **kw)` with main's logger tapped. -> (result, records)."""
    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    try:
        result = fn(*a, **kw)
    finally:
        lg.removeHandler(handler)
    return result, handler.records


def _systems(msgs):
    return [m for m in msgs if m.get("role") == "system"]


# ---------------------------------------------------------------------------
# 1. Identity protection: hash beats position.
# ---------------------------------------------------------------------------

def test_caller_hash_protects_regardless_of_position():
    print("\n[test] _droppable_system_indices — caller system message protected by hash, any position")
    caller = sysmsg("HER-SENTINEL " + "h" * 400)
    injected_a = sysmsg(facts._FACTS_BLOCK_HEADER + " fact-sentinel")
    injected_b = sysmsg(persona._PERSONA_BLOCK_HEADER + " persona-sentinel")

    # Caller's message sandwiched between two injected blocks — position
    # alone (the pre-r9 `protect_system: int` contract) would misprotect
    # here; identity must not care where it sits.
    msgs = [injected_a, caller, injected_b]
    protect = main._caller_system_hashes([caller])
    droppable = main._droppable_system_indices(msgs, protect)

    assert_eq(set(droppable), {0, 2}, "only the two injected blocks are droppable")
    assert_true(1 not in droppable, "the caller's message (index 1) is protected")


def test_caller_hash_protects_every_duplicate():
    print("\n[test] _droppable_system_indices — every copy of a duplicated caller message is protected")
    caller = sysmsg("DUP-SENTINEL " + "d" * 400)
    injected_a = sysmsg(facts._FACTS_BLOCK_HEADER + " fact-sentinel")
    msgs = [caller, injected_a, dict(caller)]  # same content, sent (or duplicated) twice
    protect = main._caller_system_hashes([caller])
    droppable = main._droppable_system_indices(msgs, protect)
    assert_eq(droppable, [1], "only the injected block is droppable; both copies of hers survive")


def test_caller_messages_survive_the_full_guard_wherever_they_sit():
    print("\n[test] _enforce_hard_budget — identity protection holds end to end under real shedding")
    caller = sysmsg("END-TO-END-SENTINEL " + "e" * 300)
    protect = main._caller_system_hashes([caller])
    msgs = (
        [sysmsg(facts._FACTS_BLOCK_HEADER + "m" * 500), caller,
         sysmsg(persona._PERSONA_BLOCK_HEADER + "m" * 500)]
        + [big("user", 2000)]
    )
    before = main.count_tokens(msgs)
    assert_true(before > main.HARD_INPUT_LIMIT, f"starts over budget ({before})")
    out = main._enforce_hard_budget(msgs, None, protect)
    sys_out = _systems(out)
    assert_true(
        any(m["content"] == caller["content"] for m in sys_out),
        "the caller's system message survived real shedding, byte-for-byte",
    )
    assert_true(
        not any(facts._FACTS_BLOCK_HEADER in m["content"] for m in sys_out)
        or not any(persona._PERSONA_BLOCK_HEADER in m["content"] for m in sys_out),
        "at least one injected block was actually spent (fixture is over budget by design)",
    )


# ---------------------------------------------------------------------------
# 2. Injected blocks are droppable.
# ---------------------------------------------------------------------------

def test_all_known_injected_headers_are_droppable_by_default():
    print("\n[test] _droppable_system_indices — every known injected header is droppable with an empty protected set")
    blocks = [
        sysmsg(main.COMPACTION_SUMMARY_HEADER + " x"),
        sysmsg(persona._PERSONA_BLOCK_HEADER + " x"),
        sysmsg(facts._FACTS_BLOCK_HEADER + " x"),
        sysmsg(retrieval._RETRIEVAL_BLOCK_HEADER + " x"),
        sysmsg(summarizer._BLOCK_HEADER + " x"),
        sysmsg(continuity.BAND_HEADER + " x"),
    ]
    for b in blocks:
        assert_true(main._is_known_injected_system_block(b), f"recognised: {b['content'][:40]!r}...")
    droppable = main._droppable_system_indices(blocks, frozenset())
    assert_eq(droppable, list(range(len(blocks))), "every injected block is droppable")


def test_injected_blocks_droppable_when_not_in_protected_set():
    print("\n[test] _droppable_system_indices — injected blocks not in the identity set are droppable")
    caller = sysmsg("PROTECTED-SENTINEL " + "p" * 300)
    injected = sysmsg(facts._FACTS_BLOCK_HEADER + "m" * 500)
    msgs = [caller, injected]
    protect = main._caller_system_hashes([caller])
    assert_eq(main._droppable_system_indices(msgs, protect), [1], "only the injected block")


# ---------------------------------------------------------------------------
# 3. The reviewer-flagged edge: her prompt CONTAINS an injected header's text.
# ---------------------------------------------------------------------------

def test_header_collision_is_protected_by_identity_not_content():
    print("\n[test] caller_system_rewritten fallback — her prompt quoting an "
          "injected header stays protected (identity, via injected_system_hashes)")
    # Two caller system messages, exactly as a real client can send:
    #   [0] an ordinary persona/system prompt (will be "edited" upstream —
    #       e.g. a future reorder/rewrite between request entry and the
    #       guard — so its hash no longer matches what was recorded)
    #   [1] a system prompt that happens to quote/discuss the wording of
    #       one of the blocks we inject (facts). Never edited: byte-
    #       identical to what was recorded at request entry.
    original_zero = sysmsg("Be a helpful, careful assistant." + "z" * 100)
    collision = sysmsg(
        "Remember: you will sometimes see a block that begins "
        + repr(facts._FACTS_BLOCK_HEADER)
        + " — that is injected memory, not something I wrote."
        + "c" * 100
    )
    assert_true(
        main._is_known_injected_system_block(collision),
        "fixture check: the collision message DOES match the facts header "
        "substring (which is exactly why content alone cannot be trusted)",
    )

    entry_messages = [original_zero, collision]
    protect = main._caller_system_hashes(entry_messages)
    assert_eq(len(protect), 2, "both her system messages were recorded at request entry")

    # Simulate the upstream rewrite of message [0] only. Message [1] (the
    # collision) reaches the guard byte-for-byte unchanged. Nothing in
    # this request actually injected anything, so injected_system_hashes
    # is empty — the fallback's whole universe of "current system
    # messages minus injected_system_hashes" comes out as "everything",
    # which is the maximally safe answer when this call registers no
    # injection at all.
    rewritten_zero = sysmsg("Be a helpful, careful assistant (REWRITTEN)." + "z" * 100)
    guard_input = [rewritten_zero, collision, {"role": "user", "content": "hi"}]

    out, records = _with_logs(main._enforce_hard_budget, guard_input, None, protect)

    assert_true(
        any("caller_system_rewritten" in r.getMessage() for r in records),
        "the assertion fired (message [0]'s hash is genuinely missing)",
    )
    sys_out = _systems(out)
    assert_true(
        any(m["content"] == collision["content"] for m in sys_out),
        "BUG CHECK: her unedited system prompt (which happens to quote an "
        "injected header) must still be protected in the fallback — if this "
        "fails, the fallback is guessing from content again",
    )
    assert_true(
        any(m["content"] == rewritten_zero["content"] for m in sys_out),
        "the edited message itself is also never dropped: it was never "
        "registered as OUR injection, so it defaults to protected",
    )


def test_header_collision_edge_survives_real_shedding_pressure():
    print("\n[test] caller_system_rewritten fallback — the collision survives "
          "even when the guard is starved and must shed everything it can")
    original_zero = sysmsg("System prompt zero." + "z" * 50)
    collision = sysmsg(
        summarizer._BLOCK_HEADER + " -- she is discussing this exact header " + "c" * 50
    )
    entry_messages = [original_zero, collision]
    protect = main._caller_system_hashes(entry_messages)

    rewritten_zero = sysmsg("System prompt zero (edited)." + "z" * 50)
    facts_block = sysmsg(facts._FACTS_BLOCK_HEADER + "m" * 600)
    # THIS is what registers the facts block as genuinely ours — exactly
    # what chat_completions now does at inject_system_block's call site
    # (main.py: `_injected_system_hashes.add(_system_msg_hash(...))`).
    # Without this, the round-2 fix's own default (empty) would protect
    # the facts block too — correct but not what this test is checking.
    injected_hashes = frozenset({main._system_msg_hash(facts_block)})

    # A huge user turn plus real injected memory so the guard has every
    # reason to look for something to shed.
    guard_input = [rewritten_zero, collision, facts_block, big("user", 3000)]
    out, records = _with_logs(
        main._enforce_hard_budget, guard_input, None, protect,
        injected_system_hashes=injected_hashes,
    )
    assert_true(
        any("caller_system_rewritten" in r.getMessage() for r in records),
        "fallback engaged",
    )
    sys_out = _systems(out)
    assert_true(
        any(m["content"] == collision["content"] for m in sys_out),
        "her collision message survives even under real shedding pressure",
    )
    assert_true(
        any(m["content"] == rewritten_zero["content"] for m in sys_out),
        "the edited-but-unregistered message survives too",
    )
    assert_true(
        not any(m["content"].startswith(facts._FACTS_BLOCK_HEADER) for m in sys_out),
        "meanwhile the genuinely injected (and REGISTERED) facts block WAS "
        "available to spend",
    )


def test_expA_case1_merge_appends_injected_block_onto_her_prompt():
    print("\n[test] round-2 expA case 1 — a pre-guard MERGE appends an "
          "injected facts block onto her prompt; her text still survives")
    # round-2 review's exact reproduction (scratchpad/probe.py CASE1): a
    # transform between request entry and the guard concatenates her
    # system message with an injected facts block into ONE system
    # message. Her original hash is now missing (the merged content
    # hashes differently) — round 1's fix fell back to a header-substring
    # check on the MERGED text, saw the facts header inside it, and
    # dropped the whole thing, her sentinel included.
    her = sysmsg("HER PROMPT: you are Lyra. " + "h" * 300)
    protect = main._caller_system_hashes([her])
    merged = sysmsg(her["content"] + "\n\n" + facts._FACTS_BLOCK_HEADER + " fact " + "m" * 300)
    guard_input = [merged, big("user", 2000)]

    # injected_system_hashes intentionally does NOT contain merged's hash:
    # nothing in THIS call actually created `merged` as an injection — a
    # real request would never merge her prompt with an injected block in
    # the first place, which is exactly why this content cannot be
    # trusted as ours just because a header substring appears inside it.
    out = main._enforce_hard_budget(guard_input, None, protect)
    sys_out = _systems(out)
    assert_true(
        any("HER PROMPT" in m["content"] for m in sys_out),
        "her prompt's text survives even merged into a message that also "
        "contains a known injected header",
    )


def test_expA_case2_whitespace_edit_of_a_header_quoting_prompt():
    print("\n[test] round-2 expA case 2 — a bare whitespace-strip edit of a "
          "header-quoting prompt; her text still survives")
    # round-2 review's exact reproduction (scratchpad/probe.py CASE2): her
    # prompt quotes a header (legitimately — see test_header_collision_
    # is_protected_by_identity_not_content above) AND gets any edit at
    # all, even one that changes nothing semantically. Round 1's fix
    # matched "still identical to the hash recorded at entry" byte-for-
    # byte, so ANY edit — not just a hostile merge — fell through to the
    # same broken content heuristic.
    q = sysmsg(
        "  Note: blocks starting " + facts._FACTS_BLOCK_HEADER + " are memory. "
        + "q" * 300 + "  "
    )
    protect = main._caller_system_hashes([q])
    edited = sysmsg(q["content"].strip())
    guard_input = [edited, big("user", 2000)]

    out = main._enforce_hard_budget(guard_input, None, protect)
    sys_out = _systems(out)
    assert_true(
        any("Note: blocks" in m["content"] for m in sys_out),
        "her edited-but-still-hers prompt survives a whitespace-only edit",
    )


# ---------------------------------------------------------------------------
# 4. The positional-int WARNING: once per process, not once per request.
# ---------------------------------------------------------------------------

def test_positional_int_warning_is_once_per_process_not_per_call():
    print("\n[test] _droppable_system_indices — positional-int WARNING is log_once, not per-call")
    logsetup._reset_log_once_for_tests()
    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    try:
        msgs = [sysmsg("a"), sysmsg("b"), {"role": "user", "content": "hi"}]
        for _ in range(5):
            main._droppable_system_indices(msgs, 1)
    finally:
        lg.removeHandler(handler)

    warnings = [
        r for r in handler.records
        if r.levelno == logging.WARNING and "protect_system count" in r.getMessage()
    ]
    assert_eq(len(warnings), 1, "exactly one WARNING across 5 positional-int calls")

    # Confirm it is genuinely logsetup.log_once (keyed by call site, not by
    # message content or conv_id) rather than some other per-call gate that
    # happens to look similar — the key main.py actually uses.
    assert_true(
        not logsetup.log_once("droppable_system_indices.positional"),
        "the log_once key main.py uses is already spent for this process",
    )


def test_positional_int_warning_key_is_call_site_not_per_request():
    print("\n[test] _droppable_system_indices — the log_once key has no per-request/conv component")
    logsetup._reset_log_once_for_tests()
    # Two calls that would carry DIFFERENT conv_ids/requests in production
    # (nothing here ties the key to either) must still only warn once,
    # proving the gate is process-wide, not per-conversation.
    main._droppable_system_indices([sysmsg("x")], 0)
    main._droppable_system_indices([sysmsg("y"), sysmsg("z")], 1)
    assert_true(
        not logsetup.log_once("droppable_system_indices.positional"),
        "one shared key regardless of which array/count was passed",
    )


# ---------------------------------------------------------------------------
# 5. Edge inputs: empty frozenset, zero caller system messages.
# ---------------------------------------------------------------------------

def test_empty_frozenset_means_nothing_protected():
    print("\n[test] _droppable_system_indices — an empty frozenset protects nothing")
    msgs = [sysmsg("one"), sysmsg("two"), {"role": "user", "content": "hi"}]
    droppable = main._droppable_system_indices(msgs, frozenset())
    assert_eq(droppable, [0, 1], "both system messages droppable — the short-circuit in the docstring")


def test_zero_caller_system_messages_is_the_empty_set():
    print("\n[test] _caller_system_hashes — a caller with no system messages at all")
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    hashes = main._caller_system_hashes(messages)
    assert_eq(hashes, frozenset(), "no system messages -> empty identity set")
    assert_true(isinstance(hashes, frozenset), "still the declared type, not a bare set or None")

    # Composes correctly with _droppable_system_indices: an array that later
    # gains injected system messages (compaction/injection) treats all of
    # them as droppable, same as passing frozenset() directly.
    later = [sysmsg(facts._FACTS_BLOCK_HEADER + "x"), sysmsg(persona._PERSONA_BLOCK_HEADER + "y")] + messages
    assert_eq(
        main._droppable_system_indices(later, hashes), [0, 1],
        "every system message injected later is droppable when she sent none",
    )


def test_empty_frozenset_and_zero_system_messages_agree():
    print("\n[test] frozenset() and _caller_system_hashes([]) are interchangeable inputs")
    msgs = [sysmsg("a"), sysmsg("b")]
    assert_eq(
        main._droppable_system_indices(msgs, frozenset()),
        main._droppable_system_indices(msgs, main._caller_system_hashes([])),
        "an explicit empty set and the hashes of a system-less array behave identically",
    )


if __name__ == "__main__":
    test_caller_hash_protects_regardless_of_position()
    test_caller_hash_protects_every_duplicate()
    test_caller_messages_survive_the_full_guard_wherever_they_sit()
    test_all_known_injected_headers_are_droppable_by_default()
    test_injected_blocks_droppable_when_not_in_protected_set()
    test_header_collision_is_protected_by_identity_not_content()
    test_header_collision_edge_survives_real_shedding_pressure()
    test_expA_case1_merge_appends_injected_block_onto_her_prompt()
    test_expA_case2_whitespace_edit_of_a_header_quoting_prompt()
    test_positional_int_warning_is_once_per_process_not_per_call()
    test_positional_int_warning_key_is_call_site_not_per_request()
    test_empty_frozenset_means_nothing_protected()
    test_zero_caller_system_messages_is_the_empty_set()
    test_empty_frozenset_and_zero_system_messages_agree()
    print("\nAll identity guard tests passed.")
