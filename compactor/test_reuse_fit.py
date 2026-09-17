"""v3.1.9.1 — reuse must actually fire at PRODUCTION numbers.

v3.1.9 went live 2026-09-16 11:06Z. On one conversation 37 of 37 requests
logged "compaction skipped: 806 turns need 45 summarization calls, over the
4-call per-request cap" — the exact failure v3.1.9 was built to remove. The
line just before it every time:

    summary block: dropped 3 tier item(s) to fit the 1846-token block budget
    ... kept 1/4 chapter(s), 1/1 scene(s) and the caller asked for
    all-or-nothing, so NOTHING is returned ...
    the stored summaries cover 792 of the turns this request would compact,
    but they do not fit whole in the 1846 token(s) TARGET (15576) leaves
    beside the system prompt, images and recent turns (12706) and one fresh
    summary (1024); summarizing from scratch ...

WHY: compact_if_needed budgeted the in-array stand-in as
`min(SUMMARY_BLOCK_MAX_TOKENS, TARGET_TOKENS - _others - SUMMARY_MAX_TOKENS)`
and rendered it all_or_nothing=True. The injection site (chat_completions)
SKIPS its own separately-injected summary block whenever the stand-in is
used (`sum(in-array)`), freeing that block's share of the injection budget —
but the stand-in budget never counted that freed share, so it was squeezed
against TARGET as though the summary were STILL going to be injected
separately. With long recent turns it could never fit, and all-or-nothing
turned "does not fit" into "reuse nothing", forever, on every request.

THE FIX: `_standin_injected_share(inject_budget)` computes the injection
site's own cap (60% of the injection budget, capped at
SUMMARY_BLOCK_MAX_TOKENS) — the exact figure the caller uses for its
non-reusing sblock — and the stand-in budget is now
`max(TARGET-based figure, that injected share)`, but ONLY when the caller
passes `inject_budget` (chat_completions always does now; a caller that
omits it — every pre-3.1.9.1 test, an old direct call — gets exactly the
old TARGET-only figure, so nothing already pinned changes).

WHAT THIS FILE PINS, matching the production numbers from the incident
(MAX_MODEL_LEN 32768, COMPACTOR_GENERATION_RESERVE 12000, TARGET 15576,
SUMMARY_MAX_TOKENS 1024, KEEP_RECENT_TURNS 4, MAX_RETAINED_IMAGES 2,
COMPACTOR_INJECTION_BUDGET_FRACTION 0.6, COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS
6230):

  [1] reuse fires on her shape: no cap refusal, no summarize() of the
      covered span, covered turns never reach the tokenizer/guard as fresh
      input.
  [2] the reusing request is never worse than the declined path for the
      turns she keeps: run BOTH paths on the same fixture, enforce the hard
      budget on both, and compare — not arithmetic.
  [3] nothing is removed without a stand-in unless the log says so; the
      existing all-or-nothing decline still fires, with the true budget
      source named, when even the injected share cannot fit the hierarchy.
  [4] controls: a short conversation with small recent turns still reuses
      exactly as before; a hierarchy that genuinely cannot fit even the
      injected share still declines with the existing log line; the
      non-reuse injection path (`inject_budget` omitted) is unchanged.
  [5] the helper is not a name that happens to agree at both call sites —
      mutating either site's use of it independently is caught.

Only synthetic conversation content appears below (project rule: this repo
is public).

    python test_reuse_fit.py
"""

import asyncio
import contextlib
import logging
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compact-reusefit-")

# --- production numbers, from the 2026-09-16 11:06Z incident -------------
os.environ["MAX_MODEL_LEN"] = "32768"
os.environ["COMPACTOR_GENERATION_RESERVE"] = "12000"
os.environ["COMPACTOR_TARGET_TOKENS"] = "15576"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "1024"
os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "4"
os.environ["COMPACTOR_MAX_RETAINED_IMAGES"] = "2"
os.environ["COMPACTOR_INJECTION_BUDGET_FRACTION"] = "0.6"
os.environ["COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS"] = "6230"
os.environ["COMPACTOR_MAX_SUMMARY_CALLS"] = "4"

import memory  # noqa: E402

memory.ensure_storage_layout()

import facts  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name: str = "compactor"):
    lg = logging.getLogger(logger_name)
    handler = _Collector()
    prev_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)


def _find(records, needle):
    return next((r for r in records if needle in r.getMessage()), None)


def _ns(msgs):
    return [m for m in msgs if m.get("role") != "system"]


def _chars_for_tokens(n: int) -> int:
    """Inverts main.count_tokens's no-tokenizer fallback for ONE message:
    `len(text) // 4 + 4`. Exact for ASCII text with no chat-template
    available, which is what this offline test process has."""
    return max(0, (n - 4) * 4)


def history(n_exchanges: int, tag: str = "", words: int = 20) -> list[dict]:
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"{tag}question {i} " + ("word " * words)})
        out.append({"role": "assistant", "content": f"{tag}answer {i} " + ("word " * words)})
    return out


def fat_turn(role: str, tokens: int, tag: str) -> dict:
    return {"role": role, "content": f"{tag} " + ("x" * _chars_for_tokens(tokens))}


def image_turn(role: str, tag: str) -> dict:
    return {
        "role": role,
        "content": [
            {"type": "text", "text": f"{tag} here is the sketch"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ],
    }


def _seed_hierarchy(conv: str, msgs: list[dict], chunks: list[dict]):
    """A hierarchy the way maybe_rollup would have left it — same writer
    test_compaction_reuse.py uses (_record_chunk_fps), so the covered-turn
    record is built the way production actually builds it, not by a route
    production does not take."""
    st = summarizer.load_state(conv)
    st["l1"] = [c for c in chunks if c.get("tier") == "l1"]
    st["l2"] = [c for c in chunks if c.get("tier") == "l2"]
    st["last_summarized_turn"] = max(c["last_turn"] for c in chunks)
    ns = _ns(msgs)
    for c in sorted(chunks, key=lambda c: c["first_turn"]):
        summarizer._record_chunk_fps(
            st, c["first_turn"], c["last_turn"],
            ns[c["first_turn"] - 1:c["last_turn"]])
    summarizer.save_state(conv, st)
    return st


def _run(msgs, conv, **kw):
    return asyncio.run(main.compact_if_needed(list(msgs), conv, **kw))


CALLS: list[list] = []
_real_summarize = main.summarize


async def _spy_summarize(client, to_summarize):
    CALLS.append(list(to_summarize))
    return "FRESHLY-SUMMARIZED", []


# ---------------------------------------------------------------------------
# Build the production-shaped fixture.
# ---------------------------------------------------------------------------
# effective_limit = 32768 - max(12000, 0) = 20768
# inject_budget    = int(20768 * 0.6)      = 12460
# injected share   = min(6230, int(12460*0.6)=7476) = 6230
EFFECTIVE_LIMIT = min(
    main.MAX_MODEL_LEN, max(256, main.MAX_MODEL_LEN - main.GENERATION_RESERVE)
)
INJECT_BUDGET = int(EFFECTIVE_LIMIT * main.INJECTION_BUDGET_FRACTION)
INJECTED_SHARE = main._standin_injected_share(INJECT_BUDGET)
check(EFFECTIVE_LIMIT == 20768, f"fixture: effective_limit (got {EFFECTIVE_LIMIT})")
check(INJECT_BUDGET == 12460, f"fixture: inject_budget (got {INJECT_BUDGET})")
check(INJECTED_SHARE == 6230, f"fixture: injected share (got {INJECTED_SHARE})")

CONV = "reuse_fit_prod"
# 300 older exchanges (600 turns) of substantial content, so the declined
# path's batch count blows well past the 4-call cap, as it did in production
# (45 calls at 806 turns) -- and so the batching never collapses to a single
# batch, which would take the direct (HTTP) path instead of the cap check.
OLDER = history(300, words=200)
# Two image turns inside the older span — preserved verbatim, never
# summarized, never covered by the hierarchy. Placed away from the tail so
# they stay out of keep_recent.
OLDER[5] = image_turn("user", "sketch-a")
OLDER[7] = image_turn("assistant", "sketch-b")
# The hierarchy: 1 L1 (~2.5k chars) + 4 L2 (~5.0k/5.1k/3.8k/3.6k chars),
# covering every OLDER text turn (600 of them - the 2 image turns already
# excluded from text_only, so the chunk chain only needs to reach the last
# non-image turn).
_older_ns = _ns(OLDER)
_older_text_positions = [
    i + 1 for i, m in enumerate(_older_ns) if not main._message_has_image(m)
]
LAST_COVERED = _older_text_positions[-1]
# Chunk boundaries are drawn on POSITION in the full (non-system) turn
# sequence, same as production and as _seed_hierarchy's siblings elsewhere.
CHUNKS = [
    {"tier": "l1", "text": "L1 " + "a" * 2500, "first_turn": 1, "last_turn": 120},
    {"tier": "l2", "text": "L2A " + "b" * 5000, "first_turn": 121, "last_turn": 240},
    {"tier": "l2", "text": "L2B " + "c" * 5100, "first_turn": 241, "last_turn": 360},
    {"tier": "l2", "text": "L2C " + "d" * 3800, "first_turn": 361, "last_turn": 480},
    {"tier": "l2", "text": "L2D " + "e" * 3600, "first_turn": 481, "last_turn": LAST_COVERED},
]
_seed_hierarchy(CONV, OLDER, CHUNKS)
_ST = summarizer.load_state(CONV)
_HIER_TOKENS = summarizer._estimate_block_tokens(
    summarizer.format_summary_block(_ST, 10**9) or ""
)
check(4500 <= _HIER_TOKENS <= 6000,
      f"fixture: the whole hierarchy renders near the 5.1k tokens the "
      f"incident measured (got {_HIER_TOKENS})")

# Recent turns: long, as in production (user 1-6k chars, replies 8-18k).
RECENT = [
    fat_turn("user", 375, "Q1"),
    fat_turn("assistant", 2500, "A1"),
    fat_turn("user", 375, "Q2"),
    fat_turn("assistant", 1900, "A2"),
]
MSGS = OLDER + RECENT

system_msgs, to_summarize, keep_recent = main.split_messages(list(MSGS))
preserved_images = [m for m in to_summarize if main._message_has_image(m)]
OTHERS = main.count_tokens(system_msgs + preserved_images + keep_recent)
check(11500 <= OTHERS <= 14000,
      f"fixture: _others lands near the 12.7k tokens the incident measured "
      f"(got {OTHERS})")
TARGET_BASED = min(
    summarizer.SUMMARY_BLOCK_MAX_TOKENS,
    main.TARGET_TOKENS - OTHERS - main.SUMMARY_MAX_TOKENS,
)
check(TARGET_BASED < INJECTED_SHARE,
      f"fixture: the TARGET-only figure ({TARGET_BASED}) is smaller than the "
      f"injected share ({INJECTED_SHARE}) — this is the shape that broke "
      f"production; if this fails, the fixture no longer reproduces it")
check(TARGET_BASED < _HIER_TOKENS <= INJECTED_SHARE,
      f"fixture: the hierarchy ({_HIER_TOKENS}) fits the injected share but "
      f"not the TARGET-only figure ({TARGET_BASED}) — the exact squeeze")

CURRENT = main.count_tokens(MSGS)
check(CURRENT > main.TARGET_TOKENS,
      f"fixture: the whole array is over TARGET, so compaction triggers "
      f"(got {CURRENT} > {main.TARGET_TOKENS})")


# ---------------------------------------------------------------------------
# [1] The declined path (inject_budget omitted) reproduces the incident.
# ---------------------------------------------------------------------------
print("[1] DECLINED PATH (no inject_budget) reproduces the 2026-09-16 outage")
main.summarize = _real_summarize  # real cap-check path, no network needed —
# the cap fires before any HTTP call.
with capture() as records:
    stored_out: list = []
    out_old = _run(MSGS, CONV, stored_turns_out=stored_out)
check(stored_out == [0],
      f"the stand-in was declined (stored_turns_out={stored_out})")
cap_line = _find(records.records, "per-request cap")
check(cap_line is not None,
      "*** the exact production failure reproduces: 'over the N-call "
      "per-request cap' fires when inject_budget is not supplied")
decline_line = _find(records.records, "do not fit whole")
check(decline_line is not None and f"{TARGET_BASED}" not in "" ,
      "and the all-or-nothing decline logged why")
check(not any("L1 " in str(m.get("content", "")) for m in out_old),
      "and the hierarchy did NOT travel into the declined-path array")

# From here on, spy on summarize() rather than hit the real (network) path —
# every remaining section either expects summarize() not to be called at all
# (reuse covers everything) or isn't testing the cap-refusal wording, which
# [1] already pinned against the real function.
main.summarize = _spy_summarize


# ---------------------------------------------------------------------------
# [2] THE FIX: reuse fires when inject_budget is supplied.
# ---------------------------------------------------------------------------
print("[2] reuse fires on her shape once inject_budget is supplied")
with capture() as records:
    stored_out2: list = []
    out_new = _run(MSGS, CONV, stored_turns_out=stored_out2, inject_budget=INJECT_BUDGET)
check(stored_out2 and stored_out2[0] > 0,
      f"*** the stand-in was ACCEPTED (stored_turns_out={stored_out2})")
check(_find(records.records, "per-request cap") is None,
      "*** no cap refusal — the covered span never reached summarize()")
check(_find(records.records, "do not fit whole") is None,
      "and no all-or-nothing decline was logged")
check(any("L1 " in str(m.get("content", "")) for m in out_new)
      and any("L2D " in str(m.get("content", "")) for m in out_new),
      "and the hierarchy DID travel into the array — first and last chunk "
      "both present, so nothing was squeezed out either")
check(stored_out2[0] == len(_older_text_positions),
      f"every older text turn was covered (stand-in replaced "
      f"{stored_out2[0]} of {len(_older_text_positions)})")


# ---------------------------------------------------------------------------
# [3] Covered turns never reach the tokenizer/guard as fresh input.
# ---------------------------------------------------------------------------
print("[3] covered turns are not forwarded to summarize() or the array")
with capture():
    CALLS.clear()
    stored_out3: list = []
    out_new2 = _run(MSGS, CONV, stored_turns_out=stored_out3, inject_budget=INJECT_BUDGET)
check(CALLS == [] or all(len(c) == 0 for c in CALLS),
      f"*** summarize() was not handed the covered span (calls: "
      f"{[len(c) for c in CALLS]})")
check(not any("question 0 " in str(m.get("content", "")) for m in out_new2),
      "the oldest covered turn is gone from the array — replaced, not kept "
      "verbatim, so the tokenizer never re-counts it")


# ---------------------------------------------------------------------------
# [4] Weak precursor only — the guard on the BARE compacted arrays, with NO
# injected memory. This is NOT invariant 2 as the brief states it ("after
# injection and _enforce_hard_budget") — it is the one case where the two
# paths cannot differ (neither has anything injected yet to compete with the
# recent turns for guard budget). Kept as a cheap sanity check; [4b] below is
# the real test.
# ---------------------------------------------------------------------------
print("[4] precursor (no injected memory) — both still fit the guard limit")
guard_report_old: dict = {}
guarded_old = main._enforce_hard_budget(
    out_old, EFFECTIVE_LIMIT, report=guard_report_old
)
guard_report_new: dict = {}
guarded_new = main._enforce_hard_budget(
    out_new, EFFECTIVE_LIMIT, report=guard_report_new
)


def _recent_verbatim_kept(guarded):
    text = " ".join(str(m.get("content", "")) for m in guarded)
    return sum(1 for tag in ("Q1", "A1", "Q2", "A2") if tag in text)


kept_old = _recent_verbatim_kept(guarded_old)
kept_new = _recent_verbatim_kept(guarded_new)
check(kept_new >= kept_old,
      f"reuse keeps at least as many recent turns as the decline with no "
      f"injection in play (reuse={kept_new}, decline={kept_old})")
check(main.count_tokens(guarded_old) <= EFFECTIVE_LIMIT,
      f"CONTROL: the declined path still fits the guard limit after "
      f"shedding (got {main.count_tokens(guarded_old)})")
check(main.count_tokens(guarded_new) <= EFFECTIVE_LIMIT,
      f"the reusing path fits the guard limit too (got "
      f"{main.count_tokens(guarded_new)})")


# ---------------------------------------------------------------------------
# [4b] Invariant 2, FOR REAL: the forwarded PAYLOAD, built the way
# chat_completions builds it — injected blocks bounded by
# `_bound_injected_blocks` (the real helper), merged in with the real
# `inject_system_block` — then `_enforce_hard_budget` on that payload, for
# BOTH paths on the same fixture.
#
# `_bound_injected_blocks` and `inject_system_block` are synchronous, pure
# functions of (blocks, budget) / (messages, text) with no I/O of their own
# (`_bound_injected_blocks` only reaches for an HTTP-backed exact count when
# the cheap local estimate is not enough to prove it fits — see its
# docstring's "Measurement discipline" paragraph; that path degrades to the
# local estimate on the offline network this test runs on, exactly like
# production degrades when /tokenize is unreachable). So both are called
# DIRECTLY here rather than driving chat_completions end-to-end through a
# stubbed vLLM transport — chat_completions' own combination of these two
# calls (see main.py, "Single inject point") is reproduced verbatim below,
# not reinvented.
# ---------------------------------------------------------------------------
print("[4b] invariant 2 on the REAL forwarded payload (injected memory + "
      "guard), both paths, same fixture")

FACTS_TOKENS = 600
RETRIEVAL_TOKENS = 3300
FACTS_BLOCK = "FACTS-BEGIN " + ("f" * _chars_for_tokens(FACTS_TOKENS)) + " FACTS-END"
RETRIEVAL_BLOCK = (
    "RETRIEVAL-BEGIN " + ("r" * _chars_for_tokens(RETRIEVAL_TOKENS)) + " RETRIEVAL-END"
)
# Real render of the actual hierarchy, at the real cap — the same call the
# injection site makes when NOT reusing (sum(in-array) not in play).
SUMMARY_BLOCK = summarizer.format_summary_block(_ST, _standin_share := main._standin_injected_share(INJECT_BUDGET))
check(SUMMARY_BLOCK is not None,
      "fixture: the declined path's separately-injected summary block "
      "renders at the injected share (it is not all_or_nothing here — "
      "only compact_if_needed's in-array stand-in is)")


def _forwarded_payload(compacted: list[dict], *, include_summary: bool) -> list[dict]:
    """Reproduces chat_completions' "Single inject point" exactly:
    _bound_injected_blocks -> join survivors -> inject_system_block."""
    blocks = [
        (main._INJECT_PRIORITY_FACTS, "facts", FACTS_BLOCK),
        (main._INJECT_PRIORITY_RETRIEVAL, "retrieval", RETRIEVAL_BLOCK),
    ]
    if include_summary:
        blocks.append((main._INJECT_PRIORITY_SUMMARY, "summary", SUMMARY_BLOCK))
    kept, dropped, cost = main._bound_injected_blocks(blocks, INJECT_BUDGET)
    payload = list(compacted)
    if kept:
        combined = "\n\n".join(kept)
        payload = main.inject_system_block(payload, combined)
    return payload


# DECLINED path: the separate summary injection is NOT skipped (no
# `_compaction_stored_turns[0] > 0`, so `sum(in-array)` never fires) — it
# renders its own copy, exactly as the injection site does when compaction
# did not reuse.
payload_old = _forwarded_payload(out_old, include_summary=True)
# REUSE path: the injection site skips its own copy because the array
# already carries the stand-in (hostile2-reuse M1 — see compact_if_needed's
# docstring). This is the behaviour `_compaction_stored_turns` exists to
# drive; reproduced directly here since this file does not run the full
# endpoint.
payload_new = _forwarded_payload(out_new, include_summary=False)

PRE_GUARD_OLD = main.count_tokens(payload_old)
PRE_GUARD_NEW = main.count_tokens(payload_new)
check(PRE_GUARD_OLD > EFFECTIVE_LIMIT and PRE_GUARD_NEW > EFFECTIVE_LIMIT,
      f"fixture: injected, BOTH payloads land above the guard limit before "
      f"the guard runs — her real shape (~22k before the guard vs "
      f"{EFFECTIVE_LIMIT}) — so the guard actually has to choose (got "
      f"declined={PRE_GUARD_OLD}, reuse={PRE_GUARD_NEW})")

guard_report_old2: dict = {}
guarded_old2 = main._enforce_hard_budget(
    payload_old, EFFECTIVE_LIMIT, report=guard_report_old2
)
guard_report_new2: dict = {}
guarded_new2 = main._enforce_hard_budget(
    payload_new, EFFECTIVE_LIMIT, report=guard_report_new2
)

# (a) both fit the limit.
check(main.count_tokens(guarded_old2) <= EFFECTIVE_LIMIT,
      f"(a) declined payload fits after the guard (got "
      f"{main.count_tokens(guarded_old2)} <= {EFFECTIVE_LIMIT})")
check(main.count_tokens(guarded_new2) <= EFFECTIVE_LIMIT,
      f"(a) reuse payload fits after the guard (got "
      f"{main.count_tokens(guarded_new2)} <= {EFFECTIVE_LIMIT})")

# (b) reuse keeps >= as many of Q1/A1/Q2/A2 verbatim, and specifically the
# newest user turn and the reply before it (Q2/A2).
kept_old2 = _recent_verbatim_kept(guarded_old2)
kept_new2 = _recent_verbatim_kept(guarded_new2)
check(kept_new2 >= kept_old2,
      f"*** (b) with real injected memory in play, reuse still keeps at "
      f"least as many recent turns as the decline (reuse={kept_new2}, "
      f"decline={kept_old2})")


def _text_of(guarded):
    return " ".join(str(m.get("content", "")) for m in guarded)


check("Q2" in _text_of(guarded_new2) and "A2" in _text_of(guarded_new2),
      "*** (b) the reuse payload specifically keeps the newest user turn "
      "(Q2) and the reply before it (A2)")

# (c) on the reuse path, the stand-in survives WHOLE — first and last chunk
# markers both present (nothing squeezed out of it by the guard).
_new2_text = _text_of(guarded_new2)
check("L1 " in _new2_text and "L2D " in _new2_text,
      "*** (c) the stand-in survives the guard whole — first (L1) and "
      "last (L2D) chunk markers both present")

# (d) token counts of facts / retrieval / summary that SURVIVE on each path.
# Each block carries a BEGIN/END marker pair; END present means the guard's
# trimming (which cuts from the end) never reached it, so it survived
# whole. END absent but BEGIN present means it was trimmed to whatever
# follows BEGIN; BEGIN absent means the layer was dropped entirely (by
# `_bound_injected_blocks` or by the guard) — 0 tokens survived.
def _survived_tokens(text: str, label: str, begin: str, end: str, original_tokens: int) -> int:
    if end in text:
        return original_tokens
    bi = text.find(begin)
    if bi == -1:
        return 0
    return main.count_tokens([{"role": "system", "content": text[bi:]}])


def _report_survival(name: str, guarded: list[dict], has_summary: bool):
    text = _text_of(guarded)
    f_tok = _survived_tokens(text, "facts", "FACTS-BEGIN", "FACTS-END", FACTS_TOKENS)
    r_tok = _survived_tokens(
        text, "retrieval", "RETRIEVAL-BEGIN", "RETRIEVAL-END", RETRIEVAL_TOKENS
    )
    if has_summary:
        # format_summary_block's render order is L3, then L2 chronological,
        # then L1 last ("most-specific last" — see its docstring), so
        # "L2A " (the earliest chunk) opens the block and "L1 " (the
        # chunk covering the MOST RECENT older turns) closes it — L1 is
        # the true end marker here, not L2D.
        _summary_tok_full = summarizer._estimate_block_tokens(SUMMARY_BLOCK)
        s_tok = _survived_tokens(text, "summary", "L2A ", "L1 ", _summary_tok_full)
    else:
        s_tok = 0
    print(f"    [4b](d) {name}: facts={f_tok} retrieval={r_tok} summary={s_tok} "
          f"(of {FACTS_TOKENS}/{RETRIEVAL_TOKENS}/"
          f"{summarizer._estimate_block_tokens(SUMMARY_BLOCK) if has_summary else 0})")
    return f_tok, r_tok, s_tok


SURVIVAL_OLD = _report_survival("declined", guarded_old2, has_summary=True)
SURVIVAL_NEW = _report_survival("reuse", guarded_new2, has_summary=False)
check(True, "(d) survival token counts printed and recorded above — no "
            "claim that reuse keeps MORE memory, only the numbers")


# ---------------------------------------------------------------------------
# [4c] The mutation target for "the guard forgets the stand-in is special":
# a minimal fixture engineered so protection ONLY matters when the excess
# cannot be covered by turns alone within the KEEP_RECENT_TURNS floor. In
# [4b] above, the excess happened to be coverable by shedding the two
# preserved image turns either way (protected or not) — that fixture cannot
# tell the two codepaths apart, which is exactly the gap the coordinator
# flagged. This one can: the array holds ONLY the standin, a small merged
# memory block, and exactly KEEP_RECENT_TURNS recent turns (so the
# floor-respecting branch cuts ZERO turns, `_n_turns - 0 > _floor` is
# false from the very first check) — the only way to free enough room is
# either (protected) trim/drop the small memory block, which is what the
# real code does, or (if the guard no longer recognises the standin) shed
# recent turns past the floor instead, which is what it must NOT do.
# ---------------------------------------------------------------------------
print("[4c] mutation target: the guard must not treat the stand-in as "
      "ordinary droppable memory")
_MM_LIMIT = 500
_mm_caller_sys = {"role": "system", "content": "you are a companion"}
_mm_standin = {
    "role": "system",
    "content": main.COMPACTION_SUMMARY_HEADER + "\n"
               + "L1 " + ("s" * _chars_for_tokens(400)),
}
_mm_recent = [
    fat_turn("user", 10, "MQ1"),
    fat_turn("assistant", 10, "MA1"),
    fat_turn("user", 10, "MQ2"),
    fat_turn("assistant", 10, "MA2"),
]
check(len(_mm_recent) == main.KEEP_RECENT_TURNS,
      f"fixture: exactly KEEP_RECENT_TURNS recent turns, so the "
      f"floor-respecting branch has nothing it may cut (got "
      f"{len(_mm_recent)} against KEEP_RECENT_TURNS="
      f"{main.KEEP_RECENT_TURNS})")
_mm_base = [_mm_caller_sys, _mm_standin] + _mm_recent
_mm_memory_text = "m" * _chars_for_tokens(300)
# Same real merge helper as [4b] and as chat_completions' own inject point.
_mm_payload = main.inject_system_block(_mm_base, _mm_memory_text)
_MM_PRE = main.count_tokens(_mm_payload)
check(_MM_PRE > _MM_LIMIT,
      f"fixture: the micro-payload is over its (deliberately small) limit "
      f"before the guard runs (got {_MM_PRE} > {_MM_LIMIT})")

_mm_report: dict = {}
_mm_guarded = main._enforce_hard_budget(
    _mm_payload, _MM_LIMIT, report=_mm_report
)
_mm_text = _text_of(_mm_guarded)
_mm_recent_kept = sum(1 for tag in ("MQ1", "MA1", "MQ2", "MA2") if tag in _mm_text)
check(_mm_recent_kept == main.KEEP_RECENT_TURNS,
      f"*** (b)-equivalent: all {main.KEEP_RECENT_TURNS} recent turns "
      f"survive — the guard freed room from memory, not from the "
      f"protected recent window (kept {_mm_recent_kept})")
check("MQ2" in _mm_text and "MA2" in _mm_text,
      "*** (b)-equivalent: the newest user turn and its paired reply "
      "specifically survive")
check(main.COMPACTION_SUMMARY_HEADER in _mm_text and "L1 " in _mm_text,
      "*** (c)-equivalent: the stand-in survives whole — this is the "
      "exact check that goes red when _is_compaction_standin's match is "
      "neutralized")
check(main.count_tokens(_mm_guarded) <= _MM_LIMIT,
      f"(a)-equivalent: the micro-payload still fits after the guard "
      f"(got {main.count_tokens(_mm_guarded)} <= {_MM_LIMIT})")


# ---------------------------------------------------------------------------
# [5] Control: a hierarchy that STILL cannot fit even the injected share
#     still declines, and the log names the injected share, not TARGET.
# ---------------------------------------------------------------------------
print("[5] CONTROL: a hierarchy too big for even the injected share still "
      "declines, honestly")
# Reuses the production-shaped OLDER/RECENT fixture (same _others ~13.4k as
# section [1]/[2]) so the TARGET-based figure is small (~1180, as measured
# above) and the injected share (6230) is the one actually doing the work —
# a single chunk bigger than 6230 tokens must still decline, even though
# it's the larger of the two budgets that binds here.
CONV_BIG = "reuse_fit_too_big"
BIG_CHUNKS = [
    {"tier": "l1", "text": "HUGE " + "z" * int(INJECTED_SHARE * 4 * 1.3),
     "first_turn": 1, "last_turn": LAST_COVERED},
]
_seed_hierarchy(CONV_BIG, OLDER, BIG_CHUNKS)
with capture() as records:
    stored_out5: list = []
    out_big = _run(MSGS, CONV_BIG, stored_turns_out=stored_out5,
                    inject_budget=INJECT_BUDGET)
check(stored_out5 == [0],
      f"*** too big for even the injected share: still declines "
      f"(stored_turns_out={stored_out5})")
decline5 = _find(records.records, "do not fit whole")
check(decline5 is not None
      and "injection budget's summary share" in decline5.getMessage(),
      "and the decline names the injected share as the real budget source, "
      "not TARGET")


# ---------------------------------------------------------------------------
# [6] Control: short conversation with small recent turns reuses exactly as
#     before (existing shape, independent of inject_budget).
# ---------------------------------------------------------------------------
print("[6] CONTROL: a short/small-recent conversation reuses whether or not "
      "inject_budget is supplied — same result either way")
CONV_SMALL = "reuse_fit_small"
# Enough turns, at production TARGET (15576), to actually trigger
# compaction — small individual turns, so unlike [1]/[2] the TARGET-based
# stand-in budget alone is already generous and the tiny seeded chunk fits
# it either way; this isolates "does supplying inject_budget change an
# ALREADY-fitting decision", which it must not.
SMALL = history(150, words=50)
_seed_hierarchy(CONV_SMALL, SMALL, [
    {"tier": "l1", "text": "SMALL-CHUNK", "first_turn": 1, "last_turn": 40},
])
out_small_a = _run(SMALL, CONV_SMALL, stored_turns_out=(sa := []))
out_small_b = _run(SMALL, CONV_SMALL, stored_turns_out=(sb := []),
                    inject_budget=INJECT_BUDGET)
check(sa == sb and sa[0] > 0,
      f"*** small-recent-turn reuse is unaffected by inject_budget "
      f"(without={sa}, with={sb})")
check(
    any("SMALL-CHUNK" in str(m.get("content", "")) for m in out_small_a)
    == any("SMALL-CHUNK" in str(m.get("content", "")) for m in out_small_b),
    "and both arrays carry (or both omit) the stand-in identically",
)


# ---------------------------------------------------------------------------
# [7] Control: the non-reuse injection path (format_summary_block at the
#     injection site) is unchanged — same budget formula, same result,
#     whether reached via the old inline expression or the new helper.
# ---------------------------------------------------------------------------
print("[7] CONTROL: the injection-site formula is unchanged by the refactor")
_old_formula = min(summarizer.SUMMARY_BLOCK_MAX_TOKENS, int(INJECT_BUDGET * 0.6))
check(main._standin_injected_share(INJECT_BUDGET) == _old_formula,
      f"the helper reproduces the injection site's original inline formula "
      f"exactly (helper={main._standin_injected_share(INJECT_BUDGET)}, "
      f"inline={_old_formula})")
for _b in (0, 1, 100, 6230 * 10 // 6, 10**9):
    check(
        main._standin_injected_share(_b)
        == min(summarizer.SUMMARY_BLOCK_MAX_TOKENS, int(_b * 0.6)),
        f"agrees with the inline formula at inject_budget={_b}",
    )


main.summarize = _real_summarize


# ---------------------------------------------------------------------------
# [8] THE WIRING, for real: drives chat_completions itself (POST
# /v1/chat/completions), not compact_if_needed directly. Every section above
# calls compact_if_needed with `inject_budget` supplied by hand — which
# proves the function is correct but NOT that chat_completions actually
# wires it through. A mutant that deletes `inject_budget=inject_budget,`
# from chat_completions' own call to compact_if_needed is invisible to every
# section above (coordinator review, 2026-09-16): production would then run
# exactly the shipped bug while this file stayed green.
#
# Pattern followed: test_budget_guard.py's endpoint-level sections
# (`_StubVLLM`, `_post_chat`, `_CaptureLogs`, `TestClient(main.app, ...)`,
# `patch.object(main.httpx, "AsyncClient", _StubVLLM)`, `X-Conversation-Id`
# header) — grepped for `TestClient`/`chat_completions(` across
# `compactor/test_*.py` and found it already driving the real endpoint with
# vLLM stubbed exactly this way, with a comment explaining why: "nothing
# that calls a function can see whether its call site is right." Reused
# rather than reinvented; the two classes below are trimmed copies (no
# streaming path, no refusal path — this file only needs the plain
# non-streaming 200).
# ---------------------------------------------------------------------------
print("[8] driving chat_completions itself: the caller must actually pass "
      "inject_budget through")

import logging as _logging_mod  # noqa: E402  (only used by [8]/[9])
from unittest.mock import patch as _patch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class _CaptureLogs(_logging_mod.Handler):
    """Trimmed copy of test_budget_guard.py's handler of the same name."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _EndpointStubResponse:
    """A minimally well-formed non-streaming completion — trimmed copy of
    test_budget_guard.py's _StubResponse — so the handler takes its normal
    200 path instead of a degraded one."""

    status_code = 200
    text = ""

    def json(self):
        return {
            "id": "stub",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        }


class _EndpointStubVLLM:
    """Stands in for httpx.AsyncClient on the request path and records every
    JSON body POSTed upstream — trimmed copy of test_budget_guard.py's
    _StubVLLM. `sent` is the assertion surface: nothing the handler returns
    reveals the forwarded payload."""

    sent: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        _EndpointStubVLLM.sent.append(json)
        return _EndpointStubResponse()

    async def aclose(self):
        pass


def _swallow_tail(coro, label=None):
    """Stand-in for main._fire_and_forget — trimmed copy of
    test_budget_guard.py's helper of the same name. The post-response
    memory tail (facts, embeddings, rollups) is not what this section is
    testing and would otherwise reach for vLLM and the scratch volume a
    second time."""
    try:
        coro.close()
    except Exception:
        pass


_client = TestClient(main.app, client=("127.0.0.1", 12399), raise_server_exceptions=False)

CONV_EP = "reuse_fit_endpoint"
_seed_hierarchy(CONV_EP, OLDER, CHUNKS)
# One fact, so memory injection actually fires and the "injected memory
# [...]" log line — the one that carries "sum(in-array)" — gets printed at
# all (that line is gated on `if kept:`; an empty injected_blocks list
# would skip it even on a reusing turn).
facts.save_facts(CONV_EP, [
    {"text": "ENDPOINT-FACT-SENTINEL", "added_turn": 1, "last_used": 0, "pin": False},
])

_ep_handler = _CaptureLogs()
_ep_logger = _logging_mod.getLogger("compactor")
_ep_logger.addHandler(_ep_handler)
try:
    _EndpointStubVLLM.sent.clear()
    # /tokenize stubbed to fail FAST (a local ConnectError raised
    # synchronously) rather than relying on this offline test network's real
    # ~2-4s dead-connect timeout, which the rest of this file pays because
    # its fixtures are small enough not to care — this fixture drives the
    # real endpoint over ~600 turns and calls count_tokens_exact/
    # count_text_tokens_exact repeatedly, so the real timeout would make
    # this section slow without testing anything different. Same fast-fail
    # shape as test_budget_guard.py's R22 comment describes and uses.
    with _patch.object(main.httpx, "AsyncClient", _EndpointStubVLLM), \
         _patch.object(main, "_fire_and_forget", _swallow_tail), \
         _patch.object(
             main.httpx, "post",
             lambda url, *a, **kw: (_ for _ in ()).throw(
                 main.httpx.ConnectError("stubbed for test speed")
             ) if "/tokenize" in url else main.httpx.post(url, *a, **kw)
         ):
        _ep_resp = _client.post(
            "/v1/chat/completions",
            json={"model": "stub-model", "messages": MSGS, "stream": False},
            headers={"X-Conversation-Id": CONV_EP},
        )
finally:
    _ep_logger.removeHandler(_ep_handler)

check(_ep_resp.status_code == 200,
      f"fixture: the endpoint request succeeded (got {_ep_resp.status_code}: "
      f"{_ep_resp.text[:200]})")
check(len(_EndpointStubVLLM.sent) == 1,
      f"fixture: exactly one upstream call — the covered span never went "
      f"through summarize() either (got {len(_EndpointStubVLLM.sent)} call(s))")
_ep_forwarded = _EndpointStubVLLM.sent[-1] if _EndpointStubVLLM.sent else None
_ep_fwd_text = (
    " ".join(str(m.get("content", "")) for m in _ep_forwarded.get("messages", []))
    if _ep_forwarded else ""
)
check(main.COMPACTION_SUMMARY_HEADER in _ep_fwd_text,
      "*** the stand-in reached the wire: the forwarded payload carries "
      "compact_if_needed's summary header")
check("L1 " in _ep_fwd_text and "L2D " in _ep_fwd_text,
      "*** and both boundary chunks (L1, L2D) are in it — the stand-in "
      "the ACTUAL endpoint forwarded, not one built by calling "
      "compact_if_needed directly")
_ep_all_text = "\n".join(r.getMessage() for r in _ep_handler.records)
check("per-request cap" not in _ep_all_text,
      "*** no cap-refusal line was logged by the real endpoint")
check("sum(in-array)" in _ep_all_text,
      "*** the endpoint's own injected-memory log line says sum(in-array) "
      "— chat_completions actually skipped its separate summary copy, "
      "which only happens when compact_if_needed reported a stand-in")


# ---------------------------------------------------------------------------
# [9] The injection-site formula, reached for REAL: on the NON-reusing path
# through chat_completions (a short conversation, under TARGET, so
# compact_if_needed returns before it ever runs — `_compaction_stored_
# turns` stays empty and the separate summary injection is NOT skipped),
# spy on summarizer.format_summary_block and assert the budget it was
# actually called with, at the real endpoint, is
# _standin_injected_share(inject_budget) — not a formula that happens to
# agree with it in a unit test that never reaches the call site (coordinator
# review: [7] pins the helper against itself, never against the site).
# ---------------------------------------------------------------------------
print("[9] driving chat_completions itself: the injection site must call "
      "the SHARED helper, not its own copy of the formula")

CONV_SITE = "reuse_fit_site"
# A hierarchy exists (so format_summary_block has something real to render)
# but is UNRELATED to this request's own (short) content — irrelevant here
# regardless, since the request below is short enough that compact_if_needed
# returns before it even reads stored state.
_seed_hierarchy(CONV_SITE, OLDER, CHUNKS)
SITE_MSGS = history(3)
check(main.count_tokens(SITE_MSGS) <= main.TARGET_TOKENS,
      f"fixture: this conversation is under TARGET, so compact_if_needed "
      f"takes its early return and never reports a stand-in (got "
      f"{main.count_tokens(SITE_MSGS)} <= {main.TARGET_TOKENS})")
# A request-supplied max_tokens, so this request's effective_limit (and
# therefore its inject_budget) is NOT the same 20,768/12,460 every other
# section in this file uses. That matters here specifically: at the
# module-level INJECT_BUDGET (12,460), 60% and 50% of it BOTH land on the
# SUMMARY_BLOCK_MAX_TOKENS cap (6,230 = 12,460 * 0.5 exactly, as well as
# min(6230, 12460*0.6)) — a numeric coincidence of the production constants
# this file mirrors, not a general truth, and it made a drifted-coefficient
# mutant (0.6 -> 0.5) INVISIBLE here on the first attempt (caught in mutation
# testing below). A smaller effective_limit moves inject_budget below the
# point where the cap dominates, so 0.6 and 0.5 diverge for real.
SITE_MAX_TOKENS = 16000
_site_effective_limit = min(
    main.MAX_MODEL_LEN,
    max(256, main.MAX_MODEL_LEN - max(main.GENERATION_RESERVE, SITE_MAX_TOKENS)),
)
_site_inject_budget = int(_site_effective_limit * main.INJECTION_BUDGET_FRACTION)
check(_site_inject_budget != INJECT_BUDGET,
      f"fixture: this request's inject_budget ({_site_inject_budget}) "
      f"differs from the module-level one ({INJECT_BUDGET}) other sections "
      f"use, so a coefficient drift at the site cannot hide behind the "
      f"SUMMARY_BLOCK_MAX_TOKENS cap the way it did at INJECT_BUDGET")
check(
    main._standin_injected_share(_site_inject_budget)
    != min(summarizer.SUMMARY_BLOCK_MAX_TOKENS, int(_site_inject_budget * 0.5)),
    f"fixture: at THIS inject_budget, 0.6 and 0.5 of it give different "
    f"results (0.6-based: {main._standin_injected_share(_site_inject_budget)}, "
    f"0.5-based: {min(summarizer.SUMMARY_BLOCK_MAX_TOKENS, int(_site_inject_budget * 0.5))}) "
    f"— this fixture can actually tell the two formulas apart"
)

_site_calls: list[tuple] = []
_real_format_summary_block = summarizer.format_summary_block


def _spy_format_summary_block(state, max_tokens=None, **kwargs):
    _site_calls.append((max_tokens, kwargs))
    return _real_format_summary_block(state, max_tokens, **kwargs)


_site_handler = _CaptureLogs()
_ep_logger.addHandler(_site_handler)
try:
    _EndpointStubVLLM.sent.clear()
    with _patch.object(main.httpx, "AsyncClient", _EndpointStubVLLM), \
         _patch.object(main, "_fire_and_forget", _swallow_tail), \
         _patch.object(summarizer, "format_summary_block", _spy_format_summary_block), \
         _patch.object(
             main.httpx, "post",
             lambda url, *a, **kw: (_ for _ in ()).throw(
                 main.httpx.ConnectError("stubbed for test speed")
             ) if "/tokenize" in url else main.httpx.post(url, *a, **kw)
         ):
        _site_resp = _client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model", "messages": SITE_MSGS, "stream": False,
                "max_tokens": SITE_MAX_TOKENS,
            },
            headers={"X-Conversation-Id": CONV_SITE},
        )
finally:
    _ep_logger.removeHandler(_site_handler)

check(_site_resp.status_code == 200,
      f"fixture: the endpoint request succeeded (got {_site_resp.status_code}: "
      f"{_site_resp.text[:200]})")
check(len(_site_calls) >= 1,
      f"fixture: format_summary_block was called at least once at the real "
      f"injection site (got {len(_site_calls)} call(s))")
_EXPECTED_SITE_BUDGET = main._standin_injected_share(_site_inject_budget)
_site_budgets = [c[0] for c in _site_calls]
check(_EXPECTED_SITE_BUDGET in _site_budgets,
      f"*** the injection site rendered the summary block with EXACTLY "
      f"_standin_injected_share(inject_budget) ({_EXPECTED_SITE_BUDGET}), "
      f"not a locally-drifted copy of the formula (got budgets "
      f"{_site_budgets})")


# ---------------------------------------------------------------------------
# [F1] p7 hostile pass #7: the compacted-branch floor (main.py:_enforce_hard_
# budget, `_floor`) used to be the raw KEEP_RECENT_TURNS MESSAGE count (4).
# split_messages ALIGNS its own kept-recent window to start on a USER turn
# (leading non-user turns move into the summarized portion — a template
# requirement), so the window it actually protects can hold FEWER than
# KEEP_RECENT_TURNS messages. With an old, unpaired assistant turn (an image,
# most often — but the bug is about POSITION, not content, so a plain heavy
# assistant turn reproduces it just as well) sitting where the 4th-from-end
# slot falls, the old floor counted it as "recent" and protected it from the
# pre-shed loop — spending injected memory (halving, then dropping facts) to
# keep a turn that was never actually inside the aligned recent window. This
# tests main._enforce_hard_budget directly (like test_p5_guard.py, which
# owns this same function's other branches), with its own deterministic
# byte-counting stand-in for count_tokens so the numbers here do not depend
# on a live tokenizer.
# ---------------------------------------------------------------------------
print("\n[F1] guard floor is ALIGNED like split_messages, not a raw message count")


def _f1_tokens(msgs) -> int:
    return sum(len(main._message_text(m).encode("utf-8")) + 4 for m in msgs)


def _f1_build(old_exchanges=30):
    """A compaction stand-in, injected memory, `old_exchanges` ordinary old
    pairs, then one UNPAIRED old assistant turn (the image stand-in) right
    before the previous exchange and the newest turn — exactly the slot
    `split_messages` would push out of its aligned keep_recent window
    (KEEP_RECENT_TURNS=4 non-system messages ending in
    [image, prev-u, prev-a, newest-u] starts on ASSISTANT, so alignment
    drops the image, leaving keep_recent=[prev-u, prev-a, newest-u], 3
    messages, not 4)."""
    facts = "[Facts]\n" + "".join(
        f"- FACT{i:02d} she likes item {i} very much indeed.\n" for i in range(9)
    )
    facts = facts + "f" * (400 - len(facts))
    msgs = [
        {"role": "system", "content": "P" * 1200},
        {"role": "system", "content": main.COMPACTION_SUMMARY_HEADER + "\n" + "s" * 6000},
        {"role": "system", "content": facts},
    ]
    for i in range(old_exchanges):
        msgs.append({"role": "user", "content": f"old-u{i} " + "u" * 150})
        msgs.append({"role": "assistant", "content": f"old-a{i} " + "a" * 1650})
    # The unpaired old turn: no user turn immediately precedes it in this
    # position (the pair above it already closed), so it is exactly the
    # shape split_messages would push into the summarized portion when the
    # 4-message tail starts on it.
    msgs.append({"role": "assistant", "content": "old-image-stub " + "i" * 30000})
    msgs.append({"role": "user", "content": "prev-u " + "u" * 150})
    msgs.append({"role": "assistant", "content": "prev-a " + "a" * 1650})
    msgs.append({"role": "user", "content": "newest " + "n" * 200})
    return msgs


_F1_LIMIT = 20768 - 90  # the shipped v3.1.9 effective limit, less a time-line reserve

_f1_saved_count_tokens = main.count_tokens
_f1_saved_count_tokens_exact = main.count_tokens_exact
_f1_saved_margin = main._BUDGET_MARGIN
main.count_tokens = _f1_tokens
main.count_tokens_exact = lambda ms, *a, **k: _f1_tokens(ms)
main._BUDGET_MARGIN = 0
try:
    _f1_msgs = _f1_build()
    _f1_rep: dict = {}
    _f1_out = main._enforce_hard_budget(_f1_msgs, _F1_LIMIT, 1, _f1_rep)
finally:
    main.count_tokens = _f1_saved_count_tokens
    main.count_tokens_exact = _f1_saved_count_tokens_exact
    main._BUDGET_MARGIN = _f1_saved_margin

_f1_mem_out = [
    x for x in _f1_out
    if x.get("role") == "system" and "[Facts]" in (x.get("content") or "")
]
_f1_facts_whole = bool(_f1_mem_out) and all(
    f"FACT{i:02d}" in _f1_mem_out[0]["content"] for i in range(9)
)
_f1_image_survived = any(
    m.get("role") == "assistant" and "old-image-stub" in (m.get("content") or "")
    for m in _f1_out
)
check(_f1_rep.get("fits"), f"fixture: the guard fit the payload ({_f1_rep})")
check(
    _f1_facts_whole,
    "*** F1: facts survive whole — the unpaired old turn is shed by the "
    "pre-shed loop instead of being counted as 'recent' and protected at "
    "memory's expense",
)
check(
    not _f1_image_survived,
    "*** F1: the unpaired old turn does NOT survive alongside whole facts "
    "— if it did, this fixture is not exercising the floor bug at all",
)

# CONTROL: the actual protected window (prev-u, prev-a, newest) always
# survives regardless of the floor fix — this proves the fix does not
# over-shed into turns split_messages really would protect.
_f1_recent_survived = all(
    any(
        m.get("role") == exp_role and m.get("content", "").startswith(exp_prefix)
        for m in _f1_out
    )
    for exp_role, exp_prefix in (
        ("user", "prev-u"), ("assistant", "prev-a"), ("user", "newest"),
    )
)
check(
    _f1_recent_survived,
    "CONTROL: the truly-recent window (prev-u, prev-a, newest) still "
    "survives — the fix narrows the floor, it does not remove protection "
    "for what split_messages actually keeps",
)


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nall checks passed")
