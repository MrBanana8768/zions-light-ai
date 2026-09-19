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
import re
import sys
import tempfile
from pathlib import Path

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
# [10] P9-1/P9-2 (hostile pass #9): the round-4 numbers
# (COMPACTOR_INJECTION_BUDGET_FRACTION=0.75, COMPACTOR_SUMMARY_BLOCK_MAX_
# TOKENS=12000, COMPACTOR_STANDIN_BUDGET_FRACTION=1.0). The SHIPPED
# SUMMARY_BLOCK_MAX_TOKENS is now higher (P10-2) and [12] reads it from the
# Dockerfile and runpod.env.template; a hierarchy that reuses at 12000
# reuses at any larger value. Not the 0.6/6230 incident numbers this file pins
# everywhere else. A hierarchy built to DOCUMENTED CAPACITY (9 L1 scenes at
# L1_MAX_TOKENS, 4 L2 chapters at L2_MAX_TOKENS, 1 L3 at L3_MAX_TOKENS — the
# same "9*L1_MAX + 4*L2_MAX + L3_MAX = 11,300" arithmetic summarizer.py's
# own format_summary_block docstring names) must REUSE at the shipped
# numbers. At the OLD 0.6/6230 pair it must still DECLINE — proving the
# fixture actually sits at the boundary the finding describes, not merely
# "large enough that any fix would pass it" — and even under the interim
# 0.75/6230 pairing (P9-1's "planned" values this release almost shipped)
# it must ALSO decline, matching P9-2's own measurement that raising
# SUMMARY_BLOCK_MAX_TOKENS alone (without also fixing the 0.6-of-
# inject_budget formula) buys nothing: the old `_standin_injected_share`
# formula pins the ceiling at 9,345 regardless of SBMAX, 1,955 short of
# capacity.
# ---------------------------------------------------------------------------
print("[10] P9-1/P9-2: a hierarchy at documented CAPACITY reuses at the "
      "shipped numbers, declines at the old ones")
main.summarize = _spy_summarize


def _g_filler(n_tokens: int) -> str:
    """ASCII text whose _estimate_block_tokens (chars//4, no tokenizer
    available in this offline test) prices at very close to n_tokens —
    same construction P9's own cliff2.py/fence2.py probes used."""
    return ("word " * n_tokens).strip()[: n_tokens * 4]


CONV_CAPACITY = "reuse_fit_capacity"
# 65 exchanges (130 non-system turns) — exactly enough turn POSITIONS for
# the 9 L1 + 4 L2 chunks below to cover with no gap, same chunk-boundary
# convention _seed_hierarchy's other callers use (positions in the full
# non-system turn sequence).
G_OLDER = history(65, words=200)
_G_L1_CHUNKS = [
    {"tier": "l1", "text": "L1scene " + _g_filler(summarizer.L1_MAX_TOKENS),
     "first_turn": i * 10 + 1, "last_turn": i * 10 + 10}
    for i in range(9)
]
_G_L2_CHUNKS = [
    {"tier": "l2", "text": "L2chapter " + _g_filler(summarizer.L2_MAX_TOKENS),
     "first_turn": 90 + i * 10 + 1, "last_turn": 90 + i * 10 + 10}
    for i in range(4)
]
_G_LAST_COVERED = _G_L2_CHUNKS[-1]["last_turn"]
check(_G_LAST_COVERED == len(_ns(G_OLDER)),
      f"fixture: the 13 chunks cover every turn position in G_OLDER with no "
      f"gap (last_turn={_G_LAST_COVERED}, non-system turns="
      f"{len(_ns(G_OLDER))})")
_G_ST = _seed_hierarchy(CONV_CAPACITY, G_OLDER, _G_L1_CHUNKS + _G_L2_CHUNKS)
# _seed_hierarchy only understands "l1"/"l2" tiers (see its own docstring);
# L3 is set directly the same way test_summary_block_budget.py's fixtures
# do (a plain {"text", "first_turn", "last_turn"} dict) — there is no
# separate "real writer" for L3 to call here without a live LLM, and this
# module-level dict assignment is exactly what maybe_rollup itself does
# after an L3 refresh call returns.
_G_ST["l3"] = {
    "text": "L3theme " + _g_filler(summarizer.L3_MAX_TOKENS),
    "first_turn": 1, "last_turn": _G_LAST_COVERED,
}
summarizer.save_state(CONV_CAPACITY, _G_ST)
_G_ST = summarizer.load_state(CONV_CAPACITY)

# Recent turns sized like the production fixture above (OTHERS ~12-14k);
# doesn't matter much here since G_OLDER alone (65 * ~200-word turns) is
# already far over TARGET_TOKENS.
G_RECENT = [
    {"role": "user", "content": "Q1 " + "u" * 1500},
    {"role": "assistant", "content": "A1 " + "a" * 10000},
    {"role": "user", "content": "Q2 " + "u" * 1500},
    {"role": "assistant", "content": "A2 " + "a" * 7600},
]
G_MSGS = G_OLDER + G_RECENT
check(main.count_tokens(G_MSGS) > main.TARGET_TOKENS,
      "fixture: G_MSGS is over TARGET, so compaction triggers")

# Measure the hierarchy's own true render cost uncapped (SUMMARY_BLOCK_MAX_
# TOKENS raised far out of the way just for this measurement) — must sit at
# or above the documented 11,300-token construction capacity, otherwise
# this fixture is not actually AT capacity and proves nothing.
_g_saved_sbmax_measure = summarizer.SUMMARY_BLOCK_MAX_TOKENS
summarizer.SUMMARY_BLOCK_MAX_TOKENS = 10**9
_G_HIER_TOKENS = summarizer._estimate_block_tokens(
    summarizer.format_summary_block(_G_ST, 10**9) or ""
)
summarizer.SUMMARY_BLOCK_MAX_TOKENS = _g_saved_sbmax_measure
_G_CAPACITY = (
    9 * summarizer.L1_MAX_TOKENS + 4 * summarizer.L2_MAX_TOKENS
    + summarizer.L3_MAX_TOKENS
)
check(_G_CAPACITY == 11300, f"fixture: documented capacity (got {_G_CAPACITY})")
check(_G_HIER_TOKENS >= _G_CAPACITY,
      f"fixture: the hierarchy's true render cost ({_G_HIER_TOKENS}) is at "
      f"or above documented capacity ({_G_CAPACITY}) — this fixture is "
      f"actually AT the boundary, not merely large")


def _g_run(sbmax: int, inject_budget: int):
    saved = summarizer.SUMMARY_BLOCK_MAX_TOKENS
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = sbmax
    try:
        stored_out: list = []
        out = _run(G_MSGS, CONV_CAPACITY, stored_turns_out=stored_out,
                    inject_budget=inject_budget)
    finally:
        summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved
    return stored_out, out


_G_OLD_INJECT = int(EFFECTIVE_LIMIT * 0.6)
_G_PLANNED_INJECT = int(EFFECTIVE_LIMIT * 0.75)  # same as the shipped fraction

_g_decline_before = main.reuse_decline_state()["declined_budget"]
_g_stored_old, _g_out_old = _g_run(6230, _G_OLD_INJECT)
check(_g_stored_old == [0],
      f"*** at the OLD 0.6/6230 numbers, a hierarchy at documented capacity "
      f"DECLINES (stored_turns_out={_g_stored_old}) — P9-1's reproduction")
check(main.reuse_decline_state()["declined_budget"] == _g_decline_before + 1,
      "*** and the decline is counted in main.reuse_decline_state() — the "
      "health signal this fix adds actually increments on this exact "
      "failure")

_g_stored_interim, _g_out_interim = _g_run(6230, _G_PLANNED_INJECT)
check(_g_stored_interim == [0],
      f"*** at 0.75 injection fraction but the OLD SUMMARY_BLOCK_MAX_TOKENS "
      f"(6230), it STILL declines (stored_turns_out={_g_stored_interim}) — "
      f"P9-2: raising the fraction alone buys nothing while "
      f"_standin_injected_share's hard-coded 0.6 keeps the ceiling pinned "
      f"under 9,345")

_g_stored_new, _g_out_new = _g_run(12000, _G_PLANNED_INJECT)
check(_g_stored_new and _g_stored_new[0] == _G_LAST_COVERED,
      f"*** at the round-4 0.75/12000/1.0 numbers, the SAME full-capacity "
      f"hierarchy REUSES completely (stored_turns_out={_g_stored_new}, "
      f"expected [{_G_LAST_COVERED}]) — the fix")
check(any("L1scene " in str(m.get("content", "")) for m in _g_out_new)
      and any("L2chapter " in str(m.get("content", "")) for m in _g_out_new)
      and any("L3theme " in str(m.get("content", "")) for m in _g_out_new),
      "and every tier — L1, L2 and L3 — actually travelled into the array, "
      "not just the newest scenes a partial fit would have kept")


# ---------------------------------------------------------------------------
# [11] P10-1 (hostile pass #10): [10] above stops at compact_if_needed — it
# proves reuse FIRES at the shipped numbers, but never calls the guard that
# runs right after it on every real request. On her real branch, 24 of 472
# positions (5.1%) reused successfully and then had the guard destroy
# persona, facts and retrieval AND shed the previous exchange anyway,
# finishing 6,187-9,213 tokens under the limit — because the compacted
# branch's turn-shed floor (main.py's `_floor`) used to gate its own
# alignment on `len(_turn_idxs) >= _floor`, which is false on exactly the
# shape a reusing request produces (`[U_prev, A_prev, U_new]`, 3 turns for
# KEEP_RECENT_TURNS=4). `_floor` stayed unaligned, the pre-shed loop never
# started, every spendable block was halved and dropped for nothing, and
# the plain shed loop after this branch dropped U_prev/A_prev anyway.
#
# The fix restructures the compacted branch to decide ONCE, by arithmetic:
# shed the next old exchange (even below the aligned floor) only when
# spending every spendable injected block down to nothing still could not
# cover the gap; otherwise let memory pay and leave the recent window
# whole. [11a] is her actual shape (a single reply bigger than all injected
# memory combined — memory cannot possibly cover the gap alone, so the
# exchange must go, and it goes WHOLE with memory untouched). [11b] is the
# CONTROL this fix must not break: a smaller gap that injected memory alone
# already covers, where U_prev/A_prev and the facts block all survive and
# no turn is shed at all — proving this is still an arithmetic choice, not
# "always shed a turn when a stand-in is present" (a fifth special case,
# which LOOPS5_BRIEF explicitly rules out).
#
# Builds on the SAME at-capacity hierarchy [10] just proved reuses
# completely at 0.75/12000/1.0, so the stand-in below is REAL
# compact_if_needed output, not a hand-built stub — [10]'s own discipline.
# ---------------------------------------------------------------------------
print("\n[11] P10-1: the guard no longer spends memory it does not need to")

_G11_FACTS = "[Facts]\n" + "".join(
    f"- FACT{i:02d} she likes item {i} very much indeed.\n" for i in range(9)
)
_G11_FACTS = _G11_FACTS + "f" * (400 - len(_G11_FACTS))
_G11_RETR = "[Retrieved]\n" + "RETRIEVAL " + "r" * (1500 - 22)
_G11_MEM = _G11_FACTS + "\n\n" + _G11_RETR


def _g11_build_and_guard(aprev_chars: int, *, expect_reuse: bool = True):
    """Reuse the CONV_CAPACITY hierarchy against a fresh recent window whose
    A_prev is `aprev_chars` long: real compact_if_needed at the shipped
    numbers builds the stand-in (or declines, see `expect_reuse` — P12-1
    made the recent window's own size, A_prev included, part of the reuse
    decision itself, so a large enough `aprev_chars` no longer reaches the
    guard with a stand-in beside it at all), then persona + injected memory
    are added the way chat_completions adds them, then the real guard runs.
    """
    recent = [
        {"role": "user", "content": "prev-u " + "u" * 300},
        {"role": "assistant", "content": "prev-a " + "a" * aprev_chars},
        {"role": "user", "content": "newest " + "n" * 400},
    ]
    msgs = G_OLDER + recent
    saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = 12000
    stored_out: list = []
    try:
        out = _run(msgs, CONV_CAPACITY, stored_turns_out=stored_out,
                    inject_budget=_G_PLANNED_INJECT)
    finally:
        summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved_sbmax
    if expect_reuse:
        check(
            stored_out == [_G_LAST_COVERED],
            f"fixture (aprev={aprev_chars}): reuse fires on the at-capacity "
            f"hierarchy (stored_turns_out={stored_out})",
        )
    else:
        check(
            stored_out == [0],
            f"*** P12-1 fixture (aprev={aprev_chars}): reuse DECLINES — "
            f"A_prev alone is now part of the P11-6 structural reserve "
            f"(`system_msgs + keep_recent`), so a reply this large no "
            f"longer reaches the guard with the stand-in still standing "
            f"beside it (stored_turns_out={stored_out})",
        )
    # `out` is compact_if_needed's real return: `history()`'s own leading
    # "you are a companion" caller system message, the real stand-in (when
    # one rendered — `None` on a decline whose fresh-span summarize() also
    # produced nothing, which this stubbed `summarize()` does not), then
    # the 3 real recent turns — find the stand-in by CONTENT (as the guard
    # itself does via `_is_compaction_standin`), not by position, and drop
    # the caller system message here (redundant with the persona line
    # added below; keeping both would leave an extra unprotected system
    # message that only muddies what this section is measuring).
    standin = next((m for m in out if main._is_compaction_standin(m)), None)
    recent = [m for m in out if m.get("role") != "system"]
    full = [{"role": "system", "content": "P" * 2500}]
    if standin is not None:
        full.append(standin)
    full += [{"role": "system", "content": _G11_MEM}] + recent
    # No /tokenize in this offline process, so this runs on main's own real
    # fallback (char/4-ish local estimate, UNCORRECTED) — the exact counter
    # state p10's own reproduction used as its second confirmation
    # ("Same result with /tokenize unavailable"). Deliberately NOT the
    # deterministic byte-count stand-in test_p5_guard.py/[F1] use: those
    # size their fixtures directly in that counter's own units, but this
    # section's fixture is sized in the REAL estimator's units (chars/4,
    # matching [10]'s own hierarchy-capacity arithmetic above), so swapping
    # in a byte-exact counter here would silently re-price every message
    # ~4x and stop measuring the scenario this section is named for.
    rep: dict = {}
    result = main._enforce_hard_budget(full, EFFECTIVE_LIMIT, 1, rep)
    mem_out = [
        m for m in result
        if m.get("role") == "system" and "[Facts]" in (m.get("content") or "")
    ]
    facts_whole = bool(mem_out) and all(
        f"FACT{i:02d}" in mem_out[0]["content"] for i in range(9)
    )
    ns = [m for m in result if m.get("role") != "system"]
    prev_survived = (
        any("prev-u" in main._message_text(m) for m in ns)
        and any("prev-a" in main._message_text(m) for m in ns)
    )
    return rep, facts_whole, prev_survived, ns


# [11a] her real shape: a single 64,000-character reply — about 16,976
# tokens at her MEASURED rate (3.77 chars/token, Tekken, her current
# branch — P11-5, hostile pass #11; this section used to cite "measured
# 26,484-39,569 chars" for "a 16k-token reply", which was itself computed
# from the STALE 2.0-2.4 chars/token figure P11-5 corrected, and did not
# match the 64,000 actually used here either way — stating the fixture's
# own size directly, and what that is in tokens at the current measured
# rate, so this comment cannot drift from the code again the same way).
# 64,000 characters is above her single largest recorded reply (51,290
# characters).
#
# P12-1 (hostile pass #12) CHANGES WHAT THIS CASE EXERCISES. Before, this
# reply outweighed persona + facts + retrieval combined but reuse still
# FIRED (the old P11-6 reserve never looked at the recent window at all),
# so the guard had to notice memory could not cover the gap and shed the
# exchange instead — P10-1/P11-4's own scenario. Now `keep_recent` — this
# very reply — is part of the reuse decision itself: a reply this large no
# longer fits beside the stand-in AT ALL, so reuse correctly declines
# before the guard ever runs, and the declined path (with nothing large
# left to forward in this offline harness, where `summarize()` is stubbed
# rather than genuinely deferring a backlog — see `main.summarize =
# _spy_summarize` above) needs to shed nothing. [11a-mid] below keeps the
# ORIGINAL scenario alive at a size P11-6 still allows through.
_g11_rep_h, _g11_facts_h, _g11_prev_h, _g11_ns_h = _g11_build_and_guard(
    64000, expect_reuse=False
)
check(_g11_rep_h.get("fits") is True, f"[11a] the guard fits the payload ({_g11_rep_h})")
check(
    (_g11_rep_h.get("dropped_turns") or 0) == 0,
    f"*** P12-5 [11a]: with reuse declined by P12-1 before the guard ever "
    f"ran, the guard has nothing left it needs to shed — U_prev/A_prev are "
    f"never at risk here in the first place ({_g11_rep_h})",
)
check(_g11_prev_h, "*** P12-1/P12-5 [11a]: U_prev/A_prev survive — reuse declined rather than firing and losing them")
check(
    any("newest" in main._message_text(m) for m in _g11_ns_h),
    "[11a]: the newest turn always survives",
)

# [11a-mid] P10-1/P11-4's ORIGINAL scenario, preserved: a reply too big for
# injected memory to cover, but still small enough that P12-1's structural
# reserve (`system + keep_recent`) fits beside the stand-in, so reuse still
# fires and the GUARD — not the reuse decision — is what has to choose
# between spending memory and shedding the exchange. 46,000 characters
# (~12.2k tokens local) clears the P11-6/P12-1 ceiling at this fixture's
# numbers (the stand-in renders ~11.5k, `effective_limit_est` here is
# ~25,960 — see the fraction note in [14] for why this file's own
# `INJECTION_BUDGET_FRACTION` differs from `_G_PLANNED_INJECT`'s) while
# still overflowing the guard's real `EFFECTIVE_LIMIT` (20,768) by more
# than persona+facts+retrieval combined (~1.1k tokens) can free — the same
# "memory cannot possibly cover the gap" shape [11a] used to test.
_g11_rep_m, _g11_facts_m, _g11_prev_m, _g11_ns_m = _g11_build_and_guard(46000)
check(_g11_rep_m.get("fits") is True, f"[11a-mid] the guard fits the payload ({_g11_rep_m})")
check(
    _g11_rep_m.get("trimmed_blocks") == 0 and _g11_rep_m.get("dropped_blocks") == 0,
    f"*** P10-1 [11a-mid]: injected memory is NOT touched — spending it "
    f"here could never have covered this A_prev's gap, so the fix does not "
    f"waste it before shedding the exchange that actually pays for the "
    f"request ({_g11_rep_m})",
)
check(_g11_facts_m, "*** P10-1 [11a-mid]: facts survive WHOLE (not halved, not dropped)")
check(
    (_g11_rep_m.get("dropped_turns") or 0) == 2,
    f"[11a-mid]: the previous exchange (U_prev+A_prev, one whole pair) is "
    f"what pays for the request ({_g11_rep_m.get('dropped_turns')} dropped)",
)
check(not _g11_prev_m, "[11a-mid]: U_prev/A_prev do NOT survive — they are what the fix sheds")
check(
    any("newest" in main._message_text(m) for m in _g11_ns_m),
    "[11a-mid]: the newest turn always survives",
)

# [11b] CONTROL, the other side of the same arithmetic: a smaller gap that
# injected memory alone already covers. The fix must NOT reach for the
# previous exchange here — U_prev/A_prev AND the facts block all survive.
# Proves the fix is an arithmetic choice, not "always shed a turn when a
# stand-in is present" (a fifth special case of the same defect).
_g11_rep_l, _g11_facts_l, _g11_prev_l, _g11_ns_l = _g11_build_and_guard(32000)
check(_g11_rep_l.get("fits") is True, f"[11b] the guard fits the payload ({_g11_rep_l})")
check(
    (_g11_rep_l.get("dropped_turns") or 0) == 0,
    f"*** P10-1 [11b] CONTROL: no turn is dropped — memory alone covers "
    f"this smaller gap ({_g11_rep_l})",
)
check(_g11_prev_l, "*** P10-1 [11b] CONTROL: U_prev AND A_prev both survive")
check(
    _g11_facts_l,
    "*** P10-1 [11b] CONTROL: the facts block survives WHOLE too (retrieval, "
    "listed after facts in the same injected block, is what a partial trim "
    "here cuts first)",
)


# ---------------------------------------------------------------------------
# [12] P10-2 (hostile pass #10): "12000 clears the hierarchy's documented
# 11,300-token construction capacity with margin" was measured against a
# fixture built at the NOMINAL per-tier maxima (9 L1 at exactly
# L1_MAX_TOKENS, 4 L2 at exactly L2_MAX_TOKENS, 1 L3 at exactly
# L3_MAX_TOKENS — [10]'s own `_G_HIER_TOKENS` fixture above). Her REAL
# tiers already exceed those maxima (measured 2026-09-17: 8 L1 chunks mean
# 561, max 792 against L1_MAX_TOKENS=500; 4 L2 chapters mean 1,102, max
# 1,271 against L2_MAX_TOKENS=1200), and `_do_l3_rollup`'s own comment
# documents that a stalled /tokenize routinely makes the L3 rollup give up
# and CONCATENATE 2-3 parts instead of summarizing them — storing that
# concatenation, which then carries into every later render. A hierarchy
# built from these MEASURED sizes plus a 2x-L3_MAX give-up concatenation
# renders close to p10's own reported 13,860 (this fixture: ~13,891 —
# small construction differences from a slightly different filler are
# expected; both exceed 12,000) — DECLINING at the old 12,000 default and
# REUSING at the new 15,000 one. This is the regression test LOOPS5_BRIEF
# asks for: it fails if a future default stops clearing this measured
# peak, the way 12,000 already did.
# ---------------------------------------------------------------------------
print("\n[12] P10-2: the ceiling clears a hierarchy sized from MEASURED tiers, "
      "not nominal maxima")

_G12_L1_MEAN, _G12_L1_MAX = 561, 792
_G12_L2_MEAN, _G12_L2_MAX = 1102, 1271

_g12_l1 = [
    {"tier": "l1", "text": "L1scene " + _g_filler(_G12_L1_MEAN),
     "first_turn": i * 10 + 1, "last_turn": i * 10 + 10}
    for i in range(8)
]
_g12_l1.append({
    "tier": "l1", "text": "L1scene " + _g_filler(_G12_L1_MAX),
    "first_turn": 81, "last_turn": 90,
})
_g12_l2 = [
    {"tier": "l2", "text": "L2chapter " + _g_filler(_G12_L2_MEAN),
     "first_turn": 90 + i * 10 + 1, "last_turn": 90 + i * 10 + 10}
    for i in range(4)
]
_G12_LAST = _g12_l2[-1]["last_turn"]
check(
    _G12_LAST == 130,
    f"fixture: [12]'s hierarchy covers the same 130 turn positions as "
    f"[10]'s (got {_G12_LAST})",
)


def _g12_shipped_sbmax_from(path: Path, pattern: str) -> int:
    """Read the SHIPPED COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS straight out of
    the deploy config, not a literal pinned in this file — so a future
    revert of the shipped value (Dockerfile or runpod.env.template) fails
    THIS test instead of leaving a stale number here to agree with it."""
    text = path.read_text(encoding="utf-8")
    m = re.search(pattern, text)
    check(bool(m), f"fixture: found {pattern!r} in {path.name}")
    return int(m.group(1)) if m else -1


_G12_ROOT = Path(__file__).resolve().parent.parent
_G12_SHIPPED_DOCKERFILE = _g12_shipped_sbmax_from(
    _G12_ROOT / "Dockerfile",
    r'ENV COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS="(\d+)"',
)
_G12_SHIPPED_RUNPOD = _g12_shipped_sbmax_from(
    _G12_ROOT / "runpod.env.template",
    r"COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS=(\d+)",
)
check(
    _G12_SHIPPED_DOCKERFILE == _G12_SHIPPED_RUNPOD,
    f"*** P10-2: Dockerfile ({_G12_SHIPPED_DOCKERFILE}) and "
    f"runpod.env.template ({_G12_SHIPPED_RUNPOD}) ship the SAME "
    f"COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS default — the two are hand-kept "
    f"in sync (see each file's own comment) and this is what would catch "
    f"them drifting apart",
)
_G12_SHIPPED_SBMAX = _G12_SHIPPED_DOCKERFILE

# The give-up L3: `_do_l3_rollup`'s own comment measures 2-3 concatenated
# parts, each near L3_MAX_TOKENS, so 2x is the routine case, not a
# hand-picked worst case.
_g12_l3_giveup = {
    "text": "L3theme " + _g_filler(2 * summarizer.L3_MAX_TOKENS),
    "first_turn": 1, "last_turn": _G12_LAST,
}
_g12_state = {"l1": _g12_l1, "l2": _g12_l2, "l3": _g12_l3_giveup}

_g12_saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
summarizer.SUMMARY_BLOCK_MAX_TOKENS = 10**9
_G12_RENDER = summarizer._estimate_block_tokens(
    summarizer.format_summary_block(_g12_state, 10**9) or ""
)
summarizer.SUMMARY_BLOCK_MAX_TOKENS = _g12_saved_sbmax
_G12_OLD_SBMAX = 12000  # the retired default (P9-1/P9-2) — a fixed
# historical reference point, not something that should track the deploy
# config the way the SHIPPED value below does.
check(
    _G12_RENDER > _G12_OLD_SBMAX,
    f"fixture: the measured-tier-plus-give-up-L3 hierarchy renders above "
    f"the OLD {_G12_OLD_SBMAX} default ({_G12_RENDER}) — otherwise this "
    f"fixture does not reproduce the regression at all",
)
check(
    _G12_RENDER < _G12_SHIPPED_SBMAX,
    f"fixture: ... and below the shipped default ({_G12_RENDER} < "
    f"{_G12_SHIPPED_SBMAX}) — otherwise this fixture cannot show the "
    f"shipped default clearing it either",
)

for _g12_budget, _g12_expect_reuse in (
    (_G12_OLD_SBMAX, False), (_G12_SHIPPED_SBMAX, True),
):
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = _g12_budget
    try:
        _g12_out = summarizer.format_summary_block(
            _g12_state, _g12_budget, all_or_nothing=True
        )
    finally:
        summarizer.SUMMARY_BLOCK_MAX_TOKENS = _g12_saved_sbmax
    _g12_reused = bool(_g12_out)
    check(
        _g12_reused == _g12_expect_reuse,
        f"*** P10-2: at SUMMARY_BLOCK_MAX_TOKENS={_g12_budget}, a hierarchy "
        f"built from her MEASURED tier sizes plus a routine give-up L3 "
        f"{'REUSES' if _g12_expect_reuse else 'DECLINES'} as expected "
        f"(render={_G12_RENDER}, got {'REUSES' if _g12_reused else 'DECLINES'})",
    )
    if _g12_expect_reuse:
        check(
            all(f"L1scene " in _g12_out for _ in [0])
            and "L2chapter " in _g12_out
            and "L3theme " in _g12_out,
            "*** P10-2: at the shipped default every tier travels into the "
            "stand-in whole — not just the newest scenes a partial fit "
            "would keep",
        )


# ---------------------------------------------------------------------------
# [F1] p7 hostile pass #7: the compacted-branch floor (main.py:_enforce_hard_
# budget, `_floor`) used to be the raw KEEP_RECENT_TURNS MESSAGE count (4).
# split_messages ALIGNS its own kept-recent window to start on a USER turn
# (leading non-user turns move into the summarized portion — a template
# requirement), so the window it actually protects can hold FEWER than
# KEEP_RECENT_TURNS messages. With an old, unpaired turn sitting where the
# 4th-from-end slot falls, the old floor counted it as "recent" and
# protected it from the pre-shed loop — spending injected memory (halving,
# then dropping facts) to keep a turn that was never actually inside the
# aligned recent window.
#
# P8-1 (hostile pass #8): the fix that landed for p7's F1 only stripped a
# leading turn of the WRONG ROLE from the aligned window — which happened
# to fully cover the ASSISTANT-role stub this section used to test with
# (an assistant turn at the front is never "user", so the existing
# `!= "user"` check already stripped it), but not what production actually
# sends. OpenWebUI puts an uploaded image on a USER turn (RUNPOD_DEPLOY.md:
# OpenAI's own multimodal format), never an assistant one, and
# compact_if_needed inserts its preserved old images directly in front of
# the true keep_recent window. So the real tail is
# [old_image(USER), prev-u(USER), prev-a(assistant), newest(USER)]: it
# ALREADY "starts on a user turn" whichever of the two front entries it is,
# so the role-only check stripped nothing and the floor stayed unaligned
# for this shape — the exact production regression the fixture below now
# exercises as the PRIMARY case. The floor now also strips a leading entry
# that shares its role with the entry right after it (a real recent window
# always alternates roles), which handles this. Both role shapes are
# tested below: "user" is production's own shape; "assistant" is kept as a
# CONTROL, the shape the pre-P8-1 fix already handled correctly (so it
# must keep passing unchanged).
#
# This tests main._enforce_hard_budget directly (like test_p5_guard.py,
# which owns this same function's other branches), with its own
# deterministic byte-counting stand-in for count_tokens so the numbers
# here do not depend on a live tokenizer.
# ---------------------------------------------------------------------------
print("\n[F1] guard floor is ALIGNED like split_messages, not a raw message count")


def _f1_tokens(msgs) -> int:
    return sum(len(main._message_text(m).encode("utf-8")) + 4 for m in msgs)


def _f1_image_turn(tag, n_chars=30000):
    """A USER turn carrying an uploaded image — OpenWebUI's real shape.
    Padded the same as the pre-P8-1 assistant stub so an incorrectly-kept
    turn still forces a real choice against memory (this test only needs
    "expensive enough to matter", not vLLM's own image-token accounting).
    `n_chars` is a parameter (P10-1, hostile pass #10) so a caller can tune
    how EXPENSIVE the misjudged turn is — see `_f1_build`'s own comment on
    `old_exchanges=0, n_chars=11000` for why the default (30000, paired
    with `old_exchanges=30`) alone stopped being enough to catch a floor
    miscalculation once the guard could pay a small overshoot with memory
    instead of shedding a turn."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": tag + " " + "i" * n_chars},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ],
    }


def _f1_build(old_exchanges=30, image_role="user", image_chars=30000):
    """A compaction stand-in, injected memory, `old_exchanges` ordinary old
    pairs, then one UNPAIRED old turn (the image stand-in, role
    `image_role`) right before the previous exchange and the newest turn —
    exactly the slot `split_messages` would push out of its aligned
    keep_recent window (KEEP_RECENT_TURNS=4 non-system messages ending in
    [image, prev-u, prev-a, newest-u]: with image_role="assistant" the
    4-message tail starts on ASSISTANT, so the role check alone already
    strips it; with image_role="user" (production's own shape, P8-1) the
    tail starts on USER either way, so only the role-ALTERNATION check
    added for P8-1 catches it — leaving keep_recent=[prev-u, prev-a,
    newest-u], 3 messages, not 4, in both cases)."""
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
    if image_role == "user":
        msgs.append(_f1_image_turn("old-image-stub", n_chars=image_chars))
    else:
        msgs.append({"role": "assistant", "content": "old-image-stub " + "i" * image_chars})
    msgs.append({"role": "user", "content": "prev-u " + "u" * 150})
    msgs.append({"role": "assistant", "content": "prev-a " + "a" * 1650})
    msgs.append({"role": "user", "content": "newest " + "n" * 200})
    return msgs


_F1_LIMIT = 20768 - 90  # the shipped v3.1.9 effective limit, less a time-line reserve

for _f1_image_role in ("user", "assistant"):
    _f1_saved_count_tokens = main.count_tokens
    _f1_saved_count_tokens_exact = main.count_tokens_exact
    _f1_saved_margin = main._BUDGET_MARGIN
    main.count_tokens = _f1_tokens
    main.count_tokens_exact = lambda ms, *a, **k: _f1_tokens(ms)
    main._BUDGET_MARGIN = 0
    try:
        _f1_msgs = _f1_build(image_role=_f1_image_role)
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
        "old-image-stub" in main._message_text(m) for m in _f1_out
    )
    check(_f1_rep.get("fits"), f"[{_f1_image_role}] fixture: the guard fit the payload ({_f1_rep})")
    check(
        _f1_facts_whole,
        f"*** F1 [{_f1_image_role}]: facts survive whole — the unpaired old "
        f"turn is shed by the pre-shed loop instead of being counted as "
        f"'recent' and protected at memory's expense",
    )
    check(
        not _f1_image_survived,
        f"*** F1 [{_f1_image_role}]: the unpaired old turn does NOT survive "
        f"alongside whole facts — if it did, this fixture is not exercising "
        f"the floor bug at all",
    )

    # CONTROL: the actual protected window (prev-u, prev-a, newest) always
    # survives regardless of the floor fix — this proves the fix does not
    # over-shed into turns split_messages really would protect.
    _f1_recent_survived = all(
        any(
            m.get("role") == exp_role and main._message_text(m).startswith(exp_prefix)
            for m in _f1_out
        )
        for exp_role, exp_prefix in (
            ("user", "prev-u"), ("assistant", "prev-a"), ("user", "newest"),
        )
    )
    check(
        _f1_recent_survived,
        f"CONTROL [{_f1_image_role}]: the truly-recent window (prev-u, "
        f"prev-a, newest) still survives — the fix narrows the floor, it "
        f"does not remove protection for what split_messages actually "
        f"keeps",
    )


# ---------------------------------------------------------------------------
# [F1-lean] P10-1 (hostile pass #10) made the section above stop catching a
# MISCOMPUTED floor. `_f1_build`'s default `old_exchanges=30` puts 30
# deferred exchanges above whatever the floor is — the unconditional
# "shed everything above the floor" pass sheds the image (and plenty more)
# regardless of whether the floor is 3 (correct) or 4 (P8-1's bug: the
# unpaired image miscounted as "recent"), so the ABOVE loop's own
# assertions stayed green even with `main.py`'s same-role alignment check
# disabled outright (checked directly against this file, needle
# `samerole` in SP\lpmut\mut.py — see this lane's report for the numbers).
# The new below-floor arithmetic (P10-1) only consults the floor's exact
# value in ONE place: whether spending every spendable injected block
# could cover the gap on its own. With no deferred history and a LARGE
# image (the section above), the gap is far bigger than the facts block
# alone could ever cover, so the image gets shed either way and a wrong
# floor is invisible. A SMALL gap — one memory alone COULD cover, if the
# floor's miscalculation let the guard reach for it — is what still
# distinguishes correct from broken: at the exact image size below,
# `_floor` correct (3) sheds the image unconditionally BEFORE memory is
# ever considered; `_floor` wrong (4, matching the pre-P8-1 bug exactly)
# lets the guard reach for memory instead, since the gap and the facts
# block happen to be comparable sizes.
# ---------------------------------------------------------------------------
print("\n[F1-lean] P10-1: a SMALL gap keeps the floor's exact value load-bearing")

_f1_saved_count_tokens = main.count_tokens
_f1_saved_count_tokens_exact = main.count_tokens_exact
_f1_saved_margin = main._BUDGET_MARGIN
main.count_tokens = _f1_tokens
main.count_tokens_exact = lambda ms, *a, **k: _f1_tokens(ms)
main._BUDGET_MARGIN = 0
try:
    _f1l_msgs = _f1_build(old_exchanges=0, image_role="user", image_chars=11000)
    _f1l_rep: dict = {}
    _f1l_out = main._enforce_hard_budget(_f1l_msgs, _F1_LIMIT, 1, _f1l_rep)
finally:
    main.count_tokens = _f1_saved_count_tokens
    main.count_tokens_exact = _f1_saved_count_tokens_exact
    main._BUDGET_MARGIN = _f1_saved_margin

_f1l_mem_out = [
    x for x in _f1l_out
    if x.get("role") == "system" and "[Facts]" in (x.get("content") or "")
]
_f1l_facts_whole = bool(_f1l_mem_out) and all(
    f"FACT{i:02d}" in _f1l_mem_out[0]["content"] for i in range(9)
)
_f1l_image_survived = any(
    "old-image-stub" in main._message_text(m) for m in _f1l_out
)
check(_f1l_rep.get("fits"), f"[F1-lean] fixture: the guard fit the payload ({_f1l_rep})")
check(
    not _f1l_image_survived,
    f"*** F1-lean: the unpaired old image does NOT survive, even though "
    f"memory alone could have covered this small a gap — the floor is "
    f"computed correctly (3, not 4), so the unconditional above-floor "
    f"shed removes it before memory is ever consulted ({_f1l_rep})",
)
check(
    _f1l_facts_whole,
    "*** F1-lean: facts survive whole — memory was never touched because "
    "the image alone already covered the gap",
)


# ---------------------------------------------------------------------------
# [13] P10-5 (hostile pass #10, LOW): the stand-in's ceiling, as a FRACTION
# of the window, grows as `max_tokens` grows past the generation reserve —
# `_standin_reuse_ceiling` is `min(SUMMARY_BLOCK_MAX_TOKENS, inject_budget)`
# with nothing bounding it relative to `effective_limit` itself, so once
# `inject_budget` (which shrinks with `effective_limit`) drops below
# SUMMARY_BLOCK_MAX_TOKENS, the ceiling becomes a flat
# INJECTION_BUDGET_FRACTION (75% shipped) of whatever window is left. Not
# reachable at RUNPOD_DEPLOY.md's recommended Max Tokens (12000, equal to
# the generation reserve, so effective_limit never shrinks there); reachable
# if Max Tokens is raised past the reserve. Documented in RUNPOD_DEPLOY.md's
# "Max Tokens" section and `_standin_reuse_ceiling`'s own docstring; pinned
# here so a future change to SUMMARY_BLOCK_MAX_TOKENS, GENERATION_RESERVE or
# INJECTION_BUDGET_FRACTION is measured, not silently different. No test
# varied max_tokens on the reuse path before this (p7's own F7 note, true
# until now).
# ---------------------------------------------------------------------------
print("\n[13] P10-5: the stand-in ceiling's share of the window across max_tokens")

_G13_SHIPPED_FRACTION = 0.75  # COMPACTOR_INJECTION_BUDGET_FRACTION, shipped


def _g13_ceiling_share(max_tokens: int):
    """(effective_limit, inject_budget, ceiling, ceiling/effective_limit)
    at the SHIPPED SUMMARY_BLOCK_MAX_TOKENS (read from Dockerfile/
    runpod.env.template above, `_G12_SHIPPED_SBMAX` — the same value [12]
    already proved the two files agree on) and the shipped injection
    fraction, for one `max_tokens` value."""
    eff_limit = min(
        main.MAX_MODEL_LEN,
        max(256, main.MAX_MODEL_LEN - max(main.GENERATION_RESERVE, max_tokens)),
    )
    inject_budget = int(eff_limit * _G13_SHIPPED_FRACTION)
    saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = _G12_SHIPPED_SBMAX
    try:
        ceiling = main._standin_reuse_ceiling(inject_budget)
    finally:
        summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved_sbmax
    return eff_limit, inject_budget, ceiling, ceiling / eff_limit


_G13_AT_RECOMMENDED = _g13_ceiling_share(12000)  # RUNPOD_DEPLOY.md's recommendation
_G13_ABOVE_RESERVE = _g13_ceiling_share(20000)   # well past the reserve

check(
    _G13_AT_RECOMMENDED[3] < _G13_SHIPPED_FRACTION - 1e-9,
    f"*** P10-5: at the RECOMMENDED Max Tokens (12000, == the generation "
    f"reserve), the stand-in ceiling stays BELOW the flat "
    f"INJECTION_BUDGET_FRACTION share {_G13_AT_RECOMMENDED} — P10-5 is "
    f"not triggered at documented settings",
)
check(
    abs(_G13_ABOVE_RESERVE[3] - _G13_SHIPPED_FRACTION) < 0.005,
    f"*** P10-5: well past the reserve, inject_budget becomes the binding "
    f"term and the ceiling converges on the flat "
    f"INJECTION_BUDGET_FRACTION share {_G13_ABOVE_RESERVE} — the "
    f"documented worst case",
)
check(
    _G13_ABOVE_RESERVE[3] > _G13_AT_RECOMMENDED[3],
    f"*** P10-5: the ceiling's share of the window is MONOTONE with "
    f"max_tokens past the reserve — raising Max Tokens only ever grows "
    f"the stand-in's share, never shrinks it (at-recommended share="
    f"{_G13_AT_RECOMMENDED[3]:.3f}, above-reserve share="
    f"{_G13_ABOVE_RESERVE[3]:.3f})",
)


# ---------------------------------------------------------------------------
# [14] P11-6 (hostile pass #11): `_standin_reuse_ceiling` never subtracted
# `_others` (system prompt + preserved images + the recent window) at all —
# deliberate, per its own docstring, so a long recent TURN could not starve
# the stand-in the way it did before P9-1/P9-2. But once the hierarchy
# passes roughly 11k real tokens (her first L3, particularly a give-up
# concatenation — P10-2), that flat ceiling — not `_target_based_budget`,
# which DOES subtract `_others` — is the one `max()` always picks, so
# `_others` stops mattering to the reuse DECISION at exactly the size where
# it starts mattering to the GUARD: the stand-in is excluded from every
# trim/drop stage in the compacted branch by design, so a large enough one
# leaves the branch nothing to spend but injected memory and then the
# previous exchange, on a hierarchy state the declined path would have
# handled by shedding old VERBATIM turns oldest-first, never touching
# keep_recent. Measured on a real branch at the ceiling's own boundary:
# reuse loses the previous exchange at 14-34 of 474 positions where
# declining would have kept it (0-1 the reverse), once the hierarchy passes
# ~11-13k real tokens.
#
# Three hierarchy states, all built the way [12] builds its measured-tier
# fixture (her OWN measured mean/max L1/L2 sizes, reused from [12] above —
# `_G12_L1_MEAN`, `_G12_L2_MEAN`), plus her measured 755-character system
# prompt and two preserved images (her MAX_RETAINED_IMAGES=2 shape,
# `image_turn` — real cost is whatever `count_tokens` prices an image at,
# not asserted here) and a REALISTIC (not the P11-4 pathological single
# ~17k-token reply) large recent exchange (~8k-token A_prev — well inside
# her measured range, short of her single largest recorded reply):
#   "today"  ~9.1k estimator tokens, no L3          — must REUSE (her state now)
#   "peakA" ~11.5k estimator tokens, an L3 (1869)   — must REUSE (CONTROL: this
#            is NOT "decline once the hierarchy is merely large" — p11's own
#            measurement has reuse still winning here, 0 vs 17 positions)
#   "peakB" ~13.6k estimator tokens, a give-up L3   — must DECLINE (this is
#            the state P11-6 is about: p11 measured reuse strictly WORSE
#            here, 16 vs 0 positions against declining)
# ---------------------------------------------------------------------------
print("\n[14] P11-6: the reuse ceiling yields once the hierarchy would squeeze "
      "the recent window it is supposed to sit beside")

# The new check recovers `effective_limit` from `inject_budget /
# INJECTION_BUDGET_FRACTION` (the two numbers `_standin_reuse_ceiling`'s
# caller has — see that check's own comment in main.py): true at every REAL
# call site, where chat_completions computes both from the SAME request
# with the SAME live fraction. This file's module-level env sets
# COMPACTOR_INJECTION_BUDGET_FRACTION="0.6" (the 2026-09-16 incident
# number, used unchanged by [1]-[13] above) while `_G_PLANNED_INJECT`
# above is a LITERAL 0.75-based figure — exactly the mismatch that
# recovery is not safe against, and proof this section would silently
# under-test the fix without correcting it here.
_G14_SAVED_FRACTION = main.INJECTION_BUDGET_FRACTION
main.INJECTION_BUDGET_FRACTION = 0.75  # matches _G_PLANNED_INJECT's own fraction

_G14_CONV = "reuse_fit_window"


def _g14_hierarchy(n_l1: int, l3_tokens: int | None):
    """A hierarchy built from her MEASURED tier sizes ([12]'s own
    `_G12_L1_MEAN`/`_G12_L2_MEAN`, not the nominal per-tier maxima [10]
    uses), `n_l1` L1 chunks (8 for "today", 9 — an extra chunk, the same
    shape [10]'s docstring describes — for the two peak states) and an
    optional L3 (a give-up-sized concatenation for peakB, same construction
    [12] uses for its own give-up L3)."""
    older = history(n_l1 * 5 + 20, words=200)
    l1 = [
        {"tier": "l1", "text": "L1scene " + _g_filler(_G12_L1_MEAN),
         "first_turn": i * 10 + 1, "last_turn": i * 10 + 10}
        for i in range(n_l1)
    ]
    l2 = [
        {"tier": "l2", "text": "L2chapter " + _g_filler(_G12_L2_MEAN),
         "first_turn": n_l1 * 10 + i * 10 + 1, "last_turn": n_l1 * 10 + i * 10 + 10}
        for i in range(4)
    ]
    last = l2[-1]["last_turn"]
    st = _seed_hierarchy(_G14_CONV, older, l1 + l2)
    if l3_tokens:
        st["l3"] = {
            "text": "L3theme " + _g_filler(l3_tokens),
            "first_turn": 1, "last_turn": last,
        }
        summarizer.save_state(_G14_CONV, st)
        st = summarizer.load_state(_G14_CONV)
    saved = summarizer.SUMMARY_BLOCK_MAX_TOKENS
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = 10**9
    render = summarizer._estimate_block_tokens(
        summarizer.format_summary_block(st, 10**9) or ""
    )
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved
    return older, last, render


def _g14_run(n_l1: int, l3_tokens: int | None, aprev_tokens: int):
    """Her measured shape end to end: a 755-char system prompt, two
    preserved images ahead of the recent window, and a REALISTIC large
    reply — real `compact_if_needed` at the shipped numbers (P10-2's
    15000/1.0), same as [10]/[11] above."""
    older, last, render = _g14_hierarchy(n_l1, l3_tokens)
    older = list(older)
    older[0] = {"role": "system", "content": "S" * 755}
    older.insert(5, image_turn("user", "sketch-a"))
    older.insert(7, image_turn("user", "sketch-b"))
    recent = [
        fat_turn("user", 100, "prev-u"),
        fat_turn("assistant", aprev_tokens, "prev-a"),
        fat_turn("user", 100, "newest-u"),
    ]
    msgs = older + recent
    saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = _G12_SHIPPED_SBMAX
    stored_out: list = []
    with capture() as records:
        try:
            out = _run(
                msgs, _G14_CONV, stored_turns_out=stored_out,
                inject_budget=_G_PLANNED_INJECT,
            )
        finally:
            summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved_sbmax
    return stored_out, out, last, render, records.records


main.summarize = _spy_summarize

_g14_today_stored, _g14_today_out, _g14_today_last, _g14_today_render, _ = (
    _g14_run(8, None, 8000)
)
check(
    9000 <= _g14_today_render <= 9300,
    f"fixture: 'today' hierarchy renders near her measured 9,047 (got "
    f"{_g14_today_render})",
)
check(
    _g14_today_stored == [_g14_today_last],
    f"*** P11-6 [14] 'today': her CURRENT hierarchy still reuses completely "
    f"(stored_turns_out={_g14_today_stored}) — this fix must not turn off "
    f"reuse for the state she is actually in",
)

_g14_pA_stored, _g14_pA_out, _g14_pA_last, _g14_pA_render, _ = (
    _g14_run(9, 1869, 8000)
)
check(
    11300 <= _g14_pA_render <= 11700,
    f"fixture: 'peakA' hierarchy renders near her projected 11,728 (got "
    f"{_g14_pA_render})",
)
check(
    _g14_pA_stored == [_g14_pA_last],
    f"*** P11-6 [14] 'peakA' CONTROL: reuse still fires here (stored_turns_"
    f"out={_g14_pA_stored}) — proves this fix is NOT 'decline once the "
    f"hierarchy is merely large': p11 measured reuse still winning at this "
    f"state (0 positions lost vs 17 for declining)",
)
_g14_pA_ns = [m for m in _g14_pA_out if m.get("role") != "system"]
check(
    any("prev-u" in main._message_text(m) for m in _g14_pA_ns)
    and any("prev-a" in main._message_text(m) for m in _g14_pA_ns),
    "[14] 'peakA': the stand-in itself still carries the recent exchange "
    "(compact_if_needed never removes keep_recent) — sanity check before "
    "the guard runs on it below",
)

_g14_pB_stored, _g14_pB_out, _g14_pB_last, _g14_pB_render, _g14_pB_log = (
    _g14_run(9, 4000, 8000)
)
check(
    13400 <= _g14_pB_render <= 13900,
    f"fixture: 'peakB' hierarchy (a give-up L3 concatenation, [12]'s own "
    f"construction) renders near p10's projected 13,859 (got "
    f"{_g14_pB_render})",
)
check(
    _g14_pB_stored == [0],
    f"*** P11-6 [14] 'peakB': the SAME shipped ceiling that reuses whole "
    f"at 'today' and 'peakA' now DECLINES (stored_turns_out="
    f"{_g14_pB_stored}) — the hierarchy alone fits under "
    f"SUMMARY_BLOCK_MAX_TOKENS (15000), so only the new structural check "
    f"(stand-in + system prompt + up to MAX_RETAINED_IMAGES images against "
    f"the request's real window) can be what declined it",
)
check(
    _find(_g14_pB_log, "no room") is not None,
    "*** P11-6 [14]: the decline is logged under the NEW reason (window "
    "squeeze), not silently folded into the old 'does not fit whole in "
    "the ... token(s) TARGET/injection budget leaves' wording",
)
# P12-2 (hostile pass #12): the log line changing wording is not proof the
# RECORDED reason changed too — `checks.reuse`/`reuse_decline_state()` is
# what an operator (and the health endpoint) actually reads, and it is a
# SEPARATE code path from the log line. Read it here, right after the
# real `compact_if_needed` call above that produced this decline, not via
# a direct `_record_reuse_outcome()` call (test_health_findings.py's
# `test_p12_2_...` already covers the RECORDER in isolation; this checks
# that `compact_if_needed` actually calls it with "window", end to end).
_g14_pB_decline_state = main.reuse_decline_state()
check(
    _g14_pB_decline_state.get("last_reason") == "window",
    f"*** P12-2 [14] 'peakB': main.reuse_decline_state() records "
    f"last_reason='window' for THIS decline, not 'budget' (got "
    f"{_g14_pB_decline_state.get('last_reason')!r}) — end-to-end proof "
    f"that compact_if_needed's window-squeeze branch actually calls "
    f"_record_reuse_outcome with the new reason, not just a differently-"
    f"worded log line",
)


def _g14_guard(out, last, tag):
    """The rest of what chat_completions does after compact_if_needed:
    persona + injected memory, then the real hard-budget guard — same
    pattern [11]'s `_g11_build_and_guard` uses."""
    standin = next((m for m in out if main._is_compaction_standin(m)), None)
    rest = [m for m in out if m.get("role") != "system"]
    full = [out[0], {"role": "system", "content": "P" * 2500}]
    if standin is not None:
        full.append(standin)
    full.append({"role": "system", "content": _G11_MEM})
    full += rest
    rep: dict = {}
    result = main._enforce_hard_budget(full, EFFECTIVE_LIMIT, 1, rep)
    ns_result = [m for m in result if m.get("role") != "system"]
    prev_survived = (
        any("prev-u" in main._message_text(m) for m in ns_result)
        and any("prev-a" in main._message_text(m) for m in ns_result)
    )
    newest_survived = any(
        "newest-u" in main._message_text(m) for m in ns_result
    )
    return rep, prev_survived, newest_survived


_g14_pB_rep, _g14_pB_prev, _g14_pB_newest = _g14_guard(
    _g14_pB_out, _g14_pB_last, "peakB"
)
check(_g14_pB_rep.get("fits") is True, f"[14] 'peakB': the guard fits the payload ({_g14_pB_rep})")
check(
    _g14_pB_prev,
    f"*** P11-6 [14] 'peakB' END TO END: with reuse declined, her previous "
    f"exchange (U_prev/A_prev) survives the real guard ({_g14_pB_rep}) — "
    f"the SAME array, still reused (not run here — see the mutation table "
    f"in SP\\fix-3193-reuse.md), loses it under the shipped code this fix "
    f"changes",
)
check(_g14_pB_newest, "[14] 'peakB': the newest turn always survives")

# Cross-pricing (P12-1, hostile pass #12): this is exactly the residual the
# shipped (90e3698) check left open, and this fix's whole point. The OLD
# check reserved `count_tokens(system + preserved_images)` — priced BY
# `count_tokens`, so it rose and fell with whichever tier of that function
# happened to run (a flat `IMAGE_TOKEN_ESTIMATE` before opencv, the
# template's own real per-resolution cost after it — measured ~3,080
# square, ~2,352 for a 4:3 photo). At the real, lower price the OLD
# reserve shrank, 'peakB' reused again, and the guard lost the exchange —
# proven on a real branch (SP\p12-findings.md, P12-1: 13-47 of 474
# positions depending on state). The fix reserves `system + keep_recent`
# instead: NOT an image price at all, so it does not move when
# `count_tokens`'s image pricing does. Pinned at all three prices the
# brief requires the decision to hold under: the shipped flat default
# (4,096, checked above at 'today'/'peakA'/'peakB'), and here the two real
# per-resolution costs p12 measured (3,080 square, 2,352 for a 4:3 photo)
# — 'peakB' must decline at EVERY one of them, and 'peakA' (a smaller
# hierarchy, well inside the reserve either way) must still reuse at every
# one, so this is not "decline whenever an image is cheap" any more than
# [14]'s original CONTROL was "decline whenever the hierarchy is large".
for _g14_price, _g14_label in ((3080, "3,080 (opencv, square)"), (2352, "2,352 (opencv, 4:3 photo)")):
    _g14_saved_image_tokens = main.IMAGE_TOKEN_ESTIMATE
    main.IMAGE_TOKEN_ESTIMATE = _g14_price
    try:
        _g14_pB2_stored, _g14_pB2_out, _g14_pB2_last, _g14_pB2_render, _ = (
            _g14_run(9, 4000, 8000)
        )
        _g14_pA2_stored, _g14_pA2_out, _g14_pA2_last, _g14_pA2_render, _ = (
            _g14_run(9, 1869, 8000)
        )
    finally:
        main.IMAGE_TOKEN_ESTIMATE = _g14_saved_image_tokens
    check(
        _g14_pB2_stored == [0],
        f"*** P12-1 [14] cross-pricing at {_g14_label}: 'peakB' still "
        f"DECLINES (stored_turns_out={_g14_pB2_stored}) — the reserve is "
        f"`system + keep_recent`, not an image price, so it does not "
        f"reopen the gap the shipped check left when images price below "
        f"the flat 4,096 estimate",
    )
    check(
        _g14_pA2_stored == [_g14_pA2_last],
        f"[14] cross-pricing at {_g14_label}: 'peakA' still REUSES "
        f"(stored_turns_out={_g14_pA2_stored}) — a cheaper image does not "
        f"make this fix decline a hierarchy it was already happy with",
    )

main.INJECTION_BUDGET_FRACTION = _G14_SAVED_FRACTION


# ---------------------------------------------------------------------------
# [15] P11-4 (hostile pass #11): [11a]/[11b] above only run with `/tokenize`
# ABSENT (`count_tokens_exact` returns `None`, so `per[]`'s local*scale
# estimate IS the only "ground truth" the branch has — self-consistent by
# construction, no mismatch possible). Production normally has `/tokenize`
# available, and P10-1's `_mem_ceiling` gate decided "would spending every
# spendable block cover the rest?" by that SAME scaled arithmetic even
# then — comparing it against the round's own EXACT verify a few lines
# down, which can disagree once an image is in play (the local counter
# prices one at the flat `IMAGE_TOKEN_ESTIMATE`; the exact count, when
# `/tokenize` works, prices it at its real, usually lower, cost). When they
# disagree enough, round 1 can wrongly decide memory covers the gap, spend
# it (trimmed here, not dropped — this fixture does not push it that far),
# and the round's own exact verify catches the shortfall anyway — so round
# 2 re-enters with memory ALREADY spent and correctly sheds the previous
# exchange regardless, making the round-1 spend pure waste. Reproduces on
# the SAME at-capacity hierarchy [10]/[11] already proved reuses, her real
# shape's two preserved OLD images ahead of the recent window (P11-1's own
# caveat: 21 of her real image turns sit exactly there), and an A_prev
# large enough that the exchange must go regardless of what memory does —
# the interesting question is only whether memory is wastefully spent
# first.
# ---------------------------------------------------------------------------
print("\n[15] P11-4: the round loop decides on GROUND TRUTH when an exact "
      "counter is present, not scaled arithmetic that can disagree on images")


def _g15_exact_counter(msgs, *_a, **_k):
    """A realistic exact counter for this offline test: ASCII text at
    chars/4 (matching the local estimator's own no-tokenizer fallback, so
    TEXT prices agree — the mismatch this section is about is IMAGES
    only), images at their measured real cost (p11: <=3,080), well under
    `IMAGE_TOKEN_ESTIMATE` (main.py's own default 4,096, the price while
    the chat template cannot be applied to an image-bearing list — lane
    opencv's concern this round, not this file's; this stub does not
    depend on which value `IMAGE_TOKEN_ESTIMATE` currently holds)."""
    total = 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    total += len(part.get("text") or "") // 4
                elif part.get("type") == "image_url":
                    total += 3080
        else:
            total += len(c or "") // 4
        total += 3
    return total


_G15_OLDER = list(G_OLDER)
_G15_OLDER.insert(5, image_turn("user", "old-sketch-a"))
_G15_OLDER.insert(7, image_turn("user", "old-sketch-b"))
_G15_CONV = "reuse_fit_p11_4_exact"
_seed_hierarchy(_G15_CONV, _G15_OLDER, _G_L1_CHUNKS + _G_L2_CHUNKS)
_g15_st = summarizer.load_state(_G15_CONV)
_g15_st["l3"] = dict(_G_ST["l3"]) if _G_ST.get("l3") else None
if _g15_st["l3"] is None:
    del _g15_st["l3"]
summarizer.save_state(_G15_CONV, _g15_st)
_G15_RECENT = [
    fat_turn("user", 100, "prev-u"),
    fat_turn("assistant", 10000, "prev-a"),
    fat_turn("user", 100, "newest-u"),
]
_G15_MSGS = _G15_OLDER + _G15_RECENT
_g15_saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
summarizer.SUMMARY_BLOCK_MAX_TOKENS = 12000
_g15_stored: list = []
try:
    _g15_out = _run(
        _G15_MSGS, _G15_CONV, stored_turns_out=_g15_stored,
        inject_budget=_G_PLANNED_INJECT,
    )
finally:
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = _g15_saved_sbmax
check(
    _g15_stored and _g15_stored[0] >= _G_LAST_COVERED - 2,
    f"[15] fixture: reuse fires on the at-capacity hierarchy with two old "
    f"images ahead of the recent window (stored_turns_out={_g15_stored}, "
    f"expected close to [{_G_LAST_COVERED}] — inserting the two images "
    f"after the chunk boundaries were drawn leaves 2 turns unpaired, "
    f"refreshed fresh rather than substituted; harmless for what this "
    f"section measures, the guard's memory-spend decision, not the exact "
    f"coverage count)",
)
_g15_standin = next(m for m in _g15_out if main._is_compaction_standin(m))
_g15_rest = [m for m in _g15_out if m.get("role") != "system"]
_G15_PERSONA = {"role": "system", "content": "P" * 2500}
_g15_full = [
    {"role": "system", "content": "you are a companion"},
    _G15_PERSONA, _g15_standin,
    {"role": "system", "content": _G11_MEM},
] + _g15_rest

_g15_saved_exact = main.count_tokens_exact
main.count_tokens_exact = _g15_exact_counter
try:
    _g15_rep: dict = {}
    _g15_result = main._enforce_hard_budget(
        _g15_full, EFFECTIVE_LIMIT, 1, _g15_rep
    )
finally:
    main.count_tokens_exact = _g15_saved_exact

check(
    _g15_rep.get("counted_by") == "vLLM's /tokenize",
    f"[15] fixture: the round actually used the exact counter, not a "
    f"fallback ({_g15_rep})",
)
check(_g15_rep.get("fits") is True, f"[15] the guard fits the payload ({_g15_rep})")
_g15_persona_out = next(
    (m for m in _g15_result if m.get("content") == _G15_PERSONA["content"]),
    None,
)
check(
    _g15_persona_out is not None,
    f"*** P11-4 [15]: persona survives WHOLE, byte for byte "
    f"(trimmed_blocks={_g15_rep.get('trimmed_blocks')}) — spending it "
    f"here could never have covered a 10k-token A_prev's gap once the two "
    f"images are already shed, so the exact-ground-truth decision does "
    f"not waste it before shedding the exchange that actually pays for "
    f"the request ({_g15_rep})",
)
_g15_ns = [m for m in _g15_result if m.get("role") != "system"]
check(
    not (
        any("prev-u" in main._message_text(m) for m in _g15_ns)
        and any("prev-a" in main._message_text(m) for m in _g15_ns)
    ),
    "[15]: the previous exchange does NOT survive — a 10k-token A_prev "
    "beside this hierarchy's stand-in genuinely cannot fit even with "
    "every spendable block gone; the fix changes WHETHER memory is spent "
    "on the way there, not THAT the exchange must go",
)
check(
    any("newest-u" in main._message_text(m) for m in _g15_ns),
    "[15]: the newest turn always survives",
)


# ---------------------------------------------------------------------------
# [16] P12-5 (hostile pass #12): the compacted-branch ordering ("shed old
# turns above the floor, THEN spend memory, THEN the floor") used to run
# only `if any(_is_compaction_standin(...) for i in _droppable_system_
# indices(...))` — i.e. only when the array carries compaction's OWN
# summary block. [F1]/[F1-lean] above both exercise that branch, but both
# of their fixtures ALWAYS include a stand-in system message
# (`main.COMPACTION_SUMMARY_HEADER`), so neither one can tell the OLD
# condition apart from the NEW, wider one (`_droppable_system_indices(...)`
# alone) — both trigger identically whenever a stand-in happens to be
# present. This section is deliberately the ONE place in this file that
# builds a guard input with injected memory (facts) and PLENTY of old
# verbatim turns to shed, but NO stand-in anywhere in the array — the
# declined-path shape P12-5 is actually about (reuse declined, or never
# attempted, so summarize() never produced a stand-in; facts/retrieval are
# still injected the way they are on every request). Before this fix, an
# array shaped like this fell through to the FLOOR-LESS generic "shed
# oldest non-system turn" loop, which has no idea a facts block sits right
# next to it and would shed the previous exchange once shedding old turns
# alone was not enough — exactly the loss measured on a real branch
# (SP\p12-findings.md, P12-5: 20-23 of 474 positions per hierarchy state,
# every one of them with 2.5-8.5k tokens of headroom left had memory been
# spent instead).
#
# Same deterministic byte-counting stand-in for count_tokens/
# count_tokens_exact as [F1]/[F1-lean] above (this tests
# main._enforce_hard_budget directly, not through compact_if_needed).
# ---------------------------------------------------------------------------
print("\n[16] P12-5: memory is spent before the previous exchange on ANY "
      "array carrying injected memory, not only a compacted one")


def _p125_build(old_exchanges: int, aprev_chars: int = 1650) -> list[dict]:
    """Persona + injected facts (NO compaction stand-in anywhere in this
    array — the declined-path shape), `old_exchanges` ordinary old pairs
    (standing in for the deferred verbatim backlog a declined request
    forwards), then the previous exchange and the newest turn. Mirrors
    `_f1_build` above minus the `COMPACTION_SUMMARY_HEADER` system
    message."""
    facts_block = "[Facts]\n" + "".join(
        f"- FACT{i:02d} she likes item {i} very much indeed.\n" for i in range(9)
    )
    facts_block = facts_block + "f" * (400 - len(facts_block))
    msgs = [
        {"role": "system", "content": "P" * 1200},
        {"role": "system", "content": facts_block},
    ]
    for i in range(old_exchanges):
        msgs.append({"role": "user", "content": f"old-u{i} " + "u" * 150})
        msgs.append({"role": "assistant", "content": f"old-a{i} " + "a" * 1650})
    msgs.append({"role": "user", "content": "prev-u " + "u" * 150})
    msgs.append({"role": "assistant", "content": "prev-a " + "a" * aprev_chars})
    msgs.append({"role": "user", "content": "newest " + "n" * 200})
    return msgs


def _p125_run(old_exchanges: int, aprev_chars: int = 1650):
    _saved_ct = main.count_tokens
    _saved_cte = main.count_tokens_exact
    _saved_margin = main._BUDGET_MARGIN
    main.count_tokens = _f1_tokens
    main.count_tokens_exact = lambda ms, *a, **k: _f1_tokens(ms)
    main._BUDGET_MARGIN = 0
    try:
        _msgs = _p125_build(old_exchanges, aprev_chars)
        check(
            not any(main._is_compaction_standin(m) for m in _msgs),
            "[16] fixture: no compaction stand-in anywhere in this array — "
            "the shape this section exists to cover",
        )
        _rep: dict = {}
        _out = main._enforce_hard_budget(_msgs, _F1_LIMIT, 1, _rep)
    finally:
        main.count_tokens = _saved_ct
        main.count_tokens_exact = _saved_cte
        main._BUDGET_MARGIN = _saved_margin
    _mem_out = [
        x for x in _out
        if x.get("role") == "system" and "[Facts]" in (x.get("content") or "")
    ]
    _facts_whole = bool(_mem_out) and all(
        f"FACT{i:02d}" in _mem_out[0]["content"] for i in range(9)
    )
    _ns = [m for m in _out if m.get("role") != "system"]
    _prev_survived = (
        any(main._message_text(m).startswith("prev-u") for m in _ns)
        and any(main._message_text(m).startswith("prev-a") for m in _ns)
    )
    return _rep, _facts_whole, _prev_survived, _ns


# [16a] A FEW old turns to shed (5 pairs — standing in for a modest
# deferred backlog) plus a previous exchange sized so that shedding EVERY
# old pair still leaves the array just over the limit, by less than what
# facts alone would free: the guard must choose between spending facts
# and shedding the previous exchange. P12-5: it must spend (or drop)
# facts, not the exchange — the SAME choice the compacted branch already
# made correctly when a stand-in was present ([F1] above). (18,900
# characters for A_prev is deliberately tuned so this is a CLOSE call at
# the floor boundary, not a case either ordering would resolve the same
# way — see the mutation note below.)
_p125a_rep, _p125a_facts, _p125a_prev, _p125a_ns = _p125_run(
    old_exchanges=5, aprev_chars=18900
)
check(_p125a_rep.get("fits") is True, f"[16a] the guard fits the payload ({_p125a_rep})")
check(
    _p125a_prev,
    f"*** P12-5 [16a]: the previous exchange (U_prev/A_prev) SURVIVES on a "
    f"NON-compacted array carrying injected memory — before this fix, the "
    f"floor-less generic shed loop (gated only on a stand-in being "
    f"present) would have reached U_prev/A_prev exactly like any other "
    f"'old' turn ({_p125a_rep})",
)
check(
    any(main._message_text(m).startswith("newest") for m in _p125a_ns),
    "[16a]: the newest turn always survives",
)

# [16b] CONTROL: with no old turns to shed at all and a previous exchange
# too large for facts alone to cover, the exchange still must pay — this
# is not "always protect the previous exchange no matter what", only
# "spend memory first". Mirrors [11a-mid]'s shape without going through
# compact_if_needed.
_p125b_rep, _p125b_facts, _p125b_prev, _p125b_ns = _p125_run(
    old_exchanges=0, aprev_chars=25000
)
check(_p125b_rep.get("fits") is True, f"[16b] CONTROL the guard fits the payload ({_p125b_rep})")
check(
    not _p125b_prev,
    f"*** P12-5 [16b] CONTROL: with nothing else left to shed and a reply "
    f"far bigger than facts alone can cover, the previous exchange is "
    f"what pays — the fix is an ordering choice (memory before the "
    f"turns above the floor), not a blanket 'never touch U_prev/A_prev' "
    f"rule ({_p125b_rep})",
)
check(
    any(main._message_text(m).startswith("newest") for m in _p125b_ns),
    "[16b] CONTROL: the newest turn always survives",
)


# ---------------------------------------------------------------------------
# [17] P12-6 (coordinator follow-up to P12-1, real-data replay, hostile
# pass #12): the P11-6/P12-1 reserve's fresh-summary allowance assumed ONE
# SUMMARY_MAX_TOKENS batch. The coordinator's real-data replay (her branch,
# 474 positions) found the "fresh+peakB" state — an uncovered tail past her
# last L1 rollup, ROUTINE per p11 (3 summarize() calls per reusing request
# between rollups), not an edge case — where the fresh summary actually
# costs 1,000-2,000 tokens (one or two UN-FOLDED map-reduce batches, not the
# single batch this reserve priced for).
#
# CORRECTION (hostile pass #13, P13-3): the coordinator's own "lost her
# previous exchange at 26 of 474 positions" number for this state was later
# found to be a replay-harness artifact, not a real measurement — the
# replay's spy appended its own ~1,000-token "fresh" text to a stand-in
# that already carried a real fresh summary for the same span, a shape no
# production request can reach (SP\p13-findings.md verified this in full:
# `_fresh_span_preview` previews exactly `fresh_input`'s own composition,
# and nothing joins the stand-in after this decision runs). The reserve
# gap this section tests for (one call priced, two or more actually
# needed) is real and still the point of [17]/[17a]/[17b] below; only the
# "26 of 474" citation was the artifact.
#
# [17a] reproduces that shape synthetically: her peakB hierarchy (~12,951
# measured tokens — [14]'s own peakA/peakB render 11,529/13,660 at
# l3_tokens 1,869/4,000, ~1:1 with l3_tokens since the L3 filler dominates
# the difference, so 3,291 interpolates to ~12,951) plus a genuinely large
# UNCOVERED TAIL (2 exchanges past the last L1/L2 chunk boundary, 4,000
# tokens each — P12-1's own comment names "an uncovered tail past
# stored_turns" as one of the two `_fresh_span_preview` triggers — sized so
# `_chunk_to_budget`'s pessimistic-scale estimate needs 2 batches, not 1:
# 16,000 raw tokens > the ~14,848-token threshold for a second 29,696-token
# batch), stubbed through a summarize() double that returns what an
# un-folded 2-batch reduce failure actually looks like: two concatenated
# near-cap chunks, ~1,800 tokens together — her measured magnitude, not
# derived from this fixture's own token count, so the stub cannot
# accidentally match whatever the fix under test predicts.
#
# [17b] confirms the OTHER half of the coordinator's ask (item 2): when a
# SMALLER fresh span (one batch, correctly reserved at flat SUMMARY_MAX_
# TOKENS either way) lets reuse fire with a fresh-summary-bearing stand-in,
# the guard's P12-5 ordering still spends injected memory before the
# previous exchange — proving the combined (stored + fresh) system message
# is still recognised as ONE stand-in by `_is_compaction_standin`, and nothing
# about attaching a fresh summary confuses the guard's own floor/memory
# choice downstream.
# ---------------------------------------------------------------------------
print("\n[17] P12-6: the fresh-summary reserve prices what summarize() can "
      "actually produce (possibly several un-folded batches), not a flat "
      "single one")


async def _g17_multibatch_summarize(client, to_summarize):
    """Simulates summarize()'s real worst case for a fresh span needing
    multiple map-reduce batches where the reduce phase never folds them —
    the reduce budget exhausted by the map phase, or two dense partials
    together still missing the reduce call's own input budget (see
    summarize()'s own "stopping the reduce" / reduce-failure branches).
    Returns TWO concatenated near-SUMMARY_MAX_TOKENS chunks (~1,800 tokens
    together) regardless of the exact input — a size chosen to be
    plausible for two un-folded SUMMARY_MAX_TOKENS-capped batches, not
    derived from this fixture's own token count (P13-3, hostile pass #13:
    the "12,951 -> 13,955-14,953" figure once cited here as a real-branch
    measurement was a replay-harness artifact, not a measurement — see
    [17]'s own correction above)."""
    CALLS.append(list(to_summarize))
    chunk = "MULTIBATCH-SUMMARY-PART " + _g_filler(900)
    return chunk + "\n\n" + chunk, []


async def _g17_singlebatch_summarize(client, to_summarize):
    """A single-batch fresh summary, capped exactly at what one real
    _summarize_once call can return — the case the flat SUMMARY_MAX_TOKENS
    allowance was always correct for."""
    CALLS.append(list(to_summarize))
    return "SINGLEBATCH-SUMMARY " + _g_filler(main.SUMMARY_MAX_TOKENS - 8), []


_G17_HIER_CONV = "reuse_fit_p12_6_hierarchy"


def _g17_hierarchy(n_l1: int = 9, l3_tokens: int | None = 3291):
    """Self-contained copy of `_g14_hierarchy`'s body (not a call to it) —
    [17a]/[17b] deliberately seed DIFFERENT hierarchy sizes (peakB-ish,
    then "today") on the same conv id in sequence; `_g14_hierarchy`
    reuses `_G14_CONV` and only SETS `st["l3"]` when `l3_tokens` is
    truthy, never clearing a PRIOR seed's l3 when this one has none — the
    exact shape [14] itself never hits (its own today/peakA/peakB calls
    only ever ADD an l3, never remove one, in that order). Reproduced here
    once, from first principles, so [17] cannot leak state into or out of
    [14]/[15]'s own conv id either."""
    older = history(n_l1 * 5 + 20, words=200)
    l1 = [
        {"tier": "l1", "text": "L1scene " + _g_filler(_G12_L1_MEAN),
         "first_turn": i * 10 + 1, "last_turn": i * 10 + 10}
        for i in range(n_l1)
    ]
    l2 = [
        {"tier": "l2", "text": "L2chapter " + _g_filler(_G12_L2_MEAN),
         "first_turn": n_l1 * 10 + i * 10 + 1, "last_turn": n_l1 * 10 + i * 10 + 10}
        for i in range(4)
    ]
    last = l2[-1]["last_turn"]
    st = _seed_hierarchy(_G17_HIER_CONV, older, l1 + l2)
    st.pop("l3", None)  # never leak a previous call's L3 on this conv id
    if l3_tokens:
        st["l3"] = {
            "text": "L3theme " + _g_filler(l3_tokens),
            "first_turn": 1, "last_turn": last,
        }
    summarizer.save_state(_G17_HIER_CONV, st)
    st = summarizer.load_state(_G17_HIER_CONV)
    saved = summarizer.SUMMARY_BLOCK_MAX_TOKENS
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = 10**9
    render = summarizer._estimate_block_tokens(
        summarizer.format_summary_block(st, 10**9) or ""
    )
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved
    return older, last, render


def _g17_build(
    n_fresh_pairs: int, fresh_pair_tokens: int, aprev_tokens: int = 8000,
    n_l1: int = 9, l3_tokens: int | None = 3291,
):
    """Her peakB-ish hierarchy by default (n_l1=9, l3_tokens=3291 -> her
    measured ~12,951), THEN an uncovered tail of `n_fresh_pairs` fat
    exchanges past the last L1/L2 chunk boundary, THEN her measured
    755-char system prompt and the same realistic recent exchange [14]/
    [16] use. [17b] passes a smaller ("today") hierarchy — even ONE
    correctly-reserved fresh batch (1,024 tokens) does not fit beside a
    12,951-token hierarchy and an 8,000-token reply at all (12,951 + 1,024
    + 128 = 14,103 > the ~12,372-token ceiling this recent window leaves —
    the SAME reason plain peakB already declines with zero fresh content),
    so proving the guard still protects a fresh-summary-bearing stand-in
    needs a hierarchy small enough for reuse to actually fire with one."""
    older, last, render = _g17_hierarchy(n_l1, l3_tokens)
    older = list(older)
    older[0] = {"role": "system", "content": "S" * 755}
    for i in range(n_fresh_pairs):
        older.append(fat_turn("user", fresh_pair_tokens, f"fresh-u{i}"))
        older.append(fat_turn("assistant", fresh_pair_tokens, f"fresh-a{i}"))
    recent = [
        fat_turn("user", 100, "prev-u"),
        fat_turn("assistant", aprev_tokens, "prev-a"),
        fat_turn("user", 100, "newest-u"),
    ]
    return older + recent, last, render


_G17_CONV = _G17_HIER_CONV  # the SAME conv id `_g17_hierarchy` seeds — a
# different one would make compact_if_needed look up an empty state


def _g17_run(
    summarize_stub, n_fresh_pairs, fresh_pair_tokens,
    n_l1=9, l3_tokens=3291, aprev_tokens=6000,
):
    # [14]'s own fraction note applies here unchanged: `_G_PLANNED_INJECT`
    # is a LITERAL 0.75-based figure, but this file's module-level env sets
    # `COMPACTOR_INJECTION_BUDGET_FRACTION="0.6"` (the sections before [14]
    # use it, and [14] itself restores it when done) — recovering
    # `effective_limit` as `inject_budget / INJECTION_BUDGET_FRACTION`
    # without matching the two is exactly the mismatch [14]'s own comment
    # warns about: an inflated `_effective_limit_est` (25,960 instead of
    # 20,768) that never declines at all, silently under-testing this
    # section's whole point. Same save/restore [14] uses.
    saved_fraction = main.INJECTION_BUDGET_FRACTION
    main.INJECTION_BUDGET_FRACTION = 0.75
    main.summarize = summarize_stub
    try:
        msgs, last, render = _g17_build(
            n_fresh_pairs, fresh_pair_tokens, aprev_tokens=aprev_tokens,
            n_l1=n_l1, l3_tokens=l3_tokens,
        )
        saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
        summarizer.SUMMARY_BLOCK_MAX_TOKENS = _G12_SHIPPED_SBMAX
        stored_out: list = []
        with capture() as records:
            try:
                out = _run(
                    msgs, _G17_CONV, stored_turns_out=stored_out,
                    inject_budget=_G_PLANNED_INJECT,
                )
            finally:
                summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved_sbmax
    finally:
        main.summarize = _spy_summarize
        main.INJECTION_BUDGET_FRACTION = saved_fraction
    return stored_out, out, last, render, records.records


# [17a] the reproduction: a fresh span big enough to need 2 un-folded
# batches (~1,800 measured tokens), at her peakB-ish hierarchy size.
_g17a_stored, _g17a_out, _g17a_last, _g17a_render, _g17a_log = _g17_run(
    _g17_multibatch_summarize, n_fresh_pairs=2, fresh_pair_tokens=4000
)
check(
    12700 <= _g17a_render <= 13200,
    f"fixture: [17a] hierarchy renders near the coordinator's measured "
    f"12,951 (got {_g17a_render})",
)
_g17a_rep, _g17a_prev, _g17a_newest = _g14_guard(_g17a_out, _g17a_last, "17a")
check(
    _g17a_rep.get("fits") is True,
    f"[17a] the guard fits the payload ({_g17a_rep})",
)
check(
    _g17a_prev,
    f"*** P12-6 [17a]: her previous exchange (U_prev/A_prev) survives "
    f"END TO END — whatever compact_if_needed decided (stored_turns_out="
    f"{_g17a_stored}), the real guard afterward does not lose it to a "
    f"fresh summary this reserve failed to price for ({_g17a_rep})",
)
check(
    _g17a_newest,
    "[17a]: the newest turn always survives",
)

# [17b] item 2: a SMALLER hierarchy ("today", ~9,075 — plain peakB already
# declines regardless of fresh content, see `_g17_build`'s own docstring)
# plus a SMALLER fresh span (one batch, correctly reserved either way)
# lets reuse fire with a fresh-summary-bearing stand-in — the guard's
# P12-5 ordering must still spend memory before the previous exchange.
_g17b_stored, _g17b_out, _g17b_last, _g17b_render, _ = _g17_run(
    _g17_singlebatch_summarize, n_fresh_pairs=1, fresh_pair_tokens=2000,
    n_l1=8, l3_tokens=None,
)
check(
    _g17b_stored == [_g17b_last],
    f"*** P12-6 [17b] fixture: reuse fires with a SMALLER fresh span "
    f"(stored_turns_out={_g17b_stored}) — the state this CONTROL needs: a "
    f"fresh-summary-bearing stand-in actually reaching the guard",
)
_g17b_standin = next(
    (m for m in _g17b_out if main._is_compaction_standin(m)), None
)
check(
    _g17b_standin is not None and "SINGLEBATCH-SUMMARY" in _g17b_standin["content"],
    "[17b] fixture: the stand-in the guard will see actually carries the "
    "fresh summary text, combined with the stored hierarchy in ONE system "
    "message (not two) — the shape `_is_compaction_standin` must still "
    "recognise",
)
_g17b_rep, _g17b_prev, _g17b_newest = _g14_guard(_g17b_out, _g17b_last, "17b")
check(
    _g17b_rep.get("fits") is True,
    f"[17b] the guard fits the payload ({_g17b_rep})",
)
check(
    _g17b_prev,
    f"*** P12-6 [17b]: her previous exchange still survives the guard "
    f"when the stand-in it protects is a COMBINED stored+fresh message, "
    f"not just the bare hierarchy — P12-5's ordering does not get confused "
    f"by what the stand-in is made of ({_g17b_rep})",
)
check(
    _g17b_newest,
    "[17b]: the newest turn always survives",
)


# ---------------------------------------------------------------------------
# [18] P12-6 coordinator follow-up #2 (hostile pass #12, real-data "unpair"
# replay — the SECOND, corrected replay after the coordinator found their
# first "fresh" harness was a spy artifact that could not test this reserve
# at all: it appended ~1,000 tokens to the stand-in AFTER compact_if_needed
# returned, where no decision inside it could ever see them). The "unpair"
# variant is real: every third assistant turn among the last 60 COVERED
# turns gets its content changed, so `_coverage_plan` genuinely stops
# pairing them and `compact_if_needed` itself summarizes them fresh
# (verified on her branch: 1 summarize() call at most positions, 3-4 at
# some). At the LIVE END of her branch — her last real message is a
# 39,569-character (~10.5k-token) reply, and the replies just before it are
# also long — reuse fired with an 11.8k-token stand-in beside a 3-4 batch
# fresh span and lost her previous exchange, where declining kept it.
#
# ROOT CAUSE, found by direct reproduction (not guessed): the fresh-span
# reserve's ESCAPE HATCH — "more than MAX_SUMMARY_CALLS_PER_REQUEST
# PESSIMISTIC-estimated batches means summarize() will cap-refuse, so
# reserve nothing" — is backwards. The pessimistic (2.0x) scale this
# preview uses to size the reserve SAFELY is not what `summarize()` itself
# uses to decide whether to proceed at all; on content whose REAL
# (undoubled) batch count is under the cap, `summarize()` proceeds and can
# still leave every batch un-folded — exactly the costly case this reserve
# exists to price for — while the PESSIMISTIC preview's own inflated count
# can land just OVER the cap on that SAME content, collapsing the reserve
# to zero at precisely the wrong moment. [18a] reproduces this directly:
# a fresh span sized so the pessimistic preview predicts battle count OVER
# the cap while a REALISTIC (capped-at-4-batches) fresh summary is what
# actually lands beside the stand-in — the previous exchange must not be
# what pays for a reserve that collapsed to zero. [18b] is the CONTROL:
# the same shape, but with a small (1-batch) fresh span, where the OLD
# formula was already correct and must stay correct.
# ---------------------------------------------------------------------------
print("\n[18] P12-6 coordinator follow-up #2: the fresh-span reserve's "
      "cap-refusal escape hatch must not collapse to zero on its own "
      "pessimism")


async def _g18_multibatch_summarize(client, to_summarize):
    """Her real magnitude: several un-folded near-cap batches concatenated
    (matching the coordinator's real SUMC=3-4, capped at MAX_SUMMARY_CALLS_
    PER_REQUEST=4 either way — summarize() itself never returns more than
    that many un-folded parts, by construction), not a single small
    summary and not an unbounded one."""
    CALLS.append(list(to_summarize))
    parts = ["UNPAIR-SUMMARY-PART " + _g_filler(900) for _ in range(4)]
    return "\n\n".join(parts), []


async def _g18_singlebatch_summarize(client, to_summarize):
    CALLS.append(list(to_summarize))
    return "UNPAIR-SUMMARY-SMALL " + _g_filler(500), []


def _g18_build(unpaired_tokens: int, n_unpaired: int, aprev_tokens: int):
    """Her peakA-ish hierarchy ([14]'s own construction, ~11.5k — the
    coordinator's real branch measured ~11.8k for this state; close
    enough that the arithmetic below is representative, not recalibrated
    to a different decimal), THEN `n_unpaired` of her last 60 COVERED
    turns (every 3rd assistant turn, matching the coordinator's own real
    methodology) get their CONTENT changed post-seed — the fingerprint
    record still holds the ORIGINAL text, so `_coverage_plan` genuinely
    stops pairing them, exactly as it would for a real edited/regenerated
    reply. THEN her measured 755-char system prompt and a tunable recent
    exchange."""
    older, last, render = _g17_hierarchy(n_l1=9, l3_tokens=1869)
    older = list(older)
    older[0] = {"role": "system", "content": "S" * 755}
    modified = 0
    for turn_no in range(last - 59, last + 1):
        if modified >= n_unpaired:
            break
        if turn_no % 6 == 0:  # assistant turns are even; every 3rd = every 6th turn number
            older[turn_no] = fat_turn(
                "assistant", unpaired_tokens, f"unpaired-{turn_no}"
            )
            modified += 1
    recent = [
        fat_turn("user", 100, "prev-u"),
        fat_turn("assistant", aprev_tokens, "prev-a"),
        fat_turn("user", 100, "newest-u"),
    ]
    return older + recent, last, render, modified


def _g18_run(summarize_stub, unpaired_tokens, n_unpaired, aprev_tokens):
    # Same fraction note as [17]/_g17_run: _G_PLANNED_INJECT is a literal
    # 0.75-based figure; this file's module env sets 0.6.
    saved_fraction = main.INJECTION_BUDGET_FRACTION
    main.INJECTION_BUDGET_FRACTION = 0.75
    main.summarize = summarize_stub
    try:
        msgs, last, render, modified = _g18_build(
            unpaired_tokens, n_unpaired, aprev_tokens
        )
        saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
        summarizer.SUMMARY_BLOCK_MAX_TOKENS = _G12_SHIPPED_SBMAX
        stored_out: list = []
        with capture() as records:
            try:
                out = _run(
                    msgs, _G17_HIER_CONV, stored_turns_out=stored_out,
                    inject_budget=_G_PLANNED_INJECT,
                )
            finally:
                summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved_sbmax
    finally:
        main.summarize = _spy_summarize
        main.INJECTION_BUDGET_FRACTION = saved_fraction
    return stored_out, out, last, render, modified, records.records


# [18a] the reproduction: 10 unpaired assistant turns at 5,900 tokens each
# (59,000 raw — pushes the PESSIMISTIC-scale batch estimate to 5, one OVER
# MAX_SUMMARY_CALLS_PER_REQUEST=4) beside a 7,000-token A_prev (tuned so
# the base hierarchy alone fits the structural ceiling — the OLD formula's
# collapse to a ZERO fresh reserve is what must be caught here; a bigger
# A_prev would decline correctly even at zero reserve, the same way plain
# peakB already does regardless of the bug, and would not discriminate the
# needle at all — found this exact trap while building [17a] the first
# time).
_g18a_stored, _g18a_out, _g18a_last, _g18a_render, _g18a_modified, _g18a_log = (
    _g18_run(_g18_multibatch_summarize, unpaired_tokens=5900, n_unpaired=10,
             aprev_tokens=7000)
)
check(
    11300 <= _g18a_render <= 11700,
    f"fixture: [18a] hierarchy renders near peakA (got {_g18a_render})",
)
check(
    _g18a_modified == 10,
    f"fixture: [18a] modified exactly 10 covered assistant turns (got "
    f"{_g18a_modified})",
)
_g18a_rep, _g18a_prev, _g18a_newest = _g14_guard(_g18a_out, _g18a_last, "18a")
check(
    _g18a_rep.get("fits") is True,
    f"[18a] the guard fits the payload ({_g18a_rep})",
)
check(
    _g18a_prev,
    f"*** P12-6 [18a]: her previous exchange (U_prev/A_prev) survives END "
    f"TO END — whatever compact_if_needed decided (stored_turns_out="
    f"{_g18a_stored}), the real guard afterward does not lose it to a "
    f"fresh-span reserve that collapsed to zero because its own pessimism "
    f"crossed the call cap ({_g18a_rep})",
)
check(
    _g18a_newest,
    "[18a]: the newest turn always survives",
)

# [18b] CONTROL: a SMALL (1-batch, well under the cap either way) unpaired
# span at the SAME hierarchy and recent-window sizes — the OLD formula was
# already correct here (n_batches <= cap), and the fix must not change
# that: reuse still fires, the fresh-summary-bearing stand-in still
# reaches the guard, and the exchange still survives.
_g18b_stored, _g18b_out, _g18b_last, _g18b_render, _g18b_modified, _ = (
    _g18_run(_g18_singlebatch_summarize, unpaired_tokens=1200, n_unpaired=2,
             aprev_tokens=7000)
)
check(
    _g18b_stored == [_g18b_last],
    f"*** P12-6 [18b] CONTROL: reuse fires with a SMALL (1-batch) unpaired "
    f"span (stored_turns_out={_g18b_stored}) — the fix must not make this "
    f"section decline MORE than the old formula did on content already "
    f"under the cap",
)
_g18b_rep, _g18b_prev, _g18b_newest = _g14_guard(_g18b_out, _g18b_last, "18b")
check(
    _g18b_rep.get("fits") is True,
    f"[18b] CONTROL the guard fits the payload ({_g18b_rep})",
)
check(
    _g18b_prev,
    f"[18b] CONTROL: her previous exchange survives — unaffected by this "
    f"fix, the same as before it ({_g18b_rep})",
)
check(
    _g18b_newest,
    "[18b] CONTROL: the newest turn always survives",
)


# ---------------------------------------------------------------------------
# [19] P13-1 (hostile pass #13, HIGH): the window check ignores the guard's
# learned `_BUDGET_MARGIN`. After a vLLM context-length 400 the guard did NOT
# predict (a `/tokenize` outage or a mispriced image — `_note_backend_
# rejection`'s own docstring), `_BUDGET_MARGIN` latches to `overshoot + 512`,
# up to the MAX_MODEL_LEN//4 ceiling, in ONE step, process-wide, released
# only after BUDGET_MARGIN_RELEASE_AFTER (default 50) consecutive accepted
# requests. `_enforce_hard_budget` (the guard) subtracts it from its own
# limit before shedding a single token (main.py: `if _BUDGET_MARGIN: limit =
# max(256, limit - _BUDGET_MARGIN)`). The window check recovered the SAME
# `effective_limit` but never subtracted the SAME margin — so while one was
# in force, this check could approve a stand-in the guard, needing that many
# more tokens of room than the check thought existed, could not actually fit
# beside the previous exchange. Reuse "succeeded" and the guard then shed
# U_prev/A_prev to find the room — exactly the loss P12-5 (above) fixed on
# the DECLINED path, given back on the REUSING one.
#
# The reviewer found no existing test can tell this apart: `[F1]`/`[F1-lean]`
# pin `main._BUDGET_MARGIN = 0` (this file's own [F1] sections above), and
# every other section here runs at the module default of 0 — a margin was
# simply never exercised through the window check before.
#
# Fixture: [14]'s own 'peakA' hierarchy (n_l1=9, l3_tokens=1869, the SAME
# CONTROL state [14] uses to prove this is not "decline once the hierarchy
# is merely large"), with `aprev_tokens=8500` — 500 tokens larger than [14]'s
# own 8,000, tuned so this state sits close enough to the window ceiling
# that margin 513 (the SMALLEST value `overshoot + 512` can latch to) is
# already enough to tip it, the same as the reviewer's real-data measurement
# (P13-1: 3 of 474 peakA positions already lose the exchange at margin 513).
# Calibrated empirically against HEAD (SP\fix-3193-r3.md's [19] proof; not
# derived from this fixture's own token count, so the fix cannot accidentally
# match what the fixture assumes): at margin 0 the previous exchange
# survives; at margin 513 AND margin 8192 (the F-02 ceiling) it does not,
# before this fix.
# ---------------------------------------------------------------------------
print("\n[19] P13-1: the window check reads _BUDGET_MARGIN the same way the "
      "guard does, at the same moment in the same request")


def _g19_run(margin, aprev_tokens=8500):
    """[14]'s own peakA construction, wrapped with the SAME local
    save/restore [17]/[18] use for `INJECTION_BUDGET_FRACTION` and
    `summarize` (by the time this section runs, [14]'s own module-level
    override has long since been restored — see [14]'s `_G14_SAVED_
    FRACTION`), plus `_BUDGET_MARGIN` for the one thing this section is
    actually about."""
    saved_fraction = main.INJECTION_BUDGET_FRACTION
    saved_summarize = main.summarize
    saved_margin = main._BUDGET_MARGIN
    main.INJECTION_BUDGET_FRACTION = 0.75
    main.summarize = _spy_summarize
    main._BUDGET_MARGIN = margin
    try:
        return _g14_run(9, 1869, aprev_tokens)
    finally:
        main.INJECTION_BUDGET_FRACTION = saved_fraction
        main.summarize = saved_summarize
        main._BUDGET_MARGIN = saved_margin


# [19a] CONTROL: healthy state (margin 0) — reuse fires and her previous
# exchange survives the real guard, same shape as [14]'s own 'peakA' check
# above, just at the slightly larger aprev_tokens this section needs.
_g19a_stored, _g19a_out, _g19a_last, _g19a_render, _ = _g19_run(0)
check(
    _g19a_stored == [_g19a_last],
    f"[19a] CONTROL (margin 0): reuse fires (stored_turns_out={_g19a_stored})",
)
main._BUDGET_MARGIN = 0
_g19a_rep, _g19a_prev, _g19a_newest = _g14_guard(_g19a_out, _g19a_last, "19a")
check(_g19a_rep.get("fits") is True, f"[19a] the guard fits the payload ({_g19a_rep})")
check(
    _g19a_prev,
    f"[19a] CONTROL: her previous exchange survives at margin 0 ({_g19a_rep})",
)

for _g19_margin in (513, 8192):
    _g19_stored, _g19_out, _g19_last, _g19_render, _ = _g19_run(_g19_margin)
    main._BUDGET_MARGIN = _g19_margin
    _g19_rep, _g19_prev, _g19_newest = _g14_guard(_g19_out, _g19_last, f"19-{_g19_margin}")
    main._BUDGET_MARGIN = 0
    check(
        _g19_prev,
        f"*** P13-1 [19] margin={_g19_margin}: her previous exchange "
        f"survives end to end — either reuse correctly declined (window "
        f"squeeze, so the declined path's P12-5 protection runs) or reuse "
        f"fired and the guard still fit the stand-in whole beside it "
        f"(stored_turns_out={_g19_stored}, guard={_g19_rep}) — before this "
        f"fix, reuse fired here (the window check never read the margin) "
        f"and the guard then shed the exchange to find the room the margin "
        f"cost it",
    )
    check(
        _g19_newest,
        f"[19] margin={_g19_margin}: the newest turn always survives",
    )

check(
    main.reuse_decline_state().get("last_reason") in ("window", "success"),
    f"[19]: reuse_decline_state() recorded a real outcome for the last "
    f"margin=8192 attempt above, not a stale value from an earlier section "
    f"(got {main.reuse_decline_state().get('last_reason')!r})",
)

# [19b] P14-4 (hostile pass #14): [19] only asserts her previous exchange
# survives, and a check that declines MORE always satisfies that — the
# margin subtracted twice passed every suite. This pins the other side:
# under a margin the stand-in still has room for, reuse still fires.
# Self-calibrating, so it does not depend on this fixture's exact sizes: a
# window decline forced at 8,192 records the recent floor and the reserve;
# with the window this fixture passes (its inject budget over the 0.75
# fraction `_g19_run` sets) they give the slack S at margin 0 without
# trusting the check's own margin arithmetic. A margin of ceil(2S/3) leaves
# room once and not twice.
_g19b_aprev = 7000
_g19_run(8192, aprev_tokens=_g19b_aprev)
_g19b_st = main.reuse_decline_state()
check(
    _g19b_st.get("last_reason") == "window",
    f"fixture: [19b] margin 8192 forces a window decline "
    f"({_g19b_st.get('last_reason')!r})",
)
_g19b_slack = (
    int(round(_G_PLANNED_INJECT / 0.75))
    - (_g19b_st.get("last_declined_others") or 0)
    - (_g19b_st.get("last_declined_reserve") or 0)
)
check(_g19b_slack > 600, f"fixture: [19b] {_g19b_slack} tokens of slack at margin 0")
_g19b_margin = -(-2 * _g19b_slack // 3)
_g19b_stored, _g19b_out, _g19b_last, _, _ = _g19_run(
    _g19b_margin, aprev_tokens=_g19b_aprev
)
check(
    _g19b_stored == [_g19b_last],
    f"*** P14-4 [19b] margin={_g19b_margin} (slack {_g19b_slack}): reuse "
    f"still fires when the stand-in fits beside the margin "
    f"(stored_turns_out={_g19b_stored}) — a margin subtracted more than "
    f"once declines here",
)
main._BUDGET_MARGIN = _g19b_margin
_g19b_rep, _g19b_prev, _g19b_newest = _g14_guard(_g19b_out, _g19b_last, "19b")
main._BUDGET_MARGIN = 0
check(
    _g19b_rep.get("fits") is True and _g19b_prev and _g19b_newest,
    f"[19b] and the guard, at the same margin, fits it with her previous "
    f"exchange and the newest turn ({_g19b_rep})",
)


# ---------------------------------------------------------------------------
# [20] P13-2 (hostile pass #13, MEDIUM): the fresh-span preview always priced
# `_fresh_span_preview` at `_PESSIMISTIC_SUMMARY_SCALE` (2.0x), even when
# `/tokenize` answers and `summarize()` itself packs the SAME list at the
# MEASURED scale (exact/local — summarize()'s own `_scale = _exact /
# _local`). Pricing the preview 2x what the real call will use inflates the
# predicted batch count and, with it, the reserve this check compares
# against the window — declining reuse on her routine between-L1-rollup
# uncovered tail even when the real (or worst-case un-folded) summarize()
# call would have fit. SP\p13-findings.md P13-2: 11-82 extra window declines
# per 474 positions, most of them losing the uncovered tail from the model's
# view (a decline hands summarize() the whole older span, that is refused
# over the call cap, and P12-5's order sheds the verbatim turns ahead of
# memory).
#
# Fixture: [17]'s own hierarchy/fresh-tail builder (`_g17_run`), her peakA
# shape (l3_tokens=1869) with a 2-pair, 3,750-token-per-message fresh tail
# and `aprev_tokens=7500` for [20a]/[20b] below (empirically, against
# HEAD; not derived from this fixture's own count) — the SAME span needs
# TWO map-reduce batches at the pessimistic 2.0x scale (reserve 13,705 >
# the 12,872-token ceiling — declines) but only ONE at her measured ~1.0x
# range (reserve 12,681 <= 12,872 — fits), because next-fit packs each
# message independently and four ~3,750-token messages clear one
# ~29,696-token batch at 1.05x but not at 2.0x.
#
# v3.1.9.4 (v3194-r3, R7): [20c]/[20e] below use a DIFFERENT, SELF-
# CALIBRATING `aprev_tokens` instead of a shared hand-picked one. G3c
# (hostile pass #14) gave `_sys_recent_floor` its own EXACT measurement
# over `system_msgs + keep_recent` — `aprev_tokens` is part of that list
# too — so [20a]/[20b]'s shared 7500 is no longer neutral background for
# every sub-section once a stub also prices THAT call: at 7500 the
# doubled floor alone can decline regardless of the fresh span's own
# scale, and at the 1000 [20c] used to compensate there is so much slack
# that neither scale nor which list gets measured changes the outcome —
# both extremes make this section stop actually testing what it claims
# to. [20c]/[20e] each binary-search their own `aprev_tokens` at runtime
# against the real `compact_if_needed`, the same kind of self-calibration
# [19b] above does by computing a margin from `reuse_decline_state()` —
# adapted here to a search because this fixture's decision is a step
# function of `aprev_tokens`, not a closed-form expression.
# ---------------------------------------------------------------------------
print("\n[20] P13-2: the fresh-span preview measures the same scale "
      "summarize() will use, instead of always pricing pessimistically")

_g20_saved_cte = main.count_tokens_exact


def _g20_stub_exact(ratio):
    """None simulates `/tokenize` not answering (summarize()'s own
    fallback path); a float simulates it answering with that exact/local
    ratio — same shape as this file's `_g15_exact_counter`/`_f1_tokens`
    stubs elsewhere, but parameterised on the ratio this section varies."""
    if ratio is None:
        return lambda ms, *a, **k: None
    return lambda ms, *a, **k: int(main.count_tokens(ms) * ratio)


def _g20_stub_exact_by_content(fresh_ratio, other_ratio):
    """P14-4 (hostile pass #14): a ratio that depends on WHAT is measured —
    `fresh_ratio` for a list made only of the fresh span's turns,
    `other_ratio` for anything else. With one ratio for every list, the
    preview's scale measured on `system + keep_recent` instead of the fresh
    span passed every suite; this way only the right list gives the passing
    answer."""
    def _exact(ms, *a, **k):
        fresh = bool(ms) and all(
            isinstance(m.get("content"), str)
            and m["content"].startswith("fresh-")
            for m in ms
        )
        return int(main.count_tokens(ms) * (fresh_ratio if fresh else other_ratio))
    return _exact


def _g20_run(exact_ratio, stub=None, aprev_tokens=7500, n_fresh_pairs=2, fresh_pair_tokens=3750):
    main.count_tokens_exact = stub or _g20_stub_exact(exact_ratio)
    try:
        return _g17_run(
            _spy_summarize, n_fresh_pairs, fresh_pair_tokens, n_l1=9, l3_tokens=1869,
            aprev_tokens=aprev_tokens,
        )
    finally:
        main.count_tokens_exact = _g20_saved_cte


# [20a] CONTROL: /tokenize down — falls back to the pessimistic 2.0x scale,
# unchanged from the shipped behaviour, and still declines. This is the one
# case the OLD comment's reasoning ("bias toward more predicted batches is
# the safe direction, the same reasoning summarize()'s own /tokenize-down
# fallback uses") was actually correct for.
_g20a_stored, _g20a_out, _g20a_last, _g20a_render, _g20a_log = _g20_run(None)
check(
    _g20a_stored == [0],
    f"[20a] CONTROL (/tokenize down): the preview still falls back to the "
    f"pessimistic scale and declines, same as before this fix "
    f"(stored_turns_out={_g20a_stored})",
)

# [20b] CONTROL: /tokenize answers, and the real measured scale genuinely IS
# 2.0x (a degraded local counter) — must still decline. The fix reads the
# real measurement; it does not turn the decline off unconditionally.
_g20b_stored, _g20b_out, _g20b_last, _g20b_render, _g20b_log = _g20_run(2.0)
check(
    _g20b_stored == [0],
    f"[20b] CONTROL: measured scale 2.0x still declines — the fix is not "
    f"'always approve', it is 'approve at the real scale' "
    f"(stored_turns_out={_g20b_stored})",
)

# [20c] *** THE FIX: /tokenize answers near her measured real range (~1.0x,
# SP\p13-findings.md's "summarize() POSTs ... token scale ~1.0x" note) —
# the SAME fresh span that declines at the pessimistic scale above now
# needs only one batch, fits, and reuses. The stub answers 1.05 only for the
# fresh span itself and 2.0 for any other list (P14-4), so the scale must be
# measured on the list summarize() will receive.
#
# v3.1.9.4 (v3194-r3, R7 — coordinator follow-up). G3c gave
# `_sys_recent_floor` its own EXACT call over `system_msgs + keep_recent`
# — a SECOND, LEGITIMATE consumer of this same stub's "2.0 for any other
# list" bucket, distinct from the fresh-span preview this section exists
# to test. `aprev_tokens` (part of `keep_recent`) was cut to 1000 so that
# doubled floor left room again — but 1000 turned out to leave SO MUCH
# room that reuse fires at 1.05x, 2.0x, AND a preview that measures the
# wrong list entirely: [20c] kept passing while no longer able to tell
# the fix from the defect it exists to catch (confirmed: `underscale`
# and `wronglist` on `main.py` both survived this check unchanged).
#
# SELF-CALIBRATED, not hand-tuned (as [19b] above does, adapted to a
# binary search since this fixture's decision is a step function of
# `aprev_tokens`, not a closed-form arithmetic expression [19b] can
# compute directly from `reuse_decline_state()`): probe the REAL
# `compact_if_needed` reuse check directly, through the SAME stub and
# fixture shape [20a]/[20b] already use, to find the largest
# `aprev_tokens` where the CORRECT scale (1.05x, measured on the fresh
# span alone) still reuses, and the largest where the WRONG scale (2.0x
# — what `wronglist` and `freshscale` both collapse to for this fixture:
# a preview that measures the wrong list lands in the stub's "any other
# list" bucket, identically to one that packs at `_PESSIMISTIC_SUMMARY_
# SCALE` unconditionally) already declines. Both searches read only
# `stored_turns_out` — the same observable [20a]-[20c]'s own checks
# already use — so this adapts automatically if MAX_MODEL_LEN,
# SUMMARY_MAX_TOKENS or any other constant this fixture depends on ever
# moves, the way a hardcoded constant would not.
def _g20c_reuses(fresh_ratio, aprev_tokens):
    _stored, *_ = _g20_run(
        None, stub=_g20_stub_exact_by_content(fresh_ratio, 2.0),
        aprev_tokens=aprev_tokens,
    )
    return _stored != [0]


def _g20_search_largest_reusing(fresh_ratio, lo=200, hi=12000):
    """Binary search: the largest aprev_tokens where reuse still fires at
    `fresh_ratio`. Assumes (and the two checks below confirm) that
    reuse-fires is monotonically non-increasing in aprev_tokens — true
    here because a bigger A_prev only ever ADDS to `_sys_recent_floor`,
    never removes from it."""
    assert _g20c_reuses(fresh_ratio, lo), (
        f"fixture: aprev_tokens={lo} must still reuse at ratio {fresh_ratio} "
        f"for this search to have a valid starting point"
    )
    assert not _g20c_reuses(fresh_ratio, hi), (
        f"fixture: aprev_tokens={hi} must already decline at ratio "
        f"{fresh_ratio} for this search to have a valid ending point"
    )
    while hi - lo > 20:
        mid = (lo + hi) // 2
        if _g20c_reuses(fresh_ratio, mid):
            lo = mid
        else:
            hi = mid
    return lo


_g20c_correct_edge = _g20_search_largest_reusing(1.05)
_g20c_wrong_edge = _g20_search_largest_reusing(2.0)
check(
    _g20c_wrong_edge < _g20c_correct_edge,
    f"fixture: a real calibration window exists — the wrong scale (2.0x) "
    f"stops reusing at a SMALLER aprev_tokens ({_g20c_wrong_edge}) than "
    f"the correct scale (1.05x) does ({_g20c_correct_edge})",
)
# Comfortably inside the window, not pinned to either edge (a search that
# happens to land within a couple of steps of its own boundary would make
# this section as fragile as the aprev_tokens=1000 it replaces).
_g20c_aprev = (_g20c_correct_edge + _g20c_wrong_edge) // 2
check(
    _g20c_reuses(1.05, _g20c_aprev) and not _g20c_reuses(2.0, _g20c_aprev),
    f"fixture: aprev_tokens={_g20c_aprev} (midpoint of "
    f"[{_g20c_wrong_edge}, {_g20c_correct_edge}]) reuses at 1.05x and "
    f"declines at 2.0x",
)

_g20c_stored, _g20c_out, _g20c_last, _g20c_render, _g20c_log = _g20_run(
    None, stub=_g20_stub_exact_by_content(1.05, 2.0), aprev_tokens=_g20c_aprev
)
check(
    _g20c_stored == [_g20c_last],
    f"*** P13-2 [20c]: the SAME fresh span that declines at the pessimistic "
    f"scale reuses once the preview measures the real scale "
    f"(stored_turns_out={_g20c_stored}) — before this fix the preview never "
    f"asked and always priced this span at 2.0x, one batch more than the "
    f"measured scale needs",
)
_g20c_rep, _g20c_prev, _g20c_newest = _g14_guard(_g20c_out, _g20c_last, "20c")
check(_g20c_rep.get("fits") is True, f"[20c] the guard fits the payload ({_g20c_rep})")
check(_g20c_prev, f"[20c]: her previous exchange survives end to end ({_g20c_rep})")
check(_g20c_newest, "[20c]: the newest turn always survives")

# [20c-freshscale] *** THE FIX, NAMED: at this SAME calibration, a preview
# that packs at `_PESSIMISTIC_SUMMARY_SCALE` UNCONDITIONALLY (freshscale —
# main.py's `_fresh_scale = _PESSIMISTIC_SUMMARY_SCALE` regardless of what
# `/tokenize` answered) is exactly the `fresh_ratio=2.0` probe the search
# above already used to find `_g20c_wrong_edge` — restated here as its own
# named, standalone check (not folded into the search) so a reader — or a
# mutation run — sees this specific regression called out on its own line.
_g20_freshscale_stored, *_ = _g20_run(
    None, stub=_g20_stub_exact_by_content(2.0, 2.0), aprev_tokens=_g20c_aprev
)
check(
    _g20_freshscale_stored == [0],
    f"*** [20c-freshscale] a preview pinned to the pessimistic scale "
    f"unconditionally DECLINES here, where the real (1.05x) scale reuses "
    f"(stored_turns_out={_g20_freshscale_stored}) — this is the exact "
    f"shape main.py's `_fresh_scale = _PESSIMISTIC_SUMMARY_SCALE` "
    f"(ignoring what /tokenize measured) would produce",
)

# [20c-wronglist] main.py's `_fresh_local`/`_fresh_exact` measuring
# `system_msgs + keep_recent` instead of `_fresh_span_preview` — P14-4's
# own "wronglist" shape — makes BOTH calls land in this stub's "any other
# list" bucket (2.0x), identically to freshscale immediately above: from
# the stub's point of view (content in, ratio out) a preview that asks the
# wrong question is indistinguishable from one that never asks at all. So
# [20c]'s own "*** P13-2 [20c]" check IS the named check wronglist trips —
# driven through the real code (a genuine textual mutation of the two call
# sites, not just this stub-level argument; see this lane's mutation
# report for the needle and the confirmed KILLED result) rather than
# invented here as a separate assertion that would just restate the same
# stub response under a different name.

# [20d] invariant (the brief's own requirement): the preview must never
# count FEWER batches than a larger scale would — next-fit bin-packing is
# monotone in the per-message price, so raising the scale can only add
# batches, never remove one. This is what makes "fall back to the
# pessimistic scale when /tokenize is down" a SAFE direction rather than a
# guess: whatever the real (measured) scale turns out to be, 2.0x reserves
# at least as much. Checked directly on `_chunk_to_budget`, independent of
# compact_if_needed, at the two scales [20a]-[20c] exercise plus the
# pessimistic constant itself.
_g20d_span = [
    fat_turn("user", 3750, "u1"), fat_turn("assistant", 3750, "a1"),
    fat_turn("user", 3750, "u2"), fat_turn("assistant", 3750, "a2"),
]
_g20d_budget = min(
    main.MAX_MODEL_LEN,
    max(256, main.MAX_MODEL_LEN - main.SUMMARY_MAX_TOKENS - main.SUMMARY_INPUT_RESERVE),
)
_g20d_batches_low = len(main._chunk_to_budget(_g20d_span, _g20d_budget, 1.0))
_g20d_batches_measured = len(main._chunk_to_budget(_g20d_span, _g20d_budget, 1.05))
_g20d_batches_pess = len(
    main._chunk_to_budget(_g20d_span, _g20d_budget, main._PESSIMISTIC_SUMMARY_SCALE)
)
check(
    _g20d_batches_low <= _g20d_batches_measured <= _g20d_batches_pess,
    f"[20d] invariant: batch count never DECREASES as the packing scale "
    f"rises (1.0x={_g20d_batches_low}, 1.05x={_g20d_batches_measured}, "
    f"{main._PESSIMISTIC_SUMMARY_SCALE}x={_g20d_batches_pess}) — the "
    f"pessimistic fallback used whenever /tokenize does not answer can only "
    f"ever reserve AS MANY OR MORE batches than the measured scale this fix "
    f"now prefers when /tokenize does answer, never fewer",
)

# ---------------------------------------------------------------------------
# [20e] v3.1.9.4 (v3194-r3, R7 — coordinator follow-up): `underscale` —
# main.py's `_fresh_scale = (_fresh_exact / _fresh_local)` becoming
# `(_fresh_exact / _fresh_local) * 0.5` — UNDER-reserves rather than
# over-reserving, so [20c]'s own calibration (correct fits, wrong
# declines) cannot catch it: halving an already-small ratio only makes
# the preview MORE willing to reuse, never less, and [20c]'s window
# only exercises "does a bigger ratio correctly decline". Needs the
# OPPOSITE shape: a fresh span where the TRUE (1.05x) scale already
# needs TWO batches and correctly DECLINES (the request genuinely does
# not fit beside a real, un-halved summarize() call), while the halved
# (0.525x) scale wrongly measures only ONE batch's worth and reuses —
# approving a stand-in reserve smaller than what summarize() will
# actually need. A BIGGER fresh span than [20c]'s (8,000 not 3,750
# tokens/message) crosses the 1-vs-2-batch line at the REAL 1.05x scale
# instead of needing 2.0x to get there; same self-calibrating binary
# search as [20c], on this span's own numbers.
# ---------------------------------------------------------------------------
print("\n[20e] underscale: halving the measured scale must not approve a "
      "reserve smaller than what summarize() will actually need")


def _g20e_reuses(fresh_ratio, aprev_tokens):
    _stored, *_ = _g20_run(
        None, stub=_g20_stub_exact_by_content(fresh_ratio, 2.0),
        aprev_tokens=aprev_tokens, n_fresh_pairs=2, fresh_pair_tokens=8000,
    )
    return _stored != [0]


def _g20e_search_largest_reusing(fresh_ratio, lo=200, hi=12000):
    assert _g20e_reuses(fresh_ratio, lo), (
        f"fixture: aprev_tokens={lo} must still reuse at ratio {fresh_ratio}"
    )
    assert not _g20e_reuses(fresh_ratio, hi), (
        f"fixture: aprev_tokens={hi} must already decline at ratio {fresh_ratio}"
    )
    while hi - lo > 20:
        mid = (lo + hi) // 2
        if _g20e_reuses(fresh_ratio, mid):
            lo = mid
        else:
            hi = mid
    return lo


# The boundary at the TRUE scale (1.05x, 2 batches once the span is large
# enough — unlike [20c]'s smaller span, which only needed 2 batches at a
# punitive 2.0x).
_g20e_true_edge = _g20e_search_largest_reusing(1.05)
# The boundary at the HALVED scale (0.525x, still only 1 batch at sizes
# well past where 1.05x already needed 2).
_g20e_half_edge = _g20e_search_largest_reusing(1.05 * 0.5)
check(
    _g20e_true_edge < _g20e_half_edge,
    f"fixture: the halved scale keeps reusing at LARGER aprev_tokens than "
    f"the true scale does (true edge {_g20e_true_edge} < halved edge "
    f"{_g20e_half_edge}) — a real gap where the true scale has already "
    f"correctly declined but the halved one has not caught up",
)
# Comfortably inside the window: correctly declines at the true scale,
# reuses at the halved one.
_g20e_aprev = (_g20e_true_edge + _g20e_half_edge) // 2
check(
    (not _g20e_reuses(1.05, _g20e_aprev)) and _g20e_reuses(1.05 * 0.5, _g20e_aprev),
    f"fixture: aprev_tokens={_g20e_aprev} (midpoint of "
    f"[{_g20e_true_edge}, {_g20e_half_edge}]) declines at the true scale "
    f"and reuses at the halved one",
)

_g20e_stored, _g20e_out, _g20e_last, _g20e_render, _g20e_log = _g20_run(
    None, stub=_g20_stub_exact_by_content(1.05, 2.0), aprev_tokens=_g20e_aprev,
    n_fresh_pairs=2, fresh_pair_tokens=8000,
)
check(
    _g20e_stored == [0],
    f"*** [20e] CONTROL: at the TRUE measured scale (1.05x), this fresh "
    f"span correctly needs 2 batches and the request declines rather than "
    f"reusing (stored_turns_out={_g20e_stored})",
)

_g20e_under_stored, *_ = _g20_run(
    None, stub=_g20_stub_exact_by_content(1.05 * 0.5, 2.0), aprev_tokens=_g20e_aprev,
    n_fresh_pairs=2, fresh_pair_tokens=8000,
)
check(
    _g20e_under_stored != [0],
    f"*** [20e-underscale] the IDENTICAL fresh span, measured at HALF the "
    f"true scale, WRONGLY reuses — approving a stand-in reserve sized for "
    f"one map-reduce batch when summarize() (at the real, un-halved scale) "
    f"will actually need two (stored_turns_out={_g20e_under_stored})",
)


# ---------------------------------------------------------------------------
# [21] P14-1 (hostile pass #14, lane v3194-guard): the P13-1 fix's own
# residual. `compact_if_needed`'s reuse window check and `_enforce_hard_
# budget`'s own read of `_BUDGET_MARGIN` are NOT atomic — this function
# awaits `summarize()` and several thread hops in between (`[19]`'s own
# section above proves the two agree WITHIN one request when nothing moves
# the global in between). If another request's unpredicted rejection
# latches a LARGER margin while this one is suspended, the window check's
# decision ("the stand-in fits beside her previous exchange") was made for
# a margin smaller than the one the guard now enforces, and the guard's
# P12-5 branch protected the stand-in at the SAME tier as the recent
# window — so it cut into U_prev/A_prev before ever touching a stand-in
# that decision no longer justified. p14 proved it end to end
# (SP\p14\mk_conc.py): (a) margin 0 throughout kept the exchange, (b)
# margin 8192 throughout kept it (the window check declined, P13-1's own
# fix), (c) 0 at the check and 8192 by the time the guard ran LOST it.
#
# THE FIX: `compact_if_needed` gained `reuse_margin_out` — the margin its
# window check actually read, reported once, at that read. `chat_
# completions` compares that to the LIVE `_BUDGET_MARGIN` right beside its
# own call to the guard; when the margin grew since the decision, it
# passes `_enforce_hard_budget`'s new `standin_protected=False`, which
# excludes the stand-in from the P12-5 branch's protected tier for THIS
# call only — spent as ordinary memory, ahead of the previous exchange,
# the same order a request that had DECLINED reuse under that larger
# margin would already get. The guard's own numeric limit is UNCHANGED —
# it still reads the live global for `limit` itself, never a stale
# snapshot (the owner-facing constraint already in OPERATIONS.md: the
# guard must never forward at a limit smaller than the live margin
# demands). A margin that FALLS mid-request needs no correction: the
# decision was already conservative, so the (now more generous) live
# limit only has more room than it assumed.
#
# This file's own `_g14_guard` cannot exercise this: it calls
# `_enforce_hard_budget` with the 4-arg shape every other section here
# uses (`standin_protected` defaults True), so this section builds the
# fixture and drives both functions directly, mirroring exactly what
# `chat_completions` now does between them — the reuse decision, the
# margin comparison, and the guard call — rather than reusing `_g14_guard`
# unmodified.
# ---------------------------------------------------------------------------
print("\n[21] P14-1: a margin latched by another request mid-request no "
      "longer costs her previous exchange")

_G21_OVERFLOW = (
    "This model's maximum context length is 32768 tokens. However, your "
    "request has 29000 input tokens. Please reduce the length of the input "
    "messages."
)


async def _g21_bumping_summarize(client, to_summarize):
    """The race: another request's unpredicted rejection lands while THIS
    request is suspended awaiting its own fresh-span summarize()."""
    main._note_backend_rejection(
        _G21_OVERFLOW, enforced_limit=20768, guard_measured_overflow=False
    )
    return await _spy_summarize(client, to_summarize)


async def _g21_falling_summarize(client, to_summarize):
    """The margin-FALLS direction: another request's accepted-streak
    release lowered the margin while this one awaited summarize(). Moved
    directly (release only happens via consecutive accepted requests, off
    this test's critical path) — same pattern [F1]/[19] use elsewhere in
    this file to pin a margin for one section."""
    main._BUDGET_MARGIN = 0
    return await _spy_summarize(client, to_summarize)


def _g21_run(stub, margin_at_start):
    """p14's own peakA fixture ([17]/[19]'s own construction, n_l1=9,
    l3_tokens=1869 — sized so the SAME span reuses at the measured ~1.05x
    scale). aprev_tokens=6500, not [19]'s 8500 or the original p14 draft's
    7500: G3c (v3.1.9.4) gave `_sys_recent_floor` its own exact call, and
    this section's `_g20_stub_exact(1.05)` answers it too (a flat ratio
    for ANY list) — a small, deliberate inflation of the recent window's
    own measured cost that ate the margin 7500 left; recalibrated down,
    same as [20c]'s own `aprev_tokens` a few sections up, for the same
    reason. Driven through the fix's own
    wiring: `reuse_margin_out` from compact_if_needed, compared against
    the LIVE margin right beside the guard call, deciding `standin_
    protected` — exactly chat_completions's own sequence between the two
    functions, copied here in shape (not by calling chat_completions
    itself, which needs a full request/app context this suite does not
    build)."""
    saved = main._BUDGET_MARGIN
    main._BUDGET_MARGIN = margin_at_start
    main.count_tokens_exact = _g20_stub_exact(1.05)
    saved_fraction = main.INJECTION_BUDGET_FRACTION
    saved_summarize = main.summarize
    try:
        main.INJECTION_BUDGET_FRACTION = 0.75
        main.summarize = stub
        msgs, last, render = _g17_build(
            2, 3750, aprev_tokens=6500, n_l1=9, l3_tokens=1869,
        )
        saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
        summarizer.SUMMARY_BLOCK_MAX_TOKENS = _G12_SHIPPED_SBMAX
        stored_out: list = []
        margin_at_decision: list = []
        try:
            out = _run(
                msgs, _G17_CONV, stored_turns_out=stored_out,
                inject_budget=_G_PLANNED_INJECT,
                reuse_margin_out=margin_at_decision,
            )
        finally:
            summarizer.SUMMARY_BLOCK_MAX_TOKENS = saved_sbmax

        # --- chat_completions's own P14-1 wiring (main.py, beside its call
        #     to _enforce_hard_budget), copied in shape ---
        margin_at_guard = main._BUDGET_MARGIN
        standin_protected = True
        if (
            stored_out and stored_out[0] > 0
            and margin_at_decision
            and margin_at_guard > margin_at_decision[0]
        ):
            standin_protected = False

        # --- the rest of what chat_completions does after compact_if_needed
        #     (persona + injected memory, then the real guard) — _g14_guard's
        #     own body, plus the new parameter it does not know about ---
        standin = next((m for m in out if main._is_compaction_standin(m)), None)
        rest = [m for m in out if m.get("role") != "system"]
        full = [out[0], {"role": "system", "content": "P" * 2500}]
        if standin is not None:
            full.append(standin)
        full.append({"role": "system", "content": _G11_MEM})
        full += rest
        rep: dict = {}
        result = main._enforce_hard_budget(
            full, EFFECTIVE_LIMIT, 1, rep, 0, standin_protected,
        )
        ns_result = [m for m in result if m.get("role") != "system"]
        prev_survived = (
            any("prev-u" in main._message_text(m) for m in ns_result)
            and any("prev-a" in main._message_text(m) for m in ns_result)
        )
        newest_survived = any(
            "newest-u" in main._message_text(m) for m in ns_result
        )
        return (
            stored_out, margin_at_decision, margin_at_guard,
            standin_protected, rep, prev_survived, newest_survived,
        )
    finally:
        main.count_tokens_exact = _g20_saved_cte
        main._BUDGET_MARGIN = saved
        main.summarize = saved_summarize
        main.INJECTION_BUDGET_FRACTION = saved_fraction


# [21a] CONTROL: margin 0 throughout — unaffected by this fix.
_g21a = _g21_run(_spy_summarize, 0)
check(_g21a[5], f"[21a] CONTROL margin 0 throughout: her previous exchange survives (rep={_g21a[4]})")
check(_g21a[6], "[21a]: the newest turn always survives")

# [21b] CONTROL: margin 8192 in force from the start — the P13-1 shape;
# also unaffected (the window check itself declines at this margin, so
# there is no stand-in to protect or demote).
_g21b = _g21_run(_spy_summarize, 8192)
check(_g21b[5], f"[21b] CONTROL margin 8192 throughout: her previous exchange survives (rep={_g21b[4]})")
check(_g21b[6], "[21b]: the newest turn always survives")

# [21c] *** THE FIX: 0 at the window check, latched to 8192 by the time the
# guard runs — before this fix: a=kept b=kept c=LOST (SP\p14-findings.md,
# P14-1). After it, the fix must see the margin grew and demote the
# stand-in's protection, and her previous exchange must survive.
_g21c = _g21_run(_g21_bumping_summarize, 0)
check(
    _g21c[0] and _g21c[0][0] > 0,
    f"[21c] fixture: reuse fires at the margin the decision saw "
    f"(stored_turns_out={_g21c[0]})",
)
check(
    _g21c[3] is False,
    f"[21c]: the fix demotes the stand-in's protection once the LIVE "
    f"margin ({_g21c[2]}) exceeds what the decision read "
    f"({_g21c[1]}) — standin_protected={_g21c[3]}",
)
check(
    _g21c[5],
    f"*** P14-1 [21c] THE FIX: her previous exchange survives the race "
    f"that lost it before this fix (a=kept b=kept c=LOST) — rep={_g21c[4]}",
)
check(_g21c[6], "[21c]: the newest turn always survives")

# [21d] CONTROL: the margin FALLS mid-request (another request's accepted
# streak released it) — must not make anything worse. The decision was
# already made under the LARGER margin, so it was already conservative;
# the fix must not demote the stand-in here, and the exchange must survive
# regardless.
_g21d = _g21_run(_g21_falling_summarize, 8192)
check(
    _g21d[3] is True,
    f"[21d]: a FALLING margin does not demote the stand-in's protection "
    f"(standin_protected={_g21d[3]}) — the decision it was made under was "
    f"already conservative",
)
check(_g21d[5], f"[21d]: her previous exchange survives when the margin only falls (rep={_g21d[4]})")
check(_g21d[6], "[21d]: the newest turn always survives")

main._BUDGET_MARGIN = 0
main._budget_ok_streak = 0


# ---------------------------------------------------------------------------
# [22] P14-1, THE WIRING FOR REAL (same doctrine as [8]/[9] above: "nothing
# that calls a function can see whether its call site is right"). [21]
# proves `compact_if_needed`'s `reuse_margin_out` and `_enforce_hard_
# budget`'s `standin_protected` are correct in themselves, but it drives
# both directly and copies chat_completions's own margin-comparison logic
# IN THE TEST — a mutant inside chat_completions's actual wiring (the
# comparison, which field it reads, whether it is wired in at all) is
# invisible to [21]. This section drives the REAL endpoint instead
# (POST /v1/chat/completions via TestClient, [8]'s own infrastructure),
# with `main.summarize` patched to bump `_BUDGET_MARGIN` mid-request the
# same way [21c]'s race does, and checks the ACTUAL forwarded payload —
# not a value this test computed, the bytes chat_completions put on the
# wire.
# ---------------------------------------------------------------------------
print("\n[22] P14-1 driving chat_completions itself: a margin latched mid-"
      "request must not cost her previous exchange on the real endpoint")

CONV_G22 = _G17_CONV


async def _g22_bumping_summarize(client, to_summarize):
    """Same race as [21c]'s, at the real endpoint: another request's
    unpredicted rejection lands while THIS request awaits its own
    fresh-span summarize()."""
    main._note_backend_rejection(
        _G21_OVERFLOW, enforced_limit=20768, guard_measured_overflow=False
    )
    return await _spy_summarize(client, to_summarize)


_g22_msgs, _g22_last, _g22_render = _g17_build(
    2, 3750, aprev_tokens=6500, n_l1=9, l3_tokens=1869,
)
_g22_saved_fraction = main.INJECTION_BUDGET_FRACTION
_g22_saved_sbmax = summarizer.SUMMARY_BLOCK_MAX_TOKENS
_g22_saved_margin = main._BUDGET_MARGIN
main.INJECTION_BUDGET_FRACTION = 0.75  # [17]/[19]/[21]'s own fixture assumes this
summarizer.SUMMARY_BLOCK_MAX_TOKENS = _G12_SHIPPED_SBMAX
main.count_tokens_exact = _g20_stub_exact(1.05)
main.summarize = _g22_bumping_summarize
main._BUDGET_MARGIN = 0
def _g22_no_redact(messages):
    """Passthrough for main._redact_forwarded_loop_replies (the fence
    lane's loop-detection code, main.py:8758-8975 — not this lane's to
    edit). Needed only because this fixture's fat_turn() content is a tag
    plus a long run of ONE repeated character, by design (cheap, exact
    token control) — which is indistinguishable, to an UNRELATED
    degeneracy detector, from a model generating a repetitive loop, and it
    redacts the "prev-a" marker this check looks for. Orthogonal to
    P14-1/G2: [8]/[9] above use realistic (non-repetitive) content and
    never trip it; this fixture just cannot use fat_turn's shape and also
    exercise that subsystem in the same request."""
    return messages, 0, 0


_g22_handler = _CaptureLogs()
_ep_logger.addHandler(_g22_handler)
try:
    _EndpointStubVLLM.sent.clear()
    with _patch.object(main.httpx, "AsyncClient", _EndpointStubVLLM), \
         _patch.object(main, "_fire_and_forget", _swallow_tail), \
         _patch.object(main, "_redact_forwarded_loop_replies", _g22_no_redact):
        _g22_resp = _client.post(
            "/v1/chat/completions",
            json={"model": "stub-model", "messages": _g22_msgs, "stream": False},
            headers={"X-Conversation-Id": CONV_G22},
        )
finally:
    _ep_logger.removeHandler(_g22_handler)
    summarizer.SUMMARY_BLOCK_MAX_TOKENS = _g22_saved_sbmax
    main.INJECTION_BUDGET_FRACTION = _g22_saved_fraction
    main.count_tokens_exact = _g20_saved_cte
    main.summarize = _spy_summarize
    main._BUDGET_MARGIN = _g22_saved_margin
    main._budget_ok_streak = 0

check(
    _g22_resp.status_code == 200,
    f"fixture: the real endpoint request succeeded (got {_g22_resp.status_code}: "
    f"{_g22_resp.text[:200]})",
)
_g22_forwarded = _EndpointStubVLLM.sent[-1] if _EndpointStubVLLM.sent else None
_g22_fwd_text = (
    " ".join(str(m.get("content", "")) for m in _g22_forwarded.get("messages", []))
    if _g22_forwarded else ""
)
check(
    main.COMPACTION_SUMMARY_HEADER in _g22_fwd_text,
    "fixture: the stand-in reached the wire — reuse fired, so this request "
    "actually exercises the race (a request that declined has no stand-in "
    "to demote)",
)
check(
    "prev-u" in _g22_fwd_text and "prev-a" in _g22_fwd_text,
    f"*** P14-1 [22] THE FIX, END TO END: her previous exchange "
    f"(U_prev/A_prev) is in the ACTUAL payload chat_completions put on "
    f"the wire, driven through the real endpoint with a margin latched by "
    f"another request while this one awaited summarize() — before this "
    f"fix, this exact race dropped it (SP\\p14-findings.md, P14-1)",
)
check(
    "newest-u" in _g22_fwd_text,
    "[22]: the newest turn always survives, at the real endpoint too",
)


# ---------------------------------------------------------------------------
# [23] G3c (hostile pass #14's "not demonstrated" list, lane v3194-guard):
# `_sys_recent_floor` (system prompt + recent window) used to be counted
# with the LOCAL count_tokens alone, while the guard downstream verifies
# the array it actually sheds with vLLM's own exact count. `keep_recent`
# carries `A_prev`, her previous reply — exactly the content this file
# documents the local tokenizer reading 34-51% low on (count_tokens_
# exact's own docstring: decorative box-drawing characters and emoji). A
# floor read too LOW makes `_standin_structural_ceiling` (`_effective_
# limit_est - _sys_recent_floor`) read too HIGH, so the window check could
# approve a stand-in that, once the guard measures honestly, does not
# actually fit beside her previous exchange — the SAME P13-1 shape, from a
# different counter being wrong.
#
# Reproduced: a synthetic counter that reads `A_prev`-tagged content at
# 49% of its true price (the documented worst case, matching this file's
# and count_tokens_exact's own "up to 51% low" language) on p14's own
# peakA fixture. Before the fix: reuse fired at aprev_tokens 10,000-14,000
# and the guard then shed her previous exchange (2 turns dropped) to
# compensate. After it: the window check declines at those same sizes
# instead (stored_turns_out=[0]) — the guard is never put in that
# position at all.
#
# THE FIX: `_sys_recent_floor` gets its own `count_tokens_exact` call
# (the SAME pattern `_fresh_scale` a few lines below it already uses),
# falling back to the PLAIN local count — unscaled, not the pessimistic
# 2.0x `_fresh_scale` falls back to — when `/tokenize` does not answer.
# That asymmetry is deliberate: a first draft used the pessimistic
# fallback here too and broke 18 of this file's OWN checks ([14] 'today'/
# 'peakA', [17b], [18b], [19a], [20c], [21c] among them) — `/tokenize`
# being unreachable is this suite's own default test posture (most
# fixtures never point VLLM_URL at a real server), so doubling the
# recent-window floor whenever it is down doubled it on states this
# file's hostile-pass history had already proven should reuse. The
# pre-fix code already forwarded the plain local count on that path; this
# fix changes nothing there, only the common case where `/tokenize` DOES
# answer.
# ---------------------------------------------------------------------------
print("\n[23] G3c: _sys_recent_floor is counted exactly when /tokenize "
      "answers, not just approximately with the local estimator")


def _g23_true(txt):
    return len(txt) // 4 + 4


def _g23_local_counter(ms):
    total = 0
    for m in ms:
        txt = main._message_text(m)
        n = _g23_true(txt)
        if "prev-a" in txt:
            n = int(n * 0.49)  # the documented worst case: 51% low
        total += n
    return total


def _g23_exact_counter(ms, *a, **k):
    return sum(_g23_true(main._message_text(m)) for m in ms)


def _g23_run(aprev_tokens, count_tokens_stub, count_tokens_exact_stub):
    saved_fraction = main.INJECTION_BUDGET_FRACTION
    saved_summarize = main.summarize
    saved_ct = main.count_tokens
    saved_cte = main.count_tokens_exact
    main.INJECTION_BUDGET_FRACTION = 0.75
    main.summarize = _spy_summarize
    main.count_tokens = count_tokens_stub
    main.count_tokens_exact = count_tokens_exact_stub
    try:
        stored, out, last, render, log = _g17_run(
            _spy_summarize, 2, 3750, n_l1=9, l3_tokens=1869,
            aprev_tokens=aprev_tokens,
        )
        rep, prev, newest = _g14_guard(out, last, "g23")
        return stored, out, last, rep, prev, newest
    finally:
        main.INJECTION_BUDGET_FRACTION = saved_fraction
        main.summarize = saved_summarize
        main.count_tokens = saved_ct
        main.count_tokens_exact = saved_cte


# [23a] *** THE FIX: at aprev_tokens=12000 — inside the window where the
# unfixed code fired reuse and the guard then shed her previous exchange
# (reproduced: stored=[130], dropped_turns=2, prev_kept=False) — the
# window check now declines instead, so the guard never has to choose.
_g23a_stored, _g23a_out, _g23a_last, _g23a_rep, _g23a_prev, _g23a_newest = _g23_run(
    12000, _g23_local_counter, _g23_exact_counter
)
check(
    _g23a_stored == [0],
    f"*** G3c [23a] THE FIX: the window check declines once the recent "
    f"floor is measured exactly (stored_turns_out={_g23a_stored}) — "
    f"before this fix it fired here and the guard then shed her previous "
    f"exchange to compensate",
)
check(
    _g23a_prev and _g23a_newest,
    f"[23a]: her previous exchange and the newest turn both survive end "
    f"to end ({_g23a_rep})",
)

# [23b] CONTROL: /tokenize down for this call specifically — falls back to
# the PLAIN local count (not a pessimistic multiplier), so behaviour here
# is UNCHANGED from before this fix: reuse still fires (the local counter
# under-reads the floor exactly as it always did on this path, and this
# fix does not touch that).
_g23b_stored, *_g23b_rest = _g23_run(
    12000, _g23_local_counter, _g20_stub_exact(None)
)
check(
    _g23b_stored == [130],
    f"[23b] CONTROL (/tokenize down): unchanged from before this fix — "
    f"the plain local count is used, same as the pre-fix code always did "
    f"on this path (stored_turns_out={_g23b_stored})",
)

# [23c] CONTROL: a SMALL A_prev, where even a 51%-low local reading is
# nowhere near enough to matter — reuse must still fire normally. The fix
# must not make this check MORE conservative than it needs to be.
_g23c_stored, _g23c_out, _g23c_last, _g23c_rep, _g23c_prev, _g23c_newest = _g23_run(
    4000, _g23_local_counter, _g23_exact_counter
)
check(
    _g23c_stored == [130],
    f"[23c] CONTROL: a small A_prev still reuses normally "
    f"(stored_turns_out={_g23c_stored}) — the fix declines only when the "
    f"REAL floor genuinely does not leave room, not unconditionally",
)
check(_g23c_prev and _g23c_newest, f"[23c]: her previous exchange and the newest turn survive ({_g23c_rep})")


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nall checks passed")
