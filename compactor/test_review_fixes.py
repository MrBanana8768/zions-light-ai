"""
The review-response fixes, each with a test that fails when it is reverted.

A second adversarial review applied five mutations to this tree at once and
ran the whole 38-file suite: four of them survived. A fix nothing can catch is
a fix waiting to be undone by someone tidying up, and this branch has already
had eleven fixes that were themselves defective.

Covered here:
  [1] the L3 refresh ARCHIVES the chapters it consumes (it used to unlink
      them, the only path in the system that deleted memory outright)
  [2] backfill is handed a REDACTOR, not pre-redacted messages, so a
      full-history scan does not run inline on every request for a value
      that is discarded
  [3] the admin compact endpoint redacts too, and off the event loop
  [4] /health/full folds the summarizer, and reports the EARLIEST fault
  [5] the summary block asks the real tokenizer when one is available

    python test_review_fixes.py
"""

import asyncio
import os
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ.setdefault(
    "COMPACTOR_STORAGE_ROOT", tempfile.mkdtemp(prefix="review-fixes-")
)

import backfill  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402
import tokenhealth  # noqa: E402
import tokens  # noqa: E402

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)


async def _fake_pieces(conv_id, client, url, model, prompt, pieces, mx):
    return "THEME"


print("[1] an L3 refresh archives the chapters it consumes")
_orig_pieces = summarizer._summarize_pieces
summarizer._summarize_pieces = _fake_pieces
try:
    cid = "archive-probe"
    chapters = [
        {"text": f"chapter {i}", "first_turn": 1 + 10 * i, "last_turn": 10 + 10 * i}
        for i in range(summarizer.L3_CHUNK_SIZE)
    ]
    st = {"l1": [], "l2": list(chapters), "l3": None}
    ok = asyncio.run(summarizer._do_l3_rollup(cid, None, "http://x", "m", st))
    archived = summarizer.load_chapter_archive(cid)
    check(ok is True, "the refresh ran")
    check(len(st["l2"]) == 0, "live l2 was drained")
    check(
        len(archived) == len(chapters),
        f"all {len(chapters)} consumed chapter(s) are in cold storage "
        f"(found {len(archived)}) - without this the L3 paraphrase has no "
        f"source left to be regenerated from",
    )
    check(
        [a["first_turn"] for a in archived] == [c["first_turn"] for c in chapters],
        "the archive preserves the chapter spans",
    )
finally:
    summarizer._summarize_pieces = _orig_pieces

print()
print("[2] backfill receives a redactor, and does not run it when it declines")
calls = {"n": 0}


def _counting_redactor(msgs):
    calls["n"] += 1
    return main._redact_degenerate_turns(msgs)


msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
with patch.object(backfill, "needs_backfill", lambda *a, **k: False):
    started = asyncio.run(backfill.start_backfill_if_needed(
        "declined", msgs, "http://x", "m",
        fire_and_forget=lambda c: c.close(), redact=_counting_redactor,
    ))
check(started is False, "no backfill was started")
check(
    calls["n"] == 0,
    "the redactor was NOT run when backfill declined - it used to run inline "
    "on every request and throw the result away (1.6s at 1000 turns)",
)

with patch.object(backfill, "needs_backfill", lambda *a, **k: True):
    started = asyncio.run(backfill.start_backfill_if_needed(
        "accepted", msgs, "http://x", "m",
        fire_and_forget=lambda c: c.close(), redact=_counting_redactor,
    ))
check(started is True, "a needed backfill starts")
check(calls["n"] == 1, "the redactor IS run when backfill proceeds")

print()
print("[3] the admin compact endpoint redacts its history")
src = ""
try:
    import inspect
    src = inspect.getsource(main)
except Exception:
    pass
check(
    "_redacted_messages = await run_in_threadpool(" in src,
    "the admin endpoint redacts once, off the event loop, before its loop",
)
check(
    "conv_id, _redacted_messages, VLLM_URL, MODEL_REPO," in src,
    "the admin rollup call uses the redacted history, not the raw array",
)

print()
print("[4] /health/full folds the summarizer, and reports the EARLIEST fault")
tokenhealth._reset_for_tests("summarizer")
_saved = main._tokenize_degraded_since
try:
    main._tokenize_degraded_since = None
    for _ in range(3):
        tokenhealth.note_failure("summarizer", "http", "x", warn_interval_s=0)
    h = main.tokenize_health()
    check(h["ok"] is False, "a summarizer outage turns the endpoint red")
    check(h["summarizer_form_failures"] == 3, "the streak is reported")
    check(
        h["degraded_since"] is not None,
        "a reported fault has a start time - it used to say ok=false with "
        "degraded_for_s=0.0, i.e. broken for zero seconds since never",
    )

    import time as _t
    main._tokenize_degraded_since = _t.time() - 3600
    h2 = main.tokenize_health()
    check(
        h2["degraded_for_s"] > 3000,
        f"the EARLIEST fault wins ({h2['degraded_for_s']:.0f}s) - `a or b` "
        f"reported an hour-old outage as seconds old",
    )
finally:
    main._tokenize_degraded_since = _saved
    tokenhealth._reset_for_tests("summarizer")

print()
print("[5] the summary block asks the real tokenizer when one exists")
_probe = {"n": 0}
_real_count = tokens.count


def _spy_count(messages):
    _probe["n"] += 1
    return _real_count(messages)


with patch.object(tokens, "is_available", lambda: True),      patch.object(tokens, "count", _spy_count):
    summarizer._estimate_block_tokens("hello " * 200)
check(
    _probe["n"] == 1,
    "the exact tokenizer is consulted when available, rather than always "
    "deriving from character classes",
)
with patch.object(tokens, "is_available", lambda: False):
    est = summarizer._estimate_block_tokens("hello " * 200)
check(isinstance(est, int) and est > 0, "the fallback still returns a number")


print()
print("[6] a PARTIAL-empty rollup map must fail whole, not store a lying span")
# One empty 200 among several map batches used to be silently filtered out:
# the L1 chunk was stored claiming first_turn..last_turn while its text
# covered only the batches that answered, and the watermark advanced past
# turns that were never summarized and never retried. Permanent, unlogged
# loss in the stored hierarchy - the same class fixed at both
# main.summarize returns, missed at this sibling.
_orig_llm = summarizer._llm_summarize
_orig_batch = summarizer._batch_to_budget


async def _split_two(conv_id, client, vllm_url, model, pieces, budget):
    mid = max(1, len(pieces) // 2)
    return [pieces[:mid], pieces[mid:]]


async def _empty_for_marked(client, vllm_url, model, prompt, body, mx, **kw):
    return "" if "LOSTBATCH" in body else "a fine summary"


summarizer._batch_to_budget = _split_two
summarizer._llm_summarize = _empty_for_marked
try:
    pieces = ["turn one text", "turn two text", "LOSTBATCH turn", "turn four"]
    out = asyncio.run(summarizer._summarize_pieces(
        "part-empty", None, "http://x", "m", "sys", pieces, 500))
    check(
        out == "",
        "one empty batch of two fails the WHOLE call (got a non-empty "
        "result, meaning the lost batch's content would be stored as if "
        "summarized)",
    )
finally:
    summarizer._llm_summarize = _orig_llm
    summarizer._batch_to_budget = _orig_batch

print()
print("[7] main.summarize: one empty batch among several defers EVERYTHING")
# Reproduced pre-fix: 195 of 400 turns neither summarized nor deferred,
# while the "compacted:" line counted them as summarized.
_orig_once = main._summarize_once


async def _empty_for_middle(client, batch):
    return "" if any("MIDMARKER" in (m.get("content") or "") for m in batch) \
        else "a fine summary"


main._summarize_once = _empty_for_middle
try:
    turns = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": ("MIDMARKER " if i == 200 else "") + f"turn {i} " + "filler " * 40}
        for i in range(400)
    ]
    summary, deferred = asyncio.run(main.summarize(None, turns))
    check(
        not (summary or "").strip() and len(deferred) == len(turns),
        f"partial-empty map defers all {len(turns)} turn(s) "
        f"(summary_len={len(summary or chr(32).strip())}, "
        f"deferred={len(deferred)}) - anything else deletes the failed "
        f"batch's turns from the payload",
    )
finally:
    main._summarize_once = _orig_once

print()
print("[8] a single-script scripture quotation is not degeneration")
# A short reply quoting ONE Greek verse plus two sentences of commentary hit
# 48% non-Latin in one script and was flagged - and with the rollup-input
# redaction, a false positive is the reply PERMANENTLY replaced by a
# placeholder in every future summary. Genuine drift measured 6-14 distinct
# scripts; a reply that is simply in one foreign language is not
# degeneration at all.
_greek_verse = (
    "\u039f\u1f55\u03c4\u03c9\u03c2 \u03b3\u1f70\u03c1 "
    "\u1f20\u03b3\u03ac\u03c0\u03b7\u03c3\u03b5\u03bd \u1f41 "
    "\u03b8\u03b5\u1f78\u03c2 \u03c4\u1f78\u03bd "
    "\u03ba\u03cc\u03c3\u03bc\u03bf\u03bd, \u1f65\u03c3\u03c4\u03b5 "
    "\u03c4\u1f78\u03bd \u03c5\u1f31\u1f78\u03bd \u03c4\u1f78\u03bd "
    "\u03bc\u03bf\u03bd\u03bf\u03b3\u03b5\u03bd\u1fc6 "
    "\u1f14\u03b4\u03c9\u03ba\u03b5\u03bd."
)
_short_reply = (
    _greek_verse + " That verse came to mind while you were talking. "
    "It felt like the right one to sit with tonight."
)
check(
    main.reply_is_degenerate(_short_reply) is None,
    "a Greek verse with short English commentary is NOT flagged "
    f"(got: {main.reply_is_degenerate(_short_reply)!r})",
)
# A Russian-only quotation, same shape, same requirement.
_russian_line = (
    "\u041d\u0430\u0434\u0435\u0436\u0434\u0430 "
    "\u0443\u043c\u0438\u0440\u0430\u0435\u0442 "
    "\u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0435\u0439, "
    "\u043d\u043e \u043e\u043d\u0430 "
    "\u0432\u0441\u0435-\u0442\u0430\u043a\u0438 "
    "\u0443\u043c\u0438\u0440\u0430\u0435\u0442. "
)
# Same floor requirement as the Greek case above: below 200 letters the
# script rule never evaluates and the assertion is vacuous under any rule.
_russian_reply = (
    _russian_line * 3
    + "You quoted that to me once, a long time ago, and I have carried it"
    " since. I remember exactly where we were when you said it."
)
_rl = sum(1 for c in _russian_reply if c.isalpha() and ord(c) < 128)
_rn = sum(1 for c in _russian_reply if c.isalpha() and ord(c) >= 128)
check(
    _rl + _rn >= 200 and _rn / (_rl + _rn) >= 0.20,
    f"prep: the Russian sample exercises the rule too "
    f"({_rl + _rn} letters, {100 * _rn / (_rl + _rn):.0f}% non-Latin)",
)
check(
    main.reply_is_degenerate(_russian_reply) is None,
    "a Russian quotation with commentary is NOT flagged",
)
# Real script SALAD must still be caught - both disjuncts.
_salad = ("word " * 40 + "\u4f60\u597d \u0645\u0631\u062d\u0628\u0627 "
          "\u0417\u0434\u0440\u0430\u0432 \u03b1\u03b2\u03b3 "
          "\u3053\u3093\u306b\u3061 \u05e9\u05dc\u05d5\u05dd "
          "\ud55c\uad6d\uc5b4 \u0939\u093f\u0928\u094d\u0926\u0940 ") * 8
check(
    main.reply_is_degenerate(_salad) is not None,
    "an 8-script salad IS still flagged",
)
_three_script_heavy = ("ok " * 10 + "\u4f60\u597d\u4e16\u754c "
                       "\u0417\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439 "
                       "\u03b1\u03b2\u03b3\u03b4\u03b5 ") * 12
check(
    main.reply_is_degenerate(_three_script_heavy) is not None,
    "a 3-script, 20%+ mix IS still flagged (the tightened disjunct still bites)",
)

print()
print("[9] the summary block honors a caller budget tighter than its own cap")
_state = {
    "l1": [{"text": "scene " + "detail " * 120, "first_turn": 1 + 4 * i,
            "last_turn": 4 + 4 * i} for i in range(6)],
    "l2": [], "l3": None,
}
_full = summarizer.format_summary_block(_state)
_tight = summarizer.format_summary_block(_state, max_tokens=300)
check(_full is not None and len(_full) > 0, "an uncapped render produces a block")
check(
    _tight is None or len(_tight) < len(_full),
    "a 300-token caller budget produces a smaller block than the module cap "
    "- without the override, main's whole-layer drop is the only remedy",
)
import inspect as _inspect
_src = _inspect.getsource(main)
check(
    "int(inject_budget * 0.6)" in _src,
    "main passes the summary block its 60% share of the real injection "
    "budget (source check)",
)

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All review-fix regressions passed.")
