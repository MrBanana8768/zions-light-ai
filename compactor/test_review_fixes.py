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
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All review-fix regressions passed.")
