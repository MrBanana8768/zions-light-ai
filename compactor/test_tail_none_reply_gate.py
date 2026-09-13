"""
compactor/test_tail_none_reply_gate.py

hostile2-config: `_facts_tail` (job 2 of `_async_tail`) spells its reply
check `(assistant_text or "").strip()`, explicitly None-tolerant, and its
own comment argued this was "DEFENCE ONLY, and currently unreachable: job
1's gate runs first and raises AttributeError on a None reply before
control ever arrives here" — job 1 (`_async_tail`'s own episodic-indexing
gate, a few lines above the call into job 2) wrote a bare
`assistant_text.strip()`. Job 2's tolerance was real code with no way to
ever run.

This drives the REAL `main._async_tail` — not a stand-in for either gate —
with `assistant_text=None`, the shape a "direct caller" (the finding's own
words, and this codebase's own docstring for `_rollup_hierarchy`, which
already accepts and documents None as "roll up the history alone") could
pass. Job 3 (`_rollup_hierarchy`) already tolerates None by design (the
skipped-tail path uses exactly that); this test's job is jobs 1 and 2.

Run:
    python test_tail_none_reply_gate.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="tail-none-reply-")
# Extraction must be ON so job 2 (_facts_tail) reaches its own
# (assistant_text or "").strip() line rather than short-circuiting earlier
# on the "extraction disabled" touched-save path, which would prove nothing
# about the line this test exists to cover.
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "true"

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


HISTORY = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "an earlier answer"},
    {"role": "user", "content": "and now, a direct call with no reply text"},
]

_real_index = retrieval.index_exchange
_real_rollup = summarizer.maybe_rollup


def _run(conv_id: str, assistant_text, *, indexed: list, rolled: list):
    """Run the real _async_tail with job 1 (episodic) and job 3 (rollup)'s
    side effects spied rather than stubbed away — only their network/DB
    calls are replaced, so both gates run exactly as shipped."""

    def _spy_index(cid, turn_index, user_text, asst_text):
        indexed.append((cid, turn_index, user_text, asst_text))
        return True

    async def _spy_rollup(cid, messages, vllm_url, model, **kw):
        rolled.append(list(messages))
        return summarizer.load_state(cid)

    retrieval.index_exchange = _spy_index
    summarizer.maybe_rollup = _spy_rollup
    try:
        asyncio.run(main._async_tail(
            conv_id, [], "and now, a direct call with no reply text",
            assistant_text, 3, list(HISTORY),
        ))
        return None
    except Exception as e:  # noqa: BLE001 - the exact thing under test
        return e
    finally:
        retrieval.index_exchange = _real_index
        summarizer.maybe_rollup = _real_rollup


print("[1] CONTROL — a real reply still runs job 1 (episodic indexing)")
indexed, rolled = [], []
raised = _run("tail_none_control", "a real, non-empty reply", indexed=indexed, rolled=rolled)
check(raised is None, f"the control call must not raise either (got {raised!r})")
check(len(indexed) == 1,
      f"job 1 (episodic indexing) ran for a real reply (got {len(indexed)} call(s))")
check(len(rolled) == 1,
      f"job 3 (rollup) ran too (got {len(rolled)} call(s))")

print("[2] assistant_text=None must not raise, and must not index a phantom exchange")
indexed, rolled = [], []
raised = _run("tail_none_direct", None, indexed=indexed, rolled=rolled)
check(raised is None,
      f"*** hostile2-config: a None reply reaching _async_tail directly "
      f"must not raise AttributeError (got {raised!r})")
check(indexed == [],
      "job 1 did not index a None reply as though it were text")
check(len(rolled) == 1,
      "job 3 still ran (it already tolerates None by design — the skipped-"
      "tail path relies on exactly that) — CONTROL that this is jobs 1/2's "
      "gates narrowing, not the whole tail being skipped")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll test_tail_none_reply_gate checks passed.")
