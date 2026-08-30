"""
Conversation forks: detecting them, and putting the halves back together.

THE FAILURE THIS IS ABOUT. The hash-fallback conv_id is
sha256(system|||first_user[:512]), so editing the SYSTEM PROMPT gives a live
conversation a brand-new identity. Facts, episodic embeddings and summaries
all keep accumulating perfectly — just under an id nothing else references.

Production, 2026-08-30: a prompt edit at ~19:08 forked a ~400-turn
conversation. The old id kept 106 facts and ~85 indexed exchanges; the new
one carried on and re-derived its own summary hierarchy. It went unnoticed
for hours because every individual signal looked healthy. Only the identity
moved, and nothing said so.

Two halves to this file:
  [1] the detector — the signature is long + hash-derived + no stored state
  [2] the merge — fold the stranded half back in, without touching summaries

    python test_conv_fork.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ.setdefault(
    "COMPACTOR_STORAGE_ROOT", tempfile.mkdtemp(prefix="conv-fork-")
)

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import portability  # noqa: E402

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)


def _warns(conv_id, source, messages):
    """Did the detector fire? Captured off main's own logger."""
    import io as _io
    import logging

    buf = _io.StringIO()
    h = logging.StreamHandler(buf)
    main.logger.addHandler(h)
    try:
        main._warn_if_conversation_forked(conv_id, source, messages)
    finally:
        main.logger.removeHandler(h)
    return "signature of a FORK" in buf.getvalue()


LONG = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
    for i in range(60)
]

print("[1] the detector")
check(
    _warns("fresh_hash_id", "hash", LONG),
    "a long, hash-derived conversation with no stored memory IS flagged - "
    "that is the fork signature and nothing else produces it",
)
check(
    not _warns("short_id", "hash", LONG[:6]),
    "a genuinely NEW conversation is not flagged (few messages)",
)
check(
    not _warns("stable_id", "header", LONG),
    "a header-sourced id is never flagged - it is stable across prompt edits",
)
check(
    not _warns("meta_id", "body_metadata.chat_id", LONG),
    "a metadata-sourced id is never flagged, same reason",
)

facts.save_facts("resumed_id", [{"text": "known", "added_turn": 1, "last_used": 1}])
check(
    not _warns("resumed_id", "hash", LONG),
    "a RESUMED conversation with stored facts is not flagged - it is only "
    "the empty-and-long combination that means the identity moved",
)

print()
print("[2] the merge")
facts.save_facts("fork_src", [
    {"text": "she grew up near the coast", "added_turn": 1, "last_used": 100},
    {"text": "a   Shared  fact", "added_turn": 2, "last_used": 100},
])
facts.save_facts("fork_dst", [
    {"text": "A Shared Fact", "added_turn": 9, "last_used": 900},
])

plan = portability.merge_conversation("fork_src", "fork_dst")
check(plan["dry_run"] is True,
      "DRY RUN BY DEFAULT - the compact endpoint defaults the other way and "
      "surprised an operator into a live run; this touches two conversations")
check(plan["facts_to_add"] == 1 and plan["facts_skipped_duplicate"] == 1,
      "the plan dedups on normalized text (case and whitespace), so the "
      "overlap both halves extracted is not imported twice")
check(len(facts.load_facts("fork_dst")) == 1,
      "a dry run writes nothing")

done = portability.merge_conversation("fork_src", "fork_dst", dry_run=False)
check(done["facts_added"] == 1 and len(facts.load_facts("fork_dst")) == 2,
      "the committed merge adds only the genuinely new fact")
check(len(facts.load_facts("fork_src")) == 2,
      "the SOURCE is left completely intact - a merge that damages its "
      "source is unrecoverable if the result is wrong")
check("not merged" in done["summaries"],
      "summaries are NOT merged - dst re-derived its own hierarchy over the "
      "same history, so folding src's in would double-count the narrative")

again = portability.merge_conversation("fork_src", "fork_dst", dry_run=False)
check(again["facts_added"] == 0 and len(facts.load_facts("fork_dst")) == 2,
      "re-running is a no-op - an operator who runs it twice does no harm")

print()
print("[3] the merge refuses rather than corrupting")


async def _while_locked():
    async with memory.conv_lock("fork_dst"):
        try:
            portability.merge_conversation("fork_src", "fork_dst", dry_run=False)
            return False
        except ValueError:
            return True


check(
    asyncio.run(_while_locked()),
    "refuses while a memory write is in flight - conv_lock is an asyncio "
    "lock and this runs in a threadpool, so it cannot await; the extraction "
    "tail would otherwise write its pre-merge snapshot back over the merge",
)

for src, dst, why in (
    ("same", "same", "src and dst are the same conversation"),
    ("", "fork_dst", "an empty src id"),
    ("no_such_conv", "fork_dst", "a src with no state at all"),
):
    try:
        portability.merge_conversation(src, dst)
        FAILED.append(f"did not refuse: {why}")
    except ValueError:
        print(f"  ok   refuses {why}")

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All conversation-fork tests passed.")
