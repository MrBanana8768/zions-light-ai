"""Compaction reuse under the traffic OpenWebUI actually produces.

hostile pass #3 (reviewer A F1/F2/F3/F7/F9, reviewer D F1). Every earlier
fixture of the reuse gate built one array, recorded it, and compacted it. The
defects that shipped were all BETWEEN requests: a record entry written from a
later request than the chunk it vouches for (F1: delete, regenerate or delete
the reply right after a chunk closes; F9: an admin rebuild, then a live turn),
and a positional compare behind a length gate that one delete, edit-and-resend,
regenerate or pair of tool messages switched off for good (F2/F3/F7; reviewer
D found five such edits in seven days of her real chat).

So this file drives a conversation the way chat_completions and the tail do,
one exchange at a time: the REAL compact_if_needed (with stored_turns_out) on
every request and the REAL _rollup_hierarchy (redaction, maybe_rollup, the raw
request, the reply as streamed) after it. Two LLM seams are stubbed and both
echo the tags of what they were handed: summarize() returns FRESH{tags} and
summarizer._summarize_pieces returns CHUNK{tags}, so the returned array says
exactly which turns it carries.

THE ORACLE, after every compaction: every turn of the request carries a unique
tag (U61v1 = turn 61, content version 1), and its CURRENT tag must be visible
in what the model receives — the returned array, whose stand-in and fresh
summary carry the tags of what they summarized. A turn whose tag appears
nowhere is lost; one where only an older version appears was replaced by a
summary of text she deleted or replaced.

And two liveness properties, because a gate that refuses everything passes the
oracle: once the record covers anything, EVERY compacting request reuses it,
and the fresh span stays bounded (one L1 chunk, the recent window, and the
turns the traffic made unmatchable).

Adapted from reviewer A's SP/p3-a/sim.py. Synthetic text only.

    python test_p3a_reuse_traffic.py
"""

import asyncio
import logging
import os
import random
import re
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="p3a-traffic-")
# Every request past ~15 turns compacts, and the stand-in (F5 budget: TARGET
# minus system, recent turns and one fresh summary) has room.
os.environ["COMPACTOR_TARGET_TOKENS"] = "1500"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "64"

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


TAG = re.compile(r"\b[UAT]\d+v\d+\b")
FRESH_INPUTS: list[list[dict]] = []


async def _spy_summarize(client, turns):
    FRESH_INPUTS.append(list(turns))
    tags = []
    for m in turns:
        tags += TAG.findall(summarizer._message_text(m))
    return "FRESH{" + ",".join(tags) + "}", []


async def _stub_pieces(conv_id, client, vllm_url, model, prompt, pieces, max_tokens):
    tags = []
    for p in pieces:
        tags += TAG.findall(p)
    return "CHUNK{" + ",".join(tags) + "}"


_real_summarize = main.summarize
_real_pieces = summarizer._summarize_pieces
main.summarize = _spy_summarize
summarizer._summarize_pieces = _stub_pieces

SYS = {"role": "system", "content": "you are a companion"}
_W = ["river", "lantern", "quiet", "morning", "garden", "letter", "stone", "window",
      "silver", "harbor", "meadow", "candle", "thread", "bridge", "winter", "orchard"]


def _pad(seed) -> str:
    # Non-repetitive: "word " * 60 is a repetition loop to reply_is_degenerate,
    # and a redacted reply would read as lost to this oracle for the wrong reason.
    rr = random.Random(str(seed))
    return " ".join(rr.choice(_W) for _ in range(60)) + "."


def turn(role: str, num: int, ver: int = 1) -> dict:
    r = {"user": "U", "assistant": "A", "tool": "T"}[role]
    return {"role": role, "content": f"{r}{num}v{ver} {_pad((num, ver))}"}


def tag_of(m: dict) -> str:
    return TAG.findall(summarizer._message_text(m))[0]


class Convo:
    def __init__(self, conv: str):
        self.conv = conv
        self.hist: list[dict] = []
        self.n = 0
        self.problems: list[str] = []
        self.rows: list[dict] = []   # one per request
        self.rollups_off = False
        self.streamed: str | None = None

    def next_num(self) -> int:
        self.n += 1
        return self.n

    async def request(self, req_turns: list[dict], reply: dict, label: str):
        req = [SYS] + req_turns
        out_n: list[int] = []
        FRESH_INPUTS.clear()
        had_record = bool(summarizer._covered_fps(summarizer.load_state(self.conv)))
        out = await main.compact_if_needed(list(req), self.conv, stored_turns_out=out_n)
        visible = set()
        for m in out:
            visible.update(TAG.findall(summarizer._message_text(m)))
        missing, stale = [], []
        for m in req_turns:
            t = tag_of(m)
            if t in visible:
                continue
            base = re.match(r"([UAT]\d+)v", t).group(1)
            if any(v.startswith(base + "v") for v in visible):
                stale.append(t)
            else:
                missing.append(t)
        if missing or stale:
            self.problems.append(f"{label}: missing={missing} stale={stale}")
        self.rows.append({
            "label": label, "req": len(req_turns), "compacted": bool(out_n),
            "stored": out_n[0] if out_n else 0, "had_record": had_record,
            "fresh": sum(len(f) for f in FRESH_INPUTS),
        })
        if not self.rollups_off:
            kw = {}
            if self.streamed is not None:
                kw["reply_as_streamed"] = self.streamed
                self.streamed = None
            await main._rollup_hierarchy(
                self.conv, req, summarizer._message_text(reply), **kw)

    async def exchange(self):
        u = turn("user", self.next_num())
        a = turn("assistant", self.next_num())
        await self.request(self.hist + [u], a, f"exchange {tag_of(u)}")
        self.hist += [u, a]

    async def regenerate_last(self):
        old = self.hist[-1]
        num, ver = map(int, re.match(r"A(\d+)v(\d+)", old["content"]).groups())
        new = turn("assistant", num, ver + 1)
        await self.request(self.hist[:-1], new, f"REGENERATE {tag_of(old)}")
        self.hist[-1] = new

    def live(self, after: int = 0):
        """Rows of compacting requests that had a record when they ran."""
        return [r for r in self.rows[after:] if r["compacted"] and r["had_record"]]


def closes_chunk(c: Convo) -> bool:
    return summarizer.load_state(c.conv).get("last_summarized_turn", 0) == len(c.hist)


def assert_healthy(c: Convo, what: str, fresh_cap: int, after: int = 0):
    live = c.live(after)
    check(c.problems == [],
          f"{what}: no turn lost or replaced by a stale version "
          f"({len(c.problems)} problem request(s): {c.problems[:2]})")
    check(bool(live) and all(r["stored"] > 0 for r in live),
          f"{what}: reuse fired on every one of the {len(live)} compacting "
          f"request(s) with a record (dead on "
          f"{sum(1 for r in live if not r['stored'])})")
    worst = max((r["fresh"] for r in live), default=0)
    check(worst <= fresh_cap,
          f"{what}: the fresh span stayed bounded (worst {worst}, cap {fresh_cap})")


_CHUNK = summarizer.L1_CHUNK_SIZE
# One unfilled L1 chunk plus the recent window's spill, the steady-state span.
_STEADY = _CHUNK


async def scenarios():
    print("[1] CONTROL: an append-only conversation reuses on every request")
    c = Convo("tr_baseline")
    for _ in range(40):
        await c.exchange()
    assert_healthy(c, "baseline", _STEADY)

    print("[2] F1: delete the last exchange right after a chunk closes")
    c = Convo("tr_delete_last_pair")
    while True:
        await c.exchange()
        if len(c.hist) >= 40 and closes_chunk(c):
            break
    deleted = [tag_of(m) for m in c.hist[-2:]]
    del c.hist[-2:]
    for _ in range(25):
        await c.exchange()
    replacement = [tag_of(m) for m in c.hist[len(c.hist) - 50:len(c.hist) - 48]]
    check(deleted != replacement, f"fixture: {deleted} deleted, {replacement} took their positions")
    assert_healthy(c, "delete last pair", _STEADY + 2)

    print("[3] F1: regenerate the reply that closed a chunk")
    c = Convo("tr_regen_boundary")
    while True:
        await c.exchange()
        if len(c.hist) >= 40 and closes_chunk(c):
            break
    await c.regenerate_last()
    for _ in range(25):
        await c.exchange()
    assert_healthy(c, "regenerate the chunk-closing reply", _STEADY + 1)

    print("[4] F1: delete only the newest reply after a chunk closes, then send")
    c = Convo("tr_delete_last_reply")
    while True:
        await c.exchange()
        if len(c.hist) >= 40 and closes_chunk(c):
            break
    del c.hist[-1:]
    for _ in range(15):
        await c.exchange()
    assert_healthy(c, "delete the newest reply", _STEADY + 2)

    print("[5] F3: every regenerate request reuses, not only the ones after it")
    c = Convo("tr_regen_mid")
    for i in range(36):
        await c.exchange()
        if i % 7 == 3:
            await c.regenerate_last()
    regen_rows = [r for r in c.live() if r["label"].startswith("REGENERATE")]
    check(len(regen_rows) >= 3 and all(r["stored"] > 0 for r in regen_rows),
          f"*** F3: {len(regen_rows)} regenerate request(s) with a record, reuse "
          f"on all of them")
    assert_healthy(c, "regenerate mid-chunk", _STEADY)

    print("[6] F2: delete ONE older message, and a pair, inside the covered span")
    for label, cut in (("one", slice(10, 11)), ("pair", slice(10, 12))):
        c = Convo(f"tr_delete_{label}")
        for _ in range(30):
            await c.exchange()
        mark = len(c.rows)
        del c.hist[cut]
        for _ in range(25):
            await c.exchange()
        assert_healthy(c, f"delete {label}", _STEADY + 2, after=mark)

    print("[7] F2: edit-and-resend an earlier message (a shorter branch)")
    c = Convo("tr_edit_resend")
    for _ in range(35):
        await c.exchange()
    mark = len(c.rows)
    c.hist = c.hist[:60]
    for _ in range(25):
        await c.exchange()
    assert_healthy(c, "edit and resend", _STEADY, after=mark)

    print("[8] reviewer D F1: an edit 80 turns back, twice, and keep chatting")
    c = Convo("tr_deep_edit")
    for _ in range(55):
        await c.exchange()
    for back in (80, 40):
        cut = len(c.hist) - back
        c.hist = c.hist[:cut - 1] + [turn("user", c.next_num()), turn("assistant", c.next_num())]
        mark = len(c.rows)
        for _ in range(25):
            await c.exchange()
        assert_healthy(c, f"edit {back} turns back", _STEADY + 2, after=mark)

    print("[9] F7: tool-role turns in the history")
    c = Convo("tr_tool_turns")
    for _ in range(3):
        await c.exchange()
    c.hist += [turn("assistant", c.next_num()), turn("tool", c.next_num()),
               turn("tool", c.next_num()), turn("assistant", c.next_num())]
    for _ in range(30):
        await c.exchange()
    assert_healthy(c, "tool turns", _STEADY)

    print("[10] a client that caps its window at 40 turns")
    c = Convo("tr_capped")
    for _ in range(30):
        await c.exchange()
    mark = len(c.rows)
    for _ in range(25):
        u = turn("user", c.next_num())
        a = turn("assistant", c.next_num())
        await c.request((c.hist + [u])[-41:], a, f"capped {tag_of(u)}")
        c.hist += [u, a]
    assert_healthy(c, "capped window", _STEADY, after=mark)

    print("[11] every reply STOPPED: memory keeps a trimmed prefix, the client re-sends all of it")
    c = Convo("tr_stopped")
    for _ in range(40):
        u = turn("user", c.next_num())
        a = turn("assistant", c.next_num())
        full = summarizer._message_text(a)
        c.streamed = full
        await c.request(c.hist + [u], {"role": "assistant", "content": full[:40]},
                        f"stopped {tag_of(u)}")
        c.hist += [u, a]
    assert_healthy(c, "stopped replies", _STEADY)


async def admin_scenario():
    """F9, through the REAL /admin/conversations/<id>/compact route."""
    from unittest.mock import patch

    import retrieval
    from fastapi.testclient import TestClient

    print("[12] F9: rollups fall behind, the operator runs the admin rebuild "
          "(two exchanges never indexed), then chat resumes")
    c = Convo("tr_admin_compact")
    for _ in range(10):
        await c.exchange()
    c.rollups_off = True
    for _ in range(20):
        await c.exchange()
    c.rollups_off = False
    skipped = {13, 17}
    rows = []
    for k in range(len(c.hist) // 2):
        if k in skipped:
            continue
        u, a = c.hist[2 * k], c.hist[2 * k + 1]
        rows.append({"turn_index": 2 + 2 * k,
                     "document": f"[user]: {u['content']}\n[assistant]: {a['content']}"})
    client = TestClient(main.app, client=("127.0.0.1", 12398), raise_server_exceptions=False)
    with patch.object(retrieval, "export_indexed_exchanges", lambda cid: rows):
        r = client.post(f"/admin/conversations/{c.conv}/compact", json={"dry_run": False})
    body = r.json() if r.status_code == 200 else {}
    check(r.status_code == 200 and body.get("gap_exchanges") == 2,
          f"fixture: the rebuild ran with 2 gap exchanges (HTTP {r.status_code}, "
          f"{ {k: body.get(k) for k in ('gap_exchanges', 'rollup_calls')} })")
    st = summarizer.load_state(c.conv)
    check(summarizer._covered_prefix(st) == 60,
          f"fixture: the rebuild's chunks cover 1-60 (got {summarizer._covered_prefix(st)})")
    placeholders = [c.hist[i] for i in (26, 27, 34, 35)]
    fps = summarizer._covered_fps(st)
    check(all(fps[i] != summarizer._covered_turn_fingerprint(m)
              for i, m in zip((26, 27, 34, 35), placeholders)),
          "*** F9: the rebuild recorded what its chunks read — its placeholders — "
          "so turns 27-28 and 35-36 are NOT recorded as the client's turns")
    mark = len(c.rows)
    for _ in range(20):
        await c.exchange()
    check(all(fps[i] == summarizer._covered_fps(summarizer.load_state(c.conv))[i]
              for i in (26, 27, 34, 35)),
          "and no later live rollup rewrote those entries")
    assert_healthy(c, "after the admin rebuild", _STEADY + 4, after=mark)


try:
    asyncio.run(scenarios())
    logging.getLogger("compactor").setLevel(logging.WARNING)
    asyncio.run(admin_scenario())
finally:
    main.summarize = _real_summarize
    summarizer._summarize_pieces = _real_pieces

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll reuse-traffic checks passed.")
