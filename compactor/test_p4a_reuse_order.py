"""Compaction reuse: a repeated message is not replaced by its twin, and the
refreshed span does not grow.

hostile pass #4, reviewer A:

  F4. The gate paired a request turn with the record by SET MEMBERSHIP: any
      turn whose content any chunk had read was replaced, wherever that twin
      sat. Her "yes" to a new question, or a paste sent twice, was replaced
      by the summary of the earlier one in another context — for good when
      it landed where no chunk would ever read it (after a delete of the
      exchange that closed a chunk).
  F6. A turn inside the covered span that pairs with nothing (the turns
      after a delete of a chunk-closing exchange, a regenerated closing
      reply, an edit) was summarized fresh on EVERY request and nothing ever
      re-read it: +2 per delete, +1 per regenerate, no decay across an L2
      rollup. Projected at her edit rate, most messages paying three
      summarization calls within about two months and the four-call refusal
      within four to six.

Driven like test_p3a_reuse_traffic.py: the REAL compact_if_needed and the
REAL _rollup_hierarchy per exchange, with the two LLM seams stubbed to echo
the tags of their input. Two oracles:

  * TAGS: every tag of every request turn is visible in the returned array
    (verbatim, in the fresh summary, or in a stored chunk). Catches a lost
    or stale turn.
  * PROBES: a probe turn carries no tag (a bare "yes" is indistinguishable
    from its twin). Each request, the probe OBJECT must be forwarded, handed
    to summarize(), or replaced LEGITIMATELY: a chunk written after the
    probe first appeared read a piece whose text is the probe's text. A
    replacement by a twin some chunk read before the probe existed is F4.

Synthetic text only.

    python test_p4a_reuse_order.py
"""

import asyncio
import os
import random
import re
import sys
import tempfile
import time

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="p4a-order-")
os.environ["COMPACTOR_TARGET_TOKENS"] = "1500"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "64"

import logging  # noqa: E402

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

logging.getLogger("compactor").setLevel(logging.WARNING)

FAILED: list[str] = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


TAG = re.compile(r"\b[UATB]\d+v\d+\b")
FRESH_INPUTS: list[list[dict]] = []
# (request number when the chunk was written, its pieces)
CHUNK_READS: list[tuple[int, list[str]]] = []
REQ_NO = [0]


async def _spy_summarize(client, turns):
    FRESH_INPUTS.append(list(turns))
    tags = []
    for m in turns:
        tags += TAG.findall(summarizer._message_text(m))
    return "FRESH{" + ",".join(tags) + "}", []


async def _stub_pieces(conv_id, client, vllm_url, model, prompt, pieces, max_tokens):
    CHUNK_READS.append((REQ_NO[0], list(pieces)))
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


def _pad(seed, n=60) -> str:
    rr = random.Random(str(seed))
    return " ".join(rr.choice(_W) for _ in range(n)) + "."


def turn(role: str, num: int, ver: int = 1) -> dict:
    r = {"user": "U", "assistant": "A"}[role]
    return {"role": role, "content": f"{r}{num}v{ver} {_pad((num, ver))}"}


def txt(m: dict) -> str:
    return summarizer._message_text(m)


def _piece_text(p: str) -> str:
    """The turn text inside one rollup piece ("[user]: ..." or the extra
    piece form "[user] (an earlier turn, as it reads now): ...")."""
    return p.split(": ", 1)[1] if ": " in p else p


class Convo:
    def __init__(self, conv: str):
        self.conv = conv
        self.hist: list[dict] = []
        self.n = 0
        self.problems: list[str] = []
        self.rows: list[dict] = []
        self.probes: dict[str, tuple[dict, int]] = {}   # name -> (object, first request)
        self.legit_hits = 0

    def next_num(self) -> int:
        self.n += 1
        return self.n

    def add_probe(self, name: str, obj: dict):
        self.probes[name] = (obj, REQ_NO[0] + 1)

    async def request(self, req_turns: list[dict], reply_text: str, label: str):
        REQ_NO[0] += 1
        req = [SYS] + req_turns
        out_n: list[int] = []
        FRESH_INPUTS.clear()
        st0 = summarizer.load_state(self.conv)
        had_record = bool(summarizer._covered_fps(st0))
        _, _ts, _ = main.split_messages(list(req))
        _cov, _chg = summarizer._coverage_plan(st0, _ts)
        out = await main.compact_if_needed(list(req), self.conv, stored_turns_out=out_n)
        visible = set()
        for m in out:
            visible.update(TAG.findall(txt(m)))
        missing, stale = [], []
        for m in req_turns:
            for t in TAG.findall(txt(m)):
                if t in visible:
                    continue
                base = re.match(r"([UATB]\d+)v", t).group(1)
                (stale if any(v.startswith(base + "v") for v in visible) else missing).append(t)
        fresh_ids = {id(m) for f in FRESH_INPUTS for m in f}
        out_ids = {id(m) for m in out}
        bad_probe = []
        for name, (obj, since) in self.probes.items():
            if not any(obj is m for m in req_turns):
                continue
            if id(obj) in fresh_ids or id(obj) in out_ids:
                continue
            legit = any(
                req_no >= since and any(_piece_text(p) == txt(obj) for p in pieces)
                for req_no, pieces in CHUNK_READS
            )
            if legit:
                self.legit_hits += 1
            else:
                bad_probe.append(name)
        if missing or stale or bad_probe:
            self.problems.append(f"{label}: missing={missing} stale={stale} twin-replaced={bad_probe}")
        self.rows.append({
            "label": label, "compacted": bool(out_n), "stored": out_n[0] if out_n else 0,
            "had_record": had_record, "fresh": sum(len(f) for f in FRESH_INPUTS),
            "prefix": summarizer._covered_prefix(st0), "refreshed": len(_chg),
        })
        await main._rollup_hierarchy(self.conv, req, reply_text)

    async def exchange(self, user: dict | None = None, reply: dict | None = None, label=None):
        u = user or turn("user", self.next_num())
        a = reply or turn("assistant", self.next_num())
        await self.request(self.hist + [u], txt(a), label or f"exchange {txt(u)[:6]}")
        self.hist += [u, a]
        return u, a

    def closed_chunk(self) -> bool:
        return summarizer.load_state(self.conv).get("last_summarized_turn", 0) == len(self.hist)


async def order_units():
    print("[1] _pairing: order-preserving, each record entry used at most once")
    fp = lambda s: summarizer._covered_turn_fingerprint({"role": "user", "content": s})  # noqa: E731
    a, b, c, d, yes = fp("a"), fp("b"), fp("c"), fp("d"), fp("yes")
    P = summarizer._pairing
    check(yes in set([a, yes, b, c, d]) and 4 not in P([a, yes, b, c, d], [a, b, c, d, yes]),
          "*** F4: a 'yes' whose only twin sits BEFORE the turns around it does not pair "
          "(membership would pair it)")
    check(P([a, yes, b, c, d], [a, b, c, d, yes]) == {0: 0, 1: 2, 2: 3, 3: 4},
          "CONTROL: every other turn still pairs, in order")
    check(P([a, yes, b], [a, yes, b]) == {0: 0, 1: 1, 2: 2},
          "CONTROL: a repeated message at its own place pairs")
    x = fp("x")
    got = P([a, x, b], [a, x, x, b])
    check(sorted(got.values()) == [0, 1, 2] and sum(1 for j in (1, 2) if j in got) == 1,
          f"a record entry vouches for at most ONE request turn (got {got})")
    cont = fp("continue")
    ans = [fp(f"answer {i}") for i in range(6)]
    rec = [v for i in range(6) for v in (cont, ans[i])]
    now = rec[2:]                                   # the first exchange deleted
    check(len(P(rec, now)) == len(now),
          "CONTROL: 'continue' between unique replies still pairs after a delete (the "
          "SequenceMatcher junk case that made the first cut refresh them forever)")
    big_rec = [f"{i:016x}" for i in range(2000)]
    big_now = [("f" * 16 if i % 2 else r) for i, r in enumerate(big_rec[1:])]
    t0 = time.perf_counter()
    P(big_rec + ["f" * 16], big_now)
    same = ["e" * 16] * 2000
    got_same = P(same, same[1:])
    dt = time.perf_counter() - t0
    check(len(got_same) == 1999, "2,000 identical turns with one deleted: every one pairs")
    check(dt < 0.5, f"pairing 2,000 turns (half one fingerprint) and 2,000 identical ones: {dt * 1000:.1f} ms")


async def f4_traffic():
    print("[2] F4: delete the exchange that closed a chunk, then answer 'yes' (an old 'yes' exists)")
    c = Convo("ord_delete_then_yes")
    for i in range(30):
        if i == 5:
            c.next_num()
            await c.exchange(user={"role": "user", "content": "yes"}, label="old yes")
        else:
            await c.exchange()
    while not c.closed_chunk():
        await c.exchange()
    del c.hist[-2:]
    yes_new = {"role": "user", "content": "yes"}
    c.add_probe("yes@new", yes_new)
    c.next_num()
    await c.exchange(user=yes_new, label="new yes")
    for _ in range(30):
        await c.exchange()
    check(c.problems == [], f"no turn lost, stale, or replaced by its twin ({c.problems[:2]})")
    check(c.legit_hits > 0, f"liveness: once a chunk re-read the new 'yes' it came off the "
                            f"shelf ({c.legit_hits} request(s))")
    _since = c.probes["yes@new"][1]
    _first = next((ps for rq, ps in CHUNK_READS if rq >= _since), [])
    check(any(_piece_text(p) == "yes" for p in _first),
          "*** F6: the FIRST L1 chunk written after it re-read it (not a later cycle): "
          "the turns after a deleted closing exchange sit past the last paired turn")
    live = [r for r in c.rows if r["compacted"] and r["had_record"]]
    check(all(r["stored"] > 0 for r in live), "reuse fired on every compacting request")

    print("[3] F4: a paste sent twice, twenty exchanges apart")
    c = Convo("ord_paste_twice")
    paste = "Here is the letter I wrote, please read it again. " + _pad("letter", 120)
    for i in range(45):
        if i == 8:
            c.next_num()
            await c.exchange(user={"role": "user", "content": paste})
        elif i == 30:
            p2 = {"role": "user", "content": paste}
            c.add_probe("paste@second", p2)
            c.next_num()
            await c.exchange(user=p2)
        else:
            await c.exchange()
    check(c.problems == [], f"the second paste is never replaced by the first ({c.problems[:2]})")

    print("[4] F4: 'yes' to a new question while an old 'yes' is covered")
    c = Convo("ord_repeat_answer")
    for i in range(40):
        if i == 4:
            c.next_num()
            await c.exchange(user={"role": "user", "content": "yes"})
        elif i == 26:
            y = {"role": "user", "content": "yes"}
            c.add_probe("yes@new", y)
            c.next_num()
            await c.exchange(user=y)
        else:
            await c.exchange()
    check(c.problems == [], f"the new 'yes' is forwarded or summarized until its own chunk "
                            f"reads it ({c.problems[:2]})")


async def f6_growth():
    print("[5] F6: a sustained delete/regenerate/edit rate does not grow the refreshed span")
    # An event right after EVERY L1 rollup for 180 exchanges (her measured
    # rate is about one boundary event in ten chunk closes; this is ten
    # times that), cycling through every shape that leaves an unpaired turn
    # inside the covered span.
    c = Convo("ord_growth")
    events: list[str] = []
    floors: list[int] = []
    kinds = ["delete-last", "regenerate-closing", "delete-reply-then-send",
             "edit-in-place", "delete-old-one", "edit-resend"]
    prev_prefix = 0
    for i in range(180):
        before = summarizer.load_state(c.conv).get("last_summarized_turn", 0)
        await c.exchange()
        row = c.rows[-1]
        if row["prefix"] != prev_prefix:
            floors.append(row["fresh"])
            prev_prefix = row["prefix"]
        rolled = summarizer.load_state(c.conv).get("last_summarized_turn", 0) != before
        if i > 12 and rolled:
            kind = kinds[len(events) % len(kinds)]
            events.append(kind)
            if kind == "delete-last":
                del c.hist[-2:]
            elif kind == "regenerate-closing":
                old = c.hist[-1]
                num, ver = map(int, re.match(r"A(\d+)v(\d+)", txt(old)).groups())
                new = turn("assistant", num, ver + 1)
                await c.request(c.hist[:-1], txt(new), "regenerate")
                c.hist[-1] = new
            elif kind == "delete-reply-then-send":
                del c.hist[-1:]
            elif kind == "edit-in-place":
                k = len(c.hist) - 30
                num, ver = map(int, re.match(r"[UA](\d+)v(\d+)", txt(c.hist[k])).groups())
                c.hist[k] = turn(c.hist[k]["role"], num, ver + 1)
            elif kind == "delete-old-one":
                del c.hist[len(c.hist) - 24]
                # keep alternation: drop its partner too
                del c.hist[len(c.hist) - 24]
            elif kind == "edit-resend":
                c.hist = c.hist[:len(c.hist) - 6]
    third = max(1, len(floors) // 3)
    refreshed = [r["refreshed"] for r in c.rows]
    rthird = max(1, len(refreshed) // 3)
    print(f"       events: {len(events)}; fresh span on the request after each L1 rollup: {floors}")
    print(f"       refreshed turns per request, max by thirds: "
          f"{max(refreshed[:rthird])} / {max(refreshed[rthird:-rthird])} / {max(refreshed[-rthird:])}")
    check(len(events) >= 12, f"fixture: {len(events)} events")
    check(c.problems == [], f"nothing lost or stale ({len(c.problems)}: {c.problems[:2]})")
    live = [r for r in c.rows if r["compacted"] and r["had_record"]]
    check(bool(live) and all(r["stored"] > 0 for r in live),
          f"reuse fired on every compacting request with a record "
          f"(dead on {sum(1 for r in live if not r['stored'])})")
    check(max(floors[-third:]) <= 4 and max(floors[-third:]) <= max(floors[:third]) + 2,
          f"*** F6: the floor does not grow (first third max {max(floors[:third])}, last third "
          f"max {max(floors[-third:])})")
    check(max(refreshed[-rthird:]) <= max(refreshed[:rthird]) + 2 and max(refreshed) <= 8,
          f"*** F6: the refreshed span per request stays bounded and does not trend up "
          f"(max {max(refreshed)}; by thirds {max(refreshed[:rthird])}/{max(refreshed[-rthird:])})")
    st = summarizer.load_state(c.conv)
    check(len(st.get("l2") or []) >= 1 or st.get("l3"),
          "fixture: the run crossed an L2 rollup, which used to carry the span forward")


try:
    asyncio.run(order_units())
    asyncio.run(f4_traffic())
    asyncio.run(f6_growth())
finally:
    main.summarize = _real_summarize
    summarizer._summarize_pieces = _real_pieces

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll reuse order/growth checks passed.")
