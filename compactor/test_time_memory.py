"""The time line never enters memory, and compaction still reuses the hierarchy.

v3.1.9 (V4_ROADMAP.md section 1.1 item 1; traps (a) and (h) of the brief).

The line is in the FORWARDED payload only. OpenWebUI never sees it and never
re-sends it, so every later request carries her turns as she typed them. If
any memory writer read the forwarded payload instead of the request, three
things would break at once:

  * facts and episodic documents would store "[Current date and time: ...]"
    as part of what she said — a stale clock, read back as memory;
  * summaries would describe a timestamp as if it were conversation;
  * the covered-turn record (summarizer._record_chunk_fps) would fingerprint a
    user turn that no later request contains, so the reuse gate would read it
    as EDITED and re-summarize it on every request — reuse switched off, the
    2026-09-11 117-second compaction back.

So this drives a conversation through the REAL route with the REAL memory tail
run to completion after every turn (facts extraction, episodic indexing, the
hierarchy rollup and the record writer), doubling only the LLM seams, and then:

  [1] every forwarded chat payload carried exactly one line, in its newest user
      message (the CONTROL that makes every absence below mean something)
  [2] what the fact extractor, the episodic indexer and the summarizer READ
      contains no line — and does contain her text
  [3] nothing on disk under the storage root contains the line
  [4] the covered-turn record is the fingerprints of the turns as the client
      holds them
  [5] compaction REUSED the hierarchy on every compacting turn once a chunk
      existed, and on at least a handful of them
  [6] a request whose compaction makes real summarization calls: those calls
      carry no line, the chat call carries exactly one

    python test_time_memory.py
"""

import asyncio
import datetime as dt
import random
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "true"
_ROOT = tempfile.mkdtemp(prefix="time-memory-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _ROOT
os.environ["COMPACTOR_TARGET_TOKENS"] = "1000"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "64"
os.environ.pop("COMPACTOR_TIMEZONE", None)
os.environ.pop("TZ", None)
os.environ.pop("COMPACTOR_TIME_INJECTION", None)

import memory  # noqa: E402

memory.ensure_storage_layout()

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import facts  # noqa: E402
import main  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []
NEEDLE = "Current date and time"
CONV = "time-memory-conv"


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


check(getattr(main, "TIME_INJECTION_ENABLED", None) is True,
      "fixture: time injection is on by default")
check(summarizer.enabled(), "fixture: the hierarchy is enabled")

# ---- the LLM seams, doubled; everything else is real ------------------------
_extract_read: list[tuple[str, str]] = []
_index_read: list[tuple[str, str]] = []
_pieces_read: list[str] = []


async def _extract(client, url, model, user_text, assistant_text, existing, conv_id=None):
    _extract_read.append((user_text, assistant_text))
    return [f"The user asked about {user_text.split('.')[0][-20:]}"]


def _index(conv_id, turn_index, user_text, assistant_text):
    _index_read.append((user_text, assistant_text))
    return True


async def _pieces(conv_id, client, vllm_url, model, prompt, pieces, max_tokens):
    _pieces_read.extend(pieces)
    # Echo the HEAD of what it read, so a leaked line (which would be at the
    # head of a user turn) reaches the stored summary too. Short: the stand-in
    # must fit whole beside the recent turns at this file's TARGET, or
    # compaction declines reuse for a reason that has nothing to do with time.
    return "SCENE " + " | ".join(p[:48] for p in pieces[:4])


async def _fresh(client, turns):
    return "FRESH-SUMMARY", []


facts.extract_facts_from_exchange = _extract
retrieval.index_exchange = _index
summarizer._summarize_pieces = _pieces


_VOCAB = ("the lantern harbor window kept quiet silver light through winter morning "
          "garden bench painted blue before rain folder letters receipts office opens "
          "nine form signed month second letter names date change began stamp page "
          "river stones path north gate old clock tower bell rings noon market bread "
          "warm kitchen table maps drawer candle shelf").split()


def prose(seed, words=60):
    """Varied synthetic prose. A repeated word is a repetition loop to
    reply_is_degenerate, which would refuse every reply and leave nothing for
    the memory writers this file is about to read."""
    rr = random.Random(str(seed))
    out = []
    while len(out) < words:
        sentence = [rr.choice(_VOCAB) for _ in range(rr.randint(7, 12))]
        out.extend(sentence[:-1] + [sentence[-1] + "."])
    return " ".join(out)


class _Backend:
    bodies: list[dict] = []
    n = 0

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        _Backend.bodies.append(json)
        _Backend.n += 1
        text = f"Item reply {_Backend.n}. " + prose(("reply", _Backend.n))
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": text}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", url))

    async def aclose(self):
        pass


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, r):
        self.lines.append(r.getMessage())


_pending: list = []


def _defer(coro, label=None):
    _pending.append(coro)
    return True


client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)


def turn(history, at, summarize_impl=_fresh):
    _Backend.bodies = []
    cap = _Lines()
    lg = logging.getLogger("compactor")
    prev = lg.level
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    try:
        with patch.object(main.httpx, "AsyncClient", _Backend), \
             patch.object(main, "summarize", summarize_impl), \
             patch.object(main, "_now_utc", lambda: at, create=True), \
             patch.object(main, "_fire_and_forget", _defer):
            r = client.post("/v1/chat/completions",
                            json={"model": "test-model", "messages": history,
                                  "stream": False},
                            headers={"X-Conversation-Id": CONV})
    finally:
        lg.removeHandler(cap)
        lg.setLevel(prev)
    while _pending:
        asyncio.run(_pending.pop(0))
    reply = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""
    return r.status_code, list(_Backend.bodies), reply, "\n".join(cap.lines)


# ===========================================================================
history = [{"role": "system", "content": "You are a patient assistant."}]
rows = []
T = dt.datetime(2026, 9, 14, 8, 0, tzinfo=dt.timezone.utc)
TURNS = 34
for n in range(1, TURNS + 1):
    history.append({"role": "user", "content": f"Tell me about item {n}. " + prose(("user", n))})
    wm_before = summarizer.load_state(CONV).get("last_summarized_turn", 0)
    status, bodies, reply, log = turn(json.loads(json.dumps(history)),
                                      T + dt.timedelta(minutes=7 * n))
    chat = bodies[-1]["messages"] if bodies else []
    newest = next((m for m in reversed(chat) if m.get("role") == "user"), {})
    rows.append({
        "n": n, "status": status, "wm_before": wm_before,
        "line_total": json.dumps(chat).count(NEEDLE),
        "line_in_newest": str(newest.get("content", "")).count(NEEDLE),
        "newest_ends": str(newest.get("content", "")).endswith(history[-1]["content"]),
        "compacted": "compacted:" in log,
        "reused": "covered by stored summaries" in log,
    })
    history.append({"role": "assistant", "content": reply})

print("[1] every forwarded chat payload was dated, once, in the newest user turn")
bad = [r for r in rows if r["status"] != 200 or r["line_total"] != 1
       or r["line_in_newest"] != 1 or not r["newest_ends"]]
check(not bad, f"all {TURNS} turns: HTTP 200, one line, in her newest message, her text intact "
               f"({bad[:3]})")

print("[2] what memory's writers read")
check(len(_extract_read) >= TURNS - 2 and len(_index_read) >= TURNS - 2,
      f"fixture: the fact extractor ran {len(_extract_read)}x and the indexer "
      f"{len(_index_read)}x")
check(all(NEEDLE not in u and NEEDLE not in a for u, a in _extract_read),
      "fact extraction never read the line")
check(all(u.startswith("Tell me about item") for u, _ in _extract_read),
      "CONTROL: fact extraction read her text as she typed it")
check(all(NEEDLE not in u and NEEDLE not in a for u, a in _index_read)
      and all(u.startswith("Tell me about item") for u, _ in _index_read),
      "the episodic indexer read her text, and never the line")
check(len(_pieces_read) >= 20 and any("Tell me about item" in p for p in _pieces_read),
      f"fixture: the hierarchy summarized {len(_pieces_read)} piece(s) of her conversation")
check(all(NEEDLE not in p for p in _pieces_read), "no summary read the line")

print("[3] nothing on disk carries the line")
hits, scanned = [], 0
for path in Path(_ROOT).rglob("*"):
    if path.is_file():
        scanned += 1
        if NEEDLE.encode() in path.read_bytes():
            hits.append(str(path))
st = summarizer.load_state(CONV)
stored_facts = facts.load_facts(CONV)
check(scanned >= 2 and st.get("l1") and stored_facts,
      f"fixture: {scanned} file(s) scanned, {len(st.get('l1') or [])} L1 chunk(s), "
      f"{len(stored_facts)} fact(s) on disk")
check(any("Tell me about item" in c.get("text", "") for c in st.get("l1") or []),
      "CONTROL: a stored summary does carry her text, so a leaked line would be on disk")
check(not hits, f"no stored file contains {NEEDLE!r} ({hits})")

print("[4] the covered-turn record describes the turns the client holds")
record = summarizer._covered_fps(st)
client_fps = summarizer._covered_turn_fingerprints(history)
check(len(record) >= summarizer.L1_CHUNK_SIZE,
      f"fixture: the record covers {len(record)} position(s)")
check(record == client_fps[:len(record)],
      "every recorded fingerprint equals the fingerprint of that turn as OpenWebUI re-sends it")

print("[5] compaction reused the hierarchy")
eligible = [r for r in rows if r["compacted"] and r["wm_before"] >= summarizer.L1_CHUNK_SIZE]
not_reused = [r["n"] for r in eligible if not r["reused"]]
check(len(eligible) >= 5, f"fixture: {len(eligible)} compacting turn(s) had a chunk to reuse")
check(not not_reused,
      f"every one of them reused the stored summaries (not reused on turns {not_reused})")

print("[6] summarization calls on the request path are not dated")


_REAL_SUMMARIZE = main.summarize
history.append({"role": "user", "content": f"Tell me about item {TURNS + 1}. " + prose("last")})
# A fresh id with the whole history: nothing stored, so compaction must call
# the backend to summarize.
CONV_SAVE, CONV = CONV, "time-memory-fresh"
status, bodies, reply, log = turn(json.loads(json.dumps(history)),
                                  T + dt.timedelta(hours=9), summarize_impl=_REAL_SUMMARIZE)
CONV = CONV_SAVE
sums = bodies[:-1]
check(status == 200 and len(sums) >= 1,
      f"fixture: {len(sums)} summarization call(s) reached the backend before the reply")
check(all(json.dumps(b).count(NEEDLE) == 0 for b in sums),
      "none of the summarization calls carries the line")
check(bodies and json.dumps(bodies[-1]["messages"]).count(NEEDLE) == 1,
      "CONTROL: the chat call that followed them carries exactly one")


print()
if FAILED:
    print(f"FAILED: {len(FAILED)} check(s)")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("All time-memory checks passed.")
