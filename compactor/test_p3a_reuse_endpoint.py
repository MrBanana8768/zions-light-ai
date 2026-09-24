"""The summary hierarchy reaches the model exactly ONCE, whatever compaction did.

hostile pass #3: reviewer A F4 + F10, reviewer E F1, and A F5 on the route.

chat_completions injects its own copy of the L1/L2/L3 hierarchy UNLESS
compact_if_needed says (stored_turns_out > 0) that the array it returned
already carries one (hostile2-reuse M1). Two defects lived in that decision
and no test could see either, because every test pinned the out-param's
VALUE and none the payload that reached the backend:

  * F10: replace the condition with `if True` (the hierarchy never injected,
    on any request) or `if False` (sent twice on every reusing turn) — the
    whole unit suite stayed green in both directions.
  * F4 / E-F1: the out-param was written before summarize() ran. A
    summarization call that raised left it saying N > 0; chat_completions
    threw the compacted array away, forwarded the original messages, and
    skipped its own copy. Reproduced: 56 turns shed with nothing standing in
    for them.

So every case here drives the REAL /v1/chat/completions route (TestClient),
with the hierarchy written by the REAL maybe_rollup and only the LLM seams
doubled, captures the payload that reached the backend, and counts how many
copies of the hierarchy it holds:

  [1] under TARGET, nothing compacted      -> injected, once
  [2] compaction reuses the hierarchy      -> in the array, once
  [3] compaction runs but reuses nothing   -> injected, once
  [4] summarize() raises mid-compaction    -> injected, once (F4)
  [5] the F5 shape at the shipped numbers: a stand-in near capacity beside
      ~10k tokens of injected memory keeps her last two turns and the
      stand-in whole (the guard spends injected memory first)

    python test_p3a_reuse_endpoint.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="p3a-endpoint-")
os.environ["COMPACTOR_TARGET_TOKENS"] = "1000"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "64"

import memory  # noqa: E402

memory.ensure_storage_layout()

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


MARK = "HIERMARK"


async def _stub_pieces(conv_id, client, vllm_url, model, prompt, pieces, max_tokens):
    return f"{MARK} scene over {len(pieces)} piece(s)"


summarizer._summarize_pieces = _stub_pieces


def history(n, tag="", fat=True):
    pad = (" " + "word " * 60) if fat else ""
    out = [{"role": "system", "content": "you are a companion"}]
    for i in range(n):
        out.append({"role": "user", "content": f"{tag}question {i}{pad}"})
        out.append({"role": "assistant", "content": f"{tag}answer {i}{pad}"})
    return out


def seed(conv, msgs):
    st = asyncio.run(summarizer.maybe_rollup(conv, msgs, "http://stub", "m",
                                             raw_messages=msgs))
    return len(st["l1"]) + len(st["l2"]) + (1 if st.get("l3") else 0)


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, r):
        self.lines.append(r.getMessage())


async def _ok_summarize(client, turns):
    return "FRESH-SUMMARY", []


async def _raising_summarize(client, turns):
    req = httpx.Request("POST", "http://stub:8000/v1/chat/completions")
    raise httpx.HTTPStatusError("400 Bad Request", request=req,
                                response=httpx.Response(400, request=req))


client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)


def post(conv, msgs, summarize_impl=_ok_summarize):
    forwarded: list[dict] = []

    async def fake_post(self, url, **kw):
        if url.endswith("/v1/chat/completions"):
            forwarded.append(kw.get("json") or {})
            return httpx.Response(200, json={"choices": [{"index": 0, "message": {
                "role": "assistant", "content": "a reply"}, "finish_reason": "stop"}]},
                request=httpx.Request("POST", url))
        raise httpx.ConnectError("no network in this test",
                                 request=httpx.Request("POST", url))

    cap = _Lines()
    lg = logging.getLogger("compactor")
    prev = lg.level
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    try:
        with patch.object(main, "summarize", summarize_impl), \
             patch.object(main, "_fire_and_forget",
                          lambda coro, label=None: (coro.close(), True)[1]), \
             patch.object(main.httpx.AsyncClient, "post", fake_post):
            r = client.post("/v1/chat/completions",
                            json={"model": "test-model", "messages": msgs, "stream": False},
                            headers={"X-Conversation-Id": conv})
    finally:
        lg.removeHandler(cap)
        lg.setLevel(prev)
    payload = forwarded[-1].get("messages", []) if forwarded else []
    blob = json.dumps(payload)
    return r.status_code, payload, blob, cap.lines


def copies(blob, n_chunks):
    return blob.count(MARK) / max(1, n_chunks)


print("[1] under TARGET: nothing compacted, the hierarchy is injected once")
CONV1 = "ep_under_target"
SHORT = history(24, fat=False)
n1 = seed(CONV1, SHORT)
status, payload, blob, lines = post(CONV1, SHORT + [{"role": "user", "content": "question 24"}])
check(status == 200 and n1 >= 2, f"fixture: HTTP {status}, {n1} stored chunk(s)")
check(main.COMPACTION_SUMMARY_HEADER not in blob, "fixture: compaction did not run")
check(copies(blob, n1) == 1,
      f"*** F10: the hierarchy reached the backend exactly once (copies={copies(blob, n1)})")

print("[2] compaction REUSES the hierarchy: it travels in the array, once")
CONV2 = "ep_reusing"
FAT = history(24)
n2 = seed(CONV2, FAT)
REQ2 = FAT + [{"role": "user", "content": "question 24 " + "word " * 60}]
status, payload, blob, lines = post(CONV2, REQ2)
check(status == 200 and any("covered by stored summaries" in l for l in lines),
      f"fixture: HTTP {status}, and compaction reused the stored summaries")
check(copies(blob, n2) == 1,
      f"*** F10: exactly one copy — not injected a second time (copies={copies(blob, n2)})")
check(main.COMPACTION_SUMMARY_HEADER in blob and "FRESH-SUMMARY" in blob,
      "and that copy is compaction's own stand-in")

print("[3] compaction runs but reuses NOTHING: the hierarchy is injected once")
ALIEN = history(24, tag="Z-") + [{"role": "user", "content": "Z-question 24 " + "word " * 60}]
status, payload, blob, lines = post(CONV2, ALIEN)
check(status == 200 and not any("covered by stored summaries" in l for l in lines)
      and "FRESH-SUMMARY" in blob,
      f"fixture: HTTP {status}, compacted without reusing")
check(copies(blob, n2) == 1,
      f"*** F10: the injected copy is still sent (copies={copies(blob, n2)})")

print("[4] summarize() RAISES after the gate chose to reuse: injected, once")
status, payload, blob, lines = post(CONV2, REQ2, _raising_summarize)
check(status == 200 and any("compaction failed" in l for l in lines),
      f"fixture: HTTP {status}, and compaction failed and fell through")
check(copies(blob, n2) == 1,
      f"*** F4/E-F1: the hierarchy still reaches the backend (copies={copies(blob, n2)}) "
      f"— an out-param written before summarize() ran used to skip it")


print("[5] F5 at the shipped numbers: the stand-in cannot cost her last two turns")
# reviewer A's guard_prod.py shape, reduced: MAX_MODEL_LEN 32768, reserve
# 12000 -> limit 20,768, TARGET 15,576, KEEP_RECENT 4, SUMMARY_MAX 1024; a
# hierarchy near its documented capacity (L3 + 4 L2 + 9 L1); ~1,650-token
# replies; ~10k tokens of injected memory. The REAL compact_if_needed,
# inject_system_block and _enforce_hard_budget, with the char/4 counter (no
# tokenizer is staged; the shed ORDER is what is asserted, not the counts).
# Run in a subprocess, because the numbers are import-time constants.
import subprocess  # noqa: E402

_CHILD = r'''
import asyncio, os, random, sys, tempfile
os.environ.update({"MODEL_REPO": "test-model", "VLLM_URL": "http://stub:8000",
    "COMPACTOR_RAG_ENABLED": "false", "COMPACTOR_STORAGE_ROOT": tempfile.mkdtemp(prefix="p3a-f5-"),
    "MAX_MODEL_LEN": "32768", "COMPACTOR_GENERATION_RESERVE": "12000",
    "COMPACTOR_KEEP_RECENT_TURNS": "4", "COMPACTOR_SUMMARY_MAX_TOKENS": "1024"})
os.environ.pop("COMPACTOR_TARGET_TOKENS", None)
import memory; memory.ensure_storage_layout()
import main, summarizer
main.count_tokens_exact = lambda ms: None
main.get_tokenizer = lambda: None
W = "the lantern by the harbor window kept its quiet silver light through winter morning".split()
def text(seed, chars):
    rr = random.Random(str(seed)); out = []; n = 0
    while n < chars:
        w = rr.choice(W); out.append(w); n += len(w) + 1
    return " ".join(out)
TURNS = 420
conv = [{"role": "system", "content": "PERSONA " + text("persona", 2000)}]
for i in range(1, TURNS + 1):
    conv.append({"role": "user" if i % 2 else "assistant",
                 "content": (f"U{i} " + text(("u", i), 400)) if i % 2 else (f"A{i} " + text(("a", i), 6600))})
req = conv + [{"role": "user", "content": f"U{TURNS + 1} " + text("new", 400)}]
covered_to = 400
st = summarizer.load_state("prod")
st["l3"] = {"text": "L3OVERVIEW " + text("l3", 8000), "first_turn": 1, "last_turn": 20}
st["l2"] = [{"text": f"L2CHAPTER{j} " + text(("l2", j), 4800), "first_turn": 21 + 50 * j, "last_turn": 70 + 50 * j} for j in range(4)]
st["l1"] = [{"text": f"L1SCENE{j} " + text(("l1", j), 2000), "first_turn": 221 + 20 * j, "last_turn": 240 + 20 * j} for j in range(9)]
st["last_summarized_turn"] = covered_to
ns = [m for m in conv if m["role"] != "system"]
summarizer._record_chunk_fps(st, 1, covered_to, ns[:covered_to])
summarizer.save_state("prod", st)
async def fresh(client, turns):
    return "FRESH " + text("fresh", 4000), []
main.summarize = fresh
out_n = []
out = asyncio.run(main.compact_if_needed(list(req), "prod", stored_turns_out=out_n))
limit = min(main.MAX_MODEL_LEN, max(256, main.MAX_MODEL_LEN - main.GENERATION_RESERVE))
msgs = main.inject_system_block(out, "[FACTS+RETRIEVAL] " + text("inj", int(sys.argv[1])))
report = {}
g = main._enforce_hard_budget(msgs, limit, 1, report)
standin_before = next(m["content"] for m in msgs if str(m.get("content")).startswith(main.COMPACTION_SUMMARY_HEADER))
standin_after = next((m["content"] for m in g if str(m.get("content")).startswith(main.COMPACTION_SUMMARY_HEADER)), "")
kept = [m["content"].split()[0] for m in g if m.get("role") != "system"]
print("RESULT", out_n, report.get("dropped_turns"), report.get("fits"),
      standin_after == standin_before, ",".join(kept),
      "L1SCENE8" in standin_after and "FRESH" in standin_after)
'''

for inject_chars in ("41000", "24000"):
    p = subprocess.run([sys.executable, "-c", _CHILD, inject_chars],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=os.path.dirname(os.path.abspath(__file__)), timeout=600,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    res = next((ln.split()[1:] for ln in p.stdout.splitlines() if ln.startswith("RESULT")), None)
    if res is None:
        check(False, f"the F5 child ran (rc {p.returncode}): {p.stderr[-600:]}")
        continue
    reused, dropped, fits, whole, kept, scenes = res[0], res[1], res[2], res[3], res[4], res[5]
    check(reused != "[0]" and fits == "True",
          f"fixture ({inject_chars} chars injected): compaction reused ({reused}) and "
          f"the guard fit the payload")
    check(dropped == "0" and kept.startswith("U419,A420,U421"),
          f"*** F5: her previous message and the reply she is answering survive "
          f"(dropped {dropped} turn(s), kept {kept})")
    check(whole == "True" and scenes == "True",
          "*** F5: and the stand-in is byte-identical — its newest scenes and the "
          "fresh summary were not halved away")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll reuse-endpoint checks passed.")
