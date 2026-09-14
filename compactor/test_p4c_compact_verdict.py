"""
Hostile pass 4 (reviewer C): F5, F6, F8(a) -- /compact's max_calls bound,
unrecognised/duplicate keys on /compact and /merge-into (the shared
_dry_run_from verdict helpers' sibling sweep), and /cleanup-test-data's
dry_run parsing.

Findings source: SP\\p4-c-findings.md F5, F6, F8(a).

    python test_p4c_compact_verdict.py
"""
import json
import os
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="p4c-verdict-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main, memory, retrieval, summarizer, facts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

memory.ensure_storage_layout()
admin = TestClient(main.app, client=("127.0.0.1", 12348), raise_server_exceptions=False)

ROLLUP_CALLS: list[str] = []
_real_maybe_rollup = summarizer.maybe_rollup


async def _counting_maybe_rollup(conv_id, *a, **kw):
    ROLLUP_CALLS.append(conv_id)
    return await _real_maybe_rollup(conv_id, *a, **kw)


summarizer.maybe_rollup = _counting_maybe_rollup
# main.py imported summarizer.maybe_rollup by reference at call time
# (summarizer.maybe_rollup(...), not a bound alias), so patching the
# module attribute is visible to it.

LLM_CALLS: list[int] = []


async def _fake_llm(client, vllm_url, model, system_prompt, body_text, max_tokens, *, timeout=300.0):
    LLM_CALLS.append(len(body_text))
    return f"summary of {len(body_text)} chars"


summarizer._llm_summarize = _fake_llm

STORE: dict[str, list[dict]] = {}


def exchanges(n):
    return [
        {"turn_index": 1 + 2 * i, "document": f"[user]: question {i}\n[assistant]: answer {i}"}
        for i in range(n)
    ]


retrieval.export_indexed_exchanges = lambda cid: list(STORE.get(cid, []))
retrieval.conversation_doc_count = lambda cid: len(STORE.get(cid, []))
retrieval.import_indexed_exchange = lambda cid, ti, doc: True
retrieval.forget_conversation = lambda cid: 0

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def set_store(cid, n):
    STORE[cid] = exchanges(n)


def compact(cid, qs="", **kw):
    return admin.post(f"/admin/conversations/{cid}/compact{qs}", **kw)


# ---------------------------------------------------------------------------
# F5: max_calls is silently ignored when passed only in the query string.
# ---------------------------------------------------------------------------
print("[F5] ?max_calls= is honoured, not silently ignored")

cid = "f5-query-only"
set_store(cid, 400)  # enough turns that max_calls=1 vs. unbounded is visible
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
ROLLUP_CALLS.clear()
LLM_CALLS.clear()
r = compact(cid, "?max_calls=1", json={"dry_run": False})
j = r.json()
check(r.status_code == 200 and j.get("rollup_calls") == 1,
      f"?max_calls=1 alone bounds the loop to 1 pass (got "
      f"rollup_calls={j.get('rollup_calls')}, status={r.status_code})")
check(len(ROLLUP_CALLS) == 1,
      f"exactly 1 call to summarizer.maybe_rollup was made (got {len(ROLLUP_CALLS)})")
# THE CORE F5 PROOF (the reviewer's own reproduction shape): max_calls now
# bounds REAL vLLM calls, not rollup PASSES. Before this fix, ONE
# maybe_rollup pass drains every L1 chunk due in its own internal loop, so
# {"max_calls": 1} on this 400-turn backlog (20 L1 chunks at the default
# L1_CHUNK_SIZE) ran the WHOLE drain -- rollup_calls=1 but llm_calls=20 (or
# however many chunks were due), the exact shape the finding's own proof
# names ("rollup_calls=1 llm_calls=44").
check(len(LLM_CALLS) == 1,
      f"F5 core proof: max_calls=1 makes AT MOST 1 real vLLM call however "
      f"many the backlog needs unbounded (got {len(LLM_CALLS)} real calls)")
check(j.get("vllm_calls") == 1,
      f"the response reports vllm_calls=1, the number this parameter now "
      f"actually bounds (got {j.get('vllm_calls')})")

# CONTROL for the check above: this exact backlog genuinely needs MORE than
# one real vLLM call when not bounded to 1 -- otherwise "max_calls=1 makes
# at most 1 call" would trivially hold for the wrong reason (nothing to
# bound in the first place).
cid = "f5-control-many-calls-needed"
set_store(cid, 400)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
LLM_CALLS.clear()
r = compact(cid, json={"dry_run": False, "max_calls": 200})
j = r.json()
check(len(LLM_CALLS) > 1,
      f"CONTROL: this backlog needs MORE than 1 real vLLM call when not "
      f"bounded to 1 (got {len(LLM_CALLS)}) -- the fixture that makes the "
      f"check above meaningful")
check(j.get("vllm_calls") == len(LLM_CALLS),
      f"the reported vllm_calls matches the real count made (got "
      f"{j.get('vllm_calls')} vs {len(LLM_CALLS)} actual)")

# Stopping mid-drain must leave CONSISTENT state -- a partial L1 chunk is
# never recorded as covering turns it did not summarize -- and the next
# call must resume correctly, reaching the same final state an unbounded
# single run would.
print()
print("[F5 mid-drain] stopping at max_calls=1 leaves consistent state; "
      "the next call resumes and reaches the same end state as an "
      "unbounded run")

cid = "f5-midpoint-partial"
set_store(cid, 400)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
LLM_CALLS.clear()
r1 = compact(cid, json={"dry_run": False, "max_calls": 1})
j1 = r1.json()
mid_state = summarizer.load_state(cid)
check(len(LLM_CALLS) == 1, f"exactly 1 real call was made (got {len(LLM_CALLS)})")
check(len(mid_state.get("l1") or []) == 1,
      f"exactly ONE L1 chunk was recorded, not a partial or duplicate one "
      f"(got {len(mid_state.get('l1') or [])})")
_chunk = (mid_state.get("l1") or [{}])[0]
check(_chunk.get("first_turn") == 1 and _chunk.get("last_turn") == summarizer.L1_CHUNK_SIZE,
      f"the one recorded chunk covers EXACTLY the turns it actually "
      f"summarized (turns {_chunk.get('first_turn')}-{_chunk.get('last_turn')}, "
      f"expected 1-{summarizer.L1_CHUNK_SIZE})")
check(mid_state.get("last_summarized_turn") == summarizer.L1_CHUNK_SIZE,
      f"the watermark advanced to exactly that chunk's last turn, not "
      f"further (got {mid_state.get('last_summarized_turn')}, expected "
      f"{summarizer.L1_CHUNK_SIZE})")

# Resume: a second call with room to finish reaches the same final state a
# single UNBOUNDED run over the same backlog would (built separately, on a
# fresh conv_id, for the comparison).
LLM_CALLS.clear()
r2 = compact(cid, json={"dry_run": False, "max_calls": 200})
final_state = summarizer.load_state(cid)
check(r2.status_code == 200, f"the resuming call succeeds (got {r2.status_code})")
check(len(LLM_CALLS) >= 1,
      f"the resume made further real calls, continuing from the watermark "
      f"(got {len(LLM_CALLS)})")

cid_unbounded = "f5-midpoint-unbounded-control"
set_store(cid_unbounded, 400)
summarizer.save_state(cid_unbounded, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
compact(cid_unbounded, json={"dry_run": False, "max_calls": 200})
unbounded_state = summarizer.load_state(cid_unbounded)
check(final_state.get("last_summarized_turn") == unbounded_state.get("last_summarized_turn"),
      f"CONTROL: split-then-resumed reaches the SAME final watermark as one "
      f"unbounded run (split={final_state.get('last_summarized_turn')}, "
      f"unbounded={unbounded_state.get('last_summarized_turn')})")
check([c.get("first_turn") for c in (final_state.get("l1") or [])]
      == [c.get("first_turn") for c in (unbounded_state.get("l1") or [])],
      "CONTROL: the same L1 chunk boundaries too (no gap, no duplicate, no "
      "overlap introduced by splitting the drain across two calls)")

cid = "f5-body-and-query-agree"
set_store(cid, 400)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
ROLLUP_CALLS.clear()
r = compact(cid, "?max_calls=1", json={"dry_run": False, "max_calls": 1})
check(r.json().get("rollup_calls") == 1,
      f"body and query agreeing on 1 still bounds to 1 (got {r.json().get('rollup_calls')})")

cid = "f5-body-and-query-disagree"
set_store(cid, 400)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
ROLLUP_CALLS.clear()
r = compact(cid, "?max_calls=1", json={"dry_run": False, "max_calls": 200})
j = r.json()
check(j.get("rollup_calls") == 1,
      f"body=200 vs. query=1 disagree -> the SMALLER (safer) wins (got "
      f"{j.get('rollup_calls')})")

# CONTROL: max_calls in the body alone still works as before (this rule
# must be able to say yes at more than one call). Not asserting an EXACT
# call count here: the loop also stops early once the watermark stops
# advancing (a real 400-turn backlog can finish in fewer than `max_calls`
# passes, each pass draining everything currently due -- see the endpoint's
# own docstring on what max_calls actually bounds, F5) -- the invariant
# that matters for a REGRESSION check is that it never exceeds the bound.
cid = "f5-control-body-only"
set_store(cid, 400)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
ROLLUP_CALLS.clear()
r = compact(cid, json={"dry_run": False, "max_calls": 3})
check(1 <= r.json().get("rollup_calls", 0) <= 3,
      f"CONTROL: body-only max_calls=3 still bounds the loop (got "
      f"{r.json().get('rollup_calls')}, expected 1-3)")

# CONTROL: an explicit null in the body is "no opinion", not a parse error.
cid = "f5-control-null-body"
set_store(cid, 60)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
r = compact(cid, json={"dry_run": False, "max_calls": None})
check(r.status_code == 200,
      f"CONTROL: {{'max_calls': null}} is not a parse error, defaults as "
      f"if absent (got {r.status_code}: {r.text[:150]!r})")

# A malformed query value is still a 400 (not silently dropped).
cid = "f5-bad-query-value"
set_store(cid, 30)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
r = compact(cid, "?max_calls=abc", json={"dry_run": False})
check(r.status_code == 400,
      f"?max_calls=abc is a 400, not silently ignored (got {r.status_code})")


# ---------------------------------------------------------------------------
# F6: /compact still runs LIVE on a dry-intent key that is not a typo of
# dry_run, and on duplicate JSON keys.
# ---------------------------------------------------------------------------
print()
print("[F6] /compact refuses (400) an unrecognised body/query key or a "
      "duplicate JSON key, rather than silently running live")


def _live_case(label, **kw):
    cid = f"f6-{abs(hash(label)) % 100000}"
    set_store(cid, 30)
    summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
    LLM_CALLS.clear()
    r = compact(cid, **kw)
    j = {}
    try:
        j = r.json()
    except Exception:
        pass
    live = isinstance(j, dict) and "rollup_calls" in j
    check(r.status_code == 400 and not live and len(LLM_CALLS) == 0,
          f"{label}: 400, never a live drain (got status={r.status_code}, "
          f"live={live}, llm_calls={len(LLM_CALLS)})")


_live_case('{"dry": true}', json={"dry": True})
_live_case('{"dry_runs": true}', json={"dry_runs": True})
_live_case('{"is_dry_run": true}', json={"is_dry_run": True})
_live_case('{"preview": true}', json={"preview": True})
_live_case('{"options": {"dry_run": true}}', json={"options": {"dry_run": True}})
_live_case("?dry=1", qs="?dry=1")
_live_case("?dryrun_mode=true", qs="?dryrun_mode=true")
_live_case('duplicate key {"dry_run": true, "dry_run": false}',
           content=b'{"dry_run": true, "dry_run": false}',
           headers={"content-type": "application/json"})

# CONTROL: a real dry_run-typo (already-covered ground, test_admin_compact.py
# [9b]) is NOT refused -- it forces dry, 200, not a 400. Restated briefly
# here so this file's own suite proves the two rules coexist correctly.
cid = "f6-control-typo-still-dry"
set_store(cid, 30)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
r = compact(cid, "?dryrun=true", json={})
check(r.status_code == 200 and r.json().get("dry_run") is True,
      f"CONTROL: ?dryrun=true (a recognised dry_run typo) still forces dry "
      f"with 200, not refused as an unknown key (got {r.status_code}: "
      f"{r.json() if r.status_code == 200 else r.text[:150]})")

# CONTROL: the real vocabulary still runs live.
cid = "f6-control-live"
set_store(cid, 30)
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
LLM_CALLS.clear()
r = compact(cid, json={"dry_run": False, "max_calls": 5})
check(r.status_code == 200 and "rollup_calls" in r.json() and len(LLM_CALLS) > 0,
      f"CONTROL: {{'dry_run': false, 'max_calls': 5}} still runs live (got "
      f"{r.status_code}: {r.json() if r.status_code == 200 else r.text[:150]})")


# --- sibling sweep: /merge-into shares _dry_run_from and must refuse the
# same shapes. ---
print()
print("[F6 sibling] /merge-into (shares _dry_run_from) also refuses an "
      "unrecognised key or a duplicate JSON key")

facts.save_facts("f6m-src", [{"text": "src fact", "added_turn": 1, "last_used": 1, "pin": False}])
facts.save_facts("f6m-dst", [{"text": "dst fact", "added_turn": 1, "last_used": 1, "pin": False}])


def _merge_refused(label, **kw):
    r = admin.post("/admin/conversations/f6m-src/merge-into/f6m-dst", **kw)
    check(r.status_code == 400,
          f"merge-into {label}: 400 (got {r.status_code}: {r.text[:150]!r})")
    # Nothing changed either way.
    check(len(facts.load_facts("f6m-dst")) == 1,
          f"merge-into {label}: dst untouched (got "
          f"{len(facts.load_facts('f6m-dst'))} facts)")


_merge_refused('{"dry": true}', json={"dry": True})
r = admin.post("/admin/conversations/f6m-src/merge-into/f6m-dst?dry=1", json={})
check(r.status_code == 400, f"merge-into ?dry=1: 400 (got {r.status_code})")
r = admin.post("/admin/conversations/f6m-src/merge-into/f6m-dst",
                content=b'{"dry_run": true, "dry_run": false}',
                headers={"content-type": "application/json"})
check(r.status_code == 400,
      f"merge-into duplicate dry_run key: 400 (got {r.status_code})")

# CONTROL: merge-into's own real vocabulary (dry_run, refresh_last_used)
# still works.
r = admin.post("/admin/conversations/f6m-src/merge-into/f6m-dst",
                json={"dry_run": True, "refresh_last_used": True})
check(r.status_code == 200,
      f"CONTROL: merge-into's real vocabulary still works (got "
      f"{r.status_code}: {r.text[:150]!r})")


# ---------------------------------------------------------------------------
# F8(a): /cleanup-test-data's dry_run used to be a bare FastAPI bool query
# param (?dry_run=true&dry_run=false commits; a JSON body was ignored).
# ---------------------------------------------------------------------------
print()
print("[F8a] /cleanup-test-data honours _dry_run_from (query multi-value "
      "and body), not a bare last-wins bool query param")

r = admin.post("/admin/conversations/cleanup-test-data?dry_run=true&dry_run=false", json={})
check(r.status_code == 200 and r.json().get("dry_run") is True,
      f"?dry_run=true&dry_run=false (disagreeing repeats) reads DRY, not "
      f"last-wins commit (got {r.status_code}: {r.json() if r.status_code == 200 else r.text[:150]})")

r = admin.post("/admin/conversations/cleanup-test-data", json={"dry_run": True})
check(r.status_code == 200 and r.json().get("dry_run") is True,
      f"a JSON body {{'dry_run': true}} is now honoured, not ignored (got "
      f"{r.status_code}: {r.json() if r.status_code == 200 else r.text[:150]})")

# A body-only COMMIT, with the default itself being dry -- the case that
# actually distinguishes "the body is read" from "the body is ignored and
# the (also-dry) default silently wins regardless". Harmless to run live:
# no seeded test-pattern conv_ids exist in this store, so `removed` is
# always empty either way.
r = admin.post("/admin/conversations/cleanup-test-data", json={"dry_run": False})
check(r.status_code == 200 and r.json().get("dry_run") is False,
      f"a JSON body {{'dry_run': false}} actually commits (dry_run: false "
      f"in the response), not silently defaulting to the (also dry) query "
      f"default (got {r.status_code}: "
      f"{r.json() if r.status_code == 200 else r.text[:150]})")

# CONTROL: the documented default (no body, no query) is still dry.
r = admin.post("/admin/conversations/cleanup-test-data")
check(r.status_code == 200 and r.json().get("dry_run") is True,
      f"CONTROL: absent body/query still defaults to dry (got {r.status_code}: "
      f"{r.json() if r.status_code == 200 else r.text[:150]})")


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll F5/F6/F8(a) (hostile pass 4) compact-verdict checks passed.")
