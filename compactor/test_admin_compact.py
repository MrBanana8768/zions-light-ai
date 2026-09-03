"""POST /admin/conversations/{conv_id}/compact — the operator's drain.

Why this file exists. The endpoint shipped on 2026-08-29 with no test at all,
and it is the only surface in the estate whose whole job is to WRITE the
summary hierarchy on demand. Everything else that writes it does so as a tail
on a request nobody is watching; this one is typed by a human, on a live
conversation, during an incident, probably at 2am. The failure modes that
matter are therefore not "does it summarize well" but "what does it do to her
memory when the operator, the store, or the model is having a bad night."

Five properties, each one a thing that would cost her something if it broke:

  1. an absent conversation 404s rather than inventing an empty one
  2. `dry_run` writes NOTHING — no state file, no LLM call, no watermark move
  3. the position guard refuses (409) rather than summarizing text that is
     not the text the chunk labels will claim
  4. a path-shaped conv_id never reaches the filesystem
  5. the drain loop terminates — on progress, on failure, and on the cap

Every assertion here was mutation-verified: the guard it names was removed
from a scratch copy of main.py one at a time and this file was confirmed to
fail each time. See the block comment at the bottom for the exact mutations.

No server, no model, no network:
    python test_admin_compact.py
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-admin-compact-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main         # noqa: E402
import memory       # noqa: E402
import retrieval    # noqa: E402
import summarizer   # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

memory.ensure_storage_layout()

# client=127.0.0.1 satisfies _require_localhost without loosening
# COMPACTOR_ADMIN_BIND. raise_server_exceptions=False so an unhandled
# exception in the handler arrives as a 500 to assert on rather than a crash.
admin = TestClient(main.app, client=("127.0.0.1", 12345),
                   raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


LLM_CALLS = []
# What was actually SENT to be summarized. LLM_CALLS answers "did a call
# happen"; this answers "which turns did it contain", which is the only way to
# see a misaligned chunk — the labels look right either way.
LLM_BODIES = []


async def _fake_llm(client, vllm_url, model, system_prompt, body_text,
                    max_tokens, *, timeout=300.0):
    """Stand in for the vLLM round trip. Records that a call happened, which
    is the thing `dry_run` has to prove it did not do."""
    LLM_CALLS.append(len(body_text))
    LLM_BODIES.append(body_text)
    return f"summary of {len(body_text)} chars"


summarizer._llm_summarize = _fake_llm


def set_store(rows):
    """Replace the episodic export with a fixture. main calls this through
    the module attribute, so rebinding it here is the whole stub."""
    retrieval.export_indexed_exchanges = lambda conv_id: list(rows)


def exchanges(n):
    """n well-formed episodic rows in the canonical _exchange_doc shape."""
    return [
        {"turn_index": 1 + 2 * i,
         "document": f"[user]: question {i}\n[assistant]: answer {i}"}
        for i in range(n)
    ]


def snapshot():
    """Every file under the storage root, by path and content hash. The only
    honest way to assert 'nothing was written' — a watermark that did not move
    is not the same claim as a store that did not change."""
    out = {}
    for dirpath, _dirs, files in os.walk(_TMP_ROOT):
        for name in files:
            p = os.path.join(dirpath, name)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, _TMP_ROOT)] = hashlib.sha256(
                    fh.read()).hexdigest()
    return out


def memory_snapshot():
    """snapshot(), minus the bookkeeping a v3.1.4 rollup writes even when it
    summarizes nothing.

    `turns_seen` / `tail_fp` are the conversation's POSITION, and recording
    where the conversation got to is not the same act as writing a summary —
    the position has to be persisted on every pass or the next one re-seeds
    from the watermark, finds no anchor, and the hierarchy stops advancing
    under a capped client window (the 2026-09-01 defect). What "a run that
    summarized nothing wrote nothing" is protecting is the CONTENT: no chunk,
    no chapter, no theme, no watermark move. So that is what this compares.
    """
    out = {}
    for path, digest in snapshot().items():
        full = os.path.join(_TMP_ROOT, path)
        if os.path.dirname(path).endswith("summaries") and path.endswith(".json"):
            with open(full, "r", encoding="utf-8") as fh:
                try:
                    st = json.load(fh)
                except Exception:
                    out[path] = digest
                    continue
            content = {k: v for k, v in st.items()
                       if k not in ("turns_seen", "tail_fp", "updated_at")}
            if not (st.get("l1") or st.get("l2") or st.get("l3")
                    or st.get("last_summarized_turn")):
                # A file holding nothing but the position is not a summary;
                # dropping the KEY as well as the value is what makes "the
                # run created no memory" testable when there was no file at
                # all beforehand.
                continue
            out[path] = json.dumps(content, sort_keys=True)
        else:
            out[path] = digest
    return out


def compact(conv_id, **body):
    return admin.post(f"/admin/conversations/{conv_id}/compact", json=body)


# ---------------------------------------------------------------------------
# 1. An absent conversation 404s
# ---------------------------------------------------------------------------

print("[1] a conversation with no indexed exchanges is a 404, not an empty run")
set_store([])
before = snapshot()
r = compact("never-existed")
check(r.status_code == 404, f"HTTP 404 for an unknown conv (got {r.status_code})")
check("no indexed exchanges" in r.text,
      "the 404 body names the reason rather than being bare")
check(snapshot() == before, "nothing was written for a conv that does not exist")
check(not summarizer.summary_path("never-existed").exists(),
      "no summary state file was created for it")

# The 404 must not be reachable by accident from a store that IS there: an
# endpoint that 404s on everything would pass the assertions above.
print()
print("[1b] a conversation that DOES have exchanges is not a 404")
set_store(exchanges(4))
r = compact("has-a-few", dry_run=True)
check(r.status_code == 200, f"HTTP 200 when exchanges exist (got {r.status_code})")


# ---------------------------------------------------------------------------
# 2. dry_run changes nothing
# ---------------------------------------------------------------------------

print()
print("[2] dry_run writes no state, moves no watermark and calls no model")
CID = "dry"
summarizer.save_state(CID, {
    "l1": [{"text": "an existing scene", "first_turn": 1, "last_turn": 20}],
    "l2": [], "l3": None, "last_summarized_turn": 20,
})
set_store(exchanges(60))            # 120 messages: five more L1 rollups' worth
before = snapshot()
LLM_CALLS.clear()
r = compact(CID, dry_run=True)
body = r.json()
check(r.status_code == 200, f"HTTP 200 (got {r.status_code})")
check(LLM_CALLS == [], f"no LLM call was made (made {len(LLM_CALLS)})")
check(snapshot() == before, "not one byte under the storage root changed")
check(summarizer.load_state(CID)["last_summarized_turn"] == 20,
      "the watermark is still 20")
check(len(summarizer.load_state(CID)["l1"]) == 1,
      "the existing L1 scene is untouched")
check(body.get("dry_run") is True, "the report says dry_run: true")
check("note" in body, "the report carries the how-to-run-it-for-real note")
check(body.get("reconstructed_messages") == 120,
      f"it still reports what it WOULD do "
      f"(reconstructed_messages={body.get('reconstructed_messages')})")

# The same request without dry_run must actually do the work — otherwise
# assertion [2] is satisfied by an endpoint that never writes at all.
print()
print("[2b] the same request WITHOUT dry_run does write")
LLM_CALLS.clear()
r = compact(CID, dry_run=False)
check(r.status_code == 200, f"HTTP 200 (got {r.status_code})")
check(len(LLM_CALLS) > 0, f"the model was called ({len(LLM_CALLS)} times)")
check(summarizer.load_state(CID)["last_summarized_turn"] == 120,
      f"the watermark advanced to 120 (got "
      f"{summarizer.load_state(CID)['last_summarized_turn']})")
check(r.json().get("dry_run") is False,
      "a real run reports dry_run: false — the report must never claim a "
      "dry run it did not perform")


# ---------------------------------------------------------------------------
# 3. The watermark guard refuses
# ---------------------------------------------------------------------------

print()
print("[3] a reconstruction shorter than the watermark is refused, not run")
CID = "watermark"
summarizer.save_state(CID, {
    "l1": [{"text": "scene", "first_turn": 1, "last_turn": 20}],
    "l2": [], "l3": None, "last_summarized_turn": 300,
})
set_store(exchanges(10))            # 20 messages, against a watermark of 300
before = snapshot()
LLM_CALLS.clear()
r = compact(CID)
check(r.status_code == 409, f"HTTP 409 (got {r.status_code})")
check("refusing" in r.text, "the 409 body says it is refusing and why")
check(LLM_CALLS == [], "no LLM call was made")
check(snapshot() == before, "the store is byte-identical after the refusal")
check(summarizer.load_state(CID)["last_summarized_turn"] == 300,
      "the watermark was NOT pulled back to 20")

print()
print("[3e] a PULLED-DOWN watermark cannot be used to permit a short rebuild")
# v3.1.7 R12. A state file written by the pre-v3.1.4 code under a cap has its
# watermark pulled BELOW the chunks it tracks, and no turns_seen at all. Both
# counters then read low, so a guard that maxes only those two permits a
# rebuild it should refuse — and the chunks come back labelled against a
# position hundreds of turns short, which the endpoint's own comment calls
# "worse than no chunk, because nothing downstream can tell".
#
# The chunk labels are the record; the watermark is a pointer derived from
# them. summarizer._recorded_position is the one function that knows that, and
# the endpoint has to use it or the two disagree about what a transcript is.
CID = "watermark-pulled-down"
summarizer.save_state(CID, {
    # chunks reach turn 660; the watermark was dragged back to a cap of 100
    "l1": [{"text": "scene", "first_turn": 641, "last_turn": 660}],
    "l2": [], "l3": None, "last_summarized_turn": 100,
})
set_store(exchanges(60))            # 120 messages: over 100, far under 660
before = snapshot()
LLM_CALLS.clear()
r = compact(CID)
check(r.status_code == 409,
      f"HTTP 409 — 120 messages cannot rebuild a conversation at turn 660 "
      f"(got {r.status_code})")
check(LLM_CALLS == [], "no LLM call was made on a position it could not trust")
check(snapshot() == before, "the store is byte-identical after the refusal")
check(summarizer.load_state(CID)["l1"][0]["last_turn"] == 660,
      "and the chunk that proved the position is untouched")

print()
print("[3b] equal length is not short — the guard must not refuse a no-op")
CID = "watermark-equal"
summarizer.save_state(CID, {
    "l1": [], "l2": [], "l3": None, "last_summarized_turn": 20,
})
set_store(exchanges(10))            # exactly 20 messages
r = compact(CID, dry_run=True)
check(r.status_code == 200,
      f"HTTP 200 when the reconstruction exactly reaches the watermark "
      f"(got {r.status_code})")

print()
print("[3c] a longer reconstruction is allowed through")
CID = "watermark-long"
summarizer.save_state(CID, {
    "l1": [], "l2": [], "l3": None, "last_summarized_turn": 20,
})
set_store(exchanges(60))
r = compact(CID, dry_run=True)
check(r.status_code == 200, f"HTTP 200 (got {r.status_code})")

print()
print("[3d] the guard is the POSITION, not the watermark")
# v3.1.4. Under a capped client window the two come apart: the watermark is how
# far the SUMMARIES got, turns_seen is how far the CONVERSATION got, and it is
# turns_seen that drives the offset _do_l1_rollup subtracts to find a chunk's
# text. A reconstruction that clears the watermark but not the position makes
# that offset point at the wrong turns, and the chunk it stores is labelled
# 21-40 with somebody else's text in it. Comparing against the watermark alone
# lets that through with a 200.
CID = "position-guard"
summarizer.save_state(CID, {
    "l1": [], "l2": [], "l3": None, "last_summarized_turn": 20,
    "turns_seen": 300, "tail_fp": ["0123456789abcdef"],
})
set_store(exchanges(30))            # 60 messages: past the watermark, not the
before_mem = memory_snapshot()      # position
LLM_CALLS.clear()
r = compact(CID)
check(r.status_code == 409,
      f"HTTP 409 for a reconstruction behind the position (got {r.status_code})")
check("recorded position is already turn 300" in r.text,
      "and the body names the position it is behind")
check(LLM_CALLS == [], "no LLM call was made")
check(memory_snapshot() == before_mem, "and no summary content was written")


# ---------------------------------------------------------------------------
# 4. A path-shaped conv_id never reaches the filesystem
# ---------------------------------------------------------------------------

print()
print("[4] a path-shaped conv_id is rejected before it can name a file")
# memory._sanitize exists and is applied to conv_ids arriving on the CHAT
# path. Admin path params never pass through it, and summary_path() does no
# validation of its own — summary_path("../../x") is a real escape from the
# summaries directory. What stops it today is the router: {conv_id} matches
# no slash. That is one route decorator away from being untrue, so it is
# asserted here rather than assumed.
outside = os.path.join(_TMP_ROOT, "escaped.json")
set_store(exchanges(60))
before = snapshot()
for cid in ("a/b", "..%2F..%2Fescaped", "%2E%2E%2F%2E%2E%2Fescaped",
            "..%2fescaped", "a%2Fb"):
    r = admin.post(f"/admin/conversations/{cid}/compact", json={})
    check(r.status_code != 200,
          f"conv_id {cid!r} does not get a 200 (got {r.status_code})")
check(not os.path.exists(outside),
      "no file was written outside the summaries directory")
check(snapshot() == before, "a path-shaped id wrote nothing anywhere")
# And the mechanism is worth stating outright, because the assertion above
# passes for the wrong reason if summary_path ever starts sanitizing.
check("summaries" in str(summarizer.summary_path("plain")),
      "summary_path still resolves under summaries/ for an ordinary id")


# ---------------------------------------------------------------------------
# 5. The loop terminates
# ---------------------------------------------------------------------------

print()
print("[5] the drain loop terminates on progress")
CID = "drain"
set_store(exchanges(60))            # 120 messages -> six L1 chunks
LLM_CALLS.clear()
r = compact(CID)
body = r.json()
check(r.status_code == 200, f"HTTP 200 (got {r.status_code})")
check(body.get("rollup_calls", 999) <= 3,
      f"it stops as soon as the watermark stops moving "
      f"(rollup_calls={body.get('rollup_calls')})")
check(body.get("stopped_because") == "the watermark stopped advancing",
      f"and says so (stopped_because={body.get('stopped_because')!r})")
check(body.get("watermark_after") == 120,
      f"the whole backlog drained (watermark_after="
      f"{body.get('watermark_after')})")
check(body.get("watermark_after")
      == summarizer.load_state(CID)["last_summarized_turn"],
      "the reported watermark matches what is actually on disk")

print()
print("[5b] the loop terminates when every summarization fails")


async def _boom(*a, **kw):
    LLM_CALLS.append(-1)
    raise RuntimeError("vLLM is down")


CID = "drain-fail"
set_store(exchanges(60))
before_mem = memory_snapshot()
LLM_CALLS.clear()
summarizer._llm_summarize = _boom
r = compact(CID)
summarizer._llm_summarize = _fake_llm
body = r.json()
check(r.status_code == 200,
      f"a dead model is a report, not a 500 (got {r.status_code})")
check(body.get("rollup_calls", 999) <= 2,
      f"the loop does not spin against a failing model "
      f"(rollup_calls={body.get('rollup_calls')})")
check(body.get("watermark_after") == 0, "the watermark did not move")
check(memory_snapshot() == before_mem,
      "a run that summarized nothing wrote no summary content")

print()
print("[5c] max_calls bounds a rollup that advances forever")
# The only bound left if the no-progress break is ever removed. Modelled with
# a maybe_rollup that always advances by one turn — nothing in the real
# summarizer promises to stop.
_real_rollup = summarizer.maybe_rollup
SPINS = [0]


async def _always_advances(conv_id, messages, vllm_url, model):
    SPINS[0] += 1
    st = summarizer.load_state(conv_id)
    st["last_summarized_turn"] = st.get("last_summarized_turn", 0) + 1
    summarizer.save_state(conv_id, st)
    return st


CID = "drain-spin"
set_store(exchanges(60))
summarizer.maybe_rollup = _always_advances
r = compact(CID, max_calls=5)
summarizer.maybe_rollup = _real_rollup
body = r.json()
check(r.status_code == 200, f"HTTP 200 (got {r.status_code})")
check(SPINS[0] == 5,
      f"exactly max_calls rollups ran (ran {SPINS[0]})")
check(body.get("rollup_calls") == 5,
      f"and the report says 5 (says {body.get('rollup_calls')})")
check("max_calls" in str(body.get("stopped_because")),
      f"stopped_because names the cap "
      f"(got {body.get('stopped_because')!r})")


# ---------------------------------------------------------------------------
# 6. R13 — a gap is FILLED at its position, not closed by concatenation
#
# `turns_seen` counts every exchange, including the ones decide_memory_tail
# skipped and the ones bgwork shed; the episodic store holds only the indexed
# ones. Concatenating what the store has therefore produced an array SHORT by
# every gap, and short by even one turn is not merely short — every pair after
# the gap sits one exchange early, so a chunk labelled 1-20 quietly swallows
# exchange 11. The endpoint was right to refuse that, and refused for every
# real conversation: 63 skips in one measured window, one gap is enough, and
# this is the only rebuild-from-store recovery path there is.
#
# So place each pair at the position its `turn_index` records and fill the
# holes with an explicit placeholder. What is left to refuse is what padding
# cannot invent: a store that does not REACH the position.
# ---------------------------------------------------------------------------

print()
print("[6] one un-indexed exchange in fifteen is rebuilt, not refused")
# The backlog's exact reproduction: turns_seen=30, watermark=20,
# concatenation=28 -> HTTP 409. Exchange 6 never reached the index, and the
# 4-wide hole it leaves in turn_index is what production produces too — the
# request seed (len(messages)+1) advances by 2 whether or not a row is written.
CID = "gap-one"
gappy = [ex for i, ex in enumerate(exchanges(15)) if i != 6]
set_store(gappy)
summarizer.save_state(CID, {
    "l1": [], "l2": [], "l3": None, "last_summarized_turn": 0, "turns_seen": 30,
})
LLM_CALLS.clear()
LLM_BODIES.clear()
r = compact(CID)
body = r.json()
check(r.status_code == 200,
      f"HTTP 200 — one gap no longer refuses the whole conversation "
      f"(got {r.status_code}: {r.text[:120]})")
check(body.get("reconstructed_messages") == 30,
      f"the rebuild spans the full 30 turns "
      f"(got {body.get('reconstructed_messages')})")
check(body.get("indexed_exchanges") == 14,
      f"...from only 14 stored rows (got {body.get('indexed_exchanges')})")
check(body.get("gap_turns") == 2 and body.get("gap_exchanges") == 1,
      f"and the plan REPORTS the gap rather than hiding it "
      f"(gap_turns={body.get('gap_turns')}, "
      f"gap_exchanges={body.get('gap_exchanges')})")
check(body.get("recorded_position") == 30,
      f"the plan names the position it was measured against "
      f"(got {body.get('recorded_position')})")

# The assertion that actually matters: WHICH turns went into chunk 1-20. The
# labels are identical whether or not the alignment is right, so only the text
# can tell. Exchanges 0-5 and 7-9 belong to it; exchange 10 does not, and
# under the old concatenation it would have been dragged in.
check(len(LLM_BODIES) >= 1, "a chunk was actually summarized")
first = LLM_BODIES[0] if LLM_BODIES else ""
check(all(f"answer {i}" in first for i in list(range(6)) + [7, 8, 9]),
      "chunk 1-20 holds exactly the exchanges that belong to turns 1-20")
check("answer 10" not in first,
      "exchange 10 was NOT pulled forward into chunk 1-20 by the gap")
check(main._UNINDEXED_TURN_PLACEHOLDER in first,
      "and the missing exchange is present as an explicit placeholder, so the "
      "summary can say a turn is unaccounted for rather than skip silently")
_l1 = summarizer.load_state(CID).get("l1") or []
check(bool(_l1) and _l1[0]["last_turn"] == 20,
      f"the chunk is still labelled 1-20 (l1={_l1[:1]})")

print()
print("[6b] the placeholder is a summarization INPUT and never enters a store")
# The house rule: a marker written into the memory store would be extracted as
# a fact and become one of her memories. This one reaches maybe_rollup (proved
# in [6] above) and nothing else. _fake_llm never echoes its input, so any
# occurrence on disk would mean the placeholder was written directly.
_leaked = []
for _dirpath, _dirs, _files in os.walk(_TMP_ROOT):
    for _name in _files:
        _p = os.path.join(_dirpath, _name)
        with open(_p, "rb") as _fh:
            if main._UNINDEXED_TURN_PLACEHOLDER.encode() in _fh.read():
                _leaked.append(os.path.relpath(_p, _TMP_ROOT))
check(_leaked == [],
      f"no file under the storage root contains the placeholder (found "
      f"{_leaked})")

print()
print("[6c] a store with more gap than transcript is still refused")
# One corrupt turn_index would otherwise open a gap as wide as the number
# itself, and a reconstruction that is majority placeholder would spend a
# summarization call per chunk to record that nothing is known.
CID = "gap-majority"
sparse = [
    {"turn_index": 1, "document": "[user]: q\n[assistant]: a"},
    {"turn_index": 3, "document": "[user]: q\n[assistant]: a"},
    {"turn_index": 61, "document": "[user]: q\n[assistant]: a"},
]
set_store(sparse)
summarizer.save_state(CID, {
    "l1": [], "l2": [], "l3": None, "last_summarized_turn": 0,
})
before = snapshot()
LLM_CALLS.clear()
r = compact(CID)
check(r.status_code == 409,
      f"HTTP 409 for a reconstruction that is mostly holes (got {r.status_code})")
check("more of this transcript is missing than is present" in r.text.lower(),
      "and the body says which refusal this is")
check(LLM_CALLS == [], "no LLM call was made")
check(snapshot() == before, "the store is byte-identical after the refusal")

print()
print("[6d] a row whose document does not parse becomes a gap, not a shift")
# Vanishing is what shifted everything after it. The slot is still consumed.
CID = "gap-unparseable"
rows = exchanges(10)
rows[3] = {"turn_index": rows[3]["turn_index"], "document": "not an exchange"}
set_store(rows)
summarizer.save_state(CID, {
    "l1": [], "l2": [], "l3": None, "last_summarized_turn": 0,
})
r = compact(CID, dry_run=True)
body = r.json()
check(r.status_code == 200, f"HTTP 200 (got {r.status_code})")
check(body.get("reconstructed_messages") == 20,
      f"the unparseable row still occupies its two slots "
      f"(got {body.get('reconstructed_messages')})")
check(body.get("gap_turns") == 2,
      f"...counted as a gap (got {body.get('gap_turns')})")


# ---------------------------------------------------------------------------

print()
if FAILED:
    print(f"{len(FAILED)} assertion(s) failed:")
    for label in FAILED:
        print(f"  - {label}")
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    sys.exit(1)
shutil.rmtree(_TMP_ROOT, ignore_errors=True)
print("All admin compact tests passed.")

# ---------------------------------------------------------------------------
# Mutation record. Each guard was removed from a scratch copy of main.py and
# this file re-run; the section named is the one that went red.
#
#   `if not exchanges:`            -> `if False:`            ->  [1]
#   `if dry_run or not messages:`  -> `if not messages:`     ->  [2]
#   `if len(messages) < _pos:`     -> `if False:`            ->  [3]
#   `_pos = max(turns_seen, watermark)` -> `= watermark`     ->  [3d]
#   `{conv_id}/compact`            -> `{conv_id:path}/compact`-> [4]
#   `if now <= prev:`              -> `if False:`            ->  [5]
#   `while calls < max_calls:`     -> `while calls < max_calls + 3:` -> [5c]
#
# v3.1.7 (R13), same treatment:
#
#   `at = max(_idx(ex) - base, cursor)` -> `at = cursor`
#     (spacing ignored, gaps closed — the pre-R13 concatenation)  ->  [6],
#     and it comes back with the ORIGINAL defect's body verbatim:
#     "rebuilds 28 messages ... including 0 placeholder turns"
#   that, PLUS the unparseable-row `continue` hoisted above the
#     slot arithmetic (the pre-R13 shape entire)                  ->  [6d]
#   `if gap_turns > _real_turns:` -> `if False:`                  ->  [6c]
#
# One mutation SURVIVED and is recorded because the reason is worth knowing:
# hoisting the unparseable-row `continue` above the slot arithmetic ON ITS OWN
# changes nothing, because a slot is derived from the row's turn_index and not
# from a running cursor — an unparseable row and a missing row are the same
# thing to this rebuild, which is the property [6d] is really pinning. It only
# becomes visible once placement is cursor-based as well, which is the
# two-part mutation above.
#
# A test whose assertions cannot be made to fail is a test that asserts
# nothing, and this branch has shipped two of those this week.
# ---------------------------------------------------------------------------
