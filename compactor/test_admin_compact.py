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
  3. the watermark guard refuses (409) rather than pulling the watermark
     backwards, which is how the same turns get summarized twice
  4. a path-shaped conv_id never reaches the filesystem
  5. the drain loop terminates — on progress, on failure, and on the cap

Every assertion here was mutation-verified: the guard it names was removed
from a scratch copy of main.py one at a time and this file was confirmed to
fail each time. See the block comment at the bottom for the exact mutations.

No server, no model, no network:
    python test_admin_compact.py
"""

import hashlib
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


async def _fake_llm(client, vllm_url, model, system_prompt, body_text,
                    max_tokens, *, timeout=300.0):
    """Stand in for the vLLM round trip. Records that a call happened, which
    is the thing `dry_run` has to prove it did not do."""
    LLM_CALLS.append(len(body_text))
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
before = snapshot()
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
check(snapshot() == before,
      "a run that summarized nothing wrote nothing")

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
#   `if len(messages) < _wm:`      -> `if False:`            ->  [3]
#   `{conv_id}/compact`            -> `{conv_id:path}/compact`-> [4]
#   `if now <= prev:`              -> `if False:`            ->  [5]
#   `while calls < max_calls:`     -> `while calls < max_calls + 3:` -> [5c]
#
# A test whose assertions cannot be made to fail is a test that asserts
# nothing, and this branch has shipped two of those this week.
# ---------------------------------------------------------------------------
