"""
Hostile pass 3 (reviewer C): F1, F2, F3 — admin body/query parsing on the
destructive admin endpoints (/admin/conversations/import overwrite,
/admin/conversations/<id>/compact dry_run + max_calls).

Adapted from the reviewer's own reproduction (SP\\p3-c\\p3c_admin_fuzz.py,
log SP\\p3-c\\adminfuzz1.log, rc 0) into assertion-based tests that live with
the code, per the fix-lane brief. Every case below is quoted, with its
expected (fixed) outcome, from that log.

    python test_p3c_admin_fuzz.py
"""
import hashlib
import os
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="p3c-adminfuzz-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main, memory, retrieval, summarizer, facts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

memory.ensure_storage_layout()
admin = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)

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


retrieval.export_indexed_exchanges = lambda cid: list(STORE.get(cid, exchanges(30)))
retrieval.conversation_doc_count = lambda cid: len(STORE.get(cid, []))


def _imp(cid, ti, doc):
    STORE.setdefault(cid, []).append({"turn_index": ti, "document": doc})
    return True


retrieval.import_indexed_exchange = _imp


def _forget(cid):
    n = len(STORE.get(cid, []))
    STORE.pop(cid, None)
    return n


retrieval.forget_conversation = _forget


def snap():
    out = {}
    for d, _, fs in os.walk(_TMP_ROOT):
        for f in fs:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, _TMP_ROOT)] = hashlib.sha256(fh.read()).hexdigest()
    return out


FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


# ---------------------------------------------------------------------------
# F1: /admin/conversations/import overwrite parsing (HIGH)
# ---------------------------------------------------------------------------
print("[F1] /admin/conversations/import overwrite must be STRICT — only "
      "true/\"true\"/\"1\"/\"yes\" ever overwrite")

bundle_src = "f1-bundle-src"
facts.save_facts(bundle_src, [{"text": "OLD bundle fact", "added_turn": 1, "last_used": 1, "pin": False}])
STORE[bundle_src] = [{"turn_index": 1, "document": "[user]: old\n[assistant]: old"}]
bundle = admin.get(f"/admin/conversations/{bundle_src}/export").json()

# CONTROL: the spellings that already correctly refused (unfixed and fixed
# code agree here — importer refuses because overwrite parses False AND the
# target has existing state).
for label, ov in [("absent", None), ("false (bool)", False), ("0 (int)", 0), ("null", "__null__"), ('""', "")]:
    victim = "f1-refuse-" + hashlib.md5(label.encode()).hexdigest()[:6]
    facts.save_facts(victim, [{"text": f"LIVE fact {n}", "added_turn": n, "last_used": 1000 + n, "pin": False}
                               for n in range(40)])
    STORE[victim] = exchanges(20)
    body = {"bundle": bundle, "target_conv_id": victim}
    if ov == "__null__":
        body["overwrite"] = None
    elif ov is not None:
        body["overwrite"] = ov
    r = admin.post("/admin/conversations/import", json=body)
    check(r.status_code == 400 and len(facts.load_facts(victim)) == 40,
          f"CONTROL overwrite={label!r}: refused (400), 40 live facts untouched")

# THE BUG (F1): non-empty strings that spell "false" used to read as
# bool()-truthy and overwrite. Must now stay refused, same as the bool False
# case above.
for label, ov in [('"false"', "false"), ('"no"', "no"), ('"0"', "0"), ('"off"', "off")]:
    victim = "f1-bug-" + hashlib.md5(label.encode()).hexdigest()[:6]
    facts.save_facts(victim, [{"text": f"LIVE fact {n}", "added_turn": n, "last_used": 1000 + n, "pin": False}
                               for n in range(40)])
    STORE[victim] = exchanges(20)
    r = admin.post("/admin/conversations/import",
                    json={"bundle": bundle, "target_conv_id": victim, "overwrite": ov})
    live_after = facts.load_facts(victim)
    check(r.status_code == 400 and len(live_after) == 40,
          f"FIXED overwrite={label} (a string spelling 'false'): still refused, "
          f"her 40 facts untouched (before this fix: 200, wiped to 1)")

# CONTROL the other direction: a real affirmative still overwrites (this
# rule must be able to say yes, not just refuse everything).
for label, ov in [(True, True), ('"true"', "true"), ('"1"', "1"), ('"yes"', "yes")]:
    victim = "f1-commit-" + hashlib.md5(str(label).encode()).hexdigest()[:6]
    facts.save_facts(victim, [{"text": f"LIVE fact {n}", "added_turn": n, "last_used": 1000 + n, "pin": False}
                               for n in range(40)])
    STORE[victim] = exchanges(20)
    r = admin.post("/admin/conversations/import",
                    json={"bundle": bundle, "target_conv_id": victim, "overwrite": ov})
    check(r.status_code == 200 and r.json().get("overwrote_existing") is True,
          f"CONTROL overwrite={label}: a real affirmative still overwrites (rule can say yes)")


# ---------------------------------------------------------------------------
# F2 / F3: /admin/conversations/<id>/compact body & max_calls parsing (HIGH/LOW)
# ---------------------------------------------------------------------------
print()
print("[F2] /compact must not run LIVE on an unparseable/misspelled/non-"
      "object body; [F3] max_calls must 400, never 500, on Infinity/bool")

cases = [
    ("form-encoded (curl -d default)", b"dry_run=true", "application/x-www-form-urlencoded", "", 400),
    ("single-quoted pseudo-JSON", b"{'dry_run': true}", "application/json", "", 400),
    ("trailing comma", b'{"dry_run": true,}', "application/json", "", 400),
    ("unquoted key", b'{dry_run: true}', "application/json", "", 400),
    ("JSON array wrapping", b'[{"dry_run": true}]', "application/json", "", 400),
    ("camelCase key dryRun", b'{"dryRun": true}', "application/json", "", None),
    ("hyphen key dry-run", b'{"dry-run": true}', "application/json", "", None),
    ("upper key DRY_RUN", b'{"DRY_RUN": true}', "application/json", "", None),
    ("key with trailing space", b'{"dry_run ": true}', "application/json", "", None),
    ("query dryRun", b"", "application/json", "?dryRun=true", None),
    ("query dry_run[]", b"", "application/json", "?dry_run[]=true", None),
    ("query 'dry_run ' (encoded space)", b"", "application/json", "?dry_run%20=true", None),
    ("body JSON string 'true'", b'"true"', "application/json", "", 400),
    ("control: {\"dry_run\": true}", b'{"dry_run": true}', "application/json", "", None),
    ("max_calls 1e999 + dry_run true", b'{"max_calls": 1e999, "dry_run": true}', "application/json", "", 400),
    ("max_calls Infinity + dry_run true", b'{"max_calls": Infinity, "dry_run": true}', "application/json", "", 400),
    ("max_calls -Infinity + dry_run true", b'{"max_calls": -Infinity, "dry_run": true}', "application/json", "", 400),
    ("max_calls NaN + dry_run true", b'{"max_calls": NaN, "dry_run": true}', "application/json", "", 400),
    ("max_calls true (bool) + dry_run true", b'{"max_calls": true, "dry_run": true}', "application/json", "", 400),
    ("max_calls 5000-digit int + dry_run true",
     ('{"max_calls": ' + "9" * 5000 + ', "dry_run": true}').encode(), "application/json", "", 400),
]

for i, (label, raw, ctype, qs, expect_status) in enumerate(cases):
    cid = f"f2-victim{i}"
    summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
    before = snap()
    LLM_CALLS.clear()
    r = admin.post(f"/admin/conversations/{cid}/compact{qs}", content=raw, headers={"content-type": ctype})
    try:
        j = r.json()
    except Exception:
        j = {}
    wrote = snap() != before
    live = isinstance(j, dict) and "rollup_calls" in j
    # The one invariant that matters for every case in this table: NEVER a
    # live drain (a write + LLM calls) from a body that could not be read,
    # was not an object, used a misspelled key, or held a value that should
    # 400. Every case here is either explicitly a 400 (expect_status=400) or
    # explicitly a documented-dry outcome (expect_status=None means "check
    # dry_run/live only, not the exact status").
    if expect_status == 400:
        check(r.status_code == 400 and not wrote and len(LLM_CALLS) == 0,
              f"{label}: 400, no write, no LLM calls (was: live drain)")
    else:
        check(not live and not wrote and len(LLM_CALLS) == 0,
              f"{label}: dry (no rollup_calls key / no write / no LLM calls), "
              f"got status={r.status_code} live={live} wrote={wrote}")

# CONTROL: a well-formed LIVE request still actually runs (this rule must be
# able to say yes, not just refuse/dry-run everything).
cid = "f2-control-live"
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
LLM_CALLS.clear()
r = admin.post(f"/admin/conversations/{cid}/compact", json={"dry_run": False})
j = r.json()
check(r.status_code == 200 and "rollup_calls" in j and len(LLM_CALLS) > 0,
      f"CONTROL: {{\"dry_run\": false}} on a well-formed body still runs live "
      f"(got status={r.status_code}, calls={len(LLM_CALLS)})")

# F3 CONTROL: max_calls=0 is a legitimate "run the guards, no calls" probe,
# not an error and not silently replaced by the 200-call default.
cid = "f3-control-zero"
summarizer.save_state(cid, {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0})
LLM_CALLS.clear()
r = admin.post(f"/admin/conversations/{cid}/compact", json={"max_calls": 0, "dry_run": False})
j = r.json()
check(r.status_code == 200 and j.get("rollup_calls") == 0 and len(LLM_CALLS) == 0,
      f"F3 CONTROL: max_calls=0 runs the guards with zero calls, not the 200 "
      f"default (got rollup_calls={j.get('rollup_calls')})")


# ---------------------------------------------------------------------------
# F1b: an overwrite whose safety snapshot cannot be written is REFUSED
# ---------------------------------------------------------------------------
print()
print("[F1b] a failed pre-overwrite snapshot refuses the overwrite; an already-"
      "unreadable store still imports as a recovery")
import portability  # noqa: E402

_real_quarantine = portability.quarantine_conversation
try:
    def _q_fails(conv_id, *, reason):
        raise portability.QuarantineError("simulated: snapshot could not be verified")

    portability.quarantine_conversation = _q_fails
    victim = "f1b-snapshot-fails"
    facts.save_facts(victim, [{"text": f"LIVE fact {n}", "added_turn": n, "last_used": 1000 + n, "pin": False}
                               for n in range(40)])
    STORE[victim] = exchanges(20)
    r = admin.post("/admin/conversations/import",
                   json={"bundle": bundle, "target_conv_id": victim, "overwrite": True})
    check(r.status_code == 409 and len(facts.load_facts(victim)) == 40,
          f"a snapshot failure refuses the overwrite (409) and her 40 facts stay "
          f"(got {r.status_code}, {len(facts.load_facts(victim))} facts)")

    def _q_unreadable(conv_id, *, reason):
        raise memory.StoreUnreadable(memory.storage_root() / "facts" / f"{conv_id}.json",
                                     ValueError("simulated torn write"))

    portability.quarantine_conversation = _q_unreadable
    victim = "f1b-store-unreadable"
    facts.save_facts(victim, [{"text": "LIVE fact", "added_turn": 1, "last_used": 1, "pin": False}])
    STORE[victim] = exchanges(2)
    r = admin.post("/admin/conversations/import",
                   json={"bundle": bundle, "target_conv_id": victim, "overwrite": True})
    check(r.status_code == 200,
          f"CONTROL: an already-unreadable store still imports as a recovery "
          f"(got {r.status_code}: {r.text[:120]})")
finally:
    portability.quarantine_conversation = _real_quarantine


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll F1/F2/F3 (hostile pass 3) admin-fuzz checks passed.")
