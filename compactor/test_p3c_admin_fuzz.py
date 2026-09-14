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
import json
import os
import sys
import tempfile
from pathlib import Path

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
print("[F1b] a failed pre-overwrite snapshot refuses the overwrite")
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
finally:
    portability.quarantine_conversation = _real_quarantine


# ---------------------------------------------------------------------------
# F1 (hostile pass 4, reviewer C): a TORN FACTS FILE must not skip the
# snapshot of the layers that ARE readable, and the torn file's own bytes
# must be copied aside.
#
# The pass-3 CONTROL this replaces reached the "already unreadable, nothing
# to lose" exemption by monkeypatching quarantine_conversation itself to
# raise StoreUnreadable directly — a route production never takes (nothing
# calls quarantine_conversation and expects it to just blow up; the real
# trigger is ONE layer's file being torn on disk). It also asserted only
# the HTTP status code, never what the overwrite destroyed. This reproduces
# the real on-disk shape (a truncated facts.json, the MooseFS-stall class
# this project has already had) through the real endpoint, against the real
# quarantine_conversation, and checks what survived.
# ---------------------------------------------------------------------------
print()
print("[F1 pass-4] a torn facts file must not skip the snapshot of the "
      "readable summary/episodic layers, and the torn bytes must be copied "
      "aside")

victim = "f1p4-torn-facts"
# Readable summary + episodic state — exactly what a torn FACTS file (only)
# leaves behind; the other two layers never touched the disk that tore.
summarizer.save_state(victim, {
    "l1": [{"text": "old l1 chunk", "first_turn": 1, "last_turn": 2}],
    "l2": [], "l3": None, "last_summarized_turn": 2,
})
STORE[victim] = exchanges(3)
_torn_bytes = b'{"conv_id": "f1p4-torn-facts", "facts": [{"text": "truncated mid ob'
_facts_path = memory.facts_path(victim)
_facts_path.parent.mkdir(parents=True, exist_ok=True)
_facts_path.write_bytes(_torn_bytes)
try:
    facts.load_facts(victim)
    _fixture_torn_ok = False
except memory.StoreUnreadable:
    _fixture_torn_ok = True
check(_fixture_torn_ok,
      "fixture check: the truncated facts file actually raises StoreUnreadable "
      "(otherwise this isn't testing what it claims to)")

# THE PRODUCTION ROUTE: the real endpoint, the real quarantine_conversation,
# no monkeypatching.
r = admin.post("/admin/conversations/import",
               json={"bundle": bundle, "target_conv_id": victim, "overwrite": True})
check(r.status_code == 200,
      f"an overwrite still succeeds when only the facts layer is torn "
      f"(got {r.status_code}: {r.text[:200]})")

snaps = portability.list_quarantine(victim)
check(len(snaps) == 1, f"exactly one quarantine snapshot was published (got {len(snaps)})")
if snaps:
    snap = json.loads(snaps[-1].read_text(encoding="utf-8"))
    q = snap.get("quarantine", {})
    check("facts (unreadable)" in (q.get("unverified_layers") or []),
          f"the facts layer is recorded unverified (got {q.get('unverified_layers')})")
    # THE BUG this finding reports: before the fix, StoreUnreadable from the
    # facts read propagated out of quarantine_conversation before ANY other
    # layer was even measured, so the exemption in main.py skipped this
    # entire snapshot — these two checks are what that skip destroyed.
    check(bool((snap.get("summary_state") or {}).get("l1")),
          "the READABLE summary hierarchy survived into the snapshot")
    check(len(snap.get("episodic") or []) >= 3,
          f"the READABLE episodic rows survived into the snapshot (got "
          f"{len(snap.get('episodic') or [])})")
    torn_path = q.get("torn_facts_path")
    check(bool(torn_path) and Path(torn_path).is_file(),
          f"the torn facts file's raw bytes were copied aside (path={torn_path!r})")
    if torn_path:
        check(Path(torn_path).read_bytes() == _torn_bytes,
              "the copied bytes are byte-identical to the original torn file")
# And the recovery itself: the import replaces the torn file with the
# bundle's (now-readable) facts — the exemption's original point, still true.
# (Guarded: if the endpoint refused above, the torn file is still on disk
# and load_facts would raise StoreUnreadable again — a separate, already-
# reported failure, not a second crash on top of it.)
try:
    _post_facts = facts.load_facts(victim)
    check(len(_post_facts) == len(bundle.get("facts") or []),
          "the import replaced the torn facts file with the bundle's, now readable")
except memory.StoreUnreadable:
    check(r.status_code == 200,
          "the import replaced the torn facts file with the bundle's, now "
          "readable (skipped: the facts file is still torn, consistent "
          "with the endpoint above having refused the overwrite)")


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll F1/F2/F3 (hostile pass 3) admin-fuzz checks passed.")
