"""
Hostile pass 4 (reviewer C): F2, F7, F8(b), F8(c) — the quarantine/import
admin surface (compactor/portability.py + main.py's
admin_import_conversation).

Findings source: SP\\p4-c-findings.md (target 41d6558; this worktree's HEAD
is later — every finding re-checked here, not assumed). F1 and F3's tests
live in test_p3c_admin_fuzz.py (F1, extending an existing CONTROL the brief
named) and test_p4c_restore.py (F3) respectively. F4 lives in
test_p4c_degeneracy.py. F5/F6/F8(a)/F8(d) live in test_p4c_compact_verdict.py.

    python test_p4c_findings.py
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP_ROOT = tempfile.mkdtemp(prefix="p4c-findings-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main, memory, portability, retrieval, summarizer, facts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

memory.ensure_storage_layout()
admin = TestClient(main.app, client=("127.0.0.1", 12346), raise_server_exceptions=False)

STORE: dict[str, list[dict]] = {}


def exchanges(n):
    return [
        {"turn_index": 1 + 2 * i, "document": f"[user]: question {i}\n[assistant]: answer {i}"}
        for i in range(n)
    ]


_real_export_indexed_exchanges = retrieval.export_indexed_exchanges = None  # set below
retrieval.export_indexed_exchanges = lambda cid: list(STORE.get(cid, []))
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

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


# A small, valid bundle to import over things below (source content doesn't
# matter for F2/F7/F8 — only the TARGET side's behavior does).
_bsrc = "p4c-bundle-src"
facts.save_facts(_bsrc, [{"text": "bundle fact", "added_turn": 1, "last_used": 1, "pin": False}])
STORE[_bsrc] = exchanges(1)
BUNDLE = admin.get(f"/admin/conversations/{_bsrc}/export").json()


# ---------------------------------------------------------------------------
# F2: the quarantine verified episodic against its OWN export, not against
# the count it measured first (conversation_doc_count) -- so a transient
# export failure (which, per export_indexed_exchanges's own documented
# contract, returns [] rather than raising) published a 0-row snapshot that
# verified clean against ITSELF, and the overwrite that followed emptied a
# real episodic index the snapshot claimed to protect.
# ---------------------------------------------------------------------------
print("[F2] a transient episodic EXPORT failure at the moment of the "
      "snapshot must not let a 0-row snapshot publish against a real, "
      "non-zero conversation_doc_count")

victim2 = "f2p4-episodic-export-fails"
facts.save_facts(victim2, [{"text": "her fact", "added_turn": 1, "last_used": 1, "pin": False}])
STORE[victim2] = exchanges(20)  # conversation_doc_count(victim2) == 20

_real_export_indexed = retrieval.export_indexed_exchanges
_export_calls = {"n": 0}


def _export_fails_once(cid):
    # Exactly export_indexed_exchanges's own documented failure contract:
    # [] on failure, never an exception. Only the FIRST call for this one
    # conv_id fails, matching "a transient Chroma error" rather than a
    # permanent one.
    if cid == victim2 and _export_calls["n"] == 0:
        _export_calls["n"] += 1
        return []
    return _real_export_indexed(cid)


retrieval.export_indexed_exchanges = _export_fails_once
try:
    r = admin.post("/admin/conversations/import",
                    json={"bundle": BUNDLE, "target_conv_id": victim2, "overwrite": True})
    check(r.status_code == 409,
          f"the overwrite is refused (409), not silently accepted, when the "
          f"snapshot's episodic export came back short of what "
          f"conversation_doc_count just measured (got {r.status_code}: "
          f"{r.text[:150]!r})")
    check(len(STORE.get(victim2, [])) == 20,
          f"and the real 20-row episodic index is UNTOUCHED by the refused "
          f"overwrite (got {len(STORE.get(victim2, []))})")
    check(len(portability.list_quarantine(victim2)) == 0,
          "no misleading 0-row snapshot was published either")
finally:
    retrieval.export_indexed_exchanges = _real_export_indexed

# CONTROL: the same shape, but the export succeeds (returns all 20) --- the
# rule must still be able to say yes, not just refuse whenever episodic is
# involved at all.
victim2b = "f2p4-episodic-export-ok"
facts.save_facts(victim2b, [{"text": "her fact", "added_turn": 1, "last_used": 1, "pin": False}])
STORE[victim2b] = exchanges(5)
r = admin.post("/admin/conversations/import",
                json={"bundle": BUNDLE, "target_conv_id": victim2b, "overwrite": True})
check(r.status_code == 200,
      f"CONTROL: a real (non-short) episodic export still overwrites "
      f"normally (got {r.status_code}: {r.text[:150]!r})")
check(len(portability.list_quarantine(victim2b)) == 1,
      "CONTROL: and it published exactly one real snapshot")


# ---------------------------------------------------------------------------
# F7: every REFUSED overwrite used to write a full quarantine snapshot
# first -- before bundle validation and the lock check -- leaving a
# never-pruned copy per refused attempt. Fixed by moving bundle validation,
# target resolution and the lock check ahead of the snapshot
# (portability._validate_target_ready, called from main.py before it ever
# touches quarantine_conversation).
# ---------------------------------------------------------------------------
print()
print("[F7] a refused overwrite (bad bundle version) must not write a "
      "quarantine snapshot at all")

victim7 = "f7p4-bad-bundle-version"
facts.save_facts(victim7, [{"text": "her fact", "added_turn": 1, "last_used": 1, "pin": False}])
STORE[victim7] = exchanges(4)
_bad_bundle = dict(BUNDLE)
_bad_bundle["version"] = "v9-does-not-exist"
for attempt in range(3):
    r = admin.post("/admin/conversations/import",
                    json={"bundle": _bad_bundle, "target_conv_id": victim7, "overwrite": True})
    check(r.status_code == 400,
          f"attempt {attempt}: a bad bundle version is refused (400) before "
          f"any snapshot (got {r.status_code})")
check(len(portability.list_quarantine(victim7)) == 0,
      f"NONE of the 3 refused attempts left a quarantine snapshot behind "
      f"(got {len(portability.list_quarantine(victim7))}; before this fix, "
      f"each one would have)")
check(len(facts.load_facts(victim7)) == 1,
      "and her fact is untouched by the refused attempts")

# CONTROL: the same conv, a VALID bundle -- overwrite still succeeds and
# still publishes exactly one real snapshot (the guarantee this fix must
# not have removed, only re-ordered relative to the validation that
# precedes it).
r = admin.post("/admin/conversations/import",
                json={"bundle": BUNDLE, "target_conv_id": victim7, "overwrite": True})
check(r.status_code == 200,
      f"CONTROL: a VALID bundle still overwrites normally after the bad "
      f"ones were refused (got {r.status_code}: {r.text[:150]!r})")
check(len(portability.list_quarantine(victim7)) == 1,
      f"CONTROL: and exactly one real snapshot exists now (got "
      f"{len(portability.list_quarantine(victim7))}) -- the snapshot-"
      f"before-destroy guarantee still holds for the attempt that actually "
      f"overwrites")


# ---------------------------------------------------------------------------
# F8(b): a whitespace-padded target_conv_id used to reach
# quarantine_conversation UNSTRIPPED while import_conversation stripped it
# -- two steps of the same request resolving to two different conv_ids.
# ---------------------------------------------------------------------------
print()
print("[F8b] a whitespace-padded target_conv_id: the snapshot and the "
      "import now agree on the same (stripped) conv_id")

facts.save_facts("padded-id", [{"text": "her fact", "added_turn": 1, "last_used": 1, "pin": False}])
STORE["padded-id"] = exchanges(2)
r = admin.post("/admin/conversations/import",
                json={"bundle": BUNDLE, "target_conv_id": "  padded-id  ", "overwrite": True})
check(r.status_code == 200,
      f"the padded id still resolves and overwrites normally (got "
      f"{r.status_code}: {r.text[:150]!r})")
check(len(portability.list_quarantine("padded-id")) == 1,
      f"the snapshot was published under the STRIPPED id 'padded-id' -- "
      f"the same one import_conversation itself writes to (got "
      f"{len(portability.list_quarantine('padded-id'))} under the stripped "
      f"name)")
check(len(portability.list_quarantine("  padded-id  ")) == 0,
      "and NOT under the raw, unstripped id (that would mean the snapshot "
      "and the import disagreed about which conversation this was)")


# ---------------------------------------------------------------------------
# F8(c): a non-string target_conv_id / bundle.source_conv_id / fork
# new_conv_id used to reach `.strip()` a few lines into portability.py and
# raise an uncaught AttributeError (a 500), instead of the 400 every other
# malformed-body case in this file gets.
# ---------------------------------------------------------------------------
print()
print("[F8c] non-string conv ids are 400s, never 500s")

_int_bundle = dict(BUNDLE)
_int_bundle["source_conv_id"] = 123  # bundle-side non-string id
cases = [
    ("target_conv_id: 123 (int)",
     {"bundle": BUNDLE, "target_conv_id": 123, "overwrite": True}),
    ("target_conv_id: ['a'] (list)",
     {"bundle": BUNDLE, "target_conv_id": ["a"], "overwrite": True}),
    ("bundle.source_conv_id: 123, no target_conv_id override",
     {"bundle": _int_bundle, "overwrite": True}),
]
for label, body in cases:
    r = admin.post("/admin/conversations/import", json=body)
    check(r.status_code == 400,
          f"{label}: 400, not a 500 (got {r.status_code}: {r.text[:150]!r})")

# The fork endpoint shares the same target-resolution code path
# (fork_conversation -> import_conversation -> _validate_target_ready) for
# new_conv_id.
facts.save_facts("f8c-fork-src", [{"text": "src fact", "added_turn": 1, "last_used": 1, "pin": False}])
r = admin.post("/admin/conversations/f8c-fork-src/fork", json={"new_conv_id": 123})
check(r.status_code == 400,
      f"fork new_conv_id: 123 (int): 400, not a 500 (got {r.status_code}: "
      f"{r.text[:150]!r})")

# CONTROL: a normal, present, non-empty string id is unaffected.
r = admin.post("/admin/conversations/f8c-fork-src/fork", json={"new_conv_id": "f8c-fork-dst"})
check(r.status_code == 200,
      f"CONTROL: a normal string new_conv_id still forks (got {r.status_code}: "
      f"{r.text[:150]!r})")


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll F2/F7/F8(b)/F8(c) (hostile pass 4) findings checks passed.")
