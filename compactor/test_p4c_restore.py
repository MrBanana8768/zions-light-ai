"""
Hostile pass 4 (reviewer C), F3: POST /admin/conversations/{id}/restore
restored EVERY archived fact on a body it could not read, a misspelled
key, or an empty/falsy text_substring -- and stamped them all used-now, so
the caller's next message archived her real active facts in their place.

Findings source: SP\\p4-c-findings.md F3. Proof reused conceptually from the
reviewer's SP\\p4-c\\p4c_admin.py section F; rebuilt as an assertion-based
test against the real endpoint per the fix-lane brief.

    python test_p4c_restore.py
"""
import os
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="p4c-restore-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main, memory, facts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

memory.ensure_storage_layout()
admin = TestClient(main.app, client=("127.0.0.1", 12347), raise_server_exceptions=False)

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def _seed(conv_id, n_active=85, n_archived=300):
    """The reviewer's own proof shape: a conv with a real active set and a
    much larger archive, so a wrongly-triggered restore-all is dramatic and
    obviously wrong, not a rounding error."""
    facts.save_facts(conv_id, [
        {"text": f"active fact {i}", "added_turn": i, "last_used": 2_000_000_000 - i, "pin": False}
        for i in range(n_active)
    ])
    facts.save_archive(conv_id, [
        {"text": f"archived fact {i}", "added_turn": i, "last_used": 1, "archived_at": 1, "pin": False}
        for i in range(n_archived)
    ])


def _restore(conv_id, **kw):
    return admin.post(f"/admin/conversations/{conv_id}/restore", **kw)


# ---------------------------------------------------------------------------
# CONTROL: a real, well-formed substring restores exactly the matching rows.
# ---------------------------------------------------------------------------
print("[F3] a real substring still restores exactly the matching rows "
      "(this rule must be able to say yes)")
cid = "f3-control-substring"
_seed(cid)
r = _restore(cid, json={"text_substring": "archived fact 299"})
j = r.json()
check(r.status_code == 200 and j.get("restored") == 1,
      f"CONTROL: a real substring restores exactly 1 (got {r.status_code}: {j})")
check(len(facts.load_facts(cid)) == 86,
      f"CONTROL: exactly one row moved to active (got {len(facts.load_facts(cid))})")

# ---------------------------------------------------------------------------
# CONTROL: the explicit, strict restore_all flag restores everything.
# ---------------------------------------------------------------------------
cid = "f3-control-restore-all"
_seed(cid)
r = _restore(cid, json={"restore_all": True})
j = r.json()
check(r.status_code == 200 and j.get("restored") == 300,
      f"CONTROL: explicit restore_all=true restores everything ({j})")
check(len(facts.load_facts(cid)) == 385,
      f"CONTROL: all 300 archived rows moved to active (got "
      f"{len(facts.load_facts(cid))})")

cid = "f3-control-restore-all-str"
_seed(cid)
r = _restore(cid, json={"restore_all": "yes"})
check(r.status_code == 200 and r.json().get("restored") == 300,
      f"CONTROL: restore_all accepts the same strict-affirmative spellings "
      f"as overwrite (got {r.status_code}: {r.json()})")


# ---------------------------------------------------------------------------
# THE BUG (F3): every one of these used to restore all 300 archived facts
# and stamp them used-now. None may restore anything now.
# ---------------------------------------------------------------------------
print()
print("[F3] none of these may restore anything -- each is a 400")

cases = [
    ("absent body", {}),
    ("empty JSON object", {"json": {}}),
    ('form-encoded text_substring=... (wrong content-type)',
     {"content": b"text_substring=fork fact 001",
      "headers": {"content-type": "application/x-www-form-urlencoded"}}),
    ("misspelled key textSubstring",
     {"json": {"textSubstring": "fork fact 001"}}),
    ('text_substring: "" (empty string)', {"json": {"text_substring": ""}}),
    ('text_substring: "   " (whitespace only)', {"json": {"text_substring": "   "}}),
    ("text_substring: 0", {"json": {"text_substring": 0}}),
    ("text_substring: false", {"json": {"text_substring": False}}),
    ("text_substring: null (explicit)", {"json": {"text_substring": None}}),
    ("text_substring: [] (list)", {"json": {"text_substring": []}}),
    ("text_substring: 123 (int -- used to 500, F8c)", {"json": {"text_substring": 123}}),
    ("trailing comma (malformed JSON)",
     {"content": b'{"text_substring": "x",}',
      "headers": {"content-type": "application/json"}}),
    ("restore_all: false, no substring", {"json": {"restore_all": False}}),
    ("restore_all: \"maybe\" (not a strict token)", {"json": {"restore_all": "maybe"}}),
    ("both text_substring and restore_all=true (ambiguous)",
     {"json": {"text_substring": "archived fact 7", "restore_all": True}}),
]

for i, (label, kw) in enumerate(cases):
    cid = f"f3-bug-{i}"
    _seed(cid)
    r = _restore(cid, **kw)
    active_after = facts.load_facts(cid)
    check(r.status_code in (400, 422) and len(active_after) == 85,
          f"{label}: refused (got {r.status_code}), her 85 active facts "
          f"untouched (got {len(active_after)}; before this fix: 200, "
          f"restored=300, active jumps to 385)")

# text_substring: 123 must specifically be a 400, not the pre-fix 500.
r = _restore("f3-bug-int-status", **{"json": {"text_substring": 123}})
# (already covered above; this restates the exact status distinction F8c
# calls out.)
check(r.status_code == 400,
      f"text_substring: 123 (int) is specifically a 400, not a 500 (got "
      f"{r.status_code})")


# ---------------------------------------------------------------------------
# facts.restore_from_archive itself: defense in depth for any caller that
# is not this endpoint.
# ---------------------------------------------------------------------------
print()
print("[F3] facts.restore_from_archive itself refuses a non-string "
      "text_substring (defense in depth, not just the HTTP layer)")
cid = "f3-direct-call"
_seed(cid)
try:
    facts.restore_from_archive(cid, text_substring=123)
    check(False, "facts.restore_from_archive(text_substring=123) should have raised TypeError")
except TypeError:
    check(True, "facts.restore_from_archive(text_substring=123) raises TypeError, not AttributeError")
except AttributeError:
    check(False, "facts.restore_from_archive(text_substring=123) still raises the OLD AttributeError")
# CONTROL: None (restore-all, the function's own direct contract) and a
# real string are both still fine.
check(facts.restore_from_archive(cid, text_substring=None) == 300,
      "CONTROL: facts.restore_from_archive(text_substring=None) still "
      "restores everything -- this is the function's OWN contract, "
      "unchanged; only the ADMIN ENDPOINT above now gates it behind an "
      "explicit flag")


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll F3 (hostile pass 4) /restore checks passed.")
