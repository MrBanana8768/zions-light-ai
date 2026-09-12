"""One surrogate rule for every JSON body, and an import that cannot wipe what
it has not yet replaced.

v3.1.9, M3 — BLOCKER in hostile pass #2. A lone surrogate ("\\ud83d") is valid
JSON and a valid Python str and cannot be encoded as UTF-8. chat_completions
refused it; its seven body-parsing siblings did not. Through
/admin/conversations/import with overwrite=true, import_conversation cleared the
facts file, cleared the episodic index, and only then wrote the bundle — which
raised UnicodeEncodeError, a ValueError the endpoint did not catch. Executed
against the real memory.py: 105 facts in, [] on disk, HTTP 500.

Three layers are pinned here, because any one alone leaves a route:

  [1] STRUCTURAL: every handler in main.py that calls request.json() also calls
      _refuse_unpaired_surrogate. A guard copied to seven places is how the
      first one was missed; this is what stops an eighth handler repeating it.
  [2] BEHAVIOURAL: each of the seven endpoints answers a surrogate body with a
      400, not a 500 — and a control body with a PAIRED surrogate (an ordinary
      emoji) is not refused.
  [3] ORDER: import_conversation writes the replacements before it clears
      anything, so ANY failure in the bundle write — not only this one — leaves
      the conversation as it was.

    python test_surrogate_guard.py
"""

import ast
import json as _json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="surrogate-guard-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "2000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import memory  # noqa: E402

memory.ensure_storage_layout()

import facts  # noqa: E402
import main  # noqa: E402
import portability  # noqa: E402
import retrieval  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

HERE = Path(__file__).resolve().parent
FAILED: list[str] = []
LONE = "\ud83d"            # half an emoji
PAIRED = "\U0001F600"      # a whole one: json.loads combines the pair


def check(cond, label):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)


# ---------------------------------------------------------------------------
print("[1] every handler that reads a JSON body refuses an unpaired surrogate")


def _json_readers_missing_guard(src: str) -> tuple[list[str], list[str]]:
    """(handlers that call request.json(), those of them that never call
    _refuse_unpaired_surrogate)."""
    readers, missing = [], []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        reads = any(
            isinstance(c.func, ast.Attribute) and c.func.attr == "json"
            and isinstance(c.func.value, ast.Name) and c.func.value.id == "request"
            for c in calls
        )
        if not reads:
            continue
        readers.append(node.name)
        guarded = any(
            isinstance(c.func, ast.Name) and c.func.id == "_refuse_unpaired_surrogate"
            for c in calls
        )
        if not guarded:
            missing.append(node.name)
    return readers, missing


# CONTROL FIRST: the detector must be able to see a handler that forgot.
_planted = (
    "async def forgot(request):\n"
    "    body = await request.json()\n"
    "    return body\n"
    "async def remembered(request):\n"
    "    body = await request.json()\n"
    "    _refuse_unpaired_surrogate(body)\n"
)
_r, _m = _json_readers_missing_guard(_planted)
check(_r == ["forgot", "remembered"] and _m == ["forgot"],
      f"CONTROL: the detector flags a planted handler that reads JSON without the "
      f"guard, and only that one (readers={_r}, missing={_m})")

readers, missing = _json_readers_missing_guard(
    (HERE / "main.py").read_text(encoding="utf-8"))
check(len(readers) >= 7,
      f"it found the body readers in main.py at all ({len(readers)}: {readers}) — "
      f"a detector that matches nothing passes everything")
check(missing == [],
      f"and every one of them calls _refuse_unpaired_surrogate (missing: {missing})")

# ---------------------------------------------------------------------------
print()
print("[2] each endpoint answers a surrogate body with 400, never 500")
admin = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)
CID, DST = "surr-conv", "surr-dst"

_ENDPOINTS = [
    ("persona", f"/admin/conversations/{CID}/persona", {"text": f"likes tea {LONE}"}),
    ("inherit-persona", f"/admin/conversations/{CID}/inherit-persona",
     {"source_conv_id": "somewhere", "note": LONE}),
    ("restore", f"/admin/conversations/{CID}/restore", {"text_substring": LONE}),
    ("import", "/admin/conversations/import",
     {"bundle": {"version": portability.BUNDLE_VERSION, "facts": [{"text": LONE}],
                 "summary_state": {}, "episodic": []},
      "target_conv_id": CID, "overwrite": True}),
    ("fork", f"/admin/conversations/{CID}/fork", {"target_conv_id": LONE}),
    ("merge-into", f"/admin/conversations/{CID}/merge-into/{DST}", {"note": LONE}),
    ("compact", f"/admin/conversations/{CID}/compact", {"dry_run": True, "note": LONE}),
]
for name, path, body in _ENDPOINTS:
    # json= would re-encode the str; send the escape as the client would.
    raw = _json.dumps(body)            # ensure_ascii=True: "\ud83d" stays an escape
    check("\\ud83d" in raw, f"fixture ({name}): the body carries the escape, not raw bytes")
    r = admin.post(path, content=raw, headers={"content-type": "application/json"})
    check(r.status_code == 400 and "surrogate" in r.text,
          f"{name}: 400 naming the surrogate (got {r.status_code}: {r.text[:90]!r})")

# And the ORIGINAL site. chat_completions has refused a lone surrogate since
# v3.1.8 and nothing in the unit suite ever proved it — the only test lived in
# the adversarial stack. This commit moved its detection onto the shared
# helper, so without this check that refactor would be unverified. It answers
# in OpenAI's error shape, not FastAPI's, which is why only the detector is
# shared.
_chat = admin.post(
    "/v1/chat/completions",
    content=_json.dumps({"model": "m", "messages": [{"role": "user", "content": f"hi {LONE}"}]}),
    headers={"content-type": "application/json"},
)
check(_chat.status_code == 400
      and (_chat.json().get("error") or {}).get("code") == "unpaired_surrogate",
      f"chat_completions: 400 with code unpaired_surrogate (got {_chat.status_code}: "
      f"{_chat.text[:90]!r})")

# CONTROL: a PAIRED surrogate is an ordinary emoji and must not be refused by
# this rule. Whatever the endpoint then says, it must not be the surrogate 400.
_r = admin.post(f"/admin/conversations/{CID}/persona",
                content=_json.dumps({"text": f"likes tea {PAIRED}"}),
                headers={"content-type": "application/json"})
check("surrogate" not in _r.text,
      f"CONTROL: a paired surrogate (an emoji) is not refused by the rule "
      f"(got {_r.status_code}: {_r.text[:90]!r})")

# ---------------------------------------------------------------------------
print()
print("[3] the import that used to delete 105 facts now deletes none")
TARGET = "surr-import-target"
_seed = [{"text": f"fact number {i}", "added_turn": i, "last_used": i} for i in range(105)]


def _reseed():
    facts.save_facts(TARGET, list(_seed))


_forgot: list[str] = []
_stub_retrieval = [
    patch.object(retrieval, "forget_conversation", lambda c: _forgot.append(c)),
    patch.object(retrieval, "conversation_doc_count", lambda c: 0),
    patch.object(retrieval, "import_indexed_exchange", lambda *a, **k: True),
]
for p in _stub_retrieval:
    p.start()

# 3a. through the endpoint
_reseed()
_forgot.clear()
raw = _json.dumps({"bundle": {"version": portability.BUNDLE_VERSION,
                              "facts": [{"text": LONE}], "summary_state": {},
                              "episodic": []},
                   "target_conv_id": TARGET, "overwrite": True})
r = admin.post("/admin/conversations/import", content=raw,
               headers={"content-type": "application/json"})
check(r.status_code == 400, f"the endpoint refuses (got {r.status_code})")
check(len(facts.load_facts(TARGET)) == 105, "and all 105 facts are still on disk")
check(_forgot == [], "and the episodic index was not cleared")

# 3b. directly — every caller that is not the endpoint, e.g. a script loading a
# bundle from disk, must be protected by import_conversation itself.
_reseed()
_forgot.clear()
_bundle = {"version": portability.BUNDLE_VERSION, "facts": [{"text": LONE}],
           "summary_state": {}, "episodic": []}
try:
    portability.import_conversation(_bundle, target_conv_id=TARGET, overwrite=True)
    raised = None
except Exception as e:
    # Broad on purpose, and the check below names the type. Catching only
    # ImportError_ meant that with the encodability check removed, the raw
    # UnicodeEncodeError escaped as a traceback: the defect was detected as a
    # crash on the next line instead of as a failed check on this one.
    raised = e
check(isinstance(raised, portability.ImportError_) and "surrogate" in str(raised),
      f"import_conversation refuses the bundle before any I/O (got {raised!r})")
check(len(facts.load_facts(TARGET)) == 105, "and nothing was deleted")
check(_forgot == [], "and the episodic index was not cleared")

# 3c. ANY failure in the bundle write, not just an encoding one. This is the
# case the reorder exists for: a disk that fills between validation and write.
_reseed()
_forgot.clear()
_real_save = facts.save_facts


def _enospc(conv_id, fl):
    # ONLY the bundle's write fails — never an empty-list write. The first
    # version failed every save_facts for TARGET, and under the shipped order
    # the first such call is the `[]` CLEAR, so the clear never happened and
    # this case passed against the very code it exists to catch. The failure
    # has to land where the shipped order had already deleted everything.
    if conv_id == TARGET and fl:
        raise OSError(28, "No space left on device")
    return _real_save(conv_id, fl)


_ok_bundle = {"version": portability.BUNDLE_VERSION,
              "facts": [{"text": "the replacement fact"}],
              "summary_state": {}, "episodic": []}
with patch.object(facts, "save_facts", _enospc):
    try:
        portability.import_conversation(_ok_bundle, target_conv_id=TARGET, overwrite=True)
        raised = None
    except OSError as e:
        raised = e
check(raised is not None, "the disk failure surfaced")
check([f["text"] for f in facts.load_facts(TARGET)][:1] == ["fact number 0"]
      and len(facts.load_facts(TARGET)) == 105,
      "and the OLD facts are intact — nothing was cleared ahead of a write that failed")
check(_forgot == [],
      "and the episodic index was not cleared either — it is emptied only after "
      "both atomic writes have landed")

# CONTROL: a clean overwrite still replaces the facts and still clears the
# episodic index, so 3a-3c are not passing because import stopped importing.
_reseed()
_forgot.clear()
res = portability.import_conversation(_ok_bundle, target_conv_id=TARGET, overwrite=True)
check([f["text"] for f in facts.load_facts(TARGET)] == ["the replacement fact"],
      "CONTROL: a clean overwrite REPLACES the 105 facts with the bundle's")
check(_forgot == [TARGET] and res.get("overwrote_existing") is True,
      "CONTROL: and clears the old episodic rows, and says it overwrote")

for p in _stub_retrieval:
    p.stop()

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll surrogate-guard checks passed.")
