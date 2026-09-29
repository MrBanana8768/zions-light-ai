"""
v3.1.9.10: /forget is POINT-IN-TIME, in every store.

Owner decision (2026-09-29): "/forget should never allow things that meet a
regex phrase to never be remembered again, rather it should wipe it from
existing memory and then allow new information to be remembered instead."

  [P1] /forget <text>: what EXISTS NOW is wiped (facts, digests, the chat
       journal); the SAME words said afterwards are remembered normally in
       facts (both write paths), the journal, the digests, the summary state
       and episodic memory; a reboot keeps the new content and removes an old
       restored copy of the journal and the digests
  [P2] /forget (full wipe): everything is wiped; the same words afterwards are
       remembered in every store; a reboot keeps them; a restored pre-wipe
       copy (journal, digests) is removed
  [P3] the source carries no write-time needle filter (a static guard, so a
       filter cannot come back unnoticed in the facts/summary/episodic paths)
  [P4] retire keeps the journal, and a reboot keeps it

Synthetic text only. Failures are monkeypatched, never chmod (the gate runs as
root).

    python test_forget_point_in_time.py
"""

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

_TMP = Path(tempfile.mkdtemp(prefix="compactor-test-pit-"))
STORE = _TMP / "store"
os.environ["COMPACTOR_STORAGE_ROOT"] = str(STORE)
os.environ["COMPACTOR_CONTROL_ROOT"] = str(_TMP / "control")
os.environ["COMPACTOR_PENDING_ROOT"] = str(_TMP / "pending")
os.environ["COMPACTOR_BACKUP_DIR"] = str(_TMP / "backups")
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_TMP / "webui.db")
os.environ["COMPACTOR_BACKUP_LOCAL_STAGING_DIR"] = str(_TMP / "staging")
os.environ["WEBUI_DB_QUARANTINE"] = str(_TMP / "quarantine")

import chatjournal  # noqa: E402
import commands  # noqa: E402
import continuity  # noqa: E402
import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

HERE = Path(__file__).resolve().parent
FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def eq(actual, expected, label):
    if actual == expected:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        FAILED.append(label)


# ---------------------------------------------------------------------------
# Mock episodic store (same shape as test_retrieval.py's), so the SAME
# retrieval.index_exchange / forget_conversation code runs, not a stand-in.
# ---------------------------------------------------------------------------

class _Vec:
    def __init__(self, d):
        self._d = d

    def tolist(self):
        return self._d


class _Embedder:
    def embed(self, texts):
        for t in texts:
            yield _Vec([float(len(t)), 0.0, 1.0])


class _Collection:
    def __init__(self):
        self._store = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, _id in enumerate(ids):
            self._store[_id] = (documents[i], metadatas[i])

    def get(self, where=None, ids=None, include=None):
        if ids is not None:
            matched = [i for i in ids if i in self._store]
        else:
            cid = (where or {}).get("conv_id")
            matched = [i for i, (_, m) in self._store.items() if m.get("conv_id") == cid]
        out = {"ids": matched}
        if include is None or "documents" in include:
            out["documents"] = [self._store[i][0] for i in matched]
        if include is None or "metadatas" in include:
            out["metadatas"] = [self._store[i][1] for i in matched]
        return out

    def delete(self, ids):
        for i in ids:
            self._store.pop(i, None)


def install_episodic():
    retrieval._available = True
    retrieval._embedder = _Embedder()
    retrieval._chroma_collection = _Collection()
    retrieval.reset_vector_cache()


def episodic_docs(conv):
    col = retrieval._chroma_collection
    return [d for (d, m) in col._store.values() if m.get("conv_id") == conv]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NEEDLE = "zebrafruit"


def reset():
    chatjournal.drain(5)
    for d in (STORE, _TMP / "control", _TMP / "pending"):
        shutil.rmtree(d, ignore_errors=True)
    memory.ensure_storage_layout()
    chatjournal._reset_for_tests()
    continuity._pending_intents.clear()
    for name in ("_ledger_write_failed", "_ledger_write_failed_ever", "_ledger_unreadable_ever",
                 "_double_failure_current", "_double_failure_ever"):
        getattr(continuity, name).clear()
    continuity._legacy_stamp.clear()
    install_episodic()


def run_cmd(name, arg, conv):
    ctx = {"clear_all_memory": lambda cid: main._clear_all_memory(cid, source="test"),
           "turn_index": 1, "persona_text": ""}
    out = asyncio.run(commands.handle_command(name, arg, conv, ctx=ctx))
    chatjournal.drain(10)
    return out


def reboot():
    return continuity.reapply_all_ledgers()


def jpath(conv):
    return STORE / "journal" / f"{conv}.jsonl"


def jraw(conv):
    p = jpath(conv)
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def jlines(conv):
    return [json.loads(x) for x in jraw(conv).splitlines() if x]


def journal_say(conv, user, assistant):
    chatjournal.request_meta_ctx.set({"chat_id": None, "message_id": None, "parent_id": None,
                                      "images": {"count": 0, "refs": []}})
    chatjournal.submit(conv, user, assistant, finished=True, truncated=False, holed=False,
                       finish_reason="stop", wipe_generation=memory.current_wipe_generation(conv))
    chatjournal.drain(5)


def old_journal_line(conv, user, assistant, ts="2000-01-01T00:00:00.000Z", epoch=0):
    return json.dumps({
        "v": 1, "conv_id": conv, "chat_id": None, "message_id": "restored-old", "parent_id": None,
        "request_id": "r", "ts": ts, "epoch": epoch, "user": user, "assistant": assistant,
        "finish_reason": "stop", "partial": False, "images": {"count": 0, "refs": []},
        "sha256": chatjournal.pair_sha(user, assistant)}) + "\n"


def digest_say(conv, key, her, reply):
    ok = asyncio.run(continuity.generate_digest(
        conv, tok=None, her_text=her, reply_text=reply, unit_number=1, key=key,
        epoch_at_submit=continuity.current_ledger_epoch(conv), vllm_url=None, model=None))
    return ok


async def _stub_extract(client, vllm_url, model, user_msg, assistant_msg, existing, conv_id=None):
    return [f"She grows {NEEDLE} in the back garden ({user_msg[:8]})"]


def extract_say(conv, user):
    real = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = _stub_extract
    try:
        return asyncio.run(facts.record_facts_for_exchange(
            conv, None, "http://stub", "m", user, "Noted, that sounds lovely.", 5))
    finally:
        facts.extract_facts_from_exchange = real


def fact_texts(conv):
    return [f["text"] for f in facts.load_facts(conv)]


def summary_text(conv):
    st = summarizer.load_state(conv)
    return json.dumps(st)


def seed_summary(conv, text):
    st = summarizer.load_state(conv)
    st.setdefault("l1", [])
    st["l1"] = [{"text": text, "first_turn": 1, "last_turn": 4}]
    summarizer.save_state(conv, st)


def digest_store(conv):
    return continuity._read_digest_store(conv)


# ===========================================================================
print("\n[P1] /forget <text> is point-in-time")
reset()
conv = "pit-1"
facts.save_facts(conv, [
    {"text": f"She grows {NEEDLE} trees", "added_turn": 1, "last_used": 1, "pin": False},
    {"text": "She likes green tea", "added_turn": 2, "last_used": 2, "pin": False},
])
journal_say(conv, f"I planted a {NEEDLE} tree", "That sounds lovely.")
journal_say(conv, "what about the gate", "It is blue.")
digest_say(conv, "old-unit", f"I planted a {NEEDLE} tree", "That sounds lovely and warm today.")
out = run_cmd("forget", NEEDLE, conv)
check(out.startswith("Forgot 1 fact(s)"), f"the reply reports the fact: {out!r}")
eq(fact_texts(conv), ["She likes green tea"], "facts: the matching fact is wiped now, the other kept")
check(NEEDLE not in jraw(conv).lower() and len(jlines(conv)) == 2, "journal: matching exchange tombstoned now")
check(digest_store(conv)["old-unit"]["suppressed"], "digest: the existing matching unit is suppressed now")

# ---- AFTER the forget she says the matching thing again: it IS remembered ----
r = run_cmd("remember", f"She waters the {NEEDLE} every morning", conv)
check(r.startswith("Remembered"), f"facts (/remember): stored: {r!r}")
n_new = extract_say(conv, f"my {NEEDLE} is flowering")
eq(n_new, 1, "facts (extraction path): a new fact containing the forgotten word is stored")
check(any(NEEDLE in t for t in fact_texts(conv)), "facts: the word is in the store again")
journal_say(conv, f"more {NEEDLE} please", f"Sure, {NEEDLE} it is.")
last = jlines(conv)[-1]
check(not last.get("redacted") and NEEDLE in last["assistant"], "journal: the new exchange is written as it is")
check(digest_say(conv, "new-unit", f"tell me about {NEEDLE} again",
                 f"Of course, {NEEDLE} grows best in the shade of the wall."),
      "digest: a new unit with the word is written (not suppressed)")
seed_summary(conv, f"She talked about her {NEEDLE} again this week.")
check(NEEDLE in summary_text(conv), "summary state: text with the word is stored (no filter)")
check(retrieval.index_exchange(conv, 9, f"do you remember my {NEEDLE}?", f"Yes, your {NEEDLE} tree."),
      "episodic: an exchange with the word is indexed")
check(any(NEEDLE in d for d in episodic_docs(conv)), "episodic: and it is in the store")

# ---- reboot: the new content survives, an old restored copy does not ----
with open(jpath(conv), "a", encoding="utf-8") as f:
    f.write(old_journal_line(conv, f"I planted a {NEEDLE} tree", "restored"))
store = digest_store(conv)
store["restored-unit"] = {"headline": f"about {NEEDLE}", "full": f"about {NEEDLE}", "epoch": 0,
                          "unit_number": 9, "suppressed": False}      # a restored old store: no `created`
continuity._write_digest_store(conv, store)
res = reboot()
check(any(NEEDLE in t for t in fact_texts(conv)), "reboot: the newly remembered facts survive")
ls = jlines(conv)
eq([bool(x.get("redacted")) for x in ls if x.get("message_id") == "restored-old"], [True],
   "reboot: the restored OLD journal line is a tombstone")
check(any((not x.get("redacted")) and NEEDLE in x.get("assistant", "") for x in ls),
      "reboot: the post-forget journal line is intact")
v = continuity.visible(conv)
check(not v["new-unit"]["suppressed"] and NEEDLE in v["new-unit"]["full"], "reboot: the post-forget digest is intact")
check(v["restored-unit"]["suppressed"], "reboot: the restored old digest is suppressed")
check(NEEDLE in summary_text(conv), "reboot: the summary state is untouched")
check(any(NEEDLE in d for d in episodic_docs(conv)), "reboot: the episodic exchange is untouched")
led = continuity._read_ledger(conv)
check(led["format"] == 2 and all(n.get("cutoff") for n in led["targeted"]), "ledger: format 2, every needle has a cutoff")

# ===========================================================================
print("\n[P2] /forget (full wipe) is point-in-time")
reset()
conv = "pit-2"
facts.save_facts(conv, [{"text": f"She grows {NEEDLE}", "added_turn": 1, "last_used": 1, "pin": False}])
journal_say(conv, f"I planted a {NEEDLE} tree", "Lovely.")
digest_say(conv, "old-unit", f"I planted a {NEEDLE} tree", "That sounds lovely and warm today.")
seed_summary(conv, f"She talked about {NEEDLE}.")
retrieval.index_exchange(conv, 1, f"my {NEEDLE}", f"your {NEEDLE}")
old_journal_copy = jraw(conv)
old_digest_copy = dict(digest_store(conv))
out = run_cmd("forget", "", conv)
eq(fact_texts(conv), [], "wipe: facts gone")
check(not jpath(conv).exists(), "wipe: journal gone")
check(not memory.digests_path(conv).exists(), "wipe: digest store gone")
check(not summarizer.summary_path(conv).exists(), "wipe: summary state gone")
eq(episodic_docs(conv), [], "wipe: episodic gone")
led = continuity._read_ledger(conv)
check(bool(led["wipe_at"]) and led["wipe_epoch"] == led["epoch"] >= 1, "ledger: wipe cutoff recorded")

# the same words afterwards, in every store
r = run_cmd("remember", f"She grows {NEEDLE} again", conv)
n_new = extract_say(conv, f"my {NEEDLE} is back")
check(n_new == 1 and any(NEEDLE in t for t in fact_texts(conv)), "wipe -> facts: remembered again")
journal_say(conv, f"I planted a {NEEDLE} tree", "Lovely, again.")
check(digest_say(conv, "old-unit", f"I planted a {NEEDLE} tree", "That sounds lovely and warm today."),
      "wipe -> digest: the SAME unit key (same words) is written again, not blocked")
seed_summary(conv, f"She talked about {NEEDLE} once more.")
retrieval.index_exchange(conv, 1, f"my {NEEDLE}", f"your {NEEDLE}")
check(any(NEEDLE in d for d in episodic_docs(conv)), "wipe -> episodic: indexed again (same content id as before)")

# reboot with a restored pre-wipe backup of the journal and the digest store
with open(jpath(conv), "a", encoding="utf-8") as f:
    f.write(old_journal_copy)
store = dict(digest_store(conv))
store["old-restored"] = old_digest_copy["old-unit"]                    # epoch 0: stamped before the wipe
continuity._write_digest_store(conv, store)
reboot()
ls = jlines(conv)
eq([x["assistant"] for x in ls], ["Lovely, again."], "reboot: the restored pre-wipe journal line is purged, the post-wipe one survives")
v = continuity.visible(conv)
eq(sorted(v), ["old-unit"], "reboot: the restored pre-wipe digest is purged, the post-wipe one survives")
check(any(NEEDLE in t for t in fact_texts(conv)), "reboot: post-wipe facts survive")
check(NEEDLE in summary_text(conv), "reboot: post-wipe summary survives")
check(any(NEEDLE in d for d in episodic_docs(conv)), "reboot: post-wipe episodic survives")

# ===========================================================================
print("\n[P3] no write-time needle filter in the source")
# The words that made a filter: the ledger's needle list. Only continuity.py
# (the ledger and the boot re-apply, cutoff-bounded), chatjournal.py (redact /
# reapply, cutoff-bounded), commands.py (the /forget command itself) may use it.
for name in ("facts.py", "summarizer.py", "retrieval.py", "persona.py", "backfill.py", "dedup.py"):
    src = (HERE / name).read_text(encoding="utf-8")
    check(not re.search(r"\bsuppress_needle\b|\bapply_needles\b|_needle_hit|\"targeted\"|record_targeted_forget", src),
          f"{name}: no ledger-needle check on its write path")
jsrc = (HERE / "chatjournal.py").read_text(encoding="utf-8")
a = jsrc.index("def append_record")
b = jsrc.index("def _run_job")
check("needle" not in jsrc[a:b].lower() and "_hits" not in jsrc[a:b], "chatjournal.append_record: no needle check")
csrc = (HERE / "continuity.py").read_text(encoding="utf-8")
a = csrc.index("async def generate_digest")
b = csrc.index("def suppress_needle")
check("_needle_hit" not in csrc[a:b].replace("v3.1.9.10", ""), "continuity.generate_digest: no needle check")
msrc = (HERE / "main.py").read_text(encoding="utf-8")
check("apply_needles" in msrc and "journal_ledger" not in msrc, "main.py: only the fail-closed band gate remains")

# ===========================================================================
print("\n[P4] retire keeps the journal, and a reboot keeps it")
reset()
src, dst = "pit-4-src", "pit-4-dst"
journal_say(src, "we talked about the tea", "It was warm.")
facts.save_facts(src, [{"text": "She likes pears", "added_turn": 1, "last_used": 1, "pin": False}])
dry = asyncio.run(commands.handle_command("retire", src, dst, ctx={}))
marker = f"/retire {src} apply "
code = dry.split(marker, 1)[1].split()[0] if marker in dry else "x"
out = asyncio.run(commands.handle_command("retire", f"{src} apply {code}", dst, ctx={}))
chatjournal.drain(5)
check(jpath(src).is_file() and "we talked about the tea" in jraw(src), "retire: the journal is kept, words intact")
eq(fact_texts(src), [], "retire: its other effects are unchanged (facts moved out)")
reboot()
check(jpath(src).is_file() and "we talked about the tea" in jraw(src), "reboot: still there after the retire's epoch bump")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  -", f)
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(1)
print("ALL OK")
shutil.rmtree(_TMP, ignore_errors=True)
