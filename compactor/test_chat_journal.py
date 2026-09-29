"""
v3.1.9.10 per-message chat journal (compactor/chatjournal.py), end to end.

Synthetic text only (the repo is public). Every write goes to a temp
STORAGE_ROOT / CONTROL_ROOT. Permission failures are simulated by
monkeypatching the call, never by chmod (the gate runs as root).

What is pinned, and by which section:

  [J1]  a completed NON-STREAMING reply writes exactly one line, with every
        field, image REFERENCES only (no data URL, no query string)
  [J2]  the same on the STREAMING path, and cut / disconnected / length
        replies are written with partial=true and their finish_reason
  [J3]  exactly one line per reply (three replies -> three lines, in order)
  [J4]  skips: self-test conv, slash command, empty reply, task traffic,
        no user text, disabled, disk pressure
  [J5]  a journal failure never touches the reply, is counted, and no text is
        logged; the file is opened O_APPEND and fsynced
  [F1]  /forget <text>: the text is gone from the journal (tombstones), also
        when NO fact matched; a restart (ledger re-apply) keeps it gone; a
        RESTORED old journal is re-redacted; a later reply that contains the
        needle is tombstoned at write time
  [F2]  /forget (full): the file is removed; a restored old journal is purged
        by the epoch; an in-flight reply from before the wipe is discarded
  [F3]  admin DELETE .../facts and /retire also remove the journal
  [F4]  fail closed: an unreadable ledger, or a forget whose ledger write
        failed, journals nothing; a failed rewrite is reported and healed at
        the next boot
  [H1]  /health/full carries chat_journal {enabled,last_write_at,lines_written,errors}
  [B1]  backup archives the journal and verify_backup accepts a .jsonl store
  [S1]  scripts/chat-journal.py: stats / verify / compare against a synthetic
        OpenWebUI database, and its pair hash equals the module's

    python test_chat_journal.py
"""

import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

_TMP = Path(tempfile.mkdtemp(prefix="compactor-test-chatjournal-"))
STORE = _TMP / "store"
CONTROL = _TMP / "control"
PENDING = _TMP / "pending"
BACKUPS = _TMP / "backups"
os.environ["COMPACTOR_STORAGE_ROOT"] = str(STORE)
os.environ["COMPACTOR_CONTROL_ROOT"] = str(CONTROL)
os.environ["COMPACTOR_PENDING_ROOT"] = str(PENDING)
os.environ["COMPACTOR_BACKUP_DIR"] = str(BACKUPS)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_TMP / "webui.db")
os.environ["COMPACTOR_BACKUP_LOCAL_STAGING_DIR"] = str(_TMP / "staging")
os.environ["WEBUI_DB_QUARANTINE"] = str(_TMP / "quarantine")

import chatjournal  # noqa: E402
import commands  # noqa: E402
import continuity  # noqa: E402
import facts  # noqa: E402
import health  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None
retrieval.conversation_doc_count = lambda conv_id: 0

client = TestClient(main.app, client=("127.0.0.1", 12360), raise_server_exceptions=False)

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
# Stubs (same shape as test_truncated_tail.py)
# ---------------------------------------------------------------------------

class _StubResponse:
    status_code = 200
    text = ""

    def __init__(self, content, finish_reason):
        self._c, self._fr = content, finish_reason

    def json(self):
        return {"id": "stub", "choices": [{"index": 0, "finish_reason": self._fr,
                                           "message": {"role": "assistant", "content": self._c}}]}


class _StubVLLM:
    reply = ""
    finish_reason = "stop"

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        return _StubResponse(_StubVLLM.reply, _StubVLLM.finish_reason)

    async def aclose(self):
        pass


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _content(t) -> bytes:
    return _sse({"choices": [{"delta": {"content": t}}]})


def _finish(r) -> bytes:
    return _sse({"choices": [{"delta": {}, "finish_reason": r}]})


DONE = b"data: [DONE]\n\n"


class _StubStreamResp:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    async def aread(self):
        return b""

    async def aiter_raw(self):
        for c in self._chunks:
            yield c


class _StubStreamCM:
    def __init__(self, chunks):
        self._r = _StubStreamResp(chunks)

    async def __aenter__(self):
        return self._r

    async def __aexit__(self, *exc):
        return False


class _StubStreamVLLM:
    chunks: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None, **kw):
        return _StubStreamCM(list(_StubStreamVLLM.chunks))

    async def aclose(self):
        pass


async def _noop_tail(*a, **k):
    return None


async def _noop_rollup(*a, **k):
    return None


class _Logs(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.text: list[str] = []

    def emit(self, record):
        try:
            self.text.append(record.getMessage())
        except Exception:
            pass


@contextlib.contextmanager
def capture_logs():
    h = _Logs()
    targets = [logging.getLogger(), logging.getLogger("compactor")]
    prev = [t.level for t in targets]
    for t in targets:
        t.addHandler(h)
        t.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        for t, lv in zip(targets, prev):
            t.removeHandler(h)
            t.setLevel(lv)


HEADERS = {"X-OpenWebUI-Chat-Id": "chat-0001", "X-OpenWebUI-Message-Id": "msg-asst-1"}


def _messages(user_text, *, image=False):
    msgs = [{"role": "user", "content": "an earlier question"},
            {"role": "assistant", "content": "an earlier answer"}]
    if image:
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAABBBBCCCC"}},
            {"type": "image_url", "image_url": {"url": "/api/v1/files/xyz789/content?token=SECRETTOKEN"}},
        ]})
    else:
        msgs.append({"role": "user", "content": user_text})
    return msgs


def _base_patches():
    return [patch.object(main, "_async_tail", _noop_tail),
            patch.object(main, "_rollup_hierarchy", _noop_rollup)]


def post_nonstream(conv, user_text, reply, *, finish="stop", image=False, headers=None, extra=None):
    _StubVLLM.reply, _StubVLLM.finish_reason = reply, finish
    body = {"model": "m", "messages": _messages(user_text, image=image), "stream": False}
    body.update(extra or {})
    with contextlib.ExitStack() as st:
        st.enter_context(patch.object(main.httpx, "AsyncClient", _StubVLLM))
        for p in _base_patches():
            st.enter_context(p)
        r = client.post("/v1/chat/completions", json=body,
                        headers={"X-Conversation-Id": conv, **(HEADERS if headers is None else headers)})
    chatjournal.drain(10)
    return r


def post_stream(conv, user_text, chunks, *, headers=None):
    _StubStreamVLLM.chunks = chunks
    body = {"model": "m", "messages": _messages(user_text), "stream": True}
    with contextlib.ExitStack() as st:
        st.enter_context(patch.object(main.httpx, "AsyncClient", _StubStreamVLLM))
        for p in _base_patches():
            st.enter_context(p)
        r = client.post("/v1/chat/completions", json=body,
                        headers={"X-Conversation-Id": conv, **(HEADERS if headers is None else headers)})
    chatjournal.drain(10)
    return r


def jfile(conv) -> Path:
    return STORE / "journal" / f"{conv}.jsonl"


def lines(conv) -> list[dict]:
    p = jfile(conv)
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


def raw(conv) -> str:
    p = jfile(conv)
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def reset():
    chatjournal.drain(5)
    for d in (STORE, CONTROL, PENDING):
        shutil.rmtree(d, ignore_errors=True)
    memory.ensure_storage_layout()
    chatjournal._reset_for_tests()
    getattr(continuity, "_pending_intents", {}).clear()
    for name in ("_ledger_write_failed", "_ledger_write_failed_ever", "_ledger_unreadable_ever",
                 "_double_failure_current", "_double_failure_ever"):
        getattr(continuity, name, set()).clear()


def run_cmd(name, arg, conv):
    ctx = {"clear_all_memory": lambda cid: main._clear_all_memory(cid, source="test"),
           "turn_index": 1, "persona_text": ""}
    out = asyncio.run(commands.handle_command(name, arg, conv, ctx=ctx))
    chatjournal.drain(10)
    return out


def restart_reapply():
    """What boot does (lifespan): re-apply every ledger."""
    return continuity.reapply_all_ledgers()


REPLY = "The garden gate was painted blue in the spring, and the tea was warm."

# ===========================================================================
print("\n[J1] non-streaming: one line, every field, references only")
reset()
conv = "conv-j1"
with capture_logs() as logs:
    r = post_nonstream(
        conv, "what colour is the gate?", REPLY, image=True,
        extra={"metadata": {"parent_message_id": "msg-user-1",
                            "files": [{"id": "file-abc", "name": "photo.png"}]}},
    )
eq(r.status_code, 200, "the reply is 200")
eq(r.json()["choices"][0]["message"]["content"], REPLY, "the reply body is untouched")
ls = lines(conv)
eq(len(ls), 1, "exactly one line")
rec = ls[0] if ls else {}
eq(rec.get("v"), 1, "schema version 1")
eq(rec.get("conv_id"), conv, "conv_id")
eq(rec.get("chat_id"), "chat-0001", "OpenWebUI chat id")
eq(rec.get("message_id"), "msg-asst-1", "OpenWebUI message id")
eq(rec.get("parent_id"), "msg-user-1", "parent message id from the payload")
check(bool(rec.get("request_id")), "request id present")
check(str(rec.get("ts", "")).endswith("Z") and "T" in rec.get("ts", ""), "UTC ISO timestamp")
eq(rec.get("epoch"), 0, "ledger epoch stamped")
eq((rec.get("user") or "").strip(), "what colour is the gate?",
   "user text (the original request's text, not the image-stripped one)")
eq(rec.get("assistant"), REPLY, "assistant text")
eq(rec.get("finish_reason"), "stop", "finish_reason")
eq(rec.get("partial"), False, "not partial")
eq(rec.get("images", {}).get("count"), 2, "image count")
refs = rec.get("images", {}).get("refs", [])
check("file-abc" in refs and "photo.png" in refs, "file id and name kept as references")
check("/api/v1/files/xyz789/content" in refs, "URL path kept")
check(not any("SECRETTOKEN" in x or "?" in x for x in refs), "query string (token) dropped")
check("data:image" not in raw(conv) and "AAAABBBB" not in raw(conv), "no data URL / bytes in the file")
eq(rec.get("sha256"), chatjournal.pair_sha(rec.get("user", ""), rec.get("assistant", "")), "sha256 of the pair")
check(oct(jfile(conv).stat().st_mode & 0o777) == oct(0o600), "file mode 0600 (her words)")
check(not any("what colour" in t or "gate was painted" in t for t in logs.text),
      "no journal text in any log line")

# ===========================================================================
print("\n[J2] streaming path, and cut replies")
reset()
conv = "conv-j2"
r = post_stream(conv, "tell me about the gate", [_content("The gate is "), _content("blue. "),
                                                  _content("It was painted in spring."),
                                                  _finish("stop"), DONE])
eq(r.status_code, 200, "stream ok")
ls = lines(conv)
eq(len(ls), 1, "streaming: exactly one line")
eq(ls[0]["assistant"] if ls else None, "The gate is blue. It was painted in spring.", "assembled reply")
eq((ls[0]["finish_reason"], ls[0]["partial"]) if ls else None, ("stop", False), "finish stop, not partial")
eq((ls[0]["chat_id"], ls[0]["message_id"]) if ls else None, ("chat-0001", "msg-asst-1"),
   "ids reach the streaming tail (contextvar survives the generator)")
# Stop pressed: no finish_reason, no [DONE]
reset()
post_stream(conv, "and the fence?", [_content("The fence is white and it was cut off mid-wo")])
ls = lines(conv)
eq(len(ls), 1, "cut stream: one line")
eq((ls[0]["partial"], ls[0]["finish_reason"]) if ls else None, (True, "incomplete"),
   "cut stream: partial=true, finish_reason=incomplete")
check(ls and ls[0]["assistant"].endswith("mid-wo"), "cut stream: the text she saw, verbatim")
# generation ceiling
reset()
post_stream(conv, "and the roof?", [_content("The roof is red. And then"), _finish("length"), DONE])
ls = lines(conv)
eq((ls[0]["partial"], ls[0]["finish_reason"]) if ls else None, (True, "length"),
   "ceiling: partial=true, finish_reason=length")
# non-streaming length
reset()
post_nonstream(conv, "and the door?", "The door is green. And th", finish="length")
ls = lines(conv)
eq((ls[0]["partial"], ls[0]["finish_reason"]) if ls else None, (True, "length"),
   "non-stream length: partial=true")
# a reply the memory tail declines (degenerate) is still journaled
reset()
post_nonstream(conv, "and the wall?", "━" * 400)
eq(len(lines(conv)), 1, "a reply the memory policy declines is still journaled (mirrors the client)")

# ===========================================================================
print("\n[J3] one line per reply, in order")
reset()
conv = "conv-j3"
for i in range(3):
    post_nonstream(conv, f"question {i}", f"answer {i} " + REPLY)
ls = lines(conv)
eq(len(ls), 3, "three replies, three lines (not more)")
eq([x["user"] for x in ls], ["question 0", "question 1", "question 2"], "in reply order")
post_stream(conv, "question 3", [_content("answer 3"), _finish("stop"), DONE])
eq(len(lines(conv)), 4, "a streamed reply adds exactly one more")
eq(chatjournal.health_snapshot()["lines_written"], 4, "the counter agrees")

# ===========================================================================
print("\n[J4] skips")
reset()
r = post_nonstream("__selftest_oneshot_abcd1234__", "selftest question", REPLY)
eq(r.status_code, 200, "selftest conv still answered")
check(not jfile("__selftest_oneshot_abcd1234__").exists(), "selftest conversation: no journal")
r = post_nonstream("conv-j4", "/list-facts", "x")
eq(r.status_code, 200, "slash command answered")
check(not jfile("conv-j4").exists(), "slash command: no journal")
post_nonstream("conv-j4", "a question", "")
check(not jfile("conv-j4").exists(), "empty reply: no journal")
post_nonstream("conv-j4", "a question", "   \n")
check(not jfile("conv-j4").exists(), "whitespace reply: no journal")
post_stream("conv-j4", "a question", [])
check(not jfile("conv-j4").exists(), "empty stream (failed request): no journal")
with patch.object(main, "_is_repeat_task_traffic", lambda c, m: True):
    post_nonstream("conv-j4", "generate a title", "A Title")
check(not jfile("conv-j4").exists(), "OpenWebUI task traffic: no journal")
post_nonstream("conv-j4", "   ", "an answer to nothing")
check(not jfile("conv-j4").exists(), "no user text: no journal")
with patch.object(chatjournal, "ENABLED", False):
    post_nonstream("conv-j4", "a question", REPLY)
    check(not jfile("conv-j4").exists(), "COMPACTOR_CHAT_JOURNAL off: no journal")
    eq(chatjournal.health_snapshot()["enabled"], False, "health says disabled")
import degrade  # noqa: E402
with patch.object(degrade, "guard", lambda *a, **k: False):
    post_nonstream("conv-j4", "a question", REPLY)
    check(not jfile("conv-j4").exists(), "disk pressure pause: no journal")
post_nonstream("conv-j4", "a question", REPLY)
eq(len(lines("conv-j4")), 1, "control: the same request outside those conditions IS journaled")

# ===========================================================================
print("\n[J5] failure never touches the reply; fsync; O_APPEND")
reset()
conv = "conv-j5"


def _boom(path, line):
    raise OSError(28, "No space left on device")


with capture_logs() as logs, patch.object(chatjournal, "_append_line", _boom):
    r = post_nonstream(conv, "a private question about pears", REPLY)
eq(r.status_code, 200, "reply is 200 although the journal write failed")
eq(r.json()["choices"][0]["message"]["content"], REPLY, "reply body intact")
eq(chatjournal.health_snapshot()["errors"], 1, "the failure is counted")
eq(chatjournal.health_snapshot()["lines_written"], 0, "and nothing counted as written")
check(any("append failed (OSError)" in t for t in logs.text), "logged by exception TYPE")
check(not any("pears" in t or "garden" in t for t in logs.text), "the failure log carries no text")
# a submit() that itself blows up must not raise into the reply path either
with patch.object(chatjournal._EXEC, "submit", side_effect=RuntimeError("boom")):
    r = post_nonstream(conv, "a second question", REPLY)
eq(r.status_code, 200, "an executor failure does not fail the reply")
# fsync + O_APPEND
reset()
calls = {"fsync": 0, "flags": []}
real_fsync, real_open = os.fsync, os.open


def _fs(fd):
    # Count only fsyncs of the journal FILE (not the directory fsync done when
    # the file is created). /proc is Linux; elsewhere count everything.
    try:
        target = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        target = ".jsonl"
    if target.endswith(".jsonl"):
        calls["fsync"] += 1
    return real_fsync(fd)


def _op(path, flags, *a, **k):
    if str(path).endswith(".jsonl"):
        calls["flags"].append(flags)
    return real_open(path, flags, *a, **k)


with patch.object(chatjournal.os, "fsync", _fs), patch.object(chatjournal.os, "open", _op):
    post_nonstream(conv, "q1", REPLY)
    first = calls["fsync"]
    post_nonstream(conv, "q2", REPLY)
check(first >= 1, "fsync is called for the first line")
check(calls["fsync"] - first >= 1, "and for every later line")
check(calls["flags"] and all(f & os.O_APPEND for f in calls["flags"]), "opened with O_APPEND")
# a huge field is clipped, and the line stays consistent
reset()
with patch.object(chatjournal, "MAX_FIELD_CHARS", 50):
    post_nonstream("conv-j5b", "u" * 200, "a" * 120)
ls = lines("conv-j5b")
check(ls and len(ls[0]["user"]) == 50 and len(ls[0]["assistant"]) == 50, "fields clipped to MAX_FIELD_CHARS")
eq(ls[0]["truncated"] if ls else None, ["user", "assistant"], "and flagged")
eq(ls[0]["sha256"] if ls else None, chatjournal.pair_sha("u" * 50, "a" * 50), "sha is over the stored text")

# ===========================================================================
print("\n[F1] /forget <text>")
reset()
conv = "conv-f1"
NEEDLE = "zebrafruit"
post_nonstream(conv, "I planted a zebrafruit tree", "Zebrafruit likes shade. " + REPLY)
post_nonstream(conv, "what about the gate", REPLY)
post_nonstream(conv, "and my ZebraFruit again", "Noted.")
before = raw(conv)
check(NEEDLE in before.lower(), "fixture: the needle is in the journal")
facts.save_facts(conv, [])  # NO fact contains the needle: it lives only in the journal
out = run_cmd("forget", NEEDLE, conv)
check("chat-journal" in out, f"the reply says so: {out!r}")
eq(out.startswith("No facts matched"), True, "no fact matched, and the journal was still cleared")
ls = lines(conv)
eq(len(ls), 3, "the exchanges stay (tombstones), nothing else is lost")
check(NEEDLE not in raw(conv).lower(), "the needle is not in the journal")
eq([bool(x.get("redacted")) for x in ls], [True, False, True], "only the matching exchanges are tombstoned")
red = ls[0]
check(red["user"] == "" and red["assistant"] == "" and red["sha256"] is None, "tombstone: words and hash gone")
check(red["message_id"] == "msg-asst-1" and red["ts"] and red["epoch"] == 0, "tombstone: ids and ts kept")
eq(ls[1]["user"], "what about the gate", "an unrelated exchange is byte-for-byte intact")
# a facts match too
reset()
post_nonstream(conv, "I planted a zebrafruit tree", REPLY)
facts.save_facts(conv, [{"text": "She planted a zebrafruit tree"}])
out = run_cmd("forget", NEEDLE, conv)
check(out.startswith("Forgot 1 fact(s)") and "chat-journal" in out, f"facts + journal: {out!r}")
check(NEEDLE not in raw(conv).lower(), "facts path: needle gone from the journal too")
# restart: re-apply on boot keeps it gone
snapshot_after_forget = raw(conv)
res = restart_reapply()
eq(raw(conv), snapshot_after_forget, "a restart (ledger re-apply) changes nothing further")
# a RESTORED older journal (backup from before the /forget) is re-redacted at boot
jfile(conv).write_text(json.dumps({
    "v": 1, "conv_id": conv, "chat_id": None, "message_id": "old-1", "parent_id": None,
    "request_id": "r", "ts": "2026-09-01T00:00:00.000Z", "epoch": 0,
    "user": "I planted a zebrafruit tree", "assistant": "ok", "finish_reason": "stop", "partial": False,
    "images": {"count": 0, "refs": []},
    "sha256": chatjournal.pair_sha("I planted a zebrafruit tree", "ok")}) + "\n", encoding="utf-8")
check(NEEDLE in raw(conv), "fixture: the restored journal has the forgotten text again")
restart_reapply()
check(NEEDLE not in raw(conv).lower(), "restart after a restore: the forgotten text does not come back")
eq(lines(conv)[0].get("redacted"), True, "...it is a tombstone")
# a later reply that contains the needle is tombstoned AT WRITE TIME
post_nonstream(conv, "more zebrafruit please", "Sure, zebrafruit it is.")
check(NEEDLE not in raw(conv).lower(), "a NEW exchange containing the forgotten text is tombstoned when written")
eq(lines(conv)[-1].get("redacted"), True, "...as a tombstone, not dropped")
# an unparseable line holding the needle is dropped, other junk kept
with open(jfile(conv), "a", encoding="utf-8") as f:
    f.write("garbage with zebrafruit inside\nother garbage\n")
restart_reapply()
check("zebrafruit" not in raw(conv).lower() and "other garbage" in raw(conv),
      "an unparseable line with the needle is dropped; others are left alone")

# ===========================================================================
print("\n[F2] /forget (full wipe)")
reset()
conv = "conv-f2"
post_nonstream(conv, "first thing", "first answer " + REPLY)
post_nonstream(conv, "second thing", "second answer " + REPLY)
old_copy = raw(conv)
out = run_cmd("forget", "", conv)
check("chat-journal" in out and "2" in out, f"reply counts the journal lines: {out!r}")
check(not jfile(conv).exists(), "the journal file is gone")
restart_reapply()
check(not jfile(conv).exists(), "and a restart does not bring it back")
# the ledger epoch moved; a restored older journal is purged by epoch at boot
led = continuity.journal_ledger(conv)
check(led is not None and led[0] >= 1, "the wipe bumped the ledger epoch")
jfile(conv).parent.mkdir(parents=True, exist_ok=True)
jfile(conv).write_text(old_copy, encoding="utf-8")
post_nonstream(conv, "third thing", "third answer " + REPLY)   # written after the wipe, epoch >= 1
restart_reapply()
ls = lines(conv)
eq([x["user"] for x in ls], ["third thing"], "restored pre-wipe lines are purged; the post-wipe line survives")
check("first thing" not in raw(conv) and "second thing" not in raw(conv), "old words not in the file")
# an in-flight reply: generation captured BEFORE the wipe, write attempted AFTER
reset()
conv = "conv-f2b"
post_nonstream(conv, "seed", REPLY)
gen0 = memory.current_wipe_generation(conv)
asyncio.run(main._clear_all_memory(conv, source="test"))
check(not jfile(conv).exists(), "wipe removed the journal")
chatjournal.request_meta_ctx.set({"chat_id": None, "message_id": None, "parent_id": None,
                                  "images": {"count": 0, "refs": []}})
chatjournal.submit(conv, "in flight question", "in flight answer", finished=True, truncated=False,
                   holed=False, finish_reason="stop", wipe_generation=gen0)
chatjournal.drain(5)
check(not jfile(conv).exists(), "an in-flight reply from BEFORE the wipe cannot resurrect the file")
chatjournal.submit(conv, "fresh question", "fresh answer", finished=True, truncated=False,
                   holed=False, finish_reason="stop", wipe_generation=memory.current_wipe_generation(conv))
chatjournal.drain(5)
eq(len(lines(conv)), 1, "a reply submitted after the wipe is journaled")

# ===========================================================================
print("\n[F3] admin wipe and /retire")
reset()
conv = "conv-f3"
post_nonstream(conv, "admin wipe me", REPLY)
r = client.delete(f"/admin/conversations/{conv}/facts")
eq(r.status_code, 200, "admin DELETE .../facts ok")
check(not jfile(conv).exists(), "admin wipe removed the journal")
eq(r.json().get("forgotten_journal"), 1, "and reports the lines")
# /retire
reset()
src, dst = "conv-f3-src", "conv-f3-dst"
post_nonstream(src, "retire me", REPLY)
facts.save_facts(src, [{"text": "She likes pears"}])
dry = asyncio.run(commands.handle_command("retire", src, dst, ctx={}))
marker = f"/retire {src} apply "
check(marker in dry, "retire dry run offers a code")
code = dry.split(marker, 1)[1].split()[0] if marker in dry else "x"
asyncio.run(commands.handle_command("retire", f"{src} apply {code}", dst, ctx={}))
chatjournal.drain(5)
check(not jfile(src).exists(), "/retire removed the retired conversation's journal")

# ===========================================================================
print("\n[F4] fail closed; failed rewrite")
reset()
conv = "conv-f4"
post_nonstream(conv, "one", REPLY)
with patch.object(continuity, "_read_ledger", side_effect=continuity.LedgerUnreadable("test")):
    post_nonstream(conv, "two, ledger unreadable", REPLY)
eq(len(lines(conv)), 1, "unreadable ledger: the exchange is NOT journaled (fail closed)")
eq(chatjournal.health_snapshot()["skipped"], 1, "...and counted as skipped")
post_nonstream(conv, "three", REPLY)
eq(len(lines(conv)), 2, "the ledger back: journaling resumes")
# a forget whose ledger write fails leaves a pending intent: journaling stays closed
reset()
conv = "conv-f4b"
post_nonstream(conv, "before the forget, mentions kiwi", REPLY)
with patch.object(continuity, "_write_ledger", side_effect=OSError(13, "denied")):
    out = run_cmd("forget", "kiwi", conv)
check("kiwi" not in raw(conv).lower(), "the journal was redacted even though the ledger write failed")
post_nonstream(conv, "after, mentions kiwi again", REPLY)
check("kiwi" not in raw(conv).lower() and len(lines(conv)) == 1,
      "pending forget: the next exchange is not journaled (no needle can slip in)")
# failed rewrite: reported, unchanged, healed at boot
reset()
conv = "conv-f4c"
post_nonstream(conv, "I love mango", REPLY)
with patch.object(chatjournal, "_rewrite", side_effect=OSError(28, "full")):
    out = run_cmd("forget", "mango", conv)
check("could not rewrite the chat journal" in out, f"the reply says the journal was not cleared: {out!r}")
check("mango" in raw(conv), "fixture: the rewrite failed, so the words are still there")
eq(chatjournal.health_snapshot()["errors"], 1, "counted as an error")
restart_reapply()
check("mango" not in raw(conv), "the next boot's re-apply clears it (the needle is in the ledger)")

# ===========================================================================
print("\n[H1] /health/full")
reset()
post_nonstream("conv-h1", "hello", REPLY)


async def _vllm_ok(url):
    return {"ok": True, "latency_ms": 1.0, "models": ["m"], "error": None}


health.probe_vllm = _vllm_ok
rep = asyncio.run(health.gather_health_full("http://fake", 4096))
cj = rep.get("chat_journal", {})
check({"enabled", "last_write_at", "lines_written", "errors"} <= set(cj), f"chat_journal block: {sorted(cj)}")
eq((cj.get("enabled"), cj.get("lines_written"), cj.get("errors")), (True, 1, 0), "values")
check(str(cj.get("last_write_at", "")).endswith("Z"), "last_write_at is a UTC timestamp")
check("conv-h1" not in json.dumps(rep.get("chat_journal")), "no conversation id in the block")

# ===========================================================================
print("\n[B1] backups carry the journal; verify_backup accepts .jsonl")
reset()
import backup  # noqa: E402

_DATA = _TMP / "data"
(_DATA).mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(str(_TMP / "webui.db"))
con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
con.execute("INSERT INTO chat (body) VALUES ('hello')")
con.commit()
con.close()
conv = "conv-b1"
facts.save_facts(conv, [{"text": "a fact"}])
post_nonstream(conv, "backed up question", REPLY)
post_nonstream(conv, "backed up question two", REPLY)
try:
    rep = backup.run_once()
    check(rep.get("ok"), f"backup cycle ok: {rep.get('detail')}")
    arch = Path(rep["archive"]) if rep.get("archive") else None
    if arch is not None and not arch.is_absolute():
        arch = BACKUPS / arch
    import tarfile

    with tarfile.open(arch) as tf:
        names = tf.getnames()
    check(any(n.endswith(f"journal/{conv}.jsonl") for n in names), "the archive contains journal/<conv>.jsonl")
    ok, detail = backup.verify_backup(arch)
    check(ok, f"verify_backup accepts an archive with a .jsonl journal: {detail}")
except Exception as e:  # pragma: no cover
    check(False, f"backup with journal raised {type(e).__name__}: {e}")

# ===========================================================================
print("\n[S1] scripts/chat-journal.py")
spec = importlib.util.spec_from_file_location(
    "chat_journal_cli", str(Path(__file__).resolve().parent.parent / "scripts" / "chat-journal.py"))
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
check(cli.pair_sha("a é", "b ✓") == chatjournal.pair_sha("a é", "b ✓"), "CLI pair hash == module pair hash")

reset()
conv = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
post_nonstream(conv, "alpha question", "alpha answer", headers={"X-OpenWebUI-Chat-Id": conv, "X-OpenWebUI-Message-Id": "a-asst-1"})
post_nonstream(conv, "beta question", "beta answer", headers={"X-OpenWebUI-Chat-Id": conv, "X-OpenWebUI-Message-Id": "a-asst-2"})
post_nonstream(conv, "gamma question", "gamma answer", headers={"X-OpenWebUI-Chat-Id": conv})  # no message id
post_nonstream(conv, "delta question", "delta answer", headers={"X-OpenWebUI-Chat-Id": conv, "X-OpenWebUI-Message-Id": "a-asst-4"})
eq(len(lines(conv)), 4, "fixture: four exchanges journaled")


def run_cli(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(list(argv))
    return code, buf.getvalue()


code, out = run_cli("stats", "--store", str(STORE))
eq(code, 0, "stats exits 0")
check(conv in out and "lines=4" in out, f"stats counts lines: {out.strip()!r}")
check("question" not in out and "answer" not in out, "stats prints no content")
code, out = run_cli("verify", "--store", str(STORE))
eq(code, 0, "verify passes on a clean journal")
# corrupt one line's text (sha no longer matches) and add junk
p = jfile(conv)
orig = p.read_text(encoding="utf-8")
bad = orig.replace("beta answer", "beta ANSWER", 1) + "not json at all\n"
p.write_text(bad, encoding="utf-8")
code, out = run_cli("verify", "--store", str(STORE))
eq(code, 1, "verify fails on a sha mismatch / unparseable line")
check("bad_lines=[2, 5]" in out, f"names the bad line numbers only: {out.strip()!r}")
check("beta" not in out, "verify prints no content")
p.write_text(orig, encoding="utf-8")

# a synthetic OpenWebUI database: JSON history has alpha+beta+delta(by content), table has alpha+gamma
db = _TMP / "owui.db"
con = sqlite3.connect(str(db))
con.execute("create table chat (id text primary key, chat text, current_message_id text, updated_at integer)")
con.execute("create table chat_message (id text primary key, chat_id text, parent_id text, role text, content text, done integer, created_at integer, updated_at integer)")
hist = {
    "u1": {"role": "user", "content": "alpha question", "parentId": None},
    "a-asst-1": {"role": "assistant", "content": "alpha answer", "parentId": "u1"},
    "u2": {"role": "user", "content": "beta question", "parentId": "a-asst-1"},
    "a-asst-2": {"role": "assistant", "content": "beta answer", "parentId": "u2"},
    "u4": {"role": "user", "content": "delta  question", "parentId": "a-asst-2"},   # extra space: normalised match
    "x4": {"role": "assistant", "content": "delta answer", "parentId": "u4"},          # id differs from the journal
}
con.execute("insert into chat values (?,?,?,?)", (conv, json.dumps({"history": {"messages": hist}}), "x4", 1))
rows = [("u1", None, "user", "alpha question"), ("a-asst-1", "u1", "assistant", "alpha answer"),
        ("u3", "a-asst-1", "user", "gamma question"), ("g3", "u3", "assistant", "gamma answer")]
for mid, parent, role, content in rows:
    con.execute("insert into chat_message values (?,?,?,?,?,?,?,?)",
                (f"{conv}-{mid}", conv, f"{conv}-{parent}" if parent else None, role, json.dumps(content), 1, 1, 1))
con.commit()
con.close()
code, out = run_cli("compare", "--store", str(STORE), "--db", str(db))
eq(code, 0, "compare exits 0")
check("journal exchanges: 4" in out, f"journal count: {out!r}")
check("missing from chat JSON: 1" in out, "gamma is missing from the chat JSON (only in the table)")
check("missing from chat_message table: 2" in out, "beta and delta are missing from the table")
check("missing from both: 0" in out, "nothing is missing from both")
check("alpha" not in out and "answer" not in out, "compare prints no content by default")
code, out2 = run_cli("compare", "--store", str(STORE), "--db", str(db), "--ids")
check("ids missing from chat JSON:" in out2, "--ids lists ids")
# the DB must not have been modified (read-only open)
check(sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute("select count(*) from chat_message").fetchone()[0] == 4,
      "the OpenWebUI DB is untouched")
# a tombstone matches by id only
run_cmd("forget", "gamma", conv)
code, out3 = run_cli("compare", "--store", str(STORE), "--db", str(db))
check("redacted" in out3 and "missing from chat JSON:" in out3, "compare still runs with tombstones")

# ===========================================================================
print("\n[D1] the images COPY the journal module and the guard lists it")
import re  # noqa: E402

_root = Path(__file__).resolve().parent.parent
for _df in ("Dockerfile", "Dockerfile.v31910"):
    _txt = (_root / _df).read_text(encoding="utf-8")
    check(re.search(r"^COPY compactor/chatjournal\.py /opt/compactor/chatjournal\.py$", _txt, re.M) is not None,
          f"{_df} COPYs compactor/chatjournal.py")
_txt = (_root / "Dockerfile.v31910").read_text(encoding="utf-8")
_guard = re.search(r'-c \\\s*"import (.*?);', _txt, re.S)
_mods = [m.strip() for m in re.sub(r"\\\s*", " ", _guard.group(1)).split(",")] if _guard else []
_copied = set(re.findall(r"^COPY compactor/([A-Za-z0-9_]+)\.py ", _txt, re.M))
eq(len(_mods), 29, "the import guard lists 29 modules")
check("chatjournal" in _mods, "...including chatjournal")
eq(set(_mods), _copied, "the guard lists exactly the modules the Dockerfile COPYs")
_all = {p.stem for p in (_root / "compactor").glob("*.py") if not p.name.startswith("test_")}
eq(_copied, _all, "and those are every non-test compactor module")

print()
if FAILED:
    print(f"FAILED {len(FAILED)}: {FAILED}")
    sys.exit(1)
print("ALL OK")
