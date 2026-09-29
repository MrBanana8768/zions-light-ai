"""
compactor.chatjournal: v3.1.9.10 per-message chat journal.

WHY. OpenWebUI stores each conversation as ONE JSON row (`chat.chat`, 63 MB
for her chat) and rewrites it whole on every save; on the network volume that
has broken three times. The compactor already sees every completed exchange
(the memory tail, `main._run_memory_tail`), so it keeps its own append-only
record: one small JSON line per completed reply, fsynced, in
`<STORAGE_ROOT>/journal/<conv_id>.jsonl`. It is a SAFETY COPY for rebuilding
a lost or damaged OpenWebUI chat later (design at the foot of this docstring),
not a memory layer: nothing in the compactor ever reads it back into a prompt.

LINE FORMAT (schema v=1), one JSON object per line, UTF-8, ensure_ascii off:

    v, conv_id, chat_id, message_id, parent_id, request_id, ts (UTC, ms, Z),
    epoch (the wipe-ledger epoch at write time; null when it could not be read), user, assistant,
    finish_reason, partial (bool), images {count, refs}, sha256,
    [truncated: ["user"|"assistant"]]   only when a field hit MAX_FIELD_CHARS
    [redacted: true]                    a /forget tombstone, see below

IDS ARE NULL ON THE DEPLOYED OPENWEBUI (0.11.4). `chat_id` / `message_id` /
`parent_id` are recorded WHEN THE REQUEST CARRIES THEM: the sanitised
X-OpenWebUI-Chat-Id / -Message-Id headers (same gate as the request-tracing
`ow_chat=` / `ow_msg=`), falling back to the body's `metadata`. OpenWebUI
0.11.4 sends neither: those headers exist only under
ENABLE_FORWARD_USER_INFO_HEADERS (kept OFF here, deliberately) and even then
there is no message-id header, and 0.11.4 strips `metadata` from the
forwarded body. So on prod every journal line has chat_id, message_id and
parent_id = null, and `scripts/chat-journal.py compare` matches on the
sha256 of the pair (user + assistant text) alone. The fields stay in the
schema for a future OpenWebUI that forwards them; nothing may rely on them.
`images` are REFERENCES only (a count, plus file ids or
names when present); bytes and data URLs are never stored. `sha256` is
`pair_sha(user, assistant)`, over the STORED texts.

WHICH REPLIES ARE JOURNALED (decided in `main._run_memory_tail`, after
`decide_memory_tail` and independent of it):
  - written: every reply that carried text, including a cut one (she pressed
    Stop, the generation ceiling, a stream that died, a client disconnect) and
    one the memory tail declines to memorize (degenerate, too short). The
    journal mirrors what the client received and OpenWebUI keeps, so it is
    not filtered by the memory policy. A reply that is not (finished and not
    truncated and not errored), or that is holed, is written with
    `partial: true` and the `finish_reason` seen ("length", "error",
    "incomplete" when the stream ended with none).
  - skipped: empty replies and failed requests (nothing to record), the
    self-test conversations, slash commands (they never reach the tail),
    OpenWebUI background task traffic (title/tag generation, not a chat
    turn), an exchange with no user text, disk-pressure pause.

DURABILITY / NEVER BLOCK THE REPLY. The reply path only hands a small task to
ONE dedicated writer thread (FIFO, so lines land in reply order); the thread
does open(O_APPEND) + write + fsync. A stalled volume therefore stalls the
journal, not a reply. Failures are logged by exception TYPE only and counted
(`/health/full` `chat_journal.errors`). The journal text is NEVER logged.

/FORGET and /RETIRE (see main._clear_all_memory, commands._handle_forget,
continuity). POINT-IN-TIME (v3.1.9.10, owner decision): /forget wipes what
exists NOW; what she says afterwards, even the same words, is journaled
normally. There is NO write-time filter.
  - `/forget` (full wipe) and the admin wipe: `wipe()` deletes the file,
    inside the same conv_lock section as every other layer, after the wipe
    generation bump, so a reply that was in flight is discarded by the
    generation check in `append_record` and cannot resurrect the file.
  - `/forget <text>`: `redact_needle()` turns every EXISTING matching
    exchange (needle found case-insensitively in the user text, the reply or
    an image ref) into a TOMBSTONE: ids, ts, epoch, finish_reason kept;
    user/assistant emptied; sha256 nulled (a hash of forgotten text would
    confirm a guess); refs dropped; `redacted: true`. The digest layer's
    semantics: the unit stays, its words go. The ledger entry records the
    forget's cutoff (UTC).
  - a restart, restore or import: the wipe ledger lives OUTSIDE STORAGE_ROOT
    (continuity.CONTROL_ROOT), so a restored older journal can carry forgotten
    words. `continuity.reapply_ledger` (run for every ledger at boot) calls
    `reapply()`, which removes only what PRE-DATES the forget: a line whose
    `ts` is before a needle's cutoff and that matches it is tombstoned;
    a line whose `ts` is before the last full wipe's `wipe_at`, or whose
    `epoch` stamp is below `wipe_epoch`, is dropped. A line written after the
    forget has a later `ts` and is left alone, so the same words said again
    later are remembered, and a restored old copy of the same words is not.
  - /retire does NOT delete the journal: it is the archive of the chat. Its
    epoch bump does not purge it either (`wipe_epoch` is not moved by a
    retire) and the needles are kept, so a restored older copy of the archive
    is still redacted. `scripts/chat-journal.py stats` lists retired
    conversations too.
  - the ledger is NOT needed to write: an unreadable ledger or a pending
    forget only leaves the line's epoch null (the `ts` rule still judges it).
    Nothing here can block a journal write on a ledger problem any more: the
    old reason (a write-time needle check needed the needles) is gone.

PRIVACY. The journal holds her words like the facts store does: STORAGE_ROOT
is inside the backups (intended) and outside `zl bundle` (ops/zlops/commands/
bundle.py forbids podenv.STORE). Files are 0600, the directory 0700.

GROWTH. ~ (user + reply) bytes plus ~400 bytes of envelope per exchange: a few
KB per message for prose (her 4,200-message chat: tens of MB at most). No
rotation; one file per conversation.

REBUILD DESIGN (later step, NOT implemented here; scripts/chat-journal.py has
no write path). Given `compare` output, an operator tool would, on a stopped
pod and against a COPY of webui.db, insert the missing exchanges as
`chat_message` rows (id `<chat>-<msg>`, JSON-encoded content, parent_id from
the journal) and history nodes in `chat.chat`, keyed by message id and
chained by parent_id; ids absent from the journal get fresh uuids chained in
`ts` order under the last matching node; then run repair-chat-tree.py.

NOTHING HERE MAY IMPORT `main`.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import datetime
import glob
import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import memory
import reqctx
from envcfg import env_bool, env_int

logger = logging.getLogger("compactor.chatjournal")

SCHEMA_VERSION = 1

ENABLED = env_bool("COMPACTOR_CHAT_JOURNAL", True)
# Per text field. A line is ~ 2 x this at worst; normal exchanges are KB.
MAX_FIELD_CHARS = env_int("COMPACTOR_CHAT_JOURNAL_MAX_CHARS", 200_000)
# Pending (not yet fsynced) writes we are willing to hold in memory if the
# volume stalls; past this a new exchange is dropped and counted.
MAX_PENDING = env_int("COMPACTOR_CHAT_JOURNAL_MAX_PENDING", 512)

SELFTEST_PREFIX = "__selftest"
_MAX_REFS = 32
_MAX_REF_CHARS = 120

# One thread: FIFO = reply order, and at most one writer touches the files.
# `_LOCK` still guards every mutation because boot-time reapply runs on a
# threadpool thread of its own.
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatjournal")
_LOCK = threading.RLock()

request_meta_ctx: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "chatjournal_request_meta", default=None
)

_stats_lock = threading.Lock()
_stats: dict[str, Any] = {
    "lines_written": 0,
    "errors": 0,
    "last_write_at": None,
    "skipped": 0,
    "dropped": 0,
    "redacted_lines": 0,
    "skipped_disk_pressure": 0,
}
_pending: set = set()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def journal_dir() -> Path:
    return memory.STORAGE_ROOT / "journal"


def journal_path(conv_id: str) -> Path:
    """`STORAGE_ROOT/journal/<conv_id>.jsonl` through memory._safe_path, the
    one conv_id validation / traversal chokepoint every store path uses."""
    return memory._safe_path("journal", conv_id, ".jsonl")


def is_selftest_conv(conv_id: str) -> bool:
    return (conv_id or "").startswith(SELFTEST_PREFIX)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def pair_sha(user: str, assistant: str) -> str:
    """sha256 over the canonical JSON of [user, assistant]. scripts/
    chat-journal.py carries an identical copy (it is stdlib-only) and a test
    pins the two together."""
    blob = json.dumps([user, assistant], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_FIELD_CHARS:
        return text, False
    return text[:MAX_FIELD_CHARS], True


def _ref(v: Any) -> str | None:
    """One image/file reference: a short id or name or URL PATH. Never a data
    URL, never a query string (it can carry tokens), never long."""
    if not isinstance(v, str):
        return None
    v = v.strip()
    if not v or v.lower().startswith("data:"):
        return None
    v = v.split("?", 1)[0].split("#", 1)[0]
    return v[:_MAX_REF_CHARS] or None


def image_refs(message: dict | None, body: dict | None = None) -> dict:
    """{"count": n, "refs": [...]} for the last user message. `count` is the
    number of image parts; refs are ids/names/URL paths when present
    (image part id/file_id/name/url, and the request's metadata.files ids and
    names). Best effort, references only."""
    count = 0
    refs: list[str] = []

    def add(v):
        r = _ref(v)
        if r and r not in refs and len(refs) < _MAX_REFS:
            refs.append(r)

    content = (message or {}).get("content")
    if isinstance(content, list):
        for c in content:
            if not isinstance(c, dict):
                continue
            t = c.get("type")
            if not (t in ("image_url", "image", "input_image") or "image_url" in c):
                continue
            count += 1
            for k in ("file_id", "id", "name"):
                add(c.get(k))
            iu = c.get("image_url")
            add(iu.get("url") if isinstance(iu, dict) else iu)
    md = (body or {}).get("metadata")
    files = md.get("files") if isinstance(md, dict) else None
    if isinstance(files, list):
        for f in files:
            if isinstance(f, dict):
                inner = f.get("file") if isinstance(f.get("file"), dict) else {}
                add(f.get("id") or inner.get("id"))
                add(f.get("name") or inner.get("filename") or inner.get("name"))
    return {"count": count, "refs": refs}


def capture_request_meta(headers, body: dict | None, messages: list | None) -> dict:
    """Ids and image references for this request, taken at the top of the
    request (before image stripping rewrites the messages). Sanitised with the
    same gate as request tracing. Never raises."""
    meta: dict = {"chat_id": None, "message_id": None, "parent_id": None,
                  "images": {"count": 0, "refs": []}}
    try:
        md = (body or {}).get("metadata")
        md = md if isinstance(md, dict) else {}

        def _h(name):
            try:
                return headers.get(name)
            except Exception:
                return None

        def _s(v):
            return reqctx.sane_incoming_id(v if isinstance(v, str) else None)

        meta["chat_id"] = _s(_h("x-openwebui-chat-id")) or _s(md.get("chat_id"))
        meta["message_id"] = _s(_h("x-openwebui-message-id")) or _s(md.get("message_id"))
        meta["parent_id"] = _s(md.get("parent_message_id")) or _s(md.get("parent_id"))
        last_user = next(
            (m for m in reversed(messages or []) if isinstance(m, dict) and m.get("role") == "user"),
            None,
        )
        meta["images"] = image_refs(last_user, body)
    except Exception as e:  # metadata is best effort
        logger.warning(f"chat journal: could not read request ids ({type(e).__name__})")
    return meta


def set_request_meta(meta: dict | None) -> None:
    request_meta_ctx.set(meta)


def _finish_reason(finished: bool, truncated: bool, holed: bool, seen: str | None) -> tuple[str, bool]:
    """(finish_reason, partial). `seen` is the model's own value when known."""
    errored = seen == "error"
    partial = bool(holed or truncated or errored or not finished)
    if seen:
        return seen, partial
    if truncated:
        return "length", True
    if not finished:
        return "incomplete", True
    return "stop", partial


def build_record(
    *, conv_id: str, user: str, assistant: str, meta: dict | None, request_id: str | None,
    ts: str, epoch: "int | None", finish_reason: str, partial: bool,
) -> dict:
    meta = meta or {}
    u, ut = _clip(user)
    a, at = _clip(assistant)
    img = meta.get("images") or {"count": 0, "refs": []}
    rec: dict = {
        "v": SCHEMA_VERSION,
        "conv_id": conv_id,
        "chat_id": meta.get("chat_id"),
        "message_id": meta.get("message_id"),
        "parent_id": meta.get("parent_id"),
        "request_id": request_id,
        "ts": ts,
        "epoch": None if epoch is None else int(epoch),
        "user": u,
        "assistant": a,
        "finish_reason": finish_reason,
        "partial": bool(partial),
        "images": {"count": int(img.get("count") or 0), "refs": list(img.get("refs") or [])},
        "sha256": pair_sha(u, a),
    }
    trunc = [n for n, t in (("user", ut), ("assistant", at)) if t]
    if trunc:
        rec["truncated"] = trunc
    return rec


def _tombstone(rec: dict) -> dict:
    out = {k: rec.get(k) for k in (
        "v", "conv_id", "chat_id", "message_id", "parent_id", "request_id", "ts",
        "epoch", "finish_reason", "partial",
    )}
    n = 0
    try:
        n = int((rec.get("images") or {}).get("count") or 0)
    except Exception:
        n = 0
    out.update({
        "user": "", "assistant": "", "images": {"count": n, "refs": []},
        "sha256": None, "redacted": True,
    })
    return out


def _dumps(rec: dict) -> str:
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Low-level file operations (all under _LOCK, all sync: run on the writer thread)
# ---------------------------------------------------------------------------

def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _append_line(path: Path, line: str) -> None:
    """One O_APPEND write of the whole line, then fsync. Split out so a test
    can make it fail (monkeypatch, never chmod: the gate runs as root)."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existed = path.exists()
    # O_RDWR (not O_WRONLY) only so the last byte can be read: see below.
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        data = (line + "\n").encode("utf-8")
        # v3.1.9.10 (R1 L3). A crash mid-write leaves a last line with no
        # newline; appending straight after it would glue THIS exchange onto
        # the torn one and lose both. Start on a fresh line instead: the torn
        # fragment stays an unparseable line of its own, this one parses.
        size = os.fstat(fd).st_size
        if size > 0 and os.pread(fd, 1, size - 1) != b"\n":
            data = b"\n" + data
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if not existed:
        _fsync_dir(path.parent)


def _bump(key: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[key] += n


def _epoch_for(conv_id: str):
    """The ledger epoch to stamp on a new line, or None when it cannot be
    known (never blocks the write; see the module docstring)."""
    import continuity  # lazy: continuity.reapply_ledger lazily imports us

    return continuity.journal_epoch(conv_id)


def _needle_list(needles) -> list[str]:
    out = []
    for n in needles or []:
        s = ((n.get("needle") if isinstance(n, dict) else n) or "").strip().lower()
        if s:
            out.append(s)
    return out


def _hits(rec: dict, needles: list[str]) -> bool:
    if not needles or rec.get("redacted"):
        return False
    hay = "\n".join([
        str(rec.get("user") or ""), str(rec.get("assistant") or ""),
        "\n".join(str(r) for r in (rec.get("images") or {}).get("refs") or []),
    ]).lower()
    return any(n in hay for n in needles)


def append_record(job: dict, wipe_generation: int | None) -> str:
    """Writer-thread half. Returns an outcome label. `job` carries the raw
    pieces captured on the event loop. Raises only on a genuine write error
    (the caller counts it)."""
    conv_id = job["conv_id"]
    with _LOCK:
        if wipe_generation is not None and wipe_generation != memory.current_wipe_generation(conv_id):
            return "skipped_wipe"
        epoch = _epoch_for(conv_id)
        rec = build_record(epoch=epoch, **{k: job[k] for k in (
            "conv_id", "user", "assistant", "meta", "request_id", "ts", "finish_reason", "partial")})
        _append_line(journal_path(conv_id), _dumps(rec))
        return "written"


def _run_job(job: dict, wipe_generation: int | None) -> None:
    try:
        outcome = append_record(job, wipe_generation)
    except Exception as e:
        _bump("errors")
        logger.error(f"chat journal: append failed ({type(e).__name__})")
        return
    if outcome == "written":
        with _stats_lock:
            _stats["lines_written"] += 1
            _stats["last_write_at"] = _utc_now_iso()
    else:
        _bump("skipped")


# ---------------------------------------------------------------------------
# Event-loop side
# ---------------------------------------------------------------------------

def submit(
    conv_id: str, user: str, assistant: str, *, finished: bool, truncated: bool,
    holed: bool, finish_reason: str | None, wipe_generation: int | None,
) -> str:
    """Hand one completed exchange to the writer thread. Returns a label
    ('queued', 'disabled', 'skipped_selftest', 'skipped_empty', 'dropped').
    NEVER raises and never blocks: this is on the reply path."""
    try:
        if not ENABLED:
            return "disabled"
        if is_selftest_conv(conv_id):
            return "skipped_selftest"
        if not (assistant or "").strip():
            return "skipped_empty"
        fr, partial = _finish_reason(finished, truncated, holed, finish_reason)
        job = {
            "conv_id": conv_id, "user": user or "", "assistant": assistant,
            "meta": request_meta_ctx.get(), "request_id": reqctx.current_request_id(),
            "ts": _utc_now_iso(), "finish_reason": fr, "partial": partial,
        }
        with _stats_lock:
            if len(_pending) >= MAX_PENDING:
                _stats["dropped"] += 1
                full = True
            else:
                full = False
        if full:
            logger.error("chat journal: writer backlog full; exchange not journaled")
            return "dropped"
        fut = _EXEC.submit(_run_job, job, wipe_generation)
        with _stats_lock:
            _pending.add(fut)
        fut.add_done_callback(_discard)
        return "queued"
    except Exception as e:
        _bump("errors")
        logger.error(f"chat journal: could not queue an exchange ({type(e).__name__})")
        return "error"


def note_skipped_disk_pressure() -> None:
    """main._run_memory_tail calls this when degrade.guard refused the
    journal write (disk pressure): counted in /health/full so a paused
    journal is visible (v3.1.9.10, R1 L6)."""
    _bump("skipped_disk_pressure")


def _discard(fut) -> None:
    with _stats_lock:
        _pending.discard(fut)


def drain(timeout: float = 10.0) -> bool:
    """Block until every queued write has finished (tests, shutdown)."""
    with _stats_lock:
        futs = list(_pending)
    done, not_done = concurrent.futures.wait(futs, timeout=timeout)
    return not not_done


# ---------------------------------------------------------------------------
# /forget: wipe, redact, reapply
# ---------------------------------------------------------------------------

def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def tmp_leftovers(conv_id: str) -> list[Path]:
    """`<conv>.jsonl.*.tmp` files: the temp copy `_rewrite` streams into. A
    kill between its first write and the os.replace leaves one behind, and it
    holds her words (v3.1.9.10, R1 M2). journal_path validates the conv_id
    (it is a plain name: [A-Za-z0-9_-]), so the glob cannot leave the dir."""
    p = journal_path(conv_id)
    try:
        return sorted(q for q in p.parent.glob(glob.escape(p.name) + ".*.tmp") if q.is_file())
    except OSError:
        return []


def wipe(conv_id: str) -> int:
    """Delete the conversation's journal AND any `.tmp` rewrite leftovers.
    Returns the number of lines removed (0 when there was none). Raises OSError
    if an unlink fails (the caller reports it; /forget's residue check also
    sees the file)."""
    p = journal_path(conv_id)
    with _LOCK:
        leftovers = tmp_leftovers(conv_id)
        if not p.is_file() and not leftovers:
            return 0
        n = 0
        if p.is_file():
            try:
                n = _count_lines(p)
            except OSError:
                n = 0
            p.unlink()
        for q in leftovers:
            if not n:
                try:
                    n = _count_lines(q)
                except OSError:
                    pass
            q.unlink()
        _fsync_dir(p.parent)
        return n


async def wipe_async(conv_id: str) -> int:
    """`wipe` on the writer thread (FIFO behind queued appends) so the event
    loop never waits on the volume. Called under conv_lock."""
    import asyncio

    return await asyncio.get_running_loop().run_in_executor(_EXEC, wipe, conv_id)


async def redact_needle_async(conv_id: str, needle: str) -> int:
    """`redact_needle` on the writer thread, FIFO behind every queued append
    (as `wipe_async` does). v3.1.9.10, R1 M1: /forget <text> used to run it on
    the general threadpool, so a reply queued BEFORE the forget could land
    AFTER the redaction with the forgotten words intact (20 of 20 trials with
    a stalled writer). On the single writer thread, everything queued earlier
    is on disk before the redaction reads the file."""
    import asyncio

    return await asyncio.get_running_loop().run_in_executor(_EXEC, redact_needle, conv_id, needle)


def has_content(conv_id: str) -> bool:
    """True if a journal file (or a `.tmp` rewrite leftover) with at least one
    byte exists (for /forget's post-wipe residue check)."""
    try:
        p = journal_path(conv_id)
        if p.is_file() and p.stat().st_size > 0:
            return True
        return any(q.stat().st_size > 0 for q in tmp_leftovers(conv_id))
    except Exception:
        return False


def _scan_would_change(path: Path, transform) -> bool:
    """True if `transform` would change or drop any line of `path`. Reads
    only: no temp file. (Blank lines never count: `_rewrite` skips them and
    replaces the file only when a real line changed or was dropped.)"""
    with open(path, "r", encoding="utf-8", errors="replace", newline="\n") as src:
        for raw in src:
            line = raw.rstrip("\n")
            if not line:
                continue
            new = transform(line)
            if new is None or new != line:
                return True
    return False


def _rewrite(path: Path, transform, on_write_pass=None) -> tuple[int, int]:
    """Stream `path` through `transform(line:str) -> str|None` (None drops the
    line, an equal string keeps it) into a temp file, fsync, os.replace.
    Returns (changed, dropped).

    SCANS FIRST (v3.1.9.10, R1 M2): when nothing would change, NO temp copy is
    made at all (the boot-time reapply used to write a full copy of every
    journal on every boot, each a window for a kill to leave her words in a
    stray .tmp). `on_write_pass`, if given, runs just before the second
    (writing) pass so a caller's counters that `transform` bumps are not
    counted twice."""
    if not _scan_would_change(path, transform):
        return 0, 0
    if on_write_pass is not None:
        on_write_pass()
    changed = dropped = 0
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out, \
                open(path, "r", encoding="utf-8", errors="replace", newline="\n") as src:
            for raw in src:
                line = raw.rstrip("\n")
                if not line:
                    continue
                new = transform(line)
                if new is None:
                    dropped += 1
                    continue
                if new != line:
                    changed += 1
                out.write(new + "\n")
            out.flush()
            os.fsync(out.fileno())
        if changed or dropped:
            os.replace(tmp, path)
            _fsync_dir(path.parent)
        else:
            os.unlink(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return changed, dropped


def redact_needle(conv_id: str, needle: str) -> int:
    """`/forget <text>`: tombstone every matching exchange. Returns how many.
    Unparseable lines that contain the needle are dropped outright. Raises
    OSError on a failed rewrite (the file is then unchanged); the needle is
    already in the ledger, so the next boot's `reapply` retries it."""
    nl = _needle_list([needle])
    if not nl:
        return 0
    p = journal_path(conv_id)
    with _LOCK:
        if not p.is_file():
            return 0

        def tf(line: str):
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError
            except Exception:
                return None if any(n in line.lower() for n in nl) else line
            if _hits(rec, nl):
                return _dumps(_tombstone(rec))
            return line

        changed, dropped = _rewrite(p, tf)
        _bump("redacted_lines", changed + dropped)
        return changed + dropped


def reapply(conv_id: str, ledger: dict) -> dict:
    """Boot-time re-apply of the wipe ledger (continuity.reapply_ledger),
    POINT-IN-TIME: it removes only content that PRE-DATES a forget.

      - full wipe: drop a line with ts < ledger.wipe_at, or with an epoch
        stamp below ledger.wipe_epoch. A /retire moves neither (the journal
        is the retired chat's archive);
      - targeted forget: tombstone a line that matches a needle AND pre-dates
        that needle's cutoff (continuity.pre_dates: ts < cutoff, and not
        stamped with a later epoch). A line with no readable ts counts as
        pre-dating (unknown age, safe side);
      - an unparseable line containing any needle is dropped.

    Idempotent. `ledger` is the normalized ledger dict."""
    import continuity

    entries = [n for n in (ledger.get("targeted") or [])
               if isinstance(n, dict) and (n.get("needle") or "").strip()]
    all_nl = _needle_list(entries)
    wipe_epoch = ledger.get("wipe_epoch")
    wipe_at = continuity._parse_ts(ledger.get("wipe_at"))
    p = journal_path(conv_id)
    with _LOCK:
        # v3.1.9.10 (R1 M2): a killed rewrite leaves `<conv>.jsonl.*.tmp`
        # with her words in it. The real file is intact (os.replace never
        # ran), so the leftover is only ever a stray copy: remove it here,
        # whether or not the journal itself still exists.
        stray = 0
        for q in tmp_leftovers(conv_id):
            try:
                q.unlink()
                stray += 1
            except OSError as e:
                logger.error(f"chat journal: could not remove a rewrite leftover ({type(e).__name__})")
        if stray:
            _fsync_dir(p.parent)
        if not p.is_file():
            return {"purged": 0, "redacted": 0, "stray_tmp_removed": stray}
        purged = [0]

        def tf(line: str):
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError
            except Exception:
                return None if any(n in line.lower() for n in all_nl) else line
            try:
                ep = None if rec.get("epoch") is None else int(rec.get("epoch"))
            except Exception:
                ep = None
            ts = rec.get("ts")
            t = continuity._parse_ts(ts)
            if (wipe_epoch is not None and ep is not None and ep < int(wipe_epoch)) or (
                wipe_at is not None and t is not None and t < wipe_at
            ):
                purged[0] += 1
                return None
            for n in entries:
                nl = _needle_list([n])
                if nl and _hits(rec, nl) and continuity.pre_dates(ts, ep, n):
                    return _dumps(_tombstone(rec))
            return line

        changed, dropped = _rewrite(p, tf, on_write_pass=lambda: purged.__setitem__(0, 0))
        if p.is_file() and p.stat().st_size == 0:
            try:
                p.unlink()
            except OSError:
                pass
        return {"purged": purged[0], "redacted": changed, "dropped": dropped,
                "stray_tmp_removed": stray}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health_snapshot() -> dict:
    """Additive /health/full block. Counters are per process. No conv ids, no
    text."""
    with _stats_lock:
        s = dict(_stats)
        pending = len(_pending)
    return {
        "enabled": bool(ENABLED),
        "last_write_at": s["last_write_at"],
        "lines_written": s["lines_written"],
        "errors": s["errors"],
        "skipped": s["skipped"],
        "dropped": s["dropped"],
        "skipped_disk_pressure": s["skipped_disk_pressure"],
        "pending": pending,
    }


def _reset_for_tests() -> None:
    drain(5.0)
    with _stats_lock:
        _stats.update(lines_written=0, errors=0, last_write_at=None, skipped=0,
                      dropped=0, redacted_lines=0, skipped_disk_pressure=0)
