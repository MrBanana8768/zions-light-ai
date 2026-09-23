"""
sync_once's publish guards, second hostile pass (v3.1.9): the unit, the row,
the generation and the clock.

The first guard suite (test_webuidb_restore_guard.py) pads every fixture with
ASCII, keeps `chat` as a handful of tiny rows, and never builds two VALID
databases of different vintage. Each of those habits hid one of these:

  [N3] _content_bytes counted CHARACTERS. SQLite's length() on TEXT is a
       character count; the value is named bytes and printed as MB. In ASCII
       the two are the same number, so no ASCII fixture can observe the unit.
  [N4] half of ONE conversation published silently. OpenWebUI keeps a whole
       conversation in one row of `chat` (the live one was measured at
       32.95 MB), so 1 row -> 1 row passes every count rule, and a 49.5% loss
       inside that row leaves the table-wide ratio above 0.5.
  [N5] the fifth state: readable, healthy, right schema, right size - and an
       OLDER GENERATION of the same database. Both existing measures are
       magnitudes; going backwards in time is not a change in magnitude.
  [N2] a snapshot mtime in the future made every cycle a silent skip, and
       copy2 carries that mtime down to every later pod.

FIXTURES ARE OPENWEBUI-SHAPED ON PURPOSE. The `chat` schema below is the one
open-webui 0.11.0 declares (models/chats.py, read out of the shipped
angreg/zions-light-ai:v3.1.8-cu12 image), and conversation bodies are built
with json.dumps defaults - which is what SQLAlchemy's JSON type writes on
SQLite, verified in the same image: {"m": "shalom"} in Hebrew is stored as
`{"m": "\\u05e9\\u05dc\\u05d5\\u05dd"}`, 33 characters and 33 bytes. So in
0.11.0 the JSON columns are ASCII and the character/byte confusion reaches
only the TEXT columns (title, summary, folder_id, ...). Where a case below
stores raw non-ASCII inside `chat` it says so, and says why.

EVERY REFUSAL HAS A CONTROL, and every guard has a fixture where IT ALONE
decides: a guard that refuses everything passes every refusal test, and a
guard shadowed by its neighbour passes a test it has nothing to do with.

    python test_webuidb_publish_guards.py
"""

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="pg-local-"))
_SNAP_VOL = Path(tempfile.mkdtemp(prefix="pg-moosefs-"))
_QUAR = Path(tempfile.mkdtemp(prefix="pg-forensics-"))
os.environ["WEBUI_LOCAL_DB"] = str(_LOCAL_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(_SNAP_VOL / "openwebui" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUAR)
for _k in list(os.environ):
    if _k.startswith("WEBUI_DB_ALLOW_") or _k in (
        "WEBUI_DB_MAX_ROW_LOSS_BYTES", "WEBUI_DB_SHRINK_REFUSE_BELOW",
        "WEBUI_DB_SHRINK_GUARD_MIN_CHATS", "WEBUI_DB_SHRINK_GUARD_MIN_BYTES",
    ):
        os.environ.pop(_k)

import webuidb  # noqa: E402

HERE = Path(__file__).resolve().parent
LOCAL = webuidb.LOCAL_DB
SNAP = webuidb.SNAPSHOT_DB
FAILED: list[str] = []
HEBREW = "שלום"          # 4 characters, 8 UTF-8 bytes
CJK = "話"                               # 1 character, 3 UTF-8 bytes
MB = 1_000_000


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[tuple[int, str]] = []

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))


CAP = _Capture()
_log = logging.getLogger("compactor.webuidb")
_log.addHandler(CAP)
_log.setLevel(logging.INFO)


def logged(level: int, needle: str) -> bool:
    return any(lv >= level and needle in msg for lv, msg in CAP.records)


# open-webui 0.11.0's `chat` table, column for column.
OWUI_CHAT = """create table chat (
    id varchar not null primary key,
    user_id varchar,
    title text,
    chat json,
    created_at bigint,
    updated_at bigint,
    share_id text unique,
    archived boolean,
    pinned boolean,
    meta json default '{}',
    variables json,
    folder_id text,
    tasks json,
    summary text,
    current_message_id text,
    last_read_at bigint
)"""


def conversation(n_msgs: int, text: str = "m" * 1000, ensure_ascii=True) -> str:
    """One conversation's `chat` JSON, in OpenWebUI's history shape and
    serialised the way SQLAlchemy's JSON type serialises it (json.dumps
    defaults) unless a case deliberately says otherwise."""
    msgs = {
        f"msg-{i:05d}": {
            "id": f"msg-{i:05d}",
            "parentId": f"msg-{i - 1:05d}" if i else None,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": text,
        }
        for i in range(n_msgs)
    }
    return json.dumps({"history": {"messages": msgs}}, ensure_ascii=ensure_ascii)


def wipe_one(path: Path) -> None:
    for suffix in ("",) + webuidb.SIDECARS:
        f = path.with_name(path.name + suffix)
        if f.exists():
            f.unlink()


def wipe():
    wipe_one(LOCAL)
    wipe_one(SNAP)
    if webuidb.EMPTY_START_MARKER.exists():
        webuidb.EMPTY_START_MARKER.unlink()
    CAP.records.clear()
    # Each scenario is its own incident and expects its own forensic copy
    # (FORENSIC_COPY_MIN_INTERVAL_S) — several run within the same process
    # well inside the default 3600s throttle window; without this reset a
    # later case's "nothing new copied" / "one copy landed" assertion would
    # depend on execution order rather than on what that case tests.
    webuidb._forensic_copy_last_monotonic = None


def owui(path: Path, rows) -> None:
    """rows: (id, updated_at, chat_json[, title]). Replaces any file there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wipe_one(path)
    con = sqlite3.connect(str(path))
    con.execute(OWUI_CHAT)
    con.execute("create table user (id text)")
    con.execute("insert into user values ('u1')")
    for row in rows:
        cid, updated_at, body = row[0], row[1], row[2]
        title = row[3] if len(row) > 3 else "a conversation"
        con.execute(
            "insert into chat (id, user_id, title, chat, created_at, updated_at, "
            "archived, pinned, meta) values (?, 'u1', ?, ?, 1, ?, 0, 0, '{}')",
            (cid, title, body, updated_at),
        )
    con.commit()
    con.close()


def chats(p):
    return webuidb._has_rows(p)


def content(p):
    return webuidb._content_bytes(p)


def snap_bytes() -> bytes:
    return SNAP.read_bytes()


def with_env(env: dict, fn):
    """Run fn() with env vars set and _reload_env() applied, then undo both.
    The finally is the point: a flag leaked into a later case disarms it."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    webuidb._reload_env()
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        webuidb._reload_env()


def run_cli(args, extra_env=None):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(HERE), PYTHONIOENCODING="utf-8")
    env.update(extra_env or {})
    r = subprocess.run(
        [sys.executable, "webuidb.py", *args], cwd=str(HERE), env=env,
        capture_output=True, text=True, errors="replace", timeout=180,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


MAX_ROW_LOSS = getattr(webuidb, "MAX_ROW_LOSS_BYTES", 1_000_000)

# ===========================================================================
print()
print("[N3] _content_bytes MEASURES BYTES, NOT CHARACTERS")
# ===========================================================================
print("    the unit, on a row open-webui 0.11.0 really writes")
# A Hebrew title is a TEXT column and is stored raw; the conversation JSON is
# stored escaped. The exact expected number is the UTF-8 length of every
# column's value as SQLite renders it (integers as their decimal text), which
# is what length(cast(x as blob)) returns.
wipe()
_title = HEBREW * 2500                        # 10,000 characters, 20,000 bytes
_body = conversation(20, text=HEBREW * 50)
owui(LOCAL, [("conv-he", 1_726_000_000, _body, _title)])
_con = sqlite3.connect(str(LOCAL))
_row = _con.execute("select * from chat").fetchone()
_con.close()
_expected = sum(len(str(v).encode("utf-8")) for v in _row if v is not None)
_chars = sum(len(str(v)) for v in _row if v is not None)
check(
    _expected - _chars == 10_000,
    f"PRECONDITION: this row's bytes and characters differ by exactly the "
    f"title's 10,000 extra bytes ({_expected} vs {_chars}) - the JSON column is "
    f"ASCII-escaped, as SQLAlchemy writes it, so ONLY the TEXT column can "
    f"separate the units",
)
check(
    content(LOCAL) == _expected,
    f"_content_bytes is the UTF-8 byte count ({content(LOCAL)} == {_expected}; "
    f"a character count would be {_chars})",
)

print("    a conversation whose bytes are two-thirds gone, at an equal character count")
# The finding's C5. Raw CJK inside `chat` is NOT what 0.11.0 writes (see the
# module docstring) - it is what any writer with ensure_ascii=False writes,
# and the refusal text prints this number as MB. Kept because it is the case
# where the unit decides a publish outright: 900 KB -> 300 KB of stored bytes,
# 300,000 -> 300,000 characters. The per-row loss (600 KB) is under
# MAX_ROW_LOSS_BYTES and the conversation is NEWER on local, so neither of
# the other two content guards can be what refuses it.
wipe()
owui(SNAP, [("conv", 100, CJK * 300_000)])
owui(LOCAL, [("conv", 200, "a" * 300_000)])
_before = snap_bytes()
check(
    (content(SNAP) - content(LOCAL)) < MAX_ROW_LOSS
    and chats(SNAP) == chats(LOCAL) == 1,
    f"PRECONDITION: one row each side, and the row lost "
    f"{content(SNAP) - content(LOCAL)} bytes, under the per-row limit "
    f"{MAX_ROW_LOSS} - the content RATIO alone has to decide this",
)
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and snap_bytes() == _before,
    f"refused, snapshot byte-for-byte untouched (synced={r['synced']}, "
    f"error={(r['error'] or '')[:90]})",
)
check(
    bool(r["error"]) and "stored content" in r["error"] and "0.9 MB" in r["error"],
    "and the refusal prints the snapshot's content as 0.9 MB - which it is, "
    "in bytes; in characters it read 0.3 MB",
)

print("    CONTROL: the same CJK conversation, grown, publishes")
wipe()
owui(SNAP, [("conv", 100, CJK * 300_000)])
owui(LOCAL, [("conv", 200, CJK * 300_000 + "a" * 5_000)])
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"growth publishes (error={r['error']})")

print("    CONTROL: fewer characters, more bytes, publishes - the unit cuts this way too")
# 300,000 ASCII characters (300 KB) replaced by 120,000 CJK characters (360 KB).
# In characters that is 0.4 of the snapshot and was REFUSED; in bytes it is
# 1.2x and nothing is smaller. A unit observer more than a production state,
# and labelled as one: it is the fixture on which a character count says no
# and a byte count says yes.
wipe()
owui(SNAP, [("conv", 100, "a" * 300_000)])
owui(LOCAL, [("conv", 200, CJK * 120_000)])
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"published (error={(r['error'] or '')[:90]})")

print("    --status prints the same unit")
wipe()
owui(SNAP, [("conv", 100, CJK * 300_000)])
owui(LOCAL, [("conv", 200, "a" * 300_000)])
rc, out = run_cli(["--status"])
_snap_line = next((ln for ln in out.splitlines() if ln.startswith("snapshot ")), "")
check(
    rc == 0 and "content=0.9 MB" in _snap_line,
    f"--status reports the snapshot's content in bytes ({_snap_line.strip()[-60:]!r})",
)

print("    PRESERVED: no `chat` table is None, an empty `chat` table is 0")
# Mutation M7 of the hostile pass survived every suite: returning 0 for a
# schema the measure does not recognise would stand the content guard down in
# silence. The rewrite of this function keeps the distinction; this pins it.
wipe()
LOCAL.parent.mkdir(parents=True, exist_ok=True)
_c = sqlite3.connect(str(LOCAL))
_c.execute("create table chats (id text, chat text)")
_c.commit()
_c.close()
check(content(LOCAL) is None, f"no `chat` table -> None ({content(LOCAL)!r})")
wipe()
owui(LOCAL, [])
check(content(LOCAL) == 0, f"`chat` with no rows -> 0 ({content(LOCAL)!r})")


# ===========================================================================
print()
print("[N4] A SURVIVING CONVERSATION THAT LOSES MORE THAN MAX_ROW_LOSS_BYTES")
# ===========================================================================
# 4,000 messages of ~1 KB each in ONE row, the live shape at a size a test can
# afford. 2,020 kept: 49.5% of it gone, content ratio 0.505, chats 1 -> 1.
wipe()
owui(SNAP, [("the-conversation", 1000, conversation(4000))])
owui(LOCAL, [("the-conversation", 2000, conversation(2020))])
_loss = content(SNAP) - content(LOCAL)
check(
    content(LOCAL) >= content(SNAP) * webuidb.SHRINK_REFUSE_BELOW
    and chats(SNAP) == chats(LOCAL) == 1
    and _loss > MAX_ROW_LOSS,
    f"PRECONDITION: ratio {content(LOCAL) / content(SNAP):.3f} is above "
    f"SHRINK_REFUSE_BELOW, 1 row -> 1 row, and the row lost {_loss} bytes "
    f"(> {MAX_ROW_LOSS}) - nothing but the per-row limit can refuse this",
)
_before = snap_bytes()
# Counted, not globbed for emptiness: [N3]'s ratio refusal above has already
# left its own forensic copy in this directory.
_quar_before = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and snap_bytes() == _before,
    f"refused, snapshot untouched (synced={r['synced']})",
)
check(
    bool(r["error"]) and "WEBUI_DB_ALLOW_ROW_LOSS" in r["error"]
    and "WEBUI_DB_ALLOW_SHRINK" not in r["error"],
    "and the way out it names is the per-row flag, not ALLOW_SHRINK - taking "
    "that advice must not disarm the empty-database guard",
)
check(
    bool(r["error"]) and "the-conv" in r["error"],
    "and it names the conversation, so a human can look at the right row",
)
_quar_rowloss = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*"))) - _quar_before
check(
    _quar_rowloss == 0,
    f"and NOTHING new was copied to quarantine ({_quar_rowloss} files): this "
    f"refusal is reached by ordinary use (a deleted branch) and repeats every "
    f"SYNC_INTERVAL_S, and the file holding the lost content - the snapshot - "
    f"is the one being left untouched",
)

print("    CONTROL: a loss under the limit publishes")
wipe()
owui(SNAP, [("the-conversation", 1000, conversation(4000))])
owui(LOCAL, [("the-conversation", 2000, conversation(3600))])
check(
    0 < content(SNAP) - content(LOCAL) < MAX_ROW_LOSS,
    f"PRECONDITION: {content(SNAP) - content(LOCAL)} bytes lost, under the limit",
)
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"published (error={(r['error'] or '')[:90]})")

print("    CONTROL: a conversation DELETED outright is not row loss")
# A row that is gone is a deletion, and deletions stay governed by the chat
# count and content ratio. 3 -> 2 large conversations: every ratio holds.
wipe()
owui(SNAP, [(f"conv-{i}", 1000 + i, conversation(1500)) for i in range(3)])
owui(LOCAL, [(f"conv-{i}", 1000 + i, conversation(1500)) for i in range(2)])
check(
    content(SNAP) - content(LOCAL) > MAX_ROW_LOSS,
    f"PRECONDITION: the deleted conversation held {content(SNAP) - content(LOCAL)} "
    f"bytes, over the per-row limit - so this passes only if a missing row is "
    f"not counted as a row that shrank to nothing",
)
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"published (error={(r['error'] or '')[:90]})")

print("    the limit is a knob, and _reload_env reloads it")
wipe()
owui(SNAP, [("the-conversation", 1000, conversation(4000))])
owui(LOCAL, [("the-conversation", 2000, conversation(3600))])
r = with_env({"WEBUI_DB_MAX_ROW_LOSS_BYTES": "100000"},
             lambda: webuidb.sync_once(force=True))
check(
    r["synced"] is False and "WEBUI_DB_ALLOW_ROW_LOSS" in (r["error"] or ""),
    f"the control above is REFUSED at a 100 KB limit (synced={r['synced']}) - "
    f"a knob _reload_env forgot would publish here",
)
wipe()
owui(SNAP, [("conv", 1000, conversation(20))])
owui(LOCAL, [("conv", 2000, conversation(20))])
r = with_env({"WEBUI_DB_MAX_ROW_LOSS_BYTES": "-5"},
             lambda: webuidb.sync_once(force=True))
check(
    r["synced"] is True,
    f"a NEGATIVE limit is clamped to 0: an unchanged conversation (loss 0) "
    f"still publishes (error={(r['error'] or '')[:90]}) - unclamped, 0 > -5 "
    f"refuses every row that did not change",
)
check(
    getattr(webuidb, "MAX_ROW_LOSS_BYTES", None) == 1_000_000,
    f"and the default is back afterwards "
    f"({getattr(webuidb, 'MAX_ROW_LOSS_BYTES', None)})",
)


def _row_loss_state():
    wipe()
    owui(SNAP, [("the-conversation", 1000, conversation(4000))])
    owui(LOCAL, [("the-conversation", 2000, conversation(2020))])


def _shrink_state():
    # Only the content RATIO trips: 300 KB -> 40 B in a surviving row is a
    # 300 KB loss, under the per-row limit; the row is newer on local.
    wipe()
    owui(SNAP, [("conv", 1000, "x" * 300_000)])
    owui(LOCAL, [("conv", 2000, "x" * 40)])


print("    each override opens only its own door (all four cells)")
_m = {}
for _flag in ("WEBUI_DB_ALLOW_ROW_LOSS", "WEBUI_DB_ALLOW_SHRINK"):
    for _name, _state in (("rowloss", _row_loss_state), ("shrink", _shrink_state)):
        _state()
        _m[(_flag, _name)] = with_env({_flag: "1"}, lambda: webuidb.sync_once(force=True))
check(_m[("WEBUI_DB_ALLOW_ROW_LOSS", "rowloss")]["synced"] is True,
      "ALLOW_ROW_LOSS opens the per-row refusal")
check(_m[("WEBUI_DB_ALLOW_ROW_LOSS", "shrink")]["synced"] is False,
      "ALLOW_ROW_LOSS does NOT open the content-ratio refusal")
check(_m[("WEBUI_DB_ALLOW_SHRINK", "shrink")]["synced"] is True,
      "ALLOW_SHRINK opens the content-ratio refusal")
check(_m[("WEBUI_DB_ALLOW_SHRINK", "rowloss")]["synced"] is False,
      "ALLOW_SHRINK does NOT open the per-row refusal")

print("    the one-shot escape the refusal prints actually works")
_row_loss_state()
# Without --force, so the command is exactly the one printed. The snapshot is
# dated earlier than local explicitly, so the mtime skip cannot be what
# answers on a filesystem with coarse timestamps.
os.utime(SNAP, (time.time() - 600, time.time() - 600))
rc, out = run_cli(["--sync-once"])
check(rc == 1, f"CONTROL: without the flag, --sync-once exits 1 (rc={rc})")
rc, out = run_cli(["--sync-once"], {"WEBUI_DB_ALLOW_ROW_LOSS": "1"})
check(
    rc == 0 and "'synced': True" in out and chats(SNAP) == 1
    and content(SNAP) == content(LOCAL),
    f"WEBUI_DB_ALLOW_ROW_LOSS=1 webuidb.py --sync-once publishes it once (rc={rc})",
)

# ===========================================================================
print()
print("[D4] `--sync-once` (the CLI action) ALWAYS forces, so a restore that "
      "preserves an old mtime cannot make it silently do nothing")
# ===========================================================================
# findings.md D4. backup.py::restore_backup (and copy2/tar restores generally)
# preserve the ORIGINAL file's mtime, so a local database restored from an
# archive can carry a mtime OLDER than (or equal to) the snapshot's, even
# though its CONTENT is what the operator just deliberately put there.
# sync_once's periodic mtime-skip (`snap_mtime >= mtime` -> "unchanged since
# last sync") then answered that as "nothing to do" - and a plain
# `webuidb.py --sync-once` (no --force) printed exit 0 with `'skipped':
# 'unchanged since last sync'`, which LOOKS like success to anything that
# checks the exit code, while publishing nothing.
wipe()
owui(SNAP, [("c1", 2000, conversation(4))])   # the snapshot: newer mtime
owui(LOCAL, [("c1", 2000, conversation(4)), ("c2", 2001, conversation(4))])
# Restored "an hour ago" - copy2/tar-preserved, older than the snapshot's
# own mtime, exactly the shape a restore leaves behind.
os.utime(LOCAL, (time.time() - 3600, time.time() - 3600))
check(
    SNAP.stat().st_mtime > LOCAL.stat().st_mtime,
    "PRECONDITION: the snapshot's mtime is NEWER than local's, despite "
    "local holding an extra conversation - this is the state a copy2/tar "
    "restore leaves and the mtime-skip alone cannot tell apart from "
    "'nothing changed'",
)
rc, out = run_cli(["--sync-once"])
check(
    rc == 0 and "'synced': True" in out and "'skipped': None" in out,
    f"webuidb.py --sync-once (no --force flag) still PUBLISHES rather than "
    f"silently skipping on the stale mtime (rc={rc}, out={out.strip()!r})",
)
check(chats(SNAP) == 2, "and the extra conversation actually landed on the snapshot")

print("    CONTROL: the DAEMON's own periodic cycle still uses the ordinary "
      "mtime skip, unaffected by this fix - only the one-shot CLI action "
      "changed")
wipe()
owui(SNAP, [("c1", 2000, conversation(4))])
owui(LOCAL, [("c1", 2000, conversation(4))])
os.utime(LOCAL, (SNAP.stat().st_mtime - 5, SNAP.stat().st_mtime - 5))
r = webuidb.sync_once()  # bare call, force defaults to False - what sync_loop uses
check(
    r["skipped"] == "unchanged since last sync" and r["synced"] is False,
    f"a bare sync_once() (the daemon's own call, sync_loop never passes "
    f"force=True on an ordinary cycle) still skips on an unchanged mtime "
    f"(got {r})",
)


# ===========================================================================
print()
print("[N5] AN OLDER GENERATION OF THE SAME DATABASE IS REFUSED")
# ===========================================================================
# The snapshot is today's. Local is last week's copy of the same database put
# back by hand: her main conversation is there but older (fewer messages, an
# earlier updated_at), and two conversations started this week do not exist
# in it. Sized so that every magnitude guard holds: 42 -> 40 chats, a content
# ratio near 0.9, and the main conversation lost ~21 KB, far under the
# per-row limit. Only TIME has gone backwards.
def _older_generation_state():
    wipe()
    base = [(f"c{i:02d}", 1000 + i, conversation(2)) for i in range(40)]
    owui(SNAP, base + [
        ("main", 5000, conversation(200)),
        ("new-1", 4500, conversation(2)),
        ("new-2", 4600, conversation(2)),
    ])
    owui(LOCAL, base + [("main", 4000, conversation(180))])


_older_generation_state()
check(
    chats(LOCAL) >= chats(SNAP) * webuidb.SHRINK_REFUSE_BELOW
    and content(LOCAL) >= content(SNAP) * webuidb.SHRINK_REFUSE_BELOW
    and content(SNAP) - content(LOCAL) < MAX_ROW_LOSS,
    f"PRECONDITION: chats {chats(SNAP)} -> {chats(LOCAL)}, content ratio "
    f"{content(LOCAL) / content(SNAP):.3f}, loss "
    f"{content(SNAP) - content(LOCAL)} B - every magnitude guard holds",
)
_before = snap_bytes()
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and snap_bytes() == _before,
    f"refused, snapshot untouched (synced={r['synced']}, "
    f"error={(r['error'] or '')[:90]})",
)
check(
    bool(r["error"]) and "WEBUI_DB_ALLOW_OLDER_GENERATION" in r["error"]
    and "'main'" in r["error"],
    "and it names the conversation that went backwards and its own flag",
)

print("    CONTROL: the same database, moving FORWARD, publishes")
wipe()
_base = [(f"c{i:02d}", 1000 + i, conversation(2)) for i in range(40)]
owui(SNAP, _base + [("main", 5000, conversation(200))])
owui(LOCAL, _base + [("main", 6000, conversation(210)), ("new-1", 6100, conversation(2))])
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"published (error={(r['error'] or '')[:90]})")

print("    CONTROL: deleting her NEWEST conversation publishes")
# The case a table-wide `max(updated_at)` comparison refuses: the newest row
# is gone, so the new image's maximum is older than the snapshot's, and
# nothing went backwards - she deleted a chat. Ordinary use has to reach the
# snapshot (A4). Comparing each SURVIVING row against itself is what tells
# these apart.
wipe()
owui(SNAP, _base + [("main", 5000, conversation(200)), ("scratch", 9000, conversation(2))])
owui(LOCAL, _base + [("main", 5000, conversation(200))])
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"published (error={(r['error'] or '')[:90]})")

print("    CONTROL: no updated_at column cannot compare - it says so and publishes")
# The restore-guard suites' own fixtures have no updated_at, and neither might
# some future schema. "Cannot compare" refusing forever would be an outage
# built out of a guard.
wipe()
for _p, _n in ((SNAP, 5), (LOCAL, 6)):
    _p.parent.mkdir(parents=True, exist_ok=True)
    _c = sqlite3.connect(str(_p))
    _c.execute("create table chat (id text, marker text, body text)")
    _c.executemany("insert into chat values (?, 'm', ?)", [(str(i), "x" * 40) for i in range(_n)])
    _c.commit()
    _c.close()
r = webuidb.sync_once(force=True)
check(r["synced"] is True, f"published (error={(r['error'] or '')[:90]})")
check(
    logged(logging.WARNING, "updated_at"),
    "and a WARNING says the generation guard could not run",
)
rc, out = run_cli(["--status"])
check(
    rc == 0 and "no updated_at" in out,
    "and --status says the same thing, so it is visible without a log",
)

print("    each override opens only its own door")
_older_generation_state()
r = with_env({"WEBUI_DB_ALLOW_OLDER_GENERATION": "1"}, lambda: webuidb.sync_once(force=True))
check(r["synced"] is True, f"ALLOW_OLDER_GENERATION opens it (error={(r['error'] or '')[:90]})")
for _flag in ("WEBUI_DB_ALLOW_SHRINK", "WEBUI_DB_ALLOW_ROW_LOSS"):
    _older_generation_state()
    r = with_env({_flag: "1"}, lambda: webuidb.sync_once(force=True))
    check(r["synced"] is False, f"{_flag} does NOT open it")
_row_loss_state()
r = with_env({"WEBUI_DB_ALLOW_OLDER_GENERATION": "1"}, lambda: webuidb.sync_once(force=True))
check(r["synced"] is False, "ALLOW_OLDER_GENERATION does NOT open the per-row refusal")

print("    a real week-old copy trips BOTH row loss and generation, and needs both flags")
wipe()
owui(SNAP, [("main", 9000, conversation(4000))])
owui(LOCAL, [("main", 2000, conversation(2020))])
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and "WEBUI_DB_ALLOW_ROW_LOSS" in (r["error"] or "")
    and "WEBUI_DB_ALLOW_OLDER_GENERATION" in (r["error"] or ""),
    "the refusal names BOTH flags - naming one would send the operator round twice",
)
r = with_env({"WEBUI_DB_ALLOW_ROW_LOSS": "1"}, lambda: webuidb.sync_once(force=True))
check(r["synced"] is False, "one flag alone is still refused")
r = with_env({"WEBUI_DB_ALLOW_ROW_LOSS": "1", "WEBUI_DB_ALLOW_OLDER_GENERATION": "1"},
             lambda: webuidb.sync_once(force=True))
check(r["synced"] is True, f"both publish (error={(r['error'] or '')[:90]})")


# ===========================================================================
print()
print("[N2] A SNAPSHOT MTIME IN THE FUTURE DOES NOT FREEZE THE DURABLE COPY")
# ===========================================================================
wipe()
owui(LOCAL, [("main", 1000, conversation(10))])
_past = time.time() - 600
os.utime(LOCAL, (_past, _past))
r = webuidb.sync_once(force=True)
check(r["synced"] is True, "first publish")
check(
    abs(SNAP.stat().st_mtime - LOCAL.stat().st_mtime) < 1e-3,
    f"the snapshot is stamped with the local mtime it was imaged from "
    f"({SNAP.stat().st_mtime:.3f} vs {LOCAL.stat().st_mtime:.3f}) - the skip "
    f"below depends on it, and deleting the stamp was a surviving mutant",
)
r = webuidb.sync_once()
check(
    r["synced"] is False and r["skipped"] == "unchanged since last sync",
    f"CONTROL: nothing written since -> skipped ({r['skipped']!r})",
)

print("    the snapshot's mtime is an hour ahead: it publishes, and says why")
_future = time.time() + 3600
os.utime(SNAP, (_future, _future))
owui(LOCAL, [("main", 2000, conversation(12))])       # she kept talking
CAP.records.clear()
r = webuidb.sync_once()
check(
    r["synced"] is True and chats(SNAP) == 1 and content(SNAP) == content(LOCAL),
    f"not a skip (synced={r['synced']}, skipped={r['skipped']!r})",
)
check(
    logged(logging.WARNING, "in the FUTURE"),
    "and a WARNING names the clock - a silent republish would hide the "
    "same fault the silent skip did",
)
check(
    SNAP.stat().st_mtime <= time.time() + 1,
    "and the snapshot no longer carries a future mtime",
)
r = webuidb.sync_once()
check(
    r["skipped"] == "unchanged since last sync",
    f"and the next cycle is an ordinary skip again, not a republish forever "
    f"({r['skipped']!r}, error={r['error']})",
)

print("    the snapshot's mtime is an hour ahead and local has NOT changed: still not a skip")
_future = time.time() + 3600
os.utime(SNAP, (_future, _future))
r = webuidb.sync_once()
check(r["synced"] is True, f"published (skipped={r['skipped']!r})")

print("    a local mtime in the future is not copied onto the snapshot")
# The poison's source. The stamp is clamped to now; costs at most a republish
# per cycle until the local clock is right again, never a freeze.
_future = time.time() + 3600
os.utime(LOCAL, (_future, _future))
r = webuidb.sync_once(force=True)
check(
    r["synced"] is True and SNAP.stat().st_mtime <= time.time() + 1,
    f"stamp clamped (snapshot mtime - now = {SNAP.stat().st_mtime - time.time():.0f}s)",
)

print("    a few seconds ahead is timestamp granularity, not a wrong clock: still skips")
os.utime(LOCAL, (time.time() - 600, time.time() - 600))
webuidb.sync_once(force=True)
_near = time.time() + 30
os.utime(SNAP, (_near, _near))
r = webuidb.sync_once()
check(
    r["skipped"] == "unchanged since last sync",
    f"30 s ahead is inside the tolerance and skips ({r['skipped']!r}) - without "
    f"a tolerance, a filesystem that rounds mtimes up would rewrite the whole "
    f"database onto /data every cycle",
)


# ===========================================================================
print()
print("[FORENSIC] the shrink refusal's forensic copy is rate-limited, not")
print("           made on every refused cycle (SP\\lane-webuidb.md #3)")
# ===========================================================================
# UNBOUNDED, BEFORE THIS FIX: every refused cycle (lost_chats or
# lost_content) copied the WHOLE local database to QUARANTINE, and a
# refusal repeats every SYNC_INTERVAL_S until an operator acts - so an
# unaddressed regression copied the live-sized database onto /data forever
# (~9.5 GB/day measured at the live 33 MB size and the default 300 s
# interval). The FIRST copy of a new refusal must still be unconditional -
# it is the earliest evidence of whatever went wrong - and every copy after
# that, for the SAME ongoing refusal, is throttled to at most one per
# FORENSIC_COPY_MIN_INTERVAL_S.
check(
    webuidb.FORENSIC_COPY_MIN_INTERVAL_S == 3600,
    f"CONTROL: the default interval is 3600s (one hour), matching "
    f"sync_loop's own failure-alarm cadence a few lines down in the same "
    f"file (got {webuidb.FORENSIC_COPY_MIN_INTERVAL_S})",
)

wipe()
owui(SNAP, [(f"conv-{i}", 1000 + i, conversation(200)) for i in range(20)])
owui(LOCAL, [(f"conv-{i}", 1000 + i, conversation(200)) for i in range(2)])
check(
    chats(LOCAL) < chats(SNAP) * webuidb.SHRINK_REFUSE_BELOW
    and chats(SNAP) >= webuidb.SHRINK_GUARD_MIN_CHATS,
    f"PRECONDITION: 20 -> 2 chats trips the SHRINK (count) refusal "
    f"({chats(LOCAL)} < {chats(SNAP)} * {webuidb.SHRINK_REFUSE_BELOW})",
)
_before = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
r = webuidb.sync_once(force=True)
check(r["synced"] is False, f"refused (synced={r['synced']})")
_after_first = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
check(
    _after_first == _before + 1,
    f"the FIRST refusal of a new incident copies unconditionally - it is "
    f"the earliest evidence of whatever went wrong ({_before} -> {_after_first})",
)

print("    the SAME ongoing refusal, retried immediately, does NOT copy again")
for _ in range(5):
    r = webuidb.sync_once(force=True)
    check(r["synced"] is False, f"still refused (synced={r['synced']})")
_after_retries = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
check(
    _after_retries == _after_first,
    f"5 more refused cycles, 0 more forensic copies ({_after_first} -> "
    f"{_after_retries}) - unbounded, this would be +5 (~165 MB in this "
    f"fixture, ~9.5 GB/day at the live 33 MB conversation size)",
)

print("    CONTROL: a DIFFERENT, later incident still gets its own copy once "
      "the interval elapses - this is a rate limit, not a one-shot switch")
# Manipulating the module's own monotonic bookkeeping directly (white-box,
# consistent with how these suites already reach into EMPTY_START_MARKER
# and the ALLOW_* globals) to simulate real time passing without an actual
# 3600s sleep.
webuidb._forensic_copy_last_monotonic = (
    time.monotonic() - webuidb.FORENSIC_COPY_MIN_INTERVAL_S - 1
)
r = webuidb.sync_once(force=True)
check(r["synced"] is False, f"still refused (synced={r['synced']})")
_after_elapsed = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
check(
    _after_elapsed == _after_retries + 1,
    f"once the interval has elapsed, the NEXT refused cycle copies again "
    f"({_after_retries} -> {_after_elapsed}) - a slowly worsening state "
    f"still leaves a trail of samples, it is not silenced forever",
)

print("    the interval is a knob, and _reload_env reloads it")
r = with_env(
    {"WEBUI_DB_FORENSIC_COPY_MIN_INTERVAL_S": "0"},
    lambda: webuidb.sync_once(force=True),
)
check(r["synced"] is False, f"still refused (synced={r['synced']})")
_after_zero = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
check(
    _after_zero == _after_elapsed + 1,
    f"interval=0 copies on every refused cycle - a knob a test (or an "
    f"operator) forgot _reload_env would ignore this until the process "
    f"restarted ({_after_elapsed} -> {_after_zero})",
)
check(
    webuidb.FORENSIC_COPY_MIN_INTERVAL_S == 3600,
    "and the default is back afterwards (with_env restores it)",
)

print("    the forensic copy takes sidecars too, not just the main file")
# hostile pass #2 LOW ("what I checked and found sound", sidecar audit):
# this copied only LOCAL_DB, never -wal/-shm/-journal. In WAL mode recently
# committed rows can live in -wal until the next checkpoint, so a copy of
# the main file alone can be MISSING data that was never actually lost -
# the wrong gap for evidence collected because something looked like loss.
#
# A plain "write garbage bytes to a -wal path" fixture does NOT reproduce
# this: sync_once's own backup step opens and closes LOCAL_DB, and SQLite
# checkpoints-and-deletes a WAL file when its connection is the LAST one
# to close a WAL-mode database (verified directly, single-connection
# script: a stray -wal survives the OPEN but is gone the instant the one
# connection that touched it closes). In production it is never the last
# connection - OpenWebUI holds LOCAL_DB open the whole time - so a SECOND,
# still-open connection is what actually leaves committed-but-not-yet-
# checkpointed data in -wal while sync_once's own connection comes and
# goes. That is the fixture below, not a garbage byte string.
wipe()
owui(SNAP, [(f"conv-{i}", 1000 + i, conversation(200)) for i in range(20)])
owui(LOCAL, [(f"conv-{i}", 1000 + i, conversation(200)) for i in range(2)])
_wal = LOCAL.with_name(LOCAL.name + "-wal")
_owui_con = sqlite3.connect(str(LOCAL))
_owui_con.execute("PRAGMA journal_mode=WAL")
# UPDATE, not INSERT: OWUI_CHAT declares 16 columns and this only needs to
# generate real WAL content, not build a valid new row.
_owui_con.execute("update chat set updated_at = updated_at + 1")
_owui_con.commit()
try:
    check(
        _wal.exists() and _wal.stat().st_size > 0,
        f"PRECONDITION: OpenWebUI's still-open connection left real "
        f"committed data in -wal ({_wal.stat().st_size if _wal.exists() else 0} bytes)",
    )
    _wal_bytes_before = _wal.read_bytes()
    webuidb._forensic_copy_last_monotonic = None
    r = webuidb.sync_once(force=True)
    check(r["synced"] is False, f"refused (synced={r['synced']})")
    check(
        _wal.exists(),
        "and the ORIGINAL -wal is untouched - OpenWebUI's connection is "
        "still open, so sync_once's own connect+close did not checkpoint "
        "it away (the precondition this case depends on)",
    )
finally:
    _owui_con.close()  # simulate OpenWebUI eventually closing too

_before_delta = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
_main_copies = sorted(_QUAR.glob(f"{LOCAL.name}.refused-*"))
_wal_copies = sorted(_QUAR.glob(f"{LOCAL.name}-wal.refused-*"))
check(len(_main_copies) >= 1, f"the main file was copied ({len(_main_copies)})")
check(
    len(_wal_copies) >= 1 and _wal_copies[-1].read_bytes() == _wal_bytes_before,
    f"and so was its -wal sidecar, with byte-identical content "
    f"({len(_wal_copies)} copies)",
)
# Guarded, not chained onto the check above: an empty _wal_copies here means
# the previous check already failed and recorded it - indexing [-1] into an
# empty list would be a crash reported as this suite's own bug, not a red
# for the property being tested (brief: "a traceback ... is a wrong-reason
# red").
if _main_copies and _wal_copies:
    check(
        _main_copies[-1].name.rsplit(".refused-", 1)[1]
        == _wal_copies[-1].name.rsplit(".refused-", 1)[1],
        f"and they share ONE stamp, so the two files are identifiable as "
        f"one snapshot ({_main_copies[-1].name} / {_wal_copies[-1].name})",
    )
else:
    check(False, "cannot compare stamps - no sidecar copy exists to compare")

print("    CONTROL: row-loss and generation refusals still copy NOTHING - "
      "unaffected by this fix, unlike the shrink refusal above")
wipe()
owui(SNAP, [("the-conversation", 1000, conversation(4000))])
owui(LOCAL, [("the-conversation", 2000, conversation(2020))])
webuidb._forensic_copy_last_monotonic = None
_before_rowloss = len(list(_QUAR.glob(f"{LOCAL.name}.refused-*")))
r = webuidb.sync_once(force=True)
check(
    r["synced"] is False and bool(r["error"]) and "WEBUI_DB_ALLOW_ROW_LOSS" in r["error"],
    f"refused on row loss, not shrink (error={(r['error'] or '')[:60]!r})",
)
check(
    len(list(_QUAR.glob(f"{LOCAL.name}.refused-*"))) == _before_rowloss,
    "and still nothing copied - the row-loss/generation refusals were "
    "already deliberately copy-free (the snapshot, not the local database, "
    "holds the content in question there) and this fix must not change that",
)


# ===========================================================================
print()
print("[13b] _reload_env reloads every WEBUI_DB_* knob read at import")
# ===========================================================================
_src = (HERE / "webuidb.py").read_text(encoding="utf-8")
_top = _src[:_src.index("def _reload_env")]
_reload = _src[_src.index("def _reload_env"):_src.index("def _stamp")]
_knobs = set(re.findall(r"""env_(?:int|float)\(\s*["'](WEBUI_DB_\w+)["']""", _top))
_knobs |= set(re.findall(r"""os\.environ\.get\(\s*["'](WEBUI_DB_ALLOW_\w+)["']""", _top))
check(
    {"WEBUI_DB_MAX_ROW_LOSS_BYTES", "WEBUI_DB_ALLOW_ROW_LOSS",
     "WEBUI_DB_ALLOW_OLDER_GENERATION"} <= _knobs,
    f"CONTROL: the new knobs are found at import ({sorted(_knobs)})",
)
_missing = sorted(k for k in _knobs if k not in _reload)
check(not _missing, f"and every one is re-read by _reload_env (missing: {_missing})")


# ===========================================================================
print()
print("[SYNC_LOOP] a run of 'no local database yet' skips is counted and "
      "shouted; 'unchanged since last sync' never is")
# ===========================================================================
# hostile pass #2, MEDIUM: sync_loop never logged or counted a skip, so a
# daemon idle for the life of the pod was invisible - only the one startup
# line. That is exactly what a wrong WEBUI_LOCAL_DB looks like from in
# here: supervisord shows RUNNING, sync_once returns cleanly every cycle
# (error=None), nothing is ever published. The other skip reason,
# "unchanged since last sync", is NOT the same thing and must stay silent -
# it is what her being asleep looks like, correctly, every quiet night this
# pod runs, and alarming on it would train an operator to ignore this log
# within a week.


class _StopLoop(Exception):
    pass


def _drive_sync_loop(results):
    """Run the REAL sync_loop() with time.sleep patched to a no-op and
    sync_once patched to return each of `results` in turn, stopping via a
    sentinel exception once exhausted - sync_loop has no exit condition of
    its own. Returns nothing; assert on CAP.records afterward."""
    it = iter(results)

    def fake_sync_once(*_a, **_kw):
        try:
            return next(it)
        except StopIteration:
            raise _StopLoop()

    orig_sleep, orig_sync_once = time.sleep, webuidb.sync_once
    time.sleep = lambda _s: None
    webuidb.sync_once = fake_sync_once
    try:
        webuidb.sync_loop()
    except _StopLoop:
        pass
    finally:
        time.sleep = orig_sleep
        webuidb.sync_once = orig_sync_once


_NO_LOCAL = {"synced": False, "skipped": "no local database yet", "error": None, "bytes": 0}
_UNCHANGED = {"synced": False, "skipped": "unchanged since last sync", "error": None, "bytes": 0}
_SYNCED = {"synced": True, "skipped": None, "error": None, "bytes": 1000}

# `consecutive_no_local` is a LOCAL inside sync_loop(), reset to 0 on every
# call - so each scenario below drives ONE call with the WHOLE sequence of
# cycles it needs, never several short calls expecting the count to carry
# over between them (it cannot: there is nothing for it to live in between
# calls, which is also exactly why this counter cannot leak past a real
# process restart).
CAP.records.clear()
_drive_sync_loop([_NO_LOCAL] * 2)
check(
    not logged(logging.ERROR, "no local database"),
    "2 skips in a row: silent, same as the failure counter's own first-two "
    "grace period",
)

CAP.records.clear()
_drive_sync_loop([_NO_LOCAL] * 3)
check(
    logged(logging.ERROR, "no local database at") and logged(logging.ERROR, "3 cycles"),
    "the 3rd consecutive 'no local database yet' skip shouts, naming the "
    "path and the count - this is the reason [C8]'s wrong-WEBUI_LOCAL_DB "
    "case was invisible",
)
check(
    sum(1 for lv, msg in CAP.records if lv >= logging.ERROR and "no local database" in msg) == 1,
    "and shouts exactly ONCE for cycles 1-3, not once per cycle",
)

CAP.records.clear()
_drive_sync_loop([_NO_LOCAL] * 11)
check(
    sum(1 for lv, msg in CAP.records if lv >= logging.ERROR and "no local database" in msg) == 1,
    "cycles 4-11: still exactly the one shout from cycle 3 - quiet again, "
    "same shout-then-wait shape as the failure counter (3, then every 12th)",
)

CAP.records.clear()
_drive_sync_loop([_NO_LOCAL] * 12)
check(
    logged(logging.ERROR, "12 cycles"),
    "the 12th consecutive skip shouts again - not silent forever after the "
    "first shout, which is the ORIGINAL failure-counter defect this mirrors",
)
check(
    sum(1 for lv, msg in CAP.records if lv >= logging.ERROR and "no local database" in msg) == 2,
    "exactly two shouts across 12 cycles (at 3 and at 12), not more",
)

print("    CONTROL: 'unchanged since last sync' is NEVER counted or shouted")
CAP.records.clear()
_drive_sync_loop([_UNCHANGED] * 40)
check(
    not any(lv >= logging.WARNING for lv, _msg in CAP.records),
    f"40 ordinary 'unchanged' skips in a row: silent at WARNING or above "
    f"({len(CAP.records)} record(s) at any level, all INFO or below) - this "
    f"is what her being asleep looks like, and alarming on it is the noise "
    f"that gets a real alarm ignored",
)

print("    CONTROL: a successful publish resets the streak - the NEXT "
      "incident shouts at its own 3rd cycle, not immediately")
CAP.records.clear()
_drive_sync_loop([_NO_LOCAL, _NO_LOCAL, _SYNCED, _NO_LOCAL, _NO_LOCAL])
check(
    not logged(logging.ERROR, "no local database"),
    "2 skips, a real publish, then 2 more skips of a NEW incident: still "
    "quiet - proves the counter was reset by the successful publish rather "
    "than continuing from 2",
)
CAP.records.clear()
_drive_sync_loop([_NO_LOCAL, _NO_LOCAL, _SYNCED, _NO_LOCAL, _NO_LOCAL, _NO_LOCAL])
check(
    logged(logging.ERROR, "no local database at") and logged(logging.ERROR, "3 cycles"),
    "...and that new incident's OWN 3rd cycle shouts, exactly like the "
    "first incident did - the counter restarted, it did not resume from 2",
)

print("    CONTROL: an actual publish FAILURE still uses the pre-existing "
      "counter and cadence, unchanged by this fix")
CAP.records.clear()
_FAILED_R = {"synced": False, "skipped": None, "error": "RuntimeError: boom", "bytes": 0}
_drive_sync_loop([_FAILED_R] * 3)
check(
    logged(logging.ERROR, "snapshot publish has failed 3 times"),
    "the failure counter's own shout-at-3 still fires - this fix must not "
    "shadow it",
)

# ===========================================================================
print()
print("[D1] SIGTERM runs a final forced sync before the loop exits")
# ===========================================================================
# findings.md D1: webuidb-sync had no SIGTERM handling at all, and a publish
# (13-23s warm) is longer than RunPod's grace period. `supervisorctl stop`
# must now capture the last write automatically rather than relying on an
# operator remembering the manual final-sync step.

_order: list[str] = []


def _tracking_sleep(_s):
    _order.append("slept")


def _drive_with_sigterm(results, *, raise_on_call: int):
    """Like _drive_sync_loop, but instead of exhausting `results` the
    (1-indexed) call number `raise_on_call` raises webuidb._ShutdownRequested
    - standing in for a real SIGTERM landing there - and every OTHER call
    returns the next canned result. Returns the list of sync_once() calls
    actually observed (including the raising one) and whatever
    sync_once(force=...) was called with on each one."""
    it = iter(results)
    calls: list[dict] = []
    n = [0]

    def fake_sync_once(force=False):
        n[0] += 1
        calls.append({"n": n[0], "force": force})
        if n[0] == raise_on_call:
            raise webuidb._ShutdownRequested()
        return next(it)

    orig_sleep, orig_sync_once = time.sleep, webuidb.sync_once
    time.sleep = _tracking_sleep
    webuidb.sync_once = fake_sync_once
    try:
        webuidb.sync_loop()
    finally:
        time.sleep = orig_sleep
        webuidb.sync_once = orig_sync_once
    return calls


CAP.records.clear()
_order.clear()
calls = _drive_with_sigterm([_SYNCED, _SYNCED], raise_on_call=2)
check(
    len(calls) == 3,
    f"3 sync_once calls total: the immediate first sync, the cycle the "
    f"simulated SIGTERM landed on, and _final_sync_on_shutdown's own "
    f"call - and nothing after that (got {len(calls)})",
)
check(
    calls[0] == {"n": 1, "force": False} and calls[1]["n"] == 2
    and calls[2] == {"n": 3, "force": True},
    f"call 1 is the D11/D14 immediate first sync (force=False, an ordinary "
    f"cycle); call 2 is where the simulated SIGTERM landed; call 3 is the "
    f"final sync, forced (got {calls})",
)
check(
    _order == ["slept"],
    f"exactly one sleep happened, BEFORE the SIGTERM's own sync_once call - "
    f"call 1 (the immediate first sync) ran with NO preceding sleep "
    f"(got {_order})",
)
check(
    logged(logging.INFO, "final sync on SIGTERM"),
    "a final sync_once(force=True) ran and its result was logged",
)
_final_calls = [c for c in CAP.records if "final sync on SIGTERM" in c[1]]
check(len(_final_calls) == 1, "logged exactly once, not once per SIGTERM path")

print("    CONTROL: a SIGTERM landing INSIDE sync_once (not just during the "
      "sleep) still reaches the final sync - not swallowed by sync_once's "
      "own `except Exception`")
CAP.records.clear()
_order.clear()
calls = _drive_with_sigterm([_SYNCED], raise_on_call=1)
check(
    len(calls) == 2 and calls[0]["force"] is False and calls[1] == {"n": 2, "force": True},
    f"the SIGTERM landed on the very first (immediate) sync_once call, "
    f"before any sleep, and the final sync still ran right after it "
    f"(got {calls})",
)
check(
    logged(logging.INFO, "final sync on SIGTERM"),
    "and the final sync still ran - webuidb._ShutdownRequested is a "
    "BaseException, not an Exception, so it is not one sync_once's own "
    "internal `except Exception` could have caught and turned into an "
    "ordinary result",
)

print("    CONTROL: the final sync always forces, even though every ordinary "
      "cycle above used force=False")
CAP.records.clear()
_forced = []
_orig_sync_once = webuidb.sync_once
webuidb.sync_once = lambda force=False: (_forced.append(force), _SYNCED)[1]
try:
    webuidb._final_sync_on_shutdown()
finally:
    webuidb.sync_once = _orig_sync_once
check(_forced == [True], f"_final_sync_on_shutdown always calls force=True (got {_forced})")

print("    CONTROL: _final_sync_on_shutdown never raises, even when "
      "sync_once itself blows up - the process is exiting either way")
webuidb.sync_once = lambda force=False: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    webuidb._final_sync_on_shutdown()
    _raised = False
except Exception:
    _raised = True
finally:
    webuidb.sync_once = _orig_sync_once
check(not _raised, "no exception escaped _final_sync_on_shutdown")
check(
    logged(logging.ERROR, "final sync on SIGTERM raised"),
    "and the failure was logged instead of silently disappearing",
)

# ---------------------------------------------------------------------------
print()
print("[SYNC_UNSTATABLE] sync_once's own guard-scan refuses an UNSTATABLE "
      "snapshot instead of guessing 'no previous copy, nothing to compare'")
# ===========================================================================
# v3.1.9 round 2 (fix-webuidb.md's own "Found, not fixed" #3, second bullet -
# arguably higher severity than the restore_on_boot site fixed alongside it:
# this one can publish COMPLETELY UNGUARDED). SNAPSHOT_DB.exists() swallows
# every OSError and answers False for "the mount errored" exactly as it does
# for "nothing published yet" - and `previous is None` a few lines below
# means "first publish, nothing to compare against", which stands EVERY
# content guard down (shrink ratio, per-row loss, generation). A TRANSIENT
# stat failure on the flakiest volume in the system used to let a publish
# through with none of them running at all.
wipe()
owui(LOCAL, [("c1", 100, conversation(20))])
owui(SNAP, [("c1", 90, conversation(20))])
_before = snap_bytes()
_orig_presence = webuidb._presence


def _unstatable_presence(path):
    # Only SNAPSHOT_DB is simulated as unstatable - LOCAL_DB's own
    # _presence() calls (there are none on this path today, but a future
    # caller should not be silently redirected) fall through to the real
    # implementation.
    if path is SNAP:
        return (False, True)
    return _orig_presence(path)


CAP.records.clear()
webuidb._presence = _unstatable_presence
try:
    r = webuidb.sync_once(force=True)
finally:
    webuidb._presence = _orig_presence
check(
    r["synced"] is False and bool(r["error"]),
    f"REFUSES the publish rather than treating an unstatable snapshot as "
    f"'absent, nothing to compare' (synced={r['synced']!r}, "
    f"error={r['error']!r})",
)
check(
    "cannot stat" in (r["error"] or ""),
    f"and names the reason (got: {r['error']!r})",
)
check(
    snap_bytes() == _before,
    "and the snapshot file on disk is byte-for-byte untouched - the refusal "
    "happens before os.replace, same as every other guard in this function",
)

print("    CONTROL: the SAME local/snapshot pair, real _presence(), "
      "publishes normally")
# Without this, [SYNC_UNSTATABLE] could be passing because sync_once now
# refuses EVERY publish against this fixture for some unrelated reason (a
# typo in the fixture, an unrelated guard tripping) rather than specifically
# because of the simulated stat failure.
r2 = webuidb.sync_once(force=True)
check(
    r2["synced"] is True,
    f"unpatched, this exact local/snapshot pair publishes fine (r={r2}) - "
    f"the refusal above was specific to the simulated stat failure, not "
    f"something else about this fixture",
)

print("    CONTROL: a genuinely ABSENT snapshot is still 'first publish, "
      "go' (A8) - this fix must not turn into 'refuse whenever previous "
      "state is uncertain'")
wipe()
owui(LOCAL, [("c1", 100, conversation(20))])
r3 = webuidb.sync_once(force=True)
check(
    r3["synced"] is True and SNAP.exists(),
    f"a real first publish (no snapshot anywhere) still goes through "
    f"(r={r3})",
)


print()
if FAILED:
    print("!" * 66)
    for f in FAILED:
        print("FAIL " + f)
    print("!" * 66)
    sys.exit(1)
print("All webui.db publish-guard checks passed.")
