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


print()
if FAILED:
    print("!" * 66)
    for f in FAILED:
        print("FAIL " + f)
    print("!" * 66)
    sys.exit(1)
print("All webui.db publish-guard checks passed.")
