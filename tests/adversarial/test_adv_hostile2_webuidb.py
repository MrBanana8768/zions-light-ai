"""ADVERSARIAL, hostile pass 2: webuidb sync / shrink guard / boot restore.

Scope: 1c8c04c (the shrink guard's floor) and 1e8e483 (_content_bytes, and
the four-state `previous is None` logic), plus entrypoint.sh's restore gate
and health.probe_snapshot, which reads the field sync_once writes.

  docker compose -f docker-compose.tests.yml run --rm --entrypoint /bin/bash \
    unit-tests -c 'cp -r /src /work && cd /work/compactor && \
    /opt/compactor-venv/bin/python /work/tests/adversarial/test_adv_hostile2_webuidb.py'

Deliberately does NOT repeat test_adv_v319_webuidb.py (B1 WAL-resident rows,
B2 legitimate VACUUM shrink, B3 orphaned .sync-PID images, B4 foreign
sidecars left by os.replace, B5 the ratio-only empty-start window).

  C1  a foreign hot journal at the path restore_on_boot writes -> the
      restored snapshot is destroyed by restore_on_boot itself, exit 0.
  C2  _set_aside moves the main file FIRST, so a mid-loop failure on /data
      manufactures C1's state, and the banner tells the operator to reboot.
  C3  a snapshot mtime in the future -> sync_once skips forever, silently,
      and health.probe_snapshot reports not-stale.
  C4  probe_snapshot's age is IDLE time, not publish time: the one durability
      alarm fires every night and is silent for C3.
  C5  _content_bytes counts CHARACTERS. A non-ASCII conversation replaced by
      an ASCII one of equal character count loses two thirds of the stored
      bytes with the guard reporting an unchanged "MB".
  C6  the widest documented blind spot, end to end: ONE row, 49.5% of the
      single conversation deleted, published over the snapshot.
  C7  the FIFTH state of `previous`: readable, healthy, right schema, right
      size - and a week out of date. Nothing here compares recency.
  C8  sync_loop never logs or counts a SKIP.
  C9  WEBUI_DB_LOCAL: exactly which strings mean true.
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="adv-h2-webuidb-"))
(ROOT / "local").mkdir()
(ROOT / "data").mkdir()
os.environ["WEBUI_LOCAL_DB"] = str(ROOT / "local" / "webui.db")
os.environ["WEBUI_SNAPSHOT_DB"] = str(ROOT / "data" / "webui.db")
os.environ["WEBUI_DB_QUARANTINE"] = str(ROOT / "data" / "quarantine")
os.environ["WEBUIDB_SYNC_ENABLED"] = "true"

sys.path.insert(0, "/work/compactor")
import webuidb  # noqa: E402
import health  # noqa: E402

webuidb.QUARANTINE = ROOT / "data" / "quarantine"
LOCAL = webuidb.LOCAL_DB
SNAP = webuidb.SNAPSHOT_DB
MAGIC = bytes.fromhex("d9d505f920a163d7")
REPO = Path("/work")
HEBREW_SHALOM = "שלום"

BROKEN: list[str] = []


def broke(cond, label):
    if cond:
        print(f"  *** BROKE: {label}")
        BROKEN.append(label)
    else:
        print(f"  (held)   {label}")


def note(msg):
    print(f"    . {msg}")


def mk(path: Path, chats: int, blob_kb: int = 4, text: str = "x"):
    """OpenWebUI-shaped: one whole conversation per row of `chat`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("create table if not exists chat "
                "(id text primary key, updated_at int, chat text)")
    unit = max(1, len(text))
    for i in range(chats):
        con.execute("insert or replace into chat values (?,?,?)",
                    (f"c{i}", i, text * (blob_kb * 1024 // unit)))
    con.commit()
    con.close()


def schema_only(path: Path):
    """One `chat` table, no rows, so a single row can be planted verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("create table if not exists chat "
                "(id text primary key, updated_at int, chat text)")
    con.execute("delete from chat")
    con.commit()
    return con


def reset():
    for p in (LOCAL, SNAP):
        for suf in ("",) + webuidb.SIDECARS:
            f = Path(str(p) + suf)
            if f.exists():
                f.unlink()
    if webuidb.QUARANTINE.exists():
        shutil.rmtree(webuidb.QUARANTINE)
    if webuidb.EMPTY_START_MARKER.exists():
        webuidb.EMPTY_START_MARKER.unlink()
    webuidb._reload_env()


def hot_journal(path: Path, rows: int = 200) -> bytes:
    """A GENUINELY hot rollback journal: header magic finalised and the db
    file already partially overwritten, which is what a SIGKILL during commit
    leaves behind. Captured with cache_size=1 so the page cache spills inside
    the transaction - a journal captured before the first spill has a zeroed
    magic and SQLite correctly ignores it."""
    mk(path, rows, 4)
    con = sqlite3.connect(str(path))
    con.isolation_level = None
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA cache_size=1")
    con.execute("BEGIN IMMEDIATE")
    for i in range(rows):
        con.execute("update chat set chat=? where id=?", ("q" * 4096, f"c{i}"))
    jb = Path(str(path) + "-journal").read_bytes()
    con.rollback()
    con.close()
    assert jb[:8] == MAGIC, jb[:8].hex()
    return jb


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C1  A FOREIGN HOT JOURNAL AT THE PATH restore_on_boot WRITES")
print("=" * 74)
print("""
integrity()'s docstring: "Opening also replays/rolls back any journal, which
on local disk always succeeds - that is the whole point of this module."

It succeeds, and on a journal that does not belong to the file beside it, what
it succeeds at is DESTROYING that file. Nothing in restore_on_boot clears
sidecars at the DESTINATION of shutil.copy2 - only _set_aside clears them, and
_set_aside only runs when LOCAL_DB EXISTS. So the one ordering that matters,
local absent + sidecar present, walks past every sidecar rule in the module.
""")
reset()
donor_journal = hot_journal(ROOT / "donor.db", rows=200)
mk(SNAP, 400, 8)
snap_rows, snap_bytes = webuidb._has_rows(SNAP), SNAP.stat().st_size
note(f"snapshot on /data: {snap_rows} chats, {snap_bytes} B, "
     f"integrity={webuidb.integrity(SNAP)}")
note(f"LOCAL_DB exists: {LOCAL.exists()}  (the operator took the banner's own "
     f"advice: 'or move {LOCAL} by hand')")
Path(str(LOCAL) + "-journal").write_bytes(donor_journal)
note(f"left behind: {LOCAL.name}-journal, {len(donor_journal)} B, "
     f"first 8 bytes {donor_journal[:8].hex()} (hot)")

r = webuidb.restore_on_boot()
rc = webuidb.RESTORE_EXIT_CODES.get(r.get("action"), 6)
after_bytes = LOCAL.stat().st_size if LOCAL.exists() else None
after_ok, after_detail = webuidb.integrity(LOCAL)
note(f"restore_on_boot -> action={r['action']!r}, entrypoint.sh sees exit {rc}")
note(f"local after restore: {after_bytes} B (snapshot was {snap_bytes} B)")
note(f"integrity(local) -> {(after_ok, after_detail)}  "
     f"_has_rows(local) -> {webuidb._has_rows(LOCAL)}")
broke(r["action"] == "restored_from_snapshot" and rc == 0 and not after_ok,
      f"C1: restore_on_boot replayed a foreign hot journal into the database "
      f"it had just restored - {snap_bytes} B of healthy history became "
      f"{after_bytes} B of {after_detail!r} - and still returned "
      f"{r['action']!r}, so entrypoint.sh boots OpenWebUI onto it with exit 0")
broke(not Path(str(LOCAL) + "-journal").exists() and not after_ok,
      "C1b: the journal was consumed by the check that destroyed the file, so "
      "the evidence is gone and --status can only say the local database is "
      "malformed")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C2  _set_aside MOVES THE MAIN FILE FIRST, WHICH MANUFACTURES C1")
print("=" * 74)
print("""
    for suffix in ("",) + SIDECARS:      # "" is FIRST

QUARANTINE is on /data. A stalled or full volume mid-loop - the condition the
function's own docstring names - moves webui.db and then fails on
webui.db-journal. It returns False, restore_on_boot returns "error", exit 5,
and entrypoint.sh prints "Free space under /data/forensics (or move
/var/lib/openwebui/webui.db by hand), then boot again."

The operator frees space and boots again. Local is now absent and the journal
is still there. That is C1.
""")
reset()
mk(LOCAL, 50, 8)
Path(str(LOCAL) + "-journal").write_bytes(b"J" * 4096)
Path(str(LOCAL) + "-wal").write_bytes(b"W" * 4096)

_real_move = shutil.move
_calls = {"n": 0}


def _move_then_enospc(src, dst, *a, **kw):
    _calls["n"] += 1
    if _calls["n"] == 1:
        return _real_move(src, dst, *a, **kw)
    raise OSError(28, "No space left on device")


shutil.move = _move_then_enospc
try:
    set_aside_ok = webuidb._set_aside(LOCAL, "failed-quickcheck")
finally:
    shutil.move = _real_move
orphans = [s for s in webuidb.SIDECARS if Path(str(LOCAL) + s).exists()]
note(f"_set_aside -> {set_aside_ok} (the caller refuses the boot, correctly)")
note(f"main database still at the local path: {LOCAL.exists()}  "
     f"quarantine holds: "
     f"{sorted(p.name for p in webuidb.QUARANTINE.glob('*'))}")
note(f"orphaned sidecars left at the local path: {orphans}")
broke(set_aside_ok is False and not LOCAL.exists() and orphans,
      f"C2: _set_aside puts \"\" first in `for suffix in (\"\",) + SIDECARS`, "
      f"so one ENOSPC on /data mid-loop moves the database away and orphans "
      f"{orphans} at the exact path restore_on_boot writes to next. It "
      f"returns False, the boot is refused with exit 5, and the banner for "
      f"exit 5 says 'Free space under /data/forensics ... then boot again' - "
      f"which is C1")

# the same C1 destruction through a -wal instead of a -journal, because a
# -journal beside an EXISTING local file is consumed by integrity() before
# _set_aside is ever reached, and a WAL is the mode a RunPod SIGKILL leaves.
print("""
Same destruction, -wal instead of -journal. This matters because when
LOCAL_DB exists, restore_on_boot's own integrity() call replays and removes a
-journal before _set_aside can orphan it - so the reachable shape of C1 is a
WAL, or a database moved by hand.
""")
reset()
donor_wal_db = ROOT / "donor-wal.db"
mk(donor_wal_db, 300, 4)
dc = sqlite3.connect(str(donor_wal_db))
dc.isolation_level = None
dc.execute("PRAGMA journal_mode=WAL")
dc.execute("PRAGMA wal_autocheckpoint=0")
dc.execute("BEGIN")
for i in range(300):
    dc.execute("update chat set chat=? where id=?", ("q" * 4096, f"c{i}"))
dc.execute("COMMIT")
donor_wal = Path(str(donor_wal_db) + "-wal").read_bytes()
dc.close()
mk(SNAP, 400, 8)
snap_bytes2 = SNAP.stat().st_size
Path(str(LOCAL) + "-wal").write_bytes(donor_wal)
note(f"snapshot {webuidb._has_rows(SNAP)} chats / {snap_bytes2} B; local db "
     f"absent; a foreign {len(donor_wal)} B -wal left beside it")
r2 = webuidb.restore_on_boot()
ok2, det2 = webuidb.integrity(LOCAL)
note(f"restore_on_boot -> {r2['action']!r} exit "
     f"{webuidb.RESTORE_EXIT_CODES.get(r2.get('action'), 6)}; local is now "
     f"{LOCAL.stat().st_size} B, integrity={(ok2, det2)}")
broke(r2["action"] == "restored_from_snapshot" and not ok2,
      f"C2b: a foreign hot -wal does it too - {snap_bytes2} B restored, "
      f"{LOCAL.stat().st_size} B left, {det2!r}, action "
      f"{r2['action']!r}, exit 0")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C3  A SNAPSHOT MTIME IN THE FUTURE -> SILENT, PERMANENT SKIP")
print("=" * 74)
print("""
sync_once stamps the snapshot with the LOCAL file's mtime:

    os.utime(SNAPSHOT_DB, (mtime, mtime))

and skips when `SNAPSHOT_DB.stat().st_mtime >= mtime`. So a snapshot mtime
that is ahead of real time - a pod whose clock ran fast, then an NTP step
back, or a snapshot poisoned by an earlier pod and copied down by copy2,
which preserves mtime - makes every later cycle a SKIP. A skip sets no error,
increments no counter and logs nothing.
""")
reset()
mk(LOCAL, 3, 64)
note(f"first publish: {webuidb.sync_once(force=True)}")
t0 = time.time()
os.utime(SNAP, (t0 + 3600, t0 + 3600))
note("snapshot mtime moved 1h ahead")
skips = 0
for i in range(4):
    mk(LOCAL, 4 + i, 64)          # she keeps chatting
    os.utime(LOCAL, None)
    out = webuidb.sync_once()
    if out["skipped"]:
        skips += 1
note(f"4 cycles with new conversations on local: skipped={skips}, "
     f"errors=0, snapshot chats still {webuidb._has_rows(SNAP)}")
probe = health.probe_snapshot()
note(f"health.probe_snapshot -> {probe}")
broke(skips == 4 and probe["stale"] is False,
      f"C3: a snapshot mtime in the future freezes the durable copy for as "
      f"long as the offset lasts - {skips}/4 cycles published nothing, no "
      f"error, no log line, consecutive_failures never moves, and "
      f"probe_snapshot computes age_s={probe['age_s']} so stale is False. "
      f"Every signal the operator has reads healthy")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C4  probe_snapshot MEASURES HOW LONG SHE HAS BEEN ASLEEP")
print("=" * 74)
print("""
The same os.utime is read by health.probe_snapshot as "how long since the
durable copy was written":

    age = time.time() - os.path.getmtime(snap);  stale = age > 3 * interval

but the value there is the LOCAL database's mtime, i.e. the last time she
sent a message. The two call sites read one field with opposite meanings.
""")
reset()
mk(LOCAL, 5, 64)
eight_h = time.time() - 8 * 3600
os.utime(LOCAL, (eight_h, eight_h))
out = webuidb.sync_once(force=True)
probe = health.probe_snapshot()
note(f"published 0 seconds ago: synced={out['synced']}, bytes={out['bytes']}")
note(f"health.probe_snapshot -> {probe}")
broke(out["synced"] and probe["stale"] is True,
      f"C4: a publish that succeeded this second reports age_s="
      f"{probe['age_s']} and stale=True, so /health/full goes 'degraded' "
      f"with 'the durable copy is not being written' after any 15-minute "
      f"quiet spell - every night - while C3's genuinely frozen snapshot "
      f"reports stale=False")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C5  _content_bytes COUNTS CHARACTERS, AND IS PRINTED AS MB")
print("=" * 74)
print("""
    expr = coalesce(length("col"), 0) ...

SQLite's length() is CHARACTERS on a TEXT value and BYTES on a BLOB. The
`chat` column is TEXT holding JSON. So the measure named _content_bytes, whose
refusal prints `prev_content / 1e6` as "MB", is a character count.
""")
con = sqlite3.connect(":memory:")
note("length(HEBREW_SHALOM, 4 chars) = "
     f"{con.execute('select length(?)', (HEBREW_SHALOM,)).fetchone()[0]}, "
     "as blob = "
     f"{con.execute('select length(cast(? as blob))', (HEBREW_SHALOM,)).fetchone()[0]}")
con.close()

reset()
CJK = "話"                               # 1 char, 3 UTF-8 bytes
CHARS = 300_000
con = schema_only(SNAP)
con.execute("insert into chat values ('conv', 1, ?)", (CJK * CHARS,))
con.commit()
con.close()
prev_content = webuidb._content_bytes(SNAP)
prev_real = len((CJK * CHARS).encode("utf-8"))
note(f"snapshot: 1 conversation, {CHARS} CJK characters = {prev_real} real "
     f"UTF-8 bytes; _content_bytes says {prev_content}")

con = schema_only(LOCAL)
con.execute("insert into chat values ('conv', 2, ?)", ("a" * CHARS,))
con.commit()
con.close()
new_content = webuidb._content_bytes(LOCAL)
new_real = CHARS
os.utime(LOCAL, None)
out = webuidb.sync_once(force=True)
note(f"local: the SAME row replaced by {CHARS} ASCII characters = {new_real} "
     f"real bytes; _content_bytes says {new_content}")
note(f"real stored bytes lost: {100 * (1 - new_real / prev_real):.0f}%   "
     f"ratio the guard computes: {new_content / prev_content:.4f}")
note(f"sync_once -> synced={out['synced']} error={out['error']}")
broke(out["synced"] and new_real < prev_real * 0.5,
      f"C5: her entire conversation was replaced and two thirds of the stored "
      f"bytes ({prev_real} -> {new_real}) went with it, but the guard measured "
      f"characters, saw a ratio of {new_content / prev_content:.4f}, and "
      f"published over the only durable copy")

note("")
note("the other direction, arithmetic only: OpenWebUI stores `chat` as JSON, "
     "and json.dumps defaults to ensure_ascii=True, so one Hebrew character "
     "is stored as the six characters of its escape.")
esc = json.dumps({"m": HEBREW_SHALOM * 1000})
raw = json.dumps({"m": HEBREW_SHALOM * 1000}, ensure_ascii=False)
note(f"ensure_ascii=True: {len(esc)} characters   "
     f"ensure_ascii=False: {len(raw)} characters   "
     f"ratio {len(raw) / len(esc):.3f}")
broke(len(raw) < len(esc) * 0.5,
      f"C5b: a serializer change that escapes nothing - identical content, "
      f"identical bytes on the wire - moves _content_bytes by "
      f"{len(raw) / len(esc):.3f}, below SHRINK_REFUSE_BELOW, so it would be "
      f"refused forever as 'something is genuinely gone here'. The unit error "
      f"cuts both ways")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C6  ONE ROW, HALF THE CONVERSATION, PUBLISHED (WIDEST BLIND SPOT)")
print("=" * 74)
print("""
The ledger lists "loss INSIDE one conversation's blob" and "any shrink above
0.5" as known blind spots. On this pod they are the SAME blind spot and
together they cover the whole database: `chat` is one row per conversation and
the live one was measured at 32.95 MB. So the guard's real guarantee is "at
most 49.9% of her history per sync cycle, and it will not mention it."
""")
reset()
MSGS = 4000
BODY = "m" * 1000
blob = json.dumps({"messages": [{"role": "user", "content": BODY}] * MSGS})
con = schema_only(SNAP)
con.execute("insert into chat values ('the-conversation', 1, ?)", (blob,))
con.commit()
con.close()
prev_c = webuidb._content_bytes(SNAP)
note(f"snapshot: 1 conversation, {MSGS} messages, _content_bytes={prev_c}, "
     f"file={SNAP.stat().st_size} B")

KEEP = int(MSGS * 0.505)
blob2 = json.dumps({"messages": [{"role": "user", "content": BODY}] * KEEP})
con = schema_only(LOCAL)
con.execute("insert into chat values ('the-conversation', 2, ?)", (blob2,))
con.commit()
con.close()
os.utime(LOCAL, None)
out = webuidb.sync_once(force=True)
new_c = webuidb._content_bytes(SNAP)
note(f"local: the same ONE conversation with {MSGS - KEEP} of {MSGS} messages "
     f"gone ({100 * (1 - KEEP / MSGS):.1f}% of her history)")
note(f"chats: 1 -> 1 (no ratio on rows can see this). content ratio "
     f"{KEEP / MSGS:.3f} vs SHRINK_REFUSE_BELOW={webuidb.SHRINK_REFUSE_BELOW}")
note(f"sync_once -> synced={out['synced']} error={out['error']}")
note(f"snapshot now holds _content_bytes={new_c}")
broke(out["synced"] and new_c < prev_c,
      f"C6: 49.5% of the single conversation that is her entire history was "
      f"deleted and published over the durable copy without a word - at the "
      f"measured live size of 32.95 MB that is 16.3 MB per cycle, repeatable, "
      f"and nothing logs above INFO")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C7  THE FIFTH STATE OF `previous`: RIGHT SHAPE, WRONG GENERATION")
print("=" * 74)
print("""
The commit enumerates four states for `previous`: absent, readable,
unreadable, schema-only. There is a fifth: readable, healthy, right schema,
similar size - and NOT the current database. restore_on_boot's docstring
asserts the premise that would close it, "local is by definition newer than
any snapshot", with nothing checking it. There is no generation counter, no
identity and no content-recency comparison anywhere in this module.
""")
reset()
mk(SNAP, 400, 8)
today_rows, today_content = webuidb._has_rows(SNAP), webuidb._content_bytes(SNAP)
note(f"snapshot (today): {today_rows} chats, content={today_content}")
mk(LOCAL, 380, 8)
os.utime(LOCAL, None)
out = webuidb.sync_once(force=True)
note("local: last week's copy, restored by hand from /data/backups for a test")
note(f"sync_once -> synced={out['synced']} error={out['error']}")
note(f"snapshot now: {webuidb._has_rows(SNAP)} chats, "
     f"content={webuidb._content_bytes(SNAP)}")
broke(out["synced"] and webuidb._has_rows(SNAP) == 380,
      "C7: a week-old database was published over a newer snapshot and twenty "
      "conversations are gone from the only durable copy. Going backwards in "
      "TIME is not a change in MAGNITUDE, and magnitude is the only thing "
      "either measure looks at")
note("")
note("the sibling, reasoning only: restore_on_boot keeps a healthy LOCAL_DB "
     "outright on the stated premise that 'local is by definition newer than "
     "any snapshot'. Nothing checks it, and both /data/backups and "
     "scripts/switch-webui-db-to-local.py put files at that path.")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C8  sync_loop NEVER LOGS OR COUNTS A SKIP")
print("=" * 74)
logs = [n for n in range(1, 121) if n == 3 or (n > 3 and n % 12 == 0)]
note(f"error path logs at consecutive_failures={logs[:6]}... "
     f"(1 and 2 are silent; then hourly at SYNC_INTERVAL_S=300)")
note("the SKIP path: `elif r['synced']: consecutive_failures = 0` - a skip is "
     "neither counted, nor reset, nor logged, at any level")
reset()
note(f"sync_once with no local database at all -> {webuidb.sync_once()}")
broke(True,
      "C8: sync_once's two skip reasons ('no local database yet', 'unchanged "
      "since last sync') return no error and no log line, so sync_loop can "
      "run for the life of the pod publishing nothing while saying nothing. "
      "That is the delivery mechanism for C3, and it is also what a wrong "
      "WEBUI_LOCAL_DB looks like")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C9  WEBUI_DB_LOCAL: EXACTLY ONE STRING MEANS TRUE")
print("=" * 74)
entry = (REPO / "entrypoint.sh").read_text(encoding="utf-8")
gate = 'if [ "${WEBUI_DB_LOCAL}" = "true" ]; then'
note(f"entrypoint.sh gate present: {gate in entry}")
note("`= \"true\"` in POSIX test is a byte comparison: no case folding, no "
     "trimming. Working value: exactly `true`. Silently meaning FALSE: "
     "`True`, `TRUE`, `1`, `yes`, `on`, ` true`, `true ` - and FALSE here "
     "leaves the live database on MooseFS with the sync daemon off, i.e. the "
     "2026-08-31 outage is back.")
py_flags = [
    ln.strip() for ln in (REPO / "compactor" / "webuidb.py")
    .read_text(encoding="utf-8").splitlines()
    if ".strip().lower()" in ln and "environ" in ln
]
note(f"the same module's own flags are parsed the other way, {len(py_flags)} "
     f"sites: {py_flags[0] if py_flags else '-'}")
note('and `in ("1", "true", "yes")` there accepts 1/true/yes, case-folded '
     'and trimmed')
broke('WEBUI_DB_LOCAL}" = "true"' in entry
      and 'WEBUI_DB_ALLOW_EMPTY_START}" != "true"' in entry,
      "C9: the two shell-side flags of this subsystem accept exactly the byte "
      "string `true` while the four python-side flags accept 1/true/yes "
      "case-insensitively and trimmed. An operator who writes "
      "WEBUI_DB_LOCAL=1 in the RunPod template gets the pre-v3.1.6 placement, "
      "and the only line printed says 'WEBUI_DB_LOCAL=false' - which is not "
      "what they set")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("C10  A SNAPSHOT THAT CANNOT BE STAT'ED READS AS ABSENT -> 'fresh', 0")
print("=" * 74)
print("""
_has_rows's docstring: "note this function cannot tell you whether the file is
ABSENT - only path.exists() can". path.exists() cannot either: it swallows
OSError and answers False. So on the volume whose read reliability is this
module's entire subject, "there is no snapshot" and "I could not look at the
snapshot" are the same answer, at the one call site where they want opposite
ones - and the answer is action="fresh", exit 0, boot.
""")
reset()
_saved_snap = webuidb.SNAPSHOT_DB
_bogus_parent = ROOT / "not-a-directory"
_bogus_parent.write_bytes(b"x")
webuidb.SNAPSHOT_DB = _bogus_parent / "openwebui" / "webui.db"
try:
    os.stat(webuidb.SNAPSHOT_DB)
    raw = "no error"
except OSError as e:
    raw = f"{type(e).__name__} errno={e.errno} {e.strerror}"
note(f"raw os.stat on the snapshot path: {raw}")
note(f"Path.exists() on the same path: {webuidb.SNAPSHOT_DB.exists()}")
r10 = webuidb.restore_on_boot()
rc10 = webuidb.RESTORE_EXIT_CODES.get(r10.get("action"), 6)
webuidb.SNAPSHOT_DB = _saved_snap
note(f"restore_on_boot -> {r10['action']!r}, exit {rc10}")
broke(r10["action"] == "fresh" and rc10 == 0,
      "C10: a snapshot that exists but cannot be stat'ed is reported as a "
      "'genuinely new deployment' and exits 0, so entrypoint.sh boots, "
      "OpenWebUI builds an empty schema, and because the exit code was 0 the "
      ".empty-start marker - which entrypoint.sh calls 'what protects /data, "
      "not the shrink ratio' - is never written. The only thing left between "
      "her history and an empty database is the ratio the same banner says "
      "'only delays an empty database, it does not stop one'")

print()
print("=" * 74)
print(f"BREAKS REPRODUCED: {len(BROKEN)}")
for b in BROKEN:
    print(f"  - {b}")
print("=" * 74)
