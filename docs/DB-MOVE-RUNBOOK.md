# Moving her chat database to the pod's local disk (v3.1.9.7)

**Status: current for v3.1.9.7.** This is the rehearsal runbook
(`/home/drew/zl-ops/dbmove/DB-MOVE-RUNBOOK.md`) brought into the repo and
refreshed against this release's actual code (review1-v3197-65ea196, B-2).
Linked from RUNPOD_DEPLOY.md's "Upgrading to v3.1.9.7" and "WEBUI_DB_LOCAL"
sections.

**What changes:**
- OpenWebUI's live database moves from the network volume (`/data/openwebui/webui.db`) to the pod's own disk (`/var/lib/openwebui/webui.db`).
- A background program, `webuidb-sync`, copies it back to `/data/openwebui/webui.db` every `WEBUI_DB_SYNC_INTERVAL_S` seconds (recommended: 120).
- That `/data` file becomes the **snapshot**: the copy that survives a pod stop.

**Why:** "database is locked" storms and failed saves come from SQLite committing on MooseFS. After the move, no commit touches MooseFS.
- In the rehearsal, a 20-second total freeze of `/data` in the middle of a copy caused **zero** failed saves.
- With the database on `/data`, the same test writer managed about a third as many commits, and some waited 10 s.

**The trade you are accepting:** see "Recovery point" at the end. In one sentence: **if the pod stops without the final-sync step below, up to about 2½ minutes of chat is lost.**

Everything here was rehearsed on a copy of her real 505 MB database, inside the real v3.1.9.7 image, with a simulated slow and stalling `/data`, and re-verified by the round-1 hostile review (`/home/drew/zl-ops/reviews/review1-v3197-65ea196.md`) against the actual shipped code. The measurements and known defects are in the "Defect status" section below.

**How long it takes:**
- about 20 minutes of pre-flight, while she can keep chatting except where noted;
- about 5 minutes of downtime;
- about 15 minutes of checks after the boot.

**Where to run commands:** everything runs in the RunPod **Web Terminal**. Start every terminal session with these lines:

```bash
PY=/opt/compactor-venv/bin/python; APY=/app/venv/bin/python; R=/opt/zl-repo/scripts
L=/var/log/zla; [ -d $L ] || L=/data/logs
```

---

## A. Pre-flight (on the pod as it runs today: v3.1.9.x image, `WEBUI_DB_LOCAL=false`)

Today `/data/openwebui/webui.db` **is** the live database. That is why the commands in this section use that path. After the move they must not; see section D.

### A1. Get the operator scripts

```bash
rm -rf /opt/zl-repo && git clone -q --depth 1 --branch feature/v3.1.9.7 https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo && git -C /opt/zl-repo log --oneline -1
```

**Expect:** feature/v3.1.9.7's HEAD, which now has feature/v3.1.9.6 merged into it (H-5, review1-v3197-65ea196) — so this one clone carries both branches' scripts.

### A2. Save two small check scripts on `/data`, so they survive the redeploy

```bash
mkdir -p /data/scripts
cat > /data/scripts/dbcounts.py <<'EOF'
import json, sqlite3, sys
DB = sys.argv[1]; CID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=60)
q = lambda s, *a: c.execute(s, a).fetchone()[0]
h = json.loads(q("select chat from chat where id=?", CID))["history"]["messages"]
print("db              ", DB)
print("quick_check     ", q("pragma quick_check"))
print("alembic         ", q("select version_num from alembic_version"))
print("chats           ", q("select count(*) from chat"))
print("message rows    ", q("select count(*) from chat_message"))
print("her msgs table  ", q("select count(*) from chat_message where chat_id=?", CID))
print("her msgs json   ", len(h))
print("her last update ", q("select updated_at from chat where id=?", CID))
EOF
```

Also copy the `uibranch.py` check from `POD-FIXES-2026-09-23.md` §4 to `/data/scripts/uibranch.py`, instead of `/tmp`.

> **From now on, always pass the database path to both scripts explicitly.** `uibranch.py`'s built-in default is `/data/openwebui/webui.db`, which after the move is the snapshot, not the live file.

### A3. Close the stale backfill records

This is unrelated to the move, but a redeploy would otherwise resume them.

```bash
$PY $R/backfill-records.py --store /data/openwebui/compactor; echo EXIT=$?
```

If it reports pending work (`EXIT=3`), apply and dry-run again:

```bash
$PY $R/backfill-records.py --store /data/openwebui/compactor --apply; echo EXIT=$?
$PY $R/backfill-records.py --store /data/openwebui/compactor; echo EXIT=$?
```

**Expect:** `EXIT=0`, and then a dry run with nothing stale and `EXIT=0`.

### A4. Chat-tree repair (JSON sync only) and stale-spinner fix

Start with the dry runs. They are read-only, and OpenWebUI can stay up:

```bash
$APY $R/repair-chat-tree.py /data/openwebui/webui.db --sync-json-only; echo EXIT=$?
$APY $R/fix-stale-unfinished.py /data/openwebui/webui.db --all-branches; echo EXIT=$?
```

- **If both print `EXIT=0` ("nothing to do"):** skip to A5.
- **If either prints `EXIT=3`:** close her tabs on every device, then run:

```bash
supervisorctl stop openwebui backup && $APY $R/repair-chat-tree.py /data/openwebui/webui.db --sync-json-only --apply && $APY $R/fix-stale-unfinished.py /data/openwebui/webui.db --all-branches --apply; echo EXIT=$?; supervisorctl start openwebui backup
```

**Expect:** "written… integrity: ok", "written: N | verification: OK", and `EXIT=0`.

**Never** run plain `repair-chat-tree.py --apply`, i.e. without `--sync-json-only`.

### A5. No VACUUM yet

Compacting the database is worth doing, but **not here** — on `/data`, a VACUUM measured 116 s of journal writes on MooseFS, and a stall in that window leaves the very hot journal we are trying to escape. After the move, the same VACUUM takes about 4 s on local disk (step C7).

### A6. A verified backup

She should not chat for about 2 minutes. The backup briefly blocks saves.

```bash
$PY /opt/compactor/backup.py --once; echo "BACKUP EXIT=$?"
A=$(ls -t /data/backups/zions-backup-*.tar.gz | head -1); echo "$A"; $PY /opt/compactor/backup.py --verify "$A"; echo "VERIFY EXIT=$?"
```

**Expect:** `[OK] zions-backup-….tar.gz  db=ok, chroma=ok, …`, `BACKUP EXIT=0`, `[OK] db=ok…`, and `VERIFY EXIT=0`.
- **Do not continue without both zeros.** Also download this archive off the pod if you can, and check afterwards that it is not all zeros.

### A7. Record the baseline

```bash
$APY /data/scripts/uibranch.py /data/openwebui/webui.db | tee /data/forensics/dbmove-baseline.txt
$PY /data/scripts/dbcounts.py /data/openwebui/webui.db | tee -a /data/forensics/dbmove-baseline.txt
df -h / | tail -1
```

- Both UI-branch lines should say "reached the first message", with 0 messages missing.
- Check the container disk has **at least 3 GB free**: after the move it holds the database, a same-size temporary copy during each sync, plus tool backups.

### A8. Quiesce, right before the redeploy

1. Close her chat on **every** device.
2. Stop OpenWebUI and the backup:

```bash
supervisorctl stop openwebui backup
ls -la /data/openwebui/ | grep webui.db
$PY /data/scripts/dbcounts.py /data/openwebui/webui.db | tee /data/forensics/dbmove-final-before.txt
```

**Expect:** exactly one `webui.db` line, with **no** `webui.db-journal` and no `-wal`, and `quick_check ok`.
- If a `-journal` is there, **stop**: follow `RUNBOOK_DB_JOURNAL.md` first.
- Do **not** start OpenWebUI again on this pod. Go straight to B.

---

## B. Template changes, then redeploy

In the RunPod template or pod environment:

| Row | Set to |
|---|---|
| image | the **v3.1.9.7** tag, which includes OpenWebUI 0.11.4 and the scroll-fix CSS |
| `WEBUI_DB_LOCAL` | `true`, lowercase, no spaces, no quotes. It was `false`. |
| `WEBUI_DB_SYNC_INTERVAL_S` | `120` (a **new** row) |
| `COMPACTOR_BACKUP_INTERVAL_HOURS` | `1` (M-4, review1-v3197-65ea196) — a redeploy with this row missing or still `6` silently REVERTS POD-FIXES-2026-09-23 §5's hand-applied hourly backups back to the template's old default. If she is on hourly backups today, this row must be `1`, not left out. |
| `COMPACTOR_MAX_RETAINED_IMAGES` | leave as the pod's current value (M-4) — **not** a database-move setting, but this redeploy is a restart, the only time this value can change. Caveat if you ARE changing it: `0` means no images at all, including a NEW upload in the current turn, not merely older ones dropping out of history — it is not a "keep 0 old images" knob. `-1` = unlimited, `N` = keep N most recent. `disconnect-2026-09-23` recommends 1 or 0 for her pod. |
| `COMPACTOR_BACKFILL_MAX_ATTEMPTS` | **Not needed** if A3 ended with nothing stale. Delete the row if it is there. If A3 could not close all stale records, keep `=0` for this deploy — it only stops stale backfills from resuming, and it has nothing to do with the database move. |

**Also, while you are in the template (M-4): delete any stale
`COMPACTOR_INJECTION_BUDGET_FRACTION=.6` row** if one is still there from
before v3.1.9.5 — it is not part of this move, but this redeploy is a
natural point to clear it, and leaving it in place pins the fraction below
this image's `0.75` default. See RUNPOD_DEPLOY.md's legacy upgrade section
for how it could still be there.

**These rows must NOT exist.** Delete any you find:
- `DATABASE_URL`: entrypoint.sh now REFUSES to boot if it disagrees with
  `WEBUI_DB_LOCAL` in either direction (D12/M-2, review1-v3197-65ea196) —
  it would otherwise silently keep OpenWebUI on `/data` while the sync
  program believes the database is local, or vice versa.
- `WEBUI_LOCAL_DB` and `WEBUI_SNAPSHOT_DB`.
- any `WEBUI_DB_ALLOW_…`, including `WEBUI_DB_ALLOW_EMPTY_START`.

Leave everything else as it is. Then redeploy.

**Boot time:** the first boot takes about 2 minutes longer than usual. It copies 505 MB off `/data` and checks it before OpenWebUI starts.

---

## C. After the boot: verification (in order; stop at the first mismatch)

Recreate the two `PY`/`APY`/`R`/`L` lines from the top first, and re-clone `/opt/zl-repo` (A1) if you will need the tools.

### C1. RunPod Logs tab

**Expect** these lines:

```
      WEBUI_DB_LOCAL=true (explicitly set (true))
[2a/3] Checking for an interrupted restore...
      no interrupted restore found
[2b/3] Placing webui.db on local disk (/var/lib/openwebui/webui.db)
… compactor.webuidb INFO restored snapshot -> local (504.9 MB, 52 chats)
{'action': 'restored_from_snapshot', 'local': '/var/lib/openwebui/webui.db', 'snapshot': '/data/openwebui/webui.db'}
```

If instead you see `RESTORE FAILED (exit N) - REFUSING TO START`, or `AN INTERRUPTED RESTORE WAS LEFT IN PROGRESS`, go to E3. Nothing has been changed in that case.

### C2. The gate and the programs

```bash
tr '\0' '\n' < /proc/1/environ | grep -E '^(WEBUI_DB_LOCAL|WEBUIDB_SYNC_ENABLED|DATABASE_URL|WEBUI_DB_SYNC_INTERVAL_S)='; supervisorctl status webuidb-sync openwebui backup
```

**Expect** the following. The order may differ.

```
WEBUI_DB_LOCAL=true
WEBUIDB_SYNC_ENABLED=true
DATABASE_URL=sqlite:////var/lib/openwebui/webui.db
WEBUI_DB_SYNC_INTERVAL_S=120
backup                           RUNNING   …
openwebui                        RUNNING   …
webuidb-sync                     RUNNING   …
```

### C3. The file OpenWebUI really has open

```bash
P=$(pgrep -f "open-webui serve" | head -1); tr '\0' '\n' < /proc/$P/environ | grep ^DATABASE_URL; ls -l /proc/$P/fd | awk '{print $NF}' | grep 'webui\.db' | sort -u
```

**Expect:** `DATABASE_URL=sqlite:////var/lib/openwebui/webui.db` and `/var/lib/openwebui/webui.db`.
- `/data/openwebui/webui.db` must **not** appear.
- `/data/openwebui/vector_db/chroma.sqlite3` may appear — that is OpenWebUI's own vector store, which stays on `/data` (D8, not fixed this release).

### C4. Both copies, and the baseline

```bash
$PY /opt/compactor/webuidb.py --status
$PY /data/scripts/dbcounts.py /var/lib/openwebui/webui.db
$APY /data/scripts/uibranch.py /var/lib/openwebui/webui.db
cat /data/forensics/dbmove-final-before.txt
```

**Expect:**
- two `--status` rows (`local` and `snapshot`), both `quick_check=ok`, with the same `chats=` and `content=`;
- `dbcounts` identical to `dbmove-final-before.txt`, apart from the `db` line;
- `uibranch` "reached the first message" twice, with 0 missing.

### C5. OpenWebUI version, and the migrations on the moved database

```bash
$APY -m pip show open-webui | grep ^Version
grep -hE "alembic|Error running migrations" $L/openwebui*.log | tail -3
```

**Expect:** `Version: 0.11.4`, then `Context impl SQLiteImpl.` and `Will assume non-transactional DDL.`. There must be **no** "Error running migrations": the database is already at 0.11.4's head, `d4c1a8e37b62`.

### C6. Disk space on the pod's own disk

```bash
df -h /var/lib/openwebui | tail -1
```

**Expect:** at least 3 GB available. This release also reserves 2× the live database's size plus a margin for the BACKUP daemon's own local staging (M-1, review1-v3197-65ea196) — room for its own copy AND a possible concurrent `webuidb-sync` cycle at once.

### C7. One-time compaction, 505 MB to about 350 MB

This is recommended. It takes about 1 minute of downtime; close her tabs first. Stop `backup` too — a scheduled backup landing mid-VACUUM would see a lock (per review1-v3197-65ea196's deploy-doc walkthrough, point 10):

```bash
supervisorctl stop openwebui webuidb-sync backup
$PY -c "import sqlite3; c=sqlite3.connect('/var/lib/openwebui/webui.db'); c.execute('VACUUM'); print('quick_check:', c.execute('pragma quick_check').fetchone()[0]); c.close()"
ls -la /var/lib/openwebui/webui.db
WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force; echo "SYNC EXIT=$?"
supervisorctl start webuidb-sync openwebui backup
```

**Expect:**
- `quick_check: ok`;
- a size of about 350,000,000;
- `{'synced': True, 'skipped': None, 'error': None, 'bytes': 35…}`;
- `SYNC EXIT=0`.

### C8. The first automatic sync

Let her chat, then wait about 3 minutes:

```bash
grep -E "published|publish failed|REFUSING|has failed" $L/webuidb-sync-error.log | tail -3
```

**Expect:** `… INFO published local -> snapshot (350.8 MB, 52 chats)`.
- Note the **`-error.log`**: that is where this program writes everything, including its normal INFO lines (D13, not fixed this release). The plain `webuidb-sync.log` stays empty.

### C9. The health endpoint

```bash
curl -s http://127.0.0.1:8080/health/full | $PY -c 'import json,sys; c=json.load(sys.stdin)["checks"]; print(json.dumps(c["snapshot"])); print(json.dumps(c["sqlite_journal"]["checked"]))'
```

**Expect:** `"watched": true, … "stale": false`, and both journal entries `"hot": false`.

**D11/H-2 is STILL a known false alarm (NOT fixed this release, review1-v3197-65ea196):** `"stale": true` with a huge `local_lag_s` right after her FIRST message post-boot is a KNOWN FALSE ALARM (the snapshot's mtime, copied from whatever she last wrote before the pod stopped, can be many hours old). A fix was attempted — a check-in sidecar (`webuidb.py`'s `SYNCED_AT_SIDECAR`) that `/health/full` reports as a new `checkin_age_s` field — but making it decide `stale` broke a real detector (`test_health_findings.py`'s F4 suite: a quiet daemon that has genuinely stopped publishing reads identically to a fresh boot's false alarm from `checkin_age_s` alone). So `stale` is still decided by `local_lag_s`/`age_s` exactly as before; `checkin_age_s` is exposed as a diagnostic only. **If `stale: true` appears right after her first post-restore message with a huge `local_lag_s`, check whether a `published` line has appeared in `$L/webuidb-sync-error.log` since — if the daemon is actually running normally, this clears on the next cycle and is the known false alarm, not real data loss.**

### C10. The first backup of the moved database

Take it while she is not chatting; it takes about 1 to 2 minutes.

```bash
$PY /opt/compactor/backup.py --once; echo "BACKUP EXIT=$?"
A=$(ls -t /data/backups/zions-backup-*.tar.gz | head -1); $PY /opt/compactor/backup.py --verify "$A"; echo "VERIFY EXIT=$?"
```

**Expect:** `[OK]…`, `BACKUP EXIT=0`, and `VERIFY EXIT=0`.
- `backup.py` reads the **live** local file automatically, stages its own copy locally first (D2), and now reserves double that copy's size plus a margin before staging, logging at ERROR (not WARNING) if it has to fall back to the old direct-to-/data behaviour (M-1).

She can use it normally now. Keep `/data/forensics/dbmove-*.txt`.

---

## D. Daily operation

### Where things are now

| | Path |
|---|---|
| Live database (what she writes to) | `/var/lib/openwebui/webui.db`, on the pod's disk, **gone when the pod stops** |
| Snapshot (what survives) | `/data/openwebui/webui.db`, refreshed every `WEBUI_DB_SYNC_INTERVAL_S`, but only when something changed |
| Sync log | `$L/webuidb-sync-error.log` |
| Backups | `/data/backups/`, unchanged |

### The one rule for every tool

**Point tools at `/var/lib/openwebui/webui.db`. Never at `/data/openwebui/webui.db`.**
- `scripts/_webui_live_path.py` (D3, `feature/v3.1.9.6`, merged into this release) now makes `repair-chat-tree.py`, `fix-stale-unfinished.py`, `fix-encoded-messages.py`, `clean-decoration.py` and `recover-webui-db.py` REFUSE a write aimed at the snapshot while local mode is active, and WARN on a read-only dry run pointed there — see CHANGELOG.md's `[3.1.9.6]`/D3 entry for exactly what each script does now. This closes the "the repair succeeded on the snapshot, the next cycle silently overwrote it, and nothing warned" failure the rehearsal found. Still point every command at the live path explicitly; do not rely on the refusal as your only guard.

### Read-only checks (safe any time)

```bash
$APY /data/scripts/uibranch.py /var/lib/openwebui/webui.db
$PY /data/scripts/dbcounts.py /var/lib/openwebui/webui.db
$APY $R/repair-chat-tree.py /var/lib/openwebui/webui.db --sync-json-only; echo EXIT=$?
$APY $R/fix-stale-unfinished.py /var/lib/openwebui/webui.db --all-branches; echo EXIT=$?
```

### Any tool that writes

The tools covered here are `repair-chat-tree --sync-json-only --apply`, `fix-stale-unfinished --apply` and `fix-encoded-messages --apply`.

Close her tabs, then:

```bash
$PY /opt/compactor/backup.py --once; echo "BACKUP EXIT=$?"
supervisorctl stop openwebui webuidb-sync
$APY $R/repair-chat-tree.py /var/lib/openwebui/webui.db --sync-json-only --apply; echo EXIT=$?
$APY $R/fix-stale-unfinished.py /var/lib/openwebui/webui.db --all-branches --apply; echo EXIT=$?
WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force; echo "SYNC EXIT=$?"
supervisorctl start webuidb-sync openwebui
```

**Expect:** each tool prints `EXIT=0`, then `'synced': True` and `SYNC EXIT=0`.
- The backup comes first because `repair-chat-tree`'s own undo copy (`webui.db.bak-<stamp>`) now sits on the pod's disk and disappears if the pod stops.
- Stop `webuidb-sync` too, not just OpenWebUI. Otherwise the tool can refuse because the sync program has the file open.

For `fix-encoded-messages.py`, use the same wrapper, with `/var/lib/openwebui/webui.db` as its `<live>` argument.

**Undoing a tool** (`--restore <stamp>`), with the same wrapper:
- Restore onto `/var/lib/openwebui/webui.db`.
- Then publish with:

```bash
WEBUI_DB_ALLOW_OLDER_GENERATION=1 WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force; echo "SYNC EXIT=$?"
```

- A plain `--sync-once` after an undo prints `skipped: unchanged since last sync` with exit 0, and **publishes nothing**.
- Plain `--force` is refused, "OLDER than the same conversation", because the undo is older on purpose.

### The other tools

| Tool | What to do |
|---|---|
| `clean-decoration.py` | Still **do not run it** (H-1 in the round-3 v3.1.9.6 review is separate from this release's H-numbering and remains unfixed). If it is ever fixed, its `--webui-db` default now resolves the live path the same way the image does (D3) rather than hardcoding the snapshot — but pass `/var/lib/openwebui/webui.db` explicitly regardless. |
| `import-history.py` | It needs an **export**, not the live file. Make the snapshot current with `WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force`, then pass `--webui-db /data/openwebui/webui.db`. The snapshot is a clean, writer-free copy. |
| `backfill-records.py` | Unchanged: `--store /data/openwebui/compactor` |
| `backup.py --once / --verify / --list` | Unchanged. It finds the live file itself. Run `--once` when she is not chatting. |

### Checking that syncs are happening

```bash
grep -E "published|publish failed|REFUSING|has failed" $L/webuidb-sync-error.log | tail -5
$PY /opt/compactor/webuidb.py --status
```

**Healthy:**
- there is a `published` line within about `WEBUI_DB_SYNC_INTERVAL_S` of her last message;
- both `--status` rows show the same `content=`;
- `/health/full` `checks.snapshot.stale` is `false` (C9).

**Normal:** no new lines at all while she isn't chatting. Unchanged cycles are silent.

**Not normal:** any `publish failed` or `REFUSING` line, or `has failed N times in a row`. Go to E1.

### BEFORE ANY stop, restart, redeploy or template edit: the final sync

**This is the only reliable way a stop loses nothing.** The pod does NOT
do this for you automatically in every case:

- `supervisorctl stop openwebui webuidb-sync` (or `stop all`) DOES now run
  an automatic final publish on SIGTERM (D1/H-1, this release) — measured
  losing nothing when the publish completes inside supervisord's
  `stopwaitsecs=120`.
- **A RunPod stop, or `docker stop` with a short grace period, does
  NOT.** `docker stop -t 10` (RunPod's own default grace period) SIGKILLs
  the process well before a 13–23s warm publish (61–71s cold) can finish
  — measured losing the last write outright (review1-v3197-65ea196, H-1).
  D1 only ever covers `supervisorctl stop`, never a container-level stop
  or an unplanned one.

So: run this by hand before ANY stop, restart, redeploy, or template edit,
regardless of which stop mechanism will be used.

Close her tabs, then:

```bash
supervisorctl stop openwebui webuidb-sync
WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force; echo "SYNC EXIT=$?"
$PY /opt/compactor/webuidb.py --status
```

**Expect:** `'synced': True`, `SYNC EXIT=0`, and two `--status` rows with the **same** `content=` and `newest_update=`. Only then stop or redeploy.
- **If `SYNC EXIT=1`, do not stop the pod.** Go to E1.
- The command is `--sync-once --force`. **Never** type `--sync-now` or `--help`: `webuidb.py` doesn't know those flags, and it silently starts a second sync loop instead (Ctrl+C it). Tracked as D6 (argparse) — fixed for `--help` (exits 0 with usage) and unknown flags (exit 2) as of this release, so this specific trap should no longer fire; keep typing the command above exactly regardless.

---

## E. Incidents

### E1. The sync log says `publish failed (… REFUSING to publish: …)`

**Nothing is lost yet.** Her live database is fine; only the durable copy has stopped moving.
1. **Don't stop or redeploy the pod** while a refusal stands.
2. Take a backup:

```bash
$PY /opt/compactor/backup.py --once; echo "BACKUP EXIT=$?"
```

3. Read the reason in the line, and act on it:

| The line says | Meaning | Do |
|---|---|---|
| `… are OLDER than the same conversation in the snapshot` | The snapshot has something newer than the live file. | If you restored or undid something **on purpose**: `WEBUI_DB_ALLOW_OLDER_GENERATION=1 WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force`. If **not**, something wrote to `/data/openwebui/webui.db`, most likely a tool run on the old path. Run `cp /data/openwebui/webui.db /data/forensics/snapshot-newer-$(date -u +%Y%m%dT%H%M%SZ).db`, compare `dbcounts.py`/`uibranch.py` on both paths, and ask for help before overriding. |
| `… lost more than 1.0 MB of stored content each` | More than 1 MB of one conversation disappeared. | If she really deleted a long stretch: `WEBUI_DB_ALLOW_ROW_LOSS=1 WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force`. If not, stop and ask. |
| `… (chat count tripped it)` or `(stored content tripped it)` | Half the conversations or half the content vanished. A copy of the live file was saved to `/data/forensics`. | Do **not** override. Ask. |
| `… failed quick_check` or `… could not be confirmed` (about `/data/openwebui/webui.db`) | The snapshot on `/data` is damaged. | Move it aside: `mv /data/openwebui/webui.db /data/forensics/webui.db.bad-$(date -u +%Y%m%dT%H%M%SZ)`. Move any `-journal` beside it the same way. Then run `WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force`, which writes a fresh snapshot. |
| `Read-only file system`, `Input/output error`, `cannot stat` | `/data` itself is misbehaving. | Nothing to do on the database. It retries every cycle. Keep the pod running until a `published` line appears. |

**Never** put any `WEBUI_DB_ALLOW_…` row on the template. They are for one command, in one shell.

### E2. The pod stopped unexpectedly, or was stopped without the final sync

- **What is gone:** whatever she wrote after the last `published` line. With a 120s interval, that is usually under about 2½ minutes (interval plus one publish). Find the time with:

```bash
grep published $L/webuidb-sync-error.log | tail -1
```

- **On the next boot:** run C1 to C4. `restored_from_snapshot` means the boot is fine.
  - A leftover `/data/openwebui/webui.db.sync-tmp` after a crash is normal. The first sync deletes it.
- **If her last messages are missing, that is the recovery point**, not damage. Tell her roughly when to pick up from.
- **Don't restore a backup archive to get them back.** Archives are older than the snapshot, and would lose more.

### E3. The boot refuses

In every one of these cases, nothing on `/data` was touched.

- **`RESTORE FAILED (exit 3)`:** the snapshot failed its check. Typically it is a hot `-journal` that could not be rolled back on MooseFS. Follow `RUNBOOK_DB_JOURNAL.md` for `/data/openwebui/webui.db`, then redeploy. **Never** set `WEBUI_DB_ALLOW_EMPTY_START`.
- **`RESTORE FAILED (exit 4)`:** the pod's disk is full. Give the pod more container disk, then redeploy.
- **`exit 5`, or `AN INTERRUPTED RESTORE WAS LEFT IN PROGRESS`:** follow the banner's own instructions, and ask for help.

---

## F. Rollback: back to the database on `/data`

This was rehearsed with zero loss: every chat and every message was identical before and after.

1. Close her tabs. Run the **final sync** (section D) and check that the two `--status` rows match.
2. Record the state:

```bash
$PY /data/scripts/dbcounts.py /var/lib/openwebui/webui.db; $PY /data/scripts/dbcounts.py /data/openwebui/webui.db
```

   The two blocks must be identical apart from the `db` line.
3. In the template, set `WEBUI_DB_LOCAL=false` (exact, lowercase). The `WEBUI_DB_SYNC_INTERVAL_S` row can stay or go. Redeploy.
4. Verify with the check from RUNPOD_DEPLOY.md:

```bash
tr '\0' '\n' < /proc/1/environ | grep -E '^(WEBUI_DB_LOCAL|WEBUIDB_SYNC_ENABLED|DATABASE_URL)='; supervisorctl status webuidb-sync
$PY /data/scripts/dbcounts.py /data/openwebui/webui.db
```

   **Expect:** `WEBUI_DB_LOCAL=false`, `WEBUIDB_SYNC_ENABLED=false`, `DATABASE_URL=sqlite:////data/openwebui/webui.db`, `webuidb-sync STOPPED Not started`, and counts identical to step 2.

**Rolling back only the image** (for example to `:v3.1.9.5-cu12` — there
is no `:v3.1.9.6-cu12`, see README.md "Image tags") needs a database
rollback ONLY if you are also reverting the placement (step 3 above); the
local-disk placement mechanism itself is v3.1.9.7-only, so an older image
has no `WEBUI_DB_LOCAL` support and simply ignores the row:

- Run the final sync first, as for any redeploy (step 1 above) —
  regardless of which image you are rolling back to.
- **H-4 (review1-v3197-65ea196): OpenWebUI 0.11.0 on this migrated
  database is NOT harmless — it is a degraded state to leave quickly.**
  Every boot logs a loud but non-fatal `Can't locate revision identified
  by 'd4c1a8e37b62'` traceback (expected). But measured directly: a
  full-chat save on 0.11.0 against the migrated schema **timed out at
  600s** in one trial with `database is locked` errors from the
  automation scheduler, and when a save DID land, the JSON `history` copy
  and the `chat_message` table diverged by one save — the exact class of
  bug that broke her UI on 2026-09-23, and `repair-chat-tree
  --sync-json-only` cannot repair it in that direction (table → JSON
  only). Get back onto 0.11.4 as soon as the reason for the rollback is
  resolved; do not leave her chatting on 0.11.0 against this schema
  longer than necessary.
- **Never use `backup.py --restore` as part of this rollback** — see
  RUNPOD_DEPLOY.md's "Rollback" section (B-1) for exactly why it produces
  a mixed-generation chat/memory split on this release, and
  `/home/drew/zl-ops/RESTORE-2026-09-23.md` ("Never") for the same rule
  stated for the pod's manual restore procedure generally.

---

## Recovery point: what she is agreeing to, in plain words

- **Planned stops lose nothing, IF the final sync ran first.** A redeploy, template edit or restart preceded by the final-sync step in section D is safe. This was rehearsed: every chat and every message was identical afterwards.
- **`supervisorctl stop`/`restart` alone (no manual final sync) now also loses nothing** (D1/H-1, this release) — SIGTERM triggers an automatic final publish that completes within supervisord's 120s `stopwaitsecs`. Still run the manual final sync anyway: it is the only thing that is ALSO safe against a stop mechanism the daemon's SIGTERM handler cannot save in time (below).
- **A RunPod stop, `docker stop -t 10`, or any stop with a grace period shorter than a warm publish (13–23s, 61–71s cold) loses the last unpublished writes.** Measured directly: `docker stop -t 10` SIGKILLs before the automatic final sync can finish. This is never more than one sync interval (`WEBUI_DB_SYNC_INTERVAL_S`, 120s recommended) plus one publish's time, UNLESS `/data` was refusing writes, in which case the window runs from the last `published` line in the sync log (E2).
- **A crash cannot damage the saved copy.** Even a crash in the middle of copying leaves the previous copy on `/data` whole. This was tested by killing the pod mid-copy, and by a genuine hot local journal.
- **What she gets in exchange:** saves stop failing when the network volume stalls. In the rehearsal, a 20-second freeze of `/data` caused zero failed saves.

---

## Defect status (as shipped in v3.1.9.7, review1-v3197-65ea196)

| D | Status |
|---|---|
| D1 final sync / stop order | PARTIAL — fixed for `supervisorctl stop`; a RunPod/`docker stop -t 10` still loses the last interval. Ignoring a second SIGTERM and not force-republishing unchanged content (this release) narrow, but do not close, the re-entrancy window. |
| D2 backup lock over /data | FIXED on normal disk; on low local disk it still falls back to the old direct-to-/data behaviour, now logged at ERROR and requiring 2× the database's size plus margin before it triggers (M-1, this release). |
| D3 tools edit the snapshot | FIXED (`feature/v3.1.9.6`, merged into this release). |
| D4 mtime-preserving restore | PARTIAL — the CLI's `--sync-once` always forces; the daemon's periodic cycle still uses the mtime skip deliberately. |
| D5 boot restore writes /data | FIXED for `restore_on_boot()` against a genuine hot journal. |
| D6 argparse | FIXED — `--help` exits 0 with usage, unknown flags exit 2. |
| D7 whole-file publish | NOT FIXED (deferred). |
| D8 chroma stays on /data | NOT FIXED (deferred). |
| D9 older local kept on restart | NOT FIXED (not made worse). |
| D10 tool backups on local disk | FIXED (`feature/v3.1.9.6`'s D3 work redirects them to `/data/forensics`). |
| D11 false stale after boot | NOT FIXED (H-2, this release — attempted, reverted). A check-in sidecar (`SYNCED_AT_SIDECAR`) is reported as a new `checkin_age_s` diagnostic field, but does not decide `stale` — doing so broke a real detector for genuine unpublished activity (see the C9 note above). The false alarm on her first write after a restore is still present. |
| D12 stray DATABASE_URL | FIXED, both directions (M-2, this release adds the `WEBUI_DB_LOCAL=false` mirror). |
| D13 log file name | NOT FIXED — `webuidb-sync-error.log` carries the normal INFO lines too. |
| D14 first publish one interval late | FIXED. |
