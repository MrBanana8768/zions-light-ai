# Her chat database on the NEW pod (v3.1.9.10): daily operation, incidents, recovery

**Scope.** This is the runbook for the NEW pod only: `angreg/zions-light-ai:v3.1.9.10-cu12`, `WEBUI_DB_LOCAL=true`,
all data under `/data/v10`. It replaces `docs/DB-MOVE-RUNBOOK.md` for that pod. **Do not use `DB-MOVE-RUNBOOK.md`
on the new pod**: it is written for the in-place move on the OLD pod layout and its paths (`/data/openwebui`,
`/data/logs`, `/data/backups`, `/data/forensics`) are the OLD pod's live tree and its rollback. Getting there
(seed, cutover, rollback) is `RUNBOOK_POD_MIGRATION.md`.

Start EVERY Web Terminal session on the new pod with these lines, and use the variables everywhere below
(nothing in this file types a path by hand):

```bash
PY=/opt/compactor-venv/bin/python; APY=/app/venv/bin/python; R=/opt/zl-repo/scripts
V=/data/v10; L=$V/logs; NEW=$V/openwebui; SNAP=$NEW/webui.db; LIVE=/var/lib/openwebui/webui.db
STORE=$NEW/compactor; BK=$V/backups; F=$V/forensics
```

**Before the first command of a session, check that this pod really is the v10 one** (a wrong shell is the one way
to aim a writer at the old pod's files):

```bash
tr '\0' '\n' < /proc/1/environ | grep -E '^(DATA_DIR|LOG_DIR|COMPACTOR_BACKUP_DIR|WEBUI_SNAPSHOT_DB|WEBUI_DB_LOCAL)='
```

**Expect** `/data/v10/openwebui`, `/data/v10/logs`, `/data/v10/backups`, `/data/v10/openwebui/webui.db` and
`WEBUI_DB_LOCAL=true`. Anything else: stop, you are not on the new pod (or its rows are wrong).

---

## 1. Where things are

| | Path | Survives a pod stop or loss? |
|---|---|---|
| Live database (what she writes to) | `$LIVE` = `/var/lib/openwebui/webui.db`, the pod's own disk | **NO** |
| Snapshot (the durable copy) | `$SNAP` = `/data/v10/openwebui/webui.db`, refreshed every `WEBUI_DB_SYNC_INTERVAL_S` (120 s), only when something changed | yes |
| Sync log | `$L/webuidb-sync-error.log` (it carries the normal INFO lines too) | yes |
| Her memory store | `$STORE` | yes |
| Chat journal (one fsynced line per completed reply) | `$STORE/journal/<conv>.jsonl` | yes |
| Uploads and the uploads mirror | `$NEW/uploads`, `$BK/uploads` | yes |
| Archives | `$BK/zions-backup-*.tar.gz` (3 hourly, 7 daily, 8 weekly: about 17 kept, about 4 GB) | yes |
| Test archives from before the cutover | `$BK/pre-cutover-test/` (outside the prune glob) | yes |
| Forensic copies, quarantines | `$F` | yes |

Two facts to hold on to: a pod stop loses the live file; and the network volume is the only thing that survives
(`~/pod-exports` on the laptop, from `zlpod pull`, is the only copy outside it).

## 2. The one rule for every tool

**Point tools at `$LIVE`. Never at `/data/openwebui/webui.db`** (that is the OLD pod's live file, and the rollback).
The repair scripts refuse a write aimed at the snapshot while local mode is on, and warn on a read pointed there.
Still pass the path explicitly.

## 3. Read-only checks (safe any time)

```bash
$APY /data/scripts/uibranch.py $LIVE
$PY /data/scripts/dbcounts.py $LIVE
$APY $R/repair-chat-tree.py $LIVE --sync-json-only; echo EXIT=$?
$APY $R/fix-stale-unfinished.py $LIVE --all-branches; echo EXIT=$?
```

(`/data/scripts/uibranch.py` and `dbcounts.py` are the small check scripts on the shared volume. They take the
database path as an argument and only read.)

Are syncs happening?

```bash
grep -E "published|publish failed|REFUSING|has failed" $L/webuidb-sync-error.log | tail -5
$PY /opt/compactor/webuidb.py --status
```

**Healthy:** a `published` line within about 120 s of her last message, and both `--status` rows showing the same
`chats=`, `content=` and `newest_update=`. **Normal:** no new lines while she is not chatting. **Not normal:** any
`publish failed`, `REFUSING`, or `has failed N times in a row`: section 6.

## 4. BEFORE ANY planned stop, restart, redeploy, template edit or reboot: the final sync

**This is the only reliable way a stop loses nothing.** `supervisorctl stop` also publishes on SIGTERM, but only
after `backup`, the tunnel and OpenWebUI have each stopped in turn (up to 10 s each), and RunPod's stop grace period is
not documented anywhere in this repo. A RunPod stop without this step loses up to about 2.5 minutes of chat.

She closes the app on every device. Then, in this order:

```bash
supervisorctl stop openwebui webuidb-sync backup     # backup too: a running backup holds the sync lock
WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force; echo "SYNC EXIT=$?"
$PY /opt/compactor/webuidb.py --status
S=$($PY /opt/compactor/webuidb.py --status | grep -E '^(local|snapshot) ' | sed -E 's/.*chats=/chats=/' | sort -u | wc -l)
[ "$S" = 1 ] && echo "STATUS ROWS MATCH" || echo "STATUS ROWS DIFFER: DO NOT STOP THE POD"
```

**Go ahead only if ALL THREE hold** (exit 0 alone is not enough):

1. `SYNC EXIT=0`. This build of `--sync-once --force` **waits for the sync lock** and exits non-zero unless it really
   synced, or verified the snapshot already identical. A run that prints "another sync is in progress" and exits 0 was the
   old (`v3.1.9.9`) behaviour: if you ever see that, it published nothing.
2. The printed dict says `'synced': True`, or says plainly that the snapshot was verified unchanged.
3. The two `--status` rows (`local`, `snapshot`) show the same `chats=`, `content=` **and** `newest_update=`
   (the one-liner prints `STATUS ROWS MATCH`).

If any of the three fails, **do not stop the pod**: section 6, E1 (with the `/data/v10` paths, below).
The command is exactly `--sync-once --force`. Never type `--sync-now` or `--help`.

To carry on running after a check (no stop planned): `supervisorctl start openwebui webuidb-sync backup`.

## 5. Any tool that writes

Covers `repair-chat-tree --sync-json-only --apply`, `fix-stale-unfinished --apply`, `fix-encoded-messages --apply`.
Close her tabs, then:

```bash
$PY /opt/compactor/backup.py --once; echo "BACKUP EXIT=$?"
supervisorctl stop openwebui webuidb-sync backup
$APY $R/repair-chat-tree.py $LIVE --sync-json-only --apply; echo EXIT=$?
$APY $R/fix-stale-unfinished.py $LIVE --all-branches --apply; echo EXIT=$?
WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force; echo "SYNC EXIT=$?"
$PY /opt/compactor/webuidb.py --status
supervisorctl start webuidb-sync openwebui backup
```

**Expect** `EXIT=0` for each tool, `SYNC EXIT=0` with `'synced': True`, and matching `--status` rows (section 4's
three conditions). The backup comes first because a repair tool's own undo copy sits on the pod's disk.

**Undoing a tool** (`--restore <stamp>`): restore onto `$LIVE`, then publish with the one-command override:

```bash
WEBUI_DB_ALLOW_OLDER_GENERATION=1 WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force; echo "SYNC EXIT=$?"
```

A plain `--sync-once` after an undo publishes nothing (unchanged); plain `--force` is refused as "OLDER" because the
undo is older on purpose.

| Tool | What to do |
|---|---|
| `clean-decoration.py` | Still **do not run it.** |
| `import-history.py` | Needs an export, not the live file. Make the snapshot current (section 4's sync line), then `--webui-db $SNAP`. The snapshot is a clean, writer-free copy. |
| `backfill-records.py` | `--store $STORE`; stop the compactor first (`--apply` refuses while it answers `/health`). |
| `backup.py --once / --verify / --list` | Finds the live file and the v10 backup dir itself. Run `--once` when she is not chatting. |
| `chat-journal.py` | Read-only (section 7.4). |

## 6. Incidents

### E1. The sync log says `publish failed (... REFUSING to publish: ...)`

**Nothing is lost yet.** Her live database is fine; only the durable copy has stopped moving.

1. **Do not stop or redeploy the pod** while a refusal stands.
2. `$PY /opt/compactor/backup.py --once; echo "BACKUP EXIT=$?"`
3. Read the reason in the line:

| The line says | Meaning | Do |
|---|---|---|
| `... are OLDER than the same conversation in the snapshot` | the snapshot has something newer than the live file | If you restored or undid something **on purpose**: `WEBUI_DB_ALLOW_OLDER_GENERATION=1 WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force`. If **not**: something wrote to `$SNAP` (a tool run on the wrong path). `mkdir -p $F && cp $SNAP $F/snapshot-newer-$(date -u +%Y%m%dT%H%M%SZ).db`, compare `dbcounts.py` on `$LIVE` and `$SNAP`, and ask before overriding. |
| `... lost more than 1.0 MB of stored content each` | more than 1 MB of one conversation disappeared | If she really deleted a long stretch: `WEBUI_DB_ALLOW_ROW_LOSS=1 WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force`. If not, stop and ask. |
| `... (chat count tripped it)` or `(stored content tripped it)` | half the conversations or half the content vanished; a copy went to `$F` | Do **not** override. Ask. |
| `... failed quick_check` or `... could not be confirmed` (about `$SNAP`) | the snapshot on the volume is damaged | Move it aside, **by variable, after the guard below**: see "Moving a damaged snapshot aside". Then run `WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force`, which writes a fresh snapshot. |
| `Read-only file system`, `Input/output error`, `cannot stat` | `/data` itself is misbehaving | Nothing to do on the database. It retries every cycle. Keep the pod running until a `published` line appears. |

**Never** put any `WEBUI_DB_ALLOW_...` row on the template. They are for one command, in one shell.

**Moving a damaged snapshot aside** (the only `mv` of a database in this file). It moves `$SNAP`, and only
after checking that `$SNAP` is under `/data/v10` and that local mode is on; it can never move the old pod's live DB:

```bash
case "$SNAP" in /data/v10/*) ;; *) echo "REFUSED: SNAP is not under /data/v10"; false;; esac \
 && tr '\0' '\n' < /proc/1/environ | grep -qx 'WEBUI_DB_LOCAL=true' \
 && [ "$(readlink -f "$SNAP")" != "$(readlink -f /data/openwebui/webui.db)" ] \
 && { mkdir -p $F && mv "$SNAP" $F/webui.db.bad-$(date -u +%Y%m%dT%H%M%SZ) && for s in -journal -wal -shm; do [ -e "$SNAP$s" ] && mv "$SNAP$s" $F/; done; echo MOVED; }
```

`MOVED` (and no `REFUSED`) means it worked. Nothing is ever deleted.

### E2. The pod was lost or stopped without the final sync: what survives, and how to recover

**What survives a pod loss**

| Where | What | Survives stop, terminate, host failure? |
|---|---|---|
| `$LIVE` (container disk) | her live chat database | **NO** |
| `$SNAP` | the snapshot, published every ~120 s when something changed | yes |
| `$STORE/journal/<conv>.jsonl` | one fsynced line per completed reply (her text and the reply) | yes |
| `$STORE` | her memory store (it may be slightly AHEAD of the snapshot) | yes |
| `$NEW/uploads`, `$BK/uploads` | uploaded images and their mirror | yes |
| `$BK/zions-backup-*.tar.gz` | the archives | yes |
| `$L` | every log | yes |
| `~/pod-exports` on the laptop | whatever `zlpod pull` last fetched | the ONLY copy outside the network volume |

**How much can be lost**

- A planned stop after a VERIFIED final sync (section 4): **nothing.**
- An unplanned loss (host failure, a RunPod stop without the final sync, a crash): whatever she wrote after the last
  `published` line. Normally at most about **2.5 minutes** (the 120 s interval plus one publish of 13 to 23 s, longer
  while the volume stalls).
- A loss while the sync log shows `publish failed` or `REFUSING`: everything since the last `published` line,
  however long ago. That is why a standing refusal is an incident (E1).
- **The network volume itself lost:** everything since the last `zlpod pull`. The snapshot, archives, journal and
  uploads all sit on the same volume.
- In every case the **replies** she received in the lost window are still in the chat journal (text readable). No tool
  writes them back into OpenWebUI yet.

**The steps**

1. **Start the pod again** (the same pod, or a new one with the SAME template rows and the same network volume). Do not
   change `DATA_DIR`, `WEBUI_SNAPSHOT_DB` or `COMPACTOR_BACKUP_DIR`. Do not add the tunnel token to any second pod
   while this one exists.
2. **RunPod Logs tab.** Expect `restored snapshot -> local (... MB, N chats)` and `restored_from_snapshot`.
   - `RESTORE FAILED (exit 3)` or `snapshot_unhealthy`: "The snapshot is damaged" below.
   - `exit 4`: the container disk is full. Give the pod more container disk and start it again.
   - `exit 5` or `AN INTERRUPTED RESTORE WAS LEFT IN PROGRESS`: stop and ask. Nothing has been changed.
   - `no local database and no snapshot`: **stop at once** (`supervisorctl stop openwebui cloudflared`). The pod is not
     seeing `/data/v10` (a wrong mount or a missing row). Her data is not lost; this boot simply cannot see it, and an
     empty OpenWebUI with the tunnel up has open sign-up.
3. **Find the recovery point:** `grep -E "published|publish failed|REFUSING" $L/webuidb-sync-error.log | tail -3`. The
   last `published` line is the moment her chat is restored to.
4. **See what was lost**, with the journal (read-only; work on a copy):

   ```bash
   cp $LIVE /tmp/webui-check.db
   $PY $R/chat-journal.py compare --store $STORE --db /tmp/webui-check.db
   $PY $R/chat-journal.py compare --store $STORE --db /tmp/webui-check.db --ids
   rm /tmp/webui-check.db
   ```

   **The limits of `compare`, so you read it correctly:**
   - It prints **counts** (and, with `--ids`, ids or the first 12 hex of a hash). Never message text.
   - It matches by **content hash**. OpenWebUI 0.11.4 does not send the message id to the compactor, so journal ids are
     always empty and id matching does not help.
   - So it can show **false "missing"**: a redacted (`/forget`) turn, an image-only turn, or a turn where document
     search (RAG) changed the prompt will not hash-match. Redacted ones are counted separately. Treat a small "missing"
     count that does not grow as noise.
   - The journal **starts at v3.1.9.10**. Nothing before her first reply on the new pod is in it. Before that first
     reply, `compare` exits 2 (there is nothing to compare).
   - It only tells you WHAT is missing. Tell her from which message to pick up. Her memory store may already know about
     those last exchanges (it is written on `/data` directly). That is expected, not damage.
5. **Do not restore a backup archive to get the lost minutes back.** Every archive is older than the snapshot.
6. Watch the first sync after her first message: `grep published $L/webuidb-sync-error.log | tail -1`. `/health/full`
   showing `stale: true` right after the first message is the known false alarm if a `published` line follows within
   about 2 minutes.

### The snapshot is damaged (the boot refused with exit 3)

The local disk is empty (the pod was lost), so the snapshot or an archive is all there is.

1. Copy it aside first: `mkdir -p $F && cp $SNAP $F/webui.db.bad-$(date -u +%Y%m%dT%H%M%SZ)` (and any `$SNAP-journal`).
2. Try the journal rollback: `RUNBOOK_DB_JOURNAL.md`, run against **`$SNAP`** (never `/data/openwebui/webui.db`).
   Then restart the pod.
3. Only if that fails: take the newest archive that verifies (`ls -t $BK/zions-backup-*.tar.gz | head -3`, then
   `$PY /opt/compactor/backup.py --verify <archive>`), and recover ONLY `webui.db` from it (its store is older than the
   live store; memory and chat would end up at different generations if both were restored). Extract `webui.db` to
   local disk, move the damaged `$SNAP` aside (the guarded command above), put the extracted file at `$SNAP`, and restart
   the pod so the boot restores it. What is lost is everything since that archive (up to about 1 hour); the journal
   (step 4 above) lists it. **Ask for help before doing this the first time.**

### Restoring an archive ON PURPOSE while the pod is running

```bash
supervisorctl stop openwebui compactor webuidb-sync backup
$PY /opt/compactor/backup.py --verify <archive>; echo VERIFY_EXIT=$?
$PY /opt/compactor/backup.py --restore <archive> --yes; echo RESTORE_EXIT=$?
$PY /opt/compactor/webuidb.py --status
```

In local mode this build's restore **publishes the restored database to `$SNAP` itself**. Check that it did: the two
`--status` rows must show the same `chats=`, `content=` and `newest_update=`. If they differ (or the restore says it could
not publish), publish by hand, straight after the restore and BEFORE starting anything:

```bash
WEBUI_DB_ALLOW_OLDER_GENERATION=1 WEBUI_DB_ALLOW_ROW_LOSS=1 WEBUI_DB_LOCAL=true \
  $PY /opt/compactor/webuidb.py --sync-once --force; echo "SYNC EXIT=$?"
$PY /opt/compactor/webuidb.py --status            # both rows identical
supervisorctl start webuidb-sync openwebui compactor backup
```

Without a published restore, a pod loss brings back the pre-restore chat next to the restored memory. A restore puts
back both her chat and her memory to the archive's moment; read `~/zl-ops/RESTORE-2026-09-23.md` "Read first" before
doing one, and re-check the model settings afterwards (they live in the database).

### E3. The boot refuses

In every one of these cases nothing under `/data/v10` was touched.

- **`RESTORE FAILED (exit 3)`**: the snapshot failed its check (typically a hot `-journal` that could not be rolled back
  on the volume). `RUNBOOK_DB_JOURNAL.md` against `$SNAP`, then restart. **Never** set `WEBUI_DB_ALLOW_EMPTY_START`.
- **`RESTORE FAILED (exit 4)`**: the pod's disk is full. Give the pod more container disk, then restart.
- **`exit 5`, or `AN INTERRUPTED RESTORE WAS LEFT IN PROGRESS`**: follow the banner's own instructions, and ask.

## 7. Backups on the new pod

1. The daemon writes to `$BK` (`COMPACTOR_BACKUP_DIR`). Retention: `COMPACTOR_BACKUP_RETAIN=3` hourly, plus 7 daily and
   8 weekly. `/data/backups` (the old pod's) stays as history; nothing on the new pod reads or prunes it.
2. **`zlpod pull backup` and the backup directory.** `zlpod` asks the pod for its backup directory (`zl disk`) and, if
   that fails, **falls back to `/data/backups`, which is the OLD pod's archives** (it prints a warning). Before you pull,
   check what the pod reports:

   ```bash
   zl disk | grep '^backups:'
   ```

   **Expect `... archive(s) in /data/v10/backups, ...`.** If it says `/data/backups`, the pod's environment is wrong;
   stop and fix the rows before pulling anything, or you will pull stale archives believing they are current.
3. **Archives made before the cutover are test archives** (the empty or seeded database, a test run). After the cutover
   they were moved out of the prune glob so retention never counts them as her history (`RUNBOOK_POD_MIGRATION.md`
   section 6): `ls $BK/pre-cutover-test/`. Do not restore from them.
4. **Reading the journal** (read-only): `$PY $R/chat-journal.py stats --store $STORE`, `verify`, `compare` (section 6, E2
   step 4 for how to read it).

## 8. Recovery point: what she is agreeing to, in plain words

- **Planned stops lose nothing, IF the final sync ran first and all three conditions held (section 4).**
- **`supervisorctl stop` alone (no manual final sync)** normally loses nothing (SIGTERM triggers a final publish), but do
  not rely on it.
- **A RunPod stop, `docker stop -t 10`, or any stop with a grace period shorter than a publish** loses the last
  unpublished writes: never more than one sync interval plus one publish, UNLESS the volume was refusing writes, in which
  case the window runs from the last `published` line.
- **A crash cannot damage the saved copy**: a crash mid-copy leaves the previous snapshot whole.
- **What she gets in exchange:** saves stop failing when the network volume stalls.
