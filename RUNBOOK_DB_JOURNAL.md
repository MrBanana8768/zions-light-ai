# Runbook — OpenWebUI database stuck on a rollback journal

For the pod as it runs today: `WEBUI_DB_LOCAL=false`, so OpenWebUI's database
lives on the RunPod network volume at `/data/openwebui/webui.db`. The commands
work on the v3.1.6.1 image and later. Run them in the pod's Web Terminal.

There is no `sqlite3` command in the image. Every database command below uses
OpenWebUI's own Python (`/app/venv/bin/python`).

---

## What this failure is, in one paragraph

The network volume (MooseFS) sometimes stalls for a moment. If that happens
while OpenWebUI is in the middle of saving, SQLite leaves a **hot rollback
journal** (`webui.db-journal`) beside the database. The next thing that opens
the database has to roll that journal back, which means writing. If the volume
is still slow, that write hangs or fails, and OpenWebUI never finishes
starting. In the browser that looks like the **"Backend Required"** screen, or a
JSON error such as `Unexpected token '<'`. The database is almost always
**not** corrupt: it is stuck halfway through its own recovery.

**The rule that matters most: the database and its journal are a matched
pair.** Deleting the journal turns a recoverable database into a corrupt one.

---

## Symptoms

- The browser shows **"Backend Required"**, or a JSON / `Unexpected token`
  error, or never gets past loading.
- `backup.py` reports `attempt to write a readonly database` or
  `database is locked`.
- `/data/logs/openwebui.log` shows `disk I/O error`,
  `attempt to write a readonly database`, or stops logging at startup.

---

## 1. Diagnose (read-only: nothing here changes anything)

**1.1 Service states**
```bash
supervisorctl status
```

**1.2 Is OpenWebUI's backend answering?**
```bash
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" -m 10 http://127.0.0.1:3000/api/config
```
- `200` means the backend is up; if the browser still errors, refresh it.
- `000` or `502` means it is not answering (still starting, or stuck).
- `500` means it answers with an error; read the log (1.3).

**1.3 OpenWebUI's recent log**
```bash
tail -n 80 /data/logs/openwebui.log
```

**1.4 The database and any sidecar files**
```bash
ls -la /data/openwebui/webui.db*
```
A `webui.db-journal` file next to `webui.db` is the signature of this failure.
(`webui.db-wal` / `webui.db-shm` should not exist on this pod; if they do,
stop and ask before going on.)

**1.5 Is the journal HOT (a real unfinished transaction)?**
```bash
/app/venv/bin/python -c "import pathlib;p=pathlib.Path('/data/openwebui/webui.db-journal');print('no journal' if not p.exists() else ('HOT journal (' + str(p.stat().st_size) + ' bytes)' if open(p,'rb').read(8)==bytes.fromhex('d9d505f920a163d7') else 'journal present but NOT hot (header zeroed): harmless leftover'))"
```
- **HOT journal**: this runbook applies. Go to section 2.
- **not hot**: a normal leftover; the problem is elsewhere, so read the log
  (1.3).
- **no journal**: not this failure.

**1.6 Which processes have the database open**
```bash
for p in /proc/[0-9]*; do ls -l "$p/fd" 2>/dev/null | grep -q 'webui.db' && echo "PID ${p#/proc/}: $(tr '\0' ' ' < "$p/cmdline" | cut -c1-120)"; done; echo done
```

**1.7 Can the volume be written to, and how fast?** Writes 16 MB with a
flush after each block, then deletes the test file.
```bash
dd if=/dev/zero of=/data/openwebui/.iotest bs=1M count=16 oflag=dsync 2>&1 | tail -1; rm -f /data/openwebui/.iotest
```
- A speed in MB/s means the volume is writable. Tens of MB/s is healthy.
- Under ~1 MB/s, or taking many seconds, means the volume is still slow.
  Recovery will be slow too; consider waiting a few minutes and re-running
  this check.
- `No space left on device` means the volume is full: do section 4 first.
- `Read-only file system` means the volume is not writable: contact RunPod;
  do not attempt recovery on it.

**1.8 Space used** (`df` on this volume shows the whole cluster, not your
quota, so use `du`)
```bash
du -sh /data/backups /data/forensics /data/openwebui /data/logs /data/models 2>/dev/null
```

**1.9 Read-only health check of the database**
```bash
/app/venv/bin/python -c "import sqlite3;c=sqlite3.connect('file:/data/openwebui/webui.db?mode=ro',uri=True,timeout=10);print(c.execute('PRAGMA quick_check').fetchone());print('chats',c.execute('select count(*) from chat').fetchone()[0])"
```
- `('ok',)` plus a chat count means the database is readable, with no hot
  journal blocking it.
- `attempt to write a readonly database` means a hot journal is waiting to be
  rolled back. That is expected with a HOT journal; go to section 2.
- `database is locked` means another process is mid-write. Wait a minute and
  re-run.
- `database disk image is malformed` means genuine corruption, NOT this
  failure. Do NOT continue here; go to section 6.

---

## 2. Recover (only with a HOT journal from 1.5)

Two ways. **2A** uses the recovery script if it is on the pod; it does the
rollback on local disk, which is safer when the volume is slow. **2B** is the
same thing by hand.

Every writer must be stopped first: OpenWebUI, the compactor and the backup
daemon. On this pod `webuidb-sync` does not run.

### 2A. With the recovery script

**Is the script on the pod?**
```bash
ls -la /data/scripts/recover-webui-db.py
```

**Stop the writers:**
```bash
supervisorctl stop openwebui compactor backup
```

**Dry run: copies to local disk, rolls back, checks; changes nothing on the
volume** (needs free space in `/tmp` of at least the database size; check
with `df -h /tmp`)
```bash
/opt/compactor-venv/bin/python /data/scripts/recover-webui-db.py --check
```
Continue only if it ends with `the database is recoverable`.

**The real run.** It renames the originals aside and never deletes them; it
refuses to swap unless the integrity check says ok; and it starts OpenWebUI at
the end, then waits for it to actually answer.
```bash
/opt/compactor-venv/bin/python /data/scripts/recover-webui-db.py --yes
```
Success ends with `RECOVERED — N chats, N users, integrity ok`. The broken
originals are moved to `/data/forensics/webui.db*.broken-<stamp>`, and a
verified copy stays in `/tmp/rescue` until the pod restarts.

**Then start the rest.** The script only ever starts/stops `openwebui` — it
never stopped `compactor` or `backup` (you did, above, by hand), so it
cannot know whether to restart them, and it says so at the end. Until you
run this, no chat reaches vLLM:
```bash
supervisorctl start compactor backup
```

Then go to section 3.

### 2B. By hand

**Stop the writers:**
```bash
supervisorctl stop openwebui compactor backup
```

**Confirm nothing still has the database open** (should print only `done`):
```bash
for p in /proc/[0-9]*; do ls -l "$p/fd" 2>/dev/null | grep -q 'webui.db' && echo "PID ${p#/proc/}: $(tr '\0' ' ' < "$p/cmdline" | cut -c1-120)"; done; echo done
```

**Copy the database AND journal together, before anything touches them:**
```bash
D=/data/forensics/journal-$(date -u +%Y%m%d-%H%M%S); mkdir -p "$D" && cp -p /data/openwebui/webui.db /data/openwebui/webui.db-journal "$D"/ && ls -la "$D"
```

**Roll the journal back and check.** Opening the database normally lets
SQLite finish its rollback. On a ~140 MB database this takes minutes on a slow
volume: **do not interrupt it.**
```bash
/app/venv/bin/python -c "import sqlite3;c=sqlite3.connect('/data/openwebui/webui.db',timeout=120);print(c.execute('PRAGMA quick_check').fetchone());print('chats',c.execute('select count(*) from chat').fetchone()[0])"
```
- `('ok',)` plus her chat count means continue.
- Anything else: STOP. Do not start OpenWebUI. Go to section 6.

**Confirm the journal is gone:**
```bash
ls -la /data/openwebui/webui.db*
```
Only `webui.db` should remain.

**Start OpenWebUI alone first:**
```bash
supervisorctl start openwebui
```

**Wait until this prints `200`** (re-run it every 30 seconds; a large database
can take a few minutes):
```bash
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" -m 10 http://127.0.0.1:3000/api/config
```

**Then start the rest:**
```bash
supervisorctl start compactor backup
```

---

## 3. Verify

**3.1 Services**
```bash
supervisorctl status
```
`openwebui`, `compactor` and `backup` should show RUNNING.

**3.2 The browser**: log in and open her conversation. The latest messages
from before the stall should be there. A message she sent during the stall
itself may be missing; that is the transaction the journal rolled back.

**3.3 No journal came back**
```bash
ls -la /data/openwebui/webui.db*
```

**3.4 Take a backup once she has chatted normally for a few minutes**, not
straight away (a backup reads the whole database, which is heavy I/O on a
volume that just stalled):
```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --once; echo "EXIT=$?"
```
Success is `EXIT=0` and a line starting `[OK] zions-backup-` (without
`--json`, `backup.py` prints `[OK] <archive> db=ok, ...`, never
`"ok": true` — that key only appears with `--json`). On v3.1.6.1–v3.1.8,
"NOT pruning — memory shrank" in that output is a known false alarm.

**3.5 Verify that archive** (use the name the backup printed):
```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --verify /data/backups/<archive-name>.tar.gz
```
It should print `[OK]`.

**3.6 Clean up** the forensic copies only after she has confirmed her history
is right:
```bash
ls -la /data/forensics/
```
```bash
rm -rf /data/forensics/journal-* /data/forensics/webui.db*.broken-*
```

---

## 4. If the volume is full

Old backups are the usual cause; on v3.1.6.1–v3.1.8 nightly backups never
prune (the "memory shrank" false alarm). Do this **with the services stopped**,
because deleting many large files is heavy work for this volume.

**Preview what would be deleted** (everything older than 2 days):
```bash
find /data/backups -maxdepth 1 -name 'zions-backup-*.tar.gz' -mtime +2 -print
```

**Delete them:**
```bash
find /data/backups -maxdepth 1 -name 'zions-backup-*.tar.gz' -mtime +2 -delete
```

**Check:**
```bash
du -sh /data/backups
```

Forensic copies from earlier incidents are the other big item:
```bash
du -sh /data/forensics/* 2>/dev/null
```

---

## 5. If it keeps coming back

A journal that returns within the hour means the volume is still stalling.

- **Do one heavy thing at a time.** Don't prune backups, take a backup and
  start OpenWebUI at the same moment.
- **Check the volume speed** with 1.7 before each recovery attempt.
- **Hold manual backups** until the volume is calm.
- **Collect evidence** for RunPod support and for us:
  ```bash
  F=/data/logs/journal-incident-$(date -u +%Y%m%d-%H%M%S).txt; { tail -n 200 /data/logs/openwebui.log; echo ----; ls -la /data/openwebui/; echo ----; supervisorctl status; } > "$F" 2>&1; echo "saved $F"
  ```
- **Check RunPod's status page** for network-volume incidents in the pod's
  data center.

The permanent way out is keeping the live database off the network volume
(`WEBUI_DB_LOCAL=true` with a snapshot on `/data`). That is a separate,
deliberate migration, not something to switch during an incident.

---

## 6. If the check says it is NOT ok

If `quick_check` reports anything other than `ok`, or the database says
`malformed`:

1. **Do not start OpenWebUI.** Leave the services stopped.
2. **Do not delete anything**, including the journal and the forensic copies.
3. **Do not run `backup.py --restore`** on v3.1.6.1–v3.1.8: it can make things
   worse. Use the manual move-aside restore in
   [OPERATIONS.md → Restore from a backup](OPERATIONS.md), which stops all
   writers and never deletes the live state.
4. Save the evidence (section 5) and ask for help first.

---

## Never do these

- `rm /data/openwebui/webui.db-journal`: deletes the half of the pair the
  database needs to recover.
- Put a different `webui.db` beside an existing journal: SQLite will "roll
  back" the new file with the old file's changes.
- Interrupt the rollback or integrity check halfway.
- Start OpenWebUI, the compactor and the backup daemon all at once right after
  a recovery.
- Run `backup.py --restore` on v3.1.6.1–v3.1.8.

---

## What v3.1.9 changes

- Nightly backups keep working through a hot journal: they copy the database
  and journal aside and back up the recovered copy, never touching the live
  pair. A failed backup retries in 15 minutes, not 24 hours.
- `/health/full` reports a hot journal held by nothing (a real stuck journal),
  and missing or stale backups.
- Nightly backups prune again.

None of that prevents the stall itself; this runbook still applies on v3.1.9.
