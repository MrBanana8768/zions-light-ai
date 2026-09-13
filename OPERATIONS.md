# Operations Runbook

For whoever is on the hook when something breaks. How to read the system's
health, recover from each known failure mode, and roll back a bad release.
Pairs with [USER_GUIDE.md](USER_GUIDE.md) (using the app) and
[RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md) (standing it up).

> Convention: commands run from inside the pod (RunPod **Web Terminal**).
> The compactor listens on `localhost:8080`; admin endpoints are
> localhost-only by design.

---

## Is it healthy right now?

```bash
# One-shot verdict: vLLM reachable + storage writable + memory + backups
curl -s http://localhost:8080/health/full | python3 -m json.tool

# Process status
supervisorctl status
```

(`jq` is not installed in the image; `python3 -m json.tool` is.)

`/health/full` returns one of:
- `"status": "ok"` — no reason fired. **Not the same as "everything works"**,
  especially on the image the pod runs today: see "Reading /health/full" below
  for what each release does and does not check.
- `"status": "degraded"` — one or more reasons in `status_reasons`. Read them;
  each names its cause (vLLM unreachable, background work shedding, memory
  tail skipping, a hot SQLite journal, unreadable memory, ...). HTTP 200 (the
  container is intentionally *not* killed — supervisord can restart vLLM
  independently).
- `"status": "down"` — **storage broken** (`/data` not writable). HTTP 503.
  Nothing useful is possible; the container should be replaced.

Deeper, on demand:
```bash
curl -s http://localhost:8080/admin/selftest | python3 -m json.tool   # real chat round-trip + facts I/O
cat /data/logs/selftest.log                                         # the boot self-test result
```

### Reading /health/full — do not trust `.status` alone

Which release the pod runs changes what `status` notices (hostile review of
v3.1.7, reviewer C, F4):

| problem | v3.1.6.1 (the pod today) and v3.1.7 | v3.1.8 | from v3.1.9 |
|---|---|---|---|
| a memory file is unreadable | `status: ok` | degrades | degrades |
| a hot SQLite journal beside `webui.db` | `status: ok` | degrades | degrades |
| no backup archives at all | `status: ok` | `status: ok` | degrades once the compactor has been up one backup interval (24 h) |
| newest backup too old | `status: ok` | `status: ok` | degrades when older than 1.5 × the interval (36 h by default) |
| the backup directory cannot be read | `status: ok` | `status: ok` | degrades: `backup status unobservable` |

(The 24 h / 36 h figures assume the default `COMPACTOR_BACKUP_INTERVAL_HOURS=24`;
if the template sets another interval, both scale with it, and so should the
30 hours below.)

So on every release, run this and read all three lines, whatever `status`
says — it is the only check that works the same on today's image and on
v3.1.9, and its 30-hour line warns six hours before v3.1.9's own reason does:

```bash
curl -s localhost:8080/health/full | python3 -c "
import json,sys,time
d=json.load(sys.stdin); b=d.get('backups') or {}; m=b.get('latest_mtime')
print('status:', d['status'], d['status_reasons'])
print('unreadable memory files:', d['stats'].get('unreadable'))
print('newest backup:', b.get('latest'), '| age (hours):', round((time.time()-m)/3600,1) if m else None, '| count:', b.get('count'))"
```

- **`unreadable memory files`** must be `{'facts': 0, 'summaries': 0, ...}` —
  every number 0. Anything above 0 is a corrupt memory file: those
  conversations are not being read and must not be written over. Ask for
  help before restarting anything.
- **`newest backup`** must name an archive, and its age must be under about
  **30 hours** (the daemon runs every 24 h). `None`, or an age over 30 hours,
  means backups have stopped. On v3.1.6.1 `status` still says `ok` when that
  happens; from v3.1.9 it degrades past 36 hours. Either way, go to "Backups
  stopped or failing" below.

#### What "memory tail skipping" means

v3.1.7 added a reason that reads `memory tail skipping: N reply(ies) not
memorized (... last outcome <label>)`. It degrades `status` for 5 minutes after
a reply did not go into memory, then clears itself. It is unchanged in v3.1.9:
the degraded windows are deliberate, because every one of these skips is a
reply she read that did not reach memory. **Most of the time it is the new
signal working, not a fault.** (v3.1.6.1, the image the pod runs today, has no
memory-tail tracking at all: the command below prints a `KeyError` there.)
To see which outcomes actually happened since the compactor started:

```bash
curl -s localhost:8080/health/full | python3 -c "import json,sys; o=json.load(sys.stdin)['memory_tail']['outcomes']; [print(f'{k:30} {v}') for k,v in o.items() if v]"
```

| outcome | what it means | action |
|---|---|---|
| `stored`, `stored_trimmed` | memorized (trimmed = a stopped reply, kept up to its last sentence) | none |
| `skipped_empty`, `skipped_task_traffic` | a Stop before the first word, or an OpenWebUI title/tag/follow-up call; nothing was lost, never degrades on its own | none |
| `skipped_degenerate`, `skipped_degenerate_partial` | the reply looped or repeated itself, so it was kept out of memory on purpose | none; expected a few times a day. Many in a row means the model is looping: read the replies |
| `skipped_too_short`, `skipped_no_boundary` | a stopped/cut reply with no complete sentence worth keeping | none; expected |
| `skipped_holed` | **fault**: the stream lost a piece of the reply | `grep -a "skipping memory tail" /data/logs/compactor.log \| tail -5`, and check `/data/logs/vllm.log` around that time |
| `skipped_disk_pressure` | **fault**: memory writes paused, disk nearly full | "Disk is filling up" below |
| `skipped_shed` | **fault**: the background pool was overloaded and dropped the write | look for the `background work shedding` reason; if it repeats, restart the compactor when she is not chatting |
| `skipped_no_user_text` | **fault-shaped**: a reply with no user message to pair it with | `grep -a "skipping memory tail" /data/logs/compactor.log \| tail -5` and report what it says |

The reason's own `last outcome` names the last skip of ANY kind, including a
harmless `skipped_empty`, so read the table above rather than that label. A
separate reason (from v3.1.9), `the backend has returned no text for N
consecutive replies`, is always a fault. Send vLLM one completion by hand (answering
`/v1/models` does not prove it is generating):

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model": "'"$MODEL_REPO"'", "prompt": "Say hello.", "max_tokens": 16}' | python3 -m json.tool
```

**Working:** a `"choices"` list whose `"text"` is not empty. **Not working:**
an error, or empty text — read `tail -100 /data/logs/vllm.log`. If `$MODEL_REPO`
is empty in your terminal, replace `"'"$MODEL_REPO"'"` with the model name in
quotes, e.g. `"coder3101/Cydonia-24B-v4.3-heretic-v4"`.

(In the `grep` commands in the table, type `|` where it shows `\|`.)

---

## Reading the logs

```bash
tail -f /data/logs/vllm.log        # inference engine
tail -f /data/logs/compactor.log   # memory + compaction + requests
tail -f /data/logs/openwebui.log   # frontend
tail -f /data/logs/selftest.log    # boot self-test (one-shot)
tail -f /data/logs/backup.log      # backup daemon
```

Logs live on the volume (`LOG_DIR`, default `/data/logs`), so they survive a
redeploy. Older docs said `/var/log/supervisor`; the service logs are not there.
Press Ctrl+C to stop a `tail -f`.

What the compactor lines mean:

| Log line | Meaning |
|---|---|
| `injected memory [persona(..) Nfact(s) Mretr sum(L1=../L2=../L3=..)]` | What was fed to the model this turn — normal |
| `extracted N new fact(s)` | Post-turn fact extraction succeeded |
| `extracted 0 fact(s) — model returned: '...'` | Extraction ran; model judged nothing memorable (or returned the raw text shown — diagnostic) |
| `indexed exchange (turn ~N)` | Episodic RAG indexed the turn — normal |
| `rollup → L1=.. L2=.. L3=..` | Hierarchical summary advanced — normal |
| `dedup merged N duplicate fact(s)` | Near-duplicate facts merged — normal |
| `... failed (non-fatal): ...` | A memory layer degraded to a no-op; **chat was not affected** |
| `Exception in ASGI application` + `ConnectError: All connection attempts failed` | The compactor couldn't reach vLLM (vLLM down/restarting) |

---

## Failure mode → recovery

### A service is FATAL (supervisord gave up restarting it)
`supervisorctl status` shows each program's state. `RUNNING` is healthy;
`FATAL` means it failed to start `startretries` times and supervisord
**stopped trying** — by design, so a genuinely-broken service stays visible
instead of fast-restart-looping and hiding the cause.

```bash
supervisorctl status
# vllm    FATAL     Exited too quickly (process log may have details)
```

1. Read that service's log: `tail -100 /data/logs/<name>.log` (and
   `/data/logs/<name>-error.log`).
2. Fix the root cause (see below for vLLM).
3. Clear FATAL and retry: `supervisorctl start <name>` (or
   `supervisorctl restart <name>`).

The background-work pool and disk-pressure state are visible in
`/health/full` (`background_work`, `memory_writes`); a FATAL *vLLM* shows as
`status: degraded` there, and a FATAL *compactor* makes `/health/full`
itself unreachable (so "curl refused on :8080" == compactor down).

### vLLM won't start / keeps restarting
1. `tail -100 /data/logs/vllm.log` — look for the real error.
2. **CUDA OOM during startup** is the most common. Cause: model too big for
   the GPU. On an A40, use `MODEL_REPO=anthracite-org/magnum-v4-12b` (the
   22B + FP8 OOMs during the marlin repack — see
   [RUNPOD_DEPLOY.md → GPU sizing](RUNPOD_DEPLOY.md#gpu-sizing)). Lower
   `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION` if still tight.
3. If a service is fast-restart-looping, stop it so the root cause stays
   visible: `supervisorctl stop vllm`, fix, `supervisorctl start vllm`.

### Chat returns errors but the pod is up
- Check `/health/full`. If `degraded`, vLLM is the problem (above). The
  compactor itself rarely 500s — memory failures degrade to no-ops.

### Disk is filling up
```bash
du -sh /data/* 2>/dev/null | sort -h
```

**Read `du`, not `df`.** `/data` is a RunPod network volume (MooseFS), and
`df -h /data` reports the free space of the whole shared cluster (hundreds of
terabytes), not your volume's quota. The quota is the size you gave the volume
in the RunPod console (e.g. 200 GB): add up the `du` lines and compare with
that. The first sign of hitting it is a write error in the live store, not a
warning, because the backup daemon's 500 MB free-space guard reads the same
misleading number and can never fire here.

- Backups (`/data/backups`) are usually the fastest-growing directory: ~116 MB
  each. **On v3.1.6.1 (the pod today) nothing has pruned them since
  2026-08-30**; v3.1.9 fixes the cause (see "Nightly 'memory shrank' alert"
  below).
- Model weights under `/data/models` are the other space hog; remove unused
  ones with `/opt/clean-models.sh` (see
  [Cleaning up old model weights](#cleaning-up-old-model-weights-on-the-volume)).

### Nightly "memory shrank" alert — noise on v3.1.6.1 to v3.1.8, a real signal from v3.1.9

**What you see:** `backup.log` says

```
backup ok: zions-backup-….tar.gz (…); NOT pruning — memory shrank since zions-backup-…: <id>.facts 193->139, <id>.summaries 8->3
```

and, if `COMPACTOR_ALERT_WEBHOOK` is set, a failure alert saying
`backup published but memory shrank …`. Either way the backup WAS made and
verified; only the cleanup of old archives was skipped.

```bash
grep -aE "NOT pruning|pruned [0-9]+" /data/logs/backup.log | tail -3
```

#### On v3.1.6.1, v3.1.7 and v3.1.8 (the pod today): mostly noise, and nothing prunes

The check on these releases compares raw fact and summary counts with the
previous night and calls any decrease "memory shrank". Decreases are normal:
when a conversation's facts reach their size cap the oldest move to an archive
file, and when 20 summaries are rolled into one chapter the count drops. On
this pod it has fired on every nightly run since 2026-08-31, so no nightly run
has pruned since 2026-08-30 (hostile review of v3.1.7, reviewer C, F1).
Archives pile up until the volume quota is hit.

How to tell noise from a real loss on these releases — read the list after
`memory shrank since …:`

- **Noise:** every item ends in `.facts N->M` or `.summaries N->M` with `M`
  above 0.
- **Real — do not prune, investigate:** any item ending in `->0`, or any
  `.episodic` item at all (the episodic index is never shrunk by normal
  operation). Leave the archives alone and ask for help; the older archives
  are the ones that hold what was lost.

#### From v3.1.9: the alert means something, and pruning resumes by itself

v3.1.9 counts what cannot come back rather than raw numbers: facts in the
active file AND its archive file together (eviction moves facts between the
two, so the total does not drop), the highest summarized turn (a rollup never
lowers it), and the archived-chapter file. So the items it can name are:

| item in the list | meaning |
|---|---|
| `<id>.facts N->0 (emptied)` | every fact of that conversation, active and archived, is gone |
| `<id>.summary_turn N->M` | that conversation's summaries now cover fewer turns than yesterday |
| `<id>.archived_chapters N->M` | archived chapter summaries were lost |
| `<id>.episodic N->M` | indexed exchanges were lost (a `/forget` or a memory reset can also do this) |

**On v3.1.9, treat every `NOT pruning — memory shrank` as real.** Do not prune
by hand; leave the archives alone and ask for help, unless you know the named
conversation was deliberately reset.

**What to look for after the v3.1.9 deploy:** the first nightly cycle (within
about 24 hours of the boot — the daemon skips the boot-time run if a backup
is less than 12 hours old) should end in

```
backup ok: zions-backup-….tar.gz (…); pruned N; …s
```

`N` can be 0 — that only means every archive still falls inside the
retention rules — but the line must say `pruned`, not `NOT pruning`. The
comparison against the last v3.1.6.1 archive is safe: the new fields read as 0
in the old manifest, so only a real episodic loss can trip it that night. If
the first v3.1.9 night says `NOT pruning`, read its list with the table above.

#### Safe manual prune (v3.1.6.1 to v3.1.8 only, and only when the check says noise)

Run it on the old image to free space before the v3.1.9 deploy if the volume
is tight; after v3.1.9 the daemon does this itself. First see what it WOULD
delete — this deletes nothing:

```bash
/opt/compactor-venv/bin/python -c "
import sys; sys.path.insert(0, '/opt/compactor')
import backup as b
a = b.list_backups(); keep = b._keep_set(a, floor=max(b.MIN_KEEP, b.RETAIN))
gone = [e for e in a if e['name'] not in keep]
print('archives:', len(a), '| keep:', len(keep), '| would delete:', len(gone), '|', round(sum(e['size_bytes'] for e in gone)/1e9, 1), 'GB')
print('rules: keep everything newer than', b.RETAIN_DAYS, 'days, one per week for', b.GFS_WEEKS, 'weeks, never fewer than', max(b.MIN_KEEP, b.RETAIN))
[print('  would delete', e['name']) for e in gone]"
```

**Check before going on:** the newest archives (today's, yesterday's) are NOT
in the "would delete" list, and "keep" is at least 3. If anything looks wrong,
stop. Then prune for real:

```bash
/opt/compactor-venv/bin/python -c "
import sys; sys.path.insert(0, '/opt/compactor')
import backup as b
removed = b.prune_old_backups(); print('deleted', len(removed)); [print('  ', n) for n in removed]"
```

**Success:** `deleted N` with the same names the preview listed. Then
`du -sh /data/backups` should be smaller. The same rules decide what is kept
as the daemon uses on a clean night, so this never goes below the retention
floor.

### Backups stopped or failing ("readonly database" / "database is locked")

**What you see:** `backup.log` has `backup failed: OperationalError: attempt
to write a readonly database` or `… database is locked`, or the "newest
backup" age from "Reading /health/full" is over 30 hours.

**On v3.1.6.1 to v3.1.8 (the pod today):** after a failed run the daemon waits
a full 24 hours before trying again, and `/health/full` still says `ok`
(reviewer C, F3). A hot rollback journal beside `webui.db` makes every attempt
fail with `readonly database`, and that is exactly the state that precedes
needing a backup.

**From v3.1.9:** a failed run is retried after 15 minutes — the log says
`backup cycle failed: …; retrying in 15 min instead of the full 24.0h
interval` — and `/health/full` degrades once the newest archive is older than
36 hours. For a hot journal, v3.1.9 also tries to back up from a copy of the
database and its journal instead of failing; when it does, `backup.log` shows
a WARNING containing `this is the hot rollback journal signature`. That backup
is fine, but **the journal on the live database is still there**: repair it
("A HOT SQLite rollback journal" below). That fallback has been tested on
the unit-test image's SQLite but not yet confirmed against the SQLite in the
production image, so do not rely on it: check by hand as below.

**On every release — every time you see one of those errors, or after any
"database is locked" episode in the OpenWebUI log, run a backup by hand and
read what it says:**

```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --once; echo "EXIT=$?"
```

- **`[OK] zions-backup-….tar.gz …` and `EXIT=0`:** fine, a fresh backup
  exists. (If the line also says `memory shrank`, see the section above —
  which release you are on decides what it means.)
- **`[FAIL] … readonly database` and `EXIT=1`:** check for a hot journal —
  "A HOT SQLite rollback journal beside `webui.db`" below — repair it, then run
  the backup command again until it prints `[OK]`.
- **`[FAIL] … database is locked` and `EXIT=1`:** OpenWebUI was mid-write.
  Wait two minutes and run it again. If it fails three times in a row, treat
  it as the hot-journal case.
- **Any other `[FAIL]`:** copy the whole line and ask for help.

Check daily, or after any volume hiccup:

```bash
ls -lt /data/backups | head -3
```

Ignore the first line (`total …`); the date on the next line, the newest
archive, should be within the last 30 hours.

### A HOT SQLite rollback journal beside `webui.db`

**Seen twice: 2026-08-31 (wedged the pod) and 2026-09-07 (silent).** Both on
the MooseFS volume, which is the actual cause — see the note at the end.

#### What it looks like

The 08-31 shape is loud: OpenWebUI answers "no backend", every query returns
`sqlite3.OperationalError: disk I/O error`, and writes report
`attempt to write a readonly database`. SQLite is trying to roll the journal
back on every open; rolling back requires WRITING; the write fails; SQLite
protects the file by reporting readonly. **The database is not corrupt** — it
is stuck mid-recovery on a filesystem that will not let it finish.

The 09-07 shape is silent, and it is the one to know about. An ORPHANED hot
journal sat beside a database that was being written to perfectly normally:
`/health/full` said `ok`, storage said writable, `indexed_exchanges_total` was
climbing, and the database's mtime advanced minute by minute while the journal
sat 36 minutes stale. Nothing was failing. The uncommitted transaction was
simply waiting for the next process to open the database and roll it back.

#### Do not diagnose it by the file's existence, or by opening the database

A `-journal` file is ORDINARY. In `delete` mode SQLite creates one around
every transaction and removes it on commit; in `persist` mode it deliberately
leaves one behind with a zeroed header. Catching one mid-write means nothing.

And a second connection **cannot** tell a hot journal from a live writer's: it
must take a write lock to find out, OpenWebUI holds that lock, and the probe
fails with the same `attempt to write a readonly database` text either way.
That ambiguity wasted real time on 09-07.

#### The check: read eight bytes

```bash
/opt/compactor-venv/bin/python -c "
import os
j='/data/openwebui/webui.db-journal'
b=open(j,'rb').read(8) if os.path.exists(j) else b''
print('header:', b.hex() or '(no journal)')
print('HOT - uncommitted transaction pending' if b.hex()=='d9d505f920a163d7' else 'clean')"
```

`d9d505f920a163d7` is SQLite's rollback-journal magic. Anything else — zeros,
or no file — is debris or nothing. Reading takes no lock and cannot be
confused by a healthy writer.

Since v3.1.8 `/health/full` does this itself and degrades on it. v3.1.6.1, the
image the pod runs today, does not report it at all (the command below prints a
`KeyError` there), so on v3.1.6.1 use the eight-byte check above. On v3.1.8
and later:

```bash
curl -s localhost:8080/health/full | python3 -c "import json,sys; print(json.load(sys.stdin)['checks']['sqlite_journal'])"
```

#### The repair: let SQLite do it

```bash
supervisorctl stop openwebui
```

```bash
/opt/compactor-venv/bin/python -c "import sqlite3; c=sqlite3.connect('/data/openwebui/webui.db'); print('journal_mode:', c.execute('PRAGMA journal_mode').fetchone()[0]); print('quick_check:', c.execute('PRAGMA quick_check').fetchone()[0]); c.close()"
```

```bash
supervisorctl start openwebui
```

Opening it read-write with the writers stopped IS the repair — SQLite rolls
the journal back and deletes it. `quick_check: ok` and a vanished journal file
mean it is done. On 09-07 that whole sequence took under a minute.

If the volume is refusing writes (the 08-31 shape) the open will fail. Then
copy the database **and its journal together** to local disk, open it there so
the rollback can complete, `PRAGMA integrity_check`, `VACUUM`, and copy back —
renaming both originals aside rather than deleting them.

#### The one thing that turns this into real damage

> **Never delete the journal by hand.** It and the database are a matched
> pair. Deleting a hot journal turns a recoverable file into a corrupt one,
> and leaving a stale journal beside a *replaced* database corrupts that one
> too. Renaming both together is what makes a swap safe.

#### Afterwards: check nothing was rolled away

A rollback undoes an incomplete transaction, so confirm the counters did not
go backwards.

```bash
curl -s localhost:8080/health/full | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d['stats'])"
```

Compare against the newest backup's own numbers. Lower means the rollback took
something and a restore is the answer.

#### Why it recurs

```
mfs#ca-mtl-1.runpod.net:9421  965T  765T  201T  80% /data
```

`webui.db` lives on MooseFS, and SQLite on a network filesystem is the
known-fragile pairing: the volume drops I/O mid-transaction and leaves a
journal behind. v3.1.6's `webuidb.py` can move the live database to the pod's
local disk and sync it to `/data`, which removes the cause — but production
deliberately runs with that move switched OFF (`WEBUI_DB_LOCAL=false`, see
RUNPOD_DEPLOY.md), so `webui.db` is still on MooseFS. Expect this on any volume
hiccup, keep the eight-byte check to hand, and run a backup by hand afterwards
("Backups stopped or failing" above).

### Memory looks wrong for one conversation
See [USER_GUIDE.md](USER_GUIDE.md). Quick: `/why` in the chat,
`/list-facts`, `/forget <substring>`, or full reset
`curl -X DELETE localhost:8080/admin/conversations/<id>/facts`.

---

## Cleaning up old model weights on the volume

When you change `MODEL_REPO` (e.g. `anthracite-org/magnum-v4-12b` → a
Cydonia-24B), vLLM downloads the new weights but the **old** ones stay cached
on the Network Volume under `HF_HOME` (`/data/models/hub/models--<org>--<name>/`).
Each model is 10–50 GB, so a few swaps can fill the volume. `df -h /data` /
`du -sh /data/* | sort -h` will show it.

The image ships `/opt/clean-models.sh` for exactly this. It is **safe by
default**: with no flags it only *lists* what's cached and marks the ACTIVE
model — nothing is deleted. The active model (derived from `$MODEL_REPO`) is
**always protected**, even if you name it explicitly. Deletion needs an
explicit `--yes` or a `y` at the confirmation prompt, and it prints the space
it freed.

```bash
# 1. See what's cached and which model is active (dry run — deletes nothing)
/opt/clean-models.sh

#   5.0G   models--TheDrummer--Cydonia-24B-v2   <== ACTIVE (protected)
#  23G    models--anthracite-org--magnum-v4-12b

# 2. Delete ONE specific old model (accepts "org/name" or the hub dir name)
/opt/clean-models.sh --delete anthracite-org/magnum-v4-12b        # prompts y/N
/opt/clean-models.sh --delete anthracite-org/magnum-v4-12b --yes  # no prompt

# 3. Keep only the active model, remove everything else
/opt/clean-models.sh --prune-others          # prompts, lists exactly what goes
/opt/clean-models.sh --prune-others --yes    # for scripts / non-interactive

# 4. Also clear vLLM's torch.compile cache (/data/vllm-compile-cache).
#    It's a REGENERABLE perf cache — safe to delete, but the next cold start
#    re-captures CUDA graphs and is 60–120s slower that one time.
/opt/clean-models.sh --compile-cache               # compile cache only
/opt/clean-models.sh --prune-others --compile-cache --yes   # both at once
```

Safety notes:
- **Dry-run default** — running it with no flag never deletes anything.
- **Active model is protected** — `--delete <active>` is refused, and
  `--prune-others` always keeps it. `--prune-others` also refuses to run if
  `MODEL_REPO` is unset (it wouldn't know what to keep).
- **Bounded deletes** — it only ever `rm -rf`s real directories named
  `models--*` directly under `$HF_HOME/hub`; it will never touch `/`, the cache
  root, or an arbitrary path.
- Run it from the pod's **Web Terminal**, or from outside with
  `docker exec <container> /opt/clean-models.sh`.
- Removing a model is harmless: if you switch `MODEL_REPO` back to it later,
  vLLM just re-downloads it on the next start.

---

## Backups & restore (data durability)

The backup daemon snapshots `webui.db` (chat history) + the `compactor/`
memory store to timestamped, **verified** archives in `/data/backups`. Every
archive is integrity-checked (SQLite `PRAGMA integrity_check` + JSON parse)
before it's published — a backup that doesn't verify is discarded, never
silently kept.

> **Scope:** backups currently live on the **same volume** as the data.
> They protect against corruption, an accidental `/forget`, and torn writes
> — **not** total volume loss. Off-volume disaster recovery is planned (see
> [ROADMAP.md → V2.3](ROADMAP.md#v23--resilience--stability)).

### Check backup status
```bash
curl -s http://localhost:8080/admin/backups | python3 -m json.tool   # list + latest summary
ls -lh /data/backups/
tail -20 /data/logs/backup.log
```

### Make / verify a backup on demand
```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --once; echo "EXIT=$?"
/opt/compactor-venv/bin/python /opt/compactor/backup.py --list
/opt/compactor-venv/bin/python /opt/compactor/backup.py --verify /data/backups/<archive>.tar.gz
```

`--once` prints `[OK] <archive> <detail>` and exits 0 when a backup was made
and verified; `[FAIL] …` and exit 1 when not (see "Backups stopped or failing"
above). `--verify` prints `[OK] db=ok, …` for a good archive.

### 🔥 Restore from a backup (recover lost/corrupted memory)

**Use the manual procedure below on every release.** It never deletes
anything: the live state is RENAMED aside, and a rename on the same volume
cannot be left half done.

> **On v3.1.6.1, v3.1.7 and v3.1.8 — the image the pod runs today — NEVER run
> `backup.py --restore`.** On those images it copies the archived database
> over the live one while leaving the live file's rollback journal beside it
> (the restored database then opens as "malformed"), and it deletes the whole
> live memory store BEFORE copying the archive's in — a restart or a full
> volume in between leaves no memory at all (hostile review of v3.1.7,
> reviewer C, F2 and Attack 5, run against the v3.1.6.1 and v3.1.7 images;
> v3.1.8 carries the same code).
>
> **From v3.1.9, `backup.py --restore` is rewritten, available, and not yet
> reviewed.** It now copies everything it will restore next to its destination
> first, moves
> the old database's journal and the old memory store into `/data/forensics`
> instead of deleting them, checks free space first, refuses to start if a
> writer is holding the database mid-write, and can list what it set aside
> (`backup.py --list-pre-restore`). It is still not the documented path, for
> reasons in the code as well as the calendar: the final hostile review of
> v3.1.9 has not cleared it; it cannot see a writer when a journal already
> exists (exactly the incident state), so stopping the writers is still on you;
> if its last step fails it can leave the restored database with the old
> memory store; its free-space check reads MooseFS's cluster-wide number; and
> it does not integrity-check the result or tell you what to start. The manual
> procedure does all of those explicitly. Until that review clears it, use the
> manual procedure on v3.1.9 too.

**This procedure is for the production placement, `WEBUI_DB_LOCAL=false`**,
where the live chat database IS `/data/openwebui/webui.db`. Every line runs in
the Web Terminal. Tell her the app will be unavailable for about 15 minutes.

**There are four writers, and all four are stopped.** `openwebui` and the
`compactor` write the data; the `backup` daemon reads it and could archive a
half-restored state; `webuidb-sync` copies the database between local disk and
`/data` on pods with `WEBUI_DB_LOCAL=true`. On this pod `webuidb-sync` should
not be running at all, and step 2 checks that; it is still named in the stop
command so the command is right on any pod.

**1. Pick an archive and verify it.**

```bash
ls -1t /data/backups/ | head -10
```

The list is newest first. Pick the newest archive from BEFORE the problem
started. Then:

```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --verify /data/backups/<archive>.tar.gz
```

**Success:** a line starting `[OK] db=ok`. **If `[FAIL]`:** do not use that
archive; verify the next older one.

**2. Check the placement.**

```bash
supervisorctl status webuidb-sync
```

**Success:** `STOPPED` and `Not started`. **If `RUNNING`:** STOP HERE. This pod
has the database on local disk and this procedure would restore the wrong
file. Ask for help.

**3. Check there is room.** The old state is kept, not deleted, so the volume
needs space for a second copy:

```bash
du -sh /data/openwebui/compactor /data/openwebui/webui.db /data/backups/<archive>.tar.gz
```

Add the first two numbers. Your volume's quota (the size set in the RunPod
console) minus the total of `du -sh /data/* 2>/dev/null` must be at least that
much. If it is not, ask for help. Do not prune backups to make room during a
restore: the older archives may be exactly the ones you need. Do not rely on
`df`; see "Disk is filling up".

**4. Stop all four writers and make sure they are gone.**

```bash
supervisorctl stop openwebui compactor backup webuidb-sync
```

**Expected:** `openwebui: stopped`, `compactor: stopped`, `backup: stopped`,
and `webuidb-sync: ERROR (not running)` — that last line is correct on this
pod. Then:

```bash
pgrep -af "open-webui serve|uvicorn main:app|backup.py --daemon|webuidb.py" || echo "nothing running - OK"
```

**Success:** `nothing running - OK`. **If any process is listed:** wait 30
seconds and run it again. `supervisorctl stop` asks a process to exit and one
that has not finished exiting still has the database open; a restore underneath
it is invisible to it. If something is still listed after two minutes, ask for
help.

**5. Unpack the archive beside the live data.**

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S); echo "STAMP=$STAMP   <- write this down"
```

If the terminal disconnects at any later step, open a new one and type
`STAMP=<the value you wrote down>` before continuing.

```bash
mkdir -p /data/restore-$STAMP && tar -xzf /data/backups/<archive>.tar.gz -C /data/restore-$STAMP; echo "EXIT=$?"; ls /data/restore-$STAMP
```

**Success:** `EXIT=0`, then `compactor  manifest.json  webui.db`. **If `EXIT`
is not 0:** the unpack failed (often: no space). Nothing live has been
touched; run `rm -rf /data/restore-$STAMP`, restart with
`supervisorctl start compactor openwebui backup`, and sort out the space.

**6. Check the unpacked database.**

```bash
/opt/compactor-venv/bin/python -c "import sqlite3,sys; c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro', uri=True); print(c.execute('PRAGMA integrity_check').fetchone()[0])" /data/restore-$STAMP/webui.db
```

**Success:** it prints `ok`. **Anything else, or an error:** do not use it.
Nothing live has been touched; restart with
`supervisorctl start compactor openwebui backup` and try an older archive.
(`sqlite3` is not installed on the pod; this is the same check through Python,
opened read-only so it cannot create or change the file.)

**7. Move the live state ASIDE — never delete it.**

```bash
mkdir -p /data/forensics/pre-restore-$STAMP && mv /data/openwebui/compactor /data/forensics/pre-restore-$STAMP/ && mv /data/openwebui/webui.db* /data/forensics/pre-restore-$STAMP/; echo "EXIT=$?"; ls -la /data/forensics/pre-restore-$STAMP/
```

**Success:** `EXIT=0`, and the listing shows `compactor` and `webui.db` — plus
`webui.db-journal` if there was one. The `*` is deliberate: a database and its
journal are a matched pair and must move together. **If `EXIT` is not 0:** go
to "Undo" below.

**8. Move the restored copy into place.**

```bash
mv /data/restore-$STAMP/compactor /data/openwebui/compactor && mv /data/restore-$STAMP/webui.db /data/openwebui/webui.db; echo "EXIT=$?"
```

```bash
/opt/compactor-venv/bin/python -c "import sqlite3,sys; c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro', uri=True); print(c.execute('PRAGMA integrity_check').fetchone()[0])" /data/openwebui/webui.db
```

**Success:** `EXIT=0`, then `ok`. These are renames on one volume, not
copies: a crash here leaves either the old path or the new one, never half a
tree. **If either fails:** go to "Undo".

**9. Start the three writers that belong on this pod.**

```bash
supervisorctl start compactor openwebui backup
```

Do NOT start `webuidb-sync` on a `WEBUI_DB_LOCAL=false` pod: it would copy a
stale local file over the database you just restored. Wait a minute, then:

```bash
supervisorctl status compactor openwebui backup
```

**Success:** all three `RUNNING`. OpenWebUI can take a minute or two; run the
status again before worrying.

**10. Confirm.**

```bash
/opt/compactor-venv/bin/python -c "import json; m=json.load(open('/data/restore-$STAMP/manifest.json')); print('archive conversations:', len(m['sources']['compactor']['conversations']))"
curl -s localhost:8080/health/full | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d['status_reasons']); print('conversations:', d['stats'].get('conversations'), '| unreadable:', d['stats'].get('unreadable'))"
```

**Success:** the two conversation numbers are within a handful of each other
(they count slightly differently), and every `unreadable` number is 0. Then
open her chat in the browser: the history should be there up to the archive's
time.

**Her conversation id after a restore.** `webui.db` also holds OpenWebUI's
settings, including the `X-Conversation-Id` connection header
(RUNBOOK_MEMORY_IDENTITY.md step 3). If the archive is older than that change,
the header is gone and her chat is back on its hash id. Before she chats, check
Admin Panel → Settings → Connections → the `localhost:8080` connection →
Headers. If it is empty, redo RUNBOOK_MEMORY_IDENTITY.md from step 1.

**The next backup cycle will say `NOT pruning — memory shrank`.** It compares
the restored (older) memory with the newest archive, taken before the restore,
so on v3.1.9 it names the turns and exchanges the restore rolled back. That is
expected once, right after a restore; the cycle after it should prune again.

**11. Clean up — only after she has confirmed** her history and memory are
right (a day later is fine):

```bash
rm -rf /data/forensics/pre-restore-$STAMP /data/restore-$STAMP
```

**Undo** (step 7, 8, 9 or 10 went wrong). Stop the writers, set the restored
state aside, and move the original back:

```bash
supervisorctl stop openwebui compactor backup
ls -la /data/openwebui/ /data/forensics/pre-restore-$STAMP/
```

Move back whatever is in `pre-restore-$STAMP`. If `/data/openwebui/compactor`
or `/data/openwebui/webui.db` exists (the restored copy), move it out of the
way first:

```bash
mkdir -p /data/forensics/failed-restore-$STAMP
mv /data/openwebui/compactor /data/forensics/failed-restore-$STAMP/ 2>/dev/null; mv /data/openwebui/webui.db* /data/forensics/failed-restore-$STAMP/ 2>/dev/null
mv /data/forensics/pre-restore-$STAMP/compactor /data/openwebui/ ; mv /data/forensics/pre-restore-$STAMP/webui.db* /data/openwebui/ ; ls -la /data/openwebui/
supervisorctl start compactor openwebui backup
```

**Success:** the final listing shows `compactor` and `webui.db` in
`/data/openwebui/`, and the three services come back `RUNNING`. You are back
where you started. Ask for help before trying again.

### Recover from a wiped / replaced volume

If the Network Volume itself was lost and you have an archive saved elsewhere
(copied off-pod): start a fresh pod with `WEBUI_DB_LOCAL=false` and the new
volume at `/data`, copy the archive into `/data/backups/`, then follow the
restore procedure above from step 1. Step 7 sets aside whatever the fresh boot
created, which is what you want.

> This is exactly why off-volume backups matter — if the only copy was on
> the lost volume, there's nothing to restore. Until off-volume DR ships,
> periodically copy `/data/backups/`'s newest archive somewhere off the pod.

---

## Logs: text vs JSON

`COMPACTOR_LOG_FORMAT` controls the compactor + sidecar (selftest, backup)
log format:
- `text` (default) — human-readable, what `tail -f` has always shown.
- `json` — one JSON object per line (`ts`/`level`/`logger`/`message`, plus
  `exc` on errors). Set this when shipping logs to an aggregator or when you
  want to `jq` them:
  ```bash
  tail -f /data/logs/compactor.log | jq 'select(.level=="WARNING")'   # jq on your own machine; it is not in the image
  ```

## Failure alerts (optional)

Set `COMPACTOR_ALERT_WEBHOOK` to a URL and the **boot self-test** and the
**backup daemon** will POST a JSON alert there on failure — so you hear
about a broken deploy or a failed backup before a user does. Off by default.

The payload carries structured fields (`service`, `status`, `detail`,
`host`, `ts`) plus `text`/`content` so the same URL works with a Slack or
Discord incoming webhook, or any generic receiver. Alerting is best-effort:
a slow or broken webhook is logged and ignored, never blocking the job.

```bash
# quick test: point it at a request-bin style URL, then force a failing
# self-test (e.g. with vLLM stopped) and watch the POST arrive
COMPACTOR_ALERT_WEBHOOK=https://hooks.example/zions ...
```

## Rolling back a bad release

Image tags are immutable snapshots (see
[README → Image tags](README.md#image-tags)). To roll back:

1. **If the History cap filter is installed, set its `max_turns` valve to 0
   first** (OpenWebUI → Admin Panel → Functions → History cap → Valves), and
   leave it at 0 until the newer image is back AND has served one uncapped
   message. Rolling back with the cap on leaves a permanent, unlogged hole in
   her summary hierarchy (hostile review of v3.1.7, reviewer C, F5). See
   RUNBOOK_MEMORY_IDENTITY.md "Rolling the IMAGE back".
2. In the RunPod template, change **Container Image** to the last-good tag
   (from v3.1.9 that is `angreg/zions-light-ai:v3.1.6.1-cu12`, the image the
   pod ran before). **Leave `WEBUI_DB_LOCAL=false` exactly as it is, spelled
   `false`** — v3.1.9 also accepts `0`/`no`/`off` and `1`/`yes`/`True`, but
   the older images read only the exact word `true` as true, so any other
   spelling can mean different things on the two sides of a rollback. See
   RUNPOD_DEPLOY.md "WEBUI_DB_LOCAL".
   A rollback is an image change only; do not restore a backup as part of it.
3. Restart the pod. The Network Volume (and all memory) is unaffected —
   only the code image changes. The `X-Conversation-Id` connection header
   lives in OpenWebUI's database, not the image, so it stays set.

The `:latest` tag is only ever promoted to a release that has passed its
boot self-test on a real pod, so `:latest` should always be safe; pinned
tags exist for deterministic rollback regardless.

---

## Escalation checklist (the 2am version)

1. `supervisorctl status` — what's actually down?
2. `curl -s localhost:8080/health/full | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d['status_reasons'])"`
   — ok / degraded / down? Then read "Reading /health/full" (above): `ok` does
   not cover stopped backups.
3. `tail -50` the log of whatever's down.
4. If data looks lost → **restore from `/data/backups`** (above) before
   doing anything else destructive.
5. If a release is the suspect → **roll back the image tag** (above).
6. If the volume is gone → fresh pod + restore from an off-pod archive copy.

---

## Shadow deploy: test a build on the pod without touching the live compactor

**Yes, this is safe, and it is the right way to validate a build before the
`:latest` flip.** A second compactor process on the same pod, on its own port,
with its own storage root, sharing the vLLM that is already running. The live
compactor is never stopped, never reconfigured, and never reads or writes the
shadow's memory.

**The one thing to be deliberate about is the GPU.** The shadow talks to the
SAME vLLM, so every generation it triggers — replies, fact extraction, dedup,
summary rollups — queues behind and alongside her real traffic. That is
usually fine for a handful of curl requests and is NOT fine for a soak. Do
this when she is not mid-conversation, and keep it short.

### 1. Get the code onto the pod, beside the live copy

```bash
# The live compactor runs from /opt/compactor. Leave it alone.
git -C /opt/compactor-shadow pull 2>/dev/null || \
  git clone -b fix/v3.1.4 <repo-url> /opt/compactor-shadow
```

### 2. Start it on its own port, with its own storage

```bash
# 8081, not $COMPACTOR_PORT. A SEPARATE storage root is the load-bearing
# part: point this at /data/openwebui/compactor and the shadow will write
# facts, summaries and episodic entries into her live memory.
COMPACTOR_STORAGE_ROOT=/data/shadow-compactor \
VLLM_URL=http://127.0.0.1:8000 \
MODEL_REPO="$MODEL_REPO" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
COMPACTOR_GENERATION_RESERVE="$COMPACTOR_GENERATION_RESERVE" \
/opt/compactor-venv/bin/uvicorn main:app \
    --app-dir /opt/compactor-shadow/compactor \
    --host 127.0.0.1 --port 8081 --log-level info \
    > /tmp/shadow.log 2>&1 &
```

It reuses `/opt/compactor-venv` on purpose: the point of a shadow deploy is to
test the CODE against the venv that is actually installed. If the branch adds
a dependency, this is where you find out — and finding out here is the whole
idea.

### 3. Hit it with curl

`--host 127.0.0.1` keeps it off the public proxy, and admin endpoints are
gated on the caller being loopback, so run these ON the pod.

```bash
# Health, and whether it thinks vLLM and storage are reachable
curl -s localhost:8081/health/full | python3 -m json.tool

# A real exchange. X-Conversation-Id keeps it out of her conversations.
curl -s localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Conversation-Id: shadow-smoke-1' \
  -d '{"model":"'"$MODEL_REPO"'","stream":false,
       "messages":[{"role":"user","content":"Say hello in one sentence."}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"][:400])'

# Did the memory tail fire, and under what outcome?
curl -s localhost:8081/health/full \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["memory_tail"])'

# What it stored
curl -s localhost:8081/admin/conversations/shadow-smoke-1/facts | python3 -m json.tool
```

### 4. Stop it and clear up

```bash
pkill -f "port 8081"
rm -rf /data/shadow-compactor      # the shadow's memory, and only the shadow's
```

### What a shadow deploy can and cannot tell you

It answers: does this build boot on the real image, against the real venv,
with the real env file? Does an exchange complete end to end? Does the memory
tail fire and store? Do the admin endpoints answer? Those are exactly the
failures that have historically shipped — v3.1.4 went down on a module missing
from the Dockerfile, and one env typo used to stop the boot.

It does not answer: anything about her actual conversation, whose state lives
in a storage root this process cannot see. To test against real data, restore
a backup into the shadow's root first — never point the shadow at the live one.
