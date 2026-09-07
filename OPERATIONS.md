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
curl -s http://localhost:8080/health/full | jq

# Process status
supervisorctl status
```

`/health/full` returns one of:
- `"status": "ok"` — everything works.
- `"status": "degraded"` — storage fine, **vLLM unreachable** (loading,
  crashed, or restarting). Chat is down; memory/admin endpoints still work.
  HTTP 200 (the container is intentionally *not* killed — supervisord can
  restart vLLM independently).
- `"status": "down"` — **storage broken** (`/data` not writable). HTTP 503.
  Nothing useful is possible; the container should be replaced.

Deeper, on demand:
```bash
curl -s http://localhost:8080/admin/selftest | jq   # real chat round-trip + facts I/O
cat /var/log/supervisor/selftest.log                # the boot self-test result
```

---

## Reading the logs

```bash
tail -f /var/log/supervisor/vllm.log        # inference engine
tail -f /var/log/supervisor/compactor.log   # memory + compaction + requests
tail -f /var/log/supervisor/openwebui.log   # frontend
tail -f /var/log/supervisor/selftest.log    # boot self-test (one-shot)
tail -f /var/log/supervisor/backup.log      # backup daemon
```

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

1. Read that service's log: `tail -100 /var/log/supervisor/<name>.log`.
2. Fix the root cause (see below for vLLM).
3. Clear FATAL and retry: `supervisorctl start <name>` (or
   `supervisorctl restart <name>`).

The background-work pool and disk-pressure state are visible in
`/health/full` (`background_work`, `memory_writes`); a FATAL *vLLM* shows as
`status: degraded` there, and a FATAL *compactor* makes `/health/full`
itself unreachable (so "curl refused on :8080" == compactor down).

### vLLM won't start / keeps restarting
1. `tail -100 /var/log/supervisor/vllm.log` — look for the real error.
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
df -h /data
du -sh /data/* | sort -h
```
- Backups (`/data/backups`) self-prune to `COMPACTOR_BACKUP_RETAIN` (7) —
  lower it or `COMPACTOR_BACKUP_INTERVAL_HOURS` if they're the bulk.
- The backup daemon **refuses to run** below `COMPACTOR_BACKUP_MIN_FREE_MB`
  (500 MB) rather than filling the disk — you'll see that in `backup.log`.
- Model weights under `/data/models` are the usual space hog; remove unused
  ones with `/opt/clean-models.sh` (see
  [Cleaning up old model weights](#cleaning-up-old-model-weights-on-the-volume)).

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

Since v3.1.8 `/health/full` does this itself and degrades on it:

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
journal behind. **v3.1.6's `webuidb.py` moves the live database to the pod's
local overlay and syncs to `/data`, which removes the cause** — but a pod
running an older image does not have it. Until that ships, expect this on any
volume hiccup, and keep the eight-byte check to hand.

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
curl -s http://localhost:8080/admin/backups | jq      # list + latest summary
ls -lh /data/backups/
cat /var/log/supervisor/backup.log
```

### Make / verify a backup on demand
```bash
curl -X POST http://localhost:8080/admin/backups | jq          # run one now
curl -s http://localhost:8080/admin/backups/verify | jq        # verify newest
# Or via the CLI:
/opt/compactor-venv/bin/python /opt/compactor/backup.py --once
/opt/compactor-venv/bin/python /opt/compactor/backup.py --list
/opt/compactor-venv/bin/python /opt/compactor/backup.py --verify /data/backups/<archive>.tar.gz
```

### 🔥 Restore from a backup (recover lost/corrupted memory)

Restore is **destructive** — it overwrites the live `webui.db` and the
compactor store with the archive's contents. It refuses to run on an archive
that doesn't verify.

```bash
# 1. Pick an archive (newest last)
ls -1t /data/backups/

# 2. Stop the writers so nothing races the restore
supervisorctl stop openwebui compactor backup

# 3. Restore (the --yes confirms the destructive op)
/opt/compactor-venv/bin/python /opt/compactor/backup.py \
    --restore /data/backups/zions-backup-YYYYMMDD-HHMMSS.tar.gz --yes

# 4. Bring the writers back
supervisorctl start compactor openwebui backup

# 5. Confirm
curl -s http://localhost:8080/health/full | jq '.stats'
```

### Recover from a wiped / replaced volume
If the Network Volume itself was lost and you have an archive saved
elsewhere (copied off-pod):
```bash
# On a fresh pod with the new volume mounted at /data:
mkdir -p /data/backups
# copy your saved archive into /data/backups/ first, then:
supervisorctl stop openwebui compactor backup
/opt/compactor-venv/bin/python /opt/compactor/backup.py \
    --restore /data/backups/<archive>.tar.gz --yes
supervisorctl start compactor openwebui backup
```
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
  tail -f /var/log/supervisor/compactor.log | jq 'select(.level=="WARNING")'
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

1. In the RunPod template, change **Container Image** to the last-good tag
   (e.g. `angreg/zions-light-ai:v2.0`, or the V1 escape hatch `:1.9.6`).
2. Restart the pod. The Network Volume (and all memory) is unaffected —
   only the code image changes.

The `:latest` tag is only ever promoted to a release that has passed its
boot self-test on a real pod, so `:latest` should always be safe; pinned
tags exist for deterministic rollback regardless.

---

## Escalation checklist (the 2am version)

1. `supervisorctl status` — what's actually down?
2. `curl -s localhost:8080/health/full | jq .status` — ok / degraded / down?
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
