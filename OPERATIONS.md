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
  ones.

### Memory looks wrong for one conversation
See [USER_GUIDE.md](USER_GUIDE.md). Quick: `/why` in the chat,
`/list-facts`, `/forget <substring>`, or full reset
`curl -X DELETE localhost:8080/admin/conversations/<id>/facts`.

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
