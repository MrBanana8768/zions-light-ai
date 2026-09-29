# Every path the pod reads or writes under `/data` (v3.1.9.10 audit)

**Why this exists.** The v3.1.9.10 plan runs a NEW pod beside the OLD one on the SAME network volume
(`/data`, for the model weights). The old pod stays live until the cutover and is the rollback after it.
The guarantee we need: **nothing the new pod runs can write, prune, lock or delete a file the old pod
owns.** This is the audit that backs that guarantee, the exact env block that delivers it, and the
boot proof that checks it.

Audited on the v3.1.9.9 code (`ccd98c6`, image `local/zla:v3199-rc`): `entrypoint.sh`, `supervisord.conf`
(every program), every `compactor/*.py` module, `ops/`, `scripts/`, OpenWebUI 0.11.4's own `env.py` /
`config.py` / `storage/provider.py` (read from the image), vLLM's cache handling. Kept true by
`compactor/test_path_isolation.py`, which fails on any NEW hardcoded `/data` literal.

**Sharing rule.** Two live pods may share a path only if NO program on EITHER pod writes it. Read-only model
weights: safe. Anything with a writer (a log, a cache, a backup directory that prunes, a marker file, a
lock, a database): not safe.

## Findings, fixed in this release (each has an env override; defaults unchanged when the env is unset)

| # | Was | Danger on a second pod | Fix |
|---|-----|------------------------|-----|
| F1 | `memory.py` / `degrade.py`: store default hardcoded `/data/openwebui/compactor` (not `DATA_DIR`) | `DATA_DIR` redirected but `COMPACTOR_STORAGE_ROOT` forgotten: the new compactor **wrote her memory into the old pod's store** while `backup.py` (which followed `DATA_DIR`) refused to back up an empty directory. Reproduced (control run below). | default now follows `DATA_DIR` |
| F2 | `uploads_backup.py`: mirror default hardcoded `/data/backups/uploads` | mirrored into, and **expired files from, the old pod's mirror** | default follows `COMPACTOR_BACKUP_DIR` |
| F3 | `COMPACTOR_BACKUP_DIR` default `/data/backups` | the backup daemon **prunes** archives in that directory: the new pod would delete the old pod's history. Reproduced: the new pod read the old archives. | env override existed; now in the block; the pod-migration runbook forbids leaving it unset |
| F4 | `wordban_runtime.py` / `word-ban-deploy.py`: `/data/zl-ops/word-ban/active.json` hardcoded | the deploy tool could overwrite the old pod's deployed grammar | default follows `ZL_HOME` |
| F5 | `entrypoint.sh`: `/data/vllm-compile-cache` hardcoded (mkdir + symlink) | two writers on one compile cache | new `VLLM_COMPILE_CACHE` (default unchanged; `clean-models.sh` already used that name) |
| F6 | `webuidb.py`, `health.py`: snapshot default hardcoded `/data/openwebui/webui.db` | **the worst one.** A shell without the pod's env running `webuidb.py --sync-once --force` (the documented "final sync") would **publish the new pod's local database over the old pod's live one** | default follows `DATA_DIR`; and see F9 |
| F7 | `ops/zlops/podenv.py`: `/data/forensics`, `/data/ssh`, `/data/shadow-compactor` hardcoded | `zl shadow down` does `rmtree(SHADOW_STORE)` | follow `WEBUI_DB_QUARANTINE`, `SETUP_SSHD_HOST_KEY_STORE`, new `ZL_SHADOW_STORE` |
| F8 | operator scripts (`backfill-records`, `import-history`, `clean-decoration`, `recover-webui-db`, `fix-stale-unfinished`, `compare-live-to-backup`, `uploads-backup`, `switch-webui-db-to-local`, `setup-sshd`, `_webui_live_path`) defaulted to `/data/openwebui/...` | a writer run without every path flag aims at the old pod's files | new `scripts/_podpaths.py`: own env, then supervisord's, then the same default |
| F9 | A RunPod Web Terminal shell may not carry the pod's env (see `podenv.py`'s own note) | every compactor CLI (`backup.py --once`, `webuidb.py --sync-once`, `uploads_backup`, anything importing `memory.py`) silently resolves to the built-in defaults, i.e. the OLD pod's paths | `envcfg.adopt_pod_env()`: when pid 1 is supervisord, fill the **path-and-gate names only** that the shell lacks from `/proc/1/environ` (never a secret, never a `WEBUI_DB_ALLOW_*` flag, never over a value the shell has). Proved in a bare `env -i` shell below. |
| F10 | uploads: OpenWebUI stores an **absolute** `file.path` (`<DATA_DIR>/uploads/<id>_<name>`) per file and serves it from that exact path | a DB copied unchanged makes the new pod **read the old pod's uploads** (and break when the old tree is cleaned up) | `pod-migrate.py` rewrites `file.path` in the target copy only (`<from>/uploads/` becomes `<to>/uploads/`); `verify` checks it |

Not changeable, and harmless: `entrypoint.sh` does `touch /data/.write-test; rm /data/.write-test` (a
"is the volume writable" check, at the volume root). Both pods do it at boot; it changes the volume root
directory's mtime and nothing else.

## The table

"Follows `DATA_DIR`" = the default is derived from `DATA_DIR`, so redirecting `DATA_DIR` alone moves it.
"Shared?" = may the old and new pod both use this exact path while both are live.

### OpenWebUI (`[program:openwebui]`, `DATA_DIR` passed explicitly by supervisord)

| Path | Env var (default) | Follows `DATA_DIR` | Shared? | New pod |
|------|-------------------|-----|---------|---------|
| snapshot DB `webui.db` (+ `.synced_at`, `.sync-tmp`, `-journal`) | `WEBUI_SNAPSHOT_DB` (`$DATA_DIR/webui.db`; exported by entrypoint) | yes | **NO** (sync daemon writes; live DB when `WEBUI_DB_LOCAL=false`) | `/data/v10/openwebui/webui.db` |
| live local DB, `.sync.lock`, `.empty-start`, `.backup-staging/`, `.migrate-*` | `WEBUI_LOCAL_DB` (`/var/lib/openwebui/webui.db`), `WEBUI_DB_EMPTY_START_MARKER`, `COMPACTOR_BACKUP_LOCAL_STAGING_DIR` | no (container-local overlay) | n/a: not on the volume | unchanged |
| `DATABASE_URL` | derived by entrypoint from the gate | | n/a | do **not** set (entrypoint refuses a disagreeing value) |
| `uploads/` | none (OpenWebUI `UPLOAD_DIR = DATA_DIR/uploads`); `COMPACTOR_UPLOADS_DIR` for the compactor's mirror source | yes | **NO** (writer; absolute path in the DB, F10) | `/data/v10/openwebui/uploads` |
| `vector_db/chroma.sqlite3` | none (`DATA_DIR/vector_db`) | yes | **NO** (writer) | `/data/v10/openwebui/vector_db` |
| `cache/` (audio, image, embedding model when RAG runs) | none (`DATA_DIR/cache`) | yes | **NO** (writer) | `/data/v10/openwebui/cache` (`--with-cache` copies it) |
| `audit.log`, `l.data` | `AUDIT_LOGS_FILE_PATH`, `LICENSE_BLOB_PATH` (`DATA_DIR/...`; not created in the observed boots) | yes | NO if ever enabled | follows `DATA_DIR` |
| **`.webui_secret_key`** | none: `Path.cwd()/.webui_secret_key` = `/app/.webui_secret_key` (**not** under `DATA_DIR`), read **only when `WEBUI_SECRET_KEY` is unset** | no | n/a | never used: entrypoint always exports `WEBUI_SECRET_KEY` |
| package files (`static/`, frontend build) | `STATIC_DIR`, `FRONTEND_BUILD_DIR` | no (image) | n/a | unchanged |

**The secret key, decided.** In this image the effective secret is the `WEBUI_SECRET_KEY` **env value**
(entrypoint generates a random one per boot if the row is empty). A key **file** under `DATA_DIR` is
never read, so the seed does **not** copy one by default (`--with-secret-key` exists for a nonstandard
setup; it copies `<from>/.webui_secret_key` at mode 0600 if present). Consequences:
* To carry her login sessions over, the new pod's `WEBUI_SECRET_KEY` must equal the old pod's (same RunPod
  secret value; never print it).
* If the old pod has no such row (a random key each boot), there is nothing to carry over.
* Either way she **logs in again** at `ai.revelationsaints.com`: the cookie belongs to the host name.

### Compactor (`[program:compactor]`, `selftest`, `backup`, and every CLI in `compactor/`)

| Path | Env var (default) | Follows `DATA_DIR` | Shared? | New pod |
|------|-------------------|-----|---------|---------|
| store: `facts/ summaries/ chromadb/ personas/ quarantine/` (+ `journal/` in the v3.1.9.10 build) | `COMPACTOR_STORAGE_ROOT` (`$DATA_DIR/compactor`) | **yes since F1** | **NO** | `/data/v10/openwebui/compactor` |
| control root (`wipes/`: the forget ledger) | `COMPACTOR_CONTROL_ROOT` (`<store parent>/compactor-control`) | yes, via the store | **NO** (two writers of one ledger = a forget applied on one pod only) | `/data/v10/openwebui/compactor-control` |
| pending root (`ledger-pending/`) | `COMPACTOR_PENDING_ROOT` (`<store parent>/compactor-pending`) | yes, via the store | **NO** (the new pod's boot consumes it: reproduced) | `/data/v10/openwebui/compactor-pending` |
| backups: archives, `.partial` staging | `COMPACTOR_BACKUP_DIR` (`/data/backups`) | **no** | **NO** (the daemon prunes here) | `/data/v10/backups` |
| uploads mirror (`manifest.json`, `objects/`, `status.json`, lock) | `COMPACTOR_UPLOADS_BACKUP_DIR` (`<backup dir>/uploads`; **F2**) | via the backup dir | **NO** (expires files) | `/data/v10/backups/uploads` |
| word-ban file `active.json` (+ previous) | `COMPACTOR_WORDBAN_FILE` (`$ZL_HOME/word-ban/active.json`; **F4**) | via `ZL_HOME` | compactor only reads; `word-ban-deploy.py` writes: **NO** | `/data/v10/zl-ops/word-ban/active.json` |
| forensics: `restore-*.inprogress` markers, set-aside DB copies | `WEBUI_DB_QUARANTINE` (`/data/forensics`); scripts: `WEBUI_DB_FORENSICS` (falls back to it) | **no** | **NO** (a marker left by one pod makes the OTHER pod's boot refuse) | `/data/v10/forensics` |
| backup source DB override | `COMPACTOR_BACKUP_WEBUI_DB` (unset: the live DB per `DATABASE_URL` / the gate) | n/a | leave unset (a value here would aim the backup at a specific file) | unset |
| free-space / quota walk | `COMPACTOR_DATA_VOLUME_ROOT` (`DATA_DIR.parent`) | yes | read-only walk | follows (`/data/v10`) |
| tokenizer files | `HF_HOME` (`/data/models`) | no | read-only | shared, offline |
| fastembed model | `FASTEMBED_CACHE_PATH` (`/opt/embeddings`, baked) | no | image-local | unchanged |

### supervisord programs, entrypoint, vLLM, ops

| Path | Env var (default) | Shared? | New pod |
|------|-------------------|---------|---------|
| all program logs, `boot.log`, `supervisord.log`, `process-events*.log` (vllm, compactor, openwebui, webuidb-sync, backup, selftest, stt, tts, cloudflared, zl-ops, the listener) | `LOG_DIR` (`/data/logs`) | **NO** (two supervisords rotate the same files; the boot marker interleaves) | `/data/v10/logs` |
| `[program:vllm]` weights | `HF_HOME` (`/data/models`) | **YES, read-only, only offline**: `COMPACTOR_FORCE_OFFLINE=true` pins `HF_HUB_OFFLINE`. Online, a moved `refs/main` would download and change the weights the old pod loads on its next start | shared, offline |
| torch.compile cache (`/root/.cache/vllm` symlink) | `VLLM_COMPILE_CACHE` (`/data/vllm-compile-cache`; **F5**) | **NO** (writes on a miss) | `/data/v10/vllm-compile-cache` (one cold start, 60-120 s of graph capture per the entrypoint's own note) |
| xgrammar cache, torch inductor | memory, `/tmp` | image-local | unchanged |
| `[program:webuidb-sync]` | reads local, writes the snapshot | see snapshot row | follows `WEBUI_SNAPSHOT_DB` |
| `[program:stt]`, `[program:tts]` | models baked in `/opt` | no `/data` path | unchanged |
| `[program:cloudflared]` | token in env; state in `/tmp` | no `/data` path. **One tunnel token must live on ONE pod**: Cloudflare balances between connectors of a tunnel | token only on the new pod |
| `[program:selftest]` | writes probe files through the compactor into the store / control root | follows those | follows |
| `[program:zl-ops]` (`ops/bootstrap.sh --boot`): `audit.log`, `bundles/`, saved bootstrap and ref | `ZL_HOME` (`/data/zl-ops`) | **NO** | `/data/v10/zl-ops` |
| `zl shadow` store (`shadow down` does `rmtree`) | `ZL_SHADOW_STORE` (`/data/shadow-compactor`; **F7**) | **NO** | `/data/v10/shadow-compactor` |
| sshd host keys (`zl sshd`, `setup-sshd.py`) | `SETUP_SSHD_HOST_KEY_STORE` (`/data/ssh`) | avoid (a shared SSH identity) | `/data/v10/ssh` |
| `/data/.write-test` (entrypoint) | fixed | transient, at the volume root | harmless |
| `/data/scripts/` (operator convenience copies) | none: no program reads it | n/a | keep the new pod's in `/data/v10/scripts` |
| `clean-models.sh` | `HF_HOME`, `VLLM_COMPILE_CACHE` | **it DELETES cached models**: never run it on the new pod while the old pod may be rolled back to | do not run |

## The exact env block for the new pod

Every row below is a **new-pod-only** template row. Everything else stays as the old pod's template has it,
except the rows in "must NOT be set". Values are literal.

```
# --- data: nothing here may equal an old-pod path ---
DATA_DIR=/data/v10/openwebui
LOG_DIR=/data/v10/logs
ZL_HOME=/data/v10/zl-ops
COMPACTOR_STORAGE_ROOT=/data/v10/openwebui/compactor
COMPACTOR_CONTROL_ROOT=/data/v10/openwebui/compactor-control
COMPACTOR_PENDING_ROOT=/data/v10/openwebui/compactor-pending
COMPACTOR_BACKUP_DIR=/data/v10/backups
COMPACTOR_UPLOADS_BACKUP_DIR=/data/v10/backups/uploads
COMPACTOR_WORDBAN_FILE=/data/v10/zl-ops/word-ban/active.json
WEBUI_SNAPSHOT_DB=/data/v10/openwebui/webui.db
WEBUI_DB_QUARANTINE=/data/v10/forensics
WEBUI_DB_FORENSICS=/data/v10/forensics
SETUP_SSHD_HOST_KEY_STORE=/data/v10/ssh
VLLM_COMPILE_CACHE=/data/v10/vllm-compile-cache
ZL_SHADOW_STORE=/data/v10/shadow-compactor

# --- the shared model weights: read-only, offline ---
COMPACTOR_FORCE_OFFLINE=true

# --- the migration settings ---
WEBUI_DB_LOCAL=true
WEBUI_DB_SYNC_INTERVAL_S=120
COMPACTOR_BACKFILL_MAX_ATTEMPTS=1

# --- secrets and URL (RunPod secrets; never print them) ---
WEBUI_SECRET_KEY=<the old pod's value, if it has one; see "The secret key, decided">
CLOUDFLARE_TUNNEL_TOKEN=<the tunnel secret; ONLY on this pod>
WEBUI_URL=https://ai.revelationsaints.com
```

**Must NOT be set on the new pod:** `DATABASE_URL`, `WEBUI_LOCAL_DB`, any `WEBUI_DB_ALLOW_*`,
`COMPACTOR_INJECTION_BUDGET_FRACTION` (delete a `.6` row: it pins the fraction below the image default),
and **no `CLOUDFLARE_TUNNEL_TOKEN` on the old pod**. `COMPACTOR_DATA_VOLUME_ROOT`,
`COMPACTOR_UPLOADS_DIR` and the snapshot/control/pending defaults follow `DATA_DIR`, so they are not needed.

With the block set, `envcfg.adopt_pod_env()` also makes a Web Terminal shell without the env resolve to the
same paths (proved below).

## The boot proof (synthetic data, real image)

Harness: `~/zl-ops/v31910/partB/harness/` (stub vLLM on :8000, no GPU, no real data). An "old pod" tree
was created by really booting `local/zla:v3199-rc` with prod-like env (`WEBUI_DB_LOCAL=false`), creating an
OpenWebUI admin, a chat, a real file upload, compactor traffic; then 69 more synthetic entries were added
across `/data` (backups, forensics, scripts, ssh, shadow, `zl-ops/word-ban`, compile cache, a stray
`.webui_secret_key`, a stale `webui.db.bak-*`, uploads, facts, summaries, personas, a pending sidecar, fake
model weights). Every file under `/data` (except `/data/v10`) was hashed (sha256 + size + mtime + mode; every
directory's mtime and entry count).

**Negative control (the v3.1.9.9 image, only `DATA_DIR` + `WEBUI_DB_LOCAL=true` set, the mistake F1 describes):**
`ADDED=7 REMOVED=1 CHANGED=27`: the old pod's logs, chroma index, facts/summaries, control ledger and pending
sidecar all changed or were consumed (`~/zl-ops/v31910/partB/control-run-diff.txt`). And a bare shell on that
image resolved `memory/backup/uploads/webuidb/wordban` to `/data/openwebui/...`, `/data/backups`, `/data/forensics`
(`control-bare-shell-v3199.txt`).

**Positive run (`local/zla:v31910-mig` = the fixed code, the env block above), PID 1 = the real entrypoint:**
first boot, `pod-migrate seed` (dry run and apply), `verify`, services started, a synthetic OpenWebUI chat +
upload, compactor chats, a manual `backup.py --once`, the uploads-mirror cycle, `webuidb.py --sync-once --force`
(the last three from a **bare `env -i` shell**), a second boot from the snapshot, `cutover` (with a stale
local DB holding a test chat), services restarted, a new chat, sync publishing normally, container stopped
with `docker stop -t 90` (final sync):

```
before entries=104 after entries=104
ADDED=0 REMOVED=0 CHANGED=1
  CHANGED ./ [mtime_ns]          <- the volume root directory itself (the /data/.write-test and mkdir /data/v10)
```

**Files changed under the old tree: 0** (`isolation-final.txt`, `isolation-after-run.txt`).
Everything the new pod wrote is under `/data/v10` (3.1 MB). A second cutover with a changed old tree
(new message rows, a new upload, a deleted upload, new and removed facts) also verified clean.

Re-run: `~/zl-ops/v31910/partB/harness/make_old_volume.sh`, then the steps in `harness/`.
