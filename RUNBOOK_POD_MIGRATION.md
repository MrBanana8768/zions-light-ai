# Moving to a NEW pod: v3.1.9.10, old pod to new pod

**What this is.** Prod today is the OLD pod (`angreg/zions-light-ai:v3.1.9-cu12`, `WEBUI_DB_LOCAL=false`, live DB
on the network volume at `/data/openwebui/webui.db`, her store at `/data/openwebui/compactor`, uploads at
`/data/openwebui/uploads`, logs and backups under `/data`). You start a NEW pod on
`angreg/zions-light-ai:v3.1.9.10-cu12`, attach the SAME network volume (for the model weights at `/data/models`),
with `WEBUI_DB_LOCAL=true` and all of ITS data in a separate folder, `/data/v10`. Seed, test, then at a
quiet moment freeze the old pod, resync, switch her over. **The old pod stays stopped afterwards, and is the
rollback.**

**Why it is safe.** Read `docs/PATHS-v31910.md`. Every path the new pod writes is redirected by the env block
below, and the boot proof shows **0 files changed under the old tree**. The tool (`pod-migrate.py`) never
writes the old tree (it opens the old database `mode=ro`, only reads its files) unless you run the reverse
cutover in "Rollback". Nothing here needs a pod restart on the old pod.

**The tool** ships in the image: `/opt/zl-repo/scripts/pod-migrate.py` (stdlib only). Every command is a
dry run until you add `--apply`. Exit codes: `0` done (dry run: the same command with `--apply` would
proceed), `1` a step failed or `verify` found a mismatch, `2` refused before touching anything. Its output is
counts only: no upload names, no conversation ids, no content.

Start every Web Terminal session on the new pod with:

```bash
PY=/opt/compactor-venv/bin/python; T="$PY /opt/zl-repo/scripts/pod-migrate.py"
OLD=/data/openwebui; NEW=/data/v10/openwebui
```

The tool finds this pod's `DATA_DIR` from supervisord even in a shell without the pod's environment, and it
**refuses to write any tree that is not this pod's own** (so a swapped `--from`/`--to` cannot touch the old pod).

---

## 0. Before you start (about 10 minutes, nothing changes)

1. **Free disk on the new pod** (it needs room for the live DB, a scratch copy while the tool runs, and the
   moved-aside copy of the previous local DB): at least **3 GB free** on `/var/lib/openwebui`
   (`df -h /var/lib/openwebui`). The tool refuses if there is not enough.
2. **The secret key** (her login sessions). On the OLD pod's terminal, without printing the value:

   ```bash
   tr '\0' '\n' < /proc/1/environ | grep -c '^WEBUI_SECRET_KEY=.'
   ```

   `1`: the old pod has a fixed key. Put the **same value** in the new pod's `WEBUI_SECRET_KEY` secret, and her
   sessions survive the move. `0`: it generates a random key each boot, there is nothing to carry over. In both
   cases she **logs in again** at `ai.revelationsaints.com` (a new host name has new cookies). The seed does
   not copy a key file: this image never reads one (`docs/PATHS-v31910.md`, "The secret key, decided").
3. **Where the old pod's uploads and DB paths are** (nothing to do, only to know): OpenWebUI stores an
   absolute path per uploaded file. The tool rewrites those paths in the NEW copy of the DB from
   `/data/openwebui/uploads/` to `/data/v10/openwebui/uploads/`, so the new pod never reads the old pod's files.
4. **Do not put `CLOUDFLARE_TUNNEL_TOKEN` on the old pod.** A tunnel balances between all connectors that hold
   the same token. It goes on the NEW pod only.

## 1. Create the new pod

RunPod: a new pod (do not touch the old one), the **same network volume mounted at `/data`**, image
`angreg/zions-light-ai:v3.1.9.10-cu12`, the old pod's template rows copied (GPU, ports, `MODEL_REPO`, ...),
**plus** these rows (they are literal; `docs/PATHS-v31910.md` explains each and lists the proof):

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

# --- secrets and URL ---
WEBUI_SECRET_KEY=<the old pod's value, or leave the row out; see step 0.2>
WEBUI_URL=https://ai.revelationsaints.com
CLOUDFLARE_TUNNEL_TOKEN=<the tunnel secret>   # add it AFTER the seed, see step 2
```

**Must NOT exist on the new pod:** `DATABASE_URL`, `WEBUI_LOCAL_DB`, any `WEBUI_DB_ALLOW_*`, and
**`COMPACTOR_INJECTION_BUDGET_FRACTION`** (delete a `.6` row: it pins the fraction below the image's 0.75).
`COMPACTOR_BACKFILL_MAX_ATTEMPTS=1` keeps her open backfill record (attempts=1) from restarting a
full backfill on the new image. Keep the old pod's other rows (backup interval, `COMPACTOR_MAX_RETAINED_IMAGES`,
`COMPACTOR_GENERATION_RESERVE`, ...) as they are, **except** `COMPACTOR_BACKUP_RETAIN`: set it to `3` on the new pod
(the old `28` is a 28-hour floor that swallows the 3-hour tier).

Then check the pod's boot log (RunPod Logs tab) and, in the Web Terminal:

```bash
tr '\0' '\n' < /proc/1/environ | grep -E '^(DATA_DIR|LOG_DIR|COMPACTOR_BACKUP_DIR|WEBUI_DB_LOCAL|COMPACTOR_STORAGE_ROOT)='
```

**Expect** the `/data/v10/...` values and `WEBUI_DB_LOCAL=true`. If any path is missing, **stop and fix the row
before going on**: a missing row is the one way the new pod could aim at the old pod's files.

## 2. First boot order: an empty first boot, then stop, seed, restart. (The decision, and why.)

**The choice.** The seed needs the new pod (it has the tool and sees both trees on the volume), but OpenWebUI
must not run on an empty DB when real data is about to arrive. supervisord starts OpenWebUI unconditionally and
no env row changes that. The alternatives were a start-command override (a RunPod-template trick, not
testable here) or accepting one empty boot. **We accept the empty boot**, because it is harmless by
construction, measured:

* It happens entirely on the new pod: an empty schema on the pod's local disk. The old tree is not touched (boot
  proof).
* `webuidb-sync` **refuses to publish a 0-chat database** ("snapshot reports 0 chats; refusing to publish",
  seen in the proof), so no empty snapshot is written to the volume either.
* The seed then **replaces the local DB** (moves the empty one aside), so the empty schema never survives.

**Leave `CLOUDFLARE_TUNNEL_TOKEN` off this first boot.** An empty OpenWebUI has open sign-up (the first person to
sign up would become admin of an empty database that is about to be replaced). Add the token after the seed
(step 4).

In the Web Terminal, when the pod is up:

```bash
supervisorctl stop openwebui compactor webuidb-sync backup      # the empty first boot is over (vllm keeps running)
```

(`webuidb-sync` logs the "refusing to publish" line on the way out. Expected.)

## 3. Seed, verify (the old pod stays live; nothing on it changes)

Ask her not to chat while the database copy runs, if you can (a few minutes). The copy is the SQLite backup
API on a read-only connection; it takes a shared lock in small steps, and retries if the old pod writes.

```bash
$T seed --from $OLD --to $NEW            # the dry run: read the plan (counts only)
$T seed --from $OLD --to $NEW --apply
echo EXIT=$?                              # 0
$T verify --from $OLD --to $NEW
echo EXIT=$?                              # 0
```

**Expect** `pod-migrate verify: all checks passed (0 failing)`. What `verify` compares, counts only: integrity_check
on both databases; the row count of every table; users, chats, `chat_message` rows and `file` rows; her chat
(`ea1494ea-...`: the size of the chat JSON, its message count and its `chat_message` rows); every uploaded file's
path in the new DB (none may still point at the old tree); facts and summaries per store; the chroma document
counts (compactor store and OpenWebUI's `vector_db`); a sha of every file in every copied tree.

* While the old pod is live and she keeps chatting, `verify` will show differences after a while. That is normal;
  the cutover closes them. Re-running `seed --apply` is a safe incremental refresh (it copies only new and
  changed files; the manifest is `<to>/.migrate/seed.json`).
* If `seed` says the target "holds data and is not a seed", something already wrote real data into `/data/v10`:
  find out what before using `--force` (with `--force` nothing is deleted either; unknown files are moved aside).
* **Not copied on purpose** (counts only in the output): OpenWebUI's `cache/` (audio/image/embedding-model cache), old
  `webui.db.bak-*` files, its audit log. If she uses document search and the new pod is offline
  (`COMPACTOR_FORCE_OFFLINE=true`), add `--with-cache` to all three commands, or the embedding model has to be
  downloaded first (temporarily remove the offline row).
* The word-ban directory (`/data/zl-ops/word-ban`) is copied if it exists on the old pod. If the old pod does not
  have one (v3.1.9 has no file deploy), deploy the grammar to the new pod with `word-ban-deploy.py` before
  switching her (OPERATIONS.md); the compactor warms it up when the pod is idle.

Reclaim local disk once you are happy: the tool moved the empty first-boot DB aside.

```bash
ls -d /var/lib/openwebui/.migrate-aside-*        # the empty first-boot database and its lock file
rm -rf /var/lib/openwebui/.migrate-aside-*       # safe to delete: it is only the empty schema
```

Now start the new pod's services and test:

```bash
supervisorctl start compactor webuidb-sync openwebui backup
```

## 4. Test the new pod (you only, never her)

1. Add the tunnel: in the RunPod template add `CLOUDFLARE_TUNNEL_TOKEN` (secret) and `WEBUI_URL`, restart the new
   pod. **Nothing is lost by the restart**: the seeded snapshot is on the volume, the boot restores it to local disk.
   (See `RUNBOOK_CLOUDFLARE_TUNNEL.md` for the dashboard side: the public hostname `ai.revelationsaints.com`
   pointing at `http://localhost:3000`.)
2. Open the **pod's proxy URL** and then **`https://ai.revelationsaints.com`**. Log in (she has to log in again anyway).
3. Check: her chat and its history load; her uploaded images show (this proves the path rewrite); one message
   round-trips (the first reply after a cold start can take up to ~100 s: the tunnel note in the tunnel runbook).
4. **Admin Panel, Settings, General, WebUI URL: set it to `https://ai.revelationsaints.com`.** The seeded DB already
   holds a `webui.url` value (or none), and OpenWebUI prefers the saved value over the env row, so `WEBUI_URL` in
   the template does not change it. After you set it:

   ```bash
   $PY - <<'EOF'
   import sqlite3
   c = sqlite3.connect("file:/var/lib/openwebui/webui.db?mode=ro", uri=True)
   print(c.execute("SELECT value FROM config WHERE key='webui.url'").fetchall())
   EOF
   ```
5. **Everything you write on the new pod during testing is discarded at the cutover** (the target DB is replaced
   from the old pod's, the new pod's own test facts are moved aside). Do not test with anything you want to keep.
   Note that URL setting from step 4 is part of the DB: **set it again after the cutover** (the last step of
   section 5), or set it only then.
6. A first backup of the new pod's data, into its own directory:

   ```bash
   $PY /opt/compactor/backup.py --once; echo BACKUP_EXIT=$?
   A=$(ls -t /data/v10/backups/zions-backup-*.tar.gz | head -1); $PY /opt/compactor/backup.py --verify "$A"; echo VERIFY_EXIT=$?
   ```

   **Expect** `0` and `0`. The pod's backup daemon uses `/data/v10/backups`; its schedule is the one in the
   template (`COMPACTOR_BACKUP_INTERVAL_HOURS` and the retention rows you copied). **`/data/backups` (the old pod's)
   stays exactly as it is, as history**: nothing on the new pod reads or prunes it.

## 5. The cutover window

Pick a quiet time. **Tell her.** It is about 10 minutes.

1. **She closes the app on every device.**
2. **Freeze the OLD pod.** On the OLD pod's terminal:

   ```bash
   supervisorctl stop openwebui compactor backup
   supervisorctl status openwebui compactor backup       # all STOPPED
   ls -la /data/openwebui | grep webui.db                # ONE line: webui.db. No -journal, no -wal.
   ```

   If a `-journal` is there: **stop**, follow `RUNBOOK_DB_JOURNAL.md`. (vLLM may keep running on the old pod; it
   writes nothing that matters here.)
3. **On the NEW pod:** stop its writers, then dry-run, then apply:

   ```bash
   supervisorctl stop openwebui compactor webuidb-sync backup
   $T cutover --from $OLD --to $NEW --old-pod-stopped                 # dry run: refusal reasons, the plan, a 60 s quiet check
   $T cutover --from $OLD --to $NEW --old-pod-stopped --apply
   echo EXIT=$?                                                       # 0; it runs verify itself at the end
   ```

   What `cutover` insists on (it exits `2` and changes nothing if any of these fail):
   * `--old-pod-stopped`: your word that the OLD pod's openwebui, compactor and backup are stopped. The tool cannot
     see the other pod, so it also checks the evidence: no `-journal`/`-wal`/`-shm` beside the old `webui.db`, and
     the old tree (database, store, uploads) shows **no change for 60 seconds** (`--quiet-seconds`).
   * The NEW pod's `openwebui`, `compactor`, `webuidb-sync` and `backup` are stopped, and nothing listens on
     :3000 / :8080. Because the tool replaces this pod's data.
   * `--to` is this pod's own `DATA_DIR`, and it is a previous seed.

   What it does: a fresh, consistent copy of `webui.db` (its `file.path` rows rewritten to the new pod), then a delta
   of the store, uploads, `vector_db` and word-ban (new and changed files copied; files deleted on the old pod, and files
   the new pod wrote during testing, are **moved** to `/data/v10/openwebui/.migrate/removed-<stamp>/`, never deleted).
   If the old tree changes while it runs, it fails (exit 1) and says so: stop the old pod properly and re-run.
4. **The database on this pod is REPLACED, not merged.** This is the dangerous step, so it is spelled out. With
   `WEBUI_DB_LOCAL=true`, the pod's boot keeps a *healthy local* DB in preference to the snapshot, and
   `webuidb-sync` would then publish that local file (your test data) over the fresh snapshot. So the tool moves
   the local DB, its `-journal`/`-wal`/`-shm`, `.sync.lock` and `.empty-start` to
   `/var/lib/openwebui/.migrate-aside-<stamp>/`, places the fresh copy at `/var/lib/openwebui/webui.db` AND at
   `/data/v10/openwebui/webui.db`, and gives both the same modification time. Check:

   ```bash
   $PY /opt/compactor/webuidb.py --status
   ```

   **Expect** `local` and `snapshot` rows with the same `chats=` and `content=`, both `quick_check=ok`.
5. **Start the new pod's services and check:**

   ```bash
   supervisorctl start compactor webuidb-sync openwebui backup
   sleep 60; supervisorctl status compactor webuidb-sync openwebui backup     # all RUNNING
   curl -s http://127.0.0.1:8080/health/full | $PY -c 'import json,sys; c=json.load(sys.stdin)["checks"]; print(json.dumps(c["snapshot"]))'
   grep -E "published|REFUSING|refus|failed" /data/v10/logs/webuidb-sync-error.log | tail -3
   ```

   Open `https://ai.revelationsaints.com` yourself: her newest message is there, her images show, a reply
   comes back. The first sync log line after her first message should be `published local -> snapshot`;
   any `REFUSING` line: **stop and do not continue** (the old pod is untouched and is your rollback).
   Then set **Admin Panel, Settings, General, WebUI URL** to `https://ai.revelationsaints.com` (step 4.4).
6. **Now stop the OLD pod entirely** (RunPod: stop, do not terminate; it stays stopped). Its volume data is
   untouched and is the rollback.
7. **She opens `https://ai.revelationsaints.com`**, logs in again, and carries on.
8. Reclaim local disk: `rm -rf /var/lib/openwebui/.migrate-aside-*` (your test data, from before the cutover).
   Delete `/var/lib/openwebui/.migrate-tmp` if it is still there.
9. Watch her first day: `/data/v10/logs/`, `zl status`, and one `backup.py --once` plus `--verify` (step 4.6).
   From now on the new pod follows `DB-MOVE-RUNBOOK.md` section D (the **final sync before any planned stop or restart**:
   `supervisorctl stop openwebui webuidb-sync`, then `WEBUI_DB_LOCAL=true $PY /opt/compactor/webuidb.py --sync-once --force`).
   A RunPod stop without it loses up to about 2.5 minutes of chat.

## 6. Rollback

**Nothing has been written on the new pod yet** (minutes after the switch, she has not sent anything): stop
the new pod (RunPod: stop it, so the tunnel leaves), start the OLD pod again. That is all: the old tree was
never modified.

**She has written on the new pod** (and you want that back on the old pod): a reverse cutover with the same tool.
This is the only time it writes the old tree.

1. She closes the app. On the NEW pod: `supervisorctl stop openwebui compactor webuidb-sync backup`, then run the
   final sync **before** stopping (the sync daemon does it on `supervisorctl stop`; check
   `$PY /opt/compactor/webuidb.py --status` shows the same `chats=` on both rows).
2. The OLD pod must be stopped (it is). Then, on the NEW pod:

   ```bash
   $T cutover --reverse --from $NEW --to $OLD --old-pod-stopped                # dry run
   $T cutover --reverse --from $NEW --to $OLD --old-pod-stopped --apply
   echo EXIT=$?                                                                # 0, verify runs at the end
   ```

   The database goes to `/data/openwebui/webui.db` (the old pod's live file, `WEBUI_DB_LOCAL=false`), its `file.path` rows are
   rewritten back to `/data/openwebui/uploads/`, and files that only exist on the new pod are copied in.
   Files the old tree has and the new one does not are moved to `/data/openwebui/.migrate/removed-<stamp>/`.
3. **Stop the new pod completely** (RunPod stop) so its tunnel leaves. Start the OLD pod. It boots on the same
   `/data/openwebui` it always used. Check her chat as in section 5.5. If the old pod's OpenWebUI is 0.11.0 (the image
   reverted a hot patch), read the note in `memory.md`, "PROD'S OPENWEBUI IS 0.11.4": a pod restart reverts it.

## 7. Cleanup later (weeks, once you are sure)

* `/data/v10/openwebui/.migrate/removed-*` (moved-aside files) and `.migrate/seed.json`.
* On the OLD tree, only when you are certain you will never roll back: take an off-pod copy of `/data/openwebui`
  first. **Never delete `/data/models`** (the new pod uses it: it is the shared, read-only weights).
* `/data/backups` (the old pod's archives) stays as history; nothing prunes it any more.
* Old `/data/logs`, `/data/forensics`, `/data/zl-ops`, `/data/vllm-compile-cache`: history, small; keep until the
  old tree goes.

## Troubleshooting

| You see | Meaning | Do |
|---|---|---|
| `REFUSED: 'openwebui' is RUNNING on this pod` | the tool replaces this pod's data | `supervisorctl stop openwebui compactor webuidb-sync backup` |
| `REFUSED: --to is not THIS pod's DATA_DIR` | wrong `--to`, or the shell cannot tell this pod's `DATA_DIR` | check step 1's `/proc/1/environ` line; pass `--local-data-dir $NEW` only if you are sure |
| `REFUSED: the tree changed within Ns of watching it` | the old pod is still writing | stop the old pod's openwebui, compactor and backup; run it again |
| `REFUSED: the source webui.db has a -journal` | the old pod did not stop cleanly | `RUNBOOK_DB_JOURNAL.md` |
| `REFUSED: --to exists, is not a previous seed and holds data` | something wrote real data under `/data/v10` | find out what; `--force` moves unknown files aside, never deletes |
| `REFUSED: another run holds .../.migrate/lock` | a run is live or died | if dead, `rm /data/v10/openwebui/.migrate/lock` |
| `FAILED: the source tree changed while the cutover ran` | the old pod was not frozen | stop it properly and re-run `cutover --apply` |
| `database copy attempt N/5 failed (OperationalError)` | the old pod (or MooseFS) had the database busy | it retries; if it gives up, retry when she is not chatting |
| `verify` FAIL on `file rows in the target` | a path prefix the tool did not know | `--path-map OLD=NEW` (repeatable), then re-run |
| `webuidb-sync` log: `refusing to publish: snapshot reports 0 chats` | expected on the empty first boot's stop | nothing |
| `webuidb-sync` log: `REFUSING to publish ... OLDER` | the local DB is older than the snapshot: do not stop the pod | `DB-MOVE-RUNBOOK.md` E1 |

**The rule for every other tool on the new pod:** point it at `/var/lib/openwebui/webui.db` (the live file) and
the `/data/v10/...` paths; never at `/data/openwebui`. The tools now default to this pod's own paths even from a
shell without the pod's environment, but say the path anyway.
