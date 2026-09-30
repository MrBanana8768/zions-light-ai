# Moving to a NEW pod: v3.1.9.10, old pod to new pod

**This runbook REPLACES the v3.1.9.8 and v3.1.9.9 deploy runbooks** (`~/zl-ops/v3198/...`, `~/zl-ops/v3199/DEPLOY-v3.1.9.9.md`,
which are not in the repo and are not edited). Do not follow them for this move: they describe an in-place redeploy of
the OLD pod. Everything they carried that still matters is in here.

**What this is.** Prod today is the OLD pod (`angreg/zions-light-ai:v3.1.9-cu12`, `WEBUI_DB_LOCAL=false`, live DB on the
network volume at `/data/openwebui/webui.db`, her store at `/data/openwebui/compactor`, uploads at
`/data/openwebui/uploads`, logs and backups under `/data`). You start a NEW pod on
`angreg/zions-light-ai:v3.1.9.10-cu12`, attach the SAME network volume (for the model weights at `/data/models`), with
`WEBUI_DB_LOCAL=true` and all of ITS data in a separate folder, `/data/v10`. Seed, test, then at a quiet moment freeze
the old pod, resync, switch her over. **The old pod stays stopped afterwards, and is the rollback.**

**Why it is safe.** Read `docs/PATHS-v31910.md`. Every path the new pod writes is redirected by the env block below, and
the boot proof shows **0 files changed under the old tree**. The tool (`pod-migrate.py`) never writes the old tree (it
opens the old database `mode=ro`, only reads its files) unless you run the reverse cutover in section 7. Nothing here
needs a restart of the old pod, until a rollback.

**The tool** ships in the image: `/opt/zl-repo/scripts/pod-migrate.py` (stdlib only). Every command is a dry run until you
add `--apply`. Its output is counts only: no upload names, no conversation ids, no content. Sub-commands: `seed`,
`cutover` (and `cutover --reverse`), `verify`, `status`. Exit codes are in the failure table (section 10).

**Day-to-day operation on the new pod** (final sync before a stop, incidents, recovery after a pod loss, restores,
backups) is **`RUNBOOK_DB_LOCAL_V10.md`**. `docs/DB-MOVE-RUNBOOK.md` is for the OLD pod's layout only: never use it on the
new pod.

**The three rules that run through everything below**

1. **Start no service on the new pod until the cutover has reported complete AND `verify` has exited 0** (section 6.4). A
   service started early on a half-copied tree is how data gets overwritten.
2. **Seed only while she is idle.** The seed reads the OLD pod's LIVE database.
3. **Once a cutover has completed, never re-run `seed` or `cutover` into the same directory.** The tool refuses, and
   NO flag (`--force`, `--discard-new-pod-writes`, `--db-mode`) overrides that: after go-live the only ways forward are the
   rollback (section 7, `--reverse`) or a fresh target directory. Before go-live, testing on the new pod is discarded with
   `--discard-new-pod-writes` (step 6.2).

Start every Web Terminal session on the new pod with (the same variables are used everywhere below):

```bash
PY=/opt/compactor-venv/bin/python; APY=/app/venv/bin/python; T="$PY /opt/zl-repo/scripts/pod-migrate.py"
OLD=/data/openwebui; NEW=/data/v10/openwebui
V=/data/v10; L=$V/logs; SNAP=$NEW/webui.db; LIVE=/var/lib/openwebui/webui.db; STORE=$NEW/compactor; BK=$V/backups
```

The tool finds this pod's `DATA_DIR` from supervisord even in a shell without the pod's environment, and it **refuses to
write any tree that is not this pod's own** (so a swapped `--from`/`--to` cannot touch the old pod).

---

## 0. Before you start (about 30 minutes, nothing changes)

1. **Build and push BOTH images now, before anything else** (you need the rollback image under pressure, not while building
   it). On the laptop:

   ```bash
   git clone --depth 1 --branch v3.1.9.10 https://github.com/MrBanana8768/zions-light-ai.git zl-v3.1.9.10 && cd zl-v3.1.9.10
   git log -1 --format='%H %s'                                         # note the commit
   taken=0
   for t in v3.1.9.10-cu12 v3.1.9.10-rollback-cu12; do
     if docker manifest inspect angreg/zions-light-ai:$t >/dev/null 2>&1; then echo "EXISTS: $t"; taken=1; else echo "free: $t"; fi
   done
   if [ "$taken" = 1 ]; then echo "STOP: a tag already exists; never overwrite one. Nothing was built or pushed."; else
     docker build --platform linux/amd64 --target rc       -t angreg/zions-light-ai:v3.1.9.10-cu12          -f Dockerfile.v31910 . \
     && docker build --platform linux/amd64 --target rollback -t angreg/zions-light-ai:v3.1.9.10-rollback-cu12 -f Dockerfile.v31910 . \
     && docker run --rm --entrypoint /app/venv/bin/pip angreg/zions-light-ai:v3.1.9.10-rollback-cu12 show open-webui | grep ^Version \
     && docker run --rm --entrypoint /app/venv/bin/pip angreg/zions-light-ai:v3.1.9.10-cu12          show open-webui | grep ^Version \
     && docker push angreg/zions-light-ai:v3.1.9.10-cu12 \
     && docker push angreg/zions-light-ai:v3.1.9.10-rollback-cu12
   fi
   ```

   (Both `Version` lines must say `0.11.4`. The `if` block is one unit: if either tag exists, nothing is built or pushed, and
   a failing build stops the chain before any push. Paste the whole block, not line by line.)

   The `rollback` target of `Dockerfile.v31910` is **prod as it runs today**: the `v3.1.9-cu12` base with OpenWebUI 0.11.4
   installed (prod's OpenWebUI is 0.11.4 only through a hot patch in the running container's overlay, which a pod restart
   throws away). It is the image the OLD pod must be started on if you ever roll back (section 7).
2. **Free disk on the new pod** (the live DB, a scratch copy while the tool runs, and the moved-aside copy of the previous
   local DB): at least **3 GB free** on `/var/lib/openwebui` (`df -h /var/lib/openwebui`). The tool refuses if there is
   not enough.
3. **The secret key** (her login sessions). **Check the RunPod TEMPLATE value, not the pod's environment.** The pod's
   entrypoint exports a generated key when the template row is empty, so a check of the running pod's environment
   always says "present" and proves nothing. In the RunPod console open the OLD pod's template (or the pod, Edit) and look
   at the environment variables list: is there a `WEBUI_SECRET_KEY` row with a value? Do not print or copy the value.
   * **Row present:** the new pod's template must use the **same** value. In RunPod, reference the same Secret in both
     templates (`{{ RUNPOD_SECRET_<name> }}`) rather than retyping it. Her sessions then survive the move.
   * **No row on the old pod:** it generates a random key each boot; there is nothing to carry over, and the new pod may
     have no row either.

   In both cases she **logs in again** at `ai.revelationsaints.com` (a new host name has new cookies). The seed does not
   copy a key file: this image never reads one (`docs/PATHS-v31910.md`, "The secret key, decided").
4. **Where the old pod's uploads and DB paths are** (nothing to do, only to know): OpenWebUI stores an absolute path per
   uploaded file. The tool rewrites those paths in the NEW copy of the DB from `/data/openwebui/uploads/` to
   `/data/v10/openwebui/uploads/`, so the new pod never reads the old pod's files.
5. **Do not put `CLOUDFLARE_TUNNEL_TOKEN` on the old pod.** A tunnel balances between all connectors that hold the same
   token. It goes on the NEW pod only.
6. **The old pod's hand edits are not template rows.** On the old pod, `COMPACTOR_MAX_RETAINED_IMAGES=1` and
   `COMPACTOR_GENERATION_RESERVE=8000` were edited into supervisord's config by hand (prod's template rows are `2` and
   `12000`), so they are lost on any old-pod restart. The new pod gets them as literal template rows (section 1), and a
   rollback needs them added to the old pod's template first (section 7).

## 1. Create the new pod

RunPod: a new pod (do not touch the old one), the **same network volume mounted at `/data`**, image
`angreg/zions-light-ai:v3.1.9.10-cu12`, the old pod's template rows copied (GPU, `MODEL_REPO`, ...), **plus** these rows (they are
literal; `docs/PATHS-v31910.md` explains each and lists the proof):

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
COMPACTOR_BACKFILL_MAX_ATTEMPTS=1        # a SECOND layer only: step 6.5 closes her open backfill record for real

# --- the compactor settings that were supervisord hand edits on the old pod: LITERAL rows here ---
COMPACTOR_MAX_RETAINED_IMAGES=1
COMPACTOR_GENERATION_RESERVE=8000
COMPACTOR_BACKUP_RETAIN=3                # the old 28 is a 28-hour floor that swallows the 3-hour tier

# --- FIRST BOOT ONLY (removed in step 3.6, in the same edit that adds the tunnel token) ---
ENABLE_LOGIN_FORM=false

# --- secrets and URL ---
WEBUI_SECRET_KEY=<the same RunPod Secret as the old pod, or leave the row out; see step 0.3>
WEBUI_URL=https://ai.revelationsaints.com
# CLOUDFLARE_TUNNEL_TOKEN is NOT here yet. It is added in step 3.6, after the seed.
```

**Must NOT exist on the new pod:** `DATABASE_URL`, `WEBUI_LOCAL_DB`, any `WEBUI_DB_ALLOW_*`, and
**`COMPACTOR_INJECTION_BUDGET_FRACTION`** (delete a `.6` row: it pins the fraction below the image's 0.75). Keep the old
pod's other rows (backup interval, ...) as they are. **`COMPACTOR_BACKUP_RETAIN=3`, `COMPACTOR_MAX_RETAINED_IMAGES=1` and
`COMPACTOR_GENERATION_RESERVE=8000` must be in the new template as written above; do not assume they carry over.**

### 1.1 SECURITY: the new pod must not be reachable from the internet

**In the new pod's template, "Expose HTTP Ports": REMOVE `3000` and `8080`** (leave those two out of the list; the
cloudflared tunnel reaches `localhost:3000` from inside the pod, it needs no exposed port). Why: the RunPod proxy URL
(`https://<pod-id>-3000.proxy.runpod.net`) is public. An OpenWebUI with no users accepts the **first sign-up as ADMIN**,
even with `ENABLE_SIGNUP=false`, and an OpenWebUI admin can run Python as root inside the pod and read or write the
network volume, including the OLD pod's live database. The first boot below is exactly that empty state. (Nothing else on
the new pod needs an exposed HTTP port. If your template also exposes other HTTP ports you do not use, e.g. the vLLM port,
remove those too. Keep TCP `22` only if you use `zlpod`.)

**Belt and braces for the same window: `ENABLE_LOGIN_FORM=false`** (the row above). Measured on 0.11.4 (R2): with it, a
sign-up on the empty database is refused (403) instead of creating an admin. It is a **first-boot-only** row:

* It is a persisted setting. On a boot OpenWebUI writes any setting that has no row in the database from the env value
  into the database (`config.seed_defaults`), and a value already in the database beats the env. Sign-in itself does not
  check it (the server accepts a password login either way), but with `false` in effect the login FORM is hidden in the
  web UI, so she could not log in.
* The seed replaces the empty database with hers. If her database already holds `ui.enable_login_form = true`, the env row
  is ignored from then on. If her database has NO such row and the env row is still `false` at the first boot on her
  database, OpenWebUI writes `false` into it and the form stays hidden until repaired.
* Therefore: the row is removed in step 3.6, **before** OpenWebUI is ever started on her database, and step 3.7 checks the
  value in her database.

### 1.2 Check the pod

When the pod is up, in its Web Terminal:

```bash
tr '\0' '\n' < /proc/1/environ | grep -E '^(DATA_DIR|LOG_DIR|COMPACTOR_BACKUP_DIR|COMPACTOR_STORAGE_ROOT|WEBUI_DB_LOCAL|WEBUI_SNAPSHOT_DB|COMPACTOR_MAX_RETAINED_IMAGES|COMPACTOR_GENERATION_RESERVE|COMPACTOR_BACKUP_RETAIN|COMPACTOR_BACKFILL_MAX_ATTEMPTS|ENABLE_LOGIN_FORM)='
```

**Expect** exactly these values: the five `/data/v10/...` paths, `WEBUI_DB_LOCAL=true`, `COMPACTOR_MAX_RETAINED_IMAGES=1`,
`COMPACTOR_GENERATION_RESERVE=8000`, `COMPACTOR_BACKUP_RETAIN=3`, `COMPACTOR_BACKFILL_MAX_ATTEMPTS=1`, and
`ENABLE_LOGIN_FORM=false`. If ANY is missing or different, **stop and fix the row before going on**: a missing path row is the one
way the new pod could aim at the old pod's files.

**Check that 3000 and 8080 are not public.** First prove the check can mean something: OpenWebUI must be UP inside the pod. In
the pod's terminal:

```bash
curl -s -o /dev/null -w '%{http_code}\n' 127.0.0.1:3000/health        # expect 200; if not, wait a minute and repeat
```

Only when that says `200`, run the external check from your laptop (the pod id is `echo $RUNPOD_POD_ID` in the pod's terminal;
a "not exposed" answer while OpenWebUI was still booting would prove nothing):

```bash
for p in 3000 8080; do curl -s -o /dev/null -w "$p -> HTTP %{http_code}\n" --max-time 20 https://<POD_ID>-$p.proxy.runpod.net/; done
```

**Expect** an error status (404 or 502 or a RunPod "not exposed" page), never an OpenWebUI page and never a `200`, `301`,
`302` or `307`. If you get one of those, the port is public: `supervisorctl stop openwebui` in the pod's terminal
at once, fix the template, and check again before continuing.

## 2. First boot: an empty first boot, then stop, then seed. (The decision, and why.)

**The choice.** The seed needs the new pod (it has the tool and sees both trees on the volume), but OpenWebUI must not
run on an empty DB when real data is about to arrive. supervisord starts OpenWebUI unconditionally and no env row changes
that. **We accept the empty boot**, because it is harmless by construction:

* It happens entirely on the new pod: an empty schema on the pod's local disk. The old tree is not touched (boot proof).
* Nothing can reach it: no exposed port (1.1), no tunnel token yet, and a sign-up is refused (`ENABLE_LOGIN_FORM=false`).
* `webuidb-sync` **refuses to publish a 0-chat database** ("snapshot reports 0 chats; refusing to publish"), so no empty
  snapshot is written to the volume either.
* The seed then **replaces the local DB** (moves the empty one aside), so the empty schema never survives.

In the Web Terminal, when the pod is up:

```bash
supervisorctl stop openwebui compactor webuidb-sync backup      # the empty first boot is over (vllm keeps running)
supervisorctl status openwebui compactor webuidb-sync backup    # all STOPPED
```

(`webuidb-sync` logs the "refusing to publish" line on the way out. Expected.)

## 3. Seed, verify (the old pod stays live; nothing on it changes)

1. **She closes the app on EVERY device, and stays away until the seed has finished: REQUIRED.** The seed reads the OLD pod's
   live database. An idle open OpenWebUI tab is enough to write it: it stores `last_active_at` about once a minute, and every
   write to the source restarts the SQLite backup from page 1. Ask her to close the app on every device (phone, tablet, laptop:
   all tabs). The tool copies in 4096-page steps with a short pause (the total pause is capped), and if the source keeps
   being written it retries with fast steps, then (only when the copy is projected to hold the lock for at most 3 s) as a
   single full copy. It prints the mode, the restarts, the attempts, the total time and the **longest time it held the old
   pod's read lock**; the old pod's own `busy_timeout` is 5 s, so a hold near that would make one of her writes fail.
   It warns when the old database was written in the last 5 minutes (measured against the volume's clock, not the pod's). If it
   warns, or says `the source database kept changing under the copy`, make sure the app is closed everywhere, wait, and run it again.
2. Dry run, then apply:

   ```bash
   $T seed --from $OLD --to $NEW                # the dry run: read the plan (counts only)
   $T seed --from $OLD --to $NEW --apply
   echo "SEED EXIT=$?"                          # 0
   $T verify --from $OLD --to $NEW
   echo "VERIFY EXIT=$?"                        # 0
   $T status --to $NEW
   echo "STATUS EXIT=$?"                        # 0, prints COMPLETE
   ```

   **Expect** `pod-migrate verify: all checks passed (0 failing)` and `pod-migrate status: COMPLETE`. What `verify` compares,
   counts only: integrity_check on both databases; the row count of every table; users, chats, `chat_message` rows and
   `file` rows; her chat (the size of the chat JSON, its message count and its `chat_message` rows); every uploaded file's
   path in the new DB (none may still point at the old tree); facts and summaries per store; the chroma document counts; and a
   hash of every file in every copied tree, **including the logical content of the SQLite files** (the chroma indexes).
3. While the old pod is live and she keeps chatting, `verify` will show differences after a while. That is normal; the
   cutover closes them. Re-running `seed --apply` is a safe incremental refresh (only while she is idle). If you have tested on the
   new pod since the last seed, the re-seed refuses like the cutover does (its local database has writes) and needs
   `--discard-new-pod-writes` too (allowed only BEFORE the cutover: after a completed cutover no flag lets a seed run).
4. If `seed` says the target "holds data and is not a seed", something already wrote real data into `/data/v10`: find out
   what before using `--force` (with `--force` nothing is deleted either; unknown files are moved aside).
5. **Not copied on purpose** (counts only in the output): OpenWebUI's `cache/`, old `webui.db.bak-*` files, its audit log. If
   she uses document search and the new pod is offline (`COMPACTOR_FORCE_OFFLINE=true`), add `--with-cache` to every
   command, or the embedding model has to be downloaded first (temporarily remove the offline row). The word-ban directory
   (`/data/zl-ops/word-ban`) is copied if it exists on the old pod. If not (v3.1.9 has no file deploy), deploy the grammar to
   the new pod with `word-ban-deploy.py` before switching her (`OPERATIONS.md`).
6. **The template edit, then a pod restart.** Now (services are still stopped) edit the new pod's template:
   * **DELETE the `ENABLE_LOGIN_FORM` row.**
   * **ADD `CLOUDFLARE_TUNNEL_TOKEN`** (secret). (`RUNBOOK_CLOUDFLARE_TUNNEL.md` covers the dashboard side: the public hostname
     `ai.revelationsaints.com` pointing at `http://localhost:3000`.)
   * Do NOT re-add ports 3000 or 8080.

   Then restart the pod. **Nothing is lost by the restart**: the seeded snapshot is on the volume, and the boot restores it to
   local disk if the local file did not survive. Do NOT start OpenWebUI by hand before this restart.
7. **After the restart, check her database's login settings** (read-only):

   ```bash
   tr '\0' '\n' < /proc/1/environ | grep -c '^ENABLE_LOGIN_FORM='          # 0: the row is gone
   $PY -c 'import sqlite3; c=sqlite3.connect("file:/var/lib/openwebui/webui.db?mode=ro", uri=True); print(c.execute("SELECT key, value FROM config WHERE key IN (?,?,?) ORDER BY key", ("ui.enable_login_form","ui.enable_signup","webui.url")).fetchall()); print("users:", c.execute("SELECT COUNT(*) FROM user").fetchone()[0])'
   ```

   **Expect** `0`, then `ui.enable_login_form true` (or no such row: then the boot has just written `true`), and `users:` at least 1.
   **If it says `ui.enable_login_form false`:** the form is hidden and she could not log in. Repair it now, with OpenWebUI
   stopped, then let the sync publish:

   ```bash
   supervisorctl stop openwebui webuidb-sync backup
   $PY -c 'import json, sqlite3; c=sqlite3.connect("/var/lib/openwebui/webui.db"); c.execute("UPDATE config SET value=? WHERE key=?", (json.dumps(True), "ui.enable_login_form")); c.commit(); print("fixed:", c.execute("SELECT value FROM config WHERE key=?", ("ui.enable_login_form",)).fetchall())'
   supervisorctl start webuidb-sync openwebui backup
   ```

   (`ui.enable_signup true` is how prod already behaves: a new sign-up is a pending user until you approve it. Set it to
   off in the Admin Panel if you want.)
8. Reclaim local disk once you are happy: the tool moved the empty first-boot DB aside. (`.migrate-aside-*` lives on the
   pod's LOCAL disk: it vanishes on any RunPod restart or stop-start, so if you restarted the pod in step 6 it is already
   gone, and the `ls` below simply prints nothing. That is fine: it was only the empty schema.)

   ```bash
   ls -d /var/lib/openwebui/.migrate-aside-*        # the empty first-boot database and its lock file
   rm -rf /var/lib/openwebui/.migrate-aside-*       # safe to delete: it is only the empty schema
   ```

## 4. Test the new pod (you only, never her)

**How you reach the new pod before she does, decided: through the tunnel itself.** The tunnel token was added in 3.6, the
pod has no public port, and `https://ai.revelationsaints.com` is not what she uses yet (she is still on the old pod's RunPod
proxy URL), so you can test it there without her seeing anything. If the tunnel misbehaves, the fallback is an SSH
port-forward from the laptop: `ssh -L 3000:localhost:3000 zlpod`, then open `http://localhost:3000` (needs `zl sshd` set up:
`ops/README.md`). Never re-expose 3000.

1. Open **`https://ai.revelationsaints.com`** (the RunPod proxy URL is intentionally dead for this pod). Log in (she has to
   log in again anyway).
2. Check: her chat and its history load; her uploaded images show (this proves the path rewrite); one message round-trips (the
   first reply after a cold start can take up to ~100 s).
3. **Admin Panel, Settings, General, WebUI URL: set it to `https://ai.revelationsaints.com`.** The seeded DB already holds a
   `webui.url` value (or none), and OpenWebUI prefers the saved value over the env row, so the template row does not change it.
   Check with the snippet in 3.7. **The cutover replaces the database, so set it again after the cutover** (step 6.7).
4. **Everything you write on the new pod during testing is discarded at the cutover** (the target DB is replaced from the old
   pod's; the new pod's own test facts are moved aside, never deleted). Do not test with anything you want to keep. The tool
   knows: testing changes the new pod's local database, and the cutover will REFUSE until you confirm with
   `--discard-new-pod-writes` (step 6.2).
5. A first backup of the new pod's data, into its own directory:

   ```bash
   $PY /opt/compactor/backup.py --once; echo BACKUP_EXIT=$?
   A=$(ls -t $BK/zions-backup-*.tar.gz | head -1); $PY /opt/compactor/backup.py --verify "$A"; echo VERIFY_EXIT=$?
   ```

   **Expect** `0` and `0`. That archive is a **test archive**; step 6.6 moves it out of the retention glob after the cutover.
   `/data/backups` (the old pod's) stays exactly as it is, as history: nothing on the new pod reads or prunes it.
6. Check that `zlpod` sees the right directory: `zl disk | grep '^backups:'` must say `/data/v10/backups`. If it says
   `/data/backups`, `zlpod pull backup` would fetch the OLD pod's archives (it falls back to that path, with only a warning):
   fix the rows first.

## 5. The cutover window: preparing

Pick a quiet time. **Tell her.** It is about 15 to 30 minutes (the cutover hashes every file: it does not trust file sizes
and times).

1. **She closes the app on every device.**
2. **Is the OLD pod idle? Run this on the OLD pod's terminal.** Freezing the old pod while it is mid-reply cuts her last
   reply. The old pod (v3.1.9) has no request tracing, and **its access log is not proof of an idle pod: uvicorn logs
   `POST /v1/chat/completions ... 200` when a reply STARTS to stream, not when it ends.** So the proof is vLLM's own metrics
   (two samples, 30 s apart); the compactor log and the open-connection count only add to it:

   ```bash
   L=/data/logs; VP=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^VLLM_PORT=//p'); VP=${VP:-8000}
   M() { curl -s http://127.0.0.1:$VP/metrics | grep -E '^vllm:num_requests_(running|waiting)[ {]'; }
   M; sleep 30; M
   ```

   **Expect** all four numbers to be `0.0` (running and waiting, both samples), i.e. running + waiting is 0 twice, 30 s apart.
   A stream in progress shows as `running` above 0. Then two extra checks. **Both `conv_id=` and `POST` lines are in
   `compactor.log`** (that one file; `compactor-error.log` is the wrong place to look):

   ```bash
   tail -n 300 $L/compactor.log | grep -E 'conv_id=|"POST /v1/chat/completions' | tail -3
   awk '$4=="01" && $2 ~ /:1F90$/' /proc/net/tcp /proc/net/tcp6 | wc -l      # established connections whose local port is 8080
   ```

   (Port 8080 in hex is `1F90`: `0x1F90` = 1*4096 + 15*256 + 9*16 + 0 = 8080. `ss` is not installed on v3.1.9, which is why this
   reads `/proc/net/tcp*`, where state `01` is ESTABLISHED. The image ships `awk`.)

   **Expect** the newest of the log lines to be a `"POST /v1/chat/completions ... 200` line, NOT a `conv_id=` line (a `conv_id=`
   line with no POST line after it means a request is in its ~100 s pre-processing phase, when vLLM shows nothing at all), and a
   count of `0`, or `1` that is gone when you repeat it (a health poll or OpenWebUI's idle keep-alive is short-lived; run it three
   times, a few seconds apart). **The metrics decide; these two can only add a reason to wait.** If any of it says a request may
   be in flight, or you are not sure (a log you cannot read): **wait until 5 minutes after her last reply**, then check again. When
   in doubt, wait the 5 minutes.
3. **Freeze the OLD pod.** On the OLD pod's terminal:

   ```bash
   supervisorctl stop openwebui compactor backup
   supervisorctl status openwebui compactor backup       # all STOPPED
   ls -la /data/openwebui | grep webui.db                # ONE line: webui.db. No -journal, no -wal.
   ```

   If a `-journal` is there: **stop**, follow `RUNBOOK_DB_JOURNAL.md`. (vLLM may keep running on the old pod; it writes nothing
   that matters here.)
4. **On the NEW pod: stop its writers**, **including `backup`** (the tool refuses to run while any of them runs):

   ```bash
   supervisorctl stop openwebui compactor webuidb-sync backup
   supervisorctl status openwebui compactor webuidb-sync backup     # all STOPPED
   ```

   (The new pod's own final sync is not needed here: the cutover REPLACES the new pod's database, so its test data is
   discarded on purpose.)

## 6. The cutover

1. **The dry run** (it changes nothing; it hashes every file, so it takes minutes, plus a 60 s quiet check of the old tree):

   ```bash
   $T cutover --from $OLD --to $NEW --old-pod-stopped
   echo "DRY EXIT=$?"
   ```

2. **Read what it says, and take the branch:**
   * `DRY EXIT=0`: go on to step 3 without the extra flag.
   * **`REFUSED: The new pod has written to its local database since this tool installed it ...`** with row counts: this is what
     testing (section 4) does. It is expected here, and only here. **A refused dry run stops at the preconditions: it printed no
     plan and did not run the 60 s quiet check of the old tree.** So run the dry run AGAIN with the flag (allowed before go-live
     only), so that the plan and the quiet check really run before you apply:

     ```bash
     $T cutover --from $OLD --to $NEW --old-pod-stopped --discard-new-pod-writes
     echo "DRY EXIT=$?"        # 0, and it prints what will be moved aside (never deleted)
     ```

     Then add `--discard-new-pod-writes` in step 3.
   * **`REFUSED: a CUTOVER ALREADY COMPLETED into this directory ...`**: **STOP.** Someone already switched her over, and the new pod
     may have served her since. No flag overrides this (not `--discard-new-pod-writes`, not `--force`, not `--db-mode`): the tool
     will not move her new messages aside. Go to section 7 (rollback) or `RUNBOOK_DB_LOCAL_V10.md` (recovery) and ask.
   * **`REFUSED: <to>/.migrate/seed.json exists but this tool cannot use it`**: the manifest was written by an older tool
     (format 1) or is damaged, and it may record a completed cutover. There is deliberately no override. Seed into a fresh
     target directory instead, or roll back (section 7).
   * Any other `REFUSED`: the failure table (section 10).
3. **Apply it, detached, so a closed terminal cannot kill it**, and watch the log:

   ```bash
   setsid nohup $T cutover --from $OLD --to $NEW --old-pod-stopped --apply [--discard-new-pod-writes] > $V/cutover.log 2>&1 &
   sleep 3; tail -f $V/cutover.log       # Ctrl+C stops only the tail, never the cutover
   ```

   (Type `--discard-new-pod-writes` only if step 2 told you to, and without the brackets. If it is on the line but no testing
   happened, it changes nothing.) The run ends with
   `pod-migrate verify: all checks passed (0 failing)`. If the terminal or pod session drops, the run continues (SIGHUP is
   ignored); `tail -n 30 $V/cutover.log` when you are back. If it is killed (Ctrl+C, `kill`, SIGTERM) it finishes the one step
   that must not be torn (installing the database), stops, and leaves the manifest marked incomplete.
4. **THE GATE. Do not start any service unless BOTH lines pass:**

   ```bash
   $T status --to $NEW --require cutover; echo "STATUS EXIT=$?"        # must print COMPLETE and EXIT=0
   $T verify --from $OLD --to $NEW --strict-local; echo "VERIFY EXIT=$?"  # must print all checks passed and EXIT=0
   ```

   `STATUS EXIT=0` means the last run was a cutover, it finished, and its own verify passed. **Anything else (INCOMPLETE, exit 1
   or 2) means: start nothing.** If the run was interrupted or failed BEFORE the database was installed, repeat the same
   `cutover ... --apply` command; it is safe to repeat (the failure table has the causes). **If `status` says `A cutover completed
   here ... but its verify did not pass`, the tool will refuse a second cutover into this directory (by design, after go-live).**
   Nothing has served her (no service was started, rule 1), so the recovery is a deliberate hand step: move the manifest aside
   with `mv $NEW/.migrate $NEW/.migrate-failed-$(date -u +%Y%m%dT%H%M%SZ)`, then `seed ... --force --apply` and a fresh
   `cutover ... --apply` (the tool treats the tree as unknown data and moves unknown files aside, never deletes). Do this ONLY if
   no service on this pod has been started since the cutover; if any was, use the rollback (section 7) and ask.
   Then check the databases:

   ```bash
   $PY /opt/compactor/webuidb.py --status
   R=$($PY /opt/compactor/webuidb.py --status | grep -E '^(local|snapshot) ' | sed -E 's/.*quick_check=/quick_check=/')
   [ "$(echo "$R" | grep -c .)" = 2 ] && [ "$(echo "$R" | sort -u | wc -l)" = 1 ] && [ "$(echo "$R" | grep -c '^quick_check=ok ')" = 2 ] \
     && echo "STATUS ROWS MATCH" || echo "STATUS ROWS DIFFER: stop"
   ```

   **Expect** `local` and `snapshot` rows with the same `chats=`, `content=` and `newest_update=`, both `quick_check=ok`, and
   `STATUS ROWS MATCH` (the one-liner keeps `quick_check` in the compared text, and needs exactly two rows, both `ok`).

   What the cutover does, for reference: a fresh, consistent copy of `webui.db` (its `file.path` rows rewritten to the new pod),
   then a delta of the store, uploads, `vector_db` and word-ban. For the delta it **hashes every non-SQLite file on both sides
   and re-copies every SQLite file** (the chroma indexes) through the SQLite backup API; it does not trust sizes and modification
   times (a same-size change in the same second would be missed). Files deleted on the old pod, and files the new pod wrote
   during testing, are **moved** to `/data/v10/openwebui/.migrate/removed-<stamp>/`, never deleted. The database on this pod is
   **replaced, not merged**: the local DB, its `-journal`/`-wal`/`-shm`, `.sync.lock` and `.empty-start` are moved to
   `/var/lib/openwebui/.migrate-aside-<stamp>/`, the fresh copy is placed at `/var/lib/openwebui/webui.db` AND at
   `/data/v10/openwebui/webui.db`, and stale sidecars beside the old snapshot are moved aside too. (If the old tree changes
   while it runs, it fails with exit 1 and says so.)
5. **Close her open backfill record. Do this NOW: after the cutover (it copies the OLD store back over the new one, record
   included), and BEFORE the compactor starts.** Her old backfill record is an `in_progress` record written by a release
   that never wrote an attempts counter, so it can resume a full backfill on the new image.

   ```bash
   COMPACTOR_BACKFILL_MAX_ATTEMPTS=3 $PY /opt/zl-repo/scripts/backfill-records.py --store $STORE; echo "DRY EXIT=$?"
   COMPACTOR_BACKFILL_MAX_ATTEMPTS=3 $PY /opt/zl-repo/scripts/backfill-records.py --store $STORE --apply; echo "APPLY EXIT=$?"
   COMPACTOR_BACKFILL_MAX_ATTEMPTS=3 $PY /opt/zl-repo/scripts/backfill-records.py --store $STORE; echo "RECHECK EXIT=$?"
   ```

   **Expect** the first (dry run) to list her record as `would-resume` (exit `3`, informational), the second to print
   `closed -> state=abandoned` and exit `0`, and the third to print `DRY RUN — nothing was written, and nothing would
   resume.` and exit `0`. If the first already exits `0` with `nothing would resume`, there was nothing to close: fine. Exit
   `1` means a record needs a human (`needs-review`, a refused `--apply`): read the reason line and ask. **The template row
   `COMPACTOR_BACKFILL_MAX_ATTEMPTS=1` stays** as a second layer; it is not what protects her record (a record without an
   attempts counter used to read as zero attempts, so the row alone did not stop it on releases before this one).
6. **Label the test archives from before the cutover, before the backup daemon runs again.** Chosen: **MOVE them into
   `$BK/pre-cutover-test/`** (outside the retention glob: retention only looks at `zions-backup-*.tar.gz` directly in `$BK`),
   not delete. They were made from the seed and from testing, not from her live history:

   ```bash
   mkdir -p $BK/pre-cutover-test
   find $BK -maxdepth 1 -name 'zions-backup-*.tar.gz' ! -newer $V/cutover.log -exec mv -t $BK/pre-cutover-test/ {} +
   ls $BK/pre-cutover-test/; ls $BK | head
   ```

   Do not restore from them. (Nothing is newer than `cutover.log` yet: the backup daemon has been stopped since 5.4.)
7. **Start the new pod's services and check**:

   ```bash
   T0=$(date -u +'%Y-%m-%d %H:%M:%S')          # the pod's clock is UTC; used below to skip lines from before the cutover
   supervisorctl start compactor webuidb-sync openwebui backup
   sleep 60; supervisorctl status compactor webuidb-sync openwebui backup     # all RUNNING
   curl -s http://127.0.0.1:8080/health/full | $PY -c 'import json,sys; c=json.load(sys.stdin)["checks"]; print(json.dumps(c["snapshot"]))'
   awk -v t0="$T0" '($1" "$2) >= t0' $L/webuidb-sync-error.log | grep -E "published|REFUSING|refus|failed" | tail -3
   ```

   (The `awk` keeps only log lines from after `T0`: the log also holds `published` lines from your testing before the cutover,
   which would look like a healthy sync when it has not run yet.)

   Open `https://ai.revelationsaints.com` yourself: her newest message is there, her images show, a reply comes back. The first sync
   log line after her first message should be `published local -> snapshot`; any `REFUSING` line: **stop and do not continue**
   (the old pod is untouched and is your rollback). Then set **Admin Panel, Settings, General, WebUI URL** to
   `https://ai.revelationsaints.com` (step 4.3): the cutover replaced the database.
8. **Now stop the OLD pod entirely** (RunPod: stop, do not terminate; it stays stopped). Its volume data is untouched and is the
   rollback.
9. **She opens `https://ai.revelationsaints.com`**, logs in again, and carries on.
10. Reclaim local disk: `rm -rf /var/lib/openwebui/.migrate-aside-*` (your test data, from before the cutover). Delete
    `/var/lib/openwebui/.migrate-tmp` if it is still there.
11. Watch her first day: `$L`, `zl status`, and one `backup.py --once` plus `--verify` (step 4.5). **From this moment the new pod
    follows `RUNBOOK_DB_LOCAL_V10.md`**, starting with the **final sync before any planned stop or restart** (section 4 there:
    stop `openwebui webuidb-sync backup`, `webuidb.py --sync-once --force`, and gate on the exit code AND the matching `--status`
    rows). A RunPod stop without it loses up to about 2.5 minutes of chat. **Never re-run `seed` or `cutover` into this directory from here on** (the tool refuses).

## 7. Rollback

**Before anything else, in EVERY rollback:** the OLD pod must start on the **rollback image**, not on its old image.

* The OLD pod's OpenWebUI is 0.11.4 only through a hot patch in the container's overlay. A restart (and starting a stopped pod
  is a restart) throws that away and reverts it to the image's 0.11.0, against a database 0.11.4 has migrated.
* **Set the OLD pod's image to `angreg/zions-light-ai:v3.1.9.10-rollback-cu12` BEFORE starting it** (built and pushed in step 0.1: it
  is the `rollback` target of `Dockerfile.v31910`, prod plus OpenWebUI 0.11.4). RunPod: edit the stopped pod or its template,
  change the image, then start.
* **And add the two hand edits to the old pod's template first** (they were supervisord edits and are gone after a restart):
  `COMPACTOR_MAX_RETAINED_IMAGES=1` and `COMPACTOR_GENERATION_RESERVE=8000` (prod's own rows are `2` and `12000`).
* **The 0.11.0 fallback, if the rollback image cannot be used** (it was not pushed, or RunPod will not take the image): starting the
  old pod on plain `v3.1.9-cu12` **does boot** (smoke-tested: a loud but non-fatal `Can't locate revision identified by
  'd4c1a8e37b62'` traceback every boot), but it is a degraded state to leave quickly. A full chat save on 0.11.0 against the migrated
  schema timed out at 600 s in one trial with `database is locked` errors, and when a save did land, the chat JSON and the
  `chat_message` table drifted apart by one save: the class of bug that broke her UI on 2026-09-23 (`repair-chat-tree
  --sync-json-only` cannot fix that direction). Use it only to get her chatting, and rebuild the rollback image straight away.

**Nothing has been written on the new pod yet** (minutes after the switch, she has not sent anything): stop the new pod
(RunPod: stop it, so the tunnel leaves), set the old pod's image and rows as above, start the OLD pod, then do the **word-ban
step** (step 4 below). The old tree was never modified.

**She has written on the new pod** (and you want that back on the old pod): a reverse cutover with the same tool. This is the
only time it writes the old tree. It copies the new pod's **LIVE LOCAL database** (read through the SQLite backup API; the new
pod's services are stopped), NOT the snapshot on the volume (the snapshot can be minutes behind), and refuses if that local
file is missing.

1. She closes the app. **On the NEW pod: the final sync first**, exactly as `RUNBOOK_DB_LOCAL_V10.md` section 4
   (`supervisorctl stop openwebui webuidb-sync backup`, then `webuidb.py --sync-once --force`, then the matching `--status`
   rows). It is a belt for the snapshot; the reverse cutover itself reads the local database. Also stop the compactor:
   `supervisorctl stop compactor`.
2. The OLD pod must be stopped (it is). Then, on the NEW pod, dry run, then apply detached:

   ```bash
   $T cutover --reverse --from $NEW --to $OLD --old-pod-stopped
   setsid nohup $T cutover --reverse --from $NEW --to $OLD --old-pod-stopped --apply > $V/reverse.log 2>&1 &
   sleep 3; tail -f $V/reverse.log
   $T status --to $OLD --require cutover; echo "STATUS EXIT=$?"        # COMPLETE, 0
   ```

   The database goes to `/data/openwebui/webui.db` (the old pod's live file, `WEBUI_DB_LOCAL=false`), its `file.path` rows are
   rewritten back to `/data/openwebui/uploads/`, and files that only exist on the new pod are copied in. Files the old tree has
   and the new one does not are moved to `/data/openwebui/.migrate/removed-<stamp>/`. The new pod's word-ban directory is copied
   to the old pod's `/data/zl-ops/word-ban` (it never copies the directory onto itself). The run's own verify compares the LOCAL
   database with what it wrote. **Do not start the old pod unless `status` says COMPLETE.**
3. **Stop the new pod completely** (RunPod stop) so its tunnel leaves. Start the OLD pod (rollback image, rows added). It boots on
   the same `/data/openwebui` it always used, and she uses its RunPod proxy URL again (the old pod has no tunnel token).
4. **Re-paste the word ban into OpenWebUI, BEFORE she chats** (every rollback). On the new pod the grammar lives in a file
   (`/data/v10/zl-ops/word-ban/active.json`) and the `structured_outputs` row in OpenWebUI's Advanced Params may have been
   deleted. The old v3.1.9 pod has no file deploy: without the row she gets no word ban. On the laptop: Admin Panel, Settings,
   Models, the model, Advanced Params, `structured_outputs`: paste the value file of the revision that is live on the new pod
   (`~/zl-ops/word-ban/word-ban-structured_outputs.value.vN.txt`; the revision number is in
   `$PY /opt/zl-repo/scripts/word-ban-deploy.py status` run on the new pod before you stop it), Save & Update. Check that the
   system-prompt paragraph is still there.
5. Check her chat as in step 6.7. Get back onto the new pod's line of releases as soon as the reason for the rollback is fixed.

## 8. If the new pod is lost later

Everything about a lost pod (what survives, how much can be lost, the exact steps, `chat-journal.py compare` and its limits,
a damaged snapshot, restoring an archive) is in **`RUNBOOK_DB_LOCAL_V10.md`, section 6 (E2)**. In one paragraph: the live DB is on the
pod's disk and is gone; the snapshot on `/data/v10` is at most about 2.5 minutes behind (a verified final sync makes it
identical); the chat journal on the volume holds every completed reply, so `chat-journal.py compare` tells you which exchanges
are missing; never restore an archive to get those minutes back.

## 9. Cleanup later (weeks, once you are sure)

* `/data/v10/openwebui/.migrate/removed-*` (moved-aside files), `.migrate/seed.json`, `$V/cutover.log`, `$BK/pre-cutover-test/`.
* On the OLD tree, only when you are certain you will never roll back: take an off-pod copy of `/data/openwebui` first. **Never
  delete `/data/models`** (the new pod uses it: it is the shared, read-only weights).
* `/data/backups` (the old pod's archives) stays as history; nothing prunes it any more.
* Old `/data/logs`, `/data/forensics`, `/data/zl-ops`, `/data/vllm-compile-cache`: history, small; keep until the old tree goes.

## 10. Failure table

`pod-migrate.py` exit codes (`seed`, `cutover`, `verify`, `status`): **0** done (a dry run: the same command with `--apply` would
proceed; `status`: COMPLETE), **1** a step failed, a check found a mismatch, the source changed during the run, the run was
interrupted, or `status` says INCOMPLETE, **2** REFUSED before touching anything (a precondition failed, bad usage, or `status`
found no manifest). **Whatever the code: start no service unless section 6.4 passes.**

| You see | Exit | Meaning | Do |
|---|---|---|---|
| `REFUSED: 'openwebui' is RUNNING on this pod` | 2 | the tool replaces this pod's data | `supervisorctl stop openwebui compactor webuidb-sync backup` |
| `REFUSED: --to is not THIS pod's DATA_DIR` | 2 | wrong `--to`, or the shell cannot tell this pod's `DATA_DIR` | check step 1.2's `/proc/1/environ` line; pass `--local-data-dir $NEW` only if you are sure |
| `REFUSED: --reverse: --from must be THIS pod's DATA_DIR` | 2 | the arguments of a rollback are swapped | `--reverse --from $NEW --to $OLD` |
| `REFUSED: the tree changed within Ns of watching it` | 2 | the old pod is still writing | stop the old pod's openwebui, compactor and backup; run it again |
| `REFUSED: the source webui.db has a -journal` | 2 | the pod that owns it did not stop cleanly | `RUNBOOK_DB_JOURNAL.md` |
| `REFUSED: --to exists, is not a previous seed and holds data` | 2 | something wrote real data under `/data/v10` | find out what; `--force` moves unknown files aside, never deletes |
| `REFUSED: The new pod has written to its local database since this tool installed it` | 2 | testing changed the new pod's database (or she used it) | after testing only: `--discard-new-pod-writes` (6.2). If she has used the pod: do not; section 7 |
| `REFUSED: a CUTOVER ALREADY COMPLETED into this directory` | 2 | a seed/cutover after a completed cutover (no flag overrides it) | **STOP.** Section 7 (`--reverse`) or a fresh target directory; `RUNBOOK_DB_LOCAL_V10.md` |
| `REFUSED: <to>/.migrate/seed.json exists but this tool cannot use it` | 2 | an older-format or damaged manifest (may hide a completed cutover) | no override: a fresh target directory, or section 7 |
| `REFUSED: this pod's local live database does not exist` (reverse) | 2 | the local DB is gone (pod restarted onto a fresh disk) | the snapshot is then the only copy: `RUNBOOK_DB_LOCAL_V10.md` E2, or (only if you accept that) `--db-mode snapshot` |
| `REFUSED: another run holds .../.migrate/lock` | 2 | a run is live or died | `pgrep -af pod-migrate`; if nothing runs, `rm $NEW/.migrate/lock`, then `status` |
| `REFUSED: --old-pod-stopped is required` | 2 | you did not confirm the freeze | freeze the old pod (5.3), add the flag |
| `FAILED: the source tree changed while the cutover ran` | 1 | the old pod was not frozen | stop it properly and re-run `cutover --apply` |
| `INTERRUPTED by SIGTERM/SIGINT` | 1 | the run was killed; the manifest says incomplete | start nothing; re-run the same command |
| `database copy attempt N/5 failed (...)` | (retries) | the old pod (or the volume) had the database busy, or the copy failed its integrity check | it retries; if it gives up (`failed integrity_check on all N attempts`, exit 1), retry when she is not chatting |
| `database copy attempt N/5 (gentle): the source was written under the copy` | (retries) | someone is using the old pod: every write restarts the copy | it retries with fast steps, then a single copy; if it gives up (`the source database kept changing under the copy`, exit 1), have her close the app on EVERY device and run it again |
| `not trying the single full copy: it would hold the old pod's read lock for about N s` | (retries) | the database is too big to copy in one lock hold under the old pod's 5 s busy_timeout | the app must be closed everywhere; then re-run the seed |
| a killed run left `.migrate/lock` | | SIGKILL or a power cut during the run | `pgrep -af pod-migrate`; if nothing runs, `rm $NEW/.migrate/lock`, run `status`, re-run the same command (a kill mid-install is recognised: no false "new pod has written" refusal) |
| `verify` FAIL on `file rows in the target` | 1 | a path prefix the tool did not know | `--path-map OLD=NEW` (repeatable), then re-run |
| `verify` FAIL on a `tree '...'` digest | 1 | a file differs between the trees | re-run `cutover --apply` (it re-hashes and re-copies); if it persists, do not switch her over |
| `status`: INCOMPLETE | 1 | a run did not finish, its verify failed, or a run holds the lock | start nothing; re-run the same cutover command |
| `status`: NO MANIFEST | 2 | nothing was seeded here | wrong `--to`, or step 3 never ran |
| `webuidb.py --sync-once --force` exits non-zero | non-0 | it could not sync (it WAITS for the lock, then gives up) | do not stop the pod: `RUNBOOK_DB_LOCAL_V10.md` E1 |
| `backfill-records.py` exit 3 (dry run) | 3 | a record would resume: expected once | `--apply` (6.5) |
| `backfill-records.py` exit 1 | 1 | a record needs a human (`needs-review`) or a refused apply | read the reason line; ask |
| `webuidb-sync` log: `refusing to publish: snapshot reports 0 chats` | | expected on the empty first boot's stop | nothing |
| `webuidb-sync` log: `REFUSING to publish ... OLDER` | | the local DB is older than the snapshot: do not stop the pod | `RUNBOOK_DB_LOCAL_V10.md` E1 |

**The rule for every other tool on the new pod:** point it at `/var/lib/openwebui/webui.db` (the live file) and the `/data/v10/...`
paths; never at `/data/openwebui`. The tools now default to this pod's own paths even from a shell without the pod's
environment, but say the path anyway.
