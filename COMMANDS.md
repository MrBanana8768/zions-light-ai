# Commands

Everything routinely run on this project, in one place. Two audiences: the
laptop (tests, analysis) and the pod (operations, incidents).

**On the pod, read the incident sections in [OPERATIONS.md](OPERATIONS.md)
before running anything destructive.** The commands here are the shape; that
file carries the reasoning and the traps.

---

## Testing — all of it runs on Linux

The production image is Ubuntu 24.04. A host run on Windows is a development
convenience, not evidence: the asyncio loop differs (Proactor vs epoll/uvloop,
which is the whole of the client-disconnect question), and Windows resolves
`localhost` to the IPv6 loopback and then the IPv4 one against separate 2 s
timeouts, which once made a suite look like it took 239 seconds.

### Unit suite

```bash
docker compose -f docker-compose.tests.yml run --rm --build unit-tests
```

```bash
# with the 200-exchange saturation run
docker compose -f docker-compose.tests.yml run --rm unit-tests --saturation
```

```bash
# one suite, by substring
docker compose -f docker-compose.tests.yml run --rm unit-tests --only summarizer
```

```bash
# skip the slow ones
docker compose -f docker-compose.tests.yml run --rm unit-tests --fast
```

Exit codes: `0` pass, `1` fail, `2` runner error, **`3` something was SKIPPED**.
A skip is never folded into a pass — three suites once exited 0 without running
their checks, and every "all suites pass" in that period excluded the tokenizer
contract.

### Tokenizer contract and soak (need their own fixture stack)

```bash
docker compose -f docker-compose.tokenizer-contract.yml up --build --exit-code-from contract-tests
```

```bash
docker compose -f docker-compose.tokenizer-contract.yml run --rm --build soak-tests
```

### Integration suite (black box, local stack)

**Recreate the compactor first.** It serves the working tree from a read-only
mount and picks it up only at container start; `run` will start a dependency
that is stopped but will NOT recreate one that is already up. That is how this
stack once served code 49 minutes older than the commit it was verifying.

```bash
docker compose -f docker-compose.integration.yml up -d --force-recreate compactor
```

```bash
docker compose -f docker-compose.integration.yml run --rm --no-deps integration-tests
```

```bash
# one file or any pytest argument
docker compose -f docker-compose.integration.yml run --rm --no-deps integration-tests -k regression_tail -v
```

### Integration with a REAL model (opt-in, slow)

A 0.5B instruct model, GGUF, CPU only. Minutes rather than seconds, because
generation is real.

```bash
docker compose -f docker-compose.integration.yml --profile model run --rm --build integration-tests-model
```

### Adversarial suites — one isolated stack per lane

Each `-p` project gets its own compactor, fixture and storage volume. They can
run at the same time and cannot see each other.

```bash
docker compose -p adv-fuzz -f docker-compose.adversarial.yml run --rm adversary tests/adversarial/test_adv_input.py -v
```

```bash
docker compose -p adv-race -f docker-compose.adversarial.yml run --rm adversary tests/adversarial/test_adv_race.py -v
```

```bash
docker compose -p adv-faults -f docker-compose.adversarial.yml run --rm adversary tests/adversarial/test_adv_faults.py -v
```

```bash
docker compose -p adv-state -f docker-compose.adversarial.yml run --rm adversary tests/adversarial/test_adv_state.py -v
```

**Tear down with `-v`** or the next run inherits a poisoned store:

```bash
docker compose -p adv-fuzz -f docker-compose.adversarial.yml down -v
```

### Mutation testing — the project standard

A test only counts if it has been watched to fail. Apply a mutation, run the
suite, watch it go red, restore, watch it go green.

```bash
python scripts/mutate.py   # if present; otherwise the pattern below
```

The pattern, when scripting it: read the file with `newline=""`, replace one
anchor, run, restore from the in-memory copy in a `finally`. **Never** use git
to restore — a `git checkout --` has previously discarded another agent's
in-flight work.

---

## Log and backup analysis (laptop)

Bundles arrive as `zla-bundle-<ts>.tar.gz` (logs) and
`zions-backup-<ts>.tar.gz` (store + `webui.db`).

```bash
tar -xzf zla-bundle-<ts>.tar.gz -C ./scratch/logs && tar -xzf zions-backup-<ts>.tar.gz -C ./scratch/bk
```

```bash
# the health snapshot the pod took with the bundle
python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['status'], d['status_reasons']); print(d['stats'])" ./scratch/logs/*/meta/health.json
```

```bash
# what actually went wrong, grouped
grep -aoE "(ERROR|WARNING) .{0,70}" ./scratch/logs/*/logs/compactor.log | sed 's/conv=[0-9a-f]*/conv=X/g; s/[0-9]\+/N/g' | sort | uniq -c | sort -rn | head -20
```

```bash
# attribute a symptom to a conversation (the compactor logs conv_id on its own line)
python -c "
import re,sys,collections
cur=None; hits=collections.Counter()
for ln in open(sys.argv[1],encoding='utf-8',errors='replace'):
    m=re.search(r'conv_id=([0-9a-f]{8})',ln)
    if m: cur=m.group(1)
    if sys.argv[2] in ln: hits[cur]+=1
print(dict(hits))
" ./scratch/logs/*/logs/compactor.log "hard budget FAILED"
```

### The repetition metrics (F2)

Pod logs carry no reply text, so this needs the backup's `webui.db`.

```bash
python scripts/chat-metrics.py ./scratch/bk/webui.db
```

Reports opening-phrase frequency and n-gram overlap between consecutive
replies — the two numbers F2 asked for and could not get from logs.

---

## Pod — health and diagnosis

```bash
curl -s localhost:8080/health/full | python3 -m json.tool | head -40
```

```bash
supervisorctl status
```

```bash
df -h /data; du -sh /data/* 2>/dev/null | sort -h | tail -10
```

```bash
# the compactor's storage root is /data/openwebui/compactor, NOT /data/compactor
ls -la /data/openwebui/
```

`sqlite3` is not installed on the pod. Use the venv's Python:

```bash
/opt/compactor-venv/bin/python -c "import sqlite3; c=sqlite3.connect('/data/openwebui/webui.db'); print(c.execute('PRAGMA quick_check').fetchone()[0])"
```

### The hot-journal check

```bash
/opt/compactor-venv/bin/python -c "
import os
j='/data/openwebui/webui.db-journal'
b=open(j,'rb').read(8) if os.path.exists(j) else b''
print('HOT - uncommitted transaction pending' if b.hex()=='d9d505f920a163d7' else 'clean')"
```

Since v3.1.8 `/health/full` reports this itself under
`checks.sqlite_journal` and degrades on it, so on a current build:

```bash
curl -s localhost:8080/health/full | python3 -c "import json,sys; print(json.load(sys.stdin)['checks']['sqlite_journal'])"
```

---

## Pod — recovery

### A hot rollback journal

```bash
supervisorctl stop openwebui
```

```bash
/opt/compactor-venv/bin/python -c "import sqlite3; c=sqlite3.connect('/data/openwebui/webui.db'); print('journal_mode:', c.execute('PRAGMA journal_mode').fetchone()[0]); print('quick_check:', c.execute('PRAGMA quick_check').fetchone()[0]); c.close()"
```

```bash
supervisorctl start openwebui
```

Opening it read-write with the writers stopped IS the repair: SQLite rolls the
journal back itself and deletes it. **Never delete the journal by hand** — it
and the database are a matched pair, and separating them turns a recoverable
file into a corrupt one.

### Backups

```bash
ls -1t /data/backups/ | head
```

```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --verify
```

```bash
supervisorctl stop openwebui compactor backup
/opt/compactor-venv/bin/python /opt/compactor/backup.py --restore /data/backups/<archive>.tar.gz --yes
supervisorctl start compactor openwebui backup
```

Restore is destructive and refuses an archive that does not verify.

### Shadow deploy (test a build beside the live compactor)

```bash
git clone -b <branch> <repo-url> /opt/compactor-shadow
```

```bash
COMPACTOR_STORAGE_ROOT=/data/shadow-compactor VLLM_URL=http://127.0.0.1:8000 MODEL_REPO="$MODEL_REPO" MAX_MODEL_LEN="$MAX_MODEL_LEN" COMPACTOR_GENERATION_RESERVE="$COMPACTOR_GENERATION_RESERVE" /opt/compactor-venv/bin/uvicorn main:app --app-dir /opt/compactor-shadow/compactor --host 127.0.0.1 --port 8081 --log-level info > /tmp/shadow.log 2>&1 &
```

```bash
curl -s localhost:8081/health/full | python3 -m json.tool
```

```bash
pkill -f "port 8081"; rm -rf /data/shadow-compactor
```

**The separate `COMPACTOR_STORAGE_ROOT` is load-bearing** — point it at the
live root and the shadow writes into her memory. It shares the running vLLM,
so its generations queue alongside real traffic: fine for a few curl calls,
not for a soak.

---

## Git, when more than one agent is working

```bash
git add <explicit> <paths>
```

Never `git add -A` or `git add .` while anything else may be writing, and give
concurrent agents their own worktree rather than a shared one. Bundling two
agents' unfinished files into a third's commit has already happened once.

---

## Gotchas that have cost real time

**Do not pipe a command whose exit code you care about through `tail`/`head`** —
you read the pipe's status, not the command's. This produced three false
"success" readings in one session, including a Docker build that had failed.

```bash
some-command > /tmp/out.txt 2>&1; echo "EXIT=$?"; tail -5 /tmp/out.txt
```

**Windows Git Bash rewrites absolute container paths.** Prefix docker commands
that pass one with `MSYS_NO_PATHCONV=1`.

**Heredocs mangle backslash escapes.** `\r`, `\n` and `\x00` written inside a
`<<'PY'` block have repeatedly landed as literal control characters in source
files. Write the script to a file and run it, or use the editing tools.

**The real interpreter on the dev machine** is
`C:\Users\rngge\AppData\Local\Programs\Python\Python314\python.exe`; a bare
`python` is a Store stub that reports itself absent.
