# The pod ops bundle (`zl` / `zlpod`)

"If I have to perform an action more than once I want it repeatable." This
directory turns the manual procedures in [COMMANDS.md](../COMMANDS.md) and
[OPERATIONS.md](../OPERATIONS.md) into one command per action, plus a live
terminal telemetry dashboard viewed inside the SSH session itself (no new
port, no image change).

**`zl` WRAPS the existing scripts; it never re-implements a repair script's
own logic.** Every fix under `zl fix` and `zl journal` runs the real,
unmodified script from `scripts/`; this bundle only picks the interpreter,
decides what to stop, confirms before it changes anything, and logs what
happened.

## What the bundle is

- `zl` -- the on-pod command (`ops/zl` + the `ops/zlops` Python package).
- `zlpod` -- the laptop-side helper (`ops/laptop/zlpod`), a thin `ssh`/`scp`
  wrapper around a `Host zlpod` entry in `~/.ssh/config`.
- `zl bundle` -- a `zla-bundle-<ts>.tar.gz` diagnostic snapshot (health,
  supervisorctl status, GPU, disk, a redacted env dump, telemetry, versions,
  and the service logs) -- the thing `zlpod pull bundle` fetches.

## First-time setup

**On image v3.1.9.8 and later, this clone-and-bootstrap one-liner is NOT
needed at all** -- the image already bakes `ops/`/`scripts/` into
`/opt/zl-repo` and boots `zl` automatically (see "After any restart"
below); just wait for the pod to come up, then skip straight to step 2.
The steps below are for an image OLDER than v3.1.9.8, which starts with
nothing at `/opt/zl-repo` at all.

**The ref you want on the pod must already be pushed to GitHub** -- the pod
clones over the network; a local, unpushed branch is invisible to it.

1. Open the pod's RunPod **Web Terminal** and run:
   ```bash
   git clone --depth 1 --branch <REF> https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
   bash /opt/zl-repo/ops/bootstrap.sh --ref <REF> --with-sshd
   ```
   This installs `/usr/local/bin/zl`, saves the ref under `/data/zl-ops`
   (so it survives a restart), adds the `zl status --brief` login hook, and
   (via `--with-sshd`) installs and hardens `openssh-server`
   (`scripts/setup-sshd.py`) so you can leave the Web Terminal.
2. On your laptop:
   ```bash
   ops/laptop/zlpod init <pod-ip> <pod-ssh-port>
   ```
   (`zl sshd --apply` prints the exact ip/port from `RUNPOD_PUBLIC_IP`/
   `RUNPOD_TCP_PORT_22`, or "see the RunPod Connect tab" if those aren't
   set, plus this exact `zlpod init` command line.)
3. `zlpod ssh`, `zlpod dash`, or `zlpod zl status` all work from here on.

## After any restart

**On image v3.1.9.8 and later: nothing to do, PROVIDED `--with-sshd` was
already saved once.** The image itself bakes `ops/`/`scripts/` into
`/opt/zl-repo` (Dockerfile) and supervisord runs a one-shot
`[program:zl-ops]` (`ops/bootstrap.sh --from-dir /opt/zl-repo --boot`,
priority 5, no dependency on vllm/compactor/openwebui) on every container
start, including after a restart. It never prompts, never reads stdin,
and does NO ref pinning at all -- it always uses the baked copy exactly
as the image ships it, no clone, ever, even if `/data/zl-ops/ref` names a
different ref from an earlier interactive install (that is only reported
as one informational log line -- "saved ref '...' not applied at boot" --
update the image, or run `zl update` by hand, to actually move to it). It
honours whatever FLAGS were saved from the first install (`--with-sshd`
re-installs/hardens `openssh-server`, bounded and fail-soft if the network
or apt isn't ready yet; `--with-tools` only runs at boot if `--tools-at-boot`
was also saved, and always last) but never rewrites them -- --boot only
ever READS $ZL_HOME/ref and $ZL_HOME/bootstrap-flags, time-bounded, and
leaves both untouched either way -- all within a fixed wall-clock budget,
after which it gives up and exits 0 regardless (never blocks vllm/
compactor/openwebui coming up; `supervisorctl status` shows `zl-ops` as
`EXITED`, never `FATAL`/`BACKOFF` -- `zl status`/`zl dash` already treat
that as normal, same as `selftest`). Its own log is
`$LOG_DIR/zl-ops.log`, including the exit code and last ~5 lines of any
`zl sshd`/`zl tools` step's own output, and (on a genuine mid-script
failure) the line and command where it aborted. If `--with-sshd` was
saved, `zlpod ssh`/`zlpod dash` work again once that log shows a fresh
`connect: ssh ...` line; if it wasn't, run `zl sshd --apply` by hand from
the Web Terminal as below.

**Saving `--with-sshd` for the FIRST time on a fresh v3.1.9.8+ pod** still
needs one manual, interactive run from the Web Terminal (there being no
flags yet for `--boot` to read back) -- the baked image is already at
`/opt/zl-repo`, so this is just:
```bash
bash /opt/zl-repo/ops/bootstrap.sh --from-dir /opt/zl-repo --with-sshd
```
run once, interactively (it will prompt/confirm as usual). From then on,
every `--boot` on this pod's `/data` volume re-applies `--with-sshd`
automatically, as described above.

**On an image OLDER than v3.1.9.8** (no baked `/opt/zl-repo`, no
`[program:zl-ops]`), the manual one-liner below still applies. The
container overlay (and `/opt`) resets on every pod restart; `/data` (and
everything `zl` saved there) survives. **`openssh-server` is part of that
overlay too, so it is gone immediately after a restart -- ONLY the RunPod
Web Terminal works until this command has re-installed it.** Run it
there, not over `zlpod ssh` (which cannot connect yet):
```bash
bash /data/zl-ops/bootstrap.sh
```
It re-clones `/opt/zl-repo` at the saved ref if needed, then hands off to
that clone's own `ops/bootstrap.sh` -- so a newer ref's bootstrap code
always wins over a stale saved copy, and (if `--with-sshd` was saved from
the first install) re-runs `zl sshd --apply` too. Only ONCE that has
finished and printed a fresh `connect: ssh ...` line does `zlpod ssh` /
`zlpod dash` / `zlpod bootstrap` work again -- `zlpod bootstrap` is for a
LATER restart's remote convenience, after the very first post-restart
bootstrap has already run from the Web Terminal and sshd is back up. If
`--with-sshd` was never saved, run `zl sshd --apply` again by hand
from the Web Terminal.

## Every `zl` command, and the manual procedure it replaces

| command | replaces | read-only? |
|---|---|---|
| `zl status [--brief]` | COMMANDS.md "Pod -- health and diagnosis" (`supervisorctl status` + the health curl-and-parse one-liner) | yes |
| `zl health [--json\|--raw]` | OPERATIONS.md "Is it healthy right now?" / "Reading /health/full" | yes |
| `zl selftest` | OPERATIONS.md "Deeper, on demand" (`/admin/selftest`) | yes |
| `zl logs [...] [--req ID]` | `scripts/tail-logs.sh` directly | yes |
| `zl dash [...]` | `tail -f` several logs + repeated health/status checks by hand | yes |
| `zl disk` | COMMANDS.md's `du -sh /data/*` / OPERATIONS.md "Disk is filling up" | yes |
| `zl bundle [--out DIR] [--keep N] [--print-path] [--req ID]` | manually collecting health.json/supervisorctl.txt/gpu.txt/disk.txt/logs by hand | writes a tarball only |
| `zl backup now\|list\|verify` | OPERATIONS.md "Backups" (`backup.py --once/--list/--verify`) | `now` writes a backup; `list`/`verify` read |
| `zl backup restore` | refuses -- see below | n/a |
| `zl journal check\|repair [--apply]` | OPERATIONS.md "A HOT SQLite rollback journal" (`scripts/recover-webui-db.py`) | `check` only |
| `zl sshd [--apply]` | OPERATIONS.md "Getting a real shell into a pod" (`scripts/setup-sshd.py`) | no |
| `zl tools [--apply]` | manually `apt-get install`ing tmux/htop/sqlite3/iproute2/less | no |
| `zl fix <name> [--apply]` | the individual repair-script procedures in OPERATIONS.md/RUNBOOK_*.md -- `zl fix` lists every entry with its source citation | depends on the entry |
| `zl shadow up\|status\|logs\|down` | COMMANDS.md "Shadow deploy" | `up`/`down` write |
| `zl wordban-check` | scripts/wordban/README.md "Steps for the owner", step 3 | yes |
| `zl models [...]` | `/opt/clean-models.sh` directly (its own dry-run default) | depends on its flags |
| `zl version` | manually checking the ref/commit/build id/package versions | yes |
| `zl history [-n N]` | reading `/data/zl-ops/audit.log` by hand | yes |
| `zl update [--ref REF]` | manually `git fetch`/`checkout` in `/opt/zl-repo` | no |

`zlpod` commands: `init`, `ssh`, `dash`, `zl ARGS...`, `status`, `pull bundle`,
`pull backup [--latest\|NAME]`, `push wordban [DIR]`, `bootstrap`, `help`.

- **`zl logs` is raw and unscrubbed** (review LOW): it is a direct
  passthrough to `scripts/tail-logs.sh`, the same as running it by hand --
  it does NOT go through `zlops.privacy`'s scrubber. That scrubber only
  ever runs on `zl bundle`'s copied-out log files and `zl dash`'s own log
  panel. `zl logs` is for an operator already looking at the pod's real
  logs in a live SSH session, same as `tail -f` always was; it is not
  something to paste elsewhere without checking it first.
- **`zl logs --req ID` / `zl bundle --req ID`** (round 3, v3.1.9.8 request
  tracing): every compactor log line belonging to one request carries
  `req=<hex id>`, and its internal vLLM calls carry child ids
  (`req=<hex>.sum1`, `.facts`, `.dedup`). `--req ID` restricts the output
  (or, for `bundle`, every copied log file) to just that one request's own
  lines -- `req=ID` and any `req=ID.xxx` child id, never a different
  request that merely shares a prefix. This is `zl`'s own filter, layered
  on top of `scripts/tail-logs.sh` (which is never modified for it); `zl
  logs --req` runs it in `--all` mode first so a request's routine lines
  are not dropped by the default signal/noise split before the `--req`
  filter ever sees them. `zl bundle --req ID` applies the filter AFTER the
  usual privacy scrub (never instead of it) and records the filter in
  `meta/req-filter.txt`. The match is on the EXACT id (or `id` + a
  `.child` suffix) -- a real id may itself contain hyphens
  (`reqctx._ID_RE` in the compactor), so `req=abc` never matches inside
  `req=abc-def`, a different id that merely shares a prefix. `zl logs
  --req` also reads each known log file directly first, for whatever
  history is older than `tail-logs.sh`'s own last-20-lines-per-file
  window, so `--req` sees more than just the last 20 lines. Matches the
  plain-text log line shape only -- a `COMPACTOR_LOG_FORMAT=json` pod's
  structured lines are not narrowed by this per-line filter (its
  logs are still fully captured by `zl bundle --req`, just unfiltered).
- **`zl journal repair`'s local-mode staleness** (review MEDIUM): on a
  `WEBUI_DB_LOCAL=true` pod, this wrapper stops `webuidb-sync` alongside
  `compactor`/`backup` and restarts it after a successful repair, but
  `webuidb-sync`'s own publish loop sleeps first, then publishes (v3.1.9
  `webuidb.py`'s sync loop) -- so `/data`'s snapshot of the just-repaired
  database stays STALE for up to `WEBUI_DB_SYNC_INTERVAL_S` (300s
  default) after the repair. If a backup or a pod stop happens in that
  window, it captures the stale snapshot, not the repair. `zl journal
  repair --apply`'s own output says this and names the documented
  one-shot command to publish immediately: `webuidb.py --sync-once
  --force` (its own `--force` only skips the mtime check -- the same
  liveness/integrity guards apply either way, so this is safe to run with
  services up, per `webuidb.py`'s own module docstring).

## The dashboard

`zl dash` (or `zlpod dash`) is a curses dashboard inside the SSH session
itself: health, services, GPU, host/data/backups, vLLM throughput, the SSH
session's own liveness, and a filtered log-signal panel. `zl dash --once`
prints one snapshot and exits (script-friendly); `zl dash --json` prints the
same snapshot as JSON (`zlops.telemetry.collect_snapshot()`, the same
function `zl bundle` uses for `meta/telemetry.json`). No new port: it reads
`/health/full`, `supervisorctl`, `/proc`, and `nvidia-smi`, all from inside
the same SSH session already open.

## Safety model

- **Dry run by default, for the commands that gate on `--apply`.** Most
  commands that change anything only show what they WOULD do until you add
  `--apply`: `zl fix`, `zl journal repair`, `zl sshd`, `zl tools`, `zl
  bundle --raw-logs` (the confirmation, not the bundle itself). A few do
  NOT gate on `--apply` at all, and the table above marks each one: `zl
  bundle` (writing a diagnostic tarball is the whole point of running it),
  `zl backup now` (its own module docstring: live-SQLite-safe by design,
  no stop needed), `zl update` (a `git fetch`+`checkout` in a clone this
  bundle owns, with a dirty-checkout refusal instead), and `zl shadow
  up`/`down` (an isolated, disposable process/storage root, never her live
  memory -- see its own refusal guard).
- **Confirm before applying.** `--apply` asks you to type `apply`, unless
  `--yes` is also given. A non-interactive session (no TTY) without `--yes`
  refuses outright (exit 1) rather than silently going ahead.
- **Waits for her in-flight request(s) before stopping anything.** (round-4
  addendum, v3.1.9.8's request tracing.) Before `zl fix <name> --apply` or
  `zl journal repair --apply` touches a single service, it reads
  `/health/full`'s `request_tracing.requests_in_flight` and, if it is
  nonzero, waits (printing a "waiting for her in-flight request(s)"
  progress line, same cadence as the restart-side heartbeat) for it to
  read 0 for a full 5 seconds straight -- not just one lucky poll -- before
  stopping anything, so a restart can never kill a request mid-stream the
  way an incident this round found. Degrades safely, never blocks
  forever: a compactor that does not report the field at all (v3.1.9), or
  flatly refuses the connection (nothing there to have a request in
  flight), proceeds immediately; a timeout or a 5xx WITHOUT the field is
  treated as busy-status-unknown and retried like a real busy reading,
  never treated as "safe to proceed" on a guess. After 5 minutes of never
  settling, it refuses (exit 1) rather than waiting forever -- pass
  `--force-busy` to skip the wait outright (still logged/printed) when you
  know better than the health check. A re-check of the SAME wait happens
  again immediately after the backup step below settles (round-6 BLOCKER:
  her next request can arrive WHILE `backup.py --once` itself is still
  running -- the first check alone only proves nothing was in flight the
  moment the backup started), so nothing is stopped until both checks
  agree it is safe.
- **Backup first.** `zl fix <name> --apply` runs `compactor/backup.py
  --once` before touching anything (after the in-flight wait above settles,
  not before it), and refuses to continue if that backup fails, unless
  `--skip-backup` is given.
- **What a dropped connection (SIGHUP) vs. Ctrl-C (SIGINT/SIGTERM) does,
  phase by phase.** The whole `--apply` flow -- BOTH in-flight waits,
  the backup, the stop, the wrapped script's own run, and the
  restart-and-verify step at the end (which can itself take up to three
  minutes waiting for OpenWebUI to answer after a database repair -- `zl`
  prints a "do not disconnect" progress line roughly every 10s while it
  waits) -- is protected, but HUP and INT/TERM are NOT treated the same,
  and the table below is deliberately per-phase because round-5 found
  the previous, single blanket sentence here actively wrong:
  - **SIGHUP (a real dropped connection, INCLUDING sshd actually closing
    the pty, not just `kill -HUP`)**: ignored by `zl` itself in EVERY
    phase, with one exception the round-7 hostile review found
    (confirmed with `pty/run_hup_backup.sh`): **the backup phase**
    (`compactor/backup.py --once`, launched with plain `run_stream`, not
    `run_protected`) runs as an ordinary child in `zl`'s OWN foreground
    process group -- unlike the wrapped repair script, it is never given
    its own session. A real pty close delivers SIGHUP to backup.py
    directly, right along with `zl`, killing it; `zl` sees the failed
    backup the same as any other backup failure and refuses to continue
    -- `EXIT_BACKUP_FAILED`, exit 5, and NOTHING is applied (nothing has
    been stopped yet at that point either). In every OTHER phase (both
    in-flight waits, the stop, the wrapped script's own run once
    `run_protected` has it in its own session, and the restart-and-verify
    step), SIGHUP is ignored outright and `zl`'s own stdout/stderr are
    dup2'd onto `/dev/null` (there is no terminal left to write to; every
    print from here on goes there silently, and to the run log under
    `ZL_HOME/runs`, the durable record once the operator can no longer
    see anything) -- the run continues COMPLETELY UNATTENDED through
    whichever phase it was in and, at the end, restores services exactly
    as any other successful run would.
  - **SIGINT/SIGTERM (Ctrl-C, or a terminated SSH session)** -- behaviour
    depends on the phase:
    - **During either in-flight wait**: ABORTS. Nothing has been stopped
      yet, so there is nothing to restore; `zl` refuses (exit 1) rather
      than going on to stop/apply anyway.
    - **During the backup itself**: also aborts, but which exit code
      depends on where the signal actually lands, for the same reason as
      the SIGHUP exception just above -- backup.py is a plain child in
      `zl`'s own foreground process group, not its own session:
      - Ctrl-C at a real terminal (SIGINT delivered to the whole
        foreground process group) reaches backup.py directly too and
        kills it the same way a pty close does -- `zl` exits 5
        (`EXIT_BACKUP_FAILED`), same as any other backup failure.
      - A signal sent to `zl`'s own pid only (e.g. `kill <pid>`, not the
        group -- as a terminated SSH session's SIGTERM to just the
        foreground job would NOT be, but a supervisor sending SIGTERM to
        one specific pid could be) never reaches backup.py: the backup
        runs to completion, and the signal is only acted on at the next
        explicit check afterward (the post-backup in-flight wait's own
        `deferred_signal` check); `zl` aborts there instead, exit 1,
        having stopped nothing.
      Both paths confirmed with a round-7 scratch probe (SIGINT to the
      whole group vs. SIGTERM to `zl`'s pid alone).
    - **During the stop itself, or in the (brief) gap after the stop has
      finished but before the wrapped script has actually started**
      (round-6 REVERSAL -- this used to let the script run anyway):
      ABORTS too. Whatever was stopped is put back, and `zl` prints
      "cancelled before the fix was applied; services restored" and
      exits `128+signal` -- the wrapped script never runs at all. Round-7
      note: this covers everything from the moment `stop_for_apply`
      returns, checked explicitly by `fix.py`/`journal.py` before ever
      calling `run_protected`. A narrower sliver remains INSIDE
      `run_protected` itself, between entering the function and `Popen`
      succeeding (`region.set_child` only registers the running child
      immediately after) -- a signal landing in that specific sliver
      (much less than one Python statement's worth of wall-clock time)
      is not caught by an explicit abort check the way the wider window
      around it is, and is deferred instead: the wrapped script goes on
      to run to completion, exactly as a restart-phase signal is
      handled. Left as-is rather than closed (the alternative would be
      moving the check to right before `Popen` inside `run_protected`):
      it happens almost never, and if it ever does land there, the fix
      still completes and the output says so (`... ran and exited N` /
      `WAS APPLIED`) -- nothing is silently lost either way.
    - **While the wrapped script is actually running**: forwarded to it
      instead (its own docstring says an interrupt is safe) -- the one
      exception is `zl journal repair`, which never forwards a signal to
      `recover-webui-db.py` at all (RUNBOOK_DB_JOURNAL.md says not to
      interrupt its database swap); it prints "repair in progress; will
      not interrupt" and the signal is acted on only once the repair
      itself has finished.
    - **During the restart-and-verify step, once the script HAS run to
      completion**: deferred, not dropped -- the restart still runs to
      completion and services are restored (or left stopped, for
      `zl journal repair`'s own leave-stopped exit codes) exactly as an
      uninterrupted run would. If the underlying script (or `zl`'s own
      refusal) returned a NONZERO code, `zl` exits with THAT code
      (printed and audited, with the deferred signal named alongside
      it) -- a signal never hides a real failure. Round-7: the note
      attached to that message differs by code, because a nonzero code
      is not necessarily a plain failure (some wrapped scripts use a
      specific nonzero code to mean "safe, or required, to re-run" -- see
      EXIT_MEANINGS) -- an unconditional "do not re-run it" would flatly
      contradict a script's own documented exit-code meaning. So: a
      NONZERO code gets a plain factual note ("the fix ran and exited
      N"), and EXIT_MEANINGS' own explanation of N stands on its own,
      unqualified either way. Only when the code was 0 does `zl` exit
      `128+signal` instead (so `$?` reflects what interrupted it), and
      only THEN is "WAS APPLIED (exit 0); do not re-run it" printed --
      the one case where that claim cannot contradict anything, since 0
      carries no script-specific meaning of its own.
- **Services put back exactly as found.** Before stopping anything, the
  wrapper records which services were actually RUNNING; afterward it starts
  exactly those back (never one that was already stopped, e.g.
  `webuidb-sync` on a `WEBUI_DB_LOCAL=false` pod) and verifies each one
  came back. `zl` says "restored" ONLY when every one of them actually did
  (round-6 HIGH) -- if even one did not, it says so LOUDLY, names the
  exact `supervisorctl start ...` command to run by hand, and exits
  nonzero even if the wrapped script's own exit code was 0. `supervisorctl
  stop`/`start` themselves run in their own session (`start_new_session`),
  so a Ctrl-C at the terminal (which targets the whole foreground process
  group) cannot kill them mid-command the way it could kill an ordinary
  child.
- **Audit log.** Every invocation, read-only included, appends one JSON
  line to `/data/zl-ops/audit.log`: UTC time, the exact command line, the
  SSH client IP (or `web-terminal`), the exit code, and how long it took.
  `zl history` reads it back. A failure to write the log (a MooseFS stall)
  never aborts the action itself -- it only warns.
- **Exit codes.** `0` done/nothing to do, `1` refusal or error, `2` usage
  error, `3` dry run found pending work (informational -- never folded into
  0 or 1), `4` partial apply (re-run). `zl health`'s own convention: `0` ok,
  `1` crit, `3` warn (documented in `ops/zlops/commands/health.py`, since
  the brief left the warn/crit split to this bundle's own judgment). A
  wrapped script's exit code always passes through unchanged; `zl` prints
  what it means afterward. `zl journal repair`'s own 4/6/7 are the ONE
  exception to the generic `4` text above: `recover-webui-db.py`'s
  RUNBOOK_DB_JOURNAL.md section 6 codes mean "writers left stopped on
  purpose, do NOT re-run" -- the opposite of the generic wording -- so
  `zl` prints that correction instead, along with openwebui's real
  per-code state (already restarted on 4; still down on 6/7).

## What is deliberately NOT automated, and why

- **`zl backup restore` always refuses.** Automated restore is out of
  scope: [OPERATIONS.md's manual procedure](../OPERATIONS.md#-restore-from-a-backup-recover-lostcorrupted-memory)
  exists precisely because `backup.py --restore` is unsafe on some
  versions of this codebase (v3.1.6.1-v3.1.8) and not yet fully reviewed
  even on v3.1.9. Follow the manual procedure, which stops all four
  writers first: `supervisorctl stop openwebui compactor backup
  webuidb-sync`.
- **Image rollback is not a `zl` command.** The image is pinned to whatever
  RunPod's template says (production currently runs
  `angreg/zions-light-ai:v3.1.9-cu12`, digest `sha256:f3fafa3e1949...` --
  see `memory.md` "Current state" for the current value); swapping it is a
  RunPod template change, not something this bundle should be able to
  trigger by accident from inside a running container.
- **Pasting the word-ban release into OpenWebUI is manual.** `zl
  wordban-check` only runs the on-pod verification script; actually
  replacing the `structured_outputs` value and the system-prompt line in
  the Admin Panel is a human paste, on purpose (`scripts/wordban/README.md`
  "Steps for the owner") -- there is no OpenWebUI admin API this bundle
  should be driving unattended.

## The ref must be pushed to GitHub before the pod can clone it

`git clone`/`git fetch` in `bootstrap.sh`/`zl update` talk to
`https://github.com/MrBanana8768/zions-light-ai.git` (or `--repo-url`) over
the network -- a commit that only exists in a local worktree is invisible
to the pod. Push first, then bootstrap or `zl update --ref <ref>`.
