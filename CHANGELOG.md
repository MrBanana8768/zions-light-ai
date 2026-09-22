# Changelog

All notable changes to Zion's Light AI. Format inspired by
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Image tags published at
[`angreg/zions-light-ai`](https://hub.docker.com/r/angreg/zions-light-ai)
on Docker Hub.

---

## [3.1.9.6] — closing stale backfill records, catching a summary
hierarchy up from webui.db, and installing sshd into a running pod

**Scripts and docs only, and NOT a new image.** Every file the image copies
(`compactor/*.py` apart from tests, `stt/`, `tts/`, `entrypoint.sh`,
`supervisord.conf`, `clean-models.sh`) and the `Dockerfile` are
byte-identical to v3.1.9.5, which was itself byte-identical to v3.1.9.4. So
`:v3.1.9.6-cu12` is published as a third tag on the existing
`:v3.1.9.4-cu12` image, digest
`sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65`
(RUNPOD_DEPLOY.md Step 3), not rebuilt, for the same reason v3.1.9.5 was
not rebuilt: a rebuild re-resolves `apt-get upgrade`, unpinned pip
dependencies, the Piper voice URL and the CUDA base tag, and produces a
different, unvalidated image.

**Why this exists.** Three independent operator gaps, each closed with a
script under `scripts/` rather than a change to what the image ships:

1. v3.1.9.4's own fix to `backfill.needs_backfill()` — reading a
   conversation's `facts/<conv>.backfill.json` RECORD before deciding
   whether it needs a lazy history backfill, rather than stopping the
   instant a facts file exists — is correct (a stale record on a
   conversation that also had live facts used to be ignored forever,
   silently), but it turns every upgrade of a pod that has ever run
   v3.1.9.3 or earlier into the TRIGGER: every stale `in_progress` or
   backed-off `failed` record already on the volume resumes the moment
   its conversation is next used. Verified on a real 2026-09-22
   production backup: four conversations carry a stale `in_progress`
   record (6/1908, 296/793, 572/732 and 589/626 exchanges) that v3.1.9.4+
   would resume all at once — roughly 2,600 background vLLM extraction
   calls competing with her live chat on the pod's one GPU.
   RUNPOD_DEPLOY.md's "Upgrading within v3.1.9.x" already named this
   cost; nothing before this release gave an operator a way to close
   those records ahead of time short of hand-editing JSON on the volume.
2. The production conversation `ea1494ea-e9d7-46fb-8b7c-3a50d685d00e`
   (the same id, coincidentally, that carries one of the four stale
   backfill records above) has roughly 3,850 messages in OpenWebUI, but
   its summary hierarchy only covers the first ~1,600 turns — a ~2,200
   turn backlog that drags the whole history around on every request and
   blocks turning on the OpenWebUI History cap. `POST
   /admin/conversations/{conv_id}/compact` runs the same rollup drain
   but rebuilds its transcript from the EPISODIC store (chromadb), which
   for this conversation holds only 70 exchanges — nowhere near enough to
   close a gap this size. The full transcript exists only in OpenWebUI's
   own `webui.db`.
3. The image has never shipped an SSH server (`Dockerfile:63-80`; never
   has), and `supervisord.conf` has no `[include]` section, so a `.conf`
   file dropped into `/etc/supervisor/conf.d/` on a running pod is
   silently ignored. Nothing before this release gave an operator a real
   shell into a live pod other than the RunPod Web Terminal — enough for
   the procedures above, but not for anything that needs a real
   interactive shell, port forwarding, or `scp`.

### Added
- **`scripts/backfill-records.py`.** Reads every `facts/*.backfill.json`
  record under a compactor store and classifies each one `leave` (already
  terminal — complete / abandoned / wiped), `would-resume` (v3.1.9.4+'s
  `needs_backfill()` will restart this), or `needs-review` (an
  `in_progress` record not yet stale, a `failed` one still inside its
  backoff window, or one this script could not read). It asks the REAL
  `compactor/backfill.py` module beside it for the verdict (`is_stale`,
  `_backoff_ready`, `_MAX_BACKFILL_ATTEMPTS`) rather than keeping a second
  copy of that logic, the same discipline `scripts/merge-conversations.py`
  already follows for `portability.merge_conversation`. Dry run (the
  default) only reports. `--apply` rewrites every `would-resume` record
  that also has a facts file to the terminal state `abandoned` — after
  backing up the original beside it, refusing outright if that exact
  backup path already exists — and prints exactly what it changed, one
  line per record. It refuses to close a `would-resume` record with NO
  facts file (that conversation has never had a fact extracted; closing it
  would cancel its one chance, not defuse a hazard) and refuses to run
  `--apply` at all while anything answers the compactor's `/health`; both
  refusals take `--force`. `--json` gives machine-readable output; `--conv`
  limits the run to specific conversations. It never touches
  `facts/<conv>.json`, `facts/<conv>.archive.json`, `summaries/`,
  `chromadb/` or `personas/`. See its own module docstring for the full
  exit-code contract (0 nothing-to-do/succeeded, 1 error, 3 dry-run-found-
  work), and `compactor/test_backfill_records_script.py` for coverage
  built from the four real stale records above — including the actual
  point of the tool: after `--apply`, `backfill.needs_backfill()` really
  does return `False` for all four.
- **`scripts/import-history.py`.** A one-shot operator tool that catches a
  conversation's L1/L2/L3 summary hierarchy up from a `webui.db` EXPORT
  (never the live database — refuses outright if a `-journal` or `-wal`
  sidecar sits beside the path given, unless `--force`). Reconstructs the
  LINEAR branch the user actually sees — OpenWebUI 0.11's `chat.chat` JSON
  stores every edit and regeneration as a tree, so this walks
  `history.messages` back from `history.currentId` via `parentId` and
  reverses it, rather than reading insertion order (which would include
  abandoned edits); falls back to the `chat_message` table when that JSON
  is missing or unreadable, and reports which source it used. A
  multimodal `content` array flattens to its text parts joined, with each
  image replaced by a `[image]` placeholder; non-alternating turns (a
  missing reply, two user turns in a row) are reported as anomalies and
  handled, never crashed on. Dry run (the default) makes NO vLLM calls at
  all and reports the source used, turns found, the existing watermark
  against what the transcript implies, how many L1/L2/L3 units are
  estimated due, an estimated (floor) vLLM-call count, and an ESTIMATED
  wall-clock. `--apply` backs up the existing `summaries/<conv_id>.json`
  beside itself before writing (refusing outright if that backup path
  already exists), then loops the REAL `compactor/summarizer.maybe_rollup`
  — the identical drain `/compact` runs, fed from this script's own
  transcript instead of the episodic store — under a `max_calls` budget
  with the same unit-boundary semantics `/compact` documents (an overshoot
  of at most one rollup unit, never more). Refuses `--apply` while
  anything answers the compactor's `/health`, unless `--force`. Never
  touches `facts/`, `chromadb/` or `personas/`, and never rebuilds the
  episodic/chromadb index — see OPERATIONS.md and the script's own module
  docstring for the full exit-code contract (0 nothing-due/succeeded, 1
  error, 3 dry-run-found-work) and for why `/compact` cannot do this job
  itself. `compactor/test_import_history_script.py` covers the branch
  walk against a forked (edited-message) history, multimodal flattening,
  the dry-run/apply/idempotent/budget-overshoot contracts, an interrupted
  and resumed `--apply`, and every refusal path.
- **`scripts/setup-sshd.py`.** Installs and hardens OpenSSH server LIVE
  inside a running container, without changing the image — necessary
  because the container filesystem resets on every pod restart, so this
  has to be re-run after each one. Dry run (the default) runs a real
  `apt-get update` (package lists only, nothing installed) and reports
  the installed/candidate `openssh-server` version, key source, and
  whether the drop-in or direct-edit case applies to THIS pod's real
  `sshd_config`. `--apply` installs it (or `--only-upgrade`s it to the
  apt candidate if already present — never `apt-get upgrade`/
  `dist-upgrade`, which would touch unrelated packages under a live vLLM
  process), resolves one usable PUBLIC key (`--authorized-key-file` >
  `--authorized-key` > the RunPod-injected `$PUBLIC_KEY` > an existing
  `authorized_keys`; every candidate validated with `ssh-keygen -l -f`,
  appended never clobbered, backed up first), writes
  `PasswordAuthentication no`, `PermitEmptyPasswords no`,
  `KbdInteractiveAuthentication no`, `ChallengeResponseAuthentication no`,
  `PubkeyAuthentication yes`, `PermitRootLogin prohibit-password` to a
  drop-in (or, if the pod's `sshd_config` does not support one —
  decided fresh every run — edits it directly with a backup), ensures
  host keys (`ssh-keygen -A`) and by default persists them to
  `/data/ssh/` so the pod's fingerprint survives a restart
  (`--no-persist-host-keys` opts out), then verifies the result for real
  with `sshd -t` then `sshd -T` — never just by reading back the file it
  wrote — refusing and rolling back if the effective config would allow
  password login. Starts sshd as a plain background daemon by default;
  `--supervise` (opt-in, OFF by default) instead adds it as a
  supervisord program, verified against a throwaway container running
  this image's real supervisord to start only the new program and leave
  every other one's pid and uptime untouched. Refuses outright if not
  root, if this does not look like the zions container (`--force`
  overrides only that check), or if no usable key exists anywhere. Never
  touches any other supervisord program, and finds sshd by its own
  pidfile — never a blanket `pkill` — to restart it. See its own module
  docstring for the full exit-code contract (0 success — including a
  no-op `--apply`; 1 refusal/failure; 3 a dry run found pending work —
  see "Fixed" below: this was briefly the inverse) and
  `compactor/test_setup_sshd_script.py`
  for coverage: a dry run that writes nothing, a clean install,
  idempotency, append-not-clobber, every refusal path, that `apt-get
  upgrade`/`dist-upgrade` is never invoked, and that no private key
  material ever reaches stdout, stderr, or `--json`.

### Documentation
- **OPERATIONS.md** gains "Closing stale backfill records before upgrading
  past v3.1.9.3", run **from a clone of this repo, not a copy** — the
  script imports the `compactor` package, so `HERE.parent/"compactor"`
  has to resolve to a real package, which a bare copy to `/data/scripts/`
  cannot supply. (This entry previously described that section as using
  the copy-to-`/data/scripts`-and-run-with-the-venv shape — the exact
  thing "Fixed" above calls a defect. Corrected here.) "Catching a
  conversation's summary hierarchy up from a webui.db export" follows
  right after it, with the real pod procedure (stop the compactor, dry
  run, `--apply`, restart, confirm `checks.hierarchy`'s lag falls on
  `/health/full`).
- **RUNPOD_DEPLOY.md**'s "Upgrading within v3.1.9.x, and rolling back" now
  tells an operator upgrading a pod that has ever run v3.1.9.3 or earlier to
  run this script (or set `COMPACTOR_BACKFILL_MAX_ATTEMPTS=0`) first, and
  flags that the live template still carries
  `COMPACTOR_INJECTION_BUDGET_FRACTION=.6` — the same stale hostile-pass-#9
  row v3.1.9.5's fix removed from this repo's `runpod.env.template`, never
  removed from the pod's actual RunPod template — to delete while there.
  It also now states plainly that the OpenWebUI History cap (`max_turns`)
  must not be turned on until `scripts/import-history.py` has caught up
  every conversation whose summary hierarchy has fallen behind: a capped
  client stops resending the turns behind the cap, which is what makes
  them unreachable by any rollup path — permanently, not just later.
- **OPERATIONS.md** gains "Getting a real shell into a pod (installing
  sshd, v3.1.9.6)": why the image has no sshd, that it must be re-run
  after every pod restart, the exact Web Terminal commands, which of the
  drop-in/direct-edit cases this pod's real shipped `sshd_config` was
  found to be in (drop-in — `Include` at line 12, before the only
  directive this script manages that ships active,
  `KbdInteractiveAuthentication no` at line 71), the RunPod template port
  change, the `/data/ssh/` private-host-key note (and that
  `compactor/backup.py` does not sweep it up), and how to undo it.
- **RUNPOD_DEPLOY.md** gets a pointer to that section right after "Access
  Your Deployment", and a row in "Upgrading within v3.1.9.x" naming
  `scripts/setup-sshd.py`. Neither implies the image changed.

### Fixed

Four defects in the three scripts above, found by the owner running
`backfill-records.py` on the real production pod **after** a 143-test
green gate — the standing lesson being that these scripts are only ever
run on a pod, against an OLDER installed compactor, from a path that is
not the repo, and the unit suite never exercised any of that combination.

- **Package resolution assumed the repo layout.** `PKG = HERE.parent /
  "compactor"` (`backfill-records.py`, `import-history.py`) resolves to
  `/data/compactor` when the script is copied to `/data/scripts/`, which
  does not exist: `ERROR: no compactor package beside this script
  (/data/compactor)`. Both scripts now resolve, in order, an explicit
  `--compactor-pkg PATH`, then `HERE.parent/"compactor"` (the repo/clone
  layout — the SUPPORTED way), then `/opt/compactor` (the image layout),
  then an already-importable module on `sys.path`, reporting which one it
  used; if every one fails, the error lists every path tried and prints
  the clone command below. `setup-sshd.py` needed no change here — it
  never imports the `compactor` package, only checks for
  `/opt/compactor/main.py` as a container marker.
- **Both scripts crash with a raw `AttributeError` on a pre-v3.1.9.4
  pod.** `backfill-records.py` reads `backfill_mod._MAX_BACKFILL_
  ATTEMPTS` and calls `backfill_mod._backoff_ready`, neither of which
  exists before v3.1.9.4 — verified across tags: v3.1.9/.1/.2/.3 have
  `is_stale`/`_STALE_SECONDS`/`atomic_write_json` but not those two. This
  is structural, not a typo: the script is a PRE-upgrade step that learns
  its rules from the INSTALLED package, which on such a pod is exactly
  the version that lacks those rules — and substituting a newer
  `backfill.py` onto the older package fails differently (verified): it
  imports `current_wipe_generation` from `memory`, which the older
  `memory.py` does not have. Both scripts now check every symbol they
  need is present BEFORE classifying or running anything, and exit 1 with
  an actionable message naming the pod's version state and the clone
  command, instead of a traceback. Neither script silently falls back to
  hardcoded constants when the package is too old — that risks closing
  (or catching up) real records against the wrong rules.
- **`import-history.py`'s guard contradicted the compactor's own
  invariant.** It refused whenever the reconstructed transcript had FEWER
  turns than the store's `recorded_position` — but `summarizer.
  _recorded_position` (via `turns_seen`) is a documented MONOTONIC LOWER
  BOUND: "a deletion, an edit, a branch switch or a bounded client window
  all shrink len(messages)+1" (`summarizer.py`, `_observed_position`,
  invariant I2). Verified against the real 2026-09-22 backup: `recorded_
  position` 3883 against a 3863-turn linear reconstruction of the SAME
  conversation, fully explained by ~153 messages sitting on abandoned
  edit/regeneration branches — not a wrong `--chat-id`, not a stale
  export. The guard now compares CONTENT instead of length: the store's
  own `tail_fp` anchor (the same fingerprint primitive `_observed_
  position` uses to align a window) is searched for inside the
  reconstructed transcript via `summarizer._align_candidates`; found
  anywhere, the run proceeds regardless of the turn-count comparison, and
  not found anywhere, it refuses — still catching a genuinely wrong
  export or chat id, now proven by a dedicated test rather than by a rule
  that flags legitimate edits too.
- **Exit-code semantics were inverted between the three scripts.**
  `backfill-records.py` and `import-history.py` already used 3 for "a dry
  run found pending work" and 0 for success (including a no-op);
  `setup-sshd.py`'s own author flagged that it had implemented 3 as
  "nothing to do" — the exact inverse. Three sibling scripts sharing the
  same flags with opposite exit meanings is exactly what burns an
  operator writing `if script; then`. `setup-sshd.py` is now normalized
  to the same rule (0 success — including a no-op `--apply`; 1 refusal;
  3 a dry run found pending work; argparse keeps its own 2), stated once
  in each script's own docstring and in OPERATIONS.md.

A hostile review of this release (2026-09-22, against the real published
digest and a copy of the real 2026-09-22 backup) then found a further
round of defects specific to `backfill-records.py`, closed below.

- **H1: `--apply` exited 0 having closed nothing.** With every targeted
  `would-resume` record refused (no facts file, a backup collision, or a
  write failure), the run reported success — an operator running
  `backfill-records.py --apply && <upgrade>` would proceed with every
  stale record still live, the exact ~2,600-call GPU storm this script
  exists to prevent. `--apply` now exits 4 when it closed SOME of its
  targeted records but at least one remains open, and 1 when it closed
  NONE — 0 is reserved for "every targeted record actually closed, or
  there was nothing to close". See the script's own docstring for the
  full table and OPERATIONS.md's "Exit codes" section for the convention
  shared with `import-history.py`/`setup-sshd.py`.
- **H4: `--compactor-pkg` silently fell back, and could report a
  provenance that was a lie.** A non-existent `--compactor-pkg` was
  silently skipped in favor of auto-detection, contradicting the
  docstring's own "explicit always wins". Worse: `pkg_source` named the
  directory this script CHOSE, never the module it actually IMPORTED — a
  resolvable-but-empty `--compactor-pkg` with a `backfill` module
  elsewhere on `sys.path` (a stray `PYTHONPATH`) let Python's import
  machinery silently fall through to that OTHER module while the NOTE
  kept naming the intended directory. An explicit `--compactor-pkg` that
  is not a real directory is now a fatal error, and after importing,
  `Path(backfill.__file__).resolve().parent` is asserted against the
  resolved directory — a mismatch refuses, naming the real `__file__`.
- **M1: the capability check did not cover every symbol the script
  actually uses.** It listed `_MAX_BACKFILL_ATTEMPTS`/`_backoff_ready`
  only; `_classify` also calls `is_stale` and `_close_record` also calls
  `atomic_write_json` — both missing from a fabricated or incomplete
  stand-in package would crash with a raw `AttributeError`, and the
  crash inside `_close_record` specifically happens AFTER the backup file
  is already written, leaving an orphan `.bak-` that arms the
  backup-already-exists refusal on every later retry. All four symbols
  are now required before anything is classified or closed, and
  `_close_record` itself removes its own just-written backup if
  `atomic_write_json` still somehow raises for an unrelated reason (disk
  full, a permissions change mid-run) — a second line of defense beyond
  the capability check.
- **M3: the live-compactor refusal failed open.** `_compactor_is_alive`
  returned "not alive" for anything but a 200 within 3 seconds — a 3-second
  timeout under GPU load, or a compactor not yet bound during boot, read
  exactly the same as one that had genuinely stopped, and `--apply`
  proceeded either way. Only an unambiguous connection refusal is now
  read as "not running"; a real response, a non-200 status, a timeout or
  any other error all refuse `--apply` the same way a confirmed-live
  compactor does. A new `--health-url` flag points the probe elsewhere.
- **M8 (backfill-records.py side): `needs-review` never affected the
  exit code, and an all-rejected `--conv` still reported "0 record(s),
  all clear".** A store with only unreadable/ambiguous records exited 0;
  every `--conv` value given being invalid or not found also exited 0
  with nothing inspected. Both now exit 1 (human attention required) on
  a dry run. (M8's other, `setup-sshd.py`-side items are closed below.)
- **Mutants BR3, BR4 and BR8 (of the M4 hostile-review mutation pass)
  survived the existing suite** — a not-yet-stale `in_progress` record
  could be classified `would-resume` and closed if `is_stale` were ever
  bypassed (BR3); the live-compactor `--apply` refusal had NO test
  exercising it at all (BR4); and a terminal (`complete`/`abandoned`/
  `wiped`) record silently misclassifying as `needs-review` instead of
  `leave` looked the same as a passing suite, because either verdict
  still leaves `--apply` untouching it (BR8). Three new tests close all
  three; re-running the reviewer's mutation harness afterward shows 0
  surviving mutants of 8 for `backfill-records.py` (BR1-BR8).
- **`setup-sshd.py` (hostile-review pass 2): B2, B3, H5, H6, H7, and the
  script's own M8 items, closed.**
  - **B2.** `sshd -T` with no `-C` evaluates NO `Match` criteria at all,
    so a pre-existing drop-in shaped like `Match Address * /
    PasswordAuthentication yes / PermitRootLogin yes` made the script
    print "password authentication is disabled" and exit 0 while a real
    connection from anywhere could log in with a password — reproduced
    for real against the published digest. The script now refuses
    outright, before writing anything, if it finds an active `Match`
    line anywhere in `sshd_config` or `sshd_config.d/*.conf`, or another
    drop-in that sorts before its own `00-zions.conf`. After writing, it
    also verifies with `sshd -T -C` for both a loopback and a
    `203.0.113.9` non-local address, not just a bare `sshd -T`.
  - **B3.** The real `ssh-keygen -l -f` returns exit 0 on a PRIVATE key
    file too (confirmed against the real image) — the test suite's own
    fake was *stricter* than the real binary, which is exactly why this
    was invisible. `_validate_key` now refuses anything that is not a
    single line starting with a known public-key type token
    (`ssh-ed25519`, `ssh-rsa`, `ecdsa-sha2-nistp256/384/521`,
    `sk-ssh-ed25519@openssh.com`, `sk-ecdsa-sha2-nistp256@openssh.com`) or
    that contains `PRIVATE KEY`, *before* ever shelling out to
    `ssh-keygen`. `keys_added` in the report/`--json` now holds
    FINGERPRINTS, never key text.
  - **H5.** A failed `--supervise` handoff (a real `supervisorctl
    reread` failure) used to leave the pod with no sshd at all: the
    standalone daemon was stopped before the handoff was attempted, and
    never restarted on failure. It is now restarted, and reconfirmed
    listening, before the run returns its refusal.
  - **H6.** "Started"/"restarted" used to be decided by `sshd`'s own
    exit status, which is 0 even on a partial bind failure. It is now
    decided by what is really LISTENING on IPv4 at the configured port,
    read from `/proc/net/tcp` (`ss` was checked against the real image
    and is not present, so it is not relied on) — a start/restart that
    is not confirmed listening is now a refusal (exit 1), not a silent
    success.
  - **H7.** `/root/.ssh` (0700) and `authorized_keys` (0600, plus
    ownership) are now enforced on every `--apply`, even when there is
    no new key to add — previously a pre-existing `0777`/`0666` pair was
    never corrected in that case.
  - **M5 (test suite gaps).** The fallback/direct-edit path
    (`_dropin_usable` false) had no end-to-end test at all; the fake
    `apt-cache` never produced an available upgrade, so
    `test_never_calls_apt_get_upgrade_or_dist_upgrade` could not fail
    even if `--only-upgrade` were replaced with a bare `apt-get
    upgrade`; and the real `DROPIN_PATH` filename's `00-` sort-order
    prefix (a security property) was never asserted outside a fixture
    that could rename it invisibly. All three are now covered, plus new
    coverage for B2/B3/H5/H6/H7 above. Re-running the reviewer's
    mutation tooling (`mutate.py`, `m2.py`, `m3.py`) against the fixed
    script now shows 0 surviving mutants (the handful of "PATTERN NOT
    FOUND" entries are old mutants whose exact target text no longer
    exists after this rewrite; the same properties are covered by new,
    still-passing tests).
  - **M8 (setup-sshd.py side).** `--port N` used to leave sshd listening
    on BOTH `N` and 22: OpenSSH ACCUMULATES `Port` directives across the
    main file and every loaded drop-in instead of first-match-wins, so
    writing `Port N` in this script's own drop-in never suppressed an
    unrelated active `Port 22` elsewhere. The script now refuses
    outright if it finds a conflicting active `Port` line anywhere it
    does not itself control, and re-verifies the real, post-write set of
    listening ports via `sshd -T` (collecting every occurrence of a
    directive, not just the first, so an accumulation is actually
    visible to the check). Refusal paths now roll back the config write
    AND an `authorized_keys` append made earlier in the same run (host
    key files under `/data/ssh` are the one deliberate exception — see
    below). A missing `supervisord.conf` under `--supervise` now refuses
    cleanly with full `--json` output instead of raising an unhandled
    `FileNotFoundError`. `_make_backup`'s same-second-stamp collision no
    longer raises an unhandled `RuntimeError` either — a numeric suffix
    resolves it.
  - Exit codes are unchanged from the table already in this script's own
    docstring and below in OPERATIONS.md, with one addition: code **4**
    ("`--apply` made progress but work remains") is now documented as
    part of the shared convention across all three operator scripts,
    even though this script never uses it — every `--apply` here either
    fully succeeds (0) or refuses (1).
  - Verified for real against the published digest, with network access
    inside the container: a real key login succeeds; a real root
    password login fails, including with the Match-block drop-in
    scenario present (the script refuses before writing, and the
    config already on disk — unmodified by that refusal — still blocks
    the login for real); a real private key is rejected with no leak; a
    second `--apply` is a true no-op; `--port 2222` leaves sshd
    listening ONLY on 2222 (confirmed via `sshd -T` and a real bound
    socket); and a real `supervisorctl reread` failure against the real
    supervisord still leaves a real, listening sshd afterward. See the
    new `compactor/test_real_image_setup_sshd.py`.
- **`import-history.py` (hostile-review pass 1): B1, B4, B5, H2, H3, M2,
  M3, and the Ctrl-C live-seam window, closed.**
  - **B1.** `--apply` computed its resume offset as ONE flat number
    (`current_turns - len(window)`), but her real store's
    position-to-branch mapping is PIECEWISE — different constant offsets
    at different ranges, from edited/abandoned messages scattered through
    the history. The flat offset left real branch turns uncovered by any
    chunk and permanently mislabelled every chunk written after them
    (chunk recording is append-only). The offset is now derived from the
    store's own covered-turn fingerprint record (growing the window
    checked against on ambiguity, refusing outright if no single
    consistent offset exists) and reported as `resume_offset` (plus
    `resume_offset_detail`) in `--json`; on the real 2026-09-22 backup it
    verifies to **22**, not the old flat 20. A belt-and-braces pass
    re-maps every newly recorded position back onto the transcript after
    the write and restores the backup if anything disagrees.
  - **B4.** `--model` never actually fell back to `$MODEL_REPO`, despite
    the docstring and every documented invocation saying it would — the
    documented `--apply` command failed every time. It now does.
  - **B5.** A `--store`/`--chat-id`/`--conv-id` typo used to silently
    build a brand-new store from scratch and report an encouraging
    "re-run with --apply" while burning real GPU time against the wrong
    place. `--store` must now already exist, contain `summaries/`, and
    already have a `summaries/<conv_id>.json` for the conversation being
    caught up, before any work — dry run or `--apply`.
  - **H2.** The architect's 0/1/2/3/4 exit-code table (see "Exit codes"
    in OPERATIONS.md) is now this script's own: 1 when `--apply` ran but
    the watermark never advanced at all, 4 when it advanced but work
    remains (most often `--max-calls` running out).
  - **H3.** `--apply` rewrites `summaries/<conv_id>.archive.json` too,
    whenever an L2 fold or L3 refresh runs, but only backed up
    `summaries/<conv_id>.json` — contradicting "its entire blast radius
    is one file". The archive sidecar now gets its own matching dated
    backup, named in the report and in `--json`'s new `archive_backup`
    field.
  - **M2.** A transcript sharing exactly ONE turn with the store's
    4-turn `tail_fp` anchor could satisfy the old content guard. B1's
    8-fingerprint offset requirement fixes this as a side effect — a
    single shared turn can no longer satisfy it.
  - **M3.** The live-compactor `--apply` refusal failed open on a
    timeout, reading it the same as a confirmed-stopped compactor. Only
    an unambiguous connection refusal is now read as "not running"; a
    timeout or any other ambiguous response refuses `--apply` unless
    `--force`. Added `--health-url`.
  - **Ctrl-C / the live-seam window.** `_run_apply_loop` no longer
    leaves a cleared live-chat anchor on disk if interrupted (Ctrl-C, a
    pod restart) before any rollup pass completes — the original is
    restored. A follow-up real-image test also confirms the NEXT live
    request after `--apply` stays contiguous with what it wrote (the
    live path's own `window_offset` lands on the same verified 22, not
    the old flat 20 or a naive 0) — this was already correct, a
    verification gap closed, not a bug.
  - **Dry-run estimate made offset-aware (architect follow-up).** The
    due/call estimate compared the raw reconstructed branch length
    against `last_summarized_turn`, undercounting whenever the store's
    `turns_seen` already sits ahead of the branch — the ordinary
    real-backlog shape. It now sizes against
    `effective_position = max(recorded_position, current_turns)`, the
    same value `--apply` already uses. On the real 2026-09-22 backup
    this moves the reported due count from 57 to the true **58 L1
    chunks** (**65** estimated calls, not 64) — corrected everywhere
    else in this changelog and in OPERATIONS.md that quoted the old
    numbers.
  - Re-ran the reviewer's full IH1-IH8 mutant set (adapted where this
    pass's code moved) plus IH3/IH4/a flat-offset-revert mutant: 0
    surviving out of 11.
  - Verified against the real 2026-09-22 backup (published digest, fake
    vLLM): the first new chunk starts at branch turn 2699 (not 2701),
    every newly recorded fingerprint matches the transcript at its
    position-offset for the whole run, only the two summary files
    (+ `.bak-`) changed, `--max-calls` exits 4, and a real SIGINT mid-run
    leaves the store re-runnable. See the new
    `compactor/test_real_image_import_apply.py`.

**New real-image test.** `compactor/test_real_image_operator_scripts.py`
builds a faithful v3.1.9 `compactor` package with `git archive v3.1.9
compactor`, bind-mounts it read-only over `/opt/compactor` in a container
started from the exact published digest
(`sha256:c1295894…`), and mounts a COPY (never the live backup, never
writable) of the real stale-record facts under `/data/openwebui/
compactor`. It asserts: a pre-v3.1.9.4 pod gives the actionable
capability error rather than a traceback; the clone invocation (`git
archive` of this same working tree standing in for a tagged release)
produces `4 would-resume / 16 leave / exit 3` and writes nothing (byte-
compared before/after); running from a copy at `/data/scripts` gives the
actionable multi-path resolution error; and the same coverage for
`import-history.py`'s dry run. A further scenario (added with the M6 fix
below) runs `--apply` for real against a full copy of the real store and
byte-compares every file, not just the dry-run ones.
`compactor/test_real_image_import_apply.py` is the dedicated, deeper
suite for `scripts/import-history.py` itself (the sentence above only
covers the one dry-run scenario `test_real_image_operator_scripts.py`
shares with `backfill-records.py`): it runs `--apply` for real against
the real backup, confirms the verified `resume_offset` (22) and that the
first new chunk lands exactly where it should with no hole or overlap,
that both `--max-calls` and a real Ctrl-C leave the store re-runnable,
and that the very next live request after `--apply` stays contiguous
with what it wrote.
`compactor/test_real_image_setup_sshd.py` is the equivalent suite for
`scripts/setup-sshd.py`: one throwaway container from the same published
digest, driven with `docker exec` across every scenario (so a real
`--supervise` handoff and real host-key persistence carry across steps
the way they would across invocations on a real pod), with the real
`openssh-server`/`openssh-client` apt packages installed for real — this
one needs network access from inside the container, which the other
real-image suites do not.

**M6: this suite is NOT "wired into the same gate as the rest of the unit
suite"** — the previous sentence here was wrong. It cannot be: it needs a
real Docker daemon, the published image, `git` with the release tag, and
the real backup path, none of which the sandboxed `unit-tests` compose
service has (`network_mode: none`, no docker socket). It is excluded from
the default `run-tests.py` selection (`NEEDS_DOCKER`) for exactly that
reason. It is instead a SEPARATE, MANDATORY, host-run step of the release
gate — see COMMANDS.md's "Real-image operator-script suite (mandatory,
host-run)" for the exact command
(`python3 scripts/run-tests.py --python /usr/bin/python3 --real-image
--only real_image_operator`, or the file run directly). Also fixed: its
own `_skip` used to exit 0 (a reported PASS) when `COMPACTOR_ALLOW_
FIXTURE_SKIP` was set in the environment — unlike `test_soak_
conversation.py`/`test_tokenizer_contract.py`, whose narrow per-suite
opt-in that variable legitimately is, this suite IS the mandatory gate
and a skip here must never read as green; it now always exits 3.
`run-tests.py`'s own `BASE_ENV` also clears that variable before invoking
any suite, as a second line of defense against a value leaking in from
the caller's shell. `NEEDS_DOCKER` also names
`compactor/test_real_image_import_apply.py` and `compactor/
test_real_image_setup_sshd.py` — both now landed alongside this one — so
`--real-image` (with no `--only` filter, or `--only real_image`) picks up
all three: `3 passed, 0 failed, 0 skipped, 0 inconclusive`.

**M7: `run-tests.py`'s default interpreter was hardcoded to a Windows
path.** On Linux (WSL, the Docker test image, a bare host run — the
platform this project actually tests on) it now defaults to
`sys.executable`, so `python3 scripts/run-tests.py` just works with no
`--python` needed; the Windows path is kept, but only as the default on
Windows. COMMANDS.md's interpreter note (previously pointing only at the
Windows path) now says so.

## [3.1.9.5] — the deploy docs match what ships

**Documentation and one test only, and NOT a new image.** Every file the image copies
(`compactor/*.py` apart from tests, `stt/`, `tts/`, `entrypoint.sh`,
`supervisord.conf`, `clean-models.sh`) and the `Dockerfile` are
byte-identical to v3.1.9.4. So `:v3.1.9.5-cu12` is published as a second tag
on the existing `:v3.1.9.4-cu12` image, digest
`sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65`
(RUNPOD_DEPLOY.md Step 3), not rebuilt. A rebuild would re-resolve
`apt-get upgrade`, unpinned pip dependencies, the Piper voice URL and the
CUDA base tag, and produce a different, unvalidated image (hostile pass
#18, B1). The release exists so the deploy docs someone follows for
v3.1.9.4's fixes stop contradicting them. The problems were found by a
release-readiness review of v3.1.9.1-v3.1.9.4 on 2026-09-21. Hostile pass
#18 then reviewed this release, and its findings (1 blocker, 7 should-fix,
5 nits) are closed below.

### Fixed (documentation)
- **runpod.env.template, "the single source of truth for the image tag",
  named `v3.1.6-cu12`.** It now names `v3.1.9.5-cu12`, and the image-variant
  list names the current tag (and that `:v3.1.9.4-cu12` is the same image,
  not a rollback) instead of v3.0 only. README no longer calls the June
  `:v3-snapshot` a rollback target.
- **The template re-added the rows v3.1.9.4's git tag annotation said to
  remove.**
  `COMPACTOR_INJECTION_BUDGET_FRACTION` and `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS`
  are now commented out there, with the reason: they equal the image
  defaults (0.75 / 15000), so the rows only pin the value against a later
  image's default, the same leftover-row shape as hostile pass #9's 0.6/6230.
  RUNPOD_DEPLOY.md's template step no longer calls adding them "optional and
  self-documenting"; it says to delete them.
- **There was no upgrade path for a pod already on v3.1.9.x.** RUNPOD_DEPLOY.md
  only covered v3.1.8 → v3.1.9 and rollback to v3.1.8. A new section,
  "Upgrading within v3.1.9.x, and rolling back", covers it. It notes that
  there is no `v3.1.9.3-cu12` image on Docker Hub. It also says what
  rolling back below v3.1.9.4 costs, all of it in `backfill.py`
  `needs_backfill` (hostile pass #18, S3):
  - Older images stop at "a facts file exists", so an unfinished backfill
    on a conversation with live facts is dropped for good (P15-2 undone).
  - They retry `"abandoned"` backfills and capped crashed ones again.
  - They honour `/forget` only through the facts file every wipe path
    leaves behind, not through the `"wiped"` record. The one gap is a wipe
    that fails with a write error, which reports itself as a failed
    `/forget` (round 2 of pass #18 corrected this sentence).

  RUNPOD_DEPLOY.md §6 (rollback to v3.1.8) points there too.
  OPERATIONS.md's rollback step named `v3.1.6.1-cu12` as the last-good
  image. It now names v3.1.9.4 by digest, says tags can be overwritten, and
  says never to roll back to `:latest`, which is still V3.0 and has no
  documented way back from a v3.1.9.x `/data` (S7).
- **The database move was called "the v3.1.9.1 feature"** in the template and
  RUNPOD_DEPLOY.md, but that number went to the reuse fix. The corrected
  wording (S6): no v3.1.9.x release is meant to run with the move, but every
  v3.1.9.x image WILL move the database if `WEBUI_DB_LOCAL` is missing,
  blank or true (entrypoint.sh). `WEBUI_DB_LOCAL=false` stays required.
- **The RunPod CLI example could not have worked, and would have moved her
  database.** It passed flags the `runpod` Python CLI's `pod create` does
  not accept (S2), and it left out `WEBUI_DB_LOCAL`, where a missing row
  means `true`. It is removed. The guide now says to deploy from the
  template.
- **Model and weights.** The volume pre-warm step and the GPU table named
  the text-only model repo. They now name the vision variant, which is both
  the image default and the template's value. The env-var table's
  `MODEL_REPO` (magnum-v4-22b, "set -12b on A40") and `VLLM_EXTRA_ARGS`
  ("do not add --quantization fp8 on A40") described a config that
  production has not run since rc8, and now match the Dockerfile and
  template (S4). The pre-warm step said "optional" and Troubleshooting said
  "vLLM downloads weights on first start". Under the template's
  `HF_HUB_OFFLINE=1` neither is true: the pre-warm is REQUIRED for the model
  `MODEL_REPO` names, and entrypoint.sh's "weights present" check passes on
  ANY cached snapshot (S5).
- **RUNPOD_DEPLOY.md said the template "does not carry" `WEBUI_DB_LOCAL`**
  (it does) and counted "42 vars" (51 uncommented rows).
- **Three settings added in v3.1.9.4 were documented nowhere:**
  `COMPACTOR_VECTOR_CACHE_MAX` (8192), `COMPACTOR_BACKFILL_MAX_ATTEMPTS` (3),
  `COMPACTOR_BACKFILL_RETRY_BACKOFF_S` (600, doubled per attempt). They are now
  commented rows in runpod.env.template. None of them needs setting.
- **README's image-tag table still called v3.0 the current release.** It
  now lists the v3.1.x and v3.0.x tags that exist, the missing v3.1.9.3
  image, and that `:latest` is still v3.0 until a v3.1.x image passes the
  on-pod gate.
- **.env.example** is used for local `docker compose` only, once copied to
  `.env`. It had no `WEBUI_DB_LOCAL` row, which moves the database on a
  local run, and it pinned `COMPACTOR_MAX_FACTS_TOKENS=1500` over the
  image's 3500. The first is added, as `false`; the second is commented out.
- **Build and deploy order.** Step 3 now publishes v3.1.9.5 as a digest
  retag, and builds any real code release from a clean checkout of its tag.
  It says to push the image before pointing the template at it (N5).

### Tests
- **`compactor/test_reuse_fit.py` checked a comment (S1).** Its
  template-vs-Dockerfile check used an unanchored
  `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS=(\d+)`, which now matched the
  commented-out example, so it no longer checked what its message claimed.
  It now asserts two things for BOTH rows of the pass-#9 pair
  (`COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` and
  `COMPACTOR_INJECTION_BUDGET_FRACTION`). The template must carry no live
  row, indented rows included. Its commented example must show the
  Dockerfile's value. Mutation-tested: a stale row (at the end, or right
  after the comment), a row at the current value, and a deleted or drifted
  example each fail. Test files are not copied into the image.
- The v3.1.9.3 entry's two "known residual" notes (P14-1, P14-2) now point at
  their fix in v3.1.9.4. V314_BACKLOG.md's OPEN list has a status note: A-05
  is fixed, A-10 (`read=None`) is still open, and the rest has not been
  re-triaged.

### Not changed, and still true
- **Not validated on a pod.** No v3.1.9.x image has a recorded Tier-2 boot
  self-test on a real pod, so `:latest` stays `:v3.0` (TESTING.md rule 5).
- **Still open:** A-10 (a paused vLLM can hang a request, by design of
  `read=None`) and the residuals listed in OPERATIONS.md and the entries
  below.

---

## [3.1.9.4] — every defect we could find

The last patch before the database move. It closes hostile pass #14's
residuals, pass #15's defect hunt and passes #16 and #17's findings, plus
the loop detector's fence readers left open since pass #9. Built in four
parallel lanes and six fix rounds; each round was reviewed and mutation-tested
before merging. Real-data replay of her branch (80 runs, margins 0 to
8,192, tails, /tokenize down, images, pricings): reuse never loses her
previous exchange where declining keeps it, and it is never worse than
v3.1.9.3 in any state.

### Fixed
- **Replies no longer wait ~30 s for fact selection (P15-4, HIGH).** Every
  request re-embedded all of her ~190 facts in one batch on the event
  loop; the pod's own log for 2026-09-17/18 shows the step between
  compaction and retrieval taking a median of 30.3 s (p90 38.8 s), before
  vLLM was called, with every other request blocked behind it. A bounded
  vector cache keyed on (embedding model, exact text), shared with dedup,
  means a request embeds only her new message and a memory tail only new
  facts; selection results are identical. Fact selection and dedup's
  clustering run off the event loop. The first request after a restart
  still pays the cold cost once.
- **A /forget cannot be undone by memory work already in flight (P15-5
  and its follow-ups).** A full /forget used to cancel every other
  conversation's pending memory writes; now it waits without cancelling,
  and a per-conversation wipe generation makes any write that began
  before a wipe (the memory tail, its rollups, the history index, a
  lazy backfill) discard itself instead of landing after it. /forget, the
  admin forget, /retire and the overwrite import all bump it, and all mark
  any backfill for that conversation finished for good, so the next
  message cannot rebuild her facts from history (P16-1); a running
  backfill stops at its next exchange (P16-2), including a /forget that
  lands during its last extraction call, and no backfill write can turn
  a "wiped" record back into a retryable one (P17-4). The admin forget now
  leaves the same empty-facts marker as chat /forget even when there were
  no facts; /tidy and archive-stale, which archive rather than delete, are
  no longer mistaken for that marker. After /forget the summary hierarchy
  still rebuilds from the chat history if she keeps chatting in the same
  chat, because the chat itself still exists; the /forget reply (now
  whenever anything was forgotten, P17-5) and USER_GUIDE.md say so.
- **Summaries cut at their length cap are no longer stored as complete
  (P15-6).** Most of her stored summaries end without punctuation, and two
  of her four live chapters sit at the cap, so this was probably routine.
  Every summary prompt now asks for a length well under its cap; a cut
  summary is retried once at the same cap with a tighter target (counted
  against the call budget), and if still cut, the longer attempt is kept,
  trimmed to a sentence, line or word boundary, logged, and counted at
  `/health/full`. A clean retry replaces the first attempt only when it is
  at least as long as that attempt trimmed (P16-7); the retry is skipped
  when the trimmed first attempt is already long enough, and an attempt
  stuck in a repetition loop never wins on length over a clean one
  (P17-1). The hierarchy never
  stalls on a cut. Compaction's own summary call gets the same handling,
  and its retry never takes a call a later batch needs (P16-4).
- **A reply vLLM cuts with a mid-stream error is no longer memorized as
  finished (P15-1),** streaming or not (P16-5).
- **Backfill:** a redeploy mid-backfill no longer abandons it for good
  (P15-2); a failing backfill retries with a back-off and stops after a
  small cap; `COMPACTOR_FACTS_EXTRACTION=false` now stops it (P15-8).
- **The chapter archive is part of her memory (P15-3):** /forget, the
  admin forget, the wipe check, /retire and the overwrite import now
  cover `summaries/<id>.archive.json`.
- **Reuse and the budget guard (P14-1, P14-2, and p14's undemonstrated
  list):** a margin learned mid-request demotes the summary stand-in
  ahead of her previous exchange instead of costing the exchange; the
  reuse preview and summarize() share one /tokenize count; the reuse
  check's recent-window floor is counted exactly; the compaction trigger
  accounts for the learned margin; the guard's last-resort pass sheds in
  the guard's own order (older turns, then injected memory, then the
  recent window except the newest, then the stand-in), only as much as
  the gap needs, and measures before touching the recent window once
  memory has been spent (P16-3) and again before touching the stand-in,
  so it never drops the stand-in when dropping her previous exchange
  alone fits; the measurement cap of 6 holds (P17-2, P17-3). The worst-case current-time line (99 tokens) was
  measured to fit its allowance; no change needed.
- **Code fences in the loop detector (P9-5, P10-4, `~~~`):** one shared
  CommonMark reader for every fence decision (backtick and tilde fences,
  closers that must match their opener), and balancing fences that match
  the fence left open. On her 1,782 real replies nothing changes.

### Tests
Two data-loss guards that nothing pinned now have tests (P15-7), and the
reuse-preview tests that a later change had weakened were recalibrated.

---

## [3.1.9.3] — the gate tells the truth

Four findings from hostile pass #11 (v3.1.9.2 at `1069da1`), closed across
four parallel lanes: the reuse ceiling accounting for the recent window
(lane reuse), a concurrent-merge fact-loss race (lane merge), the missing
`opencv` extra that keeps the real chat template from loading (lane opencv),
and — this lane's own items — a release gate that had been silently skipping
its only real-tokenizer suite for three releases, and an adversarial test
that could no longer tell a genuine reuse success from a decline. Hostile
pass #13's own real-data replay then found one more HIGH in the reuse
window check itself (P13-1, below) plus a MEDIUM and two LOWs, closed in
this same release before it shipped. Hostile pass #14 cleared the result to
ship; its four LOWs are either fixed below or listed as known residuals.

### Fixed
- **P13-1 (HIGH): the reuse window check now reads the learned budget
  margin the guard already reads.** After a vLLM context-length 400 the
  guard did not predict (a `/tokenize` outage, a mispriced image),
  `_BUDGET_MARGIN` latches to `overshoot + 512` (up to the
  `MAX_MODEL_LEN // 4` ceiling) in one step, process-wide, and
  `_enforce_hard_budget` shrinks its own limit by it before shedding a
  token — but the P11-6/P12-1 window check above recovered the same
  `effective_limit` and never subtracted the same margin. While a margin
  was in force, this check could approve a stand-in the guard, needing
  that many more tokens of room than the check thought existed, could not
  actually fit beside the previous exchange — reuse "succeeded" and the
  guard then shed U_prev/A_prev to find the room, giving back exactly what
  P12-5 (below) fixed on the declined path. Real-data replay (hostile pass
  #13, `SP\p13-findings.md`): at her current hierarchy, 1/30/313 of 474
  positions lost the exchange at margins 513/4,096/8,192; at peakA, up to
  453. Never worse than v3.1.9.2 in the same state. Fixed by subtracting
  `_BUDGET_MARGIN` from the check's own `_effective_limit_est`, the same
  global the guard reads. The two reads are not atomic (hostile pass #14,
  P14-1, known residual at v3.1.9.3; addressed in [3.1.9.4] "Reuse and the
  budget guard"): a margin latched by another request's rejection
  while this one is mid-compaction applies to this request's guard but
  not its reuse decision, and can cost her previous exchange on that one
  request. The guard keeps reading the live value on purpose; forwarding
  at the pre-rejection limit risks losing the whole reply instead. The
  learned margin is also now visible at `/health/full`'s
  `checks.budget_margin` (`main.budget_margin_state()`) — before this it
  had no field anywhere in that endpoint (the adversarial suite's own F-02
  names the gap by name).
- **P13-2 (MEDIUM): the fresh-summary reserve now prices the batch count at
  the scale `summarize()` will actually use.** The P12-6 reserve (below)
  priced its batch-count preview at the flat pessimistic scale (2.0x)
  unconditionally, reasoning that it mirrored `summarize()`'s own
  `/tokenize`-down fallback — but `summarize()` only falls back to 2.0x
  when `/tokenize` does not answer; otherwise it packs the same list at
  the measured, ~1.0x scale. In her routine between-L1-rollup state (an
  uncovered tail past the last L1 chunk, present on every request) this
  inflated the predicted batch count and declined reuse on 11-82 of 474
  positions per state that would have fit even the real call's un-folded
  worst case. Most declines lose the uncovered tail (147 of 165 at tail 20
  peakB): a window decline hands `summarize()` the whole older span, that
  is refused over the call cap, and the verbatim turns are shed ahead of
  memory. Fixed: one more `count_tokens_exact` call measures the real
  scale on the same span `summarize()` will use, falling back to the
  pessimistic 2.0x only when `/tokenize` genuinely does not answer.
  The trade, measured by hostile pass #13 on her real branch at tail 20
  peakB: the tail is lost at 69 positions instead of 147 (25 instead of 45
  at peakA), and because reuse fires more often at peak sizes, facts or
  retrieval are cut on more requests (4/243 instead of 0/166 with folded
  fresh summaries; 51/307 instead of 4/225 in the un-folded worst case).
  Her previous exchange is never lost either way. Known residual at
  v3.1.9.3 (hostile pass #14, P14-2; addressed in [3.1.9.4], where the
  preview and `summarize()` share one /tokenize count): the preview and `summarize()` ask `/tokenize`
  separately, so if it fails in the milliseconds between them,
  `summarize()` can make more batches than were reserved, up to 2,048
  tokens more on a span at exactly the call cap. That costs the one
  request.
- **P13-3 (LOW, documentation): three passages describing a
  replay-harness artifact as a measurement corrected** (see P12-6's own
  correction below); the runbook's window-decline guidance now names the
  knobs that actually move it and the fresh-summary reserve alongside
  `last_declined_ceiling`/`last_declined_others` in `checks.reuse`
  (`last_declined_reserve`); "not silently worse" replaced with what a
  window decline actually drops in the cap-refusal state P13-2 measured.
  Hostile pass #14 (P14-3) corrected six details of those corrections:
  the settings that do move the reserve and the ceiling, the images
  sentence, the declined path's cause, which number measures what, the
  log lines that used to be the only sign of a margin, and the trade in
  P13-2 above.
- **P13-4 (LOW): the hard-budget guard's memory-before-previous-exchange
  order (P12-5, below) now holds on a conversation with no system
  prompt.** `_droppable_system_indices` used to clamp with `sys_idxs[max(1,
  protect_system):]` — protecting at least the first system message even
  when the caller sent none — so on a request with no system prompt
  (`caller_system == 0`), whatever `inject_system_block` put at index 0
  (facts, if no persona precedes it) read as "the caller's own", the
  P12-5 branch never triggered, and the floor-less generic shed loop it
  exists to preempt reached her previous exchange first, with injected
  memory sitting right beside it, unspent. Fixed by trusting the
  function's own documented contract (`sys_idxs[protect_system:]`, no
  floor). The compaction stand-in keeps its protection by content
  (`_is_compaction_standin`) in the memory-first branch, and is spent
  only as the very last resort, the same with or without a caller prompt
  in front of it. The clamp used to shield it from that last resort only
  when it happened to sit at index 0. `test_budget_guard.py`'s system-less
  test pinned that position rule with a fixture that was never a real
  stand-in; it now pins the content rule and the parity instead.
- **P11-6/P12-1: reuse no longer squeezes her recent conversation, at any
  image price.** The v3.1.9.2 reuse ceiling ignored the recent window, so
  at the hierarchy size its 15000 cap was raised for (~13k tokens, hers is
  ~8.4k today) reuse lost her previous exchange where declining would have
  kept it. This entry originally said `compact_if_needed` bounded the
  stand-in by the system prompt and the retained images, "priced at the
  `IMAGE_TOKEN_ESTIMATE` constant, deliberately independent of
  `count_tokens`" — **that was not what the code did, and hostile pass #12
  (P12-1) found the gap it left**: the reserve called `count_tokens`
  itself, so it was priced by whichever tier of that function happened to
  run — the flat 4,096-token estimate before the opencv fix below, or that
  fix's own real per-resolution cost (measured ~3,080 square, ~2,352 for a
  4:3 photo) after it. Landing opencv in the SAME release made the lower,
  real price the default on the pod with no setting changed — the reserve
  shrank, and reuse started firing again at the exact hierarchy sizes this
  entry exists to protect (measured on a real branch: her previous
  exchange lost at up to 47 of 474 positions, depending on state, where
  declining or v3.1.9 kept it). The reserve now bounds the stand-in by the
  system prompt and the ACTUAL recent turns (`keep_recent` — what the
  guard, see P12-5 below, never spends before the previous exchange) plus
  a fresh summary's worst case when a fresh span is pending (see P12-6
  below for what "worst case" actually means), instead of a per-image
  estimate — so the decision no longer prices images at all, and does not
  move when `count_tokens`'s image pricing does. Verified at all three
  prices above.
- **P12-5: a DECLINED reuse request no longer sheds her previous exchange
  before injected memory.** P11-6/P12-1's structural check above (and the
  plain over-budget path) routes a growing share of requests through the
  declined path — summarize the older span from scratch, forward it
  verbatim where it will not fit a call cap, inject facts and retrieval as
  usual. The hard-budget guard's "spend memory before the turns no summary
  covers" ordering (pass-3 F5) only ran when the array carried
  compaction's OWN stand-in; a declined array carries injected memory with
  no stand-in, so it fell through to the floor-less generic shed loop,
  which sheds the OLDEST non-system turn with no idea memory sits right
  next to it — and once old turns ran out, that oldest turn was her
  previous exchange. Measured on a real branch: every such loss (20-23 of
  474 positions per hierarchy state) had 2,500-8,500 tokens of injected
  memory left unspent that would have covered it. The same ordering now
  applies to any array carrying injected memory, stand-in or not.
- **P12-2 (MEDIUM): a window-squeeze decline (P11-6/P12-1, above) no
  longer reports itself as an ordinary budget decline.** It used to record
  `last_reason=budget` with the TARGET-/injection-based ceiling and total
  — numbers this check never consults — so `checks.reuse` showed
  `last_declined_ceiling` sitting at the configured
  `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS`, and OPERATIONS.md's runbook reads
  that as "raise the setting," which moves nothing here (reproduced:
  raising it from 15000 to 20000 left the same requests declining). Now
  its own reason (`"window"`), its own counter (`declined_window`), and
  its own two numbers; OPERATIONS.md explains what actually helps instead
  (fewer/cheaper retained images, a smaller reply, an earlier L3 rollup).
- **P12-3 (LOW, test only): a merge whose commit fails now has a test
  proving `conv_lock(dst)` is released.** The shipped code was already
  correct (the release runs in a `finally`) — no fix, only a mutation-
  tested regression check, since none of the existing merge-concurrency
  cases drove a commit failure and a regression here would wedge every
  later writer to the conversation (the extraction tail, `/remember`,
  `/forget`, archive/restore/dedup, the next merge) until the process
  restarts.
- **P12-6: the fresh-summary reserve (P12-1, above) now prices what
  `summarize()` can actually produce, not a guess at one call.** A
  real-data replay of P12-1 found one state still open: with a fresh
  summary attached to the stand-in (routine — 3 `summarize()` calls per
  reusing request between L1 rollups, per hostile pass #11 — not an edge
  case), the reserve's fresh-summary allowance assumed exactly one
  `SUMMARY_MAX_TOKENS` batch. `summarize()` map-
  reduces the fresh span over budget-sized batches, and when the reduce
  phase cannot fold them (the per-request call budget exhausted by the
  map phase, or two dense partials together still missing the reduce
  call's own input budget), the returned summary is the RAW CONCATENATION
  of every un-folded batch, each up to `SUMMARY_MAX_TOKENS` — measured at
  1,000-2,000 tokens (one or two un-folded batches), not the one this
  reserve priced for. The reserve now previews the SAME batching
  `summarize()` itself will do, on the same fresh span, and reserves
  `n_batches * SUMMARY_MAX_TOKENS`. **A second real-data replay (the
  coordinator's corrected "unpair" methodology, after their first "fresh"
  harness turned out to be a spy artifact that could not test this reserve
  at all) found the cap-refusal case was itself backwards**: reserving
  NOTHING once the pessimistic batch estimate exceeded the per-request
  call cap assumed `summarize()` would refuse to run — but that estimate
  is deliberately inflated (2.0x) to size the reserve safely when
  `summarize()` DOES proceed, not to predict whether it will; on content
  whose REAL (undoubled) batch count is still under the cap, `summarize()`
  proceeds anyway and can still leave every batch un-folded. Measured at
  the live end of her branch (positions 948/952, her last real message is
  a 39,569-character reply): reuse fired with a stand-in beside a 3-4
  batch fresh span and a zero-token reserve for it, and lost her previous
  exchange with 3.9-4.7k tokens to spare. Fixed: the reserve now caps at
  the call cap's OWN worst case (`min(n_batches, MAX_SUMMARY_CALLS_PER_
  REQUEST) * SUMMARY_MAX_TOKENS`) instead of collapsing to zero — correct
  whichever way the real, non-pessimistic `summarize()` call decides.
- **P11-4: the guard no longer spends injected memory on arithmetic the
  exact counter contradicts.** A residual of v3.1.9.2's P10-1 fix: round one
  could cut memory on a local estimate that overpriced images, then shed
  the exchange anyway. It now re-checks against the exact count before
  spending memory.
- **P11-1: `checks.reuse` records the outcome after the work that can
  fail.** It used to record `success` before the fresh-span summarize ran;
  a failure there forwarded her whole original array (cut to ~3 turns by
  the guard) while `/health/full` still said `success`. Now that case is
  `error`.
- **P11-3 (LOW): `declined_recently` means what the runbook says** —
  budget declines only; a cancelled request is not counted as one.
- **Concurrent merges into one conversation no longer lose facts.**
  `merge_conversation` runs on a threadpool worker and only PROBED
  `conv_lock(dst)`, an asyncio lock it never held, so two merges into the
  same destination both passed and both did an unsynchronised
  load-modify-write: the adversarial race lost 20 of 40 acknowledged facts
  on 60 of 60 attempts, with HTTP 200 on both sides. The lock is now
  actually held for the read-modify-write — acquired on the event loop
  with a 10-second bound (on timeout, the original "retry in a moment"
  refusal), the file work kept on the worker thread so the server does not
  stall, and released in a `finally`. The reverse-merge (source) guard is
  unchanged.
- **The compactor's chat template now loads for images, and prices each
  image once.** Every boot logged `could not apply the chat template ...
  opencv is not installed`. Measured: only messages carrying an image ever
  hit it — text-only counting was already correct, so facts, retrieval and
  the summary and persona blocks do not move. `opencv-python-headless`
  (pinned) is added to the compactor venv (+152 MB unpacked, no other
  package version changed). Installing it alone would have made images
  WORSE: `count_tokens` added its flat 4,096-token estimate on top of the
  template's own image tokens, and re-encoded the image markers as literal
  text (2-2.3x each). It now prices a rendered image by its markers and
  keeps the flat estimate only for images the template did not render.
  Measured against vLLM's own per-image cost at 256/512/1024/2048 px
  (110/380/1406/3080 tokens): within 3 tokens.
- **P11-2 (hostile pass #11, re-opening P9-4): the adversarial reuse test's
  discriminator passed when reuse did not fire at all.** P10-3 (v3.1.9.2)
  gave a stored-hierarchy miss (`no_state`, a brand-new conversation),
  a coverage miss (`no_coverage`, a hierarchy that exists but covers none
  of this request) and the reuse block's own exception (`error`) their own
  counters instead of folding them into `declined_budget` — so
  `tests/adversarial/test_adv_v3192_coverage.py`'s discriminator
  (`attempted == before+1 AND declined_budget == before`) started passing
  on all three, exactly the "reuse silently did not fire" failure mode the
  test exists to catch. Fixed: the discriminator now requires `succeeded ==
  before+1 AND last_reason == "success"` — no other outcome can produce
  that pair. Two new tests (`test_reuse_discriminator_reports_no_state_
  not_success`, `..._no_coverage_not_success`) drive the `no_state` and
  `no_coverage` shapes through ordinary real HTTP requests (a brand-new
  conversation's first oversized request; a second request that replaces
  the whole history on the same conversation id). A fourth shape — reuse
  recorded "success" and then the fresh-span re-summarize call itself fails
  — is a separate finding (P11-1) in `compact_if_needed`'s recording order,
  not in this test; this same assertion will also catch it once that lands.
- **The tokenizer-contract suite (`compactor/test_tokenizer_contract.py`)
  had been SKIPPED in every release gate since v3.1.7 (P9-6), including
  this one's own predecessor — and it is the only suite that exercises a
  real `/tokenize` contract, the one that would have caught the
  fixture-handler change v3.1.9.2 itself shipped.** Not a code defect: the
  suite has always correctly reported SKIP (exit 3) with a loud reason when
  no fixture is reachable. The gap was procedural — every gate assembled
  for a release ran `docker-compose.tokenizer-contract.yml`'s `soak-tests`
  service (a scale/lag check, its own separate fixture) and never its
  `contract-tests` service (the real-tokenizer contract check), two
  different suites in the same compose file, easy to conflate by name. Ran
  directly at this release's HEAD: **rc=0, 98 checks, 0 failures, "All
  tokenizer-contract tests passed."** `scripts/run-tests.py`'s module
  docstring — the project's one canonical description of how to run
  everything — now names `contract-tests` explicitly and says plainly that
  no fixed set of compose invocations is "the gate" until that one's exit
  code has been read too.
- **P11-5 (LOW): the "2.0-2.4 characters/token" figure in `RUNPOD_DEPLOY.md`
  and `OPERATIONS.md`'s Max Tokens guidance was measured on an unusually
  decorated month.** That figure came from 2026-08-28 production replies
  carrying heavy box-drawing decoration; her current branch (476 replies,
  measured 2026-09-17 with the Tekken tokenizer) is 0.11% box-drawing and
  runs **3.77 characters/token** instead — nearly double, which made the
  old "7000 tokens routinely hits her p90 reply" claim wrong in the
  alarming direction (it does not, at the corrected rate) and understated
  how much headroom 12000 tokens actually carries (~45k characters, not
  ~24-29k). Both docs corrected with the new figure, its date, method and
  a vocabulary caveat (measured against `mistral_common`'s bundled
  `tekken_240911`, not the served model's own `tekken.json`; no pod
  `/tokenize` call was made). **The 12000 Max Tokens recommendation is
  unchanged** — it was already safe, and remains so at the corrected rate.

## [3.1.9.2] — repetition-loop hardening

**The bug (production logs):** the model (Cydonia-24B, vLLM 0.19.0) sometimes
degenerates into one token or phrase repeated, or an unbroken line of short
fragments. `reply_is_degenerate` already kept such a reply out of what gets
memorized, but OpenWebUI still re-sends it as ordinary chat history on every
later request, and the hard-budget guard's ~5-turn window makes a recent loop
reply a large fraction of everything the model is shown right after it loops
— plausibly why the reply after a loop has come back empty. Separately, the
owner's `repeat_penalty` (Ollama's name) rode through OpenWebUI's
pass-through-unknown-keys behavior to vLLM, which does not recognise it —
`repetition_penalty` silently stayed at its default, and nothing said so.

### Fixed
- **Ollama sampling-name translation.** `repeat_penalty` is translated to
  `repetition_penalty` before forwarding (coerced to a positive float; a bad
  value is dropped with a WARNING, never forwarded). If both are present,
  `repetition_penalty` wins **when it is itself valid**; otherwise the
  coerced `repeat_penalty` is used instead of losing both values.
  `repeat_last_n` has no vLLM equivalent and is dropped with a note. A
  numeric-string `repetition_penalty` is coerced to float. **Non-finite
  values are rejected** (a string `"inf"`/`"Infinity"`/`"1e999"`, or a
  numeric `1e999`, used to be coerced to a real `inf` and forwarded, which
  httpx's encoder then refused to serialize — a 500 from inside the proxy,
  after compaction and memory injection had already spent the turn; hostile
  pass #7, F4). Logged at INFO, at most once per conversation-id per process
  (bounded set — see `_translate_ollama_sampling_params` in `main.py`).
- **Detected loop replies are now kept out of what is FORWARDED to vLLM**,
  not only out of what is memorized. After compaction and memory injection,
  before the hard-budget guard, every non-newest degenerate ASSISTANT turn is
  replaced (`_redact_forwarded_loop_replies` in `main.py`) — but **only the
  flagged SPAN is collapsed, not the whole reply**, when the detector can
  locate one (a token run, a phrase repeating to the end, a single-character
  run, or an unbroken fragment line): the clean text before it, and after it
  when the span sits mid-reply, both reach the model. Real-data measurement
  (hostile pass #7, F2) showed the earlier whole-reply-replacement rule threw
  away a real, clean answer of up to 21.6k characters on 33 of 68 flagged
  replies, because a PHRASE loop ends on sentence boundaries and the old
  "trim to the last sentence, re-judge" rule could not cut it at all. The
  cut repeats until what is left is no longer flagged (the phrase rule looks
  at a 4,000-character window, so one cut can leave part of a long loop in
  place; on the 2026-09-16 backup that was 10 of 67 flagged replies before
  this, 0 after). A reply with no span to cut around (decoration fraction,
  script drift, the short-list-run backstop) falls back to its clean
  sentence head; only a reply with no clean text worth keeping gets the
  whole-reply placeholder (10 of 67 on that backup, down from 51 of 68).
  **Fenced code was excluded from the token-run rule for one release
  (hostile pass #7, F3) and is NOT any more — see the "REMOVED, not
  narrowed" entry under hostile pass #9 below; that history is kept here
  for context, not as a description of current behaviour.** User turns,
  system messages and the newest message are never touched. Logged at INFO
  as a count only (no text) — `touched=<n> whole=<k> cut=<n-k>` (hostile
  pass #8, P8-8: the line used to say "replaced N ... with a placeholder"
  unconditionally, which stopped being true once span-cutting made a
  placeholder the MINORITY outcome — 10 of 66 flagged replies on the
  2026-09-16 backup, not all of them). Apart from the fence-exemption
  history described below, the detection rules and thresholds are
  unchanged.
  A new internal helper (`_reply_degenerate_verdict`) exposes the flagged
  span alongside the same reason string, cached per 128-bit content digest
  (not the text itself, so the VERDICT cache holds no reply text) so a
  conversation OpenWebUI resends unchanged on every later request only pays
  the detection cost once per turn, not once per request (measured
  0.78-0.83s CPU per request on an 811-message real conversation before
  caching; hostile pass #7, F5). A SEPARATE cache
  (`_DEGENERATE_CUT_CACHE`) memoizes the cut-and-re-judge loop's own
  result, keyed the same way — its VALUES **are** the kept replacement
  text (a correction to this entry's earlier wording, which claimed
  neither cache held reply text; hostile pass #8, P8-8), capped at 1,024
  entries and now honouring `COMPACTOR_DEGENERATE_VERDICT_CACHE_SIZE=0`
  the same way the verdict cache does (it used to keep caching regardless
  of that setting). The ROLLUP-input redaction (`_redact_degenerate_turns`,
  memory-side, pre-existing) is UNCHANGED — this release does not have the
  real-data basis to prove the same span-cutting rule is safe there.
- **The hard-budget guard's recent-turn floor is now aligned the same way
  `split_messages` aligns its own kept-recent window** (starts on a user
  turn; an odd turn count means the real window can hold one fewer message
  than `KEEP_RECENT_TURNS`). The floor used to be the raw message count, so
  an old, unpaired turn (most often a retained image) sitting in the
  misaligned slot was protected from the pre-shed loop and could cost
  injected memory (facts/retrieval halved or dropped) to keep a turn that
  was not actually inside the real recent window — and could still be
  dropped anyway by a later shedding stage, spending the memory for nothing
  (hostile pass #7, F1). **Correction (hostile pass #8, P8-1):** the first
  version of this fix only stripped a leading turn of the WRONG role from
  the aligned window, which happened to fully cover a stale ASSISTANT-role
  test fixture but not what production actually sends — OpenWebUI puts an
  uploaded image on a USER turn, so a preserved old image sat in front of
  another USER turn and the role check alone found nothing to strip,
  leaving the floor unaligned exactly as before for that shape. The floor
  now also strips a leading turn that shares its role with the turn right
  after it (a real recent window always alternates roles; two consecutive
  user turns at the front means the first one is not actually recent).
- **Fence exemption fixes (hostile pass #8), and REMOVAL (hostile pass #9):**
  - **P8-2 (regression, was flagged correctly by v3.1.9):** the token-run
    fence exemption above treated an UNCLOSED ` ``` ` opener as fencing
    everything after it forever — this model uses bare ` ``` ` lines as
    decorative boxes (128 of 1,709 unique real replies in the 2026-09-16
    backup have an odd count), so a real identifier loop starting after
    the last unmatched opener and running to the end of the reply was
    silently exempted and stored to memory/forwarded verbatim. "Fixed" at
    the time: a run only counts as fenced when the fence actually CLOSES
    again later, and never when the run reaches the end of the reply
    either way.
  - **P9-3 (hostile pass #9): P8-2's own fix does not work, and the
    exemption is now REMOVED entirely rather than narrowed a third time.**
    The "never when the run reaches the end of the reply" half is
    logically unsatisfiable together with "only inside a fence that
    CLOSES": for a fence to be judged closed, a LATER `` ``` `` toggle
    must exist past the run, which makes "reaches the end" false by
    construction every time "closed" is true. The clause never fired —
    mutation-measured, 0 of 20,000 synthetic verdicts depended on it — so
    a loop sitting inside an ordinary CLOSED decorative box (the common
    case for this model, not the exotic one) was exempted regardless of
    position: stored to facts/episodic/rollups and forwarded on the wire
    unredacted. v3.1.9 flagged this shape; v3.1.9.2 (through this release,
    until now) silently did not. The token-run rule now judges text
    exactly as v3.1.9 did, with **no fence awareness of any kind**. The p7
    F3 complaint this exemption was originally written for (a legitimate
    repeated-value array losing the whole reply) is already answered by
    the span-cut described above, which keeps the rest of the reply and
    drops only the flagged span — so the cost of losing the exemption is
    that such a reply is skipped from MEMORY only, exactly as v3.1.9 did;
    not a regression, simply not the improvement F3/P8-2 attempted.
  - **P8-3:** the forwarded-window cut's clean-prefix rule reused
    `trim_to_last_sentence`, which refuses any boundary inside a fence —
    correct for the memory-side redaction (an unterminated opener must
    never reach fact extraction), wrong for the forwarded view, where
    nothing is extracted. This model's boxed reply style put the last
    real sentence end inside a box often enough to discard up to the
    WHOLE reply before a loop in one real case, and roughly 11k
    characters of boxes-and-prose in another. A new
    `_trim_forwarded_prefix` (forwarded path only; the memory-side rule is
    unchanged) allows a boundary inside a fence and falls back to the
    last line break when no sentence end is available, self-balancing any
    fence it leaves open.
  - **P8-4:** a mid-reply cut that splits one CLOSED fence across the kept
    prefix and suffix used to leave a stray, unbalanced ` ``` ` marker,
    misreading the rest of the reply as code. The cut now balances the
    fence count of what it actually emits.
  - **P8-5:** the cut-and-re-judge loop's pass budget is now bounded by
    total CHARACTERS scanned across all passes, not a flat pass count — a
    300k-character pathological loop (far beyond her longest real reply,
    51k) now costs a fraction of the 2.2s of GIL-bound CPU it measured
    before, with no change for anything her real conversations produce.
    Falling back to the whole-reply placeholder because the pass budget
    was exhausted is now logged at WARNING (it used to be silent).
- **`max_tokens: 1e999` (and any other numeral that overflows to `inf`, in
  any numeric request field) is now rejected at JSON-parse time with a 400**
  (hostile pass #8, P8-6), the same way the existing `NaN`/`Infinity`
  constant guard works. It used to reach `int(body.get("max_tokens") or 0)`
  and raise `OverflowError`, which the surrounding `except
  (TypeError, ValueError)` did not catch — a 500 from inside the proxy for
  a client-supplied number, now caught before compaction or memory
  injection ever run. `OverflowError` was also added to that except clause
  as a second line of defence, which now drops (and logs) an unparseable
  `max_tokens` instead of leaving the client's own bad value sitting
  untouched in the forwarded body.
- **A 4-space-indented ` ``` ` line is no longer misread as a fence
  delimiter** (hostile pass #9, P9-6): CommonMark treats text indented 4+
  spaces as an indented code block, so a ``` at that indentation is
  literal content, not markup — `_fence_toggle_offsets` used to strip all
  leading whitespace before checking, so such a line was counted as a
  toggle and could mis-pair a real fence's open/close state one line
  later than it should. Affects `trim_to_last_sentence` and
  `_trim_forwarded_prefix`'s cut-boundary decisions only; the token-run
  rule has no fence reading of its own to affect (see P9-3 above).

### Fixed (hostile pass #10)

- **P10-1 (HIGH): the hard-budget guard spent injected memory it never
  needed to spend, then shed the previous exchange anyway.** On a reusing
  request the array is usually exactly at the recent-window floor
  (`[U_prev, A_prev, U_new]`); the guard's floor alignment used to run
  only when the array held MORE turns than the floor, so it stayed
  unaligned on that exact shape, persona/facts/retrieval were halved and
  dropped for nothing, and the previous exchange was shed anyway by the
  plain fallback loop right after — finishing 6,187-9,213 tokens under the
  limit with memory gone AND the exchange gone. Measured on her real
  branch: 24 of 472 positions (5.1%). Fixed: the compacted branch now
  decides once, by arithmetic, whether spending every spendable injected
  block could ever cover the gap before it crosses into the protected
  recent window — memory pays when it can, the exchange pays only when
  memory provably cannot (`compactor/main.py`, `_enforce_hard_budget`).
- **P10-2 (HIGH): the reuse ceiling's "11,300-token capacity" was in the
  wrong unit, and 12,000 did not clear what her hierarchy actually
  renders at.** `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` raised 12000 ->
  15000, sized off a measured render (her real per-tier chunk sizes plus
  a give-up L3 concatenation — routine whenever `/tokenize` is down) with
  real margin, not the nominal per-tier maxima. Every comment and doc
  citing the old, wrong-unit figure corrected. See [Memory
  budgets](RUNPOD_DEPLOY.md#memory-budgets--raised-defaults-in-v319).
- **P10-3 (MEDIUM): `checks.reuse` could read as a healthy reuse while
  the attempt had actually crashed.** The attempt counter lived at a call
  site three `if`s deep that four of five failure shapes never reached;
  the decline counter lived in one further-nested `if`, so an exception
  left `attempted` incremented with nothing to show it had failed. Both
  now record once, at the top and bottom of the same block, with a
  `reason` (`success`/`no_state`/`no_coverage`/`budget`/`error`) — see
  OPERATIONS.md's `checks.reuse` section.
- **P10-4 (MEDIUM): the P9-6 indent fix (4-space-indented ``` lines) was
  applied to `_fence_toggle_offsets` alone; two siblings still counted
  fences the old, indent-blind way and could disagree by one, inserting
  an unmatched real fence opener into a reply that had none.**
  `_trim_forwarded_prefix` and `_cut_degenerate_span_once`'s
  belt-and-braces check now both use `_fence_toggle_offsets`. The
  fragment-line rule's two inline fence walks (P9-5) are still not
  migrated — reasoned in a code comment (main.py, above the fragment-line
  loop) and in the fix lane's own report, not silently left as-is.
- **P10-5 (LOW, informational): raising Max Tokens past
  `COMPACTOR_GENERATION_RESERVE` lets the reuse stand-in claim up to 75%
  of the window** (72% at or below the recommended Max Tokens, up from
  58% before P10-2 raised the ceiling default — a side effect worth
  knowing if you tune Max Tokens upward). Not triggered at the documented
  Max Tokens (12000). See [RUNPOD_DEPLOY.md → Max
  Tokens](RUNPOD_DEPLOY.md#sampling-parameters).

Operator note: see [RUNPOD_DEPLOY.md → Sampling parameters](RUNPOD_DEPLOY.md#sampling-parameters)
for the mapping between OpenWebUI's Advanced/Custom Parameters and vLLM's
names, and recommended starting values for this model — **Max Tokens 12000,
not 7000**: this model measures 2.0-2.4 characters/token on assistant
replies (not the 4 chars/token a naive estimate assumes), so 7000 tokens is
only ~14-17k characters and would cut some of her ordinary long replies
mid-sentence (hostile pass #7, F6). `repetition_penalty` is now recommended
at 1.05 with Frequency Penalty 0.3, because vLLM 0.19 applies
`repetition_penalty` to prompt tokens as well as output — a high value
discourages words already in the conversation/memory context, not only
words already said in this reply.

Does NOT fix: the model degenerating in the first place (that is a sampling/
model-behavior problem, mitigated by the `repetition_penalty` translation
above, not eliminated by it); the injected-memory-hierarchy's own
(`_redact_degenerate_turns`) whole-reply placeholder wording, left
unchanged; the decoration-fraction/script-drift/short-list-run verdicts,
which still fall back to whole-reply replacement in the forwarded window
because they have no single span to cut around; the pre-existing
TARGET-based stand-in budget gap above roughly 16k max_tokens (not
triggered at the recommended 12000; tracked, deferred); the
cut-and-re-judge loop's algorithm itself (hostile pass #8, P8-5) — passes
are now budgeted by total characters scanned rather than redesigned to
extend a tail cut backwards in one pass, which would need its own
mutation-tested coverage beyond this lane's scope; a merge of two
concurrent conversations losing acknowledged facts (hostile pass #8's
gate note; pre-existing, untouched by this diff, needs its own ticket);
`~~~`-delimited fences, still invisible to `_fence_toggle_offsets`
(hostile pass #9, P9-6) — CommonMark treats ``` and ~~~ as independent
fence-marker families that do not cross-close each other, and this
detector's toggle list is a single flat, character-agnostic parity count,
so adding ~~~ without also tracking marker type would let a ``` block and
a ~~~ block mis-pair under a mixed-marker reply; a real fix needs
per-marker pairing, judged not worth the redesign risk for a LOW-severity
gap on the last V3 release.

---

## [3.1.9.1] — reuse was declining on every production request

Production, 2026-09-16 11:06Z: 37 of 37 requests on one live conversation
logged `compaction skipped: 806 turns need 45 summarization calls, over the
4-call per-request cap` — the exact failure v3.1.9 shipped to remove. The
line before it, every time:

```
summary block: dropped 3 tier item(s) to fit the 1846-token block budget ...
kept 1/4 chapter(s), 1/1 scene(s) and the caller asked for all-or-nothing,
so NOTHING is returned ...
the stored summaries cover 792 of the turns this request would compact,
but they do not fit whole in the 1846 token(s) TARGET (15576) leaves
beside the system prompt, images and recent turns (12706) and one fresh
summary (1024); summarizing from scratch ...
```

**Cause**: `compact_if_needed`'s stand-in for the stored hierarchy was
budgeted against `TARGET_TOKENS` alone, as if the request would ALSO inject
a second, separate copy of the summary — but on a reusing turn that second
copy is always skipped (`sum(in-array)`), freeing its share of the injection
budget. The stand-in never got to spend that freed share, so with long
recent turns it was squeezed to a few hundred tokens and `all_or_nothing`
declined reuse on every single request.

**Fix**: the stand-in may now claim up to what the skipped summary
injection would have spent (60% of the injection budget, capped at
`COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS`) when that is larger than the
TARGET-derived figure — computed by one helper both the array's stand-in and
the separate injection path call, so they cannot drift apart again. At the
numbers above, this is the difference between a ~1,846-token squeeze and a
~6,230-token one; her hierarchy (~5.1k tokens) fits the latter and reuse
fires.

**What the operator sees now, on the same shape of request**: `compacted:
summarized N text turn(s), forwarded 0 verbatim, ... M covered by stored
summaries` instead of `compaction skipped: ... over the 4-call per-request
cap`. N is the turns the stored summaries do not cover yet (the tail past the
last L1 chunk, plus any turn the covered-turn record does not pair); it falls
as L1 rollups catch up. On a copy of the production data: 792 turns replaced,
27 summarized fresh, 1,172,733 -> ~18.7k tokens before memory injection.

**What this does NOT fix**: a hierarchy that still cannot fit even the
larger, injected-share budget still declines exactly as before (same log
line, now naming the real budget source) — no partial/squeezed stand-in was
added, to avoid removing turns the log could not honestly say were covered.

**Correction (hostile pass #9, P9-1/P9-2): the fix above stopped working on
her own conversation within days, and the "her hierarchy (~5.1k tokens) fits
... and reuse fires" claim two paragraphs up was already stale by the time
this release reached hostile review.** Both terms of the stand-in's budget
are capped by `SUMMARY_BLOCK_MAX_TOKENS`, and the fix's own 60%-of-
inject_budget share is a hard-coded `0.6` multiplier that
`SUMMARY_BLOCK_MAX_TOKENS` can only ever LOWER, never raise past. At the
values this release actually shipped (`COMPACTOR_INJECTION_BUDGET_
FRACTION=0.6`, `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS=6230`) the ceiling is a
flat 6,230 tokens — below her hierarchy one day later (~9,050 tokens, up
from ~5.1k) — so reuse silently declined on every request again, the exact
2026-09-16 failure this entry describes fixing. Raising `SUMMARY_BLOCK_MAX_
TOKENS` alone does not help: even at `0.75`/`20000` the ceiling is pinned at
~9,345 by the `0.6` multiplier, still 1,955 tokens short of the hierarchy's
own documented construction capacity (9 L1 scenes + 4 L2 chapters + 1 L3 at
their max sizes = 11,300 tokens), so reuse would turn itself off again
within one L1 rollup chunk regardless of how the fraction is tuned. **Fixed
in v3.1.9.2** (hostile pass #9): the reuse stand-in now has its OWN budget
formula (`_standin_reuse_ceiling`, `COMPACTOR_STANDIN_BUDGET_FRACTION`,
default 1.0 of the injection budget) instead of sharing the separately-
injected summary block's 60% formula — the two situations only looked
alike; nothing else spends the stand-in's share on a reusing turn, because
the separate injection is SKIPPED, not shrunk. Shipped defaults moved to
`COMPACTOR_INJECTION_BUDGET_FRACTION=0.75` and `COMPACTOR_SUMMARY_BLOCK_
MAX_TOKENS=12000`, which together clear the 11,300-token capacity with
margin (empirically measured against a hierarchy built to exactly that
capacity: true render cost 11,400-11,500 tokens). `/health/full` now
reports `checks.reuse` (`attempted`, `declined_budget`,
`declined_recently`, and the two numbers — ceiling and other-consumers
total — behind the most recent decline; no conversation text, ever), so
the next time her data outgrows the arithmetic again the operator sees it
without reading request logs.

**Correction (hostile pass #10, P10-2): "clears the 11,300-token capacity
with margin" was also wrong, in a way the "empirically measured" render
above happened to paper over.** The 11,300 figure (`9*L1_MAX_TOKENS +
4*L2_MAX_TOKENS + L3_MAX_TOKENS`) is in OUTPUT tokens; the ceiling above is
checked against `_estimate_block_tokens`, which prices non-ASCII at one
token per UTF-8 BYTE — up to 4.27x over for CJK, 2.34x for Greek — so the
two numbers were never in the same unit, and this user quotes scripture.
Separately, her real L1/L2 chunks already exceed the PER-TIER maxima that
figure assumes (measured: 8 L1 chunks mean 561, max 792 against
`L1_MAX_TOKENS=500`; 4 L2 chapters mean 1,102, max 1,271 against
`L2_MAX_TOKENS=1200`), and L3 is not bounded by `L3_MAX_TOKENS` in
practice — a stalled `/tokenize` (a live state on this pod) routinely
makes the L3 rollup give up and CONCATENATE 2-3 parts instead of
summarizing them, and that concatenation is what gets stored and carried
into every later refresh. Measured against her real chunks plus a real L3:
steady-state peak 11,728 (272 tokens of the claimed margin, not "1,955
short" nor comfortably clear); with a 2x-part give-up concatenation,
13,860 — over 12,000, and reuse declines again exactly as it did before
this entry's own fix. **Fixed in v3.1.9.2** (hostile pass #10):
`COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` raised to `15000`, sized off the
measured give-up-L3 peak (13,860) with ~1,140 tokens of real margin rather
than off the wrong-unit nominal figure — see that variable's own comment
in `Dockerfile`/`runpod.env.template` for the full arithmetic. This does
not claim the ceiling "cannot be outgrown"; a 3x-part give-up
concatenation (~15,860) still declines, safely, back to summarizing from
scratch.

---

## [3.1.9] — operator notes (the last V3 release)

Operator-facing notes only: what to check on the pod, what not to run, and
what the alarms currently mean. The code changes of the v3.1.x line are in the
git tag messages. Every item links to the runbook that carries the commands.

### Before deploying

- **`WEBUI_DB_LOCAL=false` is a hard precondition** of this and every deploy.
  Write it as exactly `false`. A missing or EMPTY row still means `true` (the
  database gets moved to local disk). New in v3.1.9: `1`/`yes`/`on`/`True`
  now also mean `true` (older images read them as false), `0`/`no`/`off`
  mean `false`, and any other value **refuses to boot** with a
  `REFUSING TO START` banner in the RunPod Logs tab — fix the row and
  redeploy; nothing was touched. After boot, the Logs tab must show
  `WEBUI_DB_LOCAL=false (explicitly set (false))`; the full check is in
  [RUNPOD_DEPLOY.md → WEBUI_DB_LOCAL](RUNPOD_DEPLOY.md#webui_db_local--a-hard-deploy-precondition).
- **Take a backup by hand on the old image** and confirm it prints `[OK]`:
  `/opt/compactor-venv/bin/python /opt/compactor/backup.py --once; echo "EXIT=$?"`.
- **If the volume is tight, prune old backups by hand before deploying.** The
  old image has not pruned since 2026-08-30 (its "memory shrank" check is
  noise); v3.1.9 resumes pruning by itself, but only on its first nightly
  cycle. The previewed manual prune is in
  [OPERATIONS.md → Nightly "memory shrank" alert](OPERATIONS.md#nightly-memory-shrank-alert--noise-on-v3161-to-v318-a-real-signal-from-v319-except-one-item).

### Verifying the deploy

- `/health/full` status is `ok`, or `degraded` only for `memory tail skipping`
  (see below) — and read the unreadable-memory and newest-backup lines in
  [OPERATIONS.md → Reading /health/full](OPERATIONS.md#reading-healthfull--do-not-trust-status-alone).
  From v3.1.9 `status` also degrades on a backup older than 36 hours or no
  backups after a day of uptime, so a backup reason right after the deploy
  means the pre-deploy backup did not happen.
- `cat /data/logs/selftest.log` ends `=== N/N passed, 0 failed ===`.
- After her first message: `/health/full`'s `memory_tail.stored` (or
  `stored_trimmed`) has gone up, and the compactor log has an
  `injected memory [...]` line for her conversation.
- **Do not judge the deploy by "hard budget enforced" lines.** The v3.1.7 tag's
  VERIFY step ("tokenize.ok; the hard budget enforced warnings drop sharply")
  cannot show the change it names: the token counting and budget path were
  unchanged in v3.1.7, so on her long chat `compaction skipped: … need N
  summarization calls` followed by `hard budget enforced: … dropped ~200 old
  turn(s)` continued after that deploy, and a drop in them can come from an
  unrelated cause (a new or shorter chat). Those lines are the context
  starvation that the identity + optional cap procedure addresses; they are not
  a deploy failure, and their disappearing is not proof of success. (hostile
  review of v3.1.7, reviewer A, F3.)
- **`memory tail skipping` degrading `/health/full` for 5 minutes at a time is
  expected** several times a day on her traffic: a reply that looped or was cut
  off without a full sentence is kept out of memory on purpose. Which outcomes
  are faults is in
  [OPERATIONS.md → What "memory tail skipping" means](OPERATIONS.md#what-memory-tail-skipping-means).
  This reason is new to the pod (v3.1.6.1 has no memory-tail tracking) and is
  unchanged in v3.1.9 by design.
- **What to expect in the logs on her first two messages after the upgrade**
  (measured against a copy of her real v3.1.7-written state): her **first**
  message under v3.1.9 may still log `compaction skipped: … turns need N
  summarization calls` / `hard budget enforced …` — it adopts the v3.1.7-era
  summary record once (a one-time legacy-adoption cost) and is not itself a
  fault. From her **second** message onward, requests reuse the stored
  summary hierarchy: no more refusals, no more shedding, a normal-sized
  forwarded payload. If refusals continue past the second message, something
  is wrong; if only the very first one refused, that is the expected shape.
- **A `hierarchy is N turns behind` reason with `verdict: unknown` right
  after the deploy (or any restart) is expected, not a fault.** The
  catch-up verdict is process-local evidence the tail records as it runs;
  a freshly started process has none yet, so any conversation already
  behind reads `unknown` until her next message on it — this is not
  itself evidence of a stall, and it typically starts advancing on her
  very next message. The reason itself offers `/compact` as an option "if
  it should not wait for that", not as a required action; prefer waiting
  for her next message right after a deploy. See "Summary hierarchy
  catch-up" below.

### The current date and time

- **The model now knows the date and time, in her browser's time zone.**
  After the deploy, and only once her chat logs `source=header`, add the line
  `User timezone: {{CURRENT_TIMEZONE}}` to her model's System Prompt in
  OpenWebUI (Admin Panel → Settings → Models). Editing the system prompt of a
  chat still on `source=hash` forks its memory. **While her chat is on
  `source=hash`, leave the system prompt alone and set
  `COMPACTOR_TIMEZONE=<her IANA zone>` (for example `America/Phoenix`) in the
  RunPod template instead:** the model is told the right time, it just does
  not follow her device if she travels. With neither, UTC.
- **Use the zone's current canonical name, not an old-style one.** The image
  now includes `tzdata-legacy`, so pre-merge spellings like `US/Arizona` or
  `Asia/Calcutta` resolve too — but do not rely on that for a zone not
  listed on this pod; prefer the canonical name (`America/Phoenix`,
  `Asia/Kolkata`, ...). A name that does not resolve at all falls back to
  UTC with one ERROR line at boot and no `/health/full` status change (see
  the Check below).
- **Check (route-dependent — the two setups above check differently):** on
  `source=header` with the system-prompt line, after her next message
  `/health/full` → `config.time_injection.last_source` is `browser` and
  `current_line` shows her local time. On `source=hash` with
  `COMPACTOR_TIMEZONE` set (today's setup), `last_source` is `env`, never
  `browser` — that is correct, not a fault; confirm the zone itself in
  `config.time_injection.last_timezone` instead. Either way, if
  `config.time_injection.fallback_error`
  is non-empty, the configured zone name did not resolve and every message is
  dated in UTC with `/health/full` `status` still `ok` — this is a silent
  failure with no `status_reasons` entry, so check `fallback_error`
  explicitly rather than trusting `status`.
- Title, tag, follow-up and query-generation calls (OpenWebUI "task
  traffic") are never dated under header identity. Under today's hash
  identity, the same calls are recognized (and left undated) only once a
  conversation has genuinely reached `COMPACTOR_TASK_TRAFFIC_MIN_POSITION`
  (default 4) turns — the first one or two task calls on a brand-new
  template may be dated, harmlessly. A real first message from her that
  happens to hash-collide with an older, already-deep conversation's opener
  is still dated correctly, but is not memorized (a known hash-identity
  limitation; header identity removes it — see RUNBOOK_MEMORY_IDENTITY.md).
- To switch it off: `COMPACTOR_TIME_INJECTION=false`. Details:
  [RUNPOD_DEPLOY.md → The current date and time](RUNPOD_DEPLOY.md#the-current-date-and-time).

### Voice

- **Read-aloud works from v3.1.9.** On v3.1.6.1-v3.1.8 the speaker button never
  played anything, because the image had no `ffmpeg` (OpenWebUI converts the
  speech to MP3 with it). After deploying, press read-aloud on any reply and
  hear audio.
- **Long recordings** (over 20 MB, about 7-8 minutes) now transcribe instead
  of failing within seconds. An 11-minute recording took about 5 minutes on
  CPU.
- **Video files:** only `.webm` video is transcribed by default. To include
  `.mp4` and iPhone `.mov` soundtracks, change the setting in the Admin Panel,
  not the RunPod template, and keep `audio/*` in it. See
  [RUNPOD_DEPLOY.md → Audio and video FILES](RUNPOD_DEPLOY.md#audio-and-video-files-attached-to-a-chat).

### Her conversation's identity

- The OpenWebUI filter `pipelines/conversation_id_header.py` **cannot** deliver
  the chat id on OpenWebUI 0.11.0 (OpenWebUI discards the metadata it writes).
  Its docstring's "interlock enforced in code" did not exist. The working route
  is an OpenWebUI connection header,
  `{"X-Conversation-Id": "{{CHAT_ID}}{{TASK}}"}` — with `{{TASK}}`, or her
  title/tag/follow-up traffic is memorized as her conversation.
- The order is: merge the old hash id into the chat uuid **before** her next
  message, then add the header, then verify `source=header`, and only then
  (optionally) the history cap, sized from her data (start at 40, not 60).
  Full procedure: [RUNBOOK_MEMORY_IDENTITY.md](RUNBOOK_MEMORY_IDENTITY.md).
- Rollback order: cap to 0 → remove the header → reverse merge.

### Summary hierarchy catch-up

- **A summary hierarchy that has fallen far behind (days of rollup failures, a
  vLLM outage, an upgrade that finds a stale watermark) now catches up in
  bounded steps instead of all at once.** Before v3.1.9, the background tail
  (and the one-shot backfill rollup for a newly-discovered V1 conversation)
  drained every L1/L2/L3 rollup a backlog needed in ONE pass — however many
  vLLM calls that took, on the same single GPU she is chatting on. Both now
  spend `COMPACTOR_TAIL_ROLLUP_MAX_CALLS` (default 4) real vLLM calls as a
  **per-turn budget that bounds where a rollup unit is allowed to START, not
  a hard ceiling on the turn**: a unit that starts always finishes, so one
  turn can spend up to `(budget − 1)` plus that one unit's own cost — normally
  a few calls, but measured at 6-16 calls for a single unit when `/tokenize`
  is down. The watermark is persisted, so this always converges over
  successive turns even when a turn overshoots.
- **`/health/full`'s `checks.hierarchy` now carries real evidence, not a
  poll-to-poll guess.** Each conversation whose recent hierarchy lag is over
  the limit gets a `verdict`: `converging` (the watermark advanced within the
  last 15 minutes — no action needed, `status` stays `ok`), `stuck` (20
  budgeted passes in a row with work due and no advance), or `unknown` (no
  evidence yet for this process — almost always a recent restart; treat as
  "wait for her next message", not as a confirmed stall). Only `stuck` and
  `unknown` add a `status_reasons` line and degrade `status`; `converging`
  never does. `checks.hierarchy.catching_up` is the single worst conversation
  (unchanged shape); the new `catching_up_all` lists every one currently over
  the limit, so a converging conversation with a large lag can no longer mask
  a genuinely stuck one with a smaller lag.
- **A summary tier (L2 fold or L3 refresh) that fails on every attempt no
  longer freezes the whole hierarchy behind it.** Before this fix, the drain
  checks tiers in priority order (L3, then L2, then L1) every call, and a
  persistently failing upper tier aborted the ENTIRE pass — so L1 (which is
  injected into every request) stopped advancing too, forever, from the turn
  the failure started. Now a failing tier is skipped for that call only, and
  the tiers below it keep advancing. Look for `conv=<id>: L3 refresh failed
  and will be retried next turn without blocking L1/L2 — ...` (or the L2
  equivalent) in the compactor log — that ERROR, once per conversation per
  tier, is what to grep for if `catching_up_all` or the `/health/full` reason
  names a failing tier. When it does, running `/compact` will NOT clear the
  backlog (it runs the identical drain and hits the same failure); fix the
  underlying cause first (commonly an unreadable archive sidecar,
  `summaries/<conv>.archive.json`).
- **The catch-up INFO log line now names what is actually pending**, instead
  of always describing the L1 watermark (which used to print "0 turn(s)
  still uncovered … ~0 more turn(s)" — misleadingly — when only an L2 fold
  or the L3 refresh was still due). It now says `L1 N turn(s) still
  uncovered`, `an L2 fold pending`, `an L3 refresh pending`, or a
  comma-joined combination, with a turn-count ETA only when L1 itself is the
  pending tier.

### Memory budgets

- **The memory-injection budgets the owner raised by hand on the running pod
  (a supervisord `environment=` edit, applied 2026-09-15, lost on every
  container restart) are now the SHIPPED DEFAULTS.** The v3.1.9 deploy
  bakes them into the image and the RunPod template, so the live edit is no
  longer needed and will not silently revert on the next restart or
  redeploy:

  | Variable | Code default | v3.1.9 shipped default |
  |---|---|---|
  | `COMPACTOR_MAX_FACTS_TOKENS` | 1500 | **3500** |
  | `COMPACTOR_INJECT_FACTS_TOKENS` | 400 | **600** |
  | `COMPACTOR_MAX_RETRIEVAL_TOKENS` | 1500 | **3500** |
  | `COMPACTOR_INJECTION_BUDGET_FRACTION` | 0.5 | **0.6** |
  | `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` | 12000 | **6230** |

  The fraction and the two raised caps move together, not independently:
  `inject_budget = effective_limit × INJECTION_BUDGET_FRACTION` is shared by
  persona + summary + facts + retrieval, and retrieval (priority 3) is
  dropped WHOLE by `_bound_injected_blocks` when it does not fit. At the
  raised facts/retrieval caps under the OLD 0.5 fraction, retrieval would
  have been silently dropped from every request; 0.6 gives it room, with the
  summary block pinned at what it measured itself needing (6,230, below its
  own 12,000 code default). Do not change one of these five without the
  others. **This changes only the environment defaults baked into the image
  and template — none of the Python code defaults (`facts.py`,
  `retrieval.py`, `main.py`, `summarizer.py`) were changed**; an operator
  who has already overridden any of these five in their own template keeps
  their own value.
- Details and the pod measurement behind these numbers:
  [RUNPOD_DEPLOY.md → Memory budgets](RUNPOD_DEPLOY.md#memory-budgets--raised-defaults-in-v319).

### Rolling back to an older image

- **Set the History cap's `max_turns` to 0 BEFORE redeploying an older image**,
  and leave it at 0 until the newer image is back and has served one uncapped
  message. Rolling back with the cap on leaves a permanent, unlogged hole in
  her summary hierarchy (reviewer C, F5). Keep `WEBUI_DB_LOCAL=false`, spelled
  exactly that way: the older images read `1`/`yes`/`True` as false and
  v3.1.9 reads them as true.

### Backups — what changes with v3.1.9

- **The "memory shrank … NOT pruning" alert now means something — with one
  known false-alarm shape, not yet fixed in code.** On v3.1.6.1–v3.1.8 it
  fired every night on normal fact eviction and summary rollups, so nothing
  pruned after 2026-08-30. v3.1.9 compares what cannot come back (active +
  archived facts together, the highest summarized turn, archived chapters,
  indexed exchanges), so **most `NOT pruning — memory shrank` alerts on
  v3.1.9 are real** and should be treated that way — **except** the item
  named `summary_active_bytes`: an ordinary L1→L2 (or L2→L3) fold routinely
  drops it by more than half (one chapter is shorter than the chunks it
  replaces) and trips the false-alarm shape on a night that lost nothing.
  Before treating a `summary_active_bytes` alert as real loss, check whether
  it is this shape — see
  [OPERATIONS.md → Nightly "memory shrank" alert](OPERATIONS.md#nightly-memory-shrank-alert--noise-on-v3161-to-v318-a-real-signal-from-v319-except-one-item).
  **After the deploy, look for the first nightly cycle** (within about a day):
  `grep -aE "NOT pruning|pruned [0-9]+" /data/logs/backup.log | tail -3` should
  show `backup ok: … pruned N; …` (N may be 0).
- **A failed backup is retried after 15 minutes**, not 24 hours (log:
  `backup cycle failed: …; retrying in 15 min`), and `/health/full` degrades
  on a stale or missing backup. A hot rollback journal no longer has to fail
  the backup: v3.1.9 backs up from a copy and logs a WARNING containing `this
  is the hot rollback journal signature` — the live journal still needs the
  repair in OPERATIONS.md. That fallback is not yet confirmed against the
  production image's SQLite, so **after any "readonly database" or "database
  is locked" episode, still run a backup by hand and read its output**
  ([OPERATIONS.md → Backups stopped or failing](OPERATIONS.md#backups-stopped-or-failing-readonly-database--database-is-locked)).
- **Restore: use the manual move-aside procedure** in
  [OPERATIONS.md → Restore from a backup](OPERATIONS.md#-restore-from-a-backup-recover-lostcorrupted-memory),
  which stops all four writers (`openwebui compactor backup webuidb-sync`).
  `backup.py --restore` is rewritten in v3.1.9 (it stages everything first and
  sets the old database journal and memory store aside in `/data/forensics`;
  `backup.py --list-pre-restore` lists them) but has not yet passed the final
  hostile review, so it is available, not recommended. On v3.1.6.1–v3.1.8,
  including an image you roll back to, **never run `backup.py --restore`**: it
  can leave `webui.db` malformed and deletes the live store before copying.
- `df -h /data` shows the whole MooseFS cluster, not your volume's quota; use
  `du -sh /data/*`.
- Service logs are in `/data/logs/`, not `/var/log/supervisor/`.

---

## [3.0.1] — patch: one image no longer poisons a conversation (2026-08-24)

**The bug (found by the test user):** uploading a picture on a text-only
backend broke the conversation *permanently* — even plain text messages after
it failed. Mechanism: OpenWebUI re-sends the full history (image included)
with every message; V3.1 compaction deliberately preserves image turns; and
vLLM 400s each request (`"…is not a multimodal model"`). vLLM never crashed —
every request carrying the image was cleanly rejected, forever. The compactor
was forwarding content the backend cannot accept: an unverified modality
boundary (the same bug class as the whole rc line).

### Fixed
- **Modality guard.** At startup the compactor resolves whether `MODEL_REPO`
  can see (HF config `vision_config`; override with
  `COMPACTOR_BACKEND_MULTIMODAL=auto|true|false`). On a text-only backend,
  image parts are replaced with an **honest placeholder** — the model is told
  an image was attached and that it cannot see it (degrade honestly, don't
  silently vanish it) — text parts are preserved, and the V3.1
  image-preserving paths simply never fire.
- **Reactive backstop:** a vLLM `not a multimodal model` 400 flips the cached
  modality, so even if startup detection is wrong the *next* message strips
  and the conversation heals instead of staying poisoned. Already-poisoned
  conversations recover automatically on their next message.
- Tier-1 `test_modality.py` covers the strip semantics and the backstop.

Operator note: pairing this with the OpenWebUI per-model **Vision** capability
toggle (off for text-only models) prevents the UI from offering uploads at
all; the guard protects any client regardless.

---

## [3.0] — V3 consolidation & dependency hardening — **released 2026-08-23**

**Goal:** stabilize the V3.x line (vision + STT + TTS, shipped incrementally as
3.1 / 3.2 / 3.3 on a rolling image) into one audited, reproducible release:
the dependency-security pass plus the bug-fix scrub that followed (eight rcs,
two production incidents, two adversarial audit rounds — see "Fixed" below).

**Released as:** `angreg/zions-light-ai:v3.0-cu12` = `:v3.0` = `:latest`
(promoted from the validated `:v3.0-rc8-cu12` build; git tag `v3.0`).
Rollback targets: `:v3.0-rc7-cu12` (pre-Cydonia-default, code-identical) and
`:v3-snapshot` (the pre-audit 3.3 image — functional but carries the
pre-audit vLLM; last resort only). Patch releases (v3.0.x) will carry any
post-release fixes.

### Security (PyPI/OSV audit, 2026-06-30)
- **vLLM `0.14.1` → `0.24.0`.** The 0.14.1 pin had accumulated ~18 CVEs + ~17
  GHSAs since it was set (it only ever fixed CVE-2026-22778); 0.24.0 is the first
  release with no known advisories. vLLM binds to localhost behind the compactor,
  which bounded exposure in the meantime.
  **Correction (rc1 → rc2):** the original note here claimed 0.24.0 kept cu128
  with "no CUDA-base change" — that was wrong. 0.24.0 ships **CUDA-13** kernels
  (rc1 failed on the CUDA-12 image with `ImportError: libcudart.so.13`), so the
  default build moved to a **CUDA 13 base (`13.0.0-runtime`) + cu130** torch
  channel and now requires a host **driver ≥580** (Ampere/A40 is supported on
  580 — the gate is the host, not the card). A documented **CUDA-12 fallback**
  profile (base `12.6.3` + cu128 + vLLM **0.19.0**, the last CUDA-12 release,
  ~32 advisories) is provided for driver-570 hosts. The three build args
  (base / `TORCH_CUDA` / `VLLM_VERSION`) are now a matched set — see the
  Dockerfile header.
- **transformers `>=4.50` → `==5.12.1`.** The unbounded floor was silently
  resolving to a 5.x major already; the last 4.x (4.57.6) carries open CVEs fixed
  only in 5.x. Pinned to the clean 5.12.1 (used tokenizer-only on a known model).
  The 4→5 major is the item the rebuild **boot self-test must confirm**; fallback
  is `transformers==4.57.6`.
- **chromadb — known/accepted, not exposed.** CVE-2026-45829 (pre-auth code
  injection) affects all 1.x with no fix yet, but only via the Chroma **HTTP
  server**; we run chromadb **embedded** (in-process, sqlite-backed), so the
  vulnerable endpoint is not present. Pinned `==1.5.9`; bump when patched.

### Changed
- **All runtime Python deps pinned to exact, audited versions** for reproducible
  builds (the Dockerfile was otherwise non-reproducible — only vLLM was pinned):
  `compactor/`, `stt/`, `tts/` requirements + `open-webui==0.11.0` in the
  Dockerfile. Pins: fastapi 0.138.2, uvicorn 0.49.0, httpx 0.28.1,
  python-multipart 0.0.32, faster-whisper 1.2.1, piper-tts 1.4.2, fastembed
  0.8.0, transformers 5.12.1, chromadb 1.5.9, open-webui 0.11.0.
- **open-webui `0.10.1` → `0.11.0` (rc3 → rc4).** rc3 pinned 0.10.1 (latest at
  audit time), which carries a **regression** (open-webui issue #20565): the
  0.10.x memory feature calls `.get()` on a model's `capabilities`, which is
  `None` for a model auto-discovered from an OpenAI connection (our vLLM) —
  crashing every chat with `'NoneType' object has no attribute 'get'`
  (`main:process_chat:1504`). Fixed in 0.10.2; **0.11.0** both fixes it *and*
  clears all **17** known advisories that 0.10.1/0.10.2 each carry (→ **0**). 0.11.0
  migrates the OpenWebUI DB schema forward (rollback to 0.10.x is unsupported
  without a DB reset — the compactor's own memory store is unaffected). Lesson
  logged: do not pin a fast-moving UI to its bleeding-edge release un-validated.

### Fixed
- **CRLF line endings baked a broken container entrypoint.** On Windows
  (`core.autocrlf=true`, no `.gitattributes`), `entrypoint.sh` and
  `supervisord.conf` checked out CRLF, so `docker build` baked a `#!/bin/bash\r`
  shebang and the container failed at start with "not found" / "bad
  interpreter" — this broke v2.2.1 and every image built on Windows. Added
  `.gitattributes` (`* text=auto eol=lf` + explicit `eol=lf` for
  `*.sh`/`entrypoint.sh`/`*.conf`/`Dockerfile`/`*.py`) and renormalized the tree
  to LF (verified byte-accurately: CR=0 across all image-copied files).
- **Dependency-boundary hardening (rc5, from a full architecture audit).** The
  recurring failure class this release kept hitting — our code trusting a
  dependency's output shape — swept systematically:
  - **Silent fact loss (data integrity).** The async tail wrote facts built on a
    snapshot read *outside* `conv_lock`, so two overlapping turns on one
    conversation could lose a fact permanently (the lock serialized the writes
    but not the read). Facts are now re-read under the lock and reconciled via
    `_merge_touched`: disk is authoritative for membership (a `/forget` is never
    resurrected), the snapshot contributes only LRU touches, and `last_used`
    only ever moves forward.
  - **Opaque 500 on a non-JSON vLLM reply** (HTML 502, truncated body):
    `r.json()` is guarded → friendly 502 instead of an unhandled decode error.
  - **Garbled stream on a vLLM 4xx/5xx:** the streaming path relayed vLLM's JSON
    error body raw into an SSE channel; it now checks status and degrades to the
    same friendly chunks the connection-error branch uses.
  - **Consecutive system messages** (V1 compaction summary + injected memory
    block) could trip Mistral-family templates' 400 — collapsed into one just
    before forwarding, with multimodal (image) content left intact.
  - **Unbounded proxy timeout:** `timeout=None` also removed the *connect*
    timeout, so a stalled vLLM socket hung a request forever. Connect/write/pool
    are now bounded; read stays unbounded for long generations.
  - **`atomic_write_json` now fsyncs the parent directory**, so the rename itself
    is durable — the previous "atomic on POSIX" claim was weaker than it read on
    a distributed network volume.
  - Tier-1 `compactor/test_concurrency_guards.py` covers both new helpers,
    including the lost-update and Mistral-400 scenarios.
- **Driver preflight is now build-arg-aware and fails closed.** `entrypoint.sh`
  Check 3 reads the baked-in `TORCH_CUDA` and requires driver ≥580 for cu130
  (CUDA 13) vs ≥525 for cu128/cu126 — instead of hardcoding cu128/≥525, which
  would have let a CUDA-13 default image false-pass a driver-570 host and then
  crash. A missing/unknown channel now defaults to the strictest floor.
- **OpenWebUI SQLite hardened for the network volume.** OpenWebUI's DB
  (`/data/openwebui/webui.db`) sits on the RunPod network volume, and 0.11.0
  defaults to **WAL** journal mode — which needs an mmap'd `-shm` shared-memory
  index that network filesystems don't support, so WAL was *causing* the
  `database is locked` errors seen in rc3 testing. Baked env: `DATABASE_ENABLE_SQLITE_WAL=false`
  (rollback journal — no `-shm`/mmap), `DATABASE_SQLITE_PRAGMA_BUSY_TIMEOUT=10000`
  (10s lock-wait; not higher, which would just mask real deadlocks), and
  `DATABASE_SQLITE_PRAGMA_MMAP_SIZE=0` (DB mmap also unreliable over network FS).
  `synchronous` stays at OWUI's `NORMAL` default — `FULL` would lengthen writes
  on slow network storage and worsen contention. All overridable via env
  (documented in `.env.example`).

### Fixed (rc5 → rc8 — on-pod incidents + two audit rounds)
- **Default model is now bootable (rc8).** The image's built-in default was
  `magnum-v4-22b` with no quantization flag — a combination that **cannot boot
  on the A40** the image is documented for (the out-of-the-box trap PR #16
  flagged back in V1). The default is now the production-validated pair:
  `MODEL_REPO=coder3101/Cydonia-24B-v4.3-heretic-v4` +
  `VLLM_EXTRA_ARGS="--quantization fp8"` (changed **together** — a 24B without
  fp8 is a boot OOM). Every real deploy still pins both via
  `runpod.env.template`, so this only changes bare-default behavior.
- **Empty-`messages` guard (rc5).** OpenWebUI 0.11 background/task calls can
  send `messages: []`, which crashed vLLM's chat templating with an opaque
  "list index out of range." Now a clean OpenAI-shaped 400
  (`code=empty_messages`) with a sender-identifying log; hardened in rc7 to
  also reject non-dict items.
- **Dependency-boundary fixes (rc5, from the first audit):** lost-update fix in
  the async facts tail (`_merge_touched`), guarded `r.json()` on the chat path,
  stream 4xx handling, adjacent-system-message merge, bounded
  connect/write/pool timeouts, parent-dir fsync in `atomic_write_json`.
- **Context-overflow cascade (rc6, the 2026-08-13 outage).** The summarize call
  itself exceeded the model window; compaction "degraded" by forwarding the
  original oversized messages; injection piled on more → hard 400. Fixed with
  a map-reduce summarizer (`_chunk_to_budget`) + `_enforce_hard_budget` final
  pre-flight + honest 4xx degradation (a healthy backend's rejection is no
  longer reported as "the model is restarting"). New env:
  `COMPACTOR_SUMMARY_INPUT_RESERVE`, `COMPACTOR_GENERATION_RESERVE`.
- **Promotion-review round (rc7) — 18 confirmed findings, the critical ones:**
  - **BLOCKER: compaction emitted assistant-first conversations.** With even
    `KEEP_RECENT_TURNS` and a real request's odd non-system count, every
    *successful* compaction violated the Mistral-family template's user-first
    rule → deterministic 400, conversation dead. Latent since V1; shielded by
    the summarize-overflow bug, exposed the moment rc6 fixed it.
    `split_messages` now aligns the keep window to a user turn.
  - `_enforce_hard_budget` could stop shedding mid-pair (assistant-first — the
    same 400 it prevents) and re-tokenized the entire list per dropped message
    (O(N²) blocking the event loop + healthchecks). Now: alternation-preserving
    shedding, per-message token arithmetic with bounded verify recounts, a
    cheap prescreen, and the whole guard runs in a threadpool.
  - The guard now honors the request's own `max_tokens` (vLLM enforces
    prompt + completion ≤ window; a fixed reserve alone wasn't enough).
  - Map-reduce summarize batches run **concurrently** (were sequential —
    multi-minute stalls) and the reduce step no longer violates its own input
    budget (hierarchical fold).
  - Text-only content-parts system messages are flattened before the
    adjacent-system merge; `/v1/models` got the same non-JSON-body guard as
    the chat path; a client disconnect mid-stream no longer memorizes the
    half-reply (facts/RAG/summaries skip incomplete streams).
- **`clean-models.sh` (rc4/rc5 era, previously unlogged):** operator tool to
  reclaim volume space from stale model caches — dry-run by default, active
  model always protected. Bundled at `/opt/clean-models.sh`.

### Validation gate (operator — before rebuild is promoted to `:latest`)
- Rebuild the image; **boot self-test must PASS** — including the STT + TTS
  probes and a chat round-trip that exercises the transformers tokenizer on 5.x.
- **Known gap (acknowledged):** the boot self-test's chat round-trip is far
  below the compaction/budget thresholds, so the guard paths above are
  validated only by Tier-1 (`test_budget_guard.py`, `test_concurrency_guards.py`)
  — promotion additionally requires a **manual long-conversation check** on the
  pod (drive a conversation past `COMPACTOR_TARGET_TOKENS` and confirm a 200 +
  the `compacted:` log line, not a 400).
- **OpenWebUI chat works end-to-end** — the 0.10.1 `capabilities`-None crash is
  fixed by the 0.11.0 pin (rc3 hit it; confirmed via the model-re-save
  workaround before the pin fix). No native-tools/`tool_choice` error (leave
  Function Calling = Default; tool calling is a V4 feature).
- Real voice round-trip works; vLLM 0.24.0 serves the configured model on a
  **driver ≥580** host (A40 or newer). On a driver-570 host, use the CUDA-12
  fallback profile (vLLM 0.19.0) instead.

---

## [3.3] — Text-to-speech (voice output)

**Goal:** let the assistant *speak* — bundle a local TTS service so OpenWebUI's
"read aloud" works, with nothing leaving the pod. The mirror of V3.2 (synthesis
instead of transcription); independent of the memory pipeline.

Image: folded into the current image line; the Piper service is on by default
(`TTS_ENABLED=true`), CPU + torch-free.

### Added
- **TTS service** (`tts/server.py`) — a thin FastAPI wrapper around **Piper**
  (onnxruntime, CPU) exposing the OpenAI audio API: `POST /v1/audio/speech`,
  `GET /health`, `GET /v1/models`. WAV native (+ pcm); mp3/opus/aac/flac via
  ffmpeg if present, else a graceful WAV fallback. Own venv (`/opt/tts-venv`),
  own supervisord program (`[program:tts]`, port 9001), default voice
  `en_US-lessac-medium` prebaked at `/opt/tts-voices`.
- **OpenWebUI wiring** — the TTS engine is pre-pointed at the local service
  (`AUDIO_TTS_*`); the read-aloud control works with no further setup.
- **Boot self-test TTS probe** (`selftest.py` `_check_tts`) — POSTs a tiny text
  and asserts non-empty audio with an `audio/*` content type. Gated on
  `TTS_ENABLED`.
- Tier-1 `tts/test_tts.py` — wav↔pcm helpers, format encode/fallback, and the
  `/v1/audio/speech` core (200 / 400 / 503 / 500) with a fake engine.
- Config (`.env.example`) + docs (RUNPOD_DEPLOY / USER_GUIDE / ROADMAP).

### Notes
- Piper (onnxruntime) chosen over Kokoro-82M to keep the aux-venv pattern
  torch-free and CPU-only; Kokoro is documented as an optional quality swap.
- ffmpeg is deliberately not bundled (keeps the image lean); WAV is what
  OpenWebUI plays, so it isn't needed for the default flow.

---

## [3.2] — Speech-to-text (voice input)

**Goal:** let the assistant *hear* — bundle a local speech-to-text service so
microphone input in OpenWebUI "just works," with nothing leaving the pod. STT is
independent of the memory pipeline (audio → text; the transcript then flows
through the compactor like any typed message).

Image: folded into the current image line; the Whisper service is on by default
(`STT_ENABLED=true`), and runs on CPU by default so it never competes with vLLM
for VRAM.

### Added
- **STT service** (`stt/server.py`) — a thin FastAPI wrapper around
  **faster-whisper** (CTranslate2) exposing the OpenAI audio API:
  `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`, `GET /health`,
  `GET /v1/models`. Renders every OpenAI response format (json / text /
  verbose_json / srt / vtt). Own venv (`/opt/whisper-venv`), own supervisord
  program (`[program:stt]`, port 9000), default `base` model prebaked at
  `/opt/whisper-models`.
- **OpenWebUI wiring** — the STT engine is pre-pointed at the local service
  (`AUDIO_STT_*`); the microphone button works with no further setup.
- **Boot self-test STT probe** (`selftest.py` `_check_stt`) — transcribes a tiny
  generated WAV on boot and asserts a well-formed response, catching the
  "service up but broken" failure a port check misses. Gated on `STT_ENABLED`.
- **Quality eval** (`tests/eval/`, excluded from the image) — a word-error-rate
  metric (`wer.py`, Tier-1-tested) + `stt_eval.py`, which scores transcription
  accuracy against operator-supplied speech clips through the live service.
- Config (`.env.example`) + docs (RUNPOD_DEPLOY / USER_GUIDE / ROADMAP):
  `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_DOWNLOAD_ROOT`, `STT_ENABLED`,
  CPU-vs-GPU guidance.

### Notes
- CPU-by-default is deliberate: vLLM reserves ~90% of the GPU, so transcribing
  on-GPU would fight it for VRAM (the A40 OOM lesson). faster-whisper is fast
  enough on CPU for the small/base models.

---

## [3.1] — Vision (image understanding)

**Goal:** make the assistant able to *see* — and make the compactor handle
images correctly so a vision-language model is safe to run. Vision is an
opt-in `MODEL_REPO` swap to a VLM, not the default (the best creative-writing
and the best vision models are not the same model today).

Image: folded into the current image line; enable by setting a VLM
`MODEL_REPO` (presets in `.env.example`).

### Added
- **Image-aware token budgeting** — `count_tokens` adds a per-image estimate
  (`COMPACTOR_IMAGE_TOKENS`, default 768) so VLM requests don't silently
  overflow the real context window.
- **Image-preserving compaction** — `compact_if_needed` keeps image-bearing
  turns verbatim and summarizes only text-only older turns; collapsing an
  image turn to text would destroy the image permanently. If every older
  turn carries an image, compaction is skipped (logged) rather than dropping
  them.
- `_message_image_count` / `_message_has_image` helpers; `test_vision.py`
  Tier-1 coverage.
- Docs: VLM presets + GPU sizing (Qwen2-VL-7B, Pixtral-12B, Llama-3.2-Vision)
  in `.env.example` / RUNPOD_DEPLOY.md; user-facing image note in USER_GUIDE.

### Notes
- Facts, RAG, injection, and streaming already degraded safely on multimodal
  content; this release closes the two real gaps (budget under-count and
  compaction discarding images).

---

## [2.3] — Resilience & Stability

**Goal:** survive failure gracefully and protect irreplaceable data, so the
pod can run unattended. The "quality and failure-tested confidence over
speed" release — every item's failure path is exercised on purpose (Tier-1
covers the unit failure modes; live restore/chaos/soak rehearsals are the
operator's on-pod gates).

Image: `angreg/zions-light-ai:v2.3` (phases `:v2.3-phase1..4`).

### Added — data durability (Theme 1)
- **Verified backups** (`compactor/backup.py` + `[program:backup]` daemon) —
  timestamped tar.gz of `webui.db` (via the SQLite online-backup API, so a
  live db isn't captured mid-write) + the `compactor/` memory store. Each
  archive is **verified before it's trusted** (`PRAGMA integrity_check` +
  JSON parse); an unverifiable archive is discarded and the cycle reports
  failure. Retention pruning, a min-free-disk guard, a gated destructive
  restore, and admin endpoints (`GET/POST /admin/backups`,
  `/admin/backups/verify`). Local-volume only for now — off-volume DR is
  flagged future work.
- **OPERATIONS.md** runbook — health interpretation, log-line reference,
  failure recovery, the restore procedure, FATAL-service handling, rollback.

### Added — graceful degradation (Theme 2)
- **Disk-pressure write-gating** (`compactor/degrade.py`) — below
  `COMPACTOR_MIN_FREE_MB_WRITES` (200), new-memory growth pauses while chat
  + explicit user writes keep working. Fails open; surfaced in
  `/health/full`.
- **vLLM-restart resilience** — an unreachable vLLM yields a clean 503
  (`model_unavailable`) on the non-stream + `/v1/models` paths and a visible
  "model is starting/restarting" message on the stream path, instead of an
  opaque 500.
- **Chaos suite** (`tests/chaos/`) — guarded, self-restoring runner that
  breaks each dependency (kill vLLM, corrupt facts, unwritable ChromaDB,
  fill disk) and asserts degraded-but-functional.

### Added — process & resource stability (Theme 3)
- **Bounded background work** (`compactor/bgwork.py`) — the async tail pool
  caps concurrency and sheds beyond a hard ceiling instead of spawning
  unboundedly under load. Stats in `/health/full`.
- **supervisord restart-policy review** — documented the
  boot-loop→FATAL-visible property; FATAL spot/recover runbook.
- **Soak monitor** (`tests/soak/`) — RSS/FD leak watch over time.

### Added — operational confidence (Theme 4)
- **Structured logging** (`compactor/logsetup.py`) — `COMPACTOR_LOG_FORMAT`
  switches the compactor + sidecars between `text` (default) and `json`.
- **Optional failure-alert webhook** (`compactor/alert.py`) —
  `COMPACTOR_ALERT_WEBHOOK`; the boot self-test + backup daemon POST a
  Slack/Discord/generic alert on failure. Off by default, best-effort.

### Notes
- Atomic-write audit confirmed every durable writer already routes through
  `memory.atomic_write_json` (no torn-write gap).
- Tier-1 grew to 18 CPU suites; new pod-local tooling under `tests/chaos/`
  and `tests/soak/` (guarded, never auto-run).

---

## [2.2] — Testing & Observability

**Goal:** make "is this deploy actually working?" answerable automatically,
and codify a testing standard every future feature must follow. No separate
image — all V2.2 code shipped inside the V2.1 image line; this release is
the standard + its tooling reaching completeness.

Image: folded into `angreg/zions-light-ai:v2.1` (and the `:v2.1-phase6`/
`6.1` tags specifically).

### Added
- **Tier-2 boot self-test** (`compactor/selftest.py`) — post-boot validation
  battery run as a non-blocking one-shot supervisord program
  (`COMPACTOR_SELFTEST_ON_BOOT=true`, logs to
  `/var/log/supervisor/selftest.log`). Checks: `/data` writable, vLLM lists
  the model, compactor `/health`, a real 1-token chat round-trip, a facts
  write/read/delete against a `__selftest__` sentinel, admin localhost
  gating. Also on-demand via `GET /admin/selftest`.
- **Two-phase vLLM readiness probe** — `--wait-for-ready` waits for an
  actual completion (`/v1/chat/completions` 200), not just an open
  `/v1/models` port, so the boot self-test can't false-fail during the
  1-5 minute cold model load.
- **`GET /health/full`** — deep probe (vLLM reachability + storage
  writability + memory-store stats). Now the Docker `HEALTHCHECK` target,
  replacing `curl :3000` which stayed green even when vLLM was FATAL.
- **`TESTING.md`** — the three-tier testing standard (Tier-1 unit / Tier-2
  boot self-test / Tier-3 integration), the per-PR requirements, and the
  exact run commands for each tier.

### Changed
- Docker `HEALTHCHECK` target switched from `http://localhost:3000/`
  (OpenWebUI login page) to `http://localhost:8080/health/full`.
- Removed dead `/app/data` mkdir cruft from the Dockerfile (pre-single-
  volume layout leftover).

---

## [2.1] — User control, portability, observability, quality

**Goal:** give the *user* agency over memory and make the system operable.
V2.0 gave the model memory; V2.1 lets the user inspect, edit, export,
deduplicate, and shape it — plus the observability surface to run it.

Images: `angreg/zions-light-ai:v2.1-phase6.1` (observability),
`:v2.1-phase7` (quality), `:v2.1-phase8` / `:v2.1-complete` (commands +
personas). Rolling tag: `:v2.1`.

### Added — Phase 5: chat commands
- In-chat slash commands intercepted by the compactor (zero LLM cost,
  instant, model never sees them): `/help`, `/list-facts`, `/list-archive`,
  `/remember <text>`, `/forget [substring]`, `/why`. Streaming and
  non-streaming response paths both synthesize an OpenAI-shaped completion.
  Conservative detection — non-command slash messages pass through to vLLM.

### Added — Phase 6: observability + portability
- `GET /health/full`, `GET /admin/selftest`, boot self-test (documented
  under [2.2] — they pair).
- **Conversation portability** — `GET /admin/conversations/<id>/export`,
  `POST /admin/conversations/import`, `POST /admin/conversations/<id>/fork`.
  Single JSON bundle per conv (facts + summary state + episodic exchanges);
  embeddings re-derived on import so bundles survive embedding-model swaps.

### Added — Phase 7: quality maintenance
- **Hybrid semantic deduplication** (`compactor/dedup.py`) — embedding
  clustering filters candidates, an LLM verification call (KEEP-on-doubt,
  temp 0.0) confirms merges. Runs inline after every fact extraction
  (0 LLM calls when no candidate clusters) and on-demand via
  `POST /admin/conversations/<id>/dedup`.
- **Stale-fact archival** — facts unused for N days (default 90) move to a
  cold-storage sidecar; recoverable via restore. `GET`/`POST
  …/archive` + `POST …/restore`.

### Added — Phase 8: personas as first-class memory
- Persona (long durable system prompt) recognized as its own memory layer:
  auto-detected from a long first system message, stored separately, exempt
  from summarizer rollup and LRU fact eviction, injected as a labeled block
  (with a double-injection guard). `GET /admin/personas` library, full
  GET/POST/DELETE per conv, and `POST …/inherit-persona` to clone across
  conversations.

### Changed
- `/admin/forget` (and the `/forget` chat command) now clear the persona
  layer too — a full memory wipe is truly full.
- `/admin/conversations/<id>` summary now reports persona presence.

---

## [2.0] — Three-layer persistent memory

**Goal:** give long creative-writing conversations memory that survives the
context window and pod restarts. A FastAPI "compactor" middleware sits
between OpenWebUI and vLLM and maintains per-conversation memory on the
network volume.

Image: `angreg/zions-light-ai:v2.0` (final: `:v2.0-phase4.3`,
`sha256:d142bf0a`).

### Added
- **Conversation identity** (`compactor/memory.py`) — resolved from an
  `X-Conversation-Id` header (set by a bundled OpenWebUI Pipeline filter),
  falling back to `body.metadata.chat_id`, then a SHA-256 fingerprint.
  Atomic JSON writes (temp + fsync + rename) and a per-conv `asyncio.Lock`
  manager serialize concurrent writers.
- **Layer 1 — facts** (`compactor/facts.py`, Phase 2) — a side LLM call
  after each turn distills durable facts; LRU-pruned to a token budget;
  injected on subsequent turns. Lazy backfill (`compactor/backfill.py`)
  extracts facts from pre-existing V1 conversations on first sight.
- **Layer 2 — RAG** (`compactor/retrieval.py`, Phase 3) — every exchange
  embedded (bge-small ONNX) into ChromaDB and retrieved by semantic
  similarity for later turns. Runs in a dedicated torch-free
  `compactor-venv` isolated from vLLM's torch stack; bge-small prebaked
  into the image.
- **Layer 3 — hierarchical summaries** (`compactor/summarizer.py`,
  Phase 4) — rolling L1→L2→L3 cascade so even very long conversations get
  a coherent context block.
- **Admin/observability endpoints** — `GET /admin/conversations`,
  `GET /admin/conversations/<id>`, `GET`/`DELETE
  /admin/conversations/<id>/facts`, all localhost-gated.
- **Tier-3 integration suite** (`tests/integration/`) — black-box
  pytest+httpx scenarios against a live pod.

### Fixed
- **Mistral chat-template rejection** (Phase 4.1) — the three memory layers
  were injected as three separate system messages, which Mistral-family
  templates (Magnum v4 12B/22B) reject once ≥2 layers populate. Combined
  into a single system message.
- **Fact extraction NONE-bias** (Phase 4.3) — Magnum-12B returned the
  literal `NONE` for ~65% of fact-rich prompts at temp 0.2. Rewrote the
  extraction prompt to bias toward extraction and dropped temperature to
  0.0; extraction is now reliable.
- Test reliability: replaced fixed async-tail sleeps with polling helpers
  (`wait_for_facts`, `wait_for_indexed_exchanges`).

---

## [1.9.6] — Final V1 release

**Goal:** close V1 cleanly. CVE remediation, parametric build foundation,
operational quality improvements. After this, the 1.9.x line is frozen
except for security patches — new feature work moves to V2.

### Security
- **Bumped vllm `0.11.0` → `0.14.1`** — resolves CVE-2026-22778 (Critical 9.8)
  and 7 other High-severity CVEs in vllm itself. Includes auto-bumps of
  torch and xgrammar transitive deps that resolve their respective Highs.
- **Added `apt-get upgrade -y` to the Dockerfile** — picks up Ubuntu CVE
  patches for installed packages (catches gnupg2 High and any future
  ones released after the base image was published).
- **Bumped `pip` / `setuptools` / `wheel`** in both venvs as part of the
  install layer — resolves 4 Highs (setuptools×2, wheel×2).
- **Bumped OpenWebUI to its latest release** — resolves Highs in pillow,
  ecdsa, pyjwt, python-multipart, nltk, pyarrow, langchain-classic,
  jaraco.context (all OpenWebUI's transitive deps).
- Bumped transformers ceiling within the `<5` range to pick up its
  flagged High.

### Foundation
- **Parametric CUDA build args** (`CUDA_BASE_IMAGE`, `TORCH_CUDA`,
  `VLLM_VERSION`). Same Dockerfile now builds cu128 (default) and cu130
  variants without source changes. Foundation for the eventual cu130
  variant once RunPod's GPU fleet broadly rolls out driver 580+.
- **Preflight checks in entrypoint.sh**: verify `/data` is writable,
  GPU is visible via nvidia-smi, driver version meets torch's minimum.
  Fails loud and fast with actionable messages instead of letting vLLM
  crash 2-3 minutes in with a cryptic stack trace.
- **Persistent torch.compile cache** at `/data/vllm-compile-cache`.
  Cold starts after the first one skip the 60-120s CUDA graph capture
  step. Symlinked from `/root/.cache/vllm` in entrypoint.sh.

### Documentation
- New top-level **README.md** with architecture diagram, quick start, and
  project structure.
- New top-level **CHANGELOG.md** (this file) documenting the entire 1.9.x
  line.
- New top-level **ROADMAP.md** with V1 → V2.0 → V2.1 → V3 → beyond plan.
- **V2_PLAN.md** updated to split V2 into V2.0 (memory architecture) and
  V2.1 (user control / portability / observability). Conv_id strategy
  upgraded from "hash with header fallback" to "header from day one,
  hash as fallback" — eliminates the collision risk class entirely.

---

## [1.9.5] — Triton JIT toolchain

### Fixed
- Added **`build-essential` + `python3-dev`** to the apt install layer.
  vLLM (via torch.compile) uses Triton to JIT-compile per-kernel C
  source at runtime during CUDA graph capture; without a compiler and
  Python headers, vLLM crashed at startup with either "Failed to find
  C compiler" or "Python.h: No such file or directory". ~200 MB image
  growth — necessary tax for vLLM on a slim base.

---

## [1.9.4] — Transformers compat for vLLM 0.11

### Fixed
- **Pinned `transformers>=4.50,<5`** in `compactor/requirements.txt`.
  vLLM 0.11 calls `tokenizer.all_special_tokens_extended`, which was
  removed in transformers 5.x. Unpinned `transformers` in 1.9.1/1.9.2
  let pip resolve to 5.9.0, causing `AttributeError` at vLLM startup.
- The compat range now keeps Gemma3Config available (the v1.9 → v1.9.1
  fix) AND keeps the tokenizer API stable for vLLM 0.11.

---

## [1.9.3] — supervisord rpcinterface syntax

### Fixed
- Corrected `supervisor.rpcinterface_factory` value to use the colon
  module:attr form (`supervisor.rpcinterface:make_main_rpcinterface`)
  instead of dotted Python path. With the wrong separator, supervisord
  crashed at config-parse time before spawning any subprocess.
- Since supervisord runs as PID 1 via entrypoint exec, that parse error
  killed the container and RunPod respawned it into a crash loop with no
  in-pod recovery short of a new image.

---

## [1.9.2] — CUDA 12 vLLM pin + compactor env handling

### Security / runtime
- **Pinned vllm `==0.11.0`** with `--extra-index-url cu128` to keep
  PyTorch on cu128 wheels. Modern vLLM (0.21+) ships cu130 wheels which
  require NVIDIA driver 580+; most RunPod hosts (including the A40 fleet)
  are still on driver 570 (CUDA 12.8 max). Without this pin, vLLM crashed
  at startup with "NVIDIA driver too old (found version 12080)".

### Fixed
- Compactor `_env_int()` helper handles empty-string env vars. `.env`
  files set keys to `""` for opt-in blanks; `os.environ.get(name, default)`
  returns `""` not the default, and `int("")` crashed at compactor module
  import time. Added regression test.

---

## [1.9.1] — Dep pin conflict + supervisorctl socket

### Fixed
- **Unpinned `fastapi` / `uvicorn` / `httpx` / `transformers`** in
  `compactor/requirements.txt`. The 1.9 pins (transformers==4.47.1
  specifically) caused pip to *downgrade* what vLLM had installed,
  leaving transformers below the 4.50 minimum that modern vLLM
  unconditionally imports (`Gemma3Config`). Result: vLLM crashed at
  import.
- Added the `[unix_http_server]` / `[supervisorctl]` /
  `[rpcinterface:supervisor]` sections to `supervisord.conf` so
  `supervisorctl` actually has a control socket to connect to. The 1.9
  image's supervisord ran fine but had no IPC surface — restart-from-pod
  required `kill`-ing PIDs manually.
- Dropped deprecated `TRANSFORMERS_CACHE` env var (transformers v5
  removes it; `HF_HOME` alone is the modern equivalent and covers both
  transformers and huggingface_hub).

---

## [1.9] — Migration from llama.cpp to vLLM

**The big rewrite.** Replaced the entire inference engine and added
the context-compactor middleware that prevents long-conversation context
loss.

### Added
- **vLLM** as the inference engine, replacing llama.cpp. Native
  HuggingFace safetensors support — any vllm-compatible HF causal-LM
  loads by repo ID with no GGUF gymnastics.
- **context-compactor** FastAPI middleware (`compactor/main.py`). Counts
  tokens with the target model's own tokenizer; when a request exceeds
  `COMPACTOR_TARGET_TOKENS` (default 75% of `MAX_MODEL_LEN`), older
  turns get summarized into a single system block via an extra LLM call
  and the original messages are replaced. Streaming responses are
  proxied verbatim. Backend-agnostic (works against any OpenAI-compatible
  endpoint).
- **Single `/data` Network Volume** layout — both model cache
  (`HF_HOME`) and OpenWebUI state (`DATA_DIR`) live on one volume.
  Simpler RunPod deploys, fewer moving parts.
- **`VLLM_EXTRA_ARGS` env var** for passing arbitrary flags to vLLM
  (`--quantization fp8`, `--tensor-parallel-size N`, etc.) without
  rebuilding.
- **Default model changed to** `anthracite-org/magnum-v4-22b` — creative
  writing fine-tune of Mistral-Small, lightly aligned. Several
  alternative presets commented in `.env.example`.

### Changed
- Dockerfile rewritten end-to-end: dropped the llama.cpp builder stage,
  switched to single-stage with atomic install+strip+cleanup per venv to
  keep the image at ~16 GB.
- Default `MAX_MODEL_LEN=32768` (vs llama.cpp's 32K cap which was a hard
  wall; with the compactor, this is the engine ceiling but conversations
  can effectively run longer via summarization).

### Documentation
- `RUNPOD_DEPLOY.md` rewritten for the vLLM + Network Volume pattern,
  including the pre-warm-on-CPU-pod cost optimization.
- `compactor/V2_PLAN.md` design spec for the next memory iteration.

---

## [1.8] — Previous llama.cpp release

Last release in the llama.cpp era. See git history for details — superseded
by 1.9's rewrite.

---

## [1.7] — Modular AI model config

Modular AI model configuration. Removed ability to run two AI models at
once in the same container due to operational complexity.

---

## Earlier versions

See git history for releases prior to 1.7.
