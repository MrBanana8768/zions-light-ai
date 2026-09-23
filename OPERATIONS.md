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

#### `checks.hierarchy` — the summary catch-up reason (v3.1.9)

```bash
curl -s localhost:8080/health/full | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('reasons:', d['status_reasons'])
print('catching_up:', d['checks']['hierarchy'].get('catching_up'))
print('catching_up_all:', d['checks']['hierarchy'].get('catching_up_all'))"
```

Each conversation whose recent hierarchy lag is over the limit gets a
`verdict`: **`converging`** (the watermark advanced within the last 15
minutes — no action, and this never adds a `status_reasons` line or
degrades `status`), **`stuck`** (20 budgeted tail passes in a row with work
due and no advance — the actionable case), or **`unknown`** (no catch-up
evidence for THIS process yet — almost always a recent restart; treat it as
"wait for her next message," not as a confirmed stall — `unknown` DOES add
a `status_reasons` line and degrade `status`, deliberately, so a lost
restart-evidence window is never read as self-healing — but its wording
says plainly that nothing has been observed yet and is not itself evidence
of a stall, distinct from `stuck`'s wording, which names the actual
failure). `catching_up`
is the single worst conversation; `catching_up_all` lists every conversation
currently over the limit, so a converging one with a large lag cannot hide a
genuinely stuck one with a smaller lag. **Do not run `/compact` on a `stuck`
verdict without checking the log first**: `grep -a "failed and will be
retried next turn" /data/logs/compactor.log | tail -5` — if a tier is named
there, `/compact` runs the identical drain and will NOT clear the backlog;
fix the named cause first (commonly an unreadable
`summaries/<conv>.archive.json`).

#### `checks.reuse` — is the reuse stand-in actually firing? (v3.1.9.2)

```bash
curl -s localhost:8080/health/full | python3 -c "
import json,sys; d=json.load(sys.stdin); r=d['checks'].get('reuse') or {}
print('attempted:', r.get('attempted'), '| succeeded:', r.get('succeeded'))
print('declined_no_state:', r.get('declined_no_state'),
      '| declined_no_coverage:', r.get('declined_no_coverage'),
      '| declined_budget:', r.get('declined_budget'),
      '| declined_window:', r.get('declined_window'),
      '| errored:', r.get('errored'))
print('last_reason:', r.get('last_reason'), '| last_attempt_age_s:', r.get('last_attempt_age_s'))
print('declined_recently:', r.get('declined_recently'))
print('last_declined_ceiling:', r.get('last_declined_ceiling'),
      '| last_declined_others:', r.get('last_declined_others'),
      '| last_declined_reserve:', r.get('last_declined_reserve'))"
```

Added after hostile pass #9 (P9-1/P9-2) found the reuse feature v3.1.9.1
introduced silently declining on every request again, at the numbers
v3.1.9.2 nearly shipped, with no signal anywhere but an INFO log line. This
is visibility-only — it never appears in `status_reasons` and never
degrades `status`, the same as `checks.tokenizer` — because a decline
still falls back to summarizing from scratch and answers the turn; it is
slower and remembers less precisely, not broken.

**Hostile pass #10 (P10-3) split what used to be one `declined_budget`
counter into five, because `attempted=0, declined_budget=0` used to mean
FOUR different things** (a fresh process; no stored hierarchy yet; a
hierarchy that covers none of this request's array; and — the one that
mattered most — an exception raised partway through the attempt, which
used to leave `attempted` incremented with nothing to show it had failed,
reading exactly like a healthy reuse). Read `last_reason` first: it names
what the MOST RECENT attempt resolved to — `"success"`, `"no_state"`
(nothing stored yet, normal for a new conversation), `"no_coverage"` (a
hierarchy exists but does not cover this array — a different branch, a
delete-and-regenerate, or a store rebuild), `"budget"` (exists, covers
this array, but the rendered stand-in does not fit the stand-in's OWN
budget whole), `"window"` (hostile pass #12, P12-2 — exists, covers this
array, the stand-in fits ITS OWN budget, but alongside the system prompt
and the recent turns it would leave no room in the request's real window
— a DIFFERENT decline from `"budget"`, see below) or `"error"` (an
exception — check `/data/logs/compactor.log` for "could not reuse stored
summaries" around `last_attempt_age_s` seconds ago). **A nonzero
`errored` count is the one that needs a log, not a shrug**: this counter
exists specifically because "the request still succeeded" (the `except`
clause's whole job) used to also mean "nothing tells you this happened."

`attempted`/`succeeded`/`declined_no_state`/`declined_no_coverage`/
`declined_budget`/`declined_window`/`errored` are all cumulative since the
process started — **a restart resets every one of them to zero, not a
rolling window** (unlike `declined_recently`, below) — so a freshly
restarted pod reading `attempted: 0` says nothing about whether reuse was
healthy or broken before the restart; use `last_attempt_age_s` (`None`
only when no candidate request has reached the reuse check yet THIS
process) rather than assuming a low `attempted` means a quiet feature.
`declined_budget == 0 and declined_window == 0` with `attempted > 0` means
every stand-in attempt that reached either check fit; that is the healthy
state to expect in normal operation. **`declined_recently`** is `true` for
`COMPACTOR_REUSE_DECLINE_DEGRADE_WINDOW_S` (default 300s) after the most
recent CAPACITY decline — `"budget"` OR `"window"` (P12-2 widened this
from "budget" specifically: `no_state`/`no_coverage`/`error` still do not
set it, but a window-squeeze decline is exactly as real a capacity squeeze
as a budget one, and hiding it here just meant an operator staring at
`declined_recently: false` minutes after a run of window declines) —
check this first if you suspect the feature just stopped working, rather
than the cumulative counters, which stay nonzero forever after even one
decline early in a long-lived process. **Read `last_declined_ceiling`/
`last_declined_others` together with `last_reason` — they mean a
DIFFERENT pair of numbers depending on which reason produced them:**

- **`last_reason == "budget"`**: the two numbers from that decline's own
  log line (`the stored summaries cover N of the turns ... but they do
  not fit whole in the <ceiling> token(s) ... leaves beside the system
  prompt, images and recent turns (<others>) and one fresh summary`) — if
  `declined_recently` is true for this reason, the summary hierarchy has
  grown past what `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS`/
  `COMPACTOR_STANDIN_BUDGET_FRACTION` can hold right now (see
  RUNPOD_DEPLOY.md's [Memory budgets](RUNPOD_DEPLOY.md#memory-budgets--raised-defaults-in-v319)
  for the exact arithmetic and what to raise). **Do not use 11,300 as the
  threshold to watch for** (hostile pass #10, P10-2 corrected this doc:
  that figure is in a different unit from what `last_declined_ceiling` is
  checked against, and her own hierarchy's measured steady-state peak with
  a real L3 is already 11,728 — above it — while still reusing at the
  shipped default). Compare `last_declined_ceiling` against
  `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` itself (15,000 shipped) instead: a
  decline with `last_declined_ceiling` at or near that configured value
  means the hierarchy has genuinely outgrown the current setting and it is
  time to raise it (together with `COMPACTOR_INJECTION_BUDGET_FRACTION`,
  which the ceiling can also never exceed).
- **`last_reason == "window"`** (hostile pass #12, P12-2): a DIFFERENT
  check, with its own log line (`the stored summaries cover N of the
  turns ... and the <rendered>-token stand-in fits the <budget>-token
  ceiling, but alongside this conversation's system prompt and recent
  turns it would leave the ~<ceiling>-token window no room for the turns
  it exists to keep (reserve <last_declined_reserve> > <last_declined_
  ceiling> available)`). `last_declined_ceiling` is `effective_limit_est
  - (system prompt + the recent turns)` — the room actually available for
  the stand-in — `last_declined_others` is what the system prompt and the
  recent turns themselves cost, and `last_declined_reserve` (v3.1.9.3,
  P13-3 — before this release the number was findable only in the log
  line) is what was actually COMPARED against the ceiling: the rendered
  stand-in, plus the fresh-summary reserve when a fresh span is pending
  (P12-6/P13-2, below), plus a fixed 128-token drift allowance. **`last_
  declined_ceiling` sitting at or near `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS`
  (or anywhere else) means nothing for THIS reason — it is not that
  number.** Neither `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` nor
  `COMPACTOR_INJECTION_BUDGET_FRACTION` can move it: raising either only
  changes how big a stand-in is ALLOWED to render before this check runs,
  never what this check compares it against (P12-2's own reproduction:
  raising `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` from 15,000 to 20,000 left
  the SAME requests declining, `last_declined_ceiling` merely following the
  new inject-budget cap).

  **What actually moves it** (corrected, hostile pass #13, P13-3 — the
  list below used to also name `COMPACTOR_IMAGE_TOKENS`, which prices
  nothing on this pod): a smaller learned budget margin (see
  `checks.budget_margin`, below — a margin in force shrinks
  `last_declined_ceiling` directly and is the single biggest lever while
  it lasts), a smaller `last_declined_reserve` (mostly what the
  conversation did between L1 rollups: a smaller uncovered tail, fewer
  un-folded fresh-summary batches), fewer retained images IF AND ONLY IF
  the recent window itself carries more images than
  `COMPACTOR_MAX_RETAINED_IMAGES` allows (`COMPACTOR_IMAGE_TOKENS` moves
  nothing on a pod with opencv installed — the image ships it from
  v3.1.9.3, so this pod prices a rendered image by its real
  per-resolution cost, not the flat estimate `COMPACTOR_IMAGE_TOKENS`
  names), a smaller recent reply, or a smaller stored hierarchy (trigger
  an L3 rollup early, `/admin/...` — see the hierarchy section above).

  Settings that move it, each with a cost (corrected, hostile pass #14,
  P14-3 — this used to say the reserve was "nothing to configure"):
  `COMPACTOR_SUMMARY_MAX_TOKENS` and `COMPACTOR_MAX_SUMMARY_CALLS` scale
  the fresh-summary part of the reserve (`min(batches, calls) x
  max tokens`; lowering either makes fresh summaries shorter or defers
  more of the span); `COMPACTOR_GENERATION_RESERVE` and the client's
  `max_tokens` set the window itself (lowering them leaves less room for
  her reply); `COMPACTOR_KEEP_RECENT_TURNS` sets the recent floor (lowering
  it forwards fewer turns verbatim). None of these is a fix for an
  occasional decline; they are for a conversation that declines on most
  requests.

  **A window decline is not a bug and is not data loss, but it is not
  free either** (corrected, hostile pass #13, P13-3 — this used to say
  "slower and forwards more raw text, not silently worse", which held for
  the hierarchy but not for what sits above it). The declined path
  (v3.1.9.3, hostile pass #12 P12-5) protects the SAME recent turns this
  check exists to protect, by spending injected memory (facts, retrieval)
  before them, and the request is always answered. But a window decline
  resets the reuse to nothing, so `summarize()` is handed the WHOLE older
  span, and at her size that exceeds `COMPACTOR_MAX_SUMMARY_CALLS` and is
  refused (corrected, hostile pass #14, P14-3 — this used to blame the
  uncovered tail alone): those turns go out VERBATIM rather than
  summarized, and P12-5's order sheds verbatim turns above the protected
  floor before it ever touches memory — so the uncovered tail can reach
  the model NEITHER summarized NOR verbatim. Real-data replay (hostile
  passes #13 and #14): after this release's fixes that still happens at
  25-69 of 474 positions with a 20-turn uncovered tail (peakA / peakB),
  down from 45-147 before them. If `declined_recently` is true for `"window"`
  and the conversation's replies seem to have forgotten something recent,
  this is the mechanism to suspect before assuming the hierarchy itself
  lost it.

#### `checks.truncated_summaries` — how many stored summaries were cut and trimmed rather than finished? (v3.1.9.4, R3/R4, P15-6)

```bash
curl -s localhost:8080/health/full | python3 -c "
import json,sys; d=json.load(sys.stdin); t=d['checks'].get('truncated_summaries') or {}
print('hierarchy (L1/L2/L3):', t.get('hierarchy'))
print('compaction (request-path):', t.get('compaction'), '| reason:', t.get('compaction_reason'))"
```

Two counters, both process-local and reset to zero on a restart. `hierarchy`
is `summarizer.truncated_summary_count()` — how many L1/L2/L3 rollup
summaries were cut at their tier's `max_tokens` (`finish_reason: "length"`)
and still had nothing better than a fallback-trimmed result after one retry
at the same cap with a tighter word target (round 2's M1, P15-6). `compaction`
is the identical counter for `main._summarize_once` — the compaction summary
`compact_if_needed` builds on the REQUEST path, not the background tail (R3,
the same fix applied to this call site's own `finish_reason=length` gap);
`compaction_reason` explains a `null` compaction value (main.py not loaded
in this process, or the counter function missing in an older build) the same
way `checks.reuse`'s `"available": false` does.

Before this release, a cut summary was stored (or forwarded, for the
compaction path) byte-for-byte as if it had finished — indistinguishable
on disk from a complete one, with no log line and no counter anywhere
(P15-6's own finding). Now every cut-and-trimmed unit is retried once and,
if still cut, logged at WARNING (naming the conversation and, for the
hierarchy path, the tier) AND counted here. **This is visibility-only — it
never appears in `status_reasons` and never degrades `status`, the same
doctrine as `checks.reuse`/`checks.budget_margin` above**: a trimmed summary
is a degraded-but-served unit (it covers slightly less than the model
tried to say, at a real sentence/line/word boundary — never mid-sentence,
never silently stalled), not a fault to alarm an operator awake for. Zero
on a healthy deployment; a count that climbs steadily is worth investigating
(a conversation whose turns consistently overflow `COMPACTOR_L1_MAX_TOKENS`/
`COMPACTOR_L2_MAX_TOKENS`/`COMPACTOR_L3_MAX_TOKENS`/`COMPACTOR_SUMMARY_MAX_TOKENS`,
or a model that is simply verbose against these caps) — raising the
relevant `*_MAX_TOKENS` env var is the fix, not a code change.

#### `checks.budget_margin` — is a learned budget correction narrowing the window right now? (v3.1.9.3, P13-1/P13-3)

```bash
curl -s localhost:8080/health/full | python3 -c "
import json,sys; d=json.load(sys.stdin); m=d['checks'].get('budget_margin') or {}
print('margin:', m.get('margin'), '/ ceiling:', m.get('ceiling'))
print('release_after:', m.get('release_after'), '| ok_streak:', m.get('ok_streak'))"
```

`main._BUDGET_MARGIN` is the degraded-mode backstop for when the local
token count has been WRONG — a vLLM context-length 400 the guard did not
already predict (a `/tokenize` outage, or a mispriced image) latches it to
`overshoot + 512`, up to `MAX_MODEL_LEN // 4` (`ceiling` above), in ONE
step, and it applies PROCESS-WIDE (one uvicorn worker, one margin, every
conversation) until `release_after` (`COMPACTOR_BUDGET_MARGIN_RELEASE_
AFTER`, default 50) consecutive ACCEPTED requests halve it (or clear it,
once it is 512 or below) — `ok_streak` is how far into that count this
process already is. Before this release the only way to learn a margin
was in force was the log: the WARNING when it latches ("Tightening the
hard limit by N for EVERY conversation"), the INFO when it is released,
or the "margin N" suffix on a hard-budget shed line if one happened to
fire while it was up (the boot-time "context calibration" line always
shows a fresh process's 0); the adversarial suite's
own F-02 finding (`tests/adversarial/test_adv_faults.py`) names the gap
explicitly ("/health/full has no margin field"). This is visibility-only —
it never appears in `status_reasons` and never degrades `status`: the
process is already correcting itself, and a 400 without it would be worse.

**While `margin` is nonzero, both `_enforce_hard_budget` (the hard-budget
guard) and the P11-6/P12-1/P13-1 reuse window check subtract it from their
own limit before deciding anything** — so a nonzero margin is the single
biggest thing that can move `checks.reuse`'s `last_declined_ceiling` for
`"window"` (see above) on a conversation that was reusing fine a moment
ago. If reuse looks like it "just stopped working" and `checks.budget_
margin.margin` is nonzero, that is very likely why, and the fix is time
(`ok_streak` accepted requests) rather than a config change — the margin
already IS the config change reacting to a real overshoot it measured.

One request can fall between the two reads (hostile pass #14, P14-1): if
a margin latches while a request is already past its reuse decision, that
request's guard applies the new margin. Fixed in v3.1.9.4: the reuse
window check now reports the margin it used for its decision, and the
guard call compares that to the LIVE margin right beside itself; if the
margin grew in between, the guard spends the compaction stand-in as
ordinary memory, ahead of her previous exchange, instead of protecting it
at the recent window's own tier — the same order a request that had
declined reuse under that larger margin would already get. The guard's
own numeric limit is still always the live value, never a stale snapshot
of the earlier one — forwarding at the old, smaller limit would risk
losing the whole reply to a second rejection on the same mechanism, which
is strictly worse than losing one exchange of context, so only the
stand-in's protection is conditional, not the limit itself. Every request
after the margin changes reads it fresh in both places, as before.

#### `config.time_injection` — the current-time feature (v3.1.9)

```bash
curl -s localhost:8080/health/full | python3 -c "
import json,sys; d=json.load(sys.stdin); t=d['config'].get('time_injection') or {}
print('last_source:', t.get('last_source'), '| last_timezone:', t.get('last_timezone'))
print('fallback_error:', t.get('fallback_error'))
print('current_line:', t.get('current_line'))"
```

`last_source` reads `browser` (header identity, system-prompt line present),
`env` (hash identity with `COMPACTOR_TIMEZONE` set — correct for that route,
not a fault), or `utc` (neither is in force). **`fallback_error` is the one
to actually check** for a misconfigured `COMPACTOR_TIMEZONE`: a name that
does not resolve falls back to UTC silently as far as `status` is concerned
— it does NOT add a `status_reasons` line or degrade `status`, only this
one field and a single `TIME ZONE NOT APPLIED` ERROR in `compactor.log` at
boot. See RUNPOD_DEPLOY.md "The current date and time" for what each route
should show.

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

### Sampling parameters

OpenWebUI's **Advanced Params** map only a fixed set of names onto the
request it sends: `temperature`, `top_p`, `min_p`, `max_tokens`,
`frequency_penalty`, `presence_penalty`, `reasoning_effort`, `seed`, `stop`,
`logit_bias`, `response_format`. Anything else — including any Ollama-style
name — has to be set under **Custom Parameters** instead, where OpenWebUI
passes it through to the request body exactly as typed.

vLLM's name is `repetition_penalty`; Ollama's name for the same idea is
`repeat_penalty`. **From v3.1.9.2 on**, the compactor translates
`repeat_penalty` to `repetition_penalty` before forwarding (a bad value is
dropped with a WARNING, never forwarded) and drops `repeat_last_n`, which has
no vLLM equivalent. Before v3.1.9.2, a Custom Parameter named
`repeat_penalty` reached vLLM unrecognised and did nothing — no error, no
log line, `repetition_penalty` just stayed at its default of 1.0.

Recommended starting values for this model (Cydonia-24B):
`repetition_penalty` 1.05 (Custom Parameter), Frequency Penalty 0.3 and Max
Tokens 12000 (both Advanced Params) — **not** the 4 chars/token rule of thumb
Max Tokens 7000 used to be picked by. **Correction (P11-5, hostile pass #11):**
this used to say "2.0-2.4 chars/token" from 2026-08-28 production data, which
was measured on unusually box-drawing-heavy replies; her current branch (476
replies, 2026-09-17, Tekken) measures **3.77 chars/token** instead (full
detail and the vocabulary caveat: [RUNPOD_DEPLOY.md → Sampling
parameters](RUNPOD_DEPLOY.md#sampling-parameters)). At that rate 7000 tokens
is ~26k characters, comfortably above her normal p90 reply length, so it is
no longer accurate to say it "would cut ordinary long replies mid-sentence"
— it still cuts her rare very-long replies, which is why 12000 remains the
recommendation, not a reason to raise it further. vLLM 0.19 also applies
`repetition_penalty` to PROMPT
tokens, not only output, so a high value discourages words already sitting
in her ~20k-token conversation/memory context, not just words the model has
already said in this reply — raise Frequency Penalty (output-only) before
raising `repetition_penalty` further if loops return. Full detail:
[RUNPOD_DEPLOY.md → Sampling parameters](RUNPOD_DEPLOY.md#sampling-parameters).

**Confirming loops are being caught.** A repetition-loop reply logs a
WARNING at detection time (`grep -a 'like a repetition loop'` matches both
wordings — the split is FINISHED vs CUT, not streamed vs non-streamed) and,
once OpenWebUI replays it back as history, an INFO line each time it is
touched in what is forwarded to vLLM: `conv=<id>: touched <N> degenerate
assistant turn(s) in the forwarded window (whole=<K> cut=<N-K>)`. **These
two counts can differ**: a CUT reply whose trimmed head reads clean is
stored TRIMMED with no loop WARNING at all (memory only judges the kept
head), but the full original text is still flagged and still touched in
the forwarded window on every later request — a `touched` count with no
matching WARNING for that turn is expected, not a sign the detector missed
it. Neither line names the reply's own text. **`whole` vs `cut`** (v3.1.9.2
hostile pass #8, P8-8): `whole` is how many of those `touched` turns lost
the ENTIRE reply to the placeholder; the rest (`cut`) kept a clean
head/tail around only the flagged span — on the 2026-09-16 backup that was
10 whole out of 66 touched, so `cut` is usually the larger number, not
`whole`.

**A non-finite numeral in the request body (v3.1.9.2 hostile pass #8,
P8-6)** — `"max_tokens": 1e999` or the same in any other numeric field —
now gets a 400 at parse time instead of a 500 from inside the proxy after
compaction and memory injection had already run.

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

### Nightly "memory shrank" alert — noise on v3.1.6.1 to v3.1.8, a real signal from v3.1.9 (except one item)

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
| `<id>.summary_active_bytes N->M (hierarchy emptied, watermark at K)` | the ENTIRE active summary hierarchy went to 0 bytes while the watermark still claims K turns covered — real loss |
| `<id>.summary_active_bytes N->M (more than half below its high-water mark)` | **known false-alarm shape, see below — do not treat as real on its own** |

**On v3.1.9, treat every `NOT pruning — memory shrank` item as real, EXCEPT
`summary_active_bytes … (more than half below its high-water mark)`.** That
one specific wording is a known false alarm, not yet fixed in code (the fix
is tracked for a later patch release): an ORDINARY L1→L2 or L2→L3 summary
fold routinely drops active summary bytes by more than half in one step —
ten L1 chunks of a few thousand characters each collapse into one shorter L2
chapter, the same shape an L2→L3 refresh repeats — and the high-water-mark
check (`backup.py`, `_census_hwm_update`) compares against the BEST size it
has ever seen for that conversation, which a routine fold will almost always
undercut. Nothing was lost; the fold is the summary hierarchy working as
designed, and the mark resets itself the next cycle (so this fires at most
once per fold, not every night after).

**How to tell this false alarm apart from a real `summary_active_bytes`
loss**, before asking for help: pull that conversation's summary state
(`curl -s localhost:8080/admin/conversations/<id> | python3 -m json.tool`,
`.summary`) and compare its L1/L2/L3 shape against yesterday's backup (or
just check whether an L2 chapter count or L3 presence went UP by one since
the last cycle). **If the chapter count went up by one (or L3 newly
appeared) in the SAME backup where `summary_active_bytes` fell** — that is
this false-alarm shape: an ordinary fold, not a loss; the backup is fine,
do not restore anything. **If the chapter/L3 count did NOT go up** (bytes
fell with no corresponding fold), or you see the OTHER `summary_active_bytes
… (hierarchy emptied, watermark at K)` wording, or any of the other items in
the table above — treat it as real: do not prune by hand, leave the
archives alone, and ask for help.

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

**`/forget` (or the DELETE above) only permanently protects the facts
layer.** It leaves an empty facts store behind specifically so the lazy
backfill cannot reconstruct facts from that conversation's history again
(`backfill.needs_backfill`'s tombstone check). The L1/L2/L3 summary
hierarchy has no equivalent tombstone: if the operator or the user keeps
that SAME conversation going, OpenWebUI resends its whole prior
transcript with the next message — it is the client, not this service,
that decides what history a request carries — and the next ordinary tail
rolls the summary hierarchy back up from watermark 0 over the full
history, as if `/forget` had never touched it (v3.1.9.4 P16-6, documented
— predates that release, not a regression). If a support request is "I
forgot X and it came back," check whether the conversation kept going
afterward before assuming the wipe itself failed; `/why` distinguishes
the two (a rebuilt summary reads as freshly-summarized content, not a
surviving fact).

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

### `COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB`: a one-time bootstrap setting

The backup daemon refuses a cycle outright (holds the prune) when it cannot
find `webui.db` at all — a missing database is treated the same as a missing
memory store, never silently backed up without her chat history. On a
genuinely brand-new pod the daemon's very first cycle can race OpenWebUI's
own first write and hit this refusal once; **the daemon's own 15-minute retry
already clears that race on its own**, so you should not normally need to
touch this setting at all.

If you do set `COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB=1` (RunPod template
variable), it only has any effect while `/data/backups/` holds **zero**
archives — the very first cycle. Once one archive has published, the setting
is inert and the refusal applies regardless: a hatch that stayed effective
forever would silently turn a later missing/unmounted/misresolved `webui.db`
into a memory-only archive that prunes real history behind it, which is the
exact failure this refusal exists to catch. **Unset it once the first backup
exists** — leaving it in the template past that point does nothing useful and
invites confusion later. The backup log names it loudly, every cycle, for as
long as it is set (`backup.py --once`'s own log line, or `tail -f
/data/logs/backup.log`), specifically so it cannot sit forgotten and silent
in a template.

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
> fully reviewed.** It copies everything it will restore next to its
> destination first, moves the old database's journal and the old memory
> store into `/data/forensics` instead of deleting them (together, in the
> order that keeps a hot journal beside the database it belongs to at every
> step — a hostile-review finding on the ROLLBACK path specifically, fixed
> before this review cycle closed), checks free space first, refuses to
> start if a writer is holding the database mid-write, refuses to start a
> second restore while an earlier one's marker is still on disk (below),
> integrity-checks the database it just landed before touching the memory
> store, and prints a restart line naming only the services that placement
> actually uses. It can list what it set aside, staged copies included
> (`backup.py --list-pre-restore`). It is still not the documented path, for
> reasons in the code as well as the calendar: the final hostile review of
> v3.1.9 has not cleared it; it cannot see a writer when a journal already
> exists (exactly the incident state), so stopping the writers is still on
> you; and its free-space check reads MooseFS's cluster-wide number. The
> manual procedure does all of those explicitly. Until that review clears
> it, use the manual procedure on v3.1.9 too.
>
> **An interrupted restore leaves a marker — `/data/forensics/restore-
> *.inprogress`.** Any `backup.py --restore` run (including the CLI's) writes
> this the moment it finishes staging, before touching anything live, and
> removes it only once the restore fully landed or a failure was fully
> rolled back. If the process is killed in between (a redeploy, an OOM
> kill), or a rollback could not fully complete, the marker survives. While
> it exists: the pod refuses to boot (entrypoint.sh checks for it before
> OpenWebUI starts, on every placement, and prints its own banner naming the
> file), `backup.py --restore` refuses to start a second restore, and
> `/health/full`'s `status_reasons` names it. It means one of three things —
> read the marker itself (`cat` it; it is JSON naming every path the restore
> planned to touch) to tell them apart:
>   1. genuinely mid-restore — webui.db and/or the compactor store may be
>      missing or at a mixed generation right now;
>   2. the restore finished and rolled itself back, but the process died
>      before it could remove its own marker — the live paths are back to
>      their pre-restore state;
>   3. the restore finished and succeeded, but the process died before it
>      could remove its own marker — the live paths are the archive's, and
>      nothing is wrong with them.
> This is deliberately not auto-recovered at boot: telling (1) apart from
> (2)/(3) needs opening webui.db, and doing that automatically at every boot
> with a marker present is not worth the risk on a volume that has already
> taken minutes to roll back a hot journal on a database this size. With
> someone watching, run `backup.py --status`, check whether the path(s)
> named in the marker's `plan` exist and pass `quick_check`, and once you are
> sure the live state is either fully restored or fully back to its
> pre-restore state, `rm` the marker file named in the banner (or by
> `find_interrupted_restore`) and redeploy or restart. A restore clears
> ITS OWN marker automatically on a full success or a fully-completed
> rollback — most of the time you will never see one. `backup.py --restore`
> also REFUSES TO START while an EARLIER run's marker is still on disk (it
> will not run a new restore over unresolved state), so clear a stale one
> by hand, as above, before trying again — a later restore never sweeps up
> a marker it did not itself write.

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

## Exit codes — the shared convention across the operator scripts

`scripts/backfill-records.py`, `scripts/import-history.py` and
`scripts/setup-sshd.py` all use the same five-value exit-code convention,
stated ONCE here — each script's own section below links back to this
table rather than repeating it, and each script's own module docstring
gives that script's exact per-code triggers.

| Code | Meaning |
|---|---|
| 0 | The desired end state is in place — a dry run that found nothing pending, or an `--apply`/install that fully succeeded (including a no-op one, because there was nothing to do). |
| 1 | A refusal or an error the operator needs to look at, OR `--apply` (or the equivalent installing step) made NO progress at all, OR the result needs human judgement (an ambiguous/needs-review record, every `--conv` value given being rejected, etc.) — never the same code as 0, however safe the refusal itself is. |
| 2 | argparse's own usage error (an unknown flag, a missing required value) — the Python standard library's own convention, unrelated to the four codes around it. |
| 3 | A DRY RUN found pending work — informational, not a failure: it means "re-run with `--apply` once you're ready". |
| 4 | `--apply` (or the equivalent) made SOME progress but work still remains — re-run (with `--force` if that is what the remaining refusal calls for) to continue. |

This normalizes what was, before v3.1.9.6, three scripts sharing the same
flags with opposite exit meanings (`setup-sshd.py` originally had 0 and 3
swapped from the other two — see CHANGELOG.md [3.1.9.6] "Fixed"), and adds
code 4 in a later pass of the same release: `backfill-records.py`'s own
`--apply` used to report the same "0 success" whether it closed every
targeted record or none at all, indistinguishable from the exact upgrade
hazard the script exists to prevent (H1). Not every script has a code-4
case — see each script's own section below for exactly which of the five
codes it actually returns and why.

## Closing stale backfill records before upgrading past v3.1.9.3

Only relevant if this pod has EVER run v3.1.9.3 or earlier. Skip this if it
has only ever run v3.1.9.4 or later.

**Why.** Every release up to v3.1.9.3 decided whether a conversation needed
a lazy history backfill by checking whether a facts file already existed —
so a stale `in_progress` or `failed` backfill record on a conversation that
also had live facts was ignored forever, silently. v3.1.9.4 fixed that:
`backfill.needs_backfill()` now reads the RECORD first, and RESUMES a stale
`in_progress` or backed-off `failed` record. Correct, but it means the
upgrade itself is the trigger: the instant each such conversation's next
eligible request arrives, its backfill restarts from scratch. A pod that
has been running v3.1.9.3 or earlier for a while can have several of these
sitting on the volume at once, and each one spends a background run of
vLLM calls proportional to that conversation's ENTIRE history — hundreds to
low thousands of calls, all competing with her live chat on the one GPU.
See RUNPOD_DEPLOY.md's "Upgrading within v3.1.9.x, and rolling back" for
the full explanation.

**Run this BEFORE upgrading, from a clone — not a copy.** This script
imports `compactor/backfill.py` (see WHY below), so `HERE.parent /
"compactor"` has to resolve to a REAL package. Copying just the one file
to `/data/scripts/` makes that resolve to `/data/compactor`, which does
not exist — and there is no `ssh`/`scp`/`rsync` in the image to copy the
rest of a release in anyway. On a pod that has ever run v3.1.9.3 or
earlier the installed `compactor/backfill.py` ALSO lacks
`_MAX_BACKFILL_ATTEMPTS`/`_backoff_ready` (v3.1.9.4 added them), so even a
correctly-resolved but too-old package cannot answer this script's
questions — the script detects that and refuses with an actionable
message rather than crashing, but the only way to actually RUN it on such
a pod is a clone, which supplies the whole self-consistent release:
```bash
git clone --depth 1 --branch <tag-or-branch> \
    https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
```
(`git` ships in the image.) `/opt` is container-local: nothing lands on
the network volume, and the clone disappears on the next restart — clone
again next time you need it. `--compactor-pkg PATH` overrides package
auto-detection entirely, if you have a package somewhere else.

**Dry run first: reports only, writes nothing.**
```bash
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py \
    --store /data/openwebui/compactor
```
(the compactor's storage root is `/data/openwebui/compactor`, **not**
`/data/compactor` — see COMMANDS.md). Read the report. Each record is
classified `leave` (already terminal — nothing to do), `would-resume`
(this is what an upgrade past v3.1.9.3 would restart), or `needs-review`
(ambiguous — read the reason printed under it). A dry run exits 3 if any
record would resume; that is informational, not a failure — a dry run
that finds nothing to resume exits 0.

**Close what's safe to close.** This rewrites every `would-resume` record
that also has a facts file to the terminal state `abandoned` — the same
state `backfill.py` itself writes once a record has spent its retries —
after backing up the original beside it (`<name>.backfill.json.bak-<UTC
stamp>`, never deleted, never overwritten by a later run):
```bash
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py \
    --store /data/openwebui/compactor --apply
```
It refuses to run while anything answers the compactor's `/health` (the
compactor could be writing these exact files right now); stop it first
(`supervisorctl stop compactor`) if the dry run above found anything to
close, or pass `--force` to override with a loud warning. It also refuses,
without `--force`, to close a `would-resume` record whose conversation has
NO facts file at all — that conversation has never had a fact extracted,
and closing its only open backfill attempt would cancel that, not defuse a
hazard.

**Machine-readable output**, for scripting a fleet of pods: add `--json`.
**One conversation only**: add `--conv <conv_id>` (repeatable). **A
different compactor to probe before `--apply`**: add `--health-url URL`
(default `http://127.0.0.1:8080/health`) — only an unambiguous connection
refusal there is read as "not running"; a real response, a non-200
status, or the probe itself timing out all refuse `--apply` the same way
a confirmed-live compactor does. See "Exit codes — the shared convention
across the operator scripts" above for what each code means, and the
script's own module docstring for exactly which of them this script
returns and why (0 succeeded or nothing to do; 1 refusal/error, or an
`--apply` that closed NONE of its targeted records, or any record left
`needs-review`, or every `--conv` value given being rejected; 3 a dry run
found `would-resume` records with nothing else needing attention; 4 an
`--apply` that closed SOME but not all of its targeted records).

**The alternative**, if you would rather not touch any records by hand:
set `COMPACTOR_BACKFILL_MAX_ATTEMPTS=0` in the template before upgrading.
That does not close the stale records — they stay `in_progress`/`failed`
on disk — but it makes `needs_backfill()` treat every one of them as
already at the attempt cap, so none of them resume. Prefer running the
script instead when you have the chance: it leaves the store in the same
terminal shape backfill.py's own retry-exhaustion path produces, rather
than relying on a template setting nobody has to remember to remove later.

**Timing, on a v3.1.9 pod specifically.** The four stale records verified
on the real 2026-09-22 backup are already INERT under v3.1.9 itself:
`needs_backfill()` on that release returns False at the facts-file check
before it ever reads `state`, so they are not resuming anything today.
`v3.1.9` also does not recognize `"abandoned"` as a state at all — it
knows only `in_progress`/`failed`/`complete` — so a record this script
writes as `abandoned` falls into v3.1.9's own retry branch if somehow
re-read by that release. Harmless there only because a facts file already
exists for all four (the same facts-file check short-circuits it), which
is exactly why this is worth doing at UPGRADE time (when the newer
`needs_backfill()` would actually act on `state`) rather than treating it
as urgent on an unpatched v3.1.9 pod.

## Catching a conversation's summary hierarchy up from a webui.db export

**Why.** `POST /admin/conversations/{conv_id}/compact` drains the same
L1/L2/L3 rollup loop this script uses, but it rebuilds the transcript to
summarize from the EPISODIC store (chromadb) — a rolling window, not an
archive. On a conversation whose hierarchy has fallen far behind (a real
example: ~3,850 messages against a summary hierarchy that only covers the
first 2,720 turns — N16: an earlier version of this section said "~1,600
turns", but the real 2026-09-22 backup's `last_summarized_turn` is
2,720), the episodic store holds as few as 70 exchanges —
nowhere near enough for `/compact` to close the gap. It either refuses
outright (its own guard against summarizing text that is not the text the
chunk labels claim) or would have to cover thousands of turns as gap
placeholders, recording them as permanently unknown. The full transcript
exists in exactly one place outside OpenWebUI's live database: an export
of `webui.db`. `scripts/import-history.py` reads that export and drives
the real rollup loop over it.

**This backlog is also why the OpenWebUI History cap cannot be turned on
yet.** With the cap off, the client resends the whole conversation on
every turn, which is the only reason the backlog is even visible to a
rollup at all. See RUNPOD_DEPLOY.md's "Upgrading within v3.1.9.x" for why
turning the cap on before this catch-up runs would strand the backlog for
good.

**Run this from a clone — not a copy.** This script imports
`compactor/summarizer.py` and `compactor/memory.py`, so, exactly like
`backfill-records.py` above, `HERE.parent / "compactor"` has to resolve
to a REAL package — copying just the one file to `/data/scripts/` makes
that resolve to `/data/compactor`, which does not exist, and there is no
`ssh`/`scp`/`rsync` in the image to copy the rest of a release in anyway:
```bash
git clone --depth 1 --branch <tag-or-branch> \
    https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
```
(`git` ships in the image.) `/opt` is container-local: nothing lands on
the network volume, and the clone disappears on the next restart — clone
again next time you need it. `--compactor-pkg PATH` overrides package
auto-detection entirely, if you have a package somewhere else.

**Get a webui.db export.** Never point this at the live database — it
refuses outright if a `-journal` or `-wal` file sits beside the path you
give it, which is what a live or crashed database leaves behind. Use a
backup snapshot, or a copy taken while OpenWebUI is stopped.

**This is a CHAT OUTAGE for the whole run, not a quiet background job
(N12).** OpenWebUI's own backend is the compactor listening on
`:8080` (`OPENAI_API_BASE_URL=http://localhost:8080/v1`), and the very
next step stops that compactor for the entire run — she cannot chat at
all until it is started again. The dry run's own estimate (below) is a
FLOOR, not the real duration: it reported ~65 calls / ~43 minutes at
`--seconds-per-call 40` on the real 2026-09-22 backup, but a round-2
hostile review measured **134 real completions** against a fake vLLM for
that exact conversation (map-reduce chunks cost more than one call each),
and her real replies run 7.5k-11k tokens, so real vLLM will cost more
still. Budget **1.5 hours or more**, and remember `--max-calls 200`
(the default) can mean this takes MULTIPLE runs if the backlog is larger
than one budget's worth (see N5/N8-adjacent notes on re-running). **Do
not start this while she might be chatting** — schedule it for a time she
is known to be away, and tell her chat will be unavailable for the
duration before you start.

**Interrupting a long run is now safe (round-3 fix pass A — see below),
so you are not committed to the full 1.5+ hours in one sitting.** If the
outage window turns out to be too short, Ctrl-C the run (or let
`--max-calls` stop it); it stops cleanly between rollup units with
everything already done genuinely saved, and re-running the exact same
command later picks up exactly where it left off — see "Interrupting
`--apply`" below for the details and the proof.

**Stop the compactor first** — this script and the live compactor would
otherwise both be writing the same `summaries/<conv_id>.json`:
```bash
supervisorctl stop compactor
```

**Before running this script, confirm `--store` points at the REAL
compactor storage root** — it must already exist, contain a
`summaries/` subdirectory, and already have a `summaries/<conv_id>.json`
for the conversation being caught up, before any work, dry run or
`--apply`. This is a catch-up tool for a conversation the compactor
already knows about, not a bootstrap tool: a `--store`/`--chat-id`/
`--conv-id` typo used to silently build a brand-new store from scratch
and report an encouraging "re-run with --apply" while burning real GPU
time against the wrong place; it now refuses outright instead (B5).

**Dry run first: reports only, makes zero vLLM calls.**
```bash
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/import-history.py \
    --webui-db /path/to/webui.db.export --chat-id <conversation id> \
    --store /data/openwebui/compactor
```
(the compactor's storage root is `/data/openwebui/compactor`, **not**
`/data/compactor` — see COMMANDS.md). Read the report: which source it
used to reconstruct the transcript (`history.messages` JSON, or the
`chat_message` table fallback), how many turns it found against the
watermark already on disk, how many L1/L2/L3 units are estimated due, an
estimated (floor) count of real vLLM calls, an ESTIMATED wall-clock
(`--seconds-per-call`, default 40s, labelled an estimate — a chunk needing
map-reduce costs more than one call), and every anomaly found in the
transcript (a missing reply, non-alternating turns). The due/call estimate
is offset-aware: it sizes itself against `effective_position =
max(recorded_position, current_turns)`, not the raw reconstructed branch
length, so it does not undercount when the store's `turns_seen` already
sits ahead of the branch (the ordinary real-backlog shape — see B1 below).
On the real 2026-09-22 backup this reports **58 L1 chunks / 6 L2 folds / 1
L3 refresh due, ~65 estimated vLLM calls** (58 and 65, not the 57/64 an
earlier build of this fix under-reported by exactly one L1 chunk). A dry
run exits 3 if anything is due; that is informational, not a failure — a
dry run that finds nothing due exits 0.

**The resume offset is verified, not assumed (B1).** Her real store's
position-to-branch mapping is PIECEWISE (different constant offsets at
different ranges of the transcript, from edited/abandoned messages
scattered through the history) — a single flat `current_turns -
len(window)` offset used to leave a real gap of turns uncovered by any
chunk, permanently mislabelling every chunk written after it (chunk
recording is append-only). `--apply` now derives the offset from the
store's own covered-turn fingerprint record, growing the window it checks
against on ambiguity and refusing outright if no single consistent offset
exists. Both dry run and `--apply` report this as `resume_offset` (and
`resume_offset_detail`) in `--json`; on the real backup this verifies to
**22**, not the flat 20 the old code computed.

Note on a short-turn-count report: this script checks the transcript's
CONTENT against the store, not merely its length, before refusing. A
conversation with edited or abandoned messages can legitimately
reconstruct SHORTER than the store's `recorded_position` — that is
expected (see the script's own `_prefix_matches_store` docstring) and is
not refused. What IS refused is content that does not match what the
store already has confirmed — the actual signature of a wrong
`--chat-id`/`--conv-id` or an export of a different conversation.

**Run it for real:**
```bash
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/import-history.py \
    --webui-db /path/to/webui.db.export --chat-id <conversation id> \
    --store /data/openwebui/compactor --apply
```
It backs up the existing `summaries/<conv_id>.json`, AND
`summaries/<conv_id>.archive.json` whenever an L2 fold or L3 refresh
actually runs (H3 — the archive sidecar changes too on those, and used to
go unbacked-up), beside themselves (`<name>.json.bak-<UTC stamp>`, never
deleted, never overwritten by a later run) before writing anything, and
refuses outright if either exact backup path already exists.
`--apply`'s report (and `--json`'s `archive_backup` field) names the
archive backup path when one was made. It refuses to run while anything
answers the compactor's `/health` — add `--health-url URL` to point the
probe elsewhere (default `http://127.0.0.1:8080/health`, the real pod's
own endpoint). A clean connection refusal there is read as "not running"
and `--apply` proceeds. Anything else refuses, but `--force` no longer
treats every non-refusal answer the same way (round-3 fix pass A, N7): a
**DETECTED** live compactor (a real response — the health endpoint itself
answered) is never overridable by `--force`, full stop, because there is
no cross-process lock the live compactor itself respects (checked: it has
none), so this health check is the only thing standing between `--apply`
and a live rollup tearing the same state file. An **AMBIGUOUS** probe (a
timeout, or any other error short of a clean connection refusal — M3) is
still the one case `--force` can override, when you have confirmed by
some other means (e.g. `supervisorctl status compactor`) that it is
really stopped.

**This run's own cross-process import lock.** Independently of the
health probe, `--apply` also takes an exclusive OS lock on
`summaries/<conv_id>.json.import.lock` for the whole run (round-3 fix
pass A, N7) — importer-vs-importer protection, not a substitute for
stopping the live compactor: the lock file holds no data, and a second
`import-history.py --apply` against the SAME conversation while one is
already running refuses outright, naming the lock path. The advisory
lock is released automatically when the holding process exits, by any
means (including a kill), so a stale lock file left behind by a killed
run is never a deadlock — it is safe to remove by hand only if you have
confirmed no `import-history.py` process is actually still running (the
refusal message says this).

**The chat_message table is cross-checked against the JSON history
(round-3 fix pass A, architect follow-up).** OpenWebUI's own code prefers
the `chat_message` SQL table over the chat's embedded JSON history when
building what a live request actually sees — this script walks the JSON
first, falling back to the table only if the JSON is unreadable, so a
real disagreement between the two would mean this script's transcript is
not what the live compactor is actually working from. `--apply` (and the
dry run) now builds both reconstructions and refuses if they disagree,
rather than silently trusting whichever one was tried first. Verified on
both the real 2026-09-22 and 2026-09-23 backups: the two reconstructions
are byte-identical for her conversation (same turn count, same ids, 0
text differences) — the cross-check exists for the day that stops being
true, not because it already isn't.

**Interrupting `--apply` — Ctrl-C (SIGINT), SIGTERM, SIGHUP, `kill -9`, or
a pod restart — is safe (round-3 fix pass A closes N1, the round-2
hostile review's BLOCKER finding).** Before a single vLLM call is made,
the on-disk anchor (`tail_fp`/`head_fp`/`window_turns`) is PRE-SEATED with
exactly what this run's own trimmed window will produce — never cleared
to blank the way the pre-round-3 code did. The drain then runs one rollup
UNIT (one L1 chunk, one L2 fold, or the L3 refresh) at a time, and state
is saved to disk after EVERY unit, never batched until the end. SIGINT,
SIGTERM and SIGHUP are all handled the same way — a flag checked between
units, never raised into the middle of one — so the run always stops
cleanly between units: exit **4** (progress made, work remains, re-run to
continue) if at least one unit had already completed and was saved, or
exit **1** if the signal landed before the very first unit finished
(nothing accomplished yet — still safe to re-run, just not "progress").
A `kill -9` or a real pod restart loses at most the one unit that was in
flight when it landed, never more, because everything before it was
already saved and the anchor was never blank to begin with. Re-running the exact same command afterward resumes from the
real watermark and never redoes finished work, and the live compactor's
next request stays contiguous with whatever the interrupted run actually
finished — even if it finished zero units. Proven directly against the
real 2026-09-22 backup and the published digest: `kill -9` at 5 distinct,
counted points plus one SIGTERM and one SIGHUP (7 kill points total),
each followed by a real live-style rollup check (no hole) and a full
re-run to completion (exit 0) — see
`compactor/test_real_image_import_apply.py`.

If you want to discard an interrupted (or any) run's progress entirely
rather than resume it, restore `summaries/<conv_id>.json.bak-<stamp>`
(and the matching `.archive.json.bak-<stamp>`, if present) by hand over
the live file — the importer writes this backup before touching
anything, on every `--apply` run. This is never necessary for a normal
interrupt-and-resume; it is only for throwing a run's progress away on
purpose.

**The feasibility check, in operator terms (round-3 fix pass A, N4/N6).**
Both the dry run and `--apply` now run through the exact same check
before either one reports anything: verify the resume offset against the
store's own evidence, then compute the window this run would actually
drain from that SAME offset, bounded so it can neither fall behind what
the store's own chunks already claim to cover nor run past how many turns
the reconstructed transcript actually has. If that check fails — an
offset that cannot be verified, or a window that falls outside those
bounds — BOTH the dry run and `--apply` refuse with the identical detail,
exit 1. Before this fix, the dry run always printed `safe_to_apply:
True` regardless, so an operator could see "safe to apply" and then have
`--apply` burn real vLLM calls before refusing on exactly the case the
dry run should have caught for free. A dry run that reports `safe_to_apply:
True` now means what it says.

**`.bak-*` files accumulate — this is by design, and needs occasional
manual pruning (N16).** Every `--apply` run that actually writes leaves
at least one new `summaries/<conv_id>.json.bak-<UTC stamp>` (plus a
matching `.archive.json.bak-<stamp>` whenever that run also did an L2
fold or L3 refresh), and NONE of them is ever deleted or overwritten by
this script — that is deliberate: a backup that could silently vanish or
get clobbered is not a backup, and it is what lets a bad run be undone by
hand. Left alone forever, they add up: a backlog large enough to need
`--max-calls 200`'s default budget spread across 11 separate runs (the
real 2026-09-22 backup's own shape) leaves 22 `.bak-` files in
`summaries/` for that one conversation by the time the catch-up finishes.
`scripts/backfill-records.py --apply` does the same thing, one dated
`.bak-` per closed record, under `facts/`.

To prune safely: keep at minimum the OLDEST backup for each conversation
(the pre-catch-up state, in case you ever need to unwind the whole run)
and the NEWEST (in case the most recent write turns out to be wrong).
Everything strictly between those two is redundant once you have
confirmed (`/health/full`'s `checks.hierarchy` lag, and a normal-looking
reply) that the catch-up worked, and is safe to delete:
```bash
# list a conversation's backups oldest-first, then remove all but the
# first (oldest) and last (newest):
ls -1tr /data/openwebui/compactor/summaries/<conv_id>.json.bak-* | \
    sed '1d;$d' | xargs -r rm -v
```
Do this per conversation, and do it for `.archive.json.bak-*` and for
`facts/*.backfill.json.bak-*` separately (the glob above only matches
one sidecar at a time on purpose — a single wildcard across sidecar
kinds risks matching more than you intended to delete). Never prune
while a catch-up run for that same conversation is in flight.

**Start the compactor again:**
```bash
supervisorctl start compactor
```

**Check afterward:**
```bash
curl -s http://127.0.0.1:8080/health/full | python3 -m json.tool
```
Look at `checks.hierarchy` — its lag for this conversation should be
falling (or gone) on subsequent requests, rather than sitting at the same
large number it was stuck at before the catch-up ran.

**What this will not do.** It never rebuilds the episodic/chromadb index
— that is a retrieval backlog, a different job. It never touches
`facts/`, `chromadb/` or `personas/` — its entire blast radius is
`summaries/<conv_id>.json` (and `summaries/<conv_id>.archive.json`
whenever an L2 fold or L3 refresh runs — H3) plus their own dated
backups. See "Exit codes — the shared convention across the operator
scripts" above for what each code means; for this script specifically, 1
is also what `--apply` returns if it ran but the watermark never
advanced at all (nothing reachable, or vLLM unreachable throughout), and
4 is what it returns if the watermark DID advance but work still remains
(most often `--max-calls` running out) — re-run to continue, the same as
`backfill-records.py`'s own code-4 case. See the script's own module
docstring for the exact per-code triggers, and
`compactor/test_import_history_script.py` for coverage, including the
fork-in-history, interrupted-and-resumed-apply, content-vs-length guard,
and verified-resume-offset cases.

---

## Getting a real shell into a pod (installing sshd, v3.1.9.6)

**Do not use `scripts/setup-sshd.py` until `v3.1.9.6` is tagged.** A
hostile review found this script could report success while a real
password login from anywhere still worked (a `Match` block sshd's own
`-T` cannot see without `-C`) and could accept and store a private key
verbatim — both fixed, but only in the commit this release tags. Running
an untagged working copy of this script is running exactly the version
those two holes were found in.

**Why the image has no sshd.** `Dockerfile:63-80` never installs
openssh-server, and it never has — `entrypoint.sh` never mentions ssh, and
`supervisord.conf` has no `[sshd]` program and, more to the point, no
`[include]` section at all: a `.conf` file dropped straight into
`/etc/supervisor/conf.d/` on a running pod is silently ignored by the
supervisord process already running there. Keeping openssh out of the
image is deliberate — v3.1.9.6 is a documentation-and-scripts release, and
adding a daemon to the image would mean rebuilding it, which is exactly
what this whole release avoids (see the release's own note at the top of
CHANGELOG.md). `scripts/setup-sshd.py` is the operator's way to get a real
shell anyway: it installs and hardens OpenSSH server LIVE, inside the
running container's own writable overlay, never touching the image.

**Re-run this after EVERY pod restart.** The container filesystem resets
on every restart — everything this script writes to `/etc/ssh` (and, with
`--supervise`, `/etc/supervisor/conf.d/`) lives on the overlay, not on
`/data`. It is idempotent and safe to run again on a pod that already has
sshd running; a re-run with nothing left to do exits **0** and changes
nothing (see EXIT CODES below — this script used to have 0 and 3 swapped
from the other two operator scripts; normalized in v3.1.9.6).

**Is the script on the pod?** Check BOTH of the paths the two methods
below actually use — a script obtained by cloning ends up at
`/opt/zl-repo/scripts/setup-sshd.py`, NOT `/data/scripts/setup-sshd.py`
(N16: an earlier version of this procedure cloned to one path and then
gave run commands for the other, which only worked by accident if you
happened to use the upload method instead of the clone):
```bash
ls -la /opt/zl-repo/scripts/setup-sshd.py /data/scripts/setup-sshd.py
```
There is no `ssh`/`scp`/`rsync` in the image. If the script is not
already at either path, get it there by either:
- cloning the repo, same as `backfill-records.py`/`import-history.py`
  below (this script does not import the `compactor` package, so it does
  not strictly need the rest of the clone, but this keeps one consistent
  method for all three):
  ```bash
  git clone --depth 1 --branch <tag-or-branch> \
      https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
  ```
  — the script then lives at `/opt/zl-repo/scripts/setup-sshd.py`, and
  every command below must use THAT path, not `/data/scripts/...`; or
- uploading just this one file through the RunPod Web Terminal, or with
  `runpodctl send`/`receive` (RunPod injects `runpodctl` into the pod at
  runtime — it is **not** shipped in this image; `/usr/local/bin` is
  empty here) to `/data/scripts/setup-sshd.py`, so a persistent copy
  survives on the volume across restarts (it still has to be RE-RUN after
  each one — see above) — every command below must use THAT path in this
  case.

The commands below use the clone path (`/opt/zl-repo/...`) — substitute
`/data/scripts/setup-sshd.py` throughout if you used the upload method
instead.

**Dry run first.** Runs a real `apt-get update` (package lists only,
nothing installed — Dockerfile prunes them, so this is required every
time) and reports the installed/candidate `openssh-server` version, which
key source it would use, whether the drop-in or the direct-edit case
applies on THIS pod's shipped `sshd_config`, and whether it would start or
restart sshd. Nothing is written, installed, or started:
```bash
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/setup-sshd.py
```

**Run it for real.** RunPod injects the operator's own public key as the
`PUBLIC_KEY` environment variable, which is the default key source (in
order: `--authorized-key-file`, `--authorized-key`, `$PUBLIC_KEY`, then an
existing non-empty `/root/.ssh/authorized_keys`):
```bash
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/setup-sshd.py --apply
```
This installs `openssh-server` (or `--only-upgrade`s it to the apt
candidate if already present — **never** `apt-get upgrade`/`dist-upgrade`,
which would touch unrelated packages under a live vLLM process), adds the
key to `authorized_keys` (append-only — an existing file is backed up
first and never clobbered), hardens the config, ensures host keys, and
starts sshd as a plain background daemon. It verifies the result for real
with `sshd -t`, then `sshd -T` with NO `-C` AND `sshd -T -C` across a full
representative address matrix (loopback v4/v6, one address in each of
10/8, 172.16/12, 192.168/16 and 100.64/10, a link-local address, a public
v4/v6 address, and this container's own address(es) via `hostname -I`),
each checked for both `user=root` and a non-root user (round 3, N2 —
round 1 checked only loopback and one public address, which missed a
`Match` reachable only from a private-range client, the shape a RunPod
proxy actually connects from) — never just by reading back the file it
wrote, and never trusting a bare `sshd -T` alone, which cannot see what a
`Match` block would do for a real connection — and refuses, rolling back
the config write and any `authorized_keys` append made this run, if any
of those contexts would allow password login or leave sshd listening on
the wrong port. "Listening" is also no longer just "something answers on
the port": the listening socket's kernel inode is mapped to the pid that
holds it (via `/proc/<pid>/fd`), and that pid must be the one this run's
own pidfile actually recorded — a foreign process (or another sshd
running with a different pidfile) already holding the port is refused
outright, before anything is written, rather than mistaken for "already
hardened" (round 3, N3). At minimum it sets: `PasswordAuthentication no`,
`PermitEmptyPasswords no`, `KbdInteractiveAuthentication no`,
`ChallengeResponseAuthentication no`, `PubkeyAuthentication yes`,
`PermitRootLogin prohibit-password`. Only a FINGERPRINT (never the key
text itself) is ever reported for a key that was added — `ssh-keygen -l
-f`'s exit code is not trusted on its own to keep a private key out (the
real binary returns 0 on a private key file too); a candidate must start
with a known public-key-type token and be a single line before that check
even runs. `/root/.ssh` and `authorized_keys` are re-hardened to
`0700`/`0600` (plus ownership) on every `--apply`, even when there is no
new key to add.

**Two more hard preconditions, refused outright rather than warned about
(round 3, N11/N15).** Before writing any key, the script reads the
EFFECTIVE `AuthorizedKeysFile` from `sshd -T` and refuses if it is not
the default `~root/.ssh/authorized_keys` — writing a key to a path sshd
was never configured to read used to report success ("APPLY complete")
while login still failed with "Permission denied (publickey)". And
`/root/.ssh`/`authorized_keys` are refused outright if either is a
SYMLINK (checked with `lstat`, plus `O_NOFOLLOW` on the actual write as a
second-layer guard against a race) — a symlinked `authorized_keys`
pointing elsewhere (e.g. into `/etc/hostname`) used to get a real key
appended into whatever it pointed at, chmodded 600, with the script still
reporting success.

**Drop-in vs. direct edit.** The script checks THIS pod's real
`sshd_config` fresh on every run rather than assuming: if it has an
`Include /etc/ssh/sshd_config.d/*.conf` line and that line appears before
any directive the script manages that is already active (sshd is
first-match-wins), it writes `/etc/ssh/sshd_config.d/00-zions.conf` as a
drop-in; otherwise it edits `sshd_config` directly, backing it up first.
**Verified against the real `angreg/zions-light-ai` image**: the shipped
config carries `Include /etc/ssh/sshd_config.d/*.conf` at line 12, and the
only directive the script manages that ships active in that file is
`KbdInteractiveAuthentication no` at line 71 — well after the Include — so
this pod always takes the drop-in case. Confirmed for real with `sshd -T`
after both a fresh `--apply` and a hostile override (an active
`PasswordAuthentication yes` line inserted after the Include still lost
to the drop-in, exactly as first-match-wins predicts).

**A pre-existing `Match` block anywhere in the config `Include` closure,
or a conflicting drop-in, refuses the run outright.** `sshd -T` with no
`-C` evaluates NO `Match` criteria, so a `Match Address * /
PasswordAuthentication yes` block left by a previous operator or pod
template made an earlier version of this script report "password
authentication is disabled" while a real connection from anywhere could
still log in with a password (a hostile review confirmed this against
the published digest). The script now refuses — before writing anything
— if it finds an active `Match` line anywhere in the FULL `Include`
closure reachable from `sshd_config` — not just `sshd_config` and
`sshd_config.d/*.conf` themselves. Round 1's fix only scanned those two
locations, which a round-2 hostile review found was not enough: a
`Match` pulled in by an `Include` from a THIRD location (e.g. a
`sshd_config.d/*.conf` drop-in that itself does nothing but
`Include /etc/ssh/site.d/*`) bypassed the refusal entirely, and a real
root password login succeeded from a private-range address through it
(N2). `Include` is now followed recursively, case-insensitively,
glob-expanded, with relative paths resolved against `/etc/ssh` the same
way sshd itself resolves them — and an `Include` that resolves to
anything OUTSIDE `/etc/ssh` is its own refusal. The script also still
refuses on another drop-in that sorts before its own `00-zions.conf`, or
a conflicting active `Port` line anywhere it does not control (`Port`
accumulates across files in OpenSSH instead of following
first-match-wins, so this script's own `Port N` never suppresses an
unrelated `Port 22` elsewhere). Remove or fix the conflicting file and
re-run.

**RunPod template.** The template's port mapping must expose the TCP port
this script configures (default 22, `--port N` to change) for it to be
reachable from outside the pod. The Web Terminal itself keeps working
either way — it does not go through sshd. `--port N` now also verifies,
for real, that sshd ends up listening ONLY on `N` — not also on 22 —
refusing if a conflicting `Port` line exists anywhere else in the loaded
config (see "Drop-in vs. direct edit" above).

**Host keys and `/data/ssh/`.** By default the script persists host keys
to `/data/ssh/` (directory `0700`, key files `0600`) and copies them into
`/etc/ssh/` on every run, so the pod's SSH host-key fingerprint stays the
same across a restart instead of changing every time — which is what
trains an operator to click through a host-key warning instead of reading
it. **`/data/ssh/` now holds private host keys on the network volume — do
not include it in any bundle, snapshot, or backup archive shared outside
the pod.** It is NOT swept into `compactor/backup.py`'s own archive:
`create_backup` copies exactly two things — the `webui.db` snapshot and
`COMPACTOR_STORAGE_ROOT` (`/data/openwebui/compactor` by default) — never
a walk of `/data` as a whole, so `/data/ssh/` sits outside its blast
radius entirely (confirmed by reading `backup.py`, not assumed). Opt out
of persistence with `--no-persist-host-keys`, at the cost of the
fingerprint changing on every restart. A refusal discovered after this
script has already written something (the config, or an
`authorized_keys` append) rolls that write back — host key files under
`/data/ssh/` are the one deliberate exception: they are never rolled
back, because they are never what causes a refusal, and deleting or
regenerating them on a refusal would only churn the pod's host-key
fingerprint for no security benefit.

**`--supervise` (opt-in, OFF by default).** Instead of a plain background
daemon, appends a `[program:sshd]` block to
`/etc/supervisor/conf.d/supervisord.conf` and runs `supervisorctl reread`
then `update` (backing the file up first; a failed `reread` restores the
backup and refuses). **Verified against a throwaway container running
this image's real supervisord** (no GPU — `vllm` alone goes FATAL, as
expected, since it needs one): adding the one new `[program:sshd]` section
and running `reread` + `update` started only that new program.
`compactor`, `openwebui` and the `processes` event listener kept their
exact same pids and continuously-growing uptimes across the whole
sequence — `supervisorctl update` never touched them. A second, idempotent
`--apply --supervise` left `sshd`'s own pid unchanged too; changing
`--port` while already supervised correctly bumped `sshd` to a new pid via
a scoped `supervisorctl restart sshd`, with the other programs again
untouched throughout. A missing `supervisord.conf` (nothing to append a
program to) now refuses cleanly with full `--json` output rather than
raising an unhandled exception. If the handoff itself fails (a real
`supervisorctl reread` failure), the standalone daemon this script
stopped to attempt the handoff is restarted and reconfirmed listening
before the run returns its refusal — this script never leaves the pod
with no sshd because of its own actions.

**Machine-readable output**, for scripting a fleet of pods: add `--json`.
See the script's own module docstring for the full exit-code contract —
see "Exit codes — the shared convention across the operator scripts"
above for what each code means — and `compactor/test_setup_sshd_script.py`
for coverage: a dry run that writes nothing, a clean install, idempotency,
append-not-clobber on an existing `authorized_keys`, every refusal path
(not root, a config-safety refusal — a `Match` block, a conflicting
drop-in, or a conflicting `Port` directive — no usable key, a malformed
or private key, a config that would allow password auth for the default
context or either `-C` address, `sshd -t` failing, a missing
`supervisord.conf`, a failed `supervisorctl reread`, and sshd not
confirmed LISTENING after a start/restart), that `apt-get
upgrade`/`dist-upgrade` is never invoked, and that no private key
material ever reaches stdout, stderr, or `--json`.

**Real-container verification.** `compactor/test_real_image_setup_sshd.py`
runs the real, unmodified script inside a throwaway container from the
published digest, with real `openssh-server`/`openssh-client` apt
installs (needs network from inside the container) — a real key login,
a real failed password login (including with the `Match`-block scenario
above), a real rejected private key, a real idempotent second `--apply`,
a real `--port 2222` listening only on 2222, and a real `--supervise`
handoff failure that still leaves a real, listening sshd afterward. This
is the suite that actually exercises what the unit suite's fakes cannot
(a real `ssh-keygen -l -f` on a private key, a real bound socket, a real
`Match` block evaluated with `-C`).

**Undo.** Stop sshd (`supervisorctl stop sshd` if `--supervise` was used,
otherwise find its pid via `/run/sshd.pid` and send it `SIGTERM`), remove
`/etc/ssh/sshd_config.d/00-zions.conf` (or restore the
`sshd_config.bak-<stamp>` this script made if it used the direct-edit
case), and remove the appended `[program:sshd]` block from
`supervisord.conf` (or restore ITS `.bak-<stamp>`) if `--supervise` was
used — then `supervisorctl reread && supervisorctl update`. None of this
survives a pod restart anyway, since it all lives on the container
overlay; the simplest "undo" on a live pod is usually just not re-running
this script after the next restart.

---

## Removing decoration from her stored replies, facts and episodic memory (v3.1.9.6)

**This is the answer to "how do I remove the symbols without running
`/forget`."** `scripts/clean-decoration.py` strips emoji, rule-line
"walls" (`═══`, `━━━`, long `====` runs) and LAW-board status tags
("✅ ... → ACTIVE (100%)") out of the three places the model re-reads its
own decoration from — her stored chat replies, active facts, and episodic
(chromadb) memory — deterministically, with no LLM call. It never touches
a word she actually wrote, and it never runs `/forget`: nothing is
deleted, only decoration is stripped from text that is kept.

**This is a SHORT chat outage, both services stopped.** `webui.db` and
the compactor's `facts/`/`summaries/`/`chromadb/` are live files each
service reads and writes on every request — the same hazard
`import-history.py` and `backfill-records.py` already refuse against, and
this script refuses the same way. Dry run makes no writes and finishes in
well under a minute; `--apply` on a store her size is a few seconds of
I/O plus one CPU embedding pass per changed episodic document — call it
under two minutes end to end:

```bash
git clone --depth 1 --branch <tag-or-branch> \
    https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
supervisorctl stop openwebui compactor
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/clean-decoration.py \
    --webui-db /data/openwebui/webui.db --store /data/openwebui/compactor \
    --conv <chat id>
# read the report, then:
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/clean-decoration.py \
    --webui-db /data/openwebui/webui.db --store /data/openwebui/compactor \
    --conv <chat id> --apply
supervisorctl start openwebui compactor
```

**The three targets** (`--only`, default all three; each independently
backed up before any write):
- **webui** — her current branch (walked via `currentId`/`parentId`,
  never insertion order). OpenWebUI 0.11 keeps an assistant turn's text in
  THREE places at once (`history.messages[id].content`, the small flat
  `messages[]` tail cache, and the `chat_message` table's own row) and
  this script updates all three together or none — it refuses outright
  if the JSON and table copies of an in-scope message already disagree,
  rather than clean one and leave the other stale. Only `role: "assistant"`
  turns are ever in scope; a `role: "user"` message is never touched.
- **facts** — `facts/<conv>.json`'s active fact text. There is no
  embedding cache on a fact record to invalidate (verified against both
  v3.1.9 and v3.1.9.4), so cleaning the text is the whole job. A fact
  that would clean to empty, or to a duplicate of another fact's text, is
  reported and left byte-for-byte as it was — never deleted; that would
  be `/forget` semantics by another name.
- **episodic** — the chromadb documents for `--conv`, in the REAL
  collection compactor's own `retrieval.py` uses
  (`retrieval.COLLECTION_NAME`, `"conversation_turns"` — an earlier
  build of this script had this hardcoded to the wrong name,
  `"episodic_memory"`, which does not error but silently creates and
  writes an empty phantom collection the live compactor never reads;
  fixed to read the real name from the module). Ids are content-addressed
  (sha256 of the stored text), so a changed document gets re-embedded
  with the compactor's own `retrieval._embed` (same model, same code path
  the live compactor uses) under a new id, upserted with the original
  metadata, and the old id deleted. Only the `[assistant]: ...` half of a
  stored exchange is ever cleaned — the `[user]: ...` half never is.
  Documents that exist for what is content-wise the same conversation but
  under a DIFFERENT, older `conv_id` (a real thing found in her store) are
  correctly left alone: her live retrieval filters strictly by exact
  `conv_id` and can never reach them, so they are genuinely out of scope,
  not a miss.

**The anchor rule (why some turns can be "skipped" even without
`--force`).** `compactor/summarizer.py` fingerprints the WHOLE,
decoration-and-all text of each turn to track where it is in the
conversation (`tail_fp`/`head_fp`/`window_turns`, the same anchor
`import-history.py`'s B1 fix made offset-aware). Rewriting a turn's text
inside that anchor's own window, without also rewriting the anchor to
match, would leave the NEXT live request unable to align — a permanent
hole in the summary hierarchy from then on. Her real store's tail anchor
already has a small pre-existing drift against a fresh branch
reconstruction (independent of anything this script does), which this
script works around rather than blindly trusting: before writing, it
asks a well-posed question — does the CURRENTLY STORED anchor still find
a real fingerprint match once these specific turns are cleaned, left
un-rewritten? If yes for the whole requested scope, everything is
cleaned and the anchor is left alone (it still works). If no, candidate
turns are excluded from cleaning — narrowest window first — until it is
yes again; every excluded turn is reported. **`--force`** cleans the full
requested scope anyway and accepts a verified REALIGNMENT of the anchor
(a different but genuinely-matched position) — but still refuses, even
under `--force`, if the new anchor would find NO fingerprint match at all
(a hole, not a realignment); text cleaning still proceeds in that case,
only the anchor is left untouched.

**Exit codes** — the same 5-value convention as the other operator
scripts (see "Exit codes" above). For this script specifically: **4**
means at least one target changed but at least one requested turn was
skipped for the anchor reason above (or a target was refused/failed); **1**
means nothing was changed at all — every candidate turn had to be
excluded, or every target refused. `--restore <stamp>` puts every target
this script backed up under that stamp back exactly as it was, and exits
0 if every backup was found and restored, 1 if any named target's backup
was missing or restoring it failed.

**Verified on the real 2026-09-23 backup, default scope (`--last 6`,
unforced):**

| target | changed | of |
|---|---|---|
| webui (assistant replies) | 4 | 6 requested (2 skipped — inside the drifted anchor window) |
| facts | 17 | 190 active facts examined |
| episodic | 81 | 84 documents for her conv_id |

That run exits 4 (real progress, 2 turns skipped, anchor left untouched:
`window_turns` unchanged at 3880). A follow-up `--force` run, with
nothing left to re-clean, accepts the realignment and rewrites the
anchor, and exits 0.

**Liveness refusals (only for `--apply`)**, all overridden together by
`--force` (never the underlying risk): a `-wal`/`-journal` file beside
`webui.db`; OpenWebUI detected via `supervisorctl status openwebui`
combined with a raw port probe (default 8080, `--openwebui-port`) so
either signal alone can't be trusted; the compactor's `/health`, same
convention as the other operator scripts; and a `/proc/<pid>/fd` scan for
any OTHER process with `webui.db` open. `--force` does NOT override a
failed `PRAGMA integrity_check` after a write, a refused resume-offset, a
history/`chat_message` disagreement, or a hole-risk anchor rewrite (the
anchor rule above) — those always refuse.

---

## Rolling back a bad release

Each release tag is pushed once and not re-pushed by this project (see
[README → Image tags](README.md#image-tags)), but Docker Hub tags CAN be
overwritten. For a rollback you must be able to trust, pin by digest
(`angreg/zions-light-ai:<tag>@sha256:...`; `docker buildx imagetools inspect
<tag>` prints it), and write the current image's digest down before every
deploy. To roll back:

1. **If the History cap filter is installed, set its `max_turns` valve to 0
   first** (OpenWebUI → Admin Panel → Functions → History cap → Valves), and
   leave it at 0 until the newer image is back AND has served one uncapped
   message. Rolling back with the cap on leaves a permanent, unlogged hole in
   her summary hierarchy (hostile review of v3.1.7, reviewer C, F5). See
   RUNBOOK_MEMORY_IDENTITY.md "Rolling the IMAGE back".
2. In the RunPod template, change **Container Image** to the last-good
   image: the one you wrote down before deploying (RUNPOD_DEPLOY.md,
   "Upgrading within v3.1.9.x", step 2). Note that `v3.1.9.6-cu12`,
   `v3.1.9.5-cu12` and `v3.1.9.4-cu12` are all the SAME image (digest
   `sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65`),
   so switching between any of those three tags rolls nothing back. Going further back within v3.1.9.x has one
   consequence to know about, and there is no `v3.1.9.3-cu12` image: see
   RUNPOD_DEPLOY.md "Upgrading within v3.1.9.x, and rolling back". Pre-v3.1.9
   targets are covered by RUNPOD_DEPLOY.md "6. Rollback to v3.1.8".
   **Leave `WEBUI_DB_LOCAL=false` exactly as it is, spelled
   `false`** — v3.1.9 also accepts `0`/`no`/`off` and `1`/`yes`/`True`, but
   the older images read only the exact word `true` as true, so any other
   spelling can mean different things on the two sides of a rollback. See
   RUNPOD_DEPLOY.md "WEBUI_DB_LOCAL".
   A rollback is an image change only; do not restore a backup as part of it.
3. Restart the pod. The Network Volume (and all memory) is unaffected —
   only the code image changes. The `X-Conversation-Id` connection header
   lives in OpenWebUI's database, not the image, so it stays set.

**Never roll back to `:latest`.** It is still `:v3.0` (the same digest) at
v3.1.9.5, because no v3.1.x image has passed the on-pod gate that promotes
it. Nothing documents or tests going from a v3.1.9.x `/data` back to V3.0
code. Roll back only to a pinned tag or digest that this document or
RUNPOD_DEPLOY.md names as a rollback target.

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
