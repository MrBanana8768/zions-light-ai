# Changelog

All notable changes to Zion's Light AI. Format inspired by
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Image tags published at
[`angreg/zions-light-ai`](https://hub.docker.com/r/angreg/zions-light-ai)
on Docker Hub.

---

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
  whole-reply placeholder (10 of 67 on that backup, down from 51 of 68). Fenced
  code is now excluded from the token-run rule (a repeated-value array in a
  ```code``` block is no longer flagged, matching the fragment-line rule's
  existing fence handling; hostile pass #7, F3). User turns, system
  messages and the newest message are never touched. Logged at INFO as a
  count only (no text) — `touched=<n> whole=<k> cut=<n-k>` (hostile pass #8,
  P8-8: the line used to say "replaced N ... with a placeholder"
  unconditionally, which stopped being true once span-cutting made a
  placeholder the MINORITY outcome — 10 of 66 flagged replies on the
  2026-09-16 backup, not all of them). Apart from the fenced-code exemption
  above and the P8-2 fix described below, the detection rules and
  thresholds are unchanged; note that the exemption
  applies wherever `reply_is_degenerate` is used, including the memory skip.
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
- **Fence exemption fixes (hostile pass #8):**
  - **P8-2 (regression, was flagged correctly by v3.1.9):** the token-run
    fence exemption above treated an UNCLOSED ` ``` ` opener as fencing
    everything after it forever — this model uses bare ` ``` ` lines as
    decorative boxes (128 of 1,709 unique real replies in the 2026-09-16
    backup have an odd count), so a real identifier loop starting after
    the last unmatched opener and running to the end of the reply was
    silently exempted and stored to memory/forwarded verbatim. Fixed: a
    run only counts as fenced when the fence actually CLOSES again later,
    and never when the run reaches the end of the reply either way. The
    p7 F3 case (a repeated-value array inside a fence that closes,
    mid-reply) is unaffected.
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
gate note; pre-existing, untouched by this diff, needs its own ticket).

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
