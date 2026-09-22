# v3.1.4 backlog — what the 08-30 gap analysis found still standing

Source: a full-bundle gap analysis (2026-08-28 17:38 → 08-30 17:58, ~295 chat
completions) mapping every production symptom against v3.1.3 (`c396f9b`).
Verdict on the release itself: **nothing in the window justified delaying the
v3.1.3 deploy** — the dominant failure clusters all map onto its fixes, and no
degeneration recurred in ~48 h under `min_p 0.05` (0 detector fires).

**Privacy:** counts and structure only; conversation ids redacted to 4 hex.

---

## N1 · Interior empty-assistant turns still poison requests — S1, first fix of the line

`_repair_template_invalid_tail` drops empty assistant turns only while they
are **last** (`msgs[-1]`). Production demonstrated the sibling shape the
repair misses, conclusively, on 08-30 06:40–06:42:

    06:40:42  stream cancelled at 0 chars (msgs=312)
              → OpenWebUI stores an EMPTY assistant turn
    user types a NEW message (not regenerate)
    06:41:06  payload is user-final, empty turn now INTERIOR (msgs=314)
              → mistral template rejects the whole request:
                "Invalid assistant message: role='assistant' content=''"
    06:42:06  user recovered by manually deleting messages (msgs back to 312)

4 such rejections in the window (~2/day), each a dead turn with no reply, no
memory write, no retry, HTTP 200 already committed. The repair's docstring
premise ("comes back as the final message") describes the regenerate flow
only. This is the fix-one-site-miss-the-sibling class, instance fifteen.

**Fix shape:** space-fill (NOT drop — dropping breaks user/assistant
alternation) any interior assistant message whose string content is empty,
in the same pass as the tail repair. The lone-empty commit `73eb22c` (first
commit of this branch) already verified against vLLM's own template stack
that whitespace content is accepted where empty content is refused.

## N2 · The truncated streams are CLIENT-SIDE CANCELLATIONS — root cause found

The 29 `stream ended without completion` events match OpenWebUI's
`middleware:response_handler - Task was cancelled!` warnings **1:1 by count
and timestamp** (OpenWebUI 2–3 ms earlier every time — the cancel propagates
downward). Discriminators: zero vLLM aborts at those times, only 2
`ServerDisconnected` in the whole window (both during compactor restarts),
and irregular durations (0–20,386 chars) rather than a proxy's fixed
wall-time. High confidence it is client-origin; medium on stop-button vs.
tab-close specifically.

The plausible driver is **latency**: reply p50 95 s, p90 138 s, max 583 s
over 85 measured turns (~20 tok/s on the A40, fp8-via-Marlin) — she stops
waiting. Each cancellation also skips the memory tail, so the assistant has
no memory of ~18% of exchanges, invisibly.

**Direction:** this is a latency/UX problem, not a stream-handling bug.
Reduce time-to-cancel pressure (see N3's GPU contention) and consider
memorizing the partial reply above some length threshold instead of
skipping the tail wholesale (a 12k-char cancelled reply is not "no
exchange"). Settling stop-vs-reload: one day of OpenWebUI DEBUG logging on
the chat-stop endpoint.

## N3 · The fact store is a revolving door, and dedup is a treadmill

Whole window: **5,341 facts extracted, 3,714 evicted (70%)** — each main-conv
turn adds 15–22 facts and immediately LRU-evicts a similar number to stay
under `COMPACTOR_MAX_FACTS_TOKENS` (default 1500). Dedup: 234 passes, 2,024
LLM calls, 55 merges (2.7% yield); deferred clusters grew 4 → 23.

Combined with extraction this is **~8 background 24B generations per user
turn on the same GPU her replies stream from** — a direct contributor to
N2's latency.

**Interaction with the deploy env:** `FACTS_EXTRACTION_MAX_TOKENS` 256→1024
removes the 99 observed truncations (good) but raises facts-per-turn into an
unchanged 1500-token store — churn gets worse unless
`COMPACTOR_MAX_FACTS_TOKENS` rises with it. Sizing evidence to gather:
fact-survival half-life from the archive sidecar for conv=4214….

## N4 · OpenWebUI task traffic is treated as a first-turn conversation

> **STATUS 2026-09-03 — half of this shipped; the fix direction below is out
> of date.** The INJECTION half is done: `_has_conversational_history`
> (main.py:2710) gates `INJECTION_NO_HISTORY_FRACTION` (0.125 against 0.5),
> and `COMPACTOR_INJECTION_NO_HISTORY_FRACTION=0` disables injection for task
> traffic with no code change. The EXTRACTION half is not. `has_history` is
> computed once at main.py:4850 and reaches only the budget fraction and a log
> line; neither `_run_memory_tail` call site (main.py:5228, 5350) is told, so
> task traffic is still fact-extracted, indexed and deduped. One
> classification, two consumers, wired to one of them. Queued as A2 in
> `V318_PLAN.md`.

conv=d5a7… (stable hash, msgs=2, fires ~90 s after every main-conv turn) is
OpenWebUI's title/tag/follow-up generation. It receives 79–95 injected facts
per request, is fact-extracted, indexed, and deduped (a second treadmill),
and its prompt grows with the conversation: 87 `injected memory over budget`
warnings, and in the fresh window it **outgrew the context window** — two
hard-budget FAILED errors (21,825 and 19,910 tokens) forwarded and 400'd.
Also on record: 08-28 20:20, a task request at 32,801 tokens against the
entire 32,768 window.

**Fix direction (mostly config, not code):** set a separate task model in
OpenWebUI admin settings, or route tasks past the compactor; in code, the
compactor already *detects* the shape (its own log line says "task traffic
or a first turn") — it could stop injecting memory into and extracting
memory from requests it has already classified as tasks.

## N5 · Backup cadence stretches under restarts — one backup in 32.5 h

After the boot-time skip, backup.py sleeps the full 24 h interval **from
boot**, so each restart pushes the next backup out; nightly restarts are
routine, worst case ~36 h RPO. One-line fix: sleep `interval − newest_age`,
not `interval`.

## N6 · Housekeeping

- 16,741 stderr lines in 48 h of transformers' `apply_chat_template(...,
  tokenize=False)` warning (main.py local-count path) — add a warnings filter.
- TTS voice listing 404s (`/v1/audio/voices` unimplemented in the piper
  sidecar) — synthesis itself works.
- Store pollution: `CLONE_CONV_ID_HERE` (a runbook placeholder executed
  literally), 17 `__selftest_oneshot_*`, ~75 `itest-*` — 129 "conversations"
  in health stats for ~26 real ones; inflates backups and stats.
- Formulaic-responses complaint: **not determinable from logs** — no reply
  text, no sampling params logged. Needs the model's params row in webui.db
  plus reply texts from two backup archives straddling the temperature
  change.

---

## F1 · The facts workstream — decouple the store from the injection

Decided 2026-08-30, from N3's numbers. The design flaw under the revolving
door is that `COMPACTOR_MAX_FACTS_TOKENS` (1500) is one knob doing two jobs:
the STORE cap (how much she can remember) and the INJECTION size (all ~80
active facts go into every prompt). Because everything is injected every
turn, everything is touched every turn, `last_used` is meaningless, and LRU
degenerates to FIFO — the 70% churn selects for nothing. Four parts:

1. **Top-K relevance injection.** Rank active facts against the user's turn
   with the SAME bge-small query embedding retrieval already computes (CPU,
   milliseconds, zero GPU) and inject the top K (~300-400 tokens) instead of
   the whole store (~1,400). Side effect, and the point: only injected facts
   get touched, so LRU starts selecting for facts that keep mattering.
2. **A pinned always-inject tier.** Identity-tier facts (who she is, who the
   owner is to her, standing preferences) bypass ranking. Pure top-K can
   drop "her name is X" on a turn about dinner — that is the "she forgot me"
   failure wearing a relevance-scoring hat.
3. **Raise the store cap once injection is decoupled** (4,000-6,000 tokens;
   eviction already archives safely, so the cap was only ever about
   injection economics).
4. **Gate the dedup treadmill** — the single largest discretionary GPU spend
   in the logs (2,024 LLM calls for 55 merges, 2.7% yield, on the same GPU
   her replies stream from). Run the LLM pass only on clusters that gained a
   member since last pass; embedding-similarity screening first. This is
   likely a bigger latency win than anything else on this list, and latency
   is N2's confirmed root cause.

Explicitly rejected: LLM-driven fact COMPACTION (summarizing facts into
profile paragraphs). Dedup's 2.7% merge yield is evidence her facts are
distinct-but-related, not redundant; compaction would spend scarce GPU to
destroy the granularity that makes the atomic layer useful, duplicating the
summary hierarchy's job — which N4's data shows is keeping pace fine.

Interim, already recommended for deploy day: `COMPACTOR_MAX_FACTS_TOKENS`
~3000 so the 1024 extraction cap does not accelerate churn into a
fixed-size store.

---

## F2 · The prompt was asking for the formulaic replies — SHIPPED in v3.1.5

Reported by the user 2026-08-31 ("just a little too repetitive"), against the
bundle `…144610Z`. **Not** a return of the 08-29 degeneration: zero drift
fires and zero repeated-run fires across the whole window, so `min_p 0.05`
is holding and this is a different failure — stylistic sameness, and ours
rather than the model's.

**What the bundle showed.** Median **91 fact bullets injected per turn**
(p90 103, max 179), plus up to 1500 tokens of the model's own verbatim past
replies from RAG, plus the hierarchical summary, plus the last four turns.
And all four injected block headers asked, in one wording or another, for
CONSISTENCY:

    facts      "…established earlier, maintain consistency with these"
    retrieval  "…use them for continuity and exact recall"
    summary    "…use them for continuity."
    persona    "…the primary identity and voice you should maintain"

Three of those four had no business asking. Shown several thousand tokens of
its own prior phrasing under an instruction to be consistent, the model was
repeating itself because the prompt requested it. The memory system that
fixed forgetting is the same system that caused the sameness.

### The work

1. **Each block claims exactly one kind of authority** — persona owns
   identity and VOICE and is now the only block that does; facts own what is
   TRUE; retrieval owns what was SAID; summary owns what HAPPENED. Full
   rationale at `persona.py`'s `_PERSONA_BLOCK_HEADER`; the other three
   headers point at it.
2. **Positive wording throughout.** The 08-29 lesson applies directly:
   naming an unwanted output puts it in context at high attention weight
   (`DO NOT USE BOX-DRAWING CHARACTERS` sat in the prompt for a day and drew
   1,710 of them). These headers say what each block is FOR and leave the
   phrasing free — "the wording is yours", "say the next thing in your own
   words" — rather than prohibiting repetition by name.
3. **`COMPACTOR_INJECT_FACTS_TOKENS` default 800 → 400.** F1 part 1
   specified 300–400 and `main.py`'s call site has documented the default as
   400 since it landed; 800 was a v3.1.4 review value the surrounding
   comments were never updated to match, so the code and its own
   documentation had disagreed since.
4. **`test_facts_wiring.py` — closes the open F1 gate finding.** Halving the
   budget is only safe because `main.py` passes `query_text` at both call
   sites, so the surviving ~26 facts rotate with the topic. Without that
   wiring the change is *strictly worse* than 800 — a fixed
   most-recently-used prefix, frozen forever — and until now deleting
   `query_text=last_user_text` left the entire suite green. Mutation-tested:
   three separate reverts (drop it, pass `""`, drop the `_async_tail` twin),
   all three caught by the new file, all three passing `test_facts_injection`
   alone.
5. **`test_block_headers.py`** pins all four header strings so prompt text
   cannot change without a reviewer seeing it, plus budget bounds so a
   future header cannot quietly eat the injection budget. Six mutations, all
   caught.

### Three fixtures were sized against the old header lengths

Worth recording, because all three failed in a way that pointed at the wrong
thing. `test_budget_guard`'s wide-budget case was passing on **~3% headroom**
(426 against 400 once the headers grew) and failed as "facts injected too",
which reads like an injection-budget regression. `test_retrieval`'s teeth
check was **19 characters** inside a 6,000-character budget. Both now carry
real headroom, and the budget-guard assertion names header growth as a cause
so the next person is not sent to the wrong file. The two token pins
(273 → 284, 1463 → 1452) were re-pinned deliberately, not loosened — their
own comments already document the same move for v3.1.3.

### Not done, and deliberately

`COMPACTOR_RAG_TOP_K` left at 5 (owner's call — retrieval is the strongest
phrase-copying feed, so it stays the first knob to reach for if this
persists).

### Sampling, as actually deployed — `repetition_penalty 1.1`

Sampling is owner-side in OpenWebUI. Recording what is SET, not what was
recommended, because this file disagreeing with production is the same
failure as `IMAGE_TOKEN_ESTIMATE`'s stale comment and the 800-vs-400
injection default: the next person reads the recommendation and believes it.

Late on 2026-08-31 a reply ended in a run of repeating tokens. The owner set
`repetition_penalty 1.1` and it resolved. That knob is live.

This document previously advised against it, and that advice was narrower
than it read. The mechanism is real — vLLM's `repetition_penalty` covers
PROMPT tokens as well as output, unlike `presence_penalty`/
`frequency_penalty`, which are output-only — so it penalises her for reusing
words from her own injected memory, names included. But 1.1 is mild, and
end-of-reply token loops are exactly what it is for.

Neither knob risks the 08-29 drift any more: vLLM applies penalties to the
logits BEFORE `min_p` filters, so the probability mass a penalty pushes into
the tail is clipped before sampling. `min_p 0.05` is what makes penalties
usable here at all.

WHAT TO WATCH, and it will not look like a sampling problem: name and fact
AVOIDANCE. If the penalty bites, she paraphrases around established facts
rather than stating them, which reads as vagueness. The last bundle showed
**183 facts injected per turn** — a large prompt surface for a
prompt-covering penalty to act across.

RE-TEST AFTER v3.1.5 DEPLOYS. F2 cuts the injected block from ~183 facts to
~26 and stops the headers asking for consistency, shrinking that surface
roughly sevenfold. 1.1 was tuned against the large-prompt regime; it should
not be assumed correct in the small one. If it needs replacing, the
output-only `presence_penalty` targets the same end-of-reply looping without
reaching into the prompt at all.

### Still open

No measurement. "A little too repetitive" is not a number, so nothing here
can be confirmed to have helped. The drift detector is the proof that this
approach works; the equivalent is n-gram overlap between consecutive replies
plus a frequency count of opening phrases over the last N replies, which is
where "every reply starts the same way" shows up first and most sharply.

---

## C1 · conv_id is unstable against system-prompt edits — SHIPPED in v3.1.5

Found the hard way, 2026-08-30. The hash fallback is
`sha256(system|||first_user[:512])`, so the SYSTEM PROMPT is part of the
conversation's identity. Editing it gives a live conversation a brand-new
conv_id and forks its memory: facts, episodic embeddings and summaries all
keep accumulating correctly, under an id nothing else references.

Observed: a prompt edit at ~19:08 forked a ~400-turn conversation. The old
id kept 106 facts and ~85 indexed exchanges; the new one carried on and
re-derived its own summary hierarchy (reaching turn 411 in three hours,
because the client resends the full array). Nothing was lost - both halves
were intact on disk - but there was no way to put them back together, and
nothing announced that it had happened.

`source=hash` means OpenWebUI is sending neither `X-Conversation-Id` nor
`metadata.chat_id`. The bundled Function filter that supplies the latter was
never installed.

Shipped in v3.1.5:

1. **Fork detection** - a long conversation resolving to a hash-derived id
   with NO stored state logs a WARNING naming the likely cause, the sibling
   id to look for, and the merge command. Every individual signal looked
   healthy during the real incident; only the identity moved, and nothing
   said so.
2. **`portability.merge_conversation`** + `POST
   /admin/conversations/{src}/merge-into/{dst}`, dry-run by default. Merges
   facts and episodic exchanges; deliberately NOT summaries (dst re-derived
   its own over the same history, so merging would double-count). Source is
   read-only.

STILL OPEN, and the actual permanent fix:

- **Install the OpenWebUI Function filter** so `metadata.chat_id` is sent.
  Then prompt edits are free forever. Note this ALSO changes ids once for
  existing conversations - which is exactly why the merge tool shipped
  first.
- **Do NOT "fix" the hash by dropping the system prompt from it.** That
  would silently re-fork every existing hash-derived conversation on
  upgrade: the same bug, shipped as a fix. If the fallback is ever changed,
  it needs an alias/migration path, not a new formula.

---

## D1 · Get `webui.db` off the network filesystem, and snapshot hourly

**Incident, 2026-08-31 02:17-02:41.** OpenWebUI went "no backend"; every
query — including plain SELECTs — returned `sqlite3.OperationalError: disk
I/O error`, 1,819 of them, and every write attempt reported "attempt to
write a readonly database".

Root cause, from the file listing: a **4.8 MB `webui.db-journal` timestamped
02:17**, two minutes before the errors began. RunPod's MooseFS volume
(`mfs#ca-mtl-1.runpod.net`, mounted at `/data`) dropped I/O while OpenWebUI
was mid-transaction. SQLite then tried to roll that journal back on every
subsequent open; rolling back requires WRITING; the write failed; SQLite
reported readonly. **The database was never corrupt** — it was stuck
mid-recovery on a filesystem that would not let it finish. "readonly" was
SQLite protecting the file, not failing.

Recovery (worked, nothing lost): stop OpenWebUI, copy `webui.db` AND its
journal together to local disk, open there so SQLite can complete the
rollback, `PRAGMA integrity_check` → `ok`, 31 chats, 2 users, VACUUM, copy
back. **The journal and its database are a matched pair**: deleting a hot
journal turns a recoverable file into a corrupt one, and leaving a stale
journal beside a *replaced* database corrupts it too. Renaming both is what
makes the swap safe.

Blast radius was one file. The compactor's own storage stayed green
throughout — 2,080 facts, 749 indexed exchanges, `unreadable: {facts: 0,
episodic: 0, summaries: 0}`. That contrast is the finding: **many small JSON
writes survived the same event that broke one large, continuously-written
SQLite file.** SQLite on a network filesystem is a known-fragile pairing,
and this is what it looks like.

### The work

1. **Move `webui.db` off MooseFS.** Run it on the pod's local overlay (20 GB,
   2% used) and sync to `/data` on a timer. The overlay is not persistent
   across pod recreation, so the sync cadence becomes the RPO — which is why
   item 2 matters. The strategic answer remains the Postgres state home
   already on the roadmap; it is built for exactly this and removes the
   class rather than shortening the window.
2. **Hourly snapshots.** `COMPACTOR_BACKUP_INTERVAL_HOURS=1` needs no code —
   the cadence is already env-tunable, chat history is already in scope
   (`COMPACTOR_BACKUP_WEBUI_DB`), and the snapshot is already live-SQLite-safe.
   Retention already tiers (`RETAIN=7` newest, `RETAIN_DAYS=14`), so hourly
   yields ~14 days of hourly archives. **The code item is the shape, not the
   cadence**: an hourly FULL tar re-writes chromadb and every facts file
   onto the same fragile volume 24x a day. Split it — an hourly light
   snapshot of `webui.db` alone, plus the existing daily full archive.
3. **Backup must not go silent when the source is sick.** During this
   incident the backup itself failed (`backup failed: OperationalError:
   attempt to write a readonly database`) and produced NOTHING — at exactly
   the moment an archive was most wanted. When the SQLite-safe path fails,
   fall back to a raw file copy of the database and its journal, label the
   archive as degraded, and alert. A byte copy of a sick database is
   forensically valuable; no archive at all is not.

### Also worth noting

- The five `webui.db.bak-*` files (2026-08-24, ~20 MB each against today's
  41 MB) had been riding along inside `/data/openwebui`, i.e. inside every
  one of the 17 backup archives. Removed. Keep `/data/openwebui` holding
  exactly one database — that directory gets read during incidents, at 2am.
- Version note: v3.1.5 is already tagged and built, so this lands as v3.1.6
  unless the tag is moved.

---

## Deploy-day notes carried from the analysis

1. The pod is already half-on the new env: `GENERATION_RESERVE=12000` went
   live at the 08-30 17:00 restart (guard limit observed at 20,768).
   `target_tokens=999999` and the 256 extraction cap were still in effect at
   capture.
2. After the v3.1.3 deploy, request-path compaction will (correctly) *skip*
   conv=4214…'s ~610k-token backlog. Drain it deliberately, once:
   `POST /admin/conversations/<id>/compact` (dry-run first).
3. Consider raising `COMPACTOR_MAX_FACTS_TOKENS` alongside the extraction
   cap (N3) — or expect louder churn.
4. The background L1 hierarchy kept pace with an all-night 372-message
   session (watermark lag ~15 turns) — the memory hierarchy works; the live
   window is the constraint.

---

# v3.1.7 — what the 2026-09-02 log sweep found

Bundle `zla-bundle-20260902T150758Z` (window 08-28 17:38 → 09-02 15:07) plus
`zions-backup-20260902-150135`. Counts only below; no conversation content and
no conversation ids — this repo is public.

The pod is on **v3.1.4 plus hot-fixes**. That matters for reading every number
here: several of these are fixed in v3.1.6/v3.1.7 and are still happening
because that build has not shipped.

## L1 · The empty-assistant 400 chain is live, and growing

The 2026-08-28 failure shape, running on nearly every turn of the long
conversation. One representative sequence, 09-02, eight seconds end to end:

```
00:00:56  /tokenize 400 "Invalid assistant message: role='assistant' content=''"
00:00:57  compaction skipped: 723 turns need 102 summarization calls, over the 4-call cap
00:01:03  token scale unavailable — budgeting this payload at 1,292,552 tokens UNCORRECTED
00:01:04  /tokenize is answering again after 2 consecutive failure(s) over 8s
00:01:04  hard budget enforced: 1,292,552 -> 18,710 tokens; dropped 724 old turn(s)
```

Per day, `Invalid assistant message`: 0, 8, 2, 19, **101**, **44** across
08-28..09-02 (09-02 is a partial day, cut at 15:07). vLLM's own log carries
343 matching `ValueError`s. `token scale unavailable` tracks it almost exactly
(0/4/1/18/99/44), because it IS the same event seen one layer up.

The turn count reached **807** by 14:58 on 09-02.

**Cause:** `count_tokens_exact` measures the raw message list. v3.1.4 repairs
the tail we FORWARD and never the payload we MEASURE. v3.1.6 introduced
`_space_fill_empty_assistant` at both sites; v3.1.7 widened it to `None`, a
missing content key, `[]` and blank-text lists, and shared one predicate with
the forward path's pop.

**Action: none in code. Ship v3.1.6+.** This is the highest-value single
action available and it closes L1 and L4 together.

## L2 · `backfill.py` does not pass `conv_id` to the extractor — one line

139 fact-extraction failures in the window; **137 of them say
`conv=? (caller passed none)`**, all on 08-30, i.e. one backfill run. So the
run that lost the facts cannot be traced to the conversation that lost them.

Three call sites take `conv_id`. `main.py:3675` and `facts.py:1689` pass it.
`backfill.py:293` does not — and it HAS the value; it uses it in its own
WARNING two statements later. The recurring defect: a parameter added at two
of the three places that needed it.

`facts.py:1515` states the intent plainly: conv_id "is what makes a failure
attributable: without it the warning this used to emit named no conversation,
so a lost extraction could not be traced to the turn that lost it."

**Action:** pass `conv_id=conv_id` at `backfill.py:293`. One line, zero risk.
Add a test asserting all three call sites pass it.

## L3 · Inline compaction is entirely off on the long conversation

`compaction skipped` per day: 0, 0, 4, 22, **102**, **44**. By 14:58 on 09-02:
`807 turns need 119 summarization calls, over the 4-call per-request cap`.

This is the documented design, not a bug — the guard sheds and the L1/L2/L3
hierarchy is supposed to carry the older context. But it means every turn
sheds ~724 turns and the model sees roughly the newest handful plus the memory
blocks. The cause is upstream: OpenWebUI resends the full transcript and the
`max_turns` filter cap is not installed.

**Action:** install the OpenWebUI filter at `max_turns=60`, but only after the
merge, per the runbook in `pipelines/conversation_id_header.py`. Note C3 from
the v3.1.7 review first: `merge_conversation` silently un-pins facts, and
capping a conversation whose watermark lags its position strands the span
between them.

## L4 · Memory tail skipped 52 times in two days

09-01 and 09-02 only: 41 `stream ended without completion`, 10 `stream
truncated at the generation ceiling`, 1 `reply truncated at the generation
ceiling`. Each one is an exchange with no fact extraction, no episodic index
and no rollup.

**Action:** shipped in v3.1.7 (`trim_to_last_sentence` + `decide_memory_tail`);
undeployed. Corpus replay says ~7 of every 17 cut replies become memorable.

## L5 · The hard budget cannot fit at all — 43 times, 6 on 09-02

```
hard budget FAILED to fit: 26,587 -> 23,442 tokens (limit 20,768);
dropped 0 old turn(s), trimmed 6 injected block(s), dropped 1 entirely
— still 2,674 token(s) over. Nothing injected remains.
```

`dropped 0 old turns` with everything injected stripped means a single turn is
larger than the whole budget, and the newest turn is never dropped. The
request still goes out and usually succeeds (23,442 fits vLLM's real 32,768 —
the 20,768 limit is `MAX_MODEL_LEN` minus the 12,000 generation reserve), so
this is not a crash. It is a **silent total memory loss for that turn**: she
gets no persona, no facts, no retrieval, no summary.

**Action:** not yet diagnosed. Worth finding what makes a single turn 23k
tokens — an image, or a long paste. Consider whether the guard should say so
distinctly rather than logging the same ERROR as an ordinary overflow.

## L6 · `/health/full` reported `"status":"ok"` through all of the above

`health.json` at capture: `status ok`, `status_reasons []`, tokenize
`ok:true consecutive_failures:0`, background work `shed:0`. Because the
tokenize streak had recovered by the moment of capture, and because nothing
counts a skipped memory tail on this build.

Confirms the v3.1.7 `memory_tail` work is the right shape — the block is
absent here, which is itself how you can tell the pod predates it.

## Not actionable, recorded so it is not re-investigated

- **21,304 `tokenize=False` nag lines** (4.4 MB of `compactor-error.log`). The
  `_DropChatTemplateNag` filter IS present in v3.1.4. The lines stop at file
  line 21,370 and the current process is quiet, so this is historical bloat.
- **228 vLLM `maximum context length` rejections.** Prompts of 36,398 and
  59,407 tokens with 1,024 output — extraction/summarizer calls, clustering
  with the 08-30 backfill run, not the chat path.
- **Disk**: `/data` 79% used, 206 TB free. Overlay 3%. Fine.
- **Store**: 138 conversations, 2,180 facts, 1,093 indexed exchanges,
  0 unreadable.

---

# v3.1.7 — the four-perspective code review

Four reviewers on the same snapshot: **A** informed (incident history, the
recurring defect, the silent-skip census), **B** and **C** cold (repo and
stakes only, run independently so divergence between them measures
single-reviewer reliability), **D** blank ("look for issues"). Each was told
to reproduce before reporting and to plan fixes rather than apply them.

Status: A and B complete. C and D not yet run.

Findings live in the session scratchpad (`review-A-report.md`,
`review-A-supplement.md`, `cold-review-B/review-B-report.md`) which is
temporary; the substance is carried here so it survives.

## FIXED in this branch

### R1 · `tailhealth.py` was never added to the Dockerfile — found by B

`main.py` imports it at module scope and `health.py` imports it inside
`gather_health_full`; it appeared in none of the 23 `COPY compactor/*.py`
lines. The image could not build.

**Third occurrence of the identical defect.** v3.1.3 added `tokenhealth.py`
and v3.1.4 shipped without its COPY line — the compactor went FATAL on boot
in production, chat path down until an operator hot-copied the file into the
running container. v3.2's `dbselect.py` was caught in review. Each time the
missing line was for a module added in the same change that needed it.

The two existing BUILD GUARDs do catch it — but only during `docker build`,
i.e. after a tag is cut, on a GPU base image, twenty minutes into a layer
cache miss. The information needed is entirely static and lives in two files.

**Fixed:** the COPY line, plus `compactor/test_image_manifest.py`, which
parses the Dockerfile's COPY list and walks the import graph from the five
entry points the image executes (`main`, `selftest`, `backup`, `pgarchive`,
`dbselect`), asserting every reachable module is copied. Uses `ast.walk`
rather than module-scope imports only, because a function-scoped
`import tailhealth` is equally fatal — just later, on a request instead of at
boot. Mutation-tested: removing the COPY line is caught and names the chain
(`selftest -> health -> tailhealth`); a COPY for a deleted module is caught.

### R2 · One env typo killed the compactor at boot; `0` silently disabled the release's own signal — found by B

`float(os.environ.get(name, "300") or 300)` in both `tailhealth.py` and
`bgwork.py`. Two failures in one line, and both modules are imported at
main.py module scope, so both are boot failures:

* **Unparseable.** `or 300` rescues only the empty string. `30O` in
  runpod.env raises ValueError at import; the compactor never starts.
* **Non-positive.** `0` or a negative parses fine and then disables the
  signal: `skipped_recently` / `shed_recently` is `since <= window`, never
  true, so `/health/full` reports ok while the memory tail is skipping or
  background work is being shed. The exact regression this release exists to
  catch, switched off by a config line nobody would look at twice.

`main._env_float` already documents the right contract ("never a crash at
import time and never a silent zero") but neither module can import main.

**Fixed:** a local `_window_s()` in each, defaulting on unset/blank/
unparseable/non-positive, with tests in `test_tailhealth.py` [9] and
`test_bgwork.py`. Five mutations, all caught — one initially survived because
bgwork had no test for it, which is why that test exists now.

### R3 · The empty-assistant repair had two holes — found by A

`_repair_template_invalid_tail`'s pop tested emptiness with
`_message_text().strip()`, which joins text parts and ignores images, so an
assistant turn carrying only an image read as empty and was popped — the
image destroyed, silently — three lines before the fill carefully refused to
touch that shape. Two emptiness rules in one function; the laxer ran first.
`tokens._sanitize` separately raised `AttributeError` on non-string content,
which `tokens.count` swallowed, making tier 2 silently unavailable.

**Fixed:** one shared predicate `main.assistant_content_is_empty(content)`
used by the fill, the pop, and (restated against reduced text)
`tokens._sanitize`. Six mutations caught.

## OPEN — highest value first

### R4 · The runbook's "commit the merge" command is a second dry run

`pipelines/conversation_id_header.py` install step 8 tells the operator to
run the merge with `?dry_run=false`. `admin_merge` reads `dry_run` from the
JSON body only; nothing in main.py reads `query_params`. The commit returns
HTTP 200 with plausible counts and changes nothing.

This is the step whose entire purpose is to un-fork her memory, and it is on
the path that must run before the `max_turns` cap is enabled (L3 above).
Fix: correct the runbook to send a JSON body, AND make `admin_merge` accept
the query form, which is what an operator reaches for under stress. No test
asserts the documented curl commands work.

### R5 · `merge_conversation` silently un-pins facts — found by A

Facts are unioned on `_fact_key` with the destination winning on collision,
and there is no pin union: a pinned source fact merged into a conversation
holding an unpinned copy comes out unpinned. `_fact_key` is casefolded and
whitespace-collapsed, so this fires on non-identical pairs too. It is on the
documented install path (step 8), so following our own runbook un-pins
identity facts. `dedup._merge_metadata` already does the pin union; the merge
path is its missed sibling. `/retire` step 3 has the same defect while step 2
was fixed. No test asserts pin survival across either.

### R6 · Three suites report PASS without running their checks

`test_tokenizer_contract.py` and `test_soak_conversation.py` exit 0 on a
whole-script skip when the docker fixture on localhost:18000 is absent;
`test_tokens.py` skips its real-tokenizer section and still prints "All
tokens tests passed". The tokenizer-contract skip text says so itself:
"Skipping it means the budget code is currently only covered by char/4
assertions — the estimator that took production down."

Every "all suites pass" in this branch, including every one reported during
this review, therefore excludes tokenizer-contract behaviour. Fix: exit with
a distinct non-zero code unless an env var opts into skipping, and make the
last stdout line say SKIPPED so a grep-based runner cannot read it as a pass.

### R7 · `SseAccumulator` corrupts multibyte characters split across reads

Decoding each chunk independently with `errors="replace"` turns a UTF-8
sequence straddling two TCP reads into U+FFFD, and the text is then stored as
a clean `stored` turn. Found independently by A and B. The `holed` flag added
for exactly this cannot fire on bytes input, because `errors="replace"` never
raises — so it only ever flags a programming error. Fix: an incremental
decoder held on the instance; set the holed flag in the JSON-drop branch when
the dropped payload carried a content key.

### R8 · `tailhealth` counts `stored` before the work runs — found by A

`_run_memory_tail` counts on `decide_memory_tail`'s verdict, then fires
`_async_tail`, which has three exits that store nothing: `degrade.guard`
false (disk pressure), extraction disabled (which also skips the rollup, a
dependency the docstring says does not exist), and empty `last_user_text`
(a bare return with no log at all). So `/health/full` can report a healthy
tail for an exchange that never reached memory. Fix: evaluate those
conditions in `_run_memory_tail` before counting, or have `_async_tail`
return an outcome the pool records.

### R9 · `reply_is_degenerate` rejects legitimate replies

A 50+ item list trips the list-run branch: a 66-item numbered list, 55 Bible
books as bullets, 60 checkbox items. "List the 66 books of the Bible" is
permanently unmemorable and, via `_redact_degenerate_turns`, replaced by a
placeholder in every future summary. B found the same rule rejecting a
coherent one-paragraph reply of short sentences that passes when the same
text has newlines. Also: tilde fences are not recognised where backtick
fences are, and 500 items of 35 chars evade both branches entirely.

### R10 · `/compact` mislabels every chunk at `len(recon) == turns_seen`

The guard is `<` where its own comment argues for `<=`, and equality is the
healthy case. Chunks are then labelled with spans they do not contain and
some turns are covered by nothing — verbatim the outcome the guard's comment
calls "worse than no chunk, because nothing downstream can tell".

### R11 · Four mutations survive the new memory-tail tests

Recorded by B; not yet enumerated here. Re-run before shipping.

## UNRESOLVED CONTRADICTION — client disconnect

A's sub-agent and B disagree, and this matters because it is the mechanism
behind the 51 stopped replies that motivated the whole memory-tail change.

* **A's sub-agent:** scenarios A/B/C fine, but a fourth — the generator
  parked at `yield` waiting on a slow consumer — defers the memory tail to
  garbage collection and leaks the upstream socket. Also demonstrated that
  the `aclose()` in the `finally` sits in front of the only bookkeeping and
  is one `await` from swallowing the tail entirely.
* **B:** the tail fires in both cancellation placements; not a defect.
* **A's own final report** filed the whole area under "checked and found
  sound", omitting its sub-agent's fourth scenario.

Both ran on Windows/Proactor with plain asyncio. Production is Linux and
likely uvloop, and a server advertising ASGI spec_version 2.4 takes the
branch that lands in the deferred case every time. **Settle this on a
production-shaped stack before building on either answer.**

## Settled, no action

* The `pipelines/conversation_id_header.py` diff is **docstring-only** —
  every executable line byte-identical, and 17 differential inlet cases agree
  on the stamped id at `max_turns` 0 and 60. No memory-wipe risk in that
  change.
* Summarizer state files are forward-compatible: a pre-v3.1.4 file loads with
  chunks, watermark and rendering intact. Rolling BACK, however, strips
  `turns_seen` and `tail_fp` permanently, because the parking mechanism for
  unrecognised data covers chunk entries and not top-level keys.

## Reviewer C (cold, run independently of B)

C went at the summarizer's position tracking and the upgrade path — the area
A explicitly marked unsettled because its sub-agent never reported. Six
findings, all reproduced with harnesses; 30 mutations applied in throwaway
worktrees, **all 30 caught**.

**The meta-finding: three suites PIN the wrong behaviour.** Not gaps —
assertions that the defective behaviour is correct. Named below at R12, R14
and R16. A mutation sweep cannot find this class, because the tests agree
with the code; only a reviewer reading intent can.

### R12 · A pulled-down watermark makes the duplicate-label guard discard every new L1 chunk — silently

**Highest priority: this fires on the documented install path.** Do not
enable `max_turns` on any pod whose state file predates this build until it
is fixed.

`_observed_position` seeds from `max(turns_seen, last_summarized_turn)`. A
state file written by the parent commit under `max_turns` has the watermark
PULLED DOWN to the cap by the old `_reconcile_watermark`, while the L1 list
still holds chunks labelled up to the real position. The position restarts at
the cap, the next chunk is labelled `cap+1..cap+20`, and the new
duplicate-chunk check finds an old chunk with that exact label — logs a
WARNING, advances the watermark, returns True. For every chunk until the
position climbs past the highest old label.

Both documented cap values (100 and 60) are multiples of `L1_CHUNK_SIZE=20`,
so the labels collide exactly. The guard's own comment says a monotonic
position makes it unreachable; the upgrade path reaches it on the first
rollup.

Reproduced (33 old chunks to turn 660, watermark 100, capped 100-turn client,
30 new exchanges): `new L1 chunks=0 LLM calls=0`. Control with the watermark
at 660: `new L1 chunks=2 LLM calls=2`. It stays stuck for ~280 exchanges.

**PINNED WRONG:** `test_a_duplicate_span_is_not_stored_twice`
(test_summarizer.py:1365) builds exactly this shape — a chunk labelled 1-4
with `last_summarized_turn = 0`, described as "a position that went
backwards" — and asserts the span is skipped "without spending an LLM call to
prove it". It never checks the turns under those labels are the turns the
existing chunk summarized.

Fix: seed the position from the chunks too
(`max(turns_seen, last_summarized_turn, max(last_turn over l1+l2+l3))`), in
`_observed_position` and `admin_compact`. Make the guard non-destructive:
ERROR, and summarize with a relabelled span, or skip only after confirming
the existing chunk covers the same turns. A duplicate chunk is recoverable; a
silently dropped span is not.

### R13 · `/admin/compact` refuses for every real conversation — and it is the endpoint the new ERROR line points at

`turns_seen` counts every exchange including the ones `decide_memory_tail`
skipped and the ones the pool shed; the episodic store holds only indexed
ones. The guard requires `len(reconstruction) >= max(turns_seen, watermark)`,
so one un-indexed exchange anywhere is enough to 409. Given 63 skipped in one
window, that is every real conversation. The old guard compared against the
watermark, which lags in steps of 20 and tolerated small gaps.

Reproduced: 15 exchanges with exchange 7 skipped →
`turns_seen=30 watermark=20 reconstruction=28 → HTTP 409`; old guard would
have run.

C is careful here: the 409 is CORRECT for the offset arithmetic, since a
gapped reconstruction would misalign chunk text. This is a design gap, not a
wrong line — the endpoint and the summarizer disagree about what a transcript
is. It is also the only rebuild-from-store recovery path, and R12's ERROR
line sends the operator straight to it.

**PINNED WRONG:** test_admin_compact [3d] asserts the 409 as desired.

Fix: rebuild by SLOT using the episodic rows' `turn_index`, filling gaps with
explicit placeholder turns so length equals position and `window_offset` is
0. Keep the 409 only when the store's highest index is below the watermark.
Report the gap count in the plan JSON.

### R14 · The multibyte corruption, confirmed by all three reviewers

Same defect as R7. C adds the detail that the suite pins it: test
[7] (test_sse_accumulator.py:155-162) asserts `holed() == False` and "the
replacement character is what arrived" — a stub laxer than production needs.
C also found a malformed `data:` event dropping its content with
`holed() == False`, pinned at test_sse_accumulator.py:97.

Found independently by A, B and C. Fix as R7, plus rewrite test [7] to split
a character across chunks.

### R15 · The position anchor cannot align across an image-only turn

`_turn_fingerprints` hashes `_message_text(m)`, which is "" for a content
list with no text part, while the episodic store remembers that turn as
`[shared N image(s)]` — which is what `admin_compact` rebuilds. A mismatch in
the newest anchor slots is absorbed by the prefix walk, but a mismatch in
`anchor[0]` defeats every prefix: the position inflates by 2 per rollup and
persists, so `window_offset` subtracts 2 forever and two turns are summarized
twice.

Fix: fingerprint user turns as the store will remember them, or fingerprint
assistant turns only (identical on both sides).

### R16 · Anchorless capped upgrade leaves a permanent 2-turn hole

The "hold" when the window is bounded and no `tail_fp` exists is measured
against a window that already contains this exchange's two turns, and is
never repaid — so the position is permanently 2 behind, every chunk labelled
`a..b` summarizes `a+2..b+2`, and two turns are never summarized. The
docstring claims "at most one turn of latency, once". Only reachable when the
cap is on at the moment of upgrade.

### R17 · `_observed_position` hashes the whole history on the event loop, every turn, inside `conv_lock`

Measured 25 ms per call at 700 messages, linear in history, ~2-4 ms under a
cap. The codebase's own doctrine treats this shape as a shipped defect. Fix:
hash only the tail slots alignment can use, or use `run_in_threadpool` as the
redaction pass already does.

## What the reviewer spread says

* **R7/R14 (multibyte) found by A, B and C independently** — the most
  strongly confirmed finding in the review, and none of them was told to look
  there.
* **C alone found R12, R13, R15, R16, R17**, all in summarizer position
  tracking. That is precisely the area A's report listed as unsettled. Cold
  review did not have a ceiling here; it filled the informed reviewer's
  largest gap.
* **A alone found R5, R8, R10** (merge un-pinning, the tailhealth counting
  order, the `/compact` equality guard). The incident history was
  load-bearing for those.
* **B alone found R1, R2, R6** (the Dockerfile omission, the env-window boot
  failure, the skip-as-pass suites) — all release-mechanics defects that
  neither A nor C looked at.

Four reviewers, four largely disjoint find-sets, one triple overlap. The
overlap is a real bug; the disjointness is the argument for having run more
than one.

## Reviewer D (blank — "look for issues", no stakes, no history, no repo description)

D was told nothing: a path and one sentence. It confirmed three findings the
other reviewers had, and produced four nobody else did. Its baseline ran all
62 scripts green.

### R18 · Two byte-identical trailing exchanges stall the position under a cap — D only

`_align_new_turns` with `_ANCHOR_TURNS = 4`: when the last two exchanges hash
identically — two consecutive `_redact_degenerate_turns` placeholders, or
"ok"/"Sure." twice — the full anchor matches at the END of the new window,
`new` comes back 0, and the position does not advance. Under `max_turns` the
position is the only thing that advances, so a repeating tail never reopens
the L1 rollup gate.

That is the frozen-hierarchy failure this release exists to fix, reached by a
different route. Reproduced against the real `_turn_fingerprints`:
`w2 = w1 + [ok/Sure.]` gives `aligned -> 0 (truth: 2)`. No test covers it.

Note the interaction with R9/R19: a degenerate reply is replaced by a
placeholder, and two placeholders in a row are byte-identical. The
degeneracy rule and the anchor can starve the hierarchy together.

Fix: fall back to `_ASSUMED_NEW_TURNS` when the anchor tuple occurs more than
once in the window, hash the ordinal into the fingerprint, or widen the
anchor.

### R19 · The list rule ignores `DEGENERATE_MIN_CHARS` — D only

`reply_is_degenerate("- a\n" * 50)` fires at 200 characters, while
`DEGENERATE_MIN_CHARS` is 300 and the comment above
`MIN_MEMORABLE_TRIMMED_CHARS` calls 300 "the floor below which this codebase
already declines to judge a reply structurally". The structural block is not
gated on it. One assert to fix, or correct the comment.

### R20 · `_align_new_turns` prefers a long early prefix over a short late one, contrary to its docstring — D only

Anchor `[A,B,C,D]` against `[A,B,C,D,X,Y,A,B,C]` returns 5; the docstring's
"latest occurrence" rule gives 0. A mutation removing recency entirely IS
caught by the unit test, but the ordering conflict is not. D's own read is
that 5 is the true answer for a full-history client, so this is a
doc/implementation ambiguity that bites only with repeated short turns —
i.e. exactly the R18 shape. Fix: compute `new` for every match and return the
minimum; pin the case.

### R21 · An empty or system-only window adds 2 phantom turns and wipes the anchor — D only

`turns_seen=100` with `tail_fp=[a,b,c,d]` and `messages=[]` (or system-only)
gives position 102 and anchor `[]`. Fix: if the fingerprint list is empty,
return `prev` and leave `tail_fp` alone.

### R22 · `test_budget_guard.py` takes ~239 s and trips a 240 s ceiling — D only

Every other script finishes in ≤ 21 s. A runner with a 4-minute timeout
records it as a HANG when it is merely slow; D confirmed it completes twice.
Fix: stub the clock behind the backoff it waits on, or split the script.
Relevant to R6 — a test harness that cannot tell slow from hung is the same
class of problem as one that cannot tell skipped from passed.

### Confirmations from D

* **R7/R14 multibyte corruption — rated HIGH.** D adds the detail that the
  client receives the correct bytes (`yield chunk` forwards raw), so what is
  fact-extracted, embedded and rolled up is *not the text she read*, and
  nothing says so. Split at every one of 107 byte offsets: `holed=True` in 0
  cases. D also identifies test [7] as a laxer-than-reality stub — a split
  character is *incomplete*, not *invalid*, UTF-8.
* **R9/R13 Bible-list false positive** — reproduced end to end at 912 chars:
  `decide: False skipped_degenerate`, then `_redact_degenerate_turns`
  replaces the turn permanently. D offers three fixes, of which the best is
  requiring the run to reach the END of the reply, since a real runaway "ran
  to its own end" and the Bible reply closes in prose.
* **R13 `/admin/compact` refusal** — same mechanism C found, independently.

### An unexplained edit, checked and cleared

D reported a transient uncommitted change to `main.py` during its review
("SseAccumulator gains an incremental utf-8 decoder") that it attributed to
nobody. Verified afterwards: the primary working tree contains no such edit,
and all four review copies are `git status` clean. Most likely one of D's own
sub-agents applied the fix against instructions and reverted it — which is
itself mild evidence for the fix, since it reached the same answer
unprompted. Recorded because an unexplained write during a review is worth
being able to rule out later.

## Final tally across the four reviewers

| finding | A | B | C | D |
|---|---|---|---|---|
| Multibyte SSE corruption (R7) | yes | yes | yes | yes |
| `/admin/compact` refuses (R13) | partial | — | yes | yes |
| Degeneracy false positive (R9) | yes | yes | yes | yes |
| Summarizer position tracking | — | — | R12, R15-R17 | R18, R20, R21 |
| Merge/retire un-pinning (R5) | yes | — | — | — |
| tailhealth counting order (R8) | yes | — | — | — |
| Dockerfile omission (R1) | — | yes | — | — |
| Env-window boot failure (R2) | — | yes | — | — |
| Skip-as-pass suites (R6) | — | yes | — | — |

Two findings were unanimous. Everything else was found by one or two
reviewers, and each of the four contributed something no other did —
including D, which was told nothing at all about what the software is or who
depends on it. The plan predicted that D-only findings would be the most
informative outcome; there are four of them, all in areas the framed
reviewers walked past.

The practical read: a single review of this codebase, however well briefed,
would have shipped most of these.

## Reviewer D's sub-agents, reported after D's own report closed

Two sub-agents returned substantial work AFTER D had already delivered, so
none of it is in `review-D-report.md`. Recorded here because it contains the
most serious single finding of the whole review.

### R23 · One client turn is silently lost the moment the `max_turns` valve engages — HIGH

**This is the exact scenario v3.1.4 exists for, and it loses a turn.**

`_observed_position`'s first regime is `if n > prev: position = n`, documented
as "the array length IS the position". That is true only while the window is
UNBOUNDED. A sliding cap does not snap from full history to short window in
one step — it engages gradually, and on the first request where it bites, `n`
is still greater than `prev` while already being SHORTER than the true
position:

* cap 100, exchange 51: client stores 100 turns, appends the new user turn
  (101), the valve trims to the last 100, the compactor appends the assistant
  turn it just produced → `n = 101`
* recorded `prev = 100`, so `n > prev` → `position = 101`
* the truth is 102. One turn of position is swallowed, permanently.

The anchor was present and would have answered "2 new turns"; the `n > prev`
branch never consults it.

Reproduced end to end against the real `maybe_rollup` with a recording fake
vLLM, 64 exchanges at cap 100, window built exactly as main.py plus the
pipeline build it:

```
client turns 1..121 that no L1 chunk summarized: [101]
label 101-120  -> real client turns 102-121
```

Two harms, both silent — no WARNING, no ERROR, and the one INFO line that is
emitted describes the healthy shape:

1. Client turn 101 is summarized by no tier, ever. Nothing downstream can
   tell and no counter exists for it.
2. The chunk LABELLED 101-120 actually summarizes 102-121 — precisely the
   failure `_do_l1_rollup`'s own comment calls "worse than summarizing
   nothing because nothing downstream can tell", and the failure its sibling
   test was written to prevent, but only for the `pos_first < 1` path.

**Test gap:** `test_switching_the_cap_on_re_summarizes_nothing` models the
transition as a full history followed immediately by a short window
(`n <= prev`), which lands in the anchor branch. Nothing exercises
`prev < n < true position`, which is what a sliding window actually produces.
The whole suite is green while the bug reproduces.

Fix: in the `n > prev` branch, still align the anchor when one exists and
take `max(n, prev + aligned)` — the anchor is computed on every call already,
so this costs nothing. Or use "the window's oldest turn is not turn 1" as the
discriminator instead of `n > prev`.

**Together with R12, this is the second independent reason not to enable
`max_turns` until the summarizer's position work is fixed.**

### R24 · The fragment-line rule counts an abbreviation's dot as a sentence end — HIGH

`reply_is_degenerate`'s structural block counts sentence breaks with bare
string counts: `line.count(". ") + line.count("! ") + ...`. So `Dr. `,
`Mrs. `, `Rev. `, `St. `, `9 a.m. ` each register as a sentence break, the
computed mean fragment length collapses, and ordinary prose trips the `<= 40`
limit.

A single paragraph of ordinary narrative prose, 1,589 characters, real mean
sentence length 93.5 characters — squarely inside what the code's own comment
calls healthy — is flagged:

```
rule's break count: 42   rule's mean: 37.0
VERDICT: an unbroken line of 1589 characters made of 43 fragments averaging 37 characters
```

**The same commit adds `_SENTENCE_ABBREVIATIONS` containing exactly
`mr mrs dr prof st rev fr a.m p.m` for precisely this reason.** Two new pieces
of code in one delta disagree about what a sentence end is.

The consequence is not a skipped write: `_redact_degenerate_turns` replaces
the reply with a placeholder in every future rollup, backfill and admin
compact, permanently.

Fix: count breaks with a regex that reuses `_SENTENCE_ABBREVIATIONS` and the
single-initial rule already written for `trim_to_last_sentence`, so the
detector and the trimmer share one definition of a sentence end — the same
"one rule, one function" argument `assistant_content_is_empty` is built on,
applied to the other duplicated rule in this delta.

### R25 · An em dash before an abbreviation bypasses the stoplist, storing fragments

`_WORD_LEAD_CHARS` contains no dash of any kind, so after `—`, `–` or `-` the
`whole_word` test fails and both the abbreviation stoplist and the
single-initial rule are skipped:

```
'Something ended properly. Then—i.e. a fragment that never fin'
   -> 'Something ended properly. Then—i.e.'
'Written by the author—J. R. R. Tolkien and oth'   -> 'Written by the author—J.'
```

The first is the damaging shape: a clean complete sentence was available and
the trim ran PAST it to end on an abbreviation — turning a correct store into
a fragment store, which is exactly what the docstring says must never happen.

The test suite covers the stoplist only with a space or start-of-text lead, so
it tests the set for being too permissive and never for being too narrow. A
mutation making `whole_word` always True IS caught; making it always False is
not.

Fix: add the dashes to `_WORD_LEAD_CHARS`, or invert the test to "the
preceding character is not alphanumeric", which is what the rule means.

### R26 · vLLM dying mid-stream skips the memory tail without counting or naming it

The streaming site guards on `if conv_id and not vllm_failed`. `vllm_failed`
is set when vLLM drops the connection PART WAY THROUGH a reply the user has
already read. At that point the accumulator holds real prose, but
`decide_memory_tail` is never called, `tailhealth.note` is never called, and
no line containing the grep phrase "skipping memory tail" is emitted:

```
decide_memory_tail called: 0 times
tailhealth snapshot changed: False
log lines with 'memory tail': []
bytes the client got contained the real prose: True
```

That is the shape this release argues against in its own comment: "a skip
that is only a log line is the defect this exists to close" — here it is not
even a log line naming the conversation.

Fix: call `_run_memory_tail` with `finished=False` on that branch (the trim
path already handles it), or add an explicit tailhealth outcome so the skip is
counted and greppable.

### R27 · An empty reply flips /health/full to degraded for five minutes

A reply with no text is `SKIPPED_EMPTY`, which `tailhealth.note` counts like
any other skip, which health.py turns into a degrade reason reading "1
reply(ies) not memorized … New memory is not being written". But there was
nothing to memorize — it is the one skip label that carries no loss. Reachable
whenever the user hits Stop before the first token, and the release's own
figure is that 51 of 63 skips in one window were manual stops.

Fix: give tailhealth a `LOSSY_SKIP_OUTCOMES` set and key the degrade decision
off that, or add `skipped_recently_lossy` alongside the existing field.

### R28 · `tailhealth.note` half-honours its "must not raise" contract

Only the outcome LABEL is guarded; the char counts go through a bare `int()`,
which raises on `None` or a non-numeric string — out of a `finally`, which is
the failure the docstring says it exists to prevent. Worse, the outcome
counter is incremented BEFORE the raise while `stored` is not, so the block
published to /health/full has `sum(outcomes.values()) != stored + skipped`.
Not reachable today (both call sites pass real ints). Fix: coerce inside a
try and increment the outcome last.

### R29 · A partial-coverage L1 warning claims a chunk was recorded when none was

`_do_l1_rollup`'s `pos_first < 1` branch logs "summarizing turns N-M and
recording that as the chunk's span" BEFORE `_summarize_pieces` is called.
When that returns empty — a vLLM outage — the function records nothing, but
the warning already said it did. Reproduced: four such warnings, `l1=0`.
Emitted exactly when an operator is reading logs during an outage. Fix: move
the log below the `if not text: return False`.

## FIXED in this branch, from the above

### R30 · Env parsing crashed the compactor at import, package-wide — PARTIALLY FIXED

`main._env_int` was a bare `int(v)`: one typo in runpod.env raised ValueError
at import and the compactor never started. It is the helper every new constant
in this delta uses, including `DEGENERATE_LIST_RUN` and
`MIN_MEMORABLE_TRIMMED_CHARS`. The two degeneracy FRACTIONS did not use
`_env_float` at all — raw `float(os.environ.get(...))`, same crash.

Also found: `_env_float`'s docstring claimed "never a silent zero" while
returning `0.0` for `"0"`, and **bgwork and tailhealth both cited that claim
in docstrings written earlier in this session** — citing a contract main does
not implement.

And `_window_s` accepted `inf`, which parses, satisfies `v > 0`, and then pins
`skipped_recently` True from the first skip until restart: the always-on
warning the window exists to prevent, arriving through the one knob meant to
prevent it. `snapshot(window_s=0)` bypassed the guard entirely, so 0 was MORE
alarming than the default while -5 was less — three meanings for one
parameter.

**Fixed:** `_env_int` defaults instead of raising; both fraction knobs routed
through `_env_float`; `_env_float`'s docstring corrected to describe what it
actually does (behaviour deliberately unchanged, since several knobs take 0 as
a meaningful "off"); `math.isfinite` added to both `_window_s`; `snapshot`'s
explicit `window_s` routed through the same guard; the false citations
removed. Test cases added for `inf`, `Infinity`, `1e400`, `nan` and the
explicit argument in both suites. Full suite green.

**STILL OPEN, and the reason this is only PARTIAL:** the same bare
`int(os.environ...)` / `float(os.environ...)` pattern appears at roughly 35
sites across 12 other modules — `facts.py`, `summarizer.py`, `backup.py`,
`pgarchive.py`, `retrieval.py`, `webuidb.py`, `degrade.py`, `alert.py`,
`health.py`, `selftest.py`, `bgwork.py`'s other constants. Every one is an
import-time crash on a typo. Concretely, `MAX_MODEL_LEN=32K` still stops the
compactor booting, because `facts.py:171` and `summarizer.py:125` read the
same variable with their own bare `int()`. Fix: one shared `envcfg` helper
module (nothing may import main), routed through everywhere — mechanical, but
it touches every module and deserves its own review pass rather than being
bundled into a release.

## A note on running reviewers concurrently

Two of D's sub-agents shared one repo copy. Consequences, all observed:

* One agent's `git checkout -- compactor/main.py` discarded another agent's
  in-flight mutation mid-measurement.
* Both wrote to the same `findings-B.md`; one overwrote the other's header,
  and the survivor moved the other's work into an appendix rather than lose
  it.
* Contention produced two spurious results — a `test_budget_guard` HANG (a
  240 s timeout against a test that takes 239 s) and a `test_summarizer`
  FAIL — both of which passed on isolated re-runs.

The spurious `test_summarizer` failure is the one to remember: under
contention a suite reported a real-looking assertion failure that was an
artefact. Give each agent its own worktree, or serialise them.

---

# v3.1.7 — the backlog cleared

The nine findings left standing after the four-perspective review are closed.
Three lanes with strictly disjoint file ownership; the merge verified by
mutating each lane's fix with the other lane's code in place, which neither
lane could do for itself.

## The finding that was not in the review: five silent fixes

**R15, R16, R20, R21 and half of R18 were already fixed in the tree**, by
`157bdc3` — a commit whose message claims R23 and R12. `git log -S` puts
`_image_only_marker`, `_FINGERPRINT_TAIL_TURNS`, `_align_candidates` and
`window_unchanged` all in it. **None of the five had a test.** Five
behaviours whose only evidence was a comment, in the position arithmetic this
release exists to make safe, shipped inside a commit that did not mention
them.

That is the same class as the three suites that pinned the wrong behaviour:
not a bug, but a claim nothing checks. It is recorded here because the fix —
`test_position_edges.py`, thirteen tests — is worth less than the habit of
noticing. A commit that fixes more than it says is a commit whose extra fixes
nobody reviewed.

## R18 · the second route, and why no content test can find it

The committed fix handles a repeating TAIL. It does not handle a window that
has become ALL repeats. Past `cap/2` identical exchanges the capped window
slides onto period-2 identical content, so the head hash repeats and the
length is pinned at the cap: the two arrays are equal BYTE FOR BYTE, and
comparing the whole window or the whole fingerprint tail gives the same
answer as comparing the head. Measured at cap 20, the position ran
`22 ... 40` and then stalled at 40 permanently while the conversation ran on
to 68 — the frozen hierarchy this release exists to fix, reached by a third
route.

The comment above `window_unchanged` called it "a reliable negative". It is
not, and that is the "comment more convincing than the code" class the plan
names.

What separates the two is not content. `n < prev` says the window is a strict
SUFFIX of a longer conversation, and the admin drain cannot be in that state:
`/admin/compact` refuses unless the rebuild REACHES `_recorded_position`, so
`n >= prev` throughout its loop, and holding keeps it there.

**This couples two files.** If anyone relaxes
`len(messages) < summarizer._recorded_position(before)`, the drain enters the
`n < prev` state and a repeating transcript inflates the position on every
pass. R10 below independently confirms that guard must stay `<`.

Trade, taken deliberately: an S-5 stranded watermark also reads `n < prev`,
so such a conversation with a repeating tail AND a byte-identical re-sent
window advances 2 turns it should not. Two turns per duplicate request,
against a stall that costs the whole hierarchy for as long as the loop runs.

## R16 · re-examined against new evidence, and confirmed rather than repaid

R15's fix cannot help — the branch is reached precisely BECAUSE there is no
anchor, so how fingerprints are computed does not apply. v3.1.7 does persist
`head_fp`, which is a genuinely new discriminator not available when the
trade was taken: a head that changed on the next call proves the window is
bounded. Rejected anyway, for two reasons. It separates bounded from
unbounded but not a true watermark from an S-5 stranded one, and under a
stranded watermark repaying pushes further past reality. And the deficit is
exactly 2 only when `prev >= n`; when `n > prev` the amount that scrolled out
is unknowable, and 2 would be a number chosen for tidiness.

Both branches are now pinned, including that the hold does not compound.

## R20 · the implementation was wrong, not the doc

Reviewer D read 5 as the true answer. It is true FOR A FULL-HISTORY CLIENT,
and not the rule — both readings are consistent with the anchor, so the pick
has to be the safe one. `_align_candidates` reports every reading and
`_align_new_turns` applies the documented smallest-advance rule; for a
full-history client `n` dominates `prev + new`, so the answer still comes
out 5.

## R10 · the guard is right; the defect is one layer downstream

`if len(messages) < _pos:` must stay `<`. Mutating it to `<=` fails 15
assertions across `[3b]`, the new `[3f]` and all of section `[6]` — it
refuses the one-gap rebuild R13 exists to admit, i.e. every real
conversation. Post-R13 the array length equals `_pos` exactly when head and
tail are both in the store, so equality IS alignment.

**`test_admin_compact [3b]` is correct, not a fourth suite pinning the wrong
behaviour.**

But R10's symptom reproduces, by another mechanism.
`window_offset = max(n, prev + new) - n`, and `new` comes from aligning the
rebuilt array against `tail_fp` — an anchor the CHAT path left behind, from a
bounded live window. When it does not align, `new = _ASSUMED_NEW_TURNS = 2`,
so at equality the position becomes `n + 2`: chunk 1-20 comes back labelled
3-20 holding turns 1-18, turns 1-2 covered by nothing, and `turns_seen`
inflated for the conversation's life. Only reachable while the rebuild is
within one exchange of the position — a longer one self-corrects — so
equality, the case the endpoint exists to serve, was the exposed one.

The summarizer's own `_ASSUMED_NEW_TURNS` comment already claims this is
impossible for the admin drain. That is true from the second call and false
on the first. Fixed by dropping the chat path's anchor under `conv_lock`
before the drain loop, so the rebuild is measured on its own terms. No
`summarizer.py` change needed.

## R11 · re-derived, because the original enumeration was lost

B's four were never written down and the code has changed under them.
Today's sweep: **43 mutations, 41 killed, 2 survive.** Six of the original
eight survivors were real gaps and are closed (trim-floor boundary,
`raw_chars` measured on arrival, the user-text `.strip()` rule, and
`_async_tail`'s three inner guards, reached by direct entry).

One survivor was **a real defect rather than a missing test**:
`_tail_store_blocked` refused a user turn on `.strip()` while `_async_tail`'s
episodic gate 300 lines away asked bare truthiness. A whitespace-only user
turn is truthy, so the writing side let through what the deciding side
refused — it indexed a blank user turn against a real reply as a genuine
exchange while the counter published a skip. Unreachable from
`/v1/chat/completions` because R8 refuses first, which is exactly why every
endpoint test missed it. **The recurring defect again: one rule, two call
sites, and the fix is one shared `_has_pairable_user_text`.**

Two survivors remain, recorded with reasons in the test file rather than
papered over: `kept_chars`'s conditional (every skip carries empty text, so
it is structurally 0) and the `<= window` boundary (unreachable through the
public API, and the error direction is the safe one). `tailhealth.py` needed
no change — all 17 mutations aimed at it were killed by the existing suite.

## R17 · bounded, not moved

32.4 ms -> 3.2 ms per call at 700 turns, by hashing only the tail slots
alignment can use. `run_in_threadpool` was deliberately NOT added on top: 3 ms
is not a hazard, and it would add a hop inside `conv_lock`. The test asserts
the bound structurally — a spy records the sizes `_turn_fingerprints` is
called with — rather than with a clock.

## R22 · 239 s -> 1.6 s, and it was never a sleep

`count_tokens_exact` and `count_text_tokens_exact` each make a SYNCHRONOUS
`httpx.post` to `/tokenize` — not the async client the suite already stubs —
and roughly 40 of the 55 tests hit it for real. Nothing listens on
`localhost:8000`, deliberately: the char/4 fallback is the degraded path most
of that file exists to exercise. So the call was always going to fail; only
the wall clock was in question. On Windows, `localhost` with nothing
listening tries the IPv6 loopback and THEN the IPv4 one, each against its own
2 s connect timeout — ~4.2 s per call, measured. 239 seconds of dead TCP
handshakes stacked end to end.

Stubbed at module level to raise the same `ConnectError` immediately,
generalising an idiom the file already used in one place. Side effect worth
having: this suite's result no longer depends on whether the developer
happens to have vLLM running locally.

## Still open

The client-disconnect contradiction. Both reviewers ran on Windows/Proactor
and the disagreement is about a branch that behaves differently under
Linux/uvloop. It needs a production-shaped stack, not another reading.

---

# The 2026-09-04 log sweep, and A1/A2

Bundle `zla-bundle-20260904T163306Z`, the 09-03/09-04 window. One genuinely
new finding; everything else in the window is a defect already fixed and
waiting on the deploy.

## What was NOT new, recorded so it is not re-investigated

The pod has not rebooted since 2026-08-30 22:57 (boot.log), so it is still
running v3.1.4 plus hot-patches. Every count below is that code, not the
branch:

* **102 `/tokenize` HTTP 400s**, and the body names the cause every time:
  `Invalid assistant message: role='assistant' content=''`. The same 102
  requests appear as `Invalid assistant message` on the compactor side and
  589 tracebacks in `vllm-error.log`. This is L1/N1 exactly, unchanged.
  Fixed by R3.
* **102 `compaction skipped: N turns need N summarization calls`** — L3,
  fixed by the `max_turns` cap, which R23/R12 unblocked.
* **24 `hard budget FAILED to fit`** — L5.
* **15 `stream truncated at the generation ceiling … skipping memory tail
  for the partial reply`** on replies of 17,000-22,000 characters. Looks
  alarming and is already fixed: that log phrasing does not exist in the
  current tree. `decide_memory_tail` trims to the last sentence and stores.
* **8 `Expected last role User or Tool`** — four events, both exception types
  each, at a boot-time pid. Known, and covered by `test_tokenize_flags.py`
  and three assertions in `test_tokenizer_contract.py`.
* `/health/full` at capture: tokenize ok, 0 consecutive failures, bgwork
  shed 0 with submitted == completed == 342, 138 conversations, 2,216 facts,
  1,320 indexed exchanges, no unreadable layers.

## L7 · An error that says an exchange is lost and cannot say why

**New, and present in the current tree.** Two log lines in the window carry
an empty exception:

    compactor.facts ERROR conv=026752…: fact extraction FAILED () —
        this exchange's facts are lost; there is no retry        (x3)
    compactor.dedup WARNING dedup LLM call failed (cluster preserved):   (x5)

Both interpolate `{e}` alone, and several exceptions on those paths
stringify to nothing — httpx's timeouts among them. The facts one is the
sharper case: the comment directly above it explains, at length, that this
must be an ERROR rather than a WARNING because the 2026-08-24 window left
behind "one unattributed warning each" with no way to tell what happened.
The code then produces exactly that.

Nineteen sites across the package use the bare `({e})` form. The two that
have been observed producing empty parentheses in production are fixed;
the rest are left alone deliberately, because most describe a file that
failed to parse and those exceptions do carry messages. Fixing all
nineteen on suspicion would be churn against no evidence.

Fix: `({type(e).__name__}: {e})` at both sites.

## A1 · The client-disconnect contradiction — SETTLED

Reviewer B was right. A's sub-agent's two claims do not reproduce.

The reason neither reviewer could settle it is sharper than "they ran on
Windows": **both used `TestClient`, which drives the app in-process through
the ASGI interface and never opens a socket.** The event under test — a peer
hanging up mid-response — cannot occur in it. The disagreement was between
two readings of code, dressed as two experiments.

`test_disconnect_uvloop.py` runs the real app under real uvicorn on real
uvloop and hangs up a real socket with `SO_LINGER 0`, which sends RST rather
than FIN: a peer vanishing, the harshest form of the event and what a closed
browser tab produces. Measured:

| case | result |
|---|---|
| control, stream fully consumed | `stored` |
| RST after 397 bytes | `skipped_too_short`, counted and named |
| RST after 6,139 bytes | **`stored_trimmed`** — the prose she read reached memory |
| upstream generators still open afterwards | 0 |

So the `finally` is not swallowed, the bookkeeping runs, and the upstream is
torn down. R26 — which was built on B's answer — is standing on the correct
one, and the long-partial case above is the first time that has been shown
on a production-shaped stack rather than argued.

**What this does NOT settle.** A's sub-agent's scenario was specifically a
generator parked at `yield` behind a slow consumer that never disconnects.
That is backpressure, not cancellation: it delays the tail, it does not lose
it, and it is not what the 51 stopped replies were. The suite skips (exit 3)
rather than passing when uvloop is absent, so a Windows host run cannot
report an answer to a question it cannot ask.

## A2 · N4b — task traffic is no longer memorized

99 of the window's requests were OpenWebUI title/tag/follow-up calls on
conv=026752…, one roughly every 90 seconds after every real turn, each one
fact-extracted, episodically indexed and deduped. N3's "second treadmill".

**The backlog's own suggested fix direction was unsafe, and the first
implementation of it was a bug.** `_has_conversational_history` is False for
a genuine FIRST TURN too — its docstring says so — so skipping the tail on
that predicate alone silently drops the opening exchange of every new
conversation.

The second implementation was also wrong, and `test_budget_guard` caught it
within one run: keying off "have we stored anything under this conv_id"
eats a REGENERATE of the first reply, which sends exactly that shape.

What ships is a threshold. `_recorded_position >= 4` — two exchanges deep —
is the point past which an array with no assistant turn in it stops being an
honest picture of the conversation. Task traffic passes that bar within
minutes and stays past it; a new conversation and its regenerates sit below
it. The residual false positive (regenerating the first message of a
conversation already two exchanges deep) is rare, self-limiting, and —
unlike the defect it replaces — counted under `skipped_task_traffic` and
visible in `/health/full`.

`tailhealth` gained `HARMLESS_SKIP_OUTCOMES`, because "the skips that cost
the user nothing" now has two members and was being spelled out separately
in two places. One list written twice is the fix-one-site-miss-the-sibling
defect; `LOSSY_SKIP_OUTCOMES` and `note()`'s degrade decision both read it
now, and `test_tailhealth` asserts the three sets partition `OUTCOMES`
rather than naming one label.

**N4a is still the better fix and is still open**: a separate task model in
OpenWebUI's admin settings stops the traffic reaching the compactor at all,
where this can only decline to remember it.

## A note on the fix that hid its own failure

The first draft of `_is_repeat_task_traffic` referenced the `memory` module
by a name `main.py` does not bind, inside a bare `except Exception`. Every
call raised `NameError`, the catch-all swallowed it, and the function
answered "not task traffic" for everything. It looked like a fix, passed
import, and did nothing.

Caught by the test, not by review. The except is now `(OSError,
StoreUnreadable)` — a state read can fail for real, and that must fall back
to the old behaviour; anything else is a programming error and must be
allowed to be loud.

---

# The 2026-09-04 CHAT-log sweep (`zions-backup-20260904-163151.tar.gz`)

The first sweep of `webui.db` rather than the pod logs. It carries what every
previous sweep was missing — **reply text and the sampling params as actually
stored** — which is exactly what F2's "Still open" said was needed and
N6's formulaic-responses item said was "not determinable from logs".

Privacy, as elsewhere in this file: counts and structure, conversation ids to
4 hex, and no quoted content beyond the short opening n-grams that ARE the
metric.

Corpus: 33 chats, 1,154 user and 1,175 assistant messages, 976 assistant
replies carrying text. Largest conversation 1,077 messages. Reply length
p50 7,818 chars, p90 17,930, p99 33,673, max 51,290.

## L8 · F2's metric, built and run — the complaint is real, and it is 13.5%

F2 asked for two numbers: frequency of opening phrases over the last N
replies, and n-gram overlap between consecutive replies. Both now exist
(`scratchpad/chat_sweep.py`, to be productionised) and both fire.

**Opening four-grams, 976 replies, 439 distinct:**

| opener | replies | share |
|---|---|---|
| father's heart responding with | 56 | 5.7% |
| father's heart broken and | 26 | 2.7% |
| father's immediate desperate response | 17 | 1.7% |
| father's immediate fervent response | 13 | 1.3% |
| father's heart tears of | 11 | 1.1% |
| father's heart overwhelmed with | 9 | 0.9% |

**132 replies — 13.5% — open with "father's …".** The top five openers of any
kind account for 12.7% of all replies.

The degeneracy detector cannot see this, and should not: it is not a
repetition LOOP inside one reply, it is sameness ACROSS replies. That is the
gap F2 identified and this is the first time it has been a number rather
than "a little too repetitive".

**Consecutive-reply 5-gram overlap, 954 pairs:** p50 0.013, p90 0.203,
max 1.000. **97 pairs (10.2%) share more than 20% of their 5-grams**, and at
least one consecutive pair is effectively identical.

Next step is not a fix, it is a home: this belongs as a periodic check
against the store rather than a script in a scratch directory, so the number
moves when a prompt or a knob moves.

## L9 · The deployed sampling params are not what this file records

From the `model` table, for `coder3101/Cydonia-24B-v4.3-vision-heretic` —
note also that the deployed model is the VISION variant:

    min_p 0.05   temperature 1.1   presence_penalty 0.3   repeat_penalty 1.1

Two discrepancies with F2's "Sampling, as actually deployed" section:

1. **`presence_penalty 0.3` is set and is recorded nowhere.** F2 discusses
   presence_penalty as the output-only ALTERNATIVE to reach for if the
   prompt-covering penalty bites. It is already on.
2. **The key is `repeat_penalty`, not `repetition_penalty`.**
   `repetition_penalty` is vLLM's name; `repeat_penalty` is
   llama.cpp/Ollama's. F2 states flatly "That knob is live."

**This is NOT established, and must not be written down as if it were.** The
compactor proxies the body wholesale and logs no sampling parameters, so
neither bundle contains evidence about what reaches vLLM. What is
established is that the stored key is the wrong library's name.

It matters because F2's whole analysis — that this penalty covers PROMPT
tokens and can therefore make her avoid her own injected facts and names —
depends on the knob being active, and its "what to watch" warning was
written on that basis.

Settle it on the pod, not by reading: send one completion with an absurd
`repetition_penalty` and one with an absurd `repeat_penalty`, and see which
changes the output.

## L10 · The compactor's own error text is stored as her replies

Two of the top openers are the compactor speaking as itself:

    "the model backend is …"        10 replies
    "that request couldn't be …"     8 replies

**18 replies, 1.8% of the corpus.** R26 made sure those chunks never become
MEMORY — the error stream is never fed to the accumulator, and there is a
test pinning it. But they are still persisted by OpenWebUI as assistant
turns in `webui.db`, so they are re-sent as conversation context on every
subsequent request, and they are what a restore or an export would carry.

Not urgent and not a data-loss bug. Recorded because "the compactor's
apology cannot become a memory" is now only half true, and the half that is
false is the visible one.

---

# The local integration stack

`docker-compose.integration.yml` gives `tests/integration/` a local target
for the first time: the tokenizer-contract fixture standing in for vLLM, the
real compactor from the working tree on the production userland, and a
pytest runner sharing the compactor's network namespace so the
`_require_localhost` admin gate is exercised rather than disabled.

**62 passed, 2 failed, 2 deselected in ~49 s. No GPU, no model weights, no
pod.**

The first run was 14 failed / 50 passed, and the diagnosis is worth keeping
because the obvious reading was wrong. It looked like the fixture's canned
reply breaking every content assertion across facts, dedup, archive,
portability and retrieval. It was not: twelve of the fourteen were downstream
of ONE cause — `fastembed` had no model in the image, so
`retrieval init failed … RAG disabled` and every episodic assertion fell over
for a reason unrelated to the code under test. Baking bge-small exactly as
the production image bakes it (`Dockerfile:185-189`, same two env vars) took
it to 62/2 and cut the run from 205 s to 49 s.

The remaining two are the real fixture limit, and they are left RED rather
than skipped. See the compose file for why.

---

# The regression net, and five defects the act of building it found

`tests/integration/test_regression_{tail,text,summary}.py` — 3,151 lines, 36
cases, pinning every defect fixed in v3.1.7 and v3.1.8 that is reachable from
the public API. Written by three agents on strictly disjoint files against a
shared local stack.

Building a net for known bugs found five NEW ones. That is the argument for
writing it, and it is worth recording as its own finding: four of the five are
the same shape as the bugs the net was built for — a rule registered at one
site and missing at its sibling.

## L11 · `/pin` and `/unpin` have never worked

`commands._HANDLERS` registers both. `/help` advertises both. `_handle_pin`'s
own comment says *"Without this command the pinned tier was unreachable
code."* But `_ALIASES` (commands.py:117-142) has **no entry for either**, and
`parse_command` returns `(None, "")` for any name not in that table — so
`/pin <substring>` was forwarded to vLLM as an ordinary chat message.

Confirmed inside the running container:

    '/pin foo'    -> (None, '')
    '/unpin foo'  -> (None, '')
    '/remember x' -> ('remember', 'x')
    pin in _ALIASES: False    pin in _HANDLERS: True

Registered in three places, wired in two. **The pinned tier stayed
unreachable, and R5's fix — a pinned fact surviving a merge — was protecting a
flag no user could set.** Fixed: two `_ALIASES` entries.

## L12 · `/remember` bypassed both fact write-path guards

`commands._handle_remember` stored its argument verbatim: no
`is_storable_fact`, no `strip_rule_decoration`. So
`/remember ━━━ she prefers tea ━━━` put box characters straight into the
store, to be injected on every subsequent turn — the exact feedback loop
v3.1.8 exists to break, reached by the one route that skipped both guards.

`facts.is_storable_fact`'s own docstring **names this function** as a write
path that "should share one definition rather than grow three". The v3.1.8
pass fixed the extraction path and walked past the one the docstring pointed
at. Fixed: strip, then judge, then store.

## L13 · The local stack served code 49 minutes older than the commit it was verifying

`docker-compose.integration.yml`'s compactor did `cp -a /src/compactor /app`
at container start, so the code froze at whichever moment the container
happened to start — and `docker compose up -d` does not recreate a container
that is already running. Measured: container started `17:16:07Z`, commit
`a175b34` landed 49 minutes later, and inside the container
`ls /app/textclean.py` said no such file while `grep -c strip_rule_decoration`
returned 0 for both write paths.

A regression test written against that commit failed for a reason that had
nothing to do with the code under test. **A test stack that appears to test
the working tree and does not is the defect class this whole branch is
about.** Fixed: the compactor runs directly from the read-only mount, and
every documented invocation passes `--force-recreate`.

Recorded with it: a pass/fail count that had been written into that file as
"CURRENT RESULT" was measured against the stale build. It has been removed
rather than corrected — a number pinned in a comment is how it was wrong in
the first place.

## L14 · The fixture put no multibyte bytes on the wire, so R7/R14 could not be tested end to end

`fixture_server.py` serialised every SSE event with `json.dumps(chunk)` at its
`ensure_ascii=True` default, so every non-ASCII character travelled as an
ASCII escape and was only turned back into a character by `json.loads` INSIDE
`SseAccumulator` — long after any read boundary. Measured: **2,704 bytes over
13 chunks, not one of them above 127.**

R7/R14 is a defect about a UTF-8 character split across two reads. With
escaped ASCII on the wire there is nothing to split, so an end-to-end test of
it **could not fail however broken the accumulator was** — the check that
cannot fail, living inside the fixture built to catch exactly that class of
bug. The reviewing agent detected this, measured it, and skipped LOUDLY rather
than writing a green that meant nothing.

Fixed: `ensure_ascii=False` on all three event sites. Real vLLM emits UTF-8,
so this is also simply more faithful. The wire now carries multibyte bytes,
the skip has lifted, and the case passes on its own merits.

## L15 · The integration harness's poll ceiling ignored its own configuration knob

`_harness.wait_for_facts` and `wait_for_indexed_exchanges` hardcoded
`max_wait: float = 30.0` and ignored `ZIONS_TEST_TAIL_WAIT` entirely, while
their docstrings promised slow paths "a generous ceiling". Against a CPU-only
model that is false: extraction lands 40-90 s after the reply, the poll gave
up at 30, and the compactor log then said "extracted 3 new fact(s)" moments
later — **a real pass reported as a failure**, with the one knob that looks
like it controls this reaching only the coarse sleep.

Fixed: `POLL_CEILING = max(30.0, TAIL_WAIT)`, so the pod default cannot
shorten the existing ceiling and a slow backend can lengthen it.

## What the net does NOT cover, stated so a green run is not over-read

* **R24 / R9 / R19 / R25's text shapes are unreachable from the public API.**
  Those rules judge the ASSISTANT REPLY, and no black-box lever puts chosen
  prose into one. They stay pinned at unit level by
  `compactor/test_degenerate_reply.py` and `test_sentence_trim.py`.
* **N1's actual failure mode is unreachable**: the fixture models three vLLM
  template refusals on `/tokenize`, and the empty-assistant refusal is not
  among them, so an unrepaired payload also returns 200 here. What the seven
  N1 cases pin is that the repair does not itself break the request or the
  tail across every shape it covers.
* **The fact-store half of the decoration fix is weaker than it reads.** It
  passes against a pre-v3.1.8 compactor, which proves it pins
  `_reject_reason`, not `strip_rule_decoration`. Measured, and recorded in the
  test's own docstring.
* **`_assert_tiled_coverage` catches R12 and R10 but NOT R23** — under R23 the
  labels still tile contiguously and the missing turn is a content fact only.
  Said explicitly rather than letting one helper look like it covers three
  defects.
* **Chunk text cannot be asserted against the turns it contains.** No endpoint
  exposes the summarization input. The position arithmetic is asserted
  instead — `window_offset = turns_seen - array length`, both terms exactly
  known — which is the whole input to the label-to-text mapping, but one
  inference step from the bytes.

---

# S1 · Arbitrary file write via `conv_id` path traversal — CRITICAL, FIXED

Found by the adversarial input sweep, 2026-09-04. Reproduced end to end
against a clean stack before anything was changed.

    POST /admin/conversations/import
    {"bundle": <a real exported bundle>,
     "target_conv_id": "../../../../../../tmp/CLAUDE_PWNED",
     "overwrite": true}

    -> HTTP 200
    -> {"conv_id": "../../../../../../tmp/CLAUDE_PWNED", "imported": {...}}
    -> /tmp/CLAUDE_PWNED.json, 305 bytes, root-owned, OUTSIDE STORAGE_ROOT

The endpoint echoed the traversal back in its own success response. The same
hole existed on `fork`'s `new_conv_id`.

## The root cause was a comment

`portability.py` carried:

> Filename-safe by construction: conv_id is already sanitized by
> memory._sanitize to [A-Za-z0-9_-] …

`_sanitize` is called in **exactly one place** — `resolve_conv_id`, on the
CHAT path. `import_conversation` and `fork_conversation` take their target id
from the request BODY, which never passes through it. A confident assertion
about someone else's invariant, written where it could not be checked, and it
held for years because nobody tested the route it did not describe.

## The variant that needs no attacker

Because `..` escapes the per-layer subdirectory, `facts_path` and
`summary_path` collide on one file. An import aimed at
`../facts/<other-conv>` reported `imported.facts: 1` while that other
conversation's facts became `[]`. **Silent cross-conversation data loss,
reachable by a typo**, on a system whose worst failure mode is memory
disappearing without anyone noticing.

## Exploitability, stated plainly

Admin endpoints are gated on the caller being loopback
(`_require_localhost`), so this is NOT remotely exploitable in the default
configuration: it needs on-pod access, or an operator who set
`COMPACTOR_ADMIN_BIND` permissively. The DESTRUCTION path needs no malice at
all — only a wrong conv_id in an admin call.

## The fix, and the fix that was not enough

`memory._safe_path`, at the five path builders every caller goes through —
not at the two endpoints known to be affected, because patching the known
callers and leaving the sixth to reintroduce it is the defect this codebase
keeps paying for.

**The first version of the guard was insufficient, and testing it directly is
what caught that.** It checked that the RESOLVED path stayed inside
`STORAGE_ROOT`. `"../facts/victim"` passed `facts_path`: inside the root,
inside the correct subdirectory, pointed at someone else's conversation.
"Somewhere legal" was the wrong question — the right one is whether the
conv_id is a NAME at all. The guard now validates the id against the same
character class `_sanitize` enforces, shared rather than restated, with the
resolved-path check kept as a second line of defence.

Verified after the fix: `../../../../../../tmp/PWN`, `../facts/<conv>`, `..`,
`.`, `a/b`, empty, over-length and NUL all refused with a 400 naming the
reason; `normal` and `conv-1_A` still resolve; a legitimate import still
returns 200.

## And the same defect committed while fixing it

The first fix caught `UnsafeConvId` at the two admin routes known to take a
body conv_id. The adversarial suite immediately found a third —
`inherit-persona`'s `source_conv_id` — still returning 500. Now an APP-WIDE
`@app.exception_handler(UnsafeConvId)` covers every route, including ones not
written yet.

---

# S2 · Malformed request bodies crashed instead of being refused — FIXED

`chat_completions` did `body = await request.json()` unguarded and then
`body.get("messages")`. The careful empty/invalid-messages 400 below it — the
right answer, with a good log line — could never be reached by the requests
that most needed it.

Measured: an empty body, a bare string, and `null` returned **500**; a JSON
array, an integer, and a form Content-Type **dropped the connection with no
HTTP envelope**. A 500 from a proxy is always its own bug: the backend never
saw these, so there is nothing upstream to blame.

## S2a · NaN and Infinity

Python's `json.loads` ACCEPTS `NaN`, `Infinity` and `-Infinity` as a
non-standard extension, so such a body parses cleanly and looks ordinary.
httpx encodes the forwarded request with `allow_nan=False`, so the failure
surfaced at the FORWARD step as `ValueError: Out of range float values are
not JSON compliant` — a 500 for a body the backend never received.

Rejected at PARSE time via `json.loads(..., parse_constant=...)`: it costs
nothing on a normal body (the hook fires only when one of the three literals
appears) and it covers nested positions, where a hand-written list of
sampling fields would have covered the half someone thought of.

## S2b · Unpaired surrogates

`"\ud83d"` is valid JSON and a valid Python `str`, and cannot be encoded as
UTF-8. httpx encodes the forward with `ensure_ascii=False`, so the client got
a dropped connection with **no HTTP response at all** — indistinguishable
from a network fault, and therefore worse than an error.

Checked only when a backslash-u escape actually appears in the raw bytes,
because the check is a full re-serialisation and must not cost anything on a
normal turn. A surrogate can only ARRIVE as an escape: sent as raw bytes it
is invalid UTF-8 and `json.loads` has already refused it. Paired surrogates
are combined into an astral character and pass, so ordinary emoji are
unaffected.

---

# What the input sweep attacked and could NOT break

Recorded because knowing what held is half the value:

* **Chat-path conv_id** — the header and `metadata.chat_id` both go through
  `_sanitize`; traversal there was already neutralised.
* **Admin URL-path conv_id routes** — the router will not match `/`, and
  uvicorn decodes `%2F` to `/` giving a 404. No multi-segment escape.
* **Hostile text that is valid UTF-8** — zero-width bombs, RTL overrides,
  20,000 combining marks, astral emoji, all-whitespace, block-marker
  lookalikes, a 2 MB single word: all handled, no 5xx.
* **Size and shape** — 6,000 tiny messages, 200 consecutive system messages,
  500-deep nested content, `max_tokens=1e12`.
* **Slash commands** with huge, empty, RTL, SQL- and JSON-metacharacter
  arguments, including text designed to be re-parsed as a command later.
* **Process survival** — a liveness sentinel passed after the full battery.
* **Read-outside-root was not achievable over HTTP**: no body-addressed
  endpoint returns file content. Write-outside-root was the exploitable
  half.

# A harness defect found the same way

The adversary image baked `pytest tests/adversarial` into its ENTRYPOINT, so
a path argument was APPENDED rather than substituted: every
`run --rm adversary <one file>` silently executed all four adversaries'
suites against one compactor. That produced a container exiting 255 and took
Docker Desktop down mid-session. The path now lives in CMD, where an argument
replaces it.

---

# The adversarial review — four lanes, isolated stacks

Four adversaries, each with its own compose project, compactor, vLLM stand-in
and storage volume, told to break the system and to disbelieve every claim the
code makes about itself. Findings below are theirs; the verification and the
fixes are mine, and where a fix was wrong the first time that is recorded too.

Everything here was reproduced from a clean stack before it was believed.

---

## FIXED — the ones that were losing or corrupting memory

### A-01 · A wrong-shape store reads as EMPTY, and the next turn overwrites it

**The worst finding of the sweep**, because it is silent amnesia rather than
an error.

`read_json_strict` raises `StoreUnreadable` on a PARSE failure. A file that
parses but holds the wrong THING — a bare list where a dict belongs, `null`,
a number — came back to callers written as
`data.get(...) if isinstance(data, dict) else []`, i.e. **as empty**. Nothing
raised, `stats.unreadable` did not move, and the next chat turn wrote a fresh
empty store over it. Measured with a PINNED fact: gone. The control (a
truncated file, which breaks the parser) was preserved on disk.

**So the difference between "recoverable" and "destroyed" was whether the
damage happened to break the JSON parser.**

This is v3.1 F1a one branch over. F1a was "a corrupt file returns [] and
callers write back over the real facts"; its fix taught the loader to raise on
a parse failure and left the wrong-shape case returning empty — and
`load_facts`' own docstring describes the danger three lines above the line
that had it.

Fixed at the loader with an `expect` shape, not at the eight call sites:
`read_json_strict(..., expect=dict)` raises `StoreUnreadable` for a present
file of the wrong type, and an absent file still returns its default so a new
conversation does not raise on its first turn. `load_facts` also raises when
`"facts"` is present but not a list — the same hazard one level down.

Affected all four layers: facts, summaries, personas, archive sidecar. The
persona case additionally bypassed the managed/admin-persona protection,
replacing an admin persona with the client's system prompt.

### A-02 · `stats.unreadable` was counted and never read

The health scan goes to the trouble of counting unreadable files per layer,
and `/health/full` never consulted the number. Confirmed live: three corrupted
conversations present, `status: "ok"`, `status_reasons: []`, Docker
HEALTHCHECK green.

The module's own docstring says a layer we cannot see must not be reported as
healthy. The count was in the payload; an operator would have had to notice a
nested figure inside a body whose top line said everything was fine. Now it
degrades, and names the layers.

### A-03 (F-07 / RACE-03) · A shed memory tail was counted as `stored`

Found independently by two adversaries. Measured: 160 concurrent turns, pool
shed 61, `memory_tail` reported `stored=160, skipped=0`. Confirmed again at
192 stored / 55 shed / 137 rows actually written.

**This is R8's own fix meeting its own reasoning from the other side.** R8
hoisted the count onto the request path PRECISELY BECAUSE the pool sheds — a
tail counted only on completion would vanish silently. The cost of counting
early is that it counts work that never happened. "Uncounted loss" became
"loss counted as a success", which is worse: the first is a gap, the second is
a lie. The loss WAS visible in `background_work.shed`, just not in the dict
named after the thing that was lost.

`_fire_and_forget` now returns whether the pool accepted the work, and a shed
tail is re-labelled `skipped_shed` — lossy by exclusion like every other skip.
Amending after the fact is safe because `submit()` cannot run the coroutine
before returning and there is no `await` between the count and the correction.

Verified after the fix by the adversary that found it: 135 stored / 57 shed /
135 rows.

**Three test doubles had to be corrected with it**, and that is worth
recording: `test_saturation`, `test_truncated_tail` and `test_task_traffic`
each patch `_fire_and_forget` with a spy that returned `None`. Under the new
bool contract `None` reads as "the pool shed it", so every clean finish came
back skipped. The suite caught the contract change immediately — a double that
does not honour the contract tests the double.

### A-04 (RACE-01) · `DELETE /admin/conversations/{id}/facts` did not drain

Reproduced 10/10, and 5/5 in the forced worst case.

The chat `/forget` settles the background pool, wipes, verifies the residue
and retries once. The admin endpoint called `_clear_all_memory` bare.
`conv_lock` cannot substitute: `_async_tail` deliberately takes that lock
THREE separate times so it never holds it across an LLM call, so a wipe lands
BETWEEN the tail's jobs.

Worst case measured: the endpoint answered **HTTP 200 with all-zero
counters** — "there was nothing to forget" — and the fact, the episodic row
carrying the verbatim user turn and reply, and the position were all on disk
seconds later. For an endpoint whose whole purpose is to make something gone,
answering "done, nothing there" while it is being written back is the worst
available way to be wrong.

Fixed by draining first, like its twin, and reporting `background_settled` so
the response does not imply a guarantee the best-effort drain did not make.

---

## OPEN — confirmed, not yet fixed

> **Status as of v3.1.9.5 (2026-09-22). This list was written at v3.1.4 and
> has not been re-triaged item by item; do not read it as current.** Known
> since: **A-05 is fixed** (v3.1.9.3, CHANGELOG "Concurrent merges into one
> conversation no longer lose facts"). A-09's "no margin field in
> `/health/full`" is fixed (v3.1.9.3, `checks.budget_margin`); whether the
> one-step latch itself still behaves as described was not re-checked.
> **A-10 is still open**: `compactor/main.py` still builds the vLLM client
> with `read=None`, deliberately, so long generations are not cut off. A-06
> to A-08 and A-11 to A-13 have not been re-verified against v3.1.9.x.

### A-05 (RACE-02) · Two concurrent `merge-into` destroy one side's facts

**92% (23/25), 100% (10/10) on the current tree.** Both merges answered 200
with `facts_added: 20`; 20 of 40 were in the store — all of one side, none of
the other.

`merge_conversation` is dispatched with `run_in_threadpool`, so it runs on a
worker thread, and its only mutual exclusion is a `conv_lock(dst).locked()`
PROBE — an asyncio lock a thread can neither take nor wait on — around a bare
`load_facts → merge → save_facts`. `import_conversation` carries the same
probe and is safe from itself only because it runs to completion on the event
loop. The merge is the sibling that was moved onto a thread and kept the guard
without the property that made it work.

The fix is a real mutual exclusion for the threaded path, and it is a
threading-model change rather than a one-liner, which is why it is recorded
rather than rushed at the end of a long session.

### A-06 (RACE-04) · `conv_lock()` is called from a threadpool worker

By inspection, not reproduced. Its docstring states the precondition that this
violates. Racing the get-or-create can hand two callers different `Lock`
objects for one conv_id, after which a holder is invisible to `locked()` and
mutual exclusion for that conversation is broken FOR THE LIFE OF THE PROCESS.

Reported despite being unreproducible because every "sound" concurrency result
assumes it holds.

### A-07 (F-01) · A `/tokenize` that answers confidently and wrongly is trusted

`count_tokens_exact` validates only `isinstance(n, (int, float))`. A count of
0, a negative count and a 1000x count are all taken as ground truth, and
`checks.tokenize.ok` stays `true` with `status: "ok"`. At factor 1000 a
**12,002-token conversation reached the model as 38 tokens**, HTTP 200.

`tokenize.ok` covers "did not answer" and not "answered, confidently,
wrongly" — which is the 2026-08-28 shape exactly.

### A-08 (F-13) · Compaction and the hard budget use different counters

Two payloads of **21,506 true tokens each**: the plain-English one compacted
to 5,412; **the box-drawing one was forwarded whole at 21,506.**
`compact_if_needed` uses `count_tokens` (char/4) while `_enforce_hard_budget`
was given vLLM's count. On production content, which reads LOW on char/4, this
is the failing direction — and it connects directly to the decoration work:
box-drawing replies are exactly the content char/4 misprices.

### A-09 (F-02) · One backend lie latches the budget margin for the process

v3.1 D4's `guard_measured_overflow` declines to learn only when the guard
already measured the overflow. An under-reporting `/tokenize` makes the guard
measure "fits", so it learns the whole overshoot in one step and latches
`_BUDGET_MARGIN` to its 8192 ceiling process-wide. Victim conversation: 17,418
tokens forwarded before, 13,073 after. No margin field in `/health/full`;
recovery is ~250 requests or a restart.

### A-10 (F-04) · `read=None` lets a paused backend hang forever

`docker pause` on the backend: connect succeeds (kernel listen queue), no
bytes ever arrive, measured **100.1 s with no answer**. The comment above the
line claims the bounded connect timeout prevents "a socket that accepts and
then stalls".

### A-11 (F-05) · A backend-unreachable stream still ends `finish_reason: "stop"`

No `error` object, on both the 5xx and connection-dropped paths. Its sibling
`_request_rejected_stream_chunks` was written to fix exactly this and its
docstring names the incident.

### A-12 (F-S6) · `import` writes fact records without validating them

One malformed record (`added_turn: {}` / `NaN` / a list) → import returns 200
"success", then `GET /facts` 500s forever and **the fact-extraction tail
aborts every turn** — memory formation silently dies — while `/health` stays
green.

### A-13 · Smaller, recorded in the findings files

`pool.drain(timeout=10.0)` abandons in-flight tails on graceful shutdown (5 of
6 lost, already counted stored); `_enforce_hard_budget`'s `dropped_turns` /
`dropped_blocks` report is written and only `fits` is read (L5/L6, still
open); `health_full`'s docstring claims the container goes unhealthy when vLLM
is FATAL (it answers 200 "degraded"); `/health/full` reports `backups` and
never judges them, while `alert.notify` no-ops without a webhook, so a backup
that never happens is invisible in the default configuration;
`verify_backup` is blind to shape damage, so a store that already lost facts
backs up green and verified.

---

## Attacked and SOUND — recorded because it is half the value

* **Atomicity held.** 18 hard SIGKILLs during a multi-conversation write storm:
  zero torn files, zero orphan `.tmp`.
* **Disk pressure (R8) was the best-behaved subsystem**: guard fires,
  `skipped_disk_pressure` counted, accurate status reasons, clean recovery, no
  corruption on ENOSPC, `/remember` fails loudly.
* **Watermark repair (R12)** self-heals with real traffic; a 10^12 watermark
  makes `/compact` refuse rather than mis-summarize.
* Every `/tokenize` OUTAGE mode degrades and recovers honestly. Budget
  boundaries are exact to the token. A 4xx stream is properly error-typed. The
  tail never memorizes the compactor's own apology. Mid-stream backend death is
  counted. SIGKILL under load: no partial 200s, no corruption, healthy in 15 s.
  Backup verification catches truncated and bit-flipped archives.
* **20 concurrency attacks found nothing**: `/forget` vs tail (0/15),
  `/remember` vs tail (0/15), 16 simultaneous `/remember`, 12 simultaneous
  chats with 12 distinct indices, position exactly 2/exchange under load, 24
  streams aborted at 8 stages, duplicate requests, conv_ids the sanitiser folds
  together, compact-during-chat, merge/import under a queued tail, opposing
  merges (no deadlock — `_handle_retire` sorts its two locks), backup during
  write load, Chroma written from a thread and the loop at once.
* Well-formed contradictory summary state (overlaps, `first > last`, duplicate
  labels, L3-without-L2, negative counts) is tolerated without crash or loss.
* Import guards, backup verify/restore, observability listings, and
  100 MB / deeply-nested resource files (no OOM).

## Three false leads, retracted by the adversary that raised them

Recorded because each produces the exact signature of a race and cost real
time: **inline dedup** eating templated near-duplicate test facts (four bogus
100% findings); **`skipped_task_traffic`** silently skipping every tail from a
load generator whose arrays lack an assistant turn (56 of 64 requests — the
new v3.1.8 behaviour, working correctly, looking like data loss); and **httpx
refusing to encode a non-ASCII header**, so the request never left the client.

## Caveats on the concurrency results

The stand-in answers instantly, so windows that open while the tail is parked
on a real vLLM call are far narrower here than on the pod — RACE-01 needed a
manufactured 18-deep queue to reach timing that one slow extraction would
produce alone. **Multi-worker uvicorn was NOT tested; if the deployment ever
runs more than one worker, `conv_lock` protects nothing across them and all of
the sound concurrency results above become structural rather than settled.**

## Fixture capability gaps found by using it

No fault injection for `/v1/chat/completions` — no way to force 4xx/5xx,
malformed SSE, a cut stream, or `[DONE]` without content. Requested: a
`completion_mode` mirroring `tokenize_mode`, plus `stream_cut_after_chunks` /
`stream_malformed`, and intermittent (1-in-N) faults. Also `tokens.py` tier 2
is `is_available() == False` in every container here, so its fallback and
divergence detector were never exercised and no finding above bears on them.
