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
