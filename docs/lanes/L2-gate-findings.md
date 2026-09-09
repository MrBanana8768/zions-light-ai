# L2 transport — adversarial gate findings

**Reviewed:** commit `44426a5`, both suites green (76/76 transport, 54/54 store).
**Verdict: the gate can be defeated, and two of the defeats are exhibited by the
suite's own passing tests.**

---

## D1 (High) — `verifyRealizedWindow` never checks the window ENDS on a `user` turn

`sendSet.ts:197-249` checks root presence, no system inside, no tombstone inside,
contiguity, **opens** on `user`, alternation, count. It never checks the last
element's role and never checks `turns` is non-empty.

`computeWindowIntent`'s derivation rests on "at gate time the chain ends on a user
turn, so `turnsOnChain` is odd" (`sendSet.ts:24-29`, `:106-108`). **That premise is
assumed and never verified.** When false, the function returns a number the
selection realizes exactly, so the gate goes green:

| `turnsOnChain` | last turn | intent | realized | gate |
|---|---|---|---|---|
| 0 | — | 0 | 0 | **ok**, `sentCount: 0` |
| 2 / 60 / 64 / 122 | assistant | 2 / 60 / 60 / 60 | same | **ok** |

`tests/compactor/sendSet.test.ts:79-86` builds exactly this: `buildConversation`
assigns `role = i % 2 === 0 ? 'user' : 'assistant'` (`helpers/fakeChain.ts:186`),
so turn 59 is **assistant** and is the leaf. A green test asserts the gate blesses
a window ending in an assistant turn.

`buildChatCompletionBody` then throws a **bare `Error`** (`request.ts:118-124`).
So the one condition the pre-send gate exists to convert into a typed, overridable
`context_truncated` escapes the gate and surfaces as an unhandled exception a layer
later — no `reasons`, no `GateFailure`, no override path, no receipt.

Reachable via: a "can I send?" precheck, a retry that re-gates before re-appending,
a branch switch to a leaf that is an assistant reply (normal after regenerate), or
any caller that appends after gating.

## D2 (High) — nothing binds the verified window to a version of the conversation

`runGate` issues three unsynchronized reads and returns. No transaction, no
snapshot, no `rev`. `getConversation` is declared in `ChainReader` (`types.ts:19`)
and **never invoked**. `conversation.rev` is never read. `audit.leaf` is fetched,
carried, and **never compared** to `turns.at(-1).id` — `verifyRealizedWindow` is
not even passed the audit (`sendSet.ts:284`).

**Scenario A (no race):** gate returns ok; between that and the POST, `selectLeaf`
moves the leaf (sibling switcher, or a second tab). Nothing re-checks. The model
gets branch A's context for a conversation now showing branch B. §4.1(b) exactly,
with no detection surface.

**Scenario B (inside the gate):** two sibling branches of equal depth. Gate audits
branch A (`chainFromCurrent = 66` → intent 59); a `selectLeaf` to branch B lands
before `readTail`; `readTail` walks **B**; realized 59; `59 === 59`; contiguity and
alternation hold within B; same root. **Green, wrong branch sent.**

## D3 (High) — in the steady state the count check carries exactly one bit

For `turnsOnChain > n`: `readTail(n+1)` returns exactly `n+1`; slice → exactly `n`;
`walkForwardOffLeadingAssistant` strips at most one. So `realized.length ∈ {n, n-1}`
and `windowIntent ∈ {n, n-1}` always. The count comparison distinguishes **one
bit** — whether audit-implied parity matches the observed first role — precisely in
the regime the 2026-08-24 incident lived in (241 messages). Any even-sized change is
invisible: equal-length branch switch, two-message race, stale audit off by 2.

The two comparisons that restore it are free and already in hand:
`audit.leaf === turns.at(-1).id`, and `turns.at(-1).role === 'user'`.

## D4 (High) — `state` is never checked

`Message.state` is `pending|streaming|complete|failed`. The gate reads `role`,
`parentId`, `deletedAt`, `id`. Never `state`. No test sets `state` to anything but
the `makeMessage` default of `complete`.

**If a failed assistant row stays on the chain:** 63 turns, send, vLLM unreachable,
`A64` marked `failed` in place, she types again → `root, U1..U63, A64(failed, ""), U65`
— 65 turns, odd, alternating, opens user, ends user. **Gate green.** The posted body
carries `{"role":"assistant","content":""}` for a turn her screen renders as a
failure notice. The model is shown a blank prior utterance of its own; the compactor
indexes it and renders `[assistant]: ` into a summary batch.

**A `streaming` row is worse for the anchor:** its content is checkpointed every
~500 ms, so its fingerprint changes between the gate reading it and the next request
re-reading it — the exact `tail_fp` drift §3.2 exists to prevent.

**If instead the failed USER turn stays** (the spec's literal words), the chain ends
on `U_fail`, her next message appends a second `user` turn, `turnsOnChain` goes even
and two consecutive user turns sit in the window. Alternation fires — a correct
refusal — but the failed pair only leaves the window after ~30 more exchanges, each
demanding an override click.

## D5 (High) — fingerprint divergence on U+FEFF, with exhibited hashes

`collapseWhitespace` (`fingerprint.ts:117-119`) uses JS `\s`; `_turn_fingerprints`
uses Python `str.split()` (`summarizer.py:728`). Same text `a﻿b`, role `user`:

```
py  8241c16cd56ac357   # U+FEFF isspace: False -> 'a﻿b'
js  1bb94d01e957ee54   # JS \s matches U+FEFF  -> 'a b'
```

U+FEFF is category `Cf`: Python says not-whitespace, JS says whitespace. A BOM or
zero-width no-break space anywhere in a turn — pasted from a Windows file, a Word
doc, a UTF-8-BOM export — desynchronises that turn's anchor. `_align_candidates`
fails, `_ASSUMED_NEW_TURNS = 2` is substituted, `window_offset` reads the wrong
slice, and the only symptom is a summary hierarchy silently behind. U+0085 (NEL)
diverges the other way. The lane doc names only `\x1c`-`\x1f`, the rarer direction.

**`_message_text` port is also incomplete.** Python's `content = m.get("content") or ""`
treats every falsy value as absent; `messageText` (`fingerprint.ts:68-73`) enumerates
`null|undefined|''|[]` and stops:

| stored `content` | `summarizer.py` | `fingerprint.ts` |
|---|---|---|
| `0` | `""` | `"0"` |
| `false` | `""` | `"false"` |
| `true` | `"True"` | `"true"` |
| `{"a":1}` | `"{'a': 1}"` | `"[object Object]"` |

Everything else matched line-by-line: the `role + "\x00"` payload, `[:16]`
truncation, `" ".join` over parts, the `if not text: image_marker` ordering, the
image predicate including the bare `"image_url" in c` case, and the marker not being
whitespace-collapsed.

## D6 (Med-High) — the "generation timeout" is total wall-clock, not idle

`transport.ts:79-82` arms one `setTimeout` before `fetch`, cleared only in `finally`.
Never re-armed inside the read loop. But `transport.ts:38-40` promises "fired when
the response body **stops arriving**" and `config.ts:23-27` calls it the generation
timeout against a hung vLLM.

Shipped behaviour: any request whose **total** duration exceeds
`DEFAULT_TIMEOUT_MS = 120_000` aborts, even while tokens arrive steadily. §3.4
records a turn costing 139.9 s of compaction alone before vLLM was contacted. A
false failure on a healthy send — the mirror of the false-refusal mode. M11 pinned
the duration, not the semantics; `timer.refresh()` per successful read (the correct
implementation) is **green** against the current suite.

## D7 (Med) — mixed streams have no stream-level state

`main.py:5469-5483`: on a mid-reply `httpx.RequestError` the compactor yields N
relay chunks of genuine prose, **then** `chatcmpl-unavail-` chunks, then `[DONE]`.
Its own comment (`main.py:5516-5521`) confirms this fires "when vLLM drops the
connection PART WAY THROUGH a reply she has already read."

`classifyChunk` is per-chunk and stateless (`sse.ts:108-127`). The obvious caller —
accumulate `delta.content`, mark `complete` on the first terminal chunk — produces
one assistant message with the outage apology welded onto the truncated real reply,
stored `complete`, re-sent next turn, fingerprinted as speech. §12 requires those be
typed notices visually distinct from assistant messages. No test uses a mixed-kind
stream.

Related: **a stream ending with no `[DONE]` and no `finish_reason` is
indistinguishable from a clean end** — `transport.ts:131-134` flushes and returns
with no "ended without terminal chunk" signal.

## D8 (Med) — the root lookup fetches the entire remaining chain on every send

`sendSet.ts:125-134` calls `readOlder(convId, fromId, ROOT_LOOKUP_LIMIT)` with
`ROOT_LOOKUP_LIMIT = 100_000`, to read one row's `id`. `readChain` bounds the
*recursion* by `limit` but returns **every visited row**, `SELECT msg.*`, `content`
included (`store.ts:515-542`). At the 2,000-message bar that is ~1,940 full rows per
outbound message — O(n) per turn, O(n²) over a conversation's life. The lane report
claims the opposite (`L2-transport.md:150-154`); the walk terminates, the result set
does not shrink. The call-counting test cannot see it. The store exposes no
"fetch this conversation's root" primitive — hand back to L1.

## D9 (Med) — `assertSendableConvId` guards emptiness, not identity

`wouldSanitizeToEmpty` (`request.ts:28-30`) only rejects ids that sanitize to empty.
`memory.py:_sanitize` strips to `[A-Za-z0-9_-]` **and truncates to 64**, then
`resolve_conv_id` returns `source="header"` — the fallback log only fires on empty.

So `"my chat"` passes, is sent verbatim, and the compactor keys memory under
`"mychat"`. The echo reports `source=header`, so §12's `conv_id_fallback` never
fires and the client has no signal. `"my chat"`, `"mychat"`, `"m ychat"` collide;
so does any pair sharing a 64-char prefix. This is the v3.0.1 orphaning defect
through a different door. Latent only because `mintConvId()` returns a UUID — but
the module documents that callers may pass `conversation.id`. Correct predicate:
`sanitize(id) === id`.

## D10 (Med) — nothing binds the gate's output to what is posted

Three couplings by convention only: `GateSuccess.turns` → `buildChatCompletionBody`
(which re-checks only non-empty and last-is-user, not count/contiguity/alternation/
tombstones); `GateSuccess.convId` → `streamChatCompletion` (unrelated parameters, no
assertion the window was built from that conversation); `systemMessage` (checked only
for `role === 'system'`, not `parentId === null`, not matching `convId`).

Also: **§4 rule 8 is unimplemented** — no request-kind marker anywhere in
`request.ts`. And `streamChatCompletion` takes `body: string`, so the "one kind of
call, no `messages: []`" guarantee lives a layer above the only function that
touches the network; the lane's own test drives `messages: []` straight through it
(`transport.test.ts:35`).

## D11 (Med) — `n` has no ceiling, no validation, no configuration

`loadTransportConfig` carries `baseUrl`, `apiKey`, `timeoutMs` and no window field,
so D2's "behind a runtime switch with a full-history setting" is unimplemented.
`runGate(chain, id, 100000)` on a 241-message conversation returns **`ok: true`**
with `sentCount: 240` — §4 rule 2's "sending full history is a spec violation," and
the gate has no opinion. `computeWindowIntent(65, 0)` returns **`-1`**;
`computeWindowIntent(5, -3)` returns `-3`, which then propagates into a
`context_truncated` notice and `OverrideRecord.originalWindowIntent`.

## D12 (Med) — the receipt cannot say "refused, nothing was sent"

`buildReceipt` reads `windowIntent`/`sentCount` off any outcome including a
`context_truncated` failure (`receipt.ts:85-91`), so a refused send produces
`{sentCount: 59, sentUnderOverride: false, messagesAdmitted: null}` —
indistinguishable from a normal send whose echo header was absent. Minor: the
fixture `makeSuccessGate()` returns `turns: []` with `sentCount: 59`
(`receipt.test.ts:26-36`), self-inconsistent, and `buildReceipt` reports it without
complaint.

---

## Mutation-log audit

**Hold up:** M1, M2, M3, M4, M9, M10, M12 (re-derived the Python cross-check vectors;
they discriminate).

**Not earned:** M11 (pins duration, not semantics — see D6). M6 (fix is right, but
`applyOverride`'s `systemMessage` is still unasserted).

**False claim:** "twelve cycles, covering every source file in this lane at least
once" (`L2-transport.md:441-442`). Six of nine. **`config.ts` has no test file and no
mutation** — its validation branch never executes, so `COMPACTOR_CLIENT_TIMEOUT_MS=0.5`
is unguarded. **`index.ts` is imported by no test**, so a missing re-export in "the
whole public surface a route handler needs" is invisible.

**Surviving mutations, in priority order:**

1. `if (turns.length !== windowIntent)` → `Math.abs(...) > 1` (`sendSet.ts:244`).
   **Green.** No test fails on a ±1 mismatch. Widening by one is the single most
   dangerous relaxation possible, because ±1 is exactly the parity quantity the
   design turns on.
2. `applyOverride`: `systemMessage: failure.systemMessage ?? null` → `null`
   (`sendSet.ts:338`). **Green** — neither override test asserts it. An override
   send would lose the persona.
3. `computeWindowIntent`'s even-`startIndex` path is never integration-tested
   (fixtures are 5, 60, 65, 501).
4. `splitSseBlocks`'s `/\r?\n\r?\n/` → `/\n\n/`. Green — every fixture uses `\n\n`.
5. `extractDataPayload`'s `.replace(/^ /, '')` → `.trim()`. Green, despite the doc
   comment claiming "not a trim."
6. **The byte-stability test is vacuous** (`fingerprint.test.ts:98-123`): `send2` is
   `JSON.parse`→`JSON.stringify` on a bare string, a no-op, and the test's own
   comment concedes it. `assert.deepEqual(anchor1, anchor2)` compares a value to
   itself. §3.2 names this as *the corrected test*. A U+FEFF vector would have
   caught D5 and does not exist.

---

## What could not be defeated

`applyOverride` genuinely does not recompute. `messagesAdmitted` genuinely never
falls back to `sentCount` (M10 well-aimed). The wire body never round-trips
`content`. Id-prefix classification cannot be defeated by model output — vLLM ids
are `chatcmpl-` + hex, and `cmd-`/`rejected-`/`unavail-` all require non-hex
characters; content is never read; chunk boundaries cannot split an id.
`chainFromCurrent` and `readChain` walk the same population (both count tombstoned
rows), so the subtraction is sound. Rule 6 holds — no path rebuilds or resends
`body`. `X-Conversation-Id` is always sent and never empty.
