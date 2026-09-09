# L2 — Transport (F8's send-set gate, the byte-stability contract, SSE classification, and the client half of the receipt)

`frontend/src/lib/server/compactor/**` and `frontend/tests/compactor/**`. Does
not touch the store, the routes, the components, the Python compactor, or
`docker-compose.tests.yml` — a new `docker-compose.compactor-tests.yml` and
`testfixtures/compactor-unit/**` carry this lane's own Docker test service
instead of editing either of those.

---

## 1. Module surface

```
frontend/src/lib/server/compactor/
  types.ts        ChainReader interface; GateOutcome/GateSuccess/GateFailure;
                  OverrideRecord; StreamShapeKind; ClassifiedChunk;
                  NormalizedError; ReceiptSnapshot
  sendSet.ts      DEFAULT_WINDOW_N; computeWindowIntent(); runGate();
                  applyOverride()                              — F8
  fingerprint.ts  messageText(); imageOnlyMarker(); turnFingerprint();
                  turnFingerprints(); trailingAnchorFingerprints();
                  utf8SurrogatepassEncode(); sha256Hex16()      — the byte-
                                                                   stability contract
  request.ts      mintConvId(); assertSendableConvId(); buildRequestHeaders();
                  buildChatCompletionBody()
  sse.ts          classifyChunkId(); classifyChunk(); splitSseBlocks();
                  extractDataPayload(); parseSseEvent(); parseErrorEnvelope()
  transport.ts    streamChatCompletion(); UpstreamHttpError;
                  GenerationTimeoutError
  config.ts       loadTransportConfig(); TransportConfig; DEFAULT_TIMEOUT_MS
  receipt.ts      buildReceipt(); readMessagesAdmitted(); PROPOSED_ADMITTED_HEADER
  index.ts        barrel export — the whole public surface a route handler needs
```

Every function takes a `ChainReader` (a narrowed, structural view of the real
`Store` — `getConversation`/`auditConversation`/`readTail`/`readOlder`) rather
than a concrete `Store`, or takes plain data. Nothing in this lane writes to
the store: `appendMessage`, `updateMessageState`, `appendStreamDelta`,
`selectLeaf`, `tombstoneSubtree` are never called from anywhere in
`compactor/**`. This is a hard boundary, not an accident — see §3 (U-2) for
where the write this lane depends on is expected to happen, and why it is
someone else's call site.

---

## 2. The send-set algorithm and how the gate verifies it (F8)

`runGate(chain, convId, n)` is the pre-send gate. It:

1. Calls `chain.auditConversation(convId)`. A missing conversation is
   `not_found`. `!audit.pass` is `chain_unsound` (the store's own five-tuple
   attached, verbatim — this is FRONTEND_SPEC.md §4.1's `chain_corrupt`
   condition, not this lane's `context_truncated`; a caller should route it to
   that notice, F14, not this one) and the gate stops here — there is no
   coherent chain to build a window from.
2. Computes `turnsOnChain = audit.chainFromCurrent - 1` (the synthetic system
   root subtracted) and `windowIntent = computeWindowIntent(turnsOnChain, n)`
   — see §2.1 below for what that function actually returns and why.
3. Does **explicit selection over the chain from the current leaf** — never
   accepts a caller-supplied message array — via exactly one or two store
   reads (§2.2), producing a `(systemMessage, turns)` pair.
4. Verifies the realized pair against five properties, named individually so
   no adjacent one is substituted (FRONTEND_SPEC.md §4.1's own discipline):
   **present** (the root was located, is genuinely a system message, is the
   conversation's actual root, is not tombstoned), **contiguous** (every
   turn's `parentId` is the previous turn's `id`), **in order** (implied by
   contiguity — there is only one order a valid parent chain admits),
   **alternation intact** (opens on `'user'`, strictly alternates, no stray
   `'system'` or tombstoned message inside the window), and **count equal to
   `windowIntent`**.
5. Any failure produces `context_truncated` with every reason listed and the
   **actual realized send set still attached** — not discarded — for D4's
   override path. No failure recomputes anything; §2.3 covers the override
   itself.

### 2.1 Why `window_intent` is not a bare `min(N, turnsOnChain)`

FRONTEND_PLAN.md §3.2 says, literally: `window_intent = min(N, turns on the
current chain)`, then "verify... count equal to `window_intent`." Taken at
face value this conflicts with FRONTEND_SPEC.md §4 rule 3 (never two
consecutive same-role turns; the window must open on `'user'`) for a reason
the plan's own N=60 derivation never mentions: **N is even.**

This lane resolves U-2 (FRONTEND_PLAN.md §3.5 — "when is the user row written,
relative to the pre-send gate") as: **the user's new turn is appended to the
chain, becoming the current leaf, *before* the gate runs.** (See §3 below for
the full reasoning and what this implies for the write side, which is not this
lane's code to write.) Under that resolution, a healthy, strictly-alternating
conversation's `turnsOnChain` is **always odd** at the moment the gate runs —
the chain starts at the first user turn and ends at the one just typed,
awaiting a reply.

Combine "always odd" with N=60 (even): whenever `turnsOnChain > N`, the naive
"last N turns" slice starts at 0-based offset `turnsOnChain - N` =
odd − even = **odd**, i.e. an `'assistant'` turn — the exact shape
`_cap()` (`pipelines/conversation_id_header.py:195-222`) exists to fix by
walking forward off the leading assistant turn. That fix is **mandatory** —
sending the unfixed window gets a 400 from vLLM's chat template — so for
**every** healthy conversation past 60 messages, the correct send set is 59
turns, not 60, forever, deterministically, for as long as N stays even.

If `window_intent` were recorded as the literal `min(N, turnsOnChain) = 60`
and compared against that mandatory 59-turn realization, **every turn past
message 60 would fail verification and demand a manual D4 override click** —
not an occasional anomaly but the steady-state UX of every long conversation.
Nothing in the plan's Phase 1–3 acceptance criteria describes that as
intended, and D4's own framing ("an override the daily user can reach") reads
as an escape hatch for a rare event, not a button on every message.

**This lane's resolution:** `computeWindowIntent(turnsOnChain, n)` already
folds in the alternation-preserving adjustment:

```
if (turnsOnChain <= n) return turnsOnChain;
const startIndex = turnsOnChain - n;
return startIndex % 2 === 0 ? n : n - 1;
```

so the routine one-turn parity trim is what "intent" means, and
`context_truncated` is reserved for what it is actually for — a realized send
set that differs from what the chain **actually** supports (a race, corruption,
or a shortfall bigger than the expected one turn). This is a **plan
correction**, made explicit rather than silently encoded — see §5 (errors
found in the plan), item 4. `computeWindowIntent` is a pure function of two
numbers and does not itself verify anything; §2.2/§2's step 4 does that
against the **actual fetched data**, which is what keeps this from being a
tautology (see the next paragraph).

**Why this is a real check, not a tautology.** `windowIntent` is computed
*before* the chain is re-read for its actual content, from the audit alone.
The realized `turns.length` comes from a **separate** later read. If the chain
changed between the two — a concurrent append, a branch switch, someone
editing the chain via `psql` mid-request — the two numbers can genuinely
disagree, and `context_truncated` fires correctly. `tests/compactor/sendSet.test.ts`'s
"a race between the audit read and the window read" test constructs exactly
this: an `auditConversation` that reports a stale, much-shorter chain than the
`readTail` call actually finds.

### 2.2 The read pattern — one or two store reads, never more

`selectFromChain` calls `chain.readTail(convId, n + 1)` exactly once. The
extra `+1` exists purely to detect, in that one round trip, whether the whole
active chain (root included) already fit inside the page:

- If the oldest message in that page has `parentId === null`, the whole chain
  was fetched — no second read.
- Otherwise (a long conversation), the oldest of the `n+1` fetched messages is
  dropped (it was only lookback for the detection above) and the persona root
  is fetched with **one** `chain.readOlder(convId, firstPage.cursor,
  ROOT_LOOKUP_LIMIT)` call. `ROOT_LOOKUP_LIMIT` (100,000) is a safety ceiling,
  not a hop budget — `readChain`'s recursive CTE (`store.ts`) stops the moment
  it reaches a row with `parent_id IS NULL` regardless of how large the
  caller's `limit` is, so this costs nothing extra for any realistic
  conversation; it only prevents a corrupt/cyclic chain from running away.

`tests/compactor/sendSet.test.ts`'s "never issues more than two chain reads"
test asserts this bound directly (a 501-turn fixture, counting calls through a
wrapping `ChainReader`), and a second test asserts a short conversation issues
**zero** `readOlder` calls at all.

The mandatory shape fix (`walkForwardOffLeadingAssistant`) mirrors `_cap()`'s
own degenerate case exactly: if stripping leading assistant turns would leave
nothing, it gives up and returns the untouched (shape-illegal) window instead
of an empty one — left unfixed, and caught by the alternation check in step 4
of §2, never silently sent.

### 2.3 D4's override — copies, never recomputes

`applyOverride(failure: GateFailure)` only accepts a `context_truncated`
failure (throws for `chain_unsound`/`not_found` — there is no coherent send
set to override for either). Every field on the returned `OverrideRecord` is
copied from the failure, not recomputed:

```ts
return {
  sentUnderOverride: true,
  convId: failure.convId,
  originalWindowIntent: failure.windowIntent ?? 0,   // the ORIGINAL intent
  systemMessage: failure.systemMessage ?? null,
  turns: failure.turns ?? [],
  sentCount: failure.sentCount ?? 0,
  reasons: failure.reasons
};
```

There is no call to `runGate` (or anything else) inside `applyOverride` — the
function has no branch that could recompute anything, which is the point:
"recompute and resend" is the defect wearing a consent dialog. Two tests exist
specifically because a first version of the "copies verbatim" test did **not**
catch a mutation that substituted `sentCount` for `windowIntent`: the original
alternation-corruption fixture happened to have `windowIntent === sentCount`
(5 and 5), so swapping the two fields was invisible to it. The fix is a second
test using the race fixture, where `windowIntent = 3` and `sentCount = 65` are
deliberately different — see §4 (mutation log), M6, for the full story; this
is exactly the kind of gap the handoff's "a test only counts if it has been
watched to fail" standard exists to catch.

---

## 3. U-2 resolved: when the user's turn is written

FRONTEND_PLAN.md §3.5 leaves this open explicitly and hands it to whoever
writes F8, because guessing wrong "means re-deriving F3's CAS semantics after
F8 is written." This lane's resolution, made necessary by §2.1's parity
argument and consistent with the store's own state machine:

- **The user's new message is appended immediately** (`appendMessage`,
  `state: 'complete'`), becoming the current leaf, **before** this module is
  ever called. This call site is not in `compactor/**` — it belongs to
  whatever route/session code assembles a turn (out of this lane's ownership;
  `src/routes/**` is explicitly not-mine).
- The store's `pending → streaming → complete|failed` state machine describes
  the **assistant's** reply lifecycle, not the user's — a user's typed message
  has no "streaming" phase. So FRONTEND_SPEC.md §4.1's "on failure, the user
  turn is retained in place... marked failed" is read here as a slightly loose
  paraphrase for "the assistant's row is marked failed; the user's turn that
  prompted it is untouched and retryable" — not as an instruction to ever
  write the user's own row with `state: 'failed'`.
- Consequence: when `runGate` refuses (`context_truncated`), **nothing new has
  been written at all** — no assistant row exists yet, so "no new root, no
  pointer move" (§4.1) holds trivially, with zero chain mutation to undo.
- A retry, or D4's override, proceeds to append a `pending` assistant message
  (state machine's own job) and call `streamChatCompletion`.

This is recorded here, not silently assumed, because it is exactly the kind of
decision the plan says costs a re-derivation if guessed wrong — and it drives
§2.1's whole argument (`turnsOnChain` is always odd at gate time only because
of this resolution).

---

## 4. The byte-stability contract and its test

`fingerprint.ts` ports two functions from `compactor/summarizer.py` byte for
byte: `_message_text` → `messageText`, `_image_only_marker` →
`imageOnlyMarker`, combined in `turnFingerprint`/`turnFingerprints` to
reproduce `_turn_fingerprints`. Every design choice traces to
`_turn_fingerprints`'s own contract: hash `role + "\x00" + " ".join(text.split())`,
falling back to the image marker when the extracted text is empty, truncated
to 16 hex chars.

**Verified against a live Python run, not just read from the source.** Five
vectors were computed by actually running the real functions
(`hashlib.sha256(...).hexdigest()[:16]`, Python 3.14.7) and cross-checked
against this module's TypeScript output:

| Case | Python | TypeScript |
|---|---|---|
| plain text, trailing `\n` | `71312c09790b940e` | `71312c09790b940e` |
| same text, different whitespace runs | `71312c09790b940e` (identical) | `71312c09790b940e` |
| image-only content part | `216f49657b130d12` | `216f49657b130d12` |
| a lone (unpaired) UTF-16 surrogate | `62ddfa15932339b6` | `62ddfa15932339b6` |
| multi-part text content | `c9c29a19bf502d37` | `c9c29a19bf502d37` |

The surrogate case additionally cross-checks the **raw bytes**, not just the
final hash: `f"user\x00abc{chr(0xD800)}def".encode("utf-8","surrogatepass").hex()`
== `7573657200616263eda080646566` on both sides. `utf8SurrogatepassEncode`
exists because Node's own `Buffer.from(str, 'utf8')` silently substitutes
U+FFFD for a lone surrogate — a message containing one (the compactor's own
`unpaired_surrogate` 400 is evidence such bodies exist) would otherwise
fingerprint differently on the two sides of the wire, with no visible symptom
beyond a drifting summary hierarchy weeks later. Iterating the string with
`for...of` (code-point iteration) hands a lone surrogate through as itself,
which the ordinary 3-byte UTF-8 branch then encodes exactly as
`surrogatepass` does — see the function's own comment for why this needs no
special-casing at all.

`trailingAnchorFingerprints` is the test named in the brief: given the same
conversation re-derived two different ways (one JSON-serialized directly, one
round-tripped through `JSON.parse`/re-`JSON.stringify`, simulating what
Postgres's own `jsonb` normalization does to key order and escapes), the
trailing-4 fingerprints are asserted `deepEqual`. An image-only turn is
fingerprinted through `_image_only_marker`'s TypeScript twin, not as the empty
string its request shape reduces to — a dedicated test asserts this
specifically, matching the brief's explicit warning that skipping it makes
"every image turn read as new."

**A known, documented limitation:** Python's `str.split()` (no arguments)
treats a slightly broader set of characters as whitespace than JS's `\s`
regex class (e.g. the ASCII file/group/record/unit separators `\x1c`–`\x1f`).
A chat turn containing one of those four rare control characters could
fingerprint differently between the two sides. This was not fixed (it would
require hand-rolling Python's exact `str.isspace()` table) because the failure
mode is the existing degraded path (`_ASSUMED_NEW_TURNS`), not data loss or a
crash — flagged rather than silently accepted.

---

## 5. Stream-shape classification (§3.4)

`classifyChunkId(id)` matches **only** the id prefix, in this fixed order:
`chatcmpl-cmd-` → `slash_command`, `chatcmpl-rejected-` → `rejected`,
`chatcmpl-unavail-` → `unavailable`, anything else → `relay`. A dedicated test
constructs a chunk whose id is an ordinary vLLM-shaped id but whose role is
`'assistant'` with apology-shaped text, and asserts it still classifies as
`relay` — the literal test for "never classify by the emoji in the text."

`classifyChunk` additionally reads `terminal` (any `finish_reason` other than
`null`/absent) and `error` (present only when a top-level `error` object
exists — contractually only on the `rejected` shape; `unavailable` carries
none, a documented **live gap**, not a bug in this module, per FRONTEND_PLAN.md
§7's "error object on the `chatcmpl-unavail-` shape" ask).

`splitSseBlocks`/`extractDataPayload`/`parseSseEvent` handle the actual wire
mechanics: blocks separated by a blank line, `data:` lines extracted (leading
single space stripped, matching the SSE spec, not a trim), `[DONE]` recognized
without attempting `JSON.parse` on it, and a malformed payload degrading to a
typed `parseError` rather than throwing. `transport.test.ts`'s relay test feeds
bytes split at an arbitrary offset **inside** a JSON object (not aligned to any
event boundary) to prove reassembly works generally, not only when byte chunks
happen to align with SSE lines.

`parseErrorEnvelope(status, bodyText)` normalizes the two shapes named in
§3.4 (`{"error":{...}}` and `{"detail": ...}`, the latter covering both the
`UnsafeConvId` handler's plain string and FastAPI's stock 422 array) plus a
non-JSON/neither-shape fallback — never rendering a raw parsed object at the
user, per the plan's explicit warning about a client that parses one shape
and chokes on the other.

---

## 6. The receipt (client half) — §3.3/§12

`buildReceipt` assembles:

| Field | Source |
|---|---|
| `messagesOnActivePath` | `audit.chainFromCurrent` |
| `messagesInConversation` | `audit.total` |
| `branchCount` | **always `null`** — see below |
| `windowIntent` / `sentCount` | the `GateOutcome` |
| `sentUnderOverride` | caller-supplied |
| `messagesAdmitted` / `admittedSource` | response headers, read defensively |

**The one property that matters most:** `messagesAdmitted` is never, under any
code path, set to `sentCount`. `readMessagesAdmitted` returns
`{ messagesAdmitted: null, admittedSource: 'not_reported' }` whenever the
proposed header is absent, empty, or fails to parse as a non-negative integer
— which is every response today, since (per FRONTEND_PLAN.md §3.4) the
compactor sets no custom response header anywhere. This is deliberately the
single most heavily mutation-tested line in this lane (§7, M10) because it is
the literal shape of the 2026-08-28 incident: substituting the sent count for
the admitted count is "accurate and worthless."

**`branchCount` is honestly `null`, not a best-effort guess.** The store's
public API exposes ancestor-walks only (`readTail`/`readOlder`) plus
`auditConversation`'s aggregate counts — there is no `getChildren`/sibling
query anywhere. Computing "how many branches exist" would mean either
reimplementing a descendant walk outside the `Store` class (this lane's brief:
"use it, do not reimplement it") or a new store method neither exists today.
`null` here is a first-class unknown, not a fabricated zero — see §8 (handed
back), item 1.

**Proposed header name**, not a settled contract: `x-context-messages-admitted`
(`PROPOSED_ADMITTED_HEADER`). The plan specifies the *fields* the echo must
carry but names no actual HTTP header (§3.3/§15); this lane had to pick
something to write a working defensive reader against. A header rename on the
server side degrades to `not_reported`, identically to a wholly absent header
— there is no special "the name changed" failure mode, by design.

---

## 7. The wire body and the raw-content obligation

`buildChatCompletionBody` builds the request as **raw JSON text**, never a
parsed-then-reserialized object. `message.content` (the store's raw-JSON-text
wire form) is spliced directly into `{"role":...,"content":<rawContent>}` via
template-string concatenation — there is no `JSON.stringify(JSON.parse(...))`
anywhere in this file. A test asserts this against a content string containing
a literal `A` escape sequence: a parse/reserialize round trip normalizes
that to the literal character `A`, changing the bytes while leaving the parsed
*value* unchanged — exactly the transformation FRONTEND_PLAN.md §3.2 forbids,
and exactly the class of change `fingerprint.ts`'s own header comment names as
irrelevant to *that* module (which only ever reads, never re-emits, content)
but load-bearing for *this* one (which writes to the wire).

The same function refuses, unconditionally, to build a request that is not
"one real user turn": zero turns, or a window whose newest message is not
`'user'`, both throw rather than producing `messages: []` or a task-shaped
call. This is FRONTEND_PLAN.md's "exactly one kind of outbound call" enforced
structurally — there is no parameter shape this function accepts that could
express a title/tag/follow-up call, and no separate function anywhere in this
lane that could construct one.

`request.ts` also carries the `X-Conversation-Id` sanitizer parity check
(`wouldSanitizeToEmpty`/`assertSendableConvId`, mirroring `memory.py`'s
`_sanitize` exactly — same disallowed-character class, same 64-char cap) and
`mintConvId()` (a real `crypto.randomUUID()` — UUIDv4). **A note on UUID
version:** FRONTEND_SPEC.md §4 rule 1 says "conv_id (UUIDv4)," but the store's
own `conversation.id` (what a natural caller already has in hand) is a
UUIDv7 (`../store/uuid7.ts`), and `Conversation` has no separate field to hold
a second, v4-specific identity. This lane's `assertSendableConvId` accepts any
string that survives the compactor's own sanitizer — the actual hard
requirement — rather than insisting on the version-4 nibble specifically,
since the compactor's sanitizer does not check it either. See §5's "handed
back" note if a dedicated, separate memory-key identity is ever wanted.

---

## 8. Transport: the client-owned timeout and the 200-proves-nothing rule

`streamChatCompletion` is the only place in this lane that touches the
network, and every call is mockable via an injected `fetchImpl` — no test
anywhere in `tests/compactor/**` makes a real HTTP request.

Two obligations from FRONTEND_PLAN.md §3.4, both load-bearing:

1. **The client owns the generation timeout.** The compactor's own `httpx`
   client sets `read=None` — no read-side timeout at all — so a hung vLLM
   hangs the stream indefinitely from the server's own perspective.
   `streamChatCompletion`'s `AbortController` + `setTimeout` is the only thing
   that ever ends such a request. Two distinct failure points are tested
   separately: a hang **before** any response headers arrive, and a stall
   **after** headers arrive but mid-body (a chunk is yielded, then nothing
   further — the exact shape of "vLLM stopped producing tokens without ever
   sending `finish_reason`"). Both tests carry an explicit `{ timeout: 2000 }`
   node-test bound — see §9 (mutation log), M11, for why an unbounded test
   would have let a real regression (the timeout firing 1000× too slowly)
   pass silently, just slowly.
2. **HTTP 200 proves nothing about the backend.** This module does not try to
   infer backend health from the status line; a 200 with a body is handed,
   chunk by chunk, to the SSE classifier (§5), and it is the caller's job
   (L4, the notice-rendering layer) to decide what a `rejected`/`unavailable`
   classified chunk means for the UI. A non-2xx status *is* meaningful here
   (it means the compactor rejected the request's **shape** before ever
   calling vLLM — `empty_messages`, malformed bodies, the 422 cases) and
   throws a typed `UpstreamHttpError` carrying the normalized envelope from
   §5, distinct from anything that could arrive mid-stream.

`buildRequestHeaders`/`assertSendableConvId` run **before** any fetch is
attempted — a test confirms the mock `fetchImpl` is never even called when
the `convId` would sanitize to empty.

---

## 9. The mutation log

Per FRONTEND_HANDOFF.md §6 / the plan's own method: mutate, run, watch red,
restore from an in-memory/on-disk copy (never `git checkout`), watch green.
Twelve cycles, covering every source file in this lane at least once. Every
cycle below was actually run against the compiled suite (locally during
development, and the whole suite again in Docker at the end — §10).

| # | File / property mutated | Mutation | Result before restore | Caught by |
|---|---|---|---|---|
| M1 | `sendSet.ts` — `computeWindowIntent` | inverted the parity branches (`n-1`/`n` swapped) | **RED** — 3 tests failed | `computeWindowIntent` unit tests, the 65-turn "not context_truncated" test, and the "≤2 reads" test's count assertion |
| M2 | `fingerprint.ts` — `collapseWhitespace` | disabled whitespace collapsing entirely | **RED** — cross-check vector v1/v2 diverged | `turnFingerprint` Python cross-check test |
| M3 | `fingerprint.ts` — image-only fallback | removed `if (!text) text = imageOnlyMarker(...)` | **RED** — v3 (image-only) vector diverged | same cross-check test |
| M4 | `fingerprint.ts` — `utf8SurrogatepassEncode` | corrupted the 3-byte branch's high-byte shift (`>>12` → `>>11`) | **RED** — v4 vector and the dedicated byte-hex test both diverged | cross-check test + dedicated surrogatepass byte test |
| M5 | `sendSet.ts` — alternation check | `if (turns[i].role === turns[i-1].role)` → `if (false)` | **RED** — 2 tests failed | "broken alternation" test, `applyOverride` verbatim test (which depends on a `context_truncated` failure existing at all) |
| M6 | `sendSet.ts` — `applyOverride` | `originalWindowIntent: failure.windowIntent` → `failure.sentCount` | **GREEN — not caught on first attempt.** The existing fixture had `windowIntent === sentCount` (5 and 5) by coincidence. **Fixed**: added a second test using the race fixture (`windowIntent=3`, `sentCount=65`, deliberately different) — re-run, now **RED**. | the new "never silently substituted" test |
| M7 | `sse.ts` — `classifyChunkId` | swapped the `rejected`/`unavailable` return values | **RED** — both shape-specific `classifyChunk` tests diverged | `classifyChunk` rejection/unavailable tests |
| M8 | `sse.ts` — `parseErrorEnvelope` | disabled the OpenAI-shape branch (guarded with an always-false runtime condition) | **RED** — 2 tests failed | the OpenAI-envelope test and `transport.test.ts`'s 4xx-error test (which reads `.code` off the parsed envelope) |
| M9 | `request.ts` — `messageToWireJson` | reintroduced `JSON.stringify(JSON.parse(rawContent))` | **RED** | the escape-normalization test (added specifically because the FIRST content test — a string with accents/newlines — did not, itself, distinguish a round trip from a verbatim splice; see the test file's own comment) |
| M10 | `receipt.ts` — `buildReceipt` | `messagesAdmitted` → `messagesAdmitted ?? sentCount` | **RED** — reproduces the exact 2026-08-28 shape | "THE property" test |
| M11 | `transport.ts` — timeout duration | `timeoutMs` → `timeoutMs * 1000` | **RED, but only after tightening the test.** With no bound on the test itself, both timeout tests still eventually resolved (20s/30s) and PASSED — slow, not caught. **Fixed**: added `{ timeout: 2000 }` to both tests. Re-run: now fails in ~2s each. | the two generation-timeout tests, once bounded |
| M12 | `sendSet.ts` — `chain_unsound` routing | `if (!audit.pass)` guarded with an always-false-at-runtime condition | **RED** — 2 tests failed | both `chain_unsound` tests and `applyOverride`'s refusal test |

Every cycle: backed up with `cp <file> <scratch>/<file>.orig` (never `git
checkout`), mutated via a small Node script matching an exact anchor string
(refusing to proceed silently if the anchor was not found), ran the full
suite, recorded the result above, restored with `cp <scratch>/<file>.orig
<file>`, re-ran to confirm green, and diffed against the backup to confirm
byte-identical restoration. No `git` command was run at any point in this
lane's work.

**M6 and M11 are the two findings worth reading twice.** Both are cases where
the mutation *should* have been caught and initially was not — not because
the production code was wrong, but because the test's own fixture happened to
be insensitive to that particular corruption. Both were fixed by strengthening
the test (a new fixture for M6, an explicit timeout bound for M11), and both
mutations were then re-run and confirmed red. This is the literal value of the
"watched to fail" standard: a test that has never actually failed is a claim,
not evidence, and two of twelve claims in this lane were false on first
inspection.

---

## 10. Verification

```bash
# All 76, Linux, in Docker — no live backend, no Postgres:
docker compose -f docker-compose.compactor-tests.yml run --rm --build compactor-unit

# one file/substring, mapped to node's --test-name-pattern:
docker compose -f docker-compose.compactor-tests.yml run --rm compactor-unit --only turnFingerprint
```

Confirmed in this session:

- **76/76 pass, 0 fail, 0 skipped**, run in Docker on `node:24-slim`
  (`docker-compose.compactor-tests.yml`, service `compactor-unit`), both after
  the initial build and again after every mutation-restore cycle above.
- The L1 store suite (`docker-compose.tests.yml`'s `client-unit`) still passes
  **54/54**, unaffected — confirmed by actually running it in this session,
  not assumed from "I didn't touch those files."
- `--only <substring that matches nothing>` exits `3` (SKIP), never folded
  into a pass; `--only <substring with no matching test title>` also exits
  `3` even when the substring appears in a **comment** (see the finding
  immediately below).
- `compactor-unit` runs with `network_mode: none` and no Postgres sidecar —
  genuinely offline, unlike `client-unit`'s own documented tension with that
  setting (see `docs/lanes/L1-store.md` §7, D-L1-3).

**A real gap found and fixed in this lane's OWN test harness while verifying
it, worth flagging for `client-unit` too (not fixed there — that file is not
mine to touch):** the first version of this lane's `--only` pre-check
(`grep -qFl -- "$ONLY_PATTERN" $TEST_FILES`, copied verbatim from
`testfixtures/client-unit/entrypoint.sh`) checks whether the pattern appears
**anywhere** in the compiled file — including doc comments, which `tsc`
preserves by default. `--only fingerprint` (lowercase) matches no actual test
title in this suite (every title is `turnFingerprint`/`imageOnlyMarker`/etc.,
capital-letter camelCase, and the match is case-sensitive) but DOES appear in
this lane's own header-comment prose in every file — so the untightened check
let all 6 files through as "matching," and node's test runner then reported 6
passing "tests," each one being an entire file substituted as a trivially
passing pseudo-test with **zero real assertions executed**, silently. Fixed
here by requiring the pattern to co-occur on a line that also contains
`test(` — closing the gap for this lane's own entrypoint
(`testfixtures/compactor-unit/entrypoint.sh`). A second, subtler version of
the same bug appeared while fixing the first: grep's own `<filename>:` prefix
(added automatically for a multi-file grep) recreated the exact false-positive
via the filename `fingerprint.test.js` itself — closed by piping through
`cat` first so no filename prefix ever reaches the second grep. `client-unit`
carries the ORIGINAL, untightened version of this check today.

---

## 11. Decisions this lane made that were not settled for it

**D-L2-1. U-2 resolved: the user's turn is written before the gate runs, by
the caller, as an ordinary `complete` message.** §3 above. Load-bearing for
§2.1's whole parity argument; flagged because guessing wrong here is exactly
what FRONTEND_PLAN.md warns costs a re-derivation of F3's CAS semantics.

**D-L2-2. `window_intent` is the alternation-preserving achievable count, not
the literal `min(N, turnsOnChain)`.** §2.1. The single largest interpretive
decision in this lane — without it, the gate would demand a manual override
on every turn past message 60 of every conversation.

**D-L2-3. `ChainReader` is a narrowed structural interface, not the concrete
`Store`.** Lets this lane's tests run against a lightweight in-memory fake
(`tests/compactor/helpers/fakeChain.ts`) with no Postgres sidecar, while
production code passes the real `Store` instance unchanged (it satisfies the
interface structurally — no adapter exists or is needed). The fake's
`auditConversation` mirrors `audit_conversation()`'s own PASS formula closely
(same five-tuple, same boolean formula) specifically so it stays a faithful
stand-in rather than a shortcut that could drift from what the real function
actually enforces.

**D-L2-4. `assertSendableConvId` accepts any UUID-shaped (or otherwise
sanitizer-surviving) string, not strictly a version-4 UUID.** §7. The
compactor's own sanitizer does not check the version nibble either; insisting
on it here would reject the store's own UUIDv7 `conversation.id`, which is
what a real caller most naturally has in hand.

**D-L2-5. `messagesAdmitted`'s header name is a proposal
(`x-context-messages-admitted`), not a settled contract.** §6. The plan
specifies the fields, not an actual header name; this lane needed one to write
a defensive reader against, and picked one that degrades identically to
"absent" if the real implementation names it differently.

**D-L2-6. A new `docker-compose.compactor-tests.yml` + `testfixtures/compactor-unit/**`,
not an edit to `docker-compose.tests.yml`.** This lane's brief lists
`docker-compose.tests.yml` as explicitly not-to-touch, while FRONTEND_PLAN.md
§6 says "add a service or extend the existing `client-unit` pattern." Read as:
follow the *pattern* (the Dockerfile/entrypoint shape both existing services
already use), in a *new* file, rather than editing the one file this lane was
told not to.

**D-L2-7. `DEFAULT_TIMEOUT_MS = 120_000`.** FRONTEND_PLAN.md §3.4 names the
obligation ("the client owns that timeout") but not a number. 120s is this
lane's own choice, overridable via `COMPACTOR_CLIENT_TIMEOUT_MS`, on the
reasoning that it should comfortably exceed a normal (if slow) generation
without making a genuinely hung backend feel responsive for two minutes.

---

## 12. What I could not verify

- **Whether the real compactor's SSE shapes match `sse.ts`'s model exactly.**
  Every classification test in this lane is against a **constructed** chunk
  matching `main.py`'s own chunk-builder functions
  (`_vllm_unreachable_stream_chunks`, `_request_rejected_stream_chunks`,
  `build_synthetic_completion_stream`), read from the source, not observed
  from a live compactor process (out of scope per this lane's brief — "mock
  the compactor's HTTP surface, do not require a live backend"). If a future
  compactor change alters those functions' shape without a corresponding spec
  update, this lane's tests would not catch the drift.
- **The Unicode-whitespace divergence between Python's `str.split()` and JS's
  `\s`** (§4) — documented as a known, low-severity gap, not fixed, because
  fixing it exactly would mean hand-porting Python's `str.isspace()` character
  table and the failure mode is already-degraded behavior, not data loss.
- **Whether `ChainReader`'s narrowed interface stays structurally compatible
  with `Store` if L1 changes a method signature.** TypeScript's structural
  typing means this is caught at compile time for any signature change, but
  this lane has no runtime test that imports the real `Store` class (by
  design — no Postgres dependency). A `tsc --noEmit` pass against both
  modules together would catch a drift; this lane did not wire that check
  into CI (it would require importing `../store/**` into this lane's own
  build, which the interface boundary in D-L2-3 deliberately avoids).
- **The real, produced end-to-end wiring** (a SvelteKit `+server.ts` route
  calling `runGate` → `buildChatCompletionBody` → `streamChatCompletion` →
  the store's write methods) — that file lives under `src/routes/**`, not
  owned by this lane, and was not written or exercised here. This lane
  verifies its own module surface is internally correct and well-typed for
  that caller to use; it cannot verify the caller does not misuse it.

---

## 13. Things wrong in the plan, found and corrected here

The task brief said three had already been found and a fourth would be
normal. This lane adds at least two more, beyond the parity issue in §2.1
(which is the headline one):

1. **§2.1's parity bug** (the big one — literal `min(N, turnsOnChain)`
   compared for exact equality against a post-alternation-fix realization
   would refuse nearly every send past 60 messages). Corrected by folding the
   fix into `window_intent`'s own definition.
2. **The plan calls `conv_id` "UUIDv4" (§4 rule 1) while the store it also
   specifies mints UUIDv7 for `conversation.id`** (D5, `uuid7.ts`), with no
   separate field anywhere to hold a second, version-4-specific identity.
   Either the header is meant to be a fresh, separately-minted UUIDv4 that
   some caller persists elsewhere (not specified where), or the intent was
   "a UUID" generically and "v4" is leftover phrasing from before the store's
   own id scheme was decided. This lane accepts either by validating against
   the compactor's actual sanitizer rule rather than the version nibble —
   see D-L2-4 — but the ambiguity itself is worth the owner's eyes.
3. **The store's public API has no way to cheaply answer "how many branches
   does this conversation have"** — needed for the receipt's `branchCount`
   field (§12's five-field list) but not obtainable from any of
   `getConversation`/`auditConversation`/`readTail`/`readOlder`. Handed back
   as `null` rather than guessed; see §14, item 1.
4. **U-2, though explicitly flagged as open rather than a plan error, still
   deserves naming here**: resolving it wrong (assistant-turn-first) would
   have inverted §2.1's entire parity argument. The plan is right to flag it
   as high-stakes; this lane's resolution is recorded in §3, not merely
   assumed.

---

## 14. Handed back

- **A cheap sibling/children-count query on the store**, or an explicit
  decision that `branchCount` stays unavailable in Phase 1. Today it is
  `null` in every receipt this lane produces — honest, but a permanent gap in
  the five-field receipt list unless L1 adds the primitive.
- **The received-context echo itself** (FRONTEND_PLAN.md §3.3/§15) — this
  lane's `readMessagesAdmitted` is ready to consume it the moment it exists,
  against the proposed header name in §6/D-L2-5; confirming or renaming that
  header is the one piece of coordination actually needed.
- **A decision on `conv_id`'s UUID version** (§13, item 2) — whether the
  compactor-facing identity should be a separately-minted UUIDv4 (and, if so,
  where it is persisted, since `Conversation` has no field for it today) or
  whether reusing the store's own UUIDv7 `conversation.id` is the accepted
  design. This lane works correctly either way (D-L2-4) but the ambiguity is
  unresolved at the data-model level.
- **A courtesy flag for `testfixtures/client-unit/entrypoint.sh`**: its
  `--only` pre-check shares the false-positive-via-comment gap this lane found
  and fixed in its own copy (§10). Not fixed there — out of this lane's
  ownership — but worth the store lane's attention next time that file is
  touched.

---

## 15. Summary

F8 (the checked send-set and pre-send gate), the byte-stability fingerprint
contract, SSE stream-shape classification, the raw-content wire-body builder,
the client-owned transport timeout, and the client half of the receipt are all
built under `frontend/src/lib/server/compactor/**`, tested under
`frontend/tests/compactor/**` (76 tests, 0 skipped), and verified on Linux in
Docker via a new `docker-compose.compactor-tests.yml` / `compactor-unit`
service that needs neither a live compactor nor a Postgres sidecar. The L1
store suite (54/54) was re-run and confirmed unaffected. Twelve mutation
cycles were run against this lane's own source, all watched red before being
restored (never via `git checkout`) and watched green again; two of them
(M6, M11) initially escaped detection and were closed by strengthening the
test rather than declaring victory early. The plan's `window_intent` formula,
taken literally, would make the gate refuse the majority of sends in any
conversation longer than 60 messages — corrected here, with the derivation
recorded in §2.1 for the Opus gate to accept or overrule.
