// Shared types for the L2 transport module. Imports FROM the store's own
// published types (../store/types.ts) — never redefines them — per this
// lane's brief: "use [the store's API], do not reimplement it."

import type { AuditResult, Conversation, Message, ReadPage } from '../store/types.js';

/** The minimal capability this module needs from a store. The concrete
 *  `Store` class (../store/store.ts) satisfies this structurally — nothing
 *  here is a new abstraction over the store, it is a narrower VIEW of its
 *  existing public methods so this module's tests can supply a lightweight
 *  fake chain (see tests/compactor/helpers/fakeChain.ts) without standing
 *  up Postgres, while production code passes the real `Store` instance
 *  directly with no adapter/wrapper in between. See
 *  docs/lanes/L2-transport.md for why the gate does not need
 *  `appendMessage`/`selectLeaf`/etc: F8 only ever READS the chain to build
 *  and verify a send set; writing the new turn onto the chain is the
 *  caller's job (see U-2's resolution in that doc), before the gate runs. */
export interface ChainReader {
	getConversation(convId: string): Promise<Conversation | null>;
	auditConversation(convId: string): Promise<AuditResult | null>;
	readTail(convId: string, limit: number): Promise<ReadPage>;
	readOlder(convId: string, beforeId: string, limit: number): Promise<ReadPage>;
	/** Gate remediation D8 (docs/lanes/L2-gate-findings.md): a one-row,
	 *  indexed lookup of the conversation's synthetic persona root —
	 *  replaces the O(n) `readOlder(convId, cursor, 100_000)` pattern
	 *  sendSet.ts used to fall back to for a long conversation's root
	 *  (backed by `message_one_root_per_conv`'s own partial unique index;
	 *  see ../store/store.ts and migrations/0001_init.sql). */
	getRoot(convId: string): Promise<Message | null>;
}

/** Why a gate attempt failed to produce a sendable window, named so a
 *  caller can route each to the right notice (FRONTEND_SPEC.md §12)
 *  without this module rendering any UI itself. */
export type GateFailureKind = 'not_found' | 'chain_unsound' | 'context_truncated';

export interface GateSuccess {
	ok: true;
	convId: string;
	/** FRONTEND_PLAN.md §3.2: min(N, turns_on_the_current_chain), already
	 *  adjusted for the mandatory alternation-preserving shape fix — see
	 *  sendSet.ts's header comment for why this module records the
	 *  ADJUSTED figure rather than the raw formula. This is the number a
	 *  receipt and an override both compare every future send against. */
	windowIntent: number;
	/** The synthetic system/persona root — always included, never subject
	 *  to windowing (FRONTEND_SPEC.md §4 rule 2 / the `_cap()` reference's
	 *  "system messages are never dropped"). */
	systemMessage: Message;
	/** The realized turn window, oldest-first, chronologically contiguous,
	 *  strictly alternating, opening on 'user'. Length equals
	 *  `windowIntent` whenever this is `ok: true` — that equality IS what
	 *  makes it `ok: true` rather than `context_truncated`. */
	turns: Message[];
	/** Always equal to `turns.length` here; carried as its own field
	 *  (rather than making every call site write `.turns.length`) because
	 *  the failure variant below needs the identical field for the same
	 *  reason, and the receipt (receipt.ts) reads this name from both. */
	sentCount: number;
}

export interface GateFailure {
	ok: false;
	convId: string;
	kind: GateFailureKind;
	/** Present for every kind. */
	reasons: string[];
	/** Present for `chain_unsound` — the store's own audit, verbatim, so a
	 *  chain_corrupt notice (F14, not this lane) has the five-tuple to
	 *  render without a second query. */
	audit?: AuditResult;
	/** Present for `context_truncated` (and absent, meaningfully, for
	 *  `not_found`/`chain_unsound`, where no coherent send set could be
	 *  built at all). This is the send set the gate actually assembled —
	 *  not thrown away — because D4's override path sends EXACTLY this,
	 *  never a recomputation. See sendSet.ts's `applyOverride`. */
	windowIntent?: number;
	systemMessage?: Message | null;
	turns?: Message[];
	sentCount?: number;
}

export type GateOutcome = GateSuccess | GateFailure;

/** D4: "the override records both numbers... must not overwrite
 *  window_intent... the receipt shows the delta against the ORIGINAL
 *  intent... the turn is marked as sent under override." Every field
 *  here is copied verbatim from the `context_truncated` GateFailure that
 *  produced it — see sendSet.ts's `applyOverride`, which contains no
 *  branch that could recompute any of them. */
export interface OverrideRecord {
	sentUnderOverride: true;
	convId: string;
	/** The ORIGINAL window_intent from the failed gate attempt — never
	 *  recomputed, never adjusted, so the receipt's delta is always
	 *  against the number the check actually failed on. */
	originalWindowIntent: number;
	systemMessage: Message | null;
	turns: Message[];
	sentCount: number;
	/** Copied from the GateFailure so the audit trail records WHY an
	 *  override was needed, not just that one occurred. */
	reasons: string[];
}

/** The three compactor-authored SSE shapes, plus the ordinary relay.
 *  FRONTEND_PLAN.md §3.4's table. Named exactly as the id prefixes they
 *  are keyed on, so a reader matching this type against that table needs
 *  no translation step. */
export type StreamShapeKind = 'relay' | 'slash_command' | 'rejected' | 'unavailable';

export interface ClassifiedChunk {
	kind: StreamShapeKind;
	/** The parsed chunk object (chat.completion.chunk / chat.completion
	 *  shape) — never re-serialized for display; callers needing the exact
	 *  bytes should read them from the raw SSE line this was parsed from,
	 *  not reconstruct them from this object (see sse.ts's own note on
	 *  why `relay` chunks are handed back with their raw line intact). */
	chunk: Record<string, unknown>;
	/** True on the chunk that ends the stream for this shape (finish_reason
	 *  set to anything other than null, OR a `[DONE]` sentinel — see sse.ts). */
	terminal: boolean;
	/** Present only when `chunk.error` exists (the `rejected` shape,
	 *  contractually; FRONTEND_PLAN.md §3.4 — `unavailable` deliberately
	 *  carries none, which is the live gap the plan hands back). */
	error?: { message: string; type?: string; code?: string; detail?: string };
}

/** Gate remediation D7 (docs/lanes/L2-gate-findings.md). `main.py`'s own
 *  comment: on a mid-reply `httpx.RequestError`, the compactor yields N
 *  relay chunks of genuine prose, THEN `chatcmpl-unavail-` chunks, THEN
 *  `[DONE]` — "when vLLM drops the connection PART WAY THROUGH a reply she
 *  has already read." A naive per-chunk caller (accumulate delta.content,
 *  mark complete on the first terminal chunk) welds the outage apology
 *  onto the truncated real reply into ONE assistant message — §12 requires
 *  those be typed notices visually distinct from assistant messages. This
 *  is the STREAM-LEVEL summary a caller needs to avoid that: where relay
 *  content stopped, what it turned into, and whether the stream ended
 *  cleanly at all. See sse.ts's `reduceStreamShape`. */
export interface StreamShapeSummary {
	/** The first classified kind other than 'relay' encountered, or null if
	 *  the whole stream (so far, or to its end) was ordinary relay. */
	firstNonRelayKind: StreamShapeKind | null;
	/** How many RELAY chunks were yielded before `firstNonRelayKind` fired
	 *  — equivalently, the 0-based index of the first non-relay chunk.
	 *  Equal to the total relay-chunk count if the stream never left relay. */
	relayContentStoppedAt: number;
	/** The concatenated `delta.content` text from relay chunks ONLY, up to
	 *  (never past) the point relay content stopped — the genuine prose a
	 *  caller has to decide what to do with, kept separate from whatever a
	 *  `rejected`/`unavailable` shape's own text says. */
	relayText: string;
	/** True iff the stream was exhausted (the generator returned) without
	 *  EVER seeing a terminal chunk (`finish_reason` set) or a `[DONE]`
	 *  sentinel — `transport.ts`'s own flush-on-exit path produces exactly
	 *  this shape for a connection that simply closes mid-reply, which is
	 *  otherwise indistinguishable from a clean end. */
	endedWithoutTerminal: boolean;
}

/** One HTTP-level failure envelope, normalized from whichever of the
 *  compactor's two shapes (plus FastAPI's stock 422) produced it —
 *  FRONTEND_PLAN.md §3.4: "Two different error envelopes exist... Handle
 *  both; a client that parses one renders a raw object at the user." */
export interface NormalizedError {
	status: number;
	/** 'openai' — `{"error": {...}}`; 'detail' — `{"detail": "..."}` or
	 *  FastAPI's `{"detail": [...]}`; 'unparseable' — the body was not
	 *  valid JSON, or was JSON but neither shape (a raw object rendered at
	 *  the user is exactly what this type exists to prevent). */
	envelope: 'openai' | 'detail' | 'unparseable';
	message: string;
	code?: string;
	type?: string;
}

/** The client half of FRONTEND_SPEC.md §12's receipt. `messagesAdmitted`
 *  is deliberately not a plain number — see receipt.ts's header comment
 *  and FRONTEND_PLAN.md §3.3: substituting the client's own sent count
 *  for this field is the exact shape of the 2026-08-28 failure. */
export interface ReceiptSnapshot {
	convId: string;
	/** audit_conversation's chain_from_current — total messages (system
	 *  root included) on the active path, i.e. what the thread renders. */
	messagesOnActivePath: number;
	/** audit_conversation's total — every message in the conversation,
	 *  every branch, tombstoned or not. */
	messagesInConversation: number;
	/** Not available from the current store API (no children/sibling
	 *  query exists — see docs/lanes/L2-transport.md, "handed back").
	 *  `null` here means exactly that, not zero. */
	branchCount: number | null;
	windowIntent: number;
	sentCount: number;
	sentUnderOverride: boolean;
	/** `null` with `admittedSource: 'not_reported'` until the
	 *  received-context echo (FRONTEND_PLAN.md §3.3 / §15) ships as a
	 *  real response header. Never populated from `sentCount` — that
	 *  substitution is the 2026-08-28 failure verbatim. */
	messagesAdmitted: number | null;
	admittedSource: 'header' | 'not_reported';
	/** Gate remediation D12 (docs/lanes/L2-gate-findings.md): explicit
	 *  "we refused and sent nothing at all" — independent of
	 *  `messagesAdmitted`'s absence, which ALSO reads `null` for a normal
	 *  send whose echo header simply hasn't shipped yet. Before this field,
	 *  those two situations were receipt-indistinguishable: `{sentCount: 59,
	 *  sentUnderOverride: false, messagesAdmitted: null}` read exactly the
	 *  same whether 59 turns were genuinely posted or the gate refused and
	 *  the network was never touched. `false` whenever the gate succeeded
	 *  OR the caller sent anyway under a D4 override (which DID reach the
	 *  network — `sentUnderOverride` already carries that distinction). */
	refused: boolean;
	/** The gate's own reasons, present iff `refused` is true — so a receipt
	 *  can explain WHY nothing was sent without a second query. */
	refusalReasons: string[] | null;
}
