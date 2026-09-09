// Builds the outbound `/v1/chat/completions` request: the wire body and
// the headers. FRONTEND_PLAN.md §3.2's byte-stability obligation applies
// here in full: "Never re-render content for the wire from the display
// form. Store the wire form; send that." `message.content` (../store/types.ts)
// is already the exact wire-form JSON text for the OpenAI "content" field's
// VALUE — this module's whole discipline is to splice that text into the
// request body VERBATIM, never `JSON.parse` it and never `JSON.stringify`
// the parsed result back. A parse/reserialize round trip can reorder object
// keys or re-render escapes even when the parsed VALUE is unchanged, and
// that is exactly the transformation §3.2 says drifts the compactor's
// tail_fp anchor — see fingerprint.ts's header comment for the other half
// of that same contract (which DOES parse `content`, but only to extract
// text for a one-way hash that is never sent anywhere).

import type { GateSuccess } from './types.js';
import type { Role } from '../store/types.js';
import { verifyChainShape } from './sendSet.js';

/** Mirrors compactor/memory.py's `_sanitize` exactly: `.strip()` (JS
 *  `.trim()`) then `[^A-Za-z0-9_-]` stripped, THEN length-capped at 64 —
 *  in that order, matching `cleaned[:_CONV_ID_MAX_LEN]` running AFTER the
 *  regex substitution, never before. A UUID (v4 or v7 — see
 *  docs/lanes/L2-transport.md's note on which this project actually sends)
 *  passes through unchanged. */
const CONV_ID_DISALLOWED = /[^A-Za-z0-9_-]/g;
const CONV_ID_MAX_LEN = 64;

export function sanitizeConvId(raw: string): string {
	if (!raw) return '';
	const cleaned = raw.trim().replace(CONV_ID_DISALLOWED, '');
	return cleaned.slice(0, CONV_ID_MAX_LEN);
}

export function wouldSanitizeToEmpty(convId: string): boolean {
	return sanitizeConvId(convId).length === 0;
}

/** Gate remediation D9 (docs/lanes/L2-gate-findings.md). The ORIGINAL
 *  version of this function only rejected an id that sanitizes to EMPTY —
 *  `memory.py`'s own fallback log line ("received X-Conversation-Id header
 *  but value sanitized to empty") only fires on THAT case, so any other
 *  sanitize-changes-the-value id ("my chat" -> "mychat", any two ids
 *  sharing a 64-char prefix) passed silently: the header would be sent
 *  verbatim, the compactor would key memory under the SANITIZED form, and
 *  `resolve_conv_id` would report `source="header"` — no signal anywhere
 *  that the identity the client believes it is using and the identity the
 *  compactor is actually keying memory under have diverged. This is the
 *  v3.0.1 orphaning defect through a different door. The correct
 *  predicate, per the findings: `sanitize(id) === id`. */
export function assertSendableConvId(convId: string): void {
	const sanitized = sanitizeConvId(convId);
	if (sanitized.length === 0) {
		throw new Error(
			`convId ${JSON.stringify(convId)} would sanitize to empty on the compactor side ` +
				`(memory.py's _sanitize strips to [A-Za-z0-9_-]) and silently fall through to the ` +
				`hash-fingerprint fallback — refusing to send it as X-Conversation-Id at all`
		);
	}
	if (sanitized !== convId) {
		throw new Error(
			`convId ${JSON.stringify(convId)} does not survive memory.py's _sanitize unchanged ` +
				`(sanitizes to ${JSON.stringify(sanitized)}) — sending it would silently key memory under a ` +
				`DIFFERENT identity than this client believes it is using (D9; the v3.0.1 orphaning defect ` +
				`through a different door), with resolve_conv_id still reporting source="header" and no ` +
				`signal that anything diverged`
		);
	}
}

/** Mints a fresh UUIDv4 for a new conversation's compactor-facing
 *  identity. FRONTEND_SPEC.md §4 rule 1 calls this "conv_id (UUIDv4)";
 *  see docs/lanes/L2-transport.md for why this module does not INSIST the
 *  header be exactly a v4 (the store's own `conversation.id` is a
 *  UUIDv7 — ../store/uuid7.ts — and there is today no separate field to
 *  hold a second, v4-specific identifier). Use this when minting a NEW
 *  identity; `assertSendableConvId` is the check every send actually
 *  enforces, and accepts either version. */
export function mintConvId(): string {
	return crypto.randomUUID();
}

export function buildRequestHeaders(convId: string, apiKey: string | undefined): Record<string, string> {
	assertSendableConvId(convId);
	const headers: Record<string, string> = {
		'Content-Type': 'application/json',
		'X-Conversation-Id': convId
	};
	// FRONTEND_PLAN.md D1: "Write the server transport-agnostic... always
	// send Authorization: Bearer when a key is configured" — sent even
	// though FRONTEND_PLAN.md §3.4 confirms no auth exists anywhere in
	// this tree today. Sending it unconditionally when configured (never
	// conditioned on some other flag) is what keeps the eventual split
	// into a separate container a deploy-config change rather than a code
	// change.
	if (apiKey) {
		headers['Authorization'] = `Bearer ${apiKey}`;
	}
	return headers;
}

/** One message, rendered onto the wire EXACTLY as
 *  `{"role":"<role>","content":<rawContent>}` — `rawContent` is spliced in
 *  verbatim, never parsed. `role` is one of the three DB-checked values
 *  ('system'|'user'|'assistant'; see the migration's CHECK constraint) so
 *  JSON.stringify-ing it is safe and never the thing being protected here. */
function messageToWireJson(role: Role, rawContent: string): string {
	return `{"role":${JSON.stringify(role)},"content":${rawContent}}`;
}

/** Gate remediation D10 (docs/lanes/L2-gate-findings.md) §4 rule 8. Every
 *  outbound call this project ever sends is "one real user turn" — never
 *  a background title/tag/follow-up call (FRONTEND_PLAN.md §3.4: "no
 *  title calls, no tag calls, no follow-up calls, no `messages: []`").
 *  §4 rule 8 asks that task traffic carry an explicit request-kind marker
 *  "so the compactor can refuse memory writes for them" — sent here even
 *  though `main.py` does not read `metadata.request_kind` today (verified
 *  against FRONTEND_PLAN.md §3.4's own inventory of what the compactor
 *  inspects: `messages`, `stream`, `model`, `max_tokens`,
 *  `metadata.{chat_id,conversation_id}`, `continue_final_message`,
 *  `add_generation_prompt` — everything else, `request_kind` included, is
 *  forwarded to vLLM untouched today). Reported as a live gap in
 *  docs/lanes/L2-transport.md's "Gate remediation" section rather than
 *  silently assumed read. */
const REQUEST_KIND = 'conversation';

/** Gate remediation D10. `buildChatCompletionBody` now demands the ACTUAL
 *  `GateSuccess` a real `runGate` call produced — not a bespoke
 *  `{systemMessage, turns}` pair a caller could assemble from unrelated
 *  sources — closing the finding's "three couplings by convention only."
 *  This is the "token buildChatCompletionBody demands": there is exactly
 *  one function in this lane that produces a `GateSuccess`
 *  (`sendSet.ts`'s `runGate`), so plumbing that SAME object through to
 *  this function is what ties the verified window to what gets posted,
 *  rather than trusting that whoever called this passed the right fields. */
export interface ChatCompletionRequestParams {
	model: string;
	gate: GateSuccess;
	stream: boolean;
	maxTokens?: number;
}

/** Builds the exact JSON text of the request body — a raw string, not a
 *  parsed object — so no serializer anywhere in this call path ever
 *  touches a message's `content`. Throws rather than building a body for
 *  anything that is not "one real user turn": FRONTEND_PLAN.md's "Exactly
 *  one kind of outbound call... No title calls, no tag calls, no
 *  follow-up calls, no `messages: []`." An empty `turns` array or a
 *  window that does not end on the user's own newest message is
 *  categorically one of those forbidden shapes, not a variant of a normal
 *  send — so this function has no parameter that could express them, and
 *  refuses rather than silently sending an empty or task-shaped
 *  `messages` array.
 *
 *  D10 also runs `verifyChainShape` again here, defensively, on
 *  `gate.turns` — the exact same structural check `sendSet.ts` ran to
 *  produce this `GateSuccess` in the first place. That is deliberate
 *  redundancy: it means this function does not merely TRUST that its
 *  input came from a real gate pass, it re-derives enough of the same
 *  answer to refuse a hand-crafted or corrupted `GateSuccess`-shaped value
 *  even without a fresh audit in hand. */
export function buildChatCompletionBody(params: ChatCompletionRequestParams): string {
	const { gate } = params;
	if (gate.systemMessage.role !== 'system') {
		throw new Error('buildChatCompletionBody: gate.systemMessage.role must be "system"');
	}
	if (gate.systemMessage.parentId !== null) {
		throw new Error(
			'buildChatCompletionBody: gate.systemMessage is not the conversation root (parentId !== null) — ' +
				'D10, refusing to post a persona that was never verified as the root'
		);
	}
	if (gate.systemMessage.convId !== gate.convId) {
		throw new Error(
			`buildChatCompletionBody: gate.systemMessage belongs to conversation ${gate.systemMessage.convId}, ` +
				`not ${gate.convId} — D10, refusing a mismatched persona/conversation pairing`
		);
	}
	if (gate.turns.length === 0) {
		throw new Error(
			'buildChatCompletionBody: refusing to build a request with zero turns — this project sends ' +
				'exactly one kind of outbound call (a real user turn), never `messages: []`'
		);
	}
	const last = gate.turns[gate.turns.length - 1];
	if (last.role !== 'user') {
		throw new Error(
			`buildChatCompletionBody: the newest message in the window is '${last.role}', not 'user' — ` +
				'refusing to send a window that is not awaiting a reply to a real user turn'
		);
	}
	if (gate.turns.length !== gate.sentCount) {
		throw new Error(
			`buildChatCompletionBody: gate.turns.length (${gate.turns.length}) !== gate.sentCount ` +
				`(${gate.sentCount}) — D10, refusing a self-inconsistent GateSuccess`
		);
	}
	const shapeReasons = verifyChainShape(gate.convId, gate.turns);
	if (shapeReasons.length > 0) {
		throw new Error(
			`buildChatCompletionBody: the gate's window failed a defensive structural re-check at the ` +
				`point of posting — ${shapeReasons.join('; ')}`
		);
	}

	const allMessages = [gate.systemMessage, ...gate.turns];
	const messagesJson = allMessages.map((m) => messageToWireJson(m.role, m.content)).join(',');

	const fields = [
		`"model":${JSON.stringify(params.model)}`,
		`"messages":[${messagesJson}]`,
		`"stream":${params.stream ? 'true' : 'false'}`,
		// §4 rule 8 — see REQUEST_KIND's own comment.
		`"metadata":{"request_kind":${JSON.stringify(REQUEST_KIND)}}`
	];
	if (params.maxTokens !== undefined) {
		fields.push(`"max_tokens":${JSON.stringify(params.maxTokens)}`);
	}
	return `{${fields.join(',')}}`;
}
