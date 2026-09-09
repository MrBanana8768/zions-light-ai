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

import type { Message, Role } from '../store/types.js';

/** Mirrors compactor/memory.py's `_CONV_ID_ALLOWED` /
 *  `_sanitize` exactly: `[^A-Za-z0-9_-]` stripped, length-capped at 64. A
 *  UUID (v4 or v7 — see docs/lanes/L2-transport.md's note on which this
 *  project actually sends) passes through unchanged; anything that would
 *  sanitize to empty is exactly the case FRONTEND_PLAN.md warns "falls
 *  through to the hash fallback with only a log line — never let that
 *  happen." This function throws rather than sending a header the
 *  compactor would silently discard. */
const CONV_ID_DISALLOWED = /[^A-Za-z0-9_-]/g;
const CONV_ID_MAX_LEN = 64;

export function wouldSanitizeToEmpty(convId: string): boolean {
	return convId.trim().replace(CONV_ID_DISALLOWED, '').length === 0;
}

export function assertSendableConvId(convId: string): void {
	if (!convId || wouldSanitizeToEmpty(convId)) {
		throw new Error(
			`convId ${JSON.stringify(convId)} would sanitize to empty on the compactor side ` +
				`(memory.py's _sanitize strips to [A-Za-z0-9_-]) and silently fall through to the ` +
				`hash-fingerprint fallback — refusing to send it as X-Conversation-Id at all`
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

export interface ChatCompletionRequestParams {
	model: string;
	systemMessage: Message;
	/** Oldest-first, strictly alternating, opening on 'user' — exactly the
	 *  shape `runGate` (sendSet.ts) guarantees on its `ok: true` branch (or
	 *  what an override explicitly accepts sending anyway; see
	 *  sendSet.ts's applyOverride). This function does NOT re-verify the
	 *  shape — that is the gate's job, done once, before this is ever
	 *  called — but it DOES refuse the one invariant that must never be
	 *  violated regardless of gate outcome: see the throw below. */
	turns: Message[];
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
 *  `messages` array. */
export function buildChatCompletionBody(params: ChatCompletionRequestParams): string {
	if (params.systemMessage.role !== 'system') {
		throw new Error('buildChatCompletionBody: systemMessage.role must be "system"');
	}
	if (params.turns.length === 0) {
		throw new Error(
			'buildChatCompletionBody: refusing to build a request with zero turns — this project sends ' +
				'exactly one kind of outbound call (a real user turn), never `messages: []`'
		);
	}
	const last = params.turns[params.turns.length - 1];
	if (last.role !== 'user') {
		throw new Error(
			`buildChatCompletionBody: the newest message in the window is '${last.role}', not 'user' — ` +
				'refusing to send a window that is not awaiting a reply to a real user turn'
		);
	}

	const allMessages = [params.systemMessage, ...params.turns];
	const messagesJson = allMessages.map((m) => messageToWireJson(m.role, m.content)).join(',');

	const fields = [
		`"model":${JSON.stringify(params.model)}`,
		`"messages":[${messagesJson}]`,
		`"stream":${params.stream ? 'true' : 'false'}`
	];
	if (params.maxTokens !== undefined) {
		fields.push(`"max_tokens":${JSON.stringify(params.maxTokens)}`);
	}
	return `{${fields.join(',')}}`;
}
