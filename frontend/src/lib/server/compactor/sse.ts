// SSE parsing and stream-shape classification. FRONTEND_PLAN.md §3.4 is the
// authority here, not FRONTEND_SPEC.md §5 (that section is titled "verified
// against the code" and is wrong in several places against `fix/v3.1.4` —
// §3.4 documents the corrections; this module follows §3.4).
//
// On success the compactor is a RAW BYTE RELAY of vLLM's own chunks
// (`aiter_raw()`, main.py) — this module never re-renders a relayed chunk;
// classifyChunk() below parses a chunk's JSON only to decide `kind` and
// `terminal`, and callers that need to forward bytes unchanged should keep
// the original block text (see ClassifiedChunk's own comment on `chunk`).
//
// Three compactor-authored shapes REPLACE the relay for specific failures,
// and they are NOT symmetrical — classify by id prefix, never by the emoji
// in the text:
//
//   | shape         | id prefix            | terminates                          | machine-readable |
//   |---------------|-----------------------|--------------------------------------|-------------------|
//   | slash command | chatcmpl-cmd-         | finish_reason: "stop"               | no (id only)      |
//   | 4xx rejection | chatcmpl-rejected-    | finish_reason:"error" + `error` obj | yes               |
//   | 5xx/unreachable | chatcmpl-unavail-   | finish_reason:"stop", no `error` obj| no (id only) — a live gap |
//
// The unavailable shape's missing `error` object is a documented, live gap
// (FRONTEND_PLAN.md §7's "error object on the chatcmpl-unavail- shape" ask)
// — an outage is machine-indistinguishable from a real reply UNLESS the
// caller also checks the id prefix, which is exactly why this module makes
// prefix-matching the primary signal rather than a fallback.

import type { ClassifiedChunk, NormalizedError, StreamShapeKind, StreamShapeSummary } from './types.js';

export const STREAM_ID_PREFIX = {
	slash_command: 'chatcmpl-cmd-',
	rejected: 'chatcmpl-rejected-',
	unavailable: 'chatcmpl-unavail-'
} as const;

export function classifyChunkId(id: unknown): StreamShapeKind {
	if (typeof id !== 'string') return 'relay';
	if (id.startsWith(STREAM_ID_PREFIX.slash_command)) return 'slash_command';
	if (id.startsWith(STREAM_ID_PREFIX.rejected)) return 'rejected';
	if (id.startsWith(STREAM_ID_PREFIX.unavailable)) return 'unavailable';
	return 'relay';
}

/** Splits an accumulated text buffer into complete SSE event blocks
 *  (separated by a blank line — `\n\n` or `\r\n\r\n`) plus whatever
 *  partial block remains at the end for the next chunk of bytes to
 *  complete. Callers feed each arriving byte chunk in, decoded to text,
 *  concatenated onto the previous remainder — see transport.ts. */
export function splitSseBlocks(buffer: string): { blocks: string[]; remainder: string } {
	const parts = buffer.split(/\r?\n\r?\n/);
	const remainder = parts.pop() ?? '';
	return { blocks: parts, remainder };
}

/** One SSE block can carry multiple `data:` lines (per the SSE spec, they
 *  concatenate, joined by `\n`) — the compactor's own chunks never do
 *  this (one `data: {json}` line per block), but vLLM's raw relay is not
 *  under this project's control, so this stays general rather than
 *  assuming one line. Returns null for a block with no `data:` line at
 *  all (a bare comment/keep-alive block, `:` per the SSE spec, or a blank
 *  block from a trailing separator). */
export function extractDataPayload(block: string): string | null {
	const dataLines = block
		.split(/\r?\n/)
		.filter((line) => line.startsWith('data:'))
		.map((line) => line.slice(5).replace(/^ /, ''));
	if (dataLines.length === 0) return null;
	return dataLines.join('\n');
}

export type SseParsedEvent =
	| { isDone: true; raw: string }
	| { isDone: false; raw: string; classified: ClassifiedChunk | null; parseError?: string };

/** Parses one `data:` payload. `raw` is always the exact text this was
 *  built from — a caller that needs to relay bytes unchanged (the success
 *  path) should hold onto the ORIGINAL block bytes fed to
 *  splitSseBlocks/extractDataPayload, never reconstruct them from the
 *  parsed `classified.chunk`, which this module does not guarantee
 *  round-trips key order or number formatting. `classified: null` with a
 *  `parseError` means the payload was not valid JSON — happens on `[DONE]`
 *  (handled separately, `isDone: true`) and, in principle, on a malformed
 *  upstream chunk; callers should treat that as "forward the bytes, do not
 *  attempt to interpret them" rather than a crash. */
export function parseSseEvent(payload: string): SseParsedEvent {
	if (payload === '[DONE]') {
		return { isDone: true, raw: payload };
	}
	let parsed: unknown;
	try {
		parsed = JSON.parse(payload);
	} catch (err) {
		return { isDone: false, raw: payload, classified: null, parseError: String(err) };
	}
	if (typeof parsed !== 'object' || parsed === null) {
		return { isDone: false, raw: payload, classified: null, parseError: 'parsed payload is not an object' };
	}
	return { isDone: false, raw: payload, classified: classifyChunk(parsed as Record<string, unknown>) };
}

function firstChoice(chunk: Record<string, unknown>): Record<string, unknown> | undefined {
	const choices = chunk.choices;
	if (!Array.isArray(choices) || choices.length === 0) return undefined;
	const c = choices[0];
	return typeof c === 'object' && c !== null ? (c as Record<string, unknown>) : undefined;
}

export function classifyChunk(chunk: Record<string, unknown>): ClassifiedChunk {
	const kind = classifyChunkId(chunk.id);
	const choice = firstChoice(chunk);
	const finishReason = choice?.finish_reason;
	const terminal = finishReason !== undefined && finishReason !== null;

	let error: ClassifiedChunk['error'];
	const rawError = chunk.error;
	if (rawError && typeof rawError === 'object') {
		const e = rawError as Record<string, unknown>;
		error = {
			message: typeof e.message === 'string' ? e.message : JSON.stringify(e),
			type: typeof e.type === 'string' ? e.type : undefined,
			code: typeof e.code === 'string' ? e.code : undefined,
			detail: typeof e.detail === 'string' ? e.detail : undefined
		};
	}

	return { kind, chunk, terminal, error };
}

function relayDeltaContent(chunk: Record<string, unknown>): string {
	const choice = firstChoice(chunk);
	const delta = choice?.delta;
	if (typeof delta !== 'object' || delta === null) return '';
	const content = (delta as Record<string, unknown>).content;
	return typeof content === 'string' ? content : '';
}

/** Gate remediation D7 (docs/lanes/L2-gate-findings.md). Consumes a stream
 *  of parsed SSE events (typically `transport.ts`'s `streamChatCompletion`
 *  output, but any sync or async iterable of `SseParsedEvent` works — see
 *  the test suite for both shapes) and reports the mixed-stream signal a
 *  per-chunk caller cannot see on its own: the first non-relay kind
 *  encountered, how much genuine relay prose came before it, and whether
 *  the stream ever reached a terminal chunk or `[DONE]` at all.
 *
 *  Deliberately STOPS accumulating `relayText` and stops advancing
 *  `relayContentStoppedAt` the instant a non-relay kind is seen — nothing
 *  after that point is treated as "the reply," which is exactly what keeps
 *  a caller from welding an outage apology onto truncated real prose into
 *  one assistant message (main.py's own documented failure mode). A
 *  malformed/unparseable event (`classified: null`) is skipped for
 *  classification purposes but does not reset or advance anything —  it is
 *  not evidence of either shape. */
export async function reduceStreamShape(
	events: AsyncIterable<SseParsedEvent> | Iterable<SseParsedEvent>
): Promise<StreamShapeSummary> {
	let firstNonRelayKind: StreamShapeKind | null = null;
	let relayContentStoppedAt = 0;
	let relayText = '';
	let sawTerminal = false;

	for await (const ev of events) {
		if (ev.isDone) {
			sawTerminal = true;
			continue;
		}
		if (!ev.classified) continue; // a parse error is not a shape signal either way

		if (firstNonRelayKind === null) {
			if (ev.classified.kind === 'relay') {
				relayContentStoppedAt += 1;
				relayText += relayDeltaContent(ev.classified.chunk);
			} else {
				firstNonRelayKind = ev.classified.kind;
			}
		}

		if (ev.classified.terminal) sawTerminal = true;
	}

	return {
		firstNonRelayKind,
		relayContentStoppedAt,
		relayText,
		endedWithoutTerminal: !sawTerminal
	};
}

/** Normalizes a non-streaming (or pre-stream) HTTP failure body into one
 *  shape. FRONTEND_PLAN.md §3.4: "Two different error envelopes exist:
 *  OpenAI-shaped `{"error":{...}}` and `{"detail":...}` (plus FastAPI's
 *  422). Handle both; a client that parses one renders a raw object at
 *  the user." `detail` can itself be a string (the app-wide `UnsafeConvId`
 *  handler's 400) or an array of FastAPI's own validation-error objects
 *  (the stock 422) — both are folded into one readable `message` here so
 *  a caller never has to branch on the shape of `detail` itself. */
export function parseErrorEnvelope(status: number, bodyText: string): NormalizedError {
	const fallback = (): NormalizedError => ({
		status,
		envelope: 'unparseable',
		message: bodyText.trim().length > 0 ? bodyText.slice(0, 2000) : `HTTP ${status} with an empty body`
	});

	let parsed: unknown;
	try {
		parsed = JSON.parse(bodyText);
	} catch {
		return fallback();
	}
	if (typeof parsed !== 'object' || parsed === null) return fallback();
	const obj = parsed as Record<string, unknown>;

	if (obj.error && typeof obj.error === 'object') {
		const err = obj.error as Record<string, unknown>;
		return {
			status,
			envelope: 'openai',
			message: typeof err.message === 'string' ? err.message : JSON.stringify(err),
			code: typeof err.code === 'string' ? err.code : undefined,
			type: typeof err.type === 'string' ? err.type : undefined
		};
	}

	if ('detail' in obj) {
		const detail = obj.detail;
		const message = typeof detail === 'string' ? detail : JSON.stringify(detail);
		return { status, envelope: 'detail', message };
	}

	return fallback();
}
