// The actual HTTP call to the compactor. FRONTEND_PLAN.md §3.4: "the
// client's own server is therefore the only auth boundary in the system"
// — this module runs server-side only (never imported by client code; see
// docs/lanes/L2-transport.md for why nothing here uses SvelteKit's public
// `$env`), and it owns two things the compactor deliberately does not:
//
//   1. The generation timeout. The compactor's own httpx client is
//      `read=None` — no timeout on the read side at all, so a hung vLLM
//      hangs the stream indefinitely. This module's AbortController is
//      the only thing standing between that and a browser tab that spins
//      forever with no error.
//
//      Gate remediation D6 (docs/lanes/L2-gate-findings.md): this is a
//      genuine IDLE timeout, not a total-wall-clock one — the timer is
//      REFRESHED (`timer.refresh()`) after the connection is established
//      and again after every successful `reader.read()`, so a healthy
//      reply that takes a long time in AGGREGATE (§3.4 records a turn
//      costing 139.9s of compaction alone before vLLM was even contacted)
//      is never aborted while tokens keep arriving. Only a genuine GAP of
//      `timeoutMs` with NOTHING arriving trips it — a hung stream is the
//      failure mode this exists to catch; a slow one is not.
//
//   2. Distinguishing "HTTP 200" from "vLLM actually answered." §3.4:
//      "HTTP 200 is committed before vLLM is contacted" — a 200 here
//      proves only that the compactor accepted the shape of the request,
//      nothing about the backend. Backend-side failure arrives as one of
//      the three synthesized SSE shapes (sse.ts), not as a non-2xx
//      status, and this module does not try to guess at the HTTP layer
//      which one is coming — it hands every event to the caller
//      classified, and lets the caller (L4) decide what a
//      'rejected'/'unavailable'/'slash_command' event means for the UI.
//
// Mock the compactor's HTTP surface in tests via `fetchImpl` — this module
// never assumes a live backend, and none of its own tests require one
// (see tests/compactor/transport.test.ts).

import { buildRequestHeaders } from './request.js';
import { extractDataPayload, parseErrorEnvelope, parseSseEvent, splitSseBlocks, type SseParsedEvent } from './sse.js';
import type { TransportConfig } from './config.js';
import type { NormalizedError } from './types.js';

export class UpstreamHttpError extends Error {
	constructor(public readonly normalized: NormalizedError) {
		super(normalized.message);
		this.name = 'UpstreamHttpError';
	}
}

/** Fired when the response body stops arriving before `timeoutMs`
 *  elapses — the client-owned substitute for the timeout the compactor's
 *  own httpx client explicitly does not set on the read side. Distinct
 *  from any other network failure so a caller can render "the model is
 *  taking too long" rather than a generic connection error. */
export class GenerationTimeoutError extends Error {
	constructor(timeoutMs: number) {
		super(`generation exceeded the client-owned timeout of ${timeoutMs}ms`);
		this.name = 'GenerationTimeoutError';
	}
}

export interface StreamChatCompletionParams {
	convId: string;
	/** The exact JSON text from request.ts's buildChatCompletionBody —
	 *  never re-parsed or re-serialized here. */
	body: string;
	config: TransportConfig;
	/** Injectable for tests — see the header comment. Defaults to the
	 *  global `fetch`. */
	fetchImpl?: typeof fetch;
}

/** POSTs the request and yields every SSE event, classified, as it
 *  arrives. Never mutates `body` and never issues a second request —
 *  FRONTEND_SPEC.md §4 rule 6, "the client must not mutate the window
 *  mid-stream," is trivially true of this function because it has no
 *  code path that could rebuild or resend the body once the request is
 *  in flight. Throws `UpstreamHttpError` for a non-2xx response (the
 *  request-shape failures the compactor rejects before ever calling
 *  vLLM — `empty_messages`, `unparseable_body`, the 422 cases, etc. —
 *  see FRONTEND_PLAN.md §3.4's corrections) and `GenerationTimeoutError`
 *  if the stream stalls past `config.timeoutMs`. */
export async function* streamChatCompletion(
	params: StreamChatCompletionParams
): AsyncGenerator<SseParsedEvent, void, unknown> {
	const headers = buildRequestHeaders(params.convId, params.config.apiKey);
	const fetchFn = params.fetchImpl ?? fetch;

	let timedOut = false;
	const controller = new AbortController();
	const timer = setTimeout(() => {
		timedOut = true;
		controller.abort();
	}, params.config.timeoutMs);

	try {
		let res: Response;
		try {
			res = await fetchFn(`${params.config.baseUrl}/v1/chat/completions`, {
				method: 'POST',
				headers,
				body: params.body,
				signal: controller.signal
			});
		} catch (err) {
			if (timedOut) throw new GenerationTimeoutError(params.config.timeoutMs);
			throw err;
		}

		// D6: the connection is up — refresh the idle budget so however long
		// the connect itself took does not eat into the read loop's own
		// allowance.
		timer.refresh();

		if (res.status >= 400) {
			const text = await res.text();
			throw new UpstreamHttpError(parseErrorEnvelope(res.status, text));
		}

		if (!res.body) return;

		const reader = res.body.getReader();
		const decoder = new TextDecoder();
		let buffer = '';
		while (true) {
			let step: { done: boolean; value?: Uint8Array };
			try {
				step = await reader.read();
			} catch (err) {
				if (timedOut) throw new GenerationTimeoutError(params.config.timeoutMs);
				throw err;
			}
			// D6: refresh on every SUCCESSFUL read, not just the first one.
			// This is the whole of what turns the timeout into a genuine
			// IDLE timeout: the deadline is always "timeoutMs since the last
			// byte arrived," never "timeoutMs since the request started."
			timer.refresh();
			if (step.done) break;
			buffer += decoder.decode(step.value, { stream: true });
			const { blocks, remainder } = splitSseBlocks(buffer);
			buffer = remainder;
			for (const block of blocks) {
				const payload = extractDataPayload(block);
				if (payload === null) continue;
				yield parseSseEvent(payload);
			}
		}
		// A well-formed stream always ends its final block with the blank
		// line splitSseBlocks splits on, so `buffer` is normally empty
		// here. Flushing a non-empty leftover anyway means a stream that
		// ends without a trailing blank line still yields its last event
		// instead of silently dropping it.
		const trailingPayload = extractDataPayload(buffer);
		if (trailingPayload !== null) {
			yield parseSseEvent(trailingPayload);
		}
	} finally {
		clearTimeout(timer);
	}
}
