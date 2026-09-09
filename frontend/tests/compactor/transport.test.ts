// The HTTP transport — mocked entirely via `fetchImpl`, per this lane's
// brief: "Mock the compactor's HTTP surface — do not require a live
// backend." No test in this file makes a real network call.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GenerationTimeoutError, streamChatCompletion, UpstreamHttpError } from '../../src/lib/server/compactor/transport.js';
import type { TransportConfig } from '../../src/lib/server/compactor/config.js';

function baseConfig(overrides: Partial<TransportConfig> = {}): TransportConfig {
	return { baseUrl: 'http://127.0.0.1:8080', timeoutMs: 5_000, windowN: 60, ...overrides };
}

async function collect<T>(iter: AsyncGenerator<T, void, unknown>): Promise<T[]> {
	const out: T[] = [];
	for await (const v of iter) out.push(v);
	return out;
}

test('streamChatCompletion: sends the exact body, headers, and URL', async () => {
	// A boxed object, not a reassigned `let`: TypeScript's control-flow
	// narrowing for a `let` captured-and-reassigned-by-a-closure loses
	// track of the reassignment at the point of the later check in this
	// exact shape (cast-to-`typeof fetch` closure); a property on a `const`
	// object sidesteps that entirely.
	const captured: { value: { url: string; init: RequestInit } | null } = { value: null };
	const fetchImpl = (async (url: string | URL | Request, init?: RequestInit) => {
		captured.value = { url: String(url), init: init as RequestInit };
		return new Response(new ReadableStream({ start: (c) => c.close() }), { status: 200 });
	}) as typeof fetch;

	await collect(
		streamChatCompletion({
			convId: 'conv-123',
			body: '{"model":"m","messages":[],"stream":true}',
			config: baseConfig({ apiKey: 'secret' }),
			fetchImpl
		})
	);

	if (!captured.value) throw new Error('fetchImpl was never called');
	assert.equal(captured.value.url, 'http://127.0.0.1:8080/v1/chat/completions');
	assert.equal(captured.value.init.method, 'POST');
	assert.equal(captured.value.init.body, '{"model":"m","messages":[],"stream":true}');
	const headers = captured.value.init.headers as Record<string, string>;
	assert.equal(headers['X-Conversation-Id'], 'conv-123');
	assert.equal(headers['Authorization'], 'Bearer secret');
});

test('streamChatCompletion: relays and classifies a normal streaming reply, split across arbitrary byte boundaries', async () => {
	const fullText =
		'data: {"id":"chatcmpl-real","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n' +
		'data: {"id":"chatcmpl-real","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n' +
		'data: [DONE]\n\n';
	// Split at an arbitrary byte offset that lands INSIDE the first JSON
	// object, not on an event boundary — proving the block splitter
	// reassembles across chunks rather than only working when bytes
	// happen to align with SSE event lines.
	const splitAt = fullText.indexOf('"delta"');
	const part1 = fullText.slice(0, splitAt);
	const part2 = fullText.slice(splitAt);

	const fetchImpl = (async () => {
		const encoder = new TextEncoder();
		const stream = new ReadableStream<Uint8Array>({
			start(controller) {
				controller.enqueue(encoder.encode(part1));
				controller.enqueue(encoder.encode(part2));
				controller.close();
			}
		});
		return new Response(stream, { status: 200 });
	}) as typeof fetch;

	const events = await collect(
		streamChatCompletion({ convId: 'c1', body: '{}', config: baseConfig(), fetchImpl })
	);

	assert.equal(events.length, 3);
	assert.equal(events[0].isDone, false);
	if (!events[0].isDone) assert.equal(events[0].classified?.kind, 'relay');
	assert.equal(events[1].isDone, false);
	if (!events[1].isDone) assert.equal(events[1].classified?.terminal, true);
	assert.equal(events[2].isDone, true);
});

test('streamChatCompletion: a 4xx/5xx response throws UpstreamHttpError with the normalized envelope, never surfaced as a stream event', async () => {
	const fetchImpl = (async () =>
		new Response(JSON.stringify({ error: { message: 'bad body', code: 'empty_messages' } }), {
			status: 400
		})) as typeof fetch;

	await assert.rejects(
		() => collect(streamChatCompletion({ convId: 'c1', body: '{}', config: baseConfig(), fetchImpl })),
		(err: unknown) => {
			assert.ok(err instanceof UpstreamHttpError);
			assert.equal(err.normalized.code, 'empty_messages');
			return true;
		}
	);
});

test(
	'streamChatCompletion: the client owns the generation timeout — a hang before headers arrive',
	// An explicit, tight upper bound is the point of this test, not a
	// convenience: a mutation that multiplies timeoutMs by even 10x still
	// EVENTUALLY resolves and would otherwise pass (just slowly), which is
	// precisely the kind of regression a timeout exists to make impossible
	// to miss. Node fails the test outright if it runs longer than this.
	{ timeout: 2_000 },
	async () => {
		const fetchImpl = (async (_url: string | URL | Request, init?: RequestInit) => {
			return new Promise<Response>((_resolve, reject) => {
				const signal = init?.signal;
				signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
			});
		}) as typeof fetch;

		await assert.rejects(
			() =>
				collect(
					streamChatCompletion({ convId: 'c1', body: '{}', config: baseConfig({ timeoutMs: 20 }), fetchImpl })
				),
			(err: unknown) => err instanceof GenerationTimeoutError
		);
	}
);

test(
	'streamChatCompletion: the client owns the generation timeout — a stall mid-stream, after headers already arrived',
	{ timeout: 2_000 }, // see the previous test's comment on why this bound is the actual assertion
	async () => {
	// Mirrors FRONTEND_PLAN.md §3.4: the compactor's own httpx client is
	// read=None (no server-side timeout at all) — this is the exact shape
	// of "vLLM hangs after committing HTTP 200." The mock ties the abort
	// signal to the stream controller itself, which is what a real
	// fetch/undici implementation does internally; this test exercises
	// transport.ts's OWN handling of that rejection, not the mock's.
	const fetchImpl = (async (_url: string | URL | Request, init?: RequestInit) => {
		const signal = init?.signal;
		let ctrl!: ReadableStreamDefaultController<Uint8Array>;
		const stream = new ReadableStream<Uint8Array>({
			start(controller) {
				ctrl = controller;
				controller.enqueue(new TextEncoder().encode('data: {"id":"chatcmpl-real","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'));
				// then nothing further — simulates a vLLM that stopped
				// producing tokens without ever sending finish_reason.
			}
		});
		signal?.addEventListener('abort', () => {
			ctrl.error(new DOMException('Aborted', 'AbortError'));
		});
		return new Response(stream, { status: 200 });
	}) as typeof fetch;

	let sawPartial = false;
	await assert.rejects(
		async () => {
			for await (const ev of streamChatCompletion({
				convId: 'c1',
				body: '{}',
				config: baseConfig({ timeoutMs: 30 }),
				fetchImpl
			})) {
				if (!ev.isDone) sawPartial = true;
			}
		},
		(err: unknown) => err instanceof GenerationTimeoutError
	);
	assert.equal(sawPartial, true, 'the partial content that DID arrive should still have been yielded before the timeout');
	}
);

test('streamChatCompletion: refuses to send a header that would sanitize to empty, before any fetch is attempted', async () => {
	let fetchCalled = false;
	const fetchImpl = (async () => {
		fetchCalled = true;
		return new Response(null, { status: 200 });
	}) as typeof fetch;

	await assert.rejects(() =>
		collect(streamChatCompletion({ convId: '###', body: '{}', config: baseConfig(), fetchImpl }))
	);
	assert.equal(fetchCalled, false);
});

test('streamChatCompletion: a response with no body ends the generator with nothing yielded', async () => {
	const fetchImpl = (async () => new Response(null, { status: 200 })) as typeof fetch;
	const events = await collect(
		streamChatCompletion({ convId: 'c1', body: '{}', config: baseConfig(), fetchImpl })
	);
	assert.deepEqual(events, []);
});

test(
	'streamChatCompletion: D6 — a genuinely IDLE timeout: a healthy, slow-but-STEADY stream survives well past the total-duration budget',
	// Generous outer bound — the test's own real delays sum to ~320ms; this
	// is a "did not hang" ceiling, not the property under test (that's the
	// explicit per-gap timeoutMs below).
	{ timeout: 5_000 },
	async () => {
		// timeoutMs=150. Four chunks, each arriving ~80ms after the reader
		// asks for it — no single GAP between arrivals ever exceeds 150ms,
		// but the stream's TOTAL duration (~320ms+) does. §3.4 records a
		// real turn costing 139.9s of compaction alone before vLLM was ever
		// contacted — this is that shape in miniature. The ORIGINAL
		// (total-wall-clock) implementation arms one `setTimeout` before
		// `fetch` and never re-arms it, so it would abort THIS stream well
		// before the final chunk arrives — a false failure on a healthy
		// send, the mirror image of the false-refusal mode this whole gate
		// exists to close.
		const chunks = [
			'data: {"id":"chatcmpl-real","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}\n\n',
			'data: {"id":"chatcmpl-real","choices":[{"index":0,"delta":{"content":"b"},"finish_reason":null}]}\n\n',
			'data: {"id":"chatcmpl-real","choices":[{"index":0,"delta":{"content":"c"},"finish_reason":null}]}\n\n',
			'data: {"id":"chatcmpl-real","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
			'data: [DONE]\n\n'
		];
		const fetchImpl = (async (_url: string | URL | Request, init?: RequestInit) => {
			const encoder = new TextEncoder();
			// Wired to the SAME AbortSignal streamChatCompletion passes to
			// fetch — a real fetch/undici implementation aborts the body
			// read the instant the signal fires, so a mock that does NOT
			// react to it would let a wrongly-non-refreshed timer's abort()
			// call go unnoticed and this test would pass for the wrong
			// reason (the mutation-testing method's own warning: "a test
			// that has never actually failed is a claim, not evidence" —
			// verified failing against exactly this gap before landing).
			let aborted = false;
			init?.signal?.addEventListener('abort', () => {
				aborted = true;
			});
			const stream = new ReadableStream<Uint8Array>({
				async pull(controller) {
					if (aborted) {
						controller.error(new DOMException('Aborted', 'AbortError'));
						return;
					}
					if (chunks.length === 0) {
						controller.close();
						return;
					}
					await new Promise((resolve) => setTimeout(resolve, 80));
					if (aborted) {
						controller.error(new DOMException('Aborted', 'AbortError'));
						return;
					}
					controller.enqueue(encoder.encode(chunks.shift() as string));
				}
			});
			return new Response(stream, { status: 200 });
		}) as typeof fetch;

		const events = await collect(
			streamChatCompletion({ convId: 'c1', body: '{}', config: baseConfig({ timeoutMs: 150 }), fetchImpl })
		);
		assert.equal(events.length, 5, 'every chunk must arrive — nothing should have been aborted mid-stream');
		assert.equal(events[events.length - 1].isDone, true);
	}
);
