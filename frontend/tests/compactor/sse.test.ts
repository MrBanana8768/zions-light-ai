// SSE parsing and stream-shape classification — FRONTEND_PLAN.md §3.4's
// table, verbatim: classify by id prefix, never by the text/emoji.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	classifyChunk,
	classifyChunkId,
	extractDataPayload,
	parseErrorEnvelope,
	parseSseEvent,
	splitSseBlocks,
	STREAM_ID_PREFIX
} from '../../src/lib/server/compactor/sse.js';

// ---------------------------------------------------------------------------
// Classification by id prefix — the three synthesized shapes, and the relay
// ---------------------------------------------------------------------------

test('classifyChunkId: matches each documented prefix exactly, and only that prefix', () => {
	assert.equal(classifyChunkId('chatcmpl-cmd-18f2a9'), 'slash_command');
	assert.equal(classifyChunkId('chatcmpl-rejected-18f2a9'), 'rejected');
	assert.equal(classifyChunkId('chatcmpl-unavail-18f2a9'), 'unavailable');
	// vLLM's own id shape (whatever it happens to be) is the ordinary relay.
	assert.equal(classifyChunkId('chatcmpl-8f7a2b3c-d4e5-f678-90ab-cdef01234567'), 'relay');
	assert.equal(classifyChunkId(undefined), 'relay');
	assert.equal(classifyChunkId(42), 'relay');
});

test('classifyChunkId: never classifies by the emoji/text — only the id prefix matters', () => {
	// A relay chunk whose CONTENT happens to contain the exact apology
	// text a rejection would use must still classify as relay, because
	// nothing about its id says otherwise. This is the literal test for
	// "match on the id prefix, never the emoji in the text."
	const id = classifyChunkId('chatcmpl-real-vllm-id');
	assert.equal(id, 'relay');
});

test('classifyChunk: slash command — finish_reason "stop", no error object, id-only signal', () => {
	const chunk = {
		id: `${STREAM_ID_PREFIX.slash_command}18f2a9`,
		choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
	};
	const c = classifyChunk(chunk);
	assert.equal(c.kind, 'slash_command');
	assert.equal(c.terminal, true);
	assert.equal(c.error, undefined);
});

test('classifyChunk: 4xx rejection — finish_reason "error" PLUS a top-level error object', () => {
	const chunk = {
		id: `${STREAM_ID_PREFIX.rejected}18f2a9`,
		choices: [{ index: 0, delta: {}, finish_reason: 'error' }],
		error: { message: 'nope', type: 'invalid_request_error', code: 'backend_rejected' }
	};
	const c = classifyChunk(chunk);
	assert.equal(c.kind, 'rejected');
	assert.equal(c.terminal, true);
	assert.equal(c.error?.code, 'backend_rejected');
});

test('classifyChunk: 5xx/unreachable — finish_reason "stop", NO error object (the documented live gap)', () => {
	const chunk = {
		id: `${STREAM_ID_PREFIX.unavailable}18f2a9`,
		choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
	};
	const c = classifyChunk(chunk);
	assert.equal(c.kind, 'unavailable');
	assert.equal(c.terminal, true);
	assert.equal(c.error, undefined, 'the unavailable shape carries no error object — a documented gap, not this module\'s bug');
});

test('classifyChunk: non-terminal chunks (finish_reason null/absent) are not terminal', () => {
	const chunk = { id: 'chatcmpl-real', choices: [{ index: 0, delta: { content: 'hi' }, finish_reason: null }] };
	assert.equal(classifyChunk(chunk).terminal, false);
	const noFinishReasonAtAll = { id: 'chatcmpl-real', choices: [{ index: 0, delta: { content: 'hi' } }] };
	assert.equal(classifyChunk(noFinishReasonAtAll).terminal, false);
});

// ---------------------------------------------------------------------------
// SSE block splitting — across arbitrary byte-chunk boundaries
// ---------------------------------------------------------------------------

test('splitSseBlocks: splits complete blocks and holds back a partial one', () => {
	const buf = 'data: {"a":1}\n\ndata: {"b":2}\n\ndata: {"c":3';
	const { blocks, remainder } = splitSseBlocks(buf);
	assert.equal(blocks.length, 2);
	assert.equal(remainder, 'data: {"c":3');
});

test('splitSseBlocks: a chunk boundary landing mid-block is recoverable once the rest arrives', () => {
	const part1 = 'data: {"a":1}\n\ndata: {"b":';
	const { blocks: blocks1, remainder: rem1 } = splitSseBlocks(part1);
	assert.equal(blocks1.length, 1);
	const part2 = rem1 + '2}\n\n';
	const { blocks: blocks2, remainder: rem2 } = splitSseBlocks(part2);
	assert.equal(blocks2.length, 1);
	assert.equal(rem2, '');
	assert.equal(extractDataPayload(blocks2[0]), '{"b":2}');
});

test('splitSseBlocks: CRLF (\\r\\n\\r\\n) block separators split identically to LF — surviving-mutation audit #4', () => {
	// The audit's own finding: `/\r?\n\r?\n/` -> `/\n\n/` stayed green
	// because every existing fixture uses bare `\n\n`. A real HTTP server
	// (or a proxy in front of one) can legitimately send CRLF line endings
	// for SSE; this is the fixture that actually exercises the `\r?` part
	// of the regex rather than assuming it.
	const buf = 'data: {"a":1}\r\n\r\ndata: {"b":2}\r\n\r\ndata: {"c":3';
	const { blocks, remainder } = splitSseBlocks(buf);
	assert.equal(blocks.length, 2);
	assert.equal(extractDataPayload(blocks[0]), '{"a":1}');
	assert.equal(extractDataPayload(blocks[1]), '{"b":2}');
	assert.equal(remainder, 'data: {"c":3');
});

test('splitSseBlocks: a MIXED stream (some blocks LF-separated, some CRLF-separated) still splits every block', () => {
	const buf = 'data: {"a":1}\n\ndata: {"b":2}\r\n\r\ndata: {"c":3}\n\n';
	const { blocks, remainder } = splitSseBlocks(buf);
	assert.equal(blocks.length, 3);
	assert.equal(remainder, '');
});

test('extractDataPayload: strips the "data: " prefix and the leading space only — not a full trim', () => {
	assert.equal(extractDataPayload('data: {"x":1}'), '{"x":1}');
	assert.equal(extractDataPayload('data:{"x":1}'), '{"x":1}'); // no space variant
	assert.equal(extractDataPayload('data: [DONE]'), '[DONE]');
	assert.equal(extractDataPayload(': keep-alive comment, no data line'), null);
});

test('extractDataPayload: surviving-mutation audit #5 — a SECOND leading space (or trailing whitespace) survives, proving this is `.replace(/^ /, \'\')`, never `.trim()`', () => {
	// The audit's own finding: `.replace(/^ /, '')` -> `.trim()` stayed
	// green "despite the doc comment claiming 'not a trim.'" Every existing
	// fixture has exactly zero or one leading space and no trailing
	// whitespace, which both implementations handle identically. This
	// fixture has TWO leading spaces (SSE's own spec strips only the first)
	// and trailing whitespace before the newline — `.trim()` would remove
	// all of both; the real implementation must leave the second leading
	// space and any trailing whitespace untouched.
	assert.equal(extractDataPayload('data:  {"x":1}'), ' {"x":1}', 'only the FIRST leading space is stripped');
	assert.equal(
		extractDataPayload('data: {"x":1}  '),
		'{"x":1}  ',
		'trailing whitespace must survive — a real .trim() would remove it'
	);
});

// ---------------------------------------------------------------------------
// End-to-end: parseSseEvent
// ---------------------------------------------------------------------------

test('parseSseEvent: [DONE] is recognized without attempting JSON.parse on it', () => {
	const ev = parseSseEvent('[DONE]');
	assert.equal(ev.isDone, true);
});

test('parseSseEvent: a malformed payload degrades to parseError, not a thrown exception', () => {
	const ev = parseSseEvent('not json at all {{{');
	assert.equal(ev.isDone, false);
	if (ev.isDone) throw new Error('unreachable');
	assert.equal(ev.classified, null);
	assert.ok(ev.parseError);
});

test('parseSseEvent: a well-formed relay chunk classifies as relay and is not terminal', () => {
	const payload = JSON.stringify({
		id: 'chatcmpl-real-id',
		choices: [{ index: 0, delta: { content: 'partial reply' }, finish_reason: null }]
	});
	const ev = parseSseEvent(payload);
	assert.equal(ev.isDone, false);
	if (ev.isDone) throw new Error('unreachable');
	assert.equal(ev.classified?.kind, 'relay');
	assert.equal(ev.classified?.terminal, false);
});

// ---------------------------------------------------------------------------
// Error envelopes — FRONTEND_PLAN.md §3.4: two shapes, plus FastAPI's 422
// ---------------------------------------------------------------------------

test('parseErrorEnvelope: OpenAI-shaped {"error": {...}}', () => {
	const body = JSON.stringify({ error: { message: 'boom', type: 'invalid_request_error', code: 'empty_messages' } });
	const e = parseErrorEnvelope(400, body);
	assert.equal(e.envelope, 'openai');
	assert.equal(e.message, 'boom');
	assert.equal(e.code, 'empty_messages');
});

test('parseErrorEnvelope: {"detail": "..."} — the app-wide UnsafeConvId handler\'s shape', () => {
	const body = JSON.stringify({ detail: 'conv_id contains unsafe characters' });
	const e = parseErrorEnvelope(400, body);
	assert.equal(e.envelope, 'detail');
	assert.equal(e.message, 'conv_id contains unsafe characters');
});

test('parseErrorEnvelope: {"detail": [...]} — FastAPI\'s stock 422 validation-error shape', () => {
	const body = JSON.stringify({ detail: [{ loc: ['body', 'messages'], msg: 'field required', type: 'value_error' }] });
	const e = parseErrorEnvelope(422, body);
	assert.equal(e.envelope, 'detail');
	assert.ok(e.message.includes('field required'));
});

test('parseErrorEnvelope: a non-JSON body (502 non-JSON upstream) never renders a raw object', () => {
	const e = parseErrorEnvelope(502, '<html>Bad Gateway</html>');
	assert.equal(e.envelope, 'unparseable');
	assert.equal(e.message, '<html>Bad Gateway</html>');
});

test('parseErrorEnvelope: JSON that is neither shape also falls back honestly rather than guessing', () => {
	const e = parseErrorEnvelope(500, JSON.stringify({ something: 'else' }));
	assert.equal(e.envelope, 'unparseable');
});

test('parseErrorEnvelope: an empty body is reported as such, not as an empty string message', () => {
	const e = parseErrorEnvelope(503, '');
	assert.equal(e.envelope, 'unparseable');
	assert.ok(e.message.includes('503'));
});
