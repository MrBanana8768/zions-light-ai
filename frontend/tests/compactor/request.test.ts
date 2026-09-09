// The wire body builder — FRONTEND_PLAN.md §3.2's "never re-render content
// for the wire from the display form." Every assertion here is about
// BYTES, not parsed-value equality: a test that parsed the body back into
// an object before comparing would pass even if this module silently
// started re-serializing content, which is exactly the defect this file
// exists to catch.
//
// Gate remediation D10 (docs/lanes/L2-gate-findings.md): buildChatCompletionBody
// now demands an actual GateSuccess (`gate: GateSuccess`), not a bespoke
// {systemMessage, turns} pair — see request.ts's own header comment. Every
// test below builds one via makeGate() rather than passing loose fields.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	assertSendableConvId,
	buildChatCompletionBody,
	buildRequestHeaders,
	mintConvId,
	sanitizeConvId,
	wouldSanitizeToEmpty
} from '../../src/lib/server/compactor/request.js';
import type { GateSuccess } from '../../src/lib/server/compactor/types.js';
import type { Message } from '../../src/lib/server/store/types.js';
import { makeMessage } from './helpers/fakeChain.js';

function makeGate(params: { systemMessage: Message; turns: Message[]; convId?: string }): GateSuccess {
	const convId = params.convId ?? 'c1';
	return {
		ok: true,
		convId,
		windowIntent: params.turns.length,
		systemMessage: params.systemMessage,
		turns: params.turns,
		sentCount: params.turns.length
	};
}

test('buildChatCompletionBody: content is spliced onto the wire verbatim, never re-serialized', () => {
	// A content string containing things a parse/reserialize round trip
	// commonly disturbs: unicode escapes, key-order-sensitive-looking
	// text, and a trailing newline that whitespace-collapse must NOT be
	// allowed to touch here (only fingerprint.ts collapses whitespace —
	// this module must not).
	const weirdRawContent = JSON.stringify('line one\nline two   with runs of spaceé');
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({
		id: 'u1',
		convId: 'c1',
		role: 'user',
		parentId: 'root',
		content: weirdRawContent
	});
	const body = buildChatCompletionBody({
		model: 'test-model',
		gate: makeGate({ systemMessage: sys, turns: [userTurn] }),
		stream: true
	});
	// The EXACT raw content bytes must appear, uninterrupted, inside the
	// built body — proving no JSON.parse/JSON.stringify round trip
	// touched them.
	assert.ok(body.includes(`"content":${weirdRawContent}`));

	const parsedBody = JSON.parse(body);
	assert.equal(parsedBody.messages.length, 2);
	assert.equal(parsedBody.messages[0].role, 'system');
	assert.equal(parsedBody.messages[1].role, 'user');
	assert.equal(parsedBody.messages[1].content, JSON.parse(weirdRawContent));
	assert.equal(parsedBody.stream, true);
	assert.equal(parsedBody.model, 'test-model');
});

test('buildChatCompletionBody: an escape form a JSON.parse/stringify round trip would normalize survives untouched', () => {
	// `A` and `A` are the same VALUE but different BYTES. A
	// parse-then-reserialize step normalizes the escape away silently —
	// exactly the transformation FRONTEND_PLAN.md §3.2 forbids. This raw
	// text is deliberately never JSON.parse'd by this test either; it
	// asserts on the literal substring the way the real byte-stability
	// property must hold, not on parsed-value equality (which this
	// mutation would still pass).
	const rawContent = '"caf\\u0041\\u0301 \\u0041"'; // note: literal escape sequences in the SOURCE JSON text
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: rawContent });
	const body = buildChatCompletionBody({
		model: 'm',
		gate: makeGate({ systemMessage: sys, turns: [userTurn] }),
		stream: true
	});
	assert.ok(
		body.includes(`"content":${rawContent}`),
		'the escape sequence must appear byte-for-byte, not normalized to its decoded characters'
	);
});

test('buildChatCompletionBody: a multimodal content-parts array passes through untouched', () => {
	const parts = [
		{ type: 'text', text: 'look at this' },
		{ type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } }
	];
	const rawContent = JSON.stringify(parts);
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: rawContent });
	const body = buildChatCompletionBody({
		model: 'm',
		gate: makeGate({ systemMessage: sys, turns: [userTurn] }),
		stream: false
	});
	assert.ok(body.includes(`"content":${rawContent}`));
});

test('buildChatCompletionBody: refuses zero turns — never `messages: []`', () => {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	assert.throws(
		() => buildChatCompletionBody({ model: 'm', gate: makeGate({ systemMessage: sys, turns: [] }), stream: true }),
		/refusing to build a request with zero turns/
	);
});

test('buildChatCompletionBody: refuses a window that does not end on a user turn', () => {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	const asstTurn = makeMessage({
		id: 'a1',
		convId: 'c1',
		role: 'assistant',
		parentId: 'u1',
		content: JSON.stringify('hello')
	});
	assert.throws(
		() =>
			buildChatCompletionBody({
				model: 'm',
				gate: makeGate({ systemMessage: sys, turns: [userTurn, asstTurn] }),
				stream: true
			}),
		/newest message in the window is 'assistant', not 'user'/
	);
});

test('buildChatCompletionBody: refuses a non-system systemMessage', () => {
	const notSystem = makeMessage({ id: 'root', convId: 'c1', role: 'user', content: JSON.stringify('oops') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	assert.throws(() =>
		buildChatCompletionBody({ model: 'm', gate: makeGate({ systemMessage: notSystem, turns: [userTurn] }), stream: true })
	);
});

test('buildChatCompletionBody: max_tokens is included only when supplied', () => {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	const gate = makeGate({ systemMessage: sys, turns: [userTurn] });
	const withoutIt = buildChatCompletionBody({ model: 'm', gate, stream: false });
	assert.ok(!JSON.parse(withoutIt).hasOwnProperty('max_tokens'));
	const withIt = buildChatCompletionBody({
		model: 'm',
		gate,
		stream: false,
		maxTokens: 512
	});
	assert.equal(JSON.parse(withIt).max_tokens, 512);
});

// ---------------------------------------------------------------------------
// D10 — the gate's window bound to what is posted
// ---------------------------------------------------------------------------

test('buildChatCompletionBody: D10 — refuses a systemMessage that is not the conversation root (parentId !== null)', () => {
	const notRoot = makeMessage({
		id: 'not-root',
		convId: 'c1',
		role: 'system',
		parentId: 'something',
		content: JSON.stringify('persona')
	});
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'not-root', content: JSON.stringify('hi') });
	assert.throws(
		() => buildChatCompletionBody({ model: 'm', gate: makeGate({ systemMessage: notRoot, turns: [userTurn] }), stream: true }),
		/gate.systemMessage is not the conversation root/
	);
});

test('buildChatCompletionBody: D10 — refuses a systemMessage/convId mismatch', () => {
	const sys = makeMessage({ id: 'root', convId: 'OTHER-CONV', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	assert.throws(
		() =>
			buildChatCompletionBody({
				model: 'm',
				gate: makeGate({ systemMessage: sys, turns: [userTurn], convId: 'c1' }),
				stream: true
			}),
		/gate.systemMessage belongs to conversation/
	);
});

test('buildChatCompletionBody: D10 — refuses a self-inconsistent GateSuccess (turns.length !== sentCount)', () => {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	const gate: GateSuccess = {
		ok: true,
		convId: 'c1',
		windowIntent: 5,
		systemMessage: sys,
		turns: [userTurn],
		sentCount: 5 // inconsistent with turns.length (1)
	};
	assert.throws(
		() => buildChatCompletionBody({ model: 'm', gate, stream: true }),
		/self-inconsistent GateSuccess/
	);
});

test('buildChatCompletionBody: D10 — the defensive verifyChainShape re-check refuses a broken-alternation window even without a fresh audit', () => {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const u1 = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	// A second consecutive 'user' turn — the shape sendSet.ts's own
	// alternation check exists to catch, reused here at the point of
	// posting.
	const u2 = makeMessage({ id: 'u2', convId: 'c1', role: 'user', parentId: 'u1', content: JSON.stringify('again') });
	const gate: GateSuccess = {
		ok: true,
		convId: 'c1',
		windowIntent: 2,
		systemMessage: sys,
		turns: [u1, u2],
		sentCount: 2
	};
	assert.throws(
		() => buildChatCompletionBody({ model: 'm', gate, stream: true }),
		/defensive structural re-check/
	);
});

test('buildChatCompletionBody: D10 — the §4 rule 8 request-kind marker is present on every send', () => {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	const body = buildChatCompletionBody({ model: 'm', gate: makeGate({ systemMessage: sys, turns: [userTurn] }), stream: true });
	const parsed = JSON.parse(body);
	assert.equal(parsed.metadata.request_kind, 'conversation');
});

// ---------------------------------------------------------------------------
// Headers / conv_id sanitization parity with compactor/memory.py's _sanitize
// ---------------------------------------------------------------------------

test('wouldSanitizeToEmpty: mirrors memory.py\'s _CONV_ID_ALLOWED charset', () => {
	assert.equal(wouldSanitizeToEmpty('a-b_c123'), false);
	assert.equal(wouldSanitizeToEmpty(''), true);
	assert.equal(wouldSanitizeToEmpty('   '), true);
	assert.equal(wouldSanitizeToEmpty('!!!###'), true);
	// A UUID (v4 or v7) survives sanitization intact — this is what makes
	// either version a legitimate conv_id on the wire.
	assert.equal(wouldSanitizeToEmpty('018f6b1a-2e3e-7c3f-8a1b-abcdef123456'), false);
});

test('sanitizeConvId: strips disallowed characters and caps at 64, matching memory.py\'s _sanitize', () => {
	assert.equal(sanitizeConvId('a-b_c123'), 'a-b_c123');
	assert.equal(sanitizeConvId('my chat'), 'mychat');
	assert.equal(sanitizeConvId(''), '');
	assert.equal(sanitizeConvId('  ###  '), '');
	const seventy = 'a'.repeat(70);
	const sanitizedSeventy = sanitizeConvId(seventy);
	assert.equal(sanitizedSeventy.length, 64);
	assert.equal(sanitizedSeventy, 'a'.repeat(64));
});

test('assertSendableConvId: throws rather than sending a header that sanitizes to empty', () => {
	assert.throws(() => assertSendableConvId('###'), /would sanitize to empty/);
	assert.throws(() => assertSendableConvId(''), /would sanitize to empty/);
	assert.doesNotThrow(() => assertSendableConvId(mintConvId()));
});

// ---------------------------------------------------------------------------
// D9 — sanitize(id) === id, not merely non-empty
// ---------------------------------------------------------------------------

test('assertSendableConvId: D9 — a non-empty id that sanitizes to a DIFFERENT value is refused, not sent silently mis-keyed', () => {
	// The ORIGINAL predicate only checked wouldSanitizeToEmpty — "my chat"
	// is non-empty even after sanitizing ("mychat"), so it used to pass
	// straight through, get sent verbatim as X-Conversation-Id, and the
	// compactor would key memory under "mychat" while resolve_conv_id
	// still reports source="header" with no signal anything diverged.
	assert.throws(() => assertSendableConvId('my chat'), /does not survive memory\.py's _sanitize unchanged/);
});

test('assertSendableConvId: D9 — two distinct ids that sanitize to the SAME value are BOTH refused (the collision, caught before either is sent)', () => {
	assert.throws(() => assertSendableConvId('my chat'));
	assert.throws(() => assertSendableConvId('m ychat'));
	// Both sanitize to "mychat" — proving the collision this predicate
	// exists to prevent is real, not hypothetical.
	assert.equal(sanitizeConvId('my chat'), sanitizeConvId('m ychat'));
});

test('assertSendableConvId: D9 — an id longer than 64 chars is refused (the truncation would silently rename it)', () => {
	const seventy = 'a'.repeat(70);
	assert.throws(() => assertSendableConvId(seventy), /does not survive memory\.py's _sanitize unchanged/);
});

test('assertSendableConvId: D9 — a UUID (the actual shape every real caller sends) survives sanitize() unchanged', () => {
	assert.doesNotThrow(() => assertSendableConvId('018f6b1a-2e3e-7c3f-8a1b-abcdef123456'));
	assert.doesNotThrow(() => assertSendableConvId(mintConvId()));
});

test('mintConvId: produces a well-formed UUIDv4', () => {
	const id = mintConvId();
	assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
});

test('buildRequestHeaders: Authorization sent only when a key is configured; X-Conversation-Id always present', () => {
	const convId = mintConvId();
	const withoutKey = buildRequestHeaders(convId, undefined);
	assert.equal(withoutKey['X-Conversation-Id'], convId);
	assert.equal('Authorization' in withoutKey, false);

	const withKey = buildRequestHeaders(convId, 'secret-key');
	assert.equal(withKey['Authorization'], 'Bearer secret-key');
});

test('buildRequestHeaders: refuses to send a header that would sanitize to empty', () => {
	assert.throws(() => buildRequestHeaders('###', undefined), /would sanitize to empty/);
});
