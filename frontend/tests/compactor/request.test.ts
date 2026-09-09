// The wire body builder — FRONTEND_PLAN.md §3.2's "never re-render content
// for the wire from the display form." Every assertion here is about
// BYTES, not parsed-value equality: a test that parsed the body back into
// an object before comparing would pass even if this module silently
// started re-serializing content, which is exactly the defect this file
// exists to catch.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	assertSendableConvId,
	buildChatCompletionBody,
	buildRequestHeaders,
	mintConvId,
	wouldSanitizeToEmpty
} from '../../src/lib/server/compactor/request.js';
import { makeMessage } from './helpers/fakeChain.js';

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
		systemMessage: sys,
		turns: [userTurn],
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
	const body = buildChatCompletionBody({ model: 'm', systemMessage: sys, turns: [userTurn], stream: true });
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
	const body = buildChatCompletionBody({ model: 'm', systemMessage: sys, turns: [userTurn], stream: false });
	assert.ok(body.includes(`"content":${rawContent}`));
});

test('buildChatCompletionBody: refuses zero turns — never `messages: []`', () => {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	assert.throws(
		() => buildChatCompletionBody({ model: 'm', systemMessage: sys, turns: [], stream: true }),
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
				systemMessage: sys,
				turns: [userTurn, asstTurn],
				stream: true
			}),
		/newest message in the window is 'assistant', not 'user'/
	);
});

test('buildChatCompletionBody: refuses a non-system systemMessage', () => {
	const notSystem = makeMessage({ id: 'root', convId: 'c1', role: 'user', content: JSON.stringify('oops') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	assert.throws(() =>
		buildChatCompletionBody({ model: 'm', systemMessage: notSystem, turns: [userTurn], stream: true })
	);
});

test('buildChatCompletionBody: max_tokens is included only when supplied', () => {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system', content: JSON.stringify('persona') });
	const userTurn = makeMessage({ id: 'u1', convId: 'c1', role: 'user', parentId: 'root', content: JSON.stringify('hi') });
	const withoutIt = buildChatCompletionBody({ model: 'm', systemMessage: sys, turns: [userTurn], stream: false });
	assert.ok(!JSON.parse(withoutIt).hasOwnProperty('max_tokens'));
	const withIt = buildChatCompletionBody({
		model: 'm',
		systemMessage: sys,
		turns: [userTurn],
		stream: false,
		maxTokens: 512
	});
	assert.equal(JSON.parse(withIt).max_tokens, 512);
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

test('assertSendableConvId: throws rather than sending a header that sanitizes to empty', () => {
	assert.throws(() => assertSendableConvId('###'), /would sanitize to empty/);
	assert.throws(() => assertSendableConvId(''), /would sanitize to empty/);
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
