// The byte-stability contract with compactor/summarizer.py's
// `_turn_fingerprints` / `_message_text` / `_image_only_marker`. Every
// vector below was cross-checked against a REAL run of the Python
// functions (see docs/lanes/L2-transport.md's "the fingerprint contract
// and its test" section for the exact script and its output) — these are
// not hashes this test invented, they are what Python actually produced.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	ANCHOR_TURNS,
	imageOnlyMarker,
	messageText,
	sha256Hex16,
	trailingAnchorFingerprints,
	turnFingerprint,
	turnFingerprints,
	utf8SurrogatepassEncode
} from '../../src/lib/server/compactor/fingerprint.js';

test('turnFingerprint: cross-checked vectors against a real Python run', () => {
	// python: hashlib.sha256(f"user\x00hello world".encode("utf-8","surrogatepass")).hexdigest()[:16]
	assert.equal(turnFingerprint('user', JSON.stringify('hello   world\n')), '71312c09790b940e');
	// whitespace-collapse: a different run of whitespace hashes IDENTICALLY.
	assert.equal(turnFingerprint('user', JSON.stringify(' hello world ')), '71312c09790b940e');
	// image-only turn, fingerprinted through the marker, not as "".
	assert.equal(
		turnFingerprint('user', JSON.stringify([{ type: 'image_url', image_url: { url: 'x' } }])),
		'216f49657b130d12'
	);
	// a lone (unpaired) UTF-16 surrogate — surrogatepass, not U+FFFD replacement.
	const lone = 'abc' + String.fromCharCode(0xd800) + 'def';
	assert.equal(turnFingerprint('user', JSON.stringify(lone)), '62ddfa15932339b6');
	// multi-part text content joins with a single space.
	assert.equal(
		turnFingerprint('assistant', JSON.stringify([{ type: 'text', text: 'hi' }, { type: 'text', text: 'there' }])),
		'c9c29a19bf502d37'
	);
});

test('turnFingerprint: system messages are skipped, not hashed as empty', () => {
	assert.equal(turnFingerprint('system', JSON.stringify('you are a helpful assistant')), null);
});

test('utf8SurrogatepassEncode: matches Python surrogatepass byte-for-byte on a lone surrogate', () => {
	// python: f"user\x00{'abc'+chr(0xD800)+'def'}".encode('utf-8','surrogatepass').hex()
	//   == '7573657200616263eda080646566'
	const bytes = utf8SurrogatepassEncode('user\x00abc' + String.fromCharCode(0xd800) + 'def');
	assert.equal(Buffer.from(bytes).toString('hex'), '7573657200616263eda080646566');
});

test('utf8SurrogatepassEncode: a valid supplementary-plane character round-trips as ordinary UTF-8', () => {
	const emoji = '\u{1F600}'; // a real, well-formed surrogate pair
	const bytes = utf8SurrogatepassEncode(emoji);
	assert.deepEqual(Buffer.from(bytes), Buffer.from(emoji, 'utf8'));
});

test('messageText: falsy content (None/""/[]) all collapse to "", matching Python\'s `or ""`', () => {
	assert.equal(messageText(JSON.stringify(null)), '');
	assert.equal(messageText(JSON.stringify('')), '');
	assert.equal(messageText(JSON.stringify([])), '');
});

test('messageText: a content-parts list joins only its text parts, dropping non-dict/non-text entries', () => {
	const content = [{ type: 'text', text: 'hello' }, 'not-a-dict', { type: 'image_url', image_url: {} }];
	assert.equal(messageText(JSON.stringify(content)), 'hello ');
});

test('imageOnlyMarker: singular vs plural, and zero images is ""', () => {
	assert.equal(imageOnlyMarker(JSON.stringify([{ type: 'image_url', image_url: {} }])), '[shared 1 image]');
	assert.equal(
		imageOnlyMarker(
			JSON.stringify([
				{ type: 'image_url', image_url: {} },
				{ type: 'image_url', image_url: {} }
			])
		),
		'[shared 2 images]'
	);
	assert.equal(imageOnlyMarker(JSON.stringify([{ type: 'text', text: 'hi' }])), '');
	assert.equal(imageOnlyMarker(JSON.stringify('plain string')), '');
});

test('turnFingerprints: skips system, preserves order, oldest-first input to oldest-first output', () => {
	const messages = [
		{ role: 'system' as const, content: JSON.stringify('persona') },
		{ role: 'user' as const, content: JSON.stringify('one') },
		{ role: 'assistant' as const, content: JSON.stringify('two') },
		{ role: 'user' as const, content: JSON.stringify('three') }
	];
	const fps = turnFingerprints(messages);
	assert.equal(fps.length, 3);
	assert.equal(fps[0], turnFingerprint('user', JSON.stringify('one')));
	assert.equal(fps[1], turnFingerprint('assistant', JSON.stringify('two')));
	assert.equal(fps[2], turnFingerprint('user', JSON.stringify('three')));
});

test('trailingAnchorFingerprints: the byte-stability invariant — identical across two sends of the same conversation', () => {
	const conversation = [
		{ role: 'system' as const, content: JSON.stringify('persona') },
		{ role: 'user' as const, content: JSON.stringify('turn 1') },
		{ role: 'assistant' as const, content: JSON.stringify('reply 1') },
		{ role: 'user' as const, content: JSON.stringify('turn 2') },
		{ role: 'assistant' as const, content: JSON.stringify('reply 2') },
		{ role: 'user' as const, content: JSON.stringify('turn 3') },
		{ role: 'assistant' as const, content: JSON.stringify('reply 3') }
	];

	// "Send 1": the client's JSON serializer happens to use one key order.
	const send1 = conversation.map((m) => ({ role: m.role, content: m.content }));
	// "Send 2": semantically IDENTICAL turns, but re-derived independently
	// (e.g. read back from Postgres jsonb, which reorders keys/re-renders
	// escapes for any multi-key object — irrelevant here since content is
	// a bare string, but exercised anyway with a fresh JSON.stringify call
	// per turn to prove this is testing re-derivation, not object identity).
	const send2 = conversation.map((m) => ({ role: m.role, content: JSON.parse(m.content) }))
		.map((m) => ({ role: m.role, content: JSON.stringify(m.content) }));

	const anchor1 = trailingAnchorFingerprints(send1);
	const anchor2 = trailingAnchorFingerprints(send2);
	assert.equal(anchor1.length, ANCHOR_TURNS);
	assert.deepEqual(anchor1, anchor2);
});

test('trailingAnchorFingerprints: fewer than ANCHOR_TURNS turns returns a short anchor, not padding', () => {
	const messages = [
		{ role: 'system' as const, content: JSON.stringify('persona') },
		{ role: 'user' as const, content: JSON.stringify('only turn') }
	];
	const anchor = trailingAnchorFingerprints(messages);
	assert.equal(anchor.length, 1);
});

test('sha256Hex16: exactly 16 lowercase hex characters', () => {
	const h = sha256Hex16(new TextEncoder().encode('anything'));
	assert.match(h, /^[0-9a-f]{16}$/);
});
