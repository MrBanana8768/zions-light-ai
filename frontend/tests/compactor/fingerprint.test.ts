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

test('messageText: D5 falsy-handling table — cross-checked against a real Python run (0/false/true/{"a":1})', () => {
	// python: {"role":"user","content":0} -> _message_text = ''
	//   (0 is falsy in Python; `or ""` substitutes it BEFORE str() ever runs
	//   — this module's earlier port only special-cased None/""/[], so 0
	//   fell through to String(0) = "0", not "")
	assert.equal(messageText(JSON.stringify(0)), '');
	// python: content=False -> _message_text = ''
	assert.equal(messageText(JSON.stringify(false)), '');
	// python: content=True -> _message_text = 'True'
	//   (True is TRUTHY in Python — `or ""` does not touch it — so this
	//   falls to str(True), which is the capitalized "True", not JS's "true")
	assert.equal(messageText(JSON.stringify(true)), 'True');
	// python: content={"a": 1} -> _message_text = "{'a': 1}"
	//   (a non-empty dict is truthy; str() of a dict is Python's own
	//   single-quoted repr, not JS's "[object Object]")
	assert.equal(messageText(JSON.stringify({ a: 1 })), "{'a': 1}");
	// python: content={} -> _message_text = ''
	//   (an EMPTY dict is falsy in Python, unlike a JS empty object, which
	//   is truthy — this is the one falsy case neither the original port
	//   nor the table in the findings names explicitly)
	assert.equal(messageText(JSON.stringify({})), '');
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

test('trailingAnchorFingerprints: the byte-stability invariant — identical across two GENUINELY different byte serializations of the same conversation', () => {
	// D5 / the surviving-mutation audit's #6: the ORIGINAL version of this
	// test built `send2` via JSON.parse then JSON.stringify on a bare
	// STRING — a no-op (JSON.stringify(JSON.parse('"turn 1"')) is
	// byte-for-byte '"turn 1"' again), so `send1` and `send2` were
	// byte-IDENTICAL and this test compared a value to itself. §3.2 names
	// this as "the corrected test" specifically because the invariant is
	// about the EXTRACTED TEXT surviving a real difference in bytes, not
	// about re-running the same serializer twice. Every turn below is
	// constructed two DIFFERENT ways that Postgres's own jsonb round trip
	// (see docs/lanes/L1-store.md §8) or a different-but-valid JSON
	// serializer could plausibly produce, while decoding to the identical
	// VALUE:
	//   - reordered object keys (jsonb does not preserve input key order)
	//   - a duplicate key jsonb collapses to "last value wins"
	//   - a \u-escaped non-ASCII character vs. the literal UTF-8 byte
	//     (Python's json.dumps default ensure_ascii=True produces the
	//     escaped form; this store's own content is never re-serialized,
	//     but a caller reading a DIFFERENT prior serialization back must
	//     still fingerprint identically)
	const send1 = [
		{ role: 'system' as const, content: JSON.stringify('persona') },
		{ role: 'user' as const, content: JSON.stringify('turn 1') },
		// key order: type, text
		{ role: 'assistant' as const, content: '[{"type":"text","text":"reply 1"}]' },
		{ role: 'user' as const, content: JSON.stringify('turn 2') },
		// a content-parts array whose single object has a duplicate "text"
		// key — jsonb's own "last value wins" behaviour, reproduced here as
		// a literal string this module must decode exactly as JSON.parse
		// (and Postgres) both do. Wrapped in an array (a real multimodal
		// content shape), not a bare object: messageText reads a bare
		// object's dict REPR, which embeds key order and is legitimately
		// NOT key-order-stable (a separate, already-known limitation of
		// that fallback path, not what this test exists to demonstrate).
		{ role: 'assistant' as const, content: '[{"text":"WRONG","type":"text","text":"reply 2"}]' },
		// literal UTF-8 non-ASCII character.
		{ role: 'user' as const, content: JSON.stringify('turn café 3') },
		{ role: 'assistant' as const, content: JSON.stringify('reply 3') }
	];
	const send2 = [
		{ role: 'system' as const, content: JSON.stringify('persona') },
		{ role: 'user' as const, content: JSON.stringify('turn 1') },
		// same VALUE, keys reordered: text, type.
		{ role: 'assistant' as const, content: '[{"text":"reply 1","type":"text"}]' },
		{ role: 'user' as const, content: JSON.stringify('turn 2') },
		// the jsonb-normalized shape: duplicate key already collapsed,
		// same content-parts array shape as send1's version.
		{ role: 'assistant' as const, content: '[{"type":"text","text":"reply 2"}]' },
		// \u-escaped form of the identical character — Python's
		// json.dumps(..., ensure_ascii=True) default, and jsonb resolves
		// \u-escapes to the literal character on output, so a reader that
		// re-derives this turn from either the client's OWN prior send or
		// a jsonb round trip must still see the same decoded string.
		{ role: 'user' as const, content: '"turn caf\\u00e9 3"' },
		{ role: 'assistant' as const, content: JSON.stringify('reply 3') }
	];

	// Prove these two conversations are genuinely byte-DIFFERENT before
	// asserting their fingerprints agree — otherwise this test could
	// silently regress back to the exact vacuity it replaces.
	assert.notEqual(JSON.stringify(send1), JSON.stringify(send2));

	const anchor1 = trailingAnchorFingerprints(send1);
	const anchor2 = trailingAnchorFingerprints(send2);
	assert.equal(anchor1.length, ANCHOR_TURNS);
	assert.deepEqual(anchor1, anchor2);
});

test('turnFingerprint: D5 cross-check — U+FEFF (BOM) is NOT whitespace in Python; the anchor must not collapse it like a space', () => {
	// python (verified against a live python.exe run, 3.14.7):
	//   'a﻿b'.split() == ['a﻿b']   (isspace() is False for FEFF)
	//   turn_fp(user, 'a﻿b') == '8241c16cd56ac357'
	//   turn_fp(user, 'a b')      == '1bb94d01e957ee54'   (an ordinary space)
	// JS's \s DOES match U+FEFF, so the module this replaces collapsed
	// 'a﻿b' down to 'a b' and produced 1bb94d01e957ee54 — the WRONG,
	// JS-\s-derived hash — for this exact input.
	const withFeff = 'a\uFEFFb'; // explicit \uFEFF escape — no invisible literal byte in the source
	assert.equal(turnFingerprint('user', JSON.stringify(withFeff)), '8241c16cd56ac357');
	assert.notEqual(
		turnFingerprint('user', JSON.stringify(withFeff)),
		turnFingerprint('user', JSON.stringify('a b')),
		'U+FEFF must NOT collapse like an ordinary space — Python does not consider it whitespace'
	);
});

test('turnFingerprint: D5 cross-check — U+0085 (NEL) IS whitespace in Python, the opposite divergence from U+FEFF', () => {
	// python: 'a\x85b'.split() == ['a', 'b']   (isspace() is True for NEL)
	//   turn_fp(user, 'a\x85b') == turn_fp(user, 'a b') == '1bb94d01e957ee54'
	// JS's \s does NOT match U+0085, so a naive JS \s-based collapse would
	// treat 'a\x85b' as ONE token and diverge from Python here — the exact
	// opposite direction of the FEFF case above, which is why the findings
	// call these "the two that diverge in opposite directions."
	const withNel = 'a\u0085b'; // explicit \u0085 escape — no invisible literal byte in the source
	assert.equal(turnFingerprint('user', JSON.stringify(withNel)), '1bb94d01e957ee54');
	assert.equal(
		turnFingerprint('user', JSON.stringify(withNel)),
		turnFingerprint('user', JSON.stringify('a b')),
		'U+0085 (NEL) must collapse exactly like an ordinary space — Python DOES consider it whitespace'
	);
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
