// The byte-stability contract with compactor/summarizer.py — FRONTEND_PLAN.md
// §3.2 ("The window is a number with a derivation, not a preference" / "The
// hard client obligation this creates").
//
// `turns_seen`'s anchor (`tail_fp`) is compared across two SEPARATE HTTP
// requests: what the compactor appended after streaming a reply on turn N,
// against what THIS client reads back out of its own store and re-sends on
// turn N+1. If the fingerprint of the trailing turns drifts for a reason that
// has nothing to do with the conversation actually changing, `_align_candidates`
// (summarizer.py) cannot find the anchor, `_ASSUMED_NEW_TURNS` (a constant, 2)
// is substituted for the real delta, and the position drifts — silently, with
// only a `log_once` warning — and `window_offset` then reads the WRONG slice
// of the transcript for the next rollup. Two turns get summarized twice, or
// none do; either way the summary hierarchy falls behind the conversation
// with no visible symptom on this side of the wire.
//
// So this module is not "a hash function" in the abstract — it is a PORT of
// exactly two functions from summarizer.py, byte for byte, verified against
// the real Python implementation (see docs/lanes/L2-transport.md's "the
// fingerprint contract and its test" section for the cross-check vectors and
// how they were produced). Anything this module computes differently from
// summarizer.py, even by one byte, is a live defect — there is no "close
// enough" for a value whose only job is equality comparison against a value
// the client will never see.
//
// WHAT MUST NEVER LEAK IN HERE: this module reads `message.content` (the
// store's raw-JSON-text wire form, see ../store/types.ts) with JSON.parse
// ONLY to extract text for hashing. The parsed value is never re-serialized
// and never sent anywhere — extraction for a one-way hash is not the
// "parse-and-reserialize-for-the-wire" transformation FRONTEND_PLAN.md §3.2
// forbids (that rule is about what actually goes out over HTTP; see
// request.ts, which never parses `content` at all).

import { createHash } from 'node:crypto';
import type { Message } from '../store/types.js';

/** How many trailing turns the anchor records — summarizer.py's
 *  `_ANCHOR_TURNS`. Not configurable: it must match the server exactly or
 *  the test this module exists to pass is meaningless. */
export const ANCHOR_TURNS = 4;

/** One non-system message's `content` value, decoded from the store's raw
 *  JSON text — never re-encoded. `content` is JSON for the OpenAI message
 *  "content" field's VALUE (a string, a content-parts array, or null/
 *  absent), matching what summarizer.py's `m.get("content")` receives for
 *  one message dict already narrowed to that same value — see
 *  frontend/tests/store/basic.test.ts's own `JSON.stringify('hello')` /
 *  `'"branch-a"'` literals for confirmation of that shape. */
function decodeContent(rawContent: string): unknown {
	return JSON.parse(rawContent);
}

/** Port of summarizer.py's `_message_text`:
 *
 *      content = m.get("content") or ""
 *      if isinstance(content, list):
 *          return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
 *      return str(content)
 *
 * Python's `or ""` treats None, "" and [] (any falsy value) as "no content
 * at all" before the isinstance check runs — so an empty list is NOT sent
 * into the list branch, it falls through to `str("")` = "". Mirrored here
 * explicitly rather than relying on JS truthiness, because a JS empty array
 * is truthy (unlike Python's empty list) and using `||` directly would take
 * the wrong branch for that one case. */
export function messageText(rawContent: string): string {
	const decoded = decodeContent(rawContent);
	const isFalsy =
		decoded === null ||
		decoded === undefined ||
		decoded === '' ||
		(Array.isArray(decoded) && decoded.length === 0);
	const content: unknown = isFalsy ? '' : decoded;
	if (Array.isArray(content)) {
		return content
			.filter((c): c is Record<string, unknown> => typeof c === 'object' && c !== null)
			.map((c) => (c.text === undefined ? '' : String(c.text)))
			.join(' ');
	}
	return String(content);
}

/** Port of summarizer.py's `_image_only_marker`. Reads the ORIGINAL decoded
 *  content (not the `or ""`-adjusted value `messageText` uses) — matching
 *  the Python function's own `m.get("content")`, which does not apply the
 *  `or ""` fallback. In practice this only differs from `messageText`'s
 *  input when content is an empty list, and an empty-list image count is
 *  always 0 either way, so the divergence is inert; kept faithful to the
 *  source rather than "simplified," because a future edit to either
 *  function must be able to diff cleanly against its Python twin. */
export function imageOnlyMarker(rawContent: string): string {
	const content = decodeContent(rawContent);
	if (!Array.isArray(content)) return '';
	let n = 0;
	for (const c of content) {
		if (typeof c !== 'object' || c === null) continue;
		const rec = c as Record<string, unknown>;
		const type = rec.type;
		if (type === 'image_url' || type === 'image' || type === 'input_image' || 'image_url' in rec) {
			n += 1;
		}
	}
	if (!n) return '';
	return `[shared ${n} image${n > 1 ? 's' : ''}]`;
}

/** Python's `str.split()` with no arguments: split on runs of whitespace,
 *  discard empty tokens, no separator survives. `"  a  b  ".split()` is
 *  `["a", "b"]` — not `["", "a", "b", ""]`, which is what
 *  `"  a  b  ".split(/\s+/)` gives in JS. `.filter(Boolean)` closes that
 *  gap. Not a byte-for-byte match for every Unicode whitespace code point
 *  Python's definition covers (see docs/lanes/L2-transport.md's limitations
 *  section) — close enough for every character a chat turn will contain in
 *  practice, and the divergence is on the ONLY-SYMPTOM side (a spurious
 *  anchor miss triggers `_ASSUMED_NEW_TURNS`, which is the existing
 *  degraded path, not a crash or data loss). */
function collapseWhitespace(text: string): string {
	return text.split(/\s+/).filter((s) => s.length > 0).join(' ');
}

/** Encodes `str` as UTF-8 with Python's `errors="surrogatepass"` semantics:
 *  a lone (unpaired) UTF-16 surrogate is encoded as the 3-byte sequence its
 *  raw 16-bit value would produce under ordinary UTF-8 rules, rather than
 *  being replaced (Node's own `Buffer.from(str, 'utf8')` substitutes U+FFFD
 *  for a lone surrogate) or throwing. This is what lets a message
 *  containing an unpaired surrogate (the compactor's own `unpaired_surrogate`
 *  400 — FRONTEND_PLAN.md §3.4 — is evidence such bodies exist) fingerprint
 *  identically on both sides instead of silently on only one.
 *
 *  Iterating a JS string with `for...of` walks by Unicode code point: a
 *  well-formed surrogate pair is yielded as one already-combined code
 *  point (>0xFFFF, encoded as 4 bytes below, standard UTF-8), while a LONE
 *  surrogate has no partner to combine with and is yielded as itself, a
 *  single code unit in the 0xD800-0xDFFF range — which the <=0xFFFF branch
 *  below encodes as an ordinary 3-byte sequence. That is exactly
 *  surrogatepass: it does not special-case the surrogate range at all, it
 *  just never refuses to encode it. Verified against a live
 *  `hashlib.sha256(...).hexdigest()` run — see
 *  docs/lanes/L2-transport.md. */
export function utf8SurrogatepassEncode(str: string): Uint8Array {
	const bytes: number[] = [];
	for (const ch of str) {
		const cp = ch.codePointAt(0) ?? 0;
		if (cp <= 0x7f) {
			bytes.push(cp);
		} else if (cp <= 0x7ff) {
			bytes.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
		} else if (cp <= 0xffff) {
			bytes.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
		} else {
			bytes.push(
				0xf0 | (cp >> 18),
				0x80 | ((cp >> 12) & 0x3f),
				0x80 | ((cp >> 6) & 0x3f),
				0x80 | (cp & 0x3f)
			);
		}
	}
	return Uint8Array.from(bytes);
}

/** sha256, truncated to 16 hex chars, matching summarizer.py's
 *  `.hexdigest()[:16]`. Kept as its own function so tests can assert the
 *  hashing step in isolation from the byte-encoding step. Uses node:crypto
 *  rather than WebCrypto's `crypto.subtle` (which is async and, without
 *  the `dom` lib, needs a type-only workaround for a plain `Uint8Array`
 *  argument) — `createHash` is synchronous, universally available in
 *  Node, and needs no import gymnastics to stay importable from a bare
 *  `node --test` run exactly like ../store/db.ts (see that file's own
 *  comment on process.env vs SvelteKit's $env/*). */
export function sha256Hex16(bytes: Uint8Array): string {
	return createHash('sha256').update(Buffer.from(bytes)).digest('hex').slice(0, 16);
}

/** Port of summarizer.py's `_turn_fingerprints`, for ONE message. `role`
 *  is passed separately rather than read off `Message` so this stays
 *  usable from a plain `{role, content}` pair in tests without needing a
 *  full store `Message` record. Returns null for a system message — the
 *  Python function skips system messages entirely (`_turn_fingerprints`
 *  loops `for m in messages: if m.get("role")=="system": continue`), and a
 *  synthetic null here (rather than a hash of an empty string) makes that
 *  skip visible to a caller that maps this over a mixed-role array instead
 *  of silently fingerprinting the persona as if it were a turn. */
export function turnFingerprint(role: string, rawContent: string): string | null {
	if (role === 'system') return null;
	let text = collapseWhitespace(messageText(rawContent));
	if (!text) text = imageOnlyMarker(rawContent);
	const payload = `${role}\x00${text}`;
	return sha256Hex16(utf8SurrogatepassEncode(payload));
}

/** Fingerprints every NON-SYSTEM message in `messages`, oldest first —
 *  matching `_turn_fingerprints(messages)`'s own skip-system-and-continue
 *  loop, which never emits a placeholder for a skipped system message
 *  (unlike the single-message `turnFingerprint` above, which returns null
 *  for exactly that case so THIS function can filter it out rather than
 *  reimplementing the skip). Callers pass messages in chronological
 *  (oldest-first) order — the same order `readTail`'s reversed page, or
 *  `readOlder`'s page, is put into before this is called; see sendSet.ts. */
export function turnFingerprints(messages: Pick<Message, 'role' | 'content'>[]): string[] {
	const out: string[] = [];
	for (const m of messages) {
		const fp = turnFingerprint(m.role, m.content);
		if (fp !== null) out.push(fp);
	}
	return out;
}

/** The trailing `ANCHOR_TURNS` fingerprints of `messages` (oldest-first
 *  input, oldest-first output) — what this client must be able to
 *  reproduce IDENTICALLY across two separate sends of the same
 *  conversation for `tail_fp` alignment to hold. Returns fewer than
 *  `ANCHOR_TURNS` entries if the conversation has fewer non-system turns
 *  than that — summarizer.py's own anchor is exactly as short on a young
 *  conversation, and `_align_candidates` (its side) handles that already;
 *  nothing here needs to compensate for it. */
export function trailingAnchorFingerprints(messages: Pick<Message, 'role' | 'content'>[]): string[] {
	return turnFingerprints(messages).slice(-ANCHOR_TURNS);
}
