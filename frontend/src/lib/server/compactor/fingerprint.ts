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

/** Gate remediation D5 (docs/lanes/L2-gate-findings.md). Python's `or`
 *  treats EVERY falsy value as "no content at all" — not just
 *  None/""/[] (this module's previous port), but also `0`, `0.0`, `False`
 *  and `{}` (an empty dict is falsy in Python; a JS empty object is
 *  truthy, so `!decoded` alone would take the wrong branch for it, exactly
 *  as a plain `||` would for an empty array). */
function isPythonFalsy(value: unknown): boolean {
	if (value === null || value === undefined) return true;
	if (value === '' || value === 0 || value === false) return true;
	if (Array.isArray(value)) return value.length === 0;
	if (typeof value === 'object') return Object.keys(value as Record<string, unknown>).length === 0;
	return false;
}

/** Python `str()` for a value that has already passed through the `or ""`
 *  falsy substitution and is known NOT to be a list (that branch is handled
 *  separately in messageText — see below). `str(x)` for a bare string is
 *  the string itself, unquoted; for anything else it is effectively
 *  `repr(x)` (Python's str() and repr() agree on every JSON-representable
 *  type except the top-level string case) — `True`/`False`/`None`,
 *  `{'a': 1}` single-quoted dict/list reprs, and so on. Exercised by D5's
 *  own cross-check table: content=0/false -> "" (via isPythonFalsy, never
 *  reaching this function); content=true -> "True"; content={"a":1} ->
 *  "{'a': 1}". */
function pythonStr(value: unknown): string {
	if (typeof value === 'string') return value;
	return pythonRepr(value);
}

function pythonRepr(value: unknown): string {
	if (value === null || value === undefined) return 'None';
	if (typeof value === 'boolean') return value ? 'True' : 'False';
	if (typeof value === 'number') return String(value);
	if (typeof value === 'string') return pythonStringRepr(value);
	if (Array.isArray(value)) return `[${value.map(pythonRepr).join(', ')}]`;
	if (typeof value === 'object') {
		const obj = value as Record<string, unknown>;
		const parts = Object.keys(obj).map((k) => `${pythonStringRepr(k)}: ${pythonRepr(obj[k])}`);
		return `{${parts.join(', ')}}`;
	}
	return String(value);
}

/** Python's `repr()` of a string: single-quoted unless the string contains
 *  a `'` and no `"` (in which case Python prefers double quotes so nothing
 *  needs escaping). Only reachable from inside a dict/list repr (see
 *  pythonRepr above) — a bare top-level string never goes through this,
 *  matching `str()`'s own asymmetry with `repr()`. */
function pythonStringRepr(s: string): string {
	const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
	let out = quote;
	for (const ch of s) {
		if (ch === quote || ch === '\\') out += '\\' + ch;
		else if (ch === '\n') out += '\\n';
		else if (ch === '\r') out += '\\r';
		else if (ch === '\t') out += '\\t';
		else out += ch;
	}
	return out + quote;
}

/** Port of summarizer.py's `_message_text`:
 *
 *      content = m.get("content") or ""
 *      if isinstance(content, list):
 *          return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
 *      return str(content)
 *
 * D5: the falsy substitution (`isPythonFalsy`) and the non-list stringify
 * (`pythonStr`) now both match Python's actual semantics, not JS
 * truthiness/String() — cross-checked against a live `python.exe` run (see
 * fingerprint.test.ts and docs/lanes/L2-transport.md's "Gate remediation"
 * section for the exact vectors: 0/false -> "", true -> "True",
 * {"a":1} -> "{'a': 1}"). */
export function messageText(rawContent: string): string {
	const decoded = decodeContent(rawContent);
	const content: unknown = isPythonFalsy(decoded) ? '' : decoded;
	if (Array.isArray(content)) {
		return content
			.filter((c): c is Record<string, unknown> => typeof c === 'object' && c !== null)
			.map((c) => (c.text === undefined ? '' : String(c.text)))
			.join(' ');
	}
	return pythonStr(content);
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

/** Gate remediation D5 (docs/lanes/L2-gate-findings.md). Every Unicode code
 *  point for which CPython's `str.isspace()` returns `True` — dumped
 *  directly from a live `python.exe` run
 *  (`[cp for cp in range(0x110000) if chr(cp).isspace()]`, Python 3.14.7;
 *  see docs/lanes/L2-transport.md's "Gate remediation" section), NOT JS's
 *  `\s` regex class. The two sets diverge in BOTH directions on characters
 *  a real chat turn can plausibly contain:
 *
 *    - U+FEFF (BOM / zero-width no-break space): JS \s treats it as
 *      whitespace (ECMA-262's own <BOM> production); Python's isspace()
 *      does NOT (category Cf, not in Python's whitespace table). A message
 *      pasted from a UTF-8-BOM file collapses one way here and not the
 *      other unless this set matches Python exactly.
 *    - U+0085 (NEL, NEXT LINE): Python's isspace() IS true for it (it is a
 *      C1 control code Python's Unicode database marks bidirectional class
 *      B); JS's \s does NOT match it.
 *
 *  Both directions matter equally — the lane report this replaces named
 *  only the ASCII separators \x1c-\x1f (also in this set, and also
 *  correctly whitespace on both sides once this list is used) and missed
 *  that FEFF/0085 diverge in OPPOSITE directions, which a single "Python's
 *  set is a superset of JS's" mental model would not predict. */
const PYTHON_WHITESPACE_CODEPOINTS = new Set<number>([
	0x09, 0x0a, 0x0b, 0x0c, 0x0d, // TAB, LF, VT, FF, CR
	0x1c, 0x1d, 0x1e, 0x1f, // ASCII FS/GS/RS/US
	0x20, // SPACE
	0x85, // NEL (NOT in JS \s)
	0xa0, // NBSP
	0x1680, // OGHAM SPACE MARK
	0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
	0x2028, 0x2029, // LINE/PARAGRAPH SEPARATOR
	0x202f, // NARROW NBSP
	0x205f, // MEDIUM MATHEMATICAL SPACE
	0x3000 // IDEOGRAPHIC SPACE
	// Deliberately NOT here: U+FEFF (BOM/ZWNBSP) — Python's isspace() is
	// False for it; JS's \s would wrongly include it.
]);

function isPythonWhitespace(codePoint: number): boolean {
	return PYTHON_WHITESPACE_CODEPOINTS.has(codePoint);
}

/** Python's `str.split()` with no arguments: split on runs of
 *  `isPythonWhitespace` code points, discard empty tokens, no separator
 *  survives. Iterates by CODE POINT (`for...of`), not UTF-16 code unit, so
 *  a supplementary-plane character is never split mid-surrogate-pair —
 *  none of the code points above are outside the BMP, so this only matters
 *  for correctness of the non-whitespace runs themselves. */
function collapseWhitespace(text: string): string {
	const tokens: string[] = [];
	let current = '';
	for (const ch of text) {
		const cp = ch.codePointAt(0) ?? 0;
		if (isPythonWhitespace(cp)) {
			if (current.length > 0) {
				tokens.push(current);
				current = '';
			}
		} else {
			current += ch;
		}
	}
	if (current.length > 0) tokens.push(current);
	return tokens.join(' ');
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
