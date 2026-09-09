// UUIDv7 (RFC 9562) — hand-rolled rather than a dependency, per
// FRONTEND_PLAN.md §3.1: "Ids are bare UUIDv7, never built by string
// concatenation anywhere." This function builds the 16 raw bytes and hex-
// formats them; nothing here ever concatenates an id from a conversation id
// or any other value — every id minted by this store comes from here alone.
//
// Layout (RFC 9562 §5.7):
//   48 bits  unix_ts_ms, big-endian
//    4 bits  version (0111)
//   12 bits  rand_a
//    2 bits  variant (10)
//   62 bits  rand_b
//
// rand_a/rand_b are filled from crypto.randomBytes — this is the "random"
// variant (no monotonic counter), which is sufficient here: ordering only
// needs millisecond resolution (append/read paths never assume sub-ms
// ordering, and Postgres's own created_at + the parent-pointer chain are
// the actual source of truth for order, not the id's bit pattern).

import { randomBytes } from 'node:crypto';

// Standard 8-4-4-4-12 hex-digit grouping (32 hex digits total = 16 bytes).
const HEX_GROUPS = [8, 4, 4, 4, 12] as const;

export function uuidv7(now: number = Date.now()): string {
	const unixTsMs = BigInt(Math.trunc(now));
	const bytes = new Uint8Array(16);

	bytes[0] = Number((unixTsMs >> 40n) & 0xffn);
	bytes[1] = Number((unixTsMs >> 32n) & 0xffn);
	bytes[2] = Number((unixTsMs >> 24n) & 0xffn);
	bytes[3] = Number((unixTsMs >> 16n) & 0xffn);
	bytes[4] = Number((unixTsMs >> 8n) & 0xffn);
	bytes[5] = Number(unixTsMs & 0xffn);

	const rand = randomBytes(10);

	// version 7 in the high nibble; top 4 bits of the 12-bit rand_a in the low nibble
	bytes[6] = 0x70 | (rand[0] & 0x0f);
	// remaining 8 bits of rand_a
	bytes[7] = rand[1];
	// variant 10 in the top 2 bits; top 6 bits of the 62-bit rand_b in the rest
	bytes[8] = 0x80 | (rand[2] & 0x3f);
	// remaining 56 bits of rand_b
	bytes[9] = rand[3];
	bytes[10] = rand[4];
	bytes[11] = rand[5];
	bytes[12] = rand[6];
	bytes[13] = rand[7];
	bytes[14] = rand[8];
	bytes[15] = rand[9];

	const hex = Buffer.from(bytes).toString('hex');
	let offset = 0;
	const parts: string[] = [];
	for (const len of HEX_GROUPS) {
		parts.push(hex.slice(offset, offset + len));
		offset += len;
	}
	return parts.join('-');
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/** True only for a well-formed UUIDv7 string (version nibble 7, variant 10xx). */
export function isUuidv7(value: string): boolean {
	return UUID_RE.test(value);
}
