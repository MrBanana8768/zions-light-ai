// The client half of FRONTEND_SPEC.md §12's receipt / §3.3's "report what
// the server ADMITTED, not what the client sent."
//
// FRONTEND_PLAN.md §3.3: "The client half ships now. The server half is
// the received-context echo... The compactor sets NO custom response
// header anywhere" today. So `messagesAdmitted` below is read
// DEFENSIVELY — a present-but-absent header is not an error, it is the
// expected state of this system until that echo ships — and this module
// has exactly one way to report that: `admittedSource: 'not_reported'`,
// `messagesAdmitted: null`. There is no code path anywhere in this file
// that falls back to `sentCount` when the header is missing. That
// substitution IS the 2026-08-28 failure: the client sent 65, the
// compactor admitted 4, and a receipt reading "65" would have been
// accurate about what was sent and worthless about what happened.
//
// PROPOSED HEADER NAMES. FRONTEND_PLAN.md §3.3/§15 specifies the FIELDS
// the echo must carry (resolved conv_id, source, messages received,
// messages admitted, exact prompt tokens, headroom, shed/trim counts) but
// names no actual HTTP header. The names below are this lane's proposal,
// not a settled contract — see docs/lanes/L2-transport.md's "handed back
// to the compactor lane" section. Reading an unrecognized/renamed header
// degrades to `not_reported` exactly like reading a wholly absent one; no
// special "the name changed" failure mode exists, by design, so a header
// rename on the server side is just this client staying honest rather
// than a runtime crash.

import type { AuditResult } from '../store/types.js';
import type { GateOutcome, ReceiptSnapshot } from './types.js';

export const PROPOSED_ADMITTED_HEADER = 'x-context-messages-admitted';

/** Accepts either a Fetch `Headers` object or a plain lowercased-key
 *  record, so tests can build a fixture without constructing a real
 *  `Headers` instance. */
export type HeaderReader = Headers | Record<string, string | undefined>;

function readHeader(headers: HeaderReader, name: string): string | null {
	if (headers instanceof Headers) return headers.get(name);
	const value = headers[name] ?? headers[name.toLowerCase()];
	return value ?? null;
}

/** Reads the messages-admitted field defensively. A header present but
 *  not a valid non-negative integer is treated the same as an absent
 *  header (`not_reported`) — a malformed value is not evidence of a real
 *  number, and this module has no way to know which malformed value would
 *  even be "closer to right," so it does not guess. */
export function readMessagesAdmitted(
	headers: HeaderReader
): { messagesAdmitted: number | null; admittedSource: 'header' | 'not_reported' } {
	const raw = readHeader(headers, PROPOSED_ADMITTED_HEADER);
	if (raw === null || raw === '') {
		return { messagesAdmitted: null, admittedSource: 'not_reported' };
	}
	const parsed = Number(raw);
	if (!Number.isInteger(parsed) || parsed < 0) {
		return { messagesAdmitted: null, admittedSource: 'not_reported' };
	}
	return { messagesAdmitted: parsed, admittedSource: 'header' };
}

export interface BuildReceiptParams {
	convId: string;
	audit: AuditResult;
	gate: GateOutcome;
	sentUnderOverride: boolean;
	/** Omit when the send never reached the network at all (a
	 *  `context_truncated` refusal with no override) — `admittedSource`
	 *  is then `not_reported` unconditionally, which is also the correct
	 *  answer, just for a different reason (no response ever arrived to
	 *  read a header from). */
	responseHeaders?: HeaderReader;
}

/** Assembles the client-side receipt fields. Deliberately takes the WHOLE
 *  `GateOutcome` rather than separate `windowIntent`/`sentCount`
 *  parameters — see the note on `branchCount` below for why this
 *  function's job is as much about naming what it CANNOT report as what
 *  it can. */
export function buildReceipt(params: BuildReceiptParams): ReceiptSnapshot {
	const { messagesAdmitted, admittedSource } = params.responseHeaders
		? readMessagesAdmitted(params.responseHeaders)
		: { messagesAdmitted: null, admittedSource: 'not_reported' as const };

	const windowIntent =
		'windowIntent' in params.gate && params.gate.windowIntent !== undefined
			? params.gate.windowIntent
			: 0;
	const sentCount =
		'sentCount' in params.gate && params.gate.sentCount !== undefined ? params.gate.sentCount : 0;

	// Gate remediation D12 (docs/lanes/L2-gate-findings.md): explicit
	// "refused, nothing was sent" — false whenever the gate succeeded, OR
	// the caller sent anyway under a D4 override (which DID touch the
	// network; `sentUnderOverride` already carries that half of the
	// distinction). Before this field, `{sentCount: 59, messagesAdmitted:
	// null}` was the SAME shape whether 59 turns were genuinely posted or
	// the gate refused and nothing ever reached the wire.
	const refused = !params.gate.ok && !params.sentUnderOverride;
	// Narrowed on `params.gate.ok` directly (not on the separately-computed
	// `refused` boolean) so TypeScript's discriminated-union narrowing
	// actually applies — `GateSuccess` has no `.reasons` field at all.
	const refusalReasons = !params.gate.ok && refused ? params.gate.reasons : null;

	return {
		convId: params.convId,
		messagesOnActivePath: params.audit.chainFromCurrent,
		messagesInConversation: params.audit.total,
		// The store's public API exposes no children/sibling query (only
		// ancestor-walks: readTail/readOlder, plus audit_conversation's own
		// aggregate counts) — there is no way to honestly compute "how many
		// branches does this conversation have" without either reimplementing
		// a descendant walk outside the Store class (this lane's brief:
		// "use it, do not reimplement it") or a new store method. `null` here
		// means exactly that: not available today, not zero. See
		// docs/lanes/L2-transport.md, "handed back."
		branchCount: null,
		windowIntent,
		sentCount,
		sentUnderOverride: params.sentUnderOverride,
		messagesAdmitted,
		admittedSource,
		refused,
		refusalReasons
	};
}
