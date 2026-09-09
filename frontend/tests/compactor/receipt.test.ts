// The client half of the receipt — FRONTEND_PLAN.md §3.3: "the receipt
// reports what the server ADMITTED, not what the client sent." The single
// most important property tested here is negative: there is no code path
// that falls back to `sentCount` when the admitted header is absent.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildReceipt, PROPOSED_ADMITTED_HEADER, readMessagesAdmitted } from '../../src/lib/server/compactor/receipt.js';
import type { AuditResult } from '../../src/lib/server/store/types.js';
import type { GateOutcome } from '../../src/lib/server/compactor/types.js';
import { makeMessage } from './helpers/fakeChain.js';

const AUDIT: AuditResult = {
	convId: 'c1',
	roots: 1,
	total: 65,
	reachableN: 65,
	missingParent: 0,
	leaf: 'leaf-id',
	leafOnTree: true,
	chainFromCurrent: 66,
	deepest: 66,
	sendableFromCurrent: 66,
	pass: true
};

function makeSuccessGate(): GateOutcome {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system' });
	// D12's minor note: the ORIGINAL fixture here returned `turns: []` with
	// `sentCount: 59` — self-inconsistent, and buildReceipt reported it
	// without complaint (buildReceipt never reads `.turns`, only
	// `.windowIntent`/`.sentCount`, so the inconsistency was harmless to
	// THIS module specifically, but still a fixture that should not exist
	// uncorrected). `turns.length` now genuinely equals `sentCount`.
	const turns = Array.from({ length: 59 }, (_, i) =>
		makeMessage({ id: `turn-${i}`, convId: 'c1', role: i % 2 === 0 ? 'user' : 'assistant' })
	);
	return {
		ok: true,
		convId: 'c1',
		windowIntent: 59,
		systemMessage: sys,
		turns,
		sentCount: turns.length
	};
}

test('readMessagesAdmitted: absent header -> not_reported, never a fabricated number', () => {
	const { messagesAdmitted, admittedSource } = readMessagesAdmitted({});
	assert.equal(messagesAdmitted, null);
	assert.equal(admittedSource, 'not_reported');
});

test('readMessagesAdmitted: present and valid -> the real value, sourced as "header"', () => {
	const headers = { [PROPOSED_ADMITTED_HEADER]: '4' };
	const { messagesAdmitted, admittedSource } = readMessagesAdmitted(headers);
	assert.equal(messagesAdmitted, 4);
	assert.equal(admittedSource, 'header');
});

test('readMessagesAdmitted: works against a real Headers instance too', () => {
	const headers = new Headers();
	headers.set(PROPOSED_ADMITTED_HEADER, '12');
	const { messagesAdmitted, admittedSource } = readMessagesAdmitted(headers);
	assert.equal(messagesAdmitted, 12);
	assert.equal(admittedSource, 'header');
});

test('readMessagesAdmitted: a malformed value degrades to not_reported rather than a guess', () => {
	assert.deepEqual(readMessagesAdmitted({ [PROPOSED_ADMITTED_HEADER]: 'not-a-number' }), {
		messagesAdmitted: null,
		admittedSource: 'not_reported'
	});
	assert.deepEqual(readMessagesAdmitted({ [PROPOSED_ADMITTED_HEADER]: '-1' }), {
		messagesAdmitted: null,
		admittedSource: 'not_reported'
	});
});

test('buildReceipt: THE property — never substitutes sentCount for a missing admitted count', () => {
	const gate = makeSuccessGate();
	// This IS the 2026-08-28 shape: the client sent 65 (well, 59 here),
	// no response header exists at all, and the receipt must say so
	// honestly rather than repeating the sent count as if it were the
	// server's own account.
	const receipt = buildReceipt({ convId: 'c1', audit: AUDIT, gate, sentUnderOverride: false });
	assert.equal(receipt.sentCount, 59);
	assert.equal(receipt.messagesAdmitted, null);
	assert.equal(receipt.admittedSource, 'not_reported');
	assert.notEqual(receipt.messagesAdmitted, receipt.sentCount);
});

test('buildReceipt: reads a real admitted count from response headers when present', () => {
	const gate = makeSuccessGate();
	const receipt = buildReceipt({
		convId: 'c1',
		audit: AUDIT,
		gate,
		sentUnderOverride: false,
		responseHeaders: { [PROPOSED_ADMITTED_HEADER]: '4' }
	});
	assert.equal(receipt.messagesAdmitted, 4);
	assert.equal(receipt.admittedSource, 'header');
	assert.notEqual(receipt.messagesAdmitted, receipt.sentCount); // 4 != 59, and that IS the point
});

test('buildReceipt: messagesOnActivePath / messagesInConversation come from the audit, not the gate', () => {
	const gate = makeSuccessGate();
	const receipt = buildReceipt({ convId: 'c1', audit: AUDIT, gate, sentUnderOverride: false });
	assert.equal(receipt.messagesOnActivePath, AUDIT.chainFromCurrent);
	assert.equal(receipt.messagesInConversation, AUDIT.total);
});

test('buildReceipt: branchCount is honestly null — no store primitive exists to compute it', () => {
	const gate = makeSuccessGate();
	const receipt = buildReceipt({ convId: 'c1', audit: AUDIT, gate, sentUnderOverride: false });
	assert.equal(receipt.branchCount, null);
});

test('buildReceipt: sentUnderOverride is carried through verbatim', () => {
	const gate = makeSuccessGate();
	const receipt = buildReceipt({ convId: 'c1', audit: AUDIT, gate, sentUnderOverride: true });
	assert.equal(receipt.sentUnderOverride, true);
});

// ---------------------------------------------------------------------------
// D12 — the explicit "refused, nothing was sent" signal
// ---------------------------------------------------------------------------

function makeFailureGate(): GateOutcome {
	return {
		ok: false,
		convId: 'c1',
		kind: 'context_truncated',
		reasons: ['realized 58 turn(s), intended 59'],
		windowIntent: 59,
		systemMessage: null,
		turns: [],
		sentCount: 58
	};
}

test('buildReceipt: refused=true when the gate failed and no override was applied — even though messagesAdmitted is ALSO null here, exactly like a normal send with no echo header yet', () => {
	const gate = makeFailureGate();
	const receipt = buildReceipt({ convId: 'c1', audit: AUDIT, gate, sentUnderOverride: false });
	assert.equal(receipt.refused, true);
	assert.deepEqual(receipt.refusalReasons, ['realized 58 turn(s), intended 59']);
	// THE property this field exists for: before it, this receipt and a
	// genuinely successful send whose echo header simply had not shipped
	// yet were INDISTINGUISHABLE (both show messagesAdmitted: null).
	assert.equal(receipt.messagesAdmitted, null);
});

test('buildReceipt: refused=false when a gate failure was sent anyway under a D4 override — it DID reach the network', () => {
	const gate = makeFailureGate();
	const receipt = buildReceipt({ convId: 'c1', audit: AUDIT, gate, sentUnderOverride: true });
	assert.equal(receipt.refused, false);
	assert.equal(receipt.refusalReasons, null);
});

test('buildReceipt: refused=false and refusalReasons=null on an ordinary successful send', () => {
	const gate = makeSuccessGate();
	const receipt = buildReceipt({ convId: 'c1', audit: AUDIT, gate, sentUnderOverride: false });
	assert.equal(receipt.refused, false);
	assert.equal(receipt.refusalReasons, null);
});
