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
	pass: true
};

function makeSuccessGate(): GateOutcome {
	const sys = makeMessage({ id: 'root', convId: 'c1', role: 'system' });
	return {
		ok: true,
		convId: 'c1',
		windowIntent: 59,
		systemMessage: sys,
		turns: [],
		sentCount: 59
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
