// F8 — the checked send-set and the pre-send gate. See sendSet.ts's own
// header comment before reading these tests: the window_intent formula
// tested here is this lane's resolution of a real ambiguity in
// FRONTEND_PLAN.md §3.2, not a restatement of the plan's literal words.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyOverride, computeWindowIntent, runGate } from '../../src/lib/server/compactor/sendSet.js';
import type { GateFailure } from '../../src/lib/server/compactor/types.js';
import { FakeChainReader, buildConversation, makeMessage } from './helpers/fakeChain.js';
import type { AuditResult, Conversation, ReadPage } from '../../src/lib/server/store/types.js';
import type { ChainReader } from '../../src/lib/server/compactor/types.js';

// ---------------------------------------------------------------------------
// computeWindowIntent — pure function
// ---------------------------------------------------------------------------

test('computeWindowIntent: whole chain fits (turnsOnChain <= n)', () => {
	assert.equal(computeWindowIntent(5, 60), 5);
	assert.equal(computeWindowIntent(0, 60), 0);
	assert.equal(computeWindowIntent(60, 60), 60);
});

test('computeWindowIntent: turnsOnChain > n, odd startIndex needs the one-turn parity trim', () => {
	// 65 - 60 = 5 (odd) -> starts on 'assistant' -> drop one -> 59.
	assert.equal(computeWindowIntent(65, 60), 59);
	// 121 - 60 = 61 (odd) -> same trim, regardless of magnitude.
	assert.equal(computeWindowIntent(121, 60), 59);
});

test('computeWindowIntent: turnsOnChain > n, even startIndex needs no trim', () => {
	// 64 - 60 = 4 (even) -> already starts on 'user' -> no trim -> 60.
	assert.equal(computeWindowIntent(64, 60), 60);
});

test('computeWindowIntent: generalizes to an odd N (no parity bug at all)', () => {
	// With N=59 (odd) and the "ends on user" invariant (turnsOnChain odd),
	// odd - odd = even -> never needs a trim.
	assert.equal(computeWindowIntent(65, 59), 59);
	assert.equal(computeWindowIntent(121, 59), 59);
});

// ---------------------------------------------------------------------------
// runGate — the happy paths
// ---------------------------------------------------------------------------

test('runGate: short conversation — whole chain sent, root included, no trim', async () => {
	const { chain, conv, root, turns } = buildConversation({ turnCount: 5 });
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, true);
	if (!outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.windowIntent, 5);
	assert.equal(outcome.sentCount, 5);
	assert.equal(outcome.systemMessage.id, root.id);
	assert.deepEqual(outcome.turns.map((m) => m.id), turns.map((m) => m.id));
	// oldest-first, opening on user, strictly alternating, 5 turns ends on
	// 'user' (the turn just typed, awaiting a reply — this lane's
	// resolution of U-2).
	assert.equal(outcome.turns[0].role, 'user');
	assert.equal(outcome.turns[4].role, 'user');
	assert.equal(outcome.turns[3].role, 'assistant');
});

test('runGate: long conversation past N=60 — the mandatory one-turn parity trim, NOT context_truncated', async () => {
	const { chain, conv } = buildConversation({ turnCount: 65 }); // odd, ends on 'user'
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, true, 'a healthy 65-turn conversation must not require an override just to send 60');
	if (!outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.windowIntent, 59);
	assert.equal(outcome.sentCount, 59);
	assert.equal(outcome.turns[0].role, 'user');
	assert.equal(outcome.turns[outcome.turns.length - 1].role, 'user'); // the newest turn is never dropped
	// contiguity: every turn's parent is the previous turn.
	for (let i = 1; i < outcome.turns.length; i++) {
		assert.equal(outcome.turns[i].parentId, outcome.turns[i - 1].id);
	}
});

test('runGate: exactly at the boundary (turnsOnChain == n) sends the whole chain untrimmed', async () => {
	const { chain, conv } = buildConversation({ turnCount: 60 });
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, true);
	if (!outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.windowIntent, 60);
	assert.equal(outcome.sentCount, 60);
});

test('runGate: never issues more than two chain reads regardless of conversation length', async () => {
	// 501, not 500: odd, so this fixture ends on 'user' — matching the
	// invariant every other fixture in this file relies on (the gate
	// always runs with a pending user turn as the current leaf).
	const { chain, conv } = buildConversation({ turnCount: 501 });
	let readTailCalls = 0;
	let readOlderCalls = 0;
	let auditCalls = 0;
	const counting: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: (id) => {
			auditCalls += 1;
			return chain.auditConversation(id);
		},
		readTail: (id, limit) => {
			readTailCalls += 1;
			return chain.readTail(id, limit);
		},
		readOlder: (id, before, limit) => {
			readOlderCalls += 1;
			return chain.readOlder(id, before, limit);
		}
	};
	const outcome = await runGate(counting, conv.id, 60);
	assert.equal(outcome.ok, true);
	assert.equal(auditCalls, 1);
	assert.equal(readTailCalls, 1);
	assert.equal(readOlderCalls, 1); // long conversation: one extra read to locate the root
	if (outcome.ok) assert.equal(outcome.sentCount, 59);
});

test('runGate: a conversation shorter than N never needs the root-lookup read at all', async () => {
	const { chain, conv } = buildConversation({ turnCount: 5 });
	let readOlderCalls = 0;
	const counting: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: (id) => chain.auditConversation(id),
		readTail: (id, limit) => chain.readTail(id, limit),
		readOlder: (id, before, limit) => {
			readOlderCalls += 1;
			return chain.readOlder(id, before, limit);
		}
	};
	await runGate(counting, conv.id, 60);
	assert.equal(readOlderCalls, 0);
});

// ---------------------------------------------------------------------------
// runGate — failure paths
// ---------------------------------------------------------------------------

test('runGate: unknown conversation is not_found, not a crash', async () => {
	const { chain } = buildConversation({ turnCount: 3 });
	const outcome = await runGate(chain, 'no-such-conversation', 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'not_found');
});

test('runGate: a second root routes to chain_unsound with the audit attached, not context_truncated', async () => {
	const { chain, conv } = buildConversation({ turnCount: 5 });
	// A second root — exactly 2026-08-24's mechanism, made structurally
	// impossible in the real schema by message_one_root_per_conv but not
	// by this fake, which is the point: this test proves the GATE also
	// refuses to treat it as a sendable window rather than relying solely
	// on the DB constraint the store lane already tests.
	chain.addMessage(makeMessage({ id: 'rogue-root', convId: conv.id, role: 'system', parentId: null }));
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'chain_unsound');
	assert.ok(outcome.audit);
	assert.equal((outcome.audit as AuditResult).roots, 2);
	assert.equal((outcome.audit as AuditResult).pass, false);
});

test('runGate: the standing 241/5-root/leaf-at-8 shape also fails as chain_unsound here', async () => {
	// Mirrors FRONTEND_PLAN.md's standing regression case at this lane's
	// own layer: a real audit_conversation() would report roots=5,
	// chain_from_current=8, deepest=208, pass=false — this fake's audit
	// agrees on the property that matters (pass=false), which is all
	// runGate is allowed to act on (never `deepest`, never `total` alone).
	const { chain, conv, turns } = buildConversation({ turnCount: 7 });
	chain.conv.currentLeafId = turns[7 - 1].id; // depth 8 counting the root
	// Add four more disconnected roots.
	for (let i = 0; i < 4; i++) {
		chain.addMessage(makeMessage({ id: `extra-root-${i}`, convId: conv.id, role: 'system', parentId: null }));
	}
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'chain_unsound');
	assert.equal((outcome.audit as AuditResult).roots, 5);
});

test('runGate: broken alternation inside the window is context_truncated, not silently sent', async () => {
	const { chain, conv, turns } = buildConversation({ turnCount: 5 });
	// Corrupt one turn's role in place — the store's own state-machine
	// trigger cannot express "two user turns in a row" as a constraint
	// (role alternation is not a column-level invariant), so this is
	// exactly the kind of anomaly this gate exists to catch independently
	// of the store's own constraints.
	const corrupted = { ...turns[3], role: 'user' as const };
	chain.addMessage(corrupted);
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.ok(outcome.reasons.some((r) => r.includes('consecutive same-role')));
	// The failed send set is still attached — not thrown away — for D4's
	// override path.
	assert.equal(outcome.sentCount, 5);
	assert.ok(outcome.turns);
});

test('runGate: a race between the audit read and the window read is caught as context_truncated', async () => {
	const { chain, conv } = buildConversation({ turnCount: 65 });
	// Simulate a concurrent append landing between this module's audit
	// call and its window read: the audit reports the OLD, shorter chain
	// (turnsOnChain=63, which predicts windowIntent 59 via the same
	// formula as the real 65-turn case — pick a value where the
	// PREDICTION genuinely diverges from what the window read will find).
	const racy: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: async (id) => {
			const real = await chain.auditConversation(id);
			if (!real) return real;
			// Report as if only 3 turns existed (chain_from_current = 4,
			// i.e. 3 turns + the root) — wildly different from the 65
			// turns readTail is about to actually find.
			return { ...real, chainFromCurrent: 4 };
		},
		readTail: (id, limit) => chain.readTail(id, limit),
		readOlder: (id, before, limit) => chain.readOlder(id, before, limit)
	};
	const outcome = await runGate(racy, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.equal(outcome.windowIntent, 3); // the (stale) audit's prediction
	assert.notEqual(outcome.sentCount, 3); // the read found the real, longer chain
	assert.ok(outcome.reasons.some((r) => r.includes('intended 3')));
});

test('runGate: a tombstoned message inside the window is context_truncated', async () => {
	const { chain, conv, turns } = buildConversation({ turnCount: 5 });
	chain.addMessage({ ...turns[2], deletedAt: new Date().toISOString() });
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.ok(outcome.reasons.some((r) => r.includes('tombstoned')));
});

test('runGate: an unlocatable root (root-lookup exhausted) is context_truncated, never a fabricated persona', async () => {
	const { chain, conv } = buildConversation({ turnCount: 65 });
	const blind: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: (id) => chain.auditConversation(id),
		readTail: (id, limit) => chain.readTail(id, limit),
		// Every readOlder call returns nothing — as if the root-lookup
		// walk hit a wall (a cyclic/corrupt chain beyond ROOT_LOOKUP_LIMIT).
		readOlder: async (): Promise<ReadPage> => ({ messages: [], cursor: null })
	};
	const outcome = await runGate(blind, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.equal(outcome.systemMessage, null);
	assert.ok(outcome.reasons.some((r) => r.includes('could not be located')));
});

// ---------------------------------------------------------------------------
// D4 — the override path
// ---------------------------------------------------------------------------

test('applyOverride: copies every field verbatim — never recomputes, never overwrites windowIntent', async () => {
	const { chain, conv, turns } = buildConversation({ turnCount: 5 });
	const corrupted = { ...turns[3], role: 'user' as const };
	chain.addMessage(corrupted);
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false);
	const failure = outcome as GateFailure;

	const record = applyOverride(failure);
	assert.equal(record.sentUnderOverride, true);
	assert.equal(record.convId, conv.id);
	assert.equal(record.originalWindowIntent, failure.windowIntent);
	assert.equal(record.sentCount, failure.sentCount);
	assert.deepEqual(record.turns, failure.turns);
	assert.deepEqual(record.reasons, failure.reasons);

	// The defining property: re-running the gate on the SAME (still
	// corrupted) chain still fails the same way — proving applyOverride
	// did not fix, mutate, or otherwise cause the underlying data to pass.
	const rerun = await runGate(chain, conv.id, 60);
	assert.equal(rerun.ok, false);
});

test('applyOverride: originalWindowIntent is the INTENT, never silently substituted with sentCount', async () => {
	// Deliberately a scenario where windowIntent and sentCount are
	// DIFFERENT numbers — the alternation-corruption fixture above leaves
	// them accidentally equal (5 and 5), which would let a mutation that
	// swapped one field for the other slip through undetected. This is
	// the race fixture from the runGate tests above, reused here
	// specifically because windowIntent=3 and sentCount=65 there.
	const { chain, conv } = buildConversation({ turnCount: 65 });
	const racy = {
		getConversation: (id: string) => chain.getConversation(id),
		auditConversation: async (id: string) => {
			const real = await chain.auditConversation(id);
			return real ? { ...real, chainFromCurrent: 4 } : real;
		},
		readTail: (id: string, limit: number) => chain.readTail(id, limit),
		readOlder: (id: string, before: string, limit: number) => chain.readOlder(id, before, limit)
	};
	const outcome = await runGate(racy, conv.id, 60);
	assert.equal(outcome.ok, false);
	const failure = outcome as GateFailure;
	assert.equal(failure.windowIntent, 3);
	assert.notEqual(failure.sentCount, 3);

	const record = applyOverride(failure);
	assert.equal(record.originalWindowIntent, 3, 'must be the ORIGINAL intent, not the sent count');
	assert.notEqual(
		record.originalWindowIntent,
		record.sentCount,
		'this fixture exists specifically because these two numbers must NOT be silently unified'
	);
});

test('applyOverride: refuses chain_unsound and not_found — there is no coherent send set to override', async () => {
	const { chain, conv } = buildConversation({ turnCount: 3 });
	chain.addMessage(makeMessage({ id: 'rogue-root', convId: conv.id, role: 'system', parentId: null }));
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false);
	assert.throws(() => applyOverride(outcome as GateFailure), /only defined for 'context_truncated'/);

	const notFound = await runGate(chain, 'nope', 60);
	assert.equal(notFound.ok, false);
	assert.throws(() => applyOverride(notFound as GateFailure));
});
