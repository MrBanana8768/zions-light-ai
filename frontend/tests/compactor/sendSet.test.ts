// F8 — the checked send-set and the pre-send gate. See sendSet.ts's own
// header comment before reading these tests: the window_intent formula
// tested here is this lane's resolution of a real ambiguity in
// FRONTEND_PLAN.md §3.2, not a restatement of the plan's literal words.
//
// Gate remediation (docs/lanes/L2-gate-findings.md) D1/D2/D3/D4/D8/D11 all
// land in this file — see each test's own comment for which finding it
// closes.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	applyOverride,
	assertValidWindowN,
	computeWindowIntent,
	MAX_WINDOW_N,
	runGate,
	verifyChainShape
} from '../../src/lib/server/compactor/sendSet.js';
import type { GateFailure } from '../../src/lib/server/compactor/types.js';
import { FakeChainReader, buildConversation, makeMessage } from './helpers/fakeChain.js';
import type { AuditResult } from '../../src/lib/server/store/types.js';
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
// D11 — n is validated: positive integer, ceiling, rejected loudly
// ---------------------------------------------------------------------------

test('assertValidWindowN: accepts any positive integer up to MAX_WINDOW_N', () => {
	assert.doesNotThrow(() => assertValidWindowN(1));
	assert.doesNotThrow(() => assertValidWindowN(60));
	assert.doesNotThrow(() => assertValidWindowN(MAX_WINDOW_N));
});

test('assertValidWindowN: rejects zero, negative, non-integer, and NaN — loudly, not by clamping', () => {
	assert.throws(() => assertValidWindowN(0), /positive integer/);
	assert.throws(() => assertValidWindowN(-5), /positive integer/);
	assert.throws(() => assertValidWindowN(0.5), /positive integer/);
	assert.throws(() => assertValidWindowN(NaN), /positive integer/);
});

test('assertValidWindowN: rejects anything over MAX_WINDOW_N — the exact D11 defect (n=100000 used to return ok:true)', () => {
	assert.throws(() => assertValidWindowN(MAX_WINDOW_N + 1), /exceeds MAX_WINDOW_N/);
	assert.throws(() => assertValidWindowN(100_000), /exceeds MAX_WINDOW_N/);
});

test('runGate: an invalid n is rejected loudly (thrown), before any chain read at all', async () => {
	const { chain, conv } = buildConversation({ turnCount: 5 });
	let auditCalls = 0;
	const counting: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: (id) => {
			auditCalls += 1;
			return chain.auditConversation(id);
		},
		readTail: (id, limit) => chain.readTail(id, limit),
		readOlder: (id, before, limit) => chain.readOlder(id, before, limit),
		getRoot: (id) => chain.getRoot(id)
	};
	await assert.rejects(() => runGate(counting, conv.id, 100_000), /exceeds MAX_WINDOW_N/);
	await assert.rejects(() => runGate(counting, conv.id, -1), /positive integer/);
	assert.equal(auditCalls, 0, 'an invalid n must be rejected before the audit is even read');
});

// ---------------------------------------------------------------------------
// verifyChainShape — the structural check, reused by request.ts (D10)
// ---------------------------------------------------------------------------

test('verifyChainShape: a healthy window has no reasons', () => {
	const { conv, turns } = buildConversation({ turnCount: 5 });
	assert.deepEqual(verifyChainShape(conv.id, turns), []);
});

test('verifyChainShape: empty turns is its own single reason, not a crash', () => {
	const reasons = verifyChainShape('c1', []);
	assert.equal(reasons.length, 1);
	assert.ok(reasons[0].includes('no turns'));
});

test('verifyChainShape: D4 — a non-complete turn anywhere in the window is named explicitly', () => {
	const { conv, turns } = buildConversation({ turnCount: 5 });
	const corrupted = [...turns];
	corrupted[3] = { ...corrupted[3], state: 'streaming' };
	const reasons = verifyChainShape(conv.id, corrupted);
	assert.ok(reasons.some((r) => r.includes('not complete') && r.includes('state=streaming')));
});

test('verifyChainShape: a turn from a different conversation is refused', () => {
	const { conv, turns } = buildConversation({ turnCount: 3 });
	const foreign = [...turns];
	foreign[1] = { ...foreign[1], convId: 'some-other-conv' };
	const reasons = verifyChainShape(conv.id, foreign);
	assert.ok(reasons.some((r) => r.includes('belongs to conversation some-other-conv')));
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
	// 59, not 60: turnsOnChain must be ODD for a healthy fixture to end on
	// 'user' at all (see this file's header comment / D1 below) — the
	// ORIGINAL version of this test used turnCount: 60 (EVEN), which put
	// the leaf on an ASSISTANT turn and only ever passed because the gate
	// did not yet check the closing role. See the D1 test immediately
	// below for that exact defect, now fixed and asserted directly.
	const { chain, conv } = buildConversation({ turnCount: 59 });
	const outcome = await runGate(chain, conv.id, 59);
	assert.equal(outcome.ok, true);
	if (!outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.windowIntent, 59);
	assert.equal(outcome.sentCount, 59);
	assert.equal(outcome.turns[outcome.turns.length - 1].role, 'user');
});

test('runGate: integration coverage for the EVEN-startIndex "no trim needed" path — closes surviving-mutation audit #3', async () => {
	// The audit's own finding: "computeWindowIntent's even-startIndex path
	// is never integration-tested (fixtures are 5, 60, 65, 501)." With
	// N=60 (even) that path is actually UNREACHABLE by any healthy fixture
	// at all: turnsOnChain is always odd for a healthy conversation (this
	// file's header comment), so odd - even is always odd — every N=60
	// integration fixture in this file exercises the ODD-startIndex
	// (trim-needed) branch, by construction, no matter how it is varied.
	// Reaching the even branch for real needs an ODD n instead: 65 - 59 =
	// 6 (even) -> no trim -> the newest 59 turns already open on 'user'.
	const { chain, conv } = buildConversation({ turnCount: 65 });
	const outcome = await runGate(chain, conv.id, 59);
	assert.equal(outcome.ok, true, 'the even-startIndex path must accept the window untrimmed');
	if (!outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.windowIntent, 59); // n itself — computeWindowIntent's even-offset branch
	assert.equal(outcome.sentCount, 59);
	assert.equal(outcome.turns[0].role, 'user', 'no leading assistant to strip on this branch');
	assert.equal(outcome.turns[outcome.turns.length - 1].role, 'user');
});

test('runGate: never issues more than two chain reads regardless of conversation length', async () => {
	// 501, not 500: odd, so this fixture ends on 'user' — matching the
	// invariant every other fixture in this file relies on (the gate
	// always runs with a pending user turn as the current leaf).
	const { chain, conv } = buildConversation({ turnCount: 501 });
	let readTailCalls = 0;
	let readOlderCalls = 0;
	let getRootCalls = 0;
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
		},
		getRoot: (id) => {
			getRootCalls += 1;
			return chain.getRoot(id);
		}
	};
	const outcome = await runGate(counting, conv.id, 60);
	assert.equal(outcome.ok, true);
	assert.equal(auditCalls, 1);
	assert.equal(readTailCalls, 1);
	// D8 (docs/lanes/L2-gate-findings.md): the root lookup for a long
	// conversation is now getRoot — a one-row, indexed read — not a
	// second O(n) readOlder walk. readOlderCalls must be ZERO: this
	// module no longer calls readOlder AT ALL. getRootCalls must be
	// exactly one. The store-level test (frontend/tests/store/
	// f11-sendable-and-root.test.ts) is what actually proves getRoot
	// touches exactly one row regardless of conversation length — this
	// fake cannot see row counts, only call counts, which is exactly the
	// limitation the findings named ("the call-counting test cannot see
	// it").
	assert.equal(readOlderCalls, 0, 'D8: the O(n) root-lookup path must never be called at all');
	assert.equal(getRootCalls, 1, 'D8: getRoot replaces it, called exactly once for a long conversation');
	if (outcome.ok) assert.equal(outcome.sentCount, 59);
});

test('runGate: a conversation shorter than N never needs the root-lookup read at all', async () => {
	const { chain, conv } = buildConversation({ turnCount: 5 });
	let readOlderCalls = 0;
	let getRootCalls = 0;
	const counting: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: (id) => chain.auditConversation(id),
		readTail: (id, limit) => chain.readTail(id, limit),
		readOlder: (id, before, limit) => {
			readOlderCalls += 1;
			return chain.readOlder(id, before, limit);
		},
		getRoot: (id) => {
			getRootCalls += 1;
			return chain.getRoot(id);
		}
	};
	await runGate(counting, conv.id, 60);
	assert.equal(readOlderCalls, 0);
	assert.equal(getRootCalls, 0, 'the whole chain already fit in the tail page — no root lookup of any kind needed');
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
	// call and its window read: the audit reports a STALE, much-shorter
	// sendable-suffix (D4: window_intent is now derived from
	// sendable_from_current, not chain_from_current — so THAT is the
	// field a race must stale for this test to still exercise the
	// intended scenario) — wildly different from the 65 turns readTail is
	// about to actually find.
	const racy: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: async (id) => {
			const real = await chain.auditConversation(id);
			if (!real) return real;
			// Report as if only 3 turns were sendable (sendable_from_current
			// = 4, i.e. 3 turns + the root).
			return { ...real, sendableFromCurrent: 4 };
		},
		readTail: (id, limit) => chain.readTail(id, limit),
		readOlder: (id, before, limit) => chain.readOlder(id, before, limit),
		getRoot: (id) => chain.getRoot(id)
	};
	const outcome = await runGate(racy, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.equal(outcome.windowIntent, 3); // the (stale) audit's prediction
	assert.notEqual(outcome.sentCount, 3); // the read found the real, longer chain
	assert.ok(outcome.reasons.some((r) => r.includes('intended 3')));
});

test('runGate: surviving-mutation audit #1 — an off-by-EXACTLY-ONE mismatch between windowIntent and the realized count must still refuse', async () => {
	// This is the test the surviving-mutation audit's own item #1 flags as
	// missing: "if (turns.length !== windowIntent) -> Math.abs(...) > 1
	// (sendSet.ts:244). GREEN. No test fails on a ±1 mismatch." Every OTHER
	// racy fixture in this file happens to produce either a genuine match
	// (0 apart) or a huge mismatch (56+ apart) — both stay correctly
	// detected (or correctly accepted) even with the count check relaxed
	// to `> 1`. This fixture is built SPECIFICALLY to differ by exactly
	// one: the real, healthy realization of a 65-turn/N=60 conversation is
	// 59 turns (the mandatory parity trim); a stale audit reporting
	// sendable_from_current as if turnsOnChain were 64 (one turn less than
	// the real 65) predicts windowIntent=60 (64 is even-offset, no trim
	// needed) — |59 - 60| = 1. The strict `!==` check catches this
	// immediately; a relaxed `Math.abs(...) > 1` would let it through as
	// `ok: true`, silently sending a window one turn short of what a
	// fresher read would have found — precisely the "1 or 2 messages
	// missing" shape the whole gate exists to make loud.
	const { chain, conv } = buildConversation({ turnCount: 65 });
	const racy: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: async (id) => {
			const real = await chain.auditConversation(id);
			if (!real) return real;
			// As if turnsOnChain were 64 (sendableFromCurrent = 65: 64 turns
			// + the root) instead of the real 65 — one turn stale.
			return { ...real, sendableFromCurrent: 65 };
		},
		readTail: (id, limit) => chain.readTail(id, limit),
		readOlder: (id, before, limit) => chain.readOlder(id, before, limit),
		getRoot: (id) => chain.getRoot(id)
	};
	const outcome = await runGate(racy, conv.id, 60);
	assert.equal(outcome.ok, false, 'a genuine off-by-one between intent and realization must still refuse');
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.equal(outcome.windowIntent, 60); // the stale audit's prediction (64 is an even offset -> no trim)
	assert.equal(outcome.sentCount, 59); // the real, healthy realization (65 is an odd offset -> the mandatory trim)
	assert.ok(outcome.reasons.some((r) => r.includes('realized 59') && r.includes('intended 60')));
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

test('runGate: an unlocatable root (getRoot returns null) is context_truncated, never a fabricated persona', async () => {
	const { chain, conv } = buildConversation({ turnCount: 65 });
	const blind: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: (id) => chain.auditConversation(id),
		readTail: (id, limit) => chain.readTail(id, limit),
		readOlder: (id, before, limit) => chain.readOlder(id, before, limit),
		// D8: the root lookup is getRoot now, not readOlder — simulate the
		// "root genuinely could not be found" case at that primitive
		// directly (e.g. a corrupt/cyclic chain the store itself flags).
		getRoot: async (): Promise<null> => null
	};
	const outcome = await runGate(blind, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.equal(outcome.systemMessage, null);
	assert.ok(outcome.reasons.some((r) => r.includes('could not be located')));
});

// ---------------------------------------------------------------------------
// D1 — the window must end on 'user', not merely open on one
// ---------------------------------------------------------------------------

test('runGate: D1 — a window ending on an assistant turn is refused, never blessed', async () => {
	// The EXACT original defect (docs/lanes/L2-gate-findings.md D1):
	// buildConversation's role assignment (i % 2 === 0 ? user : assistant)
	// puts the leaf on 'assistant' whenever turnCount is EVEN. The count
	// (6) equals windowIntent (6) — this is precisely why the count check
	// alone is only one bit (D3): it says nothing about which role sits at
	// either end. The ORIGINAL verifyRealizedWindow never checked the
	// CLOSING role at all, so this exact shape passed as `ok: true`.
	const { chain, conv } = buildConversation({ turnCount: 6 });
	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false, 'a window ending on assistant must never be ok:true');
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.ok(outcome.reasons.some((r) => r.includes('does not end on a user turn')));
	// The count check specifically must NOT be what caught this — proving
	// the new check is a genuinely separate signal, not a restatement of
	// the old one.
	assert.ok(!outcome.reasons.some((r) => r.includes('intended')));
});

// ---------------------------------------------------------------------------
// D2/D3 — the audited leaf must match the realized window's last turn
// ---------------------------------------------------------------------------

test('runGate: D2 — an equal-length branch switch between the audit read and the window read is caught via the audited leaf, not the count', async () => {
	const { chain, conv, root } = buildConversation({ turnCount: 5 }); // branch A, ends on user
	const leafA = chain.conv.currentLeafId as string;

	// Branch B: an independent branch off the SAME root, equal length and
	// equal alternation shape — count and shape checks alone cannot tell
	// these apart, which is the entire point of D2/D3.
	let parent = root.id;
	let leafB = root.id;
	for (let i = 0; i < 5; i++) {
		const role = i % 2 === 0 ? 'user' : 'assistant';
		const m = makeMessage({
			id: `branchB-${i}`,
			convId: conv.id,
			role,
			parentId: parent,
			content: JSON.stringify(`b-${i}`)
		});
		chain.addMessage(m);
		parent = m.id;
		leafB = m.id;
	}

	const racy: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		// Audits branch A — captured BEFORE the race below moves the leaf.
		auditConversation: (id) => chain.auditConversation(id),
		readTail: (id, limit) => {
			// The race: a concurrent selectLeaf moves the leaf to branch B
			// in the gap between the audit read (above) and this read.
			chain.conv.currentLeafId = leafB;
			return chain.readTail(id, limit);
		},
		readOlder: (id, before, limit) => chain.readOlder(id, before, limit),
		getRoot: (id) => chain.getRoot(id)
	};

	const outcome = await runGate(racy, conv.id, 60);
	assert.equal(outcome.ok, false, 'branch B must never be silently sent under branch A\'s verified intent');
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	// The count/shape checks alone would have passed (branch B is
	// internally healthy and the same length as branch A) — only the
	// audited-leaf comparison catches this.
	assert.ok(!outcome.reasons.some((r) => r.includes('intended')), 'the COUNT matched — this must be caught some other way');
	assert.ok(
		outcome.reasons.some((r) => r.includes('audited current leaf')),
		'the audited leaf (branch A) must not match the realized window\'s last turn (branch B)'
	);
	assert.notEqual(outcome.turns?.[outcome.turns.length - 1]?.id, leafA);
});

// ---------------------------------------------------------------------------
// D4 — non-`complete` messages
// ---------------------------------------------------------------------------

test('runGate: D4 — the current leaf is still streaming: nothing is sendable, refused with a typed reason, never a bare throw', async () => {
	const { chain, conv, turns } = buildConversation({ turnCount: 5 });
	const streamingLeaf = { ...turns[4], state: 'streaming' as const };
	chain.addMessage(streamingLeaf);

	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false, 'a streaming leaf must never be blessed as a sendable window');
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated'); // a typed reason, not a bare throw
	assert.ok(outcome.reasons.some((r) => r.includes('not complete') && r.includes('state=streaming')));
});

test('runGate: D4 — the current leaf failed and nobody has retried yet: nothing is sendable', async () => {
	const { chain, conv, turns } = buildConversation({ turnCount: 5 });
	const failedLeaf = { ...turns[4], state: 'failed' as const };
	chain.addMessage(failedLeaf);

	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, false);
	if (outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.kind, 'context_truncated');
	assert.ok(outcome.reasons.some((r) => r.includes('not complete') && r.includes('state=failed')));
});

test('runGate: D4 — a failed message on a dead SIBLING branch (never the active chain) does not block a healthy retry', async () => {
	// The documented consequence: because the store's state machine is
	// monotonic and `failed` is terminal, a retry appends a SIBLING under
	// the shared parent rather than continuing the failed row — so the
	// failed turn leaves the active chain as soon as the retry succeeds,
	// and the gate must see a perfectly healthy window.
	const { chain, conv, root } = buildConversation({ turnCount: 0 });
	const u1 = makeMessage({ id: 'u1', convId: conv.id, role: 'user', parentId: root.id, content: JSON.stringify('hi') });
	chain.addMessage(u1);
	const failedReply = makeMessage({
		id: 'failed-reply',
		convId: conv.id,
		role: 'assistant',
		parentId: u1.id,
		state: 'failed',
		content: JSON.stringify('')
	});
	chain.addMessage(failedReply);
	// Retry: a sibling under u1, never a child of the failed row.
	const retried = makeMessage({
		id: 'retried',
		convId: conv.id,
		role: 'assistant',
		parentId: u1.id,
		content: JSON.stringify('all better')
	});
	chain.addMessage(retried);
	const u2 = makeMessage({ id: 'u2', convId: conv.id, role: 'user', parentId: retried.id, content: JSON.stringify('thanks') });
	chain.addMessage(u2);
	chain.conv.currentLeafId = u2.id;

	const outcome = await runGate(chain, conv.id, 60);
	assert.equal(outcome.ok, true, 'the failed sibling never entered the active chain — nothing to refuse');
	if (!outcome.ok) throw new Error('unreachable');
	assert.equal(outcome.sentCount, 3); // u1, retried, u2 — NOT the failed reply
	assert.ok(!outcome.turns.some((t) => t.id === 'failed-reply'));
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
	// Surviving-mutation audit #2: `systemMessage: failure.systemMessage ??
	// null` -> `null` was GREEN against the suite before this assertion
	// existed — "neither override test asserts it. An override send would
	// lose the persona." This fixture's systemMessage is genuinely
	// non-null (the root was located fine; only the window's alternation
	// was corrupted), so a mutation that discards it to null is now caught
	// right here.
	assert.equal(record.systemMessage, failure.systemMessage);
	assert.ok(record.systemMessage, 'the persona must survive the override, not be discarded');

	// The defining property: re-running the gate on the SAME (still
	// corrupted) chain still fails the same way — proving applyOverride
	// did not fix, mutate, or otherwise cause the underlying data to pass.
	const rerun = await runGate(chain, conv.id, 60);
	assert.equal(rerun.ok, false);
});

test('applyOverride: a genuinely null systemMessage (root unlocatable) stays null — never fabricated', async () => {
	const { chain, conv } = buildConversation({ turnCount: 65 });
	const blind: ChainReader = {
		getConversation: (id) => chain.getConversation(id),
		auditConversation: (id) => chain.auditConversation(id),
		readTail: (id, limit) => chain.readTail(id, limit),
		readOlder: (id, before, limit) => chain.readOlder(id, before, limit),
		getRoot: async (): Promise<null> => null
	};
	const outcome = await runGate(blind, conv.id, 60);
	assert.equal(outcome.ok, false);
	const failure = outcome as GateFailure;
	assert.equal(failure.systemMessage, null);

	const record = applyOverride(failure);
	assert.equal(record.systemMessage, null, 'no persona was ever found — the override must not invent one');
});

test('applyOverride: originalWindowIntent is the INTENT, never silently substituted with sentCount', async () => {
	// Deliberately a scenario where windowIntent and sentCount are
	// DIFFERENT numbers — the alternation-corruption fixture above leaves
	// them accidentally equal (5 and 5), which would let a mutation that
	// swapped one field for the other slip through undetected. This is
	// the race fixture from the runGate tests above, reused here
	// specifically because windowIntent=3 and sentCount=59 there.
	const { chain, conv } = buildConversation({ turnCount: 65 });
	const racy = {
		getConversation: (id: string) => chain.getConversation(id),
		auditConversation: async (id: string) => {
			const real = await chain.auditConversation(id);
			return real ? { ...real, sendableFromCurrent: 4 } : real;
		},
		readTail: (id: string, limit: number) => chain.readTail(id, limit),
		readOlder: (id: string, before: string, limit: number) => chain.readOlder(id, before, limit),
		getRoot: (id: string) => chain.getRoot(id)
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
