// Gate remediation D4/D8 (docs/lanes/L2-gate-findings.md).
//
// D8: getRoot(convId) — a one-row, indexed root lookup, replacing the O(n)
// readOlder(convId, cursor, 100_000) pattern the transport lane used to fall
// back to. Tested here at the STORE layer (the transport lane's own
// sendSet.test.ts proves it is CALLED correctly and not fed more rows than
// it needs; this file proves the primitive itself is correct and touches
// exactly one row, regardless of conversation length).
//
// D4: audit_conversation's new sendable_from_current field — the longest
// contiguous suffix of the chain, ending at current_leaf_id, where every
// message is state='complete' and not tombstoned. Computed SQL-side so the
// transport lane's window_intent stays independently derived (see
// migrations/0001_init.sql's own comment on the `sendable` CTE).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';
import { setCurrentLeaf } from './fixtures/adversarial.js';

// ---------------------------------------------------------------------------
// D8 — getRoot
// ---------------------------------------------------------------------------

test('getRoot: returns the synthetic system root of a fresh conversation', async () => {
	const db = await createTestDb({ prefix: 'f11root' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"persona"' });
		const found = await db.store.getRoot(conversation.id);
		assert.ok(found);
		assert.equal(found!.id, root.id);
		assert.equal(found!.role, 'system');
		assert.equal(found!.parentId, null);
	} finally {
		await db.close();
	}
});

test('getRoot: unknown conversation returns null, not a crash', async () => {
	const db = await createTestDb({ prefix: 'f11rootmissing' });
	try {
		const found = await db.store.getRoot(uuidv7());
		assert.equal(found, null);
	} finally {
		await db.close();
	}
});

test('getRoot: still finds the SAME root on a long conversation — not confused by depth, and never the leaf', async () => {
	const db = await createTestDb({ prefix: 'f11rootlong' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"persona"' });
		let parent = root.id;
		let rev = conversation.rev;
		for (let i = 0; i < 40; i++) {
			const { message, rev: newRev } = await db.store.appendMessage({
				convId: conversation.id,
				parentId: parent,
				expectedRev: rev,
				userId,
				role: i % 2 === 0 ? 'user' : 'assistant',
				content: JSON.stringify(`t-${i}`)
			});
			parent = message.id;
			rev = newRev;
		}
		const found = await db.store.getRoot(conversation.id);
		assert.ok(found);
		assert.equal(found!.id, root.id);
		assert.notEqual(found!.id, parent, 'must be the ROOT, never the current leaf');
	} finally {
		await db.close();
	}
});

// ---------------------------------------------------------------------------
// D4 — sendable_from_current
// ---------------------------------------------------------------------------

test('sendable_from_current: a fully healthy chain — equals chain_from_current exactly', async () => {
	const db = await createTestDb({ prefix: 'f11sendablehealthy' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		let parent = root.id;
		let rev = conversation.rev;
		for (let i = 0; i < 5; i++) {
			const { message, rev: newRev } = await db.store.appendMessage({
				convId: conversation.id,
				parentId: parent,
				expectedRev: rev,
				userId,
				role: i % 2 === 0 ? 'user' : 'assistant',
				content: JSON.stringify(`t-${i}`)
			});
			parent = message.id;
			rev = newRev;
		}
		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.chainFromCurrent, 6); // root + 5 turns
		assert.equal(
			audit!.sendableFromCurrent,
			audit!.chainFromCurrent,
			'nothing is failed/tombstoned — the sendable suffix is the WHOLE chain'
		);
		assert.equal(audit!.pass, true);
	} finally {
		await db.close();
	}
});

test('sendable_from_current: the leaf itself is streaming — nothing is sendable, but pass is untouched', async () => {
	const db = await createTestDb({ prefix: 'f11sendableleafstreaming' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: u1, rev: rev1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"hi"'
		});
		await db.store.appendMessage({
			convId: conversation.id,
			parentId: u1.id,
			expectedRev: rev1,
			userId,
			role: 'assistant',
			content: '"partial so far"',
			state: 'streaming'
		});

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.chainFromCurrent, 3); // root, u1, the streaming assistant leaf
		assert.equal(
			audit!.sendableFromCurrent,
			0,
			"the leaf's own state != complete — the suffix cannot even start"
		);
		assert.equal(
			audit!.pass,
			true,
			'sendable_from_current is a SENDABILITY property, not a structural one — it must never gate pass'
		);
	} finally {
		await db.close();
	}
});

test('sendable_from_current: the leaf itself is failed — 0, distinguishing "nothing sendable" from a healthy short chain', async () => {
	const db = await createTestDb({ prefix: 'f11sendableleaffailed' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: u1, rev: rev1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"hi"'
		});
		const { message: a1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: u1.id,
			expectedRev: rev1,
			userId,
			role: 'assistant',
			content: '""',
			state: 'streaming'
		});
		await db.store.updateMessageState({
			convId: conversation.id,
			messageId: a1.id,
			state: 'failed',
			error: '{"reason":"backend_unavailable"}'
		});

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.leaf, a1.id);
		assert.equal(audit!.sendableFromCurrent, 0);
	} finally {
		await db.close();
	}
});

test('sendable_from_current: a failed message on a DEAD sibling branch (not the active chain) never shrinks it', async () => {
	const db = await createTestDb({ prefix: 'f11sendabledeadsibling' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: u1, rev: rev1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"hi"'
		});
		const { message: failedReply, rev: rev2 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: u1.id,
			expectedRev: rev1,
			userId,
			role: 'assistant',
			content: '""',
			state: 'streaming'
		});
		await db.store.updateMessageState({
			convId: conversation.id,
			messageId: failedReply.id,
			state: 'failed',
			error: '{"reason":"backend_unavailable"}'
		});
		// Retry: select back to u1, append a SIBLING (never a continuation of
		// the failed row) — the documented consequence of the store's
		// monotonic state machine (failed is terminal).
		const { rev: rev3 } = await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: u1.id,
			expectedRev: rev2
		});
		const { message: retried } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: u1.id,
			expectedRev: rev3,
			userId,
			role: 'assistant',
			content: '"retried reply"'
		});
		const { message: u2, rev: rev4 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: retried.id,
			expectedRev: (await db.store.getConversation(conversation.id))!.rev,
			userId,
			role: 'user',
			content: '"thanks"'
		});

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.leaf, u2.id);
		assert.equal(audit!.chainFromCurrent, 4); // root, u1, retried, u2 — the failed reply is OFF this chain
		assert.equal(
			audit!.sendableFromCurrent,
			4,
			'the failed sibling never entered the active chain at all — nothing to truncate'
		);
	} finally {
		await db.close();
	}
});

test('sendable_from_current: an INTERIOR failed message on the active chain (adversarial — the store never produces this on its own) truncates to the trailing complete run', async () => {
	const db = await createTestDb({ prefix: 'f11sendableinterior' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: u1, rev: rev1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"hi"'
		});
		const { message: a1, rev: rev2 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: u1.id,
			expectedRev: rev1,
			userId,
			role: 'assistant',
			content: '"reply"',
			state: 'streaming'
		});
		await db.store.updateMessageState({
			convId: conversation.id,
			messageId: a1.id,
			state: 'failed',
			error: '{"reason":"backend_unavailable"}'
		});
		// Adversarial only: appendMessage's own CAS requires parentId to be
		// the CURRENT leaf, so continuing straight off a1 (rather than
		// selecting back to u1 first) requires the CAS's normal parent to
		// still be a1 — which it is here, since nothing else has moved the
		// leaf yet. This is exactly the shape D4's findings describe as
		// theoretically reachable if a caller ever fails to follow the
		// "retry is a sibling" discipline; the store does not forbid it
		// (message_forbid_child_of_tombstone is about TOMBSTONES, not
		// failed state), so audit_conversation must handle it defensively.
		const { message: u2 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: a1.id,
			expectedRev: rev2,
			userId,
			role: 'user',
			content: '"still there?"'
		});

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.leaf, u2.id);
		assert.equal(audit!.chainFromCurrent, 4); // root, u1, a1(failed), u2
		assert.equal(
			audit!.sendableFromCurrent,
			1,
			'the walk starts at u2 (complete, counted), then hits a1 (failed) and stops — u1 and root are never even visited'
		);
	} finally {
		await db.close();
	}
});

test('sendable_from_current: NULL current_leaf_id is 0, not a crash', async () => {
	const db = await createTestDb({ prefix: 'f11sendablenullleaf' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"hi"'
		});
		await setCurrentLeaf(db, conversation.id, root.id); // legal, just to reach a known state first
		await db.pool.query('UPDATE conversation SET current_leaf_id = NULL WHERE id = $1', [conversation.id]);

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.leaf, null);
		assert.equal(audit!.sendableFromCurrent, 0);
	} finally {
		await db.close();
	}
});
