// Gate remediation F4 — selectLeaf's TOCTOU. Its reachability predicate was
// a plain SELECT under READ COMMITTED, then it CASed on `rev` alone — and
// tombstoneSubtree bumped no `rev` and took no lock on `conversation`. So:
// A reads a clean path -> B tombstones an ancestor -> A's CAS on rev still
// matches -> the leaf lands inside a tombstoned subtree. Fixed with
// `SELECT ... FOR UPDATE` on the conversation row as the FIRST statement of
// both selectLeaf's and tombstoneSubtree's transactions, so the two
// serialize instead of interleaving.
//
// Both tests below interleave two REAL connections deterministically —
// via explicit, hand-controlled statement ordering, never a timing-based
// sleep — so the outcome does not depend on scheduler luck. Each test
// forces the exact race the review named and asserts the loser fails
// LOUDLY (a typed, recognisable error) rather than corrupting state.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';

test('F4: selectLeaf, blocked behind a concurrent tombstone holding the conversation lock, re-checks FRESH state and rejects (UnreachableLeafError) rather than landing inside the now-hidden subtree', async () => {
	const db = await createTestDb({ prefix: 'f4toctou1' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });

		// root -> A (current leaf); root -> B -> C (sibling branch, never
		// current — this is the subtree a concurrent tombstone will target).
		const { message: a, rev: revA } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"a"'
		});
		const { rev: revAtRoot } = await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: root.id,
			expectedRev: revA
		});
		const { message: b, rev: revB } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: revAtRoot,
			userId,
			role: 'user',
			content: '"b"'
		});
		const { message: c } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: b.id,
			expectedRev: revB,
			userId,
			role: 'user',
			content: '"c"'
		});
		// Move current leaf back to `a`, well clear of B's branch.
		await db.store.selectLeaf({ convId: conversation.id, targetMessageId: a.id, expectedRev: (await db.store.getConversation(conversation.id))!.rev });
		const revBeforeRace = (await db.store.getConversation(conversation.id))!.rev;

		// The "tombstone" side: a raw connection replicating exactly
		// tombstoneSubtree's OWN locking discipline (SELECT ... FOR UPDATE
		// first) by hand, so this test controls precisely when the lock is
		// released — no sleep, no timing assumption.
		const lockHolder = await db.pool.connect();
		await lockHolder.query('BEGIN');
		await lockHolder.query('SELECT current_leaf_id FROM conversation WHERE id = $1 FOR UPDATE', [conversation.id]);

		// Fire the REAL selectLeaf call now, while lockHolder still holds
		// the row lock. It must block on its OWN `SELECT ... FOR UPDATE`
		// (the F4 fix) rather than proceeding on a stale read.
		const selectLeafPromise = db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: c.id,
			expectedRev: revBeforeRace
		});

		// Give the blocked query a moment to actually reach Postgres and
		// enter the lock wait queue before we decide the race by committing.
		await new Promise((resolve) => setTimeout(resolve, 150));

		// Apply the tombstone (B, C) while selectLeaf is still blocked, then
		// release the lock.
		await lockHolder.query(
			`UPDATE message SET deleted_at = now(), updated_at = now() WHERE conv_id = $1 AND id = ANY($2::uuid[])`,
			[conversation.id, [b.id, c.id]]
		);
		await lockHolder.query('COMMIT');
		lockHolder.release();

		// selectLeaf, now unblocked, must see the FRESH (tombstoned) state
		// and reject loudly — never silently CAS the leaf into a subtree
		// that was hidden while it waited.
		await assert.rejects(selectLeafPromise, /UnreachableLeafError/);

		const finalConv = await db.store.getConversation(conversation.id);
		assert.equal(finalConv!.currentLeafId, a.id, 'the leaf must still be `a` — the loser never got to write');
		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.leafOnTree, true);
		assert.equal(audit!.pass, true, 'no corruption: the current leaf is never the tombstoned node');
	} finally {
		await db.close();
	}
});

test('F4: tombstoneSubtree, blocked behind a concurrent leaf move holding the conversation lock, re-checks FRESH current_leaf_id and refuses (TombstoneRefusedError) rather than hiding the node the conversation now points at', async () => {
	const db = await createTestDb({ prefix: 'f4toctou2' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });

		// root -> A -> M (current leaf); root -> A -> P -> Q (a second
		// branch off A, not yet current — this is the subtree a concurrent
		// tombstoneSubtree(P) will target).
		const { message: aNode, rev: revA } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"a"'
		});
		const { message: m, rev: revM } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: aNode.id,
			expectedRev: revA,
			userId,
			role: 'assistant',
			content: '"m"'
		});
		const { rev: revAtA } = await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: aNode.id,
			expectedRev: revM
		});
		const { message: p, rev: revP } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: aNode.id,
			expectedRev: revAtA,
			userId,
			role: 'assistant',
			content: '"p"'
		});
		const { message: q } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: p.id,
			expectedRev: revP,
			userId,
			role: 'user',
			content: '"q"'
		});
		// Move current leaf back to M — clear of P/Q — before the race.
		const convBeforeMove = (await db.store.getConversation(conversation.id))!;
		await db.store.selectLeaf({ convId: conversation.id, targetMessageId: m.id, expectedRev: convBeforeMove.rev });
		const revBeforeRace = (await db.store.getConversation(conversation.id))!.rev;

		// The "select_leaf" side: a raw connection replicating selectLeaf's
		// OWN locking discipline by hand — hand-controlled, not
		// timing-based — so this test decides exactly when the lock is
		// released.
		const lockHolder = await db.pool.connect();
		await lockHolder.query('BEGIN');
		await lockHolder.query('SELECT id FROM conversation WHERE id = $1 FOR UPDATE', [conversation.id]);

		// Fire the REAL tombstoneSubtree call now, targeting P (whose
		// subtree is {P, Q}), while lockHolder still holds the row lock. It
		// must block on ITS OWN `SELECT ... FOR UPDATE` (the F4 fix).
		const tombstonePromise = db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: p.id });

		await new Promise((resolve) => setTimeout(resolve, 150));

		// Move the leaf to Q — INSIDE the subtree tombstoneSubtree is about
		// to act on — while it is still blocked, then release the lock.
		await lockHolder.query(
			`UPDATE conversation SET current_leaf_id = $1, rev = rev + 1, updated_at = now() WHERE id = $2 AND rev = $3`,
			[q.id, conversation.id, revBeforeRace]
		);
		await lockHolder.query('COMMIT');
		lockHolder.release();

		// tombstoneSubtree, now unblocked, must see the FRESH current_leaf_id
		// (Q) and refuse — never hide the node the conversation now points at.
		await assert.rejects(tombstonePromise, /TombstoneRefusedError/);

		const { rows } = await db.pool.query('SELECT id, deleted_at FROM message WHERE id = ANY($1::uuid[])', [[p.id, q.id]]);
		for (const row of rows) {
			assert.equal(row.deleted_at, null, `${row.id} must be untouched — the tombstone was refused`);
		}
		const finalConv = await db.store.getConversation(conversation.id);
		assert.equal(finalConv!.currentLeafId, q.id);
		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.leafOnTree, true);
		assert.equal(audit!.pass, true, 'no corruption: the current leaf is never tombstoned');
	} finally {
		await db.close();
	}
});
