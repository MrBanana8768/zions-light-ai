// F4 — the store must REJECT, and reject at the database, not at a guard
// clause in this codebase. Every test in this file issues SQL directly
// against db.pool (bypassing Store's own methods entirely) so a passing
// test can only mean the constraint itself did the rejecting. See
// docs/lanes/L1-store.md for the mutation log proving each of these five
// actually depends on its named constraint (constraint temporarily removed
// from the migration -> this exact test goes red -> constraint restored ->
// green again).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';

async function seedConversation(db: Awaited<ReturnType<typeof createTestDb>>) {
	const userId = uuidv7();
	const convId = uuidv7();
	const rootId = uuidv7();
	// One transaction, matching Store.createConversation: conversation_leaf_fk
	// is DEFERRABLE INITIALLY DEFERRED specifically so these two inserts (each
	// forward-referencing the other for an instant) only need to satisfy the
	// FK by COMMIT, not per-statement. Two separate pool.query() calls would
	// each auto-commit on their own and the first would fail outright.
	const client = await db.pool.connect();
	try {
		await client.query('BEGIN');
		await client.query(
			'INSERT INTO conversation (id, user_id, current_leaf_id) VALUES ($1,$2,$3)',
			[convId, userId, rootId]
		);
		await client.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,NULL,$3,'system','"sys"'::jsonb)`,
			[rootId, convId, userId]
		);
		await client.query('COMMIT');
	} catch (err) {
		await client.query('ROLLBACK');
		throw err;
	} finally {
		client.release();
	}
	return { userId, convId, rootId };
}

test('F4: a second root is rejected (message_one_root_per_conv)', async () => {
	const db = await createTestDb({ prefix: 'f4root' });
	try {
		const { convId, userId } = await seedConversation(db);
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,NULL,$3,'system','"root2"'::jsonb)`,
					[uuidv7(), convId, userId]
				),
			/message_one_root_per_conv/
		);
	} finally {
		await db.close();
	}
});

test('F4: a message with an unknown parent is rejected (message_parent_fk)', async () => {
	const db = await createTestDb({ prefix: 'f4unknown' });
	try {
		const { convId, userId } = await seedConversation(db);
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,$3,$4,'user','"x"'::jsonb)`,
					[uuidv7(), convId, uuidv7() /* nonexistent parent */, userId]
				),
			/message_parent_fk/
		);
	} finally {
		await db.close();
	}
});

test('F4: a message parented into another conversation is rejected (message_parent_fk, composite on conv_id)', async () => {
	const db = await createTestDb({ prefix: 'f4crossconv' });
	try {
		const { rootId: rootA } = await seedConversation(db);
		const { convId: convB, userId: userB } = await seedConversation(db);
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,$3,$4,'user','"x"'::jsonb)`,
					[uuidv7(), convB, rootA /* belongs to a DIFFERENT conversation */, userB]
				),
			/message_parent_fk/
		);
	} finally {
		await db.close();
	}
});

test('F4: a pointer move to an unreachable (nonexistent/foreign) node is rejected (conversation_leaf_fk)', async () => {
	const db = await createTestDb({ prefix: 'f4leaffk' });
	try {
		const { convId } = await seedConversation(db);
		await assert.rejects(
			() => db.pool.query('UPDATE conversation SET current_leaf_id = $1 WHERE id = $2', [uuidv7(), convId]),
			/conversation_leaf_fk/
		);
	} finally {
		await db.close();
	}
});

test('F4: an update that changes parent_id is rejected (message_forbid_structural_update trigger)', async () => {
	const db = await createTestDb({ prefix: 'f4reparent' });
	try {
		const { convId, userId, rootId } = await seedConversation(db);
		const childId = uuidv7();
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,$3,$4,'user','"hi"'::jsonb)`,
			[childId, convId, rootId, userId]
		);
		await assert.rejects(
			() => db.pool.query('UPDATE message SET parent_id = NULL WHERE id = $1', [childId]),
			/immutable/
		);
	} finally {
		await db.close();
	}
});

// Bonus, not part of F4's five: id and conv_id are covered by the same
// trigger as parent_id. Recorded here because the trigger function handles
// all three in one body and a mutation to it could plausibly break one
// column's check while leaving the others intact.
test('F4 (bonus): updates to id/conv_id are also rejected by the same trigger', async () => {
	const db = await createTestDb({ prefix: 'f4immutable' });
	try {
		const { convId, userId, rootId } = await seedConversation(db);
		const childId = uuidv7();
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,$3,$4,'user','"hi"'::jsonb)`,
			[childId, convId, rootId, userId]
		);
		await assert.rejects(
			() => db.pool.query('UPDATE message SET id = $1 WHERE id = $2', [uuidv7(), childId]),
			/immutable/
		);
		await assert.rejects(
			() => db.pool.query('UPDATE message SET conv_id = $1 WHERE id = $2', [uuidv7(), childId]),
			/immutable/
		);
	} finally {
		await db.close();
	}
});

test('F4 (bonus): a message cannot be its own parent (message_not_self_parent)', async () => {
	const db = await createTestDb({ prefix: 'f4selfparent' });
	try {
		const { convId, userId } = await seedConversation(db);
		const id = uuidv7();
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,$1,$3,'user','"x"'::jsonb)`,
					[id, convId, userId]
				),
			/message_not_self_parent/
		);
	} finally {
		await db.close();
	}
});

// Added after this lane's mutation log found the gap: NOTHING in this suite
// ever issued a raw DELETE, so message_parent_fk's `ON DELETE RESTRICT`
// (the clause standing between a hard delete and "the remaining way to
// manufacture an orphaned subtree" — FRONTEND_SPEC.md §11.1) was
// completely unexercised. Changing it to `ON DELETE CASCADE` survived the
// entire suite green before this test existed — see docs/lanes/L1-store.md's
// mutation log, mutation #6.
//
// The node being deleted is deliberately a SIBLING branch off the root, not
// on the current_leaf_id path — deleting an ANCESTOR of the current leaf
// would also trip conversation_leaf_fk (the leaf pointer itself dangling),
// which fires regardless of message_parent_fk's ON DELETE mode and would
// mask exactly the property this test exists to isolate.
test('F4 (bonus): deleting a message with a live child is rejected (message_parent_fk ON DELETE RESTRICT)', async () => {
	const db = await createTestDb({ prefix: 'f4deleterestrict' });
	try {
		const { convId, userId, rootId } = await seedConversation(db);
		// The leaf's own branch — untouched by the delete attempt below.
		const leafChildId = uuidv7();
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,$3,$4,'user','"leaf branch"'::jsonb)`,
			[leafChildId, convId, rootId, userId]
		);
		await db.pool.query('UPDATE conversation SET current_leaf_id = $1 WHERE id = $2', [leafChildId, convId]);

		// A SIBLING branch off the same root, with its own child — this is
		// what gets targeted, and it shares no row with current_leaf_id.
		const siblingId = uuidv7();
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,$3,$4,'user','"sibling"'::jsonb)`,
			[siblingId, convId, rootId, userId]
		);
		const siblingChildId = uuidv7();
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,$3,$4,'user','"sibling child"'::jsonb)`,
			[siblingChildId, convId, siblingId, userId]
		);

		// siblingId has a live child (siblingChildId) referencing it via
		// message_parent_fk — deleting it must be REFUSED, not cascade into
		// silently deleting the child too (which would manufacture exactly
		// the "hard delete orphans a subtree" failure §11.1 forbids).
		await assert.rejects(
			() => db.pool.query('DELETE FROM message WHERE id = $1', [siblingId]),
			/message_parent_fk/
		);
	} finally {
		await db.close();
	}
});

test('F4 (bonus, via the Store API): appendMessage against a stale/foreign parent loses via StaleWriteError, not a silent overwrite', async () => {
	const db = await createTestDb({ prefix: 'f4stale' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({
			userId,
			systemContent: '"sys"'
		});
		// The CAS requires parentId === current_leaf_id AND rev === expectedRev.
		// Passing the right parent but a WRONG (stale) rev must be rejected
		// even though the insert itself would satisfy every DDL constraint.
		await assert.rejects(
			() =>
				db.store.appendMessage({
					convId: conversation.id,
					parentId: root.id,
					expectedRev: '999',
					userId,
					role: 'user',
					content: '"hi"'
				}),
			/StaleWriteError/
		);
	} finally {
		await db.close();
	}
});
