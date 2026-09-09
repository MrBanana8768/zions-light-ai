// Gate remediation F1 & F10. `grep -rn "tombstone\|deleted_at"
// frontend/tests/` found only two incidental hits in column lists before
// this file existed — nothing called tombstoneSubtree at all. This file
// covers:
//   - F10: descendants are tombstoned; it is idempotent; it is
//     conv_id-anchored in both the anchor and the recursive term.
//   - F1's two new refusals: tombstoning the subtree containing the
//     current leaf, and tombstoning the synthetic root.
//   - F1's database-level enforcement: a BEFORE INSERT trigger refuses a
//     live message parented under an already-tombstoned row — the two
//     definitions of "reachable" (the audit's downward walk, selectLeaf's
//     upward walk) can no longer disagree, because the state that let them
//     disagree (a live child of a dead parent) cannot exist.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';
import { setCurrentLeafNull } from './fixtures/adversarial.js';

test('tombstoneSubtree: the target and every descendant are tombstoned; untouched siblings are not', async () => {
	const db = await createTestDb({ prefix: 'tsdescendants' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });

		// root -> a (stays leaf); root -> b -> c -> d (the subtree to hide);
		// b -> e (a second child of b, also inside the subtree).
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
		const { message: c, rev: revC } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: b.id,
			expectedRev: revB,
			userId,
			role: 'user',
			content: '"c"'
		});
		const { message: d, rev: revD } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: c.id,
			expectedRev: revC,
			userId,
			role: 'user',
			content: '"d"'
		});
		// e is a second child of b — needs the leaf back at b first.
		const { rev: revAtB } = await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: b.id,
			expectedRev: revD
		});
		const { message: e, rev: revE } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: b.id,
			expectedRev: revAtB,
			userId,
			role: 'user',
			content: '"e"'
		});

		// Leaf back to `a`, well clear of b's whole branch, before hiding it.
		await db.store.selectLeaf({ convId: conversation.id, targetMessageId: a.id, expectedRev: revE });

		const count = await db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: b.id });
		assert.equal(count, 4, 'b, c, d, e — every node in the subtree, nothing more');

		const { rows } = await db.pool.query(
			`SELECT id, deleted_at FROM message WHERE conv_id = $1`,
			[conversation.id]
		);
		const deletedAtById = new Map(rows.map((r) => [r.id as string, r.deleted_at]));
		for (const id of [b.id, c.id, d.id, e.id]) {
			assert.ok(deletedAtById.get(id) !== null, `${id} must be tombstoned`);
		}
		for (const id of [root.id, a.id]) {
			assert.equal(deletedAtById.get(id), null, `${id} is outside the subtree and must be untouched`);
		}

		// The audit sees the tombstoned subtree as still structurally
		// intact — tombstones hide content, they do not orphan it.
		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.total, 6);
		assert.equal(audit!.reachableN, 6);
		assert.equal(audit!.missingParent, 0);
		assert.equal(audit!.pass, true);
	} finally {
		await db.close();
	}
});

test('tombstoneSubtree is idempotent: a second call on an already-tombstoned subtree tombstones nothing new and does not error', async () => {
	const db = await createTestDb({ prefix: 'tsidempotent' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
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
		await db.store.selectLeaf({ convId: conversation.id, targetMessageId: a.id, expectedRev: revB });

		const first = await db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: b.id });
		assert.equal(first, 1);

		const second = await db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: b.id });
		assert.equal(second, 0, 'nothing left to tombstone — already-tombstoned rows are excluded by deleted_at IS NULL');

		const third = await db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: b.id });
		assert.equal(third, 0, 'stays idempotent on a third call too');
	} finally {
		await db.close();
	}
});

test('tombstoneSubtree is conv_id-anchored: tombstoning a subtree in conversation A never touches conversation B, even with parallel structure', async () => {
	const db = await createTestDb({ prefix: 'tsanchored' });
	try {
		const userId = uuidv7();
		const convA = await db.store.createConversation({ userId, systemContent: '"sysA"' });
		const convB = await db.store.createConversation({ userId, systemContent: '"sysB"' });

		// Build the IDENTICAL shape in both conversations: root -> x -> y.
		const { message: xA, rev: revXA } = await db.store.appendMessage({
			convId: convA.conversation.id,
			parentId: convA.root.id,
			expectedRev: convA.conversation.rev,
			userId,
			role: 'user',
			content: '"x"'
		});
		const { message: yA } = await db.store.appendMessage({
			convId: convA.conversation.id,
			parentId: xA.id,
			expectedRev: revXA,
			userId,
			role: 'user',
			content: '"y"'
		});
		const { message: xB, rev: revXB } = await db.store.appendMessage({
			convId: convB.conversation.id,
			parentId: convB.root.id,
			expectedRev: convB.conversation.rev,
			userId,
			role: 'user',
			content: '"x"'
		});
		const { message: yB } = await db.store.appendMessage({
			convId: convB.conversation.id,
			parentId: xB.id,
			expectedRev: revXB,
			userId,
			role: 'user',
			content: '"y"'
		});

		// A's current leaf is yA, which is INSIDE the subtree we're about to
		// tombstone — move it back to the root first (F1's own refusal).
		const convANow = (await db.store.getConversation(convA.conversation.id))!;
		await db.store.selectLeaf({
			convId: convA.conversation.id,
			targetMessageId: convA.root.id,
			expectedRev: convANow.rev
		});

		const count = await db.store.tombstoneSubtree({ convId: convA.conversation.id, rootMessageId: xA.id });
		assert.equal(count, 2, 'xA and yA only');

		const { rows: aRows } = await db.pool.query(
			'SELECT id, deleted_at FROM message WHERE conv_id = $1 ORDER BY created_at',
			[convA.conversation.id]
		);
		assert.ok(aRows.find((r) => r.id === xA.id)!.deleted_at !== null);
		assert.ok(aRows.find((r) => r.id === yA.id)!.deleted_at !== null);

		// B, despite sharing the identical shape and even role/content
		// values, must be completely untouched — this is the property that
		// would fail if either the anchor or the recursive term of
		// tombstoneSubtree's subtree CTE were not itself conv_id-scoped.
		const { rows: bRows } = await db.pool.query(
			'SELECT id, deleted_at FROM message WHERE conv_id = $1',
			[convB.conversation.id]
		);
		for (const row of bRows) {
			assert.equal(row.deleted_at, null, `conversation B's ${row.id} must be untouched by A's tombstone`);
		}
		const auditB = await db.store.auditConversation(convB.conversation.id);
		assert.ok(auditB);
		assert.equal(auditB!.total, 3);
		assert.equal(auditB!.pass, true);
	} finally {
		await db.close();
	}
});

// --- Gate remediation F1 -----------------------------------------------

test('F1: tombstoneSubtree refuses when the subtree contains the current leaf (TombstoneRefusedError)', async () => {
	const db = await createTestDb({ prefix: 'f1leafinsubtree' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: m1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"m1"'
		});
		// The conversation's current leaf IS m1 — appendMessage already
		// moved it there. This is exactly F1's standing scenario: create ->
		// append M1 -> tombstoneSubtree(M1) used to succeed, hanging the
		// leaf under a dead parent the moment a message was appended to it.
		await assert.rejects(
			() => db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: m1.id }),
			/TombstoneRefusedError/
		);

		// Refused BEFORE touching any row.
		const { rows } = await db.pool.query('SELECT deleted_at FROM message WHERE id = $1', [m1.id]);
		assert.equal(rows[0].deleted_at, null);
		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.pass, true);
	} finally {
		await db.close();
	}
});

test('F1: tombstoneSubtree refuses when a DESCENDANT of the target (not the target itself) is the current leaf', async () => {
	const db = await createTestDb({ prefix: 'f1leafdescendant' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: m1, rev: rev1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"m1"'
		});
		const { message: m2 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: m1.id,
			expectedRev: rev1,
			userId,
			role: 'assistant',
			content: '"m2"'
		});
		// current leaf is m2, a descendant of m1, not m1 itself.
		await assert.rejects(
			() => db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: m1.id }),
			/TombstoneRefusedError/
		);
		void m2;
	} finally {
		await db.close();
	}
});

test('F1: tombstoneSubtree refuses to tombstone the synthetic root (TombstoneRefusedError)', async () => {
	const db = await createTestDb({ prefix: 'f1rootguard' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		// The root's own subtree is the WHOLE tree, so it always contains
		// whatever the current leaf is — the leaf-containment refusal would
		// fire regardless. This test's job is only to confirm the root
		// guard exists as ITS OWN check (see the mutation log: it is
		// verified independently by tombstoning a root with NO other
		// messages at all, where the leaf IS the root and containment is
		// trivially true either way).
		const { message: m1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"m1"'
		});
		void m1;
		await assert.rejects(
			() => db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: root.id }),
			/TombstoneRefusedError/
		);

		const { rows } = await db.pool.query('SELECT deleted_at FROM message WHERE id = $1', [root.id]);
		assert.equal(rows[0].deleted_at, null);
		// Without this guard, the audit would still report pass=true against
		// a conversation whose entire content is hidden — roots/reachable_n/
		// leaf_on_tree are all structural properties, blind to deleted_at.
		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.pass, true, 'refused, so nothing changed — this assertion is about the refusal, not a hidden-root PASS');
	} finally {
		await db.close();
	}
});

// The test above cannot, by itself, prove the root guard is its OWN check
// rather than a restatement of the leaf-containment refusal: tombstoning
// the root always targets a subtree that is the WHOLE tree, which always
// contains wherever the current leaf is — so as long as current_leaf_id is
// non-null, the leaf-containment check would refuse the same call for a
// DIFFERENT reason, silently covering for a missing root guard. This test
// isolates it: with current_leaf_id forced to NULL (F2's own fixture), the
// leaf-containment check's own `currentLeafId !== null` guard short-circuits
// to false and never fires — so the ONLY thing that can refuse tombstoning
// the root here is the root guard itself.
test('F1: the root guard is its own, independent check — isolated from leaf-containment via a NULL current_leaf_id', async () => {
	const db = await createTestDb({ prefix: 'f1rootguardisolated' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		await setCurrentLeafNull(db, conversation.id);

		await assert.rejects(
			() => db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: root.id }),
			/TombstoneRefusedError/
		);

		const { rows } = await db.pool.query('SELECT deleted_at FROM message WHERE id = $1', [root.id]);
		assert.equal(rows[0].deleted_at, null);
	} finally {
		await db.close();
	}
});

// --- Gate remediation F1's database-level enforcement -------------------

test('F1: a live message cannot be appended under an already-tombstoned parent (message_forbid_child_of_tombstone trigger)', async () => {
	const db = await createTestDb({ prefix: 'f1childoftombstone' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: m1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"m1"'
		});

		// Move the leaf off m1 (tombstoneSubtree refuses to hide the
		// current leaf — F1's OTHER refusal, tested above) so m1 can
		// legitimately be tombstoned on its own.
		await db.store.selectLeaf({ convId: conversation.id, targetMessageId: root.id, expectedRev: (await db.store.getConversation(conversation.id))!.rev });
		await db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: m1.id });

		// Directly at the database, bypassing appendMessage's own CAS
		// entirely: even a raw INSERT naming m1 as parent_id must be
		// refused, because m1 is now tombstoned. This is the trigger that
		// makes "a live child of a dead parent" impossible regardless of
		// which code path (present or future) attempts it — not merely
		// something appendMessage happens to avoid today.
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,$3,$4,'user','"orphan"'::jsonb)`,
					[uuidv7(), conversation.id, m1.id, userId]
				),
			/tombstoned/
		);
	} finally {
		await db.close();
	}
});

test('F1: the same trigger rejects appendMessage() (the public API), not merely a raw INSERT', async () => {
	const db = await createTestDb({ prefix: 'f1childoftombstoneapi' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: m1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"m1"'
		});
		const convBeforeSelect = (await db.store.getConversation(conversation.id))!;
		const { rev: revAtRoot } = await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: root.id,
			expectedRev: convBeforeSelect.rev
		});
		await db.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: m1.id });

		// The exact scenario named by the review: create -> append M1 ->
		// tombstoneSubtree(M1) -> appendMessage(parent=M1). appendMessage's
		// own INSERT runs BEFORE its CAS check, so the trigger fires first
		// and raises regardless of what the CAS would otherwise have
		// decided (current_leaf_id is root here, not m1, so the CAS would
		// have rejected this too, for the unrelated reason of a stale
		// parent — the trigger firing first is what proves the DATABASE,
		// not appendMessage's own CAS logic, is what makes this
		// impossible).
		await assert.rejects(
			() =>
				db.store.appendMessage({
					convId: conversation.id,
					parentId: m1.id,
					expectedRev: revAtRoot,
					userId,
					role: 'user',
					content: '"too late"'
				}),
			/tombstoned/
		);
	} finally {
		await db.close();
	}
});
