import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';

test('create -> append -> read tail -> audit: the happy path', async () => {
	const db = await createTestDb({ prefix: 'basic' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({
			userId,
			systemContent: JSON.stringify('you are a helpful assistant')
		});
		assert.equal(conversation.rev, '0');
		assert.equal(root.parentId, null);
		assert.equal(root.role, 'system');

		const { message: userMsg, rev: rev1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: JSON.stringify('hello')
		});
		assert.equal(rev1, '1');
		assert.equal(userMsg.parentId, root.id);

		const { message: asstMsg, rev: rev2 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: userMsg.id,
			expectedRev: rev1,
			userId,
			role: 'assistant',
			content: JSON.stringify('hi there'),
			state: 'complete'
		});
		assert.equal(rev2, '2');

		const page = await db.store.readTail(conversation.id, 10);
		assert.equal(page.messages.length, 3);
		// newest first
		assert.equal(page.messages[0].id, asstMsg.id);
		assert.equal(page.messages[1].id, userMsg.id);
		assert.equal(page.messages[2].id, root.id);
		assert.equal(page.cursor, null); // root's parent is null -> no more pages

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.roots, 1);
		assert.equal(audit!.total, 3);
		assert.equal(audit!.reachableN, 3);
		assert.equal(audit!.missingParent, 0);
		assert.equal(audit!.leafOnTree, true);
		assert.equal(audit!.chainFromCurrent, 3);
		assert.equal(audit!.deepest, 3);
		assert.equal(audit!.pass, true);
	} finally {
		await db.close();
	}
});

test('pagination cursor stays valid across a branch switch', async () => {
	const db = await createTestDb({ prefix: 'branchpage' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({
			userId,
			systemContent: '"sys"'
		});
		let rev = conversation.rev;
		let parent = root.id;
		const shared: string[] = [root.id];
		for (let i = 0; i < 5; i++) {
			const { message, rev: newRev } = await db.store.appendMessage({
				convId: conversation.id,
				parentId: parent,
				expectedRev: rev,
				userId,
				role: i % 2 === 0 ? 'user' : 'assistant',
				content: JSON.stringify(`shared-${i}`)
			});
			shared.push(message.id);
			parent = message.id;
			rev = newRev;
		}
		const branchPoint = parent; // depth 5, shared by both branches
		const branchPointRev = rev;

		// branch A
		const a1 = await db.store.appendMessage({
			convId: conversation.id,
			parentId: branchPoint,
			expectedRev: branchPointRev,
			userId,
			role: 'user',
			content: '"branch-a"'
		});

		// read the tail (on branch A), then page back to somewhere in the shared prefix
		const tail = await db.store.readTail(conversation.id, 3);
		assert.equal(tail.messages[0].id, a1.message.id);
		const cursor = tail.cursor;
		assert.ok(cursor);
		const olderBeforeSwitch = await db.store.readOlder(conversation.id, cursor!, 10);
		assert.equal(olderBeforeSwitch.cursor, null); // walk reached the root

		// Now switch to a DIFFERENT branch B at the same branch point.
		await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: branchPoint,
			expectedRev: a1.rev
		});
		const b1 = await db.store.appendMessage({
			convId: conversation.id,
			parentId: branchPoint,
			expectedRev: (await db.store.getConversation(conversation.id))!.rev,
			userId,
			role: 'user',
			content: '"branch-b"'
		});
		assert.notEqual(b1.message.id, a1.message.id);

		// Continue paginating with the SAME cursor obtained while on branch A,
		// now that the current leaf points at branch B instead. It must
		// resolve to the IDENTICAL shared ancestry, because the cursor is a
		// message id / parent-pointer walk, not "wherever the leaf points
		// now" (FRONTEND_PLAN.md §3.1's "a cursor that stays valid across a
		// branch switch").
		const olderAfterSwitch = await db.store.readOlder(conversation.id, cursor!, 10);
		assert.deepEqual(
			olderAfterSwitch.messages.map((m) => m.id),
			olderBeforeSwitch.messages.map((m) => m.id)
		);
	} finally {
		await db.close();
	}
});
