// F6 — a schema test: no serialized chain exists anywhere.
// FRONTEND_SPEC.md §11.1: "The chain is represented exactly once, as
// message records each carrying its own parent_id. No record anywhere may
// hold a serialized copy of the chain: no `messages` array on the
// conversation, no `history` object, no id-keyed message map, no column
// that must be rewritten when a message is appended. A schema test must
// assert this." — written as a test because as prose it did not stop the
// failure shipping (OpenWebUI's own `chat.chat` blob column).
//
// Introspects information_schema — never eyeballs the DDL — and pairs the
// structural check with a live behavioural one: appending new messages must
// never rewrite an earlier message row's bytes.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';

// A column NAME suggesting a serialized collection of messages — the shape
// OpenWebUI's chat.chat blob and chat_message's sibling table both are.
// Deliberately does not match `message` (the table itself, singular) or
// `messages` used as a plain English word elsewhere in a comment — this
// only runs against actual column identifiers.
const SERIALIZED_CHAIN_NAME = /(^|_)(messages|history|chain|thread|msgs)($|_)/i;

test('F6 (schema): no column anywhere is an array type, and no column name suggests a serialized chain', async () => {
	const db = await createTestDb({ prefix: 'f6names' });
	try {
		const { rows } = await db.pool.query(
			`SELECT table_name, column_name, data_type, udt_name
			 FROM information_schema.columns
			 WHERE table_schema = current_schema()
			 ORDER BY table_name, ordinal_position`
		);
		assert.ok(rows.length > 0, 'sanity: the introspection query must see the migrated schema');

		for (const row of rows) {
			assert.notEqual(
				row.data_type,
				'ARRAY',
				`${row.table_name}.${row.column_name} is an array column (${row.udt_name}) — an array is ` +
					`exactly the "column that must be rewritten when a message is appended" §11.1 forbids`
			);
			assert.ok(
				!SERIALIZED_CHAIN_NAME.test(row.column_name),
				`${row.table_name}.${row.column_name}'s name suggests a serialized chain (messages/history/` +
					`chain/thread) — §11.1 forbids representing the chain anywhere but per-row parent_id links`
			);
		}
	} finally {
		await db.close();
	}
});

test('F6 (schema): the only jsonb/json columns anywhere are message.content and message.error, each scoped to one row', async () => {
	const db = await createTestDb({ prefix: 'f6jsonb' });
	try {
		const { rows } = await db.pool.query(
			`SELECT table_name, column_name
			 FROM information_schema.columns
			 WHERE table_schema = current_schema() AND data_type IN ('json', 'jsonb')
			 ORDER BY table_name, column_name`
		);
		assert.deepEqual(
			rows.map((r) => `${r.table_name}.${r.column_name}`),
			['message.content', 'message.error'],
			"a jsonb/json column outside message.content/message.error is exactly where a serialized " +
				"whole-chain blob would live (OpenWebUI's own chat.chat column is this shape)"
		);
	} finally {
		await db.close();
	}
});

test('F6 (schema): conversation carries no jsonb/json/array column at all — the leaf pointer is the only "position" it holds', async () => {
	const db = await createTestDb({ prefix: 'f6convo' });
	try {
		const { rows } = await db.pool.query(
			`SELECT column_name, data_type
			 FROM information_schema.columns
			 WHERE table_schema = current_schema() AND table_name = 'conversation'
			   AND (data_type IN ('json', 'jsonb') OR data_type = 'ARRAY')`
		);
		assert.deepEqual(rows, []);
	} finally {
		await db.close();
	}
});

test('F6 (behavioural): appending new messages never rewrites an earlier message row', async () => {
	const db = await createTestDb({ prefix: 'f6append' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		let rev = conversation.rev;
		let parent = root.id;
		const earlyIds: string[] = [root.id];
		for (let i = 0; i < 5; i++) {
			const { message, rev: newRev } = await db.store.appendMessage({
				convId: conversation.id,
				parentId: parent,
				expectedRev: rev,
				userId,
				role: i % 2 === 0 ? 'user' : 'assistant',
				content: JSON.stringify(`early-${i}`)
			});
			earlyIds.push(message.id);
			parent = message.id;
			rev = newRev;
		}

		// Snapshot the earlier rows in full BEFORE 20 more appends.
		const before = await db.pool.query(
			`SELECT id, conv_id, parent_id, user_id, role, content, state, error, created_at, updated_at, deleted_at
			 FROM message WHERE id = ANY($1) ORDER BY id`,
			[earlyIds]
		);
		assert.equal(before.rows.length, earlyIds.length);

		for (let i = 0; i < 20; i++) {
			const { message, rev: newRev } = await db.store.appendMessage({
				convId: conversation.id,
				parentId: parent,
				expectedRev: rev,
				userId,
				role: i % 2 === 0 ? 'user' : 'assistant',
				content: JSON.stringify(`later-${i}`)
			});
			parent = message.id;
			rev = newRev;
		}

		const after = await db.pool.query(
			`SELECT id, conv_id, parent_id, user_id, role, content, state, error, created_at, updated_at, deleted_at
			 FROM message WHERE id = ANY($1) ORDER BY id`,
			[earlyIds]
		);

		assert.deepEqual(
			after.rows,
			before.rows,
			'a message row already written must be byte-for-byte identical after 20 more appends elsewhere ' +
				'in the chain — nothing "rewrites" it. (conversation.rev/updated_at, the O(1) leaf pointer, ' +
				'are expected to change and are deliberately not part of this snapshot.)'
		);
	} finally {
		await db.close();
	}
});
