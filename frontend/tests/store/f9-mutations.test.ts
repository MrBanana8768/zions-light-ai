// Gate remediation F9 — untried mutations the adversarial review
// identified. Each of these survived the pre-existing suite green because
// nothing exercised the specific property named.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';

// 1. conversation_leaf_fk is a COMPOSITE FK on (id, current_leaf_id) ->
// message(conv_id, id). Dropping `id,` from that composite (leaving a bare
// FK on current_leaf_id -> message(id)) survives the suite green today
// because no test ever points one conversation's leaf at a REAL message
// that belongs to a DIFFERENT conversation — every existing "unreachable
// leaf" test uses a uuid that exists nowhere at all, which a bare FK would
// also reject. message_parent_fk's identical composite-ness IS tested
// (f4-rejection.test.ts's cross-conversation-parent case); this is that
// same rule, tested at one call site and missed at its sibling.
test('F9: the leaf pointer cannot name a real message of a DIFFERENT conversation (conversation_leaf_fk, composite on id)', async () => {
	const db = await createTestDb({ prefix: 'f9leaffk' });
	try {
		const userId = uuidv7();
		const { conversation: convA } = await db.store.createConversation({ userId, systemContent: '"sysA"' });
		const { root: rootB } = await db.store.createConversation({ userId, systemContent: '"sysB"' });

		// rootB is a REAL, existing message row — just not of convA. A bare
		// FK on current_leaf_id -> message(id) alone would accept this; the
		// composite (id, current_leaf_id) -> message(conv_id, id) must not.
		await assert.rejects(
			() =>
				db.pool.query('UPDATE conversation SET current_leaf_id = $1 WHERE id = $2', [
					rootB.id,
					convA.id
				]),
			/conversation_leaf_fk/
		);
	} finally {
		await db.close();
	}
});

// 2. message.conv_id -> conversation(id) ON DELETE RESTRICT. Changing this
// to CASCADE survives the suite green today because nothing issues DELETE
// FROM conversation — the store's own write API never does (there is no
// delete-a-conversation operation at all; tombstoning is per-message). A
// human with psql, or a future admin route, could still issue one; RESTRICT
// is what stands between that and silently vaporising every message the
// conversation ever held.
test('F9: deleting a conversation with live messages is rejected (message.conv_id ON DELETE RESTRICT, not CASCADE)', async () => {
	const db = await createTestDb({ prefix: 'f9convdelete' });
	try {
		const userId = uuidv7();
		const { conversation } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		// The synthetic root alone is enough — RESTRICT fires on any live
		// child row, including the root itself.
		await assert.rejects(
			() => db.pool.query('DELETE FROM conversation WHERE id = $1', [conversation.id]),
			/message_conv_id_fkey|violates foreign key constraint/
		);

		const { rows } = await db.pool.query('SELECT 1 FROM conversation WHERE id = $1', [conversation.id]);
		assert.equal(rows.length, 1, 'the conversation must still exist — the DELETE must have been refused, not partially applied');
	} finally {
		await db.close();
	}
});

// 3. `typeof message.content === 'string'` is never tested anywhere. The
// raw-text carriage (§3.2's byte-stability obligation) rests entirely on
// db.ts's types.setTypeParser(JSONB, ...) override — a mutable PROCESS-WIDE
// global. If that override is ever missing or overwritten by something
// else in the same process, node-postgres's DEFAULT jsonb parser kicks in,
// row.content becomes a parsed JS value (object/array/number/etc), and
// `row.content as string` (store.ts's rowToMessage) happily lies about the
// type — `as` performs no runtime check. This test proves the full round
// trip actually yields a string today, for content shapes representative
// of the wire form (§5.1 content-parts: a plain string body and a
// multi-part array body), not merely that TypeScript's type-checker is
// satisfied.
test('F9: message.content round-trips as a raw STRING end to end, for both a scalar and an array wire shape', async () => {
	const db = await createTestDb({ prefix: 'f9contentstring' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		assert.equal(typeof root.content, 'string', 'the synthetic root\'s own content must already be a string');

		const scalarWire = JSON.stringify('hello, world');
		const { message: scalarMsg } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: scalarWire
		});
		assert.equal(typeof scalarMsg.content, 'string');
		assert.equal(scalarMsg.content, scalarWire);

		const arrayWire = JSON.stringify([
			{ type: 'text', text: 'part one' },
			{ type: 'text', text: 'part two' }
		]);
		const convAfterFirst = (await db.store.getConversation(conversation.id))!;
		const { message: arrayMsg } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: scalarMsg.id,
			expectedRev: convAfterFirst.rev,
			userId,
			role: 'user',
			content: arrayWire
		});
		assert.equal(typeof arrayMsg.content, 'string');
		assert.equal(JSON.parse(arrayMsg.content).length, 2);

		// And on the READ path — readTail / readOlder — not just the write
		// return value, since a caller that never re-reads the INSERT's
		// own RETURNING row (the common case) only ever sees content via
		// these.
		const page = await db.store.readTail(conversation.id, 10);
		for (const m of page.messages) {
			assert.equal(typeof m.content, 'string', `readTail's ${m.id} must carry content as a string`);
		}

		// And through auditConversation-adjacent raw access is out of
		// scope (audit never returns content) — but getConversation's
		// sibling read of a message directly, via a fresh pool.query, is
		// the one place a MISSING type-parser override would be visible
		// even though rowToMessage is never involved: prove the RAW driver
		// read (not store.ts's own mapping) is also a string, which is
		// what actually confirms db.ts's global override, not just
		// rowToMessage's cast.
		const raw = await db.pool.query('SELECT content FROM message WHERE id = $1', [arrayMsg.id]);
		assert.equal(typeof raw.rows[0].content, 'string', 'the pg driver itself must hand back a string, not a parsed object — this is what db.ts\'s type-parser override guarantees, and what an `as string` cast cannot');
	} finally {
		await db.close();
	}
});
