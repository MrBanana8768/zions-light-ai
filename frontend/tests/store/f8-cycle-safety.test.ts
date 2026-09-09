// Gate remediation F8. audit_conversation's `down` walk is provably
// cycle-safe (see the migration SQL's own comment on that function) — it
// only ever starts at a root and moves parent -> child, and a cycle member
// necessarily has a non-null parent_id, so it can never be a root and can
// never be walked into from one. selectLeaf's `up` walk and
// tombstoneSubtree's `subtree` walk go the OTHER direction, from an
// arbitrary starting row, and have no such immunity — on cyclic data
// (constructible only by a human repairing the chain from outside the app,
// §11.5, since the schema's own constraints forbid a cycle through any
// path the public Store API can reach) an unbounded `UNION ALL` recursive
// CTE loops until the connection dies, holding an open transaction the
// whole time. Fixed with an explicit depth bound (MAX_WALK_DEPTH,
// store.ts), the same pattern readChain already uses.
//
// Both tests below use a DEDICATED connection with a short
// `statement_timeout` (5s) — not the shared test pool — as a safety net:
// if a future edit ever removes the depth bound again, the query is
// forcibly killed by Postgres itself in 5 seconds instead of hanging this
// suite (or a CI runner) indefinitely. This is belt-and-suspenders, not a
// substitute for the depth bound itself; see docs/lanes/L1-store.md's gate
// remediation section for how this was verified against a bound-removed
// mutation without ever letting a query actually hang.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import pg from 'pg';
import { createTestDb } from './helpers/testdb.js';
import { dropConstraint } from './fixtures/adversarial.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';
import { Store } from '../../src/lib/server/store/store.js';

const { Pool } = pg;

/** Builds a genuine 2-node cycle (a.parent_id = b, b.parent_id = a),
 *  disconnected from any root — constructible only by defeating
 *  message_parent_fk first, exactly like F5's other adversarial fixtures.
 *  Neither node is tombstoned and neither is the conversation's current
 *  leaf, so this exercises PURELY the depth bound, not F1's separate
 *  refusals. */
async function buildTwoCycle(
	db: Awaited<ReturnType<typeof createTestDb>>,
	convId: string,
	userId: string
): Promise<{ aId: string; bId: string }> {
	const aId = uuidv7();
	const bId = uuidv7();

	await assert.rejects(
		() =>
			db.pool.query(
				`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
				 VALUES ($1,$2,$3,$4,'user','"a"'::jsonb)`,
				[aId, convId, bId, userId]
			),
		/message_parent_fk/,
		'the guard must refuse this insert BEFORE the constraint is dropped'
	);
	await dropConstraint(db, 'message', 'message_parent_fk');

	await db.pool.query(
		`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
		 VALUES ($1,$2,$3,$4,'user','"a"'::jsonb)`,
		[aId, convId, bId, userId]
	);
	await db.pool.query(
		`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
		 VALUES ($1,$2,$3,$4,'user','"b"'::jsonb)`,
		[bId, convId, aId, userId]
	);
	return { aId, bId };
}

function guardedStore(schema: string): { store: Store; pool: pg.Pool } {
	const pool = new Pool({
		connectionString: process.env.DATABASE_URL,
		// A dedicated, short statement_timeout — see this file's header.
		options: `-c search_path=${schema} -c statement_timeout=5000`
	});
	return { store: new Store(pool), pool };
}

test('F8: the depth bound does not falsely reject a legitimate 2,000-message chain (the scale bar, 10x under MAX_WALK_DEPTH)', async () => {
	const db = await createTestDb({ prefix: 'f8legitdeep' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });

		// Bulk-build a genuine 2,000-message LINEAR chain — the scale bar
		// FRONTEND_SPEC.md §13 names — via a single multi-row INSERT
		// (mirroring f7-scale.test.ts's own bulkBuildChain), then point
		// current_leaf_id at its tail directly (bypassing appendMessage's
		// CAS, which is not what this test is about).
		const ids: string[] = [root.id];
		const values: string[] = [];
		const params: unknown[] = [];
		let p = 1;
		for (let i = 1; i < 2000; i++) {
			const id = uuidv7();
			values.push(`($${p++}, $${p++}, $${p++}, $${p++}, $${p++}, $${p++}::jsonb)`);
			params.push(id, conversation.id, ids[ids.length - 1], userId, 'user', JSON.stringify(`m-${i}`));
			ids.push(id);
		}
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content) VALUES ${values.join(',')}`,
			params
		);
		const leafId = ids[ids.length - 1];
		await db.pool.query('UPDATE conversation SET current_leaf_id = $1 WHERE id = $2', [leafId, conversation.id]);

		const convNow = (await db.store.getConversation(conversation.id))!;
		const { rev } = await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: leafId,
			expectedRev: convNow.rev
		});
		assert.ok(rev);
	} finally {
		await db.close();
	}
});

test('F8: selectLeaf on a cyclic chain terminates quickly and rejects (UnreachableLeafError) — never hangs', async () => {
	const db = await createTestDb({ prefix: 'f8cycleselect' });
	let guard: ReturnType<typeof guardedStore> | undefined;
	try {
		const userId = uuidv7();
		const { conversation } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { aId } = await buildTwoCycle(db, conversation.id, userId);

		guard = guardedStore(db.schema);
		const start = Date.now();
		await assert.rejects(
			() =>
				guard!.store.selectLeaf({
					convId: conversation.id,
					targetMessageId: aId,
					expectedRev: conversation.rev
				}),
			/UnreachableLeafError/
		);
		const elapsedMs = Date.now() - start;
		assert.ok(
			elapsedMs < 5000,
			`selectLeaf must terminate well within its depth bound's time budget, not merely before the ` +
				`5s statement_timeout safety net — took ${elapsedMs}ms`
		);
	} finally {
		if (guard) await guard.pool.end();
		await db.close();
	}
});

test('F8: tombstoneSubtree on a cyclic chain terminates quickly and tombstones exactly the finite set of real rows it can reach — never hangs', async () => {
	const db = await createTestDb({ prefix: 'f8cycletombstone' });
	let guard: ReturnType<typeof guardedStore> | undefined;
	try {
		const userId = uuidv7();
		const { conversation } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { aId, bId } = await buildTwoCycle(db, conversation.id, userId);

		guard = guardedStore(db.schema);
		const start = Date.now();
		const count = await guard.store.tombstoneSubtree({ convId: conversation.id, rootMessageId: aId });
		const elapsedMs = Date.now() - start;

		// The cyclic walk visits (id, depth) pairs up to the depth bound —
		// alternating a, b, a, b, ... — but only TWO distinct rows exist to
		// update; the UPDATE's own WHERE clause de-duplicates by row.
		assert.equal(count, 2);
		assert.ok(elapsedMs < 5000, `tombstoneSubtree must terminate well within its time budget — took ${elapsedMs}ms`);

		const { rows } = await db.pool.query('SELECT id, deleted_at FROM message WHERE id = ANY($1::uuid[])', [
			[aId, bId]
		]);
		for (const row of rows) {
			assert.ok(row.deleted_at !== null, `${row.id} must be tombstoned`);
		}
	} finally {
		if (guard) await guard.pool.end();
		await db.close();
	}
});
