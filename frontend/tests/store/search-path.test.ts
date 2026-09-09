// Gate remediation F6. db.ts pinned `search_path=${schema},public`.
// Production points this at the `openwebui` database, whose OWN tables
// live in `public` — so if the `client` schema were ever missing or
// half-migrated, `INSERT INTO message` would silently resolve to
// `public.message` instead of failing loudly. Fixed by dropping `,public`.
//
// To actually exercise the risk (not just the absence of a `message` table
// in an empty test database, which would pass even under the OLD buggy
// code), this test plants a DECOY `public.message` table — simulating
// OpenWebUI's own tables sharing the database — then points a pool at a
// schema that was created but never migrated, and proves the query fails
// with "does not exist" rather than silently hitting the decoy.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb, randomSchema } from './helpers/testdb.js';
import { createPool } from '../../src/lib/server/store/db.js';

test('search_path: a query against an unmigrated schema never falls back to a decoy public.message', async () => {
	const db = await createTestDb({ prefix: 'searchpath' });
	const emptySchema = randomSchema('searchpath_empty');
	let scopedPool: ReturnType<typeof createPool> | undefined;
	try {
		// The decoy — stands in for OpenWebUI's own public-schema tables.
		// Its mere existence is the whole point: under the OLD `,public`
		// search_path, `SELECT * FROM message` against the never-migrated
		// schema below would silently SUCCEED (0 rows) against this table
		// instead of raising "does not exist" — a resolved-but-wrong query
		// is exactly as dangerous as one that errors, for the purpose this
		// test exists to catch.
		await db.rawPool.query('DROP TABLE IF EXISTS public.message');
		await db.rawPool.query('CREATE TABLE public.message (id uuid PRIMARY KEY)');

		// A schema that exists but has NEVER been migrated — no `message`
		// table of its own anywhere in it.
		await db.rawPool.query(`CREATE SCHEMA "${emptySchema}"`);

		scopedPool = createPool({ schema: emptySchema });
		await assert.rejects(
			() => scopedPool!.query('SELECT * FROM message'),
			/relation "message" does not exist/,
			'search_path must resolve to the empty client schema ONLY — never fall back to public.message'
		);
	} finally {
		if (scopedPool) await scopedPool.end();
		await db.rawPool.query(`DROP SCHEMA IF EXISTS "${emptySchema}" CASCADE`);
		await db.rawPool.query('DROP TABLE IF EXISTS public.message');
		await db.close();
	}
});
