// TEST-ONLY. Gate remediation F7: dropSchema used to live in migrate.ts and
// be re-exported from index.ts — the store's public API surface — guarded
// only by a doc comment ("test teardown only — never call this against a
// schema you did not create for a single test run"), even though it is a
// bare `DROP SCHEMA ... CASCADE` reachable from anywhere in the app server
// that imports the store package, and is used by exactly one caller:
// frontend/tests/store/helpers/testdb.ts. A doc comment is not enforcement
// (FRONTEND_SPEC.md §11's own standard for this schema), so this function
// now lives in a module index.ts never touches. Import it directly from
// this file's path — never re-add it to index.ts's export list.

import type { Pool } from 'pg';
import { assertValidIdentifier } from './db.js';

/** Drops the schema and everything in it. Test teardown only — never call
 *  this against a schema you did not create for a single test run. */
export async function dropSchema(pool: Pool, schema: string): Promise<void> {
	assertValidIdentifier(schema, 'schema');
	await pool.query(`DROP SCHEMA IF EXISTS "${schema}" CASCADE`);
}
