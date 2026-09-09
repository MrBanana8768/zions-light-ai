// Per-test isolation: every test (or test file) that needs one gets its own
// freshly-migrated, randomly-named Postgres SCHEMA inside the same target
// database, torn down (DROP SCHEMA ... CASCADE) when the test is done.
//
// Why a schema, not the shared `openwebui`/`client` production names, and
// not a fresh DATABASE per test: creating a schema is fast (no new catalog
// entry, no new set of system files) compared to CREATE DATABASE, and it
// gives adversarial tests (F5) a place to deliberately break a constraint —
// e.g. DROP the message_one_root_per_conv index to build the standing
// 241/5-root/leaf@8 fixture, or DISABLE TRIGGER to build a "missing parent"
// fixture — without that mutation ever being visible to, or torn down by,
// any other test. See fixtures/adversarial.ts for those.
//
// DATABASE_URL must already point at a reachable Postgres 16 (the
// docker-compose.tests.yml `client-unit` service starts one; see that file
// for how). Nothing here ever assumes `client_test`/`client` specifically —
// the schema name is generated per call.

import { randomBytes } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type pg from 'pg';
import { createPool } from '../../../src/lib/server/store/db.js';
import { applyMigrations, dropSchema } from '../../../src/lib/server/store/migrate.js';
import { Store } from '../../../src/lib/server/store/store.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Compiled layout mirrors the source tree one level DEEPER than source,
// under dist-store-tests/ (tsconfig.store-tests.json's outDir with
// rootDir '.'): this file compiles to
// dist-store-tests/tests/store/helpers/testdb.js, so it takes one more
// ".." than the source path (frontend/tests/store/helpers/testdb.ts ->
// frontend/migrations) would suggest, to clear the dist-store-tests/
// directory itself and land back at frontend/migrations either way.
export const MIGRATIONS_DIR = path.join(__dirname, '..', '..', '..', '..', 'migrations');

export function randomSchema(prefix = 'test'): string {
	return `${prefix}_${randomBytes(6).toString('hex')}`;
}

export interface TestDb {
	schema: string;
	pool: pg.Pool;
	store: Store;
	migrationsDir: string;
	/** A second pool with NO search_path pinned — for adversarial fixtures
	 *  that need to run schema-qualified DDL (DROP INDEX, DISABLE TRIGGER)
	 *  against this exact schema by name, which the store's own pool
	 *  deliberately never does (see db.ts's ban on hardcoding the schema
	 *  name into application SQL). Fixture code only. */
	rawPool: pg.Pool;
	close(): Promise<void>;
}

export async function createTestDb(opts: { prefix?: string } = {}): Promise<TestDb> {
	if (!process.env.DATABASE_URL) {
		throw new Error('DATABASE_URL must be set to run store tests');
	}
	const schema = randomSchema(opts.prefix);
	const pool = createPool({ schema });
	const rawPool = createPool({ schema: 'public' }); // pinned to public; fixtures qualify explicitly
	await applyMigrations(pool, { schema, migrationsDir: MIGRATIONS_DIR });
	const store = new Store(pool);
	return {
		schema,
		pool,
		store,
		migrationsDir: MIGRATIONS_DIR,
		rawPool,
		async close() {
			await dropSchema(rawPool, schema);
			await pool.end();
			await rawPool.end();
		}
	};
}
