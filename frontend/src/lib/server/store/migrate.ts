// Applies frontend/migrations/*.sql against a target schema, tracking what
// has already run in a `schema_migrations` table (created if absent) inside
// that same schema.
//
// NOT wired into the SvelteKit server's boot path. Deliberately: this
// subtree has no access to frontend/Dockerfile or supervisord.conf (outside
// L1's ownership — see docs/lanes/L1-store.md), and there is no reliable
// way from inside a Vite/adapter-node bundle to guarantee the raw .sql file
// is even present on disk next to the running server (Vite bundles JS/TS
// imports; it does not, on its own, copy arbitrary files referenced only via
// a runtime fs.readFileSync path). The project's existing pattern for
// exactly this kind of one-time provisioning is entrypoint.sh running plain
// `psql` against the Postgres sidecar before supervisord starts anything —
// see entrypoint.sh's role/database creation for openwebui. This module is
// the equivalent for the client schema, callable either as a small script
// (`node migrate.mjs` after compiling, or ts-node/tsx in dev) or, as here,
// as the function this lane's own tests use to provision a throwaway schema
// per run. Wiring the actual invocation into entrypoint.sh (or an
// equivalent boot step) is a required integration step this lane hands
// back — see the lane doc's "what you could not verify" section.

import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import type { Pool } from 'pg';
import { assertValidIdentifier } from './db.js';

export interface MigrateOptions {
	schema: string;
	migrationsDir: string;
}

export interface MigrateResult {
	applied: string[];
	alreadyApplied: string[];
}

export async function applyMigrations(pool: Pool, opts: MigrateOptions): Promise<MigrateResult> {
	assertValidIdentifier(opts.schema, 'schema');
	const files = readdirSync(opts.migrationsDir)
		.filter((f) => f.endsWith('.sql'))
		.sort();

	const client = await pool.connect();
	const applied: string[] = [];
	const alreadyApplied: string[] = [];
	try {
		await client.query(`CREATE SCHEMA IF NOT EXISTS "${opts.schema}"`);
		await client.query(`SET search_path TO "${opts.schema}"`);
		await client.query(
			`CREATE TABLE IF NOT EXISTS schema_migrations (
				filename   text PRIMARY KEY,
				applied_at timestamptz NOT NULL DEFAULT now()
			)`
		);

		for (const file of files) {
			const { rows } = await client.query(
				'SELECT 1 FROM schema_migrations WHERE filename = $1',
				[file]
			);
			if (rows.length > 0) {
				alreadyApplied.push(file);
				continue;
			}
			const sql = readFileSync(path.join(opts.migrationsDir, file), 'utf8');
			await client.query('BEGIN');
			try {
				// One multi-statement call over the simple query protocol —
				// the same thing `psql -f` does, and what lets a human with
				// psql apply this exact file directly (FRONTEND_SPEC.md
				// §11.5's "a human with a database client must be able to
				// inspect and repair the chain from outside the app").
				await client.query(sql);
				await client.query('INSERT INTO schema_migrations (filename) VALUES ($1)', [file]);
				await client.query('COMMIT');
				applied.push(file);
			} catch (err) {
				await client.query('ROLLBACK');
				throw new Error(`migration ${file} failed: ${(err as Error).message}`, { cause: err });
			}
		}
	} finally {
		client.release();
	}
	return { applied, alreadyApplied };
}

/** Drops the schema and everything in it. Test teardown only — never call
 *  this against a schema you did not create for a single test run. */
export async function dropSchema(pool: Pool, schema: string): Promise<void> {
	assertValidIdentifier(schema, 'schema');
	await pool.query(`DROP SCHEMA IF EXISTS "${schema}" CASCADE`);
}
