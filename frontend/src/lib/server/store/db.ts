// Connection pool + type-parser setup for the L1 store.
//
// Placement (FRONTEND_PLAN.md §3.1): production points this at the existing
// `openwebui` Postgres database, in a `client` schema — NOT a new database.
// A separate database would live only on the pod's ephemeral overlay and be
// silently excluded from pgarchive.py's dumps (scoped to one database name),
// which is exactly the "OpenWebUI blob" durability failure inverted. This
// module never hardcodes "client" as a schema name in a query string —
// every SQL statement in store.ts/migrate.ts uses bare, unqualified table
// and function names and relies on `search_path`, set once per connection
// via the pool's `options` startup parameter. That is what lets tests point
// the exact same code at a fresh, throwaway schema per run (see
// tests/store/helpers/testdb.ts) instead of a second hardcoded name.
//
// Configuration is plain `process.env`, not SvelteKit's `$env/*` — this
// subtree must import cleanly under a standalone `node --test` run (no Vite,
// no SvelteKit runtime) as well as under the built adapter-node server, and
// `process.env` is the one configuration surface both environments share.

import pg from 'pg';

const { Pool, types } = pg;

// jsonb (OID 3802) — see types.ts: content/error are carried as raw JSON
// text end to end, never auto-parsed into a JS value by node-postgres's
// default jsonb type parser. This is a PROCESS-WIDE override (the `pg`
// module's type parser table is a shared, mutable global) — safe here
// because `content`/`error` are the store's only jsonb columns and nothing
// else in this process is expected to register a conflicting parser for
// OID 3802. Anyone adding a new jsonb column to this schema later inherits
// this behaviour; anyone adding an UNRELATED jsonb use elsewhere in the app
// sharing this process also inherits it — flagged here so that surprise is
// at least documented once, in the one place it is set.
types.setTypeParser(types.builtins.JSONB, (value: string) => value);

// int8 columns (bigint: conversation.rev, and audit_conversation's
// count(*)-derived columns) — parsed as a JS number is fine for `rev`
// (a per-conversation write counter, unrealistic to approach 2^53) and for
// audit's counts (bounded by message count). types.ts still carries `rev`
// as a string end to end anyway, so this override only affects the
// audit_conversation() result path and any future ad hoc bigint read.
types.setTypeParser(types.builtins.INT8, (value: string) => Number(value));

const IDENTIFIER_RE = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

export function assertValidIdentifier(name: string, what: string): void {
	if (!IDENTIFIER_RE.test(name)) {
		throw new Error(`${what} is not a safe SQL identifier: ${JSON.stringify(name)}`);
	}
}

export interface StoreConfig {
	/** Defaults to process.env.DATABASE_URL. */
	connectionString?: string;
	/** Defaults to process.env.CLIENT_STORE_SCHEMA, then 'client'. */
	schema?: string;
	/** Pool sizing knobs, rarely needed outside tests. */
	max?: number;
}

export function resolveSchema(config: StoreConfig = {}): string {
	const schema = config.schema ?? process.env.CLIENT_STORE_SCHEMA ?? 'client';
	assertValidIdentifier(schema, 'schema');
	return schema;
}

/** One Pool is bound to exactly one schema for its whole lifetime (the
 *  schema is pinned into every connection's startup `options`, via
 *  `search_path`) — never repoint a live Pool at a different schema. Tests
 *  that need N isolated schemas create N pools. */
export function createPool(config: StoreConfig = {}): pg.Pool {
	const connectionString = config.connectionString ?? process.env.DATABASE_URL;
	if (!connectionString) {
		throw new Error(
			'DATABASE_URL is not set and no connectionString was provided to createPool()'
		);
	}
	const schema = resolveSchema(config);
	return new Pool({
		connectionString,
		max: config.max,
		// libpq startup-packet options; `-c search_path=...` is the
		// standard way to pin search_path per-connection without every
		// checkout having to issue its own `SET search_path` round trip.
		options: `-c search_path=${schema},public`
	});
}
