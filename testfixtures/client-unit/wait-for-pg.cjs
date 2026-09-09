// Bounded readiness probe for the Postgres 16 sidecar (docker-compose.tests
// .yml's postgres-client-unit service). A REAL query, not a bare TCP
// connect: `docker compose run` already honours postgres-client-unit's own
// healthcheck (depends_on: condition: service_healthy) before this
// container even starts, so this is belt-and-suspenders against the one
// case that check cannot see — the operational rule behind this file is
// "never an unbounded wait; a bounded retry loop that exits non-zero when
// exhausted." 30 attempts at 1s apart, then exit 3 (SKIPPED, matching this
// project's own convention for "a fixture this suite depends on is not up"
// — see docker-compose.tests.yml's comment on the tokenizer-contract
// fixture suites for the precedent).
'use strict';
const { Client } = require('pg');

const ATTEMPTS = 30;
const DELAY_MS = 1000;

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

async function tryOnce() {
	const client = new Client({ connectionString: process.env.DATABASE_URL, connectionTimeoutMillis: 2000 });
	try {
		await client.connect();
		await client.query('SELECT 1');
		return true;
	} catch {
		return false;
	} finally {
		try {
			await client.end();
		} catch {
			/* ignore */
		}
	}
}

async function main() {
	if (!process.env.DATABASE_URL) {
		console.error('DATABASE_URL is not set.');
		process.exit(3);
	}
	for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
		// eslint-disable-next-line no-await-in-loop
		if (await tryOnce()) {
			process.exit(0);
		}
		// eslint-disable-next-line no-await-in-loop
		await sleep(DELAY_MS);
	}
	console.error(
		`SKIPPED: Postgres sidecar did not accept a real query after ${ATTEMPTS} attempts (${ATTEMPTS}s). ` +
			'This is treated as SKIP, not FAIL, per this project\'s convention for an unavailable fixture ' +
			'(see docker-compose.tests.yml) — never folded into a pass.'
	);
	process.exit(3);
}

main();
