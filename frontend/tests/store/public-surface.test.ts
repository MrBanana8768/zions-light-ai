// Gate remediation F7. dropSchema (a bare DROP SCHEMA ... CASCADE) used to
// be exported from index.ts — the store's public API surface — guarded
// only by a doc comment, despite being reachable from anywhere in the app
// server and used only by tests/store/helpers/testdb.ts. Moved to
// testSupport.ts, which index.ts never re-exports. This test asserts the
// negative directly at runtime (not merely "the code we wrote doesn't call
// it") so a future edit that re-adds the export is caught mechanically.
//
// Deliberately uses `import * as` and a runtime property check rather than
// `import { dropSchema } from '...'` — a named import of something that
// does not exist would be a COMPILE error (tsc), which would fail the
// entire suite's build step rather than this one test, and would defeat
// the mutation-testing methodology (the whole file would refuse to build,
// not go red on this one assertion).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as StoreIndex from '../../src/lib/server/store/index.js';

test('public surface: dropSchema is NOT exported from the store\'s public index', async () => {
	assert.equal(
		(StoreIndex as Record<string, unknown>).dropSchema,
		undefined,
		'dropSchema is DROP SCHEMA ... CASCADE — test-only, imported directly from testSupport.js by ' +
			'tests/store/helpers/testdb.ts, and must never be reachable from index.ts'
	);
});

test('public surface: the typed error classes ARE exported (sanity — the index still exports what application code needs)', async () => {
	assert.equal(typeof StoreIndex.StoreError, 'function');
	assert.equal(typeof StoreIndex.StaleWriteError, 'function');
	assert.equal(typeof StoreIndex.UnreachableLeafError, 'function');
	assert.equal(typeof StoreIndex.NotFoundError, 'function');
	assert.equal(typeof StoreIndex.TombstoneRefusedError, 'function');
	assert.equal(typeof StoreIndex.IllegalStateTransitionError, 'function');
});
