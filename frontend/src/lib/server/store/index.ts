export { Store } from './store.js';
export { createPool, resolveSchema, assertValidIdentifier } from './db.js';
export type { StoreConfig } from './db.js';
// Gate remediation F7: dropSchema (DROP SCHEMA ... CASCADE) is NOT
// exported here. It used to be, guarded only by a doc comment on its own
// definition — reachable from anywhere in the app server despite being
// used only by frontend/tests/store/helpers/testdb.ts. It still lives in
// ./testSupport.js, which nothing under src/routes or the compactor
// transport imports; test helpers import it directly from that module
// path, bypassing this public surface entirely. See testSupport.ts's own
// header comment.
export { applyMigrations } from './migrate.js';
export type { MigrateOptions, MigrateResult } from './migrate.js';
export { uuidv7, isUuidv7 } from './uuid7.js';
export {
	StoreError,
	StaleWriteError,
	UnreachableLeafError,
	NotFoundError,
	TombstoneRefusedError,
	IllegalStateTransitionError
} from './errors.js';
export type { Role, MessageState, Message, Conversation, AuditResult, ReadPage } from './types.js';
