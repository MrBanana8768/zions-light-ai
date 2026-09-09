// Typed errors the store's write API can throw. Deliberately few: most
// rejections (second root, unknown parent, cross-conversation parent,
// self-parent, a change to id/conv_id/parent_id) are NOT represented here —
// they are database constraint violations and are allowed to propagate as
// the raw `pg` error (with its `.code` — the Postgres SQLSTATE — intact).
// FRONTEND_HANDOFF.md §6 / this lane's brief: F4's rejection suite asserts
// the CONSTRAINTS reject bad writes, not a guard clause in this file — so
// this file must not grow a parallel validator that would let those tests
// pass even if a constraint were dropped from the migration.
//
// What IS legitimately application logic, not something Postgres's
// declarative constraint system can express on its own: the leaf-pointer
// compare-and-swap (StaleWriteError) and the tombstone-aware reachability
// walk for explicit branch selection (UnreachableLeafError). Both are
// "run inside the transaction" by design (FRONTEND_SPEC.md §11.3), not
// database constraints — see store.ts.

export class StoreError extends Error {
	constructor(message: string) {
		super(message);
		this.name = new.target.name;
	}
}

/** The append/select_leaf compare-and-swap did not match (rev or
 *  current_leaf_id changed since the caller last read the conversation).
 *  The caller lost its own write; it must refetch and retry, never overwrite. */
export class StaleWriteError extends StoreError {}

/** select_leaf's reachability predicate failed: the target message does not
 *  exist in this conversation, or it (or an ancestor of it) is tombstoned. */
export class UnreachableLeafError extends StoreError {}

/** update_message_state / append_stream_delta targeted a (conv_id, id) pair
 *  that does not exist. */
export class NotFoundError extends StoreError {}
