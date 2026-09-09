// Typed errors the store's write API can throw. Deliberately few: most
// rejections (second root, unknown parent, cross-conversation parent,
// self-parent, a change to id/conv_id/parent_id, a live child of a
// tombstoned parent) are NOT represented here — they are database
// constraint violations and are allowed to propagate as the raw `pg` error
// (with its `.code` — the Postgres SQLSTATE — intact). FRONTEND_HANDOFF.md
// §6 / this lane's brief: F4's rejection suite asserts the CONSTRAINTS
// reject bad writes, not a guard clause in this file — so this file must
// not grow a parallel validator that would let those tests pass even if a
// constraint were dropped from the migration.
//
// What IS legitimately application logic, not something Postgres's
// declarative constraint system can express on its own: the leaf-pointer
// compare-and-swap (StaleWriteError), the tombstone-aware reachability walk
// for explicit branch selection (UnreachableLeafError), and
// tombstone_subtree's own "never the current leaf, never the root"
// refusal (TombstoneRefusedError). All three are "run inside the
// transaction" by design (FRONTEND_SPEC.md §11.3), not database
// constraints — see store.ts.
//
// The one exception: IllegalStateTransitionError. The monotonic message
// state machine (pending -> streaming -> complete|failed, terminal states
// frozen) IS enforced by a database trigger (message_enforce_state_machine,
// migrations/0001_init.sql) — a pure OLD-vs-NEW check, which is exactly
// what a trigger is for. But the gate remediation brief (F5) is explicit
// that this one violation must surface to the caller as a typed error, not
// a raw driver exception with an opaque SQLSTATE — so store.ts recognises
// the trigger's distinguishing SQLSTATE (see STATE_MACHINE_SQLSTATE) and
// wraps it here. This is the one place this file's own "do not wrap
// constraint violations" rule is deliberately broken, and it is broken by
// explicit instruction, not by drift.

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
 *  that does not exist, or tombstone_subtree targeted a (conv_id, id) pair
 *  that does not exist. */
export class NotFoundError extends StoreError {}

/** tombstone_subtree refused: either the target is the conversation's
 *  synthetic root (tombstoning it would hide the entire conversation, and
 *  the audit would still report `pass = true` against an empty-looking
 *  thread), or the conversation's current `current_leaf_id` lies inside the
 *  subtree being tombstoned. Both keep "the current leaf is never
 *  tombstoned" true by construction — the caller must call select_leaf to
 *  move the pointer elsewhere first. */
export class TombstoneRefusedError extends StoreError {}

/** update_message_state attempted an illegal message state transition —
 *  backward (e.g. streaming -> pending), sideways between the two terminal
 *  states (complete <-> failed), or any change to state or content once a
 *  message is already complete or failed. Reaching a terminal state is
 *  one-way (FRONTEND_SPEC.md §11.3-adjacent gate remediation F5). Raised by
 *  the message_enforce_state_machine trigger; wrapped here so the caller
 *  sees a typed error rather than a raw pg exception. */
export class IllegalStateTransitionError extends StoreError {}
