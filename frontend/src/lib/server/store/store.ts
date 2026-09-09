// The L1 write/read API. FRONTEND_SPEC.md §11.1: "The storage API must
// expose no operation that accepts a whole chain." Every write below is a
// per-message delta. There is no PUT-a-chain endpoint anywhere in this file
// and there must never be one added to it.
//
// content/error are carried as RAW JSON TEXT (`string`), never as a parsed
// JS value, in both directions — see types.ts and db.ts's jsonb type-parser
// override. This is the store's half of FRONTEND_PLAN.md §3.2's byte-
// stability obligation: a JS JSON.parse -> re-JSON.stringify round trip can
// reorder object keys or reformat numbers even when the parsed VALUE is
// unchanged, and that is exactly the kind of transformation §3.2 says
// silently drifts the compactor's tail_fp anchor. This module never parses
// content; it hands Postgres the exact text it was given and hands back the
// exact text Postgres has stored (Postgres's own jsonb round trip is
// deterministic — see docs/lanes/L1-store.md's "byte stability" section for
// the measured finding on whether it also preserves the ORIGINAL input
// bytes, and why that distinction does not matter for the property this
// store must guarantee).

import type { Pool, PoolClient } from 'pg';
import { uuidv7 } from './uuid7.js';
import {
	StaleWriteError,
	UnreachableLeafError,
	NotFoundError,
	TombstoneRefusedError,
	IllegalStateTransitionError
} from './errors.js';
import type { Conversation, Message, AuditResult, Role, MessageState, ReadPage } from './types.js';

// Gate remediation F8: an explicit depth bound for the two recursive walks
// in this file that move against the parent-to-child direction from an
// arbitrary starting row (selectLeaf's `up` walk; tombstoneSubtree's
// `subtree` walk) rather than from a proven root (audit_conversation's
// `down` walk — see that function's own comment in the migration SQL for
// why it needs no bound). Neither walk has any structural immunity to a
// cycle: §11.5 grants a human repairing the chain from outside the app
// exactly the freedom to construct one, and an unbounded `UNION ALL`
// recursive CTE against cyclic data loops until the connection dies,
// holding an open transaction the whole time. This mirrors the bound
// readChain already applies (`chain.depth + 1 < $3`, using the caller's
// own page-size limit) — these two walks have no natural caller-supplied
// limit, so the bound is a constant instead, set far above any realistic
// conversation depth (the 2,000-message scale bar) so it can only ever
// fire on a genuine cycle, never on legitimate depth.
const MAX_WALK_DEPTH = 20_000;

// Gate remediation F5: the custom SQLSTATE the migration's
// message_enforce_state_machine trigger raises with. Matched against the
// `pg` driver's `.code` so updateMessageState can wrap that ONE violation
// as a typed IllegalStateTransitionError — every other constraint/trigger
// in this schema is deliberately left unwrapped (see errors.ts).
const STATE_MACHINE_SQLSTATE = 'ZL0ST';

function isPgErrorWithCode(err: unknown, code: string): err is { code: string; message: string } {
	return (
		typeof err === 'object' &&
		err !== null &&
		'code' in err &&
		(err as { code?: unknown }).code === code
	);
}

function rowToMessage(row: Record<string, unknown>): Message {
	return {
		id: row.id as string,
		convId: row.conv_id as string,
		parentId: (row.parent_id as string | null) ?? null,
		userId: row.user_id as string,
		role: row.role as Role,
		content: row.content as string,
		state: row.state as MessageState,
		error: (row.error as string | null) ?? null,
		createdAt: toIso(row.created_at),
		updatedAt: toIso(row.updated_at),
		deletedAt: row.deleted_at ? toIso(row.deleted_at) : null
	};
}

function rowToConversation(row: Record<string, unknown>): Conversation {
	return {
		id: row.id as string,
		userId: row.user_id as string,
		title: (row.title as string | null) ?? null,
		currentLeafId: (row.current_leaf_id as string | null) ?? null,
		rev: String(row.rev),
		createdAt: toIso(row.created_at),
		updatedAt: toIso(row.updated_at)
	};
}

function toIso(value: unknown): string {
	return value instanceof Date ? value.toISOString() : String(value);
}

async function rollbackQuietly(client: PoolClient): Promise<void> {
	try {
		await client.query('ROLLBACK');
	} catch {
		// the connection may already be unusable (e.g. the transaction was
		// aborted by a trigger-raised exception) — nothing more to do.
	}
}

export class Store {
	constructor(private readonly pool: Pool) {}

	// ---- writes --------------------------------------------------------

	/** Creates a conversation and its single synthetic `role='system'` root
	 *  in one transaction (D5). The first user turn is the root's child —
	 *  editing "the first message" therefore produces a sibling under this
	 *  root, never a second root, which is what makes `roots == 1`
	 *  enforceable by the unique index without losing that behaviour. */
	async createConversation(params: {
		userId: string;
		systemContent: string;
		title?: string | null;
	}): Promise<{ conversation: Conversation; root: Message }> {
		const convId = uuidv7();
		const rootId = uuidv7();
		const client = await this.pool.connect();
		try {
			await client.query('BEGIN');
			const convRows = await client.query(
				`INSERT INTO conversation (id, user_id, title, current_leaf_id, rev)
				 VALUES ($1, $2, $3, $4, 0)
				 RETURNING *`,
				[convId, params.userId, params.title ?? null, rootId]
			);
			const msgRows = await client.query(
				`INSERT INTO message (id, conv_id, parent_id, user_id, role, content, state)
				 VALUES ($1, $2, NULL, $3, 'system', $4::jsonb, 'complete')
				 RETURNING *`,
				[rootId, convId, params.userId, params.systemContent]
			);
			await client.query('COMMIT');
			return {
				conversation: rowToConversation(convRows.rows[0]),
				root: rowToMessage(msgRows.rows[0])
			};
		} catch (err) {
			await rollbackQuietly(client);
			throw err;
		} finally {
			client.release();
		}
	}

	/** Append is a compare-and-swap: the new message's parent must equal the
	 *  conversation's CURRENT current_leaf_id, and expectedRev must match
	 *  conversation.rev, both checked in the same transaction as the insert
	 *  (FRONTEND_SPEC.md §11.3). A branch variant (regeneration/edit) is
	 *  produced by first calling selectLeaf() to move the pointer back to
	 *  the shared parent, then appendMessage() from there — append itself
	 *  never accepts an out-of-turn parent; that is the whole of what makes
	 *  "an append can never orphan history" true. Throws StaleWriteError on
	 *  a CAS miss; throws the raw `pg` error (unmodified) on a constraint
	 *  violation — unknown parent, cross-conversation parent, etc. are
	 *  rejected by the database, not by a check in this function. */
	async appendMessage(params: {
		convId: string;
		parentId: string;
		expectedRev: string | number | bigint;
		userId: string;
		role: Role;
		content: string;
		state?: MessageState;
	}): Promise<{ message: Message; rev: string }> {
		const id = uuidv7();
		const client = await this.pool.connect();
		try {
			await client.query('BEGIN');
			const insertRes = await client.query(
				`INSERT INTO message (id, conv_id, parent_id, user_id, role, content, state)
				 VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
				 RETURNING *`,
				[
					id,
					params.convId,
					params.parentId,
					params.userId,
					params.role,
					params.content,
					params.state ?? 'complete'
				]
			);
			const casRes = await client.query(
				`UPDATE conversation
				 SET current_leaf_id = $1, rev = rev + 1, updated_at = now()
				 WHERE id = $2 AND rev = $3 AND current_leaf_id = $4
				 RETURNING rev`,
				[id, params.convId, String(params.expectedRev), params.parentId]
			);
			if (casRes.rowCount !== 1) {
				await rollbackQuietly(client);
				throw new StaleWriteError(
					`append_message CAS failed: conv=${params.convId} expectedRev=${params.expectedRev} expectedParent=${params.parentId} (rev or current_leaf_id changed since the caller last read the conversation)`
				);
			}
			await client.query('COMMIT');
			return { message: rowToMessage(insertRes.rows[0]), rev: String(casRes.rows[0].rev) };
		} catch (err) {
			if (!(err instanceof StaleWriteError)) {
				await rollbackQuietly(client);
			}
			throw err;
		} finally {
			client.release();
		}
	}

	/** Updates a message's state/content/error in place. Never touches
	 *  id/conv_id/parent_id (the BEFORE UPDATE trigger would reject it
	 *  anyway) and never moves the leaf pointer. Used both for "on failure,
	 *  the user turn is retained in place, marked failed" (§4.1) and as the
	 *  building block append_stream_delta below is a thin, named wrapper
	 *  around.
	 *
	 *  Gate remediation F5: the transition itself (pending -> streaming ->
	 *  complete|failed, terminal states frozen) is enforced by the
	 *  migration's message_enforce_state_machine trigger, not by this
	 *  function — this function's own job is only to recognise that
	 *  trigger's violation (by its distinguishing SQLSTATE) and surface it
	 *  as a typed IllegalStateTransitionError instead of a raw driver
	 *  exception. A lagging second writer that tries to flip an already-
	 *  `complete` message back to `streaming` (or change its content once
	 *  terminal) loses this call loudly, with a typed error, exactly as
	 *  §11.3's "it loses its own write, loudly" already requires for the
	 *  leaf pointer. */
	async updateMessageState(params: {
		convId: string;
		messageId: string;
		state: MessageState;
		content?: string;
		error?: string | null;
	}): Promise<Message> {
		const sets = ['state = $3', 'updated_at = now()'];
		const values: unknown[] = [params.convId, params.messageId, params.state];
		if (params.content !== undefined) {
			values.push(params.content);
			sets.push(`content = $${values.length}::jsonb`);
		}
		if (params.error !== undefined) {
			values.push(params.error);
			sets.push(`error = $${values.length}::jsonb`);
		}
		try {
			const { rows, rowCount } = await this.pool.query(
				`UPDATE message SET ${sets.join(', ')} WHERE conv_id = $1 AND id = $2 RETURNING *`,
				values
			);
			if (rowCount !== 1) {
				throw new NotFoundError(`message ${params.messageId} not found in conversation ${params.convId}`);
			}
			return rowToMessage(rows[0]);
		} catch (err) {
			if (isPgErrorWithCode(err, STATE_MACHINE_SQLSTATE)) {
				throw new IllegalStateTransitionError(err.message);
			}
			throw err;
		}
	}

	/** Decision 6 (FRONTEND_PLAN.md §3.1): the row is inserted ONCE at
	 *  stream start via appendMessage(state: 'pending'|'streaming'). This
	 *  function is the checkpoint write — called by the caller on a ~500ms
	 *  interval with the ACCUMULATED text so far, and once more at
	 *  completion with `final: true`. It does not insert a row and it does
	 *  not move the leaf; it is a single targeted UPDATE. Throttling to
	 *  ~500ms is the CALLER's responsibility (the transport lane) — this
	 *  store only guarantees the write itself is O(1) in conversation
	 *  length and measures its own per-call cost (see F7 / the streaming-
	 *  checkpoint measurement in docs/lanes/L1-store.md). Calling this once
	 *  per token instead of on an interval is exactly the O(reply²)
	 *  footgun the plan warns about — nothing in this function prevents a
	 *  caller from doing that; the granularity contract is enforced by
	 *  convention at the call site, not by this API. */
	async appendStreamDelta(params: {
		convId: string;
		messageId: string;
		content: string;
		final: boolean;
	}): Promise<Message> {
		return this.updateMessageState({
			convId: params.convId,
			messageId: params.messageId,
			state: params.final ? 'complete' : 'streaming',
			content: params.content
		});
	}

	/** Explicit branch selection. Validates a tombstone-aware reachability
	 *  predicate (walk parent_id from target to root; reject if the target
	 *  does not exist, cannot reach a root, or any node on that path is
	 *  tombstoned) and a rev CAS, both inside one transaction, matching
	 *  FRONTEND_SPEC.md §11.3 exactly. This is the ONLY other path (besides
	 *  appendMessage's CAS) that may move current_leaf_id.
	 *
	 *  Gate remediation F4 (TOCTOU): the reachability predicate below is a
	 *  plain SELECT under READ COMMITTED, and tombstone_subtree bumps no
	 *  `rev` and (before this fix) took no lock on `conversation` — so a
	 *  predicate that ran clean, followed by a CAS that only re-checks
	 *  `rev`, could still land the leaf inside a subtree a concurrent
	 *  tombstone_subtree hid in the gap between the two. `SELECT ... FOR
	 *  UPDATE` on the conversation row, taken FIRST — before the predicate
	 *  even runs — closes that gap: tombstone_subtree takes the identical
	 *  lock as its own first statement (see below), so whichever of the
	 *  two gets there first forces the other to wait, and the one that
	 *  waits re-evaluates against the FRESH, post-commit state once it
	 *  proceeds. Neither call can now act on a predicate computed against
	 *  data the other has since changed. */
	async selectLeaf(params: {
		convId: string;
		targetMessageId: string;
		expectedRev: string | number | bigint;
	}): Promise<{ rev: string }> {
		const client = await this.pool.connect();
		try {
			await client.query('BEGIN');
			const lockRes = await client.query(
				'SELECT id FROM conversation WHERE id = $1 FOR UPDATE',
				[params.convId]
			);
			if (lockRes.rowCount !== 1) {
				await rollbackQuietly(client);
				throw new NotFoundError(`conversation ${params.convId} not found`);
			}
			const { rows } = await client.query(
				`WITH RECURSIVE up(id, parent_id, deleted_at, depth) AS (
					SELECT id, parent_id, deleted_at, 1 FROM message WHERE conv_id = $1 AND id = $2
					UNION ALL
					SELECT m.id, m.parent_id, m.deleted_at, up.depth + 1
					FROM message m JOIN up ON m.id = up.parent_id
					WHERE m.conv_id = $1 AND up.depth < $3
				 )
				 SELECT count(*) AS visited,
				        count(*) FILTER (WHERE deleted_at IS NOT NULL) AS tombstoned,
				        bool_or(parent_id IS NULL) AS reached_root
				 FROM up`,
				[params.convId, params.targetMessageId, MAX_WALK_DEPTH]
			);
			const visited = Number(rows[0].visited);
			const tombstoned = Number(rows[0].tombstoned);
			const reachedRoot = rows[0].reached_root === true;
			if (visited === 0 || !reachedRoot || tombstoned > 0) {
				await rollbackQuietly(client);
				throw new UnreachableLeafError(
					`select_leaf rejected: conv=${params.convId} target=${params.targetMessageId} visited=${visited} reachedRoot=${reachedRoot} tombstoned=${tombstoned}`
				);
			}
			const casRes = await client.query(
				`UPDATE conversation
				 SET current_leaf_id = $1, rev = rev + 1, updated_at = now()
				 WHERE id = $2 AND rev = $3
				 RETURNING rev`,
				[params.targetMessageId, params.convId, String(params.expectedRev)]
			);
			if (casRes.rowCount !== 1) {
				await rollbackQuietly(client);
				throw new StaleWriteError(
					`select_leaf CAS failed: conv=${params.convId} expectedRev=${params.expectedRev} (rev changed since the caller last read the conversation)`
				);
			}
			await client.query('COMMIT');
			return { rev: String(casRes.rows[0].rev) };
		} catch (err) {
			if (
				!(err instanceof StaleWriteError) &&
				!(err instanceof UnreachableLeafError) &&
				!(err instanceof NotFoundError)
			) {
				await rollbackQuietly(client);
			}
			throw err;
		} finally {
			client.release();
		}
	}

	/** Subtree-wide tombstone. Sets deleted_at on the target AND every
	 *  descendant in one statement (a bounded recursive walk down from the
	 *  target), so hiding a message can never leave its children
	 *  reachable-but-orphaned-looking or, worse, visibly dangling in a
	 *  render while their parent is gone from view (FRONTEND_SPEC.md
	 *  §11.1).
	 *
	 *  Gate remediation F1: refuses outright — before touching any row —
	 *  in two cases, both via a typed TombstoneRefusedError:
	 *    1. The target IS the conversation's synthetic root. Tombstoning it
	 *       would hide the entire conversation while audit_conversation
	 *       still reports `pass = true` (roots/reachable_n/leaf_on_tree are
	 *       all about STRUCTURE, not visibility — a fully-tombstoned tree
	 *       is still structurally intact).
	 *    2. The conversation's current `current_leaf_id` lies inside the
	 *       subtree about to be tombstoned. The caller must select_leaf()
	 *       to a different leaf FIRST. This is what makes "the current
	 *       leaf is never tombstoned" true BY CONSTRUCTION, rather than a
	 *       property the caller was merely trusted to preserve — an
	 *       earlier version of this comment conceded the old behaviour was
	 *       wrong "in spirit" (the leaf could end up pointing at hidden
	 *       content); that state is no longer reachable, so there is
	 *       nothing left to concede.
	 *
	 *  Gate remediation F4: takes `SELECT ... FOR UPDATE` on the
	 *  conversation row as its FIRST statement — the identical lock
	 *  select_leaf takes — so the two serialize. Without this, a
	 *  concurrent select_leaf could move current_leaf_id into this exact
	 *  subtree in the gap between this function reading current_leaf_id
	 *  and applying the UPDATE below, and this function would tombstone
	 *  the node the conversation was, by the time of that UPDATE,
	 *  genuinely pointing at. See select_leaf's own comment for the other
	 *  half of this.
	 *
	 *  Gate remediation F8: the downward walk carries an explicit depth
	 *  bound (MAX_WALK_DEPTH), matching select_leaf's `up` walk and
	 *  readChain's existing pattern — see MAX_WALK_DEPTH's own comment. */
	async tombstoneSubtree(params: { convId: string; rootMessageId: string }): Promise<number> {
		const client = await this.pool.connect();
		try {
			await client.query('BEGIN');
			const lockRes = await client.query(
				'SELECT current_leaf_id FROM conversation WHERE id = $1 FOR UPDATE',
				[params.convId]
			);
			if (lockRes.rowCount !== 1) {
				await rollbackQuietly(client);
				throw new NotFoundError(`conversation ${params.convId} not found`);
			}
			const currentLeafId = (lockRes.rows[0].current_leaf_id as string | null) ?? null;

			const targetRes = await client.query(
				'SELECT parent_id FROM message WHERE conv_id = $1 AND id = $2',
				[params.convId, params.rootMessageId]
			);
			if (targetRes.rowCount !== 1) {
				await rollbackQuietly(client);
				throw new NotFoundError(
					`message ${params.rootMessageId} not found in conversation ${params.convId}`
				);
			}
			if (targetRes.rows[0].parent_id === null) {
				await rollbackQuietly(client);
				throw new TombstoneRefusedError(
					`refusing to tombstone ${params.rootMessageId}: it is the synthetic root of conversation ${params.convId} — tombstoning it would hide the entire conversation`
				);
			}

			const subtreeRes = await client.query(
				`WITH RECURSIVE subtree(id, depth) AS (
					SELECT id, 1 FROM message WHERE conv_id = $1 AND id = $2
					UNION ALL
					SELECT m.id, subtree.depth + 1
					FROM message m JOIN subtree ON m.parent_id = subtree.id
					WHERE m.conv_id = $1 AND subtree.depth < $3
				 )
				 SELECT id FROM subtree`,
				[params.convId, params.rootMessageId, MAX_WALK_DEPTH]
			);
			const subtreeIds: string[] = subtreeRes.rows.map((r) => r.id as string);

			if (currentLeafId !== null && subtreeIds.includes(currentLeafId)) {
				await rollbackQuietly(client);
				throw new TombstoneRefusedError(
					`refusing to tombstone ${params.rootMessageId}: the conversation's current_leaf_id (${currentLeafId}) is inside this subtree — select_leaf to a different leaf first`
				);
			}

			const updateRes = await client.query(
				`UPDATE message SET deleted_at = now(), updated_at = now()
				 WHERE conv_id = $1 AND id = ANY($2::uuid[]) AND deleted_at IS NULL`,
				[params.convId, subtreeIds]
			);
			await client.query('COMMIT');
			return updateRes.rowCount ?? 0;
		} catch (err) {
			if (!(err instanceof TombstoneRefusedError) && !(err instanceof NotFoundError)) {
				await rollbackQuietly(client);
			}
			throw err;
		} finally {
			client.release();
		}
	}

	// ---- reads -----------------------------------------------------------

	async getConversation(convId: string): Promise<Conversation | null> {
		const { rows } = await this.pool.query('SELECT * FROM conversation WHERE id = $1', [convId]);
		return rows.length ? rowToConversation(rows[0]) : null;
	}

	/** Gate remediation D8 (docs/lanes/L2-gate-findings.md): a one-row,
	 *  indexed lookup of the conversation's synthetic persona root —
	 *  `message_one_root_per_conv`'s own partial unique index on
	 *  `(conv_id) WHERE parent_id IS NULL` answers this directly, with no
	 *  walk. Replaces the O(n) `readOlder(convId, cursor, ROOT_LOOKUP_LIMIT)`
	 *  pattern the transport lane used to fall back to for a long
	 *  conversation's root: that call's own recursive CTE (readChain) returns
	 *  every row it visits — content included — so it cost ~1,940 full rows
	 *  per outbound message on a 2,000-message conversation, purely to read
	 *  one row's `id`. This method returns `null` if the conversation has no
	 *  root at all (should never happen for a real conversation, but this is
	 *  a plain read — it has no opinion about that, unlike auditConversation,
	 *  which is the place that DOES). Returns whichever row Postgres finds
	 *  first if, adversarially, more than one exists (message_one_root_per_conv
	 *  dropped) — callers that need to distinguish that from a healthy
	 *  single-root conversation already have auditConversation's own
	 *  `roots` count for that; this method's contract is "the root," not
	 *  "assert there is exactly one." */
	async getRoot(convId: string): Promise<Message | null> {
		const { rows } = await this.pool.query(
			'SELECT * FROM message WHERE conv_id = $1 AND parent_id IS NULL LIMIT 1',
			[convId]
		);
		return rows.length ? rowToMessage(rows[0]) : null;
	}

	/** The newest `limit` messages of the ACTIVE path (walking parent_id
	 *  from current_leaf_id toward the root), newest first. Never reads more
	 *  than `limit` rows regardless of total conversation length — see
	 *  docs/lanes/L1-store.md's "read contract" for why this satisfies
	 *  FRONTEND_SPEC.md §13's "never need the whole conversation in memory
	 *  to show the newest turn." */
	async readTail(convId: string, limit: number): Promise<ReadPage> {
		return this.readChain(convId, null, limit);
	}

	/** Continues an earlier page. `beforeId` is the oldest message id
	 *  already delivered (from a prior readTail/readOlder's last entry);
	 *  the next page starts at ITS parent. This is the store's keyset: the
	 *  "key" is a message id, resolved purely by walking immutable
	 *  parent_id pointers, so it stays valid even if current_leaf_id has
	 *  since moved to a different branch (FRONTEND_PLAN.md §3.1's read
	 *  contract: "a cursor that stays valid across a branch switch") — the
	 *  ancestors above any branch point are shared by every sibling branch. */
	async readOlder(convId: string, beforeId: string, limit: number): Promise<ReadPage> {
		return this.readChain(convId, beforeId, limit);
	}

	private async readChain(convId: string, startId: string | null, limit: number): Promise<ReadPage> {
		const { rows } = await this.pool.query(
			`WITH RECURSIVE start_node AS (
				SELECT CASE
					WHEN $2::uuid IS NULL THEN (SELECT current_leaf_id FROM conversation WHERE id = $1)
					ELSE (SELECT parent_id FROM message WHERE conv_id = $1 AND id = $2::uuid)
				END AS id
			 ),
			 chain(id, parent_id, depth) AS (
				SELECT m.id, m.parent_id, 0
				FROM message m, start_node s
				WHERE m.conv_id = $1 AND m.id = s.id
				UNION ALL
				SELECT m.id, m.parent_id, chain.depth + 1
				FROM message m
				JOIN chain ON m.id = chain.parent_id
				WHERE m.conv_id = $1 AND chain.depth + 1 < $3
			 )
			 SELECT msg.* FROM chain
			 JOIN message msg ON msg.conv_id = $1 AND msg.id = chain.id
			 ORDER BY chain.depth ASC`,
			[convId, startId, limit]
		);
		const messages = rows.map(rowToMessage);
		const last = messages[messages.length - 1];
		const cursor = last && last.parentId !== null ? last.id : null;
		return { messages, cursor };
	}

	/** FRONTEND_SPEC.md §11.4 — one function, one recursive query, defined
	 *  in the migration SQL (audit_conversation()) so the load path, this
	 *  wrapper, the future CLI (F30) and a future importer all share the
	 *  exact same logic rather than each reimplementing the traversal. */
	async auditConversation(convId: string): Promise<AuditResult | null> {
		// Gate remediation D4: MAX_WALK_DEPTH is passed explicitly rather than
		// left to the SQL function's own DEFAULT, so this one JS constant is
		// the single place that bounds every walk-away-from-a-proven-root
		// query in the system (selectLeaf's `up`, tombstoneSubtree's
		// `subtree`, and now audit_conversation's `sendable`) — see that
		// function's own comment in the migration SQL for why `sendable`
		// needs a bound at all when `down` provably does not.
		const { rows } = await this.pool.query('SELECT * FROM audit_conversation($1, $2)', [
			convId,
			MAX_WALK_DEPTH
		]);
		if (rows.length === 0) return null;
		const r = rows[0];
		return {
			convId: r.conv_id,
			roots: Number(r.roots),
			total: Number(r.total),
			reachableN: Number(r.reachable_n),
			missingParent: Number(r.missing_parent),
			leaf: r.leaf,
			leafOnTree: r.leaf_on_tree === true,
			chainFromCurrent: Number(r.chain_from_current),
			deepest: Number(r.deepest),
			sendableFromCurrent: Number(r.sendable_from_current),
			pass: r.pass === true
		};
	}

	/** No-op today — see the migration SQL's comment on
	 *  rebuild_read_models() and docs/lanes/L1-store.md's "read models"
	 *  section for why Phase 1 builds no materialized read model at all. */
	async rebuildReadModels(convId: string): Promise<void> {
		await this.pool.query('SELECT rebuild_read_models($1)', [convId]);
	}
}
