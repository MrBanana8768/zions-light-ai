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
import { StaleWriteError, UnreachableLeafError, NotFoundError } from './errors.js';
import type { Conversation, Message, AuditResult, Role, MessageState, ReadPage } from './types.js';

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
	 *  around. */
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
		const { rows, rowCount } = await this.pool.query(
			`UPDATE message SET ${sets.join(', ')} WHERE conv_id = $1 AND id = $2 RETURNING *`,
			values
		);
		if (rowCount !== 1) {
			throw new NotFoundError(`message ${params.messageId} not found in conversation ${params.convId}`);
		}
		return rowToMessage(rows[0]);
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
	 *  appendMessage's CAS) that may move current_leaf_id. */
	async selectLeaf(params: {
		convId: string;
		targetMessageId: string;
		expectedRev: string | number | bigint;
	}): Promise<{ rev: string }> {
		const client = await this.pool.connect();
		try {
			await client.query('BEGIN');
			const { rows } = await client.query(
				`WITH RECURSIVE up(id, parent_id, deleted_at) AS (
					SELECT id, parent_id, deleted_at FROM message WHERE conv_id = $1 AND id = $2
					UNION ALL
					SELECT m.id, m.parent_id, m.deleted_at
					FROM message m JOIN up ON m.id = up.parent_id
					WHERE m.conv_id = $1
				 )
				 SELECT count(*) AS visited,
				        count(*) FILTER (WHERE deleted_at IS NOT NULL) AS tombstoned,
				        bool_or(parent_id IS NULL) AS reached_root
				 FROM up`,
				[params.convId, params.targetMessageId]
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
			if (!(err instanceof StaleWriteError) && !(err instanceof UnreachableLeafError)) {
				await rollbackQuietly(client);
			}
			throw err;
		} finally {
			client.release();
		}
	}

	/** Subtree-wide tombstone. Sets deleted_at on the target AND every
	 *  descendant in one statement (a recursive walk down from the target),
	 *  so hiding a message can never leave its children reachable-but-
	 *  orphaned-looking or, worse, visibly dangling in a render while their
	 *  parent is gone from view (FRONTEND_SPEC.md §11.1). Does not touch
	 *  current_leaf_id — if the current leaf is inside the tombstoned
	 *  subtree, the conversation now legitimately fails leaf_on_tree's
	 *  spirit (the leaf points at now-hidden content) until an explicit
	 *  selectLeaf() moves it elsewhere; this store does not do that move
	 *  silently, per D6. */
	async tombstoneSubtree(params: { convId: string; rootMessageId: string }): Promise<number> {
		const { rowCount } = await this.pool.query(
			`WITH RECURSIVE subtree(id) AS (
				SELECT id FROM message WHERE conv_id = $1 AND id = $2
				UNION ALL
				SELECT m.id FROM message m JOIN subtree ON m.parent_id = subtree.id
				WHERE m.conv_id = $1
			 )
			 UPDATE message SET deleted_at = now(), updated_at = now()
			 WHERE conv_id = $1 AND id IN (SELECT id FROM subtree) AND deleted_at IS NULL`,
			[params.convId, params.rootMessageId]
		);
		return rowCount ?? 0;
	}

	// ---- reads -----------------------------------------------------------

	async getConversation(convId: string): Promise<Conversation | null> {
		const { rows } = await this.pool.query('SELECT * FROM conversation WHERE id = $1', [convId]);
		return rows.length ? rowToConversation(rows[0]) : null;
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
		const { rows } = await this.pool.query('SELECT * FROM audit_conversation($1)', [convId]);
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
