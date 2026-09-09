// A minimal, in-memory ChainReader for L2's own tests — deliberately NOT
// the real Store. This lane's tests exercise the SEND-SET ALGORITHM
// against controlled, adversarial chain shapes; the real Store (L1's own
// suite, frontend/tests/store/**) already proves the Postgres constraints
// enforce the shapes this fake merely represents. Its `auditConversation`
// mirrors the migration's `audit_conversation()` SQL function closely
// enough to be a faithful stand-in (same five-tuple, same PASS formula —
// see migrations/0001_init.sql's own comment on that function) without
// depending on a live Postgres sidecar, per this lane's brief: "Mock the
// compactor's HTTP surface — do not require a live backend." That
// instruction is about the compactor's HTTP surface specifically; this is
// its natural extension to the store dependency, so the whole suite can
// run with plain `node --test` and no Docker sidecar.

import type { AuditResult, Conversation, Message, ReadPage, Role } from '../../../src/lib/server/store/types.js';
import type { ChainReader } from '../../../src/lib/server/compactor/types.js';

let counter = 0;
function nextId(prefix: string): string {
	counter += 1;
	return `${prefix}-${counter.toString().padStart(8, '0')}`;
}

export function makeMessage(
	overrides: Partial<Message> & Pick<Message, 'id' | 'convId' | 'role'>
): Message {
	const now = new Date().toISOString();
	return {
		parentId: null,
		userId: 'user-1',
		content: JSON.stringify('x'),
		state: 'complete',
		error: null,
		createdAt: now,
		updatedAt: now,
		deletedAt: null,
		...overrides
	};
}

export class FakeChainReader implements ChainReader {
	private readonly messages = new Map<string, Message>();

	constructor(public conv: Conversation) {}

	addMessage(m: Message): void {
		this.messages.set(m.id, m);
	}

	getMessage(id: string): Message | undefined {
		return this.messages.get(id);
	}

	async getConversation(convId: string): Promise<Conversation | null> {
		return this.conv.id === convId ? { ...this.conv } : null;
	}

	/** Mirrors audit_conversation()'s own walk: a `down` traversal from
	 *  every parentId===null row, PASS iff roots===1 AND missingParent===0
	 *  AND reachableN===total AND leafOnTree — see migrations/0001_init.sql
	 *  for the SQL this is a JS stand-in for. Deliberately does not prune
	 *  tombstones from reachability (property 5 is about orphans, not
	 *  visibility — the real function's own comment explains why). */
	async auditConversation(convId: string): Promise<AuditResult | null> {
		if (this.conv.id !== convId) return null;
		const all = [...this.messages.values()].filter((m) => m.convId === convId);
		const roots = all.filter((m) => m.parentId === null);

		const byParent = new Map<string, Message[]>();
		for (const m of all) {
			if (m.parentId !== null) {
				const arr = byParent.get(m.parentId) ?? [];
				arr.push(m);
				byParent.set(m.parentId, arr);
			}
		}

		const depthOf = new Map<string, number>();
		let deepest = 0;
		const stack: Array<[string, number]> = roots.map((r) => [r.id, 1]);
		while (stack.length > 0) {
			const [id, depth] = stack.pop() as [string, number];
			if (depthOf.has(id)) continue; // cycle guard; not expected in these fixtures
			depthOf.set(id, depth);
			deepest = Math.max(deepest, depth);
			for (const child of byParent.get(id) ?? []) stack.push([child.id, depth + 1]);
		}

		const missingParent = all.filter(
			(m) => m.parentId !== null && !this.messages.has(m.parentId)
		).length;
		const leaf = this.conv.currentLeafId;
		const leafOnTree = leaf !== null && depthOf.has(leaf);
		const chainFromCurrent = leaf !== null && depthOf.has(leaf) ? (depthOf.get(leaf) as number) : 0;
		const pass = roots.length === 1 && missingParent === 0 && depthOf.size === all.length && leafOnTree;

		// Gate remediation D4 (docs/lanes/L2-gate-findings.md). Mirrors
		// migrations/0001_init.sql's `sendable` CTE: walk UPWARD from the
		// current leaf via parentId, counting while each visited message is
		// `state === 'complete'` and not tombstoned, stopping at the FIRST
		// one that is not (never visiting, and never counting, anything
		// further up). Deliberately NOT part of `pass` — see this file's own
		// header comment and the SQL function's, for why sendability and
		// structural soundness are kept independent.
		let sendableFromCurrent = 0;
		let cur: string | null = leaf;
		while (cur !== null) {
			const m = this.messages.get(cur);
			if (!m) break;
			const ok = m.state === 'complete' && m.deletedAt === null;
			if (!ok) break;
			sendableFromCurrent += 1;
			cur = m.parentId;
		}

		return {
			convId,
			roots: roots.length,
			total: all.length,
			reachableN: depthOf.size,
			missingParent,
			leaf,
			leafOnTree,
			chainFromCurrent,
			deepest,
			sendableFromCurrent,
			pass
		};
	}

	/** Gate remediation D8 (docs/lanes/L2-gate-findings.md): a one-row
	 *  lookup of the conversation's synthetic root, mirroring store.ts's
	 *  `getRoot` — used so sendSet.test.ts can assert this is called
	 *  (and readOlder is NOT) instead of the retired O(n) root-lookup
	 *  pattern. */
	async getRoot(convId: string): Promise<Message | null> {
		if (this.conv.id !== convId) return null;
		for (const m of this.messages.values()) {
			if (m.convId === convId && m.parentId === null) return m;
		}
		return null;
	}

	async readTail(convId: string, limit: number): Promise<ReadPage> {
		return this.readChain(convId, null, limit);
	}

	async readOlder(convId: string, beforeId: string, limit: number): Promise<ReadPage> {
		return this.readChain(convId, beforeId, limit);
	}

	/** Mirrors store.ts's readChain: newest-first, walking parent_id
	 *  starting at current_leaf_id (startId null) or at beforeId's PARENT
	 *  (continuing a page), bounded by `limit`, cursor = the oldest
	 *  returned message's id, or null once its parentId is null. */
	private async readChain(
		convId: string,
		startId: string | null,
		limit: number
	): Promise<ReadPage> {
		if (this.conv.id !== convId) return { messages: [], cursor: null };
		let cur: string | null;
		if (startId === null) {
			cur = this.conv.currentLeafId;
		} else {
			const startMsg = this.messages.get(startId);
			cur = startMsg ? startMsg.parentId : null;
		}
		const out: Message[] = [];
		let depth = 0;
		while (cur !== null && depth < limit) {
			const m = this.messages.get(cur);
			if (!m || m.convId !== convId) break;
			out.push(m);
			cur = m.parentId;
			depth += 1;
		}
		const last = out[out.length - 1];
		const cursor = last && last.parentId !== null ? last.id : null;
		return { messages: out, cursor };
	}
}

export interface BuiltConversation {
	chain: FakeChainReader;
	conv: Conversation;
	root: Message;
	/** oldest-first */
	turns: Message[];
}

/** A healthy, strictly-alternating conversation: one synthetic system
 *  root, then `turnCount` turns alternating user/assistant/user/... .
 *  `currentLeafId` defaults to the newest turn (or the root, if
 *  `turnCount` is 0) — matching this lane's resolution of U-2 ("the gate
 *  always runs after the user's new turn is already the current leaf"),
 *  so an odd `turnCount` is the normal case for a gate-ready fixture. */
export function buildConversation(params: {
	turnCount: number;
	convId?: string;
	userId?: string;
	systemContent?: string;
	turnContent?: (index: number, role: Role) => string;
}): BuiltConversation {
	const convId = params.convId ?? nextId('conv');
	const userId = params.userId ?? 'user-1';
	const root = makeMessage({
		id: nextId('msg'),
		convId,
		role: 'system',
		userId,
		parentId: null,
		content: JSON.stringify(params.systemContent ?? 'you are a helpful assistant')
	});

	const turns: Message[] = [];
	let parentId = root.id;
	for (let i = 0; i < params.turnCount; i++) {
		const role: Role = i % 2 === 0 ? 'user' : 'assistant';
		const text = params.turnContent ? params.turnContent(i, role) : `turn-${i}`;
		const m = makeMessage({
			id: nextId('msg'),
			convId,
			role,
			userId,
			parentId,
			content: JSON.stringify(text)
		});
		turns.push(m);
		parentId = m.id;
	}

	const now = new Date().toISOString();
	const conv: Conversation = {
		id: convId,
		userId,
		title: null,
		currentLeafId: turns.length > 0 ? turns[turns.length - 1].id : root.id,
		rev: String(turns.length),
		createdAt: now,
		updatedAt: now
	};

	const chain = new FakeChainReader(conv);
	chain.addMessage(root);
	for (const t of turns) chain.addMessage(t);

	return { chain, conv, root, turns };
}
