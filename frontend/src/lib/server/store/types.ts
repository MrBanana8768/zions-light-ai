// Public types for the L1 store. See docs/lanes/L1-store.md for the
// reasoning behind the `content`/`error` shape (raw JSON text, never a
// parsed JS value — the byte-stability obligation in FRONTEND_PLAN.md §3.2).

export type Role = 'system' | 'user' | 'assistant';
export type MessageState = 'pending' | 'streaming' | 'complete' | 'failed';

export interface Message {
	id: string;
	convId: string;
	parentId: string | null;
	userId: string;
	role: Role;
	/** The wire form, verbatim, as raw JSON text — never parsed or
	 *  re-serialized by this store on write or on read. */
	content: string;
	state: MessageState;
	/** Raw JSON text, or null. Same verbatim-text rule as `content`. */
	error: string | null;
	createdAt: string;
	updatedAt: string;
	deletedAt: string | null;
}

export interface Conversation {
	id: string;
	userId: string;
	title: string | null;
	currentLeafId: string | null;
	/** conversation.rev as a decimal string — bigint in Postgres, kept as a
	 *  string end to end (see db.ts's int8 type-parser override) so no CAS
	 *  caller can lose precision by round-tripping it through a JS number. */
	rev: string;
	createdAt: string;
	updatedAt: string;
}

export interface AuditResult {
	convId: string;
	roots: number;
	total: number;
	reachableN: number;
	missingParent: number;
	leaf: string | null;
	leafOnTree: boolean;
	/** Diagnostic only — FRONTEND_SPEC.md §13's five-tuple. Never part of
	 *  `pass`. See the forbidden-checks list in FRONTEND_SPEC.md §4.1. */
	chainFromCurrent: number;
	/** Diagnostic only, same rule as chainFromCurrent. */
	deepest: number;
	pass: boolean;
}

export interface ReadPage {
	messages: Message[];
	/** Pass as `beforeId` to readOlder() to continue. null means the walk
	 *  already reached the conversation's root — there is nothing older. */
	cursor: string | null;
}
