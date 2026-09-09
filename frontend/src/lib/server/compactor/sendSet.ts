// F8 — the checked send-set and the pre-send gate. FRONTEND_PLAN.md §3.2 /
// FRONTEND_SPEC.md §4 rules 2/3/7 and §4.1's "pre-send gate". This is the
// module the whole lane exists for: read docs/lanes/L2-transport.md's
// "the send-set algorithm and how the gate verifies it" section before
// changing anything here, it explains three decisions this file makes that
// are NOT simply "what the plan says", because the plan's literal formula
// is ambiguous (see below) in a way that would make the gate refuse a
// large fraction of ordinary sends if resolved the wrong way.
//
// ---------------------------------------------------------------------------
// WHY window_intent IS NOT A BARE min(N, turnsOnChain)
// ---------------------------------------------------------------------------
// FRONTEND_PLAN.md §3.2 says, verbatim: "window_intent = min(N, turns on the
// current chain)" and then "verify... count equal to window_intent." Taken
// completely literally, those two sentences conflict with FRONTEND_SPEC.md
// §4 rule 3 (never send two consecutive same-role turns; the window must
// open on 'user') for a reason the plan's own N=60 derivation never
// mentions: N is EVEN.
//
// U-2 (FRONTEND_PLAN.md §3.5) leaves open exactly when the user's new
// message is written relative to this gate running. This module resolves
// it: THE GATE ALWAYS RUNS AFTER THE USER'S NEW TURN IS ALREADY THE CURRENT
// LEAF (appended by the caller, as an ordinary `complete` message, before
// asking this module for a send set — see the header of transport.ts for
// where that append belongs). That means, for any healthy, strictly-
// alternating conversation, turns_on_the_current_chain is ALWAYS ODD (the
// chain starts at the first user turn and ends at the one just typed,
// awaiting a reply — an odd-length alternating sequence starting and
// ending on the same role).
//
// Combine odd-length-always with N=60 (even): whenever turnsOnChain > N,
// the naive "last N turns" slice starts at 0-based offset
// (turnsOnChain - N) = odd - even = ODD, i.e. an 'assistant' turn — the
// exact shape the Mistral template refuses outright, and the exact shape
// `_cap()` (pipelines/conversation_id_header.py:195-222) exists to fix by
// walking forward off the leading assistant turn(s). That fix is
// MANDATORY, not optional — sending the unfixed window gets a 400 from
// vLLM's chat template, which is strictly worse than sending one turn
// fewer. So for ANY healthy conversation past 60 messages, the send set
// that must actually go out is 59 turns, not 60 — deterministically, every
// single time, forever, for as long as N stays even.
//
// If window_intent were recorded as the bare min(N, turnsOnChain) = 60 and
// then compared against that mandatory 59-turn realization, EVERY turn
// past message 60 would fail verification and demand a manual D4 override
// click — not an occasional anomaly, but the steady-state UX of every long
// conversation. Nothing in the plan's Phase 1/2/3 acceptance criteria, the
// manual gates, or the six phases' definitions of done describes that as
// intended, and D4's own framing ("an override the daily user can reach")
// reads as an escape hatch for a rare event, not a button clicked on every
// message. Read together with the two corrections the plan already
// documents against itself (§7.1) and the standing invitation to find a
// fourth, this reads as exactly that: a genuine ambiguity in the plan, not
// a subtlety this module should paper over silently.
//
// THE RESOLUTION THIS MODULE SHIPS: window_intent is computed as the
// alternation-preserving achievable count — computeWindowIntent() below —
// so the ROUTINE one-turn parity trim is folded into what "intent" means,
// and `context_truncated` is reserved for what it is actually for: a
// realized send set that differs from what the CURRENT, ACTUAL chain data
// supports, i.e. races, corruption, or a shortfall bigger than the one
// mandatory turn. window_intent is still computed BEFORE the chain is
// re-read for the actual content, from the audit alone, so the equality
// check in verifySendSet() is a REAL check, not a tautology — see that
// function's own comment.
//
// This is a plan correction, made explicit rather than silently encoded,
// exactly as FRONTEND_PLAN.md §3.4's own corrections section models: state
// what the source says, state what is actually true, state the
// consequence. Recorded again, with the full derivation, in
// docs/lanes/L2-transport.md for the Opus gate to accept or override.

import type { Message } from '../store/types.js';
import type {
	ChainReader,
	GateFailure,
	GateOutcome,
	GateSuccess,
	OverrideRecord
} from './types.js';

/** D2: N = 60 non-system messages. Exported (not hardcoded at call sites)
 *  because D2 also promises "behind a runtime switch with a full-history
 *  setting" — one place to change, matching FRONTEND_PLAN.md §7's own
 *  finding ("N is coupled to L1_CHUNK_SIZE / KEEP_RECENT_TURNS /
 *  COMPACTOR_MAX_SUMMARY_CALLS... one place with a test, not two configs"). */
export const DEFAULT_WINDOW_N = 60;

/** Safety ceiling for the one-shot walk to the conversation's synthetic
 *  root when the tail page didn't already include it (long-conversation
 *  branch, below). Not a hop BUDGET — store.ts's readChain recursive CTE
 *  stops the instant it reaches a row with parent_id IS NULL regardless of
 *  how large `limit` is, so this number costs nothing when the real chain
 *  is shorter than it (every realistic conversation). It exists only so a
 *  corrupt/cyclic chain (§11.5 grants a human exactly the freedom to
 *  construct one from outside the app) fails this call cleanly instead of
 *  the query running away — mirroring store.ts's own MAX_WALK_DEPTH
 *  pattern, at 5x its value because this walk starts from wherever the
 *  window happened to end rather than from a proven root. */
const ROOT_LOOKUP_LIMIT = 100_000;

/** The alternation-preserving achievable count for a window request,
 *  computed WITHOUT reading any message content — see this file's header
 *  comment for the full derivation. Assumes (does not verify — that is
 *  verifySendSet's job) that the chain is healthy: starts at 'user',
 *  strictly alternates, and — per this module's resolution of U-2 — ends
 *  at 'user' (the turn just typed, awaiting a reply) at the moment the
 *  gate runs. Under that assumption, turnsOnChain is always odd, so the
 *  only two possible outcomes are "the whole chain fits" or "exactly one
 *  turn short of N" — this function returns the general formula rather
 *  than hardcoding the N=60 special case, so it stays correct if D2's
 *  "runtime switch" ever picks a different N. */
export function computeWindowIntent(turnsOnChain: number, n: number): number {
	if (turnsOnChain <= n) return turnsOnChain;
	const startIndex = turnsOnChain - n; // 0-based, counting from the first user turn
	return startIndex % 2 === 0 ? n : n - 1;
}

async function fetchRoot(
	chain: ChainReader,
	convId: string,
	fromId: string | null
): Promise<Message | null> {
	if (fromId === null) return null;
	const page = await chain.readOlder(convId, fromId, ROOT_LOOKUP_LIMIT);
	if (page.messages.length === 0) return null;
	const oldest = page.messages[page.messages.length - 1];
	// Defensive: if this is false, ROOT_LOOKUP_LIMIT was exhausted without
	// reaching a root — a corrupt/cyclic chain, or a conversation far
	// beyond any bar this project has stated. Either way, returning null
	// (rather than a non-root message masquerading as the persona) is what
	// turns this into a visible context_truncated instead of a silent
	// wrong system prompt.
	return oldest.parentId === null ? oldest : null;
}

/** `_cap()`'s mandatory shape fix (pipelines/conversation_id_header.py:195-222),
 *  applied to REAL fetched turns rather than derived arithmetically — see
 *  this file's header comment for why arithmetic alone cannot be trusted
 *  here (a corrupt chain might not alternate the way the formula assumes,
 *  and this walk must handle that data exactly as safely as healthy data).
 *  Mirrors the reference's own degenerate case: stripping every turn
 *  leaves nothing to send, so the reference forwards the untouched
 *  (shape-illegal) window rather than an empty one — left equally
 *  unfixed here, and caught by verifySendSet's alternation check instead
 *  of silently sent. */
function walkForwardOffLeadingAssistant(turns: Message[]): Message[] {
	const stripped = [...turns];
	while (stripped.length > 0 && stripped[0].role === 'assistant') {
		stripped.shift();
	}
	return stripped.length > 0 ? stripped : turns;
}

/** Builds the realized (system message, turn window) pair by explicit
 *  selection over the chain from the current leaf — FRONTEND_SPEC.md §4
 *  rule 7. Never accepts a caller-supplied message array: every message in
 *  the result was read from the store, via the store's own chain-walking
 *  reads, in this call. That is what makes property 3 (the render-set
 *  containment the plan's §3.1 calls out as "cannot be inherited from the
 *  DDL") true of this module's OWN output by construction — there is no
 *  code path here that could hand back a message not actually on the
 *  active chain, because there is no code path that accepts one from
 *  outside. */
async function selectFromChain(
	chain: ChainReader,
	convId: string,
	n: number
): Promise<{ systemMessage: Message | null; realized: Message[] }> {
	const firstPage = await chain.readTail(convId, n + 1); // newest-first, <= n+1 rows
	const chrono = [...firstPage.messages].reverse(); // oldest-first among what was fetched

	if (chrono.length > 0 && chrono[0].parentId === null) {
		// The whole active chain fit inside one n+1 page — root included.
		const systemMessage = chrono[0];
		const candidateTurns = chrono.slice(1);
		return { systemMessage, realized: walkForwardOffLeadingAssistant(candidateTurns) };
	}

	// Long conversation: this page is entirely non-system. It holds n+1
	// candidates purely so the check above could detect root-inclusion in
	// one round trip; drop the oldest one (kept only for that detection)
	// to get the n newest, then fetch the persona root as its own read.
	const candidateTurns = chrono.length > n ? chrono.slice(chrono.length - n) : chrono;
	const systemMessage = await fetchRoot(chain, convId, firstPage.cursor);
	return { systemMessage, realized: walkForwardOffLeadingAssistant(candidateTurns) };
}

/** All the ways a realized (systemMessage, turns) pair can fail to be a
 *  legal send set. Named as a list (rather than a single boolean) because
 *  §4.1's own repair policy is "state what was found" — a caller building
 *  a context_truncated notice should be able to show every reason, not
 *  just the first one this function happened to notice. Every property
 *  named in FRONTEND_SPEC.md §4 rules 2/3/7 is checked explicitly; none is
 *  inferred from an adjacent one (mirroring §4.1's own "no adjacent
 *  property substituted" discipline for the store's five-tuple). */
function verifyRealizedWindow(
	systemMessage: Message | null,
	turns: Message[],
	windowIntent: number
): string[] {
	const reasons: string[] = [];

	if (!systemMessage) {
		reasons.push('the system/persona message could not be located');
	} else {
		if (systemMessage.role !== 'system') reasons.push('the resolved root is not a system message');
		if (systemMessage.parentId !== null) reasons.push('the resolved root is not the conversation root');
		if (systemMessage.deletedAt) reasons.push('the resolved root is tombstoned');
	}

	for (const m of turns) {
		if (m.role === 'system') reasons.push(`turn ${m.id} is a system message inside the window`);
		if (m.deletedAt) reasons.push(`turn ${m.id} is tombstoned`);
	}

	// contiguous + in order: every turn's parent is the previous turn.
	for (let i = 1; i < turns.length; i++) {
		if (turns[i].parentId !== turns[i - 1].id) {
			reasons.push(`turn ${turns[i].id} is not contiguous with the previous turn in the window`);
			break;
		}
	}

	// alternation intact, opening on 'user'.
	if (turns.length > 0 && turns[0].role !== 'user') {
		reasons.push('the window does not open on a user turn');
	}
	for (let i = 1; i < turns.length; i++) {
		if (turns[i].role === turns[i - 1].role) {
			reasons.push(`turns ${turns[i - 1].id} and ${turns[i].id} are consecutive same-role turns`);
			break;
		}
	}

	// every element present, count equal to window_intent. This is a REAL
	// check, not a tautology: windowIntent was computed from the audit
	// BEFORE this window was fetched (see computeWindowIntent's own
	// comment), so a mismatch here means the chain actually changed
	// between the audit read and this read (a race — a concurrent append
	// or branch switch), or the data does not alternate the way a healthy
	// chain would, or ROOT_LOOKUP_LIMIT was exhausted. All three are
	// exactly what context_truncated exists to catch.
	if (turns.length !== windowIntent) {
		reasons.push(`realized ${turns.length} turn(s), intended ${windowIntent}`);
	}

	return reasons;
}

/** The pre-send gate. FRONTEND_SPEC.md §4.1: "Before every request,
 *  recompute the send set from the chain and compare it to window_intent.
 *  On mismatch, do not send." Callers must have ALREADY appended the
 *  user's new turn (making it the current leaf) before calling this — see
 *  this file's header comment on U-2. Never mutates anything: this
 *  function issues reads only (auditConversation, readTail, readOlder). */
export async function runGate(
	chain: ChainReader,
	convId: string,
	n: number = DEFAULT_WINDOW_N
): Promise<GateOutcome> {
	const audit = await chain.auditConversation(convId);
	if (!audit) {
		return { ok: false, convId, kind: 'not_found', reasons: [`conversation ${convId} not found`] };
	}
	if (!audit.pass) {
		return {
			ok: false,
			convId,
			kind: 'chain_unsound',
			reasons: [
				'audit_conversation did not pass — see the attached audit for the five-tuple ' +
					'(this is FRONTEND_SPEC.md §4.1\'s chain_corrupt condition, not this lane\'s ' +
					'context_truncated; route it to that notice instead)'
			],
			audit
		};
	}

	const turnsOnChain = Math.max(0, audit.chainFromCurrent - 1); // subtract the synthetic system root
	const windowIntent = computeWindowIntent(turnsOnChain, n);

	const { systemMessage, realized } = await selectFromChain(chain, convId, n);
	const reasons = verifyRealizedWindow(systemMessage, realized, windowIntent);

	const sentCount = realized.length;
	if (reasons.length > 0) {
		const failure: GateFailure = {
			ok: false,
			convId,
			kind: 'context_truncated',
			reasons,
			windowIntent,
			systemMessage,
			turns: realized,
			sentCount
		};
		return failure;
	}

	const success: GateSuccess = {
		ok: true,
		convId,
		windowIntent,
		// verifyRealizedWindow already required systemMessage to be
		// non-null for `reasons` to be empty, so this cast is provably
		// safe here, not an assumption.
		systemMessage: systemMessage as Message,
		turns: realized,
		sentCount
	};
	return success;
}

/** D4's override path. Takes a `context_truncated` GateFailure and turns
 *  it into a record marked "sent under override" — WITHOUT calling
 *  runGate again, without touching `windowIntent`, and without altering
 *  `turns`/`systemMessage`/`sentCount` in any way. Every field on the
 *  returned record is copied, not recomputed, from the failure that
 *  produced it: "recompute and resend" is the defect wearing a consent
 *  dialog, so there is deliberately no code path here that could recompute
 *  anything. If `failure.systemMessage` is null (the gate could not even
 *  locate a persona to send), the override still cannot fabricate one —
 *  callers must treat a null `systemMessage` as "there is nothing safe to
 *  send, override or not" rather than calling this function at all. */
export function applyOverride(failure: GateFailure): OverrideRecord {
	if (failure.kind !== 'context_truncated') {
		throw new Error(
			`applyOverride is only defined for 'context_truncated' failures, got '${failure.kind}'. ` +
				`'chain_unsound' and 'not_found' have no coherent send set to override — they route to a ` +
				`different notice (chain_corrupt / not_found), never through this function.`
		);
	}
	return {
		sentUnderOverride: true,
		convId: failure.convId,
		originalWindowIntent: failure.windowIntent ?? 0,
		systemMessage: failure.systemMessage ?? null,
		turns: failure.turns ?? [],
		sentCount: failure.sentCount ?? 0,
		reasons: failure.reasons
	};
}
