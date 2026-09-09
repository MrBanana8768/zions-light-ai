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

import type { AuditResult, Message } from '../store/types.js';
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
 *  COMPACTOR_MAX_SUMMARY_CALLS... one place with a test, not two configs").
 *  config.ts's `windowN` is what actually makes the "runtime switch" real
 *  (D11) — this is only the compiled-in default it falls back to. */
export const DEFAULT_WINDOW_N = 60;

/** Gate remediation D11 (docs/lanes/L2-gate-findings.md): a hard ceiling on
 *  `n`. `runGate(chain, id, 100000)` on a real conversation used to return
 *  `ok: true` with `sentCount` near the conversation's full length —
 *  FRONTEND_SPEC.md §4 rule 2's "sending full history is a spec violation"
 *  with the gate having no opinion at all. FRONTEND_PLAN.md §3.2's own
 *  derivation table considers, and REJECTS, N=100 as already too large for
 *  the inline-summarizer budget (§3.2: "a 100-turn cap re-latches inline
 *  summarization precisely when things are already going wrong"); 200 is
 *  2x that already-rejected value — generous headroom for legitimate
 *  runtime tuning while making "send me everything" structurally
 *  impossible to reach through this parameter. Not a value to raise
 *  casually: raising it is a deliberate decision about how much history is
 *  ever legitimate, not a workaround for a specific conversation. */
export const MAX_WINDOW_N = 200;

/** Gate remediation D11: `n` must be a positive integer at or under
 *  MAX_WINDOW_N, rejected LOUDLY (a thrown Error, not a silently-clamped
 *  value or a GateOutcome) otherwise — this is a caller/configuration
 *  defect, not a runtime chain condition, so it does not get a typed
 *  GateFailure the way a bad CHAIN does. */
export function assertValidWindowN(n: number): void {
	if (!Number.isInteger(n) || n <= 0) {
		throw new Error(`window size n must be a positive integer, got ${JSON.stringify(n)}`);
	}
	if (n > MAX_WINDOW_N) {
		throw new Error(
			`window size n=${n} exceeds MAX_WINDOW_N=${MAX_WINDOW_N} — sending anywhere near full ` +
				'history is exactly the spec violation D11 exists to prevent (FRONTEND_SPEC.md §4 rule 2); ' +
				'raise MAX_WINDOW_N deliberately if a larger window is ever genuinely wanted, do not pass a ' +
				'large n around it'
		);
	}
}

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

/** Gate remediation D8 (docs/lanes/L2-gate-findings.md). Used to be
 *  `readOlder(convId, cursor, ROOT_LOOKUP_LIMIT)` — walk up to 100,000 rows
 *  (content included; `readChain`'s recursive CTE returns every row it
 *  visits) purely to read one row's `id`. `getRoot` is a one-row, indexed
 *  lookup (`message_one_root_per_conv`'s own partial unique index answers
 *  it directly — see migrations/0001_init.sql) — O(1) regardless of
 *  conversation length, not merely bounded-and-usually-cheap. Returns null
 *  (never a non-root message masquerading as the persona) if the
 *  conversation genuinely has no root, which turns this into a visible
 *  context_truncated rather than a silent wrong system prompt. */
async function fetchRoot(chain: ChainReader, convId: string): Promise<Message | null> {
	return chain.getRoot(convId);
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
	// to get the n newest, then fetch the persona root as its own,
	// O(1)-indexed read (D8 — no longer a second O(n) chain walk).
	const candidateTurns = chrono.length > n ? chrono.slice(chrono.length - n) : chrono;
	const systemMessage = await fetchRoot(chain, convId);
	return { systemMessage, realized: walkForwardOffLeadingAssistant(candidateTurns) };
}

/** Gate remediation D1/D4/D10 (docs/lanes/L2-gate-findings.md). Every
 *  structural property a send-set window must have, independent of audit
 *  freshness or `window_intent` — contiguous, strictly alternating, opens
 *  AND ends on 'user' (D1 — the original version of this check only ever
 *  looked at the FIRST turn's role, which is exactly what let a window
 *  ending on an assistant turn through the gate green), every turn
 *  `state = 'complete'` and not tombstoned (D4), and every turn actually
 *  belongs to `convId`. Factored out of verifyRealizedWindow so the exact
 *  same check can run AGAIN, defensively, at the point of posting
 *  (request.ts's buildChatCompletionBody) — D10's "bind the gate's output
 *  to what is posted" — rather than trusting a GateSuccess's shape by
 *  convention. Named as a list, not a single boolean, matching §4.1's own
 *  "state what was found" repair policy. */
export function verifyChainShape(convId: string, turns: Message[]): string[] {
	const reasons: string[] = [];

	if (turns.length === 0) {
		reasons.push('the window contains no turns at all');
		return reasons;
	}

	for (const m of turns) {
		if (m.convId !== convId) {
			reasons.push(`turn ${m.id} belongs to conversation ${m.convId}, not ${convId}`);
		}
		if (m.role === 'system') {
			reasons.push(`turn ${m.id} is a system message inside the window`);
		}
		if (m.deletedAt) {
			reasons.push(`turn ${m.id} is tombstoned`);
		}
		// D4: a failed/streaming/pending row must never be sent — the model
		// must never be shown a blank prior utterance of its own (a failed
		// turn), and a streaming row's content is not yet stable (the
		// byte-stability obligation this same content feeds into for
		// fingerprinting). The store's monotonic state machine (failed is
		// terminal) means a healthy retry never leaves a failed row on the
		// active chain at all — see this file's header comment and
		// docs/lanes/L2-transport.md's "Gate remediation" section for why
		// this is a defensive backstop, not the primary mechanism.
		if (m.state !== 'complete') {
			reasons.push(`turn ${m.id} is not complete (state=${m.state}) — nothing built from it is sendable`);
		}
	}

	// contiguous + in order: every turn's parent is the previous turn.
	for (let i = 1; i < turns.length; i++) {
		if (turns[i].parentId !== turns[i - 1].id) {
			reasons.push(`turn ${turns[i].id} is not contiguous with the previous turn in the window`);
			break;
		}
	}

	// alternation intact, opening AND ending on 'user'.
	if (turns[0].role !== 'user') {
		reasons.push('the window does not open on a user turn');
	}
	if (turns[turns.length - 1].role !== 'user') {
		reasons.push('the window does not end on a user turn');
	}
	for (let i = 1; i < turns.length; i++) {
		if (turns[i].role === turns[i - 1].role) {
			reasons.push(`turns ${turns[i - 1].id} and ${turns[i].id} are consecutive same-role turns`);
			break;
		}
	}

	return reasons;
}

/** All the ways a realized (systemMessage, turns) pair can fail to be a
 *  legal send set. Layers the audit-specific checks (persona root
 *  soundness, freshness against the audited current leaf, count vs.
 *  window_intent) around `verifyChainShape`'s structural checks — those
 *  two categories are deliberately kept separate: this function needs an
 *  `AuditResult` to do its job; `verifyChainShape` does not, which is
 *  exactly what lets request.ts reuse it as a defensive re-check with no
 *  audit in hand (D10). Every property named in FRONTEND_SPEC.md §4 rules
 *  2/3/7 is checked explicitly; none is inferred from an adjacent one
 *  (mirroring §4.1's own "no adjacent property substituted" discipline for
 *  the store's five-tuple). */
function verifyRealizedWindow(
	systemMessage: Message | null,
	turns: Message[],
	windowIntent: number,
	audit: AuditResult
): string[] {
	const reasons: string[] = [];

	if (!systemMessage) {
		reasons.push('the system/persona message could not be located');
	} else {
		if (systemMessage.role !== 'system') reasons.push('the resolved root is not a system message');
		if (systemMessage.parentId !== null) reasons.push('the resolved root is not the conversation root');
		if (systemMessage.deletedAt) reasons.push('the resolved root is tombstoned');
	}

	reasons.push(...verifyChainShape(audit.convId, turns));

	// Gate remediation D1/D2/D3: the window's last turn must be the
	// AUDITED current leaf — not merely "some user turn." This is what
	// turns the count check back into a real, two-bit signal instead of
	// one: D2's equal-length-branch-switch race (gate audits branch A,
	// readTail walks branch B after a concurrent selectLeaf — same root,
	// same count, same alternation, DIFFERENT leaf) is caught here and
	// nowhere else, because `audit.leaf` was captured from the SAME read
	// that produced `windowIntent`, before this window was ever fetched.
	if (turns.length > 0) {
		const lastTurn = turns[turns.length - 1];
		if (audit.leaf !== lastTurn.id) {
			reasons.push(
				`the window's last turn (${lastTurn.id}) does not match the conversation's audited current ` +
					`leaf (${audit.leaf ?? 'null'})`
			);
		}
	}

	// count equal to window_intent. This is a REAL check, not a tautology:
	// windowIntent was computed from the audit BEFORE this window was
	// fetched (see computeWindowIntent's own comment), so a mismatch here
	// means the chain actually changed between the audit read and this
	// read (a race — a concurrent append or branch switch), or the data
	// does not alternate the way a healthy chain would. Both are exactly
	// what context_truncated exists to catch.
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
 *  function issues reads only (auditConversation, readTail, and — for a
 *  long conversation — getRoot; D8 retired this file's own use of
 *  readOlder). */
export async function runGate(
	chain: ChainReader,
	convId: string,
	n: number = DEFAULT_WINDOW_N
): Promise<GateOutcome> {
	// D11: rejected loudly, before a single read — a misconfigured n is a
	// caller/config defect, not a runtime chain condition.
	assertValidWindowN(n);

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

	// D4: turnsOnChain is now derived from `sendable_from_current` — the
	// SQL-side, complete-and-not-tombstoned-suffix count — not from the
	// raw structural `chain_from_current`. This is what keeps
	// window_intent an INDEPENDENTLY-derived number even in the presence
	// of a non-complete turn: if the leaf itself isn't complete (still
	// streaming, still pending, or a failed leaf nobody has retried away
	// from yet), sendableFromCurrent is 0, turnsOnChain is 0, windowIntent
	// is 0, and the REAL realized window (whatever readTail actually
	// finds) will not match it — refusing with a typed reason, never a
	// bare throw. See sendSet.ts's header comment and
	// docs/lanes/L2-gate-findings.md's D4.
	const turnsOnChain = Math.max(0, audit.sendableFromCurrent - 1); // subtract the synthetic system root
	const windowIntent = computeWindowIntent(turnsOnChain, n);

	const { systemMessage, realized } = await selectFromChain(chain, convId, n);
	const reasons = verifyRealizedWindow(systemMessage, realized, windowIntent, audit);

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
