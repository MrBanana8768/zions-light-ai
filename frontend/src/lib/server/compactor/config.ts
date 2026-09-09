// Configuration for talking to the compactor. `process.env`, not
// SvelteKit's `$env/*` — same reasoning as ../store/db.ts: this subtree
// must import cleanly under a bare `node --test` run with no Vite/SvelteKit
// runtime present, and `process.env` is the one configuration surface both
// environments share.
//
// FRONTEND_PLAN.md D1: "Write the server transport-agnostic (COMPACTOR_URL;
// always send Authorization: Bearer when a key is configured) so the split
// out is a deploy change." The compactor being co-located on
// 127.0.0.1:8080 today is a deployment fact, not something this module
// hardcodes.

import { assertValidWindowN, DEFAULT_WINDOW_N } from './sendSet.js';

export interface TransportConfig {
	/** e.g. http://127.0.0.1:8080 — no trailing slash assumed; a trailing
	 *  slash in the configured value is stripped so callers can join paths
	 *  with a plain template string. */
	baseUrl: string;
	/** Sent as `Authorization: Bearer <key>` when present. FRONTEND_PLAN.md
	 *  §3.4: no auth exists anywhere in this tree today — this is sent
	 *  unconditionally whenever configured anyway, per D1, so the day the
	 *  compactor grows real auth this client needs no code change. */
	apiKey?: string;
	/** The client-owned generation timeout. FRONTEND_PLAN.md §3.4: the
	 *  compactor's own httpx client sets `read=None` — "no generation
	 *  timeout... a hung vLLM hangs the stream indefinitely. The client
	 *  owns that timeout." The plan names the obligation but not a number;
	 *  120s is this module's own decision, exposed as its own env var so
	 *  it does not silently inherit whatever `COMPACTOR_URL` happens to be
	 *  scoped to. See docs/lanes/L2-transport.md, "decisions not settled
	 *  for me." Gate remediation D6: this is now a genuine IDLE timeout
	 *  (refreshed on every successful read — see transport.ts), not a
	 *  total-wall-clock one, so a longer number here is now safe to choose
	 *  without making a hung backend feel responsive for that long. */
	timeoutMs: number;
	/** Gate remediation D2/D11 (docs/lanes/L2-gate-findings.md): D2 calls
	 *  for the window size to sit "behind a runtime switch with a
	 *  full-history setting" — this is what makes that literally true,
	 *  rather than `DEFAULT_WINDOW_N` being the only number that has ever
	 *  existed. Validated with the EXACT SAME rule `runGate` itself applies
	 *  to any `n` it is handed (`assertValidWindowN` — positive integer,
	 *  `<= MAX_WINDOW_N`), so a misconfigured environment fails loudly at
	 *  load time, never with a silent near-full-history send three
	 *  requests later. */
	windowN: number;
}

export const DEFAULT_TIMEOUT_MS = 120_000;

export function loadTransportConfig(env: Record<string, string | undefined> = process.env): TransportConfig {
	const rawBaseUrl = env.COMPACTOR_URL;
	if (!rawBaseUrl) {
		throw new Error('COMPACTOR_URL is not set — the server holds this configuration, the browser never sees it');
	}
	const baseUrl = rawBaseUrl.replace(/\/+$/, '');
	const apiKey = env.COMPACTOR_API_KEY || undefined;
	const rawTimeout = env.COMPACTOR_CLIENT_TIMEOUT_MS;
	let timeoutMs = DEFAULT_TIMEOUT_MS;
	if (rawTimeout !== undefined && rawTimeout !== '') {
		const parsed = Number(rawTimeout);
		if (!Number.isFinite(parsed) || parsed <= 0) {
			throw new Error(`COMPACTOR_CLIENT_TIMEOUT_MS must be a positive number, got ${JSON.stringify(rawTimeout)}`);
		}
		timeoutMs = parsed;
	}

	const rawWindowN = env.COMPACTOR_WINDOW_N;
	let windowN = DEFAULT_WINDOW_N;
	if (rawWindowN !== undefined && rawWindowN !== '') {
		const parsed = Number(rawWindowN);
		if (!Number.isInteger(parsed)) {
			throw new Error(`COMPACTOR_WINDOW_N must be an integer, got ${JSON.stringify(rawWindowN)}`);
		}
		// Reuses runGate's own validator (positive, <= MAX_WINDOW_N) rather
		// than re-deriving the rule here — one place D11's ceiling is
		// enforced, matching FRONTEND_PLAN.md §7's "one place with a test,
		// not two configs" finding about N itself.
		assertValidWindowN(parsed);
		windowN = parsed;
	}

	return { baseUrl, apiKey, timeoutMs, windowN };
}
