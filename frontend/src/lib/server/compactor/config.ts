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
	 *  for me." */
	timeoutMs: number;
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
	return { baseUrl, apiKey, timeoutMs };
}
