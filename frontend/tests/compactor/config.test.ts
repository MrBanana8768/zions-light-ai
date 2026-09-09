// config.ts had NO test file and NO mutation before this pass (the
// mutation audit's own finding: "COMPACTOR_CLIENT_TIMEOUT_MS=0.5 is
// unguarded" — not because 0.5 is wrong, but because nothing ever ran
// either branch of its validation, so a regression in it was invisible).
// This file closes that gap AND covers D11's own runtime-switch validation
// (windowN).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_TIMEOUT_MS, loadTransportConfig } from '../../src/lib/server/compactor/config.js';
import { DEFAULT_WINDOW_N, MAX_WINDOW_N } from '../../src/lib/server/compactor/sendSet.js';

function env(overrides: Record<string, string | undefined> = {}): Record<string, string | undefined> {
	return { COMPACTOR_URL: 'http://127.0.0.1:8080', ...overrides };
}

// ---------------------------------------------------------------------------
// baseUrl / apiKey
// ---------------------------------------------------------------------------

test('loadTransportConfig: throws when COMPACTOR_URL is unset — the server holds this config, never the browser', () => {
	assert.throws(() => loadTransportConfig({}), /COMPACTOR_URL is not set/);
});

test('loadTransportConfig: strips a trailing slash (or several) from baseUrl', () => {
	assert.equal(loadTransportConfig(env({ COMPACTOR_URL: 'http://127.0.0.1:8080/' })).baseUrl, 'http://127.0.0.1:8080');
	assert.equal(loadTransportConfig(env({ COMPACTOR_URL: 'http://127.0.0.1:8080///' })).baseUrl, 'http://127.0.0.1:8080');
	assert.equal(loadTransportConfig(env({ COMPACTOR_URL: 'http://127.0.0.1:8080' })).baseUrl, 'http://127.0.0.1:8080');
});

test('loadTransportConfig: apiKey is undefined when unset, and present when set', () => {
	assert.equal(loadTransportConfig(env()).apiKey, undefined);
	assert.equal(loadTransportConfig(env({ COMPACTOR_API_KEY: 'secret' })).apiKey, 'secret');
	// An empty string is falsy — `||` in the source treats it as unset,
	// matching "always send Authorization: Bearer when a key is
	// configured" (an empty key is not a configured key).
	assert.equal(loadTransportConfig(env({ COMPACTOR_API_KEY: '' })).apiKey, undefined);
});

// ---------------------------------------------------------------------------
// timeoutMs
// ---------------------------------------------------------------------------

test('loadTransportConfig: timeoutMs defaults to DEFAULT_TIMEOUT_MS when unset', () => {
	assert.equal(loadTransportConfig(env()).timeoutMs, DEFAULT_TIMEOUT_MS);
	assert.equal(loadTransportConfig(env({ COMPACTOR_CLIENT_TIMEOUT_MS: '' })).timeoutMs, DEFAULT_TIMEOUT_MS);
});

test('loadTransportConfig: timeoutMs reads a positive number from COMPACTOR_CLIENT_TIMEOUT_MS, fractional included', () => {
	assert.equal(loadTransportConfig(env({ COMPACTOR_CLIENT_TIMEOUT_MS: '60000' })).timeoutMs, 60_000);
	// The exact value the mutation audit named as unguarded — verified
	// here, on BOTH sides: it is accepted (fractional ms is legal, if
	// unusual), and the validation branch that would reject a genuinely
	// bad value is exercised by the negative tests below.
	assert.equal(loadTransportConfig(env({ COMPACTOR_CLIENT_TIMEOUT_MS: '0.5' })).timeoutMs, 0.5);
});

test('loadTransportConfig: timeoutMs rejects zero, negative, and non-numeric values loudly', () => {
	assert.throws(() => loadTransportConfig(env({ COMPACTOR_CLIENT_TIMEOUT_MS: '0' })), /positive number/);
	assert.throws(() => loadTransportConfig(env({ COMPACTOR_CLIENT_TIMEOUT_MS: '-100' })), /positive number/);
	assert.throws(() => loadTransportConfig(env({ COMPACTOR_CLIENT_TIMEOUT_MS: 'not-a-number' })), /positive number/);
	assert.throws(() => loadTransportConfig(env({ COMPACTOR_CLIENT_TIMEOUT_MS: 'Infinity' })), /positive number/);
});

// ---------------------------------------------------------------------------
// windowN — D11/D2's "behind a runtime switch"
// ---------------------------------------------------------------------------

test('loadTransportConfig: windowN defaults to DEFAULT_WINDOW_N when unset', () => {
	assert.equal(loadTransportConfig(env()).windowN, DEFAULT_WINDOW_N);
	assert.equal(loadTransportConfig(env({ COMPACTOR_WINDOW_N: '' })).windowN, DEFAULT_WINDOW_N);
});

test('loadTransportConfig: windowN reads a valid override — D2\'s "runtime switch" made real', () => {
	assert.equal(loadTransportConfig(env({ COMPACTOR_WINDOW_N: '80' })).windowN, 80);
	assert.equal(loadTransportConfig(env({ COMPACTOR_WINDOW_N: String(MAX_WINDOW_N) })).windowN, MAX_WINDOW_N);
});

test('loadTransportConfig: windowN rejects zero, negative, non-integer, and anything over MAX_WINDOW_N — D11', () => {
	assert.throws(() => loadTransportConfig(env({ COMPACTOR_WINDOW_N: '0' })));
	assert.throws(() => loadTransportConfig(env({ COMPACTOR_WINDOW_N: '-10' })));
	assert.throws(() => loadTransportConfig(env({ COMPACTOR_WINDOW_N: '60.5' })), /integer/);
	assert.throws(
		() => loadTransportConfig(env({ COMPACTOR_WINDOW_N: String(MAX_WINDOW_N + 1) })),
		/exceeds MAX_WINDOW_N/
	);
	// The literal D11 defect: a config that would let `n` reach anywhere
	// near "the whole conversation" must be rejected at LOAD time, not
	// silently accepted and only misbehave three requests later.
	assert.throws(() => loadTransportConfig(env({ COMPACTOR_WINDOW_N: '100000' })), /exceeds MAX_WINDOW_N/);
});

test('loadTransportConfig: windowN and timeoutMs validation are independent — one bad value does not mask the other', () => {
	assert.throws(
		() => loadTransportConfig(env({ COMPACTOR_WINDOW_N: '100000', COMPACTOR_CLIENT_TIMEOUT_MS: '5000' })),
		/exceeds MAX_WINDOW_N/
	);
	assert.doesNotThrow(() =>
		loadTransportConfig(env({ COMPACTOR_WINDOW_N: '80', COMPACTOR_CLIENT_TIMEOUT_MS: '5000' }))
	);
});
