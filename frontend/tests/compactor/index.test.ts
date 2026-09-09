// index.ts is the barrel export — "the whole public surface a route
// handler needs" (index.ts's own header comment). The mutation audit's
// finding: "index.ts is imported by no test, so a missing re-export in
// [that surface] is invisible." This file imports it via `import * as`
// (never a named import — a named import of something missing would be a
// COMPILE error, failing the whole suite's build rather than this one
// test, and would defeat the mutation-testing methodology) and asserts
// every symbol a route handler actually needs is really there, by runtime
// shape, not by reading the source and assuming.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as CompactorIndex from '../../src/lib/server/compactor/index.js';

test('public surface: F8 gate — DEFAULT_WINDOW_N, MAX_WINDOW_N, assertValidWindowN, computeWindowIntent, verifyChainShape, runGate, applyOverride', () => {
	assert.equal(typeof CompactorIndex.DEFAULT_WINDOW_N, 'number');
	assert.equal(typeof CompactorIndex.MAX_WINDOW_N, 'number');
	assert.equal(typeof CompactorIndex.assertValidWindowN, 'function');
	assert.equal(typeof CompactorIndex.computeWindowIntent, 'function');
	assert.equal(typeof CompactorIndex.verifyChainShape, 'function');
	assert.equal(typeof CompactorIndex.runGate, 'function');
	assert.equal(typeof CompactorIndex.applyOverride, 'function');
});

test('public surface: the byte-stability fingerprint contract', () => {
	assert.equal(typeof CompactorIndex.messageText, 'function');
	assert.equal(typeof CompactorIndex.imageOnlyMarker, 'function');
	assert.equal(typeof CompactorIndex.utf8SurrogatepassEncode, 'function');
	assert.equal(typeof CompactorIndex.sha256Hex16, 'function');
	assert.equal(typeof CompactorIndex.turnFingerprint, 'function');
	assert.equal(typeof CompactorIndex.turnFingerprints, 'function');
	assert.equal(typeof CompactorIndex.trailingAnchorFingerprints, 'function');
	assert.equal(typeof CompactorIndex.ANCHOR_TURNS, 'number');
});

test('public surface: the request builder, including D9\'s sanitizeConvId', () => {
	assert.equal(typeof CompactorIndex.mintConvId, 'function');
	assert.equal(typeof CompactorIndex.assertSendableConvId, 'function');
	assert.equal(typeof CompactorIndex.sanitizeConvId, 'function');
	assert.equal(typeof CompactorIndex.wouldSanitizeToEmpty, 'function');
	assert.equal(typeof CompactorIndex.buildRequestHeaders, 'function');
	assert.equal(typeof CompactorIndex.buildChatCompletionBody, 'function');
});

test('public surface: SSE classification, including D7\'s reduceStreamShape', () => {
	assert.equal(typeof CompactorIndex.STREAM_ID_PREFIX, 'object');
	assert.equal(typeof CompactorIndex.classifyChunkId, 'function');
	assert.equal(typeof CompactorIndex.classifyChunk, 'function');
	assert.equal(typeof CompactorIndex.splitSseBlocks, 'function');
	assert.equal(typeof CompactorIndex.extractDataPayload, 'function');
	assert.equal(typeof CompactorIndex.parseSseEvent, 'function');
	assert.equal(typeof CompactorIndex.parseErrorEnvelope, 'function');
	assert.equal(typeof CompactorIndex.reduceStreamShape, 'function');
});

test('public surface: config — including D11\'s window-size validation surface', () => {
	assert.equal(typeof CompactorIndex.loadTransportConfig, 'function');
	assert.equal(typeof CompactorIndex.DEFAULT_TIMEOUT_MS, 'number');
});

test('public surface: transport — streamChatCompletion and its two typed errors', () => {
	assert.equal(typeof CompactorIndex.streamChatCompletion, 'function');
	assert.equal(typeof CompactorIndex.UpstreamHttpError, 'function');
	assert.equal(typeof CompactorIndex.GenerationTimeoutError, 'function');
});

test('public surface: the receipt', () => {
	assert.equal(typeof CompactorIndex.buildReceipt, 'function');
	assert.equal(typeof CompactorIndex.readMessagesAdmitted, 'function');
	assert.equal(typeof CompactorIndex.PROPOSED_ADMITTED_HEADER, 'string');
});
