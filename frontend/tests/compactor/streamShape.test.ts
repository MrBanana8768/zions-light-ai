// Gate remediation D7 (docs/lanes/L2-gate-findings.md). `main.py:5469-5483`:
// on a mid-reply `httpx.RequestError` the compactor yields N relay chunks of
// genuine prose, THEN `chatcmpl-unavail-` chunks, THEN `[DONE]` — its own
// comment confirms this fires "when vLLM drops the connection PART WAY
// THROUGH a reply she has already read." `classifyChunk` is per-chunk and
// stateless; the obvious caller (accumulate delta.content, mark complete on
// the first terminal chunk) would weld the outage apology onto the
// truncated real reply into ONE assistant message. `reduceStreamShape`
// (sse.ts) is the stream-level signal that lets a caller avoid that.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { reduceStreamShape } from '../../src/lib/server/compactor/sse.js';
import type { SseParsedEvent } from '../../src/lib/server/compactor/sse.js';

function relayEvent(text: string, terminal = false): SseParsedEvent {
	return {
		isDone: false,
		raw: '',
		classified: {
			kind: 'relay',
			chunk: {
				id: 'chatcmpl-real-id',
				choices: [{ index: 0, delta: { content: text }, finish_reason: terminal ? 'stop' : null }]
			},
			terminal
		}
	};
}

function unavailableEvent(): SseParsedEvent {
	return {
		isDone: false,
		raw: '',
		classified: {
			kind: 'unavailable',
			chunk: {
				id: 'chatcmpl-unavail-abc123',
				choices: [{ index: 0, delta: { content: '⚠️ connection lost' }, finish_reason: 'stop' }]
			},
			terminal: true
		}
	};
}

function doneEvent(): SseParsedEvent {
	return { isDone: true, raw: '[DONE]' };
}

// ---------------------------------------------------------------------------
// The exact mixed stream the brief names: real prose, then
// chatcmpl-unavail-, then [DONE].
// ---------------------------------------------------------------------------

test('reduceStreamShape: a mixed stream — real prose, then chatcmpl-unavail-, then [DONE]', async () => {
	const events: SseParsedEvent[] = [
		relayEvent('Once '),
		relayEvent('upon '),
		relayEvent('a time'),
		unavailableEvent(),
		doneEvent()
	];

	const summary = await reduceStreamShape(events);
	assert.equal(summary.firstNonRelayKind, 'unavailable');
	assert.equal(summary.relayContentStoppedAt, 3, 'exactly the 3 genuine relay chunks before the shape changed');
	assert.equal(summary.relayText, 'Once upon a time', 'the genuine prose, and ONLY the genuine prose');
	assert.equal(summary.endedWithoutTerminal, false, 'the unavailable chunk WAS terminal, and [DONE] also arrived');
	// The literal defect this exists to prevent: nothing here ever
	// concatenates the unavailable shape's own apology text onto relayText.
	assert.ok(!summary.relayText.includes('connection lost'));
});

test('reduceStreamShape: a purely healthy relay stream — firstNonRelayKind stays null, relayContentStoppedAt counts everything', async () => {
	const events: SseParsedEvent[] = [relayEvent('hi'), relayEvent(' there', true), doneEvent()];
	const summary = await reduceStreamShape(events);
	assert.equal(summary.firstNonRelayKind, null);
	assert.equal(summary.relayContentStoppedAt, 2);
	assert.equal(summary.relayText, 'hi there');
	assert.equal(summary.endedWithoutTerminal, false);
});

// ---------------------------------------------------------------------------
// The related, explicit "ended without a terminal chunk" signal
// ---------------------------------------------------------------------------

test('reduceStreamShape: a stream that ends with no terminal chunk and no [DONE] is flagged, not silently read as clean', async () => {
	// transport.ts's own flush-on-exit path: the connection just closes
	// mid-reply, with no finish_reason ever set and no [DONE] sentinel —
	// indistinguishable from a clean end unless this is checked explicitly.
	const events: SseParsedEvent[] = [relayEvent('partial'), relayEvent(' reply')];
	const summary = await reduceStreamShape(events);
	assert.equal(summary.endedWithoutTerminal, true);
	assert.equal(summary.firstNonRelayKind, null);
	assert.equal(summary.relayContentStoppedAt, 2);
});

test('reduceStreamShape: a slash-command stream terminates cleanly (finish_reason "stop") without ever being relay', async () => {
	const cmdEvent: SseParsedEvent = {
		isDone: false,
		raw: '',
		classified: {
			kind: 'slash_command',
			chunk: { id: 'chatcmpl-cmd-abc', choices: [{ index: 0, delta: { content: 'Pinned.' }, finish_reason: 'stop' }] },
			terminal: true
		}
	};
	const summary = await reduceStreamShape([cmdEvent, doneEvent()]);
	assert.equal(summary.firstNonRelayKind, 'slash_command');
	assert.equal(summary.relayContentStoppedAt, 0, 'not a single relay chunk arrived before the shape changed');
	assert.equal(summary.relayText, '');
	assert.equal(summary.endedWithoutTerminal, false);
});

test('reduceStreamShape: a rejected (4xx) stream carries an error object and is reported as the first non-relay kind', async () => {
	const rejectedEvent: SseParsedEvent = {
		isDone: false,
		raw: '',
		classified: {
			kind: 'rejected',
			chunk: {
				id: 'chatcmpl-rejected-abc',
				choices: [{ index: 0, delta: {}, finish_reason: 'error' }],
				error: { message: 'too long', code: 'context_length_exceeded' }
			},
			terminal: true,
			error: { message: 'too long', code: 'context_length_exceeded' }
		}
	};
	const summary = await reduceStreamShape([relayEvent('hello'), rejectedEvent]);
	assert.equal(summary.firstNonRelayKind, 'rejected');
	assert.equal(summary.relayContentStoppedAt, 1);
	assert.equal(summary.relayText, 'hello');
});

test('reduceStreamShape: a parse-error event (classified: null) is skipped, never advancing or resetting anything', async () => {
	const parseErrorEvent: SseParsedEvent = { isDone: false, raw: 'not json', classified: null, parseError: 'boom' };
	const summary = await reduceStreamShape([relayEvent('a'), parseErrorEvent, relayEvent('b'), doneEvent()]);
	assert.equal(summary.firstNonRelayKind, null);
	assert.equal(summary.relayContentStoppedAt, 2);
	assert.equal(summary.relayText, 'ab');
});

test('reduceStreamShape: content AFTER the shape changed is never appended to relayText, even if a caller kept classifying it as relay by mistake', async () => {
	// A relay chunk appearing (adversarially) AFTER an unavailable chunk —
	// reduceStreamShape must not resume accumulating once the shape has
	// changed once.
	const events: SseParsedEvent[] = [relayEvent('real'), unavailableEvent(), relayEvent('should not count'), doneEvent()];
	const summary = await reduceStreamShape(events);
	assert.equal(summary.relayContentStoppedAt, 1);
	assert.equal(summary.relayText, 'real');
	assert.equal(summary.firstNonRelayKind, 'unavailable');
});

test('reduceStreamShape: accepts an async iterable too, not only a plain array', async () => {
	async function* gen(): AsyncGenerator<SseParsedEvent> {
		yield relayEvent('async');
		yield relayEvent(' works', true);
		yield doneEvent();
	}
	const summary = await reduceStreamShape(gen());
	assert.equal(summary.relayText, 'async works');
	assert.equal(summary.endedWithoutTerminal, false);
});
