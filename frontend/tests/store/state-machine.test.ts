// Gate remediation F5. FRONTEND_SPEC.md:617-619: "never last-write-wins on
// a blob ... it must never overwrite. It loses its own write, loudly."
// updateMessageState previously took no expected-state precondition, so a
// lagging second writer could flip an already-`complete` message back to
// `streaming` and truncate its content — silently. Fixed with a monotonic
// state machine (pending -> streaming -> complete|failed, terminal states
// frozen) enforced by a BEFORE UPDATE trigger, surfaced to callers as a
// typed IllegalStateTransitionError. Both directions are tested: legal
// forward transitions succeed, and the specific "flip a complete message
// back to streaming" attack is rejected loudly rather than silently.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';

async function seedStreamingMessage(db: Awaited<ReturnType<typeof createTestDb>>) {
	const userId = uuidv7();
	const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
	const { message } = await db.store.appendMessage({
		convId: conversation.id,
		parentId: root.id,
		expectedRev: conversation.rev,
		userId,
		role: 'assistant',
		content: '""',
		state: 'pending'
	});
	return { userId, conversation, root, message };
}

test('state machine: the legal forward path pending -> streaming -> complete succeeds', async () => {
	const db = await createTestDb({ prefix: 'smforward' });
	try {
		const { conversation, message } = await seedStreamingMessage(db);
		const afterStreaming = await db.store.updateMessageState({
			convId: conversation.id,
			messageId: message.id,
			state: 'streaming',
			content: '"partial"'
		});
		assert.equal(afterStreaming.state, 'streaming');

		// Repeated same-stage checkpoints while streaming are legal.
		await db.store.updateMessageState({
			convId: conversation.id,
			messageId: message.id,
			state: 'streaming',
			content: '"more partial"'
		});

		const afterComplete = await db.store.updateMessageState({
			convId: conversation.id,
			messageId: message.id,
			state: 'complete',
			content: '"final"'
		});
		assert.equal(afterComplete.state, 'complete');
		assert.equal(afterComplete.content, '"final"');
	} finally {
		await db.close();
	}
});

test('state machine: pending -> complete directly (no intermediate streaming) succeeds', async () => {
	const db = await createTestDb({ prefix: 'smdirect' });
	try {
		const { conversation, message } = await seedStreamingMessage(db);
		const after = await db.store.updateMessageState({
			convId: conversation.id,
			messageId: message.id,
			state: 'complete',
			content: '"one shot"'
		});
		assert.equal(after.state, 'complete');
	} finally {
		await db.close();
	}
});

test('state machine: pending -> failed and streaming -> failed both succeed', async () => {
	const db = await createTestDb({ prefix: 'smfailed' });
	try {
		const { conversation, message } = await seedStreamingMessage(db);
		const afterFailedDirect = await db.store.updateMessageState({
			convId: conversation.id,
			messageId: message.id,
			state: 'failed',
			error: '{"reason":"backend_unavailable"}'
		});
		assert.equal(afterFailedDirect.state, 'failed');

		const { message: m2, rev } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: message.id,
			expectedRev: (await db.store.getConversation(conversation.id))!.rev,
			userId: (await db.store.getConversation(conversation.id))!.userId,
			role: 'assistant',
			content: '""',
			state: 'streaming'
		});
		void rev;
		const afterFailedFromStreaming = await db.store.updateMessageState({
			convId: conversation.id,
			messageId: m2.id,
			state: 'failed',
			error: '{"reason":"stopped"}'
		});
		assert.equal(afterFailedFromStreaming.state, 'failed');
	} finally {
		await db.close();
	}
});

// --- The attack this fix exists to stop ---------------------------------

test('state machine: a lagging writer cannot flip a COMPLETE message back to streaming (IllegalStateTransitionError), and content is untouched', async () => {
	const db = await createTestDb({ prefix: 'smattack' });
	try {
		const { conversation, message } = await seedStreamingMessage(db);
		await db.store.updateMessageState({
			convId: conversation.id,
			messageId: message.id,
			state: 'complete',
			content: '"the real, finished reply"'
		});

		// The exact attack named by the spec: a second, lagging writer
		// tries to resume "streaming" against an already-complete message
		// and truncate its content.
		await assert.rejects(
			() =>
				db.store.updateMessageState({
					convId: conversation.id,
					messageId: message.id,
					state: 'streaming',
					content: '"truncated"'
				}),
			/IllegalStateTransitionError/
		);

		const { rows } = await db.pool.query('SELECT state, content FROM message WHERE id = $1', [message.id]);
		assert.equal(rows[0].state, 'complete');
		assert.equal(rows[0].content, '"the real, finished reply"', 'content must be UNCHANGED, not truncated');
	} finally {
		await db.close();
	}
});

test('state machine: complete and failed cannot cross into each other, and a terminal message cannot be content-edited', async () => {
	const db = await createTestDb({ prefix: 'smterminalcross' });
	try {
		const { conversation, message } = await seedStreamingMessage(db);
		await db.store.updateMessageState({
			convId: conversation.id,
			messageId: message.id,
			state: 'complete',
			content: '"done"'
		});

		await assert.rejects(
			() => db.store.updateMessageState({ convId: conversation.id, messageId: message.id, state: 'failed' }),
			/IllegalStateTransitionError/
		);

		// Same state, but a silent content edit — also frozen.
		await assert.rejects(
			() =>
				db.store.updateMessageState({
					convId: conversation.id,
					messageId: message.id,
					state: 'complete',
					content: '"edited after the fact"'
				}),
			/IllegalStateTransitionError/
		);

		const { rows } = await db.pool.query('SELECT state, content FROM message WHERE id = $1', [message.id]);
		assert.equal(rows[0].state, 'complete');
		assert.equal(rows[0].content, '"done"');
	} finally {
		await db.close();
	}
});

test('state machine: a backward transition (streaming -> pending) is rejected', async () => {
	const db = await createTestDb({ prefix: 'smbackward' });
	try {
		const { conversation, message } = await seedStreamingMessage(db);
		await db.store.updateMessageState({
			convId: conversation.id,
			messageId: message.id,
			state: 'streaming',
			content: '"partial"'
		});
		await assert.rejects(
			() => db.store.updateMessageState({ convId: conversation.id, messageId: message.id, state: 'pending' }),
			/IllegalStateTransitionError/
		);
	} finally {
		await db.close();
	}
});
