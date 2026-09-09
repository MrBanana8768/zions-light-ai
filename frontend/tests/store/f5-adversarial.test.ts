// F5 — the adversarial chain suite. FRONTEND_PLAN.md §3.1 / FRONTEND_SPEC.md
// §4.1 & §13: "multi-root, missing parent, deep-vs-current divergence, mixed
// key spaces, mid-chain failure, and the standing 241/5/8 case."
//
// §4.1 names FIVE properties; FOUR are store properties, tested here:
//   1. roots == 1
//   2. current_leaf reachable from that root
//   4. every parent_id resolves in the same key space as the id it names
//   5. no message unreachable from the root
// Property 3 ("the chain from current_leaf contains every message the
// thread renders") is explicitly client-side per FRONTEND_PLAN.md §3.1's
// own table ("cannot be inherited from the DDL") — it is NOT retested here;
// it belongs to whichever lane owns the render path.
//
// Every fixture that must defeat a constraint to exist proves, INLINE and
// BEFORE dropping anything, that the constraint would otherwise have
// refused the exact insert being planted. That assertion — not the
// contrived shape itself — is the actual claim each such test makes.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTestDb } from './helpers/testdb.js';
import { uuidv7 } from '../../src/lib/server/store/uuid7.js';
import { dropConstraint, dropIndex, insertChain, setCurrentLeaf, setCurrentLeafNull } from './fixtures/adversarial.js';

test('F5: multi-root — a second independent root fails the audit via `roots` alone, nothing else', async () => {
	const db = await createTestDb({ prefix: 'f5multiroot' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"hi"'
		});

		// Prove the guard BEFORE touching it.
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,NULL,$3,'system','"root2"'::jsonb)`,
					[uuidv7(), conversation.id, userId]
				),
			/message_one_root_per_conv/
		);

		await dropIndex(db, 'message_one_root_per_conv');
		await insertChain(db, conversation.id, userId, 1);

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.roots, 2);
		assert.equal(audit!.missingParent, 0);
		assert.equal(audit!.reachableN, audit!.total, 'both trees are internally intact — nothing is orphaned');
		assert.equal(audit!.leafOnTree, true, 'the ORIGINAL leaf is untouched by the second root existing');
		assert.equal(audit!.pass, false, 'roots=2 alone must fail, even with every other property intact');
	} finally {
		await db.close();
	}
});

test('F5: missing parent — a dangling parent_id is caught by missing_parent and reachable_n < total; the FK would have refused it', async () => {
	const db = await createTestDb({ prefix: 'f5missingparent' });
	try {
		const userId = uuidv7();
		const { conversation } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const bogusParent = uuidv7();

		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,$3,$4,'user','"x"'::jsonb)`,
					[uuidv7(), conversation.id, bogusParent, userId]
				),
			/message_parent_fk/
		);

		await dropConstraint(db, 'message', 'message_parent_fk');
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,$3,$4,'user','"orphan"'::jsonb)`,
			[uuidv7(), conversation.id, bogusParent, userId]
		);

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.total, 2); // the synthetic root, plus the orphan
		assert.equal(audit!.roots, 1); // the real root is unaffected
		assert.equal(audit!.missingParent, 1);
		assert.equal(audit!.reachableN, 1, 'only the root is reachable from a root — the orphan is not');
		assert.equal(audit!.pass, false);
	} finally {
		await db.close();
	}
});

test('F5: deep-vs-current divergence — a long branch elsewhere never taints `pass`; the forbidden checks stay out of the boolean', async () => {
	const db = await createTestDb({ prefix: 'f5divergence' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });

		// A 30-deep primary branch off the root, branching off early (at
		// depth 5) into a short 3-message side chain that becomes current.
		let rev = conversation.rev;
		let parent = root.id;
		let branchPoint = root.id;
		for (let i = 0; i < 29; i++) {
			const { message, rev: newRev } = await db.store.appendMessage({
				convId: conversation.id,
				parentId: parent,
				expectedRev: rev,
				userId,
				role: i % 2 === 0 ? 'user' : 'assistant',
				content: JSON.stringify(`deep-${i}`)
			});
			parent = message.id;
			rev = newRev;
			if (i === 3) {
				branchPoint = message.id; // depth 5 (root=1 .. here=5)
			}
		}

		const { rev: revAfterSelect } = await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: branchPoint,
			expectedRev: rev
		});
		let shortRev = revAfterSelect;
		let shortParent = branchPoint;
		for (let i = 0; i < 3; i++) {
			const { message, rev: newRev } = await db.store.appendMessage({
				convId: conversation.id,
				parentId: shortParent,
				expectedRev: shortRev,
				userId,
				role: 'user',
				content: JSON.stringify(`short-${i}`)
			});
			shortParent = message.id;
			shortRev = newRev;
		}

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.roots, 1);
		assert.equal(audit!.missingParent, 0);
		assert.equal(audit!.total, 33); // root(1) + shared prefix(4) + long tail(25) + short tail(3)
		assert.equal(audit!.reachableN, audit!.total);
		assert.equal(audit!.leafOnTree, true);
		assert.equal(audit!.chainFromCurrent, 8); // branchPoint(depth5) + 3 more
		assert.equal(audit!.deepest, 30);
		assert.ok(audit!.deepest > audit!.chainFromCurrent);
		assert.equal(
			audit!.pass,
			true,
			'a legitimate branch selection must PASS even though deepest (30) >> chain_from_current (8) — ' +
				'depth is diagnostic only and must never gate `pass` (FRONTEND_SPEC.md §4.1 forbidden checks)'
		);
	} finally {
		await db.close();
	}
});

test('F5: mixed key spaces — a composite {conv_id}-{uuid} parent_id is rejected by the COLUMN TYPE itself, not merely a constraint', async () => {
	const db = await createTestDb({ prefix: 'f5mixedkeys' });
	try {
		const userId = uuidv7();
		const { conversation } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		// OpenWebUI's own chat_message shape: a composite {chat_id}-{uuid}
		// primary key against a bare-uuid parent_id (FRONTEND_SPEC.md
		// §11.2). This schema uses `uuid` columns throughout, so this is not
		// even an FK question — the string never survives the column's own
		// input parser.
		const compositeParent = `${conversation.id}-${uuidv7()}`;
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,$3,$4,'user','"x"'::jsonb)`,
					[uuidv7(), conversation.id, compositeParent, userId]
				),
			/invalid input syntax for type uuid/i
		);
	} finally {
		await db.close();
	}
});

test('F5: mid-chain failure — a failed CAS leaves no partial row, and a failed-state message does not corrupt the tree', async () => {
	const db = await createTestDb({ prefix: 'f5midchain' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		const { message: m1, rev: rev1 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"hi"'
		});

		// Exactly the 2026-08-24 mechanism at the store layer: a second
		// writer's INSERT half of the transaction runs against a STALE
		// parent pointer (it still names the OLD leaf, root, not m1); the
		// CAS half must then fail and the WHOLE transaction must roll back,
		// leaving zero trace — not a dangling row that "exists" but was
		// never linked in.
		await assert.rejects(
			() =>
				db.store.appendMessage({
					convId: conversation.id,
					parentId: root.id, // STALE — current leaf is m1, not root
					expectedRev: rev1, // rev IS current; only the parent is stale
					userId,
					role: 'user',
					content: '"lost race"'
				}),
			/StaleWriteError/
		);
		const auditAfterFailedCas = await db.store.auditConversation(conversation.id);
		assert.ok(auditAfterFailedCas);
		assert.equal(
			auditAfterFailedCas!.total,
			2,
			'the failed transaction (INSERT succeeded, then the CAS UPDATE failed) must not have left a partial row behind'
		);
		assert.equal(auditAfterFailedCas!.pass, true);

		// A REAL failure — vLLM died mid-stream — is a different, legitimate
		// path: the row IS kept, marked failed, IN PLACE. That must not
		// corrupt the tree, and a retry (a sibling under the same parent,
		// reached by selectLeaf back to it) must succeed normally.
		const { message: m2, rev: rev2 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: m1.id,
			expectedRev: rev1,
			userId,
			role: 'assistant',
			content: '"partial reply cut off by a dead backend"',
			state: 'streaming'
		});
		await db.store.updateMessageState({
			convId: conversation.id,
			messageId: m2.id,
			state: 'failed',
			error: '{"reason":"backend_unavailable"}'
		});

		const { rev: rev3 } = await db.store.selectLeaf({
			convId: conversation.id,
			targetMessageId: m1.id,
			expectedRev: rev2
		});
		const { message: m3 } = await db.store.appendMessage({
			convId: conversation.id,
			parentId: m1.id,
			expectedRev: rev3,
			userId,
			role: 'assistant',
			content: '"retried reply"'
		});
		assert.notEqual(m3.id, m2.id);

		const audit2 = await db.store.auditConversation(conversation.id);
		assert.ok(audit2);
		assert.equal(audit2!.total, 4); // root, m1, m2(failed, still present as a sibling), m3
		assert.equal(audit2!.roots, 1);
		assert.equal(audit2!.reachableN, 4);
		assert.equal(
			audit2!.pass,
			true,
			'a failed-state message sitting in the middle of the tree (a rejected sibling) must not fail the audit'
		);
	} finally {
		await db.close();
	}
});

test('F5: a leaf inside a wholly unreachable (rootless) subtree fails leafOnTree — existence is not reachability', async () => {
	const db = await createTestDb({ prefix: 'f5unreachableleaf' });
	try {
		const userId = uuidv7();
		const { conversation } = await db.store.createConversation({ userId, systemContent: '"sys"' });

		// A two-message "phantom" chain never connected to any root: B's
		// parent is a uuid that exists nowhere, and A's parent is B.
		const danglingParent = uuidv7();
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,$3,$4,'user','"b"'::jsonb)`,
					[uuidv7(), conversation.id, danglingParent, userId]
				),
			/message_parent_fk/
		);
		await dropConstraint(db, 'message', 'message_parent_fk');

		const bId = uuidv7();
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,$3,$4,'user','"b"'::jsonb)`,
			[bId, conversation.id, danglingParent, userId]
		);
		const aId = uuidv7();
		await db.pool.query(
			`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
			 VALUES ($1,$2,$3,$4,'user','"a"'::jsonb)`,
			[aId, conversation.id, bId, userId]
		);
		await setCurrentLeaf(db, conversation.id, aId);

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.total, 3); // root, a, b
		assert.equal(audit!.roots, 1); // the real synthetic root, untouched
		assert.equal(audit!.missingParent, 1); // b's parent does not exist
		assert.equal(audit!.reachableN, 1, 'only the real root is reachable from any root; a/b never link to one');
		assert.equal(
			audit!.leafOnTree,
			false,
			'the leaf ROW exists in `message` (a/b are real rows) but is unreachable from any root — ' +
				'property 2 is reachability, not row existence'
		);
		assert.equal(audit!.pass, false);
	} finally {
		await db.close();
	}
});

test('F5 (chain_audit): the standing case — 241 messages, 5 roots, current leaf at depth 8 — audit_conversation MUST return pass = false', async () => {
	const db = await createTestDb({ prefix: 'f5standing' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });

		// Prove the guard first, self-contained: this exact fixture is
		// possible ONLY because message_one_root_per_conv is about to be
		// dropped.
		await assert.rejects(
			() =>
				db.pool.query(
					`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
					 VALUES ($1,$2,NULL,$3,'system','"root2"'::jsonb)`,
					[uuidv7(), conversation.id, userId]
				),
			/message_one_root_per_conv/
		);
		await dropIndex(db, 'message_one_root_per_conv');

		// Tree #1: the conversation's OWN synthetic root, extended to depth
		// 208 — the exact number the real incident's own diagnostic read as
		// "healthy" (FRONTEND_SPEC.md: "deepest=208 ... true and
		// irrelevant"). 207 more nodes on top of the existing root = 208.
		let parent = root.id;
		for (let i = 1; i < 208; i++) {
			const id = uuidv7();
			await db.pool.query(
				`INSERT INTO message (id, conv_id, parent_id, user_id, role, content)
				 VALUES ($1,$2,$3,$4,'user',$5::jsonb)`,
				[id, conversation.id, parent, userId, JSON.stringify(`t1-${i}`)]
			);
			parent = id;
		}

		// Trees #2..#5: four more independent roots. Tree #2 is the CURRENT
		// branch — exactly 8 nodes deep, matching "current leaf at depth 8".
		// 208 + 8 + 10 + 8 + 7 = 241.
		const tree2 = await insertChain(db, conversation.id, userId, 8);
		await insertChain(db, conversation.id, userId, 10);
		await insertChain(db, conversation.id, userId, 8);
		await insertChain(db, conversation.id, userId, 7);

		await setCurrentLeaf(db, conversation.id, tree2.leafId);

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.total, 241, 'total must be exactly 241 — the real incident size');
		assert.equal(audit!.roots, 5);
		assert.equal(audit!.chainFromCurrent, 8);
		assert.equal(audit!.deepest, 208);
		assert.equal(audit!.missingParent, 0, 'every parent link that exists is intra-tree valid');
		assert.equal(audit!.reachableN, 241, 'every node is reachable from SOME root — nothing is orphaned');
		assert.equal(audit!.leafOnTree, true, 'the leaf itself is fine — the defect is purely structural (roots)');
		assert.equal(
			audit!.pass,
			false,
			'roots=5 alone must fail the audit — this is the exact property that would have made ' +
				'2026-08-24 structurally impossible rather than merely detected after the fact'
		);
	} finally {
		await db.close();
	}
});

// Gate remediation F2: `leaf_on_tree` was untestable-by-omission — deleting
// that conjunct from the `pass` formula (0001_init.sql) leaves the whole
// suite green, because every OTHER pass=false fixture above also fails via
// `roots` or `reachable_n`. This fixture isolates it: NULL is a legal value
// for current_leaf_id per the column's own definition (nullable, no CHECK,
// and conversation_leaf_fk is MATCH SIMPLE — a composite FK with any NULL
// column is vacuously satisfied) — the migration's own comment
// (0001_init.sql:28-29) already says a null leaf must read as
// leaf_on_tree=false, but nothing exercised that until now.
test('F2: current_leaf_id = NULL isolates leaf_on_tree as the ONLY failing signal', async () => {
	const db = await createTestDb({ prefix: 'f2leafnull' });
	try {
		const userId = uuidv7();
		const { conversation, root } = await db.store.createConversation({ userId, systemContent: '"sys"' });
		await db.store.appendMessage({
			convId: conversation.id,
			parentId: root.id,
			expectedRev: conversation.rev,
			userId,
			role: 'user',
			content: '"hi"'
		});

		await setCurrentLeafNull(db, conversation.id);

		const audit = await db.store.auditConversation(conversation.id);
		assert.ok(audit);
		assert.equal(audit!.roots, 1);
		assert.equal(audit!.missingParent, 0);
		assert.equal(audit!.reachableN, audit!.total, 'the tree itself is perfectly intact');
		assert.equal(audit!.leaf, null);
		assert.equal(audit!.leafOnTree, false, 'a NULL leaf must read as leaf_on_tree = false');
		assert.equal(
			audit!.pass,
			false,
			'roots=1, missing_parent=0 and reachable_n=total all hold — pass must fail SOLELY because leaf_on_tree is false'
		);
	} finally {
		await db.close();
	}
});
