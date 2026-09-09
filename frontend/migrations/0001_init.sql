-- L1 store — FRONTEND_PLAN.md §3.1 / FRONTEND_SPEC.md §11.
--
-- SCHEMA-AGNOSTIC ON PURPOSE. Every name below is unqualified: this file is
-- applied with `search_path` already pointed at the target schema by the
-- caller (frontend/src/lib/server/store/migrate.ts). Production points it at
-- `client`, inside the existing `openwebui` database (FRONTEND_PLAN.md §3.1 —
-- a separate database would live only on the ephemeral overlay and be
-- silently un-archived by pgarchive.py, which is scoped to one database).
-- Tests point it at a fresh, throwaway schema per run so adversarial
-- fixtures that must defeat a constraint (see docs/lanes/L1-store.md, F5) can
-- do so without touching any other schema. Do not hardcode a schema name
-- anywhere in this file or in application SQL — see db.ts.
--
-- `message` is FRONTEND_SPEC.md §11.2's DDL, verbatim (every column,
-- constraint and index name unchanged). §11.2 never defines `conversation`,
-- `rev`, or `user_id` — FRONTEND_PLAN.md §3.1 assigns writing those to this
-- lane. See docs/lanes/L1-store.md for the reasoning behind every
-- constraint below.

CREATE TABLE conversation (
    id               uuid        PRIMARY KEY,
    user_id          uuid        NOT NULL,
    title            text,
    -- Nullable only for the instant between the conversation row's own
    -- INSERT and the synthetic root's INSERT inside create_conversation's
    -- transaction; conversation_leaf_fk (below, DEFERRABLE INITIALLY
    -- DEFERRED) makes that instant legal by only checking at COMMIT. No
    -- other code path may leave it null — audit_conversation treats a null
    -- leaf as leaf_on_tree = false, i.e. a FAIL.
    current_leaf_id  uuid,
    -- Optimistic concurrency for the leaf pointer only (§11.3). Bumped by
    -- every CAS that moves current_leaf_id: append_message's compare-and-
    -- swap and select_leaf's validated branch selection. Nothing else
    -- touches it — a message-level edit (update_message_state,
    -- append_stream_delta) does not move the leaf and does not bump rev.
    rev              bigint      NOT NULL DEFAULT 0,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);

-- The conversation-list read path (F17, not this lane) needs "this user's
-- conversations, most recently updated first" without a table scan.
CREATE INDEX conversation_user_id_idx ON conversation (user_id, updated_at DESC);

CREATE TABLE message (
    id          uuid        PRIMARY KEY,
    conv_id     uuid        NOT NULL REFERENCES conversation(id) ON DELETE RESTRICT,
    parent_id   uuid        NULL,          -- NULL == the conversation root, and only it
    -- Not in §11.2's DDL text. FRONTEND_PLAN.md §3.1: "user_id on BOTH
    -- tables in Phase 1 ... so multi-user is a feature addition and not a
    -- migration." Every message carries its author's id independent of
    -- conversation.user_id (the owner), so a future multi-party thread does
    -- not need a migration to add what this column already is.
    user_id     uuid        NOT NULL,
    role        text        NOT NULL CHECK (role IN ('system','user','assistant')),
    content     jsonb       NOT NULL,      -- the wire form (§5.1 content-parts) — never re-rendered, see store.ts
    state       text        NOT NULL DEFAULT 'complete'
                            CHECK (state IN ('pending','streaming','complete','failed')),
    error       jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    -- Not in §11.2's literal text either, but required by append_stream_delta
    -- (checkpointed content needs a "when was this last written" marker) and
    -- by update_message_state. Immutable columns (id/conv_id/parent_id) are
    -- exempt from ever needing this; content/state/error are not immutable.
    updated_at  timestamptz NOT NULL DEFAULT now(),
    deleted_at  timestamptz,
    UNIQUE (conv_id, id),
    CONSTRAINT message_parent_fk
        FOREIGN KEY (conv_id, parent_id) REFERENCES message (conv_id, id)
        ON DELETE RESTRICT,
    CONSTRAINT message_not_self_parent CHECK (parent_id IS DISTINCT FROM id)
);

-- Exactly one root per conversation (D5). A partial unique index, not a
-- CHECK: a CHECK cannot see other rows, and "at most one row with
-- parent_id IS NULL per conv_id" is inherently a cross-row constraint. This
-- is what makes 2026-08-24's second root structurally impossible rather
-- than merely tested against.
CREATE UNIQUE INDEX message_one_root_per_conv
    ON message (conv_id) WHERE parent_id IS NULL;

-- The tail-first read path (store.ts readTail/readOlder) walks parent_id
-- one hop at a time via the primary key (message.id) — already O(1) per hop
-- via the PK index, needing no help from this index. This one instead backs
-- audit_conversation's full-conversation recursive walk and any future
--"list messages by wall-clock time" read model, so a 2000-message audit
-- does not fall back to a sequential scan under conv_id.
CREATE INDEX message_conv_created_at_idx ON message (conv_id, created_at, id);

-- The leaf pointer can only ever name a message of THIS conversation
-- (composite FK on conv_id, closing the exact hole conversation.rev alone
-- would not: a bare FK on current_leaf_id -> message(id) would happily
-- accept a message belonging to a different conversation).
--
-- DEFERRABLE INITIALLY DEFERRED: create_conversation writes the
-- conversation row with current_leaf_id already set to the synthetic
-- root's id, then inserts that root message, in one transaction — the two
-- rows are mutually forward-referencing for the instant between them, and
-- this constraint (checked at COMMIT, not per-statement) is what makes that
-- legal without a third, ordering-only statement.
ALTER TABLE conversation
    ADD CONSTRAINT conversation_leaf_fk
    FOREIGN KEY (id, current_leaf_id) REFERENCES message (conv_id, id)
    DEFERRABLE INITIALLY DEFERRED;

-- Structural immutability (§11.2): id, conv_id and parent_id may never
-- change after insert. "Reparenting is not an operation this system has" —
-- enforced here, not by convention, so a future call site that tries it
-- fails loudly rather than silently succeeding.
CREATE OR REPLACE FUNCTION message_forbid_structural_update() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.id IS DISTINCT FROM OLD.id THEN
        RAISE EXCEPTION 'message.id is immutable (attempted % -> %)', OLD.id, NEW.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NEW.conv_id IS DISTINCT FROM OLD.conv_id THEN
        RAISE EXCEPTION 'message.conv_id is immutable (attempted % -> %)', OLD.conv_id, NEW.conv_id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    IF NEW.parent_id IS DISTINCT FROM OLD.parent_id THEN
        RAISE EXCEPTION 'message.parent_id is immutable (attempted % -> %)', OLD.parent_id, NEW.parent_id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER message_forbid_structural_update
    BEFORE UPDATE ON message
    FOR EACH ROW
    EXECUTE FUNCTION message_forbid_structural_update();

-- audit_conversation(conv_id) — FRONTEND_SPEC.md §11.4 / FRONTEND_PLAN.md
-- §3.1: "one function, one recursive query, shared by the load path, the
-- tests, the importer and the CLI." A single SQL function (not plpgsql) so
-- it is exactly one statement: one recursive CTE (`down`), referenced three
-- times by the outer SELECT (Postgres materializes a recursive CTE once per
-- execution regardless of how many times it is referenced — it cannot be
-- inlined — so this is one traversal of the tree, not three).
--
-- reachable_n / deepest / chain_from_current all derive from that one
-- traversal:
--   - reachable_n: count of distinct ids the walk reaches from ANY root row
--     of this conversation. Tombstones do NOT prune this walk on purpose —
--     property 5 ("no message unreachable from the root") is about orphans
--     (a broken parent link), not visibility. A tombstoned subtree still has
--     an intact parent chain; it is hidden from rendering, not orphaned.
--   - deepest: the longest root-to-node chain anywhere in the tree.
--     Diagnostic only — see the PASS formula below.
-- leaf_on_tree is membership in `walk` — REACHABLE FROM A ROOT — not mere
-- existence of the row. §4.1 property 2 is "current_leaf is reachable from
-- that root", and an earlier revision of this function tested
-- `EXISTS (SELECT 1 FROM message ...)` instead, which is a different claim:
-- a leaf sitting in an orphaned subtree exists but is not on the tree. The
-- PASS formula happened to still fail such a conversation via
-- reachable_n = total, so the bug was invisible in the boolean while the
-- reported FIELD was wrong — and this function's whole purpose is that the
-- reported fields are what a human reads at 2am. `chain_from_current` on
-- the very next line already used `walk`; the two now agree.
--   - chain_from_current: the current leaf's own depth within that same
--     walk (0 if the leaf is unset or not found in it, e.g. an orphaned or
--     nonexistent leaf).
--
-- PASS is EXACTLY §11.4's formula and no other field participates in it:
-- roots = 1 AND missing_parent = 0 AND reachable_n = total AND
-- leaf_on_tree. chain_from_current and deepest are reported for
-- diagnosability (FRONTEND_SPEC.md §13's five-tuple) and MUST NOT gate
-- PASS — on 2026-08-24 a deepest=208 read as healthy while
-- chain_from_current=8; the fix is that this function never lets either
-- number near the boolean.
CREATE OR REPLACE FUNCTION audit_conversation(p_conv_id uuid)
RETURNS TABLE (
    conv_id             uuid,
    roots               bigint,
    total               bigint,
    reachable_n         bigint,
    missing_parent      bigint,
    leaf                uuid,
    leaf_on_tree        boolean,
    chain_from_current  bigint,
    deepest             bigint,
    pass                boolean
)
LANGUAGE sql
STABLE
AS $$
    WITH RECURSIVE down(id, depth) AS (
        SELECT m.id, 1
        FROM message m
        WHERE m.conv_id = p_conv_id AND m.parent_id IS NULL
        UNION ALL
        SELECT m.id, down.depth + 1
        FROM message m
        JOIN down ON m.parent_id = down.id
        WHERE m.conv_id = p_conv_id
    ),
    walk AS (
        SELECT id, min(depth) AS depth FROM down GROUP BY id
    )
    SELECT
        p_conv_id,
        (SELECT count(*) FROM message WHERE conv_id = p_conv_id AND parent_id IS NULL),
        (SELECT count(*) FROM message WHERE conv_id = p_conv_id),
        (SELECT count(*) FROM walk),
        (SELECT count(*) FROM message m
            WHERE m.conv_id = p_conv_id AND m.parent_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM message p WHERE p.conv_id = m.conv_id AND p.id = m.parent_id)),
        c.current_leaf_id,
        (c.current_leaf_id IS NOT NULL
            AND EXISTS (SELECT 1 FROM walk WHERE id = c.current_leaf_id)),
        coalesce((SELECT depth FROM walk WHERE id = c.current_leaf_id), 0),
        coalesce((SELECT max(depth) FROM walk), 0),
        -- roots = 1 AND missing_parent = 0 AND reachable_n = total AND leaf_on_tree
        (
            (SELECT count(*) FROM message WHERE conv_id = p_conv_id AND parent_id IS NULL) = 1
            AND (SELECT count(*) FROM message m
                    WHERE m.conv_id = p_conv_id AND m.parent_id IS NOT NULL
                      AND NOT EXISTS (SELECT 1 FROM message p WHERE p.conv_id = m.conv_id AND p.id = m.parent_id)) = 0
            AND (SELECT count(*) FROM walk) = (SELECT count(*) FROM message WHERE conv_id = p_conv_id)
            AND (c.current_leaf_id IS NOT NULL
                    AND EXISTS (SELECT 1 FROM walk WHERE id = c.current_leaf_id))
        )
    FROM conversation c
    WHERE c.id = p_conv_id;
$$;

-- rebuild_read_models(conv_id) — §11.1's contract requires this to exist IF
-- a denormalized read model exists. Phase 1 deliberately builds none: see
-- docs/lanes/L1-store.md ("read contract" / "read models") for the
-- reasoning — the tail-first read path is a live, bounded-depth recursive
-- walk against the base tables (store.ts readTail/readOlder), which already
-- meets the scale bar without a cache, and an out-of-band psql repair is
-- therefore honoured by the very next read with nothing to invalidate. This
-- function is a documented no-op kept present so the contract's shape
-- exists on day one; the day a cache is added, its invalidation logic goes
-- here and this function stops being a no-op.
CREATE OR REPLACE FUNCTION rebuild_read_models(p_conv_id uuid) RETURNS void
LANGUAGE sql
AS $$
    SELECT NULL::void WHERE p_conv_id IS NULL AND false;
$$;
