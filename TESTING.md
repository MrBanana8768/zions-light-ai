# Testing Standard

How Zion's Light AI is tested, and what every contributor (human or agent)
must do when adding a feature. This standard is **in force now** — even
though some of the automation it describes (the boot self-test harness,
the integration suite) lands in V2.2. Write to the standard regardless of
which tooling exists yet.

## Why three tiers

The stack has three fundamentally different kinds of correctness, each
needing a different test environment:

- **Pure logic** (token math, JSON I/O, parsers, state machines) is
  deterministic and needs no GPU — test it fast and often.
- **The live stack** (vLLM + compactor + OpenWebUI wired together) can
  only be validated against a running deployment with a real model and
  GPU — test it on the pod.
- **End-to-end behavior** ("does memory actually survive 300 turns?")
  needs the full system AND realistic scale — test it as a deliberate
  scenario, not on every commit.

Mixing these is the trap. A unit test that needs a GPU isn't a unit test;
a "does it boot" check that runs in CI without hardware proves nothing.

## The tiers

| Tier | Name | Scope | Location | Runs | Needs GPU |
|---|---|---|---|---|---|
| **1** | Unit / logic | Pure functions, file I/O, parsers, state machines, env handling | `compactor/test_*.py` | CI on every PR; locally in a CPU container | No |
| **2** | Boot self-test | Live-stack health: ports up, model loaded, real chat round-trip, facts read/write | `compactor/selftest.py` | Automatically post-boot on the pod; on-demand via `/admin/selftest` | Yes (on the pod) |
| **3** | Integration | End-to-end scenarios (e.g. facts persistence across hundreds of turns) | `tests/integration/` | Manually, or CI against an ephemeral/live pod | Yes |

## Tier 1 — Unit / logic tests

**What belongs here:** anything that can be tested without a GPU, a real
model, or the network. Token estimation, conv_id resolution, facts
load/save/prune, the SSE accumulator, backfill state transitions,
extraction-output parsing, env-var defaulting.

**Pattern:** plain `assert_eq`/`assert_true` helpers, a `__main__` runner,
exit code 1 on first failure. No pytest dependency required (keeps the
CPU container minimal). Mock the HTTP layer with `unittest.mock` when a
function calls vLLM. Redirect storage to a tempdir via
`COMPACTOR_STORAGE_ROOT` **before** importing the module under test.

**Current Tier-1 suites** (all green as of V2.0 Phase 4):
- `compactor/test_smoke.py` — core compactor logic (token counting, split,
  compaction no-op, route registration, env defaults)
- `compactor/test_memory.py` — conv_id resolution, storage layout, admin
  route registration
- `compactor/test_facts.py` — facts I/O, atomic writes, LRU pruning,
  extraction parser, extraction with mocked vLLM
- `compactor/test_backfill.py` — state machine, stale detection, pair
  extraction, needs_backfill decision matrix, end-to-end with mock vLLM
- `compactor/test_retrieval.py` — graceful degradation (deps missing),
  index/retrieve/forget with mock embedder + ChromaDB collection,
  conv-isolation via metadata filter
- `compactor/test_summarizer.py` — state I/O, threshold detection per
  tier, L1→L2→L3 cascade with shrunk thresholds + mock LLM, no-op when
  below threshold, exception-swallowing when LLM fails

**Run them** (Linux/macOS — Windows users invoke under `MSYS_NO_PATHCONV=1`
to avoid Git Bash's `/opt/...` path-mangling trap):
```bash
docker run --rm -v "$PWD/compactor:/work" -w /work python:3.12-slim bash -lc '
  pip install --quiet fastapi "uvicorn[standard]" httpx 2>/dev/null
  for t in test_smoke test_memory test_facts test_backfill test_retrieval test_summarizer; do
    python $t.py 2>&1 | tail -2
  done
'
```

## Tier 2 — Boot self-test  *(harness lands in V2.2)*

**What it is:** a validation battery that runs *inside the container after
services start*, proving the deployment actually works — not just that
processes spawned. Checks: `/data` writable, vLLM lists the model,
compactor `/health` ok, a real 1-token chat completion round-trips through
the full chain, a facts write/read/delete round-trips, admin endpoints are
localhost-gated.

**How it runs:**
- **Automatically on boot** — a one-shot `[program:selftest]` supervisord
  entry runs `compactor/selftest.py --on-boot --wait-for-ready`, logging
  PASS/FAIL to `/var/log/supervisor/selftest.log`. It's a separate process
  and **cannot** affect vLLM/compactor/OpenWebUI. Toggle with
  `COMPACTOR_SELFTEST_ON_BOOT=false`.
- **On demand** — `GET http://localhost:8080/admin/selftest` (localhost
  only) returns a JSON report. Answers "is the deploy healthy right now?"
- **From a pod shell** — `python /opt/compactor/selftest.py`.

**Check the result on a running pod:**
```bash
cat /var/log/supervisor/selftest.log              # boot result
curl -s http://localhost:8080/admin/selftest | jq # on-demand
```

## Tier 3 — Integration tests

**What it is:** deliberate end-to-end scenarios run against a live
deployment, exercising behavior that only emerges at scale. Hits the
compactor as a **black box** — never imports project code, never runs
inside the shipped image (excluded by `.dockerignore`).

**Suite lives at** `tests/integration/` — see
[`tests/integration/README.md`](tests/integration/README.md) for the full
walkthrough.

**Current coverage** (16 tests, plus 2 `slow` for the summarizer rollups):
- `test_00_smoke.py` — health, models listed, basic chat round-trip
- `test_facts.py` — facts extracted, persisted, used in next turn
- `test_retrieval.py` — exchanges indexed, distinctive content retrieved
  back via RAG
- `test_summarizer.py` — L1 rollup fires after threshold *(slow: drives
  20+ real chat turns; opt in with `-m slow`)*
- `test_forget.py` — DELETE clears facts + episodic + summary together;
  idempotent
- `test_degraded.py` — hash-fallback conv_id, empty system prompts,
  multimodal content arrays — compactor stays up

**Modes** (admin-requiring tests skip cleanly when `ZIONS_TEST_ADMIN_URL`
is unset, so a basic confidence run works from anywhere):
```bash
# Basic — no admin access needed
ZIONS_TEST_BASE_URL=https://{POD_ID}-8080.proxy.runpod.net \
  pytest tests/integration/ -v

# Full — with admin access via SSH tunnel or pod-side bind flip
ZIONS_TEST_BASE_URL=https://{POD_ID}-8080.proxy.runpod.net \
ZIONS_TEST_ADMIN_URL=http://localhost:8081 \
  pytest tests/integration/ -v

# Including the slow summarizer rollup tests
ZIONS_TEST_BASE_URL=... ZIONS_TEST_ADMIN_URL=... \
  pytest tests/integration/ -v -m "slow or not slow"
```

## The standard — what every feature PR must do

1. **New logic → Tier-1 tests in the same PR.** Any new pure function,
   parser, I/O routine, or state machine ships with `compactor/test_*.py`
   coverage. No exceptions for "it's simple" — the v1.9.2 empty-string
   crash was one line.
2. **New runtime surface → a Tier-2 self-test assertion.** Adding an
   endpoint, a service, or a new external dependency (a DB, an embedding
   model) means adding a check to `selftest.py` so a broken deploy is
   caught at boot, not by a user.
3. **Phase/feature verification → a permanent Tier-3 test.** When a
   feature's acceptance criterion is an end-to-end scenario (like the
   memory phases' verification steps), encode it in `tests/integration/`
   instead of leaving it a one-off manual checklist.
4. **All Tier-1 suites pass before merge.** Green CPU-container run is a
   merge gate.
5. **Tier-2 green before promoting `:latest`.** A release image's boot
   self-test must pass on a real pod before its tag is promoted to
   `:latest` (this becomes a CI gate per the ROADMAP CI/CD plan).

## Relationship to CI/CD

See [ROADMAP.md](ROADMAP.md) "Cross-cutting infrastructure → CI/CD
automation". In short: Tier 1 runs in GitHub Actions on every PR; Tier 2
runs against a pod before `:latest` promotion; Tier 3 runs against an
ephemeral pod for release candidates. The vulnerability gate (block
Critical CVEs) sits alongside these as a parallel merge/promote gate.
