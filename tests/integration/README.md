# Tier-3 integration validation suite

End-to-end validation that the deployed compactor's **public API** behaves
correctly. Hits the running stack as a black box — does **not** import
any compactor code, runs from outside the pod, and is **never** baked
into the Docker image.

This is the regression net for V2.0 and everything that builds on it.
Per [TESTING.md](../../TESTING.md), every V2 phase's "verification
checklist" graduates into permanent tests here.

## When to run it

- After any compactor code change, before merging
- After bumping any dep in `compactor/requirements.txt`
- After every published image tag, before flipping `:latest`
- On a schedule (e.g. weekly cron) against the live deployment to catch
  silent regressions caused by upstream changes

## Setup (one-time)

```bash
# Create a throwaway venv for the test runner.
python -m venv .test-venv
source .test-venv/bin/activate          # or .\.test-venv\Scripts\Activate.ps1
pip install -r tests/integration/requirements.txt
```

That's all. The suite has no project dependencies — just `pytest` and
`httpx`.

## Required configuration

The harness reads its connection settings from env vars (or `--base-url`
/ `--admin-url` on the pytest command line).

| Env var | Required? | Purpose |
|---|---|---|
| `ZIONS_TEST_BASE_URL` | **Yes** | Public URL of the compactor, e.g. `https://abc123-8080.proxy.runpod.net` |
| `ZIONS_TEST_ADMIN_URL` | No | URL for `/admin/*` endpoints. When unset, admin-requiring tests skip cleanly. |
| `ZIONS_TEST_MODEL` | No | Model name to use; auto-detected from `/v1/models` if unset. |
| `ZIONS_TEST_TIMEOUT` | No | HTTP timeout in seconds (default 120) |
| `ZIONS_TEST_TAIL_WAIT` | No | How long to wait for the async post-response work to finish before assertions (default 8s) |

## Four modes of running

### 1. On-pod (recommended for post-deploy validation)

The simplest path — clone the branch, install deps, run. Because requests
originate from `127.0.0.1`, admin endpoints work without any tunneling
or any change to `COMPACTOR_ADMIN_BIND` (which stays safely at `127.0.0.1`).

Open the RunPod **Web Terminal** (or `runpodctl ssh`) and paste:

```bash
# One-time setup: clone the suite onto the network volume so it
# survives pod restarts (~50KB, depth=1).
rm -rf /data/integration-tests && \
git clone --depth=1 --branch master \
    https://github.com/MrBanana8768/zions-light-ai.git /data/integration-tests && \
cd /data/integration-tests/tests/integration && \
python3 -m venv .venv && \
source .venv/bin/activate && \
pip install -q -r requirements.txt

# Run the suite — both URLs are localhost since we're inside the pod.
ZIONS_TEST_BASE_URL=http://localhost:8080 \
ZIONS_TEST_ADMIN_URL=http://localhost:8080 \
pytest -v
```

For subsequent runs on the same pod, just:
```bash
cd /data/integration-tests/tests/integration && \
source .venv/bin/activate && \
ZIONS_TEST_BASE_URL=http://localhost:8080 \
ZIONS_TEST_ADMIN_URL=http://localhost:8080 \
pytest -v
```

To validate a pre-release branch (e.g. `v2.0-phase4.2` candidate before
merge), swap `--branch master` for `--branch <branch-name>` in the clone
step.

### 2. Basic — chat round-trip only (works from anywhere)

The minimum confidence-builder. Hits public endpoints only; no admin
access needed. Validates that chat completes end-to-end and that the
compactor's request flow doesn't crash on weird-but-valid inputs.

```bash
export ZIONS_TEST_BASE_URL="https://<pod>-8080.proxy.runpod.net"
pytest tests/integration/ -v
```

Admin-requiring tests show as `SKIPPED: admin endpoint required` — that
is expected and not a failure.

### 3. Full from off-pod — with admin endpoint access

The same regression net as Mode 1, but driven from your laptop or CI
instead of the pod's Web Terminal. Useful when CI orchestrates the
validation, or when you want to exercise the public proxy URL end-to-end
including TLS / RunPod's proxy layer.

**Option A — SSH tunnel** (most secure; what production environments should use):
```bash
# In one terminal: forward the pod's localhost:8080 to your laptop's 8081
ssh -L 8081:localhost:8080 <pod-user>@<pod-ssh-host>

# In another terminal:
export ZIONS_TEST_BASE_URL="https://<pod>-8080.proxy.runpod.net"
export ZIONS_TEST_ADMIN_URL="http://localhost:8081"
pytest tests/integration/ -v
```

**Option B — temporarily flip admin bind on the pod** (easier, less secure):
```bash
# On the pod: set this env var and restart the compactor
COMPACTOR_ADMIN_BIND=0.0.0.0
supervisorctl restart compactor

# From anywhere:
export ZIONS_TEST_BASE_URL="https://<pod>-8080.proxy.runpod.net"
export ZIONS_TEST_ADMIN_URL="https://<pod>-8080.proxy.runpod.net"
pytest tests/integration/ -v
```

**Revert `COMPACTOR_ADMIN_BIND` to `127.0.0.1` afterwards** — the admin
endpoints have no auth.

### 4. Full + slow — including the summary-rollup tests

The hierarchical-summarizer tests drive ≥ 20 real chat turns each (because
that's the default L1 rollup threshold). That takes 5-15 minutes of real
inference time per test. Default `pytest` skips them; opt in:

```bash
pytest tests/integration/ -v -m slow                # ONLY slow tests
pytest tests/integration/ -v -m "slow or not slow"  # ALL tests, slow included
```

## What success looks like

Default fast run (no `-m slow`) on a healthy deployment:

```
  Deployment OK: https://...-8080.proxy.runpod.net  model=anthracite-org/magnum-v4-12b  admin=enabled

test_00_smoke.py::test_health_endpoint_responds                     PASSED
test_00_smoke.py::test_models_endpoint_lists_at_least_one_model     PASSED
test_00_smoke.py::test_minimal_chat_round_trip                      PASSED
test_00_smoke.py::test_chat_handles_system_prompt                   PASSED
test_degraded.py::test_chat_works_without_explicit_conv_id          PASSED
test_degraded.py::test_chat_handles_empty_system_prompt             PASSED
test_degraded.py::test_chat_handles_multimodal_content_array        PASSED
test_degraded.py::test_admin_endpoints_reject_unknown_conv          PASSED
test_facts.py::test_fact_extracted_and_persisted                    PASSED
test_facts.py::test_fact_used_in_next_turn                          PASSED
test_facts.py::test_admin_summary_reports_fact_count                PASSED
test_forget.py::test_forget_clears_facts_and_episodic               PASSED
test_forget.py::test_forget_response_shape                          PASSED
test_forget.py::test_forget_is_idempotent                           PASSED
test_retrieval.py::test_exchange_gets_indexed                       PASSED
test_retrieval.py::test_distinctive_content_retrieved_later         PASSED

================ 16 passed, 2 deselected in 4m12s ================
```

## What to do when something fails

Each assertion message includes enough context to diagnose without
re-running:
- `test_facts.py` failures dump the extracted facts AND the model
  response, so you can tell whether facts weren't extracted vs. weren't
  used.
- `test_retrieval.py` failures dump `indexed_exchanges` AND the probe
  response.
- `test_forget.py` failures dump the before/after admin summaries.

If the harness can't reach the deployment at all (BASE_URL wrong, pod
down, model not loaded), the session-scope fixture in `conftest.py`
fails fast with a single clear error rather than letting every test
flail.

## What this suite deliberately does NOT cover

These are scoped for later cycles, by design:

- **Streaming responses** — current tests are non-streaming for assertion
  simplicity. Streaming-shape regressions are covered by Tier-1
  (`compactor/test_smoke.py::SseAccumulator` tests).
- **Backfill behavior** — needs a pre-existing V1 conversation with many
  turns to backfill. Add when the V2.1 export/import bundles exist (so
  we can seed a deterministic state).
- **Chaos: kill vLLM mid-request, fill disk, corrupt state files** —
  that's V2.3 Resilience & Stability work. The harness here is the
  foundation those chaos tests will plug into.
- **Long-soak / leak watch** — V2.3 territory.

## Layout

```
tests/integration/
├── README.md               (this file)
├── requirements.txt        (pytest + httpx — that's all)
├── _harness.py             (shared HTTP helpers + admin helpers + skip logic)
├── conftest.py             (pytest fixtures, --base-url / --admin-url CLI)
├── test_00_smoke.py        (runs first — basic liveness + chat round-trip)
├── test_degraded.py        (weird-but-valid inputs, hash-fallback conv_id, multimodal)
├── test_facts.py           (Phase 2: facts extracted + used)
├── test_forget.py          (DELETE clears all three memory layers)
├── test_retrieval.py       (Phase 3: RAG indexes + retrieves)
└── test_summarizer.py      (Phase 4: L1 rollup fires — `-m slow`)
```
