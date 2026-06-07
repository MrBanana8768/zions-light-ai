# Chaos suite (V2.3 Theme 2)

Deliberately breaks each dependency on a **live pod** and asserts the
user-visible behavior is **degraded but functional** — never a hard 500 or
a crash loop. This is the failure-tested proof behind V2.3's graceful-
degradation guarantee.

> **This is destructive and pod-local.** It stops services and touches
> files under `/data`. It is intentionally **not** part of the auto-run
> pytest Tier-3 suite — it would be reckless to fill disks or kill vLLM in
> CI. Run it by hand, on the pod, when you want to *verify* resilience.

## What it breaks (and what "pass" means)

| Scenario | Breakage | Pass = |
|---|---|---|
| `vllm_killed` | `supervisorctl stop vllm`, chat during outage | clean **503** (`model_unavailable`), not 500; then chat **recovers to 200** after restart |
| `corrupt_facts` | garbage written into a conv's facts JSON | chat still returns **200** (memory degrades to "no facts") |
| `chromadb_unwritable` | `chmod 000` the ChromaDB dir | chat still returns **200** (retrieval/indexing degrade to no-ops) |
| `disk_fill` *(opt-in)* | balloon file consumes free space below the write threshold | `/health/full` reports `memory_writes=paused` **and** chat still serves |

Every scenario restores what it broke in a `finally` block (restarts vLLM,
rewrites/removes the corrupt file, restores dir perms, deletes the balloon).

## Running it

On the pod (RunPod Web Terminal), using the compactor venv (it has `httpx`):

```bash
ZIONS_CHAOS_CONFIRM=break-my-pod \
  /opt/compactor-venv/bin/python /opt/compactor/../tests/chaos/run_chaos.py
```

…but `tests/` is **not** in the image (excluded by `.dockerignore`), so
clone the repo onto the pod first (same pattern as the Tier-3 suite):

```bash
git clone --depth=1 --branch master \
  https://github.com/MrBanana8768/zions-light-ai.git /data/zions-src
ZIONS_CHAOS_CONFIRM=break-my-pod \
  /opt/compactor-venv/bin/python /data/zions-src/tests/chaos/run_chaos.py
```

Include the risky disk-pressure scenario:
```bash
ZIONS_CHAOS_CONFIRM=break-my-pod \
  /opt/compactor-venv/bin/python /data/zions-src/tests/chaos/run_chaos.py --with-disk-fill
```

Without `ZIONS_CHAOS_CONFIRM=break-my-pod` the script refuses to run
(exit 2). Exit 0 = all scenarios degraded gracefully; exit 1 = a hard
failure.

## Safety notes

- The **disk-fill** scenario writes a balloon file sized to drop free space
  just below the write-gate threshold, with a hard 50 MB margin and a 20 GB
  ceiling; it aborts rather than proceed if that math looks unsafe, and
  always deletes the balloon. Still — only run it when you can tolerate the
  pod briefly near-full. It's off by default for a reason.
- `vllm_killed` causes a real (brief) inference outage while vLLM restarts
  and reloads the model — expect a few minutes. Don't run during real use.
- Run on a scratch/test pod when possible, not your daily driver.

## Why this isn't pytest

The Tier-3 black-box suite (`tests/integration/`) talks to the pod over
HTTP only — it can't `chmod` ChromaDB or `supervisorctl stop vllm`. Chaos
needs filesystem + process control on the box itself, and must never run
unattended. So it's a guarded standalone runner, documented here, executed
deliberately — which is exactly the V2.3 "failure-tested on purpose" bar.
