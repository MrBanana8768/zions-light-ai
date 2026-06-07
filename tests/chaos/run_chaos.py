#!/usr/bin/env python3
"""
V2.3 Theme 2 — chaos suite. Deliberately breaks each dependency on a LIVE
pod and asserts the user-visible behavior is "degraded but functional",
never a hard 500 or a crash loop.

This is destructive and pod-local (it stops services and touches files on
/data), so it is NOT part of the auto-run pytest Tier-3 suite. It only runs
on the pod, only when you explicitly confirm:

    ZIONS_CHAOS_CONFIRM=break-my-pod \
      /opt/compactor-venv/bin/python tests/chaos/run_chaos.py

Each scenario restores what it broke in a finally block. The disk-fill
scenario is the riskiest and is OFF by default — add --with-disk-fill to
include it.

Exit 0 = every scenario degraded gracefully. Exit 1 = at least one hard
failure (a 500, a crash, or a non-restored dependency).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE = os.environ.get("ZIONS_CHAOS_BASE_URL", "http://localhost:8080").rstrip("/")
STORAGE = Path(os.environ.get("COMPACTOR_STORAGE_ROOT", "/data/openwebui/compactor"))
CONFIRM = os.environ.get("ZIONS_CHAOS_CONFIRM", "")
CONFIRM_PHRASE = "break-my-pod"


def _chat(content: str, conv_id: str, timeout: float = 60.0) -> httpx.Response:
    return httpx.post(
        f"{BASE}/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": content}],
              "stream": False, "max_tokens": 16},
        headers={"X-Conversation-Id": conv_id},
        timeout=timeout,
    )


def _supervisor(action: str, program: str) -> None:
    subprocess.run(["supervisorctl", action, program], check=False,
                   capture_output=True)


def _wait_vllm_ready(timeout_s: float = 600.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{BASE}/health/full", timeout=10.0)
            if r.status_code == 200 and r.json().get("checks", {}).get("vllm", {}).get("ok"):
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ---------------------------------------------------------------------------
# Scenarios — each returns (passed: bool, detail: str)
# ---------------------------------------------------------------------------

def scenario_vllm_killed() -> tuple[bool, str]:
    """Stop vLLM, send a chat → must be a clean 503 (not 500/crash). Then
    restart vLLM and confirm chat recovers."""
    try:
        _supervisor("stop", "vllm")
        time.sleep(3)
        r = _chat("hello while you're down", "chaos-vllm")
        if r.status_code != 503:
            return False, f"expected 503 with vLLM down, got {r.status_code}"
        body = r.json()
        if body.get("error", {}).get("code") != "model_unavailable":
            return False, f"503 but not the friendly body: {body}"
    finally:
        _supervisor("start", "vllm")
        if not _wait_vllm_ready():
            return False, "vLLM did not recover within timeout after restart"
    # Recovery check
    r2 = _chat("are you back?", "chaos-vllm")
    if r2.status_code != 200:
        return False, f"chat did not recover after vLLM restart: {r2.status_code}"
    return True, "clean 503 while down; recovered to 200 after restart"


def scenario_corrupt_facts() -> tuple[bool, str]:
    """Write garbage into a conversation's facts file → a chat for that conv
    must still return 200 (memory degrades to 'no facts', never 500)."""
    conv = "chaos-corrupt-facts"
    fpath = STORAGE / "facts" / f"{conv}.json"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    original = fpath.read_bytes() if fpath.is_file() else None
    try:
        fpath.write_text("{ this is not valid json at all ")
        r = _chat("does corrupt memory crash you?", conv)
        if r.status_code != 200:
            return False, f"corrupt facts caused {r.status_code}, expected 200"
        return True, "chat served 200 despite corrupt facts file"
    finally:
        if original is not None:
            fpath.write_bytes(original)
        elif fpath.is_file():
            fpath.unlink()


def scenario_chromadb_unwritable() -> tuple[bool, str]:
    """Make the ChromaDB dir unwritable → chat must still return 200
    (retrieval/indexing degrade to no-ops)."""
    chroma = STORAGE / "chromadb"
    chroma.mkdir(parents=True, exist_ok=True)
    original_mode = chroma.stat().st_mode
    try:
        os.chmod(chroma, 0o000)
        r = _chat("can you survive a read-only vector store?", "chaos-chroma")
        if r.status_code != 200:
            return False, f"unwritable chromadb caused {r.status_code}, expected 200"
        return True, "chat served 200 despite unwritable ChromaDB"
    finally:
        os.chmod(chroma, original_mode)


def scenario_disk_fill() -> tuple[bool, str]:
    """Consume free space below COMPACTOR_MIN_FREE_MB_WRITES with a balloon
    file → /health/full must report memory_writes=paused AND chat must still
    serve. Always deletes the balloon. RISKY — opt-in only."""
    import shutil
    balloon = STORAGE.parent / ".chaos_balloon"
    try:
        free_mb = shutil.disk_usage(str(STORAGE)).free / (1024 * 1024)
        # Read the running threshold from /health/full's report.
        hf = httpx.get(f"{BASE}/health/full", timeout=10.0).json()
        min_free = hf.get("memory_writes", {}).get("min_free_mb", 200)
        # Leave a hard 50 MB safety margin below the threshold.
        target_free = max(min_free - 50, 10)
        to_consume = int(free_mb - target_free)
        if to_consume <= 0:
            return False, f"already below threshold (free={free_mb:.0f}MB) — skipping"
        if to_consume > 20000:
            return False, f"would need to write {to_consume}MB — too large, skipping for safety"
        # Write the balloon in chunks
        with open(balloon, "wb") as f:
            chunk = b"\0" * (1024 * 1024)
            for _ in range(to_consume):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        time.sleep(12)  # let the degrade TTL cache expire
        hf2 = httpx.get(f"{BASE}/health/full", timeout=10.0).json()
        state = hf2.get("memory_writes", {}).get("new_memory_writes")
        if state != "paused":
            return False, f"expected memory_writes=paused under disk pressure, got {state!r}"
        # Chat must STILL serve (reads work, just no new persistence)
        r = _chat("you should still talk under disk pressure", "chaos-disk")
        if r.status_code != 200:
            return False, f"chat failed under disk pressure: {r.status_code}"
        return True, "writes paused + chat still served under disk pressure"
    finally:
        if balloon.exists():
            balloon.unlink()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="V2.3 chaos suite (destructive, pod-local).")
    p.add_argument("--with-disk-fill", action="store_true",
                   help="Include the risky disk-pressure balloon scenario.")
    args = p.parse_args(argv)

    if CONFIRM != CONFIRM_PHRASE:
        print(f"REFUSING TO RUN: set ZIONS_CHAOS_CONFIRM={CONFIRM_PHRASE} to confirm.")
        print("This suite stops services and touches /data — pod-local + destructive.")
        return 2

    scenarios = [
        ("vllm_killed", scenario_vllm_killed),
        ("corrupt_facts", scenario_corrupt_facts),
        ("chromadb_unwritable", scenario_chromadb_unwritable),
    ]
    if args.with_disk_fill:
        scenarios.append(("disk_fill", scenario_disk_fill))

    print(f"=== CHAOS SUITE against {BASE} ===\n")
    results = []
    for name, fn in scenarios:
        print(f"[running] {name} ...")
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"scenario raised: {type(e).__name__}: {e}"
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}\n")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"=== {passed}/{len(results)} scenarios degraded gracefully ===")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
