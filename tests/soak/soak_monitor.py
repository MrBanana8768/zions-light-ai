#!/usr/bin/env python3
"""
V2.3 Theme 3 — memory/FD leak soak watch.

Samples the compactor process's RSS (resident memory) and open file-
descriptor count over time and flags slow monotonic growth as a suspected
leak. The usual suspects in this stack: unclosed httpx clients, ChromaDB
handles, and the background-task set (now bounded by bgwork — this watch
also confirms that bound holds under sustained load).

Pod-local + Linux-only (reads /proc). It's a monitoring tool, not a pytest —
run it on the pod, ideally for hours-to-days, while real or driven traffic
flows. With --drive it sends periodic chat requests itself so you don't
need a human at the keyboard.

    /opt/compactor-venv/bin/python tests/soak/soak_monitor.py \
        --duration-hours 24 --drive

Exit 0 = stable (no leak signature). Exit 1 = suspected leak. Exit 2 = the
compactor process couldn't be found.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = os.environ.get("ZIONS_SOAK_BASE_URL", "http://localhost:8080").rstrip("/")

# Leak thresholds — conservative; tune per deployment.
RSS_GROWTH_MB = float(os.environ.get("ZIONS_SOAK_RSS_GROWTH_MB", "150"))
FD_GROWTH = int(os.environ.get("ZIONS_SOAK_FD_GROWTH", "50"))


def find_compactor_pid() -> int | None:
    """Scan /proc for the uvicorn 'main:app' process (the compactor)."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "ignore"
            )
        except (OSError, ValueError):
            continue
        if "uvicorn" in cmdline and "main:app" in cmdline:
            return int(entry.name)
    return None


def sample(pid: int) -> tuple[float, int] | None:
    """Return (rss_mb, fd_count) for pid, or None if it vanished."""
    try:
        rss_kb = 0
        for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
                break
        fd_count = len(os.listdir(Path("/proc") / str(pid) / "fd"))
        return rss_kb / 1024.0, fd_count
    except (OSError, ValueError):
        return None


def _drive_traffic(n: int) -> None:
    """Send n quick chat requests to exercise the async tail. Best-effort —
    errors (e.g. vLLM busy) are ignored; we're stressing the compactor, not
    asserting model output."""
    try:
        import httpx
    except ImportError:
        return
    for i in range(n):
        try:
            httpx.post(
                f"{BASE}/v1/chat/completions",
                json={"model": "x",
                      "messages": [{"role": "user", "content": f"soak ping {i}"}],
                      "stream": False, "max_tokens": 8},
                headers={"X-Conversation-Id": f"soak-{i % 20}"},
                timeout=60.0,
            )
        except Exception:
            pass


def _slope(samples: list[tuple[float, float]]) -> float:
    """Least-squares slope of y over x. 0 if degenerate."""
    n = len(samples)
    if n < 2:
        return 0.0
    sx = sum(x for x, _ in samples)
    sy = sum(y for _, y in samples)
    sxx = sum(x * x for x, _ in samples)
    sxy = sum(x * y for x, y in samples)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compactor RSS/FD leak soak watch.")
    p.add_argument("--duration-hours", type=float, default=1.0,
                   help="How long to sample (default 1h; use 24+ for a real soak).")
    p.add_argument("--interval-s", type=float, default=60.0,
                   help="Seconds between samples (default 60).")
    p.add_argument("--drive", action="store_true",
                   help="Send periodic chat requests to generate load.")
    p.add_argument("--drive-per-interval", type=int, default=5,
                   help="Chat requests per interval when --drive (default 5).")
    p.add_argument("--pid", type=int, default=None,
                   help="Compactor PID (default: auto-detect via /proc).")
    p.add_argument("--out", default=None, help="Append JSONL samples to this file.")
    args = p.parse_args(argv)

    pid = args.pid or find_compactor_pid()
    if not pid:
        print("Could not find the compactor process (uvicorn main:app). "
              "Pass --pid, or run this on the pod.", file=sys.stderr)
        return 2

    print(f"=== SOAK WATCH: pid={pid}, {args.duration_hours}h @ {args.interval_s}s, "
          f"drive={args.drive} ===")
    out = open(args.out, "a") if args.out else None
    t0 = time.monotonic()
    deadline = t0 + args.duration_hours * 3600.0
    rss_series: list[tuple[float, float]] = []
    fd_series: list[tuple[float, float]] = []
    first = last = None

    try:
        while time.monotonic() < deadline:
            if args.drive:
                _drive_traffic(args.drive_per_interval)
            s = sample(pid)
            if s is None:
                print("compactor process vanished — aborting", file=sys.stderr)
                return 2
            rss_mb, fd = s
            t = (time.monotonic() - t0) / 3600.0  # hours since start
            rss_series.append((t, rss_mb))
            fd_series.append((t, float(fd)))
            if first is None:
                first = (rss_mb, fd)
            last = (rss_mb, fd)
            rec = {"t_hours": round(t, 4), "rss_mb": round(rss_mb, 1), "fd": fd}
            print(f"  t={t:6.3f}h  rss={rss_mb:8.1f}MB  fd={fd}")
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\n(interrupted — evaluating samples so far)")
    finally:
        if out:
            out.close()

    if not first or len(rss_series) < 3:
        print("Not enough samples to judge — run longer.")
        return 0

    rss_delta = last[0] - first[0]
    fd_delta = last[1] - first[1]
    rss_slope = _slope(rss_series)  # MB per hour
    fd_slope = _slope(fd_series)    # FDs per hour

    print("\n=== SOAK SUMMARY ===")
    print(f"  RSS: {first[0]:.1f} → {last[0]:.1f} MB  (Δ {rss_delta:+.1f} MB, "
          f"slope {rss_slope:+.1f} MB/h)")
    print(f"  FD:  {first[1]} → {last[1]}  (Δ {fd_delta:+d}, slope {fd_slope:+.1f}/h)")

    leak = False
    if rss_delta > RSS_GROWTH_MB and rss_slope > 0:
        print(f"  ⚠ SUSPECTED MEMORY LEAK: RSS grew {rss_delta:.0f} MB "
              f"(> {RSS_GROWTH_MB:.0f}) with positive trend.")
        leak = True
    if fd_delta > FD_GROWTH and fd_slope > 0:
        print(f"  ⚠ SUSPECTED FD LEAK: open FDs grew by {fd_delta} "
              f"(> {FD_GROWTH}) with positive trend.")
        leak = True
    if not leak:
        print("  ✓ stable — no leak signature.")
    return 1 if leak else 0


if __name__ == "__main__":
    sys.exit(main())
