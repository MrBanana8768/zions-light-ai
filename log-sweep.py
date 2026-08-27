#!/usr/bin/env python3
"""Find exception handlers that swallow a failure without saying anything.

Every failure this project spent a week diagnosing had the same shape: a
degraded mode indistinguishable from a healthy one. The worst case ran for
months — `count_tokens` lost its chat-template accuracy behind a bare
`except Exception:` with no log statement (main.py:298), and nothing anywhere
reported it until a conversation grew large enough to overflow the context
window. See REMEDIATION.md P0-0 and P0-2b.

THE RULE this enforces: a handler may swallow an exception only if it does
exactly one of — logs it, re-raises it, or returns it to a caller that
surfaces it. "Returns a plausible-looking default" is none of those three.

This script finds handlers doing none of the first two. It cannot tell whether
a returned value is surfaced by the caller, so it reports candidates; triage
the output by hand. REMEDIATION.md P0-2b carries the triage as of 2026-08-27
(47 candidates: ~13 genuinely silent, ~8 legitimate cleanup, ~26 propagating
by return value).

Usage:
    python3 log-sweep.py [path ...]        # default: compactor/
    python3 log-sweep.py --count           # just the number, for CI

Exit status is 0 always — this is a review aid, not a gate. Making it a gate
would require the triage to live in code, and the triage needs a human.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXCEPT_RE = re.compile(r"^(\s*)except\b")
# A handler is "accounted for" if its body does any of these.
ACCOUNTED = ("logger.", "raise", "alert", "print(")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return (lineno, handler_line, body_preview) for silent handlers."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        print(f"  ! could not read {path}: {e}", file=sys.stderr)
        return []

    found: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = EXCEPT_RE.match(line)
        if not m:
            continue
        base = len(m.group(1))
        body: list[str] = []
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                continue
            if indent_of(nxt) <= base:
                break
            body.append(nxt)
        blob = "\n".join(body)
        if any(tok in blob for tok in ACCOUNTED):
            continue
        preview = " ; ".join(b.strip() for b in body[:2])
        found.append((i + 1, line.strip(), preview[:80]))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", default=["compactor"],
                    help="files or directories to scan (default: compactor/)")
    ap.add_argument("--count", action="store_true",
                    help="print only the total, for scripting")
    ap.add_argument("--include-tests", action="store_true",
                    help="scan test_*.py too (skipped by default)")
    args = ap.parse_args()

    targets: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            targets.extend(sorted(p.rglob("*.py")))
        elif p.is_file():
            targets.append(p)
        else:
            print(f"! no such path: {p}", file=sys.stderr)

    if not args.include_tests:
        targets = [t for t in targets if not t.name.startswith("test_")]

    total = 0
    rows: list[tuple[Path, int, str, str]] = []
    for t in targets:
        for lineno, handler, preview in scan_file(t):
            rows.append((t, lineno, handler, preview))
            total += 1

    if args.count:
        print(total)
        return 0

    print(f"Exception handlers with no logger / raise / alert: {total}")
    print("Triage by hand — see REMEDIATION.md P0-2b.\n")
    for path, lineno, handler, preview in rows:
        print(f"{str(path):<28} :{lineno:<5} {handler:<44} -> {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
