"""`zl history [-n N]` -- tail of podenv.AUDIT_LOG, human-readable."""

from __future__ import annotations

import argparse
import json

from .. import podenv


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="zl history")
    ap.add_argument("-n", type=int, default=20, metavar="N")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    if not podenv.AUDIT_LOG.is_file():
        print(f"zl history: no audit log yet ({podenv.AUDIT_LOG})")
        return 0

    try:
        lines = podenv.AUDIT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        print(f"zl history: could not read {podenv.AUDIT_LOG}: {e}")
        return 1

    for line in lines[-args.n :]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            print(f"(unparseable line) {line[:200]}")
            continue
        argv_str = " ".join(rec.get("argv", []))
        print(
            f"{rec.get('time', '?')}  exit={rec.get('exit_code', '?')}  "
            f"{rec.get('duration_s', '?')}s  {rec.get('client_ip', '?'):<15}  zl {argv_str}"
        )
    return 0
