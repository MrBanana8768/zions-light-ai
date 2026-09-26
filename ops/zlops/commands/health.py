"""`zl health [--json|--raw]` -- /health/full, interpreted by
zlops.telemetry.interpret_health (the fixed seam).

Exit code (decision, documented here and in ops/README.md -- the brief left
this to us: "pick and document; don't fold warn into ok"):
    0   status == "ok"
    1   status == "crit" (any finding at crit level, or the endpoint is
        unreachable -- interpret_health treats down/unreachable as crit)
    3   status == "warn" -- informational, matches the dry-run-pending-work
        convention: something needs attention but nothing is broken yet
"""

from __future__ import annotations

import json
import time

from .. import common, podenv


def run(argv: list[str]) -> int:
    as_json = "--json" in argv
    as_raw = "--raw" in argv

    _status_code, doc, error = common.http_get_json(f"{podenv.COMPACTOR_URL}/health/full", timeout=5.0)

    if as_raw:
        if doc is None:
            print(json.dumps({"error": error or "no response"}))
            return 1
        print(json.dumps(doc, indent=2))
        return 0

    from .. import telemetry  # lazy: B's module

    verdict = telemetry.interpret_health(doc, time.time())

    if as_json:
        print(json.dumps(verdict, indent=2))
    else:
        print(f"status: {verdict['status']}")
        if verdict["reasons"]:
            print("reasons:")
            for r in verdict["reasons"]:
                print(f"  - {r}")
        print("findings:")
        for f in verdict["findings"]:
            print(f"  [{f['level']}] {f['text']}")
        if error:
            print(f"(compactor /health/full was not reachable: {error})")

    if verdict["status"] == "crit":
        return 1
    if verdict["status"] == "warn":
        return 3
    return 0
