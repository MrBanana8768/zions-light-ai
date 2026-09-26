"""`zl selftest` -- GET /admin/selftest: a real chat round-trip + facts I/O
(OPERATIONS.md "Is it healthy right now? -- Deeper, on demand").
"""

from __future__ import annotations

import json

from .. import common, podenv


def run(argv: list[str]) -> int:
    _status_code, doc, error = common.http_get_json(f"{podenv.COMPACTOR_URL}/admin/selftest", timeout=30.0)
    if doc is None:
        print(f"zl selftest: could not reach {podenv.COMPACTOR_URL}/admin/selftest: {error}")
        return 1
    print(json.dumps(doc, indent=2))
    ok = doc.get("ok")
    if ok is False:
        return 1
    return 0
