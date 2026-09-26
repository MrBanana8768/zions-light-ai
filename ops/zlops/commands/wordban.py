"""`zl wordban-check [...]` -- runs the pod's own uploaded word-ban check
(ZL_HOME/word-ban/word-ban-check.py, pushed there by `zlpod push wordban`).
It is a self-contained, stdlib-only script (scripts/wordban/check_template.py
imports nothing beyond the standard library plus urllib), so plain `python3`
is the right interpreter -- no venv needed.
"""

from __future__ import annotations

from .. import common, podenv

_SCRIPT = podenv.ZL_HOME / "word-ban" / "word-ban-check.py"


def run(argv: list[str]) -> int:
    if not _SCRIPT.is_file():
        print(
            f"zl wordban-check: {_SCRIPT} not found.\n\n"
            "It has to be pushed from the laptop first:\n"
            "  zlpod push wordban [DIR]\n"
            "(default DIR: ~/zl-ops/word-ban, the output of `wordban.py release` --\n"
            "see scripts/wordban/README.md's Workflow section)."
        )
        return 1
    return common.run_stream(["python3", str(_SCRIPT), *argv])
