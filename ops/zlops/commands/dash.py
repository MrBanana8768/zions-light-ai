"""`zl dash [...]` -- dispatches to zlops.dash.main(argv), the fixed seam
this bundle splits on. Lazily imported so every OTHER `zl` command keeps
working while zlops/dash.py is still being written alongside this one.
"""

from __future__ import annotations


def run(argv: list[str]) -> int:
    from .. import dash

    return dash.main(argv)
