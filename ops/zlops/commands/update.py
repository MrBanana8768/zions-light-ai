"""`zl update [--ref REF]` -- fetch + checkout in podenv.REPO (the pod's own
clone), refusing if that clone is dirty. Updates ZL_HOME/ref (the same file
bootstrap.sh writes -- `zl version`'s `REF_FILE`) once the checkout succeeds.
No --apply gate: the brief's own command list shows none for `update`, and a
`git fetch`+`checkout` on a clone this bundle owns is the action itself, not
a preview of one -- the dirty-clone refusal is the safety net instead.
"""

from __future__ import annotations

import argparse

from .. import common, podenv
from .version import REF_FILE


def _current_ref() -> str | None:
    try:
        return REF_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="zl update")
    ap.add_argument("--ref", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    if not (podenv.REPO / ".git").exists():
        print(f"zl update: {podenv.REPO} is not a git checkout")
        return 1

    dirty = common.run(["git", "-C", str(podenv.REPO), "status", "--porcelain"], timeout=15)
    if dirty.returncode != 0:
        print(f"zl update: `git status` failed: {dirty.stderr or dirty.stdout}")
        return 1
    if dirty.stdout.strip():
        print(f"zl update: refusing -- {podenv.REPO} has local changes:\n{dirty.stdout}")
        return 1

    ref = args.ref or _current_ref()
    if not ref:
        print("zl update: no --ref given and no saved ref to fall back to (pass --ref)")
        return 2

    print(f"fetching {ref} into {podenv.REPO} ...")
    fetch = common.run(["git", "-C", str(podenv.REPO), "fetch", "--depth", "1", "origin", ref], timeout=120)
    if fetch.returncode != 0:
        print(f"zl update: git fetch failed:\n{fetch.stdout}\n{fetch.stderr}")
        return 1

    checkout = common.run(["git", "-C", str(podenv.REPO), "checkout", "FETCH_HEAD"], timeout=30)
    if checkout.returncode != 0:
        print(f"zl update: git checkout failed:\n{checkout.stdout}\n{checkout.stderr}")
        return 1

    commit = common.run(["git", "-C", str(podenv.REPO), "rev-parse", "HEAD"], timeout=10)
    REF_FILE.parent.mkdir(parents=True, exist_ok=True)
    REF_FILE.write_text(ref + "\n", encoding="utf-8")
    print(f"zl update: now at {ref} ({commit.stdout.strip()})")
    return 0
