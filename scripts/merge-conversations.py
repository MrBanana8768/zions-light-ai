#!/usr/bin/env python3
"""Fold one conversation's memory into another, on a pod running older code.

    supervisorctl stop compactor
    /opt/compactor-venv/bin/python /data/hotfix/scripts/merge-conversations.py \
        --src <old-conv-id> --dst <new-conv-id>            # dry run
    /opt/compactor-venv/bin/python /data/hotfix/scripts/merge-conversations.py \
        --src <old-conv-id> --dst <new-conv-id> --apply
    supervisorctl start compactor

WHY THIS EXISTS. `POST /admin/conversations/<src>/merge-into/<dst>` arrives in
v3.1.5. On a pod still running v3.1.4 the ROUTE is missing and so is
`portability.merge_conversation` itself, so there is nothing to call. This runs
the real function out of a clone of a later tag, against the live store,
without deploying that tag.

IT DOES NOT VENDOR A COPY OF THE MERGE LOGIC, deliberately. Two copies of one
rule drifting apart is this codebase's most expensive recurring defect. The
script only locates the sibling `compactor/` package of whatever checkout it
is sitting in, and calls the same `merge_conversation` the endpoint calls.

    git clone -b v3.1.8 <repo-url> /data/hotfix

THE COMPACTOR MUST BE STOPPED. `memory.conv_lock` is an `asyncio.Lock`, which
excludes coroutines inside one process and nothing else — a second process
holds no lock the service respects. Merging while the service is serving is
the adversarial finding A-05 shape: a memory tail writing the same facts file
concurrently loses whichever write lands second. So this refuses to run while
anything answers on the compactor's port, and says so rather than racing.

WHAT IT MERGES: facts and episodic exchanges. NOT summaries — the destination
derives its own hierarchy, and folding the source's in would double-count the
narrative. The SOURCE IS NEVER WRITTEN, so a merge that comes out wrong costs
nothing but the re-embedding.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent / "compactor"


def _compactor_is_live(port: int) -> bool:
    """True if anything answers /health on the compactor's port.

    Checked rather than assumed: the whole safety of this script is that no
    other process is writing the same files.
    """
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=3
        ) as r:
            return r.status == 200
    except urllib.error.URLError:
        return False
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="conv_id to merge FROM (never written)")
    ap.add_argument("--dst", required=True, help="conv_id to merge INTO")
    ap.add_argument("--apply", action="store_true",
                    help="commit. Without this it is a dry run.")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("COMPACTOR_PORT", "8080")))
    ap.add_argument("--allow-live", action="store_true",
                    help=argparse.SUPPRESS)  # escape hatch; see the docstring
    args = ap.parse_args()

    if not PKG.is_dir():
        print(f"ERROR: no compactor package beside this script ({PKG}).")
        print("Run it from a clone:  git clone -b v3.1.8 <repo-url> /data/hotfix")
        return 2

    if _compactor_is_live(args.port) and not args.allow_live:
        print(f"REFUSING: something is answering on :{args.port}.")
        print()
        print("conv_lock is an asyncio.Lock — it excludes coroutines inside one")
        print("process and nothing else. A memory tail writing the same facts")
        print("file while this runs loses whichever write lands second.")
        print()
        print("  supervisorctl stop compactor")
        print("  ...run this...")
        print("  supervisorctl start compactor")
        return 2

    sys.path.insert(0, str(PKG))
    try:
        import memory  # noqa: E402
        import portability  # noqa: E402
    except Exception as e:
        print(f"ERROR importing the compactor package from {PKG}: "
              f"{type(e).__name__}: {e}")
        return 2

    root = memory.storage_root()
    print(f"store:   {root}")
    print(f"code:    {PKG}")
    print(f"src:     {args.src}   (read-only)")
    print(f"dst:     {args.dst}")
    print(f"mode:    {'APPLY' if args.apply else 'dry run'}")
    print()

    # Back up what the merge actually writes, before it writes it. The source
    # needs no backup — merge_conversation never touches it — and summaries are
    # not merged, so dst's facts file is the whole blast radius on disk.
    if args.apply:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in (f"facts/{args.dst}.json", f"facts/{args.dst}.archive.json"):
            p = root / name
            if p.is_file():
                b = p.with_suffix(p.suffix + f".pre-merge-{stamp}")
                shutil.copy2(p, b)
                print(f"backed up {p.name} -> {b.name}")
        print()

    try:
        result = portability.merge_conversation(
            args.src, args.dst, dry_run=not args.apply
        )
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 1
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1

    print(json.dumps(result, indent=2, default=str))
    print()
    if args.apply:
        print("Committed. Verify before doing anything else:")
        print(f"  curl -s localhost:{args.port}/admin/conversations/{args.dst} "
              f"| python3 -m json.tool")
        print("Then: supervisorctl start compactor")
    else:
        print("DRY RUN — nothing was written. Read the counts above; if they")
        print("look right, re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
