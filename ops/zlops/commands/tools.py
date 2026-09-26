"""`zl tools [--apply] [--yes]` -- installs a short list of operator
conveniences the image deliberately omits: tmux, htop, sqlite3, iproute2
(`ss`/`ip`), less. NEVER upgrade or dist-upgrade -- only exactly this
install. NOT PERSISTENT: it lands on the container's writable overlay,
which resets on every restart, same as everything scripts/setup-sshd.py
does to /etc/ssh.
"""

from __future__ import annotations

from .. import common

_PACKAGES = ["tmux", "htop", "sqlite3", "iproute2", "less"]


def run(argv: list[str]) -> int:
    apply = "--apply" in argv
    yes = "--yes" in argv
    unknown = [a for a in argv if a not in ("--apply", "--yes")]
    if unknown:
        print(f"zl tools: unrecognized argument(s): {unknown}")
        return 2

    common.require_apply_confirmation(apply, yes, what=f"apt-get install {' '.join(_PACKAGES)}")

    print("NOTE: not persistent -- the container overlay resets on every pod restart; re-run after each one.")

    if apply:
        # Only --apply actually refreshes the package lists -- review LOW:
        # a dry run silently running `apt-get update` looked like a
        # read-only preview but was doing real network I/O against the
        # package mirror every time.
        update_code = common.run_stream(["apt-get", "update"])
        if update_code != 0:
            print(f"zl tools: apt-get update failed (exit {update_code})")
            return update_code
        install_flag = "-y"
    else:
        print(
            "(dry run -- NOT running `apt-get update`; using whatever package lists are already on "
            "disk, which may be stale. `apt-get install -s` below is itself a simulation: it never "
            "installs anything.)"
        )
        install_flag = "-s"

    code = common.run_stream(
        ["apt-get", "install", install_flag, "--no-install-recommends", *_PACKAGES]
    )
    if not apply and code == 0:
        print("(dry run -- pass --apply to actually install)")
    return code
