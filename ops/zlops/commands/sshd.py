"""`zl sshd [--apply] [--yes] [-- passthrough]` -- wraps
scripts/setup-sshd.py, the only way an operator gets a real shell into a
running pod (its own docstring: the image ships no openssh-server). Our own
`--apply`/`--yes` are the global confirm-before-mutating gate; everything
else (--port, --authorized-key[-file], --json, --force, --supervise,
--no-persist-host-keys) is the SCRIPT's own flag and goes after `--`
unchanged -- this wrapper never re-parses or re-validates them.
"""

from __future__ import annotations

from .. import common, podenv

_SCRIPT = podenv.scripts_dir() / "setup-sshd.py"
# setup-sshd.py's own argparse: --apply and --force are its write flags
# (--port/--authorized-key[-file]/--json/--supervise/--no-persist-host-keys
# are not). Review B2: passthrough carrying either must go through zl's own
# --apply gate.
_WRITE_FLAGS = ("--apply", "--force")
# setup-sshd.py's COMPLETE declared long-option list (its own argparse --
# review R2-B2: needed to normalize `--flag=value` and prefix abbreviations
# like `--appl`/`--forc` before the write-flag check above runs).
_ALL_FLAGS = (
    "--apply", "--authorized-key", "--authorized-key-file", "--force", "--json",
    "--no-persist-host-keys", "--port", "--supervise",
)


def run(argv: list[str]) -> int:
    own, passthrough = common.split_passthrough(argv)
    unknown = [a for a in own if a not in ("--apply", "--yes")]
    if unknown:
        print(f"zl sshd: unrecognized flag(s) before `--`: {unknown} (pass script flags after `--`)")
        return 2
    apply = "--apply" in own
    yes = "--yes" in own

    gate = common.refuse_passthrough_write_flags(
        passthrough, _WRITE_FLAGS, zl_apply=apply, what="install/harden openssh-server", all_flags=_ALL_FLAGS
    )
    if gate is not None:
        return gate

    common.require_apply_confirmation(apply, yes, what="install/harden openssh-server on this pod")

    script_args = (["--apply"] if apply else []) + passthrough
    # review B3: setup-sshd.py reads its own env directly.
    code = common.run_stream(
        [str(podenv.COMPACTOR_PY), str(_SCRIPT), *script_args], env=common.merged_environ()
    )

    if apply and code == 0:
        # podenv's own RUNPOD_PUBLIC_IP/RUNPOD_TCP_PORT_22 (env, falling back
        # to /proc/1/environ) -- the one place that looks these up, shared
        # with `zl dash`'s SSH panel.
        ip = podenv.RUNPOD_PUBLIC_IP
        port = podenv.RUNPOD_TCP_PORT_22
        if ip and port:
            print(f"\nconnect: ssh -p {port} root@{ip}")
            print(f"laptop:  zlpod init {ip} {port}")
        else:
            print("\nconnect: RUNPOD_PUBLIC_IP/RUNPOD_TCP_PORT_22 are not set here -- see the RunPod Connect tab")
            print("laptop:  zlpod init <ip-from-connect-tab> <port-from-connect-tab>")
    return code
