"""`zl models [...]` -- mostly a straight passthrough to clean-models.sh.
That script is already safe by default (lists only; deletion needs
--delete/--prune-others plus --yes or an interactive confirm), so this
wrapper adds no dry-run gate for read-only flags (--list, --compile-cache),
per the brief ("dry run default is its own"). It DOES gate the script's own
DELETE flags (review B2: a write flag of a wrapped script must go through
zl's own --apply, the same as every other command here) -- `--apply` is
this wrapper's own flag, stripped before the script ever sees it (the
script has no `--apply` of its own to collide with).
Prefers the freshly cloned ref's own copy (repo root) over the image-baked
/opt/clean-models.sh, so a fix shipped in a newer ref takes effect.
"""

from __future__ import annotations

from .. import common, podenv

_REPO_COPY = podenv.REPO / "clean-models.sh"
_IMAGE_COPY = "/opt/clean-models.sh"
# clean-models.sh's own argparse-equivalent (see its `case "$1" in` block):
# --delete/--delete=NAME and --prune-others actually remove files. Unlike
# every Python-argparse-based script this bundle wraps, clean-models.sh's
# `case "$1" in` requires an EXACT match (bash case has no abbreviation --
# `--del` falls through to its own `*) unknown argument` refusal, verified
# against clean-models.sh:73-90) -- so review R2-B2's prefix-abbreviation
# attack does not reach this script. `normalize_passthrough` is still run
# (for `--delete=NAME` and for defense-in-depth if the script ever grows
# abbreviation support), with `_ALL_FLAGS` matching its own case labels
# exactly.
_WRITE_FLAGS = ("--delete", "--prune-others")
_ALL_FLAGS = ("--list", "--prune-others", "--delete", "--compile-cache", "--yes", "--help")


def run(argv: list[str]) -> int:
    apply = "--apply" in argv
    rest = [a for a in argv if a != "--apply"]
    try:
        checked = common.normalize_passthrough(rest, _ALL_FLAGS)
    except common.AmbiguousPassthroughFlag as e:
        print(f"zl models: refusing -- {e}")
        return 2
    write_hit = [a for a in checked if a in _WRITE_FLAGS]
    if write_hit and not apply:
        print(
            f"zl models: refusing -- {write_hit} would delete model files; add zl's own --apply "
            "(the script's own --yes/interactive confirm still applies on top of that)"
        )
        return 2
    script = str(_REPO_COPY) if _REPO_COPY.is_file() else _IMAGE_COPY
    # review B3: clean-models.sh reads $HF_HOME (and friends) from its own
    # env, no fallback -- an SSH/Web Terminal session can miss a
    # customized value that only pid 1 carries.
    return common.run_stream(["bash", script, *rest], env=common.merged_environ())
