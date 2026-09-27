"""`zl version` -- what is actually running here, right now: the clone's ref
and commit, the compactor build id, the OpenWebUI/vLLM package versions, the
WEBUI_DB_LOCAL mode (and why), and the ref bootstrap.sh last saved.

`collect()` is exposed for `zl bundle`'s meta/versions.txt -- one
implementation, not two copies drifting apart (memory.md).
"""

from __future__ import annotations

import hashlib
import subprocess

from .. import common, podenv
from . import hotfix as cmd_hotfix

REF_FILE = podenv.ZL_HOME / "ref"  # written by bootstrap.sh; kept in sync there


def _git_info(repo) -> dict[str, str | None]:
    if not (repo / ".git").exists():
        return {"ref": None, "commit": None, "dirty": None, "error": f"{repo} is not a git checkout"}
    out: dict[str, str | None] = {"error": None}
    try:
        r = common.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], timeout=10)
        out["ref"] = r.stdout.strip() if r.returncode == 0 else None
        r = common.run(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=10)
        out["commit"] = r.stdout.strip() if r.returncode == 0 else None
        r = common.run(["git", "-C", str(repo), "status", "--porcelain"], timeout=10)
        out["dirty"] = bool(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired) as e:
        out["error"] = str(e)
    return out


def _compactor_build_id() -> str | None:
    main_py = podenv.COMPACTOR_PKG / "main.py"
    try:
        digest = hashlib.sha256(main_py.read_bytes()).hexdigest()
    except OSError:
        return None
    return digest[:12]


def _pkg_version(py, name: str) -> str | None:
    try:
        r = common.run(
            [str(py), "-c", f"import importlib.metadata as m; print(m.version({name!r}))"],
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _saved_bootstrap_ref() -> str | None:
    try:
        return REF_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def collect() -> dict:
    git = _git_info(podenv.REPO)
    is_local, why = podenv.detect_webui_db_local()
    return {
        "repo": str(podenv.REPO),
        "git_ref": git["ref"],
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_error": git["error"],
        "compactor_build_id": _compactor_build_id(),
        "open_webui_version": _pkg_version(podenv.WEBUI_PY, "open-webui"),
        "vllm_version": _pkg_version(podenv.VLLM_PY, "vllm"),
        "webui_db_local": is_local,
        "webui_db_local_reason": why,
        "saved_bootstrap_ref": _saved_bootstrap_ref(),
        "backup_compression": cmd_hotfix.backup_compression_status(),
    }


def render(info: dict) -> str:
    lines = [
        f"repo: {info['repo']}",
        f"git ref: {info['git_ref']!s} commit: {info['git_commit']!s}"
        + (" (dirty)" if info["git_dirty"] else ""),
    ]
    if info["git_error"]:
        lines.append(f"git error: {info['git_error']}")
    lines += [
        f"compactor build id (sha256 of main.py, 12 hex): {info['compactor_build_id']}",
        f"open-webui: {info['open_webui_version']}",
        f"vllm: {info['vllm_version']}",
        f"WEBUI_DB_LOCAL: {info['webui_db_local']} ({info['webui_db_local_reason']})",
        f"saved bootstrap ref ({REF_FILE}): {info['saved_bootstrap_ref']}",
        f"backup compression: {info['backup_compression']}",
    ]
    return "\n".join(lines) + "\n"


def run(argv: list[str]) -> int:
    print(render(collect()), end="")
    return 0
