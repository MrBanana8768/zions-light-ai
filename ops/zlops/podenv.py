"""Where things live on the pod, in one place.

Every value is read from the same environment variable the image sets
(Dockerfile ENV lines, entrypoint.sh, compactor/backup.py's own defaults), so
the bundle and the services it inspects agree on a path by construction.
Nothing here writes; importing it is free.

The rest of `zlops` imports from here. Never hard-code one of these paths or
ports anywhere else in the bundle.
"""

from __future__ import annotations

import os
from pathlib import Path


def _proc1_environ() -> dict[str, str]:
    """supervisord (pid 1 in this image) carries the environment the services
    really run with. A RunPod Web Terminal shell can lack variables that
    entrypoint.sh only wrote into supervisord's environment, so a value missing
    from this shell is looked up there before falling back to the default.
    """
    try:
        raw = Path("/proc/1/environ").read_bytes()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(b"\0"):
        key, sep, value = part.partition(b"=")
        if sep:
            out[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return out


_PROC1 = _proc1_environ()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip() or _PROC1.get(name, "").strip()
    return value or default


def _port(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


# --- ports (Dockerfile ENV) --------------------------------------------------
VLLM_PORT = _port("VLLM_PORT", 8000)
COMPACTOR_PORT = _port("COMPACTOR_PORT", 8080)
OPENWEBUI_PORT = _port("OPENWEBUI_PORT", 3000)
STT_PORT = _port("STT_PORT", 9000)
TTS_PORT = _port("TTS_PORT", 9001)
SHADOW_PORT = _port("ZL_SHADOW_PORT", 8081)

# Admin endpoints are localhost-only, so always go through the loopback.
COMPACTOR_URL = f"http://127.0.0.1:{COMPACTOR_PORT}"
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"

# --- data (network volume, survives restarts) --------------------------------
DATA = Path("/data")
WEBUI_DIR = Path(_env("DATA_DIR", "/data/openwebui"))
STORE = Path(_env("COMPACTOR_STORAGE_ROOT", str(WEBUI_DIR / "compactor")))
BACKUPS = Path(_env("COMPACTOR_BACKUP_DIR", "/data/backups"))
LOG_DIR = Path(_env("LOG_DIR", "/data/logs"))
FORENSICS = DATA / "forensics"
SSH_PERSIST = DATA / "ssh"  # private host keys: never bundle, never print
HF_HOME = Path(_env("HF_HOME", "/data/models"))  # Dockerfile:378; clean-models.sh's own default
# `zl shadow`'s storage root (COMMANDS.md "Shadow deploy"). Never STORE itself
# (that would point a shadow compactor at her live memory) -- callers must
# refuse if this resolves equal to STORE.
SHADOW_STORE = DATA / "shadow-compactor"

# The bundle's own persistent home: the saved bootstrap, the pinned ref,
# the audit log, generated log bundles, uploaded wordban checks.
ZL_HOME = Path(_env("ZL_HOME", "/data/zl-ops"))
AUDIT_LOG = ZL_HOME / "audit.log"
BUNDLES = ZL_HOME / "bundles"

# --- container overlay (reset on every restart) -------------------------------
REPO = Path(_env("ZL_REPO", "/opt/zl-repo"))  # the clone-not-copy location
COMPACTOR_PKG = Path("/opt/compactor")
COMPACTOR_PY = Path("/opt/compactor-venv/bin/python")
WEBUI_PY = Path("/app/venv/bin/python")
VLLM_PY = Path("/opt/vllm-venv/bin/python")  # supervisord.conf [program:vllm]'s own interpreter
LOCAL_WEBUI_DB = Path("/var/lib/openwebui/webui.db")
SNAPSHOT_WEBUI_DB = WEBUI_DIR / "webui.db"

# --- supervisord programs (supervisord.conf) -----------------------------------
# v3.1.9.8+: `zl-ops` (ops/bootstrap.sh --boot, priority 5, autostart=true,
# autorestart=false, startsecs=0 -- see NOTE-ops-bundle-boot.md) is a
# one-shot boot-time installer, not a long-running service; it is expected
# to run once and EXIT, same shape as `selftest`.
SERVICES = ("vllm", "compactor", "webuidb-sync", "openwebui", "stt", "tts", "selftest", "backup", "zl-ops")
# Programs that are expected to be not RUNNING and are not a fault when so:
# selftest and zl-ops both run once at boot and exit; webuidb-sync only
# runs when WEBUI_DB_LOCAL=true.
ONE_SHOT_OR_OPTIONAL = frozenset({"selftest", "webuidb-sync", "zl-ops"})

# --- compactor's own control/pending roots (continuity.py) --------------------
# Cited here, never re-derived: `compactor/continuity.py`'s own
# `_default_control_root()`/`_default_pending_root()` (v3.1.9.8,
# continuity.py:88-96) resolve to `STORAGE_ROOT.parent / "compactor-control"`
# and `STORAGE_ROOT.parent / "compactor-pending"` -- SIBLINGS of
# STORAGE_ROOT (`memory.STORAGE_ROOT`, this module's own `STORE`), never
# nested inside it, deliberately: continuity.py's own module docstring
# (line 27) says the ledger lives "OUTSIDE STORAGE_ROOT and outside" any
# backup of it, so a restore of STORAGE_ROOT alone can never resurrect a
# stale ledger. Read straight from the same env vars continuity.py reads
# (`COMPACTOR_CONTROL_ROOT`/`COMPACTOR_PENDING_ROOT`), with the SAME
# fallback shape, so this bundle can never drift from what the compactor
# actually resolved on this pod -- never hard-code a path continuity.py
# itself computes.
CONTROL_ROOT = Path(_env("COMPACTOR_CONTROL_ROOT", str(STORE.parent / "compactor-control")))
PENDING_ROOT = Path(_env("COMPACTOR_PENDING_ROOT", str(STORE.parent / "compactor-pending")))

# The her-chat default the repair scripts themselves use (--chat/--conv).
# Kept here only so help text can name it; each script owns its own default.
GITHUB_REPO_URL = _env("ZL_REPO_URL", "https://github.com/MrBanana8768/zions-light-ai.git")

# --- RunPod pod identity (injected by the platform, not by the image) --------
# Present on a real pod, absent in a plain dev container; every reader must
# tolerate "". Used by `zl dash`'s header/SSH panel only — never a path.
RUNPOD_POD_ID = _env("RUNPOD_POD_ID", "")
RUNPOD_PUBLIC_IP = _env("RUNPOD_PUBLIC_IP", "")
RUNPOD_TCP_PORT_22 = _env("RUNPOD_TCP_PORT_22", "")


def scripts_dir() -> Path:
    """The `scripts/` directory of the clone this bundle is running from
    (ops/ and scripts/ are siblings), falling back to REPO/scripts.
    """
    here = Path(__file__).resolve().parents[2] / "scripts"
    return here if here.is_dir() else REPO / "scripts"


def detect_webui_db_local() -> tuple[bool, str]:
    """Is OpenWebUI's live DB on local disk? Delegates to
    scripts/_webui_live_path.py's detect_local_mode(), the one implementation
    the repair scripts already use (shell env, DATABASE_URL, /proc/1/environ,
    open-file scan). Never re-implement the fold here: two copies of that rule
    have drifted before. Returns (is_local, how_it_was_decided).
    """
    import importlib.util

    path = scripts_dir() / "_webui_live_path.py"
    spec = importlib.util.spec_from_file_location("_zl_webui_live_path", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detect_local_mode()


def live_webui_db() -> Path:
    is_local, _ = detect_webui_db_local()
    return LOCAL_WEBUI_DB if is_local else SNAPSHOT_WEBUI_DB
