"""Where THIS pod's data lives, for the standalone operator scripts (v3.1.9.10).

Why this exists. Every operator script used to default to the literal
`/data/openwebui/...`, `/data/backups`, `/data/forensics`. On a pod whose data
lives elsewhere (the v3.1.9.10 new pod: DATA_DIR=/data/v10/openwebui, beside a
still-live old pod that owns /data/openwebui on the same network volume) a script
run without every path flag would silently aim a WRITER at the other pod's
files. And a RunPod Web Terminal shell may not carry the pod's environment at
all, so "read the env var" alone is not enough.

Resolution, per name: (1) this process's own environment; (2) when pid 1 is
supervisord (a pod), supervisord's environment (/proc/1/environ), which is what
the services really run with; (3) the built-in default, unchanged from before,
derived from DATA_DIR / COMPACTOR_BACKUP_DIR where the running system derives
it the same way (compactor/memory.py, backup.py, uploads_backup.py).

Only path names are read here. Standalone: stdlib only, imports nothing from the
compactor package. `PROC1_COMM` / `PROC1_ENVIRON` are module globals so a test
can point them at fake files.
"""

from __future__ import annotations

import os

PROC1_COMM = "/proc/1/comm"
PROC1_ENVIRON = "/proc/1/environ"

DEFAULT_DATA_DIR = "/data/openwebui"
DEFAULT_BACKUP_DIR = "/data/backups"
DEFAULT_FORENSICS = "/data/forensics"
DEFAULT_LOG_DIR = "/data/logs"
DEFAULT_ZL_HOME = "/data/zl-ops"


def _pid1_environ() -> dict[str, str]:
    try:
        with open(PROC1_COMM, "r", encoding="utf-8", errors="replace") as fh:
            if not fh.read().strip().startswith("supervisord"):
                return {}
        with open(PROC1_ENVIRON, "rb") as fh:
            raw = fh.read()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(b"\0"):
        key, sep, value = part.partition(b"=")
        if sep:
            out[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return out


def get(name: str, default: str = "") -> str:
    """Own environment, then supervisord's, then `default`. Blank counts as unset."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    value = _pid1_environ().get(name, "").strip()
    return value or default


def data_dir() -> str:
    return get("DATA_DIR", DEFAULT_DATA_DIR)


def store() -> str:
    return get("COMPACTOR_STORAGE_ROOT", data_dir().rstrip("/") + "/compactor")


def snapshot_db() -> str:
    return get("WEBUI_SNAPSHOT_DB", data_dir().rstrip("/") + "/webui.db")


def backups() -> str:
    return get("COMPACTOR_BACKUP_DIR", DEFAULT_BACKUP_DIR)


def uploads() -> str:
    return get("COMPACTOR_UPLOADS_DIR", data_dir().rstrip("/") + "/uploads")


def uploads_backup() -> str:
    return get("COMPACTOR_UPLOADS_BACKUP_DIR", backups().rstrip("/") + "/uploads")


def forensics() -> str:
    return get("WEBUI_DB_FORENSICS") or get("WEBUI_DB_QUARANTINE", DEFAULT_FORENSICS)


def log_dir() -> str:
    return get("LOG_DIR", DEFAULT_LOG_DIR)


def zl_home() -> str:
    return get("ZL_HOME", DEFAULT_ZL_HOME)


def wordban_file() -> str:
    return get("COMPACTOR_WORDBAN_FILE", zl_home().rstrip("/") + "/word-ban/active.json")
