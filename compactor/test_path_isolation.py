"""
Tests for v3.1.9.10's PATH ISOLATION: two live pods on one network volume must never
write each other's files (docs/PATHS-v31910.md is the audit; this file keeps it true).

SYNTHETIC ONLY: paths are temp directories; nothing reads a real pod.

Covers:
  * every default that used to be a bare /data literal now follows the env var the
    running system derives it from (DATA_DIR, COMPACTOR_BACKUP_DIR, ZL_HOME), and an
    unset environment still gives TODAY's value;
  * envcfg.adopt_pod_env(): an operator shell that lacks the pod's environment
    adopts the PATH names from supervisord (pid 1), never a secret, never a
    WEBUI_DB_ALLOW_* flag, never over a value the shell already has;
  * scripts/_podpaths.py (the same resolution for the standalone operator scripts);
  * ops/zlops/podenv.py's forensics/ssh/shadow paths follow their env vars;
  * entrypoint.sh's torch.compile cache directory is overridable;
  * a static sweep: a NEW hardcoded /data literal in non-test code fails this test
    until it is given an env override and listed in docs/PATHS-v31910.md.

Run: python test_path_isolation.py
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_TMP = Path(tempfile.mkdtemp(prefix="zions-path-isolation-test-"))
FAILED = 0


def assert_eq(a, b, label):
    global FAILED
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        FAILED += 1
        return
    print(f"  ok   {label}")


def assert_true(c, label):
    global FAILED
    if not c:
        print(f"FAIL {label}")
        FAILED += 1
        return
    print(f"  ok   {label}")


_CLEAN_ENV_DROP = (
    "DATA_DIR", "COMPACTOR_STORAGE_ROOT", "COMPACTOR_BACKUP_DIR", "COMPACTOR_UPLOADS_BACKUP_DIR", "COMPACTOR_UPLOADS_DIR",
    "ZL_HOME", "COMPACTOR_WORDBAN_FILE", "WEBUI_SNAPSHOT_DB", "WEBUI_DB_QUARANTINE", "WEBUI_DB_FORENSICS", "WEBUI_LOCAL_DB",
    "SETUP_SSHD_HOST_KEY_STORE", "ZL_SHADOW_STORE", "LOG_DIR", "COMPACTOR_CONTROL_ROOT", "COMPACTOR_PENDING_ROOT",
    "DATABASE_URL", "WEBUI_DB_LOCAL", "VLLM_COMPILE_CACHE",
)


def probe(code: str, **env) -> str:
    """Run `code` in a fresh interpreter with compactor/ and scripts/ importable and a
    clean path environment plus `env`. Returns stdout."""
    e = {k: v for k, v in os.environ.items() if k not in _CLEAN_ENV_DROP}
    e.update(env)
    e["PYTHONPATH"] = f"{_HERE}{os.pathsep}{_REPO / 'scripts'}"
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, cwd=str(_TMP))
    if r.returncode != 0:
        print(r.stderr[-800:])
    return r.stdout.strip()


# ---------------------------------------------------------------------------

def test_defaults_are_unchanged_when_nothing_is_set():
    assert_eq(probe("import memory; print(memory.STORAGE_ROOT)"), "/data/openwebui/compactor", "memory.STORAGE_ROOT default")
    assert_eq(probe("import degrade; print(degrade._WATCH_PATH)"), "/data/openwebui/compactor", "degrade watch path default")
    assert_eq(probe("import uploads_backup as u; print(u.UPLOADS_BACKUP_DIR)"), "/data/backups/uploads", "uploads mirror default")
    assert_eq(probe("import webuidb as w; print(w.SNAPSHOT_DB)"), "/data/openwebui/webui.db", "webuidb snapshot default")
    assert_eq(probe("import wordban_runtime as w; print(w.WORDBAN_FILE)"), "/data/zl-ops/word-ban/active.json", "word-ban file default")
    assert_eq(probe("import _podpaths as p; print(p.store(), p.snapshot_db(), p.backups(), p.forensics(), p.zl_home())"),
              "/data/openwebui/compactor /data/openwebui/webui.db /data/backups /data/forensics /data/zl-ops", "_podpaths defaults")


def test_defaults_follow_data_dir_and_friends():
    d = "/vol/v10/openwebui"
    assert_eq(probe("import memory; print(memory.STORAGE_ROOT)", DATA_DIR=d), d + "/compactor", "memory.STORAGE_ROOT follows DATA_DIR")
    assert_eq(probe("import degrade; print(degrade._WATCH_PATH)", DATA_DIR=d), d + "/compactor", "degrade follows DATA_DIR")
    assert_eq(probe("import memory; print(memory.STORAGE_ROOT)", DATA_DIR=d, COMPACTOR_STORAGE_ROOT="/x/store"), "/x/store", "the explicit store wins")
    assert_eq(probe("import uploads_backup as u; print(u.UPLOADS_BACKUP_DIR)", COMPACTOR_BACKUP_DIR="/vol/v10/backups"),
              "/vol/v10/backups/uploads", "the uploads mirror follows COMPACTOR_BACKUP_DIR")
    assert_eq(probe("import uploads_backup as u; print(u.UPLOADS_BACKUP_DIR)", COMPACTOR_BACKUP_DIR="/vol/v10/backups", COMPACTOR_UPLOADS_BACKUP_DIR="/y"),
              "/y", "the explicit mirror dir wins")
    assert_eq(probe("import uploads_backup as u; print(u.UPLOADS_DIR)", DATA_DIR=d), d + "/uploads", "uploads follow DATA_DIR")
    assert_eq(probe("import webuidb as w; print(w.SNAPSHOT_DB)", DATA_DIR=d), d + "/webui.db", "webuidb snapshot follows DATA_DIR")
    assert_eq(probe("import webuidb as w; print(w.SNAPSHOT_DB)", DATA_DIR=d, WEBUI_SNAPSHOT_DB="/z/s.db"), "/z/s.db", "the explicit snapshot wins")
    assert_eq(probe("import wordban_runtime as w; print(w.WORDBAN_FILE)", ZL_HOME="/vol/v10/zl-ops"), "/vol/v10/zl-ops/word-ban/active.json",
              "the word-ban file follows ZL_HOME")
    assert_eq(probe("import wordban_runtime as w; print(w.WORDBAN_FILE)", ZL_HOME="/vol/v10/zl-ops", COMPACTOR_WORDBAN_FILE="/w/a.json"), "/w/a.json",
              "the explicit word-ban file wins")
    assert_eq(probe("import continuity, memory; print(continuity.CONTROL_ROOT, continuity.PENDING_ROOT)", DATA_DIR=d),
              f"{d}/compactor-control {d}/compactor-pending", "control/pending roots follow DATA_DIR through the store")
    # backup.py is not edited in this release; it already followed DATA_DIR for the store
    assert_eq(probe("import backup; print(backup.STORAGE_ROOT, backup.BACKUP_DIR)", DATA_DIR=d, COMPACTOR_BACKUP_DIR="/vol/v10/backups"),
              f"{d}/compactor /vol/v10/backups", "backup.py store/backups follow their env vars")


def test_envcfg_adopts_pod_path_env_only():
    c = _TMP / "comm"
    e = _TMP / "environ"
    c.write_text("supervisord\n")
    e.write_bytes(b"\0".join([
        b"DATA_DIR=/vol/v10/openwebui", b"WEBUI_DB_LOCAL=true", b"COMPACTOR_BACKUP_DIR=/vol/v10/backups",
        b"WEBUI_SECRET_KEY=must-never-be-copied", b"CLOUDFLARE_TUNNEL_TOKEN=must-never-be-copied",
        b"WEBUI_DB_ALLOW_ROW_LOSS=1", b"WEBUI_DB_ALLOW_OLDER_GENERATION=1", b"COMPACTOR_BACKUP_INTERVAL_HOURS=1",
        b"LOG_DIR=/vol/v10/logs", b"ZL_HOME=   ", b"NOEQUALSIGN",
    ]) + b"\0")
    code = (
        "import os, envcfg; "
        "os.environ.pop('DATA_DIR', None); os.environ.pop('LOG_DIR', None); os.environ['COMPACTOR_BACKUP_DIR']='/mine'; "
        "for_names = ['WEBUI_SECRET_KEY','CLOUDFLARE_TUNNEL_TOKEN','WEBUI_DB_ALLOW_ROW_LOSS','WEBUI_DB_ALLOW_OLDER_GENERATION','COMPACTOR_BACKUP_INTERVAL_HOURS','ZL_HOME']; "
        "[os.environ.pop(n, None) for n in for_names]; "
        f"got = envcfg.adopt_pod_env({str(c)!r}, {str(e)!r}); "
        "print(sorted(got)); print(os.environ.get('DATA_DIR'), os.environ.get('COMPACTOR_BACKUP_DIR'), os.environ.get('LOG_DIR')); "
        "print([n for n in for_names if n in os.environ])"
    )
    out = probe(code).splitlines()
    assert_eq(out[0], "['DATA_DIR', 'LOG_DIR', 'WEBUI_DB_LOCAL']", "only absent PATH/gate names are adopted")
    assert_eq(out[1], "/vol/v10/openwebui /mine /vol/v10/logs", "an adopted value fills the gap; a value the shell already has wins")
    assert_eq(out[2], "[]", "secrets, WEBUI_DB_ALLOW_* flags, tuning knobs and blank values are never adopted")
    c.write_text("bash\n")
    out = probe(code).splitlines()
    assert_eq(out[0], "[]", "when pid 1 is not supervisord (a test container, a laptop) nothing is adopted")
    assert_eq(probe("import envcfg; print(envcfg.adopt_pod_env('/nonexistent/a', '/nonexistent/b'))"), "[]", "unreadable /proc files are not an error")
    names = probe("import envcfg; print(' '.join(envcfg.POD_PATH_ENV))").split()
    assert_true(not any(n.startswith("WEBUI_DB_ALLOW") for n in names), "no override flag is on the allowlist")
    assert_true(not any(n.endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")) or "SECRET" in n or "TOKEN" in n for n in names), "no secret is on the allowlist")


_FAKE_PID1 = (
    "import builtins, io\n"
    "_real = builtins.open\n"
    "def _fake(p, *a, **k):\n"
    "    if p == '/proc/1/comm': return io.StringIO('supervisord\\n')\n"
    "    if p == '/proc/1/environ': return io.BytesIO(b'DATA_DIR=/vol/v10/openwebui\\0COMPACTOR_BACKUP_DIR=/vol/v10/backups\\0"
    "WEBUI_DB_QUARANTINE=/vol/v10/forensics\\0ZL_HOME=/vol/v10/zl-ops\\0WEBUI_SECRET_KEY=never\\0')\n"
    "    return _real(p, *a, **k)\n"
    "builtins.open = _fake\n"
)


def test_import_time_adoption_reaches_every_cli_including_backup_py():
    """A bare shell on a pod (pid 1 = supervisord, no DATA_DIR etc. in the shell): importing the
    modules the operator CLIs are made of resolves to THE POD's paths. backup.py is not edited by this
    release: it works because its first compactor import is envcfg, which runs before it reads any env."""
    code = _FAKE_PID1 + (
        "import memory, backup, webuidb, uploads_backup, wordban_runtime, os\n"
        "print(memory.STORAGE_ROOT); print(backup.STORAGE_ROOT, backup.BACKUP_DIR); print(webuidb.SNAPSHOT_DB, webuidb.QUARANTINE)\n"
        "print(uploads_backup.UPLOADS_BACKUP_DIR); print(wordban_runtime.WORDBAN_FILE); print(os.environ.get('WEBUI_SECRET_KEY') == 'never')\n"
    )
    out = probe(code).splitlines()
    assert_eq(out[0], "/vol/v10/openwebui/compactor", "memory.py resolves to the pod's store from a bare shell")
    assert_eq(out[1], "/vol/v10/openwebui/compactor /vol/v10/backups", "backup.py (unedited) resolves to the pod's store and backup dir")
    assert_eq(out[2], "/vol/v10/openwebui/webui.db /vol/v10/forensics", "webuidb.py resolves to the pod's snapshot and forensics dir")
    assert_eq(out[3], "/vol/v10/backups/uploads", "the uploads mirror resolves to the pod's backup dir")
    assert_eq(out[4], "/vol/v10/zl-ops/word-ban/active.json", "the word-ban file resolves to the pod's ZL_HOME")
    assert_eq(out[5], "False", "a secret in supervisord's environment is not adopted")
    # the shell's own value still wins
    out = probe(code, DATA_DIR="/mine").splitlines()
    assert_eq(out[0], "/mine/compactor", "a DATA_DIR the shell already has wins")


def test_podpaths_resolution_order():
    c = _TMP / "comm2"
    e = _TMP / "environ2"
    c.write_text("supervisord\n")
    e.write_bytes(b"DATA_DIR=/vol/v10/openwebui\0COMPACTOR_BACKUP_DIR=/vol/v10/backups\0WEBUI_DB_QUARANTINE=/vol/v10/forensics\0")
    code = ("import _podpaths as p; p.PROC1_COMM=%r; p.PROC1_ENVIRON=%r; "
            "print(p.store(), p.snapshot_db(), p.backups(), p.uploads_backup(), p.forensics(), p.uploads())") % (str(c), str(e))
    assert_eq(probe(code), "/vol/v10/openwebui/compactor /vol/v10/openwebui/webui.db /vol/v10/backups /vol/v10/backups/uploads /vol/v10/forensics /vol/v10/openwebui/uploads",
              "with no env in the shell, supervisord's environment is used")
    assert_eq(probe(code, DATA_DIR="/shell/data"), "/shell/data/compactor /shell/data/webui.db /vol/v10/backups /vol/v10/backups/uploads /vol/v10/forensics /shell/data/uploads",
              "the shell's own value beats supervisord's")
    assert_eq(probe(code, WEBUI_DB_FORENSICS="/f"), "/vol/v10/openwebui/compactor /vol/v10/openwebui/webui.db /vol/v10/backups /vol/v10/backups/uploads /f /vol/v10/openwebui/uploads",
              "WEBUI_DB_FORENSICS beats WEBUI_DB_QUARANTINE")
    c.write_text("sleep\n")
    assert_eq(probe(code), "/data/openwebui/compactor /data/openwebui/webui.db /data/backups /data/backups/uploads /data/forensics /data/openwebui/uploads",
              "not on a pod: today's defaults")


def test_operator_scripts_default_from_the_pod_env():
    d = "/vol/v10/openwebui"
    env = dict(DATA_DIR=d, COMPACTOR_BACKUP_DIR="/vol/v10/backups", ZL_HOME="/vol/v10/zl-ops", WEBUI_DB_QUARANTINE="/vol/v10/forensics")

    def load(name, attr):
        code = ("import importlib.util, sys; "
                f"s = importlib.util.spec_from_file_location('m', {str(_REPO / 'scripts' / name)!r}); m = importlib.util.module_from_spec(s); "
                f"s.loader.exec_module(m); print({attr})")
        return probe(code, **env)

    assert_eq(load("backfill-records.py", "m.DEFAULT_STORE"), d + "/compactor", "backfill-records --store default")
    assert_eq(load("import-history.py", "m.DEFAULT_STORE"), d + "/compactor", "import-history --store default")
    assert_eq(load("clean-decoration.py", "m.DEFAULT_STORE"), d + "/compactor", "clean-decoration --store default")
    assert_eq(load("uploads-backup.py", "m.DEFAULT_BACKUP_ROOT, m.DEFAULT_UPLOADS_DIR, m.DEFAULT_DB"),
              f"/vol/v10/backups/uploads {d}/uploads {d}/webui.db", "uploads-backup defaults")
    assert_eq(load("recover-webui-db.py", "m.DB, m.FORENSICS"), f"{d}/webui.db /vol/v10/forensics", "recover-webui-db defaults")
    assert_eq(load("fix-stale-unfinished.py", "m.forensics_base()"), "/vol/v10/forensics", "fix-stale-unfinished forensics")
    assert_eq(load("word-ban-deploy.py", "m.DEFAULT_FILE"), "/vol/v10/zl-ops/word-ban/active.json", "word-ban-deploy file")
    assert_eq(load("_webui_live_path.py", "m.SNAPSHOT_DB, m.FORENSICS_ROOT"), f"{d}/webui.db /vol/v10/forensics", "_webui_live_path snapshot/forensics")
    assert_eq(load("switch-webui-db-to-local.py", "m.SNAPSHOT_DB"), f"{d}/webui.db", "switch-webui-db-to-local snapshot")
    assert_eq(load("backfill-records.py", "m.DEFAULT_STORE").startswith("/data"), False, "no script default still aims at /data/openwebui when DATA_DIR is set")


def test_podenv_paths_follow_env():
    code = "from zlops import podenv; print(podenv.FORENSICS, podenv.SSH_PERSIST, podenv.SHADOW_STORE)"
    base = probe(code.replace("from zlops", "sys_path_fix = __import__('sys').path.insert(0, %r); from zlops" % str(_REPO / "ops")))
    assert_eq(base, "/data/forensics /data/ssh /data/shadow-compactor", "podenv defaults are unchanged")
    out = probe(code.replace("from zlops", "sys_path_fix = __import__('sys').path.insert(0, %r); from zlops" % str(_REPO / "ops")),
                WEBUI_DB_QUARANTINE="/v/f", SETUP_SSHD_HOST_KEY_STORE="/v/s", ZL_SHADOW_STORE="/v/sh")
    assert_eq(out, "/v/f /v/s /v/sh", "podenv forensics/ssh/shadow follow their env vars")


def test_entrypoint_compile_cache_is_overridable():
    text = (_REPO / "entrypoint.sh").read_text()
    assert_true('export VLLM_COMPILE_CACHE="${VLLM_COMPILE_CACHE:-/data/vllm-compile-cache}"' in text, "the default is today's directory")
    assert_true('ln -s "${VLLM_COMPILE_CACHE}" /root/.cache/vllm' in text, "the symlink target follows the variable")
    assert_true('mkdir -p "${HF_HOME}" "${DATA_DIR}" "${VLLM_COMPILE_CACHE}"' in text, "the directory is created from the variable")
    assert_true("ln -s /data/vllm-compile-cache" not in text, "no literal symlink target remains")
    for env, want in (({}, "/data/vllm-compile-cache"), ({"VLLM_COMPILE_CACHE": "/data/v10/vllm-compile-cache"}, "/data/v10/vllm-compile-cache")):
        line = re.search(r'^export VLLM_COMPILE_CACHE=.*$', text, re.M).group(0)
        e = {k: v for k, v in os.environ.items() if k != "VLLM_COMPILE_CACHE"}
        e.update(env)
        r = subprocess.run(["sh", "-c", line + '; printf %s "$VLLM_COMPILE_CACHE"'], capture_output=True, text=True, env=e)
        assert_eq(r.stdout, want, f"the export resolves to {want}")


_PATH_RX = re.compile(r"^/(data|var/lib/openwebui|opt/zl-repo)(/\S*)?$")

# Every remaining hardcoded path literal in non-test code, each one a DEFAULT that has
# an env override (docs/PATHS-v31910.md, one row each). A NEW literal fails this test.
_ALLOWED_LITERALS = {
    ("compactor/backup.py", "/data/backups"), ("compactor/backup.py", "/data/openwebui"),
    ("compactor/backup.py", "/var/lib/openwebui/.backup-staging"),
    ("compactor/degrade.py", "/data/openwebui"), ("compactor/health.py", "/data/openwebui"),
    ("compactor/health.py", "/var/lib/openwebui/webui.db"), ("compactor/memory.py", "/data/openwebui"),
    ("compactor/tokens.py", "/data/models"), ("compactor/uploads_backup.py", "/data/backups"),
    ("compactor/uploads_backup.py", "/data/openwebui"), ("compactor/webuidb.py", "/data/forensics"),
    ("compactor/webuidb.py", "/data/openwebui"), ("compactor/webuidb.py", "/var/lib/openwebui/webui.db"),
    ("compactor/wordban_runtime.py", "/data/zl-ops"),
    ("ops/zlops/podenv.py", "/data"), ("ops/zlops/podenv.py", "/data/backups"), ("ops/zlops/podenv.py", "/data/logs"),
    ("ops/zlops/podenv.py", "/data/models"), ("ops/zlops/podenv.py", "/data/openwebui"), ("ops/zlops/podenv.py", "/data/zl-ops"),
    ("ops/zlops/podenv.py", "/opt/zl-repo"), ("ops/zlops/podenv.py", "/var/lib/openwebui/webui.db"),
    ("scripts/_podpaths.py", "/data/backups"), ("scripts/_podpaths.py", "/data/forensics"), ("scripts/_podpaths.py", "/data/logs"),
    ("scripts/_podpaths.py", "/data/openwebui"), ("scripts/_podpaths.py", "/data/zl-ops"),
    ("scripts/_webui_live_path.py", "/var/lib/openwebui/webui.db"),
    ("scripts/migrate-webui-sqlite-to-pg.py", "/var/lib/openwebui/webui.db"),
    ("scripts/pod-migrate.py", "/var/lib/openwebui/webui.db"),
    ("scripts/setup-sshd.py", "/data/ssh"),
    ("scripts/switch-webui-db-to-local.py", "/var/lib/openwebui/webui.db"),
}


def _literals() -> set:
    seen = set()
    files = (sorted((_REPO / "compactor").glob("*.py")) + sorted((_REPO / "scripts").glob("*.py"))
             + sorted((_REPO / "ops").rglob("*.py")))
    for f in files:
        rel = f.relative_to(_REPO).as_posix()
        if f.name.startswith("test_") or "/tests/" in rel:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        docs = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body \
                    and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
                docs.add(id(n.body[0].value))
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and _PATH_RX.match(n.value.strip()):
                seen.add((rel, n.value))
    return seen


def test_no_new_hardcoded_data_paths():
    new = sorted(_literals() - _ALLOWED_LITERALS)
    assert_eq(new, [], "no new hardcoded /data (or /var/lib/openwebui) literal: give it an env override, list it in docs/PATHS-v31910.md")
    gone = sorted(_ALLOWED_LITERALS - _literals())
    assert_eq(gone, [], "every allowed literal still exists (trim this list when a literal is removed)")


def test_the_audit_document_covers_every_adopted_variable():
    doc = (_REPO / "docs" / "PATHS-v31910.md").read_text(encoding="utf-8")
    names = probe("import envcfg; print(' '.join(envcfg.POD_PATH_ENV))").split()
    missing = [n for n in names if n not in doc]
    assert_eq(missing, [], "docs/PATHS-v31910.md names every variable in envcfg.POD_PATH_ENV")
    for n in ("VLLM_COMPILE_CACHE", "WEBUI_SECRET_KEY", "COMPACTOR_WORDBAN_FILE", "ZL_HOME", "LOG_DIR"):
        assert_true(n in doc, f"the audit names {n}")


def _all():
    return [
        test_defaults_are_unchanged_when_nothing_is_set,
        test_defaults_follow_data_dir_and_friends,
        test_envcfg_adopts_pod_path_env_only,
        test_import_time_adoption_reaches_every_cli_including_backup_py,
        test_podpaths_resolution_order,
        test_operator_scripts_default_from_the_pod_env,
        test_podenv_paths_follow_env,
        test_entrypoint_compile_cache_is_overridable,
        test_no_new_hardcoded_data_paths,
        test_the_audit_document_covers_every_adopted_variable,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            print(f"\n[{t.__name__}]")
            t()
        print()
        if FAILED:
            print(f"{FAILED} FAILED")
            sys.exit(1)
        print("All path-isolation tests passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
