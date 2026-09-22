"""The real-image test for the four defects fixed in v3.1.9.6's operator
scripts (`scripts/backfill-records.py`, `scripts/import-history.py`) —
found by the OWNER running `backfill-records.py` on the real production
pod AFTER a 143-test green gate. The standing lesson: these scripts are
only ever run ON A POD, against an OLDER installed compactor, from a path
that is NOT the repo. Every unit test before this one mocked all three of
those facts away at once.

This suite is the one that would have caught it — it runs the real
scripts, unmodified, inside real containers started from the exact
PUBLISHED digest (`sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65`),
against a REAL v3.1.9 `compactor/backfill.py` (via `git archive v3.1.9
compactor`, bind-mounted over `/opt/compactor` — the same package shape a
pod that has never run v3.1.9.4 actually has) and a COPY (never the
backup itself, never mounted writable) of the real stale backfill records
verified on the 2026-09-22 production backup.

NEEDS, and SKIPS (exit 3) HONESTLY if any is missing, same convention as
test_tokenizer_contract.py's own `_skip`:
  - a real Docker daemon
  - the published image already pulled (`docker image inspect`)
  - `git`, with the `v3.1.9` tag reachable in this checkout
  - the real 2026-09-22 pod-export backup on this host

None of that is available inside the sandboxed `unit-tests` compose
service (`network_mode: none`, no docker socket) — this suite is
DELIBERATELY excluded from that default run (see scripts/run-tests.py's
`NEEDS_DOCKER`) so the documented 145/1-skipped baseline does not drift.
Run it directly, on the host:

    python compactor/test_real_image_operator_scripts.py
    python scripts/run-tests.py --real-image --only real_image

WHY A LENGTH COMPARISON IS NOT USED FOR THE STORE BYTE-COMPARE, and other
notes specific to each scenario, are inline at each test.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = Path("/home/drew/pod-exports/2026-09-22/backup")
IMAGE_DIGEST = "sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65"
IMAGE_REF = f"angreg/zions-light-ai@{IMAGE_DIGEST}"
VENV_PY = "/opt/compactor-venv/bin/python"
CHAT_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
DOCKER_TIMEOUT_S = 180

FAILED = []


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_in(needle, haystack, label):
    if needle not in haystack:
        print(f"FAIL {label}: {needle!r} not found")
        print(f"  --- full output ---\n{haystack}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_not_in(needle, haystack, label):
    if needle in haystack:
        print(f"FAIL {label}: {needle!r} unexpectedly present")
        print(f"  --- full output ---\n{haystack}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def _skip(reason: str) -> None:
    print("=" * 72)
    print("SKIPPED: test_real_image_operator_scripts.py")
    print(f"  reason: {reason}")
    print()
    print("  This suite needs a real Docker daemon, the published image")
    print("  already pulled, git with the v3.1.9 tag, and the real")
    print("  2026-09-22 pod-export backup on this host. Run it directly:")
    print("    python compactor/test_real_image_operator_scripts.py")
    print("=" * 72)
    sys.exit(0 if os.environ.get("COMPACTOR_ALLOW_FIXTURE_SKIP") else 3)


# ---------------------------------------------------------------------------
# Preflight — every precondition checked and named, never assumed.
# ---------------------------------------------------------------------------

def _run(args, timeout=DOCKER_TIMEOUT_S, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kw)


def _preflight() -> None:
    if shutil.which("docker") is None:
        _skip("no `docker` binary on PATH")
    r = _run(["docker", "version"], timeout=15)
    if r.returncode != 0:
        _skip(f"`docker version` failed: {(r.stderr or r.stdout).strip()[:200]}")
    r = _run(["docker", "image", "inspect", IMAGE_REF], timeout=15)
    if r.returncode != 0:
        _skip(f"published image not pulled locally: {IMAGE_REF}")
    if shutil.which("git") is None:
        _skip("no `git` binary on PATH")
    r = _run(["git", "rev-parse", "--verify", "v3.1.9"], cwd=REPO_ROOT, timeout=15)
    if r.returncode != 0:
        _skip("tag v3.1.9 is not reachable in this checkout (shallow clone?)")
    if not BACKUP_ROOT.is_dir():
        _skip(f"real backup not present on this host: {BACKUP_ROOT}")
    if not (BACKUP_ROOT / "webui.db").is_file():
        _skip(f"real backup missing webui.db: {BACKUP_ROOT / 'webui.db'}")
    if not (BACKUP_ROOT / "compactor" / "facts").is_dir():
        _skip(f"real backup missing compactor/facts: {BACKUP_ROOT}")


# ---------------------------------------------------------------------------
# Fixtures — built once, into a scratch tree. The backup itself is only
# ever READ: webui.db is bind-mounted read-only (never copied — 500MB+ —
# and the script itself only ever opens it `mode=ro`); facts/summaries are
# physically COPIED into scratch before any container touches them, so
# nothing a bug in this run does can ever reach the real backup.
# ---------------------------------------------------------------------------

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="compactor-test-real-image-"))
V319_PKG = _TMP_ROOT / "v319pkg" / "compactor"
STORE_DIR = _TMP_ROOT / "store"
SINGLE_FILE_DIR = _TMP_ROOT / "single-file-copy"


def _build_v319_package() -> None:
    """A faithful v3.1.9 compactor/ tree via `git archive` — the real
    shape of a pod that has never run v3.1.9.4 (verified: it has
    `is_stale`/`_STALE_SECONDS`/`atomic_write_json` but not
    `_MAX_BACKFILL_ATTEMPTS`/`_backoff_ready` — see backfill-records.py's
    own module docstring)."""
    dest = V319_PKG.parent
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        f"git archive v3.1.9 compactor | tar -x -C {dest}",
        shell=True, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0 or not V319_PKG.is_dir():
        print("FAIL building the v3.1.9 compactor package via git archive")
        print(r.stdout, r.stderr)
        sys.exit(1)
    assert_true(
        not (V319_PKG / "backfill.py").read_text(encoding="utf-8")
        .count("_MAX_BACKFILL_ATTEMPTS"),
        "sanity: the archived v3.1.9 backfill.py really lacks _MAX_BACKFILL_ATTEMPTS",
    )


def _seed_store() -> None:
    """A COPY of the real facts/summaries for CHAT_ID and every stale
    record — never the backup itself, never mounted writable at the
    source."""
    (STORE_DIR / "facts").mkdir(parents=True, exist_ok=True)
    (STORE_DIR / "summaries").mkdir(parents=True, exist_ok=True)
    for p in (BACKUP_ROOT / "compactor" / "facts").glob("*.json"):
        shutil.copy2(p, STORE_DIR / "facts" / p.name)
    summaries_src = BACKUP_ROOT / "compactor" / "summaries"
    for name in (f"{CHAT_ID}.json", f"{CHAT_ID}.archive.json"):
        p = summaries_src / name
        if p.is_file():
            shutil.copy2(p, STORE_DIR / "summaries" / name)


def _seed_single_file_copies() -> None:
    """The "copied just this one file to /data/scripts/" scenario the
    real operator error came from — a bare file, no sibling compactor,
    no scripts/ directory at all."""
    SINGLE_FILE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "backfill-records.py", SINGLE_FILE_DIR)
    shutil.copy2(REPO_ROOT / "scripts" / "import-history.py", SINGLE_FILE_DIR)


def _store_digest() -> str:
    """A single hash of every byte under STORE_DIR, path included — the
    byte-compare in test_clone_invocation_matches_real_pod_and_writes_nothing."""
    h = hashlib.sha256()
    for p in sorted(STORE_DIR.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(STORE_DIR)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _docker_run(mounts: list[tuple[str, str, str]], image_args: list[str],
                 entrypoint: str = VENV_PY) -> subprocess.CompletedProcess:
    """`mounts` is [(host_path, container_path, mode)], mode "ro" or "rw"."""
    args = ["docker", "run", "--rm", "--entrypoint", entrypoint]
    for host, cont, mode in mounts:
        args += ["-v", f"{host}:{cont}:{mode}"]
    args += [IMAGE_REF] + image_args
    return _run(args)


# ---------------------------------------------------------------------------
# 1. A pre-v3.1.9.4 pod gets the actionable capability error, never a
#    traceback (Defect 2).
# ---------------------------------------------------------------------------

def test_pre_v3194_pod_gives_actionable_error_not_a_traceback():
    print("\n[test] backfill-records.py against a real v3.1.9 package: actionable error, not a traceback")
    # Nested one level deeper than the trap the module docstring warns
    # about: HERE (/work/opt/scripts) has a PARENT (/work/opt) distinct
    # from /opt, so HERE.parent/"compactor" (candidate 2) and /opt/compactor
    # (candidate 3, the bind-mounted v3.1.9 package) are genuinely
    # different paths — candidate 2 correctly fails to exist, and
    # resolution falls through to candidate 3 for real, rather than the
    # two candidates silently coinciding the way they would for a script
    # sitting bare at /opt/x.py.
    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(REPO_ROOT / "scripts"), "/work/opt/scripts", "ro"),
            (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
        ],
        ["/work/opt/scripts/backfill-records.py", "--store", "/data/openwebui/compactor"],
    )
    out = r.stdout + r.stderr
    assert_eq(r.returncode, 1, "exits 1 (a refusal, not a crash)")
    assert_not_in("Traceback", out, "no Python traceback")
    assert_not_in("AttributeError", out, "no raw AttributeError")
    assert_in("_MAX_BACKFILL_ATTEMPTS", out, "names the missing symbol")
    assert_in("predates v3.1.9.4", out, "names the pod's version state")
    assert_in("git clone", out, "prints the clone command")


def test_import_history_pre_v3194_pod_dry_run_is_not_a_traceback():
    print("\n[test] import-history.py against a real v3.1.9 package: dry run works or refuses cleanly, never a traceback")
    # import-history.py's own required symbols (load_state, maybe_rollup,
    # the fingerprint primitives) are, unlike backfill's, present all the
    # way back through v3.1.9 (verified) — so this proves defect 1's
    # fallback-to-/opt/compactor resolution works for THIS script too,
    # producing a real, correct dry-run report rather than any crash.
    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(REPO_ROOT / "scripts"), "/work/opt/scripts", "ro"),
            (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
            (str(BACKUP_ROOT / "webui.db"), "/data/openwebui/webui.db", "ro"),
        ],
        [
            "/work/opt/scripts/import-history.py",
            "--webui-db", "/data/openwebui/webui.db",
            "--chat-id", CHAT_ID,
            "--store", "/data/openwebui/compactor",
            "--json",
        ],
    )
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "no Python traceback")
    assert_eq(r.returncode, 3, "dry run against the real backlog finds work due -> exit 3")
    payload = json.loads(r.stdout)
    # The exact numbers the architect verified once this defect was fixed
    # (see CHANGELOG.md v3.1.9.6 "Fixed" — Defect 3): roughly 57 L1 / 6 L2
    # / 1 L3 / ~64 calls.
    assert_eq(payload["l1_chunks_due_estimate"], 57, "57 L1 chunks due")
    assert_eq(payload["l2_folds_due_estimate"], 6, "6 L2 folds due")
    assert_eq(payload["l3_refreshes_due_estimate"], 1, "1 L3 refresh due")
    assert_eq(payload["estimated_real_vllm_calls"], 64, "64 estimated calls")
    assert_true(
        payload["turns_found"] < payload["recorded_position_before"],
        "the reconstructed transcript IS shorter than recorded_position "
        "(the exact shape the old length-only guard refused) yet this "
        "still proceeded — Defect 3's content check, not length, decided it",
    )


# ---------------------------------------------------------------------------
# 2. The clone invocation reproduces the real pod's own verified numbers,
#    and writes nothing (Defect 1 + the "Supported invocation").
# ---------------------------------------------------------------------------

def test_clone_invocation_matches_real_pod_and_writes_nothing():
    print("\n[test] the clone invocation: 4 would-resume / 16 leave / exit 3, and byte-identical store")
    # "Clone" is stood in for by bind-mounting THIS checkout's own
    # scripts/ and compactor/ together at /opt/zl-repo — the real clone
    # command (git clone ... /opt/zl-repo) supplies exactly this layout;
    # a git clone step itself needs network access this sandboxed host
    # test should not depend on, and the layout is the only thing under
    # test here (a real `git clone` was separately verified in
    # OPERATIONS.md/RUNPOD_DEPLOY.md's own history against the real pod).
    before = _store_digest()
    r = _docker_run(
        [
            (str(REPO_ROOT / "scripts"), "/opt/zl-repo/scripts", "ro"),
            (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
            (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
        ],
        ["/opt/zl-repo/scripts/backfill-records.py", "--store", "/data/openwebui/compactor"],
    )
    after = _store_digest()
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "no Python traceback")
    assert_eq(r.returncode, 3, "dry run with would-resume records exits 3")
    # The exact numbers verified on the real production pod (see the
    # module docstrings' "Supported invocation").
    assert_in("4 would-resume", out, "4 would-resume (the real pod's number)")
    assert_in("16 leave", out, "16 leave (the real pod's number)")
    assert_in("0 needs-review", out, "0 needs-review (the real pod's number)")
    assert_eq(before, after, "not one byte under the store changed")


# ---------------------------------------------------------------------------
# 3. Running from a bare copy (no compactor package anywhere reachable)
#    gives the actionable multi-path error, never a traceback (Defect 1).
# ---------------------------------------------------------------------------

def test_bare_copy_with_no_package_anywhere_gives_multipath_error():
    print("\n[test] backfill-records.py copied alone, with /opt/compactor also removed: the multi-path error")
    # Simulates the ORIGINAL real operator error precisely: just the one
    # file copied to /data/scripts/, and — to exercise the true
    # every-candidate-failed path rather than falling back to the image's
    # own /opt/compactor — that path is removed first. `rm -rf` runs
    # against the container's own writable overlay; the image is
    # untouched.
    r = _docker_run(
        [
            (str(SINGLE_FILE_DIR / "backfill-records.py"), "/data/scripts/backfill-records.py", "ro"),
            (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
        ],
        [
            "-c",
            "rm -rf /opt/compactor && exec /opt/compactor-venv/bin/python "
            "/data/scripts/backfill-records.py --store /data/openwebui/compactor",
        ],
        entrypoint="sh",
    )
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "no Python traceback")
    assert_eq(r.returncode, 1, "exits 1 (a refusal, not a crash)")
    assert_in("no compactor package found", out, "names the failure plainly")
    assert_in("Tried, in order", out, "lists every path tried")
    assert_in("/data/compactor", out, "names the exact wrong path the real incident hit")
    assert_in("/opt/compactor", out, "also tried the image layout")
    assert_in("git clone", out, "prints the clone command")


def test_import_history_bare_copy_with_no_package_gives_multipath_error():
    print("\n[test] import-history.py copied alone, with /opt/compactor also removed: the multi-path error")
    r = _docker_run(
        [
            (str(SINGLE_FILE_DIR / "import-history.py"), "/data/scripts/import-history.py", "ro"),
            (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
            (str(BACKUP_ROOT / "webui.db"), "/data/openwebui/webui.db", "ro"),
        ],
        [
            "-c",
            "rm -rf /opt/compactor && exec /opt/compactor-venv/bin/python "
            "/data/scripts/import-history.py --webui-db /data/openwebui/webui.db "
            f"--chat-id {CHAT_ID} --store /data/openwebui/compactor",
        ],
        entrypoint="sh",
    )
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "no Python traceback")
    assert_eq(r.returncode, 1, "exits 1 (a refusal, not a crash)")
    assert_in("no compactor package found", out, "names the failure plainly")
    assert_in("Tried, in order", out, "lists every path tried")
    assert_in("git clone", out, "prints the clone command")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _preflight()
    try:
        _build_v319_package()
        _seed_store()
        _seed_single_file_copies()

        test_pre_v3194_pod_gives_actionable_error_not_a_traceback()
        test_import_history_pre_v3194_pod_dry_run_is_not_a_traceback()

        test_clone_invocation_matches_real_pod_and_writes_nothing()

        test_bare_copy_with_no_package_anywhere_gives_multipath_error()
        test_import_history_bare_copy_with_no_package_gives_multipath_error()

        print("\nAll real-image operator-script tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
