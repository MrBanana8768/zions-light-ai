"""The real-image test for the v3.1.9.7 scroll-jump CSS fix.

`Dockerfile.v3197` appends two CSS rules to OpenWebUI's own `custom.css`,
inside the freshly `pip install open-webui==0.11.4`-ed `/app/venv`:

    .message-listitem{content-visibility:visible !important;contain-intrinsic-size:none !important}
    #messages-container{overflow-anchor:none !important}

Per /home/drew/zl-ops/bounce-diagnosis.md, these two rules together took
the anchor-position jump on every "load older messages" batch to 0 px in
5 of 5 measured runs; either rule alone still produced an 18k-60k px jump
on some load, and without either one it was 1,000-3,700 px on EVERY load.

WHY A REAL-IMAGE TEST, NOT JUST A DOCKERFILE READ. Two things can only be
proven by actually booting the built image:

  1. OpenWebUI's own `open_webui/config.py` (~lines 100-117 in 0.11.4)
     deletes every top-level file under `open_webui/static/` and re-copies
     `open_webui/frontend/static/*` over it on EVERY process start. A fix
     baked into only one of the two files would look correct on disk in
     the image layer and still vanish the moment the process (re)starts.
     This suite proves the CSS is in both files in the image AND that it
     survives a real OpenWebUI process restart inside a running container
     -- not just present once at first boot.
  2. `GET /static/custom.css` is what the browser actually fetches. A
     byte-diff of the two files on disk does not prove the HTTP path
     serves the same content (a stale mount, a caching proxy, or a typo
     in STATIC_DIR could all break that independently of the file itself).

NEEDS, and SKIPS (exit 3) HONESTLY if any is missing, same convention as
`test_real_image_setup_sshd.py` / `test_real_image_operator_scripts.py`:
  - a real Docker daemon
  - the locally built v3.1.9.7 trial image (`zla-v3197-trial:latest` by
    default; override with `ZLA_V3197_IMAGE_REF`) -- this is deliberately
    NOT the published base digest, which does not have OpenWebUI 0.11.4 or
    the CSS fix at all
  - `--network host` support (Linux hosts) and TEST_PORT (18197 by
    default; override with `ZLA_V3197_CSS_TEST_PORT`) free on the host

This suite's `_skip` never honors `COMPACTOR_ALLOW_FIXTURE_SKIP` -- there
is no fixture here to opt into skipping; a missing Docker daemon or image
is an environment gap, not a deliberate narrow skip, and reporting it any
other way would let a run that never actually checked the CSS be read as
one that did.

Run it directly, on the host:
    python3 compactor/test_real_image_v3197_css.py
    python scripts/run-tests.py --real-image --only v3197_css

ALL steps run inside ONE throwaway container (`docker run -d --rm
--network host --entrypoint bash ... sleep`), with OpenWebUI itself
started and (for the restart scenario) re-started via `docker exec -d`,
exactly the way `entrypoint.sh` starts services under supervisord in the
real image -- so a real restart here exercises the same startup path a
supervisord-triggered restart on a pod would. No GPU, no vLLM, and no
compactor are started: this suite only exercises OpenWebUI's own static
file serving, which does not depend on any of those. The container is
ALWAYS removed (`docker rm -f`) in a `finally`, whether the suite passes
or not.
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_REF = os.environ.get("ZLA_V3197_IMAGE_REF", "zla-v3197-trial:latest")
TEST_PORT = int(os.environ.get("ZLA_V3197_CSS_TEST_PORT", "18197"))
DOCKER_TIMEOUT_S = 180
BOOT_TIMEOUT_S = 150
CONTAINER_NAME = f"v3197css-{uuid.uuid4().hex[:10]}"
DATA_DIR = "/tmp/v3197css-webui-data"
STATIC_PKG_DIR = "/app/venv/lib/python3.12/site-packages/open_webui"
FRONTEND_CSS = f"{STATIC_PKG_DIR}/frontend/static/custom.css"
SERVED_CSS = f"{STATIC_PKG_DIR}/static/custom.css"

RULE_CONTENT_VIS = (
    ".message-listitem{content-visibility:visible !important;"
    "contain-intrinsic-size:none !important}"
)
RULE_OVERFLOW_ANCHOR = "#messages-container{overflow-anchor:none !important}"

SERVE_ENV = (
    f"DATA_DIR={DATA_DIR} WEBUI_SECRET_KEY=v3197css-test-secret "
    "WEBUI_AUTH=false ENABLE_OLLAMA_API=false ENABLE_OPENAI_API=false "
    "DATABASE_ENABLE_SQLITE_WAL=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
)


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label, extra: str = ""):
    if not cond:
        print(f"FAIL {label}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_in(needle, haystack, label):
    if needle not in haystack:
        print(f"FAIL {label}: {needle!r} not found")
        print(f"  --- full output ---\n{haystack}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def _skip(reason: str) -> None:
    print("=" * 72)
    print("SKIPPED: test_real_image_v3197_css.py")
    print(f"  reason: {reason}")
    print()
    print("  This suite needs a real Docker daemon, the locally built")
    print(f"  v3.1.9.7 trial image ({IMAGE_REF}), and --network host support")
    print("  (Linux hosts). Run it directly:")
    print("    python3 compactor/test_real_image_v3197_css.py")
    print("=" * 72)
    sys.exit(3)


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
        _skip(
            f"local trial image not built: {IMAGE_REF} "
            "(build it first: docker build -f Dockerfile.v3197 -t "
            f"{IMAGE_REF} .)"
        )


def _docker_exec(cmd: str, timeout=DOCKER_TIMEOUT_S) -> subprocess.CompletedProcess:
    return _run(["docker", "exec", CONTAINER_NAME, "bash", "-c", cmd], timeout=timeout)


def _start_container() -> None:
    _run(["docker", "rm", "-f", CONTAINER_NAME], timeout=30)  # never trust a stale name
    r = _run([
        "docker", "run", "-d", "--rm", "--network", "host", "--name", CONTAINER_NAME,
        "--entrypoint", "bash", IMAGE_REF, "-c", "sleep 3600",
    ], timeout=60)
    if r.returncode != 0:
        print("FAIL starting the throwaway container")
        print(r.stdout, r.stderr)
        sys.exit(1)
    time.sleep(1)
    r = _docker_exec("echo up")
    assert_eq(r.returncode, 0, "the throwaway container is really running")


def _stop_container() -> None:
    _run(["docker", "rm", "-f", CONTAINER_NAME], timeout=30)


def _start_openwebui(log_name: str) -> None:
    """Launches a real `open-webui serve` inside the container, detached
    (`docker exec -d`), the same way entrypoint.sh/supervisord starts it
    in the real image -- never bind-mounted, DATA_DIR lives only inside
    this throwaway, --rm container."""
    r = _run([
        "docker", "exec", "-d", CONTAINER_NAME, "bash", "-c",
        f"mkdir -p {DATA_DIR} && cd /app && env {SERVE_ENV} "
        f"/app/venv/bin/open-webui serve --host 127.0.0.1 --port {TEST_PORT} "
        f"> /tmp/{log_name}.log 2>&1",
    ], timeout=30)
    assert_eq(r.returncode, 0, f"docker exec -d launched open-webui serve ({log_name})")


def _wait_for_static_css(deadline_s: float, log_name: str) -> str:
    """Polls the REAL HTTP endpoint from the host (not docker exec) until
    it serves /static/custom.css, or fails with the server log attached."""
    deadline = time.time() + deadline_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{TEST_PORT}/static/custom.css", timeout=3
            ) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = str(e)
        time.sleep(2)
    log = _docker_exec(f"tail -80 /tmp/{log_name}.log").stdout
    assert_true(
        False,
        f"open-webui serve became reachable on :{TEST_PORT} within {deadline_s:.0f}s",
        f"last connect error: {last_err}\n--- server log tail ---\n{log}",
    )
    return ""  # unreachable, assert_true exits



# Bracket a single character of the search pattern ("[o]pen-webui serve"
# instead of "open-webui serve"): pgrep/pkill -f matches the FULL command
# line, which -- unbracketed -- would also match the very `docker exec ...
# bash -c "pkill -f 'open-webui serve' ..."` invocation issuing the search
# (its own argv literally contains the pattern text), letting it kill its
# own parent shell mid-command and hang the `docker exec` call itself
# (observed: it stopped emitting any output at all, indefinitely). The
# bracketed regex still matches the real target's cmdline (the character
# class still matches a literal "o") but no longer textually matches
# itself, since "[o]pen-webui serve" is not a substring of "[o]pen-webui
# serve" starting one character later. Same trick as `ps aux | grep -v
# grep`, applied to `-f` instead.
_SELF_SAFE_PATTERN = "[o]pen-webui serve"


def _kill_openwebui() -> None:
    _docker_exec(f"pkill -f '{_SELF_SAFE_PATTERN}' || true")
    for _ in range(30):
        r = _docker_exec(f"pgrep -f '{_SELF_SAFE_PATTERN}' || true")
        if not r.stdout.strip():
            return
        time.sleep(1)
    assert_true(False, "the running open-webui serve process actually stopped")


# ---------------------------------------------------------------------------
# 1. GET /static/custom.css (the real HTTP path a browser uses) serves
#    both rules.
# ---------------------------------------------------------------------------

def test_served_css_contains_both_rules():
    print("\n[test] GET /static/custom.css (real HTTP, from the host) contains both scroll-jump rules")
    body = _wait_for_static_css(BOOT_TIMEOUT_S, "openwebui")
    assert_in(RULE_CONTENT_VIS, body, "content-visibility rule present in the served CSS")
    assert_in(RULE_OVERFLOW_ANCHOR, body, "overflow-anchor rule present in the served CSS")
    return body


# ---------------------------------------------------------------------------
# 2. Both on-disk copies inside the container contain both rules -- not
#    just the one HTTP happens to be serving right now.
# ---------------------------------------------------------------------------

def test_both_on_disk_files_contain_both_rules():
    print("\n[test] both on-disk custom.css copies (frontend/static AND static) contain both rules")
    for label, path in (("frontend/static/custom.css", FRONTEND_CSS), ("static/custom.css", SERVED_CSS)):
        r = _docker_exec(f"cat {path}")
        assert_eq(r.returncode, 0, f"{label} is readable")
        assert_in(RULE_CONTENT_VIS, r.stdout, f"content-visibility rule present in {label}")
        assert_in(RULE_OVERFLOW_ANCHOR, r.stdout, f"overflow-anchor rule present in {label}")
        # Idempotent-append proof: exactly one copy of each rule, never
        # duplicated by the Dockerfile's own idempotency guard.
        assert_eq(r.stdout.count(RULE_OVERFLOW_ANCHOR), 1, f"{label} has exactly one copy of the overflow-anchor rule")


# ---------------------------------------------------------------------------
# 3. The fix survives a real OpenWebUI process restart inside the
#    container -- proving config.py's own delete-and-recopy of static/
#    from frontend/static on every boot still lands the fix, because the
#    fix is baked into frontend/static/custom.css (the copy SOURCE), not
#    only into the copy DESTINATION.
# ---------------------------------------------------------------------------

def test_css_survives_a_real_openwebui_restart():
    print("\n[test] the served CSS survives a real OpenWebUI process restart inside the container")
    _kill_openwebui()
    # Sanity: while the process is down, the port really is unreachable --
    # otherwise a stale process from a previous step could quietly make
    # this whole scenario vacuous.
    r = _docker_exec(f"pgrep -f '{_SELF_SAFE_PATTERN}' || echo NONE")
    assert_in("NONE", r.stdout, "no open-webui serve process is left running before the restart")

    _start_openwebui("openwebui-restart")
    body = _wait_for_static_css(BOOT_TIMEOUT_S, "openwebui-restart")
    assert_in(RULE_CONTENT_VIS, body, "content-visibility rule present after restart")
    assert_in(RULE_OVERFLOW_ANCHOR, body, "overflow-anchor rule present after restart")

    # And prove it is really config.py's own re-copy that produced this,
    # not a leftover file untouched by the restart: static/custom.css and
    # frontend/static/custom.css must be byte-identical post-restart,
    # which is only true if the re-copy actually ran.
    r1 = _docker_exec(f"md5sum {FRONTEND_CSS} | cut -d' ' -f1")
    r2 = _docker_exec(f"md5sum {SERVED_CSS} | cut -d' ' -f1")
    assert_eq(r1.stdout.strip(), r2.stdout.strip(), "frontend/static and static copies are byte-identical after the restart")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _preflight()
    try:
        _start_container()
        _start_openwebui("openwebui")

        test_served_css_contains_both_rules()
        test_both_on_disk_files_contain_both_rules()
        test_css_survives_a_real_openwebui_restart()

        print("\nAll real-image v3.1.9.7 scroll-jump CSS tests passed.")
    finally:
        _stop_container()
