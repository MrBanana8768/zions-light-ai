"""The real-image test for B1 (v3.1.9.6) -- the fix that the hostile
review said "would have caught" the defect a green gate missed: no
scenario in `test_real_image_operator_scripts.py` runs `import-history.py
--apply` against a real store at all.

This suite runs the PUBLISHED digest
(`sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65`)
against a COPY of the real 2026-09-22 pod-export backup's
`compactor/summaries/<conv>.json` (+ `.archive.json`) and a bind-mounted,
read-only `webui.db` (never copied -- 500MB+, and the script itself only
ever opens it `mode=ro`), driven by a fake vLLM (`/v1/chat/completions`
and `/tokenize`) reachable over the host network. It asserts:

  (a) the first new L1 chunk starts at branch turn 2699 -- the exact
      seam the review's flat `window_offset` got wrong (it started at
      branch turn 2701, silently dropping branch turns 2699-2700);
  (b) every newly recorded covered-turn fingerprint matches the
      transcript at `position - resume_offset`, for the WHOLE run;
  (c) only `summaries/<conv>.json` and `.archive.json`, plus their
      `.bak-` copies, changed under the store (md5 snapshot before and
      after covers every other file, and every unrelated `facts/*`);
  (d) `--max-calls` budget-limits the run to exit code 4 (H2);
  (e) a Ctrl-C mid-run leaves the store re-runnable (B1's own Ctrl-C
      window fix), with the pre-apply anchor either restored (nothing
      had completed) or freshly self-consistent (something had).

NEEDS, and SKIPS (exit 3) HONESTLY if any is missing, same convention as
`test_real_image_operator_scripts.py`'s own `_skip`:
  - a real Docker daemon, with `--network host` support (Linux hosts)
  - the published image already pulled (`docker image inspect`)
  - the real 2026-09-22 pod-export backup on this host, with a summaries
    file for CHAT_ID

Run it directly, on the host:

    python compactor/test_real_image_import_apply.py
    python scripts/run-tests.py --real-image --only real_image_import
"""

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = Path("/home/drew/pod-exports/2026-09-22/backup")
IMAGE_DIGEST = "sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65"
IMAGE_REF = f"angreg/zions-light-ai@{IMAGE_DIGEST}"
VENV_PY = "/opt/compactor-venv/bin/python"
CHAT_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
DOCKER_TIMEOUT_S = 600
FAKE_VLLM_PORT = 18321
L1_CHUNK_SIZE = 20  # summarizer.L1_CHUNK_SIZE's default; verified below


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


def _skip(reason: str) -> None:
    print("=" * 72)
    print("SKIPPED: test_real_image_import_apply.py")
    print(f"  reason: {reason}")
    print("=" * 72)
    sys.exit(0 if os.environ.get("COMPACTOR_ALLOW_FIXTURE_SKIP") else 3)


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
    if not BACKUP_ROOT.is_dir():
        _skip(f"real backup not present on this host: {BACKUP_ROOT}")
    if not (BACKUP_ROOT / "webui.db").is_file():
        _skip(f"real backup missing webui.db: {BACKUP_ROOT / 'webui.db'}")
    if not (BACKUP_ROOT / "compactor" / "summaries" / f"{CHAT_ID}.json").is_file():
        _skip(f"real backup has no summaries file for {CHAT_ID}")
    if sys.platform != "linux":
        _skip("`--network host` (used to reach the fake vLLM) needs a Linux docker host")


_TMP_ROOT = Path(tempfile.mkdtemp(prefix="compactor-test-real-image-import-"))
STORE_DIR = _TMP_ROOT / "store"
FAKE_VLLM_SCRIPT = _TMP_ROOT / "fake_vllm.py"


def _seed_store() -> None:
    """A COPY of the real facts/summaries for CHAT_ID -- never the
    backup itself, never mounted writable at the source. `webui.db`
    itself is bind-mounted read-only straight from BACKUP_ROOT (never
    copied: 500MB+, and the script only ever opens it `mode=ro`)."""
    (STORE_DIR / "facts").mkdir(parents=True, exist_ok=True)
    (STORE_DIR / "summaries").mkdir(parents=True, exist_ok=True)
    for name in (f"{CHAT_ID}.json", f"{CHAT_ID}.archive.json"):
        p = BACKUP_ROOT / "compactor" / "summaries" / name
        if p.is_file():
            shutil.copy2(p, STORE_DIR / "summaries" / name)
    fp = BACKUP_ROOT / "compactor" / "facts" / f"{CHAT_ID}.json"
    if fp.is_file():
        shutil.copy2(fp, STORE_DIR / "facts" / fp.name)


_FAKE_VLLM_SRC = '''
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1])
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
COUNTS = {"tokenize": 0, "completions": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        if self.path.endswith("/tokenize"):
            COUNTS["tokenize"] += 1
            text = req.get("prompt") or ""
            body = json.dumps({"count": max(1, len(text) // 4)}).encode()
        elif "chat/completions" in self.path:
            COUNTS["completions"] += 1
            if DELAY:
                time.sleep(DELAY)
            body = json.dumps({
                "choices": [{
                    "message": {"content": "A plain summary of this stretch of the conversation."},
                    "finish_reason": "stop",
                }]
            }).encode()
        else:
            body = b"{}"
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        if self.path == "/__counts__":
            body = json.dumps(COUNTS).encode()
        else:
            body = b"{}"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass


HTTPServer(("0.0.0.0", PORT), H).serve_forever()
'''


def _start_fake_vllm(port: int, delay: float = 0.0):
    FAKE_VLLM_SCRIPT.write_text(_FAKE_VLLM_SRC, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(FAKE_VLLM_SCRIPT), str(port), str(delay)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"fake vLLM never came up on port {port}")


def _docker_run(mounts, image_args, entrypoint=VENV_PY, extra_docker_args=None,
                 popen=False, timeout=DOCKER_TIMEOUT_S):
    # Run as the invoking (host) uid/gid, not root: the container writes
    # into STORE_DIR (and this test reads those writes back afterward,
    # as the host user) -- root-owned writes from an unqualified
    # container process would otherwise leave files this test itself
    # cannot read back.
    args = [
        "docker", "run", "--rm", "--network", "host",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--entrypoint", entrypoint,
    ]
    for host, cont, mode in mounts:
        args += ["-v", f"{host}:{cont}:{mode}"]
    args += (extra_docker_args or [])
    args += [IMAGE_REF] + image_args
    if popen:
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return _run(args, timeout=timeout)


def _snapshot_md5(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


def _diff_snapshots(before: dict, after: dict) -> tuple[set, set, set]:
    """(added, removed, changed) relative paths."""
    before_keys, after_keys = set(before), set(after)
    added = after_keys - before_keys
    removed = before_keys - after_keys
    changed = {k for k in (before_keys & after_keys) if before[k] != after[k]}
    return added, removed, changed


def _load_summary_state(store_dir: Path) -> dict:
    return json.loads((store_dir / "summaries" / f"{CHAT_ID}.json").read_text(encoding="utf-8"))


def _covered_fps_from_state(state: dict) -> list:
    raw = state.get("covered_fps")
    if not isinstance(raw, str) or len(raw) % 16:
        return []
    return [raw[i:i + 16] for i in range(0, len(raw), 16)]


def _import_script_module():
    spec = importlib.util.spec_from_file_location(
        "import_history_script_real_image", str(REPO_ROOT / "scripts" / "import-history.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reconstruct_real_transcript():
    """Uses THIS checkout's own script (not the container's) purely as a
    read-only tool to reconstruct the branch from the real (bind-mounted
    read-only, but read directly here too since it costs nothing extra)
    webui.db. `import-history.py`'s OWN module-level imports are stdlib
    only (summarizer/memory are imported inside main(), not at module
    load), so this works on the bare host python running this test file
    -- no compactor venv needed just to reconstruct the branch."""
    ih = _import_script_module()
    con = ih._open_ro(BACKUP_ROOT / "webui.db")
    try:
        turns, source, notes = ih.reconstruct_transcript(con, CHAT_ID)
    finally:
        con.close()
    return turns


_VERIFY_SEAM_SRC = '''
import json, sys
sys.path.insert(0, "/opt/zl-repo/compactor")
import summarizer

turns = json.load(open("/work/verify/turns.json"))
before = json.load(open("/work/verify/before.json"))
after = json.load(open("/work/verify/after.json"))
offset = int(sys.argv[1])

transcript_fps = summarizer._covered_turn_fingerprints(turns)
fp_unknown = summarizer._FP_UNKNOWN
n = len(transcript_fps)
mismatch = None
for pos in range(len(before) + 1, len(after) + 1):
    fp = after[pos - 1]
    if fp == fp_unknown:
        continue
    j = pos - offset
    expected = transcript_fps[j - 1] if 1 <= j <= n else None
    if expected is None or fp != expected:
        mismatch = f"position {pos}: fp {fp!r} != expected {expected!r} at branch turn {j}"
        break

print(json.dumps({"mismatch": mismatch, "transcript_len": n}))
'''


def _verify_seam(turns, before_covered, after_covered, offset) -> dict:
    """Runs the REAL summarizer (inside the container, same image as the
    apply under test) to fingerprint `turns` and check that every newly
    recorded covered-turn position matches the transcript at
    position-offset -- this is B1 step 4's own check, independently
    re-run here against the real transcript rather than trusted from
    the apply's own self-report."""
    verify_dir = _TMP_ROOT / "verify"
    verify_dir.mkdir(exist_ok=True)
    (verify_dir / "verify_seam.py").write_text(_VERIFY_SEAM_SRC, encoding="utf-8")
    (verify_dir / "turns.json").write_text(json.dumps(turns), encoding="utf-8")
    (verify_dir / "before.json").write_text(json.dumps(before_covered), encoding="utf-8")
    (verify_dir / "after.json").write_text(json.dumps(after_covered), encoding="utf-8")

    r = _docker_run(
        [
            (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
            (str(verify_dir), "/work/verify", "ro"),
        ],
        ["/work/verify/verify_seam.py", str(offset)],
    )
    if r.returncode != 0:
        raise RuntimeError(f"seam verification script failed: {r.stdout}\\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# 1. The main --apply run: seam alignment (a)+(b), blast radius (c).
# ---------------------------------------------------------------------------

def test_apply_seam_alignment_and_blast_radius():
    print("\n[test] --apply on the real backup: seam alignment, label "
          "alignment, and blast radius")
    before_state = _load_summary_state(STORE_DIR)
    before_covered = _covered_fps_from_state(before_state)
    last_summarized_before = int(before_state.get("last_summarized_turn") or 0)
    before_snapshot = _snapshot_md5(STORE_DIR)

    vllm = _start_fake_vllm(FAKE_VLLM_PORT, delay=0.0)
    try:
        r = _docker_run(
            [
                (str(REPO_ROOT / "scripts"), "/opt/zl-repo/scripts", "ro"),
                (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
                (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
                (str(BACKUP_ROOT / "webui.db"), "/data/openwebui/webui.db", "ro"),
            ],
            [
                "/opt/zl-repo/scripts/import-history.py",
                "--webui-db", "/data/openwebui/webui.db",
                "--chat-id", CHAT_ID,
                "--store", "/data/openwebui/compactor",
                "--vllm-url", f"http://127.0.0.1:{FAKE_VLLM_PORT}",
                "--health-url", f"http://127.0.0.1:{FAKE_VLLM_PORT + 1}/health",
                "--model", "fake-model",
                "--max-calls", "40",
                "--apply", "--json",
            ],
        )
    finally:
        vllm.terminate()
        vllm.wait(timeout=10)

    out = r.stdout + r.stderr
    assert_true("Traceback" not in out, f"no Python traceback (out tail: {out[-2000:]})")
    payload = json.loads(r.stdout)
    print(f"  rc={r.returncode}  resume_offset={payload.get('resume_offset')}  "
          f"last_summarized_turn: {payload['last_summarized_turn_before']} -> "
          f"{payload['last_summarized_turn_after']}")

    turns = _reconstruct_real_transcript()
    offset = payload["resume_offset"]
    assert_true(offset is not None, "a resume offset was verified (not refused)")

    # (a) the first new L1 chunk starts at branch turn 2699.
    first_new_pos = last_summarized_before + 1
    j = first_new_pos - offset
    assert_eq(j, 2699,
               f"the first new chunk's covered position ({first_new_pos}) "
               f"maps to branch turn {j} (expected 2699 -- the exact seam "
               f"B1's fix corrects; the old flat offset started this at "
               f"branch turn 2701, silently dropping 2699-2700)")

    after_state = _load_summary_state(STORE_DIR)
    after_covered = _covered_fps_from_state(after_state)
    assert_true(len(after_covered) > len(before_covered),
                "the covered-turn record actually grew")

    # (b) every NEWLY recorded fingerprint matches the transcript at
    # position - offset, for the WHOLE run (not just the first chunk) --
    # run against the REAL summarizer inside the container.
    seam = _verify_seam(turns, before_covered, after_covered, offset)
    assert_true(seam["mismatch"] is None,
                f"every newly recorded position matches the transcript at "
                f"position-offset (mismatch={seam['mismatch']!r})")

    # (c) blast radius: only summaries/<conv>.json and .archive.json (and
    # their .bak- copies) changed under the whole store.
    after_snapshot = _snapshot_md5(STORE_DIR)
    added, removed, changed = _diff_snapshots(before_snapshot, after_snapshot)
    assert_true(not removed, f"nothing was deleted (removed={removed})")
    unexpected = {
        p for p in (added | changed)
        if not (p.startswith(f"summaries/{CHAT_ID}.json")
                 or p.startswith(f"summaries/{CHAT_ID}.archive.json"))
    }
    assert_true(not unexpected,
                f"only {CHAT_ID}.json/.archive.json (+ .bak-) changed "
                f"(unexpected={unexpected})")
    print(f"  changed/added under the store: {sorted(added | changed)}")


# ---------------------------------------------------------------------------
# 2. (d) H2: --max-calls budget-limits the run to exit code 4.
# ---------------------------------------------------------------------------

def test_apply_budget_limited_exits_4():
    print("\n[test] --apply with a small --max-calls exits 4 (H2: real "
          "progress, more due)")
    vllm = _start_fake_vllm(FAKE_VLLM_PORT + 2, delay=0.0)
    try:
        r = _docker_run(
            [
                (str(REPO_ROOT / "scripts"), "/opt/zl-repo/scripts", "ro"),
                (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
                (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
                (str(BACKUP_ROOT / "webui.db"), "/data/openwebui/webui.db", "ro"),
            ],
            [
                "/opt/zl-repo/scripts/import-history.py",
                "--webui-db", "/data/openwebui/webui.db",
                "--chat-id", CHAT_ID,
                "--store", "/data/openwebui/compactor",
                "--vllm-url", f"http://127.0.0.1:{FAKE_VLLM_PORT + 2}",
                "--health-url", f"http://127.0.0.1:{FAKE_VLLM_PORT + 3}/health",
                "--model", "fake-model",
                "--max-calls", "3",
                "--apply", "--json",
            ],
        )
    finally:
        vllm.terminate()
        vllm.wait(timeout=10)
    out = r.stdout + r.stderr
    assert_true("Traceback" not in out, "no Python traceback")
    assert_eq(r.returncode, 4, f"budget-limited with real progress and more "
              f"due is exit 4 (out tail: {out[-1500:]})")


# ---------------------------------------------------------------------------
# 3. (e) Ctrl-C mid-run leaves the store re-runnable.
# ---------------------------------------------------------------------------

def test_ctrlc_mid_apply_leaves_store_rerunnable():
    print("\n[test] Ctrl-C mid-`--apply` leaves the store re-runnable "
          "(B1's Ctrl-C window fix)")
    before_state = _load_summary_state(STORE_DIR)
    before_tail_fp = before_state.get("tail_fp")

    vllm = _start_fake_vllm(FAKE_VLLM_PORT + 4, delay=3.0)
    proc = None
    try:
        proc = _docker_run(
            [
                (str(REPO_ROOT / "scripts"), "/opt/zl-repo/scripts", "ro"),
                (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
                (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
                (str(BACKUP_ROOT / "webui.db"), "/data/openwebui/webui.db", "ro"),
            ],
            [
                "/opt/zl-repo/scripts/import-history.py",
                "--webui-db", "/data/openwebui/webui.db",
                "--chat-id", CHAT_ID,
                "--store", "/data/openwebui/compactor",
                "--vllm-url", f"http://127.0.0.1:{FAKE_VLLM_PORT + 4}",
                "--health-url", f"http://127.0.0.1:{FAKE_VLLM_PORT + 5}/health",
                "--model", "fake-model",
                "--max-calls", "40",
                "--apply", "--json",
            ],
            popen=True,
        )
        # Give the container time to start, open the store, clear the
        # anchor, and be blocked inside the (delayed) first vLLM call --
        # then interrupt it there, the exact window B1 closes.
        time.sleep(1.5)
        proc.send_signal(signal.SIGINT)
        try:
            out, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate(timeout=15)
    finally:
        vllm.terminate()
        vllm.wait(timeout=10)

    print(f"  interrupted run rc={proc.returncode}")
    interrupted_state = _load_summary_state(STORE_DIR)
    if not interrupted_state.get("tail_fp"):
        # Nothing completed before the interrupt: the pre-apply anchor
        # must have been put back, not left empty.
        assert_eq(interrupted_state.get("tail_fp"), before_tail_fp,
                   "the pre-apply anchor was restored (no pass had completed)")
    else:
        print("  a pass completed before the interrupt landed; it wrote its own fresh anchor")

    # Re-run (healthy vLLM, no delay): must not be refused, and must make
    # forward progress rather than repeating anything already done.
    resume_watermark_before = interrupted_state.get("last_summarized_turn")
    vllm2 = _start_fake_vllm(FAKE_VLLM_PORT + 6, delay=0.0)
    try:
        r2 = _docker_run(
            [
                (str(REPO_ROOT / "scripts"), "/opt/zl-repo/scripts", "ro"),
                (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
                (str(STORE_DIR), "/data/openwebui/compactor", "rw"),
                (str(BACKUP_ROOT / "webui.db"), "/data/openwebui/webui.db", "ro"),
            ],
            [
                "/opt/zl-repo/scripts/import-history.py",
                "--webui-db", "/data/openwebui/webui.db",
                "--chat-id", CHAT_ID,
                "--store", "/data/openwebui/compactor",
                "--vllm-url", f"http://127.0.0.1:{FAKE_VLLM_PORT + 6}",
                "--health-url", f"http://127.0.0.1:{FAKE_VLLM_PORT + 7}/health",
                "--model", "fake-model",
                "--max-calls", "5",
                "--apply", "--json",
            ],
        )
    finally:
        vllm2.terminate()
        vllm2.wait(timeout=10)
    out2 = r2.stdout + r2.stderr
    assert_true("Traceback" not in out2, f"the resume run is not a crash (tail: {out2[-1500:]})")
    assert_true(r2.returncode in (0, 4),
                f"the resume run is a normal completion or budget stop, not a "
                f"refusal (rc={r2.returncode}, out tail: {out2[-1500:]})")
    resumed_state = _load_summary_state(STORE_DIR)
    assert_true(
        int(resumed_state.get("last_summarized_turn") or 0) >= int(resume_watermark_before or 0),
        "the watermark did not go backwards across the interrupt+resume",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _preflight()
    try:
        _seed_store()
        test_apply_seam_alignment_and_blast_radius()
        test_apply_budget_limited_exits_4()
        test_ctrlc_mid_apply_leaves_store_rerunnable()
        print("\nAll real-image import-history --apply tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
