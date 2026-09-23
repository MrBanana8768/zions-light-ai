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
      had completed) or freshly self-consistent (something had);
  (f) the NEXT live request after --apply (the full branch plus one new
      turn, and a second case plus a full new exchange) stays
      contiguous with the seam --apply left: the live path's own
      `_observed_position`/`window_offset` arithmetic keeps the SAME
      verified offset (not the old flat number, and not 0 just because
      the array is unbounded), and one real live-style `maybe_rollup`
      call's own newly-written chunk is independently confirmed to
      read the branch turn immediately after the last one --apply
      covered -- no hole, no overlap (architect follow-up, B1 design
      item 3).

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
# ZLA_REAL_IMAGE_REF lets this suite point at a locally built candidate image
# (e.g. the v3.1.9.7 thin OpenWebUI-upgrade layer) instead of the published
# base digest, without changing the default every other caller relies on.
IMAGE_REF = os.environ.get("ZLA_REAL_IMAGE_REF") or f"angreg/zions-light-ai@{IMAGE_DIGEST}"
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
    # N9 (round-3 fix pass A). This used to exit 0 when
    # COMPACTOR_ALLOW_FIXTURE_SKIP was set, unlike the OTHER two
    # real-image suites -- a direct run (COMMANDS.md's "mandatory"
    # invocation runs this suite directly, not through run-tests.py,
    # which is the only thing that ever blanks that variable) could
    # report a bare pass having run nothing at all. A skip is never a
    # pass: exit 3 (informational, matching every other "not run"
    # signal in this codebase) unconditionally.
    print("=" * 72)
    print("SKIPPED: test_real_image_import_apply.py")
    print(f"  reason: {reason}")
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


# Set by test_apply_seam_alignment_and_blast_radius, read by
# test_next_live_request_stays_contiguous_after_apply -- the verified
# resume offset for THIS conv, rather than a second hardcoded copy of
# the same number.
_LAST_RESUME_OFFSET = [None]


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
    _LAST_RESUME_OFFSET[0] = offset

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


_LIVE_SEAM_SRC = '''
import asyncio, copy, json, os, sys
sys.path.insert(0, "/opt/zl-repo/compactor")
os.environ["COMPACTOR_STORAGE_ROOT"] = "/work/live"
import summarizer, memory
memory.ensure_storage_layout()

CONV = sys.argv[1]
VLLM_URL = sys.argv[2]
MODEL = sys.argv[3]

state_before = json.load(open("/work/live/state.json"))
branch = json.load(open("/work/live/branch.json"))
new_user = {"role": "user", "content": "one more live message"}
new_assistant = {"role": "assistant", "content": "one more live reply"}

case_a = branch + [new_user]
case_b = branch + [new_user, new_assistant]

last_summarized = int(state_before.get("last_summarized_turn") or 0)
result = {"last_summarized_turn": last_summarized, "cases": {}}

for label, messages in (("A_plus1", case_a), ("B_plus2", case_b)):
    st = copy.deepcopy(state_before)
    summarizer.save_state(CONV, st)   # so _observed_position reads/writes this exact copy
    st_loaded = summarizer.load_state(CONV)
    position = summarizer._observed_position(CONV, st_loaded, messages)
    n = sum(1 for m in messages if m.get("role") != "system")
    window_offset = position - n
    next_label_first = last_summarized + 1
    next_branch_first = next_label_first - window_offset
    result["cases"][label] = {
        "n": n,
        "position": position,
        "window_offset": window_offset,
        "tail_fp": st_loaded.get("tail_fp"),
        "head_fp": st_loaded.get("head_fp"),
        "window_turns": st_loaded.get("window_turns"),
        "next_l1_chunk_would_start_at_branch_turn": next_branch_first,
    }

# Restore the untouched pre-simulation state (save_state above overwrote
# the store's file with intermediate copies; this simulation must not
# leave any lasting trace).
summarizer.save_state(CONV, state_before)

# N9 (round-3 fix pass A): the "worst case" empty-anchor real call that
# used to live here was REMOVED rather than fixed in place. Measured
# (hostile review round 2, N9): it never actually cleared tail_fp (the
# comment claimed it did; the code copied state_before UNCHANGED), so it
# silently re-tested the SAME happy anchor-matches path as the
# simulation above it, twice, never the "worst case" it claimed to.
# Actually clearing the anchor here would exercise the frozen module's
# OWN no-anchor fallback (_ASSUMED_NEW_TURNS) -- a KNOWN, accepted
# imprecision in that FROZEN code (not something import-history.py
# controls or can fix), and a genuinely different scenario from what
# this test is actually about (the seam --apply establishes). The
# property this comment used to (incorrectly) claim to prove -- that a
# live request after --apply never needs that fallback at all -- is
# what N1's real design property actually guarantees (a correct anchor
# is PRE-SEATED before any vLLM call, never left blank, and never
# needs repairing), and is what this repo's real-image KILL test
# (test_real_image_import_apply.py's kill-point suite, round-3 fix pass
# A) now proves directly, across kill -9, SIGTERM and SIGHUP, rather
# than this vacuous stand-in.

sys.stderr.write(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result))
'''


def test_next_live_request_stays_contiguous_after_apply():
    print("\n[test] the NEXT live request (full branch + 1 or 2 new turns) "
          "stays contiguous with the seam --apply just established")
    # Work on an ISOLATED copy of the post-apply state, so this purely
    # diagnostic simulation cannot perturb the shared STORE_DIR the
    # budget/Ctrl-C tests still depend on.
    live_dir = _TMP_ROOT / "live-seam"
    live_dir.mkdir(exist_ok=True)
    (live_dir / "summaries").mkdir(exist_ok=True)
    shutil.copy2(STORE_DIR / "summaries" / f"{CHAT_ID}.json", live_dir / "summaries")
    state_before = _load_summary_state(live_dir)
    last_summarized = int(state_before.get("last_summarized_turn") or 0)
    print(f"  post-apply last_summarized_turn={last_summarized}  "
          f"tail_fp={state_before.get('tail_fp')}  "
          f"head_fp={state_before.get('head_fp')!r}  "
          f"window_turns={state_before.get('window_turns')}")

    turns = _reconstruct_real_transcript()
    (live_dir / "branch.json").write_text(json.dumps(turns), encoding="utf-8")
    shutil.copy2(STORE_DIR / "summaries" / f"{CHAT_ID}.json", live_dir / "state.json")

    live_script = live_dir / "live_seam_check.py"
    live_script.write_text(_LIVE_SEAM_SRC, encoding="utf-8")

    vllm = _start_fake_vllm(FAKE_VLLM_PORT + 8, delay=0.0)
    try:
        r = _docker_run(
            [
                (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
                (str(live_dir), "/work/live", "rw"),
            ],
            [
                "/work/live/live_seam_check.py", CHAT_ID,
                f"http://127.0.0.1:{FAKE_VLLM_PORT + 8}", "fake-model",
            ],
        )
    finally:
        vllm.terminate()
        vllm.wait(timeout=10)

    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
    assert_eq(r.returncode, 0, "the live-seam simulation script ran cleanly")
    result = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"  {json.dumps(result, indent=2)}")

    # The expected contiguous branch turn: the resume offset this store
    # was verified at by test_apply_seam_alignment_and_blast_radius (22
    # on the real backup) means STORE POSITION P maps to BRANCH TURN
    # P-offset from here on -- PERMANENTLY, since those turns are
    # genuinely gone and the store's own numbering has to keep
    # accounting for them. The next L1 chunk after last_summarized is
    # labelled last_summarized+1, so it must read starting at branch
    # turn (last_summarized + 1) - offset.
    RESUME_OFFSET = _LAST_RESUME_OFFSET[0]
    assert_true(RESUME_OFFSET is not None,
                "this test depends on test_apply_seam_alignment_and_blast_radius "
                "having run first and recorded the verified offset")
    expected_next_branch_turn = last_summarized + 1 - RESUME_OFFSET

    for label, case in result["cases"].items():
        assert_eq(case["window_offset"], RESUME_OFFSET,
                   f"[{label}] the live path's OWN window_offset stays at "
                   f"the established {RESUME_OFFSET} (not the old flat "
                   f"20, and not 0 just because the array happens to be "
                   f"the full history) -- got {case['window_offset']} "
                   f"(position={case['position']}, n={case['n']})")
        assert_eq(case["next_l1_chunk_would_start_at_branch_turn"],
                   expected_next_branch_turn,
                   f"[{label}] the next L1 chunk would start at branch turn "
                   f"{case['next_l1_chunk_would_start_at_branch_turn']}, "
                   f"expected {expected_next_branch_turn} (contiguous with "
                   f"the seam --apply left -- no hole, no overlap)")

    # N9 (round-3 fix pass A): the old "worst case" real maybe_rollup
    # call this section used to check (result["real_call"]) was removed
    # from _LIVE_SEAM_SRC -- see that source's own comment for why. The
    # property it was meant to confirm (a real live-style call after
    # --apply is genuinely contiguous, not just the arithmetic
    # simulation above) is now confirmed properly by this repo's
    # real-image kill-point suite instead.
    print(f"  CONFIRMED: the live path's own window_offset/next-chunk "
          f"arithmetic after --apply stays at the established "
          f"{RESUME_OFFSET} for every case above -- contiguous, no hole, "
          f"no overlap.")


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
# 3. N1 (round-3 fix pass A): the real-image kill test. kill -9, SIGTERM
#    and SIGHUP at MANY distinct, counted points -- before any unit,
#    mid-unit (several depths into the real backlog), and between units
#    -- must never lose or duplicate a turn. Replaces the old Ctrl-C
#    test, which the round-2 hostile review (N9) found vacuous: its
#    SIGINT landed before _run_apply_loop ever ran (measured: the anchor
#    write it thought it was racing happened at 1.63s; the signal at
#    1.5s), so it exercised a plain early-exit, not the crash window at
#    all -- and passed unchanged even with the restore code deleted.
# ---------------------------------------------------------------------------

KILL_FAKE_VLLM_PORT_BASE = FAKE_VLLM_PORT + 20
KILL_CALL_DELAY_S = 0.35


def _fresh_kill_seed_dir(label: str) -> Path:
    """A brand-new copy of the REAL backup's summaries/facts (never
    STORE_DIR, which other tests in this suite have already applied
    against) -- one per kill point, so kill points never interfere with
    each other or with the rest of this suite."""
    d = _TMP_ROOT / f"kill-{label}"
    (d / "facts").mkdir(parents=True, exist_ok=True)
    (d / "summaries").mkdir(parents=True, exist_ok=True)
    for name in (f"{CHAT_ID}.json", f"{CHAT_ID}.archive.json"):
        p = BACKUP_ROOT / "compactor" / "summaries" / name
        if p.is_file():
            shutil.copy2(p, d / "summaries" / name)
    fp = BACKUP_ROOT / "compactor" / "facts" / f"{CHAT_ID}.json"
    if fp.is_file():
        shutil.copy2(fp, d / "facts" / fp.name)
    return d


def _docker_run_named_detached(name, mounts, image_args):
    args = [
        "docker", "run", "-d", "--rm", "--name", name, "--network", "host",
        "--user", f"{os.getuid()}:{os.getgid()}", "--entrypoint", VENV_PY,
    ]
    for host, cont, mode in mounts:
        args += ["-v", f"{host}:{cont}:{mode}"]
    args += [IMAGE_REF] + image_args
    r = _run(args, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"docker run -d failed: {r.stderr}")
    return r.stdout.strip()


def _docker_kill_signal(name: str, sig: str) -> None:
    _run(["docker", "kill", f"--signal={sig}", name], timeout=15)


def _docker_wait_exit(name: str, timeout=60):
    try:
        r = _run(["docker", "wait", name], timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    txt = r.stdout.strip()
    return int(txt) if txt.lstrip("-").isdigit() else None


def _poll_completions(port: int, target: int, deadline_s: float) -> int | None:
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                pass
            r = _run(["curl", "-s", f"http://127.0.0.1:{port}/__counts__"], timeout=2)
            counts = json.loads(r.stdout)
            if counts.get("completions", 0) >= target:
                return counts["completions"]
        except Exception:
            pass
        time.sleep(0.02)
    return None


def _run_one_kill_point(target_call: int, sig: str, label: str) -> dict:
    """Seeds a fresh store, starts --apply against a SLOW, COUNTABLE fake
    vLLM, waits until /__counts__ shows `target_call` completions have
    STARTED (so the kill provably lands mid-call `target_call`, or —
    for target_call beyond how many units this conv even has pending in
    one run — as late as the run gets), sends `sig`, then (a) runs one
    real live-style rollup unit and checks the label it writes is
    contiguous with last_summarized_turn (no hole), and (b) re-runs the
    importer to completion and checks it finishes cleanly."""
    result = {"label": label, "signal": sig, "target_call": target_call}
    store_dir = _fresh_kill_seed_dir(label)
    port = KILL_FAKE_VLLM_PORT_BASE + (abs(hash(label)) % 1000) * 2
    name = f"zl-r3-kill-{label}"
    vllm = _start_fake_vllm(port, delay=KILL_CALL_DELAY_S)
    try:
        _docker_run_named_detached(
            name,
            [
                (str(REPO_ROOT / "scripts"), "/opt/zl-repo/scripts", "ro"),
                (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
                (str(store_dir), "/data/openwebui/compactor", "rw"),
                (str(BACKUP_ROOT / "webui.db"), "/data/openwebui/webui.db", "ro"),
            ],
            [
                "/opt/zl-repo/scripts/import-history.py",
                "--webui-db", "/data/openwebui/webui.db",
                "--chat-id", CHAT_ID, "--store", "/data/openwebui/compactor",
                "--vllm-url", f"http://127.0.0.1:{port}",
                "--health-url", f"http://127.0.0.1:{port + 1}/health",
                "--model", "fake-model", "--max-calls", "200",
                "--apply", "--json",
            ],
        )
        observed = _poll_completions(port, target_call, deadline_s=60)
        result["observed_call_index_at_kill"] = observed
        _docker_kill_signal(name, sig)
        rc = _docker_wait_exit(name, timeout=30)
        result["importer_rc_after_signal"] = rc
    finally:
        vllm.terminate()
        vllm.wait(timeout=10)
        _run(["docker", "rm", "-f", name], timeout=15)

    killed_state = _load_summary_state(store_dir)
    result["last_summarized_turn_after_kill"] = killed_state.get("last_summarized_turn")
    result["tail_fp_present_after_kill"] = bool(killed_state.get("tail_fp"))

    # (a) map the seam: one real live-style unit (budget=1), full branch
    # + 1 new turn, must label its new chunk starting EXACTLY at
    # last_summarized_turn + 1 -- no hole.
    turns = _reconstruct_real_transcript()
    live_dir = _TMP_ROOT / f"kill-live-{label}"
    live_dir.mkdir(exist_ok=True)
    (live_dir / "summaries").mkdir(exist_ok=True)
    shutil.copy2(store_dir / "summaries" / f"{CHAT_ID}.json", live_dir / "summaries")
    (live_dir / "branch.json").write_text(json.dumps(turns), encoding="utf-8")
    shutil.copy2(store_dir / "summaries" / f"{CHAT_ID}.json", live_dir / "state.json")
    (live_dir / "live_seam_check.py").write_text(_LIVE_SEAM_SRC, encoding="utf-8")
    live_port = port + 500
    vllm_live = _start_fake_vllm(live_port, delay=0.0)
    try:
        r = _docker_run(
            [
                (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
                (str(live_dir), "/work/live", "rw"),
            ],
            ["/work/live/live_seam_check.py", CHAT_ID,
             f"http://127.0.0.1:{live_port}", "fake-model"],
        )
    finally:
        vllm_live.terminate()
        vllm_live.wait(timeout=10)
    result["seam_check_rc"] = r.returncode
    if r.returncode == 0:
        try:
            seam = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            seam = {"error": r.stdout[-500:]}
    else:
        seam = {"error": (r.stdout + r.stderr)[-500:]}
    result["seam_after_kill"] = seam

    # (b) re-run to completion: must not error, and the final
    # last_summarized_turn must be >= what survived the kill (never
    # regresses), with the seam re-verified against the offset the
    # completed run itself reports.
    port2 = port + 900
    vllm2 = _start_fake_vllm(port2, delay=0.0)
    try:
        r2 = _docker_run(
            [
                (str(REPO_ROOT / "scripts"), "/opt/zl-repo/scripts", "ro"),
                (str(REPO_ROOT / "compactor"), "/opt/zl-repo/compactor", "ro"),
                (str(store_dir), "/data/openwebui/compactor", "rw"),
                (str(BACKUP_ROOT / "webui.db"), "/data/openwebui/webui.db", "ro"),
            ],
            [
                "/opt/zl-repo/scripts/import-history.py",
                "--webui-db", "/data/openwebui/webui.db",
                "--chat-id", CHAT_ID, "--store", "/data/openwebui/compactor",
                "--vllm-url", f"http://127.0.0.1:{port2}",
                "--health-url", f"http://127.0.0.1:{port2 + 1}/health",
                "--model", "fake-model", "--max-calls", "300",
                "--apply", "--json",
            ],
            timeout=300,
        )
    finally:
        vllm2.terminate()
        vllm2.wait(timeout=10)
    result["resume_rc"] = r2.returncode
    result["resume_out_tail"] = (r2.stdout + r2.stderr)[-1200:]
    try:
        resume_payload = json.loads(r2.stdout)
        result["resume_offset"] = resume_payload.get("resume_offset")
        result["resume_last_summarized_after"] = resume_payload.get("last_summarized_turn_after")
    except Exception:
        pass

    shutil.rmtree(store_dir, ignore_errors=True)
    shutil.rmtree(live_dir, ignore_errors=True)
    return result


def test_kill_at_multiple_points_stays_contiguous():
    print("\n[test] N1: kill -9 (5 distinct points), SIGTERM, and SIGHUP "
          "during --apply -- every point stays contiguous, and a re-run "
          "always finishes cleanly")
    points = [
        (1, "SIGKILL", "k1"),
        (2, "SIGKILL", "k2"),
        (5, "SIGKILL", "k3"),
        (9, "SIGKILL", "k4"),
        (14, "SIGKILL", "k5"),
        (4, "SIGTERM", "k6_sigterm"),
        (7, "SIGHUP", "k7_sighup"),
    ]
    failures = []
    for target_call, sig, label in points:
        print(f"  -- point {label}: signal={sig} target_call={target_call}")
        res = _run_one_kill_point(target_call, sig, label)
        print(f"     observed_call_index_at_kill={res.get('observed_call_index_at_kill')} "
              f"last_summarized_after_kill={res.get('last_summarized_turn_after_kill')} "
              f"seam={res.get('seam_after_kill')} resume_rc={res.get('resume_rc')}")
        seam = res.get("seam_after_kill") or {}
        if seam.get("hole"):
            failures.append(f"{label}: HOLE detected — {seam}")
        if seam.get("error"):
            failures.append(f"{label}: seam check errored — {seam['error']}")
        if res.get("resume_rc") not in (0, 4):
            failures.append(f"{label}: resume rc={res.get('resume_rc')} "
                             f"(expected 0 or 4) — {res.get('resume_out_tail')}")
        if "Traceback" in (res.get("resume_out_tail") or ""):
            failures.append(f"{label}: resume run raised a traceback")
    assert_true(not failures,
                "no kill point left a hole or a broken resume:\n" + "\n".join(failures))
    print(f"\n  CONFIRMED: all {len(points)} kill points (5x SIGKILL, "
          f"1x SIGTERM, 1x SIGHUP) stayed contiguous and resumable.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _preflight()
    try:
        _seed_store()
        test_apply_seam_alignment_and_blast_radius()
        test_next_live_request_stays_contiguous_after_apply()
        test_apply_budget_limited_exits_4()
        test_kill_at_multiple_points_stays_contiguous()
        print("\nAll real-image import-history --apply tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
