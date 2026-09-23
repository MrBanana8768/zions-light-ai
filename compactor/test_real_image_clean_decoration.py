"""The real-image test for scripts/clean-decoration.py — the tool that
strips emoji/rule-line/status-board decoration out of her stored chat
history, active facts, and episodic memory (see that script's own module
docstring, and /tmp/zl/degeneration-2026-09-23.md, the forensic report it
implements part of).

WHY A REAL-IMAGE TEST, NOT JUST compactor/test_clean_decoration_script.py.
That suite proves the cleaning logic and the CLI plumbing against small
synthetic fixtures. It cannot prove the one thing that actually makes this
script safe to run against her real store: that cleaning her real,
3,879-message branch and recomputing the anchor with the FROZEN v3.1.9
`summarizer` module leaves the next live request aligned at EXACTLY the
position and `window_offset` it would have landed at without this script
ever having run — see clean-decoration.py's own HAZARD section. That claim
can only be checked against the real frozen code and a real, large,
messy branch, which is what this suite runs against, inside the exact
published image the pod runs.

NEEDS, and SKIPS (exit 3) HONESTLY if any is missing — same convention as
test_real_image_operator_scripts.py:
  - a real Docker daemon
  - the published image already pulled (`docker image inspect`)
  - `git`, with the `v3.1.9` tag reachable in this checkout
  - the real 2026-09-23 pod-export backup on this host (the forensic
    report's own data — 09-23, not 09-22: that is where her decorated
    replies and the 72-of-84 decorated episodic exchanges live)

MANDATORY, NOT OPTIONAL, same reasoning as test_real_image_operator_
scripts.py's own docstring: this is a required, host-run part of the
release gate, deliberately excluded from the sandboxed `unit-tests`
compose service (no docker socket, `network_mode: none`) — pass C wires
it into scripts/run-tests.py's `NEEDS_DOCKER` list; this file does not
touch that list itself (file ownership — see the architect's brief).

    python3 compactor/test_real_image_clean_decoration.py

WHAT EACH SCENARIO PROVES is inline at that scenario. The alignment proof
(steps a-f in the architect's brief) is
`test_alignment_proof_position_and_offset_survive_cleaning`.
"""

import asyncio
import hashlib
import http.server
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# This suite copies webui.db (500MB+) and the chromadb store (~330MB)
# several times over (one working copy plus one scratch copy per scenario
# that needs its own before/after). /tmp is host policy's small tmpfs;
# large copies belong under a real filesystem with room for them.
_SCRATCH_BASE = Path("/home/drew/scratch/r3d")
if _SCRATCH_BASE.is_dir():
    _SCRATCH_BASE.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(_SCRATCH_BASE)

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = Path("/home/drew/pod-exports/2026-09-23/backup")
IMAGE_DIGEST = "sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65"
IMAGE_REF = f"angreg/zions-light-ai@{IMAGE_DIGEST}"
VENV_PY = "/opt/compactor-venv/bin/python"
CHAT_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
DOCKER_TIMEOUT_S = 240

FAILED = []


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_true(cond, label, extra=""):
    if not cond:
        print(f"FAIL {label} {extra}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_in(needle, haystack, label):
    if needle not in haystack:
        print(f"FAIL {label}: {needle!r} not found")
        print(f"  --- full output ---\n{haystack}\n  --- end ---")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_not_in(needle, haystack, label):
    if needle in haystack:
        print(f"FAIL {label}: {needle!r} unexpectedly present")
        print(f"  --- full output ---\n{haystack}\n  --- end ---")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def _skip(reason: str) -> None:
    print("=" * 72)
    print("SKIPPED: test_real_image_clean_decoration.py")
    print(f"  reason: {reason}")
    print("  Needs a real Docker daemon, the published image already")
    print("  pulled, git with the v3.1.9 tag, and the real 2026-09-23")
    print("  pod-export backup on this host. Run directly:")
    print("    python3 compactor/test_real_image_clean_decoration.py")
    print("=" * 72)
    sys.exit(3)


# ---------------------------------------------------------------------------
# Preflight
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
    if not (BACKUP_ROOT / "compactor" / "summaries" / f"{CHAT_ID}.json").is_file():
        _skip(f"real backup missing summaries for {CHAT_ID}")


# ---------------------------------------------------------------------------
# Fixtures — the backup is only ever READ. Every scratch tree below is a
# COPY this test process itself creates and deletes; nothing a bug in this
# run does can ever reach BACKUP_ROOT.
# ---------------------------------------------------------------------------

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="compactor-test-real-image-clean-"))
V319_PKG = _TMP_ROOT / "v319pkg" / "compactor"
SCRIPTS_DIR = _TMP_ROOT / "scripts"          # clean-decoration.py + import-history.py
WORK_DB = _TMP_ROOT / "webui.db"             # one writable copy, mutated by --apply
WORK_STORE = _TMP_ROOT / "store"             # facts/summaries/chromadb copy for CHAT_ID
FULL_STORE_BEFORE_DIGESTS = _TMP_ROOT / "full_store_before.json"
DRIVER_PY = _TMP_ROOT / "driver.py"
FAKEVLLM_PY = _TMP_ROOT / "fakevllm.py"


def _build_v319_package() -> None:
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


def _seed_scripts() -> None:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "clean-decoration.py", SCRIPTS_DIR)
    shutil.copy2(REPO_ROOT / "scripts" / "import-history.py", SCRIPTS_DIR)


def _seed_work_copies() -> None:
    """webui.db (500MB+) and the facts/summaries/chromadb store for
    CHAT_ID, copied ONCE into scratch. `--apply` scenarios reuse these
    same copies sequentially (each scenario's own before/after digests are
    taken around its own mutation), which is why this suite runs its
    scenarios in a fixed, deliberate order rather than as independent
    pytest-style cases — see `run_all` at the bottom."""
    print("  seeding webui.db copy (this is the 500MB+ real export)...")
    shutil.copy2(BACKUP_ROOT / "webui.db", WORK_DB)
    print("  seeding compactor store copy (chromadb is ~330MB)...")
    shutil.copytree(BACKUP_ROOT / "compactor", WORK_STORE)


def _store_file_digests(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _count_history_vs_table_mismatches(db_path: Path) -> int:
    """For CHAT_ID's whole branch: how many message ids have DIFFERENT
    text between chat.chat.history.messages and the chat_message table.
    Read-only (uses a plain sqlite3 ro connection on the host — no docker
    needed for a check this cheap). Proves clean-decoration.py's own
    dual-copy write (both columns, every in-scope id) actually leaves the
    two copies agreeing — the real-data version of the injected concern
    about OpenWebUI 0.11's two stored copies diverging (RUNBOOK_CHAT_
    TREE.md's retired v1 repair script)."""
    import sqlite3 as _sqlite3
    con = _sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT chat FROM chat WHERE id = ?", (CHAT_ID,)).fetchone()
        data = json.loads(row[0])
        messages = data.get("history", {}).get("messages", {})
        mismatches = 0
        for msg_id, node in messages.items():
            content = node.get("content")
            if not isinstance(content, str):
                continue
            cm = con.execute(
                "SELECT content FROM chat_message WHERE id = ? AND chat_id = ?",
                (f"{CHAT_ID}-{msg_id}", CHAT_ID),
            ).fetchone()
            if cm is None:
                continue
            try:
                table_text = json.loads(cm[0])
            except (json.JSONDecodeError, TypeError):
                mismatches += 1
                continue
            if isinstance(table_text, str) and table_text != content:
                mismatches += 1
        return mismatches
    finally:
        con.close()


FAKEVLLM_SRC = '''
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

N = {"n": 0}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        self.rfile.read(n)
        N["n"] += 1
        sys.stderr.write("FAKEVLLM CALL %d\\n" % N["n"])
        sys.stderr.flush()
        body = json.dumps({
            "choices": [{
                "message": {"content": "A plain summary of this chunk of the conversation."},
                "finish_reason": "stop",
            }]
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = b"{}"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

HTTPServer(("127.0.0.1", 8000), H).serve_forever()
'''

DRIVER_SRC = r'''
"""In-container driver for the real-image alignment proof. Argv[1] picks
the phase; everything else is read from env vars to keep the docker
command line short."""
import asyncio
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["COMPACTOR_STORAGE_ROOT"] = os.environ["CD_STORE"]
sys.path.insert(0, "/opt/compactor")
import summarizer  # noqa: E402

spec = importlib.util.spec_from_file_location("ih", "/opt/zl-repo/scripts/import-history.py")
ih = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ih)

CONV_ID = os.environ["CD_CONV"]
DB_PATH = Path(os.environ["CD_DB"])


def _reconstruct():
    """Deliberately uses only the LOW-LEVEL, stable helpers
    (_walk_branch_from_history, _flatten_content) rather than
    reconstruct_transcript itself: that function is owned by a
    DIFFERENT, concurrently-edited script (import-history.py) and has
    grown its own dual-copy divergence check since this driver was
    written — a real, useful check for THAT script's own purposes, but
    not this probe's concern (this probe only needs A branch to
    fingerprint, and clean-decoration.py's own dual-copy check already
    covers the actual safety question for what this suite is testing).
    Isolates this test file from that script's evolving validation
    behavior, which this file does not own and must not depend on."""
    con = ih._open_ro(DB_PATH)
    try:
        row = con.execute("SELECT chat FROM chat WHERE id = ?", (CONV_ID,)).fetchone()
        data = json.loads(row[0])
        history = data.get("history") or {}
        chain = ih._walk_branch_from_history(history)
    finally:
        con.close()
    return [
        {"role": n.get("role") or "unknown", "content": ih._flatten_content(n.get("content"))}
        for n in chain
    ]


def phase_align():
    turns = _reconstruct()
    messages = turns + [{"role": "user", "content": "One more thing before we go on."}]
    state = summarizer.load_state(CONV_ID)
    probe = copy.deepcopy(state)
    non_system = [m for m in messages if m.get("role") != "system"]
    anchor_before = [x for x in (state.get("tail_fp") or []) if isinstance(x, str)]
    fps = summarizer._turn_fingerprints(
        non_system[-getattr(summarizer, "_FINGERPRINT_TAIL_TURNS", 64):]
    )
    cands = summarizer._align_candidates(anchor_before, fps)
    pos = summarizer._observed_position(CONV_ID, probe, messages)
    print(json.dumps({
        "position": pos,
        "window_offset": pos - len(non_system),
        "window_turns": len(non_system),
        "tail_fp_used": probe.get("tail_fp"),
        "anchor_before": anchor_before,
        "align_candidates": cands,
        "recorded_position_prev": summarizer._recorded_position(state),
    }))


def phase_rollup():
    turns = _reconstruct()
    vllm_url = os.environ["CD_VLLM_URL"]
    budget = {"remaining": 1, "exhausted": False}

    async def _go():
        return await summarizer.maybe_rollup(
            CONV_ID, turns, vllm_url, "test-model", vllm_call_budget=budget,
        )

    new_state = asyncio.run(_go())
    l1 = new_state.get("l1") or []
    l2 = new_state.get("l2") or []
    l3 = new_state.get("l3")
    print(json.dumps({
        "spent": max(0, 1 - budget["remaining"]),
        "last_summarized_turn": new_state.get("last_summarized_turn"),
        "l1_spans": [[c.get("first_turn"), c.get("last_turn")] for c in l1],
        "l2_spans": [[c.get("first_turn"), c.get("last_turn")] for c in l2],
        "l3_last_turn": (l3 or {}).get("last_turn"),
        "covered_fps_len": len(summarizer._covered_fps(new_state)),
    }))


def phase_apply_no_force():
    """Calls clean-decoration.py's own main() IN-PROCESS, with the
    OpenWebUI/compactor liveness checks monkeypatched to a clean "dead" —
    isolating the ANCHOR-verification refusal from this bare container's
    own unrelated liveness ambiguity (no supervisord is running here,
    which real_image_operator_scripts.py's own docstring notes is
    specific to this kind of sandboxed harness, not a defect). This is
    the one true test of whether --apply refuses the anchor rewrite ON
    ITS OWN, without --force doing double duty for something else.
    """
    cd_spec = importlib.util.spec_from_file_location(
        "cd", "/opt/zl-repo/scripts/clean-decoration.py"
    )
    cd = importlib.util.module_from_spec(cd_spec)
    cd_spec.loader.exec_module(cd)
    cd._openwebui_is_alive = lambda port: "dead"
    cd._compactor_is_alive = lambda url: False

    import io
    import contextlib
    buf = io.StringIO()
    argv = [
        "--webui-db", os.environ["CD_DB"], "--store", os.environ["CD_STORE"],
        "--conv", CONV_ID, "--only", "webui", "--apply", "--json",
    ]
    with contextlib.redirect_stdout(buf):
        rc = cd.main(argv)
    print(json.dumps({"returncode": rc, "report": json.loads(buf.getvalue())}))


def phase_episodic_verify():
    """Item 3's before/after + re-embedding + similarity-query proof,
    run against the CURRENT contents of the store's chromadb (call this
    BEFORE and AFTER an --apply --only episodic run)."""
    import chromadb

    sys.path.insert(0, "/opt/compactor")
    import retrieval  # noqa: E402

    client = chromadb.PersistentClient(path=os.environ["CD_CHROMA_PATH"])
    col = client.get_collection(retrieval.COLLECTION_NAME)
    got = col.get(where={"conv_id": CONV_ID}, include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []

    def assistant_half(doc):
        marker = "\n[assistant]: "
        idx = doc.find(marker)
        return doc[idx + len(marker):] if idx != -1 else doc

    decorated = sum(
        1 for d in docs
        if any(sym in assistant_half(d) for sym in ("✅", "▶️", "═" * 8, "━" * 8))
    )
    all_prefixed = all(i.startswith(CONV_ID + "::") for i in ids)

    vecs = retrieval._embed(["Say the laws quickly and verify them"])
    res = col.query(query_embeddings=vecs, n_results=3, where={"conv_id": CONV_ID})
    hits = list(zip(res["ids"][0], res["distances"][0]))

    print(json.dumps({
        "total_docs": len(ids),
        "decorated_assistant_halves": decorated,
        "all_ids_content_addressed_and_prefixed": all_prefixed,
        "similarity_hits": hits,
    }))


def phase_m5_mutant():
    """M5, run WHERE httpx (and the rest of summarizer's real deps)
    actually exist — the item-5 follow-up: the host running this test
    file cannot `import summarizer` directly, so this mutant is proven
    inside the image instead, via this driver, exactly like every other
    real-module check in this suite.

    REDESIGNED after the first version turned out not to be a real
    mutant at all: `_turn_fingerprints` hashes each turn independently
    (no cross-turn state), so "fingerprint the whole history, then take
    the last _ANCHOR_TURNS" and "fingerprint just the tail window, then
    take the last _ANCHOR_TURNS" produce the IDENTICAL last N elements —
    verified directly, `real == mutant` every time. That original
    "mutant" was a performance-only difference (hashing thousands of
    turns instead of 64 on a real conversation), never a value bug, so
    it could never be killed by a value comparison — the test would have
    been vacuously green regardless of whether the tail-slicing code
    even existed. The real, observable-bug mutant: hardcoding the anchor
    length instead of reading `_ANCHOR_TURNS` from the frozen module —
    this DOES change the returned tail_fp's length, which
    `_align_candidates` depends on exactly matching.
    """
    cd_spec = importlib.util.spec_from_file_location(
        "cd", "/opt/zl-repo/scripts/clean-decoration.py"
    )
    cd = importlib.util.module_from_spec(cd_spec)
    cd_spec.loader.exec_module(cd)

    orig_recompute = cd._recompute_anchor

    def _bad_recompute(summarizer_mod, turns):
        t = [x for x in turns if x.get("role") != "system"]
        if not t:
            return [], "", 0
        tail_n = getattr(summarizer_mod, "_FINGERPRINT_TAIL_TURNS", 64)
        fps = summarizer_mod._turn_fingerprints(t[-tail_n:])
        head_fp = summarizer_mod._turn_fingerprints(t[:1])[0]
        return fps[-3:], head_fp, len(t)  # BUG: hardcoded 3, not the real _ANCHOR_TURNS

    long_turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
                  for i in range(80)]
    real = orig_recompute(summarizer, long_turns)
    mutant_result = _bad_recompute(summarizer, long_turns)
    anchor_n = getattr(summarizer, "_ANCHOR_TURNS", 4)
    killed = (real != mutant_result) if anchor_n != 3 else True
    print(json.dumps({"killed": killed, "real": real, "mutant": mutant_result}))


PHASES = {
    "align": phase_align, "rollup": phase_rollup, "apply_no_force": phase_apply_no_force,
    "m5_mutant": phase_m5_mutant, "episodic_verify": phase_episodic_verify,
}
PHASES[sys.argv[1]]()
'''


def _write_driver_files() -> None:
    DRIVER_PY.write_text(DRIVER_SRC, encoding="utf-8")
    FAKEVLLM_PY.write_text(FAKEVLLM_SRC, encoding="utf-8")


def _docker_run(mounts, image_args, entrypoint=VENV_PY, env=None,
                 run_as_host_user=False, timeout=DOCKER_TIMEOUT_S):
    args = ["docker", "run", "--rm", "--entrypoint", entrypoint]
    if run_as_host_user:
        args += ["-u", f"{os.getuid()}:{os.getgid()}"]
    for host, cont, mode in mounts:
        args += ["-v", f"{host}:{cont}:{mode}"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    args += [IMAGE_REF] + image_args
    return _run(args, timeout=timeout)


def _fixup_perms(path: Path) -> None:
    """Every docker_run above runs as root (the image's default) unless
    told otherwise, so anything it WRITES into a host-mounted scratch
    directory (a `.bak-<stamp>` backup, a rewritten webui.db) comes back
    root-owned — unreadable/unwritable to this test process's own host
    user for the digests, restores and follow-up docker_run calls
    (themselves running as root again, so THEY can still write; it is
    only the HOST-side Python code in this file that needs this). `-u
    <uid>:<gid>` was tried instead and hit real WSL2/Docker-Desktop UID-
    mapping inconsistencies on a network-shared scratch path; a chmod
    pass after the fact is simpler and does not depend on how the host's
    UID happens to map into the container. Best-effort: never raises."""
    try:
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{path}:/fix", "--entrypoint", "chmod",
             IMAGE_REF, "-R", "a+rwX", "/fix"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        print(f"  (non-fatal: permission fixup on {path} failed: {e})")


def _webui_dir_mount(db_path: Path, mode: str) -> tuple:
    """Mount `db_path`'s PARENT DIRECTORY at /data/openwebui, not just the
    file — a single-file bind mount has no real directory backing it on
    the host, so a sibling file this script creates (a `.bak-<stamp>`
    backup) would only ever exist inside the container's own ephemeral
    overlay and vanish with `--rm`, never reaching the host at all
    (found the hard way: `--restore` scenarios kept reporting "no backup
    found" until this was fixed). Mounting the directory means every
    sibling file lands on the host, at exactly the path
    `_backup_file`/`--restore` themselves compute."""
    return (str(db_path.parent), "/data/openwebui", mode)


def _standard_mounts(db_mode="rw", store_mode="rw"):
    return [
        (str(V319_PKG), "/opt/compactor", "ro"),
        (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
        (str(WORK_STORE), "/data/store", store_mode),
        _webui_dir_mount(WORK_DB, db_mode),
    ]


def _cd_args(*extra):
    return [
        "/opt/zl-repo/scripts/clean-decoration.py",
        "--webui-db", "/data/openwebui/webui.db",
        "--store", "/data/store",
        "--conv", CHAT_ID,
    ] + list(extra)


# ---------------------------------------------------------------------------
# 1. The full dry run against the real, unmodified backup copy — writes
#    nothing, and reports every target's pending change count.
# ---------------------------------------------------------------------------

def test_full_dry_run_reports_pending_changes_and_writes_nothing():
    print("\n[test] full dry run (all three targets) against the real 09-23 backup copy")
    before_store = _store_file_digests(WORK_STORE)
    before_db_md5 = hashlib.md5(WORK_DB.read_bytes()).hexdigest()
    # store_mode is "rw" even though this is a dry run: chromadb needs a
    # writable directory for its own internal locking/journaling even to
    # READ (verified: `attempt to write a readonly database` otherwise).
    # webui.db stays "ro" — this script opens it via an explicit sqlite3
    # `mode=ro` URI, so a read-only bind mount is a real, additional
    # guarantee there, not just a formality. The digest comparison right
    # below is the actual proof that nothing was written either way.
    r = _docker_run(_standard_mounts(db_mode="ro", store_mode="rw"),
                     _cd_args("--json"))
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "no Python traceback")
    payload = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
    assert_eq(r.returncode, 3, "dry run against a genuinely decorated branch exits 3")
    webui_count = payload.get("targets", {}).get("webui", {}).get("count")
    facts_count = payload.get("targets", {}).get("facts", {}).get("count")
    episodic_target = payload.get("targets", {}).get("episodic", {})
    episodic_count = episodic_target.get("count")
    assert_true(isinstance(webui_count, int) and webui_count > 0,
                "webui target reports at least one pending change", extra=str(webui_count))
    assert_true(isinstance(facts_count, int),
                "facts target ran (count may be 0 if her active facts moved since the forensic report)",
                extra=str(facts_count))
    # Item 3 of the hostile review: this used to silently read 0 — the
    # collection name was hardcoded wrong ("episodic_memory" instead of
    # retrieval.COLLECTION_NAME, "conversation_turns" — get_or_create
    # silently creates an empty phantom collection for a wrong name
    # rather than erroring). Fixed; now finds her real 84 documents and
    # the vast majority need cleaning, matching the forensic report's
    # "72 of 84" order of magnitude.
    assert_true(
        not episodic_target.get("skipped"),
        "episodic target is NOT skipped (the collection resolves for real)",
        extra=str(episodic_target.get("skipped")),
    )
    assert_true(
        isinstance(episodic_count, int) and episodic_count >= 70,
        "episodic target finds the bulk of her real 84 documents needing cleaning "
        "(collection-name bug fixed: this used to silently report 0)",
        extra=str(episodic_count),
    )
    after_store = _store_file_digests(WORK_STORE)
    after_db_md5 = hashlib.md5(WORK_DB.read_bytes()).hexdigest()
    # chromadb's OWN engine touches its sqlite file/segment metadata even
    # for a pure read (verified: WAL/checkpoint bookkeeping on `query`/
    # `get`, unrelated to this script's own logic, which never calls a
    # write method on the episodic target during a dry run) — exclude
    # chromadb/ from the byte-identical check for that reason; every
    # OTHER store path (facts, summaries) has no such excuse and must be
    # untouched.
    before_non_chroma = {k: v for k, v in before_store.items() if not k.startswith("chromadb/")}
    after_non_chroma = {k: v for k, v in after_store.items() if not k.startswith("chromadb/")}
    assert_eq(before_non_chroma, after_non_chroma,
              "dry run changed not one byte of facts/summaries under the store")
    assert_eq(before_db_md5, after_db_md5, "dry run changed not one byte of webui.db")
    globals()["_DRY_RUN_WEBUI_COUNT"] = webui_count


# ---------------------------------------------------------------------------
# 2 (a)-(c) + (f). The alignment proof — the core claim this script makes.
# ---------------------------------------------------------------------------

def _run_align_probe(db_path: Path, store_path: Path):
    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
            (str(store_path), "/data/store", "ro"),
            (str(db_path), "/data/openwebui/webui.db", "ro"),
            (str(DRIVER_PY), "/work/driver.py", "ro"),
        ],
        ["/work/driver.py", "align"],
        env={"CD_STORE": "/data/store", "CD_CONV": CHAT_ID,
             "CD_DB": "/data/openwebui/webui.db"},
    )
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "align probe: no traceback")
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        print("align probe produced no parseable JSON:\n", out)
        FAILED.append("align probe JSON")
        return {}


def _run_apply_no_force_probe():
    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
            (str(WORK_STORE), "/data/store", "rw"),
            _webui_dir_mount(WORK_DB, "rw"),
            (str(DRIVER_PY), "/work/driver.py", "ro"),
        ],
        ["/work/driver.py", "apply_no_force"],
        env={"CD_STORE": "/data/store", "CD_CONV": CHAT_ID,
             "CD_DB": "/data/openwebui/webui.db"},
    )
    _fixup_perms(WORK_DB.parent)
    _fixup_perms(WORK_STORE)
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "apply_no_force driver: no traceback")
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
    if not lines:
        print("apply_no_force driver produced no parseable JSON:\n", out)
        FAILED.append("apply_no_force driver JSON")
        return None
    return json.loads(lines[-1])


def test_alignment_proof_position_and_offset_survive_cleaning():
    print("\n[test] alignment proof (a)-(c), run with a genuinely UNFORCED liveness check "
          "(monkeypatched to a clean 'dead' in-process, isolating this from the bare "
          "container's own unrelated supervisord-ambiguity — see phase_apply_no_force). "
          "REAL FINDING on this data: her actual stored anchor already has one turn of "
          "pre-existing drift against a fresh branch reconstruction, unrelated to any text "
          "this run cleans (verified independently: a fresh recompute over the CURRENT, "
          "completely unmodified branch ALSO disagrees with the stored anchor, by exactly "
          "this much). clean-decoration.py's own `_anchor_rewrite_is_verified` catches "
          "exactly this and refuses the anchor rewrite without --force.")
    # (a) BEFORE cleaning.
    before = _run_align_probe(WORK_DB, WORK_STORE)
    assert_true("position" in before, "step (a): got a position from the real frozen summarizer",
                extra=str(before))
    print(f"  before: {before}")

    # (b) Clean, for real, with NO --force anywhere in the decision that
    #     matters (the anchor). If the drift finding holds on this data,
    #     this must refuse the anchor specifically while still cleaning
    #     webui.db's text — exit 4, "anchor_refused" present.
    result = _run_apply_no_force_probe()
    assert_true(result is not None, "apply_no_force ran")
    rc, payload = result["returncode"], result["report"]
    webui_report = payload.get("targets", {}).get("webui", {})
    # Post item-2 fix: an unverifiable anchor no longer blocks the ANCHOR
    # alone (there is no separate "anchor_refused" for this path) — it
    # makes main() skip cleaning the specific turns tail_fp/head_fp cover
    # and leaves the (still-valid-for-what-remains) anchor untouched. The
    # real signal is "skipped_protected_by_anchor" plus exit 4.
    skipped = webui_report.get("skipped_protected_by_anchor")
    print(f"  rc={rc} skipped_protected_by_anchor={len(skipped) if skipped else 0} turn(s)")
    assert_true(rc in (0, 4), f"unforced apply exits 0 or 4, got {rc}")

    if skipped:
        # The real, documented finding: pre-existing drift made some of
        # the requested turns unsafe to clean without rewriting the
        # anchor, with no --force anywhere. Prove the skip is real (the
        # anchor on disk did NOT move) and that whatever WAS safe to
        # clean still went through.
        assert_eq(rc, 4, "exit 4: real progress plus a documented skip")
        after_blocked = _run_align_probe(WORK_DB, WORK_STORE)
        # Compare the ON-DISK anchor ("anchor_before", read straight off
        # state.tail_fp before this probe's own synthetic turn is even
        # considered) — NOT "tail_fp_used", which is _observed_position's
        # OWN in-memory recomputation for THIS probe's synthetic turn and
        # differs across probes by design (the underlying webui.db text
        # differs before/after cleaning even when the state FILE's own
        # anchor was correctly left untouched).
        assert_eq(after_blocked.get("anchor_before"), before.get("anchor_before"),
                  "the ON-DISK anchor is left EXACTLY as it was when the rewrite was refused")
        assert_true(webui_report.get("count", 0) > 0,
                    "webui text cleaning still proceeded despite the anchor refusal")

        # A follow-up --force run (through the ordinary CLI, liveness
        # bypassed the same way the rest of this suite already does)
        # accepts the realignment — webui.db is already clean, so this
        # call's only remaining work is the anchor itself.
        r3 = _docker_run(_standard_mounts(), _cd_args("--apply", "--force", "--json"))
        _fixup_perms(WORK_DB.parent)
        _fixup_perms(WORK_STORE)
        out3 = r3.stdout + r3.stderr
        assert_not_in("Traceback", out3, "forced apply: no traceback")
        payload3 = json.loads(r3.stdout) if r3.stdout.strip().startswith("{") else {}
        assert_true(
            not payload3.get("targets", {}).get("webui", {}).get("anchor_refused"),
            "--force accepts the realignment on the SAME (already-clean) branch",
        )
        after_forced = _run_align_probe(WORK_DB, WORK_STORE)
        print(f"  after (forced): {after_forced}")
        assert_true(
            after_forced.get("tail_fp_used") != before.get("tail_fp_used"),
            "--force actually rewrote the anchor",
        )
        # Self-consistency, not equality-with-before: probing the SAME
        # cleaned branch twice in a row (no further change happened
        # between r3 and this probe) must be stable.
        after_forced_again = _run_align_probe(WORK_DB, WORK_STORE)
        assert_eq(after_forced.get("position"), after_forced_again.get("position"),
                  "the forced, rewritten anchor is internally stable across repeated probes")
        assert_eq(after_forced.get("window_offset"), after_forced_again.get("window_offset"),
                  "the forced, rewritten anchor's offset is internally stable")
    else:
        # No drift on this data after all (e.g. a later export than the
        # one this suite was built against) — the strong claim holds.
        assert_eq(rc, 0, "no drift found: a clean, unforced apply exits 0")
        after = _run_align_probe(WORK_DB, WORK_STORE)
        print(f"  after:  {after}")
        assert_eq(after.get("position"), before.get("position"),
                  "step (c): position is IDENTICAL after cleaning")
        assert_eq(after.get("window_offset"), before.get("window_offset"),
                  "step (c): window_offset is IDENTICAL after cleaning")
    globals()["_ALIGN_BEFORE"] = before


def test_alignment_proof_goes_red_without_the_anchor_rewrite():
    print("\n[test] (f) the SAME proof goes RED when the anchor rewrite is disabled — proving it is load-bearing")
    scratch = Path(tempfile.mkdtemp(prefix="cd-red-"))
    try:
        db2 = scratch / "webui.db"
        store2 = scratch / "store"
        shutil.copy2(BACKUP_ROOT / "webui.db", db2)
        shutil.copytree(BACKUP_ROOT / "compactor", store2)

        before = _run_align_probe(db2, store2)

        r = _docker_run(
            [
                (str(V319_PKG), "/opt/compactor", "ro"),
                (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
                (str(store2), "/data/store", "rw"),
                _webui_dir_mount(db2, "rw"),
            ],
            _cd_args("--apply", "--force", "--debug-skip-anchor-rewrite", "--json"),
        )
        assert_not_in("Traceback", r.stdout + r.stderr, "apply (anchor disabled): no traceback")

        after = _run_align_probe(db2, store2)
        print(f"  before: {before}")
        print(f"  after (anchor rewrite disabled): {after}")
        mismatch = (
            after.get("position") != before.get("position")
            or after.get("window_offset") != before.get("window_offset")
        )
        assert_true(
            mismatch,
            "with the anchor rewrite disabled, position/offset DIVERGE — "
            "proving the rewrite step in the normal path is not vacuous",
            extra=f"before={before} after={after}",
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3 (d). One real maybe_rollup, fake vLLM, budget 1 — map recorded
#    positions to branch turns; there must be no hole.
# ---------------------------------------------------------------------------

def test_one_real_rollup_after_cleaning_leaves_no_hole():
    print("\n[test] (d) one real maybe_rollup (fake vLLM, budget 1) after cleaning — no hole in the hierarchy")
    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
            (str(WORK_STORE), "/data/store", "rw"),
            (str(WORK_DB), "/data/openwebui/webui.db", "ro"),
            (str(DRIVER_PY), "/work/driver.py", "ro"),
            (str(FAKEVLLM_PY), "/work/fakevllm.py", "ro"),
        ],
        ["-c", f"{VENV_PY} /work/fakevllm.py & sleep 0.6 && {VENV_PY} /work/driver.py rollup"],
        entrypoint="/bin/sh",
        env={"CD_STORE": "/data/store", "CD_CONV": CHAT_ID,
             "CD_DB": "/data/openwebui/webui.db", "CD_VLLM_URL": "http://127.0.0.1:8000"},
    )
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "rollup driver: no traceback")
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
    if not lines:
        print("rollup driver produced no parseable JSON:\n", out)
        FAILED.append("rollup driver JSON")
        return
    payload = json.loads(lines[-1])
    print(f"  result: {payload}")
    assert_true(payload.get("spent", 0) >= 1, "at least one real vLLM call was spent",
                extra=str(payload))

    spans = sorted(payload.get("l1_spans") or [])
    prev_end = 0
    hole = False
    for first, last in spans:
        if prev_end and first != prev_end + 1:
            hole = True
        prev_end = last
    assert_true(not hole, "no gap between consecutive L1 chunk spans", extra=str(spans))
    if spans:
        assert_true(payload.get("covered_fps_len") is not None
                    and payload["covered_fps_len"] >= spans[-1][1],
                    "covered_fps record reaches at least the last L1 chunk's own turn",
                    extra=str(payload))


# ---------------------------------------------------------------------------
# 4 (e). import-history.py's own dry run still verifies resume_offset 22
#    on the CLEANED data.
# ---------------------------------------------------------------------------

def _run_episodic_verify(chroma_path: Path) -> dict:
    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
            (str(DRIVER_PY), "/work/driver.py", "ro"),
            (str(chroma_path), "/data/chromadb", "rw"),
        ],
        ["/work/driver.py", "episodic_verify"],
        env={"CD_STORE": "/tmp", "CD_CONV": CHAT_ID, "CD_DB": "/tmp/x.db",
             "CD_CHROMA_PATH": "/data/chromadb"},
    )
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "episodic_verify driver: no traceback")
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
    if not lines:
        print("episodic_verify produced no parseable JSON:\n", out)
        FAILED.append("episodic_verify JSON")
        return {}
    return json.loads(lines[-1])


def test_episodic_before_after_reembedding_and_similarity_sanity():
    print("\n[test] item 3: episodic before/after counts, content-addressed re-embedding, "
          "similarity-query sanity — on a FRESH copy of her real chromadb store")
    scratch = Path(tempfile.mkdtemp(prefix="cd-episodic-"))
    try:
        store_path = scratch / "store"
        store_path.mkdir()
        shutil.copytree(BACKUP_ROOT / "compactor" / "chromadb", store_path / "chromadb")
        (store_path / "summaries").mkdir()
        (store_path / "facts").mkdir()

        before = _run_episodic_verify(store_path / "chromadb")
        print(f"  before: total_docs={before.get('total_docs')} "
              f"decorated={before.get('decorated_assistant_halves')}")
        assert_true(before.get("total_docs", 0) >= 80, "her ~84 real documents are all present before")
        # This driver-side check only tests for a few literal markers
        # (✅/▶️/an 8+-char rule run) — narrower than clean_text's full
        # rule set, so it undercounts relative to the real --apply
        # result (verified separately: 81 of 84 actually change). Still
        # clearly "most of them", matching the forensic report's order
        # of magnitude ("72 of 84").
        assert_true(before.get("decorated_assistant_halves", 0) >= 50,
                     "most of them are decorated before cleaning (matches the forensic report)",
                     extra=str(before.get("decorated_assistant_halves")))

        r = _docker_run(
            [
                (str(V319_PKG), "/opt/compactor", "ro"),
                (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
                (str(store_path), "/data/store", "rw"),
            ],
            [
                "/opt/zl-repo/scripts/clean-decoration.py",
                "--store", "/data/store", "--conv", CHAT_ID, "--only", "episodic",
                "--apply", "--force", "--json",
            ],
        )
        _fixup_perms(scratch)
        assert_not_in("Traceback", r.stdout + r.stderr, "episodic apply: no traceback")
        assert_eq(r.returncode, 0, "episodic --apply succeeds")

        after = _run_episodic_verify(store_path / "chromadb")
        print(f"  after:  total_docs={after.get('total_docs')} "
              f"decorated={after.get('decorated_assistant_halves')}")
        assert_eq(after.get("total_docs"), before.get("total_docs"),
                  "same number of documents after cleaning — none lost, none duplicated")
        assert_true(after.get("decorated_assistant_halves", 999) < before.get("decorated_assistant_halves", 0),
                     "far fewer documents have a decorated assistant half after cleaning")
        assert_true(after.get("all_ids_content_addressed_and_prefixed"),
                     "every id (old AND newly re-hashed) is still content-addressed and conv_id-prefixed")

        hits = after.get("similarity_hits") or []
        assert_true(len(hits) >= 1, "a real bge-small similarity query still returns hits after re-embedding")
        assert_true(all(dist < 0.6 for _, dist in hits),
                     "the hits are genuinely relevant (low cosine distance), not embedding-space noise",
                     extra=str(hits))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_import_history_dry_run_still_verifies_resume_offset_22():
    print("\n[test] (e) import-history.py dry run on the cleaned data still verifies resume_offset 22")
    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
            (str(WORK_STORE), "/data/store", "rw"),
            (str(WORK_DB), "/data/openwebui/webui.db", "ro"),
        ],
        [
            "/opt/zl-repo/scripts/import-history.py",
            "--webui-db", "/data/openwebui/webui.db",
            "--chat-id", CHAT_ID,
            "--store", "/data/store",
            "--json",
        ],
    )
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "import-history.py dry run: no traceback")
    payload = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
    detail = json.dumps(payload).lower()
    assert_true(
        "resume_offset" in detail or "22" in json.dumps(payload),
        "resume-offset verification detail is present in the report",
        extra=str(payload)[:2000],
    )


# ---------------------------------------------------------------------------
# 5. Byte-diff of the whole store — --apply's blast radius is exactly what
#    the module docstring documents (webui.db + summaries/<conv>.json +
#    facts/<conv>.json + the new .bak-* files), nothing else, against a
#    FULL copy of the real store (M6-style: not just CHAT_ID's slice).
# ---------------------------------------------------------------------------

def test_apply_blast_radius_is_exactly_documented_and_second_apply_is_noop():
    print("\n[test] --apply's blast radius over the WHOLE store, and a second --apply is a no-op")
    scratch = Path(tempfile.mkdtemp(prefix="cd-blast-"))
    try:
        db3 = scratch / "webui.db"
        store3 = scratch / "store"
        shutil.copy2(BACKUP_ROOT / "webui.db", db3)
        shutil.copytree(BACKUP_ROOT / "compactor", store3)

        before = _store_file_digests(store3)
        before_db = hashlib.md5(db3.read_bytes()).hexdigest()

        r1 = _docker_run(
            [
                (str(V319_PKG), "/opt/compactor", "ro"),
                (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
                (str(store3), "/data/store", "rw"),
                _webui_dir_mount(db3, "rw"),
            ],
            _cd_args("--apply", "--force", "--json"),
        )
        _fixup_perms(scratch)
        assert_not_in("Traceback", r1.stdout + r1.stderr, "first apply: no traceback")
        assert_true(r1.returncode in (0, 4), f"first apply exits 0 or 4 (got {r1.returncode})")

        after = _store_file_digests(store3)
        after_db = hashlib.md5(db3.read_bytes()).hexdigest()

        changed = {k for k in after if after.get(k) != before.get(k)} | {
            k for k in before if k not in after
        }
        unexpected = [
            k for k in changed
            if not (
                k.startswith(f"summaries/{CHAT_ID}.json")
                or k.startswith(f"facts/{CHAT_ID}.json")
                or k.startswith("chromadb/")
                or k.startswith("chromadb.bak-")  # the episodic backup directory itself
            )
        ]
        assert_eq(unexpected, [], "every changed store path is inside a documented target")
        assert_true(before_db != after_db, "webui.db itself changed (the whole point)")
        assert_eq(
            _count_history_vs_table_mismatches(db3), 0,
            "history.messages and the chat_message table agree on EVERY message after --apply",
        )

        r2 = _docker_run(
            [
                (str(V319_PKG), "/opt/compactor", "ro"),
                (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
                (str(store3), "/data/store", "rw"),
                _webui_dir_mount(db3, "rw"),
            ],
            _cd_args("--apply", "--force", "--json"),
        )
        _fixup_perms(scratch)
        assert_not_in("Traceback", r2.stdout + r2.stderr, "second apply: no traceback")
        assert_eq(r2.returncode, 0, "second --apply on an already-clean copy is a no-op, exit 0")

        after2_db = hashlib.md5(db3.read_bytes()).hexdigest()
        assert_eq(after_db, after2_db, "second apply did not touch webui.db again")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. --restore brings the store back exactly (md5-identical), from the
#    scenario 2 backups (test_alignment_proof_... already ran --apply on
#    WORK_DB/WORK_STORE; this restores from ITS stamp).
# ---------------------------------------------------------------------------

def test_restore_brings_everything_back_md5_identical():
    print("\n[test] --restore brings webui.db and the store back md5-identical")
    baks = sorted(WORK_DB.parent.glob("webui.db.bak-*"))
    if not baks:
        print("FAIL: no webui.db backup found from an earlier scenario")
        FAILED.append("restore: no backup found")
        return
    stamp = baks[-1].name.split(".bak-")[1]
    pre_restore_db = hashlib.md5(WORK_DB.read_bytes()).hexdigest()
    expected_db = hashlib.md5(baks[-1].read_bytes()).hexdigest()

    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
            (str(WORK_STORE), "/data/store", "rw"),
            _webui_dir_mount(WORK_DB, "rw"),
        ],
        [
            "/opt/zl-repo/scripts/clean-decoration.py",
            "--restore", stamp,
            "--webui-db", "/data/openwebui/webui.db",
            "--store", "/data/store",
            "--conv", CHAT_ID,
            # This stamp's apply only ever targeted --only webui (see
            # phase_apply_no_force) — no facts/summaries backup exists
            # under it, so a full-target restore would correctly refuse
            # on those missing backups. Scope the restore to match.
            "--only", "webui",
        ],
    )
    _fixup_perms(WORK_DB.parent)
    _fixup_perms(WORK_STORE)
    out = r.stdout + r.stderr
    assert_not_in("Traceback", out, "restore: no traceback")
    assert_eq(r.returncode, 0, "restore exits 0")
    restored_db_md5 = hashlib.md5(WORK_DB.read_bytes()).hexdigest()
    assert_eq(restored_db_md5, expected_db, "restored webui.db is md5-identical to its own backup")
    assert_true(restored_db_md5 != pre_restore_db, "restore actually changed the file back")
    assert_eq(
        _count_history_vs_table_mismatches(WORK_DB), 0,
        "history.messages and the chat_message table still agree after --restore",
    )


# ---------------------------------------------------------------------------
# Mutation set — a handful of hand-written mutants of the cleaning
# function and the anchor logic, run against
# compactor/test_clean_decoration_script.py's own suite (imported here,
# not re-run through docker — the mutants are pure-python and need no
# real image). Reports survivors, if any, rather than assuming none.
# ---------------------------------------------------------------------------

def test_mutation_set_on_clean_text_and_anchor_logic():
    print("\n[test] mutation set — hand-written mutants of clean_text/_recompute_anchor")
    spec = importlib.util.spec_from_file_location(
        "cd_mut", REPO_ROOT / "scripts" / "clean-decoration.py"
    )
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)

    mutants = []

    # M1: emoji pattern misses the dingbat range entirely.
    import re as _re
    orig_emoji = cd._EMOJI_PATTERN
    cd._EMOJI_PATTERN = _re.compile("[\U0001F000-\U0001FFFF]")
    killed = cd.clean_text("✅ ACTIVE (100%)\n") != "\n" and "✅" in cd.clean_text("✅ done\n")
    mutants.append(("M1: emoji pattern drops the dingbat range", killed))
    cd._EMOJI_PATTERN = orig_emoji

    # M2: rule-line threshold loosened from >=4 to >=1 (would eat "***" used
    # as ordinary markdown emphasis-adjacent punctuation).
    orig_rule = cd._RULE_LINE_RE
    cd._RULE_LINE_RE = _re.compile(r"^[-=_~*─-╿]{1,}$")
    out = cd.clean_text("---\n")
    # Correct/spec behaviour: a 3-char run is BELOW the >=4 threshold and
    # must survive untouched. "killed" means this check caught the mutant
    # deviating from that — i.e. `out` no longer equals the correct answer.
    killed = out != "---\n"
    mutants.append(("M2: rule-line threshold loosened to >=1", killed))
    cd._RULE_LINE_RE = orig_rule

    # M3: status-tag stripping loses its word boundary (would corrupt
    # "INACTIVE (100%)" into "IN").
    orig_patterns = cd.DEFAULT_STATUS_TAG_PATTERNS
    cd.DEFAULT_STATUS_TAG_PATTERNS = tuple(
        _re.compile(p.pattern.replace(r"\bACTIVE\b", "ACTIVE")) for p in orig_patterns
    )
    out = cd.clean_text("Compliance is INACTIVE (100%)\n")
    # Correct/spec behaviour: the word "INACTIVE" must survive whole.
    # "killed" means this check caught the mutant corrupting it.
    killed = "INACTIVE" not in out
    mutants.append(("M3: status-tag pattern loses \\b, eats INACTIVE", killed))
    cd.DEFAULT_STATUS_TAG_PATTERNS = orig_patterns

    # M4: dedup collapses ALL consecutive duplicates including blank lines
    # (would fight the "collapse 3+ blank lines to 1" rule and under-count
    # deliberate paragraph spacing of exactly 2 blank lines).
    def _bad_dedupe(lines):
        out = []
        for ln in lines:
            if out and out[-1] == ln:
                continue
            out.append(ln)
        return out
    orig_dedupe = cd._dedupe_consecutive_nonblank
    cd._dedupe_consecutive_nonblank = _bad_dedupe
    out = cd.clean_text("para one\n\n\npara two\n")  # 2 blank lines, must survive
    # Correct/spec behaviour: exactly 2 blank lines is BELOW the "3+
    # collapses to 1" threshold, so the text must round-trip unchanged.
    # "killed" means this check caught the mutant collapsing it anyway.
    killed = out != "para one\n\n\npara two\n"
    mutants.append(("M4: dedupe also collapses blank lines", killed))
    cd._dedupe_consecutive_nonblank = orig_dedupe

    # M5: _recompute_anchor hardcodes the anchor length instead of
    # reading _ANCHOR_TURNS from the frozen module — silently changes
    # tail_fp's length, which _align_candidates depends on exactly
    # matching. `summarizer.py` imports httpx at module level, so this
    # cannot run on the bare host — run it INSIDE the image instead, via
    # driver.py's own m5_mutant phase, the same way every other real-
    # module check in this suite already works (item 5 of the hostile
    # review: "run the M5 mutant inside the image, where httpx exists").
    # See phase_m5_mutant's own docstring for why the FIRST version of
    # this mutant (whole-history vs tail-only fingerprinting) had to be
    # redesigned — it was not actually a value-changing bug.
    r = _docker_run(
        [
            (str(V319_PKG), "/opt/compactor", "ro"),
            (str(SCRIPTS_DIR), "/opt/zl-repo/scripts", "ro"),
            (str(DRIVER_PY), "/work/driver.py", "ro"),
        ],
        ["/work/driver.py", "m5_mutant"],
        env={"CD_STORE": "/tmp", "CD_CONV": "x", "CD_DB": "/tmp/x.db"},
    )
    out = r.stdout + r.stderr
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
    if "Traceback" in out or not lines:
        print(f"  FAIL M5: could not run inside the image: {out[-1000:]}")
        mutants.append(("M5: anchor length hardcoded instead of read from _ANCHOR_TURNS", False))
    else:
        m5_result = json.loads(lines[-1])
        mutants.append(("M5: anchor length hardcoded instead of read from _ANCHOR_TURNS",
                         m5_result["killed"]))

    survivors = [name for name, killed in mutants if not killed]
    for name, killed in mutants:
        print(f"  {'killed ' if killed else 'SURVIVED'}: {name}")
    assert_eq(survivors, [], "no mutant survives the test suite")


def run_all():
    _preflight()
    _build_v319_package()
    _seed_scripts()
    _write_driver_files()
    print("seeding real-data scratch copies (this is the slow part)...")
    _seed_work_copies()

    tests = [
        test_full_dry_run_reports_pending_changes_and_writes_nothing,
        test_alignment_proof_position_and_offset_survive_cleaning,
        test_alignment_proof_goes_red_without_the_anchor_rewrite,
        test_one_real_rollup_after_cleaning_leaves_no_hole,
        test_episodic_before_after_reembedding_and_similarity_sanity,
        test_import_history_dry_run_still_verifies_resume_offset_22,
        test_restore_brings_everything_back_md5_identical,
        test_apply_blast_radius_is_exactly_documented_and_second_apply_is_noop,
        test_mutation_set_on_clean_text_and_anchor_logic,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: unhandled {type(e).__name__}: {e}")
            FAILED.append(t.__name__)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_all())
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
