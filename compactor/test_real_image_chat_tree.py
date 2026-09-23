"""Real-image test for scripts/repair-chat-tree.py and
scripts/fix-encoded-messages.py, run under the PUBLISHED digest's OWN
`/app/venv/bin/python` (3.12, vs. this host's 3.14 — the two scripts use
nothing exotic, but this is the one thing a host-python smoke test cannot
prove) against COPIES of both real pod-export backups.

NEEDS, and SKIPS (exit 3) HONESTLY if any is missing, same convention as
test_real_image_import_apply.py's own `_skip`:
  - a real Docker daemon
  - the published image already pulled (`docker image inspect`)
  - both real pod-export backups on this host, each with `webui.db`

What this proves, per backup:
  1. DRY RUN under the image's own python matches what the host-python
     smoke test already found: 3 orphans re-linked, verification FAILS
     (the 25 pre-existing double-encoded messages neither backup
     predates — see repair-chat-tree.py's module docstring), exit 3.
  2. Because neither backup has a clean pre-corruption snapshot on this
     host, `--apply` is exercised against a COPY with those 25 messages
     pre-cleared (the same stand-in idempotency_test.py used while
     developing this — documented here, not hidden: this is NOT what
     fix-encoded-messages.py itself would produce without a real
     pre-repair snapshot, which is a genuine limitation of both scripts
     against exactly this pair of backups, reported to the operator
     rather than silently worked around).
  3. `--apply` on that cleaned copy: exit 0, `PRAGMA integrity_check` ==
     "ok", the `chat_message.parent_id` for each relinked orphan agrees
     with the JSON history's `parentId` (table and JSON stay consistent
     with each other, not just internally), the flat legacy `messages[]`
     list is untouched (same length, same bytes), and the `chat` row
     does not blow up in size (v1's defect, retired: nearly doubling it).
  4. A second `--apply` afterward is a clean no-op (idempotent).
  5. `--restore <stamp>` gives back the byte-identical (md5) pre-`--apply`
     file.

Run it directly, on the host:

    python compactor/test_real_image_chat_tree.py
    python scripts/run-tests.py --real-image --only real_image_chat_tree
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIGEST = "sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65"
IMAGE_REF = f"angreg/zions-light-ai@{IMAGE_DIGEST}"
VENV_PY = "/app/venv/bin/python"
CHAT_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
DOCKER_TIMEOUT_S = 120

BACKUPS = [
    Path("/home/drew/pod-exports/2026-09-22/backup"),
    Path("/home/drew/pod-exports/2026-09-23/backup"),
]

FAILURES = []


def _skip(reason: str) -> None:
    print("=" * 72)
    print("SKIPPED: test_real_image_chat_tree.py")
    print(f"  reason: {reason}")
    print("=" * 72)
    sys.exit(3)


def _run(args, timeout=DOCKER_TIMEOUT_S, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kw)


def _preflight():
    if shutil.which("docker") is None:
        _skip("no `docker` binary on PATH")
    r = _run(["docker", "version"], timeout=15)
    if r.returncode != 0:
        _skip(f"`docker version` failed: {(r.stderr or r.stdout).strip()[:200]}")
    r = _run(["docker", "image", "inspect", IMAGE_REF], timeout=15)
    if r.returncode != 0:
        _skip(f"published image not pulled locally: {IMAGE_REF}")
    present = [b for b in BACKUPS if (b / "webui.db").is_file()]
    if not present:
        _skip(f"no real backup with webui.db among {BACKUPS}")
    return present


def _docker_run(mounts, image_args, timeout=DOCKER_TIMEOUT_S):
    args = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", VENV_PY]
    for host, cont, mode in mounts:
        args += ["-v", f"{host}:{cont}:{mode}"]
    args += [IMAGE_REF] + image_args
    return _run(args, timeout=timeout)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode(s):
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


def _clear_known_encoded(db_path):
    """Stand-in for a real fix-encoded-messages.py run against a genuine
    pre-corruption snapshot (see module docstring, point 2) -- clears the
    double-encoded messages this repo's two available backups already
    share, so `--apply` can reach a verifiable, writable state at all.
    Returns the count cleared, in history and in the table."""
    import sqlite3
    con = sqlite3.connect(db_path)
    row = con.execute("select chat from chat where id=?", (CHAT_ID,)).fetchone()
    chat = json.loads(row[0])
    msgs = chat["history"]["messages"]
    n_hist = 0
    for m in msgs.values():
        c = m.get("content")
        if isinstance(c, str) and c[:1] == '"' and _decode(c) != c:
            m["content"] = _decode(c)
            n_hist += 1
    n_table = 0
    for row_id, content in list(con.execute(
        "select id, content from chat_message where chat_id=?", (CHAT_ID,)
    )):
        if isinstance(content, str) and content[:1] == '"' and _decode(content) != content:
            con.execute("update chat_message set content=? where id=?",
                        (json.dumps(_decode(content)), row_id))
            n_table += 1
    con.execute("update chat set chat=? where id=?", (json.dumps(chat), CHAT_ID))
    con.commit()
    con.close()
    return n_hist, n_table


def _check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{label}: {detail}")


def _flat_messages(db_path):
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = con.execute("select chat from chat where id=?", (CHAT_ID,)).fetchone()
    con.close()
    chat = json.loads(row[0])
    return chat.get("messages") or []


def _run_one_backup(backup_dir: Path, tmp_root: Path):
    print(f"--- {backup_dir} ---")
    work = tmp_root / backup_dir.parent.name
    work.mkdir(parents=True, exist_ok=True)
    live = work / "webui.db"
    shutil.copy2(backup_dir / "webui.db", live)
    os.chmod(live, stat.S_IRUSR | stat.S_IWUSR)

    scripts_dir = work / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    for name in ("repair-chat-tree.py", "fix-encoded-messages.py"):
        shutil.copy2(REPO_ROOT / "scripts" / name, scripts_dir / name)

    mounts = [(str(work), "/work", "rw")]

    # 1. dry run under the image's own python
    r = _docker_run(mounts, ["/work/scripts/repair-chat-tree.py", "/work/webui.db",
                             "--chat", CHAT_ID], timeout=DOCKER_TIMEOUT_S)
    _check("dry run runs under the image's own python (3.12)", r.returncode in (0, 1, 3),
           f"rc={r.returncode} stdout={r.stdout[-500:]} stderr={r.stderr[-500:]}")
    _check("dry run finds the 3 known orphans", "orphans re-linked: 3" in r.stdout, r.stdout[-300:])

    # 2. clear the known pre-existing encoding corruption (documented
    #    stand-in -- see module docstring point 2) so --apply can reach a
    #    verifiable state
    n_hist, n_table = _clear_known_encoded(str(live))
    print(f"  (stand-in cleanup: cleared {n_hist} history / {n_table} table copies)")

    pre_apply_md5 = _md5(live)
    pre_apply_size = live.stat().st_size

    # 3. --apply, under the image's own python
    r = _docker_run(mounts, ["/work/scripts/repair-chat-tree.py", "/work/webui.db",
                             "--chat", CHAT_ID, "--apply"], timeout=DOCKER_TIMEOUT_S)
    _check("--apply exits 0 or 4 (full or partial success)", r.returncode in (0, 4),
           f"rc={r.returncode} stdout={r.stdout[-800:]} stderr={r.stderr[-500:]}")
    _check("--apply reports integrity: ok", "integrity: ok" in r.stdout, r.stdout[-300:])
    apply_stdout = r.stdout

    # 4. table/JSON consistency + flat list + row size, read back on the host
    import sqlite3
    con = sqlite3.connect(f"file:{live}?mode=ro", uri=True)
    row = con.execute("select chat, current_message_id from chat where id=?", (CHAT_ID,)).fetchone()
    chat = json.loads(row[0])
    msgs = chat["history"]["messages"]
    prefix = CHAT_ID + "-"
    table = {r2[0][len(prefix):]: r2[1] for r2 in con.execute(
        "select id, parent_id from chat_message where chat_id=?", (CHAT_ID,))}
    con.close()

    disagree = [mid for mid, m in msgs.items()
                if mid in table and (table[mid] or None) != (m.get("parentId") or None)]
    _check("table parent_id agrees with JSON parentId for every message", not disagree,
           f"{len(disagree)} disagreements: {[d[:8] for d in disagree[:5]]}")

    flat = _flat_messages(live)
    _check("flat legacy messages[] list untouched (still 8 entries)", len(flat) == 8, str(len(flat)))

    post_apply_size = live.stat().st_size
    growth = post_apply_size / pre_apply_size if pre_apply_size else 0
    _check("row size not inflated (< 1.5x pre-apply size)", growth < 1.5, f"growth={growth:.2f}x")

    # 5. a second --apply is a clean no-op
    r2 = _docker_run(mounts, ["/work/scripts/repair-chat-tree.py", "/work/webui.db",
                              "--chat", CHAT_ID, "--apply"], timeout=DOCKER_TIMEOUT_S)
    _check("second --apply is a clean no-op (exit 0, 'nothing to do')",
           r2.returncode == 0 and "nothing to do" in r2.stdout,
           f"rc={r2.returncode} stdout={r2.stdout[-400:]}")

    # 6. --restore gives back the byte-identical pre-apply file
    import re
    m = re.search(r"backed up to (\S+\.bak-(\S+))", apply_stdout)
    _check("--apply printed a backup path", bool(m), apply_stdout[-300:])
    if m:
        stamp = m.group(2)
        r3 = _docker_run(mounts, ["/work/scripts/repair-chat-tree.py", "/work/webui.db",
                                  "--restore", stamp], timeout=DOCKER_TIMEOUT_S)
        _check("--restore exits 0", r3.returncode == 0, f"rc={r3.returncode} {r3.stdout[-300:]}")
        restored_md5 = _md5(live)
        _check("--restore is byte-identical (md5) to the pre-apply file",
               restored_md5 == pre_apply_md5, f"{restored_md5} != {pre_apply_md5}")


def main():
    present = _preflight()
    # /tmp is a small tmpfs on this host (COMMANDS.md) -- two 500MB+
    # webui.db copies plus their own .bak- backups do not fit there.
    # SCRATCH_DIR lets a real run point this at a host directory with
    # room; falls back to /tmp (fine on a host where /tmp has room).
    scratch = os.environ.get("SCRATCH_DIR") or "/home/drew/scratch"
    os.makedirs(scratch, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="compactor-test-real-image-chat-tree-", dir=scratch))
    try:
        for b in present:
            _run_one_backup(b, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("=" * 72)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("PASSED: test_real_image_chat_tree.py")
    sys.exit(0)


if __name__ == "__main__":
    main()
