"""
Tests for scripts/backfill-records.py — the operator tool that inspects
compactor backfill records and, only with --apply, closes the stale
`would-resume` ones before an upgrade past v3.1.9.3 resumes them (see that
script's own module docstring, and backfill.needs_backfill's).

Seeded from the REAL shape of the four stale in_progress records verified
on a 2026-09-22 production backup (conv ids and exchange counts kept
exactly as they were there), plus a handful of terminal and malformed
records built the same way backfill.py itself writes them
(_write_failed_or_abandoned, _write_wiped) to cover every verdict this
script produces.

Run inside the compactor image or any container with the requirements
installed:
    python test_backfill_records_script.py
"""

import hashlib
import http.server
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-backfill-records-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import backfill  # noqa: E402
import facts  # noqa: E402
import memory  # noqa: E402

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "backfill-records.py"

_spec = importlib.util.spec_from_file_location("backfill_records_script", _SCRIPT)
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)


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


def _wipe_storage():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


def _iso(dt):
    return dt.isoformat(timespec="seconds")


def _record_path(conv_id):
    return memory.storage_root() / "facts" / f"{conv_id}.backfill.json"


def _write_record(conv_id, **fields):
    path = _record_path(conv_id)
    data = {"conv_id": conv_id}
    data.update(fields)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_facts(conv_id, n=1):
    facts.save_facts(conv_id, [
        {"text": f"fact {i}", "added_turn": i, "last_used": 1} for i in range(n)
    ])


LONG_MESSAGES = [
    {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
]

# The four real stale in_progress records, verified on the 2026-09-22
# production backup (conv_id, exchanges_done, exchanges_total).
REAL_STALE_RECORDS = [
    ("ea1494ea-e9d7-46fb-8b7c-3a50d685d00e", 6, 1908),
    ("d6fd08f99548b229", 296, 793),
    ("3863405c69ea10ad", 572, 732),
    ("25887c0dc221f451", 589, 626),
]


def _seed_real_stale_records(old_hours=2):
    """The four real stale records, each with a facts file already on disk
    — matching the real backup, where every one of the four conversations
    already has a live tail's facts sitting alongside the crashed
    backfill."""
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=old_hours))
    for conv_id, done, total in REAL_STALE_RECORDS:
        _write_record(
            conv_id, state="in_progress", started_at=old_ts, updated_at=old_ts,
            exchanges_done=done, exchanges_total=total, attempts=1, error=None,
        )
        _write_facts(conv_id)


def _seed_terminal_records():
    """A complete, an abandoned and a wiped record, in the exact shape
    backfill.py's own writers use. Real backup only has 'complete' ones —
    'abandoned' and 'wiped' are built here, not copied, because no
    conversation in the real backup happens to be in either state."""
    now = _iso(datetime.now(timezone.utc))
    _write_record("term-complete", state="complete", started_at=now, updated_at=now,
                  exchanges_done=5, exchanges_total=5, attempts=1, error=None)
    _write_facts("term-complete")
    _write_record("term-abandoned", state="abandoned", started_at=now, updated_at=now,
                  exchanges_done=2, exchanges_total=10, attempts=3, error="boom")
    _write_facts("term-abandoned")
    # Deliberately no facts file for the wiped one — a real /forget leaves
    # an empty facts file behind, but this script only checks existence,
    # and a wiped record must stay `leave` regardless of that file.
    _write_record("term-wiped", state="wiped", started_at=now, updated_at=now,
                  exchanges_done=1, exchanges_total=10, attempts=1, error=None)


def _all_backfill_bytes():
    d = memory.storage_root() / "facts"
    return {
        p.name: p.read_bytes()
        for p in sorted(d.glob("*.backfill.json"))
    }


def _run_script(args):
    """Run the real script as a subprocess — its argparse / __main__ path,
    exactly as an operator's shell invocation would see it (same pattern as
    test_p4b_g7_switch_script.py's _run_switch_script)."""
    return subprocess.run(
        [sys.executable, str(_SCRIPT)] + list(args),
        capture_output=True, text=True, timeout=60,
    )


def _store_args():
    return ["--store", str(memory.storage_root())]


# ---------------------------------------------------------------------------
# Dry run: reports, never writes
# ---------------------------------------------------------------------------

def test_dry_run_changes_nothing_on_disk_and_exits_3():
    print("\n[test] dry run leaves every backfill record byte-identical and exits 3")
    _wipe_storage()
    _seed_real_stale_records()
    _seed_terminal_records()
    before = _all_backfill_bytes()

    r = _run_script(_store_args())
    after = _all_backfill_bytes()

    assert_eq(r.returncode, 3, "dry run with would-resume records exits 3")
    assert_eq(before, after, "no backfill record changed a single byte")
    for conv_id, _, _ in REAL_STALE_RECORDS:
        assert_true(conv_id in r.stdout, f"{conv_id} appears in the dry-run report")
    assert_true("would-resume" in r.stdout, "report names the would-resume verdict")


def test_dry_run_no_stale_records_exits_0():
    print("\n[test] dry run with nothing stale exits 0")
    _wipe_storage()
    _seed_terminal_records()
    r = _run_script(_store_args())
    assert_eq(r.returncode, 0, "no would-resume records -> exit 0")


# ---------------------------------------------------------------------------
# --apply: closes only the would-resume records that have facts
# ---------------------------------------------------------------------------

def test_apply_rewrites_only_the_four_would_resume_records():
    print("\n[test] --apply closes the four would-resume records and leaves complete/abandoned/wiped untouched")
    _wipe_storage()
    _seed_real_stale_records()
    _seed_terminal_records()
    before = _all_backfill_bytes()

    r = _run_script(_store_args() + ["--apply", "--json"])
    assert_eq(r.returncode, 0, f"--apply succeeds (stderr={r.stderr!r})")
    payload = json.loads(r.stdout)
    closed_ids = {c["conv_id"] for c in payload["changes"] if c["action"] == "closed"}
    assert_eq(closed_ids, {c for c, _, _ in REAL_STALE_RECORDS},
              "exactly the four real stale records were closed")

    after = _all_backfill_bytes()
    for name in ("term-complete.backfill.json", "term-abandoned.backfill.json",
                 "term-wiped.backfill.json"):
        assert_eq(before[name], after[name], f"{name} untouched by --apply")

    for conv_id, done, total in REAL_STALE_RECORDS:
        rec = json.loads(_record_path(conv_id).read_text())
        assert_eq(rec["state"], "abandoned", f"{conv_id} state -> abandoned")
        assert_eq(rec["exchanges_done"], done, f"{conv_id} exchanges_done preserved")
        assert_eq(rec["exchanges_total"], total, f"{conv_id} exchanges_total preserved")
        assert_eq(rec["closed_by"], "scripts/backfill-records.py", f"{conv_id} closed_by set")
        assert_true("closed_at" in rec, f"{conv_id} closed_at set")
        backups = list(_record_path(conv_id).parent.glob(
            f"{conv_id}.backfill.json.bak-*"
        ))
        assert_eq(len(backups), 1, f"{conv_id} has exactly one backup file")
        original = json.loads(backups[0].read_text())
        assert_eq(original["state"], "in_progress", "backup holds the ORIGINAL pre-apply state")


# ---------------------------------------------------------------------------
# The real point of the script: needs_backfill() agrees afterwards
# ---------------------------------------------------------------------------

def test_closed_records_make_needs_backfill_return_false():
    print("\n[test] after --apply, backfill.needs_backfill() returns False for all four conversations")
    _wipe_storage()
    _seed_real_stale_records()
    _seed_terminal_records()

    # Before closing: this is the actual hazard — v3.1.9.4+ would resume these.
    for conv_id, _, _ in REAL_STALE_RECORDS:
        assert_eq(backfill.needs_backfill(conv_id, LONG_MESSAGES), True,
                  f"{conv_id}: needs_backfill() is True before closing (the hazard)")

    r = _run_script(_store_args() + ["--apply"])
    assert_eq(r.returncode, 0, "--apply succeeds")

    for conv_id, _, _ in REAL_STALE_RECORDS:
        assert_eq(backfill.needs_backfill(conv_id, LONG_MESSAGES), False,
                  f"{conv_id}: needs_backfill() is False after closing (the fix)")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_would_resume_with_no_facts_file_refused_without_force():
    print("\n[test] a would-resume record with no facts file is refused unless --force")
    _wipe_storage()
    conv_id = "no-facts-yet"
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    _write_record(conv_id, state="in_progress", started_at=old_ts, updated_at=old_ts,
                  exchanges_done=3, exchanges_total=50, attempts=1, error=None)
    # deliberately no _write_facts(conv_id) — this is the "never backfilled
    # at all yet" case the script must not cancel.
    before = _record_path(conv_id).read_bytes()

    r = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--json"])
    assert_eq(r.returncode, 1,
              "H1: --apply that closed NONE of its would-resume targets exits 1, "
              "not 0 — refusing this one record left the hazard fully in place")
    payload = json.loads(r.stdout)
    assert_eq(payload["changes"][0]["action"], "refused-no-facts",
              "the record is refused, not closed")
    assert_eq(_record_path(conv_id).read_bytes(), before,
              "the record was not written to")
    assert_eq(list(_record_path(conv_id).parent.glob(f"{conv_id}.backfill.json.bak-*")),
              [], "no backup was created for a refused record")

    r2 = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--force", "--json"])
    assert_eq(r2.returncode, 0, "--force closes it")
    payload2 = json.loads(r2.stdout)
    assert_eq(payload2["changes"][0]["action"], "closed", "--force overrides the no-facts refusal")


def test_second_apply_refuses_to_overwrite_the_backup():
    print("\n[test] a backup file is created, and a second --apply refuses to overwrite it")
    _wipe_storage()
    conv_id = "backup-collision"
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))

    def _seed():
        _write_record(conv_id, state="in_progress", started_at=old_ts, updated_at=old_ts,
                      exchanges_done=4, exchanges_total=40, attempts=1, error=None)
        _write_facts(conv_id)

    _seed()
    fixed_stamp = "20260101T000000Z"
    argv = _store_args() + ["--conv", conv_id, "--apply", "--json"]

    with patch.object(_script, "_utc_stamp", lambda: fixed_stamp):
        rc1 = _script.main(argv)
    assert_eq(rc1, 0, "first --apply (in-process) closes the record")
    backup_path = _record_path(conv_id).with_name(
        f"{conv_id}.backfill.json.bak-{fixed_stamp}"
    )
    assert_true(backup_path.is_file(), "backup file created at the expected fixed-stamp path")
    backup_bytes_after_first = backup_path.read_bytes()

    # Put the record back to would-resume, WITHOUT touching the backup —
    # simulates a second stale record needing the exact same backup name
    # (the collision this refusal exists for).
    _seed()

    with patch.object(_script, "_utc_stamp", lambda: fixed_stamp):
        rc2 = _script.main(argv)
    assert_eq(rc2, 1, "second --apply at the same stamp is an error (refused)")
    assert_eq(backup_path.read_bytes(), backup_bytes_after_first,
              "the existing backup was NOT overwritten")
    rec = json.loads(_record_path(conv_id).read_text())
    assert_eq(rec["state"], "in_progress",
              "the record itself was NOT rewritten when the backup collided")


# ---------------------------------------------------------------------------
# Malformed records: reported, never crashed on, never rewritten
# ---------------------------------------------------------------------------

def test_malformed_records_are_reported_not_crashed_on_and_never_rewritten():
    print("\n[test] an unreadable or malformed record is reported, not crashed on, and never rewritten")
    _wipe_storage()
    _record_path("bad-json").write_text("{not valid json", encoding="utf-8")
    _record_path("bad-shape").write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    before = _all_backfill_bytes()

    r = _run_script(_store_args() + ["--json"])
    assert_eq(r.returncode, 1,
              "M8: a needs-review-only store exits 1 (human attention "
              "required) even though nothing is would-resume — it used to "
              "exit 0 as if nothing were wrong")
    payload = json.loads(r.stdout)
    by_id = {row["conv_id"]: row for row in payload["records"]}
    assert_eq(by_id["bad-json"]["verdict"], "needs-review", "invalid JSON -> needs-review")
    assert_eq(by_id["bad-shape"]["verdict"], "needs-review", "wrong JSON shape -> needs-review")
    assert_true(by_id["bad-json"]["reason"], "a reason string is given for bad-json")

    r2 = _run_script(_store_args() + ["--apply", "--json"])
    assert_eq(r2.returncode, 1,
              "N8: --apply over a needs-review-only store (here, malformed "
              "records that could not even be read) must exit 1, not 0 — "
              "matching the dry run above. Nothing was would-resume, so "
              "there was nothing to close, but 'nothing to close' is not "
              "the same as 'nothing needs a human'")
    payload2 = json.loads(r2.stdout)
    changed_ids = {c["conv_id"] for c in payload2["changes"]}
    assert_true("bad-json" not in changed_ids, "bad-json was never a target of --apply")
    assert_true("bad-shape" not in changed_ids, "bad-shape was never a target of --apply")
    assert_eq(_all_backfill_bytes(), before, "malformed records are byte-identical after --apply")


# ---------------------------------------------------------------------------
# --json output
# ---------------------------------------------------------------------------

def test_json_output_parses_and_carries_the_same_verdicts_as_text():
    print("\n[test] --json output parses and carries the same verdicts as the text report")
    _wipe_storage()
    _seed_real_stale_records()
    _seed_terminal_records()

    r_json = _run_script(_store_args() + ["--json"])
    payload = json.loads(r_json.stdout)  # raises if not valid JSON
    would_resume_json = {
        row["conv_id"] for row in payload["records"] if row["verdict"] == "would-resume"
    }
    assert_eq(would_resume_json, {c for c, _, _ in REAL_STALE_RECORDS},
              "JSON marks exactly the four real records would-resume")

    r_text = _run_script(_store_args())
    for conv_id in would_resume_json:
        assert_true(f"{conv_id}" in r_text.stdout and "would-resume" in r_text.stdout,
                    f"{conv_id} also reads would-resume in the text report")
    assert_eq(payload["counts"]["would-resume"], 4, "JSON counts match the text report's tally")


# ---------------------------------------------------------------------------
# Defect 1: package resolution — HERE.parent/"compactor" (the repo/clone
# layout) resolving to /data/compactor when copied to /data/scripts/ was
# the real operator error. Covered two ways: the pure resolution function
# (fast, deterministic, no subprocess), and --compactor-pkg end to end
# through the real CLI (a proper real-image test drives the full
# fallback-to-/opt/compactor and multi-path-failure paths against the
# real image — see compactor/test_real_image_operator_scripts.py).
# ---------------------------------------------------------------------------

def test_resolve_compactor_pkg_explicit_flag_wins_over_the_repo_layout():
    print("\n[test] --compactor-pkg wins even when the repo-layout package also exists")
    with tempfile.TemporaryDirectory() as td:
        real_pkg = Path(td) / "explicit-pkg"
        real_pkg.mkdir()
        here = Path(td) / "repo" / "scripts"
        here.mkdir(parents=True)
        (here.parent / "compactor").mkdir()  # would win as candidate 2 if not overridden
        with patch.object(_script, "HERE", here):
            pkg, tried, already = _script._resolve_compactor_pkg(str(real_pkg))
        assert_eq(pkg, real_pkg, "the explicit --compactor-pkg path wins")
        assert_true(already is False, "not reported as sys.path-importable")
        assert_true("--compactor-pkg" in tried[0], "the explicit path is listed first, and labelled")


def test_resolve_compactor_pkg_repo_layout_beats_opt_compactor():
    print("\n[test] the repo/clone layout beside the script wins over /opt/compactor")
    with tempfile.TemporaryDirectory() as td:
        here = Path(td) / "repo" / "scripts"
        here.mkdir(parents=True)
        repo_pkg = here.parent / "compactor"
        repo_pkg.mkdir()
        opt_pkg = Path(td) / "opt-compactor"
        opt_pkg.mkdir()
        with patch.object(_script, "HERE", here), patch.object(_script, "OPT_COMPACTOR", opt_pkg):
            pkg, tried, already = _script._resolve_compactor_pkg(None)
        assert_eq(pkg, repo_pkg, "the repo/clone layout resolves first")


def test_resolve_compactor_pkg_falls_back_to_opt_compactor():
    print("\n[test] with no repo-layout package (the /data/scripts/ shape), resolution falls through to /opt/compactor")
    with tempfile.TemporaryDirectory() as td:
        # HERE simulates /data/scripts: its PARENT has no "compactor" child
        # — the exact real operator error (HERE.parent/"compactor" ==
        # /data/compactor, which does not exist).
        here = Path(td) / "data" / "scripts"
        here.mkdir(parents=True)
        opt_pkg = Path(td) / "opt-compactor"
        opt_pkg.mkdir()
        with patch.object(_script, "HERE", here), patch.object(_script, "OPT_COMPACTOR", opt_pkg):
            pkg, tried, already = _script._resolve_compactor_pkg(None)
        assert_eq(pkg, opt_pkg, "falls back to the image layout")
        assert_true(
            any(str(here.parent / "compactor") in t for t in tried),
            "the failed repo-layout candidate is still named in tried",
        )


def test_resolve_compactor_pkg_reports_every_path_tried_when_none_exist():
    print("\n[test] when nothing resolves, every path tried is reported — never a bare crash")
    with tempfile.TemporaryDirectory() as td:
        here = Path(td) / "data" / "scripts"
        here.mkdir(parents=True)
        missing_opt = Path(td) / "does-not-exist"
        with patch.object(_script, "HERE", here), patch.object(_script, "OPT_COMPACTOR", missing_opt):
            pkg, tried, already = _script._resolve_compactor_pkg(None)
        # `already` (source 4, an already-importable `backfill` on
        # sys.path) is NOT asserted here either way: this test module
        # itself does `import backfill` at the top, so in THIS process
        # `backfill` really is already importable — that is a true fact
        # about the test environment, not something this function gets
        # to decide. `pkg is None` is what "no PACKAGE DIRECTORY found"
        # actually means; main()'s own multi-path error additionally
        # requires `not already_importable` (see test_capability_check_*
        # and compactor/test_real_image_operator_scripts.py for that path
        # exercised in a real subprocess with a clean sys.path).
        assert_true(pkg is None, "no package directory found")
        assert_eq(len(tried), 3,
                   "repo-layout, /opt/compactor, and the sys.path probe were all tried "
                   f"(got {tried!r})")


def test_reports_which_pkg_source_was_used():
    print("\n[test] the report names which compactor package source was resolved")
    _wipe_storage()
    _seed_terminal_records()
    r = _run_script(_store_args() + ["--json"])
    payload = json.loads(r.stdout)
    assert_true(
        any("compactor package resolved from" in w for w in payload["warnings"]),
        f"a NOTE names the resolved package source (warnings={payload['warnings']!r})",
    )


# ---------------------------------------------------------------------------
# Defect 2: a pre-v3.1.9.4 package (missing _MAX_BACKFILL_ATTEMPTS and
# _backoff_ready) must refuse with an actionable message, never a raw
# AttributeError. Simulated here with a minimal stand-in package via
# --compactor-pkg; compactor/test_real_image_operator_scripts.py proves
# the same thing against a REAL git-archived v3.1.9 compactor/backfill.py.
# ---------------------------------------------------------------------------

def _write_stub_pre_v3194_backfill_pkg(root: Path) -> Path:
    """A --compactor-pkg stand-in with `is_stale`/`atomic_write_json` (this
    script's OTHER real dependencies) but not
    `_MAX_BACKFILL_ATTEMPTS`/`_backoff_ready` — the exact shape verified
    across v3.1.9/.1/.2/.3's real backfill.py."""
    pkg = root / "compactor"
    pkg.mkdir()
    (pkg / "backfill.py").write_text(
        "_STALE_SECONDS = 600\n"
        "def is_stale(record):\n"
        "    return True\n"
        "def atomic_write_json(path, data):\n"
        "    pass\n",
        encoding="utf-8",
    )
    return pkg


def test_capability_check_refuses_cleanly_on_a_pre_v3194_package():
    print("\n[test] a --compactor-pkg missing _MAX_BACKFILL_ATTEMPTS/_backoff_ready refuses, never crashes")
    _wipe_storage()
    _seed_real_stale_records()
    with tempfile.TemporaryDirectory() as td:
        stub_pkg = _write_stub_pre_v3194_backfill_pkg(Path(td))
        r = _run_script(_store_args() + ["--compactor-pkg", str(stub_pkg)])
    assert_eq(r.returncode, 1, "refuses with exit 1, not a crash")
    assert_true("_MAX_BACKFILL_ATTEMPTS" in r.stdout, "names the missing symbol")
    assert_true("_backoff_ready" in r.stdout, "names the other missing symbol")
    assert_true("predates v3.1.9.4" in r.stdout, "names the pod's version state")
    assert_true("Traceback" not in r.stdout and "Traceback" not in r.stderr,
                "no raw traceback anywhere")
    assert_true("git clone" in r.stdout, "prints the clone command")


def test_capability_check_json_mode_also_refuses_cleanly():
    print("\n[test] the same capability refusal in --json mode is still clean JSON, not a crash")
    _wipe_storage()
    with tempfile.TemporaryDirectory() as td:
        stub_pkg = _write_stub_pre_v3194_backfill_pkg(Path(td))
        r = _run_script(_store_args() + ["--compactor-pkg", str(stub_pkg), "--json"])
    assert_eq(r.returncode, 1, "refuses with exit 1")
    payload = json.loads(r.stdout)  # raises if not valid JSON
    assert_true("_MAX_BACKFILL_ATTEMPTS" in payload["error"], "the JSON error names the missing symbol")


# ---------------------------------------------------------------------------
# H4: --compactor-pkg validation and provenance (a chosen directory is not
# necessarily the module actually imported).
# ---------------------------------------------------------------------------

def test_compactor_pkg_not_a_directory_is_refused_not_silently_skipped():
    print("\n[test] H4: an explicit --compactor-pkg that is not a directory refuses, never silently falls back to auto-detection")
    _wipe_storage()
    with tempfile.TemporaryDirectory() as td:
        not_a_dir = Path(td) / "does-not-exist" / "compactor"
        r = _run_script(_store_args() + ["--compactor-pkg", str(not_a_dir)])
    assert_eq(r.returncode, 1, "refuses with exit 1")
    assert_true(str(not_a_dir) in r.stdout, "names the bad path")
    assert_true("is not a directory" in r.stdout, "says why")
    assert_true("Traceback" not in r.stdout, "no crash")


def test_compactor_pkg_shadowed_by_pythonpath_is_refused_not_silently_wrong():
    print("\n[test] H4: an empty --compactor-pkg dir, shadowed by a real backfill.py on PYTHONPATH, refuses instead of silently trusting the shadow")
    _wipe_storage()
    _seed_real_stale_records()
    with tempfile.TemporaryDirectory() as td:
        empty_pkg = Path(td) / "empty-pkg"
        empty_pkg.mkdir()
        shadow_pkg = Path(td) / "shadow"
        shadow_pkg.mkdir()
        # A fully-capable stand-in on purpose: if the shadow module were
        # ever trusted, the CAPABILITY check would not catch it either —
        # only the provenance check (comparing backfill.__file__ against
        # the resolved directory) can.
        (shadow_pkg / "backfill.py").write_text(
            "_MAX_BACKFILL_ATTEMPTS = 3\n"
            "def _backoff_ready(record):\n    return True\n"
            "def is_stale(record):\n    return True\n"
            "def atomic_write_json(path, data):\n    pass\n",
            encoding="utf-8",
        )
        env = {**os.environ, "PYTHONPATH": str(shadow_pkg)}
        r = subprocess.run(
            [sys.executable, str(_SCRIPT)] + _store_args()
            + ["--compactor-pkg", str(empty_pkg)],
            capture_output=True, text=True, timeout=60, env=env,
        )
    assert_eq(r.returncode, 1,
              "refuses with exit 1, never silently classifies using the shadowed module")
    assert_true("shadow" in r.stdout.lower(), "names the shadow directory")
    assert_true(str(shadow_pkg) in r.stdout, "names the real (shadowing) __file__ location")
    assert_true("Traceback" not in r.stdout, "no crash")


# ---------------------------------------------------------------------------
# M1: the capability check covers every symbol used, runs before any
# backup, and _close_record itself cleans up if atomic_write_json still
# somehow raises (defense in depth beyond the capability check).
# ---------------------------------------------------------------------------

def test_atomic_write_json_failure_leaves_no_orphan_backup():
    print("\n[test] M1: if atomic_write_json still raises after the capability check passes, no orphan .bak is left and the run refuses cleanly")
    _wipe_storage()
    conv_id = "boom-on-write"
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    _write_record(conv_id, state="in_progress", started_at=old_ts, updated_at=old_ts,
                  exchanges_done=5, exchanges_total=20, attempts=1, error=None)
    _write_facts(conv_id)
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "compactor"
        pkg.mkdir()
        # is_stale/_backoff_ready implement the REAL contract here (age-based,
        # not hardcoded) -- this fixture is only exercising atomic_write_json
        # failing, and a hardcoded-True stand-in would now be refused by the
        # M1 behavioural self-check before ever reaching that write (see
        # test_fabricated_package_with_the_right_names_but_wrong_behaviour_is_refused
        # for that case on its own).
        (pkg / "backfill.py").write_text(
            "from datetime import datetime, timezone\n"
            "_MAX_BACKFILL_ATTEMPTS = 3\n"
            "_STALE_SECONDS = 600\n"
            "def is_stale(record):\n"
            "    if record.get('state') != 'in_progress':\n"
            "        return False\n"
            "    updated_at = record.get('updated_at')\n"
            "    if not updated_at:\n"
            "        return True\n"
            "    ts = datetime.fromisoformat(updated_at)\n"
            "    age = (datetime.now(timezone.utc) - ts).total_seconds()\n"
            "    return age > _STALE_SECONDS\n"
            "def _backoff_ready(record):\n"
            "    updated_at = record.get('updated_at')\n"
            "    if not updated_at:\n"
            "        return True\n"
            "    ts = datetime.fromisoformat(updated_at)\n"
            "    age = (datetime.now(timezone.utc) - ts).total_seconds()\n"
            "    attempts = max(1, int(record.get('attempts') or 1))\n"
            "    return age > (600 * (2 ** (attempts - 1)))\n"
            "def atomic_write_json(path, data):\n"
            "    raise RuntimeError('simulated write failure')\n",
            encoding="utf-8",
        )
        r = _run_script(_store_args() + ["--conv", conv_id, "--compactor-pkg", str(pkg),
                                          "--apply", "--json"])
    assert_eq(r.returncode, 1, "H1: closed none of its targets -> exit 1")
    payload = json.loads(r.stdout)
    assert_eq(payload["changes"][0]["action"], "refused-write-failed",
              "reported as a clean refusal, never a crash")
    assert_true("Traceback" not in r.stdout and "Traceback" not in r.stderr, "no raw traceback")
    backups = list(_record_path(conv_id).parent.glob(f"{conv_id}.backfill.json.bak-*"))
    assert_eq(backups, [], "no orphan backup file left behind")
    rec = json.loads(_record_path(conv_id).read_text())
    assert_eq(rec["state"], "in_progress", "the original record was never rewritten")


# ---------------------------------------------------------------------------
# M3 / BR4: the live-compactor refusal must not fail open. Only an
# unambiguous connection refusal at --health-url is safe to read as
# "not running"; a real response OR a timeout both refuse --apply.
# ---------------------------------------------------------------------------

class _Health200Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a, **kw):
        pass  # keep test output quiet


def _start_health_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Health200Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def test_apply_refuses_when_health_url_answers_and_force_overrides():
    print("\n[test] M3/BR4: --apply refuses when something answers --health-url, and --force overrides it")
    _wipe_storage()
    conv_id = "health-check-alive"
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    _write_record(conv_id, state="in_progress", started_at=old_ts, updated_at=old_ts,
                  exchanges_done=1, exchanges_total=10, attempts=1, error=None)
    _write_facts(conv_id)
    before = _record_path(conv_id).read_bytes()

    server, port = _start_health_server()
    try:
        health_url = f"http://127.0.0.1:{port}/health"
        r = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--json",
                                          "--health-url", health_url])
        assert_eq(r.returncode, 1, "refused: something answered --health-url")
        payload = json.loads(r.stdout)
        assert_true("error" in payload, "a clean JSON refusal, not a crash")
        assert_eq(_record_path(conv_id).read_bytes(), before,
                  "record untouched by the refused apply")

        r2 = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--force", "--json",
                                           "--health-url", health_url])
        assert_eq(r2.returncode, 0, "--force overrides the refusal and closes it")
        payload2 = json.loads(r2.stdout)
        assert_true(any("overrid" in w.lower() for w in payload2["warnings"]),
                    "a loud warning names the override")
    finally:
        server.shutdown()


def test_apply_proceeds_when_health_url_connection_refused():
    print("\n[test] M3: --apply proceeds when --health-url is an unambiguous connection refusal (nothing listening)")
    _wipe_storage()
    conv_id = "health-check-not-running"
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    _write_record(conv_id, state="in_progress", started_at=old_ts, updated_at=old_ts,
                  exchanges_done=1, exchanges_total=10, attempts=1, error=None)
    _write_facts(conv_id)

    # A bound-then-closed socket's port is guaranteed refused, unlike an
    # arbitrary high port that merely happens to be free right now.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    r = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--json",
                                      "--health-url", f"http://127.0.0.1:{port}/health"])
    assert_eq(r.returncode, 0, "connection refused reads as not-running; apply proceeds")
    payload = json.loads(r.stdout)
    assert_eq(payload["changes"][0]["action"], "closed", "the record was actually closed")


def test_apply_refuses_on_health_url_timeout():
    print("\n[test] M3: a --health-url probe that times out is ambiguous, not 'not running' -- --apply refuses")
    _wipe_storage()
    conv_id = "health-check-timeout"
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    _write_record(conv_id, state="in_progress", started_at=old_ts, updated_at=old_ts,
                  exchanges_done=1, exchanges_total=10, attempts=1, error=None)
    _write_facts(conv_id)
    before = _record_path(conv_id).read_bytes()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def _accept_and_hang():
        listener.settimeout(5)
        try:
            conn, _ = listener.accept()
            stop.wait(5)
            conn.close()
        except OSError:
            pass

    t = threading.Thread(target=_accept_and_hang, daemon=True)
    t.start()
    try:
        r = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--json",
                                          "--health-url", f"http://127.0.0.1:{port}/health"])
        assert_eq(r.returncode, 1, "a timeout is ambiguous, not 'not running' -- refused")
        assert_eq(_record_path(conv_id).read_bytes(), before, "record untouched")
    finally:
        stop.set()
        listener.close()


# ---------------------------------------------------------------------------
# M4 mutants BR3 and BR8: a not-yet-stale in_progress record must stay
# needs-review (never closed), and a terminal record must stay `leave`
# (never misclassified needs-review, which would otherwise still look
# untouched-by-apply for the wrong reason and mask the mutant).
# ---------------------------------------------------------------------------

def test_not_yet_stale_in_progress_record_is_needs_review_and_untouched_by_apply():
    print("\n[test] BR3: a NOT-yet-stale in_progress record is needs-review, and --apply (even --force) must never close it")
    _wipe_storage()
    conv_id = "freshly-running"
    now_ts = _iso(datetime.now(timezone.utc))
    _write_record(conv_id, state="in_progress", started_at=now_ts, updated_at=now_ts,
                  exchanges_done=1, exchanges_total=50, attempts=1, error=None)
    _write_facts(conv_id)
    before = _record_path(conv_id).read_bytes()

    r = _run_script(_store_args() + ["--conv", conv_id, "--json"])
    payload = json.loads(r.stdout)
    assert_eq(payload["records"][0]["verdict"], "needs-review",
              "a fresh in_progress record is needs-review, not would-resume")

    r2 = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--force", "--json"])
    assert_eq(r2.returncode, 1,
              "N8: nothing was would-resume, so there was nothing to close, "
              "but the needs-review record (a genuinely running backfill) "
              "is still there and --apply must exit 1, not 0, for exactly "
              "the same reason the dry run above did")
    assert_eq(json.loads(r2.stdout)["changes"], [], "nothing was closed")
    assert_eq(_record_path(conv_id).read_bytes(), before,
              "the record was never touched, even with --force")


def test_terminal_records_are_classified_leave_not_needs_review():
    print("\n[test] BR8: complete/abandoned/wiped records classify leave, never needs-review")
    _wipe_storage()
    _seed_terminal_records()
    r = _run_script(_store_args() + ["--json"])
    payload = json.loads(r.stdout)
    by_id = {row["conv_id"]: row for row in payload["records"]}
    for conv_id in ("term-complete", "term-abandoned", "term-wiped"):
        assert_eq(by_id[conv_id]["verdict"], "leave", f"{conv_id} is classified leave")
    assert_eq(payload["counts"]["needs-review"], 0,
              "no terminal record is misclassified needs-review")
    assert_eq(payload["counts"]["leave"], 3, "all three terminal records count as leave")


# ---------------------------------------------------------------------------
# M8: needs-review and an all-rejected --conv must both affect the exit
# code, never silently read as "0 record(s), all clear".
# ---------------------------------------------------------------------------

def test_all_conv_values_rejected_exits_1_not_0():
    print("\n[test] M8: every --conv value rejected (no record found) exits 1, not 0")
    _wipe_storage()
    _seed_terminal_records()  # a non-empty store; just none of these ids are in it
    r = _run_script(_store_args() + ["--conv", "does-not-exist-1", "--conv", "..", "--json"])
    assert_eq(r.returncode, 1,
              "nothing was inspected -- must read as an error, not '0 record(s), all clear'")
    payload = json.loads(r.stdout)
    assert_eq(payload["records"], [], "no records were actually inspected")
    assert_true(any("none of the given --conv" in w for w in payload["warnings"]),
                "a clear warning explains why")


# ---------------------------------------------------------------------------
# N8: --apply must exit 1 whenever a needs-review record remains, exactly
# matching the dry run's own rule and the documented exit-code table --
# even when --apply closed every would-resume record it targeted. Before
# this fix, --apply's exit code was derived only from the would-resume
# tally, so a needs-review record (a running backfill, an unreadable
# record, a capped record, a failed record still in backoff) never
# affected --apply's exit code at all.
# ---------------------------------------------------------------------------

def test_apply_exits_1_with_only_a_running_backfill_needs_review():
    print("\n[test] N8 (E1): --apply over a store with only a genuinely RUNNING backfill (needs-review) exits 1, not 0")
    _wipe_storage()
    conv_id = "genuinely-running"
    now_ts = _iso(datetime.now(timezone.utc))
    _write_record(conv_id, state="in_progress", started_at=now_ts, updated_at=now_ts,
                  exchanges_done=1, exchanges_total=50, attempts=1, error=None)
    _write_facts(conv_id)

    r_dry = _run_script(_store_args() + ["--conv", conv_id, "--json"])
    assert_eq(r_dry.returncode, 1, "dry run already exits 1 for a needs-review-only store")

    r = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--json"])
    assert_eq(r.returncode, 1,
              "N8: --apply over a needs-review-only store must exit 1 too, "
              "matching the dry run -- a running backfill is not an --apply "
              "TARGET at all, so it used to be invisible to --apply's exit code")
    payload = json.loads(r.stdout)
    assert_eq(payload["changes"], [], "nothing was closed (nothing was would-resume)")
    rec = json.loads(_record_path(conv_id).read_text())
    assert_eq(rec["state"], "in_progress", "the running backfill's record is untouched")


def test_apply_exits_1_when_would_resume_closed_but_a_needs_review_record_remains():
    print("\n[test] N8 (E2): --apply exits 1 when a needs-review record remains, even though every would-resume record was fully closed")
    _wipe_storage()
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    now_ts = _iso(datetime.now(timezone.utc))
    _write_record("stale-one", state="in_progress", started_at=old_ts, updated_at=old_ts,
                  exchanges_done=4, exchanges_total=40, attempts=1, error=None)
    _write_facts("stale-one")
    _write_record("running-one", state="in_progress", started_at=now_ts, updated_at=now_ts,
                  exchanges_done=1, exchanges_total=50, attempts=1, error=None)
    _write_facts("running-one")

    r = _run_script(_store_args() + ["--apply", "--json"])
    assert_eq(r.returncode, 1,
              "N8: a needs-review record left over makes --apply exit 1 "
              "even though TARGETED == CLOSED for the would-resume record")
    payload = json.loads(r.stdout)
    closed_ids = {c["conv_id"] for c in payload["changes"] if c["action"] == "closed"}
    assert_eq(closed_ids, {"stale-one"}, "the would-resume record was still closed")
    rec = json.loads(_record_path("running-one").read_text())
    assert_eq(rec["state"], "in_progress", "the running backfill's record is untouched")


def test_apply_exits_1_with_a_capped_stale_record_needs_review():
    print("\n[test] N8 (E9): a stale in_progress record already at the attempt cap is needs-review, and --apply exits 1")
    _wipe_storage()
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    _write_record("capped", state="in_progress", started_at=old_ts, updated_at=old_ts,
                  exchanges_done=1, exchanges_total=50, attempts=3, error=None)
    _write_facts("capped")

    r = _run_script(_store_args() + ["--conv", "capped", "--apply", "--json"])
    assert_eq(r.returncode, 1, "N8: a capped-stale needs-review record makes --apply exit 1")
    payload = json.loads(r.stdout)
    assert_eq(payload["changes"], [], "nothing was would-resume, so nothing was targeted")


def test_apply_exits_1_with_a_failed_record_still_in_backoff():
    print("\n[test] N8 (E8): a failed record still inside its retry backoff is needs-review, and --apply exits 1")
    _wipe_storage()
    now_ts = _iso(datetime.now(timezone.utc))
    _write_record("in-backoff", state="failed", started_at=now_ts, updated_at=now_ts,
                  exchanges_done=1, exchanges_total=50, attempts=1, error="boom")
    _write_facts("in-backoff")

    r = _run_script(_store_args() + ["--conv", "in-backoff", "--apply", "--json"])
    assert_eq(r.returncode, 1, "N8: a failed-but-in-backoff needs-review record makes --apply exit 1")


# ---------------------------------------------------------------------------
# M1 remainder: hasattr proves a NAME exists, not that it BEHAVES like the
# real compactor/backfill.py. A fabricated package with the four required
# symbol names, but an is_stale/_backoff_ready that simply always agree,
# used to sail through the capability check and could close a genuinely
# RUNNING backfill. A behavioural self-check runs synthetic in-memory
# records through both functions, before any write, and refuses on any
# disagreement with the documented contract.
# ---------------------------------------------------------------------------

def _write_fabricated_pkg_that_lies(root: Path) -> Path:
    """The reviewer's round-2 M1 repro: all four required symbol names are
    present (hasattr passes), but is_stale and _backoff_ready always say
    "yes, go ahead" regardless of the record -- exactly a fabricated
    stand-in that would tell this script a genuinely RUNNING backfill is
    stale and ready to close."""
    pkg = root / "compactor"
    pkg.mkdir()
    (pkg / "backfill.py").write_text(
        "_MAX_BACKFILL_ATTEMPTS = 99\n"
        "def _backoff_ready(record):\n    return True\n"
        "def is_stale(record):\n    return True\n"
        "def atomic_write_json(path, data):\n"
        "    import json\n"
        "    open(path, 'w').write(json.dumps(data))\n",
        encoding="utf-8",
    )
    return pkg


def test_fabricated_package_with_the_right_names_but_wrong_behaviour_is_refused():
    print("\n[test] M1: a fabricated package with all 4 required symbol names, but is_stale/_backoff_ready that always agree, is refused before any write")
    _wipe_storage()
    conv_id = "genuinely-running-2"
    now_ts = _iso(datetime.now(timezone.utc))
    _write_record(conv_id, state="in_progress", started_at=now_ts, updated_at=now_ts,
                  exchanges_done=1, exchanges_total=50, attempts=1, error=None)
    _write_facts(conv_id)
    before = _record_path(conv_id).read_bytes()

    with tempfile.TemporaryDirectory() as td:
        fake_pkg = _write_fabricated_pkg_that_lies(Path(td))
        r = _run_script(_store_args() + ["--conv", conv_id, "--compactor-pkg", str(fake_pkg),
                                          "--apply", "--json"])
    assert_eq(r.returncode, 1,
              "M1: hasattr-only capability checking is not enough -- a "
              "package with the right names but the wrong behaviour must "
              "still be refused, before it ever gets to close anything")
    assert_true("Traceback" not in r.stdout and "Traceback" not in r.stderr, "no raw traceback")
    assert_eq(_record_path(conv_id).read_bytes(), before,
              "the running backfill's record was never touched -- the "
              "reviewer's exact repro (a fabricated package closing a "
              "RUNNING backfill) is now refused before any write")
    backups = list(_record_path(conv_id).parent.glob(f"{conv_id}.backfill.json.bak-*"))
    assert_eq(backups, [], "no backup was ever created -- refused before any write happened")


def test_behavioural_self_check_passes_against_the_real_backfill_module():
    print("\n[test] M1: the behavioural self-check agrees with the real compactor/backfill.py")
    assert_eq(_script._check_backfill_behavior(backfill), None,
              "the real module's is_stale/_backoff_ready pass the synthetic checks")


def test_behavioural_self_check_catches_a_lying_is_stale():
    print("\n[test] M1: the behavioural self-check catches an is_stale that always says yes")
    with patch.object(backfill, "is_stale", lambda record: True):
        err = _script._check_backfill_behavior(backfill)
    assert_true(err is not None, "a lying is_stale is caught")
    assert_true("is_stale" in err, "names the offending function")


def test_behavioural_self_check_catches_a_lying_backoff_ready():
    print("\n[test] M1: the behavioural self-check catches a _backoff_ready that always says yes")
    with patch.object(backfill, "_backoff_ready", lambda record: True):
        err = _script._check_backfill_behavior(backfill)
    assert_true(err is not None, "a lying _backoff_ready is caught")
    assert_true("_backoff_ready" in err, "names the offending function")


def test_capability_check_requires_is_stale_specifically():
    print("\n[test] M1 (BM1): a --compactor-pkg missing ONLY is_stale is refused cleanly, not silently accepted")
    _wipe_storage()
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "compactor"
        pkg.mkdir()
        (pkg / "backfill.py").write_text(
            "_MAX_BACKFILL_ATTEMPTS = 3\n"
            "def _backoff_ready(record):\n    return True\n"
            "def atomic_write_json(path, data):\n    pass\n",
            encoding="utf-8",
        )
        r = _run_script(_store_args() + ["--compactor-pkg", str(pkg)])
    assert_eq(r.returncode, 1, "refuses with exit 1, not a crash")
    assert_true("is_stale" in r.stdout, "names the specifically missing is_stale symbol")
    assert_true("Traceback" not in r.stdout, "no raw traceback")


def test_provenance_note_reports_the_backfill_module_sha256():
    print("\n[test] M1: the provenance NOTE reports the compactor package's backfill.py file sha256")
    _wipe_storage()
    _seed_terminal_records()
    r = _run_script(_store_args() + ["--json"])
    payload = json.loads(r.stdout)
    note = next(w for w in payload["warnings"] if "compactor package resolved from" in w)
    assert_true("sha256=" in note, f"the NOTE reports a sha256 (note={note!r})")
    real_sha = hashlib.sha256(Path(backfill.__file__).read_bytes()).hexdigest()
    assert_true(real_sha in note, "the sha256 matches the real backfill.py file's content")


# ---------------------------------------------------------------------------
# M3 mutants BM4, BM5, BM13: the health-probe fail-safe (_compactor_is_alive)
# must be pinned on every branch, not just the connection-refused and
# timeout cases already covered above.
# ---------------------------------------------------------------------------

class _Health500Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(500)
        self.end_headers()

    def log_message(self, *a, **kw):
        pass  # keep test output quiet


def _start_health_server_with_status(handler_cls):
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def test_apply_refuses_on_a_non_200_health_response():
    print("\n[test] BM4: a real non-200 (500) --health-url response is ambiguous, not 'not running' -- --apply refuses")
    _wipe_storage()
    conv_id = "health-check-500"
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    _write_record(conv_id, state="in_progress", started_at=old_ts, updated_at=old_ts,
                  exchanges_done=1, exchanges_total=10, attempts=1, error=None)
    _write_facts(conv_id)
    before = _record_path(conv_id).read_bytes()

    server, port = _start_health_server_with_status(_Health500Handler)
    try:
        r = _run_script(_store_args() + ["--conv", conv_id, "--apply", "--json",
                                          "--health-url", f"http://127.0.0.1:{port}/health"])
        assert_eq(r.returncode, 1, "a real 500 response is a genuine HTTPError, still ambiguous -- refused")
        assert_eq(_record_path(conv_id).read_bytes(), before, "record untouched")
    finally:
        server.shutdown()


def test_health_probe_unknown_exception_is_treated_as_ambiguous_not_dead():
    print("\n[test] BM5: an exception outside HTTPError/URLError/OSError probing --health-url is ambiguous, not 'not running'")
    # A URL with no recognised scheme raises a bare ValueError straight out
    # of urllib.request.urlopen -- not wrapped in URLError -- which is
    # exactly the "any other Exception" branch BM5 targets.
    assert_eq(_script._compactor_is_alive("not-a-url-at-all"), True,
              "an unrecognised exception must read as 'might be alive', never 'not running'")


def test_health_probe_urlerror_without_connection_refused_is_ambiguous_not_dead():
    print("\n[test] BM13: a URLError whose reason is NOT a connection refusal (timeout, DNS failure) is ambiguous, not 'not running'")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
        assert_eq(_script._compactor_is_alive("http://127.0.0.1:1/health"), True,
                  "a URLError whose reason is a plain string (timeout/DNS), "
                  "not ECONNREFUSED, must read as 'might be alive'")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_dry_run_changes_nothing_on_disk_and_exits_3()
        test_dry_run_no_stale_records_exits_0()

        test_apply_rewrites_only_the_four_would_resume_records()

        test_closed_records_make_needs_backfill_return_false()

        test_would_resume_with_no_facts_file_refused_without_force()
        test_second_apply_refuses_to_overwrite_the_backup()

        test_malformed_records_are_reported_not_crashed_on_and_never_rewritten()

        test_json_output_parses_and_carries_the_same_verdicts_as_text()

        test_resolve_compactor_pkg_explicit_flag_wins_over_the_repo_layout()
        test_resolve_compactor_pkg_repo_layout_beats_opt_compactor()
        test_resolve_compactor_pkg_falls_back_to_opt_compactor()
        test_resolve_compactor_pkg_reports_every_path_tried_when_none_exist()
        test_reports_which_pkg_source_was_used()

        test_capability_check_refuses_cleanly_on_a_pre_v3194_package()
        test_capability_check_json_mode_also_refuses_cleanly()

        test_compactor_pkg_not_a_directory_is_refused_not_silently_skipped()
        test_compactor_pkg_shadowed_by_pythonpath_is_refused_not_silently_wrong()

        test_atomic_write_json_failure_leaves_no_orphan_backup()

        test_apply_refuses_when_health_url_answers_and_force_overrides()
        test_apply_proceeds_when_health_url_connection_refused()
        test_apply_refuses_on_health_url_timeout()

        test_not_yet_stale_in_progress_record_is_needs_review_and_untouched_by_apply()
        test_terminal_records_are_classified_leave_not_needs_review()

        test_all_conv_values_rejected_exits_1_not_0()

        test_apply_exits_1_with_only_a_running_backfill_needs_review()
        test_apply_exits_1_when_would_resume_closed_but_a_needs_review_record_remains()
        test_apply_exits_1_with_a_capped_stale_record_needs_review()
        test_apply_exits_1_with_a_failed_record_still_in_backoff()

        test_fabricated_package_with_the_right_names_but_wrong_behaviour_is_refused()
        test_behavioural_self_check_passes_against_the_real_backfill_module()
        test_behavioural_self_check_catches_a_lying_is_stale()
        test_behavioural_self_check_catches_a_lying_backoff_ready()
        test_capability_check_requires_is_stale_specifically()
        test_provenance_note_reports_the_backfill_module_sha256()

        test_apply_refuses_on_a_non_200_health_response()
        test_health_probe_unknown_exception_is_treated_as_ambiguous_not_dead()
        test_health_probe_urlerror_without_connection_refused_is_ambiguous_not_dead()

        print("\nAll backfill-records.py script tests passed.")
    finally:
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
