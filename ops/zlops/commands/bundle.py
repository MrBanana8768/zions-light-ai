"""`zl bundle [--out DIR] [--keep N] [--print-path] [--raw-logs [--yes]]`
-- build a `zla-bundle-<UTC ts>.tar.gz` matching the layout of the real
bundles the owner already analyses (COMMANDS.md "Log and backup analysis"):

    meta/{health.json, supervisorctl.txt, gpu.txt, disk.txt,
          env-redacted.txt, conversations.json, telemetry.json, versions.txt}
    logs/zla/*.log

NEVER included: webui.db, the compactor store, or /data/ssh -- this
command only ever reads podenv.LOG_DIR and the small admin/meta sources
listed above, never the store or the database. The env dump is redacted
(common.redact_env).

Log text is NOT safe by construction (review B4 found this docstring's
previous claim -- "log text, which carries no reply text by design" --
false: a failed write logs SQLAlchemy's `[parameters: (...)]`, which is
her chat JSON, and the compactor logs fact text and reply fragments at
INFO/WARNING in specific, known places). Every copied log is passed
through `zlops.privacy.scrub_log_text` (source = the log's own basename)
by default. `--raw-logs` copies logs UNTOUCHED instead -- only after a
typed confirmation (or `--yes`) and a loud warning that the result
contains her message text; use it only when handing a bundle to someone
who genuinely needs the raw evidence (e.g. diagnosing the scrubber
itself), never as the default hand-off.

`--req ID` (round 3, v3.1.9.8 request tracing): restricts every copied log
to just that one request's own lines (`req=ID` and its child ids
`req=ID.sum1`/`.facts`/`.dedup`/...), applied AFTER scrubbing -- a
narrowing filter, never a substitute for the privacy scrub. `meta/
req-filter.txt` records that the filter was applied and to what id, so a
bundle's own contents never silently look like "the whole log" when they
are not.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

from .. import common, podenv, privacy
from . import disk as cmd_disk, version as cmd_version

_TS_FMT = "%Y%m%dT%H%M%SZ"
_NAME_PREFIX = "zla-bundle-"
_DEFAULT_KEEP = 10
_LOG_COPY_TIMEOUT_S = 20.0  # per file -- MooseFS can stall; never let one log hang the whole bundle
_LOG_MAX_BYTES = 20 * 1024 * 1024  # tail cap per file, so one huge log cannot blow up bundle size/time


def _capture_text(argv: list[str], timeout: float = 15.0) -> str:
    try:
        r = common.run(argv, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"(could not run {argv[0]}: {e})\n"
    return (r.stdout or "") + (r.stderr or "")


def _write_meta(meta_dir: Path) -> None:
    # health.json -- the raw /health/full document (structured fields, not chat text)
    _status_code, doc, error = common.http_get_json(f"{podenv.COMPACTOR_URL}/health/full", timeout=5.0)
    (meta_dir / "health.json").write_text(
        json.dumps(doc if doc is not None else {"error": error}, indent=2), encoding="utf-8"
    )

    (meta_dir / "supervisorctl.txt").write_text(_capture_text(["supervisorctl", "status"]), encoding="utf-8")

    if shutil.which("nvidia-smi"):
        (meta_dir / "gpu.txt").write_text(_capture_text(["nvidia-smi"], timeout=20.0), encoding="utf-8")
    else:
        (meta_dir / "gpu.txt").write_text(
            "nvidia-smi not present (normal off a real pod; the NVIDIA runtime injects it)\n",
            encoding="utf-8",
        )

    disk_lines: list[str] = []
    disk_lines.append(f"du -sh of each entry under {podenv.DATA}:")
    if podenv.DATA.is_dir():
        for entry in sorted(podenv.DATA.iterdir()):
            disk_lines.append(f"  {cmd_disk.du(entry):>10}  {entry}")
    disk_lines.append("")
    disk_lines.append("df / (container overlay, NOT /data):")
    disk_lines.append(_capture_text(["df", "-h", "/"], timeout=10.0))
    disk_lines.append(f"NOTE: df {podenv.DATA} reports the whole shared MooseFS cluster, not this volume's quota.")
    count, total = cmd_disk.dir_count_and_size(podenv.BACKUPS)
    disk_lines.append(f"backups: {count} archive(s) in {podenv.BACKUPS}, {common.human_bytes(total)} total")
    # compactor's own control/pending roots -- reported only if present,
    # same `du` (time-bounded) as everything else in this file.
    for label, root in (("control root", podenv.CONTROL_ROOT), ("pending root", podenv.PENDING_ROOT)):
        if root.is_dir():
            disk_lines.append(f"{label} ({root}): {cmd_disk.du(root)}")
    (meta_dir / "disk.txt").write_text("\n".join(disk_lines) + "\n", encoding="utf-8")

    redacted = common.redact_env(common.merged_environ())
    env_lines = [f"{k}={v}" for k, v in sorted(redacted.items())]
    (meta_dir / "env-redacted.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    _status_code, conv_doc, conv_error = common.http_get_json(f"{podenv.COMPACTOR_URL}/admin/conversations", timeout=10.0)
    (meta_dir / "conversations.json").write_text(
        json.dumps(conv_doc if conv_doc is not None else {"error": conv_error}, indent=2), encoding="utf-8"
    )

    try:
        from .. import telemetry  # lazy: B's module

        snapshot = telemetry.collect_snapshot()
    except Exception as e:  # a bundle must still complete even if this collector isn't ready/errors
        snapshot = {"error": f"zlops.telemetry.collect_snapshot() failed: {e}"}
    (meta_dir / "telemetry.json").write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    (meta_dir / "versions.txt").write_text(cmd_version.render(cmd_version.collect()), encoding="utf-8")


def _forbidden_roots() -> tuple[Path, ...]:
    """Never bundled, even if LOG_DIR were ever misconfigured to overlap one
    of these -- the hard privacy rule (brief): no webui.db, no compactor
    store, no /data/ssh. Checked against each SOURCE file before it is
    copied, not just relied on by construction.
    """
    return (
        podenv.SNAPSHOT_WEBUI_DB.resolve(),
        podenv.LOCAL_WEBUI_DB.resolve(),
        podenv.STORE.resolve(),
        podenv.SSH_PERSIST.resolve(),
    )


def _is_forbidden(p: Path, roots: tuple[Path, ...]) -> bool:
    rp = p.resolve()
    for root in roots:
        if rp == root:
            return True
        try:
            if rp.is_relative_to(root):
                return True
        except ValueError:
            pass
    return False


def _read_tail(path: Path, max_bytes: int) -> tuple[bytes, bool]:
    """The last `max_bytes` of `path` (never the whole file if larger),
    and whether it was truncated to get there.
    """
    size = path.stat().st_size
    truncated = size > max_bytes
    with path.open("rb") as f:
        if truncated:
            f.seek(size - max_bytes)
        return f.read(), truncated


def _copy_one_log(src: Path, dst: Path, raw: bool, req_pattern) -> str | None:
    """Returns an error string, or None on success. Scrubs through
    `privacy.scrub_log_text` unless `raw` (review B4). `req_pattern`, when
    given (round 3, `--req`), restricts the COPIED text to just the lines
    matching it -- applied AFTER scrubbing, never instead of it: `--req`
    narrows which of her request's own lines are kept, it is not a
    substitute for the privacy scrub.
    """
    try:
        data, truncated = _read_tail(src, _LOG_MAX_BYTES)
    except OSError as e:
        return str(e)
    text = data.decode("utf-8", "replace")
    if not raw:
        text = privacy.scrub_log_text(text, source=src.name)
    if req_pattern is not None:
        text = "\n".join(ln for ln in text.splitlines() if req_pattern.search(ln))
        if text:
            text += "\n"
    if truncated:
        text = (
            f"(truncated: only the last {common.human_bytes(_LOG_MAX_BYTES)} of this file is "
            f"included -- see meta/disk.txt for its full size)\n"
        ) + text
    try:
        dst.write_text(text, encoding="utf-8")
    except OSError as e:
        return str(e)
    return None


def _copy_one_log_with_timeout(src: Path, dst: Path, raw: bool, req_pattern) -> str | None:
    """Runs `_copy_one_log` with a hard wall-clock timeout: MooseFS can
    stall a read indefinitely, and one log must never hang the whole
    bundle (review B4). The worker thread is a daemon: if it is still
    blocked when we give up on it, it does not keep this process alive --
    it is simply abandoned.
    """
    result: dict[str, str | None] = {}

    def _worker() -> None:
        result["error"] = _copy_one_log(src, dst, raw, req_pattern)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(_LOG_COPY_TIMEOUT_S)
    if t.is_alive():
        return f"copy timed out after {_LOG_COPY_TIMEOUT_S:.0f}s (MooseFS stall?) -- skipped"
    return result.get("error")


def _copy_logs(logs_dir: Path, *, raw: bool, req_pattern=None) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not podenv.LOG_DIR.is_dir():
        return
    roots = _forbidden_roots()
    for p in sorted(podenv.LOG_DIR.glob("*.log")):
        if not p.is_file():
            continue
        if _is_forbidden(p, roots):
            continue  # belt-and-braces: LOG_DIR should never overlap these, but never bundle one if it did
        error = _copy_one_log_with_timeout(p, logs_dir / p.name, raw, req_pattern)
        if error:
            (logs_dir / (p.name + ".copy-error.txt")).write_text(error, encoding="utf-8")


def _prune(out_dir: Path, keep: int, protect: Path) -> list[str]:
    """Delete only OUR OWN zla-bundle-*.tar.gz files in out_dir, oldest
    first, past `keep` -- NEVER `protect` (the bundle this run just built),
    whatever `keep` is, including 0 (review LOW: `--keep 0` must prune old
    ones, not the one just made). The timestamp format sorts
    lexicographically, so a plain sorted() on the name is a chronological
    sort.
    """
    existing = sorted(p for p in out_dir.glob(f"{_NAME_PREFIX}*.tar.gz") if p.is_file())
    doomed = existing if keep <= 0 else existing[:-keep]
    removed = []
    for p in doomed:
        if p == protect:
            continue
        try:
            p.unlink()
            removed.append(p.name)
        except OSError:
            pass
    return removed


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="zl bundle", add_help=True)
    ap.add_argument("--out", default=None, metavar="DIR")
    ap.add_argument("--keep", type=int, default=_DEFAULT_KEEP, metavar="N")
    ap.add_argument("--print-path", action="store_true")
    ap.add_argument("--raw-logs", action="store_true", help="copy logs UNSCRUBBED -- contains her message text")
    ap.add_argument("--yes", action="store_true", help="skip the --raw-logs typed confirmation")
    ap.add_argument("--req", default=None, metavar="ID",
                     help="include only this request's own log lines (v3.1.9.8 request tracing: "
                          "req=ID and its child ids req=ID.sum1/.facts/.dedup/...)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    req_pattern = None
    if args.req is not None:
        try:
            req_pattern = common.req_filter_pattern(args.req)
        except common.InvalidReqId as e:
            print(f"zl bundle: refusing -- {e}", file=sys.stderr)
            return 2

    if args.raw_logs:
        common.require_apply_confirmation(
            True, args.yes,
            what="include RAW, UNSCRUBBED logs in this bundle -- they contain her message text verbatim",
        )
        print(
            "*** WARNING: --raw-logs was given. The logs in this bundle are NOT scrubbed and MAY "
            "CONTAIN HER MESSAGE TEXT verbatim (SQL parameters from a failed write, fact text, reply "
            "fragments). Handle this bundle accordingly. ***"
        )

    out_dir = Path(args.out).expanduser() if args.out else podenv.BUNDLES
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime(_TS_FMT, time.gmtime())
    bundle_name = f"{_NAME_PREFIX}{ts}"
    archive_path = out_dir / f"{bundle_name}.tar.gz"
    partial_path = out_dir / f"{bundle_name}.tar.gz.partial"

    with tempfile.TemporaryDirectory(prefix="zl-bundle-") as tmp:
        root = Path(tmp) / bundle_name
        meta_dir = root / "meta"
        logs_dir = root / "logs" / "zla"
        meta_dir.mkdir(parents=True)
        _write_meta(meta_dir)
        _copy_logs(logs_dir, raw=args.raw_logs, req_pattern=req_pattern)
        if args.req is not None:
            (meta_dir / "req-filter.txt").write_text(
                f"--req {args.req} was applied: logs/zla/*.log were restricted to lines matching "
                f"req={args.req} or one of its child ids (req={args.req}.sum1, .facts, .dedup, ...). "
                "Lines from any other request were left out of this bundle entirely.\n",
                encoding="utf-8",
            )
        if args.raw_logs:
            (root / "RAW-LOGS-WARNING.txt").write_text(
                "These logs were copied with --raw-logs: UNSCRUBBED, and may contain her message "
                "text verbatim. See ops/zlops/privacy.py for what is normally removed.\n",
                encoding="utf-8",
            )

        # Build under a `.partial` name, then rename into place atomically --
        # a bundle that is killed mid-write (a hung MooseFS log, a signal)
        # must never leave a truncated `zla-bundle-*.tar.gz` that `zl bundle`
        # or `--latest` logic elsewhere could mistake for a complete one.
        with tarfile.open(partial_path, "w:gz") as tf:
            tf.add(root, arcname=bundle_name)
        partial_path.rename(archive_path)

    removed = _prune(out_dir, args.keep, protect=archive_path)

    if args.print_path:
        print(str(archive_path))
    else:
        print(f"bundle: {archive_path}")
        if removed:
            print(f"pruned {len(removed)} old bundle(s): {', '.join(removed)}")
    return 0
