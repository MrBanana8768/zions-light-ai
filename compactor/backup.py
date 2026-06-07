"""
compactor.backup — V2.3 Theme 1: data durability.

The `/data` volume holds two things that cannot be regenerated if lost:
OpenWebUI's `webui.db` (chat history) and `compactor/` (facts JSON, summary
state, persona text, ChromaDB vectors). Models and the torch.compile cache
are re-downloadable; these are not. This module makes timestamped,
**verified** snapshots so a corrupted file, an accidental `/forget`, or a
bad delete is recoverable.

Design principles (this is the V2.3 "failure-tested before done" release —
the safety net itself must be trustworthy):

1. **A backup that can't be verified is not a backup.** After creating an
   archive we immediately restore it to a scratch dir and assert the
   SQLite db passes `PRAGMA integrity_check` and every memory JSON parses.
   If verification fails, the archive is deleted and the run reports
   FAILURE — false confidence is worse than a known gap.
2. **Live-SQLite-safe.** `webui.db` is being written by OpenWebUI while we
   back up. A raw file copy can capture a torn page. We use SQLite's online
   backup API (`Connection.backup()`) to get a consistent snapshot.
3. **Can't fill the disk.** A min-free-space guard refuses to start a
   backup that would risk filling `/data` (a full disk is itself a failure
   mode we're trying to prevent).
4. **Atomic publish.** The archive is written to a `.partial` temp name and
   `os.replace`d into place only after it verifies — readers/pruners never
   see a half-written archive.

Scope (V2.3 Theme 1, phase 1): **local backups** to a directory on the same
volume. This protects against the common, recoverable failures (corruption,
accidental delete, torn write). It does **not** survive total volume loss —
off-volume disaster recovery (object store) is required future work and
will need a migration. The `upload_hook` below is the designed-in seam.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("compactor.backup")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# What to back up. DATA_DIR is OpenWebUI's state root; webui.db lives there.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/openwebui"))
WEBUI_DB = Path(os.environ.get("COMPACTOR_BACKUP_WEBUI_DB", str(DATA_DIR / "webui.db")))
STORAGE_ROOT = Path(
    os.environ.get("COMPACTOR_STORAGE_ROOT", str(DATA_DIR / "compactor"))
)

# Where backups land. Default is a sibling dir on the same volume.
BACKUP_DIR = Path(os.environ.get("COMPACTOR_BACKUP_DIR", "/data/backups"))

# How many archives to keep (oldest pruned past this).
RETAIN = int(os.environ.get("COMPACTOR_BACKUP_RETAIN", "7") or 7)

# Daemon cadence.
INTERVAL_HOURS = float(os.environ.get("COMPACTOR_BACKUP_INTERVAL_HOURS", "24") or 24)

# Refuse to back up if the target volume has less than this much free space.
# Prevents the backup process from being the thing that fills the disk.
MIN_FREE_MB = int(os.environ.get("COMPACTOR_BACKUP_MIN_FREE_MB", "500") or 500)

# Optional off-volume target (future work). When unset, local only.
REMOTE_TARGET = os.environ.get("COMPACTOR_BACKUP_REMOTE", "").strip()

_ARCHIVE_PREFIX = "zions-backup-"
_ARCHIVE_SUFFIX = ".tar.gz"
_MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _free_mb(path: Path) -> float:
    """Free space (MB) on the filesystem holding `path` (or its nearest
    existing ancestor, since the dir may not exist yet)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free / (1024 * 1024)
    except Exception:
        return float("inf")  # can't tell → don't block


def _snapshot_sqlite(src: Path, dest: Path) -> bool:
    """Consistent online snapshot of a (possibly live) SQLite db via the
    backup API. Returns True if a snapshot was written, False if the source
    doesn't exist. Raises on a real failure."""
    if not src.is_file():
        return False
    # Open read-only-ish; the backup API handles WAL + concurrent writers.
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(dest))
        try:
            con.backup(dst)
        finally:
            dst.close()
    finally:
        con.close()
    return True


def list_backups(backup_dir: Path | None = None) -> list[dict]:
    """Return existing archives, newest first: [{name, path, size_bytes, mtime}]."""
    d = backup_dir or BACKUP_DIR
    if not d.exists():
        return []
    out: list[dict] = []
    for f in d.glob(f"{_ARCHIVE_PREFIX}*{_ARCHIVE_SUFFIX}"):
        try:
            st = f.stat()
            out.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": st.st_size,
                "mtime": int(st.st_mtime),
            })
        except OSError:
            continue
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

def create_backup(backup_dir: Path | None = None) -> Path:
    """Build a verified-elsewhere archive of webui.db + the compactor store.

    Writes to a `.partial` temp file; the caller (run_once) verifies it and
    only then is it published via os.replace. Returns the temp path.

    Raises RuntimeError if the min-free guard trips.
    """
    d = backup_dir or BACKUP_DIR
    d.mkdir(parents=True, exist_ok=True)

    free = _free_mb(d)
    if free < MIN_FREE_MB:
        raise RuntimeError(
            f"refusing to back up: only {free:.0f} MB free at {d} "
            f"(min {MIN_FREE_MB} MB) — free space before backups can run"
        )

    stamp = _now_stamp()
    staging = Path(tempfile.mkdtemp(prefix=f"{_ARCHIVE_PREFIX}{stamp}-", dir=str(d)))
    manifest: dict = {
        "created_at": int(time.time()),
        "stamp": stamp,
        "sources": {},
        "format": "tar.gz",
        "schema": "v1",
    }
    try:
        # 1. webui.db via online snapshot (live-safe)
        db_dest = staging / "webui.db"
        if _snapshot_sqlite(WEBUI_DB, db_dest):
            manifest["sources"]["webui.db"] = {
                "present": True, "bytes": db_dest.stat().st_size,
            }
        else:
            manifest["sources"]["webui.db"] = {"present": False}
            logger.warning(f"webui.db not found at {WEBUI_DB} — backing up memory only")

        # 2. compactor/ store (atomic-written files are individually consistent)
        if STORAGE_ROOT.is_dir():
            store_dest = staging / "compactor"
            shutil.copytree(STORAGE_ROOT, store_dest)
            n_files = sum(1 for _ in store_dest.rglob("*") if _.is_file())
            manifest["sources"]["compactor"] = {"present": True, "files": n_files}
        else:
            manifest["sources"]["compactor"] = {"present": False}

        (staging / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # 3. tar.gz the staging dir to a .partial temp archive
        partial = d / f"{_ARCHIVE_PREFIX}{stamp}{_ARCHIVE_SUFFIX}.partial"
        with tarfile.open(partial, "w:gz") as tar:
            tar.add(staging, arcname=".")
        return partial
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_backup(archive_path: Path) -> tuple[bool, str]:
    """Restore an archive to a scratch dir and assert it's actually usable:
      - the tar opens and extracts
      - manifest.json is present and parses
      - if webui.db was backed up, it opens AND PRAGMA integrity_check == ok
      - every *.json under compactor/ parses

    Returns (ok, detail). Never raises — a failure to verify is a False, not
    an exception, so the caller can delete the bad archive and carry on.
    """
    scratch = Path(tempfile.mkdtemp(prefix="zions-verify-"))
    try:
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(scratch, filter="data")  # path-traversal safe
        except Exception as e:
            return False, f"tar extract failed: {type(e).__name__}: {e}"

        manifest_path = scratch / _MANIFEST_NAME
        if not manifest_path.is_file():
            return False, "manifest.json missing from archive"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"manifest unparseable: {e}"

        # SQLite integrity (only if it was supposed to be there)
        db_expected = manifest.get("sources", {}).get("webui.db", {}).get("present")
        db_path = scratch / "webui.db"
        if db_expected:
            if not db_path.is_file():
                return False, "manifest says webui.db present but it's missing"
            try:
                con = sqlite3.connect(str(db_path))
                try:
                    row = con.execute("PRAGMA integrity_check").fetchone()
                finally:
                    con.close()
                if not row or row[0] != "ok":
                    return False, f"sqlite integrity_check failed: {row}"
            except Exception as e:
                return False, f"sqlite open/check failed: {type(e).__name__}: {e}"

        # Every memory JSON must parse
        store = scratch / "compactor"
        json_checked = 0
        if store.is_dir():
            for jf in store.rglob("*.json"):
                try:
                    json.loads(jf.read_text(encoding="utf-8"))
                    json_checked += 1
                except Exception as e:
                    return False, f"corrupt memory file {jf.name}: {e}"

        return True, f"db={'ok' if db_expected else 'absent'}, {json_checked} json file(s) parsed"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------

def prune_old_backups(backup_dir: Path | None = None, retain: int | None = None) -> list[str]:
    """Delete archives older than the newest `retain`. Returns names removed."""
    keep = RETAIN if retain is None else retain
    archives = list_backups(backup_dir)
    removed: list[str] = []
    for entry in archives[keep:]:
        try:
            Path(entry["path"]).unlink()
            removed.append(entry["name"])
        except OSError as e:
            logger.warning(f"could not prune {entry['name']}: {e}")
    return removed


# ---------------------------------------------------------------------------
# Off-volume seam (future work)
# ---------------------------------------------------------------------------

def upload_hook(archive_path: Path) -> bool:
    """Designed-in seam for off-volume disaster recovery (object store).

    V2.3 phase 1 is local-only, so this is a no-op unless
    COMPACTOR_BACKUP_REMOTE is set — and even then it currently only logs,
    because true off-volume DR is deferred future work that will need a
    migration (provider choice + credentials + a real uploader, e.g. boto3
    or rclone). Wiring it here now means the create→verify→publish→upload
    pipeline already has the call site.
    """
    if not REMOTE_TARGET:
        return False
    logger.warning(
        f"COMPACTOR_BACKUP_REMOTE={REMOTE_TARGET!r} is set but off-volume "
        f"upload is not yet implemented (V2.3 future work). Archive "
        f"{archive_path.name} kept locally only."
    )
    return False


# ---------------------------------------------------------------------------
# Orchestration: create → verify → publish → prune
# ---------------------------------------------------------------------------

def run_once(backup_dir: Path | None = None) -> dict:
    """One full backup cycle. Returns a structured report. Never raises —
    failures are reported, not thrown, so the daemon keeps running."""
    d = backup_dir or BACKUP_DIR
    t0 = time.monotonic()
    report: dict = {"ok": False, "archive": None, "verified": False, "detail": ""}
    partial: Path | None = None
    try:
        partial = create_backup(d)
        ok, detail = verify_backup(partial)
        report["detail"] = detail
        if not ok:
            # No false confidence — delete the unverifiable archive.
            try:
                partial.unlink()
            except OSError:
                pass
            report["detail"] = f"VERIFICATION FAILED: {detail}"
            logger.error(f"backup verification failed, archive discarded: {detail}")
            return report
        # Publish atomically: drop the .partial suffix.
        final = partial.with_suffix("")  # strips ".partial" → ...tar.gz
        os.replace(partial, final)
        report["archive"] = final.name
        report["verified"] = True
        upload_hook(final)
        removed = prune_old_backups(d)
        report["pruned"] = removed
        report["ok"] = True
        report["elapsed_s"] = round(time.monotonic() - t0, 1)
        logger.info(
            f"backup ok: {final.name} ({detail}); pruned {len(removed)}; "
            f"{report['elapsed_s']}s"
        )
        return report
    except Exception as e:
        if partial and partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        report["detail"] = f"{type(e).__name__}: {e}"
        logger.error(f"backup failed: {report['detail']}")
        return report


def latest_backup_info(backup_dir: Path | None = None) -> dict:
    """Summary for /health/full + admin: count + newest timestamp."""
    archives = list_backups(backup_dir)
    return {
        "count": len(archives),
        "latest": archives[0]["name"] if archives else None,
        "latest_mtime": archives[0]["mtime"] if archives else None,
        "dir": str(backup_dir or BACKUP_DIR),
    }


# ---------------------------------------------------------------------------
# Restore (destructive — for the runbook / CLI, gated by --yes)
# ---------------------------------------------------------------------------

def restore_backup(
    archive_path: Path,
    *,
    data_dir: Path | None = None,
    storage_root: Path | None = None,
    confirm: bool = False,
) -> dict:
    """Restore an archive over the live data locations. DESTRUCTIVE — it
    overwrites webui.db and the compactor store. Requires confirm=True.

    Verifies the archive first (won't restore an unusable backup), then
    extracts to scratch and moves the pieces into place. Returns a report.
    """
    if not confirm:
        raise RuntimeError("restore is destructive; pass confirm=True (CLI: --yes)")
    ddir = data_dir or DATA_DIR
    sroot = storage_root or STORAGE_ROOT

    ok, detail = verify_backup(archive_path)
    if not ok:
        raise RuntimeError(f"refusing to restore an unverifiable archive: {detail}")

    scratch = Path(tempfile.mkdtemp(prefix="zions-restore-"))
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(scratch, filter="data")  # path-traversal safe
        restored: list[str] = []
        # webui.db
        src_db = scratch / "webui.db"
        if src_db.is_file():
            ddir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_db, ddir / "webui.db")
            restored.append("webui.db")
        # compactor store — replace wholesale
        src_store = scratch / "compactor"
        if src_store.is_dir():
            if sroot.exists():
                shutil.rmtree(sroot)
            shutil.copytree(src_store, sroot)
            restored.append("compactor")
        logger.info(f"restored {restored} from {archive_path.name}")
        return {"ok": True, "restored": restored, "archive": archive_path.name}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Daemon + CLI
# ---------------------------------------------------------------------------

def run_daemon(interval_hours: float | None = None) -> None:
    """Periodic loop for the supervisord sidecar. Backs up every
    `interval_hours`, forever. Each cycle is wrapped so a single failure
    doesn't kill the loop."""
    interval = (INTERVAL_HOURS if interval_hours is None else interval_hours) * 3600.0
    logger.info(
        f"backup daemon started: every {interval/3600:.1f}h → {BACKUP_DIR} "
        f"(retain {RETAIN})"
    )
    while True:
        report = run_once()
        if not report["ok"]:
            logger.error(f"backup cycle failed: {report['detail']}")
        time.sleep(interval)


def _fmt(report: dict) -> str:
    mark = "OK" if report.get("ok") else "FAIL"
    return f"[{mark}] {report.get('archive') or '-'}  {report.get('detail', '')}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Zion's Light AI data backup.")
    p.add_argument("--once", action="store_true", help="Run one backup cycle and exit.")
    p.add_argument("--daemon", action="store_true", help="Run forever on the configured interval.")
    p.add_argument("--list", action="store_true", help="List existing backups.")
    p.add_argument("--verify", metavar="ARCHIVE", help="Verify an existing archive.")
    p.add_argument("--restore", metavar="ARCHIVE", help="Restore from an archive (DESTRUCTIVE).")
    p.add_argument("--yes", action="store_true", help="Confirm a destructive --restore.")
    p.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = p.parse_args(argv)

    import logsetup
    logsetup.configure()  # honors COMPACTOR_LOG_FORMAT (text/json)

    if args.list:
        archives = list_backups()
        print(json.dumps(archives, indent=2) if args.json else
              "\n".join(f"{a['name']}  {a['size_bytes']} B" for a in archives) or "(none)")
        return 0
    if args.verify:
        ok, detail = verify_backup(Path(args.verify))
        print(json.dumps({"ok": ok, "detail": detail}) if args.json else f"[{'OK' if ok else 'FAIL'}] {detail}")
        return 0 if ok else 1
    if args.restore:
        try:
            rep = restore_backup(Path(args.restore), confirm=args.yes)
            print(json.dumps(rep) if args.json else f"restored: {rep['restored']}")
            return 0
        except Exception as e:
            print(f"restore failed: {e}", file=sys.stderr)
            return 1
    if args.daemon:
        run_daemon()
        return 0
    # default: --once
    rep = run_once()
    print(json.dumps(rep, indent=2) if args.json else _fmt(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
