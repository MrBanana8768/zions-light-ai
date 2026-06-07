"""
Tier-3 validation of the backup endpoints (V2.3 Theme 1).

Triggers a real backup on the live pod and asserts it lands AND verifies —
the durability guarantee end-to-end. Localhost-only (admin).

These exercise the same create→verify→publish pipeline the daemon runs, so
a green run here means the deployed backup path actually works, not just the
unit-mocked version.
"""

import _harness as H


def test_run_backup_creates_verified_archive():
    """POST /admin/backups → a new, verified archive (HTTP 200)."""
    H.skip_if_no_admin("backup endpoints are localhost-only")
    status, report = H.admin_run_backup()
    assert status == 200, f"backup failed: HTTP {status}, report={report!r}"
    assert report.get("ok") is True, f"report not ok: {report!r}"
    assert report.get("verified") is True, f"archive not verified: {report!r}"
    assert report.get("archive", "").endswith(".tar.gz"), report


def test_backup_appears_in_listing():
    """After a run, the archive shows up in GET /admin/backups + latest info."""
    H.skip_if_no_admin()
    status, report = H.admin_run_backup()
    assert status == 200, report
    made = report["archive"]

    listing = H.admin_list_backups()
    names = {b["name"] for b in listing.get("backups", [])}
    assert made in names, f"{made} not in listing {names}"
    info = listing.get("info", {})
    assert info.get("count", 0) >= 1, info
    assert info.get("latest") is not None, info


def test_verify_newest_backup_passes():
    """The newest archive independently passes verification."""
    H.skip_if_no_admin()
    # Ensure at least one exists
    H.admin_run_backup()
    status, body = H.admin_verify_backup()
    assert status == 200, f"verify failed: HTTP {status}, body={body!r}"
    assert body.get("ok") is True, body
    assert body.get("archive", "").endswith(".tar.gz"), body


def test_verify_missing_archive_404():
    """Verifying a non-existent named archive returns 404 (or 503), not 500."""
    H.skip_if_no_admin()
    status, body = H.admin_verify_backup(name="zions-backup-does-not-exist.tar.gz")
    # Missing named file → verify_backup returns ok=False → 503; a truly
    # absent listing → 404. Either is a clean, non-crashing answer.
    assert status in (404, 503), f"unexpected status {status}: {body!r}"


def test_health_full_reports_backups():
    """/health/full carries the backups block (count + latest)."""
    H.skip_if_no_admin()
    H.admin_run_backup()  # guarantee at least one
    status, body = H.health_full()
    assert status in (200, 503), status
    backups = body.get("backups")
    assert isinstance(backups, dict), f"no backups block: {body.get('backups')!r}"
    assert backups.get("count", 0) >= 1, backups
