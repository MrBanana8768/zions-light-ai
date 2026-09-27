"""Unit tests for ops/zlops/commands/bundle.py -- the diagnostic bundle:
layout, env redaction with planted secrets (the privacy hard rule), that
webui.db/the compactor store/ssh keys are never reachable from what it
writes, and that pruning only ever touches our own `zla-bundle-*.tar.gz`
files.

Plain-script style like compactor/test_setup_sshd_script.py: exit 0 pass,
1 fail. No docker/network needed -- http/telemetry/env are all faked.
"""

from __future__ import annotations

import io
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import tarfile
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from zlops import common, telemetry  # noqa: E402  (telemetry: B's module, only monkeypatched, never edited)
from zlops.commands import bundle, disk as cmd_disk  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label, extra=""):
    if not cond:
        print(f"FAIL {label}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


_PLANTED_SECRETS = {
    "HF_TOKEN": "hf_plantedsecret123",
    "SOME_API_KEY": "plainsecret",
}


class _PodLayout:
    """A throwaway directory tree standing in for /data, with every
    podenv path monkeypatched to point inside it, plus http/telemetry/env
    faked so `bundle.run` needs no real compactor, supervisord, or network.
    """

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        root = Path(self._td.name)
        self.data = root / "data"
        self.log_dir = self.data / "logs"
        self.backups = self.data / "backups"
        self.hf_home = self.data / "models"
        self.store = self.data / "openwebui" / "compactor"
        self.snapshot_db = self.data / "openwebui" / "webui.db"
        self.local_db = root / "var-lib-openwebui" / "webui.db"
        self.ssh_persist = self.data / "ssh"
        self.zl_home = root / "zl-ops"
        self.bundles = self.zl_home / "bundles"
        self.out_dir = root / "out"
        self.control_root = self.data / "openwebui" / "compactor-control"
        self.pending_root = self.data / "openwebui" / "compactor-pending"

        for d in (self.log_dir, self.backups, self.hf_home, self.store, self.ssh_persist, self.out_dir):
            d.mkdir(parents=True)
        self.snapshot_db.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_db.write_bytes(b"sqlite fake db -- must never appear in a bundle")
        self.local_db.parent.mkdir(parents=True, exist_ok=True)
        self.local_db.write_bytes(b"sqlite fake db -- must never appear in a bundle")
        (self.ssh_persist / "ssh_host_ed25519_key").write_text("PRIVATE KEY -- must never appear", encoding="utf-8")
        (self.store / "some_conv.facts.json").write_text('{"secret": "memory content"}', encoding="utf-8")

        for name in ("compactor.log", "vllm.log", "compactor-error.log"):
            (self.log_dir / name).write_text(f"{name}: a normal log line\n", encoding="utf-8")
        (self.log_dir / "not-a-log.txt").write_text("should never be picked up (not *.log)\n", encoding="utf-8")

        self._patches = [
            mock.patch.object(bundle.podenv, "DATA", self.data),
            mock.patch.object(bundle.podenv, "LOG_DIR", self.log_dir),
            mock.patch.object(bundle.podenv, "BACKUPS", self.backups),
            mock.patch.object(bundle.podenv, "HF_HOME", self.hf_home),
            mock.patch.object(bundle.podenv, "STORE", self.store),
            mock.patch.object(bundle.podenv, "SNAPSHOT_WEBUI_DB", self.snapshot_db),
            mock.patch.object(bundle.podenv, "LOCAL_WEBUI_DB", self.local_db),
            mock.patch.object(bundle.podenv, "SSH_PERSIST", self.ssh_persist),
            mock.patch.object(bundle.podenv, "ZL_HOME", self.zl_home),
            mock.patch.object(bundle.podenv, "BUNDLES", self.bundles),
            mock.patch.object(bundle.podenv, "CONTROL_ROOT", self.control_root),
            mock.patch.object(bundle.podenv, "PENDING_ROOT", self.pending_root),
            mock.patch.object(common, "http_get_json", side_effect=self._fake_http),
            mock.patch.object(common, "merged_environ", return_value=dict(_PLANTED_SECRETS, PATH="/usr/bin:/bin")),
            mock.patch.object(telemetry, "collect_snapshot", return_value={"fake": "snapshot"}),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        self._td.cleanup()
        return False

    @staticmethod
    def _fake_http(url, timeout=5.0):
        if "conversations" in url:
            return 200, {"conversations": []}, None
        return 200, {"status": "ok", "status_reasons": []}, None


def _extract_all(archive_path: Path, dest: Path) -> Path:
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest, filter="data")  # noqa: S202 -- our own just-built archive
        top = tf.getnames()[0].split("/")[0]
    return dest / top


def test_bundle_layout_matches_the_real_bundle():
    with _PodLayout() as fx:
        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf):
            code = bundle.run(["--out", str(fx.out_dir), "--print-path"])
        assert_eq(code, 0, "bundle.run exits 0")
        archive_path = Path(buf.getvalue().strip())
        assert_true(archive_path.is_file(), "bundle.run prints a real archive path", extra=buf.getvalue())
        assert_true(archive_path.name.startswith("zla-bundle-") and archive_path.name.endswith(".tar.gz"),
                    "the archive is named zla-bundle-<ts>.tar.gz", extra=archive_path.name)

        with tarfile.open(archive_path, "r:gz") as tf:
            names = {n.split("/", 1)[1] for n in tf.getnames() if "/" in n}
        expected_meta = {
            "meta/health.json", "meta/supervisorctl.txt", "meta/gpu.txt", "meta/disk.txt",
            "meta/env-redacted.txt", "meta/conversations.json", "meta/telemetry.json", "meta/versions.txt",
        }
        assert_true(expected_meta <= names, "every meta/ file from the brief is present",
                    extra=str(sorted(names)))
        for log_name in ("logs/zla/compactor.log", "logs/zla/vllm.log", "logs/zla/compactor-error.log"):
            assert_true(log_name in names, f"{log_name} is in the bundle", extra=str(sorted(names)))
        assert_true("logs/zla/not-a-log.txt" not in names, "a non-.log file under LOG_DIR is never bundled")


def test_disk_txt_reports_control_and_pending_roots_when_present():
    """compactor's own CONTROL_ROOT/PENDING_ROOT (continuity.py's ledger
    and durability sidecar, siblings of STORAGE_ROOT -- podenv.py's own
    comment) get their own `du`-based size line in `meta/disk.txt`, same
    as backups/HF cache already do -- but ONLY when they actually exist,
    same "if present" convention as the rest of this file.
    """
    with _PodLayout() as fx:
        fx.control_root.mkdir(parents=True)
        (fx.control_root / "ledger.json").write_text("{}", encoding="utf-8")
        # pending_root deliberately left absent.
        with mock.patch.object(sys, "stdout", io.StringIO()):
            code = bundle.run(["--out", str(fx.out_dir), "--print-path"])
        assert_eq(code, 0, "bundle.run exits 0")
        archive_path = sorted(fx.out_dir.glob("zla-bundle-*.tar.gz"))[-1]
        extracted = _extract_all(archive_path, fx.out_dir / "extracted")
        disk_txt = (extracted / "meta" / "disk.txt").read_text(encoding="utf-8")
        assert_true("control root" in disk_txt and str(fx.control_root) in disk_txt,
                    "the present control root gets its own line", extra=disk_txt)
        assert_true("pending root" not in disk_txt,
                    "the absent pending root is not mentioned at all", extra=disk_txt)


def test_zl_disk_reports_control_and_pending_roots_when_present():
    """L1/mut.py M10 regression: `zl disk` itself (`disk.py`'s own
    `run()`) has a SEPARATE copy of this same control/pending-root
    reporting logic from bundle.py's (already covered above by
    test_disk_txt_reports_control_and_pending_roots_when_present) --
    each must be tested independently, or a mutation that only guts
    disk.py's own copy survives unnoticed.
    """
    with _PodLayout() as fx:
        fx.control_root.mkdir(parents=True)
        (fx.control_root / "ledger.json").write_text("{}", encoding="utf-8")
        # pending_root deliberately left absent.
        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf):
            code = cmd_disk.run([])
        assert_eq(code, 0, "zl disk exits 0")
        out = buf.getvalue()
        assert_true("control root" in out and str(fx.control_root) in out,
                    "the present control root gets its own line in `zl disk`'s own output", extra=out)
        assert_true("pending root" not in out,
                    "the absent pending root is not mentioned at all", extra=out)


def test_bundle_redacts_env_and_never_leaks_secrets_anywhere_in_the_tree():
    with _PodLayout() as fx:
        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf):
            bundle.run(["--out", str(fx.out_dir), "--print-path"])
        archive_path = Path(buf.getvalue().strip())

        with tempfile.TemporaryDirectory() as extract_root:
            top = _extract_all(archive_path, Path(extract_root))
            env_text = (top / "meta" / "env-redacted.txt").read_text(encoding="utf-8")
            assert_true("hf_plantedsecret123" not in env_text, "the planted HF_TOKEN value never appears")
            assert_true("plainsecret" not in env_text, "the planted API key value never appears")
            assert_true("HF_TOKEN=<redacted>" in env_text, "HF_TOKEN is redacted, not dropped")
            assert_true("PATH=/usr/bin:/bin" in env_text, "a non-secret var passes through unredacted")

            # The real-image suite's own invariant, reproduced here without docker:
            # neither planted secret's value appears ANYWHERE in the extracted tree.
            offenders = []
            for p in top.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for secret in _PLANTED_SECRETS.values():
                    if secret in text:
                        offenders.append((str(p), secret))
            assert_eq(offenders, [], "no planted secret value appears anywhere in the extracted bundle")


def test_bundle_never_includes_forbidden_paths():
    with _PodLayout() as fx:
        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf):
            bundle.run(["--out", str(fx.out_dir), "--print-path"])
        archive_path = Path(buf.getvalue().strip())
        with tempfile.TemporaryDirectory() as extract_root:
            top = _extract_all(archive_path, Path(extract_root))
            all_text_and_names = []
            for p in top.rglob("*"):
                if p.is_file():
                    all_text_and_names.append(str(p))
            joined = "\n".join(all_text_and_names)
            assert_true("webui.db" not in joined, "no file named webui.db was ever copied into the bundle")
            for p in top.rglob("*"):
                if p.is_file():
                    try:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    assert_true("PRIVATE KEY -- must never appear" not in text,
                                f"the ssh host key never leaks into {p}")
                    assert_true("memory content" not in text,
                                f"the compactor store's own content never leaks into {p}")


def test_req_filter_keeps_only_the_matching_requests_lines():
    """Round-3 coordinator addendum: `zl bundle --req ID` restricts every
    copied log to just that request's own lines (its own req=ID and any
    child id req=ID.xxx), and never touches a DIFFERENT request's lines
    even when it shares a prefix.
    """
    with _PodLayout() as fx:
        (fx.log_dir / "compactor.log").write_text(
            "req=aaa111 conv=c1: dedup pass start\n"
            "req=aaa111.facts conv=c1: extracted 2 fact(s)\n"
            "req=bbb222 conv=c2: dedup pass start\n"
            "req=aaa1112 conv=c3: an unrelated request that merely shares the prefix\n"
            "req=aaa111 conv=c1 msgs=1 ttft_ms=10 total_ms=20 tokens=5 finish=stop status=200\n",
            encoding="utf-8",
        )
        code = bundle.run(["--out", str(fx.out_dir), "--req", "aaa111", "--print-path"])
        assert_eq(code, 0, "bundle.run with --req exits 0")
        archive = sorted(fx.out_dir.glob("zla-bundle-*.tar.gz"))[-1]
        with tempfile.TemporaryDirectory() as dest:
            top = _extract_all(archive, Path(dest))
            text = (top / "logs" / "zla" / "compactor.log").read_text(encoding="utf-8")
            assert_true("req=aaa111 conv=c1: dedup pass start" in text, "the matching line survives")
            assert_true("req=aaa111.facts" in text, "the matching child-id line survives")
            assert_true("req=bbb222" not in text, "a different request's line is excluded")
            assert_true("aaa1112" not in text, "a request that merely shares a prefix is excluded, not matched")
            note = (top / "meta" / "req-filter.txt").read_text(encoding="utf-8")
            assert_true("aaa111" in note, "meta/req-filter.txt records which id was applied")


def test_req_filter_keeps_the_background_form_too():
    """Round-7 coordinator addendum: a background tail's own log lines
    carry `req=ID/bg:tag` (bgwork.BackgroundPool._run) rather than the
    bare `req=ID` the request path uses -- `zl bundle --req` must keep
    those too, and still exclude an id that merely shares this prefix.
    """
    with _PodLayout() as fx:
        (fx.log_dir / "compactor.log").write_text(
            "req=aaa111 conv=c1: dedup pass start\n"
            "req=aaa111/bg:tail conv=c1: background tail running\n"
            "req=bbb222/bg:tail conv=c2: a DIFFERENT request's background tail\n"
            "req=aaa111-other conv=c3: shares a prefix, must NOT match\n",
            encoding="utf-8",
        )
        code = bundle.run(["--out", str(fx.out_dir), "--req", "aaa111", "--print-path"])
        assert_eq(code, 0, "bundle.run with --req exits 0")
        archive = sorted(fx.out_dir.glob("zla-bundle-*.tar.gz"))[-1]
        with tempfile.TemporaryDirectory() as dest:
            top = _extract_all(archive, Path(dest))
            text = (top / "logs" / "zla" / "compactor.log").read_text(encoding="utf-8")
            assert_true("req=aaa111/bg:tail" in text, "the background form survives")
            assert_true("bbb222" not in text, "a different request's background line is excluded")
            assert_true("aaa111-other" not in text, "an id that merely shares this prefix is excluded")


def test_req_filter_rejects_an_unsafe_id():
    with _PodLayout() as fx:
        code = bundle.run(["--out", str(fx.out_dir), "--req", "../../etc/passwd"])
        assert_eq(code, 2, "an unsafe --req value is refused, not built into a regex/filename unchecked")


def test_mutation_disabling_req_filter_leaks_other_requests():
    """Prove the test above bites: inline the OLD (no `--req` at all)
    `_copy_one_log` behaviour and show a different request's line would
    have survived.
    """
    with _PodLayout() as fx:
        (fx.log_dir / "compactor.log").write_text(
            "req=aaa111 conv=c1: dedup pass start\nreq=bbb222 conv=c2: dedup pass start\n",
            encoding="utf-8",
        )
        # mutated: req_pattern silently ignored (the pre-round-3 shape of _copy_one_log)
        src = fx.log_dir / "compactor.log"
        dst = Path(fx.out_dir) / "mutated-copy.log"
        bundle._copy_one_log(src, dst, raw=False, req_pattern=None)
        mutated_text = dst.read_text(encoding="utf-8")
        assert_true("req=bbb222" in mutated_text,
                    "mutated (req_pattern=None) copy includes every request -- confirms the bite")
        # restored: the real call site always passes a pattern when --req is given
        pattern = common.req_filter_pattern("aaa111")
        dst2 = Path(fx.out_dir) / "restored-copy.log"
        bundle._copy_one_log(src, dst2, raw=False, req_pattern=pattern)
        restored_text = dst2.read_text(encoding="utf-8")
        assert_true("req=bbb222" not in restored_text, "restored copy excludes the other request again")


def test_prune_keeps_only_the_newest_N_of_our_own_bundles():
    with _PodLayout() as fx:
        # 12 pre-existing bundles with sortable names, plus one unrelated file.
        for i in range(12):
            (fx.out_dir / f"zla-bundle-2026010{i:02d}T000000Z.tar.gz").write_bytes(b"x")
            time.sleep(0.001)
        (fx.out_dir / "not-ours.tar.gz").write_bytes(b"leave me alone")

        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf):
            code = bundle.run(["--out", str(fx.out_dir), "--keep", "3", "--print-path"])
        assert_eq(code, 0, "bundle.run with --keep 3 exits 0")

        remaining = sorted(p.name for p in fx.out_dir.glob("zla-bundle-*.tar.gz"))
        assert_eq(len(remaining), 3, "exactly `--keep` zla-bundle archives remain")
        newest_printed = Path(buf.getvalue().strip()).name
        assert_true(newest_printed in remaining, "the just-created bundle survives its own prune")
        assert_true((fx.out_dir / "not-ours.tar.gz").is_file(), "prune never touches a non-zla-bundle file")


if __name__ == "__main__":
    test_bundle_layout_matches_the_real_bundle()
    test_disk_txt_reports_control_and_pending_roots_when_present()
    test_zl_disk_reports_control_and_pending_roots_when_present()
    test_bundle_redacts_env_and_never_leaks_secrets_anywhere_in_the_tree()
    test_bundle_never_includes_forbidden_paths()
    test_req_filter_keeps_only_the_matching_requests_lines()
    test_req_filter_keeps_the_background_form_too()
    test_req_filter_rejects_an_unsafe_id()
    test_mutation_disabling_req_filter_leaks_other_requests()
    test_prune_keeps_only_the_newest_N_of_our_own_bundles()

    print("\nAll ops/commands/bundle.py tests passed.")
