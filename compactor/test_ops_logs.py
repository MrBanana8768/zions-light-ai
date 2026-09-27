"""Unit tests for ops/zlops/commands/logs.py -- the tail-logs.sh
passthrough, and round-3's `--req ID` addition (v3.1.9.8 request tracing):
restrict the stream to one request's own lines, entirely on this wrapper's
own side (tail-logs.sh itself is never modified -- memory.md: never
re-implement/modify a wrapped script).

Round-6 M2 rewrite: `--req`'s own history read and live follow now share
ONE byte-offset boundary per file (see logs.py's own module docstring for
why) instead of delegating the live half to tail-logs.sh's `-F -n 20`.
`_history_and_start_offsets` returns as soon as history is printed (safe
to call directly, no loop); `_follow_from_offsets` loops until
KeyboardInterrupt -- tests that need to observe a FEW iterations of it
mock `logs.time.sleep` to raise KeyboardInterrupt once nothing new is left
to read, which is also exactly how a real Ctrl-C stops it.

Plain-script style like compactor/test_setup_sshd_script.py: fakes go on
PATH-equivalent (a fake tail-logs.sh substituted via podenv.scripts_dir,
used only for the NON-`--req` passthrough path now); exit 0 pass, 1 fail.
"""

from __future__ import annotations

import io
import os
import stat
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from zlops.commands import logs  # noqa: E402


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


# A fake tail-logs.sh, used ONLY for the plain (non-`--req`) passthrough
# path now -- `--req` no longer spawns it at all.
_FAKE_TAIL_LOGS = """#!/usr/bin/env bash
if [ -n "$FAKE_TAIL_LOGS_ARGV_FILE" ]; then
    printf '%s\\n' "$*" > "$FAKE_TAIL_LOGS_ARGV_FILE"
fi
echo "ran tail-logs.sh"
"""


class _Fixture:
    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        td = Path(self._td.name)
        self.script = td / "tail-logs.sh"
        self.script.write_text(_FAKE_TAIL_LOGS, encoding="utf-8")
        self.script.chmod(self.script.stat().st_mode | stat.S_IEXEC)
        self.argv_file = td / "argv.txt"
        self._env_patch = mock.patch.dict(os.environ, {"FAKE_TAIL_LOGS_ARGV_FILE": str(self.argv_file)})
        self._env_patch.start()
        self._script_patch = mock.patch.object(logs.podenv, "scripts_dir", return_value=td)
        self._script_patch.start()
        self.log_dir = td / "logs"
        self.log_dir.mkdir()
        self._log_dir_patch = mock.patch.object(logs.podenv, "LOG_DIR", self.log_dir)
        self._log_dir_patch.start()
        return self

    def __exit__(self, *exc):
        self._log_dir_patch.stop()
        self._script_patch.stop()
        self._env_patch.stop()
        self._td.cleanup()
        return False

    def argv_seen(self) -> str:
        return self.argv_file.read_text(encoding="utf-8").strip() if self.argv_file.exists() else ""

    def write_log(self, name: str, lines: list[str]) -> None:
        (self.log_dir / name).write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")

    def append_log(self, name: str, lines: list[str]) -> None:
        with (self.log_dir / name).open("a", encoding="utf-8") as f:
            f.write("".join(ln + "\n" for ln in lines))


def _run_capturing_stdout(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with mock.patch.object(sys, "stdout", buf):
        code = logs.run(argv)
    return code, buf.getvalue()


def _run_history_capturing(req_id: str):
    """Calls only the history half (`_history_and_start_offsets`), which
    returns as soon as it is done printing -- safe to call directly
    without also starting the (infinite, Ctrl-C-only) live follower.
    """
    pattern = logs.common.req_filter_pattern(req_id)
    buf = io.StringIO()
    with mock.patch.object(sys, "stdout", buf):
        boundaries = logs._history_and_start_offsets(pattern)
    return boundaries, pattern, buf.getvalue()


def _run_follow_until_idle(pattern, boundaries) -> str:
    """Runs `_follow_from_offsets` until it has caught up with everything
    currently on disk and would otherwise sleep -- `logs.time.sleep` is
    mocked to raise KeyboardInterrupt at that point instead, which is
    exactly how a real Ctrl-C stops the real (infinite) loop.
    """
    buf = io.StringIO()
    with mock.patch.object(sys, "stdout", buf), \
         mock.patch.object(logs.time, "sleep", side_effect=KeyboardInterrupt):
        try:
            logs._follow_from_offsets(pattern, boundaries)
        except KeyboardInterrupt:
            pass
    return buf.getvalue()


def test_no_req_flag_runs_tail_logs_unfiltered():
    # No --req: the plain passthrough (`common.run_stream`) inherits stdio
    # straight through to the real terminal -- not capturable via
    # mock.patch.object(sys, "stdout"), so this checks the exit code and
    # that tail-logs.sh runs with its own default args (no forced --all).
    with _Fixture() as fx:
        code = logs.run([])
        assert_eq(code, 0, "`zl logs` with no --req exits 0")
        assert_true("--all" not in fx.argv_seen(), "no --req means tail-logs.sh runs with its own default mode")


def test_req_history_keeps_only_matching_lines():
    with _Fixture() as fx:
        fx.write_log("compactor.log", [
            "req=aaa111 conv=c1: dedup pass start",
            "req=aaa111.facts conv=c1: extracted 2 fact(s)",
            "req=bbb222 conv=c2: dedup pass start",
            "req=aaa1112 conv=c3: shares a prefix, must NOT match aaa111",
            "a plain line with no req= at all",
        ])
        _boundaries, _pattern, out = _run_history_capturing("aaa111")
        assert_true("req=aaa111 conv=c1: dedup pass start" in out, "the matching line survives")
        assert_true("req=aaa111.facts" in out, "the matching child-id line survives")
        assert_true("req=bbb222" not in out, "a different request's line is excluded")
        assert_true("aaa1112" not in out, "a request that merely shares a prefix is excluded")
        assert_true("a plain line with no req=" not in out, "a line with no req= at all is excluded")
        assert_true("[cmp] req=aaa111" in out, "tagged the same way tail-logs.sh tags compactor.log")


def test_req_history_keeps_the_background_form():
    """v3.1.9.8: a background tail's own log lines carry `req=<id>/bg:<tag>`
    (bgwork.BackgroundPool._run) instead of the bare `req=<id>` the
    request path uses -- `zl logs --req` must follow the request into the
    background too, while still excluding a merely-prefix-sharing id and
    a longer hex id.
    """
    with _Fixture() as fx:
        fx.write_log("compactor.log", [
            "req=aaa111 conv=c1: dedup pass start",
            "req=aaa111/bg:tail conv=c1: background tail running",
            "req=aaa111/bg:rollup conv=c1: background rollup running",
            "req=aaa111.facts conv=c1: extracted 2 fact(s)",
            "req=bbb222/bg:tail conv=c2: a DIFFERENT request's background tail",
            "req=aaa111-other conv=c3: shares a prefix, must NOT match",
        ])
        _boundaries, _pattern, out = _run_history_capturing("aaa111")
        assert_true("req=aaa111/bg:tail" in out, "the background form (bg:tail) survives")
        assert_true("req=aaa111/bg:rollup" in out, "the background form (bg:rollup) survives")
        assert_true("req=aaa111.facts" in out, "the plain child-id form still survives alongside it")
        assert_true("bbb222" not in out, "a different request's background line is excluded")
        assert_true("aaa111-other" not in out, "an id that merely shares this prefix is excluded")


def test_req_missing_value_is_refused():
    with _Fixture():
        code, _out = _run_capturing_stdout(["--req"])
        assert_eq(code, 2, "`zl logs --req` with no value is refused")


def test_req_rejects_an_unsafe_id():
    with _Fixture():
        code, _out = _run_capturing_stdout(["--req", "../../etc/passwd"])
        assert_eq(code, 2, "an unsafe --req value is refused")


def test_req_missing_log_file_is_skipped_silently():
    """A log file that does not exist yet (e.g. openwebui.log before
    OpenWebUI's first write) must not raise -- best-effort.
    """
    with _Fixture():
        boundaries, _pattern, _out = _run_history_capturing("aaa111")
        assert_true(len(boundaries) == len(logs._LOG_FILES), "a boundary is still recorded for every known file")


_FAKE_BINARY_GARBAGE = (
    b"req=aaa111 conv=c1: before garbage\n"
    b"\xff\xfe not valid utf-8 but req=aaa111 still\n"
    b"req=aaa111 conv=c1: after garbage\n"
)


def test_req_history_survives_non_utf8_bytes():
    """Round-4 MEDIUM, still true after the round-6 rewrite: a log line
    with non-UTF-8 bytes must not crash the read -- decode with
    errors='replace' and keep going.
    """
    with _Fixture() as fx:
        (fx.log_dir / "compactor.log").write_bytes(_FAKE_BINARY_GARBAGE)
        _boundaries, _pattern, out = _run_history_capturing("aaa111")
        assert_true("before garbage" in out, "the line before the bad bytes survives")
        assert_true("after garbage" in out, "the line after the bad bytes survives")


def test_req_reads_rotated_dot1_file_for_history():
    """Round-6 M2 requirement: history must include the ROTATED `<name>.1`
    file too (supervisord renames the old file to this on rotation), not
    just whatever is at `<name>` right now.
    """
    with _Fixture() as fx:
        (fx.log_dir / "compactor.log.1").write_text("req=aaa111 conv=old: rotated-out content\n", encoding="utf-8")
        fx.write_log("compactor.log", ["req=aaa111 conv=new: current content"])
        _boundaries, _pattern, out = _run_history_capturing("aaa111")
        assert_true("rotated-out content" in out, "the rotated .1 file's own matching line is shown", extra=out)
        assert_true("current content" in out, "the current file's own matching line is shown too", extra=out)
        # rotated (older) content must come first -- oldest history first.
        assert_true(out.index("rotated-out content") < out.index("current content"),
                    "the rotated file's content is printed BEFORE the current file's", extra=out)


def test_history_and_follow_share_one_boundary_zero_gap_zero_duplicates():
    """The round-6 fix itself: a FAST WRITER appends lines to a log file
    in the exact gap between the history read finishing and the live
    follower starting -- with a single shared byte-offset boundary
    (`_history_and_start_offsets`'s own return value), every one of those
    lines must be seen by the FOLLOWER exactly once, and lines already
    printed by history must never be printed again.
    """
    with _Fixture() as fx:
        fx.write_log("compactor.log", [f"req=aaa111 line {i:02d}" for i in range(25)])
        boundaries, pattern, history_out = _run_history_capturing("aaa111")
        for i in range(25):
            assert_true(f"line {i:02d}" in history_out, f"history line {i:02d} was printed", extra=history_out)

        # The fast writer: appends MORE lines exactly at the boundary this
        # test is about -- after the offset was recorded, before the
        # follower's first read.
        fx.append_log("compactor.log", [f"req=aaa111 line {i:02d}" for i in range(25, 30)])

        follow_out = _run_follow_until_idle(pattern, boundaries)
        for i in range(25, 30):
            assert_true(f"line {i:02d}" in follow_out, f"new line {i:02d} was picked up by the follower "
                        "(no gap)", extra=follow_out)
        combined = history_out + follow_out
        for i in range(30):
            assert_eq(combined.count(f"line {i:02d}"), 1, f"line {i:02d} appears EXACTLY once across history+follow")


def test_follow_detects_rotation_and_restarts_the_new_files_offset_at_zero():
    """Round-6 M2 requirement: the live follower must notice a rotation
    (the file at `<name>` replaced with a new, empty one -- supervisord's
    own rename-then-recreate) via its inode changing, and read the NEW
    file from its own beginning rather than seeking to the old (now
    meaningless) byte offset.
    """
    with _Fixture() as fx:
        fx.write_log("compactor.log", ["req=aaa111 pre-rotation line"])
        boundaries, pattern, _history_out = _run_history_capturing("aaa111")

        path = fx.log_dir / "compactor.log"
        path.rename(fx.log_dir / "compactor.log.1")
        path.write_text("req=aaa111 post-rotation line\n", encoding="utf-8")

        follow_out = _run_follow_until_idle(pattern, boundaries)
        assert_true("post-rotation line" in follow_out, "the new (post-rotation) file's content is followed",
                    extra=follow_out)
        assert_true("pre-rotation" not in follow_out,
                    "the old file's already-seen content is not re-printed by the follower", extra=follow_out)


def test_mutation_ignoring_rotation_reads_the_new_files_tail_instead_of_its_start():
    """Prove the test above bites: an old-offset-oblivious mutation would
    try to `seek()` past the brand-new (small) file's own end and read
    nothing at all -- confirm that failure mode, then confirm the real
    (restored) follower actually notices the rotation.
    """
    with _Fixture() as fx:
        # Chosen so that seeking to the PRE-rotation offset inside the
        # (differently-sized) POST-rotation file lands partway through
        # "req=aaa111", splitting the id off from the rest of the line --
        # exactly what a stale, rotation-blind offset would do, and why
        # the size alone (new size 30 > old offset 17, so the SEPARATE
        # truncation safeguard does not save this case) is not enough;
        # only noticing the inode actually changed is.
        fx.write_log("compactor.log", ["req=aaa111 short"])
        boundaries, pattern, _history_out = _run_history_capturing("aaa111")

        path = fx.log_dir / "compactor.log"
        old_offset = boundaries[path][0]
        path.rename(fx.log_dir / "compactor.log.1")
        path.write_text("req=aaa111 post-rotation line\n", encoding="utf-8")

        def _mutated_follow_once(pattern, path, offset) -> str:
            """A stripped copy of `_follow_from_offsets`'s own per-file
            read, MINUS the "did the inode change?" check -- it just seeks
            to the old byte offset no matter what.
            """
            buf = io.StringIO()
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                new_text = f.read()
            for line in new_text.splitlines(keepends=True):
                if pattern.search(line):
                    buf.write(f"[cmp] {line}" if line.endswith("\n") else f"[cmp] {line}\n")
            return buf.getvalue()

        mutated_out = _mutated_follow_once(pattern, path, old_offset)
        assert_true("post-rotation line" not in mutated_out,
                    "mutated (stale offset, rotation not detected) misses the new content -- confirms the bite",
                    extra=mutated_out)

        real_out = _run_follow_until_idle(pattern, boundaries)
        assert_true("post-rotation line" in real_out, "the real (restored) follower detects the rotation")


def test_trailing_partial_line_is_held_back_until_its_newline_arrives():
    """Round-7: a read that lands mid-line (the writer flushed up to some
    point inside a line but not its trailing newline yet) must not be
    emitted immediately -- the REST of that same line, appended on a
    later poll, would otherwise be printed as a spurious second "line".
    """
    with _Fixture() as fx:
        fx.write_log("compactor.log", [])
        boundaries, pattern, _history_out = _run_history_capturing("aaa111")
        path = fx.log_dir / "compactor.log"
        with path.open("a", encoding="utf-8") as f:
            f.write("req=aaa111 this line is not fin")  # deliberately no trailing \n
        first_pass_out = _run_follow_until_idle(pattern, boundaries)
        assert_true("this line is not fin" not in first_pass_out,
                    "the partial (no-newline) line is held back, not emitted early", extra=first_pass_out)
        with path.open("a", encoding="utf-8") as f:
            f.write("ished yet\n")
        second_pass_out = _run_follow_until_idle(pattern, boundaries)
        assert_true("req=aaa111 this line is not finished yet" in second_pass_out,
                    "the completed line is emitted whole, once, only once its newline arrives",
                    extra=second_pass_out)
        assert_eq(second_pass_out.count("this line is not fin"), 1,
                  "the line appears exactly once total across both polls (no early duplicate fragment)")


def test_rotation_mid_follow_drains_the_old_files_unread_tail():
    """Round-7: rotation happening WHILE the follower is already running
    (not just before it starts, which the round-6 test already covers)
    must not lose whatever was written to the OLD generation between the
    follower's last read of it and the rename -- the old handle is
    drained to its own true EOF before switching to the new file.
    """
    with _Fixture() as fx:
        fx.write_log("compactor.log", ["req=aaa111 line-A"])
        boundaries, pattern, _history_out = _run_history_capturing("aaa111")
        path = fx.log_dir / "compactor.log"
        # Written to the CURRENT (soon to be rotated) file -- never read
        # by anything before the rename below.
        with path.open("a", encoding="utf-8") as f:
            f.write("req=aaa111 line-B (written just before rotation)\n")
        path.rename(fx.log_dir / "compactor.log.1")
        path.write_text("req=aaa111 line-C (post-rotation)\n", encoding="utf-8")
        out = _run_follow_until_idle(pattern, boundaries)
        assert_true("line-B" in out, "the old generation's unread tail (written right before the rename) survives",
                    extra=out)
        assert_true("line-C" in out, "the new generation's content is also followed", extra=out)
        assert_true("line-A" not in out, "history's own already-seen line is not re-printed", extra=out)


def test_fast_writer_no_missing_no_duplicate_lines():
    """logsprobe/gap.py-style stress: a writer thread appends numbered
    lines as fast as it can while the real history+follow code runs
    concurrently -- every line must be seen exactly once (0 missing, 0
    duplicates), proving the shared byte-offset handoff plus the
    partial-line holdback do not drop or double-print anything under
    load, not just in the single-line-at-a-time cases above.
    """
    import re
    import threading
    from collections import Counter

    with _Fixture() as fx:
        fx.write_log("compactor.log", [])
        n = 4000
        stop = threading.Event()

        def writer():
            path = fx.log_dir / "compactor.log"
            with path.open("a", buffering=1, encoding="utf-8") as f:
                for i in range(n):
                    f.write(f"req=aaa111 seq={i:06d}\n")
            stop.set()

        boundaries, pattern, _history_out = _run_history_capturing("aaa111")
        buf = io.StringIO()
        real_sleep = logs.time.sleep

        def fake_sleep(s):
            if stop.is_set():
                raise KeyboardInterrupt
            real_sleep(min(s, 0.01))

        t = threading.Thread(target=writer)
        t.start()
        with mock.patch.object(sys, "stdout", buf), mock.patch.object(logs.time, "sleep", side_effect=fake_sleep):
            try:
                logs._follow_from_offsets(pattern, boundaries)
            except KeyboardInterrupt:
                pass
        t.join()
        out = buf.getvalue()
        seqs = [int(m) for m in re.findall(r"seq=(\d+)", out)]
        counts = Counter(seqs)
        missing = sorted(set(range(n)) - set(counts))
        dups = sorted(k for k, v in counts.items() if v > 1)
        assert_eq(missing, [], "0 missing lines under a fast concurrent writer")
        assert_eq(dups, [], "0 duplicate lines under a fast concurrent writer")


def test_progressed_not_set_before_a_successful_open():
    """Round-7: `progressed` must reflect an actual successful read, never
    be set merely because a file LOOKED like it had grown -- otherwise an
    open/read that then fails suppresses the poll-interval sleep on every
    iteration, spinning a tight busy loop instead of backing off. Proven
    here by making every open fail (the log directory itself is removed
    out from under the follower) and confirming `time.sleep` still gets
    called every single iteration rather than being skipped.
    """
    with _Fixture() as fx:
        fx.write_log("compactor.log", ["req=aaa111 line"])
        boundaries, pattern, _history_out = _run_history_capturing("aaa111")
        import shutil
        shutil.rmtree(fx.log_dir)
        sleep_calls = []

        def fake_sleep(s):
            sleep_calls.append(s)
            if len(sleep_calls) >= 3:
                raise KeyboardInterrupt

        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf), mock.patch.object(logs.time, "sleep", side_effect=fake_sleep):
            try:
                logs._follow_from_offsets(pattern, boundaries)
            except KeyboardInterrupt:
                pass
        assert_eq(len(sleep_calls), 3, "time.sleep is called every iteration when every open fails "
                                        "(progressed never wrongly claims otherwise)")


def test_history_never_emits_a_trailing_partial_line():
    """F3 (review follow-up): a log file ending mid-line (no trailing
    newline yet -- the writer has not flushed the rest) must not have
    that partial line printed by HISTORY, even if what's there so far
    matches the `--req` pattern -- it is not a complete record. The
    follower must pick it up whole, once its newline actually arrives.
    """
    with _Fixture() as fx:
        path = fx.log_dir / "compactor.log"
        path.write_text("req=aaa111 line 00\nreq=aaa111 line 01 not yet fin", encoding="utf-8")
        boundaries, pattern, history_out = _run_history_capturing("aaa111")
        assert_true("line 00" in history_out, "the complete first line is printed by history", extra=history_out)
        assert_true("not yet fin" not in history_out,
                    "the trailing partial line is NOT emitted by history", extra=history_out)

        # The rest of the line (and its newline) arrives later, in a
        # SEPARATE write -- a split write across the history/follow
        # boundary, exactly what F3 is about.
        with path.open("a", encoding="utf-8") as f:
            f.write("ished writing\n")

        follow_out = _run_follow_until_idle(pattern, boundaries)
        assert_true("req=aaa111 line 01 not yet finished writing" in follow_out,
                     "the follower emits the completed line WHOLE, exactly once", extra=follow_out)
        assert_eq(follow_out.count("line 01"), 1, "the split line is not duplicated or fragmented")


def test_mutation_replacing_tell_with_stat_size_creates_a_history_follow_gap():
    """F1 (review follow-up): deterministic mutation guard for
    `ops/zlops/commands/logs.py:96-101` (`_read_history_file`) -- if
    `offset = f.tell()` were replaced with a separate `path.stat().st_size`
    call made after the history handle has already read everything it is
    going to read, a FAST WRITER that appends more bytes in the narrow
    window between "history stopped reading" and "the stat() call" would
    have those extra bytes counted in the offset by the mutant (since
    stat() reads the CURRENT on-disk size) but never actually printed as
    history -- creating exactly the gap round-6/round-7 already closed
    once. This hooks `os.fstat` (called immediately after the offset line
    in the real function, for the inode) to append new content to the
    file at that exact point -- simulating the fast writer -- then proves:
    the REAL implementation's offset does not include the new bytes (the
    follower still sees them, no gap), while a `path.stat().st_size`-based
    mutant's offset WOULD already include them (the follower would then
    see nothing new -- a silent gap).
    """
    with _Fixture() as fx:
        path = fx.log_dir / "compactor.log"
        fx.write_log("compactor.log", [f"req=aaa111 line {i:02d}" for i in range(3)])
        pattern = logs.common.req_filter_pattern("aaa111")

        appended = {"done": False}
        real_fstat = os.fstat

        def _fstat_hook(fd):
            # Fires exactly once, right after the real function computes
            # `offset = f.tell()` and right before it stats the inode --
            # simulates a fast writer landing in that narrow window.
            if not appended["done"]:
                appended["done"] = True
                with path.open("a", encoding="utf-8") as g:
                    g.write("req=aaa111 line 99 (appended mid-history-read)\n")
            return real_fstat(fd)

        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf), mock.patch.object(logs.os, "fstat", side_effect=_fstat_hook):
            offset, _inode = logs._read_history_file(path, "cmp", pattern)
        history_out = buf.getvalue()
        assert_true("line 99" not in history_out,
                     "the appended line was not yet on disk when history actually read the file, "
                     "so it is correctly not printed as history", extra=history_out)

        # The REAL (f.tell()-based) offset must NOT already be past the
        # appended bytes -- the follower, seeked there, must still find them.
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            remainder = f.read()
        assert_true("line 99" in remainder,
                     "the real offset leaves the appended bytes for the follower to find (no gap)",
                     extra=remainder)

        # Now confirm the MUTANT really would create the gap this test is
        # about: a `path.stat().st_size` offset, taken AFTER the append
        # above has already landed, is past those bytes entirely.
        mutant_offset = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mutant_offset)
            mutant_remainder = f.read()
        assert_eq(mutant_remainder, "", "sanity: the stat()-based mutant offset really is past the appended "
                                          "bytes -- the follower would see nothing (the gap this test catches)")


if __name__ == "__main__":
    test_no_req_flag_runs_tail_logs_unfiltered()
    test_req_history_keeps_only_matching_lines()
    test_req_history_keeps_the_background_form()
    test_req_missing_value_is_refused()
    test_req_rejects_an_unsafe_id()
    test_req_missing_log_file_is_skipped_silently()
    test_req_history_survives_non_utf8_bytes()
    test_req_reads_rotated_dot1_file_for_history()
    test_history_and_follow_share_one_boundary_zero_gap_zero_duplicates()
    test_follow_detects_rotation_and_restarts_the_new_files_offset_at_zero()
    test_mutation_ignoring_rotation_reads_the_new_files_tail_instead_of_its_start()
    test_trailing_partial_line_is_held_back_until_its_newline_arrives()
    test_rotation_mid_follow_drains_the_old_files_unread_tail()
    test_fast_writer_no_missing_no_duplicate_lines()
    test_progressed_not_set_before_a_successful_open()
    test_history_never_emits_a_trailing_partial_line()
    test_mutation_replacing_tell_with_stat_size_creates_a_history_follow_gap()

    print("\nAll ops/commands/logs.py tests passed.")
