"""`zl logs [...]` -- passthrough to scripts/tail-logs.sh (read-only, never
exits on its own; Ctrl-C stops it, same as running it directly).

`zl logs --req ID [...]` (round 3, coordinator addendum on v3.1.9.8's
request tracing): restricts the stream to just ONE request's own lines --
`req=ID` and any of its child ids (`req=ID.sum1`, `.facts`, `.dedup`, ...).
This is OUR OWN filter, layered on top, not a new flag on tail-logs.sh
(never re-implement/modify a wrapped script -- memory.md): tail-logs.sh's
own `--grep` only ADDS a pattern to its SIGNAL set (widens what it shows);
`--req` needs the opposite, a narrowing restriction, and (round 6) its own
full history/live follower, for reasons below -- the general (non-`--req`)
path is still a plain, unmodified passthrough to tail-logs.sh.

Round-6 M2 (surviving bug from round-4's own MEDIUM fix): round-4 read
each log file's history ONCE (everything before the last 20 lines) and
THEN handed off to tail-logs.sh, whose own `follow()` re-tails the last 20
lines when ITS `tail -F` actually starts a moment later. Between "we
finished reading history" and "tail -F actually starts", a FAST writer can
append more than 20 lines (some are then never shown at all -- a GAP) or
fewer than 20 (some lines are shown TWICE -- a DUPLICATE), because the two
reads use "the last N lines" as their boundary instead of a shared byte
offset. Fixed here by doing history AND live follow ourselves, sharing one
boundary: the exact byte offset each file's history read stopped at. This
also means `--req` no longer depends on tail-logs.sh's own `-F` at all
(explicitly fine per the brief -- "kept small, handles rotation, doesn't
block the UI" -- so it stays that way deliberately, not a place to add
tail-logs.sh's other features): it re-implements just enough of "notice
new bytes, notice rotation" for this one narrow use, polling on a short
interval rather than blocking indefinitely on any one file (round-6 LOW:
if the underlying MooseFS mount itself stalls on a read, THAT read blocks
-- there is no portable way to put a timeout on a plain file read in
Python -- but Ctrl-C (KeyboardInterrupt) still reaches and unwinds this
loop normally, same as it always could here).

Rotation: supervisord rotates each log at `stdout_logfile_maxbytes` by
renaming the current file to `<name>.1` and starting a new empty one at
`<name>` (scripts/tail-logs.sh's own comment: "supervisord rotates these
at 50MB"). History reads `<name>.1` (if present) BEFORE `<name>` itself,
oldest first; the live follower detects rotation by inode (a new `<name>`
appearing with a different `st_ino` than the one history/the previous poll
saw) and restarts that file's own offset at 0.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .. import common, podenv

# The same file list and tags tail-logs.sh's own `follow` calls apply, in
# the same order (scripts/tail-logs.sh) -- kept here so `--req`'s own
# follower and history read cover exactly the same files tail-logs.sh
# does, never a second, divergent list.
_LOG_FILES = (
    ("cmp", "compactor.log"),
    ("cmp!", "compactor-error.log"),
    ("llm", "vllm-error.log"),
    ("web", "openwebui.log"),
)
_REQ_POLL_INTERVAL_S = 0.3


def _emit(tag: str, line: str) -> None:
    try:
        sys.stdout.write(f"[{tag}] {line}" if line.endswith("\n") else f"[{tag}] {line}\n")
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        pass


def _read_history_file(path: Path, tag: str, pattern) -> tuple[int, int | None]:
    """Best-effort: a missing/unreadable file is silently skipped (matches
    tail-logs.sh's own `follow()`, which prints "(no such file: ...)" only
    for the live half, not for history) -- returns `(0, None)` in that case.

    Returns `(byte_offset_at_end, inode)`, both taken from the SAME open
    handle this function just finished reading history with (`f.tell()`
    and `os.fstat(f.fileno())`) -- round-7: a separate `path.stat()` call
    made AFTER this handle was closed would reopen exactly the kind of gap
    round-6 already closed once (between "history read stopped here" and
    "here is the boundary" for the live follower to start from): a fast
    writer appending bytes in that window would see them counted in the
    stat()'s size but never actually read as history, so the live
    follower -- starting from that later offset -- would never show them
    either. Taking both from the handle that did the actual reading
    closes that window to zero. (F1, review follow-up: this is exactly
    the mutation `offset = f.tell()` -> `offset = path.stat().st_size`
    would reintroduce -- see `test_ops_logs.py`'s
    `test_mutation_replacing_tell_with_stat_size_creates_a_history_follow_gap`.)

    F3 (review follow-up): a TRAILING PARTIAL line (the writer flushed up
    to some point inside the last line but not its trailing newline yet)
    must not be emitted here even if it happens to match `pattern` -- it
    is not a complete record yet. Read line-by-line with `readline()`
    (not `for line in f`, which gives no per-line boundary to roll back
    to) and keep `offset` at the position right BEFORE any such partial
    line, so the live follower re-reads it in full, from scratch, once it
    (and its newline) actually land -- never printed twice, never printed
    as an incomplete fragment.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            offset = f.tell()
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    # Trailing partial line: leave `offset` at the position
                    # before it (recorded on the previous iteration) and
                    # stop -- do not emit it, do not advance past it.
                    break
                if pattern.search(line):
                    _emit(tag, line)
                offset = f.tell()
            try:
                inode = os.fstat(f.fileno()).st_ino
            except OSError:
                inode = None
        return offset, inode
    except OSError:
        return 0, None


def _history_and_start_offsets(pattern) -> dict[Path, tuple[int, int | None]]:
    """Prints every matching line from each log's ROTATED `.1` file (oldest
    history, if present) and then its CURRENT file, in order -- then
    returns `{path: (byte_offset_at_end, inode_at_end)}` for the CURRENT
    file only, the exact boundary the live follower below starts from.
    Reading history and recording the boundary happen back to back per
    file (never "read every file's history, then come back for offsets"),
    so there is no window between them for a fast writer to slip through
    unaccounted for.
    """
    boundaries: dict[Path, tuple[int, int | None]] = {}
    for tag, name in _LOG_FILES:
        path = podenv.LOG_DIR / name
        rotated = path.with_name(name + ".1")
        if rotated.is_file():
            _read_history_file(rotated, tag, pattern)  # its own boundary is moot -- never followed
        boundaries[path] = _read_history_file(path, tag, pattern)
    return boundaries


def _follow_from_offsets(pattern, boundaries: dict[Path, tuple[int, int | None]]) -> None:
    """Polls each log file for bytes appended past its recorded offset,
    printing only lines matching `pattern` -- the live half of `--req`,
    starting EXACTLY where history left off (see the module docstring).
    Runs until interrupted (Ctrl-C / KeyboardInterrupt); the caller lets
    that propagate.

    Round-7 rewrite: keeps ONE open file handle per path for the whole
    follow loop, reopening it only once rotation has actually been
    confirmed and fully drained, rather than reopening by PATH every
    poll. This fixes three things at once:

    - **Gapless drain on rotation.** The previous version noticed a new
      inode at `path` and went straight to reading `<name>.1` by name to
      recover whatever the old generation still had unread. That is
      itself racy under back-to-back rotations (supervisord keeps only
      ONE backup generation -- if a SECOND rotation lands before we get
      around to reading `<name>.1`, the first generation's still-unread
      tail is overwritten and gone for good, unrecoverable no matter how
      the code is written). An open handle does not have this problem:
      on Linux, renaming or unlinking the path a handle was opened
      against never affects that handle's own ability to read whatever
      was already written to the underlying file -- so draining the OLD
      handle to its own true EOF (`f.read()` returning `""`) always
      recovers 100% of what that generation held, however many further
      rotations have since happened to the NAME.
    - **Trailing partial lines** (the writer has flushed up to some point
      inside a line but not its trailing `\\n` yet) are held back in
      `pending[path]` rather than emitted immediately and re-split next
      poll as a spurious "new" fragment.
    - **`progressed` only follows a successful read**, never set before
      one is attempted -- an open/read that fails no longer suppresses
      the poll-interval sleep, which used to spin a tight busy loop.
    """
    tags = {podenv.LOG_DIR / name: tag for tag, name in _LOG_FILES}
    pending: dict[Path, str] = {path: "" for path in boundaries}

    def _drain(path: Path, f, tag: str) -> bool:
        """Reads whatever is CURRENTLY available from an already-open
        handle (never reopens), matches `pattern` against complete
        lines, and holds back a trailing partial line in `pending`.
        Returns True if any bytes were actually read.
        """
        try:
            text = f.read()
        except OSError:
            return False
        if not text:
            return False
        combined = pending[path] + text
        lines = combined.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            pending[path] = lines.pop()
        else:
            pending[path] = ""
        for line in lines:
            if pattern.search(line):
                _emit(tag, line)
        return True

    handles: dict[Path, object] = {}
    for path, (off, ino) in boundaries.items():
        tag = tags[path]
        try:
            f = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            handles[path] = None
            continue
        try:
            cur_ino = os.fstat(f.fileno()).st_ino
        except OSError:
            cur_ino = None
        if ino is not None and cur_ino is not None and cur_ino != ino:
            # Rotated in the (tiny) window between the history read
            # finishing and this follower starting -- there is no open
            # handle spanning THAT particular gap (history's own handle
            # was already closed by then), so this one case still has to
            # recover the old generation's unread tail the way supervisord
            # names it: `<name>.1`. Every LATER rotation, once we hold an
            # open handle across it, is handled the gapless way instead
            # (see below).
            rotated_path = path.with_name(path.name + ".1")
            try:
                with rotated_path.open("r", encoding="utf-8", errors="replace") as rf:
                    rf.seek(off)
                    _drain(path, rf, tag)
            except OSError:
                pass
            # `f` is already the NEW file, already positioned at its own
            # start by the fresh open above -- nothing further to seek.
        else:
            try:
                f.seek(off)
            except OSError:
                pass
        handles[path] = f

    while True:
        progressed = False
        for path, f in list(handles.items()):
            tag = tags[path]
            if f is None:
                # No handle yet (file did not exist at follow-start, or
                # was closed after a prior rotation's new open failed) --
                # retry opening it fresh every poll until it exists.
                try:
                    new_f = path.open("r", encoding="utf-8", errors="replace")
                except OSError:
                    continue
                handles[path] = new_f
                progressed = True
                continue
            if _drain(path, f, tag):
                progressed = True
            # Truncated in place (same inode, just shorter) -- our fd's
            # position does not auto-adjust for that, so a plain
            # `read()` would sit at EOF forever without ever noticing.
            # Reopen at 0 (a fresh handle is simplest and correct: there
            # is nothing meaningful left to drain from the old position
            # once the file has been truncated out from under it).
            try:
                fd_size = os.fstat(f.fileno()).st_size
            except OSError:
                fd_size = None
            if fd_size is not None and f.tell() > fd_size:
                try:
                    f.close()
                except OSError:
                    pass
                pending[path] = ""
                try:
                    handles[path] = path.open("r", encoding="utf-8", errors="replace")
                    progressed = True
                except OSError:
                    handles[path] = None
                continue
            # Rotation check: compare the HANDLE's own inode (fstat on
            # the open fd -- unaffected by any rename of the path since
            # we opened it) against the path's CURRENT inode.
            try:
                fd_ino = os.fstat(f.fileno()).st_ino
            except OSError:
                fd_ino = None
            try:
                cur_ino = os.stat(path).st_ino
            except OSError:
                cur_ino = None
            if cur_ino is None or fd_ino == cur_ino:
                continue
            # The path now points to a DIFFERENT file than our handle --
            # drain the old handle the rest of the way to its own true
            # EOF (see the docstring above: this recovers everything
            # that generation ever held, regardless of what has since
            # happened to the NAME), then switch over.
            if _drain(path, f, tag):
                progressed = True
            try:
                f.close()
            except OSError:
                pass
            pending[path] = ""  # nothing more will ever complete a partial line from a closed generation
            try:
                handles[path] = path.open("r", encoding="utf-8", errors="replace")
                progressed = True
            except OSError:
                handles[path] = None
        if not progressed:
            time.sleep(_REQ_POLL_INTERVAL_S)


def _run_with_req_filter(req_id: str) -> int:
    try:
        pattern = common.req_filter_pattern(req_id)
    except common.InvalidReqId as e:
        print(f"zl logs: refusing -- {e}", file=sys.stderr)
        return 2
    boundaries = _history_and_start_offsets(pattern)
    try:
        _follow_from_offsets(pattern, boundaries)
    except KeyboardInterrupt:
        raise
    return 0  # unreachable in practice -- _follow_from_offsets only exits via KeyboardInterrupt


def run(argv: list[str]) -> int:
    script = podenv.scripts_dir() / "tail-logs.sh"
    if not script.is_file():
        print(f"zl logs: {script} not found (this checkout is missing scripts/)")
        return 1

    req_id = None
    rest: list[str] = []
    it = iter(argv)
    for a in it:
        if a == "--req":
            try:
                req_id = next(it)
            except StopIteration:
                print("zl logs: --req needs a value (a request id)", file=sys.stderr)
                return 2
        elif a.startswith("--req="):
            req_id = a.split("=", 1)[1]
        else:
            rest.append(a)

    if req_id is None:
        return common.run_stream(["bash", str(script), *rest])
    return _run_with_req_filter(req_id)
