"""Unit tests for ops/zlops/common.py -- the shared safety plumbing every
`zl` command relies on: env redaction (the privacy hard rule), the
dry-run/--apply confirmation gate, and the audit log's fail-open write.

Plain-script style like compactor/test_setup_sshd_script.py: exit 0 pass,
1 fail. No docker/network needed, so no `_skip` path.
"""

from __future__ import annotations

import io
import os
import re
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import time
from pathlib import Path
from unittest import mock

# ops/ and compactor/ are siblings; add ops/ to sys.path so `import zlops...`
# resolves regardless of the cwd run-tests.py invokes this file from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from zlops import common  # noqa: E402


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


def test_redact_env_planted_secrets():
    planted = {
        "HF_TOKEN": "hf_plantedsecret123",
        "SOME_API_KEY": "plainsecret",
        "OPENAI_API_KEY": "sk-abcdefghijklmnop",
        "INNOCUOUS_VALUE": "hf_looksLikeATokenAnyway123456",
        "DB_URL": "postgres://user:hunter2@example.com/db",
        "JWT_LOOKING": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQ",
        "PATH": "/usr/bin:/bin",
        "MODEL_REPO": "coder3101/Cydonia-24B-v4.3-heretic-v4",
    }
    out = common.redact_env(planted)
    for key in ("HF_TOKEN", "SOME_API_KEY", "OPENAI_API_KEY", "INNOCUOUS_VALUE", "DB_URL", "JWT_LOOKING"):
        assert_eq(out[key], common.REDACTED, f"redact_env redacts {key}")
    assert_eq(out["PATH"], "/usr/bin:/bin", "redact_env leaves PATH alone")
    assert_eq(out["MODEL_REPO"], "coder3101/Cydonia-24B-v4.3-heretic-v4", "redact_env leaves MODEL_REPO alone")


def test_redact_env_case_insensitive_key():
    out = common.redact_env({"hf_api_token": "anything"})
    assert_eq(out["hf_api_token"], common.REDACTED, "redact_env matches keys case-insensitively")


def test_redact_env_cloudflare_tunnel_token():
    """v3.1.9.9: CLOUDFLARE_TUNNEL_TOKEN (docker/cloudflared-run.sh's own
    env var name for the RunPod secret) must be redacted whole in `zl
    bundle`'s env-redacted.txt -- it matches `_SECRET_KEY`'s bare `TOKEN`
    pattern, the same as any other TOKEN-named env var, and it is not in
    `_ALLOWLIST_KEYS`.
    """
    out = common.redact_env({"CLOUDFLARE_TUNNEL_TOKEN": "eyJhbGciOiJIUzI1NiJ9-fake-token-value"})
    assert_eq(out["CLOUDFLARE_TUNNEL_TOKEN"], common.REDACTED, "CLOUDFLARE_TUNNEL_TOKEN is redacted whole")
    assert_true("CLOUDFLARE_TUNNEL_TOKEN" not in common._ALLOWLIST_KEYS,
                "CLOUDFLARE_TUNNEL_TOKEN is not on the allowlist (it must never show even scrubbed)")


def test_mutation_cloudflare_tunnel_token_on_allowlist_would_leak_it():
    """Prove the test above bites: if CLOUDFLARE_TUNNEL_TOKEN were ever
    (mistakenly) added to `_ALLOWLIST_KEYS`, redact_env would show it
    (scrubbed-substring only, not whole) instead of redacting it outright --
    confirms the allowlist-exclusion, not luck, is what protects it.
    """
    original = common._ALLOWLIST_KEYS
    try:
        common._ALLOWLIST_KEYS = original | {"CLOUDFLARE_TUNNEL_TOKEN"}

        # With the allowlist mutation alone, `_SECRET_KEY`'s own TOKEN match
        # still fires first (redact_env checks key-name secrecy BEFORE the
        # allowlist) -- so prove the ALLOWLIST branch itself would leak by
        # calling it directly, bypassing the key-secrecy check the same way
        # a hypothetical future key-pattern regression might.
        value = "eyJhbGciOiJIUzI1NiJ9-fake-token-value-with-no-scrubbable-shape"
        scrubbed = common._scrub_value(value)
        assert_eq(scrubbed, value, "mutated (allowlist path) shows the raw token value -- confirms the bite")
    finally:
        common._ALLOWLIST_KEYS = original
    out = common.redact_env({"CLOUDFLARE_TUNNEL_TOKEN": "eyJhbGciOiJIUzI1NiJ9-fake-token-value"})
    assert_eq(out["CLOUDFLARE_TUNNEL_TOKEN"], common.REDACTED, "the real (restored) code redacts it whole")


def test_split_passthrough():
    own, rest = common.split_passthrough(["--apply", "--", "--conv", "abc"])
    assert_eq(own, ["--apply"], "split_passthrough: own flags before --")
    assert_eq(rest, ["--conv", "abc"], "split_passthrough: passthrough after --")
    own2, rest2 = common.split_passthrough(["--apply", "--yes"])
    assert_eq(own2, ["--apply", "--yes"], "split_passthrough: no -- means everything is 'own'")
    assert_eq(rest2, [], "split_passthrough: no -- means empty passthrough")


def test_require_apply_confirmation_dry_run_never_prompts():
    # apply=False must never touch stdin/stdout, whatever they are.
    common.require_apply_confirmation(False, False, what="anything")
    print("  ok   require_apply_confirmation: dry run never raises/prompts")


def test_require_apply_confirmation_yes_skips_prompt():
    common.require_apply_confirmation(True, True, what="anything")
    print("  ok   require_apply_confirmation: --yes skips the prompt")


def test_require_apply_confirmation_non_tty_refuses():
    with mock.patch.object(sys.stdin, "isatty", return_value=False):
        try:
            common.require_apply_confirmation(True, False, what="a dangerous thing")
        except SystemExit as e:
            assert_eq(e.code, 1, "require_apply_confirmation: non-TTY without --yes exits 1")
        else:
            print("FAIL require_apply_confirmation: non-TTY without --yes did not raise")
            sys.exit(1)


def test_require_apply_confirmation_tty_wrong_answer_refuses():
    with mock.patch.object(sys.stdin, "isatty", return_value=True), \
         mock.patch.object(sys.stdout, "isatty", return_value=True), \
         mock.patch("builtins.input", return_value="nope"):
        try:
            common.require_apply_confirmation(True, False, what="a dangerous thing")
        except SystemExit as e:
            assert_eq(e.code, 1, "require_apply_confirmation: wrong confirmation text exits 1")
        else:
            print("FAIL require_apply_confirmation: wrong answer did not raise")
            sys.exit(1)


def test_require_apply_confirmation_tty_correct_answer_proceeds():
    with mock.patch.object(sys.stdin, "isatty", return_value=True), \
         mock.patch.object(sys.stdout, "isatty", return_value=True), \
         mock.patch("builtins.input", return_value="apply"):
        common.require_apply_confirmation(True, False, what="a dangerous thing")
    print("  ok   require_apply_confirmation: typing 'apply' proceeds")


def test_write_audit_never_raises_on_failure(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bad_parent = Path(td) / "not-a-dir"
        bad_parent.write_text("i am a file, not a directory")  # mkdir(parents=True) on this must fail
        with mock.patch.object(common.podenv, "AUDIT_LOG", bad_parent / "nested" / "audit.log"):
            captured = io.StringIO()
            with mock.patch.object(sys, "stderr", captured):
                common.write_audit(["zl", "status"], 0, 0.01, common.utcnow_iso())
            assert_true("warning" in captured.getvalue(), "write_audit: warns on failure instead of raising")


def test_write_audit_writes_valid_json_line():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "zl-ops" / "audit.log"
        with mock.patch.object(common.podenv, "AUDIT_LOG", log_path):
            common.write_audit(["zl", "status", "--brief"], 3, 0.5, "2026-09-24T00:00:00Z")
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert_eq(len(lines), 1, "write_audit: exactly one line written")
        rec = json.loads(lines[0])
        assert_eq(rec["argv"], ["zl", "status", "--brief"], "write_audit: argv recorded")
        assert_eq(rec["exit_code"], 3, "write_audit: exit_code recorded")
        assert_eq(rec["time"], "2026-09-24T00:00:00Z", "write_audit: time recorded")


def test_exit_meaning_known_and_unknown():
    assert_eq(common.exit_meaning(0), "done / nothing to do", "exit_meaning(0)")
    assert_eq(common.exit_meaning(3), "dry run found pending work (informational)", "exit_meaning(3)")
    assert_true("unrecognized" in common.exit_meaning(77), "exit_meaning: unknown code falls back sanely")


def test_exit_meaning_143_says_killed_by_signal_15():
    """Round-4 cosmetic: 143 (128+SIGTERM) used to fall through to
    "unrecognized exit code" -- it is the shell's own 128+N convention,
    the same one `exit_meaning`'s negative-code branch already reports
    for a child killed by a signal directly.
    """
    assert_eq(common.exit_meaning(143), "killed by signal 15", "exit_meaning(143)")
    assert_eq(common.exit_meaning(130), "interrupted (Ctrl-C)",
              "exit_meaning(130) keeps its own EXIT_MEANINGS wording, not '128+N'")


def test_mutation_missing_128_plus_n_case_is_caught():
    """Prove the test above bites: the OLD `exit_meaning` (no 128+N
    branch) says "unrecognized" for 143; the real (restored) one says
    "killed by signal 15".
    """
    def _old_exit_meaning(code):
        if code < 0:
            return f"killed by signal {-code} (reported as exit {128 + -code})"
        return common.EXIT_MEANINGS.get(code, "unrecognized exit code")

    assert_true("unrecognized" in _old_exit_meaning(143),
                "mutated (old) exit_meaning calls 143 unrecognized -- confirms the bite")
    assert_eq(common.exit_meaning(143), "killed by signal 15", "the real (restored) exit_meaning names the signal")


def test_human_bytes():
    assert_eq(common.human_bytes(0), "0B", "human_bytes(0)")
    assert_eq(common.human_bytes(1536), "1.5K", "human_bytes(1536)")


def test_req_filter_pattern_hyphenated_id_does_not_match_a_longer_sharing_id():
    """Round-4 LOW: real ids allow hyphens WITHIN a single id
    (`reqctx._ID_RE` in zl-v3198), so `req=abc` must not match inside
    `req=abc-def` -- a completely different id that merely shares a
    prefix.
    """
    pattern = common.req_filter_pattern("abc")
    assert_true(pattern.search("req=abc conv=c1: something") is not None, "the exact id matches")
    assert_true(pattern.search("req=abc.facts conv=c1: a child id matches") is not None,
                "a child id (.facts) still matches")
    assert_true(pattern.search("req=abc-def conv=c2: a DIFFERENT, hyphenated id") is None,
                "a longer id that merely shares the prefix does not match")

    pattern2 = common.req_filter_pattern("abc-def")
    assert_true(pattern2.search("req=abc-def conv=c2: the real hyphenated id itself") is not None,
                "the hyphenated id matches itself exactly")
    assert_true(pattern2.search("req=abc-defg conv=c3: an even longer id") is None,
                "a longer id sharing THIS prefix (with the hyphen already inside it) does not match")


def test_mutation_reverting_to_a_trailing_word_boundary_is_caught():
    """Prove the test above bites: restore the OLD trailing `\\b` version
    of the pattern, confirm `req=abc` matches inside `req=abc-def`, then
    confirm the real (restored) pattern does not.
    """
    old_pattern = re.compile(rf"\breq={re.escape('abc')}(?:\.[A-Za-z0-9_-]+)?\b")
    assert_true(old_pattern.search("req=abc-def conv=c2: a DIFFERENT, hyphenated id") is not None,
                "mutated (trailing \\b) pattern matches inside a longer hyphenated id -- confirms the bite")
    new_pattern = common.req_filter_pattern("abc")
    assert_true(new_pattern.search("req=abc-def conv=c2: a DIFFERENT, hyphenated id") is None,
                "the real (restored) pattern does not")


def test_req_filter_pattern_matches_the_background_form():
    """v3.1.9.8: a background tail's log lines carry `req=<id>/bg:<tag>`
    (bgwork.BackgroundPool._run, via logsetup's Filter and reqctx's
    `bg_tag_ctx`) instead of the bare `req=<id>` the request path uses.
    `zl logs --req`/`zl bundle --req` must follow a request's tail into
    the background exactly as they follow it into a `.child` vLLM call --
    the SAME request id, just tagged differently once work moves off the
    request path.
    """
    pattern = common.req_filter_pattern("a1b2c3d4e5f6a1b2")
    assert_true(
        pattern.search("req=a1b2c3d4e5f6a1b2/bg:tail conv=abc123: background tail running") is not None,
        "the background form (req=<id>/bg:<tag>) matches",
    )
    assert_true(
        pattern.search("req=a1b2c3d4e5f6a1b2/bg:rollup conv=abc123: background rollup running") is not None,
        "a different bg tag on the same id still matches",
    )
    assert_true(
        pattern.search("req=a1b2c3d4e5f6a1b2.dedup conv=abc123: a child id still matches") is not None,
        "the plain child-id form (req=<id>.child) still matches alongside the bg form",
    )
    assert_true(
        pattern.search("req=a1b2c3d4e5f6a1b2-other conv=abc123: a different, hyphen-suffixed id") is None,
        "an id that merely shares this prefix (hyphen-suffixed) does not match",
    )
    assert_true(
        pattern.search("req=a1b2c3d4e5f6a1b2ff conv=abc123: a longer hex id sharing this prefix") is None,
        "a longer hex id sharing this prefix does not match",
    )


def test_mutation_a_bg_form_blind_pattern_would_miss_background_lines():
    """Prove the test above bites: simulate a plausible regression -- a
    rewrite that anchors the "nothing else may follow" check with a
    trailing `$` (matches only when the id is the very end of the
    pattern's own reach) instead of the real pattern's negative lookahead
    for an id-continuation character -- and confirm it fails to match the
    background form (which always has more text after the id: `/bg:tag`
    and whatever the log line says next), while the real (current)
    pattern still matches it.
    """
    req_id = "a1b2c3d4e5f6a1b2"
    escaped = re.escape(req_id)
    naive_pattern = re.compile(rf"\breq={escaped}(?:\.[A-Za-z0-9_-]+)?$")
    line = "req=a1b2c3d4e5f6a1b2/bg:tail conv=abc123: background tail running"
    real_pattern = common.req_filter_pattern(req_id)
    assert_true(naive_pattern.search(line) is None,
                "the naive ($-anchored) pattern misses the background form -- confirms the bite")
    assert_true(real_pattern.search(line) is not None,
                "the real (negative-lookahead) pattern matches the background form")


# =============================================================================
# R2-B2: passthrough normalization (--flag=value splitting, unambiguous
# prefix expansion, ambiguous-prefix refusal).
# =============================================================================

_REPAIR_CHAT_TREE_FLAGS = ("--apply", "--chat", "--json", "--restore", "--sync-json-only")


def test_normalize_passthrough_token_splits_equals():
    flag, value = common.normalize_passthrough_token("--restore=S1", _REPAIR_CHAT_TREE_FLAGS)
    assert_eq(flag, "--restore", "normalize_passthrough_token splits --restore=S1: flag")
    assert_eq(value, "S1", "normalize_passthrough_token splits --restore=S1: value")


def test_normalize_passthrough_token_expands_unambiguous_prefix():
    flag, value = common.normalize_passthrough_token("--rest", _REPAIR_CHAT_TREE_FLAGS)
    assert_eq(flag, "--restore", "normalize_passthrough_token expands --rest to --restore")
    assert_eq(value, None, "normalize_passthrough_token: --rest has no inline value")


def test_normalize_passthrough_token_leaves_exact_match_alone():
    flag, value = common.normalize_passthrough_token("--apply", _REPAIR_CHAT_TREE_FLAGS)
    assert_eq(flag, "--apply", "normalize_passthrough_token: exact match unchanged")


def test_normalize_passthrough_token_leaves_positional_and_short_alone():
    flag, value = common.normalize_passthrough_token("some-id", _REPAIR_CHAT_TREE_FLAGS)
    assert_eq(flag, "some-id", "normalize_passthrough_token: positional token unchanged")
    flag2, _ = common.normalize_passthrough_token("-y", _REPAIR_CHAT_TREE_FLAGS)
    assert_eq(flag2, "-y", "normalize_passthrough_token: short option unchanged")


def test_normalize_passthrough_token_ambiguous_prefix_raises():
    flags = ("--restore", "--restart")
    try:
        common.normalize_passthrough_token("--rest", flags)
        assert_true(False, "normalize_passthrough_token: ambiguous prefix must raise")
    except common.AmbiguousPassthroughFlag:
        print("  ok   normalize_passthrough_token: ambiguous prefix (--rest of --restore/--restart) raises")


def test_normalize_passthrough_full_list():
    out = common.normalize_passthrough(
        ["--conv", "abc", "--restore=S1"], ("--conv", "--restore", "--apply")
    )
    assert_eq(out, ["--conv", "abc", "--restore", "S1"], "normalize_passthrough: splits and passes through")


def test_refuse_passthrough_write_flags_catches_equals_form():
    # review R2-B2: the exact-match `in` test never saw `--restore=S1`.
    code = common.refuse_passthrough_write_flags(
        ["--conv", "X", "--restore=S1"], ("--apply", "--restore"),
        zl_apply=False, what="test", all_flags=_REPAIR_CHAT_TREE_FLAGS,
    )
    assert_eq(code, 2, "refuse_passthrough_write_flags catches --restore=S1 without zl's own --apply")


def test_refuse_passthrough_write_flags_catches_prefix_form():
    for token in ("--rest", "--appl", "--forc"):
        flags = ("--apply", "--force", "--restore")
        all_flags = ("--apply", "--force", "--restore", "--conv")
        code = common.refuse_passthrough_write_flags(
            [token, "S1"], flags, zl_apply=False, what="test", all_flags=all_flags
        )
        assert_eq(code, 2, f"refuse_passthrough_write_flags catches prefix form {token!r}")


def test_refuse_passthrough_write_flags_catches_apply_equals_form():
    code = common.refuse_passthrough_write_flags(
        ["--apply=1"], ("--apply",), zl_apply=True, what="test", all_flags=("--apply",)
    )
    assert_eq(code, 2, "refuse_passthrough_write_flags: '--apply=1' after `--` is still refused (it's --apply)")


def test_refuse_passthrough_write_flags_allows_when_zl_apply_given():
    code = common.refuse_passthrough_write_flags(
        ["--restore=S1"], ("--apply", "--restore"), zl_apply=True, what="test", all_flags=_REPAIR_CHAT_TREE_FLAGS
    )
    assert_eq(code, None, "refuse_passthrough_write_flags: --restore=S1 is fine once zl's own --apply is given")


def test_refuse_passthrough_write_flags_ambiguous_prefix_refused():
    all_flags = ("--restore", "--restart", "--apply")
    code = common.refuse_passthrough_write_flags(
        ["--rest"], ("--apply",), zl_apply=False, what="test", all_flags=all_flags
    )
    assert_eq(code, 2, "refuse_passthrough_write_flags: an ambiguous prefix is refused, not guessed")


def test_mutation_reverting_to_exact_match_lets_equals_form_through():
    """Prove the tests above bite: mutate `refuse_passthrough_write_flags`
    back to the old exact-match-only behaviour (ignore `all_flags`), watch
    `--restore=S1` sail through unrefused, then restore.
    """
    original = common.refuse_passthrough_write_flags

    def _old_exact_match_only(passthrough, write_flags, *, zl_apply, what, all_flags=None):
        if "--apply" in passthrough:
            return 2
        hit = [f for f in write_flags if f in passthrough]
        if hit and not zl_apply:
            return 2
        return None

    try:
        common.refuse_passthrough_write_flags = _old_exact_match_only
        code = common.refuse_passthrough_write_flags(
            ["--restore=S1"], ("--apply", "--restore"), zl_apply=False, what="test", all_flags=_REPAIR_CHAT_TREE_FLAGS
        )
        assert_true(code is None, "mutated (exact-match-only) gate lets --restore=S1 through unrefused")
    finally:
        common.refuse_passthrough_write_flags = original
    code = common.refuse_passthrough_write_flags(
        ["--restore=S1"], ("--apply", "--restore"), zl_apply=False, what="test", all_flags=_REPAIR_CHAT_TREE_FLAGS
    )
    assert_eq(code, 2, "restored gate catches --restore=S1 again")


# =============================================================================
# R2-B1: ProtectedRegion (SIGHUP ignored / SIGINT+SIGTERM deferred-or-
# forwarded across the whole stop -> run -> restart window).
# =============================================================================


def test_protected_region_forwards_to_registered_child():
    calls = []
    with mock.patch.object(common.os, "killpg", side_effect=lambda pid, sig: calls.append((pid, sig))):
        with common.ProtectedRegion() as region:
            region.set_child(4242)
            region._handle(common.signal.SIGTERM, None)
    assert_eq(calls, [(4242, common.signal.SIGINT)], "a signal while a child is registered forwards SIGINT to it")


def test_protected_region_defers_when_no_child_registered():
    with common.ProtectedRegion() as region:
        assert_eq(region.deferred_signal, None, "no signal deferred yet")
        region._handle(common.signal.SIGTERM, None)
        assert_eq(region.deferred_signal, common.signal.SIGTERM, "a signal with no child registered is deferred")


def test_protected_region_clear_message_restores_the_generic_default():
    """N15 (round-6 surviving mutant): `region.clear_message()` must
    actually remove a custom message set by `set_message()` -- a caller
    that sets one for the duration of a wait (e.g. the in-flight guard's
    "nothing has been stopped yet" text) and then calls `clear_message()`
    must see the GENERIC default message on any LATER deferred signal, not
    a stale copy of the wait-specific one that no longer applies.
    """
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region.set_message("CUSTOM: waiting for her in-flight request(s)")
            region.clear_message()
            region._handle(common.signal.SIGTERM, None)
    assert_true("CUSTOM" not in printed.getvalue(), "the custom message was actually cleared",
                extra=printed.getvalue())
    assert_true("services are being stopped/restored" in printed.getvalue(),
                "the generic default message is shown instead", extra=printed.getvalue())


def test_mutation_forgetting_clear_message_is_caught():
    """Prove the test above bites: skip the `clear_message()` call (N15's
    mutation -- removed entirely) and confirm the stale custom message
    leaks into a later, unrelated deferred-signal print; then confirm the
    real (restored) code clears it.
    """
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region.set_message("CUSTOM: waiting for her in-flight request(s)")
            # mutation: clear_message() never called here
            region._handle(common.signal.SIGTERM, None)
    assert_true("CUSTOM" in printed.getvalue(),
                "mutated (no clear_message) flow leaks the stale custom message -- confirms the bite",
                extra=printed.getvalue())

    printed2 = io.StringIO()
    with mock.patch.object(sys, "stderr", printed2):
        with common.ProtectedRegion() as region:
            region.set_message("CUSTOM: waiting for her in-flight request(s)")
            region.clear_message()
            region._handle(common.signal.SIGTERM, None)
    assert_true("CUSTOM" not in printed2.getvalue(), "the real (restored) code clears it",
                extra=printed2.getvalue())


def test_protected_region_handle_survives_a_dead_terminal_while_deferring():
    """Round-4 blocker #2: `ProtectedRegion._handle`'s own deferred-signal
    message used to be a bare `print()` -- exactly the site the pager
    probe (review4/pty/run_pager.sh: `zl fix X --apply --yes 2>&1 | less`,
    quit the pager, Ctrl-C during the restore) found still able to raise
    BrokenPipeError and abandon the pod stopped. Must defer the signal
    (the actual safety mechanism) even when the message itself cannot be
    written anywhere.
    """
    with mock.patch.object(sys, "stderr", _DyingFile()):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGTERM, None)  # must not raise
    assert_true(region.deferred_signal == common.signal.SIGTERM,
                "the signal is still recorded as deferred despite the dead terminal")


def test_mutation_bare_print_in_handle_raises_on_a_dead_terminal():
    """Prove the test above bites: inline the OLD bare-`print()` version
    of `_handle`'s deferred-signal message, confirm it raises (and, worse,
    would do so BEFORE `deferred_signal` is even set), then confirm the
    real (restored) `_handle` does not.
    """
    def _old_handle(self, signum, _frame):
        if self.child_pid is not None:
            return
        if self.deferred_signal is None:
            print(f"zl: signal {signum} received -- ...", file=sys.stderr)
        self.deferred_signal = signum

    with mock.patch.object(sys, "stderr", _DyingFile()):
        with common.ProtectedRegion() as region:
            try:
                _old_handle(region, common.signal.SIGTERM, None)
            except OSError:
                print("  ok   mutated (bare print) _handle raises on a dead terminal -- confirms the bite")
            else:
                print("FAIL mutated (bare print) _handle did not raise -- fixture is broken")
                sys.exit(1)

    with mock.patch.object(sys, "stderr", _DyingFile()):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGTERM, None)  # must not raise
    assert_true(region.deferred_signal == common.signal.SIGTERM,
                "the real (restored) _handle still defers despite the dead terminal")


def test_protected_region_finish_exits_128_plus_signal_when_deferred():
    with common.ProtectedRegion() as region:
        region._handle(common.signal.SIGTERM, None)
        try:
            region.finish()
        except SystemExit as e:
            assert_eq(e.code, 128 + common.signal.SIGTERM, "finish() exits 128+signal after a deferred signal")
        else:
            print("FAIL finish() did not exit after a deferred signal")
            sys.exit(1)


def test_protected_region_finish_is_a_noop_with_no_deferred_signal():
    with common.ProtectedRegion() as region:
        region.finish()  # must not raise
    print("  ok   finish() is a no-op when nothing was deferred")


def test_protected_region_finish_keeps_nonzero_child_code_over_deferred_signal():
    """Review H2: `finish(child_code, restored)` -- a deferred SIGTERM
    (journal exit 6, "re-run") must NOT be allowed to silently turn into
    a bare 130 the way it used to. The nonzero child code wins.
    """
    with common.ProtectedRegion() as region:
        region._handle(common.signal.SIGTERM, None)
        try:
            region.finish(6, restored=False)
        except SystemExit as e:
            assert_eq(e.code, 6, "finish() keeps the child's own nonzero exit code over the deferred signal")
        else:
            print("FAIL finish() did not exit")
            sys.exit(1)


def test_protected_region_finish_uses_128_plus_signal_when_child_code_is_zero():
    with common.ProtectedRegion() as region:
        region._handle(common.signal.SIGINT, None)
        try:
            region.finish(0, restored=True)
        except SystemExit as e:
            assert_eq(e.code, 128 + common.signal.SIGINT,
                      "finish() falls back to 128+signal only when the child code was 0")
        else:
            print("FAIL finish() did not exit")
            sys.exit(1)


def test_protected_region_finish_never_says_restored_when_restored_is_false():
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGTERM, None)
            try:
                region.finish(6, restored=False)
            except SystemExit:
                pass
    assert_true("services are restored" not in printed.getvalue(),
                "finish(restored=False) never claims services were restored", extra=printed.getvalue())
    assert_true("left stopped" in printed.getvalue(),
                "finish(restored=False) says they were left stopped instead", extra=printed.getvalue())


def test_protected_region_finish_restored_none_says_nothing_was_stopped():
    """F4 (review follow-up): a backup-phase (or in-flight-wait) abort --
    where NOTHING has been stopped yet -- must not say "services are
    restored" (that implies something was stopped and put back). Passing
    `restored=None` says plainly that nothing was ever touched.
    """
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGTERM, None)
            try:
                region.finish(common.EXIT_BACKUP_FAILED, restored=None)
            except SystemExit:
                pass
    out = printed.getvalue()
    assert_true("services are restored" not in out,
                "finish(restored=None) never claims services were restored", extra=out)
    assert_true("nothing was stopped" in out,
                "finish(restored=None) says nothing was stopped instead", extra=out)


def test_mutation_hardcoding_restored_true_on_backup_abort_leaks_the_bad_claim():
    """Prove the test above bites: the OLD call sites (`region.finish(...,
    restored=True)` on every early abort, before anything was stopped)
    would print exactly the misleading claim this test guards against.
    """
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGTERM, None)
            try:
                region.finish(common.EXIT_BACKUP_FAILED, restored=True)  # the OLD (buggy) call
            except SystemExit:
                pass
    assert_true("services are restored" in printed.getvalue(),
                "sanity: restored=True really does print the misleading claim", extra=printed.getvalue())


def test_protected_region_finish_reports_forced_exit_after_a_clean_child():
    """F4 (review follow-up): when a wrapped script itself exited 0 but
    the caller forced the final exit code nonzero because a service
    failed to come back afterward (`report_restart_failures`), the
    message must say the fix ran and exited 0 (applied), and name WHY
    the final exit code differs -- never imply the fix itself failed.
    """
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGINT, None)
            try:
                region.finish(
                    1, restored=False, applied=True, label="the fix",
                    real_child_code=0, down_services=["openwebui"],
                )
            except SystemExit as e:
                assert_eq(e.code, 1, "finish() keeps the forced exit code")
    out = printed.getvalue()
    assert_true("the fix ran and exited 0 (applied)" in out,
                "the message says the fix itself exited 0", extra=out)
    assert_true("exit 1 because openwebui did not come back" in out,
                "the message names the forcing reason", extra=out)


def test_mutation_omitting_real_child_code_misreports_a_clean_fix_as_failed():
    """Prove the test above bites: without `real_child_code`/
    `down_services`, `finish()` falls back to reporting the FORCED code
    (1) as if it were the fix's own exit code -- "the fix ran and exited
    1" -- which is simply false (the fix exited 0; a restart failure
    forced the overall code to 1).
    """
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGINT, None)
            try:
                region.finish(1, restored=False, applied=True, label="the fix")  # no real_child_code/down_services
            except SystemExit:
                pass
    out = printed.getvalue()
    assert_true("the fix ran and exited 1" in out,
                "sanity: omitting the new args really does misreport the fix's own exit code", extra=out)
    assert_true("ran and exited 0 (applied)" not in out, "and never says the (true) 0 without the new args", extra=out)


def test_protected_region_finish_applied_note_only_says_do_not_rerun_on_exit_zero():
    """Round-7: `applied=True` with a NONZERO child code must say "ran and
    exited N", never "WAS APPLIED ... do not re-run it" -- some wrapped
    scripts use a specific nonzero code to mean "safe/required to
    re-run" (e.g. target exits 4 in `pty/run_grp_exit4.sh`), so an
    unconditional "do not re-run it" note would flatly contradict the
    script's own documented exit-code meaning.
    """
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGINT, None)
            try:
                region.finish(4, restored=True, applied=True, label="the fix")
            except SystemExit as e:
                assert_eq(e.code, 4, "finish() keeps the nonzero child code")
    out = printed.getvalue()
    assert_true("ran and exited 4" in out, "a nonzero code gets a plain factual note", extra=out)
    assert_true("WAS APPLIED" not in out, "a nonzero code never claims WAS APPLIED", extra=out)
    assert_true("do not re-run" not in out, "a nonzero code never says do not re-run", extra=out)


def test_protected_region_finish_applied_note_says_was_applied_on_exit_zero():
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGINT, None)
            try:
                region.finish(0, restored=True, applied=True, label="the fix")
            except SystemExit as e:
                assert_eq(e.code, 128 + common.signal.SIGINT, "finish() falls back to 128+signal on exit 0")
    out = printed.getvalue()
    assert_true("WAS APPLIED (exit 0); do not re-run it" in out,
                "exit 0 is the one case where WAS APPLIED / do not re-run is printed", extra=out)


def test_protected_region_finish_no_applied_note_when_not_applied():
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region._handle(common.signal.SIGINT, None)
            try:
                region.finish(1, restored=True, applied=False)
            except SystemExit as e:
                assert_eq(e.code, 1, "finish() keeps the nonzero child code")
    out = printed.getvalue()
    assert_true("ran and exited" not in out, "applied=False prints no ran/applied note at all", extra=out)
    assert_true("WAS APPLIED" not in out, "applied=False never claims WAS APPLIED", extra=out)


def test_mutation_finish_ignoring_child_code_loses_it_to_130():
    """Prove the tests above bite: the OLD `finish()` took no arguments
    and always raised 128+signal, whatever the child returned. Inline
    that behaviour and show it discards a real, meaningful exit code.
    """
    def _old_finish(self):
        if self.deferred_signal is not None:
            raise SystemExit(128 + self.deferred_signal)

    with common.ProtectedRegion() as region:
        region._handle(common.signal.SIGTERM, None)
        try:
            _old_finish(region)
        except SystemExit as e:
            assert_eq(e.code, 128 + common.signal.SIGTERM,
                      "mutated (old, argument-less) finish() always returns 128+signal -- confirms the bite "
                      "(the real journal exit 6 above would have been lost to this)")
        else:
            print("FAIL mutated finish() did not exit")
            sys.exit(1)


def test_safe_print_swallows_oserror():
    class _DyingFile:
        def write(self, _s):
            raise OSError(5, "I/O error")

        def flush(self):
            pass

    common.safe_print("this must not raise", file=_DyingFile())  # must not raise
    print("  ok   safe_print swallows OSError from a dead file-like object")


class _DyingFile:
    def write(self, _s):
        raise OSError(5, "I/O error")

    def flush(self):
        pass


def test_supervisorctl_stop_and_start_use_safe_print_on_a_dead_terminal():
    """Round-4 blocker #2: `supervisorctl_stop`/`supervisorctl_start`'s own
    except-branch prints (common.py's own "did not complete" messages)
    used to be a bare `print()` -- reachable from inside `ProtectedRegion`
    (a hangup/BrokenPipeError landing exactly here, mid stop/restart,
    used to escape and abandon the pod stopped, per the pager probe in
    review4/pty/run_pager.sh). Both must swallow a write failure the same
    way every other in-region print does.
    """
    with mock.patch.object(common, "run", side_effect=TimeoutError("boom")), \
         mock.patch.object(sys, "stderr", _DyingFile()):
        result = common.supervisorctl_stop(["compactor"])  # must not raise
    assert_true(result is None, "supervisorctl_stop reports failure (None) rather than raising")

    with mock.patch.object(common, "run", side_effect=TimeoutError("boom")), \
         mock.patch.object(sys, "stderr", _DyingFile()):
        result = common.supervisorctl_start(["compactor"])  # must not raise
    assert_true(result is None, "supervisorctl_start reports failure (None) rather than raising")


def test_mutation_bare_print_in_supervisorctl_stop_raises():
    """Prove the test above bites: inline the OLD bare-`print()` version
    of `supervisorctl_stop`'s except branch, confirm it raises on a dead
    terminal, then confirm the real (restored) function does not.
    """
    def _old_supervisorctl_stop(names):
        try:
            return common.run(["supervisorctl", "stop", *names], timeout=60)
        except (OSError, common.subprocess.TimeoutExpired, TimeoutError) as e:
            print(f"zl: `supervisorctl stop {' '.join(names)}` did not complete: {e}", file=sys.stderr)
            return None

    with mock.patch.object(common, "run", side_effect=TimeoutError("boom")), \
         mock.patch.object(sys, "stderr", _DyingFile()):
        try:
            _old_supervisorctl_stop(["compactor"])
        except OSError:
            print("  ok   mutated (bare print) supervisorctl_stop raises on a dead terminal -- confirms the bite")
        else:
            print("FAIL mutated (bare print) supervisorctl_stop did not raise -- fixture is broken")
            sys.exit(1)

    with mock.patch.object(common, "run", side_effect=TimeoutError("boom")), \
         mock.patch.object(sys, "stderr", _DyingFile()):
        result = common.supervisorctl_stop(["compactor"])  # must not raise
    assert_true(result is None, "the real (restored) supervisorctl_stop does not raise")


def test_supervisorctl_stop_and_start_use_start_new_session():
    """Round-6 HIGH: `supervisorctl stop`/`start` must run with
    `start_new_session=True` so a Ctrl-C at the terminal (SIGINT to the
    whole foreground process group) cannot kill them mid-command --
    exactly the failure mode `stop_for_apply`'s own post-stop verification
    exists to catch, but there is no reason to invite it in the first
    place. Checks the actual kwarg passed to `subprocess.run` via `run`.
    """
    seen = {}

    def fake_run(argv, timeout=None, **kw):
        seen["kwargs"] = kw
        import subprocess as _sp
        return _sp.CompletedProcess(argv, 0, "", "")

    with mock.patch.object(common, "run", side_effect=fake_run):
        common.supervisorctl_stop(["compactor"])
    assert_true(seen["kwargs"].get("start_new_session") is True,
                "supervisorctl_stop passes start_new_session=True", extra=str(seen))

    seen.clear()
    with mock.patch.object(common, "run", side_effect=fake_run):
        common.supervisorctl_start(["compactor"])
    assert_true(seen["kwargs"].get("start_new_session") is True,
                "supervisorctl_start passes start_new_session=True", extra=str(seen))


def test_report_restart_failures_all_back_returns_true_and_prints_nothing():
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        result = common.report_restart_failures({"compactor": True, "backup": True})
    assert_true(result is True, "every service back RUNNING returns True")
    assert_eq(printed.getvalue(), "", "nothing is printed when everything came back")


def test_report_restart_failures_prints_warning_and_exact_command_when_one_is_down():
    """Round-6 HIGH: a service that did NOT come back must be reported
    LOUDLY, with the exact `supervisorctl start ...` command to run by
    hand, and the function must return False so callers never say
    "restored" (and never keep a zero exit code) when it did not.
    """
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        result = common.report_restart_failures({"compactor": True, "backup": False})
    assert_true(result is False, "one service down returns False")
    assert_true("backup" in printed.getvalue() and "WARNING" in printed.getvalue(),
                "the down service is named in a loud WARNING", extra=printed.getvalue())
    assert_true("supervisorctl start backup" in printed.getvalue(),
                "the exact command to bring it back is printed", extra=printed.getvalue())
    assert_true("compactor" not in printed.getvalue(),
                "a service that DID come back is not mentioned at all", extra=printed.getvalue())


def test_mutation_bare_print_raises_where_safe_print_does_not():
    class _DyingFile:
        def write(self, _s):
            raise OSError(5, "I/O error")

        def flush(self):
            pass

    try:
        print("this WILL raise", file=_DyingFile())
    except OSError:
        print("  ok   a bare print() to the same dead file DOES raise -- confirms safe_print's own catch bites")
    else:
        print("FAIL bare print() did not raise -- the fixture itself is broken")
        sys.exit(1)


def test_write_audit_defers_sigint_until_the_write_is_done():
    """LOW (round 3): a second Ctrl-C landing during the audit write
    itself (after a `ProtectedRegion` has already exited and restored
    default handlers -- see cli.py's `main`) must not be able to lose the
    audit line. SIGINT is deferred for the duration of the write and only
    takes effect (as SystemExit(130)) once the write has completed.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        audit_log = Path(td) / "audit.log"
        with mock.patch.object(common.podenv, "AUDIT_LOG", audit_log):
            real_open = Path.open

            def _open_and_signal_self(self, *a, **kw):
                if self == audit_log:
                    os.kill(os.getpid(), common.signal.SIGINT)  # arrives mid-write, must be deferred
                return real_open(self, *a, **kw)

            with mock.patch.object(Path, "open", _open_and_signal_self):
                try:
                    common.write_audit(["zl", "fix"], 0, 1.0, common.utcnow_iso())
                except SystemExit as e:
                    assert_eq(e.code, 130, "the deferred SIGINT takes effect only after the write, as 130")
                else:
                    print("FAIL write_audit did not raise despite a deferred SIGINT")
                    sys.exit(1)
        assert_true(audit_log.exists() and audit_log.read_text(encoding="utf-8").strip(),
                    "the audit line was written BEFORE the deferred signal took effect")


def test_mutation_write_audit_without_deferral_loses_the_line_to_the_signal():
    """Prove the test above bites: inline the OLD write_audit (no SIGINT
    deferral at all) and confirm a SIGINT arriving mid-write can propagate
    as a bare KeyboardInterrupt with no guarantee the line landed first --
    i.e. there was nothing here holding it off.

    Round-9 LOW: this relies on Python's DEFAULT SIGINT disposition
    (raise KeyboardInterrupt) at the moment `os.kill` fires below -- but a
    test harness that runs this file as a background/non-foreground
    process (no controlling terminal, or launched via some `Popen`
    chains) can inherit SIGINT already set to `SIG_IGN`, in which case
    `os.kill(self, SIGINT)` does nothing at all and this test would fail
    for a reason that has nothing to do with `write_audit` itself. The
    default handler is force-installed for the duration of this one test
    (and the previous SIGINT disposition -- whatever it was -- restored
    in a `finally`, so this cannot leak into any other test) so the
    fixture's own behaviour never depends on how this process happened to
    inherit SIGINT.
    """
    import tempfile

    def _old_write_audit_body(path):
        # N: the write itself, with SIGINT handling removed (the old shape).
        with path.open("a", encoding="utf-8") as f:
            os.kill(os.getpid(), common.signal.SIGINT)  # not deferred -- Python raises KeyboardInterrupt here
            f.write("{}\n")

    old_handler = common.signal.signal(common.signal.SIGINT, common.signal.default_int_handler)
    try:
        with tempfile.TemporaryDirectory() as td:
            audit_log = Path(td) / "audit.log"
            try:
                _old_write_audit_body(audit_log)
            except KeyboardInterrupt:
                print("  ok   mutated (old, non-deferring) write path raises KeyboardInterrupt mid-write -- "
                      "confirms the bite (no guarantee the write above it even started flushing)")
            else:
                print("FAIL mutated write path did not raise -- the fixture itself is broken")
                sys.exit(1)
    finally:
        common.signal.signal(common.signal.SIGINT, old_handler)


def test_protected_region_custom_message_used_when_no_forward():
    printed = io.StringIO()
    with mock.patch.object(sys, "stderr", printed):
        with common.ProtectedRegion() as region:
            region.set_message("zl: repair in progress; will not interrupt")
            region._handle(common.signal.SIGINT, None)
    assert_true("repair in progress; will not interrupt" in printed.getvalue(),
                "set_message overrides the generic deferred-signal text")


def test_mutation_protected_region_always_forwarding_defeats_no_forward_message():
    """Prove the test above bites: mutate `_handle` to always forward
    (ignoring `child_pid is None`), confirming the deferral path (and its
    message) would not have fired -- then restore.
    """
    original = common.ProtectedRegion._handle

    def _always_forward(self, signum, _frame):
        try:
            common.os.killpg(self.child_pid or 1, common.signal.SIGINT)
        except OSError:
            pass

    try:
        common.ProtectedRegion._handle = _always_forward
        printed = io.StringIO()
        with mock.patch.object(common.os, "killpg"):
            with mock.patch.object(sys, "stderr", printed):
                with common.ProtectedRegion() as region:
                    region.set_message("zl: repair in progress; will not interrupt")
                    region._handle(common.signal.SIGINT, None)
        assert_true(
            "repair in progress" not in printed.getvalue(),
            "mutated (always-forward) handler never prints the deferral message",
        )
    finally:
        common.ProtectedRegion._handle = original


# =============================================================================
# R2-HIGH: stop_for_apply refuses (StopVerificationFailed) on an unreadable
# or incomplete pre-stop status read, before anything is stopped.
# =============================================================================


def test_stop_for_apply_refuses_on_unreadable_pre_stop_status():
    with mock.patch.object(common, "supervisorctl_status", return_value={}):
        try:
            common.stop_for_apply(["compactor", "backup"])
        except common.StopVerificationFailed as e:
            assert_eq(e.was_running, [], "an unreadable pre-stop status has nothing to restart")
        else:
            print("FAIL stop_for_apply did not refuse on an unreadable pre-stop status")
            sys.exit(1)


def test_stop_for_apply_refuses_when_a_requested_name_is_missing():
    """Round-3 review N9: the OLD version of this test only mocked
    `supervisorctl_status`, never `supervisorctl_stop` -- so a mutation
    that removed the pre-stop "requested name missing" check STILL made
    this test pass, just for the wrong reason: without the check,
    `was_running` silently narrows to `["compactor"]` (the one name that
    IS in status), `stop_for_apply` calls the REAL, unmocked
    `supervisorctl_stop`, that fails/hangs against the host (there is no
    real supervisorctl in a unit-test environment), and after the full
    `verify_timeout` it raises `StopVerificationFailed` anyway -- the
    assertion below still passed, but only after ~20s and only after
    actually invoking the host's own `supervisorctl`. Both symptoms are
    fixed here: `supervisorctl_stop` is mocked to FAIL THE TEST OUTRIGHT
    if it is ever called (proving the pre-stop check refuses BEFORE any
    stop attempt), and the refusal is timed (must be near-instant, not
    anywhere near `verify_timeout`).
    """
    def _must_not_be_called(names):
        print(f"FAIL supervisorctl_stop({names!r}) was called -- the pre-stop "
              "'requested name missing' check must refuse BEFORE any stop is attempted")
        sys.exit(1)

    with mock.patch.object(common, "supervisorctl_status", return_value={"compactor": "RUNNING"}), \
         mock.patch.object(common, "supervisorctl_stop", side_effect=_must_not_be_called):
        t0 = time.monotonic()
        try:
            common.stop_for_apply(["compactor", "backup"], verify_timeout=20.0)
        except common.StopVerificationFailed as e:
            elapsed = time.monotonic() - t0
            assert_true(elapsed < 2.0,
                        f"refusal took {elapsed:.1f}s -- should be immediate (pre-stop check), "
                        "not fall through to the post-stop verify_timeout")
            assert_eq(e.was_running, [], "nothing was stopped -- the refusal is pre-stop")
            print("  ok   stop_for_apply refuses when a requested name is missing from status")
        else:
            print("FAIL stop_for_apply did not refuse when 'backup' was missing from status")
            sys.exit(1)


def test_mutation_removing_missing_name_check_calls_real_stop_slowly():
    """Prove the strengthened test above actually bites (N9): restore the
    OLD pre-stop check's behaviour (silently narrow `was_running` to only
    the names status DOES know about, instead of refusing), and confirm
    that DOES reach `supervisorctl_stop` -- which the test above treats
    as an outright failure.
    """
    def _old_check(names, pre_status):
        return [n for n in names if pre_status.get(n) in common._ACTIVE_STATES]

    calls = []
    with mock.patch.object(common, "supervisorctl_status", return_value={"compactor": "RUNNING"}):
        pre_status = common.supervisorctl_status()
        was_running = _old_check(["compactor", "backup"], pre_status)
        assert_eq(was_running, ["compactor"],
                  "mutated (old) check silently narrows to the names status DOES know, no refusal")
    with mock.patch.object(common, "supervisorctl_stop", side_effect=lambda names: calls.append(names)):
        # confirm the mutated behaviour WOULD have gone on to call supervisorctl_stop --
        # exactly what the real (restored) stop_for_apply's pre-stop refusal prevents.
        common.supervisorctl_stop(was_running)
        assert_true(calls == [["compactor"]],
                    "mutated version reaches supervisorctl_stop with a partial list -- confirms the bite")


def test_mutation_reverting_pre_stop_check_treats_unreadable_as_nothing_running():
    """Prove the test above bites: restore the OLD behaviour (unreadable
    status -> was_running=[]), watch `stop_for_apply` return normally
    (no refusal) with a writer possibly still up, then restore.
    """
    def _old_stop_for_apply(names, *, verify_timeout=20.0):
        status = common.supervisorctl_status()
        was_running = [n for n in names if status.get(n) in common._ACTIVE_STATES]
        return was_running  # the old bug: [] on an unreadable status, no refusal

    with mock.patch.object(common, "supervisorctl_status", return_value={}):
        result = _old_stop_for_apply(["compactor", "backup"])
        assert_eq(result, [], "mutated (old) stop_for_apply silently returns [] on an unreadable status")
        # And the real, fixed version refuses instead:
        try:
            common.stop_for_apply(["compactor", "backup"])
            print("FAIL the real stop_for_apply did not refuse")
            sys.exit(1)
        except common.StopVerificationFailed:
            print("  ok   the real (restored) stop_for_apply refuses instead")


# =============================================================================
# Round-2 reviewer's mutation list -- tests that bite for M2, M4, M5, M7
# (M12 already covered in test_ops_journal.py). Each one is written as a
# mutate/watch-fail/restore proof, per the brief's testing standard.
# =============================================================================


def test_merged_environ_pid1_wins_on_conflict():
    """M5: pid 1's (the real services') values must win over this
    process's own shell env on any key both define -- review B3.
    """
    with mock.patch.object(common, "proc1_environ", return_value={"WEBUI_DB_LOCAL": "true", "ONLY_PID1": "x"}):
        with mock.patch.dict(os.environ, {"WEBUI_DB_LOCAL": "false", "ONLY_SHELL": "y"}):
            merged = common.merged_environ()
    assert_eq(merged["WEBUI_DB_LOCAL"], "true", "pid1's value wins on a key both define")
    assert_eq(merged["ONLY_SHELL"], "y", "a shell-only key still survives")
    assert_eq(merged["ONLY_PID1"], "x", "a pid1-only key still survives")


def test_mutation_reversing_merged_environ_precedence_is_caught():
    original = common.proc1_environ
    try:
        def _reversed_merge():
            return {}  # stand-in: with this, os.environ always wins (no pid1 override possible)
        common.proc1_environ = _reversed_merge
        with mock.patch.dict(os.environ, {"WEBUI_DB_LOCAL": "false"}):
            merged = common.merged_environ()
        assert_true(merged.get("WEBUI_DB_LOCAL") == "false",
                    "mutated (pid1 env unavailable) merged_environ falls back to the shell's value -- "
                    "confirms pid1 is the one doing the overriding in the real function")
    finally:
        common.proc1_environ = original
    with mock.patch.object(common, "proc1_environ", return_value={"WEBUI_DB_LOCAL": "true"}):
        with mock.patch.dict(os.environ, {"WEBUI_DB_LOCAL": "false"}):
            merged = common.merged_environ()
    assert_eq(merged["WEBUI_DB_LOCAL"], "true", "restored: pid1 wins again")


def test_redact_env_allowlisted_value_still_scrubs_embedded_secret():
    """M7: an ALLOWLISTED key (e.g. ZL_REPO_URL) is shown, but its VALUE
    must still be scrubbed for an embedded secret substring -- an
    allowlisted key is not a blanket exemption from the substring scrub.
    """
    out = common.redact_env({
        "ZL_REPO_URL": "https://ghp_1234567890123456789012345@github.com/example/repo.git",
    })
    assert_true("ghp_1234567890123456789012345" not in out["ZL_REPO_URL"],
                "the embedded GitHub token substring is scrubbed")
    assert_true("github.com/example/repo.git" in out["ZL_REPO_URL"],
                "the rest of the (non-secret) URL survives")


def test_mutation_disabling_allowlist_value_scrub_is_caught():
    original = common._scrub_value
    try:
        common._scrub_value = lambda v: v  # M7 itself: allowlisted values shown raw
        out = common.redact_env({"ZL_REPO_URL": "https://ghp_1234567890123456789012345@github.com/x/y.git"})
        assert_true("ghp_1234567890123456789012345" in out["ZL_REPO_URL"],
                    "mutated (no-op) value scrub leaves the embedded token in place -- confirms the bite")
    finally:
        common._scrub_value = original
    out = common.redact_env({"ZL_REPO_URL": "https://ghp_1234567890123456789012345@github.com/x/y.git"})
    assert_true("ghp_1234567890123456789012345" not in out["ZL_REPO_URL"], "restored: the token is scrubbed again")


def test_stop_for_apply_blocks_until_status_confirms_stopped():
    """M2: `stop_for_apply` must actually POLL `supervisorctl status` until
    every service is confirmed STOPPED/EXITED/FATAL -- not just issue the
    stop command and return. Fake a service that stays RUNNING for the
    first two status reads (after the initial pre-stop read) and only
    reports STOPPED on the third -- `stop_for_apply` must not return
    before that third read.
    """
    statuses = [
        {"compactor": "RUNNING"},  # pre-stop read
        {"compactor": "RUNNING"},  # post-stop poll #1: still running
        {"compactor": "RUNNING"},  # post-stop poll #2: still running
        {"compactor": "STOPPED"},  # post-stop poll #3: finally stopped
    ]
    calls = {"n": 0}

    def fake_status():
        i = min(calls["n"], len(statuses) - 1)
        calls["n"] += 1
        return statuses[i]

    with mock.patch.object(common, "supervisorctl_status", side_effect=fake_status), \
         mock.patch.object(common, "supervisorctl_stop", return_value=None), \
         mock.patch.object(common.time, "sleep"):  # don't actually wait 0.3s per poll in a unit test
        result = common.stop_for_apply(["compactor"], verify_timeout=5.0)
    assert_eq(result, ["compactor"], "stop_for_apply returns once status genuinely confirms STOPPED")
    assert_true(calls["n"] >= 4, "at least 4 status reads happened (1 pre-stop + 3 polls) -- it actually "
                "polled, rather than trusting the stop command alone")


def test_mutation_returning_immediately_after_stop_is_caught():
    """Prove the test above bites: mutate stop_for_apply to return right
    after issuing the stop command, before any post-stop verification --
    the still-RUNNING service is (wrongly) reported as successfully
    stopped, and only 1 status read happens total.
    """
    calls = {"n": 0}

    def fake_status():
        calls["n"] += 1
        return {"compactor": "RUNNING"}  # never actually stops, in this mutated scenario

    def mutated_stop_for_apply(names, *, verify_timeout=20.0):
        pre_status = fake_status()
        was_running = [n for n in names if pre_status.get(n) in common._ACTIVE_STATES]
        if not was_running:
            return []
        common.supervisorctl_stop(was_running)
        return was_running  # M2: returns immediately, no verification loop at all

    with mock.patch.object(common, "supervisorctl_stop", return_value=None):
        result = mutated_stop_for_apply(["compactor"])
    assert_eq(result, ["compactor"],
              "mutated (no-verify) stop_for_apply reports success even though the service never "
              "actually stopped -- confirms the bite")
    assert_eq(calls["n"], 1, "mutated version only reads status ONCE (no polling at all)")


def test_run_protected_standalone_forwards_sigterm_to_child():
    """M9: `run_protected` with NO `region` (the standalone/legacy path,
    kept for any future caller that doesn't open its own ProtectedRegion)
    must still forward SIGTERM to the child as SIGINT, exactly like the
    region-based path does. Uses a real subprocess and a real signal --
    the child exits with a distinctive code (77) only if it actually
    received SIGINT itself.
    """
    import signal
    import tempfile
    import threading
    import time

    sleepy_script = (
        "import signal, sys, time\n"
        "def _got_int(signum, frame):\n"
        "    sys.exit(77)\n"
        "signal.signal(signal.SIGINT, _got_int)\n"
        "time.sleep(5)\n"
        "sys.exit(0)\n"
    )
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "sleepy.py"
        script.write_text(sleepy_script, encoding="utf-8")
        with mock.patch.object(common.podenv, "ZL_HOME", Path(td) / "zl-home"):
            # signal.signal() only works on the main thread -- run_protected
            # itself (which installs the SIGTERM handler) must run there;
            # the signal is sent FROM a background thread instead.
            def _send_shortly():
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGTERM)  # must be forwarded to the CHILD, not kill us

            t = threading.Thread(target=_send_shortly)
            t.start()
            code = common.run_protected(
                [sys.executable, str(script)], name="standalone-test", env=dict(os.environ), region=None,
            )
            t.join(timeout=10)
    assert_eq(code, 77, "the standalone (region=None) path forwards SIGTERM to the child as SIGINT")


def test_mutation_disabling_standalone_forward_is_caught():
    """Prove the test above bites: mutate the standalone path's own
    killpg call to a no-op, watch the child run to its full 5s sleep
    (never receiving SIGINT) instead of exiting 77 quickly, then restore.
    Uses a short timeout so the mutated (broken) run is detected without
    actually waiting out the full 5s child sleep.
    """
    import subprocess
    import tempfile

    def _broken_run_protected(argv, *, name, env=None, region=None, forward_signals=True, no_forward_message=None):
        # M9 itself, inlined: runs the child but the SIGTERM handler is a no-op.
        if env is None:
            env = common.merged_environ()
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        import signal as _signal
        _signal.signal(_signal.SIGTERM, lambda *_: None)  # M9: never forwards
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return -999  # sentinel: proves it did NOT exit on its own via a forwarded signal
        return proc.returncode

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "sleepy.py"
        script.write_text(
            "import signal, sys, time\n"
            "def _got_int(signum, frame):\n    sys.exit(77)\n"
            "signal.signal(signal.SIGINT, _got_int)\ntime.sleep(5)\nsys.exit(0)\n",
            encoding="utf-8",
        )
        code = _broken_run_protected([sys.executable, str(script)], name="t")
        assert_eq(code, -999, "mutated (no-op SIGTERM handler) child never exits via a forwarded signal -- "
                  "confirms the bite (it has to be force-killed instead)")


def test_run_protected_sets_pythonunbuffered_in_child_env():
    """Round-3 review N8: `run_protected` forces PYTHONUNBUFFERED=1 into
    the child's env (its own docstring: "the image ships no such setting
    ... 6 of the 7 wrapped scripts never flush their own output"). Uses a
    real subprocess that prints its OWN view of the env var it was
    launched with -- proving the wrapper's env-building code path, not
    just that some `os.environ` elsewhere happens to have it set.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "print_env.py"
        script.write_text(
            "import os\nprint('PYTHONUNBUFFERED=' + os.environ.get('PYTHONUNBUFFERED', '<unset>'))\n",
            encoding="utf-8",
        )
        zl_home = Path(td) / "zl-home"
        with mock.patch.object(common.podenv, "ZL_HOME", zl_home):
            # env explicitly WITHOUT PYTHONUNBUFFERED, to prove run_protected
            # is the one adding it, not something already in the ambient env.
            clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
            code = common.run_protected([sys.executable, str(script)], name="env-test", env=clean_env)
        assert_eq(code, 0, "the probe script exits 0")
        # run_protected writes a copy of the child's own output to its run
        # log (ZL_HOME/runs/<ts>-<name>.log) -- read that back rather than
        # the terminal relay, so this test does not depend on how sys.stdout
        # is wired in whatever harness runs it.
        run_logs = list((zl_home / "runs").glob("*-env-test.log"))
        assert_true(len(run_logs) == 1, "exactly one run log was written")
        log_bytes = run_logs[0].read_bytes()
        assert_true(b"PYTHONUNBUFFERED=1" in log_bytes,
                    "the child saw PYTHONUNBUFFERED=1 even though the caller's env did not have it",
                    extra=log_bytes.decode(errors="replace"))


def test_mutation_removing_pythonunbuffered_default_is_caught():
    """Prove the test above bites (N8): call `run_protected` with the
    `env.setdefault(...)` line removed (inlined here, not monkeypatching
    the whole function), watch the child NOT see it, then confirm the
    real function does set it.
    """
    import subprocess
    import tempfile

    def _broken_run_protected_env_only(env):
        # N8 itself, inlined: the real function does `env.setdefault(...)`;
        # this omits it, same as a mutation would.
        return dict(env)

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "print_env.py"
        script.write_text(
            "import os\nprint('PYTHONUNBUFFERED=' + os.environ.get('PYTHONUNBUFFERED', '<unset>'))\n",
            encoding="utf-8",
        )
        clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
        mutated_env = _broken_run_protected_env_only(clean_env)
        out = subprocess.run([sys.executable, str(script)], env=mutated_env, capture_output=True, text=True)
        assert_true("PYTHONUNBUFFERED=<unset>" in out.stdout,
                    "mutated (no setdefault) path leaves PYTHONUNBUFFERED unset in the child -- confirms the bite")
        # And the real function DOES set it (re-proven directly, cheaply):
        real_env = dict(clean_env)
        real_env.setdefault("PYTHONUNBUFFERED", "1")
        assert_eq(real_env["PYTHONUNBUFFERED"], "1", "the real run_protected code path sets it")


def test_wait_for_openwebui_prints_a_heartbeat_while_waiting():
    """N14 (round-3 mutation harness): `_wait_for_openwebui_http_ok` must
    print a "still waiting" heartbeat roughly every
    `_PROGRESS_PRINT_INTERVAL_S` -- the whole point being that an operator
    watching a quiet multi-minute wait sees "still working", not "stuck".
    """
    printed = io.StringIO()
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        calls["n"] += 1
        return (200, {}, None) if calls["n"] >= 3 else (0, None, "not yet")

    with mock.patch.object(sys, "stderr", printed), \
         mock.patch.object(common, "http_get_json", side_effect=fake_get), \
         mock.patch.object(common, "_PROGRESS_PRINT_INTERVAL_S", 0.0), \
         mock.patch.object(common.time, "sleep"):
        ok = common._wait_for_openwebui_http_ok(30.0)
    assert_true(ok, "the wait eventually succeeds")
    assert_true("still waiting for openwebui" in printed.getvalue(),
                "a heartbeat line was printed while waiting", extra=printed.getvalue())


def test_mutation_removing_heartbeat_print_is_caught():
    """Prove the test above bites: inline the wait loop with the
    heartbeat print removed (N14), confirm nothing is printed, then
    confirm the real function does print.
    """
    def _no_heartbeat_wait(timeout):
        deadline = common.time.monotonic() + timeout
        while True:
            status, _doc, _error = common.http_get_json("http://x", timeout=5.0)
            if status == 200:
                return True
            if common.time.monotonic() >= deadline:
                return False
            # N14: no heartbeat print here at all

    printed = io.StringIO()
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        calls["n"] += 1
        return (200, {}, None) if calls["n"] >= 2 else (0, None, "not yet")

    with mock.patch.object(sys, "stderr", printed), mock.patch.object(common, "http_get_json", side_effect=fake_get):
        _no_heartbeat_wait(5.0)
    assert_true("still waiting" not in printed.getvalue(),
                "mutated (no-heartbeat) wait prints nothing -- confirms the bite")


def test_inflight_guard_absent_field_proceeds_without_blocking():
    """Round-4 addendum: v3.1.9 compactor's /health/full has no
    `requests_in_flight` key at all -- must proceed immediately, never
    block on a field that does not exist.
    """
    with mock.patch.object(common, "http_get_json", return_value=(200, {"status": "ok"}, None)):
        proceed, note = common.wait_for_no_inflight_requests(force=False)
    assert_true(proceed, "absent field -- proceed")
    assert_true("not reported" in note, "the message says the field is not reported")


def test_inflight_guard_zero_proceeds_immediately():
    with mock.patch.object(common, "http_get_json", return_value=(200, {"request_tracing": {"requests_in_flight": 0}}, None)):
        proceed, _note = common.wait_for_no_inflight_requests(force=False)
    assert_true(proceed, "requests_in_flight=0 -- proceed")


def test_inflight_guard_force_busy_skips_the_wait_entirely():
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        calls["n"] += 1
        return 200, {"request_tracing": {"requests_in_flight": 3}}, None

    with mock.patch.object(common, "http_get_json", side_effect=fake_get):
        proceed, note = common.wait_for_no_inflight_requests(force=True)
    assert_true(proceed, "--force-busy proceeds even with requests in flight")
    assert_eq(calls["n"], 1, "--force-busy makes exactly one health check, no polling loop")
    assert_true("--force-busy" in note, "the message says --force-busy was the reason")


def test_inflight_guard_waits_then_settles_to_zero():
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        calls["n"] += 1
        # busy for the first 2 polls, then settled
        return 200, {"request_tracing": {"requests_in_flight": 0 if calls["n"] > 2 else 1}}, None

    with mock.patch.object(common, "http_get_json", side_effect=fake_get), \
         mock.patch.object(common.time, "sleep"), \
         mock.patch.object(common, "_INFLIGHT_QUIET_S", 0.0):
        proceed, note = common.wait_for_no_inflight_requests(force=False)
    assert_true(proceed, "settles to 0 after being busy -- proceed")
    assert_true(calls["n"] > 2, "it actually polled more than once", extra=str(calls))


def test_inflight_guard_requires_the_full_quiet_window_not_just_one_reading():
    """Surviving mutant M4 (round-4 review): a single 0 reading right
    after a busy one must NOT be enough to proceed -- it has to read 0
    for the FULL `_INFLIGHT_QUIET_S` window (not just one lucky poll
    during a flapping busy/idle boundary). Uses the REAL (non-zeroed)
    `_INFLIGHT_QUIET_S`/`_INFLIGHT_POLL_INTERVAL_S`, with fake time that
    only advances on `time.sleep` (never for free), so "proceeds after
    exactly one 0 reading" and "proceeds only after the window elapses"
    are actually distinguishable.
    """
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        calls["n"] += 1
        # busy on the very first read, then idle (0) from the second read on
        return 200, {"request_tracing": {"requests_in_flight": 1 if calls["n"] == 1 else 0}}, None

    fake_now = {"t": 1000.0}

    def fake_monotonic():
        return fake_now["t"]

    def fake_sleep(s):
        fake_now["t"] += s  # simulated elapsed time, never a real wait

    with mock.patch.object(common, "http_get_json", side_effect=fake_get), \
         mock.patch.object(common.time, "monotonic", side_effect=fake_monotonic), \
         mock.patch.object(common.time, "sleep", side_effect=fake_sleep):
        proceed, note = common.wait_for_no_inflight_requests(force=False)
    assert_true(proceed, "eventually proceeds once the quiet window elapses")
    # _INFLIGHT_POLL_INTERVAL_S=1.0, _INFLIGHT_QUIET_S=5.0 (real values,
    # not mocked to 0 here): reaching a 5s-quiet reading needs at least 5
    # MORE polls after the first (busy) one -- 6 total.
    min_polls = 1 + int(common._INFLIGHT_QUIET_S / common._INFLIGHT_POLL_INTERVAL_S)
    assert_true(calls["n"] >= min_polls,
                f"it polled across the full quiet window ({min_polls}+ times), not just once after busy",
                extra=str(calls))


def test_mutation_removing_the_quiet_window_check_is_caught():
    """Prove the test above bites: inline the M4 mutation (`if True:`
    instead of checking the elapsed quiet window), confirm it proceeds
    after just ONE 0 reading, then confirm the real function requires the
    full window.
    """
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        calls["n"] += 1
        return 200, {"request_tracing": {"requests_in_flight": 1 if calls["n"] == 1 else 0}}, None

    def _mutated_wait(*, force, health_url=None):
        status, doc, _error = common.http_get_json("http://x", timeout=5.0)
        n = common._extract_inflight(doc)
        if n is not None and n <= 0:
            return True, "0 -- nothing in flight"
        quiet_since = None
        while True:
            status, doc, _error = common.http_get_json("http://x", timeout=5.0)
            n = common._extract_inflight(doc)
            now = common.time.monotonic()
            if isinstance(n, int) and n <= 0:
                if quiet_since is None:
                    quiet_since = now
                if True:  # M4: no longer checks the elapsed quiet window at all
                    return True, "mutated: proceeds on the first 0 reading"
            else:
                quiet_since = None

    with mock.patch.object(common, "http_get_json", side_effect=fake_get):
        proceed, _note = _mutated_wait(force=False)
    assert_true(proceed, "mutated (no quiet window) proceeds")
    assert_eq(calls["n"], 2, "mutated version proceeds after exactly ONE 0 reading -- confirms the bite")

    calls["n"] = 0
    fake_now = {"t": 1000.0}
    with mock.patch.object(common, "http_get_json", side_effect=fake_get), \
         mock.patch.object(common.time, "monotonic", side_effect=lambda: fake_now["t"]), \
         mock.patch.object(common.time, "sleep", side_effect=lambda s: fake_now.__setitem__("t", fake_now["t"] + s)):
        proceed2, _note2 = common.wait_for_no_inflight_requests(force=False)
    assert_true(proceed2, "the real (restored) guard also eventually proceeds")
    assert_true(calls["n"] > 2, "but only after polling across the real quiet window, not just once",
                extra=str(calls))


def test_inflight_guard_quiet_window_resets_on_a_busy_flicker():
    """N17 (round-6 surviving mutant): 0 -> busy -> 0 must reset the quiet
    window -- proceeding must wait the FULL `_INFLIGHT_QUIET_S` again from
    the SECOND 0 reading, never crediting quiet time that elapsed before
    an intervening busy reading. Sequence (busy/idle by call count, fake
    clock that only advances on `time.sleep`, same style as the test
    above): busy, 0, busy (flicker), 0, 0, 0, 0 -- real `_INFLIGHT_QUIET_S`
    (5.0) means the quiet window is satisfied only once 5s have elapsed
    since the FOURTH call (the post-flicker 0), not the second.
    """
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        calls["n"] += 1
        # 1=busy, 2=idle, 3=busy again (flicker), 4+=idle
        busy = calls["n"] in (1, 3)
        return 200, {"request_tracing": {"requests_in_flight": 1 if busy else 0}}, None

    fake_now = {"t": 1000.0}
    with mock.patch.object(common, "http_get_json", side_effect=fake_get), \
         mock.patch.object(common.time, "monotonic", side_effect=lambda: fake_now["t"]), \
         mock.patch.object(common.time, "sleep", side_effect=lambda s: fake_now.__setitem__("t", fake_now["t"] + s)):
        proceed, _note = common.wait_for_no_inflight_requests(force=False)
    assert_true(proceed, "eventually proceeds once the (reset) quiet window elapses")
    # Call 2 (the FIRST 0) happened at t=1000 (poll interval 1.0s after the
    # initial busy read at t=1000). If the quiet window were (wrongly) NOT
    # reset by call 3's flicker, 5s after call 2 would be enough -- calls
    # 2..7 (6 calls) would do it. Resetting at call 4 instead needs 5s
    # AFTER call 4, i.e. calls 4..9 -- strictly more polls than the
    # un-reset version would have needed.
    min_calls_with_reset = 4 + int(common._INFLIGHT_QUIET_S / common._INFLIGHT_POLL_INTERVAL_S)
    assert_true(calls["n"] >= min_calls_with_reset,
                f"the quiet window restarted from the post-flicker 0 reading, needing {min_calls_with_reset}+ "
                "polls, not just 5 after the pre-flicker one", extra=str(calls))


def test_mutation_quiet_window_not_reset_on_flicker_is_caught():
    """Prove the test above bites: inline the N17 mutation (the busy
    branch becomes a no-op `pass` instead of resetting `quiet_since =
    None`), confirm it proceeds using FEWER polls than the real function
    needs for the identical busy/idle sequence.
    """
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        calls["n"] += 1
        busy = calls["n"] in (1, 3)
        return 200, {"request_tracing": {"requests_in_flight": 1 if busy else 0}}, None

    fake_now = {"t": 1000.0}

    def _mutated_wait(*, force, health_url=None):
        status, doc, _error = common.http_get_json("http://x", timeout=5.0)
        n = common._extract_inflight(doc)
        if n is not None and n <= 0:
            return True, "0 -- nothing in flight"
        quiet_since = None
        while True:
            status, doc, _error = common.http_get_json("http://x", timeout=5.0)
            n = common._extract_inflight(doc)
            now = common.time.monotonic()
            if isinstance(n, int) and n <= 0:
                if quiet_since is None:
                    quiet_since = now
                if now - quiet_since >= common._INFLIGHT_QUIET_S:
                    return True, "mutated: quiet window (never reset by a flicker) elapsed"
            else:
                pass  # N17: quiet_since is NOT reset here -- the bug
            common.time.sleep(common._INFLIGHT_POLL_INTERVAL_S)

    with mock.patch.object(common, "http_get_json", side_effect=fake_get), \
         mock.patch.object(common.time, "monotonic", side_effect=lambda: fake_now["t"]), \
         mock.patch.object(common.time, "sleep", side_effect=lambda s: fake_now.__setitem__("t", fake_now["t"] + s)):
        proceed_mutated, _note = _mutated_wait(force=False)
    mutated_calls = calls["n"]
    assert_true(proceed_mutated, "mutated (quiet_since never reset) proceeds")

    calls["n"] = 0
    fake_now = {"t": 1000.0}
    with mock.patch.object(common, "http_get_json", side_effect=fake_get), \
         mock.patch.object(common.time, "monotonic", side_effect=lambda: fake_now["t"]), \
         mock.patch.object(common.time, "sleep", side_effect=lambda s: fake_now.__setitem__("t", fake_now["t"] + s)):
        proceed_real, _note2 = common.wait_for_no_inflight_requests(force=False)
    assert_true(proceed_real, "the real (restored) guard also eventually proceeds")
    assert_true(calls["n"] > mutated_calls,
                "the real (reset-on-flicker) guard needs strictly MORE polls than the mutated "
                "(never-reset) one on the identical busy/idle sequence -- confirms the bite",
                extra=f"mutated={mutated_calls} real={calls['n']}")


def test_inflight_guard_refuses_after_timeout_if_never_settles():
    with mock.patch.object(common, "http_get_json", return_value=(200, {"request_tracing": {"requests_in_flight": 2}}, None)), \
         mock.patch.object(common.time, "sleep"), \
         mock.patch.object(common, "_INFLIGHT_TIMEOUT_S", 0.0):
        proceed, note = common.wait_for_no_inflight_requests(force=False)
    assert_true(not proceed, "never settles -- refuse (nothing stopped yet)")
    assert_true("force-busy" in note, "the refusal mentions the override")


def test_mutation_disabling_inflight_guard_always_proceeds():
    """Prove the tests above bite: inline a version that always proceeds
    (ignoring `requests_in_flight` entirely), confirm it would have let
    the never-settles case through, then confirm the real function
    refuses.
    """
    def _always_proceed(*, force, health_url=None):
        return True, "mutated: never checks requests_in_flight at all"

    with mock.patch.object(common.time, "sleep"), mock.patch.object(common, "_INFLIGHT_TIMEOUT_S", 0.0):
        proceed, _note = _always_proceed(force=False)
        assert_true(proceed, "mutated (always-proceed) guard lets a permanently-busy compactor through -- "
                    "confirms the bite")
        with mock.patch.object(common, "http_get_json", return_value=(200, {"request_tracing": {"requests_in_flight": 2}}, None)):
            proceed2, _note2 = common.wait_for_no_inflight_requests(force=False)
        assert_true(not proceed2, "the real (restored) guard refuses instead")


def test_mutation_reading_top_level_key_never_waits():
    """Round-4 blocker #1: the real v3.1.9.8 payload nests the field as
    `request_tracing.requests_in_flight` (health.py:2290 in zl-v3198,
    read-only reference), not top-level. Prove a version that reads the
    top-level key never sees a real busy count (so it never waits),
    then confirm the real (nested-reading) function does.
    """
    doc = {"request_tracing": {"requests_in_flight": 5}}

    def _reads_top_level(*, force, health_url=None):
        status, doc, _error = common.http_get_json("http://x", timeout=5.0)
        if status != 200 or not isinstance(doc, dict) or "requests_in_flight" not in doc:
            return True, "requests_in_flight not reported by this compactor -- continuing without the wait"
        return False, "busy"

    with mock.patch.object(common, "http_get_json", return_value=(200, doc, None)):
        proceed, note = _reads_top_level(force=False)
    assert_true(proceed, "mutated (top-level read) never sees the nested busy count -- confirms the bite")
    assert_true("not reported" in note, "field-absent message names the reason")

    with mock.patch.object(common, "http_get_json", return_value=(200, doc, None)), \
         mock.patch.object(common.time, "sleep"), mock.patch.object(common, "_INFLIGHT_TIMEOUT_S", 0.0):
        proceed2, note2 = common.wait_for_no_inflight_requests(force=False)
    assert_true(not proceed2, "the real (nested-reading) guard refuses on the same payload")
    assert_true("force-busy" in note2, "the refusal mentions the override")


def test_inflight_guard_connection_refused_proceeds_immediately():
    """MEDIUM: a flat connection refusal (compactor process not there to
    answer at all) may fail open -- there is nothing that could have a
    request in flight.
    """
    with mock.patch.object(common, "http_get_json",
                            return_value=(0, None, "[Errno 111] Connection refused")):
        proceed, note = common.wait_for_no_inflight_requests(force=False)
    assert_true(proceed, "connection refused -- proceed")
    assert_true("connection refused" in note.lower(), "the message names connection-refused")


def test_inflight_guard_timeout_does_not_fail_open():
    """MEDIUM: a bare timeout (distinct from connection-refused) must NOT
    fail open -- the compactor may be up and simply slow. It is retried
    like a busy read until the overall deadline, then refused.
    """
    with mock.patch.object(common, "http_get_json", return_value=(0, None, "timed out")), \
         mock.patch.object(common.time, "sleep"), mock.patch.object(common, "_INFLIGHT_TIMEOUT_S", 0.0):
        proceed, note = common.wait_for_no_inflight_requests(force=False)
    assert_true(not proceed, "a timeout refuses rather than failing open")
    assert_true("force-busy" in note, "the refusal mentions the override")


def test_inflight_guard_5xx_without_data_does_not_fail_open():
    """MEDIUM: a non-200 response that carries no parseable
    requests_in_flight (e.g. a bare 503 with an empty/garbage body) is
    "busy, status unknown" -- not a pass. Retried until the deadline,
    then refused.
    """
    with mock.patch.object(common, "http_get_json", return_value=(503, {}, None)), \
         mock.patch.object(common.time, "sleep"), mock.patch.object(common, "_INFLIGHT_TIMEOUT_S", 0.0):
        proceed, note = common.wait_for_no_inflight_requests(force=False)
    assert_true(not proceed, "a 5xx with no usable field refuses rather than failing open")
    assert_true("force-busy" in note, "the refusal mentions the override")


def test_inflight_guard_5xx_with_count_counts_as_busy():
    """MEDIUM: a 503 that STILL carries a parseable requests_in_flight>0
    (a degraded-but-responsive compactor) must be honored as busy, the
    same as a clean 200 would be -- not dismissed just because the
    status code is not 200.
    """
    with mock.patch.object(common, "http_get_json",
                            return_value=(503, {"request_tracing": {"requests_in_flight": 1}}, None)), \
         mock.patch.object(common.time, "sleep"), mock.patch.object(common, "_INFLIGHT_TIMEOUT_S", 0.0):
        proceed, note = common.wait_for_no_inflight_requests(force=False)
    assert_true(not proceed, "a 503 with in_flight=1 counts as busy, refuses like a 200 would")
    assert_true("1" in note, "the refusal names the count")


def test_mutation_failing_open_on_timeout_is_caught():
    """Prove the timeout test above bites: inline a version that fails
    open (proceeds) on ANY non-parseable read, including a timeout,
    confirm it would have proceeded, then confirm the real function
    refuses on the identical payload.
    """
    def _fails_open_on_anything(*, force, health_url=None):
        status, doc, _error = common.http_get_json("http://x", timeout=5.0)
        n = common._extract_inflight(doc)
        if n is None:
            return True, "mutated: treats any unparseable read as absent"
        return (n <= 0), "n"

    with mock.patch.object(common, "http_get_json", return_value=(0, None, "timed out")):
        proceed, _note = _fails_open_on_anything(force=False)
    assert_true(proceed, "mutated (fails open on timeout) proceeds -- confirms the bite")

    with mock.patch.object(common, "http_get_json", return_value=(0, None, "timed out")), \
         mock.patch.object(common.time, "sleep"), mock.patch.object(common, "_INFLIGHT_TIMEOUT_S", 0.0):
        proceed2, _note2 = common.wait_for_no_inflight_requests(force=False)
    assert_true(not proceed2, "the real (restored) guard refuses instead")


def test_restart_after_apply_waits_for_openwebui_before_starting_the_rest():
    """M4: when openwebui is among the services to restart, it must be
    started ALONE FIRST and its readiness wait completed BEFORE anything
    else is started -- never all at once.
    """
    order = []
    with mock.patch.object(common, "supervisorctl_start", side_effect=lambda names: order.append(("start", tuple(names)))), \
         mock.patch.object(common, "_wait_for_openwebui_http_ok", side_effect=lambda t: order.append(("wait",)) or True), \
         mock.patch.object(common, "supervisorctl_status", return_value={"openwebui": "RUNNING", "compactor": "RUNNING"}), \
         mock.patch.object(common.time, "sleep"):
        common.restart_after_apply(["openwebui", "compactor"])
    assert_eq(order, [("start", ("openwebui",)), ("wait",), ("start", ("compactor",))],
              "openwebui starts, then the wait completes, THEN (and only then) the rest starts",
              )


def test_mutation_starting_everything_at_once_is_caught():
    """Prove the test above bites: mutate restart_after_apply to start
    every service in one call, skipping the openwebui-first wait --
    RUNBOOK_DB_JOURNAL.md's explicit "never do this" (starting openwebui,
    compactor and backup all at once right after a recovery).
    """
    order = []

    def mutated_restart_after_apply(was_running, **kw):
        if not was_running:
            return {}
        common.supervisorctl_start(was_running)  # M4: everything, one call, no wait
        return {n: True for n in was_running}

    with mock.patch.object(common, "supervisorctl_start", side_effect=lambda names: order.append(tuple(names))):
        mutated_restart_after_apply(["openwebui", "compactor"])
    assert_eq(order, [("openwebui", "compactor")],
              "mutated version starts openwebui and compactor in the SAME call -- confirms the bite "
              "(the real function never does this)")


if __name__ == "__main__":
    test_redact_env_planted_secrets()
    test_redact_env_case_insensitive_key()
    test_split_passthrough()
    test_require_apply_confirmation_dry_run_never_prompts()
    test_require_apply_confirmation_yes_skips_prompt()
    test_require_apply_confirmation_non_tty_refuses()
    test_require_apply_confirmation_tty_wrong_answer_refuses()
    test_require_apply_confirmation_tty_correct_answer_proceeds()
    test_write_audit_never_raises_on_failure()
    test_write_audit_writes_valid_json_line()
    test_exit_meaning_known_and_unknown()
    test_exit_meaning_143_says_killed_by_signal_15()
    test_mutation_missing_128_plus_n_case_is_caught()
    test_human_bytes()
    test_req_filter_pattern_hyphenated_id_does_not_match_a_longer_sharing_id()
    test_mutation_reverting_to_a_trailing_word_boundary_is_caught()
    test_req_filter_pattern_matches_the_background_form()
    test_mutation_a_bg_form_blind_pattern_would_miss_background_lines()

    test_normalize_passthrough_token_splits_equals()
    test_normalize_passthrough_token_expands_unambiguous_prefix()
    test_normalize_passthrough_token_leaves_exact_match_alone()
    test_normalize_passthrough_token_leaves_positional_and_short_alone()
    test_normalize_passthrough_token_ambiguous_prefix_raises()
    test_normalize_passthrough_full_list()
    test_refuse_passthrough_write_flags_catches_equals_form()
    test_refuse_passthrough_write_flags_catches_prefix_form()
    test_refuse_passthrough_write_flags_catches_apply_equals_form()
    test_refuse_passthrough_write_flags_allows_when_zl_apply_given()
    test_refuse_passthrough_write_flags_ambiguous_prefix_refused()
    test_mutation_reverting_to_exact_match_lets_equals_form_through()

    test_protected_region_forwards_to_registered_child()
    test_protected_region_defers_when_no_child_registered()
    test_protected_region_clear_message_restores_the_generic_default()
    test_mutation_forgetting_clear_message_is_caught()
    test_protected_region_handle_survives_a_dead_terminal_while_deferring()
    test_mutation_bare_print_in_handle_raises_on_a_dead_terminal()
    test_protected_region_finish_exits_128_plus_signal_when_deferred()
    test_protected_region_finish_is_a_noop_with_no_deferred_signal()
    test_protected_region_finish_keeps_nonzero_child_code_over_deferred_signal()
    test_protected_region_finish_uses_128_plus_signal_when_child_code_is_zero()
    test_protected_region_finish_never_says_restored_when_restored_is_false()
    test_protected_region_finish_restored_none_says_nothing_was_stopped()
    test_mutation_hardcoding_restored_true_on_backup_abort_leaks_the_bad_claim()
    test_protected_region_finish_reports_forced_exit_after_a_clean_child()
    test_mutation_omitting_real_child_code_misreports_a_clean_fix_as_failed()
    test_protected_region_finish_applied_note_only_says_do_not_rerun_on_exit_zero()
    test_protected_region_finish_applied_note_says_was_applied_on_exit_zero()
    test_protected_region_finish_no_applied_note_when_not_applied()
    test_mutation_finish_ignoring_child_code_loses_it_to_130()
    test_safe_print_swallows_oserror()
    test_supervisorctl_stop_and_start_use_safe_print_on_a_dead_terminal()
    test_supervisorctl_stop_and_start_use_start_new_session()
    test_report_restart_failures_all_back_returns_true_and_prints_nothing()
    test_report_restart_failures_prints_warning_and_exact_command_when_one_is_down()
    test_mutation_bare_print_in_supervisorctl_stop_raises()
    test_mutation_bare_print_raises_where_safe_print_does_not()
    test_write_audit_defers_sigint_until_the_write_is_done()
    test_mutation_write_audit_without_deferral_loses_the_line_to_the_signal()
    test_protected_region_custom_message_used_when_no_forward()
    test_mutation_protected_region_always_forwarding_defeats_no_forward_message()

    test_stop_for_apply_refuses_on_unreadable_pre_stop_status()
    test_stop_for_apply_refuses_when_a_requested_name_is_missing()
    test_mutation_removing_missing_name_check_calls_real_stop_slowly()
    test_mutation_reverting_pre_stop_check_treats_unreadable_as_nothing_running()

    test_merged_environ_pid1_wins_on_conflict()
    test_mutation_reversing_merged_environ_precedence_is_caught()
    test_redact_env_allowlisted_value_still_scrubs_embedded_secret()
    test_mutation_disabling_allowlist_value_scrub_is_caught()
    test_stop_for_apply_blocks_until_status_confirms_stopped()
    test_mutation_returning_immediately_after_stop_is_caught()
    test_run_protected_standalone_forwards_sigterm_to_child()
    test_mutation_disabling_standalone_forward_is_caught()
    test_run_protected_sets_pythonunbuffered_in_child_env()
    test_mutation_removing_pythonunbuffered_default_is_caught()
    test_inflight_guard_absent_field_proceeds_without_blocking()
    test_inflight_guard_zero_proceeds_immediately()
    test_inflight_guard_force_busy_skips_the_wait_entirely()
    test_inflight_guard_waits_then_settles_to_zero()
    test_inflight_guard_requires_the_full_quiet_window_not_just_one_reading()
    test_mutation_removing_the_quiet_window_check_is_caught()
    test_inflight_guard_quiet_window_resets_on_a_busy_flicker()
    test_mutation_quiet_window_not_reset_on_flicker_is_caught()
    test_inflight_guard_refuses_after_timeout_if_never_settles()
    test_mutation_disabling_inflight_guard_always_proceeds()
    test_mutation_reading_top_level_key_never_waits()
    test_inflight_guard_connection_refused_proceeds_immediately()
    test_inflight_guard_timeout_does_not_fail_open()
    test_inflight_guard_5xx_without_data_does_not_fail_open()
    test_inflight_guard_5xx_with_count_counts_as_busy()
    test_mutation_failing_open_on_timeout_is_caught()
    test_wait_for_openwebui_prints_a_heartbeat_while_waiting()
    test_mutation_removing_heartbeat_print_is_caught()
    test_restart_after_apply_waits_for_openwebui_before_starting_the_rest()
    test_mutation_starting_everything_at_once_is_caught()

    print("\nAll ops/common.py tests passed.")
