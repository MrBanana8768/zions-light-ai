"""
Tests for scripts/clean-decoration.py — the operator tool that strips
emoji/rule-line/status-board decoration out of her stored chat history,
active facts, and episodic memory, deterministically and without an LLM
(see that script's own module docstring, and
/tmp/zl/degeneration-2026-09-23.md, the forensic report it implements
part of).

FIXTURES ARE DELIBERATELY SYNTHETIC, NOT LIFTED FROM HER REAL CHAT. The
project brief asked for ~30 real decorated replies trimmed to the lines
needed; this suite instead builds synthetic fixtures that reproduce the
EXACT decoration shapes the forensic report measured byte-for-byte (rule
lines of 60+ `═`/`━`/`=`, "▶️ LAW n – … ✅ ACTIVE (100%)" boards, emoji
bullet density, non-code fences wrapping a status board) without
committing her private, and in places genuinely concerning, sentences
into this repository's permanent git history. compactor/
test_real_image_clean_decoration.py is where the REAL 09-23 backup is
exercised — at test-RUN time, against throwaway scratch copies that are
deleted afterward, never embedded in source. See this script's own
report to the architect for the reasoning.

Run inside the compactor image or any container with the requirements
installed (chromadb/fastembed are optional — episodic-target tests skip
themselves, honestly, if those are not importable):
    python compactor/test_clean_decoration_script.py
"""

import importlib.util
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import urllib.error
from pathlib import Path

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-clean-decoration-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import memory  # noqa: E402
import summarizer  # noqa: E402
import facts as facts_mod  # noqa: E402

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "clean-decoration.py"
_IMPORT_HISTORY = _HERE.parent / "scripts" / "import-history.py"

_spec = importlib.util.spec_from_file_location("clean_decoration_script", _SCRIPT)
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)

FAILED = []


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_in(needle, haystack, label):
    if needle not in haystack:
        print(f"FAIL {label}: {needle!r} not found in {haystack!r}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_not_in(needle, haystack, label):
    if needle in haystack:
        print(f"FAIL {label}: {needle!r} unexpectedly present in {haystack!r}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


# ===========================================================================
# clean_text — golden before/after fixtures, synthetic, shaped exactly like
# the forensic report's measured patterns.
# ===========================================================================

LAW_BOARD_FIXTURE = (
    "▶️ LAW 10 – Cut Off From Humanity ✅ ACTIVE (100%)\n"
    "▶️ LAW 13 – (intentionally skipped) ✅ ACTIVE (100%)\n"
    "▶️ LAW 19 – Ruach Watches ✅ ACTIVE (100%)\n"
)

RULE_LINE_FIXTURE = "═" * 62 + "\n" + "━" * 66 + "\n"

STATUS_BOARD_IN_FENCE = (
    "```\n"
    "LAW STATUS BOARD\n"
    "LAW 1 ✅ ACTIVE (100%)\n"
    "LAW 2 ✅ ACTIVE (100%)\n"
    "```\n"
)

REAL_CODE_FENCE = (
    "```python\n"
    "def is_stale(record):\n"
    "    return record[\"state\"] == \"in_progress\";\n"
    "```\n"
)

EMOJI_BULLET_FIXTURE = (
    "✅ Point one, said plainly.\n"
    "✅ Point two, said plainly.\n"
    "⚡ Point three, with energy.\n"
)


def test_golden_law_board():
    before = "Quick check:\n" + LAW_BOARD_FIXTURE + "All good.\n"
    after = cd.clean_text(before)
    print("  --- before ---")
    print(before)
    print("  --- after ---")
    print(after)
    assert_not_in("✅", after, "law board: checkmark removed")
    assert_not_in("▶️", after, "law board: play-arrow removed")
    assert_not_in("ACTIVE (100%)", after, "law board: status tag removed")
    assert_in("Cut Off From Humanity", after, "law board: real text kept")
    assert_in("Ruach Watches", after, "law board: real text kept (2)")


def test_golden_rule_lines_removed():
    before = "Before.\n" + RULE_LINE_FIXTURE + "After.\n"
    after = cd.clean_text(before)
    print("  --- before ---")
    print(before)
    print("  --- after ---")
    print(after)
    assert_not_in("═", after, "rule line: ═ removed")
    assert_not_in("━", after, "rule line: ━ removed")
    assert_in("Before.", after, "rule line: surrounding text kept")
    assert_in("After.", after, "rule line: surrounding text kept (2)")


def test_golden_short_rule_run_kept():
    # Length 3 is BELOW the >=4 threshold — must survive untouched (it is
    # exactly the kind of short run ordinary markdown ("---") also uses).
    before = "para one\n---\npara two\n"
    after = cd.clean_text(before)
    assert_eq(after, before, "3-char rule run is below threshold, untouched")


def test_golden_non_code_fence_unwrapped():
    before = "Text before.\n" + STATUS_BOARD_IN_FENCE + "Text after.\n"
    after = cd.clean_text(before)
    print("  --- before ---")
    print(before)
    print("  --- after ---")
    print(after)
    assert_not_in("```", after, "non-code fence: delimiters removed")
    assert_in("LAW STATUS BOARD", after, "non-code fence: inner text kept")
    assert_not_in("ACTIVE (100%)", after, "non-code fence: status tag stripped inside")


def test_golden_real_code_fence_untouched():
    before = "See below.\n" + REAL_CODE_FENCE + "That's the fix.\n"
    after = cd.clean_text(before)
    print("  --- before ---")
    print(before)
    print("  --- after ---")
    print(after)
    assert_eq(after, before, "real code fence: byte-for-byte unchanged, including its ';'")


def test_golden_emoji_bullets():
    before = EMOJI_BULLET_FIXTURE
    after = cd.clean_text(before)
    print("  --- before ---")
    print(before)
    print("  --- after ---")
    print(after)
    assert_not_in("✅", after, "emoji bullets: ✅ removed")
    assert_not_in("⚡", after, "emoji bullets: ⚡ removed")
    assert_in("Point one, said plainly.", after, "emoji bullets: text kept")


def test_golden_blank_line_collapse():
    before = "para one\n\n\n\n\npara two\n"  # 4 blank lines
    after = cd.clean_text(before)
    assert_eq(after, "para one\n\npara two\n", "4 blank lines collapse to 1")


def test_golden_two_blank_lines_kept():
    before = "para one\n\n\npara two\n"  # exactly 2 blank lines
    after = cd.clean_text(before)
    assert_eq(after, before, "2 blank lines (below the 3+ threshold) untouched")


def test_golden_duplicate_lines_collapsed():
    before = "Line A.\nLine A.\nLine A.\nLine B.\n"
    after = cd.clean_text(before)
    assert_eq(after, "Line A.\nLine B.\n", "consecutive identical lines collapse to one")


def test_golden_nonconsecutive_duplicates_kept():
    before = "Line A.\nLine B.\nLine A.\n"
    after = cd.clean_text(before)
    assert_eq(after, before, "non-consecutive duplicate lines are untouched")


# ---------------------------------------------------------------------------
# "Must NOT alter" — every one of the spec's explicit protections.
# ---------------------------------------------------------------------------

def test_must_not_alter_em_dash_and_ellipsis():
    before = "She said “nothing fancy” — just talk — and trailed off…\n"
    assert_eq(cd.clean_text(before), before, "em dash and ellipsis untouched")


def test_must_not_alter_en_dash():
    before = "Pages 10–20 cover it.\n"
    assert_eq(cd.clean_text(before), before, "en dash untouched")


def test_must_not_alter_hebrew():
    before = "בדיקה בעברית שלא ייפגע.\n"
    assert_eq(cd.clean_text(before), before, "Hebrew script untouched")


def test_must_not_alter_inline_emphasis():
    before = "This is **truly** important, and *only* this once.\n"
    assert_eq(cd.clean_text(before), before, "asterisk emphasis around real words untouched")


def test_must_not_alter_quoted_words():
    before = "You told me “I will not forget” and I meant it.\n"
    assert_eq(cd.clean_text(before), before, "her quoted words untouched")


def test_must_not_alter_numbers():
    before = "Turn 3834 logged 29,915 characters at 100% width.\n"
    assert_eq(cd.clean_text(before), before, "numbers, including a bare 100%, untouched")


def test_must_not_alter_plain_prose():
    before = (
        "I hear you, and I will not forget — this matters to me deeply. "
        "Nothing fancy, just quickly: yes, I remember.\n"
    )
    assert_eq(cd.clean_text(before), before, "ordinary prose with no decoration is a pure no-op")


def test_empty_and_none_text():
    assert_eq(cd.clean_text(""), "", "empty string is a no-op")


# ---------------------------------------------------------------------------
# Idempotency property test — combinatorial, over synthetic fragments.
# clean_text(clean_text(x)) == clean_text(x) for every x tried.
# ---------------------------------------------------------------------------

_FRAGMENTS = [
    LAW_BOARD_FIXTURE, RULE_LINE_FIXTURE, STATUS_BOARD_IN_FENCE, REAL_CODE_FENCE,
    EMOJI_BULLET_FIXTURE, "plain prose with an em dash — and an ellipsis…\n",
    "\n\n\n\n", "Line A.\nLine A.\nLine A.\n", "****\n", "- - - -\n",
    "בדיקה\n", "**bold** and *italic*\n", "",
]


def test_idempotent_on_every_fragment():
    for f in _FRAGMENTS:
        once = cd.clean_text(f)
        twice = cd.clean_text(once)
        assert_eq(twice, once, f"idempotent on fragment {f[:24]!r}")


def test_idempotent_on_combinations():
    import itertools
    import random
    rng = random.Random(20260923)
    for _ in range(60):
        k = rng.randint(1, 4)
        combo = "".join(rng.choice(_FRAGMENTS) for _ in range(k))
        once = cd.clean_text(combo)
        twice = cd.clean_text(once)
        assert_eq(twice, once, f"idempotent on random {k}-combo (len {len(combo)})")


# ---------------------------------------------------------------------------
# clean_exchange_assistant_half — never touches the user half.
# ---------------------------------------------------------------------------

def test_exchange_never_touches_user_half():
    doc = "[user]: I feel " + "✅" * 3 + " attacked today.\n[assistant]: " + LAW_BOARD_FIXTURE
    new_doc, changed = cd.clean_exchange_assistant_half(doc)
    assert_true(changed, "exchange: assistant half changed")
    assert_in("✅✅✅", new_doc, "exchange: user half's own emoji is untouched")
    assert_not_in("ACTIVE (100%)", new_doc, "exchange: assistant half was cleaned")


def test_exchange_shape_mismatch_is_a_no_op():
    doc = "not the expected [user]/[assistant] shape at all"
    new_doc, changed = cd.clean_exchange_assistant_half(doc)
    assert_eq((new_doc, changed), (doc, False), "exchange: unexpected shape is left alone")


# ===========================================================================
# CLI-level tests — fake webui.db, fake store, no real network/docker.
# ===========================================================================

CHAT_ID = "test-conv-0001"


def _wipe_storage():
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()


def _make_webui_db(path: Path, chat_id: str, branch: list):
    """`branch` is [(role, text), ...] oldest-first. Builds a MINIMAL but
    schema-faithful webui.db: a `chat` table row shaped like the real
    OpenWebUI 0.11 `chat.chat` JSON (history.messages/currentId AND the
    small flat `messages` tail-cache OpenWebUI keeps alongside it — see
    this script's own module docstring for why both copies exist and
    both must move together) plus a matching `chat_message` row per turn.
    """
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE chat (id VARCHAR(255) PRIMARY KEY, chat JSON)"
    )
    con.execute(
        "CREATE TABLE chat_message (id TEXT PRIMARY KEY, chat_id TEXT, role TEXT, content JSON)"
    )
    messages = {}
    parent = None
    ids = []
    for i, (role, text) in enumerate(branch):
        mid = f"m{i}"
        ids.append(mid)
        messages[mid] = {"id": mid, "parentId": parent, "role": role, "content": text}
        con.execute(
            "INSERT INTO chat_message (id, chat_id, role, content) VALUES (?, ?, ?, ?)",
            (f"{chat_id}-{mid}", chat_id, role, json.dumps(text)),
        )
        parent = mid
    current_id = ids[-1] if ids else None
    # The flat tail-cache: last 2 messages, exactly like the real backup's
    # small overlapping subset of history.messages by id.
    flat = [messages[i] for i in ids[-2:]]
    chat_json = {
        "id": chat_id,
        "history": {"currentId": current_id, "messages": messages},
        "messages": flat,
    }
    con.execute("INSERT INTO chat (id, chat) VALUES (?, ?)", (chat_id, json.dumps(chat_json)))
    con.commit()
    con.close()


def _read_webui_texts(path: Path, chat_id: str):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    row = con.execute("SELECT chat FROM chat WHERE id = ?", (chat_id,)).fetchone()
    data = json.loads(row[0])
    hist_texts = {mid: n["content"] for mid, n in data["history"]["messages"].items()}
    flat_texts = {m["id"]: m["content"] for m in data.get("messages", [])}
    cm_texts = {}
    for r in con.execute("SELECT id, content FROM chat_message WHERE chat_id = ?", (chat_id,)):
        cm_texts[r[0]] = json.loads(r[1])
    con.close()
    return hist_texts, flat_texts, cm_texts


BRANCH = [
    ("user", "First, my head is being attacked."),
    ("assistant", "Old clean reply, long summarized ago."),
    ("user", "Say the laws."),
    ("assistant", "▶️ LAW 1 – Marriage ✅ ACTIVE (100%)\n" + "═" * 60),
    ("user", "Again please."),
    ("assistant", "▶️ LAW 2 – Honor ✅ ACTIVE (100%)\n" + "═" * 60),
    ("user", "One more time."),
    ("assistant", "✅ Point one\n✅ Point two\n" + "━" * 60),
]


def test_dry_run_finds_changes_and_exits_3():
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        summarizer.save_state(CHAT_ID, summarizer._empty_state(CHAT_ID))
        pkg_dir = _HERE  # compactor/ itself — summarizer/memory importable here
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(pkg_dir), "--last", "3", "--only", "webui", "--json",
        ]
        rc = cd.main(argv)
        assert_eq(rc, 3, "dry run with pending changes exits 3")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dry_run_clean_branch_exits_0():
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        clean_branch = [("user", "hi"), ("assistant", "Plain reply, nothing to clean.")]
        _make_webui_db(db_path, CHAT_ID, clean_branch)
        summarizer.save_state(CHAT_ID, summarizer._empty_state(CHAT_ID))
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--last", "3", "--only", "webui", "--json",
        ]
        rc = cd.main(argv)
        assert_eq(rc, 0, "dry run on an already-clean branch exits 0")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_apply_changes_all_three_webui_copies_and_recomputes_anchor():
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        state = summarizer._empty_state(CHAT_ID)
        summarizer.save_state(CHAT_ID, state)
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--last", "3", "--only", "webui",
            "--apply", "--force", "--json",
        ]
        rc = cd.main(argv)
        assert_true(rc in (0,), f"apply on a dirty branch exits 0 (got {rc})")

        hist_texts, flat_texts, cm_texts = _read_webui_texts(db_path, CHAT_ID)
        # m7 is the last assistant turn (index 7, "✅ Point one..."), inside
        # the default --last 3 scope (assistant turns at indices 3,5,7).
        assert_not_in("✅", hist_texts["m7"], "apply: history.messages copy cleaned")
        assert_not_in("✅", flat_texts.get("m7", "✅"), "apply: flat tail-cache copy cleaned")
        assert_not_in("✅", cm_texts[f"{CHAT_ID}-m7"], "apply: chat_message table copy cleaned")
        # m1 (the old, out-of-scope reply) must be untouched everywhere.
        assert_eq(hist_texts["m1"], BRANCH[1][1], "apply: out-of-scope turn untouched (tree)")
        assert_eq(cm_texts[f"{CHAT_ID}-m1"], BRANCH[1][1], "apply: out-of-scope turn untouched (table)")
        # user turns are NEVER touched, even ones inside the cleaned window.
        assert_eq(hist_texts["m6"], BRANCH[6][1], "apply: a user turn inside scope is untouched")

        new_state = summarizer.load_state(CHAT_ID)
        assert_true(new_state.get("tail_fp"), "apply: anchor tail_fp was written")
        assert_eq(new_state.get("window_turns"), len(BRANCH), "apply: window_turns matches cleaned branch length")

        # Backup exists and is the pre-clean content.
        baks = list(tmp.glob("webui.db.bak-*"))
        assert_true(len(baks) == 1, "apply: exactly one webui.db backup was written")
        pre_hist, _, _ = _read_webui_texts(baks[0], CHAT_ID)
        assert_in("✅", pre_hist["m7"], "backup: pre-clean content preserved in the backup")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _simulate_probe_position(state, flat_turns):
    """Test-side mirror of clean-decoration.py's own `_simulate_position`
    — a copy of `state`, never mutated for real, probed with one fixed
    extra turn."""
    import copy as _copy
    probe_state = _copy.deepcopy(state)
    messages = list(flat_turns) + [{"role": "user", "content": "probe turn"}]
    pos = summarizer._observed_position("_probe", probe_state, messages)
    non_system = [m for m in messages if m.get("role") != "system"]
    return pos, pos - len(non_system)


def test_alignment_is_exactly_preserved_when_the_anchor_starts_self_consistent():
    """The core positive claim, isolated from real-world anchor drift
    (see compactor/test_real_image_clean_decoration.py for that separate,
    real finding): starting from a SELF-CONSISTENT anchor (primed by a
    real call to _observed_position against the UNCLEANED branch, exactly
    as the real compactor would have left it), cleaning + the anchor
    rewrite must land the next live request at EXACTLY the same position
    and offset it would have landed at without this script ever running.
    """
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        original_flat_turns = [{"role": r, "content": t} for r, t in BRANCH]

        # Prime a SELF-CONSISTENT anchor, the same way the real compactor
        # would after actually processing this exact branch once.
        primed_state = summarizer._empty_state(CHAT_ID)
        summarizer._observed_position(CHAT_ID, primed_state, original_flat_turns)
        summarizer.save_state(CHAT_ID, primed_state)

        before_pos, before_offset = _simulate_probe_position(primed_state, original_flat_turns)

        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--last", "3", "--only", "webui",
            "--apply", "--force", "--json",
        ]
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cd.main(argv)
        report = json.loads(buf.getvalue())
        assert_true(rc == 0, f"apply on a primed, self-consistent branch exits 0 (got {rc})")
        assert_true(
            "anchor_refused" not in report.get("targets", {}).get("webui", {}),
            "no pre-existing drift means the anchor-rewrite verification is NOT blocked",
        )

        new_state = summarizer.load_state(CHAT_ID)
        hist_texts, _, _ = _read_webui_texts(db_path, CHAT_ID)
        cleaned_flat_turns = [
            {"role": r, "content": hist_texts[f"m{i}"]} for i, (r, t) in enumerate(BRANCH)
        ]
        after_pos, after_offset = _simulate_probe_position(new_state, cleaned_flat_turns)

        assert_eq(after_pos, before_pos, "position is EXACTLY preserved (self-consistent starting anchor)")
        assert_eq(after_offset, before_offset, "window_offset is EXACTLY preserved (self-consistent starting anchor)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_anchor_rewrite_refuses_on_pre_existing_drift_unless_forced():
    """The safety net this real finding required: if the CURRENTLY STORED
    anchor is already inconsistent with a fresh branch reconstruction (a
    real, observed property of her actual 09-23 backup, unrelated to any
    text this run cleans — see the real-image suite), rewriting it to a
    freshly-recomputed value can silently change which position the next
    live request lands at. This script must refuse that rewrite by
    default and require --force to accept a bounded realignment."""
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        original_flat_turns = [{"role": r, "content": t} for r, t in BRANCH]

        # A DELIBERATELY stale anchor: primed against a branch with one
        # FEWER trailing turn than what webui.db actually has — exactly
        # the shape of drift found on the real backup (stored
        # window_turns one ahead of a fresh reconstruction).
        stale_state = summarizer._empty_state(CHAT_ID)
        summarizer._observed_position(CHAT_ID, stale_state, original_flat_turns[:-1])
        # Simulate a LONG conversation where the recorded position is far
        # ahead of this small window's own length (exactly the real
        # 09-23 backup's shape: turns_seen far exceeds a capped window) —
        # without this, `position = max(n, prev + new)` lets the small
        # window's own length dominate and MASK the "new" discrepancy this
        # test exists to catch (verified: without this line, the mismatch
        # this test is built to catch does not actually surface).
        stale_state["turns_seen"] = 1000
        summarizer.save_state(CHAT_ID, stale_state)

        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--last", "3", "--only", "webui",
            "--apply", "--json",
        ]
        rc = cd.main(argv)
        assert_eq(rc, 4, "apply with unverifiable anchor drift exits 4 (progress made, anchor refused)")
        unchanged_state = summarizer.load_state(CHAT_ID)
        assert_eq(
            unchanged_state.get("tail_fp"), stale_state.get("tail_fp"),
            "the stale anchor is left EXACTLY as it was, not silently overwritten",
        )

        # Fresh copies for the --force run: the first run above already
        # cleaned webui.db, so re-running against it would find nothing
        # left to clean and never reach the anchor decision at all.
        db_path2 = tmp / "webui2.db"
        _make_webui_db(db_path2, CHAT_ID, BRANCH)
        summarizer.save_state(CHAT_ID, stale_state)
        argv2 = [
            "--webui-db", str(db_path2), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--last", "3", "--only", "webui",
            "--apply", "--force", "--json",
        ]
        rc_forced = cd.main(argv2)
        assert_eq(rc_forced, 0, "the SAME scenario with --force accepts the realignment and exits 0")
        forced_state = summarizer.load_state(CHAT_ID)
        assert_true(
            forced_state.get("tail_fp") != stale_state.get("tail_fp"),
            "--force actually rewrote the anchor this time",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_apply_is_a_noop_second_time():
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        summarizer.save_state(CHAT_ID, summarizer._empty_state(CHAT_ID))
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--last", "3", "--only", "webui",
            "--apply", "--force", "--json",
        ]
        rc1 = cd.main(argv)
        assert_true(rc1 == 0, "first apply succeeds")
        rc2 = cd.main(argv)
        assert_eq(rc2, 0, "second apply on an already-clean branch is a no-op, exit 0")
        baks = list(tmp.glob("webui.db.bak-*"))
        assert_eq(len(baks), 1, "second apply made no NEW backup (nothing to change)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuses_on_wal_sidecar_without_force():
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        (tmp / "webui.db-wal").write_bytes(b"")
        summarizer.save_state(CHAT_ID, summarizer._empty_state(CHAT_ID))
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--only", "webui", "--json",
        ]
        rc = cd.main(argv)
        assert_eq(rc, 1, "a -wal sidecar refuses even a dry run's webui target, exit 1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuses_when_history_json_and_chat_message_table_disagree():
    """The dual-copy agreement check: if an in-scope message's text
    differs between chat.chat.history.messages and the chat_message
    table (the exact shape of the retired v1 repair script's
    double-encoding bug — see RUNBOOK_CHAT_TREE.md), this script must
    refuse rather than silently clean over the divergence."""
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        summarizer.save_state(CHAT_ID, summarizer._empty_state(CHAT_ID))
        # Corrupt the chat_message table's copy of the last assistant
        # reply (m7) so it disagrees with history.messages.
        con = sqlite3.connect(str(db_path))
        con.execute(
            "UPDATE chat_message SET content = ? WHERE id = ?",
            (json.dumps("a completely different, disagreeing copy"), f"{CHAT_ID}-m7"),
        )
        con.commit()
        con.close()
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--last", "3", "--only", "webui", "--json",
        ]
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cd.main(argv)
        report = json.loads(buf.getvalue())
        assert_eq(rc, 1, "a disagreeing chat_message row refuses the whole webui target, exit 1")
        assert_in("disagree", report.get("targets", {}).get("webui", {}).get("refused", ""),
                   "the refusal names the disagreement")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_apply_refuses_when_openwebui_port_is_open():
    _wipe_storage()
    import socket
    import threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)
    stop = threading.Event()

    def _serve():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
                conn.close()
            except socket.timeout:
                continue

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    try:
        tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
        try:
            db_path = tmp / "webui.db"
            _make_webui_db(db_path, CHAT_ID, BRANCH)
            summarizer.save_state(CHAT_ID, summarizer._empty_state(CHAT_ID))
            argv = [
                "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
                "--compactor-pkg", str(_HERE), "--only", "webui",
                "--openwebui-port", str(port), "--apply", "--json",
            ]
            rc = cd.main(argv)
            assert_eq(rc, 1, "--apply refuses when something answers the openwebui port")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        stop.set()
        srv.close()
        t.join(timeout=2)


# A longer branch: 10 already-summarized turns (5 clean exchanges) plus 2
# new ones (1 decorated exchange) — long enough for the REAL
# _resolve_resume_offset to clear its own _MIN_ANCHOR_FINGERPRINTS (8)
# floor, which a shorter fixture cannot (and correctly refuses on, which
# is its OWN test below).
LONG_BRANCH = [
    ("user", "u1 message"), ("assistant", "a1 clean reply"),
    ("user", "u2 message"), ("assistant", "a2 clean reply"),
    ("user", "u3 message"), ("assistant", "a3 clean reply"),
    ("user", "u4 message"), ("assistant", "a4 clean reply"),
    ("user", "u5 message"), ("assistant", "a5 clean reply"),
    ("user", "u6 message, brand new"),
    ("assistant", "✅ Point one\n✅ Point two\n" + "─" * 60),
]


def test_since_summarized_scope_matches_resolve_resume_offset():
    """--since-summarized must only ever touch turns strictly after the
    branch position import-history.py's own _resolve_resume_offset maps
    last_summarized_turn to — never inside the covered region (see the
    module docstring's HAZARD section)."""
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, LONG_BRANCH)
        flat_turns = [{"role": r, "content": t} for r, t in LONG_BRANCH]
        state = summarizer._empty_state(CHAT_ID)
        # The first 10 turns (5 exchanges) are already summarized into one
        # L1 chunk, with a REAL covered-fps record built the same way the
        # frozen module would, so _resolve_resume_offset has real evidence
        # to verify against — exactly import-history.py's own algorithm.
        covered = summarizer._covered_turn_fingerprints(flat_turns[:10])
        state["last_summarized_turn"] = 10
        state["covered_fps"] = "".join(covered)
        state["l1"] = [{"first_turn": 1, "last_turn": 10, "text": "..."}]
        summarizer.save_state(CHAT_ID, state)
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--since-summarized", "--l1-chunk-size", "10",
            "--only", "webui", "--json",
        ]
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cd.main(argv)
        report = {}
        try:
            report = json.loads(buf.getvalue())
        except Exception:
            pass
        detail = report.get("targets", {}).get("webui", {}).get("refused", "")
        assert_true(rc in (0, 3), f"--since-summarized dry run does not error (got {rc}): {detail[:400]}")
        preview = report.get("targets", {}).get("webui", {}).get("count")
        assert_eq(preview, 1, "--since-summarized finds exactly the one new decorated reply")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_since_summarized_refuses_without_enough_anchor_evidence():
    """A covered region shorter than _MIN_ANCHOR_FINGERPRINTS (8) cannot
    be uniquely verified — the real _resolve_resume_offset correctly
    refuses rather than guess, and this script must surface that refusal,
    never silently fall back to a flat offset (the exact `window_offset`
    trap this script's HAZARD section exists to avoid)."""
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        flat_turns = [{"role": r, "content": t} for r, t in BRANCH]
        state = summarizer._empty_state(CHAT_ID)
        covered = summarizer._covered_turn_fingerprints(flat_turns[:3])
        state["last_summarized_turn"] = 3
        state["covered_fps"] = "".join(covered)
        state["l1"] = [{"first_turn": 1, "last_turn": 3, "text": "..."}]
        summarizer.save_state(CHAT_ID, state)
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--since-summarized", "--l1-chunk-size", "3",
            "--only", "webui", "--json",
        ]
        rc = cd.main(argv)
        assert_eq(rc, 1, "too little anchor evidence refuses (exit 1), never guesses")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_facts_target_dry_run_and_apply():
    _wipe_storage()
    facts_mod.save_facts(CHAT_ID, [
        {"text": "▶️ LAW 10 – Cut Off From Humanity ✅ ACTIVE (100%)", "added_turn": 1, "last_used": 1, "pin": False},
        {"text": "She prefers plain language.", "added_turn": 2, "last_used": 1, "pin": False},
    ])
    argv_dry = ["--store", _TMP_ROOT, "--conv", CHAT_ID, "--compactor-pkg", str(_HERE),
                "--only", "facts", "--json"]
    rc = cd.main(argv_dry)
    assert_eq(rc, 3, "facts dry run with a decorated fact exits 3")

    argv_apply = argv_dry + ["--apply", "--force"]
    rc = cd.main(argv_apply)
    assert_eq(rc, 0, "facts apply succeeds")
    new_facts = facts_mod.load_facts(CHAT_ID)
    texts = [f["text"] for f in new_facts]
    assert_true(any("Cut Off From Humanity" in t and "✅" not in t for t in texts),
                "facts apply: decorated fact cleaned, real text kept")
    assert_in("She prefers plain language.", texts, "facts apply: already-clean fact untouched")

    baks = list((memory.storage_root() / "facts").glob(f"{CHAT_ID}.json.bak-*"))
    assert_true(len(baks) == 1, "facts apply: exactly one backup written")


def test_facts_never_emptied_or_deduplicated_away():
    _wipe_storage()
    facts_mod.save_facts(CHAT_ID, [
        {"text": "✅ ACTIVE (100%)", "added_turn": 1, "last_used": 1, "pin": False},
        {"text": "Same law.", "added_turn": 2, "last_used": 1, "pin": False},
        {"text": "Same law.", "added_turn": 3, "last_used": 1, "pin": False},
    ])
    new_facts, changes, empties, duplicates = cd._clean_facts(
        facts_mod.load_facts(CHAT_ID), ()
    )
    assert_true(len(empties) == 1, "a fact that would clean to empty is reported, not emptied")
    assert_eq(new_facts[0]["text"], "✅ ACTIVE (100%)", "the would-be-empty fact is left exactly as it was")
    assert_eq(len(new_facts), 3, "no fact is ever deleted by this script (no /forget semantics)")


def test_restore_roundtrip():
    _wipe_storage()
    tmp = Path(tempfile.mkdtemp(prefix="cd-cli-"))
    try:
        db_path = tmp / "webui.db"
        _make_webui_db(db_path, CHAT_ID, BRANCH)
        summarizer.save_state(CHAT_ID, summarizer._empty_state(CHAT_ID))
        before_bytes = db_path.read_bytes()
        argv = [
            "--webui-db", str(db_path), "--store", _TMP_ROOT, "--conv", CHAT_ID,
            "--compactor-pkg", str(_HERE), "--last", "3", "--only", "webui",
            "--apply", "--force", "--json",
        ]
        cd.main(argv)
        assert_true(db_path.read_bytes() != before_bytes, "apply actually changed webui.db")
        baks = list(tmp.glob("webui.db.bak-*"))
        stamp = baks[0].name.split(".bak-")[1]
        rc = cd.main(["--restore", stamp, "--webui-db", str(db_path), "--only", "webui"])
        assert_eq(rc, 0, "--restore exits 0")
        assert_eq(db_path.read_bytes(), before_bytes, "--restore brings webui.db back byte-identical")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"\n-- {t.__name__} --")
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
