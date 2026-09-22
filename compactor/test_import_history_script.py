"""Tests for scripts/import-history.py — the one-shot operator tool that
catches a conversation's hierarchical summary up from a `webui.db` EXPORT,
for a backlog `POST /admin/conversations/{conv_id}/compact` cannot reach
(that endpoint rebuilds from the episodic store, which for the real
conversation this script exists for holds 70 exchanges against ~3,850
messages — see that script's own module docstring).

Builds a synthetic `webui.db` with `sqlite3` in the test itself (no real
OpenWebUI database is available in CI). Stubs vLLM at the summarizer
boundary (`summarizer._llm_summarize`), the same seam
test_admin_compact.py uses, rather than over HTTP — this module already
degrades /tokenize failures to a pessimistic estimate, so an unreachable
`--vllm-url` costs nothing but that estimate.

No server, no model, no network:
    python test_import_history_script.py
"""

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-import-history-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"

_HERE = Path(__file__).resolve().parent
_SCRIPT_PATH = _HERE.parent / "scripts" / "import-history.py"

sys.path.insert(0, str(_HERE))
import summarizer  # noqa: E402
import memory  # noqa: E402

_spec = importlib.util.spec_from_file_location("import_history_script", _SCRIPT_PATH)
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)

VLLM_URL = "http://stub:8000"
MODEL = "test-model"

FAILED = []


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


# ---------------------------------------------------------------------------
# Synthetic webui.db builder
# ---------------------------------------------------------------------------

_DB_COUNTER = [0]


def _new_db_path() -> Path:
    _DB_COUNTER[0] += 1
    return Path(_TMP_ROOT) / f"webui-{_DB_COUNTER[0]}.db"


def _build_webui_db(path: Path, chats: dict, chat_messages: list | None = None) -> None:
    """`chats`: {chat_id: chat_dict_or_None (None -> NULL column, forcing
    the chat_message fallback)}. `chat_messages`: optional list of rows
    (id, chat_id, parent_id, role, content, created_at) for the fallback
    table, independent of what `chats` holds."""
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE chat (id TEXT PRIMARY KEY, chat TEXT, updated_at INTEGER)")
    con.execute(
        "CREATE TABLE chat_message (id TEXT PRIMARY KEY, chat_id TEXT, "
        "parent_id TEXT, role TEXT, content TEXT, created_at INTEGER)"
    )
    for chat_id, chat_obj in chats.items():
        con.execute(
            "INSERT INTO chat (id, chat, updated_at) VALUES (?, ?, ?)",
            (chat_id, json.dumps(chat_obj) if chat_obj is not None else None, 0),
        )
    for row in (chat_messages or []):
        con.execute(
            "INSERT INTO chat_message (id, chat_id, parent_id, role, content, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            row,
        )
    con.commit()
    con.close()


def _msg(mid, parent, role, content):
    return {"id": mid, "parentId": parent, "role": role, "content": content}


def _history_chat(messages: dict, current_id: str) -> dict:
    return {"history": {"messages": messages, "currentId": current_id}}


def _linear_history(n_turns: int, prefix: str = "t"):
    """n_turns alternating user/assistant messages chained by parentId.
    Returns (messages_dict, current_id)."""
    messages = {}
    parent = None
    mid = None
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        mid = f"{prefix}{i}"
        messages[mid] = _msg(mid, parent, role, f"{role} turn {i}")
        parent = mid
    return messages, mid


def run_script(argv):
    """Call the real script's main() in-process (so LLM/liveness stubs
    apply), capturing stdout. Returns (returncode, stdout_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _script.main(argv)
    return rc, buf.getvalue()


def run_script_subprocess(argv, timeout=60):
    """Run the real script as a genuinely separate process. Needed
    whenever a test substitutes a DIFFERENT `summarizer`/`memory` package
    (--compactor-pkg pointed at a stub): `summarizer` and `memory` are
    already imported at this test module's top level, so `run_script`'s
    in-process `import summarizer` inside main() would just reuse the
    already-cached REAL modules from sys.modules, never the stub."""
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)] + argv,
        capture_output=True, text=True, timeout=timeout,
    )


# ---------------------------------------------------------------------------
# vLLM stub — same seam test_admin_compact.py uses
# ---------------------------------------------------------------------------

LLM_CALLS = []


async def _fake_llm(client, vllm_url, model, system_prompt, body_text,
                     max_tokens, *, timeout=300.0):
    LLM_CALLS.append(len(body_text))
    return f"summary of {len(body_text)} chars"


_REAL_LLM_SUMMARIZE = summarizer._llm_summarize
summarizer._llm_summarize = _fake_llm


# ---------------------------------------------------------------------------
# 1. Branch walk: a FORK (an edited message) must return the CURRENT
#    branch, not insertion order.
# ---------------------------------------------------------------------------

def test_fork_returns_current_branch_not_insertion_order():
    print("\n[test] a forked history.messages returns the current branch, not insertion order")
    db = _new_db_path()
    messages = {
        "m1": _msg("m1", None, "user", "original question"),
        "m2": _msg("m2", "m1", "assistant", "original answer"),
        # m1 was edited -> a SIBLING root, created later, so insertion
        # order puts it AFTER the abandoned branch above.
        "m1b": _msg("m1b", None, "user", "edited question"),
        "m2b": _msg("m2b", "m1b", "assistant", "edited answer"),
        "m3b": _msg("m3b", "m2b", "user", "followup"),
        "m4b": _msg("m4b", "m3b", "assistant", "followup answer"),
    }
    _build_webui_db(db, {"forked-chat": _history_chat(messages, "m4b")})

    con = _script._open_ro(db)
    try:
        turns, source, notes = _script.reconstruct_transcript(con, "forked-chat")
    finally:
        con.close()

    assert_eq(source, "history.messages (chat.chat JSON)", "source is the JSON history walk")
    assert_eq(len(turns), 4, f"exactly the current branch's 4 turns (got {len(turns)})")
    assert_eq(turns[0]["content"], "edited question",
               "the branch starts at the CURRENT root, not the abandoned one")
    contents = [t["content"] for t in turns]
    assert_true("original question" not in contents,
                "the abandoned branch's content never appears")
    assert_true("original answer" not in contents,
                "the abandoned branch's reply never appears")
    assert_eq(contents, ["edited question", "edited answer", "followup", "followup answer"],
              "the branch is in root-to-leaf order")


def test_chat_message_fallback_also_follows_the_current_branch():
    print("\n[test] the chat_message fallback also follows the most-recent branch")
    db = _new_db_path()
    # chat.chat is NULL -> forces the fallback.
    rows = [
        ("m1", "fb-chat", None, "user", "original question", 100),
        ("m2", "fb-chat", "m1", "assistant", "original answer", 101),
        ("m1b", "fb-chat", None, "user", "edited question", 200),
        ("m2b", "fb-chat", "m1b", "assistant", "edited answer", 201),
    ]
    _build_webui_db(db, {"fb-chat": None}, chat_messages=rows)

    con = _script._open_ro(db)
    try:
        turns, source, notes = _script.reconstruct_transcript(con, "fb-chat")
    finally:
        con.close()

    assert_eq(source, "chat_message table", "source is the fallback table")
    assert_eq([t["content"] for t in turns], ["edited question", "edited answer"],
              "the fallback picks the most recently CREATED leaf's branch")


# ---------------------------------------------------------------------------
# 2. Multimodal content flattens; images become placeholders.
# ---------------------------------------------------------------------------

def test_multimodal_content_flattens_with_image_placeholders():
    print("\n[test] a multimodal content array flattens to text + [image] placeholders")
    content = [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        {"type": "text", "text": "and this"},
    ]
    flat = _script._flatten_content(content)
    assert_true("look at this" in flat, "first text part kept")
    assert_true("and this" in flat, "second text part kept")
    assert_true("[image]" in flat, "the image part becomes a placeholder")
    assert_true("base64" not in flat, "raw image data never leaks into the flattened text")

    assert_eq(_script._flatten_content("plain string"), "plain string",
               "a plain string passes through unchanged")
    assert_eq(_script._flatten_content(None), "", "None flattens to empty string")


# ---------------------------------------------------------------------------
# 3. Dry run: zero writes, zero vLLM calls, exit 3 when work is due.
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing_and_calls_no_model_and_exits_3():
    print("\n[test] dry run makes no writes and no vLLM calls, and exits 3 when work is due")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(24)  # > L1_CHUNK_SIZE (20) -> 1 chunk due
    _build_webui_db(db, {"dry-conv": _history_chat(messages, current_id)})

    before = _snapshot()
    LLM_CALLS.clear()
    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", "dry-conv", "--store", _TMP_ROOT,
    ])
    assert_eq(rc, 3, "dry run with work due exits 3")
    assert_eq(LLM_CALLS, [], "no LLM call was made during a dry run")
    assert_eq(_snapshot(), before, "not one byte under the storage root changed")
    assert_true("DRY RUN" in out, "the report says DRY RUN")
    assert_true(not summarizer.summary_path("dry-conv").exists(),
                "no summary state file was created")


def test_dry_run_json_reports_l1_due_and_no_work_exits_0():
    print("\n[test] --json dry run reports the estimate; nothing due exits 0")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(24)
    _build_webui_db(db, {"dry-json": _history_chat(messages, current_id)})

    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", "dry-json", "--store", _TMP_ROOT, "--json",
    ])
    assert_eq(rc, 3, "24 turns against a fresh store: 1 L1 chunk due -> exit 3")
    payload = json.loads(out)
    assert_eq(payload["turns_found"], 24, "turns_found matches the reconstructed transcript")
    assert_eq(payload["l1_chunks_due_estimate"], 1, "exactly 1 L1 chunk estimated due")
    assert_true(payload["estimated_real_vllm_calls"] >= 1, "at least 1 call estimated")
    assert_true("estimate" in payload["estimated_wall_clock"].lower(),
                "the wall-clock figure is labelled an estimate")

    # A conversation too short for even one L1 chunk: nothing due, exit 0.
    db2 = _new_db_path()
    messages2, current_id2 = _linear_history(4)
    _build_webui_db(db2, {"tiny": _history_chat(messages2, current_id2)})
    rc2, out2 = run_script([
        "--webui-db", str(db2), "--chat-id", "tiny", "--store", _TMP_ROOT, "--json",
    ])
    assert_eq(rc2, 0, "a conversation shorter than one L1 chunk has nothing due -> exit 0")
    payload2 = json.loads(out2)
    assert_eq(payload2["l1_chunks_due_estimate"], 0, "0 chunks due for a 4-turn conversation")


def _snapshot() -> dict:
    out = {}
    for dirpath, _dirs, files in os.walk(_TMP_ROOT):
        for name in files:
            p = os.path.join(dirpath, name)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, _TMP_ROOT)] = fh.read()
    return out


# ---------------------------------------------------------------------------
# 4. --apply advances last_summarized_turn and writes state load_state
#    reads back.
# ---------------------------------------------------------------------------

def test_apply_advances_watermark_and_writes_readable_state():
    print("\n[test] --apply advances last_summarized_turn and writes state load_state can read")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(24)
    _build_webui_db(db, {"apply-conv": _history_chat(messages, current_id)})

    LLM_CALLS.clear()
    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", "apply-conv", "--store", _TMP_ROOT,
        "--vllm-url", VLLM_URL, "--model", MODEL, "--apply", "--json",
    ])
    assert_eq(rc, 0, f"--apply succeeds (out={out!r})")
    assert_true(len(LLM_CALLS) > 0, "the model was actually called")
    state = summarizer.load_state("apply-conv")
    assert_eq(state["last_summarized_turn"], 20,
               f"the watermark advanced to 20 (got {state['last_summarized_turn']})")
    assert_eq(len(state["l1"]), 1, "one L1 chunk was written")
    payload = json.loads(out)
    assert_eq(payload["last_summarized_turn_after"], 20, "the report matches what is on disk")
    assert_true(payload["backup"] is None,
                "no backup was made — there was no pre-existing summary file")


def test_apply_with_nothing_due_is_a_no_op_and_idempotent():
    print("\n[test] a second --apply with nothing due does nothing and says so")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(24)
    _build_webui_db(db, {"idem-conv": _history_chat(messages, current_id)})
    argv = [
        "--webui-db", str(db), "--chat-id", "idem-conv", "--store", _TMP_ROOT,
        "--vllm-url", VLLM_URL, "--model", MODEL, "--apply", "--json",
    ]
    rc1, _ = run_script(argv)
    assert_eq(rc1, 0, "first --apply succeeds")
    before = _snapshot()

    rc2, out2 = run_script(argv)
    assert_eq(rc2, 0, "second --apply (nothing left due) exits 0")
    assert_eq(_snapshot(), before, "the second --apply changed nothing on disk")
    payload2 = json.loads(out2)
    assert_true("nothing due" in payload2.get("note", ""),
                "the report says nothing was due")


# ---------------------------------------------------------------------------
# 5. max_calls budget respected, with the documented one-unit overshoot.
# ---------------------------------------------------------------------------

def test_max_calls_budget_stops_the_loop_with_documented_overshoot():
    print("\n[test] max_calls bounds real vLLM calls, overshooting by at most one unit")
    _wipe_storage()

    # A stand-in maybe_rollup where ONE call to it (one "unit") spends THREE
    # real vLLM calls against the budget contextvar — the shape a chunk
    # needing map-reduce has in the real drain — so max_calls=1 must still
    # let the unit finish (the documented guarantee) and show remaining
    # go negative (the documented overshoot), never start a second unit.
    calls_to_maybe_rollup = [0]

    async def _one_unit_costs_three_calls(conv_id, messages, vllm_url, model):
        calls_to_maybe_rollup[0] += 1
        budget = summarizer._vllm_call_budget.get()
        if budget is not None:
            budget["remaining"] -= 3
        st = summarizer.load_state(conv_id)
        st["last_summarized_turn"] = st.get("last_summarized_turn", 0) + 20
        summarizer.save_state(conv_id, st)
        return st

    conv_id = "budget-conv"
    result = asyncio.run(
        _script._run_apply_loop(
            _StubModule(_one_unit_costs_three_calls), memory,
            conv_id, [], VLLM_URL, MODEL, 1,
        )
    )
    assert_eq(calls_to_maybe_rollup[0], 1,
               "exactly one unit ran — the budget check is at the unit boundary")
    assert_eq(result["vllm_calls_spent"], 3,
               f"3 real calls were spent for max_calls=1 (got {result['vllm_calls_spent']})")
    assert_true(result["vllm_calls_spent"] > 1,
                "the overshoot is real: spent MORE than max_calls, never less")
    assert_true("max_calls=1" in str(result["stopped_because"]),
                f"stopped_because names the cap (got {result['stopped_because']!r})")
    assert_eq(summarizer.load_state(conv_id)["last_summarized_turn"], 20,
               "the one unit that started was allowed to finish")


class _StubModule:
    """Wraps a fake maybe_rollup so _run_apply_loop's
    `summarizer.maybe_rollup(...)` call reaches it, while every other
    attribute (load_state, save_state, vllm_call_budget_ctx, conv_lock use
    via `memory`) still goes to the real summarizer module — matching how
    the real drain only ever swaps out the rollup call itself in
    test_admin_compact.py's own [5c]."""

    def __init__(self, fake_maybe_rollup):
        self._fake = fake_maybe_rollup

    def __getattr__(self, name):
        return getattr(summarizer, name)

    async def maybe_rollup(self, conv_id, messages, vllm_url, model):
        return await self._fake(conv_id, messages, vllm_url, model)


# ---------------------------------------------------------------------------
# 6. Interrupting --apply leaves valid state; re-running resumes.
# ---------------------------------------------------------------------------

def test_interrupted_apply_leaves_valid_state_and_resumes():
    print("\n[test] a failure mid-loop leaves valid state, and re-running resumes without redoing work")
    _wipe_storage()
    db = _new_db_path()
    # 44 turns: two L1 chunks due (20 + 20), 4 turns left over.
    messages, current_id = _linear_history(44)
    _build_webui_db(db, {"interrupt-conv": _history_chat(messages, current_id)})

    call_count = [0]

    async def _fails_from_second_call_on(client, vllm_url, model, system_prompt,
                                          body_text, max_tokens, *, timeout=300.0):
        # Fails on the 2nd call AND every call after, for the life of this
        # stub — a ONE-SHOT failure is not enough to test "interrupted": the
        # real drain retries a failed unit fresh on the very next pass, so
        # a transient failure here would self-heal within the same --apply
        # run (chunk2 would succeed on retry) rather than leaving the run
        # actually interrupted. A persistent failure is what forces this
        # --apply run to stop after chunk 1.
        call_count[0] += 1
        if call_count[0] >= 2:
            raise RuntimeError("simulated persistent vLLM failure")
        return f"summary of {len(body_text)} chars"

    summarizer._llm_summarize = _fails_from_second_call_on
    try:
        rc, out = run_script([
            "--webui-db", str(db), "--chat-id", "interrupt-conv", "--store", _TMP_ROOT,
            "--vllm-url", VLLM_URL, "--model", MODEL, "--apply", "--json",
        ])
    finally:
        summarizer._llm_summarize = _fake_llm

    assert_eq(rc, 0, "a run that stopped on an LLM failure is still a completed --apply")
    payload = json.loads(out)
    assert_true(payload["stopped_because"], "the report names why it stopped")

    state_after_failure = summarizer.load_state("interrupt-conv")
    assert_eq(state_after_failure["last_summarized_turn"], 20,
               "exactly the first chunk's worth of progress survived the failure")
    first_chunk_text = state_after_failure["l1"][0]["text"]

    # Resume: LLM is healthy again, and a fresh backup collision must not
    # be an issue this second run creates a NEW dated backup for the file
    # that already exists (its own backup exists check is per-run, on the
    # state file, not on the backup itself here — nothing to collide with
    # since the second run's stamp differs).
    rc2, out2 = run_script([
        "--webui-db", str(db), "--chat-id", "interrupt-conv", "--store", _TMP_ROOT,
        "--vllm-url", VLLM_URL, "--model", MODEL, "--apply", "--json",
    ])
    assert_eq(rc2, 0, "the resumed --apply succeeds")
    state_after_resume = summarizer.load_state("interrupt-conv")
    assert_eq(state_after_resume["last_summarized_turn"], 40,
               f"the resume advanced past the first chunk to 40 "
               f"(got {state_after_resume['last_summarized_turn']})")
    assert_eq(state_after_resume["l1"][0]["text"], first_chunk_text,
               "the already-finished first chunk was NOT redone or altered")
    assert_true(len(state_after_resume["l1"]) >= 2,
                "a second chunk was added rather than the first being replaced")


# ---------------------------------------------------------------------------
# 7. Refusal paths.
# ---------------------------------------------------------------------------

def test_refuses_apply_while_compactor_is_live_unless_forced():
    print("\n[test] --apply refuses while the compactor answers /health, unless --force")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(24)
    _build_webui_db(db, {"live-conv": _history_chat(messages, current_id)})

    real_alive = _script._compactor_is_alive
    _script._compactor_is_alive = lambda url=_script.DEFAULT_HEALTH_URL: True
    try:
        before = _snapshot()
        rc, out = run_script([
            "--webui-db", str(db), "--chat-id", "live-conv", "--store", _TMP_ROOT,
            "--vllm-url", VLLM_URL, "--model", MODEL, "--apply",
        ])
        assert_eq(rc, 1, "refused while the compactor looks alive")
        assert_true("REFUSING" in out, "the refusal names itself")
        assert_eq(_snapshot(), before, "nothing was written on the refusal")

        rc2, out2 = run_script([
            "--webui-db", str(db), "--chat-id", "live-conv", "--store", _TMP_ROOT,
            "--vllm-url", VLLM_URL, "--model", MODEL, "--apply", "--force",
        ])
        assert_eq(rc2, 0, "--force overrides the live-compactor refusal")
        assert_true("WARNING" in out2, "the override is a loud warning, not silent")
    finally:
        _script._compactor_is_alive = real_alive


def test_refuses_when_backup_path_already_exists():
    print("\n[test] --apply refuses if the backup path already exists")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(24)
    _build_webui_db(db, {"backup-conv": _history_chat(messages, current_id)})

    # Seed an existing summary file so the FIRST apply has something to
    # back up, at a stamp we pin.
    summarizer.save_state("backup-conv", {
        "l1": [{"text": "pre-existing", "first_turn": 1, "last_turn": 4}],
        "l2": [], "l3": None, "last_summarized_turn": 4,
    })
    fixed_stamp = "20260101T000000Z"
    argv = [
        "--webui-db", str(db), "--chat-id", "backup-conv", "--store", _TMP_ROOT,
        "--vllm-url", VLLM_URL, "--model", MODEL, "--apply", "--json",
    ]

    real_stamp = _script._utc_stamp
    _script._utc_stamp = lambda: fixed_stamp
    try:
        rc1, out1 = run_script(argv)
        assert_eq(rc1, 0, "the first --apply (with a backup to make) succeeds")
        backup_path = summarizer.summary_path("backup-conv").with_name(
            summarizer.summary_path("backup-conv").name + f".bak-{fixed_stamp}"
        )
        assert_true(backup_path.is_file(), "the backup file was created at the pinned stamp")

        # Put the file back the way it was (a fresh pre-existing summary,
        # same content) WITHOUT touching the backup, to force the collision.
        summarizer.save_state("backup-conv", {
            "l1": [{"text": "pre-existing", "first_turn": 1, "last_turn": 4}],
            "l2": [], "l3": None, "last_summarized_turn": 4,
        })
        backup_bytes_before = backup_path.read_bytes()

        rc2, out2 = run_script(argv)
        assert_eq(rc2, 1, "a second --apply at the same stamp refuses")
        assert_true("REFUSING" in out2 and "backup" in out2.lower(),
                    "the refusal names the backup collision")
        assert_eq(backup_path.read_bytes(), backup_bytes_before,
                   "the existing backup was not overwritten")
    finally:
        _script._utc_stamp = real_stamp


def test_refuses_on_journal_beside_the_database_unless_forced():
    print("\n[test] a -journal file beside the database refuses unless --force")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(4)
    _build_webui_db(db, {"journal-conv": _history_chat(messages, current_id)})
    journal = db.with_name(db.name + "-journal")
    # Empty, not garbage: an empty -journal is what SQLite itself treats
    # as "no journal" once opened, so --force's forced read-only open
    # still succeeds — this test is about the REFUSAL seeing the sidecar
    # and existing, not about surviving a truly hot/corrupt journal (which
    # a read-only open cannot roll back and would fail regardless of
    # --force, correctly).
    journal.touch()

    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", "journal-conv", "--store", _TMP_ROOT,
    ])
    assert_eq(rc, 1, "refused without --force")
    assert_true("REFUSING" in out and "journal" in out.lower(),
                "the refusal names the journal sidecar")

    rc2, out2 = run_script([
        "--webui-db", str(db), "--chat-id", "journal-conv", "--store", _TMP_ROOT, "--force",
    ])
    assert_eq(rc2, 0, "--force overrides the journal refusal (dry run still completes)")
    assert_true("WARNING" in out2, "the override warns rather than staying silent")


def test_unknown_chat_id_is_an_error():
    print("\n[test] an unknown --chat-id is a clean error, not a crash")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(4)
    _build_webui_db(db, {"known-conv": _history_chat(messages, current_id)})

    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", "does-not-exist", "--store", _TMP_ROOT,
    ])
    assert_eq(rc, 1, "an unknown chat id is exit 1")
    assert_true("no chat" in out.lower() or "does-not-exist" in out,
                "the error names the missing chat id")


def test_unknown_arguments_are_rejected():
    print("\n[test] an unrecognised argument is rejected (argparse exit 2)")
    try:
        _script._build_argparser().parse_args([
            "--webui-db", "x", "--chat-id", "y", "--not-a-real-flag",
        ])
        assert_true(False, "parse_args should have raised SystemExit")
    except SystemExit as e:
        assert_eq(e.code, 2, "argparse's own usage-error exit code")


# ---------------------------------------------------------------------------
# 8. Anomalies are reported, not crashed on.
# ---------------------------------------------------------------------------

def test_anomalous_transcript_reported_not_crashed():
    print("\n[test] a missing reply and consecutive same-role turns are reported, not crashed on")
    _wipe_storage()
    db = _new_db_path()
    # user, assistant, user, user (no reply), and the transcript ends on a
    # user turn — two anomalies in four turns.
    messages = {
        "a": _msg("a", None, "user", "hello"),
        "b": _msg("b", "a", "assistant", "hi"),
        "c": _msg("c", "b", "user", "question one"),
        "d": _msg("d", "c", "user", "question two, no reply came"),
    }
    _build_webui_db(db, {"anomaly-conv": _history_chat(messages, "d")})

    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", "anomaly-conv", "--store", _TMP_ROOT, "--json",
    ])
    assert_true(rc in (0, 3), f"a short anomalous transcript still exits cleanly (got {rc})")
    payload = json.loads(out)
    assert_eq(payload["turns_found"], 4, "all four turns were still reconstructed")
    joined = " | ".join(payload["anomalies"])
    assert_true("consecutive" in joined, "the consecutive user turns are flagged")
    assert_true("no assistant reply" in joined, "the missing final reply is flagged")


# ---------------------------------------------------------------------------
# 9. Defect 1: package resolution. Same mechanism as
#    scripts/backfill-records.py's own _resolve_compactor_pkg (see that
#    script's tests for the twin coverage); compactor/
#    test_real_image_operator_scripts.py exercises the real fallback and
#    real multi-path-failure cases against the actual published image.
# ---------------------------------------------------------------------------

def test_resolve_compactor_pkg_explicit_flag_wins_over_the_repo_layout():
    print("\n[test] --compactor-pkg wins even when the repo-layout package also exists")
    with tempfile.TemporaryDirectory() as td:
        real_pkg = Path(td) / "explicit-pkg"
        real_pkg.mkdir()
        here = Path(td) / "repo" / "scripts"
        here.mkdir(parents=True)
        (here.parent / "compactor").mkdir()
        with patch.object(_script, "HERE", here):
            pkg, tried, already = _script._resolve_compactor_pkg(str(real_pkg))
        assert_eq(pkg, real_pkg, "the explicit --compactor-pkg path wins")
        assert_true("--compactor-pkg" in tried[0], "the explicit path is listed first, and labelled")


def test_resolve_compactor_pkg_falls_back_to_opt_compactor():
    print("\n[test] with no repo-layout package (the /data/scripts/ shape), resolution falls through to /opt/compactor")
    with tempfile.TemporaryDirectory() as td:
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
        assert_true(pkg is None, "no package directory found")
        assert_eq(len(tried), 3,
                   f"repo-layout, /opt/compactor, and the sys.path probe were all tried (got {tried!r})")


def test_reports_which_pkg_source_was_used():
    print("\n[test] the report names which compactor package source was resolved")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(4)
    _build_webui_db(db, {"pkg-report-conv": _history_chat(messages, current_id)})
    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", "pkg-report-conv", "--store", _TMP_ROOT, "--json",
    ])
    payload = json.loads(out)
    assert_true(
        any("compactor package resolved from" in w for w in payload["warnings"]),
        f"a NOTE names the resolved package source (warnings={payload['warnings']!r})",
    )


# ---------------------------------------------------------------------------
# 10. Defect 2: a package missing what this script needs refuses with an
#     actionable message, never a raw AttributeError.
# ---------------------------------------------------------------------------

def test_capability_check_refuses_cleanly_on_a_package_missing_required_symbols():
    print("\n[test] a --compactor-pkg missing required symbols refuses, never crashes")
    _wipe_storage()
    db = _new_db_path()
    messages, current_id = _linear_history(4)
    _build_webui_db(db, {"stub-pkg-conv": _history_chat(messages, current_id)})
    with tempfile.TemporaryDirectory() as td:
        stub_pkg = Path(td) / "compactor"
        stub_pkg.mkdir()
        # Present: load_state (this script calls it before the capability
        # check). Absent: maybe_rollup and the Defect-3 fingerprint
        # primitives this script's own content guard depends on.
        (stub_pkg / "summarizer.py").write_text(
            "def load_state(conv_id):\n"
            "    return {}\n"
            "def recorded_position(state):\n"
            "    return 0\n",
            encoding="utf-8",
        )
        (stub_pkg / "memory.py").write_text("", encoding="utf-8")
        r = run_script_subprocess([
            "--webui-db", str(db), "--chat-id", "stub-pkg-conv",
            "--store", _TMP_ROOT, "--compactor-pkg", str(stub_pkg),
        ])
    out = r.stdout + r.stderr
    assert_eq(r.returncode, 1, f"refuses with exit 1, not a crash (out={out!r})")
    assert_true("maybe_rollup" in out or "conv_lock" in out, "names at least one missing symbol")
    assert_true("Traceback" not in out, "no raw traceback")
    assert_true("git clone" in out, "prints the clone command")


# ---------------------------------------------------------------------------
# 11. Defect 3: the guard checks CONTENT, not turn COUNT, against the
#     store. A monotonic recorded_position can legitimately exceed the
#     CURRENT linear branch's turn count after an edit or an abandoned
#     regeneration (summarizer._observed_position's own invariant I2) —
#     that must proceed. A genuinely different conversation's export must
#     still refuse.
# ---------------------------------------------------------------------------

def _fp_turns(n, prefix="p"):
    """n alternating {"role","content"} dicts, content unique per index —
    the same shape reconstruct_transcript produces."""
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"{prefix} {role} turn {i}"})
    return out


def test_prefix_matches_store_true_when_anchor_found_despite_shorter_reconstruction():
    print("\n[test] _prefix_matches_store: content match proceeds even though current_turns < recorded_position")
    # The store last saw a 50-turn window (tail_fp = its last 4 turns).
    long_ago = _fp_turns(50)
    tail_fp = summarizer._turn_fingerprints(long_ago)[-4:]
    state = {"tail_fp": tail_fp}
    # The CURRENT reconstruction is only 24 turns — shorter than
    # recorded_position would be (50) — because something in the middle
    # was edited/abandoned. But its own tail is BYTE-IDENTICAL to the
    # anchor's content (the same real-world shape verified against the
    # 2026-09-22 backup: the abandoned edits sat earlier in the
    # conversation, not at the tip).
    current = long_ago[:20] + long_ago[-4:]
    matches, detail = _script._prefix_matches_store(state, current, summarizer)
    assert_eq(matches, True, "the anchor was found — content confirmed")
    assert_true("confirmed" in detail, f"detail explains the match (got {detail!r})")


def test_prefix_matches_store_false_on_a_genuinely_different_conversation():
    print("\n[test] _prefix_matches_store: refuses when the anchor is not anywhere in the reconstruction")
    long_ago = _fp_turns(50, prefix="real-conv")
    tail_fp = summarizer._turn_fingerprints(long_ago)[-4:]
    state = {"tail_fp": tail_fp}
    unrelated = _fp_turns(30, prefix="totally-different-conversation")
    matches, detail = _script._prefix_matches_store(state, unrelated, summarizer)
    assert_eq(matches, False, "the anchor is nowhere in an unrelated conversation's turns")
    assert_true("do not appear" in detail, f"detail explains the mismatch (got {detail!r})")


def test_prefix_matches_store_none_when_store_has_no_anchor_yet():
    print("\n[test] _prefix_matches_store: no evidence either way when tail_fp is empty")
    matches, detail = _script._prefix_matches_store({"tail_fp": []}, _fp_turns(10), summarizer)
    assert_true(matches is None, "no anchor recorded yet -> no verdict, caller falls back to length")


def test_apply_history_shorter_than_recorded_position_but_content_matches_is_not_refused():
    print("\n[test] end to end: a shorter-but-content-matching reconstruction is NOT refused")
    _wipe_storage()
    conv_id = "edited-history-conv"
    long_ago = _fp_turns(50, prefix=conv_id)
    tail_fp = summarizer._turn_fingerprints(long_ago)[-4:]
    # A real store state: turns_seen (and so recorded_position) at 50,
    # nothing summarized yet (last_summarized_turn=0) so there IS work
    # due once this proceeds — proving the run does real work, not just
    # skip past the guard into a no-op.
    summarizer.save_state(conv_id, {
        "l1": [], "l2": [], "l3": None,
        "last_summarized_turn": 0, "turns_seen": 50,
        "tail_fp": tail_fp, "head_fp": "", "window_turns": 0,
    })
    current = long_ago[:20] + long_ago[-4:]  # 24 turns — shorter than 50
    messages_dict, current_id = {}, None
    parent = None
    for i, t in enumerate(current):
        mid = f"m{i}"
        messages_dict[mid] = _msg(mid, parent, t["role"], t["content"])
        parent = mid
        current_id = mid
    db = _new_db_path()
    _build_webui_db(db, {conv_id: _history_chat(messages_dict, current_id)})

    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", conv_id, "--store", _TMP_ROOT, "--json",
    ])
    assert_true(rc != 1, f"NOT refused despite turns_found < recorded_position (rc={rc}, out={out!r})")
    payload = json.loads(out)
    assert_eq(payload["turns_found"], 24, "the shorter reconstruction was still used")
    assert_eq(payload["recorded_position_before"], 50, "recorded_position really is above turns_found")
    assert_true(payload["turns_found"] < payload["recorded_position_before"],
                "confirms this is exactly the shape the old length-only guard refused")


def test_apply_history_from_a_different_conversation_is_still_refused():
    print("\n[test] end to end: a genuinely wrong export is still refused")
    _wipe_storage()
    conv_id = "wrong-export-conv"
    long_ago = _fp_turns(50, prefix=conv_id)
    tail_fp = summarizer._turn_fingerprints(long_ago)[-4:]
    summarizer.save_state(conv_id, {
        "l1": [], "l2": [], "l3": None,
        "last_summarized_turn": 0, "turns_seen": 50,
        "tail_fp": tail_fp, "head_fp": "", "window_turns": 0,
    })
    # An export of a completely different conversation, reconstructed
    # under the SAME --conv-id (e.g. a wrong --chat-id was given).
    unrelated, unrelated_id = _linear_history(10, prefix="unrelated")
    db = _new_db_path()
    _build_webui_db(db, {"actually-a-different-chat": _history_chat(unrelated, unrelated_id)})

    rc, out = run_script([
        "--webui-db", str(db), "--chat-id", "actually-a-different-chat",
        "--conv-id", conv_id, "--store", _TMP_ROOT,
    ])
    assert_eq(rc, 1, "refused")
    assert_true("REFUSING" in out, "the refusal names itself")
    assert_true("do not appear" in out, "the refusal explains the content mismatch")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_fork_returns_current_branch_not_insertion_order()
        test_chat_message_fallback_also_follows_the_current_branch()

        test_multimodal_content_flattens_with_image_placeholders()

        test_dry_run_writes_nothing_and_calls_no_model_and_exits_3()
        test_dry_run_json_reports_l1_due_and_no_work_exits_0()

        test_apply_advances_watermark_and_writes_readable_state()
        test_apply_with_nothing_due_is_a_no_op_and_idempotent()

        test_max_calls_budget_stops_the_loop_with_documented_overshoot()

        test_interrupted_apply_leaves_valid_state_and_resumes()

        test_refuses_apply_while_compactor_is_live_unless_forced()
        test_refuses_when_backup_path_already_exists()
        test_refuses_on_journal_beside_the_database_unless_forced()
        test_unknown_chat_id_is_an_error()
        test_unknown_arguments_are_rejected()

        test_anomalous_transcript_reported_not_crashed()

        test_resolve_compactor_pkg_explicit_flag_wins_over_the_repo_layout()
        test_resolve_compactor_pkg_falls_back_to_opt_compactor()
        test_resolve_compactor_pkg_reports_every_path_tried_when_none_exist()
        test_reports_which_pkg_source_was_used()

        test_capability_check_refuses_cleanly_on_a_package_missing_required_symbols()

        test_prefix_matches_store_true_when_anchor_found_despite_shorter_reconstruction()
        test_prefix_matches_store_false_on_a_genuinely_different_conversation()
        test_prefix_matches_store_none_when_store_has_no_anchor_yet()
        test_apply_history_shorter_than_recorded_position_but_content_matches_is_not_refused()
        test_apply_history_from_a_different_conversation_is_still_refused()

        print("\nAll import-history.py script tests passed.")
    finally:
        summarizer._llm_summarize = _REAL_LLM_SUMMARIZE
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
