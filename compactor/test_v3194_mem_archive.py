"""
CPU-only tests for v3.1.9.4 (P15-3): the L2 chapter cold store
(summaries/<id>.archive.json) is now a memory layer wipe, verify, retire and
import all know about.

Built the same way the reviewer's SP\\p15\\p15_d1_chapter_archive.py
reproduction was: chapters written by the real writer
(summarizer._archive_chapters, what _do_l3_rollup calls), everything else
driven through the real routes (FastAPI TestClient) or the real command
dispatcher. Synthetic text only (no real user data — this repo is public).

Covers:
  - chat /forget deletes the chapter archive and reports it, and the
    verification pass (_memory_residue) actually finds nothing left
    (CONTROL: a conversation with no chapters at all still forgets cleanly).
  - DELETE /admin/conversations/<id>/facts does the same.
  - selftest._conv_artifact_paths / _purge_conv_files cover the chapter file
    (the count regression is in test_selftest.py; this file adds the
    residue-after-forget angle specific to this finding).
  - an overwrite import clears the TARGET's old chapters, facts archive and
    persona instead of leaving them beside the newly-imported state, while
    quarantine_conversation's pre-overwrite snapshot still preserves them
    (CONTROL: overwriting a target with none of those three layers is a
    no-op for all three, not an error).
  - /retire snapshots and clears the SOURCE's chapter archive too, and a
    conversation whose only memory is an orphaned chapter archive (no
    current summary state) is not reported as "nothing to retire".

Run: python test_v3194_mem_archive.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-mem-archive-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import commands  # noqa: E402
import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
import portability  # noqa: E402
import selftest  # noqa: E402
import summarizer  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

memory.ensure_storage_layout()
client = TestClient(main.app, client=("127.0.0.1", 12399), raise_server_exceptions=False)


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


def _wipe():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP)
    memory.ensure_storage_layout()


def _chapters(tag, n=3):
    return [
        {"text": f"{tag} chapter {i} synthetic summary text.", "first_turn": 1 + 200 * i, "last_turn": 200 * (i + 1)}
        for i in range(n)
    ]


def _n_chapters(conv_id):
    p = memory.summary_archive_path(conv_id)
    return len(summarizer.load_chapter_archive(conv_id)) if p.is_file() else 0


def _seed_full(conv_id, tag):
    """Every layer a wipe/retire/import touches, so a test can tell 'cleared'
    from 'never had one'."""
    now = int(time.time())
    facts.save_facts(conv_id, [{"text": f"{tag} fact one", "added_turn": 2, "last_used": now}])
    facts.archive_facts(conv_id, [{"text": f"{tag} archived fact", "added_turn": 1, "last_used": now - 99}])
    st = summarizer._empty_state(conv_id)
    st["l1"] = [{"text": f"{tag} l1 text", "first_turn": 601, "last_turn": 620}]
    st["l3"] = {"text": f"{tag} story so far", "first_turn": 1, "last_turn": 600}
    st["last_summarized_turn"] = 620
    st["turns_seen"] = 630
    summarizer.save_state(conv_id, st)
    summarizer._archive_chapters(conv_id, _chapters(tag))
    persona.save_persona(conv_id, f"{tag} persona text long enough to be a persona " * 3, source="admin")


# ---------------------------------------------------------------------------
# /forget (chat command) and the admin DELETE twin
# ---------------------------------------------------------------------------

def test_chat_forget_deletes_chapters_and_reports_them():
    print("\n[test] chat /forget deletes the chapter archive and says so, and residue verification finds nothing left")
    _wipe()
    cid = "archv-forget"
    _seed_full(cid, "F")
    assert_eq(_n_chapters(cid), 3, "seeded 3 chapters via the real writer")

    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "stream": False, "messages": [{"role": "user", "content": "/forget"}]},
        headers={"X-Conversation-Id": cid},
    )
    reply = r.json()["choices"][0]["message"]["content"]
    assert_eq(_n_chapters(cid), 0, "the chapter archive file is gone after /forget")
    assert_true("chapter archive" in reply, f"the reply names the chapter archive as forgotten: {reply!r}")
    still, unreadable = commands._memory_residue(cid)
    assert_eq(still, [], "the verification pass finds nothing left (not blind to a layer that is genuinely gone)")
    assert_eq(unreadable, [], "nothing unreadable")


def test_chat_forget_on_conversation_with_no_chapters_is_unaffected():
    print("\n[test] CONTROL: /forget on a conversation with no chapter archive at all behaves exactly as before")
    _wipe()
    cid = "archv-forget-none"
    facts.save_facts(cid, [{"text": "a lone fact", "added_turn": 1, "last_used": int(time.time())}])
    assert_eq(_n_chapters(cid), 0, "no chapters were ever written for this conversation")

    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "stream": False, "messages": [{"role": "user", "content": "/forget"}]},
        headers={"X-Conversation-Id": cid},
    )
    reply = r.json()["choices"][0]["message"]["content"]
    assert_true("chapter archive" not in reply, f"a conversation that never had chapters does not claim to have forgotten any: {reply!r}")
    assert_true("fact(s)" in reply, f"the fact is still reported forgotten: {reply!r}")


def test_admin_delete_facts_also_clears_chapters():
    print("\n[test] DELETE /admin/conversations/<id>/facts clears the chapter archive too (same chokepoint as /forget)")
    _wipe()
    cid = "archv-admin-delete"
    _seed_full(cid, "D")
    assert_eq(_n_chapters(cid), 3, "seeded 3 chapters")
    r = client.request("DELETE", f"/admin/conversations/{cid}/facts")
    assert_eq(r.status_code, 200, "the admin wipe succeeded")
    assert_eq(_n_chapters(cid), 0, "chapters are gone after the admin wipe too")
    assert_true(bool(r.json().get("forgotten_chapters")), "the response says the chapter layer was cleared")


# ---------------------------------------------------------------------------
# selftest enumeration
# ---------------------------------------------------------------------------

def test_selftest_paths_include_and_purge_the_chapter_archive():
    print("\n[test] selftest._conv_artifact_paths carries the chapter archive, and _purge_conv_files removes it")
    _wipe()
    cid = "archv-selftest"
    summarizer._archive_chapters(cid, _chapters("S"))
    paths = selftest._conv_artifact_paths(cid)
    assert_true(memory.summary_archive_path(cid) in paths, "the chapter archive path is one of the tracked artifacts")
    assert_true(memory.summary_archive_path(cid).is_file(), "it is really on disk before the purge")
    left = selftest._purge_conv_files(cid)
    assert_eq(left, [], "nothing reported as surviving the purge")
    assert_true(not memory.summary_archive_path(cid).exists(), "the chapter archive is gone after the purge")


# ---------------------------------------------------------------------------
# Import overwrite
# ---------------------------------------------------------------------------

def test_import_overwrite_clears_old_target_layers_but_preserves_them_in_quarantine():
    print("\n[test] an overwrite import clears the TARGET's old chapters/archive/persona, and the pre-overwrite snapshot still has them")
    _wipe()
    src, dst = "archv-import-src", "archv-import-dst"
    _seed_full(src, "SRC")
    _seed_full(dst, "DST")
    bundle = portability.export_conversation(src)

    r = client.post(
        "/admin/conversations/import",
        json={"bundle": bundle, "target_conv_id": dst, "overwrite": True},
    )
    assert_eq(r.status_code, 200, f"the overwrite import succeeded: {r.text[:300]}")

    dst_state = summarizer.load_state(dst)
    assert_true(dst_state["l3"]["text"].startswith("SRC "), "CONTROL: the import did replace the target's hierarchy with the source's")

    dst_chapters = summarizer.load_chapter_archive(dst)
    assert_true(
        all(c["text"].startswith("SRC ") for c in dst_chapters) if dst_chapters else True,
        "the target's chapter archive holds only the fresh state — none of DST's old chapters remain mixed in",
    )
    assert_true(
        not any(c["text"].startswith("DST ") for c in dst_chapters),
        "specifically: the OLD target's 3 chapters are gone, not just outnumbered",
    )
    dst_arch = facts.load_archive(dst)
    assert_true(
        not any(f["text"].startswith("DST ") for f in dst_arch),
        "the old target's archived fact is gone from the live facts archive",
    )
    assert_true(
        not (persona.get_persona_text(dst) or "").startswith("DST "),
        "the old target's persona is gone (import itself carries no persona, so this reads as None/empty, not DST's)",
    )

    # And none of that is actually LOST — the pre-overwrite quarantine
    # snapshot (main.admin_import_conversation takes one whenever overwrite
    # is set) still has the old target's state.
    snaps = portability.list_quarantine(dst)
    assert_true(bool(snaps), "a quarantine snapshot of the old target was written before the overwrite")
    import json
    snap_bundle = json.loads(snaps[-1].read_bytes())
    q = snap_bundle["quarantine"]
    assert_true(
        any(c["text"].startswith("DST ") for c in q["chapters"]),
        "...and it holds the OLD target's chapters, recoverable with summarizer._archive_chapters",
    )
    assert_true(
        any(f["text"].startswith("DST ") for f in q["archive"]),
        "...and the old target's archived fact",
    )
    assert_true(
        (q["persona"] or {}).get("persona_text", "").startswith("DST "),
        "...and the old target's persona",
    )


def test_import_overwrite_with_no_old_target_layers_is_a_clean_no_op():
    print("\n[test] CONTROL: overwriting a target that never had chapters/archive/persona touches none of them")
    _wipe()
    src, dst = "archv-import-src2", "archv-import-dst2"
    _seed_full(src, "SRC2")
    facts.save_facts(dst, [{"text": "bare target fact", "added_turn": 1, "last_used": int(time.time())}])
    bundle = portability.export_conversation(src)

    r = client.post(
        "/admin/conversations/import",
        json={"bundle": bundle, "target_conv_id": dst, "overwrite": True},
    )
    assert_eq(r.status_code, 200, f"the overwrite import succeeded: {r.text[:300]}")
    assert_eq(facts.load_archive(dst), [], "no facts archive existed and none exists now")
    assert_eq(summarizer.load_chapter_archive(dst), [], "no chapter archive existed and none exists now")
    assert_true(persona.get_persona_text(dst) is None, "no persona existed and none exists now")


# ---------------------------------------------------------------------------
# /retire
# ---------------------------------------------------------------------------

def _retire(arg, cid):
    return asyncio.run(commands.handle_command("retire", arg, cid, ctx={}))


def _retire_code(out, source):
    marker = f"/retire {source} apply "
    assert_true(marker in out, f"dry run offered a confirmation code: {out[-400:]}")
    return out.split(marker, 1)[1].split()[0]


def test_retire_dry_run_names_and_apply_clears_the_chapter_archive():
    print("\n[test] /retire's plan names the chapter archive, and applying it snapshots + clears the source's chapters")
    _wipe()
    src, dst = "archv-retire-src", "archv-retire-dst"
    _seed_full(src, "R")
    dry = _retire(src, dst)
    assert_true("chapter archive" in dry, f"the dry-run plan names the chapter archive layer: {dry[:600]!r}")

    applied = _retire(f"{src} apply {_retire_code(dry, src)}", dst)
    assert_true("Retired" in applied, f"the apply succeeded: {applied[:200]!r}")
    assert_eq(_n_chapters(src), 0, "the source's chapter archive is cleared after retire")

    snaps = portability.list_quarantine(src)
    assert_true(bool(snaps), "a quarantine snapshot of the source was written")
    import json
    q = json.loads(snaps[-1].read_bytes())["quarantine"]
    assert_true(any(c["text"].startswith("R ") for c in q["chapters"]), "...and it holds the source's chapters")


def test_retire_refuses_only_when_truly_empty_even_with_an_orphaned_chapter_archive():
    print("\n[test] a conversation whose ONLY memory is an orphaned chapter archive (no current summary state) is not 'nothing to retire'")
    _wipe()
    src, dst = "archv-retire-orphan", "archv-retire-dst2"
    # Write chapters directly WITHOUT going through _do_l3_rollup, so there is
    # no current l1/l2/l3 state at all — the shape a torn wipe or a partial
    # delete failure could leave behind.
    summarizer._archive_chapters(src, _chapters("ORPH"))
    assert_eq(summarizer.load_state(src).get("l3"), None, "no current summary state exists for this conv")
    dry = _retire(src, dst)
    assert_true(
        "nothing to retire" not in dry.lower(),
        f"an orphaned chapter archive is still real memory — it must not read as empty: {dry[:300]!r}",
    )
    assert_true("chapter archive" in dry, f"the plan reports the chapter layer: {dry[:600]!r}")


def main_():
    tests = [
        test_chat_forget_deletes_chapters_and_reports_them,
        test_chat_forget_on_conversation_with_no_chapters_is_unaffected,
        test_admin_delete_facts_also_clears_chapters,
        test_selftest_paths_include_and_purge_the_chapter_archive,
        test_import_overwrite_clears_old_target_layers_but_preserves_them_in_quarantine,
        test_import_overwrite_with_no_old_target_layers_is_a_clean_no_op,
        test_retire_dry_run_names_and_apply_clears_the_chapter_archive,
        test_retire_refuses_only_when_truly_empty_even_with_an_orphaned_chapter_archive,
    ]
    for t in tests:
        t()
    print("\nAll v3194-mem chapter-archive tests passed.")


if __name__ == "__main__":
    main_()
