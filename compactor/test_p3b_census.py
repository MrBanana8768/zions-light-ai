"""
p3-b (hostile pass #3, reviewer B) — F2 and F3 fix-lane tests for
compactor.backup's per-conversation census.

Two findings, one shared surface (backup._census / _census_shortfalls /
_census_regressions), so their proofs live together:

  F3 (HIGH): v3.1.9 folded archived facts INTO the `facts` field, which
  changed what an EXISTING manifest key means. v3.1.6.1's and v3.1.8's own
  verify_backup / restore_backup read `facts` by that name after a rollback
  and reject any v3.1.9 archive holding an archived fact. The fix keeps
  `facts` and `summaries` meaning exactly what they meant before v3.1.9, and
  puts every new signal behind a NEW key. `_old_style_census`/
  `_old_style_shortfalls` below are v3.1.8's OWN _census/_census_shortfalls
  (compactor/backup.py, git show v3.1.8:compactor/backup.py:274-330),
  copied verbatim, so this file proves backward compatibility without a
  docker image. The real-image proof (both directions, against the actual
  v3.1.6.1-cu12 and v3.1.8-cu12 images) is
  SP\\fix-p3b-xver-{new,old,new2,old8}.log, adapted from SP\\p3-b\\xver.py —
  not reproduced here because it needs images this suite does not have.

  F2 (HIGH): the tolerant cross-cycle census (hostile317-c F1's fix) only
  flags a layer going to exactly zero, so it is blind to any loss that
  stops short of total — SP\\p3-b\\census.py proves five partial losses read
  `ok` and prune normally. test_census_losses.py below is that same
  fixture, ported into this suite's assertions, run against the FIXED
  code. The normal-operation side (eviction, dedup, rollup, /forget) is
  covered by test_backup_v319.py's existing F1 tests (kept green by this
  fix — see its own updated assertions) plus test_forget_alarms_once_
  through_real_clear_all_memory below, which drives the REAL wipe endpoint
  handler (main._clear_all_memory) rather than a hand-built fixture.

Run inside the compactor image or any container with the requirements
installed:
    python test_p3b_census.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="zions-p3b-census-test-"))
_DATA = _TMP / "data" / "openwebui"
_STORE = _DATA / "compactor"
_BACKUPS = _TMP / "data" / "backups"
_DB = _DATA / "webui.db"
_QUARANTINE = _TMP / "quarantine"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BACKUPS)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DB)
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUARANTINE)
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "2000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_PERSONA_AUTO_DETECT_MIN_CHARS"] = "50"

import backup  # noqa: E402
import facts  # noqa: E402  (real writer)
import persona  # noqa: E402  (real writer)

CID = "0123456789abcdef0123456789abcdef"


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _wipe_store():
    if _STORE.exists():
        shutil.rmtree(_STORE)
    (_STORE / "facts").mkdir(parents=True, exist_ok=True)
    (_STORE / "summaries").mkdir(parents=True, exist_ok=True)
    (_STORE / "personas").mkdir(parents=True, exist_ok=True)


def wj(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps(obj).encode("utf-8"))


def fact(i: int) -> dict:
    return {"text": f"fact number {i} about something specific", "added_turn": i, "last_used": i}


def _build_full_conversation():
    """The exact fixture SP\\p3-b\\census.py uses: 140 active + 60 archived
    facts, a 2/4/1 L1/L2/L3 hierarchy, 12 archived chapters, and a persona."""
    _wipe_store()
    wj(_STORE / "facts" / f"{CID}.json", {"conv_id": CID, "facts": [fact(i) for i in range(140)]})
    wj(_STORE / "facts" / f"{CID}.archive.json", {"conv_id": CID, "facts": [fact(1000 + i) for i in range(60)]})
    wj(_STORE / "summaries" / f"{CID}.json", {
        "last_summarized_turn": 500,
        "l1": [{"text": "l1 " * 300, "first_turn": 481 + 10 * k, "last_turn": 490 + 10 * k} for k in range(2)],
        "l2": [{"text": "l2 " * 800, "first_turn": 1 + 100 * k, "last_turn": 100 + 100 * k} for k in range(4)],
        "l3": {"text": "l3 " * 1500, "first_turn": 1, "last_turn": 400}})
    wj(_STORE / "summaries" / f"{CID}.archive.json",
       {"chapters": [{"text": "ch " * 500, "first_turn": 1, "last_turn": 10} for _ in range(12)]})
    wj(_STORE / "personas" / f"{CID}.json", {"persona_text": "persona " * 200})


# ---------------------------------------------------------------------------
# F2 — five partial losses the exactly-zero rule missed (SP\p3-b\census.py)
# ---------------------------------------------------------------------------

def test_loss1_active_facts_truncated_is_flagged():
    print("\n[test] F2 loss1: active facts 140->3 is flagged, union+bytes both catch it")
    _build_full_conversation()
    prev = backup._census(_STORE)
    wj(_STORE / "facts" / f"{CID}.json", {"conv_id": CID, "facts": [fact(i) for i in range(3)]})
    new = backup._census(_STORE)
    regs = backup._census_regressions(prev, new)
    assert_true(any(r.startswith(f"{CID}.facts ") for r in regs), f"union rule fires (got {regs})")
    assert_true(any(r.startswith(f"{CID}.facts_bytes") for r in regs), f"bytes rule fires (got {regs})")


def test_loss2_archive_sidecar_deleted_is_flagged():
    print("\n[test] F2 loss2: facts archive sidecar deleted (active untouched) is flagged")
    _build_full_conversation()
    prev = backup._census(_STORE)
    (_STORE / "facts" / f"{CID}.archive.json").unlink()
    new = backup._census(_STORE)
    regs = backup._census_regressions(prev, new)
    assert_true(any(r.startswith(f"{CID}.archived_facts") for r in regs), f"got {regs}")
    # the union rule alone would NOT have caught this — the active file (140)
    # dwarfs the 60-fact archive, so the union only drops to 70%. Proves the
    # archived_facts-alone rule is load-bearing, not redundant with the union.
    assert_true(not any(r.startswith(f"{CID}.facts ") for r in regs),
                f"CONTROL: the union alone stays above the floor here (got {regs})")


def test_loss3_hierarchy_emptied_watermark_kept_is_flagged():
    print("\n[test] F2 loss3: L1/L2/L3 emptied, watermark kept, is flagged")
    _build_full_conversation()
    prev = backup._census(_STORE)
    wj(_STORE / "summaries" / f"{CID}.json", {"last_summarized_turn": 500, "l1": [], "l2": [], "l3": None})
    new = backup._census(_STORE)
    regs = backup._census_regressions(prev, new)
    assert_true(any(r.startswith(f"{CID}.summary_active_bytes") for r in regs), f"got {regs}")


def test_loss4_persona_deleted_is_flagged():
    print("\n[test] F2 loss4: persona deleted is flagged")
    _build_full_conversation()
    prev = backup._census(_STORE)
    (_STORE / "personas" / f"{CID}.json").unlink()
    new = backup._census(_STORE)
    regs = backup._census_regressions(prev, new)
    assert_true(any(r.startswith(f"{CID}.persona") for r in regs), f"got {regs}")


def test_loss5_fact_texts_gutted_count_unchanged_is_flagged():
    print("\n[test] F2 loss5: every fact TEXT gutted, count unchanged, is flagged")
    _build_full_conversation()
    prev = backup._census(_STORE)
    wj(_STORE / "facts" / f"{CID}.json",
       {"conv_id": CID, "facts": [{"text": "x", "added_turn": i, "last_used": i} for i in range(140)]})
    new = backup._census(_STORE)
    assert_eq(new[CID]["facts"], prev[CID]["facts"], "CONTROL: the count alone did not move")
    regs = backup._census_regressions(prev, new)
    assert_true(any(r.startswith(f"{CID}.facts_bytes") for r in regs), f"got {regs}")


def test_baseline_and_untouched_cycle_stay_clean():
    print("\n[test] F2 CONTROL: the guard can still say yes — no regression on an untouched re-census")
    _build_full_conversation()
    prev = backup._census(_STORE)
    new = backup._census(_STORE)
    assert_eq(backup._census_regressions(prev, new), [], "identical store, no regressions")


def test_l1_to_l2_rollup_does_not_trip_summary_active_bytes():
    print("\n[test] F2 CONTROL: a real L1->L2 rollup shape (bytes drop, structure intact) does not false-fire")
    # This is the shape the summary_bytes UNION rule was rejected over (see
    # backup._census_regressions's docstring): l1's small chunks collapse
    # into ONE shorter l2 chapter, so summary_active_bytes legitimately
    # drops by more than half in a single cycle, with l1 empty and l2/l3
    # holding the (compressed) survivor.
    _wipe_store()
    wj(_STORE / "summaries" / f"{CID}.json", {
        "last_summarized_turn": 200,
        "l1": [{"text": f"scene {i}" * 20, "first_turn": i * 20 + 1, "last_turn": i * 20 + 20} for i in range(10)],
        "l2": [], "l3": None,
    })
    prev = backup._census(_STORE)
    wj(_STORE / "summaries" / f"{CID}.json", {
        "last_summarized_turn": 200,
        "l1": [],
        "l2": [{"text": "chapter", "first_turn": 1, "last_turn": 200}],
        "l3": None,
    })
    new = backup._census(_STORE)
    assert_true(new[CID]["summary_active_bytes"] < prev[CID]["summary_active_bytes"] * 0.5,
                "fixture: bytes really did drop by more than half (the risky shape)")
    regs = backup._census_regressions(prev, new)
    assert_eq(regs, [], f"F2 fix: a real (non-total) rollup byte drop is not flagged (got {regs})")


# ---------------------------------------------------------------------------
# F2 — /forget alarms once, via the REAL wipe handler (main._clear_all_memory)
# ---------------------------------------------------------------------------

def test_forget_alarms_once_through_real_clear_all_memory():
    print("\n[test] F2: a real /forget (main._clear_all_memory) alarms once, via summary_turn+persona")
    import main  # noqa: E402  (imported lazily: heavier than backup/facts/persona)
    _build_full_conversation()
    prev = backup._census(_STORE)

    asyncio.run(main._clear_all_memory(CID, source="test"))

    new = backup._census(_STORE)
    regs = backup._census_regressions(prev, new)
    assert_true(any(r.startswith(f"{CID}.summary_turn") for r in regs), f"got {regs}")
    assert_true(any(r.startswith(f"{CID}.persona") for r in regs), f"got {regs}")

    # And the NEXT cycle, comparing the post-forget census against itself,
    # is clean — "alarms once" means once, not forever.
    again = backup._census(_STORE)
    assert_eq(backup._census_regressions(new, again), [],
              "the cycle AFTER the forget does not re-alarm")


# ---------------------------------------------------------------------------
# F3 — old field names keep their old meaning (v3.1.8's own _census, verbatim)
# ---------------------------------------------------------------------------

def _old_style_census(store: Path) -> dict:
    """v3.1.8's compactor/backup.py:_census, copied verbatim (git show
    v3.1.8:compactor/backup.py:274-330) — NOT imported, because this
    worktree only has the fixed v3.1.9-line code; this is what an already
    -deployed v3.1.6.1/v3.1.8 binary computes when it reads a v3.1.9
    archive after a rollback."""
    census: dict[str, dict] = {}

    def slot(conv_id: str) -> dict:
        return census.setdefault(conv_id, {"facts": 0, "summaries": 0, "episodic": 0})

    facts_dir = store / "facts"
    if facts_dir.is_dir():
        for f in sorted(facts_dir.glob("*.json")):
            if "." in f.stem:
                continue
            data = backup._read_json(f)
            if isinstance(data, dict) and isinstance(data.get("facts"), list):
                slot(f.stem)["facts"] = len(data["facts"])

    summaries_dir = store / "summaries"
    if summaries_dir.is_dir():
        for f in sorted(summaries_dir.glob("*.json")):
            if "." in f.stem:
                continue
            data = backup._read_json(f)
            if not isinstance(data, dict):
                continue
            n = 0
            for tier in ("l1", "l2"):
                if isinstance(data.get(tier), list):
                    n += len(data[tier])
            if isinstance(data.get("l3"), dict):
                n += 1
            slot(f.stem)["summaries"] = n

    episodic = backup._episodic_counts(store / "chromadb" / "chroma.sqlite3")
    if episodic:
        for conv_id, n in episodic.items():
            slot(conv_id)["episodic"] = n
    return census


def _old_style_shortfalls(expected: dict, actual: dict) -> list[str]:
    """v3.1.8's _census_shortfalls, copied verbatim."""
    out: list[str] = []
    for conv_id in sorted(expected):
        want = expected.get(conv_id) or {}
        have = actual.get(conv_id) or {}
        if not isinstance(want, dict):
            continue
        have = have if isinstance(have, dict) else {}
        for layer in ("facts", "summaries", "episodic"):
            w = int(want.get(layer) or 0)
            h = int(have.get(layer) or 0)
            if h < w:
                out.append(f"{conv_id}.{layer} {w}->{h}")
    return out


def test_old_reader_sees_the_same_facts_and_summaries_it_always_did():
    print("\n[test] F3: an old-shaped reader recomputing from the SAME tree agrees with the manifest")
    _build_full_conversation()
    new_census = backup._census(_STORE)
    old_census = _old_style_census(_STORE)

    assert_eq(new_census[CID]["facts"], old_census[CID]["facts"],
              "F3 fix: `facts` still means active-only, exactly like the old reader")
    assert_eq(new_census[CID]["summaries"], old_census[CID]["summaries"],
              "F3 fix: `summaries` still means len(l1)+len(l2)+bool(l3), exactly like the old reader")

    # The manifest (written by the fixed _census) has MORE keys than the old
    # reader looks at; an old verify_backup would recompute its OWN 3-key
    # census from the extracted tree and compare against the manifest's
    # facts/summaries/episodic — those three must never read as a shortfall.
    shortfalls = _old_style_shortfalls(new_census, old_census)
    assert_eq(shortfalls, [],
              f"F3 fix: v3.1.9's own manifest fields do not read as truncated "
              f"to an old-shaped verifier (got {shortfalls})")


def test_control_old_reader_still_catches_real_truncation():
    print("\n[test] F3 CONTROL: an old-shaped reader still catches a genuinely truncated archive")
    _build_full_conversation()
    expected = backup._census(_STORE)
    # Simulate a truncated extraction: half the active facts missing from
    # what the old reader recomputes.
    wj(_STORE / "facts" / f"{CID}.json", {"conv_id": CID, "facts": [fact(i) for i in range(70)]})
    actual = _old_style_census(_STORE)
    shortfalls = _old_style_shortfalls(expected, actual)
    assert_true(any(s.startswith(f"{CID}.facts") for s in shortfalls),
                f"a real truncation is still caught (got {shortfalls})")


if __name__ == "__main__":
    tests = [
        test_loss1_active_facts_truncated_is_flagged,
        test_loss2_archive_sidecar_deleted_is_flagged,
        test_loss3_hierarchy_emptied_watermark_kept_is_flagged,
        test_loss4_persona_deleted_is_flagged,
        test_loss5_fact_texts_gutted_count_unchanged_is_flagged,
        test_baseline_and_untouched_cycle_stay_clean,
        test_l1_to_l2_rollup_does_not_trip_summary_active_bytes,
        test_forget_alarms_once_through_real_clear_all_memory,
        test_old_reader_sees_the_same_facts_and_summaries_it_always_did,
        test_control_old_reader_still_catches_real_truncation,
    ]
    for t in tests:
        t()
    print("\nAll p3-b census (F2/F3) tests passed.")
