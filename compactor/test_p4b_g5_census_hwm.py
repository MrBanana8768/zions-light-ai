"""
p4-b (hostile pass #4, reviewer B) G5 — the partial-loss census compared only
against the CYCLE BEFORE it, so a loss spread thin enough (each single step
staying above the 0.5 floor) never alarms: facts truncated
140 -> 71 -> 36 -> 19 -> 10 -> 5 over five cycles read `census_regressions:
[]` on every single one. Persona text replaced wholesale and most of an L1
summary hierarchy dropped, each in one cycle against a field the existing
rules do not track byte-for-byte at all (persona) or only check for a total
wipe (summary_active_bytes), passed the same way.

Ported from SP\\p4-b\\census4.py, with the real writers (facts.py, persona.py,
portability.py) for the normal-operation CONTROLs — a rule that fires on
normal operation is hostile317-c F1 again.

Run inside the compactor image or any container with the requirements
installed:
    python test_p4b_g5_census_hwm.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-p4b-g5-test-"))
_DATA = _ROOT / "openwebui"
_STORE = _DATA / "compactor"
_BK = _ROOT / "bk"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BK)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DATA / "webui.db")
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"
os.environ["WEBUI_DB_LOCAL"] = "false"

import backup  # noqa: E402
import facts  # noqa: E402
import persona  # noqa: E402
import portability  # noqa: E402
import retrieval  # noqa: E402

# Same stub test_health_findings.py uses: no live vector store in this
# fixture (bare-interpreter runs have no fastembed; the unit-test Docker
# image runs retrieval "offline" instead — see census4.py's own docstring).
# fork_conversation/import_conversation's own "is the target empty" check
# reads this to decide whether it can even proceed; without it every
# portability call below refuses with "episodic (vector store unavailable)"
# regardless of anything this fix touches.
retrieval.conversation_doc_count = lambda conv_id: 0

A = "0123456789abcdef0123456789abcdef"
B = "fedcba9876543210fedcba9876543210"


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


def _wj(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps(obj).encode())


def _fact(i, t=None):
    return {"text": t or f"she said thing number {i} that matters", "added_turn": i,
            "last_used": i, "created_at": "2026-09-01T00:00:00Z", "pinned": False}


def _build():
    shutil.rmtree(_ROOT, ignore_errors=True)
    _DATA.mkdir(parents=True)
    c = sqlite3.connect(str(_DATA / "webui.db"))
    c.execute("create table chat (id text primary key, chat text)")
    c.execute("insert into chat values ('c','x')")
    c.commit()
    c.close()
    facts.save_facts(A, [_fact(i) for i in range(40)])
    facts.save_archive(A, [{**_fact(100 + i), "archived_at": 1} for i in range(60)])
    facts.save_facts(B, [_fact(i) for i in range(140)])
    _wj(_STORE / "summaries" / f"{A}.json", {
        "last_summarized_turn": 100,
        "l1": [{"text": "chunk text " * 50, "first_turn": 1 + 20 * k, "last_turn": 20 * (k + 1)}
               for k in range(5)],
        "l2": [], "l3": None,
    })
    persona.save_persona(A, "A long persona text that describes who she is. " * 5)
    _BK.mkdir()


def _cycle():
    time.sleep(1.1)
    r = backup.run_once(_BK)
    return r


def test_g5_slow_bleed_over_five_cycles_is_caught():
    print("\n[test] p4-b G5: B's facts truncated 140 -> 71 -> 36 -> 19 -> 10 -> 5 over five cycles — caught by the hwm floor")
    _build()
    _cycle()
    steps = []
    fired_at = None
    for n in (71, 36, 19, 10, 5):
        facts.save_facts(B, facts.load_facts(B)[:n])
        r = _cycle()
        held = bool(r.get("census_regressions"))
        steps.append((n, held, r.get("census_regressions")))
        if held and fired_at is None:
            fired_at = n
    print(f"      steps={steps}")
    assert_true(fired_at is not None,
                f"G5 fix: the bleed IS caught at some point across the five cycles (pre-fix: never — got {steps})")
    assert_true(any("facts_union" in x for _, held, regs in steps if held for x in (regs or [])),
                f"and the loss is named via the facts_union hwm rule (got {steps})")


def test_g5_persona_replaced_with_stub_is_caught():
    print("\n[test] p4-b G5: persona text replaced by a one-character stub — caught by persona_bytes hwm floor")
    _build()
    _cycle()
    persona.save_persona(A, "x")
    r = _cycle()
    assert_true(bool(r.get("census_regressions")),
                f"G5 fix: a persona gutted to one character is flagged (pre-fix: never — got {r})")
    assert_true(any("persona_bytes" in x for x in r["census_regressions"]),
                f"and specifically via persona_bytes (got {r['census_regressions']})")


def test_g5_most_of_the_l1_hierarchy_dropped_is_caught():
    print("\n[test] p4-b G5: 4 of 5 L1 chunks dropped, watermark kept — caught by summary_active_bytes hwm floor")
    _build()
    _cycle()
    st = json.loads((_STORE / "summaries" / f"{A}.json").read_text())
    st["l1"] = st["l1"][:1]
    _wj(_STORE / "summaries" / f"{A}.json", st)
    r = _cycle()
    assert_true(bool(r.get("census_regressions")),
                f"G5 fix: dropping 4 of 5 L1 chunks is flagged (got {r})")
    assert_true(any("summary_active_bytes" in x for x in r["census_regressions"]),
                f"and specifically via summary_active_bytes (got {r['census_regressions']})")


def test_g5_alarms_once_not_every_night():
    print("\n[test] p4-b G5 CONTROL: the hwm floor alarms ONCE, not every subsequent night")
    _build()
    _cycle()
    facts.save_facts(B, facts.load_facts(B)[:30])  # a real, one-shot loss (140 -> 30, well under half)
    r1 = _cycle()
    assert_true(bool(r1.get("census_regressions")), f"fixture: the loss fires once (got {r1})")
    r2 = _cycle()  # nothing changed since r1 — the hwm reset to the post-loss value
    assert_eq([x for x in r2.get("census_regressions", []) if "facts_union" in x and B[:8] in x], [],
              f"G5 fix: the SAME loss does not re-fire the next night (alarm-once) (got {r2.get('census_regressions')})")


def test_g5_control_real_eviction_does_not_alarm():
    print("\n[test] p4-b G5 CONTROL: real facts.prune_facts eviction does not false-alarm the hwm floor")
    _build()
    _cycle()
    kept, evicted = facts.prune_facts(facts.load_facts(A), max_tokens=50, conv_id=A)
    facts.save_facts(A, kept)
    r = _cycle()
    assert_eq([x for x in r.get("census_regressions", []) if A[:8] in x], [],
              f"G5 fix: eviction (moves facts to the archive sidecar, does not lose the union) does not false-alarm (got {r.get('census_regressions')})")


def test_g5_control_fork_and_merge_do_not_alarm():
    print("\n[test] p4-b G5 CONTROL: fork_conversation / merge_conversation (real writers) do not false-alarm")
    _build()
    _cycle()
    portability.fork_conversation(A)
    r1 = _cycle()
    assert_eq([x for x in r1.get("census_regressions", []) if A[:8] in x], [],
              f"G5 fix: fork does not false-alarm the source conversation (got {r1.get('census_regressions')})")
    portability.merge_conversation(B, A, dry_run=False, refresh_last_used=True)
    r2 = _cycle()
    assert_eq([x for x in r2.get("census_regressions", []) if A[:8] in x], [],
              f"G5 fix: merge (refresh_last_used=True) does not false-alarm the target (got {r2.get('census_regressions')})")


if __name__ == "__main__":
    tests = [
        test_g5_slow_bleed_over_five_cycles_is_caught,
        test_g5_persona_replaced_with_stub_is_caught,
        test_g5_most_of_the_l1_hierarchy_dropped_is_caught,
        test_g5_alarms_once_not_every_night,
        test_g5_control_real_eviction_does_not_alarm,
        test_g5_control_fork_and_merge_do_not_alarm,
    ]
    for t in tests:
        t()
    print("\nAll p4-b G5 (census high-water mark) tests passed.")
