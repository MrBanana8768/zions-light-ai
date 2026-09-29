"""v3.1.9.10 fix round A: tests that kill the mutants the R1 review left alive.

  M8  removing the journal reapply's `ts < wipe_at` purge rule
  M2  the pre_dates tie `<` -> `<=`
  M6  dropping `_upgrade_legacy` from the ledger WRITE path
  +   a journal line with epoch null that pre-dates a full wipe

Synthetic text only.

    python test_forget_mutants_fixa.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
_TMP = Path(tempfile.mkdtemp(prefix="compactor-test-mutfixa-"))
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_TMP / "store")
os.environ["COMPACTOR_CONTROL_ROOT"] = str(_TMP / "control")
os.environ["COMPACTOR_PENDING_ROOT"] = str(_TMP / "pending")

import chatjournal  # noqa: E402
import continuity  # noqa: E402
import memory  # noqa: E402

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def reset():
    for d in ("store", "control", "pending"):
        shutil.rmtree(_TMP / d, ignore_errors=True)
    memory.ensure_storage_layout()
    continuity._legacy_stamp.clear()


def line(conv, ts, epoch, user, assistant="reply"):
    rec = chatjournal.build_record(
        conv_id=conv, user=user, assistant=assistant, meta=None, request_id="r", ts=ts,
        epoch=epoch, finish_reason="stop", partial=False)
    return chatjournal._dumps(rec)


def write(conv, lines):
    p = chatjournal.journal_path(conv)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def users(conv):
    p = chatjournal.journal_path(conv)
    if not p.is_file():
        return None
    return [(json.loads(x)["user"], bool(json.loads(x).get("redacted"))) for x in p.read_text().splitlines() if x]


WIPE_AT = "2026-09-10T12:00:00.000Z"
BEFORE, AFTER = "2026-09-10T11:00:00.000Z", "2026-09-10T13:00:00.000Z"

print("[M8] the journal purges a line whose ts is before the full wipe's wipe_at, on the ts rule ALONE")
reset()
conv = "m8"
# epoch == wipe_epoch, so the EPOCH rule (ep < wipe_epoch) cannot purge it: only ts can.
write(conv, [line(conv, BEFORE, 2, "old line"), line(conv, AFTER, 2, "new line")])
r = chatjournal.reapply(conv, {"targeted": [], "wipe_epoch": 2, "wipe_at": WIPE_AT})
check(users(conv) == [("new line", False)], f"pre-wipe line dropped, post-wipe kept ({users(conv)})")
check(r["purged"] == 1, "purged == 1")

print("[E0] a line with epoch NULL that pre-dates a wipe is purged; one after it is kept")
reset()
conv = "e0"
write(conv, [line(conv, BEFORE, None, "old null-epoch"), line(conv, AFTER, None, "new null-epoch")])
chatjournal.reapply(conv, {"targeted": [], "wipe_epoch": 2, "wipe_at": WIPE_AT})
check(users(conv) == [("new null-epoch", False)], f"null epoch is judged by ts ({users(conv)})")

print("[M2] a line stamped EXACTLY at the forget's cutoff post-dates it (tie is not pre-dating)")
CUT = "2026-09-10T12:00:00.000Z"
entry = {"needle": "zebrafish", "epoch": 0, "cutoff": CUT}
check(continuity.pre_dates(CUT, 0, entry) is False, "pre_dates: ts == cutoff -> False")
check(continuity.pre_dates(BEFORE, 0, entry) is True, "pre_dates: ts < cutoff -> True")
check(continuity.pre_dates(AFTER, 0, entry) is False, "pre_dates: ts > cutoff -> False")
reset()
conv = "m2"
write(conv, [line(conv, CUT, 0, "zebrafish at the tie"), line(conv, BEFORE, 0, "zebrafish before")])
chatjournal.reapply(conv, {"targeted": [entry], "wipe_epoch": None, "wipe_at": None})
check(users(conv) == [("zebrafish at the tie", False), ("", True)],
      f"the tie line is kept, the earlier one tombstoned ({users(conv)})")

print("[M6] a LEGACY (.8) ledger rewritten by the WRITE path is upgraded to format 2, every needle stamped")
reset()
conv = "m6"
lp = continuity._ledger_path(conv)
lp.parent.mkdir(parents=True, exist_ok=True)
lp.write_text(json.dumps({"epoch": 1, "boundary": None, "targeted": [{"needle": "oldneedle", "epoch": 1}]}))
continuity.record_targeted_forget(conv, "newneedle")   # write path, NOT reapply_ledger (no boot stamp)
disk = json.loads(lp.read_text())
check(disk.get("format") == continuity.LEDGER_FORMAT, f"format stamped ({disk.get('format')})")
old = [n for n in disk["targeted"] if n["needle"] == "oldneedle"][0]
check(bool(old.get("cutoff")) and old.get("legacy") is True, f"the cutoff-less .8 needle got a cutoff and legacy=true ({old})")
check(all(n.get("cutoff") for n in disk["targeted"]), "no needle left without a cutoff")
check("wipe_epoch" in disk and "wipe_at" in disk, "wipe_epoch / wipe_at fields present")

shutil.rmtree(_TMP, ignore_errors=True)
print()
if FAILED:
    print(f"FAILED {len(FAILED)}: {FAILED}")
    sys.exit(1)
print("All fix-round-A mutant-killer checks passed.")
