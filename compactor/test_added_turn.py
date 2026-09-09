"""`added_turn` is allocated from the store, not from the client's array.

B2 of FRONTEND_HANDOFF, the part of it that was genuinely still open.

FRONTEND_SPEC §15 asked for a server-side turn sequence on the grounds that
three things break under a bounded-window client. Checked against the code,
all three had already been fixed independently — content-addressed episodic ids
and store-allocated ordinals (v3.1 D1), the out-of-frame cutoff guard
(v3.1 A6), and `_observed_position` owning the conversational position
(v3.1.4). Adding the `turn_seq` the spec asked for would have created a THIRD
counter beside `turns_seen` and `_stored_max_turn_index`, and two counters that
can drift is already this codebase's signature defect.

What was still reading the client's array is this one: a new fact's
`added_turn` was the request's `len(messages) + 1`. Under a bounded window that
is near-constant, so every fact in a long conversation gets the same number and
the `(pin, last_used, added_turn)` ranking loses its last term precisely where
the store is largest.

It costs ordering rather than data, which is why it is fixed with the
floor-raiser already proven in `retrieval._next_turn_index` rather than with
new machinery.

    python test_added_turn.py
"""

import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="added-turn-")

import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def facts_at(*turns):
    return [{"text": f"f{t}", "added_turn": t, "last_used": 0} for t in turns]


nxt = main._next_added_turn

print("[1] an empty store falls back to the request")
check(nxt([], 7) == 7,
      "with nothing stored the request is the only evidence there is")
check(nxt([], 1) == 1, "and a first turn starts at 1, not 0")

print("[2] the stored maximum is a FLOOR — this is the bounded-window fix")
# The shape that motivated this: a long conversation whose client sends a
# constant window, so turn_index is pinned low while the store is far along.
check(nxt(facts_at(40, 41, 42), 7) == 43,
      "a pinned request (7) does not drag the sequence back to 7 — it "
      "continues from the store at 43")
check(nxt(facts_at(200), 7) == 201,
      "and the gap does not matter: the store wins whenever it is ahead")

print("[3] but the request can still push it FORWARD")
# Deliberately not a pure counter. A conversation whose facts predate this
# would otherwise count 1, 2, 3… beside episodic rows numbered in the
# hundreds, and nothing downstream could compare them.
check(nxt(facts_at(3), 100) == 100,
      "a request ahead of the store raises the sequence to it, rather than "
      "stepping to 4 and staying in different units from everything else")

print("[4] it never repeats the highest stored value")
check(nxt(facts_at(9), 9) == 10,
      "a request EQUAL to the stored max still advances — returning 9 would "
      "stamp two facts with one number and the tiebreaker would be a tie")

print("[5] a malformed store cannot crash the memory tail")
# This runs inside _async_tail, where an exception loses the exchange's facts
# entirely. Every one of these has been seen in a real store on this project.
check(nxt([{"text": "no added_turn"}], 5) == 5,
      "a fact with no added_turn is treated as 0, not an error")
check(nxt([{"text": "null", "added_turn": None}], 5) == 5,
      "an explicit null is treated as 0")
check(nxt(["not a dict", None, 42], 5) == 5,
      "non-dict entries are skipped rather than raising — read_json_strict "
      "guards the file shape, not every element in it")
check(nxt(facts_at(3) + ["junk"], 1) == 4,
      "and a mixed list still finds the real maximum")

print("[6] monotonic over a run, which is the property that matters")
store = []
seen = []
for i in range(12):
    # A bounded-window client: turn_index pinned at the window size.
    t = nxt(store, 7)
    seen.append(t)
    store = store + facts_at(t)
check(seen == sorted(seen) and len(set(seen)) == len(seen),
      f"twelve exchanges through a CONSTANT request index produce a strictly "
      f"increasing sequence {seen[:4]}…{seen[-2:]} — under the old code every "
      f"one of these was 7")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll added_turn checks passed.")
