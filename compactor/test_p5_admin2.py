"""
Hostile pass 5 (reviewer C), lane p5-admin2: C5-4 and C5-5.

C5-4: F1's torn-bytes rule (quarantine_conversation copies a torn file's
raw bytes aside instead of just recording the layer unverified) was applied
to the facts file only. An overwrite import over a torn SUMMARY state
returned 200 and destroyed the torn file's bytes with nothing kept aside.

C5-5: F6's "refuse unknown keys / refuse duplicate JSON keys" rule landed
on /compact and /merge-into only. /restore, /import, /cleanup-test-data and
/archive (and every other admin write endpoint that reads a body or query
string at all) kept the old, permissive behaviour -- a dry_run key on an
endpoint with no dry run was silently ignored rather than refused, and a
duplicated top-level JSON key resolved last-wins with no trace either
value ever disagreed.

Findings source: SP\\p5-c-findings.md C5-4, C5-5. Reproductions adapted
from the reviewer's own SP\\p5-c\\admin_probe.py (cases A-H), rebuilt as
assertions against the real app per the fix-lane brief. Read first:
SP\\fix-p4c.md (F1/F6, the rules this finding says were applied at only
some sites) and compactor/test_p4c_findings.py /
compactor/test_p4c_compact_verdict.py (the existing F1/F2/F6/F8a coverage
this file deliberately does not re-derive).

    python test_p5_admin2.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

_TMP_ROOT = tempfile.mkdtemp(prefix="p5-admin2-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main, memory, persona, portability, retrieval, summarizer, facts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

memory.ensure_storage_layout()
admin = TestClient(main.app, client=("127.0.0.1", 12348), raise_server_exceptions=False)

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


# Episodic layer stubbed in-memory exactly as test_p4c_findings.py /
# admin_probe.py do -- no real ChromaDB in this suite.
STORE: dict[str, list[dict]] = {}
retrieval.export_indexed_exchanges = lambda cid: list(STORE.get(cid, []))
retrieval.conversation_doc_count = lambda cid: len(STORE.get(cid, []))
retrieval.import_indexed_exchange = (
    lambda cid, ti, doc: STORE.setdefault(cid, []).append(
        {"turn_index": ti, "document": doc}
    )
    or True
)
retrieval.forget_conversation = lambda cid: len(STORE.pop(cid, []))


def seed_facts(cid, n, *, last_used=None):
    now = int(time.time())
    facts.save_facts(
        cid,
        [
            {
                "text": f"{cid} fact number {i} about the lantern",
                "last_used": last_used if last_used is not None else now,
                "created": now,
            }
            for i in range(n)
        ],
    )


def raw_post(path, raw: bytes):
    return admin.post(path, content=raw, headers={"content-type": "application/json"})


def all_quarantine_bytes() -> bytes:
    d = portability.quarantine_dir()
    return b"".join(p.read_bytes() for p in d.rglob("*") if p.is_file()) if d.exists() else b""


# ===========================================================================
# C5-4: torn summary state loses its bytes on an overwrite import
# ===========================================================================
print("== C5-4: torn SUMMARY state on an overwrite import ==")

seed_facts("c54-donor", 2)
_donor_bundle = portability.export_conversation("c54-donor")

seed_facts("c54-victim", 5)
STORE["c54-victim"] = [{"turn_index": 1, "document": "[user]: q\n[assistant]: a"}]
_TORN_SUMMARY = (
    b'{"l1": [{"text": "UNIQUE-TORN-SUMMARY-MARKER chapter of her story", '
    b'"first_turn": 1, "last_turn": 20}], "l2": [], "last_summ'
)
_sp = summarizer.summary_path("c54-victim")
_sp.parent.mkdir(parents=True, exist_ok=True)
_sp.write_bytes(_TORN_SUMMARY)

r = admin.post(
    "/admin/conversations/import",
    json={"bundle": _donor_bundle, "target_conv_id": "c54-victim", "overwrite": True},
)
check(r.status_code == 200, f"[C5-4] overwrite over a torn summary still succeeds (got {r.status_code}: {r.text[:160]})")
_qs = portability.list_quarantine("c54-victim")
check(len(_qs) == 1, f"[C5-4] exactly one quarantine snapshot was published (got {len(_qs)})")
if _qs:
    _q = json.loads(Path(_qs[0]).read_bytes())["quarantine"]
    check(
        "summaries (unreadable)" in _q["unverified_layers"],
        f"[C5-4] the summary layer is recorded unverified (got {_q['unverified_layers']})",
    )
    check(
        _q.get("torn_summary_path") is not None,
        "[C5-4] the fix: torn_summary_path is present in the metadata block "
        "(was None pre-fix -- F1's rule never reached the summary layer)",
    )
    if _q.get("torn_summary_path"):
        _torn_bytes = Path(_q["torn_summary_path"]).read_bytes()
        check(
            _torn_bytes == _TORN_SUMMARY,
            "[C5-4] the torn summary file's raw bytes were copied aside "
            "byte-for-byte",
        )
check(
    b"UNIQUE-TORN-SUMMARY-MARKER" in all_quarantine_bytes(),
    "[C5-4] the torn summary's marker text is preserved SOMEWHERE in the "
    "quarantine dir (was entirely absent pre-fix)",
)
# The overwrite itself must still have gone through -- this is a safety net,
# not a refusal mechanism, for a layer that IS readable enough to overwrite.
check(
    b"UNIQUE-TORN-SUMMARY-MARKER" not in _sp.read_bytes(),
    "[C5-4] CONTROL: the live summary file was actually replaced by the "
    "overwrite (the fix must not turn this into a silent no-op)",
)

# CONTROL: an ordinary overwrite (everything readable) still needs no torn
# bytes at all, and torn_summary_path stays None -- this fix must not
# start reporting "unverified" on a healthy conversation.
seed_facts("c54-healthy", 3)
STORE["c54-healthy"] = [{"turn_index": 1, "document": "[user]: q\n[assistant]: a"}]
summarizer.save_state("c54-healthy", summarizer.load_state("c54-healthy"))
r = admin.post(
    "/admin/conversations/import",
    json={"bundle": _donor_bundle, "target_conv_id": "c54-healthy", "overwrite": True},
)
check(r.status_code == 200, f"[C5-4] CONTROL: a healthy overwrite still succeeds (got {r.status_code})")
_qs = portability.list_quarantine("c54-healthy")
if _qs:
    _q = json.loads(Path(_qs[-1]).read_bytes())["quarantine"]
    check(
        _q.get("torn_summary_path") is None and "summaries (unreadable)" not in _q["unverified_layers"],
        f"[C5-4] CONTROL: a readable summary state reports no torn path and "
        f"no unverified entry (got torn_summary_path={_q.get('torn_summary_path')!r}, "
        f"unverified={_q['unverified_layers']})",
    )

# The refuse-if-uncopyable half: if even the torn summary's raw bytes
# cannot be read back off disk, the whole quarantine (and therefore the
# overwrite behind it) must be refused with a 409 -- never proceed on "the
# store is unreadable, nothing to lose".
seed_facts("c54-uncopyable", 4)
STORE["c54-uncopyable"] = [{"turn_index": 1, "document": "[user]: q\n[assistant]: a"}]
_sp2 = summarizer.summary_path("c54-uncopyable")
_sp2.parent.mkdir(parents=True, exist_ok=True)
_sp2.write_bytes(b'{"l1": [{"broken')
import pathlib as _pathlib  # noqa: E402

_real_read_bytes = _pathlib.Path.read_bytes


def _boom_read_bytes(self):
    if self.name == _sp2.name:
        raise OSError(5, "simulated I/O error reading the torn summary file")
    return _real_read_bytes(self)


with patch.object(_pathlib.Path, "read_bytes", _boom_read_bytes):
    r = admin.post(
        "/admin/conversations/import",
        json={"bundle": _donor_bundle, "target_conv_id": "c54-uncopyable", "overwrite": True},
    )
check(
    r.status_code == 409,
    f"[C5-4] an uncopyable torn summary file refuses the overwrite entirely "
    f"(got {r.status_code}: {r.text[:160]})",
)
check(
    b"broken" in _sp2.read_bytes(),
    "[C5-4] and the original (torn) summary file on disk is untouched -- "
    "the overwrite never ran",
)

# CONTROL: torn FACTS (F1's own case) still works exactly as before -- this
# lane's generalisation must not have disturbed F1's own mechanism.
seed_facts("c54-tornfacts", 1)
_fp = memory.facts_path("c54-tornfacts") if hasattr(memory, "facts_path") else None
if _fp is not None:
    _fp.write_bytes(b'[{"text": "UNIQUE-TORN-FACTS-MARKER", "last_u')
    r = admin.post(
        "/admin/conversations/import",
        json={"bundle": _donor_bundle, "target_conv_id": "c54-tornfacts", "overwrite": True},
    )
    check(
        r.status_code == 200 and b"UNIQUE-TORN-FACTS-MARKER" in all_quarantine_bytes(),
        f"[C5-4] CONTROL: F1's torn-facts path is unaffected by this fix "
        f"(status {r.status_code}, marker kept: "
        f"{b'UNIQUE-TORN-FACTS-MARKER' in all_quarantine_bytes()})",
    )


# ===========================================================================
# C5-5: the unknown/duplicate-key rule generalised to every admin write
# endpoint
# ===========================================================================
print()
print("== C5-5: unknown-key / duplicate-key / dry_run-without-a-dry-run ==")

old_cutoff = int(time.time()) - 400 * 86400

# B. /restore?dry_run=true + {"restore_all": true} -- restore has NO dry
# run; the query key must be refused, not silently ignored.
seed_facts("c55-b", 3)
facts.archive_facts("c55-b", [{"text": f"archived {i}", "last_used": old_cutoff, "created": old_cutoff} for i in range(300)])
r = admin.post("/admin/conversations/c55-b/restore?dry_run=true", json={"restore_all": True})
check(
    r.status_code == 400 and len(facts.load_facts("c55-b")) == 3,
    f"[C5-5 B] /restore?dry_run=true is refused, not silently ignored "
    f"(got {r.status_code}, active facts {len(facts.load_facts('c55-b'))}, "
    f"was 3)",
)

# B2. Same, but the query key is a TYPO of dry_run ("dryrun") -- proves
# dry_run_typo_exempt defaults to False for an endpoint with no dry run at
# all, not just that the exact spelling is caught.
seed_facts("c55-b2", 3)
facts.archive_facts("c55-b2", [{"text": f"archived {i}", "last_used": old_cutoff, "created": old_cutoff} for i in range(300)])
r = admin.post("/admin/conversations/c55-b2/restore?dryrun=true", json={"restore_all": True})
check(
    r.status_code == 400 and len(facts.load_facts("c55-b2")) == 3,
    f"[C5-5 B2] /restore?dryrun=true (typo) is ALSO refused -- restore has "
    f"no forced-dry handling to exempt it into (got {r.status_code}, active "
    f"{len(facts.load_facts('c55-b2'))}, was 3)",
)

# C. /restore duplicate key {"restore_all": false, "restore_all": true}.
seed_facts("c55-c", 3)
facts.archive_facts("c55-c", [{"text": f"archived {i}", "last_used": old_cutoff, "created": old_cutoff} for i in range(300)])
r = raw_post("/admin/conversations/c55-c/restore", b'{"restore_all": false, "restore_all": true}')
check(
    r.status_code == 400 and len(facts.load_facts("c55-c")) == 3,
    f"[C5-5 C] a duplicated restore_all key is refused, not last-wins "
    f"(got {r.status_code}, active {len(facts.load_facts('c55-c'))}, was 3)",
)

# D. /import duplicate key {"overwrite": false, ..., "overwrite": true}.
seed_facts("c55-d", 7)
_raw_d = (
    b'{"overwrite": false, "target_conv_id": "c55-d", "bundle": '
    + json.dumps(_donor_bundle).encode() + b', "overwrite": true}'
)
r = raw_post("/admin/conversations/import", _raw_d)
check(
    r.status_code == 400 and len(facts.load_facts("c55-d")) == 7,
    f"[C5-5 D] a duplicated overwrite key is refused, not last-wins toward "
    f"the destructive value (got {r.status_code}, facts "
    f"{len(facts.load_facts('c55-d'))}, was 7)",
)

# E. /import {"overwrite": true, "dry_run": true} -- import has NO dry run.
seed_facts("c55-e", 7)
r = admin.post(
    "/admin/conversations/import",
    json={"bundle": _donor_bundle, "target_conv_id": "c55-e", "overwrite": True, "dry_run": True},
)
check(
    r.status_code == 400 and len(facts.load_facts("c55-e")) == 7,
    f"[C5-5 E] a dry_run key on /import is refused, not silently ignored "
    f"(got {r.status_code}, facts {len(facts.load_facts('c55-e'))}, was 7)",
)

# F. /cleanup-test-data duplicate {"dry_run": true, "dry_run": false}.
_tid = "itest-c55f0012"
seed_facts(_tid, 1)
r = raw_post("/admin/conversations/cleanup-test-data", b'{"dry_run": true, "dry_run": false}')
check(
    r.status_code == 400 and len(facts.load_facts(_tid)) == 1,
    f"[C5-5 F] a duplicated dry_run key is refused (got {r.status_code}, "
    f"facts left {len(facts.load_facts(_tid))}, was 1) -- this endpoint's "
    f"own docstring promises dry wins on any disagreement, which a silent "
    f"last-wins commit contradicted",
)

# G. /archive?dry_run=true&older_than_days=0 -- archive has NO dry run.
seed_facts("c55-g", 40, last_used=int(time.time()) - 10)
r = admin.post("/admin/conversations/c55-g/archive?dry_run=true&older_than_days=0")
check(
    r.status_code == 400 and len(facts.load_facts("c55-g")) == 40,
    f"[C5-5 G] /archive?dry_run=true is refused, not silently ignored while "
    f"archiving live (got {r.status_code}, active {len(facts.load_facts('c55-g'))}, was 40)",
)

# H. /archive?older_than_day=365 (typo, missing the trailing "s") -- must
# not silently fall back to the 90-day default.
seed_facts("c55-h", 10, last_used=int(time.time()) - 120 * 86400)
r = admin.post("/admin/conversations/c55-h/archive?older_than_day=365")
check(
    r.status_code == 400 and len(facts.load_facts("c55-h")) == 10,
    f"[C5-5 H] a typo'd older_than_day is refused, not treated as an absent "
    f"key with the 90-day default silently applied (got {r.status_code}, "
    f"active {len(facts.load_facts('c55-h'))}, was 10)",
)

print()
print("-- CONTROLS: the same endpoints still do their real job --")

# CONTROL: restore_all=true via body alone still restores everything.
seed_facts("c55-ctrl-restore", 2)
facts.archive_facts("c55-ctrl-restore", [{"text": f"archived {i}", "last_used": old_cutoff, "created": old_cutoff} for i in range(5)])
r = admin.post("/admin/conversations/c55-ctrl-restore/restore", json={"restore_all": True})
check(
    r.status_code == 200 and len(facts.load_facts("c55-ctrl-restore")) == 7,
    f"CONTROL: {{'restore_all': true}} with no query string still restores "
    f"everything (got {r.status_code}, active {len(facts.load_facts('c55-ctrl-restore'))}, want 7)",
)

# CONTROL: a clean, single-key import overwrite still works.
seed_facts("c55-ctrl-import", 7)
r = admin.post(
    "/admin/conversations/import",
    json={"bundle": _donor_bundle, "target_conv_id": "c55-ctrl-import", "overwrite": True},
)
check(
    r.status_code == 200 and len(facts.load_facts("c55-ctrl-import")) == len(_donor_bundle["facts"]),
    f"CONTROL: a clean overwrite import still succeeds (got {r.status_code}, "
    f"facts {len(facts.load_facts('c55-ctrl-import'))}, want {len(_donor_bundle['facts'])})",
)

# CONTROL: cleanup-test-data with a single, valid dry_run key still works
# both ways (dry, and committing).
_tid2 = "itest-c55ctrl01"
seed_facts(_tid2, 1)
r = admin.post("/admin/conversations/cleanup-test-data", json={"dry_run": True})
check(
    r.status_code == 200 and r.json().get("dry_run") is True and len(facts.load_facts(_tid2)) == 1,
    f"CONTROL: cleanup-test-data {{'dry_run': true}} stays dry (got "
    f"{r.status_code}, dry_run={r.json().get('dry_run')!r}, facts left "
    f"{len(facts.load_facts(_tid2))})",
)
r = admin.post("/admin/conversations/cleanup-test-data?dry_run=false", json={})
check(
    r.status_code == 200 and r.json().get("dry_run") is False,
    f"CONTROL: cleanup-test-data ?dry_run=false still commits (got "
    f"{r.status_code}, dry_run={r.json().get('dry_run')!r})",
)

# CONTROL: archive with only its own valid query key still archives.
seed_facts("c55-ctrl-archive", 5, last_used=int(time.time()) - 120 * 86400)
r = admin.post("/admin/conversations/c55-ctrl-archive/archive?older_than_days=90")
check(
    r.status_code == 200 and r.json().get("archived") == 5,
    f"CONTROL: /archive?older_than_days=90 still archives (got "
    f"{r.status_code}: {r.json() if r.status_code == 200 else r.text[:120]})",
)
# CONTROL: archive with no query at all still uses the real default.
seed_facts("c55-ctrl-archive2", 3, last_used=int(time.time()) - 120 * 86400)
r = admin.post("/admin/conversations/c55-ctrl-archive2/archive")
check(
    r.status_code == 200 and r.json().get("older_than_days") == facts.ARCHIVE_DEFAULT_DAYS,
    f"CONTROL: /archive with no query string still uses the real default "
    f"(got {r.status_code}: {r.json() if r.status_code == 200 else r.text[:120]})",
)

# CONTROL: fork still works with an empty body and with new_conv_id set.
r = admin.post("/admin/conversations/c55-ctrl-import/fork", json={})
check(r.status_code == 200, f"CONTROL: fork with an empty body still works (got {r.status_code})")
r = admin.post("/admin/conversations/c55-ctrl-import/fork", json={"new_conv_id": "c55-ctrl-fork-dst"})
check(r.status_code == 200, f"CONTROL: fork with new_conv_id still works (got {r.status_code})")
r = admin.post("/admin/conversations/c55-ctrl-import/fork", json={"target_conv_id": "nope"})
check(
    r.status_code == 400,
    f"[C5-5 fork] an unrecognised body key (target_conv_id, not new_conv_id) "
    f"is refused (got {r.status_code})",
)

# CONTROL / new refusal: persona set.
r = admin.post("/admin/conversations/c55-ctrl-import/persona", json={"text": "likes tea"})
check(r.status_code == 200, f"CONTROL: persona set with a valid body still works (got {r.status_code})")
r = admin.post("/admin/conversations/c55-ctrl-import/persona", json={"text": "likes tea", "note": "x"})
check(r.status_code == 400, f"[C5-5 persona] an unrecognised body key is refused (got {r.status_code})")
r = admin.post("/admin/conversations/c55-ctrl-import/persona?verbose=1", json={"text": "likes tea"})
check(r.status_code == 400, f"[C5-5 persona] an unrecognised query key is refused (got {r.status_code})")

# CONTROL / new refusal: inherit-persona.
r = admin.post(
    "/admin/conversations/c55-ctrl-import2/inherit-persona",
    json={"source_conv_id": "c55-ctrl-import"},
)
check(r.status_code == 200, f"CONTROL: inherit-persona with a valid body still works (got {r.status_code})")
r = admin.post(
    "/admin/conversations/c55-ctrl-import2/inherit-persona",
    json={"source_conv_id": "c55-ctrl-import", "note": "x"},
)
check(r.status_code == 400, f"[C5-5 inherit-persona] an unrecognised body key is refused (got {r.status_code})")

# CONTROL / new refusal: DELETE facts (forget) and dedup take no body/query.
seed_facts("c55-ctrl-forget", 1)
r = admin.delete("/admin/conversations/c55-ctrl-forget/facts")
check(r.status_code == 200, f"CONTROL: DELETE facts with no body still works (got {r.status_code})")
r = admin.delete("/admin/conversations/c55-ctrl-forget/facts?bogus=1")
check(r.status_code == 400, f"[C5-5 forget] an unrecognised query key is refused (got {r.status_code})")

seed_facts("c55-ctrl-dedup", 1)
r = admin.post("/admin/conversations/c55-ctrl-dedup/dedup")
check(r.status_code == 200, f"CONTROL: dedup with no body still works (got {r.status_code})")
r = raw_post("/admin/conversations/c55-ctrl-dedup/dedup", b'{"z": 1, "z": 2}')
check(r.status_code == 400, f"[C5-5 dedup] a duplicated (even unrecognised) key is refused (got {r.status_code})")

# CONTROL: compact and merge-into keep their EXISTING dry_run-typo
# exemption -- this lane's dry_run_typo_exempt parameterisation must not
# have disabled F6's own forced-dry handling for the endpoints that
# actually have it.
seed_facts("c55-ctrl-compact", 1)
STORE["c55-ctrl-compact"] = [{"turn_index": 1, "document": "[user]: q\n[assistant]: a"}]
r = admin.post("/admin/conversations/c55-ctrl-compact/compact?dryrun=true")
check(
    r.status_code == 200 and r.json().get("dry_run") is True,
    f"CONTROL: /compact?dryrun=true (typo) still forces dry, not a 400 "
    f"(got {r.status_code}: {r.text[:160] if r.status_code != 200 else r.json().get('dry_run')})",
)
seed_facts("c55-ctrl-merge-src", 1)
seed_facts("c55-ctrl-merge-dst", 1)
r = raw_post(
    "/admin/conversations/c55-ctrl-merge-src/merge-into/c55-ctrl-merge-dst",
    b'{"dryRun": true}',
)
check(
    r.status_code == 200 and r.json().get("dry_run") is True,
    f"CONTROL: /merge-into {{'dryRun': true}} (typo) still forces dry, not "
    f"a 400 (got {r.status_code}: {r.text[:160] if r.status_code != 200 else r.json().get('dry_run')})",
)


# ===========================================================================
# Structural: walk main.app.routes for every POST/DELETE/PUT under /admin
# and assert each refuses an unrecognised query key and a duplicated body
# key. This is the "fails if a new route skips it" check the finding asks
# for -- it does not enumerate routes by name, so a route added later that
# forgets the shared helper fails THIS loop without needing an edit here.
# ===========================================================================
print()
print("== structural: every admin write route refuses ?bogus=1 and a duplicated body key ==")

_WRITE_METHODS = {"POST", "DELETE", "PUT"}
_PROBE_CID = "p5admin2-route-probe"
_PROBE_CID2 = "p5admin2-route-probe-dst"


def _fill(path_template: str) -> str:
    return (
        path_template.replace("{conv_id}", _PROBE_CID)
        .replace("{src_conv_id}", _PROBE_CID)
        .replace("{dst_conv_id}", _PROBE_CID2)
    )


_admin_write_routes: list[tuple[str, str]] = []
for _route in main.app.routes:
    _methods = getattr(_route, "methods", None) or set()
    _path = getattr(_route, "path", None)
    if not _path or not _path.startswith("/admin"):
        continue
    _write = sorted(_methods & _WRITE_METHODS)
    if _write:
        _admin_write_routes.append((_write[0], _path))

check(
    len(_admin_write_routes) >= 13,
    f"the walk found every admin write route this lane fixed, at least "
    f"(found {len(_admin_write_routes)}: {sorted(p for _, p in _admin_write_routes)})",
)

seed_facts(_PROBE_CID, 1)
seed_facts(_PROBE_CID2, 1)

for _method, _template in _admin_write_routes:
    _url = _fill(_template)
    _label = f"{_method} {_template}"

    # (a) an unrecognised query key.
    _sep = "&" if "?" in _url else "?"
    _r = admin.request(_method, f"{_url}{_sep}bogus=1")
    check(
        _r.status_code == 400,
        f"[route-walk] {_label}: ?bogus=1 is refused (got {_r.status_code}: {_r.text[:100]!r})",
    )

    # (b) a duplicated top-level JSON body key. The key name does not need
    # to be one this route recognises -- the duplicate-key check runs
    # before the unknown-key check in every handler this lane touched, so
    # this proves the duplicate mechanism specifically, not just "some
    # check exists".
    _r2 = admin.request(
        _method, _url, content=b'{"__dup__": 1, "__dup__": 2}',
        headers={"content-type": "application/json"},
    )
    check(
        _r2.status_code == 400,
        f"[route-walk] {_label}: a duplicated body key is refused (got "
        f"{_r2.status_code}: {_r2.text[:100]!r})",
    )


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll C5-4/C5-5 (hostile pass 5) checks passed.")
