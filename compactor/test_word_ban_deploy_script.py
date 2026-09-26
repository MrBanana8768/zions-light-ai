"""
Tests for scripts/word-ban-deploy.py — the operator tool that replaces
pasting the word-ban grammar into OpenWebUI's Advanced Params with a file
the compactor injects (v3.1.9.8 FILE DEPLOY).

Covers: deploy (sha check against --sha and against a REAL release.vN.json
fixture, canonical-form check, atomic write, previous.json preserved,
--model recorded, withdrawn revisions refused), status (local file +
best-effort /health/full), rollback (swap, and its own rollback, validates
previous.json's shape/sha/withdrawn-ness before swapping — hostile review
round 1, F4), clear (refuses without --yes), and that nothing here ever
prints the grammar content itself.

    python test_word_ban_deploy_script.py
"""

import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "word-ban-deploy.py"

_spec = importlib.util.spec_from_file_location("word_ban_deploy_script", _SCRIPT)
wbd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wbd)

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


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = wbd.main(argv)
    return code, buf.getvalue()


GRAMMAR = "root ::= 'a' 'b' 'c'"  # a stand-in, never real grammar content


def make_value_file(d: Path, name="value.txt", grammar=GRAMMAR):
    text = json.dumps({"grammar": grammar}, ensure_ascii=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p, hashlib.sha256(text.encode("utf-8")).hexdigest()


# Hostile review round 1, F3: a byte-for-byte COPY of the REAL release
# manifest `wordban.py cmd_release` writes — `~/zl-ops/word-ban/
# release.v8.json` on this machine, copied here verbatim after manually
# confirming (2026-09-26) it contains no grammar/banlist content: every
# value in it is a sha256 hex digest, an ISO timestamp, an integer
# revision, or a bool (`verify.pass`, `e2e`) — walked recursively, nothing
# over 64 characters. This is the shape the old code's `rel.get(
# "value_sha256")` read (a key that plain does not exist at top level)
# and would have refused on every real release ever produced.
REAL_RELEASE_V8_JSON = {
    "revision": 8,
    "released": "2026-09-26T04:40:45Z",
    "files": {
        "word-ban-structured_outputs.value.v8.txt":
            "a40a9de837571ded8f5857e65acd156c83f4d9dd7bd8d7ef065adbf7eb89ce80",
        "wb_ban.v8.ebnf":
            "975f451e901b488528028132c1b18a87dfc637ea8cdf1f97b9a38dc9b27b07e4",
        "word-ban-structured_outputs.value.txt":
            "a40a9de837571ded8f5857e65acd156c83f4d9dd7bd8d7ef065adbf7eb89ce80",
        "word-ban-check.py":
            "22e2d675a5e4d6c0fdebb2189b7bf68eaa2f1e20d1e359416d230fd0a4bf83c3",
        "word-ban-check.v8.py":
            "22e2d675a5e4d6c0fdebb2189b7bf68eaa2f1e20d1e359416d230fd0a4bf83c3",
        "word-ban-prompt-line.v8.txt":
            "c7b91459fb2482df222fe04ad923d938bd56d16d37102ac40b936b0ca6bfad8d",
        "word-ban.md":
            "3b16dc854fe2897c7235b3446c05f112b4b3ddc1774bc1069e3455d5010f4351",
        "word-ban.v8.md":
            "3b16dc854fe2897c7235b3446c05f112b4b3ddc1774bc1069e3455d5010f4351",
    },
    "grammar_sha256": "975f451e901b488528028132c1b18a87dfc637ea8cdf1f97b9a38dc9b27b07e4",
    "banlist_sha256": "f7de74f4838bfc214140342b21bb55c7ca65c8c71f3869ea392d38f85209be08",
    "verify": {"pass": True, "when": "2026-09-26T04:32:00Z"},
    "e2e": True,
}


# ===========================================================================
# [U1] canonicalization matches compactor/wordban_runtime.py's function
# ===========================================================================

print("[U1] canonicalization matches wordban_runtime.canonical_value_bytes")
sys.path.insert(0, str(_HERE))
import wordban_runtime  # noqa: E402
so = {"grammar": GRAMMAR}
assert_eq(
    wbd.canonical_value_bytes(so), wordban_runtime.canonical_value_bytes(so),
    "identical canonicalization between the deploy script and the compactor",
)


# ===========================================================================
# [U2] deploy
# ===========================================================================

print("\n[U2] deploy: --sha check, release.vN.json check, canonical-form check, atomic write")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    active = tdp / "active" / "active.json"
    value_path, sha = make_value_file(tdp)

    # Wrong --sha -> refused, nothing written.
    code, out = run(["deploy", str(value_path), "--revision", "1",
                      "--sha", "0" * 64, "--file", str(active)])
    assert_eq(code, 1, "wrong --sha is refused (exit 1)")
    assert_true(not active.exists(), "nothing written on refusal")
    assert_true(GRAMMAR not in out, "refusal output never contains the grammar")

    # Correct --sha -> deployed.
    code, out = run(["deploy", str(value_path), "--revision", "1",
                      "--sha", sha, "--file", str(active)])
    assert_eq(code, 0, "correct --sha deploys (exit 0)")
    assert_true(active.is_file(), "active.json written")
    doc = json.loads(active.read_text(encoding="utf-8"))
    assert_eq(doc["revision"], 1, "revision recorded")
    assert_eq(doc["value_sha256"], sha, "value_sha256 recorded")
    assert_eq(doc["structured_outputs"], {"grammar": GRAMMAR}, "structured_outputs recorded (parsed, not the raw string)")
    assert_true("deployed_at" in doc, "deployed_at recorded")
    assert_true(GRAMMAR not in out, "deploy output never contains the grammar")

    # A release.vN.json beside the value file, no --sha given — built in
    # the REAL shape (files: {name: sha256_string, ...}), NOT the old
    # top-level `value_sha256` shape the pre-F3-fix code read (which no
    # release.vN.json the real tool writes has ever had — see
    # REAL_RELEASE_V8_JSON above).
    value_path2, sha2 = make_value_file(tdp, name="value2.txt", grammar=GRAMMAR + "-v2")
    real_shape_manifest = {
        "revision": 2,
        "released": "2026-09-26T00:00:00Z",
        "files": {"value2.txt": sha2, "wb_ban.v2.ebnf": "0" * 64},
        "grammar_sha256": "0" * 64,
        "banlist_sha256": "0" * 64,
        "verify": {"pass": True, "when": "2026-09-26T00:00:00Z"},
        "e2e": True,
    }
    assert_eq(
        set(real_shape_manifest.keys()), set(REAL_RELEASE_V8_JSON.keys()),
        "this test's synthetic manifest uses the SAME top-level keys as the real v8 one",
    )
    (tdp / "release.v2.json").write_text(json.dumps(real_shape_manifest), encoding="utf-8")
    code, out = run(["deploy", str(value_path2), "--revision", "2", "--file", str(active)])
    assert_eq(code, 0, "a REAL-shaped release.vN.json supplies the expected sha when --sha is omitted")
    doc2 = json.loads(active.read_text(encoding="utf-8"))
    assert_eq(doc2["revision"], 2, "revision 2 now active")
    assert_eq(doc2["model"], wbd.DEFAULT_MODEL, "model defaults to DEFAULT_MODEL when --model is omitted")
    prev = json.loads((tdp / "active" / "previous.json").read_text(encoding="utf-8"))
    assert_eq(prev["revision"], 1, "the OLD active.json was preserved as previous.json")

    # A release.vN.json for the WRONG revision (files entry present, but
    # top-level "revision" disagrees with --revision) is refused, not
    # silently trusted.
    value_path2b, sha2b = make_value_file(tdp, name="value2b.txt", grammar=GRAMMAR + "-v2b")
    wrong_rev_manifest = dict(real_shape_manifest)
    wrong_rev_manifest["files"] = {"value2b.txt": sha2b}
    wrong_rev_manifest["revision"] = 999
    (tdp / "release.v33.json").write_text(json.dumps(wrong_rev_manifest), encoding="utf-8")
    code, out = run(["deploy", str(value_path2b), "--revision", "33", "--file", str(active)])
    assert_eq(code, 1, "a release.vN.json whose OWN revision disagrees with --revision is refused")
    assert_true("revision" in out, "the refusal names the revision mismatch")

    # --model is recorded; defaults to DEFAULT_MODEL when omitted (already
    # checked above for revision 2); an explicit value is recorded too.
    value_path2c, sha2c = make_value_file(tdp, name="value2c.txt", grammar=GRAMMAR + "-v2c")
    code, out = run(["deploy", str(value_path2c), "--revision", "10", "--sha", sha2c,
                      "--model", "some-other-model", "--file", str(active)])
    assert_eq(code, 0, "deploy with an explicit --model succeeds")
    doc2c = json.loads(active.read_text(encoding="utf-8"))
    assert_eq(doc2c["model"], "some-other-model", "the explicit --model is recorded, not the default")

    # Revision 4 (withdrawn — it traps the model) is refused outright,
    # even with a correct --sha.
    value_path4, sha4 = make_value_file(tdp, name="value4.txt", grammar=GRAMMAR + "-v4")
    code, out = run(["deploy", str(value_path4), "--revision", "4", "--sha", sha4, "--file", str(active)])
    assert_eq(code, 1, "revision 4 (withdrawn) is refused by deploy, even with a correct --sha")
    assert_true("withdrawn" in out, "the refusal names it withdrawn")

    # No --sha and no release.vN.json -> refused (usage).
    value_path3, _ = make_value_file(tdp, name="value3.txt", grammar=GRAMMAR + "-v3")
    code, out = run(["deploy", str(value_path3), "--revision", "3", "--file", str(active)])
    assert_eq(code, 2, "no --sha and no release.vN.json -> usage error (exit 2)")

    # Non-canonical value.txt (a hand-edit that still hashes to --sha
    # because we pass --sha computed over the SAME broken bytes, but does
    # not round-trip through json.loads -> canonical json.dumps) is
    # refused — proves the round-trip check, not just the raw sha check.
    broken = tdp / "broken.txt"
    broken_text = json.dumps({"grammar": GRAMMAR}, ensure_ascii=True) + "\n"  # trailing newline breaks canonical form
    broken.write_text(broken_text, encoding="utf-8")
    broken_sha = hashlib.sha256(broken_text.encode("utf-8")).hexdigest()
    code, out = run(["deploy", str(broken), "--revision", "9", "--sha", broken_sha, "--file", str(active)])
    assert_eq(code, 1, "a non-canonical value.txt is refused even though its own sha256 matches --sha")
    assert_true("canonical" in out, "the refusal names the canonical-form check")


# ===========================================================================
# [U2c] Hostile review round 2, N2: a withdrawn VALUE is refused regardless
# of --revision label. WITHDRAWN_REVISIONS matches by LABEL only, so rev
# 4's actual bytes redeployed as "--revision 5" would sail past it; the
# real rev-4 content is never embedded here (no grammar in this repo — see
# module docstring), so this monkeypatches WITHDRAWN_VALUE_SHA256S with a
# synthetic stand-in sha to prove the MECHANISM: matching by value sha,
# independent of whatever revision label is passed.
# ===========================================================================

print("\n[U2c] N2 fix: a withdrawn VALUE is refused regardless of --revision label")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    active = tdp / "active.json"
    trap_path, trap_sha = make_value_file(tdp, name="trap.txt", grammar=GRAMMAR + "-trap")

    _orig_withdrawn_values = dict(wbd.WITHDRAWN_VALUE_SHA256S)
    wbd.WITHDRAWN_VALUE_SHA256S.clear()
    wbd.WITHDRAWN_VALUE_SHA256S[trap_sha] = "synthetic stand-in for a withdrawn value"
    try:
        # Deployed under revision 5 — an innocent label — the withdrawn
        # VALUE must still be refused.
        code, out = run(["deploy", str(trap_path), "--revision", "5",
                          "--sha", trap_sha, "--file", str(active)])
        assert_eq(code, 1, "a withdrawn VALUE is refused even under an innocent --revision label")
        assert_true("withdrawn" in out, "the refusal names it withdrawn")
        assert_true(not active.exists(), "nothing written on refusal")
        assert_true(GRAMMAR not in out, "the refusal never contains the grammar")
    finally:
        wbd.WITHDRAWN_VALUE_SHA256S.clear()
        wbd.WITHDRAWN_VALUE_SHA256S.update(_orig_withdrawn_values)


# ===========================================================================
# [U2d] Hostile review round 2, N1: --model defaults to $MODEL_REPO when
# set, else DEFAULT_MODEL.
# ===========================================================================

print("\n[U2d] N1 fix: --model defaults to $MODEL_REPO when set, else DEFAULT_MODEL")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    active = tdp / "active.json"
    _saved_model_repo = os.environ.pop("MODEL_REPO", None)
    try:
        v, s = make_value_file(tdp, name="md1.txt", grammar=GRAMMAR + "-md1")
        code, out = run(["deploy", str(v), "--revision", "20", "--sha", s, "--file", str(active)])
        assert_eq(code, 0, "deploy succeeds with MODEL_REPO unset")
        doc = json.loads(active.read_text(encoding="utf-8"))
        assert_eq(doc["model"], wbd.DEFAULT_MODEL,
                  "MODEL_REPO unset -> --model defaults to DEFAULT_MODEL")

        os.environ["MODEL_REPO"] = "pod-configured-model"
        v2, s2 = make_value_file(tdp, name="md2.txt", grammar=GRAMMAR + "-md2")
        code, out = run(["deploy", str(v2), "--revision", "21", "--sha", s2, "--file", str(active)])
        assert_eq(code, 0, "deploy succeeds with MODEL_REPO set")
        doc2 = json.loads(active.read_text(encoding="utf-8"))
        assert_eq(doc2["model"], "pod-configured-model",
                  "MODEL_REPO set -> --model defaults to it")

        v3, s3 = make_value_file(tdp, name="md3.txt", grammar=GRAMMAR + "-md3")
        code, out = run(["deploy", str(v3), "--revision", "22", "--sha", s3,
                          "--model", "explicit-override", "--file", str(active)])
        assert_eq(code, 0, "deploy succeeds with both MODEL_REPO set and --model given")
        doc3 = json.loads(active.read_text(encoding="utf-8"))
        assert_eq(doc3["model"], "explicit-override",
                  "an explicit --model still overrides $MODEL_REPO")
    finally:
        if _saved_model_repo is None:
            os.environ.pop("MODEL_REPO", None)
        else:
            os.environ["MODEL_REPO"] = _saved_model_repo


# ===========================================================================
# [U2b] F3 fix: _sha_from_release_manifest directly against the REAL
# release.v8.json shape (no grammar bytes needed — the function only
# reads the manifest and `value_path.name`).
# ===========================================================================

print("\n[U2b] _sha_from_release_manifest against the REAL release.v8.json shape")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    rel_path = tdp / "release.v8.json"
    rel_path.write_text(json.dumps(REAL_RELEASE_V8_JSON), encoding="utf-8")
    value_stand_in = tdp / "word-ban-structured_outputs.value.v8.txt"  # need not exist on disk

    sha, err = wbd._sha_from_release_manifest(rel_path, value_stand_in, 8)
    assert_eq(err, None, "no error parsing the REAL top-level shape (revision/released/files/...)")
    assert_eq(
        sha, "a40a9de837571ded8f5857e65acd156c83f4d9dd7bd8d7ef065adbf7eb89ce80",
        "extracts the correct sha256 from files[value_path.name], not a nonexistent top-level value_sha256",
    )

    # The OLD (pre-fix) shape this code used to expect — a bare top-level
    # value_sha256 — is what NO real release.vN.json has ever contained;
    # confirm it is correctly rejected as unusable, not silently misread.
    old_shape_path = tdp / "release.v99.json"
    old_shape_path.write_text(json.dumps({"value_sha256": "0" * 64, "revision": 99}), encoding="utf-8")
    sha2, err2 = wbd._sha_from_release_manifest(old_shape_path, value_stand_in, 99)
    assert_eq(sha2, None, "the OLD, nonexistent top-level value_sha256 shape yields no sha")
    assert_true(err2 is not None, "...with an explicit error, not a silent None")

    # A revision mismatch between the manifest and --revision is caught.
    sha3, err3 = wbd._sha_from_release_manifest(rel_path, value_stand_in, 7)
    assert_eq(sha3, None, "a release manifest for a DIFFERENT revision than --revision is refused")
    assert_true(err3 is not None and "revision" in err3, "the error names the revision mismatch")

    # A dict-shaped files entry (defensive fallback) is also accepted.
    dict_entry_path = tdp / "release.v8b.json"
    dict_shape = dict(REAL_RELEASE_V8_JSON)
    dict_shape["revision"] = 8
    dict_shape["files"] = dict(REAL_RELEASE_V8_JSON["files"])
    dict_shape["files"]["word-ban-structured_outputs.value.v8.txt"] = {"sha256": "f" * 64}
    dict_entry_path.write_text(json.dumps(dict_shape), encoding="utf-8")
    sha4, err4 = wbd._sha_from_release_manifest(dict_entry_path, value_stand_in, 8)
    assert_eq(err4, None, "a dict-shaped files entry is accepted (defensive fallback)")
    assert_eq(sha4, "f" * 64, "the dict entry's sha256 is extracted")


# ===========================================================================
# [U3] status
# ===========================================================================

print("\n[U3] status: local file, unreachable compactor is non-fatal")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    active = tdp / "active.json"
    code, out = run(["status", "--file", str(active)])
    assert_eq(code, 0, "status on a missing file is not an error")
    assert_true("no active deploy" in out, "says there is no active deploy")

    value_path, sha = make_value_file(tdp)
    run(["deploy", str(value_path), "--revision", "5", "--sha", sha, "--file", str(active)])
    code, out = run(["status", "--file", str(active), "--host", "127.0.0.1", "--port", "1",
                      "--timeout", "0.2"])
    assert_eq(code, 0, "status still succeeds when the compactor is unreachable")
    assert_true("revision=5" in out, "reports the active revision")
    assert_true(f"model={wbd.DEFAULT_MODEL}" in out, "reports the deployed model")
    assert_true(GRAMMAR not in out, "status output never contains the grammar")


# ===========================================================================
# [U4] rollback
# ===========================================================================

print("\n[U4] rollback: swaps previous back in, and can itself be rolled back")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    active = tdp / "active.json"

    code, out = run(["rollback", "--file", str(active)])
    assert_eq(code, 1, "no previous.json -> refused")

    v1, s1 = make_value_file(tdp, name="v1.txt", grammar=GRAMMAR + "-1")
    v2, s2 = make_value_file(tdp, name="v2.txt", grammar=GRAMMAR + "-2")
    run(["deploy", str(v1), "--revision", "1", "--sha", s1, "--file", str(active)])
    run(["deploy", str(v2), "--revision", "2", "--sha", s2, "--file", str(active)])

    code, out = run(["rollback", "--file", str(active)])
    assert_eq(code, 0, "rollback succeeds")
    doc = json.loads(active.read_text(encoding="utf-8"))
    assert_eq(doc["revision"], 1, "revision 1 is active again")

    # Rolling back again swaps revision 2 back in (the rollback preserved
    # what it replaced as the new previous.json).
    code, out = run(["rollback", "--file", str(active)])
    assert_eq(code, 0, "a rollback of a rollback succeeds")
    doc = json.loads(active.read_text(encoding="utf-8"))
    assert_eq(doc["revision"], 2, "revision 2 is active again (rollback undone)")


print("\n[U4b] F4 fix: rollback validates previous.json before swapping")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    active = tdp / "active.json"
    previous = tdp / "previous.json"

    # A previous.json whose value_sha256 does not match its own
    # structured_outputs (corrupted / hand-edited) is refused, not
    # trusted.
    v1, s1 = make_value_file(tdp, name="v1.txt")
    corrupt_prev = {
        "revision": 1, "value_sha256": "0" * 64,
        "structured_outputs": {"grammar": GRAMMAR}, "model": wbd.DEFAULT_MODEL,
        "deployed_at": "2026-09-26T00:00:00Z",
    }
    previous.write_text(json.dumps(corrupt_prev), encoding="utf-8")
    active.write_text(json.dumps({
        "revision": 2, "value_sha256": hashlib.sha256(
            json.dumps({"grammar": GRAMMAR + "-2"}, ensure_ascii=True).encode()).hexdigest(),
        "structured_outputs": {"grammar": GRAMMAR + "-2"}, "model": wbd.DEFAULT_MODEL,
        "deployed_at": "2026-09-26T00:00:00Z",
    }), encoding="utf-8")
    code, out = run(["rollback", "--file", str(active)])
    assert_eq(code, 1, "a previous.json whose value_sha256 does not match its own value is refused")
    assert_true("corrupted" in out or "hand-edited" in out, "the refusal explains why")
    assert_eq(json.loads(active.read_text(encoding="utf-8"))["revision"], 2,
              "active.json was NOT swapped on a refused rollback")

    # A previous.json for a WITHDRAWN revision (4) is refused, even if it
    # is otherwise well-formed and self-consistent.
    withdrawn_grammar = {"grammar": GRAMMAR + "-withdrawn"}
    withdrawn_sha = hashlib.sha256(
        json.dumps(withdrawn_grammar, ensure_ascii=True).encode()).hexdigest()
    previous.write_text(json.dumps({
        "revision": 4, "value_sha256": withdrawn_sha,
        "structured_outputs": withdrawn_grammar, "model": wbd.DEFAULT_MODEL,
        "deployed_at": "2026-09-26T00:00:00Z",
    }), encoding="utf-8")
    code, out = run(["rollback", "--file", str(active)])
    assert_eq(code, 1, "rollback refuses a withdrawn revision (4), even if internally consistent")
    assert_true("withdrawn" in out, "the refusal names it withdrawn")

    # A well-formed, non-withdrawn previous.json IS accepted.
    good_grammar = {"grammar": GRAMMAR + "-good"}
    good_sha = hashlib.sha256(json.dumps(good_grammar, ensure_ascii=True).encode()).hexdigest()
    previous.write_text(json.dumps({
        "revision": 1, "value_sha256": good_sha,
        "structured_outputs": good_grammar, "model": wbd.DEFAULT_MODEL,
        "deployed_at": "2026-09-26T00:00:00Z",
    }), encoding="utf-8")
    code, out = run(["rollback", "--file", str(active)])
    assert_eq(code, 0, "a well-formed, non-withdrawn previous.json is accepted")
    assert_eq(json.loads(active.read_text(encoding="utf-8"))["revision"], 1,
              "active.json now holds the validated previous revision")


print("\n[U4c] N2 fix: rollback refuses a withdrawn VALUE regardless of its revision label")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    active = tdp / "active.json"
    previous = tdp / "previous.json"

    trap_grammar = {"grammar": GRAMMAR + "-trap-rollback"}
    trap_sha = hashlib.sha256(
        json.dumps(trap_grammar, ensure_ascii=True).encode("utf-8")).hexdigest()

    _orig_withdrawn_values = dict(wbd.WITHDRAWN_VALUE_SHA256S)
    wbd.WITHDRAWN_VALUE_SHA256S.clear()
    wbd.WITHDRAWN_VALUE_SHA256S[trap_sha] = "synthetic stand-in for a withdrawn value"
    try:
        # previous.json is internally consistent (its value_sha256 DOES
        # match its own structured_outputs) and labelled revision 5 — an
        # innocent label, not 4 — but its VALUE is the withdrawn one.
        previous.write_text(json.dumps({
            "revision": 5, "value_sha256": trap_sha,
            "structured_outputs": trap_grammar, "model": wbd.DEFAULT_MODEL,
            "deployed_at": "2026-09-26T00:00:00Z",
        }), encoding="utf-8")
        active.write_text(json.dumps({
            "revision": 6, "value_sha256": hashlib.sha256(
                json.dumps({"grammar": GRAMMAR + "-6"}, ensure_ascii=True).encode()).hexdigest(),
            "structured_outputs": {"grammar": GRAMMAR + "-6"}, "model": wbd.DEFAULT_MODEL,
            "deployed_at": "2026-09-26T00:00:00Z",
        }), encoding="utf-8")
        code, out = run(["rollback", "--file", str(active)])
        assert_eq(code, 1, "rollback refuses a withdrawn VALUE even under an innocent revision label")
        assert_true("withdrawn" in out, "the refusal names it withdrawn")
        assert_eq(json.loads(active.read_text(encoding="utf-8"))["revision"], 6,
                  "active.json was NOT swapped on a refused rollback")
        assert_true(GRAMMAR not in out, "the refusal never contains the grammar")
    finally:
        wbd.WITHDRAWN_VALUE_SHA256S.clear()
        wbd.WITHDRAWN_VALUE_SHA256S.update(_orig_withdrawn_values)


# ===========================================================================
# [U5] clear
# ===========================================================================

print("\n[U5] clear: refuses without --yes")

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    active = tdp / "active.json"
    v1, s1 = make_value_file(tdp)
    run(["deploy", str(v1), "--revision", "1", "--sha", s1, "--file", str(active)])

    code, out = run(["clear", "--file", str(active)])
    assert_eq(code, 2, "clear without --yes is refused (usage)")
    assert_true(active.is_file(), "nothing removed without --yes")

    code, out = run(["clear", "--file", str(active), "--yes"])
    assert_eq(code, 0, "clear with --yes succeeds")
    assert_true(not active.exists(), "active.json removed")

    code, out = run(["clear", "--file", str(active), "--yes"])
    assert_eq(code, 0, "clearing an already-absent file is a no-op, not an error")


print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All test_word_ban_deploy_script.py checks passed.")
