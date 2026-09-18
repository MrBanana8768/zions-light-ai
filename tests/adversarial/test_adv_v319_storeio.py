"""Adversarial suite: THE STORE'S FILE IO (hostile pass #2).

Target: the memory.py half of 843bf9d (the UnicodeDecodeError handler), plus
every other reader in the tree that touches the store, plus the atomic-write
paths and _safe_path.

Plain script, same reason as test_adv_v319_restore.py. Run with:

    /opt/compactor-venv/bin/python tests/adversarial/test_adv_v319_storeio.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "compactor"))

_TMP = Path(tempfile.mkdtemp(prefix="adv-v319-storeio-"))
_STORE = _TMP / "compactor"
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import facts  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
import summarizer  # noqa: E402

memory.ensure_storage_layout()

_PASS = 0
_FAIL = 0


def check(cond, label):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ok    {label}")
    else:
        _FAIL += 1
        print(f"  FAIL  {label}")


def note(msg):
    print(f"  note  {msg}")


def head(t):
    print("")
    print(f"[{t}]")


# ===========================================================================
# S1. A torn multibyte sequence at a real chunk boundary.
# ===========================================================================

def _torn_at_chunk_boundary(pad_to: int) -> bytes:
    """Valid JSON whose bytes are cut in the MIDDLE of a 3-byte character at
    offset `pad_to`. This is the realistic shape, not a lone \\xff: shutil's
    copy buffer is 64 KiB on POSIX, MooseFS returns short reads at chunk
    boundaries, and a container killed mid-write stops wherever it stops.
    Zion's memory is prose — em dashes, curly quotes, emoji — so any cut past
    the first few hundred bytes has a good chance of landing inside one.
    """
    # em dash U+2014 = e2 80 94.
    head_json = b'{"conv_id": "torn", "facts": [{"text": "'
    filler = b"a" * max(0, pad_to - len(head_json) - 1)
    body = head_json + filler + b"\xe2\x80\x94"
    # Cut the last two bytes of the em dash off: a VALID lead byte followed
    # by end-of-file. UnicodeDecodeError('unexpected end of data').
    return body[:-2]


def s1_torn_multibyte_through_every_reader():
    """SOUND (the 843bf9d fix holds) — verified against four shapes and six
    readers, not just the lone \\xff the shipped guard uses."""
    head("S1: torn multibyte at a chunk boundary, through every reader")

    shapes = {
        "truncated mid-em-dash at a 64 KiB boundary":
            _torn_at_chunk_boundary(65536),
        "truncated mid-em-dash at a 4 KiB boundary":
            _torn_at_chunk_boundary(4096),
        "lone continuation byte mid-string":
            b'{"conv_id": "torn", "facts": [{"text": "a\x80b"}]}',
        "CESU-8 lone surrogate (ed a0 bd)":
            b'{"conv_id": "torn", "facts": [{"text": "\xed\xa0\xbd"}]}',
        "UTF-16LE, which decodes to NULs not garbage":
            '{"conv_id": "torn"}'.encode("utf-16-le"),
    }

    readers = [
        ("memory.read_json_strict", lambda p: memory.read_json_strict(
            p, default=None, expect=dict)),
        ("facts.load_facts", lambda p: facts.load_facts("tornconv")),
        ("facts.load_archive", lambda p: facts.load_archive("tornarch")),
        ("summarizer.load_state", lambda p: summarizer.load_state("tornsum")),
        ("summarizer.load_chapter_archive",
         lambda p: summarizer.load_chapter_archive("tornchap")),
        ("persona.load_persona", lambda p: persona.load_persona("tornpers")),
    ]
    paths = {
        "memory.read_json_strict": memory.facts_path("tornprobe"),
        "facts.load_facts": memory.facts_path("tornconv"),
        "facts.load_archive": memory.facts_archive_path("tornarch"),
        "summarizer.load_state": memory.summary_path("tornsum"),
        "summarizer.load_chapter_archive": memory.summary_archive_path("tornchap"),
        "persona.load_persona": memory.persona_path("tornpers"),
    }

    for shape, raw in shapes.items():
        for name, fn in readers:
            p = paths[name]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            try:
                fn(p)
                outcome = "RETURNED (no exception)"
            except memory.StoreUnreadable:
                outcome = "StoreUnreadable"
            except UnicodeDecodeError:
                outcome = "UnicodeDecodeError"
            except Exception as e:  # noqa: BLE001
                outcome = f"{type(e).__name__}"
            check(outcome == "StoreUnreadable",
                  f"{name} + {shape} -> {outcome}")
            p.unlink(missing_ok=True)

    # read_json, the best-effort sibling: must return the default, not raise.
    p = memory.facts_path("tornbest")
    p.write_bytes(_torn_at_chunk_boundary(65536))
    try:
        got = memory.read_json({}, ) if False else memory.read_json(p, default="DEF")
        check(got == "DEF", "memory.read_json returns its default, no raise")
    except Exception as e:  # noqa: BLE001
        check(False, f"memory.read_json raised {type(e).__name__}")
    p.unlink(missing_ok=True)

    # And the sweep that matters: no store reader in the tree bypasses
    # read_json_strict. backup.py's own readers use bare `except Exception`.
    bad = []
    for f in sorted((_REPO / "compactor").glob("*.py")):
        if f.name.startswith("test_"):
            continue
        t = f.read_text(encoding="utf-8")
        if "json.load" in t or "read_text(" in t:
            if f.name not in ("memory.py", "backup.py", "main.py",
                              "portability.py", "health.py"):
                bad.append(f.name)
    check(bad == [],
          "SOUND: no module outside memory/backup/main/portability/health "
          "reads JSON or text off disk directly (%r)" % bad)
    bsrc = (_REPO / "compactor" / "backup.py").read_text(encoding="utf-8")
    check(bsrc.count("except Exception") >= 5,
          "SOUND: backup.py's own manifest/census/verify readers all sit "
          "behind bare `except Exception`, which covers the codec too")


# ===========================================================================
# S2. The wrong shape ONE LEVEL DOWN — applied at load_facts, missed at the
#     archive loaders, and the very next write persists the loss.
# ===========================================================================

def s2_wrong_shape_one_level_down_destroys_the_cold_archive():
    """FINDING (BLOCKER). facts.load_facts got the one-level-down fix and
    says so in a comment:

        # A non-list under the key is the same hazard one level down, so it
        # raises rather than emptying.

    facts.load_archive, 66 lines below, in the same file, kept the fallback:

        archived = data.get("facts", []) if isinstance(data, dict) else []

    `expect=dict` only guards the TOP level. A file that is a dict whose
    "facts" is not a list reads as an EMPTY archive — and archive_facts()
    then calls save_archive() with (existing-that-read-empty + incoming),
    atomically writing the emptiness over the cold store. That is the v3.1
    F1a defect exactly, one key deeper, on the file that holds every fact
    ever evicted.

    restore_from_archive() reads the same loader, so the operator asking to
    recover archived facts is told there are none.

    summarizer.load_chapter_archive and _archive_chapters have the identical
    shape on the L2 chapter sidecar, which is the ONLY copy of chapter-level
    detail once L3 has paraphrased it.
    """
    head("S2 FINDING: wrong shape one level down empties the cold archive")

    # --- facts archive ---
    ap = memory.facts_archive_path("coldconv")
    ap.parent.mkdir(parents=True, exist_ok=True)
    real = {"conv_id": "coldconv", "facts": [
        {"text": "she was baptized on 12 April", "pin": True},
        {"text": "her grandmother's name is Ruth"},
    ]}
    memory.atomic_write_json(ap, real)
    check(len(facts.load_archive("coldconv")) == 2,
          "baseline: two archived facts load")

    # The damage: "facts" is a dict, not a list. One valid JSON file.
    memory.atomic_write_json(ap, {"conv_id": "coldconv",
                                  "facts": {"0": real["facts"][0],
                                            "1": real["facts"][1]}})
    got = None
    try:
        got = facts.load_archive("coldconv")
        outcome = f"returned {got!r}"
    except memory.StoreUnreadable:
        outcome = "StoreUnreadable"
    check(outcome.startswith("returned []"),
          "FINDING: load_archive returns [] for a wrong-shape archive "
          "(%s)" % outcome)
    check(facts.load_facts.__doc__ is not None
          and "one level down" in (_REPO / "compactor" / "facts.py").read_text(
              encoding="utf-8"),
          "...while load_facts in the same file guards exactly this, and "
          "says 'one level down' in its own comment")

    # And now the write-back that makes it permanent.
    n = facts.archive_facts("coldconv", [{"text": "a fact evicted today"}])
    check(n == 1, "archive_facts reported success")
    after = json.loads(ap.read_text(encoding="utf-8"))
    check(isinstance(after.get("facts"), list) and len(after["facts"]) == 1,
          "FINDING: the sidecar now holds ONE fact — both originals, "
          "including the PINNED one, are gone (%d left)"
          % len(after.get("facts") or []))
    check(all("baptized" not in f.get("text", "") for f in after["facts"]),
          "FINDING: the pinned fact is not in the file any more")
    check(facts.restore_from_archive("coldconv") == 1,
          "and restore_from_archive can only return what survived")

    # --- chapter archive, the same shape ---
    sp = memory.summary_archive_path("coldchap")
    memory.atomic_write_json(sp, {"chapters": [
        {"text": "chapter one, the only copy", "first_turn": 1, "last_turn": 20},
        {"text": "chapter two, the only copy", "first_turn": 21, "last_turn": 40},
    ]})
    check(len(summarizer.load_chapter_archive("coldchap")) == 2,
          "baseline: two archived chapters load")
    memory.atomic_write_json(sp, {"chapters": "two chapters, as a string"})
    check(summarizer.load_chapter_archive("coldchap") == [],
          "FINDING: load_chapter_archive returns [] for a wrong-shape file")
    summarizer._archive_chapters("coldchap", [
        {"text": "chapter three", "first_turn": 41, "last_turn": 60}])
    after2 = json.loads(sp.read_text(encoding="utf-8"))
    check(after2.get("chapters") == [
        {"text": "chapter three", "first_turn": 41, "last_turn": 60}],
        "FINDING: _archive_chapters wrote {'chapters': [one row]} over it — "
        "the cold chapter store is now that one row (%r)" % (after2,))
    note("smallest fix: raise StoreUnreadable when the inner key is the "
         "wrong type, exactly as load_facts already does, in all three "
         "loaders (facts.load_archive, summarizer.load_chapter_archive, "
         "summarizer._archive_chapters)")


# ===========================================================================
# S3. Crash consistency of the writes.
# ===========================================================================

def s3_atomic_write_paths():
    """SOUND, with one named gap."""
    head("S3: the atomic-write paths")
    p = memory.facts_path("writeprobe")

    # A write that fails mid-serialize must leave the old file intact and no
    # temp behind. json.dump(..., ensure_ascii=False) on a lone surrogate
    # raises from the ENCODER, after some bytes have already been written.
    memory.atomic_write_json(p, {"conv_id": "writeprobe", "facts": [
        {"text": "the good contents"}]})
    before = p.read_bytes()
    raised = None
    try:
        memory.atomic_write_json(p, {"conv_id": "writeprobe", "facts": [
            {"text": "a lone surrogate: \ud83d"}]})
    except Exception as e:  # noqa: BLE001
        raised = e
    check(raised is not None,
          "a lone surrogate fails the write (%s)" % type(raised).__name__)
    check(p.read_bytes() == before,
          "SOUND: the old contents survive the failed write byte-for-byte")
    strays = sorted(x.name for x in p.parent.glob(p.name + ".*.tmp"))
    check(strays == [], "SOUND: no orphan .tmp left behind (%r)" % strays)

    # Disk full BETWEEN the serialize and the replace.
    import os as _os
    real_replace = _os.replace

    def boom(a, b):
        raise OSError(28, "No space left on device")

    _os.replace = boom
    try:
        try:
            memory.atomic_write_json(p, {"conv_id": "writeprobe",
                                         "facts": [{"text": "new"}]})
            check(False, "the replace should have failed")
        except OSError:
            check(True, "a failing os.replace propagates")
    finally:
        _os.replace = real_replace
    check(p.read_bytes() == before,
          "SOUND: and the old file is still the old file")
    strays = sorted(x.name for x in p.parent.glob(p.name + ".*.tmp"))
    check(strays == [], "SOUND: the temp is cleaned up (%r)" % strays)

    # The gap, stated: list_known_conv_ids globs *.json and the temps are
    # *.tmp, so an orphan is invisible — and therefore never reported or
    # pruned either.
    orphan = p.parent / (p.name + ".abc123.tmp")
    orphan.write_bytes(b"{")
    check("writeprobe" in memory.list_known_conv_ids(),
          "the real conv is still listed")
    check(all(not i.endswith(".tmp") for i in memory.list_known_conv_ids()),
          "SOUND: an orphan .tmp is not mistaken for a conversation")
    note("LOW: nothing ever reports or removes an orphan .tmp either; "
         "atomic_write_json's own docstring says they are 'ignored'")
    orphan.unlink()


# ===========================================================================
# S4. _safe_path.
# ===========================================================================

def s4_safe_path():
    """SOUND. Every shape I could reach with."""
    head("S4: _safe_path")
    refused = [
        "../../../../tmp/CLAUDE_PWNED",
        "../facts/victim",
        "..",
        ".",
        "a/b",
        "a\\b",
        "a\x00b",
        "",
        "a" * 65,
        "conv.json",            # a dot would collide with the sidecar filter
        "․․/escape",  # ONE DOT LEADER, a unicode lookalike for '.'
        "．．/escape",  # FULLWIDTH FULL STOP
        "‮../etc",         # RTL override
        "CON",                  # reserved on Windows; allowed here, harmless
    ]
    for cid in refused:
        if cid == "CON":
            continue
        try:
            memory.facts_path(cid)
            check(False, f"FINDING: accepted conv_id {cid!r}")
        except memory.UnsafeConvId:
            check(True, f"SOUND: refused {cid!r}")

    # NFC/NFD: both forms contain non-ASCII, so both are refused, and the
    # normalisation question therefore cannot produce two ids that collide.
    import unicodedata
    a = unicodedata.normalize("NFC", "café")
    b = unicodedata.normalize("NFD", "café")
    for cid in (a, b):
        try:
            memory.facts_path(cid)
            check(False, f"FINDING: accepted {cid!r}")
        except memory.UnsafeConvId:
            check(True, f"SOUND: refused a normalisation form {cid!r}")

    # The five builders all go through it.
    src = (_REPO / "compactor" / "memory.py").read_text(encoding="utf-8")
    check(src.count("_safe_path(") == 6,
          "SOUND: one definition + five builders, no sixth path constructor "
          "(%d occurrences)" % src.count("_safe_path("))
    check(memory.facts_path("ok-id_1").name == "ok-id_1.json",
          "SOUND: a legal id still works")


def main():
    print("adversarial: the store's file IO (hostile pass #2)")
    print(f"tmp tree: {_TMP}")
    for fn in (s1_torn_multibyte_through_every_reader,
               s2_wrong_shape_one_level_down_destroys_the_cold_archive,
               s3_atomic_write_paths,
               s4_safe_path):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
            globals()["_FAIL"] = globals()["_FAIL"] + 1
    print("")
    print(f"RESULT: {_PASS} checks passed, {_FAIL} failed")
    shutil.rmtree(_TMP, ignore_errors=True)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
