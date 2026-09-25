#!/usr/bin/env python3
"""wordban: the word ban on the model's output, as a reproducible tool.

    wordban.py build    [--banlist banlist.yaml] [--build-dir DIR]
    wordban.py verify   [--build-dir DIR] [--corpus webui.db] [--image IMG] [--tokenizer DIR] [--no-image]
    wordban.py e2e      [--build-dir DIR] --webui-db webui.db [--compactor-ref v3.1.9]
    wordban.py diff     OLD NEW [--corpus webui.db] [--show-forms]
    wordban.py release  [--build-dir DIR] [--out-dir ~/zl-ops/word-ban] [--notes FILE]
    wordban.py cycle    --corpus webui.db [--build-dir DIR] [--base-release N] [--no-e2e]
    wordban.py tokenizer [--tokenizer DIR]

See README.md in this directory. The grammar is an xgrammar EBNF grammar that goes into the model's
`structured_outputs` custom parameter in OpenWebUI; vLLM then refuses, token by token, any continuation that
would spell a banned word. Nothing here touches the pod: every check runs locally (CPU), and `verify` / `e2e`
run vLLM 0.19's own code from the published image.

PRIVACY: `--corpus` / `--webui-db` are copied (read-only source) into a private temporary directory that is
deleted when the command ends. Only counts are printed. Nothing from them is ever written to the repo."""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wb_core as core  # noqa: E402

# rev 7 (owner, 2026-09-24): production runs v3.1.9 (not v3.1.9.4 = c1295894, the previous DEFAULT_IMAGE). A
# verification on prod's own image passed revision 6 fully: vLLM 0.19.0, xgrammar 0.2.7, transformers 4.57.6,
# mistral_common 1.11.7 and Unicode 15.0.0 are identical to c1295894; only the compactor differs. Update this
# (and DEFAULT_IMAGE_TAG below) on every deploy - verify warns loudly when --image differs from this default.
DEFAULT_IMAGE = "angreg/zions-light-ai@sha256:f3fafa3e1949115dfd2c1354969b03be8d852fc491b647884ccfe7a23cf998d2"
DEFAULT_IMAGE_TAG = "v3.1.9-cu12"
HOME = Path(os.environ.get("WORDBAN_HOME", "~/zl-ops/word-ban")).expanduser()
TOKENIZER_REPO = "coder3101/Cydonia-24B-v4.3-vision-heretic"
TOKENIZER_REV = "c1e236fb254bec28003e40da1f7e7a0bd426945f"
TOKENIZER_FILES = ["tekken.json", "params.json", "config.json", "generation_config.json"]
TEKKEN_SHA_PREFIX = "6e2501687ccd"
GRAMMAR = "wb_ban.ebnf"
VALUE = "structured_outputs.value.txt"
MANIFEST = "manifest.json"
STAMP = "verify-stamp.json"
E2E_STAMP = "e2e-stamp.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_sha():
    """one sha256 over the tool's own code: a verify stamp is only trusted by the code that wrote it"""
    h = hashlib.sha256()
    for p in sorted(list(HERE.glob("*.py")) + list(HERE.glob("e2e/*"))):
        if p.is_file():
            h.update(p.relative_to(HERE).as_posix().encode() + b"\x00" + p.read_bytes() + b"\x00")
    return h.hexdigest()


def say(*a):
    print(*a, flush=True)


# ============================================================================================================
# build
# ============================================================================================================
def in_image_python(args, argv, mounts):
    """re-run this script inside the image (its vLLM venv has the pinned Unicode version)."""
    cmd = ["docker", "run", "--rm", "--name", f"wb-{argv[0]}-{os.getpid()}", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp", "-v", f"{HERE}:/wb:ro"]
    for src, dst, mode in mounts:
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    cmd += ["--entrypoint", "/opt/vllm-venv/bin/python", args.image, "/wb/wordban.py"] + argv
    return subprocess.call(cmd)


def cmd_build(args):
    bl_path = Path(args.banlist).resolve()
    bl = core.load_yaml(bl_path)
    want = str(bl.get("unicode_version", ""))
    out = Path(args.build_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if want and unicodedata.unidata_version != want and not args.here:
        say(f"local Python has Unicode {unicodedata.unidata_version}, the banlist pins {want}: building inside the image")
        return in_image_python(args, ["build", "--here", "--banlist", "/bl/" + bl_path.name, "--build-dir", "/out"],
                               [(bl_path.parent, "/bl", "ro"), (out, "/out", "rw")])
    t0 = time.time()
    try:
        spec, ebnf = core.build(bl_path)
    except ValueError as e:
        say(f"REFUSED: {e}")
        return 1
    value = json.dumps({"grammar": ebnf}, ensure_ascii=True)
    for f in (GRAMMAR, VALUE, STAMP, E2E_STAMP):
        (out / f).unlink(missing_ok=True)
    (out / GRAMMAR).write_text(ebnf, encoding="utf-8")
    (out / VALUE).write_text(value, encoding="utf-8")
    (out / "banlist.yaml").write_bytes(bl_path.read_bytes())
    traps = spec.dfa_trap_states()
    man = {
        "revision": spec.revision, "banlist_sha256": spec.source_sha,
        "grammar_sha256": sha(out / GRAMMAR), "grammar_bytes": len(ebnf.encode()),
        "value_sha256": sha(out / VALUE), "value_bytes": len(value.encode()),
        "dfa_states": spec.dfa_n, "dfa_states_before_minimising": spec.n_raw, "classes": len(spec.classes),
        "rules": [r["id"] for _, r in spec.rules], "unicode_version": unicodedata.unidata_version,
        "python": sys.version.split()[0], "dfa_trap_states": len(traps), "built_s": round(time.time() - t0, 1),
    }
    (out / MANIFEST).write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    say(f"built revision {spec.revision}: {spec.dfa_n} states ({spec.n_raw} before minimising), "
        f"{len(spec.classes)} character classes, grammar {man['grammar_bytes']:,} B sha256 {man['grammar_sha256']}")
    say(f"  paste value {out / VALUE}: {man['value_bytes']:,} B sha256 {man['value_sha256']}")
    if traps:
        say(f"  WARNING: {len(traps)} DFA state(s) with no normal exit (a trap) - verify will fail")
    return 0


def load_build(build_dir):
    d = Path(build_dir).expanduser().resolve()
    man = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    for f, k in ((GRAMMAR, "grammar_sha256"), (VALUE, "value_sha256")):
        if sha(d / f) != man[k]:
            sys.exit(f"{d / f} does not match its manifest (edited after build?) - run build again")
    return d, man


# ============================================================================================================
# verify
# ============================================================================================================
class Results:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    @property
    def ok(self):
        return all(r["pass"] for r in self.rows)


def flat(lists):
    """a test list (or list of lists) -> strings; a YAML scalar that did not stay a string (true, no, 1.0 ...) is
    an error in the banlist, not something to guess about"""
    out = []
    for x in lists or []:
        for y in (x if isinstance(x, list) else [x]):
            if not isinstance(y, str):
                sys.exit(f"banlist test entry {y!r} is not a string - quote it in the YAML")
            out.append(y)
    return out


def build_probes(spec, bl):
    """e2e probe sentences, built only from the banlist's own test lists (never hardcoded) so they can never
    drift from the rules: every must_pass entry expects True, every must_block entry (listed + derived) expects
    False, plus one or two prose sentences that string several must_pass entries together as ordinary text."""
    T = bl.get("tests") or {}
    mp = flat(T.get("must_pass"))
    mb = flat(T.get("must_block")) + [f for _, f in core.derived_must_block(spec)]
    probes = [(t, True) for t in mp] + [(t, False) for t in mb]
    # " and " (never a separator any family recognises) keeps adjacent entries from accidentally spelling a
    # banned form at the join, e.g. must-pass "Yellow" next to must-pass "light" would otherwise glue into
    # the banned "yellow.light" family across the join punctuation alone.
    half = max(1, len(mp) // 2)
    if mp:
        probes.append((" and ".join(mp[:half]) + ".", True))
    if mp[half:]:
        probes.append((" and ".join(mp[half:]) + ".", True))
    # de-duplicate on (text, expected), keeping first-seen order: a must_block entry can coincide with a derived
    # case (or, rarely, with a must_pass entry under a different expectation, which is a banlist bug elsewhere,
    # not something to hide here) - dict.fromkeys is the same idiom cmd_diff already uses for this.
    return [list(p) for p in dict.fromkeys(probes)]


def fuzz_strings(spec, n, seed=11):
    """strings built from pieces of the banned words, the allowed words, accents, separators and marks"""
    rnd = random.Random(seed)
    words = []
    base = {}
    for rid, f in core.derived_must_block(spec):
        base.setdefault(rid, f.lower())
    for _, r in spec.rules:
        w = base.get(r["id"]) or "".join(ch for ch in r.get("stem", r.get("match", "")) if ch.isalpha())
        words.append(w)
        for a in (r.get("allow_words") or []) + (r.get("allow_prefixes") or []) + \
                 ((r.get("context") or {}).get("allow_words") or []):
            words.append(w + a)
    words += ["los ", "white ", "the ", "evangel", "tri", "Michel", "mari", "in", "en", "re", "a", "o", "s", "es",
              "ed", "ing", "Qz", "ruch", "rich", "ruin", "rue", "Los Angel", "LOS ANGELES", "strange", "change",
              "michelangelo", "marigold", "inflam", "enflam", "proinflam", "candle", "pure", "ev", "str", "flamm",
              "flammab", "ruach", "רוח", "ר", "ו", "ח", "αγγελ", "Αγγελ"]
    for e in spec.escapes:
        words += core.Reference._words(e["trie"])
    acc = ("áàâäãåāăąǎạảéèêëēĕėęěẹíìîïīĭįǐóòôöõøōŏőǒọúùûüūŭůűųǔñńņňçćĉċčșşśŝšğĝġģĺļľłďđÀÁÂÄÅÉÈÊËÍÏÓÖØÚÜÑÇĂŐŠ"
           "аеосрухіјАЕОСРНТХΑΕΟΙΝΚο" + "I1|0")
    seps = [" ", "-", "'", "’", "_", ".", "*", "(", ")", "\n", "\t", " ", "  ", "", "", "", "", ",", "!", "1", "0",
            "·", "—", "`", "|", "/", "**", "​", "‌", "‍", "­", "⁠", "\U0001f468‍",
            "‍\U0001f469", "״", "־"]
    marks = ["́", "̈", "̂", "ּ", "ָ", "ׁ"]
    out = []
    for _ in range(n):
        s = []
        for _k in range(rnd.randint(1, 4)):
            w = rnd.choice(words)
            if rnd.random() < 0.5:
                w = w[:rnd.randint(1, len(w))] + (rnd.choice(words) if rnd.random() < 0.3 else "")
            w = "".join((rnd.choice(acc) if (rnd.random() < 0.12 and c.isalpha()) else
                         (c.upper() if rnd.random() < 0.25 else c)) for c in w)
            if rnd.random() < 0.15:
                i = rnd.randint(0, len(w))
                w = w[:i] + rnd.choice(marks) * rnd.randint(1, 5) + w[i:]
            if rnd.random() < 0.15:
                i = rnd.randint(0, len(w))
                w = w[:i] + rnd.choice(seps) + w[i:]
            s.append(w)
            s.append(rnd.choice(seps))
        out.append("".join(s))
    return out


def private_dir():
    base = HOME / "private"
    base.mkdir(parents=True, exist_ok=True)
    os.chmod(base, 0o700)
    return Path(tempfile.mkdtemp(prefix="wordban-", dir=base))


def copy_db(src, dst_dir):
    dst = Path(dst_dir) / "webui.db"
    shutil.copyfile(src, dst)                      # the source is only read
    os.chmod(dst, 0o600)
    return dst


def local_checks(spec_path, build_dir, man, res, fuzz_n=40000):
    """everything that needs no tokenizer: the artifact's own automaton, the test lists, the reference."""
    bl = core.load_yaml(spec_path)
    ebnf = (build_dir / GRAMMAR).read_text(encoding="utf-8")
    if unicodedata.unidata_version == str(bl.get("unicode_version", "")):
        spec, again = core.build(spec_path)
        res.add("artifact = banlist (rebuilt here, byte-identical)", again == ebnf,
                f"sha256 {core.sha256_bytes(again.encode())[:16]} vs {man['grammar_sha256'][:16]}")
    else:
        spec = core.Spec(bl, spec_path.read_bytes())
        say(f"  (rebuild comparison skipped: this Python has Unicode {unicodedata.unidata_version}, "
            f"the banlist pins {bl.get('unicode_version')}; the image run repeats it)")
    ref = core.Reference(spec)
    t0 = time.time()
    A = core.Automaton(ebnf)
    say(f"  automaton read back from the grammar: {A.n} states, {A.n_intervals} code-point intervals "
        f"({time.time() - t0:.1f}s)")
    traps, prev, _ = A.trap_states()
    res.add("no-trap property, character level (every reachable state has an ordinary way to finish)", not traps,
            f"{len(traps)} trap state(s)" + ("; e.g. after " + ", ".join(repr(A.example_prefix(s, prev)) for s in traps[:6])
                                              if traps else f" among {A.n} states"))
    T = bl.get("tests") or {}
    mb = flat(T.get("must_block")) + [f for _, f in core.derived_must_block(spec)]
    mp = flat(T.get("must_pass"))
    bad = [t for t in mb if A.accepts(t)]
    res.add("must-block (character level)", not bad, f"{len(mb) - len(bad)}/{len(mb)} blocked" +
            (f"; ALLOWED: {bad[:10]}" if bad else ""))
    bad = [t for t in mp if not A.accepts(t)]
    res.add("must-pass (character level)", not bad, f"{len(mp) - len(bad)}/{len(mp)} allowed" +
            (f"; BLOCKED: {bad[:10]}" if bad else ""))
    dis = [t for t in mb + mp if ref.blocked(t) == A.accepts(t)]
    res.add("reference definition agrees on every listed case", not dis, f"{len(mb) + len(mp) - len(dis)}/{len(mb) + len(mp)}"
            + (f"; disagree: {dis[:10]}" if dis else ""))
    # the trap regressions: wherever the grammar stops them, a normal continuation must remain
    tl = []
    for t in flat(T.get("trap_regressions")):
        s, at = A.run(t)
        if s is None:
            s, _ = A.run(t[:at])
        tl.append((t, s in traps))
    res.add("trap regressions (angelĂ, Angelø, angeléñÖs ...): normal exit remains", not any(x for _, x in tl),
            f"{len(tl)} prefixes" + (f"; TRAPPED: {[t for t, x in tl if x]}" if any(x for _, x in tl) else ""))
    # combining-mark runs
    cap, caph = spec.mark_cap, spec.mark_cap_hebrew
    mr = []
    if not cap:
        for k in range(0, 7):
            mr.append(("e", k, True, A.accepts("x e" + "́" * k + " y")))
    if cap:
        for base, mk, lim in (("e", "́", cap), ("a", "̈", cap), ("ש", "ָ", caph or cap)):
            for k in range(0, 7):
                want = k <= lim
                got = A.accepts("x " + base + mk * k + " y")
                mr.append((base, k, want, got))
        mixed = A.accepts("x שָּ́ y")                # 2 Hebrew + 1 other = 3 > 2
        mr.append(("mixed", 3, False, mixed))
    bad = [m for m in mr if m[2] != m[3]]
    res.add(f"combining-mark cap (<= {cap} in a row" + (f", <= {caph} Hebrew points" if caph else "") + ")", not bad,
            f"{len(mr)} runs" + (f"; wrong: {bad}" if bad else ""))
    # differential fuzz: the artifact's automaton vs the independent reference
    t0 = time.time()
    fz = fuzz_strings(spec, fuzz_n)
    dis = [t for t in fz if ref.blocked(t) == A.accepts(t)]
    res.add(f"differential fuzz, grammar vs reference definition ({fuzz_n:,} strings)", not dis,
            f"{fuzz_n - len(dis):,}/{fuzz_n:,} agree ({time.time() - t0:.0f}s)" + (f"; e.g. {dis[:8]}" if dis else ""))
    # rev 7 FINAL (hostile-review fix 6): don't count this as a check at all when no family declares `like:` -
    # an always-trivially-PASSing row ("0 declared") is not a real check, and it was inflating the verify
    # PASSED-count with nothing behind it.
    targets = like_targets(bl)
    if targets:
        bad = like_mismatches(bl)
        res.add("`like:` declarations match the referenced family's separator policy in kind", not bad,
                f"{len(targets)} declared" + (f"; MISMATCH: {bad}" if bad else ""))
    return spec, ref, A, bl


_SEP_RE = re.compile(r"\{SEP:([A-Za-z0-9_]+)\??\}")


def like_rules(bl, name):
    """a family name, or a rule id, -> its raw rule dicts"""
    fams = bl.get("families") or {}
    if name in fams:
        return fams[name].get("rules") or []
    return [r for f in fams.values() for r in (f.get("rules") or []) if r["id"] == name]


def like_shape(bl, name):
    """the checkable part of a family/rule's separator policy (`like: X`, rev 7 hostile-review fix 5): a
    MULTISET (not "last rule wins") of every separator-set NAME referenced anywhere across ALL of the family's
    (or rule's) rules - inside match:/stem: pattern text, context.after, not_after, split:, spaced:,
    phrase_split's inner_sep/outer_sep - plus whether it has a glued (where: inside/anywhere) form at all. This
    is the JOINED-vs-BREAK set distinction owner decision A asks for from any two families declared `like:` each
    other: the exact same separator-set NAMES the exact same number of times each (round 2: no family currently
    declares `like:` - white_flame, the only user, was removed - but the machinery stays for a future one)."""
    names = collections.Counter()
    glued = False
    for r in like_rules(bl, name):
        for key in ("match", "stem", "not_after"):
            if r.get(key):
                names.update(_SEP_RE.findall(str(r[key])))
        ctx = r.get("context") or {}
        if ctx.get("after"):
            names.update(_SEP_RE.findall(str(ctx["after"])))
        if r.get("split"):
            names[r["split"]] += 1
        if r.get("spaced"):
            names[r["spaced"]] += 1
        if r.get("inner_sep"):
            names[r["inner_sep"]] += 1
        if r.get("outer_sep"):
            names[r["outer_sep"]] += 1
        if r.get("where") in ("inside", "anywhere"):
            glued = True
    return dict(names), glued


def like_targets(bl):
    """{declaring family name -> target} for every `like:` in the banlist (a family-level comparison: any rule
    in the family may carry the `like:` key, and they are expected to agree - this takes the family's first)"""
    out = {}
    for fam, fdef in (bl.get("families") or {}).items():
        for r in fdef.get("rules") or []:
            if r.get("like"):
                out.setdefault(fam, r["like"])
    return out


def like_mismatches(bl):
    bad = []
    for name, target in like_targets(bl).items():
        own, tgt = like_shape(bl, name), like_shape(bl, target)
        if own != tgt:
            bad.append(f"{name} like {target}: {own} != {tgt}")
    return bad


# rev 7 round 2: the yellow_light <-> white_flame `like:` pairing requirement (fix 5) was removed along with the
# white_flame family itself (a real grammar-size cost for zero behavioural effect - see banlist.yaml). The
# generic multiset machinery above (`like_shape`/`like_targets`/`like_mismatches`) is kept for a future family
# that actually needs a declared, checked separator-policy match.


def cmd_verify(args):
    build_dir, man = load_build(args.build_dir)
    spec_path = build_dir / "banlist.yaml"
    res = Results()
    say(f"verify revision {man['revision']}: grammar sha256 {man['grammar_sha256']}")
    if not args.no_image and args.image != DEFAULT_IMAGE:
        say(f"  *** WARNING: --image {args.image} is NOT the default (DEFAULT_IMAGE = {DEFAULT_IMAGE}, tag "
            f"{DEFAULT_IMAGE_TAG}) - this verify does not prove what prod actually runs ***")
    say("== local (the grammar's own automaton, the reference definition)")
    spec, ref, A, bl = local_checks(spec_path, build_dir, man, res, fuzz_n=args.fuzz)
    image_part = None
    priv = None
    corpus_meta = None
    try:
        if not args.no_image:
            priv = private_dir()
            T = bl.get("tests") or {}
            must_pass = flat(T.get("must_pass"))
            # rev 7 (Opus review follow-up): the prose probes e2e builds from the banlist's own must_pass entries
            # (build_probes' two " and "-joined sentences) go through the image's real tokenizer too, not just
            # e2e's stub backend - the same fix the probe hardcoding lesson calls for, on the other half of verify.
            must_pass += [t for t, e in build_probes(spec, bl) if e and t not in must_pass]
            tests = {
                "must_block": flat(T.get("must_block")) + [f for _, f in core.derived_must_block(spec)],
                "must_pass": must_pass, "collateral": flat(T.get("collateral")),
                "gaps": flat(T.get("gaps")), "token_paths": T.get("token_paths") or [],
                "trap_regressions": flat(T.get("trap_regressions")),
                "accent": core.accent_cases(spec, T.get("accent_words") or []),
                "mark_cap": spec.mark_cap, "mark_cap_hebrew": spec.mark_cap_hebrew,
                "funnels": bl.get("funnels") or {},
            }
            rnd = random.Random(7)
            cps = [c for c in range(0x20, 0x110000) if not (0xD800 <= c <= 0xDFFF)]
            sample = rnd.sample(cps, 3000) + [ord(c) for c in "하配иτά檢ьıмৰ正の時é😇—“”‘’✅━☼▶👾ñäáü"]
            tests["random_unicode"] = [t for t in ("x " + chr(cp) + " x" for cp in sample) if not ref.blocked(t)]
            (priv / "tests.json").write_text(json.dumps(tests, ensure_ascii=False), encoding="utf-8")
            corpus_args = []
            if args.corpus:
                say("== her corpus (READ-ONLY source copied to a private directory; counts only)")
                db = copy_db(args.corpus, priv)
                corpus_meta = load_corpus_mode(db, args)
                corpus_stats(corpus_meta["msgs"], spec, ref, A, bl, res, priv, mode_desc=corpus_meta["desc"])
                corpus_args = ["--corpus", "/priv/corpus.json", "--corpus-mode", corpus_meta["stamp"]["mode"]]
                db.unlink()
            say("== image: vLLM 0.19's own xgrammar backend + the real tokenizer (CPU)")
            tokd = Path(args.tokenizer).expanduser().resolve()
            if not (tokd / "tekken.json").exists():
                sys.exit(f"no tokenizer at {tokd} - run: wordban.py tokenizer --tokenizer {tokd}")
            cmd = ["docker", "run", "--rm", "--name", f"wb-verify-{os.getpid()}", "--network", "none",
                   "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                   "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "PYTHONUNBUFFERED=1",
                   "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "VLLM_LOGGING_LEVEL=WARNING",
                   "-v", f"{HERE}:/wb:ro", "-v", f"{tokd}:/tok:ro", "-v", f"{build_dir}:/build:ro",
                   "-v", f"{priv}:/priv:rw", "--entrypoint", "/opt/vllm-venv/bin/python", args.image,
                   "/wb/wb_image.py", "verify", "--grammar", "/build/" + GRAMMAR, "--banlist", "/build/banlist.yaml",
                   "--tests", "/priv/tests.json", "--out", "/priv/image-results.json",
                   "--walk-steps", str(args.walk_steps)] + corpus_args
            rc = subprocess.call(cmd)
            rp = priv / "image-results.json"
            if not rp.exists():
                res.add("image run", False, f"no results (exit {rc})")
            else:
                image_part = json.loads(rp.read_text(encoding="utf-8"))
                for r in image_part["rows"]:
                    res.rows.append(r)
                image_part.pop("rows")
    finally:
        if priv:
            shutil.rmtree(priv, ignore_errors=True)
    stamp = {"grammar_sha256": man["grammar_sha256"], "value_sha256": man["value_sha256"],
             "banlist_sha256": man["banlist_sha256"], "revision": man["revision"], "image": args.image,
             "image_tag": DEFAULT_IMAGE_TAG if args.image == DEFAULT_IMAGE else "unknown (--image != DEFAULT_IMAGE)",
             "tool_sha256": tool_sha(),
             "full": not args.no_image, "corpus": bool(args.corpus),
             "corpus_mode": (corpus_meta["stamp"] if corpus_meta else None),
             "pass": res.ok, "rows": res.rows,
             "image_facts": image_part, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (build_dir / STAMP).write_text(json.dumps(stamp, indent=1, ensure_ascii=False), encoding="utf-8")
    n_f = sum(1 for r in res.rows if not r["pass"])
    say(f"\nVERIFY {'PASSED' if res.ok else 'FAILED'}: {len(res.rows) - n_f}/{len(res.rows)} checks"
        + ("" if not args.no_image else "  (local part only: --no-image; release needs the full run)"))
    return 0 if res.ok else 1


CORPUS_STALE_HOURS = 12


def _corpus_age_note(newest_ts):
    """rev 7 FINAL hostile-review fix 3 (MEDIUM): print the age of the corpus's newest message and WARN if it
    is over CORPUS_STALE_HOURS old - a stale backup silently proves nothing about what she is actually seeing
    today (the exact mistake logged in memory.md: "a verifier reported prod state from an old backup")."""
    if not newest_ts:
        return "corpus newest message: unknown (no timestamped messages found)"
    age_h = (time.time() - newest_ts) / 3600
    when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest_ts))
    line = f"corpus newest message: {when} ({age_h:.1f}h old)"
    if age_h > CORPUS_STALE_HOURS:
        line += f"  *** WARNING: over {CORPUS_STALE_HOURS}h old - this may be a STALE backup, not what she is seeing now ***"
    return line


def load_corpus_mode(db, args):
    """rev 7 round 2 (owner): "just validating on the regex that is getting banned is good enough ... I only
    pull the most recent chat for behavior, so only the most recent replies should be scanned." Default: the
    single most-recently-active chat's last `--recent` messages (200 by default); `--full-corpus` opts back into
    scanning every chat's whole current branch, as before. Shared by verify/diff/cycle so their corpus-based
    checks (and stamps) agree on what "the corpus" means for a given invocation. Prints (and stamps) the age of
    the corpus's newest message either way (fix 3)."""
    full = bool(getattr(args, "full_corpus", False))
    if full:
        msgs = core.load_corpus(db)
        newest_ts = max((m["ts"] for m in msgs), default=None)
        say(_corpus_age_note(newest_ts))
        return {"msgs": msgs, "desc": "every chat's current branch (--full-corpus)",
                "stamp": {"mode": "full", "messages": len(msgs), "newest_message_ts": newest_ts}}
    n = int(getattr(args, "recent", 200) or 200)
    msgs, chat_id, chat_hash, newest_ts = core.recent_messages(db, n)
    say(_corpus_age_note(newest_ts))
    return {"msgs": msgs, "desc": f"the most recently active chat's last {n} messages",
            "stamp": {"mode": "recent", "n": n, "chat_id_sha256": chat_hash, "messages": len(msgs),
                      "newest_message_ts": newest_ts}}


def corpus_stats(msgs, spec, ref, A, bl, res, priv, mode_desc="every chat's current branch"):
    import re
    by = {"assistant": 0, "user": 0}
    fam_of = {r["id"]: fam for fam, r in spec.rules}
    fam_of["MARKS"] = "combining-mark cap"
    hits = {"assistant": {}, "user": {}}
    with_hit = {"assistant": 0, "user": 0}
    masked = []
    t0 = time.time()
    for m in msgs:
        by[m["role"]] += 1
        h = ref.hits(m["t"])
        if h:
            with_hit[m["role"]] += 1
            for _s, _e, rid in h:
                f = fam_of.get(rid, rid)
                hits[m["role"]][f] = hits[m["role"]].get(f, 0) + 1
        words = [x for x in h if x[2] != "MARKS"]
        marks = [x for x in h if x[2] == "MARKS"]
        masked.append({"role": m["role"], "t": ref.mask(m["t"]) if h else m["t"], "had_hit": bool(words),
                       "orig": ref.mask_spans(m["t"], marks) if (words and marks) else (m["t"] if words else None),
                       "mark_hit": ref.mask_spans(m["t"], words) if marks else None})
    say(f"  {len(msgs)} messages on {mode_desc} ({by['assistant']} replies, {by['user']} hers); "
        f"reference pass {time.time() - t0:.0f}s")
    say(f"  replies with a banned form: {with_hit['assistant']}; her messages with one: {with_hit['user']}")
    say(f"  banned-form occurrences by family: replies {hits['assistant']}; hers {hits['user']}")
    # character level: the masked text must pass the grammar's automaton (false blocks), the rest must not
    fb = sum(1 for m in masked if not A.accepts(m["t"]))
    miss = sum(1 for m in masked if m["had_hit"] and A.accepts(m["orig"]))
    res.add("her corpus, character level: 0 false blocks after masking the banned forms", fb == 0,
            f"{len(masked) - fb}/{len(masked)} messages accepted")
    res.add("her corpus, character level: every message with a banned word is refused", miss == 0,
            f"{sum(1 for m in masked if m['had_hit']) - miss}/{sum(1 for m in masked if m['had_hit'])} refused")
    watch = {}
    pats = [(name, re.compile(rx)) for name, rx in (bl.get("watch") or {}).items()]
    for name, _ in pats:
        watch[name] = {"assistant": 0, "user": 0}
    for m in msgs:                  # the text as written: an accented evasion (Angelâ) is not the allowed word
        for name, c in pats:
            watch[name][m["role"]] += len(c.findall(m["t"]))
    say("  watched words (not banned; counts only): " + "; ".join(f"{k}: replies {v['assistant']}, hers {v['user']}"
                                                                  for k, v in watch.items()))
    (priv / "corpus.json").write_text(json.dumps(masked, ensure_ascii=False), encoding="utf-8")
    res.rows.append({"check": "corpus facts", "pass": True, "detail": json.dumps(
        {"messages": by, "with_banned_form": with_hit, "occurrences": hits, "watch": watch})})


# ============================================================================================================
# tokenizer
# ============================================================================================================
def cmd_tokenizer(args):
    import urllib.request
    d = Path(args.tokenizer).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REV}/"
    for f in TOKENIZER_FILES:
        p = d / f
        if not p.exists() or p.stat().st_size == 0:
            say(f"downloading {base + f}")
            with urllib.request.urlopen(base + f, timeout=120) as r:
                p.write_bytes(r.read())
        say(f"  {f}: {p.stat().st_size:,} B")
    (d / "consolidated.safetensors").touch()   # empty marker: vLLM's auto mode then picks the Mistral tokenizer
    h = sha(d / "tekken.json")
    ok = h.startswith(TEKKEN_SHA_PREFIX)
    say(f"tekken.json sha256 {h} {'(the pinned tokenizer)' if ok else '(UNEXPECTED - not the pinned tokenizer)'}")
    return 0 if ok else 1


# ============================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, build=True):
        if build:
            p.add_argument("--build-dir", default=str(HOME / "build"))
        p.add_argument("--image", default=DEFAULT_IMAGE)
        p.add_argument("--tokenizer", default=str(HOME / "tokenizer"))
    p = sub.add_parser("build", help="banlist.yaml -> grammar + paste value")
    p.add_argument("--banlist", default=str(HERE / "banlist.yaml"))
    p.add_argument("--here", action="store_true", help="build with this Python even if its Unicode version differs")
    common(p)
    p = sub.add_parser("verify", help="the full verification suite (non-zero exit on any failure)")
    p.add_argument("--corpus", help="a webui.db (backup) to measure false blocks on; read-only, copied first")
    p.add_argument("--no-image", action="store_true", help="only the local checks (no tokenizer / docker)")
    p.add_argument("--fuzz", type=int, default=40000)
    p.add_argument("--walk-steps", type=int, default=120000)
    p.add_argument("--recent", type=int, default=200, help="with --corpus: only the most recently active chat's "
                   "last N messages (owner decision, rev 7 round 2; default 200)")
    p.add_argument("--full-corpus", action="store_true", help="with --corpus: scan every chat's whole current "
                   "branch instead (the old default)")
    common(p)
    p = sub.add_parser("tokenizer", help="download the pinned Tekken tokenizer files")
    common(p, build=False)
    for extra in ("e2e", "diff", "release"):
        pass
    import wb_cmds                                              # noqa: E402  (e2e, diff, release)
    wb_cmds.add_parsers(sub, common)
    args = ap.parse_args(argv)
    fn = {"build": cmd_build, "verify": cmd_verify, "tokenizer": cmd_tokenizer}.get(args.cmd) or wb_cmds.dispatch(args.cmd)
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
