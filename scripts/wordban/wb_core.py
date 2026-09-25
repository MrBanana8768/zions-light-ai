"""wordban core (pure Python, standard library + PyYAML).

  banlist.yaml --(Spec)--> character-class DFA --(emit)--> xgrammar EBNF (vLLM `structured_outputs`)
  EBNF --(Automaton)--> an automaton over code-point intervals, read back from the ARTIFACT itself, used by
                        every check (trap proof, test lists, corpus, the pod check)
  Reference           --> a second, independent implementation of the same banlist (backtracking over folded
                        text), used to cross-check the DFA builder and to mask real text

Nothing here needs vLLM, a tokenizer or a GPU. The token-level half of the verification (vLLM 0.19's own
xgrammar backend and the real Tekken tokenizer) lives in wb_image.py and runs inside the image."""
import array
import bisect
import collections
import hashlib
import itertools
import json
import os
import sqlite3
import sys
import unicodedata

DEAD = "DEAD"
FFFD_CP = 0xFFFD
# rev 7 hostile-review fix 1: NBSP (U+00A0) and narrow NBSP (U+202F) must behave EXACTLY like a plain space
# everywhere - so they are NOT part of the generic 'WS' pseudo-separator (tab, newline and the other exotic
# whitespace below); instead, wherever a separator set lists a literal plain space, NBSP_MIRROR is folded into
# that same character class automatically (see Spec._compile_classes.see_items and Reference.__init__/_item_ok).
NBSP_MIRROR = (" ", " ")
# rev 7 hostile-review fix 1 (continued) / owner decision A: line-break characters (LF, CR, VT, FF, NEL, LINE
# SEPARATOR, PARAGRAPH SEPARATOR) are excluded from the generic 'WS' pseudo-separator everywhere - a run of the
# generic separator sets (ALL, WF, PJOIN...) must never be able to sweep up TWO OR MORE line breaks (a blank
# line), which would silently defeat the yellow_light "blank line is BREAK, must pass" rule (owner
# decision A) the moment any family's separator set includes 'WS'. A family that needs to recognise a line break
# at all (yellow_light's "*-LINEBREAK" rules) does so with an explicit, dedicated, single literal
# '\n' step instead - never through a star-repeatable separator set. Tab and the other exotic horizontal
# whitespace stay in WS_EXTRA (decision A: tab is a JOINED separator for yellow_light; fix 1 only
# required removing WS - tab included - from the generic in-word SPLIT set for angel/gold/flame, done above).
LINEBREAKS = frozenset(chr(c) for c in (0x0A, 0x0B, 0x0C, 0x0D, 0x85, 0x2028, 0x2029))
# Whitespace characters other than the plain space, the two NBSP variants and line breaks (the 'WS' pseudo-sep)
WS_EXTRA = [chr(c) for c in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85, 0x1680, *range(0x2000, 0x200B), 0x2028,
                             0x2029, 0x205F, 0x3000) if chr(c) not in NBSP_MIRROR and chr(c) not in LINEBREAKS]
BIG = ["MARK", "LETTER_OTHER", "NONLETTER"]
MARK_CATS = ("Mn", "Mc", "Me")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_yaml(path):
    try:
        import yaml
    except ImportError:                                     # pragma: no cover
        sys.exit("wordban: PyYAML is required (pip install pyyaml; the image and the unit-test image have it)")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================================================
# folding
# ============================================================================================================
class Folder:
    def __init__(self, cfg):
        self.latin = [tuple(x) for x in cfg.get("latin_blocks", [])]
        self.greek = [tuple(x) for x in cfg.get("greek_blocks", [])]
        self.extra = dict(cfg.get("extra", {}))
        self.conf = {str(k): str(v) for k, v in (cfg.get("confusables") or {}).items()}
        self._cache = {}

    def confusable(self, ch):
        """the ASCII letter a look-alike from another script stands for (Cyrillic а, Greek Α ...), or None"""
        return self.conf.get(ch)

    def fold(self, ch):
        """the single letter `ch` counts as (ASCII Latin, or a base Greek letter), else None."""
        r = self._cache.get(ch, 0)
        if r != 0:
            return r
        r = None
        if ch.isascii():
            r = ch if ch.isalpha() else None
        elif ch in self.extra:
            r = self.extra[ch]
        else:
            cp = ord(ch)
            if any(a <= cp <= b for a, b in self.latin):
                d = "".join(x for x in unicodedata.normalize("NFKD", ch) if not unicodedata.category(x).startswith("M"))
                if len(d) == 1 and d.isascii() and d.isalpha():
                    r = d
            elif any(a <= cp <= b for a, b in self.greek):
                d = "".join(x for x in unicodedata.normalize("NFD", ch) if not unicodedata.category(x).startswith("M"))
                if len(d) == 1 and unicodedata.category(d).startswith("L"):
                    r = d
        self._cache[ch] = r
        return r


# ============================================================================================================
# banlist -> rules
# ============================================================================================================
class Pattern:
    """one compiled pattern: a list of steps; a step is ('c', items) or ('star', items). items are
    ('let', x) folded letter x | ('chr', c) literal non-letter | ('sep', name) | ('any',)"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def parse_pattern(s, seps):
    steps = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            j = s.index("}", i)
            body = s[i + 1:j]
            i = j + 1
            opt = body.endswith("?")
            if opt:
                body = body[:-1]
            if body == "L":
                items = [("any",)]
            elif body.startswith("SEP:"):
                name = body[4:]
                if name not in seps:
                    raise ValueError(f"pattern {s!r}: unknown separator set {name!r}")
                items = [("sep", name)]
            else:
                raise ValueError(f"pattern {s!r}: unknown {{{body}}}")
            kind = "c"
            if i < len(s) and s[i] == "*":
                kind = "star"
                i += 1
            elif i < len(s) and s[i] == "+":           # one or more
                i += 1
                steps.append(("c", items))
                kind = "star"
            steps.append(("opt" if opt else kind, items))
            continue
        if ch == "[":
            j = s.index("]", i)
            items = [("let", x.lower()) if x.isalpha() else ("chr", x) for x in s[i + 1:j]]
            i = j + 1
            steps.append(("c", items))
            continue
        if ch == "\\":
            ch = s[i + 1]
            i += 1
        steps.append(("c", [("let", ch.lower())] if ch.isalpha() else [("chr", ch)]))
        i += 1
    return steps


def expand_optional(steps):
    """a pattern with k optional steps -> 2^k patterns without them"""
    opts = [n for n, st in enumerate(steps) if st[0] == "opt"]
    out = []
    for keep in itertools.product([False, True], repeat=len(opts)):
        k = dict(zip(opts, keep))
        out.append([("c", st[1]) if st[0] == "opt" else st for n, st in enumerate(steps) if st[0] != "opt" or k[n]])
    return out


def _lets(w):
    return [("c", [("let", ch.lower())]) for ch in w]


def _sep_run(name, star=True):
    s = [("c", [("sep", name)])]
    if star:
        s.append(("star", [("sep", name)]))
    return s


def phrase_split(w1, w2, inner, outer, outer_star=True, outer_optional=False):
    """rev 7 (owner decision A): a TWO-word phrase (yellow light / white flame) where a split is also allowed
    INSIDE either word ("yel-low light", "yellow lig-ht", "yel low light" - a space inside "yellow" is fine to
    ban for THIS phrase specifically, unlike the generic angel/gold SPLIT set). One variant per split position
    in w1 (with the mandatory `outer` boundary run before w2), and one per split position in w2 (with the
    mandatory `outer` boundary run after w1); `inner` is the in-word split separator set, `outer` the
    between-words one. `outer_star=False` (rev 7 FINAL, item 4: the NL/newline outer variant) makes the outer
    boundary EXACTLY one separator, never a run - so a blank line (2+ newlines) never matches through it, the
    same reasoning as PJOIN never carrying a bare newline. `outer_optional=True` (item 4: "yel-lowlight", a
    split followed directly by NO joiner at all) also emits a second, glued (zero-outer-separator) variant per
    split position."""
    out = []
    for i in range(1, len(w1)):
        out.append(_lets(w1[:i]) + _sep_run(inner) + _lets(w1[i:]) + _sep_run(outer, outer_star) + _lets(w2))
        if outer_optional:
            out.append(_lets(w1[:i]) + _sep_run(inner) + _lets(w1[i:]) + _lets(w2))
    for j in range(1, len(w2)):
        out.append(_lets(w1) + _sep_run(outer, outer_star) + _lets(w2[:j]) + _sep_run(inner) + _lets(w2[j:]))
        if outer_optional:
            out.append(_lets(w1) + _lets(w2[:j]) + _sep_run(inner) + _lets(w2[j:]))
    return out


def spaced_word(w1, w2, which, spaced_set, boundary_sep):
    """rev 7 FINAL (item 4): a letter-spaced run WITHIN only ONE of the two words of a phrase ("y-e-l-l-o-w
    light", "yellow l-i-g-h-t"), the other word left intact, joined by `boundary_sep` (a JOIN, e.g. PJOIN)."""
    if which == "first":
        return [spaced(w1, spaced_set) + _sep_run(boundary_sep) + _lets(w2)]
    return [_lets(w1) + _sep_run(boundary_sep) + spaced(w2, spaced_set)]


def spaced(word, sepname):
    """a run of one or more separators between every two letters"""
    s = []
    for n, ch in enumerate(word):
        if n:
            s.append(("c", [("sep", sepname)]))
            s.append(("star", [("sep", sepname)]))
        s.append(("c", [("let", ch.lower())]))
    return s


class Spec:
    """the banlist, compiled: character classes + patterns + continuation tries + in-word escapes."""

    def __init__(self, bl, source_bytes=b""):
        self.bl = bl
        self.source_sha = sha256_bytes(source_bytes) if source_bytes else ""
        self.revision = bl.get("revision")
        fcfg = bl.get("folding", {})
        self.folder = Folder(fcfg)
        self.mode = fcfg.get("accented_continuation", "dead")
        if self.mode not in ("dead", "taint"):
            raise ValueError("folding.accented_continuation must be 'dead' (or 'taint', rev-4 behaviour, tests only)")
        self.fffd_default = fcfg.get("fffd", "one_run_per_word") == "one_run_per_word"
        self.mark_cap = fcfg.get("mark_cap")
        self.mark_cap_hebrew = fcfg.get("mark_cap_hebrew") if self.mark_cap else None
        self.hebrew_marks = [tuple(x) for x in fcfg.get("hebrew_marks", [])] if self.mark_cap_hebrew else []
        self.lookalikes = {str(x).lower(): [str(c) for c in v] for x, v in (fcfg.get("lookalikes") or {}).items()}
        self.forbidden = sorted(int(x) for x in (fcfg.get("forbidden_chars") or []))
        self.zwj_rule = bool(fcfg.get("zwj_not_between_letters"))
        self.marks_after_stem = fcfg.get("marks_after_stem", "allow")
        self.seps = {k: list(v) for k, v in (bl.get("separators") or {}).items()}
        self._compile_rules()
        self._compile_classes()
        self._resolve()
        self._check_funnel_prefix_lengths(bl)

    # ---------------------------------------------------------------- sanity checks
    @staticmethod
    def _check_funnel_prefix_lengths(bl, limit=20):
        """rev 7 FINAL (hostile-review fix 6): a documented funnel `after` prefix's letter count directly drives
        wb_image.py's case_variants() cost (2**letters, up to its cap) - a prefix over this limit either wastes
        that cost for nothing (case_variants' 2**20 cap already can't fully enumerate it) or, if raised further,
        gets expensive fast. Caught here at build time, not discovered later as a slow/incomplete funnel check."""
        for d in (bl.get("funnels") or {}).get("documented") or []:
            for pre in d.get("after") or []:
                n = sum(1 for c in pre if c.isalpha())
                if n > limit:
                    raise ValueError(f"funnels.documented: after prefix {pre!r} has {n} letters (limit {limit}) - "
                                     "case_variants() cannot fully enumerate its casings; shorten it or split it "
                                     "into shorter prefixes that reach the same state")

    # ---------------------------------------------------------------- rules
    def _variants(self, r, src):
        if r.get("phrase_split"):
            w1, w2 = r["phrase_split"]
            return phrase_split(w1, w2, r["inner_sep"], r["outer_sep"], outer_star=r["outer_sep"] != "NL",
                               outer_optional=bool(r.get("outer_optional")))
        if r.get("spaced_word"):
            w = r["spaced_word"]
            if src.startswith(w):
                w1, w2, which = w, src[len(w):], "first"
            elif src.endswith(w):
                w1, w2, which = src[:-len(w)], w, "second"
            else:
                raise ValueError(f"{r['id']}: spaced_word {w!r} is not a prefix or suffix of {src!r}")
            return spaced_word(w1, w2, which, r["spaced_set"], r["boundary_sep"])
        if r.get("spaced"):
            return [spaced(src, r["spaced"])]
        if r.get("split"):
            word = src
            out = []
            for i in range(1, len(word)):
                steps = [("c", [("let", ch.lower())]) for ch in word[:i]]
                steps += [("c", [("sep", r["split"])]), ("star", [("sep", r["split"])])]
                steps += [("c", [("let", ch.lower())]) for ch in word[i:]]
                out.append(steps)
            return out
        return expand_optional(parse_pattern(src, self.seps))

    def _compile_rules(self):
        self.pats = []            # Pattern objects (ban / stem / ctx)
        self.stems = []           # continuation specs, one per stem variant
        self.rules = []           # (family, rule dict) flat list, for reporting
        self.escapes = []         # in-word escape tries (word prefixes that exempt a rule)
        for fam, fdef in (self.bl.get("families") or {}).items():
            for r in fdef.get("rules", []):
                self.rules.append((fam, r))
                rid = r["id"]
                where = r.get("where", "word_start")
                if where not in ("word_start", "anywhere", "inside"):
                    raise ValueError(f"{rid}: where must be word_start, anywhere or inside")
                src = r.get("stem", r.get("match"))
                if src is None:
                    raise ValueError(f"{rid}: needs match: or stem:")
                variants = self._variants(r, src)
                esc_id = None
                if r.get("allow_in_words"):
                    ws = [w.lower() for w in r["allow_in_words"]]
                    for w in ws:
                        if not (w.isascii() and w.isalpha()):
                            raise ValueError(f"{rid}: allow_in_words must be plain ASCII letters: {w!r}")
                    esc_id = len(self.escapes)
                    self.escapes.append({"rule": rid, "trie": build_trie([], ws)})
                ctx = r.get("context")
                ctx_id = None
                if ctx:
                    ctx_id = len([p for p in self.pats if p.kind == "ctx"])
                    self.pats.append(Pattern(id=f"{rid}@ctx", family=fam, kind="ctx", where="word_start",
                                             steps=parse_pattern(ctx["after"], self.seps), wild=False, ctx_id=ctx_id,
                                             ctx_of=None, not_ctx_of=None, variant=None, stem_id=None, esc_id=None,
                                             plain=False))
                # rev 7: a NEGATIVE lookback (the opposite of context.after) - the rule does not fire when the word
                # is directly preceded by <not_after>. Built the same way as context.after (its own ctx-marker
                # pattern(s), checked one character at a time), but the consuming pattern SKIPS when a marker is
                # set instead of requiring one. Used for "G OLD" so "200 g old" (the unit gram) is not a split
                # evasion. Expanded (like any {X?} pattern) into one marker per variant, since a marker pattern
                # cannot itself end in a bare {X}* (nothing would ever complete it with zero repeats).
                not_after = r.get("not_after")
                not_ctx_ids = []
                if not_after:
                    for na_steps in expand_optional(parse_pattern(not_after, self.seps)):
                        nid = len([p for p in self.pats if p.kind == "ctx"])
                        self.pats.append(Pattern(id=f"{rid}@not-ctx{len(not_ctx_ids)}", family=fam, kind="ctx",
                                                 where="word_start", steps=na_steps, wild=False, ctx_id=nid,
                                                 ctx_of=None, not_ctx_of=None, variant=None, stem_id=None,
                                                 esc_id=None, plain=False))
                        not_ctx_ids.append(nid)
                not_ctx_of = frozenset(not_ctx_ids) if not_ctx_ids else None
                fffd = r.get("fffd", "default")
                for vi, steps in enumerate(variants):
                    if steps[0][0] == "star":
                        raise ValueError(f"{rid}: a pattern may not start with a *-step")
                    only_letters = all(it[0] in ("let", "any") for st in steps if st[0] == "c" for it in st[1])
                    if fffd == "transparent":
                        wild = "transparent"
                    elif fffd is True or (fffd == "default" and only_letters and not r.get("spaced")
                                          and not r.get("split")):
                        wild = "one" if self.fffd_default else False
                    else:
                        wild = False
                    name = rid if len(variants) == 1 else f"{rid}#{vi}"
                    base = dict(family=fam, where=where, steps=steps, wild=wild, esc_id=esc_id, ctx_id=None,
                                not_ctx_of=not_ctx_of, plain=False)
                    if "stem" in r:
                        for w in (r.get("allow_words") or []) + (r.get("allow_prefixes") or []) + (r.get("ban_words") or []):
                            if not (w.isascii() and w.isalpha()):
                                raise ValueError(f"{rid}: allowed continuations must be plain ASCII letters: {w!r}")
                        sid = len(self.stems)
                        self.stems.append(dict(alone=r.get("alone", "ban"), other=r.get("other", "ban"), rule=rid,
                                               family=fam, variant="normal",
                                               trie=build_trie(r.get("allow_words") or [], r.get("allow_prefixes") or [],
                                                              r.get("ban_words") or [])))
                        self.pats.append(Pattern(id=name, kind="stem", stem_id=sid, ctx_of=ctx_id,
                                                 variant="normal" if ctx else None, **base))
                    else:
                        self.pats.append(Pattern(id=name, kind="ban", stem_id=None, ctx_of=ctx_id,
                                                 variant="normal" if ctx else None, **base))
                    if ctx:
                        sid2 = len(self.stems)
                        self.stems.append(dict(alone="ban", other="ban", rule=rid, family=fam, variant="ctx",
                                               trie=build_trie(ctx.get("allow_words") or [], ctx.get("allow_prefixes") or [])))
                        b2 = dict(base, where="word_start", wild=False if ctx.get("plain") else wild,
                                  plain=bool(ctx.get("plain")), esc_id=None)
                        self.pats.append(Pattern(id=f"{name}@after-ctx", kind="stem", stem_id=sid2, ctx_of=ctx_id,
                                                 variant="ctx", **b2))
        for st in self.stems:
            check_trie(st)
        # one shared trie over every rule's in-word escapes: a word is tracked by ONE thread, not one per rule
        self.esc_trie = [{"ch": {}, "eids": set()}]
        for eid, e in enumerate(self.escapes):
            for w in Reference._words(e["trie"]):
                n = 0
                for ch in w:
                    nx = self.esc_trie[n]["ch"].get(ch)
                    if nx is None:
                        self.esc_trie.append({"ch": {}, "eids": set()})
                        nx = len(self.esc_trie) - 1
                        self.esc_trie[n]["ch"][ch] = nx
                    n = nx
                self.esc_trie[n]["eids"].add(eid)

    # ---------------------------------------------------------------- classes
    def _compile_classes(self):
        letters, chars, lits = set(), set(), set()
        sepmem = collections.defaultdict(set)
        uses_ws = False

        def see_items(items):
            nonlocal uses_ws
            for it in items:
                if it[0] == "let":
                    x = it[1]
                    letters.add(x if x.isascii() else (self.folder.fold(x) or x).lower())
                elif it[0] == "chr":
                    chars.add(it[1])
                    lits.add(it[1])
                elif it[0] == "sep":
                    for c in self.seps[it[1]]:
                        if c == "WS":
                            uses_ws = True
                        else:
                            chars.add(c)
                            sepmem[c].add(it[1])
                            if c == " ":                     # rev 7 fix 1: NBSP/narrow NBSP = a plain space,
                                for nb in NBSP_MIRROR:        # in every separator set that carries a plain space
                                    chars.add(nb)
                                    sepmem[nb].add(it[1])
        for p in self.pats:
            for st in p.steps:
                see_items(st[1])
        split = set()
        for st in self.stems:
            for node in st["trie"]:
                for ch in node["ch"]:
                    letters.add(ch)
                    split.add(ch)
        for p in self.pats:
            if p.plain:                        # a plain-letters-only stem: its accented forms must be told apart
                for st in p.steps:
                    split.update(it[1] for it in st[1] if it[0] == "let" and it[1].isascii())
        # look-alikes (other scripts, I/1/0) only stand in for the letters of the banned patterns themselves
        self.stem_letters = frozenset(letters)
        for e in self.escapes:
            for node in e["trie"]:
                letters.update(node["ch"])
        for x, lst in self.lookalikes.items():
            if x in self.stem_letters:
                chars.update(c for c in lst if not c.isalpha())
        self.pletters = frozenset(letters)
        # non-letter characters with the same role everywhere (same separator sets, same look-alike letter, not
        # used literally) share one class: fewer classes, a smaller grammar
        sig = {}
        for ch in sorted(chars):
            look = frozenset(x for x, lst in self.lookalikes.items() if ch in lst and x in self.stem_letters)
            key = (frozenset(sepmem[ch]), look, ch if ch in lits else None)
            sig.setdefault(key, []).append(ch)
        self.char_class = {}
        self.char_meta = {}
        for key, group in sig.items():
            name = group[0] if len(group) == 1 else "s:" + "".join(group)
            for ch in group:
                self.char_class[ch] = name
            self.char_meta[name] = {"letter": False, "ptype": "non", "key": None, "matches": key[1]}
        self.split = sorted(split)
        self.chars = sorted(chars)
        self.uses_ws = uses_ws
        self.meta = {}
        self.cps = collections.defaultdict(list)
        prev = "NONLETTER"
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            if unicodedata.category(chr(cp)) == "Cn" and cp not in self.forbidden:
                c = prev if prev in BIG else "NONLETTER"
            else:
                c = self.classify(cp)
            self.cps[c].append(cp)
            prev = c
        for c, m in (("MARK", "M"), ("HMARK", "M"), ("FFFD", "non"), ("INVIS", "non"), ("ZWJ", "zwj"),
                     ("NONLETTER", "non"), ("WS", "non"), ("LETTER_OTHER", "let"), ("LOWER_OTHER", "lo"),
                     ("UPPER_OTHER", "up")):
            if c in self.cps and c not in self.meta:
                self.meta[c] = {"letter": m in ("lo", "up", "let"), "ptype": m, "key": None, "matches": frozenset()}
        for name, m in self.char_meta.items():
            if name in self.cps:
                self.meta[name] = m
        order = {"FFFD": 0, "INVIS": 1, "ZWJ": 2, "WS": 3, "LOWER_OTHER": 4, "UPPER_OTHER": 5, "MARK": 6, "HMARK": 7,
                 "LETTER_OTHER": 8, "NONLETTER": 9}
        self.classes = sorted(self.cps, key=lambda c: (order.get(c, -1), c))
        self.cls_of = {}
        for c, l in self.cps.items():
            for cp in l:
                self.cls_of[cp] = c

    def classify(self, cp):
        ch = chr(cp)
        if cp == FFFD_CP:
            return "FFFD"
        if cp in self.forbidden:
            return "INVIS"
        if self.zwj_rule and cp == 0x200D:
            return "ZWJ"
        if ch in self.char_class and not ch.isalpha():
            return self.char_class[ch]
        if self.uses_ws and ch in WS_EXTRA:
            return "WS"
        cat = unicodedata.category(ch)
        if cat in MARK_CATS:
            if self.mark_cap_hebrew and cat == "Mn" and any(a <= cp <= b for a, b in self.hebrew_marks):
                return "HMARK"
            return "MARK"
        f = self.folder.fold(ch)
        conf = self.folder.confusable(ch)
        if conf and conf.lower() not in self.stem_letters:
            conf = None
        if not (cat.startswith("L") or (f is not None and f.isalpha()) or conf):
            return "NONLETTER"
        asc = f if (f is not None and f.isascii()) else conf
        matches = {(f or ch).lower()}
        if conf:
            matches.add(conf.lower())
        for x, lst in self.lookalikes.items():
            if ch in lst and x in self.stem_letters:
                matches.add(x)
        matches &= self.pletters
        case = ("lo" if asc.islower() else "up") if asc else "let"
        if not matches:
            return {"lo": "LOWER_OTHER", "up": "UPPER_OTHER"}.get(case, "LETTER_OTHER")
        acc = (not ch.isascii()) and asc is not None and bool(({asc.lower()} | matches) & set(self.split))
        if asc and matches == {asc.lower()}:
            name = asc + ("~" if acc else "")
        elif not asc and len(matches) == 1 and not next(iter(matches)).isascii():
            name = "g:" + next(iter(matches))
        else:
            name = f"m:{''.join(sorted(matches))}:{asc or '-'}:{case}{'~' if acc else ''}"
        if name not in self.meta:
            self.meta[name] = {"letter": True, "ptype": case, "key": (asc.lower(), acc) if asc else None,
                               "matches": frozenset(matches)}
        return name

    # ---------------------------------------------------------------- resolve items -> class sets
    def is_letter_cls(self, c):
        return self.meta[c]["letter"]

    def ptype(self, c):
        return self.meta[c]["ptype"]

    def L(self, x):
        if not x.isascii():
            x = (self.folder.fold(x) or x).lower()
        return frozenset(c for c in self.classes if x in self.meta.get(c, {}).get("matches", ()))

    def plain_L(self, x):
        return frozenset(c for c in self.classes if self.meta.get(c, {}).get("key") == (x, False)
                         and all(cp < 0x80 for cp in self.cps[c]))

    def _items_to_classes(self, items, plain=False):
        out = set()
        for it in items:
            if it[0] == "let":
                out |= (self.plain_L(it[1]) if (plain and it[1].isascii()) else self.L(it[1]))
            elif it[0] == "chr":
                out.add(self.char_class.get(it[1], it[1]))
            elif it[0] == "sep":
                for c in self.seps[it[1]]:
                    if c == "WS":
                        out.add("WS")
                        continue
                    out.add(self.char_class[c])
                    if c == " ":
                        # rev 7 fix 1: NBSP/narrow NBSP may land in a DIFFERENT character class than plain space
                        # (e.g. a literal " " elsewhere, such as context.after "los ", forces a separate class
                        # signature) even though see_items() above mirrored their separator-set membership -
                        # union the mirrors' actual class in too, so a {SEP:X} transition that accepts space
                        # always also accepts NBSP/narrow NBSP.
                        out.update(self.char_class[nb] for nb in NBSP_MIRROR)
            elif it[0] == "any":
                out |= self.any_letter
        return frozenset(out & set(self.classes))

    def _resolve(self):
        self.any_letter = frozenset(c for c in self.classes if self.meta[c]["letter"])
        for p in self.pats:
            p.csteps = [(k, self._items_to_classes(items)) for k, items in p.steps]
            p.psteps = [(k, self._items_to_classes(items, plain=True)) for k, items in p.steps] if p.plain else None
            for k, cl in p.csteps:
                if not cl:
                    raise ValueError(f"{p.id}: a step matches no character")
        self.ctx_first = {}
        for p in self.pats:
            if p.variant == "ctx":
                self.ctx_first.setdefault(p.ctx_of, set()).update((p.psteps or p.csteps)[0][1])
        # escape tries read letters by their match sets
        self.esc_letters = {c: self.meta[c]["matches"] | ({self.meta[c]["key"][0]} if self.meta[c]["key"] else set())
                            for c in self.classes}

    # ---------------------------------------------------------------- DFA
    def cont_key(self, c):
        return self.meta[c]["key"]

    def end_ok(self, st, node, taint):
        ok = (st["alone"] == "allow") if node == 0 else st["trie"][node]["end"]
        return ok and not taint

    def _put(self, new, pid, pos, flag):
        new.add(("P", pid, pos, flag))
        steps = self.pats[pid].csteps
        if pos < len(steps) and steps[pos][0] == "star":
            new.add(("P", pid, pos + 1, flag))

    def _complete(self, p, new, escaped):
        if p.kind == "ban":
            if p.esc_id is not None and p.esc_id in escaped:
                return None                                # the word is one of the allowed words
            return DEAD
        if p.kind == "stem":
            new.add(("C", p.stem_id, 0, False))
        elif p.kind == "ctx":
            new.add(("M", p.ctx_id))
        return None

    def _is_letter_step(self, st):
        return any(self.is_letter_cls(x) for x in st[1])

    def advance(self, threads, prev, c):
        meta = self.meta[c]
        is_letter = meta["letter"]
        if c == "INVIS":
            return DEAD                                    # soft hyphen, zero-width space/non-joiner, word joiner
        if c == "ZWJ" and prev in ("lo", "up", "let"):
            return DEAD                                    # zero-width joiner right after a letter
        if prev == "zwj" and is_letter:
            return DEAD                                    # ... or right before one
        up = meta["ptype"] == "up"
        ws = prev in ("non", "zwj") or (prev == "lo" and up)          # a word starts at this character
        new = set()
        # in-word escapes first: a word that begins with an allowed word is exempt from that rule
        escaped = set()
        if is_letter and self.escapes:
            letters = self.esc_letters[c]
            T = self.esc_trie
            nodes = [0] if ws else [th[1] for th in threads if th[0] == "E"]
            for th in threads:
                if th[0] == "X" and not ws:
                    new.add(th)
                    escaped.add(th[1])
            for node in nodes:
                for x in letters:
                    ch = T[node]["ch"].get(x)
                    if ch is not None:
                        for eid in T[ch]["eids"]:
                            new.add(("X", eid))
                            escaped.add(eid)
                        if T[ch]["ch"]:
                            new.add(("E", ch))
        for th in threads:
            if th[0] in ("M", "E", "X"):
                continue                                   # a context marker lives for one character
            if th[0] == "C":
                _, sid, node, taint = th
                st = self.stems[sid]
                trie = st["trie"]
                if c == "FFFD":
                    return DEAD                            # an unseen character right after a banned stem
                if is_letter:
                    k = self.cont_key(c)
                    if k and k[1]:
                        if self.mode == "dead":
                            return DEAD                    # accented letter after a banned stem: NOT ALLOWED
                        taint = True                       # rev-4 behaviour (unsafe; tests only)
                    child = trie[node]["ch"].get(k[0]) if k else None
                    if child is None:
                        if st["other"] == "ban":
                            return DEAD
                        continue                           # a different word: the thread ends
                    if trie[child]["esc"]:
                        continue
                    new.add(("C", sid, child, taint))
                    continue
                if not self.end_ok(st, node, taint):
                    return DEAD                            # the word ended here and it is a banned one
                continue
            _, pid, pos, flag = th
            p = self.pats[pid]
            steps = p.csteps
            if c == "FFFD":
                if p.plain:
                    return DEAD                            # an unseen character inside a plain-letters-only stem
                if p.wild == "transparent":
                    new.add(th)                            # an unseen point/accent: transparent
                elif flag in ("W", "T"):
                    new.add(th)                            # more bytes of the same unseen character
                elif flag == "" and p.wild == "one":
                    new.add(("P", pid, pos, "T"))          # an unseen accent inside the word
                    if self._is_letter_step(steps[pos]):
                        if pos + 1 == len(steps):
                            return DEAD                    # the last letter arrived unseen: conservative
                        self._put(new, pid, pos + 1, "W")  # ... or it WAS the next letter
                continue
            if flag in ("W", "T"):
                flag = "x"
            if steps[pos][0] == "star":
                if c in steps[pos][1]:
                    self._put(new, pid, pos, flag)
                continue
            if c in steps[pos][1]:
                if p.plain and c not in p.psteps[pos][1]:
                    return DEAD                            # the right letter, but not in plain letters
                if pos + 1 == len(steps):
                    if self._complete(p, new, escaped) is DEAD:
                        return DEAD
                    continue
                self._put(new, pid, pos + 1, flag)
        markers = {th[1] for th in threads if th[0] == "M"}
        for pid, p in enumerate(self.pats):
            if p.where == "word_start" and not ws:
                continue
            if p.where == "inside" and ws:
                continue
            if p.ctx_of is not None and ws:
                has = p.ctx_of in markers
                if p.variant == "ctx":
                    if not has:
                        continue
                elif has and c in self.ctx_first.get(p.ctx_of, ()):
                    continue                               # the context variant takes this one
            elif p.variant == "ctx":
                continue
            if p.not_ctx_of and (p.not_ctx_of & markers):
                continue                                   # not_after matched right here: this rule does not apply
            first = p.psteps[0][1] if p.plain else p.csteps[0][1]
            if c == "FFFD":
                if p.wild == "one" and self._is_letter_step(p.csteps[0]):
                    if len(p.csteps) == 1:
                        return DEAD
                    self._put(new, pid, 1, "W")
            elif c in first:
                if len(p.csteps) == 1:
                    if self._complete(p, new, escaped) is DEAD:
                        return DEAD
                else:
                    self._put(new, pid, 1, "")
        return frozenset(new)

    def accepting(self, threads):
        return all(self.end_ok(self.stems[th[1]], th[2], th[3]) for th in threads if th[0] == "C")

    def marks_ok(self, threads):
        """combining marks are refused right after a banned stem (inside its continuation) and inside a
        plain-letters-only stem"""
        if self.marks_after_stem != "refuse":
            return True
        return not any(th[0] == "C" or (th[0] == "P" and self.pats[th[1]].plain and th[2] >= 1) for th in threads)

    def build_dfa(self):
        marks = [c for c in self.classes if c in ("MARK", "HMARK")]
        ordinary = [c for c in self.classes if c not in marks]
        start = ("non", frozenset())
        states = {start: 0}
        order = [start]
        trans = {}
        i = 0
        while i < len(order):
            prev, threads = order[i]
            for c in ordinary:
                nt = self.advance(threads, prev, c)
                if nt == DEAD:
                    trans[(i, c)] = None
                    continue
                ns = (self.ptype(c), nt)
                if ns not in states:
                    states[ns] = len(order)
                    order.append(ns)
                trans[(i, c)] = states[ns]
            i += 1
        n = len(order)
        acc = [self.accepting(order[k][1]) for k in range(n)]
        mok = [self.marks_ok(order[k][1]) for k in range(n)]
        part0 = {}
        part = [part0.setdefault((acc[k], mok[k]), len(part0)) for k in range(n)]
        while True:
            sig = {}
            newp = []
            for k in range(n):
                key = (part[k],) + tuple(-1 if trans[(k, c)] is None else part[trans[(k, c)]] for c in ordinary)
                newp.append(sig.setdefault(key, len(sig)))
            done = len(set(newp)) == len(set(part))
            part = newp
            if done:
                break
        mt, macc, mmok = {}, {}, {}
        for k in range(n):
            b = part[k]
            macc[b] = acc[k]
            mmok[b] = mok[k]
            for c in ordinary:
                t = trans[(k, c)]
                mt[(b, c)] = None if t is None else part[t]
        ren = {part[0]: 0}
        q = collections.deque([part[0]])
        while q:
            b = q.popleft()
            for c in ordinary:
                t = mt[(b, c)]
                if t is not None and t not in ren:
                    ren[t] = len(ren)
                    q.append(t)
        self.n_raw = n
        self.dfa_n = len(ren)
        self.dfa_t = {(ren[b], c): (None if t is None else ren[t]) for (b, c), t in mt.items() if b in ren}
        self.dfa_acc = {ren[b]: a for b, a in macc.items() if b in ren}
        self.dfa_marks = {ren[b]: a for b, a in mmok.items() if b in ren}
        self.ordinary_classes = ordinary
        self.mark_classes = marks
        return self

    # ---------------------------------------------------------------- EBNF
    def emit(self):
        def atom_name(c):
            fixed = {"MARK": "zmk", "HMARK": "zhm", "LETTER_OTHER": "zlt", "NONLETTER": "znl", "LOWER_OTHER": "xlo",
                     "UPPER_OTHER": "xup", "WS": "zws", "INVIS": "zin", "ZWJ": "zzj", "FFFD": "zff"}
            if c in fixed:
                return fixed[c]
            if c.startswith("g:"):
                return "q" + "".join(f"{ord(x):x}" for x in c[2:])
            if c.startswith(("m:", "s:")):
                return "k" + hashlib.sha1(c.encode()).hexdigest()[:10]
            if len(c) == 2 and c[1] == "~":
                return ("vl" if c[0].islower() else "vu") + c[0].lower()
            if len(c) == 1 and c.isascii() and c.isalpha():
                return ("l" if c.islower() else "u") + c.lower()
            return None
        atom = {}
        for c in self.classes:
            if c in BIG or c in ("MARK", "HMARK") or len(self.cps[c]) > 1:
                nm = atom_name(c)
                assert nm, c
                atom[c] = nm
        used = set()
        groups_atoms = {}
        SMALL = 0                                  # measured: inlining atoms makes the grammar larger
        nranges = {c: len(ranges_of(self.cps[c])) for c in self.classes}

        def heads(cls_list):
            # One alternative per target: the classes' code points merged into a single char class (a rule that
            # is one char class is as fast as an inline one; a rule that is a UNION of rules is not - rev 4).
            if len(cls_list) == 1 and cls_list[0] in atom:
                used.add(cls_list[0])
                return [atom[cls_list[0]]]
            small = [c for c in cls_list if c not in atom or nranges[c] <= SMALL]
            big = [c for c in cls_list if c not in small]
            used.update(big)
            return ([charclass([cp for c in small for cp in self.cps[c]])] if small else []) + [atom[c] for c in big]
        cap = self.mark_cap
        full = lambda b: "root" if b == 0 else f"s{b}"
        lines = []
        for b in range(self.dfa_n):
            groups = collections.defaultdict(list)
            for c in self.ordinary_classes:
                t = self.dfa_t[(b, c)]
                if t is not None:
                    groups[t].append(c)
            alts = (['""'] if self.dfa_acc[b] else []) + [f"{h} {full(t)}" for t, cl in sorted(groups.items())
                                                           for h in heads(cl)]
            if not cap:
                if self.dfa_marks[b]:
                    alts += [f"{atom[m]} {full(b)}" for m in self.mark_classes]   # marks: transparent
                    used.update(self.mark_classes)
                lines.append(f"{full(b)} ::= " + " | ".join(alts))
                continue
            if not alts:
                raise ValueError(f"state {b} has no way to continue at all (a trap): refusing to emit a grammar "
                                 "that would force combining marks until max_tokens")
            if not self.dfa_marks[b]:
                lines.append(f"{full(b)} ::= " + " | ".join(alts))             # no marks here at all
                continue
            lines.append(f"c{b} ::= " + " | ".join(alts))
            lines += self._mark_wrappers(b, full(b), f"c{b}", atom)
            used.update(self.mark_classes)
        lines += [f"{nm} ::= {charclass(self.cps[c])}" for c, nm in atom.items() if c in used]
        order = {c: i for i, c in enumerate(self.classes)}
        for key, nm in groups_atoms.items():
            lines.append(f"{nm} ::= " + charclass([cp for c in sorted(key, key=order.get) for cp in self.cps[c]]))
        return "\n".join(lines) + "\n"

    def _mark_wrappers(self, b, N, C, atom):
        """at most `mark_cap` combining marks in a row; a run made only of Hebrew points may be up to
        `mark_cap_hebrew` long. g<i>: i marks so far, at least one non-Hebrew; h<i>: i Hebrew points so far."""
        K, KH = self.mark_cap, self.mark_cap_hebrew or 0
        zm, zh = atom.get("MARK"), atom.get("HMARK")
        name_g = lambda i: C if i >= K else (N if i == 0 else f"{N}g{i}")
        name_h = lambda i: C if i >= KH else f"{N}h{i}"
        out = []

        def rule(nm, i, heb):
            alts = [C]
            nxt = i + 1
            if heb:
                if zm and nxt <= K:
                    alts.append(f"{zm} {name_g(nxt)}")
                if zh and nxt <= KH:
                    alts.append(f"{zh} {name_h(nxt)}")
            else:
                if nxt <= K:
                    alts += [f"{zm} {name_g(nxt)}"] + ([f"{zh} {name_g(nxt)}"] if zh else [])
            out.append(f"{nm} ::= " + " | ".join(alts))
        # the state itself (0 marks)
        alts = [C]
        if K >= 1:
            alts.append(f"{zm} {name_g(1)}")
            if zh:
                alts.append(f"{zh} {name_h(1) if KH >= 1 else C}")
        out.append(f"{N} ::= " + " | ".join(alts))
        for i in range(1, K):
            rule(f"{N}g{i}", i, False)
        for i in range(1, KH):
            rule(f"{N}h{i}", i, True)
        return out

    # ---------------------------------------------------------------- DFA-level helpers
    def dfa_trap_states(self):
        """reachable states from which no ORDINARY character (not a mark, not U+FFFD) can ever reach an
        accepting state. (A state whose only exits are marks/U+FFFD, or letter fragments that all dead-end.)"""
        ords = [c for c in self.ordinary_classes if c != "FFFD"]
        good = {b for b in range(self.dfa_n) if self.dfa_acc[b]}
        changed = True
        while changed:
            changed = False
            for b in range(self.dfa_n):
                if b not in good and any(self.dfa_t[(b, c)] in good for c in ords):
                    good.add(b)
                    changed = True
        return [b for b in range(self.dfa_n) if b not in good]


def build_trie(words, prefixes, banned=()):
    """allowed whole words (end), allowed prefixes of other words (esc), and - for a stem whose other
    continuations are allowed - endings that are still banned (a path that is neither end nor esc)"""
    nodes = [{"ch": {}, "end": False, "esc": False}]

    def path(w):
        n = 0
        for ch in w.lower():
            nx = nodes[n]["ch"].get(ch)
            if nx is None:
                nodes.append({"ch": {}, "end": False, "esc": False})
                nx = len(nodes) - 1
                nodes[n]["ch"][ch] = nx
            n = nx
        return n
    for w in words:
        nodes[path(w)]["end"] = True
    for p in prefixes:
        nodes[path(p)]["esc"] = True
    for b in banned:
        path(b)
    return nodes


def check_trie(st):
    """a stem whose continuation can never complete would be a trap by construction: refuse it early."""
    trie = st["trie"]
    for n, node in enumerate(trie):
        live = node["end"] or node["esc"] or (st["other"] == "allow") or (n == 0 and st["alone"] == "allow")
        stack = list(node["ch"].values())
        seen = set()
        while not live and stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if trie[x]["end"] or trie[x]["esc"]:
                live = True
            stack += list(trie[x]["ch"].values())
        if not live:
            raise ValueError(f"stem rule {st['rule']}: a continuation state can never complete (would be a trap). "
                             "Use a plain ban rule instead, or add allow_words/allow_prefixes.")


# ---- EBNF char classes (xgrammar 0.2.7: positive ranges only, split at UTF-8 lead-byte boundaries) ----
def _u(cp):
    if 0x21 <= cp <= 0x7E and chr(cp) not in "]\\^-[\"'|":
        return chr(cp)
    return f"\\u{cp:04X}" if cp <= 0xFFFF else f"\\U{cp:08X}"


def ranges_of(cps):
    cps = sorted(cps)
    out = []
    for cp in cps:
        if out and cp == out[-1][1] + 1 and not (out[-1][1] == 0xD7FF):
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    res = []
    for a, b in out:
        for lo_, hi_, step in ((0, 0x7F, 0x80), (0x80, 0x7FF, 0x40), (0x800, 0xFFFF, 0x1000),
                               (0x10000, 0x10FFFF, 0x40000)):
            x = max(a, lo_)
            while x <= min(b, hi_):
                y = min(b, hi_, (x // step + 1) * step - 1 if lo_ else hi_)
                res.append((x, y))
                x = y + 1
    return res


def charclass(cps):
    return "[" + "".join(_u(a) if a == b else f"{_u(a)}-{_u(b)}" for a, b in ranges_of(cps)) + "]"


def build(banlist_path):
    raw = open(banlist_path, "rb").read()
    bl = load_yaml(banlist_path)
    spec = Spec(bl, raw).build_dfa()
    return spec, spec.emit()


# ============================================================================================================
# EBNF artifact -> automaton (independent of the generator: reads only the text of the grammar)
# ============================================================================================================
def _tokenize_ebnf_line(s):
    toks = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
        elif ch == "|":
            toks.append(("|",))
            i += 1
        elif s.startswith('""', i):
            toks.append(("eps",))
            i += 2
        elif ch == "[":
            i += 1
            items = []
            while s[i] != "]":
                a, i = _cc_char(s, i)
                if s[i] == "-" and s[i + 1] != "]":
                    b, i = _cc_char(s, i + 1)
                    items.append((a, b))
                else:
                    items.append((a, a))
            i += 1
            toks.append(("cc", items))
        else:
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            if j == i:
                raise ValueError(f"EBNF: unexpected {s[i:i + 20]!r}")
            toks.append(("id", s[i:j]))
            i = j
    return toks


def _cc_char(s, i):
    if s[i] == "\\":
        if s[i + 1] == "u":
            return int(s[i + 2:i + 6], 16), i + 6
        if s[i + 1] == "U":
            return int(s[i + 2:i + 10], 16), i + 10
        return ord(s[i + 1]), i + 2
    return ord(s[i]), i + 1


class Automaton:
    """a deterministic automaton over code-point intervals, read from a right-linear EBNF grammar of the
    shape this tool emits (and the earlier hand-built revisions): alternatives are "", <head> <rule>, or
    <rule> (an epsilon step); a head is a char class or a rule that is a single char class."""

    def __init__(self, ebnf):
        rules = {}
        for line in ebnf.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, body = line.partition("::=")
            toks = _tokenize_ebnf_line(body)
            alts, cur = [], []
            for t in toks:
                if t[0] == "|":
                    alts.append(cur)
                    cur = []
                else:
                    cur.append(t)
            alts.append(cur)
            rules[name.strip()] = alts
        self.rules = rules
        atoms = {}
        for nm, alts in rules.items():
            if len(alts) == 1 and len(alts[0]) == 1 and alts[0][0][0] == "cc":
                atoms[nm] = alts[0][0][1]
        self.atoms = atoms
        # elementary intervals
        pts = {0, 0x110000}
        ccs = []
        for nm, alts in rules.items():
            for a in alts:
                for t in a:
                    if t[0] == "cc":
                        ccs.append(t[1])
        for items in ccs:
            for a, b in items:
                pts.add(a)
                pts.add(b + 1)
        self.bounds = sorted(pts)
        nI = len(self.bounds) - 1
        self.n_intervals = nI

        def ivs(items):
            out = []
            for a, b in items:
                i = bisect.bisect_left(self.bounds, a)
                j = bisect.bisect_left(self.bounds, b + 1)
                out.extend(range(i, j))
            return out
        # per rule: eps targets, accepting, edges (interval list -> target)
        info = {}
        for nm, alts in rules.items():
            if nm in atoms:
                continue
            eps, acc, edges = [], False, []
            for a in alts:
                if len(a) == 1 and a[0][0] == "eps":
                    acc = True
                elif len(a) == 1 and a[0][0] == "id":
                    eps.append(a[0][1])
                elif len(a) == 2 and a[1][0] == "id" and a[0][0] in ("cc", "id"):
                    head = a[0][1] if a[0][0] == "cc" else atoms.get(a[0][1])
                    if head is None:
                        raise ValueError(f"EBNF: rule {nm}: head {a[0][1]} is not a char class")
                    edges.append((ivs(head), a[1][1]))
                else:
                    raise ValueError(f"EBNF: rule {nm}: unsupported alternative {a}")
            info[nm] = (eps, acc, edges)
        self.info = info

        def closure(names):
            seen = set()
            st = list(names)
            while st:
                x = st.pop()
                if x in seen:
                    continue
                seen.add(x)
                st += info[x][0]
            return frozenset(seen)
        start = closure(["root"])
        states = {start: 0}
        order = [start]
        T = []
        acc = []
        k = 0
        while k < len(order):
            S = order[k]
            row = [-1] * nI
            tgt = collections.defaultdict(set)
            for nm in S:
                for iv_list, t in info[nm][2]:
                    for iv in iv_list:
                        tgt[iv].add(t)
            cache = {}
            for iv, ts in tgt.items():
                key = frozenset(ts)
                if key not in cache:
                    ns = closure(ts)
                    if ns not in states:
                        states[ns] = len(order)
                        order.append(ns)
                    cache[key] = states[ns]
                row[iv] = cache[key]
            T.append(array.array('i', row))
            acc.append(any(info[nm][1] for nm in S))
            k += 1
        self.T = T
        self.acc = acc
        self.n = len(order)
        self.names = [sorted(S) for S in order]
        self._ivcache = {}

    def iv(self, ch):
        cp = ord(ch)
        r = self._ivcache.get(cp)
        if r is None:
            r = bisect.bisect_right(self.bounds, cp) - 1
            self._ivcache[cp] = r
        return r

    def run(self, text, s=0):
        """(final state, None) or (None, index where the text is refused)"""
        T = self.T
        for i, ch in enumerate(text):
            s = T[s][self.iv(ch)]
            if s < 0:
                return None, i
        return s, None

    def accepts(self, text):
        s, _ = self.run(text)
        return s is not None and self.acc[s]

    def interval_kinds(self):
        """for each interval: True if it contains at least one ORDINARY character (assigned, not a combining
        mark, not U+FFFD, not a surrogate / private-use / unassigned code point)"""
        out = []
        for i in range(self.n_intervals):
            a, b = self.bounds[i], self.bounds[i + 1]
            o = False
            for cp in range(a, b):
                if cp == FFFD_CP:
                    continue
                cat = unicodedata.category(chr(cp))
                if cat in MARK_CATS or cat in ("Cn", "Cs", "Co"):
                    continue
                o = True
                break
            out.append(o)
        return out

    def trap_states(self):
        """reachable states from which no path of ORDINARY characters reaches an accepting state."""
        ordv = self.interval_kinds()
        good = {s for s in range(self.n) if self.acc[s]}
        rev = collections.defaultdict(set)
        for s in range(self.n):
            for iv, t in enumerate(self.T[s]):
                if t >= 0 and ordv[iv]:
                    rev[t].add(s)
        st = list(good)
        while st:
            t = st.pop()
            for s in rev[t]:
                if s not in good:
                    good.add(s)
                    st.append(s)
        reach = {0}
        q = [0]
        prev = {0: None}
        while q:
            s = q.pop(0)
            for iv, t in enumerate(self.T[s]):
                if t >= 0 and t not in reach:
                    reach.add(t)
                    prev[t] = (s, iv)
                    q.append(t)
        traps = sorted(s for s in reach if s not in good)
        return traps, prev, ordv

    def example_prefix(self, s, prev):
        out = []
        while prev.get(s) is not None:
            s0, iv = prev[s]
            a = self.bounds[iv]
            # prefer a printable representative
            b = self.bounds[iv + 1]
            rep = a
            for cp in range(a, min(b, a + 256)):
                if unicodedata.category(chr(cp)) not in ("Cn", "Cs", "Co", "Cc"):
                    rep = cp
                    break
            out.append(chr(rep))
            s = s0
        return "".join(reversed(out))


# ============================================================================================================
# the reference definition (independent of the DFA): backtracking over folded text
# ============================================================================================================
class Reference:
    """the same banlist, implemented a second way: fold the text, then try every rule at every position with a
    small backtracking matcher. Shares nothing with the DFA builder except the parsed banlist and the folder."""

    def __init__(self, spec):
        self.s = spec
        self.split = set(spec.split)
        self.pl = spec.pletters
        self._cc = {}
        self.pats = [p for p in spec.pats if p.kind != "ctx"]
        self.ctx_pats = {p.ctx_id: p for p in spec.pats if p.kind == "ctx"}
        self.heb = spec.hebrew_marks
        self.forbidden = set(spec.forbidden)
        self.look = spec.lookalikes
        self.esc_words = {i: [w for w in self._words(e["trie"])] for i, e in enumerate(spec.escapes)}
        # index the patterns by what their first step can match, so most characters try none
        self.order = {id(p): i for i, p in enumerate(self.pats)}
        self.by_letter, self.by_raw, self.any_letter = {}, {}, []
        for p in self.pats:
            for it in p.steps[0][1]:
                if it[0] == "let":
                    x = it[1] if it[1].isascii() else (spec.folder.fold(it[1]) or it[1]).lower()
                    self.by_letter.setdefault(x, []).append(p)
                elif it[0] == "chr":
                    self.by_raw.setdefault(it[1], []).append(p)
                elif it[0] == "sep":
                    for c in spec.seps[it[1]]:
                        if c == "WS":
                            rs = WS_EXTRA
                        elif c == " ":
                            rs = [c, *NBSP_MIRROR]
                        else:
                            rs = [c]
                        for r in rs:
                            self.by_raw.setdefault(r, []).append(p)
                else:
                    self.any_letter.append(p)

    @staticmethod
    def _words(trie):
        out = []

        def walk(n, pre):
            if trie[n]["esc"]:
                out.append(pre)
            for ch, m in trie[n]["ch"].items():
                walk(m, pre + ch)
        walk(0, "")
        return out

    def _char(self, ch):
        r = self._cc.get(ch)
        if r is None:
            cat = unicodedata.category(ch)
            if cat in MARK_CATS:
                cp = ord(ch)
                r = ("M", cat == "Mn" and any(a <= cp <= b for a, b in self.heb))
            else:
                f = self.s.folder.fold(ch)
                conf = self.s.folder.confusable(ch)
                if conf and conf.lower() not in self.s.stem_letters:
                    conf = None                            # look-alikes only for the banned patterns' letters
                letter = (cat.startswith("L") or (f is not None and f.isalpha()) or bool(conf)) and ch != "�"
                asc = f if (f is not None and f.isascii()) else conf
                m = set()
                if letter:
                    m.add((f or ch).lower())
                    if conf:
                        m.add(conf.lower())
                m |= {x for x, lst in self.look.items() if ch in lst and x in self.s.stem_letters}
                m &= self.pl
                case = ("lo" if asc.islower() else "up") if (letter and asc) else ("let" if letter else "non")
                acc = letter and (not ch.isascii()) and asc is not None and bool(({asc.lower()} | m) & self.split)
                prim = asc.lower() if (letter and asc) else None
                r = (frozenset(m), letter, case, acc, prim)
            self._cc[ch] = r
        return r

    def units(self, text):
        idx, raw, mt, letter, case, acc, prim, mark_after = [], [], [], [], [], [], [], []
        runs = []
        i, n = 0, len(text)
        ch_ = self._char
        while i < n:
            r = ch_(text[i])
            if r[0] == "M":
                j = i
                heb = True
                while j < n:
                    rj = ch_(text[j])
                    if rj[0] != "M":
                        break
                    heb = heb and rj[1]
                    j += 1
                runs.append((i, j, j - i, heb))
                if mark_after:
                    mark_after[-1] = True
                i = j
                continue
            idx.append(i)
            raw.append(text[i])
            mt.append(r[0])
            letter.append(r[1])
            case.append(r[2])
            acc.append(r[3])
            prim.append(r[4])
            mark_after.append(False)
            i += 1
        return (idx, raw, mt, letter, case, acc, prim, mark_after), runs

    def _item_ok(self, it, U, k, plain=False):
        idx, raw, mt, letter, case, acc, prim, _ma = U
        if it[0] == "let":
            x = it[1] if it[1].isascii() else (self.s.folder.fold(it[1]) or it[1]).lower()
            if plain and x.isascii():
                return raw[k].isascii() and prim[k] == x
            return x in mt[k]
        if it[0] == "chr":
            return raw[k] == it[1]
        if it[0] == "sep":
            for c in self.s.seps[it[1]]:
                if c == "WS":
                    if raw[k] in WS_EXTRA:
                        return True
                elif c == " ":
                    if raw[k] == c or raw[k] in NBSP_MIRROR:
                        return True
                elif raw[k] == c:
                    return True
            return False
        return letter[k]

    def _match(self, steps, U, k, pos, ends, plain=False, bad=None):
        n = len(U[0])
        if pos == len(steps):
            ends.add(k)
            return
        kind, items = steps[pos]
        if kind == "star":
            self._match(steps, U, k, pos + 1, ends, plain, bad)
            j = k
            while j < n and any(self._item_ok(it, U, j) for it in items):
                j += 1
                self._match(steps, U, j, pos + 1, ends, plain, bad)
            return
        if k < n and any(self._item_ok(it, U, k, plain) for it in items):
            self._match(steps, U, k + 1, pos + 1, ends, plain, bad)
        elif plain and bad is not None and k < n and (any(self._item_ok(it, U, k) for it in items)
                                                      or U[1][k] == "�"):
            bad.add(k)                  # the right letter in a non-plain form (or unseen) inside a plain stem

    def _plain_stem(self, p, U, k):
        """a plain-letters-only stem (the context variant): (index where it is refused, None) or (None, end)"""
        n = len(U[0])
        marks = self.s.marks_after_stem == "refuse"
        for i, (kind, items) in enumerate(p.steps):
            j = k + i
            if i >= 1 and marks and U[7][j - 1]:
                return j, None                              # a combining mark inside the stem
            if j >= n:
                return None, None
            if any(self._item_ok(it, U, j, True) for it in items):
                continue
            if any(self._item_ok(it, U, j) for it in items) or U[1][j] == "�":
                return j + 1, None                          # the right letter, but not plain
            return None, None
        return None, k + len(p.steps)

    def _word_start(self, U, k):
        if k == 0:
            return True
        letter, case = U[3], U[4]
        return (not letter[k - 1]) or (case[k - 1] == "lo" and case[k] == "up")

    def _word_begin(self, U, k):
        while k > 0 and not self._word_start(U, k):
            k -= 1
        return k

    def _cont_banned(self, st, U, k):
        idx, raw, mt, letter, case, acc, prim, mark_after = U
        trie = st["trie"]
        node = 0
        n = len(idx)
        refuse_marks = self.s.marks_after_stem == "refuse"
        if refuse_marks and k > 0 and mark_after[k - 1]:
            return True                                    # a combining mark right after the stem
        while True:
            if k >= n or not letter[k]:
                ok = (st["alone"] == "allow") if node == 0 else trie[node]["end"]
                return not ok or (k < n and raw[k] == "�")
            if acc[k]:
                return True
            child = trie[node]["ch"].get(prim[k]) if prim[k] else None
            if child is None:
                return st["other"] == "ban"
            if trie[child]["esc"]:
                return False
            if refuse_marks and mark_after[k]:
                return True
            node = child
            k += 1

    def _escaped(self, p, U, k, e):
        if p.esc_id is None:
            return False
        s = self._word_begin(U, k)
        letters = U[2]
        for w in self.esc_words[p.esc_id]:
            if s + len(w) <= e and all(s + i < len(letters) and U[3][s + i]
                                       and w[i] in (letters[s + i] | ({U[6][s + i]} if U[6][s + i] else set()))
                                       for i in range(len(w)))                     and not any(self._word_start(U, j) for j in range(s + 1, e)):
                return True
        return False

    def _ctx_before(self, cp, U, k):
        for s in range(max(0, k - 16), k):
            if not self._word_start(U, s):
                continue
            ends = set()
            self._match(cp.steps, U, s, 0, ends)
            if k in ends:
                return True
        return False

    def hits(self, text):
        """[(orig start, orig end, rule id)]: banned forms, combining-mark-cap violations ('MARKS'), forbidden
        invisible characters ('INVISIBLE') and zero-width joiners touching a letter ('ZWJ')."""
        U, runs = self.units(text)
        idx, raw, mt, letter, case, acc, prim, mark_after = U
        n = len(idx)
        out = []
        for k in range(n):
            r = raw[k]
            if ord(r) in self.forbidden:
                out.append((idx[k], idx[k] + 1, "INVISIBLE"))
                continue
            if r == "‍" and self.s.zwj_rule and ((k > 0 and letter[k - 1]) or (k + 1 < n and letter[k + 1])):
                out.append((idx[k], idx[k] + 1, "ZWJ"))
                continue
            cand = []
            for x in mt[k]:
                cand += self.by_letter.get(x, [])
            if letter[k]:
                cand += self.any_letter
            cand += self.by_raw.get(r, [])
            if not cand:
                continue
            ws = self._word_start(U, k)
            for p in sorted({id(q): q for q in cand}.values(), key=lambda q: self.order[id(q)]):
                if p.where == "word_start" and not ws:
                    continue
                if p.where == "inside" and ws:
                    continue
                if p.ctx_of is not None and ws:
                    has = self._ctx_before(self.ctx_pats[p.ctx_of], U, k)
                    first_plain = any(self._item_ok(it, U, k, True) for q in self.pats
                                      if q.variant == "ctx" and q.ctx_of == p.ctx_of for it in q.steps[0][1])
                    if p.variant == "ctx" and not (has and first_plain):
                        continue
                    if p.variant == "normal" and has and first_plain:
                        continue
                elif p.variant == "ctx":
                    continue
                if p.not_ctx_of and any(self._ctx_before(self.ctx_pats[i], U, k) for i in p.not_ctx_of):
                    continue                               # not_after matched right here: this rule does not apply
                ends = set()
                hit = None
                if p.plain:
                    hit, end = self._plain_stem(p, U, k)
                    if end is not None:
                        ends.add(end)
                else:
                    self._match(p.steps, U, k, 0, ends)
                for e in sorted(ends):
                    if p.plain and self.s.marks_after_stem == "refuse" and any(mark_after[j] for j in range(k, e - 1)):
                        hit = e
                        break
                    if p.kind == "ban":
                        if not self._escaped(p, U, k, e):
                            hit = e
                            break
                    elif self._cont_banned(self.s.stems[p.stem_id], U, e):
                        hit = e
                        break
                if hit is not None:
                    g = max(hit, k + 1)
                    while g < n and letter[g]:
                        g += 1
                    out.append((idx[k], idx[min(g, n) - 1] + 1, p.id.split("#")[0].split("@")[0]))
        cap, caph = self.s.mark_cap, self.s.mark_cap_hebrew
        if cap:
            for a, b, m, heb in runs:
                if m > (caph if (heb and caph) else cap):
                    out.append((a, b, "MARKS"))
        return sorted(out)

    def blocked(self, text):
        return bool(self.hits(text))

    @staticmethod
    def _wordchar(ch):
        return ch.isalnum() or unicodedata.category(ch) in MARK_CATS or ch in "_'’-" or ch in "­​‌‍⁠﻿"

    def mask_spans(self, text, hits, repl=" Qzx "):
        """mask exactly these hits (and the rest of their words)"""
        spans = []
        for s, e, _r in sorted(hits):
            while s > 0 and self._wordchar(text[s - 1]):
                s -= 1
            while e < len(text) and self._wordchar(text[e]):
                e += 1
            if spans and s <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        for s, e in reversed(spans):
            text = text[:s] + repl + text[e:]
        return text

    def mask(self, text, repl=" Qzx "):
        """replace every word that holds a banned form with ' Qzx ' until none is left"""
        for _ in range(100):
            h = self.hits(text)
            if not h:
                return text
            text = self.mask_spans(text, h, repl)
        return text


# ============================================================================================================
# her corpus: webui.db (READ-ONLY copy) -> messages, walking each chat's current branch table-first
# ============================================================================================================
def _decode(v):
    if isinstance(v, str):
        try:
            j = json.loads(v)
        except Exception:
            return v
        return j
    return v


def _text_of(content, output):
    """the text the model actually produced: the `output` items when present (OpenWebUI 0.11
    output_text parts), else `content`."""
    o = _decode(output)
    if isinstance(o, list):
        parts = []
        for it in o:
            if isinstance(it, dict):
                for cc in it.get("content") or []:
                    if isinstance(cc, dict) and cc.get("type") in ("output_text", "text") and isinstance(cc.get("text"), str):
                        parts.append(cc["text"])
        if parts:
            return "".join(parts)
    c = _decode(content)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    if isinstance(c, dict):
        return c.get("text") or c.get("content") or ""
    return ""


def _norm_ts(t):
    try:
        t = int(t or 0)
    except Exception:
        return 0
    if t > 10 ** 14:
        return t // 10 ** 6
    if t > 10 ** 11:
        return t // 1000
    return t


def _chat_messages(cid, chat, cur_col, cols, con, scope="branch"):
    """[{role, ts, t, branch}] for ONE chat row: table rows first, JSON history fills gaps; the pointer is
    chat.current_message_id, else history.currentId. scope='all' adds every other stored message too
    (regenerations, other branches). Factored out of load_corpus so recent_messages() can reuse it per-chat."""
    try:
        j = json.loads(chat) if isinstance(chat, str) else (chat or {})
    except Exception:
        j = {}
    hist = ((j.get("history") or {}).get("messages") or {}) if isinstance(j, dict) else {}
    hist_cur = (j.get("history") or {}).get("currentId") if isinstance(j, dict) else None
    table = {}
    if cols:
        extra = "output" if "output" in cols else "null"
        prefix = cid + "-"
        for mid, pid, role, content, output, created in con.execute(
                f"select id, parent_id, role, content, {extra}, created_at from chat_message where chat_id=?", (cid,)):
            key = mid[len(prefix):] if mid.startswith(prefix) else mid
            table[key] = {"parentId": pid, "role": role, "t": _text_of(content, output), "ts": _norm_ts(created)}
    merged = {}
    for mid, m in hist.items():
        if not isinstance(m, dict):
            continue
        merged[mid] = {"parentId": m.get("parentId"), "role": m.get("role"),
                       "t": _text_of(m.get("content"), m.get("output")), "ts": _norm_ts(m.get("timestamp"))}
    for mid, r in table.items():
        merged[mid] = r                                  # table first
    ptr = cur_col if cur_col in merged else (hist_cur if hist_cur in merged else None)
    branch = []
    n = ptr
    seen = set()
    while n in merged and n not in seen:
        seen.add(n)
        branch.append(n)
        n = merged[n].get("parentId")
    ids = list(reversed(branch))
    if scope == "all":
        ids += sorted(set(merged) - set(branch))
    out = []
    for mid in ids:
        m = merged[mid]
        if m.get("role") in ("user", "assistant") and isinstance(m.get("t"), str) and m["t"].strip():
            out.append({"role": m["role"], "ts": m.get("ts") or 0, "t": m["t"], "branch": mid in seen})
    return out


def _open_chats(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(chat_message)")} if con.execute(
        "select 1 from sqlite_master where type='table' and name='chat_message'").fetchone() else set()
    chat_cols = {r[1] for r in con.execute("PRAGMA table_info(chat)")}
    sel = "id, chat, " + ("current_message_id" if "current_message_id" in chat_cols else "null")
    return con, cols, con.execute(f"select {sel} from chat").fetchall()


def _open_chats_with_title(db_path):
    """like _open_chats, but each row also carries the chat's title (for _is_test_chat) - used only by
    recent_messages, which needs it; load_corpus doesn't, so it keeps the simpler _open_chats."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(chat_message)")} if con.execute(
        "select 1 from sqlite_master where type='table' and name='chat_message'").fetchone() else set()
    chat_cols = {r[1] for r in con.execute("PRAGMA table_info(chat)")}
    sel = ("id, chat, " + ("current_message_id" if "current_message_id" in chat_cols else "null") + ", "
           + ("title" if "title" in chat_cols else "null"))
    return con, chat_cols, cols, con.execute(f"select {sel} from chat").fetchall()


def load_corpus(db_path, scope="branch"):
    """[{role, ts, t}] for every user/assistant message on EVERY chat's current branch (table rows first, JSON
    history fills gaps). scope='all' adds every other stored message too (regenerations, other branches)."""
    con, cols, rows = _open_chats(db_path)
    out = []
    for cid, chat, cur_col in rows:
        out += _chat_messages(cid, chat, cur_col, cols, con, scope)
    con.close()
    return out


# the decode-step sample floor for wb_image.py's "her corpus: sampled decode steps all have a normal exit"
# check - kept here (pure Python) so it and corpus_sample_rate() are unit-testable without the image's
# vllm/xgrammar/numpy dependencies.
CORPUS_SAMPLE_FLOOR = 100000


def corpus_sample_rate(mode, ntok, corpus_sample, floor=CORPUS_SAMPLE_FLOOR):
    """rev 7 round 2 (coordinator decision): the decode-step sample floor stays exactly as strong as ever - in
    'recent' mode (a much smaller corpus than the old full-corpus scans this floor was calibrated against), the
    SAMPLING RATE is raised instead of lowering the floor, so the sample count still reaches it: rate =
    min(1.0, (floor * 1.2) / total_tokens), capped at 1.0 for every prefix's sampling decision. 'full' mode is
    unchanged: `corpus_sample` (the --corpus-sample CLI value, default 0.03) as given, not recomputed."""
    if mode != "recent" or not ntok:
        return corpus_sample
    return min(1.0, (floor * 1.2) / ntok)


# rev 7 FINAL (hostile-review fix 6): chat ids/titles that mark a chat as a TEST fixture, not a real
# conversation - the e2e/pod-check tooling's own test conversation ids match this (e2e uses "itest-wordban-...",
# the pod check's control uses "itest-wordban-check"), so "the most recent chat" never accidentally picks one of
# THIS TOOL'S OWN test probes instead of a real conversation.
_TEST_CHAT_MARKERS = ("selftest", "itest")
_RECENT_ACTIVE_WINDOW_S = 24 * 3600


def _is_test_chat(cid, title):
    hay = f"{cid or ''} {title or ''}".lower()
    return any(m in hay for m in _TEST_CHAT_MARKERS)


def recent_messages(db_path, n=200):
    """(messages, chat_id, chat_id_sha256, newest_ts) for the OWNER'S "just the most recent chat" mode (rev 7):
    walk EVERY chat's CURRENT branch table-first (exactly like load_corpus - not its whole history, just the
    live branch, since that is what she is actually seeing), skip chats that look like this tool's own test
    fixtures (id or title containing "selftest"/"itest" - see _TEST_CHAT_MARKERS), then, THE RULE (hostile-
    review fix 6): among the chats active in the last 24h (their current branch's newest message within 24h of
    the newest message found across all of them), prefer the one with the MOST messages on its current branch -
    not simply the one with the single newest timestamp, which a lone stray ping could otherwise win. Falls back
    to the newest chat by timestamp alone if none are within the 24h window. Returns that chat's LAST n
    messages, in chronological order, plus the newest message timestamp found (for the corpus-age check).
    Only the chat id's hash is ever put in a stamp/report (never the id or her data)."""
    con, chat_cols, cols, rows = _open_chats_with_title(db_path)
    candidates = []          # (cid, msgs, newest_ts)
    global_newest = -1
    for cid, chat, cur_col, title in rows:
        msgs = _chat_messages(cid, chat, cur_col, cols, con, scope="branch")
        if not msgs or _is_test_chat(cid, title):
            continue
        newest = max(m["ts"] for m in msgs)
        candidates.append((cid, msgs, newest))
        global_newest = max(global_newest, newest)
    con.close()
    if not candidates:
        return [], None, None, None
    active = [c for c in candidates if c[2] >= global_newest - _RECENT_ACTIVE_WINDOW_S]
    pool = active or candidates
    best_cid, best_msgs, best_ts = max(pool, key=lambda c: len(c[1]))
    chat_hash = sha256_bytes(best_cid.encode())
    return best_msgs[-n:] if n else best_msgs, best_cid, chat_hash, best_ts


# ============================================================================================================
# test lists
# ============================================================================================================
def derived_must_block(spec):
    """one canonical instance per rule (plus Title and UPPER case), so a word added to the banlist is
    tested without anyone having to remember to list it."""
    out = []
    for fam, r in spec.rules:
        src = r.get("stem", r.get("match"))
        if r.get("spaced"):
            sep = [c for c in spec.seps[r["spaced"]] if c != "WS"][0]
            base = sep.join(src)
        elif r.get("split"):
            sep = [c for c in spec.seps[r["split"]] if c != "WS"][0]
            base = src[:1] + sep + src[1:]
        else:
            steps = expand_optional(parse_pattern(src, spec.seps))[0]
            s = []
            for kind, items in steps:
                if kind == "star":
                    continue
                it = items[0]
                if it[0] in ("let", "chr"):
                    s.append(it[1])
                elif it[0] == "sep":
                    s.append([c for c in spec.seps[it[1]] if c != "WS"][0])
                else:
                    s.append("x")
            base = "".join(s)
        if "stem" in r and r.get("alone", "ban") == "allow":
            continue                          # the stem alone is allowed; its banned forms are listed by hand
        if r.get("where") == "inside":
            base = "the" + base               # the banned letters glued to the end of another word
        forms = [base, base[:1].upper() + base[1:], base.upper()]
        for f in forms:
            out.append((r["id"], f))
    return out


def accent_cases(spec, words):
    """every folded variant of every letter of every listed banned word (one substitution at a time)"""
    by_letter = collections.defaultdict(list)
    for cp in range(0x80, 0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        f = spec.folder.fold(chr(cp))
        if f is not None and f.isascii():
            by_letter[f.lower()].append(chr(cp))
    out = []
    for w in words:
        for i, ch in enumerate(w):
            if not ch.isalpha():
                continue
            for v in by_letter.get(ch.lower(), []):
                out.append(w[:i] + v + w[i + 1:])
    return out
