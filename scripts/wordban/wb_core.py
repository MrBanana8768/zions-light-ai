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
import sqlite3
import sys
import unicodedata

DEAD = "DEAD"
FFFD_CP = 0xFFFD
# Whitespace characters other than the plain space (the 'WS' pseudo-separator)
WS_EXTRA = [chr(c) for c in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028,
                             0x2029, 0x202F, 0x205F, 0x3000)]
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
        self._cache = {}

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


def spaced(word, sepname):
    s = []
    for n, ch in enumerate(word):
        if n:
            s.append(("c", [("sep", sepname)]))
        s.append(("c", [("let", ch.lower())]))
    return s


class Spec:
    """the banlist, compiled: character classes + patterns + continuation tries."""

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
        self.seps = {k: list(v) for k, v in (bl.get("separators") or {}).items()}
        self._compile_rules()
        self._compile_classes()
        self._resolve()

    # ---------------------------------------------------------------- rules
    def _compile_rules(self):
        self.pats = []            # Pattern objects (ban / stem / ctx)
        self.stems = []           # continuation specs, one per stem variant
        self.rules = []           # (family, rule dict) flat list, for reporting
        for fam, fdef in (self.bl.get("families") or {}).items():
            for r in fdef.get("rules", []):
                self.rules.append((fam, r))
                rid = r["id"]
                where = r.get("where", "word_start")
                if where not in ("word_start", "anywhere"):
                    raise ValueError(f"{rid}: where must be word_start or anywhere")
                wi = where == "word_start"
                src = r.get("stem", r.get("match"))
                if src is None:
                    raise ValueError(f"{rid}: needs match: or stem:")
                if r.get("spaced"):
                    steps0 = spaced(src, r["spaced"])
                else:
                    steps0 = parse_pattern(src, self.seps)
                variants = expand_optional(steps0)
                for vi, steps in enumerate(variants):
                    if steps[0][0] == "star":
                        raise ValueError(f"{rid}: a pattern may not start with a *-step")
                    only_letters = all(it[0] in ("let", "any") for st in steps if st[0] == "c" for it in st[1])
                    wild = r.get("fffd", self.fffd_default and only_letters and not r.get("spaced"))
                    if wild and not self.fffd_default:
                        wild = False
                    name = rid if len(variants) == 1 else f"{rid}#{vi}"
                    if "stem" in r:
                        for w in (r.get("allow_words") or []) + (r.get("allow_prefixes") or []):
                            if not (w.isascii() and w.isalpha()):
                                raise ValueError(f"{rid}: allowed continuations must be plain ASCII letters: {w!r}")
                        ctx = r.get("context")
                        base = dict(alone=r.get("alone", "ban"), other=r.get("other", "ban"), rule=rid, family=fam)
                        sid = len(self.stems)
                        self.stems.append(dict(base, trie=build_trie(r.get("allow_words") or [],
                                                                     r.get("allow_prefixes") or []), variant="normal"))
                        ctx_id = None
                        if ctx:
                            ctx_id = len([p for p in self.pats if p.kind == "ctx"])
                            csteps = parse_pattern(ctx["after"], self.seps)
                            self.pats.append(Pattern(id=f"{name}@ctx", family=fam, kind="ctx", word_init=True,
                                                     steps=csteps, wild=False, ctx_id=ctx_id, ctx_of=None,
                                                     variant=None, stem_id=None))
                            sid2 = len(self.stems)
                            self.stems.append(dict(base, trie=build_trie(ctx.get("allow_words") or [],
                                                                         ctx.get("allow_prefixes") or r.get("allow_prefixes") or []),
                                                   variant="ctx"))
                            self.pats.append(Pattern(id=f"{name}@after-ctx", family=fam, kind="stem", word_init=wi,
                                                     steps=steps, wild=wild, stem_id=sid2, ctx_of=ctx_id,
                                                     variant="ctx", ctx_id=None))
                        self.pats.append(Pattern(id=name, family=fam, kind="stem", word_init=wi, steps=steps,
                                                 wild=wild, stem_id=sid, ctx_of=ctx_id,
                                                 variant="normal" if ctx else None, ctx_id=None))
                    else:
                        self.pats.append(Pattern(id=name, family=fam, kind="ban", word_init=wi, steps=steps,
                                                 wild=wild, stem_id=None, ctx_of=None, variant=None, ctx_id=None))
        for st in self.stems:
            check_trie(st)

    # ---------------------------------------------------------------- classes
    def _compile_classes(self):
        letters_ascii, letters_other, chars = set(), set(), set()
        uses_ws = False

        def see_items(items):
            nonlocal uses_ws
            for it in items:
                if it[0] == "let":
                    (letters_ascii if it[1].isascii() else letters_other).add(it[1])
                elif it[0] == "chr":
                    chars.add(it[1])
                elif it[0] == "sep":
                    for c in self.seps[it[1]]:
                        if c == "WS":
                            uses_ws = True
                        else:
                            chars.add(c)
        for p in self.pats:
            for st in p.steps:
                see_items(st[1])
        split = set()
        for st in self.stems:
            for node in st["trie"]:
                for ch in node["ch"]:
                    letters_ascii.add(ch)
                    split.add(ch)
        # non-ASCII pattern letters are matched by their folded, lower-case form
        lo = set()
        for x in letters_other:
            f = self.folder.fold(x) or x
            lo.add(f.lower())
        self.letters_ascii = sorted(letters_ascii)
        self.letters_other = sorted(lo)
        self.split = sorted(split)
        self.chars = sorted(chars)
        self.uses_ws = uses_ws
        specific = sorted(set([c for ch in self.letters_ascii for c in (ch, ch.upper())] + self.chars))
        variants = sorted(c + "~" for ch in self.split for c in (ch, ch.upper()))
        other = [f"g:{x}" for x in self.letters_other]
        marks = ["MARK"] + (["HMARK"] if self.mark_cap_hebrew else [])
        self.classes = specific + variants + other + (["WS"] if uses_ws else []) + \
            ["FFFD", "LOWER_OTHER", "UPPER_OTHER"] + marks + ["LETTER_OTHER", "NONLETTER"]
        self.cps = collections.defaultdict(list)
        prev = "NONLETTER"
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            if unicodedata.category(chr(cp)) == "Cn":
                c = prev if prev in BIG else "NONLETTER"
            else:
                c = self.classify(cp)
            self.cps[c].append(cp)
            prev = c
        missing = set(self.cps) - set(self.classes)
        assert not missing, missing
        self.classes = [c for c in self.classes if self.cps.get(c)]
        self.cls_of = {}
        for c, l in self.cps.items():
            for cp in l:
                self.cls_of[cp] = c

    def classify(self, cp):
        ch = chr(cp)
        if cp == FFFD_CP:
            return "FFFD"
        if ch in self.chars and not ch.isalpha():
            return ch
        if self.uses_ws and ch in WS_EXTRA:
            return "WS"
        cat = unicodedata.category(ch)
        if cat in MARK_CATS:
            if self.mark_cap_hebrew and cat == "Mn" and any(a <= cp <= b for a, b in self.hebrew_marks):
                return "HMARK"
            return "MARK"
        f = self.folder.fold(ch)
        if f is not None and f.isascii():
            if f.lower() in self.letters_ascii:
                return f + "~" if (not ch.isascii() and f.lower() in self.split) else f
            return "LOWER_OTHER" if f.islower() else "UPPER_OTHER"
        if cat.startswith("L"):
            key = (f or ch).lower()
            if key in self.letters_other:
                return f"g:{key}"
            return "LETTER_OTHER"
        return "NONLETTER"

    # ---------------------------------------------------------------- resolve items -> class sets
    def is_letter_cls(self, c):
        return (c in ("LOWER_OTHER", "UPPER_OTHER", "LETTER_OTHER") or c.startswith("g:")
                or (len(c) == 1 and c.isascii() and c.isalpha()) or (len(c) == 2 and c[1] == "~"))

    def ptype(self, c):
        base = c[0] if (len(c) == 2 and c[1] == "~") else c
        if c == "LOWER_OTHER" or (len(base) == 1 and base.isascii() and base.isalpha() and base.islower()):
            return "lo"
        if c == "UPPER_OTHER" or (len(base) == 1 and base.isascii() and base.isalpha() and base.isupper()):
            return "up"
        if self.is_letter_cls(c):
            return "let"
        return "non"

    def L(self, x):
        if x.isascii():
            return frozenset(y for y in (x, x.upper(), x + "~", x.upper() + "~") if y in self.classes)
        return frozenset([f"g:{(self.folder.fold(x) or x).lower()}"]) & frozenset(self.classes)

    def _items_to_classes(self, items):
        out = set()
        for it in items:
            if it[0] == "let":
                out |= self.L(it[1])
            elif it[0] == "chr":
                out.add(it[1])
            elif it[0] == "sep":
                for c in self.seps[it[1]]:
                    out.add("WS" if c == "WS" else c)
            elif it[0] == "any":
                out |= self.any_letter
        return frozenset(out & set(self.classes))

    def _resolve(self):
        self.any_letter = frozenset(c for c in self.classes if self.is_letter_cls(c))
        for p in self.pats:
            p.csteps = [(k, self._items_to_classes(items)) for k, items in p.steps]
            for k, cl in p.csteps:
                if not cl:
                    raise ValueError(f"{p.id}: a step matches no character")

    # ---------------------------------------------------------------- DFA
    def cont_key(self, c):
        if len(c) == 1 and c.isascii() and c.isalpha():
            return c.lower(), False
        if len(c) == 2 and c[1] == "~":
            return c[0].lower(), True
        return None

    def end_ok(self, st, node, taint):
        ok = (st["alone"] == "allow") if node == 0 else st["trie"][node]["end"]
        return ok and not taint

    def _put(self, new, pid, pos, flag):
        new.add(("P", pid, pos, flag))
        steps = self.pats[pid].csteps
        if pos < len(steps) and steps[pos][0] == "star":
            new.add(("P", pid, pos + 1, flag))

    def _complete(self, p, new):
        if p.kind == "ban":
            return DEAD
        if p.kind == "stem":
            new.add(("C", p.stem_id, 0, False))
        elif p.kind == "ctx":
            new.add(("M", p.ctx_id))
        return None

    def _is_letter_step(self, st):
        return any(self.is_letter_cls(x) for x in st[1])

    def advance(self, threads, prev, c):
        new = set()
        for th in threads:
            if th[0] == "M":
                continue                                   # a context marker lives for one character
            if th[0] == "C":
                _, sid, node, taint = th
                st = self.stems[sid]
                trie = st["trie"]
                if c == "FFFD":
                    return DEAD                            # an unseen character right after a banned stem
                if self.is_letter_cls(c):
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
                if flag in ("W", "T"):
                    new.add(th)                            # more bytes of the same unseen character
                elif flag == "" and p.wild:
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
                if pos + 1 == len(steps):
                    if self._complete(p, new) is DEAD:
                        return DEAD
                    continue
                self._put(new, pid, pos + 1, flag)
        markers = {th[1] for th in threads if th[0] == "M"}
        up = self.ptype(c) == "up"
        for pid, p in enumerate(self.pats):
            if p.word_init and not (prev == "non" or (prev == "lo" and up)):
                continue
            if p.ctx_of is not None:
                has = p.ctx_of in markers
                if (p.variant == "ctx") != has:
                    continue
            if c == "FFFD":
                if p.wild and self._is_letter_step(p.csteps[0]):
                    if len(p.csteps) == 1:
                        return DEAD
                    self._put(new, pid, 1, "W")
            elif c in p.csteps[0][1]:
                if len(p.csteps) == 1:
                    if self._complete(p, new) is DEAD:
                        return DEAD
                else:
                    self._put(new, pid, 1, "")
        return frozenset(new)

    def accepting(self, threads):
        return all(self.end_ok(self.stems[th[1]], th[2], th[3]) for th in threads if th[0] == "C")

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
        part = [0 if acc[k] else 1 for k in range(n)]
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
        m = len(set(part))
        mt, macc = {}, {}
        for k in range(n):
            b = part[k]
            macc[b] = acc[k]
            for c in ordinary:
                t = trans[(k, c)]
                mt[(b, c)] = None if t is None else part[t]
        # renumber: start = 0, then BFS order (deterministic)
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
        self.ordinary_classes = ordinary
        self.mark_classes = marks
        return self

    # ---------------------------------------------------------------- EBNF
    def emit(self):
        atom = {}

        def atom_name(c):
            if c == "MARK": return "zmk"
            if c == "HMARK": return "zhm"
            if c == "LETTER_OTHER": return "zlt"
            if c == "NONLETTER": return "znl"
            if c == "LOWER_OTHER": return "xlo"
            if c == "UPPER_OTHER": return "xup"
            if c == "WS": return "zws"
            if c.startswith("g:"): return "q" + "".join(f"{ord(x):x}" for x in c[2:])
            if len(c) == 2 and c[1] == "~": return ("vl" if c[0].islower() else "vu") + c[0].lower()
            if len(c) == 1 and c.isascii() and c.isalpha(): return ("l" if c.islower() else "u") + c.lower()
            return None
        for c in self.classes:
            if c in BIG or c in ("MARK", "HMARK") or len(self.cps[c]) > 1:
                nm = atom_name(c)
                assert nm, c
                atom[c] = nm

        def heads(cls_list):
            singles = [c for c in cls_list if c not in atom]
            h = [charclass([cp for c in singles for cp in self.cps[c]])] if singles else []
            return h + [atom[c] for c in cls_list if c in atom]
        cap = self.mark_cap
        wrapped = bool(cap)
        core = (lambda b: f"c{b}") if wrapped else (lambda b: "root" if b == 0 else f"s{b}")
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
            if not wrapped:
                alts += [f"{atom[m]} {full(b)}" for m in self.mark_classes]      # marks: transparent
                lines.append(f"{full(b)} ::= " + " | ".join(alts))
                continue
            if not alts:
                raise ValueError(f"state {b} has no way to continue at all (a trap): refusing to emit a grammar "
                                 "that would force combining marks until max_tokens")
            lines.append(f"{core(b)} ::= " + " | ".join(alts))
            lines += self._mark_wrappers(b, full(b), core(b), atom)
        lines += [f"{nm} ::= {charclass(self.cps[c])}" for c, nm in atom.items()]
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


def build_trie(words, prefixes):
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
        self._cc = {}
        self.pats = [p for p in spec.pats if p.kind != "ctx"]
        self.ctx_pats = {p.ctx_id: p for p in spec.pats if p.kind == "ctx"}
        self.first = [self._first_set(p) for p in self.pats]
        self.heb = spec.hebrew_marks
        # index the patterns by what their first step can match, so most characters try none
        self.by_fold, self.by_raw, self.any_letter = {}, {}, []
        for p, (folds, raws, anyl) in zip(self.pats, self.first):
            for f in folds:
                self.by_fold.setdefault(f, []).append(p)
            for r in raws:
                self.by_raw.setdefault(r, []).append(p)
            if anyl:
                self.any_letter.append(p)

    def _first_set(self, p):
        kind, items = p.steps[0]
        folds, raws, anyl = set(), set(), False
        for it in items:
            if it[0] == "let":
                folds.add(it[1])
            elif it[0] == "chr":
                raws.add(it[1])
            elif it[0] == "sep":
                for c in self.s.seps[it[1]]:
                    if c == "WS":
                        raws.update(WS_EXTRA)
                    else:
                        raws.add(c)
            else:
                anyl = True
        return folds, raws, anyl

    def _char(self, ch):
        r = self._cc.get(ch)
        if r is None:
            cat = unicodedata.category(ch)
            if cat in MARK_CATS:
                cp = ord(ch)
                r = ("M", cat == "Mn" and any(a <= cp <= b for a, b in self.heb))
            else:
                f = self.s.folder.fold(ch)
                letter = (cat.startswith("L") or (f is not None and f.isalpha())) and ch != "�"
                fold = (f or ch).lower() if letter else ch
                if f is not None and f.isascii():
                    case = "lo" if f.islower() else "up"
                else:
                    case = "let" if letter else "non"
                acc = (not ch.isascii()) and f is not None and f.isascii() and f.lower() in self.split
                r = (fold, letter, case, acc)
            self._cc[ch] = r
        return r

    def units(self, text):
        idx, fold, letter, case, acc, raw = [], [], [], [], [], []
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
                i = j
                continue
            idx.append(i)
            raw.append(text[i])
            fold.append(r[0])
            letter.append(r[1])
            case.append(r[2])
            acc.append(r[3])
            i += 1
        return (idx, raw, fold, letter, case, acc), runs

    def _item_ok(self, it, U, k):
        idx, raw, fold, letter, case, acc = U
        if it[0] == "let":
            return letter[k] and fold[k] == it[1]
        if it[0] == "chr":
            return raw[k] == it[1]
        if it[0] == "sep":
            for c in self.s.seps[it[1]]:
                if c == "WS":
                    if raw[k] in WS_EXTRA:
                        return True
                elif raw[k] == c:
                    return True
            return False
        return letter[k]

    def _match(self, steps, U, k, pos, ends):
        n = len(U[0])
        if pos == len(steps):
            ends.add(k)
            return
        kind, items = steps[pos]
        if kind == "star":
            self._match(steps, U, k, pos + 1, ends)
            j = k
            while j < n and any(self._item_ok(it, U, j) for it in items):
                j += 1
                self._match(steps, U, j, pos + 1, ends)
            return
        if k < n and any(self._item_ok(it, U, k) for it in items):
            self._match(steps, U, k + 1, pos + 1, ends)

    def _word_start(self, U, k):
        if k == 0:
            return True
        letter, case = U[3], U[4]
        return (not letter[k - 1]) or (case[k - 1] == "lo" and case[k] == "up")

    def _cont_banned(self, st, U, k):
        idx, raw, fold, letter, case, acc = U
        trie = st["trie"]
        node = 0
        n = len(idx)
        while True:
            if k >= n or not letter[k]:
                ok = (st["alone"] == "allow") if node == 0 else trie[node]["end"]
                return not ok
            if acc[k]:
                return True
            f = fold[k]
            child = trie[node]["ch"].get(f) if (f.isascii() and f.isalpha()) else None
            if child is None:
                return st["other"] == "ban"
            if trie[child]["esc"]:
                return False
            node = child
            k += 1

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
        """[(orig start, orig end, rule id)] of every banned form, plus combining-mark-cap violations ('MARKS')."""
        U, runs = self.units(text)
        idx, raw, fold, letter, case, acc = U
        n = len(idx)
        out = []
        by_fold, by_raw, anyl = self.by_fold, self.by_raw, self.any_letter
        order = {id(p): i for i, p in enumerate(self.pats)}
        for k in range(n):
            f, r, lt = fold[k], raw[k], letter[k]
            cand = (by_fold.get(f, []) + anyl) if lt else []
            if r in by_raw:
                cand = cand + by_raw[r]
            if not cand:
                continue
            if len(cand) > 1:
                cand = sorted({id(p): p for p in cand}.values(), key=lambda p: order[id(p)])
            ws = None
            for p in cand:
                if p.word_init:
                    if ws is None:
                        ws = self._word_start(U, k)
                    if not ws:
                        continue
                if p.ctx_of is not None:
                    has = self._ctx_before(self.ctx_pats[p.ctx_of], U, k)
                    if (p.variant == "ctx") != has:
                        continue
                ends = set()
                self._match(p.steps, U, k, 0, ends)
                for e in sorted(ends):
                    if p.kind == "ban" or self._cont_banned(self.s.stems[p.stem_id], U, e):
                        g = max(e, k + 1)
                        while g < n and letter[g]:
                            g += 1
                        out.append((idx[k], idx[g - 1] + 1, p.id.split("#")[0].split("@")[0]))
                        break
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
        return ch.isalnum() or unicodedata.category(ch) in MARK_CATS or ch in "_'’-"

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
            spans = []
            for s, e, _r in h:
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


def load_corpus(db_path, scope="branch"):
    """[{role, ts, t}] for every user/assistant message on each chat's current branch (table rows first,
    JSON history fills gaps; the pointer is chat.current_message_id, else history.currentId).
    scope='all' adds every other stored message too (regenerations, other branches)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(chat_message)")} if con.execute(
        "select 1 from sqlite_master where type='table' and name='chat_message'").fetchone() else set()
    chat_cols = {r[1] for r in con.execute("PRAGMA table_info(chat)")}
    out = []
    sel = "id, chat, " + ("current_message_id" if "current_message_id" in chat_cols else "null")
    for cid, chat, cur_col in con.execute(f"select {sel} from chat"):
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
        for mid in ids:
            m = merged[mid]
            if m.get("role") in ("user", "assistant") and isinstance(m.get("t"), str) and m["t"].strip():
                out.append({"role": m["role"], "ts": m.get("ts") or 0, "t": m["t"], "branch": mid in seen})
    con.close()
    return out


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
        ctx = r.get("context")
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
