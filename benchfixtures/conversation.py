"""A production-shaped conversation, generated from a fixed seed.

WHY THIS EXISTS AND WHY IT IS SO PARTICULAR ABOUT LENGTH.

The only performance numbers this tree carries are four data points in a
comment above `_redact_degenerate_turns`'s call site in main.py:

    20 turns 4.6ms, 40 turns 9.5ms, 85 turns 65ms, 170 turns 446ms

That is 0.23 ms/turn at 20 turns and 2.62 ms/turn at 170 - an eleven-fold
rise in the PER-TURN cost of a function that visits each assistant turn
exactly once and does work proportional to that turn's own length. A per-turn
cost cannot grow with the number of turns unless the turns themselves grow.
So either the detector is super-linear in ways nobody has explained, or the
fixture those four numbers came from produced longer replies at higher turn
counts and the curve is an artefact of the fixture.

The distinction decides whether v3.1.9's planned fix (bound the scan to the
last ~84 turns) is worth anything: against a genuinely super-linear scan it
is a large win at any history size, and against a linear one it is exactly
the ratio 84/n and nothing more.

This module therefore generates replies whose LENGTH IS A DECLARED PARAMETER,
never a function of position - unless `growth` is asked for explicitly, which
exists solely to reproduce the artefact and show what it looks like.

WHAT IS MEASURED AND WHAT IS ASSUMED. From the live pod: ~2,300 messages,
~1,150 assistant replies, median assistant reply 5,248 characters, one
conversation, growing ~2 MB/day. Those are the measured facts and they are
the defaults below. The SPREAD (lognormal, sigma 0.85) and the user-turn
length (600 characters) are assumptions, stated here so nobody mistakes them
for measurements; neither affects path B at all, because
`_redact_degenerate_turns` only ever runs the detector on assistant turns.

THE CONTENT MUST NOT BE DEGENERATE. `reply_is_degenerate` returns early on
its first positive rule, so a fixture that trips one measures a SHORTER code
path than production does - and `_redact_degenerate_turns` would also start
logging and copying dicts. The generator below stays clear of every rule by
construction and the harness verifies it (0 of N flagged) rather than
assuming it.

It must also not be trivially cheap. Real replies from this model are
markdown with headings, bullet lists, em-dashes and the occasional
box-drawing rule, and they run 7-14% non-ASCII on assistant turns - which
matters because the script-drift rule walks every alphabetic character
through `unicodedata.name()`, and an all-ASCII fixture would make that loop
look faster than it is. Greek and Hebrew fragments appear because this user
quotes scripture; two scripts is deliberately below the >=3 the drift rule
needs, so they cost the walk without tripping it.
"""

from __future__ import annotations

import math
import random

# Measured on the live pod, 2026-09-11.
MEDIAN_ASSISTANT_CHARS = 5248
# Assumed, not measured. See the module docstring.
MEDIAN_USER_CHARS = 600
# Lognormal sigma for the assistant-length spread. Assumed. exp(sigma^2/2) is
# the mean/median ratio, so 0.85 puts the mean at ~1.44x the median, which is
# the shape a chat model's reply lengths usually take.
LENGTH_SIGMA = 0.85

# One seed for the whole benchmark. Changing it changes every number in the
# report, which is why it is a module constant and not a call-site default.
SEED = 20260911

_WORDS = (
    "the and for that with this from have been will your when they which "
    "about because through between against before after under above into "
    "grace mercy covenant psalm scripture chapter verse prayer wisdom "
    "context window compactor summary rollup hierarchy watermark chunk "
    "request latency budget token tokenizer threshold measured production "
    "morning evening quiet faithful gentle steady patient ordinary honest "
    "remember forget carry hold notice listen answer question wonder "
    "because although however therefore meanwhile afterwards finally "
    "conversation memory attention structure paragraph sentence fragment"
).split()

# A few real non-Latin fragments. TWO scripts only: the drift rule needs
# three distinct non-Latin scripts at 20% (or five at 3%), so these cost the
# unicodedata walk what real quotations cost it and cannot flag the reply.
_GREEK = "οὕτως γὰρ ἠγάπησεν ὁ θεὸς"
_HEBREW = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"

# Box-drawing, curly quotes and em-dashes: the characters this model actually
# draws, and the ones that make an assistant turn 7-14% non-ASCII.
_RULE = "━" * 48
_SMART = ("—", "’", "“", "”", "…", "·")


def _sentence(rng: random.Random) -> str:
    n = rng.randint(8, 24)
    words = [rng.choice(_WORDS) for _ in range(n)]
    if rng.random() < 0.30:
        words.insert(rng.randrange(len(words)), rng.choice(_SMART))
    s = " ".join(words)
    return s[0].upper() + s[1:] + rng.choice((".", ".", ".", "?", "!"))


def _paragraph(rng: random.Random) -> str:
    return " ".join(_sentence(rng) for _ in range(rng.randint(2, 6)))


def _bullets(rng: random.Random) -> str:
    # Bounded at 9 items and each item deliberately longer than
    # DEGENERATE_LIST_ITEM_CHARS (30), because the list-run backstop fires at
    # 50 consecutive items of <=30 characters and a fixture that trips it
    # would be measuring the early-return path instead of the full scan.
    out = []
    for _ in range(rng.randint(3, 9)):
        out.append("- " + _sentence(rng))
    return "\n".join(out)


def _block(rng: random.Random) -> str:
    """One markdown unit of a reply, in the proportions this model writes."""
    r = rng.random()
    if r < 0.55:
        return _paragraph(rng)
    if r < 0.75:
        return "## " + " ".join(rng.choice(_WORDS) for _ in range(3)).title()
    if r < 0.90:
        return _bullets(rng)
    if r < 0.95:
        return _RULE
    if r < 0.98:
        return f"> {_GREEK}\n>\n> {_sentence(rng)}"
    return f"> {_HEBREW}\n>\n> {_sentence(rng)}"


def _text_of_length(rng: random.Random, target: int) -> str:
    """Markdown prose of exactly `target` characters.

    Exactly, not approximately: the whole point of the fixed-length sweep is
    that reply length is held constant while turn count moves, and a
    generator that overshoots by a random amount reintroduces the very
    confound the sweep exists to remove. The final trim can cut a word in
    half; that is harmless to every rule in the detector and cheaper than
    rejection sampling.
    """
    parts: list[str] = []
    size = 0
    while size < target:
        b = _block(rng)
        parts.append(b)
        size += len(b) + 2
    return "\n\n".join(parts)[:target]


def _lengths(rng: random.Random, n: int, median: int, sigma: float) -> list[int]:
    return [max(80, int(median * math.exp(rng.gauss(0.0, sigma)))) for _ in range(n)]


def conversation(
    n_assistant: int,
    *,
    assistant_chars: int | None = MEDIAN_ASSISTANT_CHARS,
    user_chars: int = MEDIAN_USER_CHARS,
    growth: bool = False,
    spread: bool = False,
    seed: int = SEED,
    system: bool = True,
) -> list[dict]:
    """`n_assistant` exchanges as an OpenAI-shaped message array.

    Exactly one knob decides reply length and the three settings are mutually
    exclusive on purpose, because mixing them is how the confound got in:

      * default          - every reply EXACTLY `assistant_chars` long.
      * spread=True      - lengths drawn lognormal around `assistant_chars`,
                           independent of position. Realistic, still not
                           correlated with turn count.
      * growth=True      - reply i is (i+1) * assistant_chars/... long, i.e.
                           length rises with position. This is the shape a
                           naive fixture has, and it exists here ONLY to
                           demonstrate what it does to the measured curve.

    A prefix of a longer conversation is the same conversation: the generator
    is driven by one seeded RNG consumed in order, so
    `conversation(20) == conversation(1200)[:41]` for the default and growth
    modes. That is what makes the turn-count sweep a controlled comparison
    rather than eight unrelated fixtures.
    """
    if growth and spread:
        raise ValueError("growth and spread are alternatives, not a combination")
    rng = random.Random(seed)
    base = assistant_chars or MEDIAN_ASSISTANT_CHARS
    if spread:
        lengths = _lengths(rng, n_assistant, base, LENGTH_SIGMA)
    elif growth:
        # Linear in position, normalised so the MEAN over the first 170 turns
        # equals `base`. Without the normalisation a growth fixture is simply
        # a bigger fixture and the comparison against the fixed-length sweep
        # says nothing; with it, both sweeps push the same total volume
        # through the detector at 170 turns and only the DISTRIBUTION differs.
        step = 2.0 * base / 171.0
        lengths = [max(80, int(step * (i + 1))) for i in range(n_assistant)]
    else:
        lengths = [base] * n_assistant

    out: list[dict] = []
    if system:
        out.append({
            "role": "system",
            "content": "You are a steady, faithful companion. Speak plainly.",
        })
    for i, ln in enumerate(lengths):
        out.append({
            "role": "user",
            "content": f"[{i}] " + _text_of_length(rng, user_chars),
        })
        out.append({
            "role": "assistant",
            "content": f"[{i}] " + _text_of_length(rng, ln),
        })
    return out


def summary_state(
    conv_id: str,
    *,
    l1_chunks: int = 10,
    l2_chapters: int = 5,
    l3: bool = True,
    tail_fp: int = 64,
    last_turn: int = 2300,
) -> dict:
    """A summary state file at FULL tier fill - the largest one this system
    can legitimately hold.

    The tiers are bounded by construction (`state["l1"] = l1[L2_CHUNK_SIZE:]`
    drops ten chunks at a time, and an L3 refresh empties l2 into a separate
    archive file), so a 1,150-reply conversation does NOT have a large state
    file - it has a small one that is read on every single turn. Sizing the
    fixture at the tier ceilings is the honest worst case, and it matters
    that the answer comes out small: it means path C's cost is open/stat
    latency on the volume, not bytes, and a slow volume is therefore the
    whole risk rather than a marginal one.

    Chunk text is sized from the tier token caps at ~4 characters per token
    (L1 500, L2 1200, L3 2000), which is the same estimator main.py falls
    back to when no tokenizer is loaded.
    """
    rng = random.Random(SEED + 1)
    span = max(1, last_turn // max(1, l1_chunks + l2_chapters))

    def chunk(idx: int, chars: int, first: int, last: int) -> dict:
        return {
            "text": _text_of_length(rng, chars),
            "first_turn": first,
            "last_turn": last,
        }

    state: dict = {
        "conv_id": conv_id,
        "updated_at": "2026-09-11T23:45:00+00:00",
        "l1": [
            chunk(i, 500 * 4, last_turn - (l1_chunks - i) * 20 + 1,
                  last_turn - (l1_chunks - i - 1) * 20)
            for i in range(l1_chunks)
        ],
        "l2": [
            chunk(i, 1200 * 4, i * span + 1, (i + 1) * span)
            for i in range(l2_chapters)
        ],
        "l3": chunk(0, 2000 * 4, 1, last_turn) if l3 else None,
        "last_summarized_turn": last_turn,
        "turns_seen": last_turn,
        "tail_fp": [f"{rng.getrandbits(128):032x}" for _ in range(tail_fp)],
        "head_fp": f"{rng.getrandbits(128):032x}",
        "window_turns": last_turn,
    }
    return state
