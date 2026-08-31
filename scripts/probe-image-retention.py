#!/usr/bin/env python3
"""Reproduce the second-image turn against the COMPACTOR, and print the reply.

    /opt/compactor-venv/bin/python /data/scripts/probe-image-retention.py

WHAT THIS SEPARATES. On 2026-08-31 the second image posted in a conversation
came back with a near-instant, near-empty reply ("follow up"), every time.
The bundle rules out the plumbing: every POST /api/chat/completions returned
200, no vLLM 400 anywhere that evening, the images arrived as real image_url
parts (image retention fired on each attempt, correctly demoting exactly one
older image), and the chat template rejection at 20:28 was a cancelled-stream
artefact from a DIFFERENT conversation. What is left is the prompt itself.

Two candidates produce that prompt, and they need different fixes:

  OURS      _apply_image_retention rewrites the older image turn into
            "[1 image shared earlier in this conversation]". That is the
            only thing that changes between the first image (which works)
            and the second (which does not) — the first upload returns
            early from retention without touching anything.

  THEIRS    OpenWebUI builds the payload differently once a conversation
            holds two images — a different encoding, an extra task
            preamble, a follow-up template leaking into the chat turn.

This sends a conversation shaped exactly like hers — system, three
exchanges, second image last — THROUGH THE COMPACTOR on 8080, so retention
runs for real. If the reply is short and useless here, the shape our code
produces is the cause and OpenWebUI is exonerated. If it answers properly,
our retention is fine and the payload OpenWebUI sends is the difference.

It also prints the ONE-image control, because a test that only exercises the
failing case cannot tell "the second image breaks it" from "images never
worked through the compactor at all".

WRITES NOTHING PERMANENT by default: it uses a throwaway conversation id, so
her real conversations are untouched. --conv overrides it if you want to aim
somewhere specific. It does leave facts/embeddings under that scratch id;
clear them with the admin endpoint afterwards if you care:

    curl -s -X POST localhost:8080/admin/conversations/<id>/clear

Stdlib only, same as probe-vision.py.
"""

import argparse
import base64
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib

COMPACTOR = "http://localhost:8080"


def say(m=""):
    print(m, flush=True)


def png(w: int, h: int, phase: int = 0) -> bytes:
    """Distinct gradient per `phase`, so the two images are not identical —
    an encoder or a cache collapsing two identical images into one would
    otherwise look like a passing test."""
    raw = b""
    for y in range(h):
        row = bytes(
            [(x * 7 + y * 3 + phase * 61) % 256 for x in range(w) for _ in (0, 1, 2)]
        )
        raw += b"\x00" + row

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def uri(dim: int, phase: int) -> str:
    return "data:image/png;base64," + base64.b64encode(png(dim, dim, phase)).decode()


def post(url: str, payload: dict, headers: dict, timeout: int = 240):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def img_turn(text: str, dim: int, phase: int) -> dict:
    parts = [{"type": "image_url", "image_url": {"url": uri(dim, phase)}}]
    if text:
        parts.insert(0, {"type": "text", "text": text})
    return {"role": "user", "content": parts}


def run(base: str, conv: str, model: str, messages: list, label: str) -> dict:
    import time

    t0 = time.monotonic()
    st, body = post(
        base + "/v1/chat/completions",
        {"model": model, "messages": messages, "stream": False, "max_tokens": 300},
        {"X-Conversation-Id": conv},
    )
    dt = time.monotonic() - t0
    say("=" * 70)
    say(f"{label}   ({len(messages)} messages sent)")
    say("=" * 70)
    if st != 200 or not isinstance(body, dict):
        say(f"  HTTP {st}: {str(body)[:400]}")
        return {"ok": False, "chars": 0, "secs": dt}
    try:
        reply = body["choices"][0]["message"]["content"]
    except Exception:
        say(f"  unexpected body: {str(body)[:400]}")
        return {"ok": False, "chars": 0, "secs": dt}
    say(f"  {dt:.1f}s, {len(reply)} chars")
    say("  ---")
    for line in (reply.strip() or "(EMPTY)").splitlines()[:12]:
        say(f"  | {line[:160]}")
    say("")
    return {"ok": True, "chars": len(reply.strip()), "secs": dt, "reply": reply}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=COMPACTOR)
    ap.add_argument("--conv", default="probe-retention-scratch")
    ap.add_argument("--model", default="probe")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    say("=" * 70)
    say("IMAGE RETENTION PROBE — through the COMPACTOR, scratch conversation")
    say("=" * 70)
    say(f"conv id: {args.conv}   (not a real conversation)")
    say("")

    SYS = {"role": "system", "content": "You are a helpful assistant."}
    A1 = {"role": "assistant", "content": "That looks like a colourful test pattern."}
    A2 = {"role": "assistant", "content": "Understood."}

    # CONTROL: one image, the case that works in production.
    one = [SYS, img_turn("What do you see in this picture?", args.size, 0)]
    r1 = run(args.url, args.conv + "-1", args.model, one, "CONTROL — first image")

    # THE FAILING SHAPE: 8 messages, second image last. Exactly what the
    # compactor logged as msgs=8 on every failed attempt, and the first
    # request where _apply_image_retention actually rewrites anything.
    two = [
        SYS,
        img_turn("What do you see in this picture?", args.size, 0),
        A1,
        {"role": "user", "content": "Thanks. Anything else worth noting?"},
        A2,
        img_turn("Here is a second picture — what is in this one?", args.size, 1),
    ]
    r2 = run(args.url, args.conv + "-2", args.model, two, "SECOND IMAGE — the failing shape")

    say("=" * 70)
    if not (r1["ok"] and r2["ok"]):
        say("A request failed outright — see the status above. That is a")
        say("different bug from the one this probe was written for.")
        return 3
    # A real answer to "what is in this one" runs to sentences. The production
    # symptom was ~1.5s and a couple of words, against ~26s for a normal turn.
    broke = r2["chars"] < max(40, r1["chars"] * 0.25)
    if broke:
        say("REPRODUCED. The second image collapses the reply through the")
        say("compactor, with no OpenWebUI involved at all.")
        say("")
        say(f"  first image : {r1['chars']} chars in {r1['secs']:.1f}s")
        say(f"  second image: {r2['chars']} chars in {r2['secs']:.1f}s")
        say("")
        say("So the cause is the shape OUR code produces — the retention")
        say("rewrite of the older image turn is the only thing that differs")
        say("between these two requests. Fix belongs in")
        say("_apply_image_retention / the payload it emits, not in OpenWebUI.")
        return 1
    say("NOT REPRODUCED. Both turns answered properly through the compactor:")
    say("")
    say(f"  first image : {r1['chars']} chars in {r1['secs']:.1f}s")
    say(f"  second image: {r2['chars']} chars in {r2['secs']:.1f}s")
    say("")
    say("Retention is not the cause. The difference is in what OpenWebUI")
    say("sends once a conversation holds two images — capture the real")
    say("payload next, rather than changing compactor code on suspicion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
