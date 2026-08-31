#!/usr/bin/env python3
"""Ask vLLM directly whether it will serve images, and what one really costs.

    /opt/compactor-venv/bin/python /data/scripts/probe-vision.py

CHANGES NOTHING. It talks to vLLM on port 8000, bypassing the compactor
entirely, so it cannot touch a conversation, a fact store, or any config.
Safe to run on a live pod while she is using it.

WHY IT EXISTS. Two things are unknown before re-enabling images, and neither
can be answered from the repo:

  1. Does this fp8-quantized vision build actually accept image input? The
     compactor's modality guard resolves `multimodal` from the HF config's
     vision tower, which says what the WEIGHTS can do — not what vLLM 0.19
     will serve after `--quantization fp8`. The compactor side is unit
     tested; this boundary is not.

  2. Is COMPACTOR_IMAGE_TOKENS=4096 right? main.py's own comment says that
     constant is "roughly half the true cost of a Mistral3 vision tile", and
     the budget margin has been absorbing the difference — dormant, because
     COMPACTOR_MAX_RETAINED_IMAGES=0 strips every image before it is ever
     charged. Turning images on wakes that path. Guessing at the number is
     what the comment warns against, so this measures it instead.

HOW IT MEASURES. vLLM reports usage.prompt_tokens on every reply. Send the
same prompt twice — once text-only, once with an image — and the difference
IS the per-image cost, exactly, with no estimating. Repeated at three
resolutions, because the cost tiles with size.

Stdlib only (urllib, zlib, struct): no httpx, no PIL, nothing to install,
and it runs under any python3 on the box.
"""

import argparse
import base64
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib

VLLM = "http://localhost:8000"


def say(m=""):
    print(m, flush=True)


def png(w: int, h: int) -> bytes:
    """A real PNG of arbitrary size, without PIL. Gradient rather than a flat
    fill so the encoder cannot collapse it into something unrepresentative."""
    raw = b""
    for y in range(h):
        row = bytes([(x * 7 + y * 3) % 256 for x in range(w) for _ in (0, 1, 2)])
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


def post(url: str, payload: dict, timeout: int = 120):
    """Returns (status, parsed_or_text). Never raises for an HTTP error — a
    4xx body is the most informative thing this script can report."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
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


def model_name(base: str) -> str | None:
    st, body = post(base + "/v1/models", {})
    if isinstance(body, dict) and body.get("data"):
        return body["data"][0].get("id")
    # /v1/models is a GET; fall back to reading it properly.
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
            data = json.loads(r.read().decode())
        return data["data"][0]["id"]
    except Exception:
        return None


def ask(base: str, model: str, parts, max_tokens: int = 8):
    return post(
        base + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
    )


def tokenize(base: str, model: str, parts):
    """What /tokenize says the same payload costs. None if it will not answer.

    This is the question the repo contradicts itself on. tokens.py says
    "vLLM's /tokenize prices vision tokens and this cannot"; main.py adds a
    flat COMPACTOR_IMAGE_TOKENS per image on the LOCAL path only, and takes
    /tokenize's number raw when it answers. So when /tokenize is up, an
    image is priced by /tokenize alone — and if /tokenize applies the chat
    template WITHOUT running the multimodal processor, that price is zero
    for the image and the budget guard is blind to the most expensive thing
    in the request.

    usage.prompt_tokens from a real completion is ground truth. Comparing
    the two settles it.
    """
    st, body = post(
        base + "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": parts}],
            "add_generation_prompt": True,
        },
    )
    if st != 200 or not isinstance(body, dict):
        return None
    n = body.get("count")
    if n is None and isinstance(body.get("tokens"), list):
        n = len(body["tokens"])
    return int(n) if n is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=VLLM)
    ap.add_argument("--sizes", default="256,512,1024",
                    help="square image edges to measure, comma separated")
    args = ap.parse_args()

    say("=" * 68)
    say("VISION PROBE — reads only. vLLM directly, compactor not involved.")
    say("=" * 68)

    model = model_name(args.url)
    if not model:
        say(f"FAIL: no model served at {args.url}. Is vLLM up?")
        return 2
    say(f"model: {model}")

    # --- baseline: the same prompt, no image ------------------------------
    PROMPT = "Reply with the single word: ok"
    st, body = ask(args.url, model, [{"type": "text", "text": PROMPT}])
    if st != 200 or not isinstance(body, dict):
        say(f"FAIL: even a TEXT request failed ({st}): {str(body)[:300]}")
        return 3
    base_tokens = body["usage"]["prompt_tokens"]
    say(f"text-only baseline: {base_tokens} prompt tokens")
    say("")

    # --- the real question ------------------------------------------------
    estimate = 4096  # COMPACTOR_IMAGE_TOKENS default
    base_tok = tokenize(args.url, model, [{"type": "text", "text": PROMPT}])
    say(f"{'image':>9}  {'REAL':>8}  {'/tokenize':>9}  {'flat 4096':>9}   verdict")
    say("-" * 68)
    results = []
    for raw in args.sizes.split(","):
        dim = int(raw.strip())
        uri = "data:image/png;base64," + base64.b64encode(png(dim, dim)).decode()
        parts = [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": uri}},
        ]
        st, body = ask(args.url, model, parts)
        if st != 200 or not isinstance(body, dict) or "usage" not in body:
            say(f"{dim}x{dim}: REFUSED ({st})")
            say("")
            say("  vLLM will not serve images on this build. That is the answer —")
            say("  do NOT raise COMPACTOR_MAX_RETAINED_IMAGES. The body was:")
            say(f"  {str(body)[:500]}")
            return 4
        real = body["usage"]["prompt_tokens"] - base_tokens
        tk = tokenize(args.url, model, parts)
        tk_cost = None if (tk is None or base_tok is None) else tk - base_tok
        # Does /tokenize see the image at all? Anything under a tenth of the
        # real cost means it priced the template and ignored the pixels.
        blind = tk_cost is not None and tk_cost < real * 0.1
        verdict = (
            "/tokenize BLIND" if blind
            else "ok" if tk_cost is not None and tk_cost >= real * 0.9
            else "/tokenize LOW" if tk_cost is not None
            else "/tokenize n/a"
        )
        results.append((dim, real, tk_cost))
        say(f"{dim:>7}px  {real:>8}  {str(tk_cost):>9}  "
            f"{real - estimate:>+9}   {verdict}")

    say("")
    worst = max(r for _, r, _ in results)
    blind_any = any(t is not None and t < r * 0.1 for _, r, t in results)
    say("=" * 68)
    say("ACCEPTED — vLLM serves images on this build.")
    say("")
    if blind_any:
        say("BUT /tokenize DOES NOT PRICE THE IMAGE. The compactor takes")
        say("/tokenize's number raw whenever it answers and adds no per-image")
        say("estimate on that path, so the budget guard is blind to the single")
        say("most expensive thing in the request — every image is effectively")
        say(f"free to it, while really costing up to {worst} tokens.")
        say("")
        say("Do NOT raise COMPACTOR_MAX_RETAINED_IMAGES on this alone: the flat")
        say("estimate would have to be applied on the /tokenize path too, which")
        say("is a code change, not a config change.")
    elif worst > estimate:
        say(f"The largest image really costs {worst} tokens against a fallback")
        say(f"estimate of {estimate}: an UNDERCOUNT of {worst - estimate} whenever")
        say("/tokenize is degraded — which it was 46 times in the 08-31 bundle.")
        say(f"Set COMPACTOR_IMAGE_TOKENS={((worst + 511) // 512) * 512} (next 512 up)")
        say("before raising COMPACTOR_MAX_RETAINED_IMAGES.")
    else:
        say(f"/tokenize prices the image, and the {estimate} fallback covers the")
        say(f"largest measured cost ({worst}). Both paths are sound —")
        say("COMPACTOR_MAX_RETAINED_IMAGES=1 is safe to set.")
    say("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
