"""
STT quality eval — the "is it actually good?" layer (Tier-3, run against a live
service). For each audio fixture, transcribe it through the running Whisper
service and score the result against a known reference transcript with WER.

This is intentionally separate from the three standard tiers: Tier-1 proves the
logic, the boot self-test proves the service decodes audio and runs, and *this*
proves the transcriptions are accurate enough to trust.

Usage
-----
    # against a local pod
    STT_URL=http://127.0.0.1:9000 python stt_eval.py
    # against a RunPod proxy
    STT_URL=https://<POD>-9000.proxy.runpod.net python stt_eval.py
    # stricter bar
    WER_THRESHOLD=0.15 STT_URL=... python stt_eval.py

Fixtures
--------
Drop matched pairs into ./fixtures/:  <name>.wav  +  <name>.txt
(the .txt holding the exact spoken words). See fixtures/README.md. Real speech
clips are operator-supplied — synthetic silence can't measure accuracy.

Exit code: 0 if every fixture is at/under the threshold (or there are none to
run), 1 if any fixture exceeds it.

Requires: httpx  (pip install httpx)
"""
from __future__ import annotations

import glob
import os
import sys

import httpx

import wer as werlib

STT_URL = (os.environ.get("STT_URL") or "http://127.0.0.1:9000").rstrip("/")
WER_THRESHOLD = float(os.environ.get("WER_THRESHOLD", "0.30"))
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
TIMEOUT_S = float(os.environ.get("STT_EVAL_TIMEOUT_S", "120.0"))


def _transcribe(path: str) -> str:
    with open(path, "rb") as f:
        audio = f.read()
    files = {"file": (os.path.basename(path), audio, "audio/wav")}
    data = {"model": "whisper-1", "response_format": "json"}
    r = httpx.post(
        f"{STT_URL}/v1/audio/transcriptions", files=files, data=data, timeout=TIMEOUT_S
    )
    r.raise_for_status()
    return r.json().get("text", "")


def _discover() -> list[tuple[str, str]]:
    """Return [(wav_path, expected_text)] for every <name>.wav with a sibling .txt."""
    pairs = []
    for wav in sorted(glob.glob(os.path.join(FIXTURES_DIR, "*.wav"))):
        txt = os.path.splitext(wav)[0] + ".txt"
        if os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                pairs.append((wav, f.read().strip()))
    return pairs


def main() -> int:
    fixtures = _discover()
    if not fixtures:
        print(f"No fixtures found in {FIXTURES_DIR} (need <name>.wav + <name>.txt).")
        print("Add real speech clips to run the quality eval — see fixtures/README.md.")
        return 0

    print(f"STT quality eval against {STT_URL}  (WER threshold {WER_THRESHOLD:.2f})")
    print("-" * 72)
    worst = 0.0
    failures = 0
    for wav, expected in fixtures:
        name = os.path.basename(wav)
        try:
            hyp = _transcribe(wav)
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failures += 1
            continue
        score = werlib.wer(expected, hyp)
        worst = max(worst, score)
        ok = score <= WER_THRESHOLD
        if not ok:
            failures += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<28} WER={score:.3f}")
        print(f"         ref: {expected}")
        print(f"         hyp: {hyp}")
    print("-" * 72)
    total = len(fixtures)
    print(f"{total - failures}/{total} within threshold; worst WER={worst:.3f}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
