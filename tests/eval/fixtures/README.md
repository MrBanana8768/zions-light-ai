# STT eval fixtures

Drop matched pairs here:

```
my_clip.wav     # the audio
my_clip.txt     # the exact words spoken, plain UTF-8 text
```

`stt_eval.py` discovers every `*.wav` that has a sibling `*.txt` and scores the
transcription against it with WER.

## Guidance

- **Format:** WAV is safest (the service decodes via ffmpeg, so mp3/m4a/ogg/webm
  also work, but WAV avoids codec surprises). Mono, 16 kHz is ideal.
- **Length:** short and clear — 5–15 seconds each.
- **Transcript:** write exactly what is said. Casing and punctuation don't
  matter (the metric normalizes them); the *words* do.
- **Coverage:** a few clean clips, plus at least one with background noise and
  one with an accent, exercises the model honestly.

These files are **not committed** by default and **not** baked into the image
(`tests/` is in `.dockerignore`) — they're yours to supply per environment.
Recording yourself reading a paragraph is a perfectly good first fixture.
