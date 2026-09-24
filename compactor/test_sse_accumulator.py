"""
SseAccumulator (v3.1.4) — the first tests it has ever had.

Until this file nothing asserted that a truncated stream reports truncated,
that a disconnect leaves complete() False, or what happens when feed() drops
a chunk. The last one is the new part: a dropped chunk used to leave text()
with a hole in it and every flag saying "fine", so the memory tail
fact-extracted, embedded and summarized a reply with a gap — on the
clean-finish path, silently, with one log line per process. holed() is the
flag the tail now skips on.

v3.1.7 (R7/R14): decoding moved from one `chunk.decode("utf-8",
errors="replace")` per feed() to a `codecs` incremental decoder held for the
accumulator's whole life, because `r.aiter_raw()` chunk boundaries fall at
arbitrary byte offsets, not character ones — decoding each chunk on its own
turned a multibyte character split across two reads into U+FFFD on BOTH
sides of the split, and `errors="replace"` never raises, so holed() (added
for exactly this) could not fire: the only path into it was `except
Exception`, and feed()'s bytes.decode() never raised on anything real —
only on a caller passing something with no .decode() at all (e.g. the None
this file used to feed it, which `aiter_raw()` can never actually yield).
[5] and [6] below now drive the flag through the two paths that ARE real:
a dropped SSE event that carried content, and finalize() finding an
incomplete sequence still buffered. [7] pins the split-character case
against BOTH the old failure (corruption) and holed() (must stay False,
since the character completes) and adds the truncated-mid-character case
(must set holed() — the character never completes).

Mutations this file exists to kill:

    _holed never set                        -> [5] the dropped event, [7c]
    _holed cleared by a good chunk          -> [6] sticky
    _truncated not set on "length"          -> [2]
    _complete set on any event              -> [3] the disconnect
    a dropped chunk raising                 -> [5] feed() never raises
    per-chunk decode instead of incremental -> [7b] the split character
    finalize() not checking the buffer      -> [7c] the truncated character

Every fixture is synthetic. The repo is public.

    python test_sse_accumulator.py
"""

import json
import os
import sys

os.environ.setdefault("MODEL_REPO", "")

import main  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _sse(obj) -> bytes:
    # ensure_ascii=False: real vLLM SSE traffic puts UTF-8 bytes on the wire
    # rather than \uXXXX escapes, and [7b]/[7c] below need actual multibyte
    # bytes to split a chunk boundary inside.
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


def _content(text: str) -> bytes:
    return _sse({"choices": [{"delta": {"content": text}}]})


def _finish(reason: str) -> bytes:
    return _sse({"choices": [{"delta": {}, "finish_reason": reason}]})


DONE = b"data: [DONE]\n\n"


def _fresh(*chunks) -> "main.SseAccumulator":
    acc = main.SseAccumulator()
    for c in chunks:
        acc.feed(c)
    acc.finalize()  # the documented contract: called once, after the last feed()
    return acc


print("[1] accumulation")
acc = _fresh(_content("Hello, "), _content("world."), _finish("stop"), DONE)
assert_eq(acc.text(), "Hello, world.", "content deltas concatenate in order")
assert_eq(acc.complete(), True, "finish_reason=stop -> complete")
assert_eq(acc.truncated(), False, "...and not truncated")
assert_eq(acc.usable(), True, "...so the stream is usable")
assert_eq(acc.holed(), False, "nothing was dropped")

# Chunk boundaries are TCP's business, not SSE's: one event split mid-JSON
# across two feeds must still parse once the \n\n delimiter arrives.
whole = _content("split ") + _content("event.")
cut = len(whole) // 2
acc = _fresh(whole[:cut], whole[cut:], _finish("stop"))
assert_eq(acc.text(), "split event.", "an event split across two chunks is reassembled")

acc = _fresh(
    _sse({"choices": [{"delta": {"role": "assistant"}}]}),  # role-only delta
    b": keep-alive\n\n",                                     # SSE comment
    b"data: \n\n",                                           # empty payload
    _content("x"),
    DONE,
)
assert_eq(acc.text(), "x", "role-only deltas, comments and empty payloads add no text")
assert_eq(acc.complete(), True, "[DONE] alone marks the stream complete")

acc = _fresh(_content("a"), b"data: {not json\n\n", _content("b"), _finish("stop"))
assert_eq(acc.text(), "ab", "one malformed event is dropped; accumulation continues")
assert_eq(acc.holed(), False, "a malformed EVENT is not a dropped CHUNK — the bytes were read")

acc = main.SseAccumulator()
assert_eq(acc.text(), "", "fresh accumulator has no text")
assert_eq((acc.complete(), acc.truncated(), acc.usable(), acc.holed()),
          (False, False, False, False), "and every flag is False")

print()
print("[2] truncation at the generation ceiling")
acc = _fresh(_content("The reply ran out of ro"), _finish("length"), DONE)
assert_eq(acc.text(), "The reply ran out of ro", "text is what arrived")
assert_eq(acc.complete(), True, "the stream DID terminate normally")
assert_eq(acc.truncated(), True, "...but finish_reason=length is recorded as truncated")
assert_eq(acc.usable(), False, "...so usable() is False: finished, but not because it was done")

print()
print("[3] a client disconnect: content, then nothing")
acc = _fresh(_content("She hit stop right abou"))
assert_eq(acc.text(), "She hit stop right abou", "the partial text is retained")
assert_eq(acc.complete(), False, "no finish_reason and no [DONE] -> not complete")
assert_eq(acc.truncated(), False, "...and not truncated either (nobody said 'length')")
assert_eq(acc.usable(), False, "...so not usable")
assert_eq(acc.holed(), False, "...and not holed: every byte that arrived was read")

print()
print("[4] usable() is the AND of complete and not-truncated, nothing more")
assert_eq(_fresh(_content("x"), DONE).usable(), True, "[DONE] without finish_reason is usable")
assert_eq(_fresh(_content("x"), _finish("length")).usable(), False, "length is not")
assert_eq(_fresh(_content("x")).usable(), False, "unfinished is not")

print()
print("[5] a dropped chunk — a malformed event that carried real content")
# v3.1.7 (R7/R14): this used to feed() a bare None to reach the except
# branch — but aiter_raw() can never yield anything but bytes, so that
# path is not the real hazard. The real one, found by C at
# test_sse_accumulator.py:97 in the earlier revision of this file, is a
# content-bearing SSE event that fails to parse: the bytes ARE read (this
# is not the decode-error branch), but its text is gone from text() and
# nothing downstream can tell — the same shape of gap as a decode hole, so
# it sets the same flag (see feed()'s except (json.JSONDecodeError, ...)
# branch).
acc = main.SseAccumulator()
acc.feed(_content("before the hole. "))
acc.feed(b'data: {"choices":[{"delta":{"content":"lost mid-event\n\n')
acc.feed(_content("after the hole."))
acc.feed(_finish("stop"))
acc.feed(DONE)
acc.finalize()
assert_eq(acc.holed(), True, "the drop is recorded")
assert_eq(acc.text(), "before the hole. after the hole.",
          "text() is the concatenation of what survived — with a gap nothing can see")
assert_eq(acc.complete(), True, "the stream still finished cleanly")
assert_eq(acc.usable(), True,
          "and usable() still says so — which is exactly why it cannot be the memory gate")

print()
print("[6] holed is sticky")
acc = main.SseAccumulator()
acc.feed(b'data: {"choices":[{"delta":{"content":"lost\n\n')
acc.finalize()
assert_eq(acc.holed(), True, "set by the drop")
for _ in range(5):
    acc.feed(_content("fine "))
acc.feed(_finish("stop"))
acc.finalize()
assert_eq(acc.holed(), True, "five good chunks and a clean finish do not clear it")

print()
print("[7] undecodable BYTES are not a hole — they are replaced, not dropped")
# errors="replace" means a chunk of invalid UTF-8 still contributes text
# (with U+FFFD), so it is read, not dropped. Pinned so a change to the decode
# call cannot silently widen or narrow what counts as a hole. This is a
# GENUINELY invalid byte (0xFF is not a valid UTF-8 lead byte at all) — a
# different case from [7b] below, where every byte is valid and the only
# problem is which feed() call it landed in.
acc = _fresh(b"data: {\"choices\":[{\"delta\":{\"content\":\"\xff\"}}]}\n\n", _finish("stop"))
assert_eq(acc.holed(), False, "invalid UTF-8 inside a chunk is replaced, not dropped")
assert_true("�" in acc.text(), "and the replacement character is what arrived")

print()
print("[7b] a character SPLIT across feed() calls is reassembled, not corrupted (R7/R14)")
# The actual production hazard: r.aiter_raw() chunk boundaries fall at
# arbitrary byte offsets. Before the incremental decoder, decoding each
# chunk on its own turned a 2-4 byte character split across two reads into
# U+FFFD on BOTH sides of the split, silently — text() disagreed with what
# the client actually received (the raw bytes are forwarded unmodified).
# Measured against real traffic: 9 of 153 split points corrupted this way,
# 0 of 107 ever set holed() under the old per-chunk decode.
event = _content("héllo wörld")
i = event.index("é".encode("utf-8"))
acc = _fresh(event[: i + 1], event[i + 1:], _finish("stop"))
assert_eq(acc.text(), "héllo wörld",
          "a multibyte character split across two feed() calls is reassembled whole")
assert_eq(acc.holed(), False, "a split-but-eventually-complete character is not a hole")
# The split can also fall inside the SECOND multibyte character.
event2 = _content("wörld")
j = event2.index("ö".encode("utf-8")) + 1  # split after the lead byte of ö
acc2 = _fresh(event2[:j], event2[j:], _finish("stop"))
assert_eq(acc2.text(), "wörld", "a split later in the same event is also reassembled")

print()
print("[7c] a stream that ends mid-character IS a hole (R7/R14)")
# Unlike [7b], here the rest of the character never arrives — the
# connection dropped mid-sequence, same as vLLM or the client disconnecting
# mid-byte. finalize() is what tells this apart from [7b]: the incomplete
# bytes are still sitting in the decoder's buffer when the stream ends.
acc = main.SseAccumulator()
event3 = _content("héllo")
k = event3.index("é".encode("utf-8"))
acc.feed(event3[: k + 1])  # cut mid-character; the event never gets its \n\n
acc.finalize()
assert_eq(acc.holed(), True,
          "an incomplete UTF-8 sequence still buffered at finalize() is a hole")

print()
print("All SseAccumulator tests passed.")
