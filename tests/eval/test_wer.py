"""
Tier-1 unit tests for the WER metric (tests/eval/wer.py).

Pure stdlib — no audio, no model, no network. Run: python test_wer.py
"""
import sys

import wer as werlib


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_close(a, b, label, tol=1e-9):
    if abs(a - b) > tol:
        print(f"FAIL {label}: expected {b}, got {a}")
        sys.exit(1)
    print(f"  ok   {label}")


def test_normalize_and_tokenize():
    print("\n[test] normalize / tokenize")
    assert_eq(werlib.normalize("Hello, World!"), "hello world", "punct + case folded")
    assert_eq(werlib.tokenize("  it's   fine. "), ["it's", "fine"], "apostrophe kept, split")
    assert_eq(werlib.tokenize(""), [], "empty -> no tokens")


def test_perfect_match():
    print("\n[test] identical text -> WER 0.0")
    assert_close(werlib.wer("the quick brown fox", "the quick brown fox"), 0.0, "exact")
    assert_close(werlib.wer("The Quick, Brown Fox!", "the quick brown fox"), 0.0,
                 "case/punct ignored")


def test_substitution():
    print("\n[test] one substitution in 4 words -> 0.25")
    assert_close(werlib.wer("the quick brown fox", "the quick red fox"), 0.25, "1 sub / 4")


def test_deletion_and_insertion():
    print("\n[test] deletion and insertion each count as one error")
    assert_close(werlib.wer("the quick brown fox", "the quick fox"), 0.25, "1 deletion / 4")
    assert_close(werlib.wer("the quick brown fox", "the quick brown red fox"), 0.25,
                 "1 insertion / 4")


def test_all_wrong():
    print("\n[test] completely different -> 1.0")
    assert_close(werlib.wer("alpha beta", "gamma delta"), 1.0, "2 subs / 2")


def test_empty_edges():
    print("\n[test] empty-reference edge cases")
    assert_close(werlib.wer("", ""), 0.0, "empty/empty -> 0.0")
    assert_close(werlib.wer("", "spurious words"), 1.0, "empty ref + hyp -> 1.0")
    assert_close(werlib.wer("hello world", ""), 1.0, "ref + empty hyp -> all deletions")


def test_edit_distance_direct():
    print("\n[test] word_edit_distance basics")
    assert_eq(werlib.word_edit_distance(["a", "b", "c"], ["a", "b", "c"]), 0, "identical")
    assert_eq(werlib.word_edit_distance(["a", "b"], []), 2, "all deletions")
    assert_eq(werlib.word_edit_distance([], ["a", "b", "c"]), 3, "all insertions")


def _all():
    return [
        test_normalize_and_tokenize,
        test_perfect_match,
        test_substitution,
        test_deletion_and_insertion,
        test_all_wrong,
        test_empty_edges,
        test_edit_distance_direct,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll WER unit tests passed.")
