"""
ngrams.py

Functions for generating n-grams and skip-grams from a token list, and for
finding the most frequent n-grams across a corpus. Pure-Python
implementations (no NLTK dependency) so the sliding-window logic is
transparent rather than hidden inside a library call.
"""

from collections import Counter


def generate_ngrams(tokens: list, n: int) -> list:
    """Return all contiguous n-grams from `tokens` as a list of tuples.

    E.g. generate_ngrams(["a", "b", "c"], 2) -> [("a", "b"), ("b", "c")]
    """
    if n <= 0 or n > len(tokens):
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def generate_skipgrams(tokens: list, n: int = 2, k: int = 1) -> list:
    """Return skip-grams: n-length tuples where up to `k` tokens may be
    skipped between each chosen token, rather than requiring contiguity.

    Only n=2 (skip-bigrams) is implemented here, since that's the
    common case (and the easiest to reason about on paper).

    E.g. generate_skipgrams(["not", "very", "good"], n=2, k=1) includes
    both ("not", "very") [adjacent] and ("not", "good") [one word skipped].
    """
    if n != 2:
        raise NotImplementedError("Only skip-bigrams (n=2) are implemented.")

    pairs = []
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 2 + k, len(tokens))):
            pairs.append((tokens[i], tokens[j]))
    return pairs


def top_ngrams(token_lists: list, n: int, top_k: int = 10) -> list:
    """Count n-gram frequency across many token lists (e.g. one per
    document) and return the `top_k` most common as (ngram, count) pairs.
    """
    counter = Counter()
    for tokens in token_lists:
        counter.update(generate_ngrams(tokens, n))
    return counter.most_common(top_k)
