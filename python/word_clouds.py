"""
word_clouds.py

Helpers for building word clouds from token lists we've already
preprocessed ourselves. This module is deliberately NOT named
`wordcloud.py` - that would shadow the `wordcloud` package it imports.

Frequencies are computed here and passed to WordCloud via
generate_from_frequencies(), rather than handing WordCloud raw text.
Given raw text, the library silently does its own tokenizing,
lowercasing, and default-stopword removal, which would hide the very
preprocessing effects this practical is trying to compare.
"""

from collections import Counter

from wordcloud import WordCloud


def word_frequencies(token_lists: list) -> Counter:
    """Count how often each token appears across all documents."""
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    return counter


def make_wordcloud(frequencies: Counter, max_words: int = 100, seed: int = 42) -> WordCloud:
    """Build a WordCloud from a {word: count} mapping. random_state is
    fixed so the same input always produces the same layout.
    """
    cloud = WordCloud(
        width=800,
        height=500,
        background_color="white",
        max_words=max_words,
        random_state=seed,
    )
    return cloud.generate_from_frequencies(dict(frequencies))


def save_wordcloud(cloud: WordCloud, path: str) -> None:
    """Write a word cloud to an image file (e.g. a PNG under images/)."""
    cloud.to_file(path)


def frequency_spread(frequencies: Counter) -> dict:
    """Summarise how flat or skewed a frequency distribution is.

    A word cloud only conveys information through size differences, so
    if most words occur just once, the cloud is nearly uniform no matter
    how it's styled.
    """
    counts = list(frequencies.values())
    unique_words = len(counts)
    appear_once = sum(1 for c in counts if c == 1)
    return {
        "unique_words": unique_words,
        "appear_once": appear_once,
        "share_appearing_once": round(appear_once / unique_words, 3) if unique_words else 0.0,
        "max_count": max(counts) if counts else 0,
    }
