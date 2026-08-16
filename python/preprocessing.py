"""
preprocessing.py

Reusable text-cleaning functions for the NLP practicals/notebooks in this repo.
Each function does ONE job so they can be composed and tested independently
(and so it's obvious in the notebook which step is doing what).
"""

import re


def to_lowercase(text: str) -> str:
    """Convert all characters in `text` to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Strip standard punctuation characters from `text`.

    Uses a regex character class rather than str.translate so it's easy to
    read/adjust which punctuation is kept vs removed.
    """
    return re.sub(r"[^\w\s]", "", text)


def remove_numbers(text: str) -> str:
    """Remove standalone digits from `text` (e.g. review scores like '7/10')."""
    return re.sub(r"\d+", "", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_characters(text: str) -> str:
    """Remove characters that aren't letters, numbers, or whitespace
    (e.g. $, %, --, !! survive punctuation removal in some pipelines,
    this catches the rest).
    """
    return re.sub(r"[^A-Za-z0-9\s]", "", text)


def clean_text(text: str, keep_numbers: bool = False) -> str:
    """Full preprocessing pipeline: lowercase -> remove punctuation/specials
    -> optionally remove numbers -> collapse whitespace.

    Args:
        text: raw input string.
        keep_numbers: if False (default), digits are stripped too.

    Returns:
        Cleaned string ready for tokenization.
    """
    text = to_lowercase(text)
    text = remove_special_characters(text)
    if not keep_numbers:
        text = remove_numbers(text)
    text = remove_extra_whitespace(text)
    return text


def get_nltk_stopwords() -> set:
    """Return NLTK's built-in English stopword list as a set.

    Requires nltk.download('stopwords') to have been run at least once
    (see the notebook's setup cell).
    """
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


def load_custom_stopwords(path: str) -> set:
    """Load a custom stopword list from a text file (one word per line).

    Used to layer domain-specific filler words (e.g. "movie", "watch" for
    a movie-review corpus) on top of, or instead of, NLTK's general list.
    """
    with open(path) as f:
        return {line.strip().lower() for line in f if line.strip()}


def remove_stopwords(tokens: list, stopword_set: set) -> list:
    """Filter a list of tokens, dropping any token present in `stopword_set`.

    Args:
        tokens: list of word tokens (already lowercased, ideally).
        stopword_set: set of words to remove.

    Returns:
        New list with stopwords removed, order preserved.
    """
    return [t for t in tokens if t not in stopword_set]
