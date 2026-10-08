"""
spellcheck.py

A from-scratch spelling corrector based on edit distance, in the style of
Peter Norvig's well-known "How to Write a Spelling Corrector" essay.

Idea: given a word that isn't in our vocabulary, generate every string
that is one edit away (delete, transpose, replace, or insert one letter),
keep only the ones that ARE real words, and pick the most frequent. If
nothing is one edit away, try two edits.

Two limits are built into this approach, and the practical tests both:
- It only checks whether a word EXISTS in the vocabulary. A typo that
  happens to spell a different real word ("sea" for "see") is invisible.
- It looks at one word at a time, with no sentence context.
"""

import random
from collections import Counter

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def build_word_counts(words, min_count: int = 1) -> Counter:
    """Count word frequencies from an iterable of words, keeping only
    lowercase alphabetic words that occur at least `min_count` times.

    Raising min_count drops rare words, which matters because a large
    real-world corpus contains typos of its own - a misspelling that
    appears once or twice would otherwise count as a "real" word.
    """
    counts = Counter(w.lower() for w in words if w.isalpha())
    if min_count > 1:
        counts = Counter({w: c for w, c in counts.items() if c >= min_count})
    return counts


def edits1(word: str) -> set:
    """All strings exactly one edit away from `word`: deletions,
    transpositions of adjacent letters, single-letter replacements,
    and single-letter insertions.
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [left + right[1:] for left, right in splits if right]
    transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
    replaces = [left + c + right[1:] for left, right in splits if right for c in LETTERS]
    inserts = [left + c + right for left, right in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


def edits2(word: str) -> set:
    """All strings two edits away (an edit of an edit)."""
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


def known(candidates, counts: Counter) -> set:
    """Filter `candidates` down to those that are real words in `counts`."""
    return {w for w in candidates if w in counts}


def correct(word: str, counts: Counter, max_distance: int = 2) -> str:
    """Return the most likely correction for `word`.

    Priority: the word itself if it's already known, then known words one
    edit away, then (if max_distance allows) known words two edits away.
    Among equally close candidates, the most frequent word wins. If no
    candidate exists, the word is returned unchanged.
    """
    if word in counts:
        return word

    candidates = known(edits1(word), counts)
    if not candidates and max_distance >= 2:
        candidates = known(edits2(word), counts)

    if not candidates:
        return word
    # Ties on frequency are broken alphabetically so results are reproducible
    # (set iteration order otherwise varies between Python runs).
    return max(candidates, key=lambda w: (counts[w], w))


def find_unknown_words(token_lists, counts: Counter) -> Counter:
    """Count tokens (across many documents) that are NOT in the vocabulary -
    i.e. the words a spell checker would flag.
    """
    unknown = Counter()
    for tokens in token_lists:
        unknown.update(t for t in tokens if t not in counts)
    return unknown


def introduce_typo(word: str, rng: random.Random) -> str:
    """Apply one random single-letter edit (delete, transpose, replace, or
    insert) to `word`, to create a synthetic misspelling with a known
    correct answer.
    """
    kind = rng.choice(["delete", "transpose", "replace", "insert"])
    i = rng.randrange(len(word))
    if kind == "delete":
        return word[:i] + word[i + 1:]
    if kind == "transpose" and len(word) > 1:
        i = rng.randrange(len(word) - 1)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "replace":
        return word[:i] + rng.choice(LETTERS) + word[i + 1:]
    return word[:i] + rng.choice(LETTERS) + word[i:]


def evaluate_corrector(words, counts: Counter, max_distance: int = 2,
                       n_edits: int = 1, seed: int = 42) -> dict:
    """Corrupt each word with `n_edits` random single-letter typos, run the
    corrector, and report how often it recovers the original word.
    Typos that happen to produce another real word are skipped, since they
    would not be detectable as errors at all (see the module docstring).

    n_edits=1 makes typos exactly one edit from the original; n_edits=2
    makes them two edits away, which a max_distance=1 corrector cannot reach.
    """
    rng = random.Random(seed)
    tested = recovered = 0
    for word in words:
        typo = word
        for _ in range(n_edits):
            typo = introduce_typo(typo, rng)
        if typo == word or typo in counts:
            continue
        tested += 1
        if correct(typo, counts, max_distance=max_distance) == word:
            recovered += 1
    return {
        "tested": tested,
        "recovered": recovered,
        "accuracy": round(recovered / tested, 3) if tested else 0.0,
    }
