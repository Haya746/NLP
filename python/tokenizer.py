"""
tokenizer.py

Wraps three different tokenization approaches so the notebook can compare
them side by side on the same input: NLTK, a plain regex tokenizer, and
spaCy. Requires `nltk` and `spacy` to be installed (see requirements.txt)
and their respective data/models downloaded (see README installation steps).
"""

import re

import nltk
import spacy

# Loaded once at import time and reused, rather than reloading per call.
_nlp = spacy.load("en_core_web_sm")


def nltk_sentence_tokenize(text: str) -> list[str]:
    """Split `text` into sentences using NLTK's Punkt tokenizer."""
    return nltk.sent_tokenize(text)


def nltk_word_tokenize(text: str) -> list[str]:
    """Split `text` into word tokens using NLTK's word tokenizer."""
    return nltk.word_tokenize(text)


def regex_word_tokenize(text: str) -> list[str]:
    """Split `text` into word tokens using a simple regex: sequences of
    alphanumeric characters. Deliberately naive - no handling of
    contractions, hyphenated words, etc. - so its output can be
    contrasted with NLTK/spaCy in the notebook.
    """
    return re.findall(r"\b\w+\b", text)


def spacy_word_tokenize(text: str) -> list[str]:
    """Split `text` into word tokens using spaCy's tokenizer (rule + model
    based, handles contractions like "don't" -> "do", "n't" differently
    from NLTK).
    """
    doc = _nlp(text)
    return [token.text for token in doc]


def compare_tokenizers(text: str) -> dict:
    """Run all three tokenizers on the same input and return their outputs
    together, for side-by-side comparison in the notebook.
    """
    return {
        "regex": regex_word_tokenize(text),
        "nltk": nltk_word_tokenize(text),
        "spacy": spacy_word_tokenize(text),
    }
