"""
stemming_lemmatization.py

Wraps NLTK's Porter stemmer and WordNet lemmatizer so they can be compared
side by side. Lemmatization accuracy depends heavily on knowing each word's
part of speech (POS) - without it, WordNetLemmatizer assumes every word is
a noun, which silently fails on verbs/adjectives. `get_wordnet_pos` bridges
NLTK's POS tagger output to the tags WordNetLemmatizer expects.
"""

import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def porter_stem(tokens: list) -> list:
    """Apply NLTK's Porter stemmer to each token. Rule-based suffix
    stripping - fast, but can produce forms that aren't real words
    (e.g. "studies" -> "studi"), and doesn't know about irregular forms.
    """
    return [_stemmer.stem(t) for t in tokens]


def get_wordnet_pos(treebank_tag: str) -> str:
    """Map an NLTK/Penn Treebank POS tag (e.g. 'VBD', 'NN', 'JJ') to the
    single-letter POS constant WordNetLemmatizer expects. Defaults to
    noun, since that's WordNetLemmatizer's own default and it's the
    safest fallback for tags this function doesn't recognize.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_tokens(tokens: list, use_pos: bool = True) -> list:
    """Lemmatize a list of tokens.

    Args:
        tokens: word tokens to lemmatize.
        use_pos: if True (default), runs NLTK's POS tagger first and uses
            each word's actual part of speech for accurate lemmatization.
            If False, every token is lemmatized as if it were a noun
            (WordNetLemmatizer's default), which is faster but frequently
            wrong for verbs and adjectives - kept here specifically so the
            notebook can demonstrate that difference.
    """
    if not use_pos:
        return [_lemmatizer.lemmatize(t) for t in tokens]

    tagged = nltk.pos_tag(tokens)
    return [_lemmatizer.lemmatize(t, get_wordnet_pos(tag)) for t, tag in tagged]


def compare_stem_lemma(tokens: list) -> dict:
    """Run stemming and both lemmatization modes on the same tokens for
    side-by-side comparison.
    """
    return {
        "stemmed": porter_stem(tokens),
        "lemmatized_no_pos": lemmatize_tokens(tokens, use_pos=False),
        "lemmatized_with_pos": lemmatize_tokens(tokens, use_pos=True),
    }
