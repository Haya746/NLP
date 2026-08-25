"""
embeddings.py

Word2Vec (custom-trained on our own tiny corpus) vs pretrained GloVe
vectors (trained on billions of words) vs FastText (subword-aware,
custom-trained). The point of comparing all three is data scale and
out-of-vocabulary (OOV) handling, not picking a "winner."
"""

import gensim.downloader as gensim_api
from gensim.models import FastText, Word2Vec


def train_word2vec(token_lists: list, vector_size: int = 50, window: int = 5,
                    min_count: int = 1, sg: int = 1) -> Word2Vec:
    """Train a Word2Vec model directly on our own corpus.

    min_count=1 is deliberately permissive (default gensim behaviour
    requires min_count=5, which would discard almost every word in a
    15-document corpus). sg=1 selects Skip-gram; sg=0 would be CBOW.
    """
    return Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
    )


def train_fasttext(token_lists: list, vector_size: int = 50, window: int = 5,
                    min_count: int = 1) -> FastText:
    """Train a FastText model on our own corpus. Unlike Word2Vec, FastText
    represents each word as a bag of character n-grams, so it can produce
    a (rough) vector even for a word it never saw during training, by
    combining the n-gram vectors it does know.
    """
    return FastText(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
    )


def load_pretrained_glove(name: str = "glove-wiki-gigaword-50"):
    """Download (and locally cache under ~/gensim-data) a pretrained GloVe
    model via gensim's downloader API. Requires an internet connection the
    first time it's run for a given model name; cached after that.
    """
    return gensim_api.load(name)


def safe_most_similar(model, word: str, topn: int = 5):
    """Call most_similar() on a gensim model/KeyedVectors, but don't crash
    on an out-of-vocabulary word - return a clear marker instead so OOV
    behaviour can be inspected directly rather than causing a traceback.
    """
    try:
        return model.wv.most_similar(word, topn=topn)
    except AttributeError:
        # KeyedVectors objects (e.g. loaded GloVe) don't have a .wv wrapper -
        # they ARE the vectors object directly.
        try:
            return model.most_similar(word, topn=topn)
        except KeyError:
            return f"OOV: '{word}' not in vocabulary"
    except KeyError:
        return f"OOV: '{word}' not in vocabulary"
