"""
vectorizers.py

Wraps scikit-learn's CountVectorizer (Bag of Words), TfidfVectorizer, and
HashingVectorizer so the three can be compared on the same corpus. Unlike
Practicals 5-6, these are generally fine to run on Practical 1's cleaned,
lowercased text - word order/capitalization don't matter for a bag-of-words
style representation the way they mattered for POS tagging and NER.
"""

from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)


def bow_vectorize(corpus: list):
    """Fit a CountVectorizer (Bag of Words) on `corpus` (a list of raw
    document strings). Returns (matrix, feature_names).
    """
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer.get_feature_names_out()


def tfidf_vectorize(corpus: list):
    """Fit a TfidfVectorizer on `corpus`. Returns (matrix, feature_names)."""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer.get_feature_names_out()


def top_tfidf_terms(tfidf_matrix, feature_names, doc_index: int, top_k: int = 10) -> list:
    """Return the top-`top_k` (term, score) pairs for one document (row)
    in a fitted TF-IDF matrix, sorted by score descending. Terms with a
    score of 0 (not present in this document) are excluded.
    """
    row = tfidf_matrix[doc_index].toarray().flatten()
    ranked_indices = row.argsort()[::-1]
    results = []
    for idx in ranked_indices:
        if row[idx] <= 0:
            break
        results.append((feature_names[idx], round(float(row[idx]), 4)))
        if len(results) >= top_k:
            break
    return results


def hashing_vectorize(corpus: list, n_features: int = 20):
    """Fit a HashingVectorizer on `corpus` with a given feature-space size.
    Returns just the matrix - HashingVectorizer has no vocabulary_ and no
    get_feature_names_out(), since tokens are hashed directly to indices
    with no stored mapping back to the original words.
    """
    vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
    return vectorizer.fit_transform(corpus)
