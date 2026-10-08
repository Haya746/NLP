"""
similarity.py

Cosine similarity over TF-IDF vectors, for finding which reviews are most
alike. Unlike Word2Vec (Practical 8) or LDA (Practical 10), this is a
direct geometric computation on vectors that already exist - there's no
statistical model being trained/learned from co-occurrence patterns, so
this technique doesn't fail on a small corpus the same way those did.
Low scores here would reflect genuinely low term overlap, not insufficient
training data.
"""

from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(tfidf_matrix):
    """Compute the full pairwise cosine similarity matrix for a TF-IDF
    matrix - one row/column per document, values in [0, 1] since TF-IDF
    weights are non-negative. The diagonal should be 1.0 (every document
    is maximally similar to itself).
    """
    return cosine_similarity(tfidf_matrix)


def most_similar_documents(query_index: int, similarity_matrix, top_k: int = 3, exclude_self: bool = True) -> list:
    """Return the `top_k` most similar documents to `query_index`, as a
    list of (document_index, score) tuples sorted by score descending.
    """
    scores = similarity_matrix[query_index].copy()
    if exclude_self:
        scores[query_index] = -1  # so it never gets picked as its own "most similar"

    ranked_indices = scores.argsort()[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked_indices]


def similarity_between(doc_a_index: int, doc_b_index: int, similarity_matrix) -> float:
    """Return the cosine similarity between two specific documents."""
    return float(similarity_matrix[doc_a_index, doc_b_index])


def average_pairwise_similarity(similarity_matrix) -> float:
    """Return the average similarity across all distinct document pairs
    (excluding the diagonal self-similarities), as a baseline for what
    counts as "actually similar" vs. typical background similarity for
    this corpus.
    """
    n = similarity_matrix.shape[0]
    total = similarity_matrix.sum() - n  # subtract the n diagonal 1.0s
    count = n * n - n
    return float(total / count)
