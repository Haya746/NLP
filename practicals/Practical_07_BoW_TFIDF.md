# Practical 7 — Bag of Words, TF-IDF & HashingVectorizer

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To represent the review corpus as numerical feature vectors using CountVectorizer (Bag of Words), TfidfVectorizer, and HashingVectorizer, and to test whether Practical 1's cleaning pipeline — which broke POS tagging and NER — is actually appropriate for this kind of representation.

## Theory

Bag of Words (BoW) represents each document as a vector of word counts, with word order discarded entirely — "not good" and "good not" produce identical vectors. `CountVectorizer` builds this automatically, learning a vocabulary from the corpus and producing one row per document, one column per vocabulary word.

TF-IDF improves on raw counts by weighting each term by how distinctive it is: a word appearing in every document gets down-weighted even without an explicit stopword list, because IDF naturally penalizes terms that show up everywhere. A word appearing often in one document but rarely elsewhere gets a high score.

HashingVectorizer skips building an explicit vocabulary — it hashes each token directly to a fixed-size feature index. Memory-efficient for very large corpora, but there's no way to map a feature index back to the word that produced it, and with a small feature space, two different words can collide into the same index.

Practicals 5 and 6 showed Practical 1's aggressive cleaning actively broke POS tagging and NER, since those tasks depend on the exact surface form of text. This practical checks whether that same cleaning is actually fine — or helpful — here instead.

## Algorithm

1. Build two versions of the corpus: raw text, and cleaned + stopword-removed text.
2. Vectorize both with CountVectorizer and compare vocabulary size.
3. Vectorize both with TfidfVectorizer and extract top terms for review 4 from each.
4. Compare whether stopword removal actually changed the top-ranked TF-IDF terms.
5. Run HashingVectorizer with a small feature space (checking for collisions) and a larger one.

## Code

Full implementation lives in:
- `python/vectorizers.py` — `bow_vectorize`, `tfidf_vectorize`, `top_tfidf_terms`, `hashing_vectorize`
- `notebooks/07_BoW_TFIDF.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/07_BoW_TFIDF.ipynb top to bottom, then paste your actual
output here — the two vocab sizes, the raw-vs-cleaned top TF-IDF terms for
review 4, and the HashingVectorizer shapes/attribute checks.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Did sklearn's CountVectorizer vocab size on cleaned text match Practical
  2's number (94)? If not, why might they differ?
- Did the top TF-IDF terms for review 4 actually change between raw and
  stopword-removed versions, or was TF-IDF's own weighting doing similar
  work already?
- What did the HashingVectorizer shapes and hasattr checks confirm?
- Pulling this together with Practicals 5-6: does cleaning help or hurt,
  and does that depend entirely on the downstream task? Give one example
  of each.
-->

## Viva Questions

1. What information does Bag of Words permanently discard, and why might that matter?
2. How does TF-IDF's weighting differ from a plain word count?
3. What's the main trade-off of using HashingVectorizer instead of CountVectorizer?
4. Why might cleaning/lowercasing text be appropriate for Bag of Words / TF-IDF but not for POS tagging or NER?
5. Why might explicit stopword removal have less effect on TF-IDF results than on a raw Bag of Words count?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
