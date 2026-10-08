# Practical 12 — Text Similarity & Cosine Similarity

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To compute pairwise cosine similarity between reviews using TF-IDF vectors, find which reviews are most similar to a given query review, and test whether thematically related reviews actually score as more similar to each other.

## Theory

Cosine similarity measures how alike two vectors are by the angle between them, not their magnitude, giving a score from 0 to 1 for non-negative vectors like TF-IDF weights. This matters because two documents of very different lengths can still be considered similar if they use words in similar proportions.

Worth noting given the last two practicals: this is a genuinely different kind of technique than Word2Vec (Practical 8) or LDA (Practical 10). Those are statistical models that need to learn patterns from co-occurrence across many documents — which is why they broke down on our 15-review corpus. Cosine similarity is a direct geometric computation on vectors that already exist — nothing is being learned or estimated from limited data. So low similarity scores here aren't necessarily a small-data failure; they may honestly reflect that these reviews don't share much vocabulary.

## Algorithm

1. Build a TF-IDF matrix and compute the full pairwise cosine similarity matrix.
2. Confirm the diagonal (self-similarity) is 1.0.
3. Query review 4 (acting-focused) and find its most similar other reviews.
4. Query review 10 (price/value-focused) and do the same.
5. Compute the average pairwise similarity across the whole corpus as a baseline.

## Code

Full implementation lives in:
- `python/similarity.py` — `build_similarity_matrix`, `most_similar_documents`, `similarity_between`, `average_pairwise_similarity`
- `notebooks/12_Text_Similarity.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/12_Text_Similarity.ipynb top to bottom, then paste your
actual output here — the diagonal check, both queries' top-3 matches with
scores, and the average pairwise similarity baseline.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Did review 4's top matches actually relate to acting, or were they just
  least-dissimilar options? Same question for review 10.
- How do the top-match scores compare to the average baseline?
- Is low similarity here a small-data problem like Practicals 8/10, or a
  legitimate result given real vocabulary differences between short,
  topically varied reviews?
-->

## Viva Questions

1. Why is cosine similarity preferred over a raw distance metric (like Euclidean distance) for comparing TF-IDF vectors?
2. What should the diagonal of a cosine similarity matrix always be, and why?
3. Why doesn't cosine similarity on TF-IDF vectors suffer from the same small-data problem as Word2Vec or LDA?
4. If two documents share zero vocabulary, what will their cosine similarity be, and why?
5. Why is it important to compare a "top match" score against a corpus-wide average baseline, rather than just trusting the ranking?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
