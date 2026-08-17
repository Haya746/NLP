# Practical 3 — N-grams & Skip-grams

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To generate n-grams (unigrams, bigrams, trigrams) and skip-grams from the review corpus, and specifically to check whether bigrams can preserve negation context (e.g. "not good") that Practical 2 showed stopword removal puts at risk.

## Theory

A bag-of-words / unigram representation treats each word independently with no positional information — "not good" and "good not" look identical, and a "not" token removed by stopword filtering vanishes with no trace of what it was negating.

N-grams are contiguous sequences of *n* tokens, used to partially recover local word order:
- Unigrams (n=1): single tokens.
- Bigrams (n=2): adjacent token pairs, e.g. `("not", "good")`.
- Trigrams (n=3): three adjacent tokens.

If bigrams are generated **before** stopword removal, a pair like `("not", "recommend")` stays intact even though "not" alone would normally be stripped — one common mitigation for the negation-loss problem identified in Practical 2.

Trade-off: larger n captures more context, but the number of possible n-grams grows fast, and most specific n-grams appear only once or twice in a small corpus (sparsity).

Skip-grams relax the adjacency requirement, pairing tokens within a gap of up to *k* tokens — useful when a modifier sits between a negation and the word it negates (e.g. "not very good").

## Algorithm

1. Reuse cleaned tokens from Practical 1, and stopword-removed tokens from Practical 2.
2. Generate unigrams/bigrams/trigrams for one sample review.
3. For the negation reviews from Practical 2 (7, 9, 13), generate bigrams before and after stopword removal, and check whether the negation bigram survives.
4. Find the most frequent bigrams across the corpus, before and after stopword removal.
5. Generate skip-bigrams (k=1) and compare against plain bigrams for a case with a modifier in between.

## Code

Full implementation lives in:
- `python/ngrams.py` — `generate_ngrams`, `generate_skipgrams`, `top_ngrams`
- `notebooks/03_Ngrams.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/03_Ngrams.ipynb top to bottom, then paste your actual output
here — the unigram/bigram/trigram example, whether negation bigrams survived
for reviews 7/9/13, the top-10 bigram lists before/after stopword removal,
and the skip-bigram result.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Did bigrams generated before stopword removal actually preserve negation
  context for reviews 7, 9, 13? Give a specific example.
- Did the top-10 bigram list get more or less informative after stopword removal?
- Did skip-bigrams catch ("not", "good") where a plain bigram missed it?
- Bigrams before or after stopword removal — which would you recommend for
  sentiment analysis on this data, and why?
-->

## Viva Questions

1. Why does a bag-of-words / unigram representation struggle with negation?
2. What's the practical trade-off of increasing n in n-grams?
3. How does a skip-gram differ from a regular n-gram?
4. Why generate bigrams before stopword removal rather than after, for a sentiment task?
5. What is the sparsity problem in the context of n-grams, and why does it get worse as n increases?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
