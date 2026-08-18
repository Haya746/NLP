# Practical 3 — N-grams & Skip-grams

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim

To generate n-grams (unigrams, bigrams, trigrams) and skip-grams from the review corpus, and specifically to check whether bigrams can preserve negation context (e.g. "not good") that Practical 2 showed stopword removal puts at risk.

## Theory

A bag-of-words / unigram representation treats each word independently with no positional information — "not good" and "good not" look identical, and a "not" token removed by stopword filtering vanishes with no trace of what it was negating.

N-grams are contiguous sequences of _n_ tokens, used to partially recover local word order:

- Unigrams (n=1): single tokens.
- Bigrams (n=2): adjacent token pairs, e.g. `("not", "good")`.
- Trigrams (n=3): three adjacent tokens.

If bigrams are generated **before** stopword removal, a pair like `("not", "recommend")` stays intact even though "not" alone would normally be stripped — one common mitigation for the negation-loss problem identified in Practical 2.

Trade-off: larger n captures more context, but the number of possible n-grams grows fast, and most specific n-grams appear only once or twice in a small corpus (sparsity).

Skip-grams relax the adjacency requirement, pairing tokens within a gap of up to _k_ tokens — useful when a modifier sits between a negation and the word it negates (e.g. "not very good").

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

Test tokens: ['it', 'was', 'not', 'very', 'good']
Plain bigrams: [('it', 'was'), ('was', 'not'), ('not', 'very'), ('very', 'good')]
Skip-bigrams (k=1): [('it', 'was'), ('it', 'not'), ('was', 'not'), ('was', 'very'), ('not', 'very'), ('not', 'good'), ('very', 'good')]
Does a plain bigram directly pair "not" and "good"? False
Does a skip-bigram directly pair "not" and "good"? True
TOP BIGRAMS (before stopword removal):
('like', 'it'): 3
('it', 'was'): 2
('of', 'the'): 2
('this', 'movie'): 1
('movie', 'was'): 1
('was', 'absolutely'): 1
('absolutely', 'fantastic'): 1
('fantastic', 'ive'): 1
('ive', 'never'): 1
('never', 'seen'): 1
TOP BIGRAMS (after stopword removal):
('movie', 'absolutely'): 1
('absolutely', 'fantastic'): 1
('fantastic', 'ive'): 1
('ive', 'never'): 1
('never', 'seen'): 1
('seen', 'anything'): 1
('anything', 'like'): 1
('worst', 'film'): 1
('film', 'dont'): 1
('dont', 'waste'): 1Review 7: Two hours and 15 mins of pure boredom. 2/10 would not recommend to anyone.
Negation bigrams BEFORE stopword removal: [('would', 'not'), ('not', 'recommend')]
Negation bigrams AFTER stopword removal: []
Review 9: It's okay... not great, not terrible. Somewhere in the middle I'd say.
Negation bigrams BEFORE stopword removal: [('okay', 'not'), ('not', 'great'), ('great', 'not'), ('not', 'terrible')]
Negation bigrams AFTER stopword removal: []
Review 13: Don't get me wrong, it's not BAD, but I expected way more given the hype.
Negation bigrams BEFORE stopword removal: [('its', 'not'), ('not', 'bad')]
Negation bigrams AFTER stopword removal: []

## Conclusion

Skip-bigrams proved more effective than plain bigrams at capturing relationships separated by an intermediate word: for "it was not very good," a plain bigram could not directly pair "not" and "good" (False), while a skip-bigram (k=1) did (True) — making skip-bigrams better suited to catching modified negation phrases. Comparing the top-10 bigram lists, stopword removal eliminated uninformative pairs like ("it", "was") and ("of", "the"), but also produced an unintended artifact: ("film", "dont") appeared in the "after" list only because removing the stopword "of" from between them created a new adjacency that didn't exist in the original text. Most importantly, all three negation bigrams found intact before stopword removal — ("would", "not")+("not", "recommend") in review 7, ("not", "great")+("not", "terrible") in review 9, and ("not", "bad") in review 13 — were completely eliminated after stopword removal, confirming that generating bigrams before removing stopwords is what actually preserves that context. This shows indiscriminate stopword removal can destroy sentiment-critical information, and that for this dataset, negation words should either be excluded from the stopword list or bigrams should be generated before stopword removal, not after.

## Viva Questions

1. Why does a bag-of-words / unigram representation struggle with negation?
2. What's the practical trade-off of increasing n in n-grams?
3. How does a skip-gram differ from a regular n-gram?
4. Why generate bigrams before stopword removal rather than after, for a sentiment task?
5. What is the sparsity problem in the context of n-grams, and why does it get worse as n increases?

_(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)_
