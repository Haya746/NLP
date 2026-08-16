# Practical 2 — Stopword Removal

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To remove stopwords from the cleaned/tokenized review corpus using NLTK's default English stopword list, and to compare that against a custom domain-specific stopword list — including checking what stopword removal does to negation words.

## Theory

Stopwords are high-frequency words (articles, prepositions, common pronouns/verbs — "the", "is", "at", "which") that appear in almost every document and typically carry little topic-specific meaning on their own. Removing them before tasks like classification or topic modelling reduces vocabulary size and lets a model focus on words that actually discriminate between documents.

Two caveats matter in practice:
- **Stopword lists are task-dependent.** A general list isn't tuned to any particular domain. A movie-review corpus might have its own high-frequency, low-signal words ("movie", "film", "watch") that a general list wouldn't flag.
- **Stopword removal can delete meaningful signal**, especially for sentiment analysis. Standard lists include negation words ("not", "no", "nor") and contraction fragments ("wasn", "wouldn", "shouldn"). Removing "not" from "not good" leaves "good" — the opposite meaning. This connects directly to Practical 1: how a contraction was tokenized upstream affects whether stopword removal even recognizes it downstream.

## Algorithm

1. Reuse the cleaned + tokenized reviews from Practical 1.
2. Load NLTK's default English stopword list and check its size.
3. Remove default stopwords from the tokens; compare before/after.
4. Inspect what happened specifically to negation-related words.
5. Load a custom domain stopword list and remove those too.
6. Recompute vocabulary size at each stage and compare to Practical 1's baseline.

## Code

Full implementation lives in:
- `python/preprocessing.py` — extended with `get_nltk_stopwords`, `load_custom_stopwords`, `remove_stopwords`
- `datasets/stopwords_custom.txt` — domain-specific stopword list for this review corpus
- `notebooks/02_Stopword_Removal.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/02_Stopword_Removal.ipynb top to bottom, then paste your actual
output here — the NLTK stopword list size, before/after examples, what you
found when checking for negation words, and the three vocabulary size numbers.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Did negation words actually get removed, or had Practical 1's cleaning
  already changed their form so they didn't match the stopword list?
- How much did vocabulary size shrink at each stage?
- Was stopword removal net helpful or net risky for this dataset if the
  eventual goal were sentiment analysis?
-->

## Viva Questions

1. What is a stopword, and why are they usually removed before text classification?
2. Why might a general-purpose stopword list be a bad fit for a specific domain?
3. Why can stopword removal be risky for sentiment analysis specifically?
4. Why does the order of preprocessing steps matter here?
5. What's the tradeoff of adding a custom domain stopword list on top of a general one?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
