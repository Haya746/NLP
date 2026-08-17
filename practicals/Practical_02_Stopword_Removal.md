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

- Review 7: found ['not'] in tokens -> ['two', 'hours', 'and', 'mins', 'of', 'pure', 'boredom', 'would', 'not', 'recommend', 'to', 'anyone']
- Review 8: found ['no'] in tokens -> ['wow', 'best', 'film', 'ive', 'seen', 'in', 'years', 'hands', 'down', 'no', 'notes']
- Review 9: found ['not', 'not'] in tokens -> ['its', 'okay', 'not', 'great', 'not', 'terrible', 'somewhere', 'in', 'the', 'middle', 'id', 'say']
- Review 13: found ['not'] in tokens -> ['dont', 'get', 'me', 'wrong', 'its', 'not', 'bad', 'but', 'i', 'expected', 'way', 'more', 'given', 'the', 'hype']
- Raw vocabulary (Practical 1 baseline):  136
- After NLTK stopword removal: 102
- After NLTK + custom domain stopword removal:  94

## Conclusion

Step 3 showed that literal negation words like "not" and "no" appear intact in the tokens (e.g. review 7: "would not recommend"; review 9: "not great, not terrible"), and both are confirmed present in NLTK's stopword list — meaning stopword removal would strip them outright and could flip a review's apparent sentiment. Separately, review 13 also contained "dont" (from "Don't get me wrong"), but this form wasn't flagged at all, since apostrophe removal in Practical 1's cleaning step turned it into a form that no longer matches stopword-list entries like "don". This shows two distinct risks: stopword removal can directly delete real negation words, and preprocessing order can also cause negation words to silently escape detection in a mangled form. Vocabulary size dropped from 136 (raw) to 102 after NLTK stopword removal, and to 94 after adding the custom domain list — a modest additional cut compared to NLTK's list alone. For sentiment analysis specifically, stopword removal reduces noise but is genuinely risky here, given that real negation words were confirmed present and removable — it would need to be applied carefully, likely by excluding negation words from the stopword list before removing the rest.

## Viva Questions

1. What is a stopword, and why are they usually removed before text classification?
2. Why might a general-purpose stopword list be a bad fit for a specific domain?
3. Why can stopword removal be risky for sentiment analysis specifically?
4. Why does the order of preprocessing steps matter here?
5. What's the tradeoff of adding a custom domain stopword list on top of a general one?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
