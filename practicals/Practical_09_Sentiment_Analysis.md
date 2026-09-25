# Practical 9 — Sentiment Analysis (VADER & TextBlob)

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To score review sentiment using VADER and TextBlob, and to directly test — using the actual functions built in Practicals 1 and 2 — whether preprocessing (cleaning, stopword removal) helps or hurts sentiment scoring accuracy.

## Theory

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon + rule-based sentiment tool built for informal, social-media-style text. It produces a `compound` score from -1 to +1, plus neg/neu/pos proportions. VADER's rules explicitly account for punctuation emphasis ("great!!!"), capitalization emphasis ("GREAT"), negation ("not good"), and degree modifiers ("very good").

TextBlob takes a simpler pattern-based approach: a lexicon of pre-scored adjectives with simpler rules, producing `polarity` (-1 to +1) and `subjectivity` (0 to 1).

This practical directly tests whether VADER's rules — which depend on punctuation, capitalization, and intact negation words — actually get disrupted by Practical 1's `clean_text()` and Practical 2's stopword removal. This closes the loop opened in Practical 2, where negation loss was only a hypothesis about a downstream task.

## Algorithm

1. Predict, before running anything, whether cleaning will meaningfully change VADER's scores.
2. Score two punctuation/caps-heavy reviews with VADER, raw vs cleaned.
3. Score review 7 ("...would not recommend...") three ways: raw, cleaned, and stopword-removed.
4. Score all 15 raw reviews with both VADER and TextBlob and compare agreement.

## Code

Full implementation lives in:
- `python/sentiment.py` — `vader_score`, `textblob_score`, `classify_compound`, `compare_vader_textblob`
- `notebooks/09_Sentiment_Analysis.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/09_Sentiment_Analysis.ipynb top to bottom, then paste your
actual output here — your prediction, the raw-vs-cleaned scores for
reviews 1 and 8, the three-way score for review 7, and the VADER/TextBlob
agreement rate plus disagreements.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Was your prediction correct? Did cleaning change VADER's scores for
  reviews 1 and 8?
- Did removing "not" from review 7 actually change its VADER score
  meaningfully? Give the three real numbers.
- What was the VADER/TextBlob agreement rate, and do disagreements happen
  on genuinely ambiguous reviews?
- Overall recommendation: how much preprocessing should a sentiment
  pipeline actually do, given everything from Practicals 1-9?
-->

## Viva Questions

1. What makes VADER specifically well-suited to informal/social-media text, compared to a general sentiment lexicon?
2. Why would stopword removal specifically risk changing a VADER sentiment score, when VADER wasn't part of any earlier practical?
3. What's the practical difference between VADER's compound score and TextBlob's polarity score?
4. If two sentiment tools disagree on a review, does that necessarily mean one of them is "wrong"?
5. Based on everything from Practicals 1-9, should a sentiment-analysis pipeline use the same cleaned text as a Bag-of-Words/TF-IDF pipeline?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
