# Practical 14 — Spell Checking

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To build an edit-distance spelling corrector from scratch, measure how accurate it is on synthetic typos, and test what happens when it is run on the review corpus after Practical 1's cleaning — including what its "corrections" do to words that carry meaning.

## Theory

A spell checker has two jobs: detect a word that is probably wrong, then suggest the word the writer most likely meant.

This practical builds the classic edit-distance approach:
- **Detect:** a word is flagged if it is not in a vocabulary of known words.
- **Generate candidates:** every string one edit away from the flagged word, where an edit is a deletion, transposition (swap of two adjacent letters), replacement, or insertion of one letter. If nothing one edit away is a real word, try two edits.
- **Choose:** among the real-word candidates, pick the most frequent one in a large reference corpus, as a stand-in for "which word did the writer most likely mean".

The vocabulary and frequencies come from NLTK's `movie_reviews` corpus (about 1.6 million words). A `min_count` setting drops rare words, because a large real-world corpus contains typos of its own.

Two limits are built into this approach: it only asks whether a word exists (so a typo that spells a different real word is invisible), and it judges one word at a time with no sentence context.

There is also a link back to the earlier practicals: Practical 1's `clean_text()` strips apostrophes, so "can't" becomes "cant" before any later step sees it, and a spell checker cannot know "cant" was once a contraction.

## Algorithm

1. Build word-frequency tables from the movie review corpus, with and without a minimum-count filter.
2. Correct a sentence containing genuine typos.
3. Run the spell checker over the cleaned review tokens, list every flagged word, and inspect each proposed correction.
4. Test a sentence made of valid words used wrongly, and see whether anything is flagged.
5. Measure accuracy on synthetic typos (one edit vs two edits away, correcting up to distance 1 vs 2) and note the time taken.
6. Repeat the accuracy test across five random seeds and look at the spread.

## Code

Full implementation lives in:
- `python/spellcheck.py` — `build_word_counts`, `edits1`, `edits2`, `known`, `correct`, `find_unknown_words`, `introduce_typo`, `evaluate_corrector`
- `notebooks/14_Spell_Checking.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/14_Spell_Checking.ipynb top to bottom, then paste your actual
output here — the vocabulary sizes, the typo-sentence corrections, the
flagged words and proposed corrections for both min_count settings, the
real-word-error test, the four accuracy lines with timings, and the
five-seed spread.
-->

## Conclusion

<!--
Write 5-7 sentences in your own words, based on what you actually observed:
- Did the corrector fix the genuine typos correctly?
- Which cleaned tokens were flagged, why, and did the proposed corrections
  preserve or change the reviewer's meaning? Trace one back to its review.
- What did the real-word-error test show about what this approach cannot catch?
- How did accuracy and running time change with typo size and max_distance?
- How stable was accuracy across seeds, compared with Practical 11?
- Where should spell checking go relative to Practical 1's cleaning?
-->

## Viva Questions

1. What four kinds of single edit does an edit-distance spell corrector consider?
2. Why choose the most frequent candidate among the real words one edit away?
3. What is the difference between a non-word error and a real-word error, and which one can this corrector catch?
4. Why does correcting up to two edits take so much longer than one?
5. Why can running a spell checker after aggressive text cleaning be risky?
6. Why did the accuracy number stay stable across random seeds here when it did not in Practical 11?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
