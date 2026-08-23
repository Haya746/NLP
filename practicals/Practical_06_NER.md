# Practical 6 — Named Entity Recognition (NER)

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To identify named entities (people, organizations, dates, money, etc.) in review text using NLTK and spaCy, and to test what happens to NER accuracy when it's run on Practical 1's cleaned/lowercased tokens instead of raw text.

## Theory

Named Entity Recognition (NER) locates and classifies spans of text that refer to real-world entities — people (PERSON), organizations (ORG), locations (GPE), dates (DATE), monetary values (MONEY), and so on.

Two approaches are compared here: NLTK's `ne_chunk` (statistical, built on top of POS tagging) and spaCy's NER pipeline (a trained neural model, generally more accurate on real-world text).

Every practical so far has reused Practical 1's cleaned tokens (lowercased, punctuation/numbers stripped) as the starting point. NER is different in one important way: capitalization is one of the strongest signals a NER system uses to recognize a proper noun ("Robert De Niro" vs "robert de niro"), and punctuation/number preservation matters for recognizing dates and monetary values. This practical tests whether feeding NER the same cleaned tokens used everywhere else was ever actually a good idea.

## Algorithm

1. Pick a review containing a real named entity (review 4 — "Robert De Niro").
2. Run both NLTK and spaCy NER on the raw, original review text.
3. Run both on the same review after being passed through `clean_text()`.
4. Compare all four results directly.
5. Run spaCy NER across all 15 raw reviews and tally what entity types actually appear.

## Code

Full implementation lives in:
- `python/ner.py` — `nltk_ner`, `spacy_ner`, `compare_raw_vs_cleaned`
- `notebooks/06_NER.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/06_NER.ipynb top to bottom, then paste your actual output
here — the four-way raw-vs-cleaned comparison for review 4, and the entity
type counts / per-review breakdown from Step 2.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Did NER correctly identify "Robert De Niro" on raw text, and what
  happened once the text was cleaned?
- Did NLTK or spaCy handle the raw text noticeably better?
- What entity types actually turned up across the 15 raw reviews?
- Should NER run on raw or cleaned text in a real pipeline, and how does
  that change how you'd order Practicals 1-6 if designing from scratch?
-->

## Viva Questions

1. What is Named Entity Recognition, and name three common entity types.
2. Why does capitalization matter so much for NER, specifically for English text?
3. Why would running NER on Practical 1's cleaned tokens be a bad idea?
4. What's the difference between NLTK's `ne_chunk` and spaCy's NER pipeline, mechanically?
5. If you had to redesign Practicals 1-6 as a single pipeline, where would NER need to run relative to the cleaning step?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
