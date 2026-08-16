# Practical 1 — Text Preprocessing & Tokenization

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To perform text preprocessing (case normalization, punctuation/number/whitespace removal) and tokenization (sentence-level and word-level, comparing regex, NLTK, and spaCy tokenizers) on a small sample review corpus.

## Theory

Text preprocessing is the set of steps used to turn raw, noisy text into a clean, consistent form before any downstream NLP task. Raw text usually contains inconsistencies that don't carry useful signal — mixed casing, punctuation, stray digits, extra whitespace — which can cause the same word to be treated as different tokens if left unaddressed (e.g. `"Great!!"` vs `"great"`).

Common preprocessing steps:
- **Lowercasing** — normalizes case so `"Great"` and `"great"` are treated identically.
- **Punctuation/special character removal** — strips symbols that usually don't carry semantic meaning on their own.
- **Number removal** — optional; useful when digits are noise, but harmful when numbers carry meaning (dates, quantities, ratings).
- **Whitespace normalization** — collapses multiple spaces/newlines into one.

Tokenization is the process of splitting text into smaller units (tokens) that downstream algorithms operate on:
- **Sentence tokenization** splits a document into sentences — has to correctly handle cases like abbreviations and decimals that contain periods without being sentence boundaries.
- **Word tokenization** splits text into words/punctuation tokens. Different tokenizers disagree on edge cases:
  - A **regex tokenizer** is fast but naive — no linguistic rules, so it can't tell that `"don't"` should be handled specially.
  - **NLTK's** word tokenizer follows Penn Treebank conventions (splits contractions, separates punctuation as its own tokens).
  - **spaCy's** tokenizer is rule-based and model-informed, generally handling contractions and edge cases more consistently.

## Algorithm

1. Load the sample review dataset.
2. For each review, apply the cleaning pipeline: lowercase → remove special characters → remove numbers → collapse whitespace.
3. Compare a few reviews before and after cleaning.
4. Apply sentence tokenization to a multi-sentence example.
5. Apply word tokenization three ways (regex, NLTK, spaCy) to the same sentence and compare.
6. Run the full pipeline across the dataset and compute token count / vocabulary size statistics.

## Code

Full implementation lives in:
- `python/preprocessing.py` — cleaning functions (`to_lowercase`, `remove_punctuation`, `remove_numbers`, `remove_extra_whitespace`, `clean_text`)
- `python/tokenizer.py` — tokenization functions (`nltk_sentence_tokenize`, `nltk_word_tokenize`, `regex_word_tokenize`, `spacy_word_tokenize`, `compare_tokenizers`)
- `notebooks/01_Text_Preprocessing.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/01_Text_Preprocessing.ipynb top to bottom, then paste your actual
output here (the before/after cleaning examples, the tokenizer comparison,
the average token count and vocabulary size). Screenshots or copy-pasted
text both work — check what your instructor expects.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Which preprocessing step changed the text the most, and did it lose any
  information you didn't expect?
- Where did the three tokenizers disagree, and why?
- What did the token count / vocabulary size numbers tell you about the dataset?
-->

## Viva Questions

1. Why is lowercasing usually done before other preprocessing steps?
2. Give an example where removing numbers would lose important information.
3. What's the difference between sentence tokenization and word tokenization?
4. Why might NLTK and spaCy tokenize a contraction like "don't" differently from a plain regex tokenizer?
5. Why do we compute vocabulary size, and what does a small vs large vocabulary tell you about a dataset?

*(Study notes for these questions are in the notebook's last section — work through them in your own words rather than memorizing.)*
