# Practical 13 — Word Clouds

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To visualise word frequencies in the review corpus as word clouds, compare how stopword removal (NLTK's list, then the custom domain list from Practical 2) changes what the cloud shows, and test how much information a word cloud can actually carry on a 15-review corpus.

## Theory

A word cloud draws each word at a size proportional to its frequency, so the most common words stand out visually. It only ever shows one thing: relative word frequency. It says nothing about word order, context, or sentiment.

Two things determine whether a word cloud is useful:
- **Preprocessing.** Without stopword removal, the biggest words in almost any English text are function words like "the" and "and". Domain-specific filler matters too: in a movie-review corpus, "movie" and "film" can dominate even after a standard stopword list is applied.
- **Frequency spread.** A word cloud communicates only through size differences. If almost every word appears once, almost every word is drawn at the same size, and the cloud looks like decoration rather than a summary.

`WordCloud.generate(text)` quietly tokenizes, lowercases, and removes its own default stopwords before drawing. To keep control over preprocessing (and keep it comparable with Practicals 1-2), this practical computes frequencies itself and passes them in with `generate_from_frequencies()`.

## Algorithm

1. Build three versions of the corpus with identical cleaning: (A) cleaned only, (B) cleaned + NLTK stopwords removed, (C) cleaned + NLTK and custom domain stopwords removed.
2. Print the top 10 words for each version.
3. Measure how flat each frequency distribution is.
4. Draw all three word clouds side by side and save the figure to `images/`.
5. Generate a cloud the "default" way from raw text and inspect what the library did on its own.

## Code

Full implementation lives in:
- `python/word_clouds.py` — `word_frequencies`, `make_wordcloud`, `save_wordcloud`, `frequency_spread`
- `notebooks/13_WordCloud.ipynb` — full walkthrough with all steps run in order
- `images/wordcloud_comparison.png` — the saved figure, produced when the notebook is run

## Output

<!--
Run notebooks/13_WordCloud.ipynb top to bottom, then paste your actual
output here — the three top-10 lists, the three frequency-spread
dictionaries, a screenshot or description of the three clouds, and what
the default-library cloud contained in Step 4.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- How did the top-10 lists change from A to B to C?
- What did the frequency-spread numbers say about how much size variation
  the clouds could show?
- Were any of the clouds actually a useful summary of the corpus?
- What did the library do on its own in Step 4 that you didn't ask for?
- Is a word cloud a good tool for a corpus this small?
-->

## Viva Questions

1. What does a word cloud actually encode, and what does it leave out?
2. Why are word clouds usually made after stopword removal?
3. Why might a domain-specific stopword list be needed on top of a standard one?
4. Why does a word cloud look uninformative when almost every word occurs once?
5. Why compute frequencies yourself and use `generate_from_frequencies()` instead of passing raw text to `generate()`?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
