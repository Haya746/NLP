# Practical 4 — Stemming & Lemmatization

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim

To reduce words to their base/root form using stemming (Porter) and lemmatization (WordNet, with and without POS information), compare their outputs on irregular forms, and measure the effect on vocabulary size.

## Theory

Both stemming and lemmatization aim to collapse different inflected forms of a word ("run", "running", "ran") down to one representative form, so they're counted as the same feature rather than three separate ones.

Stemming (e.g. the Porter algorithm) applies a fixed set of suffix-stripping rules, with no real knowledge of the language beyond those rules. This makes it fast, but it can produce forms that aren't real words (e.g. "studies" -> "studi"), and it has no concept of irregular forms — an irregular verb like "seen" or "outdid" won't be reduced, since there's no suffix pattern to strip.

Lemmatization (e.g. NLTK's WordNet lemmatizer) instead looks a word up against a real vocabulary/morphological database, returning an actual dictionary base form. This is more linguistically correct, but it needs to know the word's part of speech to work correctly — without it, WordNetLemmatizer assumes every word is a noun, so a verb like "seen" won't be correctly reduced. With the correct POS supplied, it can resolve irregular forms that stemming cannot.

## Algorithm

1. Reuse cleaned tokens from Practical 1.
2. Apply Porter stemming to words with irregular forms ("seen", "outdid", "given", "dragged").
3. Apply lemmatization to the same words with and without POS information, and compare all three outputs.
4. Run stemming and POS-aware lemmatization across the whole dataset and compute vocabulary size for each, compared to the Practical 1 baseline (136).
5. Identify words where stemming and lemmatization disagree, and check whether stemmed forms are real words.

## Code

Full implementation lives in:

- `python/stemming_lemmatization.py` — `porter_stem`, `get_wordnet_pos`, `lemmatize_tokens`, `compare_stem_lemma`
- `notebooks/04_Stemming_Lemmatization.ipynb` — full walkthrough with all steps run in order

## Output

seen -> stemmed: seen lemma (no POS): seen lemma (with POS): see
outdid -> stemmed: outdid lemma (no POS): outdid lemma (with POS): outdid
given -> stemmed: given lemma (no POS): given lemma (with POS): give
dragged -> stemmed: drag lemma (no POS): dragged lemma (with POS): drag

Raw vocabulary (Practical 1 baseline): 136
Vocabulary after stemming: 133
Vocabulary after POS-aware lemmatization: 130

38 words where stemmed and lemmatized forms differ:
absolutely -> stem: absolut lemma: absolutely
acting -> stem: act lemma: acting
admission -> stem: admiss lemma: admission
alone -> stem: alon lemma: alone
anyone -> stem: anyon lemma: anyone
anything -> stem: anyth lemma: anything
are -> stem: are lemma: be
before -> stem: befor lemma: before
believe -> stem: believ lemma: believe
confusing -> stem: confus lemma: confuse
decade -> stem: decad lemma: decade
dialogue -> stem: dialogu lemma: dialogue
every -> stem: everi lemma: every
fantastic -> stem: fantast lemma: fantastic
given -> stem: given lemma: give
his -> stem: hi lemma: his
honestly -> stem: honestli lemma: honestly
insane -> stem: insan lemma: insane
is -> stem: is lemma: be
masterpiece -> stem: masterpiec lemma: masterpiece
middle -> stem: middl lemma: middle
movie -> stem: movi lemma: movie
nothing -> stem: noth lemma: nothing
outdid -> stem: outdid lemma: outdo
pacing -> stem: pace lemma: pacing
penny -> stem: penni lemma: penny
promise -> stem: promis lemma: promise
really -> stem: realli lemma: really
seen -> stem: seen lemma: see
simply -> stem: simpli lemma: simply
somewhere -> stem: somewher lemma: somewhere
terrible -> stem: terribl lemma: terrible
this -> stem: thi lemma: this
very -> stem: veri lemma: very
visuals -> stem: visual lemma: visuals
was -> stem: wa lemma: be
waste -> stem: wast lemma: waste
worst -> stem: worst lemma: bad

## Conclusion

The results show that stemming and lemmatization both reduce vocabulary size, but lemmatization performs a more linguistically accurate normalization. The original vocabulary of 136 words was reduced to 133 after stemming and further to 130 after POS-aware lemmatization, indicating that lemmatization merged more word variants into their correct base forms. Stemming often produced truncated or non-dictionary words such as "absolut", "fantast", "movi", and "terribl", whereas lemmatization preserved meaningful dictionary forms like "absolutely", "fantastic", "movie", and "terrible". It also correctly converted inflected words based on their grammatical role, for example "seen" → "see", "given" → "give", "was" → "be", and "worst" → "bad", while stemming either left some words unchanged or simply removed suffixes. Overall, the comparison demonstrates that although stemming is faster and slightly reduces vocabulary, POS-aware lemmatization produces cleaner and more semantically meaningful tokens, making it a better choice for most NLP tasks such as sentiment analysis and text classification.

## Viva Questions

1. What's the fundamental difference in approach between stemming and lemmatization?
2. Why does WordNetLemmatizer need a part-of-speech tag to work correctly?
3. Give an example of over-stemming (or explain what it means).
4. Why might a stemmer fail on an irregular verb like "outdid" or "given"?
5. What's the practical trade-off between choosing stemming vs lemmatization for a real pipeline?

_(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)_
