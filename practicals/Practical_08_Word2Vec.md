# Practical 8 — Word2Vec & Word Embeddings

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To train a Word2Vec model on our own review corpus, compare it against pretrained GloVe embeddings trained on a much larger corpus, and test how each handles out-of-vocabulary words — including a brief look at FastText's subword approach as a partial fix.

## Theory

Word2Vec learns dense, low-dimensional vectors for words based on the distributional hypothesis: words appearing in similar contexts tend to have similar meanings, so words used in similar contexts end up with similar vectors. Two training modes exist — Skip-gram (predict context from a target word) and CBOW (predict a target word from context).

Unlike TF-IDF/Bag-of-Words (Practical 7), which only capture exact word matches, embeddings can capture semantic relationships between words that never literally co-occur.

The catch: this only works with a lot of data. Meaningful embeddings emerge from statistical co-occurrence patterns across millions of words. Our dataset has 15 short reviews and about 136 unique words — nowhere close to enough. This practical is built to make that limitation concrete, comparing our own tiny trained model directly against GloVe vectors pretrained on 6 billion tokens.

A second issue specific to Word2Vec: it can only produce a vector for a word it saw during training — anything else is completely out-of-vocabulary (OOV), with no vector at all. FastText addresses this by representing each word as a bag of character n-grams, so it can approximate a vector for an unseen word from the n-grams it does recognize.

## Algorithm

1. Train a Word2Vec model on our own corpus and query `most_similar` for a few words.
2. Download pretrained GloVe vectors and run the identical queries.
3. Compare the two sets of results directly.
4. Test an out-of-vocabulary query against GloVe.
5. Train a small FastText model on our own corpus and test it on an unseen word variant, comparing against Word2Vec on the same query.

## Code

Full implementation lives in:
- `python/embeddings.py` — `train_word2vec`, `train_fasttext`, `load_pretrained_glove`, `safe_most_similar`
- `notebooks/08_Word2Vec.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/08_Word2Vec.ipynb top to bottom, then paste your actual
output here — the Word2Vec similarity results, GloVe's results for the same
words, the OOV test result, and the FastText vs Word2Vec vector-availability
test.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Did the custom Word2Vec model's results look semantically meaningful or
  essentially arbitrary? What does that say about corpus size requirements?
- Did GloVe's results look noticeably better? Give a specific example.
- Was "wasnt" in GloVe's vocabulary? What does either answer tell you?
- Did FastText succeed where Word2Vec failed, and why specifically (in
  terms of character n-grams, not just "it worked")?
-->

## Viva Questions

1. What is the distributional hypothesis, and how does Word2Vec use it?
2. Why did our custom-trained Word2Vec model likely produce poor-quality similarity results?
3. What's the difference between Skip-gram and CBOW?
4. Why can't Word2Vec produce a vector for an out-of-vocabulary word, and how does FastText solve this?
5. What's the practical trade-off of using pretrained embeddings (like GloVe) instead of training your own on a small, domain-specific corpus?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
