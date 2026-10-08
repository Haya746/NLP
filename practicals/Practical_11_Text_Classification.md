# Practical 11 — Text Classification (Naive Bayes & SVM)

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To train Naive Bayes and SVM classifiers on TF-IDF features to predict review sentiment, and to test whether a 15-example labeled dataset is actually large enough to evaluate a classifier's real-world accuracy — continuing the same data-scale question from Practicals 8 and 10, this time for supervised learning.

## Theory

Naive Bayes and SVM are both standard supervised classifiers for text, typically fed TF-IDF or Bag-of-Words features. Naive Bayes assumes feature independence and models class-conditional word probabilities; SVM finds a decision boundary that maximizes the margin between classes.

Both need labeled data — unlike every previous practical, this one requires a ground-truth sentiment label for each review, which our dataset doesn't come with.

The real point of this practical is a different data-scale problem than Practicals 8 and 10: even if a classifier trains "successfully," can you trust an accuracy number computed from a test set as small as 4-5 examples? A single example changes the reported accuracy by roughly 20-25 percentage points at that size. This practical tests that directly.

## Algorithm

1. Assign a ground-truth sentiment label to each of the 15 reviews (starting from a suggested draft, checked against your own reading).
2. Build TF-IDF features and train Naive Bayes + SVM on all 15 labeled examples, evaluating on that same data.
3. Do a real 70/30 train/test split and evaluate on the held-out portion.
4. Repeat the split 10 times with different random seeds and examine the spread of resulting accuracy.

## Code

Full implementation lives in:
- `python/classifiers.py` — `train_naive_bayes`, `train_svm`, `evaluate_classifier`, `repeated_holdout_variance`, `summarize_variance`
- `notebooks/11_Text_Classification.ipynb` — full walkthrough, including the label-assignment table, with all steps run in order

## Output

<!--
Run notebooks/11_Text_Classification.ipynb top to bottom, then paste your
actual output here — your final label list (noting any changes from the
draft), the in-sample accuracies, the single split accuracies, and the
full spread of 10 accuracy values (mean/std/min/max) for both classifiers.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Did you change either ambiguous label (reviews 9, 13)? Why?
- Were the in-sample accuracies suspiciously high? Does that mean the
  model generalizes?
- How different was the single split accuracy from in-sample accuracy?
- How much did accuracy vary across the 10 repeated splits? What does
  that say about trusting a single accuracy number here?
- Is this the same "not enough data" problem as Practicals 8 and 10, or
  a distinct issue specific to supervised evaluation?
-->

## Viva Questions

1. Why can't accuracy on the training set itself be trusted as a measure of a classifier's real-world performance?
2. Why is a 70/30 train/test split especially unreliable on a 15-example dataset?
3. What does a wide spread of accuracy values across repeated random splits actually demonstrate?
4. What's the practical difference between how Naive Bayes and SVM make predictions?
5. How does this practical's finding relate to Practical 8 (Word2Vec) and Practical 10 (LDA)?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
