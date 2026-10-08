# Movie Review Sentiment Classifier

A real (not toy) sentiment classifier trained on NLTK's `movie_reviews` corpus — 2000 labeled reviews (1000 positive, 1000 negative), the standard benchmark dataset for this exact task.

## Why this uses a different dataset than the practicals/ notebooks

Practicals 1-11 deliberately used a tiny 15-review dataset to isolate specific preprocessing effects (that's what made it possible to trace one specific bug — like an apostrophe getting stripped — all the way through six later notebooks). Practical 11 specifically showed that with only 15 examples, a single train/test split's accuracy number swings wildly depending on which few examples land in the test set, and can't be trusted in isolation.

This project exists to show the other side of that finding: with a properly-sized dataset (2000 reviews, an 80/20 split giving 400 real test examples), a train/test accuracy number is actually stable and meaningful — this isn't a toy demonstration, it's a real classifier.

## Aim
To train and compare Naive Bayes and SVM classifiers on TF-IDF features for binary movie review sentiment classification, and save the better-performing model for reuse.

## Dataset
NLTK's built-in `movie_reviews` corpus — downloaded automatically the first time `train.py` runs (cached afterward). No dataset file is checked into this repo; it's fetched at runtime rather than committing ~2000 review files to git.

## How to run

```bash
pip install -r requirements.txt
python train.py
```

This downloads the corpus (first run only), trains both classifiers, prints test accuracy and a full classification report for each, and saves whichever model scored higher to `model/sentiment_model.pkl`.

Then, to predict on a new review:

```bash
python predict.py "This movie was absolutely fantastic, best film I've seen all year"
```

or run `python predict.py` with no argument to be prompted for input interactively.

## Files
- `train.py` — loads the corpus, trains Naive Bayes + SVM on TF-IDF features, evaluates both, saves the better one
- `predict.py` — loads the saved model and predicts sentiment for a new review
- `model/` — created by `train.py`; holds the saved vectorizer + model bundle (not checked into git — see `.gitignore`)

## Notes
- `TfidfVectorizer(max_features=5000, stop_words="english")` caps the vocabulary and removes English stopwords — reasonable defaults for a corpus this size, unlike the practicals' notebooks, which deliberately handled stopword removal manually and tracked its effects step by step.
- Both models are evaluated with a single 80/20 split here. If you want to see whether *this* dataset's accuracy is as unstable as Practical 11 found for the 15-review one, try adapting `repeated_holdout_variance` from `python/classifiers.py` to this corpus and compare the spread.
