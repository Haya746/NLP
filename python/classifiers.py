"""
classifiers.py

Naive Bayes and SVM text classification on top of TF-IDF features, plus a
function specifically for stress-testing evaluation reliability on a very
small labeled dataset by repeating the train/test split many times with
different random splits and looking at how much the resulting accuracy
number actually varies.
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def train_naive_bayes(X, y) -> MultinomialNB:
    """Fit a Multinomial Naive Bayes classifier on TF-IDF/count features."""
    model = MultinomialNB()
    model.fit(X, y)
    return model


def train_svm(X, y) -> SVC:
    """Fit a linear-kernel SVM classifier."""
    model = SVC(kernel="linear")
    model.fit(X, y)
    return model


def evaluate_classifier(model, X, y) -> dict:
    """Return accuracy and a full precision/recall/F1 classification
    report for `model` on the given features/labels.
    """
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "report": classification_report(y, predictions, zero_division=0),
    }


def repeated_holdout_variance(X, y, model_fn, test_size: float = 0.3, n_repeats: int = 10) -> list:
    """Repeat a train/test split `n_repeats` times, each with a different
    random_state, training and evaluating a fresh model each time.
    Returns the list of test accuracies - meant to make visible how much
    a single accuracy number can swing on a very small dataset, purely
    from which few examples happened to land in the test set.

    `model_fn` should be a zero-argument callable that returns an unfit
    model (e.g. lambda: MultinomialNB()), so a fresh model is trained on
    each split rather than reusing one that's already seen data.
    """
    accuracies = []
    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None
        )
        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
    return accuracies


def summarize_variance(accuracies: list) -> dict:
    """Return mean, std, min, max for a list of accuracy values."""
    arr = np.array(accuracies)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
