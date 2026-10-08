"""
train.py

Trains a movie review sentiment classifier on NLTK's `movie_reviews`
corpus - 2000 labeled reviews (1000 positive, 1000 negative), a standard,
properly-sized benchmark dataset. This is a deliberate contrast with
Practical 11's 15-review toy dataset, which was only ever meant to
demonstrate that accuracy is unstable and untrustworthy at that scale -
with 2000 examples and a real 400-example test set, the accuracy number
this script reports actually means something.

Usage:
    python train.py
"""

import os
import random

import joblib
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

nltk.download("movie_reviews")


def load_dataset():
    """Load NLTK's movie_reviews corpus as parallel (texts, labels) lists,
    shuffled with a fixed seed for reproducibility.
    """
    documents = []
    for category in movie_reviews.categories():  # ['neg', 'pos']
        for fileid in movie_reviews.fileids(category):
            documents.append((movie_reviews.raw(fileid), category))

    random.seed(42)
    random.shuffle(documents)

    texts = [text for text, label in documents]
    labels = [label for text, label in documents]
    return texts, labels


def main():
    print("Loading dataset (downloads on first run, cached after)...")
    texts, labels = load_dataset()
    print(f"Loaded {len(texts)} reviews "
          f"({labels.count('pos')} positive, {labels.count('neg')} negative)")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\n--- Naive Bayes ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)
    nb_accuracy = accuracy_score(y_test, nb_preds)
    print(f"Test accuracy: {nb_accuracy:.2%}")
    print(classification_report(y_test, nb_preds))

    print("--- SVM (linear kernel) ---")
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train_vec, y_train)
    svm_preds = svm_model.predict(X_test_vec)
    svm_accuracy = accuracy_score(y_test, svm_preds)
    print(f"Test accuracy: {svm_accuracy:.2%}")
    print(classification_report(y_test, svm_preds))

    if svm_accuracy >= nb_accuracy:
        best_model, best_name = svm_model, "svm"
    else:
        best_model, best_name = nb_model, "naive_bayes"

    print(f"\nSaving best model ({best_name}, test accuracy "
          f"{max(nb_accuracy, svm_accuracy):.2%})...")

    os.makedirs("model", exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "model": best_model, "model_name": best_name},
        "model/sentiment_model.pkl",
    )
    print("Saved to model/sentiment_model.pkl")


if __name__ == "__main__":
    main()
