"""
predict.py

Loads the model trained by train.py and predicts sentiment for a review,
either passed as a command-line argument or typed in interactively.

Usage:
    python predict.py "This movie was absolutely fantastic"
    python predict.py
    (then type a review when prompted)
"""

import sys

import joblib


def load_model(path: str = "model/sentiment_model.pkl") -> dict:
    """Load the vectorizer + model bundle saved by train.py."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"No trained model found at '{path}'. Run train.py first.")
        sys.exit(1)


def predict(review_text: str, bundle: dict) -> str:
    """Vectorize `review_text` with the saved vectorizer and return the
    saved model's predicted label ('pos' or 'neg').
    """
    vector = bundle["vectorizer"].transform([review_text])
    return bundle["model"].predict(vector)[0]


def main():
    bundle = load_model()

    if len(sys.argv) > 1:
        review_text = " ".join(sys.argv[1:])
    else:
        review_text = input("Enter a movie review: ")

    prediction = predict(review_text, bundle)
    label = "positive" if prediction == "pos" else "negative"
    print(f"\nPredicted sentiment: {label}")


if __name__ == "__main__":
    main()
