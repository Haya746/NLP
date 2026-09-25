"""
sentiment.py

VADER and TextBlob sentiment scoring, wrapped for direct comparison. VADER
is specifically rule-based around punctuation emphasis, capitalization,
degree modifiers, and negation - all signals Practical 1's clean_text()
strips out - so this module is meant to be tested on both raw and cleaned
text, not just one.
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


def vader_score(text: str) -> dict:
    """Return VADER's full score dict: neg/neu/pos proportions plus the
    overall 'compound' score (-1 to +1).
    """
    return _vader.polarity_scores(text)


def textblob_score(text: str) -> dict:
    """Return TextBlob's polarity (-1 to +1) and subjectivity (0 to 1)."""
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}


def classify_compound(compound: float, threshold: float = 0.05) -> str:
    """Bucket a VADER compound score into positive/negative/neutral using
    VADER's own conventional threshold (+-0.05).
    """
    if compound >= threshold:
        return "positive"
    elif compound <= -threshold:
        return "negative"
    else:
        return "neutral"


def compare_vader_textblob(text: str) -> dict:
    """Score the same text with both tools for direct comparison."""
    return {
        "vader": vader_score(text),
        "textblob": textblob_score(text),
    }
