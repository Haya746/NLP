"""
ner.py

Named Entity Recognition via two approaches: NLTK's built-in chunker
(nltk.ne_chunk, statistical, trained on top of POS tags) and spaCy's
neural NER pipeline. Deliberately operates on raw, unprocessed text -
NER depends heavily on capitalization and punctuation as signals, both
of which Practical 1's clean_text() strips out, so these functions
are NOT meant to be fed cleaned/lowercased tokens.
"""

import nltk
import spacy

_nlp = spacy.load("en_core_web_sm")


def nltk_ner(text: str) -> list:
    """Run NLTK's NER chunker on raw text. Returns a list of
    (entity_text, entity_label) tuples. Requires the 'words',
    'maxent_ne_chunker' (or '..._tab') NLTK resources to be downloaded.
    """
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tagged)

    entities = []
    for subtree in tree.subtrees(filter=lambda t: t.label() != "S"):
        entity_text = " ".join(word for word, tag in subtree.leaves())
        entities.append((entity_text, subtree.label()))
    return entities


def spacy_ner(text: str) -> list:
    """Run spaCy's neural NER pipeline on raw text. Returns a list of
    (entity_text, entity_label) tuples.
    """
    doc = _nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def compare_raw_vs_cleaned(raw_text: str, cleaned_text: str) -> dict:
    """Run both NER approaches on a raw version of some text and on an
    already-cleaned (lowercased, punctuation-stripped) version of the same
    text, so the two can be directly compared.
    """
    return {
        "nltk_on_raw": nltk_ner(raw_text),
        "nltk_on_cleaned": nltk_ner(cleaned_text),
        "spacy_on_raw": spacy_ner(raw_text),
        "spacy_on_cleaned": spacy_ner(cleaned_text),
    }
