"""
pos_chunking.py

POS tagging via NLTK's pre-trained tagger, and shallow parsing (chunking)
on top of it to extract noun phrases - useful as a lightweight way to pull
out "what is being talked about" (aspects) from a sentence, without needing
a full parser.
"""

from collections import Counter

import nltk
from nltk.chunk import RegexpParser

# Grammar: an optional determiner (DT), any number of adjectives (JJ*),
# followed by one or more nouns (NN, NNS, NNP, NNPS - the NN.* pattern
# catches all noun subtypes). This is intentionally simple - it will
# miss more complex noun phrases (e.g. with prepositional attachments)
# but is enough to demonstrate the technique.
_NP_GRAMMAR = r"NP: {<DT>?<JJ>*<NN.*>+}"
_parser = RegexpParser(_NP_GRAMMAR)


def pos_tag_tokens(tokens: list) -> list:
    """Tag each token with its part of speech using NLTK's pre-trained
    tagger. Returns a list of (word, tag) tuples.
    """
    return nltk.pos_tag(tokens)


def parse_tree(tagged_tokens: list):
    """Return the full chunk parse tree (not just extracted phrases) -
    useful for inspecting the tree structure directly, e.g. via print()
    or tree.pretty_print().
    """
    return _parser.parse(tagged_tokens)


def extract_noun_phrases(tagged_tokens: list) -> list:
    """Run the noun-phrase chunking grammar over already-POS-tagged tokens
    and return the extracted noun phrases as plain strings (words joined
    by spaces).
    """
    tree = parse_tree(tagged_tokens)
    phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join(word for word, tag in subtree.leaves())
        phrases.append(phrase)
    return phrases


def top_noun_phrases(list_of_token_lists: list, top_k: int = 10) -> list:
    """POS-tag and chunk each token list, then return the most common noun
    phrases across all of them - a rough proxy for "what's frequently
    talked about" across a corpus.
    """
    counter = Counter()
    for tokens in list_of_token_lists:
        tagged = pos_tag_tokens(tokens)
        phrases = extract_noun_phrases(tagged)
        counter.update(phrases)
    return counter.most_common(top_k)
