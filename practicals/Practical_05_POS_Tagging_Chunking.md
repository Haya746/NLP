# Practical 5 — POS Tagging & Chunking

**Name:** <!-- fill in -->
**Course:** NLP
**Date:** <!-- fill in -->

## Aim
To assign part-of-speech (POS) tags to tokens, demonstrate how the same word can be tagged differently depending on sentence context, and use chunking to extract noun phrases from the review corpus.

## Theory

POS tagging assigns a grammatical category (noun, verb, adjective, etc.) to each token, based on both the word itself and its surrounding context. NLTK's tagger uses the Penn Treebank tagset (NN = singular noun, VB = base verb, JJ = adjective, DT = determiner, and so on).

Practical 4 already surfaced why context matters here: a POS tagger given an isolated word list has far less to work with than one given a full sentence. This practical tests that directly with a genuinely ambiguous word ("watch," which can be a verb or a noun).

Chunking (shallow parsing) groups tagged tokens into meaningful phrases without building a full syntactic parse tree. A simple grammar like `NP: {<DT>?<JJ>*<NN.*>+}` — an optional determiner, any number of adjectives, followed by one or more nouns — is enough to pull out noun phrases like "the acting" or "a great soundtrack." For review text, the noun phrases mentioned are a rough signal of what aspects of the movie people are actually discussing.

## Algorithm

1. POS-tag a full sample review and inspect the tags.
2. Tag the word "watch" in two constructed sentences where it plays a different grammatical role, and check whether the tagger correctly distinguishes them.
3. Define a noun-phrase chunking grammar and apply it to a tagged review, inspecting the parse tree.
4. Extract noun phrases from that review as plain text.
5. Run chunking across the whole dataset and find the most frequently mentioned noun phrases.

## Code

Full implementation lives in:
- `python/pos_chunking.py` — `pos_tag_tokens`, `parse_tree`, `extract_noun_phrases`, `top_noun_phrases`
- `notebooks/05_POS_Tagging_Chunking.ipynb` — full walkthrough with all steps run in order

## Output

<!--
Run notebooks/05_POS_Tagging_Chunking.ipynb top to bottom, then paste your
actual output here — the tagged review, whether "watch" was tagged
differently across the two sentences, the parse tree, the noun phrases from
the sample review, and the top-15 noun phrase list.
-->

## Conclusion

<!--
Write 4-6 sentences in your own words, based on what you actually observed:
- Did the tagger actually distinguish "watch" as verb vs noun across the
  two sentences? If not, why might this case be harder than "outdid" was?
- Were the noun phrases from the sample review sensible, or did the simple
  grammar miss/mangle anything?
- Did the top-15 noun phrase list actually reflect what these reviews are
  about?
- Would this be useful groundwork for aspect-based sentiment analysis on
  this data? What would need fixing first?
-->

## Viva Questions

1. What does a POS tagger actually predict, and what does it use to make that prediction?
2. What is chunking, and how is it different from full parsing?
3. Explain the noun-phrase grammar `NP: {<DT>?<JJ>*<NN.*>+}` in your own words.
4. Why is noun-phrase extraction useful for review-style text specifically?
5. What's a limitation of this simple regex-based chunking grammar?

*(Study notes for these are in the notebook's last section — work through them in your own words rather than memorizing.)*
