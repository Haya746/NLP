# NLP — Natural Language Processing Coursework & Practice

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=flat&logo=spacy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data-013243?style=flat&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat)

## Overview

This repository is where I'm working through Natural Language Processing — starting
from my mid-semester practicals and building out from there, topic by topic, as I
learn each one. It's a working repo, not a finished product: notebooks get added
as I actually complete them, so the commit history reflects the order I learned
things in rather than a dump of pre-written content.

## Topics covered (updated as notebooks are added)

- [x] Text preprocessing & normalization
- [x] Tokenization (word, sentence, regex-based)
- [ ] Stopword removal
- [ ] Stemming & lemmatization
- [ ] POS tagging & chunking
- [ ] Named Entity Recognition (NER)
- [ ] Bag of Words / CountVectorizer / HashingVectorizer
- [ ] TF-IDF
- [ ] N-grams & skip-grams
- [ ] Word2Vec / Doc2Vec / FastText (intro)
- [ ] Word clouds
- [ ] Sentiment analysis (VADER, TextBlob)
- [ ] Text classification (Naive Bayes, Logistic Regression, SVM, Random Forest)
- [ ] Topic modelling (LDA)
- [ ] Text similarity & cosine similarity
- [ ] Spell checking
- [ ] Text summarization (basic)
- [ ] Chatbot basics

## Tech stack

Python · Jupyter · NLTK · spaCy · scikit-learn · Gensim · TextBlob · pandas · NumPy · matplotlib/seaborn

## Folder structure

```
NLP/
├── README.md
├── requirements.txt
├── .gitignore
├── datasets/          # small sample datasets used across notebooks
├── notebooks/         # one notebook per topic, added as completed
├── python/            # reusable helper modules (preprocessing, vectorizers, etc.)
├── practicals/         # write-ups tied to actual coursework practicals
├── mini_projects/      # complete end-to-end projects
└── images/             # plots, wordclouds, diagrams used in notebooks/README
```

## Installation

```bash
git clone https://github.com/Haya746/NLP.git
cd NLP
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```

## How to run

```bash
jupyter notebook
```
Open any notebook under `notebooks/` — each one is self-contained and documents
theory, code, and output for that topic.

## Mini projects

Added as they're completed:
- [ ] Movie review sentiment classifier
- [ ] Email spam detector
- [ ] Fake news detector
- [ ] Resume keyword extractor

Each will have its own folder with a README, requirements, dataset, and
training/prediction scripts.

## Future improvements

- Transformer-based fine-tuning (BERT/DistilBERT) once covered in coursework
- Streamlit demo app for the sentiment/spam classifiers
- Expand dataset variety beyond the sample sets

## License

MIT — see [LICENSE](LICENSE).
