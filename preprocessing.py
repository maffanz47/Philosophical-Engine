"""
preprocessing.py — NLP Preprocessing & Dual Vectorization (v4.3).

Changes vs previous versions:
  • TfidfVectorizer constraints updated to stop cluster collapse:
    - stop_words='english'
    - max_df=0.60 (acts as dynamic stop-word filter for cross-domain bleed)
    - min_df=5
  • ngram_range=(1, 3) retained.
  • max_features reduced to 10,000 to shrink dense array memory by 1/3.

Math — TF-IDF:
  w(t,d) = log(1 + tf(t,d)) × log(N / df(t))
  sublinear_tf=True applies log(1+tf) to dampen high-freq term dominance.
"""

import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK assets are present
for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(_pkg, quiet=True)

_STOP  = set(stopwords.words("english"))
_LEMMA = WordNetLemmatizer()

# ── Updated constants (v3) ─────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10_000      # reduced to fit RAM (1/3 smaller)
TFIDF_NGRAM_RANGE  = (1, 3)      # unigrams, bigrams, trigrams


def clean_and_lemmatize(text: str) -> str:
    """
    Tokenize → lowercase → strip non-alpha → remove stop-words → lemmatize.
    Returns a cleaned string suitable for TF-IDF and vocab encoding.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    return " ".join(
        _LEMMA.lemmatize(t)
        for t in tokens
        if t not in _STOP and len(t) > 2
    )


def build_tfidf(clean_texts: list[str],
                max_features: int = TFIDF_MAX_FEATURES,
                ngram_range: tuple = TFIDF_NGRAM_RANGE):
    """
    Fit a TF-IDF vectorizer with trigram support.

    Why trigrams?
      Single words like "reason" appear across all philosophy.
      Bigrams like "pure reason" narrow to Kant.
      Trigrams like "critique pure reason" are almost uniquely Kantian.

    min_df=3: ignore n-grams seen in fewer than 3 docs (prunes hapax n-grams).
    max_df=0.85: ignore n-grams in >85% of docs (too common to discriminate).

    Returns:
        matrix     – sparse CSR [n_samples × max_features]
        vectorizer – fitted TfidfVectorizer (reuse for inference)
    """
    vec = TfidfVectorizer(
        stop_words='english',
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=0.60,
        min_df=5,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(clean_texts)
    print(f"  [TF-IDF] Matrix shape: {matrix.shape}  "
          f"(ngram_range={ngram_range}, max_features={max_features:,})")
    return matrix, vec


