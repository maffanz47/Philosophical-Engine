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


# ── Dense Vocabulary Encoder (for PyTorch Embedding layers) ───────────────

class VocabEncoder:
    """
    Builds a word→index mapping for PyTorch Embedding layers.
    Index 0 = <PAD>,  Index 1 = <UNK>.
    Note: the Embedding layer operates on unigrams only;
    n-gram context is captured by the LSTM's recurrence.
    """

    PAD, UNK = 0, 1

    def __init__(self, max_vocab: int = 20_000):
        self.max_vocab = max_vocab
        self.word2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, clean_texts: list[str]) -> "VocabEncoder":
        """Build vocab from top-frequency tokens across all cleaned texts."""
        from collections import Counter
        freq = Counter()
        for doc in clean_texts:
            freq.update(doc.split())
        for i, (word, _) in enumerate(freq.most_common(self.max_vocab - 2)):
            idx = i + 2
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
        print(f"  [Vocab] Size: {len(self.word2idx):,} tokens")
        return self

    def encode(self, text: str, max_len: int = 200) -> list[int]:
        """Map tokens to indices; pad or truncate to max_len."""
        tokens = text.split()[:max_len]
        ids    = [self.word2idx.get(t, self.UNK) for t in tokens]
        ids   += [self.PAD] * (max_len - len(ids))
        return ids

    def encode_batch(self, texts: list[str], max_len: int = 200) -> np.ndarray:
        """Encode a list of texts → numpy int array [n × max_len]."""
        return np.array([self.encode(t, max_len) for t in texts], dtype=np.int64)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
