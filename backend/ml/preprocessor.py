import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, max_features=10000, min_df=2, max_df=0.95, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        words = nltk.word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in self.stop_words]
        return ' '.join(words)

    def fit(self, texts):
        cleaned = [self.clean_text(text) for text in texts]
        self.vectorizer.fit(cleaned)
        return self

    def transform(self, texts):
        cleaned = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned)

    def fit_transform(self, texts):
        cleaned = [self.clean_text(text) for text in texts]
        return self.vectorizer.fit_transform(cleaned)