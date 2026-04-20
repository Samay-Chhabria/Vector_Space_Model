from pydoc import text
import re
from typing import List, Set

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class Preprocessor:
    """Text preprocessing pipeline for VSM-ready indexing and querying."""

    def __init__(self, stopword_file: str):
        """Initialize stopword set and lemmatizer."""
        self.stopwords: Set[str] = set()
        self.lemmatizer = WordNetLemmatizer()
        self._load_stopwords(stopword_file)

    def _load_stopwords(self, filepath: str) -> None:
        """Load stopwords from a text file into a set."""
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip().lower()
                if word:
                    self.stopwords.add(word)

    def preprocess_text(self, text: str) -> List[str]:
        """
        Apply lowercase normalization, tokenization, stopword removal,
        and lemmatization.
        """
        normalized = text.lower()
        normalized=re.sub(r"[^a-zA-Z0-9]", " ", normalized)

        tokens = word_tokenize(normalized)

        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stopwords 
        ]

        return tokens

    def preprocess(self, text: str) -> List[str]:
        """Backward-compatible wrapper around `preprocess_text`."""
        return self.preprocess_text(text) 

