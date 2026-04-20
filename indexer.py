import pickle
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List
from pre_processor import Preprocessor

class Indexer:
    """Build term-frequency and document-frequency indexes for VSM."""

    def __init__(self, preprocessor: Preprocessor):
        """Initialize indexer with a reusable preprocessor instance."""
        self.preprocessor = preprocessor

    def preprocess_documents(self, documents: Dict[int, str]) -> Dict[int, List[str]]:
        """Preprocess all documents once."""
        return {
            doc_id: self.preprocessor.preprocess_text(text)
            for doc_id, text in documents.items()
        }
    
    def build_positional_index(self, tokenized_docs):
        positional_index = defaultdict(lambda: defaultdict(list))

        for doc_id, tokens in tokenized_docs.items():
            for position, term in enumerate(tokens):
                positional_index[term][doc_id].append(position)

        return dict(positional_index)

    def save_index(self, index: Dict, filepath: str) -> None:
        """Save index to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(index, f)

    def load_index(self, filepath: str) -> Dict:
        """Load index from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
        
    def build_tf_index_from_tokens(self, tokenized_docs: Dict[int, List[str]]) -> Dict[str, Dict[int, int]]:
        tf_index: DefaultDict[str, Dict[int, int]] = defaultdict(dict)

        for doc_id, tokens in tokenized_docs.items():
            term_counts = Counter(tokens)
            for term, tf in term_counts.items():
                tf_index[term][doc_id] = tf

        return dict(tf_index)

    def compute_df(self, tf_index: Dict[str, Dict[int, int]]) -> Dict[str, int]:
        """Compute document frequency for each term: {term: df}."""
        return {term: len(postings) for term, postings in tf_index.items()}  