from typing import Dict, List, Tuple

from pre_processor import Preprocessor
from vsm_model import VSMRetrievalModel


class QueryProcessor:
    """Query processor for VSM ranking with TF-IDF and cosine similarity."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        tfidf_index: Dict[str, Dict[int, float]],
        idf_dict: Dict[str, float],
        doc_magnitudes: Dict[int, float],
        documents: Dict[int, str],
        alpha: float = 0.005,
    ):
        """Initialize query processor with precomputed corpus statistics."""
        self.preprocessor = preprocessor
        self.tfidf_index = tfidf_index
        self.idf_dict = idf_dict
        self.doc_magnitudes = doc_magnitudes
        self.documents = documents
        self.retriever = VSMRetrievalModel(alpha=alpha)

    def _sanitize_query(self, query: str) -> str:
        """
        Basic query parsing: normalize whitespace and remove empty terms.
        """
        return " ".join(query.strip().split())

    def process_query(self, query: str) -> List[Tuple[int, float]]:
        """Preprocess query, build query vector, and return ranked results."""
        cleaned_query = self._sanitize_query(query)
        if not cleaned_query:
            return []

        query_tokens = self.preprocessor.preprocess_text(cleaned_query)

        # Filter tokens not in index
        query_tokens = [
            token for token in query_tokens if token in self.idf_dict
        ]

        # If nothing useful remains
        if not query_tokens:
            return []

        query_vector = self.retriever.build_query_vector(query_tokens, self.idf_dict)

        return self.retriever.rank_documents(
            query_vector=query_vector,
            tfidf_index=self.tfidf_index,
            doc_magnitudes=self.doc_magnitudes,
        )

    def process(self, query: str) -> List[Tuple[int, float]]:
        """Backward-compatible alias for `process_query`."""
        return self.process_query(query) 

