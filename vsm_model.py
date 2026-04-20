import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class VSMRetrievalModel:
    """Vector Space Model retrieval using sparse TF-IDF vectors."""

    def __init__(self, alpha: float = 0.005):
        """Initialize retrieval model with alpha filtering threshold."""
        self.alpha = alpha


    def build_query_vector(
        self, query_tokens: List[str], idf_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """Build sparse TF-IDF query vector using corpus IDF values."""
        if not query_tokens:
            return {}

        term_counts = Counter(query_tokens)
        query_vector: Dict[str, float] = {}
        for term, raw_tf in term_counts.items():
            idf_value = idf_dict.get(term, 0.0)
            query_vector[term] = raw_tf * idf_value
        return query_vector

    def rank_documents(
        self,
        query_vector: Dict[str, float],
        tfidf_index: Dict[str, Dict[int, float]],
        doc_magnitudes: Dict[int, float],
    ) -> List[Tuple[int, float]]:
        """
        Rank documents by cosine similarity and apply alpha threshold.
        Uses inverted index postings for efficient dot-product accumulation.
        """
        if not query_vector:
            return []
        
        query_norm_sq = sum(weight * weight for weight in query_vector.values())
        if query_norm_sq <= 0.0:
            return []
        query_magnitude = math.sqrt(query_norm_sq)

        dot_scores: Dict[int, float] = defaultdict(float)
        
        for term, query_weight in query_vector.items():
            postings = tfidf_index.get(term)
            if not postings:
                continue
            for doc_id, doc_weight in postings.items():
                dot_scores[doc_id] += query_weight * doc_weight

        ranked_results: List[Tuple[int, float]] = []
        for doc_id, dot_product in dot_scores.items():
            doc_magnitude = doc_magnitudes.get(doc_id, 0.0)
            if doc_magnitude <= 0.0:
                continue
            similarity = dot_product / (query_magnitude * doc_magnitude)
            if similarity > self.alpha:
                ranked_results.append((doc_id, similarity))

        ranked_results.sort(key=lambda item: item[1], reverse=True)
        return ranked_results 
