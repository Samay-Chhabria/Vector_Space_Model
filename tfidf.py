import math
from collections import defaultdict
from typing import Dict, Tuple  


def compute_idf(df_dict: Dict[str, int], total_docs: int) -> Dict[str, float]:
    if total_docs <= 0:
        return {}

    idf_dict = {}
    for term, df in df_dict.items():
        if df > 0:
            idf_dict[term] = math.log(total_docs / df)
        elif df == 0:
            idf_dict[term] = math.log(total_docs / (df + 1))
    return idf_dict


def compute_tfidf(
    tf_index: Dict[str, Dict[int, int]], idf_dict: Dict[str, float]
) -> Dict[str, Dict[int, float]]:
    """Compute sparse TF-IDF index: {term: {doc_id: tf-idf}}."""
    tfidf_index: Dict[str, Dict[int, float]] = {}
    for term, postings in tf_index.items():
        idf_value = idf_dict.get(term, 0.0)
        if idf_value == 0.0:
            continue
        weighted_postings: Dict[int, float] = {}
        for doc_id, raw_tf in postings.items():
            tf_weight=raw_tf
            tfidf_weight = tf_weight * idf_value
            if tfidf_weight > 0.0:
                weighted_postings[doc_id] = tfidf_weight
        if weighted_postings:
            tfidf_index[term] = weighted_postings
    return tfidf_index

def normalize_document_vectors(tfidf_index):
    squared_norms = defaultdict(float)

    for term, postings in tfidf_index.items():
        for doc_id, weight in postings.items():
            squared_norms[doc_id] += weight * weight

    return {doc_id: math.sqrt(value) for doc_id, value in squared_norms.items()} 
