import os
import nltk
import tkinter as tk
from typing import Dict, List, Tuple

from gui import SearchInterface
from indexer import Indexer
from pre_processor import Preprocessor
from query_processor import QueryProcessor
from file_handler import load_index, load_queries, read_documents, save_index
from tfidf import compute_idf, compute_tfidf, normalize_document_vectors




def _ensure_nltk_resource(resource_path: str, download_name: str):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading {download_name}...")
        nltk.download(download_name)

def _resolve_data_folder(base_dir: str) -> str:
    
    primary = os.path.join(base_dir, "data", "TrumpSpeeches")
    if os.path.isdir(primary) and any(name.endswith(".txt") for name in os.listdir(primary)):
        return primary

    legacy = os.path.join(base_dir, "..", "Trump_Speeches")
    if os.path.isdir(legacy):
        return legacy

    return primary


def _extract_document_names(data_folder: str) -> Dict[int, str]:
    """Create deterministic mapping from doc_id to original filename."""
    files = sorted(
        [name for name in os.listdir(data_folder) if name.lower().endswith(".txt")],
        key=lambda filename: int(filename.split("_")[1].split(".")[0])
        if "_" in filename and filename.split("_")[1].split(".")[0].isdigit()
        else filename,
    )
    return {doc_id: filename for doc_id, filename in enumerate(files)}


def _build_pipeline(base_dir: str) -> Tuple[QueryProcessor, Dict[int, str], Dict[str, object]]:
    """Build the full VSM pipeline and return processor plus metadata."""
    stopword_file = os.path.join(base_dir, "data", "Stopword-List.txt")
    data_folder = _resolve_data_folder(base_dir)
    index_file = os.path.join(base_dir, "index.json")
    preprocessor = Preprocessor(stopword_file)

    if os.path.exists(index_file):
        print("Loading existing index...")
        loaded_payload = load_index(index_file)
    else:
        print("Building index (first run only)...")
        documents = read_documents(data_folder)
        document_names = _extract_document_names(data_folder)

        indexer = Indexer(preprocessor)

        tokenized_docs = indexer.preprocess_documents(documents)

        tf_index = indexer.build_tf_index_from_tokens(tokenized_docs)
        positional_index = indexer.build_positional_index(tokenized_docs)
        df_dict = indexer.compute_df(tf_index)
        idf_dict = compute_idf(df_dict, len(documents))
        tfidf_index = compute_tfidf(tf_index, idf_dict)
        doc_magnitudes = normalize_document_vectors(tfidf_index)
        payload = {
            "tf_index": tf_index,
            "positional_index": positional_index,
            "df_dict": df_dict,
            "idf_dict": idf_dict,
            "tfidf_index": tfidf_index,
            "doc_magnitudes": doc_magnitudes,
            "documents": document_names,
        }

        save_index(payload, index_file)
        loaded_payload = payload
    
    tfidf_loaded = {
        term: {int(doc_id): float(weight) for doc_id, weight in postings.items()}
        for term, postings in loaded_payload.get("tfidf_index", {}).items()
    }
    idf_loaded = {
        term: float(weight) for term, weight in loaded_payload.get("idf_dict", {}).items()
    }
    doc_mag_loaded = {
        int(doc_id): float(mag)
        for doc_id, mag in loaded_payload.get("doc_magnitudes", {}).items()
    }
    docs_loaded = {
        int(doc_id): filename
        for doc_id, filename in loaded_payload.get("documents", {}).items()
    }

    processor = QueryProcessor(
        preprocessor=preprocessor,
        tfidf_index=tfidf_loaded,
        idf_dict=idf_loaded,
        doc_magnitudes=doc_mag_loaded,
        documents=docs_loaded,
    )
    return processor, docs_loaded, loaded_payload


def _print_ranked_results(
    query: str, results: List[Tuple[int, float]], documents: Dict[int, str], limit: int = 10
) -> None:
    """Print ranked results in assignment-required query/length/set-like format."""
    doc_ids = [str(doc_id) for doc_id, _score in results]
    length = len(doc_ids)

    if length == 0:
        doc_id_set_string = "{}"
    else:
        doc_id_set_string = "{'" + "', '".join(doc_ids) + "'}"

    print(f"Query: {query}")
    print()
    print(f"Length={length}")
    print(doc_id_set_string)


def _run_queries_from_file(
    queries_file: str, processor: QueryProcessor, documents: Dict[int, str]
) -> None:
    """Run all queries from file and print ranked retrieval results."""
    queries = load_queries(queries_file)
    print(f"\nLoaded {len(queries)} query/queries from file.")
    for query in queries:
        results = processor.process_query(query)
        _print_ranked_results(query, results, documents)


def _run_cli(processor: QueryProcessor, documents: Dict[int, str]) -> None:
    """Start interactive CLI loop for custom query input."""
    print("\nInteractive mode: type a query and press Enter.")
    print("Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter query: ").strip()
        if user_query.lower() == "exit":
            print("Exiting interactive mode.")
            break
        results = processor.process_query(user_query)
        _print_ranked_results(user_query, results, documents)


def _run_gui(processor: QueryProcessor, documents: Dict[int, str]) -> None:
    """Launch Tkinter GUI for ranked retrieval."""
    root = tk.Tk()
    SearchInterface(root=root, query_processor=processor, documents=documents)
    root.mainloop()


def main() -> None:
    """Run full VSM pipeline, batch query evaluation, CLI, and GUI."""
    _ensure_nltk_resource('corpora/wordnet', 'wordnet')
    _ensure_nltk_resource('corpora/omw-1.4', 'omw-1.4')
    _ensure_nltk_resource('tokenizers/punkt', 'punkt')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    queries_file = os.path.join(base_dir, "data", "Query List VSM.txt")

    processor, documents, _ = _build_pipeline(base_dir)
    _run_queries_from_file(queries_file, processor, documents)
    _run_gui(processor, documents)
    


if __name__ == "__main__":
    main() 
