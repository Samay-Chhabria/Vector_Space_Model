import json
import os
from typing import Any, Dict, List


def read_text_file(file_path: str) -> str:
    """Read a UTF-8 text file and return its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Text file not found: {file_path}") from exc


def write_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """Write dictionary data to a JSON file."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(f"Could not write JSON file: {file_path}") from exc


def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format in file: {file_path}") from exc


def read_documents(folder_path: str) -> Dict[int, str]:
    """Read .txt documents from a folder and return {doc_id: text}."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Documents folder not found: {folder_path}")

    files = sorted(
        [name for name in os.listdir(folder_path) if name.lower().endswith(".txt")],
        key=lambda filename: int(filename.split("_")[1].split(".")[0])
        if "_" in filename and filename.split("_")[1].split(".")[0].isdigit()
        else filename,
    )

    documents: Dict[int, str] = {}
    for doc_id, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        documents[doc_id] = read_text_file(file_path)
    return documents


def load_queries(file_path: str) -> List[str]:
    """Load non-empty query lines from a UTF-8 text file."""
    content = read_text_file(file_path)
    return [line.strip() for line in content.splitlines() if line.strip()]


def save_index(index: Dict[str, Any], path: str) -> None:
    """Persist index payload to disk as JSON."""
    write_json_file(path, index)


def load_index(path: str) -> Dict[str, Any]:
    """Load index payload from JSON file."""
    return read_json_file(path) 
