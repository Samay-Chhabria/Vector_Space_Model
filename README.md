# 📘 Assignment-2: Vector Space Model (TF-IDF + Cosine Similarity)

## 📌 Project Overview

This project implements a complete **Vector Space Model (VSM)** based Information Retrieval system using the **Trump Speeches dataset**.

The system ranks documents using:

* **TF-IDF weighting**
* **Cosine similarity**
* **Alpha threshold filtering** (`alpha = 0.005`)

---

## ⚙️ VSM Workflow

### 1. Text Preprocessing

Each document is normalized using:

* Lowercasing
* Tokenization (NLTK)
* Stopword removal (custom list)
* Lemmatization (WordNet)

---

### 2. Term Frequency (TF) Index

Sparse representation:

```
{ term: { doc_id: raw_tf } }
```

* TF uses **raw frequency (no log scaling)**

---

### 3. Document Frequency (DF)

```
df(t) = number of documents containing term t
```

---

### 4. Inverse Document Frequency (IDF)

```
idf(t) = log(N / df(t))
```

> Note: IDF values are precomputed and reused during query processing.

---

### 5. TF-IDF Index

Sparse weighted index:

```
{ term: { doc_id: tf * idf } }
```

---

### 6. Document Vector Magnitudes

Precomputed for cosine normalization:

```
||d|| = sqrt(sum(tfidf²))
```

Stored as:

```
{ doc_id: magnitude }
```

---

### 7. Query Processing

For each query:

* Apply same preprocessing as documents
* Filter tokens not in vocabulary
* Build query TF-IDF vector
* Compute cosine similarity using inverted index
* Apply alpha threshold (`> 0.005`)
* Rank results in descending order

---

## 📊 Retrieval Model

### Cosine Similarity

```
sim(q, d) = (q · d) / (||q|| * ||d||)
```

Efficient computation is done using:

* Sparse vectors
* Inverted index traversal (term-wise accumulation)

---

## 🚀 Implemented Features

* Modular architecture (clean separation of components)
* Efficient sparse vector representation (dictionary-based)
* Precomputed:

  * IDF values
  * Document magnitudes
* Fast ranking using inverted index
* Alpha threshold filtering for noise reduction
* Batch query execution from file
* Interactive CLI search mode
* GUI interface
* Index persistence using `index.json`

---

## 🧩 Additional Internal Components

### Positional Index (Built but Not Used)

Created during indexing phase:

```
{ term: { doc_id: [positions] } }
```

Included for potential **phrase query extension**.

---

## 📁 Folder Structure

```
data/
│── TrumpSpeeches/            # Document collection
│── Query List VSM.txt        # Input queries
│── Stopword-List.txt         # Stopword list

preprocessing/
│── pre_processor.py          # Text preprocessing

indexing/
│── indexer.py                # TF, DF, positional index

weighting/
│── tfidf.py                  # TF, IDF, TF-IDF, normalization

retrieval/
│── vsm_model.py              # Query vector + ranking

query/
│── query_processor.py        # Query pipeline

utils/
│── file_handler.py           # File and JSON handling

gui.py                        # Tkinter GUI
main.py                       # Entry point
index.json                    # Saved index data
```

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install nltk
```

### 2. Run the system

```
python main.py
```

---

## 🔄 System Execution Flow

* Loads existing index (or builds it on first run)
* Executes all queries from file
* launches GUI

---

## 📤 Output Format (Assignment Requirement)

For each query:

```
Query: <query text>

Length=<number_of_documents>
{'doc_id1', 'doc_id2', ...}
```

---

## 🖥️ GUI

The GUI provides:

* Query input field
* Search and clear buttons
* Keyboard shortcuts (Enter / Esc)
* Status feedback (Searching, Done, Error)
* Scrollable results display
* Output in assignment-required format

---

## ✅ Final Notes

* The system follows the **Vector Space Model pipeline**
* Designed for **efficient retrieval using sparse data structures**
* Easily extendable for advanced IR features
