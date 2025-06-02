# BM25Spyrs: A rust ported version of BM25S

BM25Spyrs (combination of Sparse Matrix, Python and Rust) is a lightweight library
that ports the essential features from the library [BM25S](https://bm25s.github.io/) to Rust.

### Python binded

The project ships a binding to python via [Pyo3](https://pyo3.rs/), enabling the usage of the library in python (intended usage).

### Features

This project focuses on BM25s's core features :
* Tokenization : using [Scikit-learn](https://scikit-learn.org/stable/index.html) Regex (r"(?u)\b\w\w+\b"), a stemmer (snowball) and stop-words filtering ([NLTK](https://www.nltk.org/search.html?q=stopwords)'s list).
* Indexing : leveraging sparse matrices.
* Single and batch retrieval : supporting multi-threading retrieval and a simple heap based top k algorithm to sorts the results.

By it simplicity it only supports the BM25 Atire version of the formula

### Limitations
* This project is currently CPU optimized (SIMD, cache accessing, etc...) and do not leverage GPU optimizations.
* It lacks customizations whether it is from the tokenization, indexing or retrieval side.

## Quick Installation (for python usage)
### Requirements
* python 3.11+
* Rust toolchain (install from rustup.rs)
```shell
pip install git+https://github.com/stefan-rvic/bm25-spyrs.git
```
## Usage

### Initialization

First, we prepare instances of the tokenizer and bm25spyrs class.

```python
import bm25spyrs

tokenizer = bm25spyrs.Tokenizer()

k1, b = 1.5, 0.75 # Common parameters for BM25 Atire
retriever = bm25spyrs.Retriever(k1, b) # Common BM25 Atire 
```

### Tokenizing documents
We will follow Beir's format for their corpora.
Unlike BM25F, BM25 Atire don't take into account fields like title, body or abstract, so we can join titles and texts to avoid information loss.
```python
beir_corpus = {
    '1': {'title': "Fast Cars", 'text': "Red cars drive very fast."},
    '2': {'title': "Fast Animals", 'text': "Cheetahs run fast like cars."},
    '3': {'title': "Red Fruits", 'text': "Apples are red and sweet."}
}
corpus = [f'{doc["title"]} {doc["text"]}' for doc in beir_corpus.values()]

tokenized_corpus = tokenizer.perform(corpus)
#{
#  'corpus': [[8, 1, 0, 1, 3, 8], [8, 6, 7, 5, 8, 4, 1], [0, 2, 10, 0, 9]], 
#  'vocab': {'cheetah': 7, 'anim': 6, 'fast': 8, 'fruit': 2, 'drive': 3, 'car': 1, 'like': 4, 'sweet': 9, 'red': 0, 'appl': 10, 'run': 5}
#}
```
Each words end up being turned into an id (saving memory vs using Strings). And a vocab to map the ids to their actual word.
### Indexing / Scoring
```python
retriever.index(tokenized_corpus)
```
Tokens composing the corpus are being scored and stored in a sparse matrix.
### Retrieving top 2 elements

When tokenizing queries, we only want the stemmed / stop-words filtered version.
```python
query = "fast cars"
n = 2

tokenized_query = tokenizer.perform_simple(query)
# ['fast', 'car']

hits = retriever.top_n(tokenized_query, n)
# [(0, 1.1584718227386475), (1, 0.9269601106643677)]
```

The document 1 is the most relevant result to the given query. This is because "fast" and "car" appears 4 times (title and text) vs 3 occurences in document 2.
### Retrieving top 2 elements for batch of queries
Same process when processing multiple queries
```python
queries = [
    "fast cars",
    "red",
    "cheetahs run"  
]
n = 2

tokenized_queries = [tokenizer.perform_simple(query) for query in queries]
# [['fast', 'car'], ['red'], ['cheetah', 'run']]

hits = retriever.top_n_batched(tokenized_query, n)
# [
#   [(0, 1.1584718227386475), (1, 0.9269601106643677)], 
#   [(2, 0.6120228171348572), (0, 0.40546512603759766)], 
#   [(1, 2.0439298152923584), (0, 0.0)]
# ]
```