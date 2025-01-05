import math
from collections import Counter
import numpy as np
from scipy.sparse import csc_matrix
import re


def tokenize(texts, stopwords=None):
    word_pattern = re.compile(r'\b\w+\b')

    corpus = []
    term_to_id = {}

    for text in texts:

        tokens = word_pattern.findall(text.lower())
        terms = []
        for token in tokens:
            if token in stopwords:
                continue

            if token not in term_to_id:
                token_id = len(term_to_id)
                term_to_id[token] = token_id

            terms.append(term_to_id[token])

        corpus.append(terms)

    return corpus, term_to_id

class Bm25:
    def __init__(self, tokenizer, k1=1.5, b=0.75, epsilon=0.25):
        self.indptr = None
        self.indices = None
        self.data = None
        self.score_matrix = None
        self.n_docs = None
        self.vocab = None
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.tokenizer = tokenizer

    def index(self, raw_corpus):
        corpus, self.vocab = self.tokenizer([doc['text'] for doc in raw_corpus.values()])

        doc_frequencies, term_frequencies = self._compute_frequencies(corpus)

        doc_lengths = np.array([len(doc) for doc in corpus])
        avg_doc_len = np.mean(doc_lengths)
        self.n_docs = len(corpus)
        n_terms = len(self.vocab)

        idf_array = self._compute_idf_array(n_terms, self.n_docs, doc_frequencies)

        scores, rows, cols = self._prepare_sparse_matrix(idf_array, doc_frequencies, term_frequencies, doc_lengths, avg_doc_len)

        self.score_matrix = csc_matrix(
            (scores, (rows, cols)),
            shape=(self.n_docs, n_terms)
        )

        self.data = self.score_matrix.data
        self.indices = self.score_matrix.indices
        self.indptr = self.score_matrix.indptr

    @staticmethod
    def _compute_frequencies(corpus):
        doc_frequencies = Counter()
        term_frequencies = []

        for terms in corpus:
            unique_term_count = Counter(terms)
            term_frequencies.append(
                (
                    np.array(list(unique_term_count.keys())),
                    np.array(list(unique_term_count.values()))
                )
            )
            doc_frequencies.update(unique_term_count.keys())

        return doc_frequencies, term_frequencies

    @staticmethod
    def _compute_idf_array(n_terms, n_docs, doc_frequencies):
        idf_array = np.zeros(n_terms)

        for term, freq in doc_frequencies.items():
            idf_array[term] = math.log(n_docs) - math.log(freq)

        return idf_array

    def _prepare_sparse_matrix(self, idf_array, doc_frequencies, term_frequencies, doc_lengths, avg_doc_len):
        size = sum(doc_frequencies.values()) # compute before

        rows = np.empty(size, dtype="int32")
        cols = np.empty(size, dtype="int32")
        scores = np.empty(size, dtype="float32")

        step = 0
        for i, (terms, tf_array) in enumerate(term_frequencies):
            doc_len = doc_lengths[i]
            tfc = (tf_array * (self.k1 + 1)) / (tf_array + self.k1 * (1 - self.b + self.b * (doc_len / avg_doc_len)))
            idf = idf_array[terms]
            score = idf * tfc

            start = step
            end = step = step + len(score)
            rows[start:end] = i
            cols[start:end] = terms
            scores[start:end] = score

        return scores, rows, cols


    def top_n(self, query, n=5):
        _, query_terms = self.tokenizer([query])
        query_indices = [self.vocab[term] for term in query_terms.keys()
                         if term in self.vocab]

        if not query_indices:
            return []

        scores = np.zeros(self.n_docs)

        for term_idx in query_indices:
            start = self.indptr[term_idx]
            end = self.indptr[term_idx + 1]

            doc_indices = self.indices[start:end]
            term_scores = self.data[start:end]

            scores[doc_indices] += term_scores

        if n < len(scores):
            top_indices = np.argpartition(scores, -n)[-n:]
            top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
        else:
            top_indices = np.argsort(scores)[::-1]

        return [(idx, scores[idx]) for idx in top_indices]


if __name__ == "__main__":
    # beir_corpus = {
    #     '1' : {'title': "hw", 'text': "hello world"},
    #     '2' : {'title': "hp", 'text': "hello python"},
    #     '3' : {'title': "pw", 'text': "python world"},
    #     '4' : {'title': "mlw", 'text': "machine learning world"},
    #     '5' : {'title': "dlp", 'text': "deep learning python"},
    # }

    from nltk.corpus import stopwords

    import nltk

    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    import time
    import logging
    import os

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    dataset = "scifact"
    data_path = os.path.join("datasets", dataset)

    if not os.path.exists(data_path) or not all(
            os.path.exists(os.path.join(data_path, f)) for f in ["corpus.jsonl", "queries.jsonl"]):
        logging.info(f"Dataset {dataset} not found. Downloading...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, "datasets")
    else:
        logging.info(f"Dataset {dataset} found at {data_path}")


    ir_corpus, queries, _ = GenericDataLoader(data_path).load(split="test")

    bm25 = Bm25(tokenizer=lambda text : tokenize(text, stopwords=stop_words))

    start_time = time.time()
    bm25.index(ir_corpus)
    indexing_time = time.time() - start_time
    logging.info(f"Indexing completed in {indexing_time:.2f} seconds")

    # query = "python world"
    # results = bm25.top_n(query)
    # print(results)