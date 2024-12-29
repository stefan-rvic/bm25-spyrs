import math
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix


class Bm25Sparse:
    def __init__(self, raw_corpus, tokenizer, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.tokenizer = tokenizer
        self.raw_corpus = raw_corpus
        self.corpus_size = len(raw_corpus)

        self._prepare_corpus()
        self._build_vocab()
        self._calculate_doc_lengths()
        self._calculate_idf()

        self._build_sparse_matrix()

    def _prepare_corpus(self):
        self.corpus = [self.tokenizer(doc) for doc in self.raw_corpus]
        self.corpus_frequencies = [Counter(terms) for terms in self.corpus]

    def _build_vocab(self):
        unique_terms = set()
        for doc in self.corpus:
            unique_terms.update(doc)

        self.term_to_idx = {term: idx for idx, term in enumerate(sorted(unique_terms))}
        self.idx_to_term = {idx: term for term, idx in self.term_to_idx.items()}
        self.vocab_size = len(self.term_to_idx)

    def _calculate_doc_lengths(self):
        self.doc_lengths = np.array([len(doc) for doc in self.corpus])
        self.avg_doc_len = np.mean(self.doc_lengths)

    def _calculate_idf(self):
        self.idf = np.zeros(self.vocab_size)
        docs_per_term = Counter()

        for doc_freq in self.corpus_frequencies:
            docs_per_term.update(doc_freq.keys())

        total_idf = 0.0
        terms_with_negative_idf = []

        for term, freq in docs_per_term.items():
            term_idx = self.term_to_idx[term]
            term_idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[term_idx] = term_idf
            total_idf += term_idf

            if term_idf < 0:
                terms_with_negative_idf.append(term_idx)

        eps = self.epsilon * (total_idf / self.vocab_size)
        for term_idx in terms_with_negative_idf:
            self.idf[term_idx] = eps

    def _build_sparse_matrix(self):
        rows = []
        cols = []
        data = []

        for doc_idx, doc_freq in enumerate(self.corpus_frequencies):
            doc_len = self.doc_lengths[doc_idx]
            len_norm = 1 - self.b + self.b * (doc_len / self.avg_doc_len)

            for term, freq in doc_freq.items():
                term_idx = self.term_to_idx[term]

                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * len_norm
                score = numerator / denominator

                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(score)

        self.score_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.corpus_size, self.vocab_size)
        )

    def top_n(self, query, n=5):
        query_terms = self.tokenizer(query)
        query_indices = [self.term_to_idx[term] for term in query_terms
                         if term in self.term_to_idx]

        if not query_indices:
            return []

        query_idfs = self.idf[query_indices]

        scores = self.score_matrix[:, query_indices].multiply(query_idfs).sum(axis=1).A1

        top_indices = np.argsort(scores)[::-1][:n]
        return list(zip(top_indices, scores[top_indices]))


if __name__ == "__main__":
    corpus = [
        "hello world",
        "hello python",
        "python world",
        "machine learning world",
        "deep learning python"
    ]


    def simple_tokenizer(text):
        return text.lower().split()


    bm25 = Bm25Sparse(corpus, simple_tokenizer)
    query = "python world"
    results = bm25.top_n(query)
    print(results)