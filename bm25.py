import math
from collections import Counter

import numpy as np


class Bm25:

    def __init__(self, raw_corpus, tokenizer, k1=1.5, b=0.75, epsilon=0.25):
        self.total_term_count = 0
        self.avg_doc_len = 0.0
        self.idf = {}
        self.raw_corpus = raw_corpus
        self.tokenizer = tokenizer

        self.corpus_size = len(raw_corpus)
        self.corpus = []
        self.corpus_frequencies = []
        self.corpus_lengths = []

        self.docs_per_term = Counter()
        self.idf = {}

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self._prepare_corpus()
        self._prepare_frequencies()
        self._calculate_idf()

    def _prepare_corpus(self):
        self.corpus = [self.tokenizer(doc) for doc in self.raw_corpus]

    def _prepare_frequencies(self):
        self.docs_per_term = Counter()

        for terms in self.corpus:
            length = len(terms)
            self.corpus_lengths.append(length)
            self.total_term_count += length

            freq = Counter(terms)
            self.corpus_frequencies.append(freq)
            self.docs_per_term.update(freq.keys())

        self.avg_doc_len = self.total_term_count / self.corpus_size
        return self

    def _calculate_idf(self):
        total_idf = 0.0

        terms_with_negative_idf = []
        for term, freq in self.docs_per_term.items():
            term_idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[term] = term_idf
            total_idf += term_idf

            if term_idf < 0:
                terms_with_negative_idf.append(term)

        eps = self.epsilon * (total_idf / self.total_term_count)
        for term in terms_with_negative_idf:
            self.idf[term] = eps

        return self

    def _calculate_score(self, query_term, corpus_lengths):
        q_frequencies = np.array([doc_freq.get(query_term, 0) for doc_freq in self.corpus_frequencies])
        return self.idf.get(query_term, 0) * (q_frequencies * (self.k1 + 1) /
                                          (q_frequencies + self.k1 * (
                                                      1 - self.b + self.b * corpus_lengths / self.avg_doc_len)))

    def _sum_scores(self, query):
        scores = np.zeros(self.corpus_size)

        q_terms = self.tokenizer(query)
        corpus_lengths = np.array(self.corpus_lengths)

        for term in q_terms:
            scores += self._calculate_score(term, corpus_lengths)

        return scores

    def top_n(self, query, n = 5):
        scores = self._sum_scores(query)
        top_indices  = np.argsort(scores)[::-1][:n]
        return list(zip(top_indices , scores[top_indices ]))


corpus = [
    "hello world",
    "hello python",
    "python world",
    "machine learning world",
    "deep learning python"
]


def simple_tokenizer(text):
    return text.lower().split()


bm25 = Bm25(corpus, simple_tokenizer)
query = "python world"
results = bm25.top_n(query)
print(results)
