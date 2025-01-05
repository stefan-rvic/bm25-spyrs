import math
from collections import Counter
import numpy as np
from scipy.sparse import csc_matrix
import re


def tokenize(text, stemmer=None, stopwords=None):
    word_pattern = re.compile(r'\b\w+\b')

    tokens = word_pattern.findall(text.lower())

    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    if stemmer:
        tokens = [stemmer(token) for token in tokens]

    return tokens

class Bm25:
    def __init__(self, tokenizer, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.tokenizer = tokenizer

        self.corpus = []
        self.corpus_frequencies = []
        self.corpus_size = 0

        self.doc_id_to_idx = {}
        self.idx_to_doc_id = {}
        self.idx_to_term = {}
        self.term_to_idx = {}

        self.docs_per_term = Counter()

    def _prepare_corpus(self, corpus):
        self.corpus_size = len(corpus)

        for idx, doc in enumerate(corpus):
            doc_id = doc.get('_id', idx)
            self.doc_id_to_idx[doc_id] = idx
            self.idx_to_doc_id[idx] = doc_id

            # title = doc.get('title', '').strip()
            text = doc.get('text', '').strip()
            # full_text = f"{title} {text}".strip()

            tokens = self.tokenizer(text)
            count = Counter(tokens)

            self.corpus_frequencies.append(count)
            self.corpus.append(tokens)
            self.docs_per_term.update(count.keys())

    def _build_vocab(self):
        self.vocab_size = len(self.docs_per_term)

        for idx, term in enumerate(self.docs_per_term.keys()):
            self.term_to_idx[term] = idx
            self.idx_to_term[idx] = term

    def _calculate_doc_lengths(self):
        self.doc_lengths = np.array([len(doc) for doc in self.corpus])
        self.avg_doc_len = np.mean(self.doc_lengths)

    def _calculate_idf(self):
        self.idf = np.zeros(self.vocab_size)

        for term, freq in self.docs_per_term.items():
            term_idx = self.term_to_idx[term]
            self.idf[term_idx] = math.log(self.corpus_size) - math.log(freq)

    def _release_memory(self):
        self.docs_per_term = None

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

        self.score_matrix = csc_matrix(
            (data, (rows, cols)),
            shape=(self.corpus_size, self.vocab_size)
        )

        self.data = self.score_matrix.data
        self.indices = self.score_matrix.indices
        self.indptr = self.score_matrix.indptr

    def index(self, corpus):
        self._prepare_corpus(corpus)
        self._build_vocab()
        self._calculate_doc_lengths()
        self._calculate_idf()
        self._release_memory()
        self._build_sparse_matrix()

    def top_n(self, query, n=5):
        query_terms = self.tokenizer(query)
        query_indices = [self.term_to_idx[term] for term in query_terms
                         if term in self.term_to_idx]

        if not query_indices:
            return []

        scores = np.zeros(self.corpus_size)
        query_idfs = self.idf[query_indices]

        for idx, term_idx in enumerate(query_indices):
            start = self.indptr[term_idx]
            end = self.indptr[term_idx + 1]
            term_scores = self.data[start:end] * query_idfs[idx]
            term_indices = self.indices[start:end]
            np.add.at(scores, term_indices, term_scores)

        top_indices = np.argsort(scores)[::-1][:n]
        results = [(self.idx_to_doc_id[idx], scores[idx]) for idx in top_indices]
        return results


if __name__ == "__main__":
    beir_corpus = [
        {'_id': 1, 'title': "hw", 'text': "hello world"},
        {'_id': 2, 'title': "hp", 'text': "hello python"},
        {'_id': 3, 'title': "pw", 'text': "python world"},
        {'_id': 4, 'title': "mlw", 'text': "machine learning world"},
        {'_id': 5, 'title': "dlp", 'text': "deep learning python"},
    ]

    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords

    import nltk

    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    stemmer = SnowballStemmer("english")
    tokenizer = lambda text : tokenize(text, lambda token: stemmer.stem(token), stop_words)

    bm25 = Bm25(tokenizer=tokenizer)
    bm25.index(beir_corpus)
    query = "python world"
    results = bm25.top_n(query)
    print(results)