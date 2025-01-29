import argparse
import logging
import sys
import time

import bm25s
import numpy as np
from nltk import SnowballStemmer
from nltk.corpus import stopwords

from benchmark import Benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

stopwords = stopwords.words('english')

class BenchmarkBm25s(Benchmark):
    def __init__(self, dataset):
        super(BenchmarkBm25s, self).__init__(dataset)
        self.result_tracker['model_name'] = 'BM25s'
        stemmer = SnowballStemmer("english")
        self.stemming = lambda texts: [stemmer.stem(text) for text in texts]
        self.model = bm25s.BM25(method='atire', backend='numpy')

    def indexing_method(self, texts):
        corpus_tokens = bm25s.tokenize(texts, stemmer=self.stemming, stopwords=stopwords, allow_empty=False, show_progress=False)
        self.model.index(corpus_tokens, show_progress=False)

    def compute_mat_size(self):
        scores = self.model.scores
        data_size = scores["data"].nbytes if isinstance(scores["data"], np.ndarray) else sys.getsizeof(scores["data"])
        indices_size = scores["indices"].nbytes if isinstance(scores["indices"], np.ndarray) else sys.getsizeof(
            scores["indices"])
        indptr_size = scores["indptr"].nbytes if isinstance(scores["indptr"], np.ndarray) else sys.getsizeof(
            scores["indptr"])

        mem = (data_size + indices_size + indptr_size) / 1024 / 1024
        logger.info(f"sparse matrix size: {mem :.2f} MB")
        self.result_tracker['matrix_size'] = mem

    def scoring_method(self, queries, doc_ids, k):
        results = {}

        chunk_size = 100
        query_ids = list(queries.keys())
        queries = list(queries.values())

        total_time = 0.0
        for i in range(0, len(queries), chunk_size):
            batch_queries = queries[i:i + chunk_size]

            start_time = time.time()
            tokenized_chunk = bm25s.tokenize(batch_queries, stemmer=self.stemming, stopwords=stopwords, allow_empty=False, show_progress=False)
            hits = self.model.retrieve(tokenized_chunk, k=k, n_threads=-1, chunksize=chunk_size, show_progress=False)
            total_time += time.time() - start_time

            for batch_i, qid in enumerate(query_ids[i:i + chunk_size]):
                results[qid] = {doc_ids[index]: float(score) for index, score in zip(hits.documents[batch_i], hits.scores[batch_i])}

        self.result_tracker['queries_count'] = len(queries)
        self.result_tracker['retrieval_total_time'] = total_time
        logging.info(f"Total retrieving time for {len(queries)} queries: {total_time} seconds")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the BEIR dataset")
    args = parser.parse_args()

    benchmark = BenchmarkBm25s(args.dataset)
    benchmark.perform()
