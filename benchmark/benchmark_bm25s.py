import logging
import time

import bm25s

import argparse

from nltk import SnowballStemmer

from benchmark import Benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkBm25s(Benchmark):
    def __init__(self, dataset):
        super(BenchmarkBm25s, self).__init__(dataset)
        stemmer = SnowballStemmer("english")
        self.stemming = lambda texts: [stemmer.stem(text) for text in texts]
        self.model = bm25s.BM25(method='atire', backend='numpy')

    def indexing_method(self, texts):
        corpus_tokens = bm25s.tokenize(texts, stemmer=self.stemming, allow_empty=False)
        self.model.index(corpus_tokens)

    def scoring_method(self, queries, doc_ids, k):
        results = {}

        chunk_size = 100
        query_ids = list(queries.keys())
        queries = list(queries.values())

        total_time = 0.0
        for i in range(0, len(queries), chunk_size):
            batch_queries = queries[i:i + chunk_size]

            start_time = time.time()
            tokenized_chunk = bm25s.tokenize(batch_queries, stemmer=self.stemming, allow_empty=False)
            hits = self.model.retrieve(tokenized_chunk, k=k, n_threads=-1, chunksize=chunk_size)
            total_time += time.time() - start_time

            for batch_i, qid in enumerate(query_ids[i:i + chunk_size]):
                results[qid] = {doc_ids[index]: float(score) for index, score in zip(hits.documents[batch_i], hits.scores[batch_i])}

        logging.info(f"Total retrieving time for {len(queries)} queries: {total_time} seconds")
        return results

    # def scoring_method(self, queries, doc_ids, k):
    #     results = {}
    #     total_time = 0.0
    #
    #     for qid, query in queries.items():
    #         start_time = time.time()
    #         tokenized_query = bm25s.tokenize(query, stemmer=self.stemming)
    #         hits = self.model.retrieve(tokenized_query, k=k)
    #         total_time += time.time() - start_time
    #
    #         results[qid] = {doc_ids[index]: float(score) for index, score in zip(hits.documents[0], hits.scores[0])}
    #
    #     logging.info(f"Total retrieving time for {len(queries)} queries: {total_time} seconds")
    #     return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the BEIR dataset")
    args = parser.parse_args()

    benchmark = BenchmarkBm25s(args.dataset)
    benchmark.perform()
