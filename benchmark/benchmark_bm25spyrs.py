import logging
import time

import bm25spyrs
import argparse

from benchmark import Benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkBm25Spyrs(Benchmark):
    def __init__(self, dataset):
        super(BenchmarkBm25Spyrs, self).__init__(dataset)
        self.model = bm25spyrs.Retriever(1.5, 0.75)

    def indexing_method(self, texts):
        self.model.index(texts)

    def scoring_method(self, queries, doc_ids, k):
        results = {}

        chunk_size = 100
        query_ids = list(queries.keys())
        queries = list(queries.values())

        total_time = 0.0

        for i in range(0, len(queries), chunk_size):
            batch_queries = queries[i:i + chunk_size]
            start_time = time.time()
            hits = self.model.top_n_batched(batch_queries, k)
            total_time += time.time() - start_time

            for batch_i, qid in enumerate(query_ids[i:i + chunk_size]):
                results[qid] = {doc_ids[index]: score for index, score in hits[batch_i]}

        logging.info(f"Total retrieving time for {len(queries)} queries: {total_time} seconds")
        return results

    # def scoring_method(self, queries, doc_ids, k):
    #     results = {}
    #     total_time = 0.0
    #
    #     for qid, query in queries.items():
    #         start_time = time.time()
    #         hits = self.model.top_n(query, k)
    #         total_time += time.time() - start_time
    #         results[qid] = {doc_ids[hit[0]]: hit[1] for hit in hits}
    #
    #     logging.info(f"Total retrieving time for {len(queries)} queries: {total_time} seconds")
    #     return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the BEIR dataset")
    args = parser.parse_args()

    benchmark = BenchmarkBm25Spyrs(args.dataset)
    benchmark.perform()
