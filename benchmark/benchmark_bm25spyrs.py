import bm25spyrs
import argparse

from benchmark import Benchmark


class BenchmarkBm25Spyrs(Benchmark):
    def __init__(self, dataset):
        super(BenchmarkBm25Spyrs, self).__init__(dataset)
        self.model = bm25spyrs.Retriever(1.5, 0.75)

    def indexing_method(self, texts):
        self.model.index(texts)

    def scoring_method(self, queries, doc_ids, k):
        results = {}

        for qid, query in queries.items():
            hits = self.model.top_n(query, k)
            results[qid] = {doc_ids[hit[0]]: hit[1] for hit in hits}

        return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the BEIR dataset")
    args = parser.parse_args()

    benchmark = BenchmarkBm25Spyrs(args.dataset)
    benchmark.perform()
