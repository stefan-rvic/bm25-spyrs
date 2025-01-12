import bm25s

import argparse

from nltk import SnowballStemmer

from benchmark import Benchmark


class BenchmarkBm25s(Benchmark):
    def __init__(self, dataset):
        super(BenchmarkBm25s, self).__init__(dataset)
        stemmer = SnowballStemmer("english")
        self.stemming = lambda texts: [stemmer.stem(text) for text in texts]
        self.model = bm25s.BM25(method='atire')

    def indexing_method(self, texts):
        corpus_tokens = bm25s.tokenize(texts, stemmer=self.stemming)
        self.model.index(corpus_tokens)

    def scoring_method(self, queries, doc_ids, k):
        results = {}

        for qid, query in queries.items():
            tokenized_query = bm25s.tokenize(query, stemmer=self.stemming)
            hits = self.model.retrieve(tokenized_query, k=k)
            results[qid] = {doc_ids[index]: float(score) for index, score in zip(hits.documents[0], hits.scores[0])}

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the BEIR dataset")
    args = parser.parse_args()

    benchmark = BenchmarkBm25s(args.dataset)
    benchmark.perform()
