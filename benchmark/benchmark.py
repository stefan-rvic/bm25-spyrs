import logging
import os
import time

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Benchmark:
    def __init__(self, dataset):
        data_path = os.path.join("beir_datasets", dataset)
        self.corpus, self.queries, self.qrels = GenericDataLoader(data_path).load(split="test")
        self.doc_ids = list(self.corpus.keys())
        self.dataset = dataset

    def benchmark_indexing(self):
        logging.info(f"indexing {self.dataset}")
        texts = [f'{doc["title"]} {doc["text"]}' for doc in self.corpus.values()]

        start_time = time.time()
        self.indexing_method(texts)
        indexing_time = time.time() - start_time
        del texts
        logging.info(f"Indexing completed in {indexing_time:.2f} seconds")

    def indexing_method(self, texts):
        pass

    def benchmark_scoring(self):
        logging.info(f"scoring {self.dataset} queries")
        evaluator = EvaluateRetrieval()
        results = self.scoring_method(self.queries, self.doc_ids, 100)
        evaluator.evaluate(self.qrels, results, [1, 10, 100])

    def scoring_method(self, queries, doc_ids, k):
        pass

    def perform(self):
        logging.info("started benchmark")
        self.benchmark_indexing()
        self.benchmark_scoring()
        logging.info("ended benchmark")
