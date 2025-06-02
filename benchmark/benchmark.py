import json
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
        self.model = None
        data_path = os.path.join("beir_datasets", dataset)
        self.corpus, self.queries, self.qrels = GenericDataLoader(data_path).load(split="test")
        self.doc_ids = list(self.corpus.keys())
        self.dataset = dataset
        self.result_tracker = {
            'dataset': dataset
        }

    def benchmark_indexing(self):
        logging.info(f"indexing {self.dataset}")
        texts = [f'{doc["title"]} {doc["text"]}' for doc in self.corpus.values()]

        tokenized_texts = self.tokenize_corpus(texts)
        start_time = time.time()
        self.indexing_method(tokenized_texts)
        indexing_time = time.time() - start_time
        del texts

        self.result_tracker['indexing_time'] = indexing_time
        logging.info(f"Indexing completed in {indexing_time:.2f} seconds")

    def tokenize_corpus(self, texts):
        pass

    def indexing_method(self, tokenized_texts):
        pass

    def compute_mat_size(self):
        pass

    def benchmark_scoring(self):
        logging.info(f"scoring {self.dataset} queries")
        evaluator = EvaluateRetrieval()
        results = self.scoring_method(self.queries, self.doc_ids, 100)
        (self.result_tracker['ndcg'],
         self.result_tracker['map'],
         self.result_tracker['recall'],
         self.result_tracker['p']) = evaluator.evaluate(self.qrels, results, [1, 10, 100])

    def scoring_method(self, queries, doc_ids, k):
        pass

    def perform(self):
        logging.info("started benchmark")
        self.benchmark_indexing()
        self.compute_mat_size()
        self.benchmark_scoring()
        self.write_results(f'results_{self.result_tracker["model_name"]}.json')
        logging.info("ended benchmark")

    def write_results(self, filename):
        if not os.path.exists(filename):
            data = []
        else:
            with open(filename, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []

        data.append(self.result_tracker)

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
