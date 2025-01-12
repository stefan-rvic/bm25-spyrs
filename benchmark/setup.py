import logging

from beir import util

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASETS = [
    "msmarco",
    "trec-covid",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact"
]


def download_and_unpack(dataset: str, base_path: str = "beir_datasets") -> bool:
    try:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, base_path)
        logger.info(f"Successfully downloaded and extracted {dataset} to {data_path}")
    except Exception as e:
        logger.error(f"Error processing {dataset}: {str(e)}")

if __name__ == '__main__':
    for dataset in DATASETS:
        download_and_unpack(dataset)