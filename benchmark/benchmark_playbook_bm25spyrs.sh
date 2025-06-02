#!/bin/bash

source /venv/Scripts/activate


combinations=(
    "arguana"
    "scifact"
    "nfcorpus"
    "quora"
    "fiqa"
    "scidocs"
    "trec-covid"
    "webis-touche2020"
    "nq"
    "hotpotqa"
    "fever"
)

for combo in "${combinations[@]}"; do
    read -r dataset backend <<< "$combo"
    echo "Running: $dataset with $backend"
    python benchmark_bm25spyrs.py --dataset "$dataset"
done

deactivate

