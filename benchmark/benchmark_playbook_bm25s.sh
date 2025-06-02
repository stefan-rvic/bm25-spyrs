#!/bin/bash

source /venv/Scripts/activate


combinations=(
    "arguana jax"
    "scifact jax"
    "nfcorpus jax"
    "quora jax"
    "fiqa jax"
    "scidocs jax"
    "trec-covid jax"
    "webis-touche2020 jax"
    "nq jax"
    "hotpotqa jax"
    "fever jax"
)

for combo in "${combinations[@]}"; do
    read -r dataset backend <<< "$combo"
    echo "Running: $dataset with $backend"
    python benchmark_bm25s.py --dataset "$dataset" --backend "$backend"
done

deactivate

