#!/bin/bash

MODEL=${1:-"qwen3-4b_test"}
RESULTS_DIR="results/${MODEL}"

DATASETS=(
    "AIME_2024"
    "AIME_2025"
    "AIME_2026"
    "AMC23_test"
    "AMC24_test"
    "brumo_2025"
    "hmmt25"
)

for DATASET in "${DATASETS[@]}"; do
    INPUT="${RESULTS_DIR}/${MODEL}_${DATASET}.jsonl"

    if [ ! -f "$INPUT" ]; then
        echo "[SKIP] $INPUT not found"
        continue
    fi

    echo "[RUN] $INPUT"
    python3 eval_metrics.py --input "$INPUT"
done
