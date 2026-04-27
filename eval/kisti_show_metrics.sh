#!/bin/bash

export MY_DIR="/scratch/x3326a36"
export SIF_PATH="$MY_DIR/verl_spo.sif"
export PYTHONUSERBASE="$MY_DIR/.local"
export HF_HOME="$MY_DIR/.cache/huggingface"
export XDG_CACHE_HOME="$MY_DIR/.cache"
export TMPDIR="$MY_DIR/tmp"
export FLASHINFER_CACHE_DIR="$MY_DIR/.cache/flashinfer"

module purge
module load singularity

MODEL_NAME=${1:-"qwen3-4b_test"}
ROOT="/scratch/x3326a36/spo/eval"
RESULTS_DIR="$ROOT/results/$MODEL_NAME"

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
    INPUT="$RESULTS_DIR/${MODEL_NAME}_${DATASET}.jsonl"

    if [ ! -f "$INPUT" ]; then
        echo "[SKIP] $INPUT not found"
        continue
    fi

    echo "[RUN] $DATASET"

    singularity exec --nv \
        --bind $MY_DIR \
        --bind $MY_DIR/.cache/flashinfer:/home01/x3326a36/.cache/flashinfer \
        $SIF_PATH \
        env PYTHONUSERBASE=$PYTHONUSERBASE \
            FLASHINFER_CACHE_DIR="$FLASHINFER_CACHE_DIR" \
        python3 "$ROOT/eval_metrics.py" --input "$INPUT"

    echo "[DONE] $DATASET"
    echo ""
done
