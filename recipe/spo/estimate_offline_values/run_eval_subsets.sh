#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-spo_verl_pr}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/DAPO-Math-17k-Processed_Splits}"

cd "${REPO_ROOT}"

for i in $(seq 2 14); do
    data_file="${DATA_DIR}/subset_${i}.parquet"
    exp_name="offline_value_estimation_subset_${i}"

    if [[ ! -f "${data_file}" ]]; then
        echo "[ERROR] Missing file: ${data_file}"
        exit 1
    fi

    echo "[RUN] subset_${i} -> ${exp_name}"
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DATA_FILE="${data_file}" \
    MODEL_PATH="${MODEL_PATH}" \
    EXP_NAME="${exp_name}" \
    bash "${SCRIPT_DIR}/eval.sh"
done

echo "[DONE] Completed subset_2 through subset_14"
