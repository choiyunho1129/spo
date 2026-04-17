#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-spo_verl_pr}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/DAPO-Math-17k-Processed_Splits}"
TARGET_TRAJECTORIES="${TARGET_TRAJECTORIES:-4}"
SUBSET_START="${SUBSET_START:-}"
SUBSET_END="${SUBSET_END:-}"
INCREMENTAL_UTIL="${SCRIPT_DIR}/incremental_eval.py"

cd "${REPO_ROOT}"

mapfile -t subset_files < <(find "${DATA_DIR}" -maxdepth 1 -type f -name 'subset_*.parquet' | sort -V)

if [[ ${#subset_files[@]} -eq 0 ]]; then
    echo "[ERROR] No subset parquet files found in ${DATA_DIR}"
    exit 1
fi

failed_subsets=()
processed_count=0
skip_count=0

for data_file in "${subset_files[@]}"; do
    subset_name="$(basename "${data_file}")"
    subset_id="${subset_name#subset_}"
    subset_id="${subset_id%.parquet}"

    if [[ ! "${subset_id}" =~ ^[0-9]+$ ]]; then
        echo "[WARN] Skip non-numeric subset file: ${subset_name}"
        continue
    fi
    if [[ -n "${SUBSET_START}" && "${subset_id}" -lt "${SUBSET_START}" ]]; then
        continue
    fi
    if [[ -n "${SUBSET_END}" && "${subset_id}" -gt "${SUBSET_END}" ]]; then
        continue
    fi

    processed_count=$((processed_count + 1))

    # subset 0, 1 only -> target trajectories = 32
    if [[ "${subset_id}" == "0" || "${subset_id}" == "1" ]]; then
        run_target_trajectories=32
    else
        run_target_trajectories="${TARGET_TRAJECTORIES}"
    fi

    base_exp="offline_value_estimation_subset_${subset_id}"
    base_jsonl="${OUTPUT_DIR}/spo/${base_exp}/validation_data/0.jsonl"

    run_data_file="${data_file}"
    run_n_val="${run_target_trajectories}"
    run_exp="${base_exp}"
    merge_from=""
    temp_subset_file=""

    if [[ -f "${base_jsonl}" ]]; then
        temp_subset_file="$(mktemp "/tmp/subset_${subset_id}_missing_XXXX.parquet")"
        set +e
        python3 "${INCREMENTAL_UTIL}" prepare \
            --subset-file "${data_file}" \
            --existing-jsonl "${base_jsonl}" \
            --target-per-input "${run_target_trajectories}" \
            --output-file "${temp_subset_file}"
        prepare_rc=$?
        set -e

        if [[ ${prepare_rc} -eq 3 ]]; then
            echo "[SKIP] subset_${subset_id}: already has >= ${run_target_trajectories} trajectories per input."
            rm -f "${temp_subset_file}"
            skip_count=$((skip_count + 1))
            continue
        elif [[ ${prepare_rc} -ne 0 ]]; then
            echo "[ERROR] subset_${subset_id}: failed to prepare incremental parquet."
            rm -f "${temp_subset_file}"
            failed_subsets+=("${subset_id}")
            continue
        fi

        run_data_file="${temp_subset_file}"
        run_n_val=1
        run_exp="${base_exp}_append_$(date +%Y%m%d_%H%M%S)"
        merge_from="${OUTPUT_DIR}/spo/${run_exp}/validation_data/0.jsonl"
    fi

    echo "[RUN] subset_${subset_id} -> ${run_exp} (TARGET=${run_target_trajectories}, N_VAL=${run_n_val})"
    set +e
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DATA_FILE="${run_data_file}" \
    MODEL_PATH="${MODEL_PATH}" \
    EXP_NAME="${run_exp}" \
    N_VAL="${run_n_val}" \
    bash "${SCRIPT_DIR}/eval.sh"
    eval_rc=$?
    set -e

    if [[ -n "${temp_subset_file}" ]]; then
        rm -f "${temp_subset_file}"
    fi

    if [[ ${eval_rc} -ne 0 ]]; then
        echo "[ERROR] subset_${subset_id}: eval failed."
        failed_subsets+=("${subset_id}")
        continue
    fi

    if [[ -n "${merge_from}" ]]; then
        if [[ ! -f "${merge_from}" ]]; then
            echo "[ERROR] subset_${subset_id}: incremental output not found: ${merge_from}"
            failed_subsets+=("${subset_id}")
            continue
        fi
        python3 "${INCREMENTAL_UTIL}" merge \
            --base-jsonl "${base_jsonl}" \
            --incremental-jsonl "${merge_from}" \
            --target-per-input "${run_target_trajectories}"
    fi
done

if [[ ${processed_count} -eq 0 ]]; then
    echo "[WARN] No subsets matched SUBSET_START/SUBSET_END filters."
fi

if [[ ${#failed_subsets[@]} -gt 0 ]]; then
    echo "[FAILED] subset(s): ${failed_subsets[*]}"
    exit 1
fi

echo "[DONE] processed=${processed_count}, skipped=${skip_count}, default_target=${TARGET_TRAJECTORIES} (subset_0,1=32)"