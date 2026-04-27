#!/bin/bash

# 모델 선택
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
MODEL_NAME=qwen2.5-0.5B-Instruct

# 모델 선택
MODEL_PATH=Qwen/Qwen3-4B
MODEL_NAME=qwen3-4b

# 평가 설정
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=-1
MAX_TOKENS=8192
N=8

ROOT=${ROOT:-./}
OUTPUT_DIR=./results/$MODEL_NAME
mkdir -p $OUTPUT_DIR
# ============================================
# 데이터셋 목록 (추가 시 여기에만 추가)
# ============================================
DATASETS=(
    "$ROOT/data/AIME_2024.parquet"
    "$ROOT/data/AIME_2025.parquet"
    "$ROOT/data/AIME_2026.parquet"
    "$ROOT/data/AMC23_test.parquet"
    "$ROOT/data/AMC24_test.parquet"
    "$ROOT/data/brumo_2025.parquet"
    "$ROOT/data/hmmt25.parquet"
)

# ============================================
# 평가 실행
# ============================================
for DATA in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename $DATA .parquet)
    OUTPUT_FILE=$OUTPUT_DIR/${MODEL_NAME}_${DATASET_NAME}.jsonl

    echo "======================================"
    echo "Evaluating: $DATASET_NAME"
    echo "Model: $MODEL_PATH"
    echo "Output: $OUTPUT_FILE"
    echo "======================================"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval_vllm.py \
        --input_file $DATA \
        --output_file $OUTPUT_FILE \
        --model_path $MODEL_PATH \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --top_k $TOP_K \
        --max_tokens $MAX_TOKENS \
        --n $N \
        2>&1 | tee $OUTPUT_DIR/${MODEL_NAME}_${DATASET_NAME}.log

    echo "======================================"
    echo "Done: $DATASET_NAME"
    echo "======================================"
done

echo "======================================"
echo "All Evaluations Complete"
echo "======================================"
