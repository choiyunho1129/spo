#!/bin/bash

# eval.sh - VERL val_dataloader 방식의 평가 스크립트

# ============================================
# 사용자 설정
# ============================================

ROOT=${ROOT:-./}
# DATA=${DATA:-$ROOT/data/AIME_2025_verl_format.parquet}
# DATA=${DATA:-$ROOT/data/AIME_2025.parquet}
# DATA=${DATA:-$ROOT/data/gsm8k.parquet}
DATA=${DATA:-$ROOT/data/AIME_2025.parquet}
# DATA=${DATA:-$ROOT/data/math500.parquet}
# DATA=${DATA:-$ROOT/data/hmmt25.parquet}
OUTPUT_DIR=./results/
mkdir -p $OUTPUT_DIR

# 모델 선택
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
# MODEL_PATH=Qwen/Qwen3-4B
MODEL_NAME=qwen2.5-0.5B-Instruct
# MODEL_NAME=qwen3-4B-tmp
# 평가 설정 (crrl.sh와 동일)
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=-1
MAX_TOKENS=8192
N=8

# ============================================
# 평가 실행
# ============================================

echo "======================================"
echo "Starting Evaluation"
echo "======================================"
echo "Data: $DATA"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR/$MODEL_NAME.jsonl"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Top-k: $TOP_K"
echo "max_tokens: $MAX_TOKENS"
echo "N: $N"
echo "======================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval_vllm.py \
  --input_file $DATA \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --model_path $MODEL_PATH \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --top_k $TOP_K \
  --max_tokens $MAX_TOKENS \
  --n $N \
  2>&1 | tee $OUTPUT_DIR/$MODEL_NAME.log

echo "======================================"
echo "Evaluation Complete"
echo "Results: $OUTPUT_DIR/$MODEL_NAME.jsonl"
echo "Log: $OUTPUT_DIR/$MODEL_NAME.log"
echo "======================================"
