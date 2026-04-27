#!/bin/bash

# ==========================================
# 0. 환경 및 경로 설정 (절대 경로 추천)
# ==========================================
export MY_DIR="/scratch/x3326a36"
export SIF_PATH="$MY_DIR/verl_spo.sif"
export PYTHONUSERBASE="$MY_DIR/.local"

# 🛑 [추가] 홈 디렉토리 쿼터 에러 방지를 위한 경로 변경
export HF_HOME="$MY_DIR/.cache/huggingface"
export XDG_CACHE_HOME="$MY_DIR/.cache"
export TMPDIR="$MY_DIR/tmp"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

# 서버 모듈 로드
module purge
module load singularity

# 작업 루트 경로 (eval_vllm.py가 있는 위치)
# 현재 스크립트 위치가 /scratch/x3326a36/onscripts 라면 아래와 같이 설정
ROOT="/scratch/x3326a36/spo/eval"
DATA="$ROOT/data/AIME_2025.parquet"
OUTPUT_DIR="$ROOT/results2"
PYTHON_SCRIPT="$ROOT/eval_vllm.py"

mkdir -p "$OUTPUT_DIR"

# 모델 및 파라미터 설정
MODEL_PATH="Qwen/Qwen3-4B"
MODEL_NAME="qwen3-4B"
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=-1
MAX_TOKENS=8192
N=8

# ============================================
# 실행 전 체크
# ============================================
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ ERROR: 파이썬 스크립트를 찾을 수 없습니다: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -f "$SIF_PATH" ]; then
    echo "❌ ERROR: Singularity 이미지가 없습니다: $SIF_PATH"
    exit 1
fi

# ============================================
# 평가 실행 (Singularity)
# ============================================

echo "======================================"
echo "🚀 Evaluation 시작 (Singularity 내부)"
echo "======================================"
echo "📍 데이터: $DATA"
echo "📍 모델: $MODEL_PATH"
echo "📍 로그: $OUTPUT_DIR/$MODEL_NAME.log"
echo "======================================"

# GPU 할당 및 Singularity 실행
# env 명령어를 통해 컨테이너 내부로 변수들을 명확히 전달합니다.
CUDA_VISIBLE_DEVICES=0,1,2,3 singularity exec --nv \
    --bind $MY_DIR \
    $SIF_PATH \
    env PYTHONUSERBASE=$PYTHONUSERBASE \
    python3 "$PYTHON_SCRIPT" \
        --input_file "$DATA" \
        --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
        --model_path "$MODEL_PATH" \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --top_k $TOP_K \
        --max_tokens $MAX_TOKENS \
        --n $N \
        2>&1 | tee "$OUTPUT_DIR/$MODEL_NAME.log"

echo "======================================"
echo "✅ Evaluation 완료"
echo "======================================"
