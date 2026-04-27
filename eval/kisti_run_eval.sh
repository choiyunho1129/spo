#!/bin/bash
#SBATCH -J myapp
#SBATCH -p amd_a100nv_8
#SBATCH --comment etc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=awj1204@snu.ac.kr

# ==========================================
# 0. 환경 및 경로 설정 (절대 경로 추천)
# ==========================================
export MY_DIR="/scratch/x3326a36"
export SIF_PATH="$MY_DIR/verl_spo.sif"
export PYTHONUSERBASE="$MY_DIR/.local"

# 🛑 홈 디렉토리 쿼터 에러 방지를 위한 경로 변경
export HF_HOME="$MY_DIR/.cache/huggingface"
export XDG_CACHE_HOME="$MY_DIR/.cache"
export TMPDIR="$MY_DIR/tmp"
export VLLM_NO_USAGE_STATS=1
export FLASHINFER_CACHE_DIR="$MY_DIR/.cache/flashinfer"
mkdir -p "$FLASHINFER_CACHE_DIR"


mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

# 서버 모듈 로드
module purge
module load singularity


# ============================================
# 모델 선택 (둘 중 하나 선택)
# ============================================
# MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
# MODEL_NAME=qwen2.5-0.5B-Instruct

MODEL_PATH=Qwen/Qwen3-4B
MODEL_NAME=qwen3-4b_test


# 작업 루트 경로
ROOT="/scratch/x3326a36/spo/eval"
DATA_DIR="$ROOT/data"
OUTPUT_DIR="$ROOT/results/$MODEL_NAME"
PYTHON_SCRIPT="$ROOT/eval_vllm.py"

mkdir -p "$OUTPUT_DIR"

# ============================================
# 평가 설정
# ============================================
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=-1
MAX_TOKENS=8192
N=8

# ============================================
# 데이터셋 목록 (추가 시 여기에만 추가)
# ============================================
DATASETS=(
    "$DATA_DIR/AIME_2024.parquet"
    "$DATA_DIR/AIME_2025.parquet"
    "$DATA_DIR/AMC23_test.parquet"
    "$DATA_DIR/AMC24_test.parquet"
    "$DATA_DIR/brumo_2025.parquet"
    "$DATA_DIR/hmmt25.parquet"
)

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
# 데이터셋별 평가 실행
# ============================================
for DATA in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename "$DATA" .parquet)
    OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME}_${DATASET_NAME}.jsonl"
    LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_${DATASET_NAME}.log"

    echo "======================================"
    echo "🚀 Evaluating: $DATASET_NAME"
    echo "📍 Model: $MODEL_PATH"
    echo "📍 Data: $DATA"
    echo "📍 Output: $OUTPUT_FILE"
    echo "======================================"

    # 데이터셋 존재 여부 확인
    if [ ! -f "$DATA" ]; then
        echo "⚠️  WARNING: 데이터셋을 찾을 수 없습니다: $DATA"
        echo "스킵합니다..."
        echo ""
        continue
    fi

    # GPU 할당 및 Singularity 실행
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 singularity exec --nv \
    --bind $MY_DIR \
    --bind $MY_DIR/.cache/flashinfer:/home01/x3326a36/.cache/flashinfer \
    $SIF_PATH \
    env PYTHONUSERBASE=$PYTHONUSERBASE \
        FLASHINFER_CACHE_DIR="$MY_DIR/.cache/flashinfer" \
    python3 "$PYTHON_SCRIPT" \
            --input_file "$DATA" \
            --output_file "$OUTPUT_FILE" \
            --model_path "$MODEL_PATH" \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --top_k $TOP_K \
            --max_tokens $MAX_TOKENS \
            --n $N \
            2>&1 | tee "$LOG_FILE"

    echo "======================================"
    echo "✅ Done: $DATASET_NAME"
    echo "======================================"
    echo ""
done

echo "======================================"
echo "✅ All Evaluations Complete"
echo "======================================"
