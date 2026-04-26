# 기본 (n=8이 default)
# python eval_metrics.py --input results/qwen2.5-0.5B-Instruct.jsonl
python3 eval_metrics.py --input results/qwen3-4b/qwen3-4b_AIME_2024.jsonl
python3 eval_metrics.py --input results/qwen3-4b/qwen3-4b_AIME_2025.jsonl
# n 명시
# python eval_metrics.py --input results/qwen2.5-0.5B-Instruct.jsonl --n 8

# 결과 JSON으로 저장도 같이
# python eval_metrics.py --input results/qwen2.5-0.5B-Instruct.jsonl --n 8 --output results/qwen2.5-0.5B-Instruct_metrics.json