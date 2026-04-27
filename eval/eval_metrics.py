"""
Compute validation metrics from eval JSONL output.

Replicates the metric logic from RayPPOTrainer._validate() / process_validation_metrics().

Metrics computed per data_source:
  - mean@N / mean@N/std (SEM)
  - maj@N  / maj@N/std  (SEM)
  - best@N / best@N/std (SEM)

Usage:
    python eval_metrics.py --input results/model.jsonl --n 8
    python eval_metrics.py --input results/model.jsonl --n 8 --output results/model_metrics.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def compute_group_metrics(correctness_list):
    arr = np.array(correctness_list, dtype=np.float32)
    n = len(arr)
    return {
        "mean": float(arr.mean()),
        "maj":  float(arr.sum() > n / 2),   # strict majority
        "best": float(arr.sum() > 0),        # any correct
    }


def process_metrics(records, n):
    """
    Group records by (data_source, prompt) and compute mean/maj/best@N + SEM.
    std = std / sqrt(n_prompts)  →  Standard Error of Mean (matches wandb trainer values)
    """
    ds_prompt_correct = defaultdict(lambda: defaultdict(list))

    for rec in records:
        ds      = rec.get("data_source", "unknown")
        prompt  = rec.get("prompt", "")
        correct = int(rec.get("correctness", 0))
        ds_prompt_correct[ds][prompt].append(correct)

    results = {}
    for ds, prompt_map in sorted(ds_prompt_correct.items()):
        mean_accs, maj_accs, best_accs = [], [], []
        for corr_list in prompt_map.values():
            g = compute_group_metrics(corr_list)
            mean_accs.append(g["mean"])
            maj_accs.append(g["maj"])
            best_accs.append(g["best"])

        mean_arr = np.array(mean_accs, dtype=np.float32) if mean_accs else np.zeros(1)
        maj_arr  = np.array(maj_accs,  dtype=np.float32) if maj_accs  else np.zeros(1)
        best_arr = np.array(best_accs, dtype=np.float32) if best_accs else np.zeros(1)

        n_prompts = len(mean_accs)
        sqrt_n = np.sqrt(max(n_prompts, 1))

        results[ds] = {
            f"mean@{n}":      float(mean_arr.mean()),
            f"mean@{n}/std":  float(mean_arr.std() / sqrt_n),   # SEM
            f"maj@{n}":       float(maj_arr.mean()),
            f"maj@{n}/std":   float(maj_arr.std() / sqrt_n),    # SEM
            f"best@{n}":      float(best_arr.mean()),
            f"best@{n}/std":  float(best_arr.std() / sqrt_n),   # SEM
            "prompt_count":   n_prompts,
            "response_count": sum(len(v) for v in prompt_map.values()),
        }

    return results


# ---------------------------------------------------------------------------
# Trainer-style metric key formatter
# ---------------------------------------------------------------------------

def format_trainer_metrics(per_ds_metrics, n):
    """
    Format metrics to match the trainer's wandb logging format:
        val-core/{data_source}/acc/mean@N
        val-core/{data_source}/acc/mean@N/std  (SEM)
        val-core/{data_source}/acc/maj@N
        val-core/{data_source}/acc/maj@N/std   (SEM)
        val-core/{data_source}/acc/best@N
        val-core/{data_source}/acc/best@N/std  (SEM)
        val-aux/{data_source}/acc/prompt_count
        val-aux/{data_source}/acc/response_count
    """
    core_keys = {
        f"mean@{n}", f"mean@{n}/std",
        f"maj@{n}",  f"maj@{n}/std",
        f"best@{n}", f"best@{n}/std",
    }
    metric_dict = {}
    for ds, m in per_ds_metrics.items():
        for key, val in m.items():
            section = "val-core" if key in core_keys else "val-aux"
            metric_dict[f"{section}/{ds}/acc/{key}"] = val
    return metric_dict


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(per_ds_metrics, n, trainer_metrics):
    print("\n" + "=" * 65)
    print("EVALUATION METRICS")
    print("=" * 65)

    for ds, m in per_ds_metrics.items():
        print(f"\n[{ds}]")
        print(f"  Prompts   : {m['prompt_count']}")
        print(f"  Responses : {m['response_count']}")
        print(f"  mean@{n:<3}  : {m[f'mean@{n}']:.4f}  (sem={m[f'mean@{n}/std']:.4f})")
        print(f"  maj@{n:<4}  : {m[f'maj@{n}']:.4f}  (sem={m[f'maj@{n}/std']:.4f})")
        print(f"  best@{n:<3}  : {m[f'best@{n}']:.4f}  (sem={m[f'best@{n}/std']:.4f})")

    print("\n" + "-" * 65)
    print("Trainer-style metric keys (val-core only):")
    for k, v in sorted(trainer_metrics.items()):
        if "val-core" in k:
            print(f"  {k}: {v:.4f}")

    all_mean = [m[f"mean@{n}"] for m in per_ds_metrics.values()]
    all_maj  = [m[f"maj@{n}"]  for m in per_ds_metrics.values()]
    all_best = [m[f"best@{n}"] for m in per_ds_metrics.values()]
    print("\n" + "-" * 65)
    print("Overall (macro avg across data sources):")
    print(f"  mean@{n} : {np.mean(all_mean):.4f}")
    print(f"  maj@{n}  : {np.mean(all_maj):.4f}")
    print(f"  best@{n} : {np.mean(all_best):.4f}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute val metrics from eval JSONL.")
    parser.add_argument("--input",  "-i", required=True, help="Path to eval JSONL file")
    parser.add_argument("--output", "-o", default=None,  help="(Optional) save metrics to JSON")
    parser.add_argument("--n",      "-n", type=int, default=8,
                        help="Number of responses per prompt (default: 8)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {input_path}")

    per_ds_metrics  = process_metrics(records, n=args.n)
    trainer_metrics = format_trainer_metrics(per_ds_metrics, n=args.n)

    print_results(per_ds_metrics, n=args.n, trainer_metrics=trainer_metrics)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "per_data_source": per_ds_metrics,
            "trainer_style":   trainer_metrics,
            "n": args.n,
            "total_records": len(records),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()