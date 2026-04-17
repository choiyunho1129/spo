#!/usr/bin/env python3

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq


ANSWER_FORMAT = "\nThe answer format must be: \\boxed{'The final answer goes here.'}"


def prompt_to_input(prompt: str) -> str:
    return f"user\n{prompt}{ANSWER_FORMAT}\nassistant\n"


def read_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    if not path or not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def load_input_counts(path: str) -> Counter:
    counts: Counter = Counter()
    for row in read_jsonl(path):
        input_text = row.get("input")
        if isinstance(input_text, str):
            counts[input_text] += 1
    return counts


def cmd_prepare(args: argparse.Namespace) -> int:
    subset_table = pq.read_table(args.subset_file)
    if "prompt" not in subset_table.column_names:
        print(
            f"[ERROR] Missing 'prompt' column in {args.subset_file}. "
            f"Available columns: {subset_table.column_names}",
            file=sys.stderr,
        )
        return 2

    prompt_column = subset_table.column("prompt")
    existing_counts = load_input_counts(args.existing_jsonl)

    input_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx in range(subset_table.num_rows):
        prompt = prompt_column[idx].as_py()
        input_key = prompt_to_input(prompt)
        input_to_indices[input_key].append(idx)

    take_indices: list[int] = []
    for input_key, row_indices in input_to_indices.items():
        have = existing_counts.get(input_key, 0)
        missing = max(args.target_per_input - have, 0)
        # If duplicated prompts exist, distribute missing samples across duplicated rows.
        for j in range(missing):
            take_indices.append(row_indices[j % len(row_indices)])

    if not take_indices:
        print(
            f"[PREPARE] subset={args.subset_file} already complete "
            f"(prompts={len(input_to_indices)}, target={args.target_per_input})"
        )
        return 3

    output_table = subset_table.take(pa.array(take_indices, type=pa.int64()))
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pq.write_table(output_table, args.output_file)

    print(
        f"[PREPARE] subset={args.subset_file} prompts={len(input_to_indices)} "
        f"existing_rows={sum(existing_counts.values())} missing_rows={len(take_indices)} "
        f"output={args.output_file}"
    )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    base_records = read_jsonl(args.base_jsonl)
    incremental_records = read_jsonl(args.incremental_jsonl)

    counts: Counter = Counter()
    merged_records: list[dict] = []

    dropped_base = 0
    for row in base_records:
        input_text = row.get("input")
        if not isinstance(input_text, str):
            continue
        if counts[input_text] < args.target_per_input:
            merged_records.append(row)
            counts[input_text] += 1
        else:
            dropped_base += 1

    added_incremental = 0
    skipped_incremental = 0
    for row in incremental_records:
        input_text = row.get("input")
        if not isinstance(input_text, str):
            continue
        if counts[input_text] < args.target_per_input:
            merged_records.append(row)
            counts[input_text] += 1
            added_incremental += 1
        else:
            skipped_incremental += 1

    write_jsonl(args.base_jsonl, merged_records)

    count_values = list(counts.values())
    min_count = min(count_values) if count_values else 0
    max_count = max(count_values) if count_values else 0
    underfilled = sum(1 for x in count_values if x < args.target_per_input)
    print(
        f"[MERGE] base={args.base_jsonl} kept={len(merged_records)} "
        f"dropped_base={dropped_base} added_incremental={added_incremental} "
        f"skipped_incremental={skipped_incremental} prompts={len(counts)} "
        f"min_per_input={min_count} max_per_input={max_count} underfilled={underfilled}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental utility for SPO offline-value subset evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Create a parquet with only missing samples needed to reach target trajectories per input."
    )
    prepare_parser.add_argument("--subset-file", type=str, required=True)
    prepare_parser.add_argument("--existing-jsonl", type=str, required=True)
    prepare_parser.add_argument("--target-per-input", type=int, default=4)
    prepare_parser.add_argument("--output-file", type=str, required=True)
    prepare_parser.set_defaults(func=cmd_prepare)

    merge_parser = subparsers.add_parser(
        "merge", help="Merge incremental JSONL into base JSONL while keeping at most target trajectories per input."
    )
    merge_parser.add_argument("--base-jsonl", type=str, required=True)
    merge_parser.add_argument("--incremental-jsonl", type=str, required=True)
    merge_parser.add_argument("--target-per-input", type=int, default=4)
    merge_parser.set_defaults(func=cmd_merge)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
