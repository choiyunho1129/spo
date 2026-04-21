#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add local estimator modules to import path.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from single_trajectory_estimator_support.feature_builder.builder import (  # noqa: E402
    SingleTrajectoryFeatureBuilder,
)
from single_trajectory_estimator_support.feature_builder.config import (  # noqa: E402
    FeatureBuilderConfig,
)
from single_trajectory_estimator_support.feature_builder.features import (  # noqa: E402
    extract_reasoning_and_answer,
)
from single_trajectory_estimator_support.value_estimator.training import (  # noqa: E402
    SingleTrajectoryEstimatorFitConfig,
    build_training_matrix,
    fit_single_trajectory_estimator,
    save_single_trajectory_estimator_bundle,
)

THINK_CONTENT_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_CONTENT_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class RolloutRecord:
    subset_id: int
    prompt: str
    output: str
    score: float


@dataclass(frozen=True)
class PromptPair:
    subset_id: int
    prompt: str
    selected_rollouts: tuple[RolloutRecord, RolloutRecord]
    avg_score: float


@dataclass(frozen=True)
class RolloutFailure:
    subset_id: int
    prompt_idx: int
    stage: str
    reason: str
    detail: str


class LayerCapture:
    def __init__(self, model: torch.nn.Module, layer_indices: Iterable[int]) -> None:
        self.outputs: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        layers = resolve_transformer_layers(model)
        for layer_idx in sorted(set(int(i) for i in layer_indices)):
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"Invalid layer index {layer_idx} for model with {len(layers)} layers.")
            handle = layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self.outputs[layer_idx] = hidden.detach()

        return _hook

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.outputs.clear()


def resolve_transformer_layers(model: torch.nn.Module) -> Any:
    base_model = model.module if hasattr(model, "module") else model
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model.layers
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        return base_model.transformer.h
    if hasattr(base_model, "gpt_neox") and hasattr(base_model.gpt_neox, "layers"):
        return base_model.gpt_neox.layers
    raise ValueError(
        "Could not resolve transformer layer stack. "
        "Expected one of model.layers / transformer.h / gpt_neox.layers."
    )


def parse_args() -> argparse.Namespace:
    default_feature_cfg = THIS_DIR / "single_trajectory_estimator_support/default_feature_builder_config.json"
    default_fit_cfg = THIS_DIR / "single_trajectory_estimator_support/default_estimator_fit_config.json"
    default_output_model = THIS_DIR / "artifacts/qwen3_4b_subset3_6_pairavg_estimator.joblib"

    parser = argparse.ArgumentParser(
        description=(
            "Train single-trajectory estimator with prompt-wise pair-average targets from "
            "offline_value_estimation_subset_{3,4,5,6}/validation_data/0.jsonl"
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data1/home/yunhochoi/verl/crrl_verl_pr/crrl"),
        help="Root directory containing offline_value_estimation_subset_*/validation_data/0.jsonl.",
    )
    parser.add_argument(
        "--subset-ids",
        type=str,
        default="3,4,5,6",
        help="Comma-separated subset IDs.",
    )
    parser.add_argument(
        "--rollouts-per-prompt",
        type=int,
        default=2,
        help="Number of rollouts selected per prompt. For current setup this should be 2.",
    )
    parser.add_argument(
        "--pair-selection",
        choices=["random", "first_two"],
        default="random",
        help="How to select 2 rollouts among 4 per prompt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for rollout selection.")
    parser.add_argument(
        "--prompt-selection",
        choices=["first", "random"],
        default="first",
        help="How to select prompts when --max-prompts is set.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HF model path for teacher-forcing feature extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g., cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to trust remote code when loading tokenizer/model.",
    )
    parser.add_argument(
        "--feature-builder-config",
        type=Path,
        default=default_feature_cfg,
        help="Path to feature builder config JSON.",
    )
    parser.add_argument(
        "--fit-config",
        type=Path,
        default=default_fit_cfg,
        help="Path to estimator fit config JSON.",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=default_output_model,
        help="Output path for estimator bundle (.joblib).",
    )
    parser.add_argument(
        "--output-meta-path",
        type=Path,
        default=None,
        help="Optional output metadata path. Defaults to <output_model_path>.meta.json",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap on prompt count for quick tests.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Log progress every N prompts.",
    )
    parser.add_argument(
        "--error-log-limit",
        type=int,
        default=20,
        help="Maximum number of detailed rollout extraction errors to print.",
    )
    return parser.parse_args()


def parse_subset_ids(subset_ids_raw: str) -> list[int]:
    parsed = []
    for item in subset_ids_raw.split(","):
        token = item.strip()
        if not token:
            continue
        parsed.append(int(token))
    if not parsed:
        raise ValueError("No subset IDs were provided.")
    return parsed


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32

    if dtype_arg != "auto":
        raise ValueError(f"Unsupported dtype: {dtype_arg}")

    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_pairs(
    *,
    data_root: Path,
    subset_ids: list[int],
    rollouts_per_prompt: int,
    pair_selection: str,
    prompt_selection: str,
    seed: int,
    max_prompts: int | None,
) -> list[PromptPair]:
    rng = random.Random(seed)
    pairs: list[PromptPair] = []

    for subset_id in subset_ids:
        jsonl_path = data_root / f"offline_value_estimation_subset_{subset_id}/validation_data/0.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing JSONL: {jsonl_path}")

        prompt2rows: dict[str, list[RolloutRecord]] = {}
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = str(row.get("input", ""))
                output = str(row.get("output", ""))
                if not prompt:
                    raise ValueError(f"Missing 'input' at {jsonl_path}:{line_idx}")
                if not output:
                    raise ValueError(f"Missing 'output' at {jsonl_path}:{line_idx}")
                if "score" in row:
                    score = float(row["score"])
                elif "reward" in row:
                    score = float(row["reward"])
                else:
                    raise ValueError(f"Missing score/reward at {jsonl_path}:{line_idx}")

                prompt2rows.setdefault(prompt, []).append(
                    RolloutRecord(subset_id=subset_id, prompt=prompt, output=output, score=score)
                )

        for prompt in sorted(prompt2rows.keys()):
            rows = prompt2rows[prompt]
            if len(rows) < rollouts_per_prompt:
                continue
            if pair_selection == "first_two":
                selected = rows[:rollouts_per_prompt]
            else:
                selected = rng.sample(rows, k=rollouts_per_prompt)

            if len(selected) != 2:
                raise ValueError(
                    f"This training script currently expects exactly 2 selected rollouts, got {len(selected)}"
                )

            avg_score = float((selected[0].score + selected[1].score) / 2.0)
            pairs.append(
                PromptPair(
                    subset_id=subset_id,
                    prompt=prompt,
                    selected_rollouts=(selected[0], selected[1]),
                    avg_score=avg_score,
                )
            )

    if max_prompts is not None:
        max_prompts = max(0, int(max_prompts))
        if prompt_selection == "random":
            if max_prompts < len(pairs):
                pairs = rng.sample(pairs, k=max_prompts)
        else:
            pairs = pairs[:max_prompts]

    return pairs


def find_reasoning_answer_spans(text: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    reasoning_span = None
    answer_span = None

    think_match = THINK_CONTENT_PATTERN.search(text)
    if think_match:
        reasoning_span = (think_match.start(1), think_match.end(1))

    answer_match = ANSWER_CONTENT_PATTERN.search(text)
    if answer_match:
        answer_span = (answer_match.start(1), answer_match.end(1))
    elif "</think>" in text:
        start = text.find("</think>") + len("</think>")
        end = len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            answer_span = (start, end)

    return reasoning_span, answer_span


def span_to_token_indices(offsets: list[tuple[int, int]], span: tuple[int, int] | None) -> list[int]:
    if span is None:
        return []
    span_start, span_end = span
    indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= span_start or tok_start >= span_end:
            continue
        if tok_end > tok_start:
            indices.append(idx)
    return indices


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]

    normalized_offsets = [(int(start), int(end)) for start, end in offsets]
    return [int(tid) for tid in token_ids], normalized_offsets


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def mean_at_indices(values: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    clipped = [idx for idx in indices if 0 <= idx < values.shape[0]]
    if not clipped:
        return 0.0
    return float(values[clipped].mean())


def classify_exception(exc: Exception) -> tuple[str, str]:
    message = str(exc).strip().replace("\n", " ")
    lowered = message.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "oom", message
    if "token indices sequence length" in lowered:
        return "seq_len", message
    if "position embeddings" in lowered or "max_position_embeddings" in lowered:
        return "seq_len", message
    if "index out of range in self" in lowered and "position" in lowered:
        return "seq_len", message
    if "hook output missing" in lowered:
        return "hook", message
    return type(exc).__name__, message


def fallback_segment_indices(tokenizer: Any, generated_text: str, total_tokens: int) -> tuple[list[int], list[int]]:
    _, reasoning_text, answer_text = extract_reasoning_and_answer({"generated_text": generated_text})
    reasoning_count = len(tokenizer.encode(reasoning_text, add_special_tokens=False)) if reasoning_text else 0
    answer_count = len(tokenizer.encode(answer_text, add_special_tokens=False)) if answer_text else 0

    reasoning_count = min(reasoning_count, total_tokens)
    answer_count = min(answer_count, total_tokens)

    reasoning_indices = list(range(reasoning_count))
    answer_indices = list(range(max(0, total_tokens - answer_count), total_tokens))
    return reasoning_indices, answer_indices


def compute_rollout_entropy_features(
    *,
    tokenizer: Any,
    generated_text: str,
    response_ids: list[int],
    response_entropies: np.ndarray,
) -> dict[str, float]:
    output_mean = float(response_entropies.mean()) if response_entropies.size > 0 else 0.0
    reasoning_mean = 0.0
    answer_mean = 0.0

    reasoning_indices: list[int] = []
    answer_indices: list[int] = []

    use_offset_fallback = False
    try:
        token_ids_by_offset, offsets = tokenize_with_offsets(tokenizer, generated_text)
        if len(token_ids_by_offset) != len(response_ids):
            use_offset_fallback = True
        else:
            reasoning_span, answer_span = find_reasoning_answer_spans(generated_text)
            reasoning_indices = span_to_token_indices(offsets, reasoning_span)
            answer_indices = span_to_token_indices(offsets, answer_span)
    except Exception:
        use_offset_fallback = True

    if use_offset_fallback:
        reasoning_indices, answer_indices = fallback_segment_indices(tokenizer, generated_text, len(response_ids))

    reasoning_mean = mean_at_indices(response_entropies, reasoning_indices)
    answer_mean = mean_at_indices(response_entropies, answer_indices)

    return {
        "output_mean_token_entropy": output_mean,
        "reasoning_mean_token_entropy": reasoning_mean,
        "answer_mean_token_entropy": answer_mean,
    }


def extract_rollout_features(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    layer_capture: LayerCapture,
    prompt_layer_index: int,
    response_layer_index: int,
    builder: SingleTrajectoryFeatureBuilder,
    prompt: str,
    output: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(output, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt token length is zero.")
    if not response_ids:
        raise ValueError("Response token length is zero.")

    full_ids = prompt_ids + response_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    layer_capture.clear()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    if prompt_layer_index not in layer_capture.outputs:
        raise RuntimeError(f"Hook output missing for prompt layer {prompt_layer_index}.")
    if response_layer_index not in layer_capture.outputs:
        raise RuntimeError(f"Hook output missing for response layer {response_layer_index}.")

    prompt_layer_hidden = layer_capture.outputs[prompt_layer_index][0]
    response_layer_hidden = layer_capture.outputs[response_layer_index][0]

    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    total_len = prompt_len + response_len

    if prompt_layer_hidden.shape[0] < total_len or response_layer_hidden.shape[0] < total_len:
        raise RuntimeError(
            "Hook output seq_len is shorter than input token length: "
            f"hidden={prompt_layer_hidden.shape[0]}/{response_layer_hidden.shape[0]}, total={total_len}"
        )

    # bfloat16 tensors cannot be converted to numpy directly in some environments.
    prompt_hidden_tokens = prompt_layer_hidden[:prompt_len].detach().to(torch.float32).cpu().numpy()
    response_hidden_tokens = (
        response_layer_hidden[prompt_len : prompt_len + response_len].detach().to(torch.float32).cpu().numpy()
    )

    logits = outputs.logits[0]
    if prompt_len <= 0:
        raise RuntimeError("prompt_len must be positive for causal entropy slice.")
    response_logits = logits[prompt_len - 1 : prompt_len - 1 + response_len]
    if response_logits.shape[0] != response_len:
        raise RuntimeError(
            f"Entropy logits length mismatch: expected {response_len}, got {response_logits.shape[0]}"
        )
    response_entropies = entropy_from_logits(response_logits).detach().cpu().numpy()

    rollout_features = compute_rollout_entropy_features(
        tokenizer=tokenizer,
        generated_text=output,
        response_ids=response_ids,
        response_entropies=response_entropies,
    )

    prompt_hidden, response_hidden, response_features = builder.build_inputs(
        prompt_hidden_layers=prompt_hidden_tokens,
        response_hidden_layers=response_hidden_tokens,
        generated_text=output,
        response_ids=response_ids,
        tokenizer=tokenizer,
        rollout_features=rollout_features,
    )
    return prompt_hidden, response_hidden, response_features


def main() -> None:
    args = parse_args()
    subset_ids = parse_subset_ids(args.subset_ids)
    if args.rollouts_per_prompt != 2:
        raise ValueError("This script currently supports rollouts_per_prompt=2 only.")

    feature_builder_payload = load_json(args.feature_builder_config)
    fit_payload = load_json(args.fit_config)
    feature_builder_config = FeatureBuilderConfig.from_dict(feature_builder_payload)
    fit_config = SingleTrajectoryEstimatorFitConfig(**fit_payload)
    builder = SingleTrajectoryFeatureBuilder(feature_builder_config)

    prompt_layer_index = int(feature_builder_config.prompt_hidden.layer_index)
    response_layer_index = int(feature_builder_config.response_hidden.layer_index)

    prompt_pairs = build_prompt_pairs(
        data_root=args.data_root,
        subset_ids=subset_ids,
        rollouts_per_prompt=args.rollouts_per_prompt,
        pair_selection=args.pair_selection,
        prompt_selection=args.prompt_selection,
        seed=args.seed,
        max_prompts=args.max_prompts,
    )
    if not prompt_pairs:
        raise ValueError("No prompt pairs were built from the provided subsets.")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, args.device)

    print(
        f"[INFO] Loading model/tokenizer: model={args.model_path}, device={device}, dtype={dtype}, "
        f"pairs={len(prompt_pairs)}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    layer_capture = LayerCapture(model, [prompt_layer_index, response_layer_index])

    prompt_hidden_rows: list[np.ndarray] = []
    response_hidden_rows: list[np.ndarray] = []
    response_feature_rows: list[dict[str, float]] = []
    targets: list[float] = []

    start_time = time.time()
    skipped_prompts = 0
    used_prompts = 0
    failure_counter: Counter[str] = Counter()
    failure_examples: list[RolloutFailure] = []

    try:
        for prompt_idx, pair in enumerate(prompt_pairs, start=1):
            pair_features = []
            pair_failed = False

            for rollout in pair.selected_rollouts:
                try:
                    feat = extract_rollout_features(
                        model=model,
                        tokenizer=tokenizer,
                        layer_capture=layer_capture,
                        prompt_layer_index=prompt_layer_index,
                        response_layer_index=response_layer_index,
                        builder=builder,
                        prompt=rollout.prompt,
                        output=rollout.output,
                        device=device,
                    )
                except Exception as exc:
                    pair_failed = True
                    reason, detail = classify_exception(exc)
                    failure_counter[reason] += 1
                    if len(failure_examples) < max(0, int(args.error_log_limit)):
                        failure_examples.append(
                            RolloutFailure(
                                subset_id=int(pair.subset_id),
                                prompt_idx=int(prompt_idx),
                                stage="extract_rollout_features",
                                reason=reason,
                                detail=detail,
                            )
                        )
                        print(
                            f"[WARN] subset={pair.subset_id} prompt_idx={prompt_idx} "
                            f"reason={reason} detail={detail}"
                        )
                    break
                pair_features.append(feat)

            if pair_failed or len(pair_features) != 2:
                skipped_prompts += 1
            else:
                for prompt_hidden, response_hidden, response_features in pair_features:
                    prompt_hidden_rows.append(prompt_hidden)
                    response_hidden_rows.append(response_hidden)
                    response_feature_rows.append(response_features)
                    targets.append(pair.avg_score)
                used_prompts += 1

            if args.log_every > 0 and prompt_idx % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[INFO] processed={prompt_idx}/{len(prompt_pairs)} prompts, "
                    f"used={used_prompts}, skipped={skipped_prompts}, rows={len(targets)}, "
                    f"elapsed={elapsed:.1f}s"
                )
                if failure_counter:
                    top_reason, top_count = failure_counter.most_common(1)[0]
                    print(f"[INFO] top_failure={top_reason} ({top_count})")
    finally:
        layer_capture.close()

    if not targets:
        reason_summary = ", ".join(f"{k}:{v}" for k, v in failure_counter.most_common()) or "none"
        detail_lines = []
        for idx, item in enumerate(failure_examples[:5], start=1):
            detail_lines.append(
                f"  {idx}. subset={item.subset_id}, prompt_idx={item.prompt_idx}, "
                f"reason={item.reason}, detail={item.detail}"
            )
        details = "\n".join(detail_lines) if detail_lines else "  (no captured exceptions)"
        raise RuntimeError(
            "No training rows were collected.\n"
            f"Failure summary: {reason_summary}\n"
            "Sample failures:\n"
            f"{details}"
        )

    print(f"[INFO] Fitting estimator on {len(targets)} rows ({used_prompts} prompts x 2 rollouts).")
    bundle = fit_single_trajectory_estimator(
        prompt_hidden_rows=prompt_hidden_rows,
        response_hidden_rows=response_hidden_rows,
        response_feature_rows=response_feature_rows,
        targets=targets,
        feature_builder_config=feature_builder_config,
        fit_config=fit_config,
    )

    output_model_path = args.output_model_path.expanduser().resolve()
    save_single_trajectory_estimator_bundle(bundle, output_model_path)
    print(f"[INFO] Saved estimator bundle: {output_model_path}")

    train_x, _ = build_training_matrix(
        prompt_hidden_rows=prompt_hidden_rows,
        response_hidden_rows=response_hidden_rows,
        response_feature_rows=response_feature_rows,
        feature_builder_config=feature_builder_config,
        fit_config=fit_config,
        prompt_hidden_projection=bundle.get("prompt_hidden_pca"),
        response_hidden_projection=bundle.get("response_hidden_pca"),
    )
    train_y = np.asarray(targets, dtype=np.float32)
    pred_y = np.asarray(bundle["estimator"].predict(train_x), dtype=np.float32)
    train_mae = float(np.mean(np.abs(pred_y - train_y)))
    train_rmse = float(math.sqrt(float(np.mean((pred_y - train_y) ** 2))))

    output_meta_path = (
        args.output_meta_path.expanduser().resolve()
        if args.output_meta_path is not None
        else output_model_path.with_suffix(output_model_path.suffix + ".meta.json")
    )
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_path": args.model_path,
        "device": str(device),
        "dtype": str(dtype),
        "subset_ids": subset_ids,
        "pair_selection": args.pair_selection,
        "prompt_selection": args.prompt_selection,
        "seed": int(args.seed),
        "rollouts_per_prompt": int(args.rollouts_per_prompt),
        "total_prompts": int(len(prompt_pairs)),
        "used_prompts": int(used_prompts),
        "skipped_prompts": int(skipped_prompts),
        "failure_counter": dict(failure_counter),
        "num_rows": int(len(targets)),
        "train_label_mean": float(train_y.mean()),
        "train_label_min": float(train_y.min()),
        "train_label_max": float(train_y.max()),
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "feature_builder_config_path": str(args.feature_builder_config.expanduser().resolve()),
        "fit_config_path": str(args.fit_config.expanduser().resolve()),
        "output_model_path": str(output_model_path),
    }
    with output_meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved metadata: {output_meta_path}")


if __name__ == "__main__":
    main()
