from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS = {
    "output_mean_token_entropy",
    "reasoning_mean_token_entropy",
    "answer_mean_token_entropy",
}

THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            converted = float(stripped)
        except ValueError:
            return None
        return converted if math.isfinite(converted) else None
    return None


def tokenize_whitespace(text: str) -> list[str]:
    return [token for token in re.findall(r"\S+", text.lower()) if token]


def unique_token_ratio(text: str) -> float | None:
    tokens = tokenize_whitespace(text)
    if not tokens:
        return None
    return len(set(tokens)) / len(tokens)


def repetition_ratio(text: str) -> float | None:
    unique_ratio = unique_token_ratio(text)
    if unique_ratio is None:
        return None
    return 1.0 - unique_ratio


def duplicate_line_ratio(text: str) -> float | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return 1.0 - (len(set(lines)) / len(lines))


def extract_reasoning_and_answer(record: dict[str, Any]) -> tuple[str, str, str]:
    generated_text = str(record.get("generated_text", ""))
    reasoning_content = str(record.get("reasoning_content", ""))
    answer_content = str(record.get("answer_content", ""))

    if not reasoning_content:
        match = THINK_PATTERN.search(generated_text)
        if match:
            reasoning_content = match.group(1)
    if not answer_content:
        match = ANSWER_PATTERN.search(generated_text)
        if match:
            answer_content = match.group(1)
        elif "</think>" in generated_text:
            answer_content = generated_text.split("</think>", maxsplit=1)[1].strip()
    return generated_text, reasoning_content, answer_content


def extract_actual_entropy_features(record: dict[str, Any]) -> dict[str, float]:
    rollout_features = record.get("rollout_features")
    if not isinstance(rollout_features, dict):
        return {}
    features: dict[str, float] = {}
    for key in REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS:
        numeric = coerce_float(rollout_features.get(key))
        if numeric is not None:
            features[key] = numeric
    return features


def extract_rollout_numeric_features(record: dict[str, Any]) -> dict[str, float]:
    token_stats = record.get("token_stats") or {}
    generated_text, reasoning_content, answer_content = extract_reasoning_and_answer(record)

    features = extract_actual_entropy_features(record)
    builtins = {
        "output_length": coerce_float(record.get("output_length")),
        "think_tokens": coerce_float(token_stats.get("think_tokens")),
        "answer_tokens": coerce_float(token_stats.get("answer_tokens")),
        "has_complete_answer": 1.0 if record.get("has_complete_answer") else 0.0,
        "has_reasoning_content": 1.0 if reasoning_content.strip() else 0.0,
        "output_unique_token_ratio": unique_token_ratio(generated_text),
        "answer_unique_token_ratio": unique_token_ratio(answer_content),
        "output_repetition_ratio": repetition_ratio(generated_text),
        "reasoning_repetition_ratio": repetition_ratio(reasoning_content),
        "duplicate_line_ratio": duplicate_line_ratio(generated_text),
        # Internal helper for derived features.
        "reasoning_unique_token_ratio": unique_token_ratio(reasoning_content),
    }
    features.update({key: value for key, value in builtins.items() if value is not None})
    return features


def extract_derived_rollout_features(feature_map: dict[str, float]) -> dict[str, float]:
    output_length = max(float(feature_map.get("output_length", 0.0)), 1.0)
    think_tokens = float(feature_map.get("think_tokens", 0.0))
    answer_tokens = float(feature_map.get("answer_tokens", 0.0))
    has_reasoning_content = float(feature_map.get("has_reasoning_content", 0.0))
    output_entropy = float(feature_map.get("output_mean_token_entropy", 0.0))
    reasoning_entropy = float(feature_map.get("reasoning_mean_token_entropy", 0.0))
    answer_entropy = float(feature_map.get("answer_mean_token_entropy", 0.0))
    output_unique = float(feature_map.get("output_unique_token_ratio", 0.0))
    reasoning_unique = float(feature_map.get("reasoning_unique_token_ratio", 0.0))
    output_repeat = float(feature_map.get("output_repetition_ratio", 0.0))
    reasoning_repeat = float(feature_map.get("reasoning_repetition_ratio", 0.0))
    return {
        "think_ratio": think_tokens / output_length,
        "answer_ratio": answer_tokens / output_length,
        "entropy_gap_reasoning_answer": reasoning_entropy - answer_entropy,
        "unique_gap_reasoning_output": reasoning_unique - output_unique,
        "repetition_gap_reasoning_output": reasoning_repeat - output_repeat,
        "reasoning_x_log_output_length": has_reasoning_content * float(np.log1p(output_length)),
        "answer_entropy_gap_vs_output": answer_entropy - output_entropy,
    }
