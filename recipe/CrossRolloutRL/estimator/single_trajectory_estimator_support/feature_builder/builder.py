from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .config import FeatureBuilderConfig, HiddenSequenceConfig
from .features import (
    extract_derived_rollout_features,
    extract_reasoning_and_answer,
    extract_rollout_numeric_features,
)


def _as_float32_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _resolve_layer_hidden(
    hidden_layers: Sequence[np.ndarray | Sequence[float]] | np.ndarray,
    layer_index: int,
) -> np.ndarray:
    if isinstance(hidden_layers, np.ndarray):
        array = _as_float32_array(hidden_layers)
        if array.ndim == 1:
            return array
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            if layer_index < 0 or layer_index >= array.shape[0]:
                raise ValueError(
                    f"Requested layer_index={layer_index}, got shape={tuple(array.shape)}. "
                    "Expected [num_layers, num_tokens, hidden_dim] for one example."
                )
            return array[layer_index]
        raise ValueError(f"Unsupported hidden array shape {tuple(array.shape)}")

    hidden_list = list(hidden_layers)
    if not hidden_list:
        raise ValueError("No hidden layers were provided.")
    if layer_index < 0 or layer_index >= len(hidden_list):
        raise IndexError(f"Requested layer_index={layer_index}, available={len(hidden_list)}")
    return _as_float32_array(hidden_list[layer_index])


def pool_hidden_tokens(hidden_tokens: np.ndarray | Sequence[float], spec: HiddenSequenceConfig) -> np.ndarray:
    hidden = _as_float32_array(hidden_tokens)
    if hidden.ndim == 1:
        return hidden.reshape(-1)
    if hidden.ndim != 2:
        raise ValueError(f"Expected 1D or 2D hidden tokens, got shape={tuple(hidden.shape)}")
    if spec.pooling.type != "last_n_mean":
        raise ValueError(f"Unsupported pooling type {spec.pooling.type!r}")
    if spec.pooling.n is None or spec.pooling.n <= 0:
        raise ValueError("last_n_mean pooling requires a positive `n`.")
    pooled = hidden[-int(spec.pooling.n) :].mean(axis=0)
    return pooled.astype(np.float32, copy=False).reshape(-1)


def _normalize_token_ids(response_ids: Sequence[int] | np.ndarray) -> list[int]:
    array = np.asarray(response_ids).reshape(-1)
    return [int(token_id) for token_id in array.tolist()]


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text.strip():
        return 0
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            token_ids = encoded["input_ids"]
        else:
            token_ids = encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return int(len(token_ids))


class SingleTrajectoryFeatureBuilder:
    """Turn raw hidden states and rollout metadata into estimator inputs."""

    def __init__(self, config: FeatureBuilderConfig) -> None:
        self.config = config

    def build_hidden_vector(
        self,
        hidden_layers: Sequence[np.ndarray | Sequence[float]] | np.ndarray,
        spec: HiddenSequenceConfig,
    ) -> np.ndarray:
        """Pick one layer and pool it into one vector."""
        layer_hidden = _resolve_layer_hidden(hidden_layers, spec.layer_index)
        return pool_hidden_tokens(layer_hidden, spec)

    def build_prompt_hidden(
        self,
        prompt_hidden_layers: Sequence[np.ndarray | Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        return self.build_hidden_vector(prompt_hidden_layers, self.config.prompt_hidden)

    def build_response_hidden(
        self,
        response_hidden_layers: Sequence[np.ndarray | Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        return self.build_hidden_vector(response_hidden_layers, self.config.response_hidden)

    def build_response_record(
        self,
        *,
        generated_text: str,
        response_ids: Sequence[int] | np.ndarray,
        tokenizer: Any,
        reasoning_content: str = "",
        answer_content: str = "",
        rollout_features: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the internal rollout record used to compute scalar features."""
        normalized_response_ids = _normalize_token_ids(response_ids)
        _, reasoning_text, answer_text = extract_reasoning_and_answer(
            {
                "generated_text": generated_text,
                "reasoning_content": reasoning_content,
                "answer_content": answer_content,
            }
        )
        return {
            "generated_text": generated_text,
            "reasoning_content": reasoning_text,
            "answer_content": answer_text,
            "output_length": int(len(normalized_response_ids)),
            "has_complete_answer": bool(answer_text),
            "token_stats": {
                "think_tokens": _count_text_tokens(tokenizer, reasoning_text),
                "answer_tokens": _count_text_tokens(tokenizer, answer_text),
            },
            "rollout_features": dict(rollout_features),
        }

    def build_response_features(
        self,
        response_record: dict[str, Any],
    ) -> dict[str, float]:
        """Return the scalar features used by the estimator."""
        feature_map = extract_rollout_numeric_features(response_record)
        feature_map.update(extract_derived_rollout_features(feature_map))
        ordered_keys = list(self.config.rollout_scalars.scalar_keys)
        ordered_keys.extend(self.config.rollout_scalars.derived_scalar_keys)
        return {key: float(feature_map.get(key, 0.0)) for key in ordered_keys}

    def build_inputs(
        self,
        *,
        prompt_hidden_layers: Sequence[np.ndarray | Sequence[float]] | np.ndarray,
        response_hidden_layers: Sequence[np.ndarray | Sequence[float]] | np.ndarray,
        generated_text: str,
        response_ids: Sequence[int] | np.ndarray,
        tokenizer: Any,
        reasoning_content: str = "",
        answer_content: str = "",
        rollout_features: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Return prompt hidden, response hidden, and response features."""
        prompt_hidden = self.build_prompt_hidden(prompt_hidden_layers)
        response_hidden = self.build_response_hidden(response_hidden_layers)
        response_record = self.build_response_record(
            generated_text=generated_text,
            response_ids=response_ids,
            tokenizer=tokenizer,
            reasoning_content=reasoning_content,
            answer_content=answer_content,
            rollout_features=rollout_features,
        )
        response_features = self.build_response_features(response_record)
        return prompt_hidden, response_hidden, response_features
