from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import EstimatorModelConfig, ProjectionConfig, SingleTrajectoryEstimatorConfig
from ..feature_builder.features import REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS


class FastPCA:
    """Compatibility shim for legacy bundles that serialized __main__.FastPCA."""

    def transform(self, x: np.ndarray | Sequence[float]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2:
            raise ValueError(f"Expected shape [batch, hidden_dim], got {tuple(x_arr.shape)}")

        components = np.asarray(getattr(self, "components_"), dtype=np.float32)
        if components.ndim != 2:
            raise ValueError(f"Expected PCA components with rank=2, got shape {tuple(components.shape)}")

        mean = getattr(self, "mean_", None)
        if mean is None:
            mean_arr = np.zeros((components.shape[1],), dtype=np.float32)
        else:
            mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)

        if x_arr.shape[1] != components.shape[1]:
            raise ValueError(
                f"Input hidden_dim mismatch: got {x_arr.shape[1]}, expected {components.shape[1]} from PCA."
            )

        projected = (x_arr - mean_arr.reshape(1, -1)) @ components.T
        return projected.astype(np.float32, copy=False)


def _register_legacy_pickle_aliases() -> None:
    # Legacy estimator bundles stored PCA objects as __main__.FastPCA.
    # In Ray workers __main__ points to default_worker.py, so bind the class before joblib.load.
    import __main__

    if not hasattr(__main__, "FastPCA"):
        setattr(__main__, "FastPCA", FastPCA)


def _resolve_response_hidden_projection(bundle: dict[str, Any]) -> Any | None:
    return bundle.get(
        "response_hidden_pca",
        bundle.get(
            "think_end_hidden_pca",
            bundle.get("trajectory_hidden_pca", bundle.get("rollout_hidden_pca")),
        ),
    )


def _projection_config_from_projection(projection: Any | None) -> ProjectionConfig:
    if projection is None:
        return ProjectionConfig(type=None, input_dim=None, output_dim=None)

    input_dim = getattr(projection, "n_features_in_", None)
    output_dim = getattr(projection, "n_components_", None)
    components = getattr(projection, "components_", None)
    if isinstance(components, np.ndarray) and components.ndim == 2:
        if input_dim is None:
            input_dim = int(components.shape[1])
        if output_dim is None:
            output_dim = int(components.shape[0])
    return ProjectionConfig(
        type="pca",
        input_dim=None if input_dim is None else int(input_dim),
        output_dim=None if output_dim is None else int(output_dim),
    )


def _extract_alpha_from_estimator(estimator: Any) -> float:
    model = None
    named_steps = getattr(estimator, "named_steps", None)
    if isinstance(named_steps, dict):
        model = named_steps.get("model")
    if model is None:
        model = estimator
    alpha = getattr(model, "alpha", None)
    return float(alpha) if alpha is not None else 300.0


def _extract_feature_dim(config_payload: dict[str, Any], estimator: Any) -> int:
    feature_dim = config_payload.get("feature_dim")
    if feature_dim is not None:
        return int(feature_dim)

    estimator_feature_dim = getattr(estimator, "n_features_in_", None)
    if estimator_feature_dim is not None:
        return int(estimator_feature_dim)
    return 0


def _extract_clip_bounds(config_payload: dict[str, Any]) -> tuple[float, float]:
    if "clip_min" in config_payload and "clip_max" in config_payload:
        return float(config_payload["clip_min"]), float(config_payload["clip_max"])

    model_desc = config_payload.get("model")
    if isinstance(model_desc, str):
        match = re.search(r"clip\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]", model_desc)
        if match:
            return float(match.group(1)), float(match.group(2))

    return 0.0, 1.0


def _load_estimator_config(
    config_payload: Any,
    *,
    estimator: Any,
    prompt_hidden_projection: Any | None,
    response_hidden_projection: Any | None,
) -> SingleTrajectoryEstimatorConfig:
    if not isinstance(config_payload, dict):
        raise ValueError("Model bundle does not contain a valid config payload.")

    try:
        return SingleTrajectoryEstimatorConfig.from_dict(config_payload)
    except Exception:
        response_feature_keys = tuple(
            config_payload.get(
                "response_feature_keys",
                config_payload.get("rollout_scalar_keys", config_payload.get("trajectory_scalar_keys", [])),
            )
        )
        derived_response_feature_keys = tuple(
            config_payload.get(
                "derived_response_feature_keys",
                config_payload.get("trajectory_derived_scalar_keys", []),
            )
        )
        clip_min, clip_max = _extract_clip_bounds(config_payload)

        return SingleTrajectoryEstimatorConfig(
            prompt_hidden_projection=_projection_config_from_projection(prompt_hidden_projection),
            response_hidden_projection=_projection_config_from_projection(response_hidden_projection),
            response_feature_keys=response_feature_keys,
            derived_response_feature_keys=derived_response_feature_keys,
            model=EstimatorModelConfig(
                alpha=_extract_alpha_from_estimator(estimator),
                clip_min=clip_min,
                clip_max=clip_max,
                feature_dim=_extract_feature_dim(config_payload, estimator),
            ),
        )


class SingleTrajectoryEstimator:
    """Predict value from prompt hidden, response hidden, and response features.

    The loaded bundle contains:
    - prompt PCA
    - response PCA
    - regressor
    """

    def __init__(
        self,
        *,
        config: SingleTrajectoryEstimatorConfig,
        estimator: Any,
        prompt_hidden_projection: Any | None = None,
        response_hidden_projection: Any | None = None,
    ) -> None:
        self.config = config
        self.estimator = estimator
        self.prompt_hidden_projection = prompt_hidden_projection
        self.response_hidden_projection = response_hidden_projection

    @classmethod
    def load(cls, model_path: str | Path) -> "SingleTrajectoryEstimator":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("`joblib` is required to load estimator bundles. Install `joblib`.") from exc
        _register_legacy_pickle_aliases()
        bundle = joblib.load(Path(model_path).expanduser().resolve())
        config_payload = bundle.get("config")
        if config_payload is None:
            raise ValueError("Model bundle does not contain a standalone config payload.")
        prompt_hidden_projection = bundle.get("prompt_hidden_pca")
        response_hidden_projection = _resolve_response_hidden_projection(bundle)
        estimator = bundle["estimator"]
        return cls(
            config=_load_estimator_config(
                config_payload,
                estimator=estimator,
                prompt_hidden_projection=prompt_hidden_projection,
                response_hidden_projection=response_hidden_projection,
            ),
            estimator=estimator,
            prompt_hidden_projection=prompt_hidden_projection,
            response_hidden_projection=response_hidden_projection,
        )

    def _transform_hidden(
        self,
        hidden: np.ndarray | Sequence[float],
        projection: Any | None,
    ) -> np.ndarray:
        """Project one hidden vector if needed."""
        vector = np.asarray(hidden, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"Expected a single hidden vector with shape [hidden_dim], got {tuple(vector.shape)}")
        vector = vector.reshape(1, -1)
        if projection is not None:
            vector = projection.transform(vector).astype(np.float32, copy=False)
        return vector.reshape(-1)

    def build_response_feature_vector(self, response_features: dict[str, float]) -> np.ndarray:
        """Convert the response feature dict to a vector."""
        feature_map = dict(response_features)
        required_entropy_keys = REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS.intersection(self.config.response_feature_keys)
        missing_entropy_keys = sorted(required_entropy_keys - feature_map.keys())
        if missing_entropy_keys:
            raise ValueError(f"Missing actual token entropy features {missing_entropy_keys} in response_features.")
        ordered_keys = list(self.config.response_feature_keys)
        ordered_keys.extend(self.config.derived_response_feature_keys)
        return np.asarray([float(feature_map.get(key, 0.0)) for key in ordered_keys], dtype=np.float32)

    def build_feature_vector(
        self,
        *,
        prompt_hidden: np.ndarray | Sequence[float],
        response_hidden: np.ndarray | Sequence[float],
        response_features: dict[str, float],
    ) -> np.ndarray:
        """Build the final input vector."""
        pieces = [
            self._transform_hidden(prompt_hidden, self.prompt_hidden_projection),
            self._transform_hidden(response_hidden, self.response_hidden_projection),
            self.build_response_feature_vector(response_features),
        ]
        feature_vector = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
        if self.config.model.feature_dim > 0 and feature_vector.shape[0] != self.config.model.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.config.model.feature_dim}, got {feature_vector.shape[0]}"
            )
        return feature_vector

    def predict_value(
        self,
        *,
        prompt_hidden: np.ndarray | Sequence[float],
        response_hidden: np.ndarray | Sequence[float],
        response_features: dict[str, float],
    ) -> float:
        """Predict one value."""
        x = self.build_feature_vector(
            prompt_hidden=prompt_hidden,
            response_hidden=response_hidden,
            response_features=response_features,
        ).reshape(1, -1)
        prediction = float(np.asarray(self.estimator.predict(x), dtype=np.float32).reshape(-1)[0])
        return float(np.clip(prediction, self.config.model.clip_min, self.config.model.clip_max))


def load_single_trajectory_estimator(model_path: str | Path) -> SingleTrajectoryEstimator:
    return SingleTrajectoryEstimator.load(model_path)
