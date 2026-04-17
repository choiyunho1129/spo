from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import SingleTrajectoryEstimatorConfig
from ..feature_builder.features import REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS


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
        bundle = joblib.load(Path(model_path).expanduser().resolve())
        config_payload = bundle.get("config")
        if config_payload is None:
            raise ValueError("Model bundle does not contain a standalone config payload.")
        return cls(
            config=SingleTrajectoryEstimatorConfig.from_dict(config_payload),
            estimator=bundle["estimator"],
            prompt_hidden_projection=bundle.get("prompt_hidden_pca"),
            response_hidden_projection=bundle.get(
                "response_hidden_pca",
                bundle.get("think_end_hidden_pca", bundle.get("trajectory_hidden_pca")),
            ),
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
