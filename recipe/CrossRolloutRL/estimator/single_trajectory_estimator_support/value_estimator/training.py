from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..feature_builder.config import FeatureBuilderConfig
from .config import (
    EstimatorModelConfig,
    ProjectionConfig,
    SingleTrajectoryEstimatorConfig,
)
from .runtime import SingleTrajectoryEstimator


@dataclass(frozen=True)
class SingleTrajectoryEstimatorFitConfig:
    prompt_hidden_pca_dim: int = 0
    response_hidden_pca_dim: int = 0
    alpha: float = 300.0
    random_seed: int = 42
    clip_min: float = 0.0
    clip_max: float = 1.0


def fit_hidden_pca(
    hidden_rows: Sequence[np.ndarray | Sequence[float]],
    pca_dim: int,
    random_seed: int,
) -> Any | None:
    if pca_dim <= 0:
        return None

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("`scikit-learn` is required for PCA fitting. Install `scikit-learn`.") from exc

    hidden_matrix = np.stack(
        [np.asarray(row, dtype=np.float32).reshape(-1) for row in hidden_rows],
        axis=0,
    )
    effective_dim = min(int(pca_dim), int(hidden_matrix.shape[0]), int(hidden_matrix.shape[1]))
    if effective_dim <= 0:
        raise ValueError(
            f"Invalid effective PCA dim computed from requested={pca_dim}, "
            f"shape={tuple(hidden_matrix.shape)}."
        )
    pca = PCA(n_components=effective_dim, svd_solver="randomized", random_state=random_seed)
    pca.fit(hidden_matrix)
    return pca


def build_training_matrix(
    *,
    prompt_hidden_rows: Sequence[np.ndarray | Sequence[float]],
    response_hidden_rows: Sequence[np.ndarray | Sequence[float]],
    response_feature_rows: Sequence[dict[str, float]],
    feature_builder_config: FeatureBuilderConfig,
    fit_config: SingleTrajectoryEstimatorFitConfig,
    prompt_hidden_projection: Any | None = None,
    response_hidden_projection: Any | None = None,
) -> tuple[np.ndarray, SingleTrajectoryEstimatorConfig]:
    if not prompt_hidden_rows:
        raise ValueError("No training rows were provided.")
    if len(prompt_hidden_rows) != len(response_hidden_rows):
        raise ValueError("prompt_hidden_rows and response_hidden_rows must have the same length.")
    if len(prompt_hidden_rows) != len(response_feature_rows):
        raise ValueError("response_feature_rows must match prompt_hidden_rows length.")

    estimator_config = SingleTrajectoryEstimatorConfig(
        prompt_hidden_projection=ProjectionConfig(
            type=None if prompt_hidden_projection is None else "pca",
            input_dim=None if prompt_hidden_projection is None else int(prompt_hidden_projection.n_features_in_),
            output_dim=None if prompt_hidden_projection is None else int(prompt_hidden_projection.n_components_),
        ),
        response_hidden_projection=ProjectionConfig(
            type=None if response_hidden_projection is None else "pca",
            input_dim=None if response_hidden_projection is None else int(response_hidden_projection.n_features_in_),
            output_dim=None if response_hidden_projection is None else int(response_hidden_projection.n_components_),
        ),
        response_feature_keys=tuple(feature_builder_config.rollout_scalars.scalar_keys),
        derived_response_feature_keys=tuple(feature_builder_config.rollout_scalars.derived_scalar_keys),
        model=EstimatorModelConfig(
            alpha=float(fit_config.alpha),
            clip_min=float(fit_config.clip_min),
            clip_max=float(fit_config.clip_max),
            feature_dim=0,
        ),
    )

    bootstrap_estimator = SingleTrajectoryEstimator(
        config=estimator_config,
        estimator=None,
        prompt_hidden_projection=prompt_hidden_projection,
        response_hidden_projection=response_hidden_projection,
    )
    rows = []
    for prompt_hidden, response_hidden, response_features in zip(
        prompt_hidden_rows,
        response_hidden_rows,
        response_feature_rows,
        strict=True,
    ):
        rows.append(
            bootstrap_estimator.build_feature_vector(
                prompt_hidden=prompt_hidden,
                response_hidden=response_hidden,
                response_features=response_features,
            ).astype(np.float32, copy=False)
        )
    feature_matrix = np.stack(rows, axis=0).astype(np.float32, copy=False)

    estimator_config = SingleTrajectoryEstimatorConfig(
        prompt_hidden_projection=estimator_config.prompt_hidden_projection,
        response_hidden_projection=estimator_config.response_hidden_projection,
        response_feature_keys=estimator_config.response_feature_keys,
        derived_response_feature_keys=estimator_config.derived_response_feature_keys,
        model=EstimatorModelConfig(
            alpha=estimator_config.model.alpha,
            clip_min=estimator_config.model.clip_min,
            clip_max=estimator_config.model.clip_max,
            feature_dim=int(feature_matrix.shape[1]),
        ),
    )
    return feature_matrix, estimator_config


def fit_single_trajectory_estimator(
    *,
    prompt_hidden_rows: Sequence[np.ndarray | Sequence[float]],
    response_hidden_rows: Sequence[np.ndarray | Sequence[float]],
    response_feature_rows: Sequence[dict[str, float]],
    targets: Sequence[float],
    feature_builder_config: FeatureBuilderConfig,
    fit_config: SingleTrajectoryEstimatorFitConfig,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "`scikit-learn` is required for estimator training. Install `scikit-learn`."
        ) from exc

    y = np.asarray(targets, dtype=np.float32).reshape(-1)
    if len(prompt_hidden_rows) != int(y.shape[0]):
        raise ValueError("targets length must match prompt_hidden_rows length.")

    prompt_hidden_projection = fit_hidden_pca(
        prompt_hidden_rows,
        fit_config.prompt_hidden_pca_dim,
        fit_config.random_seed,
    )
    response_hidden_projection = fit_hidden_pca(
        response_hidden_rows,
        fit_config.response_hidden_pca_dim,
        fit_config.random_seed,
    )
    feature_matrix, estimator_config = build_training_matrix(
        prompt_hidden_rows=prompt_hidden_rows,
        response_hidden_rows=response_hidden_rows,
        response_feature_rows=response_feature_rows,
        feature_builder_config=feature_builder_config,
        fit_config=fit_config,
        prompt_hidden_projection=prompt_hidden_projection,
        response_hidden_projection=response_hidden_projection,
    )

    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=fit_config.alpha, random_state=fit_config.random_seed)),
        ]
    )
    estimator.fit(feature_matrix, y)

    return {
        "bundle_type": "single_trajectory_estimator",
        "bundle_version": 1,
        "config": estimator_config.to_dict(),
        "estimator": estimator,
        "prompt_hidden_pca": prompt_hidden_projection,
        "response_hidden_pca": response_hidden_projection,
    }


def save_single_trajectory_estimator_bundle(bundle: dict[str, Any], model_path: str | Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("`joblib` is required to save estimator bundles. Install `joblib`.") from exc

    model_path = Path(model_path).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
