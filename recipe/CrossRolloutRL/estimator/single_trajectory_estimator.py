from __future__ import annotations

"""Public entry point for the single-trajectory value estimator."""

if __package__:
    from .single_trajectory_estimator_support.feature_builder import (
        FeatureBuilderConfig,
        HiddenSequenceConfig,
        PoolingConfig,
        REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS,
        RolloutScalarConfig,
        SingleTrajectoryFeatureBuilder,
        extract_derived_rollout_features,
        extract_reasoning_and_answer,
        extract_rollout_numeric_features,
        pool_hidden_tokens,
    )
    from .single_trajectory_estimator_support.value_estimator import (
        EstimatorModelConfig,
        ProjectionConfig,
        SingleTrajectoryEstimator,
        SingleTrajectoryEstimatorConfig,
        SingleTrajectoryEstimatorFitConfig,
        build_training_matrix,
        fit_single_trajectory_estimator,
        load_single_trajectory_estimator,
        save_single_trajectory_estimator_bundle,
    )
else:
    from single_trajectory_estimator_support.feature_builder import (
        FeatureBuilderConfig,
        HiddenSequenceConfig,
        PoolingConfig,
        REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS,
        RolloutScalarConfig,
        SingleTrajectoryFeatureBuilder,
        extract_derived_rollout_features,
        extract_reasoning_and_answer,
        extract_rollout_numeric_features,
        pool_hidden_tokens,
    )
    from single_trajectory_estimator_support.value_estimator import (
        EstimatorModelConfig,
        ProjectionConfig,
        SingleTrajectoryEstimator,
        SingleTrajectoryEstimatorConfig,
        SingleTrajectoryEstimatorFitConfig,
        build_training_matrix,
        fit_single_trajectory_estimator,
        load_single_trajectory_estimator,
        save_single_trajectory_estimator_bundle,
    )

__all__ = [
    "EstimatorModelConfig",
    "FeatureBuilderConfig",
    "HiddenSequenceConfig",
    "PoolingConfig",
    "ProjectionConfig",
    "REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS",
    "RolloutScalarConfig",
    "SingleTrajectoryEstimator",
    "SingleTrajectoryEstimatorConfig",
    "SingleTrajectoryEstimatorFitConfig",
    "SingleTrajectoryFeatureBuilder",
    "build_training_matrix",
    "extract_derived_rollout_features",
    "extract_reasoning_and_answer",
    "extract_rollout_numeric_features",
    "fit_single_trajectory_estimator",
    "load_single_trajectory_estimator",
    "pool_hidden_tokens",
    "save_single_trajectory_estimator_bundle",
]
