# Value Estimator

The estimator is the final model.

It takes:

- `prompt_hidden`
- `response_hidden`
- `response_features`

and returns:

- scalar `value`

## What You Pass In

Current setting:

- `prompt_hidden`: `[2560]`
- `response_hidden`: `[2560]`
- `response_features`: `20` scalars

## What Is Inside The Saved Model

- fitted prompt PCA
- fitted trajectory PCA
- fitted regressor

So inside the model pipeline:

- `16 + 16 + 20 = 52`

## Runtime Example

```python
from recipe.CrossRolloutRL.estimator.single_trajectory_estimator import load_single_trajectory_estimator

estimator = load_single_trajectory_estimator(model_path)
value = estimator.predict_value(
    prompt_hidden=prompt_hidden,
    response_hidden=response_hidden,
    response_features=response_features,
)
```

## Feature Keys

Raw rollout features:

- `output_length`
- `think_tokens`
- `answer_tokens`
- `has_complete_answer`
- `has_reasoning_content`
- `output_mean_token_entropy`
- `reasoning_mean_token_entropy`
- `answer_mean_token_entropy`
- `output_unique_token_ratio`
- `answer_unique_token_ratio`
- `output_repetition_ratio`
- `reasoning_repetition_ratio`
- `duplicate_line_ratio`

Derived rollout features:

- `think_ratio`
- `answer_ratio`
- `entropy_gap_reasoning_answer`
- `unique_gap_reasoning_output`
- `repetition_gap_reasoning_output`
- `reasoning_x_log_output_length`
- `answer_entropy_gap_vs_output`

## Training Example

```python
from recipe.CrossRolloutRL.estimator.single_trajectory_estimator import (
    FeatureBuilderConfig,
    SingleTrajectoryEstimatorFitConfig,
    fit_single_trajectory_estimator,
    save_single_trajectory_estimator_bundle,
)

bundle = fit_single_trajectory_estimator(
    prompt_hidden_rows=prompt_hidden_rows,
    response_hidden_rows=response_hidden_rows,
    response_feature_rows=response_feature_rows,
    targets=value_targets,
    feature_builder_config=FeatureBuilderConfig.from_dict(builder_config),
    fit_config=SingleTrajectoryEstimatorFitConfig(
        prompt_hidden_pca_dim=16,
        response_hidden_pca_dim=16,
        alpha=300.0,
    ),
)
save_single_trajectory_estimator_bundle(bundle, model_path)
```
