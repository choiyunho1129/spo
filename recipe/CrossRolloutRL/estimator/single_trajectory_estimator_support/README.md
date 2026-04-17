# Single Trajectory Estimator Support

Public import path:

- `recipe/CrossRolloutRL/estimator/single_trajectory_estimator.py`

This code has two parts:

- `recipe/CrossRolloutRL/estimator/single_trajectory_estimator_support/feature_builder/`
- `recipe/CrossRolloutRL/estimator/single_trajectory_estimator_support/value_estimator/`

## What Each Part Does

- `feature_builder`
  Takes raw hidden states and rollout metadata, then builds:
  `prompt_hidden`, `response_hidden`, `response_features`.
- `value_estimator`
  Takes those three inputs and returns one scalar `value`.

## End-To-End Flow

1. Start with raw hidden states from the model.
2. Start with one rollout result: text, token ids, and entropy summaries.
3. Use the builder to make pooled vectors and scalar features.
4. Use the estimator to predict `value`.

## Minimal Usage

```python
from recipe.CrossRolloutRL.estimator.single_trajectory_estimator import (
    FeatureBuilderConfig,
    SingleTrajectoryFeatureBuilder,
    load_single_trajectory_estimator,
)

builder = SingleTrajectoryFeatureBuilder(FeatureBuilderConfig.from_dict(builder_config))
prompt_hidden, response_hidden, response_features = builder.build_inputs(
    prompt_hidden_layers=prompt_hidden_layers,
    response_hidden_layers=response_hidden_layers,
    generated_text=generated_text,
    response_ids=response_ids,
    tokenizer=tokenizer,
    rollout_features=rollout_features,
)

estimator = load_single_trajectory_estimator(model_path)
value = estimator.predict_value(
    prompt_hidden=prompt_hidden,
    response_hidden=response_hidden,
    response_features=response_features,
)
```

## Current Default Shapes

- raw `prompt_hidden_layers`: `[36, prompt_len, 2560]`
- raw `response_hidden_layers`: `[36, response_len_or_span_len, 2560]`
- pooled `prompt_hidden`: `[2560]`
- pooled `response_hidden`: `[2560]`
- `response_features`: `20` scalars
- estimator input after fitted PCA: `[52]`
- estimator output: scalar `value`
