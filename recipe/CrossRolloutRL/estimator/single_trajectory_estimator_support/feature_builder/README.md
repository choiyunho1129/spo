# Feature Builder

The builder is the step that turns raw model outputs into estimator inputs.

It returns:

- `prompt_hidden`
- `response_hidden`
- `response_features`

## What You Pass In

Two hidden-state inputs:

- `prompt_hidden_layers`
- `response_hidden_layers`

Accepted shapes:

- `[num_layers, num_tokens, hidden_dim]`
- `[num_tokens, hidden_dim]`
- `[hidden_dim]`

Common shape in the current setting:

- `prompt_hidden_layers`: `[36, prompt_len, 2560]`
- `response_hidden_layers`: `[36, response_len_or_span_len, 2560]`

One rollout result:

- `generated_text`: decoded response string
- `response_ids`: token ids for the same response
- `tokenizer`
- `rollout_features`

Required rollout feature keys:

- `output_mean_token_entropy`
- `reasoning_mean_token_entropy`
- `answer_mean_token_entropy`

The builder computes these for you:

- `output_length`
- `has_complete_answer`
- `think_tokens`
- `answer_tokens`
- all derived scalar features

## What You Get Back

In the current setting:

- `prompt_hidden`: `[2560]`
- `response_hidden`: `[2560]`
- `response_features`: dict with `20` scalar values

## Minimal Example

```python
from recipe.CrossRolloutRL.estimator.single_trajectory_estimator import (
    FeatureBuilderConfig,
    SingleTrajectoryFeatureBuilder,
)

builder = SingleTrajectoryFeatureBuilder(FeatureBuilderConfig.from_dict(builder_config))
prompt_hidden, response_hidden, response_features = builder.build_inputs(
    prompt_hidden_layers=prompt_hidden_layers,
    response_hidden_layers=response_hidden_layers,
    generated_text=generated_text,
    response_ids=response_ids,
    tokenizer=tokenizer,
    rollout_features={
        "output_mean_token_entropy": output_mean_token_entropy,
        "reasoning_mean_token_entropy": reasoning_mean_token_entropy,
        "answer_mean_token_entropy": answer_mean_token_entropy,
    },
)
```

## Where The Hidden States Usually Come From

If you already run a teacher-forced forward on `prompt + response`, use that forward.

Example:

```python
captured = {}

### function for capturing hidden states ###

def save_layer26(_module, _inputs, output):  # create "hook"
    hidden = output[0] if isinstance(output, tuple) else output
    captured["layer26"] = hidden.detach()

base_model = self.actor_module.module if hasattr(self.actor_module, "module") else self.actor_module
handle = base_model.model.layers[26].register_forward_hook(save_layer26) ### register "hook" 


output = self.actor_module(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    use_cache=False,
    return_dict=True,
)

handle.remove() ### delete "hook" -> important!!!

response_length = response_ids.shape[-1]
logits = output.logits[:, -response_length - 1 : -1, :]
entropy = entropy_from_logits(logits) ### some pseudocode for calculating entropy from logits 

layer26 = captured["layer26"]  # [bs, seq_len, hidden_dim]
```

`output_hidden_states=True` also works, but it returns every layer for every token. For this estimator, a one-layer hook is usually cheaper.
