from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PoolingConfig:
    type: str
    n: int | None = None


@dataclass(frozen=True)
class HiddenSequenceConfig:
    input_field: str
    layer_index: int
    pooling: PoolingConfig


@dataclass(frozen=True)
class RolloutScalarConfig:
    scalar_keys: tuple[str, ...]
    derived_scalar_keys: tuple[str, ...]
    extra_scalar_field_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureBuilderConfig:
    prompt_hidden: HiddenSequenceConfig
    response_hidden: HiddenSequenceConfig
    rollout_scalars: RolloutScalarConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureBuilderConfig":
        prompt_payload = dict(payload["prompt_hidden"])
        response_payload = dict(
            payload.get("response_hidden", payload.get("think_end_hidden", payload.get("trajectory_hidden")))
        )
        rollout_payload = dict(payload["rollout_scalars"])
        return cls(
            prompt_hidden=HiddenSequenceConfig(
                input_field=str(prompt_payload["input_field"]),
                layer_index=int(prompt_payload["layer_index"]),
                pooling=PoolingConfig(**dict(prompt_payload["pooling"])),
            ),
            response_hidden=HiddenSequenceConfig(
                input_field=str(response_payload["input_field"]),
                layer_index=int(response_payload["layer_index"]),
                pooling=PoolingConfig(**dict(response_payload["pooling"])),
            ),
            rollout_scalars=RolloutScalarConfig(
                scalar_keys=tuple(rollout_payload.get("scalar_keys", [])),
                derived_scalar_keys=tuple(rollout_payload.get("derived_scalar_keys", [])),
                extra_scalar_field_paths=tuple(rollout_payload.get("extra_scalar_field_paths", [])),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
