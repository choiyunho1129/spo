from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectionConfig:
    type: str | None
    input_dim: int | None
    output_dim: int | None


@dataclass(frozen=True)
class EstimatorModelConfig:
    alpha: float
    clip_min: float
    clip_max: float
    feature_dim: int


@dataclass(frozen=True)
class SingleTrajectoryEstimatorConfig:
    prompt_hidden_projection: ProjectionConfig
    response_hidden_projection: ProjectionConfig
    response_feature_keys: tuple[str, ...]
    derived_response_feature_keys: tuple[str, ...]
    model: EstimatorModelConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SingleTrajectoryEstimatorConfig":
        return cls(
            prompt_hidden_projection=ProjectionConfig(**dict(payload["prompt_hidden_projection"])),
            response_hidden_projection=ProjectionConfig(
                **dict(
                    payload.get(
                        "response_hidden_projection",
                        payload.get("think_end_hidden_projection", payload.get("trajectory_hidden_projection")),
                    )
                )
            ),
            response_feature_keys=tuple(payload.get("response_feature_keys", payload.get("trajectory_scalar_keys", []))),
            derived_response_feature_keys=tuple(
                payload.get("derived_response_feature_keys", payload.get("trajectory_derived_scalar_keys", []))
            ),
            model=EstimatorModelConfig(**dict(payload["model"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
